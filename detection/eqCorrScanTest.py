import os
import multiprocessing
from multiprocessing import Manager
import time
import obspy
from obspy import read
import obspyh5
import eqcorrscan
from eqcorrscan.core.template_gen import template_gen
from eqcorrscan.core.match_filter import Tribe
from eqcorrscan.core.match_filter import Template
from eqcorrscan.core.match_filter import match_filter

# functions used in this code are defined below
def getFname(path,stat,chan,startTime):

    # extract year string
    tempYear = startTime.year
    tempYearStr = str(tempYear)

    # extract month string and pad with zeros if necessary
    tempMonth = startTime.month
    if tempMonth < 10:
        tempMonthStr = "0" + str(tempMonth)
    else:
        tempMonthStr = str(tempMonth)

    # extract day string and pad with zeros if necessary
    tempDay = startTime.day
    if tempDay < 10:
        tempDayStr = "0" + str(tempDay)
    else:
        tempDayStr = str(tempDay)

    # construct date string
    tempDate = tempYearStr + "-" + tempMonthStr + "-" + tempDayStr

    # read in data for template
    fname = path + stat + "/" + chan + "/" + tempDate + "." + stat + "." + chan + ".noIR.MSEED"

    return fname

def makeTemplates(path,stat,chan,tempLims,freq,filtType):

    fname = getFname(path,stat,chan,tempLims[0])

    # read in data for template
    stTemp = obspy.read(fname)

    # basic preprocessing
    stTemp.detrend("demean")
    stTemp.detrend("linear")
    stTemp.taper(max_percentage=0.01, max_length=10.)

    # filter and resample the data to each band
    if filtType == "bandpass":
        stTemp.filter(filtType,freqmin=freq[0],freqmax=freq[1])
        stTemp.resample(freq[1]*2)
    elif filtType == "lowpass":
        stTemp.filter(filtType,freq=freq[0])
        stTemp.resample(freq[0]*2)
    elif filtType == "highpass":
        stTemp.filter(filtType,freq=freq[0])

    # trim the data to the time ranges of template for each band
    stTemp.trim(starttime=tempLims[0],endtime=tempLims[1])

    return stTemp

# start timer and give output
timer = time.time()
print("Loading templates...")

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/"

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
tempH5 = obspy.read(templatePath + 'conservativeWaveforms.h5')[:100]

# define parallel parameters
readPar = 0
nproc = 8

# define station and channel parameters
stat = ["PIG2"]
chan = "HH*"
freq = [1,10]
filtType = "bandpass"

# enter the buffer used on the front and back end to produce templates
buffFront = 2*60
buffEnd = 3*60

# make empty containers for templates and names
tempTimes = []
templates = []
template_names = []

# get template starttimes, endtimes and index
for t in range(len(tempH5)):
    tempTimes.append([tempH5[t].stats.starttime+buffFront*3/4,tempH5[t].stats.endtime-buffEnd*2/3,t])

# clear initial template catalog for memory
del tempH5

# read and process templates in parallel
if readPar:
    # start parallel pool and make shared memory containers
    manager = Manager()
    templates_shared = manager.list([])
    template_names_shared = manager.list([])

    # define loading function to be parallelized
    def parFunc(tempTimes):
        tempLims = [tempTimes[0],tempTimes[1]]
        st = read()
        st.clear()
        for s in stat:
            st += makeTemplates(path,s,chan,tempLims,freq,filtType)
        #print(st)
        #st.plot()
        templates_shared.append(st)
        template_names_shared.append(s.lower()+"temp"+str(tempTimes[2]))

    # load the templates in parallel
    p = multiprocessing.Pool(processes=nproc)
    p.map(parFunc,tempTimes)

    # make normal lists from shared memory containers
    template_names.append([t for t in template_names_shared])
    templates.append([t for t in templates_shared])
    templates = templates[0]
    template_names = template_names[0]

    # clear variables
    del template_names_shared
    del templates_shared

# read and process templates in serial
else:
    #fill each template with data for all desired staions and components
    for t in range(len(tempTimes)):
        tempLims = [tempTimes[t][0],tempTimes[t][1]]
        st = read()
        st.clear()
        for s in stat:
            st += makeTemplates(path,s,chan,tempLims,freq,filtType)
        #print(st)
        #st.plot()
        templates.append(st)
        template_names.append(s.lower()+"temp"+str(t))

# stop timer and give output
runtime = time.time() - timer
print("Loaded " + str(len(templates)) + " templates in " + str(runtime) + " seconds")
print("Loading 1 day of data...")

# read in data to scan
st = read()
st.clear()
for s in stat:
  fname = getFname(path,s,chan,obspy.UTCDateTime(2012,1,20,0,1))
  st += obspy.read(fname)

# basic preprocessing
st.detrend("demean")
st.detrend("linear")
st.taper(max_percentage=0.01, max_length=10.)
if filtType == "bandpass":
    st.filter(filtType,freqmin=freq[0],freqmax=freq[1])
    st.resample(freq[1]*2)
elif filtType == "lowpass":
    st.filter(filtType,freq=freq[0])
    st.resample(freq[0]*2)
elif filtType == "highpass":
    st.filter(filtType,freq=freq[0])

# start timer and give output
timer = time.time()
print("Starting scan...")

# run eqcorrscan's match filter routine
detections = match_filter(template_names=template_names,template_list=templates,st=st,threshold=8,threshold_type="MAD",trig_int=6,cores=20)

# stop timer and give output
runtime = time.time() - timer
print(detections)
print("Scanned 1 day of data with " + str(len(templates)) + " templates in " + str(runtime) + " seconds and found " + str(len(detections)) + " detections")
