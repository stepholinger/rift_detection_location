from obspy import read
import obspy
import obspyh5
from eqcorrscan.core.template_gen import template_gen
import eqcorrscan
import os
from eqcorrscan.core.match_filter import Tribe
from eqcorrscan.core.match_filter import Template
from eqcorrscan.core.match_filter import match_filter
from obspy.clients.fdsn import Client

def makeTemplates(path,stat,chan,tempLims,freq):

    # extract year string
    tempYear = tempLims[0].year
    tempYearStr = str(tempYear)

    # extract month string and pad with zeros if necessary
    tempMonth = tempLims[0].month
    if tempMonth < 10:
        tempMonthStr = "0" + str(tempMonth)
    else:
        tempMonthStr = str(tempMonth)

    # extract day string and pad with zeros if necessary
    tempDay = tempLims[0].day
    if tempDay < 10:
        tempDayStr = "0" + str(tempDay)
    else:
        tempDayStr = str(tempDay)

    # construct date string
    tempDate = tempYearStr + "-" + tempMonthStr + "-" + tempDayStr

    # read in data for template
    stTemp = obspy.read(path + stat + "/" + chan + "/" + tempDate + "." + stat + "." + chan + ".noIR.MSEED")

    # basic preprocessing
    stTemp.detrend("demean")
    stTemp.detrend("linear")
    stTemp.taper(max_percentage=0.01, max_length=10.)

    # filter the data to each band
    stTemp.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # trim the data to the time ranges of template for each band
    stTemp.trim(starttime=tempLims[0],endtime=tempLims[1])

    # plot templates (for testing)
    #stTemp.plot()

    return stTemp

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/"
temp = obspy.read(templatePath + 'conservativeWaveforms.h5')[:1]

# define station and channel parameters
stat = ["PIG2"]
chans = ["HHZ","HHE","HHN"]

freq = [1,10]
templates = []
template_names = []
templateList = []
for t in temp:
    tempLims = [t.stats.starttime, t.stats.endtime]

    # make empty stream
    st = read()
    st.clear()

    for s in stat:
        for c in chans:
            st += makeTemplates(path,s,c,tempLims,freq)
    templateObj = Template(name=t.stats.starttime.isoformat()[-6:]+s.lower()+c.lower(),st=st,lowcut=freq[0],highcut=freq[1],samp_rate=t.stats.sampling_rate,filt_order=4,prepick=0)
    templateList.append(st)
    templates.append(templateObj)
    template_names.append(t.stats.starttime.isoformat()[-6:]+s.lower()+c.lower())
templates = Tribe(templates=templates)

# extract year string
tempYear = tempLims[0].year
tempYearStr = str(tempYear)

# extract month string and pad with zeros if necessary
tempMonth = tempLims[0].month
if tempMonth < 10:
    tempMonthStr = "0" + str(tempMonth)
else:
    tempMonthStr = str(tempMonth)

# extract day string and pad with zeros if necessary
tempDay = tempLims[0].day
if tempDay < 10:
    tempDayStr = "0" + str(tempDay)
else:
    tempDayStr = str(tempDay)

# construct date string
tempDate = tempYearStr + "-" + tempMonthStr + "-" + tempDayStr

# read in data for template
st = read()
st.clear()
for c in chans:
    st += obspy.read(path + stat[0] + "/" + c + "/" + tempDate + "." + stat[0] + "." + c + ".noIR.MSEED")

# basic preprocessing
st.detrend("demean")
st.detrend("linear")
st.taper(max_percentage=0.01, max_length=10.)
st.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

print(templateList)
print(template_names)
print(templateList[0])
print(st)

#detections = match_filter(template_names=template_names,template_list=templateList,st=st,threshold = 5,threshold_type="MAD",trig_int=6,plotvar =False,cores=8)
#print(detections)
