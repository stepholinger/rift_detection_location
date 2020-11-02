import os
import multiprocessing
from multiprocessing import Manager
import time
import numpy as np

import obspy
from obspy import read
import obspyh5

import h5py

# functions to construct filename string of file to load
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

# function to make templates from a UTCDateTime object specifying bounds
def makeTemplate(path,stat,chan,tempLims,freq,filtType):

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
    #stTemp.plot()
    return stTemp

# parallelized implementation of the above
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

def makeTemplateList(tempH5,path,stat,chan,freq,filtType,readPar,nproc):

    # start timer and give output
    timer = time.time()
    print("Loading templates...")

    # make empty containers for templates and names
    tempTimes = []
    template_names = []

    # get template starttimes, endtimes and index
    for t in range(len(tempH5)):
        tempTimes.append([tempH5[t].stats.starttime,tempH5[t].stats.endtime,t])

    # clear initial template catalog for memory
    del tempH5

    # read and process templates in parallel
    if readPar:
        # start parallel pool and make shared memory containers
        manager = Manager()
        templates_shared = manager.list([])
        template_names_shared = manager.list([])

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
                st += makeTemplate(path,s,chan,tempLims,freq,filtType)
            #print(st)
            #st.plot()
            st.write('templates/' + str(freq[0]) + '-' + str(freq[1]) + 'Hz/template_'+str(t)+'.h5','H5',mode='a')
            template_names.append(s.lower()+"temp"+str(t))

    # stop timer and give output
    runtime = time.time() - timer
    #print("Loaded " + str(len(templates)) + " templates in " + str(runtime) + " seconds")

    # save list of template names
    hf = h5py.File('template_names.h5','w')
    hf.create_dataset('template_names',data=np.array(template_names,dtype='S'))
