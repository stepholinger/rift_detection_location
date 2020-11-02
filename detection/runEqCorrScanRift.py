import os
import multiprocessing
from multiprocessing import Manager
import time

import obspy
from obspy import read
import obspyh5

import h5py
import numpy as np

import eqcorrscan
from eqcorrscan.core.template_gen import template_gen
from eqcorrscan.core.match_filter import Tribe
from eqcorrscan.core.match_filter import Template
from eqcorrscan.core.match_filter import match_filter

from eqCorrScanUtils import getFname
from eqCorrScanUtils import makeTemplate
from eqCorrScanUtils import parFunc
from eqCorrScanUtils import makeTemplateList

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/run3/"

# set cross correlation coefficient threshold
ccThresh = 0.5

# read hdf5 file of results from correlation
output = h5py.File(templatePath + 'short_correlations.h5','r')

# extract data from hdf5 file
corrCoefs = list(output['corrCoefs'])
corrCoefs = np.array(corrCoefs)

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
tempH5_all_chan = obspy.read(templatePath + 'short_waveforms.h5')

# just get desired channel
chan = 'HHZ'
tempH5_all_cc = []
for f in range(len(tempH5_all_chan)):
    if tempH5_all_chan[f].stats.channel == chan:
        tempH5_all_cc.append(tempH5_all_chan[f])

# sort waveforms and shifts by correlation coeficient
sortInd = np.argsort(abs(corrCoefs))[::-1]
sortCorrCoefs = corrCoefs[sortInd]

# filter by xcorr coeff
tempH5 = []
for f in range(len(tempH5_all_cc)):
    if abs(sortCorrCoefs[f]) > ccThresh:
        tempH5.append(tempH5_all_cc[sortInd[f]])

numTemp = len(tempH5)
print(str(numTemp) + " templates will be made")
#tempH5 = obspy.read("/home/setholinger/Documents/Code/python/detection/templates/0.01-1Hz/2012-04-02.h5")
#numTemp = 1

# define parallel parameters
readPar = 0
nproc = 8

# define station and channel parameters
stat = ["PIG2"]
chan = "HH*"
numChan = 3
freq = [0.01,1]
#freq = [1,10]
filtType = "bandpass"

# enter the buffer used on the front and back end to produce templates
# this duration will be removed after filtering from the front and back ends of the template
#buff = [2*60,2*60]
trimIdx = 600

# make and save templates (don't re run this)
#makeTemplateList(tempH5,path,stat,chan,freq,filtType,readPar,nproc)

# set start date for scan
startDate = obspy.UTCDateTime(2012,11,21,0,0)
numDays = 720

# set template chunk size
blockSize = 50

# set minimum distance in seconds between detections
tolerance = 60

# set threshold for detector
threshold = 6

for i in range(numDays):

    # make variable to store detections
    detections = []

    # get UTCDateTime for current day
    currentDate = startDate + 86400*i

    try:

        # read in data to scan
        st = read()
        st.clear()
        for s in stat:
          fname = getFname(path,s,chan,currentDate)
          print("Loading " + fname + "...")
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

        for j in range(int(numTemp/blockSize)+1):
        #for j in range(1):

            # start timer and give output
            timer = time.time()

    	    # get templates and template names
            templates = []
            template_names = []
            for k in range(j*blockSize,(j+1)*blockSize):

                try:
                    # read template file
                    stTemp = obspy.read('templates/' + str(freq[0]) + '-' + str(freq[1]) + 'Hz/template_'+str(k)+'.h5')

                    # occasionally templates are one sample too short- trim to the correct number of samples
                    # this handling is a bit clumsy- redo later
                    for c in range(numChan):
                        stTemp[c].data=stTemp[c].data[:trimIdx]
                    templates.append(stTemp)
                    template_names.append("template_" + str(k))
                except:
                     pass

            # run eqcorrscan's match filter routine
            det = match_filter(template_names=template_names,template_list=templates,st=st,threshold=threshold,threshold_type="MAD",trig_int=tolerance,cores=20)

            # append to list
            detections.extend(det)

            # stop timer and give output
            runtime = time.time() - timer

            # give some output
            if blockSize*(j+1) >= numTemp:
                print("Scanned " + currentDate.date.strftime("%Y-%m-%d") + " with " + str(len(templates)) + " templates (" + str(numTemp) + "/" + str(numTemp) + ") in " + str(round(runtime,2)) + " seconds and found " + str(len(det)) + " detections")
            else:
                print("Scanned " + currentDate.date.strftime("%Y-%m-%d") + " with " + str(len(templates)) + " templates (" + str(blockSize*(j+1)) + "/" + str(numTemp) + ") in " + str(round(runtime,2)) + " seconds and found " + str(len(det)) + " detections")

        # sort detections chronologically
        detections.sort()

        # loop through all detections and eliminate redundant detections
        for d in range(len(detections)-1):
            if abs(detections[d].detect_time - detections[d+1].detect_time) < tolerance:
                detections[d] = []
        detections = list(filter(None,detections))

        print(str(len(detections)) + " detections found on " + currentDate.date.strftime("%Y-%m-%d") + " after removing duplicates")

        for detections in detections:
            detections.write('templateDetections.csv',append=True)
    except:
        print("Skipping " + currentDate.date.strftime("%Y-%m-%d"))
