import obspy
import collections
from obspy.signal.cross_correlation import correlation_detector
import glob
import h5py
import numpy as np
import multiprocessing
from multiprocessing import Manager
import multiTemplateMatch

# this code produces the detections in the folder 'multiTemplate'
# choose number of processors
nproc = 10

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/run3/"

# define station and channel parameters
stat = "PIG2"
chans = ["HHZ","HHE","HHN"]

# define template match parameters for first template
threshLow = 0.6
threshHigh = 0.2
tolerance = 60

# define filter parameters for templates
freqLow = [0.01,0.1]
freqHigh = [1, 10]

# read in tempates from energy detector
templates_all = obspy.read(templatePath + 'short_waveforms.h5')

# only need one channel here as templates get made in the code
templates_cc = []
for f in templates_all:
    if f.stats.channel == "HHZ":
        templates_cc.append(f)

# read in cross correlation results
cc_output = h5py.File(templatePath + "short_correlations_0.01-1Hz.h5",'r')
corrCoefs = np.array(list(cc_output['corrCoefs']))
sortInd = np.argsort(abs(corrCoefs))[::-1]
sortCorrCoefs = corrCoefs[sortInd]

# filter by cc coefficient
templates = []
ccThresh = 0.5
for i in range(len(templates_cc)):
      if abs(sortCorrCoefs[i]) > ccThresh:
          templates.append(templates_cc[sortInd[i]])
print("Using " + str(len(templates)) + " templates")

# make array for storing all detections
allDetections = []

# define function for pmap
def parFunc(template):
    tempLims = [template.stats.starttime, template.stats.endtime]
    detections = multiTemplateMatch.multiTemplateMatch(path,stat,chans,tempLims,freqLow,threshLow,tempLims,freqHigh,threshHigh,tolerance)
    detections_shared.extend(detections)

# start parallel pool and make shared memory containers
manager = Manager()
detections_shared = manager.list([])

# call the function in parallel
p = multiprocessing.Pool(processes=nproc)
p.map(parFunc,templates)

print("Finished template matching- consolidating detections")

# make normal lists from shared memory containers
allDetections.extend([d for d in detections_shared])

print("Finished consolidating detections- removing redundant detections")

# sort results
def func(t):
    return t.ns
allDetections.sort(key=func)

# remove redundant detections
finalDetections = [allDetections[0]]
for d in range(len(allDetections)):
        if allDetections[d] - allDetections[d-1] > 60:
            finalDetections.append(allDetections[d])

# save results
with open('/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/multiTemplateDetections.txt', 'w') as f:
    for item in finalDetections:
        f.write("%s\n" % item)
