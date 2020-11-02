import obspy
import collections
from obspy.signal.cross_correlation import correlation_detector
import glob
import h5py
import numpy as np
import multiprocessing
from multiprocessing import Manager
from multiprocessing import set_start_method
import multiTemplateMatch
from multiTemplateMatch import multiTemplateMatch
from multiTemplateMatch import readInputTemplates
from multiTemplateMatch import makeTemplates
from multiTemplateMatch import readData
from multiTemplateMatch import removeRedundant
import time

# this code produces the detections in the folder 'multiTemplate'
# choose number of processors
nproc = 18

# define template match parameters for templates
ccThresh = 0.1
numTemp = 1
readTemplates = 1
threshLow = 0.4
threshHigh = 0.2
tolerance = 60
distance = 10
numComp = 2

# define path to data and templates
dataPath = "/media/Data/Data/PIG/MSEED/noIR/"
inputTemplatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/run3/"
outPath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"

# define station and channel parameters
stat = "PIG2"
chan = "H*"

# define filter parameters for templates
freqLow = [0.05,0.1]
freqHigh = [1, 10]

# get energy detector output to make new templates
templateIDs = range(numTemp)
if readTemplates == 0:
    energy_det_cc = readInputTemplates(inputTemplatePath,'short_waveforms.h5',"short_correlations_0.01-1Hz.h5",ccThresh)
    numTemp = len(energy_det_cc)
    makeTemplates(dataPath,outPath,stat,freqLow,freqHigh,energy_det_cc,numTemp)

else:
    print("Using " + str(numTemp) + " templates...")

# make vector of all filenames for a single channel
fileMat = []
dummyChan = "HHZ"
filePath = "/media/Data/Data/PIG/MSEED/noIR/" + stat + "/" + dummyChan + "/*"
files = glob.glob(filePath)
files.sort()
fileMat.extend(files)

# specify a specific file (for testing)
fileMat = ["/media/Data/Data/PIG/MSEED/noIR/PIG2/HHZ/2013-05-22.PIG2.HHZ.noIR.MSEED"]

# loop through all files- we will replace channel string with wildcard as we go
for f in fileMat:

    # start timer
    timer = time.time()

    # pull out day string for user output
    day = f.split("/")[9].split(".")[0]

    # read continuous data to scan
    stLow,stHigh = readData(f,freqLow,freqHigh,dummyChan)

    # start parallel pool and make shared memory containers
    detections = []
    detID = []
    manager = Manager()
    detections_shared = manager.list([])
    detID_shared = manager.list([])

    # define function for pmap
    def parFunc(templateID):
        # read in each band of templates
        tempLow = obspy.read(outPath + 'shortTemplates_resample/' + 'tempLow_' + str(templateID) +'.h5')
        tempHigh = obspy.read(outPath + 'shortTemplates_resample/' + 'tempHigh_' + str(templateID) +'.h5')

        # run the actual template matching code
        detections = multiTemplateMatch(tempLow,stLow,threshLow,tempHigh,stHigh,threshHigh,numComp,tolerance,distance)

        # give output
        print("Found " + str(len(detections)) + " detections on " + day + " with template " + str(templateID))

        # store results
        detections_shared.extend(detections)
        detID_shared.extend(templateID*np.ones((len(detections))))

    # call parallel function in properly guarded statement
    if __name__ == '__main__':
        multiprocessing.freeze_support()
        try:
            set_start_method("spawn")
        except:
            pass
        p = multiprocessing.Pool(processes=nproc)
        p.map(parFunc,templateIDs)
        p.close()
        p.join()

    # make normal lists from shared memory containers
    detections.extend([d for d in detections_shared])
    detID = detID_shared

    # sort results
    dTimes = []
    for d in detections:
        dTimes.append(d.ns)
    def func(t):
        return t.ns
    detections.sort(key=func)
    sortIdx = np.array(np.argsort(dTimes))
    detID = np.array(detID)[sortIdx]

    if len(detections) > 0:

        # remove redundant detections
        finalDetections,finalDetID = removeRedundant(detections,detID,tolerance)

        # save results
        #with open(outPath + 'multiTemplateDetections.txt', 'a') as f:
        #    for item in finalDetections:
        #        f.write("%s\n" % item)
        #with open(outPath + 'templateIDs.txt', 'a') as f:
        #    for item in finalDetID:
        #        f.write("%s\n" % item)

    # stop timer
    runtime = time.time() - timer

    if len(detections) > 0:
        print("Finished template matching on " + day + " in " + str(runtime) + " seconds and found " + str(len(finalDetections)) + " detections with " + str(numTemp) + " templates")
    else:
        print("Finished template matching on " + day + " in " + str(runtime) + " seconds and found 0 detections with " + str(numTemp) + " templates")
