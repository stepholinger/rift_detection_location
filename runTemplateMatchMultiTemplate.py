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
import time

# this code produces the detections in the folder 'multiTemplate'
# choose number of processors
nproc = 18

# define template match parameters for templates
readTemplates = 1
threshLow = 0.3
threshHigh = 0.2
tolerance = 60
numComp = 2

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/run3/"
outPath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/"

# define station and channel parameters
stat = "PIG2"
chan = "H*"

# define filter parameters for templates
freqLow = [0.01,0.1]
freqHigh = [1, 10]

# read in tempates from energy detector
energy_det_all = obspy.read(templatePath + 'short_waveforms.h5')

# only need one channel here as multiband, 3-component templates get made in the code
energy_det = []
for d in energy_det_all:
    if d.stats.channel == "HHZ":
        energy_det.append(d)

# read in cross correlation results
cc_output = h5py.File(templatePath + "short_correlations_0.01-1Hz.h5",'r')
corrCoefs = np.array(list(cc_output['corrCoefs']))
sortInd = np.argsort(abs(corrCoefs))[::-1]
sortCorrCoefs = corrCoefs[sortInd]

# filter by cc coefficient
ccThresh = 0.5
energy_det_cc = []
for c in range(len(sortCorrCoefs)):
      if abs(sortCorrCoefs[c]) > ccThresh:
          energy_det_cc.append(energy_det[sortInd[c]])

# make and save templates using time limits from detections above cc coefficient threshold
numTemp = len(energy_det_cc)
templateIDs = range(numTemp)
if readTemplates == 0:
    print("Making " + str(len(energy_det_cc)) + " templates...")
    for d in range(len(energy_det_cc)):
        tempLims = [energy_det_cc[d].stats.starttime, energy_det_cc[d].stats.endtime]
        stTempLow = multiTemplateMatch.makeTemplates(path,stat,"H*",tempLims,freqLow)
        stTempHigh = multiTemplateMatch.makeTemplates(path,stat,"H*",tempLims,freqHigh)
        stTempLow.write(outPath + '/shortTemplates/' + 'tempLow_' + str(d) +'.h5','H5',mode='a')
        stTempHigh.write(outPath + '/shortTemplates/' + 'tempHigh_' + str(d) +'.h5','H5',mode='a')
else:
    print("Using " + str(len(energy_det_cc)) + " templates...")

# make vector of all filenames for a single channel
fileMat = []
dummyChan = "HHZ"
filePath = "/media/Data/Data/PIG/MSEED/noIR/" + stat + "/" + dummyChan + "/*"
files = glob.glob(filePath)
files.sort()
fileMat.extend(files)

# specify a specific file (for testing)
#fileMat = ["/media/Data/Data/PIG/MSEED/noIR/PIG2/HHZ/2012-05-22.PIG2.HHZ.noIR.MSEED"]
#fileMat = fileMat[:10]

# loop through all files- we will replace channel string with wildcard as we go
for f in fileMat:

    # start timer
    timer = time.time()

    # make filename with wildcard channel
    fname = f.replace(dummyChan,"H*")

    # pull out day string for user output
    day = fname.split("/")[9].split(".")[0]

    # read files and do basic preprocessing
    stRaw = obspy.read(fname)
    stRaw.detrend("demean")
    stRaw.detrend("linear")
    stRaw.taper(max_percentage=0.01, max_length=10.)

    # copy the file
    stLow = stRaw.copy()
    stHigh = stRaw.copy()

    # filter and downsample the data to each band
    stLow.filter("bandpass",freqmin=freqLow[0],freqmax=freqLow[1])
    stHigh.filter("bandpass",freqmin=freqHigh[0],freqmax=freqHigh[1])
    stLow.resample(freqLow[1]*2)
    stHigh.resample(freqHigh[1]*2)

    # define function for pmap
    def parFunc(templateID):
        tempLow = obspy.read(outPath + '/shortTemplates/' + 'tempLow_' + str(templateID) +'.h5')
        tempHigh = obspy.read(outPath + '/shortTemplates/' + 'tempHigh_' + str(templateID) +'.h5')
        tempLow.resample(freqLow[1]*2)
        tempHigh.resample(freqHigh[1]*2)
        #tempLow.plot()
        #tempHigh.plot()
        detections = multiTemplateMatch.multiTemplateMatch(tempLow,stLow,threshLow,tempHigh,stHigh,threshHigh,numComp,tolerance)
        print("Found " + str(len(detections)) + " detections on " + day + " with template " + str(templateID))
        detections_shared.extend(detections)
        detID_shared.extend(templateID*np.ones((len(detections))))

    # start parallel pool and make shared memory containers
    detections = []
    detID = []
    manager = Manager()
    detections_shared = manager.list([])
    detID_shared = manager.list([])

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

    # remove redundant detections
    finalDetections = []
    finalDetID = []
    if len(detections) > 0:
        for d in range(len(detections)-1):
                if detections[d+1] - detections[d] > tolerance:
                    finalDetections.append(detections[d])
                    finalDetID.append(detID[d])

        # save results
        with open(outPath + 'multiTemplateDetections.txt', 'a') as f:
            for item in finalDetections:
                f.write("%s\n" % item)
        with open(outPath + 'templateIDs.txt', 'a') as f:
            for item in finalDetID:
                f.write("%s\n" % item)
    # stop timer
    runtime = time.time() - timer

    print("Finished template matching on " + day + " in " + str(runtime) + " seconds and found " + str(len(finalDetections)) + " detections with " + str(numTemp) + " templates")


# # define function for pmap
# def parFunc(template):
#     tempLims = [template.stats.starttime, template.stats.endtime]
#     detections = multiTemplateMatch.multiTemplateMatch(path,stat,chans,tempLims,freqLow,threshLow,freqHigh,threshHigh,tolerance)
#     detections_shared.extend(detections)
#
# def parFuncTest(template):
#     tempLims = [template.stats.starttime, template.stats.endtime]
#
# #detections_shared = []
# #for t in range(len(templates)):
# #    tempLims = [templates[t].stats.starttime, templates[t].stats.endtime]
# #    detections = multiTemplateMatch.multiTemplateMatch(path,stat,chans,tempLims,freqLow,threshLow,freqHigh,threshHigh,tolerance)
# #    detections_shared.extend(detections)
#
# # start parallel pool and make shared memory containers
# manager = Manager()
# detections_shared = manager.list([])
#
# #def startProc():
# #    print ('Starting ' + multiprocessing.current_process().name)
#
# if __name__ == '__main__':
#     import obspy
#     import collections
#     from obspy.signal.cross_correlation import correlation_detector
#     import glob
#     import h5py
#     import numpy as np
#     import multiprocessing
#     from multiprocessing import Manager
#     from multiprocessing import set_start_method
#     import multiTemplateMatch
#
#     multiprocessing.freeze_support()
#     try:
#         set_start_method("spawn")
#     except:
#         pass
#     p = multiprocessing.Pool(processes=nproc)
#     p.map(parFuncTest,templates)
#     p.close()
#     p.join()
#
# print("Finished template matching- consolidating detections")
#
# # make normal lists from shared memory containers
# allDetections.extend([d for d in detections_shared])
#
# print("Finished consolidating detections- removing redundant detections")
# # sort results
# def func(t):
#     return t.ns
# allDetections.sort(key=func)
#
# # remove redundant detections
# finalDetections = [allDetections[0]]
# for d in range(len(allDetections)):
#         if allDetections[d] - allDetections[d-1] > 60:
#             finalDetections.append(allDetections[d])
#
# # save results
# with open('/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/multiTemplateDetections.txt', 'w') as f:
#     for item in finalDetections:
#         f.write("%s\n" % item)
