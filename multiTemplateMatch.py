import obspy
from obspy.signal.cross_correlation import correlation_detector
import glob
import numpy as np
import copy
import time

def readInputTemplates(path,wave_fname,corr_fname,ccThresh):
    # read in templates from energy detector
    energy_det_all = obspy.read(path + wave_fname)

    # only need one channel here as multiband, 3-component templates get made in the code
    energy_det = []
    for d in energy_det_all:
        if d.stats.channel == "HHZ":
            energy_det.append(d)

    # read in cross correlation results
    cc_output = h5py.File(path + corr_fname,'r')
    corrCoefs = np.array(list(cc_output['corrCoefs']))
    sortInd = np.argsort(abs(corrCoefs))[::-1]
    sortCorrCoefs = corrCoefs[sortInd]

    # filter by cc coefficient
    energy_det_cc = []
    for c in range(len(sortCorrCoefs)):
          if abs(sortCorrCoefs[c]) > ccThresh:
              energy_det_cc.append(energy_det[sortInd[c]])

    return energy_det_cc


def makeTemplates(dataPath,outPath,stat,freqLow,freqHigh,energy_det_cc,numTemp):
    # make and save templates using time limits from detections above cc coefficient threshold
    print("Making " + str(len(energy_det_cc)) + " templates...")
    for d in range(numTemp):
        tempLims = [energy_det_cc[d].stats.starttime, energy_det_cc[d].stats.endtime]
        stTempLow = makeSingleTemplates(dataPath,stat,"H*",tempLims,freqLow)
        stTempHigh = makeSingleTemplates(dataPath,stat,"H*",tempLims,freqHigh)
        stTempLow.write(outPath + '/shortTemplates_resample/' + 'tempLow_' + str(d) +'.h5','H5',mode='a')
        stTempHigh.write(outPath + '/shortTemplates_resample/' + 'tempHigh_' + str(d) +'.h5','H5',mode='a')


def readData(filename,freqLow,freqHigh,dummyChan):
    # make filename with wildcard channel
    fname = filename.replace(dummyChan,"H*")

    # read files and do basic preprocessing
    stRaw = obspy.read(fname)
    stRaw.detrend("demean")
    stRaw.detrend("linear")
    stRaw.taper(max_percentage=0.01, max_length=10.)

    # copy the file
    stLow = stRaw.copy()
    stHigh = stRaw.copy()

    # filter and downsample the data to each band
    stLow.resample(freqLow[1]*3)
    stHigh.resample(freqHigh[1]*3)
    stLow.filter("bandpass",freqmin=freqLow[0],freqmax=freqLow[1])
    stHigh.filter("bandpass",freqmin=freqHigh[0],freqmax=freqHigh[1])

    return stLow,stHigh


def removeRedundant(detections,detID,tolerance):
    # remove redundant detections
    finalDetections = []
    finalDetID = []
    for d in range(len(detections)-1):
            if detections[d+1] - detections[d] > tolerance:
                finalDetections.append(detections[d])
                finalDetID.append(detID[d])

    return finalDetections,finalDetID


def makeSingleTemplate(path,stat,chan,tempLims,freq):

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
    stTemp.resample(freq[1]*3)

    # filter the data to each band
    stTemp.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # trim the data to the time ranges of template for each band
    stTemp.trim(starttime=tempLims[0],endtime=tempLims[1])

    # plot templates (for testing)
    #stTemp.plot()

    return stTemp

def multiTemplateMatch(stTempLow,stLow,threshLow,stTempHigh,stHigh,threshHigh,numComp,tolerance,distance):

    # make a couple useful list
    detectionsTemp = []
    detections = []

    # iterate through each channel
    for s in range(len(stTempLow)):

        # call the template matching function in each band
        #detectionsLow,sl = correlation_detector(obspy.Stream(stLow[s]),obspy.Stream(stTempLow[s]),threshLow,tolerance)
        #detectionsHigh,sh = correlation_detector(obspy.Stream(stHigh[s]),obspy.Stream(stTempHigh[s]),threshHigh,tolerance)
        detectionsLow,sl = correlation_detector(obspy.Stream(stLow[s]),obspy.Stream(stTempLow[s]),threshLow,distance)
        detectionsHigh,sh = correlation_detector(obspy.Stream(stHigh[s]),obspy.Stream(stTempHigh[s]),threshHigh,distance)
        #print(len(detectionsLow))
        #print(len(detectionsHigh))

        # get all high frequency trigger times for today
        detHighTimes = []
        for i in range(len(detectionsHigh)):
            detHighTimes.append(detectionsHigh[i].get("time"))

        # loop through all low frequency triggers for today
        for i in range(len(detectionsLow)):
            detLowTime = detectionsLow[i].get("time")

            # calculate time difference between low freq trigger and all high freq triggers
            diffs = np.subtract(detLowTime,detHighTimes)

            # only interested in positive values of 'diffs', which indicates high freq trigger first
            diffs[diffs < -1*tolerance] = float("nan")

            # save low freq trigger if a high freq trigger is sufficiently close
            if len(diffs) > 0:
                if min(diffs) < tolerance:
                    detectionsTemp.append(detLowTime)

    # sort detections chronologically
    detectionsTemp.sort()

    #print(detectionsTemp)
    # save detections if they show up on desired number of components
    if len(detectionsTemp) > 0:
        for d in range(len(detectionsTemp)-numComp-1):
            #print(detectionsTemp[d+numComp-1] - detectionsTemp[d])
            if detectionsTemp[d+numComp-1] - detectionsTemp[d] < tolerance:
                detections.append(detectionsTemp[d])

    return detections


def multiTemplateMatchDeprecated(path,stat,chans,tempLims,freqLow,threshLow,freqHigh,threshHigh,tolerance):

    # internal control over detection plotting
    plotting = False

    # define cross correlation search parameters
    distance = 10

    # make a couple useful list
    detectionArray = [[],[],[]]
    allDetections = []

    # make vector of all filenames for a single channel
    fileMat = []
    filePath = "/media/Data/Data/PIG/MSEED/noIR/" + stat + "/" + chans[0] + "/*"
    files = glob.glob(filePath)
    files.sort()
    fileMat.extend(files)

    # specify a specific file (for testing)
    fileMat = ["/media/Data/Data/PIG/MSEED/noIR/PIG2/" + chans[0] + "/2012-05-22." + stat + "." + chans[0] + ".noIR.MSEED"]

    # get templates for low and high frequency bands
    stTempLow = makeTemplates(path,stat,"H*",tempLims,freqLow)
    stTempHigh = makeTemplates(path,stat,"H*",tempLims,freqHigh)

    # loop through all files- we will replace channel string with wildcard as we go
    for f in range(len(fileMat)):

        timer = time.time()

        # get filename from full path
        fname = fileMat[f]

        # make filename with wildcard channel
        fname = fname.replace(chans[0],"H*")

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

        # filter the data to each band
        stLow.filter("bandpass",freqmin=freqLow[0],freqmax=freqLow[1])
        stHigh.filter("bandpass",freqmin=freqHigh[0],freqmax=freqHigh[1])

        # call the template matching function in each band
        detectionsLow,sl = correlation_detector(stLow,stTempLow,threshLow,distance)
        detectionsHigh,sh = correlation_detector(stHigh,stTempHigh,threshHigh,distance)
        print(detectionsLow)
        print(detectionsHigh)
        # at this point, channel consolidation is roughly done
        # carry out multiband consolidation

        # get all high frequency times
        detHighTimes = []
        for i in range(len(detectionsHigh)):
            detHighTimes.append(detectionsHigh[i].get("time"))

        for i in range(len(detectionsLow)):
            detLowTime = detectionsLow[i].get("time")
            # calculate difference between current detection and all high frequency detections
            diffs = abs(np.subtract(detLowTime,detHighTimes))

            # save low frequency detections if minimum difference is less than threshold time
            if any(diffs) and min(diffs) < tolerance:
                allDetections.append(detLowTime)
        runtime = time.time() - timer
        print("Finished detections for " + day + " in " + str(runtime) + " seconds\n")

    # sort detections chronologically
    allDetections.sort()

    # get indices of redundant detections
    removeInd = []
    for d in range(len(allDetections)-1):
        if allDetections[d+1] - allDetections[d] < 60:
            removeInd.append(d+1)

    # replace redundant detections with arbitrary placeholder
    for r in removeInd:
        allDetections[r] = obspy.UTCDateTime(0)

    # make final list of detections and fill with all values not equal to placeholder
    finalDetections = []
    finalDetections[:] = [x for x in allDetections if x != obspy.UTCDateTime(0)]

    # give the user some output
    print(str(len(finalDetections)) + " detections found over " + str(len(fileMat[0])) + " days \n")

    if plotting:

        # load all traces into one stream for plotting
        stHigh += stLow
        stHigh += stRaw
        plotWinLen = tempLimsLow[1] - tempLimsLow[0]

        # plot each detection
        for d in range(len(finalDetections)):
            stHigh.plot(starttime = finalDetections[d], endtime = finalDetections[d]+plotWinLen,equal_scale = False)

    return finalDetections
