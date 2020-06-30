import obspy
from obspy.signal.cross_correlation import correlation_detector
import glob
import numpy as np
import copy
import time

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

def similarity_component_thres(ccs, thres, num_components):
    ccmatrix = np.array([tr.data for tr in ccs])
    header = dict(sampling_rate=ccs[0].stats.sampling_rate,
                  starttime=ccs[0].stats.starttime)
    comp_thres = np.sum(ccmatrix > thres, axis=0) >= num_components
    data = np.mean(ccmatrix, axis=0) * comp_thres
    return obspy.Trace(data=data, header=header)

def multiTemplateMatch(stTempLow,stLow,threshLow,stTempHigh,stHigh,threshHigh,numComp,tolerance):

    # make a useful list
    detections = []

    # define similary functions
    def simfLow(ccs):
        return similarity_component_thres(ccs,threshLow,numComp)

    def simfHigh(ccs):
        return similarity_component_thres(ccs,threshHigh,numComp)

    # call the template matching function in each band
    detectionsLow,sl = correlation_detector(stLow,stTempLow,threshLow,tolerance,similarity_func=simfLow)
    detectionsHigh,sh = correlation_detector(stHigh,stTempHigh,threshHigh,tolerance,similarity_func=simfHigh)

    #print(detectionsLow)
    #print(detectionsHigh)

    # get all high frequency trigger times for today
    detHighTimes = []
    for i in range(len(detectionsHigh)):
        detHighTimes.append(detectionsHigh[i].get("time"))

    # loop through all low frequency triggers for today
    for i in range(len(detectionsLow)):
        detLowTime = detectionsLow[i].get("time")

        # calculate time difference between low freq trigger and all high freq triggers
        diffs = abs(np.subtract(detLowTime,detHighTimes))

        # save low freq trigger if a high freq trigger is sufficiently close
        if any(diffs) and min(diffs) < tolerance:
            detections.append(detLowTime)
            #stLow.plot(starttime=detLowTime,endtime=detLowTime+300)

    # sort detections chronologically
    detections.sort()

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
