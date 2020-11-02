import obspy
from obspy.signal.cross_correlation import correlation_detector
import glob
import numpy as np
import copy
import thread

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

def multiTemplateMatch(path,stat,chans,tempLimsLow,freqLow,threshLow,tempLimsHigh,freqHigh,threshHigh):

    # make empty vector of detections
    detections = []

    # get templates for low and high frequency bands
    stTempLow = makeTemplates(path,stat,chans[c],tempLimsLow,freqLow)
    stTempHigh = makeTemplates(path,stat,chans[c],tempLimsHigh,freqHigh)

    # define cross correlation search parameters
    distance = 10
    tolerance = 10*60

    # make vector of all filenames
    fileMat = []
    filePath = "/media/Data/Data/PIG/MSEED/noIR/" + stat + "/" + chans[c] + "/*"
    files = glob.glob(filePath)
    files.sort()
    fileMat.append(files)

    # specify a specific file (for testing)
    fileMat = [["/media/Data/Data/PIG/MSEED/noIR/PIG2/" + chans[c] + "/2012-05-09." + stat + "." + chans[c] + ".noIR.MSEED"]]
                #"/media/Data/Data/PIG/MSEED/noIR/PIG2/HHZ/2012-08-21.PIG2.HHZ.noIR.MSEED",
                #"/media/Data/Data/PIG/MSEED/noIR/PIG2/HHZ/2012-11-10.PIG2.HHZ.noIR.MSEED"]]

    # loop through all files
    for f in range(len(fileMat[0])):

        # get filename from full path
        fname = fileMat[0][f]

        # pull out day string for output
        day = fname.split("/")[9].split(".")[0]

        # read file
        stLow = obspy.read(fname)

        # basic preprocessing
        stLow.detrend("demean")
        stLow.detrend("linear")
        stLow.taper(max_percentage=0.01, max_length=10.)

        # copy the file
        stHigh = stLow.copy()
        stRaw = stLow.copy()

        # filter the data to each band
        stLow.filter("bandpass",freqmin=freqLow[0],freqmax=freqLow[1])
        stHigh.filter("bandpass",freqmin=freqHigh[0],freqmax=freqHigh[1])

        # call the template matching function in each band
        detectionsLow,sl = correlation_detector(stLow,stTempLow,threshLow,distance)
        detectionsHigh,sh = correlation_detector(stHigh,stTempHigh,threshHigh,distance)

        # extract time values from detections
        differences = np.zeros((len(detectionsLow),len(detectionsHigh)))

        # calculate detection time difference for each pair of detections
        for n in range(len(detectionsLow)):
            for m in range(len(detectionsHigh)):
                differences[n,m] = detectionsLow[n].get("time")-detectionsHigh[m].get("time")

        # replace pairs where low frequency detection is first with nan
        differences[differences < -2] = np.nan

        # find closest low frequency detection for each high frequency detection
        minDiffs = np.nanmin(differences,axis=0)

        # keep if the time difference is less than user-inputted threshold
        detectDiffs = np.where(minDiffs < tolerance,1,0)

        # fill detection vector with times returned by high frequency template match
        for d in range(len(detectDiffs)):
            if detectDiffs[d] == 1:
                detections.append(detectionsHigh[d].get("time"))

    return detections


def detect(path,stat,chans,tempLimsLow,freqLow,threshLow,tempLimsHigh,freqHigh,threshHigh):

    # make a couple useful list
    detectionArray = [[],[],[]]
    allDetections = []

    for c in range(len(chans)):

        detections = thread.start_new_thread(multiTemplateMatch, (path,stat,chans[c],tempLimsLow,freqLow,threshLow,tempLimsHigh,freqHigh,threshHigh))

        # fill array with
        detectionArray[c].append(detections)

    for c in range(len(chans)-1):
        for i in range(len(detectionArray[c][0])):
            for j in range(len(detectionArray[c+1][0])):
                if np.abs(detectionArray[c][0][i] - detectionArray[c+1][0][j]) < 60:
                    allDetections.append(detectionArray[c][0][i])
            if c  == 0:
                for j in range(len(detectionArray[c+2][0])):
                    if np.abs(detectionArray[c][0][i] - detectionArray[c+2][0][j]) < 60:
                        allDetections.append(detectionArray[c+2][0][j])

    # sort detections chronologically
    allDetections.sort()

    # remove redundant detections
    removeInd = []
    for d in range(len(allDetections)-1):
        if allDetections[d+1] - allDetections[d] < 60:
            removeInd.append(d+1)

    for r in removeInd:
        allDetections[r] = obspy.UTCDateTime(0)

    finalDetections = []
    finalDetections[:] = [x for x in allDetections if x != obspy.UTCDateTime(0)]

    # load all traces into one stream for plotting
    stHigh += stLow
    stHigh += stRaw
    plotWinLen = 300

    # give the user some output
    print(str(len(finalDetections)) + " detections found over " + str(len(fileMat)) + " days")

    # plot each detection
    for d in range(len(finalDetections)):
        stHigh.plot(starttime = finalDetections[d], endtime = finalDetections[d]+plotWinLen,equal_scale = False)

    return finalDetections
