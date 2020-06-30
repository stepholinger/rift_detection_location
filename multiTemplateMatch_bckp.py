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

def multiTemplateMatch(path,stat,chans,tempLimsLow,freqLow,threshLow,tempLimsHigh,freqHigh,threshHigh,tolerance):

    # internal control over detection plotting
    plotting = False

    # make a couple useful list
    detectionArray = [[],[],[]]
    allDetections = []

    for c in range(len(chans)):

        # make empty vector of detections
        detections = []

        # get templates for low and high frequency bands
        stTempLow = makeTemplates(path,stat,chans[c],tempLimsLow,freqLow)
        stTempHigh = makeTemplates(path,stat,chans[c],tempLimsHigh,freqHigh)
        #stTempLow.plot()

        # define cross correlation search parameters
        distance = 10

        # make vector of all filenames
        fileMat = []
        filePath = "/media/Data/Data/PIG/MSEED/noIR/" + stat + "/" + chans[c] + "/*"
        files = glob.glob(filePath)
        files.sort()
        fileMat.append(files)

        # specify a specific file (for testing)
        fileMat = [["/media/Data/Data/PIG/MSEED/noIR/PIG2/" + chans[c] + "/2012-05-22." + stat + "." + chans[c] + ".noIR.MSEED"]]

        # loop through all files
        for f in range(len(fileMat[0])):

            try:

                timer = time.time()

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
                differences[differences < -1*distance] = np.nan

                try:
                    # find closest low frequency detection for each high frequency detection
                    minDiffs = np.nanmin(differences,axis=0)

                    # keep if the time difference is less than user-inputted threshold
                    detectDiffs = np.where(minDiffs < tolerance,1,0)

                    # fill detection vector with times returned by high frequency template match
                    for d in range(len(detectDiffs)):
                        if detectDiffs[d] == 1:
                            detections.append(detectionsHigh[d].get("time"))
                    print("Found " + str(len(detections)) + " detections in " + fname)
                except:
                        print("No detections for " + fname + "\n")

            except:
                print("Skipping " + fname + "\n")

        # fill array with detections from each file
        detectionArray[c].append(detections)

    # iterate through first two channels
    for c in range(len(chans)-1):

        # iterate through list of detections from channel c
        for i in range(len(detectionArray[c][0])):

        # iterate through list of detections from channel c+1
            for j in range(len(detectionArray[c+1][0])):

                # if both detections are closely spaced in time, add to list of all detections
                if np.abs(detectionArray[c][0][i] - detectionArray[c+1][0][j]) < tolerance:
                    allDetections.append(detectionArray[c][0][i])

            if c == 0:
                # iterate through list of detections from channel c+2 if c == 0 (compare first and third channel)
                for j in range(len(detectionArray[c+2][0])):

                    # if both detections are closely spaced in time, add to list of all detections
                    if np.abs(detectionArray[c][0][i] - detectionArray[c+2][0][j]) < tolerance:
                        allDetections.append(detectionArray[c+2][0][j])

        runtime = time.time() - timer
        print("Finished detections for " + fname + " in " + str(runtime) + " seconds\n")

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
