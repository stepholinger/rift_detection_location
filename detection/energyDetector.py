import numpy as np
import obspy
import obspyh5
from energyDetectorUtils import removeDoubleCounting
from energyDetectorUtils import getFiles
from energyDetectorUtils import getTriggers
from energyDetectorUtils import getEnergyPeaks
from energyDetectorUtils import saveWaveforms
from energyDetectorUtils import saveDetections
from energyDetectorUtils import testPlot

path = "/media/Data/Data/PIG/MSEED/noIR/"
outPath = "/home/setholinger/Documents/Projects/PIG/detections/energy/run3/"
stat = "PIG2"
chan = "all"
fileType = "MSEED"
fs = 100

# specify two frequency bands, prominence, and allowable number of seconds between low and high frequency detections
freqLow = [0.01,1]
freqHigh = [1,10]
prominence = 0.1
tolerance = 120
multiplier = 10

# specify window to pull template around detection in seconds
buffer = [2*60,3*60]

# get all files of desired station and channel
files = getFiles(chan,path,stat)

# first day is garbage, so remove it
files = files[1:]

# scan a specific day (for testing)
#day = "2012-05-09"
#dayFile = path + stat + "/HH*/" + day + "." + stat + ".HH*.noIR.MSEED"
#files = [dayFile]

# make empty arrays to store detection times
detShort = []
detLong = []

# iterate through all filestrings
for f in files:

    # make empty arrays to store detection times
    detShortTemp = []
    detLongTemp = []

    # give some output
    print("Scanning " + f + "...")

	# read data files for all channels into one stream object
    st = obspy.read(f)

    # basic preprocessing
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # copy for other bands
    stLow = st.copy()
    stHigh = st.copy()

    # filter the data
    stLow.filter("bandpass",freqmin=freqLow[0],freqmax=freqLow[1])
    stHigh.filter("bandpass",freqmin=freqHigh[0],freqmax=freqHigh[1])

    # run trigger-finding algorithm for each channel
    for s in range(len(st)):

        # make empty arrays to store detections from current channel
        detShortChan = []
        detLongChan = []
        detShortDay = []
        detLongDay = []

        # calculate kinetic energy and find peaks
        peaksLow,energyLow = getEnergyPeaks(stLow[s],prominence,tolerance,fs)
        peaksHigh,energyHigh = getEnergyPeaks(stHigh[s],prominence,tolerance,fs)

        # plot trace and energy peaks (for testing)
        #testPlot(energyHigh,peaksHigh,energyLow,peaksLow)

        # check if peaks are concurrent in each band
        for h in range(len(peaksHigh)):
            for l in range(len(peaksLow)):

                # skip to next iteration if low frequency detection is first
                if peaksLow[l] - peaksHigh[h] < 0:
                    continue

                # get triggers when peaks are sufficiently close to each other
                detShortChan,detLongChan = getTriggers(st[s],energyLow,peaksLow[l:l+2],peaksHigh[h],tolerance,buffer[0],fs,detShortChan,detLongChan,multiplier*0.75)

        # remove double counting within current channel
        detShortChan = removeDoubleCounting(detShortChan,tolerance)
        detLongChan = removeDoubleCounting(detLongChan,tolerance*multiplier)

        # append to list for current channel
        detShortTemp.extend(detShortChan)
        detLongTemp.extend(detLongChan)

    # sort detections
    detShortTemp.sort()
    detLongTemp.sort()

    # if a detection is repeated 2 times, save it
    for d in range(len(detShortTemp)-1):
        if detShortTemp[d+1] - detShortTemp[d] < tolerance:
            detShortDay.append(detShortTemp[d])
    for d in range(len(detLongTemp)-1):
        if detLongTemp[d+1] - detLongTemp[d] < tolerance:
            detLongDay.append(detLongTemp[d])

    # remove double counting from daily list
    detShortDay = removeDoubleCounting(detShortDay,tolerance)
    detLongDay = removeDoubleCounting(detLongDay,tolerance*multiplier)

    # append to final list of detections
    detShort.extend(detShortDay)
    detLong.extend(detLongDay)

    # save waveform snippets of detections from current day
    saveWaveforms(detShortDay,st,buffer,outPath,'short')
    saveWaveforms(detLongDay,st,[buffer[0]*multiplier*0.75,buffer[1]*multiplier*1.5],outPath,'long')

# save list of final detections
saveDetections(detShort,outPath,'short')
saveDetections(detLong,outPath,'long')
