import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import numpy as np
import h5py
from eqcorrscan.core.match_filter import read_detections
import matplotlib.pyplot as plt
import scipy

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

    # make a copy so we only have to read once for all detections on a given day
    st = stTemp.copy()

    # trim the data to the time ranges of template for each band
    stTemp.trim(starttime=tempLims[0],endtime=tempLims[1])

    return stTemp,st

# set path
dataPath = "/media/Data/Data/PIG/MSEED/noIR/"
outPath = '/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/'

# read detection file
det = [line.rstrip('\n') for line in open(outPath + 'multiTemplateDetections.txt')]

# set snippet length in seconds
fs = 100
snipLen = 300

# set filter and filter waveforms- prefilt indicates waveforms already filtered
freq = [0.05,0.1]
filtType = "bandpass"

# set desired channel
stat = 'PIG2'
chan = 'HHZ'

# open file for output
outFile = h5py.File(outPath + "KE_" + str(freq[0]) + "-" + str(freq[1]) + "Hz.h5","w")

# make arrays for storing output
eventKE = np.zeros((len(det)))
noiseKE = np.zeros((len(det)))
dailyMeanEventKE = []
dailyMeanNoiseKE = []

for i in range(len(det)):

    tempLims = [obspy.UTCDateTime(det[i]),obspy.UTCDateTime(det[i])+snipLen]

    if i > 0 and tempLims[0].day != st[0].stats.starttime.day:
        dailyMeanEventKE.append(np.mean(dailyMeanEnergy_event))
        dailyMeanNoiseKE.append(np.mean(dailyMeanEnergy_noise))

    if i == 0 or tempLims[0].day != st[0].stats.starttime.day:
        dailyMeanEnergy_event = []
        dailyMeanEnergy_noise = []
        event,st = makeTemplate(dataPath,stat,chan,tempLims,freq,filtType)
        #event = event[0]
    else:
        event = st.copy()
        event.trim(starttime=tempLims[0],endtime=tempLims[1])
        #event = event[0]

    # get noise window preceding event
    noise = st.copy()
    noise.trim(starttime=tempLims[0]-snipLen,endtime=tempLims[0])

    # calculate KE for both
    event_energy = scipy.integrate.trapz(np.multiply(event[0].data,event[0].data))
    noise_energy = scipy.integrate.trapz(np.multiply(noise[0].data,noise[0].data))
    dailyMeanEnergy_event.append(event_energy)
    dailyMeanEnergy_noise.append(noise_energy)

    # save output
    eventKE[i] = event_energy
    noiseKE[i] = noise_energy

    # give the user some output
    print("Retrieved energy for " + str(round(i/len(det)*100)) + "% of events")

# make plot
plt.plot(dailyMeanEventKE,color='k')
plt.plot(dailyMeanNoiseKE,color='g')
plt.yscale('log')
plt.show()

# write output to file
outFile.create_dataset("events",data=eventKE)
outFile.create_dataset("noise",data=noiseKE)
outFile.create_dataset("events_daily_mean",data=dailyMeanEventKE)
outFile.create_dataset("noise_daily_mean",data=dailyMeanNoiseKE)

# close output file
outFile.close()
