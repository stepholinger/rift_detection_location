import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import numpy as np
import h5py
from eqcorrscan.core.match_filter import read_detections


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
    elif filtType == "none":
        pass
    # make a copy so we only have to read once for all detections on a given day
    st = stTemp.copy()

    # trim the data to the time ranges of template for each band
    stTemp.trim(starttime=tempLims[0],endtime=tempLims[1])

    return stTemp,st

# set path
dataPath = "/media/Data/Data/PIG/MSEED/noIR/"
outPath = '/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/'
type = 'short'

# read detection file
csv = 0
txt = 1
if csv:
    det = read_detections(outPath + 'templateDetections_0.01-1Hz.csv')
if txt:
    det = [line.rstrip('\n') for line in open(outPath + 'multiTemplateDetections_' + type + '.txt')]

# set snippet length in seconds
snipLen = 300

# add buffer in front of waveform
buffer = 200

# set filter and filter waveforms- prefilt indicates waveforms already filtered
freq = [0.05,1]
filtType = "bandpass"

# set desired channel
stat = 'PIG2'
chan = 'HHE'

waveform_matrix = np.zeros((len(det),(snipLen+buffer)*freq[1]*2+1),'float64')

for i in range(len(det)):

    if csv:
        tempLims = [det[i].detect_time-buffer,det[i].detect_time+snipLen]
    if txt:
        tempLims = [obspy.UTCDateTime(det[i])-buffer,obspy.UTCDateTime(det[i])+snipLen]

    #read in whole day and preprocess if current detection is not on same day as last one
    if i == 0 or tempLims[0].day != st[0].stats.starttime.day:
        event,st = makeTemplate(dataPath,stat,chan,tempLims,freq,filtType)
    else:
        event = st.copy()
        event.trim(starttime=tempLims[0],endtime=tempLims[1])

    event.write(outPath + type + '_waveforms_' + chan + '_' + str(freq[0]) + "-" + str(freq[1]) + 'Hz.h5','H5',mode='a')

    waveform_matrix[i,:event[0].stats.npts] = event[0].data

    # give the user some output
    print("Retreived " + str(round(i/len(det)*100)) + "% of events")

waveFile = h5py.File(outPath + type + "_waveform_matrix_" + chan + '_' + str(freq[0]) + "-" + str(freq[1]) + "Hz.h5",'w')
waveFile.create_dataset("waveforms",data=waveform_matrix)
