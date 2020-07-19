import obspy
import obspyh5
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py
from scipy import stats

# define path to data and templates
type = "short"
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/stack/"
fs = 100
stack = 1

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]
all_waveforms = obspy.read(templatePath + type + '_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')

#filter waveforms
freq = [0.05,1]
#all_waveforms.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

# just get desired channel
chan = 'HHZ'
waveforms = []
for f in all_waveforms:
    if f.stats.channel == chan:
        waveforms.append(f)

numTemp = len(waveforms)
traceLen = waveforms[0].stats.npts

# run 1
#masterInd = 3189
# run 2
#masterInd = 1307
# run 3
masterInd = 10679

if stack:
    print("Using stack as master...")

else:
    print("Using event at " + waveforms[masterInd].stats.starttime.isoformat() + " as master...")

# set cross correlation coefficient threshold- use MAD or set manually
madFlag = 0
if madFlag == 0:
    ccThresh = 0.2

# read hdf5 file of results from correlation
output = h5py.File(templatePath + type + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')

# extract data from hdf5 file
corrCoefs = list(output['corrCoefs'])
shifts = list(output['shifts'])

# load data into array
corrCoefs = np.array(corrCoefs)
shifts = np.array(shifts)

# sort waveforms and shifts by correlation coeficient
sortInd = np.argsort(abs(corrCoefs))[::-1]
sortCorrCoefs = corrCoefs[sortInd]
sortShifts = shifts[sortInd]

if madFlag == 1:
    mad = stats.median_absolute_deviation(sortCorrCoefs)
    mad = round(mad,2)
    print("Median absolute deviation of cross correlation coefficients: " + str(mad))
    ccThresh = mad

# make array to store waveform data and times
waveformDataAll = np.zeros((numTemp,traceLen))
waveformData = np.zeros((numTemp,traceLen))
detTimesAll = []
detTimes = []

count = 0

for i in range(numTemp):

    try:

        # get trace from obspy stream
        event = waveforms[sortInd[i]]
        trace = event.data

        # get datetimes
        detTimesAll.append(event.stats.starttime.datetime)

        # flip polarity if necessary
        if sortCorrCoefs[i] < 0:
            trace = trace * -1

        if sortShifts[i] > 0:
            alignedTrace = np.append(np.zeros(abs(int(sortShifts[i]))),trace)
            alignedTrace = alignedTrace[:traceLen]
            waveformDataAll[i,:] = alignedTrace/np.max(abs(alignedTrace))

        else:
            alignedTrace = trace[abs(int(sortShifts[i])):]
            waveformDataAll[i,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))

        # separate list for events exceeding threshold
        if abs(sortCorrCoefs[i]) > ccThresh:

            # get datetimes
            detTimes.append(event.stats.starttime.datetime)

            if sortShifts[i] > 0:
                waveformData[i,:] = alignedTrace/np.max(abs(alignedTrace))
            else:
                waveformData[i,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))

            count += 1

        print("Aligned " + str(round(i/len(waveforms)*100)) + "% of events")
    except:
        pass

# make plot of all events
fig,ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,gridspec_kw={'width_ratios':[1,4]})
ax[0].hist(abs(sortCorrCoefs),50,orientation='horizontal')
ax[0].invert_xaxis()
corrCoefs[masterInd] = 0
ax[0].set(ylim = [min(abs(sortCorrCoefs)),max(abs(sortCorrCoefs))])
ax[0].set(ylabel = "Cross correlation coefficient")
ax[0].axhline(y=ccThresh,color='red',linestyle='dashed')
ax[1].imshow(waveformDataAll, aspect = 'auto',extent=[0,300,numTemp,0])
ax[1].set(xlabel = "Time (s)")
ax[1].axhline(y=count,color='red',linestyle='dashed')
plt.title("All detections (" + str(freq[0]) + "-" + str(freq[1]) + " Hz)")
fig.tight_layout()
plt.show()
#plt.savefig(path + 'waveformPlot.png')

# make simple histogram of all times
startTime = waveforms[0].stats.starttime.datetime
endTime = waveforms[-1].stats.starttime.datetime
numDays = (endTime-startTime).days+1
plotDates = date2num(detTimesAll)
plt.hist(plotDates,numDays)
ax = plt.gca()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.title("Histogram of all detections (" + str(freq[0]) + "-" + str(freq[1]) + " Hz)")
plt.xlabel("Date")
plt.ylabel("Detection count")
plt.show()

# make plot of events exceeding xcorr threshold
fig,ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,gridspec_kw={'width_ratios':[1,4]})
ax[0].hist(abs(sortCorrCoefs[:count]),50,orientation='horizontal')
ax[0].invert_xaxis()
corrCoefs[masterInd] = 0
ax[0].set(ylim = [min(abs(sortCorrCoefs[:count])),max(abs(sortCorrCoefs[:count]))])
ax[0].set(ylabel = "Cross correlation coefficient")
ax[1].imshow(waveformData[:count,:], aspect = 'auto',extent=[0,300,count,0])
ax[1].set(xlabel = "Time (s)")
plt.title("Detections with CC > " + str(ccThresh) + " (" + str(freq[0]) + "-" + str(freq[1]) + " Hz)")
fig.tight_layout()
plt.show()

# make simple histogram of times exceeding xcorr threshold
plotDates = date2num(detTimes)
plt.hist(plotDates,numDays)
ax = plt.gca()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.title("Histogram of detections with CC > " + str(ccThresh) + " (" + str(freq[0]) + "-" + str(freq[1]) + " Hz)")
plt.xlabel("Date")
plt.ylabel("Detection count")
plt.show()

# close hdf5 file
output.close()
