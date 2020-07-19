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
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 100

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

numEvent = len(waveforms)
traceLen = waveforms[0].stats.npts

# read hdf5 file of results from correlation
output = h5py.File(templatePath + type + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')

# extract data from hdf5 file
shifts = list(output['shifts'])
corrCoefs = list(output['corrCoefs'])

# load data into array
shifts = np.array(shifts)
corrCoefs = np.array(corrCoefs)

# make waveform storage
waveformDataAll = np.zeros((numEvent,traceLen))

for i in range(numEvent):

    # get trace from obspy stream
    event = waveforms[i]
    trace = event.data

    # flip polarity if necessary
    if corrCoefs[i] < 0:
        trace = trace * -1

    if shifts[i] > 0:
        alignedTrace = np.append(np.zeros(abs(int(shifts[i]))),trace)
        alignedTrace = alignedTrace[:traceLen]
        waveformDataAll[i,:] = alignedTrace/np.max(abs(alignedTrace))

    else:
        alignedTrace = trace[abs(int(shifts[i])):]
        waveformDataAll[i,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))

    print("Stacked " + str(round(i/len(waveforms)*100)) + "% of events")

# actually make stack
stackArray = np.sum(waveformDataAll,0)/numEvent

# shove it into an obspy object and save it
stack = waveforms[0]
stack.data = stackArray
stack.plot()
stack.write(templatePath + 'stack_' + str(freq[0]) + "-" + str(freq[1]) + '.h5','H5',mode='a')
