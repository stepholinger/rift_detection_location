import obspy
import obspyh5
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py

# define path to data and templates
type = "short"
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/"
fs = 100

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
all_waveforms = obspy.read(templatePath + type + '_waveforms.h5')
prefiltFreq = [0.01,1]

#filter waveforms
freq = [0.05,0.1]
all_waveforms.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

# just get desired channel
chan = 'HHZ'
waveforms = []
for f in all_waveforms:
    if f.stats.channel == chan:
        waveforms.append(f)

numTemp = len(waveforms)
traceLen = waveforms[0].stats.npts
#masterInd = 11
masterInd = 3189


# set cross correlation coefficient threshold
ccThresh = 0.4

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

# make array to store waveform data
waveformData = np.zeros((numTemp,traceLen))

count = 0

for i in range(numTemp):

  try:

      if abs(sortCorrCoefs[i]) > ccThresh:

          # plot waveform (for testing/checking)
          #waveforms[sortInd[i]].plot()

          # get trace from obspy stream
          event = waveforms[sortInd[i]]
          trace = event.data

          # flip polarity if necessary
          if sortCorrCoefs[i] < 0:
              trace = trace * -1

          if sortShifts[i] > 0:
              alignedTrace = np.append(np.zeros(abs(int(sortShifts[i]))),trace)
              alignedTrace = alignedTrace[:traceLen]
              waveformData[i,:] = alignedTrace/np.max(abs(alignedTrace))

          else:
              alignedTrace = trace[abs(int(sortShifts[i])):]
              waveformData[i,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))

          print("Aligned " + str(round(i/len(waveforms)*100)) + "% of events")
          count += 1
  except:
      pass

# make empty figure
fig,ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,gridspec_kw={'width_ratios':[1,4]})

# plot histogram on left side of figure
ax[0].hist(abs(sortCorrCoefs[:count]),50,orientation='horizontal')
ax[0].invert_xaxis()
corrCoefs[masterInd] = 0
ax[0].set(ylim = [min(abs(sortCorrCoefs[:count])),max(abs(sortCorrCoefs[:count]))])

# make plot of all waveforms
ax[1].imshow(waveformData[:count,:], aspect = 'auto')

# add title
plt.title(type + " K.E. Detections with CC > " + str(ccThresh) + " (" + str(freq[0]) + "-" + str(freq[1]) + " Hz)")

# correct wonky formatting
fig.tight_layout()

# display and save plot
plt.show()
#plt.savefig(path + 'waveformPlot.png')

# close hdf5 file
output.close()
