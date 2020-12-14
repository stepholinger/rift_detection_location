import tslearn
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
import time
import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import h5py

# read in waveforms
# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_means"
norm_component = 1
type = "short"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
print("Loading and normalizing input data...")
for c in range(len(chans)):
    waveform_file = h5py.File(templatePath + type + "_waveform_matrix_" + chans[c] + "_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
    waveform_matrix = list(waveform_file['waveforms'])
    if c == 0:
        waves = np.empty((len(waveform_matrix),0),'float64')

    # normalize each component
    if norm_component:
        waveform_matrix = np.divide(waveform_matrix,np.amax(np.abs(waveform_matrix),axis=1,keepdims=True))

    waves = np.hstack((waves,waveform_matrix))

    # close h5 file
    waveform_file.close()

# calculate average waveform
avg_wave = np.mean(waves,axis=0)

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/" + method + "/"
else:
    templatePath = templatePath + type + "_3D_clustering/" + method + "/"

# load centroid into dummy obspy event as master waveform for cross correlation
masterEvent = obspy.read(templatePath + "dummy_2Hz.h5")
masterEvent[0].data = avg_wave

# load in an obspy dummy event
event = obspy.read(templatePath + "dummy_2Hz.h5")

# iterate through all waves in the current cluster
eventsAligned = np.zeros((len(waves),(snipLen*fs+1)*3))
corrCoefs = np.zeros((len(waves)))
shifts = np.zeros((len(waves)))
for w in range(len(waves)):

    # load dummy obspy event (for cross correlation) and fill with current event data
    event[0].data = waves[w]
    trace = waves[w]

    # correlate centroid with event
    corr = correlate(masterEvent[0],event[0],snipLen*fs,normalize='naive',demean=False,method='auto')
    shift, corrCoef = xcorr_max(corr)
    corrCoefs[w] = corrCoef
    shifts[w] = shift

    if shift > 0:

        alignedTrace = np.append(np.zeros(abs(int(shift))),trace)
        alignedTrace = alignedTrace[:(snipLen*fs+1)*3]
        eventsAligned[w,:len(alignedTrace)] = alignedTrace

    else:

        alignedTrace = trace[abs(int(shift)):]
        eventsAligned[w,:len(alignedTrace)] = alignedTrace

    print("Aligned " + str(round(w/len(waves)*100)) + "% of events")

# save cross correlation results
corrFile = h5py.File(templatePath + "/all_wave_avg_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
corrFile.create_dataset("corrCoefs",data=corrCoefs)
corrFile.create_dataset("shifts",data=shifts)
corrFile.close()

# save aligned cluster waveforms
waveFile = h5py.File(templatePath + "/aligned_all_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
waveFile.create_dataset("waveforms",data=eventsAligned)
waveFile.close()

# get xcorr quality order (this is only for plotting so we dont screw up the event indices)
sortIdx = np.array(np.argsort(abs(corrCoefs))[::-1])

norm_waves = np.divide(eventsAligned,np.amax(np.abs(eventsAligned),axis=1,keepdims=True))
plt.imshow(norm_waves[sortIdx],vmin=-0.1,vmax=0.1,aspect = 'auto',extent=[0,snipLen*3,len(eventsAligned),0],cmap='seismic')
plt.gca().set_xticks([0,snipLen/2,snipLen,snipLen*3/2,snipLen*2,snipLen*5/2,snipLen*3])
xPos = [snipLen,snipLen*2]
for xc in xPos:
    plt.gca().axvline(x=xc,color='k',linestyle='--')
plt.gca().set_xticklabels(['0','250\n'+chans[0],'500  0   ','250\n'+chans[1],'500  0   ','250\n'+chans[2],'500'])

plt.xlabel("Time (seconds)")
plt.ylabel("Event Number")
plt.tight_layout(h_pad=1.0)
plt.savefig(templatePath + "all_aligned_waves.png")
