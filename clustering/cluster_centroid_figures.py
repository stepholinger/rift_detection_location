import tslearn
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
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

# set paramters
numCluster = 10

# set length of wave snippets in seconds
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
waveform_file = h5py.File(templatePath +"short_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
waveform_matrix = list(waveform_file['waveforms'])
waves = np.array(waveform_matrix.copy())

# close h5 file
waveform_file.close()

# load median amplitude stats
medFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/" + str(numCluster) + "_cluster_median_amplitudes.h5","r")
medAmps = np.array(list(medFile["median_amplitudes"]))
medFile.close()

# scale mean around zero
input_waves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(waves)

# normalize everything
print("Normalizing input data...")
for i in range(len(input_waves)):
    input_waves[i,:] = input_waves[i,:]/max(input_waves[i,:])

# load clustering results
outFile = h5py.File(templatePath + "clustering/" + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
pred = np.array(list(outFile["cluster_index"]))
centroids = list(outFile["centroids"])
outFile.close()

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
for c in range(numCluster):

    # load centroid into dummy obspy event as master waveform for cross correlation
    masterEvent = obspy.read(templatePath + "dummy_2Hz.h5")
    masterEvent[0].data = centroids[c].ravel()

    # make empty array for storage
    clusterEvents = waves[pred == c]
    clusterEvents_norm = input_waves[pred == c]
    clusterEventsAligned = np.zeros((len(waves[pred == c]),snipLen*fs+1))
    clusterEventsAligned_norm = np.zeros((len(input_waves[pred == c]),snipLen*fs+1))
    corrCoefs = np.zeros((len(input_waves[pred == c])))

    # iterate through all waves in the current cluster
    for w in range(len(clusterEvents)):

        # get current event
        trace_norm = clusterEvents_norm[w].ravel()
        trace = clusterEvents[w].ravel()

        # load dummy obspy event (for cross correlation) and fill with current event data
        event_norm = obspy.read(templatePath + "dummy_2Hz.h5")
        event_norm[0].data = trace_norm

        # correlate centroid with event
        corr = correlate(masterEvent[0],event_norm[0],event_norm[0].stats.npts,normalize='naive',demean=False,method='auto')
        shift, corrCoef = xcorr_max(corr)
        corrCoefs[w] = corrCoef

        # flip polarity if necessary
        if corrCoef < 0:
            trace_norm = trace_norm * -1
            trace = trace * -1

        if shift > 0:
            alignedTrace_norm = np.append(np.zeros(abs(int(shift))),trace_norm)
            alignedTrace_norm = alignedTrace_norm[:snipLen*fs]
            clusterEventsAligned_norm[w,:len(alignedTrace_norm)] = alignedTrace_norm/np.max(abs(alignedTrace_norm))

            alignedTrace = np.append(np.zeros(abs(int(shift))),trace)
            alignedTrace = alignedTrace[:snipLen*fs]
            clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

        else:
            alignedTrace_norm = trace_norm[abs(int(shift)):]
            clusterEventsAligned_norm[w,:len(alignedTrace_norm)] = alignedTrace_norm/np.max(abs(alignedTrace_norm))

            alignedTrace = trace[abs(int(shift)):]
            clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

        print("Aligned " + str(round(w/len(clusterEvents_norm)*100)) + "% of events for cluster " + str(c+1))

    # make plots
    fig,ax = plt.subplots(nrows=3,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[1,3,8]})
    sortIdx = np.array(np.argsort(abs(corrCoefs))[::-1])
    t = np.linspace(0,snipLen,snipLen*fs+1)

    ax[0].plot(t,centroids[c].ravel()/np.max(abs(centroids[c].ravel()))*medAmps[c])
    ax[0].title.set_text('Centroid Waveform scaled by median $A_{max}$')
    ax[0].set_ylim([-1*medAmps[c],medAmps[c]])
    ax[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))

    # plot all waves and mean waveform (amplitudes preserved)
    for w in range(len(clusterEventsAligned)):
        ax[1].plot(t,clusterEventsAligned[w],'k',alpha=0.005)
    cluster_mean_wave = np.mean(clusterEventsAligned[sortIdx,:],axis=0)
    ax[1].plot(t,cluster_mean_wave)
    ax[1].set_ylim([-4*max(abs(cluster_mean_wave)),4*max(abs(cluster_mean_wave))])
    ax[1].title.set_text('Mean Waveform')
    ax[1].ticklabel_format(style='sci', axis='y',scilimits=(0,0))

    ax[2].imshow(clusterEventsAligned_norm[sortIdx,:],aspect = 'auto',extent=[0,500,len(clusterEvents_norm),0])
    ax[2].title.set_text('Cluster Waveforms')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Event Number")
    plt.tight_layout(h_pad=1.0)
    plt.savefig(templatePath + "clustering/" + str(numCluster) + "/" + str(numCluster)+ "_cluster_clust" + str(c) + ".png")
