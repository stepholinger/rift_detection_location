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
maxClusters = 10

# set length of wave snippets in seconds
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# make plot
fig,ax = plt.subplots(nrows=maxClusters,ncols=maxClusters,sharex=False,sharey=False,figsize=(20,20))
[axi.set_axis_off() for axi in ax.ravel()]

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
for n in range(1,maxClusters):

    # load clustering results
    numCluster = n+1

    # give output
    print("Working on " + str(numCluster) + " cluster results")

    for c in range(numCluster):

        # read in aligned waveforms for current cluster
        waveFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/aligned_cluster" + str(c) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
        clusterEventsAligned = np.array(list(waveFile["waveforms"]))
        waveFile.close()

        # plot all waves and mean waveform (amplitudes preserved)
        t = np.linspace(0,snipLen,snipLen*fs+1)
        ax[c,n].plot(t,clusterEventsAligned[:5000,:].T,'k',alpha=0.005)
        cluster_mean_wave = np.mean(clusterEventsAligned,axis=0)
        ax[c,n].axis('on')
        ax[c,n].plot(t,cluster_mean_wave)
        ax[c,n].set_ylim([-4*max(abs(cluster_mean_wave)),4*max(abs(cluster_mean_wave))])
        ax[c,n].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
        if c<n:
            ax[c,n].set_xticklabels([])

    # add individual subplot titles
    ax[0,n].title.set_text('N = ' + str(n+1))

# add whole-plot axis labels
fig.suptitle("Cluster Centroids and 5000 Waveforms", fontsize=24)
fig.text(0.5, 0.01, 'Time (s)', ha='center', fontsize=16)
fig.text(0.05, 0.5, 'Velocity (m/s)', va='center', rotation='vertical', fontsize=16)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig(templatePath + "clustering/" + "all_clusters" + ".png")
