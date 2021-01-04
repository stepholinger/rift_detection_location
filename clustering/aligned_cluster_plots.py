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
method = "modified_k_shape"
norm_component = 0
numClusters = range(20,21)
type = "short"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
print("Loading and normalizing input data...")

# read in pre-aligned 3-component traces
if method == "k_means":
        waveform_file = h5py.File(templatePath + type + "_3D_clustering/" + method + "/aligned_all_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
        input_waves = np.array(list(waveform_file['waveforms']))
        waveform_file.close()
else:
    # read in trace for each component and concatenate
    for c in range(len(chans)):
        waveform_file = h5py.File(templatePath + type + "_waveform_matrix_" + chans[c] + "_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
        waveform_matrix = list(waveform_file['waveforms'])
        if c == 0:
            input_waves = np.empty((len(waveform_matrix),0),'float64')

        input_waves = np.hstack((input_waves,waveform_matrix))

        # close h5 file
        waveform_file.close()

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/" + method + "/"
else:
    templatePath = templatePath + type + "_3D_clustering/" + method + "/"

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
for n in numClusters:

    print("Making plots for n = " + str(n) + "...")

    for c in range(n):
    #    try:

        # load clustering results
        try:
            clustFile = h5py.File(templatePath + str(n) +  "/" + str(n) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
            pred = np.array(list(clustFile["cluster_index"]))
            centroids = list(clustFile["centroids"])
            clustFile.close()
        except:
            modelFile = h5py.File(templatePath + str(n) +  "/" + str(n) + "_cluster_model_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
            model = modelFile['data']
            pred = np.array(list(model['model_params']['labels_']))
            centroids = np.array(list(model['model_params']['cluster_centers_']))
            modelFile.close()

        # read in correlation results for the current cluster
        corrFile = h5py.File(templatePath + str(n) + "/centroid" + str(c) + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
        corrCoefs = np.array(list(corrFile["corrCoefs"]))
        shifts = np.array(list(corrFile["shifts"]))
        corrFile.close()

        # make empty array for storage
        clusterEvents = input_waves[pred == c]
        clusterEventsAligned = np.zeros((len(input_waves[pred == c]),(snipLen*fs+1)*3))

        # iterate through all waves in the current cluster
        for w in range(len(clusterEvents)):

            # get cross correlation results
            shift = shifts[w]
            corrCoef = corrCoefs[w]

            # get current event
            trace = clusterEvents[w].ravel()

            # flip polarity if necessary
            if corrCoef < 0:
                trace = trace * -1

            if shift > 0:
                alignedTrace = np.append(np.zeros(abs(int(shift))),trace)
                alignedTrace = alignedTrace[:(snipLen*fs+1)*3]
                clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

            else:
                alignedTrace = trace[abs(int(shift)):]
                clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

        # save aligned cluster waveforms
        #waveFile = h5py.File(templatePath + str(n) + "/aligned_cluster" + str(c) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
        #waveFile.create_dataset("waveforms",data=clusterEventsAligned)
        #waveFile.close()

        # make plot version 1; shows difference in amplitudes on different components
        fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[1,2]})
        sortIdx = np.array(np.argsort(abs(corrCoefs))[::-1])
        t = np.linspace(0,snipLen*3,(snipLen*fs+1)*3)

        # plot all waves and mean waveform (amplitudes preserved)
        for w in range(len(clusterEventsAligned)):
            ax[0].plot(t,clusterEventsAligned[w],'k',alpha=0.05)
        cluster_mean_wave = np.nanmean(clusterEventsAligned[sortIdx,:],axis=0)
        ax[0].plot(t,cluster_mean_wave*5,linewidth=1)
        ax[0].set_ylim([-10*np.nanmax(abs(cluster_mean_wave)),10*np.nanmax(abs(cluster_mean_wave))])
        xPos = [snipLen,snipLen*2]
        for xc in xPos:
            ax[0].axvline(x=xc,color='k',linestyle='--')
        ax[0].title.set_text('Centroid and Cluster Waveforms (Cluster ' + str(c) + ')')

        ax[1].imshow(np.divide(clusterEventsAligned[sortIdx,:],np.amax(np.abs(clusterEventsAligned[sortIdx,:]),axis=1,keepdims=True)),vmin=-0.25,vmax=0.25,aspect = 'auto',extent=[0,snipLen*3,len(clusterEvents),0],cmap='seismic')
        ax[1].set_xticks([0,snipLen/2,snipLen,snipLen*3/2,snipLen*2,snipLen*5/2,snipLen*3])
        xPos = [snipLen,snipLen*2]
        for xc in xPos:
            ax[1].axvline(x=xc,color='k',linestyle='--')
        ax[1].set_xticklabels(['0','250\n'+chans[0],'500  0   ','250\n'+chans[1],'500  0   ','250\n'+chans[2],'500'])

        plt.xlabel("Time (seconds)")
        plt.ylabel("Event Number")
        plt.tight_layout(h_pad=1.0)
        #plt.savefig(templatePath + str(n) + "/" + str(n)+ "_cluster_clust" + str(c) + ".png",dpi=400,transparent=True)
        plt.savefig(templatePath + str(n) + "/" + str(n)+ "_cluster_clust" + str(c) + ".png",dpi=400)
        plt.close()
        #except:
        #    pass
