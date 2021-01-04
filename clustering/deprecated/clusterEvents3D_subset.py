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
chans = ["HHZ","HHN","HHE"]

# set paramters
norm_component = 1
skipClustering = 1

# choose how many clusters initially were used, which clusters to re-cluster, and how many clusters to form for each (from polarization results)
numClusterInit = 10
clustIndVect = [0,1,2,3,4,5,6,7,8,9]
numClusterVect = [3,2,2,4,2,2,2,2,5,2]
type = "short"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
for c in range(len(chans)):
    waveform_file = h5py.File(templatePath + type + "_waveform_matrix_" + chans[c] + "_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
    waveform_matrix = list(waveform_file['waveforms'])
    if c == 0:
        waves = np.empty((len(waveform_matrix),0),'float64')
    waves = np.hstack((waves,waveform_matrix))

    # close h5 file
    waveform_file.close()

# read initial clustering results
clustFile = h5py.File(templatePath + type + "_clustering/" + str(numClusterInit) +  "/" + str(numClusterInit) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
predInit = np.array(list(clustFile["cluster_index"]))
centroidsInit = list(clustFile["centroids"])
clustFile.close()

# change path variables
if norm_component:
    templateOutPath = templatePath + "subset_clustering/" + type + "_normalized_3D_clustering/" + str(numClusterInit) + "/"
else:
    templateOutPath + "subset_clustering/" + type + "_3D_clustering/" + str(numClusterInit) + "/"

# iterate through each cluster from the initial run
for n in range(numClusterInit):

    # get desired number of clusters for this run of clustering
    numCluster = numClusterVect[n]
    clusterInd = clustIndVect[n]

    # just get waveforms in current cluster
    clust_waves = waves[predInit == clusterInd]

    # give output
    print("Algorithm will run on " + str(len(clust_waves)) + " waveforms")

    # scale mean around zero
    input_waves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(clust_waves)

    # normalize everything
    print("Normalizing input data...")
    if norm_component:
        for i in range(len(input_waves)):
            for c in range(len(chans)):
                input_waves[i,(snipLen*fs+1)*c:(snipLen*fs+1)*(c+1)] = input_waves[i,(snipLen*fs+1)*c:(snipLen*fs+1)*(c+1)]/max(np.abs(input_waves[i,(snipLen*fs+1)*c:(snipLen*fs+1)*(c+1)]))
    else:
        for i in range(len(input_waves)):
            input_waves[i,:] = input_waves[i,:]/max(np.abs(input_waves[i,:]))

    # save result
    if skipClustering == 0:

        # run clustering
        print("Clustering...")
        ks = KShape(n_clusters=numCluster, n_init=1, random_state=0).fit(input_waves)
        pred = ks.fit_predict(input_waves)

        clustFile = h5py.File(templateOutPath + str(clusterInd) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
        clustFile.create_dataset("cluster_index",data=pred)
        clustFile.create_dataset("centroids",data=ks.cluster_centers_)
        clustFile.create_dataset("inertia",data=ks.inertia_)
        clustFile.close()

        # load some variables
        centroids = ks.cluster_centers_

    if skipClustering:
        clustFile = h5py.File(templateOutPath + str(clusterInd) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
        pred = np.array(list(clustFile["cluster_index"]))
        centroids = list(clustFile["centroids"])
        clustFile.close()

    # for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
    for c in range(numCluster):

        # load centroid into dummy obspy event as master waveform for cross correlation
        masterEvent = obspy.read(templatePath + "dummy_2Hz.h5")
        masterEvent[0].data = centroids[c].ravel()

        # make empty array for storage
        clusterEvents = clust_waves[pred == c]
        clusterEvents_norm = input_waves[pred == c]
        clusterEventsAligned = np.zeros((len(clust_waves[pred == c]),(snipLen*fs+1)*3))
        clusterEventsAligned_norm = np.zeros((len(input_waves[pred == c]),(snipLen*fs+1)*3))
        corrCoefs = np.zeros((len(input_waves[pred == c])))
        shifts = np.zeros((len(input_waves[pred == c])))

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
            shifts[w] = shift

            # flip polarity if necessary
            if corrCoef < 0:
                trace_norm = trace_norm * -1
                trace = trace * -1

            if shift > 0:
                alignedTrace_norm = np.append(np.zeros(abs(int(shift))),trace_norm)
                alignedTrace_norm = alignedTrace_norm[:(snipLen*fs+1)*3]
                clusterEventsAligned_norm[w,:len(alignedTrace_norm)] = alignedTrace_norm/np.max(abs(alignedTrace_norm))

                alignedTrace = np.append(np.zeros(abs(int(shift))),trace)
                alignedTrace = alignedTrace[:(snipLen*fs+1)*3]
                clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

            else:
                alignedTrace_norm = trace_norm[abs(int(shift)):]
                clusterEventsAligned_norm[w,:len(alignedTrace_norm)] = alignedTrace_norm/np.max(abs(alignedTrace_norm))

                alignedTrace = trace[abs(int(shift)):]
                clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

            print("Aligned " + str(round(w/len(clusterEvents_norm)*100)) + "% of events for cluster " + str(c+1))

        # save cross correlation results
        corrFile = h5py.File(templateOutPath + str(clusterInd) + "/centroid" + str(c) + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
        corrFile.create_dataset("corrCoefs",data=corrCoefs)
        corrFile.create_dataset("shifts",data=shifts)
        corrFile.close()

        # save aligned cluster waveforms
        waveFile = h5py.File(templateOutPath + str(clusterInd) + "/aligned_cluster" + str(c) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
        waveFile.create_dataset("waveforms",data=clusterEventsAligned)
        waveFile.close()

        # make plot version 1; shows difference in amplitudes on different components
        fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[1,2]})
        sortIdx = np.array(np.argsort(abs(corrCoefs))[::-1])
        t = np.linspace(0,snipLen*3,(snipLen*fs+1)*3)

        # plot all waves and mean waveform (amplitudes preserved)
        if norm_component:
            for w in range(round(len(clusterEventsAligned_norm)/100)):
            #for w in range(len(clusterEventsAligned_norm)):
                ax[0].plot(t,clusterEventsAligned_norm[w],'k',alpha=0.01)
            cluster_mean_wave = np.mean(clusterEventsAligned_norm,axis=0)
            ax[0].plot(t,cluster_mean_wave)
        else:
            for w in range(len(clusterEventsAligned)):
                ax[0].plot(t,clusterEventsAligned[w],'k',alpha=0.01)
            cluster_mean_wave = np.mean(clusterEventsAligned[sortIdx,:],axis=0)
            ax[0].plot(t,cluster_mean_wave*5)
            ax[0].set_ylim([-10*max(abs(cluster_mean_wave)),10*max(abs(cluster_mean_wave))])
        xPos = [snipLen,snipLen*2]
        for xc in xPos:
            ax[0].axvline(x=xc,color='k',linestyle='--')
        ax[0].title.set_text('Centroid and Cluster Waveforms (Cluster ' + str(c) + ')')

        ax[1].imshow(clusterEventsAligned_norm[sortIdx,:],vmin=-0.25,vmax=0.25,aspect = 'auto',extent=[0,snipLen*3,len(clusterEvents_norm),0],cmap='seismic')
        ax[1].set_xticks([0,snipLen/2,snipLen,snipLen*3/2,snipLen*2,snipLen*5/2,snipLen*3])
        xPos = [snipLen,snipLen*2]
        for xc in xPos:
            ax[1].axvline(x=xc,color='k',linestyle='--')
        ax[1].set_xticklabels(['0','250\n'+chans[0],'500  0   ','250\n'+chans[1],'500  0   ','250\n'+chans[2],'500'])

        plt.xlabel("Time (seconds)")
        plt.ylabel("Event Number")
        plt.tight_layout(h_pad=1.0)
        plt.savefig(templateOutPath + str(clusterInd) + "/" + str(numCluster)+ "_cluster_clust" + str(c) + ".png")
