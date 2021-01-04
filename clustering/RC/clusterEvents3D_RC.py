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
templatePath = "/n/home01/setholinger/clustering/kmeans/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_means"
norm_component = 1
skipClustering = 0
numCluster = 2
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
        if norm_component:
            waveform_file = h5py.File(templatePath + type + "_normalized_3D_clustering/aligned_all_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
            waves = np.array(list(waveform_file['waveforms']))
            waveform_file.close()
        else:
            waveform_file = h5py.File(templatePath + type + "_3D_clustering/aligned_all_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
            waves = np.array(list(waveform_file['waveforms']))
            waveform_file.close()
else:
    # read in trace for each component and concatenate
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

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/"
else:
    templatePath = templatePath + type + "_3D_clustering/"

# give output
print("Algorithm will run on " + str(len(waves)) + " waveforms")

# scale mean around zero
input_waves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(waves)

# run clustering or skip and load results if desired
if skipClustering:
    clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    pred = np.array(list(clustFile["cluster_index"]))
    centroids = list(clustFile["centroids"])
    clustFile.close()
else:
    print("Clustering...")
    ks = KShape(n_clusters=numCluster, n_init=1, random_state=0)
    pred = ks.fit_predict(input_waves)

    clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
    clustFile.create_dataset("cluster_index",data=pred)
    clustFile.create_dataset("centroids",data=ks.cluster_centers_)
    clustFile.create_dataset("inertia",data=ks.inertia_)
    clustFile.close()

    modelFile = templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_model_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5"
    ks.to_hdf5(modelFile)

    # load some variables
    centroids = ks.cluster_centers_

# try silhouette calculation
#sil = tslearn.clustering.silhouette_score(input_waves,pred,metric='dtw')
#print(sil)

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
for c in range(numCluster):

    # load centroid into dummy obspy event as master waveform for cross correlation
    masterEvent = obspy.read(templatePath + "dummy_2Hz.h5")
    masterEvent[0].data = centroids[c].ravel()

    # make empty array for storage
    clusterEvents = waves[pred == c]
    clusterEvents_norm = input_waves[pred == c]
    clusterEventsAligned = np.zeros((len(waves[pred == c]),(snipLen*fs+1)*3))
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

    print("Finished correlations for cluster " + str(c))

    # save cross correlation results
    corrFile = h5py.File(templatePath + str(numCluster) + "/centroid" + str(c) + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
    corrFile.create_dataset("corrCoefs",data=corrCoefs)
    corrFile.create_dataset("shifts",data=shifts)
    corrFile.close()
