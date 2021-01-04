import tslearn
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
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
from matplotlib import cm

# read in waveforms
# define path to data and templates
templatePath = "/n/home01/setholinger/clustering/kshape/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_shape"
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

# load clustering results
try:
    clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    cluster_labels = np.array(list(clustFile["cluster_index"]))
    centers = list(clustFile["centroids"])
    clustFile.close()

except:
    modelFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_model_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    model = modelFile['data']
    cluster_labels = np.array(list(model['model_params']['labels_']))
    centers = np.array(list(model['model_params']['cluster_centers_']))
    inertia = list(model['model_params']['inertia_'])
    modelFile.close()

    # save to regular h5 for convenience
    clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
    clustFile.create_dataset("cluster_index",data=pred)
    clustFile.create_dataset("centroids",data=centroids)
    clustFile.create_dataset("inertia",data=inertia)
    clustFile.close()


# Compute the silhouette scores for each sample and save results
print("Starting silhouette value calculation for " + str(numCluster) + " clusters...")
sample_silhouette_values = silhouette_samples(waves, cluster_labels,metric='correlation')
silFile = h5py.File(templatePath + str(numCluster)  +  "/" + str(numCluster) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
silFile.create_dataset("scores",data=sample_silhouette_values)
silFile.close()
