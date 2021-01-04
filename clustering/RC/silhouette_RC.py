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
templatePath = "/n/holyscratch01/denolle_lab/solinger/clustering/kshape/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_shape"
norm_component = 1
skipClustering = 0
numCluster = 2
type = "short"

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/"
else:
    templatePath = templatePath + type + "_3D_clustering/"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
print("Loading and normalizing input data...")

# read in waveform file and make labels that correspond (since order is not preserved when we read the separately
# clustered aligned waves)
waves = np.empty((0,(snipLen*fs+1)*3),'float64')
cluster_labels = np.empty((0,1),'float64')
for n in range(numCluster):
    # save aligned cluster waveforms
    waveFile = h5py.File(templatePath + str(numCluster) + "/aligned_cluster" + str(n) + "scaled_input_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    cluster_waves = np.array(list(waveFile['waveforms']))
    waveFile.close()
    waves = np.vstack((waves,cluster_waves))
    ind = np.ones((len(cluster_waves),1))*n
    cluster_labels = np.vstack((cluster_labels,ind))

# Compute the silhouette scores for each sample and save results
print("Starting silhouette value calculation for " + str(numCluster) + " clusters...")
sample_silhouette_values = silhouette_samples(waves, cluster_labels,metric='correlation')
silFile = h5py.File(templatePath + str(numCluster)  +  "/" + str(numCluster) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
silFile.create_dataset("scores",data=sample_silhouette_values)
silFile.close()
