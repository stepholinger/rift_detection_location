import tslearn
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

# NOTE: the aligned waves produced by this code are the ACTUAL DATA, not the preprocessed input for clustering.
# This means it's suitable for seismic analysis and plotting but NOT silhouettes!

# read in waveforms
# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_shape"
numCluster = 10
numEventsPerClust = 1000
norm_component = 0
type = "short"

# set number of components to tell K-shape- this determines if we get modified or base behavior
n_component = 3

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# change path variables
templatePath = templatePath + type + "_modified_3D_kshape/"

# load matrix of waveforms
print("Loading and normalizing input data...")

# read in trace for each component and concatenate
if norm_component:
    waveform_file = h5py.File(templatePath + str(numCluster) +  "/" + str(numEventsPerClust*numCluster) + "_events/" + type + "normalized_test_" + str(numEventsPerClust*numCluster) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
    waves = list(waveform_file['waveforms'])
else:
    waveform_file = h5py.File(templatePath + str(numCluster) +  "/" + str(numEventsPerClust*numCluster) + "_events/" + type + "_test_" + str(numEventsPerClust*numCluster) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
    waves = list(waveform_file['waveforms'])

# give output
print("Algorithm will run on " + str(len(waves)) + " waveforms")

# scale mean around zero
input_waves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(waves)

print("Clustering...")
ks = KShape(n_clusters=numCluster, n_init=1, random_state=0,n_component=n_component)
pred = ks.fit_predict(input_waves)

clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz_ncomp=" + str(n_component) + ".h5","w")
clustFile.create_dataset("cluster_index",data=pred)
clustFile.create_dataset("centroids",data=ks.cluster_centers_)
clustFile.create_dataset("inertia",data=ks.inertia_)
clustFile.close()

modelFile = templatePath + str(numCluster) +  "/" + str(numEventsPerClust*numCluster) + "_events/" + str(numCluster) + "_cluster_model_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz_ncomp=" + str(n_component) + ".h5"
ks.to_hdf5(modelFile)
