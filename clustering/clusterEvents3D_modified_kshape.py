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
method = "modified_k_shape"
norm_component = 0
numCluster = 20
type = "short"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
print("Loading and normalizing input data...")

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

# small subset for testing
#waves = waves[:1000,:]

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/" + method + "/"
else:
    templatePath = templatePath + type + "_3D_clustering/" + method + "/"

# give output
print("Algorithm will run on " + str(len(waves)) + " waveforms")

# scale mean around zero
input_waves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(waves)

print("Clustering...")
ks = KShape(n_clusters=numCluster, n_init=1, random_state=0,n_component=len(chans))
pred = ks.fit_predict(input_waves)

if norm_component:
    clustFile = h5py.File(templatePath + str(numCluster) +  "/normalized_" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
else:
    clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
clustFile.create_dataset("cluster_index",data=pred)
clustFile.create_dataset("centroids",data=ks.cluster_centers_)
clustFile.create_dataset("inertia",data=ks.inertia_)
clustFile.close()

if norm_component:
    modelFile = templatePath + str(numCluster) +  "/normalized_" + str(numCluster) + "_cluster_model_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5"
else:
    modelFile = templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_model_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5"
ks.to_hdf5(modelFile)
