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
inPath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
outPath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"

fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_shape"
norm_component = 0
numClusters = 10
numEventsPerClust = 1000
type = "short"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
print("Loading and normalizing input data...")

# change path variables
if norm_component:
    inPath = inPath + type + "_normalized_3D_clustering/" + method + "/"
else:
    inPath = inPath + type + "_3D_clustering/" + method + "/"
outPath = outPath + type + "_modified_3D_kshape/"

waves = np.empty((0,(snipLen*fs+1)*3),'float64')
for n in range(numClusters):
    # load waves
    waveFile = h5py.File(inPath + str(numClusters) + "/aligned_cluster" + str(n) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    cluster_waves = np.array(list(waveFile['waveforms']))
    waveFile.close()

    # load correlation results
    corrFile = h5py.File(inPath + str(numClusters) + "/centroid" + str(n) + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    corrCoefs = np.array(list(corrFile["corrCoefs"]))
    shifts = np.array(list(corrFile["shifts"]))
    corrFile.close()

    # sort and get best waves into test matrix
    sort_idx = np.array(np.argsort(abs(corrCoefs))[::-1])
    sort_waves = cluster_waves[sort_idx,:]
    waves = np.vstack((waves,sort_waves[:numEventsPerClust,:]))

waveFile = h5py.File(outPath + str(numClusters) + "/" + type + "_test_" + str(numEventsPerClust*numClusters) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
waveFile.create_dataset("waveforms",data=waves)
waveFile.close()
