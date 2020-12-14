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
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_shape"
norm_component = 0
skipClustering = 0
numClusters = range(3,4)
type = "short"

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/" + method + "/"
else:
    templatePath = templatePath + type + "_3D_clustering/" + method + "/"

for n_clusters in numClusters:

    # load silhouette values
    silFile = h5py.File(templatePath + str(n_clusters)  +  "/" + str(n_clusters) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    sample_silhouette_values = np.array(list(silFile["scores"]))
    silFile.close()

    # calculate stats
    silMean = np.mean(sample_silhouette_values)
    silMed = np.median(sample_silhouette_values)
    silMax = np.max(sample_silhouette_values)
    silMin = np.min(sample_silhouette_values)
    silQ1 = np.quantile(sample_silhouette_values,.25)
    silQ3 = np.quantile(sample_silhouette_values,.75)

    statsFile = h5py.File(templatePath + str(n_clusters)  +  "/" + str(n_clusters) + "_cluster_silhoutte_stats.h5","w")
    statsFile.create_dataset("mean",data=silMean)
    statsFile.create_dataset("median",data=silMed)
    statsFile.create_dataset("max",data=silMax)
    statsFile.create_dataset("min",data=silMin)
    statsFile.create_dataset("Q1",data=silQ1)
    statsFile.create_dataset("Q3",data=silQ3)
    statsFile.close()
