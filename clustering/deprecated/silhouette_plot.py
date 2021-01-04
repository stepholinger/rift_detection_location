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
norm_component = 1
skipClustering = 0
numClusters = range(2,3)
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

    # load clustering results
    clustFile = h5py.File(templatePath + str(n_clusters) +  "/" + str(n_clusters) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    cluster_labels = np.array(list(clustFile["cluster_index"]))
    centers = list(clustFile["centroids"])
    clustFile.close()

    # load silhouette values
    silFile = h5py.File(templatePath + str(n_clusters)  +  "/" + str(n_clusters) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    sample_silhouette_values = np.array(list(silFile["scores"]))
    silFile.close()

    # get max and min silhouette values
    minVal = min(sample_silhouette_values)
    maxVal = max(sample_silhouette_values)

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(minVal/2, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("Silhouette Values")
    ax1.set_ylabel("Event Number")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
    #ax1.set_yticks([])  # Clear the yaxis labels / ticks
    #ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    ax1.set_xlim([minVal*2,maxVal*2])


    plt.suptitle(("Silhouette Analysis for Normalized K-Shape "
                  "with n = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.savefig(templatePath + str(n_clusters) + "/" + str(n_clusters) + "_cluster_silhoutte" + ".png")
    plt.close()
