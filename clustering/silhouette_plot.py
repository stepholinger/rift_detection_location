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

    # read in waveform file and make labels that correspond (since order is not preserved when we read the separately
    # clustered aligned waves)
    cluster_labels = np.empty((0,1),'float64')
    outFile = h5py.File(templatePath + str(n_clusters) +  "/" + str(n_clusters) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    pred = np.array(list(outFile["cluster_index"]))
    outFile.close()

    for n in range(n_clusters):
        num_waves = sum(pred==n)
        ind = np.ones((num_waves,1))*n
        cluster_labels = np.vstack((cluster_labels,ind))

    # load silhouette values
    silFile = h5py.File(templatePath + str(n_clusters)  +  "/" + str(n_clusters) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    sample_silhouette_values = np.array(list(silFile["scores"]))
    silFile.close()

    # load silhouette stats
    statFile = h5py.File(templatePath + str(n_clusters) + "/" + str(n_clusters) + "_cluster_silhoutte_stats.h5","r")
    sil_mean = statFile["mean"][()]
    sil_median = statFile["median"][()]
    sil_max = statFile["max"][()]
    sil_min = statFile["min"][()]
    sil_Q1 = statFile["Q1"][()]
    sil_Q3 = statFile["Q3"][()]
    statFile.close()

    # get max and min silhouette values
    minVal = sil_min
    maxVal = sil_max

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels.flatten() == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(0.01, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("Silhouette Values")
    ax1.set_ylabel("Event Number")

    # The vertical line for average silhouette score of all the values
    if sil_mean > sil_median:
        shift_mean = 0.003
        shift_median = -0.012
    else:
        shift_mean = -0.012
        shift_median = 0.003
    ax1.axvline(x = sil_mean, color="red", linestyle="--")
    ax1.axvline(x = sil_median, color="blue", linestyle="--")
    ax1.text(sil_mean+shift_mean, len(cluster_labels)*0.5, "Mean Score: " + str(np.round(sil_mean,3)),color='red',rotation='vertical')
    ax1.text(sil_median+shift_median, len(cluster_labels)*0.5, "Median Score: " + str(np.round(sil_median,3)),color='blue',rotation='vertical')

    ax1.axvline(x = sil_Q1, color="black", linestyle="--")
    ax1.axvline(x = sil_Q3, color="black", linestyle="--")
    ax1.text(sil_Q1-0.012, len(cluster_labels)*0.5, "Q1: " + str(np.round(sil_Q1,3)),color='black',rotation='vertical')
    ax1.text(sil_Q3+0.003, len(cluster_labels)*0.5, "Q3: " + str(np.round(sil_Q3,3)),color='black',rotation='vertical')
    ax1.set_xlim([-0.5,0.5])
    plt.xticks([-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])

    if norm_component:
        plt.suptitle(("Silhouette Analysis for Normalized K-Shape with n = %d" % n_clusters), fontsize=14, fontweight='bold')
    else:
        plt.suptitle(("Silhouette Analysis for K-Shape with n = %d" % n_clusters), fontsize=14, fontweight='bold')

    plt.savefig(templatePath + str(n_clusters) + "/" + str(n_clusters) + "_cluster_silhoutte" + ".png")
    plt.close()
