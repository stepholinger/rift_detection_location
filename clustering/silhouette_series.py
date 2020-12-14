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
numClusters = range(21,40)
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

for c in numClusters:

        try:

            # load matrix of waveforms
            print("Loading input data for cluster for " + str(c) + " clusters...")
            # read in waveform file and make labels that correspond (since order is not preserved when we read the separately
            # clustered aligned waves)
            waves = np.empty((0,(snipLen*fs+1)*3),'float64')
            cluster_labels = np.empty((0,1),'float64')
            for n in range(c):
                waveFile = h5py.File(templatePath + str(c) + "/aligned_cluster" + str(n) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
                cluster_waves = np.array(list(waveFile['waveforms']))
                waveFile.close()
                waves = np.vstack((waves,cluster_waves))
                ind = np.ones((len(cluster_waves),1))*n
                cluster_labels = np.vstack((cluster_labels,ind))
            cluster_labels = np.array(cluster_labels.flatten())

            # test the method by making random indices
            #cluster_labels = np.random.randint(0,high=c,size=len(waves))
            #print(min(cluster_labels))
            #print(max(cluster_labels))

            # rename variable
            X = waves

            # Compute the silhouette scores for each sample and save results
            print("Starting silhouette value calculation for " + str(c) + " clusters...")
            sample_silhouette_values = silhouette_samples(X, cluster_labels,metric='correlation')
            silFile = h5py.File(templatePath + str(c)  +  "/" + str(c) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
            silFile.create_dataset("scores",data=sample_silhouette_values)
            silFile.close()

            # load silhouette values
            silFile = h5py.File(templatePath + str(c)  +  "/" + str(c) + "_cluster_silhoutte_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
            sample_silhouette_values = np.array(list(silFile["scores"]))
            silFile.close()

            # get max and min silhouette values
            minVal = min(sample_silhouette_values)
            maxVal = max(sample_silhouette_values)

            # Create a subplot with 1 row and 2 columns
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)

            y_lower = 10
            for i in range(c):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels.flatten() == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / c)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(maxVal/10, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_xlabel("Silhouette Values")
            ax1.set_ylabel("Event Number")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")

            if minVal < 0:
                ax1.set_xlim([minVal*2,maxVal*2])
            else:
                ax1.set_xlim([0,maxVal*2])


            plt.suptitle(("Silhouette Analysis for Normalized K-Shape "
                          "with n = %d" % c),
                         fontsize=14, fontweight='bold')

            plt.savefig(templatePath + str(c) + "/" + str(c) + "_cluster_silhoutte" + ".png")
            plt.close()

        except:
            print("Skipped " + str(c) + " clusters (probably wan't run)")
