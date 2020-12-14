import tslearn
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
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
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set paramters
method = "k_shape"
norm_component = 0
skipClustering = 0
numClusters = range(2,41)
type = "short"

# change path variables
if norm_component:
    templatePath = templatePath + type + "_normalized_3D_clustering/" + method + "/"
else:
    templatePath = templatePath + type + "_3D_clustering/" + method + "/"

# first version, using inertia values from clustering
sil_mean_vect = np.zeros((len(numClusters),1))
sil_median_vect = np.zeros((len(numClusters),1))
sil_max_vect = np.zeros((len(numClusters),1))
sil_min_vect = np.zeros((len(numClusters),1))
sil_Q1_vect = np.zeros((len(numClusters),1))
sil_Q3_vect = np.zeros((len(numClusters),1))

c = 0
errLim = 0
for f in numClusters:
    try:
        statFile = h5py.File(templatePath + str(f) + "/" + str(f) + "_cluster_silhoutte_stats.h5","r")
        sil_mean_vect[c] = statFile["mean"][()]
        sil_median_vect[c] = statFile["median"][()]
        sil_max_vect[c] = statFile["max"][()]
        sil_min_vect[c] = statFile["min"][()]
        sil_Q1_vect[c] = statFile["Q1"][()]
        sil_Q3_vect[c] = statFile["Q3"][()]
        statFile.close()
        print("Ran " + str(f) + " clusters")
        #print(sil_median_vect[c])
    except:
        print("Error on " + str(f) + " clusters")
        errLim = c
    c = c+1

if errLim == len(numClusters)-1:
    axLim = errLim-1
else:
    axLim = len(numClusters)-1

# plot 1- show full range of score distribution
# plot all the lines
plt.plot(numClusters[:axLim],sil_mean_vect[:axLim],color='red')
plt.plot(numClusters[:axLim],sil_median_vect[:axLim],color='blue')
plt.plot(numClusters[:axLim],sil_Q1_vect[:axLim],color='black',linestyle="--")
plt.plot(numClusters[:axLim],sil_Q3_vect[:axLim],color='black',linestyle="--")
plt.plot(numClusters[:axLim],sil_max_vect[:axLim],color='gray',linestyle="--")
plt.plot(numClusters[:axLim],sil_min_vect[:axLim],color='gray',linestyle="--")

# label them
plt.text(numClusters[axLim], sil_mean_vect[axLim]-0.008, "Mean",color='red')
plt.text(numClusters[axLim], sil_median_vect[axLim]-0.008, "Median",color='blue')
plt.text(numClusters[axLim], sil_Q1_vect[axLim]-0.008, "Q1",color='black')
plt.text(numClusters[axLim], sil_Q3_vect[axLim]-0.008, "Q3",color='black')
plt.text(numClusters[axLim], sil_max_vect[axLim]-0.008, "Max",color='gray')
plt.text(numClusters[axLim], sil_min_vect[axLim]-0.008, "Min",color='gray')

# get max value and its index
mean_max_ind = np.argmax(sil_mean_vect)
mean_max = sil_mean_vect[mean_max_ind]
median_max_ind = np.argmax(sil_median_vect)
median_max = sil_median_vect[median_max_ind]

# put a point at the max value for each curve
plt.scatter(numClusters[mean_max_ind],mean_max,s=100,marker='*',color='red',edgecolors='black',zorder=100)
plt.scatter(numClusters[median_max_ind],median_max,s=100,marker='*',color='blue',edgecolors='black',zorder=100)
plt.text(numClusters[mean_max_ind],mean_max+0.05,"N=" + str(numClusters[mean_max_ind]) + ": " + str(np.round(mean_max[0],3)),color='red',zorder=100)
plt.text(numClusters[median_max_ind],median_max+0.05,"N=" + str(numClusters[median_max_ind]) + ": " + str(np.round(median_max[0],3)),color='blue',zorder=100)

plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.gca().set_ylim([-0.1,0.5])
plt.gca().set_xlim([0,numClusters[-1]+5])

if norm_component:
    plt.title(("Silhouette Curve for Normalized K-Shape"), fontsize=14, fontweight='bold')
else:
    plt.title(("Silhouette Curve for K-Shape"), fontsize=14, fontweight='bold')

#plt.show()
plt.savefig(templatePath + "/silhouette_l_curve.png")

# second scaled version of plot
plt.gca().set_ylim([-0.1,0.1])

plt.savefig(templatePath + "/silhouette_l_curve_zoomed.png")


plt.close()
