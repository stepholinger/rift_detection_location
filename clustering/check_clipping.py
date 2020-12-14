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

# set paramters
numCluster = 30
cluster = 18
type = "short"

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
figPath = templatePath + "cluster_clipping/" + str(numCluster) + "/" + str(cluster) + "/"
fs = 2
chans = ["HHZ","HHN","HHE"]

# set length of wave snippets in seconds
#snipLen = 360001
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of HHZ-only waveforms
waveform_file = h5py.File(templatePath + type + "_waveform_matrix_" + chans[0] + "_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
events = np.array(list(waveform_file['waveforms']))
waveform_file.close()

# load matrix of 3D waveforms
for c in range(len(chans)):
    waveform_file = h5py.File(templatePath + type + "_waveform_matrix_" + chans[c] + "_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
    waveform_matrix = list(waveform_file['waveforms'])
    if c == 0:
        events3D = np.empty((len(waveform_matrix),0),'float64')
    events3D = np.hstack((events3D,waveform_matrix))

    # close h5 file
    waveform_file.close()

# load clustering results
clustFile = h5py.File(templatePath + type + "_3D_clustering/" + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
pred = np.array(list(clustFile["cluster_index"]))
centroids = list(clustFile["centroids"])
clustFile.close()

# get cluster events
clusterEvents = events[pred == cluster]
clusterEvents3D = events3D[pred == cluster]

# just plot HHZ and see
for w in range(len(clusterEvents)):
    plt.plot(clusterEvents[w]/max(abs(clusterEvents[w])),'k',alpha=0.0025)
plt.plot(centroids[cluster][1:snipLen*fs+1]/max(abs(centroids[cluster][1:snipLen*fs+1])))
plt.title("Normalized Centroid and Normalized HHZ Traces")
plt.savefig(figPath + "fig1.png")
plt.close()

# make normalized traces
clusterEvents3D_norm = clusterEvents3D
for i in range(len(clusterEvents3D)):
    clusterEvents3D_norm[i,:] = clusterEvents3D[i,:]/max(np.abs(clusterEvents3D[i,:]))

# scale mean around zero (before and after normalization respectively)
clusterEvents3D = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(clusterEvents3D)
clusterEvents3D_norm = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(clusterEvents3D_norm)

# plot normalized -> TSSMV events
for w in range(len(clusterEvents3D)):
    plt.plot(clusterEvents3D_norm[w],'k',alpha=0.0025)
plt.plot(centroids[cluster])
plt.title("Centroid and 3D Trace Normalized to 1 before TSSMV")
plt.savefig(figPath + "fig2.png")
plt.close()

# plot TSSMV events (nonnormalized amplitude)
for w in range(len(clusterEvents3D)):
    plt.plot(clusterEvents3D[w],'k',alpha=0.0025)
plt.title("3D Trace after TSSMV")
plt.savefig(figPath + "fig3.png")
plt.close()

# plot TSSMV -> normalized events
for w in range(len(clusterEvents3D)):
    plt.plot(clusterEvents3D[w]/max(abs(clusterEvents3D[w])),'k',alpha=0.0025)
plt.plot(centroids[cluster]/max(abs(centroids[cluster])))
plt.title("Centroid and 3D Trace Normalized to 1 after TSSMV")
plt.savefig(figPath + "fig4.png")
plt.close()
