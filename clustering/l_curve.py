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

path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/short_3D_clustering/"

prefiltFreq = [0.05,1]

numClusters = range(2,31)

# first version, using inertia values from clustering
inertia_vect = np.zeros((len(numClusters),1))
d_inertia_dt = np.zeros((len(numClusters),1))

for f in range(len(numClusters)):
    try:
        clustFile = h5py.File(templatePath + str(numClusters[f]) + "/" + str(numClusters[f]) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
        inertia = clustFile['inertia']
        inertia_vect[f] = inertia[()]
        if f > 0:
            d_inertia_dt[f-1] = inertia_vect[f]-inertia_vect[f-1]
        clustFile.close()
    except:
        pass
plt.plot(numClusters,inertia_vect)
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.gca().set_ylim([np.min(inertia_vect[np.nonzero(inertia_vect)]),0.7])
#plt.show()
plt.savefig(templatePath + "inertia_l_curve.png")
plt.close()
#plt.plot(numClusters,d_inertia_dt)
#plt.show()
#
# # second version, using actual norm calculated from centroid and cluster members
# clustNormSums = np.zeros((len(numClusters),1))
#
# # weird check garbage
# all_weighted_avgs = np.zeros((len(numClusters),1))
#
# for c in range(len(numClusters)):
#
#     # get centroids for current cluster number
#     clustFile = h5py.File(templatePath + str(numClusters[c]) + "/" + str(numClusters[c]) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
#     pred = np.array(list(clustFile["cluster_index"]))
#     centroids = list(clustFile["centroids"])
#     clustFile.close()
#
#     # make array for storing sum of norms for each cluster in the c-cluster run
#     normSum = 0
#
#     # weird check garbage
#     avg_centroid_amp = 0
#
#     for cn in range(numClusters[c]):
#
#         # get waveforms already aligned with cluster centroid
#         waveFile = h5py.File(templatePath + str(numClusters[c]) + "/aligned_cluster" + str(cn) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
#         alignedWaves = np.array(list(waveFile["waveforms"]))
#         waveFile.close()
#
#         # get median cluster A_max
#         a_max = np.amax(abs(alignedWaves),1)
#         medAmp = np.median(a_max)
#
#         # get current cluster centroid
#         centroid = centroids[cn].ravel()
#
#         # get weighted average component
#         amp = max(abs(centroid))
#         weighted_centroid_amp = amp * len(alignedWaves)/137000
#
#         # normalization
#         #centroid = centroid/np.max(abs(centroid))
#         #centroid = centroid/np.max(abs(centroid))*medAmp
#         alignedWaves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(alignedWaves)
#         alignedWaves = alignedWaves[:,:,0]
#         a_max = np.amax(abs(alignedWaves),1)
#         alignedWaves = alignedWaves/a_max[:,None]
#
#         # calculate L2 norm- find distance from centroid (elementwise difference), square each value, sum, then square root
#         # something like: sqrt(sum(np.square(abs(alignedWaves[w] - centroid))))
#         res = abs(alignedWaves-centroid)
#         squared_res = np.square(res)
#         sum_squared_res = np.sum(squared_res,1)
#         norms = np.sqrt(sum_squared_res)
#         normSum += np.sum(norms)
#
#         # weighted centroid average amplitude
#         avg_centroid_amp += weighted_centroid_amp
#
#         print("Finished calculating norms for run " + str(numClusters[c]) + " cluster " + str(cn))
#
#     # fill vector
#     clustNormSums[c] = normSum
#
#     # weird check garbage
#     all_weighted_avgs[c] = avg_centroid_amp
#
# plt.plot(numClusters,all_weighted_avgs)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Weighted Average of Centroid A_max")
# plt.show()
#
# plt.plot(numClusters,clustNormSums)
# plt.xlabel("Number of clusters")
# plt.ylabel("Sum of L2 norms")
# plt.savefig(templatePath + "residual_l_curve.png")
