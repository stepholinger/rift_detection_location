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

# set paramters
maxClusters = 10

# set length of wave snippets in seconds
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

# load matrix of waveforms
waveform_file = h5py.File(templatePath +"short_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
waveform_matrix = list(waveform_file['waveforms'])
waves = np.array(waveform_matrix.copy())

# close h5 file
waveform_file.close()

# make plot
fig,ax = plt.subplots(nrows=maxClusters,ncols=maxClusters,sharex=False,sharey=False,figsize=(20,20))
[axi.set_axis_off() for axi in ax.ravel()]

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
for n in range(1,maxClusters):

    # load clustering results
    numCluster = n+1
    outFile = h5py.File(templatePath + "clustering/" + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    pred = np.array(list(outFile["cluster_index"]))
    centroids = list(outFile["centroids"])
    outFile.close()

    # give output
    print("Working on " + str(numCluster) + " cluster results")

    for c in range(numCluster):

        # load centroid into dummy obspy event as master waveform for cross correlation
        masterEvent = obspy.read(templatePath + "dummy_2Hz.h5")
        masterEvent[0].data = centroids[c].ravel()

        # make empty array for storage
        clusterEvents = waves[pred == c]
        clusterEventsAligned = np.zeros((len(waves[pred == c]),snipLen*fs+1))
        corrCoefs = np.zeros((len(waves[pred == c])))
        shifts = np.zeros((len(waves[pred == c])))

        # iterate through all waves in the current cluster
        for w in range(len(clusterEvents)):

            # get current event
            trace = clusterEvents[w].ravel()

            # load dummy obspy event (for cross correlation) and fill with current event data
            event = obspy.read(templatePath + "dummy_2Hz.h5")
            event[0].data = trace

            # correlate centroid with event
            corr = correlate(masterEvent[0],event[0],event[0].stats.npts,normalize='naive',demean=False,method='auto')
            shift, corrCoef = xcorr_max(corr)
            corrCoefs[w] = corrCoef
            shifts[w] = shift

            # flip polarity if necessary
            if corrCoef < 0:
                trace = trace * -1

            if shift > 0:

                alignedTrace = np.append(np.zeros(abs(int(shift))),trace)
                alignedTrace = alignedTrace[:snipLen*fs]
                clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

            else:

                alignedTrace = trace[abs(int(shift)):]
                clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace

            # give output at % completion increase
            if round(w/len(clusterEvents)*100) > round((w-1)/len(clusterEvents)*100):
                print("Aligned " + str(round(w/len(clusterEvents)*100)) + "% of events for cluster " + str(c))

        # sort by cross correlation coefficient
        sortIdx = np.array(np.argsort(abs(corrCoefs))[::-1])

        # save cross correlation results
        outFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/centroid" + str(c) + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
        outFile.create_dataset("corrCoefs",data=corrCoefs)
        outFile.create_dataset("shifts",data=shifts)
        outFile.close()

        # save aligned cluster waveforms
        outFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/aligned_cluster" + str(c) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
        outFile.create_dataset("waveforms",data=clusterEventsAligned)
        outFile.close()
