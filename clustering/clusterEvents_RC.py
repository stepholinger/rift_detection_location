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
fs = 2

# set paramters
numCluster = 3

# set length of wave snippets in seconds
snipLen = 500

# read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
prefiltFreq = [0.05,1]

waveform_file = h5py.File("short_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5",'r')
waveform_matrix = list(waveform_file['waveforms'])
input_waves = waveform_matrix.copy()

# close h5 file
waveform_file.close()

# get a subset of the waveforms for testing
print("Algorithm will run on " + str(len(input_waves)) + " waveforms")

# scale mean around zero
input_waves = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(input_waves)

# normalize everything
print("Normalizing input data...")
for i in range(len(input_waves)):
    input_waves[i,:] = input_waves[i,:]/max(input_waves[i,:])

# run clustering
print("Clustering...")
ks = KShape(n_clusters=numCluster, n_init=1, random_state=0).fit(input_waves)
pred = ks.fit_predict(input_waves)

# save result
outFile = h5py.File(str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","w")
outFile.create_dataset("cluster_index",data=pred)
outFile.create_dataset("centroids",data=ks.cluster_centers_)
outFile.create_dataset("inertia",data=ks.inertia_)
outFile.close()

# make plot of results
#plt.figure()
#for i in range(numCluster):
#    plt.subplot(10, 1, 1 + i)
#    for n in input_waves[pred == i]:
#        plt.plot(n.ravel(), "k-", alpha=.2)
#    plt.plot(ks.cluster_centers_[i].ravel(), "r-")
#    plt.ylim(-4, 4)
#    plt.title("Cluster %d" % (i + 1))

#plt.show()

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
for c in range(numCluster):

    # load centroid into dummy obspy event as master waveform for cross correlation
    masterEvent = obspy.read(templatePath + "dummy_2Hz.h5")
    masterEvent[0].data = ks.cluster_centers_[c].ravel()

    # make empty array for storage
    clusterEvents = input_waves[pred == c]
    clusterEventsAligned = np.zeros((len(input_waves[pred == c]),snipLen*fs+1))
    corrCoefs = np.zeros((len(input_waves[pred == c])))

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

        # flip polarity if necessary
        if corrCoef < 0:
            trace = trace * -1

        if shift > 0:
            alignedTrace = np.append(np.zeros(abs(int(shift))),trace)
            alignedTrace = alignedTrace[:snipLen*fs]
            clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))

        else:
            alignedTrace = trace[abs(int(shift)):]
            clusterEventsAligned[w,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))
        print("Aligned " + str(round(w/len(clusterEvents)*100)) + "% of events for cluster " + str(c+1))

    # sort by cross correlation coefficient
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[1,4]})
    sortIdx = np.array(np.argsort(abs(corrCoefs))[::-1])
    ax[0].plot(ks.cluster_centers_[c].ravel())
    ax[1].imshow(clusterEventsAligned[sortIdx,:],aspect = 'auto')
    #plt.show()
    plt.savefig(str(numCluster) + "_cluster_clust" + str(c) + ".png")
