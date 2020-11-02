import matplotlib.pyplot as plt
import time
import numpy as np
import obspy
import h5py
from readEvent import readEvent
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max

# read in waveforms
# define path to data and templates
dataPath = "/media/Data/Data/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"

# set paramters
numCluster = 10
xcorr_thresh = 0.9

# set length of wave snippets in seconds
snipLen = 500
preBuff = 0

# set frequency bands for a few things
prefiltFreq = [0.05,1]
centroidFreq = [0.05,1]
plotFreq = [0.05,1]

# load full matrix of waveforms
print("Loading waveforms...")
waves = obspy.read(templatePath + 'short_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')

# get starttime for each event
event_times = []
for w in range(len(waves)):
    event_times.append(waves[w].stats.starttime)
print("Retrieved all event times")

# clear obspy waveform for memory
del waves

# load clustering results
cluster_results = h5py.File(templatePath + "clustering/" + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(centroidFreq[0]) + "-" + str(centroidFreq[1]) + "Hz.h5","r")
pred = np.array(list(cluster_results["cluster_index"]))
cluster_results.close()

# for each cluster, find max amplitude event that's well correlated with centroid
for c in range(numCluster):

    # give output
    print("Retrieving max amplitude event for cluster " + str(c))

    # read in correlation results for the current cluster
    corrFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/centroid" + str(c) + "_correlations_" + str(centroidFreq[0]) + "-" + str(centroidFreq[1]) + "Hz.h5","r")
    cluster_xcorr_coef = np.array(list(corrFile["corrCoefs"]))
    corrFile.close()

    # read in waveforms for current cluster
    waveFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/aligned_cluster" + str(c) + "_waveform_matrix_" + str(centroidFreq[0]) + "-" + str(centroidFreq[1]) + "Hz.h5","r")
    clusterEvents = np.array(list(waveFile["waveforms"]))
    waveFile.close()

    # get event times for events in current cluster
    clusterEvent_times = []
    for i in range(len(event_times)):
        if pred[i] == c:
            clusterEvent_times.append(event_times[i])
    numEvents_cluster = len(clusterEvent_times)

    # get max amplitude for each event
    maxAmps = np.amax(clusterEvents,axis=1)

    # replace max amp with 0 for events below xcorr_thresh
    for w in range(len(clusterEvent_times)):
        if abs(cluster_xcorr_coef[w]) < xcorr_thresh:
            maxAmps[w] = 0

    # get index of largest event above cross correlation threshold
    maxIdx = np.argmax(maxAmps)

    # get bounds around event
    starttime = clusterEvent_times[maxIdx]
    eventLims = [starttime - preBuff, starttime + snipLen]

    # make empty obspy stream
    events = obspy.read()
    events.clear()

    # read event on PIG and regional stations
    stats = ["PIG1","PIG2","PIG3","PIG4","PIG5","BEAR","THUR","DNTW","UPTW"]
    nets = ["PIG","PIG","PIG","PIG","PIG","YT","YT","YT","YT"]
    chans = ["HHZ","HHZ","HHZ","HHZ","HHZ","BHZ","BHZ","BHZ","BHZ"]
    for n in range(len(stats)):
        try:
            event = readEvent(dataPath + nets[n] + "/MSEED/noIR/",stats[n],chans[n],eventLims,plotFreq)
            events += event
        except:
            pass

    # make plot
    events.plot(outfile=templatePath + "clustering/" + str(numCluster) +  "/regionalPlots/clust" + str(c) + "_maxEvent_xcorr>" + str(xcorr_thresh) + "_" + str(plotFreq[0]) + "-" + str(plotFreq[1]) + "Hz.png",equal_scale=False)

    # save waveform
    event.write(templatePath + "clustering/" + str(numCluster) +  "/clust" + str(c) + "_representative_event_" + str(plotFreq[0]) + "-" + str(plotFreq[1]) + 'Hz.h5','H5',mode='a')
