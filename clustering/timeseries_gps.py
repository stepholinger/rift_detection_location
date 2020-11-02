import matplotlib.pyplot as plt
import time
import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import h5py
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter
from datetime import datetime

# read in waveforms
# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 2
numCluster = 14
snipLen = 500

# load detection times
outFile = h5py.File(templatePath + "short_3D_clustering/detection_times.h5","r")
detTimes = list(outFile["times"])
outFile.close()

prefiltFreq = [0.05,1]
#waves = obspy.read(templatePath + 'short_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')

# load clustering results
outFile = h5py.File(templatePath + "short_3D_clustering/" + str(numCluster) + "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
pred = list(outFile["cluster_index"])
centroids = list(outFile["centroids"])
outFile.close()

# load gps velocities
outFile = h5py.File("gps_velocity.h5","r")
gps_time = list(outFile["time"])[0]
gps_vel = list(outFile["velocity"])[0]
outFile.close()

# choose cluster
c = 3

# get events in current cluster
clusterTimes = []
for i in range(len(detTimes)):
    if pred[i] == c:
        clusterTimes.append(datetime.utcfromtimestamp(detTimes[i]))

# iterate through all waves in the current cluster
#detTimes = []
#for w in range(len(clusterEvents)):
#    detTimes.append(clusterEvents[w].stats.starttime.datetime)

# make simple histogram of times
startTime = clusterTimes[0]
endTime = clusterTimes[-1]
numDays = (endTime-startTime).days+1
plt.hist(clusterTimes,numDays)
ax = plt.gca()
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.title("Event Timeseries and GPS Velocity (Cluster " + str(c) + ")")
plt.xlabel("Date")
plt.ylabel("Detection count")
plt.gcf().autofmt_xdate()

# plot gps data
ax2 = ax.twinx()
plotDates = []
for t in gps_time:
    plotDates.append(datetime.utcfromtimestamp(t))
ax2.plot(plotDates,gps_vel*86400*365, c = 'k')
ax2.set_ylabel("Velocity (m/year)")
ax2.set_ylim(3750,4025)

plt.show()
#plt.savefig(templatePath + "short_clustering/" + str(numCluster) + "/" + "cluster_" + str(c) + "_timing_and_gps.png")
#plt.close()
