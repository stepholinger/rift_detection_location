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

type = "short"
method = "modified_k_shape"
norm_component = 0
numClusters = range(20,21)

fs = 2
snipLen = 500

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

# load detection times
outFile = h5py.File(templatePath + "detection_times.h5","r")
detTimes = list(outFile["times"])
outFile.close()

prefiltFreq = [0.05,1]
#waves = obspy.read(templatePath + 'short_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')

# load gps velocities
outFile = h5py.File("gps_velocity.h5","r")
gps_time = list(outFile["time"])[0]
gps_vel = list(outFile["velocity"])[0]
outFile.close()

for n_clusters in numClusters:

    # load clustering results
    outFile = h5py.File(templatePath + str(n_clusters) + "/" + str(n_clusters) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
    pred = list(outFile["cluster_index"])
    centroids = list(outFile["centroids"])
    outFile.close()

    for c in range(n_clusters):

        print("Plotting timeseries for cluster " + str(c) + " (" + str(n_clusters) + " clusters)")
        try:
            # get events in current cluster
            clusterTimes = []
            for i in range(len(detTimes)):
                if pred[i] == c:
                    clusterTimes.append(datetime.utcfromtimestamp(detTimes[i]))

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

            #plt.show()
            plt.savefig(templatePath + str(n_clusters) + "/" + "cluster_" + str(c) + "_timing_and_gps.png")
            plt.close()
        except:
            pass
