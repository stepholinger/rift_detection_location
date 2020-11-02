import matplotlib.pyplot as plt
import time
import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import h5py
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter

# read in waveforms
# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
fs = 2
numCluster = 10
snipLen = 500

# load matrix of waveforms
prefiltFreq = [0.05,1]
waves = obspy.read(templatePath + 'short_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')

# load clustering results
outFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
pred = list(outFile["cluster_index"])
centroids = list(outFile["centroids"])
outFile.close()

# for each cluster, cross correlate, align and plot each event in the cluster in reference to the centroid
medAmpArray = []
for c in range(numCluster):

    # get events in current cluster
    clusterEvents = []
    for i in range(len(waves)):
        if pred[i] == c:
            clusterEvents.append(waves[i])

    # make empty array for storage
    maxAmps = np.zeros((len(clusterEvents)))
    cluster_spectra = []
    detTimes = []

    # iterate through all waves in the current cluster
    for w in range(len(clusterEvents)):
        detTimes.append(clusterEvents[w].stats.starttime.datetime)
        trace = clusterEvents[w].data
        maxAmps[w] = np.max(abs(trace))

        # calculate Fourier amplitude spectra for current event
        event_spectra = abs(np.fft.fft(trace))
        cluster_spectra.append(event_spectra)

    # get median of waveform max amplitudes
    medAmp = np.median(maxAmps)
    medAmpArray.append(medAmp)

    # calculate Fourier amplitude spectra for centroid
    cluster_centroid = centroids[c]/max(centroids[c])*medAmp
    centroid_spectra = np.array(abs(np.fft.fft(cluster_centroid)))

    # make simple histogram of max amplitudes
    hist, bins = np.histogram(maxAmps, bins=25)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(maxAmps, bins=logbins)
    ax = plt.gca()
    plt.title("Histogram of $A_{max}$ for events in cluster " + str(c))
    plt.xlabel("Max amplitude (m/s)")
    plt.ylabel("Detection count")
    plt.xlim([min(maxAmps),max(maxAmps)])
    plt.xscale('log')
    plt.yscale('log')
    ax.axvline(x=medAmp,color='red',linestyle='dashed')
    plt.text(3/2*medAmp,np.log10(max(hist)),"Median $A_{max}$: " + str(np.round(medAmp,decimals=10)) + " m/s")
    plt.savefig(templatePath + "clustering/" + str(numCluster) + "/" + "cluster_" + str(c) + "_amplitude_distribution.png")
    plt.close()

    # make simple histogram of times
    startTime = waves[0].stats.starttime.datetime
    endTime = waves[-1].stats.starttime.datetime
    numDays = (endTime-startTime).days+1
    plotDates = date2num(detTimes)
    plt.hist(plotDates,numDays)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.title("Timeseries of events in cluster " + str(c))
    plt.xlabel("Date")
    plt.ylabel("Detection count")
    plt.gcf().autofmt_xdate()
    plt.savefig(templatePath + "clustering/" + str(numCluster) + "/" + "cluster_" + str(c) + "_time_distribution.png")
    plt.close()

    # make spectra figure
    freq = np.fft.fftfreq(snipLen*fs+1,1/fs)
    for n in range(len(clusterEvents)):
        try:
            plt.plot(freq[0:int(snipLen*fs/2)],cluster_spectra[n][0:int(snipLen*fs/2)],'k',alpha=0.01)
        except:
            pass
    plt.plot(freq[0:int(snipLen*fs/2)],centroid_spectra[0:int(snipLen*fs/2)],'r')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (m/s)')
    plt.title("Cluster " + str(c) + " spectra (centroid scaled by cluster median amplitude")
    plt.savefig(templatePath + "clustering/" + str(numCluster) + "/" + "cluster_" + str(c) + "_spectra.png")
    plt.close()

# save median amplitudes as hdf5 file
medFile = h5py.File(templatePath + "clustering/" + str(numCluster) + "/" + str(numCluster) + "_cluster_median_amplitudes.h5","w")
medFile.create_dataset("median_amplitudes",data=medAmpArray)
medFile.close()
