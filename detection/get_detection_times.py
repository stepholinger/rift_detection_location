import matplotlib.pyplot as plt
import numpy as np
import obspy
import h5py
from datetime import datetime
from matplotlib.dates import date2num

# set number of clusters
numCluster = 10

# define path to data and templates
path = "/media/Data/Data/PIG/MSEED/noIR/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"

# load matrix of waveforms
prefiltFreq = [0.05,1]
waves = obspy.read(templatePath + 'short_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')

detTimes = []
for w in range(len(waves)):
    detTimes.append(waves[w].stats.starttime.ns/1e9)

# save times
outFile = h5py.File(templatePath + "short_clustering/" + str(numCluster) + "/detection_times.h5","w")
outFile.create_dataset("times",data=detTimes)
outFile.close()
