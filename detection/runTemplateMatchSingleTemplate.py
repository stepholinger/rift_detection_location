import obspy
from obspy.signal.cross_correlation import correlation_detector
import glob
import numpy as np
import multiTemplateMatch_bckp

# this code produce the ~7000 detections in the folder 'template1'

# define path to data
path = "/media/Data/Data/PIG/MSEED/noIR/"

# define station and channel parameters
stat = "PIG2"
chans = ["HHZ","HHE","HHN"]

# define time limits for first template in each frequency band
tempLimsLow = [obspy.UTCDateTime(2012,4,2,15,18,00),obspy.UTCDateTime(2012,4,2,15,23,00)]
tempLimsHigh = [obspy.UTCDateTime(2012,4,2,15,18,00),obspy.UTCDateTime(2012,4,2,15,21,00)]

# define filter parameters for first template
freqLow = [0.05,0.1]
freqHigh = [1, 10]

# define template match parameters for first template
threshLow = 0.6
threshHigh = 0.2
tolerance = 60

# run the template matching algorithm for first template
detections = multiTemplateMatch_bckp.multiTemplateMatch(path,stat,chans,tempLimsLow,freqLow,threshLow,tempLimsHigh,freqHigh,threshHigh,tolerance)

# save results
with open('/home/setholinger/Documents/Projects/PIG/detections/templateMatch/test.txt', 'w') as f:
    for item in detections:
        f.write("%s\n" % item)
