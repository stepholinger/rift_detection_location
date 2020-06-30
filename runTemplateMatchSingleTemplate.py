import obspy
from obspy.signal.cross_correlation import correlation_detector
import glob
import numpy as np
import multiTemplateMatch

# this code produce the ~7000 detections in the folder 'template1'

# define path to data
path = "/media/Data/Data/PIG/MSEED/noIR/"

# define station and channel parameters
stat = "PIG2"
chans = ["HHZ","HHE","HHN"]

# define template match parameters for first template
threshLow = 0.6
threshHigh = 0.2
tolerance = 60

# define time limits for template in each frequency band
tempLimsLow = [obspy.UTCDateTime(2012,5,9,18,1,0),obspy.UTCDateTime(2012,5,9,19,30,0)]
tempLimsHigh = [obspy.UTCDateTime(2012,5,9,18,1,0),obspy.UTCDateTime(2012,5,9,18,8,0)]

# define filter parameters for first template
freqLow = [0.001,0.1]
freqHigh = [1, 10]

# run the template matching algorithm for first template
detections = multiTemplateMatch.multiTemplateMatch(path,stat,chans,tempLimsLow,freqLow,threshLow,tempLimsHigh,freqHigh,threshHigh,tolerance)

# save results
with open('/home/setholinger/Documents/Projects/PIG/detections/templateMatch/template2/template2Detections.txt', 'w') as f:
    for item in detections:
        f.write("%s\n" % item)
