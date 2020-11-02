import obspy
from obspy.signal.cross_correlation import correlate_template
from obspy.signal.cross_correlation import xcorr_max
import numpy as np
import h5py

# set path
path = '/home/setholinger/Documents/Projects/PIG/detections/energy/run3/'

# load waveforms
#waveforms = obspy.read(path + 'waveforms.h5')
waveforms = obspy.read('/home/setholinger/Documents/Projects/PIG/detections/energy/run3/long_waveforms.h5')

# set filter parameters and filter waveforms
freq = [0.01,1]
waveforms.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

print(len(waveforms))

np.sort(waveforms)
for i in waveforms:
    print(i)

# read hdf5 file of results from correlation
det = h5py.File(path + 'long_detections.h5','r')

# extract data from hdf5 file
detTimestamp = list(det['detections'])
print(detTimestamp)
# close file
det.close
