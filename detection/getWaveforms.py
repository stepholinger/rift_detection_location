import obspy
import copy
import obspyh5

# set paths
dataPath = "/media/Data/Data/PIG/MSEED/noIR/"
path = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/template1/"

# set station parameters
stat = "PIG2"
chan = "HHZ"
fs = 100

# set length of detection to extract in seconds
detLenSec = 300
detLenIdx = detLenSec*fs
buffer = 100

# make empty array for storing detections timestrings
detections = []

# open file and read the content in a list
with open(path + 'template1Detections.txt', 'r') as f:
    for line in f:
        currentPlace = line[:-1]
        detections.append(currentPlace)

# make bogus date for first iteration
lastDate = obspy.UTCDateTime(0)

# loop through all detections
for d in range(len(detections)):

    # get date string
    date = detections[d].split('T')[0]

    # get UTCDateTime for start of detection window
    year = date.split("-")[0]
    month = date.split("-")[1]
    day = date.split("-")[2]
    hour = detections[d].split('T')[1].split(":")[0]
    min = detections[d].split('T')[1].split(":")[1]
    sec = detections[d].split('T')[1].split(":")[2].split(".")[0]
    detectionStart = obspy.UTCDateTime(int(year),int(month),int(day),int(hour),int(min),int(sec))

    # if detection is on same day as last detection, don't read data again
    if date != lastDate:

        # output that the previous day is finished (only start after first day finishes)
        if lastDate != obspy.UTCDateTime(0):
            print("Finished extracting waveforms from " + lastDate)

        # read in data
        stRaw = obspy.read(dataPath + stat + "/" + chan + "/" + date + "." + stat + "." + chan + ".noIR.MSEED")

    # just get the data surrounding each detection
    det = stRaw.slice(starttime=detectionStart-buffer,endtime=detectionStart+detLenSec+buffer)

    # write the stream to hdf5
    det.write(path + 'waveforms.h5','H5',mode='a')

    # copy the date string for comparison with next detection
    lastDate = copy.deepcopy(date)
