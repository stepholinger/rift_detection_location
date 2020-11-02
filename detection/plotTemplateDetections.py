import obspy
import copy

path = "/media/Data/Data/PIG/MSEED/noIR/"
outPath = "/home/setholinger/Documents/Projects/PIG/detectionPlots/templateMatch/PIG2/HHZ/"
stat = "PIG2"
chan = "HHZ"

# define filter parameters
freqLow = [0.05,0.1]
freqHigh = [1, 10]

# define an empty list
detections = []

# open file and read the content in a list
with open('/home/setholinger/Documents/Projects/PIG/detections/templateMatch/template1Detections.txt', 'r') as f:
    for line in f:
        currentPlace = line[:-1]
        detections.append(currentPlace)

# make bogus date for first iteration
lastDate = obspy.UTCDateTime(0)

# loop through all detections
for d in detections:

    # get date string
    date = d.split('T')[0]

    # get UTCDateTime for start of detection window
    year = date.split("-")[0]
    month = date.split("-")[1]
    day = date.split("-")[2]
    hour = d.split('T')[1].split(":")[0]
    min = d.split('T')[1].split(":")[1]
    sec = d.split('T')[1].split(":")[2].split(".")[0]
    detectionStart = obspy.UTCDateTime(int(year),int(month),int(day),int(hour),int(min),int(sec))

    # if same as last detection's date, don't read data again
    if date != lastDate:

        # read in data
        stRaw = obspy.read(path + stat + "/" + chan + "/" + date + "." + stat + "." + chan + ".noIR.MSEED")

        # basic preprocessing
        stRaw.detrend("demean")
        stRaw.detrend("linear")
        stRaw.taper(max_percentage=0.01, max_length=10.)

        # copy for other bands
        stLow = stRaw.copy()
        stHigh = stRaw.copy()

        # filter the data
        stLow.filter("bandpass",freqmin=freqLow[0],freqmax=freqLow[1])
        stHigh.filter("bandpass",freqmin=freqHigh[0],freqmax=freqHigh[1])

        # combine into one stream for plotting
        stRaw += stLow
        stRaw += stHigh

    # make plot of the detection
    stRaw.plot(starttime=detectionStart,endtime=detectionStart+300,outfile = outPath + date + "_" + str(detectionStart.hour) + ":" + str(detectionStart.minute) + ":00.png",equal_scale=False,method='full')

    # copy the date string for comparison with next detection
    lastDate = copy.deepcopy(date)
