import numpy as np
import obspy

def readEvent(path,stat,chan,tempLims,freq):

    # extract strings for date and make date string
    tempDate = tempLims[0].isoformat().split("T")[0]

    # read in data for template
    event = obspy.read(path + stat + "/" + chan + "/" + tempDate + "." + stat + "." + chan + ".noIR.MSEED")

    # basic preprocessing and filtering
    event.detrend("demean")
    event.detrend("linear")
    event.taper(max_percentage=0.01, max_length=10.)
    event.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # trim the data to the time ranges of template for each band
    event.trim(starttime=tempLims[0],endtime=tempLims[1])

    return event
