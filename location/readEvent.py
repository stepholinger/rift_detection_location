import numpy as np
import obspy

def readEvent(path,stat,chan,tempLims,freq):

    # extract year string
    tempYear = tempLims[0].year
    tempYearStr = str(tempYear)

    # extract month string and pad with zeros if necessary
    tempMonth = tempLims[0].month
    if tempMonth < 10:
        tempMonthStr = "0" + str(tempMonth)
    else:
        tempMonthStr = str(tempMonth)

    # extract day string and pad with zeros if necessary
    tempDay = tempLims[0].day
    if tempDay < 10:
        tempDayStr = "0" + str(tempDay)
    else:
        tempDayStr = str(tempDay)

    # construct date string
    tempDate = tempYearStr + "-" + tempMonthStr + "-" + tempDayStr

    # read in data for template
    event = obspy.read(path + stat + "/" + chan + "/" + tempDate + "." + stat + "." + chan + ".noIR.MSEED")

    # basic preprocessing
    event.detrend("demean")
    event.detrend("linear")
    event.taper(max_percentage=0.01, max_length=10.)

    # filter the data to each band
    event.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # trim the data to the time ranges of template for each band
    event.trim(starttime=tempLims[0],endtime=tempLims[1])

    # plot templates (for testing)
    #event.plot()

    return event
