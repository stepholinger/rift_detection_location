import obspy
import obspyh5
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py

mode = "h5"

if mode == "read":

    # set path
    path = '/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/'

    # set file parameters
    fs = 100

    # set trace length and buffer in seconds
    traceLenSec = 300
    bufferLenSec = 100
    traceLen = traceLenSec * fs
    bufferLen = bufferLenSec * fs

    # load waveforms
    waveforms = obspy.read(path + 'waveforms.h5')

    # filter waveforms
    freq = [0.01,0.1]
    waveforms.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # read hdf5 file of results from correlation
    output = h5py.File(path + 'correlations.h5','r')

    # extract data from hdf5 file
    corrCoefs = list(output['corrCoefs'])
    shifts = list(output['shifts'])

    # load data into array
    corrCoefs = np.array(corrCoefs)
    shifts = np.array(shifts)

    # sort waveforms and shifts by correlation coeficient
    sortInd = np.argsort(abs(corrCoefs))[::-1]
    sortCorrCoefs = corrCoefs[sortInd]
    sortShifts = shifts[sortInd]

    # make array to store waveform data
    waveformData = np.zeros((len(waveforms),traceLen))

    for i in range(len(sortInd)):

        # get trace from obspy stream
        trace = waveforms[sortInd[i]].data

        # flip polarity if necessary
        if sortCorrCoefs[i] < 0:
            trace = trace * -1

        # adjust buffer length with event shift
        bufferShift = bufferLen - int(sortShifts[i])

        if sortShifts[i] > 0 and abs(sortShifts[i]) > bufferLen:
            alignedTrace = np.append(np.zeros(abs(bufferShift)),trace)
            alignedTrace = alignedTrace[:traceLen]
            waveformData[i,:] = alignedTrace/np.max(abs(alignedTrace))

        else:
            alignedTrace = trace[bufferShift:bufferShift + traceLen]
            waveformData[i,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))

        print("Aligned " + str(round(i/len(waveforms)*100)) + "% of events")

    # make empty figure
    fig,ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,gridspec_kw={'width_ratios':[1,4]})

    # plot histogram on left side of figure
    ax[0].hist(abs(corrCoefs),100,orientation='horizontal')
    ax[0].invert_xaxis()
    ax[0].set(ylim = [min(abs(corrCoefs)),max(abs(corrCoefs))])

    # make plot of all waveforms
    ax[1].imshow(waveformData, aspect = 'auto')

    # correct wonky formatting
    fig.tight_layout()

    # display and save plot
    plt.show()
    #plt.savefig(path + 'waveformPlot.png')

    # close hdf5 file
    output.close()

if mode == "h5":

    # define path to data and templates
    path = "/media/Data/Data/PIG/MSEED/noIR/"
    templatePath = "/home/setholinger/Documents/Projects/PIG/detections/energy/run2/"
    fs = 100

    # read in h5 file of single channel templates- we will use the start and end times to make 3-component templates
    waveforms = obspy.read(templatePath + 'long_waveforms.h5')
    numTemp = len(waveforms)
    traceLen = waveforms[0].stats.npts
    bufferLen = 120*5*fs
    masterInd = 52

    #filter waveforms
    #freq = [0.001,0.005]
    #waveforms.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # read hdf5 file of results from correlation
    output = h5py.File(templatePath + 'long_correlations.h5','r')

    # extract data from hdf5 file
    corrCoefs = list(output['corrCoefs'])
    shifts = list(output['shifts'])

    # load data into array
    corrCoefs = np.array(corrCoefs)
    shifts = np.array(shifts)

    # sort waveforms and shifts by correlation coeficient
    sortInd = np.argsort(abs(corrCoefs))[::-1]
    sortCorrCoefs = corrCoefs[sortInd]
    sortShifts = shifts[sortInd]

    # make array to store waveform data
    waveformData = np.zeros((numTemp,traceLen))

    for i in range(numTemp):

        try:

            # get trace from obspy stream
            trace = waveforms[sortInd[i]].data
            # flip polarity if necessary
            if sortCorrCoefs[i] < 0:
                trace = trace * -1

            # adjust buffer length with event shift
            bufferShift = bufferLen - int(sortShifts[i])

            if sortShifts[i] > 0 and abs(sortShifts[i]) > bufferLen:
                alignedTrace = np.append(np.zeros(abs(bufferShift)),trace)
                alignedTrace = alignedTrace[:traceLen]
                waveformData[i,:] = alignedTrace/np.max(abs(alignedTrace))

            else:
                alignedTrace = trace[bufferShift:bufferShift + traceLen]
                waveformData[i,:len(alignedTrace)] = alignedTrace/np.max(abs(alignedTrace))
            print("Aligned " + str(round(i/len(waveforms)*100)) + "% of events")

        except:
            pass

    # make empty figure
    fig,ax = plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,gridspec_kw={'width_ratios':[1,4]})

    # plot histogram on left side of figure
    ax[0].hist(abs(corrCoefs),50,orientation='horizontal')
    ax[0].invert_xaxis()
    corrCoefs[masterInd] = 0
    ax[0].set(ylim = [min(abs(corrCoefs)),max(abs(corrCoefs))])

    # make plot of all waveforms
    ax[1].imshow(waveformData, aspect = 'auto')

    # add title
    plt.title("Long K.E. Detections (" + str(freq[0]) + "-" + str(freq[1]) + " Hz)")

    # correct wonky formatting
    fig.tight_layout()

    # display and save plot
    plt.show()
    #plt.savefig(path + 'waveformPlot.png')

    # close hdf5 file
    output.close()
