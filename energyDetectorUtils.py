import obspy
import obspyh5
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
from scipy.signal import find_peaks

def removeDoubleCounting(det,tolerance):
    # remove double counting from final list
    for d in range(len(det)-1):
        if det[d+1] - det[d] < tolerance:
            det[d] = []

    # remove empty elements
    det = list(filter(None,det))
    return det

def getFiles(chan,path,stat):
    if chan == 'all':
        files = glob.glob(path + stat + "/" + "HHZ" + "/*", recursive=True)
        files = [f.replace("Z","*") for f in files]
    else:
        files = glob.glob(path + stat + "/" + chan + "/*", recursive=True)
    files.sort()
    return files

def getEnergyPeaks(st,prominence,tolerance,fs):
    # square trace to get kinetic energy
    energy = np.square(np.array(st.data,dtype='float64'))

    # normalize amplitudes (helps with peak finding)
    energy = energy/np.max(energy)

    # find maxima in both bands
    peaks,_ = find_peaks(energy,prominence=prominence,distance=fs*tolerance)
    return peaks,energy

def getTriggers(st,energyLow,peaksLow,peakHigh,tolerance,buffer,fs,detShortChan,detLongChan,multiplier):

    # check if biggest low freq peak of day
    if energyLow[peaksLow[0]]/np.max(energyLow) == 1:
        try:
            # check if at least two low freq peaks are within tolerance*multiplier*fs seconds of the high freq peak
            if peaksLow[0] - peakHigh < tolerance*multiplier*fs and peaksLow[1] - peakHigh < tolerance*multiplier*fs:

                # append to list of detections for this channel
                detLongChan.append(st.stats.starttime + peakHigh/fs - buffer*multiplier)

            # if not, check if normal detection criteria is met
            else:
                if peaksLow[0] - peakHigh < tolerance*fs:

                    # append to list of detections for this channel
                    detShortChan.append(st.stats.starttime + peakHigh/fs - buffer*multiplier)
        except:
            pass

    # if not, check if normal detection criteria is met
    else:
        if peaksLow[0] - peakHigh < tolerance*fs:

            # append to list of detections for this channel
            detShortChan.append(st.stats.starttime + peakHigh/fs - buffer)
    return detShortChan,detLongChan

def saveWaveforms(detections,st,buffer,outPath,type):
    for d in detections:
        det = st.slice(starttime=d,endtime=d+buffer[0]+buffer[1])
        det.write(outPath + type + '_waveforms.h5','H5',mode='a')

def saveDetections(detections,outPath,type):
    detectionTimestamps = []
    for d in detections:
        detectionTimestamps.append(d.timestamp)
    outFile = h5py.File(outPath + type + "_detections.h5","w")
    outFile.create_dataset("detections",data=detectionTimestamps)
    outFile.close()

def testPlot(energyHigh,peaksHigh,energyLow,peaksLow):
    plt.plot(energyHigh)
    plt.plot(energyLow)
    plt.plot(peaksHigh,energyHigh[peaksHigh],"^")
    plt.plot(peaksLow,energyLow[peaksLow],"v")
    plt.show()
