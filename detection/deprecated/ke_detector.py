import pathlib
import obspy
import obspyh5
import numpy as np
import h5py
import glob
from scipy.signal import find_peaks
import multiprocessing
from multiprocessing import set_start_method


def get_files(path,station):
    files = glob.glob(path + station + "/" + "HHZ" + "/*", recursive=True)
    files = [f.replace("Z","*") for f in files]
    files.sort()
    return files


def ke_peaks(st,prominence,distance,fs):
    # square trace to get kinetic energy
    energy = np.square(np.array(st.data,dtype='float64'))

    # normalize amplitudes (helps with peak finding)
    energy = energy/np.max(energy)

    # find maxima in both bands
    peaks,_ = find_peaks(energy,prominence=prominence,distance=fs*distance)
    return peaks,energy


def triggers(trace,low_freq_peak,high_freq_peak,tolerance,buffer):
    # check if we have a high frequency ke peak followed by a low frequency ke peak
    if low_freq_peak - high_freq_peak < tolerance*trace.stats.sampling_rate:

        # append to list of detections for this channel
        trigger = trace.stats.starttime + high_freq_peak/trace.stats.sampling_rate - buffer       
        return trigger


def remove_repeats(det,distance):
    # remove double counting from final list
    for d in range(len(det)-1):
        if det[d+1] - det[d] < distance:
            det[d] = []

    # remove empty elements
    det = list(filter(None,det))
    return det


def get_waveforms(detections,st,buffer):
    detected_waveforms = obspy.Stream(traces=[])
    for d in detections:
        snippet = st.slice(starttime=d,endtime=d+buffer[0]+buffer[1])
        detected_waveforms += snippet
    return detected_waveforms


def save_detections(detections):
    home_dir = str(pathlib.Path().absolute())
    detection_timestamps = []
    for d in detections:
        detection_timestamps.append(d.strftime("%Y-%m-%dT%H:%M:%S"))
    out_file = h5py.File(home_dir + "/outputs/detections/ke_detections.h5","w")
    out_file.create_dataset("detections",data=detection_timestamps)
    out_file.close()

    
def save_waveforms(waveforms):
    home_dir = str(pathlib.Path().absolute())
    waveforms.write(home_dir + "/outputs/detections/ke_waveforms.h5",'H5',mode='a')
    
    
def detect(f,station,low_freq,high_freq,prominence,tolerance,distance,buffer):

    # make empty arrays to store detection times
    det = []

    # give some output
    print("Scanning " + f + "...\n")

    # read data files for all channels into one stream object
    st = obspy.read(f)

    # basic preprocessing
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # copy stream and filter to low and high frequency bands
    st_low = st.copy().filter("bandpass",freqmin=low_freq[0],freqmax=low_freq[1])
    st_high =  st.copy().filter("bandpass",freqmin=high_freq[0],freqmax=high_freq[1])

    # run trigger-finding algorithm for each channel
    for s in range(len(st)):

        # make empty arrays to store detections from current channel
        single_chan_det = []

        # find peaks in kinetic energy for each frequency band
        low_freq_peaks,low_freq_ke = ke_peaks(st_low[s],prominence,distance,st_low[s].stats.sampling_rate)
        high_freq_peaks,high_freq_ke = ke_peaks(st_high[s],prominence,distance,st_high[s].stats.sampling_rate)

        # check if peaks are concurrent in each band
        for h in range(len(high_freq_peaks)):
            for l in range(len(low_freq_peaks)):

                # skip to next iteration if low frequency detection is first
                if low_freq_peaks[l] - high_freq_peaks[h] < 0:
                    continue

                # get triggers if peaks are sufficiently close to each other
                trigger = triggers(st[s],low_freq_peaks[l],high_freq_peaks[h],tolerance,buffer[0])
                if trigger is not None:
                    single_chan_det.append(trigger)

        # remove double counting within current channel
        single_chan_det = remove_repeats(single_chan_det,tolerance)

        # append to list for current channel
        det.extend(single_chan_det)

    # sort detections
    det.sort()

    # if a detection is repeated 2 times (meaning it showed up on at least two channels), save it
    daily_det = []
    for d in range(len(det)-1):
        if det[d+1] - det[d] < tolerance:
            daily_det.append(det[d])

    # remove double counting from daily list
    detections = remove_repeats(daily_det,tolerance)
    
    # retrieve waveform snippets of detections from current day
    waveforms = get_waveforms(detections,st,buffer)

    return {'detections':detections, 'waveforms':waveforms}
    
    
def ke_detector(data_path,station,low_freq,high_freq,prominence,tolerance,distance,buffer):

    # get list of files to run detections on
    files = get_files(data_path,station)
    
    # construct iterable list of inputs for starmap
    inputs = [[f,station,low_freq,high_freq,prominence,tolerance,distance,buffer] for f in files]
    
    # choose number of processors; this defaults to the number of availble processors - 2
    n_procs = multiprocessing.cpu_count() - 2
    
    # map inputs to ke_detector
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=n_procs)
    outputs = p.starmap(detect,inputs)
    p.close()
    p.join()

    # extract detections and waveforms from list of outputs
    detections = []
    waveforms = obspy.Stream(traces=[])
    for output in outputs:
        detections.extend(output['detections'])
        waveforms += (output['waveforms'])

    # save list of detections
    save_detections(detections)
    
    # save list of waveforms
    save_waveforms(waveforms)

