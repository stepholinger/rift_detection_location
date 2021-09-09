import pathlib
import multiprocessing
import glob
import gc
import copy
import numpy as np
import pyasdf
import obspy
from obspy.signal.trigger import coincidence_trigger 
from obspy.core.event import Event
from obspy.core.event import Origin



def get_files(path):
    files = glob.glob(path + "PIG2/" + "HHZ" + "/*", recursive=True)
    files = [f.replace("PIG2","*") for f in files]
    files.sort()
    return files



def read(f):
    # read data and handle basic preprocessing
    st = obspy.read(f)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)
    st.merge(fill_value='interpolate')
    return st



def filter_stream(st,freq):
    # filter to the desired band and downsample to just above Nyquist to conserve memory
    st.filter("bandpass",freqmin=freq[0],freqmax=freq[1])
    st.resample(freq[1]*2.1)
    return st



def get_data(st):
    data = []
    for s in range(len(st)):
        data.append(st[s].data)
    return data



def reset_stream(st,data,fs):
    for s in range(len(st)):
        st[s].data = data[s]
        st[s].stats.sampling_rate = fs
        delattr(st[s].stats,"processing")
    return st
    
    
    
def remove_repeats(detections,distance):
    # exit the method if list is empty
    if not detections:
        return
    
    # iterate through list, keeping earliest detection if there's double counting
    for d in range(len(detections)-1):
        if detections[d+1] - detections[d] < distance:
            detections[d+1] = detections[d]

    # remove empty elements
    detections = np.unique(detections)
    return detections.tolist()



def get_waveforms(detections,st,buffer):
    # exit the method if list is empty
    if not detections:
        return
    
    waveform_list = []
    for detection_time in detections:
        waveform = st.slice(starttime=detection_time-buffer[0],endtime=detection_time+buffer[1])
        waveform_list.append(waveform)
    return waveform_list



def save_waveforms(waveform_list,ds):
    # exit the method if list is empty
    if not waveform_list:
        return
    
    # add an event for each detection
    for waveform in waveform_list:
        # make obspy event for association with event waveforms
        event = Event()
        event.event_type = "ice quake"
        origin = Origin()
        origin.time = waveform[0].stats.starttime
        event.origins = [origin]

        # add waveform to ASDF dataset
        ds.add_waveforms(waveform,tag="stream",event_id=event)
        ds.add_quakeml(event)


        
def make_dataset(xml_path,filename):
    # open ASDF files for output 
    home_dir = str(pathlib.Path().absolute())
    ds = pyasdf.ASDFDataSet(home_dir + "/outputs/detections/" + filename + ".h5",compression="gzip-3")
    
    # add station XML information
    files = glob.glob(xml_path + "*")
    for f in files:
        ds.add_stationxml(f)
    return ds



def triggers(low_freq_detections,high_freq_detections,tolerance):
    # exit the method if either list is empty
    if not low_freq_detections or not high_freq_detections:
        return
    
    # make empty list for detections
    detections = []
    
    # iterate through list of low frequency detections
    for det_low in low_freq_detections:
        
        # calculate time difference between current low frequency detection and each high frequency detection
        diffs = np.array([det_low['time']-det_high['time'] for det_high in high_freq_detections])    
        
        # discard negative time differences to ensure we only get times with high frequency detections first
        diffs[diffs < -1] = float("nan")
        
        # keep detection if there's a high frequency detection within the time tolerance of the low frequency detection
        if min(diffs) < tolerance:
            detections.append(det_low['time'])    
    return detections



def detect(d):
    # give some output
    print("Scanning " + d.f + "...\n")

    # read data files for all channels into one stream object
    st = read(d.f)
    
    # save data and sampling rate of unfiltered data- this avoids having to copy streams, which is wasteful of memory 
    data = get_data(st)
    fs = st[0].stats.sampling_rate
    
    # filter to low frequency band, resample to save memory/increase speed, and run stalta detections
    st = filter_stream(st,d.low_freq)
    fs_low = st[0].stats.sampling_rate
    low_freq_detections = coincidence_trigger("classicstalta",d.low_thresh_on,d.low_thresh_off,st,d.num_stations,nsta=fs_low*d.sta_len,nlta=fs_low*d.lta_len)

    # reset stream to unfiltered state with original sampling rate
    st = reset_stream(st,data,fs)
    
    # filter to high frequency band, resample to save memory/increase speed, and run stalta detections
    st = filter_stream(st,d.high_freq)
    fs_high = st[0].stats.sampling_rate
    high_freq_detections = coincidence_trigger("classicstalta",d.high_thresh_on,d.high_thresh_off,st,d.num_stations,nsta=fs_high*d.sta_len,nlta=fs_high*d.lta_len)

    # reset stream to unfiltered state with original sampling rate
    st = reset_stream(st,data,fs)
    
    # run trigger-finding algorithm 
    detections = triggers(low_freq_detections,high_freq_detections,d.tolerance)
    
    # remove double counting from daily list
    detections = remove_repeats(detections,d.tolerance)
    
    # get list of waveforms using detection times
    waveforms = get_waveforms(detections,st,d.buffer)
    
    return waveforms

    
    
def stalta_detector(d):
    # get list of files to run detections on
    files = get_files(d.data_path)

    # create ASDF dataset and load station XML metadata 
    ds = make_dataset(d.xml_path,"stalta_catalog")
    
    # construct iterable list of detection parameter objects for imap
    inputs = []
    for f in files:
        d.f = f
        inputs.append(copy.deepcopy(d))

    # map inputs to stalta_detector and save as each call finishes
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=d.n_procs)
    for result in p.imap_unordered(detect,inputs):
        save_waveforms(result,ds)
    p.close()
    p.join()

