import pathlib
import time
import obspy
from obspy.core.event import Event
from obspy.core.event import Catalog
from obspy.core.event import Origin
from obspy.signal.cross_correlation import correlate_template
from obspy.signal.trigger import coincidence_trigger
import obspyh5
import pyasdf
import numpy as np
import h5py
import glob
import copy
from scipy.signal import find_peaks
import multiprocessing
from multiprocessing import Manager
from multiprocessing import set_start_method
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt



def write_parameters(d):
    home_dir = str(pathlib.Path().absolute())
    with open(home_dir + "/outputs/detections/params.txt", 'w') as f:
        print(d.__dict__, file=f)
    
    

def get_files(path):
    files = glob.glob(path + "PIG2/" + "HHZ" + "/*", recursive=True)
    files = [f.replace("PIG2","*") for f in files]
    files.sort()
    return files



def make_dataset(xml_path,filename):
    # open ASDF files for output 
    home_dir = str(pathlib.Path().absolute())
    ds = pyasdf.ASDFDataSet(home_dir + "/outputs/detections/" + filename + ".h5",compression="gzip-3")
    
    # add station XML information
    files = glob.glob(xml_path + "*")
    for f in files:
        ds.add_stationxml(f)
    return ds



def make_templates(ds_catalog,template_times,high_freq,low_freq,xml_path):
    # make a new ASDF dataset for templates
    ds_template = make_dataset(xml_path,"templates")
        
    for i in range(len(template_times)):
        
        # make empty template stream and fill with appropriate traces
        template_stream = obspy.Stream(traces=[])
        for station in ds_catalog.ifilter(ds_catalog.q.starttime == obspy.UTCDateTime(template_times[i])):
            template_stream += station.stream
        
        # make obspy event for association with event waveforms and add to ASDF dataset
        event = Event()
        event.event_type = "ice quake"
        origin = Origin()
        origin.time = template_stream[0].stats.starttime
        event.origins = [origin]
        ds_template.add_quakeml(event)

        # process templates in both frequency bands
        template_stream.taper(max_percentage=0.1, max_length=30.)
        template_high = template_stream.copy()
        template_high.filter("bandpass",freqmin=high_freq[0],freqmax=high_freq[1])
        template_high.resample(high_freq[1]*2.1)
        template_low = template_stream
        template_low.filter("bandpass",freqmin=low_freq[0],freqmax=low_freq[1])
        template_low.resample(low_freq[1]*2.1)

        # add both bands of waveforms to ASDF dataset
        ds_template.add_waveforms(template_low,tag="low",event_id=event)
        ds_template.add_waveforms(template_high,tag="high",event_id=event)

        
        
def get_template_list(ds):
    low_freq_templates = []
    high_freq_templates = []
    for event in ds.events:
        template_low_stream = obspy.Stream(traces=[])
        template_high_stream = obspy.Stream(traces=[])
        for station in ds.ifilter(ds.q.event == event):
            template_low_stream += station.low
            template_high_stream += station.high
        low_freq_templates.append(template_low_stream)
        high_freq_templates.append(template_high_stream)
    return low_freq_templates, high_freq_templates


        
def read(f):
    # read data and handle basic preprocessing
    st = obspy.read(f)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)
    return st



def filter_stream(st,freq):
    # filter to the desired band and downsample to just above Nyquist to conserve memory
    st_filt = st.copy()
    st_filt.filter("bandpass",freqmin=freq[0],freqmax=freq[1])
    st_filt.resample(freq[1]*2.1)
    st_filt.merge()
    return st_filt



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
        diffs[diffs < -1*tolerance] = float("nan")

        # keep detection if there's a high frequency detection within the time tolerance of the low frequency detection
        if min(diffs) < tolerance:
            detections.append(det_low['time'])    
    return detections



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
    
    
    
def clean_and_sort_detections(detection_list):
    detections = []
    for det in detection_list:
        if det:
            detections.extend(det)

    # sort results
    detection_timestamps = []
    for det in detections:
        detection_timestamps.append(det.ns)
    def sort_func(d):
        return d.ns
    detections.sort(key=sort_func)
    
    return detections
     
    

def save_detections(detections):
    home_dir = str(pathlib.Path().absolute())
    detection_timestamps = []
    for d in detections:
        detection_timestamps.append(d.strftime("%Y-%m-%dT%H:%M:%S"))
    out_file = h5py.File(home_dir + "/outputs/detections/template_matching_detections.h5","w")
    out_file.create_dataset("detections",data=detection_timestamps)
    out_file.close()

    
    
def save_waveforms(detections,st,buffer,ds):
    # exit the method if list is empty
    if not detections:
        return
    
    # add an event for each detection
    event_list = []
    for detection_time in detections:
        
        # get waveform
        waveform = st.slice(starttime=detection_time-buffer[0],endtime=detection_time+buffer[1])
        
        # make obspy event for association with event waveforms
        event = Event()
        event.event_type = "ice quake"
        origin = Origin()
        origin.time = waveform[0].stats.starttime
        event.origins = [origin]

        # add event to ASDF dataset and event list
        event_list.append(event)
        ds.add_waveforms(waveform,tag="stream",event_id=event)
    return event_list
    
    

def detect(d):
    # make stream for storing cross correlation functions from each station
    low_freq_corr_stream = obspy.Stream(traces=[])
    high_freq_corr_stream = obspy.Stream(traces=[])    
        
    # iterate through stations in the template
    for s in range(len(d.template_low)):
        
        # check if stream has data on the current station of the template
        if len(d.stream_low.select(station=d.template_low[s].stats.station)) > 0:
           
            # cross correlate template with continuous data in both frequency bands
            low_freq_corr_func = correlate_template(d.stream_low.select(station=d.template_low[s].stats.station)[0],d.template_low[s],"same")
            high_freq_corr_func = correlate_template(d.stream_high.select(station=d.template_high[s].stats.station)[0],d.template_high[s],"same")

            # load resulting cross correlation functions into appropriate streams and copy metadata from stream of continuous data
            low_freq_corr_trace = obspy.Trace(np.abs(low_freq_corr_func))
            low_freq_corr_trace.stats = d.stream_low.select(station=d.template_low[s].stats.station)[0].stats
            low_freq_corr_stream += low_freq_corr_trace

            high_freq_corr_trace = obspy.Trace(np.abs(high_freq_corr_func))
            high_freq_corr_trace.stats = d.stream_high.select(station=d.template_high[s].stats.station)[0].stats
            high_freq_corr_stream += high_freq_corr_trace

    # find times when triggers occur on the specified number of stations in both frequency bands                   
    low_freq_detections = coincidence_trigger(None,d.low_thresh_on,d.low_thresh_off,low_freq_corr_stream,d.num_stations)
    high_freq_detections = coincidence_trigger(None,d.high_thresh_on,d.high_thresh_off,high_freq_corr_stream,d.num_stations)

    # run trigger-finding algorithm 
    detections = triggers(low_freq_detections,high_freq_detections,d.tolerance)

    # remove double counting from daily list
    detections = remove_repeats(detections,d.tolerance)
    
    return detections



def template_match(d): 
    # get home directory path
    home_dir = str(pathlib.Path().absolute())
    
    # write file with parameters for this run
    write_parameters(d)
    
    # create ASDF dataset and load station XML metadata 
    ds_catalog = make_dataset(d.xml_path,"template_matching_catalog")
    #ds_catalog = make_dataset(d.xml_path,"test")
    
    # read template ASDF dataset
    ds_templates = pyasdf.ASDFDataSet(home_dir + "/outputs/detections/templates.h5",mode='r')

    # make list of template streams
    low_freq_templates, high_freq_templates = get_template_list(ds_templates)
    
    # make vector of all filenames for a single channel
    files = get_files(d.data_path)

    # specify a specific file (for testing)
    #files = ["/media/Data/Data/PIG/MSEED/noIR/PIG*/HHZ/2012-12-19.PIG*.HHZ.noIR.MSEED"]

    # loop through all files and run detector parallel across templates
    event_list = []
    for f in files:

        # start timer
        timer = time.time()

        # read data files for all stations into one stream object
        st = read(f)
                
        # copy stream, filter to low frequency band
        st_low = filter_stream(st,d.low_freq)
        st_high = filter_stream(st,d.high_freq)

        inputs = []
        for i in range(len(low_freq_templates)):
            d.template_low = low_freq_templates[i]
            d.template_high = high_freq_templates[i]
            d.stream_low = st_low
            d.stream_high = st_high
            inputs.append(copy.deepcopy(d))
        
        # call parallel function
        multiprocessing.freeze_support()
        p = multiprocessing.Pool(processes=d.n_procs)
        detection_list = p.imap_unordered(detect,inputs)        
        p.close()
        p.join()
  
        # get rid of redundant detections and sort
        detections = clean_and_sort_detections(detection_list)

        # save detections and give output to user
        day_string = f.split("/")[9].split(".")[0]
        if len(detections) > 0:

            # remove redundant detections
            detections = remove_repeats(detections,d.tolerance)
            
            # save results by appending to open h5 file
            events = save_waveforms(detections,st,d.buffer,ds_catalog)
            
            # append events to overall event list (much faster to add quakeml to ASDF as a big obpsy catalog than individual events)
            event_list.extend(events)
            
            runtime = np.round(time.time() - timer)
            print("Finished template matching on " + day_string + " in " + str(runtime) + "s and found " + str(len(detections)) + " detections")
        else:
            runtime = np.round(time.time() - timer)
            print("Finished template matching on " +  day_string + " in " + str(runtime) + "s and found 0 detections")
    
    # add all obspy events to the ASDF dataset
    catalog = Catalog(event_list)
    ds_catalog.add_quakeml(catalog)
            
        
            
def plot_detections(ds):    
    for event in ds.events:
        for station in ds.ifilter(ds.q.event == event):
            station.stream.plot()    

    
            
def detection_timeseries(ds,fname,cluster=None,predictions=None):
    
    # extract times for each event in the dataset
    detection_times = []
    for event in ds.events:
        detection_times.append(event.origins[0].time.datetime)

    # subset to get only waves in a particular cluster, if provided
    if cluster is not None:
        detection_times = np.array(detection_times)[predictions == cluster]
        
    # get earliest and latest day of detections to make vector of bin edges
    start_day = min(detection_times).date()
    end_day = max(detection_times).date()+timedelta(days=1)
    binedges = [start_day + timedelta(days=x) for x in range(0,(end_day-start_day).days+1,1)]
        
    fig,ax = plt.subplots(figsize=[20,10])
    ax.hist(np.array(detection_times),binedges)
    ax.set(xlabel="Date")
    ax.set(ylabel="Number of Events")
    if cluster is not None:
        ax.set_title("Detection timeseries (cluster " + str(cluster) + ")")
    else:
        ax.set_title("Detection timeseries")       
    plt.savefig(fname)
