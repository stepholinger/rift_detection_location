import obspy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import date2num, DateFormatter
from obspy.signal.cross_correlation import correlate, xcorr_max
import types
import time
from collections import Counter, namedtuple
import scipy
import pathlib
import glob
import copy
import multiprocessing
import pickle
 
    
    
def get_files(l):
    files = glob.glob(l.data_path + "PIG2/" + "HHZ" + "/*", recursive=True)
    files = [f.replace("HHZ","HH*") for f in files]
    files.sort()
    start_date = l.detection_times[0].strftime("%Y-%m-%d")
    start_index = [s for s in range(len(files)) if start_date in files[s]][0]
    end_date = l.detection_times[-1].strftime("%Y-%m-%d")
    end_index = [s for s in range(len(files)) if end_date in files[s]][0]+1
    files = files[start_index:end_index]
    return files



def get_detections_today(l):
    current_date = datetime.strptime(l.f.split("/")[9].split(".")[0],"%Y-%m-%d")
    bool_indices = np.logical_and(l.detection_times>=current_date,l.detection_times<current_date + timedelta(days=1))
    detections_today = l.detection_times[bool_indices]
    if sum(bool_indices) == 0:
        indices = [0,0]
    else:
        indices = [[i for i, x in enumerate(bool_indices) if x][0],[i for i, x in enumerate(bool_indices) if x][-1]+1]
    return detections_today, indices
  
    
    
def load_waveform(r,detection_time):
    # read in data for template
    date_string = detection_time.date().strftime("%Y-%m-%d")
    fname = r.data_path + r.station + "/HHZ/*" + date_string + "*"
    st = obspy.read(fname)

    # basic preprocessing
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # filter and resample the data to each band
    st.filter("bandpass",freqmin=r.freq[0],freqmax=r.freq[1])
    st.resample(r.freq[1]*2.1)
    return st



def get_trace(r,event):
    fs = r.freq[1]*2.1
    waveform = np.zeros((int(r.trace_length*fs+1)))
    trace = event[0].data
    waveform[:len(trace)] = trace
    return waveform



def read_waveforms(r):
    # get detection times for current file
    detection_times_today, indices = get_detections_today(r)
    num_detections_today = len(detection_times_today)
    
    if detection_times_today.size:
        # read current file
        st = load_waveform(r,detection_times_today[0])

        # make container for storing rotated waveforms
        waveform_matrix = np.zeros((num_detections_today,int(r.trace_length*r.fs+1)))

        for i in range(num_detections_today):
            # trim trace of entire day and rotate
            event_st = st.copy()
            event_st.trim(starttime=obspy.UTCDateTime(detection_times_today[i]),endtime=obspy.UTCDateTime(detection_times_today[i]) + r.trace_length)
            try:
                #event_st.rotate('NE->RT',back_azimuth=r.backazimuths[indices[0]:indices[1]][i])
                waveform = get_trace(r,event_st)
                waveform_matrix[i,:] = waveform  
            except:
                waveform_matrix[i,:] = np.zeros((int(r.trace_length*r.fs+1)))
    else:
        waveform_matrix = []
        indices = []
    return waveform_matrix, indices



def get_waveforms(r): 
    
    # make output array
    r.fs = r.freq[1]*2.1
    waveform_matrix = np.zeros((len(r.detection_times),int(r.trace_length*r.fs+1)))
    
    # make vector of all filenames
    files = get_files(r)
    
    # construct iterable list of detection parameter objects for imap
    inputs = []
    for f in files:
        r.f = f
        inputs.append(copy.deepcopy(r))

    # map inputs to polarization_analysis and save as each call finishes
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=r.n_procs)
    for result in p.imap_unordered(read_waveforms,inputs):
        if result[1]:
            waveform_matrix[result[1][0]:result[1][1],:] = result[0]

    return waveform_matrix



def get_stacks(waveforms_stack,waveforms_corr,detection_dates,correlation_coefficients,cluster,shifts,fs,trace_length):    

    # make empty array for storage
    aligned_cluster_events = np.zeros((len(waveforms_stack),int(trace_length*fs+1)))
    master_event = waveforms_corr[np.argmax(np.abs(correlation_coefficients))]
    
    #polarity_count = 0
    # iterate through all waves in the current cluster
    for w in range(len(waveforms_stack)):

        # put both waveforms into an obspy stream
        master_trace = obspy.Trace(master_event)
        trace = obspy.Trace(waveforms_corr[w])

        # cross correlate with master event
        correlation_timeseries = correlate(master_trace,trace,1000)
        shift, correlation_coefficient = xcorr_max(correlation_timeseries)

        # get waveform from stacking frequency band
        trace = waveforms_stack[w]

        # flip polarity if necessary
        if correlation_coefficient < 0:
            #polarity_count += 1/3
            trace = trace * -1

        if shift > 0:
            aligned_trace = np.append(np.zeros(abs(int(shift))),trace)
            aligned_trace = aligned_trace[:int(trace_length*fs+1)]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

        else:
            aligned_trace = trace[abs(int(shift)):]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

    start_date = detection_dates[0]
    end_date = detection_dates[-1]
    num_days = end_date-start_date

    daily_stacks = []
    for d in range(num_days.days):
        day = start_date + timedelta(days=d)
        waves_today = aligned_cluster_events[detection_dates == day,:]
        daily_stack = np.divide(np.nansum(waves_today,axis = 0),np.sum([detection_dates == day]))
        daily_stacks.append(daily_stack)
    stack = np.nanmean(aligned_cluster_events,axis=0)
    #print(polarity_count/len(waveforms_stack))
    return daily_stacks,stack
    
    

def get_stacks_master(waveforms_stack,master_event,detection_dates,fs,trace_length):    

    # make empty array for storage
    aligned_cluster_events = np.zeros((len(waveforms_stack),int(trace_length*fs+1)))
    
    #polarity_count = 0
    # iterate through all waves in the current cluster
    for w in range(len(waveforms_stack)):

        # put both waveforms into an obspy stream
        master_trace = obspy.Trace(master_event)
        trace = obspy.Trace(waveforms_stack[w])

        # cross correlate with master event
        correlation_timeseries = correlate(master_trace,trace,1000)
        shift, correlation_coefficient = xcorr_max(correlation_timeseries)

        # get waveform from stacking frequency band
        trace = waveforms_stack[w]

        # flip polarity if necessary
        if correlation_coefficient < 0:
            #polarity_count += 1/3
            trace = trace * -1

        if shift > 0:
            aligned_trace = np.append(np.zeros(abs(int(shift))),trace)
            aligned_trace = aligned_trace[:int(trace_length*fs+1)]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

        else:
            aligned_trace = trace[abs(int(shift)):]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

    start_date = detection_dates[0]
    end_date = detection_dates[-1]
    num_days = end_date-start_date

    daily_stacks = []
    for d in range(num_days.days):
        day = start_date + timedelta(days=d)
        waves_today = aligned_cluster_events[detection_dates == day,:]
        daily_stack = np.divide(np.nansum(waves_today,axis = 0),np.sum([detection_dates == day]))
        daily_stacks.append(daily_stack)
    stack = np.nanmean(aligned_cluster_events,axis=0)
    #print(polarity_count/len(waveforms_stack))
    return daily_stacks,stack
    
    
    
def get_radial_tilt_stack(waveforms_stack,waveforms_corr,correlation_coefficients,cluster,shifts,freq,trace_length):    
    
    # identify "master event" which will be the event from that was best correlated with the cluster centroid earlier
    master_event = waveforms_corr[np.argmax(np.abs(correlation_coefficients))]
    master_event_tilt = np.gradient(master_event[1],1/2.1)/(9.8)
    master_trace = obspy.Trace(master_event_tilt)

    # make empty array for storage
    fs = freq[1]*2.1
    aligned_cluster_events = np.zeros((len(waveforms_stack),int(trace_length*fs+1)))
    
    # iterate through all waves in the current cluster
    for w in range(len(waveforms_stack)):

        # cross correlate and align each component of the trace w.r.t the master event
        event_tilt = np.gradient(waveforms_corr[w][1],1/2.1)/(9.8)
        trace = obspy.Trace(event_tilt)

        # cross correlate with master event
        correlation_timeseries = correlate(master_trace,trace,500)
        shift, correlation_coefficient = xcorr_max(correlation_timeseries)

        # get waveform from stacking frequency band
        trace = np.gradient(waveforms_stack[w][1],1/2.1)/(9.8)

        # flip polarity if necessary
        if correlation_coefficient < 0:
            trace = trace * -1

        if shift > 0:
            aligned_trace = np.append(np.zeros(abs(int(shift))),trace)
            aligned_trace = aligned_trace[:int(trace_length*fs+1)]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

        else:
            aligned_trace = trace[abs(int(shift)):]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

    stack = np.nanmean(aligned_cluster_events,axis=0)
    return stack
    
    