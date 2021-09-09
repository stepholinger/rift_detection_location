import pathlib
import time
import obspy
from obspy.signal.cross_correlation import correlation_detector
import obspyh5
import numpy as np
import h5py
import glob
from scipy.signal import find_peaks
import multiprocessing
from multiprocessing import Manager
from multiprocessing import set_start_method


def make_templates(waveforms,template_indices,high_freq,low_freq):
    home_dir = str(pathlib.Path().absolute())
    for i in range(len(template_indices)):
        index = template_indices[i]
        template = obspy.Stream(traces=[])
        for chan in ["Z","N","E"]:
            component_waves = waveforms.select(component=chan)
            template += component_waves[index]

        # copy, taper, filter, and downsample to both the low and high frequency bands
        temp_high = template.copy()
        temp_high.taper(max_percentage=0.1, max_length=30.)
        temp_high.filter("bandpass",freqmin=high_freq[0],freqmax=high_freq[1])
        temp_high.resample(high_freq[1]*2.1)
        temp_low = template
        temp_low.taper(max_percentage=0.1, max_length=30.)
        temp_low.filter("bandpass",freqmin=low_freq[0],freqmax=low_freq[1])
        temp_low.resample(low_freq[1]*2.1)

        # write both templates out to file
        temp_high.write(home_dir + "/outputs/detections/templates/template_" + str(i) + "_high.h5",'H5',mode='w')
        temp_low.write(home_dir + "/outputs/detections/templates/template_" + str(i) + "_low.h5",'H5',mode='w')
        

def read_data(file,low_freq,high_freq):
    
    # make filename with wildcard channel
    fname = file.replace("HHZ","H*")

    # read files and do basic preprocessing
    st = obspy.read(fname)
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # copy the file
    stream_low = st.copy()
    stream_high = st

    # filter and downsample the data to each band
    stream_low.filter("bandpass",freqmin=low_freq[0],freqmax=low_freq[1])
    stream_low.resample(low_freq[1]*2.1)
    stream_high.filter("bandpass",freqmin=high_freq[0],freqmax=high_freq[1])
    stream_high.resample(high_freq[1]*2.1)

    return stream_low,stream_high


def remove_repeats(detections,distance):
    final_detections = []
    final_detections.append(detections[0])
    for d in range(len(detections)-1):
        if detections[d+1] - detections[d] > distance:
            final_detections.append(detections[d+1])

    return final_detections


def save_detections(detections,f):
    
    # convert UTCDateTime objects to strings
    detection_timestamps = []
    for d in detections:
        detection_timestamps.append(d.strftime("%Y-%m-%dT%H:%M:%S"))
    
    # append and resize dataset
    f['detections'].resize((len(f['detections']) + len(detection_timestamps)), axis=0)
    f['detections'][-len(detection_timestamps):] = detection_timestamps


def detect(template_id,stream_low,cc_threshold_low,stream_high,cc_threshold_high,num_components,tolerance,distance):

    # get home directory path
    home_dir = str(pathlib.Path().absolute())
    
    # read in each band of templates
    template_low = obspy.read(home_dir + '/outputs/detections/templates/template_' + str(template_id) +'_low.h5')
    template_high = obspy.read(home_dir + '/outputs/detections/templates/template_' + str(template_id) +'_high.h5')
    #template_low = obspy.read(home_dir + '/outputs/detections/templates/test/tempLow_' + str(template_id) +'.h5')
    #template_high = obspy.read(home_dir + '/outputs/detections/templates/test/tempHigh_' + str(template_id) +'.h5')

    # make a couple useful list
    triggers = []
    detections = []

    # iterate through each channel
    for s in range(len(template_low)):

        # call the template matching function in each band
        low_freq_detections,_ = correlation_detector(obspy.Stream(stream_low[s]),obspy.Stream(template_low[s]),cc_threshold_low,distance)
        high_freq_detections,_ = correlation_detector(obspy.Stream(stream_high[s]),obspy.Stream(template_high[s]),cc_threshold_high,distance)

        # get all high frequency trigger times for today
        high_freq_detection_times = []
        for i in range(len(high_freq_detections)):
            high_freq_detection_times.append(high_freq_detections[i].get("time"))

        # loop through all low frequency triggers for today
        for i in range(len(low_freq_detections)):
            low_freq_detection_time = low_freq_detections[i].get("time")

            # calculate time difference between low freq trigger and all high freq triggers
            diffs = np.subtract(low_freq_detection_time,high_freq_detection_times)

            # only interested in positive values of 'diffs', which indicates high freq trigger first
            diffs[diffs < -1*tolerance] = float("nan")

            # save low freq trigger if a high freq trigger is sufficiently close
            if len(diffs) > 0:
                if min(diffs) < tolerance:
                    triggers.append(low_freq_detection_time)

    # sort detections chronologically
    triggers.sort()

    # save detections if they show up on desired number of components
    if len(triggers) > 0:
        for d in range(len(triggers)-num_components-1):
            if triggers[d+num_components-1] - triggers[d] < tolerance:
                detections.append(triggers[d])
        
        #print("Found " + str(len(detections)) + " detections with template " + str(template_id))

    return detections


def template_match(data_path,station,high_freq,low_freq,cc_threshold_low,cc_threshold_high,num_components,tolerance,distance):
    
    # get home directory path
    home_dir = str(pathlib.Path().absolute())
    
    # get number of templates and list of template ids
    template_list = glob.glob(home_dir + '/outputs/detections/templates/*')
    #template_list = glob.glob(home_dir + '/outputs/detections/templates/test/*')
    num_templates =  int(len(template_list)/2)
    template_ids = range(num_templates)
                          
    # make vector of all filenames for a single channel
    files = []
    file_list = glob.glob(data_path + station + "/HHZ/*")
    file_list.sort()
    files.extend(file_list)

    # specify a specific file (for testing)
    #files = ["/media/Data/Data/PIG/MSEED/noIR/PIG2/HHZ/2012-01-10.PIG2.HHZ.noIR.MSEED"]

    # open h5 file for output
    detection_file = h5py.File(home_dir + "/outputs/detections/template_detections.h5","w")
    detection_file.create_dataset('detections',data=[],compression="gzip",chunks=True, maxshape=(None,),dtype=h5py.string_dtype(encoding='utf-8'))
    
    # loop through all files and run detector parallel across templates
    for f in range(len(files)):

        # start timer
        timer = time.time()

        # read continuous data to scan
        stream_low,stream_high = read_data(files[f],low_freq,high_freq)

        # construct iterable list of inputs for starmap
        inputs = [[id,stream_low,cc_threshold_low,stream_high,cc_threshold_high,num_components,tolerance,distance] for id in template_ids]

        # choose number of processors; this defaults to the number of availble processors - 2
        n_procs = multiprocessing.cpu_count() - 2
            
        # call parallel function
        multiprocessing.freeze_support()
        p = multiprocessing.Pool(processes=n_procs)
        detection_list = p.starmap(detect,inputs)
        p.close()
        p.join()
        
        # extract detections from starmap output
        detections = []
        for d in detection_list:
            detections.extend(d)
        
        # sort results
        detection_timestamps = []
        for d in detections:
            detection_timestamps.append(d.ns)
        def sort_func(d):
            return d.ns
        detections.sort(key=sort_func)
     
        # stop timer
        runtime = time.time() - timer

        # save detections and give output to user
        if len(detections) > 0:

            # remove redundant detections
            detections = remove_repeats(detections,tolerance)

            # save results by appending to open h5 file
            save_detections(detections,detection_file)
            
            print("Finished template matching on " + files[f].split("/")[9].split(".")[0] + " in " + str(np.round(runtime)) + "s and found " + str(len(detections)) + " detections")
        else:
            print("Finished template matching on " + files[f].split("/")[9].split(".")[0] + " in " + str(np.round(runtime)) + "s and found 0 detections")
    
    # close h5 file 
    detection_file.close()