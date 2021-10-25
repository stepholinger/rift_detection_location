import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
from sklearn.decomposition import PCA
from pyproj import Proj,transform
from datetime import datetime
from datetime import timedelta
import types
import time
from collections import Counter
import scipy
import pathlib
import glob
import copy
import multiprocessing
import pickle



def write_parameters(d):
    home_dir = str(pathlib.Path().absolute())
    with open(home_dir + "/outputs/locations/params.txt", 'w') as f:
        print(d.__dict__, file=f)

        

def get_files(path):
    files = glob.glob(path + "PIG2/" + "HHZ" + "/*", recursive=True)
    files = [f.replace("PIG2","PIG*") for f in files]
    files = [f.replace("HHZ","HH*") for f in files]
    files.sort()
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



def first_observed_arrival(st,l):
    # cross correlate all traces and find station with largest shift
    channels = ["HHZ","HHN","HHE"]
    first_stat_vector = []
    for chan in channels:
        st_chan = st.select(channel=chan)
        shifts = np.zeros(len(st_chan))
        corrs = np.zeros(len(st_chan))
        for j in range(len(st_chan)):
            #corr = correlate(st_single_chan[0].data,st_single_chan[j].data)
            corr = correlate(st_chan[0], st_chan[j], l.max_shift)
            shift, correlation_coefficient = xcorr_max(corr,abs_max=True)
            shifts[j] = shift
            corrs[j] = correlation_coefficient
        stat_idx = np.argmax(shifts)
        first_stat_vector.append(st_chan[stat_idx].stats.station)
    counts = Counter(first_stat_vector).most_common(2)
    if len(counts) > 1:
        if counts[0][1] == counts[1][1]:
            first_stat = []
        else:
            first_stat = counts[0][0]
    else:
        first_stat = counts[0][0]
    return first_stat



def get_detection_times(ds):
    # extract times for each event in the dataset
    detection_times = []
    for event in ds.events:
        detection_times.append(event.origins[0].time.datetime)
    return np.array(detection_times)



def get_station_coordinates(l):
    stat_coords = []
    inv = obspy.read_inventory(l.xml_path + "/*")
    channels = inv.get_contents()['channels']
    for channel in channels:
        lat = inv.get_coordinates(channel)["latitude"]
        lon = inv.get_coordinates(channel)["longitude"]
        stat_coords.append([lon,lat])
    _, idx = np.unique(stat_coords,axis=0,return_index=True)
    stat_coords = np.array(stat_coords)[np.sort(idx)]
    return stat_coords



def get_array_centroid(l):
    # convert station coordinates to x and y and take average station location
    avg_stat_x = np.mean(l.station_grid_coords[:,0])
    avg_stat_y = np.mean(l.station_grid_coords[:,1])
    return [avg_stat_x,avg_stat_y]



def get_station_locations(l):
    # convert station coordinates to x and y and take average station location
    p2 = Proj("EPSG:3031",preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)
    [stat_x,stat_y] = transform(p1,p2,l.station_lon_lat_coords[:,0],l.station_lon_lat_coords[:,1])
    return np.transpose(np.array([stat_x,stat_y]))



def get_station_angles(l):
    # compute angles from North (0 deg) to each station, centered about the array centroid
    # deal with the fact that EPSG:3031 is on a grid like [meters N, meters E] with the first axis positive-South
    station_locations_corrected = np.transpose(np.array([l.station_grid_coords[:,1],l.station_grid_coords[:,0]*-1]))
    array_centroid_corrected = [l.array_centroid[1],l.array_centroid[0]*-1]
    station_angles = []
    for i in range(len(station_locations_corrected)):
        angle = get_baz(station_locations_corrected[i,:]-array_centroid_corrected)
        station_angles.append(angle)
    return station_angles



def make_results_object(l):
    baz_object = types.SimpleNamespace()
    #baz_object.all_first_components = np.empty((l.num_steps*l.num_detections,2,len(l.stations)),"float64")
    #baz_object.all_first_components[:,:] = np.NaN
    #baz_object.norms = np.empty((l.num_steps*l.num_detections),"float64")
    #baz_object.norms[:] = np.NaN
    #baz_object.indices = np.empty((l.num_steps*l.num_detections),"float64")
    #baz_object.indices[:] = np.NaN
    baz_object.backazimuths = np.empty((l.num_detections),'float64')
    baz_object.backazimuths[:] = np.NaN
    baz_object.uncertainties = np.empty((l.num_detections),'float64')
    baz_object.uncertainties[:] = np.NaN
    #baz_object.errors = ["" for x in range(l.num_detections)]
    return baz_object



def check_data_quality(st,l):
    stations_to_remove = []
    for i in range(len(st)):
        snr = max(abs(st[i].data))/np.mean(abs(st[i].data))
        if snr < l.snr_threshold:
            stations_to_remove.append(st[i].stats.station)
    stations_to_remove = np.unique(stations_to_remove)
    for stat in stations_to_remove:
        for trace in st.select(station=stat):
            st.remove(trace)
    return st



def check_trace_length(st):
    uneven_length = 0
    first_length = len(st[0].data)
    for trace in st:
        if not len(trace) == first_length:
            uneven_length = 1
    return uneven_length



def get_baz(pca_components):
    # based on which quadrant we're in, use the appropriate triangle side ratios for arctan
    if pca_components[0] > 0 and pca_components[1] > 0:
        baz = np.arctan(abs(pca_components[0]/pca_components[1]))*180/np.pi
    if pca_components[0] > 0 and pca_components[1] < 0:
        baz = np.arctan(abs(pca_components[1]/pca_components[0]))*180/np.pi + 90
    if pca_components[0] < 0 and pca_components[1] < 0:
        baz = np.arctan(abs(pca_components[0]/pca_components[1]))*180/np.pi + 180
    if pca_components[0] < 0 and pca_components[1] > 0:
        baz = np.arctan(abs(pca_components[1]/pca_components[0]))*180/np.pi + 270
    return baz



def compute_pca(st,l):
    # make array for storage of pca components
    first_component_vect = np.empty((0,2),"float64")

    # get mean amplitude for whole trace (average of both components)
    horz_data = np.transpose(np.concatenate(([st.select(channel="HHE")[0].data],[st.select(channel="HHN")[0].data])))

    # itertate through data in windows
    for n in range(l.num_steps):
        # get current window
        start_ind = n * l.slide * l.fs
        end_ind = start_ind + l.win_len*l.fs
        X = horz_data[start_ind:end_ind,:]

        # only progress if matrix of data is not empty
        if X.size > 0:
            # normalize and compute the PCA if staLta criteria is met for BOTH components
            if np.mean(abs(X[:,0])) > l.stalta_threshold*np.mean(abs(horz_data[:,0])) and np.mean(abs(X[:,1])) > l.stalta_threshold*np.mean(abs(horz_data[:,1])):

                # find component with max amplitude, normalize both traces by that max value, and compute PCA
                max_amp = np.amax(abs(X))
                X_norm = np.divide(X,max_amp)
                pca = PCA(n_components = 2)
                pca.fit(X_norm)

                # flip pca components based on station of first arrival
                first_components = correct_pca(pca.components_[0,:],l)

                # save result
                first_component_vect = np.vstack((first_component_vect,first_components))

            else:
                # add zeros if we didn't run PCA on the window due to low STALTA
                first_component_vect = np.vstack((first_component_vect,[np.nan,np.nan]))
        else:
            # add zeros if we didn't run PCA on the window due to emptiness
            first_component_vect = np.vstack((first_component_vect,[np.nan,np.nan]))
    return first_component_vect



def angle_difference(angle_1,angle_2):
    diff_1 = abs(angle_1 - angle_2)
    diff_2 = 360 - diff_1
    return min(diff_1,diff_2)



def closest_station(baz,l):
    radial_diffs = []
    for i in range(len(l.stations)):
        radial_diffs.append(angle_difference(baz,l.station_angles[i]))
    station = l.stations[np.argmin(radial_diffs)]
    return station



def correct_pca(pca_components,l):
    # get the backazimuth corresponding to the initial pca first components
    baz = get_baz(pca_components)
    
    # get the other possible backazimuth
    if baz < 180:
        baz_180 = baz + 180
    if baz > 180:
        baz_180 = baz - 180

    # get the stations closest to these backazimuths
    predicted_station = closest_station(baz,l)
    predicted_station_180 = closest_station(baz_180,l)
    
    # check if the observed station of first arrival agrees with either of these predicted backazimuths
    if l.first_stat == predicted_station:
        corrected_pca_components = pca_components 
    if l.first_stat == predicted_station_180:
        corrected_pca_components = pca_components*-1
    else:
        corrected_pca_components = [np.nan,np.nan]
    return corrected_pca_components



def circular_stdev(pca_components,weights):
    cos_sum = 0
    sin_sum = 0
    for i in range(len(pca_components)):
        if not np.isnan(pca_components[i,0]):
            cos = weights[i]*pca_components[i,0]
            cos_sum += cos
            sin = weights[i]*pca_components[i,1]
            sin_sum += sin
    cos_avg = cos_sum/np.sum(weights)
    sin_avg = sin_sum/np.sum(weights)
    R = np.sqrt(cos_avg*cos_avg+sin_avg*sin_avg)
    stdev = np.sqrt(-2*np.log(R))
    #stdev = np.sqrt(2*(1-R))
    return stdev



def polarization_analysis(l):

    # get detection times from ASDF dataset and insert dummy time at the end for convenience
    detection_times_today, indices = get_detections_today(l)
    num_detections_today = len(detection_times_today)
    detection_times_today = np.append(detection_times_today,datetime(1970,1,1,0,0,0))

    # read and filter the day of data
    st = obspy.read(l.f)
    st.filter("bandpass",freqmin=l.freq[0],freqmax=l.freq[1])

    # make containers for today's results
    event_baz_vect = np.empty((num_detections_today),'float64')
    event_baz_vect[:] = np.nan
    event_uncertainty_vect = np.empty((num_detections_today),'float64')
    event_uncertainty_vect[:] = np.nan
    
    # run polarization analysis for all events in the current file / on the current day
    for i in range(num_detections_today):
    
        # make arrays for storing PCA results
        all_first_components = np.empty((l.num_steps,2,len(l.stations)),"float64")
        all_first_components[:,:,:] = np.nan
    
        # get UTCDateTime and current date for convenience
        detection_utc_time = obspy.UTCDateTime(detection_times_today[i])

        # check if more than one event on current day; if so, read entire day. If not, just read the event.
        st_event = st.copy()
        st_event.trim(starttime=detection_utc_time,endtime=detection_utc_time+l.trace_len)
        st_event.taper(max_percentage=0.1, max_length=30.)

        # check for gaps and remove stations with bad data quality for this event
        start_time = st_event[0].stats.starttime
        if check_trace_length(st_event):
            #print("Skipped event at "  + str(start_time) + " due to traces with uneven length")
            #baz_object.errors[i] = "Skipped event at "  + str(start_time) + " due to traces with uneven length"
            continue

        # check SNR and if all traces were removed due to poor snr, skip this event
        st_event = check_data_quality(st_event,l)
        if not st_event:
            #print("Skipped event at "  + str(start_time) + " due to poor SNR on all stations")
            #baz_object.errors[i] = "Skipped event at "  + str(start_time) + " due to poor SNR on all stations"
            continue
            
        # loop through stations to get one trace from each to find earliest arrival
        l.first_stat = first_observed_arrival(st_event,l)
        if not l.first_stat:
            #print("Skipped event at "  + str(start_time) + " due to indeterminate first arrival")
            #baz_object.errors[i] = "Skipped event at "  + str(start_time) + " due to indeterminate first arrival"
            continue
            
        # loop though stations to perform PCA on all windows in the event on each station's data
        for s in range(len(l.stations)):
            
            # check if there's data and skip current station if not
            if not st_event.select(station=l.stations[s]):
                continue
                
            # compute pca components for all windows in the event
            pca_first_components = compute_pca(st_event.select(station=l.stations[s]),l)

            # fill array in results object
            #baz_object.all_first_components[i*l.num_steps:(i+1)*l.num_steps,:,s] = pca_first_components
            all_first_components[:,:,s] = pca_first_components
        
        # sum results (this is vector sum across stations of pca first components for each window)
        first_component_sums = np.nansum(all_first_components,axis=2)

        # take average weighted by norm of PCA component sums to get single mean event backazimuth
        norms = np.linalg.norm(first_component_sums,axis=1)
        denom = np.sum(norms)
        avg_weighted_x = np.sum(first_component_sums[:,0]*norms)/denom
        avg_weighted_y = np.sum(first_component_sums[:,1]*norms)/denom
        if not np.sum(first_component_sums) == 0:
            event_baz_vect[i] = get_baz([avg_weighted_x,avg_weighted_y])
        
        # rescale norms to get weight vector whose sum equals the number of windows for which we calculated PCA (this is necessary for the sqrt(-2log(R)) in the circular standard deviation calculation)
        weights = norms/(np.sum(norms)/np.sum([norms != 0]))
        normalized_first_component_sums = first_component_sums
        for n in range(len(normalized_first_component_sums)):
            if not np.sum(first_component_sums[n,0]) == 0:
                vect_len = np.sqrt(first_component_sums[n,0]*first_component_sums[n,0]+first_component_sums[n,1]*first_component_sums[n,1])
                normalized_first_component_sums[n,0] = first_component_sums[n,0]/vect_len
                normalized_first_component_sums[n,1] = first_component_sums[n,1]/vect_len
        event_uncertainty_vect[i] = (circular_stdev(normalized_first_component_sums,weights)*180/np.pi)
        print("Finished with " + str(detection_times_today[i].date()))
    return event_baz_vect,event_uncertainty_vect,indices


#def compute_backazimuths(l,detection_times): 
def compute_backazimuths(l,ds): 
    
    # get home directory path
    home_dir = str(pathlib.Path().absolute())
    
    # get useful geometric information about the array
    l.station_lon_lat_coords = get_station_coordinates(l)
    l.station_grid_coords = get_station_locations(l)
    l.array_centroid = get_array_centroid(l)
    l.station_angles = get_station_angles(l)
    
    # get all detection times
    l.detection_times = get_detection_times(ds)
    #l.detection_times = detection_times
    l.num_detections = len(l.detection_times)
        
    # make object for storing pca vector sums and storing data to plot
    b = make_results_object(l)

    # write file with parameters for this run
    write_parameters(l)
    
    # open output file
    baz_file = open(l.filename, "wb")
    
    # make vector of all filenames
    files = get_files(l.data_path)
    print("Got all files...")
    # construct iterable list of detection parameter objects for imap
    inputs = []
    for f in files:
        l.f = f
        inputs.append(copy.deepcopy(l))
    print("Made inputs...")        
    # map inputs to polarization_analysis and save as each call finishes
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=l.n_procs)
    for result in p.imap_unordered(polarization_analysis,inputs):
        b.backazimuths[result[2][0]:result[2][1]] = result[0]
        b.uncertainties[result[2][0]:result[2][1]] = result[1]
        pickle.dump(b, baz_file)
    return b






#def correct_pca_sector(pca_components,l):
#     # get the backazimuth corresponding to the observed polarization direction
#     polarization_direction = get_baz(pca_components)
    
#     # get angles orthogonal to polarization direction
#     if polarization_direction < 270 and polarization_direction > 90:
#         sector_angles = [polarization_direction - 90, polarization_direction + 90]
#     if polarization_direction > 270:
#         sector_angles = [polarization_direction - 270, polarization_direction - 90]
#     if polarization_direction < 90:
#         sector_angles = [polarization_direction + 90, polarization_direction + 270]       
      
#     # make a list of stations in each sector
#     sector_1_stations = []
#     sector_2_stations = []
#     for s in range(len(l.stations)):
#         if l.station_angles[s] > sector_angles[0] and l.station_angles[s] < sector_angles[1]:
#             sector_1_stations.append(l.stations[s])
#         else:
#             sector_2_stations.append(l.stations[s])
    
#     # check if observed station of first arrival is in sector 1
#     if l.first_stat in sector_1_stations:
#         # if initially-calculated polarization direction is also this sector, don't flip pca components
#         if polarization_direction > sector_angles[0] and polarization_direction < sector_angles[1]:
#             corrected_pca_components = pca_components
#         # if initially-calculated polarization direction is not in this sector, flip pca components
#         if polarization_direction < sector_angles[0] or polarization_direction > sector_angles[1]:
#             corrected_pca_components = pca_components*-1
    
#     # check if observed station of first arrival is in sector 2       
#     if l.first_stat in sector_2_stations:
#         # if initially-calculated polarization direction is also this sector, don't flip pca components
#         if polarization_direction < sector_angles[0] or polarization_direction > sector_angles[1]:
#             corrected_pca_components = pca_components
#         # if initially-calculated polarization direction is not in this sector, flip pca components
#         if polarization_direction > sector_angles[0] and polarization_direction < sector_angles[1]:
#             corrected_pca_components = pca_components*-1       
#     return corrected_pca_components



# simple 1 channel correlation function, closely based on tslearn's correlation code
# def correlate(trace_1,trace_2):

#     # get input vectors into correct form
#     trace_1 = np.array([[i] for i in trace_1])
#     trace_2 = np.array([[i] for i in trace_2])

#     # get some useful values
#     sz = int(trace_1.shape[0])

#     # get some useful values
#     n = 2*sz-1;
#     fft_sz = 1 << n.bit_length()
#     denom = 0.

#     # compute norms of each vector
#     norm_1 = np.linalg.norm(trace_1)
#     norm_2 = np.linalg.norm(trace_2)
#     denom = norm_1 * norm_2

#     # compute cross correlation
#     cc = np.real(np.fft.ifft(np.fft.fft(trace_1, fft_sz, axis=0) *
#                                    np.conj(np.fft.fft(trace_2, fft_sz, axis=0)), axis=0))
#     cc = np.vstack((cc[-(sz-1):], cc[:sz]))
#     cc = np.real(cc).sum(axis=-1) / denom
#     return cc



# def compute_backazimuths_test(l,detection_times):

#     # get detection times from ASDF dataset and insert dummy time at the end for convenience
#     #detection_times = get_detection_times(ds)
#     l.num_detections = len(detection_times)
#     detection_times.append(datetime(1970,1,1,0,0,0))
    
#     # get useful geometric information about the array
#     l.station_lon_lat_coords = get_station_coordinates(l)
#     l.station_grid_coords = get_station_locations(l)
#     l.array_centroid = get_array_centroid(l)
#     l.station_angles = get_station_angles(l)

#     # make object for storing pca vector sums and storing data to plot
#     baz_object = make_results_object(l)

#     # for each cluster, run polarization analysis for all events
#     for i in range(len(detection_times)-1):
#         print(detection_times[i])
#         t = time.time()
    
#         # get UTCDateTime and current date for convenience
#         current_date = detection_times[i].date()
#         detection_utc_time = obspy.UTCDateTime(detection_times[i])

#         # check if more than one event on current day; if so, read entire day. If not, just read the event.
#         #if detection_times[i+1].date() == current_date and detection_times[i-1].date() != current_date:
#         st = obspy.read(l.data_path+"PIG*/HH*/*" + current_date.strftime("%Y-%m-%d") + "*")
#         st.filter("bandpass",freqmin=l.freq[0],freqmax=l.freq[1])
#         st_event = st.copy()
#         st_event.trim(starttime=detection_utc_time,endtime=detection_utc_time+l.trace_len)
#         st_event.taper(max_percentage=0.1, max_length=30.)
# #         elif detection_times[i-1].date() == current_date:
# #             st_event = st.copy()
# #             st_event.trim(starttime=detection_utc_time,endtime=detection_utc_time+l.trace_len)
# #             st_event.taper(max_percentage=0.1, max_length=30.)
# #         else:
# #             st_event = obspy.read(l.data_path+"PIG*/HH*/*" + current_date.strftime("%Y-%m-%d") + "*",starttime=detection_utc_time,endtime=detection_utc_time+l.trace_len)
# #             st_event.taper(max_percentage=0.1, max_length=30.)
# #             st_event.filter("bandpass",freqmin=l.freq[0],freqmax=l.freq[1])
        
#         # check for gaps and remove stations with bad data quality for this event
#         start_time = st_event[0].stats.starttime
#         if check_trace_length(st_event):
#             print("Skipped event at "  + str(start_time) + " due to traces with uneven length")
#             baz_object.errors[i] = "Skipped event at "  + str(start_time) + " due to traces with uneven length"
#             continue

#         # check SNR and if all traces were removed due to poor snr, skip this event
#         st_event = check_data_quality(st_event,l)
#         if not st_event:
#             print("Skipped event at "  + str(start_time) + " due to poor SNR on all stations")
#             baz_object.errors[i] = "Skipped event at "  + str(start_time) + " due to poor SNR on all stations"
#             continue
            
#         # loop through stations to get one trace from each to find earliest arrival
#         l.first_stat = first_observed_arrival(st_event,l)
#         print(l.first_stat)
#         if not l.first_stat:
#             print("Skipped event at "  + str(start_time) + " due to indeterminate first arrival")
#             baz_object.errors[i] = "Skipped event at "  + str(start_time) + " due to indeterminate first arrival"
#             continue
            
#         # loop though stations to perform PCA on all windows in the event on each station's data
#         for s in range(len(l.stations)):
            
#             # check if there's data and skip current station if not
#             if not st_event.select(station=l.stations[s]):
#                 continue
                
#             # compute pca components for all windows in the event
#             pca_first_components = compute_pca(st_event.select(station=l.stations[s]),l)

#             # fill array in results object
#             baz_object.all_first_components[i*l.num_steps:(i+1)*l.num_steps,:,s] = pca_first_components
        
#         # sum results (this is vector sum across stations of pca first components for each window)
#         first_component_sums = np.nansum(baz_object.all_first_components[i*l.num_steps:(i+1)*l.num_steps,:,:],axis=2)

#         # take average weighted by norm of PCA component sums to get single mean event backazimuth
#         norms = np.linalg.norm(first_component_sums,axis=1)
#         denom = np.sum(norms)
#         avg_weighted_x = np.sum(first_component_sums[:,0]*norms)/denom
#         avg_weighted_y = np.sum(first_component_sums[:,1]*norms)/denom
#         if not np.sum(first_component_sums) == 0:
#             event_baz = get_baz([avg_weighted_x,avg_weighted_y])
#         else:
#             event_baz = np.nan
        
#         # rescale norms to get weight vector whose sum equals the number of windows for which we calculated PCA (this is necessary for the sqrt(-2log(R)) in the circular standard deviation calculation)
#         weights = norms/(np.sum(norms)/np.sum([norms != 0]))
#         normalized_first_component_sums = first_component_sums
#         for n in range(len(normalized_first_component_sums)):
#             if not np.sum(first_component_sums[n,0]) == 0:
#                 vect_len = np.sqrt(first_component_sums[n,0]*first_component_sums[n,0]+first_component_sums[n,1]*first_component_sums[n,1])
#                 normalized_first_component_sums[n,0] = first_component_sums[n,0]/vect_len
#                 normalized_first_component_sums[n,1] = first_component_sums[n,1]/vect_len

#         # fill the results object
#         baz_object.norms[i*l.num_steps:(i+1)*l.num_steps] = norms
#         baz_object.indices[i*l.num_steps:(i+1)*l.num_steps] = i
#         baz_object.event_backazimuths[i] = event_baz
#         baz_object.event_uncertainties[i] = circular_stdev(normalized_first_component_sums,weights)*180/np.pi

#         print("BAZ: " + str(event_baz))
#         print("runtime: " + str(time.time()-t))
#         print(circular_stdev(normalized_first_component_sums,weights)*180/np.pi)

