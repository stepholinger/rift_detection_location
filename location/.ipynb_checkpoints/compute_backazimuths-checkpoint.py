import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
from sklearn.decomposition import PCA
from pyproj import Proj,transform,Geod
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

        
        
def get_files(l):
    files = glob.glob(l.data_path + "PIG2/" + "HHZ" + "/*", recursive=True)
    files = [f.replace("PIG2","PIG*") for f in files]
    files = [f.replace("HHZ","HH*") for f in files]
    files.sort()
    start_date = l.detection_times[0].strftime("%Y-%m-%d")
    start_index = [s for s in range(len(files)) if start_date in files[s]][0]
    end_date = l.detection_times[-1].strftime("%Y-%m-%d")
    end_index = [s for s in range(len(files)) if end_date in files[s]][0]+1
    files = files[start_index:end_index]
    return files



def make_results_object(l):
    baz_object = types.SimpleNamespace()
    baz_object.backazimuths = np.empty((l.num_detections),'float64')
    baz_object.backazimuths[:] = np.NaN
    baz_object.uncertainties = np.empty((l.num_detections),'float64')
    baz_object.uncertainties[:] = np.NaN
    return baz_object



def get_detection_times(ds):
    # extract times for each event in the dataset
    detection_times = []
    for event in ds.events:
        detection_times.append(event.origins[0].time.datetime)
    return np.array(detection_times)



def get_detections_today(l):
    current_date = datetime.strptime(l.f.split("/")[9].split(".")[0],"%Y-%m-%d")
    bool_indices = np.logical_and(l.detection_times>=current_date,l.detection_times<current_date + timedelta(days=1))
    detections_today = l.detection_times[bool_indices]
    if sum(bool_indices) == 0:
        indices = [0,0]
    else:
        indices = [[i for i, x in enumerate(bool_indices) if x][0],[i for i, x in enumerate(bool_indices) if x][-1]+1]
    return detections_today, indices



def get_stations_to_use(st,l):
    available_stations = []
    for trace in st:
        available_stations.append(trace.stats.station)
    available_stations = np.unique(available_stations)
    stations_to_use = list(set(l.stations).intersection(set(available_stations)))
    return np.sort(stations_to_use)



def get_data_to_use(st_all,l):
    st = obspy.Stream()
    for s in l.stations:
        st += st_all.select(station=s)
    return st



def get_station_coordinates(l):
    stat_coords = []
    inv = obspy.read_inventory(l.xml_path + "/*")
    for s in l.stations:
        channel = l.network + "." + s + "..HHZ"
        lat = inv.get_coordinates(channel)["latitude"]
        lon = inv.get_coordinates(channel)["longitude"]
        stat_coords.append([lon,lat])
    _, idx = np.unique(stat_coords,axis=0,return_index=True)
    stat_coords = np.array(stat_coords)[np.sort(idx)]
    return stat_coords



def get_station_grid_locations(l):
    # convert station coordinates to x and y and take average station location
    p2 = Proj(l.crs,preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)
    [stat_x,stat_y] = transform(p1,p2,l.station_lon_lat_coords[:,0],l.station_lon_lat_coords[:,1])
    return np.stack((stat_x,stat_y),axis=1)



def get_station_angles(l):
    station_angles = []
    for i in range(len(l.station_grid_coords)):
        x = (l.station_grid_coords[i,0]-l.array_centroid[0])
        y = (l.station_grid_coords[i,1]-l.array_centroid[1])
        angle = np.arctan2(y,x)*180/np.pi
        
        # subtract from 90 since the returned angles are in relation to 0 on the unit circle
        # we want them in relation to true north
        angle = 90-angle
        if angle < 0:
            angle = angle + 360
        
        station_angles.append(angle)
    return station_angles



def first_observed_arrival(st,l):
    # cross correlate all traces and find station with largest shift
    channels = ["HHZ","HHN","HHE"]
    first_stat_vector = []
    for chan in channels:
        st_chan = st.select(channel=chan)
        shifts = np.zeros(len(st_chan))
        corrs = np.zeros(len(st_chan))
        for j in range(len(st_chan)):
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
            # normalize and compute the PCA if staLta criteria is met
            if  np.mean(abs(X)) > l.stalta_threshold*np.mean(abs(horz_data)):
                # find component with max amplitude, normalize both traces by that max value, and compute PCA
                max_amp = np.amax(abs(X))
                X_norm = np.divide(X,max_amp)
                pca = PCA(n_components = 2)
                pca.fit(X_norm)

                # flip pca components based on station of first arrival
                if l.pca_correction == "radial":
                    first_components = correct_pca_radial(pca.components_[0,:],l)
                if l.pca_correction == "distance":
                    first_components = correct_pca_distance(pca.components_[0,:],l)
                if l.pca_correction == "sector":
                    first_components = correct_pca_sector(pca.components_[0,:],l)

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



def correct_pca_radial(pca_components,l):
    # get the backazimuth corresponding to the initial pca first components
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360
        
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

             
    
def correct_pca_distance(pca_components,l):
    # get the backazimuth corresponding to the observed polarization direction
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360

    # get distances to each station from array centroid
    station_distances = np.hypot(l.station_grid_coords[:,0]-l.array_centroid[0],l.station_grid_coords[:,1]-l.array_centroid[1])

    # calculate the distance from the array centroid to each station IN THE DIRECTION OF THE CALCULATED BACKAZIMUTH
    sector_1_stations = []
    sector_1_distances = []
    sector_2_stations = []    
    sector_2_distances = []

    for s in range(len(l.station_angles)):
        if angle_difference(l.station_angles[s],baz) < 90:
            sector_1_stations.append(l.stations[s])
            theta = abs(baz-l.station_angles[s])
            sector_1_distances.append(station_distances[s]*np.cos(theta*np.pi/180))
        if angle_difference(l.station_angles[s],baz) > 90:
            sector_2_stations.append(l.stations[s])
            theta = abs(baz-180-l.station_angles[s])
            sector_2_distances.append(station_distances[s]*np.cos(theta*np.pi/180))

    # now find farthest station from centroid (which would be first to see an incoming plane wave)
    if sector_1_distances:
        sector_1_first_arrival = sector_1_stations[np.argmax(sector_1_distances)]
    else:
        sector_1_first_arrival = []
    if sector_2_distances:
        sector_2_first_arrival = sector_2_stations[np.argmax(sector_2_distances)]
    else:
        sector_2_first_arrival = []
    if l.first_stat == sector_1_first_arrival:
        corrected_pca_components = pca_components 
    if l.first_stat == sector_2_first_arrival:
        corrected_pca_components = pca_components*-1
    else:
        corrected_pca_components = [np.nan,np.nan]
    return corrected_pca_components



def correct_pca_sector(pca_components,l):
    # get the backazimuth corresponding to the observed polarization direction
    baz = 90 - np.arctan2(pca_components[1],pca_components[0])*180/np.pi
    if baz < 0:
        baz = baz + 360
    
    # calculate the distance from the array centroid to each station IN THE DIRECTION OF THE CALCULATED BACKAZIMUTH
    sector_1_stations = []
    sector_2_stations = []    
    for s in range(len(l.station_angles)):
        if angle_difference(l.station_angles[s],baz) < 90:
            sector_1_stations.append(l.stations[s])
        if angle_difference(l.station_angles[s],baz) > 90:
            sector_2_stations.append(l.stations[s])

    if l.first_stat in sector_1_stations:
        corrected_pca_components = pca_components 
    if l.first_stat in sector_2_stations:
        corrected_pca_components = pca_components*-1
    else:
        corrected_pca_components = [np.nan,np.nan]
    return corrected_pca_components



def calculate_event_baz(first_component_sums,norms):
    denom = np.sum(norms)
    avg_weighted_x = np.nansum(first_component_sums[:,0])/denom
    avg_weighted_y = np.nansum(first_component_sums[:,1])/denom
    event_baz = 90 - np.arctan2(avg_weighted_y,avg_weighted_x)*180/np.pi
    if event_baz < 0:
        event_baz = event_baz + 360
    return event_baz



def calculate_uncertainty(first_component_sums,norms):
    # rescale norms to get weight vector whose sum equals the number of windows for which we calculated PCA (this is necessary for the sqrt(-2log(R)) in the circular standard deviation calculation)
    weights = norms/(np.sum(norms)/np.sum([norms != 0]))
    normalized_first_component_sums = first_component_sums
    for n in range(len(normalized_first_component_sums)):
        if not np.sum(first_component_sums[n,0]) == 0:
            vect_len = np.sqrt(first_component_sums[n,0]*first_component_sums[n,0]+first_component_sums[n,1]*first_component_sums[n,1])
            normalized_first_component_sums[n,0] = first_component_sums[n,0]/vect_len
            normalized_first_component_sums[n,1] = first_component_sums[n,1]/vect_len
    event_uncertainty = (circular_stdev(normalized_first_component_sums,weights)*180/np.pi)
    return event_uncertainty



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
    return stdev



def polarization_analysis(l):

    # get detection times from ASDF dataset and insert dummy time at the end for convenience
    detection_times_today, indices = get_detections_today(l)
    num_detections_today = len(detection_times_today)
    detection_times_today = np.append(detection_times_today,datetime(1970,1,1,0,0,0))

    # read all available data for the current day
    st = obspy.read(l.f)
    
    # get stations that are (1) available on current day and (2) within the user-specified list of desired stations
    l.stations = get_stations_to_use(st,l)
    
    # only keep the data from these stations for use in backaziumuth calculations
    st = get_data_to_use(st,l)
    st.filter("bandpass",freqmin=l.freq[0],freqmax=l.freq[1])

    # get geometrical parameters for the functional "array", which is made up of only the stations that are available
    l.station_lon_lat_coords = get_station_coordinates(l)
    l.station_grid_coords = get_station_grid_locations(l)
   
    # update array centroid to reflect subset of stations being used for this particular event or keep it fixed
    if l.centroid == "moving":
        l.array_centroid = np.mean(l.station_grid_coords,axis=0)
        
    # get angles to each station that's desired and available on this particular day)
    l.station_angles = get_station_angles(l)

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
        l.current_detection = detection_utc_time
        
        # check if more than one event on current day; if so, read entire day. If not, just read the event.
        st_event = st.copy()
        st_event.trim(starttime=detection_utc_time,endtime=detection_utc_time+l.trace_len)
        st_event.taper(max_percentage=0.1, max_length=30.)
        
        # check for gaps and remove stations with bad data quality for this event
        start_time = st_event[0].stats.starttime
        if check_trace_length(st_event):
            print("Skipped event at "  + str(start_time) + " due to traces with uneven length\n")
            continue

        # check SNR and if all traces were removed due to poor snr, skip this event
        st_event = check_data_quality(st_event,l)
        if not st_event:
            print("Skipped event at "  + str(start_time) + " due to poor SNR on all stations\n")
            continue
            
        # loop through stations to get one trace from each to find earliest arrival
        l.first_stat = first_observed_arrival(st_event,l)
        if not l.first_stat:
            print("Skipped event at "  + str(start_time) + " due to indeterminate first arrival\n")
            continue
            
        # loop though stations to perform PCA on all windows in the event on each station's data
        for s in range(len(l.stations)):

            # check if there's data and skip current station if not
            if not st_event.select(station=l.stations[s]):
                continue
                
            # compute pca components for all windows in the event
            all_first_components[:,:,s] = compute_pca(st_event.select(station=l.stations[s]),l)
        
        # sum results (this is vector sum across stations of pca first components for each window)
        first_component_sums = np.nansum(all_first_components,axis=2)
        
        # take average weighted by norm of PCA component sums to get single mean event backazimuth
        norms = np.linalg.norm(first_component_sums,axis=1)
        if not np.sum(first_component_sums) == 0:
            event_baz_vect[i] = calculate_event_baz(first_component_sums,norms)
            event_uncertainty_vect[i] = calculate_uncertainty(first_component_sums,norms)
    if num_detections_today == 0:
         print("Finished with " + str(st[0].stats.starttime.date)+" (no detections) \n")   
    else:
        print("Finished with " + str(detection_times_today[0].date())+"\n")
    return event_baz_vect,event_uncertainty_vect,indices



def compute_backazimuths(l): 
    
    # get home directory path
    home_dir = str(pathlib.Path().absolute())
    
    # get centroid for the desired stations
    l.station_lon_lat_coords = get_station_coordinates(l)
    l.station_grid_coords = get_station_grid_locations(l)
    l.array_centroid = np.mean(l.station_grid_coords,axis=0)

    # get all detection times
    l.num_detections = len(l.detection_times)
        
    # make object for storing pca vector sums and storing data to plot
    b = make_results_object(l)

    # write file with parameters for this run
    write_parameters(l)
    
    # make vector of all filenames
    files = get_files(l)
    
    print("Got all files...\n")
    # construct iterable list of detection parameter objects for imap
    inputs = []
    for f in files:
        l.f = f
        inputs.append(copy.deepcopy(l))
    print("Made inputs...\n")        
    # map inputs to polarization_analysis and save as each call finishes
    multiprocessing.freeze_support()
    p = multiprocessing.Pool(processes=l.n_procs)
    for result in p.imap_unordered(polarization_analysis,inputs):
        b.backazimuths[result[2][0]:result[2][1]] = result[0]
        b.uncertainties[result[2][0]:result[2][1]] = result[1]

        # open output file, save result vector, and close output file
        baz_file = open(l.filename+".pickle", "wb")
        pickle.dump(b, baz_file)
        baz_file.close()
        
    return b