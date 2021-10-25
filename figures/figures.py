import obspy
import numpy as np
import pathlib
from pyproj import Proj,transform
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from rasterio.plot import show
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter



def get_station_coordinates(xml_path):
    station_lon_lat_coords = []
    inv = obspy.read_inventory(xml_path + "/*")
    channels = inv.get_contents()['channels']
    for channel in channels:
        lat = inv.get_coordinates(channel)["latitude"]
        lon = inv.get_coordinates(channel)["longitude"]
        station_lon_lat_coords.append([lon,lat])
    _, idx = np.unique(station_lon_lat_coords,axis=0,return_index=True)
    station_lon_lat_coords = np.array(station_lon_lat_coords)[np.sort(idx)]
    return station_lon_lat_coords



def get_array_centroid(station_grid_coords):
    # convert station coordinates to x and y and take average station location
    avg_stat_x = np.mean(station_grid_coords[:,0])
    avg_stat_y = np.mean(station_grid_coords[:,1])
    return [avg_stat_x,avg_stat_y]



def get_station_locations(station_lon_lat_coords):
    # convert station coordinates to x and y and take average station location
    p2 = Proj("EPSG:3031",preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)
    [stat_x,stat_y] = transform(p1,p2,station_lon_lat_coords[:,0],station_lon_lat_coords[:,1])
    return np.transpose(np.array([stat_x,stat_y]))



def compute_rays(angle):
    ray = np.zeros((1,2),'float64')
    if angle >= 0 and angle < 90:
        ray[0,0] = -1*np.cos(angle*np.pi/180)
        ray[0,1] = np.sin(angle*np.pi/180)
    if angle >= 90 and angle < 180:
        ray[0,0] = np.sin((angle-90)*np.pi/180)
        ray[0,1] = np.cos((angle-90)*np.pi/180)
    if angle >= 180 and angle < 270:
        ray[0,0] = np.cos((angle-180)*np.pi/180)
        ray[0,1] = -1*np.sin((angle-180)*np.pi/180)
    if angle >= 270 and angle < 360:
        ray[0,0] = -1*np.sin((angle-270)*np.pi/180)
        ray[0,1] = -1*np.cos((angle-270)*np.pi/180)
    return ray



def plot_backazimuths_on_imagery(backazimuths,sat_imagery,xml_path):

    # get station locations and array centroids
    station_lon_lat_coords = get_station_coordinates(xml_path)
    station_grid_coords = get_station_locations(station_lon_lat_coords)
    array_centroid = get_array_centroid(station_grid_coords)

    # get corners of imagery extent
    corners = np.array([[sat_imagery.bounds[0],sat_imagery.bounds[1]],   # bottom left
                        [sat_imagery.bounds[0],sat_imagery.bounds[3]],   # top left
                        [sat_imagery.bounds[2],sat_imagery.bounds[1]],   # bottom right
                        [sat_imagery.bounds[2],sat_imagery.bounds[3]]])  # top right
    p2 = Proj("EPSG:3031",preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)
    corners_lon,corners_lat = transform(p2,p1,corners[:,0],corners[:,1])

    # make plots
    fig,ax = plt.subplots(figsize=(10,10))

    # plot imagery
    show(sat_imagery,ax=ax,cmap="gray")

    # define, transform, and plot lat/lon grid
    lat = [-74,-75]
    lon = [-98,-100,-102,-104]
    x_lab_pos=[]
    y_lab_pos=[]
    line = np.linspace(corners_lat[0]+1,corners_lat[2]-1,100)
    for l in lon:
        line_x,line_y = transform(p1,p2,np.linspace(l,l,100),line)
        ax.plot(line_x,line_y,linestyle='--',linewidth=0.25,c='gray',alpha=1)
        y_lab_pos.append(line_y[np.argmin(np.abs(line_x-corners[0,0]))])
    line = np.linspace(corners_lon[0]-2,corners_lon[1]+1,100)
    for l in lat:
        line_x,line_y = transform(p1,p2,line,np.linspace(l,l,100))
        ax.plot(line_x,line_y,linestyle='--',linewidth=0.25,c='gray',alpha=1)
        x_lab_pos.append(line_x[np.argmin(np.abs(line_y-corners[0,1]))])
    ax.set_xlim([corners[0,0],corners[2,0]])
    ax.set_ylim([corners[0,1],corners[1,1]])

    # set ticks and labels for lat/lon grid
    ax.set_xticks(x_lab_pos)
    ax.set_xticklabels(labels=[str(lat[0]) + '$^\circ$',str(lat[1]) + '$^\circ$'])
    ax.set_xlabel("Latitude")
    ax.set_yticks(y_lab_pos)
    ax.set_yticklabels(labels=[str(lon[0]) + '$^\circ$',str(lon[1]) + '$^\circ$',str(lon[2]) + '$^\circ$',str(lon[3]) + '$^\circ$'])
    ax.set_ylabel("Longitude")

    # colors
    k1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    k2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

    # plot station locations
    ax.scatter(station_grid_coords[:,0],station_grid_coords[:,1],marker="^",c='black',zorder=10000)

    # compute histogram of rift backazimuths
    baz_hist, bins = np.histogram(backazimuths,bins=np.linspace(0,360,37))

    #plot all rays in 10-degree bins with length proportional to # of windows in that bin
    rays = np.zeros((36,2),'float64')
    scale = 40000
    max_width = 2*np.pi*scale/36
    max_width = 7.5
    for i in range(36):
        angle = i*10
        rays[i,:] = compute_rays(angle)
        rayLength = baz_hist[i]/max(baz_hist)*scale
        [x,y] = [np.linspace(array_centroid[0],array_centroid[0]+rays[i,0]*rayLength,100),
                np.linspace(array_centroid[1],array_centroid[1]+rays[i,1]*rayLength,100)]
        lwidths=np.linspace(0,max_width,100)*rayLength/scale
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=lwidths,color="#d95f02",alpha=0.35,zorder=1000)
        ax.add_collection(lc)

    plt.show()
    
    
    
def load_waveform(detection_time,s):
    
    # read in data for template
    date_string = detection_time.date().strftime("%Y-%m-%d")
    fname = s.data_path + s.station + "/" + s.channel + "/*" + date_string + "*"
    st = obspy.read(fname)

    # basic preprocessing
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # filter and resample the data to each band
    st.filter("bandpass",freqmin=s.freq[0],freqmax=s.freq[1])
    
    # make a copy so we only have to read once for all detections on a given day
    event = st.copy()

    # trim the data to the time ranges of template for each band
    event.trim(starttime=obspy.UTCDateTime(detection_time),endtime=obspy.UTCDateTime(detection_time) + s.trace_length)

    return st,event

    
    
def get_stacks(s):

    home_dir = str(pathlib.Path().absolute())

    # take care of some timing details we need 
    start_date = s.detection_times[0].date()
    start_time = datetime(start_date.year,start_date.month,start_date.day)
    end_date = s.detection_times[-1].date()    
    end_time = datetime(end_date.year,end_date.month,end_date.day)+timedelta(days=1)
    num_days = (end_time-start_time).days                          
    
    # make output array
    daily_stacks = np.zeros((num_days,s.trace_length*s.fs+1),"float64")
    stack = np.zeros((s.trace_length*s.fs+1),"float64")
    counts = np.zeros(num_days)
    d = 0
    
    for i in range(len(s.detection_times)):
        # only read file if date of event is different than last event
        if i == 0 or s.detection_times[i].date != st[0].stats.starttime.date:
            st,event_st = load_waveform(s.detection_times[i],s)
        else:
            event_st = st.copy()
            event_st.trim(starttime=s.detection_times[i],endtime=s.detection_times[i] + s.trace_length)

        # get vector of data from stream
        try:
            trace = event_st[0].data
        except:
            continue
            
        # get cross correlation results
        shift = s.shifts[i]
        correlation_coefficient = s.correlation_coefficients[i]
        
        # flip polarity if necessary
        if correlation_coefficient < 0:
            trace = trace * -1

        if shift > 0:
            aligned_trace = np.append(np.zeros(abs(int(shift))),trace)
            aligned_trace = aligned_trace[:s.trace_length*s.fs]

        else:
            aligned_trace = np.append(trace[abs(int(shift)):],np.zeros(abs(int(shift))))
        
        # check if event is on current day
        if s.detection_times[i].date() == s.detection_times[0].date() + timedelta(days=d):

            # sum the normalized waveforms from current day
            stack[:len(aligned_trace)] = stack[:len(aligned_trace)] + aligned_trace
            counts[d] += 1

        # no more events on current day, or last event in the catalog
        if s.detection_times[i].date() != s.detection_times[0].date() + timedelta(days=d) or i == len(s.detection_times)-1:

            # save stack from current day
            daily_stacks[d,:] = stack

            # update day count and reset stack
            stack = np.zeros((s.trace_length*s.fs+1),"float64")
            stack[:len(aligned_trace)] = stack[:len(aligned_trace)] + aligned_trace
            counts[d] += 1
            d += 1

            # give output
            print(str(d)+"/"+str(num_days)+" days of events stacked...")
        
    # divide daily stacks by number of events per day to get daily average waveforms
    daily_stacks = np.divide(daily_stacks,counts[:,None])
        
    return daily_stacks

  

def plot_events_and_gps(gps_velocity,gps_time_vect,detection_times,daily_stacks):
    
    # take care of some timing details we need for plotting≠
    start_date = detection_times[0].date()
    start_time = datetime(start_date.year,start_date.month,start_date.day)
    end_date = detection_times[-1].date()
    end_time = datetime(end_date.year,end_date.month,end_date.day)+timedelta(days=1)
    binedges = [start_time + timedelta(days=x) for x in range(0,(end_time-start_time).days+1,1)]
    num_days = (end_time-start_time).days                          
    trace_length = len(daily_stacks[0,:])
    
    # make empty array for storage
  #  aligned_events = np.zeros((len(waveforms),len(waveforms[0,:])))
  #  trace_len = len(waveforms[0,:])
    
    # iterate through all waves and align them
  #  for w in range(len(waveforms)):

        # get cross correlation results
  #      shift = shifts[w]
  #      correlation_coefficient = correlation_coefficients[w]
        
        # get current event
 #       trace = waveforms[w]

        # flip polarity if necessary
  #      if correlation_coefficient < 0:
 #           trace = trace * -1

  #      if shift > 0:
  #          aligned_trace = np.append(np.zeros(abs(int(shift))),trace)
 #           aligned_trace = aligned_trace[:trace_len]
  #          aligned_events[w,:len(aligned_trace)] = aligned_trace

 #       else:
 #           aligned_trace = trace[abs(int(shift)):]
  #          aligned_events[w,:len(aligned_trace)] = aligned_trace

    # make some containers for daily waveform stacks
  #  daily_stacks = np.zeros((num_days,int(trace_len/3)),"float64")
 #   stack = np.zeros((1,int(trace_len/3)),"float64")
   # counts = np.zeros(num_days)
  #  d = 0
    
    # iterate through list of detection times and stack events on the same day                       
    #for i in range(len(detection_times)):

        # check if event is on current day
   #     if detection_times[i].date() == detection_times[0].date() + timedelta(days=d):

            # sum the normalized waveforms from current day
  #          stack = stack + aligned_events[i,:int(trace_len/3)]

            # count number of events on current day
 #           counts[d] += 1

        # no more events on current day
#        else:

            # save stack from current day
 #           daily_stacks[d,:] = stack

            # update day count and reset stack
 #           stack = np.zeros((1,int(trace_len/3)),"float64")
#            d += 1

            # count current time and add to new day's stack
#            stack = stack + stack + aligned_events[i,:int(trace_len/3)]
#            counts[d] += 1

    # divide daily stacks by number of events per day to get daily average waveforms
#    daily_stacks = np.divide(daily_stacks,counts[:,None])
    norm_daily_stacks = np.divide(daily_stacks,np.amax(np.abs(daily_stacks),axis=1,keepdims=True))
    norm_daily_stacks = np.transpose(norm_daily_stacks)

    # make plot
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[2,1]},figsize=(15,15))

    # plot evernt timeseries                       
    ax[0].hist(detection_times,binedges)
    ax[0].set_ylabel("Detection count")

    # plot gps ice velocity                         
    ax2 = ax[0].twinx()
    ax2.plot(gps_time_vect,gps_velocity, c = 'k')
    ax2.set_ylabel("Velocity (m/year)           ",loc='top')
    ax2.set_ylim(3700,4025)
    ax2.set_yticks([3850,3900,3950,4000])

    # plot rms noise level                       
    #    ax3 = ax[0].twinx()
    #    ax3.spines.right.set_position(("axes",1))
    #    ax3.set_yticks([0,2e-6,4e-6])
    #    ax3.set_yticklabels(['0','2','4'],c='grey')
    #    ax3.set_ylim(0,12e-6)
    #    ax3.set_ylabel("\n  Noise RMS \n($10^{-6}$ m/s)",loc="bottom",c="grey")
    ax[0].set_title("(a) Event timeseries and GPS ice velocity")

    # plot waveforms and configure labels
    ax[1].imshow(norm_daily_stacks[:,100:300],vmin=-0.25,vmax=0.25,aspect = 'auto',extent=[date2num(start_time),date2num(end_time),0,trace_length],cmap='seismic')
    ax[1].set_yticks([0,trace_length/2,trace_length])
    ax[1].set_yticklabels(['200','100','0'])
    ax[1].set_ylabel("Time (seconds)")
    ax[1].set_xlabel("Date")
    ax[1].set_title("(b) Daily vertical waveform stacks")

    
    
    plt.show()