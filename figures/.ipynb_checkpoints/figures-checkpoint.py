import obspy
import numpy as np
import pathlib
from pyproj import Proj,transform,Geod
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter
from location.compute_backazimuths import get_station_coordinates
from location.compute_backazimuths import get_station_grid_locations
import geopandas as gpd
import cartopy
import cartopy.crs as ccrs
from shapely import geometry
from collections import namedtuple


def transform_imagery(file,dst_crs):
    with rasterio.open(file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open('data/imagery/' + dst_crs + '.TIF', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            
            
            
def get_station_coordinates(xml_path):
    stat_coords = []
    inv = obspy.read_inventory(xml_path + "/*")
    for s in inv.get_contents()['stations']:
        channel = inv.get_contents()['networks'][0] + "." + s.split(' ')[0].split('.')[1] + "..HHZ"
        lat = inv.get_coordinates(channel)["latitude"]
        lon = inv.get_coordinates(channel)["longitude"]
        stat_coords.append([lon,lat])
    _, idx = np.unique(stat_coords,axis=0,return_index=True)
    stat_coords = np.array(stat_coords)[np.sort(idx)]
    return stat_coords



def get_station_grid_locations(station_lon_lat_coords,crs):
    # convert station coordinates to x and y and take average station location
    p2 = Proj(crs,preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)
    [stat_x,stat_y] = transform(p1,p2,station_lon_lat_coords[:,0],station_lon_lat_coords[:,1])
    return np.stack((stat_x,stat_y),axis=1)            
            
    
    
def plot_backazimuths_on_imagery(backazimuths,array_centroid,station_grid_coords,color_bounds,colors):

    # open LANDSAT imagery file and plot as it is, in EPSG:3031
    original_file = "data/imagery/epsg:3245.TIF"

    sat_imagery = rasterio.open(original_file)
    sat_data = sat_imagery.read(1)

    # Construct figure and axis to plot on
    fig,ax = plt.subplots(figsize=(15,15))
    axes_coords = np.array([0, 0, 1, 1])
    ax_image = fig.add_axes(axes_coords)

    # get corners of imagery extent
    p2 = Proj("EPSG:3245",preserve_units=False)
    p1 = Proj(proj='latlong',preserve_units=False)

    # plot imagery
    bounds = sat_imagery.bounds
    horz_len = bounds[2]-bounds[0]
    vert_len = bounds[3]-bounds[1]
    plot_bounds = [bounds[0]+0.35*horz_len,bounds[2]-0.25*horz_len,bounds[1]+0.25*vert_len,bounds[3]-0.4*vert_len]
    ax_image.imshow(sat_data,cmap='gray',extent=[bounds[0],bounds[2],bounds[1],bounds[3]])


    # define, transform, and plot lat/lon grid
    lat = [-74,-74.5,-75,-75.5]
    lon = [-98,-100,-102,-104]
    x_lab_pos=[]
    y_lab_pos=[]
    line = np.linspace(-110,-90,100)
    for i in lat:
        line_x,line_y = transform(p1,p2,line,np.linspace(i,i,100))
        ax_image.plot(line_x,line_y,linestyle='--',linewidth=2,c='gray',alpha=1)
        y_lab_pos.append(line_y[np.argmin(np.abs(line_x-plot_bounds[0]))])
    line = np.linspace(-80,-70,100)
    for i in lon:
        line_x,line_y = transform(p1,p2,np.linspace(i,i,100),line)
        ax_image.plot(line_x,line_y,linestyle='--',linewidth=2,c='gray',alpha=1)
        x_lab_pos.append(line_x[np.argmin(np.abs(line_y-plot_bounds[2]))])

    # set ticks and labels for lat/lon grid
    ax_image.set_xticks(x_lab_pos)
    ax_image.set_xticklabels(labels=[str(lon[0]) + '$^\circ$',str(lon[1]) + '$^\circ$',str(lon[2]) + '$^\circ$',str(lon[3]) + '$^\circ$'],fontsize=25)
    ax_image.set_xlabel("Longitude",fontsize=25)
    ax_image.set_yticks(y_lab_pos)
    ax_image.set_yticklabels(labels=[str(lat[0]) + '$^\circ$',str(lat[1]) + '$^\circ$',str(lat[2]) + '$^\circ$',str(lat[3]) + '$^\circ$'],fontsize=25)
    ax_image.set_ylabel("Latitude",fontsize=25)
    ax_image.set_xlim([plot_bounds[0],plot_bounds[1]])
    ax_image.set_ylim([plot_bounds[2],plot_bounds[3]])
    ax_image.set_title("Event backazimuths",fontsize=25)
    
    # properly center the polar plot on the array centroid
    x_pos = (array_centroid[0]-plot_bounds[0])/(plot_bounds[1]-plot_bounds[0])
    y_pos = (array_centroid[1]-plot_bounds[2])/(plot_bounds[3]-plot_bounds[2])
    width = 0.3

    # make polar plot centered at array centroid
    ax_polar = fig.add_axes([x_pos-width/2,y_pos-width/2,width,width], projection = 'polar')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    radius,bins = np.histogram(backazimuths[~np.isnan(backazimuths)]*np.pi/180,bins=np.linspace(0,2*np.pi,37))
    patches = ax_polar.bar(bins[:-1], radius, zorder=1, align='edge', width=np.diff(bins),
                     edgecolor='black', fill=True, linewidth=1,alpha = .5)
    for i in range(len(color_bounds)):
        for j in range(color_bounds[i][0]//10,color_bounds[i][1]//10):
            patches[j].set_facecolor(colors[i])

    # Remove ylabels for area plots (they are mostly obstructive)
    ax_polar.set_yticks([])
    ax_polar.axis('off')

    # plot station locations
    ax_stats = fig.add_axes(axes_coords)
    ax_stats.scatter(station_grid_coords[:,0],station_grid_coords[:,1],marker="^",c='black',s=100)
    ax_stats.set_xlim([plot_bounds[0],plot_bounds[1]])
    ax_stats.set_ylim([plot_bounds[2],plot_bounds[3]])
    ax_stats.axis('off')
    
    # add North arrow
    line_x,line_y = transform(p1,p2,np.linspace(-102.2,-102.2,100),np.linspace(-74.65,-74.6,100))
    ax_stats.plot(line_x,line_y,color='w',linewidth = 5)
    ax_stats.scatter(line_x[-1],line_y[-1],marker=(3,0,4),c='w',s=400)
    ax_stats.text(line_x[-1]-3500,line_y[-1]-2500,"N",color='w',fontsize=25)
    
    # add scale bar
    ax_stats.plot([plot_bounds[0]+(plot_bounds[1]-plot_bounds[0])/2-10000,plot_bounds[0]+(plot_bounds[1]-plot_bounds[0])/2+10000],[plot_bounds[2]+17500,plot_bounds[2]+17500],color='w',linewidth = 5)
    ax_stats.text(plot_bounds[0]+(plot_bounds[1]-plot_bounds[0])/2-4000,plot_bounds[2]+14000,"20 km",color='w',fontsize=25)

    # add inset figure of antarctica
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax_inset = fig.add_axes([0.765,0.7,0.275,0.275],projection = ccrs.SouthPolarStereo())
    ax_inset.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
    geom = geometry.box(minx=-103,maxx=-99,miny=-75.5,maxy=-74.5)
    ax_inset.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='k',facecolor='none', linewidth=1)
    ax_inset.add_feature(cartopy.feature.OCEAN, facecolor='#A8C5DD', edgecolor='none')
    
    plt.savefig("outputs/figures/backazimuths.png",bbox_inches="tight")

    
    
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



def get_stacks(waveforms,detection_dates,correlation_coefficients,cluster,shifts,freq,trace_length):    
    
    # make empty array for storage
    fs = freq[1]*2.1
    aligned_cluster_events = np.zeros((len(waveforms),int(trace_length*fs+1)*3))

    # iterate through all waves in the current cluster
    for w in range(len(waveforms)):

        # get cross correlation results
        shift = shifts[w]
        correlation_coefficient = correlation_coefficients[w]

        # get current event
        trace = waveforms[w]

        # flip polarity if necessary
        if correlation_coefficient < 0:
            trace = trace * -1

        if shift > 0:
            aligned_trace = np.append(np.zeros(abs(int(shift))),trace)
            aligned_trace = aligned_trace[:int(trace_length*fs+1)*3]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

        else:
            aligned_trace = trace[abs(int(shift)):]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

    start_date = detection_dates[0]
    end_date = detection_dates[-1]
    num_days = end_date-start_date

    stacks = []
    for d in range(num_days.days):
        day = start_date + timedelta(days=d)
        waves_today = aligned_cluster_events[detection_dates == day,:]
        #waves_today = np.divide(waves_today,np.nanmax(np.abs(waves_today),axis=1,keepdims=True))
        stack = np.divide(np.nansum(waves_today,axis = 0),np.sum([detection_dates == day]))
        stacks.append(stack)

    return stacks
    
    
    
def plot_events_and_gps(gps_velocity,gps_time_vect,noise_vect,noise_date_vect,detection_times,daily_stacks,colors,color_bounds,labels,backazimuths,plot_type):
    # take care of some timing details we need for plotting
    start_date = detection_times[0].date()
    start_time = datetime(start_date.year,start_date.month,start_date.day)
    end_date = detection_times[-1].date()
    end_time = datetime(end_date.year,end_date.month,end_date.day)+timedelta(days=1)
    binedges = [start_time + timedelta(days=x) for x in range(0,(end_time-start_time).days+1,1)]
    num_days = (end_time-start_time).days                          
    trace_length = len(daily_stacks[0,:])
  
    # normalize the stacks for plotting
    daily_stacks = np.array(daily_stacks)[:,200:620]
    norm_daily_stacks = np.divide(daily_stacks,np.amax(np.abs(daily_stacks),axis=1,keepdims=True))
    norm_daily_stacks = np.transpose(norm_daily_stacks)

    # make plot
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[2,1]},figsize=(15,15))

    # plot event timeseries                       
    for i in range(len(color_bounds)):
        bool_indices = np.logical_and(backazimuths>=color_bounds[i][0],backazimuths<color_bounds[i][1])
        baz_group_times = detection_times[bool_indices]
        if plot_type == "timeseries":
            ax[0].hist(baz_group_times,binedges,facecolor=colors[i],label=labels[i])
        if plot_type == "cumulative":
            events_per_day,_ = np.histogram(baz_group_times,binedges)
            cumulative_event_count = np.cumsum(events_per_day)
            ax[0].plot(binedges[:-1],cumulative_event_count,color=colors[i],label=labels[i])
    if plot_type == "timeseries":
        ax[0].set_ylabel("Events per day")
    if plot_type == "cumulative":
        ax[0].set_ylabel("Cumulative event count")
    ax[0].legend(loc="upper right")
    ax[0].set_title("(a) Event timeseries and GPS ice velocity")
    
    # plot gps ice velocity                         
    ax2 = ax[0].twinx()
    ax2.plot(gps_time_vect,gps_velocity, c = 'k')
    ax2.set_ylabel("Velocity (m/year)           ",loc='top')
    ax2.set_ylim(3700,4025)
    ax2.set_yticks([3850,3900,3950,4000])

    # plot rms noise level                       
    ax3 = ax[0].twinx()
    ax3.spines.right.set_position(("axes",1))
    ax3.set_yticks([0,2e-6,4e-6])
    ax3.set_yticklabels(['0','2','4'],c='grey')
    ax3.set_ylim(0,12e-6)
    ax3.set_ylabel("\n  Noise RMS \n($10^{-6}$ m/s)",loc="bottom",c="grey")
    ax3.plot(noise_date_vect,noise_vect,linewidth="0.75",c='silver')
    
    # plot waveforms and configure labels
    ax[1].imshow(norm_daily_stacks,vmin=-0.25,vmax=0.25,aspect = 'auto',extent=[date2num(start_time),date2num(end_time),0,trace_length],cmap='seismic')
    ax[1].set_yticks([0,trace_length/2,trace_length])
    ax[1].set_yticklabels(['200','100','0'])
    ax[1].set_ylabel("Time (seconds)")
    ax[1].set_xlabel("Date")
    ax[1].set_title("(b) Daily vertical waveform stacks")

    if plot_type == "cumulative":
        plt.savefig("outputs/figures/gps_and_cumulative_event_count.png")
    else:
        plt.savefig("outputs/figures/gps_and_event_timeseries.png")
    #plt.show()