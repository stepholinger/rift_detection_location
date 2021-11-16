import obspy
import numpy as np
import pathlib
from pyproj import Proj,transform,Geod
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import date2num, DateFormatter
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
    c_map = [np.min(sat_data),np.max(sat_data)]
    c_map_range = c_map[1]-c_map[0]
    ax_image.imshow(sat_data,cmap='gray',extent=[bounds[0],bounds[2],bounds[1],bounds[3]],interpolation='none', vmin=np.min(sat_data), vmax=np.max(sat_data)-0.2*c_map_range)
    #ax_image.imshow(sat_data,cmap='gray',extent=[bounds[0],bounds[2],bounds[1],bounds[3]])


    # define, transform, and plot lat/lon grid
    lat = [-74,-74.5,-75,-75.5]
    lon = [-98,-100,-102,-104]
    x_lab_pos=[]
    y_lab_pos=[]
    line = np.linspace(-110,-90,100)
    for i in lat:
        line_x,line_y = transform(p1,p2,line,np.linspace(i,i,100))
        ax_image.plot(line_x,line_y,linestyle='--',linewidth=2,c='w',alpha=0.75)
        y_lab_pos.append(line_y[np.argmin(np.abs(line_x-plot_bounds[0]))])
    line = np.linspace(-80,-70,100)
    for i in lon:
        line_x,line_y = transform(p1,p2,np.linspace(i,i,100),line)
        ax_image.plot(line_x,line_y,linestyle='--',linewidth=2,c='w',alpha=0.75)
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
    #ax_image.set_title("Event Backazimuths",fontsize=25)
    
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
    
    # add legend for spatial groups
    x,y = transform(p1,p2,-102.75,-75.175)
    box = matplotlib.patches.Rectangle((x-1000, y-7000), 27000, 10000, linewidth=1, edgecolor='k', facecolor='w')
    ax_stats.add_patch(box)
    ax_stats.text(x, y, "Rift tip",c="#d95f02",fontsize=25)
    ax_stats.text(x, y-3000, "Rift/margin",c="#1b9e77",fontsize=25)
    ax_stats.text(x, y-6000, "Northeast margin",c="#7570b3",fontsize=25)
   
    # add North arrow
    line_x,line_y = transform(p1,p2,np.linspace(-102.2,-102.2,100),np.linspace(-74.65,-74.6,100))
    ax_stats.plot(line_x,line_y,color='w',linewidth = 5)
    ax_stats.scatter(line_x[-1],line_y[-1],marker=(3,0,4),c='w',s=400)
    ax_stats.text(line_x[-1]-3500,line_y[-1]-2500,"N",color='w',fontsize=25)
    
    # add scale bar
    ax_stats.plot([plot_bounds[0]+(plot_bounds[1]-plot_bounds[0])/2-5000,plot_bounds[0]+(plot_bounds[1]-plot_bounds[0])/2+15000],[plot_bounds[2]+12000,plot_bounds[2]+12000],color='w',linewidth = 5)
    ax_stats.text(plot_bounds[0]+(plot_bounds[1]-plot_bounds[0])/2,plot_bounds[2]+9000,"20 km",color='w',fontsize=25)

    # add inset figure of antarctica
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    ax_inset = fig.add_axes([0.765,0.7,0.275,0.275],projection = ccrs.SouthPolarStereo())
    ax_inset.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
    geom = geometry.box(minx=-103,maxx=-99,miny=-75.5,maxy=-74.5)
    ax_inset.add_geometries([geom], crs=ccrs.PlateCarree(), edgecolor='r',facecolor='none', linewidth=1)
    ax_inset.add_feature(cartopy.feature.OCEAN, facecolor='#A8C5DD', edgecolor='none')
    
    plt.savefig("outputs/figures/backazimuths.png",bbox_inches="tight")

    
    
def load_waveform(r,detection_time):
    
    # read in data for template
    date_string = detection_time.date.strftime("%Y-%m-%d")
    fname = r.data_path + r.station + "/HH*/*" + date_string + "*"
    st = obspy.read(fname)

    # basic preprocessing
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # filter and resample the data to each band
    st.filter("bandpass",freqmin=r.freq[0],freqmax=r.freq[1])
    st.resample(r.freq[1]*2.1)
    
    # make a copy so we only have to read once for all detections on a given day
    event = st.copy()

    # trim the data to the time ranges of template for each band
    event.trim(starttime=detection_time,endtime=detection_time + r.trace_length)

    return st,event



def get_3D_trace(r,event):
    fs = r.freq[1]*2.1
    waveform = np.zeros((3*int(r.trace_length*fs+1)))
    for i in range(len(event)):
        trace = event.select(component=r.component_order[i])[0].data
        waveform[i*int(r.trace_length*fs+1):i*int(r.trace_length*fs+1)+len(trace)] = trace
    return waveform



def get_rotated_waveforms(r):

    # make output array
    fs = r.freq[1]*2.1
    waveform_matrix = np.zeros((len(r.detection_times),3*int(r.trace_length*fs+1)))
    
    for i in range(len(r.detection_times)):
        # only read file if date of event is different than last event
        if i == 0 or r.detection_times[i].date != st[0].stats.starttime.date:
            st,event_st = load_waveform(r,r.detection_times[i])
        else:
            event_st = st.copy()
            event_st.trim(starttime=r.detection_times[i],endtime=r.detection_times[i] + r.trace_length)

        # rotate the waveform
        try:
            event_st.rotate('NE->RT',back_azimuth=r.backazimuths[i])
            waveform = get_3D_trace(r,event_st)
            waveform_matrix[i,:] = waveform   
        except:
            waveform_matrix[i,:] = np.zeros((3*int(r.trace_length*fs+1)))
        # give output
        print(str(i)+"/"+str(len(waveform_matrix))+" event waveforms retrieved...")
    return waveform_matrix



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

    daily_stacks = []
    for d in range(num_days.days):
        day = start_date + timedelta(days=d)
        waves_today = aligned_cluster_events[detection_dates == day,:]
        daily_stack = np.divide(np.nansum(waves_today,axis = 0),np.sum([detection_dates == day]))
        daily_stacks.append(daily_stack)
    stack = np.nanmean(aligned_cluster_events,axis=0)
    return daily_stacks,stack
    
    
    
def plot_daily_events_and_gps(gps_velocity,gps_time_vect,noise_vect,noise_date_vect,detection_times,daily_stacks,stacks,colors,baz_bounds,backazimuths):
    # take care of some timing details we need for plotting
    start_date = detection_times[0].date()
    start_time = datetime(start_date.year,start_date.month,start_date.day)
    end_date = detection_times[-1].date()
    end_time = datetime(end_date.year,end_date.month,end_date.day)+timedelta(days=1)
    binedges = [start_time + timedelta(days=x) for x in range(0,(end_time-start_time).days+1,1)]
    num_days = (end_time-start_time).days                          
    trace_length = len(daily_stacks[0][0,:])

    # make plot
    fig = plt.figure(figsize=(15,15))
    gs = fig.add_gridspec(nrows=10,ncols=2, hspace=0,height_ratios=[2,0.5,1,1,0.5,1,1,0.5,1,1],width_ratios=[1,0.1])
    ax = gs.subplots(sharex=False,sharey=False)
    
    # plot gps ice velocity
    ax[0,0].plot(gps_time_vect,gps_velocity, c = 'k')
    ax[0,0].set_ylabel("Velocity (m/year)")
    ax[0,0].set_ylim(3800,4050)
    ax[0,0].set_yticks([3850,3900,3950,4000])
    ax[0,0].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0,0].set_xlim(gps_time_vect[0],gps_time_vect[-1])
    ax[0,0].set_title("a",loc="left",fontsize=25)

    # plot rms noise level                       
    ax_twin = ax[0,0].twinx()
    ax_twin.spines.right.set_position(("axes",1))
    ax_twin.set_yticks([0,2e-6,4e-6,6e-6,8e-6])
    ax_twin.set_yticklabels(['0','2','4','6','8'],c='grey')
    ax_twin.set_ylim(0,12e-6)
    ax_twin.set_ylabel("\n  Noise RMS \n($10^{-6}$ m/s)",c="grey")
    ax_twin.plot(noise_date_vect,noise_vect,linewidth="0.75",c='silver')

    # make spacing subplot invisible
    ax[1,0].set_visible(0)
    ax[0,1].set_visible(0)
    ax[1,1].set_visible(0)

    # plot event timeseries
    ax_ind = [2,5,8]
    letters = ['b','c','d']
    for i in range(len(baz_bounds)):
        bool_indices = np.logical_and(backazimuths>=baz_bounds[i][0],backazimuths<baz_bounds[i][1])
        baz_group_times = detection_times[bool_indices]
        ax[ax_ind[i],0].hist(baz_group_times,binedges,facecolor=colors[i])
        ax[ax_ind[i],0].set_ylabel("Events per day")
        ax[ax_ind[i],0].set_ylim(0,40)
        ax[ax_ind[i],0].spines['right'].set_visible(False)
        ax[ax_ind[i],0].spines['top'].set_visible(False)    
        ax[ax_ind[i],0].set_title(letters[i],loc="left",fontsize=25)

        # normalize the stacks for plotting
        trim_daily_stacks = np.array(daily_stacks[i])[:,200:515]
        norm_daily_stacks = np.divide(trim_daily_stacks,np.amax(np.abs(trim_daily_stacks),axis=1,keepdims=True))
        norm_daily_stacks = np.transpose(norm_daily_stacks)
        
        # plot waveforms and configure labels
        ax[ax_ind[i]+1,0].imshow(norm_daily_stacks,vmin=-0.25,vmax=0.25,aspect = 'auto',extent=[date2num(start_time),date2num(end_time),0,trace_length],cmap='seismic')
        ax[ax_ind[i]+1,0].set_yticks([0,trace_length/2,trace_length])
        ax[ax_ind[i]+1,0].set_yticklabels(['150','75','0'])
        ax[ax_ind[i]+1,0].set_ylabel("Time (s)")
        ax[ax_ind[i]+1,0].xaxis.set_tick_params(which='both', labelbottom=True)
        ax[ax_ind[i]+1,0].xaxis_date()

        # plot individual overall stacks on right axis
        ax[ax_ind[i]+1,1].plot(stacks[i][200:515],np.flip(np.arange(315)))
        box = ax[ax_ind[i]+1,0].get_position()
        box.x0 = box.x0 + 0.65
        box.x1 = box.x0 + 0.05
        ax[ax_ind[i]+1,1].set_position(box)
        ax[ax_ind[i]+1,1].axis('off')

        # make spacing subplots invisible
        ax[ax_ind[i],1].set_visible(0)
        if ax_ind[i]<8:
            ax[ax_ind[i]+2,1].set_visible(0)
            ax[ax_ind[i]+2,0].set_visible(0)
            
    ax[9,0].set_xlabel("Date")

    plt.savefig("outputs/figures/gps_and_daily_event_timeseries.png")
    #plt.show()
    
    
    
def plot_weekly_events_and_gps(gps_speed,gps_time_vect,noise_vect,noise_date_vect,detection_times,daily_stacks,stacks,colors,baz_bounds,backazimuths):
    # take care of some timing details we need for plotting
    start_date = detection_times[0].date()
    start_time = datetime(start_date.year,start_date.month,start_date.day)
    end_date = detection_times[-1].date()
    end_time = datetime(end_date.year,end_date.month,end_date.day)+timedelta(days=1)
    binedges = [start_time + timedelta(days=x) for x in range(0,(end_time-start_time).days+1,7)]
    num_days = (end_time-start_time).days                          
    trace_length = len(daily_stacks[0][0,:])

    # make plot
    fig = plt.figure(figsize=(15,15))
    gs = fig.add_gridspec(nrows=10,ncols=2, hspace=0,height_ratios=[2,0.6,1,1,0.6,1,1,0.6,1,1],width_ratios=[1,0.1])
    ax = gs.subplots(sharex=False,sharey=False)
    
    # plot gps ice velocity
    ax[0,0].plot(gps_time_vect,gps_speed, c = 'k')
    ax[0,0].set_ylabel("Speed (m/year)",fontsize=15)
    ax[0,0].set_ylim(3800,4050)
    ax[0,0].set_yticks([3850,3900,3950,4000])
    ax[0,0].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0,0].set_xlim(gps_time_vect[0],gps_time_vect[-1])
    ax[0,0].set_title("a. GPS ice speed and timeseries of noise RMS",loc="left",fontsize=15)

    # plot rms noise level                       
    ax_twin = ax[0,0].twinx()
    ax_twin.spines.right.set_position(("axes",1))
    ax_twin.set_yticks([0,2e-6,4e-6,6e-6,8e-6])
    ax_twin.set_yticklabels(['0','2','4','6','8'])
    ax_twin.set_ylim(0,12e-6)
    ax_twin.set_ylabel("\n  Noise RMS \n($10^{-6}$ m/s)",fontsize=15)
    ax_twin.bar(noise_date_vect,noise_vect,width=7,facecolor='silver',edgecolor='k')
    
    # add lines for important days and label them
    ax_twin.vlines([datetime(2012,5,9),datetime(2013,11,7)],0,120,colors=['dimgray','dimgray'],linestyles='dashed')
    ax_twin.text(datetime(2012,5,15),3e-6,"May 9 riftquake")
    ax_twin.text(datetime(2013,8,25),3e-6,"Iceberg B-31 \ncalves")
    
    # make spacing subplot invisible
    ax[1,0].set_visible(0)
    ax[0,1].set_visible(0)
    ax[1,1].set_visible(0)

    # plot event timeseries
    ax_ind = [2,5,8]
    letters = ['b. Timeseries and waveforms of rift tip events','c. Timeseries and waveforms of rift/margin events','d. Timeseries and waveforms of northeast margin events']
    for i in range(len(baz_bounds)):
        bool_indices = np.logical_and(backazimuths>=baz_bounds[i][0],backazimuths<baz_bounds[i][1])
        baz_group_times = detection_times[bool_indices]
        ax[ax_ind[i],0].hist(baz_group_times,binedges,facecolor=colors[i],edgecolor='k')
        ax[ax_ind[i],0].set_xlim(gps_time_vect[0],gps_time_vect[-1])
        ax[ax_ind[i],0].set_ylabel("Events",fontsize=15)
        ax[ax_ind[i],0].set_ylim(0,120)
        ax[ax_ind[i],0].set_title(letters[i],loc="left",fontsize=15)
        ax[ax_ind[i],0].vlines([datetime(2012,5,9),datetime(2013,11,7)],0,120,colors=['dimgray','dimgray'],linestyles='dashed')
        
        # normalize the stacks for plotting
        trim_daily_stacks = np.array(daily_stacks[i])[:,0,200:515]
        norm_daily_stacks = np.divide(trim_daily_stacks,np.amax(np.abs(trim_daily_stacks),axis=1,keepdims=True))
        norm_daily_stacks = np.transpose(norm_daily_stacks)
        
        # plot waveforms and configure labels
        ax[ax_ind[i]+1,0].imshow(norm_daily_stacks,vmin=-0.5,vmax=0.5,aspect = 'auto',extent=[date2num(start_time),date2num(end_time),0,trace_length],cmap='seismic')
        ax[ax_ind[i]+1,0].set_yticks([0,trace_length/2,trace_length])
        ax[ax_ind[i]+1,0].set_yticklabels(['150','75','0'])
        ax[ax_ind[i]+1,0].set_ylabel("Time (s)",fontsize=15)
        ax[ax_ind[i]+1,0].xaxis.set_tick_params(which='both', labelbottom=True)
        ax[ax_ind[i]+1,0].xaxis_date()

        # plot individual overall stacks on right axis
        ax[ax_ind[i]+1,1].plot(stacks[i][0,200:515],np.flip(np.arange(315)),c='k')
        box = ax[ax_ind[i]+1,0].get_position()
        box.x0 = box.x0 + 0.65
        box.x1 = box.x0 + 0.05
        ax[ax_ind[i]+1,1].set_position(box)
        ax[ax_ind[i]+1,1].axis('off')

        # make spacing subplots invisible
        ax[ax_ind[i],1].set_visible(0)
        if ax_ind[i]<8:
            ax[ax_ind[i]+2,1].set_visible(0)
            ax[ax_ind[i]+2,0].set_visible(0)
            
    ax[9,0].set_xlabel("Date",fontsize=15)

    plt.savefig("outputs/figures/gps_and_weekly_event_timeseries.png")
    #plt.show()