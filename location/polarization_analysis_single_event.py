import obspy
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter
from matplotlib.dates import num2date
from polarization_utils import readEvent
from polarization_utils import predict_first_arrival
from polarization_utils import observed_first_arrival
from polarization_utils import correct_polarization
from polarization_utils import compute_baz
from polarization_utils import compute_pca
from polarization_utils import compute_rays
from pyproj import Proj,transform
import rasterio
from rasterio.plot import show
from matplotlib import cm

# set path
dataPath = "/media/Data/Data/PIG/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
figPath = "/home/setholinger/Documents/Projects/PIG/location/polarization/"

# set data paramters
fs = 100

# set threshold parameters
norm_thresh = 2.75

# set windowing parameters
snipLen = 3600
winLen = 10
slide = 5
numSteps = int((snipLen-winLen)/slide)

# set stations and components
chans = ["HHN","HHE","HHZ"]
stats = ["PIG2","PIG4","PIG5"]
#stat_coords = np.array([[-100.748596,-75.016701],[-100.786598,-75.010696],[-100.730904,-75.009201],[-100.723701,-75.020302],[-100.802696,-75.020103]])
stat_coords = np.array([[-100.786598,-75.010696],[-100.723701,-75.020302],[-100.802696,-75.020103]])

# read imagery data, get coordinate system, convert station coordinates to x and y, and take average station location
file = "/media/Data/Data/PIG/TIF/LC08_L1GT_001113_20131012_20170429_01_T2_B4.TIF"
sat_data = rasterio.open(file)
p2 = Proj(sat_data.crs,preserve_units=False)
p1 = Proj(proj='latlong',preserve_units=False)
[stat_x,stat_y] = transform(p1,p2,stat_coords[:,0],stat_coords[:,1])
avg_stat_x = np.mean(stat_x)
avg_stat_y = np.mean(stat_y)

# set frequency
freq = [0.001,1]

# get times bounds for current event
eventLims = [obspy.UTCDateTime(2012,5,9,18,00,00),obspy.UTCDateTime(2012,5,9,19,00,00)]

# make arrays for storing pca vector sums and pca components
all_first_components = np.zeros((numSteps),"float64")
first_component_sums = np.zeros((numSteps,2),"float64")

# make empty obspy stream for storing one trace from each stations
event_stat = obspy.read()
event_stat.clear()

# loop through stations to get one trace from each to find earliest arrival
for stat in range(len(stats)):

    # get times bounds for current event and read event
    event_stat += readEvent(dataPath + "MSEED/noIR/",stats[stat],chans[1],eventLims,freq)

# find station with earliest arrival
first_stat = observed_first_arrival(event_stat)

# loop though stations to perform PCA on all windows in the event on each station's data
for stat in range(len(stats)):

    # compute pca components for all windows in the event
    first_components = compute_pca(dataPath,stats[stat],chans,fs,winLen,slide,numSteps,freq,eventLims)

    # correct polarization direction based on first arrival
    first_components_corrected = correct_polarization(first_components,stats,first_stat,avg_stat_x,avg_stat_y,stat_x,stat_y)

    # sum results (this is vector sum across stations of pca first components for each window)
    first_component_sums = first_component_sums + first_components_corrected

# fill results vector
all_first_components = first_component_sums

# make plots
fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8, 8),gridspec_kw={'height_ratios':[1,0.4]})

# do some stuff
corners = np.array([[sat_data.bounds[0],sat_data.bounds[1]],   # bottom left
            [sat_data.bounds[0],sat_data.bounds[3]],   # top left
            [sat_data.bounds[2],sat_data.bounds[1]],   # bottom right
            [sat_data.bounds[2],sat_data.bounds[3]]])  # top right
corners_lon,corners_lat = transform(p2,p1,corners[:,0],corners[:,1])

# plot imagery
show(sat_data,ax=ax[0],cmap="gray")

# handle axes
plt.suptitle("May 9 Event Polarizations")

# make array for storage
back_azimuths = np.empty((0,1),"float64")

# plot pca compoments that exceed norm threshold
# be wary- the transformed coordinate system's x-axis is meters north and the y-axis is meters east, so the pca_first_component[~,0] (which is cartesian x) is in [L] east
# and therefore along the transformed y-axis and the pca_first_component[~,1] (which is cartesian y) is in [L] north and therefore along the transformed x-axis
count = 0
for s in range(len(all_first_components)):

    # only plot and save results if length of resultant vector has a norm exceeding the threshold
    if np.linalg.norm(all_first_components[s,:]) > norm_thresh:

        # calculate back azimuths and save in array
        baz = compute_baz(all_first_components[s,:])
        back_azimuths = np.vstack((back_azimuths,baz))
        count += 1

# compute histogram of back azimuths
baz_hist,edges = np.histogram(back_azimuths,bins=np.linspace(0,360,37))

# set up colormap
colors = [ cm.plasma(x) for x in np.linspace(0,1,max(baz_hist)+1)]

# plot all rays in 10-degree bins with length proportional to # of windows in that bin
rays = np.zeros((36,2),'float64')
scale = 40000
max_width = 2*np.pi*scale/36
max_width = 8
for i in range(36):
    angle = i*10
    rays[i,:] = compute_rays(angle)
    rayLength = baz_hist[i]/max(baz_hist)*scale
    [x,y] = [np.linspace(avg_stat_x,avg_stat_x+rays[i,0]*rayLength,100),
             np.linspace(avg_stat_y,avg_stat_y+rays[i,1]*rayLength,100)]
    lwidths=np.linspace(0,max_width,100)*rayLength/scale
    print(lwidths)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    print(segments)
    lc = LineCollection(segments, linewidths=lwidths,color='maroon',alpha=0.5)
    ax[0].add_collection(lc)

# define, transform, and plot lat/lon grid
lat = [-74,-75]
lon = [-98,-100,-102,-104]
x_lab_pos=[]
y_lab_pos=[]
line = np.linspace(corners_lat[0]+1,corners_lat[2]-1,100)
for l in lon:
    line_x,line_y = transform(p1,p2,np.linspace(l,l,100),line)
    ax[0].plot(line_x,line_y,linestyle='--',linewidth=1,c='gray',alpha=1)
    y_lab_pos.append(line_y[np.argmin(np.abs(line_x-corners[0,0]))])
line = np.linspace(corners_lon[0]-2,corners_lon[1]+1,100)
for l in lat:
    line_x,line_y = transform(p1,p2,line,np.linspace(l,l,100))
    ax[0].plot(line_x,line_y,linestyle='--',linewidth=1,c='gray',alpha=1)
    x_lab_pos.append(line_x[np.argmin(np.abs(line_y-corners[0,1]))])
ax[0].set_xlim([corners[0,0],corners[2,0]])
ax[0].set_ylim([corners[0,1],corners[1,1]])

ax[0].set_xticks(x_lab_pos)
ax[0].set_xticklabels(labels=[str(lat[0]) + '$^\circ$',str(lat[1]) + '$^\circ$'])
ax[0].set_xlabel("Latitude")
ax[0].set_yticks(y_lab_pos)
ax[0].set_yticklabels(labels=[str(lon[0]) + '$^\circ$',str(lon[1]) + '$^\circ$',str(lon[2]) + '$^\circ$',str(lon[3]) + '$^\circ$'])
ax[0].set_ylabel("Longitude")

# colors
k1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
k2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

# plot ice front
front_x = [-1.644e6,-1.64e6,-1.638e6,-1.626e6,-1.611e6,-1.6095e6,-1.6055e6,-1.6038e6,-1.598e6,-1.6005e6,-1.6e6,-1.595e6]
front_y = [-3.34e5,-3.33e5,-3.44e5,-3.445e5,-3.475e5,-3.43e5,-3.4e5,-3.413e5,-3.356e5,-3.32e5,-3.289e5,-3.29e5]
ax[0].plot(front_x, front_y,c=k1,zorder=len(all_first_components)*10)

# plot rift
rift1_x = [-1.63e6,-1.6233e6,-1.6132e6,-1.6027e6]
rift1_y = [-3.255e5,-3.237e5,-3.236e5,-3.281e5]
ax[0].plot(rift1_x,rift1_y,c=k2,zorder=len(all_first_components)*10)
rift2_x = [-1.63e6,-1.6232e6,-1.6132e6]
rift2_y = [-3.28e5,-3.2706e5,-3.236e5]
ax[0].plot(rift2_x,rift2_y,c=k2,zorder=len(all_first_components)*10)

# plot station locations
ax[0].scatter(stat_x,stat_y,marker="^",c='black',zorder=count*10)

# add north arrow
ax[0].arrow(avg_stat_x-65000,avg_stat_y+70000,-10000,0,width = 500,head_width=3000,head_length=3000,fc="k", ec="k",zorder=len(all_first_components)*10)
ax[0].text(avg_stat_x-74000,avg_stat_y+73000,"N",size="large",zorder=len(all_first_components)*10)

# add distance scale
ax[0].plot(np.linspace(avg_stat_x-60000,avg_stat_x-80000,10),np.ones(10)*avg_stat_y-30000,c="k",zorder=len(all_first_components)*10)
ax[0].text(avg_stat_x-82000,avg_stat_y-26000,"20 km",size="medium")

# plot event waveform
event_Z = readEvent(dataPath + "MSEED/noIR/","PIG2","HHZ",eventLims,freq)
start = pd.Timestamp(event_Z[0].stats.starttime.isoformat())
end = pd.Timestamp(event_Z[0].stats.endtime.isoformat())
t = np.linspace(start.value, end.value,event_Z[0].stats.npts)
t = pd.to_datetime(t)
ax[1].plot(t,event_Z[0].data)
ax[1].set_ylabel("Velocity (m/s)")
ax[1].set_xlabel("Time")
a = ax[1].get_xticks().tolist()
a[0] = str(num2date(a[0])).split('+')[0].split(' ')[0] + "\n" + str(num2date(a[0])).split('+')[0].split(' ')[1][:-3]
for l in range(len(a)-1):
    a[l+1] = str(num2date(a[l+1])).split(' ')[1].split('+')[0][:-3]
ax[1].set_xticklabels(a)
ax[1].grid(linestyle=":")
ax[1].set_xlim([t[0],t[-1]])

#plt.show()
#plt.close()
plt.savefig(figPath + "win_len_" + str(winLen) + "/norm>" + str(norm_thresh) + "/may_9_event_polarizations_" + str(freq[0]) + "-" + str(freq[1]) + "Hz.png",dpi=400)
