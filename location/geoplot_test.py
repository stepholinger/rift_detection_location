import matplotlib.pyplot as plt
import numpy as np
from pyproj import Proj,transform
import rasterio
from rasterio.plot import show
from polarization_utils import compute_baz
from matplotlib import cm

# make plots
fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8, 8),gridspec_kw={'height_ratios':[1,0.4]})
file = "/media/Data/Data/PIG/TIF/LC08_L1GT_001113_20131012_20170429_01_T2_B4.TIF"
sat_data = rasterio.open(file)

# Convert our dataset from lat-lon to projected coordinates, then plot
#stat_coords = np.array([[-100.748596,-75.016701],[-100.786598,-75.010696],[-100.730904,-75.009201],[-100.723701,-75.020302],[-100.802696,-75.020103]])
stat_coords = np.array([[-100.786598,-75.010696],[-100.723701,-75.020302],[-100.802696,-75.020103]])
p2 = Proj(sat_data.crs,preserve_units=False)
p1 = Proj(proj='latlong',preserve_units=False)
x1,y1 = p1(stat_coords[:,0],stat_coords[:,1])
[stat_x,stat_y] = transform(p1,p2,x1,y1)
avg_stat_x = np.mean(stat_x)
avg_stat_y = np.mean(stat_y)

corners = np.array([[sat_data.bounds[0],sat_data.bounds[1]],   # bottom left
                    [sat_data.bounds[0],sat_data.bounds[3]],   # top left
                    [sat_data.bounds[2],sat_data.bounds[1]],   # bottom right
                    [sat_data.bounds[2],sat_data.bounds[3]]])  # top right
corners_lon,corners_lat = transform(p2,p1,corners[:,0],corners[:,1])

# plot imagery and station locations
show(sat_data,ax=ax[0],cmap="gray")

ax[0].xaxis.set_ticks_position('both')
ax[0].set_xticks([corners[0,0],corners[2,0]])
ax[0].set_xticklabels(labels=[str(round(corners_lat[0],2)) + '$^\circ$',str(round(corners_lat[2],2)) + '$^\circ$'])
ax[0].set_xlabel("Latitude")
ax[0].yaxis.set_ticks_position('both')
ax[0].set_yticks([corners[0,1],corners[1,1]])
ax[0].set_yticklabels(labels=[str(round(corners_lon[0],2)) + '$^\circ$',str(round(corners_lon[1],2)) + '$^\circ$'])
ax[0].set_ylabel("Longitude")

dataPath = "/media/Data/Data/PIG/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
figPath = "/home/setholinger/Documents/Projects/PIG/location/polarization/"

# set data paramters
type = "short"
fs = 100
numCluster = 10

# set threshold parameters
# xcorr_percent_thresh = 0.1 will compute polarizations for the 10% best correlated events
xcorr_percent_thresh = 0.001
norm_thresh = 2
percent = xcorr_percent_thresh*100

# make dummy arrays for test
all_first_components = (np.random.rand(1000,2)-0.5)*2
back_azimuths = np.empty((0,1),"float64")

# set up colormap
colors1 = [ cm.plasma(x) for x in np.linspace(0,1,180)]
colors2 = [ cm.plasma(x) for x in np.linspace(0,1,181)]
colors = np.append(colors1,colors2[::-1],0)

for s in range(len(all_first_components)):

    baz = compute_baz(all_first_components[s,:])
    back_azimuths = np.vstack((back_azimuths,baz))
    # make beams/rays using the pca components for plotting
    [x,y] = [np.linspace(avg_stat_x,avg_stat_x-all_first_components[s,1]*10000,10),
            np.linspace(avg_stat_y,avg_stat_y+all_first_components[s,0]*10000,10)]

    colorInd = int(np.round(back_azimuths[s]))
    ax[0].plot(x,y,color=colors[colorInd], alpha=0.01, linewidth=10,zorder=s)

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

ax[0].scatter(stat_x,stat_y,marker="^",c='black',zorder=len(all_first_components)*10)

# add north arrow
ax[0].arrow(avg_stat_x-65000,avg_stat_y+70000,-10000,0,width = 500,head_width=3000,head_length=3000,fc="k", ec="k")
ax[0].text(avg_stat_x-74000,avg_stat_y+73000,"N",size="large")

# add distance scale
ax[0].plot(np.linspace(avg_stat_x-60000,avg_stat_x-80000,10),np.ones(10)*avg_stat_y-30000,c="k")
ax[0].text(avg_stat_x-82000,avg_stat_y-26000,"20 km",size="medium")

# plot histogram of back azimuths for cluster
n,bins,patches = ax[1].hist(back_azimuths,36)
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', colors[int(bins[i])])
hist,edges = np.histogram(back_azimuths,36)
ax[1].set_xlabel("Back Azimuth (degrees)")
ax[1].set_xlim(0,360)
ax[1].set_ylim(0,max(hist))
ax[1].set_ylabel("Number of Windows")
#ax[1].axis('square')
#plt.savefig(figPath + "norm>" + str(norm_thresh) + "/top_" + str(percent) + "%/cluster_0_polarizations.png")
plt.show()
