import obspy
from obspy import UTCDateTime
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
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
outPath = "/home/setholinger/Documents/Projects/PIG/location/polarization/"

norm_component = 0
type = "short"
method = 'k_shape'
fs = 100
numCluster = 14
c = 3

outPath = outPath + "3D_clustering/"
if norm_component:
    outPath = outPath + "normalized_components/"
    templatePath = templatePath + type + "_normalized_3D_clustering/" + method + "/"
else:
    templatePath = templatePath + type + "_3D_clustering/" + method + "/"

# set threshold parameters
# xcorr_percent_thresh = 0.1 will compute polarizations for the 10% best correlated events
norm_thresh = 2.75
MAD = 0
xcorr_percent_thresh = 0.1

# set windowing parameters
snipLen = 500
winLen = 10
slide = 5
numSteps = int((snipLen-winLen)/slide)

# set stations and components
chans = ["HHN","HHE","HHZ"]
stations = ["PIG2","PIG4","PIG5"]
#stat_coords = np.array([[-100.748596,-75.016701],[-100.786598,-75.010696],[-100.730904,-75.009201],[-100.723701,-75.020302],[-100.802696,-75.020103]])
stat_coords = np.array([[-100.786598,-75.010696],[-100.723701,-75.020302],[-100.802696,-75.020103]])
plotStat = 'PIG2'

# read imagery data, get coordinate system, convert station coordinates to x and y, and take average station location
file = "/media/Data/Data/PIG/TIF/LC08_L1GT_001113_20131012_20170429_01_T2_B4.TIF"
sat_data = rasterio.open(file)
p2 = Proj(sat_data.crs,preserve_units=False)
p1 = Proj(proj='latlong',preserve_units=False)
[stat_x,stat_y] = transform(p1,p2,stat_coords[:,0],stat_coords[:,1])
avg_stat_x = np.mean(stat_x)
avg_stat_y = np.mean(stat_y)

# set frequency
prefiltFreq = [0.05,1]
freq = [0.05,1]

# load waveforms
#waves = obspy.read(templatePath + type + '_waveforms_' + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + 'Hz.h5')
# load detection times
detFile = h5py.File(templatePath + "detection_times.h5","r")
detTimes = list(detFile["times"])
detFile.close()

# load clustering results
clustFile = h5py.File(templatePath + str(numCluster) +  "/" + str(numCluster) + "_cluster_predictions_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
pred = np.array(list(clustFile["cluster_index"]))
centroids = list(clustFile["centroids"])
clustFile.close()

# read in correlation results for the current cluster
corrFile = h5py.File(templatePath + str(numCluster) + "/centroid" + str(c) + "_correlations_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
cluster_xcorr_coef = np.array(list(corrFile["corrCoefs"]))
corrFile.close()

# read backazimuth data
percent = xcorr_percent_thresh*100
outFile = h5py.File(outPath + "win_len_" + str(winLen) + "/norm>" + str(norm_thresh) + "/top_" + str(percent) + "%/" + str(numCluster) + "/cluster_" + str(c) + "_backazimuths.h5","r")
back_azimuths = np.array(list(outFile["backazimuths"]))
outFile.close()

# make plots
#fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8, 8),gridspec_kw={'height_ratios':[1,0.4]})
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8, 8))

# do some stuff
corners = np.array([[sat_data.bounds[0],sat_data.bounds[1]],   # bottom left
                    [sat_data.bounds[0],sat_data.bounds[3]],   # top left
                    [sat_data.bounds[2],sat_data.bounds[1]],   # bottom right
                    [sat_data.bounds[2],sat_data.bounds[3]]])  # top right
corners_lon,corners_lat = transform(p2,p1,corners[:,0],corners[:,1])

# get indices of waveforms in current cluster and events in cluster above xcorr threshold
clusterInd = [i for i, x in enumerate(pred==c) if x]
n_events = round(xcorr_percent_thresh*len(clusterInd))
threshInd = abs(cluster_xcorr_coef).argsort()[-1*n_events:][::-1]

# plot imagery
show(sat_data,ax=ax,cmap="gray")

# handle axes
percent = xcorr_percent_thresh*100
plt.suptitle("Cluster " + str(c) + " Polarizations (top " + str(percent) + "% of events)")

# compute histogram of back azimuths
baz_hist,edges = np.histogram(back_azimuths,bins=np.linspace(0,360,37))
all_first_components = np.zeros((100000,1))

# set up colormap
colors = [ cm.plasma(x) for x in np.linspace(0,1,max(baz_hist)+1)]

# plot all rays in 10-degree bins with length proportional to # of windows in that bin
rays = np.zeros((36,2),'float64')
scale = 25000
max_width = 2*np.pi*scale/36
max_width = 8
for i in range(36):
  angle = i*10
  rays[i,:] = compute_rays(angle)
  rayLength = baz_hist[i]/max(baz_hist)*scale
  [x,y] = [np.linspace(avg_stat_x,avg_stat_x+rays[i,0]*rayLength,100),
           np.linspace(avg_stat_y,avg_stat_y+rays[i,1]*rayLength,100)]
  lwidths=np.linspace(0,max_width,100)*rayLength/scale
  points = np.array([x, y]).T.reshape(-1, 1, 2)
  segments = np.concatenate([points[:-1], points[1:]], axis=1)
  lc = LineCollection(segments, linewidths=lwidths,color='maroon',alpha=0.5,zorder=len(all_first_components)*10)
  ax.add_collection(lc)

# define, transform, and plot lat/lon grid
lat = [-74,-75]
lon = [-98,-100,-102,-104]
x_lab_pos=[]
y_lab_pos=[]
line = np.linspace(corners_lat[0]+1,corners_lat[2]-1,100)
for l in lon:
  line_x,line_y = transform(p1,p2,np.linspace(l,l,100),line)
  ax.plot(line_x,line_y,linestyle='--',linewidth=1,c='gray',alpha=1)
  y_lab_pos.append(line_y[np.argmin(np.abs(line_x-corners[0,0]))])
line = np.linspace(corners_lon[0]-2,corners_lon[1]+1,100)
for l in lat:
  line_x,line_y = transform(p1,p2,line,np.linspace(l,l,100))
  ax.plot(line_x,line_y,linestyle='--',linewidth=1,c='gray',alpha=1)
  x_lab_pos.append(line_x[np.argmin(np.abs(line_y-corners[0,1]))])
ax.set_xlim([corners[0,0],corners[2,0]])
ax.set_ylim([corners[0,1],corners[1,1]])

ax.set_xticks(x_lab_pos)
ax.set_xticklabels(labels=[str(lat[0]) + '$^\circ$',str(lat[1]) + '$^\circ$'])
ax.set_xlabel("Latitude")
ax.set_yticks(y_lab_pos)
ax.set_yticklabels(labels=[str(lon[0]) + '$^\circ$',str(lon[1]) + '$^\circ$',str(lon[2]) + '$^\circ$',str(lon[3]) + '$^\circ$'])
ax.set_ylabel("Longitude")

# colors
k1 = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
k2 = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]

# plot ice front
front_x = [-1.644e6,-1.64e6,-1.638e6,-1.626e6,-1.611e6,-1.6095e6,-1.6055e6,-1.6038e6,-1.598e6,-1.6005e6,-1.6e6,-1.595e6]
front_y = [-3.34e5,-3.33e5,-3.44e5,-3.445e5,-3.475e5,-3.43e5,-3.4e5,-3.413e5,-3.356e5,-3.32e5,-3.289e5,-3.29e5]
ax.plot(front_x, front_y,c=k1,zorder=len(all_first_components)*10)

# plot rift
rift1_x = [-1.63e6,-1.6233e6,-1.6132e6,-1.6027e6]
rift1_y = [-3.255e5,-3.237e5,-3.236e5,-3.281e5]
ax.plot(rift1_x,rift1_y,c=k2,zorder=len(all_first_components)*10)
rift2_x = [-1.63e6,-1.6232e6,-1.6132e6]
rift2_y = [-3.28e5,-3.2706e5,-3.236e5]
ax.plot(rift2_x,rift2_y,c=k2,zorder=len(all_first_components)*10)

# plot station locations
ax.scatter(stat_x,stat_y,marker="^",c='black',zorder=len(all_first_components)*10)

# add north arrow
#ax.arrow(avg_stat_x-65000,avg_stat_y+70000,-10000,0,width = 500,head_width=3000,head_length=3000,fc="k", ec="k",zorder=len(all_first_components)*10)
#ax.text(avg_stat_x-74000,avg_stat_y+73000,"N",size="large",zorder=len(all_first_components)*10)

# add distance scale
#ax.plot(np.linspace(avg_stat_x-60000,avg_stat_x-80000,10),np.ones(10)*avg_stat_y-30000,c="k")
#ax.text(avg_stat_x-82000,avg_stat_y-26000,"20 km",size="medium")

# plot events and centroid
waveFile = h5py.File(templatePath + str(numCluster) + "/aligned_cluster" + str(c) + "_waveform_matrix_" + str(prefiltFreq[0]) + "-" + str(prefiltFreq[1]) + "Hz.h5","r")
alignedWaves = np.array(list(waveFile["waveforms"]))
waveFile.close()
alignedWave_fs = 2
t = np.linspace(0,snipLen*3,(snipLen*alignedWave_fs+1)*3)
maxAmps = np.zeros((len(threshInd),1),'float64')
#for w in range(len(threshInd)):
#  ax[1].plot(t,alignedWaves[threshInd[w]],'k',alpha=0.1)
#  maxAmps[w] = np.max(np.abs(alignedWaves[threshInd[w]]))
#ax[1].plot(t,centroids[c].ravel()/np.max(abs(centroids[c].ravel()))*np.median(maxAmps))
#ax[1].set_ylim([-4*np.median(maxAmps),4*np.median(maxAmps)])
#ax[1].title.set_text('Centroid and Events Used in Polarization Analysis')
#ax[1].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
#ax[1].set_xlabel("Time (seconds)")
#ax[1].set_ylabel("Velocity (m/s)")
#ax[1].set_xlim([t[0],t[-1]])
#clusterChans = ["HHZ","HHN","HHE"]
#ax[1].set_xticks([0,snipLen/2,snipLen,snipLen*3/2,snipLen*2,snipLen*5/2,snipLen*3])
#xPos = [snipLen,snipLen*2]
#for xc in xPos:
#  ax[1].axvline(x=xc,color='k',linestyle='--')
#ax[1].set_xticklabels(['0','250\n'+clusterChans[0],'500  0   ','250\n'+clusterChans[1],'500  0   ','250\n'+clusterChans[2],'500'])
#ax[1].grid(linestyle=":")
#ax[1].grid()

plt.tight_layout()

#plt.show()
plt.savefig(outPath + "win_len_" + str(winLen) + "/norm>" + str(norm_thresh) + "/top_" + str(percent) + "%/cluster_" + str(c) + "_polarizations_plot_only.png",dpi=400)
plt.close()
