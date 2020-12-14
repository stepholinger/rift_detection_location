import obspy
from obspy import UTCDateTime
import h5py
import time
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
import multiprocessing
from multiprocessing import Manager
from multiprocessing import set_start_method

# define function for parallelization
def run_polarization(args):

    # get inputs
    c = args[0]
    numCluster = args[1]
    nproc = args[2]
    clust_method = args[3]
    type = args[4]
    fs = args[5]
    dataPath = args[6]
    templatePath = args[7]
    outPath = args[8]
    norm_component = args[9]
    MAD = args[10]
    norm_thresh = args[11]
    xcorr_percent_thresh = args[12]
    snipLen = args[13]
    winLen = args[14]
    slide = args[15]

    if norm_component:
        outPath = outPath + "normalized_components/"
        templatePath = templatePath + type + "_normalized_3D_clustering/" + clust_method + "/"
    else:
        templatePath = templatePath + type + "_3D_clustering/" + clust_method + "/"

    # get window parameters
    numSteps = int((snipLen-winLen)/slide)

    # set stations and components
    chans = ["HHN","HHE","HHZ"]
    stations = ["PIG2","PIG4","PIG5"]
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

    # get indices of waveforms in current cluster and events in cluster above xcorr threshold
    clusterInd = [i for i, x in enumerate(pred==c) if x]
    if MAD:
        mad = stats.median_absolute_deviation(abs(cluster_xcorr_coef))
        mad = round(mad,2)
        xcorr_percent_thresh = mad/max(abs(cluster_xcorr_coef))
        print(mad)
        print(xcorr_percent_thresh)
        n_events = round(xcorr_percent_thresh*len(clusterInd))
    else:
        n_events = round(xcorr_percent_thresh*len(clusterInd))
    threshInd = abs(cluster_xcorr_coef).argsort()[-1*n_events:][::-1]

    # make array for storing pca vector sums and storing data to plot
    event_index = np.zeros((numSteps*len(threshInd),1),"float64")
    all_first_components = np.zeros((numSteps*len(threshInd),2),"float64")
    clusterEventsAligned = np.zeros((len(threshInd),snipLen*fs),'float64')

    # loop through indices of events in current cluster
    for i in range(len(threshInd)):

        # make array for storage of pca components and empty obspy stream for storing one trace from each station
        first_component_sums = np.zeros((numSteps,2),"float64")
        event_stat = obspy.read()
        event_stat.clear()

        # loop through stations to get one trace from each to find earliest arrival
        #try:
        for stat in range(len(stations)):

            # get times bounds for current event and read event
            #eventLims = [waves[clusterInd[threshInd[i]]].stations.starttime,waves[clusterInd[threshInd[i]]].stations.starttime + snipLen]
            starttime = UTCDateTime(detTimes[clusterInd[threshInd[i]]])
            endtime = starttime + snipLen
            eventLims = [starttime,endtime]
            event_stat += readEvent(dataPath + "MSEED/noIR/",stations[stat],chans[1],eventLims,freq)

        # find station with earliest arrival
        first_stat = observed_first_arrival(event_stat)

        # loop though stations to perform PCA on all windows in the event on each station's data
        for stat in range(len(stations)):

            # compute pca components for all windows in the event
            first_components = compute_pca(dataPath,stations[stat],chans,fs,winLen,slide,numSteps,freq,eventLims)

            # correct polarization direction based on first arrival
            first_components_corrected = correct_polarization(first_components,stations,first_stat,avg_stat_x,avg_stat_y,stat_x,stat_y)

            # sum results (this is vector sum across stations of pca first components for each window)
            first_component_sums = first_component_sums + first_components_corrected

        # give user output every % complete
        if round(i/len(threshInd)*100) > round((i-1)/len(threshInd)*100):
                print(str(round(i/len(threshInd)*100)) + " % complete (cluster " + str(c) +")")

        # fill results vector
        all_first_components[i*numSteps:(i+1)*numSteps,:] = first_component_sums
        event_index[i*numSteps:(i+1)*numSteps,1] = clusterInd[threshInd[i]]*np.ones((numSteps,1))

        #except:
            # give output if no data on current station
            #print("Skipping cluster " + str(c) + " event " + str(i) + " (missing data on " + stations[stat] + ")")

    # make array for storage
    back_azimuths = np.empty((0,1),"float64")
    baz_event_index = np.empty((0,1),"float64")

    # plot pca compoments that exceed norm threshold
    # be wary- the transformed coordinate system's x-axis is meters north and the y-axis is meters east, so the pca_first_component[~,0] (which is cartesian x) is in [L] east
    # and therefore along the transformed y-axis and the pca_first_component[~,1] (which is cartesian y) is in [L] north and therefore along the transformed x-axis
    for s in range(len(all_first_components)):

        # only plot and save results if length of resultant vector has a norm exceeding the threshold
        if np.linalg.norm(all_first_components[s,:]) > norm_thresh:

            # calculate back azimuths and save in array
            baz = compute_baz(all_first_components[s,:])
            back_azimuths = np.vstack((back_azimuths,baz))
            baz_event_index = np.vstack(event_index[s])

    # save actual backazimuth data
    outFile = h5py.File(outPath + "win_len_" + str(winLen) + "/norm>" + str(norm_thresh) + "/top_" + str(percent) + "%/" + str(numCluster) + "/cluster_" + str(c) + "_backazimuths.h5","w")
    outFile.create_dataset("backazimuths",data=back_azimuths)
    outFile.create_dataset("index",data=baz_event_index)
    outFile.close()
