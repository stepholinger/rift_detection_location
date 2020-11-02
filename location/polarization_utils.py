import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
from sklearn.decomposition import PCA
import pyproj
from pyproj import Proj,transform
import obspy


def observed_first_arrival(stream):
    # cross correlate all traces and find station with largest shift
    shifts = np.zeros((len(stream),1))
    maxAmp = np.zeros((len(stream),1))
    for j in range(len(stream)):
        corr = correlate(stream[0],stream[j],stream[0].stats.npts,normalize='naive',demean=False,method='auto')
        shift, corrCoef = xcorr_max(corr)
        shifts[j] = shift
    stat_idx = np.argmax(shifts)
    first_stat = stream[stat_idx].stats.station
    return first_stat


def predict_first_arrival(pca_component,avg_stat_x,avg_stat_y,stat_x,stat_y,stats):
    # be wary- the transformed coordinate system's x-axis is meters north and the y-axis is meters east!
    # so the pca_first_component[~,0] (which is cartesian x) is meters east and therefore along the transformed y-axis
    # and the pca_first_component[~,1] (which is cartesian y) is meters north and therefore along the transformed x-axis
    # (and needs sign flipped since transformed x-axis is positive south)
    potential_location_x_initial = avg_stat_x - pca_component[1]*10000
    potential_location_y_initial = avg_stat_y + pca_component[0]*10000
    potential_location_x_reversed = avg_stat_x + pca_component[1]*10000
    potential_location_y_reversed = avg_stat_y - pca_component[0]*10000
    distance_initial =  np.zeros((len(stat_x),1))
    distance_reversed =  np.zeros((len(stat_x),1))
    for i in range(len(stat_x)):
        distance_initial[i] = np.sqrt((potential_location_x_initial-stat_x[i])**2 + (potential_location_y_initial-stat_y[i])**2)
        distance_reversed[i] = np.sqrt((potential_location_x_reversed-stat_x[i])**2 + (potential_location_y_reversed-stat_y[i])**2)
    initial_pred = stats[np.argmin(distance_initial)]
    reverse_pred = stats[np.argmin(distance_reversed)]
    pred_symmetric = [initial_pred,reverse_pred]
    return pred_symmetric,initial_pred


def correct_polarization(pca_components,stats,first_stat,avg_stat_x,avg_stat_y,stat_x,stat_y):
    # make copy of pca components
    pca_components_corrected = pca_components

    for i in range(len(pca_components)):
        # get which stations see first arrival for source at either side of polarization direction
        pred_symmetric,initial_pred = predict_first_arrival(pca_components[i,:],avg_stat_x,avg_stat_y,stat_x,stat_y,stats)

        # if observed first station is not either of those, we aren't seeing a phase polarized in the propagation direction
        # if observed first station is one of those two, check whether we need to flip sign
        if pred_symmetric.count(first_stat) > 0:
            # if predicted first station from initial pca answer agrees with observed first station, don't flip sign
            # if predicted first station from initial pca answer disagrees with observed first station, flip sign
            if first_stat != initial_pred:
                #print("Flipped")
                pca_components_corrected[i,:] = -1*pca_components_corrected[i,:]
        else:
            pca_components_corrected[i,:] = 0
    return pca_components_corrected


def compute_pca(dataPath,stat,chans,fs,winLen,slide,numSteps,freq,eventLims):
    # make array for storage of pca components
    first_components = np.empty((0,2),"float64")

    # read event and get data
    event_N = readEvent(dataPath + "MSEED/noIR/",stat,chans[0],eventLims,freq)
    event_E = readEvent(dataPath + "MSEED/noIR/",stat,chans[1],eventLims,freq)
    data_N = event_N[0].data
    data_E = event_E[0].data

    # put data into matrix for PCA
    data_matrix = np.vstack((data_E,data_N)).T

    # get mean amplitude for whole trace (average of both components)
    mean_amp = np.mean(abs(data_matrix))

    # itertate through data in windows
    for n in range(numSteps):
        # get current window
        startInd = n*slide*fs
        endInd = startInd + winLen*fs
        X = data_matrix[startInd:endInd,:]

        # get average amplitude on for window
        mean_amp_win = np.mean(abs(X))

        # apply staLta criteria
        if  np.mean(abs(X)) > mean_amp:
            # find component with max amplitude, normalize both traces by that max value, and compute PCA
            maxAmp = np.amax(abs(X))
            X_norm = np.divide(X,maxAmp)
            pca = PCA(n_components = 2)
            pca.fit(X_norm)

            # save result
            first_components = np.vstack((first_components,pca.components_[0,:]))

        else:
            # add zeros if we didn't run PCA on the window due to low STALTA
            first_components = np.vstack((first_components,[0,0]))
    return first_components


def compute_baz(component):
    # based on which quadrant we're in, use the appropriate triangle side ratios for arctan
    if component[0] > 0 and component[1] > 0:
        baz = np.arctan(abs(component[0]/component[1]))*180/np.pi
    if component[0] > 0 and component[1] < 0:
        baz = np.arctan(abs(component[1]/component[0]))*180/np.pi + 90
    if component[0] < 0 and component[1] < 0:
        baz = np.arctan(abs(component[0]/component[1]))*180/np.pi + 180
    if component[0] < 0 and component[1] > 0:
        baz = np.arctan(abs(component[1]/component[0]))*180/np.pi + 270
    return baz


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

def readEvent(path,stat,chan,tempLims,freq):
    # extract strings for date and make date string
    tempDate = tempLims[0].isoformat().split("T")[0]

    # read in data for template
    event = obspy.read(path + stat + "/" + chan + "/" + tempDate + "." + stat + "." + chan + ".noIR.MSEED")

    # basic preprocessing and filtering
    event.detrend("demean")
    event.detrend("linear")
    event.taper(max_percentage=0.01, max_length=10.)
    event.filter("bandpass",freqmin=freq[0],freqmax=freq[1])

    # trim the data to the time ranges of template for each band
    event.trim(starttime=tempLims[0],endtime=tempLims[1])
    return event


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def plot_pca(stat,data_Z,data_N,data_E,startInd,endInd,X_norm,pca):
    fig,ax = plt.subplots(nrows=4,ncols=1,sharex=False,sharey=False,gridspec_kw={'height_ratios':[1,1,1,8]},figsize = (7,10))
    ax[0].plot(data_Z)
    ax[0].text(0,max(data_Z)*0.5,"HHZ")
    ax[0].set_ylim([-1*max(abs(data_Z)),max(abs(data_Z))])
    ax[0].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    ax[1].plot(data_N)
    ax[1].text(0,max(data_N)*0.5,"HHN")
    ax[1].axvspan(startInd, endInd, color='red', alpha=0.5)
    ax[1].set_ylim([-1*max(abs(data_N)),max(abs(data_N))])
    ax[1].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    ax[2].plot(data_E)
    ax[2].text(0,max(data_E)*0.5,"HHE")
    ax[2].axvspan(startInd, endInd, color='red', alpha=0.5)
    ax[2].set_ylim([-1*max(abs(data_E)),max(abs(data_E))])
    ax[2].ticklabel_format(style='sci', axis='y',scilimits=(0,0))
    ax[3].scatter(X_norm[:, 0], X_norm[:, 1], alpha = 0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    ax[3].set(xlim=(-2, 2),ylim=(-2, 2))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(stat + " Polarization")

# plot histogram of back azimuths for cluster
#n,bins,patches = ax[1].hist(back_azimuths,bins=np.linspace(0,360,37))
#for i, p in enumerate(patches):
#    plt.setp(p, 'facecolor', colors[baz_hist[i]])
#ax[1].set_xlabel("Back Azimuth (degrees)")
#ax[1].set_xlim(0,360)
#ax[1].set_ylim(0,max(baz_hist))
#ax[1].set_ylabel("Number of Windows")
