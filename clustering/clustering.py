import tslearn
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape
import numpy as np
import obspy
from obspy.signal.cross_correlation import correlate
from obspy.signal.cross_correlation import xcorr_max
import h5py
import pathlib
import matplotlib.pyplot as plt



def load_waveform(c,detection_time):
    
    # read in data for template
    date_string = detection_time.date.strftime("%Y-%m-%d")
    fname = c.data_path + c.station + "/HH*/*" + date_string + "*"
    st = obspy.read(fname)

    # basic preprocessing
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, max_length=10.)

    # filter and resample the data to each band
    st.filter("bandpass",freqmin=c.freq[0],freqmax=c.freq[1])
    st.resample(c.freq[1]*2.1)
    
    # make a copy so we only have to read once for all detections on a given day
    event = st.copy()

    # trim the data to the time ranges of template for each band
    event.trim(starttime=detection_time,endtime=detection_time + c.trace_length)

    return st,event



def get_3D_trace(c,event):
    fs = c.freq[1]*2.1
    waveform = np.zeros((3*int(c.trace_length*fs+1)))
    for i in range(len(event)):
        trace = event.select(component=c.component_order[i])[0].data
        waveform[i*int(c.trace_length*fs+1):i*int(c.trace_length*fs+1)+len(trace)] = trace
    return waveform



def get_input_waveforms(ds,c):

    home_dir = str(pathlib.Path().absolute())
    #waveform_file = h5py.File(home_dir + "/outputs/clustering/input_waveforms.h5",'w') 

    # make output array
    fs = c.freq[1]*2.1
    waveform_matrix = np.zeros((len(ds.events),3*int(c.trace_length*fs+1)))
    
    # extract times for each event in the dataset
    detection_times = []
    for event in ds.events:
        detection_times.append(event.origins[0].time)
    
    for i in range(len(detection_times)):
        # only read file if date of event is different than last event
        if i == 0 or detection_times[i].date != st[0].stats.starttime.date:
            st,event_st = load_waveform(c,detection_times[i])
        else:
            event_st = st.copy()
            event_st.trim(starttime=detection_times[i],endtime=detection_times[i] + c.trace_length)

        #event.write(outPath + type + '_waveforms_' + chan + '_' + str(freq[0]) + "-" + str(freq[1]) + 'Hz.h5','H5',mode='a')
        waveform = get_3D_trace(c,event_st)
        waveform_matrix[i,:] = waveform    

        # give output
        print(str(i)+"/"+str(len(waveform_matrix))+" event waveforms retrieved...")
    return waveform_matrix


    
def cluster_events(c,waveforms):

    # scale mean around zero
    scaled_waveforms = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(waveforms)

    ks = KShape(n_clusters=c.num_clusters, n_init=1, random_state=1,n_component=3)
    predictions = ks.fit_predict(scaled_waveforms)

    # save results
    home_dir = str(pathlib.Path().absolute())
    cluster_file = h5py.File(home_dir + "/outputs/clustering/" + str(c.num_clusters) + "_cluster_results.h5","w")
    cluster_file.create_dataset("cluster_index",data=predictions)
    cluster_file.create_dataset("centroids",data=ks.cluster_centers_)
    cluster_file.create_dataset("inertia",data=ks.inertia_)
    cluster_file.close()
    ks.to_hdf5(home_dir + "/outputs/clustering/" + str(c.num_clusters) + "_cluster_model.h5")
    
    
    
def plot_clusters(c,cluster,centroid,waveforms,correlation_coefficients,shifts):

    # make empty array for storage
    fs = c.freq[1]*2.1
    aligned_cluster_events = np.zeros((len(waveforms),int(c.trace_length*fs+1)*3))

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
            aligned_trace = aligned_trace[:int(c.trace_length*fs+1)*3]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

        else:
            aligned_trace = trace[abs(int(shift)):]
            aligned_cluster_events[w,:len(aligned_trace)] = aligned_trace

    # make plot version 1; shows difference in amplitudes on different components
    fig,ax = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=False,gridspec_kw={'height_ratios':[1,2]})
    sort_index = np.array(np.argsort(abs(correlation_coefficients))[::-1])
    t = np.linspace(0,c.trace_length*3,int(c.trace_length*fs+1)*3)

    # plot all waves and mean waveform (amplitudes preserved)
    for w in range(len(aligned_cluster_events)):
        ax[0].plot(t,aligned_cluster_events[w],'k',alpha=0.005)
    ax[0].plot(t,centroid,linewidth=1)
    ax[0].set_ylim([-10*np.nanmax(abs(centroid)),10*np.nanmax(abs(centroid))])
    for x in [c.trace_length,c.trace_length*2]:
        ax[0].axvline(x=x,color='k',linestyle='--')
    ax[0].title.set_text('Centroid and Cluster Waveforms (Cluster ' + str(cluster) + ')')
    normalized_aligned_cluster_events = np.divide(aligned_cluster_events[sort_index,:],np.amax(np.abs(aligned_cluster_events[sort_index,:]),axis=1,keepdims=True))
    ax[1].imshow(normalized_aligned_cluster_events,vmin=-0.25,vmax=0.25,aspect = 'auto',extent=[0,c.trace_length*3,len(waveforms),0],cmap='seismic')
    ax[1].set_xticks([0,c.trace_length/2,c.trace_length,c.trace_length*3/2,c.trace_length*2,c.trace_length*5/2,c.trace_length*3])
    for x in [c.trace_length,c.trace_length*2]:
        ax[1].axvline(x=x,color='k',linestyle='--')
    ax[1].set_xticklabels(['0','250\n'+c.component_order[0],'500  0   ','250\n'+c.component_order[1],'500  0   ','250\n'+c.component_order[2],'500'])

    plt.xlabel("Time (seconds)")
    plt.ylabel("Event Number")
    plt.tight_layout(h_pad=1.0)
    home_dir = str(pathlib.Path().absolute())
    plt.savefig(home_dir + "/outputs/clustering/cluster_" + str(cluster) + "_waveforms.png",dpi=400)
    plt.close()
