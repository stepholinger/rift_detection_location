import scipy
import datetime
import pandas as pd
import pathlib
import numpy as np
from sklearn.neighbors import KernelDensity



def gps_ice_speed(station,peak_height,kde_width):
    home_dir = str(pathlib.Path().absolute())
    data = pd.read_csv(home_dir+"/data/gps/" + station + "_5minavg_neu.txt",names=["seconds","days","n","e","u","lat","lon","alt"],delim_whitespace=True)

    time_vect = [datetime.datetime(2012,1,9,0) + datetime.timedelta(days=data["days"][0])]
    date_vect = []
    for d in range(len(data["days"]))[1:]:
        time_vect.append(datetime.datetime(2012,1,9,0) + datetime.timedelta(days=data["days"][d]))
    
    dist = np.hypot(data["n"],data["e"])        
    daily_dist_vect = []
    daily_dist = []
    for t in range(len(time_vect)-1):
        if time_vect[t].date() == time_vect[t+1].date():
            daily_dist.append(dist[t])
        if time_vect[t].date() < time_vect[t+1].date():
            daily_dist.append(dist[t])
            daily_dist_vect.append(np.mean(daily_dist))
            daily_dist = []
            date_vect.append(time_vect[t].date())

    speed = np.gradient(daily_dist_vect,86400) 
    speed = remove_artifacts(speed,peak_height,kde_width)
    
    return speed,date_vect



def remove_artifacts(data,peak_height,kde_width):

    env = np.abs(scipy.signal.hilbert(data-np.mean(data)))
    peaks = scipy.signal.find_peaks(env,height=2.5e-5)
    kde = KernelDensity(kernel='epanechnikov', bandwidth=2.5).fit(peaks[0].reshape(-1,1))
    density = kde.score_samples(np.linspace(0,len(data),len(data)).reshape(-1,1))
    ind = np.where(np.isfinite(density))[0]
    bounds = [ind[0]]
    for i in range(len(ind))[1:]:
        if ind[i]-ind[i-1] > 1:
            bounds.append(ind[i-1])
            bounds.append(ind[i])
    bounds.append(ind[-1])
    bounds = np.array(bounds)
    bounds = bounds.reshape((int(len(bounds)/2),2))
    for b in range(len(bounds)):
        x0 = bounds[b,0]-1
        x1 = bounds[b,1]+1
        y0 = data[x0]
        y1 = data[x1]
        m = (y1-y0)/(x1-x0)
        line = np.linspace(x0,x1,x1-x0)
        line1 = line*m  
        intercept = line1[0]-y0
        line = line*m -intercept
        data[x0:x1] = line
    return data

