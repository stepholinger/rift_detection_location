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
from polarization_analysis_par_fun_RC import *

# input parameters used in clustering
numCluster = 7
nproc = 2
clust_method = "k_shape"
type = "short"
fs = 100

# set paths
dataPath = "/media/Data/Data/PIG/"
templatePath = "/home/setholinger/Documents/Projects/PIG/detections/templateMatch/multiTemplate/run3/"
outPath = "/home/setholinger/Documents/Projects/PIG/location/polarization/3D_clustering/"

# set parameters for polarization calculation
norm_component = 0
MAD = 0
norm_thresh = 2.75
xcorr_percent_thresh = 1
snipLen = 500
winLen = 10
slide = 5

# call the function
if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        set_start_method("spawn")
    except:
        pass
    p = multiprocessing.Pool(processes=nproc)

    inputs = [[] for i in range(numCluster)]
    for i in range(numCluster):
        inputs[i].append(range(numCluster)[i])
        inputs[i].append(int(numCluster))
        inputs[i].append(int(nproc))
        inputs[i].append(clust_method)
        inputs[i].append(type)
        inputs[i].append(fs)
        inputs[i].append(dataPath)
        inputs[i].append(templatePath)
        inputs[i].append(outPath)
        inputs[i].append(norm_component)
        inputs[i].append(MAD)
        inputs[i].append(norm_thresh)
        inputs[i].append(xcorr_percent_thresh)
        inputs[i].append(snipLen)
        inputs[i].append(winLen)
        inputs[i].append(slide)
    p.map(run_polarization,inputs)
    p.close()
    p.join()
