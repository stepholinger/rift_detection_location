import tslearn
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import KShape
import matplotlib.pyplot as plt
import time
import numpy as np

# code to test runtime of kshape

# note on data: we will need to downsample and trim
# lets assume we downsample to 1 hz and get the 100s containing the event
fs = 2
snipLen = 300
numCluster = 10

# generate n 5 minute (300*fs) long time series
n_vect = [10,100,1000,10000]
runtimes = np.zeros((len(n_vect),1))
for n in range(len(n_vect)):
    timer = time.time()
    X = random_walks(n_ts=n_vect[n], sz=snipLen*fs, d=1)
    X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    ks = KShape(n_clusters=numCluster, n_init=1, random_state=0).fit(X)
    runtimes[n] = time.time()-timer
    print("Finished clustering (" + str(numCluster) + " clusters) for " + str(n_vect[n]) + " events in " + str(runtimes[n]) + " seconds")

plt.scatter(n_vect,runtimes)
plt.xscale('log')
plt.xlabel("Number of waveforms")
plt.ylabel("Runtime")
plt.show()
