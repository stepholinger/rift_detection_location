import numpy as np

# define function to compute rays for plotting from backazimuths- this is currently made specifically for north = right
def compute_rays(angle):
    ray = np.zeros((1,2),'float64')
    if angle >= 0 and angle < 90:
        ray[0,0] = np.cos(angle*np.pi/180)
        ray[0,1] = np.sin(angle*np.pi/180)
    if angle >= 90 and angle < 180:
        ray[0,0] = -1*np.sin((angle-90)*np.pi/180)
        ray[0,1] = np.cos((angle-90)*np.pi/180)
    if angle >= 180 and angle < 270:
        ray[0,0] = -1*np.cos((angle-180)*np.pi/180)
        ray[0,1] = -1*np.sin((angle-180)*np.pi/180)
    if angle >= 270 and angle < 360:
        ray[0,0] = np.sin((angle-270)*np.pi/180)
        ray[0,1] = -1*np.cos((angle-270)*np.pi/180)
    return ray
