from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt

# Create a Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1)

kf.x = np.array([2., 0.])  # initial state (location and velocity)

kf.F = np.array([[1., 1.], # state transition matrix
                 [0., 1.]])  

kf.H = np.array([[1., 0.]])  # Measurement function

kf.P *= np.array([[1000., 0.], [0., 1000.]])  # covariance matrix

kf.R = np.array([[5.]])  # measurement uncertainty

kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)  # process uncertainty

