from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt
import random


observed_wait_times = [11, 15, 19, 8, 10, 13, 9, 20, 15, 13, 15, 16]
# Initialize the Kalman Filter
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([observed_wait_times[0], 0.])  # initial state (location and velocity)
kf.F = np.array([[1., 1.],  # state transition matrix
                 [0., 1.]])
kf.H = np.array([[1., 0.]])  # measurement function
kf.P *= np.array([[1000., 0.],
                  [0., 1000.]])  # covariance matrix
kf.R = np.array([[5.]])  # measurement uncertainty
kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.1)  # process uncertainty



# Prepare to plot
kalman_estimates = []
average_estimates = []

# Initialize variables for averaging
total_wait_time = 0
num_observations = 0

# Initialize lists to store errors
kalman_errors = []
average_errors = []

# Run the Kalman Filter and calculate averages
for wait_time in observed_wait_times:
    # Step 1: Update with a new measurement
    kf.update(np.array([wait_time]))

    # Step 2: Predict the next state
    kf.predict()

    # Store the Kalman filter estimate
    kalman_estimates.append(kf.x[0])

    # Update total wait time and number of observations
    total_wait_time += wait_time
    num_observations += 1

    # Calculate the average wait time
    average_wait_time = total_wait_time / num_observations
    average_estimates.append(average_wait_time)
    
    # Calculate errors
    kalman_error = (wait_time - kf.x[0])**2
    average_error = (wait_time - average_wait_time)**2
    
    kalman_errors.append(kalman_error)
    average_errors.append(average_error)

# Calculate Mean Squared Errors
kalman_mse = np.mean(kalman_errors)
average_mse = np.mean(average_errors)

print("Kalman Filter MSE:", kalman_mse)
print("Averaging Method MSE:", average_mse)

# Plotting
plt.figure(figsize=(12, 4))
plt.plot(range(1, len(observed_wait_times) + 1), observed_wait_times, label='Observed Wait Times')
plt.plot(range(1, len(kalman_estimates) + 1), kalman_estimates, label='Kalman Filter Estimates')
plt.plot(range(1, len(average_estimates) + 1), average_estimates, label='Average Wait Times')
plt.legend()
plt.xlabel('Observation Number')
plt.ylabel('Wait time (minutes)')
plt.title('Comparison of Wait Time Estimation Methods')
plt.xticks(range(1, len(observed_wait_times) + 1))
plt.grid(True)
plt.text(0.7, 0.20, f'Kalman Filter MSE: {kalman_mse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.text(0.7, 0.13, f'Averaging Method MSE: {average_mse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
plt.show()
