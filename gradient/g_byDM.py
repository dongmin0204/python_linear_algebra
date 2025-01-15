import numpy as np
import matplotlib.pyplot as plt

H0 = 100  # Initial height in meters
G_MEASURED = 9.93  # True gravitational acceleration
LEARNING_RATE = 1e-5  # Further reduced step size for gradient descent
TOLERANCE = 1e-6  # Tolerance for stopping condition
MAX_ITERATION = 1000  # Max number of iterations

# Generate time points for simulation
time_end = np.sqrt(2 * H0 / G_MEASURED) # Time
time_points = np.linspace(0, time_end, 100)  # Reduced number of points to make it faster

# Simulate the real measured heights using the true gravitational acceleration
height_measured = H0 - 0.5 * G_MEASURED * time_points**2

def objective_function(g, time_points):
    h0 = 100
    height_simulated = h0 - 0.5 * g * time_points**2
    return np.sum((height_measured - height_simulated) ** 2)



for i in gravity_points:
    objective_function(i,time_points)
