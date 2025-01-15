import numpy as np
import matplotlib.pyplot as plt


# Constants
h0 = 100  # Initial height in meters
g_measured = 9.93  # True gravitational acceleration
learning_rate = 1e-5  # Further reduced step size for gradient descent
tolerance = 1e-6  # Tolerance for stopping condition
max_iterations = 1000  # Max number of iterations

# Generate time points for simulation
time_end = np.sqrt(2 * h0 / g_measured)
time_points = np.linspace(0, time_end, 100)  # Reduced number of points to make it faster

# Simulate the real measured heights using the true gravitational acceleration
height_measured = h0 - 0.5 * g_measured * time_points**2

# Objective function: calculate the sum of squared differences
def objective_function(g):
    height_simulated = h0 - 0.5 * g * time_points**2
    return np.sum((height_measured - height_simulated) ** 2)

# Gradient calculation: derivative of the objective function with respect to g
def gradient(g):
    height_simulated = h0 - 0.5 * g * time_points**2
    gradient_value = np.sum(time_points**2 * (height_measured - height_simulated))
    return gradient_value

# Gradient descent function to optimize g
def gradient_descent(initial_g, learning_rate, tolerance, max_iterations):
    g = initial_g
    for iteration in range(max_iterations):
        grad = gradient(g)
        new_g = g - learning_rate * grad
        if abs(new_g - g) < tolerance:
            break
        g = new_g
    return g, iteration

# Initial guess for g
initial_g = 9.7  # A value closer to the expected result

# Perform gradient descent optimization
optimized_g, iterations = gradient_descent(initial_g, learning_rate, tolerance, max_iterations)

# Simulate the candidate height trajectory using the optimized g value
height_optimized = h0 - 0.5 * optimized_g * time_points**2

# Plot the measured heights and the optimized heights
plt.figure(figsize=(10, 6))
plt.plot(time_points, height_measured, label='Measured (g = 9.93)', color='blue')
plt.plot(time_points, height_optimized, label=f'Optimized (g = {optimized_g:.5f})', color='red', linestyle='--')
plt.title('Comparison of Measured and Optimized Heights')
plt.xlabel('Time (seconds)')
plt.ylabel('Height (meters)')
plt.legend()
plt.grid(True)
plt.show()
