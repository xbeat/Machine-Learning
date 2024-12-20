## Introduction to Math and Statistics for Steam AI in Python
Slide 1: Introduction to Mathematics and Statistics for Steam AI using Python

Python has become an essential tool in the field of Steam AI, offering powerful libraries and frameworks for mathematical and statistical analysis. This presentation will cover fundamental concepts and practical applications, demonstrating how Python can be used to solve complex problems in Steam AI.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Example: Simple neural network for Steam AI
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

Slide 2: Basic Statistical Concepts

Understanding central tendency and dispersion is crucial in Steam AI. We'll use Python to calculate and visualize these measures for a dataset of steam turbine efficiency readings.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
efficiency_readings = np.random.normal(85, 5, 1000)

# Calculate measures of central tendency and dispersion
mean = np.mean(efficiency_readings)
median = np.median(efficiency_readings)
std_dev = np.std(efficiency_readings)

# Visualize the distribution
plt.hist(efficiency_readings, bins=30)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
plt.title('Distribution of Steam Turbine Efficiency Readings')
plt.xlabel('Efficiency (%)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f"Standard Deviation: {std_dev:.2f}")
```

Slide 3: Linear Algebra Fundamentals

Linear algebra is the backbone of many machine learning algorithms used in Steam AI. Let's explore matrix operations and their applications in solving systems of equations related to steam flow.

```python
import numpy as np

# Define matrices representing steam flow equations
A = np.array([[2, 1, -1],
              [1, 3, 2],
              [-1, 2, 4]])

b = np.array([8, 14, 18])

# Solve the system of equations
x = np.linalg.solve(A, b)

print("Solution to the steam flow equations:")
print(f"x = {x[0]:.2f}, y = {x[1]:.2f}, z = {x[2]:.2f}")

# Verify the solution
verification = np.allclose(np.dot(A, x), b)
print(f"Solution verified: {verification}")
```

Slide 4: Probability Distributions

Probability distributions play a crucial role in modeling uncertainties in Steam AI. We'll use Python to generate and visualize common distributions used in this field.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data for normal and exponential distributions
x = np.linspace(0, 10, 1000)
normal_dist = stats.norm.pdf(x, loc=5, scale=1)
exp_dist = stats.expon.pdf(x, scale=2)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, normal_dist, label='Normal (μ=5, σ=1)')
plt.plot(x, exp_dist, label='Exponential (λ=0.5)')
plt.title('Probability Distributions in Steam AI')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: Time Series Analysis

Time series analysis is essential for predicting steam turbine performance and detecting anomalies. Let's use Python to decompose a time series into its trend, seasonal, and residual components.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate synthetic time series data
np.random.seed(0)
time = np.arange(0, 365)
trend = 0.1 * time
seasonal = 10 * np.sin(2 * np.pi * time / 365)
noise = np.random.normal(0, 3, 365)
ts = trend + seasonal + noise

# Decompose the time series
result = seasonal_decompose(ts, model='additive', period=365)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
ax1.plot(time, ts)
ax1.set_title('Original Time Series')
ax2.plot(time, result.trend)
ax2.set_title('Trend')
ax3.plot(time, result.seasonal)
ax3.set_title('Seasonal')
ax4.plot(time, result.resid)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 6: Optimization Techniques

Optimization is crucial in Steam AI for maximizing efficiency and minimizing costs. We'll demonstrate gradient descent, a common optimization algorithm, to find the minimum of a simple function.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5*np.sin(x)

def df(x):
    return 2*x + 5*np.cos(x)

def gradient_descent(start, learn_rate, num_iterations):
    x = start
    x_history = [x]
    
    for _ in range(num_iterations):
        x = x - learn_rate * df(x)
        x_history.append(x)
    
    return x, x_history

# Perform gradient descent
x_min, x_history = gradient_descent(start=3, learn_rate=0.1, num_iterations=50)

# Visualize the optimization process
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', label='f(x)')
plt.plot(x_history, [f(x) for x in x_history], 'ro-', label='Gradient Descent Path')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Minimum found at x = {x_min:.4f}")
```

Slide 7: Hypothesis Testing

Hypothesis testing is used in Steam AI to make decisions based on data. Let's perform a t-test to compare the efficiency of two different steam turbine designs.

```python
import numpy as np
from scipy import stats

# Generate sample data for two turbine designs
np.random.seed(0)
design_a = np.random.normal(85, 2, 100)
design_b = np.random.normal(86, 2, 100)

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(design_a, design_b)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between the two designs.")
else:
    print("Fail to reject the null hypothesis. There is not enough evidence to conclude a significant difference.")

# Visualize the distributions
plt.figure(figsize=(10, 6))
plt.hist(design_a, bins=20, alpha=0.5, label='Design A')
plt.hist(design_b, bins=20, alpha=0.5, label='Design B')
plt.title('Distribution of Efficiency for Two Turbine Designs')
plt.xlabel('Efficiency (%)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

Slide 8: Machine Learning Basics

Machine learning is a powerful tool in Steam AI for predictive maintenance and performance optimization. Let's implement a simple linear regression model to predict steam turbine output based on input features.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Input feature: steam pressure
y = 2 * X + 1 + np.random.randn(100, 1)  # Output: turbine power output

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Steam Turbine Output Prediction')
plt.xlabel('Steam Pressure')
plt.ylabel('Power Output')
plt.legend()
plt.show()
```

Slide 9: Dimensionality Reduction

In Steam AI, we often deal with high-dimensional data. Principal Component Analysis (PCA) is a technique used to reduce dimensionality while preserving important information. Let's apply PCA to a dataset of steam turbine sensor readings.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate synthetic high-dimensional data
np.random.seed(0)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA on Steam Turbine Sensor Readings')
plt.grid(True)
plt.show()

# Determine the number of components needed to explain 95% of the variance
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"Number of components needed to explain 95% of variance: {n_components}")
```

Slide 10: Numerical Integration

Numerical integration is essential in Steam AI for calculating various thermodynamic properties. Let's implement the trapezoidal rule to approximate the integral of a function representing steam enthalpy.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 * np.sin(x)  # Example function representing steam enthalpy

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    return (b - a) / (2 * n) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])

# Define integration limits and number of intervals
a, b = 0, np.pi
n = 1000

# Compute the integral
integral_approx = trapezoidal_rule(f, a, b, n)

print(f"Approximate integral: {integral_approx:.6f}")

# Visualize the function and its integral
x = np.linspace(a, b, 1000)
y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x)')
plt.fill_between(x, y, alpha=0.3)
plt.title('Numerical Integration of Steam Enthalpy Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Fourier Analysis

Fourier analysis is crucial in Steam AI for analyzing periodic signals from turbines and identifying potential issues. Let's use the Fast Fourier Transform (FFT) to analyze a simulated vibration signal from a steam turbine.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a simulated vibration signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.1, 1000)

# Perform FFT
fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

# Plot the original signal and its frequency spectrum
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ax1.plot(t, signal)
ax1.set_title('Simulated Turbine Vibration Signal')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')

ax2.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2])
ax2.set_title('Frequency Spectrum')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_xlim(0, 30)

plt.tight_layout()
plt.show()

# Identify dominant frequencies
dominant_freqs = freqs[np.argsort(np.abs(fft))[-3:]]
print("Dominant frequencies (Hz):", np.abs(dominant_freqs))
```

Slide 12: Monte Carlo Simulation

Monte Carlo simulations are valuable in Steam AI for risk assessment and decision-making under uncertainty. Let's simulate the reliability of a steam turbine system with multiple components.

```python
import numpy as np
import matplotlib.pyplot as plt

def system_reliability(component_reliabilities):
    return np.prod(component_reliabilities)

# Set up the simulation
n_simulations = 10000
n_components = 5

# Run the Monte Carlo simulation
system_reliabilities = []
for _ in range(n_simulations):
    component_reliabilities = np.random.uniform(0.9, 0.99, n_components)
    system_reliabilities.append(system_reliability(component_reliabilities))

# Analyze the results
mean_reliability = np.mean(system_reliabilities)
std_reliability = np.std(system_reliabilities)

print(f"Mean system reliability: {mean_reliability:.4f}")
print(f"Standard deviation of system reliability: {std_reliability:.4f}")

# Visualize the distribution of system reliabilities
plt.figure(figsize=(10, 6))
plt.hist(system_reliabilities, bins=50, edgecolor='black')
plt.title('Distribution of Steam Turbine System Reliability')
plt.xlabel('System Reliability')
plt.ylabel('Frequency')
plt.axvline(mean_reliability, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_reliability:.4f}')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 13: Control Systems and Feedback Loops

Control systems are essential in Steam AI for maintaining optimal performance. Let's simulate a simple PID controller for regulating steam turbine speed.

```python
import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def simulate_turbine(initial_speed, setpoint, controller, simulation_time, dt):
    time = np.arange(0, simulation_time, dt)
    speed = np.zeros_like(time)
    speed[0] = initial_speed

    for i in range(1, len(time)):
        control_output = controller.update(setpoint, speed[i-1], dt)
        speed[i] = speed[i-1] + control_output * dt

    return time, speed

# Simulation parameters
controller = PIDController(Kp=0.5, Ki=0.1, Kd=0.1)
initial_speed, setpoint = 3000, 3600  # RPM
simulation_time, dt = 100, 0.1

time, speed = simulate_turbine(initial_speed, setpoint, controller, simulation_time, dt)

plt.figure(figsize=(10, 6))
plt.plot(time, speed, label='Turbine Speed')
plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
plt.title('Steam Turbine Speed Control with PID')
plt.xlabel('Time (s)')
plt.ylabel('Speed (RPM)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 14: Real-life Example: Steam Turbine Efficiency Optimization

In this example, we'll use machine learning to optimize steam turbine efficiency based on various operational parameters.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# Input features: steam pressure, temperature, flow rate, and ambient temperature
X = np.column_stack((
    np.random.uniform(10, 20, n_samples),  # Steam pressure (MPa)
    np.random.uniform(500, 600, n_samples),  # Steam temperature (°C)
    np.random.uniform(100, 200, n_samples),  # Steam flow rate (kg/s)
    np.random.uniform(10, 35, n_samples)  # Ambient temperature (°C)
))

# Target variable: turbine efficiency (%)
y = 85 + 2*X[:, 0] - 0.01*X[:, 1] + 0.05*X[:, 2] - 0.1*X[:, 3] + np.random.normal(0, 1, n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Feature importance
importance = model.feature_importance_
features = ['Steam Pressure', 'Steam Temperature', 'Flow Rate', 'Ambient Temperature']

plt.figure(figsize=(10, 6))
plt.bar(features, importance)
plt.title('Feature Importance for Steam Turbine Efficiency')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()
```

Slide 15: Real-life Example: Anomaly Detection in Steam Turbine Vibrations

In this example, we'll use statistical methods to detect anomalies in steam turbine vibration data, which can indicate potential mechanical issues.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate synthetic vibration data
np.random.seed(42)
n_samples = 1000
normal_vibrations = np.random.normal(0, 1, n_samples)

# Introduce anomalies
anomaly_indices = np.random.choice(n_samples, 50, replace=False)
anomalies = np.random.normal(5, 2, 50)
normal_vibrations[anomaly_indices] = anomalies

# Calculate z-scores
z_scores = np.abs(stats.zscore(normal_vibrations))

# Define threshold for anomalies
threshold = 3

# Detect anomalies
detected_anomalies = np.where(z_scores > threshold)[0]

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(normal_vibrations, label='Vibration Data')
plt.scatter(detected_anomalies, normal_vibrations[detected_anomalies], color='red', label='Detected Anomalies')
plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='green', linestyle='--')
plt.title('Anomaly Detection in Steam Turbine Vibrations')
plt.xlabel('Time')
plt.ylabel('Vibration Amplitude')
plt.legend()
plt.show()

print(f"Number of detected anomalies: {len(detected_anomalies)}")
```

Slide 16: Additional Resources

For further exploration of mathematics and statistics in Steam AI, consider the following resources:

1. ArXiv.org: "Machine Learning for Fluid Mechanics" ([https://arxiv.org/abs/1905.11075](https://arxiv.org/abs/1905.11075))
2. ArXiv.org: "Deep learning in fluid dynamics" ([https://arxiv.org/abs/1806.02071](https://arxiv.org/abs/1806.02071))
3. ArXiv.org: "Data-driven modeling and prediction of non-linear multiscale dynamical systems" ([https://arxiv.org/abs/1707.07998](https://arxiv.org/abs/1707.07998))

These papers provide in-depth discussions on advanced topics related to fluid dynamics and machine learning, which are highly relevant to Steam AI applications.

