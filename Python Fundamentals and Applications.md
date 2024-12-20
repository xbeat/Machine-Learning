## Python Fundamentals and Applications
Slide 1: Introduction to Advanced Time Series Analysis

Time series analysis is a crucial technique for understanding and predicting sequential data. This presentation explores advanced methods using TensorFlow, Fourier Transforms, and Cohomology Groups. We'll dive into practical examples and code snippets to illustrate these concepts.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate a sample time series
t = np.linspace(0, 10, 1000)
y = np.sin(t) + 0.1 * np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.title('Sample Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

Slide 2: TensorFlow for Time Series Analysis

TensorFlow provides powerful tools for building and training neural networks for time series forecasting. We'll use a simple LSTM model to demonstrate its capabilities.

```python
# Prepare data for LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_train, y_train = create_dataset(y, y, time_steps)

# Build LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# Make predictions
X_test = X_train[-100:]
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(y[-100:], label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('LSTM Time Series Prediction')
plt.show()
```

Slide 3: Fourier Transforms in Time Series Analysis

Fourier transforms are essential for analyzing the frequency components of a time series. We'll use the Fast Fourier Transform (FFT) to decompose our sample time series.

```python
# Perform FFT
fft_result = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(t), t[1] - t[0])

# Plot the frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(fft_result))
plt.title('Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.xlim(0, 5)  # Limit x-axis for better visualization
plt.show()

# Reconstruct the signal using inverse FFT
reconstructed_signal = np.fft.ifft(fft_result)

plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Original')
plt.plot(t, reconstructed_signal.real, label='Reconstructed')
plt.legend()
plt.title('Original vs Reconstructed Signal')
plt.show()
```

Slide 4: Introduction to Cohomology Groups

Cohomology groups are abstract algebraic structures used in topology and can be applied to time series analysis for detecting topological features. While not commonly used in traditional time series analysis, they offer a novel approach to understanding data structure.

```python
import networkx as nx

def create_time_series_graph(time_series, threshold):
    G = nx.Graph()
    for i in range(len(time_series)):
        for j in range(i+1, len(time_series)):
            if abs(time_series[i] - time_series[j]) < threshold:
                G.add_edge(i, j)
    return G

# Create a graph from our time series
graph = create_time_series_graph(y, threshold=0.1)

plt.figure(figsize=(10, 6))
nx.draw(graph, node_size=20, node_color='blue', with_labels=False)
plt.title('Time Series Graph')
plt.show()
```

Slide 5: Persistent Homology for Time Series

Persistent homology, a concept related to cohomology groups, can be used to analyze the topological structure of time series data. We'll use the ripser library to compute persistent homology.

```python
!pip install ripser

from ripser import ripser
from persim import plot_diagrams

# Compute persistent homology
diagrams = ripser(y.reshape(-1, 1))['dgms']

# Plot persistence diagram
plot_diagrams(diagrams, show=True)
plt.title('Persistence Diagram')
plt.show()
```

Slide 6: Wavelet Transform for Time Series Analysis

Wavelet transforms provide a multi-resolution analysis of time series, allowing us to capture both frequency and time information simultaneously.

```python
from pywt import wavedec

# Perform wavelet decomposition
coeffs = wavedec(y, 'db4', level=5)

# Plot wavelet coefficients
plt.figure(figsize=(12, 8))
for i, coeff in enumerate(coeffs):
    plt.subplot(len(coeffs), 1, i+1)
    plt.plot(coeff)
    plt.title(f'Wavelet Coefficients Level {i}')
plt.tight_layout()
plt.show()
```

Slide 7: Seasonal Decomposition of Time Series

Seasonal decomposition helps in understanding the underlying patterns in time series data by separating it into trend, seasonal, and residual components.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate a seasonal time series
t = np.linspace(0, 5, 1000)
seasonal = 2 * np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)
trend = 0.05 * t
noise = 0.1 * np.random.randn(1000)
y = seasonal + trend + noise

# Perform seasonal decomposition
result = seasonal_decompose(y, model='additive', period=200)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 8: Granger Causality in Time Series

Granger causality is a statistical concept used to determine whether one time series can be used to forecast another. It's particularly useful in analyzing relationships between multiple time series.

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Generate two related time series
np.random.seed(42)
x = np.cumsum(np.random.randn(1000))
y = np.roll(x, 5) + 0.1 * np.random.randn(1000)

# Perform Granger causality test
data = np.column_stack((x, y))
results = grangercausalitytests(data, maxlag=10, verbose=False)

# Print results for lag 5
print(f"Granger Causality Test Results (Lag 5):")
print(f"F-statistic: {results[5][0]['ssr_ftest'][0]:.4f}")
print(f"p-value: {results[5][0]['ssr_ftest'][1]:.4f}")
```

Slide 9: Dynamic Time Warping (DTW)

Dynamic Time Warping is an algorithm for measuring similarity between two temporal sequences, which may vary in speed. It's particularly useful for comparing time series with different lengths or phases.

```python
from dtaidistance import dtw
import numpy as np
import matplotlib.pyplot as plt

# Generate two similar but out-of-phase time series
t1 = np.linspace(0, 10, 100)
t2 = np.linspace(0, 10, 120)
s1 = np.sin(t1) + 0.1 * np.random.randn(100)
s2 = np.sin(t2 - 1) + 0.1 * np.random.randn(120)

# Compute DTW distance
distance = dtw.distance(s1, s2)

# Compute DTW path
path = dtw.warping_path(s1, s2)

# Plot the sequences and their alignment
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(s1, label='Series 1')
plt.plot(s2, label='Series 2')
plt.legend()
plt.title('Original Time Series')

plt.subplot(122)
for [i, j] in path:
    plt.plot([i, j], [s1[i], s2[j]], 'k-', alpha=0.1)
plt.plot(s1, label='Series 1')
plt.plot(s2, label='Series 2')
plt.legend()
plt.title(f'DTW Alignment (Distance: {distance:.2f})')

plt.tight_layout()
plt.show()
```

Slide 10: Recurrent Plot Analysis

Recurrence plots are a way to visualize the recurrence of states in a dynamical system. They can reveal hidden patterns and structures in time series data.

```python
from pyts.image import RecurrencePlot
import numpy as np
import matplotlib.pyplot as plt

# Generate a chaotic time series (Logistic Map)
def logistic_map(x0, r, n):
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

# Generate the time series
x0, r, n = 0.5, 3.8, 1000
time_series = logistic_map(x0, r, n)

# Create and fit the RecurrencePlot instance
rp = RecurrencePlot(threshold='point', percentage=20)
recurrence_plot = rp.fit_transform(time_series.reshape(1, -1))[0]

# Plot the recurrence plot
plt.figure(figsize=(10, 8))
plt.imshow(recurrence_plot, cmap='binary', origin='lower')
plt.title('Recurrence Plot of Logistic Map')
plt.xlabel('Time')
plt.ylabel('Time')
plt.colorbar(label='Recurrence')
plt.show()
```

Slide 11: Long Short-Term Memory (LSTM) Networks

LSTM networks are a type of recurrent neural network capable of learning long-term dependencies, making them particularly suitable for time series forecasting.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
y = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)

# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 20
X, y = create_dataset(y.reshape(-1, 1), look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions
y_pred = model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('LSTM Time Series Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()
```

Slide 12: Real-Life Example: Weather Forecasting

Weather forecasting is a classic application of time series analysis. We'll use historical temperature data to predict future temperatures.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Generate synthetic weather data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
temperatures = 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.randn(len(dates)) * 3
df = pd.DataFrame({'date': dates, 'temperature': temperatures})

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['temperature']])

# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 30
X, y = create_dataset(scaled_data, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(df['date'][-len(y_test_inv):], y_test_inv, label='Actual')
plt.plot(df['date'][-len(y_pred_inv):], y_pred_inv, label='Predicted')
plt.legend()
plt.title('Weather Forecast: Actual vs Predicted Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()
```

Slide 13: Real-Life Example: Earthquake Detection

Seismic time series analysis is crucial for earthquake detection and prediction. We'll use a simplified example to demonstrate how we might process and analyze seismic data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Generate synthetic seismic data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
background = np.sin(0.5 * t) + 0.5 * np.sin(1.5 * t)
earthquakes = np.zeros_like(t)
earthquake_times = [20, 50, 80]
for eq_time in earthquake_times:
    earthquakes += 5 * np.exp(-0.5 * (t - eq_time)**2 / 2**2)
noise = 0.5 * np.random.randn(len(t))
seismic_signal = background + earthquakes + noise

# Find peaks (potential earthquakes)
peaks, _ = find_peaks(seismic_signal, height=2, distance=50)

# Plot the seismic signal and detected peaks
plt.figure(figsize=(12, 6))
plt.plot(t, seismic_signal, label='Seismic Signal')
plt.plot(t[peaks], seismic_signal[peaks], "x", color='red', label='Detected Earthquakes')
plt.title('Seismic Signal Analysis')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

# Analyze frequency content
from scipy.fft import fft, fftfreq

N = len(t)
yf = fft(seismic_signal)
xf = fftfreq(N, t[1] - t[0])

plt.figure(figsize=(12, 6))
plt.plot(xf[:N//2], 2.0/N * np.abs(yf[:N//2]))
plt.title('Frequency Spectrum of Seismic Signal')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.xlim(0, 5)
plt.show()
```

Slide 14: Conclusion and Future Directions

We've explored various advanced techniques for time series analysis, including:

* TensorFlow for neural network-based forecasting
* Fourier transforms for frequency analysis
* Cohomology groups and persistent homology for topological features
* Wavelet transforms for multi-resolution analysis
* Seasonal decomposition
* Granger causality for relationship analysis
* Dynamic Time Warping for sequence comparison
* Recurrence plots for visualizing dynamical systems
* LSTM networks for long-term dependency learning

Future directions in time series analysis may include:

* Integration of machine learning with traditional statistical methods
* Exploration of more advanced topological data analysis techniques
* Development of interpretable AI models for time series forecasting
* Application of quantum computing algorithms to time series problems

As the field continues to evolve, these techniques will become increasingly important in various domains, from climate science to finance, healthcare, and beyond.

Slide 15: Additional Resources

For those interested in diving deeper into the topics covered, here are some valuable resources:

1. "Time Series Analysis and Its Applications" by Robert H. Shumway and David S. Stoffer ArXiv: [https://arxiv.org/abs/1802.07900](https://arxiv.org/abs/1802.07900)
2. "Forecasting: Principles and Practice" by Rob J. Hyndman and George Athanasopoulos Available online: [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
3. "Deep Learning for Time Series Forecasting" by Jason Brownlee Available as an e-book
4. "Topological Data Analysis for Scientific Visualization" by Julien Tierny ArXiv: [https://arxiv.org/abs/1709.03571](https://arxiv.org/abs/1709.03571)
5. "A Wavelet Tour of Signal Processing" by Stéphane Mallat Academic textbook

These resources provide in-depth explanations and advanced techniques for time series analysis, covering both theoretical foundations and practical applications.

