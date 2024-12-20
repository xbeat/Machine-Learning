## Time Series Analysis for Better Decisions
Slide 1: Time Series Fundamentals in Python

Time series analysis begins with understanding the basic components of temporal data structures. We'll explore how to create, manipulate and visualize time series data using pandas, focusing on essential data handling techniques for temporal analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a basic time series dataset
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.normal(loc=100, scale=10, size=len(dates))
ts_data = pd.Series(values, index=dates)

# Basic time series operations
print("Time Series Head:")
print(ts_data.head())
print("\nTime Series Resample (Monthly):")
print(ts_data.resample('M').mean())

# Visualization
plt.figure(figsize=(12, 6))
ts_data.plot(title='Daily Values Throughout 2023')
plt.grid(True)
plt.show()
```

Slide 2: Seasonal Decomposition Analysis

Understanding the underlying patterns in time series data requires decomposing it into trend, seasonal, and residual components. This implementation demonstrates STL decomposition using statsmodels for complex pattern identification.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate seasonal data
time_index = pd.date_range('2023-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365)/365)
noise = np.random.normal(0, 1, 365)
data = trend + seasonal + noise

# Create time series
ts = pd.Series(data, index=time_index)

# Perform decomposition
decomposition = seasonal_decompose(ts, period=365)

# Plot components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=ax1, title='Original')
decomposition.trend.plot(ax=ax2, title='Trend')
decomposition.seasonal.plot(ax=ax3, title='Seasonal')
decomposition.resid.plot(ax=ax4, title='Residual')
plt.tight_layout()
plt.show()
```

Slide 3: ARIMA Model Implementation

ARIMA (AutoRegressive Integrated Moving Average) models are fundamental for time series forecasting. This implementation shows how to identify model parameters, fit the model, and make predictions using a systematic approach.

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Generate sample data
np.random.seed(42)
n_points = 1000
ar_params = [0.7, -0.2]
ma_params = [0.2, 0.1]
ar = np.random.randn(n_points)
for i in range(2, n_points):
    ar[i] += ar_params[0] * ar[i-1] + ar_params[1] * ar[i-2]

# Create time series
dates = pd.date_range(start='2023-01-01', periods=n_points, freq='D')
ts_data = pd.Series(ar, index=dates)

# Fit ARIMA model
model = ARIMA(ts_data, order=(2, 0, 2))
results = model.fit()

# Make predictions
forecast = results.forecast(steps=30)
print("\nModel Summary:")
print(results.summary().tables[1])
print("\nForecast next 5 days:")
print(forecast[:5])
```

Slide 4: Advanced Time Series Preprocessing

Effective time series analysis requires robust preprocessing techniques. This implementation demonstrates handling missing values, resampling, and dealing with outliers using statistical methods.

```python
# Create sample time series with gaps and outliers
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
data = np.random.normal(100, 10, len(dates))
data[::10] = np.nan  # Create missing values
data[::20] = 200  # Create outliers

ts = pd.Series(data, index=dates)

def preprocess_timeseries(ts, outlier_threshold=3):
    # Handle missing values using forward fill and backward fill
    ts_cleaned = ts.copy()
    ts_cleaned = ts_cleaned.fillna(method='ffill').fillna(method='bfill')
    
    # Remove outliers using Z-score
    z_scores = np.abs((ts_cleaned - ts_cleaned.mean()) / ts_cleaned.std())
    ts_cleaned[z_scores > outlier_threshold] = np.nan
    ts_cleaned = ts_cleaned.interpolate(method='time')
    
    return ts_cleaned

# Apply preprocessing
ts_preprocessed = preprocess_timeseries(ts)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ts.plot(ax=ax1, title='Original Time Series with Gaps and Outliers')
ts_preprocessed.plot(ax=ax2, title='Preprocessed Time Series')
plt.tight_layout()
plt.show()
```

Slide 5: Prophet Model for Time Series Forecasting

Facebook's Prophet model excels at handling seasonality and holiday effects in time series data. This implementation shows how to use Prophet for robust forecasting with real-world considerations like holidays and special events.

```python
from prophet import Prophet
import pandas as pd

# Prepare data in Prophet format
df = pd.DataFrame({
    'ds': pd.date_range(start='2023-01-01', end='2023-12-31'),
    'y': np.random.normal(100, 10, 365) + \
         np.sin(np.linspace(0, 4*np.pi, 365)) * 10
})

# Add holiday effects
holidays = pd.DataFrame({
    'holiday': 'special_event',
    'ds': pd.to_datetime(['2023-07-04', '2023-12-25']),
    'lower_window': 0,
    'upper_window': 1
})

# Create and fit model
model = Prophet(holidays=holidays, 
               yearly_seasonality=True,
               weekly_seasonality=True,
               daily_seasonality=False)
model.fit(df)

# Make future predictions
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Prophet Forecast with Holiday Effects')
components = model.plot_components(forecast)
```

Slide 6: Dynamic Time Warping Implementation

Dynamic Time Warping (DTW) is crucial for comparing time series patterns regardless of speed variations. This implementation shows a from-scratch DTW algorithm with visualization capabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

def dtw(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                        dtw_matrix[i, j-1],    # deletion
                                        dtw_matrix[i-1, j-1])  # match
    
    # Backtrack to find the warping path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        possible_moves = [
            (i-1, j-1),  # diagonal
            (i-1, j),    # vertical
            (i, j-1)     # horizontal
        ]
        i, j = min(possible_moves, 
                  key=lambda x: dtw_matrix[x[0], x[1]])
    
    return dtw_matrix[n, m], path[::-1]

# Example usage
ts1 = np.sin(np.linspace(0, 4*np.pi, 100))
ts2 = np.sin(np.linspace(0, 4*np.pi, 120))

distance, path = dtw(ts1, ts2)

# Visualization
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(ts1, label='Series 1')
plt.plot(ts2, label='Series 2')
plt.legend()
plt.title(f'Original Series (DTW Distance: {distance:.2f})')

plt.subplot(212)
plt.plot([p[0] for p in path], [p[1] for p in path], 'r-')
plt.xlabel('Series 1')
plt.ylabel('Series 2')
plt.title('DTW Warping Path')
plt.tight_layout()
plt.show()
```

Slide 7: LSTM for Time Series Prediction

Long Short-Term Memory networks are powerful for capturing long-term dependencies in time series data. This implementation demonstrates a complete LSTM model for multivariate time series forecasting.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Generate multivariate time series data
def generate_data(n_samples=1000):
    t = np.linspace(0, 100, n_samples)
    x1 = np.sin(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    x2 = np.cos(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    y = np.sin(0.1 * (t + 5))  # Future values
    return np.column_stack([x1, x2]), y

# Prepare data
X, y = generate_data()
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Create sequences
def create_sequences(X, y, seq_length=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled)

# Build and train LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2]), 
         return_sequences=True),
    Dropout(0.2),
    LSTM(30, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(X_seq, y_seq, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=1)

# Make predictions
predictions = model.predict(X_seq)
predictions = scaler_y.inverse_transform(predictions)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(scaler_y.inverse_transform(y_seq), label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('LSTM Time Series Prediction')
plt.show()
```

Slide 8: Time Series Feature Engineering

Advanced feature engineering techniques specific to temporal data can significantly improve model performance. This implementation demonstrates creating lag features, rolling statistics, and time-based features.

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Create sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.normal(100, 10, len(dates)) + \
             np.sin(np.linspace(0, 8*np.pi, len(dates))) * 20
})

def create_features(df, target_col, lags=[1, 24, 168]):
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    windows = [6, 12, 24]
    for window in windows:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window).max()
    
    # Custom features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    return df

# Apply feature engineering
df_featured = create_features(df, 'value')

# Display results
print("Original features shape:", df.shape)
print("Enhanced features shape:", df_featured.shape)
print("\nNew features created:")
print(df_featured.columns.tolist())

# Correlation analysis
correlation_matrix = df_featured.corr()['value'].sort_values(ascending=False)
print("\nTop 5 correlated features:")
print(correlation_matrix[:5])
```

Slide 9: Fourier Transform Analysis for Time Series

Spectral analysis using Fourier transforms helps identify periodic patterns and dominant frequencies in time series data. This implementation shows how to perform and visualize frequency domain analysis.

```python
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def spectral_analysis(time_series, sampling_rate):
    # Compute FFT
    n = len(time_series)
    fft_values = fft(time_series)
    frequencies = fftfreq(n, 1/sampling_rate)
    
    # Compute power spectrum
    power_spectrum = np.abs(fft_values)**2
    
    # Find dominant frequencies
    dominant_freqs = frequencies[np.argsort(power_spectrum)[-5:]]
    
    return frequencies, power_spectrum, dominant_freqs

# Generate sample data with multiple frequencies
t = np.linspace(0, 100, 1000)
signal = (np.sin(2*np.pi*0.1*t) + 
         0.5*np.sin(2*np.pi*0.05*t) + 
         0.3*np.sin(2*np.pi*0.02*t) + 
         np.random.normal(0, 0.1, len(t)))

# Perform spectral analysis
freq, power, dom_freq = spectral_analysis(signal, sampling_rate=10)

# Visualization
plt.figure(figsize=(15, 10))

# Original signal
plt.subplot(211)
plt.plot(t, signal)
plt.title('Original Time Series')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Power spectrum
plt.subplot(212)
plt.plot(freq[:len(freq)//2], power[:len(freq)//2])
plt.title('Power Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Power')

# Mark dominant frequencies
for f in dom_freq:
    if f > 0:
        plt.axvline(x=f, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

print("Dominant frequencies found:", dom_freq[dom_freq > 0])
```

Slide 10: Wavelet Transform Analysis

Wavelet transforms provide time-frequency localization, enabling analysis of non-stationary signals and detection of local patterns. This implementation demonstrates continuous wavelet transform analysis using PyWavelets.

```python
import pywt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def wavelet_analysis(signal, scales, wavelet='cmor1.5-1.0'):
    # Perform continuous wavelet transform
    coef, freqs = pywt.cwt(signal, scales, wavelet)
    
    # Calculate wavelet power spectrum
    power = (abs(coef)) ** 2
    
    return coef, freqs, power

# Generate sample signal with varying frequency
t = np.linspace(0, 1, 1000)
freq = np.zeros_like(t)
freq[t < 0.3] = 10
freq[t >= 0.3] = 25
freq[t >= 0.6] = 50
signal = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.1, len(t))

# Perform wavelet transform
scales = np.arange(1, 128)
coef, freqs, power = wavelet_analysis(signal, scales)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Original signal
ax1.plot(t, signal)
ax1.set_title('Original Signal')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')

# Wavelet transform
im = ax2.imshow(power, aspect='auto', cmap='jet',
                extent=[t[0], t[-1], freqs[-1], freqs[0]])
ax2.set_title('Wavelet Transform Power Spectrum')
ax2.set_xlabel('Time')
ax2.set_ylabel('Frequency')

# Add colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

plt.tight_layout()
plt.show()

# Extract significant features
power_threshold = np.percentile(power, 95)
significant_times = t[np.any(power > power_threshold, axis=0)]
print(f"Significant time points detected: {len(significant_times)}")
```

Slide 11: Kalman Filter Implementation

Kalman filtering provides optimal state estimation for time series with noise and uncertainty. This implementation shows a basic Kalman filter for time series smoothing and prediction.

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = 1.0
        
    def update(self, measurement):
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate

# Generate noisy data
true_signal = np.sin(np.linspace(0, 4*np.pi, 100))
measurements = true_signal + np.random.normal(0, 0.3, size=len(true_signal))

# Apply Kalman filtering
kf = KalmanFilter(process_variance=0.001, measurement_variance=0.1)
filtered_values = []
for measurement in measurements:
    filtered_values.append(kf.update(measurement))

# Calculate metrics
mse_original = np.mean((measurements - true_signal)**2)
mse_filtered = np.mean((filtered_values - true_signal)**2)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(true_signal, 'g-', label='True Signal')
plt.plot(measurements, 'r.', label='Noisy Measurements')
plt.plot(filtered_values, 'b-', label='Kalman Filter Estimate')
plt.title('Kalman Filter for Time Series Smoothing')
plt.legend()
plt.grid(True)

print(f"Original MSE: {mse_original:.4f}")
print(f"Filtered MSE: {mse_filtered:.4f}")

plt.show()
```

Slide 12: Real-world Application: Energy Consumption Forecasting

This implementation demonstrates a complete pipeline for energy consumption forecasting using multiple models and incorporating external factors like weather and time patterns.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Generate synthetic energy consumption data
def generate_energy_data(days=365):
    date_rng = pd.date_range(start='2023-01-01', periods=days*24, freq='H')
    
    # Base load pattern
    hour_pattern = np.sin(np.pi * np.arange(24) / 12) + 1
    base_load = np.tile(hour_pattern, days)
    
    # Seasonal pattern
    seasonal = np.sin(2 * np.pi * np.arange(len(date_rng)) / (365 * 24)) * 0.5
    
    # Temperature effect
    temp = 20 + 10 * np.sin(2 * np.pi * np.arange(len(date_rng)) / (365 * 24))
    temp_effect = 0.1 * (temp - 20)**2
    
    # Combine patterns with noise
    consumption = base_load + seasonal + temp_effect + np.random.normal(0, 0.1, len(date_rng))
    
    df = pd.DataFrame({
        'timestamp': date_rng,
        'consumption': consumption,
        'temperature': temp
    })
    return df

# Feature engineering for energy data
def create_energy_features(df):
    df = df.copy()
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Lag features
    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        df[f'consumption_lag_{lag}'] = df['consumption'].shift(lag)
    
    # Rolling statistics
    for window in [24, 168]:
        df[f'consumption_rolling_mean_{window}'] = df['consumption'].rolling(window).mean()
        df[f'consumption_rolling_std_{window}'] = df['consumption'].rolling(window).std()
    
    return df

# Create and prepare dataset
df = generate_energy_data()
df_featured = create_energy_features(df)
df_featured = df_featured.dropna()

# Prepare data for modeling
features = ['hour', 'dayofweek', 'month', 'is_weekend', 'temperature',
           'consumption_lag_1', 'consumption_lag_24', 'consumption_lag_168',
           'consumption_rolling_mean_24', 'consumption_rolling_std_24']
X = df_featured[features].values
y = df_featured['consumption'].values

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Reshape data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Train model
history = model.fit(X_train_reshaped, y_train_scaled, 
                   epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions
predictions_scaled = model.predict(X_test_reshaped)
predictions = scaler_y.inverse_transform(predictions_scaled)

# Calculate metrics
mape = mean_absolute_percentage_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAPE: {mape*100:.2f}%")
print(f"RMSE: {rmse:.2f}")

# Visualization
plt.figure(figsize=(15, 6))
plt.plot(y_test[:168], label='Actual')
plt.plot(predictions[:168], label='Predicted')
plt.title('Energy Consumption Forecast (1 Week)')
plt.xlabel('Hours')
plt.ylabel('Consumption')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 13: Real-world Application: Stock Market Technical Analysis

This implementation demonstrates a comprehensive technical analysis system for financial time series, including multiple technical indicators and trading signal generation.

```python
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import find_peaks

class TechnicalAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
    
    def add_moving_averages(self, windows=[20, 50, 200]):
        for window in windows:
            self.data[f'MA_{window}'] = self.data['Close'].rolling(window=window).mean()
    
    def add_bollinger_bands(self, window=20, std_dev=2):
        self.data['BB_middle'] = self.data['Close'].rolling(window=window).mean()
        rolling_std = self.data['Close'].rolling(window=window).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (rolling_std * std_dev)
        self.data['BB_lower'] = self.data['BB_middle'] - (rolling_std * std_dev)
    
    def add_rsi(self, period=14):
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
    
    def add_macd(self, fast=12, slow=26, signal=9):
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=signal).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal_Line']
    
    def generate_signals(self):
        # RSI signals
        self.data['RSI_Signal'] = 0
        self.data.loc[self.data['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold
        self.data.loc[self.data['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought
        
        # MACD signals
        self.data['MACD_Signal'] = 0
        self.data.loc[self.data['MACD'] > self.data['Signal_Line'], 'MACD_Signal'] = 1
        self.data.loc[self.data['MACD'] < self.data['Signal_Line'], 'MACD_Signal'] = -1
        
        # Bollinger Bands signals
        self.data['BB_Signal'] = 0
        self.data.loc[self.data['Close'] < self.data['BB_lower'], 'BB_Signal'] = 1
        self.data.loc[self.data['Close'] > self.data['BB_upper'], 'BB_Signal'] = -1
        
        # Combined signal
        self.data['Combined_Signal'] = (self.data['RSI_Signal'] + 
                                      self.data['MACD_Signal'] + 
                                      self.data['BB_Signal'])
    
    def analyze(self):
        self.add_moving_averages()
        self.add_bollinger_bands()
        self.add_rsi()
        self.add_macd()
        self.generate_signals()
        return self.data

# Example usage with visualization
def plot_analysis(data):
    plt.figure(figsize=(15, 12))
    
    # Price and MA
    plt.subplot(411)
    plt.plot(data['Close'], label='Close')
    plt.plot(data['MA_20'], label='MA 20')
    plt.plot(data['MA_50'], label='MA 50')
    plt.plot(data['BB_upper'], 'r--', label='BB Upper')
    plt.plot(data['BB_lower'], 'r--', label='BB Lower')
    plt.title('Price Action with Technical Indicators')
    plt.legend()
    
    # RSI
    plt.subplot(412)
    plt.plot(data['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI')
    plt.legend()
    
    # MACD
    plt.subplot(413)
    plt.plot(data['MACD'], label='MACD')
    plt.plot(data['Signal_Line'], label='Signal Line')
    plt.bar(data.index, data['MACD_Histogram'], label='MACD Histogram')
    plt.title('MACD')
    plt.legend()
    
    # Combined Signal
    plt.subplot(414)
    plt.plot(data['Combined_Signal'], label='Combined Signal')
    plt.title('Combined Trading Signal')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Generate sample data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
data = pd.DataFrame({
    'Close': prices,
    'Volume': np.random.randint(1000000, 10000000, len(dates))
}, index=dates)

# Perform analysis
analyzer = TechnicalAnalyzer(data)
analyzed_data = analyzer.analyze()
plot_analysis(analyzed_data)

# Print trading statistics
signal_stats = pd.DataFrame({
    'Buy Signals': (analyzed_data['Combined_Signal'] > 0).sum(),
    'Sell Signals': (analyzed_data['Combined_Signal'] < 0).sum(),
    'Neutral Signals': (analyzed_data['Combined_Signal'] == 0).sum()
}, index=['Count'])

print("\nTrading Signal Statistics:")
print(signal_stats)
```

Slide 14: Bayesian Structural Time Series Analysis

Bayesian structural time series models provide a flexible framework for analyzing and forecasting time series data with uncertainty quantification. This implementation demonstrates model composition and inference.

```python
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy import stats

def create_structural_model(data, seasons=12):
    with pm.Model() as model:
        # Prior for observation noise
        σ_obs = pm.HalfNormal('σ_obs', sigma=1)
        
        # Level component
        σ_level = pm.HalfNormal('σ_level', sigma=1)
        level = pm.GaussianRandomWalk('level', 
                                     sigma=σ_level,
                                     shape=len(data))
        
        # Seasonal component
        σ_seasonal = pm.HalfNormal('σ_seasonal', sigma=1)
        seasonal_raw = pm.Normal('seasonal_raw', 
                               mu=0, 
                               sigma=σ_seasonal,
                               shape=seasons)
        seasonal = pm.Deterministic('seasonal',
                                  seasonal_raw - pm.math.mean(seasonal_raw))
        
        # Cycle through seasonal effects
        seasonal_pattern = pm.Deterministic(
            'seasonal_pattern',
            pm.math.tile(seasonal, len(data)//seasons + 1)[:len(data)]
        )
        
        # Combine components
        μ = level + seasonal_pattern
        
        # Likelihood
        y = pm.Normal('y', 
                     mu=μ,
                     sigma=σ_obs,
                     observed=data)
        
    return model

# Generate sample data
np.random.seed(42)
n_points = 144  # 12 years of monthly data
t = np.arange(n_points)

# True components
trend = 0.1 * t
seasonal = 5 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 1, n_points)

# Combine components
y = trend + seasonal + noise

# Fit model
with create_structural_model(y) as model:
    # Inference
    trace = pm.sample(2000, 
                     tune=1000, 
                     return_inferencedata=True,
                     target_accept=0.9)

# Extract components
level_samples = trace.posterior['level'].mean(dim=['chain', 'draw']).values
seasonal_pattern = trace.posterior['seasonal_pattern'].mean(dim=['chain', 'draw']).values

# Calculate prediction intervals
level_hpd = pm.hdi(trace.posterior['level'])
y_pred = trace.posterior['level'] + trace.posterior['seasonal_pattern']
y_hpd = pm.hdi(y_pred)

# Visualization
plt.figure(figsize=(15, 10))

# Original data and fit
plt.subplot(311)
plt.plot(t, y, 'k.', label='Observed')
plt.plot(t, level_samples + seasonal_pattern, 'r-', label='Fitted')
plt.fill_between(t, y_hpd[0], y_hpd[1], color='r', alpha=0.2)
plt.title('Observed Data and Model Fit')
plt.legend()

# Level component
plt.subplot(312)
plt.plot(t, level_samples, 'b-', label='Level')
plt.fill_between(t, level_hpd[0], level_hpd[1], color='b', alpha=0.2)
plt.title('Level Component')
plt.legend()

# Seasonal component
plt.subplot(313)
plt.plot(t, seasonal_pattern, 'g-', label='Seasonal')
plt.title('Seasonal Component')
plt.legend()

plt.tight_layout()
plt.show()

# Print model diagnostics
print("\nModel Diagnostics:")
print(f"Number of divergences: {trace['diverging'].sum()}")
print(f"Mean acceptance rate: {trace['accept_stat'].mean():.2f}")
```

Slide 15: Additional Resources

1.  "Deep Learning for Time Series Forecasting: A Survey" [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
2.  "Probabilistic Forecasting with Temporal Convolutional Neural Networks" [https://arxiv.org/abs/1906.04397](https://arxiv.org/abs/1906.04397)
3.  "A Review of Trends and Perspectives in Time Series Forecasting" [https://arxiv.org/abs/2103.12057](https://arxiv.org/abs/2103.12057)
4.  "Bayesian Structural Time Series Models" [https://arxiv.org/abs/1802.05692](https://arxiv.org/abs/1802.05692)
5.  "Time Series Analysis Using Neural Networks: A Survey" [https://arxiv.org/abs/2101.02118](https://arxiv.org/abs/2101.02118)

