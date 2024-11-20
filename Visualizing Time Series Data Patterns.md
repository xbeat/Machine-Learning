## Visualizing Time Series Data Patterns
Slide 1: Time Series Data Preprocessing

Time series data often requires careful preprocessing before visualization or analysis. This includes handling missing values, resampling to ensure consistent intervals, and smoothing noisy data. Here we implement essential preprocessing functions for time series analysis.

```python
import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_timeseries(df, date_column, value_column):
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Handle missing values using forward fill
    df[value_column] = df[value_column].fillna(method='ffill')
    
    # Resample to regular intervals (daily)
    df = df.set_index(date_column)
    df = df.resample('D').mean()
    
    # Apply smoothing using rolling average
    df['smoothed'] = df[value_column].rolling(window=7).mean()
    
    return df

# Example usage
data = {
    'date': ['2023-01-01', '2023-01-03', '2023-01-04'],
    'value': [100, np.nan, 120]
}
df = pd.DataFrame(data)
processed_df = preprocess_timeseries(df, 'date', 'value')
print(processed_df.head())
```

Slide 2: Basic Time Series Visualization

Matplotlib and Seaborn provide powerful tools for visualizing time series data. This example demonstrates creating a comprehensive visualization with trend lines, confidence intervals, and key statistics.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_timeseries(df, value_column, title):
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.plot(df.index, df[value_column], 
             alpha=0.5, label='Raw Data')
    
    # Plot smoothed trend
    plt.plot(df.index, df['smoothed'], 
             'r-', label='7-day Moving Average')
    
    # Add confidence interval
    rolling_std = df[value_column].rolling(window=7).std()
    plt.fill_between(df.index,
                     df['smoothed'] - 2*rolling_std,
                     df['smoothed'] + 2*rolling_std,
                     alpha=0.2, color='r',
                     label='95% Confidence Interval')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt

# Example usage with previous preprocessed data
fig = visualize_timeseries(processed_df, 'value', 
                          'Time Series Analysis')
plt.show()
```

Slide 3: Detecting Seasonality

Seasonality analysis is crucial for understanding periodic patterns in time series data. This implementation uses both time domain and frequency domain approaches to identify seasonal components.

```python
def detect_seasonality(data, freq=7):
    """
    Detect seasonality using autocorrelation and spectral analysis
    """
    # Calculate autocorrelation
    autocorr = pd.Series(data).autocorr(lag=freq)
    
    # Perform spectral analysis using FFT
    fft = np.fft.fft(data)
    power = np.abs(fft)**2
    freqs = np.fft.fftfreq(len(data))
    
    # Find dominant frequencies
    dominant_freq = freqs[np.argmax(power[1:])+1]
    seasonal_period = int(1/abs(dominant_freq))
    
    return {
        'autocorrelation': autocorr,
        'seasonal_period': seasonal_period,
        'is_seasonal': autocorr > 0.7
    }

# Example with synthetic data
np.random.seed(42)
t = np.linspace(0, 365, 365)
seasonal = 10 * np.sin(2*np.pi*t/7) + np.random.normal(0, 1, 365)
results = detect_seasonality(seasonal)
print(f"Seasonality detected: {results['is_seasonal']}")
print(f"Period: {results['seasonal_period']} days")
```

Slide 4: Statistical Decomposition

Time series decomposition separates data into trend, seasonal, and residual components, enabling deeper analysis of each pattern independently. This implementation uses both additive and multiplicative decomposition methods.

```python
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

def advanced_decomposition(data, period=7, model='additive'):
    """
    Perform statistical decomposition of time series data
    """
    # Ensure data is regular and complete
    if isinstance(data, pd.Series):
        data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Perform decomposition
    decomposition = seasonal_decompose(data, 
                                     period=period, 
                                     model=model)
    
    # Calculate strength of seasonality
    seasonal_strength = 1 - (np.var(decomposition.resid) / 
                           np.var(decomposition.seasonal + decomposition.resid))
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'seasonal_strength': seasonal_strength
    }

# Example usage
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
data = pd.Series(np.random.normal(0, 1, 365) + \
                 10 * np.sin(np.arange(365) * 2 * np.pi / 7), 
                 index=dates)

results = advanced_decomposition(data)
print(f"Seasonal Strength: {results['seasonal_strength']:.2f}")
```

Slide 5: Trend Analysis and Forecasting

Advanced trend analysis combines statistical tests and forecasting techniques to identify significant trends and make predictions. This implementation includes Mann-Kendall test and prophet-style forecasting.

```python
from scipy import stats
import numpy as np

class TrendAnalyzer:
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(data)
    
    def mann_kendall_test(self):
        """
        Perform Mann-Kendall trend test
        """
        s = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                s += np.sign(self.data[j] - self.data[i])
        
        # Calculate variance
        var_s = (self.n * (self.n - 1) * (2 * self.n + 5)) / 18
        
        # Calculate Z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
            
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'statistic': s,
            'z_score': z,
            'p_value': p_value,
            'trend': 'increasing' if z > 0 else 'decreasing' if z < 0 else 'no trend'
        }
    
    def forecast_trend(self, horizon=30):
        """
        Simple trend forecasting using linear regression
        """
        X = np.arange(self.n).reshape(-1, 1)
        y = self.data
        
        # Fit linear regression
        slope, intercept = np.polyfit(X.flatten(), y, 1)
        
        # Generate forecast
        forecast_x = np.arange(self.n, self.n + horizon)
        forecast_y = slope * forecast_x + intercept
        
        return {
            'slope': slope,
            'intercept': intercept,
            'forecast': forecast_y
        }

# Example usage
data = np.cumsum(np.random.normal(0.1, 1, 100))  # Random walk with drift
analyzer = TrendAnalyzer(data)

trend_test = analyzer.mann_kendall_test()
forecast = analyzer.forecast_trend()

print(f"Trend Detection: {trend_test['trend']}")
print(f"Trend Slope: {forecast['slope']:.4f}")
```

Slide 6: Feature Engineering for Time Series

Feature engineering transforms raw time series data into meaningful features that capture temporal patterns and relationships. This implementation creates advanced time-based features for machine learning models.

```python
def engineer_time_features(df, date_column):
    """
    Create comprehensive time-based features for machine learning
    """
    # Ensure datetime
    df = df.copy()
    if not isinstance(df[date_column], pd.DatetimeIndex):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Basic time features
    df['year'] = df[date_column].year
    df['month'] = df[date_column].month
    df['day'] = df[date_column].day
    df['day_of_week'] = df[date_column].dayofweek
    df['day_of_year'] = df[date_column].dayofyear
    
    # Cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # Lag features
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Rolling statistics
    for window in [7, 30]:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
    
    return df

# Example usage
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
values = np.random.normal(100, 10, 365)
df = pd.DataFrame({'date': dates, 'value': values})

featured_df = engineer_time_features(df, 'date')
print("\nFeature Names:")
print(featured_df.columns.tolist())
print("\nSample Data:")
print(featured_df.head())
```

Slide 7: Anomaly Detection in Time Series

Anomaly detection identifies unusual patterns or outliers in time series data using statistical methods and machine learning approaches. This implementation combines multiple techniques for robust anomaly detection.

```python
class TimeSeriesAnomalyDetector:
    def __init__(self, data):
        self.data = np.array(data)
        self.mean = np.mean(data)
        self.std = np.std(data)
    
    def statistical_detection(self, threshold=3):
        """
        Z-score based anomaly detection
        """
        z_scores = np.abs((self.data - self.mean) / self.std)
        return z_scores > threshold
    
    def isolation_forest_detection(self, contamination=0.1):
        """
        Isolation Forest based anomaly detection
        """
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=contamination, 
                            random_state=42)
        
        # Reshape for sklearn
        X = self.data.reshape(-1, 1)
        
        # Fit and predict (-1 for anomalies, 1 for normal)
        predictions = clf.fit_predict(X)
        return predictions == -1
    
    def detect_anomalies(self, methods=['statistical', 'isolation']):
        results = {}
        
        if 'statistical' in methods:
            results['statistical'] = self.statistical_detection()
            
        if 'isolation' in methods:
            results['isolation'] = self.isolation_forest_detection()
            
        # Combine results (consider point anomalous if detected by any method)
        combined = np.zeros_like(self.data, dtype=bool)
        for method_results in results.values():
            combined = combined | method_results
            
        return {
            'detailed': results,
            'combined': combined,
            'indices': np.where(combined)[0]
        }

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
# Insert anomalies
normal_data[100] = 10
normal_data[500] = -8

detector = TimeSeriesAnomalyDetector(normal_data)
anomalies = detector.detect_anomalies()

print(f"Number of anomalies detected: {len(anomalies['indices'])}")
print(f"Anomaly indices: {anomalies['indices']}")
```

Slide 8: Real-time Time Series Processing

Real-time processing requires efficient algorithms for updating statistics and detecting patterns as new data arrives. This implementation provides a streaming time series processor with online learning capabilities.

```python
class StreamingTimeSeriesProcessor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffer = []
        self.online_mean = 0
        self.online_var = 0
        self.n = 0
        
    def update_statistics(self, new_value):
        """
        Update running statistics using Welford's algorithm
        """
        self.n += 1
        delta = new_value - self.online_mean
        self.online_mean += delta / self.n
        delta2 = new_value - self.online_mean
        self.online_var += delta * delta2
        
    def add_point(self, value, timestamp):
        """
        Process new data point
        """
        # Update buffer
        self.buffer.append((timestamp, value))
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
        # Update statistics
        self.update_statistics(value)
        
        # Calculate current metrics
        current_std = np.sqrt(self.online_var / self.n)
        is_anomaly = abs(value - self.online_mean) > 3 * current_std
        
        return {
            'mean': self.online_mean,
            'std': current_std,
            'is_anomaly': is_anomaly,
            'buffer_size': len(self.buffer)
        }

# Example usage
processor = StreamingTimeSeriesProcessor(window_size=50)

# Simulate streaming data
for i in range(200):
    # Generate normal data with occasional spikes
    if i % 50 == 0:
        value = np.random.normal(10, 1)  # Anomaly
    else:
        value = np.random.normal(0, 1)
        
    result = processor.add_point(value, 
                               timestamp=pd.Timestamp.now())
    
    if result['is_anomaly']:
        print(f"Anomaly detected at point {i}: {value:.2f}")
    
print(f"\nFinal statistics:")
print(f"Mean: {result['mean']:.2f}")
print(f"Std: {result['std']:.2f}")
```

Slide 9: Change Point Detection

Change point detection identifies significant shifts in time series behavior, crucial for understanding structural changes in the data. This implementation uses both statistical and algorithmic approaches to detect regime changes.

```python
class ChangePointDetector:
    def __init__(self, data):
        self.data = np.array(data)
        self.n = len(data)
        
    def cusum_detection(self, threshold=1.0):
        """
        Cumulative Sum (CUSUM) change point detection
        """
        mean = np.mean(self.data)
        std = np.std(self.data)
        
        # Standardize data
        s_pos = np.zeros(self.n)
        s_neg = np.zeros(self.n)
        
        # CUSUM recursion
        for i in range(1, self.n):
            s_pos[i] = max(0, s_pos[i-1] + 
                          (self.data[i] - mean)/std - threshold)
            s_neg[i] = max(0, s_neg[i-1] - 
                          (self.data[i] - mean)/std - threshold)
        
        # Detect change points
        change_points = np.where((s_pos > threshold) | 
                               (s_neg > threshold))[0]
        
        return change_points
    
    def binary_segmentation(self, min_size=30):
        """
        Binary segmentation for multiple change point detection
        """
        def calculate_contrast(start, end):
            if end - start < min_size:
                return -1, None
            
            n_points = end - start
            means = np.zeros(n_points)
            
            for i in range(start, end-1):
                left_mean = np.mean(self.data[start:i+1])
                right_mean = np.mean(self.data[i+1:end])
                means[i-start] = abs(left_mean - right_mean)
            
            max_idx = np.argmax(means)
            return means[max_idx], start + max_idx
        
        change_points = []
        segments = [(0, self.n)]
        
        while segments:
            start, end = segments.pop(0)
            contrast, cp = calculate_contrast(start, end)
            
            if contrast > 0:
                change_points.append(cp)
                segments.append((start, cp))
                segments.append((cp, end))
        
        return np.sort(change_points)

# Example usage
np.random.seed(42)
# Generate data with regime changes
data = np.concatenate([
    np.random.normal(0, 1, 200),
    np.random.normal(3, 1.5, 200),
    np.random.normal(-1, 1, 200)
])

detector = ChangePointDetector(data)
cusum_cp = detector.cusum_detection()
binary_cp = detector.binary_segmentation()

print("CUSUM Change Points:", cusum_cp[:5])
print("Binary Segmentation Change Points:", binary_cp)
```

Slide 10: Wavelet Analysis

Wavelet analysis decomposes time series into multiple frequency bands, enabling multi-scale analysis of temporal patterns. This implementation provides tools for continuous wavelet transform and spectral analysis.

```python
import pywt
import scipy.signal as signal

class WaveletAnalyzer:
    def __init__(self, data, sampling_rate=1.0):
        self.data = np.array(data)
        self.sampling_rate = sampling_rate
        
    def continuous_wavelet_transform(self, wavelet='cmor1.5-1.0', 
                                   scales=None):
        """
        Perform Continuous Wavelet Transform
        """
        if scales is None:
            scales = np.arange(1, min(len(self.data)//2, 128))
            
        # Compute CWT
        coef, freqs = pywt.cwt(self.data, 
                              scales, 
                              wavelet)
        
        # Calculate power spectrum
        power = np.abs(coef) ** 2
        
        return {
            'coefficients': coef,
            'frequencies': freqs,
            'power': power
        }
    
    def spectral_analysis(self, nperseg=256):
        """
        Compute spectral density and significant frequencies
        """
        freqs, times, Sxx = signal.spectrogram(
            self.data,
            fs=self.sampling_rate,
            nperseg=nperseg
        )
        
        # Find dominant frequencies
        mean_power = np.mean(Sxx, axis=1)
        dominant_freq_idx = np.argsort(mean_power)[-3:]
        dominant_freqs = freqs[dominant_freq_idx]
        
        return {
            'frequencies': freqs,
            'times': times,
            'power': Sxx,
            'dominant_frequencies': dominant_freqs
        }

# Example usage
# Generate test signal with multiple frequencies
t = np.linspace(0, 10, 1000)
signal = (np.sin(2*np.pi*2*t) + 
         0.5*np.sin(2*np.pi*10*t) + 
         0.25*np.random.randn(len(t)))

analyzer = WaveletAnalyzer(signal, sampling_rate=100)
cwt_results = analyzer.continuous_wavelet_transform()
spectral_results = analyzer.spectral_analysis()

print("Dominant Frequencies:", 
      spectral_results['dominant_frequencies'])
```

Slide 11: Advanced Forecasting with Neural Networks

Time series forecasting using deep learning combines LSTM networks with attention mechanisms for improved prediction accuracy. This implementation provides a comprehensive neural forecasting framework.

```python
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, 
                 output_dim, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Combine and predict
        output = self.dropout(attn_out.transpose(0, 1))
        return self.fc(output[:, -1, :])

def train_model(model, train_loader, val_loader, epochs=100):
    """
    Training loop with validation
    """
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item()
                
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, '
                  f'Val Loss = {val_loss:.4f}')

# Example usage
sequence_length = 50
input_dim = 1
hidden_dim = 64
num_layers = 2
output_dim = 1

model = TimeSeriesTransformer(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    output_dim=output_dim
)

# Example data preparation code
def prepare_data(data, seq_length):
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
        
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)
```

Slide 12: Real-world Application: Stock Market Analysis

This comprehensive example demonstrates time series analysis applied to stock market data, including preprocessing, feature engineering, and predictive modeling.

```python
class StockMarketAnalyzer:
    def __init__(self, data):
        self.data = data
        self.features = None
        self.model = None
        
    def preprocess(self):
        """
        Preprocess stock market data
        """
        df = self.data.copy()
        
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        self.features = df.dropna()
        return self.features
    
    def create_prediction_features(self, target_days=5):
        """
        Create features for prediction
        """
        df = self.features.copy()
        
        # Target variable (future returns)
        df['Target'] = df['Close'].shift(-target_days) / df['Close'] - 1
        
        # Additional features
        df['Returns'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Remove NaN values
        return df.dropna()
    
    def train_model(self, X, y):
        """
        Train prediction model
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        
        self.model.fit(X, y)
        return self.model
    
    def evaluate_predictions(self, X, y):
        """
        Evaluate model performance
        """
        from sklearn.metrics import mean_squared_error, r2_score
        
        predictions = self.model.predict(X)
        
        return {
            'MSE': mean_squared_error(y, predictions),
            'R2': r2_score(y, predictions),
            'Feature_Importance': dict(zip(
                X.columns, 
                self.model.feature_importances_
            ))
        }

# Example usage
# Assuming you have stock data in a pandas DataFrame
stock_data = pd.DataFrame({
    'Date': pd.date_range(start='2020-01-01', periods=500),
    'Close': np.random.normal(100, 10, 500).cumsum(),
    'Volume': np.random.randint(1000000, 5000000, 500)
})

analyzer = StockMarketAnalyzer(stock_data)
features = analyzer.preprocess()
prediction_data = analyzer.create_prediction_features()

# Split features and target
X = prediction_data.drop(['Target', 'Date', 'Close'], axis=1)
y = prediction_data['Target']

# Train and evaluate
model = analyzer.train_model(X, y)
results = analyzer.evaluate_predictions(X, y)

print("Model Performance:")
print(f"MSE: {results['MSE']:.6f}")
print(f"R2: {results['R2']:.6f}")
print("\nTop Features by Importance:")
for feat, imp in sorted(
    results['Feature_Importance'].items(), 
    key=lambda x: x[1], 
    reverse=True
)[:3]:
    print(f"{feat}: {imp:.4f}")
```

Slide 13: Complex Event Processing in Time Series

Complex event processing identifies patterns of events in real-time data streams, crucial for monitoring and alerting systems. This implementation provides a framework for detecting complex patterns in time series data.

```python
class ComplexEventProcessor:
    def __init__(self):
        self.patterns = {}
        self.event_buffer = []
        self.max_buffer_size = 1000
        
    def define_pattern(self, name, conditions):
        """
        Define a complex event pattern
        """
        self.patterns[name] = {
            'conditions': conditions,
            'window': conditions.get('window', 100),
            'threshold': conditions.get('threshold', 0.8)
        }
    
    def check_sequence_pattern(self, events, pattern):
        """
        Check if a sequence of events matches a pattern
        """
        matched = 0
        required = len(pattern['sequence'])
        
        for i, event in enumerate(events):
            if i + required > len(events):
                break
                
            match = True
            for j, condition in enumerate(pattern['sequence']):
                if not condition(events[i + j]):
                    match = False
                    break
                    
            if match:
                matched += 1
                
        return matched / (len(events) - required + 1) >= pattern['threshold']
    
    def process_event(self, event):
        """
        Process a new event and check for pattern matches
        """
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.max_buffer_size:
            self.event_buffer.pop(0)
            
        matches = {}
        for name, pattern in self.patterns.items():
            window = self.event_buffer[-pattern['window']:]
            matches[name] = self.check_sequence_pattern(window, pattern)
            
        return matches
    
    def create_alert(self, pattern_name, events):
        """
        Create alert for matched pattern
        """
        return {
            'pattern': pattern_name,
            'timestamp': pd.Timestamp.now(),
            'events': events,
            'severity': self.patterns[pattern_name].get('severity', 'medium')
        }

# Example usage
def price_spike(event):
    return event['price_change'] > 0.05

def volume_surge(event):
    return event['volume'] > event['avg_volume'] * 2

def price_reversal(event):
    return event['price_change'] < -0.03

# Initialize processor
processor = ComplexEventProcessor()

# Define patterns
processor.define_pattern('market_manipulation', {
    'sequence': [price_spike, volume_surge, price_reversal],
    'window': 5,
    'threshold': 0.8,
    'severity': 'high'
})

# Simulate event stream
events = []
for i in range(100):
    event = {
        'timestamp': pd.Timestamp.now() + pd.Timedelta(minutes=i),
        'price_change': np.random.normal(0, 0.02),
        'volume': np.random.normal(1000, 200),
        'avg_volume': 1000
    }
    
    # Inject pattern
    if i in [50, 51, 52]:
        event['price_change'] = 0.06 if i == 50 else \
                               0.02 if i == 51 else -0.04
        event['volume'] = 2500 if i == 51 else 1000
        
    matches = processor.process_event(event)
    
    if any(matches.values()):
        alert = processor.create_alert(
            next(name for name, matched in matches.items() if matched),
            events[-5:]
        )
        print(f"Alert generated: {alert['pattern']} "
              f"at {alert['timestamp']}")
```

Slide 14: Additional Resources

*   "Deep Learning for Time Series Forecasting"
    *   [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
*   "A Survey of Time Series Anomaly Detection Methods"
    *   [https://arxiv.org/abs/2009.09517](https://arxiv.org/abs/2009.09517)
*   "Attention-based Models for Time Series Prediction"
    *   [https://arxiv.org/abs/2101.02288](https://arxiv.org/abs/2101.02288)
*   Recommended search terms for further research:
    *   "Time series analysis methods in Python"
    *   "Neural networks for financial time series"
    *   "Complex event processing algorithms"
    *   "Wavelet analysis for time series"
*   Useful Python libraries:
    *   statsmodels
    *   prophet
    *   sktime
    *   tsfresh
    *   pymoo (for optimization)

