## Top 4 Effective Noise Filtering Techniques for Time Series
Slide 1: Mean Filter Implementation

The mean filter is a fundamental smoothing technique that operates by replacing each data point with the average of its neighboring values within a specified window size. This method effectively reduces random noise while preserving underlying trends.

```python
import numpy as np
import matplotlib.pyplot as plt

def mean_filter(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Example usage
np.random.seed(42)
t = np.linspace(0, 10, 1000)
signal = np.sin(t) + np.random.normal(0, 0.2, len(t))
filtered = mean_filter(signal, window_size=20)

plt.figure(figsize=(10, 6))
plt.plot(t[:len(filtered)], signal[:len(filtered)], 'b-', alpha=0.5, label='Noisy')
plt.plot(t[:len(filtered)], filtered, 'r-', label='Filtered')
plt.legend()
```

Slide 2: Median Filter for Outlier Removal

The median filter excels at removing spike noise and outliers while preserving edge information better than the mean filter. It's particularly effective for dealing with impulse noise in time series data.

```python
def median_filter(data, window_size):
    result = np.zeros_like(data)
    half_window = window_size // 2
    
    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)
        result[i] = np.median(data[start_idx:end_idx])
    
    return result

# Example with outliers
signal_with_outliers = signal.copy()
outlier_positions = np.random.randint(0, len(signal), 50)
signal_with_outliers[outlier_positions] = 5
filtered_median = median_filter(signal_with_outliers, window_size=5)
```

Slide 3: Exponential Smoothing

Exponential smoothing assigns exponentially decreasing weights to observations, giving more importance to recent data points. This method is particularly useful for time series with trends and seasonal patterns.

```python
def exponential_smoothing(data, alpha):
    """
    $$y_t = \alpha x_t + (1-\alpha)y_{t-1}$$
    where alpha is the smoothing factor (0 < alpha < 1)
    """
    result = np.zeros_like(data)
    result[0] = data[0]
    
    for t in range(1, len(data)):
        result[t] = alpha * data[t] + (1 - alpha) * result[t-1]
    
    return result

# Example usage
exp_smoothed = exponential_smoothing(signal, alpha=0.1)
```

Slide 4: Gaussian Filter Implementation

Gaussian filtering applies a weighted average where the weights follow a Gaussian distribution, providing smooth noise reduction while preserving important signal features. This method is particularly effective for continuous signals with normally distributed noise.

```python
def gaussian_filter(data, sigma, window_size=None):
    if window_size is None:
        window_size = int(6 * sigma)  # 3 sigma on each side
    
    x = np.linspace(-3, 3, window_size)
    gaussian_weights = np.exp(-x**2 / (2 * sigma**2))
    gaussian_weights /= np.sum(gaussian_weights)
    
    return np.convolve(data, gaussian_weights, mode='valid')

# Example implementation
filtered_gaussian = gaussian_filter(signal, sigma=1.0, window_size=21)
```

Slide 5: Real-World Application - Stock Price Smoothing

A practical application of time series filtering in financial analysis, demonstrating the effectiveness of different filtering methods on daily stock price data to identify underlying trends.

```python
import yfinance as yf
import pandas as pd

# Download sample stock data
stock_data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
prices = stock_data['Close'].values

# Apply different filters
mean_filtered = mean_filter(prices, window_size=10)
median_filtered = median_filter(prices, window_size=10)
exp_filtered = exponential_smoothing(prices, alpha=0.1)
gauss_filtered = gaussian_filter(prices, sigma=2.0)

# Calculate performance metrics
def calculate_metrics(original, filtered):
    mse = np.mean((original[:len(filtered)] - filtered)**2)
    mae = np.mean(np.abs(original[:len(filtered)] - filtered))
    return {'MSE': mse, 'MAE': mae}
```

Slide 6: Handling Missing Data in Time Series

Missing data handling is crucial in real-world time series analysis. This implementation demonstrates robust filtering techniques that account for NaN values and data gaps.

```python
def robust_filter(data, window_size, method='median'):
    data = np.array(data)
    result = np.full_like(data, np.nan, dtype=float)
    
    for i in range(len(data)):
        window = data[max(0, i-window_size//2):min(len(data), i+window_size//2+1)]
        valid_data = window[~np.isnan(window)]
        
        if len(valid_data) > 0:
            if method == 'median':
                result[i] = np.median(valid_data)
            elif method == 'mean':
                result[i] = np.mean(valid_data)
                
    return result

# Example with missing data
data_with_gaps = signal.copy()
data_with_gaps[np.random.choice(len(signal), 100)] = np.nan
filtered_robust = robust_filter(data_with_gaps, window_size=5)
```

Slide 7: Kalman Filter for Time Series

Kalman filtering provides optimal estimates of states in linear dynamic systems with Gaussian noise. This recursive filter combines predictions with measurements to minimize mean square error.

```python
def kalman_filter(measurements, process_variance, measurement_variance):
    """
    $$x_t = Ax_{t-1} + w_t$$
    $$z_t = Hx_t + v_t$$
    """
    n_measurements = len(measurements)
    x_hat = np.zeros(n_measurements)
    p = np.zeros(n_measurements)
    
    # Initialize
    x_hat[0] = measurements[0]
    p[0] = 1.0
    
    for t in range(1, n_measurements):
        # Predict
        x_hat_minus = x_hat[t-1]
        p_minus = p[t-1] + process_variance
        
        # Update
        k = p_minus / (p_minus + measurement_variance)
        x_hat[t] = x_hat_minus + k * (measurements[t] - x_hat_minus)
        p[t] = (1 - k) * p_minus
        
    return x_hat

# Example usage
noisy_data = signal + np.random.normal(0, 0.5, len(signal))
kalman_filtered = kalman_filter(noisy_data, 0.1, 1.0)
```

Slide 8: Savitzky-Golay Filter Implementation

The Savitzky-Golay filter performs local polynomial regression on a series of values to determine the smoothed value for each point, preserving higher moments of the data.

```python
from scipy.signal import savgol_filter

def custom_savgol_filter(data, window_size, poly_order):
    if window_size % 2 == 0:
        window_size += 1
    
    filtered = savgol_filter(data, window_size, poly_order)
    
    # Calculate error metrics
    mse = np.mean((data - filtered)**2)
    rmse = np.sqrt(mse)
    
    return filtered, {'MSE': mse, 'RMSE': rmse}

# Example usage
sg_filtered, metrics = custom_savgol_filter(signal, window_size=21, poly_order=3)
```

Slide 9: Adaptive Filtering

Adaptive filtering adjusts filter parameters based on the signal characteristics, making it particularly useful for non-stationary time series data.

```python
def adaptive_mean_filter(data, initial_window_size, threshold):
    result = np.zeros_like(data)
    window_sizes = np.zeros_like(data)
    
    for i in range(len(data)):
        window_size = initial_window_size
        while True:
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(data), i + window_size//2 + 1)
            window = data[start_idx:end_idx]
            
            if np.std(window) < threshold or window_size >= len(data):
                result[i] = np.mean(window)
                window_sizes[i] = window_size
                break
            
            window_size += 2
            
    return result, window_sizes

# Example usage
adaptive_filtered, used_windows = adaptive_mean_filter(signal, 5, 0.5)
```

Slide 10: Real-Time Filtering Pipeline

This implementation demonstrates a complete real-time filtering pipeline that processes streaming data, handles outliers, and applies adaptive filtering techniques for optimal noise reduction.

```python
class RealTimeFilter:
    def __init__(self, buffer_size=100):
        self.buffer = np.zeros(buffer_size)
        self.position = 0
        self.buffer_size = buffer_size
        
    def update(self, new_value):
        self.buffer[self.position % self.buffer_size] = new_value
        self.position += 1
        
    def get_filtered_value(self, method='median', window_size=5):
        start_idx = max(0, self.position - window_size)
        current_buffer = self.buffer[start_idx:self.position]
        
        if method == 'median':
            return np.median(current_buffer)
        elif method == 'mean':
            return np.mean(current_buffer)
        elif method == 'exp':
            return exponential_smoothing(current_buffer, 0.1)[-1]
            
# Example usage
rt_filter = RealTimeFilter()
filtered_stream = []

for value in signal:
    rt_filter.update(value)
    filtered_stream.append(rt_filter.get_filtered_value())
```

Slide 11: Wavelet Denoising

Wavelet denoising provides multi-resolution analysis capabilities, effectively separating noise from signal components at different frequency scales.

```python
import pywt

def wavelet_denoise(data, wavelet='db4', level=1):
    """
    $$y[n] = \sum_{k} w_{j,k} \psi_{j,k}[n]$$
    """
    # Decompose signal
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # Threshold determination
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # Apply threshold
    coeffs_thresholded = list(coeffs)
    for i in range(1, len(coeffs)):
        coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold)
    
    # Reconstruct signal
    return pywt.waverec(coeffs_thresholded, wavelet)

# Example usage
denoised = wavelet_denoise(signal, level=3)
```

Slide 12: Frequency Domain Filtering

Frequency domain filtering transforms time series data using FFT, applies filtering in frequency space, and transforms back to time domain for effective noise reduction across different frequency bands.

```python
def frequency_domain_filter(data, cutoff_freq, sampling_rate):
    """
    $$X(f) = \int_{-\infty}^{\infty} x(t)e^{-2\pi ift}dt$$
    """
    # FFT transform
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
    
    # Create mask for frequencies
    mask = np.abs(freqs) <= cutoff_freq
    filtered_fft = fft_data * mask
    
    # Inverse FFT
    return np.real(np.fft.ifft(filtered_fft))

# Example usage
sampling_rate = 100  # Hz
cutoff = 10  # Hz
freq_filtered = frequency_domain_filter(signal, cutoff, sampling_rate)
```

Slide 13: Performance Metrics and Filter Comparison

A comprehensive comparison framework for evaluating different filtering methods using multiple performance metrics and visualization techniques.

```python
def compare_filters(original_signal, noisy_signal, window_size=20):
    # Apply different filters
    filters = {
        'Mean': mean_filter(noisy_signal, window_size),
        'Median': median_filter(noisy_signal, window_size),
        'Gaussian': gaussian_filter(noisy_signal, sigma=1.0),
        'Exponential': exponential_smoothing(noisy_signal, alpha=0.1),
        'Wavelet': wavelet_denoise(noisy_signal)
    }
    
    # Calculate metrics
    metrics = {}
    for name, filtered in filters.items():
        valid_length = min(len(original_signal), len(filtered))
        metrics[name] = {
            'MSE': np.mean((original_signal[:valid_length] - filtered[:valid_length])**2),
            'MAE': np.mean(np.abs(original_signal[:valid_length] - filtered[:valid_length])),
            'PSNR': 10 * np.log10(1 / np.mean((original_signal[:valid_length] - filtered[:valid_length])**2))
        }
    
    return filters, metrics

# Example usage and visualization
filters_result, metrics_result = compare_filters(signal, signal + np.random.normal(0, 0.2, len(signal)))
```

Slide 14: Additional Resources

*   "A Review of Time Series Smoothing Methods" - arXiv:2104.xxxxx
*   "Adaptive Filtering Techniques for Non-stationary Data" - [https://doi.org/10.1016/j.sigpro.2019.xxx](https://doi.org/10.1016/j.sigpro.2019.xxx)
*   "Modern Approaches to Time Series Filtering" - arXiv:2103.xxxxx
*   Search terms for further research:
    *   "Robust time series filtering techniques"
    *   "Advanced signal processing methods"
    *   "Real-time data filtering algorithms"

