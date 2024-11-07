## Smoothing Time Series Data with Gaussian Filters
Slide 1: Gaussian Filter Fundamentals

The Gaussian filter is a crucial tool in time series analysis that applies a weighted moving average using the normal distribution. It effectively reduces noise while preserving underlying trends by giving more weight to nearby points and less weight to distant ones.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    """Generate 1D Gaussian kernel"""
    x = np.linspace(-size//2, size//2, size)
    kernel = np.exp(-x**2/(2*sigma**2))
    return kernel/kernel.sum()

# Example kernel visualization
kernel = gaussian_kernel(21, 3)
plt.plot(kernel)
plt.title('Gaussian Kernel (σ=3)')
plt.show()
```

Slide 2: Basic Time Series Generation

To demonstrate Gaussian filtering, we first generate a synthetic time series with added noise. This creates a controlled environment where we can clearly observe the filter's effects on both the signal and noise components.

```python
def generate_noisy_series(n_points=1000):
    t = np.linspace(0, 10, n_points)
    # Generate clean signal
    clean = np.sin(t) + 0.5*np.sin(3*t)
    # Add noise
    noisy = clean + np.random.normal(0, 0.2, n_points)
    return t, clean, noisy

t, clean, noisy = generate_noisy_series()
```

Slide 3: Implementing Gaussian Filter

The implementation uses convolution between the input signal and a Gaussian kernel. The kernel size determines the window of influence, while sigma controls the spread of weights within that window.

```python
def gaussian_filter(data, kernel_size=21, sigma=3):
    """Apply Gaussian filter to 1D data"""
    kernel = gaussian_kernel(kernel_size, sigma)
    # Pad the data to handle edges
    pad_size = kernel_size // 2
    padded = np.pad(data, (pad_size, pad_size), mode='edge')
    # Apply convolution
    filtered = np.convolve(padded, kernel, mode='valid')
    return filtered
```

Slide 4: Impact of Different Sigma Values

Understanding how sigma affects filtering is crucial. Larger sigma values produce smoother results but may lose important details, while smaller values preserve more detail but may retain unwanted noise.

```python
def compare_sigmas(data, sigmas=[1, 3, 5]):
    plt.figure(figsize=(12, 6))
    for sigma in sigmas:
        filtered = gaussian_filter(data, sigma=sigma)
        plt.plot(filtered, label=f'σ={sigma}')
    plt.plot(data, 'gray', alpha=0.5, label='Original')
    plt.legend()
    plt.title('Comparison of Different Sigma Values')
    plt.show()
```

Slide 5: Edge Effects and Padding

Edge effects are a critical consideration in time series filtering. Different padding strategies can significantly impact the quality of filtering near the boundaries of the data.

```python
def compare_padding_methods(data):
    padding_modes = ['edge', 'reflect', 'symmetric']
    plt.figure(figsize=(12, 6))
    for mode in padding_modes:
        padded = np.pad(data, (10, 10), mode=mode)
        plt.plot(padded, label=f'Padding: {mode}')
    plt.legend()
    plt.title('Comparison of Padding Methods')
    plt.show()
```

Slide 6: Real-world Application: Stock Price Smoothing

Applying Gaussian filtering to financial time series demonstrates its practical utility. This implementation shows how to process daily stock prices to identify underlying trends.

```python
import yfinance as yf
from datetime import datetime, timedelta

def smooth_stock_prices(symbol, period='1y', sigma=5):
    # Download stock data
    stock = yf.download(symbol, 
                       start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       end=datetime.now().strftime('%Y-%m-%d'))
    
    # Apply Gaussian filter to closing prices
    prices = stock['Close'].values
    smoothed = gaussian_filter(prices, sigma=sigma)
    
    return prices, smoothed
```

Slide 7: Real-time Gaussian Filter Implementation

For real-time applications, we modify the Gaussian filter to work with only historical data. This implementation uses a one-sided kernel and updates the filtered value incrementally as new data arrives.

```python
def realtime_gaussian_filter(data, sigma=2, window_size=None):
    if window_size is None:
        window_size = int(6 * sigma)  # Cover 99.7% of distribution
    
    kernel = gaussian_kernel(window_size, sigma)[:window_size//2 + 1]
    kernel = kernel / kernel.sum()  # Renormalize truncated kernel
    
    filtered = np.zeros_like(data)
    for i in range(len(data)):
        window = data[max(0, i-window_size//2):i+1]
        k = kernel[-len(window):]
        filtered[i] = np.sum(window * k) / np.sum(k)
    
    return filtered
```

Slide 8: Signal-to-Noise Ratio Analysis

Evaluating filter performance requires measuring the signal-to-noise ratio improvement. This implementation quantifies the effectiveness of different filter parameters.

```python
def calculate_snr(original, filtered, true_signal):
    """Calculate Signal-to-Noise Ratio improvement"""
    noise_before = original - true_signal
    noise_after = filtered - true_signal
    
    snr_before = 10 * np.log10(np.var(true_signal) / np.var(noise_before))
    snr_after = 10 * np.log10(np.var(true_signal) / np.var(noise_after))
    
    return snr_before, snr_after

# Example usage
snr_orig, snr_filt = calculate_snr(noisy, filtered_signal, clean)
print(f"SNR Improvement: {snr_filt - snr_orig:.2f} dB")
```

Slide 9: Adaptive Sigma Selection

An advanced implementation that automatically selects the optimal sigma value based on the local characteristics of the time series data.

```python
def adaptive_gaussian_filter(data, min_sigma=1, max_sigma=5):
    # Calculate local variance in sliding windows
    window_size = 21
    local_std = np.array([np.std(data[max(0, i-window_size//2):min(len(data), i+window_size//2)])
                         for i in range(len(data))])
    
    # Scale sigma inversely with local standard deviation
    adaptive_sigma = min_sigma + (max_sigma - min_sigma) * (1 - local_std/np.max(local_std))
    
    # Apply filter with varying sigma
    filtered = np.zeros_like(data)
    for i in range(len(data)):
        kernel = gaussian_kernel(window_size, adaptive_sigma[i])
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        k = kernel[window_size//2 - (i-start):window_size//2 + (end-i)]
        filtered[i] = np.sum(data[start:end] * k) / np.sum(k)
    
    return filtered
```

Slide 10: Processing Multiple Time Series

Implementation for simultaneously filtering multiple time series with shared characteristics, optimized for computational efficiency.

```python
def batch_gaussian_filter(data_matrix, sigma=3):
    """
    Filter multiple time series simultaneously
    data_matrix: shape (n_series, n_timesteps)
    """
    kernel = gaussian_kernel(21, sigma)
    
    # Use broadcasting for efficient computation
    padded = np.pad(data_matrix, ((0,0), (10,10)), mode='edge')
    filtered = np.array([np.convolve(series, kernel, mode='valid') 
                        for series in padded])
    
    return filtered
```

Slide 11: Fast Fourier Transform Implementation

An alternative implementation using FFT for faster computation of Gaussian filtering on large datasets.

```python
def fft_gaussian_filter(data, sigma):
    """Gaussian filter implementation using FFT"""
    # Generate frequency domain gaussian kernel
    n = len(data)
    freq = np.fft.fftfreq(n)
    gaussian_freq = np.exp(-0.5 * (2 * np.pi * freq * sigma)**2)
    
    # Apply filter in frequency domain
    data_fft = np.fft.fft(data)
    filtered_fft = data_fft * gaussian_freq
    filtered = np.real(np.fft.ifft(filtered_fft))
    
    return filtered
```

Slide 12: Performance Benchmarking

A comprehensive comparison of different Gaussian filtering implementations, measuring execution time and accuracy across various data sizes and sigma values.

```python
def benchmark_filters(data_sizes=[1000, 10000, 100000], sigmas=[1, 3, 5]):
    results = {}
    for size in data_sizes:
        # Generate test data
        t, clean, noisy = generate_noisy_series(size)
        
        for sigma in sigmas:
            # Time standard convolution
            start = time.time()
            conv_filtered = gaussian_filter(noisy, sigma=sigma)
            conv_time = time.time() - start
            
            # Time FFT method
            start = time.time()
            fft_filtered = fft_gaussian_filter(noisy, sigma)
            fft_time = time.time() - start
            
            results[(size, sigma)] = {
                'conv_time': conv_time,
                'fft_time': fft_time,
                'conv_mse': np.mean((conv_filtered - clean)**2),
                'fft_mse': np.mean((fft_filtered - clean)**2)
            }
    
    return results
```

Slide 13: Handling Missing Data

Implementation of Gaussian filtering for time series with missing values, using sophisticated interpolation techniques before applying the filter.

```python
def gaussian_filter_with_missing(data, sigma=3):
    """Handle missing values (NaN) in time series data"""
    # Create mask of valid values
    mask = ~np.isnan(data)
    x = np.arange(len(data))
    
    # Interpolate missing values
    valid_x = x[mask]
    valid_y = data[mask]
    interp = np.interp(x, valid_x, valid_y)
    
    # Apply Gaussian filter
    filtered = gaussian_filter(interp, sigma=sigma)
    
    # Restore missing value markers
    filtered[~mask] = np.nan
    
    return filtered, interp
```

Slide 14: Real-world Application: EEG Signal Processing

Practical implementation showing Gaussian filtering application to electroencephalogram (EEG) data, demonstrating noise reduction while preserving important brain activity patterns.

```python
def process_eeg_signal(eeg_data, sampling_rate=256):
    """
    Process EEG signal with adaptive Gaussian filtering
    sampling_rate: Hz
    """
    # Define frequency bands
    delta_sigma = sampling_rate / 30  # For delta waves (0.5-4 Hz)
    beta_sigma = sampling_rate / 100  # For beta waves (13-30 Hz)
    
    # Apply different filters for different frequency bands
    delta_filtered = gaussian_filter(eeg_data, sigma=delta_sigma)
    beta_filtered = gaussian_filter(eeg_data, sigma=beta_sigma)
    
    # Combine filtered signals
    combined = 0.7 * delta_filtered + 0.3 * beta_filtered
    
    return {
        'delta': delta_filtered,
        'beta': beta_filtered,
        'combined': combined
    }
```

Slide 15: Additional Resources

*   "Gaussian Processes for Time Series Analysis" - [https://arxiv.org/abs/1906.08329](https://arxiv.org/abs/1906.08329)
*   "Adaptive Gaussian Filtering for Real-time Applications" - [https://arxiv.org/abs/2003.05798](https://arxiv.org/abs/2003.05798)
*   "On the Optimal Selection of Gaussian Filter Parameters" - [https://arxiv.org/abs/1912.09347](https://arxiv.org/abs/1912.09347)
*   "Fast Gaussian Filtering using FFT-based Methods" - [https://arxiv.org/abs/2105.12456](https://arxiv.org/abs/2105.12456)
*   "Missing Data Imputation in Time Series using Gaussian Processes" - [https://arxiv.org/abs/2008.07593](https://arxiv.org/abs/2008.07593)

