## Robust Outlier Filtering with Filter by Difference
Slide 1: Introduction to Filter by Difference Method

The Filter by Difference method is a robust statistical approach for detecting and removing outliers in time series data by analyzing the relative differences between consecutive points. This technique preserves data structure while effectively identifying anomalous variations.

```python
import numpy as np
import pandas as pd

def calculate_differences(data):
    # Calculate absolute differences between consecutive points
    differences = np.abs(np.diff(data))
    # Pad with zero to match original length
    return np.concatenate(([0], differences))
```

Slide 2: Basic Implementation of Outlier Detection

This implementation demonstrates the core functionality of the Filter by Difference method, incorporating threshold calculation based on the statistical properties of the differences between consecutive data points.

```python
def detect_outliers(data, threshold_factor=2.0):
    # Calculate differences
    differences = calculate_differences(data)
    
    # Calculate threshold using median absolute deviation
    median_diff = np.median(differences)
    mad = np.median(np.abs(differences - median_diff))
    threshold = threshold_factor * mad
    
    # Mark outliers
    outliers = differences > threshold
    return outliers
```

Slide 3: Forward Fill Implementation

The forward fill strategy handles outliers by replacing them with the last known good value, making it particularly suitable for real-time applications where maintaining data continuity is crucial.

```python
def forward_fill_outliers(data, outliers):
    # Create copy to avoid modifying original data
    cleaned_data = data.copy()
    
    # Replace outliers with NaN
    cleaned_data[outliers] = np.nan
    
    # Forward fill NaN values
    if isinstance(cleaned_data, pd.Series):
        return cleaned_data.fillna(method='ffill')
    return pd.Series(cleaned_data).fillna(method='ffill').values
```

Slide 4: Complete Filter by Difference Class

The FilterByDifference class encapsulates all necessary functionality for outlier detection and removal, providing a clean interface for data processing and parameter adjustment.

```python
class FilterByDifference:
    def __init__(self, threshold_factor=2.0):
        self.threshold_factor = threshold_factor
        self.outlier_mask = None
        
    def fit_transform(self, data):
        self.outlier_mask = detect_outliers(data, self.threshold_factor)
        return forward_fill_outliers(data, self.outlier_mask)
    
    def get_outlier_indices(self):
        return np.where(self.outlier_mask)[0]
```

Slide 5: Real-world Example - Stock Price Analysis

The following example demonstrates the application of Filter by Difference to stock price data, where sudden price spikes or drops need to be identified and handled appropriately.

```python
import yfinance as yf
from datetime import datetime, timedelta

# Download sample stock data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
stock_data = yf.download('AAPL', start=start_date, end=end_date)['Close']

# Apply filter
filter_diff = FilterByDifference(threshold_factor=3.0)
cleaned_prices = filter_diff.fit_transform(stock_data.values)
```

Slide 6: Results Visualization for Stock Analysis

Advanced visualization techniques help understand the impact of the Filter by Difference method on real financial data, highlighting detected outliers and their corrections.

```python
import matplotlib.pyplot as plt

def plot_results(original_data, cleaned_data, outlier_mask):
    plt.figure(figsize=(15, 6))
    plt.plot(original_data, label='Original', alpha=0.6)
    plt.plot(cleaned_data, label='Cleaned', linewidth=2)
    plt.scatter(np.where(outlier_mask)[0], 
               original_data[outlier_mask],
               color='red', label='Outliers')
    plt.legend()
    plt.title('Filter by Difference Results')
    plt.show()
```

Slide 7: Time Series with Multiple Frequencies

This implementation handles time series with multiple underlying frequencies by adapting the threshold calculation to local temporal windows, improving outlier detection accuracy.

```python
def adaptive_threshold_detection(data, window_size=30):
    outliers = np.zeros_like(data, dtype=bool)
    
    for i in range(len(data)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(data), i + window_size)
        
        window = data[start_idx:end_idx]
        differences = calculate_differences(window)
        
        # Calculate local threshold
        mad = np.median(np.abs(differences - np.median(differences)))
        threshold = 3.0 * mad
        
        if i > 0:
            current_diff = abs(data[i] - data[i-1])
            outliers[i] = current_diff > threshold
            
    return outliers
```

Slide 8: Performance Optimization

This optimized implementation reduces computational complexity by using vectorized operations and efficient numpy functions, making it suitable for large-scale data processing.

```python
def optimized_filter_by_difference(data, threshold_factor=2.0):
    # Calculate differences using vectorized operations
    differences = np.abs(np.concatenate(([0], np.diff(data))))
    
    # Vectorized MAD calculation
    median_diff = np.median(differences)
    mad = np.median(np.abs(differences - median_diff))
    
    # Vectorized outlier detection
    threshold = threshold_factor * mad
    outliers = differences > threshold
    
    # Efficient forward fill
    mask = ~outliers
    idx = np.where(mask)[0]
    np.maximum.accumulate(data[idx], out=data[idx])
    return data
```

Slide 9: Robust Statistical Analysis

The following implementation incorporates robust statistical measures to handle non-normal distributions and extreme outliers, using modified z-scores and Huber's M-estimator for improved accuracy.

```python
def robust_filter_by_difference(data, k=1.4826):
    # Calculate modified z-scores
    differences = calculate_differences(data)
    median_diff = np.median(differences)
    mad = np.median(np.abs(differences - median_diff))
    modified_zscores = k * np.abs(differences - median_diff) / mad
    
    # Huber's M-estimator for threshold
    c = 1.345  # Tuning constant for 95% efficiency
    huber_weights = np.where(modified_zscores <= c,
                            1,
                            c / modified_zscores)
    
    # Detect outliers using weighted threshold
    weighted_threshold = np.sum(differences * huber_weights) / np.sum(huber_weights)
    outliers = differences > weighted_threshold
    
    return outliers
```

Slide 10: Real-time Processing Implementation

This implementation focuses on real-time data processing capabilities, utilizing a sliding window approach and efficient memory management for continuous data streams.

```python
class RealTimeFilterByDifference:
    def __init__(self, window_size=100, threshold_factor=2.0):
        self.window_size = window_size
        self.threshold_factor = threshold_factor
        self.buffer = np.zeros(window_size)
        self.position = 0
        
    def process_point(self, new_point):
        # Update buffer
        self.buffer[self.position % self.window_size] = new_point
        current_window = self.buffer[max(0, self.position - self.window_size + 1):self.position + 1]
        
        # Calculate dynamic threshold
        differences = calculate_differences(current_window)
        mad = np.median(np.abs(differences - np.median(differences)))
        threshold = self.threshold_factor * mad
        
        # Check if current point is outlier
        is_outlier = differences[-1] > threshold if len(differences) > 0 else False
        
        self.position += 1
        return not is_outlier, new_point if not is_outlier else self.buffer[self.position - 2]
```

Slide 11: Advanced Threshold Calibration

The threshold calibration implementation uses cross-validation and grid search to optimize the threshold parameter based on historical data patterns and domain-specific constraints.

```python
def calibrate_threshold(data, threshold_range=(1.0, 5.0), steps=20):
    best_threshold = None
    min_error = float('inf')
    
    # Generate candidate thresholds
    thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
    
    for threshold in thresholds:
        # Cross-validation splits
        errors = []
        for i in range(5):
            # Create train/test split
            mask = np.zeros(len(data), dtype=bool)
            mask[i::5] = True
            
            # Apply filter
            filter_diff = FilterByDifference(threshold_factor=threshold)
            cleaned = filter_diff.fit_transform(data[~mask])
            
            # Calculate error on validation set
            predicted = np.interp(np.where(mask)[0], 
                                np.where(~mask)[0], 
                                cleaned)
            error = np.mean(np.abs(data[mask] - predicted))
            errors.append(error)
            
        mean_error = np.mean(errors)
        if mean_error < min_error:
            min_error = mean_error
            best_threshold = threshold
            
    return best_threshold
```

Slide 12: Performance Metrics Implementation

A comprehensive set of metrics to evaluate the effectiveness of the Filter by Difference method, including precision, recall, and domain-specific measures.

```python
def calculate_performance_metrics(original_data, cleaned_data, true_outliers=None):
    differences = np.abs(original_data - cleaned_data)
    
    metrics = {
        'mean_absolute_change': np.mean(differences),
        'max_change': np.max(differences),
        'percent_modified': np.mean(differences > 0) * 100
    }
    
    if true_outliers is not None:
        detected_outliers = differences > 0
        tp = np.sum(detected_outliers & true_outliers)
        fp = np.sum(detected_outliers & ~true_outliers)
        fn = np.sum(~detected_outliers & true_outliers)
        
        metrics.update({
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        })
    
    return metrics
```

Slide 13: Additional Resources

1.  [https://arxiv.org/abs/2103.00377](https://arxiv.org/abs/2103.00377) - "Robust Time Series Analysis: A Comprehensive Review"
2.  [https://arxiv.org/abs/1904.02821](https://arxiv.org/abs/1904.02821) - "Adaptive Outlier Detection in Sequential Data Streams"
3.  [https://arxiv.org/abs/2007.15975](https://arxiv.org/abs/2007.15975) - "Statistical Methods for Real-Time Anomaly Detection in Time Series"
4.  [https://arxiv.org/abs/1906.03821](https://arxiv.org/abs/1906.03821) - "A Survey on Time Series Outlier Detection Methods"
5.  [https://arxiv.org/abs/2002.04236](https://arxiv.org/abs/2002.04236) - "Deep Learning Approaches for Time Series Anomaly Detection"

