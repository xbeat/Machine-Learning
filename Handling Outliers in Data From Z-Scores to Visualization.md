## Handling Outliers in Data From Z-Scores to Visualization
Slide 1: Understanding Z-Scores for Outlier Detection

Z-scores represent the number of standard deviations a data point lies from the mean. This statistical measure helps identify potential outliers by quantifying how extreme each value is relative to the overall distribution, with values beyond ±3 typically considered outliers.

```python
import numpy as np
import pandas as pd

def calculate_zscores(data):
    # Calculate z-scores for each data point
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    
    # Create DataFrame for better visualization
    df = pd.DataFrame({'original_data': data, 'z_scores': z_scores})
    
    # Identify outliers using |z| > 3 threshold
    outliers = df[abs(df['z_scores']) > 3]
    
    return df, outliers

# Example usage
data = np.array([1, 2, 2.5, 2.7, 3, 15, 2.8, 2.9, 3.1, 2.6])
results, outliers = calculate_zscores(data)
print("Full Dataset with Z-scores:")
print(results)
print("\nOutliers (|z| > 3):")
print(outliers)
```

Slide 2: Advanced Z-Score Analysis with Modified Z-Scores

The modified Z-score approach uses median and median absolute deviation instead of mean and standard deviation, making it more robust against extreme values and multiple outliers in the dataset.

```python
def modified_zscore(data):
    # Calculate median and MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median)) * 1.4826
    
    # Calculate modified z-scores
    modified_zscores = 0.6745 * (data - median) / mad
    
    # Create results DataFrame
    results = pd.DataFrame({
        'data': data,
        'modified_zscore': modified_zscores,
        'is_outlier': abs(modified_zscores) > 3.5
    })
    
    return results

# Example with skewed data
data = np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 10, 15, 20])
results = modified_zscore(data)
print("Modified Z-score Analysis:")
print(results)
```

Slide 3: IQR Method Implementation

The Interquartile Range (IQR) method defines outliers as values falling outside 1.5 times the IQR below the first quartile or above the third quartile, providing a robust approach less sensitive to extreme values.

```python
def iqr_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Define bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Create mask for outliers
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    
    results = pd.DataFrame({
        'value': data,
        'is_outlier': outlier_mask,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    
    return results

# Example usage
data = np.array([1, 2, 2.5, 2.7, 3, 25, 2.8, 2.9, 3.1, 2.6, 30, 1.8])
results = iqr_outliers(data)
print("IQR-based Outlier Detection:")
print(results)
```

Slide 4: Data Visualization for Outlier Detection

Understanding the distribution of data through visualization is crucial for outlier detection. This implementation combines box plots and scatter plots to provide a comprehensive view of potential outliers.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_outliers(data, title="Outlier Visualization"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Box plot
    sns.boxplot(x=data, ax=ax1)
    ax1.set_title("Box Plot with Outliers")
    
    # Scatter plot with z-scores
    z_scores = (data - np.mean(data)) / np.std(data)
    ax2.scatter(range(len(data)), z_scores)
    ax2.axhline(y=3, color='r', linestyle='--', label='Upper Threshold (z=3)')
    ax2.axhline(y=-3, color='r', linestyle='--', label='Lower Threshold (z=-3)')
    ax2.set_title("Z-Score Distribution")
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
data = np.concatenate([
    np.random.normal(10, 2, 100),
    np.random.normal(30, 1, 5)  # Outliers
])
fig = visualize_outliers(data)
plt.show()
```

Slide 5: Outlier Transformation Techniques

Data transformation techniques can help minimize the impact of outliers while preserving their relative positions in the dataset. Common methods include logarithmic, square root, and Box-Cox transformations.

```python
import scipy.stats as stats

def transform_outliers(data):
    # Create different transformations
    log_transform = np.log1p(data - min(data) + 1)
    sqrt_transform = np.sqrt(data - min(data))
    boxcox_transform, lambda_param = stats.boxcox(data - min(data) + 1)
    
    results = pd.DataFrame({
        'original': data,
        'log_transform': log_transform,
        'sqrt_transform': sqrt_transform,
        'boxcox_transform': boxcox_transform
    })
    
    return results, lambda_param

# Example usage
data = np.array([2, 3, 4, 5, 6, 100, 4, 5, 6, 7, 200, 5])
transformed_data, lambda_param = transform_outliers(data)
print("Transformed Data:")
print(transformed_data)
print(f"\nBox-Cox transformation lambda: {lambda_param:.3f}")
```

Slide 6: Robust Statistical Methods for Outlier Handling

Robust statistical methods provide reliable estimates of central tendency and dispersion even in the presence of outliers. This implementation demonstrates the use of robust estimators for location and scale.

```python
from scipy.stats import trim_mean, iqr
from sklearn.covariance import MinCovDet

def robust_statistics(data):
    # Calculate robust estimates
    trimmed_mean = trim_mean(data, 0.1)  # 10% trimming
    winsorized_mean = stats.mstats.winsorize(data, limits=[0.05, 0.05]).mean()
    huber_location = stats.huber(data).mu
    
    # Create robustness comparison
    results = pd.DataFrame({
        'statistic': ['mean', 'median', 'trimmed_mean', 'winsorized_mean', 'huber_location'],
        'value': [
            np.mean(data),
            np.median(data),
            trimmed_mean,
            winsorized_mean,
            huber_location
        ]
    })
    
    return results

# Example with contaminated data
np.random.seed(42)
normal_data = np.random.normal(10, 2, 100)
outliers = np.array([50, 60, 70, -20, -30])
data = np.concatenate([normal_data, outliers])

results = robust_statistics(data)
print("Robust Statistics Comparison:")
print(results)
```

Slide 7: Automated Outlier Detection with Isolation Forest

The Isolation Forest algorithm isolates outliers by randomly selecting a feature and split value, making it particularly effective for high-dimensional datasets and requiring minimal assumptions about the data distribution.

```python
from sklearn.ensemble import IsolationForest
import numpy as np

def isolation_forest_detector(data, contamination=0.1):
    # Reshape data for sklearn
    X = data.reshape(-1, 1)
    
    # Initialize and fit the Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    # Fit and predict
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.score_samples(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'value': data,
        'is_outlier': predictions == -1,
        'anomaly_score': -scores  # Higher score = more likely to be outlier
    }).sort_values('anomaly_score', ascending=False)
    
    return results

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 100)
outliers = np.array([5, 7, -6, 8, -7])
data = np.concatenate([normal_data, outliers])

results = isolation_forest_detector(data)
print("Isolation Forest Results (Top 10 potential outliers):")
print(results.head(10))
```

Slide 8: Local Outlier Factor (LOF) Implementation

LOF identifies outliers by measuring the local deviation of a point with respect to its neighbors, making it effective for detecting outliers in datasets with varying densities.

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_detector(data, n_neighbors=20):
    # Reshape data for sklearn
    X = data.reshape(-1, 1)
    
    # Initialize and fit LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination='auto',
        novelty=False
    )
    
    # Predict and get negative outlier scores
    predictions = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_
    
    # Create results DataFrame
    results = pd.DataFrame({
        'value': data,
        'is_outlier': predictions == -1,
        'lof_score': -scores  # Convert to positive scores for consistency
    }).sort_values('lof_score', ascending=False)
    
    return results

# Example with clustered data and outliers
np.random.seed(42)
cluster1 = np.random.normal(0, 0.5, 50)
cluster2 = np.random.normal(5, 0.5, 50)
outliers = np.array([-2, 7, 2.5])
data = np.concatenate([cluster1, cluster2, outliers])

results = lof_detector(data)
print("LOF Detection Results (Top 10 potential outliers):")
print(results.head(10))
```

Slide 9: DBSCAN for Density-Based Outlier Detection

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) effectively identifies outliers as points that don't belong to any cluster, particularly useful for datasets with clusters of varying shapes and densities.

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def dbscan_outlier_detector(data, eps=0.5, min_samples=5):
    # Standardize and reshape data
    X = StandardScaler().fit_transform(data.reshape(-1, 1))
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'value': data,
        'cluster': clusters,
        'is_outlier': clusters == -1
    })
    
    # Calculate additional statistics
    cluster_stats = results.groupby('cluster').agg({
        'value': ['count', 'mean', 'std']
    }).round(3)
    
    return results, cluster_stats

# Example usage
np.random.seed(42)
cluster1 = np.random.normal(0, 0.5, 50)
cluster2 = np.random.normal(5, 0.5, 50)
outliers = np.array([-2, 7, 2.5, 8, -3])
data = np.concatenate([cluster1, cluster2, outliers])

results, stats = dbscan_outlier_detector(data)
print("DBSCAN Outlier Detection Results:")
print(results[results['is_outlier']].sort_values('value'))
print("\nCluster Statistics:")
print(stats)
```

Slide 10: Real-World Application - Financial Time Series Outliers

Financial data often contains anomalies due to market events or recording errors. This implementation demonstrates a comprehensive approach to detecting and handling outliers in stock price data.

```python
import pandas as pd
import numpy as np
from scipy import stats

def analyze_financial_outliers(prices, window=20):
    # Calculate returns
    returns = np.log(prices / prices.shift(1))
    
    # Rolling statistics
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    # Calculate rolling z-scores
    z_scores = (returns - rolling_mean) / rolling_std
    
    # Multiple detection methods
    results = pd.DataFrame({
        'price': prices,
        'returns': returns,
        'z_score': z_scores,
        'is_zscore_outlier': abs(z_scores) > 3,
        'is_mad_outlier': abs(returns - returns.median()) > 3 * stats.median_abs_deviation(returns.dropna())
    })
    
    # Add volatility regime detection
    results['volatility'] = rolling_std
    results['high_volatility'] = results['volatility'] > results['volatility'].quantile(0.95)
    
    return results

# Example with simulated stock data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
prices = 100 * (1 + np.random.normal(0.0002, 0.01, 252)).cumprod()
# Add some artificial outliers
prices[50] *= 1.15  # Sudden jump
prices[150] *= 0.85  # Sudden drop

results = analyze_financial_outliers(prices)
print("Financial Outlier Analysis Results:")
print(results[results['is_zscore_outlier']].head())
```

Slide 11: Source Code for Financial Time Series Visualization

```python
def visualize_financial_outliers(results):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Price plot with outliers highlighted
    ax1.plot(results.index, results['price'], label='Price')
    outliers = results[results['is_zscore_outlier']]
    ax1.scatter(outliers.index, outliers['price'], 
                color='red', label='Outliers', zorder=5)
    ax1.set_title('Price Series with Outliers')
    ax1.legend()
    
    # Returns distribution
    sns.histplot(data=results['returns'].dropna(), ax=ax2, bins=50)
    ax2.axvline(results['returns'].mean(), color='r', linestyle='--', 
                label='Mean')
    ax2.axvline(results['returns'].median(), color='g', linestyle='--', 
                label='Median')
    ax2.set_title('Returns Distribution')
    ax2.legend()
    
    # Volatility regime
    ax3.plot(results.index, results['volatility'], label='Volatility')
    ax3.axhline(results['volatility'].quantile(0.95), color='r', 
                linestyle='--', label='95th Percentile')
    ax3.set_title('Volatility Regime')
    ax3.legend()
    
    plt.tight_layout()
    return fig

# Visualize results
fig = visualize_financial_outliers(results)
plt.show()
```

Slide 12: Real-World Application - Sensor Data Anomaly Detection

Sensor networks often produce data with various types of anomalies. This implementation shows how to detect and classify different types of sensor data outliers.

```python
def analyze_sensor_data(timestamps, values, window_size=12):
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Calculate rolling statistics
    df['rolling_mean'] = df['value'].rolling(window=window_size).mean()
    df['rolling_std'] = df['value'].rolling(window=window_size).std()
    
    # Different types of anomalies
    df['spike'] = abs(df['value'] - df['rolling_mean']) > 3 * df['rolling_std']
    df['level_shift'] = abs(df['rolling_mean'].diff()) > 2 * df['rolling_std']
    df['variance_change'] = df['rolling_std'] > 2 * df['rolling_std'].mean()
    
    # Seasonal adjustment
    seasonal_means = df.groupby('hour')['value'].transform('mean')
    seasonal_std = df.groupby('hour')['value'].transform('std')
    df['seasonal_residual'] = (df['value'] - seasonal_means) / seasonal_std
    
    return df

# Generate example sensor data
np.random.seed(42)
timestamps = pd.date_range('2024-01-01', periods=720, freq='H')
base_signal = 100 + 10 * np.sin(np.pi * np.arange(720) / 24)  # Daily cycle
noise = np.random.normal(0, 1, 720)
anomalies = np.zeros(720)
anomalies[100:105] = 30  # Spike
anomalies[300:400] += np.linspace(0, 20, 100)  # Level shift
values = base_signal + noise + anomalies

results = analyze_sensor_data(timestamps, values)
print("Sensor Data Analysis Results:")
print(results[results[['spike', 'level_shift', 'variance_change']].any(axis=1)].head())
```

Slide 13: Source Code for Sensor Data Visualization

```python
def visualize_sensor_anomalies(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw data with detected anomalies
    ax1.plot(results['timestamp'], results['value'], label='Raw Signal')
    spikes = results[results['spike']]
    level_shifts = results[results['level_shift']]
    ax1.scatter(spikes['timestamp'], spikes['value'], 
                color='red', label='Spikes', zorder=5)
    ax1.scatter(level_shifts['timestamp'], level_shifts['value'], 
                color='orange', label='Level Shifts', zorder=5)
    ax1.set_title('Sensor Data with Detected Anomalies')
    ax1.legend()
    
    # Rolling statistics
    ax2.plot(results['timestamp'], results['rolling_mean'], label='Rolling Mean')
    ax2.fill_between(results['timestamp'], 
                     results['rolling_mean'] - 2*results['rolling_std'],
                     results['rolling_mean'] + 2*results['rolling_std'],
                     alpha=0.2, label='±2σ Band')
    ax2.set_title('Rolling Statistics')
    ax2.legend()
    
    # Seasonal pattern
    hourly_mean = results.groupby('hour')['value'].mean()
    hourly_std = results.groupby('hour')['value'].std()
    ax3.plot(hourly_mean.index, hourly_mean.values, label='Hourly Mean')
    ax3.fill_between(hourly_mean.index, 
                     hourly_mean - hourly_std,
                     hourly_mean + hourly_std,
                     alpha=0.2, label='±1σ Band')
    ax3.set_title('Daily Pattern')
    ax3.legend()
    
    # Seasonal residuals distribution
    sns.histplot(data=results['seasonal_residual'].dropna(), ax=ax4, bins=50)
    ax4.axvline(0, color='r', linestyle='--', label='Mean')
    ax4.axvline(-3, color='g', linestyle='--', label='-3σ')
    ax4.axvline(3, color='g', linestyle='--', label='+3σ')
    ax4.set_title('Seasonal Residuals Distribution')
    ax4.legend()
    
    plt.tight_layout()
    return fig

# Visualize the results
fig = visualize_sensor_anomalies(results)
plt.show()
```

Slide 14: Ensemble Method for Robust Outlier Detection

This implementation combines multiple outlier detection methods to create a more robust and reliable detection system, using a voting mechanism to reduce false positives.

```python
class EnsembleOutlierDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detectors = {
            'isolation_forest': IsolationForest(contamination=contamination),
            'lof': LocalOutlierFactor(contamination=contamination, novelty=True),
            'robust_covariance': MinCovDet(contamination=contamination)
        }
    
    def fit(self, X):
        # Ensure 2D array
        X = np.atleast_2d(X)
        if X.shape[1] == 1:
            X = np.hstack([X, np.zeros_like(X)])
        
        # Fit all detectors
        for name, detector in self.detectors.items():
            try:
                detector.fit(X)
            except Exception as e:
                print(f"Warning: {name} fitting failed: {e}")
        return self
    
    def predict(self, X):
        X = np.atleast_2d(X)
        if X.shape[1] == 1:
            X = np.hstack([X, np.zeros_like(X)])
        
        # Collect predictions from all detectors
        predictions = {}
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'predict'):
                    predictions[name] = detector.predict(X)
                else:
                    predictions[name] = detector.fit_predict(X)
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
        
        # Combine predictions using majority voting
        votes = np.zeros(X.shape[0])
        for pred in predictions.values():
            votes += (pred == -1)
        
        # Final decision: point is outlier if majority says so
        return (votes > len(predictions) / 2).astype(int) * -2 + 1

# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
outliers = np.random.normal(4, 0.5, 50)
data = np.concatenate([normal_data, outliers])

detector = EnsembleOutlierDetector()
predictions = detector.fit_predict(data.reshape(-1, 1))

results = pd.DataFrame({
    'value': data,
    'is_outlier': predictions == -1
})
print("Ensemble Detector Results:")
print(f"Total outliers detected: {sum(predictions == -1)}")
print(results[results['is_outlier']].describe())
```

Slide 15: Additional Resources

*   "A Study of Outlier Detection Methods and Their Applications" - [https://arxiv.org/abs/2202.01048](https://arxiv.org/abs/2202.01048)
*   "Isolation Forest Algorithm and Its Applications" - [https://arxiv.org/abs/1811.02141](https://arxiv.org/abs/1811.02141)
*   "Survey on Deep Learning Methods for Anomaly Detection" - [https://arxiv.org/abs/2009.14017](https://arxiv.org/abs/2009.14017)
*   "Robust Statistics for Outlier Detection: A Comparative Study" - [https://arxiv.org/abs/1904.02181](https://arxiv.org/abs/1904.02181)
*   "Local Outlier Factor: A Density-Based Approach to Outlier Detection" - [https://arxiv.org/abs/1906.03509](https://arxiv.org/abs/1906.03509)

