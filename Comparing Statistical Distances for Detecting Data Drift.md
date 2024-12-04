## Comparing Statistical Distances for Detecting Data Drift
Slide 1: Understanding Statistical Distances for Data Drift

Statistical distances provide mathematical frameworks for quantifying the difference between probability distributions. These metrics are crucial for detecting and monitoring data drift in machine learning systems, where the statistical properties of production data may deviate from training data over time.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate example distributions
np.random.seed(42)
reference = np.random.normal(0, 1, 1000)
production = np.random.normal(0.5, 1.2, 1000)

# Helper function to estimate probability density
def estimate_density(data, points=100):
    density = stats.gaussian_kde(data)
    xs = np.linspace(min(data), max(data), points)
    return xs, density(xs)
```

Slide 2: Implementing Jensen-Shannon Distance

The Jensen-Shannon distance is derived from the Jensen-Shannon divergence and provides a symmetric measure bounded between 0 and 1. This implementation demonstrates the calculation between two probability distributions using numpy arrays.

```python
def jensen_shannon_distance(p, q):
    # Convert to probability distributions
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate mid-point distribution
    m = 0.5 * (p + q)
    
    # Calculate JS divergence
    divergence = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
    
    # Convert divergence to distance
    distance = np.sqrt(divergence)
    return distance

# Example usage
x_ref, p = estimate_density(reference)
x_prod, q = estimate_density(production)
js_dist = jensen_shannon_distance(p, q)
print(f"Jensen-Shannon Distance: {js_dist:.4f}")
```

Slide 3: Kullback-Leibler Divergence Implementation

KL Divergence measures the relative entropy between two probability distributions. While not symmetric, it's particularly sensitive to distribution changes and useful for early drift detection in monitoring systems.

```python
def kl_divergence(p, q):
    # Ensure valid probability distributions
    p = np.asarray(p)
    q = np.asarray(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Add small constant to prevent division by zero
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    return np.sum(p * np.log(p / q))

# Calculate KL divergence for our distributions
kl_div = kl_divergence(p, q)
print(f"KL Divergence: {kl_div:.4f}")
```

Slide 4: Wasserstein Distance Calculation

The Wasserstein distance, also known as Earth Mover's Distance, measures the minimum "work" required to transform one distribution into another. This implementation uses the scipy.stats module for efficient computation.

```python
from scipy.stats import wasserstein_distance

def calculate_wasserstein(x1, x2):
    # Ensure inputs are numpy arrays
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Calculate Wasserstein distance
    w_distance = wasserstein_distance(x1, x2)
    return w_distance

# Calculate Wasserstein distance
w_dist = calculate_wasserstein(reference, production)
print(f"Wasserstein Distance: {w_dist:.4f}")
```

Slide 5: Real-time Drift Detection System

This implementation creates a real-time drift detection system that monitors incoming data streams and calculates multiple distance metrics simultaneously, providing a comprehensive view of distribution changes.

```python
class DriftDetector:
    def __init__(self, reference_data, window_size=1000):
        self.reference_data = reference_data
        self.window_size = window_size
        self.current_window = []
        
    def update(self, new_sample):
        self.current_window.append(new_sample)
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
            
        return self.calculate_distances()
    
    def calculate_distances(self):
        if len(self.current_window) < self.window_size:
            return None
            
        x_ref, p = estimate_density(self.reference_data)
        x_prod, q = estimate_density(self.current_window)
        
        return {
            'js_distance': jensen_shannon_distance(p, q),
            'kl_divergence': kl_divergence(p, q),
            'wasserstein': calculate_wasserstein(self.reference_data, 
                                               self.current_window)
        }

# Initialize detector
detector = DriftDetector(reference)
```

Slide 6: Multi-dimensional Data Drift Detection

Data drift detection becomes more complex with high-dimensional data. This implementation extends our previous metrics to handle multi-dimensional features using dimensionality reduction techniques before calculating distances.

```python
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class MultiDimensionalDriftDetector:
    def __init__(self, reference_data, n_components=2):
        self.pca = PCA(n_components=n_components)
        self.reference_transformed = self.pca.fit_transform(reference_data)
        
    def detect_drift(self, production_data):
        # Transform production data using fitted PCA
        prod_transformed = self.pca.transform(production_data)
        
        # Calculate distances in reduced space
        distances = {
            'wasserstein': np.mean([
                calculate_wasserstein(
                    self.reference_transformed[:, i],
                    prod_transformed[:, i]
                ) for i in range(self.pca.n_components_)
            ]),
            'js_distance': np.mean([
                jensen_shannon_distance(
                    *estimate_density(self.reference_transformed[:, i])[1],
                    *estimate_density(prod_transformed[:, i])[1]
                ) for i in range(self.pca.n_components_)
            ])
        }
        return distances
```

Slide 7: Statistical Hypothesis Testing for Drift Detection

Augmenting distance metrics with statistical hypothesis testing provides a more robust approach to drift detection. This implementation uses the Kolmogorov-Smirnov test alongside our distance metrics.

```python
from scipy import stats

def statistical_drift_test(reference, production, alpha=0.05):
    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(reference, production)
    
    # Calculate effect size (Cohen's d)
    cohens_d = (np.mean(production) - np.mean(reference)) / \
               np.sqrt((np.var(production) + np.var(reference)) / 2)
    
    results = {
        'drift_detected': p_value < alpha,
        'p_value': p_value,
        'ks_statistic': ks_statistic,
        'effect_size': cohens_d
    }
    return results

# Example usage
drift_results = statistical_drift_test(reference, production)
print(f"Drift Test Results: {drift_results}")
```

Slide 8: Visualization of Distribution Shifts

Creating effective visualizations helps in understanding the nature and magnitude of data drift. This implementation provides multiple visualization techniques for monitoring distribution changes.

```python
def visualize_distribution_shift(reference, production, title="Distribution Shift"):
    plt.figure(figsize=(12, 6))
    
    # Kernel Density Estimation
    x_ref, p = estimate_density(reference)
    x_prod, q = estimate_density(production)
    
    plt.subplot(1, 2, 1)
    plt.plot(x_ref, p, label='Reference', color='blue', alpha=0.7)
    plt.plot(x_prod, q, label='Production', color='red', alpha=0.7)
    plt.fill_between(x_ref, p, alpha=0.3, color='blue')
    plt.fill_between(x_prod, q, alpha=0.3, color='red')
    plt.title('Density Comparison')
    plt.legend()
    
    # QQ Plot
    plt.subplot(1, 2, 2)
    stats.probplot(production, dist="norm", plot=plt)
    plt.title('Q-Q Plot vs Normal')
    
    plt.tight_layout()
    plt.show()
```

Slide 9: Implementing Adaptive Thresholds

Static thresholds for drift detection can be problematic. This implementation uses adaptive thresholds based on historical distance measurements and statistical control charts.

```python
class AdaptiveThresholdDetector:
    def __init__(self, window_size=100, z_score_threshold=3):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.distance_history = []
        
    def update_threshold(self, new_distance):
        self.distance_history.append(new_distance)
        if len(self.distance_history) > self.window_size:
            self.distance_history.pop(0)
            
        mean_dist = np.mean(self.distance_history)
        std_dist = np.std(self.distance_history)
        
        threshold = mean_dist + self.z_score_threshold * std_dist
        return threshold
    
    def is_drift_detected(self, current_distance):
        if len(self.distance_history) < self.window_size:
            return False
            
        threshold = self.update_threshold(current_distance)
        return current_distance > threshold

# Example usage
adaptive_detector = AdaptiveThresholdDetector()
```

Slide 10: Mathematical Foundations of Distance Metrics

This slide presents the formal mathematical definitions of the key distance metrics used in drift detection. The formulas are presented in their theoretical form before practical implementation.

```python
# Mathematical definitions in LaTeX format (not rendered):

"""
Jensen-Shannon Distance:
$$JSD(P||Q) = \sqrt{\frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)}$$
where $$M = \frac{1}{2}(P + Q)$$

Kullback-Leibler Divergence:
$$D_{KL}(P||Q) = \sum_{i} P(i) \log\frac{P(i)}{Q(i)}$$

Wasserstein Distance:
$$W_p(P,Q) = \inf_{\gamma \in \Gamma(P,Q)} (\int d(x,y)^p d\gamma(x,y))^{1/p}$$
"""

# Implementation of mathematical definitions
def mathematical_distances():
    # Example probability distributions
    P = np.array([0.2, 0.3, 0.5])
    Q = np.array([0.1, 0.4, 0.5])
    
    # Calculate M (middle distribution)
    M = 0.5 * (P + Q)
    
    return {
        'distributions': {
            'P': P,
            'Q': Q,
            'M': M
        }
    }
```

Slide 11: Real-world Application: Financial Data Drift Detection

This implementation demonstrates drift detection in financial time series data, specifically focusing on stock price distributions and their changes over time.

```python
import pandas as pd
from datetime import datetime, timedelta

class FinancialDriftDetector:
    def __init__(self, lookback_window=30, threshold=0.1):
        self.lookback_window = lookback_window
        self.threshold = threshold
        self.reference_window = None
        
    def detect_price_drift(self, prices):
        if len(prices) < self.lookback_window * 2:
            return False
            
        # Split into reference and current windows
        reference = prices[-2*self.lookback_window:-self.lookback_window]
        current = prices[-self.lookback_window:]
        
        # Calculate returns
        ref_returns = np.diff(reference) / reference[:-1]
        curr_returns = np.diff(current) / current[:-1]
        
        # Calculate distances
        w_dist = calculate_wasserstein(ref_returns, curr_returns)
        js_dist = jensen_shannon_distance(
            *estimate_density(ref_returns)[1],
            *estimate_density(curr_returns)[1]
        )
        
        return {
            'wasserstein_drift': w_dist > self.threshold,
            'js_drift': js_dist > self.threshold,
            'metrics': {
                'wasserstein': w_dist,
                'js_distance': js_dist
            }
        }

# Example usage with synthetic data
prices = np.exp(np.random.normal(0, 0.01, 1000)).cumprod()
detector = FinancialDriftDetector()
result = detector.detect_price_drift(prices)
print(f"Financial Drift Detection Results: {result}")
```

Slide 12: Implementation of Ensemble Drift Detection

This system combines multiple drift detection methods to create a more robust detection mechanism, using weighted voting to make final decisions about drift presence.

```python
class EnsembleDriftDetector:
    def __init__(self, reference_data, weights=None):
        self.reference_data = reference_data
        self.detectors = {
            'ks_test': lambda x: statistical_drift_test(reference_data, x),
            'wasserstein': lambda x: calculate_wasserstein(reference_data, x),
            'js_distance': lambda x: jensen_shannon_distance(
                *estimate_density(reference_data)[1],
                *estimate_density(x)[1]
            )
        }
        self.weights = weights or {k: 1/len(self.detectors) 
                                 for k in self.detectors.keys()}
        
    def detect_drift(self, production_data, threshold=0.05):
        results = {}
        weighted_score = 0
        
        for name, detector in self.detectors.items():
            score = detector(production_data)
            results[name] = score
            weighted_score += score * self.weights[name]
            
        return {
            'drift_detected': weighted_score > threshold,
            'weighted_score': weighted_score,
            'individual_scores': results
        }

# Example usage
ensemble = EnsembleDriftDetector(reference)
ensemble_results = ensemble.detect_drift(production)
print(f"Ensemble Detection Results: {ensemble_results}")
```

Slide 13: Implementation of Time-Series Specific Drift Detection

This implementation focuses on detecting drift in time series data by considering both distributional changes and temporal dependencies through sequential analysis.

```python
class TimeSeriesDriftDetector:
    def __init__(self, window_size=100, n_lags=5):
        self.window_size = window_size
        self.n_lags = n_lags
        
    def create_features(self, data):
        features = []
        for i in range(len(data) - self.n_lags):
            features.append(data[i:i + self.n_lags])
        return np.array(features)
    
    def detect_drift(self, reference, production):
        # Create lagged features
        ref_features = self.create_features(reference)
        prod_features = self.create_features(production)
        
        # Calculate multivariate distances
        distances = {
            'wasserstein': np.mean([
                calculate_wasserstein(ref_features[:, i], 
                                   prod_features[:, i])
                for i in range(self.n_lags)
            ]),
            'temporal_correlation': np.corrcoef(
                np.mean(ref_features, axis=0),
                np.mean(prod_features, axis=0)
            )[0, 1]
        }
        
        return distances

# Example usage
ts_detector = TimeSeriesDriftDetector()
ts_results = ts_detector.detect_drift(
    reference[-500:], 
    production[-500:]
)
print(f"Time Series Drift Results: {ts_results}")
```

Slide 14: Performance Evaluation of Drift Detection Methods

This implementation provides a framework for evaluating and comparing different drift detection methods using synthetic data with known drift points.

```python
def evaluate_drift_detectors(n_samples=1000, n_experiments=100):
    results = {
        'js_distance': {'true_positives': 0, 'false_positives': 0},
        'kl_divergence': {'true_positives': 0, 'false_positives': 0},
        'wasserstein': {'true_positives': 0, 'false_positives': 0}
    }
    
    for _ in range(n_experiments):
        # Generate data with known drift point
        reference = np.random.normal(0, 1, n_samples)
        drift_data = np.random.normal(0.5, 1.2, n_samples)
        no_drift_data = np.random.normal(0, 1, n_samples)
        
        # Test with drift
        for method in results.keys():
            if method == 'js_distance':
                score = jensen_shannon_distance(
                    *estimate_density(reference)[1],
                    *estimate_density(drift_data)[1]
                )
            elif method == 'kl_divergence':
                score = kl_divergence(
                    *estimate_density(reference)[1],
                    *estimate_density(drift_data)[1]
                )
            else:
                score = calculate_wasserstein(reference, drift_data)
                
            results[method]['true_positives'] += score > 0.1
            
            # Test without drift
            if method == 'js_distance':
                score = jensen_shannon_distance(
                    *estimate_density(reference)[1],
                    *estimate_density(no_drift_data)[1]
                )
            elif method == 'kl_divergence':
                score = kl_divergence(
                    *estimate_density(reference)[1],
                    *estimate_density(no_drift_data)[1]
                )
            else:
                score = calculate_wasserstein(reference, no_drift_data)
                
            results[method]['false_positives'] += score > 0.1
    
    # Calculate metrics
    for method in results.keys():
        results[method]['precision'] = results[method]['true_positives'] / \
            (results[method]['true_positives'] + results[method]['false_positives'])
        results[method]['recall'] = results[method]['true_positives'] / n_experiments
        
    return results

evaluation_results = evaluate_drift_detectors()
print("Evaluation Results:", evaluation_results)
```

Slide 15: Additional Resources

*   ArXiv Papers:
*   "Deep Learning Approaches to Data Drift Detection" - [https://arxiv.org/abs/2107.14075](https://arxiv.org/abs/2107.14075)
*   "A Survey on Concept Drift Adaptation" - [https://arxiv.org/abs/1010.4784](https://arxiv.org/abs/1010.4784)
*   "Robust Drift Detection Using Statistical Learning Methods" - [https://arxiv.org/abs/1904.09587](https://arxiv.org/abs/1904.09587)
*   "Distribution Drift Detection for Deep Learning Systems" - [https://arxiv.org/abs/2206.14511](https://arxiv.org/abs/2206.14511)
*   Suggested Search Terms:
*   "Data drift detection machine learning"
*   "Concept drift adaptation methods"
*   "Statistical distance metrics for distribution comparison"
*   "Real-time drift detection systems"

