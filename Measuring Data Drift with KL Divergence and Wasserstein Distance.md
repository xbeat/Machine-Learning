## Measuring Data Drift with KL Divergence and Wasserstein Distance
Slide 1: Understanding Data Drift Detection

Data drift represents the change in data distribution over time that can degrade model performance. Detection requires comparing probability distributions between a reference dataset and production data using statistical measures to quantify the magnitude of drift.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def detect_data_drift(reference_data, production_data, threshold=0.05):
    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(reference_data, production_data)
    
    # Check if drift is detected
    drift_detected = p_value < threshold
    
    return {
        'drift_detected': drift_detected,
        'ks_statistic': ks_statistic,
        'p_value': p_value
    }
```

Slide 2: KL Divergence Implementation

The Kullback-Leibler divergence measures relative entropy between two probability distributions. It quantifies how one probability distribution differs from another reference distribution, providing insights into distribution shifts.

```python
def kl_divergence(p, q):
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Normalize to ensure valid probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))

def calculate_kl_drift(reference_hist, production_hist):
    """
    Calculate KL divergence between reference and production histograms
    """
    bins = len(reference_hist)
    return kl_divergence(reference_hist, production_hist)
```

Slide 3: Wasserstein Distance Calculator

The Wasserstein distance, also known as Earth Mover's Distance, measures the minimum "work" required to transform one distribution into another. It's more robust to small distribution changes and handles non-overlapping distributions better.

```python
from scipy.stats import wasserstein_distance

def calculate_wasserstein(reference_data, production_data):
    # Calculate Wasserstein distance
    distance = wasserstein_distance(reference_data, production_data)
    
    # Normalize the distance by the data range
    data_range = max(np.max(reference_data), np.max(production_data)) - \
                 min(np.min(reference_data), np.min(production_data))
    
    normalized_distance = distance / data_range
    return normalized_distance
```

Slide 4: Distribution Comparison Visualization

This visualization tool helps compare reference and production distributions visually while displaying both KL divergence and Wasserstein distance metrics for comprehensive drift analysis.

```python
def plot_distribution_comparison(reference_data, production_data, title="Distribution Comparison"):
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(reference_data, bins=30, alpha=0.5, label='Reference', density=True)
    plt.hist(production_data, bins=30, alpha=0.5, label='Production', density=True)
    
    # Calculate metrics
    w_distance = calculate_wasserstein(reference_data, production_data)
    kl_div = kl_divergence(*np.histogram(reference_data, bins=30)[0],
                          *np.histogram(production_data, bins=30)[0])
    
    plt.title(f'{title}\nWasserstein: {w_distance:.4f}, KL: {kl_div:.4f}')
    plt.legend()
    plt.show()
```

Slide 5: Real-world Example - Feature Drift Detection

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def monitor_feature_drift(reference_df, production_df, features, threshold=0.1):
    drift_results = {}
    scaler = StandardScaler()
    
    for feature in features:
        ref_data = scaler.fit_transform(reference_df[feature].values.reshape(-1, 1)).ravel()
        prod_data = scaler.transform(production_df[feature].values.reshape(-1, 1)).ravel()
        
        w_distance = calculate_wasserstein(ref_data, prod_data)
        kl_div = kl_divergence(*np.histogram(ref_data, bins=30)[0],
                              *np.histogram(prod_data, bins=30)[0])
        
        drift_results[feature] = {
            'wasserstein': w_distance,
            'kl_divergence': kl_div,
            'drift_detected': w_distance > threshold
        }
    
    return drift_results
```

Slide 6: Multivariate Drift Detection

Detecting drift across multiple features simultaneously requires more sophisticated approaches. This implementation uses a combination of statistical tests and distance metrics to identify multivariate distribution changes.

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def multivariate_drift_detector(reference_data, production_data, alpha=0.05):
    # Calculate mean and covariance of reference data
    ref_mean = np.mean(reference_data, axis=0)
    ref_cov = np.cov(reference_data, rowvar=False)
    ref_cov_inv = np.linalg.inv(ref_cov)
    
    # Calculate Mahalanobis distances
    m_distances = np.array([
        mahalanobis(x, ref_mean, ref_cov_inv) 
        for x in production_data
    ])
    
    # Chi-square test for multivariate normality
    degrees_of_freedom = reference_data.shape[1]
    critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
    
    return {
        'drift_detected': np.mean(m_distances > critical_value) > alpha,
        'drift_score': np.mean(m_distances),
        'critical_value': critical_value
    }
```

Slide 7: Time-Window Based Drift Analysis

This implementation enables continuous monitoring of data drift using sliding time windows, providing insights into gradual distribution changes over time.

```python
def time_window_drift_analysis(data, window_size, reference_window, feature_col, time_col):
    results = []
    
    # Create reference distribution
    ref_data = data[data[time_col].isin(reference_window)][feature_col]
    
    # Sliding window analysis
    for start_time in data[time_col].unique()[:-window_size]:
        window_data = data[
            (data[time_col] >= start_time) & 
            (data[time_col] < start_time + window_size)
        ][feature_col]
        
        w_distance = calculate_wasserstein(ref_data, window_data)
        
        results.append({
            'start_time': start_time,
            'wasserstein_distance': w_distance
        })
    
    return pd.DataFrame(results)
```

Slide 8: Concept Drift Detection Framework

A comprehensive framework for detecting both data and concept drift, incorporating multiple statistical tests and visualization capabilities for production monitoring.

```python
class DriftDetectionFramework:
    def __init__(self, reference_data, drift_threshold=0.1):
        self.reference_data = reference_data
        self.threshold = drift_threshold
        self.metrics_history = []
        
    def detect_drift(self, production_data):
        wasserstein_score = calculate_wasserstein(
            self.reference_data, 
            production_data
        )
        
        kl_score = kl_divergence(
            *np.histogram(self.reference_data, bins=30)[0],
            *np.histogram(production_data, bins=30)[0]
        )
        
        drift_detected = wasserstein_score > self.threshold
        
        results = {
            'timestamp': pd.Timestamp.now(),
            'wasserstein_distance': wasserstein_score,
            'kl_divergence': kl_score,
            'drift_detected': drift_detected
        }
        
        self.metrics_history.append(results)
        return results
        
    def plot_metrics_history(self):
        history_df = pd.DataFrame(self.metrics_history)
        
        plt.figure(figsize=(12, 6))
        plt.plot(history_df['timestamp'], 
                history_df['wasserstein_distance'], 
                label='Wasserstein')
        plt.axhline(y=self.threshold, color='r', 
                   linestyle='--', label='Threshold')
        plt.title('Drift Metrics Over Time')
        plt.legend()
        plt.show()
```

Slide 9: Implementation of Advanced Statistical Tests

Advanced statistical tests provide additional layers of validation for drift detection. This implementation combines multiple statistical approaches to create a more robust drift detection system.

```python
from scipy.stats import anderson_ksamp, ks_2samp, mannwhitneyu

class AdvancedDriftTests:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def perform_all_tests(self, reference_data, production_data):
        # Anderson-Darling test
        ad_statistic, _, ad_p_value = anderson_ksamp([reference_data, production_data])
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = ks_2samp(reference_data, production_data)
        
        # Mann-Whitney U test
        mw_statistic, mw_p_value = mannwhitneyu(reference_data, production_data)
        
        return {
            'anderson_darling': {
                'statistic': ad_statistic,
                'p_value': ad_p_value,
                'drift_detected': ad_p_value < self.alpha
            },
            'kolmogorov_smirnov': {
                'statistic': ks_statistic,
                'p_value': ks_p_value,
                'drift_detected': ks_p_value < self.alpha
            },
            'mann_whitney': {
                'statistic': mw_statistic,
                'p_value': mw_p_value,
                'drift_detected': mw_p_value < self.alpha
            }
        }
```

Slide 10: Real-time Drift Monitoring System

A production-ready system for monitoring data drift in real-time streams, incorporating adaptive thresholds and automated alerts for detected distribution changes.

```python
class RealTimeDriftMonitor:
    def __init__(self, reference_data, window_size=1000):
        self.reference_data = reference_data
        self.window_size = window_size
        self.current_window = []
        self.drift_threshold = self._calculate_initial_threshold()
        
    def _calculate_initial_threshold(self):
        # Calculate baseline drift using bootstrap
        n_bootstrap = 100
        bootstrap_drifts = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(
                self.reference_data, 
                size=len(self.reference_data)//2, 
                replace=True
            )
            drift = calculate_wasserstein(self.reference_data, sample)
            bootstrap_drifts.append(drift)
            
        return np.percentile(bootstrap_drifts, 95)
    
    def process_new_data(self, new_data_point):
        self.current_window.append(new_data_point)
        
        if len(self.current_window) >= self.window_size:
            drift_metrics = self.calculate_window_drift()
            self.current_window = self.current_window[1:]
            return drift_metrics
        
        return None
    
    def calculate_window_drift(self):
        window_data = np.array(self.current_window)
        w_distance = calculate_wasserstein(self.reference_data, window_data)
        
        return {
            'timestamp': pd.Timestamp.now(),
            'wasserstein_distance': w_distance,
            'threshold': self.drift_threshold,
            'drift_detected': w_distance > self.drift_threshold
        }
```

Slide 11: Feature Importance in Drift Detection

This implementation analyzes the contribution of individual features to overall drift, helping identify which features are most responsible for distribution changes.

```python
def analyze_feature_drift_importance(reference_df, production_df, features):
    drift_importance = {}
    
    # Calculate total drift across all features
    total_drift = 0
    feature_drifts = {}
    
    for feature in features:
        ref_data = reference_df[feature].values
        prod_data = production_df[feature].values
        
        # Calculate both metrics for comprehensive analysis
        w_distance = calculate_wasserstein(ref_data, prod_data)
        kl_div = kl_divergence(*np.histogram(ref_data, bins=30)[0],
                              *np.histogram(prod_data, bins=30)[0])
        
        feature_drifts[feature] = {
            'wasserstein': w_distance,
            'kl_divergence': kl_div
        }
        total_drift += w_distance
    
    # Calculate relative importance
    for feature in features:
        drift_importance[feature] = {
            'relative_importance': feature_drifts[feature]['wasserstein'] / total_drift,
            'wasserstein': feature_drifts[feature]['wasserstein'],
            'kl_divergence': feature_drifts[feature]['kl_divergence']
        }
    
    return pd.DataFrame.from_dict(drift_importance, orient='index')
```

Slide 12: Drift Visualization Dashboard

A comprehensive visualization system that combines multiple drift metrics and provides interactive plotting capabilities for monitoring distribution changes over time.

```python
import seaborn as sns
from matplotlib.gridspec import GridSpec

class DriftVisualizationDashboard:
    def __init__(self, reference_data, production_data, features):
        self.reference_data = reference_data
        self.production_data = production_data
        self.features = features
        
    def create_dashboard(self):
        n_features = len(self.features)
        fig = plt.figure(figsize=(15, 5 * n_features))
        gs = GridSpec(n_features, 2)
        
        for idx, feature in enumerate(self.features):
            # Distribution comparison
            ax1 = fig.add_subplot(gs[idx, 0])
            sns.kdeplot(data=self.reference_data[feature], 
                       label='Reference', ax=ax1)
            sns.kdeplot(data=self.production_data[feature], 
                       label='Production', ax=ax1)
            
            # Calculate drift metrics
            w_dist = calculate_wasserstein(
                self.reference_data[feature],
                self.production_data[feature]
            )
            
            # QQ Plot
            ax2 = fig.add_subplot(gs[idx, 1])
            stats.probplot(self.production_data[feature], 
                         dist="norm", plot=ax2)
            
            ax1.set_title(f'{feature} Distribution\nWasserstein Distance: {w_dist:.4f}')
            ax2.set_title(f'{feature} Q-Q Plot')
            
        plt.tight_layout()
        return fig
```

Slide 13: Mathematical Foundations of Drift Metrics

Implementation of core mathematical concepts behind drift detection metrics, including probability density estimation and statistical distance calculations.

```python
def mathematical_drift_metrics():
    """
    Mathematical formulations of drift metrics
    Formulas are represented in LaTeX notation
    """
    formulas = {
        'kl_divergence': """
        # Kullback-Leibler Divergence
        $$KL(P||Q) = \sum_{x} P(x) \log(\frac{P(x)}{Q(x)})$$
        """,
        
        'wasserstein': """
        # Wasserstein Distance (1-dimensional)
        $$W_1(P,Q) = \int_{-\infty}^{\infty} |F_P(x) - F_Q(x)| dx$$
        """,
        
        'jensen_shannon': """
        # Jensen-Shannon Divergence
        $$JSD(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)$$
        where M = \frac{1}{2}(P + Q)
        """
    }
    return formulas

def calculate_jensen_shannon(p, q):
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate midpoint distribution
    m = (p + q) / 2
    
    # Calculate Jensen-Shannon divergence
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2
```

Slide 14: Production Monitoring Integration

A complete system for integrating drift detection into production ML pipelines, including monitoring, logging, and alerting capabilities.

```python
import logging
from datetime import datetime

class ProductionDriftMonitor:
    def __init__(self, reference_data, alert_threshold=0.1):
        self.reference_data = reference_data
        self.alert_threshold = alert_threshold
        self.logger = self._setup_logger()
        self.drift_history = []
        
    def _setup_logger(self):
        logger = logging.getLogger('DriftMonitor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('drift_monitoring.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def monitor_batch(self, production_batch):
        drift_metrics = self._calculate_drift_metrics(production_batch)
        self.drift_history.append(drift_metrics)
        
        if self._should_alert(drift_metrics):
            self._trigger_alert(drift_metrics)
        
        return drift_metrics
    
    def _calculate_drift_metrics(self, batch):
        w_distance = calculate_wasserstein(self.reference_data, batch)
        kl_div = kl_divergence(
            *np.histogram(self.reference_data, bins=30)[0],
            *np.histogram(batch, bins=30)[0]
        )
        
        return {
            'timestamp': datetime.now(),
            'wasserstein': w_distance,
            'kl_divergence': kl_div,
            'sample_size': len(batch)
        }
    
    def _should_alert(self, metrics):
        return metrics['wasserstein'] > self.alert_threshold
    
    def _trigger_alert(self, metrics):
        alert_msg = (f"Drift Alert: Wasserstein distance {metrics['wasserstein']:.4f} "
                    f"exceeds threshold {self.alert_threshold}")
        self.logger.warning(alert_msg)
```

Slide 15: Additional Resources

*   [https://arxiv.org/abs/2108.13985](https://arxiv.org/abs/2108.13985) - "A Survey on Data Drift Detection Methods"
*   [https://arxiv.org/abs/2004.03045](https://arxiv.org/abs/2004.03045) - "Monitoring and Adapting to Concept Drift in Data Streams"
*   [https://arxiv.org/abs/2007.14101](https://arxiv.org/abs/2007.14101) - "Robust Drift Detection Under Label Shift"
*   [https://arxiv.org/abs/1910.11656](https://arxiv.org/abs/1910.11656) - "Understanding and Detecting Concept Drift"
*   [https://arxiv.org/abs/2107.07421](https://arxiv.org/abs/2107.07421) - "On the Mathematics of Data Drift Detection"

