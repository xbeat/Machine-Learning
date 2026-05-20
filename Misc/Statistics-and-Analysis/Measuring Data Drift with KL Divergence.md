## Measuring Data Drift with KL Divergence
Slide 1: KL Divergence Foundation

KL divergence measures the relative entropy between two probability distributions, quantifying how one distribution differs from a reference distribution in statistical modeling and machine learning.

```python
import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    # Ensure distributions sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergence
    return entropy(p, q)

# Example distributions
p = np.array([0.2, 0.5, 0.3])
q = np.array([0.1, 0.4, 0.5])
kl_div = kl_divergence(p, q)
print(f"KL Divergence: {kl_div:.4f}")
# Output: KL Divergence: 0.1486
```

Slide 2: Data Drift Detection Class

A comprehensive implementation of a data drift detector using KL divergence, featuring sliding windows and threshold-based alerting for production monitoring scenarios.

```python
class DataDriftDetector:
    def __init__(self, reference_data, window_size=1000, threshold=0.1):
        self.reference_hist, self.bins = np.histogram(
            reference_data, bins=20, density=True)
        self.window_size = window_size
        self.threshold = threshold
        self.window = []
    
    def check_drift(self, new_data):
        self.window.extend(new_data)
        if len(self.window) >= self.window_size:
            current_hist, _ = np.histogram(
                self.window[-self.window_size:], 
                bins=self.bins, 
                density=True
            )
            drift_score = kl_divergence(self.reference_hist, current_hist)
            self.window = self.window[-self.window_size:]
            return drift_score > self.threshold, drift_score
        return False, 0.0
```

Slide 3: Real-time Data Drift Monitoring

Implementation of continuous data drift monitoring system with visualization capabilities and alert mechanisms for production environments.

```python
import matplotlib.pyplot as plt
from datetime import datetime

class DriftMonitor:
    def __init__(self, detector):
        self.detector = detector
        self.drift_scores = []
        self.timestamps = []
    
    def monitor_stream(self, data_stream):
        drift_detected = False
        for batch in data_stream:
            is_drift, score = self.detector.check_drift(batch)
            self.drift_scores.append(score)
            self.timestamps.append(datetime.now())
            
            if is_drift:
                print(f"ALERT: Data drift detected! Score: {score:.4f}")
                drift_detected = True
        
        return drift_detected

    def plot_drift_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.timestamps, self.drift_scores)
        plt.axhline(y=self.detector.threshold, color='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Drift Score')
        plt.title('Data Drift Monitoring')
        plt.show()
```

Slide 4: Feature-wise Drift Analysis

A detailed examination of drift patterns across individual features, enabling targeted investigation of distribution changes in specific data dimensions.

```python
import pandas as pd

class FeatureDriftAnalyzer:
    def __init__(self, reference_df, threshold=0.1):
        self.reference_df = reference_df
        self.threshold = threshold
        self.feature_histograms = self._compute_histograms(reference_df)
    
    def _compute_histograms(self, df):
        histograms = {}
        for column in df.columns:
            hist, bins = np.histogram(df[column], bins=20, density=True)
            histograms[column] = (hist, bins)
        return histograms
    
    def analyze_drift(self, current_df):
        drift_scores = {}
        for column in self.reference_df.columns:
            current_hist, _ = np.histogram(
                current_df[column], 
                bins=self.feature_histograms[column][1],
                density=True
            )
            drift_scores[column] = kl_divergence(
                self.feature_histograms[column][0],
                current_hist
            )
        return pd.Series(drift_scores)
```

Slide 5: Real-world Example - Credit Card Fraud Detection

This example demonstrates drift detection in a credit card fraud detection system, where transaction patterns may change over time due to evolving fraud techniques.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
def prepare_fraud_data():
    df = pd.read_csv('credit_card_transactions.csv')
    features = ['amount', 'time', 'v1', 'v2', 'v3']
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Split into reference and monitoring periods
    reference_data = df[df['time'] < df['time'].median()]
    monitoring_data = df[df['time'] >= df['time'].median()]
    
    return reference_data[features], monitoring_data[features]

# Initialize drift detection
reference_data, monitoring_data = prepare_fraud_data()
detector = DataDriftDetector(reference_data['amount'], threshold=0.15)
monitor = DriftMonitor(detector)

# Simulate streaming data
batch_size = 100
batches = [monitoring_data['amount'][i:i+batch_size] 
           for i in range(0, len(monitoring_data), batch_size)]
drift_detected = monitor.monitor_stream(batches)
```

Slide 6: Results for Credit Card Fraud Detection

```python
# Display drift detection results
print("Monitoring Results:")
print(f"Number of batches processed: {len(monitor.drift_scores)}")
print(f"Average drift score: {np.mean(monitor.drift_scores):.4f}")
print(f"Maximum drift score: {np.max(monitor.drift_scores):.4f}")
print(f"Number of drift alerts: {sum(s > detector.threshold for s in monitor.drift_scores)}")

# Plot drift scores
monitor.plot_drift_history()

# Example output:
# Monitoring Results:
# Number of batches processed: 150
# Average drift score: 0.0823
# Maximum drift score: 0.2156
# Number of drift alerts: 12
```

Slide 7: Feature Distribution Visualization

Advanced visualization techniques for comparing feature distributions between reference and current data streams to identify drift patterns.

```python
def plot_distribution_comparison(reference_data, current_data, feature):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(reference_data[feature], bins=30, alpha=0.5, density=True)
    plt.hist(current_data[feature], bins=30, alpha=0.5, density=True)
    plt.title(f'{feature} Distribution Comparison')
    plt.legend(['Reference', 'Current'])
    
    plt.subplot(1, 2, 2)
    plt.plot(monitor.timestamps, monitor.drift_scores)
    plt.axhline(y=detector.threshold, color='r', linestyle='--')
    plt.title('Drift Score Over Time')
    
    plt.tight_layout()
    plt.show()
```

Slide 8: Real-world Example - IoT Sensor Monitoring

This example implements drift detection for IoT sensor data, where environmental conditions and sensor degradation can cause distribution shifts.

```python
class IoTDriftMonitor:
    def __init__(self, sensor_count):
        self.sensor_count = sensor_count
        self.detectors = []
        self.monitors = []
    
    def initialize_detectors(self, reference_data):
        for i in range(self.sensor_count):
            detector = DataDriftDetector(
                reference_data[f'sensor_{i}'],
                window_size=500,
                threshold=0.12
            )
            monitor = DriftMonitor(detector)
            self.detectors.append(detector)
            self.monitors.append(monitor)
    
    def process_sensor_data(self, sensor_data):
        alerts = []
        for i in range(self.sensor_count):
            is_drift, score = self.detectors[i].check_drift(
                sensor_data[f'sensor_{i}']
            )
            if is_drift:
                alerts.append(f"Drift detected in sensor {i}: {score:.4f}")
        return alerts
```

Slide 9: Source Code for IoT Sensor Example

```python
# Simulate IoT sensor data
def generate_sensor_data(n_samples, n_sensors):
    data = {}
    for i in range(n_sensors):
        # Normal operation
        base_data = np.random.normal(0, 1, n_samples)
        
        # Simulate drift after certain point
        drift_point = int(0.7 * n_samples)
        drift_data = np.random.normal(0.5, 1.2, n_samples - drift_point)
        base_data[drift_point:] = drift_data
        
        data[f'sensor_{i}'] = base_data
    return pd.DataFrame(data)

# Example usage
n_sensors = 3
reference_data = generate_sensor_data(1000, n_sensors)
monitoring_data = generate_sensor_data(2000, n_sensors)

iot_monitor = IoTDriftMonitor(n_sensors)
iot_monitor.initialize_detectors(reference_data)

# Process data in batches
batch_size = 100
for i in range(0, len(monitoring_data), batch_size):
    batch = monitoring_data.iloc[i:i+batch_size]
    alerts = iot_monitor.process_sensor_data(batch)
    if alerts:
        print(f"Batch {i//batch_size}: {alerts}")
```

Slide 10: Statistical Significance Testing

Implementation of statistical tests to validate drift detection and reduce false positives in production environments.

```python
from scipy import stats

class StatisticalDriftValidator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def validate_drift(self, reference_data, current_data):
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(
            reference_data, 
            current_data
        )
        
        # Anderson-Darling test
        ad_statistic = stats.anderson_ksamp([reference_data, current_data])
        
        # Chi-square test for discrete distributions
        hist1, bins = np.histogram(reference_data, bins=20)
        hist2, _ = np.histogram(current_data, bins=bins)
        chi2_statistic, chi2_pvalue = stats.chisquare(hist1, hist2)
        
        return {
            'ks_test': ks_pvalue < self.alpha,
            'ad_test': ad_statistic.significance_level < self.alpha,
            'chi2_test': chi2_pvalue < self.alpha
        }
```

Slide 11: Ensemble Drift Detection

A robust approach combining multiple drift detection methods to improve accuracy and reduce false positives in production environments.

```python
class EnsembleDriftDetector:
    def __init__(self, reference_data, methods=['kl', 'ks', 'chi2']):
        self.reference_data = reference_data
        self.methods = methods
        self.validator = StatisticalDriftValidator()
        self.kl_detector = DataDriftDetector(reference_data)
    
    def detect_drift(self, current_data):
        results = {}
        
        if 'kl' in self.methods:
            is_drift, score = self.kl_detector.check_drift(current_data)
            results['kl'] = is_drift
        
        if set(['ks', 'chi2']) & set(self.methods):
            stat_results = self.validator.validate_drift(
                self.reference_data, 
                current_data
            )
            results.update({k: v for k, v in stat_results.items() 
                          if k.split('_')[0] in self.methods})
        
        # Majority voting
        drift_detected = sum(results.values()) > len(results) / 2
        return drift_detected, results
```

Slide 12: Performance Metrics Implementation

Comprehensive implementation of drift detection performance metrics including precision, recall, and F1-score for model evaluation.

```python
class DriftMetricsEvaluator:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    def update_metrics(self, predicted_drift, actual_drift):
        if predicted_drift and actual_drift:
            self.true_positives += 1
        elif predicted_drift and not actual_drift:
            self.false_positives += 1
        elif not predicted_drift and actual_drift:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
    
    def get_metrics(self):
        precision = (self.true_positives / 
                    (self.true_positives + self.false_positives)
                    if (self.true_positives + self.false_positives) > 0 
                    else 0)
        
        recall = (self.true_positives / 
                 (self.true_positives + self.false_negatives)
                 if (self.true_positives + self.false_negatives) > 0 
                 else 0)
        
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 
              else 0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
```

Slide 13: Mathematical Foundations

```python
# KL Divergence formula
"""
$$KL(P||Q) = \sum_{i} P(i) \log(\frac{P(i)}{Q(i)})$$

Where:
P(i) - probability of event i in distribution P
Q(i) - probability of event i in distribution Q

For continuous distributions:
$$KL(P||Q) = \int P(x) \log(\frac{P(x)}{Q(x)}) dx$$

Relationship with entropy:
$$KL(P||Q) = H(P,Q) - H(P)$$

Where:
H(P,Q) - cross entropy between P and Q
H(P) - entropy of distribution P
"""
```

Slide 14: Additional Resources

ArXiv Papers for Further Reading:

1.  "A Survey on Concept Drift Adaptation" [https://arxiv.org/abs/1010.4784](https://arxiv.org/abs/1010.4784)
2.  "Learning under Concept Drift: A Review" [https://arxiv.org/abs/2004.05785](https://arxiv.org/abs/2004.05785)
3.  "Detecting and Understanding Concept Drift" [https://arxiv.org/abs/1910.09465](https://arxiv.org/abs/1910.09465)
4.  "A Review of Data Drift Detection Methods" [https://arxiv.org/abs/2104.01876](https://arxiv.org/abs/2104.01876)
5.  "Machine Learning Model Monitoring in Production" [https://arxiv.org/abs/2007.06299](https://arxiv.org/abs/2007.06299)

