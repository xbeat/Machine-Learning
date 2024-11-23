## Concept and Data Drift in Machine Learning
Slide 1: Understanding Data Drift Detection

Data drift detection requires statistical methods to monitor changes in feature distributions over time. We'll implement a basic drift detector using Kolmogorov-Smirnov test to compare distributions between training and current data windows, with visualization capabilities.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class DataDriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, current_data):
        statistic, p_value = stats.ks_2samp(self.reference_data, current_data)
        is_drift = p_value < self.threshold
        return {
            'drift_detected': is_drift,
            'p_value': p_value,
            'statistic': statistic
        }
    
    def visualize_distributions(self, current_data):
        plt.figure(figsize=(10, 6))
        plt.hist(self.reference_data, bins=30, alpha=0.5, label='Reference')
        plt.hist(current_data, bins=30, alpha=0.5, label='Current')
        plt.legend()
        plt.title('Distribution Comparison')
        plt.show()

# Example usage
reference_data = np.random.normal(0, 1, 1000)
current_data = np.random.normal(0.5, 1, 1000)  # Shifted distribution

detector = DataDriftDetector(reference_data)
result = detector.detect_drift(current_data)
print(f"Drift detected: {result['drift_detected']}")
detector.visualize_distributions(current_data)
```

Slide 2: Implementing Concept Drift Detection

Concept drift detection involves monitoring prediction errors over time. This implementation uses the Page-Hinkley test, which can detect changes in the probability distribution of a time series, particularly useful for online learning scenarios.

```python
class PageHinkleyTest:
    def __init__(self, threshold=50, alpha=0.005):
        self.threshold = threshold
        self.alpha = alpha
        self.running_mean = 0
        self.sum = 0
        self.sample_count = 0
        self.min_value = 0
        
    def update(self, value):
        self.sample_count += 1
        self.running_mean = (self.running_mean * (self.sample_count - 1) + 
                           value) / self.sample_count
        self.sum = max(0, self.sum + value - self.running_mean - self.alpha)
        self.min_value = min(self.sum, self.min_value)
        
        return self.sum - self.min_value > self.threshold
    
# Example usage
ph_test = PageHinkleyTest()
# Simulate concept drift with changing error patterns
errors = np.concatenate([np.random.normal(0, 1, 100), 
                        np.random.normal(2, 1, 100)])

drift_points = []
for i, error in enumerate(errors):
    if ph_test.update(error):
        drift_points.append(i)
        print(f"Concept drift detected at point {i}")
```

Slide 3: Feature Distribution Analysis

Understanding how feature distributions change over time is crucial for drift detection. This implementation creates a comprehensive feature analyzer that tracks statistical properties and visualizes distribution changes across multiple time windows.

```python
import pandas as pd
from scipy import stats

class FeatureDistributionAnalyzer:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.reference_stats = {}
        
    def compute_statistics(self, data):
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skew': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75)
        }
    
    def set_reference(self, feature_data):
        self.reference_stats = self.compute_statistics(feature_data)
    
    def analyze_drift(self, current_data):
        current_stats = self.compute_statistics(current_data)
        differences = {}
        for metric in self.reference_stats.keys():
            diff_pct = ((current_stats[metric] - self.reference_stats[metric]) / 
                       self.reference_stats[metric] * 100)
            differences[metric] = diff_pct
        return differences

# Example usage
analyzer = FeatureDistributionAnalyzer()
reference_data = np.random.normal(0, 1, 1000)
current_data = np.random.normal(0.5, 1.2, 1000)

analyzer.set_reference(reference_data)
differences = analyzer.analyze_drift(current_data)
print("Distribution changes (%):")
for metric, change in differences.items():
    print(f"{metric}: {change:.2f}%")
```

Slide 4: Real-Time Drift Monitoring System

A comprehensive drift monitoring system that combines both data and concept drift detection. This implementation uses sliding windows and multiple statistical tests to provide real-time alerts and monitoring capabilities for production ML systems.

```python
import numpy as np
from collections import deque
from sklearn.metrics import accuracy_score

class DriftMonitor:
    def __init__(self, window_size=1000, drift_threshold=0.05):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.performance_window = deque(maxlen=window_size)
        
    def add_observation(self, features, target, prediction):
        # Track feature distributions
        self.current_window.append(features)
        
        # Track performance
        self.performance_window.append(int(target == prediction))
        
        if len(self.current_window) == self.window_size:
            return self._check_drift()
        return None
    
    def _check_drift(self):
        # Check data drift using KS test
        ks_statistic, p_value = stats.ks_2samp(
            np.array(self.reference_window).flatten(),
            np.array(self.current_window).flatten()
        )
        
        # Check concept drift using performance degradation
        current_performance = np.mean(self.performance_window)
        reference_performance = self.initial_performance
        
        return {
            'data_drift': p_value < self.drift_threshold,
            'concept_drift': (reference_performance - current_performance) > 
                            self.drift_threshold,
            'performance_drop': reference_performance - current_performance,
            'p_value': p_value
        }

    def set_reference(self, initial_data, initial_performance):
        self.reference_window.extend(initial_data)
        self.initial_performance = initial_performance

# Example usage
monitor = DriftMonitor()
initial_data = np.random.normal(0, 1, 1000)
initial_performance = 0.95
monitor.set_reference(initial_data, initial_performance)

# Simulate streaming data
for i in range(100):
    features = np.random.normal(0.1 * i, 1, 10)  # Gradually shifting distribution
    target = 1 if np.mean(features) > 0 else 0
    prediction = 1 if np.random.random() > 0.2 else 0  # Simulated predictions
    
    result = monitor.add_observation(features, target, prediction)
    if result:
        print(f"Iteration {i}: {result}")
```

Slide 5: Advanced Concept Drift Detection Using ADWIN

The Adaptive Windowing (ADWIN) algorithm is a sophisticated approach for detecting concept drift by maintaining a variable-size window of recent examples and automatically growing or shrinking it based on observed changes.

```python
class ADWIN:
    def __init__(self, delta=0.002):
        self.delta = delta
        self.bucket_row = []
        self.bucket_sizes = []
        self.total = 0
        self.variance = 0
        self.width = 0
        
    def update(self, value):
        self._insert_element(value)
        self._compress_buckets()
        return self._check_drift()
        
    def _insert_element(self, value):
        self.bucket_row.append([value])
        self.bucket_sizes.append(1)
        self.total += value
        self.width += 1
        
    def _compress_buckets(self):
        i = 0
        while i < len(self.bucket_row):
            if i + 1 < len(self.bucket_row):
                if len(self.bucket_row[i]) == len(self.bucket_row[i + 1]):
                    new_bucket = self._merge_buckets(
                        self.bucket_row[i], 
                        self.bucket_row[i + 1]
                    )
                    self.bucket_row.pop(i)
                    self.bucket_row.pop(i)
                    self.bucket_row.insert(i, new_bucket)
                    self.bucket_sizes.pop(i)
                    self.bucket_sizes.pop(i)
                    self.bucket_sizes.insert(
                        i, 
                        len(new_bucket)
                    )
                    i -= 1
            i += 1
            
    def _merge_buckets(self, bucket1, bucket2):
        return bucket1 + bucket2
        
    def _check_drift(self):
        for i in range(len(self.bucket_row)):
            if self._cut_expression(i):
                self.bucket_row = self.bucket_row[i + 1:]
                self.bucket_sizes = self.bucket_sizes[i + 1:]
                return True
        return False
        
    def _cut_expression(self, i):
        if i + 1 >= len(self.bucket_row):
            return False
            
        n0 = sum(self.bucket_sizes[:i + 1])
        n1 = sum(self.bucket_sizes[i + 1:])
        
        if n0 < 1 or n1 < 1:
            return False
            
        total0 = sum(sum(bucket) for bucket in self.bucket_row[:i + 1])
        total1 = sum(sum(bucket) for bucket in self.bucket_row[i + 1:])
        
        μ0 = total0 / n0
        μ1 = total1 / n1
        
        m = (1 / (1 / n0 + 1 / n1)) * np.log(4 / self.delta)
        
        return abs(μ0 - μ1) > np.sqrt(2 * m * self.variance)

# Example usage
adwin = ADWIN()
# Generate data with concept drift
data = np.concatenate([
    np.random.normal(0, 1, 1000),
    np.random.normal(2, 1, 1000)
])

drift_points = []
for i, value in enumerate(data):
    if adwin.update(value):
        drift_points.append(i)
        print(f"Concept drift detected at point {i}")
```

Slide 6: Feature Importance Drift Analysis

This implementation focuses on detecting changes in feature importance over time, which can indicate subtle forms of concept drift not captured by traditional methods. It uses permutation importance comparison between time windows.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

class FeatureImportanceDrift:
    def __init__(self, model, feature_names, window_size=1000):
        self.model = model
        self.feature_names = feature_names
        self.window_size = window_size
        self.reference_importance = None
        
    def compute_importance(self, X, y):
        result = permutation_importance(
            self.model, X, y,
            n_repeats=10,
            random_state=42
        )
        return dict(zip(
            self.feature_names,
            result.importances_mean
        ))
        
    def set_reference(self, X_ref, y_ref):
        self.model.fit(X_ref, y_ref)
        self.reference_importance = self.compute_importance(X_ref, y_ref)
        
    def detect_importance_drift(self, X_current, y_current, threshold=0.1):
        current_importance = self.compute_importance(X_current, y_current)
        
        drifts = {}
        for feature in self.feature_names:
            ref_imp = self.reference_importance[feature]
            cur_imp = current_importance[feature]
            
            if ref_imp > 0:  # Avoid division by zero
                relative_change = abs(cur_imp - ref_imp) / ref_imp
                drifts[feature] = {
                    'relative_change': relative_change,
                    'is_drift': relative_change > threshold,
                    'reference_importance': ref_imp,
                    'current_importance': cur_imp
                }
                
        return drifts

# Example usage
from sklearn.datasets import make_classification

# Generate synthetic data
X_ref, y_ref = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    random_state=42
)

X_cur, y_cur = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=2,  # Changed importance
    random_state=43
)

feature_names = [f'feature_{i}' for i in range(5)]
model = RandomForestClassifier(random_state=42)

drift_detector = FeatureImportanceDrift(model, feature_names)
drift_detector.set_reference(X_ref, y_ref)

results = drift_detector.detect_importance_drift(X_cur, y_cur)
for feature, metrics in results.items():
    print(f"\n{feature}:")
    print(f"Drift detected: {metrics['is_drift']}")
    print(f"Relative change: {metrics['relative_change']:.2f}")
```

Slide 7: Statistical Process Control for Drift Detection

Statistical Process Control (SPC) provides a robust framework for monitoring model performance over time using control charts. This implementation uses CUSUM (Cumulative Sum) charts to detect subtle, persistent changes in model metrics.

```python
import numpy as np
from scipy import stats

class CUSUMDriftDetector:
    def __init__(self, target_mean, std_dev, drift_threshold=4.0, slack=0.5):
        self.target_mean = target_mean
        self.std_dev = std_dev
        self.threshold = drift_threshold * std_dev
        self.slack = slack * std_dev
        self.reset_stats()
        
    def reset_stats(self):
        self.pos_cusums = [0]
        self.neg_cusums = [0]
        self.means = []
        
    def update(self, new_value):
        self.means.append(new_value)
        
        # Calculate standardized distance from target
        pos_diff = new_value - (self.target_mean + self.slack)
        neg_diff = (self.target_mean - self.slack) - new_value
        
        # Update CUSUM values
        self.pos_cusums.append(max(0, self.pos_cusums[-1] + 
                                 pos_diff/self.std_dev))
        self.neg_cusums.append(max(0, self.neg_cusums[-1] + 
                                 neg_diff/self.std_dev))
        
        # Check for drift
        drift_status = {
            'positive_drift': self.pos_cusums[-1] > self.threshold,
            'negative_drift': self.neg_cusums[-1] > self.threshold,
            'current_pos_cusum': self.pos_cusums[-1],
            'current_neg_cusum': self.neg_cusums[-1]
        }
        
        return drift_status

# Example usage with simulated model accuracy
np.random.seed(42)
detector = CUSUMDriftDetector(
    target_mean=0.85,  # Expected accuracy
    std_dev=0.02,      # Expected variation
    drift_threshold=3.0
)

# Simulate gradual accuracy degradation
n_points = 100
base_accuracy = np.random.normal(0.85, 0.02, n_points)
drift_accuracy = np.random.normal(0.80, 0.02, n_points)  # Lower accuracy
combined_accuracy = np.concatenate([base_accuracy, drift_accuracy])

drift_detected = False
for i, acc in enumerate(combined_accuracy):
    result = detector.update(acc)
    if (result['positive_drift'] or result['negative_drift']) and not drift_detected:
        print(f"Drift detected at point {i}")
        print(f"CUSUM stats: {result}")
        drift_detected = True
```

Slide 8: Multivariate Drift Detection Using MMD

Maximum Mean Discrepancy (MMD) is a powerful method for detecting multivariate distribution changes. This implementation uses kernel-based MMD to compare feature distributions between reference and current windows.

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class MMDDriftDetector:
    def __init__(self, window_size=1000, alpha=0.05, kernel_bandwidth='median'):
        self.window_size = window_size
        self.alpha = alpha
        self.kernel_bandwidth = kernel_bandwidth
        self.reference_window = None
        
    def set_reference(self, reference_data):
        self.reference_window = reference_data
        if self.kernel_bandwidth == 'median':
            self.kernel_bandwidth = self._compute_kernel_bandwidth(reference_data)
    
    def _compute_kernel_bandwidth(self, data):
        # Median heuristic for kernel bandwidth
        pairwise_dists = np.linalg.norm(
            data[:, None, :] - data[None, :, :], 
            axis=-1
        )
        return np.median(pairwise_dists[pairwise_dists > 0])
    
    def _compute_mmd(self, X, Y):
        # Compute MMD^2 statistic
        K_XX = rbf_kernel(X, X, gamma=1.0/self.kernel_bandwidth)
        K_YY = rbf_kernel(Y, Y, gamma=1.0/self.kernel_bandwidth)
        K_XY = rbf_kernel(X, Y, gamma=1.0/self.kernel_bandwidth)
        
        mmd2 = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
        return np.sqrt(max(mmd2, 0))
    
    def detect_drift(self, current_data):
        if self.reference_window is None:
            raise ValueError("Reference window not set")
            
        mmd_value = self._compute_mmd(self.reference_window, current_data)
        
        # Permutation test for significance
        n_permutations = 100
        combined = np.vstack([self.reference_window, current_data])
        n = len(self.reference_window)
        permutation_mmd = []
        
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_mmd = self._compute_mmd(combined[:n], combined[n:])
            permutation_mmd.append(perm_mmd)
            
        p_value = np.mean(np.array(permutation_mmd) >= mmd_value)
        
        return {
            'drift_detected': p_value < self.alpha,
            'mmd_value': mmd_value,
            'p_value': p_value
        }

# Example usage
np.random.seed(42)

# Generate reference and current data
reference_data = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=1000
)

# Current data with drift
current_data = np.random.multivariate_normal(
    mean=[0.5, 0.5],
    cov=[[1.2, 0.7], [0.7, 1.2]],
    size=1000
)

detector = MMDDriftDetector()
detector.set_reference(reference_data)
result = detector.detect_drift(current_data)

print("Drift detection results:")
print(f"Drift detected: {result['drift_detected']}")
print(f"MMD value: {result['mmd_value']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
```

Slide 9: Ensemble Drift Detection System

This implementation combines multiple drift detection methods to create a robust ensemble system. It uses weighted voting and maintains confidence scores for each detector based on their historical performance.

```python
import numpy as np
from sklearn.base import BaseEstimator
from collections import defaultdict

class EnsembleDriftDetector:
    def __init__(self, detectors, weights=None):
        self.detectors = detectors
        self.weights = weights if weights else [1/len(detectors)] * len(detectors)
        self.performance_history = defaultdict(list)
        self.confidence_scores = np.array(self.weights)
        
    def update_detector_confidence(self, detector_idx, correct_detection):
        # Update confidence using exponential moving average
        alpha = 0.1
        current_score = self.confidence_scores[detector_idx]
        new_score = alpha * float(correct_detection) + (1 - alpha) * current_score
        self.confidence_scores[detector_idx] = new_score
        
    def detect_drift(self, reference_data, current_data):
        detector_votes = []
        
        for idx, detector in enumerate(self.detectors):
            try:
                result = detector.detect_drift(reference_data, current_data)
                detector_votes.append({
                    'detector': type(detector).__name__,
                    'drift_detected': result['drift_detected'],
                    'confidence': self.confidence_scores[idx],
                    'metadata': result
                })
            except Exception as e:
                print(f"Detector {type(detector).__name__} failed: {str(e)}")
                
        # Weighted voting
        weighted_vote = 0
        total_confidence = sum(self.confidence_scores)
        
        for idx, vote in enumerate(detector_votes):
            if vote['drift_detected']:
                weighted_vote += self.confidence_scores[idx]
                
        drift_detected = weighted_vote / total_confidence > 0.5
        
        return {
            'drift_detected': drift_detected,
            'weighted_score': weighted_vote / total_confidence,
            'individual_votes': detector_votes
        }
    
    def update_ensemble(self, correct_drift):
        # Update detector confidence based on performance
        for idx, detector in enumerate(self.detectors):
            detector_correct = (detector.last_prediction == correct_drift)
            self.update_detector_confidence(idx, detector_correct)

# Example implementation of base detectors
class KSDetector:
    def detect_drift(self, reference_data, current_data):
        statistic, p_value = stats.ks_2samp(reference_data, current_data)
        drift_detected = p_value < 0.05
        self.last_prediction = drift_detected
        return {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'statistic': statistic
        }

class CUSUMDetector:
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        
    def detect_drift(self, reference_data, current_data):
        cusum = np.cumsum(current_data - np.mean(reference_data))
        drift_detected = np.abs(cusum).max() > self.threshold
        self.last_prediction = drift_detected
        return {
            'drift_detected': drift_detected,
            'max_cusum': np.abs(cusum).max()
        }

# Example usage
np.random.seed(42)

# Create ensemble
detectors = [
    KSDetector(),
    CUSUMDetector(threshold=2.0),
    MMDDriftDetector()  # From previous slide
]

ensemble = EnsembleDriftDetector(detectors)

# Generate test data
reference_data = np.random.normal(0, 1, 1000)
current_data = np.random.normal(0.5, 1.2, 1000)  # With drift

# Detect drift
result = ensemble.detect_drift(reference_data, current_data)

print("Ensemble Drift Detection Results:")
print(f"Overall drift detected: {result['drift_detected']}")
print(f"Weighted confidence score: {result['weighted_score']:.3f}")
print("\nIndividual detector votes:")
for vote in result['individual_votes']:
    print(f"{vote['detector']}: {vote['drift_detected']} "
          f"(confidence: {vote['confidence']:.3f})")
```

Slide 10: Online Concept Drift Adaptation

This implementation creates an adaptive learning system that can automatically update itself when concept drift is detected, using a sliding window approach and model retraining strategies.

```python
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import numpy as np
from collections import deque

class AdaptiveLearningSystem:
    def __init__(self, base_model, window_size=1000, 
                 drift_threshold=0.05, adaptation_rate=0.3):
        self.base_model = base_model
        self.current_model = clone(base_model)
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.adaptation_rate = adaptation_rate
        
        # Initialize windows
        self.X_window = deque(maxlen=window_size)
        self.y_window = deque(maxlen=window_size)
        self.performance_window = deque(maxlen=window_size)
        
        self.baseline_performance = None
        self.retrain_count = 0
        
    def partial_fit(self, X, y):
        # Update windows
        for x_i, y_i in zip(X, y):
            self.X_window.append(x_i)
            self.y_window.append(y_i)
            
            # Make prediction and update performance
            if len(self.X_window) > 1:
                pred = self.current_model.predict([x_i])[0]
                acc = int(pred == y_i)
                self.performance_window.append(acc)
                
                # Check for drift
                if self._detect_performance_drift():
                    self._adapt_model()
                    
        # Initial fit or retrain on window
        if len(self.X_window) >= self.window_size:
            self._retrain_model()
            
    def _detect_performance_drift(self):
        if len(self.performance_window) < self.window_size:
            return False
            
        current_performance = np.mean(list(self.performance_window)[-100:])
        
        if self.baseline_performance is None:
            self.baseline_performance = current_performance
            return False
            
        performance_drop = self.baseline_performance - current_performance
        return performance_drop > self.drift_threshold
        
    def _adapt_model(self):
        # Implement adaptation strategy
        recent_X = np.array(list(self.X_window)[-int(self.window_size *
                                                    self.adaptation_rate):])
        recent_y = np.array(list(self.y_window)[-int(self.window_size * 
                                                    self.adaptation_rate):])
        
        # Create new model instance
        new_model = clone(self.base_model)
        new_model.fit(recent_X, recent_y)
        
        # Evaluate new model
        new_performance = accuracy_score(
            recent_y,
            new_model.predict(recent_X)
        )
        
        if new_performance > self.baseline_performance:
            self.current_model = new_model
            self.baseline_performance = new_performance
            self.retrain_count += 1
            
    def _retrain_model(self):
        X_array = np.array(list(self.X_window))
        y_array = np.array(list(self.y_window))
        self.current_model.fit(X_array, y_array)
        self.baseline_performance = accuracy_score(
            y_array,
            self.current_model.predict(X_array)
        )

# Example usage
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create synthetic data with concept drift
np.random.seed(42)

def generate_data(n_samples, concept=0):
    if concept == 0:
        X = np.random.normal(0, 1, (n_samples, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:
        X = np.random.normal(0, 1, (n_samples, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

# Initialize adaptive system
base_model = DecisionTreeClassifier(random_state=42)
adaptive_system = AdaptiveLearningSystem(base_model)

# Training with concept drift
n_samples = 5000
X1, y1 = generate_data(n_samples, concept=0)
X2, y2 = generate_data(n_samples, concept=1)

# Online learning
for i in range(n_samples):
    if i < n_samples // 2:
        adaptive_system.partial_fit(X1[i:i+1], y1[i:i+1])
    else:
        adaptive_system.partial_fit(X2[i-n_samples//2:i-n_samples//2+1], 
                                  y2[i-n_samples//2:i-n_samples//2+1])
        
print(f"Model retrains due to drift: {adaptive_system.retrain_count}")
```

Slide 11: Visualization Dashboard for Drift Monitoring

This implementation creates a comprehensive visualization system for monitoring different types of drift in real-time, including distribution shifts, performance metrics, and feature importance changes.

```python
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DriftVisualizationDashboard:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.performance_history = []
        self.distribution_metrics = {feat: [] for feat in feature_names}
        self.drift_alerts = []
        self.timestamps = []
        
    def update(self, timestamp, performance, distribution_data, drift_detected=False):
        self.timestamps.append(timestamp)
        self.performance_history.append(performance)
        
        for feat, value in distribution_data.items():
            self.distribution_metrics[feat].append(value)
            
        if drift_detected:
            self.drift_alerts.append(timestamp)
            
    def plot_dashboard(self, window_size=100):
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 10))
        
        # Performance Timeline
        ax1 = plt.subplot(311)
        ax1.plot(self.timestamps[-window_size:], 
                self.performance_history[-window_size:], 
                label='Model Performance')
        ax1.set_title('Performance Timeline')
        ax1.set_ylabel('Accuracy')
        
        # Mark drift points
        for drift_time in self.drift_alerts:
            if drift_time in self.timestamps[-window_size:]:
                ax1.axvline(x=drift_time, color='r', linestyle='--', alpha=0.5)
        
        # Feature Distribution Changes
        ax2 = plt.subplot(312)
        for feat in self.feature_names:
            ax2.plot(self.timestamps[-window_size:], 
                    self.distribution_metrics[feat][-window_size:], 
                    label=feat)
        ax2.set_title('Feature Distribution Changes')
        ax2.set_ylabel('Distribution Metric')
        ax2.legend()
        
        # Drift Probability Heatmap
        ax3 = plt.subplot(313)
        drift_matrix = self._compute_drift_probability_matrix(window_size)
        im = ax3.imshow(drift_matrix, aspect='auto', cmap='YlOrRd')
        ax3.set_title('Feature Drift Probability Heatmap')
        ax3.set_ylabel('Features')
        ax3.set_yticks(range(len(self.feature_names)))
        ax3.set_yticklabels(self.feature_names)
        plt.colorbar(im)
        
        plt.tight_layout()
        return fig
    
    def _compute_drift_probability_matrix(self, window_size):
        matrix = np.zeros((len(self.feature_names), window_size))
        for i, feat in enumerate(self.feature_names):
            values = self.distribution_metrics[feat][-window_size:]
            # Normalize to [0,1] range for probability visualization
            if values:
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    matrix[i] = [(v - min_val)/(max_val - min_val) 
                               for v in values]
        return matrix

# Example usage
np.random.seed(42)

# Generate synthetic monitoring data
feature_names = ['feature_1', 'feature_2', 'feature_3']
dashboard = DriftVisualizationDashboard(feature_names)

# Simulate real-time monitoring
start_time = datetime.now()
n_points = 200

for i in range(n_points):
    # Simulate time passing
    current_time = start_time + timedelta(hours=i)
    
    # Simulate performance metrics
    base_performance = 0.85
    if i > n_points/2:  # Simulate performance degradation
        base_performance = 0.75
    performance = base_performance + np.random.normal(0, 0.05)
    
    # Simulate distribution metrics
    distribution_data = {
        'feature_1': np.random.normal(0, 1 + i/100),
        'feature_2': np.random.normal(0, 1 + (i > n_points/2) * 0.5),
        'feature_3': np.random.normal(0, 1)
    }
    
    # Detect drift based on threshold
    drift_detected = any(abs(v) > 2 for v in distribution_data.values())
    
    dashboard.update(current_time, performance, distribution_data, drift_detected)

# Generate visualization
fig = dashboard.plot_dashboard(window_size=100)
plt.show()

# Print summary statistics
print("\nDrift Monitoring Summary:")
print(f"Total monitoring period: {n_points} hours")
print(f"Number of drift alerts: {len(dashboard.drift_alerts)}")
print(f"Average performance: {np.mean(dashboard.performance_history):.3f}")
```

Slide 12: Supervised Drift Detection with Label Information

This implementation leverages label information to detect concept drift by monitoring class-conditional distributions and error patterns, providing more precise drift detection in supervised learning contexts.

```python
import numpy as np
from scipy import stats
from collections import defaultdict
from sklearn.metrics import confusion_matrix

class SupervisedDriftDetector:
    def __init__(self, n_classes, window_size=1000, alpha=0.05):
        self.n_classes = n_classes
        self.window_size = window_size
        self.alpha = alpha
        
        # Initialize storage for class-conditional statistics
        self.reference_stats = defaultdict(dict)
        self.current_window = defaultdict(list)
        self.error_patterns = []
        
    def fit_reference(self, X, y):
        """Compute reference statistics for each class"""
        for class_label in range(self.n_classes):
            class_samples = X[y == class_label]
            
            if len(class_samples) > 0:
                self.reference_stats[class_label] = {
                    'mean': np.mean(class_samples, axis=0),
                    'cov': np.cov(class_samples.T),
                    'size': len(class_samples)
                }
    
    def update(self, X, y, y_pred):
        """Update detection statistics with new samples"""
        # Update class-conditional windows
        for i, (x, true_label, pred_label) in enumerate(zip(X, y, y_pred)):
            self.current_window[true_label].append(x)
            
            # Track error patterns
            self.error_patterns.append(int(true_label != pred_label))
            
            # Maintain window size
            if len(self.error_patterns) > self.window_size:
                self.error_patterns.pop(0)
                for class_label in range(self.n_classes):
                    if len(self.current_window[class_label]) > 0:
                        self.current_window[class_label].pop(0)
        
        return self._check_drift()
    
    def _check_drift(self):
        drift_results = {
            'concept_drift': False,
            'class_drifts': defaultdict(bool),
            'error_rate_change': False,
            'details': {}
        }
        
        # Check class-conditional distribution changes
        for class_label in range(self.n_classes):
            if (len(self.current_window[class_label]) >= 30 and 
                class_label in self.reference_stats):
                
                current_samples = np.array(self.current_window[class_label])
                
                # Hotelling's T-squared test for multivariate drift
                t2_stat, p_value = self._hotelling_t2_test(
                    self.reference_stats[class_label],
                    current_samples
                )
                
                drift_results['class_drifts'][class_label] = p_value < self.alpha
                drift_results['details'][f'class_{class_label}_pvalue'] = p_value
        
        # Check error pattern changes
        if len(self.error_patterns) >= self.window_size:
            baseline_error = np.mean(self.error_patterns[:self.window_size//2])
            current_error = np.mean(self.error_patterns[self.window_size//2:])
            
            error_change = abs(current_error - baseline_error)
            drift_results['error_rate_change'] = error_change > 0.1
            drift_results['details']['error_rate_change'] = error_change
        
        # Overall drift decision
        drift_results['concept_drift'] = (
            any(drift_results['class_drifts'].values()) or 
            drift_results['error_rate_change']
        )
        
        return drift_results
    
    def _hotelling_t2_test(self, reference_stats, current_samples):
        """Compute Hotelling's T-squared statistic for multivariate drift"""
        current_mean = np.mean(current_samples, axis=0)
        diff = current_mean - reference_stats['mean']
        
        # Pooled covariance
        n1 = reference_stats['size']
        n2 = len(current_samples)
        pooled_cov = ((n1 - 1) * reference_stats['cov'] + 
                     (n2 - 1) * np.cov(current_samples.T)) / (n1 + n2 - 2)
        
        # Compute T-squared statistic
        t2_stat = n1 * n2 / (n1 + n2) * diff.dot(
            np.linalg.pinv(pooled_cov)).dot(diff)
        
        # Convert to F-statistic
        p = len(diff)
        f_stat = t2_stat * (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)
        p_value = 1 - stats.f.cdf(f_stat, p, n1 + n2 - p - 1)
        
        return t2_stat, p_value

# Example usage
np.random.seed(42)

# Generate synthetic data with concept drift
def generate_synthetic_data(n_samples, n_classes=3, drift=False):
    if not drift:
        centers = [[0, 0], [2, 2], [-2, 2]]
    else:
        centers = [[1, 1], [3, 1], [-1, 3]]  # Shifted centers
        
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples)
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        start_idx = i * samples_per_class
        end_idx = (i + 1) * samples_per_class
        
        X[start_idx:end_idx] = np.random.multivariate_normal(
            centers[i], 
            np.eye(2) * (1.0 + 0.5 * drift),
            samples_per_class
        )
        y[start_idx:end_idx] = i
        
    return X, y

# Create detector
detector = SupervisedDriftDetector(n_classes=3)

# Generate initial data and fit reference
X_ref, y_ref = generate_synthetic_data(1000, drift=False)
detector.fit_reference(X_ref, y_ref)

# Simulate streaming data with drift
window_size = 100
results_log = []

for i in range(20):  # Simulate 20 windows
    drift = i >= 10  # Introduce drift halfway
    X_new, y_new = generate_synthetic_data(window_size, drift=drift)
    # Simulate predictions (with some errors)
    y_pred = y_new.copy()
    error_mask = np.random.random(window_size) < (0.1 + 0.1 * drift)
    y_pred[error_mask] = (y_pred[error_mask] + 1) % 3
    
    result = detector.update(X_new, y_new, y_pred)
    results_log.append(result)
    
    print(f"\nWindow {i+1} Results:")
    print(f"Concept Drift Detected: {result['concept_drift']}")
    print(f"Error Rate Change: {result['error_rate_change']}")
    print("Class-wise Drift Status:", dict(result['class_drifts']))
```

Slide 13: Advanced Preprocessing for Drift Detection

This implementation focuses on robust preprocessing techniques for drift detection, including automated feature scaling, dimensionality reduction, and handling of categorical variables with dynamic encoding updates.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from scipy import stats

class DriftPreprocessor:
    def __init__(self, categorical_features=None, n_components=0.95):
        self.categorical_features = categorical_features or []
        self.n_components = n_components
        self.fitted = False
        
        # Initialize transformers
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.pca = PCA(n_components=n_components)
        
        # Storage for distribution parameters
        self.feature_distributions = {}
        self.encoded_categories = {}
        
    def fit(self, X, update_distributions=True):
        """Fit preprocessor on reference data"""
        numerical_mask = [i for i in range(X.shape[1]) 
                         if i not in self.categorical_features]
        
        # Split features
        X_num = X[:, numerical_mask] if len(numerical_mask) > 0 else None
        X_cat = X[:, self.categorical_features] if self.categorical_features else None
        
        # Fit transformers
        if X_num is not None:
            self.numerical_scaler.fit(X_num)
            X_num_scaled = self.numerical_scaler.transform(X_num)
            
            # Store distribution parameters for numerical features
            if update_distributions:
                for i, feat_idx in enumerate(numerical_mask):
                    self.feature_distributions[feat_idx] = {
                        'mean': np.mean(X_num_scaled[:, i]),
                        'std': np.std(X_num_scaled[:, i]),
                        'skew': stats.skew(X_num_scaled[:, i]),
                        'kurtosis': stats.kurtosis(X_num_scaled[:, i])
                    }
        
        if X_cat is not None:
            self.categorical_encoder.fit(X_cat)
            
            # Store category frequencies
            if update_distributions:
                for i, feat_idx in enumerate(self.categorical_features):
                    unique, counts = np.unique(X_cat[:, i], return_counts=True)
                    self.encoded_categories[feat_idx] = dict(zip(unique, 
                                                               counts / len(X_cat)))
        
        # Fit PCA on combined features
        X_transformed = self.transform(X, update_encodings=False)
        self.pca.fit(X_transformed)
        
        self.fitted = True
        return self
    
    def transform(self, X, update_encodings=True):
        """Transform new data and optionally update categorical encodings"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
            
        numerical_mask = [i for i in range(X.shape[1]) 
                         if i not in self.categorical_features]
        
        # Transform numerical features
        X_num = X[:, numerical_mask] if len(numerical_mask) > 0 else None
        X_cat = X[:, self.categorical_features] if self.categorical_features else None
        
        transformed_parts = []
        
        if X_num is not None:
            X_num_scaled = self.numerical_scaler.transform(X_num)
            transformed_parts.append(X_num_scaled)
        
        if X_cat is not None:
            # Handle new categories if needed
            if update_encodings:
                self._update_categorical_encodings(X_cat)
            X_cat_encoded = self.categorical_encoder.transform(X_cat)
            transformed_parts.append(X_cat_encoded)
        
        # Combine transformed features
        X_combined = np.hstack(transformed_parts)
        
        # Apply PCA
        X_reduced = self.pca.transform(X_combined)
        
        return X_reduced
    
    def _update_categorical_encodings(self, X_cat):
        """Update category frequencies with new data"""
        for i, feat_idx in enumerate(self.categorical_features):
            unique, counts = np.unique(X_cat[:, i], return_counts=True)
            current_freqs = dict(zip(unique, counts / len(X_cat)))
            
            # Update stored frequencies
            if feat_idx in self.encoded_categories:
                for category, freq in current_freqs.items():
                    if category in self.encoded_categories[feat_idx]:
                        # Exponential moving average update
                        alpha = 0.1
                        old_freq = self.encoded_categories[feat_idx][category]
                        self.encoded_categories[feat_idx][category] = (
                            alpha * freq + (1 - alpha) * old_freq
                        )
                    else:
                        # Add new category
                        self.encoded_categories[feat_idx][category] = freq
    
    def compute_distribution_changes(self, X):
        """Compute distribution changes for both numerical and categorical features"""
        changes = {}
        
        # Numerical features
        numerical_mask = [i for i in range(X.shape[1]) 
                         if i not in self.categorical_features]
        X_num = X[:, numerical_mask] if len(numerical_mask) > 0 else None
        
        if X_num is not None:
            X_num_scaled = self.numerical_scaler.transform(X_num)
            
            for i, feat_idx in enumerate(numerical_mask):
                current_stats = {
                    'mean': np.mean(X_num_scaled[:, i]),
                    'std': np.std(X_num_scaled[:, i]),
                    'skew': stats.skew(X_num_scaled[:, i]),
                    'kurtosis': stats.kurtosis(X_num_scaled[:, i])
                }
                
                # Compute relative changes
                ref_stats = self.feature_distributions[feat_idx]
                stat_changes = {}
                
                for stat, value in current_stats.items():
                    if ref_stats[stat] != 0:
                        rel_change = abs(value - ref_stats[stat]) / abs(ref_stats[stat])
                        stat_changes[stat] = rel_change
                    
                changes[f'numerical_{feat_idx}'] = stat_changes
        
        # Categorical features
        X_cat = X[:, self.categorical_features] if self.categorical_features else None
        
        if X_cat is not None:
            for i, feat_idx in enumerate(self.categorical_features):
                unique, counts = np.unique(X_cat[:, i], return_counts=True)
                current_freqs = dict(zip(unique, counts / len(X_cat)))
                
                # Compute Jensen-Shannon divergence
                ref_freqs = self.encoded_categories[feat_idx]
                js_div = self._jensen_shannon_divergence(ref_freqs, current_freqs)
                
                changes[f'categorical_{feat_idx}'] = {
                    'js_divergence': js_div
                }
        
        return changes
    
    def _jensen_shannon_divergence(self, P, Q):
        """Compute Jensen-Shannon divergence between two probability distributions"""
        # Get all categories
        categories = set(list(P.keys()) + list(Q.keys()))
        
        # Convert to arrays with zeros for missing categories
        p = np.array([P.get(cat, 0) for cat in categories])
        q = np.array([Q.get(cat, 0) for cat in categories])
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute mean distribution
        m = (p + q) / 2
        
        # Compute JS divergence
        return (stats.entropy(p, m) + stats.entropy(q, m)) / 2

# Example usage
np.random.seed(42)

# Generate synthetic data with mixed types
n_samples = 1000
n_numerical = 3
n_categorical = 2

# Generate reference data
X_num_ref = np.random.normal(0, 1, (n_samples, n_numerical))
X_cat_ref = np.random.choice(['A', 'B', 'C'], (n_samples, n_categorical))
X_ref = np.hstack([X_num_ref, X_cat_ref])

# Generate current data with drift
X_num_cur = np.random.normal(0.5, 1.2, (n_samples, n_numerical))
X_cat_cur = np.random.choice(['A', 'B', 'C', 'D'], (n_samples, n_categorical))
X_cur = np.hstack([X_num_cur, X_cat_cur])

# Initialize and fit preprocessor
categorical_features = list(range(n_numerical, n_numerical + n_categorical))
preprocessor = DriftPreprocessor(categorical_features=categorical_features)
preprocessor.fit(X_ref)

# Transform data and compute changes
X_ref_transformed = preprocessor.transform(X_ref)
X_cur_transformed = preprocessor.transform(X_cur)
distribution_changes = preprocessor.compute_distribution_changes(X_cur)

print("\nDistribution Changes:")
for feature, changes in distribution_changes.items():
    print(f"\n{feature}:")
    for metric, value in changes.items():
        print(f"  {metric}: {value:.4f}")
```

Slide 14: Model-Agnostic Drift Detection System

This implementation creates a model-agnostic system that can detect drift across different types of models by monitoring prediction distributions and feature importance patterns.

```python
import numpy as np
from scipy import stats
from sklearn.inspection import permutation_importance
from typing import Any, Dict, List, Optional
import warnings

class ModelAgnosticDriftDetector:
    def __init__(self, 
                 model: Any,
                 feature_names: List[str],
                 window_size: int = 1000,
                 significance_level: float = 0.05):
        self.model = model
        self.feature_names = feature_names
        self.window_size = window_size
        self.significance_level = significance_level
        
        # Storage for reference distributions
        self.reference_predictions = None
        self.reference_importances = None
        self.prediction_history = []
        self.importance_history = []
        
    def set_reference(self, X_ref: np.ndarray, y_ref: np.ndarray) -> None:
        """Establish reference distribution for predictions and feature importance"""
        # Get reference predictions
        self.reference_predictions = self._get_prediction_distribution(X_ref)
        
        # Calculate initial feature importance
        self.reference_importances = self._calculate_feature_importance(X_ref, y_ref)
        
    def detect_drift(self, X_current: np.ndarray, 
                    y_current: Optional[np.ndarray] = None) -> Dict:
        """Detect drift in current data compared to reference"""
        if self.reference_predictions is None:
            raise ValueError("Reference distribution not set. Call set_reference first.")
            
        current_predictions = self._get_prediction_distribution(X_current)
        
        # Store history
        self.prediction_history.append(current_predictions)
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            
        # Calculate current feature importance if labels available
        current_importances = None
        if y_current is not None:
            current_importances = self._calculate_feature_importance(
                X_current, 
                y_current
            )
            self.importance_history.append(current_importances)
            
            if len(self.importance_history) > self.window_size:
                self.importance_history.pop(0)
        
        return self._analyze_drift(current_predictions, current_importances)
    
    def _get_prediction_distribution(self, X: np.ndarray) -> np.ndarray:
        """Get distribution of model predictions"""
        try:
            # Try probability predictions first
            predictions = self.model.predict_proba(X)
        except (AttributeError, NotImplementedError):
            # Fall back to regular predictions
            predictions = self.model.predict(X)
        return predictions
    
    def _calculate_feature_importance(self, X: np.ndarray, 
                                    y: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using permutation importance"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = permutation_importance(
                self.model, X, y,
                n_repeats=5,
                random_state=42
            )
        
        return dict(zip(self.feature_names, result.importances_mean))
    
    def _analyze_drift(self, current_predictions: np.ndarray,
                      current_importances: Optional[Dict[str, float]]) -> Dict:
        """Analyze different types of drift"""
        results = {
            'prediction_drift': False,
            'feature_importance_drift': False,
            'details': {}
        }
        
        # Analyze prediction drift
        pred_ks_stat, pred_p_value = stats.ks_2samp(
            self.reference_predictions.flatten(),
            current_predictions.flatten()
        )
        
        results['details']['prediction_drift'] = {
            'statistic': pred_ks_stat,
            'p_value': pred_p_value
        }
        results['prediction_drift'] = pred_p_value < self.significance_level
        
        # Analyze feature importance drift if available
        if current_importances is not None:
            importance_changes = {}
            for feature in self.feature_names:
                ref_imp = self.reference_importances[feature]
                curr_imp = current_importances[feature]
                
                if ref_imp != 0:
                    relative_change = abs(curr_imp - ref_imp) / abs(ref_imp)
                else:
                    relative_change = abs(curr_imp - ref_imp)
                    
                importance_changes[feature] = relative_change
                
            # Consider drift if any feature importance changed significantly
            max_importance_change = max(importance_changes.values())
            results['feature_importance_drift'] = max_importance_change > 0.5
            results['details']['importance_changes'] = importance_changes
        
        # Additional metrics
        if len(self.prediction_history) >= self.window_size:
            results['details']['trend_analysis'] = self._analyze_trends()
        
        return results
    
    def _analyze_trends(self) -> Dict:
        """Analyze trends in prediction and importance histories"""
        trends = {}
        
        # Analyze prediction distribution trends
        pred_means = [np.mean(preds) for preds in self.prediction_history]
        pred_trend = np.polyfit(range(len(pred_means)), pred_means, 1)[0]
        trends['prediction_trend'] = pred_trend
        
        # Analyze importance trends if available
        if self.importance_history:
            importance_trends = {}
            for feature in self.feature_names:
                feature_imps = [imp[feature] for imp in self.importance_history]
                imp_trend = np.polyfit(range(len(feature_imps)), feature_imps, 1)[0]
                importance_trends[feature] = imp_trend
            trends['importance_trends'] = importance_trends
            
        return trends

# Example usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic data
X_ref, y_ref = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Create and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_ref, y_ref)

# Initialize detector
feature_names = [f'feature_{i}' for i in range(X_ref.shape[1])]
detector = ModelAgnosticDriftDetector(model, feature_names)
detector.set_reference(X_ref, y_ref)

# Generate drift data
X_drift, y_drift = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=43,
    flip_y=0.2  # Introduce label noise
)

# Detect drift
results = detector.detect_drift(X_drift, y_drift)

print("\nDrift Detection Results:")
print(f"Prediction Distribution Drift: {results['prediction_drift']}")
print(f"Feature Importance Drift: {results['feature_importance_drift']}")
print("\nDetailed Statistics:")
print(f"Prediction Drift p-value: {results['details']['prediction_drift']['p_value']:.4f}")

if 'importance_changes' in results['details']:
    print("\nFeature Importance Changes:")
    for feature, change in results['details']['importance_changes'].items():
        print(f"{feature}: {change:.4f}")
```

Slide 15: Additional Resources

*   "Dealing with Concept Drift: Importance of Data Distribution Monitoring" - [https://arxiv.org/abs/2004.12800](https://arxiv.org/abs/2004.12800)
*   "A Survey on Concept Drift Adaptation" - [https://arxiv.org/abs/1010.4784](https://arxiv.org/abs/1010.4784)
*   "Learning under Concept Drift: A Review" - [https://arxiv.org/abs/2004.05785](https://arxiv.org/abs/2004.05785)
*   "Online Learning and Concept Drift: An Overview" - \[Search "concept drift overview" on Google Scholar\]
*   "Adaptive Learning Systems: Beyond Supervised Learning" - \[Visit IEEE Xplore Digital Library\]

Key search areas for further research:

*   Concept drift detection algorithms
*   Online learning with drift adaptation
*   Distribution shift in machine learning
*   Adaptive model updating strategies
*   Real-time drift monitoring systems

