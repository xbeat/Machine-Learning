## Detecting Concept and Data Drift in Machine Learning
Slide 1: Understanding Data Drift Detection

Data drift detection is crucial for maintaining model performance in production. We'll implement a basic drift detector using the Kolmogorov-Smirnov test to identify statistical differences between training and production data distributions.

```python
import numpy as np
from scipy import stats
import pandas as pd

class DataDriftDetector:
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.baseline_data = None
        
    def set_baseline(self, data):
        """Store baseline (training) data distribution"""
        self.baseline_data = data
        
    def detect_drift(self, production_data):
        """Perform KS test between baseline and production data"""
        statistic, p_value = stats.ks_2samp(self.baseline_data, production_data)
        return {
            'drift_detected': p_value < self.threshold,
            'p_value': p_value,
            'statistic': statistic
        }

# Example usage
np.random.seed(42)
baseline = np.random.normal(0, 1, 1000)  # Training data
production = np.random.normal(0.5, 1, 1000)  # Shifted production data

detector = DataDriftDetector()
detector.set_baseline(baseline)
result = detector.detect_drift(production)
print(f"Drift detection results: {result}")
```

Slide 2: Implementing Concept Drift Detection

Concept drift detection requires monitoring the relationship between features and target variables. Here we implement the CUSUM (Cumulative Sum) algorithm to detect changes in prediction error patterns.

```python
class ConceptDriftDetector:
    def __init__(self, threshold=1.0, drift_threshold=2.0):
        self.threshold = threshold
        self.drift_threshold = drift_threshold
        self.mean = 0
        self.sum = 0
        self.n = 0
        
    def update(self, error):
        """Update detector with new prediction error"""
        self.n += 1
        old_mean = self.mean
        self.mean += (error - old_mean) / self.n
        self.sum = max(0, self.sum + error - (self.mean + self.threshold))
        
        return self.sum > self.drift_threshold
    
# Example usage
import numpy as np

detector = ConceptDriftDetector()
errors = np.concatenate([
    np.random.normal(0, 1, 100),  # Normal errors
    np.random.normal(2, 1, 50)    # Drift period
])

drift_points = []
for i, error in enumerate(errors):
    if detector.update(error):
        drift_points.append(i)
        print(f"Concept drift detected at point {i}")
```

Slide 3: Multivariate Data Drift Analysis

A comprehensive approach to detecting multivariate data drift using Maximum Mean Discrepancy (MMD), which can capture complex distribution changes across multiple features simultaneously.

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def compute_mmd(X, Y, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy between two samples
    X, Y: numpy arrays of shape (n_samples, n_features)
    """
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    
    mmd = (K_XX.mean() + K_YY.mean() - 2 * K_XY.mean())
    return np.sqrt(max(mmd, 0))

# Example usage
np.random.seed(42)
X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
Y = np.random.multivariate_normal([0.5, 0.5], [[1, 0.5], [0.5, 1]], 100)

mmd_score = compute_mmd(X, Y)
print(f"MMD Score: {mmd_score:.4f}")
```

Slide 4: Real-time Concept Drift Detection System

```python
class RealTimeConceptDriftDetector:
    def __init__(self, window_size=100, alpha=3.0):
        self.window_size = window_size
        self.alpha = alpha
        self.errors = []
        self.baseline_mean = None
        self.baseline_std = None
        
    def initialize(self, initial_errors):
        """Initialize baseline statistics"""
        self.baseline_mean = np.mean(initial_errors)
        self.baseline_std = np.std(initial_errors)
        self.errors = list(initial_errors)
        
    def detect_drift(self, new_error):
        """Update window and check for drift"""
        self.errors.append(new_error)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
            
        current_mean = np.mean(self.errors)
        z_score = abs(current_mean - self.baseline_mean) / self.baseline_std
        
        return {
            'drift_detected': z_score > self.alpha,
            'z_score': z_score,
            'current_mean': current_mean
        }

# Example usage
np.random.seed(42)
initial_errors = np.random.normal(0, 1, 100)
detector = RealTimeConceptDriftDetector()
detector.initialize(initial_errors)

# Simulate concept drift
drift_errors = np.random.normal(2, 1, 50)
for error in drift_errors:
    result = detector.detect_drift(error)
    if result['drift_detected']:
        print(f"Drift detected! Z-score: {result['z_score']:.2f}")
```

Slide 5: Feature-wise Drift Analysis

Implementing a comprehensive feature-wise drift analyzer that tracks individual feature distributions over time using statistical tests and visualization capabilities.

```python
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

class FeatureDriftAnalyzer:
    def __init__(self, feature_names: List[str], significance_level: float = 0.05):
        self.feature_names = feature_names
        self.significance_level = significance_level
        self.reference_distributions = {}
        
    def set_reference(self, data: pd.DataFrame):
        """Store reference distributions for each feature"""
        for feature in self.feature_names:
            self.reference_distributions[feature] = {
                'data': data[feature].values,
                'mean': data[feature].mean(),
                'std': data[feature].std()
            }
    
    def analyze_drift(self, new_data: pd.DataFrame) -> Dict:
        results = {}
        for feature in self.feature_names:
            ref_data = self.reference_distributions[feature]['data']
            new_values = new_data[feature].values
            
            # Perform statistical tests
            ks_stat, p_value = stats.ks_2samp(ref_data, new_values)
            mean_shift = abs(new_values.mean() - 
                           self.reference_distributions[feature]['mean'])
            
            results[feature] = {
                'drift_detected': p_value < self.significance_level,
                'p_value': p_value,
                'ks_statistic': ks_stat,
                'mean_shift': mean_shift
            }
            
        return results

# Example usage
np.random.seed(42)
features = ['income', 'age', 'credit_score']
n_samples = 1000

# Generate reference data
reference_data = pd.DataFrame({
    'income': np.random.normal(50000, 10000, n_samples),
    'age': np.random.normal(35, 8, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples)
})

# Generate drift data with shift in income
drift_data = pd.DataFrame({
    'income': np.random.normal(55000, 10000, n_samples),  # Shifted
    'age': np.random.normal(35, 8, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples)
})

analyzer = FeatureDriftAnalyzer(features)
analyzer.set_reference(reference_data)
drift_results = analyzer.analyze_drift(drift_data)

for feature, result in drift_results.items():
    print(f"\n{feature} drift analysis:")
    print(f"Drift detected: {result['drift_detected']}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Mean shift: {result['mean_shift']:.2f}")
```

Slide 6: Adaptive Learning with Concept Drift

Implementation of an adaptive learning system that automatically updates its model when concept drift is detected, using a sliding window approach and performance monitoring.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from collections import deque

class AdaptiveModelManager:
    def __init__(self, 
                 base_model=RandomForestClassifier(),
                 window_size=1000,
                 drift_threshold=0.1):
        self.base_model = base_model
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.X_window = deque(maxlen=window_size)
        self.y_window = deque(maxlen=window_size)
        self.performance_history = []
        
    def update_window(self, X_batch, y_batch):
        """Add new samples to sliding window"""
        for x, y in zip(X_batch, y_batch):
            self.X_window.append(x)
            self.y_window.append(y)
            
    def check_drift(self, recent_performance):
        """Check if model performance indicates concept drift"""
        if len(self.performance_history) < 2:
            return False
            
        baseline = np.mean(self.performance_history[:-1])
        performance_drop = baseline - recent_performance
        return performance_drop > self.drift_threshold
        
    def adapt(self, X_batch, y_batch):
        """Update model if drift is detected"""
        self.update_window(X_batch, y_batch)
        
        # Calculate performance on new batch
        y_pred = self.base_model.predict(X_batch)
        current_performance = accuracy_score(y_batch, y_pred)
        self.performance_history.append(current_performance)
        
        if self.check_drift(current_performance):
            print("Drift detected - Retraining model...")
            # Retrain model on current window
            X_window_array = np.array(list(self.X_window))
            y_window_array = np.array(list(self.y_window))
            self.base_model.fit(X_window_array, y_window_array)
            
        return current_performance

# Example usage
np.random.seed(42)

# Generate initial training data
X_train = np.random.rand(1000, 5)
y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)

# Initialize adaptive model
adaptive_model = AdaptiveModelManager()
adaptive_model.base_model.fit(X_train, y_train)

# Simulate concept drift
for i in range(5):
    # Generate batch with gradually changing concept
    X_batch = np.random.rand(200, 5)
    noise = np.random.normal(0, 0.1 * i, 200)  # Increasing noise
    y_batch = (X_batch[:, 0] + X_batch[:, 1] + noise > 1).astype(int)
    
    performance = adaptive_model.adapt(X_batch, y_batch)
    print(f"Batch {i+1} Performance: {performance:.4f}")
```

Slide 7: Sequential Drift Detection

A sophisticated sequential drift detection mechanism that uses both CUSUM and EWMA (Exponentially Weighted Moving Average) for robust drift identification in streaming data.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DriftMetrics:
    cusum_pos: float
    cusum_neg: float
    ewma: float
    drift_detected: bool

class SequentialDriftDetector:
    def __init__(self, 
                 lambda_param: float = 0.1,
                 cusum_threshold: float = 5.0,
                 ewma_threshold: float = 3.0):
        self.lambda_param = lambda_param
        self.cusum_threshold = cusum_threshold
        self.ewma_threshold = ewma_threshold
        
        # Initialize tracking variables
        self.mean = 0
        self.std = 0
        self.n_samples = 0
        self.cusum_pos = 0
        self.cusum_neg = 0
        self.ewma = 0
        self.initialization_phase = True
        
    def _update_statistics(self, value: float) -> None:
        """Update running statistics"""
        self.n_samples += 1
        delta = value - self.mean
        self.mean += delta / self.n_samples
        self.std = np.sqrt(
            (self.std ** 2 * (self.n_samples - 2) +
             delta * (value - self.mean)) / (self.n_samples - 1)
        ) if self.n_samples > 1 else 0
        
    def _update_drift_metrics(self, value: float) -> DriftMetrics:
        """Update drift detection metrics"""
        if self.initialization_phase:
            self._update_statistics(value)
            if self.n_samples >= 30:  # Minimum samples for initialization
                self.initialization_phase = False
            return DriftMetrics(0, 0, 0, False)
            
        # Standardize value
        std_value = (value - self.mean) / (self.std + 1e-8)
        
        # Update CUSUM
        self.cusum_pos = max(0, self.cusum_pos + std_value - 0.005)
        self.cusum_neg = max(0, self.cusum_neg - std_value - 0.005)
        
        # Update EWMA
        self.ewma = (
            self.lambda_param * std_value +
            (1 - self.lambda_param) * self.ewma
        )
        
        # Check for drift
        drift_detected = (
            abs(self.ewma) > self.ewma_threshold or
            self.cusum_pos > self.cusum_threshold or
            self.cusum_neg > self.cusum_threshold
        )
        
        return DriftMetrics(
            self.cusum_pos,
            self.cusum_neg,
            self.ewma,
            drift_detected
        )
    
    def process_value(self, value: float) -> DriftMetrics:
        """Process new value and return drift metrics"""
        return self._update_drift_metrics(value)

# Example usage
np.random.seed(42)

# Generate data with drift
n_samples = 1000
normal_data = np.random.normal(0, 1, n_samples)
drift_data = np.random.normal(2, 1.5, n_samples)
all_data = np.concatenate([normal_data, drift_data])

# Process data
detector = SequentialDriftDetector()
drift_points = []

for i, value in enumerate(all_data):
    metrics = detector.process_value(value)
    if metrics.drift_detected:
        drift_points.append(i)
        print(f"Drift detected at point {i}")
        print(f"CUSUM+: {metrics.cusum_pos:.2f}")
        print(f"CUSUM-: {metrics.cusum_neg:.2f}")
        print(f"EWMA: {metrics.ewma:.2f}")
```

Slide 8: Time-Series Drift Detection

Implementation of a specialized drift detector for time series data that combines change point detection with seasonal decomposition to identify both gradual and sudden distribution changes.

```python
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Tuple, List

class TimeSeriesDriftDetector:
    def __init__(self, 
                 window_size: int = 30,
                 seasonal_period: int = 7,
                 change_threshold: float = 0.01):
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.change_threshold = change_threshold
        self.history = []
        self.baseline_stats = None
        
    def decompose_series(self, data: np.array) -> Dict[str, np.array]:
        """Perform seasonal decomposition"""
        if len(data) < 2 * self.seasonal_period:
            return None
            
        result = seasonal_decompose(
            data, 
            period=self.seasonal_period,
            extrapolate_trend=True
        )
        
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }
    
    def compute_distribution_metrics(self, 
                                  data: np.array) -> Dict[str, float]:
        """Calculate key distribution metrics"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skew': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
    def detect_drift(self, new_data: np.array) -> Dict:
        """Detect drift in new time series data"""
        self.history.extend(new_data)
        
        if len(self.history) < 2 * self.window_size:
            return {'drift_detected': False, 'message': 'Insufficient data'}
            
        # Get recent window
        recent_window = self.history[-self.window_size:]
        
        # Perform decomposition
        decomp = self.decompose_series(
            np.array(self.history[-2 * self.seasonal_period:])
        )
        
        if decomp is None:
            return {'drift_detected': False, 'message': 'Unable to decompose'}
            
        # Initialize baseline if needed
        if self.baseline_stats is None:
            self.baseline_stats = self.compute_distribution_metrics(
                self.history[:self.window_size]
            )
            return {'drift_detected': False, 'message': 'Baseline initialized'}
            
        # Compute current metrics
        current_stats = self.compute_distribution_metrics(recent_window)
        
        # Check for significant changes
        drift_detected = False
        drift_metrics = {}
        
        for metric in ['mean', 'std', 'skew', 'kurtosis']:
            relative_change = abs(
                current_stats[metric] - self.baseline_stats[metric]
            ) / (abs(self.baseline_stats[metric]) + 1e-10)
            
            drift_metrics[f'{metric}_change'] = relative_change
            if relative_change > self.change_threshold:
                drift_detected = True
                
        return {
            'drift_detected': drift_detected,
            'metrics': drift_metrics,
            'decomposition': {
                'trend_last': decomp['trend'][-1],
                'seasonal_strength': np.std(decomp['seasonal']),
                'residual_std': np.std(decomp['residual'])
            }
        }

# Example usage
np.random.seed(42)

# Generate synthetic time series with drift
def generate_time_series(n_points: int,
                        drift_point: int) -> np.array:
    t = np.arange(n_points)
    seasonal = 2 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    trend = np.zeros(n_points)
    trend[drift_point:] = 0.05 * (t[drift_point:] - drift_point)  # Drift
    noise = np.random.normal(0, 0.5, n_points)
    return seasonal + trend + noise

# Generate data
n_points = 200
drift_point = 100
data = generate_time_series(n_points, drift_point)

# Detect drift
detector = TimeSeriesDriftDetector()
batch_size = 20

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    result = detector.detect_drift(batch)
    
    if result['drift_detected']:
        print(f"\nDrift detected at time {i}")
        print("Drift metrics:")
        for metric, value in result['metrics'].items():
            print(f"{metric}: {value:.4f}")
```

Slide 9: Incremental Learning with Drift Adaptation

An advanced implementation of an incremental learning system that adapts to both data and concept drift while maintaining model performance through selective retraining and ensemble methods.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import List, Tuple, Optional

class AdaptiveIncrementalLearner(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators: int = 3,
                 max_models: int = 10,
                 drift_threshold: float = 0.1):
        self.n_estimators = n_estimators
        self.max_models = max_models
        self.drift_threshold = drift_threshold
        self.models: List[Dict] = []
        self.performance_history = []
        
    def _create_model(self) -> Dict:
        """Create a new base model"""
        return {
            'model': RandomForestClassifier(n_estimators=self.n_estimators),
            'weight': 1.0,
            'performance': []
        }
        
    def _update_weights(self, X: np.array, y: np.array) -> None:
        """Update model weights based on recent performance"""
        for model_dict in self.models:
            y_pred = model_dict['model'].predict(X)
            accuracy = np.mean(y_pred == y)
            model_dict['performance'].append(accuracy)
            
            # Update weight using exponential decay
            recent_perf = np.mean(model_dict['performance'][-5:])
            model_dict['weight'] = np.exp(recent_perf - 1)
            
    def _check_drift(self, 
                    current_performance: float) -> bool:
        """Check if drift has occurred"""
        if len(self.performance_history) < 5:
            return False
            
        baseline = np.mean(self.performance_history[-5:])
        return abs(current_performance - baseline) > self.drift_threshold
        
    def fit(self, X: np.array, y: np.array) -> 'AdaptiveIncrementalLearner':
        """Initial fit of the model"""
        initial_model = self._create_model()
        initial_model['model'].fit(X, y)
        self.models.append(initial_model)
        return self
        
    def partial_fit(self, 
                   X: np.array,
                   y: np.array,
                   classes: Optional[np.array] = None) -> None:
        """Update the model with new data"""
        # Get current performance
        y_pred = self.predict(X)
        current_performance = np.mean(y_pred == y)
        self.performance_history.append(current_performance)
        
        # Check for drift
        if self._check_drift(current_performance):
            # Create new model if needed
            if len(self.models) < self.max_models:
                new_model = self._create_model()
                new_model['model'].fit(X, y)
                self.models.append(new_model)
            else:
                # Replace worst performing model
                worst_idx = np.argmin([
                    np.mean(m['performance'][-5:]) 
                    for m in self.models
                ])
                self.models[worst_idx] = self._create_model()
                self.models[worst_idx]['model'].fit(X, y)
                
        # Update all models
        self._update_weights(X, y)
        
    def predict(self, X: np.array) -> np.array:
        """Weighted prediction from all models"""
        predictions = np.array([
            model_dict['model'].predict(X) 
            for model_dict in self.models
        ])
        weights = np.array([
            model_dict['weight'] 
            for model_dict in self.models
        ])
        
        # Weighted voting
        weighted_votes = np.zeros((X.shape[0], len(np.unique(predictions))))
        for i, weight in enumerate(weights):
            for j in range(X.shape[0]):
                weighted_votes[j, predictions[i, j]] += weight
                
        return np.argmax(weighted_votes, axis=1)

# Example usage
np.random.seed(42)

# Generate synthetic data with concept drift
def generate_drift_data(n_samples: int,
                       n_features: int,
                       drift_point: int) -> Tuple[np.array, np.array]:
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    
    # Initial concept
    y[:drift_point] = (X[:drift_point, 0] + X[:drift_point, 1] > 0).astype(int)
    
    # Drifted concept
    y[drift_point:] = (X[drift_point:, 0] - X[drift_point:, 1] > 0).astype(int)
    
    return X, y

# Generate data
X, y = generate_drift_data(1000, 5, 500)

# Train and evaluate
learner = AdaptiveIncrementalLearner()
learner.fit(X[:100], y[:100])

# Process remaining data in batches
batch_size = 50
accuracies = []

for i in range(100, len(X), batch_size):
    X_batch = X[i:i+batch_size]
    y_batch = y[i:i+batch_size]
    
    # Predict before update
    y_pred = learner.predict(X_batch)
    accuracy = np.mean(y_pred == y_batch)
    accuracies.append(accuracy)
    
    # Update model
    learner.partial_fit(X_batch, y_batch)
    
    if i % 200 == 0:
        print(f"Batch {i//batch_size}, Accuracy: {accuracy:.4f}")
```

Slide 10: Ensemble-based Drift Detection

Implementation of an ensemble approach that combines multiple drift detection methods to provide more robust and reliable drift detection in complex environments.

```python
import numpy as np
from scipy import stats
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class DriftResult:
    detected: bool
    confidence: float
    detector_votes: Dict[str, bool]
    statistics: Dict[str, float]

class EnsembleDriftDetector:
    def __init__(self,
                 window_size: int = 100,
                 confidence_threshold: float = 0.6,
                 detectors: Optional[List[str]] = None):
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.detectors = detectors or ['ks', 'mann_whitney', 'levene', 'mood']
        self.reference_window = None
        self.statistics_history = []
        
    def _ks_test(self, 
                 reference: np.array, 
                 current: np.array) -> Tuple[bool, float]:
        """Kolmogorov-Smirnov test"""
        statistic, p_value = stats.ks_2samp(reference, current)
        return p_value < 0.05, statistic

    def _mann_whitney_test(self, 
                          reference: np.array, 
                          current: np.array) -> Tuple[bool, float]:
        """Mann-Whitney U test"""
        statistic, p_value = stats.mannwhitneyu(
            reference, current, alternative='two-sided'
        )
        return p_value < 0.05, statistic

    def _levene_test(self, 
                     reference: np.array, 
                     current: np.array) -> Tuple[bool, float]:
        """Levene test for variance equality"""
        statistic, p_value = stats.levene(reference, current)
        return p_value < 0.05, statistic

    def _mood_test(self, 
                   reference: np.array, 
                   current: np.array) -> Tuple[bool, float]:
        """Mood test for scale differences"""
        statistic, p_value = stats.mood(reference, current)
        return p_value < 0.05, statistic

    def update_reference(self, data: np.array) -> None:
        """Update reference window"""
        self.reference_window = data[-self.window_size:]

    def detect_drift(self, current_data: np.array) -> DriftResult:
        """Detect drift using ensemble of methods"""
        if self.reference_window is None:
            self.update_reference(current_data)
            return DriftResult(
                detected=False,
                confidence=0.0,
                detector_votes={},
                statistics={}
            )

        # Initialize results
        detector_votes = {}
        statistics = {}

        # Run all detectors
        test_methods = {
            'ks': self._ks_test,
            'mann_whitney': self._mann_whitney_test,
            'levene': self._levene_test,
            'mood': self._mood_test
        }

        for detector in self.detectors:
            if detector in test_methods:
                detected, statistic = test_methods[detector](
                    self.reference_window, 
                    current_data
                )
                detector_votes[detector] = detected
                statistics[detector] = statistic

        # Calculate ensemble confidence
        positive_votes = sum(detector_votes.values())
        confidence = positive_votes / len(self.detectors)

        # Overall drift decision
        drift_detected = confidence >= self.confidence_threshold

        return DriftResult(
            detected=drift_detected,
            confidence=confidence,
            detector_votes=detector_votes,
            statistics=statistics
        )

    def process_batch(self, 
                     batch_data: np.array,
                     update_reference: bool = True) -> DriftResult:
        """Process a new batch of data"""
        result = self.detect_drift(batch_data)
        
        if update_reference and not result.detected:
            self.update_reference(batch_data)
            
        return result

# Example usage
np.random.seed(42)

# Generate synthetic data with gradual drift
def generate_gradual_drift_data(n_samples: int,
                              drift_start: int,
                              drift_length: int) -> np.array:
    data = np.random.normal(0, 1, n_samples)
    
    # Add gradual drift
    if drift_start + drift_length <= n_samples:
        drift_increment = 2.0 / drift_length
        for i in range(drift_length):
            idx = drift_start + i
            data[idx] += i * drift_increment
            
    return data

# Generate data
n_samples = 1000
drift_start = 400
drift_length = 200
data = generate_gradual_drift_data(n_samples, drift_start, drift_length)

# Initialize detector
detector = EnsembleDriftDetector()

# Process data in batches
batch_size = 50
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    result = detector.process_batch(batch)
    
    if result.detected:
        print(f"\nDrift detected at batch {i//batch_size}")
        print(f"Confidence: {result.confidence:.3f}")
        print("Detector votes:")
        for detector, vote in result.detector_votes.items():
            print(f"{detector}: {vote}")
        print("Statistics:")
        for detector, stat in result.statistics.items():
            print(f"{detector}: {stat:.3f}")
```

Slide 11: Visualization and Monitoring Dashboard

Implementation of a comprehensive drift monitoring dashboard that provides real-time visualization and analysis of different types of drift.

```python
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any

class DriftMonitoringDashboard:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.metrics_history = []
        self.alerts = []
        self.feature_stats = {
            feature: {
                'drift_count': 0,
                'last_drift': None,
                'severity_history': []
            } for feature in feature_names
        }

    def _calculate_severity(self, 
                          metric_value: float, 
                          threshold: float) -> str:
        """Calculate drift severity level"""
        if metric_value > threshold * 2:
            return 'HIGH'
        elif metric_value > threshold:
            return 'MEDIUM'
        return 'LOW'

    def update_metrics(self, 
                      timestamp: datetime,
                      metrics: Dict[str, Dict[str, float]]) -> None:
        """Update monitoring metrics"""
        current_metrics = {
            'timestamp': timestamp.isoformat(),
            'features': {}
        }

        for feature in self.feature_names:
            if feature in metrics:
                feature_metrics = metrics[feature]
                severity = self._calculate_severity(
                    feature_metrics.get('drift_score', 0),
                    feature_metrics.get('threshold', 0.05)
                )

                if severity != 'LOW':
                    self.feature_stats[feature]['drift_count'] += 1
                    self.feature_stats[feature]['last_drift'] = timestamp
                    self.alerts.append({
                        'timestamp': timestamp.isoformat(),
                        'feature': feature,
                        'severity': severity,
                        'metrics': feature_metrics
                    })

                self.feature_stats[feature]['severity_history'].append(severity)
                current_metrics['features'][feature] = {
                    'metrics': feature_metrics,
                    'severity': severity
                }

        self.metrics_history.append(current_metrics)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        return {
            'summary': {
                'total_points': len(self.metrics_history),
                'total_alerts': len(self.alerts),
                'feature_stats': self.feature_stats
            },
            'recent_alerts': self.alerts[-10:],
            'metrics_history': self.metrics_history[-100:]
        }

    def export_dashboard_data(self, 
                            filepath: str) -> None:
        """Export dashboard data to JSON"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

# Example usage
np.random.seed(42)

# Generate synthetic monitoring data
features = ['feature_1', 'feature_2', 'feature_3']
dashboard = DriftMonitoringDashboard(features)

def generate_drift_metrics(time_idx: int) -> Dict[str, Dict[str, float]]:
    """Generate synthetic drift metrics"""
    metrics = {}
    for feature in features:
        base_drift = np.sin(time_idx / 10) * 0.05
        noise = np.random.normal(0, 0.01)
        
        # Add sudden drift for feature_2 at specific points
        if feature == 'feature_2' and 30 <= time_idx <= 35:
            base_drift += 0.1
            
        metrics[feature] = {
            'drift_score': abs(base_drift + noise),
            'threshold': 0.05,
            'p_value': max(0, 1 - abs(base_drift + noise)),
            'distribution_distance': abs(base_drift + noise) * 2
        }
    return metrics

# Simulate monitoring over time
for i in range(50):
    timestamp = datetime.now()
    metrics = generate_drift_metrics(i)
    dashboard.update_metrics(timestamp, metrics)

# Generate and export report
report = dashboard.generate_report()

# Print summary
print("\nMonitoring Summary:")
print(f"Total data points: {report['summary']['total_points']}")
print(f"Total alerts: {report['summary']['total_alerts']}")

print("\nFeature Statistics:")
for feature, stats in report['summary']['feature_stats'].items():
    print(f"\n{feature}:")
    print(f"Drift count: {stats['drift_count']}")
    if stats['last_drift']:
        print(f"Last drift: {stats['last_drift']}")
```

Slide 12: Drift Detection with Deep Learning

Implementation of a neural network-based drift detector that uses representation learning to identify complex distribution changes in high-dimensional data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List

class DriftEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

class DeepDriftDetector:
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 threshold: float = 0.1):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = DriftEncoder(input_dim, latent_dim).to(self.device)
        self.threshold = threshold
        self.optimizer = optim.Adam(self.model.parameters())
        self.reference_distribution = None
        self.reference_loss = None
        
    def _compute_mmd(self,
                    x: torch.Tensor,
                    y: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy"""
        def gaussian_kernel(x: torch.Tensor,
                          y: torch.Tensor,
                          sigma: float = 1.0) -> torch.Tensor:
            norm = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=-1)
            return torch.exp(-norm / (2 * sigma ** 2))
        
        xx = gaussian_kernel(x, x)
        yy = gaussian_kernel(y, y)
        xy = gaussian_kernel(x, y)
        
        return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
    
    def fit_reference(self,
                     data: np.ndarray,
                     epochs: int = 50,
                     batch_size: int = 32) -> None:
        """Fit the model on reference data"""
        torch_data = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(torch_data, torch_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        reconstruction_criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                self.optimizer.zero_grad()
                
                latent, reconstructed = self.model(batch_x)
                loss = reconstruction_criterion(reconstructed, batch_x)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
                
        # Store reference distribution
        with torch.no_grad():
            latent, _ = self.model(torch_data)
            self.reference_distribution = latent.cpu().numpy()
            self.reference_loss = total_loss/len(dataloader)
            
    def detect_drift(self,
                    data: np.ndarray,
                    batch_size: int = 32) -> dict:
        """Detect drift in new data"""
        if self.reference_distribution is None:
            raise ValueError("Must fit reference distribution first")
            
        self.model.eval()
        torch_data = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            latent, reconstructed = self.model(torch_data)
            current_distribution = latent.cpu().numpy()
            
            # Compute MMD between distributions
            mmd_score = self._compute_mmd(
                torch.FloatTensor(self.reference_distribution),
                torch.FloatTensor(current_distribution)
            ).item()
            
            # Compute reconstruction error
            rec_error = nn.MSELoss()(reconstructed, torch_data).item()
            
            # Detect drift based on both metrics
            drift_detected = (
                mmd_score > self.threshold or
                rec_error > self.reference_loss * 1.5
            )
            
            return {
                'drift_detected': drift_detected,
                'mmd_score': mmd_score,
                'reconstruction_error': rec_error,
                'reference_error': self.reference_loss
            }

# Example usage
np.random.seed(42)

# Generate synthetic data
def generate_high_dim_data(n_samples: int,
                          n_features: int,
                          drift: bool = False) -> np.ndarray:
    """Generate high-dimensional data with optional drift"""
    if not drift:
        return np.random.normal(0, 1, (n_samples, n_features))
    else:
        # Add correlation and shift in distribution
        base = np.random.normal(0, 1, (n_samples, n_features))
        correlation = np.random.normal(0, 0.5, (n_features, n_features))
        return np.dot(base, correlation) + 0.5

# Generate reference and drift data
n_features = 50
n_samples = 1000

reference_data = generate_high_dim_data(n_samples, n_features, drift=False)
drift_data = generate_high_dim_data(n_samples, n_features, drift=True)

# Initialize and train detector
detector = DeepDriftDetector(input_dim=n_features)
detector.fit_reference(reference_data)

# Test on both reference and drift data
print("\nTesting on reference data:")
ref_result = detector.detect_drift(reference_data[:100])
for key, value in ref_result.items():
    print(f"{key}: {value}")

print("\nTesting on drift data:")
drift_result = detector.detect_drift(drift_data[:100])
for key, value in drift_result.items():
    print(f"{key}: {value}")
```

Slide 13: Additional Resources

*   "A Survey on Concept Drift Adaptation"
    *   [https://arxiv.org/abs/1801.00977](https://arxiv.org/abs/1801.00977)
*   "Learning under Concept Drift: A Review"
    *   [https://arxiv.org/abs/2004.05785](https://arxiv.org/abs/2004.05785)
*   "Deep Learning for Concept Drift Detection in Streaming Data"
    *   [https://arxiv.org/abs/2012.12970](https://arxiv.org/abs/2012.12970)
*   Recommended Search Terms:
    *   "Adaptive learning systems concept drift"
    *   "Online machine learning drift detection"
    *   "Real-time distribution shift detection"
    *   "Neural drift detection methods"
*   Additional Tools and Libraries:
    *   River (Python library for online ML): [https://riverml.xyz](https://riverml.xyz)
    *   Alibi-detect (Drift detection): [https://github.com/SeldonIO/alibi-detect](https://github.com/SeldonIO/alibi-detect)
    *   Scikit-multiflow: [https://scikit-multiflow.github.io](https://scikit-multiflow.github.io)

Note: Some URLs may need verification. Please check current documentation for most up-to-date resources.

