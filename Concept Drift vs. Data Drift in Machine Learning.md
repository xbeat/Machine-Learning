## Concept Drift vs. Data Drift in Machine Learning
Slide 1: Understanding Concept Drift Detection

Concept drift detection requires monitoring changes in the relationship between features and target variables over time. This implementation demonstrates a basic statistical approach using a sliding window to detect significant changes in prediction error patterns.

```python
import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple

class ConceptDriftDetector(BaseEstimator):
    def __init__(self, window_size: int = 100, threshold: float = 2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.error_window = []
        
    def add_error(self, error: float) -> Tuple[bool, float]:
        self.error_window.append(error)
        if len(self.error_window) > self.window_size:
            self.error_window.pop(0)
            
        if len(self.error_window) == self.window_size:
            z_score = self._calculate_zscore()
            return abs(z_score) > self.threshold, z_score
        return False, 0.0
    
    def _calculate_zscore(self) -> float:
        mean_error = np.mean(self.error_window)
        std_error = np.std(self.error_window)
        recent_mean = np.mean(self.error_window[-10:])
        return (recent_mean - mean_error) / (std_error + 1e-8)
```

Slide 2: Implementing Adaptive Model Retraining

A practical approach to handling concept drift involves implementing an adaptive retraining strategy. This system monitors prediction errors and automatically triggers model updates when significant drift is detected.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class AdaptiveModelTrainer:
    def __init__(self, base_model, drift_detector, retrain_size=1000):
        self.model = base_model
        self.drift_detector = drift_detector
        self.retrain_size = retrain_size
        self.recent_data = []
        self.recent_labels = []
        
    def update(self, X, y, prediction):
        error = int(prediction != y)
        drift_detected, _ = self.drift_detector.add_error(error)
        
        self.recent_data.append(X)
        self.recent_labels.append(y)
        
        if drift_detected and len(self.recent_data) >= self.retrain_size:
            self._retrain()
            return True
        return False
        
    def _retrain(self):
        X_retrain = pd.DataFrame(self.recent_data[-self.retrain_size:])
        y_retrain = np.array(self.recent_labels[-self.retrain_size:])
        self.model.fit(X_retrain, y_retrain)
        self.recent_data = []
        self.recent_labels = []
```

Slide 3: Feature-Target Relationship Analysis

This implementation provides tools to analyze and visualize changes in feature-target relationships over time, helping identify specific features contributing to concept drift through correlation analysis.

```python
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

class FeatureTargetAnalyzer:
    def __init__(self, window_size=500):
        self.window_size = window_size
        self.correlation_history = {}
        
    def update_correlations(self, X, y):
        for column in X.columns:
            if column not in self.correlation_history:
                self.correlation_history[column] = []
            
            corr, _ = pearsonr(X[column].values, y)
            self.correlation_history[column].append(corr)
            
            if len(self.correlation_history[column]) > self.window_size:
                self.correlation_history[column].pop(0)
                
    def plot_correlation_trends(self):
        plt.figure(figsize=(12, 6))
        for feature, correlations in self.correlation_history.items():
            plt.plot(correlations, label=feature)
        plt.xlabel('Time Window')
        plt.ylabel('Correlation with Target')
        plt.legend()
        plt.title('Feature-Target Correlation Over Time')
        plt.show()
```

Slide 4: Real-time Drift Monitoring System

This comprehensive system implements real-time monitoring of both concept and data drift, using statistical tests and visualization tools to track model performance degradation and data distribution changes.

```python
from scipy.stats import ks_2samp
import numpy as np
from typing import Dict, List

class DriftMonitoringSystem:
    def __init__(self, reference_data: pd.DataFrame, confidence_level: float = 0.05):
        self.reference_distributions = self._compute_distributions(reference_data)
        self.confidence_level = confidence_level
        self.drift_scores = {col: [] for col in reference_data.columns}
        
    def _compute_distributions(self, data: pd.DataFrame) -> Dict:
        return {col: data[col].values for col in data.columns}
    
    def check_drift(self, new_data: pd.DataFrame) -> Dict[str, bool]:
        drift_detected = {}
        for column in self.reference_distributions.keys():
            statistic, p_value = ks_2samp(
                self.reference_distributions[column],
                new_data[column].values
            )
            drift_detected[column] = p_value < self.confidence_level
            self.drift_scores[column].append(statistic)
        return drift_detected
```

Slide 5: Statistical Process Control for Drift Detection

Statistical Process Control (SPC) provides a robust framework for detecting concept drift by monitoring the stability of model predictions. This implementation uses CUSUM (Cumulative Sum) charts to detect subtle changes in prediction patterns.

```python
import numpy as np
from typing import Tuple, List

class CUSUMDriftDetector:
    def __init__(self, threshold: float = 5.0, drift_value: float = 1.0):
        self.threshold = threshold
        self.drift_value = drift_value
        self.pos_cusum = 0
        self.neg_cusum = 0
        self.means: List[float] = []
        
    def update(self, value: float) -> Tuple[bool, float]:
        self.means.append(value)
        if len(self.means) < 50:  # Warmup period
            return False, 0
            
        target = np.mean(self.means[:-20])  # Reference mean
        deviation = value - target
        
        self.pos_cusum = max(0, self.pos_cusum + deviation - self.drift_value)
        self.neg_cusum = max(0, self.neg_cusum - deviation - self.drift_value)
        
        return (self.pos_cusum > self.threshold or 
                self.neg_cusum > self.threshold), max(self.pos_cusum, self.neg_cusum)
```

Slide 6: Incremental Learning with Concept Drift

This implementation showcases an incremental learning approach that adapts to concept drift by maintaining an ensemble of base learners and dynamically adjusting their weights based on recent performance.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List

class IncrementalDriftLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators: List = []
        self.weights = np.ones(n_estimators) / n_estimators
        self.performance_window = []
        
    def partial_fit(self, X, y):
        # Add new estimator if needed
        if len(self.estimators) < self.n_estimators:
            self.estimators.append(clone(self.base_estimator))
        
        # Train newest estimator
        newest_estimator = self.estimators[-1]
        newest_estimator.fit(X, y)
        
        # Update weights based on performance
        for i, est in enumerate(self.estimators):
            pred = est.predict(X)
            accuracy = np.mean(pred == y)
            self.weights[i] *= (1 + accuracy)
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.estimators])
        return np.average(predictions, axis=0, weights=self.weights)
```

Slide 7: Temporal Validation Framework

Implementing a temporal validation framework is crucial for evaluating models under concept drift. This implementation creates time-based cross-validation splits while maintaining temporal order.

```python
import pandas as pd
from typing import Generator, Tuple
import numpy as np

class TemporalValidator:
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap
        
    def split(self, X: pd.DataFrame, date_column: str) -> Generator[Tuple, None, None]:
        dates = X[date_column].sort_values().unique()
        split_size = len(dates) // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = dates[split_size * (i + 1)]
            test_start = dates[split_size * (i + 1) + self.gap]
            test_end = dates[split_size * (i + 2)]
            
            train_mask = X[date_column] <= train_end
            test_mask = (X[date_column] >= test_start) & (X[date_column] <= test_end)
            
            yield np.where(train_mask)[0], np.where(test_mask)[0]
```

Slide 8: Feature Importance Drift Monitor

This implementation tracks changes in feature importance over time to identify which features are becoming more or less relevant, helping understand the nature of concept drift.

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from typing import Dict, List

class FeatureImportanceDriftMonitor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.importance_history: Dict[str, List[float]] = {}
        self.rf_model = RandomForestClassifier(n_estimators=100)
        
    def update(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        self.rf_model.fit(X, y)
        current_importances = dict(zip(X.columns, 
                                     self.rf_model.feature_importances_))
        
        for feature, importance in current_importances.items():
            if feature not in self.importance_history:
                self.importance_history[feature] = []
            
            self.importance_history[feature].append(importance)
            if len(self.importance_history[feature]) > self.window_size:
                self.importance_history[feature].pop(0)
                
        return self._calculate_importance_drift()
        
    def _calculate_importance_drift(self) -> Dict[str, float]:
        drift_scores = {}
        for feature, history in self.importance_history.items():
            if len(history) >= 2:
                recent = np.mean(history[-10:])
                overall = np.mean(history)
                drift_scores[feature] = abs(recent - overall) / overall
        return drift_scores
```

Slide 9: Concept Drift Visualization System

This implementation creates an interactive visualization system for monitoring concept drift patterns, including distribution shifts, performance metrics, and feature importance changes over time.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict

class DriftVisualizerSystem:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.performance_history = []
        self.drift_scores_history = []
        self.feature_importances_history = []
        
    def update(self, performance: float, drift_scores: Dict[str, float], 
               feature_importances: Dict[str, float]):
        self.performance_history.append(performance)
        self.drift_scores_history.append(drift_scores)
        self.feature_importances_history.append(feature_importances)
        
    def create_dashboard(self):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Model Performance', 'Drift Scores', 'Feature Importance')
        )
        
        # Performance Timeline
        fig.add_trace(
            go.Scatter(y=self.performance_history, name="Performance"),
            row=1, col=1
        )
        
        # Drift Scores
        for feature in self.feature_names:
            drift_values = [d[feature] for d in self.drift_scores_history]
            fig.add_trace(
                go.Scatter(y=drift_values, name=f"Drift-{feature}"),
                row=2, col=1
            )
            
        # Feature Importance
        for feature in self.feature_names:
            importance_values = [d[feature] for d in self.feature_importances_history]
            fig.add_trace(
                go.Scatter(y=importance_values, name=f"Importance-{feature}"),
                row=3, col=1
            )
            
        fig.update_layout(height=900, showlegend=True)
        return fig
```

Slide 10: Real-World Application: Credit Scoring Model

Implementation of a credit scoring system that handles concept drift in customer behavior patterns, demonstrating practical application in financial services.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

class AdaptiveCreditScoringSystem:
    def __init__(self, drift_threshold: float = 0.1):
        self.model = LogisticRegression(warm_start=True)
        self.scaler = StandardScaler()
        self.drift_detector = ConceptDriftDetector()
        self.drift_threshold = drift_threshold
        self.feature_importance_monitor = FeatureImportanceDriftMonitor()
        
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        numeric_features = ['income', 'debt_ratio', 'credit_history_length']
        categorical_features = ['employment_type', 'housing_status']
        
        # Handle missing values
        data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())
        data[categorical_features] = data[categorical_features].fillna('unknown')
        
        # Create dummy variables for categorical features
        data_encoded = pd.get_dummies(data[categorical_features], drop_first=True)
        
        # Combine numeric and encoded categorical features
        X = pd.concat([data[numeric_features], data_encoded], axis=1)
        return self.scaler.fit_transform(X)
        
    def update_model(self, X: np.ndarray, y: np.ndarray):
        # Check for concept drift
        predictions = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, predictions)
        
        drift_detected = self.drift_detector.add_error(1 - auc)[0]
        
        if drift_detected:
            # Retrain model with recent data
            self.model.fit(X, y)
            
            # Monitor feature importance changes
            importance_drift = self.feature_importance_monitor.update(
                pd.DataFrame(X), y
            )
            
            return {
                'retrained': True,
                'auc_score': auc,
                'importance_drift': importance_drift
            }
        
        return {
            'retrained': False,
            'auc_score': auc,
            'importance_drift': None
        }
```

Slide 11: Source Code for Credit Scoring Model Results

```python
# Example usage and results for Credit Scoring System
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample data loading and preprocessing
data = pd.read_csv('credit_data.csv')
X = adaptive_credit_system.preprocess_features(data)
y = data['default'].values

# Initial training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
initial_results = adaptive_credit_system.update_model(X_train, y_train)

print("Initial Training Results:")
print(f"AUC Score: {initial_results['auc_score']:.3f}")
print(f"Model Retrained: {initial_results['retrained']}")

# Simulate concept drift with new data
new_data = pd.read_csv('credit_data_new.csv')
X_new = adaptive_credit_system.preprocess_features(new_data)
y_new = new_data['default'].values

drift_results = adaptive_credit_system.update_model(X_new, y_new)

print("\nDrift Detection Results:")
print(f"AUC Score: {drift_results['auc_score']:.3f}")
print(f"Model Retrained: {drift_results['retrained']}")
if drift_results['importance_drift']:
    print("\nFeature Importance Changes:")
    for feature, drift in drift_results['importance_drift'].items():
        print(f"{feature}: {drift:.3f}")
```

Slide 12: Windowed Probability Distribution Tracker

This implementation monitors changes in probability distributions over time using sliding windows and statistical tests, providing detailed insights into the nature and magnitude of concept drift.

```python
import numpy as np
from scipy.stats import wasserstein_distance
from collections import deque
from typing import Dict, Optional

class ProbabilityDistributionTracker:
    def __init__(self, window_size: int = 1000, n_bins: int = 50):
        self.window_size = window_size
        self.n_bins = n_bins
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.distribution_distances: Dict[str, float] = {}
        
    def update(self, new_value: float, window_type: str = 'current') -> Optional[float]:
        if window_type == 'reference':
            self.reference_window.append(new_value)
        else:
            self.current_window.append(new_value)
            
        if len(self.reference_window) >= self.window_size and \
           len(self.current_window) >= self.window_size:
            return self._calculate_distribution_distance()
        return None
        
    def _calculate_distribution_distance(self) -> float:
        ref_hist, ref_bins = np.histogram(self.reference_window, 
                                        bins=self.n_bins, 
                                        density=True)
        curr_hist, _ = np.histogram(self.current_window, 
                                  bins=ref_bins, 
                                  density=True)
        
        distance = wasserstein_distance(ref_hist, curr_hist)
        self.distribution_distances[len(self.distribution_distances)] = distance
        return distance
```

Slide 13: Real-World Application: Customer Churn Prediction

A comprehensive implementation of a churn prediction system that adapts to changing customer behavior patterns while maintaining interpretability and performance monitoring.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class AdaptiveChurnPredictor:
    def __init__(self, 
                 drift_threshold: float = 0.1,
                 retraining_window: int = 5000):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.drift_detector = CUSUMDriftDetector()
        self.distribution_tracker = ProbabilityDistributionTracker()
        self.drift_threshold = drift_threshold
        self.retraining_window = retraining_window
        self.feature_history: Dict[str, deque] = {}
        
    def preprocess_and_train(self, 
                            data: pd.DataFrame, 
                            target: str = 'churned') -> Tuple[np.ndarray, np.ndarray]:
        # Feature engineering
        data['tenure_months'] = pd.to_numeric(data['tenure_months'])
        data['total_charges'] = pd.to_numeric(data['total_charges'].replace(' ', '0'))
        
        # Calculate customer lifetime value
        data['customer_lifetime_value'] = data['tenure_months'] * data['total_charges']
        
        # Create interaction features
        data['usage_per_charge'] = data['monthly_usage'] / data['total_charges']
        
        # Encode categorical variables
        categorical_cols = ['contract_type', 'payment_method', 'service_plan']
        data_encoded = pd.get_dummies(data[categorical_cols], drop_first=True)
        
        # Combine features
        numeric_cols = ['tenure_months', 'total_charges', 'customer_lifetime_value', 
                       'usage_per_charge']
        X = pd.concat([data[numeric_cols], data_encoded], axis=1)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y = data[target].values
        
        # Initial model training
        self.model.fit(X_scaled, y)
        
        return X_scaled, y
    
    def predict_and_update(self, 
                          X: np.ndarray, 
                          y: np.ndarray = None) -> Dict[str, float]:
        predictions = self.model.predict_proba(X)[:, 1]
        
        if y is not None:
            # Update drift detection
            error = np.mean(np.abs(predictions - y))
            drift_detected, drift_score = self.drift_detector.update(error)
            
            if drift_detected:
                # Retrain model
                self.model.fit(X, y)
                
            return {
                'predictions': predictions,
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'error_rate': error
            }
        
        return {'predictions': predictions}
```

Slide 14: Additional Resources

1.  "Adaptive Concept Drift Detection via Online Learning" [https://arxiv.org/abs/2105.07742](https://arxiv.org/abs/2105.07742)
2.  "Deep Learning for Concept Drift Detection in Streaming Data" [https://arxiv.org/abs/2004.00066](https://arxiv.org/abs/2004.00066)
3.  "A Survey on Concept Drift Adaptation" [https://arxiv.org/abs/1010.4784](https://arxiv.org/abs/1010.4784)
4.  "Learning under Concept Drift: A Review" [https://arxiv.org/abs/2004.05785](https://arxiv.org/abs/2004.05785)
5.  "Concept Drift Detection Through Resampling" [https://arxiv.org/abs/1704.00023](https://arxiv.org/abs/1704.00023)

