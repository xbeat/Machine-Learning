## Understanding and Mitigating Data Drift in ML
Slide 1: Understanding Data Drift Fundamentals

Data drift occurs when statistical properties of input features change over time in production environments compared to training data. This fundamental concept is crucial for maintaining model performance and requires continuous monitoring of feature distributions and model predictions.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def detect_data_drift(reference_data, current_data, threshold=0.05):
    # Perform Kolmogorov-Smirnov test for distribution comparison
    statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    # Check if drift is detected based on p-value threshold
    drift_detected = p_value < threshold
    
    return {
        'drift_detected': drift_detected,
        'p_value': p_value,
        'statistic': statistic
    }

# Example usage
np.random.seed(42)
reference_data = np.random.normal(0, 1, 1000)  # Training data distribution
current_data = np.random.normal(0.5, 1.2, 1000)  # Shifted distribution

result = detect_data_drift(reference_data, current_data)
print(f"Drift detected: {result['drift_detected']}")
print(f"P-value: {result['p_value']:.6f}")
```

Slide 2: Types of Data Drift

Data drift manifests in several forms: concept drift (relationship changes between features and target), covariate shift (feature distribution changes), and prior probability shift (target distribution changes). Understanding these distinctions is essential for appropriate monitoring strategies.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def analyze_drift_types(reference_df, current_df, feature_cols):
    drift_analysis = {}
    scaler = StandardScaler()
    
    for feature in feature_cols:
        # Calculate distribution statistics
        ref_mean = reference_df[feature].mean()
        ref_std = reference_df[feature].std()
        curr_mean = current_df[feature].mean()
        curr_std = current_df[feature].std()
        
        # Calculate relative changes
        mean_change = abs((curr_mean - ref_mean) / ref_mean) * 100
        std_change = abs((curr_std - ref_std) / ref_std) * 100
        
        drift_analysis[feature] = {
            'mean_change_percent': mean_change,
            'std_change_percent': std_change,
            'significant_drift': mean_change > 10 or std_change > 15
        }
    
    return drift_analysis

# Example usage
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.exponential(2, 1000)
})

current_data = pd.DataFrame({
    'feature1': np.random.normal(0.3, 1.2, 1000),
    'feature2': np.random.exponential(2.5, 1000)
})

results = analyze_drift_types(reference_data, current_data, ['feature1', 'feature2'])
print(pd.DataFrame(results).T)
```

Slide 3: Statistical Methods for Drift Detection

Statistical tests provide quantitative measures to detect significant distribution changes. Common approaches include Kolmogorov-Smirnov test, Chi-squared test, and Jensen-Shannon divergence, each suitable for different types of data and drift patterns.

```python
from scipy.stats import chi2_contingency
from scipy.spatial.distance import jensenshannon

def comprehensive_drift_detection(reference_data, current_data, categorical_cols=None):
    if categorical_cols is None:
        categorical_cols = []
    
    results = {}
    
    # KS test for numerical features
    numerical_cols = [col for col in reference_data.columns if col not in categorical_cols]
    for col in numerical_cols:
        ks_stat, p_value = stats.ks_2samp(reference_data[col], current_data[col])
        results[f'{col}_ks_test'] = {'statistic': ks_stat, 'p_value': p_value}
    
    # Chi-squared test for categorical features
    for col in categorical_cols:
        ref_counts = pd.crosstab(index=reference_data[col], columns='count')
        curr_counts = pd.crosstab(index=current_data[col], columns='count')
        contingency = pd.concat([ref_counts, curr_counts], axis=1).fillna(0)
        chi2_stat, p_value, _, _ = chi2_contingency(contingency)
        results[f'{col}_chi2_test'] = {'statistic': chi2_stat, 'p_value': p_value}
    
    return results

# Example usage
np.random.seed(42)
reference_df = pd.DataFrame({
    'numeric_feat': np.random.normal(0, 1, 1000),
    'categorical_feat': np.random.choice(['A', 'B', 'C'], 1000)
})

current_df = pd.DataFrame({
    'numeric_feat': np.random.normal(0.5, 1.2, 1000),
    'categorical_feat': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
})

drift_results = comprehensive_drift_detection(
    reference_df, 
    current_df, 
    categorical_cols=['categorical_feat']
)
print(pd.DataFrame(drift_results))
```

Slide 4: Feature Distribution Monitoring

Continuous monitoring of feature distributions enables early detection of data drift. This implementation demonstrates how to track and visualize distribution changes using kernel density estimation and statistical metrics.

```python
from scipy.stats import gaussian_kde
import seaborn as sns

class DistributionMonitor:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_kdes = {}
        self._fit_reference_kdes()
    
    def _fit_reference_kdes(self):
        for column in self.reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_kdes[column] = gaussian_kde(self.reference_data[column])
    
    def monitor_distributions(self, current_data, threshold=0.1):
        monitoring_results = {}
        
        for column, ref_kde in self.reference_kdes.items():
            if column not in current_data.columns:
                continue
                
            current_kde = gaussian_kde(current_data[column])
            
            # Calculate JS divergence between distributions
            x_eval = np.linspace(
                min(self.reference_data[column].min(), current_data[column].min()),
                max(self.reference_data[column].max(), current_data[column].max()),
                1000
            )
            ref_pdf = ref_kde(x_eval)
            current_pdf = current_kde(x_eval)
            js_div = jensenshannon(ref_pdf, current_pdf)
            
            monitoring_results[column] = {
                'js_divergence': js_div,
                'drift_detected': js_div > threshold
            }
        
        return monitoring_results

# Example usage
monitor = DistributionMonitor(reference_df)
distribution_changes = monitor.monitor_distributions(current_df)
print("\nDistribution Monitoring Results:")
print(pd.DataFrame(distribution_changes).T)
```

Slide 5: Real-time Drift Detection System

This implementation showcases a production-ready drift detection system that processes incoming data streams and triggers alerts when significant distributional changes are detected, incorporating both statistical tests and practical thresholds.

```python
import time
from collections import deque
from datetime import datetime

class RealTimeDriftDetector:
    def __init__(self, reference_data, window_size=1000, alert_threshold=0.05):
        self.reference_data = reference_data
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.data_window = {col: deque(maxlen=window_size) 
                           for col in reference_data.columns}
        self.alerts = []
        
    def process_datapoint(self, datapoint):
        timestamp = datetime.now()
        drift_detected = False
        
        # Update sliding windows
        for col, value in datapoint.items():
            if col in self.data_window:
                self.data_window[col].append(value)
                
                # Only check for drift if window is full
                if len(self.data_window[col]) == self.window_size:
                    current_data = np.array(self.data_window[col])
                    
                    # Perform KS test
                    statistic, p_value = stats.ks_2samp(
                        self.reference_data[col],
                        current_data
                    )
                    
                    if p_value < self.alert_threshold:
                        drift_detected = True
                        self.alerts.append({
                            'timestamp': timestamp,
                            'feature': col,
                            'p_value': p_value,
                            'statistic': statistic
                        })
        
        return drift_detected, self.alerts[-1] if drift_detected else None

# Example usage
np.random.seed(42)
detector = RealTimeDriftDetector(reference_df)

# Simulate streaming data
for i in range(1200):
    # Generate synthetic datapoint with increasing drift
    drift_factor = i / 1000
    synthetic_datapoint = {
        'numeric_feat': np.random.normal(drift_factor, 1 + drift_factor/2),
        'categorical_feat': np.random.choice(['A', 'B', 'C'])
    }
    
    drift_detected, alert = detector.process_datapoint(synthetic_datapoint)
    if drift_detected:
        print(f"Drift detected at iteration {i}:")
        print(f"Feature: {alert['feature']}")
        print(f"P-value: {alert['p_value']:.6f}\n")
```

Slide 6: Concept Drift Detection

Concept drift occurs when the relationship between input features and target variable changes over time. This implementation focuses on monitoring prediction error patterns to detect concept drift in classification models.

```python
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ConceptDriftMonitor:
    def __init__(self, base_model, window_size=500, error_threshold=0.1):
        self.base_model = base_model
        self.window_size = window_size
        self.error_threshold = error_threshold
        self.prediction_errors = deque(maxlen=window_size)
        self.baseline_error = None
        
    def set_baseline_performance(self, X_baseline, y_baseline):
        baseline_pred = self.base_model.predict_proba(X_baseline)[:, 1]
        self.baseline_error = 1 - roc_auc_score(y_baseline, baseline_pred)
        
    def monitor_predictions(self, X_current, y_true):
        current_pred = self.base_model.predict_proba(X_current)[:, 1]
        current_error = 1 - roc_auc_score(y_true, current_pred)
        self.prediction_errors.append(current_error)
        
        if len(self.prediction_errors) == self.window_size:
            avg_error = np.mean(self.prediction_errors)
            error_increase = (avg_error - self.baseline_error) / self.baseline_error
            
            return {
                'concept_drift_detected': error_increase > self.error_threshold,
                'error_increase': error_increase,
                'current_error': current_error,
                'baseline_error': self.baseline_error
            }
        return None

# Example usage with synthetic data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
X_train, y_train = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

# Train base model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Initialize concept drift monitor
drift_monitor = ConceptDriftMonitor(model, window_size=200)
drift_monitor.set_baseline_performance(X_train, y_train)

# Simulate concept drift by monitoring batches
batch_size = 50
for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    
    # Add synthetic concept drift
    if i > len(X_test)//2:
        X_batch = X_batch * 1.2  # Simulate feature relationship change
        
    result = drift_monitor.monitor_predictions(X_batch, y_batch)
    if result and result['concept_drift_detected']:
        print(f"Concept drift detected in batch {i//batch_size}")
        print(f"Error increase: {result['error_increase']:.2%}")
```

Slide 7: Data Drift Visualization

Effective visualization of data drift helps in understanding the nature and extent of distribution changes. This implementation provides multiple visualization techniques for both numerical and categorical features.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

class DriftVisualizer:
    def __init__(self, reference_data, current_data):
        self.reference_data = reference_data
        self.current_data = current_data
        plt.style.use('seaborn')
        
    def plot_distribution_comparison(self, feature, figsize=(12, 6)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # KDE plot
        sns.kdeplot(data=self.reference_data[feature], ax=ax1, 
                   label='Reference', color='blue')
        sns.kdeplot(data=self.current_data[feature], ax=ax1, 
                   label='Current', color='red')
        ax1.set_title(f'KDE Plot - {feature}')
        ax1.legend()
        
        # QQ plot
        ref_quantiles = np.quantile(self.reference_data[feature], 
                                  np.linspace(0, 1, 100))
        curr_quantiles = np.quantile(self.current_data[feature], 
                                   np.linspace(0, 1, 100))
        ax2.scatter(ref_quantiles, curr_quantiles, alpha=0.5)
        ax2.plot([min(ref_quantiles), max(ref_quantiles)], 
                [min(ref_quantiles), max(ref_quantiles)], 
                'r--', label='No Drift Line')
        ax2.set_title(f'Q-Q Plot - {feature}')
        ax2.set_xlabel('Reference Quantiles')
        ax2.set_ylabel('Current Quantiles')
        
        plt.tight_layout()
        return fig

    def plot_categorical_drift(self, feature):
        ref_props = self.reference_data[feature].value_counts(normalize=True)
        curr_props = self.current_data[feature].value_counts(normalize=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(ref_props))
        
        ax.bar(x - width/2, ref_props, width, label='Reference', color='blue', alpha=0.6)
        ax.bar(x + width/2, curr_props, width, label='Current', color='red', alpha=0.6)
        
        ax.set_xticks(x)
        ax.set_xticklabels(ref_props.index)
        ax.set_title(f'Categorical Distribution Changes - {feature}')
        ax.legend()
        
        return fig

# Example usage
visualizer = DriftVisualizer(reference_df, current_df)

# Visualize numerical feature drift
num_drift_plot = visualizer.plot_distribution_comparison('numeric_feat')
plt.close()

# Visualize categorical feature drift
cat_drift_plot = visualizer.plot_categorical_drift('categorical_feat')
plt.close()
```

Slide 8: Advanced Drift Detection Metrics

Implementation of sophisticated drift detection metrics including Population Stability Index (PSI) and Characteristic Stability Index (CSI) for comprehensive drift analysis across multiple features and time periods.

```python
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

class AdvancedDriftMetrics:
    def __init__(self, n_bins=10, threshold_psi=0.2, threshold_csi=0.15):
        self.n_bins = n_bins
        self.threshold_psi = threshold_psi
        self.threshold_csi = threshold_csi
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        
    def calculate_psi(self, reference_data, current_data):
        """Population Stability Index calculation"""
        # Bin the reference data
        self.discretizer.fit(reference_data.reshape(-1, 1))
        ref_bins = self.discretizer.transform(reference_data.reshape(-1, 1))
        curr_bins = self.discretizer.transform(current_data.reshape(-1, 1))
        
        ref_percents = np.histogram(ref_bins, bins=self.n_bins)[0] / len(ref_bins)
        curr_percents = np.histogram(curr_bins, bins=self.n_bins)[0] / len(curr_bins)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_percents = np.array([x + epsilon if x == 0 else x for x in ref_percents])
        curr_percents = np.array([x + epsilon if x == 0 else x for x in curr_percents])
        
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        return psi
    
    def calculate_csi(self, reference_data, current_data):
        """Characteristic Stability Index calculation"""
        ref_mean = np.mean(reference_data)
        ref_std = np.std(reference_data)
        curr_mean = np.mean(current_data)
        curr_std = np.std(current_data)
        
        # Calculate CSI components
        mean_shift = abs(ref_mean - curr_mean) / ref_std
        std_ratio = max(ref_std, curr_std) / min(ref_std, curr_std)
        
        csi = mean_shift * np.log(std_ratio)
        return csi
    
    def analyze_feature_drift(self, reference_df, current_df):
        results = {}
        
        for column in reference_df.columns:
            if np.issubdtype(reference_df[column].dtype, np.number):
                psi = self.calculate_psi(reference_df[column].values, 
                                      current_df[column].values)
                csi = self.calculate_csi(reference_df[column].values, 
                                      current_df[column].values)
                
                results[column] = {
                    'psi': psi,
                    'csi': csi,
                    'psi_drift_detected': psi > self.threshold_psi,
                    'csi_drift_detected': csi > self.threshold_csi
                }
        
        return pd.DataFrame(results).T

# Example usage
np.random.seed(42)

# Generate synthetic data with drift
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.exponential(2, 1000),
    'feature3': np.random.gamma(2, 2, 1000)
})

current_data = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.2, 1000),  # Significant drift
    'feature2': np.random.exponential(2.1, 1000),  # Minor drift
    'feature3': np.random.gamma(2, 2.5, 1000)      # Moderate drift
})

drift_analyzer = AdvancedDriftMetrics()
results = drift_analyzer.analyze_feature_drift(reference_data, current_data)
print("\nDrift Analysis Results:")
print(results)
```

Slide 9: Model Performance Impact Analysis

This implementation focuses on quantifying the impact of data drift on model performance through various metrics and provides insights into which features contribute most to performance degradation.

```python
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

class DriftImpactAnalyzer:
    def __init__(self, model, performance_threshold=0.1):
        self.model = model
        self.performance_threshold = performance_threshold
        self.baseline_performance = None
        self.feature_importance = None
        
    def set_baseline(self, X_baseline, y_baseline):
        """Establish baseline performance metrics"""
        self.baseline_predictions = self.model.predict(X_baseline)
        self.baseline_performance = {
            'mse': mean_squared_error(y_baseline, self.baseline_predictions),
            'r2': r2_score(y_baseline, self.baseline_predictions)
        }
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X_baseline.columns, 
                                             self.model.feature_importances_))
    
    def analyze_impact(self, X_current, y_current):
        """Analyze performance degradation and feature contribution"""
        current_predictions = self.model.predict(X_current)
        current_performance = {
            'mse': mean_squared_error(y_current, current_predictions),
            'r2': r2_score(y_current, current_predictions)
        }
        
        # Calculate performance degradation
        degradation = {
            'mse_change': (current_performance['mse'] - self.baseline_performance['mse']) 
                         / self.baseline_performance['mse'],
            'r2_change': (self.baseline_performance['r2'] - current_performance['r2'])
        }
        
        # Analyze feature-wise impact
        feature_impact = {}
        for feature in X_current.columns:
            # Calculate performance impact by permuting each feature
            X_permuted = X_current.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_predictions = self.model.predict(X_permuted)
            permuted_mse = mean_squared_error(y_current, permuted_predictions)
            
            # Calculate relative impact
            impact = (permuted_mse - current_performance['mse']) / current_performance['mse']
            feature_impact[feature] = {
                'importance': self.feature_importance.get(feature, 0),
                'drift_impact': impact
            }
        
        return {
            'performance_degradation': degradation,
            'feature_impact': pd.DataFrame(feature_impact).T,
            'significant_drift': degradation['r2_change'] > self.performance_threshold
        }

# Example usage
from sklearn.datasets import make_regression

# Generate synthetic dataset
X, y = make_regression(n_samples=2000, n_features=10, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

# Split into reference and current datasets
X_reference, y_reference = X[:1000], y[:1000]
X_current, y_current = X[1000:], y[1000:]

# Introduce drift in current dataset
X_current['feature_0'] = X_current['feature_0'] * 1.5
X_current['feature_1'] = X_current['feature_1'] + 2

# Train model and analyze impact
model = RandomForestRegressor(random_state=42)
model.fit(X_reference, y_reference)

analyzer = DriftImpactAnalyzer(model)
analyzer.set_baseline(X_reference, y_reference)
impact_analysis = analyzer.analyze_impact(X_current, y_current)

print("\nPerformance Degradation:")
print(pd.DataFrame([impact_analysis['performance_degradation']]))
print("\nFeature Impact Analysis:")
print(impact_analysis['feature_impact'].sort_values('drift_impact', ascending=False))
```

Slide 10: Time Series Drift Detection

Implementation of specialized drift detection methods for time series data, incorporating both distribution-based and temporal pattern analysis to identify significant changes in sequential data patterns.

```python
import numpy as np
import pandas as pd
from statstools import adfuller
from scipy.stats import ks_2samp

class TimeSeriesDriftDetector:
    def __init__(self, window_size=100, significance_level=0.05):
        self.window_size = window_size
        self.significance_level = significance_level
        self.baseline_stats = None
        
    def extract_features(self, time_series):
        """Extract statistical features from time series window"""
        return {
            'mean': np.mean(time_series),
            'std': np.std(time_series),
            'skew': pd.Series(time_series).skew(),
            'kurtosis': pd.Series(time_series).kurtosis(),
            'adf_pvalue': adfuller(time_series)[1]
        }
    
    def set_baseline(self, reference_series):
        """Establish baseline characteristics"""
        self.baseline_series = reference_series
        self.baseline_stats = self.extract_features(reference_series)
        
        # Calculate baseline seasonal patterns if applicable
        if len(reference_series) >= 2 * self.window_size:
            self.baseline_acf = np.correlate(reference_series, reference_series, mode='full')
            self.baseline_acf = self.baseline_acf[len(self.baseline_acf)//2:]
    
    def detect_drift(self, current_window):
        """Detect drift in current time window"""
        if len(current_window) < self.window_size:
            raise ValueError("Current window too small for analysis")
            
        current_stats = self.extract_features(current_window)
        
        # Statistical tests
        ks_statistic, ks_pvalue = ks_2samp(self.baseline_series, current_window)
        
        # Calculate temporal pattern changes
        current_acf = np.correlate(current_window, current_window, mode='full')
        current_acf = current_acf[len(current_acf)//2:]
        
        pattern_correlation = np.corrcoef(
            self.baseline_acf[:self.window_size],
            current_acf[:self.window_size]
        )[0,1]
        
        # Analyze feature differences
        feature_drifts = {}
        for feature in self.baseline_stats.keys():
            relative_change = abs(current_stats[feature] - self.baseline_stats[feature])
            if self.baseline_stats[feature] != 0:
                relative_change /= abs(self.baseline_stats[feature])
            feature_drifts[feature] = relative_change
        
        return {
            'distribution_drift': {
                'detected': ks_pvalue < self.significance_level,
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue
            },
            'pattern_drift': {
                'detected': pattern_correlation < 0.7,
                'correlation': pattern_correlation
            },
            'feature_drifts': feature_drifts,
            'overall_drift_detected': (ks_pvalue < self.significance_level) or 
                                    (pattern_correlation < 0.7) or
                                    (any(v > 0.3 for v in feature_drifts.values()))
        }

# Example usage
np.random.seed(42)

# Generate synthetic time series with drift
def generate_time_series(n_points, base_freq=0.1, drift_factor=0):
    t = np.linspace(0, n_points, n_points)
    series = np.sin(2 * np.pi * base_freq * t) + \
             0.5 * np.sin(2 * np.pi * 0.05 * t) + \
             np.random.normal(0, 0.1, n_points)
    if drift_factor > 0:
        series += drift_factor * t/n_points
    return series

# Generate reference and current data
reference_data = generate_time_series(500)
current_data_nodrift = generate_time_series(200)
current_data_drift = generate_time_series(200, drift_factor=0.5)

# Initialize detector
detector = TimeSeriesDriftDetector(window_size=100)
detector.set_baseline(reference_data)

# Analyze both scenarios
result_nodrift = detector.detect_drift(current_data_nodrift)
result_drift = detector.detect_drift(current_data_drift)

print("\nNo Drift Scenario Results:")
print(f"Overall drift detected: {result_nodrift['overall_drift_detected']}")
print("\nDrift Scenario Results:")
print(f"Overall drift detected: {result_drift['overall_drift_detected']}")
print("Feature drifts:", result_drift['feature_drifts'])
```

Slide 11: Automated Drift Response System

This implementation provides an autonomous system that monitors data drift and automatically triggers appropriate responses, including model retraining, feature importance analysis, and alerting mechanisms.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

class AutomatedDriftResponse:
    def __init__(self, 
                 model, 
                 drift_threshold=0.1,
                 performance_threshold=0.05,
                 retrain_window=1000):
        self.model = model
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.retrain_window = retrain_window
        self.baseline_metrics = None
        self.alert_history = []
        self.retraining_history = []
        
    def initialize_baseline(self, X_baseline, y_baseline):
        """Initialize baseline metrics and model performance"""
        self.X_baseline = X_baseline
        self.y_baseline = y_baseline
        
        # Calculate baseline performance
        X_train, X_val, y_train, y_val = train_test_split(
            X_baseline, y_baseline, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        baseline_score = self.model.score(X_val, y_val)
        
        # Calculate baseline feature distributions
        self.baseline_distributions = {
            col: {
                'mean': X_baseline[col].mean(),
                'std': X_baseline[col].std(),
                'quantiles': np.percentile(X_baseline[col], [25, 50, 75])
            }
            for col in X_baseline.columns
        }
        
        self.baseline_metrics = {
            'performance': baseline_score,
            'timestamp': datetime.now()
        }
    
    def evaluate_drift(self, X_current, y_current):
        """Evaluate current data for drift and trigger responses"""
        current_distributions = {
            col: {
                'mean': X_current[col].mean(),
                'std': X_current[col].std(),
                'quantiles': np.percentile(X_current[col], [25, 50, 75])
            }
            for col in X_current.columns
        }
        
        # Calculate distribution changes
        distribution_changes = {}
        for col in X_current.columns:
            mean_change = abs(current_distributions[col]['mean'] - 
                            self.baseline_distributions[col]['mean']) / \
                         self.baseline_distributions[col]['std']
            
            distribution_changes[col] = mean_change
        
        # Evaluate current performance
        current_score = self.model.score(X_current, y_current)
        performance_degradation = (self.baseline_metrics['performance'] - 
                                 current_score) / \
                                self.baseline_metrics['performance']
        
        # Determine if action is needed
        needs_action = (performance_degradation > self.performance_threshold or
                       any(change > self.drift_threshold 
                           for change in distribution_changes.values()))
        
        if needs_action:
            response = self._generate_response(
                X_current, y_current, 
                distribution_changes, 
                performance_degradation
            )
            return response
        
        return {
            'action_needed': False,
            'distribution_changes': distribution_changes,
            'performance_degradation': performance_degradation
        }
    
    def _generate_response(self, X_current, y_current, 
                          distribution_changes, performance_degradation):
        """Generate appropriate response to detected drift"""
        timestamp = datetime.now()
        
        # Determine most affected features
        affected_features = {k: v for k, v in distribution_changes.items() 
                           if v > self.drift_threshold}
        
        # Check if retraining is needed
        should_retrain = (len(self.retraining_history) == 0 or 
                         (timestamp - self.retraining_history[-1]['timestamp'])
                         .total_seconds() > self.retrain_window)
        
        response = {
            'action_needed': True,
            'timestamp': timestamp,
            'affected_features': affected_features,
            'performance_degradation': performance_degradation,
            'recommended_actions': []
        }
        
        # Add recommended actions
        if should_retrain:
            response['recommended_actions'].append({
                'action': 'retrain',
                'details': 'Model retraining recommended due to significant drift'
            })
            
            # Perform retraining
            X_combined = pd.concat([self.X_baseline, X_current])
            y_combined = pd.concat([self.y_baseline, y_current])
            self.model.fit(X_combined, y_combined)
            
            self.retraining_history.append({
                'timestamp': timestamp,
                'performance_before': self.baseline_metrics['performance'],
                'performance_after': self.model.score(X_current, y_current)
            })
        
        # Add monitoring alerts
        self.alert_history.append({
            'timestamp': timestamp,
            'drift_detected': True,
            'affected_features': affected_features,
            'performance_impact': performance_degradation
        })
        
        return response

# Example usage
np.random.seed(42)

# Generate synthetic data
X_baseline = pd.DataFrame(np.random.normal(0, 1, (1000, 5)), 
                         columns=[f'feature_{i}' for i in range(5)])
y_baseline = (X_baseline.sum(axis=1) > 0).astype(int)

X_current = pd.DataFrame(np.random.normal(0.5, 1.2, (500, 5)), 
                        columns=[f'feature_{i}' for i in range(5)])
y_current = (X_current.sum(axis=1) > 0).astype(int)

# Initialize system
model = RandomForestClassifier(random_state=42)
drift_system = AutomatedDriftResponse(model)
drift_system.initialize_baseline(X_baseline, y_baseline)

# Evaluate current data
response = drift_system.evaluate_drift(X_current, y_current)
print("\nDrift Response System Output:")
print(json.dumps(response, indent=2, default=str))
```

Slide 12: Online Learning with Drift Adaptation

Implementation of an online learning system that continuously adapts to data drift while maintaining model performance through incremental updates and adaptive learning rates.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # For sigmoid function

class OnlineDriftAdaptiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 learning_rate=0.01, 
                 forgetting_factor=0.99, 
                 drift_threshold=0.1):
        self.learning_rate = learning_rate
        self.forgetting_factor = forgetting_factor
        self.drift_threshold = drift_threshold
        self.weights = None
        self.scaler = StandardScaler()
        self.drift_history = []
        
    def _initialize_weights(self, n_features):
        """Initialize model weights"""
        self.weights = np.random.randn(n_features) / np.sqrt(n_features)
        self.bias = 0.0
        self.feature_importance = np.zeros(n_features)
        
    def _update_learning_rate(self, drift_magnitude):
        """Adapt learning rate based on drift magnitude"""
        return self.learning_rate * (1 + drift_magnitude * 2)
    
    def _compute_gradient(self, X, y, predictions):
        """Compute gradients for model update"""
        errors = y - predictions
        grad_w = -np.mean(errors.reshape(-1, 1) * X, axis=0)
        grad_b = -np.mean(errors)
        return grad_w, grad_b
    
    def _detect_drift(self, X_batch, y_batch):
        """Detect drift in incoming batch"""
        predictions = self.predict_proba(X_batch)
        current_loss = -np.mean(y_batch * np.log(predictions + 1e-10) + 
                              (1 - y_batch) * np.log(1 - predictions + 1e-10))
        
        if hasattr(self, 'previous_loss'):
            drift_magnitude = abs(current_loss - self.previous_loss) / max(self.previous_loss, 1e-10)
            drift_detected = drift_magnitude > self.drift_threshold
        else:
            drift_magnitude = 0
            drift_detected = False
            
        self.previous_loss = current_loss
        return drift_detected, drift_magnitude
    
    def partial_fit(self, X_batch, y_batch):
        """Update model with new batch of data"""
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Initialize weights if first batch
        if self.weights is None:
            self._initialize_weights(X_batch.shape[1])
            self.scaler.partial_fit(X_batch)
            
        # Scale features
        X_scaled = self.scaler.transform(X_batch)
        
        # Detect drift
        drift_detected, drift_magnitude = self._detect_drift(X_scaled, y_batch)
        
        if drift_detected:
            # Adapt learning rate
            adaptive_lr = self._update_learning_rate(drift_magnitude)
            
            # Update feature importance
            predictions = self.predict_proba(X_scaled)
            feature_impacts = np.abs(self.weights * np.std(X_scaled, axis=0))
            self.feature_importance = (self.forgetting_factor * self.feature_importance + 
                                    (1 - self.forgetting_factor) * feature_impacts)
            
            # Log drift event
            self.drift_history.append({
                'magnitude': drift_magnitude,
                'adaptive_lr': adaptive_lr,
                'feature_impacts': dict(zip(range(len(self.weights)), 
                                         self.feature_importance))
            })
        else:
            adaptive_lr = self.learning_rate
            
        # Compute predictions and gradients
        predictions = self.predict_proba(X_scaled)
        grad_w, grad_b = self._compute_gradient(X_scaled, y_batch, predictions)
        
        # Update weights with momentum
        if not hasattr(self, 'momentum_w'):
            self.momentum_w = np.zeros_like(self.weights)
            self.momentum_b = 0.0
            
        momentum_factor = 0.9
        self.momentum_w = (momentum_factor * self.momentum_w - 
                         adaptive_lr * grad_w)
        self.momentum_b = (momentum_factor * self.momentum_b - 
                         adaptive_lr * grad_b)
        
        self.weights += self.momentum_w
        self.bias += self.momentum_b
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        return expit(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X):
        """Predict classes"""
        return (self.predict_proba(X) >= 0.5).astype(int)

# Example usage with simulated drift
np.random.seed(42)

# Generate initial data
def generate_drift_data(n_samples, n_features, drift_factor=0):
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = (np.dot(X, true_weights) + drift_factor * 
         np.random.randn(n_samples) > 0).astype(int)
    return X, y

# Initialize model
model = OnlineDriftAdaptiveClassifier()

# Training with drift simulation
n_batches = 10
batch_size = 100
n_features = 5

print("Training with drift simulation:")
for i in range(n_batches):
    # Increase drift factor over time
    drift_factor = i * 0.2
    X_batch, y_batch = generate_drift_data(batch_size, n_features, drift_factor)
    
    # Update model
    model.partial_fit(X_batch, y_batch)
    
    # Print drift detection results
    if model.drift_history:
        latest_drift = model.drift_history[-1]
        print(f"\nBatch {i+1}:")
        print(f"Drift magnitude: {latest_drift['magnitude']:.4f}")
        print(f"Adaptive learning rate: {latest_drift['adaptive_lr']:.4f}")
```

Slide 13: Multivariate Drift Detection System

A sophisticated implementation for detecting and analyzing drift in multiple variables simultaneously, considering both individual feature changes and their interactions.

```python
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

class MultivariateDriftDetector:
    def __init__(self, significance_level=0.01, n_components=0.95):
        self.significance_level = significance_level
        self.n_components = n_components
        self.reference_distribution = None
        self.pca = None
        self.feature_correlations = None
        
    def fit_reference(self, reference_data):
        """Fit reference distribution and correlation structure"""
        self.reference_data = reference_data
        
        # Fit PCA for dimensionality reduction
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(reference_data)
        
        # Calculate reference distribution parameters
        self.reference_transformed = self.pca.transform(reference_data)
        self.reference_mean = np.mean(self.reference_transformed, axis=0)
        self.reference_cov = EmpiricalCovariance().fit(self.reference_transformed)
        
        # Calculate feature correlations
        self.feature_correlations = pd.DataFrame(
            np.corrcoef(reference_data.T),
            columns=reference_data.columns,
            index=reference_data.columns
        )
        
    def calculate_hotelling_t2(self, data):
        """Calculate Hotelling's T2 statistic"""
        transformed_data = self.pca.transform(data)
        diff = transformed_data - self.reference_mean
        
        t2_stats = []
        for i in range(len(diff)):
            t2 = diff[i].dot(
                np.linalg.inv(self.reference_cov.covariance_)
            ).dot(diff[i])
            t2_stats.append(t2)
            
        return np.array(t2_stats)
    
    def detect_drift(self, current_data):
        """Detect multivariate drift in current data"""
        # Calculate Hotelling's T2 statistics
        t2_stats = self.calculate_hotelling_t2(current_data)
        
        # Calculate critical value
        n_components = self.reference_transformed.shape[1]
        n_samples = len(self.reference_transformed)
        
        f_critical = stats.f.ppf(
            1 - self.significance_level,
            n_components,
            n_samples - n_components
        )
        
        critical_value = (
            n_components * (n_samples - 1) / 
            (n_samples - n_components)
        ) * f_critical
        
        # Detect drift
        drift_points = t2_stats > critical_value
        total_drift = np.mean(drift_points)
        
        # Calculate contribution of each feature
        feature_contributions = self._calculate_feature_contributions(
            current_data
        )
        
        # Analyze correlation changes
        correlation_changes = self._analyze_correlation_changes(
            current_data
        )
        
        return {
            'drift_detected': total_drift > self.significance_level,
            'drift_magnitude': total_drift,
            'critical_value': critical_value,
            'feature_contributions': feature_contributions,
            'correlation_changes': correlation_changes,
            'n_drift_points': sum(drift_points),
            'total_points': len(drift_points)
        }
    
    def _calculate_feature_contributions(self, current_data):
        """Calculate contribution of each feature to drift"""
        feature_contributions = {}
        
        for feature in current_data.columns:
            # Calculate KL divergence for each feature
            reference_dist = self.reference_data[feature]
            current_dist = current_data[feature]
            
            kl_div = self._calculate_kl_divergence(
                reference_dist,
                current_dist
            )
            
            feature_contributions[feature] = {
                'kl_divergence': kl_div,
                'mean_shift': (current_dist.mean() - reference_dist.mean()) / 
                             reference_dist.std(),
                'std_ratio': current_dist.std() / reference_dist.std()
            }
            
        return feature_contributions
    
    def _analyze_correlation_changes(self, current_data):
        """Analyze changes in feature correlations"""
        current_corr = pd.DataFrame(
            np.corrcoef(current_data.T),
            columns=current_data.columns,
            index=current_data.columns
        )
        
        correlation_changes = pd.DataFrame(
            np.abs(current_corr - self.feature_correlations),
            columns=current_data.columns,
            index=current_data.columns
        )
        
        return {
            'max_change': correlation_changes.max().max(),
            'mean_change': correlation_changes.mean().mean(),
            'significant_changes': self._get_significant_correlation_changes(
                correlation_changes
            )
        }
    
    def _calculate_kl_divergence(self, p, q):
        """Calculate KL divergence between two distributions"""
        # Use kernel density estimation for continuous distributions
        p_kde = stats.gaussian_kde(p)
        q_kde = stats.gaussian_kde(q)
        
        # Evaluate on a grid
        x_grid = np.linspace(
            min(p.min(), q.min()),
            max(p.max(), q.max()),
            100
        )
        
        p_pdf = p_kde(x_grid)
        q_pdf = q_kde(x_grid)
        
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        kl_div = np.sum(p_pdf * np.log((p_pdf + epsilon) / (q_pdf + epsilon)))
        
        return kl_div
    
    def _get_significant_correlation_changes(self, correlation_changes):
        """Identify significant correlation changes"""
        significant_changes = []
        
        for i in range(len(correlation_changes)):
            for j in range(i + 1, len(correlation_changes)):
                if correlation_changes.iloc[i, j] > 0.2:  # Threshold
                    significant_changes.append({
                        'features': (
                            correlation_changes.index[i],
                            correlation_changes.columns[j]
                        ),
                        'change': correlation_changes.iloc[i, j]
                    })
                    
        return sorted(
            significant_changes,
            key=lambda x: x['change'],
            reverse=True
        )

# Example usage
np.random.seed(42)

# Generate synthetic data with drift
def generate_multivariate_data(n_samples, n_features, drift_factor=0):
    # Generate correlated features
    cov_matrix = np.eye(n_features)
    cov_matrix[cov_matrix == 0] = 0.3
    
    # Add drift to mean and covariance
    mean = np.zeros(n_features) + drift_factor
    cov_matrix = cov_matrix * (1 + drift_factor * 0.2)
    
    data = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    return pd.DataFrame(
        data,
        columns=[f'feature_{i}' for i in range(n_features)]
    )

# Generate reference and current data
reference_data = generate_multivariate_data(1000, 5)
current_data = generate_multivariate_data(500, 5, drift_factor=0.5)

# Initialize and use detector
detector = MultivariateDriftDetector()
detector.fit_reference(reference_data)
results = detector.detect_drift(current_data)

print("\nMultivariate Drift Detection Results:")
print(f"Drift Detected: {results['drift_detected']}")
print(f"Drift Magnitude: {results['drift_magnitude']:.4f}")
print("\nFeature Contributions:")
for feature, metrics in results['feature_contributions'].items():
    print(f"\n{feature}:")
    print(f"KL Divergence: {metrics['kl_divergence']:.4f}")
    print(f"Mean Shift: {metrics['mean_shift']:.4f}")
    print(f"Std Ratio: {metrics['std_ratio']:.4f}")
```

Slide 14: Ensemble-based Drift Detection

Implementation of an ensemble approach that combines multiple drift detection methods to provide more robust and reliable drift detection across different types of changes in data distribution.

```python
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mutual_info_score

class EnsembleDriftDetector:
    def __init__(self, methods=['ks', 'isolation', 'correlation', 'mutual_info'],
                 threshold=0.6):
        self.methods = methods
        self.threshold = threshold
        self.baseline_stats = {}
        self.detector_weights = {
            'ks': 0.3,
            'isolation': 0.25,
            'correlation': 0.25,
            'mutual_info': 0.2
        }
        
    def fit_reference(self, reference_data):
        """Initialize reference statistics for all detection methods"""
        self.reference_data = reference_data
        
        if 'isolation' in self.methods:
            self.isolation_forest = IsolationForest(random_state=42)
            self.isolation_forest.fit(reference_data)
            
        if 'correlation' in self.methods:
            self.baseline_stats['correlation'] = reference_data.corr()
            
        if 'mutual_info' in self.methods:
            self.baseline_stats['mutual_info'] = self._calculate_mutual_info_matrix(
                reference_data
            )
            
        # Store reference distributions
        self.reference_distributions = {
            col: {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std(),
                'quantiles': np.percentile(reference_data[col], [25, 50, 75])
            }
            for col in reference_data.columns
        }
        
    def _calculate_mutual_info_matrix(self, data):
        """Calculate mutual information between all feature pairs"""
        n_features = data.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                mi_matrix[i, j] = mutual_info_score(
                    data.iloc[:, i],
                    data.iloc[:, j]
                )
                
        return pd.DataFrame(
            mi_matrix,
            index=data.columns,
            columns=data.columns
        )
    
    def detect_drift(self, current_data):
        """Perform ensemble drift detection"""
        detector_results = {}
        
        if 'ks' in self.methods:
            detector_results['ks'] = self._detect_ks_drift(current_data)
            
        if 'isolation' in self.methods:
            detector_results['isolation'] = self._detect_isolation_drift(
                current_data
            )
            
        if 'correlation' in self.methods:
            detector_results['correlation'] = self._detect_correlation_drift(
                current_data
            )
            
        if 'mutual_info' in self.methods:
            detector_results['mutual_info'] = self._detect_mutual_info_drift(
                current_data
            )
            
        # Combine detector results
        ensemble_score = self._combine_detector_results(detector_results)
        
        return {
            'ensemble_drift_score': ensemble_score,
            'drift_detected': ensemble_score > self.threshold,
            'detector_results': detector_results,
            'feature_drifts': self._analyze_feature_drifts(
                current_data,
                detector_results
            )
        }
        
    def _detect_ks_drift(self, current_data):
        """Detect drift using Kolmogorov-Smirnov test"""
        ks_results = {}
        
        for column in current_data.columns:
            statistic, p_value = stats.ks_2samp(
                self.reference_data[column],
                current_data[column]
            )
            
            ks_results[column] = {
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            }
            
        return {
            'feature_results': ks_results,
            'drift_score': np.mean([
                res['drift_detected'] for res in ks_results.values()
            ])
        }
        
    def _detect_isolation_drift(self, current_data):
        """Detect drift using Isolation Forest"""
        # Get anomaly scores for current data
        anomaly_scores = -self.isolation_forest.score_samples(current_data)
        reference_scores = -self.isolation_forest.score_samples(self.reference_data)
        
        # Compare score distributions
        statistic, p_value = stats.ks_2samp(reference_scores, anomaly_scores)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'drift_score': statistic if p_value < 0.05 else 0,
            'anomaly_ratio': np.mean(anomaly_scores > np.percentile(reference_scores, 95))
        }
        
    def _detect_correlation_drift(self, current_data):
        """Detect drift in feature correlations"""
        current_corr = current_data.corr()
        correlation_changes = np.abs(
            current_corr - self.baseline_stats['correlation']
        )
        
        return {
            'max_change': correlation_changes.max().max(),
            'mean_change': correlation_changes.mean().mean(),
            'drift_score': correlation_changes.mean().mean() > 0.2
        }
        
    def _detect_mutual_info_drift(self, current_data):
        """Detect drift in mutual information between features"""
        current_mi = self._calculate_mutual_info_matrix(current_data)
        mi_changes = np.abs(
            current_mi - self.baseline_stats['mutual_info']
        )
        
        return {
            'max_change': mi_changes.max().max(),
            'mean_change': mi_changes.mean().mean(),
            'drift_score': mi_changes.mean().mean() > 0.2
        }
        
    def _combine_detector_results(self, detector_results):
        """Combine results from all detectors using weighted voting"""
        weighted_scores = []
        
        for method, result in detector_results.items():
            if isinstance(result.get('drift_score'), (int, float)):
                weighted_scores.append(
                    result['drift_score'] * self.detector_weights[method]
                )
                
        return np.mean(weighted_scores) if weighted_scores else 0
        
    def _analyze_feature_drifts(self, current_data, detector_results):
        """Analyze drift patterns for individual features"""
        feature_analysis = {}
        
        for column in current_data.columns:
            feature_scores = []
            
            # Combine evidence from different detectors
            if 'ks' in detector_results:
                feature_scores.append(
                    detector_results['ks']['feature_results'][column]['drift_detected']
                )
                
            # Add distribution changes
            current_stats = {
                'mean': current_data[column].mean(),
                'std': current_data[column].std(),
                'quantiles': np.percentile(current_data[column], [25, 50, 75])
            }
            
            ref_stats = self.reference_distributions[column]
            
            feature_analysis[column] = {
                'drift_score': np.mean(feature_scores),
                'mean_shift': abs(
                    current_stats['mean'] - ref_stats['mean']
                ) / ref_stats['std'],
                'std_ratio': current_stats['std'] / ref_stats['std'],
                'quantile_shifts': [
                    abs(c - r) / r for c, r in zip(
                        current_stats['quantiles'],
                        ref_stats['quantiles']
                    )
                ]
            }
            
        return feature_analysis

# Example usage
np.random.seed(42)

# Generate synthetic data with drift
def generate_drifted_data(n_samples, n_features, drift_type='mean_shift'):
    if drift_type == 'mean_shift':
        reference = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        current = pd.DataFrame(
            np.random.normal(0.5, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
    elif drift_type == 'correlation_change':
        # Generate correlated data for reference
        cov_matrix = np.eye(n_features)
        cov_matrix[cov_matrix == 0] = 0.7
        reference = pd.DataFrame(
            np.random.multivariate_normal(
                np.zeros(n_features),
                cov_matrix,
                n_samples
            ),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate less correlated data for current
        cov_matrix[cov_matrix == 0.7] = 0.2
        current = pd.DataFrame(
            np.random.multivariate_normal(
                np.zeros(n_features),
                cov_matrix,
                n_samples
            ),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
    
    return reference, current

# Test with different drift types
reference_data, current_data_mean = generate_drifted_data(1000, 5, 'mean_shift')
_, current_data_corr = generate_drifted_data(1000, 5, 'correlation_change')

# Initialize and use ensemble detector
detector = EnsembleDriftDetector()
detector.fit_reference(reference_data)

# Test mean shift drift
mean_shift_results = detector.detect_drift(current_data_mean)
correlation_shift_results = detector.detect_drift(current_data_corr)

print("\nMean Shift Drift Results:")
print(f"Ensemble Drift Score: {mean_shift_results['ensemble_drift_score']:.4f}")
print(f"Drift Detected: {mean_shift_results['drift_detected']}")

print("\nCorrelation Shift Drift Results:")
print(f"Ensemble Drift Score: {correlation_shift_results['ensemble_drift_score']:.4f}")
print(f"Drift Detected: {correlation_shift_results['drift_detected']}")
```

Slide 15: Additional Resources

*   Learning with Concept Drift: A Comprehensive Survey [https://arxiv.org/abs/2104.05785](https://arxiv.org/abs/2104.05785)
*   Detecting and Correcting for Label Shift with Black Box Predictors [https://arxiv.org/abs/1802.03916](https://arxiv.org/abs/1802.03916)
*   A Survey on Concept Drift Adaptation [https://arxiv.org/abs/1010.4784](https://arxiv.org/abs/1010.4784)
*   Adaptive Learning Under Covariate Shift [https://arxiv.org/abs/1903.01872](https://arxiv.org/abs/1903.01872)
*   Real-time Adaptive Detection of Data Drift and Anomalies [https://arxiv.org/abs/2009.09795](https://arxiv.org/abs/2009.09795)

