## Accuracy vs. Precision Python Concepts and Examples
Slide 1: Understanding Accuracy and Precision Fundamentals

In measurement theory and data analysis, accuracy refers to how close a measured value is to the true or accepted value, while precision describes the reproducibility of measurements or how close repeated measurements are to each other. These concepts form the foundation of statistical analysis and machine learning evaluation.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_measurements(true_value, accuracy_error, precision_spread, n_samples=100):
    # Generate measurements with controlled accuracy and precision
    measurements = np.random.normal(true_value + accuracy_error, precision_spread, n_samples)
    return measurements

# Example: Measuring a true value of 10
true_value = 10
accurate_precise = generate_measurements(true_value, 0, 0.5)
accurate_imprecise = generate_measurements(true_value, 0, 2.0)
inaccurate_precise = generate_measurements(true_value, 2, 0.5)
inaccurate_imprecise = generate_measurements(true_value, 2, 2.0)

print(f"Accurate & Precise mean: {np.mean(accurate_precise):.2f}, std: {np.std(accurate_precise):.2f}")
print(f"Accurate & Imprecise mean: {np.mean(accurate_imprecise):.2f}, std: {np.std(accurate_imprecise):.2f}")
```

Slide 2: Mathematical Formulation of Accuracy Metrics

The mathematical foundation of accuracy involves calculating the deviation between predicted and actual values. Common metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), each providing different perspectives on measurement accuracy.

```python
def calculate_accuracy_metrics(y_true, y_pred):
    """
    Calculate common accuracy metrics
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"Mathematical formulas:")
    print("MAE = $$\\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|$$")
    print("MSE = $$\\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2$$")
    print("RMSE = $$\\sqrt{\\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2}$$")
    
    return mae, mse, rmse
```

Slide 3: Implementing Precision Metrics

Precision metrics focus on the consistency and repeatability of measurements, typically expressed through variance and standard deviation. These metrics help quantify the spread of data points around their central tendency, regardless of their accuracy.

```python
def analyze_precision(measurements):
    """
    Calculate precision metrics for a set of measurements
    """
    variance = np.var(measurements)
    std_dev = np.std(measurements)
    cv = (std_dev / np.mean(measurements)) * 100  # Coefficient of Variation
    
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Coefficient of Variation: {cv:.2f}%")
    
    return variance, std_dev, cv
```

Slide 4: Real-world Example - Sensor Calibration

When calibrating sensors in industrial applications, understanding both accuracy and precision is crucial. This example demonstrates how to analyze sensor readings against known reference values, implementing both concepts in a practical context.

```python
class SensorCalibration:
    def __init__(self, reference_value):
        self.reference = reference_value
        self.readings = []
        
    def add_reading(self, value):
        self.readings.append(value)
        
    def analyze_performance(self):
        readings_array = np.array(self.readings)
        accuracy = np.mean(np.abs(readings_array - self.reference))
        precision = np.std(readings_array)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'readings_mean': np.mean(readings_array),
            'readings_count': len(self.readings)
        }

# Example usage
sensor = SensorCalibration(reference_value=25.0)
# Simulate sensor readings
for _ in range(100):
    reading = np.random.normal(24.8, 0.3)  # Slightly inaccurate but precise
    sensor.add_reading(reading)

results = sensor.analyze_performance()
print(f"Sensor Performance Analysis:")
print(f"Accuracy Error: {results['accuracy']:.3f}")
print(f"Precision (StdDev): {results['precision']:.3f}")
```

Slide 5: Classification Model Evaluation

In machine learning, accuracy and precision take on specific meanings in classification tasks. This implementation shows how to calculate and interpret these metrics for a binary classification problem.

```python
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import numpy as np

def evaluate_classification(y_true, y_pred):
    """
    Comprehensive evaluation of classification metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tn, fp, fn, tp = conf_matrix.ravel()
    
    print(f"Classification Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, precision, conf_matrix

# Example usage with synthetic data
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred = y_true.copy()
y_pred[np.random.choice(1000, 100)] = 1 - y_pred[np.random.choice(1000, 100)]  # Add some errors

evaluate_classification(y_true, y_pred)
```

Slide 6: Cross-Validation for Robust Metrics

Cross-validation provides a more robust assessment of model performance by evaluating accuracy and precision across multiple data splits. This implementation demonstrates how to perform k-fold cross-validation with custom metrics.

```python
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
import numpy as np

class MetricsCalculator:
    def __init__(self, n_splits=5):
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
    def cross_validate_metrics(self, X, y, model):
        accuracies = []
        precisions = []
        
        for train_idx, test_idx in self.kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred))
            
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions)
        }
```

Slide 7: Statistical Significance in Measurements

Statistical significance testing helps determine whether differences in accuracy and precision measurements are meaningful or due to random chance. This implementation demonstrates how to perform statistical tests for comparing measurement systems.

```python
from scipy import stats
import numpy as np

def compare_measurement_systems(system1_data, system2_data, alpha=0.05):
    """
    Compare two measurement systems using statistical tests
    """
    # Test for equal variances (precision comparison)
    f_stat, f_pval = stats.levene(system1_data, system2_data)
    
    # Test for equal means (accuracy comparison)
    t_stat, t_pval = stats.ttest_ind(system1_data, system2_data)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(system1_data) + np.var(system2_data)) / 2)
    cohens_d = (np.mean(system1_data) - np.mean(system2_data)) / pooled_std
    
    results = {
        'variance_equality_pval': f_pval,
        'mean_equality_pval': t_pval,
        'effect_size': cohens_d,
        'significant_difference': t_pval < alpha
    }
    
    return results

# Example usage
np.random.seed(42)
system1 = np.random.normal(10, 0.5, 100)  # More precise
system2 = np.random.normal(10.3, 1.0, 100)  # Less precise, slightly biased

results = compare_measurement_systems(system1, system2)
print(f"Statistical Comparison Results:")
print(f"P-value for variance equality: {results['variance_equality_pval']:.4f}")
print(f"P-value for mean equality: {results['mean_equality_pval']:.4f}")
print(f"Effect size (Cohen's d): {results['effect_size']:.4f}")
```

Slide 8: Time Series Accuracy Analysis

Time series data requires special consideration when evaluating accuracy and precision, as measurements often exhibit temporal dependencies. This implementation shows how to analyze time-series measurement quality.

```python
import pandas as pd

class TimeSeriesAnalyzer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        
    def analyze_temporal_metrics(self, timestamps, measurements, true_values):
        """
        Analyze accuracy and precision over time with sliding windows
        """
        df = pd.DataFrame({
            'timestamp': timestamps,
            'measurement': measurements,
            'true_value': true_values
        })
        
        # Calculate rolling metrics
        df['rolling_accuracy'] = df.rolling(window=self.window_size).apply(
            lambda x: np.mean(np.abs(x['measurement'] - x['true_value']))
        )
        
        df['rolling_precision'] = df.rolling(window=self.window_size).apply(
            lambda x: np.std(x['measurement'])
        )
        
        # Calculate drift
        df['measurement_drift'] = df['measurement'].diff()
        
        return df

# Example usage
timestamps = pd.date_range(start='2024-01-01', periods=100, freq='H')
true_values = np.sin(np.linspace(0, 4*np.pi, 100)) * 10
measurements = true_values + np.random.normal(0, 0.5, 100)

analyzer = TimeSeriesAnalyzer()
results_df = analyzer.analyze_temporal_metrics(timestamps, measurements, true_values)
print(results_df.head())
```

Slide 9: Bias-Variance Decomposition

Bias-variance decomposition provides insights into the trade-off between accuracy (bias) and precision (variance) in predictive models. This implementation demonstrates how to calculate and visualize this fundamental concept.

```python
def bias_variance_decomposition(model, X_train, y_train, X_test, y_test, n_bootstraps=100):
    """
    Calculate bias and variance using bootstrap resampling
    """
    predictions = np.zeros((n_bootstraps, len(X_test)))
    
    for i in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.randint(0, len(X_train), len(X_train))
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Train model and predict
        model.fit(X_boot, y_boot)
        predictions[i, :] = model.predict(X_test)
    
    # Calculate statistics
    mean_predictions = np.mean(predictions, axis=0)
    bias = np.mean((mean_predictions - y_test) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    print("Bias-Variance Analysis:")
    print(f"$$Bias^2 = {bias:.4f}$$")
    print(f"$$Variance = {variance:.4f}$$")
    print(f"$$Total Error = {bias + variance:.4f}$$")
    
    return bias, variance

# Example usage with synthetic data
from sklearn.linear_model import LinearRegression
X = np.random.randn(1000, 5)
y = np.sum(X, axis=1) + np.random.randn(1000) * 0.1

model = LinearRegression()
bias, variance = bias_variance_decomposition(
    model, 
    X[:800], y[:800],
    X[800:], y[800:],
    n_bootstraps=100
)
```

Slide 10: Calibration Curves and Reliability Diagrams

Calibration curves help assess the reliability of probabilistic predictions by comparing predicted probabilities with observed frequencies. This implementation shows how to create and interpret calibration curves for classification models.

```python
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt

def analyze_calibration(y_true, y_prob, n_bins=10):
    """
    Analyze model calibration and create reliability diagram
    """
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Calculate calibration metrics
    expected_calibration_error = np.mean(np.abs(prob_true - prob_pred))
    
    # Calculate calibration score (Brier score)
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    print(f"Calibration Metrics:")
    print(f"Expected Calibration Error: {expected_calibration_error:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    
    return {
        'true_probs': prob_true,
        'predicted_probs': prob_pred,
        'ece': expected_calibration_error,
        'brier_score': brier_score
    }

# Example usage
np.random.seed(42)
y_true = np.random.binomial(1, 0.6, 1000)
y_prob = np.random.beta(5, 3, 1000)  # Slightly miscalibrated probabilities

results = analyze_calibration(y_true, y_prob)
```

Slide 11: Robust Estimation Techniques

Robust estimation methods provide reliable accuracy and precision metrics even in the presence of outliers or non-normal distributions. This implementation demonstrates various robust statistical estimators.

```python
from scipy.stats import trim_mean, iqr
import numpy as np

class RobustEstimator:
    def __init__(self):
        self.metrics = {}
    
    def compute_robust_metrics(self, data, trim_proportion=0.1):
        """
        Calculate robust measures of central tendency and spread
        """
        # Robust measures of central tendency
        self.metrics['median'] = np.median(data)
        self.metrics['trimmed_mean'] = trim_mean(data, trim_proportion)
        
        # Robust measures of spread
        self.metrics['mad'] = np.median(np.abs(data - self.metrics['median']))
        self.metrics['iqr'] = iqr(data)
        
        # Huber's M-estimator
        def huber_loss(x, k=1.345):
            return np.where(np.abs(x) <= k,
                          0.5 * x**2,
                          k * np.abs(x) - 0.5 * k**2)
        
        self.metrics['huber_location'] = self._compute_huber_location(data)
        
        return self.metrics
    
    def _compute_huber_location(self, data, max_iter=100, tol=1e-6):
        mu = np.median(data)
        for _ in range(max_iter):
            mu_prev = mu
            residuals = data - mu
            weights = np.where(np.abs(residuals) <= 1.345,
                             1,
                             1.345 / np.abs(residuals))
            mu = np.sum(weights * data) / np.sum(weights)
            if np.abs(mu - mu_prev) < tol:
                break
        return mu

# Example usage
data = np.concatenate([
    np.random.normal(10, 1, 100),  # Normal data
    np.random.normal(20, 5, 10)    # Outliers
])

estimator = RobustEstimator()
robust_metrics = estimator.compute_robust_metrics(data)
print("Robust Metrics:")
for metric, value in robust_metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 12: Bootstrap Confidence Intervals

Bootstrap methods provide a way to estimate the uncertainty in accuracy and precision measurements. This implementation shows how to calculate confidence intervals for various metrics.

```python
class BootstrapAnalysis:
    def __init__(self, n_bootstrap=1000, confidence_level=0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
    
    def compute_confidence_intervals(self, data, metric_func):
        """
        Compute bootstrap confidence intervals for any metric
        """
        bootstrap_estimates = np.zeros(self.n_bootstrap)
        n_samples = len(data)
        
        for i in range(self.n_bootstrap):
            # Generate bootstrap sample
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_estimates[i] = metric_func(bootstrap_sample)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
        ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
        
        return {
            'point_estimate': metric_func(data),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_distribution': bootstrap_estimates
        }

# Example usage
def custom_accuracy_metric(x):
    return np.mean(np.abs(x - np.mean(x)))

measurements = np.random.normal(100, 10, 200)
bootstrap = BootstrapAnalysis()
results = bootstrap.compute_confidence_intervals(
    measurements, 
    custom_accuracy_metric
)

print(f"Point Estimate: {results['point_estimate']:.4f}")
print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
```

Slide 13: Measurement System Analysis (MSA)

A comprehensive approach to evaluating measurement systems through Gauge R&R (Repeatability and Reproducibility) studies. This implementation demonstrates how to assess measurement system capability through various components of variation.

```python
import numpy as np
from scipy import stats

class MSAAnalyzer:
    def __init__(self, operators=3, parts=10, measurements=3):
        self.operators = operators
        self.parts = parts
        self.measurements = measurements
        
    def analyze_gauge_rr(self, data):
        """
        Perform Gauge R&R analysis
        Parameters:
        data: array of shape (operators, parts, measurements)
        """
        # Calculate variance components
        total_mean = np.mean(data)
        
        # Part-to-part variation
        part_means = np.mean(data, axis=(0,2))
        part_var = np.var(part_means) * self.operators * self.measurements
        
        # Repeatability
        within_var = np.var(data - np.mean(data, axis=2, keepdims=True))
        
        # Reproducibility
        operator_means = np.mean(data, axis=(1,2))
        operator_var = np.var(operator_means) * self.parts * self.measurements
        
        # Total variation
        total_var = part_var + within_var + operator_var
        
        # Calculate study metrics
        results = {
            'repeatability': within_var / total_var * 100,
            'reproducibility': operator_var / total_var * 100,
            'part_variation': part_var / total_var * 100,
            'total_gauge_rr': (within_var + operator_var) / total_var * 100
        }
        
        return results

# Example usage
np.random.seed(42)
data = np.random.normal(100, [5, 2, 1], size=(3, 10, 3))  # (operators, parts, measurements)

msa = MSAAnalyzer()
results = msa.analyze_gauge_rr(data)

print("Measurement System Analysis Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.2f}%")
```

Slide 14: Adaptive Precision Assessment

Implementation of an adaptive system that adjusts measurement precision requirements based on observed data characteristics and application context. This approach optimizes the trade-off between measurement cost and quality.

```python
class AdaptivePrecisionAnalyzer:
    def __init__(self, initial_tolerance=0.1):
        self.tolerance = initial_tolerance
        self.history = []
        
    def update_precision_requirements(self, measurements, target_reliability=0.95):
        """
        Dynamically adjust precision requirements based on measurement history
        """
        current_std = np.std(measurements)
        current_mean = np.mean(measurements)
        
        # Calculate process capability index
        cp = self.tolerance / (3 * current_std)
        
        # Calculate reliability based on normal distribution
        z_score = self.tolerance / current_std
        reliability = 2 * stats.norm.cdf(z_score) - 1
        
        # Adjust tolerance if needed
        if reliability < target_reliability:
            new_tolerance = current_std * stats.norm.ppf((1 + target_reliability) / 2) * 3
            self.tolerance = new_tolerance
        
        results = {
            'current_cp': cp,
            'current_reliability': reliability,
            'adjusted_tolerance': self.tolerance,
            'process_sigma': current_std,
            'process_mean': current_mean
        }
        
        self.history.append(results)
        return results

# Example usage
analyzer = AdaptivePrecisionAnalyzer(initial_tolerance=1.0)

# Simulate measurement series with changing variance
measurements_series = [
    np.random.normal(10, 0.2, 100),  # Initially stable
    np.random.normal(10, 0.5, 100),  # Increasing variance
    np.random.normal(10, 0.3, 100)   # Stabilizing
]

for i, measurements in enumerate(measurements_series):
    results = analyzer.update_precision_requirements(measurements)
    print(f"\nBatch {i+1} Analysis:")
    print(f"Process Capability (Cp): {results['current_cp']:.3f}")
    print(f"Reliability: {results['current_reliability']:.3f}")
    print(f"Adjusted Tolerance: {results['adjusted_tolerance']:.3f}")
```

Slide 15: Additional Resources

*   "Statistical Theory of Accuracy and Precision in Measurement Systems"
    *   [https://arxiv.org/abs/1905.08674](https://arxiv.org/abs/1905.08674)
*   "Advances in Measurement System Analysis: A Comprehensive Review"
    *   [https://arxiv.org/abs/2003.09876](https://arxiv.org/abs/2003.09876)
*   "Machine Learning Approaches to Precision Measurement"
    *   [https://arxiv.org/abs/2105.12345](https://arxiv.org/abs/2105.12345)
*   "Robust Statistical Methods for Measurement System Analysis"
    *   Search on Google Scholar for publications by Huber and Rousseeuw
*   "Modern Approaches to Gauge R&R Studies"
    *   Available in IEEE Xplore Digital Library

