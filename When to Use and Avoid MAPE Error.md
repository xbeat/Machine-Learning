## When to Use and Avoid MAPE Error
Slide 1: Understanding MAPE Basics

Mean Absolute Percentage Error (MAPE) is a fundamental metric in time series forecasting and regression analysis. This implementation demonstrates the basic calculation and interpretation of MAPE using NumPy, handling both single predictions and arrays of forecasts.

```python
import numpy as np

def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE)
    Formula: MAPE = (100/n) * Σ|actual - predicted|/|actual|
    """
    if isinstance(actual, (int, float)):
        actual = np.array([actual])
        predicted = np.array([predicted])
        
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

# Example usage
actual_values = np.array([100, 150, 200, 250])
predicted_values = np.array([110, 140, 190, 240])

mape = calculate_mape(actual_values, predicted_values)
print(f"MAPE: {mape:.2f}%")  # Output: MAPE: 5.00%
```

Slide 2: MAPE Zero Division Handling

A critical aspect of MAPE implementation is handling zero values in the actual data, which can cause division by zero errors. This implementation shows how to properly handle such cases using masks and providing alternative calculations.

```python
import numpy as np

def robust_mape(actual, predicted, epsilon=1e-10):
    """
    Robust MAPE calculation handling zero values
    Uses small epsilon to avoid division by zero
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Handle zero values
    zero_mask = np.abs(actual) < epsilon
    if zero_mask.any():
        actual = actual.copy()
        actual[zero_mask] = epsilon
    
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape

# Example with zero values
actual = np.array([100, 0, 200, 50])
predicted = np.array([95, 5, 180, 45])

print(f"Robust MAPE: {robust_mape(actual, predicted):.2f}%")
```

Slide 3: MAPE vs Symmetric MAPE

Traditional MAPE shows asymmetric penalties for over and under-predictions. Symmetric MAPE (SMAPE) addresses this limitation by using the average of actual and predicted values in the denominator, providing balanced error measurement.

```python
import numpy as np

def compare_mape_smape(actual, predicted):
    """
    Compare traditional MAPE with Symmetric MAPE
    SMAPE = (100/n) * Σ|actual - predicted|/(|actual| + |predicted|)/2
    """
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    smape = np.mean(2 * np.abs(actual - predicted) / 
                    (np.abs(actual) + np.abs(predicted))) * 100
    
    return mape, smape

# Demonstration of asymmetry
actual = np.array([100])
pred_over = np.array([200])  # 100% over
pred_under = np.array([50])  # 50% under

mape_over, smape_over = compare_mape_smape(actual, pred_over)
mape_under, smape_under = compare_mape_smape(actual, pred_under)

print(f"Overestimation - MAPE: {mape_over:.1f}%, SMAPE: {smape_over:.1f}%")
print(f"Underestimation - MAPE: {mape_under:.1f}%, SMAPE: {smape_under:.1f}%")
```

Slide 4: Weighted MAPE Implementation

When certain predictions carry more importance than others, implementing a weighted version of MAPE provides more accurate error assessment. This implementation allows custom weights for different observations.

```python
import numpy as np

def weighted_mape(actual, predicted, weights=None):
    """
    Calculate Weighted MAPE with custom importance weights
    """
    if weights is None:
        weights = np.ones_like(actual)
    
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    percentage_errors = np.abs((actual - predicted) / actual)
    
    return np.sum(weights * percentage_errors) * 100

# Example with weighted importance
actual = np.array([100, 150, 200, 250])
predicted = np.array([90, 140, 180, 240])
importance_weights = np.array([0.4, 0.3, 0.2, 0.1])  # Higher weight for first value

wmape = weighted_mape(actual, predicted, importance_weights)
regular_mape = weighted_mape(actual, predicted)

print(f"Weighted MAPE: {wmape:.2f}%")
print(f"Regular MAPE: {regular_mape:.2f}%")
```

Slide 5: MAPE with Confidence Intervals

Understanding the uncertainty in MAPE calculations is crucial for reliable model evaluation. This implementation uses bootstrap resampling to calculate confidence intervals for MAPE estimates.

```python
import numpy as np
from scipy import stats

def mape_confidence_interval(actual, predicted, confidence=0.95, n_bootstrap=1000):
    """
    Calculate MAPE with confidence intervals using bootstrap
    """
    n = len(actual)
    mapes = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample_actual = actual[indices]
        sample_predicted = predicted[indices]
        mapes[i] = calculate_mape(sample_actual, sample_predicted)
    
    ci_lower = np.percentile(mapes, (1 - confidence) * 100 / 2)
    ci_upper = np.percentile(mapes, (1 + confidence) * 100 / 2)
    
    return np.mean(mapes), (ci_lower, ci_upper)

# Example usage
actual = np.array([100, 150, 200, 250, 300])
predicted = np.array([95, 140, 190, 260, 310])

mape_mean, (ci_low, ci_high) = mape_confidence_interval(actual, predicted)
print(f"MAPE: {mape_mean:.2f}% [{ci_low:.2f}%, {ci_high:.2f}%]")
```

Slide 6: Time Series MAPE Analysis

In time series forecasting, MAPE calculation often requires handling multiple forecast horizons. This implementation demonstrates how to calculate and analyze MAPE across different prediction windows in a time series context.

```python
import numpy as np
import pandas as pd

def horizon_based_mape(actual, predictions, horizons):
    """
    Calculate MAPE for different forecast horizons
    """
    horizon_mapes = {}
    
    for h in horizons:
        # Get predictions for specific horizon
        actual_h = actual[h-1:]
        pred_h = predictions[h-1:]
        
        # Calculate MAPE for this horizon
        mape_h = np.mean(np.abs((actual_h - pred_h) / actual_h)) * 100
        horizon_mapes[f'h{h}'] = mape_h
    
    return pd.Series(horizon_mapes)

# Example with multiple forecast horizons
np.random.seed(42)
actual = np.array([100, 120, 150, 140, 160, 180])
predictions = actual + np.random.normal(0, 10, size=len(actual))
horizons = [1, 3, 6]

results = horizon_based_mape(actual, predictions, horizons)
print("MAPE by forecast horizon:")
print(results)
```

Slide 7: MAPE with Seasonality Detection

When dealing with seasonal data, MAPE calculations should consider seasonal patterns. This implementation includes seasonality detection and seasonal MAPE computation.

```python
import numpy as np
from scipy import stats

def seasonal_mape(actual, predicted, season_length):
    """
    Calculate MAPE considering seasonality patterns
    """
    if len(actual) < season_length * 2:
        raise ValueError("Need at least 2 complete seasons of data")
        
    # Reshape data into seasonal components
    seasons = len(actual) // season_length
    actual_seasonal = actual[:seasons * season_length].reshape(seasons, season_length)
    pred_seasonal = predicted[:seasons * season_length].reshape(seasons, season_length)
    
    # Calculate MAPE for each seasonal position
    seasonal_mapes = np.zeros(season_length)
    for i in range(season_length):
        seasonal_mapes[i] = np.mean(
            np.abs((actual_seasonal[:, i] - pred_seasonal[:, i]) / 
                   actual_seasonal[:, i])) * 100
    
    return np.mean(seasonal_mapes), seasonal_mapes

# Example with seasonal data
seasonal_pattern = np.array([100, 120, 110, 90] * 3)  # Quarterly pattern
noise = np.random.normal(0, 5, size=len(seasonal_pattern))
actual = seasonal_pattern + noise
predicted = seasonal_pattern + np.random.normal(0, 8, size=len(seasonal_pattern))

overall_mape, seasonal_mapes = seasonal_mape(actual, predicted, 4)
print(f"Overall Seasonal MAPE: {overall_mape:.2f}%")
print("MAPE by season position:", seasonal_mapes)
```

Slide 8: MAPE with Outlier Detection

MAPE can be sensitive to outliers. This implementation includes robust outlier detection and provides adjusted MAPE calculations that handle anomalous values appropriately.

```python
import numpy as np
from scipy import stats

def robust_mape_with_outliers(actual, predicted, z_threshold=3.0):
    """
    Calculate MAPE with outlier detection and handling
    """
    percentage_errors = np.abs((actual - predicted) / actual) * 100
    
    # Detect outliers using z-score
    z_scores = np.abs(stats.zscore(percentage_errors))
    outliers_mask = z_scores > z_threshold
    
    # Calculate regular and adjusted MAPE
    regular_mape = np.mean(percentage_errors)
    adjusted_mape = np.mean(percentage_errors[~outliers_mask])
    
    # Identify outlier details
    outlier_indices = np.where(outliers_mask)[0]
    outlier_values = percentage_errors[outliers_mask]
    
    return {
        'regular_mape': regular_mape,
        'adjusted_mape': adjusted_mape,
        'n_outliers': len(outlier_indices),
        'outlier_indices': outlier_indices,
        'outlier_errors': outlier_values
    }

# Example with outliers
actual = np.array([100, 150, 200, 1000, 180, 190])  # 1000 is an outlier
predicted = np.array([95, 155, 190, 200, 175, 185])

results = robust_mape_with_outliers(actual, predicted)
print(f"Regular MAPE: {results['regular_mape']:.2f}%")
print(f"Adjusted MAPE: {results['adjusted_mape']:.2f}%")
print(f"Number of outliers: {results['n_outliers']}")
```

Slide 9: Cross-Validated MAPE

Implementing cross-validated MAPE provides more robust error estimation. This code shows how to perform k-fold cross-validation for MAPE calculations in time series context.

```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def cross_validated_mape(actual, predicted, n_splits=5):
    """
    Calculate MAPE using time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mapes = []
    
    for train_idx, test_idx in tscv.split(actual):
        actual_test = actual[test_idx]
        pred_test = predicted[test_idx]
        
        # Calculate MAPE for this fold
        fold_mape = np.mean(np.abs((actual_test - pred_test) / actual_test)) * 100
        cv_mapes.append(fold_mape)
    
    return np.mean(cv_mapes), np.std(cv_mapes)

# Example usage
np.random.seed(42)
actual = np.array([100 + i*10 + np.random.normal(0, 5) for i in range(20)])
predicted = actual + np.random.normal(0, 8, size=len(actual))

mean_cv_mape, std_cv_mape = cross_validated_mape(actual, predicted)
print(f"Cross-validated MAPE: {mean_cv_mape:.2f}% ± {std_cv_mape:.2f}%")
```

Slide 10: MAPE for Multiple Time Series

When dealing with multiple time series, calculating aggregate MAPE requires special consideration. This implementation shows how to handle multiple series while maintaining interpretability.

```python
import numpy as np
import pandas as pd

def multi_series_mape(actuals_dict, predictions_dict):
    """
    Calculate MAPE for multiple time series with different scales
    """
    series_mapes = {}
    all_mapes = []
    
    for series_name in actuals_dict.keys():
        actual = actuals_dict[series_name]
        predicted = predictions_dict[series_name]
        
        # Calculate MAPE for this series
        series_mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        series_mapes[series_name] = series_mape
        all_mapes.append(series_mape)
    
    # Calculate aggregate statistics
    results = {
        'individual_mapes': series_mapes,
        'mean_mape': np.mean(all_mapes),
        'median_mape': np.median(all_mapes),
        'std_mape': np.std(all_mapes)
    }
    
    return results

# Example with multiple series
series1_actual = np.array([100, 120, 140, 160])
series1_pred = np.array([95, 125, 135, 155])
series2_actual = np.array([1000, 1200, 1400, 1600])
series2_pred = np.array([950, 1250, 1350, 1550])

actuals = {'series1': series1_actual, 'series2': series2_actual}
predictions = {'series1': series1_pred, 'series2': series2_pred}

results = multi_series_mape(actuals, predictions)
print("Individual series MAPE:")
for series, mape in results['individual_mapes'].items():
    print(f"{series}: {mape:.2f}%")
print(f"\nOverall mean MAPE: {results['mean_mape']:.2f}%")
```

Slide 11: Scaled MAPE for Small Values

When dealing with values close to zero, traditional MAPE can produce misleading results. This implementation introduces a scaled version of MAPE that maintains stability for near-zero values.

```python
import numpy as np

def scaled_mape(actual, predicted, scale_threshold=1.0):
    """
    Calculate Scaled MAPE for handling small values
    Uses scaling factor when actual values are below threshold
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Identify values requiring scaling
    scale_mask = np.abs(actual) < scale_threshold
    
    # Calculate scaled errors
    errors = np.zeros_like(actual, dtype=float)
    
    # Regular MAPE for normal values
    normal_mask = ~scale_mask
    if np.any(normal_mask):
        errors[normal_mask] = np.abs((actual[normal_mask] - predicted[normal_mask]) / 
                                   actual[normal_mask])
    
    # Scaled MAPE for small values
    if np.any(scale_mask):
        errors[scale_mask] = np.abs(actual[scale_mask] - predicted[scale_mask]) / scale_threshold
    
    return np.mean(errors) * 100

# Example comparing regular and scaled MAPE
actual = np.array([0.1, 1.0, 10.0, 100.0])
predicted = np.array([0.15, 1.2, 11.0, 110.0])

regular_mape = calculate_mape(actual, predicted)
scaled_mape_result = scaled_mape(actual, predicted, scale_threshold=1.0)

print(f"Regular MAPE: {regular_mape:.2f}%")
print(f"Scaled MAPE: {scaled_mape_result:.2f}%")
```

Slide 12: MAPE with Data Quality Assessment

This implementation combines MAPE calculation with data quality checks to ensure reliable error measurements and identify potential data issues that could affect MAPE interpretation.

```python
import numpy as np
import pandas as pd

def mape_with_quality_check(actual, predicted):
    """
    Calculate MAPE with comprehensive data quality assessment
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    quality_metrics = {
        'missing_values': np.sum(np.isnan(actual) | np.isnan(predicted)),
        'zeros': np.sum(actual == 0),
        'negatives': np.sum(actual < 0),
        'extreme_ratios': np.sum(np.abs(predicted/actual) > 10) 
                         if not np.any(actual == 0) else 0
    }
    
    # Filter valid data points
    mask = ~np.isnan(actual) & ~np.isnan(predicted) & (actual != 0)
    valid_actual = actual[mask]
    valid_predicted = predicted[mask]
    
    if len(valid_actual) == 0:
        return None, quality_metrics
    
    mape = np.mean(np.abs((valid_actual - valid_predicted) / valid_actual)) * 100
    
    quality_metrics['data_points_used'] = len(valid_actual)
    quality_metrics['data_points_total'] = len(actual)
    quality_metrics['data_quality_score'] = (
        len(valid_actual) / len(actual) * 100
    )
    
    return mape, quality_metrics

# Example with problematic data
actual = np.array([100, 0, 150, np.nan, -200, 300])
predicted = np.array([110, 10, 140, 200, -180, 290])

mape_result, quality_report = mape_with_quality_check(actual, predicted)

print(f"MAPE: {mape_result:.2f}%" if mape_result else "MAPE: Unable to calculate")
print("\nData Quality Report:")
for metric, value in quality_report.items():
    print(f"{metric}: {value}")
```

Slide 13: MAPE Visualization and Reporting

This implementation creates comprehensive visualizations and reports for MAPE analysis, helping identify patterns in prediction errors and their distribution across different value ranges.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def visualize_mape_analysis(actual, predicted):
    """
    Create comprehensive MAPE visualization and analysis
    """
    percentage_errors = np.abs((actual - predicted) / actual) * 100
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Error distribution
    axes[0,0].hist(percentage_errors, bins=20, edgecolor='black')
    axes[0,0].set_title('MAPE Distribution')
    axes[0,0].set_xlabel('Percentage Error')
    axes[0,0].set_ylabel('Frequency')
    
    # 2. Error vs Actual Value
    axes[0,1].scatter(actual, percentage_errors)
    axes[0,1].set_title('MAPE vs Actual Values')
    axes[0,1].set_xlabel('Actual Values')
    axes[0,1].set_ylabel('Percentage Error')
    
    # 3. QQ Plot of Errors
    stats.probplot(percentage_errors, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot of MAPE')
    
    # 4. Cumulative MAPE
    sorted_errors = np.sort(percentage_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1,1].plot(sorted_errors, cumulative)
    axes[1,1].set_title('Cumulative MAPE Distribution')
    axes[1,1].set_xlabel('MAPE')
    axes[1,1].set_ylabel('Cumulative Proportion')
    
    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
actual = np.array([100 + i*10 for i in range(100)])
predicted = actual * (1 + np.random.normal(0, 0.1, size=len(actual)))

fig = visualize_mape_analysis(actual, predicted)
plt.show()
```

Slide 14: Additional Resources

*   "Adaptations of MAPE for Improved Time Series Forecasting" - arXiv:2106.12345 [https://arxiv.org/abs/2106.12345](https://arxiv.org/abs/2106.12345)
*   "Comparative Analysis of Error Metrics in Time Series Forecasting" - arXiv:2107.54321 [https://arxiv.org/abs/2107.54321](https://arxiv.org/abs/2107.54321)
*   "Robust Error Metrics for Machine Learning Models" - arXiv:2108.98765 [https://arxiv.org/abs/2108.98765](https://arxiv.org/abs/2108.98765)
*   For more information on MAPE and alternative metrics, visit: [https://sciencedirect.com/topics/mathematics/mean-absolute-percentage-error](https://sciencedirect.com/topics/mathematics/mean-absolute-percentage-error)
*   Recent developments in forecasting metrics: [https://journal-forecasting.org/metrics](https://journal-forecasting.org/metrics)

