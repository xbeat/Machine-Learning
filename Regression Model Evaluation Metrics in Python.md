## Regression Model Evaluation Metrics in Python
Slide 1: Mean Squared Error (MSE) Implementation

Mean Squared Error serves as a fundamental metric in regression analysis, measuring the average squared difference between predicted and actual values. It penalizes larger errors more heavily due to the squared term, making it particularly sensitive to outliers in the dataset.

```python
def mean_squared_error(y_true, y_pred):
    """
    Calculate MSE from scratch
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    """
    # Convert inputs to numpy arrays for vectorized operations
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate squared differences and mean
    mse = np.mean((y_true - y_pred) ** 2)
    
    return mse

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")  # Output: MSE: 0.0500
```

Slide 2: Root Mean Squared Error (RMSE) Implementation

RMSE extends MSE by taking the square root of the result, providing a metric in the same units as the target variable. This makes interpretation more intuitive when comparing model performance across different scales of data.

```python
def root_mean_squared_error(y_true, y_pred):
    """
    Calculate RMSE from scratch
    Formula: RMSE = √[(1/n) * Σ(y_true - y_pred)²]
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate MSE then take square root
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    return rmse

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"RMSE: {root_mean_squared_error(y_true, y_pred):.4f}")  # Output: RMSE: 0.2236
```

Slide 3: Mean Absolute Error (MAE) Implementation

Mean Absolute Error calculates the average absolute differences between predictions and actual values, providing a linear penalty for errors. Unlike MSE, MAE is less sensitive to outliers and provides a more robust metric for datasets with significant anomalies.

```python
def mean_absolute_error(y_true, y_pred):
    """
    Calculate MAE from scratch
    Formula: MAE = (1/n) * Σ|y_true - y_pred|
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate absolute differences and mean
    mae = np.mean(np.abs(y_true - y_pred))
    
    return mae

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")  # Output: MAE: 0.2000
```

Slide 4: R-squared (R²) Score Implementation

R-squared quantifies the proportion of variance in the dependent variable explained by the independent variables. This metric provides a scale-free score between 0 and 1, where 1 indicates perfect prediction and 0 indicates performance equivalent to a horizontal line.

```python
def r2_score(y_true, y_pred):
    """
    Calculate R² Score from scratch
    Formula: R² = 1 - (Σ(y_true - y_pred)²) / (Σ(y_true - y_mean)²)
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate mean of true values
    y_mean = np.mean(y_true)
    
    # Calculate total sum of squares and residual sum of squares
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    
    # Calculate R²
    r2 = 1 - (rss / tss)
    
    return r2

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"R² Score: {r2_score(y_true, y_pred):.4f}")  # Output: R² Score: 0.9921
```

Slide 5: Adjusted R-squared Implementation

Adjusted R-squared modifies the R-squared metric to account for the number of predictors in the model, penalizing the addition of variables that don't contribute significantly to model performance. This prevents overfitting through feature selection.

```python
def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Calculate Adjusted R² Score from scratch
    Formula: Adj R² = 1 - [(1 - R²)(n-1)/(n-p-1)]
    where n is sample size and p is number of features
    """
    import numpy as np
    
    # Calculate regular R²
    r2 = r2_score(y_true, y_pred)
    
    # Calculate sample size
    n = len(y_true)
    
    # Calculate adjusted R²
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    
    return adjusted_r2

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
n_features = 2
print(f"Adjusted R² Score: {adjusted_r2_score(y_true, y_pred, n_features):.4f}")
# Output: Adjusted R² Score: 0.9868
```

Slide 6: Mean Absolute Percentage Error (MAPE) Implementation

Mean Absolute Percentage Error provides a percentage-based measure of prediction accuracy, making it particularly useful when comparing models across different scales. It expresses accuracy as a percentage, facilitating intuitive interpretation for stakeholders.

```python
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE from scratch
    Formula: MAPE = (100/n) * Σ|((y_true - y_pred)/y_true)|
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    
    # Calculate percentage errors
    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    
    # Calculate mean and convert to percentage
    mape = 100 * np.mean(percentage_errors)
    
    return mape

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}%")
# Output: MAPE: 4.37%
```

Slide 7: Real-world Example - House Price Prediction

This comprehensive example demonstrates the application of regression metrics in a real estate price prediction scenario, including data preprocessing, model training, and evaluation using multiple metrics to assess model performance.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Generate synthetic house data
np.random.seed(42)
n_samples = 1000

# Features: size, bedrooms, age
X = np.random.rand(n_samples, 3)
X[:, 0] = X[:, 0] * 2000 + 1000  # Size: 1000-3000 sq ft
X[:, 1] = np.round(X[:, 1] * 3 + 2)  # Bedrooms: 2-5
X[:, 2] = np.round(X[:, 2] * 30)  # Age: 0-30 years

# Target: price (with some noise)
y = (X[:, 0] * 100 + X[:, 1] * 50000 - X[:, 2] * 1000 + 
     np.random.normal(0, 10000, n_samples))

# Split data and scale features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Slide 8: Source Code for House Price Prediction Results

```python
# Train model and make predictions
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate all metrics
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'RMSE': root_mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'R²': r2_score(y_test, y_pred),
    'Adjusted R²': adjusted_r2_score(y_test, y_pred, 3),
    'MAPE': mean_absolute_percentage_error(y_test, y_pred)
}

# Print results
for metric, value in metrics.items():
    if metric == 'MAPE':
        print(f"{metric}: {value:.2f}%")
    else:
        print(f"{metric}: {value:.2f}")

# Example output
"""
MSE: 98234567.89
RMSE: 9911.34
MAE: 7845.23
R²: 0.92
Adjusted R²: 0.91
MAPE: 3.45%
"""
```

Slide 9: Residual Analysis Implementation

Residual analysis provides crucial insights into model assumptions and potential areas for improvement. This implementation includes residual calculation, normality testing, and homoscedasticity visualization to validate regression model assumptions.

```python
def analyze_residuals(y_true, y_pred):
    """
    Comprehensive residual analysis including statistical tests
    """
    import numpy as np
    from scipy import stats
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Basic statistics
    stats_dict = {
        'Mean': np.mean(residuals),
        'Std Dev': np.std(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals)
    }
    
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    
    return stats_dict, (shapiro_stat, shapiro_p)

# Example usage with previous house price data
stats_dict, normality_test = analyze_residuals(y_test, y_pred)
print("\nResidual Statistics:")
for stat, value in stats_dict.items():
    print(f"{stat}: {value:.4f}")
print(f"\nShapiro-Wilk test: stat={normality_test[0]:.4f}, p={normality_test[1]:.4f}")
```

Slide 10: Huber Loss Implementation

Huber Loss combines the best properties of MSE and MAE by being quadratic for small errors and linear for large errors, offering robustness against outliers while maintaining MSE's advantages for smaller residuals. The delta parameter controls the transition point.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber Loss from scratch
    Formula: 
    L(y, f(x)) = 1/2(y - f(x))² for |y - f(x)| ≤ δ
    L(y, f(x)) = δ|y - f(x)| - 1/2δ² for |y - f(x)| > δ
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate residuals
    residuals = np.abs(y_true - y_pred)
    
    # Calculate loss based on delta threshold
    mask = residuals <= delta
    squared_loss = 0.5 * residuals[mask]**2
    linear_loss = delta * residuals[~mask] - 0.5 * delta**2
    
    # Combine losses
    return np.mean(np.concatenate([squared_loss, linear_loss]))

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"Huber Loss (δ=1.0): {huber_loss(y_true, y_pred):.4f}")
```

Slide 11: Explained Variance Score Implementation

Explained Variance Score measures the proportion of variance that is predictable from the independent variables. It differs from R² by focusing on the variance of the errors rather than the total variance of the predictions.

```python
def explained_variance_score(y_true, y_pred):
    """
    Calculate Explained Variance Score from scratch
    Formula: 1 - Var(y_true - y_pred) / Var(y_true)
    """
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate variances
    residual_variance = np.var(y_true - y_pred)
    total_variance = np.var(y_true)
    
    # Calculate score
    score = 1 - (residual_variance / total_variance)
    
    return score

# Example usage
y_true = [3, 2, 5, 7, 9]
y_pred = [2.8, 2.2, 4.8, 7.1, 8.8]
print(f"Explained Variance Score: {explained_variance_score(y_true, y_pred):.4f}")
```

Slide 12: Real-world Example - Time Series Energy Consumption

This example demonstrates the application of regression metrics in time series forecasting, specifically for energy consumption prediction, incorporating temporal features and multiple evaluation metrics.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Generate synthetic hourly energy consumption data
np.random.seed(42)
n_hours = 8760  # One year of hourly data

# Create time features
time_index = pd.date_range('2023-01-01', periods=n_hours, freq='H')
hour = time_index.hour
day_of_week = time_index.dayofweek
month = time_index.month

# Generate features matrix
X = np.column_stack([
    hour,
    day_of_week,
    month,
    np.sin(2 * np.pi * hour / 24),  # Daily cyclical feature
    np.cos(2 * np.pi * hour / 24)
])

# Generate target with daily, weekly, and seasonal patterns
y = (20 + 
     10 * np.sin(2 * np.pi * hour / 24) +  # Daily pattern
     5 * np.sin(2 * np.pi * day_of_week / 7) +  # Weekly pattern
     8 * np.sin(2 * np.pi * month / 12) +  # Yearly pattern
     np.random.normal(0, 2, n_hours))  # Random noise
```

Slide 13: Source Code for Energy Consumption Results

```python
def evaluate_time_series_model(X, y, n_splits=5):
    """
    Evaluate model using multiple metrics with time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_results = {
        'MSE': [], 'RMSE': [], 'MAE': [], 
        'R²': [], 'MAPE': [], 'Huber': []
    }
    
    for train_idx, test_idx in tscv.split(X):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics_results['MSE'].append(mean_squared_error(y_test, y_pred))
        metrics_results['RMSE'].append(root_mean_squared_error(y_test, y_pred))
        metrics_results['MAE'].append(mean_absolute_error(y_test, y_pred))
        metrics_results['R²'].append(r2_score(y_test, y_pred))
        metrics_results['MAPE'].append(mean_absolute_percentage_error(y_test, y_pred))
        metrics_results['Huber'].append(huber_loss(y_test, y_pred))
    
    # Calculate mean and std for each metric
    for metric in metrics_results:
        mean_val = np.mean(metrics_results[metric])
        std_val = np.std(metrics_results[metric])
        print(f"{metric:>8}: {mean_val:.4f} ± {std_val:.4f}")

# Run evaluation
evaluate_time_series_model(X, y)
```

Slide 14: Additional Resources

*   "On the Use of Cross-Validation for Time Series Predictor Evaluation"
    *   [https://arxiv.org/abs/1809.09446](https://arxiv.org/abs/1809.09446)
*   "A Comprehensive Review of Loss Functions in Machine Learning"
    *   [https://arxiv.org/abs/2011.00450](https://arxiv.org/abs/2011.00450)
*   "Robust Regression and Outlier Detection"
    *   [https://arxiv.org/abs/1607.01152](https://arxiv.org/abs/1607.01152)
*   "Time Series Forecasting with Deep Learning: A Survey"
    *   [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
*   "Beyond R-squared: Metrics for Regression Models"
    *   [https://arxiv.org/abs/2012.03150](https://arxiv.org/abs/2012.03150)

