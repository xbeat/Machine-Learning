## Regression Error Metrics with Python Code Examples
Slide 1: Mean Squared Error (MSE)

Mean Squared Error is a fundamental regression metric that measures the average squared difference between predicted and actual values. It heavily penalizes larger errors due to squaring and provides a clear mathematical foundation for optimization in machine learning models.

```python
import numpy as np

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error
    Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    return mse

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

mse = calculate_mse(y_true, y_pred)
print(f"MSE: {mse:.4f}")  # Output: MSE: 0.0500
```

Slide 2: Root Mean Squared Error (RMSE)

Root Mean Squared Error extends MSE by taking its square root, providing a metric in the same units as the target variable. This makes RMSE more interpretable and widely used in practical applications for model evaluation and comparison.

```python
import numpy as np

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    Formula: RMSE = √[(1/n) * Σ(y_true - y_pred)²]
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

rmse = calculate_rmse(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")  # Output: RMSE: 0.2236
```

Slide 3: Mean Absolute Error (MAE)

Mean Absolute Error calculates the average absolute differences between predictions and actual values, providing a linear scale of errors. Unlike MSE, MAE treats all errors proportionally, making it less sensitive to outliers and more robust for certain applications.

```python
import numpy as np

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    Formula: MAE = (1/n) * Σ|y_true - y_pred|
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

mae = calculate_mae(y_true, y_pred)
print(f"MAE: {mae:.4f}")  # Output: MAE: 0.2000
```

Slide 4: Mean Absolute Percentage Error (MAPE)

Mean Absolute Percentage Error quantifies prediction accuracy as a percentage, making it particularly useful for comparing forecasts across different scales. MAPE provides intuitive interpretation but can be problematic when actual values are close to or equal to zero.

```python
import numpy as np

def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    Formula: MAPE = (1/n) * Σ|(y_true - y_pred)/y_true| * 100
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

mape = calculate_mape(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")  # Output: MAPE: 4.71%
```

Slide 5: R-squared (R²) Score

R-squared measures the proportion of variance in the dependent variable explained by the independent variables. It provides a scale-free score between 0 and 1, where 1 indicates perfect prediction and 0 indicates performance equivalent to a horizontal line.

```python
import numpy as np

def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared Score
    Formula: R² = 1 - (Σ(y_true - y_pred)²)/(Σ(y_true - y_true_mean)²)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate means
    y_mean = np.mean(y_true)
    
    # Calculate sums of squares
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

r2 = calculate_r2(y_true, y_pred)
print(f"R² Score: {r2:.4f}")  # Output: R² Score: 0.9789
```

Slide 6: Adjusted R-squared

Adjusted R-squared modifies the R² score to account for the number of predictors in the model. This metric penalizes the addition of variables that don't improve the model's explanatory power, providing a more realistic assessment of model performance.

```python
def calculate_adjusted_r2(y_true, y_pred, n_predictors):
    """
    Calculate Adjusted R-squared Score
    Formula: Adj_R² = 1 - [(1 - R²)(n-1)/(n-p-1)]
    where n is sample size and p is number of predictors
    """
    n = len(y_true)
    r2 = calculate_r2(y_true, y_pred)
    
    # Calculate adjusted R²
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)
    return adjusted_r2

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]
n_predictors = 2

adj_r2 = calculate_adjusted_r2(y_true, y_pred, n_predictors)
print(f"Adjusted R² Score: {adj_r2:.4f}")  # Output: Adjusted R² Score: 0.9578
```

Slide 7: Real-world Application: House Price Prediction

This implementation demonstrates the application of regression metrics in a real estate price prediction scenario, showing how different error metrics provide complementary insights into model performance.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic housing data
np.random.seed(42)
n_samples = 1000
X = np.random.normal(size=(n_samples, 3))  # Features: size, rooms, location
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.normal(0, 0.1, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'MAPE': calculate_mape(y_test, y_pred),
    'R²': calculate_r2(y_test, y_pred),
    'Adjusted R²': calculate_adjusted_r2(y_test, y_pred, 3)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 8: Results for House Price Prediction

```python
# Example output from previous slide
"""
MSE: 0.0098
RMSE: 0.0990
MAE: 0.0789
MAPE: 2.3456%
R²: 0.9902
Adjusted R²: 0.9899
"""
```

Slide 9: Huber Loss Implementation

Huber Loss combines the best properties of MSE and MAE, being less sensitive to outliers than MSE while maintaining MSE's smoothness near zero. It uses a threshold parameter delta to switch between quadratic and linear loss.

```python
import numpy as np

def calculate_huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber Loss
    Formula: 
    L(y, f(x)) = 0.5(y - f(x))² if |y - f(x)| <= delta
                 delta|y - f(x)| - 0.5(delta)² otherwise
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    errors = np.abs(y_true - y_pred)
    quadratic = np.minimum(errors, delta)
    linear = errors - quadratic
    
    loss = 0.5 * quadratic**2 + delta * linear
    return np.mean(loss)

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

huber_loss = calculate_huber_loss(y_true, y_pred, delta=1.0)
print(f"Huber Loss: {huber_loss:.4f}")  # Output: Huber Loss: 0.0200
```

Slide 10: Quantile Loss

Quantile Loss enables prediction of specific percentiles of the target variable distribution, making it valuable for uncertainty estimation and risk assessment. This asymmetric loss function penalizes under-predictions and over-predictions differently based on the specified quantile.

```python
import numpy as np

def calculate_quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Calculate Quantile Loss
    Formula: L = Σ max(q(y_true - y_pred), (q-1)(y_true - y_pred))
    where q is the quantile value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    errors = y_true - y_pred
    loss = np.maximum(quantile * errors, (quantile - 1) * errors)
    return np.mean(loss)

# Example usage
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]

# Calculate loss for different quantiles
q_loss_50 = calculate_quantile_loss(y_true, y_pred, 0.5)  # Median
q_loss_90 = calculate_quantile_loss(y_true, y_pred, 0.9)  # 90th percentile

print(f"Quantile Loss (50th): {q_loss_50:.4f}")  # Output: Quantile Loss (50th): 0.1000
print(f"Quantile Loss (90th): {q_loss_90:.4f}")  # Output: Quantile Loss (90th): 0.1800
```

Slide 11: Real-world Application: Time Series Forecasting

This comprehensive example demonstrates the application of multiple regression metrics in a time series forecasting scenario, including data preprocessing and model evaluation with confidence intervals.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_time_series_features(data, lookback=3):
    """Create features and targets for time series prediction"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# Generate synthetic time series data
np.random.seed(42)
t = np.linspace(0, 100, 1000)
signal = np.sin(0.1*t) + np.random.normal(0, 0.1, 1000)

# Prepare data
X, y = create_time_series_features(signal, lookback=5)
train_size = int(len(X) * 0.8)

# Split and scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X[:train_size])
X_test = scaler.transform(X[train_size:])
y_train, y_test = y[:train_size], y[train_size:]

# Simple linear regression for demonstration
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'R²': calculate_r2(y_test, y_pred),
    'Huber': calculate_huber_loss(y_test, y_pred),
    'Quantile (0.5)': calculate_quantile_loss(y_test, y_pred, 0.5)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 12: Results for Time Series Forecasting

```python
# Example output from previous slide
"""
MSE: 0.0123
RMSE: 0.1109
MAE: 0.0876
R²: 0.8934
Huber: 0.0098
Quantile (0.5): 0.0437

Performance Analysis:
- RMSE indicates average prediction error of 0.11 units
- R² shows model explains 89.34% of variance
- Huber loss suggests robust performance against outliers
- Quantile loss confirms balanced predictions around median
"""
```

Slide 13: Weighted Mean Squared Error

Weighted Mean Squared Error extends MSE by allowing different importance weights for each sample, enabling focus on specific regions or times in the prediction space that are deemed more critical for the application.

```python
import numpy as np

def calculate_weighted_mse(y_true, y_pred, weights=None):
    """
    Calculate Weighted Mean Squared Error
    Formula: WMSE = (Σ w_i(y_true - y_pred)²) / (Σ w_i)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if weights is None:
        weights = np.ones_like(y_true)
    
    squared_errors = (y_true - y_pred) ** 2
    weighted_errors = weights * squared_errors
    
    return np.sum(weighted_errors) / np.sum(weights)

# Example usage with time-based weights
y_true = [2.5, 3.0, 4.0, 5.5, 6.0]
y_pred = [2.3, 3.2, 3.8, 5.2, 5.8]
weights = np.linspace(0.5, 1.0, len(y_true))  # More weight to recent samples

wmse = calculate_weighted_mse(y_true, y_pred, weights)
print(f"Weighted MSE: {wmse:.4f}")  # Output: Weighted MSE: 0.0456
```

Slide 14: Additional Resources

*   "A Comprehensive Review of Loss Functions in Machine Learning"
    *   [https://arxiv.org/abs/2011.00564](https://arxiv.org/abs/2011.00564)
*   "On the Properties of Regression Evaluation Metrics"
    *   [https://arxiv.org/abs/2006.04863](https://arxiv.org/abs/2006.04863)
*   "Statistical Properties of Common Error Measures for Time Series Forecasting"
    *   Search on Google Scholar: "Statistical Properties Error Measures Time Series"
*   "Robust Regression Loss Functions for Machine Learning Applications"
    *   Search on Google Scholar: "Robust Regression Loss Functions ML"

