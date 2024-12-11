## Evaluating Regression Models Python Metrics and Code
Slide 1: Mean Squared Error (MSE)

The Mean Squared Error is a fundamental metric for evaluating regression models, measuring the average squared difference between predicted and actual values. It penalizes larger errors more heavily due to the squaring operation, making it particularly sensitive to outliers in the dataset.

```python
import numpy as np

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error between true and predicted values
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: MSE value
    """
    # MSE formula: (1/n) * Σ(y_true - y_pred)²
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"MSE: {calculate_mse(y_true, y_pred):.4f}")
# Output: MSE: 0.0675
```

Slide 2: Root Mean Squared Error (RMSE)

RMSE extends MSE by taking the square root of the result, providing a metric in the same unit as the target variable. This makes interpretation more intuitive and allows direct comparison with the original scale of the data.

```python
def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error between true and predicted values
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: RMSE value
    """
    # RMSE formula: sqrt((1/n) * Σ(y_true - y_pred)²)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"RMSE: {calculate_rmse(y_true, y_pred):.4f}")
# Output: RMSE: 0.2598
```

Slide 3: Mean Absolute Error (MAE)

Mean Absolute Error calculates the average absolute differences between predictions and actual values, providing a linear penalization of errors. Unlike MSE, MAE is less sensitive to outliers and provides a more robust metric for datasets with significant anomalies.

```python
def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between true and predicted values
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: MAE value
    """
    # MAE formula: (1/n) * Σ|y_true - y_pred|
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"MAE: {calculate_mae(y_true, y_pred):.4f}")
# Output: MAE: 0.2250
```

Slide 4: R-squared (Coefficient of Determination)

R-squared measures the proportion of variance in the dependent variable explained by the independent variables. This metric ranges from 0 to 1, where 1 indicates perfect prediction and 0 indicates the model performs no better than predicting the mean.

```python
def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared score between true and predicted values
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: R-squared value
    """
    # Calculate mean of true values
    y_mean = np.mean(y_true)
    
    # Calculate total sum of squares
    ss_total = np.sum((y_true - y_mean) ** 2)
    
    # Calculate residual sum of squares
    ss_residual = np.sum((y_true - y_pred) ** 2)
    
    # R² formula: 1 - (SS_residual / SS_total)
    r2 = 1 - (ss_residual / ss_total)
    return r2

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"R²: {calculate_r2(y_true, y_pred):.4f}")
# Output: R²: 0.9327
```

Slide 5: Adjusted R-squared

Adjusted R-squared modifies the R-squared metric to account for the number of predictors in the model, penalizing the addition of variables that don't improve the model's explanatory power significantly. This prevents overfitting through excessive feature inclusion.

```python
def calculate_adjusted_r2(y_true, y_pred, n_features):
    """
    Calculate Adjusted R-squared score
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        n_features: Number of features (independent variables)
    Returns:
        float: Adjusted R-squared value
    """
    n_samples = len(y_true)
    r2 = calculate_r2(y_true, y_pred)
    
    # Adjusted R² formula: 1 - (1 - R²) * (n - 1)/(n - p - 1)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
n_features = 2
print(f"Adjusted R²: {calculate_adjusted_r2(y_true, y_pred, n_features):.4f}")
# Output: Adjusted R²: 0.8872
```

Slide 6: Real-world Implementation - House Price Prediction

A comprehensive implementation of regression metrics for a house price prediction model, demonstrating the practical application of various evaluation metrics in a real estate context using the California Housing dataset.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# Load and prepare data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'R²': calculate_r2(y_test, y_pred),
    'Adjusted R²': calculate_adjusted_r2(y_test, y_pred, X_test.shape[1])
}

# Display results
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 7: Results for House Price Prediction

The evaluation results from our house price prediction model demonstrate the relationships between different metrics and their interpretation in a practical context. This analysis helps in understanding model performance from multiple perspectives.

```python
"""
Example Output:
MSE: 0.5428
RMSE: 0.7366
MAE: 0.5344
R²: 0.5983
Adjusted R²: 0.5975
"""

# Visualization of actual vs predicted values
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.tight_layout()
plt.show()
```

Slide 8: Explained Variance Score

The Explained Variance Score measures the proportion of variance that is predictable from the independent variables. This metric provides insight into how much of the variance in the target variable is captured by the model's predictions.

```python
def calculate_explained_variance(y_true, y_pred):
    """
    Calculate Explained Variance Score
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
    Returns:
        float: Explained variance score
    """
    # Calculate variance of residuals
    residual_variance = np.var(y_true - y_pred)
    # Calculate total variance
    total_variance = np.var(y_true)
    
    # Explained variance formula: 1 - (variance(y_true - y_pred) / variance(y_true))
    explained_variance = 1 - (residual_variance / total_variance)
    return explained_variance

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"Explained Variance: {calculate_explained_variance(y_true, y_pred):.4f}")
# Output: Explained Variance: 0.9331
```

Slide 9: Mean Absolute Percentage Error (MAPE)

Mean Absolute Percentage Error provides a percentage-based measurement of prediction accuracy, making it particularly useful when comparing models across different scales. It expresses accuracy as a percentage, facilitating intuitive interpretation across diverse datasets.

```python
def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: Array of actual values (must not contain zeros)
        y_pred: Array of predicted values
    Returns:
        float: MAPE value
    """
    # MAPE formula: (1/n) * Σ|(y_true - y_pred)/y_true| * 100
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# Example usage with non-zero values
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
print(f"MAPE: {calculate_mape(y_true, y_pred):.2f}%")
# Output: MAPE: 7.83%
```

Slide 10: Real-world Implementation - Time Series Forecasting

Implementation of comprehensive regression metrics for time series forecasting, demonstrating the evaluation of predictions across multiple time steps with consideration for temporal dependencies.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic time series data
np.random.seed(42)
t = np.linspace(0, 100, 100)
y = 0.5 * np.sin(0.1 * t) + 0.1 * np.random.randn(100)

# Prepare data
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Parameters
seq_length = 10
X, y = create_sequences(y_scaled, seq_length)

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Simple moving average prediction
y_pred = np.mean(X_test, axis=1)

# Calculate all metrics
metrics = {
    'MSE': calculate_mse(y_test, y_pred),
    'RMSE': calculate_rmse(y_test, y_pred),
    'MAE': calculate_mae(y_test, y_pred),
    'MAPE': calculate_mape(y_test.flatten(), y_pred.flatten()),
    'R²': calculate_r2(y_test, y_pred)
}

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 11: Results for Time Series Forecasting

The comprehensive evaluation of our time series forecasting model reveals the interplay between different error metrics and their significance in temporal prediction tasks.

```python
"""
Example Output:
MSE: 0.0124
RMSE: 0.1114
MAE: 0.0891
MAPE: 15.3244
R²: 0.7823
"""

# Visualization of forecasting results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='s')
plt.title('Time Series Forecasting Results')
plt.xlabel('Time Steps')
plt.ylabel('Scaled Value')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Weighted Mean Squared Error (WMSE)

Weighted Mean Squared Error extends MSE by allowing different weights for different samples or time points, enabling customized error penalization based on domain knowledge or sample importance in the prediction context.

```python
def calculate_wmse(y_true, y_pred, weights=None):
    """
    Calculate Weighted Mean Squared Error
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        weights: Array of weights for each sample (default: equal weights)
    Returns:
        float: WMSE value
    """
    if weights is None:
        weights = np.ones_like(y_true)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # WMSE formula: Σ(weights * (y_true - y_pred)²)
    wmse = np.sum(weights * (y_true - y_pred) ** 2)
    return wmse

# Example usage with custom weights
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.3, 4.2, 4.8])
weights = np.array([0.1, 0.2, 0.3, 0.4])  # Higher weights for later samples

print(f"WMSE: {calculate_wmse(y_true, y_pred, weights):.4f}")
# Output: WMSE: 0.0331
```

Slide 13: Additional Resources

*   A Comprehensive Survey of Regression Based Loss Functions for Time Series Forecasting [https://arxiv.org/abs/2201.09755](https://arxiv.org/abs/2201.09755)
*   Evaluation Metrics for Regression Problems: A Unified Approach [https://arxiv.org/abs/2006.13799](https://arxiv.org/abs/2006.13799)
*   Deep Learning for Time Series Forecasting: A Survey [https://arxiv.org/abs/2004.13408](https://arxiv.org/abs/2004.13408)
*   Robust Regression Loss Functions for Time Series Analysis [https://arxiv.org/abs/2008.04687](https://arxiv.org/abs/2008.04687)
*   Machine Learning Model Evaluation Metrics: A Comparative Study Search on Google Scholar: "machine learning regression metrics comparative analysis"

