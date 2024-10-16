## Regression Models and Outlier Sensitivity
Slide 1: Understanding Regression and Outliers

Regression models are powerful tools for predicting numerical values, but they often struggle with outliers. Outliers are data points that significantly differ from other observations, potentially skewing the model's performance. Let's explore this concept with a simple linear regression example.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 2
y[80:85] += 30  # Add outliers

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Plot results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.legend()
plt.title('Linear Regression with Outliers')
plt.show()
```

Slide 2: The Impact of Outliers on Linear Regression

Linear regression aims to find the best-fitting line through data points by minimizing the sum of squared residuals. However, this approach is sensitive to outliers because squaring large residuals amplifies their impact on the model.

```python
# Calculate residuals
residuals = y - model.predict(X)

# Plot residuals
plt.scatter(X, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.ylabel('Residuals')
plt.show()

# Print mean squared error
print(f"Mean Squared Error: {np.mean(residuals**2):.2f}")
```

Slide 3: Mean Squared Error (MSE) and Its Limitations

The Mean Squared Error (MSE) is commonly used as the loss function in linear regression. While effective for many scenarios, MSE is particularly sensitive to outliers due to its quadratic nature.

```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Calculate MSE for our model
mse_value = mse(y, model.predict(X))
print(f"MSE: {mse_value:.2f}")

# Visualize MSE growth
errors = np.linspace(0, 10, 100)
mse_values = mse(np.zeros_like(errors), errors)

plt.plot(errors, mse_values)
plt.title('MSE Growth with Error Magnitude')
plt.xlabel('Error')
plt.ylabel('MSE')
plt.show()
```

Slide 4: Introducing Huber Loss

Huber loss addresses the outlier sensitivity problem by combining the best properties of MSE and Mean Absolute Error (MAE). It uses MSE for small residuals and a linear loss for large residuals, making it more robust against outliers.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    residuals = y_true - y_pred
    condition = np.abs(residuals) <= delta
    squared_loss = 0.5 * residuals**2
    linear_loss = delta * (np.abs(residuals) - 0.5 * delta)
    return np.where(condition, squared_loss, linear_loss)

# Compare MSE and Huber Loss
errors = np.linspace(-10, 10, 1000)
mse_values = 0.5 * errors**2
huber_values = huber_loss(errors, np.zeros_like(errors))

plt.plot(errors, mse_values, label='MSE')
plt.plot(errors, huber_values, label='Huber Loss')
plt.legend()
plt.title('MSE vs Huber Loss')
plt.xlabel('Error')
plt.ylabel('Loss')
plt.show()
```

Slide 5: Implementing Huber Regression

Huber Regression uses Huber loss as its objective function. While not directly available in scikit-learn, we can implement it using the HuberRegressor class, which combines L2 regularization with Huber loss.

```python
from sklearn.linear_model import HuberRegressor

# Fit Huber Regression
huber_model = HuberRegressor(epsilon=1.35)
huber_model.fit(X, y.ravel())

# Plot results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.plot(X, huber_model.predict(X), color='green', label='Huber Regression')
plt.legend()
plt.title('Linear vs Huber Regression')
plt.show()
```

Slide 6: Comparing Linear and Huber Regression Performance

Let's compare the performance of Linear Regression and Huber Regression on our dataset with outliers.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predictions
linear_pred = model.predict(X)
huber_pred = huber_model.predict(X)

# Compute metrics
linear_mse = mean_squared_error(y, linear_pred)
huber_mse = mean_squared_error(y, huber_pred)
linear_mae = mean_absolute_error(y, linear_pred)
huber_mae = mean_absolute_error(y, huber_pred)

print(f"Linear Regression - MSE: {linear_mse:.2f}, MAE: {linear_mae:.2f}")
print(f"Huber Regression - MSE: {huber_mse:.2f}, MAE: {huber_mae:.2f}")
```

Slide 7: Visualizing Residuals

Comparing residuals helps us understand how each model handles outliers.

```python
linear_residuals = y.ravel() - linear_pred.ravel()
huber_residuals = y.ravel() - huber_pred.ravel()

plt.scatter(X, linear_residuals, color='red', label='Linear Regression')
plt.scatter(X, huber_residuals, color='green', label='Huber Regression')
plt.axhline(y=0, color='blue', linestyle='--')
plt.legend()
plt.title('Residuals Comparison')
plt.ylabel('Residuals')
plt.show()
```

Slide 8: Real-Life Example: Temperature Prediction

Imagine we're predicting daily temperatures. Unusual weather events might introduce outliers that could skew traditional linear regression models.

```python
np.random.seed(42)
days = np.arange(1, 31)
temperatures = 20 + 0.5 * days + np.random.randn(30) * 2
temperatures[25:28] -= 15  # Sudden cold snap

plt.scatter(days, temperatures)
plt.title('Daily Temperatures with Cold Snap')
plt.xlabel('Day of Month')
plt.ylabel('Temperature (°C)')
plt.show()

# Fit and compare models
X = days.reshape(-1, 1)
y = temperatures.reshape(-1, 1)

linear_model = LinearRegression().fit(X, y)
huber_model = HuberRegressor().fit(X, y.ravel())

plt.scatter(days, temperatures, label='Actual')
plt.plot(days, linear_model.predict(X), color='red', label='Linear Regression')
plt.plot(days, huber_model.predict(X), color='green', label='Huber Regression')
plt.legend()
plt.title('Temperature Prediction: Linear vs Huber')
plt.xlabel('Day of Month')
plt.ylabel('Temperature (°C)')
plt.show()
```

Slide 9: Real-Life Example: Sensor Calibration

In industrial settings, sensor calibration often involves regression. Faulty sensors or environmental disturbances can introduce outliers.

```python
np.random.seed(42)
true_values = np.linspace(0, 100, 50)
measured_values = true_values + np.random.randn(50) * 5
measured_values[35:40] += 30  # Simulating sensor malfunction

plt.scatter(true_values, measured_values)
plt.title('Sensor Readings vs True Values')
plt.xlabel('True Value')
plt.ylabel('Measured Value')
plt.show()

# Fit and compare models
X = true_values.reshape(-1, 1)
y = measured_values.reshape(-1, 1)

linear_model = LinearRegression().fit(X, y)
huber_model = HuberRegressor().fit(X, y.ravel())

plt.scatter(true_values, measured_values, label='Data')
plt.plot(true_values, linear_model.predict(X), color='red', label='Linear Regression')
plt.plot(true_values, huber_model.predict(X), color='green', label='Huber Regression')
plt.plot(true_values, true_values, color='blue', linestyle='--', label='Ideal Calibration')
plt.legend()
plt.title('Sensor Calibration: Linear vs Huber')
plt.xlabel('True Value')
plt.ylabel('Measured Value')
plt.show()
```

Slide 10: Tuning Huber Regression: The Epsilon Parameter

The epsilon parameter in Huber Regression determines the threshold between squared and linear loss. Let's explore its impact.

```python
epsilons = [1.0, 1.35, 2.0, 3.0]
plt.figure(figsize=(12, 8))

for i, eps in enumerate(epsilons, 1):
    huber_model = HuberRegressor(epsilon=eps).fit(X, y.ravel())
    
    plt.subplot(2, 2, i)
    plt.scatter(X, y, label='Data')
    plt.plot(X, huber_model.predict(X), color='green', label=f'Huber (ε={eps})')
    plt.title(f'Huber Regression (ε={eps})')
    plt.legend()

plt.tight_layout()
plt.show()
```

Slide 11: Cross-Validation for Huber Regression

To find the optimal epsilon value, we can use cross-validation.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'epsilon': np.linspace(1.0, 3.0, 10)}
huber = HuberRegressor()
grid_search = GridSearchCV(huber, param_grid, cv=5)
grid_search.fit(X, y.ravel())

print(f"Best epsilon: {grid_search.best_params_['epsilon']:.2f}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Plot results
plt.scatter(X, y, label='Data')
plt.plot(X, grid_search.predict(X), color='green', label='Best Huber Model')
plt.legend()
plt.title('Optimal Huber Regression Model')
plt.show()
```

Slide 12: Limitations of Huber Regression

While Huber Regression is more robust than Linear Regression, it's not without limitations:

1. It's more computationally expensive than simple linear regression.
2. The epsilon parameter needs tuning, which can be time-consuming.
3. For extremely large outliers, even Huber Regression may struggle.

```python
# Simulate data with extreme outliers
X_extreme = np.linspace(0, 10, 100).reshape(-1, 1)
y_extreme = 2 * X_extreme + 1 + np.random.randn(100, 1) * 2
y_extreme[90:95] += 100  # Add extreme outliers

linear_model = LinearRegression().fit(X_extreme, y_extreme)
huber_model = HuberRegressor().fit(X_extreme, y_extreme.ravel())

plt.scatter(X_extreme, y_extreme, label='Data')
plt.plot(X_extreme, linear_model.predict(X_extreme), color='red', label='Linear')
plt.plot(X_extreme, huber_model.predict(X_extreme), color='green', label='Huber')
plt.legend()
plt.title('Performance with Extreme Outliers')
plt.ylim(-10, 110)
plt.show()
```

Slide 13: Alternative Robust Regression Techniques

While Huber Regression is effective, other robust regression techniques exist:

1. RANSAC (Random Sample Consensus)
2. Theil-Sen Regression
3. Quantile Regression

```python
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor

ransac = RANSACRegressor().fit(X, y.ravel())
theilsen = TheilSenRegressor().fit(X, y.ravel())

plt.scatter(X, y, label='Data')
plt.plot(X, linear_model.predict(X), color='red', label='Linear')
plt.plot(X, huber_model.predict(X), color='green', label='Huber')
plt.plot(X, ransac.predict(X), color='blue', label='RANSAC')
plt.plot(X, theilsen.predict(X), color='purple', label='Theil-Sen')
plt.legend()
plt.title('Comparison of Robust Regression Techniques')
plt.show()
```

Slide 14: Conclusion and Best Practices

Huber Regression offers a robust alternative to Linear Regression when dealing with outliers. To make the most of it:

1. Always visualize your data to understand the presence and impact of outliers.
2. Use cross-validation to tune the epsilon parameter.
3. Compare multiple robust regression techniques for your specific dataset.
4. Consider the trade-off between robustness and computational complexity.

```python
# Function to evaluate multiple models
def evaluate_models(X, y, models):
    for name, model in models.items():
        model.fit(X, y.ravel())
        mse = mean_squared_error(y, model.predict(X))
        print(f"{name} - MSE: {mse:.4f}")

models = {
    'Linear Regression': LinearRegression(),
    'Huber Regression': HuberRegressor(),
    'RANSAC': RANSACRegressor(),
    'Theil-Sen': TheilSenRegressor()
}

evaluate_models(X, y, models)
```

Slide 15: Additional Resources

For those interested in diving deeper into robust regression techniques and handling outliers, consider exploring these resources:

1. Robust Regression Methods in Machine Learning: A Survey (ArXiv:2007.04124)
2. Huber Regression: Robust Regression for Outliers (ArXiv:1811.03455)
3. Comparing Different Robust Linear Regression Methods (ArXiv:2007.01146)

These papers provide comprehensive overviews and comparisons of various robust regression techniques, including Huber Regression, and offer insights into their theoretical foundations and practical applications.

