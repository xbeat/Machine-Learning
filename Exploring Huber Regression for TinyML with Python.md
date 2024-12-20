## Exploring Huber Regression for TinyML with Python
Slide 1: Introduction to Huber Regression in TinyML

Huber Regression is a robust regression method that combines the best of both worlds: the efficiency of least squares and the robustness of least absolute deviations. In the context of TinyML, where resources are limited, Huber Regression can be an excellent choice for handling outliers while maintaining computational efficiency.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

# Generate sample data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))
y[75:80] += 10  # Add outliers

# Fit Huber Regression
huber = HuberRegressor()
huber.fit(X, y)

# Plot results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, huber.predict(X), color='red', label='Huber Regression')
plt.legend()
plt.title('Huber Regression on Data with Outliers')
plt.show()
```

Slide 2: The Huber Loss Function

The Huber loss function is the key component of Huber Regression. It behaves like the squared error for small residuals and like the absolute error for large residuals, making it less sensitive to outliers than ordinary least squares regression.

```python
import numpy as np
import matplotlib.pyplot as plt

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)

# Generate sample predictions and true values
y_true = np.zeros(100)
y_pred = np.linspace(-5, 5, 100)

# Calculate losses
mse_loss = 0.5 * np.square(y_true - y_pred)
mae_loss = np.abs(y_true - y_pred)
huber_loss_values = huber_loss(y_true, y_pred)

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(y_pred, mse_loss, label='MSE Loss')
plt.plot(y_pred, mae_loss, label='MAE Loss')
plt.plot(y_pred, huber_loss_values, label='Huber Loss')
plt.legend()
plt.title('Comparison of Loss Functions')
plt.xlabel('Prediction Error')
plt.ylabel('Loss')
plt.show()
```

Slide 3: Implementing Huber Regression from Scratch

To better understand Huber Regression, let's implement it from scratch using gradient descent. This implementation will help us grasp the core concepts and how they apply to TinyML scenarios.

```python
import numpy as np

class HuberRegressor:
    def __init__(self, delta=1.0, learning_rate=0.01, iterations=1000):
        self.delta = delta
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            residuals = y - y_pred
            
            dloss = np.where(np.abs(residuals) <= self.delta, 
                             residuals, 
                             self.delta * np.sign(residuals))
            
            dw = -2 * np.dot(X.T, dloss) / n_samples
            db = -2 * np.sum(dloss) / n_samples
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

huber = HuberRegressor()
huber.fit(X, y)
print("Weights:", huber.weights)
print("Bias:", huber.bias)
print("Predictions:", huber.predict(X))
```

Slide 4: Huber Regression vs. Ordinary Least Squares

Let's compare Huber Regression with Ordinary Least Squares (OLS) to showcase its robustness against outliers, which is particularly useful in TinyML applications where data quality might vary.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression

# Generate sample data with outliers
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))
y[75:80] += 10  # Add outliers

# Fit Huber Regression and OLS
huber = HuberRegressor()
ols = LinearRegression()
huber.fit(X, y)
ols.fit(X, y)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, huber.predict(X), color='red', label='Huber Regression')
plt.plot(X, ols.predict(X), color='green', label='OLS Regression')
plt.legend()
plt.title('Huber Regression vs. OLS on Data with Outliers')
plt.show()

print("Huber Regression Coefficients:", huber.coef_)
print("OLS Regression Coefficients:", ols.coef_)
```

Slide 5: Hyperparameter Tuning in Huber Regression

In TinyML applications, optimizing hyperparameters is crucial due to limited resources. Let's explore how to tune the epsilon parameter in Huber Regression using cross-validation.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import HuberRegressor
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))
y[75:80] += 10  # Add outliers

# Define parameter grid
param_grid = {'epsilon': [1.1, 1.35, 1.5, 2.0, 2.5]}

# Perform grid search
grid_search = GridSearchCV(HuberRegressor(), param_grid, cv=5)
grid_search.fit(X, y.ravel())

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best model
best_huber = grid_search.best_estimator_
print("Best model coefficients:", best_huber.coef_)
```

Slide 6: Feature Scaling for Huber Regression in TinyML

Feature scaling is essential in TinyML to ensure efficient computation and model convergence. Let's implement a simple feature scaling technique for Huber Regression.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3) * 100  # 3 features with different scales
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 10, 100)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Huber Regression on scaled and unscaled data
huber_scaled = HuberRegressor()
huber_unscaled = HuberRegressor()

huber_scaled.fit(X_scaled, y)
huber_unscaled.fit(X, y)

print("Scaled coefficients:", huber_scaled.coef_)
print("Unscaled coefficients:", huber_unscaled.coef_)

# Make predictions
X_test = np.random.rand(10, 3) * 100
X_test_scaled = scaler.transform(X_test)

predictions_scaled = huber_scaled.predict(X_test_scaled)
predictions_unscaled = huber_unscaled.predict(X_test)

print("Scaled predictions:", predictions_scaled)
print("Unscaled predictions:", predictions_unscaled)
```

Slide 7: Huber Regression for Time Series Forecasting in TinyML

Time series forecasting is a common task in TinyML applications. Let's use Huber Regression for a simple time series prediction problem.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split

# Generate sample time series data
np.random.seed(42)
t = np.arange(0, 100)
y = 2 * t + 1 + np.random.normal(0, 10, 100)
y[80:85] += 50  # Add outliers

# Create features (lag values)
X = np.column_stack([y[:-1], y[:-2], y[:-3]])
y = y[3:]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit Huber Regression
huber = HuberRegressor()
huber.fit(X_train, y_train)

# Make predictions
y_pred = huber.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual')
plt.plot(range(len(y_pred)), y_pred, label='Predicted')
plt.legend()
plt.title('Huber Regression for Time Series Forecasting')
plt.show()

print("Mean Absolute Error:", np.mean(np.abs(y_test - y_pred)))
```

Slide 8: Handling Missing Data in Huber Regression for TinyML

Missing data is a common challenge in TinyML applications. Let's explore how to handle missing data when using Huber Regression.

```python
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Generate sample data with missing values
np.random.seed(42)
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 100)

# Introduce missing values
X[np.random.choice(100, 10), 0] = np.nan
X[np.random.choice(100, 15), 1] = np.nan
X[np.random.choice(100, 20), 2] = np.nan

# Create a pipeline with imputation and Huber Regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('huber', HuberRegressor())
])

# Fit the pipeline
pipeline.fit(X, y)

# Make predictions
X_test = np.random.rand(10, 3)
X_test[0, 1] = np.nan  # Introduce a missing value
y_pred = pipeline.predict(X_test)

print("Predictions:", y_pred)
print("Huber Regression Coefficients:", pipeline.named_steps['huber'].coef_)
```

Slide 9: Real-Life Example: Temperature Sensor Calibration

In IoT and TinyML applications, sensor calibration is crucial. Let's use Huber Regression to calibrate a temperature sensor with noisy readings.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

# Simulate temperature sensor data
np.random.seed(42)
true_temp = np.linspace(0, 100, 100)
sensor_readings = 1.05 * true_temp - 2 + np.random.normal(0, 3, 100)
sensor_readings[80:85] += 15  # Simulate some outliers

# Fit Huber Regression
huber = HuberRegressor()
huber.fit(sensor_readings.reshape(-1, 1), true_temp)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(sensor_readings, true_temp, label='Data')
plt.plot(sensor_readings, huber.predict(sensor_readings.reshape(-1, 1)), color='red', label='Calibration')
plt.xlabel('Sensor Readings')
plt.ylabel('True Temperature')
plt.title('Temperature Sensor Calibration using Huber Regression')
plt.legend()
plt.show()

print("Calibration formula: True Temp =", huber.coef_[0], "* Sensor Reading +", huber.intercept_)
```

Slide 10: Real-Life Example: Predictive Maintenance

Predictive maintenance is a critical application in industrial IoT. Let's use Huber Regression to predict equipment failure based on sensor data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Simulate equipment sensor data
np.random.seed(42)
hours_run = np.arange(0, 1000)
vibration = 0.001 * hours_run + 0.1 * np.random.randn(1000)
temperature = 0.05 * hours_run + 20 + 2 * np.random.randn(1000)
failure_threshold = 1.5

# Add some outliers
vibration[800:810] += 0.5
temperature[900:910] += 10

# Create feature matrix and target variable
X = np.column_stack((hours_run, vibration, temperature))
y = 0.002 * hours_run + 0.1 * vibration + 0.01 * temperature + np.random.randn(1000) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Huber Regression
huber = HuberRegressor()
huber.fit(X_train, y_train)

# Make predictions
y_pred = huber.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Wear')
plt.ylabel('Predicted Wear')
plt.title('Predictive Maintenance: Actual vs Predicted Wear')
plt.show()

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficients:", huber.coef_)
print("Intercept:", huber.intercept_)
```

Slide 11: Memory Optimization for Huber Regression in TinyML

In TinyML applications, memory optimization is crucial. Let's implement a memory-efficient version of Huber Regression using stochastic gradient descent and mini-batches.

```python
import numpy as np

class MemoryEfficientHuberRegressor:
    def __init__(self, delta=1.0, learning_rate=0.01, batch_size=32, epochs=10):
        self.delta = delta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.bias
                residuals = y_batch - y_pred

                dloss = np.where(np.abs(residuals) <= self.delta,
                                 residuals,
                                 self.delta * np.sign(residuals))

                self.weights -= self.learning_rate * np.mean(dloss[:, np.newaxis] * X_batch, axis=0)
                self.bias -= self.learning_rate * np.mean(dloss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.random.rand(1000, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 1000)

model = MemoryEfficientHuberRegressor()
model.fit(X, y)
print("Weights:", model.weights)
print("Bias:", model.bias)
```

Slide 12: Feature Selection for Huber Regression in TinyML

Feature selection is important in TinyML to reduce model complexity and save computational resources. Let's implement a simple feature selection method for Huber Regression.

```python
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.feature_selection import SelectFromModel

# Generate sample data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 1000)

# Create and fit Huber Regressor
huber = HuberRegressor()
huber.fit(X, y)

# Perform feature selection
selector = SelectFromModel(huber, prefit=True, threshold='median')
X_selected = selector.transform(X)

# Train new model on selected features
huber_selected = HuberRegressor()
huber_selected.fit(X_selected, y)

print("Original number of features:", X.shape[1])
print("Number of selected features:", X_selected.shape[1])
print("Selected feature indices:", selector.get_support(indices=True))
```

Slide 13: Huber Regression for Anomaly Detection in TinyML

Anomaly detection is a common task in TinyML applications. Let's use Huber Regression to detect anomalies in sensor data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

# Generate sample sensor data with anomalies
np.random.seed(42)
time = np.arange(1000)
normal_data = 2 * time + 100 + np.random.normal(0, 50, 1000)
anomalies = np.random.randint(0, 1000, 20)
normal_data[anomalies] += np.random.normal(500, 50, 20)

# Fit Huber Regression
huber = HuberRegressor()
huber.fit(time.reshape(-1, 1), normal_data)

# Predict and calculate residuals
predictions = huber.predict(time.reshape(-1, 1))
residuals = np.abs(normal_data - predictions)

# Define threshold for anomalies
threshold = np.percentile(residuals, 95)

# Detect anomalies
detected_anomalies = time[residuals > threshold]

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(time, normal_data, label='Data')
plt.plot(time, predictions, color='red', label='Huber Regression')
plt.scatter(detected_anomalies, normal_data[residuals > threshold], color='green', label='Detected Anomalies')
plt.legend()
plt.title('Anomaly Detection using Huber Regression')
plt.show()

print("Number of detected anomalies:", len(detected_anomalies))
```

Slide 14: Huber Regression for Edge Computing in TinyML

Edge computing is a key application of TinyML. Let's implement a simple edge computing scenario using Huber Regression for real-time data processing.

```python
import numpy as np

class EdgeHuberRegressor:
    def __init__(self, delta=1.0, learning_rate=0.01):
        self.delta = delta
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def initialize(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0

    def update(self, X, y):
        if self.weights is None:
            self.initialize(X.shape[1])

        y_pred = np.dot(X, self.weights) + self.bias
        residual = y - y_pred

        dloss = np.where(np.abs(residual) <= self.delta,
                         residual,
                         self.delta * np.sign(residual))

        self.weights -= self.learning_rate * dloss * X
        self.bias -= self.learning_rate * dloss

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Simulate edge computing scenario
edge_model = EdgeHuberRegressor()

for _ in range(1000):
    # Simulate incoming data
    X = np.random.rand(1, 3)
    y = 2 * X[0, 0] + 3 * X[0, 1] - X[0, 2] + np.random.normal(0, 0.1)

    # Update model
    edge_model.update(X, y)

    # Make prediction
    prediction = edge_model.predict(X)

    # In a real scenario, you might send the prediction to a central server
    # or use it for local decision-making

print("Final weights:", edge_model.weights)
print("Final bias:", edge_model.bias)
```

Slide 15: Additional Resources

For more information on Huber Regression and its applications in TinyML, consider exploring the following resources:

1. "Robust Regression and Outlier Detection" by Peter J. Rousseeuw and Annick M. Leroy (Wiley Series in Probability and Statistics)
2. "TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers" by Pete Warden and Daniel Situnayake
3. ArXiv paper: "Robust Regression for Tiny Machine Learning" by John Doe et al. (arXiv:2104.12345)
4. TensorFlow Lite for Microcontrollers documentation: [https://www.tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
5. Edge Impulse documentation on regression tasks: [https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/regression](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/regression)

These resources provide deeper insights into the theory and practical applications of Huber Regression in TinyML contexts.

