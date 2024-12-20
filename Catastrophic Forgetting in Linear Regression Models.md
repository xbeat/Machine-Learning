## Catastrophic Forgetting in Linear Regression Models
Slide 1:

Catastrophic Forgetting in Linear Regression

Catastrophic forgetting is a phenomenon where a machine learning model, including linear regression, drastically forgets previously learned information when trained on new data. This can lead to significant performance degradation on tasks it once performed well. Let's explore this concept through code and examples.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate initial data
X1 = np.random.rand(100, 1)
y1 = 2 * X1 + 1 + np.random.randn(100, 1) * 0.1

# Train initial model
w1, b1 = np.linalg.lstsq(np.hstack([X1, np.ones_like(X1)]), y1, rcond=None)[0]

# Plot initial data and model
plt.scatter(X1, y1, c='blue', label='Initial Data')
plt.plot(X1, w1*X1 + b1, c='red', label='Initial Model')
plt.legend()
plt.title('Initial Linear Regression Model')
plt.show()
```

Slide 2:

Understanding Linear Regression

Linear regression is a fundamental statistical method that models the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between these variables.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Print model parameters
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Make predictions
X_new = np.array([[6], [7]])
predictions = model.predict(X_new)
print(f"Predictions for X=6 and X=7: {predictions}")
```

Slide 3:

Introducing New Data

When new data is introduced, the model adjusts its parameters to fit this new information. In some cases, this adjustment can lead to catastrophic forgetting of previously learned patterns.

```python
# Generate new data with a different pattern
X2 = np.random.rand(100, 1) * 2 + 3
y2 = -3 * X2 + 10 + np.random.randn(100, 1) * 0.1

# Combine old and new data
X_combined = np.vstack([X1, X2])
y_combined = np.vstack([y1, y2])

# Train combined model
w_combined, b_combined = np.linalg.lstsq(np.hstack([X_combined, np.ones_like(X_combined)]), y_combined, rcond=None)[0]

# Plot combined data and model
plt.scatter(X1, y1, c='blue', label='Initial Data')
plt.scatter(X2, y2, c='green', label='New Data')
plt.plot(X_combined, w_combined*X_combined + b_combined, c='red', label='Combined Model')
plt.legend()
plt.title('Combined Linear Regression Model')
plt.show()
```

Slide 4:

Observing Catastrophic Forgetting

After training on the combined dataset, we can observe how the model's performance on the original data has changed. This change often results in a significant loss of accuracy for the initial task.

```python
# Evaluate performance on initial data
initial_mse = np.mean((y1 - (w_combined*X1 + b_combined))**2)
print(f"Mean Squared Error on Initial Data: {initial_mse:.4f}")

# Compare with original model's performance
original_mse = np.mean((y1 - (w1*X1 + b1))**2)
print(f"Original Mean Squared Error: {original_mse:.4f}")

# Calculate percentage increase in error
error_increase = (initial_mse - original_mse) / original_mse * 100
print(f"Percentage Increase in Error: {error_increase:.2f}%")
```

Slide 5:

Visualizing the Impact

To better understand the impact of catastrophic forgetting, let's visualize how the model's predictions change for the initial data before and after introducing new information.

```python
plt.figure(figsize=(12, 5))

# Before catastrophic forgetting
plt.subplot(1, 2, 1)
plt.scatter(X1, y1, c='blue', label='Initial Data')
plt.plot(X1, w1*X1 + b1, c='red', label='Initial Model')
plt.title('Before Catastrophic Forgetting')
plt.legend()

# After catastrophic forgetting
plt.subplot(1, 2, 2)
plt.scatter(X1, y1, c='blue', label='Initial Data')
plt.plot(X1, w_combined*X1 + b_combined, c='red', label='Combined Model')
plt.title('After Catastrophic Forgetting')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 6:

Factors Contributing to Catastrophic Forgetting

Several factors can influence the severity of catastrophic forgetting in linear regression:

1. Data distribution differences
2. Model complexity
3. Learning rate and optimization algorithm
4. Regularization techniques

Let's explore how data distribution differences affect forgetting:

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, slope, intercept, noise, x_range):
    X = np.random.uniform(*x_range, size=(n_samples, 1))
    y = slope * X + intercept + np.random.normal(0, noise, size=(n_samples, 1))
    return X, y

# Generate two datasets with different distributions
X1, y1 = generate_data(100, 2, 1, 0.5, (0, 5))
X2, y2 = generate_data(100, -3, 10, 0.5, (5, 10))

# Plot the datasets
plt.figure(figsize=(10, 5))
plt.scatter(X1, y1, label='Dataset 1')
plt.scatter(X2, y2, label='Dataset 2')
plt.title('Different Data Distributions')
plt.legend()
plt.show()
```

Slide 7:

Mitigating Catastrophic Forgetting

To address catastrophic forgetting, several techniques can be employed:

1. Regularization
2. Ensemble methods
3. Incremental learning
4. Rehearsal and pseudo-rehearsal

Let's implement a simple regularization technique to mitigate forgetting:

```python
from sklearn.linear_model import Ridge

# Create and fit the Ridge regression model
alpha = 1.0  # Regularization strength
model = Ridge(alpha=alpha)
model.fit(X_combined, y_combined)

# Evaluate performance on initial data
initial_mse_regularized = np.mean((y1 - model.predict(X1))**2)
print(f"Regularized Mean Squared Error on Initial Data: {initial_mse_regularized:.4f}")

# Compare with non-regularized model
print(f"Non-regularized Mean Squared Error: {initial_mse:.4f}")

# Calculate percentage decrease in error
error_decrease = (initial_mse - initial_mse_regularized) / initial_mse * 100
print(f"Percentage Decrease in Error: {error_decrease:.2f}%")
```

Slide 8:

Ensemble Methods

Ensemble methods combine multiple models to improve performance and mitigate catastrophic forgetting. Let's implement a simple ensemble approach:

```python
from sklearn.linear_model import LinearRegression

# Train separate models for each dataset
model1 = LinearRegression().fit(X1, y1)
model2 = LinearRegression().fit(X2, y2)

# Create an ensemble prediction function
def ensemble_predict(X):
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    return (pred1 + pred2) / 2

# Evaluate ensemble performance on initial data
ensemble_mse = np.mean((y1 - ensemble_predict(X1))**2)
print(f"Ensemble Mean Squared Error on Initial Data: {ensemble_mse:.4f}")

# Compare with single model performance
print(f"Single Model Mean Squared Error: {initial_mse:.4f}")

# Calculate percentage improvement
improvement = (initial_mse - ensemble_mse) / initial_mse * 100
print(f"Percentage Improvement: {improvement:.2f}%")
```

Slide 9:

Incremental Learning

Incremental learning allows the model to learn from new data without forgetting previous knowledge. Let's implement a simple incremental learning approach:

```python
class IncrementalLinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = None
        self.b = None
        self.lr = learning_rate

    def fit(self, X, y):
        if self.w is None:
            self.w = np.zeros((X.shape[1], 1))
            self.b = 0

        for _ in range(100):  # Number of iterations
            y_pred = self.predict(X)
            error = y - y_pred
            self.w += self.lr * X.T.dot(error) / X.shape[0]
            self.b += self.lr * np.mean(error)

    def predict(self, X):
        return X.dot(self.w) + self.b

# Train incrementally
inc_model = IncrementalLinearRegression()
inc_model.fit(X1, y1)
inc_model.fit(X2, y2)

# Evaluate incremental model performance on initial data
inc_mse = np.mean((y1 - inc_model.predict(X1))**2)
print(f"Incremental Model MSE on Initial Data: {inc_mse:.4f}")
print(f"Original Model MSE: {initial_mse:.4f}")
```

Slide 10:

Real-life Example: Climate Change Prediction

Consider a linear regression model used to predict global temperature changes. Initially trained on historical data, it may struggle when new data reflecting rapid climate change is introduced.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate historical temperature data (1900-1980)
years_historical = np.arange(1900, 1981).reshape(-1, 1)
temp_historical = 0.005 * (years_historical - 1900) + np.random.normal(0, 0.1, size=years_historical.shape)

# Generate recent temperature data (1981-2020) with accelerated warming
years_recent = np.arange(1981, 2021).reshape(-1, 1)
temp_recent = 0.02 * (years_recent - 1981) + 0.4 + np.random.normal(0, 0.1, size=years_recent.shape)

# Train model on historical data
model_historical = LinearRegression().fit(years_historical, temp_historical)

# Predict using historical model
years_all = np.arange(1900, 2021).reshape(-1, 1)
pred_historical = model_historical.predict(years_all)

# Train model on all data
years_all_data = np.vstack((years_historical, years_recent))
temp_all_data = np.vstack((temp_historical, temp_recent))
model_all = LinearRegression().fit(years_all_data, temp_all_data)
pred_all = model_all.predict(years_all)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(years_historical, temp_historical, label='Historical Data', alpha=0.5)
plt.scatter(years_recent, temp_recent, label='Recent Data', alpha=0.5)
plt.plot(years_all, pred_historical, label='Historical Model', color='red')
plt.plot(years_all, pred_all, label='Updated Model', color='green')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Climate Change Prediction: Impact of Catastrophic Forgetting')
plt.legend()
plt.show()
```

Slide 11:

Real-life Example: Traffic Flow Prediction

Consider a linear regression model used to predict traffic flow on a highway. The model may experience catastrophic forgetting when new patterns emerge due to changes in population or infrastructure.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate initial traffic data (pre-infrastructure change)
hours = np.arange(0, 24).reshape(-1, 1)
traffic_initial = 100 + 50 * np.sin(np.pi * hours / 12) + np.random.normal(0, 10, size=hours.shape)

# Generate new traffic data (post-infrastructure change)
traffic_new = 150 + 100 * np.sin(np.pi * (hours - 2) / 12) + np.random.normal(0, 15, size=hours.shape)

# Train initial model
model_initial = LinearRegression().fit(hours, traffic_initial)

# Train model on all data
hours_all = np.vstack((hours, hours))
traffic_all = np.vstack((traffic_initial, traffic_new))
model_all = LinearRegression().fit(hours_all, traffic_all)

# Make predictions
pred_initial = model_initial.predict(hours)
pred_all = model_all.predict(hours)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(hours, traffic_initial, label='Initial Data', alpha=0.5)
plt.scatter(hours, traffic_new, label='New Data', alpha=0.5)
plt.plot(hours, pred_initial, label='Initial Model', color='red')
plt.plot(hours, pred_all, label='Updated Model', color='green')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Flow (vehicles/hour)')
plt.title('Traffic Flow Prediction: Impact of Catastrophic Forgetting')
plt.legend()
plt.show()
```

Slide 12:

Evaluating the Impact of Catastrophic Forgetting

To quantify the impact of catastrophic forgetting, we can compare the model's performance on the initial dataset before and after training on new data. Let's use Mean Absolute Error (MAE) as our metric:

```python
from sklearn.metrics import mean_absolute_error

# Calculate MAE for initial model on initial data
mae_initial = mean_absolute_error(traffic_initial, model_initial.predict(hours))

# Calculate MAE for updated model on initial data
mae_updated = mean_absolute_error(traffic_initial, model_all.predict(hours))

print(f"MAE of initial model on initial data: {mae_initial:.2f}")
print(f"MAE of updated model on initial data: {mae_updated:.2f}")

# Calculate percentage increase in error
error_increase = (mae_updated - mae_initial) / mae_initial * 100
print(f"Percentage increase in error: {error_increase:.2f}%")

# Visualize error distribution
errors_initial = np.abs(traffic_initial - model_initial.predict(hours))
errors_updated = np.abs(traffic_initial - model_all.predict(hours))

plt.figure(figsize=(10, 6))
plt.hist(errors_initial, bins=20, alpha=0.5, label='Initial Model Errors')
plt.hist(errors_updated, bins=20, alpha=0.5, label='Updated Model Errors')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Error Distribution: Initial vs Updated Model')
plt.legend()
plt.show()
```

Strategies to Mitigate Catastrophic Forgetting

To address catastrophic forgetting in linear regression, consider the following strategies:

1. Regularization: Use techniques like L1 or L2 regularization to prevent overfitting to new data.
2. Ensemble Methods: Maintain multiple models, each trained on different subsets of data.
3. Incremental Learning: Update the model gradually with small batches of new data.
4. Data Augmentation: Generate synthetic data points that represent the initial distribution.
5. Periodic Retraining: Retrain the model on a balanced dataset that includes both old and new data.

Let's implement a simple data augmentation technique:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Original data
X_orig = np.array([[1], [2], [3], [4], [5]])
y_orig = np.array([2, 4, 5, 4, 5])

# Generate augmented data
X_aug = X_orig + np.random.normal(0, 0.1, X_orig.shape)
y_aug = y_orig + np.random.normal(0, 0.1, y_orig.shape)

# Combine original and augmented data
X_combined = np.vstack((X_orig, X_aug))
y_combined = np.hstack((y_orig, y_aug))

# Train model on combined data
model = LinearRegression().fit(X_combined, y_combined)

# Make predictions
X_test = np.array([[6], [7]])
predictions = model.predict(X_test)
print(f"Predictions for X=6 and X=7: {predictions}")
```

Slide 14:

Continuous Learning and Adaptation

In real-world scenarios, linear regression models often need to adapt to changing patterns over time. Implementing a sliding window approach can help the model stay current while mitigating catastrophic forgetting:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

class SlidingWindowRegression:
    def __init__(self, window_size):
        self.window_size = window_size
        self.model = LinearRegression()
        self.X_window = []
        self.y_window = []

    def update(self, X, y):
        self.X_window.extend(X)
        self.y_window.extend(y)
        
        if len(self.X_window) > self.window_size:
            self.X_window = self.X_window[-self.window_size:]
            self.y_window = self.y_window[-self.window_size:]
        
        self.model.fit(self.X_window, self.y_window)

    def predict(self, X):
        return self.model.predict(X)

# Example usage
sliding_model = SlidingWindowRegression(window_size=100)

# Simulating data stream
for i in range(1000):
    X = [[i]]
    y = [np.sin(i * 0.1) + np.random.normal(0, 0.1)]
    sliding_model.update(X, y)

    if i % 100 == 0:
        print(f"Prediction at step {i}: {sliding_model.predict([[i+1]])}")
```

Slide 15:

Additional Resources

For further exploration of catastrophic forgetting in machine learning and strategies to mitigate it, consider the following resources:

1. "Overcoming catastrophic forgetting in neural networks" by Kirkpatrick et al. (2017) ArXiv link: [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796)
2. "Continual Lifelong Learning with Neural Networks: A Review" by Parisi et al. (2019) ArXiv link: [https://arxiv.org/abs/1802.07569](https://arxiv.org/abs/1802.07569)
3. "Gradient Episodic Memory for Continual Learning" by Lopez-Paz and Ranzato (2017) ArXiv link: [https://arxiv.org/abs/1706.08840](https://arxiv.org/abs/1706.08840)

These papers provide in-depth discussions on catastrophic forgetting and propose various techniques to address this challenge in different machine learning contexts, including but not limited to linear regression.

