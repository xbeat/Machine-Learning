## Confidence and Prediction Intervals in Data Science
Slide 1: Understanding Confidence and Prediction Intervals

Confidence and prediction intervals are statistical tools used to quantify uncertainty in estimates and predictions. They provide a range of values that are likely to contain the true population parameter or a future observation, respectively. These intervals are crucial for making informed decisions based on statistical models.

Slide 2: Source Code for Understanding Confidence and Prediction Intervals

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Fit linear regression model
coeffs = np.polyfit(X, y, 1)
line = np.poly1d(coeffs)

# Plot data and regression line
plt.scatter(X, y, alpha=0.5)
plt.plot(X, line(X), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Confidence and Prediction Intervals')
plt.show()
```

Slide 3: Confidence Intervals

Confidence intervals estimate the range where the true population parameter is likely to fall. For a linear regression model, the confidence interval represents the uncertainty in estimating the mean response at a given input value. It helps answer questions about the reliability of the estimated relationship between variables.

Slide 4: Source Code for Confidence Intervals

```python
from scipy import stats

# Calculate confidence intervals
def confidence_interval(x, y, prediction, confidence=0.95):
    n = len(x)
    mean_x = np.mean(x)
    std_err = np.sqrt(np.sum((y - prediction)**2) / (n-2))
    t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
    se = std_err * np.sqrt(1/n + (x - mean_x)**2 / np.sum((x - mean_x)**2))
    return t_value * se

# Calculate and plot confidence intervals
ci = confidence_interval(X, y, line(X))
plt.fill_between(X, line(X) - ci, line(X) + ci, color='red', alpha=0.1, label='Confidence Interval')
plt.legend()
plt.show()
```

Slide 5: Prediction Intervals

Prediction intervals estimate the range where a future individual observation is likely to fall. They account for both the uncertainty in estimating the mean response and the inherent variability of individual observations. Prediction intervals are wider than confidence intervals because they include additional uncertainty.

Slide 6: Source Code for Prediction Intervals

```python
# Calculate prediction intervals
def prediction_interval(x, y, prediction, confidence=0.95):
    n = len(x)
    mean_x = np.mean(x)
    std_err = np.sqrt(np.sum((y - prediction)**2) / (n-2))
    t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
    se = std_err * np.sqrt(1 + 1/n + (x - mean_x)**2 / np.sum((x - mean_x)**2))
    return t_value * se

# Calculate and plot prediction intervals
pi = prediction_interval(X, y, line(X))
plt.fill_between(X, line(X) - pi, line(X) + pi, color='blue', alpha=0.1, label='Prediction Interval')
plt.legend()
plt.show()
```

Slide 7: Comparing Confidence and Prediction Intervals

Confidence intervals are narrower than prediction intervals because they only account for uncertainty in estimating the mean response. Prediction intervals are wider as they also include the variability of individual observations. Understanding this difference is crucial for proper interpretation and decision-making based on statistical models.

Slide 8: Source Code for Comparing Confidence and Prediction Intervals

```python
# Plot both confidence and prediction intervals
plt.scatter(X, y, alpha=0.5)
plt.plot(X, line(X), color='red', label='Regression Line')
plt.fill_between(X, line(X) - ci, line(X) + ci, color='red', alpha=0.1, label='Confidence Interval')
plt.fill_between(X, line(X) - pi, line(X) + pi, color='blue', alpha=0.1, label='Prediction Interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Confidence and Prediction Intervals')
plt.legend()
plt.show()
```

Slide 9: Interpreting Confidence Intervals

A 95% confidence interval means that if we repeated the sampling process many times and calculated the interval each time, about 95% of these intervals would contain the true population parameter. It does not mean there's a 95% probability that the true parameter is within a single calculated interval.

Slide 10: Source Code for Interpreting Confidence Intervals

```python
# Simulate multiple samples and calculate confidence intervals
num_simulations = 1000
contains_true_param = 0

true_intercept, true_slope = 1, 2
confidence_level = 0.95

for _ in range(num_simulations):
    X = np.random.uniform(0, 10, 100)
    y = true_intercept + true_slope * X + np.random.normal(0, 1, 100)
    
    coeffs = np.polyfit(X, y, 1)
    line = np.poly1d(coeffs)
    
    ci = confidence_interval(X, y, line(X), confidence_level)
    
    if (line(X) - ci <= true_intercept + true_slope * X).all() and \
       (line(X) + ci >= true_intercept + true_slope * X).all():
        contains_true_param += 1

print(f"Percentage of intervals containing true parameter: {contains_true_param / num_simulations:.2%}")
```

Slide 11: Interpreting Prediction Intervals

A 95% prediction interval means that we can expect 95% of future observations to fall within this range, given the current model. It accounts for both the uncertainty in the model's parameters and the inherent variability in individual observations.

Slide 12: Source Code for Interpreting Prediction Intervals

```python
# Simulate future observations and check if they fall within prediction intervals
num_simulations = 1000
within_interval = 0

for _ in range(num_simulations):
    X_new = np.random.uniform(0, 10, 1)[0]
    y_new = true_intercept + true_slope * X_new + np.random.normal(0, 1)
    
    y_pred = line(X_new)
    pi = prediction_interval(X, y, y_pred)
    
    if y_pred - pi <= y_new <= y_pred + pi:
        within_interval += 1

print(f"Percentage of new observations within prediction interval: {within_interval / num_simulations:.2%}")
```

Slide 13: Real-life Example: Weather Forecasting

In weather forecasting, confidence intervals can be used to estimate the range of average temperatures for a given day, while prediction intervals can estimate the range of actual temperatures that might occur. This information helps people plan their activities and dress appropriately.

Slide 14: Source Code for Weather Forecasting Example

```python
import random

def simulate_weather(days=30):
    avg_temp = 20  # Average temperature in Celsius
    temp_variation = 5  # Daily temperature variation
    forecast_error = 2  # Forecast error standard deviation
    
    actual_temps = [avg_temp + random.gauss(0, temp_variation) for _ in range(days)]
    forecasted_temps = [temp + random.gauss(0, forecast_error) for temp in actual_temps]
    
    return actual_temps, forecasted_temps

actual, forecast = simulate_weather()

plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual Temperature')
plt.plot(forecast, label='Forecasted Temperature')
plt.fill_between(range(30), 
                 [f - 1.96*2 for f in forecast], 
                 [f + 1.96*2 for f in forecast], 
                 alpha=0.2, label='95% Prediction Interval')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Weather Forecast with Prediction Intervals')
plt.legend()
plt.show()
```

Slide 15: Real-life Example: Quality Control in Manufacturing

In manufacturing, confidence intervals can be used to estimate the average weight of products, while prediction intervals can estimate the range of weights for individual products. This information is crucial for quality control and ensuring products meet specifications.

Slide 16: Source Code for Manufacturing Example

```python
def simulate_manufacturing(n_samples=100):
    target_weight = 500  # Target weight in grams
    process_variation = 5  # Standard deviation of the manufacturing process
    
    weights = [random.gauss(target_weight, process_variation) for _ in range(n_samples)]
    
    mean_weight = np.mean(weights)
    std_error = np.std(weights, ddof=1) / np.sqrt(n_samples)
    
    confidence_interval = stats.t.interval(0.95, n_samples-1, loc=mean_weight, scale=std_error)
    prediction_interval = stats.t.interval(0.95, n_samples-1, loc=mean_weight, scale=np.std(weights, ddof=1))
    
    return weights, confidence_interval, prediction_interval

weights, ci, pi = simulate_manufacturing()

plt.figure(figsize=(12, 6))
plt.hist(weights, bins=20, alpha=0.5, label='Product Weights')
plt.axvline(np.mean(weights), color='red', linestyle='dashed', label='Mean Weight')
plt.axvspan(ci[0], ci[1], alpha=0.2, color='green', label='95% Confidence Interval')
plt.axvspan(pi[0], pi[1], alpha=0.2, color='blue', label='95% Prediction Interval')
plt.xlabel('Weight (g)')
plt.ylabel('Frequency')
plt.title('Product Weight Distribution with Confidence and Prediction Intervals')
plt.legend()
plt.show()
```

Slide 17: Additional Resources

For more information on confidence and prediction intervals, consider the following resources:

1.  "Statistical Intervals: A Guide for Practitioners and Researchers" by W. Q. Meeker et al. (2017)
2.  "Introduction to Statistical Learning" by G. James et al. (2013)
3.  ArXiv paper: "Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife" by S. Wager et al. (2014) URL: [https://arxiv.org/abs/1311.4555](https://arxiv.org/abs/1311.4555)

