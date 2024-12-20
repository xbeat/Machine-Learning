## Confidence and Prediction Intervals with Python
Slide 1: Understanding Confidence and Prediction Intervals

Confidence and prediction intervals are statistical tools used to quantify uncertainty in estimates and forecasts. They provide a range of values that likely contain the true population parameter or future observations. These intervals are crucial for making informed decisions based on data analysis and predictive modeling.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sample Data for Interval Demonstration')
plt.show()
```

Slide 2: Confidence Intervals: Definition and Purpose

A confidence interval is a range of values that is likely to contain the true population parameter with a certain level of confidence. It provides information about the precision of an estimate and helps researchers understand the reliability of their findings.

```python
import scipy.stats as stats

# Calculate mean and standard error
mean = np.mean(y)
se = stats.sem(y)

# Calculate 95% confidence interval
ci = stats.t.interval(alpha=0.95, df=len(y)-1, loc=mean, scale=se)

print(f"Sample mean: {mean:.2f}")
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

Slide 3: Interpreting Confidence Intervals

The interpretation of a confidence interval is often misunderstood. A 95% confidence interval does not mean there's a 95% probability that the true parameter lies within the interval. Instead, it means that if we were to repeat the sampling process many times and calculate the interval each time, about 95% of these intervals would contain the true parameter.

```python
# Simulate confidence interval coverage
n_simulations = 1000
n_samples = 100
true_mean = 5
coverage = 0

for _ in range(n_simulations):
    sample = np.random.normal(true_mean, 1, n_samples)
    sample_mean = np.mean(sample)
    sample_se = stats.sem(sample)
    ci = stats.t.interval(0.95, len(sample)-1, sample_mean, sample_se)
    if ci[0] <= true_mean <= ci[1]:
        coverage += 1

print(f"Observed coverage: {coverage/n_simulations:.2%}")
```

Slide 4: Factors Affecting Confidence Interval Width

The width of a confidence interval is influenced by several factors: sample size, variability in the data, and desired confidence level. Understanding these factors helps in designing studies and interpreting results.

```python
def ci_width(n, std_dev, confidence):
    z = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z * (std_dev / np.sqrt(n))
    return 2 * margin_of_error

sample_sizes = [10, 50, 100, 500, 1000]
std_dev = 1
confidence = 0.95

widths = [ci_width(n, std_dev, confidence) for n in sample_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, widths, marker='o')
plt.xlabel('Sample Size')
plt.ylabel('Confidence Interval Width')
plt.title('CI Width vs. Sample Size')
plt.xscale('log')
plt.show()
```

Slide 5: Prediction Intervals: Definition and Purpose

A prediction interval is a range of values that is likely to contain future observations with a certain level of confidence. Unlike confidence intervals, which estimate population parameters, prediction intervals account for both the uncertainty in the population parameters and the variability of individual observations.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fit a linear regression model
model = LinearRegression()
X = x.reshape(-1, 1)
model.fit(X, y)

# Calculate prediction interval
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
std_error = np.sqrt(mse)

# 95% prediction interval
z = stats.norm.ppf(0.975)
pi = np.column_stack((y_pred - z * std_error, y_pred + z * std_error))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.fill_between(x, pi[:, 0], pi[:, 1], alpha=0.2, color='gray', label='95% Prediction Interval')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Prediction Interval')
plt.legend()
plt.show()
```

Slide 6: Comparing Confidence and Prediction Intervals

While both confidence and prediction intervals provide information about uncertainty, they serve different purposes. Confidence intervals focus on estimating population parameters, while prediction intervals are used for forecasting individual observations.

```python
# Generate new data for comparison
np.random.seed(123)
x_new = np.linspace(0, 10, 100)
X_new = x_new.reshape(-1, 1)
y_new = model.predict(X_new)

# Calculate confidence interval for the mean response
se_mean = std_error / np.sqrt(len(x))
ci = np.column_stack((y_new - z * se_mean, y_new + z * se_mean))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label='Data')
plt.plot(x_new, y_new, color='red', label='Regression Line')
plt.fill_between(x_new, ci[:, 0], ci[:, 1], alpha=0.2, color='blue', label='95% Confidence Interval')
plt.fill_between(x_new, pi[:, 0], pi[:, 1], alpha=0.1, color='gray', label='95% Prediction Interval')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Confidence vs. Prediction Intervals')
plt.legend()
plt.show()
```

Slide 7: Bootstrap Method for Confidence Intervals

The bootstrap method is a powerful technique for estimating confidence intervals without making strong assumptions about the underlying distribution of the data. It involves resampling with replacement from the original dataset to create multiple bootstrap samples.

```python
def bootstrap_ci(data, statistic, n_bootstrap=1000, confidence=0.95):
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats[i] = statistic(bootstrap_sample)
    
    lower_percentile = (1 - confidence) / 2
    upper_percentile = 1 - lower_percentile
    return np.percentile(bootstrap_stats, [lower_percentile * 100, upper_percentile * 100])

# Example: Bootstrap CI for the mean
data = np.random.normal(5, 2, 100)
bootstrap_ci_result = bootstrap_ci(data, np.mean)

print(f"Bootstrap 95% CI for the mean: ({bootstrap_ci_result[0]:.2f}, {bootstrap_ci_result[1]:.2f})")
```

Slide 8: Confidence Intervals for Proportions

Confidence intervals can also be calculated for proportions, which is particularly useful in survey research and hypothesis testing for categorical data. The Wilson score interval is a popular method for calculating confidence intervals for proportions, especially with small sample sizes.

```python
def wilson_score_interval(p, n, z=1.96):
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    return (center - spread, center + spread)

# Example: 95% CI for a proportion
successes = 40
total = 100
p_hat = successes / total

ci = wilson_score_interval(p_hat, total)
print(f"Observed proportion: {p_hat:.2f}")
print(f"95% Wilson score interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

Slide 9: Real-Life Example: Quality Control in Manufacturing

In a manufacturing process, confidence intervals can be used to monitor the quality of produced items. For example, let's consider a production line of electronic components where we want to estimate the mean resistance of resistors.

```python
# Simulate resistor measurements
np.random.seed(42)
resistances = np.random.normal(100, 5, 50)  # 50 resistors, mean 100 ohms, std 5 ohms

mean_resistance = np.mean(resistances)
se_resistance = stats.sem(resistances)
ci_resistance = stats.t.interval(0.95, len(resistances)-1, mean_resistance, se_resistance)

print(f"Mean resistance: {mean_resistance:.2f} ohms")
print(f"95% CI for mean resistance: ({ci_resistance[0]:.2f}, {ci_resistance[1]:.2f}) ohms")

plt.figure(figsize=(10, 6))
plt.hist(resistances, bins=15, edgecolor='black')
plt.axvline(mean_resistance, color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(ci_resistance[0], color='green', linestyle='dashed', linewidth=2, label='95% CI')
plt.axvline(ci_resistance[1], color='green', linestyle='dashed', linewidth=2)
plt.xlabel('Resistance (ohms)')
plt.ylabel('Frequency')
plt.title('Distribution of Resistor Measurements')
plt.legend()
plt.show()
```

Slide 10: Real-Life Example: Weather Forecasting

Prediction intervals are commonly used in weather forecasting to provide a range of likely temperatures or precipitation amounts. This helps communicate the uncertainty in weather predictions to the public.

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Simulate daily temperature data for a year
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
temperatures = pd.Series(np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10 + 20 + np.random.normal(0, 2, len(dates)), index=dates)

# Fit ARIMA model
model = ARIMA(temperatures, order=(1, 1, 1))
results = model.fit()

# Make predictions with prediction intervals
forecast = results.get_forecast(steps=7)
mean_forecast = forecast.predicted_mean
prediction_intervals = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(temperatures.index, temperatures, label='Historical Temperatures')
plt.plot(mean_forecast.index, mean_forecast, color='red', label='Forecast')
plt.fill_between(mean_forecast.index, prediction_intervals.iloc[:, 0], prediction_intervals.iloc[:, 1], color='pink', alpha=0.3, label='95% Prediction Interval')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Forecast with Prediction Interval')
plt.legend()
plt.show()
```

Slide 11: Bayesian Credible Intervals

Bayesian credible intervals are the Bayesian analog to frequentist confidence intervals. They represent a range of values that contain the true parameter with a certain probability, given the observed data and prior beliefs. Credible intervals have a more intuitive interpretation than confidence intervals.

```python
import pymc3 as pm

# Generate sample data
np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, 100)

# Bayesian model
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
    
    trace = pm.sample(2000, return_inferencedata=False)

# Calculate credible interval
mu_samples = trace['mu']
credible_interval = np.percentile(mu_samples, [2.5, 97.5])

print(f"95% Credible Interval for mu: ({credible_interval[0]:.2f}, {credible_interval[1]:.2f})")

plt.figure(figsize=(10, 6))
pm.plot_posterior(trace, var_names=['mu'], credible_interval=0.95)
plt.title('Posterior Distribution and 95% Credible Interval for mu')
plt.show()
```

Slide 12: Limitations and Considerations

While confidence and prediction intervals are valuable tools, they have limitations. Assumptions about data distribution, sample representativeness, and model adequacy must be considered. It's crucial to interpret intervals in context and be aware of potential biases or violations of assumptions.

```python
# Demonstrate the impact of non-normality on confidence intervals
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
skewed_data = np.random.exponential(1, 1000)

def compare_intervals(data, name):
    mean = np.mean(data)
    se = stats.sem(data)
    ci_normal = stats.t.interval(0.95, len(data)-1, mean, se)
    ci_bootstrap = bootstrap_ci(data, np.mean)
    
    print(f"{name} data:")
    print(f"Normal theory CI: ({ci_normal[0]:.2f}, {ci_normal[1]:.2f})")
    print(f"Bootstrap CI: ({ci_bootstrap[0]:.2f}, {ci_bootstrap[1]:.2f})")

compare_intervals(normal_data, "Normal")
compare_intervals(skewed_data, "Skewed")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(normal_data, bins=30, edgecolor='black')
plt.title('Normal Data')
plt.subplot(1, 2, 2)
plt.hist(skewed_data, bins=30, edgecolor='black')
plt.title('Skewed Data')
plt.tight_layout()
plt.show()
```

Slide 13: Best Practices and Recommendations

To effectively use confidence and prediction intervals, consider the following best practices:

1. Clearly state the confidence level and interpretation.
2. Account for sample size and its impact on interval width.
3. Verify assumptions and use robust methods when necessary.
4. Employ appropriate visualization techniques to communicate uncertainty.
5. Combine intervals with other statistical tools for comprehensive analysis.
6. Regularly update and refine models as new data becomes available.

```python
# Example of visualizing confidence intervals for multiple groups
np.random.seed(42)
groups = ['A', 'B', 'C', 'D']
means = [10, 12, 9, 11]
std_devs = [2, 3, 1.5, 2.5]
sample_sizes = [50, 40, 60, 45]

data = [np.random.normal(mu, sigma, size) for mu, sigma, size in zip(means, std_devs, sample_sizes)]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (group, group_data) in enumerate(zip(groups, data)):
    mean = np.mean(group_data)
    ci = stats.t.interval(0.95, len(group_data)-1, mean, stats.sem(group_data))
    ax.errorbar(i, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], fmt='o', capsize=5, capthick=2)

ax.set_xlabel('Group')
ax.set_ylabel('Mean Value')
ax.set_title('Comparison of Group Means with 95% Confidence Intervals')
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(groups)
plt.show()
```

Slide 14: Reporting and Communicating Intervals

Effective communication of confidence and prediction intervals is crucial for informed decision-making. When reporting results, provide context, explain limitations, and use visual aids to enhance understanding. Avoid overstating certainty and encourage proper interpretation of intervals.

```python
def report_interval(point_estimate, interval, interval_type, confidence_level=0.95):
    print(f"Point estimate: {point_estimate:.2f}")
    print(f"{confidence_level*100:.0f}% {interval_type} Interval: ({interval[0]:.2f}, {interval[1]:.2f})")
    print(f"Interpretation: We are {confidence_level*100:.0f}% confident that the true value")
    print(f"lies between {interval[0]:.2f} and {interval[1]:.2f}.")

# Example usage
sample_mean = 10.5
ci = (9.8, 11.2)
report_interval(sample_mean, ci, "Confidence")

# Visualization
plt.figure(figsize=(10, 4))
plt.errorbar(sample_mean, 0, xerr=[[sample_mean-ci[0]], [ci[1]-sample_mean]], 
             fmt='o', capsize=5, capthick=2, color='blue')
plt.axvline(sample_mean, color='red', linestyle='--', label='Point Estimate')
plt.xlabel('Value')
plt.yticks([])
plt.title('Visualization of Confidence Interval')
plt.legend()
plt.show()
```

Slide 15: Additional Resources

For further exploration of confidence and prediction intervals, consider the following resources:

1. Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on p-values: Context, process, and purpose. The American Statistician, 70(2), 129-133. ArXiv: [https://arxiv.org/abs/1603.00505](https://arxiv.org/abs/1603.00505)
2. Morey, R. D., Hoekstra, R., Rouder, J. N., Lee, M. D., & Wagenmakers, E. J. (2016). The fallacy of placing confidence in confidence intervals. Psychonomic Bulletin & Review, 23(1), 103-123. ArXiv: [https://arxiv.org/abs/1407.5296](https://arxiv.org/abs/1407.5296)
3. Greenland, S., Senn, S. J., Rothman, K. J., Carlin, J. B., Poole, C., Goodman, S. N., & Altman, D. G. (2016). Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations. European Journal of Epidemiology, 31(4), 337-350. ArXiv: [https://arxiv.org/abs/1603.07532](https://arxiv.org/abs/1603.07532)

These papers provide in-depth discussions on the interpretation and use of confidence intervals in statistical analysis.

