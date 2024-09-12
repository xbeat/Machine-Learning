## Confidence Interval Explained with Python
Slide 1: Understanding Confidence Intervals

Confidence intervals provide a range of values that likely contain the true population parameter. They are crucial in statistical inference, helping us estimate unknown population parameters based on sample data. This concept is fundamental in data analysis, scientific research, and decision-making processes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
sample = np.random.normal(loc=100, scale=15, size=100)

# Calculate mean and standard error
sample_mean = np.mean(sample)
standard_error = np.std(sample, ddof=1) / np.sqrt(len(sample))

# Calculate 95% confidence interval
confidence_level = 0.95
degrees_freedom = len(sample) - 1
t_value = np.abs(np.random.standard_t(degrees_freedom, size=1000000))
margin_of_error = np.percentile(t_value, confidence_level * 100) * standard_error

lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")
```

Slide 2: Components of a Confidence Interval

A confidence interval consists of three main components: the point estimate, the margin of error, and the confidence level. The point estimate is typically the sample mean, while the margin of error accounts for sampling variability. The confidence level, often 95%, indicates the probability that the interval contains the true population parameter.

```python
import scipy.stats as stats

# Sample data
data = [23, 25, 28, 30, 32, 35, 37, 39]

# Calculate confidence interval
confidence_level = 0.95
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
sample_size = len(data)

margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=sample_size-1) * (sample_std / np.sqrt(sample_size))

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Point estimate: {sample_mean:.2f}")
print(f"Margin of error: {margin_of_error:.2f}")
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
```

Slide 3: Interpreting Confidence Intervals

Confidence intervals are often misinterpreted. A 95% confidence interval does not mean there's a 95% chance the true parameter lies within the interval. Instead, it means that if we repeated the sampling process many times and calculated the confidence interval each time, about 95% of these intervals would contain the true parameter.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_confidence_intervals(num_simulations, sample_size, population_mean, population_std):
    contains_true_mean = 0
    for _ in range(num_simulations):
        sample = np.random.normal(population_mean, population_std, sample_size)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        margin_of_error = 1.96 * (sample_std / np.sqrt(sample_size))
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error
        if lower_bound <= population_mean <= upper_bound:
            contains_true_mean += 1
    return contains_true_mean / num_simulations

num_simulations = 10000
sample_size = 30
population_mean = 100
population_std = 15

proportion = simulate_confidence_intervals(num_simulations, sample_size, population_mean, population_std)
print(f"Proportion of intervals containing the true mean: {proportion:.2f}")
```

Slide 4: Factors Affecting Confidence Interval Width

The width of a confidence interval is influenced by several factors: sample size, variability in the data, and desired confidence level. A larger sample size or lower variability leads to narrower intervals, while a higher confidence level results in wider intervals.

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_ci_width(sample_size, std_dev, confidence_level):
    z_score = np.abs(np.random.normal(0, 1, 1000000))
    critical_value = np.percentile(z_score, (1 + confidence_level) * 100 / 2)
    margin_of_error = critical_value * (std_dev / np.sqrt(sample_size))
    return 2 * margin_of_error

sample_sizes = range(10, 1001, 10)
std_dev = 15
confidence_levels = [0.90, 0.95, 0.99]

for conf_level in confidence_levels:
    widths = [calculate_ci_width(n, std_dev, conf_level) for n in sample_sizes]
    plt.plot(sample_sizes, widths, label=f"{conf_level*100}% Confidence")

plt.xlabel("Sample Size")
plt.ylabel("Confidence Interval Width")
plt.title("CI Width vs. Sample Size for Different Confidence Levels")
plt.legend()
plt.show()
```

Slide 5: Confidence Intervals for Proportions

Confidence intervals can also be calculated for proportions, which is useful when dealing with categorical data. The process is similar to that for means, but uses a different formula for the standard error.

```python
import numpy as np
import scipy.stats as stats

def proportion_ci(success, total, confidence_level):
    p_hat = success / total
    z = stats.norm.ppf((1 + confidence_level) / 2)
    standard_error = np.sqrt(p_hat * (1 - p_hat) / total)
    margin_of_error = z * standard_error
    lower_bound = max(0, p_hat - margin_of_error)
    upper_bound = min(1, p_hat + margin_of_error)
    return lower_bound, upper_bound

# Example: 40 successes out of 100 trials
success = 40
total = 100
confidence_level = 0.95

lower, upper = proportion_ci(success, total, confidence_level)
print(f"95% CI for proportion: ({lower:.4f}, {upper:.4f})")
```

Slide 6: Bootstrap Confidence Intervals

Bootstrap is a resampling technique used to estimate the sampling distribution of a statistic. It's particularly useful for calculating confidence intervals when the underlying distribution is unknown or complex.

```python
import numpy as np

def bootstrap_ci(data, num_bootstrap_samples=10000, confidence_level=0.95):
    bootstrap_means = np.zeros(num_bootstrap_samples)
    for i in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    return np.percentile(bootstrap_means, [lower_percentile * 100, upper_percentile * 100])

# Example data
data = [23, 25, 28, 30, 32, 35, 37, 39]

lower, upper = bootstrap_ci(data)
print(f"95% Bootstrap CI: ({lower:.2f}, {upper:.2f})")
```

Slide 7: One-sided Confidence Intervals

Sometimes, we're only interested in an upper or lower bound rather than a range. One-sided confidence intervals provide this information, which can be useful in certain scenarios such as quality control or safety standards.

```python
import scipy.stats as stats

def one_sided_ci(data, confidence_level, side='upper'):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    
    if side == 'upper':
        t_value = stats.t.ppf(confidence_level, df=sample_size-1)
    elif side == 'lower':
        t_value = stats.t.ppf(1 - confidence_level, df=sample_size-1)
    else:
        raise ValueError("Side must be 'upper' or 'lower'")
    
    margin_of_error = t_value * (sample_std / np.sqrt(sample_size))
    bound = sample_mean + margin_of_error if side == 'upper' else sample_mean - margin_of_error
    
    return bound

# Example data
data = [23, 25, 28, 30, 32, 35, 37, 39]

upper_bound = one_sided_ci(data, 0.95, 'upper')
lower_bound = one_sided_ci(data, 0.95, 'lower')

print(f"95% Upper bound: {upper_bound:.2f}")
print(f"95% Lower bound: {lower_bound:.2f}")
```

Slide 8: Confidence Intervals for Difference in Means

When comparing two groups, we often want to estimate the difference between their population means. A confidence interval for this difference helps us assess whether the groups are significantly different.

```python
import scipy.stats as stats

def ci_difference_in_means(group1, group2, confidence_level=0.95):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    difference = mean1 - mean2
    
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_value = stats.t.ppf((1 + confidence_level) / 2, df)
    
    margin_of_error = t_value * pooled_se
    lower = difference - margin_of_error
    upper = difference + margin_of_error
    
    return lower, upper

# Example data
group1 = [23, 25, 28, 30, 32]
group2 = [18, 21, 24, 27, 29]

lower, upper = ci_difference_in_means(group1, group2)
print(f"95% CI for difference in means: ({lower:.2f}, {upper:.2f})")
```

Slide 9: Confidence Intervals in Regression Analysis

In regression analysis, confidence intervals can be calculated for the regression coefficients. These intervals provide a range of plausible values for the true population parameters, helping us assess the reliability of our model.

```python
import numpy as np
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.normal(0, 0.5, (100, 1))

# Add constant term to X for intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print confidence intervals for coefficients
print(model.conf_int())

# Plot the regression line with confidence interval
import matplotlib.pyplot as plt

plt.scatter(X[:, 1], y)
plt.plot(X[:, 1], model.predict(X), color='red')
plt.fill_between(X[:, 1], 
                 model.get_prediction(X).conf_int()[:, 0], 
                 model.get_prediction(X).conf_int()[:, 1], 
                 color='pink', alpha=0.3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression Line with 95% Confidence Interval')
plt.show()
```

Slide 10: Non-parametric Confidence Intervals

When the underlying distribution of the data is unknown or non-normal, non-parametric methods for constructing confidence intervals can be useful. These methods make fewer assumptions about the data distribution.

```python
import numpy as np
from scipy import stats

def sign_test_ci(data, confidence_level=0.95):
    n = len(data)
    median = np.median(data)
    
    # Calculate the critical value
    k = stats.binom.ppf((1 + confidence_level) / 2, n, 0.5)
    
    # Sort the data
    sorted_data = np.sort(data)
    
    # Find the lower and upper bounds
    lower = sorted_data[int(n - k)]
    upper = sorted_data[int(k - 1)]
    
    return lower, upper

# Example data
data = [23, 25, 28, 30, 32, 35, 37, 39]

lower, upper = sign_test_ci(data)
print(f"95% Sign Test CI for median: ({lower:.2f}, {upper:.2f})")
```

Slide 11: Real-life Example: Estimating Average Height

Suppose we want to estimate the average height of adults in a city. We collect a random sample of 100 adults and calculate their average height. Using a confidence interval, we can provide a range of plausible values for the true population mean height.

```python
import numpy as np
import scipy.stats as stats

# Simulated height data (in cm)
np.random.seed(42)
heights = np.random.normal(170, 10, 100)

# Calculate the confidence interval
sample_mean = np.mean(heights)
sample_std = np.std(heights, ddof=1)
confidence_level = 0.95
sample_size = len(heights)

margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=sample_size-1) * (sample_std / np.sqrt(sample_size))

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Sample mean height: {sample_mean:.2f} cm")
print(f"95% Confidence Interval: ({ci_lower:.2f} cm, {ci_upper:.2f} cm)")

# Visualize the data and confidence interval
import matplotlib.pyplot as plt

plt.hist(heights, bins=20, edgecolor='black')
plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(ci_lower, color='green', linestyle='dotted', linewidth=2, label='95% CI')
plt.axvline(ci_upper, color='green', linestyle='dotted', linewidth=2)
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.title('Distribution of Heights with 95% Confidence Interval')
plt.legend()
plt.show()
```

Slide 12: Real-life Example: Quality Control in Manufacturing

In a manufacturing process, we want to ensure that the diameter of produced bolts falls within a specified range. We can use confidence intervals to estimate the true mean diameter and assess whether the process is meeting quality standards.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulated bolt diameter data (in mm)
np.random.seed(42)
diameters = np.random.normal(10, 0.1, 50)

# Calculate the confidence interval
sample_mean = np.mean(diameters)
sample_std = np.std(diameters, ddof=1)
confidence_level = 0.95
sample_size = len(diameters)

margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=sample_size-1) * (sample_std / np.sqrt(sample_size))

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Sample mean diameter: {sample_mean:.4f} mm")
print(f"95% Confidence Interval: ({ci_lower:.4f} mm, {ci_upper:.4f} mm)")

# Visualize the data and confidence interval
plt.hist(diameters, bins=15, edgecolor='black')
plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(ci_lower, color='green', linestyle='dotted', linewidth=2, label='95% CI')
plt.axvline(ci_upper, color='green', linestyle='dotted', linewidth=2)
plt.xlabel('Diameter (mm)')
plt.ylabel('Frequency')
plt.title('Distribution of Bolt Diameters with 95% Confidence Interval')
plt.legend()
plt.show()
```

Slide 13: Confidence Intervals for Paired Data

When dealing with paired data, such as before-and-after measurements, we can use a paired t-test and construct confidence intervals for the mean difference. This approach accounts for the dependency between paired observations.

```python
import numpy as np
import scipy.stats as stats

def paired_data_ci(before, after, confidence_level=0.95):
    differences = after - before
    n = len(differences)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
    margin_of_error = t_value * (std_diff / np.sqrt(n))
    
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error
    
    return mean_diff, ci_lower, ci_upper

# Example data: Weight loss program (weights in kg)
before = np.array([70, 75, 80, 85, 90])
after = np.array([68, 72, 77, 81, 86])

mean_diff, ci_lower, ci_upper = paired_data_ci(before, after)

print(f"Mean weight loss: {mean_diff:.2f} kg")
print(f"95% CI for weight loss: ({ci_lower:.2f} kg, {ci_upper:.2f} kg)")
```

Slide 14: Confidence Intervals in Hypothesis Testing

Confidence intervals complement hypothesis testing by providing a range of plausible values for the parameter of interest. If a hypothesized value falls outside the confidence interval, we can reject the null hypothesis at the corresponding significance level.

```python
import numpy as np
import scipy.stats as stats

def perform_t_test_with_ci(sample, hypothesized_mean, confidence_level=0.95):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    sample_size = len(sample)
    
    # Calculate t-statistic and p-value
    t_statistic = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(sample_size))
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=sample_size-1))
    
    # Calculate confidence interval
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=sample_size-1) * (sample_std / np.sqrt(sample_size))
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return t_statistic, p_value, ci_lower, ci_upper

# Example: Test if the mean is different from 100
sample = np.random.normal(102, 5, 30)
hypothesized_mean = 100

t_stat, p_val, ci_lower, ci_upper = perform_t_test_with_ci(sample, hypothesized_mean)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"Reject null hypothesis: {p_val < 0.05}")
print(f"Hypothesized mean outside CI: {hypothesized_mean < ci_lower or hypothesized_mean > ci_upper}")
```

Slide 15: Additional Resources

For those interested in delving deeper into confidence intervals and statistical inference, here are some recommended resources:

1. "Statistical Inference" by Casella and Berger - A comprehensive textbook covering statistical theory and methods.
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani - Provides a practical approach to statistical learning with applications in R.
3. "Bootstrap Methods and Their Application" by Davison and Hinkley - An in-depth exploration of bootstrap techniques for inference.
4. ArXiv.org articles:
   * "Confidence Intervals: From Tests of Statistical Significance to Confidence Intervals, Range Hypotheses and Substantial Effects" by Geoff Cumming and Sue Finch ([https://arxiv.org/abs/math/0409297](https://arxiv.org/abs/math/0409297))
   * "Why the Standard Error of Measurement is Vital to Understanding Assessment" by Leslie Rutkowski ([https://arxiv.org/abs/1612.08988](https://arxiv.org/abs/1612.08988))

These resources offer a mix of theoretical foundations and practical applications, suitable for readers looking to expand their understanding of confidence intervals and related statistical concepts.

