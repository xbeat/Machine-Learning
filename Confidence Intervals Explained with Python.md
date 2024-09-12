## Confidence Intervals Explained with Python
Slide 1: Understanding Confidence Intervals

Confidence intervals are a fundamental concept in statistics that provide a range of plausible values for an unknown population parameter. They help us quantify the uncertainty in our estimates and make inferences about populations based on sample data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
sample = np.random.normal(loc=10, scale=2, size=100)

# Calculate mean and standard error
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
standard_error = sample_std / np.sqrt(len(sample))

# Calculate 95% confidence interval
confidence_level = 0.95
degrees_freedom = len(sample) - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_value * standard_error
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Sample Mean: {sample_mean:.2f}")
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")

# Plot histogram with confidence interval
plt.hist(sample, bins=20, density=True, alpha=0.7)
plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(ci_lower, color='green', linestyle='dotted', linewidth=2, label='95% CI')
plt.axvline(ci_upper, color='green', linestyle='dotted', linewidth=2)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Sample Distribution with 95% Confidence Interval')
plt.legend()
plt.show()
```

Slide 2: Components of a Confidence Interval

A confidence interval consists of three main components: the point estimate, the margin of error, and the confidence level. The point estimate is typically the sample mean, while the margin of error represents the range of uncertainty around that estimate. The confidence level, usually expressed as a percentage, indicates the probability that the interval contains the true population parameter.

```python
import numpy as np
from scipy import stats

def calculate_confidence_interval(sample, confidence_level):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    sample_size = len(sample)
    
    standard_error = sample_std / np.sqrt(sample_size)
    degrees_freedom = sample_size - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error = t_value * standard_error
    
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return sample_mean, ci_lower, ci_upper, margin_of_error

# Example usage
sample_data = [10, 12, 9, 11, 13, 15, 8, 14, 10, 11]
confidence_level = 0.95

mean, lower, upper, moe = calculate_confidence_interval(sample_data, confidence_level)

print(f"Sample Mean: {mean:.2f}")
print(f"Margin of Error: {moe:.2f}")
print(f"{confidence_level*100}% Confidence Interval: ({lower:.2f}, {upper:.2f})")
```

Slide 3: Interpreting Confidence Intervals

Confidence intervals are often misinterpreted. A 95% confidence interval does not mean there's a 95% chance that the true population parameter lies within the interval. Instead, it means that if we were to repeat the sampling process many times and calculate the confidence interval each time, about 95% of these intervals would contain the true population parameter.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_confidence_intervals(population_mean, population_std, sample_size, num_simulations, confidence_level):
    contains_true_mean = 0
    
    for _ in range(num_simulations):
        sample = np.random.normal(population_mean, population_std, sample_size)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1) * (sample_std / np.sqrt(sample_size))
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        if ci_lower <= population_mean <= ci_upper:
            contains_true_mean += 1
    
    return contains_true_mean / num_simulations

# Simulation parameters
population_mean = 10
population_std = 2
sample_size = 30
num_simulations = 10000
confidence_level = 0.95

coverage_probability = simulate_confidence_intervals(population_mean, population_std, sample_size, num_simulations, confidence_level)

print(f"Simulated coverage probability: {coverage_probability:.4f}")
print(f"Expected coverage probability: {confidence_level:.4f}")
```

Slide 4: Factors Affecting Confidence Interval Width

The width of a confidence interval is influenced by several factors, including sample size, variability in the data, and the desired confidence level. Larger sample sizes and lower variability lead to narrower intervals, while higher confidence levels result in wider intervals.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ci_width(sample_size, sample_std, confidence_level):
    t_value = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1)
    margin_of_error = t_value * (sample_std / np.sqrt(sample_size))
    return 2 * margin_of_error

sample_sizes = np.arange(10, 201, 10)
std_dev = 2
confidence_levels = [0.90, 0.95, 0.99]

plt.figure(figsize=(10, 6))

for conf_level in confidence_levels:
    widths = [ci_width(n, std_dev, conf_level) for n in sample_sizes]
    plt.plot(sample_sizes, widths, label=f"{conf_level:.0%} Confidence Level")

plt.xlabel("Sample Size")
plt.ylabel("Confidence Interval Width")
plt.title("Confidence Interval Width vs. Sample Size")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: Calculating Confidence Intervals for Proportions

Confidence intervals can also be calculated for proportions, which is useful when dealing with categorical data. The formula for the margin of error in this case is slightly different, using the concept of standard error for proportions.

```python
import numpy as np
from scipy import stats

def proportion_confidence_interval(successes, total, confidence_level):
    p_hat = successes / total
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    standard_error = np.sqrt((p_hat * (1 - p_hat)) / total)
    margin_of_error = z_score * standard_error
    
    ci_lower = max(0, p_hat - margin_of_error)
    ci_upper = min(1, p_hat + margin_of_error)
    
    return p_hat, ci_lower, ci_upper

# Example: 60 successes out of 100 trials
successes = 60
total = 100
confidence_level = 0.95

p_hat, ci_lower, ci_upper = proportion_confidence_interval(successes, total, confidence_level)

print(f"Sample Proportion: {p_hat:.2f}")
print(f"{confidence_level*100}% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
```

Slide 6: Confidence Intervals for Small Samples

When dealing with small samples (typically n < 30), we use the t-distribution instead of the normal distribution to calculate confidence intervals. This accounts for the increased uncertainty in small samples.

```python
import numpy as np
from scipy import stats

def small_sample_ci(sample, confidence_level):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    sample_size = len(sample)
    
    degrees_freedom = sample_size - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error = t_value * (sample_std / np.sqrt(sample_size))
    
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return sample_mean, ci_lower, ci_upper

# Example with a small sample
small_sample = [22, 25, 17, 24, 16, 29, 20, 23, 19, 21]
confidence_level = 0.95

mean, lower, upper = small_sample_ci(small_sample, confidence_level)

print(f"Sample Mean: {mean:.2f}")
print(f"{confidence_level*100}% Confidence Interval: ({lower:.2f}, {upper:.2f})")
```

Slide 7: One-Sided Confidence Intervals

Sometimes we're interested in setting a lower or upper bound on a parameter, rather than both. In these cases, we use one-sided confidence intervals. These are particularly useful when we want to make claims about minimum or maximum values.

```python
import numpy as np
from scipy import stats

def one_sided_ci(sample, confidence_level, side='lower'):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    sample_size = len(sample)
    
    degrees_freedom = sample_size - 1
    
    if side == 'lower':
        t_value = stats.t.ppf(confidence_level, degrees_freedom)
        ci_bound = sample_mean - t_value * (sample_std / np.sqrt(sample_size))
        return sample_mean, ci_bound, np.inf
    elif side == 'upper':
        t_value = stats.t.ppf(1 - confidence_level, degrees_freedom)
        ci_bound = sample_mean + t_value * (sample_std / np.sqrt(sample_size))
        return sample_mean, -np.inf, ci_bound
    else:
        raise ValueError("Side must be 'lower' or 'upper'")

# Example usage
sample_data = [105, 98, 103, 110, 108, 100, 95, 102, 106, 109]
confidence_level = 0.95

mean, lower, _ = one_sided_ci(sample_data, confidence_level, side='lower')
print(f"Sample Mean: {mean:.2f}")
print(f"{confidence_level*100}% Lower Confidence Bound: {lower:.2f}")

mean, _, upper = one_sided_ci(sample_data, confidence_level, side='upper')
print(f"{confidence_level*100}% Upper Confidence Bound: {upper:.2f}")
```

Slide 8: Bootstrap Confidence Intervals

When the underlying distribution of the data is unknown or non-normal, we can use bootstrap methods to construct confidence intervals. This involves resampling the data with replacement to estimate the sampling distribution of the statistic of interest.

```python
import numpy as np
from scipy import stats

def bootstrap_ci(data, num_bootstrap_samples=10000, confidence_level=0.95):
    bootstrap_means = np.zeros(num_bootstrap_samples)
    
    for i in range(num_bootstrap_samples):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    
    ci_lower = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
    
    return np.mean(data), ci_lower, ci_upper

# Example usage
np.random.seed(42)
non_normal_data = np.random.exponential(scale=2, size=1000)

mean, lower, upper = bootstrap_ci(non_normal_data)

print(f"Sample Mean: {mean:.2f}")
print(f"95% Bootstrap Confidence Interval: ({lower:.2f}, {upper:.2f})")
```

Slide 9: Confidence Intervals for Difference in Means

When comparing two populations, we often want to estimate the difference between their means. Confidence intervals for the difference in means help us assess whether there's a significant difference between the two groups.

```python
import numpy as np
from scipy import stats

def ci_difference_in_means(sample1, sample2, confidence_level=0.95):
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    difference = mean1 - mean2
    
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_value = stats.t.ppf((1 + confidence_level) / 2, df)
    
    margin_of_error = t_value * pooled_se
    ci_lower = difference - margin_of_error
    ci_upper = difference + margin_of_error
    
    return difference, ci_lower, ci_upper

# Example: Comparing two groups
group1 = [25, 28, 22, 24, 26, 27, 23, 25, 29, 30]
group2 = [20, 22, 19, 21, 23, 18, 24, 20, 21, 22]

diff, lower, upper = ci_difference_in_means(group1, group2)

print(f"Difference in Means: {diff:.2f}")
print(f"95% Confidence Interval for Difference: ({lower:.2f}, {upper:.2f})")
```

Slide 10: Confidence Intervals in Regression Analysis

In regression analysis, confidence intervals can be calculated for the regression coefficients. These intervals provide a range of plausible values for the true population parameters and help assess the precision of our estimates.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

# Calculate confidence intervals for slope and intercept
n = len(X)
dof = n - 2
t_value = stats.t.ppf(0.975, dof)  # 95% CI

slope_se = std_err
intercept_se = std_err * np.sqrt(np.sum(X**2) / (n * np.sum((X - np.mean(X))**2)))

slope_ci = (slope - t_value * slope_se, slope + t_value * slope_se)
intercept_ci = (intercept - t_value * intercept_se, intercept + t_value * intercept_se)

# Plot the data and regression line with confidence intervals
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, intercept + slope * X, color='red', label='Regression Line')
plt.fill_between(X, 
                 (intercept_ci[0] + slope_ci[0] * X),
                 (intercept_ci[1] + slope_ci[1] * X),
                 alpha=0.2, color='red', label='95% CI')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with 95% Confidence Interval')
plt.legend()
plt.show()

print(f"Slope: {slope:.4f}, 95% CI: {slope_ci}")
print(f"Intercept: {intercept:.4f}, 95% CI: {intercept_ci}")
```

Slide 11: Real-Life Example: Estimating Average Daily Steps

A health researcher wants to estimate the average number of daily steps taken by adults in a city. They collect data from a sample of 100 participants using fitness trackers over a week.

```python
import numpy as np
from scipy import stats

# Simulated daily step counts for 100 participants
np.random.seed(42)
step_counts = np.random.normal(loc=8000, scale=2000, size=100)

# Calculate the confidence interval
sample_mean = np.mean(step_counts)
sample_std = np.std(step_counts, ddof=1)
sample_size = len(step_counts)

confidence_level = 0.95
degrees_freedom = sample_size - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_value * (sample_std / np.sqrt(sample_size))

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Sample Mean: {sample_mean:.0f} steps")
print(f"95% Confidence Interval: ({ci_lower:.0f}, {ci_upper:.0f}) steps")
```

Slide 12: Real-Life Example: Estimating Germination Rate

A botanist is studying the germination rate of a rare plant species. They plant 200 seeds and observe that 140 of them germinate successfully. They want to estimate the true germination rate with a confidence interval.

```python
import numpy as np
from scipy import stats

total_seeds = 200
germinated_seeds = 140

# Calculate the proportion and confidence interval
p_hat = germinated_seeds / total_seeds
confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)
standard_error = np.sqrt((p_hat * (1 - p_hat)) / total_seeds)
margin_of_error = z_score * standard_error

ci_lower = max(0, p_hat - margin_of_error)
ci_upper = min(1, p_hat + margin_of_error)

print(f"Sample Germination Rate: {p_hat:.2%}")
print(f"95% Confidence Interval: ({ci_lower:.2%}, {ci_upper:.2%})")
```

Slide 13: Confidence Intervals in Hypothesis Testing

Confidence intervals complement hypothesis testing by providing a range of plausible values for the parameter of interest. If the hypothesized value falls outside the confidence interval, we reject the null hypothesis at the corresponding significance level.

```python
import numpy as np
from scipy import stats

def perform_t_test(sample, hypothesized_mean, confidence_level=0.95):
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    sample_size = len(sample)
    
    t_statistic = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(sample_size))
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=sample_size-1))
    
    # Calculate confidence interval
    t_value = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1)
    margin_of_error = t_value * (sample_std / np.sqrt(sample_size))
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    return t_statistic, p_value, ci_lower, ci_upper

# Example: Testing if the average height of a plant species is 25 cm
plant_heights = [23.5, 25.2, 24.7, 26.1, 23.9, 25.8, 24.3, 25.5, 24.9, 26.3]
hypothesized_mean = 25

t_stat, p_val, ci_lower, ci_upper = perform_t_test(plant_heights, hypothesized_mean)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
print(f"Hypothesized mean {'is' if ci_lower <= hypothesized_mean <= ci_upper else 'is not'} in the confidence interval")
```

Slide 14: Limitations and Considerations

While confidence intervals are powerful tools, they have limitations. They assume random sampling, do not account for systematic errors or biases, and their interpretation can be counterintuitive. It's crucial to consider the context and assumptions when using and interpreting confidence intervals.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_ci_coverage(sample_size, num_simulations, confidence_level):
    true_mean = 0
    true_std = 1
    covered = 0
    
    for _ in range(num_simulations):
        sample = np.random.normal(true_mean, true_std, sample_size)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1) * (sample_std / np.sqrt(sample_size))
        ci_lower = sample_mean - margin_of_error
        ci_upper = sample_mean + margin_of_error
        
        if ci_lower <= true_mean <= ci_upper:
            covered += 1
    
    return covered / num_simulations

sample_sizes = [10, 30, 50, 100, 200, 500]
simulations = 10000
confidence_level = 0.95

coverage_rates = [simulate_ci_coverage(n, simulations, confidence_level) for n in sample_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, coverage_rates, marker='o')
plt.axhline(y=confidence_level, color='r', linestyle='--', label='Expected coverage')
plt.xlabel('Sample Size')
plt.ylabel('Actual Coverage Rate')
plt.title(f'CI Coverage Rate vs Sample Size ({confidence_level*100}% CI)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into confidence intervals and statistical inference, the following resources are recommended:

1. "Statistical Inference" by Casella and Berger - A comprehensive textbook on statistical theory and methods.
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani - Covers statistical learning techniques with applications in R.
3. "Confidence Intervals: From Intuition to Optimal Likelihood Methods" by Held and Bové (arXiv:1306.2237) - An in-depth exploration of confidence interval methods.
4. "Bootstrap Methods and Their Application" by Davison and Hinkley - A thorough treatment of bootstrap techniques for inference.
5. Online courses on platforms like Coursera, edX, or DataCamp that cover statistical inference and confidence intervals in various programming languages.

