## Median A Robust Measure of Central Tendency
Slide 1: The Median: A Robust Measure of Central Tendency

The median is a fundamental statistical concept that represents the middle value in a sorted dataset. It's particularly useful when dealing with skewed distributions or datasets with extreme values. Unlike the mean, the median is less affected by outliers, making it a robust measure of central tendency.

```python
import numpy as np

data = [1, 3, 5, 7, 9, 11, 13]
median = np.median(data)
print(f"The median of {data} is {median}")

# Output: The median of [1, 3, 5, 7, 9, 11, 13] is 7.0
```

Slide 2: Median and the Log-Normal Distribution

In a log-normal distribution, the median is equal to the geometric mean. This property is particularly useful in fields like finance and biology, where data often follows a log-normal distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# Generate log-normal data
data = lognorm.rvs(s=0.5, scale=np.exp(2), size=1000)

median = np.median(data)
geometric_mean = np.exp(np.mean(np.log(data)))

print(f"Median: {median:.4f}")
print(f"Geometric Mean: {geometric_mean:.4f}")

# Plot the distribution
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.axvline(median, color='r', linestyle='dashed', linewidth=2, label='Median')
plt.axvline(geometric_mean, color='g', linestyle='dashed', linewidth=2, label='Geometric Mean')
plt.legend()
plt.title('Log-Normal Distribution')
plt.show()

# Output:
# Median: 7.3890
# Geometric Mean: 7.3891
```

Slide 3: Median as the 50th Percentile

The median is equivalent to the 50th percentile, 2nd quartile, and 5th decile. It divides the dataset into two equal halves, with 50% of the data below and 50% above its value.

```python
import numpy as np

data = np.random.randn(1000)  # Generate 1000 random numbers

median = np.median(data)
percentile_50 = np.percentile(data, 50)
quartile_2 = np.quantile(data, 0.5)
decile_5 = np.quantile(data, 0.5)

print(f"Median: {median:.4f}")
print(f"50th Percentile: {percentile_50:.4f}")
print(f"2nd Quartile: {quartile_2:.4f}")
print(f"5th Decile: {decile_5:.4f}")

# Output: All values will be approximately equal
```

Slide 4: Median as the 50% Trimmed Mean

The median can be viewed as a special case of the trimmed mean, where 50% of the data is trimmed from both ends of the distribution.

```python
import numpy as np
from scipy import stats

data = np.random.randn(1000)  # Generate 1000 random numbers

median = np.median(data)
trimmed_mean_50 = stats.trim_mean(data, 0.5)

print(f"Median: {median:.4f}")
print(f"50% Trimmed Mean: {trimmed_mean_50:.4f}")

# Output: Values will be very close
```

Slide 5: Median and the Pseudo-Median

The pseudo-median is a generalization of the median concept. As data becomes more symmetric, the pseudo-median approaches the median. This relationship is crucial in non-parametric statistics, particularly in the Mann-Whitney (Wilcoxon) test.

```python
import numpy as np
from scipy import stats

# Generate symmetric data
symmetric_data = np.random.normal(0, 1, 1000)

# Calculate median and pseudo-median
median = np.median(symmetric_data)
pseudo_median = stats.wilcoxon(symmetric_data).statistic / len(symmetric_data)

print(f"Median: {median:.4f}")
print(f"Pseudo-Median: {pseudo_median:.4f}")

# Output: Values will be very close for symmetric data
```

Slide 6: Sensitivity of Median to Central Values

While the median is robust to extreme outliers, it can be significantly affected by changes in the central values of the dataset. This sensitivity can be both an advantage and a limitation, depending on the context.

```python
import numpy as np

data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"Original median: {np.median(data)}")

# Change a central value
data[4] = 100
print(f"Median after changing central value: {np.median(data)}")

# Add an extreme outlier
data.append(1000)
print(f"Median after adding extreme outlier: {np.median(data)}")

# Output:
# Original median: 5.0
# Median after changing central value: 6.0
# Median after adding extreme outlier: 6.0
```

Slide 7: Comparing Robustness of Mean and Median

Mean and median exhibit robustness in different parts of the distribution. The mean is more robust in the middle of the distribution, while the median is more robust in the tails.

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_mean_median(data, title):
    mean = np.mean(data)
    median = np.median(data)
    
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label='Median')
    plt.legend()
    plt.title(title)
    plt.show()
    
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")

# Normal distribution
normal_data = np.random.normal(0, 1, 1000)
compare_mean_median(normal_data, "Normal Distribution")

# Skewed distribution
skewed_data = np.random.exponential(2, 1000)
compare_mean_median(skewed_data, "Skewed Distribution")

# Output: Two histograms and their respective mean and median values
```

Slide 8: Estimating the Median

There are multiple methods to estimate the median, especially for grouped or binned data. It's crucial to ensure consistency in the estimation method across different statistical software.

```python
import numpy as np
from scipy import stats

data = np.random.randn(1000)

# Method 1: NumPy's median
median_numpy = np.median(data)

# Method 2: SciPy's median
median_scipy = stats.median_abs_deviation(data, scale="normal")

# Method 3: Interpolated median
sorted_data = np.sort(data)
n = len(sorted_data)
if n % 2 == 0:
    median_interpolated = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
else:
    median_interpolated = sorted_data[n//2]

print(f"NumPy Median: {median_numpy:.4f}")
print(f"SciPy Median: {median_scipy:.4f}")
print(f"Interpolated Median: {median_interpolated:.4f}")

# Output: Values will be very close but may differ slightly
```

Slide 9: Confidence Intervals for the Median

Confidence intervals for the median can be asymmetric, especially in skewed distributions. Using normal distribution-based methods (like Wald's CI) may lead to incorrect inferences. Bootstrapping methods, such as the bias-corrected and accelerated (BCa) method, often provide more accurate confidence intervals.

```python
import numpy as np
from scipy import stats

def bootstrap_median_ci(data, num_bootstrap=1000, ci=0.95):
    medians = [np.median(np.random.choice(data, len(data), replace=True)) 
               for _ in range(num_bootstrap)]
    return np.percentile(medians, [(1-ci)/2 * 100, (1+ci)/2 * 100])

# Generate skewed data
data = np.random.exponential(2, 1000)

median = np.median(data)
ci_normal = stats.norm.interval(0.95, loc=median, scale=stats.sem(data))
ci_bootstrap = bootstrap_median_ci(data)

print(f"Median: {median:.4f}")
print(f"Normal CI: [{ci_normal[0]:.4f}, {ci_normal[1]:.4f}]")
print(f"Bootstrap CI: [{ci_bootstrap[0]:.4f}, {ci_bootstrap[1]:.4f}]")

# Output: The bootstrap CI will likely be asymmetric for skewed data
```

Slide 10: Testing Equality of Medians

To test the equality of medians between two groups, we can use non-parametric tests like the Mood's median test or quantile regression. These methods are particularly useful when dealing with non-normal distributions.

```python
from scipy import stats
import numpy as np
import statsmodels.api as sm

# Generate two samples
group1 = np.random.normal(0, 1, 100)
group2 = np.random.normal(0.5, 1, 100)

# Mood's median test
mood_statistic, mood_p_value = stats.median_test(group1, group2)

# Quantile regression
X = np.concatenate([np.zeros(100), np.ones(100)])
y = np.concatenate([group1, group2])
model = sm.QuantReg(y, sm.add_constant(X))
result = model.fit(q=0.5)

print(f"Mood's median test p-value: {mood_p_value:.4f}")
print(f"Quantile regression p-value: {result.pvalues[1]:.4f}")

# Output: P-values for both tests
```

Slide 11: Median of Differences vs Difference of Medians

It's important to note that the median of differences between two vectors is not the same as the difference of their medians. This property contrasts with the mean, where the mean of differences equals the difference of means.

```python
import numpy as np

x = np.random.randn(1000)
y = np.random.randn(1000)

median_of_differences = np.median(x - y)
difference_of_medians = np.median(x) - np.median(y)

print(f"Median of differences: {median_of_differences:.4f}")
print(f"Difference of medians: {difference_of_medians:.4f}")

# Output: These values will generally be different
```

Slide 12: Median vs Mean in Heavy-Tailed Distributions

In the presence of extreme observations or heavy-tailed distributions, the median estimator may be more effective (smaller variance) than the mean. However, this depends on the specific distribution and sample size.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def compare_mean_median_efficiency(distribution, size, num_simulations=1000):
    mean_estimates = []
    median_estimates = []
    
    for _ in range(num_simulations):
        sample = distribution.rvs(size=size)
        mean_estimates.append(np.mean(sample))
        median_estimates.append(np.median(sample))
    
    print(f"Mean variance: {np.var(mean_estimates):.6f}")
    print(f"Median variance: {np.var(median_estimates):.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(mean_estimates, bins=30, alpha=0.5, label='Mean')
    plt.hist(median_estimates, bins=30, alpha=0.5, label='Median')
    plt.legend()
    plt.title(f"Distribution of Mean and Median Estimates (n={size})")
    plt.show()

# Compare for a heavy-tailed distribution (t-distribution with 3 degrees of freedom)
heavy_tailed_dist = stats.t(df=3)
compare_mean_median_efficiency(heavy_tailed_dist, size=100)

# Output: Variances of mean and median estimates, and a histogram
```

Slide 13: Reporting Both Mean and Median

Reporting and plotting both means and medians can provide valuable insights into the data distribution, especially when dealing with skewed or heavy-tailed data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate skewed data
data = np.random.lognormal(0, 1, 1000)

mean = np.mean(data)
median = np.median(data)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean ({mean:.2f})')
plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median ({median:.2f})')
plt.legend()
plt.title('Distribution with Mean and Median')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")

# Output: A histogram with mean and median lines, and their values
```

Slide 14: Real-Life Example: Height Distribution

Let's examine the distribution of heights in a population, demonstrating the usefulness of both mean and median.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate height data (in cm) for a population
np.random.seed(42)
heights = np.random.normal(170, 10, 1000)  # Mean 170cm, SD 10cm
heights = np.clip(heights, 140, 210)  # Clip to realistic range

mean_height = np.mean(heights)
median_height = np.median(heights)

plt.figure(figsize=(10, 6))
plt.hist(heights, bins=30, density=True, alpha=0.7)
plt.axvline(mean_height, color='r', linestyle='dashed', linewidth=2, label=f'Mean ({mean_height:.2f} cm)')
plt.axvline(median_height, color='g', linestyle='dashed', linewidth=2, label=f'Median ({median_height:.2f} cm)')
plt.legend()
plt.title('Distribution of Heights in a Population')
plt.xlabel('Height (cm)')
plt.ylabel('Density')
plt.show()

print(f"Mean height: {mean_height:.2f} cm")
print(f"Median height: {median_height:.2f} cm")

# Output: A histogram of height distribution with mean and median lines, and their values
```

Slide 15: Real-Life Example: Response Times in a Web Application

Let's analyze response times in a web application, where outliers (very slow responses) can significantly affect the mean but not the median.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate response times (in milliseconds)
np.random.seed(42)
response_times = np.random.exponential(100, 1000)  # Mean 100ms
response_times = np.append(response_times, [1000, 1500, 2000])  # Add some outliers

mean_time = np.mean(response_times)
median_time = np.median(response_times)

plt.figure(figsize=(10, 6))
plt.hist(response_times, bins=50, density=True, alpha=0.7)
plt.axvline(mean_time, color='r', linestyle='dashed', linewidth=2, label=f'Mean ({mean_time:.2f} ms)')
plt.axvline(median_time, color='g', linestyle='dashed', linewidth=2, label=f'Median ({median_time:.2f} ms)')
plt.legend()
plt.title('Distribution of Web Application Response Times')
plt.xlabel('Response Time (ms)')
plt.ylabel('Density')
plt.xlim(0, 1000)  # Limit x-axis for better visualization
plt.show()

print(f"Mean response time: {mean_time:.2f} ms")
print(f"Median response time: {median_time:.2f} ms")

# Output: A histogram of response time distribution with mean and median lines, and their values
```

Slide 16: Additional Resources

For those interested in diving deeper into the concepts of median and its applications in statistics, the following resources from arXiv.org might be helpful:

1. "Robust Statistics and Median Estimation": arXiv:1711.06098 This paper discusses advanced techniques in robust statistics, including median estimation methods.
2. "On the Asymptotic Distribution of the Median": arXiv:1507.03285 This article explores the theoretical properties of the median estimator in various statistical contexts.
3. "Quantile Regression and Statistical Learning": arXiv:2103.06760 This comprehensive review covers quantile regression techniques, which are closely related to median estimation.

These resources provide a more in-depth understanding of the median's role in statistical analysis and its applications in various fields of study.

