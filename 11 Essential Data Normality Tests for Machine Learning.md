## 11 Essential Data Normality Tests for Machine Learning
Slide 1: Introduction to Data Normality Testing

Data normality testing is crucial in many machine learning models and statistical analyses. This slideshow will explore 11 essential methods to test for normality in data distributions. We'll cover both visual and quantitative approaches, providing code examples and practical applications for each method.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate a normal distribution for demonstration
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(normal_data, bins=30, density=True, alpha=0.7)
plt.title("Histogram of Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Visual Methods - Q-Q Plot

A Q-Q (Quantile-Quantile) plot is a graphical tool to assess if a dataset follows a normal distribution. It compares the quantiles of the data against the quantiles of a theoretical normal distribution.

```python
import statsmodels.api as sm

plt.figure(figsize=(10, 6))
sm.qqplot(normal_data, line='s')
plt.title("Q-Q Plot")
plt.show()
```

Slide 3: Visual Methods - Probability Plot

A probability plot is similar to a Q-Q plot but uses a different scale on the y-axis. It's useful for identifying deviations from normality, especially in the tails of the distribution.

```python
import scipy.stats as stats

fig, ax = plt.subplots(figsize=(10, 6))
res = stats.probplot(normal_data, plot=ax)
ax.set_title("Probability Plot")
plt.show()
```

Slide 4: Shapiro-Wilk Test

The Shapiro-Wilk test is a statistical method to test the null hypothesis that a sample comes from a normally distributed population. It's particularly effective for small sample sizes.

```python
from scipy.stats import shapiro

stat, p_value = shapiro(normal_data)
print(f"Shapiro-Wilk test statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The data is likely normally distributed (fail to reject H0)")
else:
    print("The data is likely not normally distributed (reject H0)")
```

Slide 5: Kolmogorov-Smirnov Test

The Kolmogorov-Smirnov (K-S) test compares the cumulative distribution function of the data with that of a normal distribution. It's useful for larger sample sizes.

```python
from scipy.stats import kstest

stat, p_value = kstest(normal_data, 'norm')
print(f"K-S test statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The data is likely normally distributed (fail to reject H0)")
else:
    print("The data is likely not normally distributed (reject H0)")
```

Slide 6: Anderson-Darling Test

The Anderson-Darling test is another statistical method for testing normality. It's more sensitive to deviations in the tails of the distribution compared to the K-S test.

```python
from scipy.stats import anderson

result = anderson(normal_data)
print(f"Anderson-Darling test statistic: {result.statistic:.4f}")
print("Critical values:", result.critical_values)
print("Significance levels:", result.significance_level)

# Interpret the result
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print(f"At {sl}% significance level, the data is normally distributed (fail to reject H0)")
    else:
        print(f"At {sl}% significance level, the data is not normally distributed (reject H0)")
```

Slide 7: Lilliefors Test

The Lilliefors test is a modification of the Kolmogorov-Smirnov test. It's used when the parameters of the normal distribution are not known and must be estimated from the sample.

```python
from scipy.stats import lilliefors

stat, p_value = lilliefors(normal_data)
print(f"Lilliefors test statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The data is likely normally distributed (fail to reject H0)")
else:
    print("The data is likely not normally distributed (reject H0)")
```

Slide 8: Skewness and Kurtosis

Skewness measures the asymmetry of the distribution, while kurtosis measures the "tailedness" of the distribution. Normal distributions have a skewness of 0 and a kurtosis of 3.

```python
from scipy.stats import skew, kurtosis

skewness = skew(normal_data)
kurt = kurtosis(normal_data)

print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurt:.4f}")

# Interpret the results
if abs(skewness) < 0.5 and abs(kurt) < 0.5:
    print("The data is approximately normally distributed")
else:
    print("The data may not be normally distributed")
```

Slide 9: Jarque-Bera Test

The Jarque-Bera test is based on the sample skewness and kurtosis. It tests whether the sample skewness and kurtosis match those of a normal distribution.

```python
from scipy.stats import jarque_bera

stat, p_value = jarque_bera(normal_data)
print(f"Jarque-Bera test statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The data is likely normally distributed (fail to reject H0)")
else:
    print("The data is likely not normally distributed (reject H0)")
```

Slide 10: D'Agostino's K^2 Test

D'Agostino's K^2 test combines skewness and kurtosis to produce an omnibus test of normality. It's effective at detecting deviations from normality due to either skewness or kurtosis.

```python
from scipy.stats import normaltest

stat, p_value = normaltest(normal_data)
print(f"D'Agostino's K^2 test statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value > alpha:
    print("The data is likely normally distributed (fail to reject H0)")
else:
    print("The data is likely not normally distributed (reject H0)")
```

Slide 11: Real-Life Example: Height Distribution

Let's apply some of these tests to a real-world example: the distribution of heights in a population.

```python
# Simulate height data (in cm) for a population
np.random.seed(42)
heights = np.random.normal(loc=170, scale=10, size=1000)

# Visual check
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=30, density=True, alpha=0.7)
plt.title("Distribution of Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.show()

# Perform normality tests
_, p_shapiro = shapiro(heights)
_, p_kstest = kstest(heights, 'norm')
_, p_normaltest = normaltest(heights)

print(f"Shapiro-Wilk p-value: {p_shapiro:.4f}")
print(f"Kolmogorov-Smirnov p-value: {p_kstest:.4f}")
print(f"D'Agostino's K^2 p-value: {p_normaltest:.4f}")
```

Slide 12: Real-Life Example: Reaction Times

Another practical application: testing the normality of reaction times in a psychological experiment.

```python
# Simulate reaction time data (in milliseconds)
np.random.seed(42)
reaction_times = np.random.lognormal(mean=5.5, sigma=0.4, size=1000)

# Visual check
plt.figure(figsize=(10, 6))
plt.hist(reaction_times, bins=30, density=True, alpha=0.7)
plt.title("Distribution of Reaction Times")
plt.xlabel("Reaction Time (ms)")
plt.ylabel("Frequency")
plt.show()

# Perform normality tests
_, p_shapiro = shapiro(reaction_times)
_, p_kstest = kstest(reaction_times, 'norm')
_, p_normaltest = normaltest(reaction_times)

print(f"Shapiro-Wilk p-value: {p_shapiro:.4f}")
print(f"Kolmogorov-Smirnov p-value: {p_kstest:.4f}")
print(f"D'Agostino's K^2 p-value: {p_normaltest:.4f}")
```

Slide 13: Transforming Non-Normal Data

When data is not normally distributed, we can sometimes apply transformations to achieve normality. Common transformations include log, square root, and Box-Cox.

```python
from scipy.stats import boxcox

# Apply log transformation
log_reaction_times = np.log(reaction_times)

# Apply Box-Cox transformation
boxcox_reaction_times, lambda_param = boxcox(reaction_times)

# Compare distributions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.hist(reaction_times, bins=30, density=True, alpha=0.7)
ax1.set_title("Original Data")

ax2.hist(log_reaction_times, bins=30, density=True, alpha=0.7)
ax2.set_title("Log Transformed")

ax3.hist(boxcox_reaction_times, bins=30, density=True, alpha=0.7)
ax3.set_title("Box-Cox Transformed")

plt.tight_layout()
plt.show()

# Test normality of transformed data
_, p_log = shapiro(log_reaction_times)
_, p_boxcox = shapiro(boxcox_reaction_times)

print(f"Log transform Shapiro-Wilk p-value: {p_log:.4f}")
print(f"Box-Cox transform Shapiro-Wilk p-value: {p_boxcox:.4f}")
```

Slide 14: Additional Resources

For further exploration of normality testing and its applications in machine learning:

1. "Testing for Normality" by Ralph B. D'Agostino (1986) ArXiv: [https://arxiv.org/abs/1011.2375](https://arxiv.org/abs/1011.2375)
2. "A Study of the Power of Some Tests for Normality" by Nornadiah Mohd Razali and Yap Bee Wah (2011) ArXiv: [https://arxiv.org/abs/1012.2754](https://arxiv.org/abs/1012.2754)
3. "Normality Tests for Statistical Analysis: A Guide for Non-Statisticians" by Ghasemi and Zahediasl (2012) DOI: 10.5812/ijem.3505

These resources provide in-depth discussions on various normality tests, their power, and applications in different fields of study.

