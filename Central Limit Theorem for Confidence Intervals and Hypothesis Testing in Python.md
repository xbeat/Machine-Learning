## Central Limit Theorem for Confidence Intervals and Hypothesis Testing in Python
Slide 1: Introduction to the Central Limit Theorem

The Central Limit Theorem (CLT) is a fundamental concept in probability theory and statistics. It states that the distribution of sample means approximates a normal distribution as the sample size becomes larger, regardless of the population's original distribution. This theorem forms the basis for many statistical inferences and is crucial in understanding confidence intervals and hypothesis testing.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a non-normal population
population = np.random.exponential(scale=2, size=10000)

# Function to calculate sample means
def sample_means(population, sample_size, num_samples):
    return [np.mean(np.random.choice(population, sample_size)) for _ in range(num_samples)]

# Calculate sample means for different sample sizes
sample_sizes = [5, 30, 100]
num_samples = 1000

plt.figure(figsize=(15, 5))
for i, size in enumerate(sample_sizes):
    means = sample_means(population, size, num_samples)
    plt.subplot(1, 3, i+1)
    plt.hist(means, bins=30, edgecolor='black')
    plt.title(f'Sample Size: {size}')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 2: Properties of the Central Limit Theorem

The CLT has several important properties that make it a powerful tool in statistical analysis:

1. It applies to any population distribution, as long as the population has a finite variance.
2. The sample size needed for the CLT to hold depends on the shape of the original distribution.
3. The mean of the sampling distribution is equal to the population mean.
4. The standard error of the sampling distribution decreases as the sample size increases.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a non-normal population
population = np.random.exponential(scale=2, size=10000)

# Function to calculate standard error
def standard_error(sample_size):
    return np.std(population) / np.sqrt(sample_size)

# Calculate standard errors for different sample sizes
sample_sizes = np.arange(10, 1000, 10)
std_errors = [standard_error(size) for size in sample_sizes]

plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, std_errors)
plt.title('Standard Error vs. Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Standard Error')
plt.show()
```

Slide 3: Applications of the Central Limit Theorem

The CLT has numerous applications in various fields, including:

1. Quality control in manufacturing
2. Medical research and clinical trials
3. Environmental science and climate studies
4. Social sciences and opinion polls
5. Economics and market research

These applications rely on the CLT to make inferences about population parameters from sample statistics, enabling researchers and analysts to draw meaningful conclusions from limited data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate a quality control process
def manufacturing_process(batch_size, defect_rate):
    return np.random.binomial(batch_size, defect_rate)

# Parameters
batch_size = 1000
true_defect_rate = 0.05
num_batches = 100

# Simulate batches and calculate defect rates
defect_rates = [manufacturing_process(batch_size, true_defect_rate) / batch_size for _ in range(num_batches)]

plt.figure(figsize=(10, 6))
plt.hist(defect_rates, bins=20, edgecolor='black')
plt.title('Distribution of Defect Rates in Manufacturing')
plt.xlabel('Defect Rate')
plt.ylabel('Frequency')
plt.axvline(true_defect_rate, color='red', linestyle='dashed', linewidth=2, label='True Defect Rate')
plt.legend()
plt.show()
```

Slide 4: Confidence Intervals: Definition and Interpretation

A confidence interval is a range of values that is likely to contain the true population parameter with a certain level of confidence. It provides a measure of the uncertainty associated with a sample estimate. The interpretation of a 95% confidence interval is that if we were to repeat the sampling process many times and calculate the interval each time, about 95% of these intervals would contain the true population parameter.

```python
import numpy as np
from scipy import stats

# Generate sample data
sample_size = 100
sample_mean = 25
sample_std = 5

# Calculate confidence interval
confidence_level = 0.95
degrees_freedom = sample_size - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
margin_of_error = t_value * (sample_std / np.sqrt(sample_size))

lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")
```

Slide 5: Constructing Confidence Intervals

To construct a confidence interval, we need:

1. The sample mean
2. The standard error of the mean
3. The critical value from the t-distribution (for small samples) or the standard normal distribution (for large samples)

The formula for a confidence interval is: CI = sample mean ± (critical value \* standard error)

```python
import numpy as np
from scipy import stats

def confidence_interval(data, confidence=0.95):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    sample_size = len(data)
    
    degrees_freedom = sample_size - 1
    t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
    margin_of_error = t_value * (sample_std / np.sqrt(sample_size))
    
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    
    return (lower_bound, upper_bound)

# Example usage
data = np.random.normal(loc=100, scale=15, size=30)
ci = confidence_interval(data)
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

Slide 6: Factors Affecting Confidence Interval Width

The width of a confidence interval is influenced by several factors:

1. Sample size: Larger samples lead to narrower intervals
2. Variability in the data: More variable data results in wider intervals
3. Confidence level: Higher confidence levels produce wider intervals

Understanding these factors helps in designing studies and interpreting results.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ci_width(sample_size, std_dev, confidence):
    t_value = stats.t.ppf((1 + confidence) / 2, sample_size - 1)
    margin_of_error = t_value * (std_dev / np.sqrt(sample_size))
    return 2 * margin_of_error

sample_sizes = np.arange(10, 200, 10)
std_dev = 15
confidence_levels = [0.90, 0.95, 0.99]

plt.figure(figsize=(10, 6))
for confidence in confidence_levels:
    widths = [ci_width(n, std_dev, confidence) for n in sample_sizes]
    plt.plot(sample_sizes, widths, label=f'{confidence*100}% Confidence')

plt.title('Confidence Interval Width vs. Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Confidence Interval Width')
plt.legend()
plt.show()
```

Slide 7: Hypothesis Testing: Basic Concepts

Hypothesis testing is a statistical method used to make inferences about population parameters based on sample data. It involves:

1. Null hypothesis (H0): A statement of no effect or no difference
2. Alternative hypothesis (H1): A statement of an effect or a difference
3. Test statistic: A value calculated from the sample data
4. p-value: The probability of obtaining results as extreme as the observed data, assuming the null hypothesis is true

```python
import numpy as np
from scipy import stats

# Example: Testing if a coin is fair
# H0: p = 0.5 (fair coin)
# H1: p ≠ 0.5 (biased coin)

# Simulated data
n_flips = 100
n_heads = 60

# Calculate test statistic (z-score)
p_hat = n_heads / n_flips
p_null = 0.5
se = np.sqrt(p_null * (1 - p_null) / n_flips)
z_score = (p_hat - p_null) / se

# Calculate p-value (two-tailed test)
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 8: Types of Hypothesis Tests

There are various types of hypothesis tests, each suited for different scenarios:

1. One-sample t-test: Compares a sample mean to a known population mean
2. Two-sample t-test: Compares means of two independent groups
3. Paired t-test: Compares means of two related groups
4. Chi-square test: Analyzes categorical data
5. ANOVA: Compares means of three or more groups

The choice of test depends on the research question and the nature of the data.

```python
import numpy as np
from scipy import stats

# Example: One-sample t-test
# H0: μ = 100
# H1: μ ≠ 100

sample_data = np.random.normal(loc=105, scale=15, size=30)
population_mean = 100

t_statistic, p_value = stats.ttest_1samp(sample_data, population_mean)

print(f"One-sample t-test:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Example: Two-sample t-test
# H0: μ1 = μ2
# H1: μ1 ≠ μ2

group1 = np.random.normal(loc=100, scale=15, size=30)
group2 = np.random.normal(loc=110, scale=15, size=30)

t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"\nTwo-sample t-test:")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 9: Type I and Type II Errors

In hypothesis testing, two types of errors can occur:

1. Type I Error: Rejecting the null hypothesis when it's actually true (false positive)
2. Type II Error: Failing to reject the null hypothesis when it's actually false (false negative)

The probability of a Type I error is denoted by α (significance level), while the probability of a Type II error is denoted by β. The power of a test (1 - β) is the probability of correctly rejecting a false null hypothesis.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_hypothesis_test(mu0, mu1, sigma, alpha):
    x = np.linspace(mu0 - 4*sigma, mu1 + 4*sigma, 1000)
    y0 = stats.norm.pdf(x, mu0, sigma)
    y1 = stats.norm.pdf(x, mu1, sigma)
    
    critical_value = stats.norm.ppf(1 - alpha/2, mu0, sigma)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y0, label='Null Distribution')
    plt.plot(x, y1, label='Alternative Distribution')
    plt.fill_between(x, 0, y0, where=(x <= -critical_value) | (x >= critical_value), alpha=0.3, color='red', label='Type I Error')
    plt.fill_between(x, 0, y1, where=(x > -critical_value) & (x < critical_value), alpha=0.3, color='blue', label='Type II Error')
    plt.axvline(-critical_value, color='black', linestyle='--')
    plt.axvline(critical_value, color='black', linestyle='--')
    plt.title('Type I and Type II Errors')
    plt.xlabel('Test Statistic')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

plot_hypothesis_test(mu0=0, mu1=2, sigma=1, alpha=0.05)
```

Slide 10: Power Analysis

Power analysis is a statistical method used to determine the sample size required to detect an effect of a given size with a certain level of confidence. It helps researchers design studies that are neither underpowered (risking Type II errors) nor overpowered (wasting resources).

Key components of power analysis:

1. Effect size: The magnitude of the difference you want to detect
2. Significance level (α): The probability of Type I error
3. Power (1 - β): The probability of correctly rejecting a false null hypothesis
4. Sample size: The number of observations needed

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

def calculate_power(effect_size, sample_size, alpha=0.05):
    critical_value = stats.norm.ppf(1 - alpha/2)
    beta = stats.norm.cdf(critical_value - effect_size * np.sqrt(sample_size))
    power = 1 - beta
    return power

effect_sizes = np.linspace(0.1, 1, 10)
sample_sizes = np.arange(10, 200, 10)

power_matrix = np.array([[calculate_power(es, n) for n in sample_sizes] for es in effect_sizes])

plt.figure(figsize=(12, 8))
plt.imshow(power_matrix, aspect='auto', extent=[sample_sizes[0], sample_sizes[-1], effect_sizes[0], effect_sizes[-1]], 
           origin='lower', cmap='viridis')
plt.colorbar(label='Power')
plt.title('Power Analysis Heatmap')
plt.xlabel('Sample Size')
plt.ylabel('Effect Size')
plt.show()
```

Slide 11: Real-Life Example: Environmental Impact Assessment

Environmental scientists are studying the impact of a new water treatment method on river water quality. They want to determine if there's a significant difference in dissolved oxygen levels before and after implementing the treatment.

Null Hypothesis (H0): There is no difference in dissolved oxygen levels before and after treatment. Alternative Hypothesis (H1): There is a difference in dissolved oxygen levels before and after treatment.

```python
import numpy as np
from scipy import stats

# Simulate dissolved oxygen levels (mg/L)
before_treatment = np.random.normal(loc=7.5, scale=0.5, size=30)
after_treatment = np.random.normal(loc=8.0, scale=0.5, size=30)

# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(before_treatment, after_treatment)

print(f"Mean before treatment: {np.mean(before_treatment):.2f} mg/L")
print(f"Mean after treatment: {np.mean(after_treatment):.2f} mg/L")
print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The treatment appears to have a significant effect.")
else:
    print("Fail to reject the null hypothesis. No significant effect detected.")
```

Slide 12: Bootstrapping: A Nonparametric Approach

Bootstrapping is a resampling technique used to estimate the sampling distribution of a statistic. It's particularly useful when the underlying distribution is unknown or when working with small sample sizes. The method involves:

1. Repeatedly sampling with replacement from the original dataset
2. Calculating the statistic of interest for each resampled dataset
3. Using the distribution of these statistics to make inferences

```python
import numpy as np
import matplotlib.pyplot as plt

def bootstrap_mean(data, num_bootstraps=10000):
    bootstrap_means = np.zeros(num_bootstraps)
    for i in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means[i] = np.mean(bootstrap_sample)
    return bootstrap_means

# Generate sample data
original_data = np.random.exponential(scale=2, size=50)

# Perform bootstrapping
bootstrap_results = bootstrap_mean(original_data)

# Calculate confidence interval
ci_lower, ci_upper = np.percentile(bootstrap_results, [2.5, 97.5])

plt.figure(figsize=(10, 6))
plt.hist(bootstrap_results, bins=50, edgecolor='black')
plt.axvline(np.mean(original_data), color='red', linestyle='dashed', label='Sample Mean')
plt.axvline(ci_lower, color='green', linestyle='dashed', label='95% CI')
plt.axvline(ci_upper, color='green', linestyle='dashed')
plt.title('Bootstrap Distribution of Sample Mean')
plt.xlabel('Mean')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

Slide 13: Limitations and Assumptions

While the Central Limit Theorem, confidence intervals, and hypothesis testing are powerful tools, they have limitations and rely on certain assumptions:

1. Sample size: The CLT approximation improves with larger sample sizes.
2. Independence: Observations should be independent of each other.
3. Random sampling: The sample should be representative of the population.
4. Normality: Some tests assume normality of the underlying population or sampling distribution.
5. Equal variances: Certain tests (e.g., t-test) assume equal variances between groups.

Violating these assumptions can lead to incorrect conclusions. It's crucial to check these assumptions and consider alternative methods when they're not met.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate samples from different distributions
normal_data = np.random.normal(loc=0, scale=1, size=1000)
skewed_data = np.random.exponential(scale=1, size=1000)
bimodal_data = np.concatenate([np.random.normal(loc=-2, scale=0.5, size=500),
                               np.random.normal(loc=2, scale=0.5, size=500)])

# Function to plot QQ plot
def plot_qq(data, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(title)
    plt.show()

# Plot QQ plots for each distribution
plot_qq(normal_data, "Normal Distribution")
plot_qq(skewed_data, "Skewed Distribution")
plot_qq(bimodal_data, "Bimodal Distribution")
```

Slide 14: Advanced Topics and Further Reading

To deepen your understanding of the Central Limit Theorem, confidence intervals, and hypothesis testing, consider exploring these advanced topics:

1. Bayesian inference and credible intervals
2. Nonparametric methods (e.g., permutation tests)
3. Multiple comparisons and false discovery rate
4. Effect sizes and practical significance
5. Meta-analysis and combining results across studies

For further reading, you may refer to the following resources:

1. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall/CRC.
2. Wasserman, L. (2004). All of Statistics: A Concise Course in Statistical Inference. Springer.
3. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis. Chapman and Hall/CRC.

For the latest research on statistical methods, you can explore preprints on arXiv.org in the Statistics category: [https://arxiv.org/list/stat/recent](https://arxiv.org/list/stat/recent)

