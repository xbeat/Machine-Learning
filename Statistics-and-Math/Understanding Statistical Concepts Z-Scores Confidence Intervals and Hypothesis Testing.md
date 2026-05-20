## Understanding Statistical Concepts Z-Scores Confidence Intervals and Hypothesis Testing
Slide 1: Z-Score Implementation from Scratch

Implementing Z-score calculation for understanding data point positions relative to the mean in standard deviation units.

```python
import numpy as np

def calculate_zscore(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    z_scores = [(x - mean) / std for x in data]
    return z_scores

# Example usage
data = [2, 4, 6, 8, 10]
z_scores = calculate_zscore(data)
print(f"Data: {data}\nZ-scores: {[round(z, 2) for z in z_scores]}")
```

Slide 2: Confidence Interval Computation

Statistical implementation of confidence intervals using Student's t-distribution for small sample sizes.

```python
import numpy as np
from scipy import stats

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * sem
    return mean - margin_error, mean + margin_error

# Example
sample_data = [23, 25, 21, 24, 26, 22, 25]
ci_lower, ci_upper = confidence_interval(sample_data)
print(f"95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
```

Slide 3: Hypothesis Testing Framework

Implementation of a two-tailed t-test for comparing sample means against a null hypothesis.

```python
def two_tailed_ttest(sample, population_mean, alpha=0.05):
    t_stat, p_value = stats.ttest_1samp(sample, population_mean)
    
    result = {
        't_statistic': t_stat,
        'p_value': p_value,
        'reject_null': p_value < alpha,
        'confidence_level': 1 - alpha
    }
    return result

# Example
sample = [102, 98, 103, 95, 98, 102, 97, 99]
test_result = two_tailed_ttest(sample, population_mean=100)
print(f"Test Results:\n{test_result}")
```

Slide 4: P-Value Visualization

Creating visual representation of p-value regions in normal distribution using matplotlib.

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_pvalue_region(z_score):
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x, 0, 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Normal Distribution')
    plt.fill_between(x[x >= abs(z_score)], y[x >= abs(z_score)], 
                    color='r', alpha=0.3, label='P-value region')
    plt.fill_between(x[x <= -abs(z_score)], y[x <= -abs(z_score)], 
                    color='r', alpha=0.3)
    plt.title('P-value Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example
plot_pvalue_region(1.96)  # 95% confidence level
```

Slide 5: Sampling Distribution Simulation

Demonstrating the central limit theorem through Monte Carlo simulation of sampling distributions.

```python
def simulate_sampling_distribution(population, sample_size, n_samples):
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(population, size=sample_size)
        sample_means.append(np.mean(sample))
    return np.array(sample_means)

# Simulation
population = np.random.exponential(size=10000)
sample_means = simulate_sampling_distribution(population, sample_size=30, n_samples=1000)

plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=50, density=True)
plt.title('Sampling Distribution of Sample Means')
plt.show()
```

Slide 6: Effect Size Calculator

Computing Cohen's d effect size for quantifying the magnitude of differences between groups.

```python
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_se
    
    return cohen_d

# Example
control = [23, 25, 21, 24]
treatment = [27, 29, 26, 28]
effect = cohens_d(control, treatment)
print(f"Cohen's d: {effect:.3f}")
```

Slide 7: Statistical Power Analysis

Implementation of power analysis for determining sample size requirements.

```python
from scipy.stats import norm

def calculate_power(effect_size, n, alpha=0.05):
    critical_value = norm.ppf(1 - alpha/2)
    ncp = effect_size * np.sqrt(n)
    power = 1 - norm.cdf(critical_value - ncp)
    return power

def required_sample_size(effect_size, desired_power=0.8, alpha=0.05):
    n = 1
    while calculate_power(effect_size, n, alpha) < desired_power:
        n += 1
    return n

# Example
effect_size = 0.5
n = required_sample_size(effect_size)
print(f"Required sample size for effect size {effect_size}: {n}")
```

Slide 8: Bootstrap Confidence Intervals

Non-parametric bootstrap method for estimating confidence intervals.

```python
def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1 - confidence) * 100 / 2)
    upper = np.percentile(bootstrap_means, (1 + confidence) * 100 / 2)
    return lower, upper

# Example
data = [12, 14, 13, 15, 16, 12, 14]
ci_lower, ci_upper = bootstrap_ci(data)
print(f"Bootstrap 95% CI: ({ci_lower:.2f}, {ci_upper:.2f})")
```

Slide 9: One-Way ANOVA Implementation

Implementing one-way ANOVA to compare means across multiple groups with F-statistic calculation.

```python
def one_way_anova(groups):
    k = len(groups)
    N = sum(len(group) for group in groups)
    
    grand_mean = np.mean([x for group in groups for x in group])
    between_ss = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    within_ss = sum(sum((x - np.mean(g))**2 for x in g) for g in groups)
    
    between_df = k - 1
    within_df = N - k
    
    f_stat = (between_ss/between_df)/(within_ss/within_df)
    p_value = 1 - stats.f.cdf(f_stat, between_df, within_df)
    
    return {'f_statistic': f_stat, 'p_value': p_value}

# Example
group1 = [4, 5, 6, 5]
group2 = [6, 7, 8, 7]
group3 = [8, 9, 8, 9]
result = one_way_anova([group1, group2, group3])
print(f"ANOVA Results: {result}")
```

Slide 10: Bayesian Parameter Estimation

Implementation of Bayesian estimation for mean and standard deviation using MCMC sampling.

```python
def bayesian_estimation(data, n_samples=10000):
    def log_likelihood(params):
        mu, sigma = params
        return sum(stats.norm.logpdf(x, mu, sigma) for x in data)
    
    current = [np.mean(data), np.std(data)]
    chains = []
    
    for _ in range(n_samples):
        proposal = [
            current[0] + np.random.normal(0, 0.1),
            current[1] + np.random.normal(0, 0.1)
        ]
        
        if proposal[1] <= 0:
            continue
            
        current_ll = log_likelihood(current)
        proposal_ll = log_likelihood(proposal)
        
        if np.log(np.random.random()) < proposal_ll - current_ll:
            current = proposal
        
        chains.append(current.copy())
    
    return np.array(chains)

# Example
data = [2.1, 2.3, 2.2, 2.4, 2.3]
posterior_samples = bayesian_estimation(data)
print(f"Posterior mean: {np.mean(posterior_samples[:, 0]):.3f}")
print(f"Posterior std: {np.mean(posterior_samples[:, 1]):.3f}")
```

Slide 11: Statistical Power Simulation

Comprehensive simulation for understanding statistical power under different conditions.

```python
def power_simulation(effect_size, sample_size, n_simulations=10000, alpha=0.05):
    significant_tests = 0
    
    for _ in range(n_simulations):
        control = np.random.normal(0, 1, sample_size)
        treatment = np.random.normal(effect_size, 1, sample_size)
        
        _, p_value = stats.ttest_ind(control, treatment)
        if p_value < alpha:
            significant_tests += 1
    
    return significant_tests / n_simulations

# Example
effect_sizes = [0.2, 0.5, 0.8]
sample_sizes = [20, 50, 100]

results = {}
for effect in effect_sizes:
    for size in sample_sizes:
        power = power_simulation(effect, size)
        results[f"Effect:{effect}_Size:{size}"] = power

print("Power Analysis Results:")
for k, v in results.items():
    print(f"{k}: {v:.3f}")
```

Slide 12: Non-parametric Statistical Tests

Implementation of Mann-Whitney U test and Wilcoxon signed-rank test for non-normally distributed data.

```python
def mann_whitney_u(group1, group2, alternative='two-sided'):
    stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
    return {'statistic': stat, 'p_value': p_value}

def wilcoxon_test(pre_treatment, post_treatment):
    stat, p_value = stats.wilcoxon(pre_treatment, post_treatment)
    return {'statistic': stat, 'p_value': p_value}

# Example usage
control = [45, 38, 52, 48, 47]
treatment = [53, 55, 51, 49, 54]
mw_result = mann_whitney_u(control, treatment)
print(f"Mann-Whitney U test: {mw_result}")
```

Slide 13: Multiple Comparison Correction

Implementing Bonferroni and Benjamini-Hochberg corrections for multiple hypothesis testing.

```python
def multiple_comparison_correction(p_values, method='bonferroni'):
    if method == 'bonferroni':
        return np.minimum(p_values * len(p_values), 1.0)
    elif method == 'benjamini-hochberg':
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        n = len(p_values)
        adjusted_p = np.minimum(1, sorted_p * n / np.arange(1, n + 1))
        for i in range(n-2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i+1])
        return adjusted_p[np.argsort(sorted_idx)]

# Example
p_values = np.array([0.01, 0.04, 0.03, 0.005])
bonf_corrected = multiple_comparison_correction(p_values, 'bonferroni')
bh_corrected = multiple_comparison_correction(p_values, 'benjamini-hochberg')
print(f"Original p-values: {p_values}")
print(f"Bonferroni corrected: {bonf_corrected}")
print(f"Benjamini-Hochberg corrected: {bh_corrected}")
```

Slide 14: Additional Resources

*   arXiv:1901.09342 - "A Comprehensive Review of Statistical Testing Methods"
*   arXiv:2003.12867 - "Modern Approaches to Multiple Comparison Problems"
*   arXiv:1904.10220 - "Statistical Power Analysis: Methods and Applications"
*   [https://www.nature.com/articles/s41467-019-09785-8](https://www.nature.com/articles/s41467-019-09785-8) - "Best Practices in Statistical Analysis"
*   [https://www.sciencedirect.com/science/article/pii/S0169743919308098](https://www.sciencedirect.com/science/article/pii/S0169743919308098) - "Advanced Statistical Methods in Data Science"

