## Proving the Central Limit Theorem
Slide 1: Understanding Central Limit Theorem Fundamentals

The Central Limit Theorem is a foundational concept in statistics stating that the sampling distribution of means approaches normality as sample size increases, regardless of the underlying population distribution. This implementation demonstrates basic CLT concepts.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data from different distributions
np.random.seed(42)
sample_sizes = [5, 30, 100, 1000]
n_simulations = 10000

# Create samples from exponential distribution
population = np.random.exponential(size=100000)
sample_means = []

for size in sample_sizes:
    means = [np.mean(np.random.choice(population, size=size)) 
             for _ in range(n_simulations)]
    sample_means.append(means)

# Calculate theoretical normal distribution parameters
pop_mean = np.mean(population)
pop_std = np.std(population)
```

Slide 2: Visualizing CLT in Action

This implementation creates visualizations demonstrating how sample means converge to normal distribution as sample size increases, providing clear evidence of the Central Limit Theorem in practice.

```python
def plot_sampling_distribution(sample_means, sample_sizes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (means, size) in enumerate(zip(sample_means, sample_sizes)):
        # Plot histogram of sample means
        axes[i].hist(means, bins=50, density=True, alpha=0.7)
        
        # Overlay theoretical normal distribution
        x = np.linspace(min(means), max(means), 100)
        theoretical_normal = stats.norm.pdf(x, 
                                         loc=pop_mean,
                                         scale=pop_std/np.sqrt(size))
        axes[i].plot(x, theoretical_normal, 'r-', lw=2)
        axes[i].set_title(f'Sample Size = {size}')
        
    plt.tight_layout()
    return fig
```

Slide 3: Mathematical Foundation of CLT

The mathematical basis of CLT involves moment-generating functions and probability theory. This code implements the theoretical foundations using LaTeX notation for mathematical expressions and numerical verification.

```python
# Mathematical representation of CLT
"""
$$
\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0,1)
$$

Where:
$$
\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i
$$
"""

def verify_clt_conditions(data, confidence_level=0.95):
    sample_mean = np.mean(data)
    sample_std = np.std(data)
    n = len(data)
    
    # Calculate confidence interval
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_error = z_score * (sample_std / np.sqrt(n))
    
    ci_lower = sample_mean - margin_error
    ci_upper = sample_mean + margin_error
    
    return sample_mean, (ci_lower, ci_upper)
```

Slide 4: Implementing Sample Size Analysis

This implementation explores how different sample sizes affect the convergence to normality, providing quantitative measures through statistical tests and visualization of the sampling distribution.

```python
def analyze_sample_size_effect(population, sizes=[30, 100, 500, 1000], 
                             n_iterations=1000):
    results = {}
    
    for size in sizes:
        normality_scores = []
        for _ in range(n_iterations):
            sample = np.random.choice(population, size=size)
            # Shapiro-Wilk test for normality
            _, p_value = stats.shapiro(sample)
            normality_scores.append(p_value)
            
        results[size] = {
            'mean_p_value': np.mean(normality_scores),
            'std_p_value': np.std(normality_scores),
            'normality_rate': np.mean([p > 0.05 for p in normality_scores])
        }
    
    return results
```

Slide 5: Real-world Application - Stock Returns Analysis

Implementing CLT to analyze daily stock returns, demonstrating how financial data approaches normal distribution when aggregated over different time periods.

```python
import pandas as pd
import yfinance as yf

def analyze_stock_returns(ticker='AAPL', period='1y'):
    # Download stock data
    stock = yf.download(ticker, period=period)
    
    # Calculate daily returns
    returns = stock['Adj Close'].pct_change().dropna()
    
    # Create different sampling periods
    weekly_means = returns.rolling(window=5).mean().dropna()
    monthly_means = returns.rolling(window=21).mean().dropna()
    
    # Test for normality
    daily_stats = stats.normaltest(returns)
    weekly_stats = stats.normaltest(weekly_means)
    monthly_stats = stats.normaltest(monthly_means)
    
    return {
        'daily': {'data': returns, 'stats': daily_stats},
        'weekly': {'data': weekly_means, 'stats': weekly_stats},
        'monthly': {'data': monthly_means, 'stats': monthly_stats}
    }
```

Slide 6: Implementation of Monte Carlo CLT Verification

This implementation uses Monte Carlo simulation to verify CLT empirically by sampling from various non-normal distributions and demonstrating convergence to normality as sample size increases.

```python
def monte_carlo_clt_verification(distribution_func, params, 
                               sample_sizes=[10, 50, 200, 1000],
                               n_simulations=10000):
    results = {}
    
    for size in sample_sizes:
        sample_means = []
        for _ in range(n_simulations):
            # Generate sample from specified distribution
            sample = distribution_func(*params, size=size)
            sample_means.append(np.mean(sample))
            
        # Analyze normality of sampling distribution
        _, p_value = stats.normaltest(sample_means)
        skew = stats.skew(sample_means)
        kurtosis = stats.kurtosis(sample_means)
        
        results[size] = {
            'p_value': p_value,
            'skewness': skew,
            'kurtosis': kurtosis,
            'means': sample_means
        }
    
    return results
```

Slide 7: Confidence Interval Estimation Using CLT

The CLT enables accurate confidence interval estimation for population parameters. This implementation demonstrates how to calculate and validate confidence intervals using CLT principles.

```python
def calculate_ci_metrics(data, confidence_levels=[0.90, 0.95, 0.99]):
    results = {}
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    for conf_level in confidence_levels:
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        margin_error = z_score * (sample_std / np.sqrt(n))
        
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error
        
        results[conf_level] = {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_error': margin_error,
            'z_score': z_score
        }
    
    return results
```

Slide 8: Simulation of Different Parent Distributions

This code demonstrates how CLT applies to different underlying distributions by simulating samples from various probability distributions and analyzing their sampling distributions.

```python
def simulate_parent_distributions():
    # Define sample parameters
    n_samples = 1000
    sample_size = 30
    distributions = {
        'exponential': np.random.exponential,
        'uniform': np.random.uniform,
        'gamma': lambda size: np.random.gamma(2, 2, size),
        'lognormal': np.random.lognormal
    }
    
    results = {}
    for dist_name, dist_func in distributions.items():
        sample_means = []
        for _ in range(n_samples):
            sample = dist_func(size=sample_size)
            sample_means.append(np.mean(sample))
            
        results[dist_name] = {
            'means': sample_means,
            'normality_test': stats.normaltest(sample_means),
            'qq_plot_data': stats.probplot(sample_means)
        }
    
    return results
```

Slide 9: Real-world Application - Clinical Trial Analysis

Implementation of CLT principles in analyzing clinical trial data, demonstrating how to handle multiple treatment groups and assess statistical significance.

```python
def analyze_clinical_trial(control_group, treatment_group, 
                         bootstrap_iterations=10000):
    def bootstrap_mean(data):
        return np.random.choice(data, size=len(data), replace=True).mean()
    
    # Calculate observed difference
    observed_diff = np.mean(treatment_group) - np.mean(control_group)
    
    # Bootstrap sampling distributions
    bootstrap_diffs = []
    for _ in range(bootstrap_iterations):
        control_mean = bootstrap_mean(control_group)
        treatment_mean = bootstrap_mean(treatment_group)
        bootstrap_diffs.append(treatment_mean - control_mean)
    
    # Calculate confidence interval
    ci = np.percentile(bootstrap_diffs, [2.5, 97.5])
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_difference': observed_diff,
        'confidence_interval': ci,
        'p_value': p_value,
        'bootstrap_distribution': bootstrap_diffs
    }
```

Slide 10: Variance Analysis and Sample Size Determination

This implementation focuses on analyzing how sample size affects variance and helps determine optimal sample sizes for different statistical scenarios.

```python
def analyze_variance_sample_size(population, target_precision=0.05,
                               confidence_level=0.95,
                               size_range=range(10, 1001, 10)):
    pop_variance = np.var(population)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    results = {}
    for n in size_range:
        # Standard error of the mean
        sem = np.sqrt(pop_variance / n)
        
        # Margin of error
        margin_error = z_score * sem
        
        # Relative precision
        rel_precision = margin_error / np.mean(population)
        
        results[n] = {
            'sem': sem,
            'margin_error': margin_error,
            'relative_precision': rel_precision,
            'meets_target': rel_precision <= target_precision
        }
    
    # Find minimum sample size meeting target precision
    min_n = min((n for n in size_range 
                 if results[n]['relative_precision'] <= target_precision),
                default=None)
    
    return results, min_n
```

Slide 11: Demonstrating CLT with Non-Standard Distributions

This implementation explores CLT's application to highly skewed and multimodal distributions, showing how the theorem holds even for complex probability distributions that deviate significantly from normality.

```python
def analyze_complex_distributions(n_samples=10000):
    # Create a mixture of distributions
    def generate_mixture():
        components = [
            np.random.exponential(scale=2, size=n_samples),
            np.random.gamma(2, 2, size=n_samples),
            np.random.lognormal(0, 0.5, size=n_samples)
        ]
        weights = [0.4, 0.3, 0.3]
        indices = np.random.choice(len(components), 
                                 size=n_samples, 
                                 p=weights)
        return np.array([components[i][j] 
                        for j, i in enumerate(indices)])
    
    sample_sizes = [10, 30, 100, 500]
    results = {}
    
    for size in sample_sizes:
        means = []
        for _ in range(n_samples):
            sample = generate_mixture()
            means.append(np.mean(sample[:size]))
            
        results[size] = {
            'means': means,
            'normality_test': stats.normaltest(means),
            'skewness': stats.skew(means),
            'kurtosis': stats.kurtosis(means)
        }
    
    return results
```

Slide 12: Performance Metrics and CLT Validation

This implementation provides comprehensive metrics to validate CLT assumptions and quantify the convergence to normality across different sample sizes and distributions.

```python
def calculate_clt_metrics(data_generator, 
                         sample_sizes=[30, 100, 500, 1000],
                         n_iterations=10000):
    metrics = {}
    
    for size in sample_sizes:
        sample_means = []
        qq_correlations = []
        
        for _ in range(n_iterations):
            sample = data_generator(size)
            sample_means.append(np.mean(sample))
            
            # Calculate Q-Q plot correlation
            theoretical_quantiles = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(sample_means)))
            qq_corr = np.corrcoef(
                theoretical_quantiles,
                np.sort(stats.zscore(sample_means)))[0,1]
            qq_correlations.append(qq_corr)
        
        # Calculate comprehensive metrics
        metrics[size] = {
            'mean_qq_correlation': np.mean(qq_correlations),
            'normality_test': stats.normaltest(sample_means),
            'anderson_darling': stats.anderson(sample_means),
            'jarque_bera': stats.jarque_bera(sample_means),
            'descriptive_stats': {
                'mean': np.mean(sample_means),
                'std': np.std(sample_means),
                'skew': stats.skew(sample_means),
                'kurtosis': stats.kurtosis(sample_means)
            }
        }
    
    return metrics
```

Slide 13: Real-world Application - Quality Control Analysis

Implementation of CLT in manufacturing quality control, demonstrating how to analyze production measurements and establish control limits based on sampling distributions.

```python
def quality_control_analysis(measurements, spec_limits, 
                           subgroup_size=25):
    def calculate_control_limits(data, size):
        mean = np.mean(data)
        std = np.std(data)
        return {
            'ucl': mean + 3 * (std / np.sqrt(size)),
            'lcl': mean - 3 * (std / np.sqrt(size)),
            'mean': mean,
            'std': std
        }
    
    # Split data into subgroups
    n_subgroups = len(measurements) // subgroup_size
    subgroups = np.array_split(measurements, n_subgroups)
    subgroup_means = [np.mean(subgroup) for subgroup in subgroups]
    
    # Calculate control limits
    control_limits = calculate_control_limits(
        measurements, subgroup_size)
    
    # Process capability analysis
    cp = (spec_limits['upper'] - spec_limits['lower']) / (6 * control_limits['std'])
    cpk = min(
        (spec_limits['upper'] - control_limits['mean']) / (3 * control_limits['std']),
        (control_limits['mean'] - spec_limits['lower']) / (3 * control_limits['std'])
    )
    
    return {
        'control_limits': control_limits,
        'capability_indices': {'cp': cp, 'cpk': cpk},
        'subgroup_means': subgroup_means,
        'normality_test': stats.normaltest(subgroup_means)
    }
```

Slide 14: Additional Resources

*   Understanding the Central Limit Theorem: A Comprehensive Review [https://arxiv.org/abs/1808.07383](https://arxiv.org/abs/1808.07383)
*   Modern Applications of the Central Limit Theorem in Machine Learning [https://arxiv.org/abs/2001.07268](https://arxiv.org/abs/2001.07268)
*   Non-asymptotic Analysis of the Central Limit Theorem [https://arxiv.org/abs/1910.03832](https://arxiv.org/abs/1910.03832)
*   Statistical Learning Theory and the Central Limit Theorem [https://www.science.org/doi/10.1126/science.statistics.review](https://www.science.org/doi/10.1126/science.statistics.review)
*   Practical Applications of CLT in Big Data Analytics [https://www.researchgate.net/publication/statistical\_applications](https://www.researchgate.net/publication/statistical_applications)

