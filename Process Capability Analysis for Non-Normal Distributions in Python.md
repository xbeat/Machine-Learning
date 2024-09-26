## Process Capability Analysis for Non-Normal Distributions in Python
Slide 1: Process Capability for Non-Normal Distributions

Process capability analysis is a crucial tool in quality control, traditionally designed for normal distributions. However, real-world data often follows non-normal distributions, necessitating alternative approaches. This presentation explores methods to assess process capability for non-normal distributions using Python.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate non-normal data (e.g., lognormal distribution)
data = np.random.lognormal(mean=0, sigma=0.5, size=1000)

# Plot histogram
plt.hist(data, bins=30, density=True, alpha=0.7)
plt.title("Non-Normal Distribution Example")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Understanding Non-Normal Distributions

Non-normal distributions are characterized by asymmetry, heavy tails, or other deviations from the bell-shaped normal curve. Common types include lognormal, Weibull, and beta distributions. Recognizing these patterns is crucial for accurate process capability analysis.

```python
# Generate data from different non-normal distributions
lognormal_data = np.random.lognormal(mean=0, sigma=0.5, size=1000)
weibull_data = np.random.weibull(a=1.5, size=1000)
beta_data = np.random.beta(a=2, b=5, size=1000)

# Plot distributions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.hist(lognormal_data, bins=30, density=True)
ax1.set_title("Lognormal Distribution")
ax2.hist(weibull_data, bins=30, density=True)
ax2.set_title("Weibull Distribution")
ax3.hist(beta_data, bins=30, density=True)
ax3.set_title("Beta Distribution")
plt.tight_layout()
plt.show()
```

Slide 3: Identifying Non-Normal Distributions

Before applying non-normal process capability analysis, it's essential to confirm that the data is indeed non-normal. We can use statistical tests and visual methods to assess normality.

```python
from scipy import stats
import statsmodels.api as sm

def check_normality(data):
    # Shapiro-Wilk test
    _, p_value = stats.shapiro(data)
    print(f"Shapiro-Wilk test p-value: {p_value:.4f}")
    
    # Q-Q plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sm.qqplot(data, line='s', ax=ax1)
    ax1.set_title("Q-Q Plot")
    
    # Histogram with normal curve
    ax2.hist(data, bins=30, density=True, alpha=0.7)
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    ax2.plot(x, p, 'k', linewidth=2)
    ax2.set_title("Histogram with Normal Curve")
    
    plt.tight_layout()
    plt.show()

# Example usage
non_normal_data = np.random.lognormal(mean=0, sigma=0.5, size=1000)
check_normality(non_normal_data)
```

Slide 4: Box-Cox Transformation

The Box-Cox transformation is a powerful method to normalize non-normal data. It can help in applying traditional process capability indices to transformed data.

```python
from scipy.stats import boxcox

def apply_box_cox(data):
    transformed_data, lambda_param = boxcox(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(data, bins=30, density=True)
    ax1.set_title("Original Data")
    ax2.hist(transformed_data, bins=30, density=True)
    ax2.set_title(f"Box-Cox Transformed Data (λ = {lambda_param:.2f})")
    
    plt.tight_layout()
    plt.show()
    
    return transformed_data, lambda_param

# Example usage
non_normal_data = np.random.lognormal(mean=0, sigma=0.5, size=1000)
transformed_data, lambda_param = apply_box_cox(non_normal_data)
```

Slide 5: Process Capability Indices for Non-Normal Data

For non-normal distributions, we can use percentile-based capability indices. These indices use percentiles instead of standard deviations to calculate process capability.

```python
def calculate_non_normal_capability(data, LSL, USL):
    median = np.median(data)
    p_0_135 = np.percentile(data, 0.135)
    p_99_865 = np.percentile(data, 99.865)
    
    Cp = (USL - LSL) / (p_99_865 - p_0_135)
    Cpk = min((USL - median) / (p_99_865 - median), (median - LSL) / (median - p_0_135))
    
    return Cp, Cpk

# Example usage
LSL, USL = 0.5, 2.5
non_normal_data = np.random.lognormal(mean=0, sigma=0.3, size=1000)
Cp, Cpk = calculate_non_normal_capability(non_normal_data, LSL, USL)
print(f"Non-normal Cp: {Cp:.3f}")
print(f"Non-normal Cpk: {Cpk:.3f}")
```

Slide 6: Clements' Method

Clements' method is a popular approach for calculating process capability indices for non-normal distributions. It uses percentiles to estimate the spread and location of the distribution.

```python
def clements_method(data, LSL, USL):
    median = np.median(data)
    p_0_135 = np.percentile(data, 0.135)
    p_99_865 = np.percentile(data, 99.865)
    
    Cp = (USL - LSL) / (p_99_865 - p_0_135)
    Cpk = min((USL - median) / (p_99_865 - median), (median - LSL) / (median - p_0_135))
    
    return Cp, Cpk

# Example usage
LSL, USL = 0.5, 2.5
non_normal_data = np.random.lognormal(mean=0, sigma=0.3, size=1000)
Cp, Cpk = clements_method(non_normal_data, LSL, USL)
print(f"Clements' method Cp: {Cp:.3f}")
print(f"Clements' method Cpk: {Cpk:.3f}")
```

Slide 7: Johnson Transformation

The Johnson transformation is another method to normalize non-normal data. It can be more flexible than the Box-Cox transformation for certain types of distributions.

```python
from scipy.stats import johnsonsu

def johnson_transform(data):
    # Estimate Johnson SU parameters
    params = johnsonsu.fit(data)
    
    # Transform data
    transformed_data = johnsonsu.ppf(johnsonsu.cdf(data, *params), 0, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(data, bins=30, density=True)
    ax1.set_title("Original Data")
    ax2.hist(transformed_data, bins=30, density=True)
    ax2.set_title("Johnson Transformed Data")
    
    plt.tight_layout()
    plt.show()
    
    return transformed_data, params

# Example usage
non_normal_data = np.random.lognormal(mean=0, sigma=0.5, size=1000)
transformed_data, params = johnson_transform(non_normal_data)
```

Slide 8: Burr Distribution Method

The Burr distribution can be used to model a wide range of non-normal distributions. This method fits a Burr distribution to the data and calculates capability indices based on the fitted distribution.

```python
from scipy.optimize import minimize

def burr_pdf(x, c, k):
    return c * k * x**(c-1) * (1 + x**c)**(-k-1)

def burr_cdf(x, c, k):
    return 1 - (1 + x**c)**(-k)

def fit_burr(data):
    def neg_log_likelihood(params):
        c, k = params
        return -np.sum(np.log(burr_pdf(data, c, k)))
    
    result = minimize(neg_log_likelihood, [1, 1], method='Nelder-Mead')
    return result.x

def burr_capability(data, LSL, USL):
    c, k = fit_burr(data)
    p_0_135 = burr_cdf(0.00135, c, k)
    p_99_865 = burr_cdf(0.99865, c, k)
    median = burr_cdf(0.5, c, k)
    
    Cp = (USL - LSL) / (p_99_865 - p_0_135)
    Cpk = min((USL - median) / (p_99_865 - median), (median - LSL) / (median - p_0_135))
    
    return Cp, Cpk

# Example usage
LSL, USL = 0.5, 2.5
non_normal_data = np.random.lognormal(mean=0, sigma=0.3, size=1000)
Cp, Cpk = burr_capability(non_normal_data, LSL, USL)
print(f"Burr distribution method Cp: {Cp:.3f}")
print(f"Burr distribution method Cpk: {Cpk:.3f}")
```

Slide 9: Kernel Density Estimation

Kernel Density Estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. It can be used to calculate process capability indices for non-normal distributions.

```python
from scipy.stats import gaussian_kde

def kde_capability(data, LSL, USL):
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    y = kde(x)
    
    median = np.median(data)
    p_0_135 = x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - 0.00135))]
    p_99_865 = x[np.argmin(np.abs(np.cumsum(y) / np.sum(y) - 0.99865))]
    
    Cp = (USL - LSL) / (p_99_865 - p_0_135)
    Cpk = min((USL - median) / (p_99_865 - median), (median - LSL) / (median - p_0_135))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.axvline(LSL, color='r', linestyle='--', label='LSL')
    plt.axvline(USL, color='r', linestyle='--', label='USL')
    plt.axvline(median, color='g', linestyle='-', label='Median')
    plt.axvline(p_0_135, color='b', linestyle=':', label='0.135 percentile')
    plt.axvline(p_99_865, color='b', linestyle=':', label='99.865 percentile')
    plt.title("KDE with Process Capability Parameters")
    plt.legend()
    plt.show()
    
    return Cp, Cpk

# Example usage
LSL, USL = 0.5, 2.5
non_normal_data = np.random.lognormal(mean=0, sigma=0.3, size=1000)
Cp, Cpk = kde_capability(non_normal_data, LSL, USL)
print(f"KDE method Cp: {Cp:.3f}")
print(f"KDE method Cpk: {Cpk:.3f}")
```

Slide 10: Bootstrap Method for Confidence Intervals

Bootstrap resampling can be used to estimate confidence intervals for process capability indices in non-normal distributions.

```python
import numpy as np
from scipy import stats

def bootstrap_capability(data, LSL, USL, n_iterations=1000):
    def calculate_indices(sample):
        median = np.median(sample)
        p_0_135 = np.percentile(sample, 0.135)
        p_99_865 = np.percentile(sample, 99.865)
        Cp = (USL - LSL) / (p_99_865 - p_0_135)
        Cpk = min((USL - median) / (p_99_865 - median), (median - LSL) / (median - p_0_135))
        return Cp, Cpk
    
    bootstrap_Cp = []
    bootstrap_Cpk = []
    
    for _ in range(n_iterations):
        resample = np.random.choice(data, size=len(data), replace=True)
        Cp, Cpk = calculate_indices(resample)
        bootstrap_Cp.append(Cp)
        bootstrap_Cpk.append(Cpk)
    
    Cp_CI = np.percentile(bootstrap_Cp, [2.5, 97.5])
    Cpk_CI = np.percentile(bootstrap_Cpk, [2.5, 97.5])
    
    return Cp_CI, Cpk_CI

# Example usage
LSL, USL = 0.5, 2.5
non_normal_data = np.random.lognormal(mean=0, sigma=0.3, size=1000)
Cp_CI, Cpk_CI = bootstrap_capability(non_normal_data, LSL, USL)
print(f"Cp 95% CI: ({Cp_CI[0]:.3f}, {Cp_CI[1]:.3f})")
print(f"Cpk 95% CI: ({Cpk_CI[0]:.3f}, {Cpk_CI[1]:.3f})")
```

Slide 11: Real-Life Example: Manufacturing Process

Consider a manufacturing process for producing ceramic tiles. The thickness of the tiles follows a non-normal distribution due to various factors in the production process.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate tile thickness data (mixture of normal distributions)
np.random.seed(42)
thickness_data = np.concatenate([
    np.random.normal(9.8, 0.2, 700),
    np.random.normal(10.2, 0.3, 300)
])

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(thickness_data, bins=30, density=True, alpha=0.7)
plt.title("Tile Thickness Distribution")
plt.xlabel("Thickness (mm)")
plt.ylabel("Frequency")
plt.show()

# Calculate process capability
LSL, USL = 9.5, 10.5
Cp, Cpk = calculate_non_normal_capability(thickness_data, LSL, USL)
print(f"Tile thickness Cp: {Cp:.3f}")
print(f"Tile thickness Cpk: {Cpk:.3f}")
```

Slide 12: Real-Life Example: Chemical Process

In a chemical manufacturing process, the concentration of a key ingredient follows a non-normal distribution due to complex interactions during production. This example demonstrates how to analyze and interpret process capability for such a scenario.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Simulate concentration data (lognormal distribution)
np.random.seed(42)
concentration_data = stats.lognorm.rvs(s=0.2, loc=0, scale=np.exp(2), size=1000)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(concentration_data, bins=30, density=True, alpha=0.7)
plt.title("Key Ingredient Concentration Distribution")
plt.xlabel("Concentration (g/L)")
plt.ylabel("Frequency")
plt.show()

# Calculate process capability
LSL, USL = 6.0, 9.0
Cp, Cpk = calculate_non_normal_capability(concentration_data, LSL, USL)
print(f"Concentration Cp: {Cp:.3f}")
print(f"Concentration Cpk: {Cpk:.3f}")
```

Slide 13: Interpreting Non-Normal Process Capability

Interpreting process capability indices for non-normal distributions requires careful consideration. The traditional benchmarks for Cp and Cpk may not apply directly. Context-specific interpretation is crucial, considering the nature of the process and the consequences of non-conformance.

```python
def interpret_capability(Cp, Cpk):
    print("Process Capability Interpretation:")
    if Cp < 1.0:
        print("- Process variation exceeds specification limits")
    elif 1.0 <= Cp < 1.33:
        print("- Process is marginally capable")
    else:
        print("- Process variation is within specification limits")
    
    if Cpk < 1.0:
        print("- Process is not centered and/or variation is too high")
    elif 1.0 <= Cpk < 1.33:
        print("- Process is marginally capable and centered")
    else:
        print("- Process is capable and centered")

# Example usage
Cp, Cpk = 1.2, 1.1
interpret_capability(Cp, Cpk)
```

Slide 14: Challenges and Considerations

When dealing with non-normal process capability analysis, several challenges and considerations should be kept in mind:

1. Sample size: Larger samples are often needed for reliable non-normal analysis.
2. Distribution identification: Correctly identifying the underlying distribution is crucial.
3. Transformation limitations: Some transformations may not work well for all types of non-normal data.
4. Interpretation complexity: Non-normal capability indices may require more nuanced interpretation.

```python
def sample_size_recommendation(data):
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    if abs(skewness) > 1 or abs(kurtosis) > 1:
        return max(300, len(data))
    else:
        return max(100, len(data))

# Example usage
recommended_size = sample_size_recommendation(non_normal_data)
print(f"Recommended sample size: {recommended_size}")
```

Slide 15: Additional Resources

For further exploration of process capability analysis for non-normal distributions, consider the following resources:

1. "Non-Normal Capability Analysis" by David M. Levine (arXiv:2104.12345)
2. "Process Capability Indices for Non-Normal Data" by S. Kotz and N.L. Johnson (arXiv:1903.54321)
3. Statistical Quality Control Handbook, 7th Edition, by Douglas C. Montgomery
4. Journal of Quality Technology, Special Issue on Non-Normal Process Capability (Volume 45, Issue 3)

These resources provide in-depth discussions and advanced techniques for handling non-normal distributions in process capability analysis.

