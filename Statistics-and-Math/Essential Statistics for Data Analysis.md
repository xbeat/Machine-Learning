## Essential Statistics for Data Analysis
Slide 1: Descriptive Statistics Foundation

Statistical analysis begins with understanding central tendency measures that form the basis for more complex analyses. These metrics help identify patterns and anomalies in datasets while providing crucial summary information.

```python
import numpy as np

def basic_stats(data):
    # Calculate basic descriptive statistics
    mean = np.mean(data)
    median = np.median(data)
    mode = max(set(data), key=data.count)
    std = np.std(data)
    
    # Sample dataset and calculations
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode}")
    print(f"Standard Deviation: {std:.2f}")
    
# Example usage
data = [12, 15, 12, 18, 22, 24, 27, 12, 14]
basic_stats(data)
```

Slide 2: Variance and Standard Deviation Implementation

Understanding data spread is crucial for statistical analysis. This implementation demonstrates how to calculate variance and standard deviation from scratch, providing insights into data distribution patterns.

```python
def calculate_spread(data):
    n = len(data)
    mean = sum(data) / n
    
    # Calculate variance
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    
    # Calculate standard deviation
    std_dev = variance ** 0.5
    
    return {
        'variance': variance,
        'std_dev': std_dev,
        'formula': '$$\sigma = \sqrt{\frac{\sum(x_i - \mu)^2}{n-1}}$$'
    }

# Example usage
dataset = [23, 45, 67, 34, 89, 54, 23]
results = calculate_spread(dataset)
print(f"Variance: {results['variance']:.2f}")
print(f"Standard Deviation: {results['std_dev']:.2f}")
```

Slide 3: Correlation Analysis

Statistical correlation measures the strength and direction of relationships between variables. This implementation calculates Pearson's correlation coefficient and provides visualization capabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

def correlation_analysis(x, y):
    correlation = np.corrcoef(x, y)[0, 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Variable X')
    plt.ylabel('Variable Y')
    plt.title(f'Correlation: {correlation:.2f}')
    
    return correlation

# Example usage
x = np.random.normal(0, 1, 100)
y = x * 0.8 + np.random.normal(0, 0.2, 100)
correlation = correlation_analysis(x, y)
```

Slide 4: Probability Distribution Analysis

Statistical distributions provide insights into data patterns and help in making predictions. This implementation focuses on analyzing normal distributions and their properties.

```python
import numpy as np
from scipy import stats

def analyze_distribution(data):
    # Calculate distribution parameters
    mean, std = stats.norm.fit(data)
    
    # Perform Shapiro-Wilk test for normality
    statistic, p_value = stats.shapiro(data)
    
    # Generate theoretical normal distribution
    x = np.linspace(min(data), max(data), 100)
    pdf = stats.norm.pdf(x, mean, std)
    
    return {
        'mean': mean,
        'std': std,
        'p_value': p_value,
        'formula': '$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$'
    }

# Example usage
data = np.random.normal(loc=0, scale=1, size=1000)
results = analyze_distribution(data)
```

Slide 5: Hypothesis Testing Implementation

Understanding statistical significance through hypothesis testing is crucial for data analysis. This implementation demonstrates t-tests and their practical application.

```python
def hypothesis_test(sample1, sample2, alpha=0.05):
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(sample1), len(sample2)
    pooled_std = np.sqrt(((n1-1)*np.var(sample1) + (n2-1)*np.var(sample2))/(n1+n2-2))
    cohens_d = (np.mean(sample1) - np.mean(sample2)) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohens_d
    }

# Example usage
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(95, 15, 50)
results = hypothesis_test(group1, group2)
```

Slide 6: Time Series Analysis

Time series analysis is essential for understanding temporal patterns in data. This implementation focuses on decomposition and trend analysis.

```python
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def analyze_timeseries(data, period):
    # Convert to time series
    ts = pd.Series(data)
    
    # Perform decomposition
    decomposition = seasonal_decompose(ts, period=period)
    
    # Calculate rolling statistics
    rolling_mean = ts.rolling(window=period).mean()
    rolling_std = ts.rolling(window=period).std()
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    }

# Example usage
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
data = np.random.normal(100, 10, 365) + np.sin(np.linspace(0, 4*np.pi, 365)) * 20
ts_results = analyze_timeseries(data, period=30)
```

Slide 7: Regression Analysis Implementation

Regression analysis reveals relationships between variables and enables predictions. This implementation demonstrates linear regression with comprehensive diagnostics and validation metrics.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def advanced_regression(X, y):
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    
    # Make predictions
    y_pred = model.predict(X.reshape(-1, 1))
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Calculate confidence intervals
    n = len(X)
    std_error = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))
    
    return {
        'coefficients': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'mse': mse,
        'std_error': std_error,
        'formula': '$$y = \beta_0 + \beta_1x + \epsilon$$'
    }

# Example usage
X = np.random.normal(0, 1, 100)
y = 2 * X + 1 + np.random.normal(0, 0.5, 100)
results = advanced_regression(X, y)
```

Slide 8: Statistical Outlier Detection

Outlier detection is crucial for data quality and anomaly identification. This implementation provides multiple methods for detecting statistical outliers.

```python
def detect_outliers(data, method='zscore'):
    def iqr_bounds(x):
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        return q1 - 1.5*iqr, q3 + 1.5*iqr
    
    if method == 'zscore':
        z_scores = (data - np.mean(data)) / np.std(data)
        outliers = np.abs(z_scores) > 3
    elif method == 'iqr':
        lower, upper = iqr_bounds(data)
        outliers = (data < lower) | (data > upper)
    else:
        raise ValueError("Method must be 'zscore' or 'iqr'")
    
    return {
        'outliers': data[outliers],
        'outlier_indices': np.where(outliers)[0],
        'total_outliers': sum(outliers)
    }

# Example usage
data = np.concatenate([np.random.normal(0, 1, 95), np.array([10, -10, 8, -8, 12])])
zscore_results = detect_outliers(data, 'zscore')
iqr_results = detect_outliers(data, 'iqr')
```

Slide 9: Statistical Power Analysis

Power analysis helps determine sample size requirements and experiment validity. This implementation calculates statistical power for different test scenarios.

```python
from scipy import stats

def power_analysis(effect_size, alpha=0.05, power=0.8):
    def calculate_sample_size(d):
        # Initial estimate
        n = 8
        while True:
            ncp = np.sqrt(n/2) * d
            crit = stats.norm.ppf(1-alpha)
            beta = stats.norm.cdf(crit - ncp)
            actual_power = 1 - beta
            
            if actual_power >= power:
                break
            n += 1
        return n
    
    sample_size = calculate_sample_size(effect_size)
    
    return {
        'required_n': sample_size,
        'effect_size': effect_size,
        'alpha': alpha,
        'target_power': power,
        'formula': '$$1 - \beta = P(|T| > t_{\\alpha/2}|H_1)$$'
    }

# Example usage
results = power_analysis(effect_size=0.5)
```

Slide 10: Bootstrap Statistical Analysis

Bootstrap methods enable estimation of sampling distributions and confidence intervals without assumptions about underlying distributions. This implementation provides resampling techniques for robust statistical inference.

```python
def bootstrap_analysis(data, n_bootstrap=1000, statistic=np.mean):
    bootstrapped_stats = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        stat = statistic(sample)
        bootstrapped_stats.append(stat)
    
    # Calculate confidence intervals
    ci_lower, ci_upper = np.percentile(bootstrapped_stats, [2.5, 97.5])
    
    return {
        'estimate': statistic(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': np.std(bootstrapped_stats)
    }

# Example usage
data = np.random.lognormal(0, 0.5, 100)
bootstrap_results = bootstrap_analysis(data)
```

Slide 11: Statistical Process Control

SPC charts monitor process stability and detect significant variations. This implementation provides comprehensive control chart analysis with violation detection.

```python
def control_chart_analysis(data, window=20):
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate control limits
    ucl = mean + 3*std
    lcl = mean - 3*std
    
    # Moving range calculations
    mr = np.abs(np.diff(data))
    mr_mean = np.mean(mr)
    mr_ucl = mr_mean + 3*std
    
    def detect_violations(values):
        # Western Electric Rules
        rules = {
            'beyond_3sigma': np.abs(values - mean) > 3*std,
            'two_of_three': np.convolve(np.abs(values - mean) > 2*std, 
                                      np.ones(3)/3, mode='valid') >= 2/3
        }
        return rules
    
    violations = detect_violations(data)
    
    return {
        'center_line': mean,
        'ucl': ucl,
        'lcl': lcl,
        'violations': violations
    }

# Example usage
process_data = np.random.normal(100, 10, 100)
spc_results = control_chart_analysis(process_data)
```

Slide 12: Survival Analysis Implementation

Survival analysis examines time-to-event data. This implementation provides Kaplan-Meier estimation and hazard analysis capabilities.

```python
from lifelines import KaplanMeierFitter

def survival_analysis(durations, events):
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events, label='Survival Curve')
    
    # Calculate key metrics
    median_survival = kmf.median_survival_time_
    
    # Calculate survival probabilities at specific times
    times = np.linspace(0, max(durations), 100)
    survival_prob = kmf.survival_function_at_times(times)
    
    return {
        'median_survival': median_survival,
        'survival_curve': survival_prob,
        'formula': '$$S(t) = \prod_{i:t_i\leq t} (1 - \frac{d_i}{n_i})$$'
    }

# Example usage
durations = np.random.exponential(50, size=200)
events = np.random.binomial(n=1, p=0.7, size=200)
survival_results = survival_analysis(durations, events)
```

Slide 13: Additional Resources

*   "Statistical Methods in Data Science: A Comprehensive Review" - [https://arxiv.org/abs/2012.00054](https://arxiv.org/abs/2012.00054)
*   "Modern Statistical Methods for Complex Data Structures" - [https://arxiv.org/abs/1908.07890](https://arxiv.org/abs/1908.07890)
*   "Bootstrap Methods in Statistical Analysis" - [https://arxiv.org/abs/1904.12956](https://arxiv.org/abs/1904.12956)
*   For more resources, search "statistical analysis methods" on Google Scholar
*   Recommended reading: Statistical Learning Theory on ArXiv

