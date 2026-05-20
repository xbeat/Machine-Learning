## Differences in Standard Deviation Calculations Pandas vs NumPy
Slide 1: Understanding Standard Deviation Differences

Statistical computations in Python can yield different results depending on the library used. The key distinction between NumPy and Pandas lies in their default behavior regarding degrees of freedom (ddof) parameter when calculating standard deviation.

```python
import numpy as np
import pandas as pd

# Create sample data
data = [2, 4, 4, 4, 5, 5, 7, 9]

# NumPy std calculation (default ddof=0)
np_std = np.std(data)

# Pandas std calculation (default ddof=1)
pd_std = pd.Series(data).std()

print(f"NumPy std (ddof=0): {np_std:.6f}")
print(f"Pandas std (ddof=1): {pd_std:.6f}")
```

Slide 2: Mathematical Foundation

The fundamental difference stems from the population versus sample standard deviation formulas. The mathematical expressions showcase how degrees of freedom impacts the final calculation.

```python
# Mathematical formulas in LaTeX notation (not rendered)
$$\text{Population SD} = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}$$

$$\text{Sample SD} = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \bar{x})^2}{N-1}}$$

# Implementation from scratch
def calculate_std(data, ddof=0):
    mean = sum(data) / len(data)
    squared_diff_sum = sum((x - mean) ** 2 for x in data)
    return (squared_diff_sum / (len(data) - ddof)) ** 0.5
```

Slide 3: Population Standard Deviation

The population standard deviation assumes we have complete data about the entire population. NumPy's default implementation (ddof=0) uses this approach, dividing by N in the denominator.

```python
import numpy as np

data = [2, 4, 4, 4, 5, 5, 7, 9]

# NumPy population std
pop_std = np.std(data, ddof=0)

# Custom implementation
def population_std(data):
    mean = np.mean(data)
    squared_diff = [(x - mean) ** 2 for x in data]
    return np.sqrt(sum(squared_diff) / len(data))

print(f"NumPy population std: {pop_std:.6f}")
print(f"Custom population std: {population_std(data):.6f}")
```

Slide 4: Sample Standard Deviation

When working with sample data, statisticians often prefer using N-1 degrees of freedom (Bessel's correction). Pandas adopts this convention by default, which explains the different results.

```python
import pandas as pd

data = [2, 4, 4, 4, 5, 5, 7, 9]

# Pandas sample std
sample_std = pd.Series(data).std()

# Custom implementation
def sample_std(data):
    mean = sum(data) / len(data)
    squared_diff = [(x - mean) ** 2 for x in data]
    return np.sqrt(sum(squared_diff) / (len(data) - 1))

print(f"Pandas sample std: {sample_std:.6f}")
print(f"Custom sample std: {sample_std(data):.6f}")
```

Slide 5: Real-world Example: Stock Price Analysis

Financial analysts often use standard deviation to measure market volatility. This example demonstrates how different std calculations affect risk assessment.

```python
import numpy as np
import pandas as pd

# Sample daily stock returns
stock_returns = [0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.03, 0.02, 0.01, -0.01]

# Calculate volatility using both methods
np_volatility = np.std(stock_returns) * np.sqrt(252)  # Annualized
pd_volatility = pd.Series(stock_returns).std() * np.sqrt(252)

print(f"NumPy Annualized Volatility: {np_volatility:.4f}")
print(f"Pandas Annualized Volatility: {pd_volatility:.4f}")
```

Slide 6: Impact on Research Analysis

The choice between population and sample standard deviation significantly impacts research conclusions, especially in small datasets. Understanding these differences is crucial for accurate statistical inference and experimental design.

```python
import numpy as np
import pandas as pd

# Compare impact on different sample sizes
sample_sizes = [5, 10, 30, 100]

for size in sample_sizes:
    data = np.random.normal(0, 1, size)
    np_std = np.std(data)
    pd_std = pd.Series(data).std()
    diff_percent = ((pd_std - np_std) / np_std) * 100
    
    print(f"Sample size: {size}")
    print(f"NumPy std: {np_std:.6f}")
    print(f"Pandas std: {pd_std:.6f}")
    print(f"Difference: {diff_percent:.2f}%\n")
```

Slide 7: Healthcare Data Analysis Example

Real-world application demonstrating the impact of different standard deviation calculations on patient vital signs monitoring and clinical decision-making processes.

```python
import numpy as np
import pandas as pd

# Simulated patient blood pressure readings
bp_readings = [120, 122, 118, 125, 119, 121, 123, 120]

def analyze_vitals(data):
    np_std = np.std(data, ddof=0)
    pd_std = pd.Series(data).std()
    
    # Calculate reference ranges
    np_range = (np.mean(data) - 2*np_std, np.mean(data) + 2*np_std)
    pd_range = (np.mean(data) - 2*pd_std, np.mean(data) + 2*pd_std)
    
    return {
        'pop_std': np_std,
        'sample_std': pd_std,
        'pop_range': np_range,
        'sample_range': pd_range
    }

results = analyze_vitals(bp_readings)
for key, value in results.items():
    print(f"{key}: {value}")
```

Slide 8: Effect of Sample Size on Standard Deviation

A comprehensive analysis of how sample size affects the difference between population and sample standard deviation calculations, with visualization code.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_size_effect(min_size=5, max_size=100, steps=20):
    sizes = np.linspace(min_size, max_size, steps, dtype=int)
    differences = []
    
    for size in sizes:
        data = np.random.normal(0, 1, size)
        np_std = np.std(data, ddof=0)
        pd_std = pd.Series(data).std()
        diff = ((pd_std - np_std) / np_std) * 100
        differences.append(diff)
    
    return sizes, differences

sizes, diffs = analyze_size_effect()
print("Sample Size | Difference (%)")
print("-" * 25)
for size, diff in zip(sizes, diffs):
    print(f"{size:10d} | {diff:12.2f}")
```

Slide 9: Handling Missing Values

Different standard deviation calculations handle missing values differently, which can significantly impact analysis results in real-world datasets.

```python
import numpy as np
import pandas as pd

# Dataset with missing values
data_with_nan = [1, 2, np.nan, 4, 5, 6, np.nan, 8]

# NumPy approach
np_clean = np.array(data_with_nan)[~np.isnan(data_with_nan)]
np_std = np.std(np_clean)

# Pandas approach
pd_std = pd.Series(data_with_nan).std()

# Custom implementation with missing value handling
def robust_std(data, ddof=1):
    clean_data = [x for x in data if not pd.isna(x)]
    mean = sum(clean_data) / len(clean_data)
    squared_diff = [(x - mean) ** 2 for x in clean_data]
    return np.sqrt(sum(squared_diff) / (len(clean_data) - ddof))

print(f"NumPy std: {np_std:.6f}")
print(f"Pandas std: {pd_std:.6f}")
print(f"Custom robust std: {robust_std(data_with_nan):.6f}")
```

Slide 10: Parallel Computing Considerations

Standard deviation calculations in distributed computing environments require special attention to maintain numerical stability and accuracy across different computation methods.

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import math

def parallel_std(data, chunks=4, ddof=1):
    chunk_size = math.ceil(len(data) / chunks)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    def chunk_stats(chunk):
        return len(chunk), np.sum(chunk), np.sum(np.square(chunk))
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(chunk_stats, chunks))
    
    total_n = sum(r[0] for r in results)
    total_sum = sum(r[1] for r in results)
    total_sq_sum = sum(r[2] for r in results)
    
    mean = total_sum / total_n
    variance = (total_sq_sum - (total_sum ** 2) / total_n) / (total_n - ddof)
    return np.sqrt(variance)

# Example usage
data = np.random.normal(0, 1, 1000000)
print(f"Parallel std: {parallel_std(data.tolist()):.6f}")
print(f"Pandas std: {pd.Series(data).std():.6f}")
```

Slide 11: Time Series Standard Deviation

Time series data requires special consideration when calculating standard deviation, as temporal dependencies can affect the interpretation of variability measurements.

```python
import numpy as np
import pandas as pd

# Create time series data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.random.normal(100, 15, 100)
ts_data = pd.Series(values, index=dates)

# Calculate rolling standard deviation
def analyze_time_series_std(data, windows=[7, 14, 30]):
    results = {}
    for window in windows:
        # NumPy approach (manual rolling)
        np_rolling = [np.std(values[max(0, i-window):i]) 
                     for i in range(1, len(values)+1)]
        
        # Pandas approach
        pd_rolling = data.rolling(window=window).std()
        
        results[f'window_{window}'] = {
            'numpy': np_rolling[-1],
            'pandas': pd_rolling.iloc[-1]
        }
    
    return results

results = analyze_time_series_std(ts_data)
for window, stats in results.items():
    print(f"\n{window}:")
    print(f"NumPy rolling std: {stats['numpy']:.6f}")
    print(f"Pandas rolling std: {stats['pandas']:.6f}")
```

Slide 12: Weighted Standard Deviation

When observations have different importance levels, weighted standard deviation provides a more accurate measure of dispersion considering the relative significance of each data point.

```python
import numpy as np
import pandas as pd

def weighted_std(values, weights, ddof=1):
    """
    Calculate weighted standard deviation with specified degrees of freedom
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance * len(weights) / (len(weights) - ddof))

# Example with student grades and credit weights
grades = [85, 92, 78, 90, 88]
credits = [3, 4, 2, 4, 3]

# Calculate using different methods
numpy_weighted = np.sqrt(np.cov(grades, aweights=credits))
custom_weighted = weighted_std(grades, credits)
simple_std = pd.Series(grades).std()

print(f"Weighted std (custom): {custom_weighted:.6f}")
print(f"Weighted std (numpy): {numpy_weighted[0][0]:.6f}")
print(f"Unweighted std: {simple_std:.6f}")
```

Slide 13: Robust Standard Deviation

Real-world data often contains outliers that can significantly impact standard deviation calculations. Robust methods provide more reliable measures of variability in such cases.

```python
import numpy as np
from scipy import stats

def robust_statistics(data):
    # Regular std
    standard_std = np.std(data, ddof=1)
    
    # Median Absolute Deviation (MAD)
    mad = stats.median_abs_deviation(data)
    
    # Interquartile Range based std
    q75, q25 = np.percentile(data, [75, 25])
    iqr_std = (q75 - q25) / 1.349
    
    # Trimmed std (removing 10% from each end)
    trimmed_std = stats.trim_mean(np.square(data - np.mean(data)), 0.1) ** 0.5
    
    return {
        'standard': standard_std,
        'mad': mad,
        'iqr_based': iqr_std,
        'trimmed': trimmed_std
    }

# Example with outliers
data_with_outliers = [1, 2, 2, 3, 3, 4, 4, 100]
results = robust_statistics(data_with_outliers)

for method, value in results.items():
    print(f"{method} std: {value:.6f}")
```

Slide 14: Additional Resources

*   [https://arxiv.org/abs/1906.07101](https://arxiv.org/abs/1906.07101) - "A New Look at Standard Deviation: Generalizing to Weighted Observations" 
*   [https://arxiv.org/abs/1811.02891](https://arxiv.org/abs/1811.02891) - "Robust Statistics for Outlier Detection in Big Data" 
*   [https://arxiv.org/abs/2003.06663](https://arxiv.org/abs/2003.06663) - "On the Choice of the Number of Degrees of Freedom in Statistical Estimation" 
*   [https://arxiv.org/abs/1712.04788](https://arxiv.org/abs/1712.04788) - "Statistical Analysis of Time Series Data: A Comprehensive Guide" 
*   [https://arxiv.org/abs/1902.06021](https://arxiv.org/abs/1902.06021) - "Efficient Computation of Standard Deviation in Distributed Systems"

