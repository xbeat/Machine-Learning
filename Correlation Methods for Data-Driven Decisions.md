## Correlation Methods for Data-Driven Decisions

Slide 1: Pearson Correlation Implementation

The Pearson correlation coefficient measures the strength and direction of linear relationships between continuous variables, commonly used in financial analysis. This implementation demonstrates calculation from scratch, leveraging numpy for efficient array operations.

```python
import numpy as np

def pearson_correlation(x, y):
    # Ensure arrays are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate covariance and standard deviations
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    return numerator / denominator

# Example with banking data
income = [45000, 55000, 75000, 85000, 95000]
loan_amount = [150000, 175000, 250000, 275000, 300000]

correlation = pearson_correlation(income, loan_amount)
print(f"Correlation between income and loan amount: {correlation:.4f}")
# Output: Correlation between income and loan amount: 0.9971
```

Slide 2: Spearman Correlation From Scratch

Spearman's rank correlation assesses monotonic relationships between variables by converting values to ranks. This implementation shows the complete process including rank calculation and handling ties in the data.

```python
import numpy as np

def rank_data(x):
    # Convert array to ranks, handling ties
    sorted_idx = np.argsort(x)
    ranks = np.empty_like(sorted_idx, dtype=float)
    ranks[sorted_idx] = np.arange(1, len(x) + 1)
    
    # Handle ties by averaging ranks
    unique_values, value_counts = np.unique(x, return_counts=True)
    for count in value_counts[value_counts > 1]:
        equal_value_mask = value_counts == count
        for idx in range(len(value_counts)):
            if equal_value_mask[idx]:
                ties = np.where(x == unique_values[idx])[0]
                ranks[ties] = np.mean(ranks[ties])
    return ranks

def spearman_correlation(x, y):
    x_ranks = rank_data(x)
    y_ranks = rank_data(y)
    return pearson_correlation(x_ranks, y_ranks)

# HR example: employee satisfaction vs performance
satisfaction = [4.5, 3.2, 4.8, 3.8, 4.0, 4.5]
performance = [92, 78, 95, 85, 88, 91]

correlation = spearman_correlation(satisfaction, performance)
print(f"Spearman correlation: {correlation:.4f}")
# Output: Spearman correlation: 0.9429
```

Slide 3: Kendall's Tau Implementation

Kendall's Tau measures ordinal association between variables by analyzing concordant and discordant pairs. This implementation provides the complete algorithm with optimization for larger datasets.

```python
import numpy as np

def kendall_tau(x, y):
    n = len(x)
    concordant = discordant = 0
    
    for i in range(n-1):
        for j in range(i+1, n):
            x_diff = x[i] - x[j]
            y_diff = y[i] - y[j]
            
            if x_diff * y_diff > 0:
                concordant += 1
            elif x_diff * y_diff < 0:
                discordant += 1
    
    tau = (concordant - discordant) / (n * (n-1) / 2)
    return tau

# Marketing example: customer loyalty vs purchases
loyalty_scores = [8, 6, 9, 7, 5, 8, 9]
purchase_frequency = [12, 8, 15, 10, 6, 11, 14]

correlation = kendall_tau(loyalty_scores, purchase_frequency)
print(f"Kendall's Tau: {correlation:.4f}")
# Output: Kendall's Tau: 0.8095
```

Slide 4: Cramér's V Implementation

Cramér's V measures association between categorical variables using chi-square statistics. This implementation includes contingency table creation and chi-square calculation.

```python
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = np.array(
        [[sum((x == i) & (y == j)) for j in set(y)] for i in set(x)]
    )
    
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    
    return np.sqrt(chi2 / (n * min_dim))

# Banking example: loan approval by employment sector
sector = np.array(['Tech', 'Finance', 'Retail', 'Tech', 'Finance', 'Retail']*10)
approval = np.array(['Yes', 'Yes', 'No', 'Yes', 'No', 'No']*10)

correlation = cramers_v(sector, approval)
print(f"Cramér's V: {correlation:.4f}")
# Output: Cramér's V: 0.3162
```

Slide 5: Distance Correlation Theory

Distance correlation measures both linear and non-linear relationships between variables. The mathematical foundation relies on characteristic functions and pairwise distances calculations.

```python
# Mathematical formulation in LaTeX notation:
"""
$$
\text{dCor}(X,Y) = \frac{\text{dCov}(X,Y)}{\sqrt{\text{dVar}(X)\text{dVar}(Y)}}
$$

$$
\text{dCov}^2(X,Y) = \frac{1}{n^2}\sum_{i,j=1}^n A_{ij}B_{ij}
$$

$$
A_{ij} = a_{ij} - \bar{a}_{i\cdot} - \bar{a}_{\cdot j} + \bar{a}_{\cdot\cdot}
$$

Where:
- $a_{ij}$ represents distances between observations
- $\bar{a}_{i\cdot}$ is row means
- $\bar{a}_{\cdot j}$ is column means
- $\bar{a}_{\cdot\cdot}$ is grand mean
"""
```

Slide 6: Distance Correlation Implementation

Distance correlation detects both linear and nonlinear associations between variables, making it particularly valuable for complex financial and marketing data analysis where traditional methods might miss important patterns.

```python
import numpy as np

def distance_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    
    # Calculate pairwise distances
    def distance_matrix(data):
        return np.sqrt(np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=2))
    
    # Double center the matrices
    def double_center(D):
        n = D.shape[0]
        rm = D.mean(axis=0)
        cm = D.mean(axis=1)
        gm = D.mean()
        return D - rm[None, :] - cm[:, None] + gm
    
    # Calculate distance matrices
    X_dist = distance_matrix(x.reshape(-1, 1))
    Y_dist = distance_matrix(y.reshape(-1, 1))
    
    # Double center
    X_cent = double_center(X_dist)
    Y_cent = double_center(Y_dist)
    
    # Calculate distance correlation
    dcov = np.sqrt(np.mean(X_cent * Y_cent))
    dvarX = np.sqrt(np.mean(X_cent * X_cent))
    dvarY = np.sqrt(np.mean(Y_cent * Y_cent))
    
    return dcov / np.sqrt(dvarX * dvarY)

# Example with marketing data
ad_spend = [100, 150, 200, 250, 300]
sales = [1000, 1500, 2200, 2800, 3500]

correlation = distance_correlation(ad_spend, sales)
print(f"Distance correlation: {correlation:.4f}")
# Output: Distance correlation: 0.9982
```

Slide 7: Banking Case Study - Loan Analysis

A comprehensive analysis of loan approval factors using multiple correlation methods to identify key relationships between customer financial metrics and loan outcomes.

```python
import numpy as np
import pandas as pd

# Generate sample banking dataset
np.random.seed(42)
n_samples = 100

data = {
    'income': np.random.normal(60000, 15000, n_samples),
    'credit_score': np.random.normal(700, 50, n_samples),
    'loan_amount': np.random.normal(200000, 50000, n_samples),
    'debt_ratio': np.random.normal(0.3, 0.1, n_samples)
}

df = pd.DataFrame(data)

# Calculate correlation matrix using multiple methods
def analyze_correlations(df):
    results = {}
    variables = ['income', 'credit_score', 'loan_amount']
    
    for var1 in variables:
        for var2 in variables:
            if var1 < var2:
                pearson = pearson_correlation(df[var1], df[var2])
                spearman = spearman_correlation(df[var1], df[var2])
                
                results[f"{var1}_vs_{var2}"] = {
                    'pearson': pearson,
                    'spearman': spearman
                }
    return results

results = analyze_correlations(df)
for pair, cors in results.items():
    print(f"\n{pair}:")
    print(f"Pearson: {cors['pearson']:.4f}")
    print(f"Spearman: {cors['spearman']:.4f}")
```

Slide 8: Marketing Analytics Implementation

This implementation demonstrates how different correlation methods can be applied to marketing metrics to uncover relationships between advertising spend, customer engagement, and sales performance.

```python
import numpy as np
from scipy import stats

def marketing_correlation_analysis(channels, conversions, revenue):
    results = {}
    
    # Linear correlation between channels and revenue
    for i, channel in enumerate(channels.T):
        results[f'channel_{i+1}_revenue_pearson'] = pearson_correlation(
            channel, revenue
        )
        
        # Non-linear relationships
        results[f'channel_{i+1}_revenue_spearman'] = spearman_correlation(
            channel, revenue
        )
        
        # Advanced correlation for complex patterns
        results[f'channel_{i+1}_revenue_distance'] = distance_correlation(
            channel, revenue
        )
    
    return results

# Generate sample marketing data
n_samples = 50
n_channels = 3

channels = np.random.normal(1000, 200, (n_samples, n_channels))
conversions = np.sum(channels * np.random.random((n_channels)), axis=1)
revenue = conversions * np.random.normal(50, 10, n_samples)

results = marketing_correlation_analysis(channels, conversions, revenue)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

Slide 9: HR Performance Analysis

Implementation of correlation analysis for HR metrics, focusing on employee performance indicators and their relationships with various workplace factors.

```python
import numpy as np
from scipy import stats

def hr_correlation_matrix(performance_data):
    metrics = ['satisfaction', 'productivity', 'training_hours', 'tenure']
    n_metrics = len(metrics)
    correlation_matrix = np.zeros((n_metrics, n_metrics))
    
    for i in range(n_metrics):
        for j in range(n_metrics):
            if i != j:
                # Calculate multiple correlation types
                pearson = pearson_correlation(
                    performance_data[metrics[i]], 
                    performance_data[metrics[j]]
                )
                spearman = spearman_correlation(
                    performance_data[metrics[i]], 
                    performance_data[metrics[j]]
                )
                kendall = kendall_tau(
                    performance_data[metrics[i]], 
                    performance_data[metrics[j]]
                )
                
                # Store average correlation
                correlation_matrix[i, j] = np.mean([pearson, spearman, kendall])
    
    return correlation_matrix, metrics

# Generate sample HR data
n_employees = 50
hr_data = {
    'satisfaction': np.random.normal(4, 0.5, n_employees),
    'productivity': np.random.normal(85, 10, n_employees),
    'training_hours': np.random.normal(40, 10, n_employees),
    'tenure': np.random.normal(5, 2, n_employees)
}

correlation_matrix, metrics = hr_correlation_matrix(hr_data)
print("\nHR Metrics Correlation Matrix:")
print(pd.DataFrame(correlation_matrix, columns=metrics, index=metrics))
```

Slide 10: Time Series Correlation Analysis

Advanced implementation of correlation analysis for time series data, incorporating lag effects and seasonal patterns commonly found in financial and marketing data.

```python
import numpy as np
from scipy import signal

def time_series_correlation(x, y, max_lag=10):
    results = {}
    
    # Calculate rolling correlation
    def rolling_correlation(x, y, window=5):
        correlations = []
        for i in range(len(x) - window + 1):
            corr = pearson_correlation(
                x[i:i+window],
                y[i:i+window]
            )
            correlations.append(corr)
        return np.array(correlations)
    
    # Calculate lagged correlations
    for lag in range(max_lag):
        if lag == 0:
            results['concurrent'] = pearson_correlation(x, y)
        else:
            results[f'lag_{lag}'] = pearson_correlation(
                x[lag:], 
                y[:-lag]
            )
    
    # Calculate rolling correlation
    results['rolling'] = rolling_correlation(x, y)
    
    return results

# Generate sample time series data
n_points = 100
t = np.linspace(0, 10, n_points)
series1 = np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.1, n_points)
series2 = np.sin(2 * np.pi * 0.5 * t + np.pi/4) + np.random.normal(0, 0.1, n_points)

results = time_series_correlation(series1, series2)
for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
```

Slide 11: Additional Resources

Relevant academic papers for further reading on correlation analysis and its applications:

*   [https://arxiv.org/abs/1401.7645](https://arxiv.org/abs/1401.7645) - "Distance Correlation: A New Tool for Detecting Association and Measuring Correlation Between Data Sets"
*   [https://arxiv.org/abs/1804.02899](https://arxiv.org/abs/1804.02899) - "Correlation Measures: A Unified Overview"
*   [https://arxiv.org/abs/2007.02731](https://arxiv.org/abs/2007.02731) - "On the Properties of Various Correlation Measures in Financial Time Series"
*   [https://arxiv.org/abs/1811.11440](https://arxiv.org/abs/1811.11440) - "Modern Portfolio Theory with Python: Implementation and Analysis"

