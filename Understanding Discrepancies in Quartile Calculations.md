## Understanding Discrepancies in Quartile Calculations
Slide 1: Understanding Quartile Calculation Methods

The fundamental difference between manual and Python quartile calculations lies in their underlying methodologies. Manual calculations often use position-based formulas, while Python's numpy implements linear interpolation between closest ranks by default.

```python
import numpy as np

# Sample dataset
data = [2, 4, 6, 8, 10, 12, 14]

# Manual quartile calculation
def manual_quartiles(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    # Position calculations
    q1_pos = (n + 1) / 4
    q2_pos = (n + 1) / 2
    q3_pos = 3 * (n + 1) / 4
    
    # Linear interpolation for non-integer positions
    def get_value(pos):
        if pos.is_integer():
            return sorted_data[int(pos) - 1]
        lower = sorted_data[int(pos) - 1]
        upper = sorted_data[int(pos)]
        fraction = pos - int(pos)
        return lower + fraction * (upper - lower)
    
    return get_value(q1_pos), get_value(q2_pos), get_value(q3_pos)

# Compare with numpy
manual_q1, manual_q2, manual_q3 = manual_quartiles(data)
numpy_q1, numpy_q2, numpy_q3 = np.quantile(data, [0.25, 0.5, 0.75])

print(f"Manual Q1, Q2, Q3: {manual_q1:.2f}, {manual_q2:.2f}, {manual_q3:.2f}")
print(f"Numpy  Q1, Q2, Q3: {numpy_q1:.2f}, {numpy_q2:.2f}, {numpy_q3:.2f}")
```

Slide 2: Different Quartile Methods in NumPy

NumPy offers nine different methods for computing quartiles through the interpolation parameter, each producing slightly different results based on their statistical approach and handling of edge cases.

```python
import numpy as np

data = [2, 4, 6, 8, 10, 12, 14]
methods = ['linear', 'lower', 'higher', 'midpoint', 'nearest']

print("Quartile calculations using different methods:")
for method in methods:
    q1, q2, q3 = np.quantile(data, [0.25, 0.5, 0.75], 
                            interpolation=method)
    print(f"\nMethod: {method}")
    print(f"Q1: {q1:.2f}, Q2: {q2:.2f}, Q3: {q3:.2f}")
```

Slide 3: Mathematical Foundation of Quartile Calculations

Understanding the mathematical basis for quartile calculations helps explain the variations between different methods. The key lies in how we handle non-integer position values.

```python
# Mathematical representation of quartile calculations
"""
Manual Method Formulas:
$$Q1_{pos} = \frac{n + 1}{4}$$
$$Q2_{pos} = \frac{n + 1}{2}$$
$$Q3_{pos} = \frac{3(n + 1)}{4}$$

Linear Interpolation Formula:
$$Q_i = x_j + (x_{j+1} - x_j) \times (p - j)$$
where:
- $x_j$ is the value at position j
- $p$ is the target position
- $j$ is the floor of $p$
"""

def theoretical_quartile_pos(n):
    q1_pos = (n + 1) / 4
    q2_pos = (n + 1) / 2
    q3_pos = 3 * (n + 1) / 4
    return q1_pos, q2_pos, q3_pos

# Example with n = 10
n = 10
q1, q2, q3 = theoretical_quartile_pos(n)
print(f"For n={n}:")
print(f"Q1 position: {q1:.2f}")
print(f"Q2 position: {q2:.2f}")
print(f"Q3 position: {q3:.2f}")
```

Slide 4: Implementing Type 7 Quartile Method

The Type 7 method, used by default in NumPy, implements a specific interpolation strategy that differs from traditional manual calculations but offers consistent results across different dataset sizes.

```python
def type7_quartiles(data):
    """
    Implements Type 7 quartile calculation method
    p(k) = (k - 1)/(n - 1)
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    def get_quantile(p):
        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
            
        if p == 0:
            return sorted_data[0]
        if p == 1:
            return sorted_data[-1]
            
        h = (n - 1) * p
        i = int(h)
        return sorted_data[i] + (h - i) * (sorted_data[i + 1] - sorted_data[i])
    
    return (get_quantile(0.25), 
            get_quantile(0.50), 
            get_quantile(0.75))

# Example usage
data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
q1, q2, q3 = type7_quartiles(data)
print(f"Type 7 Quartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
```

Slide 5: Handling Edge Cases in Quartile Calculations

Edge cases, such as datasets with even or odd lengths, require special consideration. Different methods handle these cases differently, which can lead to varying results between manual and automated calculations.

```python
def compare_edge_cases():
    # Test cases for different dataset lengths
    odd_dataset = [1, 2, 3, 4, 5]
    even_dataset = [1, 2, 3, 4, 5, 6]
    
    def calculate_quartiles(data, method='linear'):
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Manual calculation
        def manual_method():
            q1_pos = (n + 1) / 4 - 1
            q2_pos = (n + 1) / 2 - 1
            q3_pos = 3 * (n + 1) / 4 - 1
            return q1_pos, q2_pos, q3_pos
        
        positions = manual_method()
        manual_results = [sorted_data[int(np.ceil(p))] for p in positions]
        numpy_results = np.quantile(data, [0.25, 0.5, 0.75], 
                                  interpolation=method)
        
        return manual_results, numpy_results

    # Compare results
    for dataset, name in [(odd_dataset, "Odd"), (even_dataset, "Even")]:
        manual, numpy_res = calculate_quartiles(dataset)
        print(f"\n{name} length dataset {dataset}")
        print(f"Manual: {manual}")
        print(f"NumPy:  {numpy_res}")

compare_edge_cases()
```

Slide 6: Real-world Example - Stock Price Analysis

Analyzing stock price quartiles helps identify market trends and outliers. This example demonstrates how different quartile calculation methods can affect financial analysis results.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Generate sample stock price data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
prices = np.random.normal(100, 15, 100).cumsum() + 1000

def analyze_stock_quartiles(dates, prices):
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    # Calculate quartiles using different methods
    methods = ['linear', 'midpoint']
    results = {}
    
    for method in methods:
        q1, q2, q3 = np.quantile(prices, [0.25, 0.5, 0.75], 
                                interpolation=method)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        results[method] = {
            'Q1': q1,
            'Median': q2,
            'Q3': q3,
            'IQR': iqr,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        }
    
    return results

results = analyze_stock_quartiles(dates, prices)
for method, values in results.items():
    print(f"\nMethod: {method}")
    for metric, value in values.items():
        print(f"{metric}: {value:.2f}")
```

Slide 7: Impact of Sample Size on Quartile Calculations

The relationship between sample size and quartile calculation accuracy reveals important considerations for choosing the appropriate method based on dataset characteristics.

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_sample_size_impact(min_size=10, max_size=1000, steps=5):
    sizes = np.logspace(np.log10(min_size), np.log10(max_size), steps).astype(int)
    results = []
    
    for size in sizes:
        # Generate random data
        data = np.random.normal(0, 1, size)
        
        # Calculate quartiles using different methods
        manual_q1 = np.percentile(data, 25, interpolation='linear')
        numpy_q1 = np.quantile(data, 0.25)
        
        # Calculate relative difference
        diff = abs(manual_q1 - numpy_q1) / abs(manual_q1)
        results.append({
            'size': size,
            'difference': diff
        })
    
    # Print results
    print("Sample Size Impact Analysis:")
    print("Size\tRelative Difference")
    print("-" * 30)
    for r in results:
        print(f"{r['size']}\t{r['difference']:.6f}")
    
    return results

results = analyze_sample_size_impact()
```

Slide 8: Quartile-based Outlier Detection

Implementing robust outlier detection using quartile-based methods requires understanding the nuances of different calculation approaches to ensure accurate anomaly identification.

```python
def quartile_outlier_detector(data, method='linear', threshold=1.5):
    """
    Implements quartile-based outlier detection with configurable method
    """
    q1, q3 = np.quantile(data, [0.25, 0.75], interpolation=method)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = []
    inliers = []
    
    for value in data:
        if value < lower_bound or value > upper_bound:
            outliers.append(value)
        else:
            inliers.append(value)
    
    stats = {
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Outlier Count': len(outliers),
        'Outlier Percentage': len(outliers) / len(data) * 100
    }
    
    return outliers, inliers, stats

# Example usage with synthetic data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(100, 15, 95),  # Normal data
    np.random.normal(200, 5, 5)     # Outliers
])

outliers, inliers, stats = quartile_outlier_detector(data)
for key, value in stats.items():
    print(f"{key}: {value:.2f}")
```

Slide 9: Performance Comparison of Different Methods

A systematic comparison of different quartile calculation methods reveals their computational efficiency and accuracy trade-offs, crucial for large-scale data processing applications.

```python
import time
import numpy as np
from typing import List, Tuple

def benchmark_quartile_methods(data_sizes: List[int], 
                             methods: List[str]) -> List[dict]:
    results = []
    
    for size in data_sizes:
        # Generate random data
        data = np.random.normal(0, 1, size)
        method_times = {}
        method_results = {}
        
        for method in methods:
            start_time = time.perf_counter()
            quartiles = np.quantile(data, [0.25, 0.5, 0.75], 
                                  interpolation=method)
            end_time = time.perf_counter()
            
            method_times[method] = end_time - start_time
            method_results[method] = quartiles
        
        results.append({
            'size': size,
            'times': method_times,
            'quartiles': method_results
        })
    
    # Print results
    for result in results:
        print(f"\nDataset size: {result['size']}")
        for method in methods:
            print(f"{method}:")
            print(f"  Time: {result['times'][method]:.6f} seconds")
            print(f"  Q1, Q2, Q3: {result['quartiles'][method]}")
            
    return results

# Benchmark different methods
data_sizes = [1000, 10000, 100000]
methods = ['linear', 'lower', 'higher', 'midpoint', 'nearest']
benchmark_results = benchmark_quartile_methods(data_sizes, methods)
```

Slide 10: Weighted Quartile Implementation

Implementing weighted quartiles provides a more sophisticated approach for datasets where observations have different importance levels or frequencies.

```python
def weighted_quartiles(values: np.ndarray, 
                      weights: np.ndarray, 
                      q: List[float]) -> np.ndarray:
    """
    Calculate weighted quartiles using cumulative weights
    
    Args:
        values: Array of values
        weights: Array of weights (same length as values)
        q: List of quantiles to compute (between 0 and 1)
    """
    # Sort values and weights together
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate cumulative weights
    cumsum_weights = np.cumsum(sorted_weights)
    total_weight = cumsum_weights[-1]
    
    # Normalize cumulative weights
    normalized_cumsum = cumsum_weights / total_weight
    
    # Calculate weighted quartiles
    weighted_quartiles = []
    for quantile in q:
        if quantile < 0 or quantile > 1:
            raise ValueError("Quantiles must be between 0 and 1")
            
        # Find indices where cumsum exceeds quantile
        exceed_indices = normalized_cumsum >= quantile
        if not exceed_indices.any():
            weighted_quartiles.append(sorted_values[-1])
        else:
            idx = exceed_indices.nonzero()[0][0]
            if idx == 0:
                weighted_quartiles.append(sorted_values[0])
            else:
                # Linear interpolation
                prev = idx - 1
                fraction = ((quantile - normalized_cumsum[prev]) / 
                          (normalized_cumsum[idx] - normalized_cumsum[prev]))
                value = (sorted_values[prev] + 
                        fraction * (sorted_values[idx] - sorted_values[prev]))
                weighted_quartiles.append(value)
    
    return np.array(weighted_quartiles)

# Example usage
values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
weights = np.array([1, 1, 2, 3, 4, 4, 3, 2, 1, 1])
quartiles = [0.25, 0.5, 0.75]

weighted_results = weighted_quartiles(values, weights, quartiles)
print("Weighted Quartiles:")
for q, result in zip(quartiles, weighted_results):
    print(f"Q{int(q*100)}: {result:.2f}")
```

Slide 11: Robust Quartile Estimation

Implementing robust quartile estimation techniques that handle outliers and non-normal distributions effectively while maintaining computational efficiency.

```python
def robust_quartile_estimation(data: np.ndarray, 
                             bootstrap_samples: int = 1000,
                             confidence_level: float = 0.95) -> dict:
    """
    Robust quartile estimation using bootstrap resampling
    """
    n_samples = len(data)
    quartile_bootstraps = np.zeros((bootstrap_samples, 3))
    
    for i in range(bootstrap_samples):
        # Bootstrap resampling
        bootstrap_sample = np.random.choice(data, 
                                          size=n_samples, 
                                          replace=True)
        quartile_bootstraps[i] = np.quantile(bootstrap_sample, 
                                           [0.25, 0.5, 0.75])
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(quartile_bootstraps, 
                            alpha/2 * 100, axis=0)
    ci_upper = np.percentile(quartile_bootstraps, 
                            (1-alpha/2) * 100, axis=0)
    
    # Calculate point estimates and standard errors
    point_estimates = np.mean(quartile_bootstraps, axis=0)
    standard_errors = np.std(quartile_bootstraps, axis=0)
    
    return {
        'point_estimates': point_estimates,
        'confidence_intervals': list(zip(ci_lower, ci_upper)),
        'standard_errors': standard_errors
    }

# Example usage
np.random.seed(42)
data = np.concatenate([
    np.random.normal(100, 15, 95),
    np.random.normal(200, 5, 5)  # Outliers
])

results = robust_quartile_estimation(data)
print("\nRobust Quartile Estimation Results:")
for i, q in enumerate(['Q1', 'Q2', 'Q3']):
    print(f"\n{q}:")
    print(f"Point estimate: {results['point_estimates'][i]:.2f}")
    print(f"95% CI: ({results['confidence_intervals'][i][0]:.2f}, "
          f"{results['confidence_intervals'][i][1]:.2f})")
    print(f"Standard error: {results['standard_errors'][i]:.2f}")
```

Slide 12: Real-time Streaming Quartile Calculation

Implementing an efficient streaming algorithm for calculating quartiles in real-time data streams requires maintaining a sorted summary structure while minimizing memory usage.

```python
class StreamingQuartiles:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.buffer = []
        self.sorted_buffer = []
        
    def update(self, value: float) -> dict:
        """
        Update streaming quartiles with new value
        """
        # Add new value and maintain window size
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
        # Update sorted buffer efficiently
        self.sorted_buffer = sorted(self.buffer)
        
        # Calculate quartiles
        quartiles = {}
        for q, label in zip([0.25, 0.5, 0.75], ['Q1', 'Q2', 'Q3']):
            pos = int(q * (len(self.sorted_buffer) - 1))
            quartiles[label] = self.sorted_buffer[pos]
            
        # Calculate IQR and bounds
        iqr = quartiles['Q3'] - quartiles['Q1']
        quartiles['IQR'] = iqr
        quartiles['Lower_Bound'] = quartiles['Q1'] - 1.5 * iqr
        quartiles['Upper_Bound'] = quartiles['Q3'] + 1.5 * iqr
        
        return quartiles

# Example usage with streaming data
import random
streamer = StreamingQuartiles(window_size=100)

# Simulate streaming data
print("Streaming Quartile Updates:")
for i in range(5):
    value = random.normalvariate(100, 15)
    results = streamer.update(value)
    print(f"\nUpdate {i+1} - New value: {value:.2f}")
    for key, val in results.items():
        print(f"{key}: {val:.2f}")
```

Slide 13: Fast Approximate Quartiles using P2 Algorithm

Implementation of the P2 algorithm for fast approximate quartile calculation, suitable for large-scale data processing with controlled error bounds.

```python
class P2Quartiles:
    def __init__(self, n_markers: int = 5):
        self.n = 0  # count of observations
        self.desired_positions = [0, 0.25, 0.5, 0.75, 1]
        self.positions = [1, 2, 3, 4, 5]  # marker positions
        self.heights = [0.0] * n_markers  # marker heights
        self.initialized = False
        
    def update(self, value: float) -> None:
        """
        Update P2 algorithm with new value
        """
        if not self.initialized:
            self.heights = [value] * len(self.positions)
            self.initialized = True
            return
            
        self.n += 1
        
        # Initial guess for positions
        desired = [self.n * p for p in self.desired_positions]
        
        # Update height
        if value < self.heights[0]:
            self.heights[0] = value
        elif value >= self.heights[-1]:
            self.heights[-1] = value
        else:
            for i in range(1, len(self.heights) - 1):
                if (self.heights[i-1] <= value) and (value < self.heights[i]):
                    self.heights[i] = value
                    
        # Update positions
        for i in range(1, len(self.positions) - 1):
            d = desired[i] - self.positions[i]
            if (d >= 1 and (self.positions[i+1] - self.positions[i] > 1)) or \
               (d <= -1 and (self.positions[i] - self.positions[i-1] > 1)):
                d = int(np.sign(d))
                parabolic = (self.heights[i+1] - self.heights[i-1]) / \
                           (self.positions[i+1] - self.positions[i-1])
                linear = (self.heights[i+d] - self.heights[i]) / \
                        (self.positions[i+d] - self.positions[i])
                height = self.heights[i] + d * \
                        (linear + parabolic * (d - 1) / 2)
                
                if height > self.heights[i-1] and height < self.heights[i+1]:
                    self.heights[i] = height
                    self.positions[i] += d
    
    def get_quartiles(self) -> dict:
        """
        Return current quartile estimates
        """
        return {
            'Q1': self.heights[1],
            'Q2': self.heights[2],
            'Q3': self.heights[3]
        }

# Example usage
p2 = P2Quartiles()
data = np.random.normal(100, 15, 1000)

for value in data:
    p2.update(value)

results = p2.get_quartiles()
print("\nP2 Algorithm Results:")
for quartile, value in results.items():
    print(f"{quartile}: {value:.2f}")

# Compare with numpy
np_quartiles = np.quantile(data, [0.25, 0.5, 0.75])
print("\nNumPy Quartiles for comparison:")
for q, value in zip(['Q1', 'Q2', 'Q3'], np_quartiles):
    print(f"{q}: {value:.2f}")
```

Slide 14: Additional Resources

*   "A Fast Algorithm for Approximate Quantiles in High Speed Data Streams" [https://arxiv.org/abs/1907.04227](https://arxiv.org/abs/1907.04227)
*   "Efficient Computation of Robust Weighted Quartiles" [https://arxiv.org/abs/1912.01767](https://arxiv.org/abs/1912.01767)
*   "On the Choice of Sample Quantiles for Robust Estimation" [https://arxiv.org/abs/1903.09942](https://arxiv.org/abs/1903.09942)
*   For practical implementations and further research:
    *   Search for "P2 Algorithm implementations" on GitHub
    *   Review Statistical Computing documentation
    *   Explore SciPy and NumPy documentation for quartile calculations

