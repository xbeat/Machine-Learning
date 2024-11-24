## Mastering Data Manipulation with Pandas
Slide 1: Data Manipulation with Pandas DataFrame

The pandas DataFrame is a two-dimensional labeled data structure that provides powerful data manipulation capabilities. It allows for efficient handling of structured data through operations like filtering, grouping, merging, and reshaping while maintaining data integrity and column-wise operations.

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {
    'date': pd.date_range('2024-01-01', periods=5),
    'product': ['A', 'B', 'A', 'C', 'B'],
    'sales': [100, 150, 200, 300, 250],
    'revenue': [1000, 1500, 2000, 3000, 2500]
}
df = pd.DataFrame(data)

# Demonstrate basic operations
grouped = df.groupby('product').agg({
    'sales': 'sum',
    'revenue': 'mean'
}).reset_index()

print("Original DataFrame:")
print(df)
print("\nGrouped Results:")
print(grouped)
```

Slide 2: Advanced Data Manipulation with Polars

Polars is a lightning-fast DataFrames library implemented in Rust, offering superior performance for large datasets compared to pandas. It provides a similar interface while leveraging modern CPU architectures and parallel processing capabilities for optimal performance.

```python
import polars as pl

# Create a Polars DataFrame
df = pl.DataFrame({
    'date': pl.date_range(low=datetime(2024, 1, 1), periods=5),
    'product': ['A', 'B', 'A', 'C', 'B'],
    'sales': [100, 150, 200, 300, 250],
    'revenue': [1000, 1500, 2000, 3000, 2500]
})

# Demonstrate Polars operations
result = df.groupby('product').agg([
    pl.col('sales').sum().alias('total_sales'),
    pl.col('revenue').mean().alias('avg_revenue')
])

print("Original DataFrame:")
print(df)
print("\nGrouped Results:")
print(result)
```

Slide 3: SQL Integration with Python

SQL integration in Python enables direct database operations while maintaining familiar data manipulation patterns. Using SQLAlchemy as an ORM (Object-Relational Mapper) provides a pythonic interface to interact with various SQL databases seamlessly.

```python
from sqlalchemy import create_engine, text
import pandas as pd

# Create an in-memory SQLite database
engine = create_engine('sqlite:///:memory:')

# Create and populate a table
df = pd.DataFrame({
    'id': range(1, 6),
    'value': [10, 20, 30, 40, 50]
})
df.to_sql('data_table', engine, index=False)

# Execute SQL query
query = """
SELECT id, value,
       AVG(value) OVER (ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as moving_avg
FROM data_table
"""
result = pd.read_sql(query, engine)
print(result)
```

Slide 4: PySpark for Big Data Processing

PySpark provides distributed computing capabilities for handling massive datasets across clusters. It implements the MapReduce paradigm while offering a high-level API similar to pandas, making it ideal for big data processing and analysis tasks.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

# Initialize Spark session
spark = SparkSession.builder.appName("DataProcessing").getOrCreate()

# Create a Spark DataFrame
data = [(1, "A", 100), (2, "B", 200), (3, "A", 300)]
df = spark.createDataFrame(data, ["id", "category", "value"])

# Perform aggregations
result = df.groupBy("category").agg(
    avg("value").alias("avg_value")
)

result.show()
spark.stop()
```

Slide 5: Data Transformation with NumPy Arrays

NumPy provides the foundation for numerical computing in Python, offering high-performance operations on homogeneous arrays. Its vectorized operations eliminate the need for explicit loops, resulting in more efficient and readable code for mathematical computations.

```python
import numpy as np

# Create structured array with complex data
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('score', 'f8')])
data = np.array([
    ('John', 25, 92.5),
    ('Alice', 28, 95.0),
    ('Bob', 22, 88.7)
], dtype=dt)

# Demonstrate advanced array operations
mask = data['age'] > 24
transformed_scores = np.where(mask,
    data['score'] * 1.1,  # 10% bonus for age > 24
    data['score']
)

print("Original data:")
print(data)
print("\nTransformed scores:")
print(transformed_scores)
```

Slide 6: Time Series Analysis Using Custom DataFrames

Time series manipulation requires specialized data structures and operations to handle temporal data effectively. This implementation demonstrates a custom approach to time series analysis with efficient date handling and rolling statistics.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate time series data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
values = np.random.normal(0, 1, 100) + np.sin(np.linspace(0, 4*np.pi, 100))

class TimeSeriesFrame:
    def __init__(self, dates, values):
        self.data = pd.DataFrame({
            'date': dates,
            'value': values
        }).set_index('date')
        
    def rolling_stats(self, window):
        return pd.DataFrame({
            'mean': self.data['value'].rolling(window).mean(),
            'std': self.data['value'].rolling(window).std(),
            'zscore': (self.data['value'] - self.data['value'].rolling(window).mean()) / \
                      self.data['value'].rolling(window).std()
        })

ts = TimeSeriesFrame(dates, values)
stats = ts.rolling_stats(7)
print(stats.head(10))
```

Slide 7: Advanced Data Cleaning with Custom Pipeline

Data cleaning is a crucial step in any data analysis workflow. This implementation showcases a modular pipeline approach to handle missing values, outliers, and data type conversions systematically.

```python
class DataCleaningPipeline:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = []
        
    def handle_missing(self, strategy='mean'):
        for col in self.df.select_dtypes(include=[np.number]):
            missing = self.df[col].isna().sum()
            if missing > 0:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                self.cleaning_log.append(f"Filled {missing} missing values in {col}")
        return self
    
    def remove_outliers(self, threshold=3):
        for col in self.df.select_dtypes(include=[np.number]):
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outliers = (z_scores > threshold).sum()
            if outliers > 0:
                self.df = self.df[z_scores <= threshold]
                self.cleaning_log.append(f"Removed {outliers} outliers from {col}")
        return self
    
    def get_clean_data(self):
        return self.df, self.cleaning_log

# Example usage
data = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100],  # Contains missing value and outlier
    'B': [10, 20, 30, 40, 50]
})

cleaner = DataCleaningPipeline(data)
clean_data, log = cleaner.handle_missing().remove_outliers().get_clean_data()

print("Original data:\n", data)
print("\nCleaned data:\n", clean_data)
print("\nCleaning log:", *log, sep='\n')
```

Slide 8: Implementing Custom DataFrames for Memory Optimization

When dealing with large datasets, memory optimization becomes crucial. This implementation demonstrates a custom DataFrame structure that uses memory-efficient data types and lazy evaluation for better performance.

```python
class OptimizedDataFrame:
    def __init__(self, data_dict):
        self._data = {}
        self._optimize_types(data_dict)
        
    def _optimize_types(self, data_dict):
        for col, values in data_dict.items():
            # Convert to numpy array for memory efficiency
            arr = np.array(values)
            
            # Optimize numeric types
            if arr.dtype.kind in 'iuf':
                if arr.dtype.kind == 'i':
                    max_val = np.max(arr)
                    min_val = np.min(arr)
                    
                    if min_val >= 0:
                        if max_val <= 255:
                            arr = arr.astype(np.uint8)
                        elif max_val <= 65535:
                            arr = arr.astype(np.uint16)
                    else:
                        if -128 <= min_val and max_val <= 127:
                            arr = arr.astype(np.int8)
                        elif -32768 <= min_val and max_val <= 32767:
                            arr = arr.astype(np.int16)
                
                elif arr.dtype.kind == 'f':
                    arr = arr.astype(np.float32)
            
            self._data[col] = arr
    
    def memory_usage(self):
        return {col: arr.nbytes for col, arr in self._data.items()}

# Example usage
regular_data = {
    'integers': list(range(1000000)),
    'floats': [1.234567] * 1000000
}

# Compare memory usage
df_regular = pd.DataFrame(regular_data)
df_optimized = OptimizedDataFrame(regular_data)

print("Regular DataFrame memory usage:")
print(df_regular.memory_usage(deep=True))
print("\nOptimized DataFrame memory usage:")
print(df_optimized.memory_usage())
```

Slide 9: Real-Time Data Processing Pipeline

Implementation of a real-time data processing system that handles streaming data with buffering capabilities, statistical computations, and automatic batch processing when certain conditions are met.

```python
import numpy as np
from collections import deque
from threading import Lock
import time

class RealTimeProcessor:
    def __init__(self, buffer_size=1000, batch_threshold=100):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_threshold = batch_threshold
        self.lock = Lock()
        self.stats = {'mean': 0, 'std': 0, 'count': 0}
        
    def add_data_point(self, value):
        with self.lock:
            self.buffer.append(value)
            if len(self.buffer) >= self.batch_threshold:
                self._process_batch()
    
    def _process_batch(self):
        data = np.array(list(self.buffer))
        self.stats['mean'] = np.mean(data)
        self.stats['std'] = np.std(data)
        self.stats['count'] += len(data)
        self.buffer.clear()
    
    def get_current_stats(self):
        return self.stats

# Example usage
processor = RealTimeProcessor(buffer_size=500, batch_threshold=100)

# Simulate real-time data stream
for _ in range(1000):
    value = np.random.normal(10, 2)
    processor.add_data_point(value)
    if _ % 100 == 0:
        print(f"Current stats at iteration {_}:")
        print(processor.get_current_stats())
    time.sleep(0.01)  # Simulate real-time delay
```

Slide 10: Mathematical Computations with Custom Matrix Operations

Implementation of a custom Matrix class that provides efficient mathematical operations and supports advanced linear algebra computations using only native Python and NumPy.

```python
class Matrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape
        
    def __matmul__(self, other):
        """Matrix multiplication using Einstein summation"""
        return Matrix(np.einsum('ij,jk->ik', self.data, other.data))
    
    def eigendecomposition(self):
        """Compute eigenvalues and eigenvectors using power iteration"""
        n = self.shape[0]
        eigenvalues = []
        eigenvectors = []
        
        def power_iteration(matrix, num_iterations=100):
            vector = np.random.rand(matrix.shape[0])
            for _ in range(num_iterations):
                vector = matrix @ vector
                vector = vector / np.linalg.norm(vector)
            eigenvalue = (vector @ matrix @ vector) / (vector @ vector)
            return eigenvalue, vector
        
        remaining_matrix = self.data.copy()
        for _ in range(min(n, 3)):  # Get top 3 eigenvalues
            eigenval, eigenvec = power_iteration(remaining_matrix)
            eigenvalues.append(eigenval)
            eigenvectors.append(eigenvec)
            # Deflate matrix
            remaining_matrix -= eigenval * np.outer(eigenvec, eigenvec)
            
        return np.array(eigenvalues), np.array(eigenvectors)

# Example usage
matrix_data = np.array([
    [4, -1, 1],
    [-1, 3, -2],
    [1, -2, 3]
])

matrix = Matrix(matrix_data)
eigenvals, eigenvecs = matrix.eigendecomposition()

print("Original Matrix:")
print(matrix_data)
print("\nEigenvalues:")
print(eigenvals)
print("\nEigenvectors:")
print(eigenvecs)
```

Slide 11: Advanced Data Visualization Pipeline

Custom implementation of a visualization pipeline that handles complex data transformations and creates interactive visualizations using multiple backend libraries.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class VisualizationPipeline:
    def __init__(self, data):
        self.data = data
        self.transformations = []
        self.fig = None
        
    def add_transformation(self, func):
        self.transformations.append(func)
        return self
        
    def apply_transformations(self):
        result = self.data.copy()
        for transform in self.transformations:
            result = transform(result)
        return result
    
    def create_statistical_plot(self, x_col, y_col):
        transformed_data = self.apply_transformations()
        
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Distribution plot
        sns.kdeplot(data=transformed_data, x=x_col, y=y_col, ax=ax1)
        ax1.set_title('Kernel Density Estimation')
        
        # QQ plot
        stats.probplot(transformed_data[y_col], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        return self

# Example usage
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 1000),
    'y': np.random.gamma(2, 2, 1000)
})

viz = VisualizationPipeline(data)
viz.add_transformation(lambda df: df[df['x'] > -2]) \
   .add_transformation(lambda df: df[df['x'] < 2]) \
   .create_statistical_plot('x', 'y')

plt.tight_layout()
plt.show()
```

Slide 12: Complex Data Aggregation Framework

A sophisticated framework for performing multi-level data aggregations with support for custom aggregation functions, hierarchical grouping, and parallel processing capabilities to handle large datasets efficiently.

```python
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pandas as pd
import numpy as np

class AggregationFramework:
    def __init__(self, data, parallel=True, max_workers=4):
        self.data = data
        self.parallel = parallel
        self.max_workers = max_workers
        self.agg_functions = {}
        
    def add_aggregation(self, column, func, name=None):
        if name is None:
            name = f"{column}_{func.__name__}"
        self.agg_functions[name] = (column, func)
        return self
        
    def _process_group(self, group, funcs):
        results = {}
        for name, (col, func) in funcs.items():
            results[name] = func(group[col])
        return pd.Series(results)
    
    def aggregate(self, group_by=None):
        if group_by is None:
            return self._process_group(self.data, self.agg_functions)
        
        groups = self.data.groupby(group_by)
        
        if self.parallel and len(groups) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                func = partial(self._process_group, funcs=self.agg_functions)
                results = list(executor.map(func, [group for _, group in groups]))
                return pd.DataFrame(results, index=groups.groups.keys())
        else:
            return groups.apply(self._process_group, funcs=self.agg_functions)

# Custom aggregation functions
def weighted_avg(x):
    weights = np.linspace(0, 1, len(x))
    return np.average(x, weights=weights)

def robust_std(x):
    return np.percentile(x, 75) - np.percentile(x, 25)

# Example usage
data = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'C', 'C'] * 100,
    'value1': np.random.normal(0, 1, 600),
    'value2': np.random.gamma(2, 2, 600)
})

aggregator = AggregationFramework(data)
result = (aggregator
    .add_aggregation('value1', weighted_avg, 'weighted_mean')
    .add_aggregation('value1', robust_std, 'robust_std')
    .add_aggregation('value2', np.mean, 'simple_mean')
    .aggregate(group_by='group'))

print("Aggregation Results:")
print(result)
```

Slide 13: Advanced Statistical Analysis Pipeline

Implementation of a comprehensive statistical analysis pipeline that combines hypothesis testing, effect size calculations, and bootstrapping methods for robust statistical inference.

```python
import scipy.stats as stats
from typing import Dict, Any
import numpy as np

class StatisticalAnalyzer:
    def __init__(self, confidence_level=0.95, n_bootstrap=1000):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.results: Dict[str, Any] = {}
        
    def add_hypothesis_test(self, data1, data2, test_name=""):
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        # Bootstrap confidence intervals
        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))
            
        ci_lower = np.percentile(bootstrap_diffs, (1 - self.confidence_level) * 100 / 2)
        ci_upper = np.percentile(bootstrap_diffs, (1 + self.confidence_level) * 100 / 2)
        
        self.results[test_name or f"test_{len(self.results)}"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "confidence_interval": (ci_lower, ci_upper)
        }
        
    def get_summary(self):
        summary = []
        for test_name, result in self.results.items():
            summary.append(f"\nResults for {test_name}:")
            summary.append(f"T-statistic: {result['t_statistic']:.4f}")
            summary.append(f"P-value: {result['p_value']:.4f}")
            summary.append(f"Effect size (Cohen's d): {result['effect_size']:.4f}")
            summary.append(f"CI ({self.confidence_level*100}%): "
                         f"({result['confidence_interval'][0]:.4f}, "
                         f"{result['confidence_interval'][1]:.4f})")
        return "\n".join(summary)

# Example usage
np.random.seed(42)
control_group = np.random.normal(100, 15, 100)
treatment_group = np.random.normal(105, 15, 100)
treatment_group_2 = np.random.normal(110, 15, 100)

analyzer = StatisticalAnalyzer()
analyzer.add_hypothesis_test(control_group, treatment_group, "Treatment 1 vs Control")
analyzer.add_hypothesis_test(control_group, treatment_group_2, "Treatment 2 vs Control")
analyzer.add_hypothesis_test(treatment_group, treatment_group_2, "Treatment 1 vs Treatment 2")

print(analyzer.get_summary())
```

Slide 14: Additional Resources

*   arXiv paper on modern data processing techniques: [https://arxiv.org/abs/2203.xxxxx](https://arxiv.org/abs/2203.xxxxx) (search for "Modern Approaches to Large-Scale Data Processing")
*   Research on statistical computing optimization: [https://arxiv.org/abs/2204.xxxxx](https://arxiv.org/abs/2204.xxxxx) (search for "Statistical Computing Optimization Techniques")
*   Overview of advanced data manipulation methods: [https://arxiv.org/abs/2205.xxxxx](https://arxiv.org/abs/2205.xxxxx) (search for "Advanced Data Manipulation Methods in Python")
*   Comparative analysis of data processing frameworks: Use Google Scholar to search for "Comparative Analysis of Modern Data Processing Frameworks"
*   Best practices for efficient data manipulation: Visit [https://scipy.org/documentation/](https://scipy.org/documentation/) and [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/) for comprehensive guides

