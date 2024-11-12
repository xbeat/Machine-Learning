## Choosing Between NumPy and Pandas for Python Data Processing
Slide 1: NumPy Fundamentals - Array Operations

NumPy arrays provide efficient storage and operations for numerical data through contiguous memory allocation. Unlike Python lists, NumPy arrays enforce homogeneous data types, enabling vectorized operations that significantly boost computational performance for mathematical calculations.

```python
import numpy as np

# Creating arrays and basic operations
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

# Vectorized operations - no explicit loops needed
addition = arr1 + arr2
multiplication = arr1 * arr2
power = arr1 ** 2

print(f"Addition: {addition}")
print(f"Multiplication: {multiplication}")
print(f"Power: {power}")

# Output:
# Addition: [ 7  9 11 13 15]
# Multiplication: [ 6 14 24 36 50]
# Power: [ 1  4  9 16 25]
```

Slide 2: Pandas Series and DataFrame Basics

Pandas introduces two primary data structures: Series (1-dimensional) and DataFrame (2-dimensional), both built on top of NumPy arrays. These structures add powerful indexing, data alignment, and handling of missing values capabilities essential for data analysis.

```python
import pandas as pd

# Creating Series and DataFrame
series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
df = pd.DataFrame({
    'numbers': [1, 2, 3, 4],
    'letters': ['a', 'b', 'c', 'd'],
    'values': [1.1, 2.2, 3.3, 4.4]
})

print("Series:\n", series)
print("\nDataFrame:\n", df)

# Accessing data
print("\nAccessing column:", df['numbers'])
print("\nFiltering:", df[df['values'] > 2.5])
```

Slide 3: NumPy Performance Analysis

Understanding performance differences between NumPy and pure Python operations is crucial for optimization. NumPy's vectorized operations execute at the C level, avoiding Python's loop overhead and providing significant speedup for large-scale numerical computations.

```python
import numpy as np
import time

# Performance comparison: NumPy vs Python lists
size = 1000000

# Python list operation
python_list = list(range(size))
start_time = time.time()
python_result = [x**2 for x in python_list]
python_time = time.time() - start_time

# NumPy operation
numpy_array = np.arange(size)
start_time = time.time()
numpy_result = numpy_array**2
numpy_time = time.time() - start_time

print(f"Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"Speed improvement: {python_time/numpy_time:.2f}x")
```

Slide 4: Pandas Data Cleaning and Preprocessing

Data cleaning is a critical step in any data analysis pipeline. Pandas provides comprehensive tools for handling missing values, removing duplicates, and transforming data formats, making it indispensable for preparing real-world datasets.

```python
import pandas as pd
import numpy as np

# Create sample dataset with issues
df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', None, '2023-01-03'],
    'value': [1.0, np.nan, 3.0, 3.0],
    'category': ['A', 'B', 'B', 'A']
})

# Clean the data
cleaned_df = df.copy()
cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])  # Convert to datetime
cleaned_df['value'].fillna(cleaned_df['value'].mean(), inplace=True)  # Fill NaN
cleaned_df.dropna(subset=['date'], inplace=True)  # Remove rows with missing dates
cleaned_df.drop_duplicates(subset=['value', 'category'], inplace=True)  # Remove duplicates

print("Original DataFrame:\n", df)
print("\nCleaned DataFrame:\n", cleaned_df)
```

Slide 5: NumPy Matrix Operations

Matrix operations form the backbone of scientific computing and machine learning algorithms. NumPy provides highly optimized implementations of matrix operations, leveraging efficient BLAS and LAPACK libraries for linear algebra computations.

```python
import numpy as np

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
matrix_product = np.dot(A, B)  # Matrix multiplication
eigenvalues, eigenvectors = np.linalg.eig(A)  # Eigendecomposition
inverse = np.linalg.inv(A)  # Matrix inverse
determinant = np.linalg.det(A)  # Determinant

print("Matrix Product:\n", matrix_product)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\nInverse:\n", inverse)
print("\nDeterminant:", determinant)
```

Slide 6: Pandas Advanced Data Aggregation

Pandas provides powerful grouping and aggregation capabilities through the GroupBy operation. This functionality enables complex data analysis by splitting data into groups, applying functions, and combining results efficiently for insightful analytics.

```python
import pandas as pd
import numpy as np

# Create sample sales data
sales_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], size=365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], size=365),
    'sales': np.random.normal(1000, 200, size=365),
    'units': np.random.randint(10, 100, size=365)
})

# Complex aggregation
agg_results = sales_data.groupby(['product', 'region']).agg({
    'sales': ['mean', 'sum', 'std'],
    'units': ['count', 'max']
}).round(2)

# Calculate monthly trends
monthly_trends = sales_data.set_index('date').resample('M').agg({
    'sales': 'sum',
    'units': 'mean'
}).round(2)

print("Aggregated Results:\n", agg_results)
print("\nMonthly Trends:\n", monthly_trends)
```

Slide 7: NumPy Broadcasting and Vectorization

Broadcasting is a powerful mechanism that enables NumPy to perform operations on arrays of different shapes efficiently. Understanding broadcasting rules is crucial for writing optimized numerical computations without explicit loops.

```python
import numpy as np

# Broadcasting examples
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])  # Shape: (2, 3)
vector = np.array([10, 20, 30])   # Shape: (3,)

# Broadcasting in action
broadcast_add = array_2d + vector
broadcast_multiply = array_2d * vector

# Complex broadcasting example
coords = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2]])  # Shape: (3, 3)
weights = np.array([1, 2, 3]).reshape(3, 1)  # Shape: (3, 1)
weighted_coords = coords * weights  # Shape: (3, 3)

print("Original array:\n", array_2d)
print("\nBroadcast addition:\n", broadcast_add)
print("\nBroadcast multiplication:\n", broadcast_multiply)
print("\nWeighted coordinates:\n", weighted_coords)
```

Slide 8: Pandas Time Series Analysis

Time series analysis is a cornerstone of data science, and Pandas excels at handling temporal data with its sophisticated datetime functionality, resampling operations, and rolling window calculations.

```python
import pandas as pd
import numpy as np

# Generate time series data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
ts_data = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

# Time series operations
rolling_mean = ts_data.rolling(window=7).mean()  # 7-day moving average
monthly_data = ts_data.resample('M').agg(['mean', 'std'])
year_to_date = ts_data.cumsum()

# Calculate seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data, period=30, model='additive')

print("Original Time Series Head:\n", ts_data.head())
print("\nRolling Mean Head:\n", rolling_mean.head())
print("\nMonthly Statistics:\n", monthly_data)

# Plot components (commented out as per requirements)
# decomposition.plot()
```

Slide 9: Real-world Application - Portfolio Analysis

This implementation demonstrates a practical application combining NumPy and Pandas for financial portfolio analysis, showcasing how both libraries complement each other in real-world scenarios.

```python
import numpy as np
import pandas as pd

# Generate sample stock data
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
prices = pd.DataFrame(
    np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100,
    index=dates,
    columns=stocks
)

# Calculate daily returns
returns = prices.pct_change()

# Portfolio analysis
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
sharpe_ratio = portfolio_return / portfolio_vol

print("Portfolio Metrics:")
print(f"Annual Return: {portfolio_return:.4f}")
print(f"Annual Volatility: {portfolio_vol:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
```

Slide 10: Real-world Application - Market Basket Analysis

Implementing market basket analysis using Pandas demonstrates the library's strength in handling categorical data and computing complex relationships between items in transaction datasets.

```python
import pandas as pd
import numpy as np
from itertools import combinations

# Generate sample transaction data
transactions = pd.DataFrame({
    'transaction_id': np.repeat(range(1000), 3),
    'item': np.random.choice(['bread', 'milk', 'eggs', 'cheese', 'butter'], 3000)
})

# Create item pairs and calculate support
def calculate_support(transactions):
    # Convert to binary purchase matrix
    purchase_matrix = pd.crosstab(transactions['transaction_id'], transactions['item'])
    
    # Calculate item pair frequencies
    n_transactions = len(purchase_matrix)
    item_pairs = []
    support_values = []
    
    for item1, item2 in combinations(purchase_matrix.columns, 2):
        both_purchased = purchase_matrix[purchase_matrix[item1] & purchase_matrix[item2]].shape[0]
        support = both_purchased / n_transactions
        item_pairs.append(f"{item1} -> {item2}")
        support_values.append(support)
    
    return pd.DataFrame({
        'item_pair': item_pairs,
        'support': support_values
    }).sort_values('support', ascending=False)

results = calculate_support(transactions)
print("Top 5 Item Pairs by Support:\n", results.head())
```

Slide 11: NumPy Performance Optimization Techniques

Advanced optimization techniques in NumPy can significantly improve computational efficiency through memory management, vectorization, and proper array operations that minimize temporary array creation.

```python
import numpy as np
import time

# Optimization example: comparing different approaches
size = 1000000

# Inefficient approach with temporary arrays
def inefficient_calculation(arr):
    temp1 = arr * 2
    temp2 = temp1 + 3
    return temp2 ** 2

# Optimized approach without temporary arrays
def efficient_calculation(arr):
    return np.square(np.add(np.multiply(arr, 2), 3))

# Memory pre-allocation example
def optimized_growth():
    result = np.zeros(size, dtype=np.float64)
    for i in range(size):
        result[i] = i * 2
    return result

# Benchmark
arr = np.random.rand(size)

start = time.time()
result1 = inefficient_calculation(arr)
time1 = time.time() - start

start = time.time()
result2 = efficient_calculation(arr)
time2 = time.time() - start

print(f"Inefficient approach time: {time1:.4f} seconds")
print(f"Efficient approach time: {time2:.4f} seconds")
print(f"Speed improvement: {time1/time2:.2f}x")
```

Slide 12: Pandas Advanced Indexing and Selection

Advanced indexing techniques in Pandas enable sophisticated data selection and filtering operations, crucial for complex data analysis tasks and feature engineering in machine learning pipelines.

```python
import pandas as pd
import numpy as np

# Create complex dataset
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'value': np.random.randn(1000),
    'flag': np.random.choice([True, False], 1000),
    'group': np.random.randint(1, 5, 1000)
})

# Advanced indexing examples
mask = (df['value'] > 0) & (df['flag']) & (df['category'].isin(['A', 'B']))
filtered = df.loc[mask]

# Multi-level indexing
df.set_index(['date', 'category'], inplace=True)
df.sort_index(inplace=True)

# Complex selection
slice_selection = df.loc['2023-01':'2023-02', 'A':'B']
value_selection = df.xs('A', level='category')

print("Filtered Data:\n", filtered.head())
print("\nMulti-index Selection:\n", slice_selection.head())
print("\nCross-section Selection:\n", value_selection.head())
```

Slide 13: Memory Management and Performance Optimization

Advanced memory management techniques are crucial when working with large datasets. Understanding how NumPy and Pandas handle memory internally enables optimization of data processing pipelines for better performance.

```python
import numpy as np
import pandas as pd
import sys

# Memory usage analysis
def analyze_memory_usage(obj, name="Object"):
    size_bytes = sys.getsizeof(obj)
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        size_bytes = obj.memory_usage(deep=True).sum()
    
    # Convert to readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{name} size: {size_bytes:.2f} {unit}"
        size_bytes /= 1024

# Compare different data types
df_float64 = pd.DataFrame(np.random.randn(100000, 4), columns=['A', 'B', 'C', 'D'])
df_float32 = df_float64.astype(np.float32)
df_sparse = pd.DataFrame(np.random.choice([0, 1], size=(100000, 4), p=[0.99, 0.01]))
df_sparse = df_sparse.astype(pd.SparseDtype("int", fill_value=0))

print(analyze_memory_usage(df_float64, "Float64 DataFrame"))
print(analyze_memory_usage(df_float32, "Float32 DataFrame"))
print(analyze_memory_usage(df_sparse, "Sparse DataFrame"))

# Memory-efficient operations
def efficient_operation(df):
    return df.groupby('A')['B'].transform('mean')

def inefficient_operation(df):
    return df.apply(lambda x: x['B'] - x['B'].mean())

# Example of memory-efficient chunking
def process_large_csv(filename, chunksize=10000):
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        processed = chunk.value.mean()  # Example operation
        chunks.append(processed)
    return pd.concat(chunks)
```

Slide 14: Integrated NumPy and Pandas Pipeline

A comprehensive example demonstrating how to effectively combine NumPy and Pandas in a real-world data processing pipeline, leveraging the strengths of both libraries for optimal performance.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create sample dataset
np.random.seed(42)
n_samples = 100000

# Generate data using NumPy (efficient for numerical computations)
numeric_features = np.random.randn(n_samples, 3)
categorical_features = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))
timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')

# Convert to Pandas for data manipulation
df = pd.DataFrame(
    np.hstack([numeric_features, categorical_features]),
    columns=['value1', 'value2', 'value3', 'cat1', 'cat2']
)
df['timestamp'] = timestamps

# Preprocessing pipeline
def preprocess_pipeline(df):
    # Use NumPy for numerical calculations
    numeric_cols = ['value1', 'value2', 'value3']
    numeric_data = df[numeric_cols].values
    
    # Standardize using NumPy operations
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Back to Pandas for feature engineering
    df[numeric_cols] = scaled_data
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # One-hot encoding using Pandas
    categorical_dummies = pd.get_dummies(df[['cat1', 'cat2']], prefix=['cat1', 'cat2'])
    
    # Combine features
    final_df = pd.concat([df[numeric_cols], 
                         df[['hour', 'dayofweek']], 
                         categorical_dummies], axis=1)
    
    return final_df

# Process data
processed_df = preprocess_pipeline(df)
print("Processed DataFrame Shape:", processed_df.shape)
print("\nFeature Names:", processed_df.columns.tolist())
print("\nMemory Usage:", processed_df.memory_usage().sum() / 1024 / 1024, "MB")
```

Slide 15: Additional Resources

*   Machine Learning with NumPy and Pandas:
    *   [https://arxiv.org/abs/2306.15561](https://arxiv.org/abs/2306.15561)
    *   [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267642](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267642)
*   Performance Optimization:
    *   [https://www.nature.com/articles/s41598-020-76767-0](https://www.nature.com/articles/s41598-020-76767-0)
    *   [https://academic.oup.com/gigascience/article/9/10/giaa102/5918883](https://academic.oup.com/gigascience/article/9/10/giaa102/5918883)
*   Best Practices and Tutorials:
    *   [https://scipy.org/](https://scipy.org/)
    *   [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
    *   [https://numpy.org/doc/stable/user/](https://numpy.org/doc/stable/user/)

