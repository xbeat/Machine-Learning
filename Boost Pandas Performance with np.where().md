## Boost Pandas Performance with np.where()
Slide 1: Understanding apply() vs where() Performance

The fundamental difference between apply() and where() lies in their execution models. While apply() processes data row by row using Python's interpreter, where() leverages NumPy's optimized C-level operations for vectorized computation, resulting in significant performance improvements.

```python
import pandas as pd
import numpy as np
import time

# Create sample dataset
df = pd.DataFrame({
    'value': np.random.randint(0, 100, 1000000)
})

# Using apply()
start = time.time()
df['category_apply'] = df['value'].apply(lambda x: 'High' if x > 50 else 'Low')
print(f"apply() time: {time.time() - start:.4f} seconds")

# Using where()
start = time.time()
df['category_where'] = np.where(df['value'] > 50, 'High', 'Low')
print(f"where() time: {time.time() - start:.4f} seconds")
```

Slide 2: Simple Conditional Logic with where()

Numpy's where() function excels at handling straightforward conditional operations by evaluating conditions element-wise across the entire array simultaneously, making it particularly efficient for large datasets with binary decision logic.

```python
import pandas as pd
import numpy as np

# Sample financial dataset
df = pd.DataFrame({
    'price': [100, 150, 80, 200, 120],
    'volume': [1000, 1500, 800, 2000, 1200]
})

# Calculate trading signals using where()
df['signal'] = np.where(
    (df['price'] > 100) & (df['volume'] > 1000),
    'Buy',
    'Hold'
)

print(df)
```

Slide 3: Multiple Conditions with where()

When dealing with multiple conditions, where() can be nested to create complex decision trees while maintaining vectorized performance. This approach is significantly faster than using multiple apply() operations or chained lambda functions.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'score': [85, 92, 78, 95, 65, 88],
    'attendance': [90, 95, 85, 92, 70, 88]
})

df['grade'] = np.where(
    (df['score'] >= 90) & (df['attendance'] >= 90), 'A',
    np.where(
        (df['score'] >= 80) & (df['attendance'] >= 85), 'B',
        'C'
    )
)

print(df)
```

Slide 4: Performance Benchmarking

Understanding the performance implications of different approaches is crucial for optimizing data processing pipelines. This comprehensive benchmark compares execution times across various dataset sizes to demonstrate where()'s superiority.

```python
import pandas as pd
import numpy as np
import time

def benchmark_operations(size):
    df = pd.DataFrame({
        'value': np.random.randint(0, 100, size)
    })
    
    # apply() benchmark
    start = time.time()
    df['result_apply'] = df['value'].apply(
        lambda x: 'Category A' if x > 75 
        else 'Category B' if x > 50 
        else 'Category C'
    )
    apply_time = time.time() - start
    
    # where() benchmark
    start = time.time()
    df['result_where'] = np.where(
        df['value'] > 75, 'Category A',
        np.where(df['value'] > 50, 'Category B', 'Category C')
    )
    where_time = time.time() - start
    
    return apply_time, where_time

sizes = [10000, 100000, 1000000]
for size in sizes:
    apply_t, where_t = benchmark_operations(size)
    print(f"\nDataset size: {size:,}")
    print(f"apply() time: {apply_t:.4f} seconds")
    print(f"where() time: {where_t:.4f} seconds")
    print(f"Speed improvement: {(apply_t/where_t):.2f}x")
```

Slide 5: Memory Efficiency Comparison

Memory usage optimization is crucial when working with large datasets. where() operations typically consume less memory than apply() due to their vectorized nature and ability to operate on the underlying NumPy arrays directly.

```python
import pandas as pd
import numpy as np
import memory_profiler

@memory_profiler.profile
def compare_memory_usage(size=1000000):
    df = pd.DataFrame({
        'value': np.random.randint(0, 100, size)
    })
    
    # Memory usage with apply()
    result_apply = df['value'].apply(
        lambda x: x * 2 if x > 50 else x
    )
    
    # Memory usage with where()
    result_where = np.where(
        df['value'] > 50,
        df['value'] * 2,
        df['value']
    )
    
    return result_apply, result_where

# Run memory comparison
results = compare_memory_usage()
```

Slide 6: Real-World Example: Financial Data Analysis

Processing financial market data requires efficient computation for timely decision-making. This implementation demonstrates how where() can optimize the calculation of technical indicators and trading signals in a high-frequency trading context.

```python
import pandas as pd
import numpy as np

# Generate sample market data
np.random.seed(42)
df = pd.DataFrame({
    'price': np.random.normal(100, 5, 1000000),
    'volume': np.random.normal(10000, 1000, 1000000),
    'volatility': np.random.normal(0.02, 0.005, 1000000)
})

# Calculate multiple trading signals using where()
df['signal'] = np.where(
    (df['price'] > df['price'].rolling(20).mean()) &
    (df['volume'] > df['volume'].rolling(20).mean()) &
    (df['volatility'] < df['volatility'].rolling(20).mean()),
    1,  # Buy signal
    np.where(
        (df['price'] < df['price'].rolling(20).mean()) &
        (df['volume'] > df['volume'].rolling(20).mean()),
        -1,  # Sell signal
        0   # Hold
    )
)

print(f"Processing time for {len(df):,} rows")
print(f"Signal distribution:\n{df['signal'].value_counts()}")
```

Slide 7: Handling Missing Data with where()

The where() function provides elegant solutions for handling missing data while maintaining computational efficiency. This example demonstrates advanced missing data imputation techniques using vectorized operations.

```python
import pandas as pd
import numpy as np

# Create dataset with missing values
df = pd.DataFrame({
    'sales': [100, np.nan, 150, np.nan, 200, 180, np.nan],
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C']
})

# Complex imputation logic with where()
category_means = df.groupby('category')['sales'].transform('mean')
global_mean = df['sales'].mean()

df['sales_cleaned'] = np.where(
    df['sales'].isna(),
    np.where(
        category_means.notna(),
        category_means,
        global_mean
    ),
    df['sales']
)

print("Original vs Cleaned Data:")
print(pd.concat([df['sales'], df['sales_cleaned']], axis=1))
```

Slide 8: Vectorized Date Processing

Date operations can be computationally expensive when using apply(). This example shows how to leverage where() for efficient date-based calculations and transformations.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample datetime data
dates = pd.date_range(start='2023-01-01', periods=1000000, freq='1min')
df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.normal(100, 10, 1000000)
})

# Extract hour component
df['hour'] = df['timestamp'].dt.hour

# Complex time-based logic using where()
df['time_category'] = np.where(
    (df['hour'] >= 9) & (df['hour'] < 16),
    'Trading Hours',
    np.where(
        (df['hour'] >= 16) & (df['hour'] < 20),
        'After Market',
        'Off Hours'
    )
)

# Calculate time-weighted adjustments
df['adjusted_value'] = np.where(
    df['time_category'] == 'Trading Hours', 
    df['value'],
    np.where(
        df['time_category'] == 'After Market',
        df['value'] * 0.8,
        df['value'] * 0.5
    )
)

print(df.groupby('time_category').agg({
    'value': 'mean',
    'adjusted_value': 'mean'
}).round(2))
```

Slide 9: Optimizing Categorical Data Processing

When working with categorical data, where() can significantly improve performance compared to traditional mapping approaches. This implementation shows advanced categorical data transformation techniques.

```python
import pandas as pd
import numpy as np

# Create large categorical dataset
categories = ['A', 'B', 'C', 'D', 'E']
df = pd.DataFrame({
    'category': np.random.choice(categories, 1000000),
    'value': np.random.normal(100, 15, 1000000)
})

# Complex categorical transformations
df['risk_score'] = np.where(
    df['category'].isin(['A', 'B']),
    np.where(
        df['value'] > 100,
        'High Risk - Premium',
        'Low Risk - Premium'
    ),
    np.where(
        df['value'] > 90,
        'High Risk - Standard',
        'Low Risk - Standard'
    )
)

# Calculate risk metrics
risk_metrics = df.groupby('risk_score')['value'].agg(['count', 'mean', 'std'])
print("\nRisk Distribution:")
print(risk_metrics.round(2))
```

Slide 10: Optimization for Large-Scale Data Processing

When processing large-scale datasets, memory management becomes crucial. This implementation demonstrates how to efficiently handle millions of rows using chunked processing with where() operations.

```python
import pandas as pd
import numpy as np
from typing import Generator

def process_chunks(df_size: int, chunk_size: int) -> Generator:
    # Generate data in chunks
    chunks_num = df_size // chunk_size
    
    for i in range(chunks_num):
        chunk = pd.DataFrame({
            'value': np.random.normal(100, 15, chunk_size),
            'category': np.random.choice(['A', 'B', 'C'], chunk_size)
        })
        
        # Apply vectorized transformations
        chunk['transformed'] = np.where(
            (chunk['value'] > 100) & (chunk['category'] == 'A'),
            chunk['value'] * 1.5,
            np.where(
                (chunk['value'] <= 100) & (chunk['category'] == 'B'),
                chunk['value'] * 0.8,
                chunk['value']
            )
        )
        yield chunk

# Process 10 million rows in chunks
total_size = 10_000_000
chunk_size = 1_000_000

results = []
for chunk in process_chunks(total_size, chunk_size):
    results.append(chunk['transformed'].mean())

print(f"Average transformed value: {np.mean(results):.2f}")
```

Slide 11: Advanced Pattern Recognition with where()

Pattern recognition in time series data can be optimized using where() for identifying complex sequences and trends while maintaining high performance.

```python
import pandas as pd
import numpy as np

# Generate time series data
n_points = 1000000
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1min'),
    'price': np.random.normal(100, 5, n_points)
})

# Calculate moving averages
df['sma_20'] = df['price'].rolling(20).mean()
df['sma_50'] = df['price'].rolling(50).mean()

# Identify complex patterns using where()
df['pattern'] = np.where(
    (df['sma_20'] > df['sma_50']) & 
    (df['sma_20'].shift(1) <= df['sma_50'].shift(1)),
    'Golden Cross',
    np.where(
        (df['sma_20'] < df['sma_50']) & 
        (df['sma_20'].shift(1) >= df['sma_50'].shift(1)),
        'Death Cross',
        'No Pattern'
    )
)

# Calculate pattern statistics
pattern_stats = df.groupby('pattern').agg({
    'price': ['count', 'mean', 'std']
}).round(2)

print("\nPattern Analysis:")
print(pattern_stats)
```

Slide 12: Real-World Example: Customer Segmentation

This implementation showcases how where() can optimize customer segmentation algorithms by efficiently processing multiple conditional criteria across large customer datasets.

```python
import pandas as pd
import numpy as np

# Generate customer data
n_customers = 1000000
df = pd.DataFrame({
    'customer_id': range(n_customers),
    'age': np.random.normal(40, 15, n_customers),
    'spending': np.random.normal(1000, 500, n_customers),
    'frequency': np.random.normal(10, 5, n_customers)
})

# Clean and bound values
df['age'] = np.clip(df['age'], 18, 100)
df['spending'] = np.clip(df['spending'], 0, None)
df['frequency'] = np.clip(df['frequency'], 0, None)

# Complex customer segmentation using where()
df['segment'] = np.where(
    (df['spending'] > df['spending'].quantile(0.75)) &
    (df['frequency'] > df['frequency'].quantile(0.75)),
    'Premium',
    np.where(
        (df['spending'] > df['spending'].quantile(0.5)) &
        (df['frequency'] > df['frequency'].quantile(0.5)),
        'Regular',
        np.where(
            (df['age'] < 25) & (df['spending'] > df['spending'].mean()),
            'Young High-Value',
            'Standard'
        )
    )
)

# Calculate segment metrics
segment_analysis = df.groupby('segment').agg({
    'customer_id': 'count',
    'spending': 'mean',
    'frequency': 'mean',
    'age': 'mean'
}).round(2)

print("\nCustomer Segment Analysis:")
print(segment_analysis)
```

Slide 13: Performance Optimization Techniques

This implementation demonstrates advanced techniques for optimizing where() operations, including parallel processing and memory management strategies for handling extremely large datasets efficiently.

```python
import pandas as pd
import numpy as np
from multiprocessing import Pool
import functools

def process_partition(df_partition: pd.DataFrame) -> pd.DataFrame:
    # Complex conditions using where()
    return pd.DataFrame({
        'result': np.where(
            (df_partition['value'] > df_partition['value'].mean()) &
            (df_partition['value'] < df_partition['value'].mean() + 
             2 * df_partition['value'].std()),
            df_partition['value'] * 1.1,
            np.where(
                df_partition['value'] < df_partition['value'].mean(),
                df_partition['value'] * 0.9,
                df_partition['value']
            )
        )
    })

# Generate large dataset
n_rows = 5_000_000
df = pd.DataFrame({
    'value': np.random.normal(100, 15, n_rows)
})

# Split into partitions
n_partitions = 4
df_split = np.array_split(df, n_partitions)

# Parallel processing
with Pool(processes=n_partitions) as pool:
    results = pool.map(process_partition, df_split)

# Combine results
final_df = pd.concat(results)

print(f"Processed {n_rows:,} rows using {n_partitions} partitions")
print("\nResults Summary:")
print(final_df['result'].describe().round(2))
```

Slide 14: Advanced Data Type Optimization

Understanding how where() interacts with different data types is crucial for optimal performance. This example shows best practices for handling various data types while maintaining vectorized operations.

```python
import pandas as pd
import numpy as np

# Create dataset with multiple data types
df = pd.DataFrame({
    'numeric': np.random.normal(100, 15, 1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'date': pd.date_range('2023-01-01', periods=1000000)
})

# Optimize memory usage
df['category'] = df['category'].astype('category')
df['date'] = pd.to_datetime(df['date'])

# Complex type-aware transformations
df['result'] = np.where(
    (df['numeric'] > 100) & (df['category'] == 'A'),
    df['numeric'].astype(np.float32),  # Reduce precision for memory
    np.where(
        df['date'] > pd.Timestamp('2023-06-01'),
        df['numeric'] * 1.5,
        df['numeric']
    )
).astype(np.float32)

# Memory usage analysis
memory_usage = df.memory_usage(deep=True) / 1024**2  # Convert to MB
print("\nMemory Usage (MB):")
for column, usage in memory_usage.items():
    print(f"{column}: {usage:.2f}")
```

Slide 15: Additional Resources

*   arxiv.org/abs/2301.00215 - "Optimizing DataFrame Operations: A Comprehensive Study of Vectorization Techniques"
*   arxiv.org/abs/2207.12505 - "High-Performance Data Processing in Python: Benchmarking Vectorized Operations"
*   Search Google Scholar for: "Pandas Performance Optimization Techniques"
*   Documentation resources:
    *   pandas.pydata.org/docs/user\_guide/enhancingperf.html
    *   numpy.org/doc/stable/reference/generated/numpy.where.html
*   GitHub repositories for benchmarks:
    *   github.com/pandas-dev/pandas
    *   github.com/numpy/numpy

