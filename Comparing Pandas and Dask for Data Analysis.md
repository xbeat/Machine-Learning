## Comparing Pandas and Dask for Data Analysis
Slide 1: Introduction to Pandas and Dask

Pandas and Dask are powerful libraries in Python for data manipulation and analysis. While Pandas excels at handling smaller datasets in memory, Dask is designed for processing large datasets that don't fit in memory. This slideshow will compare these two libraries, highlighting their strengths and use cases.

```python
import pandas as pd
import dask.dataframe as dd

# Creating a small dataset with Pandas
pandas_df = pd.DataFrame({'A': range(5), 'B': range(5, 10)})

# Creating a large dataset with Dask
dask_df = dd.from_pandas(pandas_df, npartitions=2)

print("Pandas DataFrame:")
print(pandas_df)
print("\nDask DataFrame:")
print(dask_df)
```

Slide 2: Data Structures

Pandas primarily uses DataFrame and Series objects for data manipulation. Dask extends these concepts to handle larger datasets by breaking them into partitions that can be processed in parallel.

```python
# Pandas DataFrame and Series
pandas_df = pd.DataFrame({'A': range(5), 'B': range(5, 10)})
pandas_series = pd.Series(range(5))

# Dask DataFrame and Series
dask_df = dd.from_pandas(pandas_df, npartitions=2)
dask_series = dd.from_pandas(pandas_series, npartitions=2)

print("Pandas DataFrame shape:", pandas_df.shape)
print("Dask DataFrame shape:", dask_df.shape.compute())
print("\nPandas Series:")
print(pandas_series)
print("\nDask Series:")
print(dask_series.compute())
```

Slide 3: Loading Data

Both Pandas and Dask offer methods to load data from various sources. Pandas loads data entirely into memory, while Dask can work with data that doesn't fit in memory by loading it in chunks.

```python
import pandas as pd
import dask.dataframe as dd

# Loading CSV with Pandas
pandas_df = pd.read_csv('large_file.csv')

# Loading CSV with Dask
dask_df = dd.read_csv('large_file.csv')

print("Pandas DataFrame info:")
print(pandas_df.info())
print("\nDask DataFrame info:")
print(dask_df.info())
```

Slide 4: Basic Operations

Both libraries support similar operations like filtering, sorting, and aggregation. However, Dask operations are lazy and only computed when explicitly called.

```python
import pandas as pd
import dask.dataframe as dd

# Creating sample data
pandas_df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
dask_df = dd.from_pandas(pandas_df, npartitions=2)

# Filtering
pandas_filtered = pandas_df[pandas_df['A'] > 5]
dask_filtered = dask_df[dask_df['A'] > 5]

# Sorting
pandas_sorted = pandas_df.sort_values('B')
dask_sorted = dask_df.sort_values('B')

print("Pandas filtered and sorted:")
print(pandas_filtered)
print(pandas_sorted)

print("\nDask filtered and sorted:")
print(dask_filtered.compute())
print(dask_sorted.compute())
```

Slide 5: Memory Usage

Pandas loads all data into memory, which can be a limitation for large datasets. Dask, on the other hand, can work with datasets larger than available memory by processing data in chunks.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create a large DataFrame
large_df = pd.DataFrame(np.random.randn(1000000, 4), columns=list('ABCD'))

# Convert to Dask DataFrame
dask_df = dd.from_pandas(large_df, npartitions=10)

print("Pandas DataFrame memory usage:")
print(large_df.memory_usage(deep=True).sum() / 1e6, "MB")

print("\nDask DataFrame memory usage (estimated):")
print(dask_df.memory_usage(deep=True).sum().compute() / 1e6, "MB")
```

Slide 6: Parallel Processing

Dask leverages parallel processing to handle large datasets efficiently. It can distribute computations across multiple cores or even a cluster of machines.

```python
import pandas as pd
import dask.dataframe as dd
import time

# Create a large DataFrame
large_df = pd.DataFrame({'A': range(10000000), 'B': range(10000000, 20000000)})

# Convert to Dask DataFrame
dask_df = dd.from_pandas(large_df, npartitions=4)

# Measure time for Pandas operation
start_time = time.time()
pandas_result = large_df['A'].mean()
pandas_time = time.time() - start_time

# Measure time for Dask operation
start_time = time.time()
dask_result = dask_df['A'].mean().compute()
dask_time = time.time() - start_time

print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Dask time: {dask_time:.2f} seconds")
print(f"Speedup: {pandas_time / dask_time:.2f}x")
```

Slide 7: Lazy Evaluation

Dask uses lazy evaluation, meaning operations are only performed when results are explicitly requested. This allows for optimized computation plans and efficient memory usage.

```python
import pandas as pd
import dask.dataframe as dd

# Create DataFrames
pandas_df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
dask_df = dd.from_pandas(pandas_df, npartitions=2)

# Define operations
pandas_result = (pandas_df['A'] * 2).mean()
dask_result = (dask_df['A'] * 2).mean()

print("Pandas result (computed immediately):")
print(pandas_result)

print("\nDask result (not computed yet):")
print(dask_result)

print("\nDask result (after computation):")
print(dask_result.compute())
```

Slide 8: Handling Time Series Data

Both Pandas and Dask provide powerful tools for working with time series data, but Dask can handle much larger time series datasets.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create a time series DataFrame
dates = pd.date_range('2023-01-01', periods=1000000, freq='T')
pandas_ts = pd.DataFrame({'timestamp': dates, 'value': np.random.randn(1000000)})

# Convert to Dask DataFrame
dask_ts = dd.from_pandas(pandas_ts, npartitions=10)

# Resample and compute mean
pandas_result = pandas_ts.set_index('timestamp').resample('D').mean()
dask_result = dask_ts.set_index('timestamp').resample('D').mean().compute()

print("Pandas result:")
print(pandas_result.head())
print("\nDask result:")
print(dask_result.head())
```

Slide 9: Handling Missing Data

Both Pandas and Dask provide methods for handling missing data, but Dask can process larger datasets with missing values more efficiently.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create DataFrames with missing values
pandas_df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5], 'B': [np.nan, 2, 3, np.nan, 5]})
dask_df = dd.from_pandas(pandas_df, npartitions=2)

# Fill missing values
pandas_filled = pandas_df.fillna(0)
dask_filled = dask_df.fillna(0)

print("Pandas DataFrame with filled values:")
print(pandas_filled)

print("\nDask DataFrame with filled values:")
print(dask_filled.compute())
```

Slide 10: Grouping and Aggregation

Both libraries support grouping and aggregation operations, but Dask can handle these operations on much larger datasets.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create sample data
pandas_df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'value': np.random.randn(1000000)
})
dask_df = dd.from_pandas(pandas_df, npartitions=10)

# Perform groupby and aggregation
pandas_result = pandas_df.groupby('category')['value'].mean()
dask_result = dask_df.groupby('category')['value'].mean().compute()

print("Pandas groupby result:")
print(pandas_result)
print("\nDask groupby result:")
print(dask_result)
```

Slide 11: Visualization

Pandas integrates well with plotting libraries like Matplotlib, while Dask requires computation before visualization. However, Dask can handle preprocessing of larger datasets for visualization.

```python
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

# Create sample data
pandas_df = pd.DataFrame({'x': range(1000), 'y': np.random.randn(1000)})
dask_df = dd.from_pandas(pandas_df, npartitions=10)

# Pandas plot
plt.figure(figsize=(10, 5))
pandas_df.plot(x='x', y='y', ax=plt.subplot(121), title='Pandas Plot')

# Dask plot (compute first)
dask_result = dask_df.compute()
dask_result.plot(x='x', y='y', ax=plt.subplot(122), title='Dask Plot')

plt.tight_layout()
plt.show()
```

Slide 12: Real-life Example: Log Analysis

Analyzing server logs is a common task that can benefit from both Pandas and Dask, depending on the size of the log files.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Generate sample log data
log_entries = [
    f"{i},2023-05-{np.random.randint(1, 32):02d} {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d},{'GET' if np.random.random() > 0.3 else 'POST'},{np.random.choice([200, 404, 500])}"
    for i in range(1000000)
]

# Write to CSV
with open('server_logs.csv', 'w') as f:
    f.write("id,timestamp,method,status\n")
    f.write("\n".join(log_entries))

# Read with Pandas
pandas_logs = pd.read_csv('server_logs.csv', parse_dates=['timestamp'])

# Read with Dask
dask_logs = dd.read_csv('server_logs.csv', parse_dates=['timestamp'])

# Analyze HTTP status codes
pandas_status = pandas_logs['status'].value_counts()
dask_status = dask_logs['status'].value_counts().compute()

print("Pandas status code counts:")
print(pandas_status)
print("\nDask status code counts:")
print(dask_status)
```

Slide 13: Real-life Example: Geospatial Analysis

Geospatial analysis often involves large datasets, making it an ideal use case for comparing Pandas and Dask performance.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
import time

# Generate sample geospatial data
num_points = 1000000
lat = np.random.uniform(30, 50, num_points)
lon = np.random.uniform(-120, -70, num_points)
data = pd.DataFrame({'lat': lat, 'lon': lon})

# Function to check if a point is within a bounding box
def within_bbox(row, bbox):
    return (bbox[0] <= row['lon'] <= bbox[2]) and (bbox[1] <= row['lat'] <= bbox[3])

# Define a bounding box (min_lon, min_lat, max_lon, max_lat)
bbox = (-100, 35, -90, 45)

# Pandas analysis
start_time = time.time()
pandas_result = data[data.apply(within_bbox, axis=1, bbox=bbox)]
pandas_time = time.time() - start_time

# Dask analysis
dask_data = dd.from_pandas(data, npartitions=10)
start_time = time.time()
dask_result = dask_data[dask_data.apply(within_bbox, axis=1, bbox=bbox, meta=('bool')).compute()]
dask_time = time.time() - start_time

print(f"Pandas processing time: {pandas_time:.2f} seconds")
print(f"Dask processing time: {dask_time:.2f} seconds")
print(f"Points within bounding box (Pandas): {len(pandas_result)}")
print(f"Points within bounding box (Dask): {len(dask_result)}")
```

Slide 14: Conclusion

Pandas and Dask are both powerful tools for data analysis in Python. Pandas is excellent for smaller datasets that fit in memory and offers a rich set of features for data manipulation. Dask extends these capabilities to larger datasets by enabling parallel processing and out-of-core computation. Choose Pandas for quick analysis of smaller datasets and prototyping, and consider Dask when dealing with large-scale data processing tasks or when you need to leverage distributed computing resources.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
sizes = [1e3, 1e4, 1e5, 1e6]
pandas_times = []
dask_times = []

for size in sizes:
    data = pd.DataFrame({'A': np.random.randn(int(size)), 'B': np.random.randn(int(size))})
    
    # Pandas performance
    start = time.time()
    _ = data.groupby('A').B.mean()
    pandas_times.append(time.time() - start)
    
    # Dask performance
    dask_data = dd.from_pandas(data, npartitions=4)
    start = time.time()
    _ = dask_data.groupby('A').B.mean().compute()
    dask_times.append(time.time() - start)

# Plot performance comparison
plt.figure(figsize=(10, 6))
plt.plot(sizes, pandas_times, label='Pandas')
plt.plot(sizes, dask_times, label='Dask')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (s)')
plt.title('Pandas vs Dask Performance')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For more information on Pandas and Dask, consider exploring the following resources:

1. Pandas documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. Dask documentation: [https://docs.dask.org/en/latest/](https://docs.dask.org/en/latest/)
3. "Scaling Pandas: Comparing Dask, Ray, Modin, Vaex, and RAPIDS" (arXiv:2202.04935): [https://arxiv.org/abs/2202.04935](https://arxiv.org/abs/2202.04935)
4. "A Comparative Study of Distributed DataFrame Systems" (arXiv:2011.00719): [https://arxiv.org/abs/2011.00719](https://arxiv.org/abs/2011.00719)

These resources provide in-depth information on both libraries and comparative studies of distributed DataFrame systems, which can help you make informed decisions about which tool to use for your specific data analysis needs.

