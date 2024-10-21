## Polars vs. Pandas Accelerating Data Analysis

Slide 1: Introduction to Polars and Pandas

Polars and Pandas are both powerful data manipulation libraries in Python. While Pandas has been the go-to library for data analysis for years, Polars is a newer contender that promises significant performance improvements. This presentation will compare these two libraries, focusing on their strengths, differences, and how Polars can potentially accelerate your data analysis tasks.

```python
import polars as pl
import pandas as pd

# Create sample dataframes
polars_df = pl.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
pandas_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

print("Polars DataFrame:")
print(polars_df)
print("\nPandas DataFrame:")
print(pandas_df)
```

Slide 2: Performance Comparison

One of the main advantages of Polars over Pandas is its superior performance. Polars is built on Apache Arrow and is designed to be memory-efficient and leverage modern CPU architectures. This often results in significantly faster operations, especially for large datasets.

```python
import time
import polars as pl
import pandas as pd
import numpy as np

# Generate large dataset
size = 1_000_000
data = {'A': np.random.rand(size), 'B': np.random.rand(size)}

# Pandas
start = time.time()
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas.groupby('A').agg({'B': ['mean', 'max']})
pandas_time = time.time() - start

# Polars
start = time.time()
df_polars = pl.DataFrame(data)
result_polars = df_polars.groupby('A').agg([
    pl.col('B').mean(),
    pl.col('B').max()
])
polars_time = time.time() - start

print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Polars time: {polars_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

Slide 3: Memory Efficiency

Polars is designed to be memory-efficient, which is crucial when working with large datasets. It uses Apache Arrow as its memory model, allowing for zero-copy operations and efficient memory usage.

```python
import polars as pl
import pandas as pd
import sys

# Create large dataframes
size = 1_000_000
data = {'A': range(size), 'B': range(size)}

df_pandas = pd.DataFrame(data)
df_polars = pl.DataFrame(data)

# Check memory usage
pandas_memory = sys.getsizeof(df_pandas)
polars_memory = sys.getsizeof(df_polars)

print(f"Pandas DataFrame size: {pandas_memory / 1e6:.2f} MB")
print(f"Polars DataFrame size: {polars_memory / 1e6:.2f} MB")
print(f"Memory reduction: {pandas_memory / polars_memory:.2f}x")
```

Slide 4: Lazy Evaluation in Polars

Polars introduces the concept of lazy evaluation, which allows for query optimization before execution. This can lead to significant performance improvements, especially for complex operations on large datasets.

```python
import polars as pl

# Create a large dataframe
df = pl.DataFrame({'A': range(1_000_000), 'B': range(1_000_000)})

# Define a lazy computation
lazy_result = (
    df.lazy()
    .filter(pl.col('A') > 500_000)
    .groupby('B')
    .agg([pl.col('A').mean().alias('A_mean')])
    .sort('A_mean', descending=True)
    .limit(10)
)

# Execute the lazy computation
result = lazy_result.collect()
print(result)
```

Slide 5: Data Types and Schema

Polars uses a strongly typed schema, which contributes to its performance advantages. It supports a wide range of data types, including temporal types, and allows for easy schema manipulation.

```python
import polars as pl

# Create a dataframe with various data types
df = pl.DataFrame({
    'int_col': [1, 2, 3],
    'float_col': [1.1, 2.2, 3.3],
    'str_col': ['a', 'b', 'c'],
    'date_col': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'bool_col': [True, False, True]
})

# Convert date_col to Date type
df = df.with_columns(pl.col('date_col').cast(pl.Date))

# Print schema
print(df.schema)

# Change data type of float_col to integer
df = df.with_columns(pl.col('float_col').cast(pl.Int64))

# Print updated schema
print(df.schema)
```

Slide 6: Handling Missing Data

Both Polars and Pandas provide methods for handling missing data, but Polars offers some unique features. For example, Polars uses None to represent missing values for all data types, which can be more intuitive than Pandas' NaN for numeric types.

```python
import polars as pl
import pandas as pd
import numpy as np

# Create dataframes with missing values
polars_df = pl.DataFrame({
    'A': [1, None, 3],
    'B': [4.0, 5.0, None],
    'C': ['x', None, 'z']
})

pandas_df = pd.DataFrame({
    'A': [1, np.nan, 3],
    'B': [4.0, 5.0, np.nan],
    'C': ['x', None, 'z']
})

print("Polars DataFrame:")
print(polars_df)
print("\nPandas DataFrame:")
print(pandas_df)

# Fill missing values
polars_filled = polars_df.fill_null(strategy='forward')
pandas_filled = pandas_df.fillna(method='ffill')

print("\nPolars Filled:")
print(polars_filled)
print("\nPandas Filled:")
print(pandas_filled)
```

Slide 7: Grouping and Aggregation

Both Polars and Pandas offer powerful grouping and aggregation capabilities, but Polars' syntax can be more expressive and its performance is often superior.

```python
import polars as pl
import pandas as pd
import numpy as np

# Create sample data
data = {
    'category': ['A', 'B', 'A', 'B', 'A', 'B'] * 1000,
    'value1': np.random.rand(6000),
    'value2': np.random.rand(6000)
}

# Polars
df_polars = pl.DataFrame(data)
result_polars = df_polars.groupby('category').agg([
    pl.col('value1').mean().alias('value1_mean'),
    pl.col('value2').sum().alias('value2_sum'),
    pl.col('value1').count().alias('count')
])

# Pandas
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas.groupby('category').agg({
    'value1': 'mean',
    'value2': 'sum',
    'value1': 'count'
}).rename(columns={'value1': 'value1_mean', 'value2': 'value2_sum', 'value1': 'count'})

print("Polars result:")
print(result_polars)
print("\nPandas result:")
print(result_pandas)
```

Slide 8: Joining DataFrames

Joining dataframes is a common operation in data analysis. Both Polars and Pandas support various types of joins, but Polars often performs these operations faster, especially on large datasets.

```python
import polars as pl
import pandas as pd
import time

# Create sample dataframes
df1 = pl.DataFrame({'key': range(1000000), 'value_a': range(1000000)})
df2 = pl.DataFrame({'key': range(500000, 1500000), 'value_b': range(1000000)})

# Polars join
start = time.time()
result_polars = df1.join(df2, on='key', how='inner')
polars_time = time.time() - start

# Convert to Pandas for comparison
pdf1 = df1.to_pandas()
pdf2 = df2.to_pandas()

# Pandas join
start = time.time()
result_pandas = pdf1.merge(pdf2, on='key', how='inner')
pandas_time = time.time() - start

print(f"Polars join time: {polars_time:.2f} seconds")
print(f"Pandas join time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

Slide 9: Handling Time Series Data

Time series data is common in many fields. Both Polars and Pandas offer functionalities for working with temporal data, but Polars' performance can be particularly advantageous for large time series datasets.

```python
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate time series data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(365)]
values = np.random.randn(365)

# Polars
df_polars = pl.DataFrame({'date': dates, 'value': values})
df_polars = df_polars.with_columns(pl.col('date').cast(pl.Date))

# Resample to monthly frequency and calculate mean
result_polars = df_polars.groupby_dynamic('date', every='1mo').agg([
    pl.col('value').mean().alias('monthly_mean')
])

# Pandas
df_pandas = pd.DataFrame({'date': dates, 'value': values})
df_pandas.set_index('date', inplace=True)

# Resample to monthly frequency and calculate mean
result_pandas = df_pandas.resample('M').mean()

print("Polars result:")
print(result_polars)
print("\nPandas result:")
print(result_pandas)
```

Slide 10: String Operations

Efficient string operations are crucial for text data processing. Polars provides fast string manipulation functions that can significantly speed up text-based computations.

```python
import polars as pl
import pandas as pd
import time

# Create a large dataframe with string data
size = 1_000_000
data = {'text': ['Hello world! ' * 10] * size}

# Polars
df_polars = pl.DataFrame(data)
start = time.time()
result_polars = df_polars.with_columns(
    pl.col('text').str.to_uppercase().alias('upper'),
    pl.col('text').str.contains('world').alias('contains_world'),
    pl.col('text').str.length().alias('length')
)
polars_time = time.time() - start

# Pandas
df_pandas = pd.DataFrame(data)
start = time.time()
result_pandas = df_pandas.copy()
result_pandas['upper'] = df_pandas['text'].str.upper()
result_pandas['contains_world'] = df_pandas['text'].str.contains('world')
result_pandas['length'] = df_pandas['text'].str.len()
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")
```

Slide 11: Handling Large Datasets

One of Polars' key strengths is its ability to efficiently handle large datasets that might not fit into memory. Its out-of-core processing capabilities allow it to work with datasets larger than available RAM.

```python
import polars as pl
import os

# Function to generate a large CSV file
def generate_large_csv(filename, size):
    with open(filename, 'w') as f:
        f.write('id,value\n')
        for i in range(size):
            f.write(f'{i},{i*2}\n')

# Generate a 1GB CSV file
filename = 'large_file.csv'
generate_large_csv(filename, 50_000_000)

# Read and process the file using Polars' lazy evaluation
df = pl.scan_csv(filename)
result = df.filter(pl.col('value') > 1_000_000).select([
    pl.col('id'),
    pl.col('value'),
    (pl.col('value') * 2).alias('double_value')
]).collect()

print(result.head())

# Clean up
os.remove(filename)
```

Slide 12: Real-Life Example: Weather Data Analysis

Let's compare Polars and Pandas in a real-life scenario: analyzing a large weather dataset. We'll perform some common operations like filtering, grouping, and aggregation.

```python
import polars as pl
import pandas as pd
import numpy as np
import time

# Generate synthetic weather data
size = 1_000_000
dates = pd.date_range('2020-01-01', periods=size)
cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], size)
temperatures = np.random.normal(20, 10, size)
humidity = np.random.uniform(30, 80, size)

# Polars
start = time.time()
df_polars = pl.DataFrame({
    'date': dates,
    'city': cities,
    'temperature': temperatures,
    'humidity': humidity
})

result_polars = (
    df_polars.filter(pl.col('temperature') > 25)
    .groupby(['city', pl.col('date').dt.month()])
    .agg([
        pl.col('temperature').mean().alias('avg_temp'),
        pl.col('humidity').mean().alias('avg_humidity')
    ])
    .sort(['city', 'date'])
)
polars_time = time.time() - start

# Pandas
start = time.time()
df_pandas = pd.DataFrame({
    'date': dates,
    'city': cities,
    'temperature': temperatures,
    'humidity': humidity
})

result_pandas = (
    df_pandas[df_pandas['temperature'] > 25]
    .groupby(['city', df_pandas['date'].dt.month])
    .agg({'temperature': 'mean', 'humidity': 'mean'})
    .reset_index()
    .sort_values(['city', 'date'])
)
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")

print("\nPolars result (first 5 rows):")
print(result_polars.head())
print("\nPandas result (first 5 rows):")
print(result_pandas.head())
```

Slide 13: Real-Life Example: Text Processing

Text processing is a common task in data analysis. Let's compare Polars and Pandas in a scenario where we need to analyze word frequencies in a large corpus of text.

```python
import polars as pl
import pandas as pd
import time
import re

# Generate a large corpus of text
corpus = " ".join(["The quick brown fox jumps over the lazy dog"] * 100000)

# Polars
start = time.time()
df_polars = pl.DataFrame({'text': [corpus]})
words_polars = (
    df_polars.select(pl.col('text').str.to_lowercase().str.split_whitespace())
    .explode('text')
    .select(pl.col('text').str.replace_all(r'[^\w\s]', '').alias('word'))
    .filter(pl.col('word') != '')
    .groupby('word')
    .count()
    .sort('count', descending=True)
    .limit(10)
)
polars_time = time.time() - start

# Pandas
start = time.time()
df_pandas = pd.DataFrame({'text': [corpus]})
words_pandas = (
    df_pandas['text'].str.lower()
    .str.split()
    .explode()
    .str.replace(r'[^\w\s]', '', regex=True)
    .value_counts()
    .reset_index()
    .rename(columns={'index': 'word', 'text': 'count'})
    .head(10)
)
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Speedup: {pandas_time / polars_time:.2f}x")

print("\nPolars result:")
print(words_polars)
print("\nPandas result:")
print(words_pandas)
```

Slide 14: Conclusion

Throughout this presentation, we've explored the key differences between Polars and Pandas. Polars offers significant performance improvements, especially for large datasets, due to its efficient memory management and leverage of modern CPU architectures. Its lazy evaluation feature allows for query optimization, further enhancing performance for complex operations.

While Pandas remains a robust and widely-used library with a mature ecosystem, Polars presents a compelling alternative for data scientists and analysts working with large datasets or requiring high-performance computations. The choice between Polars and Pandas depends on your specific use case, dataset size, and performance requirements.

```python
import polars as pl
import pandas as pd
import numpy as np
import time

# Generate a large dataset
size = 10_000_000
data = {'A': np.random.rand(size), 'B': np.random.rand(size)}

# Polars
start = time.time()
df_polars = pl.DataFrame(data)
result_polars = df_polars.filter(pl.col('A') > 0.5).groupby('B').agg(pl.col('A').mean())
polars_time = time.time() - start

# Pandas
start = time.time()
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas[df_pandas['A'] > 0.5].groupby('B')['A'].mean().reset_index()
pandas_time = time.time() - start

print(f"Polars time: {polars_time:.2f} seconds")
print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Overall speedup: {pandas_time / polars_time:.2f}x")
```

Slide 15: Additional Resources

For those interested in learning more about Polars and its capabilities, here are some valuable resources:

1.  Polars Documentation: [https://pola-rs.github.io/polars-book/](https://pola-rs.github.io/polars-book/)
2.  Polars GitHub Repository: [https://github.com/pola-rs/polars](https://github.com/pola-rs/polars)
3.  "Polars: Lightning-fast DataFrame library for Rust and Python" (arXiv:2211.14502): [https://arxiv.org/abs/2211.14502](https://arxiv.org/abs/2211.14502)
4.  Pandas Documentation (for comparison): [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

These resources provide in-depth information about Polars' features, performance characteristics, and usage examples. The arXiv paper offers a technical overview of Polars' design and performance benchmarks.

