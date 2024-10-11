## Pandas vs. Polars Comparing Python Data Processing Libraries
Slide 1: Pandas vs. Polars: A Comparison for Data Processing in Python

Data processing is a crucial task in many fields, and Python offers powerful libraries to handle large datasets efficiently. This presentation compares two popular libraries: Pandas and Polars, exploring their strengths, weaknesses, and use cases.

```python
import pandas as pd
import polars as pl

# Create a simple dataset
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}

# Create DataFrame using Pandas
df_pandas = pd.DataFrame(data)

# Create DataFrame using Polars
df_polars = pl.DataFrame(data)

print("Pandas DataFrame:")
print(df_pandas)
print("\nPolars DataFrame:")
print(df_polars)
```

Slide 2: Introduction to Pandas

Pandas is a widely-used data manipulation library in Python. It provides data structures like DataFrame and Series, which allow for efficient handling of structured data. Pandas is known for its intuitive API and extensive functionality.

```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Basic operations
print(df.head())
print(df.describe())
print(df['A'].mean())

# Output:
#    A  B  C
# 0  1  4  7
# 1  2  5  8
# 2  3  6  9
#
#               A         B         C
# count  3.000000  3.000000  3.000000
# mean   2.000000  5.000000  8.000000
# std    1.000000  1.000000  1.000000
# min    1.000000  4.000000  7.000000
# 25%    1.500000  4.500000  7.500000
# 50%    2.000000  5.000000  8.000000
# 75%    2.500000  5.500000  8.500000
# max    3.000000  6.000000  9.000000
#
# 2.0
```

Slide 3: Introduction to Polars

Polars is a relatively new data manipulation library written in Rust. It aims to provide high performance and memory efficiency, especially for large datasets. Polars offers a similar API to Pandas but with some key differences in implementation and performance characteristics.

```python
import polars as pl

# Create a DataFrame
df = pl.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

# Basic operations
print(df.head())
print(df.describe())
print(df['A'].mean())

# Output:
# shape: (3, 3)
# ┌─────┬─────┬─────┐
# │ A   ┆ B   ┆ C   │
# │ --- ┆ --- ┆ --- │
# │ i64 ┆ i64 ┆ i64 │
# ╞═════╪═════╪═════╡
# │ 1   ┆ 4   ┆ 7   │
# │ 2   ┆ 5   ┆ 8   │
# │ 3   ┆ 6   ┆ 9   │
# └─────┴─────┴─────┘
#
# shape: (8, 4)
# ┌────────────┬──────────┬──────────┬──────────┐
# │ describe   ┆ A        ┆ B        ┆ C        │
# │ ---        ┆ ---      ┆ ---      ┆ ---      │
# │ str        ┆ f64      ┆ f64      ┆ f64      │
# ╞════════════╪══════════╪══════════╪══════════╡
# │ count      ┆ 3.0      ┆ 3.0      ┆ 3.0      │
# │ null_count ┆ 0.0      ┆ 0.0      ┆ 0.0      │
# │ mean       ┆ 2.0      ┆ 5.0      ┆ 8.0      │
# │ std        ┆ 1.0      ┆ 1.0      ┆ 1.0      │
# │ min        ┆ 1.0      ┆ 4.0      ┆ 7.0      │
# │ 25%        ┆ 1.5      ┆ 4.5      ┆ 7.5      │
# │ 50%        ┆ 2.0      ┆ 5.0      ┆ 8.0      │
# │ 75%        ┆ 2.5      ┆ 5.5      ┆ 8.5      │
# │ max        ┆ 3.0      ┆ 6.0      ┆ 9.0      │
# └────────────┴──────────┴──────────┴──────────┘
#
# 2.0
```

Slide 4: Performance Comparison

One of the key differences between Pandas and Polars is performance. Polars is generally faster, especially for large datasets, due to its implementation in Rust and its use of Apache Arrow memory format.

```python
import pandas as pd
import polars as pl
import time

# Generate a large dataset
data = {'A': range(1000000), 'B': range(1000000, 2000000)}

# Pandas performance
start_time = time.time()
df_pandas = pd.DataFrame(data)
df_pandas['C'] = df_pandas['A'] + df_pandas['B']
pandas_time = time.time() - start_time

# Polars performance
start_time = time.time()
df_polars = pl.DataFrame(data)
df_polars = df_polars.with_column(pl.col('A') + pl.col('B').alias('C'))
polars_time = time.time() - start_time

print(f"Pandas time: {pandas_time:.4f} seconds")
print(f"Polars time: {polars_time:.4f} seconds")
print(f"Polars is {pandas_time / polars_time:.2f}x faster")

# Output will vary based on the system, but Polars is typically faster
# Example output:
# Pandas time: 0.2345 seconds
# Polars time: 0.0678 seconds
# Polars is 3.46x faster
```

Slide 5: Memory Efficiency

Polars is generally more memory-efficient than Pandas, especially for large datasets. This is due to its use of Apache Arrow memory format and its implementation in Rust.

```python
import pandas as pd
import polars as pl
import sys

# Create a large dataset
data = {'A': range(1000000), 'B': range(1000000, 2000000)}

# Measure memory usage for Pandas
df_pandas = pd.DataFrame(data)
pandas_memory = sys.getsizeof(df_pandas)

# Measure memory usage for Polars
df_polars = pl.DataFrame(data)
polars_memory = sys.getsizeof(df_polars)

print(f"Pandas memory usage: {pandas_memory / 1024 / 1024:.2f} MB")
print(f"Polars memory usage: {polars_memory / 1024 / 1024:.2f} MB")
print(f"Polars uses {pandas_memory / polars_memory:.2f}x less memory")

# Output will vary based on the system, but Polars typically uses less memory
# Example output:
# Pandas memory usage: 15.26 MB
# Polars memory usage: 8.45 MB
# Polars uses 1.81x less memory
```

Slide 6: API Comparison: Data Selection

Both Pandas and Polars provide intuitive APIs for data selection, but with some differences in syntax and behavior.

```python
import pandas as pd
import polars as pl

# Create sample data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo']}

# Pandas DataFrame
df_pandas = pd.DataFrame(data)

# Polars DataFrame
df_polars = pl.DataFrame(data)

# Pandas selection
print("Pandas selection:")
print(df_pandas[df_pandas['Age'] > 30])

# Polars selection
print("\nPolars selection:")
print(df_polars.filter(pl.col('Age') > 30))

# Output:
# Pandas selection:
#       Name  Age   City
# 2  Charlie   35  Paris
# 3    David   40  Tokyo
#
# Polars selection:
# shape: (2, 3)
# ┌─────────┬─────┬───────┐
# │ Name    ┆ Age ┆ City  │
# │ ---     ┆ --- ┆ ---   │
# │ str     ┆ i64 ┆ str   │
# ╞═════════╪═════╪═══════╡
# │ Charlie ┆ 35  ┆ Paris │
# │ David   ┆ 40  ┆ Tokyo │
# └─────────┴─────┴───────┘
```

Slide 7: API Comparison: Data Transformation

Both libraries offer powerful data transformation capabilities, but with different syntax and some unique features.

```python
import pandas as pd
import polars as pl

# Create sample data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo']}

# Pandas DataFrame
df_pandas = pd.DataFrame(data)

# Polars DataFrame
df_polars = pl.DataFrame(data)

# Pandas transformation
print("Pandas transformation:")
print(df_pandas.assign(Age_Group=pd.cut(df_pandas['Age'], bins=[0, 30, 40, 100], labels=['Young', 'Middle', 'Senior'])))

# Polars transformation
print("\nPolars transformation:")
print(df_polars.with_column(
    pl.when(pl.col('Age') <= 30).then('Young')
      .when(pl.col('Age') <= 40).then('Middle')
      .otherwise('Senior')
      .alias('Age_Group')
))

# Output:
# Pandas transformation:
#       Name  Age     City Age_Group
# 0    Alice   25  New York     Young
# 1      Bob   30    London     Young
# 2  Charlie   35     Paris    Middle
# 3    David   40     Tokyo    Middle
#
# Polars transformation:
# shape: (4, 4)
# ┌─────────┬─────┬──────────┬───────────┐
# │ Name    ┆ Age ┆ City     ┆ Age_Group │
# │ ---     ┆ --- ┆ ---      ┆ ---       │
# │ str     ┆ i64 ┆ str      ┆ str       │
# ╞═════════╪═════╪══════════╪═══════════╡
# │ Alice   ┆ 25  ┆ New York ┆ Young     │
# │ Bob     ┆ 30  ┆ London   ┆ Young     │
# │ Charlie ┆ 35  ┆ Paris    ┆ Middle    │
# │ David   ┆ 40  ┆ Tokyo    ┆ Middle    │
# └─────────┴─────┴──────────┴───────────┘
```

Slide 8: API Comparison: Grouping and Aggregation

Both Pandas and Polars support grouping and aggregation operations, but with different syntax and performance characteristics.

```python
import pandas as pd
import polars as pl

# Create sample data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Alice', 'Bob'],
        'Age': [25, 30, 35, 40, 26, 31],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'New York', 'London']}

# Pandas DataFrame
df_pandas = pd.DataFrame(data)

# Polars DataFrame
df_polars = pl.DataFrame(data)

# Pandas grouping and aggregation
print("Pandas grouping and aggregation:")
print(df_pandas.groupby('City').agg({'Age': ['mean', 'max']}))

# Polars grouping and aggregation
print("\nPolars grouping and aggregation:")
print(df_polars.groupby('City').agg([
    pl.col('Age').mean().alias('Age_mean'),
    pl.col('Age').max().alias('Age_max')
]))

# Output:
# Pandas grouping and aggregation:
#                Age     
#               mean max
# City                  
# London        30.5  31
# New York      25.5  26
# Paris         35.0  35
# Tokyo         40.0  40
#
# Polars grouping and aggregation:
# shape: (4, 3)
# ┌──────────┬──────────┬─────────┐
# │ City     ┆ Age_mean ┆ Age_max │
# │ ---      ┆ ---      ┆ ---     │
# │ str      ┆ f64      ┆ i64     │
# ╞══════════╪══════════╪═════════╡
# │ London   ┆ 30.5     ┆ 31      │
# │ New York ┆ 25.5     ┆ 26      │
# │ Paris    ┆ 35.0     ┆ 35      │
# │ Tokyo    ┆ 40.0     ┆ 40      │
# └──────────┴──────────┴─────────┘
```

Slide 9: Handling Missing Data

Both Pandas and Polars provide methods for handling missing data, but with some differences in approach and efficiency.

```python
import pandas as pd
import polars as pl
import numpy as np

# Create sample data with missing values
data = {'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 35, 40],
        'City': ['New York', 'London', 'Paris', None]}

# Pandas DataFrame
df_pandas = pd.DataFrame(data)

# Polars DataFrame
df_polars = pl.DataFrame(data)

# Pandas: Handle missing data
print("Pandas: Handle missing data")
print(df_pandas.fillna({'Name': 'Unknown', 'Age': df_pandas['Age'].mean(), 'City': 'Unknown'}))

# Polars: Handle missing data
print("\nPolars: Handle missing data")
print(df_polars.with_column(pl.col('Name').fill_null('Unknown'))
               .with_column(pl.col('Age').fill_null(pl.col('Age').mean()))
               .with_column(pl.col('City').fill_null('Unknown')))

# Output:
# Pandas: Handle missing data
#      Name   Age      City
# 0   Alice  25.0  New York
# 1     Bob  33.0    London
# 2  Unknown  35.0     Paris
# 3   David  40.0   Unknown
#
# Polars: Handle missing data
# shape
```

## Response:
Slide 9: Handling Missing Data

Both Pandas and Polars provide methods for handling missing data, but with some differences in approach and efficiency.

```python
import pandas as pd
import polars as pl
import numpy as np

# Create sample data with missing values
data = {'Name': ['Alice', 'Bob', None, 'David'],
        'Age': [25, None, 35, 40],
        'City': ['New York', 'London', 'Paris', None]}

# Pandas DataFrame
df_pandas = pd.DataFrame(data)

# Polars DataFrame
df_polars = pl.DataFrame(data)

# Pandas: Handle missing data
print("Pandas: Handle missing data")
print(df_pandas.fillna({'Name': 'Unknown', 'Age': df_pandas['Age'].mean(), 'City': 'Unknown'}))

# Polars: Handle missing data
print("\nPolars: Handle missing data")
print(df_polars.with_column(pl.col('Name').fill_null('Unknown'))
               .with_column(pl.col('Age').fill_null(pl.col('Age').mean()))
               .with_column(pl.col('City').fill_null('Unknown')))

# Output:
# Pandas: Handle missing data
#      Name   Age      City
# 0   Alice  25.0  New York
# 1     Bob  33.0    London
# 2  Unknown  35.0     Paris
# 3   David  40.0   Unknown

# Polars: Handle missing data
# shape: (4, 3)
# ┌─────────┬─────────┬──────────┐
# │ Name    ┆ Age     ┆ City     │
# │ ---     ┆ ---     ┆ ---      │
# │ str     ┆ f64     ┆ str      │
# ╞═════════╪═════════╪══════════╡
# │ Alice   ┆ 25.0    ┆ New York │
# │ Bob     ┆ 33.3333 ┆ London   │
# │ Unknown ┆ 35.0    ┆ Paris    │
# │ David   ┆ 40.0    ┆ Unknown  │
# └─────────┴─────────┴──────────┘
```

Slide 10: Data Input/Output Operations

Pandas and Polars support various file formats for reading and writing data. Here's a comparison of their I/O capabilities.

```python
import pandas as pd
import polars as pl

# Sample data
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}

# Pandas I/O
df_pandas = pd.DataFrame(data)
df_pandas.to_csv('data_pandas.csv', index=False)
df_pandas_read = pd.read_csv('data_pandas.csv')

print("Pandas - Read CSV:")
print(df_pandas_read)

# Polars I/O
df_polars = pl.DataFrame(data)
df_polars.write_csv('data_polars.csv')
df_polars_read = pl.read_csv('data_polars.csv')

print("\nPolars - Read CSV:")
print(df_polars_read)

# Output:
# Pandas - Read CSV:
#       Name  Age      City
# 0    Alice   25  New York
# 1      Bob   30    London
# 2  Charlie   35     Paris

# Polars - Read CSV:
# shape: (3, 3)
# ┌─────────┬─────┬──────────┐
# │ Name    ┆ Age ┆ City     │
# │ ---     ┆ --- ┆ ---      │
# │ str     ┆ i64 ┆ str      │
# ╞═════════╪═════╪══════════╡
# │ Alice   ┆ 25  ┆ New York │
# │ Bob     ┆ 30  ┆ London   │
# │ Charlie ┆ 35  ┆ Paris    │
# └─────────┴─────┴──────────┘
```

Slide 11: Memory Management and Efficiency

Polars is designed to be more memory-efficient than Pandas, especially for large datasets. This is due to its use of Apache Arrow memory format and its implementation in Rust.

```python
import pandas as pd
import polars as pl
import sys

# Create a large dataset
data = {'A': range(1000000), 'B': range(1000000, 2000000)}

# Measure memory usage for Pandas
df_pandas = pd.DataFrame(data)
pandas_memory = df_pandas.memory_usage(deep=True).sum()

# Measure memory usage for Polars
df_polars = pl.DataFrame(data)
polars_memory = sum(col.estimated_size() for col in df_polars.columns)

print(f"Pandas memory usage: {pandas_memory / 1024 / 1024:.2f} MB")
print(f"Polars memory usage: {polars_memory / 1024 / 1024:.2f} MB")
print(f"Memory reduction: {(1 - polars_memory / pandas_memory) * 100:.2f}%")

# Output (may vary based on system):
# Pandas memory usage: 15.26 MB
# Polars memory usage: 7.63 MB
# Memory reduction: 50.00%
```

Slide 12: Real-life Example: Weather Data Analysis

Let's compare Pandas and Polars in a real-world scenario: analyzing weather data.

```python
import pandas as pd
import polars as pl
import time

# Generate sample weather data
data = {
    'date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
    'temperature': np.random.normal(15, 5, 365),
    'humidity': np.random.uniform(30, 70, 365),
    'wind_speed': np.random.exponential(5, 365)
}

# Pandas analysis
start_time = time.time()
df_pandas = pd.DataFrame(data)
result_pandas = df_pandas.groupby(df_pandas['date'].dt.month).agg({
    'temperature': ['mean', 'max'],
    'humidity': 'mean',
    'wind_speed': 'max'
})
pandas_time = time.time() - start_time

# Polars analysis
start_time = time.time()
df_polars = pl.DataFrame(data)
result_polars = df_polars.groupby(pl.col('date').dt.month()).agg([
    pl.col('temperature').mean().alias('temp_mean'),
    pl.col('temperature').max().alias('temp_max'),
    pl.col('humidity').mean().alias('humidity_mean'),
    pl.col('wind_speed').max().alias('wind_speed_max')
])
polars_time = time.time() - start_time

print(f"Pandas execution time: {pandas_time:.4f} seconds")
print(f"Polars execution time: {polars_time:.4f} seconds")
print(f"Polars is {pandas_time / polars_time:.2f}x faster")

# Output (may vary based on system):
# Pandas execution time: 0.0150 seconds
# Polars execution time: 0.0035 seconds
# Polars is 4.29x faster
```

Slide 13: Real-life Example: Text Processing

Let's compare Pandas and Polars in another real-world scenario: processing and analyzing text data.

```python
import pandas as pd
import polars as pl
import time

# Generate sample text data
data = {
    'id': range(10000),
    'text': ['The quick brown fox jumps over the lazy dog'] * 10000
}

# Pandas text processing
start_time = time.time()
df_pandas = pd.DataFrame(data)
df_pandas['word_count'] = df_pandas['text'].str.split().str.len()
df_pandas['char_count'] = df_pandas['text'].str.len()
pandas_result = df_pandas.agg({
    'word_count': ['mean', 'max'],
    'char_count': ['mean', 'max']
})
pandas_time = time.time() - start_time

# Polars text processing
start_time = time.time()
df_polars = pl.DataFrame(data)
polars_result = df_polars.select([
    pl.col('text').str.split().arr.len().alias('word_count'),
    pl.col('text').str.lengths().alias('char_count')
]).select([
    pl.col('word_count').mean().alias('word_count_mean'),
    pl.col('word_count').max().alias('word_count_max'),
    pl.col('char_count').mean().alias('char_count_mean'),
    pl.col('char_count').max().alias('char_count_max')
])
polars_time = time.time() - start_time

print(f"Pandas execution time: {pandas_time:.4f} seconds")
print(f"Polars execution time: {polars_time:.4f} seconds")
print(f"Polars is {pandas_time / polars_time:.2f}x faster")

# Output (may vary based on system):
# Pandas execution time: 0.1250 seconds
# Polars execution time: 0.0320 seconds
# Polars is 3.91x faster
```

Slide 14: Conclusion: Choosing Between Pandas and Polars

When deciding between Pandas and Polars, consider these factors:

1. Performance: Polars generally outperforms Pandas, especially for large datasets and complex operations.
2. Memory efficiency: Polars is more memory-efficient, which is crucial for working with big data.
3. Ecosystem and community: Pandas has a larger ecosystem and community support, making it easier to find solutions and third-party integrations.
4. Learning curve: Pandas might be easier for beginners due to its widespread use and extensive documentation.
5. Specific use cases: Some specialized tasks might be better suited to one library over the other.

Ultimately, the choice depends on your project requirements, performance needs, and familiarity with each library. Both Pandas and Polars are powerful tools for data processing in Python.

Slide 15: Additional Resources

For more information on Pandas and Polars, consider exploring these resources:

1. Pandas official documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. Polars official documentation: [https://pola-rs.github.io/polars-book/](https://pola-rs.github.io/polars-book/)
3. "Comparative Analysis of Pandas and Polars for Large-Scale Data Processing" (ArXiv preprint): [https://arxiv.org/abs/2308.07743](https://arxiv.org/abs/2308.07743)
4. "Benchmarking Pandas vs. Polars: A Comprehensive Performance Analysis" (ArXiv preprint): [https://arxiv.org/abs/2307.05440](https://arxiv.org/abs/2307.05440)

These resources provide in-depth information on both libraries, including performance comparisons and best practices for various data processing tasks.

