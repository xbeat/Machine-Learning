## Polars A Beginners Guide to the Python Dataframe Library

Slide 1: Introduction to Polars Polars is a fast and efficient DataFrame library for Python, written in Rust. It provides a user-friendly interface similar to pandas but with better performance and memory efficiency.

```python
import polars as pl

# Create a DataFrame from a Python dictionary
data = {"names": ["Alice", "Bob", "Charlie"], "ages": [25, 32, 18]}
df = pl.DataFrame(data)
print(df)
```

Slide 2: Reading Data from CSV Polars can read data from various sources, including CSV files. It provides a convenient function to read CSV data into a DataFrame.

```python
# Read a CSV file
df = pl.read_csv("data.csv")
print(df)
```

Slide 3: Data Selection Polars supports familiar indexing and selection operations, similar to pandas, to access specific rows and columns of a DataFrame.

```python
# Select a column
ages = df["ages"]
print(ages)

# Select rows based on a condition
young_people = df[df["ages"] < 25]
print(young_people)
```

Slide 4: Data Manipulation Polars provides various methods to manipulate and transform data in a DataFrame, such as filtering, sorting, and applying functions.

```python
# Filter rows
filtered_df = df.filter(pl.col("ages") > 25)
print(filtered_df)

# Sort DataFrame
sorted_df = df.sort("ages", descending=True)
print(sorted_df)
```

Slide 5: Aggregations Polars supports efficient aggregations on DataFrames, including sum, mean, min, max, and more.

```python
# Calculate the mean age
mean_age = df.select(pl.col("ages").mean().alias("mean_age"))
print(mean_age)

# Group by and aggregate
grouped = df.groupby("ages").agg(pl.count().alias("count"))
print(grouped)
```

Slide 6: Joining DataFrames Polars allows you to join multiple DataFrames based on common columns, similar to SQL-like joins.

```python
# Create another DataFrame
df2 = pl.DataFrame({"names": ["Alice", "Eve", "Frank"], "cities": ["New York", "London", "Paris"]})

# Join DataFrames
joined_df = df.join(df2, on="names", how="left")
print(joined_df)
```

Slide 7: Missing Data Handling Polars provides methods to handle missing data in DataFrames, such as filling or dropping rows/columns with null values.

```python
# Drop rows with null values
df_no_nulls = df.drop_nulls()
print(df_no_nulls)

# Fill null values
filled_df = df.fill_null(0)
print(filled_df)
```

Slide 8: Data Visualization Polars integrates with popular data visualization libraries like Matplotlib and Plotly, allowing you to create visualizations directly from DataFrames.

```python
import matplotlib.pyplot as plt

# Plot a histogram
df.plot.histogram("ages", buckets=5)
plt.show()
```

Slide 9: Performance Comparison Polars aims to provide better performance compared to pandas, especially for larger datasets. Here's an example to compare the performance of common operations.

```python
import pandas as pd
import timeit

# Create a large DataFrame
large_df = pl.DataFrame({"values": [i for i in range(1_000_000)]})
pandas_df = pd.DataFrame(large_df.to_dict())

# Compare performance
polars_time = timeit.timeit(lambda: large_df.filter(pl.col("values") > 500_000), number=1000)
pandas_time = timeit.timeit(lambda: pandas_df[pandas_df["values"] > 500_000], number=1000)

print(f"Polars time: {polars_time:.6f} seconds")
print(f"Pandas time: {pandas_time:.6f} seconds")
```

Slide 10: Lazy Evaluation Polars supports lazy evaluation, which means operations are not executed until necessary. This can lead to improved performance for complex operations.

```python
# Create a lazy expression
lazy_df = pl.scan_csv("data.csv") \
            .filter(pl.col("age") > 25) \
            .select(["name", "age"])

# Execute the lazy expression
result = lazy_df.collect()
print(result)
```

Slide 11: Data Types Polars supports a variety of data types, including numeric, string, date/time, and categorical types. It also provides methods for type conversion and handling missing values.

```python
# Create a DataFrame with mixed data types
data = {"names": ["Alice", "Bob", "Charlie"], "ages": [25, None, 18], "dates": ["2022-01-01", "2021-05-15", None]}
df = pl.DataFrame(data)

# Convert data types
df = df.with_columns(
    pl.col("ages").cast(pl.Int64),
    pl.col("dates").str.strptime(pl.Date, "%Y-%m-%d")
)
print(df)
```

Slide 12: Parallel Processing Polars can leverage multiple CPU cores for parallel processing, leading to faster computations on larger datasets.

```python
# Set the number of threads for parallel processing
pl.set_env_threaded(True)
pl.set_env_threading_max_num_threads(4)

# Perform a computation in parallel
result = df.select(pl.col("values").apply(lambda x: x * 2).parallel())
print(result)
```

Slide 13: User-Defined Functions (UDFs) Polars allows you to define custom functions (UDFs) and apply them to DataFrames. UDFs can be written in Rust or Python.

```python
# Define a Python UDF
import polars as pl

def square(x):
    return x ** 2

# Apply the UDF to a column
df = pl.DataFrame({"values": [1, 2, 3, 4, 5]})
squared_df = df.with_column(pl.col("values").apply(square))
print(squared_df)
```

Slide 14: Integration with Other Libraries Polars can integrate with other Python libraries, such as NumPy, Scikit-learn, and Dask, enabling seamless data processing workflows.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Convert Polars DataFrame to NumPy array
X = df["features"].to_numpy()
y = df["target"].to_numpy()

# Train a linear regression model
model = LinearRegression().fit(X, y)

# Make predictions on a new DataFrame
new_df = pl.DataFrame({"features": [[1, 2], [3, 4]]})
predictions = model.predict(new_df["features"].to_numpy())
print(predictions)
```

This slideshow covers various topics related to Polars, including data manipulation, aggregations, joining, visualization, performance comparison, lazy evaluation, data types, parallel processing, user-defined functions, and integration with other libraries. The examples are designed to be actionable and suitable for beginner to intermediate level users.

