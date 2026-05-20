## Data Wrangling with Dask in Python
Slide 1: Introduction to Data Wrangling with Dask

Data wrangling is the process of cleaning, structuring, and enriching raw data into a desired format for better decision making in less time. Dask is a flexible library for parallel computing in Python that makes data wrangling of large datasets easier and more efficient.

```python
import dask.dataframe as dd
import pandas as pd

# Create a sample dataset
data = {'A': range(1000), 'B': range(1000, 2000)}
df = pd.DataFrame(data)

# Convert pandas DataFrame to Dask DataFrame
dask_df = dd.from_pandas(df, npartitions=4)

print(dask_df.head())
```

Slide 2: Why Use Dask for Data Wrangling?

Dask allows you to work with larger-than-memory datasets by breaking them into smaller chunks. It provides a familiar API similar to pandas, making it easier for data scientists to transition from small to big data processing.

```python
import dask.dataframe as dd
import numpy as np

# Create a large Dask DataFrame
dask_df = dd.from_array(np.random.randn(1000000, 3), columns=['A', 'B', 'C'])

# Perform operations on the large dataset
result = dask_df.mean().compute()
print(result)
```

Slide 3: Loading Data with Dask

Dask can efficiently load data from various sources, including CSV files, Parquet files, and databases. It uses lazy evaluation, meaning it only loads and processes data when necessary.

```python
import dask.dataframe as dd

# Load a large CSV file
df = dd.read_csv('large_dataset.csv')

# Load multiple CSV files
df_multi = dd.read_csv('data_*.csv')

# Load a Parquet file
df_parquet = dd.read_parquet('large_dataset.parquet')

print(df.head())
```

Slide 4: Basic Data Manipulation with Dask

Dask provides familiar pandas-like operations for data manipulation, such as filtering, selecting columns, and creating new columns.

```python
import dask.dataframe as dd

df = dd.read_csv('large_dataset.csv')

# Filter rows
filtered_df = df[df['age'] > 30]

# Select columns
selected_df = df[['name', 'age', 'salary']]

# Create a new column
df['age_squared'] = df['age'] ** 2

print(filtered_df.head())
print(selected_df.head())
print(df.head())
```

Slide 5: Aggregations and Grouping

Dask supports various aggregation operations and grouping, allowing you to summarize and analyze large datasets efficiently.

```python
import dask.dataframe as dd

df = dd.read_csv('large_dataset.csv')

# Calculate mean age
mean_age = df['age'].mean().compute()

# Group by department and calculate average salary
avg_salary_by_dept = df.groupby('department')['salary'].mean().compute()

print(f"Mean age: {mean_age}")
print("Average salary by department:")
print(avg_salary_by_dept)
```

Slide 6: Handling Missing Data

Dask provides methods to handle missing data in large datasets, similar to pandas.

```python
import dask.dataframe as dd
import numpy as np

df = dd.read_csv('large_dataset.csv')

# Check for missing values
missing_values = df.isnull().sum().compute()

# Fill missing values
df['age'] = df['age'].fillna(df['age'].mean())

# Drop rows with missing values
df_cleaned = df.dropna()

print("Missing values:")
print(missing_values)
print("\nCleaned data:")
print(df_cleaned.head())
```

Slide 7: Merging and Joining DataFrames

Dask allows you to merge and join large DataFrames efficiently, enabling complex data integration tasks.

```python
import dask.dataframe as dd

# Load two DataFrames
df1 = dd.read_csv('employees.csv')
df2 = dd.read_csv('departments.csv')

# Merge DataFrames
merged_df = dd.merge(df1, df2, on='department_id')

# Perform left join
left_joined_df = dd.merge(df1, df2, on='department_id', how='left')

print(merged_df.head())
print(left_joined_df.head())
```

Slide 8: Data Reshaping with Dask

Dask supports various data reshaping operations, including pivoting and melting, which are useful for transforming data structures.

```python
import dask.dataframe as dd
import pandas as pd

# Create a sample DataFrame
data = {'date': pd.date_range('2023-01-01', periods=1000),
        'product': ['A', 'B', 'C'] * 334,
        'sales': range(1000)}
df = dd.from_pandas(pd.DataFrame(data), npartitions=4)

# Pivot the DataFrame
pivoted_df = df.pivot_table(values='sales', index='date', columns='product')

# Melt the pivoted DataFrame back
melted_df = pivoted_df.reset_index().melt(id_vars=['date'], var_name='product', value_name='sales')

print(pivoted_df.head())
print(melted_df.head())
```

Slide 9: Time Series Analysis with Dask

Dask provides functionality for working with time series data, including resampling and rolling window calculations.

```python
import dask.dataframe as dd
import pandas as pd

# Create a sample time series DataFrame
dates = pd.date_range('2023-01-01', periods=1000, freq='H')
df = dd.from_pandas(pd.DataFrame({'timestamp': dates, 'value': range(1000)}), npartitions=4)

# Resample to daily frequency
daily_df = df.set_index('timestamp').resample('D').mean()

# Calculate 7-day rolling average
rolling_avg = df.set_index('timestamp')['value'].rolling(window='7D').mean()

print(daily_df.head())
print(rolling_avg.head())
```

Slide 10: Data Visualization with Dask

While Dask itself doesn't provide plotting capabilities, you can use it in combination with other libraries like matplotlib or seaborn for data visualization.

```python
import dask.dataframe as dd
import matplotlib.pyplot as plt

df = dd.read_csv('large_dataset.csv')

# Calculate average age by department
avg_age_by_dept = df.groupby('department')['age'].mean().compute()

# Create a bar plot
plt.figure(figsize=(10, 6))
avg_age_by_dept.plot(kind='bar')
plt.title('Average Age by Department')
plt.xlabel('Department')
plt.ylabel('Average Age')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Weather Data Analysis

In this example, we'll analyze a large weather dataset using Dask, demonstrating its capabilities in handling time series data and performing aggregations.

```python
import dask.dataframe as dd
import matplotlib.pyplot as plt

# Load weather data
weather_df = dd.read_csv('weather_data_*.csv', parse_dates=['date'])

# Calculate average temperature by month
monthly_temp = weather_df.groupby(weather_df.date.dt.to_period('M'))['temperature'].mean().compute()

# Plot average temperature by month
plt.figure(figsize=(12, 6))
monthly_temp.plot(kind='line')
plt.title('Average Monthly Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.tight_layout()
plt.show()

print(monthly_temp)
```

Slide 12: Real-Life Example: Log File Analysis

In this example, we'll use Dask to analyze large log files, demonstrating its efficiency in processing text data and performing aggregations.

```python
import dask.dataframe as dd
import matplotlib.pyplot as plt

# Load log files
logs_df = dd.read_csv('server_logs_*.log', sep='\s+', 
                      names=['timestamp', 'ip', 'status', 'bytes'])

# Convert timestamp to datetime
logs_df['timestamp'] = dd.to_datetime(logs_df['timestamp'], format='%Y-%m-%d:%H:%M:%S')

# Count requests by status code
status_counts = logs_df.groupby('status').size().compute()

# Plot status code distribution
plt.figure(figsize=(10, 6))
status_counts.plot(kind='bar')
plt.title('HTTP Status Code Distribution')
plt.xlabel('Status Code')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

print(status_counts)
```

Slide 13: Performance Optimization with Dask

Dask offers various ways to optimize performance, including adjusting the number of partitions and using the persist() method to keep data in memory.

```python
import dask.dataframe as dd
import time

df = dd.read_csv('large_dataset.csv')

# Measure execution time without optimization
start_time = time.time()
result1 = df.groupby('category')['value'].mean().compute()
end_time = time.time()
print(f"Execution time without optimization: {end_time - start_time:.2f} seconds")

# Optimize by persisting the DataFrame in memory
df_persisted = df.persist()

# Measure execution time with optimization
start_time = time.time()
result2 = df_persisted.groupby('category')['value'].mean().compute()
end_time = time.time()
print(f"Execution time with optimization: {end_time - start_time:.2f} seconds")
```

Slide 14: Error Handling and Debugging in Dask

When working with large datasets, it's crucial to handle errors gracefully and debug issues effectively. Dask provides tools to help with these tasks.

```python
import dask.dataframe as dd

def process_data(x):
    if x < 0:
        raise ValueError("Negative value encountered")
    return x ** 2

df = dd.read_csv('large_dataset.csv')

try:
    # Apply a function that may raise an error
    result = df['value'].map(process_data).compute()
except ValueError as e:
    print(f"Error occurred: {e}")
    # Handle the error, e.g., by filtering out negative values
    result = df[df['value'] >= 0]['value'].map(process_data).compute()

print(result.head())

# Debugging: Visualize the task graph
df['squared_value'] = df['value'].map(process_data)
df['squared_value'].visualize(filename='task_graph.png')
print("Task graph saved as 'task_graph.png'")
```

Slide 15: Additional Resources

For more information on Data Wrangling with Dask, consider exploring the following resources:

1. Dask Documentation: [https://docs.dask.org/](https://docs.dask.org/)
2. "Scaling Python with Dask" by Matthew Rocklin (ArXiv:1901.10124): [https://arxiv.org/abs/1901.10124](https://arxiv.org/abs/1901.10124)
3. "Dask: Parallel Computation with Blocked algorithms and Task Scheduling" by Rocklin (ArXiv:1505.08025): [https://arxiv.org/abs/1505.08025](https://arxiv.org/abs/1505.08025)
4. Dask Examples Repository: [https://github.com/dask/dask-examples](https://github.com/dask/dask-examples)
5. Dask Tutorial: [https://tutorial.dask.org/](https://tutorial.dask.org/)

These resources provide in-depth explanations, tutorials, and research papers to help you master data wrangling with Dask.

