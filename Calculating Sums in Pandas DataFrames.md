## Calculating Sums in Pandas DataFrames
Slide 1: Basic DataFrame Sum Operations

The pandas sum() function is a versatile method for calculating column-wise or row-wise sums in a DataFrame. It can handle numeric data types while automatically excluding NaN values, making it robust for real-world data analysis scenarios.

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, np.nan],
    'B': [4, 5, 6, 7],
    'C': [8, 9, 10, 11]
})

# Column-wise sum
col_sums = df.sum()
print("Column sums:\n", col_sums)

# Row-wise sum
row_sums = df.sum(axis=1)
print("\nRow sums:\n", row_sums)
```

Slide 2: Advanced Sum Operations with Custom Parameters

The sum() function provides additional parameters for handling missing values, data types, and numeric precision. Understanding these parameters is crucial for accurate calculations in data analysis pipelines.

```python
# Create DataFrame with mixed types
df_mixed = pd.DataFrame({
    'A': [1, 2, 'NA', 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, 12]
})

# Sum with skipna and numeric_only parameters
sum_skipna = df_mixed.sum(skipna=True, numeric_only=True)
print("Sum with skipna:\n", sum_skipna)

# Sum with min_count parameter
sum_min_count = df_mixed.sum(min_count=3)
print("\nSum with min_count=3:\n", sum_min_count)
```

Slide 3: Groupwise Sum Operations

Combining groupby() with sum() enables powerful aggregation operations across categorical variables. This combination is essential for analyzing patterns and trends within subsets of your data.

```python
# Create DataFrame with categories
df_grouped = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C'],
    'value1': [10, 20, 30, 40, 50],
    'value2': [1, 2, 3, 4, 5]
})

# Group by category and sum
group_sums = df_grouped.groupby('category').sum()
print("Group sums:\n", group_sums)
```

Slide 4: Rolling Sum Implementation

Rolling sums provide insights into cumulative patterns and trends over specified windows of data. This technique is particularly useful in time series analysis and financial applications.

```python
# Create time series data
dates = pd.date_range('2024-01-01', periods=10)
ts_df = pd.DataFrame({
    'values': range(10)
}, index=dates)

# Calculate rolling sum with window size 3
rolling_sum = ts_df.rolling(window=3).sum()
print("Rolling sum:\n", rolling_sum)

# Calculate rolling sum with min_periods
rolling_sum_min = ts_df.rolling(window=3, min_periods=1).sum()
print("\nRolling sum with min_periods=1:\n", rolling_sum_min)
```

Slide 5: Cumulative Sum Operations

Cumulative sums track running totals across DataFrame elements, essential for analyzing progressive accumulation of values in time series or sequential data analysis.

```python
# Create sample DataFrame
df_cumsum = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

# Calculate cumulative sum
cumsum = df_cumsum.cumsum()
print("Cumulative sum:\n", cumsum)

# Calculate cumulative sum by axis
cumsum_axis = df_cumsum.cumsum(axis=1)
print("\nCumulative sum by row:\n", cumsum_axis)
```

Slide 6: Real-world Example - Sales Analysis

A practical implementation of sum operations analyzing daily sales data across different product categories and regions, demonstrating data preprocessing and aggregation techniques.

```python
# Create sales dataset
sales_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'product': np.random.choice(['A', 'B', 'C'], 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'sales': np.random.normal(1000, 100, 100),
    'units': np.random.randint(10, 100, 100)
})

# Calculate total sales by product and region
total_sales = sales_data.groupby(['product', 'region'])['sales'].sum()
print("Total sales by product and region:\n", total_sales)

# Calculate daily running total
daily_running_total = sales_data.groupby('date')['sales'].sum().cumsum()
print("\nDaily running total:\n", daily_running_total)
```

Slide 7: Sum Operations with Time-based Grouping

Analyzing time-series data requires specialized grouping operations combined with sum calculations. This implementation showcases resampling and period-based aggregations.

```python
# Create time series sales data
ts_sales = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='H'),
    'sales': np.random.normal(1000, 100, 100)
}).set_index('date')

# Calculate daily sums
daily_sums = ts_sales.resample('D').sum()
print("Daily sums:\n", daily_sums.head())

# Calculate weekly sums with custom week start
weekly_sums = ts_sales.resample('W-MON').sum()
print("\nWeekly sums:\n", weekly_sums.head())
```

Slide 8: Custom Sum Aggregations

Implementing custom sum operations with complex conditions and multiple aggregation levels provides deeper insights into data patterns and relationships.

```python
import pandas as pd
import numpy as np

# Create multi-level DataFrame
multi_df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B'] * 3,
    'subgroup': ['X', 'Y'] * 6,
    'value': np.random.randn(12) * 100
})

# Custom sum aggregation
custom_sum = multi_df.groupby(['group', 'subgroup']).agg({
    'value': [
        ('sum', 'sum'),
        ('positive_sum', lambda x: x[x > 0].sum()),
        ('negative_sum', lambda x: x[x < 0].sum())
    ]
})

print("Custom aggregation results:\n", custom_sum)
```

Slide 9: Real-world Example - Financial Analysis

Implementing sum operations for financial data analysis, including calculation of portfolio returns and risk metrics using rolling and cumulative sums.

```python
# Create financial dataset
np.random.seed(42)
financial_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=252),  # Trading days in a year
    'returns': np.random.normal(0.001, 0.02, 252),
    'volume': np.random.lognormal(10, 1, 252)
})

# Calculate cumulative returns
financial_data['cum_returns'] = (1 + financial_data['returns']).cumprod()

# Calculate rolling volatility (20-day window)
financial_data['volatility'] = financial_data['returns'].rolling(20).std() * np.sqrt(252)

print("Financial metrics:\n", financial_data.tail())
```

Slide 10: Handling Missing Values in Sum Operations

Understanding the impact of missing values on sum calculations and implementing appropriate strategies for handling them in different analytical scenarios.

```python
# Create DataFrame with missing values
missing_df = pd.DataFrame({
    'A': [1, np.nan, 3, np.nan, 5],
    'B': [np.nan, 2, np.nan, 4, 5],
    'C': [1, 2, 3, 4, np.nan]
})

# Different approaches to handling missing values
print("Default sum:\n", missing_df.sum())
print("\nSum without NA:\n", missing_df.sum(skipna=False))
print("\nSum with min_count:\n", missing_df.sum(min_count=3))

# Fill missing values with different strategies
print("\nSum after forward fill:\n", missing_df.ffill().sum())
print("\nSum after backward fill:\n", missing_df.bfill().sum())
```

Slide 11: Memory-Efficient Sum Operations

Implementing efficient sum operations for large datasets using chunking and optimization techniques to manage memory usage while maintaining accuracy.

```python
# Function for memory-efficient sum calculation
def efficient_sum(df, chunk_size=1000):
    total_sum = 0
    for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
        total_sum += chunk['value'].sum()
    return total_sum

# Example with generated large dataset
large_df = pd.DataFrame({
    'value': np.random.randn(10000)
})

# Compare memory usage
def compare_sum_methods(df):
    import memory_profiler
    
    @memory_profiler.profile
    def direct_sum():
        return df['value'].sum()
    
    @memory_profiler.profile
    def chunked_sum():
        return sum(chunk['value'].sum() 
                  for chunk in np.array_split(df['value'], 10))
    
    print("Direct sum:", direct_sum())
    print("Chunked sum:", chunked_sum())

compare_sum_methods(large_df)
```

Slide 12: Sum Operations with Complex Data Types

Implementing sum operations for complex data types including strings, timestamps, and custom objects while handling type conversion and aggregation rules.

```python
# Create DataFrame with complex types
complex_df = pd.DataFrame({
    'text_length': ['abc', 'defgh', 'ijklmno'],
    'timestamp': pd.date_range('2024-01-01', periods=3),
    'custom': [
        {'value': 1},
        {'value': 2},
        {'value': 3}
    ]
})

# Custom sum functions
def text_sum(series):
    return sum(len(x) for x in series)

def custom_sum(series):
    return sum(x['value'] for x in series)

# Apply custom sum operations
results = {
    'text_length_sum': text_sum(complex_df['text_length']),
    'custom_sum': custom_sum(complex_df['custom'])
}

print("Complex type sums:\n", results)
```

Slide 13: Additional Resources

*   "Efficient Statistical Computing in Pandas: A Comprehensive Review" [https://arxiv.org/abs/2305.12872](https://arxiv.org/abs/2305.12872)
*   "Optimizing DataFrame Operations: Memory and Performance Analysis" [https://arxiv.org/abs/2301.09324](https://arxiv.org/abs/2301.09324)
*   "Large-Scale Data Analysis with Pandas: Challenges and Solutions" [https://arxiv.org/abs/2204.15136](https://arxiv.org/abs/2204.15136)
*   "Statistical Computing with Missing Data: A Comparative Study" [https://arxiv.org/abs/2203.14281](https://arxiv.org/abs/2203.14281)

