## Default Join Type in pandas.merge()
Slide 1: Default Join Type in pandas.merge()

The default join type in pandas.merge() is an inner join, which returns only the matching records from both DataFrames being merged. This behavior retains only rows where the specified key columns have values present in both DataFrames.

```python
import pandas as pd

# Create sample DataFrames
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'value': ['A', 'B', 'C', 'D']
})

df2 = pd.DataFrame({
    'id': [1, 2, 5, 6],
    'score': [100, 200, 300, 400]
})

# Default merge (inner join)
result = pd.merge(df1, df2)
print("Default merge result:")
print(result)

# Output:
#    id value  score
# 0   1    A    100
# 1   2    B    200
```

Slide 2: Inner Join Behavior Analysis

Inner joins filter out non-matching records entirely, which can lead to data loss if not handled carefully. Understanding this default behavior is crucial for data integrity, especially when dealing with large datasets where missing matches might indicate data quality issues.

```python
import pandas as pd
import numpy as np

# Create DataFrames with missing values
df1 = pd.DataFrame({
    'key': ['A', 'B', 'C', np.nan, 'D'],
    'value1': range(5)
})

df2 = pd.DataFrame({
    'key': ['B', 'D', 'E', np.nan, 'F'],
    'value2': range(5, 10)
})

# Demonstrate default merge behavior
result = pd.merge(df1, df2)
print("Results with NaN values:")
print(result)

# Output shows only matching non-null keys
```

Slide 3: Comparison with Other Join Types

Understanding how the default inner join differs from other join types helps in making informed decisions about data merging strategies. Left, right, and outer joins provide different approaches to handling unmatched records compared to the default inner join.

```python
import pandas as pd

# Sample DataFrames
left = pd.DataFrame({'id': [1, 2, 3], 'value': ['a', 'b', 'c']})
right = pd.DataFrame({'id': [2, 3, 4], 'score': [20, 30, 40]})

# Compare different join types
inner_join = pd.merge(left, right, how='inner')
left_join = pd.merge(left, right, how='left')
right_join = pd.merge(left, right, how='right')
outer_join = pd.merge(left, right, how='outer')

print("Inner Join (Default):", inner_join.shape[0], "rows")
print("Left Join:", left_join.shape[0], "rows")
print("Right Join:", right_join.shape[0], "rows")
print("Outer Join:", outer_join.shape[0], "rows")
```

Slide 4: Real-world Example - Customer Orders Analysis

When analyzing customer purchase data, understanding merge behavior becomes crucial. This example demonstrates merging customer information with their order history, showing how the default inner join affects business analytics.

```python
import pandas as pd
import numpy as np

# Create customer data
customers = pd.DataFrame({
    'customer_id': range(1, 6),
    'name': ['John', 'Alice', 'Bob', 'Carol', 'David'],
    'signup_date': pd.date_range('2023-01-01', periods=5)
})

# Create orders data
orders = pd.DataFrame({
    'order_id': range(1, 8),
    'customer_id': [1, 2, 2, 3, 3, 3, 6],  # Note: customer_id 6 doesn't exist
    'order_date': pd.date_range('2023-02-01', periods=7),
    'amount': np.random.randint(100, 1000, 7)
})

# Merge using default inner join
customer_orders = pd.merge(customers, orders)
print("Customer Orders Analysis:")
print(customer_orders)
```

Slide 5: Performance Implications of Default Join

The default inner join typically provides better performance compared to other join types because it processes only matching records. This becomes significant when working with large datasets where memory usage and processing time are critical concerns.

```python
import pandas as pd
import time
import numpy as np

# Generate large sample datasets
n_rows = 100000
df1 = pd.DataFrame({
    'key': np.random.choice(range(n_rows//2), n_rows),
    'value1': np.random.randn(n_rows)
})

df2 = pd.DataFrame({
    'key': np.random.choice(range(n_rows//2), n_rows),
    'value2': np.random.randn(n_rows)
})

# Measure execution time for different join types
joins = ['inner', 'outer', 'left', 'right']
times = {}

for join_type in joins:
    start_time = time.time()
    _ = pd.merge(df1, df2, on='key', how=join_type)
    times[join_type] = time.time() - start_time

print("Execution times (seconds):")
for join_type, exec_time in times.items():
    print(f"{join_type}: {exec_time:.4f}")
```

Slide 6: Handling Multiple Key Columns

The default merge behavior becomes more complex when dealing with multiple key columns. Understanding how pandas handles multiple keys in the default inner join is essential for accurate data combination and analysis.

```python
import pandas as pd

# Create DataFrames with multiple key columns
df1 = pd.DataFrame({
    'dept_id': [1, 1, 2, 2],
    'year': [2022, 2023, 2022, 2023],
    'budget': [100, 150, 200, 250]
})

df2 = pd.DataFrame({
    'dept_id': [1, 1, 2, 3],
    'year': [2022, 2023, 2022, 2022],
    'expenses': [80, 120, 180, 90]
})

# Merge on multiple keys
result = pd.merge(df1, df2, on=['dept_id', 'year'])
print("Multiple key merge result:")
print(result)

# Calculate budget vs expenses
result['surplus'] = result['budget'] - result['expenses']
print("\nBudget analysis:")
print(result)
```

Slide 7: Handling Index-based Merging

When merging DataFrames using index-based joins, the default behavior differs slightly from column-based merging. Understanding these nuances is crucial for maintaining data integrity during index-based operations.

```python
import pandas as pd

# Create DataFrames with meaningful indices
df1 = pd.DataFrame({
    'value': ['A', 'B', 'C', 'D']
}, index=[1, 2, 3, 4])

df2 = pd.DataFrame({
    'score': [100, 200, 300, 400]
}, index=[1, 2, 5, 6])

# Demonstrate index-based merging
result_left_index = pd.merge(df1, df2, left_index=True, right_index=True)
print("Index-based merge result:")
print(result_left_index)

# Compare with reset index
result_reset = pd.merge(df1.reset_index(), df2.reset_index())
print("\nColumn-based merge after reset_index:")
print(result_reset)
```

Slide 8: Managing Duplicate Keys

The default merge behavior with duplicate keys can lead to cartesian products, potentially causing unexpected data expansion. Understanding and managing this behavior is crucial for maintaining data integrity.

```python
import pandas as pd

# Create DataFrames with duplicate keys
df1 = pd.DataFrame({
    'key': ['A', 'A', 'B'],
    'value': [1, 2, 3]
})

df2 = pd.DataFrame({
    'key': ['A', 'A', 'C'],
    'score': [10, 20, 30]
})

# Show cartesian product with duplicate keys
result = pd.merge(df1, df2)
print("Merge result with duplicate keys:")
print(result)

# Calculate result size
print("\nInput shapes:", df1.shape[0], "x", df2.shape[0])
print("Output shape:", result.shape[0])
```

Slide 9: Real-world Example - Sales Data Integration

This example demonstrates a practical scenario of merging sales transactions with product information, showing how the default inner join affects business reporting and analysis.

```python
import pandas as pd
import numpy as np

# Create sales transactions data
sales = pd.DataFrame({
    'transaction_id': range(1, 8),
    'product_id': ['P1', 'P2', 'P1', 'P3', 'P4', 'P2', 'P5'],
    'quantity': np.random.randint(1, 10, 7),
    'sale_date': pd.date_range('2024-01-01', periods=7)
})

# Create product catalog
products = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P3', 'P4'],
    'product_name': ['Widget A', 'Widget B', 'Widget C', 'Widget D'],
    'unit_price': [10.99, 15.99, 20.99, 25.99]
})

# Merge and calculate total sales
merged_sales = pd.merge(sales, products)
merged_sales['total_amount'] = merged_sales['quantity'] * merged_sales['unit_price']

print("Sales Analysis:")
print(merged_sales)
print("\nTotal Sales by Product:")
print(merged_sales.groupby('product_name')['total_amount'].sum())
```

Slide 10: Memory Efficiency in Default Joins

The default inner join's memory usage characteristics are important when working with large datasets. Understanding these patterns helps in optimizing merge operations for better performance.

```python
import pandas as pd
import numpy as np
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Create large DataFrames with different overlap ratios
size = 100000
df1 = pd.DataFrame({
    'key': np.random.randint(0, size//2, size),
    'value1': np.random.randn(size)
})

df2 = pd.DataFrame({
    'key': np.random.randint(0, size//2, size),
    'value2': np.random.randn(size)
})

# Measure memory before and after merge
initial_memory = get_memory_usage()
result = pd.merge(df1, df2)
final_memory = get_memory_usage()

print(f"Initial memory: {initial_memory:.2f} MB")
print(f"Final memory: {final_memory:.2f} MB")
print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
print(f"Result shape: {result.shape}")
```

Slide 11: Merge Indicator Feature

The merge indicator functionality helps track the source of rows in the merged DataFrame, providing valuable insights into the matching process even with the default inner join behavior.

```python
import pandas as pd

# Create sample DataFrames with partially overlapping data
df1 = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'value': ['A', 'B', 'C', 'D']
})

df2 = pd.DataFrame({
    'id': [1, 2, 5, 6],
    'score': [100, 200, 300, 400]
})

# Merge with indicator
result = pd.merge(df1, df2, indicator=True)
print("Merge result with indicator:")
print(result)

# Analyze merge results
merge_counts = result['_merge'].value_counts()
print("\nMerge statistics:")
print(merge_counts)
```

Slide 12: Handling Column Name Conflicts

When merging DataFrames with identical column names, understanding the default suffixing behavior is crucial for maintaining data clarity and preventing column name conflicts.

```python
import pandas as pd

# Create DataFrames with overlapping column names
df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [10, 20, 30],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03']
})

df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'value': [100, 200, 400],
    'date': ['2024-01-01', '2024-01-02', '2024-01-04']
})

# Default merge with overlapping columns
default_merge = pd.merge(df1, df2, on='id')
print("Default merge with suffixes:")
print(default_merge)

# Custom suffixes
custom_merge = pd.merge(df1, df2, on='id', suffixes=('_first', '_second'))
print("\nMerge with custom suffixes:")
print(custom_merge)
```

Slide 13: Performance Optimization Techniques

Understanding how to optimize merge operations with the default join type can significantly improve performance when working with large datasets or frequent merge operations.

```python
import pandas as pd
import numpy as np
from time import time

# Create large DataFrames with sorted and unsorted keys
n_rows = 100000
keys = np.random.randint(0, n_rows//2, n_rows)

# Sorted DataFrames
df1_sorted = pd.DataFrame({
    'key': np.sort(keys),
    'value1': np.random.randn(n_rows)
})

df2_sorted = pd.DataFrame({
    'key': np.sort(keys),
    'value2': np.random.randn(n_rows)
})

# Unsorted DataFrames
df1_unsorted = df1_sorted.sample(frac=1).reset_index(drop=True)
df2_unsorted = df2_sorted.sample(frac=1).reset_index(drop=True)

# Compare merge performance
def time_merge(df1, df2):
    start = time()
    _ = pd.merge(df1, df2, on='key')
    return time() - start

sorted_time = time_merge(df1_sorted, df2_sorted)
unsorted_time = time_merge(df1_unsorted, df2_unsorted)

print(f"Sorted merge time: {sorted_time:.4f} seconds")
print(f"Unsorted merge time: {unsorted_time:.4f} seconds")
print(f"Performance improvement: {((unsorted_time-sorted_time)/unsorted_time)*100:.2f}%")
```

Slide 14: Additional Resources

*   ArXiv paper on data integration techniques:
    *   [https://arxiv.org/abs/2107.00782](https://arxiv.org/abs/2107.00782)
*   Efficient data merging strategies research:
    *   [https://arxiv.org/abs/1908.08283](https://arxiv.org/abs/1908.08283)
*   Performance optimization for large-scale data merging:
    *   [https://dl.acm.org/doi/10.1145/3318464.3389742](https://dl.acm.org/doi/10.1145/3318464.3389742)
*   For more detailed information about pandas merging operations:
    *   [https://pandas.pydata.org/docs/user\_guide/merging.html](https://pandas.pydata.org/docs/user_guide/merging.html)
*   Advanced data manipulation techniques:
    *   [https://scipy-lectures.org/packages/statistics/index.html](https://scipy-lectures.org/packages/statistics/index.html)

