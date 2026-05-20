## Default Aggregation Methods in Pandas GroupBy
Slide 1: Default Aggregation in Pandas GroupBy

The default aggregation method in pandas groupby() operations is determined by the data type of each column. For numeric data, it defaults to mean(), while for object/string columns, it uses first(). This behavior can be observed through practical examples.

```python
import pandas as df
import numpy as np

# Create sample dataset
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C'],
    'values': [1, 2, 3, 4, 5],
    'text': ['foo', 'bar', 'baz', 'qux', 'quux']
})

# Default groupby behavior
grouped = df.groupby('category').agg()
print("Default aggregation results:")
print(grouped)
```

Slide 2: Column-Specific Default Aggregations

Understanding how different data types are aggregated is crucial for data analysis. Numeric columns (int, float) use mean(), datetime columns use first(), and object/string columns use first(). This behavior affects how your data is summarized.

```python
import pandas as pd
import numpy as np

# Create mixed-type dataset
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'numeric': [1, 2, 3, 4],
    'string': ['x', 'y', 'z', 'w'],
    'date': pd.date_range('2024-01-01', periods=4)
})

# Show default aggregation for each type
result = df.groupby('group').agg()
print("Default aggregations by type:")
print(result)
```

Slide 3: Numeric Data Aggregation Analysis

When grouping numeric data, the mean() function calculates the arithmetic average for each group. This behavior is particularly important in financial and statistical analysis where averages are commonly used as summary statistics.

```python
import pandas as pd
import numpy as np

# Financial data example
financial_data = pd.DataFrame({
    'department': ['Sales', 'Sales', 'IT', 'IT', 'HR'],
    'revenue': [100000, 150000, 200000, 250000, 80000],
    'expenses': [50000, 75000, 100000, 125000, 40000]
})

# Default groupby on numeric columns
dept_summary = financial_data.groupby('department').agg()
print("Department Financial Summary:")
print(dept_summary)
```

Slide 4: String and Object Data Handling

String and object data types are aggregated using first() by default, which retains the first occurrence within each group. This behavior is essential to understand when dealing with categorical or text data in grouped operations.

```python
import pandas as pd

# Customer data example
customer_data = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'North'],
    'product': ['Laptop', 'Phone', 'Tablet', 'Phone', 'Laptop'],
    'customer_type': ['Premium', 'Regular', 'Premium', 'Premium', 'Regular']
})

# Default groupby on string columns
region_summary = customer_data.groupby('region').agg()
print("Region Summary (First Values):")
print(region_summary)
```

Slide 5: Handling Missing Values in Default Aggregation

The default aggregation method handles missing values (NaN) differently based on data types. For numeric data, NaN values are excluded from mean calculations, while for object types, NaN values are preserved if they are the first occurrence.

```python
import pandas as pd
import numpy as np

# Dataset with missing values
df_missing = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'C'],
    'values': [1, np.nan, 3, np.nan, 5],
    'text': ['foo', None, 'baz', np.nan, 'quux']
})

# Default aggregation with missing values
result_missing = df_missing.groupby('group').agg()
print("Aggregation with Missing Values:")
print(result_missing)
```

Slide 6: Multiple Column Grouping Default Behavior

When grouping by multiple columns, the default aggregation applies independently to each remaining column. This hierarchical grouping is particularly useful for complex data analysis where multiple categories need to be considered simultaneously.

```python
import pandas as pd

# Multi-level grouping dataset
sales_data = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'East'],
    'quarter': ['Q1', 'Q1', 'Q2', 'Q2', 'Q1'],
    'sales': [1000, 2000, 1500, 2500, 3000],
    'units': [100, 200, 150, 250, 300]
})

# Multiple column groupby
multi_group = sales_data.groupby(['region', 'quarter']).agg()
print("Multi-level Grouping Results:")
print(multi_group)
```

Slide 7: Time Series Default Aggregation

When working with time series data, the default aggregation becomes particularly important for resampling and time-based grouping operations. DateTime columns use first() by default, while associated numeric values use mean().

```python
import pandas as pd
import numpy as np

# Time series dataset
dates = pd.date_range('2024-01-01', periods=6)
time_series = pd.DataFrame({
    'date': dates,
    'value': [10, 15, 12, 18, 20, 22],
    'category': ['A', 'A', 'B', 'B', 'A', 'B']
})

# Time-based grouping
time_grouped = time_series.groupby(pd.Grouper(key='date', freq='2D')).agg()
print("Time Series Aggregation:")
print(time_grouped)
```

Slide 8: Impact of Data Types on Default Aggregation

The data type of columns significantly influences the default aggregation behavior. This example demonstrates how changing data types can alter the groupby results without explicitly specifying an aggregation method.

```python
import pandas as pd

# Create dataset with different data types
mixed_df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B'],
    'int_val': [1, 2, 3, 4],
    'float_val': [1.0, 2.0, 3.0, 4.0],
    'str_as_num': ['1', '2', '3', '4']
})

# Convert string to numeric
mixed_df['str_as_num'] = pd.to_numeric(mixed_df['str_as_num'])

# Compare aggregations
result = mixed_df.groupby('group').agg()
print("Type-based Aggregation Results:")
print(result)
```

Slide 9: Real-world Example: Sales Analysis

This comprehensive example demonstrates default aggregation in a real-world sales analysis scenario, showing how different data types are handled in a complex business dataset.

```python
import pandas as pd
import numpy as np

# Create realistic sales dataset
sales_analysis = pd.DataFrame({
    'store_id': np.repeat(['S001', 'S002', 'S003'], 4),
    'date': pd.date_range('2024-01-01', periods=12),
    'daily_sales': np.random.normal(1000, 100, 12),
    'customer_count': np.random.randint(50, 150, 12),
    'promotion_active': np.random.choice(['Yes', 'No'], 12)
})

# Complex grouping analysis
store_performance = sales_analysis.groupby(['store_id', 
    pd.Grouper(key='date', freq='W')]).agg()

print("Store Performance Analysis:")
print(store_performance)
```

Slide 10: Performance Metrics for Sales Analysis

When analyzing the results of our previous sales analysis, we can observe how default aggregation affects different metrics and their interpretation in a business context.

```python
# Continue from previous example
store_stats = store_performance.describe()
print("\nStatistical Summary of Store Performance:")
print(store_stats)

# Calculate additional metrics
store_variance = store_performance.var()
print("\nVariance Analysis:")
print(store_variance)
```

Slide 11: Working with Custom Objects and Default Aggregation

When dealing with custom objects or complex data structures, understanding how pandas handles default aggregation becomes crucial. This example demonstrates the behavior with custom classes and specialized data types.

```python
import pandas as pd
import numpy as np

class Revenue:
    def __init__(self, amount):
        self.amount = amount
    def __repr__(self):
        return f"Revenue({self.amount})"

# Create dataset with custom objects
custom_df = pd.DataFrame({
    'department': ['Sales', 'Sales', 'IT', 'IT'],
    'revenue': [Revenue(1000), Revenue(2000), Revenue(3000), Revenue(4000)],
    'count': [1, 2, 3, 4]
})

# Default groupby behavior with custom objects
grouped_custom = custom_df.groupby('department').agg()
print("Custom Object Aggregation:")
print(grouped_custom)
```

Slide 12: Advanced Time-Based Aggregation Patterns

Default aggregation becomes more complex when dealing with multiple time-based hierarchies and mixed data types. This example shows how pandas handles such scenarios in financial data analysis.

```python
import pandas as pd
import numpy as np

# Create financial timeseries data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
financial_ts = pd.DataFrame({
    'date': dates,
    'instrument': np.random.choice(['Bond', 'Stock', 'Option'], 100),
    'price': np.random.normal(100, 10, 100),
    'volume': np.random.randint(1000, 5000, 100),
    'trader': np.random.choice(['A', 'B', 'C'], 100)
})

# Multi-level time-based grouping
time_hierarchy = financial_ts.groupby([
    pd.Grouper(key='date', freq='M'),
    'instrument'
]).agg()

print("Financial Time Series Aggregation:")
print(time_hierarchy)
```

Slide 13: Real-world Application: Environmental Data Analysis

This comprehensive example shows how default aggregation handles environmental monitoring data with mixed frequencies and multiple sensor types.

```python
import pandas as pd
import numpy as np

# Create environmental monitoring dataset
n_samples = 1000
env_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
    'location': np.random.choice(['Site1', 'Site2', 'Site3'], n_samples),
    'temperature': np.random.normal(20, 5, n_samples),
    'humidity': np.random.normal(60, 10, n_samples),
    'sensor_status': np.random.choice(['active', 'maintenance'], n_samples)
})

# Complex environmental analysis
site_metrics = env_data.groupby([
    'location',
    pd.Grouper(key='timestamp', freq='D')
]).agg()

print("Environmental Monitoring Results:")
print(site_metrics.head())

# Calculate daily statistics
daily_stats = site_metrics.describe()
print("\nDaily Environmental Statistics:")
print(daily_stats)
```

Slide 14: Additional Resources

*   Understanding Pandas GroupBy: A Deep Dive into Data Aggregation
    *   [https://pandas.pydata.org/docs/user\_guide/groupby.html](https://pandas.pydata.org/docs/user_guide/groupby.html)
*   Efficient Data Aggregation Methods in Time Series Analysis
    *   [https://arxiv.org/abs/2103.05071](https://arxiv.org/abs/2103.05071)
*   Advanced Techniques in DataFrame Operations and Aggregations
    *   [https://towardsdatascience.com/advanced-pandas-groupby-techniques](https://towardsdatascience.com/advanced-pandas-groupby-techniques)
*   Statistical Computing in Python with Pandas
    *   [https://scipy.org/](https://scipy.org/)
*   Performance Optimization in Pandas Aggregations
    *   [https://engineering.google.com/blog/pandas-optimization](https://engineering.google.com/blog/pandas-optimization)

