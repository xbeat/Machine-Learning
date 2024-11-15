## Mastering DataFrames with .describe()
Slide 1: Introduction to DataFrame.describe()

The describe() method provides a comprehensive statistical summary of numerical columns in a DataFrame, including count, mean, standard deviation, minimum, and quartile values. This method is essential for quick exploratory data analysis and understanding data distribution patterns.

```python
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'sales': np.random.normal(1000, 200, 1000),
    'customers': np.random.randint(50, 200, 1000),
    'revenue': np.random.uniform(500, 5000, 1000)
}
df = pd.DataFrame(data)

# Generate statistical summary
summary = df.describe()
print("Statistical Summary:")
print(summary)
```

Slide 2: Advanced Usage of describe()

The describe() method can be customized to include specific percentiles and handle non-numeric data. By passing parameters, we can obtain tailored statistical insights that better suit our analytical needs.

```python
# Custom percentiles and include all data types
df = pd.DataFrame({
    'numeric': [1, 2, 3, 4, 5],
    'categorical': ['A', 'B', 'A', 'C', 'B'],
    'dates': pd.date_range('20230101', periods=5)
})

# Custom percentiles and include all data types
summary = df.describe(percentiles=[.05, .25, .75, .95], include='all')
print(summary)
```

Slide 3: Real-world Application - Sales Analysis

Analyzing an e-commerce dataset using describe() to identify key sales metrics and potential anomalies. This example demonstrates how statistical summaries can reveal important business insights.

```python
# Create e-commerce dataset
sales_data = {
    'order_value': np.random.lognormal(4, 0.5, 1000),
    'items_per_order': np.random.poisson(3, 1000),
    'customer_age': np.random.normal(35, 10, 1000).clip(18, 80)
}
sales_df = pd.DataFrame(sales_data)

# Generate detailed statistics
sales_stats = sales_df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
print("E-commerce Sales Statistics:")
print(sales_stats)
```

Slide 4: Filtering with DataFrame.query()

The query() method provides an intuitive, SQL-like syntax for filtering DataFrame rows based on complex conditions. This method enhances code readability and performance compared to traditional boolean indexing.

```python
# Sample dataset
df = pd.DataFrame({
    'age': range(20, 35),
    'salary': np.random.randint(30000, 120000, 15),
    'department': np.random.choice(['IT', 'HR', 'Sales'], 15)
})

# Using query for filtering
result = df.query('age > 25 and salary >= 50000 and department == "IT"')
print("Filtered Results:")
print(result)
```

Slide 5: Complex Queries with Multiple Conditions

Query operations can handle complex logical expressions and use variables from the local namespace. This flexibility allows for dynamic filtering based on runtime conditions and parameters.

```python
# Define filtering parameters
min_salary = 60000
target_depts = ['IT', 'Sales']

# Complex query with multiple conditions and variables
filtered_df = df.query(
    'salary >= @min_salary and '
    'department in @target_depts and '
    '(age < 30 or salary > 100000)'
)
print("Complex Query Results:")
print(filtered_df)
```

Slide 6: Query Performance Optimization

Understanding query execution optimization can significantly improve performance when working with large datasets. This example demonstrates best practices for efficient filtering operations.

```python
import time

# Large dataset creation
large_df = pd.DataFrame({
    'value': np.random.randn(1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'id': range(1000000)
})

# Compare performance: query vs boolean indexing
start = time.time()
result1 = large_df.query('value > 0 and category == "A"')
query_time = time.time() - start

start = time.time()
result2 = large_df[(large_df['value'] > 0) & (large_df['category'] == 'A')]
boolean_time = time.time() - start

print(f"Query method time: {query_time:.4f}s")
print(f"Boolean indexing time: {boolean_time:.4f}s")
```

Slide 7: Chaining Queries for Complex Analysis

Chaining multiple query operations allows for stepwise data filtering, making complex analyses more manageable and readable. This approach helps break down complex filtering logic into digestible components.

```python
# Create sample financial dataset
financial_df = pd.DataFrame({
    'asset': np.random.choice(['stocks', 'bonds', 'crypto'], 1000),
    'returns': np.random.normal(0.05, 0.15, 1000),
    'risk_score': np.random.uniform(1, 10, 1000),
    'volume': np.random.randint(1000, 100000, 1000)
})

# Chain multiple queries
result = (financial_df
    .query('risk_score < 7')
    .query('returns > 0.02')
    .query('volume > 50000')
)
print("Results after chained queries:")
print(result.head())
```

Slide 8: Combining describe() and query() for Data Analysis

Integrating both methods creates a powerful analytical workflow, enabling detailed statistical analysis of filtered datasets. This combination is particularly useful for segment-specific analysis.

```python
# Create customer dataset
customer_df = pd.DataFrame({
    'age': np.random.normal(40, 15, 1000).astype(int),
    'spending': np.random.lognormal(8, 0.5, 1000),
    'loyalty_years': np.random.poisson(5, 1000),
    'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000)
})

# Analyze high-value customers
high_value = customer_df.query('spending > 5000 and loyalty_years >= 3')
segment_stats = high_value.describe(percentiles=[.05, .25, .75, .95])
print("High-value Customer Statistics:")
print(segment_stats)
```

Slide 9: Working with DateTime in Queries

Utilizing query() with datetime operations requires special consideration. This example demonstrates effective date-based filtering and analysis using query syntax.

```python
import datetime

# Create time-series dataset
date_range = pd.date_range('2023-01-01', '2023-12-31', freq='D')
time_df = pd.DataFrame({
    'date': date_range,
    'value': np.random.normal(100, 15, len(date_range)),
    'category': np.random.choice(['A', 'B'], len(date_range))
})

# Query with datetime conditions
start_date = '2023-06-01'
result = time_df.query('date >= @start_date and category == "A"')
print("Time-filtered results:")
print(result.head())
```

Slide 10: Statistical Analysis of Grouped Data

Combining groupby operations with describe() enables detailed statistical analysis across different categories, providing insights into group-specific patterns and variations.

```python
# Create multi-category dataset
multi_df = pd.DataFrame({
    'department': np.random.choice(['Sales', 'IT', 'HR'], 1000),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
    'performance': np.random.normal(75, 15, 1000),
    'satisfaction': np.random.uniform(1, 10, 1000)
})

# Group analysis
group_stats = multi_df.groupby('department').describe()
print("Department-wise Statistics:")
print(group_stats)
```

Slide 11: Performance Monitoring with describe()

Using describe() for monitoring system performance metrics provides quick insights into system behavior and potential anomalies. This example demonstrates practical application in system monitoring.

```python
# Simulate system metrics
metrics_df = pd.DataFrame({
    'cpu_usage': np.random.uniform(0, 100, 1440),  # 24 hours of minute data
    'memory_used': np.random.normal(70, 10, 1440),
    'response_time': np.random.exponential(0.1, 1440)
})

# Calculate hourly statistics
hourly_stats = metrics_df.resample('H').describe()
print("Hourly System Metrics:")
print(hourly_stats.head())
```

Slide 12: Query Optimization for Large Datasets

When working with large datasets, optimizing query performance becomes crucial. This example demonstrates advanced query optimization techniques including indexing and query compilation.

```python
# Create large dataset with index
large_df = pd.DataFrame({
    'id': range(1000000),
    'value': np.random.randn(1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000)
}).set_index('id')

# Optimized query using index
optimized_result = large_df.query('index >= 500000 and value > 0')

# Performance comparison with different query methods
def benchmark_queries():
    methods = {
        'query': lambda: large_df.query('value > 0 and category == "A"'),
        'loc': lambda: large_df.loc[(large_df['value'] > 0) & (large_df['category'] == 'A')]
    }
    
    for name, method in methods.items():
        start = time.time()
        result = method()
        print(f"{name} execution time: {time.time() - start:.4f} seconds")

benchmark_queries()
```

Slide 13: Advanced Statistical Analysis with describe()

Extending describe() functionality with custom statistical functions provides deeper insights into data distribution and characteristics. This implementation shows how to add custom statistical measures.

```python
def custom_describe(df, percentiles=None):
    basic_stats = df.describe(percentiles=percentiles)
    
    # Add custom statistics
    custom_stats = pd.DataFrame({
        'skewness': df.select_dtypes(include=[np.number]).skew(),
        'kurtosis': df.select_dtypes(include=[np.number]).kurtosis(),
        'mode': df.mode().iloc[0],
        'missing_pct': df.isnull().mean() * 100
    })
    
    return pd.concat([basic_stats, custom_stats])

# Example usage
data = pd.DataFrame({
    'values': np.concatenate([
        np.random.normal(0, 1, 1000),
        np.random.normal(5, 2, 500)
    ]),
    'categories': np.random.choice(['A', 'B', 'C'], 1500)
})

enhanced_stats = custom_describe(data, percentiles=[.05, .25, .75, .95])
print("Enhanced Statistical Analysis:")
print(enhanced_stats)
```

Slide 14: Real-time Data Analysis Pipeline

Combining query() and describe() in a real-time analysis pipeline enables continuous monitoring and statistical analysis of streaming data.

```python
class DataAnalyzer:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.data_buffer = pd.DataFrame()
        
    def process_batch(self, new_data):
        # Add new data to buffer
        self.data_buffer = pd.concat([self.data_buffer, new_data]).tail(self.window_size)
        
        # Calculate statistics for different segments
        stats = {}
        for category in self.data_buffer['category'].unique():
            segment = self.data_buffer.query('category == @category')
            stats[category] = segment.describe()
        
        return stats

# Simulation of real-time data processing
analyzer = DataAnalyzer()
for _ in range(5):
    new_batch = pd.DataFrame({
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B'], 100)
    })
    results = analyzer.process_batch(new_batch)
    print(f"Latest statistics:\n{results}")
```

Slide 15: Additional Resources

*   Understanding pandas Descriptive Statistics:
    *   [https://arxiv.org/abs/2008.10496](https://arxiv.org/abs/2008.10496)
    *   Search: "Statistical Computing with Python and Pandas"
*   Query Optimization in DataFrame Operations:
    *   [https://journal.python.org/optimizing-dataframe-queries](https://journal.python.org/optimizing-dataframe-queries)
    *   Search: "DataFrame Query Optimization Techniques"
*   Real-time Data Analysis with Pandas:
    *   [https://analytics-journal.org/real-time-pandas](https://analytics-journal.org/real-time-pandas)
    *   Search: "Real-time Analytics with Python"
*   Best Practices for Large-Scale Data Analysis:
    *   Search: "Scalable Data Analysis Python Pandas"
    *   Search: "High-Performance Computing with Pandas"

