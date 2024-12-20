## Efficient DataFrame Filtering with Pandas Query
Slide 1: Introduction to DataFrame Query Method

The query method in pandas provides a powerful and efficient way to filter DataFrames using string expressions. Unlike traditional boolean indexing, query leverages computation optimization and offers a more readable syntax for complex filtering operations.

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'A': np.random.randint(1, 100, 1000),
    'B': np.random.choice(['X', 'Y', 'Z'], 1000),
    'C': np.random.uniform(0, 1, 1000)
})

# Traditional boolean indexing
result1 = df[(df['A'] > 50) & (df['B'] == 'X')]

# Using query method
result2 = df.query('A > 50 and B == "X"')
```

Slide 2: Query Syntax and String Expressions

Query accepts string expressions that can reference column names directly without DataFrame prefix. The method supports complex logical operations, comparison operators, and even inline variable references using '@' symbol.

```python
# Sample DataFrame
df = pd.DataFrame({
    'age': range(20, 30),
    'salary': range(30000, 40000, 1000),
    'department': ['IT', 'HR', 'Sales', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT']
})

# Multiple conditions
filtered_df = df.query('age >= 25 and salary < 35000')

# Using in/not in operations
filtered_df2 = df.query('department in ["IT", "Sales"]')

# Variable reference
min_age = 22
filtered_df3 = df.query('age > @min_age')
```

Slide 3: Performance Optimization in Query

Query method internally compiles the string expression into bytecode, making it significantly faster than traditional filtering for large datasets. It also reduces memory usage by avoiding intermediate boolean mask creation.

```python
import time
import pandas as pd
import numpy as np

# Create large DataFrame
large_df = pd.DataFrame({
    'value': np.random.randn(1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'id': range(1000000)
})

# Measure traditional filtering
start = time.time()
result1 = large_df[(large_df['value'] > 0) & (large_df['category'] == 'A')]
traditional_time = time.time() - start

# Measure query method
start = time.time()
result2 = large_df.query('value > 0 and category == "A"')
query_time = time.time() - start

print(f"Traditional: {traditional_time:.4f}s")
print(f"Query: {query_time:.4f}s")
```

Slide 4: Complex Filtering with Mathematical Operations

The query method supports complex mathematical operations and functions within the string expression. This enables sophisticated filtering conditions without nested boolean operations.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x': np.random.uniform(-10, 10, 1000),
    'y': np.random.uniform(-10, 10, 1000),
    'z': np.random.uniform(-10, 10, 1000)
})

# Complex mathematical filtering
result = df.query('(x**2 + y**2) <= 100 and abs(z) < 5')

# Combined with string operations
df['category'] = np.random.choice(['type_A', 'type_B', 'type_C'], 1000)
filtered = df.query('category.str.contains("type_") and z >= 0')
```

Slide 5: Real-world Example - Financial Data Analysis

In this practical example, we'll analyze financial transaction data using query method for efficient filtering and analysis of large-scale financial records.

```python
import pandas as pd
import numpy as np

# Create sample financial dataset
np.random.seed(42)
n_records = 100000

transactions_df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n_records, freq='5min'),
    'amount': np.random.normal(1000, 500, n_records),
    'transaction_type': np.random.choice(['purchase', 'refund', 'transfer'], n_records),
    'account_type': np.random.choice(['savings', 'checking', 'investment'], n_records),
    'risk_score': np.random.uniform(0, 100, n_records)
})

# Complex financial analysis using query
high_risk_transfers = transactions_df.query(
    'risk_score > 80 and '
    'transaction_type == "transfer" and '
    'amount > 1500 and '
    'account_type != "investment"'
)

print(f"Suspicious transactions found: {len(high_risk_transfers)}")
```

Slide 6: Working with DateTime in Query

The query method seamlessly integrates with pandas DateTime functionality, allowing complex temporal filtering operations. This is particularly useful when analyzing time-series data or event-based datasets.

```python
import pandas as pd
import numpy as np

# Create time-series dataset
dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.normal(100, 15, len(dates)),
    'event_type': np.random.choice(['A', 'B', 'C'], len(dates))
})

# Convert timestamp to datetime index
df.set_index('timestamp', inplace=True)

# Query with datetime operations
morning_events = df.query('index.hour >= 9 and index.hour <= 17')
summer_data = df.query('index.month >= 6 and index.month <= 8')

# Complex datetime filtering
busy_periods = df.query(
    'index.hour.between(9, 17) and '
    'index.dayofweek < 5 and '  # Monday = 0, Friday = 4
    'value > 110'
)
```

Slide 7: Query with Regular Expressions

Query method supports string operations and regular expressions through the str accessor, enabling powerful text-based filtering capabilities while maintaining performance benefits.

```python
import pandas as pd

# Create dataset with text data
df = pd.DataFrame({
    'email': ['john.doe@company.com', 'jane@gmail.com', 
              'support@company.com', 'info@website.org'],
    'message': ['Hello World', 'Query test', 
                'Important notice', 'Regular expression'],
    'priority': [1, 2, 1, 3]
})

# String pattern matching
company_emails = df.query('email.str.contains("company.com")')

# Combined regex and numerical filtering
important_company = df.query(
    'email.str.contains("company.com") and '
    'priority < 2 and '
    'message.str.contains("Important|Urgent", case=False)'
)

print("Filtered results:")
print(important_company)
```

Slide 8: Nested Query Operations

Complex data analysis often requires multiple filtering steps. Query method can be chained efficiently, allowing for sequential filtering operations while maintaining code readability.

```python
import pandas as pd
import numpy as np

# Create hierarchical dataset
df = pd.DataFrame({
    'department': np.random.choice(['Sales', 'IT', 'HR'], 1000),
    'team': np.random.choice(['Alpha', 'Beta', 'Gamma'], 1000),
    'performance': np.random.uniform(0, 100, 1000),
    'years_exp': np.random.randint(1, 15, 1000),
    'salary': np.random.uniform(40000, 100000, 1000)
})

# Multiple sequential queries
result = (df.query('department == "Sales"')
           .query('performance > 80')
           .query('years_exp >= 5')
           .query('salary < 80000'))

# Alternative single complex query
result_alternative = df.query(
    'department == "Sales" and '
    'performance > 80 and '
    'years_exp >= 5 and '
    'salary < 80000'
)

print(f"Found {len(result)} matching records")
```

Slide 9: Query Performance Optimization Techniques

Understanding query optimization techniques is crucial for handling large datasets efficiently. This example demonstrates various approaches to optimize query performance.

```python
import pandas as pd
import numpy as np
import time

# Create large dataset
large_df = pd.DataFrame({
    'id': range(1000000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000000),
    'value': np.random.uniform(0, 1000, 1000000),
    'text': np.random.choice(['abc', 'def', 'ghi', 'jkl'], 1000000)
})

# Optimization technique 1: Index-based filtering
large_df.set_index('id', inplace=True)

# Optimization technique 2: Pre-computed conditions
threshold = 500
categories = ['A', 'B']

def measure_query_time(query_func):
    start = time.time()
    result = query_func()
    return time.time() - start, len(result)

# Standard query
t1, n1 = measure_query_time(
    lambda: large_df.query('value > @threshold and category in @categories')
)

print(f"Query execution time: {t1:.4f}s, Records found: {n1}")
```

Slide 10: Query with Grouping Operations

The query method can be effectively combined with groupby operations for sophisticated data analysis. This approach allows for filtered aggregations while maintaining computational efficiency.

```python
import pandas as pd
import numpy as np

# Create sales dataset
df = pd.DataFrame({
    'product_id': np.random.randint(1000, 2000, 10000),
    'store_id': np.random.randint(1, 50, 10000),
    'sales': np.random.uniform(10, 1000, 10000),
    'date': pd.date_range('2023-01-01', periods=10000, freq='H'),
    'promotion': np.random.choice([True, False], 10000)
})

# Complex query with grouping
result = (df.query('sales > 500 and promotion == True')
          .groupby('store_id')
          .agg({
              'sales': ['count', 'mean', 'sum'],
              'product_id': 'nunique'
          }))

# Time-based grouped analysis
monthly_analysis = (df.query('sales > @df.sales.mean()')
                   .set_index('date')
                   .groupby(pd.Grouper(freq='M'))
                   .agg({'sales': 'sum', 'promotion': 'sum'}))

print("High-value sales analysis:")
print(result.head())
```

Slide 11: Real-world Example - Customer Segmentation

Implementing customer segmentation using query method for efficient filtering and analysis of customer behavior patterns in a large-scale e-commerce dataset.

```python
import pandas as pd
import numpy as np

# Generate customer dataset
n_customers = 100000
customer_data = pd.DataFrame({
    'customer_id': range(n_customers),
    'total_purchases': np.random.normal(500, 200, n_customers),
    'avg_order_value': np.random.normal(100, 30, n_customers),
    'days_since_last_purchase': np.random.randint(1, 365, n_customers),
    'loyalty_score': np.random.uniform(0, 100, n_customers),
    'age': np.random.normal(35, 12, n_customers).astype(int)
})

# Define segment criteria
vip_customers = customer_data.query(
    'total_purchases > @customer_data.total_purchases.quantile(0.9) and '
    'loyalty_score > 80 and '
    'days_since_last_purchase < 30'
)

# At-risk customers
at_risk = customer_data.query(
    'loyalty_score < 40 and '
    'days_since_last_purchase > 60 and '
    'total_purchases > @customer_data.total_purchases.mean()'
)

print(f"VIP Customers: {len(vip_customers)}")
print(f"At-risk Customers: {len(at_risk)}")
```

Slide 12: Advanced Query Optimization Patterns

Understanding advanced query patterns helps in optimizing complex filtering operations while maintaining code readability and performance.

```python
import pandas as pd
import numpy as np

# Create complex dataset
df = pd.DataFrame({
    'metric_a': np.random.normal(100, 15, 100000),
    'metric_b': np.random.normal(50, 10, 100000),
    'category': np.random.choice(['X', 'Y', 'Z'], 100000),
    'subcategory': np.random.choice(['A1', 'A2', 'B1', 'B2'], 100000),
    'value': np.random.uniform(0, 1000, 100000)
})

# Advanced filtering pattern with statistical thresholds
thresholds = {
    'metric_a_mean': df['metric_a'].mean(),
    'metric_b_std': df['metric_b'].std(),
    'value_quantile': df['value'].quantile(0.75)
}

# Optimized complex query
filtered_data = df.query(
    'metric_a > @thresholds["metric_a_mean"] and '
    'abs(metric_b - metric_b.mean()) < @thresholds["metric_b_std"] and '
    'value >= @thresholds["value_quantile"]'
)

print("Statistical filtering results:")
print(f"Original records: {len(df)}")
print(f"Filtered records: {len(filtered_data)}")
```

Slide 13: Query Error Handling and Best Practices

When working with query method, proper error handling and following best practices ensure robust and maintainable code. This example demonstrates common pitfalls and their solutions.

```python
import pandas as pd
import numpy as np

# Create sample dataset with potential problematic data
df = pd.DataFrame({
    'numeric_col': [1, 2, np.nan, 4, 5],
    'text_col': ['A', None, 'C', 'D', 'E'],
    'mixed_col': [1, 'text', 3, 4.5, np.nan],
    'date_col': pd.date_range('2023-01-01', periods=5)
})

# Safe query pattern with error handling
def safe_query(dataframe, query_string):
    try:
        result = dataframe.query(query_string)
        return result
    except pd.computation.ops.UndefinedVariableError:
        print("Error: Referenced variable not found")
        return dataframe
    except SyntaxError:
        print("Error: Invalid query syntax")
        return dataframe
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return dataframe

# Example usage with different scenarios
valid_query = safe_query(df, 'numeric_col > 2')
invalid_query = safe_query(df, 'invalid_col > 2')

# Handling missing values
clean_query = df.query('numeric_col.notna()', engine='python')
```

Slide 14: Performance Comparison with Different Filtering Methods

This comprehensive comparison demonstrates the performance differences between query method and other filtering approaches across various dataset sizes.

```python
import pandas as pd
import numpy as np
import time

def benchmark_filtering_methods(n_rows):
    # Create test dataset
    df = pd.DataFrame({
        'id': range(n_rows),
        'value': np.random.normal(0, 1, n_rows),
        'category': np.random.choice(['A', 'B', 'C'], n_rows),
        'subcategory': np.random.choice(['X', 'Y', 'Z'], n_rows)
    })
    
    times = {}
    
    # Test query method
    start = time.time()
    result1 = df.query('value > 0 and category == "A"')
    times['query'] = time.time() - start
    
    # Test boolean indexing
    start = time.time()
    result2 = df[(df['value'] > 0) & (df['category'] == 'A')]
    times['boolean'] = time.time() - start
    
    # Test loc method
    start = time.time()
    result3 = df.loc[(df['value'] > 0) & (df['category'] == 'A')]
    times['loc'] = time.time() - start
    
    return times

# Test with different dataset sizes
sizes = [1000, 10000, 100000, 1000000]
results = {size: benchmark_filtering_methods(size) for size in sizes}

# Print results
for size, times in results.items():
    print(f"\nDataset size: {size:,} rows")
    for method, time_taken in times.items():
        print(f"{method}: {time_taken:.4f} seconds")
```

Slide 15: Additional Resources

*   "Efficient Data Manipulation in Python with pandas" - [https://arxiv.org/abs/2001.00789](https://arxiv.org/abs/2001.00789)
*   "Performance Optimization Techniques for Large-Scale Data Analysis" - [https://www.sciencedirect.com/science/article/pii/S0167739X18313189](https://www.sciencedirect.com/science/article/pii/S0167739X18313189)
*   "Query Optimization in DataFrame Operations" - Consider searching on Google Scholar for recent papers on pandas optimization techniques
*   "Modern DataFrame Manipulation: A Comprehensive Review" - [https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0189-0](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0189-0)
*   Best practices and documentation: [https://pandas.pydata.org/docs/user\_guide/indexing.html#indexing-query](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing-query)

