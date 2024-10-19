## Mastering SQL in Python with PandaSQL
Slide 1: Introduction to PandaSQL

PandaSQL is a powerful library that bridges the gap between SQL and pandas DataFrames in Python. It allows users to write SQL queries directly on pandas DataFrames, combining the familiarity of SQL with the flexibility of pandas.

```python
import pandas as pd
import pandasql as ps

# Create a sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Paris']
})

# Run a SQL query on the DataFrame
query = "SELECT * FROM df WHERE age > 28"
result = ps.sqldf(query, locals())
print(result)
```

Slide 2: Setting Up PandaSQL

To get started with PandaSQL, you need to install it using pip. Once installed, you can import it alongside pandas to begin querying your DataFrames.

```python
# Install PandaSQL
!pip install pandasql

# Import necessary libraries
import pandas as pd
import pandasql as ps

# Create a sample DataFrame
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'A', 'B'],
    'quantity': [10, 20, 15, 5, 25],
    'price': [100, 200, 150, 100, 180]
})

print(df)
```

Slide 3: Basic SQL Queries with PandaSQL

PandaSQL allows you to write SQL queries as strings and execute them on pandas DataFrames. Let's start with a simple SELECT query to retrieve all rows from our DataFrame.

```python
# Basic SELECT query
query = "SELECT * FROM df"
result = ps.sqldf(query, locals())
print(result)
```

Slide 4: Filtering Data with WHERE Clause

You can use the WHERE clause in your SQL queries to filter data based on specific conditions. This is equivalent to using boolean indexing in pandas.

```python
# Filtering data with WHERE clause
query = "SELECT * FROM df WHERE quantity > 15"
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
pandas_result = df[df['quantity'] > 15]
print("\nPandas equivalent:")
print(pandas_result)
```

Slide 5: Aggregating Data with GROUP BY

PandaSQL supports SQL aggregations using GROUP BY, which is similar to pandas' groupby() method followed by aggregation functions.

```python
# Aggregating data with GROUP BY
query = """
SELECT product, SUM(quantity) as total_quantity, AVG(price) as avg_price
FROM df
GROUP BY product
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
pandas_result = df.groupby('product').agg({'quantity': 'sum', 'price': 'mean'})
pandas_result.columns = ['total_quantity', 'avg_price']
print("\nPandas equivalent:")
print(pandas_result)
```

Slide 6: Joining DataFrames

PandaSQL allows you to join multiple DataFrames using SQL JOIN syntax, which can be more intuitive for those familiar with SQL compared to pandas merge() function.

```python
# Create two sample DataFrames
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'id': [2, 3, 4], 'city': ['London', 'Paris', 'Berlin']})

# Join DataFrames using SQL
query = """
SELECT df1.id, df1.name, df2.city
FROM df1
LEFT JOIN df2 ON df1.id = df2.id
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
pandas_result = pd.merge(df1, df2, on='id', how='left')
print("\nPandas equivalent:")
print(pandas_result)
```

Slide 7: Subqueries and Complex Operations

PandaSQL supports subqueries and complex SQL operations, which can sometimes be more straightforward than nested pandas operations.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'department': ['A', 'A', 'B', 'B', 'C'],
    'employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'salary': [50000, 60000, 55000, 65000, 70000]
})

# Use a subquery to find employees with above-average salary
query = """
SELECT department, employee, salary
FROM df
WHERE salary > (SELECT AVG(salary) FROM df)
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
avg_salary = df['salary'].mean()
pandas_result = df[df['salary'] > avg_salary]
print("\nPandas equivalent:")
print(pandas_result)
```

Slide 8: Window Functions in PandaSQL

PandaSQL supports window functions, which can be used for operations like running totals or ranking. These are similar to pandas' expanding() and rank() methods.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=5),
    'sales': [100, 150, 200, 120, 180]
})

# Use window function for cumulative sum
query = """
SELECT date, sales,
       SUM(sales) OVER (ORDER BY date) as cumulative_sales
FROM df
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
df['cumulative_sales'] = df['sales'].cumsum()
print("\nPandas equivalent:")
print(df)
```

Slide 9: Real-Life Example: Analyzing Student Performance

Let's use PandaSQL to analyze student performance data, demonstrating how it can be used in educational contexts.

```python
# Create a sample DataFrame of student scores
students_df = pd.DataFrame({
    'student_id': range(1, 11),
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 
             'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'math_score': [85, 92, 78, 95, 88, 72, 90, 83, 79, 94],
    'science_score': [92, 88, 75, 89, 95, 80, 85, 88, 92, 86],
    'literature_score': [78, 85, 90, 82, 87, 88, 91, 76, 84, 89]
})

# Calculate average scores and rank students
query = """
SELECT name,
       (math_score + science_score + literature_score) / 3.0 as avg_score,
       RANK() OVER (ORDER BY (math_score + science_score + literature_score) DESC) as rank
FROM students_df
ORDER BY avg_score DESC
"""
result = ps.sqldf(query, locals())
print(result)
```

Slide 10: Real-Life Example: Analyzing Sensor Data

In this example, we'll use PandaSQL to analyze sensor data, demonstrating its application in IoT and environmental monitoring scenarios.

```python
# Create a sample DataFrame of sensor readings
import numpy as np

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
sensor_df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.random.normal(20, 5, len(dates)),
    'humidity': np.random.normal(60, 10, len(dates)),
    'air_quality': np.random.normal(50, 20, len(dates))
})

# Analyze daily averages and flag unusual readings
query = """
SELECT 
    DATE(timestamp) as date,
    AVG(temperature) as avg_temp,
    AVG(humidity) as avg_humidity,
    AVG(air_quality) as avg_air_quality,
    CASE 
        WHEN AVG(temperature) > 25 OR AVG(humidity) > 70 OR AVG(air_quality) > 100 
        THEN 'Alert' 
        ELSE 'Normal' 
    END as status
FROM sensor_df
GROUP BY DATE(timestamp)
HAVING status = 'Alert'
ORDER BY date
"""
result = ps.sqldf(query, locals())
print(result)
```

Slide 11: Performance Considerations

While PandaSQL provides a familiar SQL interface, it's important to consider performance implications, especially for large datasets or complex queries.

```python
import time

# Create a larger DataFrame
large_df = pd.DataFrame({
    'id': range(100000),
    'value': np.random.randn(100000)
})

# Measure time for PandaSQL query
start_time = time.time()
query = "SELECT * FROM large_df WHERE value > 0"
result = ps.sqldf(query, locals())
pandasql_time = time.time() - start_time

# Measure time for equivalent pandas operation
start_time = time.time()
pandas_result = large_df[large_df['value'] > 0]
pandas_time = time.time() - start_time

print(f"PandaSQL time: {pandasql_time:.4f} seconds")
print(f"Pandas time: {pandas_time:.4f} seconds")
```

Slide 12: Best Practices and Tips

When using PandaSQL, consider these best practices to optimize your workflow and query performance:

1.  Use PandaSQL for complex queries where SQL syntax is more intuitive.
2.  For simple operations, stick to native pandas methods for better performance.
3.  Leverage PandaSQL's support for window functions and subqueries when appropriate.
4.  Be mindful of memory usage, especially with large datasets.
5.  Use appropriate indexing in your pandas DataFrames to speed up PandaSQL queries.
6.  Always compare the performance of PandaSQL queries with equivalent pandas operations for critical tasks.

```python
# Example of using appropriate indexing
df = pd.DataFrame({
    'id': range(1000000),
    'value': np.random.randn(1000000)
})
df.set_index('id', inplace=True)

# PandaSQL query using the index
query = "SELECT * FROM df WHERE id BETWEEN 500000 AND 500010"
result = ps.sqldf(query, locals())
print(result)
```

Slide 13: Conclusion and Future Directions

PandaSQL bridges the gap between SQL and pandas, offering a powerful tool for data analysis in Python. It's particularly useful for those transitioning from SQL to pandas or working in environments where SQL is the primary query language. As data processing needs evolve, libraries like PandaSQL may continue to adapt, potentially incorporating features like:

1.  Support for more advanced SQL features
2.  Improved performance optimizations
3.  Integration with big data technologies

Keep an eye on the PandaSQL project for future updates and enhancements.

```python
# Example of a more advanced query combining multiple features
query = """
WITH ranked_data AS (
    SELECT *,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
    FROM df
)
SELECT department, employee, salary, salary_rank
FROM ranked_data
WHERE salary_rank <= 2
ORDER BY department, salary_rank
"""
result = ps.sqldf(query, locals())
print(result)
```

Slide 14: Additional Resources

For those interested in diving deeper into PandaSQL and related topics, here are some valuable resources:

1.  PandaSQL GitHub Repository: [https://github.com/yhat/pandasql](https://github.com/yhat/pandasql)
2.  "Pandas: Powerful Python Data Analysis Toolkit" by Wes McKinney (ArXiv:1402.1726): [https://arxiv.org/abs/1402.1726](https://arxiv.org/abs/1402.1726)
3.  "SQL for Data Scientists: A Beginner's Guide for Building Datasets for Analysis" by Renee M. P. Teate
4.  Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
5.  SQLite Documentation (PandaSQL uses SQLite under the hood): [https://www.sqlite.org/docs.html](https://www.sqlite.org/docs.html)

These resources will help you further explore the capabilities of PandaSQL and enhance your data analysis skills in Python.

