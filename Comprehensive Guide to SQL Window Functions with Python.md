## Comprehensive Guide to SQL Window Functions with Python
Slide 1: Introduction to Window Functions in SQL

Window functions are a powerful feature in SQL that allow you to perform calculations across a set of rows that are related to the current row. They provide a way to analyze data within a specific "window" of rows, enabling complex analytical queries and advanced data manipulation.

```python
import pandas as pd

# Create a sample dataset
data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'sales': [100, 150, 200, 120, 180]
}
df = pd.DataFrame(data)

# Calculate a 3-day moving average using a window function
df['moving_avg'] = df['sales'].rolling(window=3).mean()

print(df)
```

Slide 2: Basic Syntax of Window Functions

Window functions in SQL follow a specific syntax. They are typically used in the SELECT clause and include an OVER clause that defines the window of rows on which the function operates. The OVER clause can include PARTITION BY and ORDER BY to further refine the window.

```sql
SELECT column1, column2,
       WINDOW_FUNCTION(column3) OVER (
           PARTITION BY column4
           ORDER BY column5
           ROWS BETWEEN start_point AND end_point
       ) AS result_column
FROM table_name;
```

Slide 3: Types of Window Functions

SQL provides various types of window functions, including:

1. Aggregate functions (SUM, AVG, COUNT, etc.)
2. Ranking functions (ROW\_NUMBER, RANK, DENSE\_RANK)
3. Offset functions (LAG, LEAD)
4. Statistical functions (PERCENTILE\_CONT, CUME\_DIST)

```python
import pandas as pd

data = {
    'department': ['A', 'A', 'B', 'B', 'C'],
    'employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'salary': [50000, 55000, 60000, 65000, 70000]
}
df = pd.DataFrame(data)

# Example of a ranking function (ROW_NUMBER)
df['rank'] = df.groupby('department')['salary'].rank(method='dense', ascending=False)

print(df)
```

Slide 4: PARTITION BY Clause

The PARTITION BY clause divides the result set into partitions to which the window function is applied separately. It's similar to GROUP BY, but doesn't reduce the number of rows returned.

```sql
SELECT employee_name, department, salary,
       AVG(salary) OVER (PARTITION BY department) AS dept_avg_salary
FROM employees;
```

Slide 5: ORDER BY Clause in Window Functions

The ORDER BY clause within the OVER clause determines the order in which rows are processed by the window function. This is crucial for functions like ROW\_NUMBER or running totals.

```python
import pandas as pd

data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'sales': [100, 150, 200, 120, 180]
}
df = pd.DataFrame(data)

# Calculate cumulative sum of sales
df['cumulative_sales'] = df['sales'].cumsum()

print(df)
```

Slide 6: ROW\_NUMBER() Function

ROW\_NUMBER() assigns a unique integer to each row within its partition. It's often used for generating sequential numbers or identifying duplicate rows.

```sql
SELECT ROW_NUMBER() OVER (ORDER BY date) AS row_num,
       date, sales
FROM sales_data;
```

Slide 7: RANK() and DENSE\_RANK() Functions

RANK() and DENSE\_RANK() are similar to ROW\_NUMBER() but handle ties differently. RANK() leaves gaps in the ranking when there are ties, while DENSE\_RANK() doesn't.

```python
import pandas as pd

data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'score': [85, 92, 92, 88, 95]
}
df = pd.DataFrame(data)

df['rank'] = df['score'].rank(method='min', ascending=False)
df['dense_rank'] = df['score'].rank(method='dense', ascending=False)

print(df)
```

Slide 8: LAG() and LEAD() Functions

LAG() and LEAD() allow you to access data from other rows in relation to the current row. LAG() looks at previous rows, while LEAD() looks at subsequent rows.

```sql
SELECT date, sales,
       LAG(sales) OVER (ORDER BY date) AS previous_day_sales,
       LEAD(sales) OVER (ORDER BY date) AS next_day_sales
FROM sales_data;
```

Slide 9: Running Totals and Moving Averages

Window functions are excellent for calculating running totals and moving averages, which are common in time series analysis.

```python
import pandas as pd

data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'sales': [100, 150, 200, 120, 180]
}
df = pd.DataFrame(data)

# Calculate running total and 3-day moving average
df['running_total'] = df['sales'].cumsum()
df['moving_avg'] = df['sales'].rolling(window=3).mean()

print(df)
```

Slide 10: FIRST\_VALUE() and LAST\_VALUE() Functions

These functions allow you to retrieve the first or last value in an ordered set of values. They're useful for comparing current values with starting or ending values in a window.

```sql
SELECT date, product, sales,
       FIRST_VALUE(sales) OVER (PARTITION BY product ORDER BY date) AS first_sale,
       LAST_VALUE(sales) OVER (PARTITION BY product ORDER BY date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_sale
FROM sales_data;
```

Slide 11: NTILE() Function

NTILE() divides the rows into a specified number of ranked groups. It's useful for creating equal-sized buckets of data.

```python
import pandas as pd
import numpy as np

data = {
    'name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'score': [85, 92, 78, 95, 88, 72, 93, 87, 81, 79]
}
df = pd.DataFrame(data)

# Divide into 4 quartiles
df['quartile'] = pd.qcut(df['score'], q=4, labels=False)

print(df)
```

Slide 12: Window Frame Clause

The window frame clause (ROWS or RANGE) defines the set of rows within a partition. It allows for more granular control over which rows are included in window function calculations.

```sql
SELECT date, sales,
       AVG(sales) OVER (ORDER BY date
                        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM sales_data;
```

Slide 13: Real-Life Example: Customer Loyalty Program

Imagine a customer loyalty program where we want to rank customers based on their total purchases and identify top spenders in each category.

```python
import pandas as pd

data = {
    'customer_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'category': ['Electronics', 'Books', 'Electronics', 'Clothing', 'Books', 'Electronics', 'Clothing', 'Books'],
    'purchase_amount': [500, 50, 300, 100, 75, 200, 150, 80]
}
df = pd.DataFrame(data)

# Calculate total purchases and rank customers
df['total_purchases'] = df.groupby('customer_id')['purchase_amount'].transform('sum')
df['overall_rank'] = df['total_purchases'].rank(method='dense', ascending=False)
df['category_rank'] = df.groupby('category')['purchase_amount'].rank(method='dense', ascending=False)

print(df)
```

Slide 14: Real-Life Example: Website Traffic Analysis

Let's analyze website traffic data to identify trends and patterns in page views over time.

```python
import pandas as pd

data = {
    'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
    'page_views': [1000, 1200, 1100, 1300, 1500, 1400, 1600, 1800, 1700, 1900]
}
df = pd.DataFrame(data)

# Calculate 3-day moving average and daily change
df['moving_avg'] = df['page_views'].rolling(window=3).mean()
df['daily_change'] = df['page_views'].diff()
df['percent_change'] = df['page_views'].pct_change() * 100

# Rank days by page views
df['rank'] = df['page_views'].rank(method='dense', ascending=False)

print(df)
```

Slide 15: Additional Resources

For more in-depth information on Window Functions in SQL, consider exploring these resources:

1. "SQL Window Functions: Frames, Partitions and Groups" by David Brinton (arXiv:2008.09907) URL: [https://arxiv.org/abs/2008.09907](https://arxiv.org/abs/2008.09907)
2. "SQL Performance Explained: Everything Developers Need to Know about SQL Performance" by Markus Winand (While not on arXiv, this book is a valuable resource for understanding SQL performance, including window functions)

Remember to always refer to the official documentation of your specific SQL database system for the most accurate and up-to-date information on window function implementation and syntax.

