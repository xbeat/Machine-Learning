## Mastering SQL Query Execution Order
Slide 1: SQL Query Execution Order

Understanding the execution order of SQL queries is crucial for writing efficient and correct database operations. SQL follows a logical order when processing queries, which may differ from the written order.

```python
# Simplified representation of SQL query execution order
def sql_execution_order(query):
    steps = [
        "FROM",
        "WHERE",
        "GROUP BY",
        "HAVING",
        "SELECT",
        "ORDER BY",
        "LIMIT"
    ]
    
    for step in steps:
        if step.lower() in query.lower():
            print(f"Execute {step}")

# Example usage
sample_query = """
SELECT column1, AVG(column2) AS avg_col2
FROM table1
WHERE condition
GROUP BY column1
HAVING AVG(column2) > 100
ORDER BY avg_col2 DESC
LIMIT 10
"""

sql_execution_order(sample_query)
```

Slide 2: FROM Clause

The FROM clause is the starting point of query execution. It specifies the table(s) from which data will be retrieved. In cases of multiple tables, joins are performed at this stage.

```python
import sqlite3

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER
    )
''')

# Insert sample data
cursor.executemany('INSERT INTO employees VALUES (?, ?, ?, ?)', [
    (1, 'Alice', 'HR', 50000),
    (2, 'Bob', 'IT', 60000),
    (3, 'Charlie', 'Finance', 55000)
])

# Execute a query with FROM clause
cursor.execute('SELECT * FROM employees')
print(cursor.fetchall())
```

Slide 3: WHERE Clause

After the FROM clause, the WHERE clause filters rows based on specified conditions. This step reduces the dataset before further processing.

```python
# Continuing from the previous example
cursor.execute('SELECT * FROM employees WHERE salary > 55000')
print(cursor.fetchall())
```

Slide 4: GROUP BY Clause

The GROUP BY clause groups rows that have the same values in specified columns. It's often used with aggregate functions to perform calculations on each group.

```python
cursor.execute('''
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
''')
print(cursor.fetchall())
```

Slide 5: HAVING Clause

The HAVING clause filters groups based on specified conditions. It's similar to WHERE, but operates on grouped data rather than individual rows.

```python
cursor.execute('''
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
    HAVING AVG(salary) > 52000
''')
print(cursor.fetchall())
```

Slide 6: SELECT Clause

The SELECT clause determines which columns will be included in the final result set. It's executed after filtering and grouping operations.

```python
cursor.execute('''
    SELECT name, department
    FROM employees
    WHERE salary > 52000
''')
print(cursor.fetchall())
```

Slide 7: ORDER BY Clause

The ORDER BY clause sorts the result set based on specified columns. It's one of the last operations performed before returning results.

```python
cursor.execute('''
    SELECT *
    FROM employees
    ORDER BY salary DESC
''')
print(cursor.fetchall())
```

Slide 8: LIMIT Clause

The LIMIT clause restricts the number of rows returned in the result set. It's typically the last operation performed in query execution.

```python
cursor.execute('''
    SELECT *
    FROM employees
    ORDER BY salary DESC
    LIMIT 2
''')
print(cursor.fetchall())
```

Slide 9: Subqueries and Execution Order

Subqueries are executed before the main query. They can appear in various parts of the main query, affecting the execution order.

```python
cursor.execute('''
    SELECT name, salary
    FROM employees
    WHERE salary > (SELECT AVG(salary) FROM employees)
''')
print(cursor.fetchall())
```

Slide 10: Join Operations

Join operations are performed early in the query execution process, typically right after the FROM clause.

```python
# Create a new table for departments
cursor.execute('''
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT,
        location TEXT
    )
''')

cursor.executemany('INSERT INTO departments VALUES (?, ?, ?)', [
    (1, 'HR', 'New York'),
    (2, 'IT', 'San Francisco'),
    (3, 'Finance', 'Chicago')
])

# Perform a join operation
cursor.execute('''
    SELECT e.name, e.salary, d.location
    FROM employees e
    JOIN departments d ON e.department = d.name
''')
print(cursor.fetchall())
```

Slide 11: Window Functions

Window functions are processed after the WHERE clause but before the SELECT list is processed. They allow calculations across sets of rows that are related to the current row.

```python
# SQLite doesn't support window functions, so we'll use a Python equivalent
import pandas as pd

# Convert SQLite results to a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM employees", conn)

# Add a column with cumulative sum of salary within each department
df['cumulative_salary'] = df.groupby('department')['salary'].cumsum()

print(df)
```

Slide 12: Real-life Example: Customer Order Analysis

Let's analyze customer orders to find the top products by revenue for each category.

```python
# Create tables for products and orders
cursor.executescript('''
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        category TEXT,
        price REAL
    );
    
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        product_id INTEGER,
        quantity INTEGER,
        FOREIGN KEY (product_id) REFERENCES products (id)
    );

    INSERT INTO products VALUES
    (1, 'Laptop', 'Electronics', 1000),
    (2, 'Smartphone', 'Electronics', 500),
    (3, 'T-shirt', 'Clothing', 20),
    (4, 'Jeans', 'Clothing', 50);

    INSERT INTO orders VALUES
    (1, 1, 5),
    (2, 2, 10),
    (3, 3, 20),
    (4, 4, 15);
''')

cursor.execute('''
    SELECT p.category, p.name, SUM(p.price * o.quantity) as revenue
    FROM products p
    JOIN orders o ON p.id = o.product_id
    GROUP BY p.category, p.name
    HAVING revenue = (
        SELECT MAX(sub.revenue)
        FROM (
            SELECT p2.category, p2.name, SUM(p2.price * o2.quantity) as revenue
            FROM products p2
            JOIN orders o2 ON p2.id = o2.product_id
            GROUP BY p2.category, p2.name
        ) sub
        WHERE sub.category = p.category
    )
    ORDER BY revenue DESC
''')

print(cursor.fetchall())
```

Slide 13: Real-life Example: Employee Performance Analysis

Analyze employee performance by comparing individual salaries to department averages.

```python
cursor.execute('''
    WITH dept_avg AS (
        SELECT department, AVG(salary) as avg_salary
        FROM employees
        GROUP BY department
    )
    SELECT e.name, e.department, e.salary,
           d.avg_salary,
           (e.salary - d.avg_salary) as salary_diff
    FROM employees e
    JOIN dept_avg d ON e.department = d.department
    ORDER BY salary_diff DESC
''')

print(cursor.fetchall())
```

Slide 14: Additional Resources

For more in-depth information on SQL query execution and optimization, consider exploring the following resources:

1.  "Query Execution in Database Systems" by Goetz Graefe (ArXiv:1208.6445) URL: [https://arxiv.org/abs/1208.6445](https://arxiv.org/abs/1208.6445)
2.  "A Survey of Query Optimization in Parallel Database Systems" by Weiping Yan and Per-Åke Larson (ArXiv:cs/9501102) URL: [https://arxiv.org/abs/cs/9501102](https://arxiv.org/abs/cs/9501102)

These papers provide comprehensive insights into query execution strategies and optimization techniques in database systems.

