## Advanced SQL Concepts with Python
Slide 1: Introduction to Advanced SQL Concepts with Python

Advanced SQL concepts can be leveraged effectively using Python, combining the power of relational databases with a versatile programming language. This slideshow explores various advanced SQL techniques and how to implement them using Python's database libraries.

```python
import sqlite3

# Establish a connection to the database
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary REAL
    )
''')

conn.commit()
conn.close()
```

Slide 2: Complex Joins

Complex joins allow us to combine data from multiple tables based on related columns. Let's explore a scenario with employees and their projects.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, department TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS projects
                  (id INTEGER PRIMARY KEY, name TEXT, employee_id INTEGER)''')

# Insert sample data
cursor.executemany('INSERT INTO employees VALUES (?,?,?)',
                   [(1, 'Alice', 'IT'), (2, 'Bob', 'HR'), (3, 'Charlie', 'IT')])
cursor.executemany('INSERT INTO projects VALUES (?,?,?)',
                   [(1, 'Website Redesign', 1), (2, 'Database Upgrade', 1),
                    (3, 'Recruitment Drive', 2)])

# Perform a complex join
cursor.execute('''
    SELECT e.name, e.department, p.name as project_name
    FROM employees e
    LEFT JOIN projects p ON e.id = p.employee_id
''')

for row in cursor.fetchall():
    print(row)

conn.close()
```

Slide 3: Subqueries

Subqueries are queries nested within another query, allowing for more complex data retrieval and manipulation. Let's use a subquery to find employees with above-average salaries.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Ensure the employees table exists and has salary data
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, salary REAL)''')
cursor.executemany('INSERT INTO employees VALUES (?,?,?)',
                   [(1, 'Alice', 75000), (2, 'Bob', 65000), (3, 'Charlie', 80000)])

# Subquery to find employees with above-average salary
cursor.execute('''
    SELECT name, salary
    FROM employees
    WHERE salary > (SELECT AVG(salary) FROM employees)
''')

print("Employees with above-average salary:")
for row in cursor.fetchall():
    print(f"{row[0]}: ${row[1]:.2f}")

conn.close()
```

Slide 4: Window Functions

Window functions perform calculations across a set of rows that are related to the current row. They are powerful for analytics and reporting. Let's use a window function to rank employees by salary within their department.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Ensure the employees table exists with necessary data
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL)''')
cursor.executemany('INSERT INTO employees VALUES (?,?,?,?)',
                   [(1, 'Alice', 'IT', 75000), (2, 'Bob', 'HR', 65000),
                    (3, 'Charlie', 'IT', 80000), (4, 'David', 'HR', 70000)])

# Use a window function to rank employees by salary within their department
cursor.execute('''
    SELECT name, department, salary,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dept_rank
    FROM employees
''')

print("Employee rankings by salary within department:")
for row in cursor.fetchall():
    print(f"{row[0]} ({row[1]}): ${row[2]:.2f} - Rank: {row[3]}")

conn.close()
```

Slide 5: Common Table Expressions (CTEs)

Common Table Expressions (CTEs) are named temporary result sets that exist within the scope of a single SQL statement. They can simplify complex queries and improve readability. Let's use a CTE to find the top performer in each department.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Ensure the employees table exists with necessary data
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, department TEXT, performance_score REAL)''')
cursor.executemany('INSERT INTO employees VALUES (?,?,?,?)',
                   [(1, 'Alice', 'IT', 95), (2, 'Bob', 'HR', 88),
                    (3, 'Charlie', 'IT', 92), (4, 'David', 'HR', 90)])

# Use a CTE to find the top performer in each department
cursor.execute('''
    WITH RankedEmployees AS (
        SELECT name, department, performance_score,
               RANK() OVER (PARTITION BY department ORDER BY performance_score DESC) as dept_rank
        FROM employees
    )
    SELECT name, department, performance_score
    FROM RankedEmployees
    WHERE dept_rank = 1
''')

print("Top performers by department:")
for row in cursor.fetchall():
    print(f"{row[0]} ({row[1]}): Score {row[2]}")

conn.close()
```

Slide 6: Pivoting Data

Pivoting data involves transforming rows into columns, which can be useful for reporting and data analysis. Let's pivot employee data to show department totals.

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Ensure the employees table exists with necessary data
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL)''')
cursor.executemany('INSERT INTO employees VALUES (?,?,?,?)',
                   [(1, 'Alice', 'IT', 75000), (2, 'Bob', 'HR', 65000),
                    (3, 'Charlie', 'IT', 80000), (4, 'David', 'HR', 70000),
                    (5, 'Eve', 'Marketing', 72000)])

# Fetch data and create a DataFrame
df = pd.read_sql_query("SELECT * FROM employees", conn)

# Pivot the data
pivoted = df.pivot_table(values='salary', index='department', aggfunc='sum')

print("Department salary totals:")
print(pivoted)

conn.close()
```

Slide 7: Recursive Queries

Recursive queries are useful for working with hierarchical or tree-structured data. Let's use a recursive query to traverse an employee hierarchy.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create a table for employee hierarchy
cursor.execute('''CREATE TABLE IF NOT EXISTS employee_hierarchy
                  (id INTEGER PRIMARY KEY, name TEXT, manager_id INTEGER)''')
cursor.executemany('INSERT INTO employee_hierarchy VALUES (?,?,?)',
                   [(1, 'Alice', NULL), (2, 'Bob', 1), (3, 'Charlie', 1),
                    (4, 'David', 2), (5, 'Eve', 2)])

# Recursive query to get all subordinates of Alice
cursor.execute('''
    WITH RECURSIVE subordinates AS (
        SELECT id, name, manager_id
        FROM employee_hierarchy
        WHERE name = 'Alice'
        UNION ALL
        SELECT e.id, e.name, e.manager_id
        FROM employee_hierarchy e
        JOIN subordinates s ON e.manager_id = s.id
    )
    SELECT name FROM subordinates
''')

print("Alice's subordinates (direct and indirect):")
for row in cursor.fetchall():
    print(row[0])

conn.close()
```

Slide 8: Full-Text Search

Full-text search allows efficient searching of text content. SQLite provides FTS5, a powerful full-text search engine. Let's implement a simple full-text search on a articles table.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create a virtual table using FTS5
cursor.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(title, content)''')

# Insert sample data
cursor.executemany('INSERT INTO articles_fts (title, content) VALUES (?, ?)',
                   [('Python Basics', 'Python is a versatile programming language.'),
                    ('Advanced SQL', 'SQL is powerful for data manipulation and analysis.'),
                    ('Data Science', 'Data science combines statistics, programming, and domain knowledge.')])

# Perform a full-text search
search_term = 'programming'
cursor.execute('SELECT title, content FROM articles_fts WHERE articles_fts MATCH ?', (search_term,))

print(f"Search results for '{search_term}':")
for row in cursor.fetchall():
    print(f"Title: {row[0]}")
    print(f"Content: {row[1]}\n")

conn.close()
```

Slide 9: Stored Procedures

While SQLite doesn't support stored procedures natively, we can simulate them using user-defined functions in Python. Let's create a function to calculate the average salary by department.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Ensure the employees table exists with necessary data
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL)''')
cursor.executemany('INSERT INTO employees VALUES (?,?,?,?)',
                   [(1, 'Alice', 'IT', 75000), (2, 'Bob', 'HR', 65000),
                    (3, 'Charlie', 'IT', 80000), (4, 'David', 'HR', 70000)])

# Define a function to calculate average salary by department
def avg_salary_by_dept(dept):
    cursor.execute('SELECT AVG(salary) FROM employees WHERE department = ?', (dept,))
    return cursor.fetchone()[0]

# Register the function with SQLite
conn.create_function('avg_salary_by_dept', 1, avg_salary_by_dept)

# Use the function in a query
cursor.execute('''
    SELECT DISTINCT department, avg_salary_by_dept(department) as avg_salary
    FROM employees
''')

print("Average salary by department:")
for row in cursor.fetchall():
    print(f"{row[0]}: ${row[1]:.2f}")

conn.close()
```

Slide 10: Triggers

Triggers are database operations that are automatically performed when a specified database event occurs. Let's create a trigger that logs employee salary changes.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create employees and salary_log tables
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, salary REAL)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS salary_log
                  (id INTEGER PRIMARY KEY, employee_id INTEGER, old_salary REAL, new_salary REAL, change_date TEXT)''')

# Create a trigger
cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS log_salary_changes
    AFTER UPDATE OF salary ON employees
    BEGIN
        INSERT INTO salary_log (employee_id, old_salary, new_salary, change_date)
        VALUES (OLD.id, OLD.salary, NEW.salary, DATETIME('now'));
    END
''')

# Insert and update data
cursor.execute('INSERT INTO employees (name, salary) VALUES (?, ?)', ('Alice', 75000))
cursor.execute('UPDATE employees SET salary = ? WHERE name = ?', (80000, 'Alice'))

# Check the log
cursor.execute('SELECT * FROM salary_log')
print("Salary change log:")
for row in cursor.fetchall():
    print(f"Employee ID: {row[1]}, Old Salary: ${row[2]:.2f}, New Salary: ${row[3]:.2f}, Change Date: {row[4]}")

conn.close()
```

Slide 11: Real-Life Example: Customer Order Analysis

Let's analyze customer orders to find the top-selling products and the most valuable customers.

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS customers
                  (id INTEGER PRIMARY KEY, name TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS products
                  (id INTEGER PRIMARY KEY, name TEXT, price REAL)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS orders
                  (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER)''')

# Insert sample data
cursor.executemany('INSERT INTO customers VALUES (?,?)',
                   [(1, 'John'), (2, 'Jane'), (3, 'Bob')])
cursor.executemany('INSERT INTO products VALUES (?,?,?)',
                   [(1, 'Widget', 10.0), (2, 'Gadget', 20.0), (3, 'Doohickey', 15.0)])
cursor.executemany('INSERT INTO orders VALUES (?,?,?,?)',
                   [(1, 1, 1, 5), (2, 1, 2, 2), (3, 2, 3, 3), (4, 3, 1, 10)])

# Analyze top-selling products
cursor.execute('''
    SELECT p.name, SUM(o.quantity) as total_sold
    FROM products p
    JOIN orders o ON p.id = o.product_id
    GROUP BY p.id
    ORDER BY total_sold DESC
''')

print("Top-selling products:")
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]} units")

# Analyze most valuable customers
cursor.execute('''
    SELECT c.name, SUM(p.price * o.quantity) as total_spent
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN products p ON o.product_id = p.id
    GROUP BY c.id
    ORDER BY total_spent DESC
''')

print("\nMost valuable customers:")
for row in cursor.fetchall():
    print(f"{row[0]}: ${row[1]:.2f}")

conn.close()
```

Slide 12: Real-Life Example: Social Network Analysis

Let's analyze a simple social network to find the most connected users and suggest potential connections.

```python
import sqlite3
import networkx as nx

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''CREATE TABLE IF NOT EXISTS users
                  (id INTEGER PRIMARY KEY, name TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS connections
                  (user_id INTEGER, friend_id INTEGER)''')

# Insert sample data
cursor.executemany('INSERT INTO users VALUES (?,?)',
                   [(1, 'Alice'), (2, 'Bob'), (3, 'Charlie'), (4, 'David'), (5, 'Eve')])
cursor.executemany('INSERT INTO connections VALUES (?,?)',
                   [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5)])

# Find most connected users
cursor.execute('''
    SELECT u.name, COUNT(c.friend_id) as connection_count
    FROM users u
    LEFT JOIN connections c ON u.id = c.user_id
    GROUP BY u.id
    ORDER BY connection_count DESC
''')

print("Most connected users:")
for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]} connections")

# Build a graph
G = nx.Graph()
cursor.execute('SELECT user_id, friend_id FROM connections')
for row in cursor.fetchall():
    G.add_edge(row[0], row[1])

# Suggest potential connections
print("\nPotential connections:")
for user in G.nodes():
    friends = set(G.neighbors(user))
    friends_of_friends = set()
    for friend in friends:
        friends_of_friends.update(G.neighbors(friend))
    potential_connections = friends_of_friends - friends - {user}
    if potential_connections:
        user_name = cursor.execute('SELECT name FROM users WHERE id = ?', (user,)).fetchone()[0]
        print(f"{user_name}: {', '.join(cursor.execute('SELECT name FROM users WHERE id IN ({})'.format(','.join('?' * len(potential_connections))), list(potential_connections)).fetchall()[0])}")

conn.close()
```

Slide 13: Advanced Indexing Techniques

Proper indexing is crucial for optimizing SQL query performance. Let's explore some advanced indexing techniques using SQLite.

```python
import sqlite3
import time

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create a table with a large number of rows
cursor.execute('''CREATE TABLE IF NOT EXISTS large_table
                  (id INTEGER PRIMARY KEY, data TEXT, category TEXT)''')

# Insert sample data
data = [('data' + str(i), 'category' + str(i % 5)) for i in range(100000)]
cursor.executemany('INSERT INTO large_table (data, category) VALUES (?, ?)', data)

# Query without index
start_time = time.time()
cursor.execute('SELECT * FROM large_table WHERE category = "category3"')
cursor.fetchall()
print(f"Query without index: {time.time() - start_time:.4f} seconds")

# Create an index on the category column
cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON large_table (category)')

# Query with index
start_time = time.time()
cursor.execute('SELECT * FROM large_table WHERE category = "category3"')
cursor.fetchall()
print(f"Query with index: {time.time() - start_time:.4f} seconds")

# Create a composite index
cursor.execute('CREATE INDEX IF NOT EXISTS idx_category_data ON large_table (category, data)')

# Query using the composite index
start_time = time.time()
cursor.execute('SELECT * FROM large_table WHERE category = "category3" AND data LIKE "data1%"')
cursor.fetchall()
print(f"Query with composite index: {time.time() - start_time:.4f} seconds")

conn.close()
```

Slide 14: Query Optimization Techniques

Optimizing SQL queries is essential for improving database performance. Let's explore some techniques to optimize complex queries.

```python
import sqlite3

conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Create sample tables
cursor.execute('''CREATE TABLE IF NOT EXISTS orders
                  (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS order_items
                  (id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS products
                  (id INTEGER PRIMARY KEY, name TEXT, price REAL)''')

# Insert sample data (omitted for brevity)

# Unoptimized query
cursor.execute('''
    SELECT o.id, SUM(oi.quantity * p.price) as total_value
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE o.order_date >= '2023-01-01'
    GROUP BY o.id
''')

# Optimized query using subquery
cursor.execute('''
    SELECT o.id, 
           (SELECT SUM(oi.quantity * p.price)
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE oi.order_id = o.id) as total_value
    FROM orders o
    WHERE o.order_date >= '2023-01-01'
''')

# Using EXPLAIN QUERY PLAN to analyze query performance
cursor.execute('EXPLAIN QUERY PLAN ' + '''
    SELECT o.id, 
           (SELECT SUM(oi.quantity * p.price)
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE oi.order_id = o.id) as total_value
    FROM orders o
    WHERE o.order_date >= '2023-01-01'
''')
print("Query execution plan:")
for row in cursor.fetchall():
    print(row)

conn.close()
```

Slide 15: Additional Resources

For further exploration of advanced SQL concepts and their implementation in Python, consider the following resources:

1. SQLite Documentation: [https://www.sqlite.org/docs.html](https://www.sqlite.org/docs.html)
2. Python SQLite3 Module Documentation: [https://docs.python.org/3/library/sqlite3.html](https://docs.python.org/3/library/sqlite3.html)
3. "Advanced SQL for Data Scientists" by Zachary Thomas (arXiv:2004.07703): [https://arxiv.org/abs/2004.07703](https://arxiv.org/abs/2004.07703)
4. "Query Optimization Techniques in Database Systems" by Surajit Chaudhuri (arXiv:1806.00948): [https://arxiv.org/abs/1806.00948](https://arxiv.org/abs/1806.00948)

These resources provide in-depth information on SQL optimization, advanced querying techniques, and integration with Python for data analysis and manipulation.

