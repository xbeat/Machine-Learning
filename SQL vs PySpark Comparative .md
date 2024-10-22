## SQL vs PySpark Comparative 
Slide 1: Introduction to SQL and PySpark

Data manipulation and analysis can be performed using both SQL and PySpark. These technologies serve similar purposes but operate differently. SQL is a standard language for relational databases, while PySpark is a Python API for Apache Spark, designed for big data processing.

```python
# SQL Example
sql_query = """
SELECT name, age 
FROM users 
WHERE age > 25"""

# PySpark Equivalent
from pyspark.sql import SparkSession
spark_df.select("name", "age").filter("age > 25")
```

Slide 2: Creating Tables and DataFrames

SQL creates tables in a relational database, while PySpark creates distributed DataFrames in memory.

```python
# SQL
create_table = """
CREATE TABLE employees (
    id INT,
    name VARCHAR(50),
    department VARCHAR(50)
)"""

# PySpark
from pyspark.sql.types import *
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("department", StringType(), True)
])
df = spark.createDataFrame([], schema)
```

Slide 3: Data Selection

Both SQL and PySpark offer ways to select specific columns and filter data. The syntax differs but the concept remains similar.

```python
# SQL
sql_query = """
SELECT name, age 
FROM employees 
WHERE department = 'IT'"""

# PySpark
df.select("name", "age").filter(col("department") == "IT")
```

Slide 4: Aggregations

Performing grouping operations and aggregations is fundamental in data analysis. Both technologies provide robust aggregation capabilities.

```python
# SQL
sql_query = """
SELECT department, COUNT(*) as count, AVG(salary) as avg_salary
FROM employees
GROUP BY department"""

# PySpark
df.groupBy("department").agg(
    count("*").alias("count"),
    avg("salary").alias("avg_salary")
)
```

Slide 5: Real-Life Example - Weather

Analysis Analyzing temperature readings from multiple weather stations across different cities.

```python
# SQL
sql_query = """
SELECT city, 
       AVG(temperature) as avg_temp,
       COUNT(*) as readings
FROM weather_readings
WHERE year = 2023
GROUP BY city
HAVING COUNT(*) > 100"""

# PySpark
weather_df.filter(col("year") == 2023)\
    .groupBy("city")\
    .agg(
        avg("temperature").alias("avg_temp"),
        count("*").alias("readings")
    )\
    .filter(col("readings") > 100)
```

Slide 6: Real-Life Example - Student

Performance Analysis Analyzing student grades across different subjects and calculating performance metrics.

```python
# SQL
sql_query = """
SELECT subject,
       AVG(score) as avg_score,
       COUNT(DISTINCT student_id) as student_count
FROM exam_results
GROUP BY subject
HAVING AVG(score) < 75"""

# PySpark
exam_df.groupBy("subject")\
    .agg(
        avg("score").alias("avg_score"),
        countDistinct("student_id").alias("student_count")
    )\
    .filter(col("avg_score") < 75)
```

Slide 7: Joins in SQL and PySpark

Both platforms support various types of joins to combine data from multiple sources.

```python
# SQL
sql_query = """
SELECT s.name, c.course_name
FROM students s
LEFT JOIN courses c
ON s.course_id = c.id"""

# PySpark
students_df.join(
    courses_df,
    students_df.course_id == courses_df.id,
    "left"
)
```

Slide 8: Window Functions

Window functions allow calculations across a set of rows related to the current row.

```python
# SQL
sql_query = """
SELECT name,
       score,
       AVG(score) OVER (PARTITION BY subject) as avg_subject_score
FROM exam_results"""

# PySpark
from pyspark.sql.window import Window
window_spec = Window.partitionBy("subject")
exam_df.withColumn(
    "avg_subject_score",
    avg("score").over(window_spec)
)
```

Slide 9: Handling Missing

Values Different approaches to handle null values in both SQL and PySpark.

```python
# SQL
sql_query = """
SELECT name,
       COALESCE(age, 0) as age,
       NULLIF(department, 'Unknown') as dept
FROM employees"""

# PySpark
df.na.fill({"age": 0})\
    .withColumn(
        "dept",
        when(col("department") == "Unknown", None)\
        .otherwise(col("department"))
    )
```

Slide 10: String Operations

Both SQL and PySpark provide functions for string manipulation.

```python
# SQL
sql_query = """
SELECT UPPER(name) as upper_name,
       SUBSTRING(description, 1, 10) as short_desc
FROM products"""

# PySpark
from pyspark.sql.functions import upper, substring
df.select(
    upper("name").alias("upper_name"),
    substring("description", 1, 10).alias("short_desc")
)
```

Slide 11: Complex Data Types

Handling arrays and structs in both platforms.

```python
# SQL
sql_query = """
SELECT name,
       tags[1] as first_tag,
       metadata->>'city' as city
FROM products"""

# PySpark
df.select(
    "name",
    col("tags").getItem(0).alias("first_tag"),
    col("metadata.city")
)
```

Slide 12: Performance Optimization

Both SQL and PySpark offer ways to optimize query performance.

```python
# SQL with indexing
sql_query = """
CREATE INDEX idx_department
ON employees(department)
WHERE department IS NOT NULL"""

# PySpark with caching
df.cache()  # Cache DataFrame in memory
df.repartition(10)  # Optimize partitioning
```

Slide 13: Additional Resources

For more detailed information about SQL and PySpark integration, refer to:

*   "Distributed Computing with PySpark SQL: A Comparative Study" (arXiv:2103.07538)
*   "Performance Analysis of SparkSQL vs Traditional SQL" (arXiv:1906.04516)

These papers provide comprehensive comparisons and performance analyses of both technologies.

