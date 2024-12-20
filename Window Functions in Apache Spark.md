## Window Functions in Apache Spark
Slide 1: Introduction to Window Functions in Apache Spark

Window functions in Apache Spark allow you to perform calculations across a set of rows that are related to the current row. These functions operate on a window of data, defined by partitioning and ordering specifications. They are powerful tools for complex analytics, enabling operations like running totals, rankings, and moving averages.

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, rank, sum

# Create a SparkSession
spark = SparkSession.builder.appName("WindowFunctionsIntro").getOrCreate()

# Sample data
data = [("Alice", "Sales", 1000), ("Bob", "Sales", 1500), 
        ("Charlie", "Marketing", 2000), ("David", "Marketing", 2500)]
df = spark.createDataFrame(data, ["name", "department", "salary"])

# Define a window specification
windowSpec = Window.partitionBy("department").orderBy("salary")

# Apply a window function
df_with_rank = df.withColumn("rank", rank().over(windowSpec))

df_with_rank.show()
```

Slide 2: Window Specification

A window specification defines how to group and order the data for window function calculations. It typically includes partitioning (grouping) and ordering clauses. The Window.partitionBy() method specifies the grouping, while orderBy() determines the order within each partition.

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import col

# Define a window specification
windowSpec = Window.partitionBy("department").orderBy(col("salary").desc())

# Use the window specification with a window function
df_with_window = df.withColumn("max_salary", max("salary").over(windowSpec))

df_with_window.show()
```

Slide 3: Ranking Functions

Ranking functions assign ranks to rows within a partition. Common ranking functions include rank(), dense\_rank(), and row\_number(). These functions are useful for identifying top performers, creating leaderboards, or finding the nth highest/lowest values.

```python
from pyspark.sql.functions import rank, dense_rank, row_number

# Apply different ranking functions
df_ranked = df.withColumn("rank", rank().over(windowSpec)) \
              .withColumn("dense_rank", dense_rank().over(windowSpec)) \
              .withColumn("row_number", row_number().over(windowSpec))

df_ranked.show()
```

Slide 4: Aggregate Functions

Window functions can use aggregate functions like sum(), avg(), and count() to compute running totals, moving averages, or cumulative counts. These are particularly useful for time-series analysis and cumulative calculations.

```python
from pyspark.sql.functions import sum, avg, count

# Calculate running total and moving average
df_agg = df.withColumn("running_total", sum("salary").over(windowSpec)) \
           .withColumn("moving_avg", avg("salary").over(windowSpec))

df_agg.show()
```

Slide 5: Offset Functions

Offset functions allow access to row values at specified offsets from the current row. Common offset functions include lag() and lead(). These are useful for comparing current values with previous or future values in a sequence.

```python
from pyspark.sql.functions import lag, lead

# Calculate difference from previous and next salary
df_offset = df.withColumn("prev_salary", lag("salary").over(windowSpec)) \
               .withColumn("next_salary", lead("salary").over(windowSpec)) \
               .withColumn("diff_prev", col("salary") - col("prev_salary")) \
               .withColumn("diff_next", col("next_salary") - col("salary"))

df_offset.show()
```

Slide 6: Unbounded Windows

Unbounded windows allow calculations across all rows in a partition, regardless of the current row's position. This is useful for computing overall aggregates within each partition.

```python
from pyspark.sql.functions import sum

# Define an unbounded window
unboundedWindow = Window.partitionBy("department").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

# Calculate total salary and percentage of total for each employee
df_unbounded = df.withColumn("total_salary", sum("salary").over(unboundedWindow)) \
                 .withColumn("percentage", (col("salary") / col("total_salary")) * 100)

df_unbounded.show()
```

Slide 7: Sliding Windows

Sliding windows define a range of rows around the current row for calculations. This is particularly useful for moving averages or running totals over a specific number of rows.

```python
from pyspark.sql.functions import avg

# Define a sliding window of 3 rows
slidingWindow = Window.partitionBy("department").orderBy("salary").rowsBetween(-1, 1)

# Calculate 3-row moving average
df_sliding = df.withColumn("moving_avg_3", avg("salary").over(slidingWindow))

df_sliding.show()
```

Slide 8: Multiple Window Functions

You can apply multiple window functions in a single transformation, allowing for complex analytics in a concise manner. This is useful when you need to compute several metrics simultaneously.

```python
from pyspark.sql.functions import rank, sum, avg

# Apply multiple window functions
df_multi = df.withColumn("rank", rank().over(windowSpec)) \
             .withColumn("running_total", sum("salary").over(windowSpec)) \
             .withColumn("dept_avg", avg("salary").over(Window.partitionBy("department")))

df_multi.show()
```

Slide 9: Window Functions with Aggregations

Window functions can be combined with group-by aggregations to perform multi-level analytics. This allows for comparisons between individual rows and their respective groups.

```python
from pyspark.sql.functions import sum, col

# Perform group-by aggregation
df_agg = df.groupBy("department").agg(sum("salary").alias("dept_total"))

# Join with original dataframe and apply window function
df_combined = df.join(df_agg, "department") \
                .withColumn("percentage", (col("salary") / col("dept_total")) * 100) \
                .withColumn("rank_in_dept", rank().over(Window.partitionBy("department").orderBy(col("salary").desc())))

df_combined.show()
```

Slide 10: Real-Life Example: Employee Performance Analysis

In this example, we'll use window functions to analyze employee performance across different departments, calculating rankings, running totals, and performance percentiles.

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, rank, sum, percent_rank

# Create a SparkSession
spark = SparkSession.builder.appName("EmployeePerformance").getOrCreate()

# Sample employee data
data = [
    ("Alice", "Sales", 100, 5000),
    ("Bob", "Sales", 150, 6000),
    ("Charlie", "Marketing", 200, 4500),
    ("David", "Marketing", 180, 4000),
    ("Eve", "Engineering", 300, 7000),
    ("Frank", "Engineering", 280, 6500)
]
df = spark.createDataFrame(data, ["name", "department", "tasks_completed", "project_value"])

# Define window specifications
dept_window = Window.partitionBy("department").orderBy(col("tasks_completed").desc())
overall_window = Window.orderBy(col("project_value").desc())

# Apply window functions
df_performance = df.withColumn("dept_rank", rank().over(dept_window)) \
                   .withColumn("overall_rank", rank().over(overall_window)) \
                   .withColumn("running_total_tasks", sum("tasks_completed").over(dept_window)) \
                   .withColumn("percentile", percent_rank().over(overall_window))

df_performance.show()
```

Slide 11: Real-Life Example: Product Inventory Analysis

In this example, we'll use window functions to analyze product inventory, calculating running totals, identifying top-selling products, and determining restocking needs.

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, sum, rank, lag

# Create a SparkSession
spark = SparkSession.builder.appName("ProductInventory").getOrCreate()

# Sample product inventory data
data = [
    ("ProductA", "Electronics", 100, 50),
    ("ProductB", "Electronics", 150, 30),
    ("ProductC", "Clothing", 200, 100),
    ("ProductD", "Clothing", 180, 80),
    ("ProductE", "Home", 120, 60),
    ("ProductF", "Home", 90, 40)
]
df = spark.createDataFrame(data, ["product", "category", "quantity_sold", "stock_remaining"])

# Define window specifications
category_window = Window.partitionBy("category").orderBy(col("quantity_sold").desc())
overall_window = Window.orderBy(col("quantity_sold").desc())

# Apply window functions
df_inventory = df.withColumn("category_rank", rank().over(category_window)) \
                 .withColumn("overall_rank", rank().over(overall_window)) \
                 .withColumn("running_total_sold", sum("quantity_sold").over(category_window)) \
                 .withColumn("prev_stock", lag("stock_remaining").over(category_window)) \
                 .withColumn("stock_difference", col("stock_remaining") - col("prev_stock"))

df_inventory.show()
```

Slide 12: Performance Considerations

When using window functions, consider these performance tips:

1. Minimize the number of window functions in a single query.
2. Use appropriate partitioning to reduce data shuffling.
3. Order data efficiently within partitions.
4. Consider using caching for frequently accessed windowed results.
5. Monitor query execution plans to identify potential bottlenecks.

Slide 13: Performance Considerations

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, sum

spark = SparkSession.builder.appName("PerformanceExample").getOrCreate()

# Sample large dataset
large_df = spark.range(0, 1000000).withColumn("group", (col("id") % 100).cast("integer"))

# Efficient window specification
efficient_window = Window.partitionBy("group").orderBy("id")

# Combine multiple window functions in one pass
result_df = large_df.withColumn("rank", rank().over(efficient_window)) \
                    .withColumn("running_sum", sum("id").over(efficient_window))

# Cache the result if it will be used multiple times
result_df.cache()

# Show execution plan
result_df.explain()

result_df.show(5)
```

Slide 14: Debugging and Troubleshooting

When working with window functions, common issues include:

1. Incorrect partitioning or ordering
2. Unexpected null values in window calculations
3. Performance issues with large datasets

Slide 15: Debugging and Troubleshooting

1. Use .explain() to examine the logical and physical plans
2. Check intermediate results with .show()
3. Verify window specifications and function arguments
4. Monitor Spark UI for performance metrics

Slide 16: Debugging and Troubleshooting

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, sum, when

spark = SparkSession.builder.appName("DebuggingExample").getOrCreate()

# Sample data with potential issues
data = [("A", 1, 100), ("A", 2, None), ("B", 1, 200), ("B", 2, 300)]
df = spark.createDataFrame(data, ["group", "id", "value"])

# Window specification
window_spec = Window.partitionBy("group").orderBy("id")

# Apply window functions with null handling
result_df = df.withColumn("rank", rank().over(window_spec)) \
              .withColumn("running_sum", sum(when(col("value").isNotNull(), col("value")).otherwise(0)).over(window_spec))

# Show execution plan
print("Execution Plan:")
result_df.explain()

# Show results
print("\nResults:")
result_df.show()

# Check for null values
print("\nNull Value Check:")
result_df.filter(col("value").isNull()).show()
```

Slide 17: Advanced Window Function Techniques

Advanced techniques for window functions include:

1. Using multiple window specifications in a single query
2. Combining window functions with User-Defined Functions (UDFs)
3. Applying window functions to complex data types like arrays and structs
4. Utilizing window functions in streaming contexts

Slide 18: Advanced Window Function Techniques

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, rank, sum, array, struct, explode

spark = SparkSession.builder.appName("AdvancedWindowFunctions").getOrCreate()

# Sample data with complex types
data = [
    ("A", [1, 2, 3], {"x": 10, "y": 20}),
    ("B", [4, 5, 6], {"x": 30, "y": 40}),
    ("A", [7, 8, 9], {"x": 50, "y": 60})
]
df = spark.createDataFrame(data, ["group", "array_col", "struct_col"])

# Multiple window specifications
window1 = Window.partitionBy("group")
window2 = Window.partitionBy("group").orderBy("array_col[0]")

# Applying window functions to complex types
result_df = df.withColumn("array_sum", sum(df.array_col[0]).over(window1)) \
              .withColumn("struct_x_rank", rank().over(window2.orderBy(col("struct_col.x").desc()))) \
              .withColumn("exploded_array", explode("array_col")) \
              .withColumn("running_sum_exploded", sum("exploded_array").over(window2))

result_df.show(truncate=False)
```

Slide 19: Additional Resources

For more in-depth information on Apache Spark Window Functions:

1. Apache Spark Documentation: [https://spark.apache.org/docs/latest/sql-programming-guide.html#window-functions](https://spark.apache.org/docs/latest/sql-programming-guide.html#window-functions)
2. Research Paper: "Efficient Processing of Window Functions in Analytical SQL Queries" by Georgios Giannikis et al. (2019) ArXiv URL: [https://arxiv.org/abs/1909.03642](https://arxiv.org/abs/1909.03642)
3. Apache Spark: The Definitive Guide (book) by Bill Chambers and Matei Zaharia
4. Spark Summit presentations and videos: [https://databricks.com/sparkaisummit](https://databricks.com/sparkaisummit)

These resources provide comprehensive coverage of window functions in Apache Spark, including advanced techniques, optimization strategies, and real-world use cases.

