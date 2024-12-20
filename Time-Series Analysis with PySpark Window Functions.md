## Time-Series Analysis with PySpark Window Functions
Slide 1: Introduction to Window Functions in PySpark

Window functions are powerful tools for performing calculations across a set of rows that are related to the current row. In time-series analysis, they allow us to compute moving averages, running totals, and other time-based aggregations efficiently. This slideshow will explore how to leverage window functions for time-series analysis in PySpark using Python.

```python
from pyspark.sql import SparkSession
from pyspark.sql import Window
import pyspark.sql.functions as F

# Initialize SparkSession
spark = SparkSession.builder.appName("WindowFunctionsDemo").getOrCreate()
```

Slide 2: Creating a Sample Time-Series Dataset

Let's create a sample dataset representing daily temperature readings for a weather station. We'll use this dataset throughout our examples to demonstrate various window function operations.

```python
# Create sample data
data = [
    ("2023-01-01", 5.2),
    ("2023-01-02", 6.1),
    ("2023-01-03", 4.8),
    ("2023-01-04", 7.3),
    ("2023-01-05", 8.2),
    ("2023-01-06", 6.5),
    ("2023-01-07", 5.9)
]

# Create DataFrame
df = spark.createDataFrame(data, ["date", "temperature"])
df = df.withColumn("date", F.to_date("date"))
df.show()
```

Slide 3: Basic Window Function: Moving Average

One common operation in time-series analysis is calculating a moving average. Let's compute a 3-day moving average of temperature using window functions.

```python
# Define the window specification
window_spec = Window.orderBy("date").rowsBetween(-2, 0)

# Calculate 3-day moving average
df_with_ma = df.withColumn("moving_avg_temp", 
                           F.avg("temperature").over(window_spec))

df_with_ma.show()
```

Slide 4: Cumulative Sum Using Window Functions

Another useful operation is calculating cumulative sums. Let's compute the cumulative sum of temperature readings over time.

```python
# Define window specification for cumulative sum
window_spec_cum = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, 0)

# Calculate cumulative sum
df_with_cumsum = df.withColumn("cumulative_temp_sum", 
                               F.sum("temperature").over(window_spec_cum))

df_with_cumsum.show()
```

Slide 5: Ranking and Dense Ranking

Window functions allow us to rank data points within a specified window. Let's rank the temperatures from coldest to warmest and use dense ranking to handle ties.

```python
# Define window specification for ranking
window_spec_rank = Window.orderBy(F.desc("temperature"))

# Apply ranking functions
df_ranked = df.withColumn("rank", F.rank().over(window_spec_rank)) \
              .withColumn("dense_rank", F.dense_rank().over(window_spec_rank))

df_ranked.show()
```

Slide 6: Lagged and Lead Values

In time-series analysis, it's often useful to compare current values with previous (lagged) or future (lead) values. Let's calculate the temperature difference from the previous day.

```python
# Define window specification for lag
window_spec_lag = Window.orderBy("date")

# Calculate temperature difference from previous day
df_with_diff = df.withColumn("prev_day_temp", F.lag("temperature", 1).over(window_spec_lag)) \
                 .withColumn("temp_diff", F.col("temperature") - F.col("prev_day_temp"))

df_with_diff.show()
```

Slide 7: Partitioning Window Functions

When dealing with multiple time series, we can use partitioning to apply window functions separately to each series. Let's add a location column and calculate moving averages for each location.

```python
# Add location column to our dataset
data_with_location = [
    ("2023-01-01", "New York", 5.2),
    ("2023-01-02", "New York", 6.1),
    ("2023-01-03", "New York", 4.8),
    ("2023-01-01", "Los Angeles", 15.2),
    ("2023-01-02", "Los Angeles", 16.5),
    ("2023-01-03", "Los Angeles", 14.8)
]

df_multi = spark.createDataFrame(data_with_location, ["date", "location", "temperature"])
df_multi = df_multi.withColumn("date", F.to_date("date"))

# Define window specification with partitioning
window_spec_partition = Window.partitionBy("location").orderBy("date").rowsBetween(-1, 0)

# Calculate 2-day moving average for each location
df_multi_ma = df_multi.withColumn("moving_avg_temp", 
                                  F.avg("temperature").over(window_spec_partition))

df_multi_ma.show()
```

Slide 8: Real-Life Example: Analyzing Sensor Data

Let's consider a scenario where we have sensor data from multiple IoT devices measuring air quality (PM2.5 levels) over time. We'll use window functions to analyze this data.

```python
# Create sample IoT sensor data
iot_data = [
    ("Device1", "2023-01-01 00:00:00", 25.3),
    ("Device1", "2023-01-01 01:00:00", 27.1),
    ("Device1", "2023-01-01 02:00:00", 26.8),
    ("Device2", "2023-01-01 00:00:00", 30.2),
    ("Device2", "2023-01-01 01:00:00", 31.5),
    ("Device2", "2023-01-01 02:00:00", 29.8)
]

df_iot = spark.createDataFrame(iot_data, ["device_id", "timestamp", "pm25"])
df_iot = df_iot.withColumn("timestamp", F.to_timestamp("timestamp"))

# Calculate hourly change and 3-hour moving average
window_spec_iot = Window.partitionBy("device_id").orderBy("timestamp")
window_spec_ma = window_spec_iot.rangeBetween(-7200, 0)  # 2 hours in seconds

df_iot_analyzed = df_iot.withColumn("prev_hour_pm25", F.lag("pm25", 1).over(window_spec_iot)) \
                        .withColumn("hourly_change", F.col("pm25") - F.col("prev_hour_pm25")) \
                        .withColumn("3hr_moving_avg", F.avg("pm25").over(window_spec_ma))

df_iot_analyzed.show()
```

Slide 9: Time-Based Window Functions

PySpark allows us to define windows based on time intervals rather than row counts. This is particularly useful for irregular time series data.

```python
# Define a time-based window specification
window_spec_time = Window.partitionBy("device_id") \
                         .orderBy("timestamp") \
                         .rangeBetween("-1 hour", Window.currentRow)

# Calculate aggregations over the time-based window
df_iot_time_window = df_iot.withColumn("avg_last_hour", F.avg("pm25").over(window_spec_time)) \
                           .withColumn("max_last_hour", F.max("pm25").over(window_spec_time))

df_iot_time_window.show()
```

Slide 10: Handling Missing Data in Time Series

In real-world scenarios, time series data often contains gaps. Let's explore how to use window functions to handle missing data by forward-filling values.

```python
# Create sample data with missing values
data_with_gaps = [
    ("2023-01-01", 5.2),
    ("2023-01-02", None),
    ("2023-01-03", 4.8),
    ("2023-01-04", None),
    ("2023-01-05", 8.2)
]

df_gaps = spark.createDataFrame(data_with_gaps, ["date", "temperature"])
df_gaps = df_gaps.withColumn("date", F.to_date("date"))

# Define window for forward fill
window_spec_fill = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, 0)

# Perform forward fill
df_filled = df_gaps.withColumn("filled_temp", 
                               F.last("temperature", ignorenulls=True).over(window_spec_fill))

df_filled.show()
```

Slide 11: Detecting Outliers with Window Functions

Window functions can be used to detect outliers in time series data. Let's implement a simple outlier detection method using the interquartile range (IQR).

```python
# Define window for outlier detection
window_spec_outlier = Window.orderBy("date").rowsBetween(-2, 2)

# Calculate Q1, Q3, and IQR
df_with_stats = df.withColumn("Q1", F.expr("percentile_approx(temperature, 0.25)").over(window_spec_outlier)) \
                  .withColumn("Q3", F.expr("percentile_approx(temperature, 0.75)").over(window_spec_outlier)) \
                  .withColumn("IQR", F.col("Q3") - F.col("Q1"))

# Detect outliers
df_outliers = df_with_stats.withColumn("is_outlier", 
    (F.col("temperature") < (F.col("Q1") - 1.5 * F.col("IQR"))) | 
    (F.col("temperature") > (F.col("Q3") + 1.5 * F.col("IQR")))
)

df_outliers.show()
```

Slide 12: Real-Life Example: Analyzing Website Traffic

Let's analyze website traffic data to identify trends and anomalies using window functions.

```python
# Create sample website traffic data
traffic_data = [
    ("2023-01-01", 1000),
    ("2023-01-02", 1200),
    ("2023-01-03", 980),
    ("2023-01-04", 1100),
    ("2023-01-05", 1500),
    ("2023-01-06", 1300),
    ("2023-01-07", 1100)
]

df_traffic = spark.createDataFrame(traffic_data, ["date", "visitors"])
df_traffic = df_traffic.withColumn("date", F.to_date("date"))

# Calculate 7-day moving average and percent change
window_spec_traffic = Window.orderBy("date").rowsBetween(-6, 0)

df_traffic_analyzed = df_traffic.withColumn("7day_avg", F.avg("visitors").over(window_spec_traffic)) \
                                .withColumn("prev_day_visitors", F.lag("visitors").over(Window.orderBy("date"))) \
                                .withColumn("percent_change", 
                                    F.when(F.col("prev_day_visitors").isNotNull(),
                                           ((F.col("visitors") - F.col("prev_day_visitors")) / F.col("prev_day_visitors")) * 100
                                    ).otherwise(None)
                                )

df_traffic_analyzed.show()
```

Slide 13: Optimizing Window Function Performance

When working with large datasets, window functions can be computationally expensive. Here are some tips to optimize performance:

1. Minimize the window size when possible.
2. Use partitioning to reduce the amount of data processed in each window.
3. Consider using approximate functions for large datasets (e.g., approx\_count\_distinct instead of count\_distinct).

```python
# Example of using partitioning and a smaller window for better performance
window_spec_optimized = Window.partitionBy(F.dayofweek("date")) \
                              .orderBy("date") \
                              .rowsBetween(-2, 2)

df_optimized = df_traffic.withColumn("day_of_week_avg", 
                                     F.avg("visitors").over(window_spec_optimized))

df_optimized.show()
```

Slide 14: Additional Resources

For more information on window functions and time-series analysis in PySpark, consider exploring the following resources:

1. Apache Spark Documentation on Window Functions: [https://spark.apache.org/docs/latest/sql-ref-functions-window.html](https://spark.apache.org/docs/latest/sql-ref-functions-window.html)
2. "Efficient Time Series Analysis Using Apache Spark" (ArXiv paper): [https://arxiv.org/abs/2005.06115](https://arxiv.org/abs/2005.06115)
3. PySpark DataFrame API Documentation: [https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html)

These resources provide in-depth explanations and additional examples to further your understanding of window functions in PySpark for time-series analysis.

