## Spark Window Functions for Time-Series Analysis in PySpark
Slide 1: Introduction to Window Functions in PySpark

Window functions in PySpark allow us to perform calculations across a set of rows that are related to the current row. They are particularly useful for time-series analysis, enabling operations like running totals, moving averages, and rank calculations. These functions provide a powerful way to analyze data within a specific window or frame of rows.

```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Create a SparkSession
spark = SparkSession.builder.appName("WindowFunctionsDemo").getOrCreate()

# Sample data
data = [("2023-01-01", 10), ("2023-01-02", 15), ("2023-01-03", 20), 
        ("2023-01-04", 25), ("2023-01-05", 30)]
df = spark.createDataFrame(data, ["date", "value"])
df.show()
```

Slide 2: Setting Up the Window Specification

The Window specification defines the partitioning, ordering, and frame for window functions. It determines how rows are grouped and ordered within each partition.

```python
# Define a window specification
window_spec = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)

# Apply a window function (cumulative sum)
df_with_cumsum = df.withColumn("cumulative_sum", F.sum("value").over(window_spec))
df_with_cumsum.show()
```

Slide 3: Moving Average Calculation

A moving average smooths out short-term fluctuations and highlights longer-term trends in time series data. It's commonly used in various fields, including weather forecasting and stock market analysis.

```python
# Define a 3-day moving average window
moving_avg_window = Window.orderBy("date").rowsBetween(-2, 0)

# Calculate 3-day moving average
df_with_moving_avg = df.withColumn("moving_avg", F.avg("value").over(moving_avg_window))
df_with_moving_avg.show()
```

Slide 4: Ranking Functions

Ranking functions assign ranks to rows within a partition. They're useful for identifying top performers, outliers, or relative positions in a dataset.

```python
# Define a window specification for ranking
rank_window = Window.orderBy(F.desc("value"))

# Apply ranking functions
df_with_ranks = df.withColumn("rank", F.rank().over(rank_window)) \
                  .withColumn("dense_rank", F.dense_rank().over(rank_window)) \
                  .withColumn("row_number", F.row_number().over(rank_window))
df_with_ranks.show()
```

Slide 5: Lag and Lead Functions

Lag and lead functions access data from previous or subsequent rows, enabling comparisons and calculations based on neighboring values in the time series.

```python
# Define a window specification for lag and lead
lag_lead_window = Window.orderBy("date")

# Apply lag and lead functions
df_with_lag_lead = df.withColumn("prev_value", F.lag("value", 1).over(lag_lead_window)) \
                     .withColumn("next_value", F.lead("value", 1).over(lag_lead_window))
df_with_lag_lead.show()
```

Slide 6: Calculating Percentage Change

Percentage change is a common metric in time series analysis, showing the relative change between consecutive periods.

```python
# Calculate percentage change
df_with_pct_change = df_with_lag_lead.withColumn(
    "pct_change",
    F.when(F.col("prev_value").isNotNull(),
           ((F.col("value") - F.col("prev_value")) / F.col("prev_value")) * 100
    ).otherwise(None)
)
df_with_pct_change.show()
```

Slide 7: Cumulative Distribution

The cumulative distribution function helps understand the distribution of values over time, useful for analyzing trends and patterns.

```python
# Define a window for cumulative distribution
cume_dist_window = Window.orderBy("value")

# Calculate cumulative distribution
df_with_cume_dist = df.withColumn("cume_dist", F.cume_dist().over(cume_dist_window))
df_with_cume_dist.show()
```

Slide 8: Time-based Windows

Time-based windows allow for calculations over specific time intervals, crucial for analyzing periodic patterns or trends.

```python
from pyspark.sql.types import TimestampType

# Convert date to timestamp
df_ts = df.withColumn("timestamp", F.to_timestamp("date"))

# Define a 3-day time-based window
time_window = Window.orderBy("timestamp") \
    .rangeBetween(-2 * 86400, 0)  # 2 days in seconds

# Calculate 3-day time-based moving average
df_time_avg = df_ts.withColumn("time_based_avg", F.avg("value").over(time_window))
df_time_avg.show()
```

Slide 9: Partitioned Windows

Partitioned windows allow for separate calculations within different categories or groups in your data.

```python
# Add a category column to our dataset
df_cat = df.withColumn("category", F.when(F.col("value") < 20, "low").otherwise("high"))

# Define a partitioned window
part_window = Window.partitionBy("category").orderBy("date")

# Calculate rank within each category
df_part_rank = df_cat.withColumn("category_rank", F.rank().over(part_window))
df_part_rank.show()
```

Slide 10: Real-life Example: Weather Analysis

Let's analyze daily temperature data to identify trends and anomalies.

```python
# Sample weather data
weather_data = [
    ("2023-01-01", 5), ("2023-01-02", 7), ("2023-01-03", 6),
    ("2023-01-04", 8), ("2023-01-05", 10), ("2023-01-06", 12),
    ("2023-01-07", 9), ("2023-01-08", 7), ("2023-01-09", 6),
    ("2023-01-10", 5)
]
weather_df = spark.createDataFrame(weather_data, ["date", "temperature"])

# Calculate 3-day moving average and temperature change
weather_window = Window.orderBy("date").rowsBetween(-2, 0)
lag_window = Window.orderBy("date")

weather_analysis = weather_df \
    .withColumn("moving_avg_temp", F.avg("temperature").over(weather_window)) \
    .withColumn("prev_temp", F.lag("temperature").over(lag_window)) \
    .withColumn("temp_change", F.col("temperature") - F.col("prev_temp"))

weather_analysis.show()
```

Slide 11: Real-life Example: Sensor Data Analysis

Analyze sensor data from an industrial machine to detect anomalies and performance trends.

```python
# Sample sensor data
sensor_data = [
    ("2023-01-01 08:00:00", 100, 50), ("2023-01-01 09:00:00", 102, 52),
    ("2023-01-01 10:00:00", 98, 49), ("2023-01-01 11:00:00", 103, 51),
    ("2023-01-01 12:00:00", 97, 48), ("2023-01-01 13:00:00", 105, 53),
    ("2023-01-01 14:00:00", 101, 50), ("2023-01-01 15:00:00", 99, 49)
]
sensor_df = spark.createDataFrame(sensor_data, ["timestamp", "pressure", "temperature"])

# Convert timestamp to proper format
sensor_df = sensor_df.withColumn("timestamp", F.to_timestamp("timestamp"))

# Define windows for calculations
time_window = Window.orderBy("timestamp").rangeBetween(-2 * 3600, 0)  # 2-hour window
lag_window = Window.orderBy("timestamp")

# Perform analysis
sensor_analysis = sensor_df \
    .withColumn("avg_pressure", F.avg("pressure").over(time_window)) \
    .withColumn("avg_temperature", F.avg("temperature").over(time_window)) \
    .withColumn("pressure_change", F.col("pressure") - F.lag("pressure").over(lag_window)) \
    .withColumn("temp_change", F.col("temperature") - F.lag("temperature").over(lag_window))

sensor_analysis.show()
```

Slide 12: Handling Missing Data in Time Series

Missing data is common in time series analysis. Let's explore how to handle it using window functions.

```python
# Sample data with missing values
missing_data = [
    ("2023-01-01", 10), ("2023-01-02", None), ("2023-01-03", 20),
    ("2023-01-04", None), ("2023-01-05", 30), ("2023-01-06", 25)
]
missing_df = spark.createDataFrame(missing_data, ["date", "value"])

# Define window for interpolation
interp_window = Window.orderBy("date").rowsBetween(-1, 1)

# Interpolate missing values
interpolated_df = missing_df.withColumn(
    "interpolated_value",
    F.when(F.col("value").isNotNull(), F.col("value"))
     .otherwise(F.avg("value").over(interp_window))
)

interpolated_df.show()
```

Slide 13: Optimizing Window Function Performance

Window functions can be computationally expensive. Here are some tips to optimize their performance:

1. Minimize the window size when possible.
2. Use partitioning to reduce the amount of data processed in each window.
3. Consider caching intermediate results for frequently used window calculations.

```python
# Example of partitioned and cached window operation
from pyspark.sql.types import IntegerType

# Create a larger dataset
large_df = spark.range(0, 1000000).withColumn("value", (F.rand() * 100).cast(IntegerType()))

# Add a partition column
large_df = large_df.withColumn("partition", F.col("id") % 10)

# Define a partitioned window
opt_window = Window.partitionBy("partition").orderBy("id")

# Cache the dataframe
large_df.cache()

# Perform window operation
result_df = large_df.withColumn("running_sum", F.sum("value").over(opt_window))

# Show a sample of the result
result_df.sample(0.001).show()
```

Slide 14: Additional Resources

For more advanced topics and in-depth analysis of time series data using PySpark, consider exploring the following resources:

1. "Distributed Time Series Analysis with Apache Spark" by Li et al. (2020) ArXiv URL: [https://arxiv.org/abs/2003.03810](https://arxiv.org/abs/2003.03810)
2. "Scalable Time Series Classification for PySpark" by Schäfer et al. (2021) ArXiv URL: [https://arxiv.org/abs/2106.11497](https://arxiv.org/abs/2106.11497)

These papers provide insights into advanced techniques for time series analysis in distributed computing environments using Apache Spark.

