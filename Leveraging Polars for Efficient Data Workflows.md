## Leveraging Polars for Efficient Data Workflows
Slide 1: Lazy Evaluation in Polars

Polars leverages lazy evaluation to optimize query execution plans before actually processing the data. This approach allows for better memory management and performance optimization by building a computation graph that can be analyzed and optimized before execution.

```python
import polars as pl

# Create a lazy DataFrame
df = pl.scan_csv("large_dataset.csv")  
                                      
# Chain operations without immediate execution
query = (df
    .filter(pl.col("value") > 100)
    .groupby("category")
    .agg([
        pl.col("amount").sum().alias("total"),
        pl.col("amount").mean().alias("average")
    ]))

# Examine the execution plan
print(query.describe_optimization_plan())

# Execute the optimized query
result = query.collect()
```

Slide 2: Efficient Memory Management with Streaming

Polars streaming capabilities enable processing of large datasets that exceed available RAM by reading data in chunks while maintaining high performance through vectorized operations and parallel processing.

```python
import polars as pl

# Stream large CSV file in chunks
df_stream = pl.scan_csv("huge_dataset.csv")
            .filter(pl.col("timestamp").is_between("2023-01-01", "2023-12-31"))
            .groupby_dynamic(
                "timestamp",
                every="1w",
                by="user_id"
            ).agg([
                pl.col("value").sum(),
                pl.col("value").mean()
            ])
            .collect(streaming=True)

# Process results
for batch in df_stream:
    print(f"Processed batch shape: {batch.shape}")
```

Slide 3: Advanced String Operations

Polars provides powerful string manipulation capabilities through expression contexts, allowing for complex pattern matching, extraction, and transformation operations that can be executed efficiently on large text datasets.

```python
import polars as pl

df = pl.DataFrame({
    "text": ["user123_data", "admin456_log", "guest789_info"]
})

result = df.select([
    pl.col("text").str.extract(r"(\w+)(\d+)_(\w+)", 1).alias("user_type"),
    pl.col("text").str.extract(r"(\w+)(\d+)_(\w+)", 2).alias("id"),
    pl.col("text").str.extract(r"(\w+)(\d+)_(\w+)", 3).alias("category")
])

print(result)
```

Slide 4: High-Performance Joins

Polars implements sophisticated join algorithms that outperform traditional pandas operations by utilizing parallel processing and optimized memory management strategies for combining large datasets efficiently.

```python
import polars as pl

# Create sample DataFrames
customers = pl.DataFrame({
    "customer_id": range(1000000),
    "name": ["Customer_" + str(i) for i in range(1000000)]
})

orders = pl.DataFrame({
    "order_id": range(5000000),
    "customer_id": np.random.randint(0, 1000000, 5000000),
    "amount": np.random.uniform(10, 1000, 5000000)
})

# Perform optimized join
result = customers.join(
    orders,
    on="customer_id",
    how="left"
).groupby("customer_id").agg([
    pl.col("amount").sum().alias("total_spent"),
    pl.col("order_id").count().alias("num_orders")
])
```

Slide 5: Time Series Operations

Polars excels in time series analysis through its specialized datetime functions and window operations, providing efficient tools for temporal data manipulation and analysis at scale.

```python
import polars as pl
import numpy as np

# Create time series data
dates = pl.date_range(
    start="2023-01-01",
    end="2023-12-31",
    interval="1d"
)

df = pl.DataFrame({
    "date": dates,
    "value": np.random.normal(0, 1, len(dates))
})

# Perform time-based operations
result = df.with_columns([
    pl.col("value")
        .rolling_mean(window_size="30d")
        .alias("30d_moving_avg"),
    pl.col("value")
        .rolling_std(window_size="30d")
        .alias("30d_volatility")
])
```

Slide 6: Custom Aggregations with Expressions

Polars expression system enables the creation of complex custom aggregations that combine multiple operations while maintaining high performance through vectorized computations and optimal memory usage patterns.

```python
import polars as pl
import numpy as np

df = pl.DataFrame({
    "group": np.random.choice(["A", "B", "C"], 1000000),
    "value": np.random.normal(100, 15, 1000000)
})

result = df.groupby("group").agg([
    (pl.col("value").filter(pl.col("value") > pl.col("value").mean())
        .count() / pl.col("value").count() * 100)
        .alias("pct_above_mean"),
    ((pl.col("value") - pl.col("value").mean()) / pl.col("value").std())
        .abs()
        .mean()
        .alias("mean_abs_zscore")
])
```

Slide 7: Parallel Processing with Polars

Polars maximizes computational efficiency by automatically leveraging multiple CPU cores for data processing tasks, implementing parallel execution strategies for operations like groupby, join, and aggregations.

```python
import polars as pl

# Configure thread pool size
pl.Config.set_num_threads(8)

# Create large DataFrame
df = pl.DataFrame({
    "id": range(10000000),
    "category": np.random.choice(["A", "B", "C", "D"], 10000000),
    "value": np.random.random(10000000)
})

# Parallel execution of complex operations
result = (df.lazy()
    .groupby("category")
    .agg([
        pl.col("value").quantile(0.95).alias("p95"),
        pl.col("value").filter(pl.col("value") > 0.5).mean().alias("high_value_mean"),
        pl.col("id").n_unique().alias("unique_ids")
    ])
    .collect())
```

Slide 8: Working with Missing Data

Polars provides sophisticated methods for handling missing data through efficient null representation and specialized functions that maintain high performance while dealing with incomplete datasets.

```python
import polars as pl
import numpy as np

# Create DataFrame with missing values
df = pl.DataFrame({
    "A": [1, None, 3, None, 5],
    "B": [None, 2, None, 4, 5],
    "C": ["a", None, "c", "d", None]
})

# Advanced missing value handling
result = df.select([
    pl.col("A").fill_null(strategy="forward").alias("A_forward_fill"),
    pl.col("B").fill_null(pl.col("A")).alias("B_filled_from_A"),
    pl.col("*").drop_nulls().over("C").alias("drop_null_only_in_C"),
    pl.col("*").null_count().alias("null_counts")
])
```

Slide 9: Advanced Window Functions

Polars window functions provide powerful tools for calculating rolling statistics, cumulative values, and relative metrics while maintaining exceptional performance through optimized implementations.

```python
import polars as pl
import numpy as np

df = pl.DataFrame({
    "date": pl.date_range(start="2023-01-01", end="2023-12-31", interval="1d"),
    "group": np.random.choice(["A", "B"], 365),
    "value": np.random.normal(100, 10, 365)
})

result = df.with_columns([
    pl.col("value")
        .rolling_mean(window_size=7)
        .over("group")
        .alias("7d_moving_avg_by_group"),
    pl.col("value")
        .pct_change(periods=1)
        .over("group")
        .alias("daily_returns"),
    pl.col("value")
        .rank(method="dense")
        .over("group")
        .alias("rank_within_group")
])
```

Slide 10: Real-world Example: Financial Analysis

Processing and analyzing high-frequency trading data requires efficient handling of large time-series datasets with complex calculations and grouping operations, showcasing Polars' performance advantages.

```python
import polars as pl
import numpy as np

# Simulate trading data
trades = pl.DataFrame({
    "timestamp": pl.datetime_range(
        start="2023-01-01", 
        end="2023-12-31", 
        interval="1m"
    ),
    "symbol": np.random.choice(["AAPL", "GOOGL", "MSFT"], 525600),
    "price": np.random.normal(100, 5, 525600),
    "volume": np.random.exponential(1000, 525600).astype(int)
})

# Complex financial calculations
analysis = (trades.lazy()
    .groupby_dynamic(
        "timestamp",
        every="1h",
        by="symbol"
    )
    .agg([
        pl.col("price").mean().alias("vwap"),
        (pl.col("price") * pl.col("volume")).sum() / pl.col("volume").sum()
            .alias("vwap"),
        pl.col("volume").sum().alias("total_volume"),
        (pl.col("price").max() - pl.col("price").min()) / pl.col("price").min() * 100
            .alias("price_range_pct")
    ])
    .collect())
```

Slide 11: Real-world Example: Sensor Data Processing

Processing IoT sensor data requires efficient handling of time-series data with multiple measurements and complex aggregations across different time windows and device groups.

```python
import polars as pl
import numpy as np

# Simulate sensor readings
sensor_data = pl.DataFrame({
    "timestamp": pl.date_range(
        start="2023-01-01", 
        end="2023-12-31", 
        interval="5m"
    ),
    "device_id": np.random.choice(range(100), 105120),
    "temperature": np.random.normal(25, 3, 105120),
    "humidity": np.random.normal(60, 10, 105120),
    "pressure": np.random.normal(1013, 5, 105120)
})

# Complex sensor analysis
analysis = (sensor_data.lazy()
    .groupby_dynamic(
        "timestamp",
        every="1h",
        by="device_id",
        closed="right"
    )
    .agg([
        pl.all().mean().suffix("_avg"),
        pl.all().std().suffix("_std"),
        pl.col("temperature").filter(
            pl.col("temperature") > pl.col("temperature").mean() + 2 * pl.col("temperature").std()
        ).count().alias("temperature_anomalies")
    ])
    .filter(pl.col("temperature_anomalies") > 0)
    .sort(["device_id", "timestamp"])
    .collect())
```

Slide 12: Handling Large-Scale Categorical Data

Polars efficiently processes categorical data through optimized memory layouts and specialized operations for grouping, counting, and transforming categorical variables in large datasets.

```python
import polars as pl
import numpy as np

# Create large categorical dataset
categories = ["cat_" + str(i) for i in range(1000)]
df = pl.DataFrame({
    "category_1": np.random.choice(categories, 1000000),
    "category_2": np.random.choice(categories, 1000000),
    "value": np.random.random(1000000)
})

# Efficient categorical operations
result = (df.lazy()
    .with_columns([
        pl.col("category_1").cast(pl.Categorical).alias("category_1_opt"),
        pl.col("category_2").cast(pl.Categorical).alias("category_2_opt")
    ])
    .groupby(["category_1_opt", "category_2_opt"])
    .agg([
        pl.count().alias("count"),
        pl.col("value").mean().alias("avg_value"),
        pl.col("value").std().alias("std_value")
    ])
    .sort("count", descending=True)
    .head(100)
    .collect())
```

Slide 13: Advanced Data Reshaping

Polars provides efficient methods for complex data reshaping operations, including pivot tables, melting, and dynamic column transformations while maintaining high performance.

```python
import polars as pl
import numpy as np

# Create sample data
df = pl.DataFrame({
    "date": pl.date_range(start="2023-01-01", end="2023-12-31", interval="1d"),
    "product": np.random.choice(["A", "B", "C"], 365),
    "region": np.random.choice(["North", "South", "East", "West"], 365),
    "sales": np.random.randint(100, 1000, 365),
    "returns": np.random.randint(0, 50, 365)
})

# Complex reshaping operations
pivot_result = (df.pivot(
    values=["sales", "returns"],
    index=["date"],
    columns="product",
    aggregate_function="sum"
)
.with_columns([
    pl.col("^sales_.*$").sum(axis=1).alias("total_sales"),
    pl.col("^returns_.*$").sum(axis=1).alias("total_returns")
]))

# Melt operation for long format
melted = df.melt(
    id_vars=["date", "region"],
    value_vars=["sales", "returns"],
    variable_name="metric",
    value_name="value"
)
```

Slide 14: Additional Resources

*   A Comprehensive Guide to Polars for Large-Scale Data Processing [https://www.analyticsvidhya.com/blog/guide-to-polars](https://www.analyticsvidhya.com/blog/guide-to-polars)
*   Benchmarking Polars against pandas and other frameworks [https://h2oai.github.io/db-benchmark](https://h2oai.github.io/db-benchmark)
*   Getting Started with Polars for Data Science [https://towardsdatascience.com/getting-started-with-polars-for-data-science](https://towardsdatascience.com/getting-started-with-polars-for-data-science)
*   Data Analysis with Polars: Best Practices and Performance Tips [https://medium.com/towards-data-science/polars-best-practices](https://medium.com/towards-data-science/polars-best-practices)
*   Polars Official Documentation and Tutorials [https://pola-rs.github.io/polars-book/](https://pola-rs.github.io/polars-book/)

