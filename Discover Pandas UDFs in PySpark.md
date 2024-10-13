## Discover Pandas UDFs in PySpark
Slide 1: Introduction to Pandas UDFs in PySpark

Pandas UDFs (User Defined Functions) in PySpark combine the flexibility of Python with the distributed computing power of Spark. They allow you to apply custom Python functions to large datasets efficiently, leveraging Pandas' optimized operations.

```python
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType

# Define a Pandas UDF
@pandas_udf(IntegerType())
def square(x: pd.Series) -> pd.Series:
    return x * x

# Apply the UDF to a Spark DataFrame
df = spark.createDataFrame([(1,), (2,), (3,)], ["num"])
result = df.select(square("num").alias("squared"))
result.show()
```

Slide 2: Types of Pandas UDFs

PySpark supports three main types of Pandas UDFs: Scalar, Grouped Map, and Grouped Aggregate. Each type serves different use cases and operates on data differently.

```python
from pyspark.sql.functions import pandas_udf, PandasUDFType

# Scalar UDF
@pandas_udf(IntegerType())
def scalar_square(x: pd.Series) -> pd.Series:
    return x * x

# Grouped Map UDF
@pandas_udf("id long, val double", PandasUDFType.GROUPED_MAP)
def grouped_map_example(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(val=df['val'] * 2)

# Grouped Aggregate UDF
@pandas_udf("double", PandasUDFType.GROUPED_AGG)
def grouped_agg_example(v: pd.Series) -> float:
    return v.mean()
```

Slide 3: Scalar Pandas UDFs

Scalar Pandas UDFs operate on individual columns, transforming input Series to output Series. They're ideal for element-wise operations and can significantly improve performance over regular Python UDFs.

```python
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def celsius_to_fahrenheit(temp: pd.Series) -> pd.Series:
    return (temp * 9/5) + 32

# Create a sample DataFrame
df = spark.createDataFrame([(0,), (10,), (20,), (30,)], ["celsius"])

# Apply the UDF
result = df.withColumn("fahrenheit", celsius_to_fahrenheit(col("celsius")))
result.show()
```

Slide 4: Grouped Map Pandas UDFs

Grouped Map UDFs allow you to perform operations on grouped data, where each group is processed as a separate Pandas DataFrame. This is useful for complex transformations that require context from multiple rows.

```python
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("id", StringType()),
    StructField("value", IntegerType()),
    StructField("rank", IntegerType())
])

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def rank_within_group(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf['rank'] = pdf['value'].rank(method='dense', ascending=False)
    return pdf

# Create a sample DataFrame
df = spark.createDataFrame([
    ("A", 1), ("A", 2), ("A", 3),
    ("B", 10), ("B", 20), ("B", 30)
], ["id", "value"])

# Apply the UDF
result = df.groupBy("id").apply(rank_within_group)
result.show()
```

Slide 5: Grouped Aggregate Pandas UDFs

Grouped Aggregate UDFs are used for aggregating data within groups. They operate on a group of rows and return a single aggregated value for each group.

```python
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType(), PandasUDFType.GROUPED_AGG)
def weighted_average(v: pd.Series, w: pd.Series) -> float:
    return np.average(v, weights=w)

# Create a sample DataFrame
df = spark.createDataFrame([
    ("A", 10, 0.5), ("A", 20, 0.3), ("A", 30, 0.2),
    ("B", 15, 0.6), ("B", 25, 0.4)
], ["id", "value", "weight"])

# Apply the UDF
result = df.groupBy("id").agg(weighted_average("value", "weight").alias("weighted_avg"))
result.show()
```

Slide 6: Performance Considerations

Pandas UDFs offer significant performance improvements over regular Python UDFs by leveraging vectorized operations and minimizing data movement between JVM and Python processes.

```python
import time
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import IntegerType

# Regular Python UDF
@udf(IntegerType())
def regular_square(x):
    return x * x

# Pandas UDF
@pandas_udf(IntegerType())
def pandas_square(x: pd.Series) -> pd.Series:
    return x * x

# Create a large DataFrame
df = spark.range(1000000)

# Measure performance
start = time.time()
df.select(regular_square("id")).count()
regular_time = time.time() - start

start = time.time()
df.select(pandas_square("id")).count()
pandas_time = time.time() - start

print(f"Regular UDF time: {regular_time:.2f}s")
print(f"Pandas UDF time: {pandas_time:.2f}s")
print(f"Speedup: {regular_time / pandas_time:.2f}x")
```

Slide 7: Error Handling in Pandas UDFs

Proper error handling is crucial in Pandas UDFs to ensure robustness and provide meaningful feedback. Here's an example of how to handle errors within a Pandas UDF:

```python
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def safe_division(a: pd.Series, b: pd.Series) -> pd.Series:
    def divide(x, y):
        try:
            return str(x / y)
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return a.combine(b, divide)

# Create a sample DataFrame
df = spark.createDataFrame([(10, 2), (8, 0), (15, 3)], ["a", "b"])

# Apply the UDF
result = df.withColumn("result", safe_division(col("a"), col("b")))
result.show()
```

Slide 8: Working with Complex Data Types

Pandas UDFs can handle complex data types like arrays and structs. Here's an example of processing arrays using a Pandas UDF:

```python
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, DoubleType

@pandas_udf(ArrayType(DoubleType()))
def normalize_array(arr: pd.Series) -> pd.Series:
    def normalize(x):
        return (x - np.mean(x)) / np.std(x)
    
    return arr.apply(normalize)

# Create a sample DataFrame with array column
df = spark.createDataFrame([
    ([1.0, 2.0, 3.0],),
    ([4.0, 5.0, 6.0],),
    ([7.0, 8.0, 9.0],)
], ["values"])

# Apply the UDF
result = df.withColumn("normalized", normalize_array(col("values")))
result.show(truncate=False)
```

Slide 9: Chaining Pandas UDFs

Pandas UDFs can be chained together to perform complex operations. This allows you to break down complex logic into smaller, reusable functions:

```python
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def celsius_to_fahrenheit(temp: pd.Series) -> pd.Series:
    return (temp * 9/5) + 32

@pandas_udf(DoubleType())
def fahrenheit_to_kelvin(temp: pd.Series) -> pd.Series:
    return (temp - 32) * 5/9 + 273.15

# Create a sample DataFrame
df = spark.createDataFrame([(0,), (10,), (20,), (30,)], ["celsius"])

# Chain UDFs
result = df.withColumn("fahrenheit", celsius_to_fahrenheit(col("celsius"))) \
           .withColumn("kelvin", fahrenheit_to_kelvin(col("fahrenheit")))
result.show()
```

Slide 10: Real-Life Example: Text Processing

Let's use Pandas UDFs for a text processing task, such as calculating the sentiment score of product reviews:

```python
from textblob import TextBlob
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def sentiment_score(text: pd.Series) -> pd.Series:
    return text.apply(lambda x: TextBlob(x).sentiment.polarity)

# Create a sample DataFrame with product reviews
df = spark.createDataFrame([
    ("This product is amazing!", ),
    ("I'm disappointed with the quality.", ),
    ("It's okay, but could be better.", )
], ["review"])

# Apply the UDF
result = df.withColumn("sentiment", sentiment_score(col("review")))
result.show(truncate=False)
```

Slide 11: Real-Life Example: Time Series Analysis

Here's an example of using Pandas UDFs for time series analysis, specifically for calculating moving averages:

```python
from pyspark.sql.functions import pandas_udf, window
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def moving_average(values: pd.Series, window_size: int) -> pd.Series:
    return values.rolling(window=window_size).mean()

# Create a sample DataFrame with time series data
df = spark.createDataFrame([
    ("2023-01-01", 10),
    ("2023-01-02", 15),
    ("2023-01-03", 20),
    ("2023-01-04", 25),
    ("2023-01-05", 30)
], ["date", "value"])

# Convert string to timestamp
df = df.withColumn("date", col("date").cast("timestamp"))

# Apply the UDF
result = df.withColumn("moving_avg", moving_average("value", lit(3)))
result.show()
```

Slide 12: Debugging Pandas UDFs

Debugging Pandas UDFs can be challenging due to their distributed nature. Here are some techniques to help with debugging:

```python
import sys
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import StringType

@pandas_udf(StringType())
def debug_udf(x: pd.Series) -> pd.Series:
    def process(val):
        try:
            # Your processing logic here
            result = str(int(val) * 2)
            
            # Add debug information
            print(f"Processing value: {val}, Result: {result}", file=sys.stderr)
            
            return result
        except Exception as e:
            error_msg = f"Error processing {val}: {str(e)}"
            print(error_msg, file=sys.stderr)
            return error_msg
    
    return x.apply(process)

# Create a sample DataFrame
df = spark.createDataFrame([("1",), ("2",), ("3",), ("invalid",)], ["value"])

# Apply the UDF
result = df.withColumn("processed", debug_udf(col("value")))
result.show()
```

Slide 13: Best Practices for Pandas UDFs

When working with Pandas UDFs, consider these best practices:

1. Use appropriate UDF types based on your use case.
2. Vectorize operations within UDFs for better performance.
3. Handle errors gracefully to prevent job failures.
4. Monitor memory usage, especially when working with large datasets.
5. Use caching strategically to optimize performance.
6. Test UDFs thoroughly with various input scenarios.

```python
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def optimized_udf(x: pd.Series, y: pd.Series) -> pd.Series:
    # Vectorized operation
    result = np.log(x) + np.sqrt(y)
    
    # Handle potential errors
    result = np.where(np.isfinite(result), result, np.nan)
    
    return pd.Series(result)

# Create a sample DataFrame
df = spark.createDataFrame([(1, 4), (2, 9), (3, 16), (4, 25)], ["x", "y"])

# Apply the UDF
result = df.withColumn("result", optimized_udf(col("x"), col("y")))
result.show()
```

Slide 14: Additional Resources

For further exploration of Pandas UDFs in PySpark, consider these resources:

1. Apache Spark Documentation: [https://spark.apache.org/docs/latest/api/python/user\_guide/sql/arrow\_pandas.html](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html)
2. "Scalable Machine Learning with Apache Spark" by ArXiv: [https://arxiv.org/abs/2207.07466](https://arxiv.org/abs/2207.07466)
3. "Efficient Data Processing in Apache Spark" by ArXiv: [https://arxiv.org/abs/2106.12167](https://arxiv.org/abs/2106.12167)

These resources provide in-depth information on advanced usage, performance optimization, and best practices for Pandas UDFs in PySpark.

