## Spark vs MapReduce Python-Powered Distributed Data Processing
Slide 1: Introduction to Spark and MapReduce

Spark and MapReduce are powerful frameworks for distributed data processing. While MapReduce was pioneering, Spark has become more popular due to its speed and versatility. This presentation will explore both, focusing on their implementation in Python.

```python
# Simple word count example in PySpark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WordCount").getOrCreate()
text_file = spark.read.text("path/to/file.txt")
word_counts = text_file.rdd.flatMap(lambda line: line.value.split(" ")) \
                   .map(lambda word: (word, 1)) \
                   .reduceByKey(lambda a, b: a + b)
word_counts.saveAsTextFile("output")
```

Slide 2: MapReduce Basics

MapReduce is a programming model for processing large datasets in parallel across a distributed cluster. It consists of two main phases: Map and Reduce. The Map phase processes input data and generates key-value pairs, while the Reduce phase aggregates these pairs to produce the final output.

```python
# MapReduce-style word count in Python
def mapper(text):
    words = text.split()
    for word in words:
        yield (word, 1)

def reducer(key, values):
    return key, sum(values)

# Usage
text = "hello world hello python"
mapped = list(mapper(text))
reduced = {}
for word, count in mapped:
    if word in reduced:
        reduced[word] += count
    else:
        reduced[word] = count

print(reduced)
```

Slide 3: Spark Overview

Apache Spark is a unified analytics engine for large-scale data processing. It provides high-level APIs in Java, Scala, Python, and R. Spark's core concept is the Resilient Distributed Dataset (RDD), which allows for in-memory processing and faster computations compared to MapReduce.

```python
# Creating an RDD in PySpark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkExample").getOrCreate()
data = [1, 2, 3, 4, 5]
rdd = spark.sparkContext.parallelize(data)
result = rdd.map(lambda x: x * 2).collect()
print(result)
```

Slide 4: Spark DataFrames

Spark DataFrames provide a higher-level abstraction built on top of RDDs. They offer a more intuitive API for working with structured data and enable optimizations through Spark SQL's Catalyst optimizer.

```python
# Creating and querying a DataFrame in PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("DataFrameExample").getOrCreate()

# Create a DataFrame from a list of tuples
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])

# Perform operations on the DataFrame
result = df.filter(col("age") > 28).select("name").show()
```

Slide 5: Real-Life Example: Log Analysis

Log analysis is a common use case for both MapReduce and Spark. Let's consider a scenario where we need to analyze web server logs to count the number of visits per IP address.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col

spark = SparkSession.builder.appName("LogAnalysis").getOrCreate()

# Assume logs are in the format: IP_ADDRESS - - [DATE] "REQUEST" STATUS SIZE
logs = spark.read.text("path/to/access.log")

parsed_logs = logs.select(
    split(col("value"), " ").getItem(0).alias("ip"),
    split(col("value"), '"').getItem(1).alias("request")
)

ip_counts = parsed_logs.groupBy("ip").count().orderBy("count", ascending=False)
ip_counts.show()
```

Slide 6: Spark SQL

Spark SQL allows you to query structured data using SQL within your Spark applications. It provides a seamless way to mix SQL queries with programmatic data manipulations.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SparkSQLExample").getOrCreate()

# Create a temporary view of a DataFrame
df = spark.read.json("path/to/people.json")
df.createOrReplaceTempView("people")

# Run SQL query
result = spark.sql("SELECT name, age FROM people WHERE age > 20")
result.show()
```

Slide 7: Spark Streaming

Spark Streaming enables scalable, high-throughput, fault-tolerant stream processing of live data streams. It can ingest data from various sources like Kafka, Flume, or TCP sockets.

```python
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext("local[2]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

lines = ssc.socketTextStream("localhost", 9999)
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
word_counts = pairs.reduceByKey(lambda x, y: x + y)

word_counts.pprint()

ssc.start()
ssc.awaitTermination()
```

Slide 8: MapReduce Shuffling and Sorting

In MapReduce, the shuffling and sorting phase occurs between the Map and Reduce stages. It's crucial for grouping all values associated with the same key before the Reduce phase begins.

```python
import itertools

def shuffle_sort(mapped_data):
    # Sort the mapped data by key
    sorted_data = sorted(mapped_data, key=lambda x: x[0])
    
    # Group data by key
    grouped_data = itertools.groupby(sorted_data, key=lambda x: x[0])
    
    # Prepare data for reduce phase
    return [(key, [value for _, value in group]) for key, group in grouped_data]

# Example usage
mapped_data = [("apple", 1), ("banana", 1), ("apple", 1), ("cherry", 1), ("banana", 1)]
shuffled_data = shuffle_sort(mapped_data)
print(shuffled_data)
```

Slide 9: Spark RDD Operations: Transformations and Actions

Spark RDDs support two types of operations: transformations and actions. Transformations create a new RDD from an existing one, while actions return a value to the driver program after running a computation on the RDD.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("RDDOperations").getOrCreate()
sc = spark.sparkContext

# Create an RDD
rdd = sc.parallelize([1, 2, 3, 4, 5])

# Transformation: map
squared_rdd = rdd.map(lambda x: x ** 2)

# Transformation: filter
even_rdd = squared_rdd.filter(lambda x: x % 2 == 0)

# Action: collect
result = even_rdd.collect()
print(result)

# Action: reduce
sum_of_evens = even_rdd.reduce(lambda a, b: a + b)
print(sum_of_evens)
```

Slide 10: Real-Life Example: Sentiment Analysis

Sentiment analysis is another common use case for big data processing. Let's use Spark to perform a simple sentiment analysis on a dataset of movie reviews.

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# Assume we have a DataFrame 'reviews' with columns 'text' and 'sentiment'
reviews = spark.read.csv("path/to/reviews.csv", header=True, inferSchema=True)

# Create a pipeline
hashingTF = HashingTF(inputCol="text", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[hashingTF, idf, lr])

# Split the data and train the model
(trainingData, testData) = reviews.randomSplit([0.7, 0.3], seed=100)
model = pipeline.fit(trainingData)

# Make predictions
predictions = model.transform(testData)
predictions.select("text", "sentiment", "prediction").show()
```

Slide 11: Spark vs MapReduce: Performance Comparison

Spark often outperforms MapReduce due to its in-memory processing capabilities. Here's a simple benchmark to compare their performance on a word count task.

```python
import time
from pyspark.sql import SparkSession

def mapreduce_word_count(file_path):
    with open(file_path, 'r') as file:
        words = file.read().split()
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    return word_counts

def spark_word_count(file_path):
    spark = SparkSession.builder.appName("WordCount").getOrCreate()
    text_file = spark.read.text(file_path)
    counts = text_file.rdd.flatMap(lambda line: line.value.split(" ")) \
                 .map(lambda word: (word, 1)) \
                 .reduceByKey(lambda a, b: a + b)
    return counts.collect()

# Benchmark
file_path = "path/to/large_text_file.txt"

start_time = time.time()
mapreduce_result = mapreduce_word_count(file_path)
mapreduce_time = time.time() - start_time

start_time = time.time()
spark_result = spark_word_count(file_path)
spark_time = time.time() - start_time

print(f"MapReduce time: {mapreduce_time:.2f} seconds")
print(f"Spark time: {spark_time:.2f} seconds")
```

Slide 12: Fault Tolerance in Spark and MapReduce

Both Spark and MapReduce provide fault tolerance, but they use different mechanisms. MapReduce achieves fault tolerance through data replication and re-execution of failed tasks. Spark uses lineage information to recover lost data.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FaultTolerance").getOrCreate()

# Create an RDD with lineage
rdd1 = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
rdd2 = rdd1.map(lambda x: x * 2)
rdd3 = rdd2.filter(lambda x: x > 5)

# If a partition of rdd3 is lost, Spark can recompute it using the lineage
# No need for explicit code to handle failures

# Action to trigger computation
result = rdd3.collect()
print(result)
```

Slide 13: Optimizing Spark Applications

Optimizing Spark applications involves various techniques such as proper data partitioning, minimizing data shuffling, and caching frequently used data.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("SparkOptimization").getOrCreate()

# Read a large dataset
df = spark.read.csv("path/to/large_dataset.csv", header=True, inferSchema=True)

# Partition the data by a frequently used column
df_partitioned = df.repartition(100, "frequently_used_column")

# Cache the DataFrame if it's used multiple times
df_partitioned.cache()

# Use broadcast join for small datasets
small_df = spark.read.csv("path/to/small_dataset.csv", header=True, inferSchema=True)
result = df_partitioned.join(spark.broadcast(small_df), "join_column")

# Show the execution plan
result.explain()
```

Slide 14: Additional Resources

For further exploration of Spark and MapReduce, consider these peer-reviewed resources:

1. "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing" by Zaharia et al. (2012) ArXiv URL: [https://arxiv.org/abs/1203.6959](https://arxiv.org/abs/1203.6959)
2. "Spark SQL: Relational Data Processing in Spark" by Armbrust et al. (2015) ArXiv URL: [https://arxiv.org/abs/1507.08204](https://arxiv.org/abs/1507.08204)
3. "GraphX: Graph Processing in a Distributed Dataflow Framework" by Gonzalez et al. (2014) ArXiv URL: [https://arxiv.org/abs/1402.2394](https://arxiv.org/abs/1402.2394)

These papers provide in-depth insights into the architecture and algorithms behind Spark and its various components.

