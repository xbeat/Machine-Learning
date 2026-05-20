## Memory-Efficient Duplicate Removal in Python
Slide 1: Memory-Efficient Duplicate Removal Overview

Memory efficiency in duplicate removal operations is critical when dealing with large datasets in Python. The choice of method can significantly impact both memory usage and processing speed, especially when handling millions of records.

```python
# Basic comparison of memory usage for different methods
import sys
import pandas as pd
import numpy as np

def measure_memory(obj):
    return sys.getsizeof(obj) / (1024 * 1024)  # Convert to MB

# Create sample dataset
df = pd.DataFrame({'A': np.random.randint(0, 1000, 1000000)})
print(f"Original memory usage: {measure_memory(df):.2f} MB")
```

Slide 2: Hash-Based Duplicate Removal

Hash-based deduplication leverages Python's built-in hash table implementation for efficient duplicate detection. This method provides O(n) time complexity but requires additional memory proportional to the number of unique values.

```python
def hash_based_dedup(data):
    seen = set()
    unique = []
    
    for item in data:
        if hash(str(item)) not in seen:
            seen.add(hash(str(item)))
            unique.append(item)
            
    return unique

# Example usage
data = [1, 2, 2, 3, 3, 3, 4]
result = hash_based_dedup(data)
print(f"Original: {data}\nDeduplicated: {result}")
```

Slide 3: Pandas Drop Duplicates with Index

The pandas drop\_duplicates method offers an efficient way to remove duplicates while maintaining the original index structure. This approach is particularly useful when preserving data relationships is important.

```python
import pandas as pd

def index_aware_dedup(df):
    # Keep track of memory usage
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Drop duplicates while keeping first occurrence
    df_dedup = df.drop_duplicates(keep='first')
    
    final_memory = df_dedup.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Memory usage: {initial_memory:.2f} MB -> {final_memory:.2f} MB")
    return df_dedup

# Example usage
df = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['a', 'b', 'b', 'c']})
result = index_aware_dedup(df)
```

Slide 4: Numpy-Based Deduplication

Using NumPy's unique function provides significant performance benefits for large numerical arrays. This method is particularly memory-efficient when working with homogeneous data types.

```python
import numpy as np

def numpy_dedup(array):
    # Convert to numpy array if not already
    arr = np.array(array)
    
    # Measure initial memory
    initial_mem = arr.nbytes / 1024**2
    
    # Perform deduplication
    unique_arr = np.unique(arr)
    
    # Measure final memory
    final_mem = unique_arr.nbytes / 1024**2
    
    print(f"Memory usage: {initial_mem:.2f} MB -> {final_mem:.2f} MB")
    return unique_arr

# Example with large array
data = np.random.randint(0, 1000, 1000000)
result = numpy_dedup(data)
```

Slide 5: Generator-Based Duplicate Removal

Generator-based approaches offer excellent memory efficiency by processing data in chunks without loading the entire dataset into memory at once. This method is ideal for very large datasets.

```python
def generator_dedup(iterable, chunk_size=1000):
    seen = set()
    chunk = []
    
    for item in iterable:
        if item not in seen:
            seen.add(item)
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                yield from chunk
                chunk = []
    
    if chunk:
        yield from chunk

# Example usage
large_data = range(1000000)
dedup_gen = generator_dedup(large_data)
# Process in chunks
first_chunk = list(itertools.islice(dedup_gen, 10))
print(f"First 10 unique items: {first_chunk}")
```

Slide 6: Real-World Example: Log Deduplication

Log file processing often requires efficient duplicate removal while maintaining chronological order. This implementation demonstrates a memory-efficient approach for large log files.

```python
def process_log_file(filepath, chunk_size=1000):
    seen_entries = set()
    unique_entries = []
    
    with open(filepath, 'r') as file:
        for line in file:
            # Hash the relevant parts of the log entry
            entry_hash = hash(line.strip())
            
            if entry_hash not in seen_entries:
                seen_entries.add(entry_hash)
                unique_entries.append(line)
                
                if len(unique_entries) >= chunk_size:
                    yield from unique_entries
                    unique_entries = []
    
    if unique_entries:
        yield from unique_entries

# Example usage
log_file = 'sample.log'
unique_logs = process_log_file(log_file)
```

Slide 7: Memory-Optimized DataFrame Deduplication

When working with large DataFrames, optimizing memory usage during deduplication requires careful consideration of data types and chunk processing.

```python
def optimized_df_dedup(df, chunk_size=100000):
    # Optimize dtypes before deduplication
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # High cardinality check
            df[col] = df[col].astype('category')
    
    # Process in chunks
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    result = pd.concat([chunk.drop_duplicates() for chunk in chunks]).drop_duplicates()
    
    return result

# Example with large DataFrame
df = pd.DataFrame({
    'A': np.random.choice(['x', 'y', 'z'], 1000000),
    'B': np.random.randint(0, 1000, 1000000)
})
optimized_result = optimized_df_dedup(df)
```

Slide 8: Comparison of Deduplication Methods

Different deduplication methods exhibit varying performance characteristics based on data size and available memory. This implementation provides a comprehensive comparison framework.

```python
def compare_dedup_methods(data, methods):
    results = {}
    
    for method_name, method_func in methods.items():
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024**2
        
        # Execute deduplication
        result = method_func(data.copy())
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024**2
        
        results[method_name] = {
            'time': end_time - start_time,
            'memory': end_memory - start_memory,
            'unique_count': len(result)
        }
    
    return pd.DataFrame(results).T

# Example comparison
methods = {
    'hash_based': hash_based_dedup,
    'pandas': lambda x: x.drop_duplicates(),
    'numpy': lambda x: np.unique(x)
}

data = pd.Series(np.random.randint(0, 1000, 1000000))
comparison = compare_dedup_methods(data, methods)
print(comparison)
```

Slide 9: SQL-Based Deduplication Strategy

For database-backed applications, SQL-based deduplication can be more efficient than in-memory processing, especially for very large datasets.

```python
import sqlite3
import pandas as pd

def sql_dedup(df, key_columns):
    # Create temporary SQLite database
    conn = sqlite3.connect(':memory:')
    
    # Write data to SQL
    df.to_sql('temp_table', conn, index=False)
    
    # Perform deduplication using SQL
    key_cols = ', '.join(key_columns)
    query = f"""
    SELECT *
    FROM temp_table
    WHERE rowid IN (
        SELECT MIN(rowid)
        FROM temp_table
        GROUP BY {key_cols}
    )
    """
    
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    return result

# Example usage
df = pd.DataFrame({
    'id': range(1000),
    'value': np.random.randint(0, 100, 1000)
})
deduped_df = sql_dedup(df, ['value'])
```

Slide 10: In-Place Deduplication for Large Arrays

In-place deduplication minimizes memory overhead by modifying the original data structure directly. This approach is particularly useful when working with memory constraints on large datasets.

```python
def inplace_dedup(arr):
    if not arr:
        return 0
    
    # Sort array in-place
    arr.sort()
    
    # In-place deduplication
    write_pos = 1
    for read_pos in range(1, len(arr)):
        if arr[read_pos] != arr[write_pos - 1]:
            arr[write_pos] = arr[read_pos]
            write_pos += 1
    
    # Truncate array to remove duplicates
    del arr[write_pos:]
    return len(arr)

# Example usage
data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
new_length = inplace_dedup(data)
print(f"Deduplicated array: {data[:new_length]}")
```

Slide 11: Probabilistic Deduplication Using Bloom Filters

Bloom filters provide a memory-efficient probabilistic approach to deduplication, trading perfect accuracy for significantly reduced memory usage in large-scale applications.

```python
class BloomDeduplicator:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size
        
    def _hash_functions(self, item):
        # Simple hash function implementation for demonstration
        hash_values = []
        for i in range(self.hash_count):
            hash_val = hash(str(item) + str(i)) % self.size
            hash_values.append(hash_val)
        return hash_values
    
    def add_and_check(self, item):
        hash_values = self._hash_functions(item)
        exists = all(self.bit_array[h] for h in hash_values)
        
        if not exists:
            for h in hash_values:
                self.bit_array[h] = True
                
        return exists

# Example usage
dedup = BloomDeduplicator(size=1000, hash_count=3)
data = [1, 2, 2, 3, 3, 3, 4]
unique_items = [x for x in data if not dedup.add_and_check(x)]
print(f"Approximately unique items: {unique_items}")
```

Slide 12: Real-Time Streaming Deduplication

Real-time deduplication for streaming data requires efficient handling of continuous data flows while maintaining memory constraints and processing speed.

```python
from collections import deque
import time

class StreamDeduplicator:
    def __init__(self, window_size=1000, time_window=60):
        self.seen_items = deque(maxlen=window_size)
        self.time_window = time_window
        self.timestamps = deque(maxlen=window_size)
        
    def process_item(self, item):
        current_time = time.time()
        
        # Remove old items from window
        while self.timestamps and current_time - self.timestamps[0] > self.time_window:
            self.timestamps.popleft()
            self.seen_items.popleft()
        
        # Check if item is duplicate
        if item not in self.seen_items:
            self.seen_items.append(item)
            self.timestamps.append(current_time)
            return True
        return False

# Example usage
dedup = StreamDeduplicator(window_size=5, time_window=10)
stream_data = [1, 2, 2, 3, 3, 3, 4]
for item in stream_data:
    is_unique = dedup.process_item(item)
    print(f"Item: {item}, Is Unique: {is_unique}")
```

Slide 13: Advanced Memory Profiling

Understanding memory usage patterns during deduplication is crucial for optimization. This implementation provides detailed memory profiling capabilities.

```python
import tracemalloc
import functools

def profile_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()
        
        # Get initial memory snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        result = func(*args, **kwargs)
        
        # Get final memory snapshot
        snapshot2 = tracemalloc.take_snapshot()
        execution_time = time.time() - start_time
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        print(f"\nMemory profile for {func.__name__}:")
        for stat in top_stats[:3]:
            print(f"{stat}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        tracemalloc.stop()
        return result
    return wrapper

@profile_memory
def dedup_with_profiling(data):
    return list(set(data))

# Example usage
test_data = list(range(1000000)) * 2
result = dedup_with_profiling(test_data)
```

Slide 14: Additional Resources

*   Efficient Deduplication Techniques for Modern Backup Storage Systems [https://arxiv.org/abs/1908.11470](https://arxiv.org/abs/1908.11470)
*   Memory-Efficient Deduplication for Large-Scale Data Processing [https://dl.acm.org/doi/10.1145/3183713.3196930](https://dl.acm.org/doi/10.1145/3183713.3196930)
*   Streaming Data Deduplication with Dynamic Bloom Filters [https://www.sciencedirect.com/science/article/abs/pii/S0743731520303166](https://www.sciencedirect.com/science/article/abs/pii/S0743731520303166)
*   Guidelines for searching more resources:
    *   Google Scholar: "memory efficient deduplication algorithms"
    *   ACM Digital Library: "streaming deduplication techniques"
    *   IEEE Xplore: "probabilistic deduplication methods"

