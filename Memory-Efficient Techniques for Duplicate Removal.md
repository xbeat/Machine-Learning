## Memory-Efficient Techniques for Duplicate Removal
Slide 1: Memory-Efficient Duplicate Removal Basics

Memory efficiency in duplicate removal operations is crucial for large datasets. The choice between different methods can significantly impact both memory usage and execution speed. Let's explore the foundational approach using Python's built-in set data structure.

```python
# Basic duplicate removal using set
def basic_dedup(data):
    # Convert list to set (memory intensive for large datasets)
    unique_data = list(set(data))
    
    # Example usage
    sample_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    result = basic_dedup(sample_data)
    print(f"Original data: {sample_data}")
    print(f"Deduplicated: {result}")
    print(f"Memory usage: {sys.getsizeof(result)} bytes")
```

Slide 2: Generator-Based Deduplication

Generators provide a memory-efficient way to process large datasets by yielding one item at a time. This approach maintains a smaller memory footprint compared to set-based methods, especially when dealing with massive collections.

```python
def generator_dedup(data):
    seen = set()
    for item in data:
        if item not in seen:
            seen.add(item)
            yield item

# Example usage
large_data = range(1000000)
dedup_gen = generator_dedup(large_data)
# Process items one at a time
for i, item in enumerate(dedup_gen):
    if i < 5:  # Show first 5 items
        print(item)
```

Slide 3: NumPy-Based Deduplication for Numerical Data

NumPy provides specialized methods for handling numerical arrays, offering both memory efficiency and computational speed advantages through vectorized operations and optimized memory layouts.

```python
import numpy as np

def numpy_dedup(data):
    # Convert to numpy array and use unique
    arr = np.array(data)
    unique_vals = np.unique(arr)
    
    # Example with timing
    import time
    start = time.time()
    sample = np.random.randint(0, 100, 1000000)
    result = np.unique(sample)
    print(f"Time taken: {time.time() - start:.4f} seconds")
    print(f"Memory usage: {result.nbytes} bytes")
    return result
```

Slide 4: Pandas DataFrame Deduplication Strategies

When working with structured data, Pandas offers sophisticated deduplication methods. The drop\_duplicates() function provides various strategies for handling duplicate rows while maintaining data integrity.

```python
import pandas as pd

def pandas_dedup(df, subset=None, keep='first'):
    # Memory usage before
    mem_before = df.memory_usage(deep=True).sum()
    
    # Perform deduplication
    df_dedup = df.drop_duplicates(subset=subset, keep=keep)
    
    # Memory usage after
    mem_after = df_dedup.memory_usage(deep=True).sum()
    
    print(f"Memory reduced by: {(mem_before - mem_after) / 1024**2:.2f} MB")
    return df_dedup
```

Slide 5: Streaming Deduplication for Large Files

When dealing with files too large to fit in memory, streaming deduplication becomes essential. This implementation reads the file in chunks and maintains a rolling hash set for efficiency.

```python
def stream_dedup(filename, chunk_size=1024):
    seen = set()
    
    def process_chunk(chunk):
        return [line for line in chunk.split('\n') 
                if line and hash(line) not in seen]
    
    with open(filename, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            unique_lines = process_chunk(chunk)
            for line in unique_lines:
                seen.add(hash(line))
                yield line
```

Slide 6: Hash-Based Deduplication with Memory Control

This advanced implementation uses a hash-based approach with a memory ceiling, automatically adjusting the hash table size to prevent memory overflow while maintaining deduplication efficiency.

```python
class MemoryControlledDedup:
    def __init__(self, max_memory_mb=100):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.hash_table = {}
        
    def dedup(self, data):
        current_memory = 0
        for item in data:
            item_hash = hash(str(item))
            mem_impact = sys.getsizeof(item_hash) + sys.getsizeof(item)
            
            if current_memory + mem_impact > self.max_memory:
                # Flush oldest entries
                self._flush_oldest(mem_impact)
            
            if item_hash not in self.hash_table:
                self.hash_table[item_hash] = item
                current_memory += mem_impact
                yield item
```

Slide 7: Real-World Example: Log File Deduplication

This implementation demonstrates practical log file deduplication, considering timestamps and message content while maintaining memory efficiency for large log files.

```python
import re
from datetime import datetime

class LogDeduplicator:
    def __init__(self, time_window_seconds=3600):
        self.seen_messages = {}
        self.time_window = time_window_seconds
    
    def process_log_line(self, line):
        # Extract timestamp and message
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if timestamp_match:
            timestamp = datetime.strptime(timestamp_match.group(1), 
                                        '%Y-%m-%d %H:%M:%S')
            message = line[timestamp_match.end():].strip()
            
            # Check for duplicates within time window
            if message in self.seen_messages:
                last_seen = self.seen_messages[message]
                if (timestamp - last_seen).seconds <= self.time_window:
                    return None
            
            self.seen_messages[message] = timestamp
            return line
        return line
```

Slide 8: Benchmark Results for Deduplication Methods

Comparing various deduplication methods reveals significant differences in memory usage and processing speed. Let's analyze the performance metrics for different data sizes.

```python
import memory_profiler
import time

def benchmark_dedup_methods(data_size=1000000):
    methods = {
        'set_based': basic_dedup,
        'generator': generator_dedup,
        'numpy': numpy_dedup,
        'memory_controlled': MemoryControlledDedup().dedup
    }
    
    results = {}
    for name, method in methods.items():
        start_time = time.time()
        memory_usage = memory_profiler.memory_usage((method, (range(data_size),)))
        end_time = time.time()
        
        results[name] = {
            'time': end_time - start_time,
            'max_memory': max(memory_usage) - min(memory_usage)
        }
    
    return results
```

Slide 9: Optimized String Deduplication

String deduplication requires special consideration due to Python's string interning behavior. This implementation optimizes memory usage for string-heavy datasets.

```python
class StringDeduplicator:
    def __init__(self):
        self.intern_dict = {}
        
    def dedup_strings(self, strings):
        result = []
        for s in strings:
            # Use hash as key to save memory
            h = hash(s)
            if h not in self.intern_dict:
                self.intern_dict[h] = sys.intern(s)
            result.append(self.intern_dict[h])
        
        # Memory usage statistics
        orig_mem = sum(sys.getsizeof(s) for s in strings)
        dedup_mem = sum(sys.getsizeof(s) for s in result)
        print(f"Memory saved: {(orig_mem - dedup_mem) / 1024:.2f} KB")
        return result
```

Slide 10: Database-Backed Deduplication

For extremely large datasets, using a database as a backing store provides scalability while maintaining reasonable memory usage. This implementation uses SQLite for persistence.

```python
import sqlite3
from contextlib import contextmanager

class DatabaseDeduplicator:
    def __init__(self, db_path=':memory:'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS unique_items
                (hash TEXT PRIMARY KEY, value TEXT)
            ''')
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def dedup(self, items):
        with self.get_connection() as conn:
            for item in items:
                item_hash = hash(str(item))
                conn.execute('''
                    INSERT OR IGNORE INTO unique_items (hash, value)
                    VALUES (?, ?)
                ''', (str(item_hash), str(item)))
            
            for row in conn.execute('SELECT value FROM unique_items'):
                yield row[0]
```

Slide 11: Probabilistic Deduplication using Bloom Filters

For scenarios where approximate deduplication is acceptable, Bloom filters offer extremely memory-efficient solution with controllable false-positive rates.

```python
from math import log, ceil
import mmh3  # MurmurHash3 for better hash distribution

class BloomDeduplicator:
    def __init__(self, expected_items, false_positive_rate=0.01):
        self.size = self._optimal_size(expected_items, false_positive_rate)
        self.hash_count = self._optimal_hash_count(expected_items)
        self.bit_array = [0] * self.size
    
    def _optimal_size(self, n, p):
        return ceil(-n * log(p) / (log(2) ** 2))
    
    def _optimal_hash_count(self, n):
        return ceil((self.size / n) * log(2))
    
    def add(self, item):
        for seed in range(self.hash_count):
            index = mmh3.hash(str(item), seed) % self.size
            self.bit_array[index] = 1
    
    def probably_seen(self, item):
        return all(self.bit_array[mmh3.hash(str(item), seed) % self.size]
                  for seed in range(self.hash_count))
```

Slide 12: Time-Window Based Deduplication

This implementation maintains a sliding window of unique items, automatically expiring old entries to control memory usage while preserving recent data.

```python
from collections import OrderedDict
from time import time

class TimeWindowDeduplicator:
    def __init__(self, window_seconds=3600):
        self.window = window_seconds
        self.items = OrderedDict()
    
    def add_item(self, item):
        current_time = time()
        
        # Remove expired items
        cutoff_time = current_time - self.window
        while self.items and next(iter(self.items.values())) < cutoff_time:
            self.items.popitem(last=False)
        
        # Add new item if not already present
        if item not in self.items:
            self.items[item] = current_time
            return True
        return False
```

Slide 13: Additional Resources

*   Building Memory-Efficient Data Structures [https://arxiv.org/abs/2106.16214](https://arxiv.org/abs/2106.16214)
*   Probabilistic Data Structures for Big Data Analytics [https://arxiv.org/abs/1902.10778](https://arxiv.org/abs/1902.10778)
*   Optimizing Python Memory Usage [https://realpython.com/python-memory-management/](https://realpython.com/python-memory-management/)
*   Efficient String Deduplication Techniques [https://medium.com/engineering/string-deduplication-algorithms](https://medium.com/engineering/string-deduplication-algorithms)
*   Advanced Data Deduplication Methods [https://www.sciencedirect.com/topics/computer-science/data-deduplication](https://www.sciencedirect.com/topics/computer-science/data-deduplication)

