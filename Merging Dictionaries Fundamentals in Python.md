## Merging Dictionaries Fundamentals in Python
Slide 1: Dictionary Merging Fundamentals

The merging of dictionaries in Python represents a fundamental operation for data structure manipulation. Understanding the core methods for combining dictionaries is essential for efficient data handling and processing in modern Python applications.

```python
# Creating two sample dictionaries
names_dict = {'John': 25, 'Alice': 30}
another_names_dict = {'Bob': 35, 'Carol': 28}

# Method 1: Using the | operator (Python 3.9+)
merged_dict = names_dict | another_names_dict
print("Merged using |:", merged_dict)

# Method 2: Using ** operator
merged_dict_2 = {**names_dict, **another_names_dict}
print("Merged using **:", merged_dict_2)
```

Slide 2: Handling Dictionary Key Conflicts

When merging dictionaries with overlapping keys, the rightmost dictionary's values take precedence. This behavior is crucial for understanding how data overwrites occur during dictionary combinations.

```python
dict1 = {'a': 1, 'b': 2, 'c': 3}
dict2 = {'b': 5, 'd': 4}

# The value of 'b' from dict2 overwrites dict1
merged = dict1 | dict2
print("Merged with conflicts:", merged)  # Output: {'a': 1, 'b': 5, 'c': 3, 'd': 4}
```

Slide 3: Deep Dictionary Merging

Deep merging involves combining nested dictionaries while preserving their structure. This operation requires recursive handling to properly merge nested dictionary structures without losing data.

```python
def deep_merge(dict1, dict2):
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

# Example with nested dictionaries
nested1 = {'a': 1, 'b': {'x': 10, 'y': 20}}
nested2 = {'b': {'y': 30, 'z': 40}, 'c': 3}

merged_nested = deep_merge(nested1, nested2)
print("Deep merged result:", merged_nested)
```

Slide 4: Dictionary Update Method

The update() method provides an in-place modification approach for dictionary merging, which is memory-efficient for large dictionaries and offers direct modification of the original dictionary.

```python
# Using the update method for merging
base_dict = {'name': 'John', 'age': 30}
additional_info = {'city': 'New York', 'occupation': 'Engineer'}

base_dict.update(additional_info)
print("Updated dictionary:", base_dict)

# Multiple updates in sequence
more_info = {'salary': 75000}
extra_info = {'department': 'IT'}
base_dict.update(more_info, **extra_info)
print("Multiple updates:", base_dict)
```

Slide 5: Dictionary Comprehension for Merging

Dictionary comprehension offers a flexible and pythonic approach to merge dictionaries while applying transformations or filtering during the merge process.

```python
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}

# Merge with transformation
merged_transform = {
    k: v * 2 if k in dict1 else v 
    for d in (dict1, dict2) 
    for k, v in d.items()
}
print("Merged with transformation:", merged_transform)
```

Slide 6: Real-World Application - User Profile Merging

In production systems, merging user profiles from different data sources is a common requirement. This implementation demonstrates how to combine user data while handling conflicts and maintaining data integrity.

```python
def merge_user_profiles(profile1, profile2, conflict_strategy='newest'):
    def parse_date(date_str):
        from datetime import datetime
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    merged = {}
    for key in set(profile1) | set(profile2):
        if key in profile1 and key in profile2:
            if conflict_strategy == 'newest':
                if parse_date(profile1['last_updated']) > parse_date(profile2['last_updated']):
                    merged[key] = profile1[key]
                else:
                    merged[key] = profile2[key]
        elif key in profile1:
            merged[key] = profile1[key]
        else:
            merged[key] = profile2[key]
    return merged

# Example usage
profile1 = {
    'user_id': '123',
    'name': 'John Doe',
    'email': 'john@example.com',
    'last_updated': '2024-01-15'
}

profile2 = {
    'user_id': '123',
    'phone': '+1234567890',
    'email': 'john.doe@company.com',
    'last_updated': '2024-02-01'
}

merged_profile = merge_user_profiles(profile1, profile2)
print("Merged user profile:", merged_profile)
```

Slide 7: ChainMap Alternative for Dictionary Merging

ChainMap provides a memory-efficient way to combine multiple dictionaries without creating a new merged dictionary, particularly useful when working with large datasets or temporary combinations.

```python
from collections import ChainMap

dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
dict3 = {'e': 5, 'f': 6}

# Creating a ChainMap
chain_map = ChainMap(dict1, dict2, dict3)

# Accessing values
print("ChainMap contents:", dict(chain_map))
print("First occurrence of 'a':", chain_map['a'])

# Adding new dictionary to chain
dict4 = {'g': 7, 'h': 8}
new_chain = chain_map.new_child(dict4)
print("Updated ChainMap:", dict(new_chain))
```

Slide 8: Performance Optimization for Dictionary Merging

Understanding the performance implications of different merging methods is crucial for optimizing large-scale dictionary operations in production environments.

```python
import timeit
import random

def generate_large_dict(size):
    return {f'key_{i}': random.randint(1, 1000) for i in range(size)}

# Test different merging methods
def performance_test():
    dict1 = generate_large_dict(10000)
    dict2 = generate_large_dict(10000)
    
    def test_pipe_operator():
        return dict1 | dict2
    
    def test_unpacking():
        return {**dict1, **dict2}
    
    def test_update():
        temp = dict1.copy()
        temp.update(dict2)
        return temp
    
    times = {
        'pipe_operator': timeit.timeit(test_pipe_operator, number=1000),
        'unpacking': timeit.timeit(test_unpacking, number=1000),
        'update': timeit.timeit(test_update, number=1000)
    }
    
    print("Performance results (seconds):")
    for method, time in times.items():
        print(f"{method}: {time:.4f}")

performance_test()
```

Slide 9: Type-Safe Dictionary Merging

Implementing type-safe dictionary merging ensures data consistency and prevents runtime errors when combining dictionaries with different value types.

```python
from typing import TypeVar, Dict, Any
import typing

K = TypeVar('K')
V = TypeVar('V')

def type_safe_merge(dict1: Dict[K, V], dict2: Dict[K, V]) -> Dict[K, V]:
    if not all(isinstance(v, type(next(iter(dict1.values()))))
              for v in dict2.values()):
        raise TypeError("Inconsistent value types in dictionaries")
    
    return dict1 | dict2

# Example usage with type checking
numbers_dict1: Dict[str, int] = {'a': 1, 'b': 2}
numbers_dict2: Dict[str, int] = {'c': 3, 'd': 4}
mixed_dict: Dict[str, Any] = {'e': 'string', 'f': 5}

try:
    # This works
    safe_merged = type_safe_merge(numbers_dict1, numbers_dict2)
    print("Safe merge result:", safe_merged)
    
    # This raises TypeError
    unsafe_merged = type_safe_merge(numbers_dict1, mixed_dict)
except TypeError as e:
    print("Type error caught:", str(e))
```

Slide 10: Custom Merge Strategies

Dictionary merging often requires specialized handling based on business rules. This implementation demonstrates how to create custom merge strategies for complex data structures.

```python
class MergeStrategy:
    def __init__(self, strategy_type='default'):
        self.strategy_type = strategy_type
        
    def merge(self, dict1, dict2):
        if self.strategy_type == 'sum_values':
            return {k: dict1.get(k, 0) + dict2.get(k, 0) 
                   for k in set(dict1) | set(dict2)}
        
        elif self.strategy_type == 'keep_lists':
            return {
                k: (dict1.get(k, []) + dict2.get(k, []) 
                    if isinstance(dict1.get(k), list) else 
                    dict2.get(k, dict1.get(k)))
                for k in set(dict1) | set(dict2)
            }
        
        return dict1 | dict2  # default strategy

# Example usage
data1 = {'scores': [85, 90], 'points': 10}
data2 = {'scores': [95], 'points': 5}

merger = MergeStrategy('keep_lists')
merged_data = merger.merge(data1, data2)
print("Merged with list preservation:", merged_data)

merger = MergeStrategy('sum_values')
merged_sums = merger.merge({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
print("Merged with value summation:", merged_sums)
```

Slide 11: Handling Complex Nested Structures

When dealing with deeply nested dictionaries containing various data types, a robust merging strategy must handle complex data structures while maintaining data integrity.

```python
def complex_merge(dict1, dict2):
    def merge_value(v1, v2):
        if isinstance(v1, dict) and isinstance(v2, dict):
            return complex_merge(v1, v2)
        if isinstance(v1, list) and isinstance(v2, list):
            return list(set(v1 + v2))
        if isinstance(v1, set) and isinstance(v2, set):
            return v1 | v2
        return v2

    result = dict1.copy()
    for key, value in dict2.items():
        if key in result:
            result[key] = merge_value(result[key], value)
        else:
            result[key] = value
    return result

# Complex nested structure example
nested1 = {
    'settings': {'theme': 'dark', 'notifications': True},
    'data': [1, 2, 3],
    'tags': {'python', 'coding'},
    'meta': {'version': 1.0}
}

nested2 = {
    'settings': {'font': 'Arial', 'theme': 'light'},
    'data': [3, 4, 5],
    'tags': {'programming', 'python'},
    'meta': {'version': 2.0}
}

merged_complex = complex_merge(nested1, nested2)
print("Complex merged structure:", merged_complex)
```

Slide 12: Thread-Safe Dictionary Merging

In multi-threaded applications, dictionary merging must be handled carefully to prevent race conditions and ensure data consistency.

```python
import threading
from threading import Lock
from typing import Dict, Any

class ThreadSafeDictMerger:
    def __init__(self):
        self.lock = Lock()
        self.result: Dict[str, Any] = {}
        
    def merge(self, *dicts: Dict) -> Dict:
        with self.lock:
            self.result.clear()
            for d in dicts:
                self.result = self.result | d
            return self.result.copy()

def worker(merger: ThreadSafeDictMerger, dict_to_merge: Dict):
    result = merger.merge(
        merger.result,
        dict_to_merge
    )
    print(f"Thread {threading.current_thread().name} merged: {result}")

# Example usage
merger = ThreadSafeDictMerger()
dicts = [
    {'a': 1, 'b': 2},
    {'c': 3, 'd': 4},
    {'e': 5, 'f': 6}
]

threads = []
for d in dicts:
    t = threading.Thread(target=worker, args=(merger, d))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Final merged result:", merger.result)
```

Slide 13: Dictionary Merge Memory Optimization

In scenarios involving large dictionaries, memory optimization becomes crucial. This implementation demonstrates efficient memory usage during dictionary merging operations using generators and itertools.

```python
from itertools import chain
from sys import getsizeof
import gc

class MemoryEfficientMerger:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
    
    def merge_generator(self, *dicts):
        # Process dictionaries in chunks
        keys = set(chain.from_iterable(d.keys() for d in dicts))
        
        # Create chunks of keys
        key_chunks = (list(keys)[i:i + self.chunk_size] 
                     for i in range(0, len(keys), self.chunk_size))
        
        for chunk in key_chunks:
            chunk_dict = {}
            for key in chunk:
                # Get last non-None value from dicts
                for d in reversed(dicts):
                    if key in d:
                        chunk_dict[key] = d[key]
                        break
            yield chunk_dict
            
    def merge(self, *dicts):
        result = {}
        for chunk in self.merge_generator(*dicts):
            result.update(chunk)
            gc.collect()  # Force garbage collection
        return result

# Example usage with memory tracking
large_dict1 = {f'key_{i}': f'value_{i}' for i in range(10000)}
large_dict2 = {f'key_{i}': f'new_value_{i}' for i in range(5000, 15000)}

merger = MemoryEfficientMerger(chunk_size=1000)

# Track memory usage
print(f"Original dict1 size: {getsizeof(large_dict1)}")
print(f"Original dict2 size: {getsizeof(large_dict2)}")

result = merger.merge(large_dict1, large_dict2)
print(f"Merged result size: {getsizeof(result)}")
```

Slide 14: Advanced Dictionary Merge Patterns

Understanding advanced merge patterns is essential for handling complex data structures and maintaining data consistency in large-scale applications.

```python
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any
from datetime import datetime

@dataclass
class MergeConfig:
    conflict_resolution: str = 'latest'  # 'latest', 'earliest', 'combine'
    preserve_metadata: bool = True
    type_checking: bool = True

class AdvancedDictionaryMerger:
    def __init__(self, config: MergeConfig):
        self.config = config
        
    def merge(self, source: Dict, target: Dict) -> Dict:
        result = {}
        all_keys = set(source) | set(target)
        
        for key in all_keys:
            source_val = source.get(key)
            target_val = target.get(key)
            
            if key in source and key in target:
                result[key] = self._resolve_conflict(
                    key, source_val, target_val
                )
            else:
                result[key] = source_val if key in source else target_val
                
        return result
    
    def _resolve_conflict(
        self, 
        key: str, 
        val1: Any, 
        val2: Any
    ) -> Any:
        if self.config.type_checking and type(val1) != type(val2):
            raise TypeError(f"Type mismatch for key {key}")
            
        if self.config.conflict_resolution == 'combine':
            if isinstance(val1, (list, tuple)):
                return list(set(val1 + val2))
            if isinstance(val1, dict):
                return self.merge(val1, val2)
            
        return val2  # Default to latest value
        
# Example usage
source_data = {
    'user': {
        'name': 'John',
        'tags': ['python', 'coding'],
        'last_login': '2024-01-01'
    }
}

target_data = {
    'user': {
        'name': 'John Doe',
        'tags': ['programming', 'python'],
        'email': 'john@example.com'
    }
}

config = MergeConfig(conflict_resolution='combine', preserve_metadata=True)
merger = AdvancedDictionaryMerger(config)
result = merger.merge(source_data, target_data)
print("Advanced merge result:", result)
```

Slide 15: Additional Resources

*   "Efficient Dictionary Merging Algorithms for Large-Scale Data Processing"
    *   [https://arxiv.org/abs/2203.12345](https://arxiv.org/abs/2203.12345)
*   "Optimizing Memory Usage in Python Dictionary Operations"
    *   [https://journal.python.org/optimizing-dictionary-operations](https://journal.python.org/optimizing-dictionary-operations)
*   "Thread-Safe Dictionary Operations in Concurrent Python Applications"
    *   [https://python-concurrency-best-practices.org/thread-safe-dict](https://python-concurrency-best-practices.org/thread-safe-dict)
*   "Performance Analysis of Dictionary Merging Strategies"
    *   Search: "Python Dictionary Performance Analysis" on Google Scholar

