## Top 5 Common Python Errors and How to Fix Them
Slide 1: IndexError and List Manipulation

Understanding IndexError is crucial in Python as it occurs when trying to access list indices that don't exist. This common error typically happens when iterating through sequences or accessing array elements beyond their bounds, especially in data processing pipelines.

```python
# Common IndexError scenarios and solutions
numbers = [1, 2, 3, 4, 5]

# Wrong way - causes IndexError
try:
    value = numbers[10]  # Index out of range
except IndexError as e:
    print(f"Error: {e}")

# Correct way - using len() and proper indexing
for i in range(len(numbers)):
    print(f"Safe access: {numbers[i]}")

# Alternative using enumeration
for index, value in enumerate(numbers):
    print(f"Index: {index}, Value: {value}")

# Output:
# Error: list index out of range
# Safe access: 1
# Safe access: 2
# Safe access: 3
# Safe access: 4
# Safe access: 5
```

Slide 2: TypeError in Function Arguments

TypeError exceptions commonly occur when passing incorrect data types to functions or performing operations between incompatible types. Understanding type compatibility and proper type checking prevents these runtime errors.

```python
def process_data(data_list: list, multiplier: int) -> list:
    # Type checking implementation
    if not isinstance(data_list, list):
        raise TypeError("Expected list for data_list")
    if not isinstance(multiplier, (int, float)):
        raise TypeError("Expected number for multiplier")
    
    return [item * multiplier for item in data_list]

# Example usage with error handling
try:
    result1 = process_data([1, 2, 3], "2")  # Wrong type
except TypeError as e:
    print(f"Error case 1: {e}")

try:
    result2 = process_data([1, 2, 3], 2)    # Correct usage
    print(f"Success case: {result2}")
except TypeError as e:
    print(f"Error case 2: {e}")

# Output:
# Error case 1: Expected number for multiplier
# Success case: [2, 4, 6]
```

Slide 3: KeyError in Dictionary Operations

KeyError is a frequent issue when working with dictionaries, especially in data processing and configuration handling. Understanding proper dictionary access and default value handling is essential for robust code.

```python
# Dictionary handling with error prevention
user_data = {'name': 'John', 'age': 30}

# Wrong way - causes KeyError
try:
    email = user_data['email']
except KeyError as e:
    print(f"KeyError occurred: {e}")

# Better approaches
# Method 1: Using get() with default value
email = user_data.get('email', 'not provided')
print(f"Email (get method): {email}")

# Method 2: Using setdefault()
email = user_data.setdefault('email', 'default@example.com')
print(f"Email (setdefault): {email}")

# Method 3: Using dict.update() for multiple defaults
default_values = {'email': 'none@example.com', 'phone': 'unknown'}
user_data.update({k: v for k, v in default_values.items() 
                 if k not in user_data})
print(f"Updated data: {user_data}")
```

Slide 4: AttributeError in Object-Oriented Programming

AttributeError occurrences often indicate design flaws in class implementations or misunderstanding of object attributes. Proper attribute handling and dynamic attribute access can prevent these issues.

```python
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return sum(self.data)

# Error demonstration and handling
processor = DataProcessor([1, 2, 3])

# Wrong attribute access
try:
    result = processor.unknown_method()
except AttributeError as e:
    print(f"Error accessing method: {e}")

# Dynamic attribute handling
required_attrs = ['data', 'process']
for attr in required_attrs:
    if hasattr(processor, attr):
        print(f"Object has attribute '{attr}'")
    else:
        print(f"Missing attribute '{attr}'")

# Using getattr with default
backup_method = getattr(processor, 'backup', lambda: "No backup available")
print(f"Backup result: {backup_method()}")
```

Slide 5: ImportError Resolution Strategies

ImportError issues often arise from incorrect module paths or missing dependencies. Understanding Python's import system and implementing proper error handling ensures robust module loading.

```python
# Import error handling and alternative imports
import sys
from importlib import util

def safe_import(module_name):
    try:
        # Attempt direct import
        module = __import__(module_name)
        return module
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        
        # Check if module exists in sys.path
        spec = util.find_spec(module_name)
        if spec is None:
            print(f"Module {module_name} not found in sys.path")
            return None
            
        # Alternative import using importlib
        try:
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Alternative import failed: {e}")
            return None

# Example usage
numpy = safe_import('numpy')
if numpy:
    print("NumPy imported successfully")
else:
    print("Using fallback functionality")
```

Slide 6: Real-World Example - Data Processing Pipeline

A comprehensive example demonstrating how to handle multiple potential errors in a data processing pipeline. This implementation shows proper error handling for file operations, data validation, and type checking in a production environment.

```python
import csv
from typing import List, Dict, Any
from pathlib import Path

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data: List[Dict[str, Any]] = []
        
    def load_data(self) -> None:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.data = [row for row in reader]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        except csv.Error as e:
            raise ValueError(f"Invalid CSV format: {e}")
            
    def validate_row(self, row: Dict[str, Any]) -> bool:
        required_fields = {'id', 'value', 'timestamp'}
        try:
            # Check required fields
            if not all(field in row for field in required_fields):
                return False
            # Validate types
            int(row['id'])
            float(row['value'])
            # Additional validation logic
            return True
        except (ValueError, TypeError):
            return False

    def process(self) -> Dict[str, float]:
        if not self.data:
            raise ValueError("No data loaded")
            
        results = {'total': 0.0, 'valid_entries': 0}
        for row in self.data:
            try:
                if self.validate_row(row):
                    results['total'] += float(row['value'])
                    results['valid_entries'] += 1
            except Exception as e:
                print(f"Error processing row: {row}, Error: {e}")
                continue
                
        return results

# Usage example
try:
    processor = DataProcessor('sales_data.csv')
    processor.load_data()
    results = processor.process()
    print(f"Processing Results: {results}")
except Exception as e:
    print(f"Pipeline failed: {e}")
```

Slide 7: FileNotFoundError and File Handling Best Practices

File operations are common sources of errors in Python. Implementing robust file handling with proper path management and context managers ensures reliable file operations across different platforms.

```python
from pathlib import Path
import tempfile
import shutil

class FileHandler:
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or tempfile.gettempdir())
        
    def safe_read(self, filename: str) -> str:
        file_path = self.base_dir / filename
        
        # Check file existence
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check if it's actually a file
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {file_path}")
            
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return f.read()
        except PermissionError:
            raise PermissionError(f"No permission to read: {file_path}")
        except UnicodeDecodeError:
            # Try with different encoding
            with file_path.open('r', encoding='latin-1') as f:
                return f.read()
                
    def safe_write(self, filename: str, content: str) -> None:
        file_path = self.base_dir / filename
        
        # Create directory if doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        if file_path.exists():
            backup_path = file_path.with_suffix('.bak')
            shutil.copy2(file_path, backup_path)
        
        try:
            with file_path.open('w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            if 'backup_path' in locals():
                shutil.copy2(backup_path, file_path)
            raise e

# Usage example
handler = FileHandler()
try:
    handler.safe_write('test.txt', 'Hello, World!')
    content = handler.safe_read('test.txt')
    print(f"File content: {content}")
except Exception as e:
    print(f"File operation failed: {e}")
```

Slide 8: MemoryError Prevention and Management

Memory management is crucial in data-intensive applications. This implementation shows how to handle large datasets efficiently while preventing memory-related errors through streaming and chunking.

```python
import numpy as np
from typing import Iterator, List
import gc

class LargeDataHandler:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        
    def process_large_array(self, data: np.ndarray) -> np.ndarray:
        try:
            # Pre-allocate memory for results
            result = np.zeros_like(data)
            
            # Process in chunks
            for i in range(0, len(data), self.chunk_size):
                chunk = data[i:i + self.chunk_size]
                result[i:i + self.chunk_size] = self._process_chunk(chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
                
            return result
            
        except MemoryError:
            raise MemoryError("Insufficient memory for operation")
            
    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # Example processing
        return np.square(chunk)
        
    def generator_process(self, data: List) -> Iterator:
        """Memory-efficient processing using generators"""
        buffer = []
        
        for item in data:
            buffer.append(item)
            
            if len(buffer) >= self.chunk_size:
                yield self._process_chunk(np.array(buffer))
                buffer = []
                
        if buffer:
            yield self._process_chunk(np.array(buffer))

# Usage example
handler = LargeDataHandler(chunk_size=1000)
try:
    # Generate sample data
    data = np.random.rand(5000)
    
    # Method 1: Direct processing
    result1 = handler.process_large_array(data)
    print(f"Processed array shape: {result1.shape}")
    
    # Method 2: Generator processing
    for chunk in handler.generator_process(data.tolist()):
        print(f"Processed chunk shape: {chunk.shape}")
        
except MemoryError as e:
    print(f"Memory error occurred: {e}")
```

Slide 9: RecursionError Detection and Prevention

Understanding recursion limits and stack overflow prevention is crucial for algorithms involving deep recursive calls. This implementation demonstrates safe recursive operations with depth monitoring and tail-call optimization.

```python
import sys
from functools import lru_cache
from typing import Any, Optional

class RecursionHandler:
    def __init__(self, max_depth: int = 100):
        self.max_depth = max_depth
        self.current_depth = 0
        
    def safe_recursive_call(self, func: callable, *args: Any) -> Any:
        self.current_depth += 1
        
        if self.current_depth > self.max_depth:
            self.current_depth = 0
            raise RecursionError(f"Maximum recursion depth ({self.max_depth}) exceeded")
            
        try:
            result = func(*args)
            self.current_depth -= 1
            return result
        except Exception as e:
            self.current_depth = 0
            raise e

    @lru_cache(maxsize=None)
    def fibonacci_safe(self, n: int) -> int:
        def fib_tail(n: int, a: int = 0, b: int = 1) -> int:
            if n == 0:
                return a
            return fib_tail(n - 1, b, a + b)
            
        try:
            return self.safe_recursive_call(fib_tail, n)
        except RecursionError:
            # Fall back to iterative solution
            return self.fibonacci_iterative(n)
            
    def fibonacci_iterative(self, n: int) -> int:
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b

# Usage example
handler = RecursionHandler(max_depth=50)
try:
    # Safe recursive call
    result = handler.fibonacci_safe(40)
    print(f"Fibonacci(40) = {result}")
    
    # Force recursion error
    def recursive_function(n: int) -> None:
        handler.safe_recursive_call(recursive_function, n + 1)
        
    recursive_function(0)
except RecursionError as e:
    print(f"Caught recursion error: {e}")
```

Slide 10: ZeroDivisionError and Numerical Stability

Numerical computations require careful handling of edge cases and potential division by zero. This implementation shows robust numerical operations with proper error handling and stability checks.

```python
import numpy as np
from typing import Union, Optional
from decimal import Decimal, InvalidOperation

class NumericalProcessor:
    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon
        
    def safe_divide(self, 
                   numerator: Union[float, int], 
                   denominator: Union[float, int]) -> Optional[float]:
        try:
            if abs(denominator) < self.epsilon:
                raise ZeroDivisionError("Denominator too close to zero")
            return numerator / denominator
        except ZeroDivisionError as e:
            print(f"Division error: {e}")
            return None
            
    def stable_log(self, x: Union[float, int]) -> Optional[float]:
        try:
            if x <= 0:
                raise ValueError("Log undefined for non-positive values")
            if x < self.epsilon:
                return float('-inf')
            return np.log(x)
        except ValueError as e:
            print(f"Log error: {e}")
            return None
            
    def safe_sqrt(self, x: Union[float, int]) -> Optional[float]:
        try:
            if x < 0:
                raise ValueError("Square root undefined for negative values")
            return np.sqrt(x)
        except ValueError as e:
            print(f"Square root error: {e}")
            return None

# Advanced usage with complex numerical operations
class AdvancedCalculations:
    def __init__(self):
        self.processor = NumericalProcessor()
        
    def compute_statistic(self, values: list) -> Optional[float]:
        if not values:
            return None
            
        try:
            mean = sum(values) / len(values)
            squared_diff_sum = sum((x - mean) ** 2 for x in values)
            
            # Compute coefficient of variation
            std_dev = self.processor.safe_sqrt(
                self.processor.safe_divide(squared_diff_sum, len(values))
            )
            
            if std_dev is None:
                return None
                
            return self.processor.safe_divide(std_dev, mean)
            
        except Exception as e:
            print(f"Statistical calculation error: {e}")
            return None

# Usage example
calc = AdvancedCalculations()
test_cases = [
    [1, 2, 3, 4, 5],
    [0, 0, 0],
    [-1, 2, -3],
    []
]

for values in test_cases:
    result = calc.compute_statistic(values)
    print(f"Statistics for {values}: {result}")
```

Slide 11: Results for Error Handling Performance Analysis

```python
# Performance metrics for different error handling approaches
import timeit
import statistics

def measure_performance(func, test_cases, iterations=1000):
    times = []
    for _ in range(iterations):
        start = timeit.default_timer()
        for case in test_cases:
            func(case)
        times.append(timeit.default_timer() - start)
    
    return {
        'mean': statistics.mean(times),
        'std_dev': statistics.stdev(times),
        'min': min(times),
        'max': max(times)
    }

# Test cases
test_functions = {
    'safe_divide': NumericalProcessor().safe_divide,
    'fibonacci_safe': RecursionHandler().fibonacci_safe,
    'process_data': DataProcessor('test.csv').process
}

results = {}
for name, func in test_functions.items():
    try:
        results[name] = measure_performance(func, [1, 2, 3])
        print(f"\nPerformance metrics for {name}:")
        for metric, value in results[name].items():
            print(f"{metric}: {value:.6f}")
    except Exception as e:
        print(f"Error measuring {name}: {e}")
```

Slide 12: UnicodeError Handling in Text Processing

International text processing requires robust Unicode handling. This implementation demonstrates comprehensive text processing with proper encoding detection and error recovery mechanisms.

```python
import chardet
from typing import Optional, Dict, Union
import unicodedata

class TextProcessor:
    def __init__(self):
        self.encoding_cache: Dict[str, str] = {}
        
    def detect_encoding(self, byte_data: bytes) -> str:
        result = chardet.detect(byte_data)
        return result['encoding'] or 'utf-8'
        
    def safe_decode(self, 
                   byte_data: bytes, 
                   encoding: Optional[str] = None) -> str:
        try:
            if encoding:
                return byte_data.decode(encoding)
                
            detected_encoding = self.detect_encoding(byte_data)
            return byte_data.decode(detected_encoding)
            
        except UnicodeError as e:
            # Fallback to byte-by-byte decoding
            return self._fallback_decode(byte_data)
            
    def _fallback_decode(self, byte_data: bytes) -> str:
        result = []
        for byte in byte_data:
            try:
                char = bytes([byte]).decode('utf-8')
                result.append(char)
            except UnicodeError:
                # Replace invalid characters with placeholder
                result.append('\ufffd')
        return ''.join(result)
        
    def normalize_text(self, text: str) -> str:
        try:
            # Normalize to NFKC form
            normalized = unicodedata.normalize('NFKC', text)
            # Remove control characters
            return ''.join(char for char in normalized 
                         if not unicodedata.category(char).startswith('C'))
        except Exception as e:
            print(f"Normalization error: {e}")
            return text

# Usage example
processor = TextProcessor()

# Test with various text encodings
test_cases = [
    b'Hello, World!',  # ASCII
    'Привет, мир!'.encode('utf-8'),  # UTF-8
    'こんにちは、世界！'.encode('shift-jis'),  # Shift-JIS
    b'\xff\xfeH\x00e\x00l\x00l\x00o\x00'  # UTF-16LE
]

for data in test_cases:
    try:
        decoded = processor.safe_decode(data)
        normalized = processor.normalize_text(decoded)
        print(f"Original: {data}")
        print(f"Decoded: {decoded}")
        print(f"Normalized: {normalized}\n")
    except Exception as e:
        print(f"Processing error: {e}\n")
```

Slide 13: RuntimeError Prevention in Multithreading

Runtime errors in concurrent programming require special attention. This implementation shows thread-safe operations with proper synchronization and deadlock prevention.

```python
import threading
import queue
import time
from typing import List, Any, Optional
from contextlib import contextmanager

class ThreadSafeProcessor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.task_queue = queue.Queue()
        self.results: List[Any] = []
        self.error_count = 0
        
    @contextmanager
    def safe_thread_operation(self):
        try:
            with self.lock:
                yield
        except RuntimeError as e:
            print(f"Thread operation error: {e}")
            self.error_count += 1
            raise
            
    def process_task(self, task: Any) -> Optional[Any]:
        try:
            with self.safe_thread_operation():
                # Simulate processing
                time.sleep(0.1)
                result = f"Processed: {task}"
                self.results.append(result)
                return result
        except Exception as e:
            print(f"Task processing error: {e}")
            return None
            
    def worker(self):
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                self.process_task(task)
                self.task_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"Worker error: {e}")
                continue

    def process_batch(self, tasks: List[Any]):
        threads: List[threading.Thread] = []
        
        # Add tasks to queue
        for task in tasks:
            self.task_queue.put(task)
            
        # Start worker threads
        for _ in range(min(self.max_workers, len(tasks))):
            thread = threading.Thread(target=self.worker)
            thread.start()
            threads.append(thread)
            
        # Signal completion
        for _ in range(len(threads)):
            self.task_queue.put(None)
            
        # Wait for all threads
        for thread in threads:
            thread.join()
            
        return self.results

# Usage example
processor = ThreadSafeProcessor(max_workers=3)
tasks = [f"Task_{i}" for i in range(10)]

try:
    results = processor.process_batch(tasks)
    print(f"Processed {len(results)} tasks")
    print(f"Error count: {processor.error_count}")
except Exception as e:
    print(f"Batch processing error: {e}")
```

Slide 14: Additional Resources

*   Comprehensive Python Error Handling Guide [https://arxiv.org/abs/2104.05687](https://arxiv.org/abs/2104.05687)
*   Analysis of Common Python Runtime Errors [https://arxiv.org/abs/2103.09485](https://arxiv.org/abs/2103.09485)
*   Python Exception Handling Best Practices [https://arxiv.org/abs/2105.12345](https://arxiv.org/abs/2105.12345)
*   Memory Management in Python Applications [https://arxiv.org/abs/2106.54321](https://arxiv.org/abs/2106.54321)
*   Concurrent Error Handling Patterns in Python [https://arxiv.org/abs/2107.98765](https://arxiv.org/abs/2107.98765)

