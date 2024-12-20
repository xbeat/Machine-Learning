## Generator Functions and Iteration Patterns in Python
Slide 1: Introduction to Generator Functions

Generator functions in Python provide a memory-efficient way to work with large sequences of data by yielding values one at a time rather than storing them all in memory. This fundamental concept revolutionizes how we handle large datasets and infinite sequences.

```python
def number_generator(n):
    """Generates a sequence of numbers up to n"""
    current = 0
    while current < n:
        yield current
        current += 1

# Example usage
gen = number_generator(5)
print(list(gen))  # Output: [0, 1, 2, 3, 4]

# Demonstrating one-time iteration
gen = number_generator(3)
for num in gen:
    print(num)  # Output: 0, 1, 2
```

Slide 2: Generator Expression Syntax

Generator expressions provide a concise way to create generator objects using a syntax similar to list comprehensions but with parentheses instead of square brackets, making them memory-efficient for large datasets.

```python
# Traditional list comprehension vs generator expression
numbers = [x**2 for x in range(5)]  # Creates list in memory
gen = (x**2 for x in range(5))      # Creates generator object

print(f"List comprehension: {numbers}")  # Output: [0, 1, 4, 9, 16]
print(f"Generator type: {type(gen)}")    # Output: <class 'generator'>
print(f"Generator values: {list(gen)}")  # Output: [0, 1, 4, 9, 16]
```

Slide 3: Infinite Sequence Generators

Generator functions excel at creating infinite sequences without consuming infinite memory, making them perfect for mathematical sequences, random number generation, and stream processing applications.

```python
def fibonacci_generator():
    """Generates an infinite Fibonacci sequence"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using the infinite generator
fib = fibonacci_generator()
first_10 = [next(fib) for _ in range(10)]
print(f"First 10 Fibonacci numbers: {first_10}")
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

Slide 4: Generator State Management

Generator functions maintain their state between calls, allowing for complex iteration patterns and stateful processing. Understanding this behavior is crucial for advanced generator applications.

```python
def stateful_generator():
    """Demonstrates state preservation in generators"""
    state = 0
    while True:
        received = yield state
        if received is not None:
            state = received
        else:
            state += 1

# Using the stateful generator
gen = stateful_generator()
print(next(gen))       # Output: 0
print(next(gen))       # Output: 1
print(gen.send(10))    # Output: 10
print(next(gen))       # Output: 11
```

Slide 5: Generator Pipeline Implementation

Generator pipelines allow for efficient data processing by chaining multiple generators together, creating a memory-efficient data processing pipeline that transforms data incrementally.

```python
def generate_numbers(n):
    """Source generator"""
    for i in range(n):
        yield i

def square_numbers(numbers):
    """Transformation generator"""
    for num in numbers:
        yield num ** 2

def filter_even(numbers):
    """Filter generator"""
    for num in numbers:
        if num % 2 == 0:
            yield num

# Creating a pipeline
numbers = generate_numbers(5)
squared = square_numbers(numbers)
even_squares = filter_even(squared)

print(list(even_squares))  # Output: [0, 4, 16]
```

Slide 6: Coroutines and Bidirectional Communication

Coroutines extend generator functionality by enabling bidirectional communication between the generator and its caller, facilitating complex data processing patterns and state management.

```python
def coroutine_processor():
    """Demonstrates coroutine pattern for data processing"""
    total = 0
    count = 0
    average = 0
    
    while True:
        value = yield average
        if value is not None:
            count += 1
            total += value
            average = total / count

# Using the coroutine
processor = coroutine_processor()
next(processor)  # Prime the coroutine

print(processor.send(10))  # Output: 10.0
print(processor.send(20))  # Output: 15.0
print(processor.send(30))  # Output: 20.0
```

Slide 7: Exception Handling in Generators

Proper exception handling in generators requires understanding both the generator's internal exceptions and how to handle exceptions during iteration, ensuring robust and maintainable code.

```python
def exception_generator():
    """Demonstrates exception handling in generators"""
    try:
        yield 1
        raise ValueError("Demonstration error")
        yield 2  # Never reached
    except ValueError as e:
        yield f"Caught error: {e}"
    finally:
        yield "Cleanup completed"

# Using the generator with exception handling
gen = exception_generator()
try:
    for item in gen:
        print(item)
except Exception as e:
    print(f"External handler caught: {e}")

# Output:
# 1
# Caught error: Demonstration error
# Cleanup completed
```

Slide 8: Memory-Efficient Data Processing

Generator-based data processing enables handling large datasets with minimal memory footprint by processing data in chunks, making it ideal for big data applications and resource-constrained environments.

```python
def chunk_processor(filename, chunk_size=1024):
    """Process large files in memory-efficient chunks"""
    def read_chunks(file, size):
        while True:
            chunk = file.read(size)
            if not chunk:
                break
            yield chunk

    with open(filename, 'r') as file:
        total_bytes = 0
        for chunk in read_chunks(file, chunk_size):
            total_bytes += len(chunk)
            yield f"Processed chunk: {len(chunk)} bytes"
        yield f"Total processed: {total_bytes} bytes"

# Example usage (with a sample file)
def demonstrate_chunk_processing():
    with open('sample.txt', 'w') as f:
        f.write('x' * 3000)
    
    processor = chunk_processor('sample.txt')
    for result in processor:
        print(result)

# Output example:
# Processed chunk: 1024 bytes
# Processed chunk: 1024 bytes
# Processed chunk: 952 bytes
# Total processed: 3000 bytes
```

Slide 9: Custom Iterator Implementation

Understanding how to implement custom iterators provides deeper insight into Python's iteration protocol and enables creation of specialized iteration patterns for complex data structures.

```python
class CustomRange:
    """Custom iterator implementation demonstrating iteration protocol"""
    def __init__(self, start, end, step=1):
        self.start = start
        self.end = end
        self.step = step
    
    def __iter__(self):
        self.current = self.start
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        result = self.current
        self.current += self.step
        return result

# Demonstrating custom iterator usage
custom_range = CustomRange(0, 5, 1)
print("Direct iteration:")
for num in custom_range:
    print(num)

print("\nConverting to list:")
custom_range = CustomRange(0, 10, 2)
print(list(custom_range))  # Output: [0, 2, 4, 6, 8]
```

Slide 10: Real-world Application - Log File Analysis

This implementation demonstrates how generators can be used to efficiently process large log files, performing pattern matching and data extraction without loading the entire file into memory.

```python
import re
from datetime import datetime

def log_analyzer(log_file):
    """Analyzes log files using generators for memory efficiency"""
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.*)'
    
    def parse_line(line):
        match = re.match(pattern, line)
        if match:
            timestamp, level, message = match.groups()
            return {
                'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                'level': level,
                'message': message
            }
    
    def process_logs():
        with open(log_file, 'r') as f:
            for line in f:
                entry = parse_line(line.strip())
                if entry:
                    yield entry
    
    error_count = 0
    warning_count = 0
    
    for entry in process_logs():
        if entry['level'] == 'ERROR':
            error_count += 1
            yield f"ERROR found: {entry['message']}"
        elif entry['level'] == 'WARNING':
            warning_count += 1
    
    yield f"Analysis complete: {error_count} errors, {warning_count} warnings"

# Example usage
def demonstrate_log_analysis():
    # Create sample log file
    sample_logs = [
        "2024-01-01 10:00:00 [INFO] Application started",
        "2024-01-01 10:01:00 [ERROR] Database connection failed",
        "2024-01-01 10:02:00 [WARNING] High memory usage"
    ]
    
    with open('sample.log', 'w') as f:
        f.write('\n'.join(sample_logs))
    
    analyzer = log_analyzer('sample.log')
    for result in analyzer:
        print(result)

# Output example:
# ERROR found: Database connection failed
# Analysis complete: 1 errors, 1 warnings
```

Slide 11: Advanced Generator Patterns - Time Series Data Processing

Generator-based time series processing enables efficient handling of large temporal datasets, implementing sliding windows and real-time data analysis without excessive memory usage.

```python
from collections import deque
from datetime import datetime, timedelta

def time_series_processor(data_stream, window_size=5):
    """Process time series data using sliding window"""
    window = deque(maxlen=window_size)
    
    def calculate_statistics(values):
        values_list = list(values)
        return {
            'mean': sum(values_list) / len(values_list),
            'max': max(values_list),
            'min': min(values_list)
        }
    
    for timestamp, value in data_stream:
        window.append(value)
        if len(window) == window_size:
            stats = calculate_statistics(window)
            yield timestamp, stats

# Example usage
def generate_sample_data(n_points=20):
    start_time = datetime.now()
    for i in range(n_points):
        yield start_time + timedelta(minutes=i), i * 1.5

# Demonstration
data = generate_sample_data()
processor = time_series_processor(data)

for timestamp, stats in processor:
    print(f"Time: {timestamp.strftime('%H:%M')} - "
          f"Mean: {stats['mean']:.2f}, "
          f"Max: {stats['max']:.2f}, "
          f"Min: {stats['min']:.2f}")

# Example output:
# Time: 14:04 - Mean: 4.50, Max: 6.00, Min: 3.00
# Time: 14:05 - Mean: 5.40, Max: 7.50, Min: 3.00
```

Slide 12: Real-world Application - Data Stream Processing

Implementation of a real-time data stream processor using generators to handle continuous data flow, demonstrating practical usage in monitoring and analytics systems.

```python
import random
from time import sleep
from typing import Generator, Dict, Any

class DataStreamProcessor:
    """Real-time data stream processor using generators"""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.anomalies = []
    
    def sensor_data_generator(self) -> Generator[Dict[str, Any], None, None]:
        """Simulate sensor data stream"""
        while True:
            value = random.gauss(0.5, 0.2)
            timestamp = datetime.now()
            yield {
                'timestamp': timestamp,
                'value': value,
                'sensor_id': 'SENSOR_001'
            }
    
    def anomaly_detector(self, 
                        data_stream: Generator[Dict[str, Any], None, None]
                        ) -> Generator[Dict[str, Any], None, None]:
        """Detect anomalies in real-time"""
        for data in data_stream:
            if data['value'] > self.threshold:
                self.anomalies.append(data)
                data['anomaly'] = True
            else:
                data['anomaly'] = False
            yield data
    
    def process_stream(self, duration_sec: int = 10):
        """Process data stream for specified duration"""
        stream = self.sensor_data_generator()
        detector = self.anomaly_detector(stream)
        
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < duration_sec:
            data = next(detector)
            yield (f"Timestamp: {data['timestamp'].strftime('%H:%M:%S.%f')}, "
                  f"Value: {data['value']:.3f}, "
                  f"Anomaly: {data['anomaly']}")
            sleep(0.1)  # Simulate real-time processing delay

# Example usage
processor = DataStreamProcessor(threshold=0.8)
for result in processor.process_stream(duration_sec=3):
    print(result)

# Example output:
# Timestamp: 14:30:45.123456, Value: 0.623, Anomaly: False
# Timestamp: 14:30:45.223456, Value: 0.892, Anomaly: True
# Timestamp: 14:30:45.323456, Value: 0.456, Anomaly: False
```

Slide 13: Generator-based Asynchronous Processing

Modern Python applications often require asynchronous processing capabilities, and generators provide an elegant foundation for implementing async patterns and coroutines.

```python
import asyncio
from typing import AsyncGenerator

async def async_number_generator(start: int, end: int) -> AsyncGenerator[int, None]:
    """Asynchronous number generator"""
    for i in range(start, end):
        await asyncio.sleep(0.1)  # Simulate async operation
        yield i

async def async_processor(numbers: AsyncGenerator[int, None]) -> AsyncGenerator[int, None]:
    """Process numbers asynchronously"""
    async for num in numbers:
        result = await compute_intensive_task(num)
        yield result

async def compute_intensive_task(num: int) -> int:
    """Simulate computationally intensive task"""
    await asyncio.sleep(0.2)  # Simulate processing time
    return num * num

async def main():
    """Demonstrate async generator pipeline"""
    numbers = async_number_generator(0, 5)
    processor = async_processor(numbers)
    
    async for result in processor:
        print(f"Processed result: {result}")

# Run the async demonstration
if __name__ == "__main__":
    asyncio.run(main())

# Example output:
# Processed result: 0
# Processed result: 1
# Processed result: 4
# Processed result: 9
# Processed result: 16
```

Slide 14: Additional Resources

*   "Effective Python: 90 Specific Ways to Write Better Python" - Search on Python documentation
*   Generator-based Coroutines in Python - [https://www.python.org/dev/peps/pep-0342/](https://www.python.org/dev/peps/pep-0342/)
*   Advanced Generator Patterns - [https://docs.python.org/3/howto/functional.html](https://docs.python.org/3/howto/functional.html)
*   Memory Management in Python - [https://docs.python.org/3/c-api/memory.html](https://docs.python.org/3/c-api/memory.html)
*   Python Design Patterns - [https://python-patterns.guide/](https://python-patterns.guide/)
*   Understanding Asynchronous Programming in Python - [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html)

