## Explaining Generators in Python with Fibonacci
Slide 1: Generator Factory Concept

A generator in Python is a special type of function that allows for lazy evaluation of values, creating elements only when needed. Unlike regular functions that return all values at once, generators yield values one at a time, making them memory-efficient for large sequences.

```python
def simple_generator(n):
    """Basic generator that yields numbers from 0 to n-1"""
    current = 0
    while current < n:
        yield current
        current += 1

# Example usage
gen = simple_generator(3)
print(next(gen))  # Output: 0
print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
```

Slide 2: Fibonacci Generator Implementation

The Fibonacci generator demonstrates the power of lazy evaluation by generating Fibonacci numbers on-demand. This approach is particularly efficient as it only calculates values when requested, maintaining minimal memory usage regardless of sequence length.

```python
def fibonacci_generator(n):
    """Generates n Fibonacci numbers"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Example usage
fib = fibonacci_generator(5)
sequence = [next(fib) for _ in range(5)]
print(sequence)  # Output: [0, 1, 1, 2, 3]
```

Slide 3: Generator State Management

Generators maintain their internal state between calls, remembering the last position and all local variables. This feature makes them ideal for implementing iterative algorithms where state preservation is crucial for generating the next value.

```python
def stateful_generator():
    """Demonstrates state preservation in generators"""
    state = 0
    while True:
        received = yield state
        state += 10 if received else 1

# Example usage
gen = stateful_generator()
print(next(gen))      # Output: 0
print(gen.send(True)) # Output: 10
print(gen.send(False))# Output: 11
```

Slide 4: Mathematical Series Generator

A practical application of generators for computing mathematical series demonstrates their utility in mathematical computations. This implementation shows how to generate terms of a power series efficiently.

```python
def power_series_generator(x, terms):
    """Generates terms of the power series for e^x"""
    n = 0
    factorial = 1
    while n < terms:
        term = (x ** n) / factorial
        yield term
        n += 1
        factorial *= (n + 1)

# Calculate first 5 terms of e^2
series = power_series_generator(2, 5)
partial_sum = sum([next(series) for _ in range(5)])
print(f"Partial sum: {partial_sum}")  # Output approximates e^2
```

Slide 5: Generator Expression Optimization

Generator expressions provide a more concise syntax for creating generators, offering memory efficiency compared to list comprehensions. This is particularly useful when working with large datasets or infinite sequences.

```python
# Memory-efficient generator expression
gen_exp = (x**2 for x in range(10**6))
print(next(gen_exp))  # Output: 0
print(next(gen_exp))  # Output: 1

# Memory comparison
import sys
list_comp = [x**2 for x in range(10**6)]
gen_exp = (x**2 for x in range(10**6))
print(f"List size: {sys.getsizeof(list_comp)}")
print(f"Generator size: {sys.getsizeof(gen_exp)}")
```

Slide 6: Infinite Sequence Generator

Generators excel at handling infinite sequences, as they only generate values when requested. This implementation shows how to create an infinite sequence generator while maintaining constant memory usage.

```python
def infinite_primes():
    """Generates an infinite sequence of prime numbers"""
    def is_prime(n):
        return all(n % i != 0 for i in range(2, int(n**0.5) + 1))
    
    n = 2
    while True:
        if is_prime(n):
            yield n
        n += 1

# Example usage
primes = infinite_primes()
first_five = [next(primes) for _ in range(5)]
print(first_five)  # Output: [2, 3, 5, 7, 11]
```

Slide 7: Generator Pipeline Construction

Generator pipelines allow for the creation of complex data processing workflows where each generator transforms data from the previous one. This pattern enables memory-efficient processing of large datasets through composition.

```python
def read_data():
    """Simulates reading large data"""
    for i in range(1000):
        yield i

def filter_even(numbers):
    """Filters even numbers"""
    for num in numbers:
        if num % 2 == 0:
            yield num

def multiply_by_three(numbers):
    """Multiplies each number by 3"""
    for num in numbers:
        yield num * 3

# Pipeline construction
data = read_data()
filtered = filter_even(data)
result = multiply_by_three(filtered)

# Process first 5 results
print([next(result) for _ in range(5)])  # Output: [0, 6, 12, 18, 24]
```

Slide 8: Real-world Example - Data Streaming

This practical implementation demonstrates using generators for processing large datasets in chunks, simulating real-time data streaming scenarios common in data engineering applications.

```python
def stream_data_processor(chunk_size=1000):
    """Simulates processing streaming data in chunks"""
    def generate_data():
        for i in range(10000):
            yield {'id': i, 'value': i * 2}
    
    def process_chunk(chunk):
        return sum(item['value'] for item in chunk)
    
    current_chunk = []
    for item in generate_data():
        current_chunk.append(item)
        if len(current_chunk) == chunk_size:
            yield process_chunk(current_chunk)
            current_chunk = []
            
    if current_chunk:  # Process remaining items
        yield process_chunk(current_chunk)

# Example usage
processor = stream_data_processor(chunk_size=2500)
chunk_sums = list(processor)
print(f"Number of chunks processed: {len(chunk_sums)}")
print(f"Sum of all chunks: {sum(chunk_sums)}")
```

Slide 9: Generator-based Time Series Analysis

A practical implementation showing how generators can be used for time series analysis, demonstrating moving average calculation with minimal memory footprint.

```python
from collections import deque
from datetime import datetime, timedelta

def moving_average_generator(window_size):
    """Generates moving averages for streaming time series data"""
    window = deque(maxlen=window_size)
    
    while True:
        new_value = yield None if len(window) < window_size else sum(window)/window_size
        window.append(new_value)

# Example usage with time series data
def simulate_time_series():
    start_time = datetime.now()
    for i in range(100):
        yield (start_time + timedelta(minutes=i), i + (i % 5))

# Process time series
ma_gen = moving_average_generator(5)
next(ma_gen)  # Initialize generator

for timestamp, value in simulate_time_series():
    ma = ma_gen.send(value)
    if ma is not None:
        print(f"{timestamp}: MA = {ma:.2f}")
```

Slide 10: Memory-Efficient Data Processing

This implementation showcases how generators can be used to process large datasets efficiently, demonstrating memory usage optimization for big data scenarios.

```python
def process_large_dataset(filename, chunk_size=1000):
    """Process large datasets in memory-efficient chunks"""
    def read_chunks():
        with open(filename, 'r') as f:
            chunk = []
            for line in f:
                chunk.append(float(line.strip()))
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk
    
    def calculate_statistics(chunk):
        return {
            'count': len(chunk),
            'mean': sum(chunk) / len(chunk),
            'max': max(chunk),
            'min': min(chunk)
        }
    
    for chunk in read_chunks():
        yield calculate_statistics(chunk)

# Example usage with file creation
import random

# Create sample data file
with open('large_dataset.txt', 'w') as f:
    for _ in range(10000):
        f.write(f"{random.random()}\n")

# Process data
stats_generator = process_large_dataset('large_dataset.txt')
chunk_stats = list(stats_generator)
print(f"Processed {len(chunk_stats)} chunks")
```

Slide 11: Generator-based Custom Iterator Pattern

Generators provide an elegant way to implement custom iterators, simplifying the implementation of complex iteration patterns while maintaining clean, readable code. This example demonstrates a custom range iterator with step control.

```python
def custom_range_iterator(start, end, step=1):
    """Custom range iterator with dynamic step control"""
    current = start
    step_size = step
    
    while current < end:
        step_control = yield current
        if step_control is not None:
            step_size = step_control
        current += step_size

# Example usage
def demonstrate_custom_range():
    iterator = custom_range_iterator(0, 10)
    next(iterator)  # Initialize
    
    print(iterator.send(None))    # Output: 0
    print(iterator.send(2))       # Change step to 2
    print(iterator.send(None))    # Continue with step 2
    print(iterator.send(0.5))     # Change step to 0.5

demonstrate_custom_range()
```

Slide 12: Real-time Data Processing with Generators

This implementation demonstrates how generators can be used for real-time data processing, showing a practical application in monitoring and analyzing streaming data with minimal latency.

```python
import time
from collections import deque

def sensor_data_generator(sampling_rate=0.1):
    """Simulates continuous sensor data stream"""
    while True:
        yield {
            'timestamp': time.time(),
            'temperature': 20 + (time.time() % 5),
            'humidity': 50 + (time.time() % 10)
        }
        time.sleep(sampling_rate)

def analyze_sensor_data(window_size=5):
    """Analyzes streaming sensor data"""
    temp_window = deque(maxlen=window_size)
    humid_window = deque(maxlen=window_size)
    
    sensor = sensor_data_generator()
    
    while True:
        data = next(sensor)
        temp_window.append(data['temperature'])
        humid_window.append(data['humidity'])
        
        if len(temp_window) == window_size:
            yield {
                'avg_temp': sum(temp_window) / window_size,
                'avg_humidity': sum(humid_window) / window_size,
                'timestamp': data['timestamp']
            }

# Example usage
analyzer = analyze_sensor_data()
for _ in range(3):  # Analyze 3 windows of data
    analysis = next(analyzer)
    print(f"Time: {analysis['timestamp']:.2f}, "
          f"Avg Temp: {analysis['avg_temp']:.2f}, "
          f"Avg Humidity: {analysis['avg_humidity']:.2f}")
```

Slide 13: Advanced Mathematical Series Generator

A sophisticated implementation of a mathematical series generator that can handle various types of series with configurable parameters and convergence criteria.

```python
def math_series_generator(series_type='geometric', first_term=1, ratio=0.5, tolerance=1e-10):
    """
    Generates terms of various mathematical series
    Supports: geometric, arithmetic, and harmonic series
    """
    current_term = first_term
    position = 1
    
    while abs(current_term) > tolerance:
        yield current_term
        
        if series_type == 'geometric':
            current_term *= ratio
        elif series_type == 'arithmetic':
            current_term += ratio
        elif series_type == 'harmonic':
            position += 1
            current_term = first_term / position

# Example usage
def demonstrate_series():
    # Geometric series
    geometric = math_series_generator(series_type='geometric')
    geometric_terms = [next(geometric) for _ in range(5)]
    
    # Harmonic series
    harmonic = math_series_generator(series_type='harmonic')
    harmonic_terms = [next(harmonic) for _ in range(5)]
    
    print(f"Geometric series: {geometric_terms}")
    print(f"Harmonic series: {harmonic_terms}")

demonstrate_series()
```

Slide 14: Additional Resources

*   "Generator Patterns in Python" - [https://www.python.org/dev/peps/pep-0255/](https://www.python.org/dev/peps/pep-0255/)
*   "Understanding Python Generators" - [https://docs.python.org/3/howto/functional.html](https://docs.python.org/3/howto/functional.html)
*   "Memory Management in Python" - [https://docs.python.org/3/c-api/memory.html](https://docs.python.org/3/c-api/memory.html)
*   "Efficient Data Processing in Python" - [https://realpython.com/introduction-to-python-generators/](https://realpython.com/introduction-to-python-generators/)
*   "Advanced Iterator Patterns" - [https://www.python.org/dev/peps/pep-0289/](https://www.python.org/dev/peps/pep-0289/)

