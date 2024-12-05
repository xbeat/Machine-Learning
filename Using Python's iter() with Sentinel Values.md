## Using Python's iter() with Sentinel Values
Slide 1: Understanding Python's iter() with Sentinel Values

The iter() function in Python can accept two arguments: a callable object and a sentinel value. When called this way, iter() creates an iterator that repeatedly calls the function until it returns the sentinel value, making it powerful for creating custom iteration patterns.

```python
# Creating an iterator that yields random numbers until 100 is generated
import random

def random_until_100():
    iterator = iter(lambda: random.randint(1, 100), 100)
    return list(iterator)

# Example usage
result = random_until_100()
print(f"Generated numbers until 100: {result}")
```

Slide 2: File Reading with iter() and Sentinel

The iter() function with sentinel values provides an elegant way to read files until a specific condition is met, like encountering an empty string or specific marker, avoiding explicit loop constructs and making code more pythonic.

```python
def read_until_empty(file_obj):
    # Create iterator that reads lines until empty string
    lines = iter(file_obj.readline, '')
    return list(lines)

# Example usage
with open('sample.txt', 'w') as f:
    f.write('Line 1\nLine 2\nLine 3')

with open('sample.txt', 'r') as f:
    content = read_until_empty(f)
    print(f"File contents: {content}")
```

Slide 3: Custom Number Generator with iter()

Creating a custom number sequence generator using iter() demonstrates how to implement mathematical sequences without storing all values in memory, making it memory-efficient for large or infinite sequences.

```python
def fibonacci_generator():
    a, b = 0, 1
    def get_next():
        nonlocal a, b
        current = a
        a, b = b, a + b
        return current
    
    # Create iterator that never reaches sentinel (-1)
    return iter(get_next, -1)

# Example usage
fib = fibonacci_generator()
first_10 = [next(fib) for _ in range(10)]
print(f"First 10 Fibonacci numbers: {first_10}")
```

Slide 4: Network Data Streaming with iter()

Using iter() for network data streaming demonstrates its practical application in handling continuous data streams, where the sentinel value acts as a termination signal or timeout indicator.

```python
import socket
import time

def simulate_network_stream():
    def receive_data():
        # Simulate network data reception
        time.sleep(0.1)
        return round(time.time() % 10, 2)
    
    # Create iterator that stops when value is 0.0
    return iter(receive_data, 0.0)

# Example usage
stream = simulate_network_stream()
data = []
for value in stream:
    data.append(value)
    if len(data) >= 10:
        break
print(f"Received data stream: {data}")
```

Slide 5: Database Cursor Implementation

Implementing a database-like cursor using iter() shows how to create memory-efficient data access patterns for large datasets, yielding one record at a time instead of loading everything into memory.

```python
class DatabaseSimulator:
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def fetch_next(self):
        if self.index >= len(self.data):
            return None
        result = self.data[self.index]
        self.index += 1
        return result

    def cursor(self):
        return iter(self.fetch_next, None)

# Example usage
db = DatabaseSimulator([
    {'id': 1, 'name': 'John'},
    {'id': 2, 'name': 'Jane'},
    {'id': 3, 'name': 'Bob'}
])

cursor = db.cursor()
for record in cursor:
    print(f"Processing record: {record}")
```

Slide 6: Time-based Data Collection

Implementing a time-based data collection system using iter() demonstrates how to create controlled sampling intervals with automatic termination based on specific conditions.

```python
import time
from datetime import datetime

def time_series_collector(duration_seconds):
    start_time = time.time()
    
    def collect_sample():
        if time.time() - start_time > duration_seconds:
            return None
        return (datetime.now(), time.time() - start_time)
    
    return iter(collect_sample, None)

# Example usage
collector = time_series_collector(2)
samples = list(collector)
print(f"Collected {len(samples)} samples over 2 seconds:")
for timestamp, elapsed in samples[:5]:
    print(f"Time: {timestamp}, Elapsed: {elapsed:.2f}s")
```

Slide 7: Custom Token Parser

Creating a custom token parser using iter() demonstrates its application in text processing and compiler design, where the sentinel value marks the end of input or specific token boundaries.

```python
def tokenizer(text):
    position = 0
    
    def next_token():
        nonlocal position
        if position >= len(text):
            return None
            
        # Skip whitespace
        while position < len(text) and text[position].isspace():
            position += 1
            
        if position >= len(text):
            return None
            
        # Collect token
        start = position
        while position < len(text) and not text[position].isspace():
            position += 1
            
        return text[start:position]
    
    return iter(next_token, None)

# Example usage
text = "def calculate_sum(a, b):"
parser = tokenizer(text)
tokens = list(parser)
print(f"Parsed tokens: {tokens}")
```

Slide 8: Mathematical Sequence Generator

The iter() function can be used to generate mathematical sequences based on recurrence relations. This implementation shows how to create a generator for sequences defined by mathematical formulas, with automatic termination conditions.

```python
def sequence_generator(formula, initial_value, max_terms=None):
    current = initial_value
    count = 0
    
    def next_term():
        nonlocal current, count
        if max_terms and count >= max_terms:
            return None
        result = current
        current = formula(current)
        count += 1
        return result
    
    return iter(next_term, None)

# Example usage: Generate powers of 2
powers_of_2 = sequence_generator(lambda x: x * 2, 1, 10)
sequence = list(powers_of_2)
print(f"Powers of 2: {sequence}")

# Example usage: Generate collatz sequence
def collatz(n):
    return n // 2 if n % 2 == 0 else 3 * n + 1
    
collatz_seq = sequence_generator(collatz, 27, 20)
sequence = list(collatz_seq)
print(f"Collatz sequence starting at 27: {sequence}")
```

Slide 9: Event Stream Processing

Implementing an event stream processor using iter() demonstrates how to handle asynchronous events with conditional termination, useful in real-time monitoring and data processing systems.

```python
import random
from datetime import datetime, timedelta

class EventStreamProcessor:
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.start_time = datetime.now()
        
    def get_event(self):
        if datetime.now() - self.start_time > timedelta(seconds=5):
            return None
            
        event = {
            'timestamp': datetime.now(),
            'value': random.random(),
            'type': random.choice(['A', 'B', 'C'])
        }
        
        return event if event['value'] < self.threshold else None

# Example usage
processor = EventStreamProcessor()
event_stream = iter(processor.get_event, None)

events = []
for event in event_stream:
    events.append(event)
    print(f"Processed event: {event}")
```

Slide 10: Custom Range Implementation

A custom range implementation using iter() showcases how to create numeric sequences with complex step patterns that standard range() cannot handle, such as geometric progressions or custom increments.

```python
def custom_range(start, end, step_func):
    current = start
    
    def get_next():
        nonlocal current
        if current >= end:
            return None
        result = current
        current = step_func(current)
        return result
    
    return iter(get_next, None)

# Example 1: Geometric progression
geometric = custom_range(1, 100, lambda x: x * 2)
geom_sequence = list(geometric)
print(f"Geometric sequence: {geom_sequence}")

# Example 2: Custom increment pattern
def fibonacci_step(x):
    # Returns next number based on digit sum
    return x + sum(int(d) for d in str(x))

fib_range = custom_range(10, 100, fibonacci_step)
fib_sequence = list(fib_range)
print(f"Custom increment sequence: {fib_sequence}")
```

Slide 11: Lazy Data Processing Pipeline

Creating a lazy data processing pipeline demonstrates how iter() can be used to build memory-efficient data transformation chains, processing large datasets without loading them entirely into memory.

```python
class DataPipeline:
    def __init__(self, data_source):
        self.source = data_source
        self.transformations = []
    
    def add_transformation(self, func):
        self.transformations.append(func)
        return self
    
    def process(self):
        def generator():
            for item in self.source:
                result = item
                for transform in self.transformations:
                    result = transform(result)
                    if result is None:
                        return None
                return result
        
        return iter(generator, None)

# Example usage
data = range(1, 11)
pipeline = DataPipeline(data)
processed = pipeline.add_transformation(lambda x: x * 2)\
                   .add_transformation(lambda x: x + 1)\
                   .add_transformation(lambda x: x if x < 20 else None)\
                   .process()

results = list(processed)
print(f"Processed data: {results}")
```

Slide 12: Time Series Data Simulator

A time series data simulator using iter() shows how to generate continuous data streams with time-based patterns, useful for testing time series analysis algorithms and monitoring systems.

```python
import math
from datetime import datetime, timedelta

class TimeSeriesSimulator:
    def __init__(self, duration_seconds=10, frequency=1.0):
        self.start_time = datetime.now()
        self.duration = timedelta(seconds=duration_seconds)
        self.frequency = frequency
        self.t = 0
        
    def next_value(self):
        current_time = datetime.now()
        if current_time - self.start_time > self.duration:
            return None
            
        value = math.sin(2 * math.pi * self.frequency * self.t) + \
                random.uniform(-0.1, 0.1)
        self.t += 0.1
        return (current_time, value)

# Example usage
simulator = TimeSeriesSimulator(duration_seconds=5, frequency=0.5)
time_series = iter(simulator.next_value, None)

data_points = []
for timestamp, value in time_series:
    data_points.append((timestamp, round(value, 3)))
    if len(data_points) <= 5:  # Show first 5 points
        print(f"Time: {timestamp}, Value: {value:.3f}")
```

Slide 13: Additional Resources

*   Research paper on Iterator Patterns in Python: [https://www.researchgate.net/publication/Python\_Iterator\_Patterns](https://www.researchgate.net/publication/Python_Iterator_Patterns)
*   Advanced Iterator Implementation Techniques: [https://realpython.com/python-iterators-iterables/](https://realpython.com/python-iterators-iterables/)
*   Python Documentation on Iterators and Generators: [https://docs.python.org/3/library/itertools.html](https://docs.python.org/3/library/itertools.html)
*   Data Processing with Python Iterators: [https://pydata.org/documentation/iterators-guide](https://pydata.org/documentation/iterators-guide)
*   Performance Analysis of Iterator Patterns: [https://www.python.org/dev/peps/pep-0234/](https://www.python.org/dev/peps/pep-0234/)

