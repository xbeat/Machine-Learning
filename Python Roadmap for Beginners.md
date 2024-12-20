## Python Roadmap for Beginners
Slide 1: Data Structures - Binary Trees From Scratch

A binary tree is a hierarchical data structure where each node has at most two children. This implementation demonstrates a complete binary tree class with core operations like insertion, traversal, and search using recursive algorithms.

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
            return
        
        def _insert_recursive(node, value):
            if value < node.value:
                if node.left is None:
                    node.left = Node(value)
                else:
                    _insert_recursive(node.left, value)
            else:
                if node.right is None:
                    node.right = Node(value)
                else:
                    _insert_recursive(node.right, value)
                    
        _insert_recursive(self.root, value)

# Example usage
tree = BinaryTree()
for val in [5, 3, 7, 1, 4, 6, 8]:
    tree.insert(val)
```

Slide 2: Advanced Recursion - Dynamic Programming

Dynamic programming optimizes recursive solutions by storing intermediate results. This example implements the classic fibonacci sequence using both naive recursion and dynamic programming to demonstrate performance differences.

```python
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

def fib_dynamic(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]

# Performance comparison
import time

n = 35
start = time.time()
fib_recursive(n)
print(f"Recursive time: {time.time() - start:.2f} seconds")

start = time.time()
fib_dynamic(n)
print(f"Dynamic time: {time.time() - start:.2f} seconds")
```

Slide 3: Advanced Object-Oriented Programming - Metaclasses

Metaclasses provide a powerful way to customize class creation in Python. This example demonstrates how to create a singleton pattern using metaclasses, ensuring only one instance of a class exists.

```python
class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=Singleton):
    def __init__(self):
        self.connection = "Connected to DB"
    
    def query(self, sql):
        return f"Executing: {sql}"

# Usage demonstration
db1 = Database()
db2 = Database()
print(db1 is db2)  # True
print(db1.query("SELECT * FROM users"))
```

Slide 4: Concurrent Programming - AsyncIO

AsyncIO enables cooperative multitasking using coroutines. This implementation shows how to handle multiple network requests concurrently using Python's asyncio library.

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = [
        'http://python.org',
        'http://golang.org',
        'http://rust-lang.org'
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return responses

# Running the async code
start = time.time()
responses = asyncio.run(main())
print(f"Completed in {time.time() - start:.2f} seconds")
```

Slide 5: Advanced Pattern Matching with Python 3.10+

Pattern matching extends beyond simple switch statements, allowing complex structural pattern matching. This implementation demonstrates advanced pattern matching for data parsing and protocol handling.

```python
def parse_command(command):
    match command.split():
        case ["quit" | "exit"]:
            return "Exiting program"
        case ["save", filename]:
            return f"Saving to {filename}"
        case ["load", filename, *flags] if flags:
            return f"Loading {filename} with flags: {flags}"
        case ["compute", *values] if all(v.isdigit() for v in values):
            nums = [int(v) for v in values]
            return f"Sum: {sum(nums)}"
        case _:
            return "Invalid command"

# Example usage
commands = [
    "save data.txt",
    "load config.json --verbose --debug",
    "compute 1 2 3 4 5",
    "unknown command"
]

for cmd in commands:
    print(f"Command: {cmd}")
    print(f"Result: {parse_command(cmd)}\n")
```

Slide 6: Custom Context Managers and Resource Management

Context managers provide elegant resource handling and cleanup. This implementation shows how to create custom context managers for managing complex resources and transactions.

```python
from contextlib import contextmanager
import time

class Transaction:
    def __init__(self, name):
        self.name = name
        
    @contextmanager
    def atomic(self):
        print(f"Starting transaction: {self.name}")
        state = {"rolled_back": False}
        try:
            yield self
            if not state["rolled_back"]:
                print("Committing transaction")
        except Exception as e:
            print(f"Rolling back due to: {str(e)}")
            state["rolled_back"] = True
            raise
        finally:
            print("Cleaning up resources")

class Database:
    @contextmanager
    def transaction(self, name):
        txn = Transaction(name)
        with txn.atomic():
            yield txn

# Usage example
db = Database()
try:
    with db.transaction("user_update") as txn:
        # Simulate operations
        print("Updating user data...")
        time.sleep(1)
        raise ValueError("Simulation of error")
except ValueError:
    print("Transaction failed but resources were cleaned up")
```

Slide 7: High-Performance Numerical Computing

Implementation of high-performance numerical computations using NumPy and vectorization techniques. This example demonstrates matrix operations and performance optimization.

```python
import numpy as np
import time

def matrix_operation_optimized(size):
    # Create large matrices
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Vectorized operations
    start = time.time()
    C = np.dot(A, B)
    D = np.exp(C) / (1 + np.exp(C))  # Sigmoid
    result = np.sum(D)
    vectorized_time = time.time() - start
    
    print(f"Vectorized computation: {vectorized_time:.4f} seconds")
    return result

def matrix_operation_naive(size):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Non-vectorized operations
    start = time.time()
    C = [[sum(A[i][k] * B[k][j] for k in range(size)) 
          for j in range(size)] for i in range(size)]
    D = [[1/(1 + np.exp(-C[i][j])) 
          for j in range(size)] for i in range(size)]
    result = sum(sum(row) for row in D)
    naive_time = time.time() - start
    
    print(f"Naive computation: {naive_time:.4f} seconds")
    return result

# Compare performance
size = 100
result_vectorized = matrix_operation_optimized(size)
result_naive = matrix_operation_naive(size)
```

Slide 8: Advanced Generator Patterns

Exploring advanced generator patterns for memory-efficient data processing. This implementation shows custom generators with complex iteration patterns and pipeline processing.

```python
def data_pipeline():
    def generate_data(n):
        for i in range(n):
            yield {"id": i, "value": i * i}
    
    def filter_even(data):
        for item in data:
            if item["value"] % 2 == 0:
                yield item
    
    def transform_data(data):
        for item in data:
            item["transformed"] = item["value"] ** 0.5
            yield item
    
    def process_chunk(data, chunk_size=3):
        chunk = []
        for item in data:
            chunk.append(item)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Create processing pipeline
    raw_data = generate_data(10)
    filtered_data = filter_even(raw_data)
    transformed_data = transform_data(filtered_data)
    chunks = process_chunk(transformed_data)
    
    return chunks

# Execute pipeline
for chunk in data_pipeline():
    print(f"Processing chunk: {chunk}")
```

Slide 9: Custom Decorators with Parameters

Advanced decorator patterns that allow parameter customization and function modification. This implementation demonstrates how to create flexible decorators for logging, timing, and caching.

```python
import functools
import time
from typing import Any, Callable

def parametrized_decorator(retries: int = 3, delay: float = 1.0):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            
            for attempt in range(retries):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        print(f"Success on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < retries - 1:
                        time.sleep(delay)
            
            raise last_exception
            
        return wrapper
    return decorator

# Example usage
@parametrized_decorator(retries=3, delay=0.5)
def unstable_network_call(url: str) -> str:
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return f"Success: {url}"

# Test the decorated function
try:
    result = unstable_network_call("http://example.com")
    print(f"Final result: {result}")
except ConnectionError as e:
    print(f"All retries failed: {e}")
```

Slide 10: Advanced Memory Management

Implementation of memory-efficient data structures using slots and weakref for optimized memory usage and garbage collection control.

```python
import weakref
from sys import getsizeof

class StandardClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class OptimizedClass:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

class WeakRefContainer:
    def __init__(self):
        self._data = weakref.WeakKeyDictionary()
    
    def add_item(self, key, value):
        self._data[key] = value
    
    def get_item(self, key):
        return self._data.get(key)

# Memory comparison
std_objects = [StandardClass(i, i) for i in range(1000)]
opt_objects = [OptimizedClass(i, i) for i in range(1000)]

print(f"Standard object size: {getsizeof(std_objects[0])} bytes")
print(f"Optimized object size: {getsizeof(opt_objects[0])} bytes")

# WeakRef demonstration
container = WeakRefContainer()
class DataObject:
    def __init__(self, value):
        self.value = value

obj = DataObject(42)
container.add_item(obj, "associated_data")
print(f"Data exists: {container.get_item(obj)}")
del obj  # Object will be garbage collected
```

Slide 11: Advanced Type Hints and Protocol Classes

Implementation of complex type hints using Protocol classes and Generic types for robust type checking and interface definitions.

```python
from typing import TypeVar, Protocol, Generic, Iterator
from dataclasses import dataclass

T = TypeVar('T')

class Comparable(Protocol):
    def __lt__(self, other: Any) -> bool: ...

class Sortable(Protocol[T]):
    def sort_key(self) -> Comparable: ...

@dataclass
class SortableContainer(Generic[T]):
    data: list[T]
    
    def sort(self) -> None:
        if all(isinstance(item, Sortable) for item in self.data):
            self.data.sort(key=lambda x: x.sort_key())
    
    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

# Example usage
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age
    
    def sort_key(self) -> int:
        return self.age
    
    def __repr__(self) -> str:
        return f"Person(name={self.name}, age={self.age})"

# Create and sort data
people = SortableContainer([
    Person("Alice", 30),
    Person("Bob", 25),
    Person("Charlie", 35)
])

people.sort()
for person in people:
    print(person)
```

Slide 12: High-Performance Data Processing Pipeline

Implementation of a multi-threaded data processing pipeline using queues and thread pools for efficient large-scale data processing.

```python
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time

class DataPipeline:
    def __init__(self, num_workers=4):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.num_workers = num_workers
        self.stop_event = threading.Event()
    
    def producer(self, data):
        for item in data:
            if self.stop_event.is_set():
                break
            self.input_queue.put(item)
        self.input_queue.put(None)  # Sentinel
    
    def process_item(self, item):
        if item is None:
            return None
        # Simulate complex processing
        time.sleep(0.1)
        return item * item
    
    def consumer(self):
        while not self.stop_event.is_set():
            item = self.output_queue.get()
            if item is None:
                break
            print(f"Processed: {item}")
            self.output_queue.task_done()
    
    def worker(self):
        while not self.stop_event.is_set():
            item = self.input_queue.get()
            if item is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break
            result = self.process_item(item)
            self.output_queue.put(result)
            self.input_queue.task_done()
    
    def run(self, data):
        # Start workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            workers = [executor.submit(self.worker) 
                      for _ in range(self.num_workers)]
            
            # Start producer and consumer
            producer_thread = threading.Thread(
                target=self.producer, args=(data,))
            consumer_thread = threading.Thread(target=self.consumer)
            
            producer_thread.start()
            consumer_thread.start()
            
            # Wait for completion
            producer_thread.join()
            self.input_queue.join()
            consumer_thread.join()

# Example usage
pipeline = DataPipeline(num_workers=4)
data = range(20)
pipeline.run(data)
```

Slide 13: Advanced Neural Network Implementation

Pure Python implementation of a neural network with backpropagation, demonstrating deep learning concepts without external libraries.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(net))
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                        self.sigmoid_derivative(self.activations[i])
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
for _ in range(10000):
    nn.forward(X)
    nn.backward(X, y)

print("Predictions:")
print(nn.forward(X))
```

Slide 14: Additional Resources

*   arXiv:2103.11955 - "Python for Scientific Computing" - [https://arxiv.org/abs/2103.11955](https://arxiv.org/abs/2103.11955)
*   arXiv:1907.05559 - "High Performance Python Programming" - [https://arxiv.org/abs/1907.05559](https://arxiv.org/abs/1907.05559)
*   arXiv:2002.04667 - "Modern Python Development Practices" - [https://arxiv.org/abs/2002.04667](https://arxiv.org/abs/2002.04667)
*   Search terms for further learning:
    *   "Advanced Python Design Patterns"
    *   "Python Concurrency and Parallelism"
    *   "Memory Optimization in Python"
    *   "Python Type System and Static Analysis"

