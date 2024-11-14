## Roadmap to Becoming a Python Pro
Slide 1: Advanced List Comprehensions and Generators

List comprehensions and generators are powerful Python features that enable concise, memory-efficient data transformations. While basic comprehensions create lists, generator expressions produce values on-demand, reducing memory usage for large datasets and enabling infinite sequences.

```python
# Advanced list comprehension with multiple conditions
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [x for row in matrix for x in row if x % 2 == 0]
print(f"Filtered flat list: {flat}")  # Output: [2, 4, 6, 8]

# Generator for Fibonacci sequence
def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using the generator
fib = fibonacci_generator()
first_10 = [next(fib) for _ in range(10)]
print(f"First 10 Fibonacci numbers: {first_10}")
```

Slide 2: Custom Decorators with Parameters

Decorators are metaprogramming tools that modify function behavior. Parameter-accepting decorators add flexibility by allowing runtime configuration of the decoration process, enabling reusable code patterns and aspect-oriented programming concepts.

```python
def retry(max_attempts=3, delay_seconds=1):
    import time
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    time.sleep(delay_seconds)
            return None
        return wrapper
    return decorator

@retry(max_attempts=2, delay_seconds=0.1)
def unstable_network_call(url):
    import random
    if random.random() < 0.5:
        raise ConnectionError("Network unstable")
    return f"Success: {url}"

# Example usage
try:
    result = unstable_network_call("api.example.com")
    print(result)
except ConnectionError as e:
    print(f"Failed after retries: {e}")
```

Slide 3: Context Managers From Scratch

Context managers provide a robust way to handle resource acquisition and release. Understanding their implementation helps create clean, maintainable code that properly manages system resources and ensures cleanup operations.

```python
from typing import Optional
import time

class Timer:
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.description} took {duration:.2f} seconds")
        return False  # Don't suppress exceptions

# Example usage
with Timer("Complex calculation"):
    # Simulate complex operation
    time.sleep(1.5)
    result = sum(i * i for i in range(1000000))
```

Slide 4: Metaclasses and Class Creation

Metaclasses control class creation and behavior, offering powerful ways to modify class definitions at runtime. They enable framework development, attribute validation, and automatic registration of classes in a system.

```python
class ValidateAttributes(type):
    def __new__(cls, name, bases, attrs):
        # Validate attributes during class creation
        for key, value in attrs.items():
            if key.startswith('__'):
                continue
            if isinstance(value, str):
                attrs[key] = value.strip()
            elif isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(f"Negative values not allowed: {key}")
        return super().__new__(cls, name, bases, attrs)

class Product(metaclass=ValidateAttributes):
    name = "  Sample Product  "  # Will be stripped
    price = 99.99  # Will be validated
    stock = 50     # Will be validated

# Example usage
product = Product()
print(f"Product name: {product.name}")  # Output: "Sample Product"
```

Slide 5: Advanced Error Handling and Custom Exceptions

Modern error handling extends beyond basic try-except blocks. Custom exception hierarchies, contextual error information, and chained exceptions provide robust error management for complex applications while maintaining debugging capabilities.

```python
class DomainError(Exception):
    """Base exception for all domain-specific errors"""
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = time.time()

class ValidationError(DomainError):
    """Raised when data validation fails"""
    def __init__(self, field, value, reason, **kwargs):
        context = {
            'field': field,
            'invalid_value': value,
            'reason': reason,
            **kwargs
        }
        super().__init__(f"Validation failed for {field}: {reason}", context)

def process_user_data(data: dict):
    try:
        if not data.get('email'):
            raise ValidationError('email', data.get('email'), 'Email is required')
        if len(data.get('password', '')) < 8:
            raise ValidationError('password', '***', 'Password too short',
                               min_length=8)
    except ValidationError as e:
        print(f"Error: {e}")
        print(f"Context: {e.context}")
        raise

# Example usage
try:
    process_user_data({'email': '', 'password': '123'})
except ValidationError as e:
    print(f"Validation failed at {time.ctime(e.timestamp)}")
```

Slide 6: Memory-Efficient Data Processing

Processing large datasets requires careful memory management. Using generators, itertools, and chunked processing enables handling massive data volumes without overwhelming system resources.

```python
from itertools import islice
from typing import Iterator, Any
import sys

def memory_efficient_reader(file_path: str, chunk_size: int = 1024) -> Iterator[str]:
    """Read large files in chunks to minimize memory usage"""
    with open(file_path, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def process_in_batches(data: Iterator[Any], batch_size: int = 100) -> Iterator[list]:
    """Process iterator data in fixed-size batches"""
    batch = list(islice(data, batch_size))
    while batch:
        yield batch
        batch = list(islice(data, batch_size))

# Example usage with memory tracking
def memory_usage() -> float:
    """Get current memory usage in MB"""
    return sys.getsizeof(0) * len(globals()) / 1024 / 1024

# Simulate processing large dataset
def process_large_dataset():
    data_generator = (i for i in range(1000000))
    memory_before = memory_usage()
    
    for batch in process_in_batches(data_generator, 1000):
        # Process batch
        result = sum(batch)
        
    memory_after = memory_usage()
    print(f"Memory usage: {memory_after - memory_before:.2f} MB")

process_large_dataset()
```

Slide 7: Advanced Concurrency with asyncio

Modern Python applications leverage asynchronous programming for improved performance. The asyncio framework enables concurrent execution of I/O-bound operations while maintaining readable, sequential-looking code.

```python
import asyncio
import aiohttp
import time
from typing import List, Dict

async def fetch_data(session: aiohttp.ClientSession, url: str) -> Dict:
    """Asynchronously fetch data from URL"""
    async with session.get(url) as response:
        return {
            'url': url,
            'status': response.status,
            'data': await response.json()
        }

async def process_urls(urls: List[str]) -> List[Dict]:
    """Process multiple URLs concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Example usage
async def main():
    urls = [
        'https://api.github.com/users/github',
        'https://api.github.com/users/python',
        'https://api.github.com/users/django'
    ]
    
    start_time = time.time()
    results = await process_urls(urls)
    duration = time.time() - start_time
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Error: {result}")
        else:
            print(f"Successfully fetched: {result['url']}")
    
    print(f"Completed in {duration:.2f} seconds")

# Run the async code
if __name__ == "__main__":
    asyncio.run(main())
```

Slide 8: Advanced Object-Oriented Design Patterns

Design patterns provide proven solutions to common software engineering challenges. Understanding and implementing these patterns enables creation of maintainable, scalable, and flexible software systems.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import json

# Command Pattern with Builder
@dataclass
class Document:
    content: str = ""
    
    def append(self, text: str) -> None:
        self.content += text

class Command(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass
    
    @abstractmethod
    def undo(self) -> None:
        pass

class AppendCommand(Command):
    def __init__(self, document: Document, text: str):
        self.document = document
        self.text = text
        self._previous_state = None
    
    def execute(self) -> None:
        self._previous_state = self.document.content
        self.document.append(self.text)
    
    def undo(self) -> None:
        if self._previous_state is not None:
            self.document.content = self._previous_state

class DocumentBuilder:
    def __init__(self):
        self.document = Document()
        self.commands: List[Command] = []
    
    def append_text(self, text: str) -> 'DocumentBuilder':
        command = AppendCommand(self.document, text)
        command.execute()
        self.commands.append(command)
        return self
    
    def undo_last(self) -> 'DocumentBuilder':
        if self.commands:
            command = self.commands.pop()
            command.undo()
        return self
    
    def build(self) -> Document:
        return self.document

# Example usage
builder = DocumentBuilder()
doc = (builder.append_text("Hello ")
              .append_text("World!")
              .undo_last()
              .append_text("Python!")
              .build())

print(f"Final document: {doc.content}")  # Output: Hello Python!
```

Slide 9: Advanced Data Structures Implementation

Custom data structures implementation provides deep understanding of algorithmic complexity and memory management. Building efficient data structures requires careful consideration of access patterns and performance characteristics.

```python
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None
    
    def height(self, node):
        if not node:
            return 0
        return node.height
    
    def balance_factor(self, node):
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)
    
    def rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = max(self.height(y.left), self.height(y.right)) + 1
        x.height = max(self.height(x.left), self.height(x.right)) + 1
        return x
    
    def rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = max(self.height(x.left), self.height(x.right)) + 1
        y.height = max(self.height(y.left), self.height(y.right)) + 1
        return y
    
    def insert(self, root, key):
        if not root:
            return AVLNode(key)
        
        if key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:
            return root
        
        root.height = max(self.height(root.left), self.height(root.right)) + 1
        balance = self.balance_factor(root)
        
        # Left Left Case
        if balance > 1 and key < root.left.key:
            return self.rotate_right(root)
        
        # Right Right Case
        if balance < -1 and key > root.right.key:
            return self.rotate_left(root)
        
        # Left Right Case
        if balance > 1 and key > root.left.key:
            root.left = self.rotate_left(root.left)
            return self.rotate_right(root)
        
        # Right Left Case
        if balance < -1 and key < root.right.key:
            root.right = self.rotate_right(root.right)
            return self.rotate_left(root)
        
        return root

# Example usage
avl = AVLTree()
root = None
keys = [10, 20, 30, 40, 50, 25]

for key in keys:
    root = avl.insert(root, key)

print("AVL Tree created successfully")
```

Slide 10: Advanced Python Metaprogramming

Metaprogramming allows programs to analyze and modify their own structure and behavior at runtime. This powerful feature enables creation of flexible APIs, dynamic code generation, and sophisticated framework development.

```python
class AutoProperty:
    def __init__(self, validator=None):
        self.validator = validator
        self._name = None
    
    def __set_name__(self, owner, name):
        self._name = f'_{name}'
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self._name, None)
    
    def __set__(self, instance, value):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self._name}: {value}")
        setattr(instance, self._name, value)

def validate_positive(value):
    return isinstance(value, (int, float)) and value > 0

class MetaValidator(type):
    def __new__(cls, name, bases, namespace):
        # Add validation to all numeric attributes
        for key, value in namespace.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                namespace[key] = AutoProperty(validate_positive)
        return super().__new__(cls, name, bases, namespace)

class Product(metaclass=MetaValidator):
    price = 0  # Will be converted to AutoProperty with validation
    quantity = 0  # Will be converted to AutoProperty with validation
    
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity
    
    @property
    def total(self):
        return self.price * self.quantity

# Example usage
try:
    product = Product(price=10.5, quantity=5)
    print(f"Total: {product.total}")
    
    # This will raise ValueError
    product.price = -20
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 11: Advanced Pattern Matching with Match-Case

Python 3.10 introduced pattern matching, enabling sophisticated data structure decomposition and control flow. This feature provides elegant solutions for complex branching logic and data processing.

```python
from dataclasses import dataclass
from typing import List, Union, Optional

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    radius: float

@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point

def analyze_shape(shape: Union[Circle, Rectangle, Point, List[Point]]) -> str:
    match shape:
        case Circle(center=Point(x=0, y=0), radius=r):
            return f"Circle centered at origin with radius {r}"
        
        case Circle(center=Point(x=x, y=y), radius=r) if r > 10:
            return f"Large circle at ({x}, {y}) with radius {r}"
        
        case Rectangle(top_left=Point(x1, y1), bottom_right=Point(x2, y2)):
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            return f"Rectangle {width}x{height}"
        
        case Point(x, y):
            return f"Single point at ({x}, {y})"
        
        case [Point(x=x1, y=y1), Point(x=x2, y=y2)]:
            return f"Line from ({x1}, {y1}) to ({x2}, {y2})"
        
        case [Point(_, _), *rest] if len(rest) > 0:
            return f"Polygon with {len(rest) + 1} points"
        
        case _:
            return "Unknown shape"

# Example usage
shapes = [
    Circle(Point(0, 0), 5),
    Circle(Point(1, 1), 15),
    Rectangle(Point(0, 0), Point(3, 4)),
    Point(1, 1),
    [Point(0, 0), Point(1, 1), Point(2, 2)]
]

for shape in shapes:
    print(analyze_shape(shape))
```

Slide 12: Advanced Network Programming

Network programming in Python extends beyond basic socket operations. Modern implementations handle complex protocols, asynchronous communications, and robust error handling for distributed systems development.

```python
import socket
import selectors
import types
from typing import Dict, Optional
import json

class NonBlockingServer:
    def __init__(self, host: str, port: int):
        self.sel = selectors.DefaultSelector()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()
        self.sock.setblocking(False)
        self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        
    def accept_connection(self, sock: socket.socket):
        conn, addr = sock.accept()
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, events, data=data)
        print(f"Accepted connection from {addr}")
        
    def service_connection(self, key: selectors.SelectorKey, mask: int):
        sock = key.fileobj
        data = key.data
        
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)
            if recv_data:
                data.outb += self.process_request(recv_data)
            else:
                self.sel.unregister(sock)
                sock.close()
                
        if mask & selectors.EVENT_WRITE:
            if data.outb:
                sent = sock.send(data.outb)
                data.outb = data.outb[sent:]
    
    def process_request(self, data: bytes) -> bytes:
        try:
            request = json.loads(data.decode())
            response = {
                "status": "success",
                "message": f"Processed {request.get('command', 'unknown')}"
            }
        except json.JSONDecodeError:
            response = {
                "status": "error",
                "message": "Invalid JSON format"
            }
        return json.dumps(response).encode()
    
    def serve_forever(self):
        print("Server started...")
        try:
            while True:
                events = self.sel.select(timeout=None)
                for key, mask in events:
                    if key.data is None:
                        self.accept_connection(key.fileobj)
                    else:
                        self.service_connection(key, mask)
        finally:
            self.sel.close()

# Example client code
def create_client_connection(host: str, port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    request = json.dumps({"command": "ping"}).encode()
    sock.send(request)
    response = sock.recv(1024).decode()
    print(f"Server response: {response}")
    sock.close()

# Usage example
if __name__ == "__main__":
    import threading
    
    server = NonBlockingServer('localhost', 12345)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    
    # Simulate client connections
    create_client_connection('localhost', 12345)
```

Slide 13: Advanced Mathematical Computations

Complex mathematical operations require careful implementation of numerical algorithms. This implementation showcases matrix operations and numerical methods with pure Python.

```python
import math
from typing import List, Tuple
import random

class NumericalMethods:
    @staticmethod
    def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        if not A or not B or len(A[0]) != len(B):
            raise ValueError("Invalid matrix dimensions")
        
        result = [[0.0 for _ in range(len(B[0]))] for _ in range(len(A))]
        
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    
    @staticmethod
    def newton_raphson(f, f_prime, x0: float, tolerance: float = 1e-7, max_iter: int = 100) -> Tuple[float, int]:
        """
        Find root using Newton-Raphson method
        f: function to find root of
        f_prime: derivative of f
        """
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < tolerance:
                return x, i
            
            fp = f_prime(x)
            if fp == 0:
                raise ValueError("Derivative is zero")
            
            x = x - fx/fp
        
        raise ValueError("Failed to converge")
    
    @staticmethod
    def monte_carlo_integration(f, a: float, b: float, n: int = 10000) -> Tuple[float, float]:
        """
        Compute integral using Monte Carlo method
        Returns (estimate, error_estimate)
        """
        total = 0.0
        total_squared = 0.0
        
        for _ in range(n):
            x = random.uniform(a, b)
            fx = f(x)
            total += fx
            total_squared += fx * fx
        
        mean = total / n
        variance = (total_squared / n - mean * mean) / (n - 1)
        integral = (b - a) * mean
        error = (b - a) * math.sqrt(variance / n)
        
        return integral, error

# Example usage
def example_calculations():
    # Matrix multiplication
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = NumericalMethods.matrix_multiply(A, B)
    print(f"Matrix multiplication result: {result}")
    
    # Root finding
    f = lambda x: x**2 - 4
    f_prime = lambda x: 2*x
    root, iterations = NumericalMethods.newton_raphson(f, f_prime, 3.0)
    print(f"Root found: {root} in {iterations} iterations")
    
    # Integration
    f = lambda x: math.sin(x)
    integral, error = NumericalMethods.monte_carlo_integration(f, 0, math.pi)
    print(f"Integral estimate: {integral:.6f} ± {error:.6f}")

if __name__ == "__main__":
    example_calculations()
```

Slide 14: Additional Resources

*   Advanced Python Programming:
    *   [https://arxiv.org/abs/2207.09467](https://arxiv.org/abs/2207.09467) (Advanced Python Design Patterns)
    *   [https://arxiv.org/abs/2103.11928](https://arxiv.org/abs/2103.11928) (Python Performance Optimization)
    *   [https://arxiv.org/abs/2106.14938](https://arxiv.org/abs/2106.14938) (Modern Python Concurrency Patterns)
*   Further Reading:
    *   [https://python.org/dev/peps/](https://python.org/dev/peps/) (Python Enhancement Proposals)
    *   [https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html) (AsyncIO Documentation)
    *   [https://realpython.com/advanced-python-patterns/](https://realpython.com/advanced-python-patterns/)
*   Code Examples and Tutorials:
    *   [https://github.com/python/cpython](https://github.com/python/cpython) (CPython Source Code)
    *   [https://github.com/jupyter/jupyter](https://github.com/jupyter/jupyter) (Jupyter Notebooks)
    *   [https://scipy-lectures.org/](https://scipy-lectures.org/) (Scientific Python)

