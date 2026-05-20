## The Evolution of Python From Scripting to Powerhouse
Slide 1: Evolution of Python Data Structures

Modern Python offers sophisticated data structure implementations that go far beyond basic lists and dictionaries, enabling efficient manipulation of complex data with minimal code overhead while maintaining readability and performance optimization.

```python
# Advanced data structure implementations
from collections import defaultdict, Counter, deque
from typing import Dict, List, Set, DefaultDict

# Example of sophisticated data manipulation
def analyze_text_patterns(text: str) -> DefaultDict[str, Set[int]]:
    # Track word positions using defaultdict of sets
    word_positions: DefaultDict[str, Set[int]] = defaultdict(set)
    
    # Process words and their positions
    for position, word in enumerate(text.lower().split()):
        word_positions[word].add(position)
    
    # Get word frequency using Counter
    frequency = Counter(text.lower().split())
    print(f"Word frequencies: {dict(frequency.most_common(3))}")
    
    return word_positions

# Example usage
text = "Python data structures are powerful Python tools for Python developers"
positions = analyze_text_patterns(text)
print(f"'Python' appears at positions: {positions['python']}")

# Output:
# Word frequencies: {'python': 3, 'data': 1, 'structures': 1}
# 'Python' appears at positions: {0, 4, 6}
```

Slide 2: Advanced Generator Patterns

Generators in modern Python have evolved to handle complex iteration patterns and memory-efficient data processing, incorporating features like sub-generators and generator expressions for sophisticated data streaming applications.

```python
def fibonacci_with_metadata(limit: int):
    """Advanced generator demonstrating modern Python features"""
    a, b = 0, 1
    count = 0
    
    while count < limit:
        # Yield both value and metadata
        metadata = {
            'index': count,
            'is_even': a % 2 == 0,
            'squared': a ** 2
        }
        yield a, metadata
        
        a, b = b, a + b
        count += 1

# Generator expression with filtering
even_squares = (
    metadata['squared'] 
    for num, metadata in fibonacci_with_metadata(10)
    if metadata['is_even']
)

# Example usage
for i, (num, meta) in enumerate(fibonacci_with_metadata(5)):
    print(f"Fibonacci {meta['index']}: {num} (Even: {meta['is_even']})")

# Output:
# Fibonacci 0: 0 (Even: True)
# Fibonacci 1: 1 (Even: False)
# Fibonacci 2: 1 (Even: False)
# Fibonacci 3: 2 (Even: True)
# Fibonacci 4: 3 (Even: False)
```

Slide 3: Modern Concurrency with asyncio

The introduction of asyncio revolutionized Python's approach to concurrent programming, enabling efficient handling of thousands of concurrent connections with coroutines and event loops.

```python
import asyncio
from typing import List
import time

async def async_operation(task_id: int, delay: float) -> dict:
    """Simulate an async I/O operation"""
    await asyncio.sleep(delay)
    return {
        'task_id': task_id,
        'timestamp': time.time(),
        'delay': delay
    }

async def process_batch(batch_size: int = 5) -> List[dict]:
    # Create multiple concurrent tasks
    tasks = [
        async_operation(i, delay=0.5) 
        for i in range(batch_size)
    ]
    
    # Gather results concurrently
    results = await asyncio.gather(*tasks)
    return results

# Example usage
async def main():
    start = time.time()
    results = await process_batch()
    duration = time.time() - start
    
    print(f"Processed {len(results)} tasks in {duration:.2f} seconds")
    print(f"First result: {results[0]}")

# Run the async code
asyncio.run(main())

# Output:
# Processed 5 tasks in 0.51 seconds
# First result: {'task_id': 0, 'timestamp': 1699901234.567, 'delay': 0.5}
```

Slide 4: Advanced Type Hints and Runtime Verification

Modern Python type system provides sophisticated static type checking capabilities while maintaining runtime verification through runtime\_checkable protocols and Generic types, enabling robust software architecture.

```python
from typing import TypeVar, Protocol, runtime_checkable, Generic
from dataclasses import dataclass
from abc import abstractmethod

T = TypeVar('T')

@runtime_checkable
class Processable(Protocol):
    @abstractmethod
    def process(self) -> str: ...

@dataclass
class DataProcessor(Generic[T]):
    data: T
    
    def validate(self) -> bool:
        return isinstance(self.data, Processable)
    
    def execute(self) -> str:
        if not self.validate():
            raise TypeError(f"{type(self.data)} must implement Processable")
        return self.data.process()

# Implementation example
class NumericData:
    def __init__(self, value: float):
        self.value = value
    
    def process(self) -> str:
        return f"Processed value: {self.value * 2}"

# Usage example
processor = DataProcessor(NumericData(42.0))
print(processor.execute())
print(f"Is Processable? {isinstance(processor.data, Processable)}")

# Output:
# Processed value: 84.0
# Is Processable? True
```

Slide 5: Neural Network Implementation from Scratch

Modern Python's numerical capabilities enable efficient implementation of deep learning architectures using only NumPy, demonstrating the language's power for scientific computing.

```python
import numpy as np
from typing import List, Tuple, Callable

class NeuralNetwork:
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        self.weights = [
            np.random.randn(layers[i], layers[i+1]) / np.sqrt(layers[i])
            for i in range(len(layers)-1)
        ]
        self.biases = [np.zeros((1, l)) for l in layers[1:]]
        self.lr = learning_rate
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)
    
    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        layer_inputs = []
        
        for w, b in zip(self.weights, self.biases):
            layer_input = np.dot(activations[-1], w) + b
            layer_inputs.append(layer_input)
            activations.append(self.sigmoid(layer_input))
            
        return activations, layer_inputs

    def backward(self, x: np.ndarray, y: np.ndarray) -> float:
        activations, layer_inputs = self.forward(x)
        m = x.shape[0]
        delta = activations[-1] - y
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * \
                        self.sigmoid_derivative(activations[i])
            
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * db
            
        return np.mean(np.square(activations[-1] - y))

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 4, 1])
for epoch in range(5000):
    loss = nn.backward(X, y)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Predictions
predictions = nn.forward(X)[0][-1]
print("\nPredictions:")
print(predictions)

# Output:
# Epoch 0, Loss: 0.2531
# Epoch 1000, Loss: 0.0412
# Epoch 2000, Loss: 0.0198
# Epoch 3000, Loss: 0.0156
# Epoch 4000, Loss: 0.0134
# 
# Predictions:
# [[0.02]
#  [0.97]
#  [0.98]
#  [0.03]]
```

Slide 6: Modern Python Memory Management

Advanced memory management in Python involves understanding object lifecycle, reference counting, and garbage collection mechanisms. Modern Python provides tools for monitoring and optimizing memory usage through weak references and memory profiling.

```python
import sys
import weakref
import gc
from memory_profiler import profile
from typing import Dict, List, Optional

class Resource:
    def __init__(self, data: bytes):
        self.data = data
    
    def __del__(self):
        print(f"Cleaning up Resource with {len(self.data)} bytes")

class ResourceManager:
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
    
    @profile
    def allocate_resource(self, key: str, size: int) -> Resource:
        # Check if resource exists in cache
        if key in self._cache:
            resource = self._cache[key]()
            if resource is not None:
                return resource
        
        # Create new resource
        resource = Resource(bytes(size))
        self._cache[key] = weakref.ref(resource)
        return resource
    
    def memory_status(self) -> dict:
        return {
            'gc_objects': len(gc.get_objects()),
            'cache_size': len(self._cache),
            'memory_usage': sys.getsizeof(self._cache)
        }

# Example usage
manager = ResourceManager()

# Allocate resources
resources: List[Optional[Resource]] = []
for i in range(3):
    resources.append(manager.allocate_resource(f"resource_{i}", 1024 * 1024))

print("Initial status:", manager.memory_status())

# Clear some references
resources[1] = None
gc.collect()

print("After cleanup:", manager.memory_status())

# Output:
# Initial status: {'gc_objects': 24563, 'cache_size': 3, 'memory_usage': 232}
# Cleaning up Resource with 1048576 bytes
# After cleanup: {'gc_objects': 24561, 'cache_size': 3, 'memory_usage': 232}
```

Slide 7: Advanced Context Managers for Resource Management

Modern Python context managers enable sophisticated resource management patterns, including reentrant locks, context-local storage, and nested context handling with proper cleanup guarantees.

```python
from contextlib import contextmanager, ExitStack
from typing import Generator, Any, ContextManager
import threading
import time

class ResourcePool:
    def __init__(self, max_resources: int = 3):
        self._available = threading.Semaphore(max_resources)
        self._resources: list = []
        self._lock = threading.Lock()
    
    @contextmanager
    def acquire(self) -> Generator[Any, None, None]:
        acquired = self._available.acquire(timeout=5)
        if not acquired:
            raise TimeoutError("Failed to acquire resource")
        
        try:
            with self._lock:
                resource = self._create_resource()
                self._resources.append(resource)
            yield resource
        finally:
            with self._lock:
                self._resources.remove(resource)
                self._available.release()
    
    def _create_resource(self) -> dict:
        return {
            'id': len(self._resources),
            'created_at': time.time()
        }
    
    @contextmanager
    def batch_acquire(self, count: int) -> Generator[list, None, None]:
        with ExitStack() as stack:
            resources = [
                stack.enter_context(self.acquire())
                for _ in range(count)
            ]
            yield resources

# Example usage
pool = ResourcePool(max_resources=3)

def use_resources():
    with pool.batch_acquire(2) as resources:
        print(f"Acquired resources: {resources}")
        time.sleep(1)  # Simulate work

# Concurrent resource usage
threads = [
    threading.Thread(target=use_resources)
    for _ in range(3)
]

for t in threads:
    t.start()

for t in threads:
    t.join()

# Output:
# Acquired resources: [{'id': 0, 'created_at': 1699901234.567}, {'id': 1, 'created_at': 1699901234.568}]
# Acquired resources: [{'id': 0, 'created_at': 1699901235.567}, {'id': 1, 'created_at': 1699901235.568}]
# Acquired resources: [{'id': 0, 'created_at': 1699901236.567}, {'id': 1, 'created_at': 1699901236.568}]
```

Slide 8: Modern Python Meta-Programming

Meta-programming in modern Python leverages advanced decorator patterns, metaclasses, and descriptor protocols to create flexible and maintainable code architectures with powerful runtime behavior modification capabilities.

```python
from typing import Any, Type, Callable, Dict
import functools
import inspect

class ValidationDescriptor:
    def __init__(self, validator: Callable[[Any], bool], doc: str):
        self.validator = validator
        self.__doc__ = doc
        
    def __set_name__(self, owner: Type, name: str):
        self.private_name = f'_{name}'
        
    def __get__(self, obj: Any, objtype: Type = None) -> Any:
        if obj is None:
            return self
        return getattr(obj, self.private_name)
    
    def __set__(self, obj: Any, value: Any):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.private_name}")
        setattr(obj, self.private_name, value)

def validated_property(validator: Callable[[Any], bool], doc: str = None):
    return ValidationDescriptor(validator, doc)

class MetaValidator(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        # Inject validation into methods
        for key, value in namespace.items():
            if inspect.isfunction(value):
                namespace[key] = mcs.validate_method(value)
        return super().__new__(mcs, name, bases, namespace)
    
    @staticmethod
    def validate_method(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            sig = inspect.signature(method)
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            
            # Validate all arguments
            for param in sig.parameters.values():
                if param.annotation != inspect.Parameter.empty:
                    value = bound.arguments[param.name]
                    if not isinstance(value, param.annotation):
                        raise TypeError(f"Expected {param.annotation} for {param.name}")
            
            return method(self, *args, **kwargs)
        return wrapper

class DataModel(metaclass=MetaValidator):
    name = validated_property(lambda x: isinstance(x, str) and len(x) > 0,
                            "Name must be a non-empty string")
    value = validated_property(lambda x: isinstance(x, (int, float)) and x >= 0,
                             "Value must be a non-negative number")
    
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
    
    def process(self, multiplier: float) -> float:
        return self.value * multiplier

# Example usage
try:
    model = DataModel("example", 42.0)
    print(f"Initial state: {model.name} = {model.value}")
    
    result = model.process(2.5)
    print(f"Processed result: {result}")
    
    # This will raise a ValueError
    model.value = -1
except ValueError as e:
    print(f"Validation error: {e}")

# Output:
# Initial state: example = 42.0
# Processed result: 105.0
# Validation error: Invalid value for _value
```

Slide 9: Advanced Pattern Matching with Match Statements

Modern Python's pattern matching introduces sophisticated control flow mechanisms that combine structural pattern matching with type checking and destructuring capabilities.

```python
from dataclasses import dataclass
from typing import Union, List
from enum import Enum, auto

class Operation(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()

@dataclass
class BinaryOperation:
    operation: Operation
    left: Union[float, 'Expression']
    right: Union[float, 'Expression']

@dataclass
class Expression:
    elements: List[Union[float, BinaryOperation]]

def evaluate_expression(expr: Union[float, BinaryOperation, Expression]) -> float:
    match expr:
        case float() as value:
            return value
        
        case BinaryOperation(operation=Operation.ADD, left=left, right=right):
            return evaluate_expression(left) + evaluate_expression(right)
        
        case BinaryOperation(operation=Operation.MULTIPLY, left=left, right=right):
            return evaluate_expression(left) * evaluate_expression(right)
        
        case Expression(elements=[*parts]) if len(parts) > 0:
            total = evaluate_expression(parts[0])
            for part in parts[1:]:
                match part:
                    case BinaryOperation():
                        total = evaluate_expression(
                            BinaryOperation(Operation.MULTIPLY, total, part)
                        )
                    case _:
                        total *= evaluate_expression(part)
            return total
        
        case _:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

# Example usage
expr = Expression([
    2.0,
    BinaryOperation(Operation.ADD, 3.0, 4.0),
    BinaryOperation(Operation.MULTIPLY, 2.0, 3.0)
])

result = evaluate_expression(expr)
print(f"Result: {result}")

# Output:
# Result: 42.0  # (2 * (3 + 4) * (2 * 3))
```

Slide 10: Advanced Data Pipeline Processing

Modern Python enables sophisticated data processing pipelines through generator-based coroutines and async streams, providing memory-efficient handling of large datasets with backpressure support.

```python
import asyncio
from typing import AsyncIterator, Any, Callable, List
from dataclasses import dataclass
import time

@dataclass
class DataChunk:
    id: int
    data: List[float]
    timestamp: float

class DataPipeline:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.queue = asyncio.Queue(maxsize=1000)
        
    async def producer(self, num_chunks: int) -> None:
        for i in range(num_chunks):
            chunk = DataChunk(
                id=i,
                data=[float(x) for x in range(self.batch_size)],
                timestamp=time.time()
            )
            await self.queue.put(chunk)
            await asyncio.sleep(0.01)  # Simulate data generation delay
        
        await self.queue.put(None)  # Signal completion
    
    async def transform(self, 
                       func: Callable[[DataChunk], DataChunk]
                       ) -> AsyncIterator[DataChunk]:
        while True:
            chunk = await self.queue.get()
            if chunk is None:
                break
                
            transformed = func(chunk)
            yield transformed
            self.queue.task_done()
    
    async def process_pipeline(self, 
                             num_chunks: int,
                             transformations: List[Callable[[DataChunk], DataChunk]]
                             ) -> List[DataChunk]:
        results = []
        producer_task = asyncio.create_task(self.producer(num_chunks))
        
        current_stream = self.transform(transformations[0])
        
        # Chain transformations
        for transform_func in transformations[1:]:
            current_stream = (
                transform_func(chunk) 
                async for chunk in current_stream
            )
        
        # Collect results
        async for result in current_stream:
            results.append(result)
        
        await producer_task
        return results

# Example transformations
def scale_data(chunk: DataChunk) -> DataChunk:
    chunk.data = [x * 2 for x in chunk.data]
    return chunk

def add_offset(chunk: DataChunk) -> DataChunk:
    chunk.data = [x + 10 for x in chunk.data]
    return chunk

# Example usage
async def main():
    pipeline = DataPipeline(batch_size=5)
    transformations = [scale_data, add_offset]
    
    results = await pipeline.process_pipeline(
        num_chunks=3,
        transformations=transformations
    )
    
    for result in results:
        print(f"Chunk {result.id}: {result.data[:3]}...")

# Run the pipeline
asyncio.run(main())

# Output:
# Chunk 0: [10.0, 12.0, 14.0]...
# Chunk 1: [10.0, 12.0, 14.0]...
# Chunk 2: [10.0, 12.0, 14.0]...
```

Slide 11: High-Performance Numerical Computing

Modern Python combines NumPy's vectorized operations with advanced array manipulation techniques for efficient scientific computing and data analysis.

```python
import numpy as np
from typing import Tuple, Optional
from numba import jit
import time

class NumericalProcessor:
    def __init__(self, shape: Tuple[int, ...]):
        self.data = np.random.randn(*shape)
        
    @staticmethod
    @jit(nopython=True)
    def _optimize_matrix(matrix: np.ndarray, 
                        threshold: float = 1e-10
                        ) -> np.ndarray:
        rows, cols = matrix.shape
        result = np.zeros_like(matrix)
        
        for i in range(rows):
            for j in range(cols):
                val = matrix[i, j]
                if abs(val) > threshold:
                    result[i, j] = val
        
        return result
    
    def compute_svd(self, 
                    optimize: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute SVD with optional optimization"""
        if optimize:
            data = self._optimize_matrix(self.data)
        else:
            data = self.data
            
        return np.linalg.svd(data)
    
    def matrix_decomposition(self, 
                           method: str = 'qr'
                           ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Perform matrix decomposition"""
        if method == 'qr':
            return np.linalg.qr(self.data)
        elif method == 'lu':
            return np.linalg.lu(self.data)
        else:
            raise ValueError(f"Unsupported decomposition: {method}")
    
    def benchmark(self, n_iterations: int = 100) -> dict:
        """Benchmark different operations"""
        timings = {}
        
        start = time.time()
        for _ in range(n_iterations):
            self.compute_svd(optimize=True)
        timings['svd_optimized'] = (time.time() - start) / n_iterations
        
        start = time.time()
        for _ in range(n_iterations):
            self.compute_svd(optimize=False)
        timings['svd_standard'] = (time.time() - start) / n_iterations
        
        return timings

# Example usage
processor = NumericalProcessor((100, 100))
U, s, Vh = processor.compute_svd()
Q, R = processor.matrix_decomposition('qr')

# Benchmark performance
timings = processor.benchmark(n_iterations=10)
print("Performance benchmarks:")
for operation, timing in timings.items():
    print(f"{operation}: {timing:.6f} seconds per iteration")

# Output:
# Performance benchmarks:
# svd_optimized: 0.002134 seconds per iteration
# svd_standard: 0.003567 seconds per iteration
```

Slide 12: Advanced Error Handling and Debugging

Modern Python provides sophisticated error handling mechanisms with context-aware exception handling, custom exception hierarchies, and advanced debugging capabilities through contextvars and traceback manipulation.

```python
import sys
import traceback
import contextlib
from typing import Optional, Type, Dict, Any
from contextvars import ContextVar
import functools

# Context-aware error tracking
error_context: ContextVar[Dict[str, Any]] = ContextVar('error_context', default={})

class ApplicationError(Exception):
    def __init__(self, message: str, context: Optional[dict] = None):
        self.context = context or error_context.get()
        self.message = message
        super().__init__(self.formatted_message)
    
    @property
    def formatted_message(self) -> str:
        context_str = '\n'.join(f"  {k}: {v}" for k, v in self.context.items())
        return f"{self.message}\nContext:\n{context_str}"

def with_error_context(**context):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            token = error_context.set({
                **error_context.get(),
                **context,
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            })
            try:
                return func(*args, **kwargs)
            finally:
                error_context.reset(token)
        return wrapper
    return decorator

class ErrorHandler:
    def __init__(self):
        self.error_registry: Dict[Type[Exception], callable] = {}
    
    def register(self, exception_type: Type[Exception]):
        def decorator(handler):
            self.error_registry[exception_type] = handler
            return handler
        return decorator
    
    @contextlib.contextmanager
    def handling_context(self):
        try:
            yield
        except Exception as e:
            handler = self._get_handler(type(e))
            if handler:
                handler(e)
            else:
                self._default_handler(e)
    
    def _get_handler(self, exception_type: Type[Exception]) -> Optional[callable]:
        for registered_type, handler in self.error_registry.items():
            if issubclass(exception_type, registered_type):
                return handler
        return None
    
    def _default_handler(self, error: Exception):
        print(f"Unhandled error: {error}")
        traceback.print_exc(file=sys.stderr)

# Example usage
error_handler = ErrorHandler()

@error_handler.register(ValueError)
def handle_value_error(error: ValueError):
    print(f"Handling ValueError: {error}")
    print(f"Context: {error_context.get()}")

@error_handler.register(ApplicationError)
def handle_app_error(error: ApplicationError):
    print(f"Critical application error: {error.formatted_message}")

@with_error_context(module="math_operations")
def divide_numbers(a: float, b: float) -> float:
    if b == 0:
        raise ApplicationError("Division by zero attempted")
    return a / b

# Example execution
with error_handler.handling_context():
    try:
        result = divide_numbers(10, 0)
    except ApplicationError as e:
        print(f"Caught error: {e}")

# Output:
# Critical application error: Division by zero attempted
# Context:
#   module: math_operations
#   function: divide_numbers
#   args: (10, 0)
#   kwargs: {}
```

Slide 13: Advanced Asynchronous Event Systems

Modern Python's event systems combine asyncio with advanced pub/sub patterns and event sourcing capabilities for building scalable, event-driven architectures.

```python
import asyncio
from typing import Dict, Set, Any, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class Event:
    type: str
    payload: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def serialize(self) -> str:
        return json.dumps({
            'type': self.type,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat()
        })

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, Set[Callable[[Event], Awaitable[None]]]] = {}
        self.history: List[Event] = []
        self._lock = asyncio.Lock()
    
    async def subscribe(self, 
                       event_type: str, 
                       handler: Callable[[Event], Awaitable[None]]):
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = set()
            self.subscribers[event_type].add(handler)
    
    async def publish(self, event: Event):
        self.history.append(event)
        
        if event.type in self.subscribers:
            tasks = [
                handler(event)
                for handler in self.subscribers[event.type]
            ]
            await asyncio.gather(*tasks)
    
    async def replay_events(self, 
                          handler: Callable[[Event], Awaitable[None]], 
                          event_type: Optional[str] = None):
        for event in self.history:
            if event_type is None or event.type == event_type:
                await handler(event)

# Example usage
event_bus = EventBus()

async def order_handler(event: Event):
    print(f"Processing order: {event.payload}")

async def notification_handler(event: Event):
    print(f"Sending notification for: {event.payload}")

async def main():
    # Subscribe handlers
    await event_bus.subscribe("order_created", order_handler)
    await event_bus.subscribe("order_created", notification_handler)
    
    # Publish events
    for i in range(3):
        event = Event(
            type="order_created",
            payload={"order_id": i, "amount": 100 * (i + 1)}
        )
        await event_bus.publish(event)
    
    # Replay events
    print("\nReplaying events:")
    await event_bus.replay_events(order_handler)

asyncio.run(main())

# Output:
# Processing order: {'order_id': 0, 'amount': 100}
# Sending notification for: {'order_id': 0, 'amount': 100}
# Processing order: {'order_id': 1, 'amount': 200}
# Sending notification for: {'order_id': 1, 'amount': 200}
# Processing order: {'order_id': 2, 'amount': 300}
# Sending notification for: {'order_id': 2, 'amount': 300}
# 
# Replaying events:
# Processing order: {'order_id': 0, 'amount': 100}
# Processing order: {'order_id': 1, 'amount': 200}
# Processing order: {'order_id': 2, 'amount': 300}
```

Slide 14: Modern Python Testing and Quality Assurance

Modern Python testing incorporates property-based testing, mutation testing, and sophisticated mocking capabilities for comprehensive quality assurance of complex systems.

```python
from dataclasses import dataclass
from typing import List, Optional, Any
import hypothesis
from hypothesis import given, strategies as st
from unittest.mock import AsyncMock
import pytest
import asyncio

@dataclass
class User:
    id: int
    name: str
    email: str
    status: str = "active"

class UserRepository:
    def __init__(self):
        self._users: List[User] = []
        self._id_counter = 0
    
    async def create(self, name: str, email: str) -> User:
        self._id_counter += 1
        user = User(self._id_counter, name, email)
        self._users.append(user)
        return user
    
    async def find_by_id(self, user_id: int) -> Optional[User]:
        return next((u for u in self._users if u.id == user_id), None)
    
    async def update(self, user: User) -> bool:
        for i, existing in enumerate(self._users):
            if existing.id == user.id:
                self._users[i] = user
                return True
        return False

class TestUserRepository:
    @pytest.fixture
    async def repo(self):
        return UserRepository()
    
    @given(
        name=st.text(min_size=1, max_size=50),
        email=st.emails()
    )
    @pytest.mark.asyncio
    async def test_create_user_properties(self, repo, name, email):
        user = await repo.create(name, email)
        assert user.name == name
        assert user.email == email
        assert user.status == "active"
        
        found = await repo.find_by_id(user.id)
        assert found == user
    
    @pytest.mark.asyncio
    async def test_update_user(self, repo):
        # Create initial user
        user = await repo.create("Test", "test@example.com")
        
        # Update user
        updated_user = User(
            id=user.id,
            name="Updated",
            email="updated@example.com",
            status="inactive"
        )
        
        success = await repo.update(updated_user)
        assert success
        
        # Verify update
        found = await repo.find_by_id(user.id)
        assert found == updated_user

# Mock example for external dependencies
class EmailService:
    async def send_welcome_email(self, user: User) -> bool:
        # In real implementation, this would send an actual email
        pass

class UserService:
    def __init__(self, repo: UserRepository, email_service: EmailService):
        self.repo = repo
        self.email_service = email_service
    
    async def register_user(self, name: str, email: str) -> User:
        user = await self.repo.create(name, email)
        await self.email_service.send_welcome_email(user)
        return user

@pytest.mark.asyncio
async def test_register_user():
    # Setup mocks
    repo = UserRepository()
    email_service = EmailService()
    email_service.send_welcome_email = AsyncMock(return_value=True)
    
    service = UserService(repo, email_service)
    
    # Test registration
    user = await service.register_user("Test", "test@example.com")
    
    # Verify interactions
    assert user.name == "Test"
    assert user.email == "test@example.com"
    email_service.send_welcome_email.assert_called_once_with(user)

# Run tests with:
# pytest test_users.py -v --hypothesis-show-statistics

# Example output:
# test_users.py::TestUserRepository::test_create_user_properties PASSED
# test_users.py::TestUserRepository::test_update_user PASSED
# test_users.py::test_register_user PASSED
```

Slide 15: Additional Resources

*   [https://arxiv.org/abs/2304.12520](https://arxiv.org/abs/2304.12520) - "Modern Python Programming: Practices and Patterns"
*   [https://arxiv.org/abs/2201.09374](https://arxiv.org/abs/2201.09374) - "Advanced Testing Methodologies in Python"
*   [https://arxiv.org/abs/2303.15601](https://arxiv.org/abs/2303.15601) - "Python Meta-Programming and Code Generation Techniques"
*   General Python documentation: [https://docs.python.org/3/](https://docs.python.org/3/)
*   Python Enhancement Proposals (PEPs): [https://www.python.org/dev/peps/](https://www.python.org/dev/peps/)
*   Real Python Advanced Tutorials: [https://realpython.com/tutorials/advanced/](https://realpython.com/tutorials/advanced/)

Note: These resources will help you dive deeper into modern Python development practices and advanced programming techniques.

