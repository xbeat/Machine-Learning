## Python Decorators Basic and Advanced Usage
Slide 1: Understanding Python Decorators

A decorator is a design pattern in Python that allows modifying the behavior of functions or classes without directly changing their source code. Decorators provide a clean and elegant way to wrap additional functionality around existing code, following the principle of separation of concerns.

```python
# Basic decorator structure
def my_decorator(func):
    def wrapper():
        print("Something happens before the function is called")
        func()
        print("Something happens after the function is called")
    return wrapper

# Using the decorator
@my_decorator
def say_hello():
    print("Hello!")

# Output when calling say_hello():
# Something happens before the function is called
# Hello!
# Something happens after the function is called
```

Slide 2: Decorator Syntax Deep Dive

The @ symbol in Python is syntactic sugar for function decoration. When we use @decorator\_name above a function definition, Python automatically applies the decorator to the function. Understanding this syntax is crucial for both using and creating decorators effectively.

```python
# These two code blocks are equivalent:

# Method 1: Using @ syntax
@my_decorator
def function_a():
    pass

# Method 2: Manual decoration
def function_b():
    pass
function_b = my_decorator(function_b)

# Both methods produce identical results
# The @ syntax is preferred for clarity and readability
```

Slide 3: Decorators with Arguments

Decorators can accept arguments that modify their behavior, requiring an additional level of nesting in the decorator definition. This pattern allows for more flexible and reusable decorator implementations across different scenarios.

```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    print(f"Hello, {name}")
    
# When called: greet("Alice")
# Output:
# Hello, Alice
# Hello, Alice
# Hello, Alice
```

Slide 4: Class-Based Decorators

Class-based decorators provide an object-oriented approach to function modification. They're particularly useful when the decorator needs to maintain state or when you want to provide additional methods alongside the wrapped function.

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0
        
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call count: {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def my_function():
    print("Function executed")

# Calling my_function() three times:
# Call count: 1
# Function executed
# Call count: 2
# Function executed
# Call count: 3
# Function executed
```

Slide 5: Real-World Example - Performance Monitoring

In production environments, monitoring function execution time is crucial for performance optimization. This practical example demonstrates how decorators can be used to implement timing functionality without modifying existing code.

```python
import time
import functools

def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

@measure_time
def complex_calculation(n):
    return sum(i * i for i in range(n))

# Usage:
result = complex_calculation(1000000)
# Output: complex_calculation took 0.1234 seconds to execute
```

Slide 6: Function Metadata Preservation

When creating decorators, it's important to preserve the original function's metadata such as docstrings and function names. The functools.wraps decorator helps maintain this information for proper debugging and documentation.

```python
from functools import wraps

def log_function_call(func):
    @wraps(func)  # Preserves metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} completed")
        return result
    return wrapper

@log_function_call
def calculate_square(n):
    """Returns the square of a number."""
    return n * n

# Metadata is preserved
print(calculate_square.__name__)  # Output: calculate_square
print(calculate_square.__doc__)   # Output: Returns the square of a number.
```

Slide 7: Decorator Chaining

Multiple decorators can be applied to a single function, creating a chain of transformations. The decorators are applied from bottom to top, with each decorator wrapping the result of the previous one in sequence.

```python
def bold(func):
    def wrapper():
        return f"<b>{func()}</b>"
    return wrapper

def italic(func):
    def wrapper():
        return f"<i>{func()}</i>"
    return wrapper

@bold
@italic
def greet():
    return "Hello, World!"

print(greet())  
# Output: <b><i>Hello, World!</i></b>
# Order matters: @bold(@italic(greet()))
```

Slide 8: Stateful Decorators

Decorators can maintain state across function calls, making them useful for caching, counting, or tracking various metrics. This example implements a memoization decorator for optimizing recursive functions.

```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache  # Expose cache for inspection
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Instant result due to caching
print(fibonacci.cache)  # View cached values
```

Slide 9: Decorators with Parameters for Class Methods

When decorating class methods, we need to handle the self parameter correctly. This example shows how to create a decorator that can work with both standalone functions and class methods.

```python
def validate_types(*types):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Skip self for methods
            actual_args = args[1:] if args and isinstance(args[0], object) else args
            for arg, expected_type in zip(actual_args, types):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Expected {expected_type}, got {type(arg)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Calculator:
    @validate_types(int, int)
    def add(self, x, y):
        return x + y

calc = Calculator()
print(calc.add(1, 2))  # Output: 3
# calc.add(1, "2")  # Raises TypeError
```

Slide 10: Real-World Example - API Rate Limiting

This practical implementation shows how decorators can be used to implement rate limiting for API endpoints, a common requirement in web applications and microservices.

```python
import time
from collections import deque
from functools import wraps

class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and self.calls[0] <= now - self.time_window:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")

            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

# Example usage
@RateLimiter(max_calls=3, time_window=1)  # 3 calls per second
def api_endpoint():
    return "API response"

# Test rate limiting
for _ in range(4):
    try:
        print(api_endpoint())
    except Exception as e:
        print(f"Error: {e}")
```

Slide 11: Advanced Error Handling in Decorators

Decorators can provide sophisticated error handling and retry logic for functions that might fail. This implementation shows how to create a robust retry mechanism with exponential backoff for handling transient failures.

```python
import time
from functools import wraps
import random

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = retries
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise e
                    sleep_time = (backoff_in_seconds * 2 ** retry_count + 
                                random.uniform(0, 1))
                    print(f"Retry {retry_count} after {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            return None
        return wrapper
    return decorator

@retry_with_backoff(retries=3, backoff_in_seconds=1)
def unstable_network_call():
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return "Success!"

# Example usage
try:
    result = unstable_network_call()
    print(f"Result: {result}")
except ConnectionError as e:
    print(f"Final failure: {e}")
```

Slide 12: Context-Aware Decorators

Context-aware decorators can modify their behavior based on runtime conditions or configuration settings. This pattern is particularly useful for feature flags, debugging, and environment-specific behaviors.

```python
import os
from functools import wraps
from typing import Optional, Callable

class FeatureToggle:
    def __init__(self, feature_name: str, fallback: Optional[Callable] = None):
        self.feature_name = feature_name
        self.fallback = fallback

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.getenv(f"FEATURE_{self.feature_name.upper()}", "false").lower() == "true":
                return func(*args, **kwargs)
            elif self.fallback:
                return self.fallback(*args, **kwargs)
            else:
                raise NotImplementedError(f"Feature {self.feature_name} is not enabled")
        return wrapper

def old_implementation(x: int, y: int) -> int:
    return x + y

@FeatureToggle("new_math", fallback=old_implementation)
def advanced_math(x: int, y: int) -> int:
    return (x ** 2 + y ** 2) ** 0.5

# Usage:
# os.environ["FEATURE_NEW_MATH"] = "true"
result = advanced_math(3, 4)
print(f"Result: {result}")  # Uses new or old implementation based on env var
```

Slide 13: Performance Optimization with Descriptors

This advanced example combines decorators with descriptors to create a powerful caching mechanism for class properties, demonstrating how decorators can interact with Python's descriptor protocol.

```python
class CachedProperty:
    def __init__(self, func):
        self.func = func
        self.cache = {}
        
    def __get__(self, obj, cls):
        if obj is None:
            return self
            
        if obj not in self.cache:
            self.cache[obj] = self.func(obj)
            
        return self.cache[obj]
        
    def __set__(self, obj, value):
        raise AttributeError("Can't modify cached property")

class DataProcessor:
    def __init__(self, data):
        self.data = data
        
    @CachedProperty
    def expensive_calculation(self):
        print("Performing expensive calculation...")
        return sum(x * x for x in self.data)

# Usage demonstration
processor = DataProcessor(range(1000000))
print(processor.expensive_calculation)  # First call: performs calculation
print(processor.expensive_calculation)  # Second call: returns cached result
```

Slide 14: Additional Resources

*   Performance Optimization with Python Decorators
    *   Search: "Python Decorator Performance Optimization Patterns"
    *   [https://realpython.com/primer-on-python-decorators/](https://realpython.com/primer-on-python-decorators/)
*   Advanced Python Decorators: From Decorators to Frameworks
    *   [https://docs.python.org/3/reference/datamodel.html#decorators](https://docs.python.org/3/reference/datamodel.html#decorators)
*   Understanding Python Descriptors and Meta-Programming
    *   [https://python-course.eu/oop/python-descriptors.php](https://python-course.eu/oop/python-descriptors.php)
*   Design Patterns in Python: The Decorator Pattern
    *   Search: "Python Design Patterns Decorator Implementation"
*   Python Enhancement Proposals (PEPs) Related to Decorators
    *   [https://peps.python.org/pep-0318/](https://peps.python.org/pep-0318/)

