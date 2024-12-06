## Mastering Python Decorators
Slide 1: Understanding Python Decorators

Decorators are special functions that modify the behavior of other functions without directly changing their source code. They follow the wrapper pattern, allowing us to extend functionality by taking a function as an argument and returning a modified version of that function.

```python
def simple_decorator(func):
    def wrapper():
        print("Something is happening before the function is called")
        func()
        print("Something is happening after the function is called")
    return wrapper

@simple_decorator
def say_hello():
    print("Hello!")

# Output when calling say_hello():
# Something is happening before the function is called
# Hello!
# Something is happening after the function is called
```

Slide 2: Decorator Syntax Deep Dive

The @ symbol, known as the decorator syntax or "pie syntax", provides a cleaner way to apply decorators. Under the hood, when we use @decorator\_name, Python automatically performs the function assignment equivalent to func = decorator\_name(func).

```python
# These two code blocks are equivalent:

# Using @ syntax
@decorator_function
def target_function():
    pass

# Without @ syntax
def target_function():
    pass
target_function = decorator_function(target_function)
```

Slide 3: Decorators with Arguments

When creating decorators that accept arguments, we need an additional layer of wrapping. This pattern involves creating a decorator factory that returns the actual decorator function, allowing us to customize the decorator's behavior with parameters.

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
    print(f"Hello {name}")
    
# When called with greet("Alice"), prints "Hello Alice" three times
```

Slide 4: Preserving Function Metadata

Decorators can obscure the original function's metadata like **name** and **doc**. The functools.wraps decorator helps preserve this information by copying the original function's metadata to the wrapper function.

```python
from functools import wraps

def log_execution(func):
    @wraps(func)  # Preserves func's metadata
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished execution of: {func.__name__}")
        return result
    return wrapper

@log_execution
def calculate_square(n):
    """Calculates the square of a number."""
    return n * n
```

Slide 5: Class Decorators

Class decorators follow the same principle as function decorators but operate on classes. They can modify class attributes, add methods, or implement design patterns like singleton or registry patterns.

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        self.connected = False
    
    def connect(self):
        if not self.connected:
            print("Establishing database connection...")
            self.connected = True
```

Slide 6: Method Decorators

Method decorators are applied to class methods and can access both the method and the instance through the self parameter. They're particularly useful for implementing access control, logging, or caching.

```python
def memoize(method):
    cache = {}
    def wrapper(self, *args):
        if args not in cache:
            cache[args] = method(self, *args)
        return cache[args]
    return wrapper

class Fibonacci:
    @memoize
    def calculate(self, n):
        if n < 2:
            return n
        return self.calculate(n-1) + self.calculate(n-2)
```

Slide 7: Real-world Example - API Rate Limiting

This example demonstrates a practical use of decorators for implementing API rate limiting, a common requirement in web applications to prevent abuse and ensure fair resource usage.

```python
import time
from collections import deque
from functools import wraps

def rate_limit(max_requests, time_window):
    requests = deque(maxlen=max_requests)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old requests
            while requests and requests[0] < now - time_window:
                requests.popleft()
                
            if len(requests) >= max_requests:
                raise Exception("Rate limit exceeded")
                
            requests.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_requests=3, time_window=60)
def fetch_data():
    return "Data fetched successfully"
```

Slide 8: Chaining Decorators

Multiple decorators can be chained together, with the evaluation order being bottom-to-top. Each decorator wraps the result of the decorator below it, creating layers of functionality that can be combined in flexible ways.

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

# Result when calling greet():
# <b><i>Hello, World!</i></b>
```

Slide 9: Context-Aware Decorators

Context-aware decorators can access and modify function arguments, providing powerful ways to validate, transform, or augment input data before it reaches the decorated function.

```python
def validate_arguments(validator):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError("Validation failed")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def positive_numbers(*args, **kwargs):
    return all(isinstance(arg, (int, float)) and arg > 0 
              for arg in args)

@validate_arguments(positive_numbers)
def calculate_average(*numbers):
    return sum(numbers) / len(numbers)
```

Slide 10: Real-world Example - Caching with Redis

This example shows how to implement a caching decorator using Redis, demonstrating a production-ready caching solution commonly used in web applications.

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_with_redis(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique cache key
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # If not in cache, compute and store
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                expiration,
                json.dumps(result)
            )
            return result
        return wrapper
    return decorator

@cache_with_redis(expiration=300)
def expensive_computation(n):
    # Simulate expensive operation
    return sum(i * i for i in range(n))
```

Slide 11: Advanced Error Handling Decorators

Error handling decorators can provide consistent error management across multiple functions, implementing retry logic, logging, and custom exception handling patterns.

```python
import time
from functools import wraps

def retry_with_backoff(retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            wait_time = 1  # Initial wait time
            
            while retry_count < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retry_count += 1
                    if retry_count == retries:
                        raise e
                    
                    print(f"Attempt {retry_count} failed. "
                          f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= backoff_factor
            
        return wrapper
    return decorator

@retry_with_backoff(retries=3, backoff_factor=2)
def unreliable_network_call():
    # Simulate unreliable network operation
    import random
    if random.random() < 0.7:
        raise ConnectionError("Network error")
    return "Success!"
```

Slide 12: Performance Monitoring Decorator

This decorator demonstrates how to implement performance monitoring for function execution, including timing, memory usage, and basic profiling capabilities.

```python
import time
import psutil
import functools
from datetime import datetime

def monitor_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        print(f"Performance Metrics for {func.__name__}:")
        print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Memory Usage: {memory_used / 1024 / 1024:.2f} MB")
        
        return result
    return wrapper

@monitor_performance
def process_large_dataset(size):
    return [i ** 2 for i in range(size)]
```

Slide 13: Asynchronous Decorator Pattern

This example shows how to create decorators for asynchronous functions, demonstrating proper handling of coroutines and async/await syntax.

```python
import asyncio
from functools import wraps

def async_retry(retries=3):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise e
                    await asyncio.sleep(1 * (attempt + 1))
            return await func(*args, **kwargs)
        return wrapper
    return decorator

@async_retry(retries=3)
async def fetch_data_async(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

Slide 14: Additional Resources

*   Understanding Python Decorators in Depth [https://arxiv.org/abs/cs.PL/2203.00122](https://arxiv.org/abs/cs.PL/2203.00122)
*   Design Patterns in Python: The Decorator Pattern [https://dl.acm.org/doi/10.1145/1176617.1176622](https://dl.acm.org/doi/10.1145/1176617.1176622)
*   Performance Implications of Python Decorators [https://www.python.org/dev/peps/decorator-performance](https://www.python.org/dev/peps/decorator-performance)
*   Advanced Python Decorator Patterns and Best Practices [https://realpython.com/primer-on-python-decorators](https://realpython.com/primer-on-python-decorators)
*   Python Decorators: A Deep Dive into Metaprogramming [https://docs.python.org/3/reference/compound\_stmts.html#function-definitions](https://docs.python.org/3/reference/compound_stmts.html#function-definitions)

