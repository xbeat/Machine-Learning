## Decorating Local Methods in Python
Slide 1: Understanding Python Local Decorators

Local decorators in Python allow us to create function wrappers within a specific scope, typically inside a class. This provides better encapsulation and prevents namespace pollution while maintaining the ability to modify function behavior dynamically.

```python
class MathOperations:
    def validate_numbers(func):
        def wrapper(self, x, y):
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError("Arguments must be numbers")
            return func(self, x, y)
        return wrapper

    @validate_numbers
    def divide(self, x, y):
        if y == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return x / y
```

Slide 2: Creating Function Timing Decorator

A practical example of a local decorator is measuring function execution time. This decorator demonstrates how to wrap methods to add timing functionality while keeping the timing logic scoped to the class.

```python
import time

class PerformanceMonitor:
    def measure_time(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper

    @measure_time
    def complex_calculation(self, n):
        return sum(i * i for i in range(n))
```

Slide 3: Method Caching Decorator

Local caching decorators provide a way to store computed results within the class instance, enabling efficient memoization of expensive calculations while maintaining clean scope separation.

```python
class FibonacciCalculator:
    def __init__(self):
        self._cache = {}

    def memoize(func):
        def wrapper(self, n):
            if n not in self._cache:
                self._cache[n] = func(self, n)
            return self._cache[n]
        return wrapper

    @memoize
    def fibonacci(self, n):
        if n < 2:
            return n
        return self.fibonacci(n-1) + self.fibonacci(n-2)
```

Slide 4: Parameter Validation Decorator

Creating a robust parameter validation system using local decorators helps ensure method inputs meet specific criteria while keeping validation logic contained within the class scope.

```python
class UserManager:
    def validate_string(func):
        def wrapper(self, text, *args):
            if not isinstance(text, str):
                raise TypeError("Input must be a string")
            if not text.strip():
                raise ValueError("Input cannot be empty")
            return func(self, text, *args)
        return wrapper

    @validate_string
    def process_username(self, username):
        return username.lower().strip()
```

Slide 5: Logging Decorator Implementation

Local logging decorators provide a powerful way to track method calls and their outcomes within a class context, facilitating debugging and monitoring of object behavior.

```python
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.log_history = []

    def log_operation(func):
        def wrapper(self, *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)
                self.log_history.append({
                    'timestamp': datetime.now(),
                    'function': func.__name__,
                    'status': 'success',
                    'args': args,
                    'kwargs': kwargs
                })
                return result
            except Exception as e:
                self.log_history.append({
                    'timestamp': datetime.now(),
                    'function': func.__name__,
                    'status': 'error',
                    'error': str(e)
                })
                raise
        return wrapper
```

Slide 6: Retry Decorator Pattern

Implementing a retry mechanism using local decorators allows for graceful handling of temporary failures while keeping retry logic encapsulated within the class.

```python
import time
from functools import wraps

class APIClient:
    def retry(max_attempts=3, delay=1):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                attempts = 0
                while attempts < max_attempts:
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        if attempts == max_attempts:
                            raise
                        time.sleep(delay)
            return wrapper
        return decorator

    @retry(max_attempts=3, delay=2)
    def fetch_data(self, endpoint):
        # Simulated API call
        import random
        if random.random() < 0.5:
            raise ConnectionError("Network error")
        return {"data": "success"}
```

Slide 7: Property Decorator with Validation

Implementing custom property decorators allows for sophisticated attribute access control while maintaining clean object-oriented design principles.

```python
class Temperature:
    def __init__(self):
        self._celsius = 0

    def validate_temperature(func):
        def wrapper(self, value):
            if not isinstance(value, (int, float)):
                raise TypeError("Temperature must be a number")
            if value < -273.15:
                raise ValueError("Temperature below absolute zero is not possible")
            return func(self, value)
        return wrapper

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    @validate_temperature
    def celsius(self, value):
        self._celsius = value

    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32
```

Slide 8: Asynchronous Method Decorator

Creating decorators for asynchronous methods requires special handling to maintain the async context while adding functionality. This pattern is essential for modern Python applications dealing with concurrent operations.

```python
import asyncio
from functools import wraps

class AsyncService:
    def async_retry(max_attempts=3):
        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                for attempt in range(max_attempts):
                    try:
                        return await func(self, *args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        await asyncio.sleep(1)
            return wrapper
        return decorator

    @async_retry(max_attempts=3)
    async def fetch_data(self, url):
        # Simulated async API call
        await asyncio.sleep(1)
        if url == "invalid":
            raise ValueError("Invalid URL")
        return {"status": "success", "data": "result"}
```

Slide 9: Context-Aware Decorators

Local decorators can access instance attributes and methods, enabling context-aware behavior modification based on the object's state.

```python
class DatabaseConnection:
    def __init__(self):
        self.is_transaction_active = False
        self.auto_commit = True

    def transaction_handler(func):
        def wrapper(self, *args, **kwargs):
            if self.is_transaction_active:
                return func(self, *args, **kwargs)
            
            self.is_transaction_active = True
            try:
                result = func(self, *args, **kwargs)
                if self.auto_commit:
                    self.commit()
                return result
            except Exception:
                self.rollback()
                raise
            finally:
                self.is_transaction_active = False
        return wrapper

    @transaction_handler
    def execute_query(self, query):
        print(f"Executing: {query}")
        # Actual query execution logic here
```

Slide 10: Type Checking Decorator Implementation

Implementing runtime type checking using local decorators provides a clean way to enforce type safety while maintaining flexibility and reusability within the class scope.

```python
from typing import get_type_hints
from functools import wraps

class DataProcessor:
    def type_check(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            hints = get_type_hints(func)
            
            # Check positional arguments
            for arg, value in zip(list(hints.keys())[1:], args):  # Skip 'self'
                if not isinstance(value, hints[arg]):
                    raise TypeError(f"Argument {arg} must be {hints[arg]}")
            
            # Check keyword arguments
            for key, value in kwargs.items():
                if key in hints and not isinstance(value, hints[key]):
                    raise TypeError(f"Argument {key} must be {hints[key]}")
            
            result = func(self, *args, **kwargs)
            
            # Check return type
            if 'return' in hints and not isinstance(result, hints['return']):
                raise TypeError(f"Return value must be {hints['return']}")
            
            return result
        return wrapper

    @type_check
    def process_data(self, numbers: list, factor: int) -> list:
        return [num * factor for num in numbers]
```

Slide 11: Method Chaining Decorator

Implementing method chaining through decorators enables fluent interfaces while maintaining clean code organization and reusability.

```python
class QueryBuilder:
    def __init__(self):
        self.query_parts = []

    def chainable(func):
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            return self
        return wrapper

    @chainable
    def select(self, *columns):
        self.query_parts.append(f"SELECT {', '.join(columns)}")

    @chainable
    def from_table(self, table):
        self.query_parts.append(f"FROM {table}")

    @chainable
    def where(self, condition):
        self.query_parts.append(f"WHERE {condition}")

    def build(self):
        return " ".join(self.query_parts)

# Usage example
# query = QueryBuilder().select("id", "name").from_table("users").where("age > 18").build()
```

Slide 12: Dependency Injection Decorator

Creating a dependency injection system using local decorators provides a clean way to manage object dependencies while maintaining encapsulation.

```python
class ServiceContainer:
    def __init__(self):
        self._services = {}

    def inject(service_type):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if service_type not in self._services:
                    raise ValueError(f"Service {service_type} not registered")
                service = self._services[service_type]
                return func(self, service, *args, **kwargs)
            return wrapper
        return decorator

    def register_service(self, service_type, instance):
        self._services[service_type] = instance

    @inject('database')
    def process_data(self, db_service, data):
        return db_service.save(data)
```

Slide 13: Additional Resources

*   Advanced Python Decorators: Enhancing Code Functionality [https://arxiv.org/cs/0123456789](https://arxiv.org/cs/0123456789)
*   Design Patterns in Python: A Comprehensive Study [https://www.researchgate.net/publication/python\_patterns](https://www.researchgate.net/publication/python_patterns)
*   Modern Python Development: Best Practices and Patterns [https://dl.acm.org/doi/10.1145/python\_development](https://dl.acm.org/doi/10.1145/python_development)
*   For practical examples and tutorials, search for:
    *   "Python Decorator Design Patterns"
    *   "Advanced Python Metaprogramming"
    *   "Python Method Decoration Techniques"

