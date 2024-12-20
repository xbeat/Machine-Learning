## Python Error Handling LBYL vs EAFP Approaches
Slide 1: Understanding LBYL vs EAFP Programming Paradigms

Python offers two main approaches to handling potential errors: Look Before You Leap (LBYL) and Easier to Ask for Forgiveness than Permission (EAFP). These paradigms represent fundamentally different philosophies in error handling and code design.

```python
# LBYL Example (Look Before You Leap)
def divide_lbyl(x, y):
    if y != 0:  # Check condition before proceeding
        return x / y
    else:
        return "Cannot divide by zero"

# EAFP Example (Easier to Ask for Forgiveness than Permission)
def divide_eafp(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return "Cannot divide by zero"

# Example usage
print(divide_lbyl(10, 2))  # Output: 5.0
print(divide_lbyl(10, 0))  # Output: Cannot divide by zero
print(divide_eafp(10, 2))  # Output: 5.0
print(divide_eafp(10, 0))  # Output: Cannot divide by zero
```

Slide 2: Performance Analysis of LBYL vs EAFP

EAFP generally performs better in Python because it aligns with Python's internal design. The overhead of checking conditions in LBYL can accumulate, especially when dealing with multiple conditions or nested structures.

```python
import timeit
import statistics

def benchmark_approaches():
    # Setup dictionary for testing
    data = {'key1': 'value1', 'key2': 'value2'}
    
    # LBYL approach
    def lbyl_test():
        if 'key1' in data and data['key1'] is not None:
            return data['key1']
        return None
    
    # EAFP approach
    def eafp_test():
        try:
            return data['key1']
        except (KeyError, TypeError):
            return None
    
    # Benchmark both approaches
    lbyl_time = timeit.repeat(lbyl_test, number=1000000)
    eafp_time = timeit.repeat(eafp_test, number=1000000)
    
    print(f"LBYL average: {statistics.mean(lbyl_time):.6f} seconds")
    print(f"EAFP average: {statistics.mean(eafp_time):.6f} seconds")

benchmark_approaches()
```

Slide 3: Context Managers and EAFP

Context managers exemplify EAFP principles by handling resource management and cleanup automatically. This approach ensures proper resource handling even if exceptions occur during execution.

```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        
    def __enter__(self):
        print(f"Connecting to database: {self.connection_string}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        if exc_type is not None:
            print(f"An error occurred: {exc_val}")
        return False  # Propagate exceptions
        
# Usage example
try:
    with DatabaseConnection("postgresql://localhost:5432/mydb") as db:
        raise ValueError("Simulated error")
except ValueError as e:
    print(f"Caught error: {e}")
```

Slide 4: Practical Implementation in File Handling

File operations demonstrate the superiority of EAFP in real-world scenarios. This implementation shows how to handle multiple potential errors while maintaining clean, readable code.

```python
def process_file_eafp(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            numbers = [float(num) for num in content.split()]
            return sum(numbers) / len(numbers)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except ValueError:
        print("File contains invalid numbers")
        return None
    except ZeroDivisionError:
        print("File is empty")
        return None

# Example usage with different scenarios
print(process_file_eafp("valid.txt"))    # Processes valid file
print(process_file_eafp("missing.txt"))  # Handles missing file
print(process_file_eafp("invalid.txt"))  # Handles invalid content
```

Slide 5: Dynamic Attribute Access Using EAFP

The EAFP paradigm shines when dealing with dynamic attribute access and method calls, providing a more pythonic and efficient approach to handling object interactions.

```python
class DynamicObject:
    def __init__(self):
        self.existing_attr = "I exist"
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return f"Created dynamic attribute: {name}"

obj = DynamicObject()

# Demonstrate dynamic attribute access
print(obj.existing_attr)      # Output: I exist
print(obj.nonexistent_attr)   # Output: Created dynamic attribute: nonexistent_attr

# Example with method calls
def safe_call(obj, method_name, *args, **kwargs):
    try:
        method = getattr(obj, method_name)
        return method(*args, **kwargs)
    except AttributeError:
        return f"Method {method_name} not found"
    except Exception as e:
        return f"Error executing {method_name}: {str(e)}"

print(safe_call(obj, "existing_attr"))
print(safe_call(obj, "unknown_method"))
```

Slide 6: Exception Hierarchy for Custom Error Handling

Understanding and implementing custom exception hierarchies enables robust error handling in larger applications. This implementation demonstrates how to create domain-specific exceptions while maintaining the EAFP philosophy.

```python
class DataProcessingError(Exception):
    """Base exception for data processing errors"""
    pass

class ValidationError(DataProcessingError):
    """Raised when data validation fails"""
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class ProcessingError(DataProcessingError):
    """Raised when data processing fails"""
    pass

def process_data(data):
    try:
        if not isinstance(data, dict):
            raise ValidationError("input", "Must be a dictionary")
        if "age" in data and (not isinstance(data["age"], int) or data["age"] < 0):
            raise ValidationError("age", "Must be a positive integer")
            
        # Process the data
        return {"processed": True, "data": data}
    
    except ValidationError as e:
        print(f"Validation failed: {e}")
        return None
    except Exception as e:
        raise ProcessingError(f"Unexpected error: {str(e)}")

# Example usage
print(process_data({"name": "John", "age": 30}))  # Valid data
print(process_data({"name": "John", "age": -5}))  # Invalid age
print(process_data([1, 2, 3]))  # Invalid input type
```

Slide 7: Real-world Application: Data Processing Pipeline

A practical implementation of EAFP principles in a data processing pipeline, demonstrating how to handle various error conditions while maintaining code readability and robustness.

```python
import json
from datetime import datetime

class DataPipeline:
    def __init__(self):
        self.transformations = []
        self.error_log = []
    
    def add_transformation(self, func):
        self.transformations.append(func)
    
    def log_error(self, step, error):
        self.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'error': str(error)
        })
    
    def process(self, data):
        result = data
        for idx, transform in enumerate(self.transformations):
            try:
                result = transform(result)
            except Exception as e:
                self.log_error(f"Step {idx}", e)
                return None
        return result

# Example transformations
def validate_json(data):
    return json.loads(data) if isinstance(data, str) else data

def normalize_dates(data):
    if 'date' in data:
        data['date'] = datetime.strptime(
            data['date'], '%Y-%m-%d'
        ).isoformat()
    return data

# Usage example
pipeline = DataPipeline()
pipeline.add_transformation(validate_json)
pipeline.add_transformation(normalize_dates)

# Process valid data
valid_data = '{"date": "2024-01-01", "value": 100}'
print(pipeline.process(valid_data))

# Process invalid data
invalid_data = '{"date": "invalid-date", "value": 100}'
print(pipeline.process(invalid_data))
print(f"Errors: {pipeline.error_log}")
```

Slide 8: Decorators with EAFP for Function Validation

Implementing decorators using EAFP principles provides a clean way to handle function input validation and error handling without cluttering the main function logic.

```python
from functools import wraps
import time

def retry_with_backoff(max_retries=3, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            
            raise last_exception
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, initial_delay=1)
def unstable_network_call(url):
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return f"Success: {url}"

# Example usage
try:
    result = unstable_network_call("http://example.com")
    print(result)
except ConnectionError as e:
    print(f"Final failure: {e}")
```

Slide 9: Advanced Exception Chaining

Exception chaining allows preservation of the original error context while raising more specific exceptions. This implementation demonstrates sophisticated error handling in a modular system.

```python
class DatabaseError(Exception):
    pass

class NetworkError(Exception):
    pass

class ServiceError(Exception):
    pass

def fetch_data_from_db():
    try:
        # Simulate database operation
        raise ConnectionError("Database connection failed")
    except ConnectionError as e:
        raise DatabaseError("Database operation failed") from e

def fetch_from_network():
    try:
        # Simulate network call
        raise TimeoutError("Network timeout")
    except TimeoutError as e:
        raise NetworkError("Network operation failed") from e

def service_operation():
    try:
        fetch_data_from_db()
        fetch_from_network()
    except (DatabaseError, NetworkError) as e:
        raise ServiceError("Service operation failed") from e

# Example usage with full traceback
try:
    service_operation()
except ServiceError as e:
    print(f"Top level error: {e}")
    print("\nOriginal cause:", e.__cause__)
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
```

Slide 10: Context-Aware Error Handling

Implementing context-aware error handling allows for dynamic error responses based on the execution environment and operation context.

```python
import contextlib
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum, auto

class Environment(Enum):
    DEVELOPMENT = auto()
    STAGING = auto()
    PRODUCTION = auto()

@dataclass
class ExecutionContext:
    environment: Environment
    debug: bool
    user_id: Optional[str] = None

class ContextualErrorHandler:
    def __init__(self, context: ExecutionContext):
        self.context = context
        self.errors = []
    
    @contextlib.contextmanager
    def handle_errors(self, operation_name: str):
        try:
            yield
        except Exception as e:
            self.errors.append({
                'operation': operation_name,
                'error': str(e),
                'type': type(e).__name__
            })
            
            if self.context.environment == Environment.DEVELOPMENT:
                print(f"Debug info for {operation_name}:", str(e))
            elif self.context.environment == Environment.PRODUCTION:
                if self.context.user_id:
                    print(f"Error for user {self.context.user_id}")
                else:
                    print("System error occurred")
            raise

# Example usage
context = ExecutionContext(
    environment=Environment.DEVELOPMENT,
    debug=True,
    user_id="user123"
)

handler = ContextualErrorHandler(context)

def risky_operation():
    with handler.handle_errors("data_processing"):
        raise ValueError("Invalid data format")

try:
    risky_operation()
except ValueError:
    print("Error log:", handler.errors)
```

Slide 11: Asynchronous Error Handling

Managing errors in asynchronous code requires special attention to ensure proper error propagation and handling across coroutines.

```python
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring async resource")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing async resource")
        if exc_type is not None:
            print(f"Handled error: {exc_val}")
        
    async def process(self):
        await asyncio.sleep(1)
        return "Processed data"

async def process_with_timeout(timeout: float) -> Optional[str]:
    try:
        async with AsyncExitStack() as stack:
            resource = await stack.enter_async_context(AsyncResource())
            
            # Run with timeout
            result = await asyncio.wait_for(
                resource.process(),
                timeout=timeout
            )
            return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
async def main():
    # Successful case
    result1 = await process_with_timeout(2.0)
    print("Result 1:", result1)
    
    # Timeout case
    result2 = await process_with_timeout(0.5)
    print("Result 2:", result2)

asyncio.run(main())
```

Slide 12: Property-Based Testing with EAFP

Property-based testing combined with EAFP principles ensures robust code behavior across a wide range of inputs while maintaining pythonic error handling patterns.

```python
from hypothesis import given, strategies as st
from typing import List, Any
import math

class NumberProcessor:
    def process_numbers(self, numbers: List[float]) -> float:
        try:
            cleaned = [n for n in numbers if isinstance(n, (int, float))]
            if not cleaned:
                raise ValueError("No valid numbers provided")
            return sum(cleaned) / len(cleaned)
        except TypeError:
            raise ValueError("Invalid input type")

    def calculate_statistics(self, numbers: List[float]) -> dict:
        try:
            mean = self.process_numbers(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            return {
                'mean': mean,
                'std_dev': math.sqrt(variance),
                'count': len(numbers)
            }
        except Exception as e:
            return {'error': str(e)}

# Property-based tests
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_processor(numbers):
    processor = NumberProcessor()
    try:
        result = processor.calculate_statistics(numbers)
        assert 'mean' in result
        assert 'std_dev' in result
        assert result['count'] == len(numbers)
    except Exception as e:
        assert 'error' in result

# Example usage
processor = NumberProcessor()
print(processor.calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0]))
print(processor.calculate_statistics([]))  # Handles empty list
print(processor.calculate_statistics(['invalid']))  # Handles invalid input
```

Slide 13: Real-time Data Validation with EAFP

Implementation of real-time data validation system using EAFP principles, demonstrating how to handle streaming data with complex validation requirements.

```python
from datetime import datetime
from typing import Generator, Dict, Any
import json

class DataValidator:
    def __init__(self):
        self._validators = {}
        self.setup_validators()
    
    def setup_validators(self):
        def validate_timestamp(value: str) -> datetime:
            return datetime.fromisoformat(value)
            
        def validate_numeric(value: Any) -> float:
            return float(value)
            
        self._validators = {
            'timestamp': validate_timestamp,
            'value': validate_numeric
        }
    
    def validate_stream(self, data_stream: Generator[Dict, None, None]):
        for item in data_stream:
            try:
                validated = {}
                for field, validator in self._validators.items():
                    if field in item:
                        validated[field] = validator(item[field])
                yield validated
            except Exception as e:
                yield {'error': f"Validation failed: {str(e)}", 'data': item}

# Example usage
def generate_test_data():
    test_data = [
        {'timestamp': '2024-01-01T12:00:00', 'value': '42.5'},
        {'timestamp': 'invalid', 'value': '42.5'},
        {'timestamp': '2024-01-01T12:00:00', 'value': 'not_a_number'},
        {'timestamp': '2024-01-01T12:00:00', 'value': '43.2'}
    ]
    for item in test_data:
        yield item

validator = DataValidator()
for result in validator.validate_stream(generate_test_data()):
    print(json.dumps(result, default=str, indent=2))
```

Slide 14: Additional Resources

*   Research papers and documentation for deeper understanding:

*   arxiv.org/abs/computing/0702072 - "Exception Handling: Issues and a Proposed Notation"
*   [https://peps.python.org/pep-0463/](https://peps.python.org/pep-0463/) - Python Exception Handling PEP
*   [https://dl.acm.org/doi/10.1145/1988042.1988046](https://dl.acm.org/doi/10.1145/1988042.1988046) - "Exception Handling: A Field Study in Java"

*   Recommended reading for advanced concepts:

*   [https://www.python.org/dev/peps/pep-3134/](https://www.python.org/dev/peps/pep-3134/) - Exception Chaining and Embedded Tracebacks
*   [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html) - Python Exception Handling Documentation
*   [https://google.github.io/styleguide/pyguide.html#24-exceptions](https://google.github.io/styleguide/pyguide.html#24-exceptions) - Google Python Style Guide on Exceptions

*   Community resources:

*   [https://stackoverflow.com/questions/tagged/python+exception-handling](https://stackoverflow.com/questions/tagged/python+exception-handling) - Stack Overflow Python Exception Handling
*   [https://realpython.com/python-exceptions/](https://realpython.com/python-exceptions/) - Real Python Exception Handling Guide
*   [https://pypi.org/project/better-exceptions/](https://pypi.org/project/better-exceptions/) - Better Exceptions Package Documentation

