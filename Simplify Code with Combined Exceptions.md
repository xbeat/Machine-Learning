## Simplify Code with Combined Exceptions
Slide 1: Understanding Combined Exception Handling

Exception handling in Python allows grouping multiple exceptions into a single except block, reducing code redundancy and improving readability. This approach maintains functionality while simplifying error management through the strategic combination of related exceptions.

```python
# Traditional approach with separate handlers
def traditional_handling(x):
    try:
        result = 10 / x
        numbers = [1, 2, 3]
        value = numbers[x]
    except ZeroDivisionError:
        print("Cannot divide by zero")
    except IndexError:
        print("Index out of range")
        
# Combined approach
def combined_handling(x):
    try:
        result = 10 / x
        numbers = [1, 2, 3]
        value = numbers[x]
    except (ZeroDivisionError, IndexError) as e:
        print(f"Operation failed: {str(e)}")

# Example usage
combined_handling(0)  # Output: Operation failed: division by zero
combined_handling(5)  # Output: Operation failed: list index out of range
```

Slide 2: Exception Hierarchies and Grouping

Python's exception hierarchy allows for intelligent grouping based on exception types. Understanding the relationship between exceptions helps in creating effective combined handlers that maintain specific error handling while reducing code complexity.

```python
def process_data(data):
    try:
        # Multiple operations that might fail
        parsed = int(data)
        result = 100 / parsed
        assert result > 0, "Negative results not allowed"
    except (ValueError, ZeroDivisionError, AssertionError) as e:
        print(f"Data processing error: {type(e).__name__} - {str(e)}")
        return None
    else:
        return result

# Example usage with different scenarios
print(process_data("abc"))    # ValueError
print(process_data("0"))      # ZeroDivisionError
print(process_data("-5"))     # AssertionError
```

Slide 3: Custom Exception Groups

Creating custom exception hierarchies enables organized error handling for domain-specific problems. This approach allows for granular control while maintaining the benefits of combined exception handling through inheritance.

```python
class DataProcessingError(Exception):
    """Base class for data processing exceptions"""
    pass

class ValidationError(DataProcessingError):
    """Raised when data validation fails"""
    pass

class TransformationError(DataProcessingError):
    """Raised when data transformation fails"""
    pass

def process_record(record):
    try:
        if not isinstance(record, dict):
            raise ValidationError("Record must be a dictionary")
        if 'value' not in record:
            raise ValidationError("Missing 'value' field")
        record['transformed'] = record['value'] * 2
    except DataProcessingError as e:
        print(f"Processing failed: {str(e)}")
        return None
    return record

# Example usage
print(process_record([]))           # ValidationError
print(process_record({"key": 1}))   # ValidationError
print(process_record({"value": 5})) # Success
```

Slide 4: Context-Aware Exception Handling

Context-aware exception handling combines multiple exceptions while maintaining detailed error information. This approach provides comprehensive error reporting while keeping the code structure clean and maintainable.

```python
import contextlib
import logging

class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @contextlib.contextmanager
    def error_context(self, operation_name):
        try:
            yield
        except (ValueError, TypeError) as e:
            self.logger.error(f"{operation_name} failed: {str(e)}")
            raise DataProcessingError(f"{operation_name}: {str(e)}")
    
    def process_value(self, value):
        with self.error_context("Value processing"):
            return int(value) * 2
    
    def process_list(self, items):
        results = []
        for idx, item in enumerate(items):
            with self.error_context(f"Item {idx} processing"):
                results.append(self.process_value(item))
        return results

# Example usage
processor = DataProcessor()
try:
    print(processor.process_list(['1', '2', 'abc', '4']))
except DataProcessingError as e:
    print(f"Processing failed: {e}")
```

Slide 5: Exception Chaining and Information Preservation

Exception chaining allows preservation of the original error context while raising new exceptions. This technique enables meaningful error reporting in complex operations while maintaining the benefits of combined exception handling.

```python
class ETLError(Exception):
    pass

def extract(data):
    try:
        return float(data)
    except (ValueError, TypeError) as e:
        raise ETLError("Extraction failed") from e

def transform(value):
    try:
        return value * 2
    except Exception as e:
        raise ETLError("Transformation failed") from e

def load(value):
    try:
        if value < 0:
            raise ValueError("Cannot load negative values")
        return f"Loaded: {value}"
    except Exception as e:
        raise ETLError("Loading failed") from e

def etl_pipeline(data):
    try:
        extracted = extract(data)
        transformed = transform(extracted)
        result = load(transformed)
        return result
    except ETLError as e:
        print(f"Pipeline error: {e}")
        print(f"Original error: {e.__cause__}")
        return None

# Example usage
print(etl_pipeline("abc"))    # Extraction error
print(etl_pipeline("10"))     # Success
print(etl_pipeline("-5"))     # Loading error
```

Slide 6: Resource Management with Combined Exceptions

Resource management often requires handling multiple potential failures during acquisition and release. Combined exception handling streamlines cleanup operations while ensuring proper resource management across different error scenarios.

```python
class Resource:
    def __init__(self, name):
        self.name = name
        
    def acquire(self):
        if self.name == "invalid":
            raise ValueError("Cannot acquire invalid resource")
        print(f"Resource {self.name} acquired")
        
    def release(self):
        if self.name == "locked":
            raise RuntimeError("Cannot release locked resource")
        print(f"Resource {self.name} released")

def manage_resources(resources):
    acquired = []
    try:
        # Acquire phase
        for resource in resources:
            resource.acquire()
            acquired.append(resource)
            
        # Process phase
        print("Processing resources...")
            
    except (ValueError, RuntimeError) as e:
        print(f"Operation failed: {str(e)}")
    finally:
        # Release phase
        for resource in reversed(acquired):
            try:
                resource.release()
            except Exception as e:
                print(f"Release failed for {resource.name}: {str(e)}")

# Example usage
resources = [
    Resource("db"),
    Resource("invalid"),
    Resource("locked")
]
manage_resources(resources)
```

Slide 7: Network Operations Error Handling

Network operations often encounter various types of failures. Using combined exception handling for network-related errors provides a robust way to handle different failure modes while maintaining clean and maintainable code.

```python
import socket
import json
from urllib.error import URLError, HTTPError
from json.decoder import JSONDecodeError

class NetworkClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        
    def fetch_data(self):
        try:
            # Simulate network operations
            sock = socket.create_connection((self.host, self.port), 
                                         timeout=5)
            data = sock.recv(1024)
            return json.loads(data.decode())
            
        except (socket.timeout, ConnectionRefusedError) as e:
            raise ConnectionError(f"Network error: {str(e)}")
        except JSONDecodeError as e:
            raise ValueError(f"Invalid data format: {str(e)}")
        finally:
            sock.close() if 'sock' in locals() else None

    def safe_fetch(self):
        try:
            return self.fetch_data()
        except (ConnectionError, ValueError) as e:
            print(f"Fetch failed: {str(e)}")
            return None

# Example usage
client = NetworkClient("localhost", 8080)
result = client.safe_fetch()
```

Slide 8: Database Operations with Combined Error Handling

When working with databases, multiple types of errors can occur during connection, query execution, and data retrieval. Combined exception handling provides a clean way to manage these diverse error scenarios.

```python
class DatabaseError(Exception):
    pass

class Database:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
        
    def execute_query(self, query, parameters=None):
        try:
            # Simulate database operations
            if "invalid" in self.connection_string:
                raise ConnectionError("Failed to connect to database")
                
            if "SELECT" not in query.upper():
                raise ValueError("Only SELECT queries allowed")
                
            if parameters and not isinstance(parameters, dict):
                raise TypeError("Parameters must be a dictionary")
                
            return [{"id": 1, "data": "sample"}]
            
        except (ConnectionError, ValueError, TypeError) as e:
            raise DatabaseError(f"Query execution failed: {str(e)}")
            
    def safe_query(self, query, parameters=None):
        try:
            return self.execute_query(query, parameters)
        except DatabaseError as e:
            print(f"Database operation failed: {str(e)}")
            return None

# Example usage
db = Database("mysql://localhost/mydb")
result = db.safe_query("SELECT * FROM users", {"id": 1})
print(result)

db_invalid = Database("invalid://localhost/mydb")
result = db_invalid.safe_query("INSERT INTO users", [1, 2])
print(result)
```

Slide 9: File Operations with Combined Exceptions

File operations can fail in multiple ways, from permission issues to encoding problems. Implementing combined exception handling for file operations ensures robust error management while maintaining code clarity.

```python
class FileProcessor:
    def __init__(self, filename):
        self.filename = filename
        
    def process_file(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Process content
                if len(content) == 0:
                    raise ValueError("Empty file")
                    
                numbers = [int(line) for line in content.split('\n')]
                return sum(numbers)
                
        except (FileNotFoundError, PermissionError) as e:
            print(f"File access error: {str(e)}")
            return None
        except (ValueError, UnicodeDecodeError) as e:
            print(f"File content error: {str(e)}")
            return None

# Example usage
processor = FileProcessor("numbers.txt")
result = processor.process_file()

processor_invalid = FileProcessor("/root/restricted.txt")
result_invalid = processor_invalid.process_file()
```

Slide 10: Asynchronous Operations Error Handling

Asynchronous operations require special attention to exception handling due to their non-linear execution flow. Combined exception handling in async contexts helps manage multiple failure modes while maintaining code clarity.

```python
import asyncio
from typing import List, Any

class AsyncProcessor:
    async def process_item(self, item: Any) -> Any:
        try:
            if isinstance(item, str):
                await asyncio.sleep(0.1)  # Simulate processing
                return item.upper()
            elif isinstance(item, (int, float)):
                await asyncio.sleep(0.1)  # Simulate processing
                return item * 2
            else:
                raise TypeError(f"Unsupported type: {type(item)}")
        except (ValueError, TypeError) as e:
            raise ProcessingError(f"Processing failed: {str(e)}")

    async def process_batch(self, items: List[Any]) -> List[Any]:
        try:
            tasks = [self.process_item(item) for item in items]
            return await asyncio.gather(*tasks)
        except (ProcessingError, asyncio.TimeoutError) as e:
            print(f"Batch processing failed: {str(e)}")
            return []

# Example usage
async def main():
    processor = AsyncProcessor()
    items = ["test", 42, 3.14, None]
    result = await processor.process_batch(items)
    print(f"Processed results: {result}")

# Run the async code
asyncio.run(main())
```

Slide 11: Real-world Example: Data Pipeline with Error Recovery

This comprehensive example demonstrates a data pipeline that processes multiple data sources with combined exception handling and automatic recovery mechanisms for various failure scenarios.

```python
from typing import Dict, List, Optional
import json
import time

class DataPipeline:
    def __init__(self, retry_attempts: int = 3):
        self.retry_attempts = retry_attempts
        self.error_log: List[Dict] = []

    def process_record(self, record: Dict) -> Optional[Dict]:
        for attempt in range(self.retry_attempts):
            try:
                # Validate record
                if not isinstance(record, dict):
                    raise ValueError("Record must be a dictionary")

                # Transform data
                processed = {
                    'id': str(record.get('id', '')),
                    'timestamp': time.time(),
                    'value': float(record.get('value', 0)) * 1.5,
                    'status': 'processed'
                }

                # Validate output
                if processed['value'] < 0:
                    raise ValueError("Negative values not allowed")

                return processed

            except (ValueError, TypeError, KeyError) as e:
                self.error_log.append({
                    'error': str(e),
                    'attempt': attempt + 1,
                    'record': record
                })
                if attempt == self.retry_attempts - 1:
                    print(f"Failed to process record after {self.retry_attempts} attempts")
                    return None
                time.sleep(0.1 * attempt)  # Exponential backoff

    def process_batch(self, records: List[Dict]) -> Dict[str, List]:
        results = {
            'successful': [],
            'failed': [],
            'error_log': self.error_log
        }

        for record in records:
            try:
                processed = self.process_record(record)
                if processed:
                    results['successful'].append(processed)
                else:
                    results['failed'].append(record)
            except Exception as e:
                results['failed'].append(record)
                self.error_log.append({
                    'error': str(e),
                    'record': record,
                    'type': 'unexpected'
                })

        return results

# Example usage
pipeline = DataPipeline()
test_data = [
    {'id': 1, 'value': '10.5'},
    {'id': 2, 'value': '-5.0'},
    {'id': 3, 'value': 'invalid'},
    {'id': 4, 'value': '15.7'}
]

results = pipeline.process_batch(test_data)
print(json.dumps(results, indent=2))
```

Slide 12: Real-world Example: API Client with Robust Error Handling

This example shows a production-ready API client implementation that handles various network, authentication, and data processing errors while maintaining clean error reporting.

```python
from typing import Optional, Dict, Any
import time
import json

class APIClientError(Exception):
    pass

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session_valid = True
        self.retry_count = 3

    def _validate_response(self, response: Dict) -> None:
        if not isinstance(response, dict):
            raise ValueError("Invalid response format")
        if response.get('status') == 'error':
            raise APIClientError(response.get('message', 'Unknown error'))

    def make_request(self, endpoint: str, data: Dict) -> Optional[Dict[str, Any]]:
        for attempt in range(self.retry_count):
            try:
                # Simulate network request
                if not self.session_valid:
                    raise ConnectionError("Invalid session")

                # Simulate API call
                if 'invalid' in str(data):
                    raise ValueError("Invalid request data")

                # Simulate response
                response = {
                    'status': 'success',
                    'data': {
                        'request_id': f"req_{time.time()}",
                        'result': data
                    }
                }

                self._validate_response(response)
                return response['data']

            except (ConnectionError, ValueError, APIClientError) as e:
                if isinstance(e, ConnectionError):
                    self.session_valid = False
                    print(f"Connection error: {str(e)}")
                elif isinstance(e, ValueError):
                    print(f"Validation error: {str(e)}")
                else:
                    print(f"API error: {str(e)}")

                if attempt == self.retry_count - 1:
                    return None

                time.sleep(0.5 * (attempt + 1))
                continue

    def safe_request(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        try:
            result = self.make_request(endpoint, data)
            if result is None:
                return {'status': 'error', 'message': 'Request failed'}
            return {'status': 'success', 'data': result}
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Unexpected error: {str(e)}"
            }

# Example usage
client = APIClient("https://api.example.com", "secret_key")
valid_request = client.safe_request("/data", {"key": "value"})
print("Valid request:", json.dumps(valid_request, indent=2))

invalid_request = client.safe_request("/data", {"invalid": "data"})
print("Invalid request:", json.dumps(invalid_request, indent=2))
```

Slide 13: Advanced Pattern Matching with Exception Groups

Python's exception groups feature enables sophisticated error handling patterns for complex operations. This approach allows handling multiple exceptions while maintaining granular control over error processing.

```python
from typing import List, Dict
from dataclasses import dataclass
import sys

@dataclass
class ValidationError(Exception):
    field: str
    message: str

class ProcessingError(Exception):
    pass

class DataValidator:
    def validate_record(self, record: Dict) -> None:
        errors = []
        
        # Validate multiple fields
        try:
            if not isinstance(record.get('id'), int):
                errors.append(ValidationError('id', 'Must be integer'))
            if not isinstance(record.get('name'), str):
                errors.append(ValidationError('name', 'Must be string'))
            if record.get('age', 0) < 0:
                errors.append(ValidationError('age', 'Must be positive'))
                
            if errors:
                if sys.version_info >= (3, 11):  # Python 3.11+ support
                    raise ExceptionGroup("Validation failed", errors)
                else:
                    raise ValidationError('multiple', str(errors))
                    
        except ExceptionGroup as eg:
            print("Multiple validation errors:")
            for error in eg.exceptions:
                print(f"- {error.field}: {error.message}")
            raise
        except ValidationError as e:
            print(f"Validation error: {e.message}")
            raise

# Example usage
validator = DataValidator()
try:
    validator.validate_record({
        'id': '123',  # Should be int
        'name': 42,   # Should be string
        'age': -5     # Should be positive
    })
except (ExceptionGroup, ValidationError):
    print("Record validation failed")
```

Slide 14: Defensive Programming with Exception Chains

This advanced technique demonstrates how to implement defensive programming using exception chains, allowing for detailed error tracking while maintaining clean code structure.

```python
from typing import Any, Optional
from datetime import datetime

class OperationError(Exception):
    def __init__(self, message: str, operation: str, 
                 timestamp: Optional[datetime] = None):
        self.operation = operation
        self.timestamp = timestamp or datetime.now()
        super().__init__(f"{message} in {operation} at {self.timestamp}")

class ErrorChain:
    def __init__(self):
        self.error_stack = []
        
    def execute_step(self, step_name: str, 
                    operation: callable, *args, **kwargs) -> Any:
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            error = OperationError(str(e), step_name)
            self.error_stack.append(error)
            raise error from e
            
    def process_with_recovery(self, steps: dict) -> Optional[Any]:
        result = None
        for step_name, (operation, recovery) in steps.items():
            try:
                result = self.execute_step(step_name, operation, result)
            except OperationError as e:
                if recovery:
                    try:
                        result = recovery(e, result)
                        print(f"Recovered from {step_name} error")
                    except Exception as recovery_error:
                        print(f"Recovery failed: {recovery_error}")
                        return None
                else:
                    return None
        return result

# Example usage
def process_data(data: Optional[Any] = None) -> int:
    if not isinstance(data, (int, type(None))):
        raise ValueError("Invalid input type")
    return (data or 0) + 1

def recover_process(error: Exception, data: Optional[Any]) -> int:
    return 0

steps = {
    'step1': (lambda x: int('invalid'), None),  # Will fail
    'step2': (process_data, recover_process),   # Has recovery
    'step3': (lambda x: x * 2, None)           # Final step
}

chain = ErrorChain()
result = chain.process_with_recovery(steps)
print(f"Final result: {result}")
print("Error stack:")
for error in chain.error_stack:
    print(f"- {error.operation}: {str(error)}")
```

Slide 15: Additional Resources

*   Exception Handling Best Practices in Python
    *   [https://arxiv.org/abs/cs/0701072](https://arxiv.org/abs/cs/0701072)
    *   [https://www.python.org/dev/peps/pep-3134/](https://www.python.org/dev/peps/pep-3134/)
    *   [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
*   Advanced Error Handling Patterns
    *   [https://www.python.org/dev/peps/pep-0654/](https://www.python.org/dev/peps/pep-0654/)
    *   [https://www.python.org/dev/peps/pep-3134/](https://www.python.org/dev/peps/pep-3134/)
    *   For more resources, search "Python Exception Handling Patterns" on Google Scholar
*   Defensive Programming Techniques
    *   [https://www.sciencedirect.com/topics/computer-science/defensive-programming](https://www.sciencedirect.com/topics/computer-science/defensive-programming)
    *   For implementation examples, visit Python's official documentation
    *   Search "Defensive Programming in Python" on academic databases

Note: Due to my training cutoff date, please verify all URLs and refer to the most current documentation for up-to-date information.

