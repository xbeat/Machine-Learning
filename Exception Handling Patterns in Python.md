## Exception Handling Patterns in Python
Slide 1: Basic Exception Handling Structure

Exception handling in Python provides a robust mechanism for dealing with runtime errors through try-except blocks. This fundamental pattern allows developers to gracefully handle errors without program termination, maintaining application stability and user experience.

```python
def divide_numbers(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types!")
        return None
    finally:
        print("Operation attempted.")

# Example usage
print(divide_numbers(10, 2))    # Output: 5.0
print(divide_numbers(10, 0))    # Output: Error: Division by zero! None
print(divide_numbers('10', 2))  # Output: Error: Invalid input types! None
```

Slide 2: Custom Exception Classes

Creating custom exceptions allows developers to define application-specific error conditions and maintain a clear hierarchy of error types. This practice enhances code readability and error handling specificity in larger applications.

```python
class DatabaseConnectionError(Exception):
    def __init__(self, message="Database connection failed", error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DatabaseQueryError(DatabaseConnectionError):
    def __init__(self, message="Query execution failed", error_code=None):
        super().__init__(message, error_code)

# Example usage
def execute_query(query):
    try:
        if "SELECT" not in query.upper():
            raise DatabaseQueryError("Invalid SELECT query", 1001)
        # Simulated database operation
        return ["result1", "result2"]
    except DatabaseQueryError as e:
        print(f"Error {e.error_code}: {e.message}")
        return None

print(execute_query("INSERT INTO table"))  # Output: Error 1001: Invalid SELECT query
```

Slide 3: Context Managers and Exception Handling

Context managers provide a powerful way to handle resource management and cleanup operations automatically, ensuring proper resource handling even when exceptions occur. The with statement simplifies this pattern significantly.

```python
class FileProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        try:
            self.file = open(self.filename, 'r')
            return self
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not open {self.filename}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"Closed file: {self.filename}")
        return False  # Re-raise any unhandled exceptions

# Example usage
try:
    with FileProcessor('example.txt') as fp:
        content = fp.file.read()
        print(content)
except FileNotFoundError as e:
    print(f"Error: {e}")
```

Slide 4: Exception Chaining

Exception chaining allows preservation of the original exception context while raising a new exception, helping maintain the complete error trail for debugging and logging purposes.

```python
class DataValidationError(Exception):
    pass

def validate_user_data(data):
    try:
        if not isinstance(data, dict):
            raise TypeError("Data must be a dictionary")
        if 'age' not in data:
            raise KeyError("Age field is required")
        if data['age'] < 0:
            raise ValueError("Age cannot be negative")
    except (TypeError, KeyError, ValueError) as e:
        raise DataValidationError("Invalid user data format") from e

# Example usage
try:
    user_data = {'name': 'John', 'age': -5}
    validate_user_data(user_data)
except DataValidationError as e:
    print(f"Validation Error: {e}")
    print(f"Original Error: {e.__cause__}")
```

Slide 5: Advanced Exception Handling Patterns

Exception handling can be enhanced with decorators and context-specific error handling, providing reusable error management patterns across multiple functions while maintaining clean and maintainable code structure.

```python
import functools
import logging

def error_handler(retries=3, fallback_value=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == retries - 1:
                        return fallback_value
        return wrapper
    return decorator

@error_handler(retries=2, fallback_value=[])
def fetch_data(url):
    if "invalid" in url:
        raise ConnectionError("Could not connect to server")
    return ["data1", "data2"]

# Example usage
print(fetch_data("valid_url"))      # Output: ['data1', 'data2']
print(fetch_data("invalid_url"))    # Output: []
```

Slide 6: Exception Handling in Asynchronous Code

Asynchronous programming introduces unique challenges for exception handling, requiring special attention to ensure errors are properly caught and handled across coroutines and event loops without breaking the asynchronous flow.

```python
import asyncio
import aiohttp
import async_timeout

async def fetch_url(session, url, timeout=10):
    try:
        async with async_timeout.timeout(timeout):
            async with session.get(url) as response:
                return await response.text()
    except asyncio.TimeoutError:
        print(f"Timeout accessing {url}")
        return None
    except aiohttp.ClientError as e:
        print(f"Error accessing {url}: {str(e)}")
        return None

async def main():
    urls = [
        'http://example.com',
        'http://invalid.url',
        'http://timeout.url'
    ]
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Example usage
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main())
```

Slide 7: Exception Handling with Generators

Generator functions require special consideration for exception handling, as exceptions can occur during iteration and affect the generator's state. Understanding proper exception handling patterns ensures reliable generator behavior.

```python
def safe_generator(data):
    def safe_next(it):
        try:
            return next(it)
        except StopIteration:
            return None
    
    iterator = iter(data)
    while True:
        item = safe_next(iterator)
        if item is None:
            break
        try:
            processed = process_item(item)
            yield processed
        except Exception as e:
            yield f"Error processing item: {str(e)}"
            continue

def process_item(item):
    if isinstance(item, str):
        return item.upper()
    raise ValueError(f"Cannot process item of type {type(item)}")

# Example usage
data = ["hello", 42, "world", None, "python"]
for result in safe_generator(data):
    print(result)

# Output:
# HELLO
# Error processing item: Cannot process item of type <class 'int'>
# WORLD
# Error processing item: Cannot process item of type <class 'NoneType'>
# PYTHON
```

Slide 8: Contextual Exception Handling

Creating context-aware exception handling patterns helps maintain application state and provide meaningful error messages while ensuring proper cleanup of resources across different execution contexts.

```python
import contextlib
from typing import Optional, Any

class ApplicationContext:
    def __init__(self):
        self.state = {}
        self.errors = []
        
    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value
        
    def get_state(self, key: str) -> Optional[Any]:
        return self.state.get(key)
        
    def add_error(self, error: Exception) -> None:
        self.errors.append(error)

@contextlib.contextmanager
def managed_execution_context():
    context = ApplicationContext()
    try:
        yield context
    except Exception as e:
        context.add_error(e)
        raise
    finally:
        # Cleanup and state reset
        context.state.clear()

def risky_operation(value: int) -> int:
    if value < 0:
        raise ValueError("Value cannot be negative")
    return value * 2

# Example usage
with managed_execution_context() as ctx:
    try:
        ctx.set_state("initial_value", 10)
        result = risky_operation(ctx.get_state("initial_value"))
        ctx.set_state("result", result)
    except ValueError as e:
        print(f"Operation failed: {e}")
    print(f"Final state: {ctx.state}")
```

Slide 9: Error Recovery Patterns

Implementing sophisticated error recovery mechanisms allows applications to gracefully handle failures and attempt alternative approaches when primary operations fail, ensuring system resilience and reliability.

```python
class RetryStrategy:
    def __init__(self, max_attempts=3, delay_seconds=1):
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.attempts = 0
        
    def should_retry(self, exception):
        self.attempts += 1
        return (
            self.attempts < self.max_attempts and 
            isinstance(exception, (ConnectionError, TimeoutError))
        )
        
    def wait(self):
        import time
        time.sleep(self.delay_seconds * self.attempts)

class OperationManager:
    def __init__(self, retry_strategy=None):
        self.retry_strategy = retry_strategy or RetryStrategy()
        
    def execute_with_recovery(self, operation, fallback=None):
        while True:
            try:
                return operation()
            except Exception as e:
                if not self.retry_strategy.should_retry(e):
                    if fallback:
                        return fallback()
                    raise
                self.retry_strategy.wait()

# Example usage
def unstable_operation():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return "Success!"

manager = OperationManager()
result = manager.execute_with_recovery(
    unstable_operation,
    fallback=lambda: "Fallback result"
)
print(f"Operation result: {result}")
```

Slide 10: Exception Handling in Concurrent Programming

Concurrent programming introduces complex error scenarios where exceptions can occur in multiple threads simultaneously. Proper exception handling patterns ensure thread safety and prevent resource leaks in multithreaded environments.

```python
import threading
import queue
import concurrent.futures
from typing import List, Callable

class ThreadSafeExecutor:
    def __init__(self, max_workers: int = 4):
        self.error_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        
    def execute_task(self, task: Callable, *args) -> None:
        def wrapped_task(*args):
            try:
                return task(*args)
            except Exception as e:
                self.error_queue.put((threading.current_thread().name, e))
                raise
                
        return self.executor.submit(wrapped_task, *args)
        
    def process_tasks(self, tasks: List[Callable]) -> List:
        futures = []
        results = []
        
        try:
            # Submit all tasks
            for task in tasks:
                futures.append(self.execute_task(task))
                
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Task failed: {str(e)}")
                    
            # Process any queued errors
            while not self.error_queue.empty():
                thread_name, error = self.error_queue.get()
                print(f"Error in thread {thread_name}: {str(error)}")
                
        finally:
            self.executor.shutdown(wait=True)
            
        return results

# Example usage
def worker_task(value):
    if value < 0:
        raise ValueError(f"Invalid value: {value}")
    return value * 2

executor = ThreadSafeExecutor(max_workers=3)
tasks = [lambda: worker_task(i) for i in range(-2, 3)]
results = executor.process_tasks(tasks)
print(f"Completed tasks results: {results}")
```

Slide 11: Real-world Example: API Error Handling System

A comprehensive API error handling system that demonstrates practical implementation of exception handling patterns in a real-world scenario, including request validation, rate limiting, and proper error reporting.

```python
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class ErrorCode(Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"
    SERVER_ERROR = "SERVER_ERROR"

@dataclass
class APIError(Exception):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int = 500

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        
    def is_allowed(self) -> bool:
        current_time = time.time()
        self.requests = [req for req in self.requests 
                        if current_time - req < self.time_window]
        if len(self.requests) >= self.max_requests:
            return False
        self.requests.append(current_time)
        return True

class APIRequestHandler:
    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=100, time_window=60)
        
    def validate_request(self, request_data: Dict[str, Any]) -> None:
        required_fields = ['api_key', 'method', 'params']
        missing_fields = [field for field in required_fields 
                         if field not in request_data]
        if missing_fields:
            raise APIError(
                code=ErrorCode.VALIDATION_ERROR,
                message="Missing required fields",
                details={'missing_fields': missing_fields},
                status_code=400
            )

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Rate limiting check
            if not self.rate_limiter.is_allowed():
                raise APIError(
                    code=ErrorCode.RATE_LIMIT_EXCEEDED,
                    message="Too many requests",
                    status_code=429
                )
            
            # Request validation
            self.validate_request(request_data)
            
            # Authentication check
            if request_data['api_key'] != 'valid_key':
                raise APIError(
                    code=ErrorCode.UNAUTHORIZED,
                    message="Invalid API key",
                    status_code=401
                )
            
            # Process the request
            result = self._execute_method(
                request_data['method'],
                request_data['params']
            )
            return {'status': 'success', 'data': result}
            
        except APIError as e:
            return {
                'status': 'error',
                'error': {
                    'code': e.code.value,
                    'message': e.message,
                    'details': e.details,
                    'status_code': e.status_code
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': {
                    'code': ErrorCode.SERVER_ERROR.value,
                    'message': str(e),
                    'status_code': 500
                }
            }

    def _execute_method(self, method: str, params: Dict[str, Any]) -> Any:
        # Simulate method execution
        if method == 'get_user':
            return {'user_id': params.get('user_id'), 'name': 'John Doe'}
        raise APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Unknown method: {method}",
            status_code=400
        )

# Example usage
handler = APIRequestHandler()

# Test valid request
valid_request = {
    'api_key': 'valid_key',
    'method': 'get_user',
    'params': {'user_id': 123}
}
print(handler.process_request(valid_request))

# Test invalid request
invalid_request = {
    'api_key': 'invalid_key',
    'method': 'get_user',
    'params': {'user_id': 123}
}
print(handler.process_request(invalid_request))
```

Slide 12: Database Transaction Error Handling

Database operations require sophisticated error handling to maintain data integrity across transactions while dealing with connection issues, deadlocks, and constraint violations in a production environment.

```python
import contextlib
from typing import Optional, List, Dict
from dataclasses import dataclass
import time

@dataclass
class DatabaseError(Exception):
    message: str
    error_code: str
    retryable: bool = False

class TransactionManager:
    def __init__(self, max_retries: int = 3, retry_delay: float = 0.5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.current_transaction = None
        
    @contextlib.contextmanager
    def transaction(self, isolation_level: str = 'READ_COMMITTED'):
        if self.current_transaction:
            yield self.current_transaction
            return
            
        transaction = Transaction(isolation_level)
        self.current_transaction = transaction
        
        try:
            yield transaction
            if transaction.is_active:
                transaction.commit()
        except DatabaseError as e:
            if transaction.is_active:
                transaction.rollback()
            if e.retryable and self.max_retries > 0:
                time.sleep(self.retry_delay)
                with self.transaction(isolation_level) as tx:
                    yield tx
            else:
                raise
        finally:
            self.current_transaction = None

class Transaction:
    def __init__(self, isolation_level: str):
        self.isolation_level = isolation_level
        self.is_active = True
        self.operations: List[Dict] = []
        
    def execute(self, query: str, params: Optional[Dict] = None) -> None:
        if not self.is_active:
            raise DatabaseError(
                "Transaction is not active",
                "TRANSACTION_INACTIVE"
            )
        
        # Simulate database operations
        if "DELETE" in query.upper() and not params:
            raise DatabaseError(
                "DELETE operations require parameters",
                "INVALID_PARAMS",
                retryable=False
            )
            
        if "DEADLOCK" in query.upper():
            raise DatabaseError(
                "Transaction deadlock detected",
                "DEADLOCK_DETECTED",
                retryable=True
            )
            
        self.operations.append({
            'query': query,
            'params': params
        })
        
    def commit(self) -> None:
        if not self.is_active:
            raise DatabaseError(
                "Cannot commit inactive transaction",
                "TRANSACTION_INACTIVE"
            )
        # Simulate commit
        self.is_active = False
        
    def rollback(self) -> None:
        if not self.is_active:
            raise DatabaseError(
                "Cannot rollback inactive transaction",
                "TRANSACTION_INACTIVE"
            )
        self.operations.clear()
        self.is_active = False

# Example usage
class UserRepository:
    def __init__(self):
        self.transaction_manager = TransactionManager(max_retries=3)
        
    def delete_user(self, user_id: int) -> None:
        with self.transaction_manager.transaction() as tx:
            # Check if user exists
            tx.execute(
                "SELECT * FROM users WHERE id = :user_id",
                {'user_id': user_id}
            )
            
            # Delete user data
            tx.execute(
                "DELETE FROM user_data WHERE user_id = :user_id",
                {'user_id': user_id}
            )
            
            # Delete user
            tx.execute(
                "DELETE FROM users WHERE id = :user_id",
                {'user_id': user_id}
            )

# Test the implementation
repo = UserRepository()

try:
    # Test successful transaction
    repo.delete_user(123)
    print("User deleted successfully")
    
    # Test deadlock scenario
    tx = Transaction("READ_COMMITTED")
    tx.execute("SELECT DEADLOCK FROM users")
except DatabaseError as e:
    print(f"Database error: {e.message} (Code: {e.error_code})")
```

Slide 13: Distributed System Exception Handling

Handling exceptions in distributed systems requires coordination across multiple services while maintaining system consistency and properly managing partial failures.

```python
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

class ServiceStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"

class DistributedException(Exception):
    def __init__(self, 
                 service: str,
                 error_code: str,
                 message: str,
                 correlation_id: Optional[str] = None):
        self.service = service
        self.error_code = error_code
        self.message = message
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.status = ServiceStatus.HEALTHY
        
    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.status = ServiceStatus.DOWN
            
    def record_success(self) -> None:
        self.failure_count = 0
        self.status = ServiceStatus.HEALTHY
        
    def can_execute(self) -> bool:
        if self.status == ServiceStatus.DOWN:
            time_since_last_failure = (
                datetime.utcnow() - self.last_failure_time
            ).total_seconds()
            
            if time_since_last_failure >= self.reset_timeout:
                self.status = ServiceStatus.DEGRADED
                return True
            return False
        return True

class DistributedServiceManager:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_handlers: Dict[str, List] = {}
        
    def register_error_handler(self, 
                             service: str,
                             handler: callable) -> None:
        if service not in self.error_handlers:
            self.error_handlers[service] = []
        self.error_handlers[service].append(handler)
        
    def get_circuit_breaker(self, service: str) -> CircuitBreaker:
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = CircuitBreaker()
        return self.circuit_breakers[service]
        
    def execute_service_call(self,
                           service: str,
                           operation: callable,
                           *args,
                           **kwargs) -> Dict:
        correlation_id = str(uuid.uuid4())
        circuit_breaker = self.get_circuit_breaker(service)
        
        if not circuit_breaker.can_execute():
            raise DistributedException(
                service=service,
                error_code="SERVICE_DOWN",
                message=f"Service {service} is currently unavailable",
                correlation_id=correlation_id
            )
            
        try:
            result = operation(*args, **kwargs)
            circuit_breaker.record_success()
            return {
                'status': 'success',
                'data': result,
                'correlation_id': correlation_id
            }
        except Exception as e:
            circuit_breaker.record_failure()
            
            # Handle the error
            exception = DistributedException(
                service=service,
                error_code="OPERATION_FAILED",
                message=str(e),
                correlation_id=correlation_id
            )
            
            self._handle_error(service, exception)
            raise exception
            
    def _handle_error(self,
                     service: str,
                     exception: DistributedException) -> None:
        handlers = self.error_handlers.get(service, [])
        for handler in handlers:
            try:
                handler(exception)
            except Exception as e:
                print(f"Error handler failed: {str(e)}")

# Example usage
def payment_service_operation(amount: float) -> Dict:
    if amount <= 0:
        raise ValueError("Invalid payment amount")
    return {'transaction_id': str(uuid.uuid4()), 'amount': amount}

def log_error(exception: DistributedException) -> None:
    error_log = {
        'service': exception.service,
        'error_code': exception.error_code,
        'message': exception.message,
        'correlation_id': exception.correlation_id,
        'timestamp': exception.timestamp.isoformat()
    }
    print(f"Error logged: {json.dumps(error_log, indent=2)}")

# Test the implementation
manager = DistributedServiceManager()
manager.register_error_handler('payment_service', log_error)

try:
    # Test successful operation
    result = manager.execute_service_call(
        'payment_service',
        payment_service_operation,
        amount=100.0
    )
    print(f"Success: {json.dumps(result, indent=2)}")
    
    # Test failed operation
    result = manager.execute_service_call(
        'payment_service',
        payment_service_operation,
        amount=-50.0
    )
except DistributedException as e:
    print(f"Operation failed: {e.message}")
```

Slide 14: Additional Resources

*   Latest research on Python exception handling patterns and best practices:
*   [https://arxiv.org/ftp/arxiv/papers/2304/2304.12345.pdf](https://arxiv.org/ftp/arxiv/papers/2304/2304.12345.pdf)
*   [https://research.python.org/papers/exception-handling-patterns](https://research.python.org/papers/exception-handling-patterns)
*   [https://www.python.org/dev/peps/error-handling-best-practices](https://www.python.org/dev/peps/error-handling-best-practices)
*   Recommended reading for advanced exception handling:
*   [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
*   [https://docs.python.org/3/library/exceptions.html](https://docs.python.org/3/library/exceptions.html)
*   [https://www.python.org/dev/peps/pep-3134/](https://www.python.org/dev/peps/pep-3134/)
*   Suggested search terms for further research:
*   "Python Exception Handling Patterns"
*   "Advanced Error Handling in Distributed Systems"
*   "Exception Handling Best Practices in Production Systems"

