## How to Create Custom Exception in Python
Slide 1: Understanding Custom Exceptions

Custom exceptions in Python extend the built-in Exception class to create specialized error handling mechanisms. These allow developers to define application-specific error conditions and provide meaningful error messages tailored to their program's requirements.

```python
# Basic structure of a custom exception
class CustomError(Exception):
    def __init__(self, message="A custom error occurred"):
        self.message = message
        super().__init__(self.message)
```

Slide 2: Creating a Domain-Specific Exception

A well-designed custom exception should encapsulate domain-specific error conditions and relevant error data. This enables precise error handling and debugging by capturing contextual information about the error state.

```python
class InvalidWeightError(Exception):
    def __init__(self, weight, message=None):
        self.weight = weight
        self.message = message or f"Invalid weight value: {weight}"
        super().__init__(self.message)

    def __str__(self):
        return f"Weight Error: {self.message}"
```

Slide 3: Implementing Weight Calculator with Custom Exception

The weight calculator demonstrates practical usage of custom exceptions by validating input parameters and raising appropriate errors when conditions are not met. This ensures robust error handling in real-world applications.

```python
def calculate_moon_weight(earth_weight):
    try:
        if not isinstance(earth_weight, (int, float)):
            raise InvalidWeightError(earth_weight, "Weight must be a number")
        if earth_weight < 0 or earth_weight > 300:
            raise InvalidWeightError(earth_weight, "Weight must be between 0 and 300 kg")
        
        return earth_weight * 0.165
    except InvalidWeightError as e:
        print(f"Error: {e}")
        return None
```

Slide 4: Hierarchical Exception Structure

Custom exceptions can form a hierarchy to represent different categories of errors while maintaining a common base. This approach enables more granular error handling and improved code organization.

```python
class WeightError(Exception):
    """Base exception for weight-related errors"""
    pass

class NegativeWeightError(WeightError):
    def __init__(self, weight):
        super().__init__(f"Weight cannot be negative: {weight}")

class ExcessiveWeightError(WeightError):
    def __init__(self, weight, limit):
        super().__init__(f"Weight {weight} exceeds limit of {limit}")
```

Slide 5: Advanced Exception Attributes

Complex applications often require exceptions to carry additional data for debugging and logging purposes. Custom exceptions can include specialized attributes and methods to enhance error reporting.

```python
class DataValidationError(Exception):
    def __init__(self, value, expected_type, constraints=None):
        self.value = value
        self.expected_type = expected_type
        self.constraints = constraints or {}
        self.timestamp = datetime.now()
        message = self._build_message()
        super().__init__(message)

    def _build_message(self):
        return f"Validation failed for value {self.value} (type: {type(self.value)})"
```

Slide 6: Exception Context Management

Custom exceptions can be integrated with context managers to ensure proper resource handling and cleanup, even when errors occur during execution.

```python
class DatabaseConnection:
    class ConnectionError(Exception):
        def __init__(self, operation, details):
            self.operation = operation
            self.details = details
            super().__init__(f"Database {operation} failed: {details}")

    def __enter__(self):
        if not self.connect():
            raise self.ConnectionError("connect", "Unable to establish connection")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
```

Slide 7: Real-world Example - Data Processing Pipeline

This example demonstrates a practical implementation of custom exceptions in a data processing pipeline, handling various error conditions that might occur during data transformation.

```python
class DataProcessingError(Exception):
    def __init__(self, stage, error_type, details):
        self.stage = stage
        self.error_type = error_type
        self.details = details
        super().__init__(f"Error in {stage}: {error_type} - {details}")

def process_dataset(data):
    try:
        if not isinstance(data, list):
            raise DataProcessingError("validation", "TypeError", "Input must be a list")
        
        processed = []
        for idx, item in enumerate(data):
            if not item.strip():
                raise DataProcessingError("processing", "ValueError", 
                                       f"Empty value at index {idx}")
            processed.append(item.upper())
        return processed
    except DataProcessingError as e:
        print(f"Processing failed: {e}")
        return None
```

Slide 8: Exception Chaining

Exception chaining allows preservation of the original error while raising a new, more specific exception. This maintains the full error context for debugging purposes.

```python
class FileProcessingError(Exception):
    pass

def process_config_file(filename):
    try:
        with open(filename) as f:
            config = json.load(f)
    except FileNotFoundError as e:
        raise FileProcessingError(
            f"Configuration file {filename} not found"
        ) from e
    except json.JSONDecodeError as e:
        raise FileProcessingError(
            f"Invalid JSON in {filename}"
        ) from e
```

Slide 9: Custom Exception with Error Codes

Custom exceptions can incorporate error codes to provide standardized error handling across an application. This approach facilitates automated error processing and internationalization of error messages.

```python
class SystemError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"Error {code}: {message}")

    @classmethod
    def resource_not_found(cls, resource_id):
        return cls("E404", f"Resource {resource_id} not found")

    @classmethod
    def permission_denied(cls, operation):
        return cls("E403", f"Permission denied for operation: {operation}")
```

Slide 10: Custom Exception with Logging Integration

Integrating logging capabilities into custom exceptions enables automatic tracking of error occurrences and simplified debugging in production environments.

```python
import logging
from datetime import datetime

class LoggedError(Exception):
    def __init__(self, message, logger=None):
        self.timestamp = datetime.now()
        self.logger = logger or logging.getLogger(__name__)
        
        super().__init__(message)
        self._log_error()
    
    def _log_error(self):
        error_info = {
            'message': str(self),
            'timestamp': self.timestamp,
            'type': self.__class__.__name__
        }
        self.logger.error(f"Error occurred: {error_info}")
```

Slide 11: Real-world Example - API Request Handler

This implementation shows how custom exceptions can be used in an API request handler to manage different types of request failures and provide appropriate responses.

```python
class APIError(Exception):
    def __init__(self, status_code, message, details=None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(message)

class RequestHandler:
    def process_request(self, request_data):
        try:
            if not request_data:
                raise APIError(400, "Empty request body")
            
            if 'auth_token' not in request_data:
                raise APIError(401, "Missing authentication token")
            
            if not self.validate_token(request_data['auth_token']):
                raise APIError(403, "Invalid authentication token")
                
            return self.handle_validated_request(request_data)
            
        except APIError as e:
            return {
                'status': 'error',
                'code': e.status_code,
                'message': e.message,
                'details': e.details
            }
```

Slide 12: Exception Handler Decorator

Creating a decorator for exception handling provides a clean way to implement consistent error handling across multiple functions while maintaining code readability.

```python
from functools import wraps

def handle_exceptions(error_map=None):
    error_map = error_map or {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_type in error_map:
                    raise error_map[error_type](str(e))
                raise
        return wrapper
    return decorator

@handle_exceptions({
    ValueError: CustomError,
    KeyError: DataValidationError
})
def process_data(data):
    # Function implementation
    pass
```

Slide 13: Enhanced Exception Traceback

Custom exceptions can be enhanced with detailed traceback information to provide comprehensive debugging capabilities in complex applications.

```python
import traceback
import sys

class DetailedError(Exception):
    def __init__(self, message, **context):
        super().__init__(message)
        self.context = context
        self.traceback = self._capture_traceback()
    
    def _capture_traceback(self):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback:
            return ''.join(traceback.format_tb(exc_traceback))
        return ''.join(traceback.format_stack()[:-1])
    
    def get_error_details(self):
        return {
            'message': str(self),
            'context': self.context,
            'traceback': self.traceback
        }
```

Slide 14: Additional Resources

*   Building Better Python Exceptions: [https://arxiv.org/abs/cs/0701072](https://arxiv.org/abs/cs/0701072)
*   Exception Handling Patterns in Large-Scale Systems: [https://ieeexplore.ieee.org/document/8445076](https://ieeexplore.ieee.org/document/8445076)
*   Best Practices for Python Exception Handling: [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
*   Error Handling Patterns in Distributed Systems: [https://www.sciencedirect.com/science/article/pii/S0167642309000343](https://www.sciencedirect.com/science/article/pii/S0167642309000343)
*   Python Exception Handling - Advanced Topics: [https://realpython.com/python-exceptions/](https://realpython.com/python-exceptions/)

