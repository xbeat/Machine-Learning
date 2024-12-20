## Mastering isinstance() and issubclass() in Python
Slide 1: Understanding isinstance() Fundamentals

The isinstance() function is a crucial Python built-in that checks if an object belongs to a specified class or type. It returns True if the object is an instance of the class or any of its subclasses, making it essential for type checking and polymorphic code.

```python
# Demonstrating basic isinstance() usage
class Animal:
    pass

class Dog(Animal):
    pass

dog = Dog()
number = 42
text = "Hello"

# Check instance types
print(isinstance(dog, Dog))      # Output: True
print(isinstance(dog, Animal))   # Output: True
print(isinstance(number, int))   # Output: True
print(isinstance(text, float))   # Output: False
```

Slide 2: Multiple Type Checking with isinstance()

The isinstance() function can check multiple types simultaneously using a tuple of types as its second argument. This feature is particularly useful when implementing flexible functions that can handle various input types.

```python
def process_data(value):
    # Check if value is either int, float, or str
    if isinstance(value, (int, float)):
        return value * 2
    elif isinstance(value, str):
        return value.upper()
    return None

# Example usage
print(process_data(5))       # Output: 10
print(process_data(3.14))    # Output: 6.28
print(process_data("test"))  # Output: "TEST"
```

Slide 3: Custom Class Type Checking

Understanding how isinstance() works with custom class hierarchies is essential for implementing robust object-oriented designs. The function respects the inheritance chain and can verify complex class relationships.

```python
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

class Car(Vehicle):
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

class ElectricCar(Car):
    def __init__(self, brand, model, battery_capacity):
        super().__init__(brand, model)
        self.battery_capacity = battery_capacity

# Create instances
tesla = ElectricCar("Tesla", "Model S", 100)
regular_car = Car("Toyota", "Camry")

# Check inheritance chain
print(isinstance(tesla, ElectricCar))  # Output: True
print(isinstance(tesla, Car))          # Output: True
print(isinstance(tesla, Vehicle))      # Output: True
print(isinstance(regular_car, ElectricCar))  # Output: False
```

Slide 4: Advanced Type Checking with Abstract Base Classes

The isinstance() function seamlessly integrates with Python's Abstract Base Classes (ABC), enabling powerful interface checking and ensuring proper implementation of abstract methods.

```python
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data):
        pass

class NumericProcessor(DataProcessor):
    def process(self, data):
        return data * 2

class StringProcessor(DataProcessor):
    def process(self, data):
        return data.lower()

# Create instances and check types
num_processor = NumericProcessor()
str_processor = StringProcessor()

print(isinstance(num_processor, DataProcessor))  # Output: True
print(isinstance(str_processor, DataProcessor))  # Output: True
```

Slide 5: Understanding issubclass() Fundamentals

The issubclass() function determines if a class is a subclass of another class. Unlike isinstance(), it works with class objects rather than instances, making it valuable for metaclass programming and class relationship verification.

```python
class Animal:
    pass

class Mammal(Animal):
    pass

class Dog(Mammal):
    pass

# Check class relationships
print(issubclass(Dog, Mammal))      # Output: True
print(issubclass(Dog, Animal))      # Output: True
print(issubclass(Mammal, Animal))   # Output: True
print(issubclass(Animal, Mammal))   # Output: False
```

Slide 6: Multiple Base Class Checking with issubclass()

The issubclass() function effectively handles multiple inheritance scenarios, allowing developers to verify relationships between classes that inherit from multiple base classes. This is crucial for complex class hierarchies.

```python
class Flyable:
    def fly(self):
        pass

class Swimmable:
    def swim(self):
        pass

class Duck(Flyable, Swimmable):
    pass

class Penguin(Swimmable):
    pass

# Check multiple inheritance relationships
print(issubclass(Duck, Flyable))     # Output: True
print(issubclass(Duck, Swimmable))   # Output: True
print(issubclass(Penguin, Flyable))  # Output: False
print(issubclass(Duck, (Flyable, Swimmable)))  # Output: True
```

Slide 7: Type Checking in Generic Functions

Implementing type-safe generic functions using isinstance() enables robust code that can handle multiple data types while maintaining type safety and providing meaningful error messages.

```python
def safe_operation(value):
    """Performs type-safe operations on different data types"""
    try:
        if isinstance(value, (int, float)):
            return value ** 2
        elif isinstance(value, str):
            if value.isdigit():
                return int(value) ** 2
            return len(value)
        elif isinstance(value, (list, tuple)):
            return sum(x for x in value if isinstance(x, (int, float)))
        raise TypeError(f"Unsupported type: {type(value)}")
    except Exception as e:
        return f"Error: {str(e)}"

# Test with different types
print(safe_operation(5))          # Output: 25
print(safe_operation("123"))      # Output: 15129
print(safe_operation([1,2,"3",4.0])) # Output: 7.0
print(safe_operation(complex(1,2)))  # Output: "Error: Unsupported type: <class 'complex'>"
```

Slide 8: Runtime Type Verification System

Creating a decorator-based type verification system using isinstance() enables automatic type checking for function parameters and return values during runtime execution.

```python
from functools import wraps
from typing import get_type_hints

def type_check(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints for the function
        hints = get_type_hints(func)
        
        # Check positional arguments
        for arg, value in zip(func.__code__.co_varnames, args):
            if arg in hints:
                if not isinstance(value, hints[arg]):
                    raise TypeError(f"Argument {arg} must be {hints[arg]}, got {type(value)}")
        
        # Check keyword arguments
        for key, value in kwargs.items():
            if key in hints:
                if not isinstance(value, hints[key]):
                    raise TypeError(f"Argument {key} must be {hints[key]}, got {type(value)}")
        
        result = func(*args, **kwargs)
        
        # Check return type
        if 'return' in hints and not isinstance(result, hints['return']):
            raise TypeError(f"Return value must be {hints['return']}, got {type(result)}")
        
        return result
    return wrapper

# Example usage
@type_check
def process_user_data(name: str, age: int, scores: list) -> dict:
    return {
        'name': name,
        'age': age,
        'average_score': sum(scores) / len(scores)
    }

# Test the function
try:
    result = process_user_data("John", 25, [90, 85, 88])
    print(result)  # Output: {'name': 'John', 'age': 25, 'average_score': 87.66666666666667}
    
    # This will raise a TypeError
    result = process_user_data("Jane", "30", [85, 90])
except TypeError as e:
    print(f"Type Error: {e}")
```

Slide 9: Dynamic Method Resolution

Using isinstance() for dynamic method resolution enables polymorphic behavior and flexible handling of different object types while maintaining code maintainability.

```python
class DataHandler:
    def handle_data(self, data):
        # Find the appropriate handler method based on data type
        for cls in type(data).__mro__:
            handler_name = f'handle_{cls.__name__.lower()}'
            handler = getattr(self, handler_name, None)
            if handler and callable(handler):
                return handler(data)
        raise ValueError(f"No handler found for type: {type(data)}")
    
    def handle_int(self, data):
        return data * 2
    
    def handle_float(self, data):
        return round(data, 2)
    
    def handle_str(self, data):
        return data.upper()
    
    def handle_list(self, data):
        return [self.handle_data(item) for item in data]

# Test the dynamic handler
handler = DataHandler()
print(handler.handle_data(42))        # Output: 84
print(handler.handle_data(3.14159))   # Output: 3.14
print(handler.handle_data("hello"))   # Output: "HELLO"
print(handler.handle_data([1, "test", 3.14]))  # Output: [2, "TEST", 3.14]
```

Slide 10: Custom Type Checking Protocol

Creating a custom type checking protocol demonstrates advanced usage of isinstance() and issubclass() for implementing duck typing with explicit interface verification.

```python
class ValidationProtocol:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'validate') and 
                callable(subclass.validate) and 
                hasattr(subclass, 'clean') and 
                callable(subclass.clean) or 
                NotImplemented)

class DataValidator(ValidationProtocol):
    def validate(self, data):
        raise NotImplementedError
    
    def clean(self, data):
        raise NotImplementedError

class EmailValidator:
    def validate(self, email):
        return '@' in email and '.' in email
    
    def clean(self, email):
        return email.strip().lower()

# Test the protocol
validator = EmailValidator()
print(isinstance(validator, ValidationProtocol))  # Output: True
print(issubclass(EmailValidator, ValidationProtocol))  # Output: True

# Example usage
email = " Test@Example.com "
if isinstance(validator, ValidationProtocol):
    if validator.validate(email):
        clean_email = validator.clean(email)
        print(f"Cleaned email: {clean_email}")  # Output: "test@example.com"
```

Slide 11: Type-Safe Dispatch System

Implementing a type-safe dispatch system showcases advanced usage of isinstance() for creating a flexible and extensible method dispatch mechanism based on argument types.

```python
class TypeDispatcher:
    def __init__(self):
        self.registry = {}
    
    def register(self, *types):
        def decorator(func):
            self.registry[types] = func
            return func
        return decorator
    
    def dispatch(self, *args):
        for types, func in self.registry.items():
            if len(args) == len(types) and all(isinstance(arg, t) for arg, t in zip(args, types)):
                return func(*args)
        raise TypeError(f"No matching function for types: {tuple(type(arg) for arg in args)}")

# Example usage
dispatcher = TypeDispatcher()

@dispatcher.register(int, int)
def add_ints(x, y):
    return x + y

@dispatcher.register(str, str)
def concat_strings(x, y):
    return f"{x} {y}"

@dispatcher.register(list, list)
def merge_lists(x, y):
    return [*x, *y]

# Test dispatching
print(dispatcher.dispatch(1, 2))           # Output: 3
print(dispatcher.dispatch("Hello", "World"))  # Output: "Hello World"
print(dispatcher.dispatch([1, 2], [3, 4]))    # Output: [1, 2, 3, 4]

try:
    dispatcher.dispatch(1, "test")  # This will raise TypeError
except TypeError as e:
    print(f"Error: {e}")
```

Slide 12: Real-world Application: Data Pipeline Validation

This example demonstrates a practical implementation of type checking in a data processing pipeline, ensuring data integrity through different transformation stages.

```python
from typing import Union, Dict, List
import json

class DataProcessor:
    def __init__(self):
        self.transformations = []
    
    def add_transformation(self, func, expected_type):
        self.transformations.append((func, expected_type))
    
    def process(self, data: Union[Dict, List]) -> Dict:
        current_data = data
        
        for step, (transform_func, expected_type) in enumerate(self.transformations, 1):
            if not isinstance(current_data, expected_type):
                raise TypeError(
                    f"Step {step}: Expected {expected_type.__name__}, "
                    f"got {type(current_data).__name__}"
                )
            current_data = transform_func(current_data)
        
        return current_data

# Example pipeline setup
processor = DataProcessor()

def parse_json(data: str) -> Dict:
    return json.loads(data)

def validate_fields(data: Dict) -> Dict:
    required = {'name', 'age', 'email'}
    if not all(field in data for field in required):
        raise ValueError("Missing required fields")
    return data

def transform_data(data: Dict) -> Dict:
    return {
        'user_info': {
            'full_name': data['name'].upper(),
            'age_group': f"{data['age'] // 10 * 10}s",
            'contact': data['email'].lower()
        }
    }

# Register transformations
processor.add_transformation(parse_json, str)
processor.add_transformation(validate_fields, dict)
processor.add_transformation(transform_data, dict)

# Test the pipeline
input_data = '{"name": "John Doe", "age": 35, "email": "John@Example.com"}'

try:
    result = processor.process(input_data)
    print(json.dumps(result, indent=2))
except (TypeError, ValueError) as e:
    print(f"Error: {e}")
```

Slide 13: Real-world Application: Type-Safe API Request Handler

This implementation showcases a robust API request handler using isinstance() and issubclass() for request validation, parameter checking, and response formatting in a production environment.

```python
from datetime import datetime
from typing import Any, Dict, Optional, Union
import json

class RequestValidator:
    def __init__(self):
        self._validators = {}
        
    def register_validator(self, param_name: str, expected_type: type):
        self._validators[param_name] = expected_type
    
    def validate(self, params: Dict[str, Any]) -> bool:
        for param_name, value in params.items():
            if param_name in self._validators:
                if not isinstance(value, self._validators[param_name]):
                    raise ValueError(
                        f"Parameter '{param_name}' must be of type "
                        f"{self._validators[param_name].__name__}, got {type(value).__name__}"
                    )
        return True

class APIResponse:
    def __init__(self, status: int, data: Optional[Dict] = None, error: Optional[str] = None):
        self.status = status
        self.data = data
        self.error = error
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp
        }

class APIRequestHandler:
    def __init__(self):
        self.validator = RequestValidator()
        
    def setup_validation(self):
        self.validator.register_validator('user_id', int)
        self.validator.register_validator('email', str)
        self.validator.register_validator('active', bool)
        self.validator.register_validator('metadata', dict)
    
    def process_request(self, request_data: Dict[str, Any]) -> APIResponse:
        try:
            # Validate request parameters
            self.validator.validate(request_data)
            
            # Process the request
            processed_data = self._process_user_data(request_data)
            return APIResponse(status=200, data=processed_data)
            
        except ValueError as e:
            return APIResponse(status=400, error=str(e))
        except Exception as e:
            return APIResponse(status=500, error="Internal server error")
    
    def _process_user_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'user_id': data['user_id'],
            'email_verified': '@' in data['email'],
            'status': 'active' if data['active'] else 'inactive',
            'metadata': {
                k: v for k, v in data['metadata'].items()
                if isinstance(v, (str, int, bool, float))
            }
        }

# Example usage
handler = APIRequestHandler()
handler.setup_validation()

# Test valid request
valid_request = {
    'user_id': 12345,
    'email': 'user@example.com',
    'active': True,
    'metadata': {
        'last_login': '2024-01-01',
        'login_count': 42,
        'preferences': {'theme': 'dark'}
    }
}

# Test invalid request
invalid_request = {
    'user_id': '12345',  # Wrong type (string instead of int)
    'email': 'user@example.com',
    'active': True,
    'metadata': {}
}

# Process requests and print responses
print("Valid Request Response:")
print(json.dumps(handler.process_request(valid_request).to_dict(), indent=2))

print("\nInvalid Request Response:")
print(json.dumps(handler.process_request(invalid_request).to_dict(), indent=2))
```

Slide 14: Additional Resources

*   arXiv Papers and Related Resources:

*   General Type Systems in Python:
    *   "Type Systems for Python: Past, Present and Future"
    *   Search: [https://arxiv.org/search/?query=python+type+system&searchtype=all](https://arxiv.org/search/?query=python+type+system&searchtype=all)
*   Advanced Object-Oriented Design:
    *   "Design Patterns in Dynamic Programming"
    *   Search: [https://arxiv.org/search/?query=design+patterns+dynamic+programming&searchtype=all](https://arxiv.org/search/?query=design+patterns+dynamic+programming&searchtype=all)
*   Type Checking in Production Systems:
    *   "Runtime Type Checking for Production Systems"
    *   Search: [https://scholar.google.com/scholar?q=runtime+type+checking+production+systems](https://scholar.google.com/scholar?q=runtime+type+checking+production+systems)
*   Recommended Documentation:
    *   Python Official Documentation: [https://docs.python.org/3/library/functions.html#isinstance](https://docs.python.org/3/library/functions.html#isinstance)
    *   Python Type Hints PEP 484: [https://www.python.org/dev/peps/pep-0484/](https://www.python.org/dev/peps/pep-0484/)
    *   Python Abstract Base Classes: [https://docs.python.org/3/library/abc.html](https://docs.python.org/3/library/abc.html)

