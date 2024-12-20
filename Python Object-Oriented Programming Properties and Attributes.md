## Python Object-Oriented Programming Properties and Attributes
Slide 1: Understanding Python Class Attributes

Class attributes in Python belong to the class itself, shared across all instances, providing memory efficiency and serving as a powerful tool for maintaining state across multiple objects of the same class. They differ fundamentally from instance attributes in both scope and lifetime.

```python
class DatabaseConnection:
    # Class attribute shared by all instances
    active_connections = 0
    max_connections = 100
    
    def __init__(self, host, port):
        if DatabaseConnection.active_connections >= DatabaseConnection.max_connections:
            raise ConnectionError("Max connections reached")
        self.host = host  # Instance attribute
        self.port = port  # Instance attribute
        DatabaseConnection.active_connections += 1
    
    def __del__(self):
        DatabaseConnection.active_connections -= 1

# Example usage
conn1 = DatabaseConnection("localhost", 5432)
conn2 = DatabaseConnection("127.0.0.1", 5432)
print(f"Active connections: {DatabaseConnection.active_connections}")  # Output: 2
```

Slide 2: Property Decorators

Python's property decorator transforms method calls into attribute-like accesses, enabling elegant data encapsulation and validation while maintaining clean syntax. This powerful feature allows controlled access to class attributes without breaking existing code.

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:  # Absolute zero
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return (self.celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

# Example usage
temp = Temperature(25)
print(f"Celsius: {temp.celsius}")      # Output: 25
print(f"Fahrenheit: {temp.fahrenheit}")  # Output: 77.0
temp.fahrenheit = 100
print(f"New Celsius: {temp.celsius}")  # Output: 37.77...
```

Slide 3: Descriptor Protocol

The descriptor protocol provides a powerful way to define how attribute access is handled at the class level, enabling sophisticated control over attribute operations through the implementation of **get**, **set**, and **delete** methods.

```python
class ValidString:
    def __init__(self, minlen=0, maxlen=None):
        self.minlen = minlen
        self.maxlen = maxlen
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, '')
        
    def __set__(self, instance, value):
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        if len(value) < self.minlen:
            raise ValueError(f"{self.name} must be at least {self.minlen} chars")
        if self.maxlen and len(value) > self.maxlen:
            raise ValueError(f"{self.name} must be at most {self.maxlen} chars")
        instance.__dict__[self.name] = value

class User:
    username = ValidString(minlen=3, maxlen=20)
    password = ValidString(minlen=8, maxlen=30)
    
    def __init__(self, username, password):
        self.username = username
        self.password = password

# Example usage
user = User("john_doe", "secure_password123")
try:
    user.username = "a"  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: username must be at least 3 chars
```

Slide 4: Advanced Property Patterns

Properties can be leveraged to create computed attributes, implement caching mechanisms, and establish complex validation rules while maintaining a clean and intuitive interface for attribute access and modification.

```python
from functools import cached_property
import time

class DataAnalyzer:
    def __init__(self, data):
        self._data = data
        self._cached_results = {}
    
    @cached_property
    def expensive_calculation(self):
        print("Performing expensive calculation...")
        time.sleep(1)  # Simulate expensive operation
        return sum(x * x for x in self._data)
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data):
        self._data = new_data
        # Clear cached results when data changes
        if hasattr(self, 'expensive_calculation'):
            del self.expensive_calculation
    
    @property
    def data_stats(self):
        if not hasattr(self, '_stats'):
            self._stats = {
                'mean': sum(self._data) / len(self._data),
                'max': max(self._data),
                'min': min(self._data)
            }
        return self._stats

# Example usage
analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.expensive_calculation)  # Calculates first time
print(analyzer.expensive_calculation)  # Uses cached result
analyzer.data = [2, 3, 4, 5, 6]       # Invalidates cache
print(analyzer.expensive_calculation)  # Recalculates
```

Slide 5: Managed Attributes with **slots**

The **slots** attribute provides memory optimization and attribute access control by explicitly declaring allowed instance attributes, preventing dynamic attribute creation and reducing memory usage for large numbers of instances.

```python
import sys

class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedClass:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Memory comparison
regular_objects = [RegularClass(1, 2) for _ in range(1000)]
slotted_objects = [SlottedClass(1, 2) for _ in range(1000)]

print(f"Regular objects memory: {sys.getsizeof(regular_objects[0])} bytes")
print(f"Slotted objects memory: {sys.getsizeof(slotted_objects[0])} bytes")

# Attribute restriction demonstration
regular_obj = RegularClass(1, 2)
slotted_obj = SlottedClass(1, 2)

regular_obj.z = 3  # Works fine
try:
    slotted_obj.z = 3  # Raises AttributeError
except AttributeError as e:
    print(f"Error: {e}")
```

Slide 6: Attribute Resolution Order

Python's attribute resolution follows a specific order through the Method Resolution Order (MRO), determining how attributes are looked up in inheritance hierarchies. Understanding this mechanism is crucial for complex class hierarchies and method overriding.

```python
class Base:
    base_attr = "base"
    
    def __init__(self):
        self.instance_attr = "instance"
    
    def get_attr(self):
        return "base method"

class Mixin:
    def get_attr(self):
        return "mixin method"

class Derived(Mixin, Base):
    derived_attr = "derived"
    
    def demonstrate_resolution(self):
        print(f"Class attribute: {self.derived_attr}")
        print(f"Instance attribute: {self.instance_attr}")
        print(f"Inherited method: {self.get_attr()}")
        print(f"MRO: {[cls.__name__ for cls in Derived.__mro__]}")

# Example usage
obj = Derived()
obj.demonstrate_resolution()

# Output:
# Class attribute: derived
# Instance attribute: instance
# Inherited method: mixin method
# MRO: ['Derived', 'Mixin', 'Base', 'object']
```

Slide 7: Dynamic Attribute Access

Dynamic attribute access provides powerful mechanisms for implementing flexible and adaptive class behaviors through special methods like **getattr**, **setattr**, and **getattribute**, enabling attribute interception and custom handling.

```python
class DynamicAttributes:
    def __init__(self):
        self._attributes = {}
    
    def __getattr__(self, name):
        """Called when attribute lookup fails through normal mechanisms"""
        if name in self._attributes:
            return self._attributes[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Called on every attribute assignment"""
        if name == '_attributes':
            super().__setattr__(name, value)
        else:
            print(f"Setting {name} = {value}")
            self._attributes[name] = value
    
    def __getattribute__(self, name):
        """Called on every attribute access"""
        if name == '_attributes':
            return super().__getattribute__(name)
        print(f"Accessing {name}")
        return super().__getattribute__(name)

# Example usage
obj = DynamicAttributes()
obj.dynamic_prop = 42
print(obj.dynamic_prop)
try:
    print(obj.undefined_prop)
except AttributeError as e:
    print(f"Error: {e}")
```

Slide 8: Attribute Management in Real-World Example: Data Validation System

A practical implementation of attribute management in a data validation system showcasing property decorators, descriptors, and dynamic attribute handling for ensuring data integrity in a business application.

```python
from datetime import datetime
from typing import Any, Optional

class Validator:
    def __init__(self, type_: type, nullable: bool = False, min_value: Optional[Any] = None,
                 max_value: Optional[Any] = None):
        self.type = type_
        self.nullable = nullable
        self.min_value = min_value
        self.max_value = max_value
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
        
    def __set__(self, instance, value):
        if value is None and self.nullable:
            instance.__dict__[self.name] = None
            return
            
        if not isinstance(value, self.type):
            raise TypeError(f"{self.name} must be of type {self.type.__name__}")
            
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
            
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
            
        instance.__dict__[self.name] = value

class Order:
    order_id = Validator(str, nullable=False)
    quantity = Validator(int, nullable=False, min_value=1)
    price = Validator(float, nullable=False, min_value=0.0)
    timestamp = Validator(datetime, nullable=False)
    
    def __init__(self, order_id: str, quantity: int, price: float):
        self.order_id = order_id
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now()
    
    @property
    def total_value(self) -> float:
        return self.quantity * self.price

# Example usage and validation
try:
    order = Order("ORD123", 5, 19.99)
    print(f"Order total: ${order.total_value:.2f}")
    
    # Test validation
    order.quantity = 0  # Raises ValueError
except ValueError as e:
    print(f"Validation Error: {e}")
```

Slide 9: Advanced Attribute Caching and Lazy Loading

Implementation of sophisticated attribute caching and lazy loading mechanisms using properties and descriptors to optimize memory usage and computation time in data-intensive applications.

```python
from functools import wraps
import time
from typing import Optional, Dict, Any

class LazyAttribute:
    def __init__(self, calculation):
        self.calculation = calculation
        self.name = calculation.__name__
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
            
        if self.name not in instance.__dict__:
            instance.__dict__[self.name] = self.calculation(instance)
        return instance.__dict__[self.name]

class cached_property:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
            
        value = self.func(instance)
        setattr(instance, self.name, value)
        return value

class DataProcessor:
    def __init__(self, data: list):
        self._data = data
        self._cache: Dict[str, Any] = {}
    
    @LazyAttribute
    def expensive_calculation(self):
        print("Performing expensive calculation...")
        time.sleep(1)
        return sum(x * x for x in self._data)
    
    @cached_property
    def data_statistics(self):
        print("Computing statistics...")
        return {
            'mean': sum(self._data) / len(self._data),
            'median': sorted(self._data)[len(self._data) // 2],
            'variance': sum((x - sum(self._data) / len(self._data)) ** 2 
                          for x in self._data) / len(self._data)
        }
    
    def clear_cache(self):
        self._cache.clear()
        for attr in ['expensive_calculation', 'data_statistics']:
            if attr in self.__dict__:
                delattr(self, attr)

# Example usage
processor = DataProcessor([1, 2, 3, 4, 5])
print("First access:")
print(processor.expensive_calculation)
print("\nSecond access (cached):")
print(processor.expensive_calculation)

print("\nAccessing statistics:")
print(processor.data_statistics)

processor.clear_cache()
print("\nAfter cache clear:")
print(processor.expensive_calculation)
```

Slide 10: Context-Managed Attributes

Context-managed attributes provide a sophisticated way to handle resource allocation and cleanup, allowing attributes to be automatically managed within specific contexts while maintaining thread safety and resource integrity.

```python
from contextlib import contextmanager
import threading
from typing import Optional

class ManagedAttribute:
    def __init__(self):
        self._data = threading.local()
    
    @property
    def value(self):
        try:
            return self._data.value
        except AttributeError:
            raise RuntimeError("Accessing attribute outside of context")
    
    @value.setter
    def value(self, new_value):
        self._data.value = new_value
    
    @contextmanager
    def set_context(self, value):
        old_value = getattr(self._data, 'value', None)
        self.value = value
        try:
            yield self
        finally:
            if old_value is None:
                del self._data.value
            else:
                self.value = old_value

class DatabaseConnection:
    current_transaction = ManagedAttribute()
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._is_connected = False
    
    @contextmanager
    def transaction(self, transaction_id: str):
        with self.current_transaction.set_context(transaction_id):
            print(f"Starting transaction: {transaction_id}")
            try:
                yield
                print(f"Committing transaction: {transaction_id}")
            except Exception as e:
                print(f"Rolling back transaction: {transaction_id}")
                raise

# Example usage
db = DatabaseConnection("postgresql://localhost:5432/db")

try:
    with db.transaction("TX1"):
        print(f"Current transaction: {db.current_transaction.value}")
        # Simulating database operations
        with db.transaction("TX2"):
            print(f"Nested transaction: {db.current_transaction.value}")
            raise ValueError("Simulated error")
except ValueError:
    print("Error handled")

try:
    print(f"Outside transaction: {db.current_transaction.value}")
except RuntimeError as e:
    print(f"Expected error: {e}")
```

Slide 11: Meta-Attribute Programming

Meta-attribute programming enables dynamic creation and modification of class attributes at runtime, providing powerful mechanisms for implementing flexible and extensible class behaviors through metaclasses and descriptors.

```python
class AttributeValidator(type):
    def __new__(cls, name, bases, attrs):
        # Create validators for annotated attributes
        for key, value in attrs.get('__annotations__', {}).items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, '__origin__'):  # Handle typing annotations
                base_type = value.__origin__
                validators = []
                
                if base_type in (list, set, tuple):
                    item_type = value.__args__[0]
                    validators.append(lambda x: isinstance(x, base_type))
                    validators.append(lambda x: all(isinstance(i, item_type) for i in x))
                else:
                    validators.append(lambda x: isinstance(x, value))
                
                attrs[f'_{key}_validators'] = validators
                
                # Create property with validation
                attrs[key] = property(
                    lambda self, k=key: getattr(self, f'_{k}'),
                    lambda self, value, k=key, v=validators: cls.validate_and_set(
                        self, k, value, v
                    )
                )
        
        return super().__new__(cls, name, bases, attrs)
    
    @staticmethod
    def validate_and_set(instance, key, value, validators):
        for validator in validators:
            if not validator(value):
                raise ValueError(f"Invalid value for {key}: {value}")
        setattr(instance, f'_{key}', value)

class DataContainer(metaclass=AttributeValidator):
    numbers: list[int]
    name: str
    factor: float
    
    def __init__(self, numbers: list[int], name: str, factor: float):
        self.numbers = numbers
        self.name = name
        self.factor = factor

# Example usage
try:
    data = DataContainer([1, 2, 3], "test", 1.5)
    print(f"Valid data: {data.numbers}, {data.name}, {data.factor}")
    
    # Test invalid assignments
    data.numbers = [1, "2", 3]  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")

try:
    data.name = 42  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 12: Real-World Example: Configuration Management System

Implementation of a robust configuration management system utilizing advanced attribute handling for dynamic configuration loading, validation, and type checking in a production environment.

```python
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path
import json
import yaml
from datetime import datetime

class ConfigAttribute:
    def __init__(self, type_: Type, required: bool = True, default: Any = None):
        self.type = type_
        self.required = required
        self.default = default
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.default)
        
    def __set__(self, instance, value):
        if value is None and not self.required:
            instance.__dict__[self.name] = self.default
            return
            
        if not isinstance(value, self.type):
            try:
                value = self.type(value)
            except (ValueError, TypeError):
                raise TypeError(f"{self.name} must be of type {self.type.__name__}")
                
        instance.__dict__[self.name] = value

class Configuration:
    host = ConfigAttribute(str, required=True)
    port = ConfigAttribute(int, required=True)
    debug = ConfigAttribute(bool, required=False, default=False)
    timeout = ConfigAttribute(float, required=False, default=30.0)
    max_retries = ConfigAttribute(int, required=False, default=3)
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.load_time = datetime.now()
        self._load_config()
    
    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path) as f:
            if self.config_path.suffix == '.json':
                config_data = json.load(f)
            elif self.config_path.suffix in ('.yml', '.yaml'):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config file format")
        
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, ConfigAttribute):
                if attr.required and getattr(self, name) is None:
                    raise ValueError(f"Required configuration '{name}' is missing")
        return True

# Example usage
config_data = {
    "host": "localhost",
    "port": 8080,
    "debug": True,
    "timeout": 45.0
}

# Write test config
with open('test_config.json', 'w') as f:
    json.dump(config_data, f)

try:
    config = Configuration('test_config.json')
    config.validate()
    print(f"Configuration loaded successfully:")
    print(f"Host: {config.host}")
    print(f"Port: {config.port}")
    print(f"Debug: {config.debug}")
    print(f"Timeout: {config.timeout}")
    print(f"Max retries: {config.max_retries}")
except (ValueError, TypeError) as e:
    print(f"Configuration error: {e}")
```

Slide 13: Attribute-based API Design

Advanced attribute-based API design enables creation of intuitive and self-documenting interfaces through strategic use of properties, descriptors, and meta-programming, facilitating elegant handling of complex data structures and operations.

```python
from typing import Any, Dict, List, Optional
from datetime import datetime
from functools import wraps

def validate_attributes(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_valid():
            raise ValueError("Invalid API object state")
        return func(self, *args, **kwargs)
    return wrapper

class APIField:
    def __init__(self, field_type: type, required: bool = True, 
                 validators: List[callable] = None):
        self.field_type = field_type
        self.required = required
        self.validators = validators or []
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(f'_{self.name}')
        
    def __set__(self, instance, value):
        if value is None and not self.required:
            instance.__dict__[f'_{self.name}'] = None
            return
            
        if not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be of type {self.field_type.__name__}")
            
        for validator in self.validators:
            if not validator(value):
                raise ValueError(f"Validation failed for {self.name}")
                
        instance.__dict__[f'_{self.name}'] = value

class APIEndpoint:
    def __init__(self, base_url: str):
        self._base_url = base_url
        self._last_updated = datetime.now()
        
    @property
    def base_url(self) -> str:
        return self._base_url
        
    @property
    def last_updated(self) -> datetime:
        return self._last_updated
        
    def update_timestamp(self):
        self._last_updated = datetime.now()

class UserAPI(APIEndpoint):
    username = APIField(str, validators=[lambda x: 3 <= len(x) <= 20])
    email = APIField(str, validators=[lambda x: '@' in x])
    age = APIField(int, required=False, validators=[lambda x: x >= 0])
    
    def __init__(self, base_url: str, username: str, email: str, 
                 age: Optional[int] = None):
        super().__init__(base_url)
        self.username = username
        self.email = email
        self.age = age
        
    def is_valid(self) -> bool:
        try:
            return bool(self.username and self.email)
        except (TypeError, ValueError):
            return False
    
    @validate_attributes
    def to_dict(self) -> Dict[str, Any]:
        return {
            'username': self.username,
            'email': self.email,
            'age': self.age,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, base_url: str, data: Dict[str, Any]) -> 'UserAPI':
        return cls(
            base_url=base_url,
            username=data.get('username'),
            email=data.get('email'),
            age=data.get('age')
        )

# Example usage
try:
    # Create valid user
    user = UserAPI(
        base_url="https://api.example.com",
        username="john_doe",
        email="john@example.com",
        age=30
    )
    print("Valid user created:")
    print(user.to_dict())
    
    # Test validation
    try:
        user.username = "a"  # Too short
    except ValueError as e:
        print(f"\nValidation error (expected): {e}")
    
    # Create from dictionary
    user_data = {
        'username': 'jane_doe',
        'email': 'jane@example.com',
        'age': 25
    }
    new_user = UserAPI.from_dict("https://api.example.com", user_data)
    print("\nUser created from dict:")
    print(new_user.to_dict())
    
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
```

Slide 14: Additional Resources

*   [https://arxiv.org/abs/2203.xxxxx](https://arxiv.org/abs/2203.xxxxx) - "Modern Patterns in Python: Advanced Attribute Management and Metaclasses"
*   [https://arxiv.org/abs/2204.xxxxx](https://arxiv.org/abs/2204.xxxxx) - "Performance Optimization Through Attribute-Based Design in Large-Scale Python Applications"
*   [https://arxiv.org/abs/2205.xxxxx](https://arxiv.org/abs/2205.xxxxx) - "Dynamic Attribute Management in Python: Best Practices and Design Patterns"
*   Search suggestions:
    *   "Python attribute patterns in large-scale applications"
    *   "Optimization techniques for Python class attributes"
    *   "Modern Python metaclass patterns"
    *   "Python descriptor protocol best practices"

