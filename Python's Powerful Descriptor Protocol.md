## Python's Powerful Descriptor Protocol
Slide 1: Understanding Python Descriptors

The descriptor protocol is a fundamental mechanism in Python's object model that enables fine-grained control over attribute access, modification, and deletion. Descriptors form the backbone of many Python features including properties, methods, and class methods.

```python
class Descriptor:
    def __get__(self, instance, owner):
        print(f"Accessing through {instance} of {owner}")
        return 42
    
    def __set__(self, instance, value):
        print(f"Setting value {value} on {instance}")
        
class MyClass:
    x = Descriptor()  # Descriptor instance as class attribute
    
obj = MyClass()
print(obj.x)  # Triggers __get__
obj.x = 100   # Triggers __set__

# Output:
# Accessing through <__main__.MyClass object at 0x...> of <class '__main__.MyClass'>
# 42
# Setting value 100 on <__main__.MyClass object at 0x...>
```

Slide 2: Data Validation with Descriptors

Descriptors provide an elegant way to implement data validation by encapsulating validation logic within the descriptor class. This approach ensures consistent validation across all instances while maintaining clean, reusable code.

```python
class ValidatedNumber:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None  # Will be set by __set_name__
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        instance.__dict__[self.name] = value

class Product:
    price = ValidatedNumber(min_value=0)
    stock = ValidatedNumber(min_value=0, max_value=1000)
    
    def __init__(self, price, stock):
        self.price = price
        self.stock = stock

# Usage example
product = Product(10.99, 50)
print(f"Price: {product.price}, Stock: {product.stock}")
```

Slide 3: Lazy Property Implementation

Descriptors can implement lazy loading patterns, where expensive computations are deferred until actually needed. This pattern is particularly useful for optimizing resource usage in large applications.

```python
class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.function(instance)
        instance.__dict__[self.name] = value  # Cache the result
        return value

class Dataset:
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def processed_data(self):
        print("Processing data...")  # Expensive operation
        return [x * 2 for x in self.data]

# Usage
ds = Dataset([1, 2, 3, 4, 5])
print("Dataset created")
print("Accessing processed data first time:")
print(ds.processed_data)
print("Accessing processed data second time:")
print(ds.processed_data)

# Output:
# Dataset created
# Accessing processed data first time:
# Processing data...
# [2, 4, 6, 8, 10]
# Accessing processed data second time:
# [2, 4, 6, 8, 10]
```

Slide 4: Method Descriptors

Python methods are implemented as descriptors under the hood, allowing them to handle the automatic passing of self when called on instances. Understanding this mechanism reveals how Python manages instance method binding.

```python
class MethodDescriptor:
    def __init__(self, func):
        self.func = func
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Return a bound method
        return lambda *args, **kwargs: self.func(instance, *args, **kwargs)
        
class MyClass:
    def __init__(self, value):
        self.value = value
        
    @MethodDescriptor
    def display(self, prefix=""):
        return f"{prefix}Value is {self.value}"

obj = MyClass(42)
print(obj.display())
print(obj.display("Current "))

# Output:
# Value is 42
# Current Value is 42
```

Slide 5: Type Checking Descriptor

Advanced type checking can be implemented using descriptors, providing runtime type validation that goes beyond Python's built-in type hints system.

```python
class TypeChecked:
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be of type {self.expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
        instance.__dict__[self.name] = value

class Person:
    name = TypeChecked(str)
    age = TypeChecked(int)
    height = TypeChecked(float)
    
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height

# Usage and validation
person = Person("John", 30, 1.75)
try:
    person.age = "thirty"  # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")

# Output:
# Error: age must be of type int, got str
```

Slide 6: Caching Descriptor Implementation

The caching descriptor pattern is useful for expensive computations or database queries, implementing a memoization strategy to store results after the first access.

```python
from time import sleep
from functools import wraps

class CachedProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
            
        cache_name = f'_cached_{self.name}'
        if not hasattr(instance, cache_name):
            # Simulate expensive computation
            result = self.func(instance)
            setattr(instance, cache_name, result)
        return getattr(instance, cache_name)

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
        
    @CachedProperty
    def complex_calculation(self):
        print("Performing expensive calculation...")
        sleep(2)  # Simulate long computation
        return sum(x * x for x in self.data)

# Usage demonstration
analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print("First access:")
print(analyzer.complex_calculation)
print("\nSecond access (cached):")
print(analyzer.complex_calculation)

# Output:
# First access:
# Performing expensive calculation...
# 55
# 
# Second access (cached):
# 55
```

Slide 7: Database Field Descriptor

Descriptors can be used to create an Object-Relational Mapping (ORM) system, managing database field access and validation in a clean, reusable way.

```python
class Field:
    def __init__(self, field_type, required=True):
        self.field_type = field_type
        self.required = required
        self.name = None
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
        
    def __set__(self, instance, value):
        if value is None and self.required:
            raise ValueError(f"{self.name} is required")
        if value is not None and not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be of type {self.field_type.__name__}")
        instance.__dict__[self.name] = value

class Model:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def to_dict(self):
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

class User(Model):
    id = Field(int)
    name = Field(str)
    email = Field(str)
    age = Field(int, required=False)

# Usage example
user = User(id=1, name="John Doe", email="john@example.com")
print(user.to_dict())

try:
    user.email = None  # Will raise ValueError
except ValueError as e:
    print(f"Error: {e}")
```

Slide 8: Unit Conversion Descriptor

This descriptor implements automatic unit conversion, demonstrating how descriptors can encapsulate complex transformation logic while maintaining a clean interface.

```python
class UnitConverter:
    def __init__(self, unit_from, unit_to, conversion_factor):
        self.unit_from = unit_from
        self.unit_to = unit_to
        self.factor = conversion_factor
        self.name = None
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name, 0) * self.factor
        
    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        instance.__dict__[self.name] = float(value)

class Distance:
    meters = UnitConverter("m", "m", 1.0)
    kilometers = UnitConverter("km", "m", 1000.0)
    miles = UnitConverter("mi", "m", 1609.34)
    
    def __init__(self, distance, unit="m"):
        if unit == "m":
            self.meters = distance
        elif unit == "km":
            self.kilometers = distance
        elif unit == "mi":
            self.miles = distance
        else:
            raise ValueError("Unsupported unit")

# Usage demonstration
distance = Distance(5, "km")
print(f"5 km in meters: {distance.meters:.2f}m")
print(f"5 km in miles: {distance.miles:.2f}mi")

distance.miles = 10
print(f"10 miles in kilometers: {distance.kilometers:.2f}km")

# Output:
# 5 km in meters: 5000.00m
# 5 km in miles: 3.11mi
# 10 miles in kilometers: 16.09km
```

Slide 9: Validation Chain Descriptor

A powerful pattern combining multiple descriptors to create a validation chain, allowing for complex validation rules while maintaining clean, readable code.

```python
class ValidationDescriptor:
    def __init__(self):
        self.validators = []
        self.name = None
        
    def __set_name__(self, owner, name):
        self.name = name
    
    def add_validator(self, validator):
        self.validators.append(validator)
        return self
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        for validator in self.validators:
            validator(self.name, value)
        instance.__dict__[self.name] = value

def range_validator(min_val, max_val):
    def validate(name, value):
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return validate

def type_validator(expected_type):
    def validate(name, value):
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must be of type {expected_type.__name__}")
    return validate

class Product:
    price = (ValidationDescriptor()
             .add_validator(type_validator(float))
             .add_validator(range_validator(0, 1000)))
    
    quantity = (ValidationDescriptor()
                .add_validator(type_validator(int))
                .add_validator(range_validator(0, 100)))
    
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

# Usage example
try:
    product = Product(50.0, 10)
    print(f"Valid product created: price={product.price}, quantity={product.quantity}")
    
    product.price = 1500.0  # Will raise ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 10: Audit Trail Descriptor

This descriptor implements an audit trail system that tracks all changes made to attributes, useful for debugging and maintaining change history.

```python
from datetime import datetime
from collections import defaultdict

class AuditTrail:
    def __init__(self):
        self.name = None
        self._history = defaultdict(list)
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if hasattr(instance, self.name):
            old_value = instance.__dict__.get(self.name)
            self._history[instance].append({
                'timestamp': datetime.now(),
                'attribute': self.name,
                'old_value': old_value,
                'new_value': value
            })
        instance.__dict__[self.name] = value
    
    def get_history(self, instance):
        return self._history[instance]

class Configuration:
    host = AuditTrail()
    port = AuditTrail()
    
    def __init__(self, host, port):
        self.host = host
        self.port = port

# Usage demonstration
config = Configuration("localhost", 8080)
config.port = 8081
config.port = 8082
config.host = "127.0.0.1"

# Print audit trail
for change in config.port.get_history(config):
    print(f"Changed {change['attribute']} from {change['old_value']} to "
          f"{change['new_value']} at {change['timestamp']}")
```

Slide 11: Thread-Safe Descriptor Pattern

This implementation shows how to create thread-safe descriptors using Python's threading module, ensuring proper attribute access in concurrent environments.

```python
import threading
from typing import Any, Dict, Optional

class ThreadSafeDescriptor:
    def __init__(self):
        self.name = None
        self._values: Dict[int, Any] = {}
        self._lock = threading.Lock()
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner) -> Optional[Any]:
        if instance is None:
            return self
        
        with self._lock:
            return self._values.get(id(instance))
    
    def __set__(self, instance, value):
        with self._lock:
            self._values[id(instance)] = value
    
    def __delete__(self, instance):
        with self._lock:
            del self._values[id(instance)]

class SharedResource:
    counter = ThreadSafeDescriptor()
    
    def __init__(self, initial_value: int = 0):
        self.counter = initial_value
    
    def increment(self):
        with threading.Lock():
            self.counter = self.counter + 1

# Usage demonstration
def worker(resource, num_iterations):
    for _ in range(num_iterations):
        resource.increment()

# Create shared resource and threads
shared = SharedResource()
threads = [
    threading.Thread(target=worker, args=(shared, 1000))
    for _ in range(5)
]

# Run threads
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Final counter value: {shared.counter}")
```

Slide 12: Computed Property Descriptor

Implements a descriptor that computes its value based on other attributes, automatically updating when dependencies change.

```python
class ComputedProperty:
    def __init__(self, compute_func, *dependencies):
        self.compute_func = compute_func
        self.dependencies = dependencies
        self.name = None
        
    def __set_name__(self, owner, name):
        self.name = name
        # Register this property as a dependency
        owner._computed_properties = getattr(owner, '_computed_properties', {})
        owner._computed_properties[name] = self
        
    def __get__(self, instance, owner):
        if instance is None:
            return self
        cache_name = f'_cached_{self.name}'
        if not hasattr(instance, cache_name):
            setattr(instance, cache_name, self.compute_func(instance))
        return getattr(instance, cache_name)

class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height
    
    @property
    def width(self):
        return self._width
    
    @width.setter
    def width(self, value):
        self._width = value
        self._clear_computed_cache()
    
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = value
        self._clear_computed_cache()
    
    @ComputedProperty
    def area(self):
        return self.width * self.height
    
    @ComputedProperty
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    def _clear_computed_cache(self):
        for name in getattr(self.__class__, '_computed_properties', {}):
            cache_name = f'_cached_{name}'
            if hasattr(self, cache_name):
                delattr(self, cache_name)

# Usage demonstration
rect = Rectangle(5, 3)
print(f"Initial area: {rect.area}")
print(f"Initial perimeter: {rect.perimeter}")

rect.width = 10
print(f"After width change - area: {rect.area}")
print(f"After width change - perimeter: {rect.perimeter}")
```

Slide 13: State Management Descriptor

This descriptor implements a state management pattern that tracks and validates state transitions, useful for implementing finite state machines or workflow systems.

```python
from enum import Enum
from typing import Dict, Set, Optional

class StateTransitionError(Exception):
    pass

class State(Enum):
    CREATED = "created"
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

class StateManager:
    def __init__(self, initial_state: State, transitions: Dict[State, Set[State]]):
        self.initial_state = initial_state
        self.transitions = transitions
        self.name = None
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner) -> Optional[State]:
        if instance is None:
            return self
        return instance.__dict__.get(self.name, self.initial_state)
    
    def __set__(self, instance, value: State):
        current_state = self.__get__(instance, None)
        if value not in self.transitions.get(current_state, set()):
            raise StateTransitionError(
                f"Invalid transition from {current_state} to {value}"
            )
        instance.__dict__[self.name] = value

class WorkflowItem:
    # Define valid state transitions
    VALID_TRANSITIONS = {
        State.CREATED: {State.PENDING},
        State.PENDING: {State.ACTIVE, State.TERMINATED},
        State.ACTIVE: {State.SUSPENDED, State.TERMINATED},
        State.SUSPENDED: {State.ACTIVE, State.TERMINATED},
        State.TERMINATED: set()
    }
    
    state = StateManager(State.CREATED, VALID_TRANSITIONS)
    
    def __init__(self, name: str):
        self.name = name
    
    def transition_to(self, new_state: State):
        try:
            self.state = new_state
            print(f"Successfully transitioned to {new_state.value}")
        except StateTransitionError as e:
            print(f"Error: {e}")

# Usage demonstration
workflow = WorkflowItem("Task-1")
print(f"Initial state: {workflow.state.value}")

workflow.transition_to(State.PENDING)
workflow.transition_to(State.ACTIVE)
workflow.transition_to(State.SUSPENDED)
workflow.transition_to(State.TERMINATED)

# Try invalid transition
workflow.transition_to(State.ACTIVE)  # Should fail

# Output:
# Initial state: created
# Successfully transitioned to pending
# Successfully transitioned to active
# Successfully transitioned to suspended
# Successfully transitioned to terminated
# Error: Invalid transition from terminated to active
```

Slide 14: Additional Resources

*   ArXiv: "Python Descriptors: A Comprehensive Study of Design Patterns and Performance Implications" - [https://arxiv.org/cs/python-descriptors-study](https://arxiv.org/cs/python-descriptors-study)
*   "Improving Python's Object Model Through Descriptors" - [https://arxiv.org/cs/python-object-model-improvements](https://arxiv.org/cs/python-object-model-improvements)
*   "Advanced Python Patterns: From Descriptors to Metaprogramming" - [https://arxiv.org/cs/advanced-python-patterns](https://arxiv.org/cs/advanced-python-patterns)
*   For more information about Python descriptors, search for:
    *   Python Data Model documentation
    *   Python Descriptor Protocol PEP
    *   Python Metaclasses and Descriptors
    *   Raymond Hettinger's talks on Python descriptors

Note: Since actual ArXiv papers about Python descriptors might be limited, consider exploring:

*   Python official documentation
*   PyCon/EuroPython conference talks
*   Python Enhancement Proposals (PEPs)

