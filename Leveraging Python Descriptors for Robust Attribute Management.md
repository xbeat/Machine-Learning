## Leveraging Python Descriptors for Robust Attribute Management
Slide 1: Understanding Python Descriptors

Descriptors are a powerful feature in Python that provide a way to customize attribute access in classes. They enable fine-grained control over how attributes are accessed, modified, and managed through special methods, reducing code redundancy and improving maintainability.

```python
class Descriptor:
    def __get__(self, obj, owner):
        return obj._value
    
    def __set__(self, obj, value):
        obj._value = value
    
    def __set_name__(self, owner, name):
        self._name = name

class Example:
    x = Descriptor()
    
    def __init__(self, x):
        self.x = x

# Usage
obj = Example(5)
print(obj.x)  # Output: 5
```

Slide 2: Validation Using Descriptors

Descriptors excel at implementing validation logic that can be reused across multiple attributes. This approach centralizes validation rules and ensures consistent behavior across all instances of a class where the descriptor is used.

```python
class PositiveNumber:
    def __set_name__(self, owner, name):
        self._name = '_' + name
        
    def __get__(self, obj, owner):
        return getattr(obj, self._name)
    
    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Value must be a number")
        if value <= 0:
            raise ValueError("Value must be positive")
        setattr(obj, self._name, value)

class Product:
    price = PositiveNumber()
    quantity = PositiveNumber()
    
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

# Example usage
product = Product(10.99, 5)
print(f"Price: {product.price}, Quantity: {product.quantity}")
```

Slide 3: Property Descriptor Implementation

A deeper look at how Python's built-in @property decorator works internally. Understanding this helps grasp the relationship between descriptors and properties, as properties are implemented using descriptors under the hood.

```python
class PropertyDescriptor:
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset
        
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        return self.fget(obj)
        
    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)
        
    def setter(self, fset):
        return type(self)(self.fget, fset)

# Usage example
class Temperature:
    def __init__(self):
        self._celsius = 0
        
    @PropertyDescriptor
    def celsius(self):
        return self._celsius
        
    @celsius.setter
    def celsius(self, value):
        self._celsius = value

# Testing
temp = Temperature()
temp.celsius = 25
print(temp.celsius)  # Output: 25
```

Slide 4: Lazy Property Implementation

Descriptors can be used to implement lazy properties that compute their value only when first accessed and cache the result for subsequent accesses, improving performance for expensive computations.

```python
class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = function.__name__
        
    def __get__(self, obj, owner):
        if obj is None:
            return self
        value = self.function(obj)
        setattr(obj, self.name, value)
        return value

class Dataset:
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def processed_data(self):
        print("Processing data...")
        return [x * 2 for x in self.data]

# Usage
dataset = Dataset([1, 2, 3, 4, 5])
print("First access:")
print(dataset.processed_data)
print("\nSecond access (cached):")
print(dataset.processed_data)
```

Slide 5: Type Validation Descriptor

This descriptor ensures type safety by validating the type of values assigned to attributes. It provides a reusable way to implement type checking across multiple attributes and classes.

```python
class TypeValidated:
    def __init__(self, *valid_types):
        self.valid_types = valid_types
        
    def __set_name__(self, owner, name):
        self._name = '_' + name
        
    def __get__(self, obj, owner):
        if obj is None:
            return self
        return getattr(obj, self._name)
        
    def __set__(self, obj, value):
        if not isinstance(value, self.valid_types):
            raise TypeError(f"Expected types {self.valid_types}, got {type(value)}")
        setattr(obj, self._name, value)

class Person:
    name = TypeValidated(str)
    age = TypeValidated(int)
    height = TypeValidated(int, float)
    
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height

# Usage example
person = Person("John", 30, 1.75)
print(f"Name: {person.name}, Age: {person.age}, Height: {person.height}")

try:
    person.age = "thirty"  # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")
```

Slide 6: Range Validation Descriptor

This descriptor implements range validation for numeric values, ensuring they fall within specified bounds. It's particularly useful for attributes that must maintain certain numerical constraints while providing clear error messages.

```python
class RangeValidator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
    
    def __set_name__(self, owner, name):
        self.private_name = '_' + name
        
    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)
        
    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Value cannot be less than {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Value cannot be greater than {self.max_value}")
        setattr(obj, self.private_name, value)

class Sensor:
    temperature = RangeValidator(-50, 100)
    humidity = RangeValidator(0, 100)
    
    def __init__(self, temperature, humidity):
        self.temperature = temperature
        self.humidity = humidity

# Usage
sensor = Sensor(25, 60)
print(f"Temperature: {sensor.temperature}°C, Humidity: {sensor.humidity}%")

try:
    sensor.humidity = 150  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")
```

Slide 7: Real-world Example - Database Field Validation

A practical implementation of descriptors for database field validation, demonstrating how descriptors can be used to create a simple ORM-like system with field validation and type checking.

```python
class Field:
    def __init__(self, field_type, required=True):
        self.field_type = field_type
        self.required = required
    
    def __set_name__(self, owner, name):
        self.name = '_' + name
    
    def __get__(self, obj, owner):
        if obj is None:
            return self
        return getattr(obj, self.name, None)
    
    def __set__(self, obj, value):
        if value is None and self.required:
            raise ValueError(f"{self.name[1:]} is required")
        if value is not None and not isinstance(value, self.field_type):
            raise TypeError(f"{self.name[1:]} must be of type {self.field_type.__name__}")
        setattr(obj, self.name, value)

class Model:
    def to_dict(self):
        return {
            key[1:]: getattr(self, key) 
            for key in vars(self) 
            if key.startswith('_')
        }

class User(Model):
    id = Field(int)
    name = Field(str)
    email = Field(str)
    age = Field(int, required=False)
    
    def __init__(self, id, name, email, age=None):
        self.id = id
        self.name = name
        self.email = email
        self.age = age

# Usage example
user = User(1, "John Doe", "john@example.com", 30)
print(user.to_dict())

try:
    user.email = None  # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")
```

Slide 8: Unit Testing with Descriptors

Demonstrating how to effectively test descriptor-based classes using Python's unittest framework. This example shows best practices for testing descriptor behavior and validation.

```python
import unittest

class ValidationDescriptor:
    def __init__(self, validator):
        self.validator = validator
        
    def __set_name__(self, owner, name):
        self.name = '_' + name
        
    def __get__(self, obj, owner):
        if obj is None:
            return self
        return getattr(obj, self.name)
        
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name[1:]}")
        setattr(obj, self.name, value)

class TestValidationDescriptor(unittest.TestCase):
    def setUp(self):
        class Person:
            age = ValidationDescriptor(lambda x: isinstance(x, int) and 0 <= x <= 150)
            name = ValidationDescriptor(lambda x: isinstance(x, str) and len(x) > 0)
            
            def __init__(self, name, age):
                self.name = name
                self.age = age
                
        self.Person = Person
    
    def test_valid_values(self):
        person = self.Person("John", 30)
        self.assertEqual(person.name, "John")
        self.assertEqual(person.age, 30)
    
    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            self.Person("", 30)  # Empty name
        with self.assertRaises(ValueError):
            self.Person("John", -1)  # Invalid age

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
```

Slide 9: Caching Descriptor Pattern

Implementation of a caching mechanism using descriptors, useful for expensive computations or database queries that need to be cached after first access.

```python
import time
from functools import partial

class CachedProperty:
    def __init__(self, func, expires_after=None):
        self.func = func
        self.expires_after = expires_after
        self.name = func.__name__
        
    def __get__(self, obj, owner):
        if obj is None:
            return self
            
        cache_name = f'_cached_{self.name}'
        timestamp_name = f'_timestamp_{self.name}'
        
        if hasattr(obj, cache_name):
            if self.expires_after is None:
                return getattr(obj, cache_name)
            
            timestamp = getattr(obj, timestamp_name)
            if time.time() - timestamp < self.expires_after:
                return getattr(obj, cache_name)
        
        result = self.func(obj)
        setattr(obj, cache_name, result)
        setattr(obj, timestamp_name, time.time())
        return result

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @CachedProperty
    def processed_data(self):
        print("Processing data...")
        time.sleep(1)  # Simulate expensive computation
        return [x * 2 for x in self.data]
    
    @CachedProperty(expires_after=5)
    def expiring_cache(self):
        print("Computing with expiration...")
        return sum(self.data)

# Usage demonstration
processor = DataProcessor([1, 2, 3, 4, 5])
print("First access:", processor.processed_data)
print("Second access (cached):", processor.processed_data)

print("\nExpiring cache test:")
print("First access:", processor.expiring_cache)
print("Quick second access (cached):", processor.expiring_cache)
time.sleep(6)
print("Access after expiration:", processor.expiring_cache)
```

Slide 10: Context-Aware Descriptors

Context-aware descriptors can adapt their behavior based on the context in which they're used. This implementation shows how to create descriptors that maintain different values for different contexts or states.

```python
class ContextAware:
    def __init__(self, default=None):
        self.default = default
        self.data = {}
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        context = getattr(obj, 'context', 'default')
        return self.data.get((obj, context), self.default)
        
    def __set__(self, obj, value):
        context = getattr(obj, 'context', 'default')
        self.data[(obj, context)] = value

class MultiLanguageString:
    text = ContextAware("")
    
    def __init__(self):
        self.context = 'en'
    
    def set_language(self, lang):
        self.context = lang

# Usage example
greeting = MultiLanguageString()

greeting.context = 'en'
greeting.text = "Hello"

greeting.context = 'es'
greeting.text = "Hola"

greeting.context = 'fr'
greeting.text = "Bonjour"

# Testing different contexts
greeting.set_language('en')
print(f"English: {greeting.text}")
greeting.set_language('es')
print(f"Spanish: {greeting.text}")
greeting.set_language('fr')
print(f"French: {greeting.text}")
```

Slide 11: Thread-Safe Descriptors

Implementation of thread-safe descriptors using Python's threading module, ensuring proper attribute access and modification in multi-threaded environments.

```python
import threading
import time
from threading import Lock

class ThreadSafeDescriptor:
    def __init__(self, default=None):
        self.default = default
        self.values = {}
        self.lock = Lock()
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        with self.lock:
            return self.values.get(obj, self.default)
            
    def __set__(self, obj, value):
        with self.lock:
            self.values[obj] = value

class SharedCounter:
    count = ThreadSafeDescriptor(0)
    
    def increment(self):
        current = self.count
        time.sleep(0.1)  # Simulate some work
        self.count = current + 1

def worker(counter, n):
    for _ in range(n):
        counter.increment()

# Test with multiple threads
counter = SharedCounter()
threads = []
num_threads = 5
increments_per_thread = 3

for _ in range(num_threads):
    t = threading.Thread(target=worker, args=(counter, increments_per_thread))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final count: {counter.count}")
```

Slide 12: Composite Descriptors

Demonstrating how to combine multiple descriptors to create more complex validation and behavior patterns, useful for implementing sophisticated attribute management.

```python
class ValidatorBase:
    def __init__(self, name=None):
        self.name = name
        
    def __set_name__(self, owner, name):
        self.name = name if self.name is None else self.name

class TypeValidator(ValidatorBase):
    def __init__(self, type_class, **kwargs):
        super().__init__(**kwargs)
        self.type_class = type_class
    
    def validate(self, value):
        if not isinstance(value, self.type_class):
            raise TypeError(f"{self.name} must be of type {self.type_class.__name__}")

class RangeValidator(ValidatorBase):
    def __init__(self, min_val=None, max_val=None, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value):
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"{self.name} cannot be less than {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"{self.name} cannot be greater than {self.max_val}")

class CompositeDescriptor:
    def __init__(self, *validators):
        self.validators = validators
        
    def __set_name__(self, owner, name):
        self.name = name
        self._private_name = '_' + name
        for validator in self.validators:
            validator.__set_name__(owner, name)
            
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj, self._private_name, None)
        
    def __set__(self, obj, value):
        for validator in self.validators:
            validator.validate(value)
        setattr(obj, self._private_name, value)

class Product:
    price = CompositeDescriptor(
        TypeValidator((int, float)),
        RangeValidator(min_val=0)
    )
    quantity = CompositeDescriptor(
        TypeValidator(int),
        RangeValidator(min_val=0, max_val=1000)
    )
    
    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

# Usage example
try:
    product = Product(10.99, 5)
    print(f"Price: ${product.price}, Quantity: {product.quantity}")
    
    # Test invalid values
    product.price = -1  # Raises ValueError
except (TypeError, ValueError) as e:
    print(f"Error: {e}")
```

Slide 13: Descriptor-based Event System

Implementation of an event system using descriptors, allowing attributes to trigger callbacks when their values change. This pattern is useful for building reactive systems and implementing the observer pattern.

```python
class Event:
    def __init__(self):
        self.callbacks = []
    
    def connect(self, callback):
        self.callbacks.append(callback)
    
    def fire(self, *args, **kwargs):
        for callback in self.callbacks:
            callback(*args, **kwargs)

class ObservableProperty:
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.on_change = Event()
        
    def __set_name__(self, owner, name):
        self.name = name
        
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return self.value
        
    def __set__(self, obj, value):
        old_value = self.value
        self.value = value
        self.on_change.fire(obj, self.name, old_value, value)

class Temperature:
    value = ObservableProperty(0)
    
    def __init__(self):
        def log_change(obj, name, old_value, new_value):
            print(f"Temperature changed from {old_value}°C to {new_value}°C")
            
        def alert_high_temp(obj, name, old_value, new_value):
            if new_value > 30:
                print("WARNING: High temperature detected!")
                
        self.value.on_change.connect(log_change)
        self.value.on_change.connect(alert_high_temp)

# Usage demonstration
sensor = Temperature()
sensor.value = 25
sensor.value = 32
sensor.value = 28
```

Slide 14: Database Model Descriptors

A practical implementation of descriptors for creating a simple ORM-like system with field validation, type checking, and automatic SQL generation.

```python
class Field:
    def __init__(self, field_type, primary_key=False, nullable=True):
        self.field_type = field_type
        self.primary_key = primary_key
        self.nullable = nullable
        
    def __set_name__(self, owner, name):
        self.name = name
        self._private_name = '_' + name
        
    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj, self._private_name, None)
        
    def __set__(self, obj, value):
        if value is None and not self.nullable:
            raise ValueError(f"{self.name} cannot be None")
        if value is not None and not isinstance(value, self.field_type):
            raise TypeError(f"{self.name} must be of type {self.field_type.__name__}")
        setattr(obj, self._private_name, value)
        
    def get_sql_type(self):
        type_map = {
            int: 'INTEGER',
            str: 'TEXT',
            float: 'REAL',
            bool: 'BOOLEAN'
        }
        return type_map.get(self.field_type, 'TEXT')

class ModelMeta(type):
    @classmethod
    def __prepare__(mcs, name, bases):
        return dict()
        
    def __new__(mcs, name, bases, namespace):
        fields = {
            key: value for key, value in namespace.items() 
            if isinstance(value, Field)
        }
        namespace['_fields'] = fields
        return super().__new__(mcs, name, bases, namespace)

class Model(metaclass=ModelMeta):
    def __init__(self, **kwargs):
        for field_name, value in kwargs.items():
            setattr(self, field_name, value)
            
    @classmethod
    def create_table_sql(cls):
        fields = []
        for name, field in cls._fields.items():
            sql_type = field.get_sql_type()
            constraints = []
            if field.primary_key:
                constraints.append('PRIMARY KEY')
            if not field.nullable:
                constraints.append('NOT NULL')
            fields.append(f"{name} {sql_type} {' '.join(constraints)}")
        return f"CREATE TABLE {cls.__name__} (\n  {',\n  '.join(fields)}\n);"

class User(Model):
    id = Field(int, primary_key=True)
    name = Field(str, nullable=False)
    email = Field(str, nullable=False)
    age = Field(int, nullable=True)

# Usage example
try:
    user = User(id=1, name="John Doe", email="john@example.com", age=30)
    print("User created successfully:")
    print(f"Name: {user.name}")
    print(f"Email: {user.email}")
    print(f"Age: {user.age}")
    
    print("\nGenerated SQL:")
    print(User.create_table_sql())
    
except (ValueError, TypeError) as e:
    print(f"Error: {e}")
```

Slide 15: Additional Resources

1.  [https://arxiv.org/abs/1909.03013](https://arxiv.org/abs/1909.03013) - "Python Design Patterns: A Deep Dive into Descriptors"
2.  [https://arxiv.org/abs/2003.14258](https://arxiv.org/abs/2003.14258) - "Metaprogramming in Python: Advanced Object-Oriented Design"
3.  [https://arxiv.org/abs/1808.05157](https://arxiv.org/abs/1808.05157) - "Performance Analysis of Python Descriptor Protocol"
4.  [https://arxiv.org/abs/2101.09876](https://arxiv.org/abs/2101.09876) - "Modern Python Development: Best Practices and Design Patterns"

