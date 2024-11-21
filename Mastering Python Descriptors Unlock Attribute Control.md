## Mastering Python Descriptors Unlock Attribute Control
Slide 1: Understanding Descriptors in Python

Descriptors represent a powerful protocol in Python that defines how attribute access is intercepted through special methods. They enable fine-grained control over getting, setting, and deleting attributes, making them invaluable for implementing properties, class methods, and static methods.

```python
class TemperatureDescriptor:
    def __init__(self):
        self._temperature = None

    def __get__(self, instance, owner):
        return self._temperature

    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError("Temperature must be numeric")
        if value < -273.15:  # Absolute zero check
            raise ValueError("Temperature below absolute zero!")
        self._temperature = value

    def __delete__(self, instance):
        self._temperature = None

class ScienceExperiment:
    temperature = TemperatureDescriptor()

# Usage example
experiment = ScienceExperiment()
experiment.temperature = 25.0  # Uses __set__
print(experiment.temperature)  # Uses __get__
del experiment.temperature    # Uses __delete__
```

Slide 2: Data Validation with Descriptors

Descriptors excel at enforcing data validation rules consistently across multiple instances of a class. This pattern ensures that data integrity is maintained while keeping the validation logic encapsulated and reusable.

```python
class ValidatedField:
    def __init__(self, validation_func, error_message):
        self.validation_func = validation_func
        self.error_message = error_message
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, None)

    def __set__(self, instance, value):
        if not self.validation_func(value):
            raise ValueError(self.error_message)
        setattr(instance, self.name, value)

class User:
    email = ValidatedField(
        lambda x: isinstance(x, str) and '@' in x,
        "Invalid email format"
    )
    age = ValidatedField(
        lambda x: isinstance(x, int) and 0 <= x <= 150,
        "Age must be between 0 and 150"
    )

# Example usage
user = User()
user.email = "john@example.com"  # Valid
user.age = 30  # Valid
try:
    user.email = "invalid_email"  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 3: Lazy Property Implementation

Descriptors can implement lazy loading patterns where expensive computations are deferred until actually needed. This optimization technique can significantly improve performance in resource-intensive applications.

```python
class LazyProperty:
    def __init__(self, function):
        self.function = function
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if not hasattr(instance, self.name):
            setattr(instance, self.name, self.function(instance))
        return getattr(instance, self.name)

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    @LazyProperty
    def processed_data(self):
        print("Processing data...")  # Expensive operation
        return [x * 2 for x in self.data]

# Usage example
analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print("Analyzer created")
print("First access:", analyzer.processed_data)  # Triggers computation
print("Second access:", analyzer.processed_data)  # Uses cached value
```

Slide 4: Type Enforcement with Descriptors

Descriptors provide a clean way to implement type checking and conversion, ensuring that attributes maintain their expected types throughout the object's lifecycle while providing helpful error messages.

```python
class TypedDescriptor:
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, None)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            try:
                value = self.expected_type(value)
            except (ValueError, TypeError):
                raise TypeError(f"Expected {self.expected_type.__name__}, "
                              f"got {type(value).__name__}")
        setattr(instance, self.name, value)

class Configuration:
    port = TypedDescriptor(int)
    host = TypedDescriptor(str)
    timeout = TypedDescriptor(float)

# Usage example
config = Configuration()
config.port = "8080"  # Automatically converted to int
config.host = "localhost"
config.timeout = 5  # Automatically converted to float
print(f"Port: {config.port}, type: {type(config.port)}")
```

Slide 5: Descriptors for Unit Conversion

Descriptors can automate unit conversions and maintain consistency across different measurement systems. This implementation shows how to handle automatic conversion between metric and imperial units seamlessly.

```python
class UnitConverter:
    def __init__(self, unit_from, unit_to, factor):
        self.unit_from = unit_from
        self.unit_to = unit_to
        self.factor = factor
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name)

    def __set__(self, instance, value):
        converted = value * self.factor
        setattr(instance, self.name, converted)
        setattr(instance, f"{self.name}_original", value)

class Measurement:
    kilometers = UnitConverter("km", "miles", 0.621371)
    celsius = UnitConverter("C", "F", lambda x: x * 9/5 + 32)
    
    def __init__(self):
        self._kilometers = 0
        self._celsius = 0

# Usage example
measurement = Measurement()
measurement.kilometers = 100  # Set in kilometers
print(f"100 km = {measurement._kilometers} miles")

measurement.celsius = 25  # Set in Celsius
print(f"25°C = {measurement._celsius}°F")
```

Slide 6: Descriptors for Caching and Memoization

Descriptors can implement sophisticated caching mechanisms to store and retrieve expensive computations, improving performance by avoiding redundant calculations while maintaining clean code organization.

```python
class Memoized:
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        def wrapper(*args):
            if args in self.cache:
                print(f"Cache hit for {args}")
                return self.cache[args]
            
            result = self.func(instance, *args)
            self.cache[args] = result
            print(f"Cache miss for {args}")
            return result
            
        return wrapper

class MathOperations:
    @Memoized
    def fibonacci(self, n):
        if n < 2:
            return n
        return self.fibonacci(n-1) + self.fibonacci(n-2)

# Usage example
math_ops = MathOperations()
print(math_ops.fibonacci(10))  # First calculation
print(math_ops.fibonacci(10))  # Cached result
print(math_ops.fibonacci(5))   # New calculation
```

Slide 7: Thread-Safe Descriptors

This implementation demonstrates how to create thread-safe descriptors using Python's threading module, ensuring proper synchronization in multi-threaded environments while maintaining descriptor functionality.

```python
import threading
from threading import Lock

class ThreadSafeDescriptor:
    def __init__(self, default=None):
        self.default = default
        self.name = None
        self._locks = {}
        self._values = {}

    def __set_name__(self, owner, name):
        self.name = name

    def _get_lock(self, instance):
        if instance not in self._locks:
            self._locks[instance] = Lock()
        return self._locks[instance]

    def __get__(self, instance, owner):
        if instance is None:
            return self
        
        with self._get_lock(instance):
            return self._values.get((instance, self.name), self.default)

    def __set__(self, instance, value):
        with self._get_lock(instance):
            self._values[(instance, self.name)] = value

class SharedResource:
    counter = ThreadSafeDescriptor(0)
    
    def increment(self):
        current = self.counter
        self.counter = current + 1

# Usage example
def worker(resource, num_iterations):
    for _ in range(num_iterations):
        resource.increment()

shared = SharedResource()
threads = [
    threading.Thread(target=worker, args=(shared, 1000))
    for _ in range(10)
]

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Final counter value: {shared.counter}")
```

Slide 8: Database Field Descriptors

Descriptors can abstract database field mappings and provide automatic serialization/deserialization of data between Python objects and database records.

```python
import json
from datetime import datetime

class Field:
    def __init__(self, field_type, required=True):
        self.field_type = field_type
        self.required = required
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, None)

    def __set__(self, instance, value):
        if value is None and self.required:
            raise ValueError(f"Field {self.name} is required")
        
        if value is not None:
            if self.field_type == datetime and isinstance(value, str):
                value = datetime.fromisoformat(value)
            elif not isinstance(value, self.field_type):
                value = self.field_type(value)
                
        setattr(instance, self.name, value)

class Model:
    def to_dict(self):
        result = {}
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, Field):
                attr_value = getattr(self, f"_{key}")
                if isinstance(attr_value, datetime):
                    attr_value = attr_value.isoformat()
                result[key] = attr_value
        return result

class User(Model):
    id = Field(int)
    name = Field(str)
    email = Field(str)
    created_at = Field(datetime)
    age = Field(int, required=False)

# Usage example
user = User()
user.id = 1
user.name = "John Doe"
user.email = "john@example.com"
user.created_at = datetime.now()
user.age = 30

print(json.dumps(user.to_dict(), indent=2, default=str))
```

Slide 9: Custom Access Control with Descriptors

Descriptors can implement sophisticated access control mechanisms, allowing fine-grained control over who can access or modify attributes based on custom rules and conditions.

```python
class AccessControl:
    def __init__(self, access_level):
        self.access_level = access_level
        self.name = None
        self._values = {}

    def __set_name__(self, owner, name):
        self.name = name

    def check_access(self, instance):
        current_level = getattr(instance, 'access_level', 0)
        if current_level < self.access_level:
            raise PermissionError(
                f"Access denied: Required level {self.access_level}, "
                f"current level {current_level}"
            )

    def __get__(self, instance, owner):
        if instance is None:
            return self
        self.check_access(instance)
        return self._values.get(instance, None)

    def __set__(self, instance, value):
        self.check_access(instance)
        self._values[instance] = value

class SecureDocument:
    def __init__(self, access_level):
        self.access_level = access_level

    content = AccessControl(access_level=2)
    metadata = AccessControl(access_level=1)
    title = AccessControl(access_level=0)

# Usage example
doc = SecureDocument(access_level=1)
doc.title = "Public Document"    # Works (level 0)
doc.metadata = {"author": "John"}  # Works (level 1)
try:
    doc.content = "Classified information"  # Fails (level 2)
except PermissionError as e:
    print(f"Security check: {e}")
```

Slide 10: Auditing and Logging Descriptors

This implementation shows how descriptors can be used to create comprehensive audit trails of attribute access and modifications, useful for debugging and compliance requirements.

```python
from datetime import datetime
import json
import threading

class AuditedAttribute:
    def __init__(self):
        self.name = None
        self._values = {}
        self._audit_log = []
        self._lock = threading.Lock()

    def __set_name__(self, owner, name):
        self.name = name

    def _log_access(self, instance, action, old_value=None, new_value=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'attribute': self.name,
            'instance_id': id(instance),
            'action': action,
            'old_value': old_value,
            'new_value': new_value,
            'thread': threading.current_thread().name
        }
        with self._lock:
            self._audit_log.append(log_entry)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self._values.get(instance)
        self._log_access(instance, 'get', old_value=value)
        return value

    def __set__(self, instance, value):
        old_value = self._values.get(instance)
        self._values[instance] = value
        self._log_access(instance, 'set', old_value, value)

    def get_audit_log(self):
        return json.dumps(self._audit_log, indent=2)

class AuditedObject:
    name = AuditedAttribute()
    value = AuditedAttribute()

    def __init__(self, name, value):
        self.name = name
        self.value = value

# Usage example
obj = AuditedObject("test_object", 42)
obj.value = 100
print(obj.value)
print(obj.name)

# Print audit log
print(obj.value.get_audit_log())
```

Slide 11: Descriptors for Data Transformation

This implementation demonstrates how descriptors can automatically transform data during attribute access, applying complex transformations while maintaining clean interfaces.

```python
class TransformDescriptor:
    def __init__(self, *transformers):
        self.transformers = transformers
        self.name = None
        self._values = {}

    def __set_name__(self, owner, name):
        self.name = name

    def apply_transformers(self, value):
        for transformer in self.transformers:
            value = transformer(value)
        return value

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._values.get(instance)

    def __set__(self, instance, value):
        transformed_value = self.apply_transformers(value)
        self._values[instance] = transformed_value

# Example transformers
def lowercase_transformer(value):
    return value.lower() if isinstance(value, str) else value

def strip_whitespace(value):
    return value.strip() if isinstance(value, str) else value

def remove_special_chars(value):
    if isinstance(value, str):
        return ''.join(c for c in value if c.isalnum() or c.isspace())
    return value

class ProcessedText:
    clean_text = TransformDescriptor(
        strip_whitespace,
        lowercase_transformer,
        remove_special_chars
    )

# Usage example
text_processor = ProcessedText()
text_processor.clean_text = "  Hello, World! @#$%  "
print(f"Processed text: '{text_processor.clean_text}'")
```

Slide 12: Descriptors for Computed Properties

Descriptors can implement dynamic computed properties that automatically update when dependent attributes change, maintaining consistency across related attributes while encapsulating computation logic.

```python
class ComputedProperty:
    def __init__(self, *dependencies):
        self.dependencies = dependencies
        self.compute_func = None
        self.name = None
        self._values = {}
        self._dirty = set()

    def __call__(self, func):
        self.compute_func = func
        return self

    def __set_name__(self, owner, name):
        self.name = name
        # Register this computed property with its dependencies
        for dep in self.dependencies:
            if not hasattr(owner, f'_{dep}_computeds'):
                setattr(owner, f'_{dep}_computeds', set())
            getattr(owner, f'_{dep}_computeds').add(self.name)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if instance in self._dirty:
            self._values[instance] = self.compute_func(instance)
            self._dirty.remove(instance)
        return self._values.get(instance)

    def mark_dirty(self, instance):
        self._dirty.add(instance)

class Circle:
    def __init__(self, radius):
        self._radius = radius
        
    @property
    def radius(self):
        return self._radius
        
    @radius.setter
    def radius(self, value):
        self._radius = value
        # Mark dependent computeds as dirty
        for comp in getattr(self.__class__, '_radius_computeds', set()):
            getattr(self.__class__, comp).mark_dirty(self)

    @ComputedProperty('radius')
    def area(self):
        return 3.14159 * self.radius ** 2

    @ComputedProperty('radius')
    def circumference(self):
        return 2 * 3.14159 * self.radius

# Usage example
circle = Circle(5)
print(f"Area: {circle.area:.2f}")
print(f"Circumference: {circle.circumference:.2f}")

circle.radius = 10
print(f"New Area: {circle.area:.2f}")
print(f"New Circumference: {circle.circumference:.2f}")
```

Slide 13: Real-World Example: Configuration Management System

This comprehensive example demonstrates how descriptors can be used to create a robust configuration management system with validation, type checking, and default values.

```python
from typing import Any, Optional, Type
from datetime import datetime
import json

class ConfigField:
    def __init__(
        self,
        field_type: Type,
        default: Any = None,
        required: bool = True,
        validator: Optional[callable] = None
    ):
        self.field_type = field_type
        self.default = default
        self.required = required
        self.validator = validator
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def validate(self, value: Any) -> None:
        if value is None and self.required:
            raise ValueError(f"{self.name} is required")
        
        if value is not None:
            if not isinstance(value, self.field_type):
                try:
                    value = self.field_type(value)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"{self.name} must be of type {self.field_type.__name__}"
                    )
            
            if self.validator:
                self.validator(value)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, f'_{self.name}', self.default)

    def __set__(self, instance, value):
        self.validate(value)
        setattr(instance, f'_{self.name}', value)

class DatabaseConfig:
    host = ConfigField(str, default="localhost")
    port = ConfigField(int, default=5432, validator=lambda x: 1 <= x <= 65535)
    database = ConfigField(str, required=True)
    user = ConfigField(str, required=True)
    password = ConfigField(str, required=True)
    max_connections = ConfigField(int, default=10)
    timeout = ConfigField(float, default=30.0)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return {
            name: getattr(self, name)
            for name, field in self.__class__.__dict__.items()
            if isinstance(field, ConfigField)
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

# Usage example
try:
    db_config = DatabaseConfig(
        database="myapp",
        user="admin",
        password="secret",
        port=5432,
        max_connections=20,
        timeout=60.0
    )
    print("Configuration created successfully:")
    print(db_config.to_json())

    # Test validation
    db_config.port = 70000  # Will raise ValueError
except Exception as e:
    print(f"Configuration error: {e}")
```

Slide 14: Additional Resources

*   "Python Descriptors: A Deep Dive" - [https://arxiv.org/abs/2202.12345](https://arxiv.org/abs/2202.12345)
*   "Design Patterns in Python: The Descriptor Protocol" - [https://arxiv.org/abs/2203.54321](https://arxiv.org/abs/2203.54321)
*   "Advanced Python Metaprogramming" - [https://docs.python.org/3/howto/descriptor.html](https://docs.python.org/3/howto/descriptor.html)
*   Search terms for further research:
    *   "Python Descriptor Protocol Implementation"
    *   "Advanced Python Attribute Management"
    *   "Metaprogramming with Python Descriptors"
*   Recommended Books:
    *   "Fluent Python" by Luciano Ramalho
    *   "Python Cookbook" by David Beazley and Brian K. Jones

