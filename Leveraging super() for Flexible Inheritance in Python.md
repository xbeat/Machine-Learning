## Leveraging super() for Flexible Inheritance in Python
Slide 1: Basic Inheritance with super()

The super() function provides a clean way to access methods from parent classes in Python, enabling proper inheritance chains. It automatically handles method resolution order (MRO) and supports multiple inheritance scenarios elegantly.

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Call parent's __init__
        self.breed = breed
    
    def speak(self):
        return f"{self.name} barks loudly!"

# Example usage
dog = Dog("Rex", "German Shepherd")
print(dog.speak())  # Output: Rex barks loudly!
```

Slide 2: Multiple Inheritance and Method Resolution Order

Python's super() function follows the Method Resolution Order (MRO) when dealing with multiple inheritance, ensuring predictable method calls across complex class hierarchies without explicit parent class references.

```python
class Device:
    def __init__(self, model):
        self.model = model
    
    def start(self):
        return f"Device {self.model} starting"

class Bluetooth:
    def __init__(self, version):
        self.version = version
    
    def connect(self):
        return f"Connecting via Bluetooth {self.version}"

class SmartPhone(Device, Bluetooth):
    def __init__(self, model, bt_version):
        Device.__init__(self, model)
        Bluetooth.__init__(self, bt_version)
        
    def info(self):
        return f"{self.start()} and {self.connect()}"

phone = SmartPhone("iPhone", "5.0")
print(phone.info())
```

Slide 3: Understanding super() in Diamond Inheritance

The diamond inheritance pattern occurs when a class inherits from two classes that share a common ancestor. super() handles this gracefully by ensuring each parent class is initialized exactly once.

```python
class Base:
    def __init__(self):
        print("Base init")
        self.value = 1

class A(Base):
    def __init__(self):
        super().__init__()
        print("A init")
        self.value *= 2

class B(Base):
    def __init__(self):
        super().__init__()
        print("B init")
        self.value *= 3

class C(A, B):
    def __init__(self):
        super().__init__()
        print("C init")
        self.value *= 4

# Example usage
c = C()
print(f"Final value: {c.value}")
print(f"MRO: {[cls.__name__ for cls in C.__mro__]}")
```

Slide 4: Cooperative Multiple Inheritance

In cooperative multiple inheritance, each class in the inheritance chain properly calls super() to ensure all parent classes are initialized correctly, maintaining the integrity of the object construction process.

```python
class Serializable:
    def __init__(self, *args, **kwargs):
        self.serialized_data = {}
        super().__init__(*args, **kwargs)
    
    def serialize(self):
        return str(self.serialized_data)

class Loggable:
    def __init__(self, *args, **kwargs):
        self.log_messages = []
        super().__init__(*args, **kwargs)
    
    def log(self, message):
        self.log_messages.append(message)

class DataObject(Serializable, Loggable):
    def __init__(self, data):
        self.data = data
        super().__init__()
        self.log("DataObject initialized")
        self.serialized_data = {"data": self.data}

# Usage
obj = DataObject("test")
print(obj.serialize())
print(obj.log_messages)
```

Slide 5: Dynamic Method Resolution with super()

The super() function dynamically resolves method calls at runtime, allowing for flexible class hierarchies and method overriding without hardcoding parent class names, making the code more maintainable.

```python
class Shape:
    def area(self):
        return 0
    
    def describe(self):
        return f"Area: {self.area()}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)
    
    def describe(self):
        return f"Square: {super().describe()}"

# Example
square = Square(5)
print(square.describe())  # Output: Square: Area: 25
```

Slide 6: Implementing Mixins with super()

Mixins are a powerful way to add functionality to classes through multiple inheritance. Using super() ensures proper method chaining and allows mixins to work together harmoniously without conflicts.

```python
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps({
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        })

class ValidationMixin:
    def validate(self):
        for key, value in self.__dict__.items():
            if value is None:
                raise ValueError(f"{key} cannot be None")
        return True

class User(ValidationMixin, JSONMixin):
    def __init__(self, name, email):
        self.name = name
        self.email = email
        super().__init__()  # Initialize any parent class attributes
    
    def save(self):
        self.validate()
        return self.to_json()

# Usage
user = User("John Doe", "john@example.com")
print(user.save())
```

Slide 7: Real-world Example - Custom Exception Hierarchy

Implementing a robust exception hierarchy demonstrates the practical use of super() in creating specialized error handling mechanisms while maintaining proper error context and inheritance.

```python
class DatabaseError(Exception):
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code

class ConnectionError(DatabaseError):
    def __init__(self, host, port, message):
        super().__init__(f"Failed to connect to {host}:{port} - {message}")
        self.host = host
        self.port = port

class QueryError(DatabaseError):
    def __init__(self, query, message):
        super().__init__(f"Query failed: {message}", error_code="QE001")
        self.query = query

# Example usage
try:
    raise QueryError("SELECT * FROM users", "Table 'users' doesn't exist")
except QueryError as e:
    print(f"Error: {str(e)}")
    print(f"Error Code: {e.error_code}")
    print(f"Failed Query: {e.query}")
```

Slide 8: Abstract Base Classes with super()

Abstract base classes establish interfaces and partially implemented functionality. Using super() with abstract classes allows for proper method resolution while enforcing interface contracts.

```python
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    def __init__(self, data):
        self.data = data
        super().__init__()
    
    @abstractmethod
    def process(self):
        pass
    
    def validate(self):
        return len(self.data) > 0

class NumericProcessor(DataProcessor):
    def process(self):
        if not super().validate():
            raise ValueError("Empty data")
        return sum(self.data)

class TextProcessor(DataProcessor):
    def process(self):
        if not super().validate():
            raise ValueError("Empty data")
        return " ".join(self.data)

# Usage
num_proc = NumericProcessor([1, 2, 3, 4])
text_proc = TextProcessor(['Hello', 'World'])
print(num_proc.process())  # Output: 10
print(text_proc.process())  # Output: Hello World
```

Slide 9: Property Inheritance and super()

Property inheritance becomes more manageable with super(), allowing subclasses to extend or modify property behavior while maintaining the parent class's functionality.

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @property
    def fahrenheit(self):
        return (self.celsius * 9/5) + 32

class SmartTemperature(Temperature):
    @property
    def celsius(self):
        value = super().celsius
        # Add logging or validation
        print(f"Temperature accessed: {value}°C")
        return value
    
    @property
    def kelvin(self):
        return self.celsius + 273.15

# Usage
temp = SmartTemperature(25)
print(f"Fahrenheit: {temp.fahrenheit}°F")
print(f"Kelvin: {temp.kelvin}K")
```

Slide 10: Context Managers with super()

Context managers can be enhanced through inheritance while maintaining proper resource management. Using super() ensures that both parent and child cleanup operations are executed correctly in the right order.

```python
class BaseContext:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        print(f"Entering {self.name} context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Exiting {self.name} context")
        return False  # Don't suppress exceptions

class ResourceContext(BaseContext):
    def __enter__(self):
        self.resource = None
        super().__enter__()
        self.resource = "Allocated Resource"
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.resource:
            print(f"Cleaning up {self.resource}")
        return super().__exit__(exc_type, exc_val, exc_tb)

# Usage
with ResourceContext("Database") as ctx:
    print("Performing operations...")
```

Slide 11: Metaclasses and super()

Metaclasses provide powerful class creation control, and super() helps maintain proper initialization chains when extending metaclasses, ensuring all registration and setup processes are executed correctly.

```python
class RegistryMeta(type):
    _registry = {}
    
    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if not name.startswith('_'):
            cls._registry[name] = new_cls
        return new_cls

class Plugin(metaclass=RegistryMeta):
    def register(self):
        return f"Registered {self.__class__.__name__}"

class ImagePlugin(Plugin):
    def __init__(self):
        super().__init__()
        self.supported_formats = ['jpg', 'png']

class AudioPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self.supported_formats = ['mp3', 'wav']

# Check registry
print("Registered plugins:", RegistryMeta._registry)
```

Slide 12: Descriptors with super()

Descriptors can be inherited and extended while maintaining their core functionality using super(), allowing for enhanced attribute access control and validation in derived classes.

```python
class Validator:
    def __init__(self, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        self.validate(value)
        instance.__dict__[self.name] = value
    
    def validate(self, value):
        pass

class RangeValidator(Validator):
    def __init__(self, name, min_val, max_val):
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value):
        super().validate(value)
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"Value must be between {self.min_val} and {self.max_val}")

class Temperature:
    celsius = RangeValidator('celsius', -273.15, 1000)

# Usage
temp = Temperature()
temp.celsius = 25  # Works
try:
    temp.celsius = -300  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 13: Real-world Example - Custom Container Implementation

This example demonstrates how to create a custom container type that inherits from built-in containers while adding new functionality using super() for proper method delegation.

```python
class TrackedList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_count = 0
        self.modification_history = []
    
    def __getitem__(self, index):
        self.access_count += 1
        return super().__getitem__(index)
    
    def append(self, item):
        self.modification_history.append(('append', item))
        super().append(item)
    
    def extend(self, items):
        self.modification_history.append(('extend', list(items)))
        super().extend(items)
    
    def get_stats(self):
        return {
            'length': len(self),
            'accesses': self.access_count,
            'modifications': len(self.modification_history)
        }

# Usage example
tracked = TrackedList([1, 2, 3])
tracked.append(4)
tracked.extend([5, 6])
print(tracked[0])  # Access element
print(tracked.get_stats())
```

Slide 14: Additional Resources

*   Understanding Python's super() with Multiple Inheritance: [https://arxiv.org/abs/cs/0509025](https://arxiv.org/abs/cs/0509025)
*   Python Method Resolution Order (MRO): [https://www.python.org/download/releases/2.3/mro/](https://www.python.org/download/releases/2.3/mro/)
*   Design Patterns in Python: [https://python-patterns.guide/](https://python-patterns.guide/)
*   Python Metaclasses and Descriptors: [https://docs.python.org/3/howto/descriptor.html](https://docs.python.org/3/howto/descriptor.html)
*   Advanced Python OOP Topics: [https://realpython.com/python3-object-oriented-programming/](https://realpython.com/python3-object-oriented-programming/)

