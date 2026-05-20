## Python Advanced Concepts! Decorators, Generators, and Context Managers

Slide 1: Function Decorators

Function decorators are a powerful feature in Python that allow you to modify or enhance the behavior of functions without changing their source code. They are essentially functions that take another function as an argument and return a new function with added functionality.

```python
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

@uppercase_decorator
def greet():
    return "hello, world!"

print(greet())  # Output: HELLO, WORLD!
```

Slide 2: Class Decorators

Class decorators are similar to function decorators but are applied to classes instead. They can be used to modify or enhance the behavior of a class without altering its source code.

```python
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        self.connection = "Connected to database"

db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # Output: True
```

Slide 3: Generators

Generators are functions that allow you to declare a function that behaves like an iterator. They use the yield keyword to produce a series of values over time, rather than computing them all at once and returning them in a list.

```python
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for num in fibonacci(10):
    print(num, end=" ")
# Output: 0 1 1 2 3 5 8 13 21 34
```

Slide 4: Custom Context Managers

Context managers are objects that define the methods **enter**() and **exit**(), allowing you to allocate and release resources precisely using the 'with' statement. They are commonly used for file handling, database connections, and other resource management tasks.

```python
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print(f"Execution time: {self.end - self.start:.2f} seconds")

with Timer():
    # Simulate some work
    time.sleep(2)
# Output: Execution time: 2.00 seconds
```

Slide 5: Custom Managers with contextlib

The contextlib module provides utilities for working with context managers. It includes the @contextmanager decorator, which allows you to create context managers using generator syntax, simplifying the process of creating custom context managers.

```python

@contextmanager
def tempdir():
    import tempfile
    import shutil
    import os

    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)

with tempdir() as temp:
    print(f"Created temporary directory: {temp}")
    # Use the temporary directory
# Directory is automatically removed after the 'with' block
```

Slide 6: Callable Objects

In Python, any object that implements the **call**() method is considered callable. This allows instances of a class to be called like functions, providing a powerful way to create objects that act like functions but maintain state.

```python
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))  # Output: 10
print(triple(5))  # Output: 15
```

Slide 7: Property Decorators

Property decorators provide a way to define methods that can be accessed like attributes. They allow you to add getter, setter, and deleter functionality to class attributes, enabling you to control access and modify behavior when getting or setting values.

```python
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.fahrenheit)  # Output: 77.0
temp.fahrenheit = 100
print(temp._celsius)  # Output: 37.77777777777778
```

Slide 8: Abstract Base Classes

Abstract Base Classes (ABCs) provide a way to define interfaces in Python. They allow you to create base classes that can't be instantiated and enforce the implementation of certain methods in derived classes.

```python

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

rect = Rectangle(5, 3)
print(rect.area())  # Output: 15
```

Slide 9: Metaclasses

Metaclasses are classes for classes. They define how classes are created and can be used to modify class creation behavior. Metaclasses allow you to implement patterns like singleton, interface checking, or automatic property creation.

```python
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # Output: True
```

Slide 10: Enforcing Naming Conventions

Metaclasses can be used to enforce naming conventions in classes. This example demonstrates how to ensure all method names in a class are in snake\_case.

```python

class SnakeCaseEnforcer(type):
    def __new__(cls, name, bases, dct):
        for attr_name, attr_value in dct.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                if not re.match(r'^[a-z_]+$', attr_name):
                    raise ValueError(f"Method '{attr_name}' must be in snake_case")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=SnakeCaseEnforcer):
    def valid_method(self):
        pass

    def invalidMethod(self):  # This will raise a ValueError
        pass
```

Slide 11: Descriptors

Descriptors are objects that define how attribute access is handled in classes. They implement methods like **get**(), **set**(), and **delete**() to customize attribute behavior.

```python
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        return obj.__dict__[self.name]

    def __set__(self, obj, value):
        if not (self.min_value <= value <= self.max_value):
            raise ValueError(f"{self.name} must be between {self.min_value} and {self.max_value}")
        obj.__dict__[self.name] = value

class Person:
    age = ValidatedAttribute(0, 120)

person = Person()
person.age = 30  # Valid
try:
    person.age = 150  # Raises ValueError
except ValueError as e:
    print(e)  # Output: age must be between 0 and 120
```

Slide 12: Data Classes

Data Classes, introduced in Python 3.7, provide a concise way to create classes that are primarily used to store data. They automatically generate methods like **init**(), **repr**(), and **eq**() based on the class attributes.

```python

@dataclass
class Point:
    x: float
    y: float

    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p1 = Point(3, 4)
p2 = Point(3, 4)

print(p1)  # Output: Point(x=3, y=4)
print(p1 == p2)  # Output: True
print(p1.distance_from_origin())  # Output: 5.0
```

Slide 13: Function Annotations

Function annotations provide a way to attach metadata to function parameters and return values. While they don't affect the function's behavior directly, they can be used for documentation, type checking, or other custom processing.

```python
    """
    Calculate Body Mass Index (BMI)
    
    :param weight: Weight in kilograms
    :param height: Height in meters
    :return: BMI value
    """
    return weight / (height ** 2)

import inspect

print(calculate_bmi.__annotations__)
# Output: {'weight': <class 'float'>, 'height': <class 'float'>, 'return': <class 'float'>}

print(inspect.signature(calculate_bmi))
# Output: (weight: float, height: float) -> float
```

Slide 14: Using functools for Memoization

Memoization is an optimization technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again. The functools.lru\_cache decorator provides an easy way to add memoization to functions.

```python

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Output: 354224848179261915075 (calculated quickly)
print(fibonacci.cache_info())  # Output: CacheInfo(hits=98, misses=101, maxsize=None, currsize=101)
```

Slide 15: **slots**

The **slots** attribute allows you to explicitly declare data members in a class, optimizing memory usage and improving attribute access speed. It creates a static structure for instances, preventing the addition of new attributes not specified in **slots**.

```python
    def __init__(self, name, age):
        self.name = name
        self.age = age

class SlottedPerson:
    __slots__ = ['name', 'age']
    def __init__(self, name, age):
        self.name = name
        self.age = age

regular = RegularPerson("Alice", 30)
slotted = SlottedPerson("Bob", 25)

regular.new_attr = "This works"
try:
    slotted.new_attr = "This raises an AttributeError"
except AttributeError as e:
    print(e)  # Output: 'SlottedPerson' object has no attribute 'new_attr'

import sys
print(sys.getsizeof(regular))  # Output: 48 (or similar, may vary)
print(sys.getsizeof(slotted))  # Output: 40 (or similar, may vary)
```

Slide 16: Additional Resources

For more in-depth information on these Python topics, consider exploring the following resources:

1. "Python Data Model" by Luciano Ramalho ([https://arxiv.org/abs/1406.4671](https://arxiv.org/abs/1406.4671))
2. "Metaclasses in Python" by Michele Simionato ([https://arxiv.org/abs/1101.4349](https://arxiv.org/abs/1101.4349))
3. Official Python Documentation ([https://docs.python.org/](https://docs.python.org/))

These resources provide comprehensive explanations and advanced usage examples of the concepts covered in this presentation.


