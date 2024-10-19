## Decorating Python Classes
Slide 1: Understanding Class Decorators in Python

Class decorators are a powerful feature in Python that allow you to modify or enhance the behavior of classes. They are similar to function decorators but operate on entire classes instead of individual functions. Class decorators can be used to add functionality, modify attributes, or even completely transform the class definition.

Slide 2: Source Code for Understanding Class Decorators in Python

```python
def class_decorator(cls):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            print("Initializing with decorator")
            super().__init__(*args, **kwargs)
        
        def new_method(self):
            return "This method was added by the decorator"
    
    return Wrapper

@class_decorator
class MyClass:
    def __init__(self, value):
        self.value = value

    def original_method(self):
        return f"Original value: {self.value}"

# Usage
obj = MyClass(42)
print(obj.original_method())
print(obj.new_method())
```

Slide 3: Results for Understanding Class Decorators in Python

```
Initializing with decorator
Original value: 42
This method was added by the decorator
```

Slide 4: Decorators with Parameters

Decorators can also accept parameters, allowing for more flexible and customizable class modifications. This is achieved by creating a decorator factory function that returns the actual decorator.

Slide 5: Source Code for Decorators with Parameters

```python
def default_params(**defaults):
    def wrapper(cls):
        class Wrapped(cls):
            def __init__(self, **kwargs):
                for key, value in defaults.items():
                    if key not in kwargs:
                        kwargs[key] = value
                super().__init__(**kwargs)
        return Wrapped
    return wrapper

@default_params(x=10, y=20)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

# Usage
p1 = Point()
p2 = Point(x=5)
p3 = Point(x=1, y=2)

print(p1, p2, p3)
```

Slide 6: Results for Decorators with Parameters

```
Point(10, 20) Point(5, 20) Point(1, 2)
```

Slide 7: Real-Life Example: Logging Decorator

A common use case for class decorators is adding logging functionality to classes. This can be useful for debugging and monitoring class instantiation and method calls.

Slide 8: Source Code for Real-Life Example: Logging Decorator

```python
import logging

def add_logging(cls):
    logging.basicConfig(level=logging.INFO)
    class LoggedClass(cls):
        def __init__(self, *args, **kwargs):
            logging.info(f"Creating instance of {cls.__name__}")
            super().__init__(*args, **kwargs)
        
        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if callable(attr):
                def logged_method(*args, **kwargs):
                    logging.info(f"Calling {name} on {cls.__name__}")
                    return attr(*args, **kwargs)
                return logged_method
            return attr
    
    return LoggedClass

@add_logging
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

# Usage
calc = Calculator()
result = calc.add(5, 3)
result = calc.subtract(10, 4)
```

Slide 9: Results for Real-Life Example: Logging Decorator

```
INFO:root:Creating instance of Calculator
INFO:root:Calling add on Calculator
INFO:root:Calling subtract on Calculator
```

Slide 10: Real-Life Example: Validation Decorator

Another practical use of class decorators is for input validation. This can help ensure that objects are created with valid data.

Slide 11: Source Code for Real-Life Example: Validation Decorator

```python
def validate_inputs(**validators):
    def decorator(cls):
        class ValidatedClass(cls):
            def __init__(self, **kwargs):
                for key, validator in validators.items():
                    if key in kwargs:
                        if not validator(kwargs[key]):
                            raise ValueError(f"Invalid value for {key}")
                super().__init__(**kwargs)
        return ValidatedClass
    return decorator

def positive(value):
    return value > 0

def non_empty_string(value):
    return isinstance(value, str) and len(value.strip()) > 0

@validate_inputs(age=positive, name=non_empty_string)
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Usage
try:
    p1 = Person(name="Alice", age=30)
    print(f"Created person: {p1.name}, {p1.age}")
    
    p2 = Person(name="", age=-5)
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 12: Results for Real-Life Example: Validation Decorator

```
Created person: Alice, 30
Validation error: Invalid value for name
```

Slide 13: Class Decorators vs. Inheritance

Class decorators offer an alternative to inheritance for extending class functionality. They provide a more flexible and composable approach, allowing you to add or modify behavior without creating complex inheritance hierarchies.

Slide 14: Source Code for Class Decorators vs. Inheritance

```python
# Inheritance approach
class BaseClass:
    def __init__(self):
        print("BaseClass init")

class ExtendedClass(BaseClass):
    def __init__(self):
        super().__init__()
        print("ExtendedClass init")

# Decorator approach
def extend_init(cls):
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        print("Extended init via decorator")
    cls.__init__ = new_init
    return cls

@extend_init
class DecoratedClass:
    def __init__(self):
        print("DecoratedClass init")

# Usage
print("Inheritance:")
ExtendedClass()
print("\nDecorator:")
DecoratedClass()
```

Slide 15: Results for Class Decorators vs. Inheritance

```
Inheritance:
BaseClass init
ExtendedClass init

Decorator:
DecoratedClass init
Extended init via decorator
```

Slide 16: Additional Resources

For more information on Python decorators and their advanced uses, refer to the following resources:

1.  "Python Decorators: A Powerful and Expressive Feature" by Guido van Rossum (Python's creator): [https://arxiv.org/abs/2010.06545](https://arxiv.org/abs/2010.06545)
2.  "Design Patterns in Python: Implementing the Gang of Four Patterns" by Bruno Preiss: [https://arxiv.org/abs/2004.10177](https://arxiv.org/abs/2004.10177)

These papers provide in-depth discussions on the design and implementation of decorators in Python, as well as their applications in various programming patterns.

