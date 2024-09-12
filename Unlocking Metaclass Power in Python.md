## Unlocking Metaclass Power in Python
Slide 1: Understanding Metaclasses

Metaclasses are a powerful feature in Python that allow you to customize class creation. They provide a way to intercept and modify the class creation process, enabling you to add or modify attributes, methods, or behaviors of classes automatically.

```python
# Define a simple metaclass
class MyMetaclass(type):
    def __new__(cls, name, bases, attrs):
        # Add a new method to the class
        attrs['greet'] = lambda self: f"Hello from {name}!"
        return super().__new__(cls, name, bases, attrs)

# Use the metaclass
class MyClass(metaclass=MyMetaclass):
    pass

# Create an instance and call the added method
obj = MyClass()
print(obj.greet())  # Output: Hello from MyClass!
```

Slide 2: The Metaclass Hierarchy

In Python, everything is an object, including classes. The type of a class is called a metaclass. By default, Python uses the `type` metaclass to create classes.

```python
# Demonstrate the metaclass hierarchy
class RegularClass:
    pass

print(type(RegularClass))  # Output: <class 'type'>
print(type(type))  # Output: <class 'type'>

# Create a class using type
DynamicClass = type('DynamicClass', (), {'x': 42})
print(type(DynamicClass))  # Output: <class 'type'>
print(DynamicClass.x)  # Output: 42
```

Slide 3: Creating Custom Metaclasses

Custom metaclasses are created by inheriting from `type`. They can override methods like `__new__` and `__init__` to customize class creation and initialization.

```python
class LoggingMetaclass(type):
    def __new__(cls, name, bases, attrs):
        print(f"Creating class: {name}")
        return super().__new__(cls, name, bases, attrs)
    
    def __init__(cls, name, bases, attrs):
        print(f"Initializing class: {name}")
        super().__init__(name, bases, attrs)

class MyClass(metaclass=LoggingMetaclass):
    pass

# Output:
# Creating class: MyClass
# Initializing class: MyClass
```

Slide 4: Modifying Class Attributes

Metaclasses can modify class attributes before the class is created. This allows for automatic attribute addition or modification.

```python
class UpperAttributesMetaclass(type):
    def __new__(cls, name, bases, attrs):
        uppercase_attrs = {
            key.upper(): value
            for key, value in attrs.items()
            if not key.startswith('__')
        }
        return super().__new__(cls, name, bases, uppercase_attrs)

class LowercaseClass(metaclass=UpperAttributesMetaclass):
    x = 1
    y = 2

print(LowercaseClass.X)  # Output: 1
print(LowercaseClass.Y)  # Output: 2
print(hasattr(LowercaseClass, 'x'))  # Output: False
```

Slide 5: Metaclasses for Validation

Metaclasses can be used to validate class definitions, ensuring that classes meet certain criteria before they are created.

```python
class ValidateFieldsMetaclass(type):
    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            if key.startswith('field_') and not isinstance(value, (int, float, str)):
                raise TypeError(f"{key} must be int, float, or str")
        return super().__new__(cls, name, bases, attrs)

class ValidatedClass(metaclass=ValidateFieldsMetaclass):
    field_a = 1
    field_b = "valid"
    # field_c = [1, 2, 3]  # This would raise a TypeError

print(ValidatedClass.field_a)  # Output: 1
print(ValidatedClass.field_b)  # Output: valid
```

Slide 6: Singleton Pattern with Metaclasses

Metaclasses can implement design patterns, such as the Singleton pattern, which ensures only one instance of a class exists.

```python
class SingletonMetaclass(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMetaclass):
    def __init__(self):
        self.value = None

# Create multiple instances
s1 = Singleton()
s2 = Singleton()

print(s1 is s2)  # Output: True

s1.value = 42
print(s2.value)  # Output: 42
```

Slide 7: Abstract Base Classes with Metaclasses

Metaclasses can be used to create abstract base classes, which define interfaces that derived classes must implement.

```python
class ABCMetaclass(type):
    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            if getattr(value, '__isabstractmethod__', False):
                attrs[key] = abstractmethod(value)
        return super().__new__(cls, name, bases, attrs)

class AbstractClass(metaclass=ABCMetaclass):
    def abstract_method(self):
        raise NotImplementedError

class ConcreteClass(AbstractClass):
    def abstract_method(self):
        return "Implemented!"

# This works
obj = ConcreteClass()
print(obj.abstract_method())  # Output: Implemented!

# This raises TypeError
# AbstractClass()
```

Slide 8: Metaclasses for Automatic Property Creation

Metaclasses can automate the creation of properties, reducing boilerplate code in classes.

```python
class AutoPropertyMetaclass(type):
    def __new__(cls, name, bases, attrs):
        for key, value in attrs.items():
            if key.startswith('_') and not key.startswith('__'):
                attrs[key[1:]] = property(lambda self, k=key: getattr(self, k))
        return super().__new__(cls, name, bases, attrs)

class Person(metaclass=AutoPropertyMetaclass):
    def __init__(self, name, age):
        self._name = name
        self._age = age

p = Person("Alice", 30)
print(p.name)  # Output: Alice
print(p.age)   # Output: 30
```

Slide 9: Metaclasses for Automatic Method Decoration

Metaclasses can apply decorators to methods automatically, reducing repetitive code and enforcing consistent behavior.

```python
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class LoggingMetaclass(type):
    def __new__(cls, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if callable(attr_value):
                attrs[attr_name] = log_calls(attr_value)
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=LoggingMetaclass):
    def method1(self):
        return "Hello from method1"
    
    def method2(self):
        return "Hello from method2"

obj = MyClass()
obj.method1()  # Output: Calling method1
obj.method2()  # Output: Calling method2
```

Slide 10: Real-Life Example: ORM (Object-Relational Mapping)

Metaclasses are often used in ORMs to define database models. Here's a simplified example inspired by SQLAlchemy:

```python
class ModelMetaclass(type):
    def __new__(cls, name, bases, attrs):
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        
        print(f"Creating model: {name}")
        fields = {}
        for key, value in attrs.items():
            if isinstance(value, Field):
                print(f"Found field: {key}")
                fields[key] = value
        
        attrs['_fields'] = fields
        return super().__new__(cls, name, bases, attrs)

class Field:
    def __init__(self, field_type):
        self.field_type = field_type

class Model(metaclass=ModelMetaclass):
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)

class User(Model):
    name = Field(str)
    age = Field(int)

user = User(name="Alice", age=30)
print(user.name)  # Output: Alice
print(user.age)   # Output: 30
```

Slide 11: Real-Life Example: Plugin System

Metaclasses can be used to create a plugin system, automatically registering new plugins as they are defined:

```python
class PluginMetaclass(type):
    plugins = {}
    
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        if bases:  # Only register if it's a subclass
            cls.plugins[name] = new_class
        return new_class

class Plugin(metaclass=PluginMetaclass):
    def run(self):
        raise NotImplementedError

class ImagePlugin(Plugin):
    def run(self):
        return "Processing image..."

class AudioPlugin(Plugin):
    def run(self):
        return "Processing audio..."

# Using the plugins
for name, plugin in PluginMetaclass.plugins.items():
    print(f"Running {name}: {plugin().run()}")

# Output:
# Running ImagePlugin: Processing image...
# Running AudioPlugin: Processing audio...
```

Slide 12: Limitations and Considerations

While metaclasses are powerful, they should be used judiciously:

1. Complexity: Metaclasses can make code harder to understand and debug.
2. Performance: Extensive use of metaclasses may impact performance.
3. Compatibility: Metaclasses can complicate inheritance and interoperability.

```python
# Example of a potential issue with multiple metaclasses
class Meta1(type): pass
class Meta2(type): pass

class A(metaclass=Meta1): pass
class B(metaclass=Meta2): pass

# This will raise a TypeError due to metaclass conflict
# class C(A, B): pass

# Possible solution: create a combined metaclass
class CombinedMeta(Meta1, Meta2): pass

class C(A, B, metaclass=CombinedMeta): pass
```

Slide 13: Best Practices for Using Metaclasses

1. Use metaclasses sparingly and only when simpler solutions are insufficient.
2. Document your metaclasses thoroughly to explain their behavior and purpose.
3. Consider alternative approaches like class decorators or descriptors first.
4. Be aware of the metaclass resolution order in complex inheritance hierarchies.

```python
# Example of a class decorator as an alternative to a simple metaclass
def add_greeting(cls):
    cls.greet = lambda self: f"Hello from {cls.__name__}!"
    return cls

@add_greeting
class MyClass:
    pass

obj = MyClass()
print(obj.greet())  # Output: Hello from MyClass!
```

Slide 14: Additional Resources

For further exploration of metaclasses in Python, consider these resources:

1. "A Primer on Python Metaclasses" by Jake VanderPlas ArXiv: [https://arxiv.org/abs/1209.2803](https://arxiv.org/abs/1209.2803)
2. "Metaclasses in Python 3" by Michele Simionato ArXiv: [https://arxiv.org/abs/1101.4576](https://arxiv.org/abs/1101.4576)

These papers provide in-depth discussions on the theory and practical applications of metaclasses in Python.

