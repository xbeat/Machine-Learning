## Simplifying Python Code with Attribute Functions
Slide 1: Introduction to Python's Attribute Functions

Python provides built-in functions for dynamic attribute manipulation: getattr(), setattr(), hasattr(), and delattr(). These functions offer flexibility and efficiency when working with object attributes, making code more concise and maintainable.

Slide 2: Source Code for Introduction to Python's Attribute Functions

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Creating an instance
person = Person("Alice", 30)

# Using attribute functions
print(getattr(person, "name"))  # Get attribute value
setattr(person, "job", "Developer")  # Set new attribute
print(hasattr(person, "age"))  # Check if attribute exists
delattr(person, "age")  # Delete an attribute

print(person.__dict__)  # View all attributes
```

Slide 3: Results for: Source Code for Introduction to Python's Attribute Functions

```
Alice
True
{'name': 'Alice', 'job': 'Developer'}
```

Slide 4: getattr() Function

The getattr() function retrieves the value of an object's attribute. It takes two required arguments: the object and the attribute name as a string. An optional third argument can be provided as a default value if the attribute doesn't exist.

Slide 5: Source Code for getattr() Function

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

book = Book("1984", "George Orwell")

# Using getattr()
title = getattr(book, "title")
genre = getattr(book, "genre", "Unknown")  # Using default value

print(f"Title: {title}")
print(f"Genre: {genre}")
```

Slide 6: Results for: Source Code for getattr() Function

```
Title: 1984
Genre: Unknown
```

Slide 7: setattr() Function

setattr() allows dynamic assignment of attribute values. It takes three arguments: the object, the attribute name as a string, and the value to assign. This function is particularly useful when working with attributes whose names are determined at runtime.

Slide 8: Source Code for setattr() Function

```python
class Car:
    pass

car = Car()

# Using setattr()
setattr(car, "brand", "Toyota")
setattr(car, "model", "Corolla")
setattr(car, "year", 2023)

print(f"Car: {car.brand} {car.model} ({car.year})")

# Dynamic attribute assignment
attributes = ["color", "fuel_type", "mileage"]
values = ["Red", "Gasoline", 15000]

for attr, value in zip(attributes, values):
    setattr(car, attr, value)

print(car.__dict__)
```

Slide 9: Results for: Source Code for setattr() Function

```
Car: Toyota Corolla (2023)
{'brand': 'Toyota', 'model': 'Corolla', 'year': 2023, 'color': 'Red', 'fuel_type': 'Gasoline', 'mileage': 15000}
```

Slide 10: hasattr() Function

hasattr() checks if an object has a specific attribute. It returns True if the attribute exists, and False otherwise. This function is useful for conditional logic based on attribute presence.

Slide 11: Source Code for hasattr() Function

```python
class Smartphone:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

phone = Smartphone("Apple", "iPhone 13")

# Using hasattr()
if hasattr(phone, "brand"):
    print(f"Brand: {phone.brand}")

if hasattr(phone, "os"):
    print(f"OS: {phone.os}")
else:
    setattr(phone, "os", "iOS")
    print(f"OS set to: {phone.os}")

# Checking multiple attributes
required_attrs = ["brand", "model", "os", "price"]
missing_attrs = [attr for attr in required_attrs if not hasattr(phone, attr)]

print(f"Missing attributes: {missing_attrs}")
```

Slide 12: Results for: Source Code for hasattr() Function

```
Brand: Apple
OS set to: iOS
Missing attributes: ['price']
```

Slide 13: delattr() Function

delattr() removes an attribute from an object. It takes two arguments: the object and the attribute name as a string. This function is useful for cleaning up or modifying object structures dynamically.

Slide 14: Source Code for delattr() Function

```python
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

student = Student("Bob", 16, "10th")

print("Before deletion:", student.__dict__)

# Using delattr()
delattr(student, "age")

print("After deletion:", student.__dict__)

# Attempting to delete a non-existent attribute
try:
    delattr(student, "school")
except AttributeError as e:
    print(f"Error: {e}")

# Conditional attribute deletion
if hasattr(student, "grade"):
    delattr(student, "grade")

print("Final state:", student.__dict__)
```

Slide 15: Results for: Source Code for delattr() Function

```
Before deletion: {'name': 'Bob', 'age': 16, 'grade': '10th'}
After deletion: {'name': 'Bob', 'grade': '10th'}
Error: 'Student' object has no attribute 'school'
Final state: {'name': 'Bob'}
```

Slide 16: Real-Life Example: Dynamic Configuration

In this example, we'll use attribute functions to create a flexible configuration system for a web application.

Slide 17: Source Code for Real-Life Example: Dynamic Configuration

```python
class AppConfig:
    def __init__(self):
        self.default_settings = {
            "debug_mode": False,
            "max_users": 100,
            "theme": "default"
        }

    def __getattr__(self, name):
        return self.default_settings.get(name, None)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self, setting):
        if hasattr(self, setting):
            delattr(self, setting)

# Usage
config = AppConfig()

print("Initial debug_mode:", config.debug_mode)

config.update_config(debug_mode=True, theme="dark")
print("Updated config:", config.__dict__)

config.reset("theme")
print("After reset:", config.__dict__)

print("max_users:", getattr(config, "max_users"))
```

Slide 18: Results for: Source Code for Real-Life Example: Dynamic Configuration

```
Initial debug_mode: False
Updated config: {'debug_mode': True, 'theme': 'dark'}
After reset: {'debug_mode': True}
max_users: 100
```

Slide 19: Real-Life Example: Dynamic Form Handling

This example demonstrates how attribute functions can be used to handle dynamic form data in a web application.

Slide 20: Source Code for Real-Life Example: Dynamic Form Handling

```python
class DynamicForm:
    def __init__(self):
        self.fields = {}

    def add_field(self, name, value=None):
        setattr(self, name, value)
        self.fields[name] = value

    def update_field(self, name, value):
        if hasattr(self, name):
            setattr(self, name, value)
            self.fields[name] = value
        else:
            raise AttributeError(f"Field '{name}' does not exist")

    def remove_field(self, name):
        if hasattr(self, name):
            delattr(self, name)
            del self.fields[name]
        else:
            raise AttributeError(f"Field '{name}' does not exist")

    def get_field(self, name):
        return getattr(self, name, None)

    def __str__(self):
        return str(self.fields)

# Usage
form = DynamicForm()

form.add_field("username", "john_doe")
form.add_field("email", "john@example.com")
print("Initial form:", form)

form.update_field("email", "john.doe@example.com")
print("Updated form:", form)

print("Username:", form.get_field("username"))

form.remove_field("email")
print("Form after removal:", form)
```

Slide 21: Results for: Source Code for Real-Life Example: Dynamic Form Handling

```
Initial form: {'username': 'john_doe', 'email': 'john@example.com'}
Updated form: {'username': 'john_doe', 'email': 'john.doe@example.com'}
Username: john_doe
Form after removal: {'username': 'john_doe'}
```

Slide 22: Additional Resources

For more information on Python's attribute functions and advanced object-oriented programming concepts, consider exploring these resources:

1.  Python Documentation: [https://docs.python.org/3/library/functions.html](https://docs.python.org/3/library/functions.html)
2.  "Fluent Python" by Luciano Ramalho
3.  "Python Cookbook" by David Beazley and Brian K. Jones

These resources provide in-depth explanations and advanced use cases for attribute manipulation in Python.

