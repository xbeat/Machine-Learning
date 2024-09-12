## Understanding Encapsulation in Python
Slide 1: What is Encapsulation?

Encapsulation is a fundamental concept in object-oriented programming that bundles data and the methods that operate on that data within a single unit or object. It provides data hiding and access control, allowing you to restrict direct access to an object's internal state.

```python
class BankAccount:
    def __init__(self):
        self.__balance = 0  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount

    def get_balance(self):
        return self.__balance

account = BankAccount()
account.deposit(100)
print(account.get_balance())  # Output: 100
# print(account.__balance)  # This would raise an AttributeError
```

Slide 2: Public vs. Private Attributes

In Python, we use naming conventions to indicate the accessibility of attributes. Public attributes are directly accessible, while private attributes are prefixed with double underscores.

```python
class Car:
    def __init__(self):
        self.color = "red"  # Public attribute
        self.__mileage = 0  # Private attribute

    def drive(self, distance):
        self.__mileage += distance

    def get_mileage(self):
        return self.__mileage

car = Car()
print(car.color)  # Output: red
car.drive(100)
print(car.get_mileage())  # Output: 100
# print(car.__mileage)  # This would raise an AttributeError
```

Slide 3: Name Mangling

Python uses name mangling for private attributes. It prefixes the attribute name with \_ClassName to make it harder to access from outside the class.

```python
class Example:
    def __init__(self):
        self.__private = "I'm private"

obj = Example()
print(dir(obj))  # Output: [..., '_Example__private', ...]
print(obj._Example__private)  # Output: I'm private
```

Slide 4: Property Decorators

Property decorators provide a way to use methods as attributes, allowing for controlled access to private attributes.

```python
class Temperature:
    def __init__(self):
        self.__celsius = 0

    @property
    def celsius(self):
        return self.__celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self.__celsius = value

    @property
    def fahrenheit(self):
        return (self.__celsius * 9/5) + 32

temp = Temperature()
temp.celsius = 25
print(temp.celsius)    # Output: 25
print(temp.fahrenheit) # Output: 77.0
```

Slide 5: Getters and Setters

Getters and setters are methods used to access and modify private attributes, providing an additional layer of control.

```python
class Circle:
    def __init__(self, radius):
        self.__radius = radius

    def get_radius(self):
        return self.__radius

    def set_radius(self, value):
        if value > 0:
            self.__radius = value
        else:
            raise ValueError("Radius must be positive")

    def area(self):
        return 3.14 * self.__radius ** 2

circle = Circle(5)
print(circle.get_radius())  # Output: 5
circle.set_radius(7)
print(circle.area())  # Output: 153.86
```

Slide 6: Encapsulation in Inheritance

Encapsulation also plays a role in inheritance. Private attributes are not directly accessible in child classes, but protected attributes (prefixed with a single underscore) are.

```python
class Animal:
    def __init__(self):
        self.__private_attr = "I'm private"
        self._protected_attr = "I'm protected"

class Dog(Animal):
    def access_attributes(self):
        # print(self.__private_attr)  # This would raise an AttributeError
        print(self._protected_attr)  # This works

dog = Dog()
dog.access_attributes()  # Output: I'm protected
```

Slide 7: Real-life Example: Smart Home System

A smart home system demonstrates encapsulation by hiding complex operations and providing a simple interface.

```python
class SmartHome:
    def __init__(self):
        self.__temperature = 20
        self.__lights_on = False

    def set_temperature(self, temp):
        if 15 <= temp <= 30:
            self.__temperature = temp
            self.__adjust_hvac()
        else:
            print("Temperature out of range")

    def __adjust_hvac(self):
        print(f"Adjusting HVAC to {self.__temperature}°C")

    def toggle_lights(self):
        self.__lights_on = not self.__lights_on
        print(f"Lights are {'on' if self.__lights_on else 'off'}")

home = SmartHome()
home.set_temperature(23)  # Output: Adjusting HVAC to 23°C
home.toggle_lights()  # Output: Lights are on
```

Slide 8: Real-life Example: Library Management System

A library management system showcases encapsulation by managing book information and lending processes internally.

```python
class Library:
    def __init__(self):
        self.__books = {}

    def add_book(self, title, author):
        if title not in self.__books:
            self.__books[title] = {"author": author, "available": True}
            print(f"Added: {title} by {author}")
        else:
            print("Book already exists")

    def borrow_book(self, title):
        if title in self.__books and self.__books[title]["available"]:
            self.__books[title]["available"] = False
            print(f"Borrowed: {title}")
        else:
            print("Book not available")

    def return_book(self, title):
        if title in self.__books:
            self.__books[title]["available"] = True
            print(f"Returned: {title}")
        else:
            print("This book doesn't belong to our library")

library = Library()
library.add_book("1984", "George Orwell")
library.borrow_book("1984")
library.return_book("1984")
```

Slide 9: Encapsulation and Data Validation

Encapsulation allows for data validation, ensuring that object attributes maintain valid states.

```python
class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age

    @property
    def age(self):
        return self.__age

    @age.setter
    def age(self, value):
        if 0 <= value <= 150:
            self.__age = value
        else:
            raise ValueError("Invalid age")

    def __str__(self):
        return f"{self.__name} is {self.__age} years old"

person = Person("Alice", 30)
print(person)  # Output: Alice is 30 years old
person.age = 35
print(person)  # Output: Alice is 35 years old
# person.age = 200  # This would raise a ValueError
```

Slide 10: Encapsulation in Context Managers

Context managers use encapsulation to manage resources, ensuring proper setup and cleanup.

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileManager('example.txt', 'w') as f:
    f.write('Hello, World!')

# File is automatically closed after the with block
```

Slide 11: Encapsulation and Method Chaining

Encapsulation can facilitate method chaining, allowing for more fluent and readable code.

```python
class StringBuilder:
    def __init__(self):
        self.__string = ""

    def append(self, text):
        self.__string += text
        return self

    def remove(self, text):
        self.__string = self.__string.replace(text, "")
        return self

    def upper(self):
        self.__string = self.__string.upper()
        return self

    def __str__(self):
        return self.__string

result = (StringBuilder()
          .append("Hello")
          .append(" ")
          .append("World")
          .remove("o")
          .upper())

print(result)  # Output: HELL WRLD
```

Slide 12: Encapsulation and Decorators

Decorators can be used to add encapsulation-like behavior to functions and methods.

```python
def validate_args(func):
    def wrapper(self, *args):
        if len(args) != 2 or not all(isinstance(arg, (int, float)) for arg in args):
            raise ValueError("Two numeric arguments are required")
        return func(self, *args)
    return wrapper

class Calculator:
    @validate_args
    def add(self, x, y):
        return x + y

calc = Calculator()
print(calc.add(5, 3))  # Output: 8
# print(calc.add("5", "3"))  # This would raise a ValueError
```

Slide 13: Encapsulation and Abstraction

Encapsulation supports abstraction by hiding implementation details and exposing only necessary interfaces.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.__side = side

    def area(self):
        return self.__side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.__radius = radius

    def area(self):
        return 3.14 * self.__radius ** 2

shapes = [Square(5), Circle(3)]
for shape in shapes:
    print(f"Area: {shape.area()}")

# Output:
# Area: 25
# Area: 28.26
```

Slide 14: Encapsulation and Design Patterns

Encapsulation plays a crucial role in various design patterns, such as the Singleton pattern, which ensures only one instance of a class exists.

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.__initialized = True
            self.__data = []

    def add_data(self, item):
        self.__data.append(item)

    def get_data(self):
        return self.__data.()

s1 = Singleton()
s2 = Singleton()
s1.add_data("Item 1")
print(s2.get_data())  # Output: ['Item 1']
print(s1 is s2)  # Output: True
```

Slide 15: Encapsulation in Asynchronous Programming

Encapsulation can be applied in asynchronous programming to manage shared state and ensure thread safety.

```python
import asyncio

class AsyncCounter:
    def __init__(self):
        self.__count = 0
        self.__lock = asyncio.Lock()

    async def increment(self):
        async with self.__lock:
            self.__count += 1
            await asyncio.sleep(0.1)  # Simulate some work
            return self.__count

    async def get_count(self):
        async with self.__lock:
            return self.__count

async def worker(counter, name):
    for _ in range(3):
        value = await counter.increment()
        print(f"{name}: {value}")

async def main():
    counter = AsyncCounter()
    await asyncio.gather(
        worker(counter, "Worker 1"),
        worker(counter, "Worker 2")
    )
    print(f"Final count: {await counter.get_count()}")

asyncio.run(main())

# Sample Output:
# Worker 1: 1
# Worker 2: 2
# Worker 1: 3
# Worker 2: 4
# Worker 1: 5
# Worker 2: 6
# Final count: 6
```

Slide 16: Additional Resources

For further exploration of encapsulation and related topics in Python:

1. "Data Encapsulation in Python" - arXiv:1803.04644 \[cs.PL\] [https://arxiv.org/abs/1803.04644](https://arxiv.org/abs/1803.04644)
2. "Design Patterns: Elements of Reusable Object-Oriented Software" by Erich Gamma et al. This book, while not Python-specific, provides valuable insights into design patterns that leverage encapsulation.
3. "Fluent Python" by Luciano Ramalho This book offers an in-depth look at Python's object model and how to effectively use encapsulation in Python.
4. "Python in Practice" by Mark Summerfield This book includes practical examples of design patterns and best practices in Python, including proper use of encapsulation.

