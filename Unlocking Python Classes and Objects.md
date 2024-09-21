## Unlocking Python Classes and Objects
Slide 1: Introduction to Classes and Objects

In Python, classes and objects are fundamental concepts of object-oriented programming (OOP). A class is a blueprint for creating objects, while an object is an instance of a class. This powerful paradigm allows us to structure our code in a more organized and reusable manner.

```python
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model

    def display_info(self):
        return f"{self.make} {self.model}"

# Creating an object
my_car = Car("Toyota", "Corolla")
print(my_car.display_info())  # Output: Toyota Corolla
```

Slide 2: Class Attributes and Methods

Class attributes are shared among all instances of a class, while instance attributes are unique to each object. Methods are functions defined within a class that can perform operations on the object's data.

```python
class Rectangle:
    shape_name = "Rectangle"  # Class attribute

    def __init__(self, width, height):
        self.width = width    # Instance attribute
        self.height = height  # Instance attribute

    def area(self):  # Method
        return self.width * self.height

rect = Rectangle(5, 3)
print(Rectangle.shape_name)  # Output: Rectangle
print(rect.area())           # Output: 15
```

Slide 3: Inheritance

Inheritance allows a class to inherit attributes and methods from another class. This promotes code reuse and establishes a hierarchy between classes.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.speak())  # Output: Buddy says Woof!
print(cat.speak())  # Output: Whiskers says Meow!
```

Slide 4: Encapsulation

Encapsulation is the bundling of data and the methods that operate on that data within a single unit (class). It restricts direct access to some of an object's components, which is a fundamental principle of OOP.

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private attribute

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
account.deposit(500)
print(account.get_balance())  # Output: 1500
# print(account.__balance)  # This would raise an AttributeError
```

Slide 5: Polymorphism

Polymorphism allows objects of different classes to be treated as objects of a common base class. It enables you to use a single interface with different underlying forms.

```python
class Shape:
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2

shapes = [Circle(5), Square(4)]
for shape in shapes:
    print(f"Area: {shape.area()}")

# Output:
# Area: 78.5
# Area: 16
```

Slide 6: Class Methods and Static Methods

Class methods and static methods are special types of methods in Python classes. Class methods can access and modify class state, while static methods don't access class or instance state.

```python
class MathOperations:
    pi = 3.14

    @classmethod
    def circle_area(cls, radius):
        return cls.pi * radius ** 2

    @staticmethod
    def add(a, b):
        return a + b

print(MathOperations.circle_area(5))  # Output: 78.5
print(MathOperations.add(3, 4))       # Output: 7
```

Slide 7: Property Decorators

Property decorators provide a way to customize access to instance attributes. They allow you to define methods that behave like attributes, enabling you to implement getters, setters, and deleters.

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value

    @property
    def fahrenheit(self):
        return (self.celsius * 9/5) + 32

temp = Temperature(25)
print(temp.celsius)     # Output: 25
print(temp.fahrenheit)  # Output: 77.0
temp.celsius = 30
print(temp.celsius)     # Output: 30
```

Slide 8: Magic Methods

Magic methods, also known as dunder methods, are special methods in Python that have double underscores before and after their names. They allow you to define how objects of your class behave in various situations.

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):
        return f"{self.title} by {self.author}"

    def __len__(self):
        return self.pages

    def __eq__(self, other):
        return self.title == other.title and self.author == other.author

book1 = Book("Python Crash Course", "Eric Matthes", 544)
book2 = Book("Python Crash Course", "Eric Matthes", 544)

print(str(book1))  # Output: Python Crash Course by Eric Matthes
print(len(book1))  # Output: 544
print(book1 == book2)  # Output: True
```

Slide 9: Abstract Base Classes

Abstract Base Classes (ABCs) provide a way to define interfaces in Python. They allow you to create base classes that can't be instantiated and force derived classes to implement certain methods.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # This would raise TypeError
rect = Rectangle(5, 3)
print(f"Area: {rect.area()}, Perimeter: {rect.perimeter()}")
# Output: Area: 15, Perimeter: 16
```

Slide 10: Composition

Composition is a design principle that suggests creating complex objects by combining simpler ones. It's an alternative to inheritance and often provides more flexibility.

```python
class Engine:
    def start(self):
        return "Engine started"

class Wheels:
    def rotate(self):
        return "Wheels rotating"

class Car:
    def __init__(self):
        self.engine = Engine()
        self.wheels = Wheels()

    def drive(self):
        return f"{self.engine.start()}, {self.wheels.rotate()}"

my_car = Car()
print(my_car.drive())
# Output: Engine started, Wheels rotating
```

Slide 11: Real-Life Example: Library Management System

Let's create a simple library management system to demonstrate the practical use of classes and objects.

```python
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False

    def __str__(self):
        return f"{self.title} by {self.author}"

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def borrow_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn and not book.is_borrowed:
                book.is_borrowed = True
                return f"Successfully borrowed: {book}"
        return "Book not available"

    def return_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn and book.is_borrowed:
                book.is_borrowed = False
                return f"Successfully returned: {book}"
        return "Invalid return"

library = Library()
library.add_book(Book("The Great Gatsby", "F. Scott Fitzgerald", "9780743273565"))
library.add_book(Book("To Kill a Mockingbird", "Harper Lee", "9780446310789"))

print(library.borrow_book("9780743273565"))
print(library.borrow_book("9780743273565"))
print(library.return_book("9780743273565"))

# Output:
# Successfully borrowed: The Great Gatsby by F. Scott Fitzgerald
# Book not available
# Successfully returned: The Great Gatsby by F. Scott Fitzgerald
```

Slide 12: Real-Life Example: Temperature Converter

Let's create a temperature converter class that demonstrates the use of properties and error handling.

```python
class TemperatureConverter:
    def __init__(self, celsius=0):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero is not possible")
        self._celsius = value

    @property
    def fahrenheit(self):
        return (self.celsius * 9/5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

    @property
    def kelvin(self):
        return self.celsius + 273.15

    @kelvin.setter
    def kelvin(self, value):
        self.celsius = value - 273.15

converter = TemperatureConverter()
converter.celsius = 25
print(f"Celsius: {converter.celsius}°C")
print(f"Fahrenheit: {converter.fahrenheit}°F")
print(f"Kelvin: {converter.kelvin}K")

converter.fahrenheit = 68
print(f"Celsius: {converter.celsius}°C")

# Output:
# Celsius: 25°C
# Fahrenheit: 77.0°F
# Kelvin: 298.15K
# Celsius: 20.0°C
```

Slide 13: Best Practices and Tips

1. Follow the PEP 8 style guide for Python code.
2. Use meaningful names for classes, methods, and attributes.
3. Keep your classes focused on a single responsibility (Single Responsibility Principle).
4. Use inheritance judiciously; favor composition over inheritance when appropriate.
5. Document your classes and methods using docstrings.
6. Use type hints to improve code readability and catch potential errors.
7. Write unit tests for your classes to ensure they behave as expected.
8. Consider using dataclasses for simple data containers (Python 3.7+).

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
print(f"Distance from origin: {p.distance_from_origin()}")
# Output: Distance from origin: 5.0
```

Slide 14: Advanced Topics in Python OOP

As you progress in your Python journey, consider exploring these advanced topics:

1. Metaclasses
2. Descriptors
3. Context Managers
4. Multiple Inheritance and Method Resolution Order (MRO)
5. Design Patterns in Python
6. Asynchronous Programming with async and await

Here's a quick example of a context manager:

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

with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')

# File is automatically closed after the with block
```

Slide 15: Additional Resources

To further enhance your understanding of Python classes and objects, consider exploring these resources:

1. "Python Object-Oriented Programming" by Steven F. Lott and Dusty Phillips (Book)
2. "Fluent Python" by Luciano Ramalho (Book)
3. "Object-Oriented Programming (OOP) in Python 3" course on Real Python (Online Tutorial)
4. "Python Beyond the Basics - Object-Oriented Programming" course on Udemy
5. Python official documentation on classes: [https://docs.python.org/3/tutorial/classes.html](https://docs.python.org/3/tutorial/classes.html)

For academic papers related to object-oriented programming and Python, you can explore these ArXiv references:

1. "A Principled Approach to Object-Oriented Programming in Python" by J. Vanderplas (2019) ArXiv URL: [https://arxiv.org/abs/1901.03757](https://arxiv.org/abs/1901.03757)
2. "Design Patterns in Python" by A. Mishchenko and G. Reznik (2021) ArXiv URL: [https://arxiv.org/abs/2106.09015](https://arxiv.org/abs/2106.09015)

Remember to verify the relevance and currency of these resources as you use them in your learning journey.


