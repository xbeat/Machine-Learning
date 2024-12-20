## SOLID Design Patterns in Python

Slide 1: Single Responsibility Principle (SRP)

The Single Responsibility Principle states that a class should have only one reason to change, meaning it should have a single, well-defined responsibility or job. This principle promotes code organization, maintainability, and testability.

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def display_book_info(self):
        print(f"{self.title}, Author: {self.author}")

class BookPersistence:
    def save_book(self, book, file_path):
        # Code to save book data to a file

    def load_book(self, file_path):
        # Code to load book data from a file
```

In this example, the `Book` class is responsible for holding book data, while the `BookPersistence` class handles the persistence of book data. Each class has a single responsibility, promoting code organization and maintainability.

Slide 2: Open/Closed Principle (OCP)

The Open/Closed Principle states that a class should be open for extension but closed for modification. In other words, you should be able to extend the behavior of a class without modifying its source code.

```python
from abc import ABC, abstractmethod

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

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

# Extension: Adding a new shape without modifying existing code
class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height
```

In this example, the `Shape` class is an abstract base class with an abstract `area` method. The `Rectangle` and `Circle` classes inherit from `Shape` and implement the `area` method. To add a new shape (e.g., `Triangle`), we can simply create a new class that inherits from `Shape` and implements the `area` method, without modifying the existing code.

Slide 3: Liskov Substitution Principle (LSP)

The Liskov Substitution Principle states that objects of a superclass should be replaceable with objects of its subclasses without affecting the correctness of the program.

```python
class Vehicle:
    def start_engine(self):
        pass

class Car(Vehicle):
    def start_engine(self):
        print("Starting car engine.")

class ElectricCar(Car):
    def start_engine(self):
        print("Electric cars don't have engines.")

# Violates LSP
def start_vehicles(vehicles):
    for vehicle in vehicles:
        vehicle.start_engine()

car = Car()
electric_car = ElectricCar()
vehicles = [car, electric_car]
start_vehicles(vehicles)  # Raises an error for ElectricCar
```

In this example, the `ElectricCar` class violates the Liskov Substitution Principle because it overrides the `start_engine` method in an unexpected way, leading to a runtime error when used in place of a `Car` object. To adhere to LSP, we can refactor the code to use more appropriate method names or introduce additional abstractions.

Slide 4: Interface Segregation Principle (ISP)

The Interface Segregation Principle states that a client should not be forced to depend on methods it does not use. Instead, interfaces should be split into smaller, more specific ones.

```python
from abc import ABC, abstractmethod

class Printer(ABC):
    @abstractmethod
    def print_document(self):
        pass

class Scanner(ABC):
    @abstractmethod
    def scan_document(self):
        pass

class MultiFunctionPrinter(Printer, Scanner):
    def print_document(self):
        # Code to print a document

    def scan_document(self):
        # Code to scan a document

class SimplePrinter(Printer):
    def print_document(self):
        # Code to print a document
```

In this example, the `Printer` and `Scanner` interfaces are segregated, allowing clients to depend only on the functionality they need. The `MultiFunctionPrinter` class implements both interfaces, while the `SimplePrinter` class only implements the `Printer` interface.

Slide 5: Dependency Inversion Principle (DIP)

The Dependency Inversion Principle states that high-level modules should not depend on low-level modules. Both should depend on abstractions. Abstractions should not depend on details. Details should depend on abstractions.

```python
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, message):
        pass

class ConsoleLogger(Logger):
    def log(self, message):
        print(message)

class FileLogger(Logger):
    def __init__(self, file_path):
        self.file_path = file_path

    def log(self, message):
        with open(self.file_path, 'a') as file:
            file.write(message + '\n')

class PaymentProcessor:
    def __init__(self, logger: Logger):
        self.logger = logger

    def process_payment(self, amount):
        self.logger.log(f"Processing payment of ${amount}")
        # Payment processing code

console_logger = ConsoleLogger()
payment_processor = PaymentProcessor(console_logger)
payment_processor.process_payment(100)
```

In this example, the `PaymentProcessor` class depends on the `Logger` abstraction (interface), allowing flexibility to switch between different logging implementations without modifying the `PaymentProcessor` class. The `ConsoleLogger` and `FileLogger` classes implement the `Logger` interface, providing concrete logging implementations.

Slide 6: SRP Example: Separation of Concerns

The Single Responsibility Principle promotes the separation of concerns, where each class or module has a single, well-defined responsibility.

```python
class BookStore:
    def __init__(self):
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def remove_book(self, book):
        self.books.remove(book)

    def display_books(self):
        for book in self.books:
            print(book)

class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return f"{self.title} by {self.author}"

# Usage
bookstore = BookStore()
book1 = Book("The Great Gatsby", "F. Scott Fitzgerald")
book2 = Book("Pride and Prejudice", "Jane Austen")

bookstore.add_book(book1)
bookstore.add_book(book2)
bookstore.display_books()
```

In this example, the `BookStore` class is responsible for managing the collection of books, while the `Book` class is responsible for representing a single book. Each class has a single responsibility, promoting code organization and maintainability.

Slide 7: OCP Example: Extension without Modification

The Open/Closed Principle allows extending the functionality of a class without modifying its source code, promoting code reusability and maintainability.

```python
from abc import ABC, abstractmethod

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

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

class AreaCalculator:
    def calculate_area(self, shapes):
        total_area = 0
        for shape in shapes:
            total_area += shape.area()
        return total_area

# Usage
rectangle = Rectangle(5, 3)
circle = Circle(2)
shapes = [rectangle, circle]

calculator = AreaCalculator()
total_area = calculator.calculate_area(shapes)
print(f"Total area: {total_area}")
```

In this example, the `Shape` class is an abstract base class with an abstract `area` method. The `Rectangle` and `Circle` classes inherit from `Shape` and implement the `area` method. The `AreaCalculator` class calculates the total area of a list of shapes without knowing their specific types. To add a new shape (e.g., `Triangle`), we can simply create a new class that inherits from `Shape` and implements the `area` method, without modifying the existing code.

Slide 8: LSP Example: Substitutability

The Liskov Substitution Principle ensures that objects of a superclass can be replaced with objects of its subclasses without affecting the correctness of the program.

```python
class Vehicle:
    def start_engine(self):
        pass

    def drive(self):
        print("Driving...")

class Car(Vehicle):
    def start_engine(self):
        print("Starting car engine.")

class ElectricCar(Car):
    def start_engine(self):
        print("Electric cars don't have engines.")

def start_and_drive(vehicle):
    vehicle.start_engine()
    vehicle.drive()

# Usage
car = Car()
electric_car = ElectricCar()

start_and_drive(car)       # Output: Starting car engine. Driving...
start_and_drive(electric_car)  # Output: Electric cars don't have engines. Driving...
```

In this example, the `ElectricCar` class adheres to the Liskov Substitution Principle because it overrides the `start_engine` method in a way that still makes sense for its type. The `start_and_drive` function can work with both `Car` and `ElectricCar` objects without any issues.

Slide 9: ISP Example: Interface Segregation

The Interface Segregation Principle promotes the separation of interfaces into smaller, more specific ones to avoid clients from depending on methods they don't use.

```python
from abc import ABC, abstractmethod

class Printer(ABC):
    @abstractmethod
    def print_document(self):
        pass

class Scanner(ABC):
    @abstractmethod
    def scan_document(self):
        pass

class MultiFunctionPrinter(Printer, Scanner):
    def print_document(self):
        print("Printing document...")

    def scan_document(self):
        print("Scanning document...")

class SimplePrinter(Printer):
    def print_document(self):
        print("Printing document...")

def print_documents(printers):
    for printer in printers:
        printer.print_document()

# Usage
multi_function_printer = MultiFunctionPrinter()
simple_printer = SimplePrinter()

printers = [multi_function_printer, simple_printer]
print_documents(printers)
```

In this example, the `Printer` and `Scanner` interfaces are segregated, allowing clients to depend only on the functionality they need. The `SimplePrinter` class only implements the `Printer` interface, while the `MultiFunctionPrinter` class implements both `Printer` and `Scanner` interfaces. The `print_documents` function can work with any object that implements the `Printer` interface, without being affected by the presence or absence of the `Scanner` interface.

Slide 10: DIP Example: Decoupling with Abstractions

The Dependency Inversion Principle promotes decoupling high-level and low-level modules by introducing abstractions that they both depend on.

```python
from abc import ABC, abstractmethod

class Logger(ABC):
    @abstractmethod
    def log(self, message):
        pass

class ConsoleLogger(Logger):
    def log(self, message):
        print(message)

class FileLogger(Logger):
    def __init__(self, file_path):
        self.file_path = file_path

    def log(self, message):
        with open(self.file_path, 'a') as file:
            file.write(message + '\n')

class PaymentProcessor:
    def __init__(self, logger: Logger):
        self.logger = logger

    def process_payment(self, amount):
        self.logger.log(f"Processing payment of ${amount}")
        print("Payment processed successfully.")

# Usage
console_logger = ConsoleLogger()
payment_processor = PaymentProcessor(console_logger)
payment_processor.process_payment(100)

file_logger = FileLogger("payment_logs.txt")
payment_processor = PaymentProcessor(file_logger)
payment_processor.process_payment(200)
```

In this example, the `PaymentProcessor` class depends on the `Logger` abstraction (interface) instead of a concrete implementation. This allows flexibility to switch between different logging implementations, such as `ConsoleLogger` or `FileLogger`, without modifying the `PaymentProcessor` class. The high-level module (`PaymentProcessor`) depends on the abstraction (`Logger`), adhering to the Dependency Inversion Principle.

Slide 11: SRP and OCP in Django

The Single Responsibility Principle and Open/Closed Principle are commonly applied in the Django web framework through the separation of concerns and the use of modular design patterns.

```python
# models.py
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)

# views.py
from django.shortcuts import render
from .models import Book

def book_list(request):
    books = Book.objects.all()
    return render(request, 'book_list.html', {'books': books})

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('books/', views.book_list, name='book_list'),
]
```

In this example, the `Book` model in `models.py` follows the Single Responsibility Principle by representing the book data. The `views.py` file handles the business logic for rendering the book list view. The `urls.py` file defines the URL patterns for the application. This separation of concerns promotes code organization, maintainability, and adherence to the Open/Closed Principle, as new functionality can be added without modifying existing code.

Slide 12: SOLID Principles in Practice

Applying the SOLID principles in your Python projects can lead to more maintainable, extensible, and testable code. While the principles may seem abstract at first, practical examples and code reviews can help reinforce their understanding and application.

Here are some tips for incorporating SOLID principles into your projects:

1. Conduct code reviews and refactoring sessions to identify potential violations and opportunities for improvement.
2. Encourage discussion and knowledge sharing among team members to foster a deeper understanding of the principles.
3. Continuously strive for simplicity and clarity in your code, as this often aligns with the goals of SOLID principles.
4. Embrace design patterns and architectural patterns that promote adherence to SOLID principles, such as the Model-View-Controller (MVC) pattern or the Repository pattern.
5. Continuously learn and practice by applying the principles in various projects, as the more you practice, the more natural it will become.

Remember, SOLID principles are not strict rules but rather guidelines to help you write better, more maintainable code. Applying them judiciously and adapting them to your specific project needs is key to reaping their benefits.

## Meta:
Certainly! Here's an example that adheres to the Liskov Substitution Principle (LSP):

```python
class Vehicle:
    def start(self):
        pass

    def drive(self):
        print("Driving...")

class Car(Vehicle):
    def start(self):
        print("Starting car engine.")

class ElectricCar(Vehicle):
    def start(self):
        print("Initializing electric motor.")

def start_and_drive(vehicle):
    vehicle.start()
    vehicle.drive()

# Usage
car = Car()
electric_car = ElectricCar()

start_and_drive(car)       # Output: Starting car engine. Driving...
start_and_drive(electric_car)  # Output: Initializing electric motor. Driving...
```

In this example, we have an abstract base class `Vehicle` with two methods: `start` and `drive`. The `Car` class inherits from `Vehicle` and overrides the `start` method to print "Starting car engine." The `ElectricCar` class also inherits from `Vehicle` but overrides the `start` method to print "Initializing electric motor."

The `start_and_drive` function takes a `Vehicle` object as an argument and calls its `start` and `drive` methods. When we call `start_and_drive` with a `Car` object, it prints "Starting car engine. Driving..." When we call it with an `ElectricCar` object, it prints "Initializing electric motor. Driving..."

By using a more generic method name `start` instead of `start_engine`, we avoid violating the Liskov Substitution Principle. Both `Car` and `ElectricCar` implement the `start` method in a way that makes sense for their respective types, and they can be used interchangeably without causing any unexpected behavior.

This example demonstrates how adhering to LSP can lead to more flexible and extensible code. If we need to add a new type of vehicle in the future, such as a hybrid car or a hydrogen-powered car, we can create a new subclass of `Vehicle` and implement the `start` method accordingly, without modifying the existing classes or the `start_and_drive` function.

