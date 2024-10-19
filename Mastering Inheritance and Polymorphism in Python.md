## Mastering Inheritance and Polymorphism in Python

Slide 1: Introduction to Inheritance and Polymorphism

Inheritance and polymorphism are fundamental concepts in object-oriented programming (OOP) that enhance code reusability, flexibility, and maintainability. These powerful features allow developers to create hierarchical relationships between classes and write more modular, extensible code. Let's explore these concepts in Python with practical examples.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass  # To be implemented by subclasses

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Creating instances
dog = Dog("Buddy")
cat = Cat("Whiskers")

# Demonstrating polymorphism
animals = [dog, cat]
for animal in animals:
    print(animal.speak())

# Output:
# Buddy says Woof!
# Whiskers says Meow!
```

Slide 2: Understanding Inheritance

Inheritance allows a new class (subclass) to inherit attributes and methods from an existing class (superclass). This promotes code reuse and establishes a hierarchical relationship between classes. In Python, we define a subclass by placing the superclass name in parentheses after the subclass name.

```python
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start_engine(self):
        return f"The {self.brand} {self.model}'s engine is starting."

class Car(Vehicle):
    def __init__(self, brand, model, fuel_type):
        super().__init__(brand, model)
        self.fuel_type = fuel_type

    def honk(self):
        return "Beep beep!"

# Creating an instance of Car
my_car = Car("Toyota", "Corolla", "Gasoline")

# Using inherited and subclass-specific methods
print(my_car.start_engine())
print(my_car.honk())

# Output:
# The Toyota Corolla's engine is starting.
# Beep beep!
```

Slide 3: Types of Inheritance

Python supports various types of inheritance, including single, multiple, and multilevel inheritance. Single inheritance involves a subclass inheriting from one superclass, while multiple inheritance allows a subclass to inherit from multiple superclasses. Multilevel inheritance creates a chain of inheritance with multiple levels.

```python
# Single Inheritance
class Animal:
    def __init__(self, species):
        self.species = species

class Dog(Animal):
    def bark(self):
        return "Woof!"

# Multiple Inheritance
class Flyer:
    def fly(self):
        return "I can fly!"

class Swimmer:
    def swim(self):
        return "I can swim!"

class Duck(Animal, Flyer, Swimmer):
    pass

# Multilevel Inheritance
class Mammal(Animal):
    def feed_young(self):
        return "Feeding with milk"

class Cat(Mammal):
    def purr(self):
        return "Purr..."

# Creating instances
dog = Dog("Canine")
duck = Duck("Anatidae")
cat = Cat("Feline")

# Demonstrating different types of inheritance
print(dog.species, dog.bark())
print(duck.species, duck.fly(), duck.swim())
print(cat.species, cat.feed_young(), cat.purr())

# Output:
# Canine Woof!
# Anatidae I can fly! I can swim!
# Feline Feeding with milk Purr...
```

Slide 4: Method Overriding

Method overriding allows a subclass to provide a specific implementation for a method that is already defined in its superclass. This enables customization of inherited behavior while maintaining the same method signature.

```python
class Shape:
    def __init__(self, color):
        self.color = color

    def area(self):
        return "Area calculation not implemented for this shape"

class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, color, width, height):
        super().__init__(color)
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

# Creating instances
circle = Circle("Red", 5)
rectangle = Rectangle("Blue", 4, 6)

# Demonstrating method overriding
print(f"Circle area: {circle.area()}")
print(f"Rectangle area: {rectangle.area()}")

# Output:
# Circle area: 78.5
# Rectangle area: 24
```

Slide 5: The super() Function

The `super()` function is used to call methods from a superclass in the subclass. It provides a clean way to extend or modify the behavior of inherited methods without completely replacing them.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id

    def introduce(self):
        base_intro = super().introduce()
        return f"{base_intro} My student ID is {self.student_id}."

# Creating instances
person = Person("Alice", 30)
student = Student("Bob", 20, "S12345")

# Demonstrating the use of super()
print(person.introduce())
print(student.introduce())

# Output:
# Hi, I'm Alice and I'm 30 years old.
# Hi, I'm Bob and I'm 20 years old. My student ID is S12345.
```

Slide 6: Understanding Polymorphism

Polymorphism allows objects of different classes to be treated as objects of a common superclass. It enables the use of a single interface to represent different underlying forms (data types or classes). In Python, polymorphism is achieved through method overriding and duck typing.

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

def print_area(shape):
    print(f"The area is: {shape.area()}")

# Creating instances
circle = Circle(5)
square = Square(4)

# Demonstrating polymorphism
print_area(circle)
print_area(square)

# Output:
# The area is: 78.5
# The area is: 16
```

Slide 7: Duck Typing

Duck typing is a concept in Python that focuses on the behavior of an object rather than its type. If an object has the methods and properties required by a function, it can be used regardless of its actual type.

```python
class Duck:
    def speak(self):
        return "Quack quack!"

class Dog:
    def speak(self):
        return "Woof woof!"

class Cat:
    def speak(self):
        return "Meow meow!"

def animal_sound(animal):
    return animal.speak()

# Creating instances
duck = Duck()
dog = Dog()
cat = Cat()

# Demonstrating duck typing
animals = [duck, dog, cat]
for animal in animals:
    print(animal_sound(animal))

# Output:
# Quack quack!
# Woof woof!
# Meow meow!
```

Slide 8: Abstract Base Classes

Abstract Base Classes (ABCs) provide a way to define interfaces in Python. They cannot be instantiated and may contain abstract methods that must be implemented by concrete subclasses.

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

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

    def perimeter(self):
        return 2 * 3.14 * self.radius

# Creating instances
rectangle = Rectangle(5, 3)
circle = Circle(4)

# Using abstract methods
shapes = [rectangle, circle]
for shape in shapes:
    print(f"Area: {shape.area()}, Perimeter: {shape.perimeter()}")

# Output:
# Area: 15, Perimeter: 16
# Area: 50.24, Perimeter: 25.12
```

Slide 9: Method Resolution Order (MRO)

Method Resolution Order (MRO) determines the order in which Python searches for methods in a class hierarchy, especially important in multiple inheritance scenarios.

```python
class A:
    def method(self):
        return "Method from A"

class B(A):
    def method(self):
        return "Method from B"

class C(A):
    def method(self):
        return "Method from C"

class D(B, C):
    pass

# Creating an instance of D
d = D()

# Demonstrating MRO
print(d.method())
print(D.mro())

# Output:
# Method from B
# [<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>]
```

Slide 10: Real-life Example: File System

Let's model a simple file system using inheritance and polymorphism. This example demonstrates how these concepts can be applied to represent different types of file system entities.

```python
from abc import ABC, abstractmethod

class FileSystemEntity(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_size(self):
        pass

class File(FileSystemEntity):
    def __init__(self, name, size):
        super().__init__(name)
        self.size = size

    def get_size(self):
        return self.size

class Directory(FileSystemEntity):
    def __init__(self, name):
        super().__init__(name)
        self.contents = []

    def add(self, entity):
        self.contents.append(entity)

    def get_size(self):
        return sum(entity.get_size() for entity in self.contents)

# Creating a file system structure
root = Directory("root")
documents = Directory("documents")
images = Directory("images")

root.add(documents)
root.add(images)

documents.add(File("report.doc", 1000))
documents.add(File("presentation.ppt", 2000))
images.add(File("photo.jpg", 1500))

# Calculating total size
print(f"Total size: {root.get_size()} bytes")

# Output:
# Total size: 4500 bytes
```

Slide 11: Real-life Example: Shape Drawing System

Let's create a simple shape drawing system that demonstrates inheritance, polymorphism, and method overriding. This example shows how these concepts can be applied in a graphical context.

```python
import math

class Shape:
    def __init__(self, color):
        self.color = color

    def draw(self):
        pass

    def area(self):
        pass

class Circle(Shape):
    def __init__(self, color, radius):
        super().__init__(color)
        self.radius = radius

    def draw(self):
        return f"Drawing a {self.color} circle with radius {self.radius}"

    def area(self):
        return math.pi * self.radius ** 2

class Rectangle(Shape):
    def __init__(self, color, width, height):
        super().__init__(color)
        self.width = width
        self.height = height

    def draw(self):
        return f"Drawing a {self.color} rectangle with width {self.width} and height {self.height}"

    def area(self):
        return self.width * self.height

class Triangle(Shape):
    def __init__(self, color, base, height):
        super().__init__(color)
        self.base = base
        self.height = height

    def draw(self):
        return f"Drawing a {self.color} triangle with base {self.base} and height {self.height}"

    def area(self):
        return 0.5 * self.base * self.height

# Creating shapes
shapes = [
    Circle("red", 5),
    Rectangle("blue", 4, 6),
    Triangle("green", 3, 4)
]

# Drawing shapes and calculating areas
for shape in shapes:
    print(shape.draw())
    print(f"Area: {shape.area():.2f}")
    print()

# Output:
# Drawing a red circle with radius 5
# Area: 78.54

# Drawing a blue rectangle with width 4 and height 6
# Area: 24.00

# Drawing a green triangle with base 3 and height 4
# Area: 6.00
```

Slide 12: Best Practices for Inheritance and Polymorphism

When working with inheritance and polymorphism, it's important to follow best practices to ensure clean, maintainable, and efficient code. Here are some guidelines:

1.  Follow the Liskov Substitution Principle (LSP): Subclasses should be substitutable for their base classes without affecting the correctness of the program.
2.  Use composition over inheritance when appropriate: Sometimes, it's better to compose objects rather than inherit from them.
3.  Keep the inheritance hierarchy shallow: Deep inheritance hierarchies can become complex and difficult to maintain.
4.  Use abstract base classes to define interfaces: This ensures that derived classes implement the required methods.
5.  Avoid multiple inheritance when possible: It can lead to the "diamond problem" and make the code harder to understand.
6.  Use method overriding judiciously: Override methods only when necessary and ensure that the overridden method's behavior is consistent with the base class.

Slide 13: Best Practices for Inheritance and Polymorphism

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

class Car(Vehicle):
    def start(self):
        return "Car engine started"

    def stop(self):
        return "Car engine stopped"

class Bicycle(Vehicle):
    def start(self):
        return "Bicycle started moving"

    def stop(self):
        return "Bicycle stopped moving"

def operate_vehicle(vehicle):
    print(vehicle.start())
    print(vehicle.stop())

# Using the vehicles
car = Car()
bicycle = Bicycle()

operate_vehicle(car)
print()
operate_vehicle(bicycle)

# Output:
# Car engine started
# Car engine stopped

# Bicycle started moving
# Bicycle stopped moving
```

Slide 14: Common Pitfalls and How to Avoid Them

When working with inheritance and polymorphism, developers may encounter several common pitfalls. Here are some issues to watch out for and how to avoid them:

1.  Overuse of inheritance: Avoid creating deep inheritance hierarchies. Instead, consider using composition or interfaces.
2.  Violating the Liskov Substitution Principle: Ensure that subclasses can be used interchangeably with their base classes without breaking the program's behavior.
3.  Tight coupling: Avoid creating strong dependencies between classes. Use dependency injection or inversion of control to reduce coupling.
4.  Incorrect method overriding: Make sure overridden methods have the same signature as the base class methods and maintain consistent behavior.
5.  Ignoring the "is-a" relationship: Only use inheritance when there's a true "is-a" relationship between the subclass and the superclass.

Slide 15: Common Pitfalls and How to Avoid Them

```python
# Problematic code
class Bird:
    def fly(self):
        return "Flying"

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins can't fly")
```

Slide 16: Common Pitfalls and How to Avoid Them

When working with inheritance and polymorphism, developers may encounter several common pitfalls. Here are some issues to watch out for and how to avoid them:

1.  Overuse of inheritance: Avoid creating deep inheritance hierarchies. Instead, consider using composition or interfaces.
2.  Violating the Liskov Substitution Principle: Ensure that subclasses can be used interchangeably with their base classes without breaking the program's behavior.
3.  Tight coupling: Avoid creating strong dependencies between classes. Use dependency injection or inversion of control to reduce coupling.
4.  Incorrect method overriding: Make sure overridden methods have the same signature as the base class methods and maintain consistent behavior.
5.  Ignoring the "is-a" relationship: Only use inheritance when there's a true "is-a" relationship between the subclass and the superclass.

Slide 17: Common Pitfalls and How to Avoid Them

```python
# Problematic code
class Bird:
    def fly(self):
        return "Flying"

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("Penguins can't fly")

# Better approach
class Animal:
    def move(self):
        pass

class Bird(Animal):
    def move(self):
        return "Flying"

class Penguin(Animal):
    def move(self):
        return "Swimming"

# Usage
def animal_movement(animal):
    print(animal.move())

bird = Bird()
penguin = Penguin()

animal_movement(bird)     # Output: Flying
animal_movement(penguin)  # Output: Swimming
```

Slide 18: Advanced Inheritance Techniques

Python offers advanced inheritance techniques that can be useful in specific scenarios. Let's explore some of these techniques:

1.  Mixins: Mixins are classes that provide additional functionality to other classes without being meant for instantiation themselves.
2.  Properties: Properties allow you to define methods that behave like attributes, providing getter, setter, and deleter functionality.
3.  Descriptors: Descriptors are objects that define how attribute access is handled, offering fine-grained control over attribute behavior.

Slide 19: Advanced Inheritance Techniques

```python
# Mixin example
class LoggingMixin:
    def log(self, message):
        print(f"Log: {message}")

class User(LoggingMixin):
    def __init__(self, name):
        self.name = name

    def greet(self):
        message = f"Hello, {self.name}!"
        self.log(message)
        return message

# Property example
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        return (self._celsius * 9/5) + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

# Usage
user = User("Alice")
print(user.greet())  # Output: Log: Hello, Alice!
                     #         Hello, Alice!

temp = Temperature(25)
print(f"{temp.fahrenheit:.1f}°F")  # Output: 77.0°F
temp.fahrenheit = 68
print(f"{temp._celsius:.1f}°C")    # Output: 20.0°C
```

Slide 20: Additional Resources

For further exploration of inheritance and polymorphism in Python, consider the following resources:

1.  Python's official documentation on classes: [https://docs.python.org/3/tutorial/classes.html](https://docs.python.org/3/tutorial/classes.html)
2.  "Design Patterns: Elements of Reusable Object-Oriented Software" by Gamma et al. - A classic book on OOP design patterns.
3.  "Fluent Python" by Luciano Ramalho - An in-depth guide to Python's object model.
4.  "Clean Code: A Handbook of Agile Software Craftsmanship" by Robert C. Martin - Provides principles for writing clean, maintainable code.
5.  Online courses on platforms like Coursera, edX, or Udacity that cover advanced Python OOP concepts.

Remember to always refer to the most up-to-date documentation and resources as programming languages and best practices evolve over time.

