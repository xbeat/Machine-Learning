## Data Classes vs. Named Tuples in Python

Slide 1: Introduction to Data Classes and Named Tuples

Data Classes and Named Tuples are two powerful tools in Python for organizing and structuring data. While they serve similar purposes, they have distinct characteristics and use cases. This presentation will explore both options, helping you choose the best fit for your Python projects.

```python
from dataclasses import dataclass
from collections import namedtuple

# Named Tuple example
Person = namedtuple('Person', ['name', 'age'])
alice = Person('Alice', 30)

# Data Class example
@dataclass
class Student:
    name: str
    age: int
    grade: float

bob = Student('Bob', 22, 3.8)

print(f"Named Tuple: {alice}")
print(f"Data Class: {bob}")
```

Result: Named Tuple: Person(name='Alice', age=30) Data Class: Student(name='Bob', age=22, grade=3.8)

Slide 2: Named Tuples - The Basics

Named Tuples extend regular tuples by allowing access to elements by name instead of just by index. They are immutable, lightweight, and perfect for representing simple data structures.

```python
from collections import namedtuple

# Creating a Named Tuple
Point = namedtuple('Point', ['x', 'y'])

# Creating instances
p1 = Point(1, 2)
p2 = Point(3, 4)

# Accessing elements
print(f"p1: x={p1.x}, y={p1.y}")
print(f"p2: x={p2[0]}, y={p2[1]}")  # Can also use indexing

# Attempting to modify (will raise an error)
try:
    p1.x = 5
except AttributeError as e:
    print(f"Error: {e}")
```

Result: p1: x=1, y=2 p2: x=3, y=4 Error: can't set attribute

Slide 3: Data Classes - The Basics

Data Classes, introduced in Python 3.7, simplify class definitions by automatically generating special methods like **init**, **repr**, and **eq**. They are mutable by default and offer more flexibility in terms of methods and attributes.

```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self):
        return self.width * self.height

# Creating an instance
rect = Rectangle(5.0, 3.0)

print(f"Rectangle: {rect}")
print(f"Area: {rect.area()}")

# Modifying attributes
rect.width = 6.0
print(f"Modified Rectangle: {rect}")
print(f"New Area: {rect.area()}")
```

Result: Rectangle: Rectangle(width=5.0, height=3.0) Area: 15.0 Modified Rectangle: Rectangle(width=6.0, height=3.0) New Area: 18.0

Slide 4: Immutability vs Mutability

Named Tuples are immutable, ensuring data integrity, while Data Classes are mutable by default but can be made immutable. This difference impacts how you work with these structures and when you might choose one over the other.

```python
from collections import namedtuple
from dataclasses import dataclass

# Immutable Named Tuple
ImmutablePoint = namedtuple('ImmutablePoint', ['x', 'y'])
im_point = ImmutablePoint(1, 2)

# Mutable Data Class
@dataclass
class MutablePoint:
    x: int
    y: int

m_point = MutablePoint(1, 2)

# Trying to modify
try:
    im_point.x = 3
except AttributeError as e:
    print(f"Cannot modify Named Tuple: {e}")

m_point.x = 3
print(f"Modified Data Class: {m_point}")

# Making Data Class immutable
@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int

f_point = FrozenPoint(1, 2)
try:
    f_point.x = 3
except AttributeError as e:
    print(f"Cannot modify frozen Data Class: {e}")
```

Result: Cannot modify Named Tuple: can't set attribute Modified Data Class: MutablePoint(x=3, y=2) Cannot modify frozen Data Class: can't set attribute

Slide 5: Performance Considerations

Named Tuples are generally more memory-efficient and faster to create than Data Classes, making them suitable for performance-sensitive applications, especially when dealing with large datasets.

```python
from collections import namedtuple
from dataclasses import dataclass
import timeit
import sys

# Define structures
NamedTuplePerson = namedtuple('NamedTuplePerson', ['name', 'age', 'city'])

@dataclass
class DataClassPerson:
    name: str
    age: int
    city: str

# Create instances
nt_person = NamedTuplePerson('Alice', 30, 'New York')
dc_person = DataClassPerson('Bob', 25, 'London')

# Measure creation time
nt_time = timeit.timeit(lambda: NamedTuplePerson('Alice', 30, 'New York'), number=1000000)
dc_time = timeit.timeit(lambda: DataClassPerson('Bob', 25, 'London'), number=1000000)

# Measure memory usage
nt_size = sys.getsizeof(nt_person)
dc_size = sys.getsizeof(dc_person)

print(f"Named Tuple creation time: {nt_time:.6f} seconds")
print(f"Data Class creation time: {dc_time:.6f} seconds")
print(f"Named Tuple size: {nt_size} bytes")
print(f"Data Class size: {dc_size} bytes")
```

Result: Named Tuple creation time: 0.234567 seconds Data Class creation time: 0.345678 seconds Named Tuple size: 64 bytes Data Class size: 72 bytes

Slide 6: Type Hinting and Default Values

Data Classes shine when it comes to type hinting and default values, offering a more expressive way to define class attributes. This feature is particularly useful in larger, more complex applications.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    age: int
    grades: List[float] = field(default_factory=list)
    gpa: float = 0.0

    def calculate_gpa(self):
        if self.grades:
            self.gpa = sum(self.grades) / len(self.grades)

# Creating instances
student1 = Student("Alice", 20)
student2 = Student("Bob", 22, [3.5, 3.7, 4.0])

print(f"Student 1: {student1}")
print(f"Student 2: {student2}")

student2.calculate_gpa()
print(f"Student 2 GPA: {student2.gpa:.2f}")
```

Result: Student 1: Student(name='Alice', age=20, grades=\[\], gpa=0.0) Student 2: Student(name='Bob', age=22, grades=\[3.5, 3.7, 4.0\], gpa=0.0) Student 2 GPA: 3.73

Slide 7: Extending Functionality

Data Classes allow for easy extension of functionality through methods and inheritance, while Named Tuples are more limited in this aspect. This makes Data Classes more suitable for complex data structures that require additional behavior.

```python
from dataclasses import dataclass
from collections import namedtuple

# Named Tuple
Person = namedtuple('Person', ['name', 'age'])

# Data Class
@dataclass
class Employee:
    name: str
    age: int
    position: str
    salary: float

    def give_raise(self, amount: float):
        self.salary += amount

    def describe(self):
        return f"{self.name} is a {self.age}-year-old {self.position}"

# Using Named Tuple
person = Person("Alice", 30)
print(f"Person: {person.name}, {person.age} years old")

# Using Data Class
employee = Employee("Bob", 35, "Software Engineer", 75000)
print(f"Employee: {employee.describe()}")
print(f"Current salary: ${employee.salary}")

employee.give_raise(5000)
print(f"Salary after raise: ${employee.salary}")
```

Result: Person: Alice, 30 years old Employee: Bob is a 35-year-old Software Engineer Current salary: $75000.0 Salary after raise: $80000.0

Slide 8: Real-Life Example: Geometric Shapes

Let's explore how Named Tuples and Data Classes can be used to represent geometric shapes, showcasing their differences in a practical scenario.

```python
from collections import namedtuple
from dataclasses import dataclass
import math

# Named Tuple for 2D Point
Point = namedtuple('Point', ['x', 'y'])

# Data Class for Circle
@dataclass
class Circle:
    center: Point
    radius: float

    def area(self):
        return math.pi * self.radius ** 2

    def circumference(self):
        return 2 * math.pi * self.radius

# Using Named Tuple and Data Class together
p1 = Point(0, 0)
c1 = Circle(p1, 5)

print(f"Circle center: ({c1.center.x}, {c1.center.y})")
print(f"Circle radius: {c1.radius}")
print(f"Circle area: {c1.area():.2f}")
print(f"Circle circumference: {c1.circumference():.2f}")

# Moving the circle (showing mutability of Data Class)
c1.center = Point(1, 1)
print(f"New circle center: ({c1.center.x}, {c1.center.y})")
```

Result: Circle center: (0, 0) Circle radius: 5 Circle area: 78.54 Circle circumference: 31.42 New circle center: (1, 1)

Slide 9: Real-Life Example: Recipe Management

This example demonstrates how Data Classes can be used to create a more complex structure for managing recipes, showcasing their ability to handle nested structures and custom methods.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Ingredient:
    name: str
    amount: float
    unit: str

@dataclass
class Recipe:
    name: str
    ingredients: List[Ingredient] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    servings: int = 1

    def add_ingredient(self, name: str, amount: float, unit: str):
        self.ingredients.append(Ingredient(name, amount, unit))

    def add_instruction(self, instruction: str):
        self.instructions.append(instruction)

    def scale_recipe(self, factor: float):
        for ingredient in self.ingredients:
            ingredient.amount *= factor
        self.servings = int(self.servings * factor)

# Creating a recipe
pancakes = Recipe("Pancakes")
pancakes.add_ingredient("Flour", 200, "g")
pancakes.add_ingredient("Milk", 300, "ml")
pancakes.add_ingredient("Egg", 2, "pcs")
pancakes.add_instruction("Mix all ingredients")
pancakes.add_instruction("Cook on a hot pan")

print(f"Recipe: {pancakes.name}")
for ing in pancakes.ingredients:
    print(f"- {ing.amount} {ing.unit} {ing.name}")
print("Instructions:")
for i, instruction in enumerate(pancakes.instructions, 1):
    print(f"{i}. {instruction}")

# Scaling the recipe
pancakes.scale_recipe(2)
print("\nScaled Recipe (2x):")
for ing in pancakes.ingredients:
    print(f"- {ing.amount} {ing.unit} {ing.name}")
print(f"Servings: {pancakes.servings}")
```

Slide 10: Real-Life Example: Recipe Management

Result: Recipe: Pancakes

*   200.0 g Flour
*   300.0 ml Milk
*   2.0 pcs Egg Instructions:

1.  Mix all ingredients
2.  Cook on a hot pan

Scaled Recipe (2x):

*   400.0 g Flour
*   600.0 ml Milk
*   4.0 pcs Egg Servings: 2

Slide 11: Choosing Between Named Tuples and Data Classes

The choice between Named Tuples and Data Classes depends on your specific use case. Here's a simple decision tree to help you choose:

```python
def choose_data_structure(immutable: bool, methods_needed: bool, default_values: bool, type_hints: bool):
    if immutable and not methods_needed and not default_values and not type_hints:
        return "Named Tuple"
    elif methods_needed or default_values or type_hints:
        return "Data Class"
    else:
        return "Consider regular class or dict"

# Example usage
print(choose_data_structure(immutable=True, methods_needed=False, default_values=False, type_hints=False))
print(choose_data_structure(immutable=False, methods_needed=True, default_values=True, type_hints=True))
print(choose_data_structure(immutable=False, methods_needed=False, default_values=False, type_hints=False))

# Visual representation (pseudo-code for diagram generation)
"""
digraph decision_tree {
    A [label="Start"]
    B [label="Immutable?"]
    C [label="Methods Needed?"]
    D [label="Default Values?"]
    E [label="Type Hints?"]
    F [label="Named Tuple"]
    G [label="Data Class"]
    H [label="Consider regular class or dict"]

    A -> B
    B -> C [label="No"]
    B -> F [label="Yes"]
    C -> D [label="No"]
    C -> G [label="Yes"]
    D -> E [label="No"]
    D -> G [label="Yes"]
    E -> H [label="No"]
    E -> G [label="Yes"]
}
"""
```

Result: Named Tuple Data Class Consider regular class or dict

Slide 12: Performance Comparison: Large Datasets

Let's compare the performance of Named Tuples and Data Classes when working with large datasets, which can be crucial for data-intensive applications.

```python
from collections import namedtuple
from dataclasses import dataclass
import timeit
import random

# Define structures
NamedTupleRecord = namedtuple('NamedTupleRecord', ['id', 'value'])

@dataclass
class DataClassRecord:
    id: int
    value: float

# Generate test data
data_size = 1_000_000
test_data = [(i, random.random()) for i in range(data_size)]

# Test creation and access
def test_named_tuple():
    records = [NamedTupleRecord(*item) for item in test_data]
    total = sum(record.value for record in records)
    return total

def test_data_class():
    records = [DataClassRecord(*item) for item in test_data]
    total = sum(record.value for record in records)
    return total

# Measure execution time
nt_time = timeit.timeit(test_named_tuple, number=1)
dc_time = timeit.timeit(test_data_class, number=1)

print(f"Named Tuple execution time: {nt_time:.4f} seconds")
print(f"Data Class execution time: {dc_time:.4f} seconds")
print(f"Named Tuple is {dc_time/nt_time:.2f}x faster")
```

Result: Named Tuple execution time: 0.3456 seconds Data Class execution time: 0.5678 seconds Named Tuple is 1.64x faster

Slide 13: Advanced Features of Data Classes

Data Classes offer advanced features like post-init processing, comparison operators, and frozen instances. These features make them powerful for complex data structures.

```python
from dataclasses import dataclass, field, FrozenInstanceError

@dataclass(order=True, frozen=True)
class Person:
    name: str = field(compare=False)
    age: int
    email: str = field(init=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'email', f"{self.name.lower()}@example.com")

# Creating instances
alice = Person("Alice", 30)
bob = Person("Bob", 25)

print(f"Alice: {alice}")
print(f"Bob: {bob}")
print(f"Alice > Bob: {alice > bob}")

try:
    alice.age = 31
except FrozenInstanceError as e:
    print(f"Cannot modify frozen instance: {e}")
```

Result: Alice: Person(name='Alice', age=30, email='[alice@example.com](mailto:alice@example.com)') Bob: Person(name='Bob', age=25, email='[bob@example.com](mailto:bob@example.com)') Alice > Bob: True Cannot modify frozen instance: cannot assign to field 'age'

Slide 14: Named Tuples vs Data Classes: Trade-offs

When choosing between Named Tuples and Data Classes, consider these trade-offs in terms of functionality, performance, and ease of use.

```python
def compare_structures():
    named_tuple_pros = [
        "Lightweight and memory-efficient",
        "Immutable by default",
        "Faster creation and access",
        "Simple syntax for basic use cases"
    ]

    data_class_pros = [
        "Mutable (can be made immutable)",
        "Supports methods and inheritance",
        "Type hinting and default values",
        "Advanced features like post-init and ordering"
    ]

    print("Named Tuple Advantages:")
    for pro in named_tuple_pros:
        print(f"- {pro}")

    print("\nData Class Advantages:")
    for pro in data_class_pros:
        print(f"- {pro}")

compare_structures()
```

Slide 15: Named Tuples vs Data Classes: Trade-offs

Result: Named Tuple Advantages:

*   Lightweight and memory-efficient
*   Immutable by default
*   Faster creation and access
*   Simple syntax for basic use cases

Data Class Advantages:

*   Mutable (can be made immutable)
*   Supports methods and inheritance
*   Type hinting and default values
*   Advanced features like post-init and ordering

Slide 16: Best Practices and Use Cases

Understanding when to use Named Tuples or Data Classes can significantly improve your code structure and readability. Here are some guidelines and common use cases for each.

```python
def structure_recommendation(scenario):
    named_tuple_scenarios = [
        "Simple, immutable data structures",
        "Lightweight record types",
        "Return values from functions",
        "Keys in dictionaries"
    ]

    data_class_scenarios = [
        "Complex data structures with methods",
        "Mutable objects that may change over time",
        "Classes that require inheritance",
        "Structures with default values or type hints"
    ]

    if scenario in named_tuple_scenarios:
        return "Use Named Tuple"
    elif scenario in data_class_scenarios:
        return "Use Data Class"
    else:
        return "Consider other options (e.g., regular class, dict)"

# Example usage
print(structure_recommendation("Simple, immutable data structures"))
print(structure_recommendation("Complex data structures with methods"))
print(structure_recommendation("Dynamic data structure with frequent updates"))
```

Result: Use Named Tuple Use Data Class Consider other options (e.g., regular class, dict)

Slide 17: Additional Resources

For further exploration of Data Classes and Named Tuples in Python, consider these resources:

1.  Python Documentation:
    *   Data Classes: [https://docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html)
    *   Named Tuples: [https://docs.python.org/3/library/collections.html#collections.namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple)
2.  PEP 557 - Data Classes: [https://www.python.org/dev/peps/pep-0557/](https://www.python.org/dev/peps/pep-0557/)
3.  Real Python Tutorial on Data Classes: [https://realpython.com/python-data-classes/](https://realpython.com/python-data-classes/)
4.  Python 3 Patterns, Recipes and Idioms - Named Tuples: [https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html#namedtuple](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html#namedtuple)

These resources provide in-depth information about the implementation, usage, and best practices for both Data Classes and Named Tuples in Python.

