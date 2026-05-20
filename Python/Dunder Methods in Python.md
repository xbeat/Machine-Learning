## Dunder Methods in Python
Slide 1: Introduction to Dunder Methods in Python

Dunder methods, short for "double underscore" methods, are special methods in Python that allow you to define how objects of a class behave in various situations. These methods are surrounded by double underscores on both sides, hence the name. They are also known as magic methods or special methods. Dunder methods enable you to customize the behavior of your objects, making them more intuitive and powerful.

```python
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return f"MyClass instance with value: {self.value}"

obj = MyClass(42)
print(obj)  # Output: MyClass instance with value: 42
```

Slide 2: The **init** Method: Object Initialization

The **init** method is one of the most commonly used dunder methods. It's called when an object is created and is used to initialize the object's attributes. This method allows you to set up the initial state of your object.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)
print(f"{person.name} is {person.age} years old")
# Output: Alice is 30 years old
```

Slide 3: The **str** and **repr** Methods: String Representation

The **str** method defines the "informal" string representation of an object, typically used for display to end-users. The **repr** method provides a more detailed, "official" string representation, often used for debugging and development.

```python
class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
    
    def __str__(self):
        return f"{self.title} by {self.author}"
    
    def __repr__(self):
        return f"Book(title='{self.title}', author='{self.author}')"

book = Book("1984", "George Orwell")
print(str(book))  # Output: 1984 by George Orwell
print(repr(book))  # Output: Book(title='1984', author='George Orwell')
```

Slide 4: The **len** Method: Object Length

The **len** method allows you to define what the len() function returns when called on your object. This is particularly useful for container-like objects.

```python
class Playlist:
    def __init__(self):
        self.songs = []
    
    def add_song(self, song):
        self.songs.append(song)
    
    def __len__(self):
        return len(self.songs)

playlist = Playlist()
playlist.add_song("Song 1")
playlist.add_song("Song 2")
playlist.add_song("Song 3")

print(f"The playlist has {len(playlist)} songs")
# Output: The playlist has 3 songs
```

Slide 5: The **getitem** and **setitem** Methods: Indexing and Slicing

These methods allow you to define how your object behaves when accessed using square bracket notation, enabling indexing and slicing operations.

```python
class CustomList:
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value

custom_list = CustomList([1, 2, 3, 4, 5])
print(custom_list[2])  # Output: 3
custom_list[2] = 10
print(custom_list[2])  # Output: 10
```

Slide 6: The **iter** and **next** Methods: Iteration

These methods make your objects iterable, allowing them to be used in for loops and with other iteration tools.

```python
class Countdown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

for num in Countdown(5):
    print(num)
# Output:
# 5
# 4
# 3
# 2
# 1
```

Slide 7: The **call** Method: Making Objects Callable

The **call** method allows you to make your objects callable, just like functions.

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, x):
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))  # Output: 10
print(triple(5))  # Output: 15
```

Slide 8: The **enter** and **exit** Methods: Context Management

These methods allow your objects to be used with the 'with' statement, enabling resource management and cleanup.

```python
class File:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

with File('example.txt', 'w') as f:
    f.write('Hello, World!')
# File is automatically closed after the 'with' block
```

Slide 9: The **add** and **radd** Methods: Operator Overloading

These methods allow you to define how the '+' operator behaves with your objects. **radd** is used when your object is on the right side of the '+' operator.

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __str__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(1, 2)
p2 = Point(3, 4)
print(p1 + p2)  # Output: Point(4, 6)
```

Slide 10: The **eq** and **hash** Methods: Equality and Hashing

These methods allow you to define how your objects are compared for equality and how they are hashed, which is important for using them in sets or as dictionary keys.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __eq__(self, other):
        return self.name == other.name and self.age == other.age
    
    def __hash__(self):
        return hash((self.name, self.age))

person1 = Person("Alice", 30)
person2 = Person("Alice", 30)
person3 = Person("Bob", 25)

print(person1 == person2)  # Output: True
print(person1 == person3)  # Output: False

person_set = {person1, person2, person3}
print(len(person_set))  # Output: 2 (person1 and person2 are considered the same)
```

Slide 11: The **bool** Method: Truth Value Testing

The **bool** method allows you to define the truth value of your objects when used in boolean contexts.

```python
class Container:
    def __init__(self, items):
        self.items = items
    
    def __bool__(self):
        return len(self.items) > 0

empty_container = Container([])
full_container = Container([1, 2, 3])

print(bool(empty_container))  # Output: False
print(bool(full_container))   # Output: True

if full_container:
    print("The container is not empty")
# Output: The container is not empty
```

Slide 12: Real-Life Example: Custom Temperature Class

Let's create a Temperature class that uses dunder methods to enable intuitive operations and conversions.

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius
    
    def __str__(self):
        return f"{self.celsius}°C"
    
    def __add__(self, other):
        return Temperature(self.celsius + other.celsius)
    
    def __lt__(self, other):
        return self.celsius < other.celsius
    
    def to_fahrenheit(self):
        return (self.celsius * 9/5) + 32

t1 = Temperature(20)
t2 = Temperature(30)

print(t1)  # Output: 20°C
print(t1 + t2)  # Output: 50°C
print(t1 < t2)  # Output: True
print(f"{t1} in Fahrenheit is {t1.to_fahrenheit()}°F")
# Output: 20°C in Fahrenheit is 68.0°F
```

Slide 13: Real-Life Example: Custom Data Structure

Let's implement a simple priority queue using dunder methods to make it behave like a native Python container.

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
    
    def __len__(self):
        return len(self._queue)
    
    def __iter__(self):
        return iter(sorted(self._queue))
    
    def add(self, item, priority):
        heapq.heappush(self._queue, (priority, item))
    
    def pop(self):
        return heapq.heappop(self._queue)[1]

tasks = PriorityQueue()
tasks.add("Complete project", 2)
tasks.add("Review code", 1)
tasks.add("Write tests", 3)

print(f"Number of tasks: {len(tasks)}")
for task in tasks:
    print(task[1])

# Output:
# Number of tasks: 3
# Review code
# Complete project
# Write tests
```

Slide 14: Additional Resources

For more in-depth information about Python's dunder methods and their usage, consider exploring the following resources:

1. Python Data Model documentation: [https://docs.python.org/3/reference/datamodel.html](https://docs.python.org/3/reference/datamodel.html)
2. "A Guide to Python's Magic Methods" by Rafe Kettler: [https://rszalski.github.io/magicmethods/](https://rszalski.github.io/magicmethods/)
3. "Fluent Python" by Luciano Ramalho, which covers advanced Python concepts including dunder methods.
4. Python's official tutorial on classes: [https://docs.python.org/3/tutorial/classes.html](https://docs.python.org/3/tutorial/classes.html)

These resources provide comprehensive explanations and examples of dunder methods and their applications in Python programming.

