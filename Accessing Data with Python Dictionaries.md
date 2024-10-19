## Accessing Data with Python Dictionaries
Slide 1: What are Python Dictionaries?

Python dictionaries are ordered collections of key-value pairs. They provide a flexible and efficient way to store and retrieve data using unique keys instead of numeric indices. Dictionaries are mutable, meaning their contents can be modified after creation. They are particularly useful for representing structured data or creating lookup tables.

```python
# Creating a dictionary
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

# Accessing values
print(f"Student name: {student['name']}")
print(f"Student age: {student['age']}")

# Output:
# Student name: Alice
# Student age: 20
```

Slide 2: Creating and Modifying Dictionaries

Dictionaries can be created using curly braces {} or the dict() constructor. Keys and values are separated by colons, and key-value pairs are separated by commas. You can add, modify, or remove key-value pairs after creation.

```python
# Creating a dictionary
fruits = {"apple": 3, "banana": 2, "orange": 5}

# Adding a new key-value pair
fruits["grape"] = 4

# Modifying an existing value
fruits["banana"] = 6

# Removing a key-value pair
del fruits["orange"]

print(fruits)
# Output: {'apple': 3, 'banana': 6, 'grape': 4}
```

Slide 3: Accessing Dictionary Values

Dictionary values can be accessed using their corresponding keys. If a key doesn't exist, attempting to access it will raise a KeyError. To avoid this, you can use the get() method, which returns a default value if the key is not found.

```python
car = {"make": "Toyota", "model": "Corolla", "year": 2022}

# Accessing values
print(f"Car make: {car['make']}")

# Using get() method
color = car.get("color", "Not specified")
print(f"Car color: {color}")

# Output:
# Car make: Toyota
# Car color: Not specified
```

Slide 4: Iterating Through Dictionaries

Python provides several ways to iterate through dictionaries. You can loop through keys, values, or key-value pairs using the keys(), values(), and items() methods, respectively.

```python
grades = {"Math": 90, "English": 85, "Science": 95}

# Iterating through keys
for subject in grades.keys():
    print(f"Subject: {subject}")

# Iterating through values
for score in grades.values():
    print(f"Score: {score}")

# Iterating through key-value pairs
for subject, score in grades.items():
    print(f"{subject}: {score}")

# Output:
# Subject: Math
# Subject: English
# Subject: Science
# Score: 90
# Score: 85
# Score: 95
# Math: 90
# English: 85
# Science: 95
```

Slide 5: Dictionary Comprehension

Dictionary comprehension is a concise way to create dictionaries using a single line of code. It follows a similar syntax to list comprehension but uses curly braces instead of square brackets.

```python
# Creating a dictionary of squares
numbers = [1, 2, 3, 4, 5]
squares = {num: num**2 for num in numbers}

print(squares)
# Output: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filtering even numbers and squaring them
even_squares = {num: num**2 for num in numbers if num % 2 == 0}

print(even_squares)
# Output: {2: 4, 4: 16}
```

Slide 6: Nested Dictionaries

Dictionaries can contain other dictionaries as values, creating nested structures. This is useful for representing complex, hierarchical data.

```python
# Nested dictionary representing a bookstore
bookstore = {
    "fiction": {
        "sci-fi": ["Dune", "Neuromancer"],
        "fantasy": ["The Hobbit", "Harry Potter"]
    },
    "non-fiction": {
        "science": ["A Brief History of Time", "The Selfish Gene"],
        "history": ["Guns, Germs, and Steel", "1776"]
    }
}

# Accessing nested values
print(f"Science fiction books: {bookstore['fiction']['sci-fi']}")
print(f"History books: {bookstore['non-fiction']['history']}")

# Output:
# Science fiction books: ['Dune', 'Neuromancer']
# History books: ['Guns, Germs, and Steel', '1776']
```

Slide 7: Dictionary Methods - pop() and popitem()

The pop() method removes and returns the value for a specified key, while popitem() removes and returns the last inserted key-value pair as a tuple.

```python
inventory = {"apples": 50, "bananas": 30, "oranges": 40}

# Using pop()
removed_item = inventory.pop("bananas")
print(f"Removed item: {removed_item}")
print(f"Updated inventory: {inventory}")

# Using popitem()
last_item = inventory.popitem()
print(f"Last item removed: {last_item}")
print(f"Final inventory: {inventory}")

# Output:
# Removed item: 30
# Updated inventory: {'apples': 50, 'oranges': 40}
# Last item removed: ('oranges', 40)
# Final inventory: {'apples': 50}
```

Slide 8: Dictionary Methods - update() and clear()

The update() method adds or updates multiple key-value pairs from another dictionary or an iterable of key-value pairs. The clear() method removes all items from the dictionary.

```python
original = {"a": 1, "b": 2}
new_items = {"b": 3, "c": 4}

# Using update()
original.update(new_items)
print(f"Updated dictionary: {original}")

# Using clear()
original.clear()
print(f"Cleared dictionary: {original}")

# Output:
# Updated dictionary: {'a': 1, 'b': 3, 'c': 4}
# Cleared dictionary: {}
```

Slide 9: Real-Life Example: Student Management System

Let's create a simple student management system using dictionaries to store and manipulate student data.

```python
# Student management system
students = {}

def add_student(name, age, grade):
    students[name] = {"age": age, "grade": grade}

def update_grade(name, new_grade):
    if name in students:
        students[name]["grade"] = new_grade
    else:
        print(f"Student {name} not found.")

def display_students():
    for name, info in students.items():
        print(f"{name}: Age {info['age']}, Grade {info['grade']}")

# Using the system
add_student("Alice", 15, "A")
add_student("Bob", 16, "B")
update_grade("Alice", "A+")
display_students()

# Output:
# Alice: Age 15, Grade A+
# Bob: Age 16, Grade B
```

Slide 10: Real-Life Example: Recipe Book

Let's create a simple recipe book using nested dictionaries to store and manage recipes.

```python
recipe_book = {}

def add_recipe(name, ingredients, instructions):
    recipe_book[name] = {
        "ingredients": ingredients,
        "instructions": instructions
    }

def display_recipe(name):
    if name in recipe_book:
        recipe = recipe_book[name]
        print(f"Recipe: {name}")
        print("Ingredients:")
        for ingredient in recipe["ingredients"]:
            print(f"- {ingredient}")
        print("Instructions:")
        for step, instruction in enumerate(recipe["instructions"], 1):
            print(f"{step}. {instruction}")
    else:
        print(f"Recipe '{name}' not found.")

# Using the recipe book
add_recipe("Pancakes", 
           ["2 cups flour", "2 eggs", "1 cup milk", "2 tbsp sugar"],
           ["Mix dry ingredients", "Add wet ingredients", "Cook on griddle"])

display_recipe("Pancakes")

# Output:
# Recipe: Pancakes
# Ingredients:
# - 2 cups flour
# - 2 eggs
# - 1 cup milk
# - 2 tbsp sugar
# Instructions:
# 1. Mix dry ingredients
# 2. Add wet ingredients
# 3. Cook on griddle
```

Slide 11: Dictionary Performance

Dictionaries in Python are implemented using hash tables, which provide constant-time (O(1)) average-case performance for insertion, deletion, and lookup operations. This makes them highly efficient for large datasets.

```python
import time

# Comparing list search vs. dictionary lookup
def list_search(n):
    list_data = list(range(n))
    start = time.time()
    _ = 999999 in list_data
    return time.time() - start

def dict_search(n):
    dict_data = {i: i for i in range(n)}
    start = time.time()
    _ = 999999 in dict_data
    return time.time() - start

n = 10_000_000
list_time = list_search(n)
dict_time = dict_search(n)

print(f"List search time: {list_time:.6f} seconds")
print(f"Dictionary search time: {dict_time:.6f} seconds")

# Output (results may vary):
# List search time: 0.135293 seconds
# Dictionary search time: 0.000001 seconds
```

Slide 12: Advanced Dictionary Operations

Python dictionaries support various advanced operations, including merging dictionaries, creating default dictionaries, and using dictionary views.

```python
from collections import defaultdict

# Merging dictionaries (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = dict1 | dict2
print(f"Merged dictionary: {merged}")

# Default dictionary
word_count = defaultdict(int)
sentence = "the quick brown fox jumps over the lazy dog"
for word in sentence.split():
    word_count[word] += 1
print(f"Word count: {dict(word_count)}")

# Dictionary views
prices = {"apple": 0.5, "banana": 0.3, "orange": 0.6}
items_view = prices.items()
prices["grape"] = 0.8
print(f"Updated items view: {list(items_view)}")

# Output:
# Merged dictionary: {'a': 1, 'b': 3, 'c': 4}
# Word count: {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1}
# Updated items view: [('apple', 0.5), ('banana', 0.3), ('orange', 0.6), ('grape', 0.8)]
```

Slide 13: Dictionary Unpacking and Keyword Arguments

Dictionaries can be unpacked using the \*\* operator, which is useful for passing keyword arguments to functions or combining dictionaries.

```python
def create_person(name, age, city):
    return f"{name} is {age} years old and lives in {city}."

person_info = {"name": "Alice", "age": 30, "city": "New York"}

# Unpacking dictionary as keyword arguments
result = create_person(**person_info)
print(result)

# Combining dictionaries
defaults = {"company": "Tech Corp", "position": "Engineer"}
custom = {"name": "Bob", "age": 28}
employee = {**defaults, **custom}
print(f"Employee info: {employee}")

# Output:
# Alice is 30 years old and lives in New York.
# Employee info: {'company': 'Tech Corp', 'position': 'Engineer', 'name': 'Bob', 'age': 28}
```

Slide 14: Additional Resources

For further exploration of Python dictionaries and their applications, consider the following resources:

1.  Python Official Documentation: [https://docs.python.org/3/tutorial/datastructures.html#dictionaries](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)
2.  Real Python - Python Dictionary: [https://realpython.com/python-dicts/](https://realpython.com/python-dicts/)
3.  ArXiv paper on efficient dictionary implementations: [https://arxiv.org/abs/1908.08037](https://arxiv.org/abs/1908.08037)

These resources provide in-depth explanations, advanced techniques, and research on dictionary implementations in Python and other programming languages.

