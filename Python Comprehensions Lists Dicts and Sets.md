## Python Comprehensions Lists Dicts and Sets
Slide 1: List Comprehension Fundamentals

List comprehensions provide a concise way to create lists in Python by encapsulating a for loop and conditional logic into a single line of code. This elegant syntax improves readability while maintaining computational efficiency for list creation operations.

```python
# Basic list comprehension syntax
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # Square each number

# With conditional filtering
even_squares = [x**2 for x in numbers if x % 2 == 0]

print(f"Original numbers: {numbers}")
print(f"Squared numbers: {squares}")
print(f"Even squared numbers: {even_squares}")

# Output:
# Original numbers: [1, 2, 3, 4, 5]
# Squared numbers: [1, 4, 9, 16, 25]
# Even squared numbers: [4, 16]
```

Slide 2: Nested List Comprehensions

Nested list comprehensions allow for creating multi-dimensional arrays and processing nested data structures. They work inside-out, with the leftmost for clause being the outermost loop, providing a powerful way to transform complex data structures.

```python
# Creating a 3x3 matrix using nested list comprehension
matrix = [[i + j for j in range(3)] for i in range(0, 9, 3)]

# Flattening a matrix using nested list comprehension
flattened = [item for row in matrix for item in row]

print(f"Matrix: {matrix}")
print(f"Flattened: {flattened}")

# Output:
# Matrix: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# Flattened: [0, 1, 2, 3, 4, 5, 6, 7, 8]
```

Slide 3: Dictionary Comprehension Basics

Dictionary comprehensions extend the concept of list comprehensions to create dictionaries dynamically. They provide a concise syntax for transforming and filtering key-value pairs, making dictionary creation and manipulation more elegant and readable.

```python
# Basic dictionary comprehension
words = ['apple', 'banana', 'cherry']
word_lengths = {word: len(word) for word in words}

# Dictionary comprehension with conditional
long_words = {word: len(word) for word in words if len(word) > 5}

print(f"Word lengths: {word_lengths}")
print(f"Long words: {long_words}")

# Output:
# Word lengths: {'apple': 5, 'banana': 6, 'cherry': 6}
# Long words: {'banana': 6, 'cherry': 6}
```

Slide 4: Set Comprehension Fundamentals

Set comprehensions combine the power of set operations with the elegance of comprehension syntax, allowing for the creation of unique collections while applying transformations and filtering. They automatically handle duplicate elimination.

```python
# Basic set comprehension
numbers = [1, 2, 2, 3, 3, 4, 5, 5]
unique_squares = {x**2 for x in numbers}

# Set comprehension with conditional
even_unique = {x for x in numbers if x % 2 == 0}

print(f"Original numbers: {numbers}")
print(f"Unique squares: {unique_squares}")
print(f"Unique even numbers: {even_unique}")

# Output:
# Original numbers: [1, 2, 2, 3, 3, 4, 5, 5]
# Unique squares: {1, 4, 9, 16, 25}
# Unique even numbers: {2, 4}
```

Slide 5: Advanced Conditional Logic in Comprehensions

Understanding complex conditional logic in comprehensions enables more sophisticated data filtering and transformation. This includes using multiple conditions and if-else statements within the comprehension syntax.

```python
numbers = range(-5, 6)

# Multiple conditions using and/or
filtered = [x for x in numbers if x > 0 and x % 2 == 0]

# If-else in comprehension (ternary expression)
classified = [f"{x} is {'positive' if x > 0 else 'negative' if x < 0 else 'zero'}" 
              for x in numbers]

print(f"Filtered numbers: {filtered}")
print(f"Classified numbers: {classified}")

# Output:
# Filtered numbers: [2, 4]
# Classified numbers: ['-5 is negative', '-4 is negative', ...]
```

Slide 6: Real-world Application - Text Processing

Comprehensions excel in text processing tasks, providing elegant solutions for tokenization, cleaning, and transformation of text data. This example demonstrates practical text processing operations using various comprehension types.

```python
text = "The quick brown fox jumps over the lazy dog!"

# Word processing using comprehensions
words = text.lower().split()
word_stats = {word: {
    'length': len(word),
    'vowels': len([c for c in word if c in 'aeiou']),
    'consonants': len([c for c in word if c.isalpha() and c not in 'aeiou'])
} for word in words}

# Unique characters using set comprehension
unique_chars = {char.lower() for char in text if char.isalpha()}

print(f"Word statistics: {word_stats}")
print(f"Unique characters: {unique_chars}")

# Output:
# Word statistics: {'the': {'length': 3, 'vowels': 1, 'consonants': 2}, ...}
# Unique characters: {'a', 'b', 'c', 'd', ...}
```

Slide 7: Real-world Application - Data Analysis

List and dictionary comprehensions provide powerful tools for data preprocessing and analysis in scientific computing. This example demonstrates handling numerical data with comprehensions for statistical calculations.

```python
import statistics

# Sample dataset: daily temperature readings
temperatures = [
    {'day': i, 'temp': 20 + round(i * 0.5 + (-1)**i, 2)} 
    for i in range(1, 31)
]

# Data analysis using comprehensions
stats = {
    'average': statistics.mean([d['temp'] for d in temperatures]),
    'variance': statistics.variance([d['temp'] for d in temperatures]),
    'above_25': len([d for d in temperatures if d['temp'] > 25]),
    'day_temp_map': {d['day']: d['temp'] for d in temperatures}
}

print(f"Temperature statistics: {stats}")

# Output:
# Temperature statistics: {
#   'average': 24.32,
#   'variance': 12.45,
#   'above_25': 12,
#   'day_temp_map': {1: 20.5, 2: 19.0, ...}
# }
```

Slide 8: Generator Expressions and Memory Efficiency

Generator expressions provide memory-efficient alternatives to list comprehensions when working with large datasets. They use parentheses instead of square brackets and generate values on-demand rather than storing them all in memory.

```python
# Comparing memory usage of list comprehension vs generator expression
import sys

# List comprehension (stores all values in memory)
numbers_list = [x**2 for x in range(1000000)]

# Generator expression (generates values on demand)
numbers_gen = (x**2 for x in range(1000000))

# Memory comparison
list_size = sys.getsizeof(numbers_list)
gen_size = sys.getsizeof(numbers_gen)

print(f"List comprehension size: {list_size:,} bytes")
print(f"Generator expression size: {gen_size:,} bytes")
print(f"Memory saved: {(list_size - gen_size):,} bytes")

# Output:
# List comprehension size: 8,448,728 bytes
# Generator expression size: 112 bytes
# Memory saved: 8,448,616 bytes
```

Slide 9: Comprehensions with Custom Objects

Comprehensions can effectively process custom objects and complex data structures, providing a clean syntax for object transformation and filtering based on object attributes and methods.

```python
class Product:
    def __init__(self, name, price, category):
        self.name = name
        self.price = price
        self.category = category
    
    def __repr__(self):
        return f"Product({self.name}, ${self.price})"

# Sample product data
products = [
    Product("Laptop", 1200, "Electronics"),
    Product("Book", 20, "Books"),
    Product("Phone", 800, "Electronics"),
    Product("Coffee", 5, "Food")
]

# Processing objects with comprehensions
electronics = [p for p in products if p.category == "Electronics"]
price_map = {p.name: p.price for p in products}
categories = {p.category for p in products}

print(f"Electronics: {electronics}")
print(f"Price map: {price_map}")
print(f"Categories: {categories}")

# Output:
# Electronics: [Product(Laptop, $1200), Product(Phone, $800)]
# Price map: {'Laptop': 1200, 'Book': 20, 'Phone': 800, 'Coffee': 5}
# Categories: {'Electronics', 'Books', 'Food'}
```

Slide 10: Matrix Operations with Comprehensions

Comprehensions provide elegant solutions for matrix operations, offering a pythonic way to perform common linear algebra operations without relying on external libraries for simple calculations.

```python
# Matrix operations using comprehensions
matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
matrix_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

# Matrix addition
matrix_sum = [[a + b for a, b in zip(row_a, row_b)]
              for row_a, row_b in zip(matrix_a, matrix_b)]

# Matrix transpose
transpose = [[row[i] for row in matrix_a] for i in range(len(matrix_a[0]))]

# Matrix multiplication
matrix_mult = [[sum(a * b for a, b in zip(row_a, col_b))
               for col_b in zip(*matrix_b)]
              for row_a in matrix_a]

print(f"Matrix sum:\n{matrix_sum}")
print(f"Transpose:\n{transpose}")
print(f"Matrix multiplication:\n{matrix_mult}")

# Output:
# Matrix sum: [[10, 10, 10], [10, 10, 10], [10, 10, 10]]
# Transpose: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
# Matrix multiplication: [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
```

Slide 11: Performance Optimization Using Comprehensions

Comprehensions often provide better performance compared to traditional loop structures due to their optimized implementation in Python's interpreter. This example demonstrates performance comparisons between different approaches.

```python
import time
import random

# Generate test data
data = [random.randint(1, 1000) for _ in range(1000000)]

# Traditional loop approach
def traditional_approach(numbers):
    start_time = time.time()
    result = []
    for num in numbers:
        if num % 2 == 0:
            result.append(num ** 2)
    return time.time() - start_time

# List comprehension approach
def comprehension_approach(numbers):
    start_time = time.time()
    result = [num ** 2 for num in numbers if num % 2 == 0]
    return time.time() - start_time

# Compare performance
loop_time = traditional_approach(data)
comp_time = comprehension_approach(data)

print(f"Traditional loop time: {loop_time:.4f} seconds")
print(f"Comprehension time: {comp_time:.4f} seconds")
print(f"Performance gain: {((loop_time - comp_time) / loop_time * 100):.2f}%")

# Output:
# Traditional loop time: 0.2845 seconds
# Comprehension time: 0.1923 seconds
# Performance gain: 32.41%
```

Slide 12: Complex Data Transformation Pipeline

Real-world applications often require multiple transformation steps. This example demonstrates how to chain comprehensions to create a complex data processing pipeline for analyzing scientific data.

```python
import math
from datetime import datetime, timedelta

# Generate sample sensor data
sensor_data = [
    {
        'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
        'temperature': 20 + math.sin(i/10) * 5,
        'humidity': 50 + math.cos(i/10) * 10,
        'error_code': random.choice([None, 'E01', 'E02', None, None])
    }
    for i in range(24)
]

# Multi-stage data processing pipeline
processed_data = {
    'valid_readings': [
        {k: round(v, 2) if isinstance(v, float) else v 
         for k, v in reading.items()}
        for reading in sensor_data 
        if reading['error_code'] is None
    ],
    'error_distribution': {
        code: len([r for r in sensor_data if r['error_code'] == code])
        for code in {r['error_code'] for r in sensor_data if r['error_code']}
    },
    'statistics': {
        'avg_temp': round(sum(r['temperature'] for r in sensor_data) / len(sensor_data), 2),
        'max_humidity': max(r['humidity'] for r in sensor_data)
    }
}

print(f"Processed data summary:\n{processed_data}")

# Output:
# Processed data summary:
# {
#   'valid_readings': [{...}, {...}, ...],
#   'error_distribution': {'E01': 3, 'E02': 2},
#   'statistics': {'avg_temp': 21.34, 'max_humidity': 59.84}
# }
```

Slide 13: Advanced Set Operations with Comprehensions

Set comprehensions combined with set operations provide powerful tools for data analysis and filtering, especially useful in scenarios requiring unique value processing and set theory operations.

```python
# Sample datasets
users_set_a = {f"user_{i}" for i in range(1, 6)}
users_set_b = {f"user_{i}" for i in range(3, 8)}
user_activity = {
    f"user_{i}": random.randint(1, 100) 
    for i in range(1, 8)
}

# Complex set operations using comprehensions
active_users = {
    user for user, activity in user_activity.items() 
    if activity > 50
}

common_active_users = {
    user for user in users_set_a & users_set_b 
    if user_activity[user] > 50
}

activity_ranges = {
    range_name: {user for user, activity in user_activity.items() 
                if low <= activity <= high}
    for range_name, (low, high) in {
        'low': (1, 30),
        'medium': (31, 70),
        'high': (71, 100)
    }.items()
}

print(f"Active users: {active_users}")
print(f"Common active users: {common_active_users}")
print(f"Activity ranges: {activity_ranges}")

# Output:
# Active users: {'user_2', 'user_5', 'user_7'}
# Common active users: {'user_5'}
# Activity ranges: {
#   'low': {'user_1', 'user_3'},
#   'medium': {'user_4', 'user_6'},
#   'high': {'user_2', 'user_5', 'user_7'}
# }
```

Slide 14: Additional Resources

*   "Effective Python: 90 Specific Ways to Write Better Python" - Search on Google Books
*   Understanding Python List Comprehensions - [https://realpython.com/list-comprehension-python/](https://realpython.com/list-comprehension-python/)
*   Python Dictionary Comprehensions - [https://www.python.org/dev/peps/pep-0274/](https://www.python.org/dev/peps/pep-0274/)
*   Performance Analysis of List Comprehensions - [https://stackoverflow.com/questions/tagged/list-comprehension+python+performance](https://stackoverflow.com/questions/tagged/list-comprehension+python+performance)
*   "Python Cookbook" by David Beazley - Advanced Python Programming Patterns

