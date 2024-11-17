## Essential Python Data Types
Slide 1: Python Numbers - Core Numeric Types

Numbers in Python are fundamental data types representing mathematical values. The language implements three distinct numeric types: integers for whole numbers, floating-point numbers for decimals, and complex numbers for mathematical operations involving imaginary components.

```python
# Integer examples with binary, octal and hexadecimal
decimal_num = 255
binary_num = 0b11111111  # Binary (base 2)
octal_num = 0o377       # Octal (base 8)
hex_num = 0xFF         # Hexadecimal (base 16)

# Floating-point examples with scientific notation
float_num = 3.14159
scientific = 2.5e-3    # 0.0025

# Complex number operations
z1 = 3 + 4j
z2 = complex(1, 2)
magnitude = abs(z1)    # sqrt(3² + 4²) = 5.0

print(f"Different number representations of 255: {decimal_num}")
print(f"Binary: {binary_num}, Octal: {octal_num}, Hex: {hex_num}")
print(f"Complex magnitude: {magnitude}")
```

Slide 2: Advanced Boolean Operations

Boolean algebra in Python extends beyond simple True/False values, implementing short-circuit evaluation and bitwise operations. Understanding these concepts is crucial for optimization and low-level programming tasks.

```python
# Short-circuit evaluation demonstration
def expensive_operation():
    print("Expensive operation executed")
    return False

# Short-circuit OR (expensive_operation won't execute)
result1 = True or expensive_operation()
print(f"Result 1: {result1}")

# Bitwise operations on integers
a, b = 60, 13
print(f"a = {bin(a)}, b = {bin(b)}")
print(f"AND: {bin(a & b)}")
print(f"OR:  {bin(a | b)}")
print(f"XOR: {bin(a ^ b)}")
print(f"NOT: {bin(~a)}")
```

Slide 3: Advanced List Comprehensions

List comprehensions provide a powerful and expressive way to create and transform lists. They can incorporate multiple conditions, nested loops, and conditional expressions, offering superior performance compared to traditional loops.

```python
# Advanced list comprehension examples
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Flatten matrix using nested comprehension
flattened = [num for row in matrix for num in row]

# Conditional comprehension with multiple conditions
evens_squared = [x**2 for x in range(20) if x % 2 == 0 if x > 4]

# Nested comprehension creating multiplication table
mult_table = [[i*j for j in range(1, 6)] for i in range(1, 6)]

print(f"Flattened matrix: {flattened}")
print(f"Evens squared: {evens_squared}")
print("Multiplication table:")
for row in mult_table:
    print(row)
```

Slide 4: Advanced Tuple Operations

Tuples in Python offer immutable sequences with unique properties and optimizations. Their immutability enables their use as dictionary keys and provides performance benefits in certain scenarios through internal optimizations.

```python
# Advanced tuple operations and patterns
from collections import namedtuple

# Named tuples for structured data
Point = namedtuple('Point', ['x', 'y', 'z'])
p1 = Point(1, 2, 3)

# Tuple unpacking with asterisk
first, *middle, last = (1, 2, 3, 4, 5)

# Tuple as dictionary key
tuple_dict = {(1, 2): 'value'}

# Multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

numbers = [1, 2, 3, 4, 5]
minimum, maximum, average = get_stats(numbers)

print(f"Named tuple: {p1}")
print(f"Middle values: {middle}")
print(f"Stats - Min: {minimum}, Max: {maximum}, Avg: {average}")
```

Slide 5: Dictionary Advanced Features

Modern Python dictionaries maintain insertion order and provide specialized methods for advanced operations. Understanding these features enables efficient data manipulation and improved memory usage in complex applications.

```python
# Advanced dictionary features
from collections import defaultdict, ChainMap

# Dictionary merge and update patterns
dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged = {**dict1, **dict2}  # Dictionary unpacking

# Default dictionaries for automatic initialization
word_count = defaultdict(int)
words = "the quick brown fox jumps over the lazy dog".split()
for word in words:
    word_count[word] += 1

# Dictionary views and set operations
keys1 = dict1.keys()
keys2 = dict2.keys()
common_keys = keys1 & keys2

print(f"Merged dict: {merged}")
print(f"Word count: {dict(word_count)}")
print(f"Common keys: {common_keys}")
```

Slide 6: Advanced Set Operations

Sets provide powerful mathematical operations for working with unique collections. Understanding set algebra and its implementation in Python allows for efficient data deduplication and comparison operations in real-world applications.

```python
# Advanced set operations and performance optimization
# Creating sets with comprehension
squares = {x**2 for x in range(10)}
cubes = {x**3 for x in range(10)}

# Advanced set operations
intersection = squares & cubes
symmetric_diff = squares ^ cubes
relative_complement = squares - cubes

# Frozen sets for immutable set operations
immutable_set = frozenset([1, 2, 3, 4])
set_of_frozensets = {frozenset([1, 2]), frozenset([3, 4])}

print(f"Squares: {squares}")
print(f"Cubes: {cubes}")
print(f"Intersection: {intersection}")
print(f"Symmetric Difference: {symmetric_diff}")
print(f"Set of Frozensets: {set_of_frozensets}")
```

Slide 7: Advanced String Manipulation

String manipulation in Python extends beyond basic operations, incorporating advanced formatting, encoding handling, and regular expressions for complex text processing tasks that are common in data cleaning and analysis.

```python
import re
from string import Template

# Advanced string formatting
data = {'name': 'John', 'age': 30}
template = Template('Name: $name, Age: $age')
formatted = template.substitute(data)

# Complex string operations with regex
text = "Python3.9 was released in 2020, Python3.10 in 2021"
pattern = r'Python(\d+\.\d+)'
versions = re.findall(pattern, text)

# Unicode and encoding handling
unicode_text = "Hello, 世界"
encoded = unicode_text.encode('utf-8')
decoded = encoded.decode('utf-8')

print(f"Template formatting: {formatted}")
print(f"Found Python versions: {versions}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

Slide 8: Mathematical Operations Implementation

Python provides robust support for mathematical operations through its built-in numeric types and the math module. This implementation demonstrates advanced mathematical concepts and their practical applications.

```python
import math
import cmath  # For complex number operations

# Custom implementation of mathematical functions
def newton_sqrt(n, precision=1e-10):
    x = n / 2  # Initial guess
    while True:
        root = 0.5 * (x + n / x)
        if abs(root - x) < precision:
            return root
        x = root

# Complex number operations
def complex_roots(a, b, c):
    discriminant = cmath.sqrt(b**2 - 4*a*c)
    root1 = (-b + discriminant)/(2*a)
    root2 = (-b - discriminant)/(2*a)
    return root1, root2

# Example usage
number = 16
custom_sqrt = newton_sqrt(number)
math_sqrt = math.sqrt(number)
roots = complex_roots(1, -5, 6)  # x² - 5x + 6

print(f"Custom sqrt(16): {custom_sqrt}")
print(f"Math module sqrt(16): {math_sqrt}")
print(f"Quadratic roots: {roots}")
```

Slide 9: Advanced Iteration Patterns

Modern Python offers sophisticated iteration patterns that go beyond basic loops. Understanding these patterns enables writing more efficient and expressive code for complex data processing tasks.

```python
from itertools import islice, chain, combinations

# Custom iterator class
class FibonacciIterator:
    def __init__(self, limit):
        self.limit = limit
        self.previous = 0
        self.current = 1
        self.count = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.count >= self.limit:
            raise StopIteration
            
        result = self.previous
        self.previous, self.current = (
            self.current,
            self.previous + self.current
        )
        self.count += 1
        return result

# Using various iteration patterns
numbers = range(10)
fibonacci = FibonacciIterator(8)

# Combination of iterators
combined = chain(numbers, fibonacci)
windowed = list(islice(combined, 5))
combos = list(combinations(range(4), 2))

print(f"Windowed iteration: {windowed}")
print(f"Combinations: {combos}")
print(f"Fibonacci sequence: {list(fibonacci)}")
```

Slide 10: Real-world Application - Data Analysis

A practical implementation of data analysis using Python's core data types. This example demonstrates data preprocessing, statistical analysis, and result visualization using only built-in features.

```python
# Sample dataset: Sales data analysis
sales_data = [
    {'date': '2024-01', 'product': 'A', 'revenue': 1200},
    {'date': '2024-01', 'product': 'B', 'revenue': 850},
    {'date': '2024-02', 'product': 'A', 'revenue': 1400},
    {'date': '2024-02', 'product': 'B', 'revenue': 950}
]

# Data processing and analysis
def analyze_sales(data):
    # Group by product
    product_sales = {}
    for entry in data:
        product = entry['product']
        revenue = entry['revenue']
        if product not in product_sales:
            product_sales[product] = []
        product_sales[product].append(revenue)
    
    # Calculate statistics
    statistics = {}
    for product, revenues in product_sales.items():
        statistics[product] = {
            'total': sum(revenues),
            'average': sum(revenues) / len(revenues),
            'growth': (revenues[-1] - revenues[0]) / revenues[0] * 100
        }
    
    return statistics

results = analyze_sales(sales_data)
for product, stats in results.items():
    print(f"\nProduct {product} Analysis:")
    print(f"Total Revenue: ${stats['total']:,.2f}")
    print(f"Average Revenue: ${stats['average']:,.2f}")
    print(f"Growth Rate: {stats['growth']:,.1f}%")
```

Slide 11: Real-world Application - Text Processing Engine

This implementation showcases a practical text processing engine utilizing Python's core data types for document analysis and feature extraction in natural language processing tasks.

```python
from collections import Counter
import re

class TextProcessor:
    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words) if stop_words else set()
        
    def preprocess(self, text):
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove stop words and short words
        return [w for w in words if w not in self.stop_words and len(w) > 2]
    
    def extract_features(self, text):
        words = self.preprocess(text)
        return {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'word_frequency': Counter(words).most_common(5),
            'average_word_length': sum(len(w) for w in words) / len(words)
        }

# Example usage
processor = TextProcessor(stop_words={'the', 'and', 'is', 'in', 'it'})
sample_text = """
Python is a versatile programming language that emphasizes code readability. 
It supports multiple programming paradigms and has a comprehensive standard library.
"""

features = processor.extract_features(sample_text)
for feature, value in features.items():
    print(f"\n{feature.replace('_', ' ').title()}:")
    print(f"{value}")
```

Slide 12: Advanced Dictionary Applications

Dictionary implementations in Python can be extended to create sophisticated data structures for complex applications. This example demonstrates creating a persistent cache with timeout functionality.

```python
from time import time
from typing import Any, Optional

class TimedCache:
    def __init__(self, default_timeout: int = 3600):
        self._cache: dict = {}
        self._timeouts: dict = {}
        self._default_timeout = default_timeout
    
    def set(self, key: str, value: Any, timeout: Optional[int] = None) -> None:
        self._cache[key] = value
        self._timeouts[key] = time() + (timeout or self._default_timeout)
    
    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._cache:
            return default
            
        if time() > self._timeouts[key]:
            del self._cache[key]
            del self._timeouts[key]
            return default
            
        return self._cache[key]
    
    def clear_expired(self) -> int:
        current_time = time()
        expired = [
            key for key, timeout in self._timeouts.items()
            if current_time > timeout
        ]
        
        for key in expired:
            del self._cache[key]
            del self._timeouts[key]
            
        return len(expired)

# Example usage
cache = TimedCache(default_timeout=5)
cache.set('short_lived', 'This expires quickly', timeout=2)
cache.set('long_lived', 'This stays longer', timeout=10)

print(f"Initial cache entries:")
print(f"Short lived: {cache.get('short_lived')}")
print(f"Long lived: {cache.get('long_lived')}")

import time
time.sleep(3)  # Wait for short_lived to expire

print(f"\nAfter 3 seconds:")
print(f"Short lived: {cache.get('short_lived')}")
print(f"Long lived: {cache.get('long_lived')}")
```

Slide 13: Mathematical Optimization Implementation

This implementation demonstrates advanced mathematical concepts using Python's core data types, showcasing numerical optimization techniques commonly used in scientific computing.

```python
def gradient_descent(f, df, start, learning_rate=0.1, tolerance=1e-6, max_iterations=1000):
    """
    Generic gradient descent implementation
    f: objective function
    df: derivative of objective function
    """
    x = start
    history = [(x, f(x))]
    
    for i in range(max_iterations):
        gradient = df(x)
        new_x = x - learning_rate * gradient
        
        # Check convergence
        if abs(new_x - x) < tolerance:
            break
            
        x = new_x
        history.append((x, f(x)))
    
    return x, history

# Example: Find minimum of x² + 2x + 1
def objective(x):
    return x**2 + 2*x + 1

def derivative(x):
    return 2*x + 2

# Run optimization
x_min, history = gradient_descent(objective, derivative, start=5.0)

print(f"Minimum found at x = {x_min:.6f}")
print(f"Minimum value = {objective(x_min):.6f}")
print("\nOptimization path:")
for i, (x, fx) in enumerate(history[:5]):  # Show first 5 iterations
    print(f"Iteration {i}: x = {x:.6f}, f(x) = {fx:.6f}")
```

Slide 14: Additional Resources

*   Advanced Python Programming:
    *   [https://arxiv.org/abs/2207.09273](https://arxiv.org/abs/2207.09273)
    *   [https://arxiv.org/abs/1907.05611](https://arxiv.org/abs/1907.05611)
*   Scientific Computing with Python:
    *   [https://www.python.org/doc/essays/numerical](https://www.python.org/doc/essays/numerical)
    *   [https://docs.scipy.org/doc/scipy/reference/tutorial](https://docs.scipy.org/doc/scipy/reference/tutorial)
*   Data Structures and Algorithms:
    *   [https://www.geeksforgeeks.org/python-programming-language](https://www.geeksforgeeks.org/python-programming-language)
    *   [https://realpython.com/python-data-structures](https://realpython.com/python-data-structures)
*   Machine Learning with Python:
    *   [https://scikit-learn.org/stable/tutorial](https://scikit-learn.org/stable/tutorial)
    *   [https://pytorch.org/tutorials](https://pytorch.org/tutorials)

