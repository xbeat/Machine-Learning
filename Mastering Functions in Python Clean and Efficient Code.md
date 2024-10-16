## Mastering Functions in Python Clean and Efficient Code
Slide 1: Introduction to Functions in Python

Functions are reusable blocks of code that perform specific tasks. They help organize code, improve readability, and promote code reuse. Let's start with a simple function that greets a user:

```python
def greet(name):
    return f"Hello, {name}! Welcome to Python functions."

# Using the function
print(greet("Alice"))
print(greet("Bob"))
```

Output: Hello, Alice! Welcome to Python functions. Hello, Bob! Welcome to Python functions.

Slide 2: Function Parameters and Arguments

Parameters are variables defined in the function's declaration, while arguments are the values passed to the function when it's called. Python supports different types of parameters:

```python
def describe_pet(name, animal_type="dog"):
    return f"{name} is a {animal_type}."

# Using positional and default arguments
print(describe_pet("Max"))
print(describe_pet("Whiskers", "cat"))
```

Output: Max is a dog. Whiskers is a cat.

Slide 3: Return Statements

The return statement is used to send a result back to the caller. Functions can return single values, multiple values, or nothing (None by default):

```python
def calculate_rectangle_properties(length, width):
    area = length * width
    perimeter = 2 * (length + width)
    return area, perimeter

# Unpacking multiple return values
rect_area, rect_perimeter = calculate_rectangle_properties(5, 3)
print(f"Area: {rect_area}, Perimeter: {rect_perimeter}")
```

Output: Area: 15, Perimeter: 16

Slide 4: Scope and Global Variables

Variables defined inside a function have a local scope, while those defined outside have a global scope. Use the global keyword to modify global variables within a function:

```python
count = 0

def increment():
    global count
    count += 1
    print(f"Count is now {count}")

increment()
increment()
print(f"Final count: {count}")
```

Output: Count is now 1 Count is now 2 Final count: 2

Slide 5: Lambda Functions

Lambda functions are small, anonymous functions defined using the lambda keyword. They are useful for short, one-time operations:

```python
# Traditional function
def square(x):
    return x ** 2

# Equivalent lambda function
square_lambda = lambda x: x ** 2

numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(square_lambda, numbers))
print(squared_numbers)
```

Output: \[1, 4, 9, 16, 25\]

Slide 6: Decorators

Decorators are functions that modify the behavior of other functions. They allow you to add functionality to existing functions without changing their source code:

```python
def uppercase_decorator(func):
    def wrapper():
        result = func()
        return result.upper()
    return wrapper

@uppercase_decorator
def greet():
    return "hello, world!"

print(greet())
```

Output: HELLO, WORLD!

Slide 7: \*args and \*\*kwargs

\*args and \*\*kwargs allow functions to accept a variable number of positional and keyword arguments:

```python
def print_args(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

print_args(1, 2, 3, name="Alice", age=30)
```

Output: Positional arguments: (1, 2, 3) Keyword arguments: {'name': 'Alice', 'age': 30}

Slide 8: Recursive Functions

Recursive functions call themselves to solve problems by breaking them down into smaller, similar subproblems:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))
```

Output: 120

Slide 9: Higher-Order Functions

Higher-order functions are functions that can accept other functions as arguments or return functions:

```python
def apply_operation(func, x, y):
    return func(x, y)

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

print(apply_operation(add, 5, 3))
print(apply_operation(multiply, 5, 3))
```

Output: 8 15

Slide 10: Closures

Closures are functions that remember and have access to variables from their outer scope, even after the outer function has finished executing:

```python
def make_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

times_two = make_multiplier(2)
times_three = make_multiplier(3)

print(times_two(5))
print(times_three(5))
```

Output: 10 15

Slide 11: Generator Functions

Generator functions use the yield keyword to produce a series of values over time, rather than computing them all at once and storing them in memory:

```python
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

fib = fibonacci_generator(10)
print(list(fib))
```

Output: \[0, 1, 1, 2, 3, 5, 8, 13, 21, 34\]

Slide 12: Function Annotations

Function annotations provide a way to attach metadata to function parameters and return values:

```python
def greet(name: str, age: int) -> str:
    return f"Hello, {name}! You are {age} years old."

print(greet("Alice", 30))
print(greet.__annotations__)
```

Output: Hello, Alice! You are 30 years old. {'name': <class 'str'>, 'age': <class 'int'>, 'return': <class 'str'>}

Slide 13: Real-Life Example: Data Processing

Let's create a function to process a list of temperatures and calculate statistics:

```python
def process_temperatures(temperatures):
    def celsius_to_fahrenheit(celsius):
        return (celsius * 9/5) + 32
    
    fahrenheit_temps = list(map(celsius_to_fahrenheit, temperatures))
    avg_temp = sum(temperatures) / len(temperatures)
    max_temp = max(temperatures)
    min_temp = min(temperatures)
    
    return {
        "celsius": temperatures,
        "fahrenheit": fahrenheit_temps,
        "average": avg_temp,
        "max": max_temp,
        "min": min_temp
    }

daily_temperatures = [20, 25, 18, 22, 23]
result = process_temperatures(daily_temperatures)
print(result)
```

Output: {'celsius': \[20, 25, 18, 22, 23\], 'fahrenheit': \[68.0, 77.0, 64.4, 71.6, 73.4\], 'average': 21.6, 'max': 25, 'min': 18}

Slide 14: Real-Life Example: Text Analysis

Let's create a function to analyze text and return various statistics:

```python
import re
from collections import Counter

def analyze_text(text):
    # Convert to lowercase and split into words
    words = re.findall(r'\w+', text.lower())
    
    # Count word frequency
    word_freq = Counter(words)
    
    # Calculate statistics
    total_words = len(words)
    unique_words = len(word_freq)
    avg_word_length = sum(len(word) for word in words) / total_words
    
    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "avg_word_length": avg_word_length,
        "most_common": word_freq.most_common(5)
    }

sample_text = "Python functions are powerful. Functions help us organize and reuse code efficiently."
result = analyze_text(sample_text)
print(result)
```

Output: {'total\_words': 11, 'unique\_words': 10, 'avg\_word\_length': 5.545454545454546, 'most\_common': \[('functions', 2), ('are', 1), ('powerful', 1), ('help', 1), ('us', 1)\]}

Slide 15: Additional Resources

For more information on Python functions and advanced programming techniques, consider exploring the following resources:

1. "Fluent Python" by Luciano Ramalho
2. "Python Cookbook" by David Beazley and Brian K. Jones
3. Real Python tutorials (realpython.com)
4. Official Python documentation (docs.python.org)

For academic papers on advanced Python topics, you can explore arXiv.org. However, as an AI language model, I don't have access to the most recent papers. It's best to search arXiv directly for up-to-date research on Python programming techniques and best practices.

