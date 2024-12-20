## Python Generators and Iterators with yield
Slide 1: Introduction to Python Generators and Iterators

Generators and iterators are powerful features in Python that allow for efficient handling of large datasets and creation of custom sequences. They provide a way to generate values on-the-fly, saving memory and improving performance. This presentation will explore these concepts, their implementation, and practical applications.

```python
# Simple generator function
def countdown(n):
    while n > 0:
        yield n
        n -= 1

# Using the generator
for num in countdown(5):
    print(num)
```

Slide 2: What are Iterators?

Iterators are objects that implement the iterator protocol, consisting of the **iter**() and **next**() methods. They allow you to traverse through a sequence of elements, one at a time, without loading the entire sequence into memory. Iterators are the foundation for many Python features, including for loops and list comprehensions.

```python
class CountDown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        current = self.start
        self.start -= 1
        return current

# Using the iterator
for num in CountDown(5):
    print(num)
```

Slide 3: Understanding Generators

Generators are a special type of iterator that are defined using functions with the 'yield' keyword. They allow you to generate a sequence of values over time, rather than computing them all at once and storing them in memory. Generators are memory-efficient and can be used to represent infinite sequences.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Using the generator
fib = fibonacci()
for _ in range(10):
    print(next(fib))
```

Slide 4: The 'yield' Keyword

The 'yield' keyword is used in generator functions to define points where the function should pause and yield a value. When the generator function is called, it returns a generator object without executing the function body. The function's state is saved and resumed on subsequent calls to next().

```python
def square_numbers(n):
    for i in range(n):
        yield i ** 2

# Using the generator
squares = square_numbers(5)
print(next(squares))  # 0
print(next(squares))  # 1
print(next(squares))  # 4
```

Slide 5: Generator Expressions

Generator expressions are a concise way to create generators using a syntax similar to list comprehensions. They are memory-efficient alternatives to list comprehensions when you don't need to store all the generated values at once.

```python
# List comprehension
squares_list = [x**2 for x in range(10)]

# Generator expression
squares_gen = (x**2 for x in range(10))

print(squares_list)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares_gen)   # <generator object <genexpr> at 0x...>
```

Slide 6: Infinite Sequences with Generators

Generators are particularly useful for creating infinite sequences, as they generate values on-demand without storing the entire sequence in memory. This allows for efficient handling of potentially infinite data streams.

```python
def primes():
    yield 2
    primes_list = [2]
    num = 3
    while True:
        if all(num % p != 0 for p in primes_list):
            primes_list.append(num)
            yield num
        num += 2

prime_gen = primes()
for _ in range(10):
    print(next(prime_gen))
```

Slide 7: Combining Generators

Generators can be combined using various techniques to create more complex data processing pipelines. This allows for efficient and modular data manipulation.

```python
def numbers():
    for i in range(1, 11):
        yield i

def squared(gen):
    for num in gen:
        yield num ** 2

def even_numbers(gen):
    for num in gen:
        if num % 2 == 0:
            yield num

pipeline = even_numbers(squared(numbers()))
print(list(pipeline))  # [4, 16, 36, 64, 100]
```

Slide 8: Real-life Example: Processing Large Files

Generators are excellent for processing large files, as they allow you to read and process the file line by line without loading the entire file into memory.

```python
def process_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            # Process each line
            yield line.strip().upper()

# Usage
for processed_line in process_large_file('large_file.txt'):
    print(processed_line)
```

Slide 9: Real-life Example: Pagination

Generators can be used to implement efficient pagination for large datasets, allowing you to retrieve data in chunks without loading the entire dataset into memory.

```python
def paginate(data, page_size):
    for i in range(0, len(data), page_size):
        yield data[i:i + page_size]

# Sample data
items = list(range(1, 101))

# Usage
for page in paginate(items, 10):
    print(f"Page: {page}")
```

Slide 10: Sending Values to Generators

Generators can receive values using the send() method, allowing for two-way communication between the generator and the caller. This feature enables the creation of coroutines and more complex generator-based workflows.

```python
def echo_generator():
    while True:
        value = yield
        yield value

echo = echo_generator()
next(echo)  # Prime the generator
print(echo.send("Hello"))  # Hello
print(echo.send("World"))  # World
```

Slide 11: Generator Delegation with 'yield from'

The 'yield from' statement allows one generator to delegate part of its operations to another generator. This enables the creation of more complex generator hierarchies and supports better code organization.

```python
def subgenerator():
    yield 1
    yield 2
    yield 3

def main_generator():
    yield 'A'
    yield from subgenerator()
    yield 'B'

for item in main_generator():
    print(item)
```

Slide 12: Exception Handling in Generators

Generators can handle exceptions using try-except blocks, allowing for graceful error handling and cleanup operations.

```python
def div_generator(a, b):
    try:
        yield a / b
    except ZeroDivisionError:
        yield "Cannot divide by zero"

for result in div_generator(10, 2):
    print(result)  # 5.0

for result in div_generator(10, 0):
    print(result)  # Cannot divide by zero
```

Slide 13: Asynchronous Generators

Python 3.6 introduced asynchronous generators, which combine the power of generators with asynchronous programming. They are defined using 'async def' and 'yield', and are used with 'async for' loops.

```python
import asyncio

async def async_range(start, stop):
    for i in range(start, stop):
        await asyncio.sleep(0.1)
        yield i

async def main():
    async for num in async_range(0, 5):
        print(num)

asyncio.run(main())
```

Slide 14: Performance Comparison: Generators vs Lists

Generators often provide better performance and memory usage compared to lists, especially when dealing with large datasets. Here's a simple comparison:

```python
import sys

# List
def get_squares_list(n):
    return [i**2 for i in range(n)]

# Generator
def get_squares_gen(n):
    for i in range(n):
        yield i**2

n = 1000000
squares_list = get_squares_list(n)
squares_gen = get_squares_gen(n)

print(f"List size: {sys.getsizeof(squares_list)} bytes")
print(f"Generator size: {sys.getsizeof(squares_gen)} bytes")
```

Slide 15: Additional Resources

For more information on Python generators and iterators, consider exploring the following resources:

1. Python Documentation: [https://docs.python.org/3/howto/functional.html#generators](https://docs.python.org/3/howto/functional.html#generators)
2. Real Python Tutorial: [https://realpython.com/introduction-to-python-generators/](https://realpython.com/introduction-to-python-generators/)
3. Python Cookbook by David Beazley and Brian K. Jones (O'Reilly Media)
4. Fluent Python by Luciano Ramalho (O'Reilly Media)

These resources provide in-depth explanations, advanced techniques, and best practices for working with generators and iterators in Python.

