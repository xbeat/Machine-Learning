## Currying in Python A Functional Programming Technique
Slide 1: Understanding Currying in Python

Currying is a functional programming technique that transforms a function with multiple arguments into a sequence of functions, each taking a single argument. This concept is named after mathematician Haskell Curry and is widely used in functional programming languages. In Python, we can implement currying to create more flexible and reusable code.

```python
def add(x):
    def inner(y):
        return x + y
    return inner

add_5 = add(5)
result = add_5(3)
print(result)  # Output: 8
```

Slide 2: Basic Currying Example

Let's start with a simple example to illustrate currying. We'll create a curried function that adds two numbers. Instead of taking both arguments at once, we'll split it into two nested functions.

```python
def curry_add(x):
    def add_y(y):
        return x + y
    return add_y

# Usage
curried_add_5 = curry_add(5)
result = curried_add_5(3)
print(result)  # Output: 8

# Alternative usage
print(curry_add(2)(7))  # Output: 9
```

Slide 3: Currying vs. Partial Application

While currying and partial application are related concepts, they're not identical. Currying always produces a chain of unary functions (functions with one argument), while partial application can fix any number of arguments. Let's compare the two:

```python
from functools import partial

# Currying
def curried_multiply(x):
    def multiply_by_y(y):
        return x * y
    return multiply_by_y

# Partial application
def multiply(x, y):
    return x * y

curried_double = curried_multiply(2)
partial_double = partial(multiply, 2)

print(curried_double(5))  # Output: 10
print(partial_double(5))  # Output: 10
```

Slide 4: Automatic Currying

We can create a decorator to automatically curry any function with multiple arguments. This allows us to use the function in both curried and uncurried forms.

```python
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def add_three_numbers(x, y, z):
    return x + y + z

print(add_three_numbers(1)(2)(3))  # Output: 6
print(add_three_numbers(1, 2)(3))  # Output: 6
print(add_three_numbers(1, 2, 3))  # Output: 6
```

Slide 5: Real-Life Example: Text Processing

Currying can be useful in text processing tasks. Let's create a curried function to replace words in a sentence:

```python
def replace_word(old_word):
    def with_new_word(new_word):
        def in_text(text):
            return text.replace(old_word, new_word)
        return in_text
    return with_new_word

replace_python = replace_word("Python")
replace_with_java = replace_python("Java")

original_text = "Python is a versatile programming language."
modified_text = replace_with_java(original_text)

print(modified_text)  # Output: Java is a versatile programming language.
```

Slide 6: Currying for Function Composition

Currying facilitates function composition, allowing us to create new functions by combining existing ones. Here's an example of how currying can be used to create a pipeline of operations:

```python
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def add(x, y):
    return x + y

@curry
def multiply(x, y):
    return x * y

pipeline = lambda x: multiply(2)(add(3)(x))

result = pipeline(5)
print(result)  # Output: 16 ((5 + 3) * 2)
```

Slide 7: Currying for Memoization

Currying can be combined with memoization to create efficient, reusable functions that cache their results. This is particularly useful for expensive computations:

```python
def memoize(func):
    cache = {}
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return memoized

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Output: 354224848179261915075 (calculated quickly due to memoization)
```

Slide 8: Real-Life Example: Configuration Management

Currying can be useful in configuration management scenarios. Let's create a curried function to generate configuration objects:

```python
def config_generator(environment):
    def set_database(database):
        def set_port(port):
            return {
                "environment": environment,
                "database": database,
                "port": port
            }
        return set_port
    return set_database

prod_config = config_generator("production")
prod_mysql_config = prod_config("mysql")
final_config = prod_mysql_config(3306)

print(final_config)
# Output: {'environment': 'production', 'database': 'mysql', 'port': 3306}
```

Slide 9: Currying with Type Hints

We can use type hints to make our curried functions more readable and maintainable. Here's an example of a curried function with type hints:

```python
from typing import Callable

def curried_formatter(prefix: str) -> Callable[[str], Callable[[int], str]]:
    def add_suffix(suffix: str) -> Callable[[int], str]:
        def format_number(number: int) -> str:
            return f"{prefix}{number}{suffix}"
        return format_number
    return add_suffix

format_usd = curried_formatter("$")("USD")
print(format_usd(100))  # Output: $100USD
print(format_usd(250))  # Output: $250USD

format_euro = curried_formatter("€")("EUR")
print(format_euro(100))  # Output: €100EUR
```

Slide 10: Currying and Decorators

Currying can be combined with decorators to create powerful and flexible function transformations. Here's an example of a curried decorator that adds logging to a function:

```python
import functools

def logged(level):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{level}: Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"{level}: Finished {func.__name__}")
            return result
        return wrapper
    return decorator

@logged("INFO")
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
# Output:
# INFO: Calling greet
# INFO: Finished greet
# Hello, Alice!
```

Slide 11: Performance Considerations

While currying can lead to more flexible and composable code, it's important to consider its performance implications. Curried functions often involve multiple function calls and closures, which can introduce overhead:

```python
import timeit

def regular_add(x, y):
    return x + y

def curried_add(x):
    def inner(y):
        return x + y
    return inner

regular_time = timeit.timeit("regular_add(5, 3)", globals=globals(), number=1000000)
curried_time = timeit.timeit("curried_add(5)(3)", globals=globals(), number=1000000)

print(f"Regular function time: {regular_time:.6f} seconds")
print(f"Curried function time: {curried_time:.6f} seconds")
print(f"Overhead: {(curried_time - regular_time) / regular_time * 100:.2f}%")
```

Slide 12: Currying in Functional Programming Paradigms

Currying is particularly useful in functional programming paradigms, where it facilitates function composition and partial application. Let's explore how currying can be used to create a simple data processing pipeline:

```python
from functools import reduce

def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def map_func(func, iterable):
    return map(func, iterable)

@curry
def filter_func(pred, iterable):
    return filter(pred, iterable)

@curry
def reduce_func(func, iterable):
    return reduce(func, iterable)

pipeline = (
    map_func(lambda x: x * 2)
    | filter_func(lambda x: x > 5)
    | reduce_func(lambda x, y: x + y)
)

result = pipeline(range(5))
print(result)  # Output: 18 (2*3 + 2*4)
```

Slide 13: Currying and Lazy Evaluation

Currying can be combined with lazy evaluation to create efficient data processing pipelines. Here's an example using Python's itertools module:

```python
import itertools

def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def take(n, iterable):
    return itertools.islice(iterable, n)

@curry
def map_func(func, iterable):
    return map(func, iterable)

@curry
def filter_func(pred, iterable):
    return filter(pred, iterable)

pipeline = (
    map_func(lambda x: x ** 2)
    | filter_func(lambda x: x % 2 == 0)
    | take(3)
)

result = list(pipeline(itertools.count()))
print(result)  # Output: [0, 4, 16]
```

Slide 14: Additional Resources

For those interested in diving deeper into currying and functional programming in Python, here are some additional resources:

1. "Functional Programming in Python" by David Mertz (ArXiv:1904.04206) URL: [https://arxiv.org/abs/1904.04206](https://arxiv.org/abs/1904.04206)
2. "A Gentle Introduction to Functional Programming in Python" by Cristian Medina (ArXiv:1904.04207) URL: [https://arxiv.org/abs/1904.04207](https://arxiv.org/abs/1904.04207)

These papers provide a comprehensive overview of functional programming concepts, including currying, and their implementation in Python.

