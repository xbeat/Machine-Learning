## Partial Functions in Python for Code Modularity
Slide 1: Introduction to Partial Functions in Python

Partial functions in Python allow us to create new functions by fixing a subset of arguments of an existing function. This technique enhances code modularity and reusability.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# Creating a partial function
square = partial(power, exponent=2)

print(square(4))  # Output: 16
print(square(5))  # Output: 25
```

Slide 2: The functools.partial Function

The `functools.partial` function is the key to creating partial functions in Python. It takes a function and some arguments, returning a new function with those arguments pre-set.

```python
from functools import partial

def greet(greeting, name):
    return f"{greeting}, {name}!"

# Creating partial functions
say_hello = partial(greet, "Hello")
say_hi = partial(greet, "Hi")

print(say_hello("Alice"))  # Output: Hello, Alice!
print(say_hi("Bob"))       # Output: Hi, Bob!
```

Slide 3: Partial Functions and Default Arguments

Partial functions differ from default arguments. While default arguments are set at function definition, partial functions create new function objects with pre-set arguments.

```python
def multiply(a, b=2):
    return a * b

double = partial(multiply, b=2)

print(multiply(3))    # Output: 6 (using default argument)
print(double(3))      # Output: 6 (using partial function)
print(multiply(3, 4)) # Output: 12 (overriding default argument)
print(double(3, 4))   # Output: 12 (overriding partial function)
```

Slide 4: Partial Functions with Positional Arguments

Partial functions can also be created with positional arguments. The new function will have fewer required arguments than the original.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# Creating a partial function with a positional argument
cube = partial(power, 3)

print(cube(2))  # Output: 8 (3^2)
print(cube(3))  # Output: 27 (3^3)
```

Slide 5: Partial Functions in Event Handlers

Partial functions are particularly useful in event-driven programming, allowing us to pass additional arguments to callback functions.

```python
import tkinter as tk
from functools import partial

def on_button_click(message, event):
    print(f"Button clicked! Message: {message}")

root = tk.Tk()
button = tk.Button(root, text="Click me!")
button.bind("<Button-1>", partial(on_button_click, "Hello, World!"))
button.pack()
root.mainloop()
```

Slide 6: Partial Functions for Configuration

Partial functions can be used to create pre-configured versions of functions, enhancing code modularity.

```python
from functools import partial

def connect_to_database(host, port, user, password):
    # Simulating database connection
    return f"Connected to {host}:{port} as {user}"

# Pre-configured connection function
connect_to_prod = partial(connect_to_database, 
                          host="prod.example.com", 
                          port=5432, 
                          user="admin")

print(connect_to_prod(password="secret"))
# Output: Connected to prod.example.com:5432 as admin
```

Slide 7: Partial Functions in Functional Programming

Partial functions play a crucial role in functional programming paradigms, enabling function composition and currying.

```python
from functools import partial

def compose(f, g):
    return lambda x: f(g(x))

def add(a, b):
    return a + b

increment = partial(add, 1)
double = lambda x: x * 2

increment_and_double = compose(double, increment)

print(increment_and_double(3))  # Output: 8 ((3 + 1) * 2)
```

Slide 8: Partial Functions for Parameter Binding

Partial functions can bind parameters to create more specific versions of general functions.

```python
from functools import partial

def filter_by_attribute(items, attr, value):
    return [item for item in items if getattr(item, attr) == value]

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person("Alice", 30), Person("Bob", 25), Person("Charlie", 30)]

filter_by_age = partial(filter_by_attribute, attr="age")
thirty_year_olds = filter_by_age(people, value=30)

for person in thirty_year_olds:
    print(person.name)  # Output: Alice, Charlie
```

Slide 9: Partial Functions in Decorators

Partial functions can be used to create flexible decorators that accept arguments.

```python
from functools import partial, wraps

def retry(max_attempts, exceptions):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator

# Creating a partial function for a specific retry decorator
retry_network_errors = partial(retry, exceptions=(ConnectionError, TimeoutError))

@retry_network_errors(max_attempts=3)
def fetch_data():
    # Simulating a network operation
    import random
    if random.random() < 0.5:
        raise ConnectionError("Network error occurred")
    return "Data fetched successfully"

print(fetch_data())  # May print: Data fetched successfully
```

Slide 10: Partial Functions for Memoization

Partial functions can be used to implement memoization, a technique to cache expensive function calls.

```python
from functools import partial, lru_cache

def memoize(func):
    return lru_cache(maxsize=None)(func)

@memoize
def expensive_computation(a, b):
    print(f"Computing {a} + {b}")
    return a + b

# Create partial functions for specific computations
compute_with_5 = partial(expensive_computation, 5)
compute_with_10 = partial(expensive_computation, 10)

print(compute_with_5(3))  # Output: Computing 5 + 3 \n 8
print(compute_with_5(3))  # Output: 8 (cached result)
print(compute_with_10(7))  # Output: Computing 10 + 7 \n 17
```

Slide 11: Partial Functions in Testing

Partial functions can simplify test setup by creating pre-configured test functions.

```python
from functools import partial
import unittest

def validate_user(name, age, email):
    if len(name) < 2:
        raise ValueError("Name too short")
    if age < 18:
        raise ValueError("User too young")
    if "@" not in email:
        raise ValueError("Invalid email")
    return True

class TestUserValidation(unittest.TestCase):
    def setUp(self):
        self.validate = partial(validate_user, name="John", age=25)

    def test_valid_user(self):
        self.assertTrue(self.validate(email="john@example.com"))

    def test_invalid_email(self):
        with self.assertRaises(ValueError):
            self.validate(email="invalid-email")

if __name__ == "__main__":
    unittest.main()
```

Slide 12: Real-Life Example: Image Processing

Partial functions can be used in image processing to create reusable filter functions.

```python
from functools import partial
from PIL import Image, ImageEnhance

def adjust_image(image, brightness=1.0, contrast=1.0, color=1.0):
    img = Image.open(image)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    return img

# Create partial functions for specific adjustments
increase_brightness = partial(adjust_image, brightness=1.5)
increase_contrast = partial(adjust_image, contrast=1.5)
grayscale = partial(adjust_image, color=0)

# Apply filters
original_image = "path/to/image.jpg"
bright_image = increase_brightness(original_image)
high_contrast_image = increase_contrast(original_image)
gray_image = grayscale(original_image)

bright_image.save("bright_image.jpg")
high_contrast_image.save("high_contrast_image.jpg")
gray_image.save("gray_image.jpg")
```

Slide 13: Real-Life Example: Custom Sorting

Partial functions can be used to create custom sorting functions for complex data structures.

```python
from functools import partial

class Product:
    def __init__(self, name, price, rating):
        self.name = name
        self.price = price
        self.rating = rating

    def __repr__(self):
        return f"Product({self.name}, ${self.price}, {self.rating}★)"

def sort_products(products, key_func, reverse=False):
    return sorted(products, key=key_func, reverse=reverse)

# Create partial functions for different sorting criteria
sort_by_price = partial(sort_products, key_func=lambda p: p.price)
sort_by_rating = partial(sort_products, key_func=lambda p: p.rating, reverse=True)

products = [
    Product("Laptop", 1200, 4.5),
    Product("Phone", 800, 4.2),
    Product("Tablet", 500, 4.0),
    Product("Smartwatch", 300, 4.3)
]

print("Sorted by price (low to high):")
print(sort_by_price(products))

print("\nSorted by rating (high to low):")
print(sort_by_rating(products))
```

Slide 14: Additional Resources

For further exploration of partial functions and functional programming in Python:

1. "Functional Programming in Python" by David Mertz (O'Reilly)
2. "Python Cookbook" by David Beazley and Brian K. Jones (O'Reilly)
3. Official Python documentation on functools: [https://docs.python.org/3/library/functools.html](https://docs.python.org/3/library/functools.html)
4. "Higher-order functions and operations on callable objects" (PEP 309): [https://www.python.org/dev/peps/pep-0309/](https://www.python.org/dev/peps/pep-0309/)

