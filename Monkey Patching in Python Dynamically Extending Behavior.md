## Monkey Patching in Python Dynamically Extending Behavior
Slide 1: Understanding Monkey Patching in Python

Monkey patching is a dynamic technique to modify or extend the behavior of a program at runtime. In Python, it allows you to change or add attributes and methods to existing classes or modules without altering their source code. This powerful feature can be both useful and dangerous, depending on how it's applied.

```python
# Original class
class Calculator:
    def add(self, a, b):
        return a + b

# Monkey patching to add a new method
def multiply(self, a, b):
    return a * b

Calculator.multiply = multiply

calc = Calculator()
print(calc.add(2, 3))      # Output: 5
print(calc.multiply(2, 3)) # Output: 6
```

Slide 2: How Monkey Patching Works

Monkey patching exploits Python's dynamic nature. Objects in Python are mutable, allowing us to modify their attributes and methods at runtime. When we patch a class or module, we're essentially modifying its namespace, adding or replacing existing elements.

```python
import math

# Original function
print(math.pi)  # Output: 3.141592653589793

# Monkey patching math.pi
math.pi = 3.14
print(math.pi)  # Output: 3.14

# Restoring the original value
import importlib
importlib.reload(math)
print(math.pi)  # Output: 3.141592653589793
```

Slide 3: Adding New Methods

One common use of monkey patching is to add new methods to existing classes. This can be particularly useful when working with third-party libraries or built-in types that you can't modify directly.

```python
# Adding a new method to the built-in list type
def sum_positive(self):
    return sum(x for x in self if x > 0)

list.sum_positive = sum_positive

numbers = [1, -2, 3, -4, 5]
print(numbers.sum_positive())  # Output: 9
```

Slide 4: Modifying Existing Methods

Monkey patching can also be used to modify the behavior of existing methods. This technique is often employed for debugging, testing, or adapting third-party code to specific needs.

```python
# Original function
def greet(name):
    return f"Hello, {name}!"

# Store the original function
original_greet = greet

# Modify the function
def new_greet(name):
    return f"Greetings, {name}! How are you today?"

greet = new_greet

print(greet("Alice"))  # Output: Greetings, Alice! How are you today?

# Restore the original function
greet = original_greet
print(greet("Bob"))    # Output: Hello, Bob!
```

Slide 5: Patching Built-in Functions

Even Python's built-in functions can be monkey patched. However, this should be done with extreme caution as it can lead to unexpected behavior across your entire program.

```python
# Original behavior
print(len([1, 2, 3]))  # Output: 3

# Monkey patching the built-in len() function
original_len = len

def new_len(obj):
    if isinstance(obj, list):
        return original_len(obj) * 2
    return original_len(obj)

builtins.len = new_len

print(len([1, 2, 3]))  # Output: 6
print(len("hello"))    # Output: 5

# Restore the original len() function
builtins.len = original_len
```

Slide 6: Real-Life Example: Logging Function Calls

Monkey patching can be used to add logging to existing functions without modifying their source code. This is particularly useful for debugging or monitoring third-party libraries.

```python
import functools

def log_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

# Existing function in a third-party library
def calculate_area(length, width):
    return length * width

# Monkey patch to add logging
calculate_area = log_calls(calculate_area)

# Using the patched function
area = calculate_area(5, 3)
# Output:
# Calling calculate_area with args: (5, 3), kwargs: {}
# calculate_area returned: 15
```

Slide 7: Real-Life Example: Feature Flags

Monkey patching can be used to implement feature flags, allowing you to enable or disable features at runtime without changing the codebase.

```python
class FeatureFlags:
    def __init__(self):
        self.flags = {}

    def set_flag(self, feature, enabled):
        self.flags[feature] = enabled

    def is_enabled(self, feature):
        return self.flags.get(feature, False)

feature_flags = FeatureFlags()

def send_notification(user, message):
    if feature_flags.is_enabled("new_notification_system"):
        print(f"Sending notification to {user} using new system: {message}")
    else:
        print(f"Sending notification to {user} using old system: {message}")

# Initially using the old system
send_notification("Alice", "Hello!")
# Output: Sending notification to Alice using old system: Hello!

# Enable the new notification system
feature_flags.set_flag("new_notification_system", True)

# Now using the new system
send_notification("Bob", "Hi there!")
# Output: Sending notification to Bob using new system: Hi there!
```

Slide 8: Dangers of Monkey Patching

While powerful, monkey patching can lead to maintainability issues, unexpected behavior, and hard-to-debug problems. It can make code harder to understand and reason about, especially in large projects or when working in teams.

```python
# Original implementation
def divide(a, b):
    return a / b

# Monkey patching with unexpected behavior
def unsafe_divide(a, b):
    if b == 0:
        return float('inf')  # Silently handle division by zero
    return a / b

# Replacing the original function
divide = unsafe_divide

# This now silently returns infinity instead of raising an exception
result = divide(5, 0)
print(result)  # Output: inf

# This unexpected behavior might lead to bugs in other parts of the code
# that expect a ZeroDivisionError to be raised
```

Slide 9: Best Practices for Monkey Patching

To mitigate the risks associated with monkey patching, follow these best practices:

1. Document all monkey patches clearly.
2. Use monkey patching sparingly and only when necessary.
3. Consider using decorators or inheritance instead, if possible.
4. Be cautious when patching built-in functions or third-party libraries.
5. Implement patches in a centralized location for better maintainability.

```python
# Example of a well-documented monkey patch
import some_library

def patched_function(arg1, arg2):
    """
    Monkey patch for some_library.original_function
    
    This patch adds logging and error handling to the original function.
    """
    try:
        print(f"Calling original_function with args: {arg1}, {arg2}")
        result = some_library.original_function(arg1, arg2)
        print(f"original_function returned: {result}")
        return result
    except Exception as e:
        print(f"Error in original_function: {e}")
        raise

# Apply the patch
some_library.original_function = patched_function
```

Slide 10: Testing with Monkey Patching

Monkey patching is often used in unit testing to replace parts of the system with mock objects or to isolate specific components for testing.

```python
import unittest
from unittest.mock import patch

def get_weather(city):
    # Assume this function makes an API call to get weather data
    pass

class TestWeatherFunction(unittest.TestCase):
    @patch('__main__.get_weather')
    def test_get_weather(self, mock_get_weather):
        mock_get_weather.return_value = {"temperature": 25, "condition": "Sunny"}
        
        result = get_weather("London")
        
        self.assertEqual(result, {"temperature": 25, "condition": "Sunny"})
        mock_get_weather.assert_called_once_with("London")

if __name__ == '__main__':
    unittest.main()

# This test uses monkey patching to replace the get_weather function
# with a mock object, allowing us to test the function's behavior
# without making actual API calls.
```

Slide 11: Monkey Patching in Popular Libraries

Many popular Python libraries use monkey patching internally or provide APIs for users to monkey patch their behavior. For example, the `gevent` library uses monkey patching to make synchronous code work asynchronously.

```python
import time
import requests

def download_sites(sites):
    for site in sites:
        print(f"Downloading {site}")
        requests.get(site)

sites = ["https://www.example.com", "https://www.python.org", "https://www.openai.com"]

start_time = time.time()
download_sites(sites)
duration = time.time() - start_time
print(f"Downloaded {len(sites)} sites in {duration} seconds")

# Now, let's use gevent to make it asynchronous
from gevent import monkey
monkey.patch_all()

start_time = time.time()
download_sites(sites)
duration = time.time() - start_time
print(f"Downloaded {len(sites)} sites in {duration} seconds (with gevent)")

# The second run with gevent will be significantly faster
# as it uses monkey patching to make the requests asynchronous
```

Slide 12: Alternatives to Monkey Patching

While monkey patching can be powerful, there are often cleaner alternatives that can achieve similar results with less risk:

1. Inheritance: Extend classes to add or modify behavior.
2. Composition: Use wrapper classes or the Decorator pattern.
3. Dependency Injection: Pass modified objects as arguments.

```python
# Original class
class Calculator:
    def add(self, a, b):
        return a + b

# Inheritance
class EnhancedCalculator(Calculator):
    def multiply(self, a, b):
        return a * b

# Composition (Decorator pattern)
class CalculatorLogger:
    def __init__(self, calculator):
        self.calculator = calculator

    def add(self, a, b):
        result = self.calculator.add(a, b)
        print(f"Addition: {a} + {b} = {result}")
        return result

# Usage
calc = EnhancedCalculator()
print(calc.add(2, 3))      # Output: 5
print(calc.multiply(2, 3)) # Output: 6

logged_calc = CalculatorLogger(Calculator())
logged_calc.add(2, 3)      # Output: Addition: 2 + 3 = 5
```

Slide 13: Conclusion

Monkey patching is a powerful technique in Python that allows for runtime modification of code. While it can be useful for debugging, testing, and adapting third-party libraries, it should be used judiciously. Understanding the principles, benefits, and risks of monkey patching is crucial for writing maintainable and robust Python code.

Remember:

* Use monkey patching sparingly and document it well.
* Consider alternatives like inheritance or composition when possible.
* Be aware of the potential risks and unexpected behaviors.
* Leverage monkey patching for testing and debugging, but be cautious in production code.

Slide 14: Additional Resources

For more information on monkey patching and advanced Python techniques, consider the following resources:

1. "Python in a Nutshell" by Alex Martelli, Anna Ravenscroft, and Steve Holden
2. "Fluent Python" by Luciano Ramalho
3. "Expert Python Programming" by Michal Jaworski and Tarek Ziade

For academic papers related to dynamic programming languages and runtime modifications:

1. "An Empirical Study on the Use of Dynamic Features in Python" (arXiv:1907.11751) URL: [https://arxiv.org/abs/1907.11751](https://arxiv.org/abs/1907.11751)
2. "Dynamic Languages and Software Development" (arXiv:2102.01725) URL: [https://arxiv.org/abs/2102.01725](https://arxiv.org/abs/2102.01725)

These resources provide deeper insights into Python's dynamic nature and advanced programming techniques, including monkey patching.

