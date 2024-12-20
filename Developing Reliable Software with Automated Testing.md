## Developing Reliable Software with Automated Testing
Slide 1: The Importance of Testing in Software Development

Testing is a crucial aspect of software development that ensures code quality, reliability, and maintainability. By writing and executing tests, developers can catch bugs early, validate functionality, and gain confidence in their code. Let's explore the benefits of testing through practical examples and real-world scenarios.

```python
def add_numbers(a, b):
    return a + b

# Test the add_numbers function
def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0

test_add_numbers()
print("All tests passed!")
```

Slide 2: Catching Bugs Early

Testing helps detect issues during development, preventing costly fixes after deployment. By writing tests as you code, you can identify and resolve problems immediately.

```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# Test the calculate_average function
def test_calculate_average():
    assert calculate_average([1, 2, 3, 4, 5]) == 3
    assert calculate_average([0, 10]) == 5
    assert calculate_average([-1, 1]) == 0

    # This test will fail, catching a potential bug
    assert calculate_average([]) == 0  # ZeroDivisionError

test_calculate_average()
```

Slide 3: Ensuring Code Quality

Tests provide a safety net, ensuring that code behaves as expected. They help maintain code quality by validating functionality and preventing regressions.

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

# Test the Rectangle class
def test_rectangle():
    rect = Rectangle(5, 3)
    assert rect.area() == 15
    
    rect.width = 10
    assert rect.area() == 30

test_rectangle()
print("Rectangle tests passed!")
```

Slide 4: Confidence in Refactoring

Tests allow developers to modify code without fear of breaking existing functionality. They provide a safety net when refactoring or optimizing code.

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the fibonacci function
def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55

test_fibonacci()
print("Fibonacci tests passed!")

# Refactor the fibonacci function to improve performance
def fibonacci_optimized(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Test the optimized version
def test_fibonacci_optimized():
    assert fibonacci_optimized(0) == 0
    assert fibonacci_optimized(1) == 1
    assert fibonacci_optimized(5) == 5
    assert fibonacci_optimized(10) == 55

test_fibonacci_optimized()
print("Optimized Fibonacci tests passed!")
```

Slide 5: Documenting Behavior

Tests serve as living documentation for how the system is supposed to work. They provide clear examples of expected inputs and outputs.

```python
def is_palindrome(s):
    """
    Check if a string is a palindrome.
    
    A palindrome is a word, phrase, number, or other sequence of characters
    that reads the same forward and backward, ignoring spaces, punctuation,
    and capitalization.
    
    Args:
        s (str): The string to check.
    
    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

# Test the is_palindrome function
def test_is_palindrome():
    assert is_palindrome("A man a plan a canal Panama")
    assert is_palindrome("race a car") == False
    assert is_palindrome("Was it a car or a cat I saw?")
    assert is_palindrome("hello") == False

test_is_palindrome()
print("Palindrome tests passed!")
```

Slide 6: Faster Debugging

Isolated failures in tests can pinpoint exactly where bugs occur, making debugging more efficient and less time-consuming.

```python
def divide_numbers(a, b):
    return a / b

# Test the divide_numbers function
def test_divide_numbers():
    assert divide_numbers(10, 2) == 5
    assert divide_numbers(-6, 3) == -2
    assert divide_numbers(0, 5) == 0
    
    try:
        divide_numbers(5, 0)
    except ZeroDivisionError:
        print("Caught ZeroDivisionError as expected")
    else:
        raise AssertionError("Expected ZeroDivisionError, but no exception was raised")

test_divide_numbers()
print("Division tests passed!")
```

Slide 7: Better Collaboration

Tests define clear expectations for how code should behave, aiding teamwork and reducing misunderstandings between developers.

```python
class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add_item(self, item, quantity):
        self.items[item] = self.items.get(item, 0) + quantity

    def remove_item(self, item, quantity):
        if item in self.items:
            self.items[item] = max(0, self.items[item] - quantity)
            if self.items[item] == 0:
                del self.items[item]

    def get_total_items(self):
        return sum(self.items.values())

# Test the ShoppingCart class
def test_shopping_cart():
    cart = ShoppingCart()
    
    cart.add_item("apple", 3)
    assert cart.get_total_items() == 3
    
    cart.add_item("banana", 2)
    assert cart.get_total_items() == 5
    
    cart.remove_item("apple", 1)
    assert cart.get_total_items() == 4
    
    cart.remove_item("banana", 3)  # Should remove all bananas
    assert cart.get_total_items() == 2
    assert "banana" not in cart.items

test_shopping_cart()
print("Shopping cart tests passed!")
```

Slide 8: Reduce Regression Issues

Automated tests prevent old bugs from reappearing after updates, ensuring that fixed issues stay fixed.

```python
class TemperatureConverter:
    @staticmethod
    def celsius_to_fahrenheit(celsius):
        return (celsius * 9/5) + 32

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit):
        return (fahrenheit - 32) * 5/9

# Test the TemperatureConverter class
def test_temperature_converter():
    converter = TemperatureConverter()
    
    # Test Celsius to Fahrenheit
    assert round(converter.celsius_to_fahrenheit(0), 2) == 32
    assert round(converter.celsius_to_fahrenheit(100), 2) == 212
    assert round(converter.celsius_to_fahrenheit(-40), 2) == -40
    
    # Test Fahrenheit to Celsius
    assert round(converter.fahrenheit_to_celsius(32), 2) == 0
    assert round(converter.fahrenheit_to_celsius(212), 2) == 100
    assert round(converter.fahrenheit_to_celsius(-40), 2) == -40

test_temperature_converter()
print("Temperature converter tests passed!")
```

Slide 9: Improve Design

Writing tests often leads to cleaner, more modular code structures. It encourages developers to think about edge cases and potential issues.

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Test the Stack class
def test_stack():
    stack = Stack()
    
    assert stack.is_empty()
    assert stack.size() == 0
    
    stack.push(1)
    stack.push(2)
    stack.push(3)
    
    assert not stack.is_empty()
    assert stack.size() == 3
    assert stack.peek() == 3
    
    assert stack.pop() == 3
    assert stack.size() == 2
    
    stack.push(4)
    assert stack.peek() == 4
    
    while not stack.is_empty():
        stack.pop()
    
    assert stack.is_empty()
    
    try:
        stack.pop()
    except IndexError as e:
        assert str(e) == "Stack is empty"
    else:
        raise AssertionError("Expected IndexError, but no exception was raised")

test_stack()
print("Stack tests passed!")
```

Slide 10: Boost Productivity

Though upfront time is required, testing saves time in the long run by reducing manual QA efforts and preventing bugs from reaching production.

```python
import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def memoized_fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = memoized_fibonacci(n-1, memo) + memoized_fibonacci(n-2, memo)
    return memo[n]

# Test and compare the performance of both implementations
def test_fibonacci_performance():
    n = 30
    
    start_time = time.time()
    result1 = fibonacci(n)
    end_time = time.time()
    time1 = end_time - start_time
    
    start_time = time.time()
    result2 = memoized_fibonacci(n)
    end_time = time.time()
    time2 = end_time - start_time
    
    assert result1 == result2
    print(f"Regular fibonacci took {time1:.6f} seconds")
    print(f"Memoized fibonacci took {time2:.6f} seconds")
    print(f"Speedup: {time1 / time2:.2f}x")

test_fibonacci_performance()
```

Slide 11: Meet Business Requirements

Tests ensure that critical business functionality works as intended, reducing the risk of costly errors and improving customer satisfaction.

```python
class DiscountCalculator:
    def apply_discount(self, original_price, discount_percentage):
        if discount_percentage < 0 or discount_percentage > 100:
            raise ValueError("Discount percentage must be between 0 and 100")
        discount_amount = original_price * (discount_percentage / 100)
        return round(original_price - discount_amount, 2)

# Test the DiscountCalculator class
def test_discount_calculator():
    calculator = DiscountCalculator()
    
    assert calculator.apply_discount(100, 20) == 80.00
    assert calculator.apply_discount(50, 10) == 45.00
    assert calculator.apply_discount(75.50, 15) == 64.17
    
    try:
        calculator.apply_discount(100, -10)
    except ValueError as e:
        assert str(e) == "Discount percentage must be between 0 and 100"
    
    try:
        calculator.apply_discount(100, 110)
    except ValueError as e:
        assert str(e) == "Discount percentage must be between 0 and 100"

test_discount_calculator()
print("Discount calculator tests passed!")
```

Slide 12: Real-Life Example: Weather Forecast

Let's consider a weather forecasting application that predicts temperature and precipitation for the next day.

```python
import random

class WeatherForecaster:
    def __init__(self):
        self.temperature = 0
        self.precipitation = 0

    def predict_temperature(self):
        # Simulate temperature prediction (in Celsius)
        self.temperature = round(random.uniform(-10, 35), 1)
        return self.temperature

    def predict_precipitation(self):
        # Simulate precipitation prediction (in mm)
        self.precipitation = round(random.uniform(0, 50), 1)
        return self.precipitation

    def get_forecast(self):
        temp = self.predict_temperature()
        precip = self.predict_precipitation()
        return f"Tomorrow's forecast: {temp}°C, {precip}mm precipitation"

# Test the WeatherForecaster class
def test_weather_forecaster():
    forecaster = WeatherForecaster()
    
    # Test temperature prediction
    temp = forecaster.predict_temperature()
    assert -10 <= temp <= 35
    
    # Test precipitation prediction
    precip = forecaster.predict_precipitation()
    assert 0 <= precip <= 50
    
    # Test forecast string
    forecast = forecaster.get_forecast()
    assert "Tomorrow's forecast:" in forecast
    assert "°C" in forecast
    assert "mm precipitation" in forecast

    print("Sample forecast:", forecast)

test_weather_forecaster()
print("Weather forecaster tests passed!")
```

Slide 13: Real-Life Example: Password Strength Checker

Let's implement a simple password strength checker and test its functionality.

```python
import re

class PasswordChecker:
    @staticmethod
    def check_strength(password):
        if len(password) < 8:
            return "Weak: Password should be at least 8 characters long"
        
        if not re.search(r"[A-Z]", password):
            return "Weak: Password should contain at least one uppercase letter"
        
        if not re.search(r"[a-z]", password):
            return "Weak: Password should contain at least one lowercase letter"
        
        if not re.search(r"\d", password):
            return "Weak: Password should contain at least one digit"
        
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return "Moderate: Password should contain at least one special character for better security"
        
        return "Strong: Password meets all criteria"

# Test the PasswordChecker class
def test_password_checker():
    checker = PasswordChecker()
    
    assert "Weak" in checker.check_strength("pass")
    assert "Weak" in checker.check_strength("password")
    assert "Weak" in checker.check_strength("PASSWORD123")
    assert "Weak" in checker.check_strength("Password")
    assert "Moderate" in checker.check_strength("Password123")
    assert "Strong" in checker.check_strength("P@ssw0rd!")
    
    print("Sample strength check:", checker.check_strength("MyP@ssw0rd!"))

test_password_checker()
print("Password checker tests passed!")
```

Slide 14: Conclusion

Testing is an essential practice in software development that offers numerous benefits. By writing and maintaining tests, developers can improve code quality, catch bugs early, and streamline the development process. Remember that testing is an ongoing process, and it's important to continuously update and expand your test suite as your codebase evolves.

Slide 15: Additional Resources

For more information on software testing and best practices, consider exploring the following resources:

1. "Software Testing: A Craftsman's Approach" by Paul C. Jorgensen
2. "Test Driven Development: By Example" by Kent Beck
3. Python's unittest framework documentation: [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)
4. pytest documentation: [https://docs.pytest.org/](https://docs.pytest.org/)
5. ArXiv paper: "A Survey of Automated Software Testing: Techniques, Applications, and Challenges" by Dongdong Wang, et al. ([https://arxiv.org/abs/2101.04373](https://arxiv.org/abs/2101.04373))

These resources can help you

