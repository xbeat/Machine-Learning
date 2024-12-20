## Debugging Tips for Coding Challenges with Python
Slide 1: The Disappearing Act

Reproducibility Problem Keep a log of actions when the bug appears to recreate conditions

```python
# Don't: Inconsistent bug reporting
def process_data(data):
    # Intermittent bug occurs here
    result = complex_calculation(data)
    return result

# Do: Log actions for reproducibility
import logging

logging.basicConfig(level=logging.DEBUG)

def process_data(data):
    logging.debug(f"Processing data: {data}")
    result = complex_calculation(data)
    logging.debug(f"Result: {result}")
    return result
```

Slide 2: Spaghetti Code

Tangled codebase makes bug tracking difficult Break code into smaller, manageable pieces and document connections

```python
# Don't: Monolithic function
def do_everything(data):
    # Process data
    # Perform calculations
    # Generate report
    # Send emails
    # Update database
    pass

# Do: Modular functions with clear responsibilities
def process_data(data):
    # Process data
    return processed_data

def perform_calculations(processed_data):
    # Perform calculations
    return results

def generate_report(results):
    # Generate report
    return report

# Main function calls modular components
def main(data):
    processed_data = process_data(data)
    results = perform_calculations(processed_data)
    report = generate_report(results)
    send_email(report)
    update_database(results)
```

Slide 3: Lack of Documentation

Insufficient documentation hinders bug fixing Write and update documentation regularly

```python
# Don't: Undocumented function
def calculate_risk(value, factor):
    return value * factor / 100

# Do: Well-documented function
def calculate_risk(value: float, factor: float) -> float:
    """
    Calculate risk based on value and risk factor.

    Args:
        value (float): The base value to calculate risk on.
        factor (float): The risk factor as a percentage.

    Returns:
        float: The calculated risk value.

    Example:
        >>> calculate_risk(1000, 5)
        50.0
    """
    return value * factor / 100
```

Slide 4: Environment-Specific Bugs

Bugs appear only in certain setups Use Docker for consistent development and testing environments

```python
# Don't: Hardcoded paths or environment-specific code
DATABASE_PATH = "C:\Users\JohnDoe\Documents\mydb.sqlite"

# Do: Use environment variables and Docker
import os

DATABASE_PATH = os.getenv("DATABASE_PATH", "/app/data/mydb.sqlite")

# Dockerfile
# FROM python:3.9
# WORKDIR /app
#  . /app
# ENV DATABASE_PATH=/app/data/mydb.sqlite
# CMD ["python", "main.py"]
```

Slide 5: Testing Shortfalls

Insufficient testing allows bugs to slip through Implement comprehensive automated testing

```python
# Don't: Manual testing only
def add_numbers(a, b):
    return a + b

# Manually test
print(add_numbers(2, 3))  # Output: 5

# Do: Automated testing with multiple scenarios
import unittest

class TestAddNumbers(unittest.TestCase):
    def test_positive_numbers(self):
        self.assertEqual(add_numbers(2, 3), 5)
    
    def test_negative_numbers(self):
        self.assertEqual(add_numbers(-1, -1), -2)
    
    def test_zero(self):
        self.assertEqual(add_numbers(0, 0), 0)
    
    def test_large_numbers(self):
        self.assertEqual(add_numbers(1000000, 2000000), 3000000)

if __name__ == '__main__':
    unittest.main()
```

Slide 6: Dependency Issues

Bugs caused by outdated or incompatible dependencies Regularly update and test dependencies

```python
# Don't: Neglect dependency management
# requirements.txt
# requests
# numpy
# pandas

# Do: Specify versions and regularly update
# requirements.txt
requests==2.26.0
numpy==1.21.2
pandas==1.3.3

# In your code
import pkg_resources

def check_dependencies():
    required = {'requests': '2.26.0', 'numpy': '1.21.2', 'pandas': '1.3.3'}
    for package, version in required.items():
        pkg_resources.require(f"{package}=={version}")

check_dependencies()
```

Slide 7: Human Error

Mistakes due to typos or misunderstandings Implement code reviews and clear communication

```python
# Don't: Push code without review
def calcula_total(items):  # Typo in function name
    total = 0
    for item in items:
        total += iten.price  # Typo in variable name
    return total

# Do: Use code reviews and linters
def calculate_total(items):
    """
    Calculate the total price of all items.
    
    Args:
        items (list): List of Item objects with 'price' attribute.
    
    Returns:
        float: Total price of all items.
    """
    return sum(item.price for item in items)

# Use a linter like pylint to catch typos and style issues
# Run: pylint your_module.py
```

Slide 8: Legacy Code Nightmares

Difficulties in maintaining and updating old code Make gradual, well-tested updates to legacy code

```python
# Don't: Overhaul legacy code all at once
def old_complex_function():
    # Hundreds of lines of outdated, poorly documented code
    pass

# Do: Refactor gradually with tests
def legacy_wrapper(input_data):
    """Wrapper for old_complex_function with new interface."""
    # Prepare input for old function
    result = old_complex_function(input_data)
    # Process result to match new expected output
    return processed_result

# New function replacing part of the old one
def new_improved_function(input_data):
    """New implementation of a part of old_complex_function."""
    # New, cleaner implementation
    return result

# Tests to ensure new function matches old behavior
def test_new_matches_old():
    assert legacy_wrapper(test_data) == new_improved_function(test_data)
```

Slide 9: Additional Resources

1. "Best Practices for Scientific Computing" - [https://arxiv.org/abs/1210.0530](https://arxiv.org/abs/1210.0530)
2. "The Art of Readable Code" by Dustin Boswell and Trevor Foucher
3. "Clean Code: A Handbook of Agile Software Craftsmanship" by Robert C. Martin

