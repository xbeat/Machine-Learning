## Mastering Python Error Handling with Try Except Else and Finally
Slide 1: Introduction to Exception Handling in Python

Exception handling is a crucial aspect of writing robust Python code. It allows developers to gracefully manage errors and unexpected situations that may occur during program execution. Python provides a structured approach to handle exceptions using the try, except, else, and finally blocks. This slideshow will explore these concepts, their usage, and practical examples to help you master exception handling in Python.

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
    else:
        print(f"The result is: {result}")
    finally:
        print("Division operation completed.")

divide(10, 2)  # Normal case
divide(10, 0)  # Error case
```

Slide 2: The Try Block

The try block is used to enclose the code that might raise an exception. It allows you to test a block of code for potential errors. If an exception occurs within the try block, the program flow immediately transfers to the corresponding except block.

```python
try:
    # Code that might raise an exception
    user_input = input("Enter a number: ")
    number = int(user_input)
    print(f"You entered: {number}")
except ValueError:
    print("Invalid input. Please enter a valid number.")
```

Slide 3: The Except Block

The except block catches and handles exceptions that occur in the try block. You can specify which type of exception to catch, or use a general except clause to catch all exceptions. Multiple except blocks can be used to handle different types of exceptions.

```python
try:
    file = open("nonexistent_file.txt", "r")
    content = file.read()
    file.close()
except FileNotFoundError:
    print("Error: The file does not exist.")
except IOError:
    print("Error: An I/O error occurred.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

Slide 4: The Else Block

The else block is executed if no exceptions were raised in the try block. It's useful for code that should only run when the try block succeeds. This helps separate the main logic from the error-handling code, making your program more readable and maintainable.

```python
def get_positive_number():
    try:
        number = float(input("Enter a positive number: "))
        if number <= 0:
            raise ValueError("Number must be positive")
    except ValueError as e:
        print(f"Error: {e}")
        return None
    else:
        print("Input successful!")
        return number

result = get_positive_number()
print(f"Result: {result}")
```

Slide 5: The Finally Block

The finally block is always executed, regardless of whether an exception occurred or not. It's typically used for cleanup operations, such as closing files or releasing resources, ensuring that these actions are performed even if an exception is raised.

```python
def read_file(filename):
    try:
        file = open(filename, "r")
        content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
    finally:
        print("Attempting to close the file...")
        try:
            file.close()
            print("File closed successfully.")
        except NameError:
            print("The file was never opened.")

content = read_file("example.txt")
print(f"File content: {content}")
```

Slide 6: Handling Multiple Exceptions

Python allows you to handle multiple exceptions in a single except block or use multiple except blocks for different exception types. This flexibility enables you to create more specific error-handling strategies based on the type of exception that occurs.

```python
def process_data(data):
    try:
        value = int(data)
        result = 100 / value
        print(f"Result: {result}")
    except ValueError:
        print("Error: Invalid input. Please enter a number.")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

process_data("10")   # Valid input
process_data("abc")  # ValueError
process_data("0")    # ZeroDivisionError
```

Slide 7: Raising Exceptions

In addition to handling exceptions, Python allows you to raise exceptions explicitly using the `raise` keyword. This is useful when you want to signal that an error condition has occurred in your code.

```python
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    elif age > 120:
        raise ValueError("Age is too high")
    else:
        print(f"Age {age} is valid")

try:
    validate_age(25)   # Valid age
    validate_age(-5)   # Raises ValueError
except ValueError as e:
    print(f"Error: {e}")
```

Slide 8: Custom Exceptions

Python allows you to create custom exception classes by inheriting from the built-in Exception class or its subclasses. This enables you to define application-specific exceptions that can provide more context about errors in your code.

```python
class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: balance {balance}, withdrawal amount {amount}")

def withdraw(balance, amount):
    if amount > balance:
        raise InsufficientFundsError(balance, amount)
    return balance - amount

try:
    new_balance = withdraw(100, 150)
except InsufficientFundsError as e:
    print(f"Error: {e}")
    print(f"Current balance: {e.balance}")
    print(f"Attempted withdrawal: {e.amount}")
```

Slide 9: Using Context Managers

Context managers, implemented using the `with` statement, provide a clean and efficient way to handle resource management and exception handling. They ensure that resources are properly acquired and released, even if exceptions occur.

```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        return True

with FileManager("example.txt", "w") as file:
    file.write("Hello, World!")
    raise ValueError("Simulated error")

print("File operation completed.")
```

Slide 10: Exception Chaining

Exception chaining allows you to preserve the original exception when raising a new one. This is useful for providing additional context about an error without losing information about the original cause.

```python
def fetch_data():
    try:
        # Simulate a network error
        raise ConnectionError("Unable to connect to the server")
    except ConnectionError as e:
        raise RuntimeError("Failed to fetch data") from e

try:
    fetch_data()
except RuntimeError as e:
    print(f"Error: {e}")
    if e.__cause__:
        print(f"Caused by: {e.__cause__}")
```

Slide 11: Handling Asynchronous Exceptions

When working with asynchronous code, such as coroutines in Python's asyncio framework, exception handling requires special consideration. The `try`/`except` blocks work similarly, but you need to use `await` with asynchronous operations.

```python
import asyncio

async def fetch_data(url):
    # Simulating an asynchronous operation
    await asyncio.sleep(1)
    if "error" in url:
        raise ValueError("Error in URL")
    return f"Data from {url}"

async def process_url(url):
    try:
        data = await fetch_data(url)
        print(f"Processed: {data}")
    except ValueError as e:
        print(f"Error processing {url}: {e}")

async def main():
    urls = ["https://example.com", "https://error.com", "https://test.com"]
    tasks = [process_url(url) for url in urls]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

Slide 12: Real-Life Example: API Request Handling

In this example, we'll simulate making API requests and handling various exceptions that might occur during the process. This demonstrates how exception handling can be used in real-world scenarios to create more robust applications.

```python
import random
import time

class APIError(Exception):
    pass

def simulate_api_request(endpoint):
    # Simulate network latency
    time.sleep(random.uniform(0.1, 0.5))
    
    # Simulate various API responses
    if random.random() < 0.2:
        raise ConnectionError("Network error occurred")
    elif random.random() < 0.1:
        raise TimeoutError("Request timed out")
    elif endpoint == "/error":
        raise APIError("Internal server error")
    else:
        return f"Data from {endpoint}"

def fetch_data(endpoint, retries=3):
    for attempt in range(retries):
        try:
            data = simulate_api_request(endpoint)
            return data
        except (ConnectionError, TimeoutError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise
        except APIError as e:
            print(f"API error occurred: {e}")
            raise
    
    raise RuntimeError("Max retries reached")

try:
    result = fetch_data("/users")
    print(f"Fetched data: {result}")
except Exception as e:
    print(f"Failed to fetch data: {e}")
```

Slide 13: Real-Life Example: Configuration File Parsing

This example demonstrates how exception handling can be used when parsing a configuration file. It shows how to handle various errors that might occur during file operations and data processing.

```python
import json

def load_config(filename):
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        
        # Validate required fields
        required_fields = ['database', 'server', 'port']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate port number
        if not isinstance(config['port'], int) or config['port'] <= 0:
            raise ValueError("Invalid port number")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {filename}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")

try:
    config = load_config('config.json')
    print("Configuration loaded successfully:")
    print(f"Database: {config['database']}")
    print(f"Server: {config['server']}")
    print(f"Port: {config['port']}")
except Exception as e:
    print(f"Error: {e}")
```

Slide 14: Additional Resources

To further enhance your understanding of exception handling in Python, consider exploring the following resources:

1.  Python's official documentation on Errors and Exceptions
2.  "Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin
3.  "Python Cookbook" by David Beazley and Brian K. Jones
4.  Online courses on platforms like Coursera, edX, or Udacity
5.  Python community forums and discussion groups

Remember to practice regularly and experiment with different scenarios to become proficient in exception handling.

