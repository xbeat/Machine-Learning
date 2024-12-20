## Exception Handling in Python A Comprehensive Guide
Slide 1: Understanding Exceptions in Python

Exceptions are events that occur during program execution which disrupt the normal flow of instructions. In Python, exceptions are objects that represent these error conditions. When an exception is raised, it propagates up the call stack until it's caught by an exception handler or causes the program to terminate.

```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(f"Error occurred: {e}")
```

Slide 2: Built-in Exceptions

Python provides a variety of built-in exceptions to handle different error scenarios. Some common ones include TypeError, ValueError, IndexError, and KeyError. These exceptions help in identifying and handling specific error conditions in your code.

```python
# Examples of built-in exceptions
try:
    # TypeError
    "2" + 2

    # ValueError
    int("abc")

    # IndexError
    list_example = [1, 2, 3]
    print(list_example[5])

    # KeyError
    dict_example = {"a": 1, "b": 2}
    print(dict_example["c"])
except (TypeError, ValueError, IndexError, KeyError) as e:
    print(f"Caught an exception: {type(e).__name__} - {e}")
```

Slide 3: Custom Exceptions

Custom exceptions allow you to define application-specific error conditions. They are created by subclassing the Exception class or any of its subclasses. Custom exceptions can carry additional information about the error and make your code more readable and maintainable.

```python
class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: balance {balance}, tried to withdraw {amount}")

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

Slide 4: Try-Except Blocks

Try-except blocks are the foundation of exception handling in Python. The try block contains the code that might raise an exception, while the except block specifies how to handle the exception if it occurs. This structure allows you to gracefully manage errors and prevent your program from crashing.

```python
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
        result = None
    except TypeError:
        print("Error: Invalid operand types!")
        result = None
    else:
        print("Division successful!")
    finally:
        print("Operation complete.")
    return result

print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide("10", 2))
```

Slide 5: Handling Multiple Exceptions

Python allows you to handle multiple exceptions in a single except block or in separate blocks. This feature enables you to provide different responses based on the type of exception that occurs, making your error handling more precise and informative.

```python
def process_data(data):
    try:
        if len(data) > 5:
            raise ValueError("Data too long")
        result = 100 / len(data)
        print(data[10])
    except ZeroDivisionError:
        print("Error: Empty data")
    except ValueError as ve:
        print(f"Error: {ve}")
    except IndexError:
        print("Error: Accessing invalid index")
    except Exception as e:
        print(f"Unexpected error: {e}")

process_data([])
process_data([1, 2, 3, 4, 5, 6])
process_data([1, 2, 3])
```

Slide 6: The `else` and `finally` Clauses

The `else` clause in a try-except block is executed if no exceptions are raised. The `finally` clause is always executed, regardless of whether an exception occurred or not. These clauses help in structuring your code and ensuring that certain operations are always performed.

```python
def read_file(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except IOError:
        print(f"Error: Unable to read file '{filename}'")
    else:
        print(f"Successfully read {len(content)} characters from '{filename}'")
        return content
    finally:
        print("File operation attempt completed")

read_file("existing_file.txt")
read_file("non_existent_file.txt")
```

Slide 7: Raising Exceptions

In Python, you can raise exceptions explicitly using the `raise` statement. This is useful when you want to signal an error condition in your own functions or methods. You can raise built-in exceptions or custom exceptions that you've defined.

```python
def validate_age(age):
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age is unrealistically high")
    print(f"Age {age} is valid")

try:
    validate_age(25)
    validate_age(-5)
    validate_age("thirty")
    validate_age(200)
except (TypeError, ValueError) as e:
    print(f"Validation error: {e}")
```

Slide 8: Exception Chaining

Exception chaining allows you to associate a new exception with a previous one. This is useful when you want to raise a different type of exception while preserving information about the original error. Python provides the `raise ... from ...` syntax for this purpose.

```python
def fetch_data():
    raise ConnectionError("Unable to connect to server")

def process_data():
    try:
        fetch_data()
    except ConnectionError as ce:
        raise ValueError("Data processing failed") from ce

try:
    process_data()
except ValueError as ve:
    print(f"Error: {ve}")
    if ve.__cause__:
        print(f"Caused by: {ve.__cause__}")
```

Slide 9: Context Managers and the `with` Statement

Context managers, used with the `with` statement, provide a clean way to manage resources like file handles or network connections. They ensure that resources are properly acquired and released, even if exceptions occur. This helps prevent resource leaks and makes your code more robust.

```python
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name

    def __enter__(self):
        print(f"Connecting to database '{self.db_name}'")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Closing connection to database '{self.db_name}'")
        if exc_type:
            print(f"An error occurred: {exc_value}")
        return False  # Propagate exceptions

    def query(self, sql):
        if "DROP" in sql.upper():
            raise ValueError("DROP statements are not allowed")
        print(f"Executing query: {sql}")

with DatabaseConnection("mydb") as db:
    db.query("SELECT * FROM users")
    db.query("DROP TABLE users")
```

Slide 10: Logging Exceptions

Logging is crucial for debugging and monitoring applications. Python's logging module provides a flexible framework for generating log messages. When combined with exception handling, it allows you to record detailed error information for later analysis.

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        logging.error("Division by zero attempted", exc_info=True)
        result = None
    except TypeError:
        logging.error("Invalid types for division", exc_info=True)
        result = None
    else:
        logging.info(f"Successfully divided {a} by {b}")
    return result

divide(10, 2)
divide(10, 0)
divide("10", 2)
```

Slide 11: Handling Exceptions in Asynchronous Code

Asynchronous programming in Python, often using the `asyncio` module, requires special consideration for exception handling. Exceptions in asynchronous code can be tricky to debug and manage, but Python provides tools to handle them effectively.

```python
import asyncio

async def risky_operation(task_id):
    if task_id % 2 == 0:
        raise ValueError(f"Even task ID {task_id} not allowed")
    await asyncio.sleep(1)
    return f"Task {task_id} completed"

async def main():
    tasks = [risky_operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(result)

asyncio.run(main())
```

Slide 12: Best Practices in Exception Handling

Effective exception handling involves following best practices to make your code more robust and maintainable. These practices include being specific with exception types, avoiding bare except clauses, and not suppressing exceptions unnecessarily.

```python
def process_user_input(user_input):
    try:
        # Convert input to integer
        value = int(user_input)
        
        # Perform some operation
        result = 100 / value
        
        # Write result to file
        with open("results.txt", "w") as file:
            file.write(str(result))
        
        return result
    except ValueError:
        print("Invalid input: Please enter a valid integer")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
    except IOError as e:
        print(f"File error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise  # Re-raise unexpected exceptions

# Example usage
user_inputs = ["10", "0", "abc", "5"]
for input_value in user_inputs:
    try:
        result = process_user_input(input_value)
        if result:
            print(f"Result: {result}")
    except Exception as e:
        print(f"Unhandled exception: {e}")
```

Slide 13: Real-Life Example: Web Scraping with Error Handling

Web scraping often involves dealing with network issues, parsing errors, and unexpected page structures. Proper exception handling can make your scraping scripts more resilient and informative.

```python
import urllib.request
from urllib.error import URLError, HTTPError
from html.parser import HTMLParser

class TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = None

    def handle_starttag(self, tag, attrs):
        if tag == 'title':
            self.title = ""

    def handle_data(self, data):
        if self.title is not None:
            self.title += data

    def handle_endtag(self, tag):
        if tag == 'title':
            self.title = self.title.strip()

def get_website_title(url):
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')
            parser = TitleParser()
            parser.feed(html)
            return parser.title
    except HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
    except URLError as e:
        print(f"URL Error: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# Example usage
urls = [
    "https://www.python.org",
    "https://www.nonexistentwebsite123456.com",
    "https://httpstat.us/404",
    "https://httpstat.us/500"
]

for url in urls:
    title = get_website_title(url)
    if title:
        print(f"Title of {url}: {title}")
    else:
        print(f"Failed to retrieve title for {url}")
```

Slide 14: Real-Life Example: File Processing with Exception Handling

File processing is a common task that can benefit greatly from robust exception handling. This example demonstrates how to handle various exceptions that might occur when reading, processing, and writing files.

```python
import os
import csv
from datetime import datetime

def process_csv_file(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            headers = next(reader)
            writer.writerow(headers + ['Processed'])
            
            for row in reader:
                try:
                    # Assume the last column is a date
                    date_str = row[-1]
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    # Add a day to the date
                    new_date = date_obj + datetime.timedelta(days=1)
                    row[-1] = new_date.strftime('%Y-%m-%d')
                    
                    writer.writerow(row + ['Yes'])
                except ValueError as ve:
                    print(f"Error processing row {reader.line_num}: {ve}")
                    writer.writerow(row + ['No'])
        
        print(f"Processing complete. Output written to {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except PermissionError:
        print(f"Error: Permission denied when accessing '{output_file}'")
    except csv.Error as e:
        print(f"CSV Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
input_csv = 'input_data.csv'
output_csv = 'processed_data.csv'

# Create a sample input CSV file
with open(input_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age', 'Date'])
    writer.writerow(['Alice', '30', '2023-01-15'])
    writer.writerow(['Bob', '25', '2023-02-20'])
    writer.writerow(['Charlie', '35', 'invalid-date'])

process_csv_file(input_csv, output_csv)

# Clean up the created files
os.remove(input_csv)
os.remove(output_csv)
```

Slide 15: Additional Resources

For further exploration of exception handling in Python, consider the following resources:

1.  Python Official Documentation on Errors and Exceptions: [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
2.  PEP 3134 -- Exception Chaining and Embedded Tracebacks: [https://www.python.org/dev/peps/pep-3134/](https://www.python.org/dev/peps/pep-3134/)
3.  Real Python's Guide to Python Exceptions: [https://realpython.com/python-exceptions/](https://realpython.com/python-exceptions/)
4.  "Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin (Book)
5.  "Python Cookbook" by David Beazley and Brian K. Jones (Book)

These resources provide in-depth explanations, best practices, and advanced techniques for mastering exception handling in Python.

