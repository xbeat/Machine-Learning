## Common Python Interview Questions with Code Examples

Slide 1: What is Python?

Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python's syntax emphasizes code readability, often using English keywords where other languages use punctuation.

```python
def greet(name):
    """This function greets the person passed in as a parameter"""
    print(f"Hello, {name}! Welcome to Python programming.")

# Call the function
greet("Alice")

# Output: Hello, Alice! Welcome to Python programming.
```

Slide 2: Variables and Data Types

Python is dynamically typed, meaning you don't need to declare variable types explicitly. It supports various data types, including integers, floats, strings, lists, tuples, and dictionaries.

```python
integer_var = 42
float_var = 3.14
string_var = "Hello, Python!"
list_var = [1, 2, 3, 4, 5]
tuple_var = (1, "two", 3.0)
dict_var = {"name": "John", "age": 30}

print(f"Integer: {integer_var}, Type: {type(integer_var)}")
print(f"Float: {float_var}, Type: {type(float_var)}")
print(f"String: {string_var}, Type: {type(string_var)}")
print(f"List: {list_var}, Type: {type(list_var)}")
print(f"Tuple: {tuple_var}, Type: {type(tuple_var)}")
print(f"Dictionary: {dict_var}, Type: {type(dict_var)}")

# Output:
# Integer: 42, Type: <class 'int'>
# Float: 3.14, Type: <class 'float'>
# String: Hello, Python!, Type: <class 'str'>
# List: [1, 2, 3, 4, 5], Type: <class 'list'>
# Tuple: (1, 'two', 3.0), Type: <class 'tuple'>
# Dictionary: {'name': 'John', 'age': 30}, Type: <class 'dict'>
```

Slide 3: Control Flow: Conditional Statements

Python uses indentation to define code blocks. The if-elif-else structure is used for conditional execution. This allows for clear and readable code structure.

```python
    if temp > 30:
        return "It's hot outside!"
    elif 20 <= temp <= 30:
        return "The weather is pleasant."
    else:
        return "It's cold, bring a jacket!"

# Test the function with different temperatures
print(check_temperature(35))  # Output: It's hot outside!
print(check_temperature(25))  # Output: The weather is pleasant.
print(check_temperature(15))  # Output: It's cold, bring a jacket!
```

Slide 4: Loops in Python

Python supports two main types of loops: for loops and while loops. For loops are commonly used for iterating over sequences, while while loops continue execution as long as a condition is true.

```python
print("For loop example:")
for i in range(5):
    print(f"Iteration {i}")

# Demonstrating while loop
print("\nWhile loop example:")
count = 0
while count < 5:
    print(f"Count is {count}")
    count += 1

# Output:
# For loop example:
# Iteration 0
# Iteration 1
# Iteration 2
# Iteration 3
# Iteration 4
# 
# While loop example:
# Count is 0
# Count is 1
# Count is 2
# Count is 3
# Count is 4
```

Slide 5: Functions in Python

Functions in Python are defined using the def keyword. They can take parameters and return values. Python also supports default arguments, keyword arguments, and variable-length arguments.

```python
    """Calculate the area of a rectangle."""
    return length * width

# Using the function with different arguments
print(calculate_area(5, 3))  # Output: 15
print(calculate_area(4))     # Output: 4 (using default width)
print(calculate_area(width=2, length=6))  # Output: 12 (using keyword arguments)

# Function with variable-length arguments
def sum_all(*args):
    """Sum all arguments passed to the function."""
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # Output: 15
```

Slide 6: List Comprehensions

List comprehensions provide a concise way to create lists based on existing lists. They can replace loops and map() calls with more readable and expressive code.

```python
squares_loop = []
for i in range(10):
    squares_loop.append(i**2)
print("Squares using loop:", squares_loop)

# Creating the same list using list comprehension
squares_comprehension = [i**2 for i in range(10)]
print("Squares using comprehension:", squares_comprehension)

# List comprehension with condition
even_squares = [i**2 for i in range(10) if i % 2 == 0]
print("Even squares:", even_squares)

# Output:
# Squares using loop: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# Squares using comprehension: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# Even squares: [0, 4, 16, 36, 64]
```

Slide 7: Exception Handling

Exception handling in Python uses try-except blocks. This allows developers to gracefully handle errors and unexpected situations in their code.

```python
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError:
        print("Error: Invalid input types!")
        return None
    else:
        print("Division successful")
        return result
    finally:
        print("Division operation completed")

# Test the function with different inputs
print(divide(10, 2))   # Successful division
print(divide(10, 0))   # Division by zero
print(divide('10', 2)) # Type error

# Output:
# Division successful
# Division operation completed
# 5.0
# Error: Division by zero!
# Division operation completed
# None
# Error: Invalid input types!
# Division operation completed
# None
```

Slide 8: Object-Oriented Programming

Python supports object-oriented programming (OOP) with classes and objects. Classes are used to create user-defined data structures and can contain attributes and methods.

```python
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.odometer = 0
    
    def describe_car(self):
        return f"{self.year} {self.make} {self.model}"
    
    def drive(self, distance):
        self.odometer += distance
        return f"Drove {distance} km. Total: {self.odometer} km"

# Creating and using a Car object
my_car = Car("Toyota", "Corolla", 2022)
print(my_car.describe_car())
print(my_car.drive(100))
print(my_car.drive(50))

# Output:
# 2022 Toyota Corolla
# Drove 100 km. Total: 100 km
# Drove 50 km. Total: 150 km
```

Slide 9: File Handling

Python provides easy-to-use functions for file operations. The with statement ensures proper handling of file resources.

```python
with open('example.txt', 'w') as file:
    file.write("Hello, Python!\n")
    file.write("This is a file handling example.")

# Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print("File contents:")
    print(content)

# Appending to a file
with open('example.txt', 'a') as file:
    file.write("\nAppending new content.")

# Reading lines from a file
print("\nReading lines:")
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())

# Output:
# File contents:
# Hello, Python!
# This is a file handling example.
# 
# Reading lines:
# Hello, Python!
# This is a file handling example.
# Appending new content.
```

Slide 10: Modules and Packages

Python's modular design allows code to be organized into modules and packages. This promotes code reusability and maintainability.

```python
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# main.py
import math_operations

result_add = math_operations.add(5, 3)
result_subtract = math_operations.subtract(10, 4)

print(f"Addition result: {result_add}")
print(f"Subtraction result: {result_subtract}")

# Output:
# Addition result: 8
# Subtraction result: 6
```

Slide 11: Lambda Functions

Lambda functions are small, anonymous functions defined using the lambda keyword. They are often used for short operations and can be passed as arguments to other functions.

```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print("Squared numbers:", squared)

# Using a lambda function with filter()
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print("Even numbers:", even_numbers)

# Using a lambda function with sorted()
students = [('Alice', 22), ('Bob', 19), ('Charlie', 24)]
sorted_students = sorted(students, key=lambda x: x[1])
print("Sorted students:", sorted_students)

# Output:
# Squared numbers: [1, 4, 9, 16, 25]
# Even numbers: [2, 4]
# Sorted students: [('Bob', 19), ('Alice', 22), ('Charlie', 24)]
```

Slide 12: Decorators

Decorators are a powerful feature in Python that allow the modification of functions or classes without directly changing their source code. They are often used for logging, timing functions, or adding authentication.

```python

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.5f} seconds to execute.")
        return result
    return wrapper

@timer_decorator
def slow_function():
    time.sleep(2)
    print("Function executed")

slow_function()

# Output:
# Function executed
# slow_function took 2.00309 seconds to execute.
```

Slide 13: Context Managers

Context managers in Python, implemented using the with statement, provide a convenient way to manage resources like file handles or database connections. They ensure proper setup and cleanup of resources.

```python
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}, {exc_value}")
        return False  # Propagate exceptions

# Using the custom context manager
with CustomContextManager() as cm:
    print("Inside the context")
    # Uncomment the next line to see exception handling
    # raise ValueError("An error occurred")

print("After the context")

# Output:
# Entering the context
# Inside the context
# Exiting the context
# After the context
```

Slide 14: Real-life Example: Web Scraping

Web scraping is a common task in Python. Here's a simple example using the requests and BeautifulSoup libraries to extract information from a website.

```python
from bs4 import BeautifulSoup

# Fetch the HTML content of a web page
url = "https://example.com"
response = requests.get(url)
html_content = response.text

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Extract all paragraph texts
paragraphs = soup.find_all('p')
for i, paragraph in enumerate(paragraphs, 1):
    print(f"Paragraph {i}: {paragraph.text.strip()}")

# Extract the title of the page
title = soup.title.string if soup.title else "No title found"
print(f"\nPage Title: {title}")

# Note: This code assumes you have installed the requests and beautifulsoup4 libraries
# You can install them using: pip install requests beautifulsoup4
```

Slide 15: Real-life Example: Data Analysis

Python is widely used for data analysis. Here's a simple example using pandas to analyze a dataset.

```python
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 22],
    'City': ['New York', 'San Francisco', 'London', 'Paris', 'Tokyo'],
    'Salary': [50000, 70000, 65000, 55000, 45000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display basic information about the dataset
print(df.describe())

# Calculate and print the average salary
average_salary = df['Salary'].mean()
print(f"\nAverage Salary: ${average_salary:.2f}")

# Find the person with the highest salary
highest_paid = df.loc[df['Salary'].idxmax()]
print(f"\nHighest paid person:\n{highest_paid}")

# Create a bar plot of salaries
plt.figure(figsize=(10, 6))
plt.bar(df['Name'], df['Salary'])
plt.title('Salaries by Person')
plt.xlabel('Name')
plt.ylabel('Salary ($)')
plt.show()

# Note: This code assumes you have installed pandas and matplotlib
# You can install them using: pip install pandas matplotlib
```

Slide 16: Additional Resources

For further learning about Python and its applications, consider exploring these resources:

1. Official Python Documentation: [https://docs.python.org/3/](https://docs.python.org/3/)
2. Python for Data Science Handbook: [https://jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/)
3. Real Python Tutorials: [https://realpython.com/](https://realpython.com/)
4. ArXiv paper on Python in Scientific Computing: [https://arxiv.org/abs/1807.04806](https://arxiv.org/abs/1807.04806)
5. ArXiv paper on Python for Machine Learning: [https://arxiv.org/abs/2009.04806](https://arxiv.org/abs/2009.04806)

These resources provide in-depth information on various aspects of Python programming and its applications in different fields.


