## Python for Data Science and Basics

Slide 1: Introduction to Python

Python is a versatile, high-level programming language known for its simplicity and readability. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python's extensive standard library and third-party packages make it suitable for various applications, from web development to scientific computing.

```python
# A simple "Hello, World!" program in Python
print("Hello, World!")

# Basic arithmetic operations
result = 5 + 3 * 2
print(f"5 + 3 * 2 = {result}")

# Defining a function
def greet(name):
    return f"Hello, {name}!"

# Calling the function
message = greet("Python Learner")
print(message)
```

Slide 2: Variables and Data Types

Python uses dynamic typing, allowing variables to hold different types of data without explicit declaration. Common data types include integers, floating-point numbers, strings, and booleans. Variables are created by assigning a value to a name.

```python
# Integer
age = 25
print(f"Age: {age}, Type: {type(age)}")

# Float
height = 1.75
print(f"Height: {height}, Type: {type(height)}")

# String
name = "Alice"
print(f"Name: {name}, Type: {type(name)}")

# Boolean
is_student = True
print(f"Is student: {is_student}, Type: {type(is_student)}")

# Dynamic typing
x = 10
print(f"x is {x}, Type: {type(x)}")
x = "Hello"
print(f"Now x is {x}, Type: {type(x)}")
```

Slide 3: Control Flow: Conditional Statements

Conditional statements allow programs to make decisions based on certain conditions. Python uses if, elif (else if), and else keywords for this purpose. The indentation is crucial in Python as it defines the block of code associated with each condition.

```python
# Example: Determining the state of water based on temperature
temperature = 25  # in Celsius

if temperature <= 0:
    state = "solid (ice)"
elif temperature < 100:
    state = "liquid (water)"
else:
    state = "gas (steam)"

print(f"At {temperature}°C, water is in a {state} state.")

# Nested conditions
is_weekend = True
is_sunny = False

if is_weekend:
    if is_sunny:
        activity = "go to the beach"
    else:
        activity = "watch a movie at home"
else:
    activity = "go to work"

print(f"Today's activity: {activity}")
```

Slide 4: Loops in Python

Loops allow repetition of code blocks. Python provides two main types of loops: for and while. The for loop is used for iterating over a sequence (list, tuple, string, etc.), while the while loop repeats as long as a condition is true.

```python
# For loop example
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(f"I like {fruit}")

# While loop example
countdown = 5
while countdown > 0:
    print(countdown)
    countdown -= 1
print("Liftoff!")

# Using range() function with for loop
for i in range(1, 6):
    print(f"{i} squared is {i**2}")

# Break and continue statements
for num in range(1, 11):
    if num == 5:
        continue  # Skip 5
    if num == 8:
        break  # Stop at 8
    print(num)
```

Slide 5: Functions in Python

Functions are reusable blocks of code that perform specific tasks. They help in organizing code, improving readability, and reducing repetition. Functions can accept parameters and return values.

```python
# Defining a function with parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Calling the function
print(greet("Alice"))
print(greet("Bob", "Hi"))

# Function with multiple return values
def divide_and_remainder(dividend, divisor):
    quotient = dividend // divisor
    remainder = dividend % divisor
    return quotient, remainder

result = divide_and_remainder(17, 5)
print(f"17 divided by 5 gives quotient {result[0]} and remainder {result[1]}")

# Lambda functions (anonymous functions)
square = lambda x: x**2
print(f"The square of 7 is {square(7)}")
```

Slide 6: Lists and List Comprehensions

Lists are versatile, mutable sequences in Python. They can contain elements of different types and support various operations like indexing, slicing, and methods for modification. List comprehensions provide a concise way to create lists based on existing lists.

```python
# Creating and manipulating lists
numbers = [1, 2, 3, 4, 5]
print(f"Original list: {numbers}")

numbers.append(6)
print(f"After appending 6: {numbers}")

numbers.extend([7, 8])
print(f"After extending with [7, 8]: {numbers}")

popped_item = numbers.pop()
print(f"Popped item: {popped_item}")
print(f"List after pop: {numbers}")

# Slicing
print(f"First three elements: {numbers[:3]}")
print(f"Every other element: {numbers[::2]}")

# List comprehension
squares = [x**2 for x in range(1, 6)]
print(f"Squares of numbers 1 to 5: {squares}")

# Filtering with list comprehension
even_numbers = [x for x in range(1, 11) if x % 2 == 0]
print(f"Even numbers from 1 to 10: {even_numbers}")
```

Slide 7: Dictionaries and Sets

Dictionaries are key-value pairs that allow fast lookup of values. Sets are unordered collections of unique elements, useful for membership testing and eliminating duplicates.

```python
# Creating and using dictionaries
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

print(f"Person: {person}")
print(f"Name: {person['name']}")

# Adding and modifying entries
person["job"] = "Engineer"
person["age"] = 31
print(f"Updated person: {person}")

# Dictionary methods
print(f"Keys: {person.keys()}")
print(f"Values: {person.values()}")

# Sets
fruits = {"apple", "banana", "cherry"}
print(f"Fruits set: {fruits}")

# Set operations
more_fruits = {"orange", "banana", "kiwi"}
print(f"Union: {fruits.union(more_fruits)}")
print(f"Intersection: {fruits.intersection(more_fruits)}")
print(f"Difference: {fruits.difference(more_fruits)}")

# Checking membership
print(f"Is 'apple' in fruits? {'apple' in fruits}")
```

Slide 8: File Handling in Python

Python provides built-in functions for reading from and writing to files. This is essential for data persistence and processing large amounts of information.

```python
# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, File!\n")
    file.write("This is a test file.\n")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()
    print("File contents:")
    print(content)

# Reading line by line
print("Reading line by line:")
with open("example.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip() removes leading/trailing whitespace

# Appending to a file
with open("example.txt", "a") as file:
    file.write("This line is appended.\n")

# Reading the updated file
with open("example.txt", "r") as file:
    updated_content = file.read()
    print("Updated file contents:")
    print(updated_content)
```

Slide 9: Exception Handling

Exception handling allows programs to deal with unexpected errors gracefully. Python uses try, except, else, and finally blocks for this purpose.

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Error: Division by zero!")
        return None
    except TypeError:
        print("Error: Invalid types for division!")
        return None
    else:
        print("Division successful")
        return result
    finally:
        print("Division operation completed")

# Test cases
print(divide(10, 2))
print(divide(10, 0))
print(divide("10", 2))

# Custom exception
class NegativeNumberError(Exception):
    pass

def square_root(n):
    if n < 0:
        raise NegativeNumberError("Cannot calculate square root of a negative number")
    return n ** 0.5

try:
    print(square_root(16))
    print(square_root(-4))
except NegativeNumberError as e:
    print(f"Error: {e}")
```

Slide 10: Introduction to Python for Data Science

Python has become a leading language for data science due to its rich ecosystem of libraries and tools. Key libraries include NumPy for numerical computing, Pandas for data manipulation, and Matplotlib for data visualization. These libraries provide efficient data structures and algorithms for handling large datasets and complex computations.

```python
# Importing common data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating a NumPy array
arr = np.array([1, 2, 3, 4, 5])
print("NumPy array:", arr)
print("Mean:", np.mean(arr))
print("Standard deviation:", np.std(arr))

# Creating a Pandas DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)
print("\nPandas DataFrame:")
print(df)

# Simple data visualization with Matplotlib
plt.figure(figsize=(8, 6))
plt.plot(arr, arr**2, 'b-', label='y = x^2')
plt.title('Simple Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Data Manipulation with Pandas

Pandas is a powerful library for data manipulation and analysis. It provides data structures like DataFrame and Series, which allow efficient handling of structured data.

```python
import pandas as pd

# Creating a DataFrame
data = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Temperature': [12, 14, 11, 15, 13],
    'Humidity': [65, 70, 75, 60, 68],
    'WindSpeed': [10, 8, 12, 7, 9]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print("Original DataFrame:")
print(df)

# Basic operations
print("\nMean temperature:", df['Temperature'].mean())
print("Maximum humidity:", df['Humidity'].max())

# Filtering
high_temp_days = df[df['Temperature'] > 13]
print("\nDays with temperature > 13°C:")
print(high_temp_days)

# Grouping and aggregation
monthly_avg = df.resample('M').mean()
print("\nMonthly averages:")
print(monthly_avg)

# Adding a new column
df['HeatIndex'] = df['Temperature'] + 0.5 * df['Humidity']
print("\nDataFrame with Heat Index:")
print(df)
```

Slide 12: Data Visualization with Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It provides a MATLAB-like interface for creating plots and figures.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot sine wave
ax1.plot(x, y1, 'b-', label='sin(x)')
ax1.set_title('Sine Wave')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)

# Plot cosine wave
ax2.plot(x, y2, 'r-', label='cos(x)')
ax2.set_title('Cosine Wave')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Scatter plot with color mapping
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sizes = 1000 * np.random.rand(N)

plt.figure(figsize=(10, 8))
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.colorbar()
plt.title('Scatter Plot with Color Mapping')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
```

Slide 13: Real-Life Example: Weather Data Analysis

This example demonstrates how to use Python for data science in a real-world scenario: analyzing weather data. We'll use Pandas for data manipulation and Matplotlib for visualization.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample weather data (you would typically load this from a file)
data = {
    'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
    'Temperature': np.random.normal(15, 5, 365),  # Mean 15°C, std 5°C
    'Precipitation': np.random.exponential(2, 365)  # Mean 2mm
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Calculate monthly averages
monthly_avg = df.resample('M').mean()

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Temperature plot
ax1.plot(monthly_avg.index, monthly_avg['Temperature'], 'r-')
ax1.set_title('Monthly Average Temperature (2023)')
ax1.set_xlabel('Month')
ax1.set_ylabel('Temperature (°C)')
ax1.grid(True)

# Precipitation plot
ax2.bar(monthly_avg.index, monthly_avg['Precipitation'], width=20)
ax2.set_title('Monthly Average Precipitation (2023)')
ax2.set_xlabel('Month')
ax2.set_ylabel('Precipitation (mm)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print some statistics
print(f"Yearly average temperature: {df['Temperature'].mean():.2f}°C")
print(f"Hottest day: {df['Temperature'].max():.2f}°C on {df['Temperature'].idxmax().strftime('%Y-%m-%d')}")
print(f"Coldest day: {df['Temperature'].min():.2f}°C on {df['Temperature'].idxmin().strftime('%Y-%m-%d')}")
print(f"Total yearly precipitation: {df['Precipitation'].sum():.2f}mm")
```

Slide 14: Real-Life Example: Text Analysis

This example showcases how Python can be used for basic text analysis, a common task in data science and natural language processing. We'll analyze a sample text to count words, find the most common words, and calculate some basic statistics.

```python
import re
from collections import Counter

def analyze_text(text):
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count words
    word_count = len(words)
    
    # Find most common words
    word_freq = Counter(words)
    most_common = word_freq.most_common(5)
    
    # Calculate average word length
    avg_length = sum(len(word) for word in words) / word_count
    
    return word_count, most_common, avg_length

# Sample text
sample_text = """
Python is a versatile programming language. It is widely used in data science,
web development, artificial intelligence, and more. Python's simplicity and 
readability make it a popular choice for beginners and experts alike.
"""

# Analyze the text
total_words, top_words, avg_word_length = analyze_text(sample_text)

# Print results
print(f"Total words: {total_words}")
print(f"Top 5 words: {top_words}")
print(f"Average word length: {avg_word_length:.2f}")
```

Slide 15: Additional Resources

For those interested in deepening their knowledge of Python and its applications in data science, the following resources from arXiv.org may be helpful:

1.  "Python for Scientific Computing" by K. Jarrod Millman and Michael Aivazis arXiv:1102.1523 \[cs.MS\] URL: [https://arxiv.org/abs/1102.1523](https://arxiv.org/abs/1102.1523)
2.  "Effective Computation in Physics: Field Guide to Research with Python" by Anthony Scopatz and Kathryn D. Huff arXiv:1510.00002 \[physics.comp-ph\] URL: [https://arxiv.org/abs/1510.00002](https://arxiv.org/abs/1510.00002)

These papers provide in-depth discussions on using Python for scientific computing and research, which are fundamental to data science applications.

