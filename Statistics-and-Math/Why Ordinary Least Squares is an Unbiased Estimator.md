## Why Ordinary Least Squares is an Unbiased Estimator
Slide 1: Introduction to Python Variables and Data Types

Python's type system provides flexible and dynamic variable handling. Variables are containers that store data, and their type is determined automatically when you assign a value. Python supports common data types like integers, floating-point numbers, strings, and booleans.

```python
# Basic variable assignments
name = "Alice"          # String type
age = 25               # Integer type
height = 1.75          # Float type
is_student = True      # Boolean type

print(f"Type of name: {type(name)}")
print(f"Type of age: {type(age)}")
```

Slide 2: Control Flow with If Statements

Control flow determines how your program executes based on conditions. If statements allow you to make decisions in your code based on boolean expressions.

```python
time = 14  # 24-hour format

if time < 12:
    print("Good morning!")
elif time < 18:
    print("Good afternoon!")
else:
    print("Good evening!")
```

Slide 3: Loops in Python

Loops let you repeat code blocks efficiently. Python provides two main types of loops: for and while. For loops are typically used with sequences, while while loops continue until a condition becomes false.

```python
# Print first 5 square numbers
for i in range(1, 6):
    square = i ** 2
    print(f"{i} squared is {square}")
```

Slide 4: Working with Lists

Lists are ordered, mutable sequences that can store multiple items of different types. They support various operations like appending, extending, and slicing.

```python
# Creating and manipulating lists
fruits = ["apple", "banana", "orange"]
fruits.append("grape")
fruits.insert(1, "mango")

print("Original list:", fruits)
print("Sliced list:", fruits[1:3])
```

Slide 5: Functions and Return Values

Functions are reusable blocks of code that perform specific tasks. They can accept parameters and return values, making code modular and maintainable.

```python
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

scores = [85, 92, 78, 90, 88]
avg = calculate_average(scores)
print(f"Average score: {avg}")
```

Slide 6: Real-Life Example - Temperature Converter

This practical example shows how to create a temperature converter between Celsius and Fahrenheit scales.

```python
def convert_temperature(value, scale):
    if scale.lower() == 'c':
        fahrenheit = (value * 9/5) + 32
        return f"{value}°C is {fahrenheit:.1f}°F"
    elif scale.lower() == 'f':
        celsius = (value - 32) * 5/9
        return f"{value}°F is {celsius:.1f}°C"

print(convert_temperature(25, 'c'))
print(convert_temperature(77, 'f'))
```

Slide 7: String Manipulation

Strings in Python are immutable sequences of characters that support various operations and methods for text processing.

```python
text = "Python Programming"
print(f"Uppercase: {text.upper()}")
print(f"Lowercase: {text.lower()}")
print(f"Split words: {text.split()}")
print(f"Replace: {text.replace('Python', 'Basic')}")
```

Slide 8: Error Handling with Try-Except

Error handling prevents program crashes by catching and handling exceptions gracefully.

```python
def safe_division(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return "Cannot divide by zero"
    except TypeError:
        return "Invalid input types"

print(safe_division(10, 2))
print(safe_division(10, 0))
```

Slide 9: Real-Life Example - Word Counter

This example demonstrates how to count word frequencies in a text document.

```python
def count_words(text):
    words = text.lower().split()
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    return word_count

sample_text = "The quick brown fox jumps over the lazy dog"
print(count_words(sample_text))
```

Slide 10: File Operations

Python provides simple ways to read from and write to files, essential for data processing and storage.

```python
# Writing and reading a file
with open('example.txt', 'w') as file:
    file.write('Hello, Python!\n')
    file.write('File operations are easy.')

with open('example.txt', 'r') as file:
    content = file.read()
    print(content)
```

Slide 11: List Comprehensions

List comprehensions provide a concise way to create lists based on existing sequences.

```python
# Generate squares of even numbers from 0 to 9
squares = [x**2 for x in range(10) if x % 2 == 0]
print(f"Squares of even numbers: {squares}")

# Create a list of vowels from a string
text = "Hello World"
vowels = [char for char in text if char.lower() in 'aeiou']
print(f"Vowels found: {vowels}")
```

Slide 12: Dictionary Comprehensions

Dictionary comprehensions allow for creating dictionaries using a compact syntax.

```python
# Create a dictionary of number:square pairs
squares_dict = {x: x**2 for x in range(5)}
print(f"Number to square mapping: {squares_dict}")

# Create a character frequency dictionary
text = "hello"
char_freq = {char: text.count(char) for char in text}
print(f"Character frequencies: {char_freq}")
```

Slide 13: Additional Resources

For further learning and advanced topics in Python programming, consider these peer-reviewed resources:

1.  "Python in Science Education" - arXiv:1905.10262
2.  "Teaching Programming with Python" - arXiv:2007.07012
3.  Official Python Documentation at python.org
4.  "Algorithmic Thinking and Problem Solving in Python" - arXiv:1911.02151

