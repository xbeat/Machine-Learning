## Mastering Python Strings for Text Manipulation
Slide 1: Introduction to Python Strings

Python strings are sequences of characters, enclosed in single or double quotes. They are immutable, meaning their contents cannot be changed after creation. Strings are versatile and essential for text manipulation in Python.

```python
# Creating strings
single_quoted = 'Hello, World!'
double_quoted = "Python Strings"
multi_line = '''This is a
multi-line string'''

print(single_quoted)
print(double_quoted)
print(multi_line)
```

Slide 2: String Indexing and Slicing

Strings can be accessed using indexing and slicing. Indexing starts at 0 for the first character, and -1 for the last. Slicing allows you to extract substrings using a start:stop:step syntax.

```python
text = "Python Strings"

# Indexing
print(text[0])    # First character
print(text[-1])   # Last character

# Slicing
print(text[0:6])   # Characters from index 0 to 5
print(text[7:])    # Characters from index 7 to the end
print(text[::-1])  # Reverse the string
```

Slide 3: String Concatenation and Repetition

Strings can be combined using the + operator (concatenation) or repeated using the \* operator (repetition). These operations create new strings without modifying the originals.

```python
str1 = "Hello"
str2 = "World"

# Concatenation
combined = str1 + " " + str2
print(combined)

# Repetition
repeated = str1 * 3
print(repeated)
```

Slide 4: String Methods: Case Manipulation

Python provides methods to change the case of strings. These methods return new strings without modifying the original.

```python
text = "Python Strings"

# Uppercase
print(text.upper())

# Lowercase
print(text.lower())

# Title case
print(text.title())

# Capitalize
print(text.capitalize())
```

Slide 5: String Methods: Searching and Replacing

Python offers methods to search for substrings and replace text within strings. These operations are case-sensitive by default.

```python
text = "Python is amazing. Python is versatile."

# Find the first occurrence
print(text.find("Python"))

# Count occurrences
print(text.count("Python"))

# Replace occurrences
new_text = text.replace("Python", "Java")
print(new_text)
```

Slide 6: String Methods: Splitting and Joining

Splitting strings into lists and joining lists into strings are common operations in text processing. The split() method separates a string based on a delimiter, while join() combines list elements into a single string.

```python
# Splitting
sentence = "Python is a powerful programming language"
words = sentence.split()
print(words)

# Joining
joined = " ".join(words)
print(joined)

# Splitting with custom delimiter
csv_data = "apple,banana,cherry"
fruits = csv_data.split(",")
print(fruits)
```

Slide 7: String Formatting

Python provides multiple ways to format strings, including the % operator, the format() method, and f-strings (formatted string literals). F-strings are the most recent and often the most convenient method.

```python
name = "Alice"
age = 30

# % operator
print("My name is %s and I'm %d years old." % (name, age))

# format() method
print("My name is {} and I'm {} years old.".format(name, age))

# f-string (Python 3.6+)
print(f"My name is {name} and I'm {age} years old.")
```

Slide 8: String Methods: Stripping Whitespace

Removing leading and trailing whitespace from strings is a common task in data cleaning. Python provides methods to strip whitespace from the left, right, or both sides of a string.

```python
text = "   Python Strings   "

# Remove leading and trailing whitespace
print(text.strip())

# Remove leading whitespace
print(text.lstrip())

# Remove trailing whitespace
print(text.rstrip())

# Strip custom characters
custom_text = "***Python***"
print(custom_text.strip("*"))
```

Slide 9: String Methods: Checking String Properties

Python provides methods to check various properties of strings, such as whether they contain only alphabetic characters, digits, or are in a specific case.

```python
# Check if string is alphabetic
print("Python".isalpha())

# Check if string is numeric
print("12345".isnumeric())

# Check if string is alphanumeric
print("Python3".isalnum())

# Check if string is lowercase
print("python".islower())

# Check if string is uppercase
print("PYTHON".isupper())
```

Slide 10: Real-Life Example: Name Formatting

Let's create a function to format names for a user directory. This example demonstrates the practical use of string methods for text processing.

```python
def format_name(first_name, last_name):
    # Capitalize first letter, convert rest to lowercase
    formatted_first = first_name[0].upper() + first_name[1:].lower()
    formatted_last = last_name[0].upper() + last_name[1:].lower()
    
    return f"{formatted_last}, {formatted_first}"

# Test the function
print(format_name("jOHn", "DOE"))
print(format_name("aLiCe", "SMITH"))
```

Slide 11: Real-Life Example: Simple Text Analysis

Let's create a function to analyze text, counting words and calculating the average word length. This example showcases string splitting and list comprehension.

```python
def analyze_text(text):
    # Remove punctuation and convert to lowercase
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum() or char.isspace())
    
    # Split into words
    words = cleaned_text.split()
    
    # Count words
    word_count = len(words)
    
    # Calculate average word length
    avg_length = sum(len(word) for word in words) / word_count
    
    return f"Word count: {word_count}, Average word length: {avg_length:.2f}"

# Test the function
sample_text = "Python strings are powerful tools for text manipulation!"
print(analyze_text(sample_text))
```

Slide 12: String Encoding and Decoding

Strings in Python 3 are Unicode by default. However, when working with external systems or files, you may need to encode strings to bytes or decode bytes to strings.

```python
# String to bytes
text = "Pythön"
encoded = text.encode('utf-8')
print(encoded)

# Bytes to string
decoded = encoded.decode('utf-8')
print(decoded)

# Error handling
try:
    wrong_decode = encoded.decode('ascii')
except UnicodeDecodeError as e:
    print(f"Error: {e}")
```

Slide 13: String Interpolation with f-strings

F-strings, introduced in Python 3.6, offer a concise and readable way to embed expressions inside string literals. They can include variables, expressions, and even function calls.

```python
import math

radius = 5
pi = math.pi

# Basic variable interpolation
print(f"The radius is {radius}")

# Expression in f-string
print(f"The area of the circle is {pi * radius**2:.2f}")

# Calling functions in f-string
print(f"The sine of 30 degrees is {math.sin(math.radians(30)):.4f}")

# Formatting options
for x in range(1, 11):
    print(f"{x:2d} {x*x:3d} {x*x*x:4d}")
```

Slide 14: Additional Resources

For those interested in diving deeper into Python strings and text processing, here are some valuable resources:

1.  Python's official documentation on string methods: [https://docs.python.org/3/library/stdtypes.html#string-methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
2.  "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper: [https://www.nltk.org/book/](https://www.nltk.org/book/)
3.  "Text Analysis in Python for Social Scientists" by Minsuk Seo, Philipp Singer, and Stephan Giest: [https://arxiv.org/abs/2104.01385](https://arxiv.org/abs/2104.01385)

These resources provide comprehensive coverage of string manipulation techniques and their applications in various domains.

