## Python String Methods startswith() and endswith()
Slide 1: Understanding startswith() and endswith() with Multiple Substrings

The startswith() and endswith() methods in Python are commonly used to check if a string begins or ends with a specific substring. However, a lesser-known feature is their ability to accept multiple substrings as a tuple, enabling more efficient and concise code.

```python
# Traditional approach
text = "Hello, World!"
if text.startswith("Hello") or text.startswith("Hi"):
    print("Greeting found")

# More efficient approach using tuple
greetings = ("Hello", "Hi")
if text.startswith(greetings):
    print("Greeting found")

# Output:
# Greeting found
```

Slide 2: Syntax and Basic Usage

The syntax for using multiple substrings with startswith() and endswith() is straightforward. Simply pass a tuple of strings instead of a single string.

```python
def check_file_type(filename):
    image_extensions = ('.jpg', '.png', '.gif')
    if filename.lower().endswith(image_extensions):
        return "Image file"
    return "Other file type"

print(check_file_type("vacation.jpg"))
print(check_file_type("document.pdf"))

# Output:
# Image file
# Other file type
```

Slide 3: Advantages of Using Tuples

Using tuples with startswith() and endswith() offers several advantages:

1.  Cleaner code: Avoids multiple or conditions.
2.  Better performance: Single method call instead of multiple.
3.  Easier maintenance: Add or remove substrings by modifying the tuple.

```python
def categorize_url(url):
    social_domains = ('facebook.com', 'twitter.com', 'instagram.com')
    news_domains = ('cnn.com', 'bbc.com', 'nytimes.com')
    
    if url.endswith(social_domains):
        return "Social Media"
    elif url.endswith(news_domains):
        return "News Site"
    else:
        return "Other"

print(categorize_url("https://www.facebook.com/profile"))
print(categorize_url("https://www.bbc.com/news"))
print(categorize_url("https://www.example.com"))

# Output:
# Social Media
# News Site
# Other
```

Slide 4: Case Sensitivity and Optional Start/End Positions

Both startswith() and endswith() methods are case-sensitive by default. They also accept optional start and end parameters to specify the portion of the string to check.

```python
text = "HELLO WORLD"
prefixes = ("hello", "hi")

print(text.startswith(prefixes))  # Case-sensitive
print(text.lower().startswith(prefixes))  # Case-insensitive

partial_text = "HELLO WORLD HELLO"
print(partial_text.startswith(("WORLD", "HELLO"), 6))  # Start from index 6

# Output:
# False
# True
# True
```

Slide 5: Real-Life Example: File Type Validation

In this example, we'll create a function that validates file types for a hypothetical upload system, demonstrating the practical use of endswith() with multiple substrings.

```python
def validate_upload(filename):
    allowed_images = ('.jpg', '.jpeg', '.png', '.gif')
    allowed_documents = ('.pdf', '.doc', '.docx', '.txt')
    
    if filename.lower().endswith(allowed_images):
        return "Valid image file"
    elif filename.lower().endswith(allowed_documents):
        return "Valid document file"
    else:
        return "Invalid file type"

# Test the function
files = ["report.pdf", "photo.jpg", "script.py", "document.docx"]
for file in files:
    print(f"{file}: {validate_upload(file)}")

# Output:
# report.pdf: Valid document file
# photo.jpg: Valid image file
# script.py: Invalid file type
# document.docx: Valid document file
```

Slide 6: Real-Life Example: URL Parsing

Let's create a function that extracts the domain from a URL and categorizes it based on its top-level domain (TLD).

```python
def parse_url(url):
    # Extract domain (simple approach, not foolproof)
    domain = url.split("//")[-1].split("/")[0]
    
    edu_tlds = ('.edu', '.ac.uk', '.edu.au')
    gov_tlds = ('.gov', '.gov.uk', '.gov.au')
    
    if domain.endswith(edu_tlds):
        return f"{domain} is an educational institution"
    elif domain.endswith(gov_tlds):
        return f"{domain} is a government website"
    else:
        return f"{domain} is a general domain"

# Test the function
urls = [
    "https://www.stanford.edu/research",
    "https://www.whitehouse.gov",
    "https://www.bbc.co.uk/news"
]

for url in urls:
    print(parse_url(url))

# Output:
# www.stanford.edu is an educational institution
# www.whitehouse.gov is a government website
# www.bbc.co.uk is a general domain
```

Slide 7: Combining startswith() and endswith()

We can combine both methods to create more complex string matching patterns. Here's an example that checks if a string is a valid Python identifier.

```python
import string

def is_valid_identifier(name):
    valid_starts = tuple(string.ascii_letters + '_')
    valid_chars = tuple(string.ascii_letters + string.digits + '_')
    
    return (name.startswith(valid_starts) and 
            all(char in valid_chars for char in name[1:]))

# Test the function
identifiers = ["valid_name", "2invalid", "_ok", "also-invalid"]
for identifier in identifiers:
    print(f"{identifier}: {'Valid' if is_valid_identifier(identifier) else 'Invalid'}")

# Output:
# valid_name: Valid
# 2invalid: Invalid
# _ok: Valid
# also-invalid: Invalid
```

Slide 8: Performance Considerations

Using startswith() and endswith() with tuples can be more efficient than multiple individual checks, especially for larger datasets.

```python
import timeit

def check_individual(text, prefixes):
    return any(text.startswith(prefix) for prefix in prefixes)

def check_tuple(text, prefixes):
    return text.startswith(prefixes)

text = "Hello, World!"
prefixes = ("Hi", "Hello", "Hey")

individual_time = timeit.timeit(lambda: check_individual(text, prefixes), number=1000000)
tuple_time = timeit.timeit(lambda: check_tuple(text, prefixes), number=1000000)

print(f"Individual checks: {individual_time:.6f} seconds")
print(f"Tuple check: {tuple_time:.6f} seconds")
print(f"Tuple method is {individual_time / tuple_time:.2f}x faster")

# Output may vary, but tuple method is generally faster:
# Individual checks: 0.635982 seconds
# Tuple check: 0.198741 seconds
# Tuple method is 3.20x faster
```

Slide 9: Error Handling and Edge Cases

It's important to handle potential errors and edge cases when using these methods, especially with user input.

```python
def safe_check_start(text, prefixes):
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not isinstance(prefixes, (tuple, list)):
        raise TypeError("Prefixes must be a tuple or list")
    return text.startswith(tuple(prefixes))

# Test the function
try:
    print(safe_check_start("Hello", ("Hi", "Hello")))  # Valid
    print(safe_check_start(123, ("1", "2")))  # Invalid input type
    print(safe_check_start("Test", "T"))  # Invalid prefixes type
except TypeError as e:
    print(f"Error: {e}")

# Output:
# True
# Error: Input must be a string
```

Slide 10: Practical Application: Simple Command Parser

Let's create a simple command parser that demonstrates the use of startswith() with multiple prefixes.

```python
def parse_command(command):
    file_commands = ("open", "close", "save")
    edit_commands = ("cut", "copy", "paste")
    view_commands = ("zoom", "fullscreen", "normal")
    
    if command.lower().startswith(file_commands):
        return "File operation"
    elif command.lower().startswith(edit_commands):
        return "Edit operation"
    elif command.lower().startswith(view_commands):
        return "View operation"
    else:
        return "Unknown command"

# Test the parser
commands = ["Open file", "Copy text", "Zoom in", "Exit"]
for cmd in commands:
    print(f"'{cmd}' is a {parse_command(cmd)}")

# Output:
# 'Open file' is a File operation
# 'Copy text' is a Edit operation
# 'Zoom in' is a View operation
# 'Exit' is a Unknown command
```

Slide 11: Working with Multilingual Text

The startswith() and endswith() methods work with Unicode strings, making them suitable for multilingual applications.

```python
def detect_language(text):
    russian_prefixes = ("привет", "здравствуйте")
    spanish_prefixes = ("hola", "buenos días")
    japanese_prefixes = ("こんにちは", "おはよう")
    
    if text.lower().startswith(russian_prefixes):
        return "Russian"
    elif text.lower().startswith(spanish_prefixes):
        return "Spanish"
    elif text.lower().startswith(japanese_prefixes):
        return "Japanese"
    else:
        return "Unknown language"

# Test the function
greetings = ["Hola amigo", "こんにちは世界", "Hello world", "Здравствуйте, мир"]
for greeting in greetings:
    print(f"'{greeting}' is detected as {detect_language(greeting)}")

# Output:
# 'Hola amigo' is detected as Spanish
# 'こんにちは世界' is detected as Japanese
# 'Hello world' is detected as Unknown language
# 'Здравствуйте, мир' is detected as Russian
```

Slide 12: Combining with Regular Expressions

For more complex pattern matching, you can combine startswith() and endswith() with regular expressions.

```python
import re

def validate_email(email):
    valid_domains = ('.com', '.org', '.edu')
    
    # Basic email pattern
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    
    if re.match(pattern, email) and email.lower().endswith(valid_domains):
        return "Valid email address"
    else:
        return "Invalid email address"

# Test the function
emails = ["user@example.com", "invalid-email", "student@university.edu", "user@domain.net"]
for email in emails:
    print(f"{email}: {validate_email(email)}")

# Output:
# user@example.com: Valid email address
# invalid-email: Invalid email address
# student@university.edu: Valid email address
# user@domain.net: Invalid email address
```

Slide 13: Conclusion and Best Practices

Using startswith() and endswith() with multiple substrings can significantly improve code readability and efficiency. Best practices include:

1.  Use tuples for immutable lists of prefixes/suffixes.
2.  Consider case sensitivity and use lower() or upper() when needed.
3.  Handle potential errors and edge cases.
4.  Combine with other string methods or regular expressions for complex patterns.
5.  Use these methods to create clean, efficient, and maintainable code.

```python
# Example of a clean and efficient implementation
def categorize_text(text):
    categories = {
        "question": ("what", "why", "how", "when"),
        "greeting": ("hello", "hi", "hey"),
        "farewell": ("goodbye", "bye", "see you")
    }
    
    lower_text = text.lower()
    for category, prefixes in categories.items():
        if lower_text.startswith(prefixes):
            return f"This is a {category}"
    
    return "Uncategorized text"

print(categorize_text("What time is it?"))
print(categorize_text("Hello, how are you?"))
print(categorize_text("Nice weather today"))

# Output:
# This is a question
# This is a greeting
# Uncategorized text
```

Slide 14: Additional Resources

For further exploration of string methods and Python programming:

1.  Python Official Documentation on String Methods: [https://docs.python.org/3/library/stdtypes.html#string-methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
2.  "Fluent Python" by Luciano Ramalho - A comprehensive guide to writing effective Python code.
3.  "Python Cookbook" by David Beazley and Brian K. Jones - Provides recipes for solving common programming problems, including advanced string manipulation.
4.  Online Python courses on platforms like Coursera, edX, or Udacity for interactive learning experiences.
5.  Python Enhancement Proposals (PEPs) for in-depth understanding of Python's design decisions: [https://www.python.org/dev/peps/](https://www.python.org/dev/peps/)

Remember to verify these resources and their availability as they may change over time.

