## Python String Fundamentals Enhance Your Coding Skills
Slide 1: String Creation and Assignment

String creation in Python offers multiple approaches for declaring and initializing string variables. Understanding the fundamental syntax differences between single quotes, double quotes, and triple quotes enables developers to handle text data effectively while maintaining code readability.

```python
# Different ways to create strings
single_quoted = 'Python String'
double_quoted = "Python String"
triple_quoted = '''Multiple
line string'''

# String assignment and concatenation
name = "Alice"
greeting = "Hello, " + name
print(greeting)  # Output: Hello, Alice

# Raw string for escaping characters
raw_string = r"C:\Users\Documents"
print(raw_string)  # Output: C:\Users\Documents
```

Slide 2: String Methods for Case Manipulation

Python provides built-in string methods that facilitate case transformations, essential for text processing and standardization. These methods create new string objects rather than modifying the original, maintaining string immutability principles.

```python
# Case manipulation methods
text = "Python Programming"

upper_case = text.upper()
lower_case = text.lower()
title_case = text.title()
swapped_case = text.swapcase()

print(f"Original: {text}")        # Output: Python Programming
print(f"Upper: {upper_case}")     # Output: PYTHON PROGRAMMING
print(f"Lower: {lower_case}")     # Output: python programming
print(f"Title: {title_case}")     # Output: Python Programming
print(f"Swapped: {swapped_case}") # Output: pYTHON pROGRAMMING
```

Slide 3: String Indexing and Slicing

String indexing and slicing operations allow precise extraction of substrings using Python's powerful slice notation. Understanding zero-based indexing and negative indices is crucial for effective string manipulation.

```python
# String indexing and slicing
text = "Python Programming"

# Basic indexing
first_char = text[0]     # 'P'
last_char = text[-1]     # 'g'

# Slicing with different steps
substring = text[0:6]    # 'Python'
reverse = text[::-1]     # 'gnimmargorP nohtyP'
skip_chars = text[::2]   # 'Pto rgamn'

print(f"Original: {text}")
print(f"First character: {first_char}")
print(f"Last character: {last_char}")
print(f"Substring: {substring}")
print(f"Reversed: {reverse}")
```

Slide 4: String Formatting Techniques

Modern Python string formatting encompasses multiple approaches, from traditional %-formatting to f-strings introduced in Python 3.6+. Each method offers unique advantages for creating formatted text output.

```python
name = "Alice"
age = 25
height = 1.75

# %-formatting
old_style = "Name: %s, Age: %d, Height: %.2f" % (name, age, height)

# str.format() method
format_method = "Name: {}, Age: {}, Height: {:.2f}".format(name, age, height)

# f-strings (Python 3.6+)
f_string = f"Name: {name}, Age: {age}, Height: {height:.2f}"

# Format specifiers in f-strings
number = 42
binary = f"{number:b}"  # binary
hex_num = f"{number:x}" # hexadecimal
scientific = f"{height:e}"  # scientific notation

print(f_string)
print(f"Binary: {binary}, Hex: {hex_num}, Scientific: {scientific}")
```

Slide 5: String Search and Validation Methods

Python's comprehensive set of string methods enables efficient text searching, validation, and pattern matching without regular expressions. These built-in methods provide essential functionality for string analysis and manipulation.

```python
text = "Python Programming Language"
search_term = "Programming"

# Search methods
contains = search_term in text
position = text.find(search_term)
count_p = text.count('P')

# Validation methods
is_alpha = text.replace(" ", "").isalpha()
starts_with = text.startswith("Python")
ends_with = text.endswith("Language")

print(f"Contains '{search_term}': {contains}")  # True
print(f"Position of '{search_term}': {position}")  # 7
print(f"Count of 'P': {count_p}")  # 2
print(f"Is alphabetic: {is_alpha}")  # True
```

Slide 6: String Splitting and Joining

String manipulation often requires decomposing strings into components or combining multiple strings. Python's split() and join() methods provide powerful tools for these operations with customizable delimiters.

```python
# String splitting
text = "Python,Java,C++,JavaScript"
languages = text.split(',')

# Multi-delimiter splitting
complex_text = "Python;Java:C++,JavaScript"
all_langs = complex_text.replace(';', ',').replace(':', ',').split(',')

# String joining
delimiter = ' | '
joined_text = delimiter.join(languages)

# Line-based operations
multiline = """Line 1
Line 2
Line 3"""
lines = multiline.splitlines()

print(f"Split result: {languages}")
print(f"Joined result: {joined_text}")
print(f"Lines: {lines}")
```

Slide 7: Advanced String Operations

Advanced string operations enable complex text transformations through built-in methods that handle padding, alignment, and character replacement. These operations are crucial for formatting output and text processing.

```python
text = "python"

# Padding and alignment
left_pad = text.ljust(10, '*')    # 'python****'
right_pad = text.rjust(10, '*')   # '****python'
center_pad = text.center(10, '*')  # '**python**'

# Character replacement
mapped = text.maketrans('ptn', '123')
translated = text.translate(mapped)

# Strip operations
whitespace_text = "   python   "
stripped = whitespace_text.strip()
lstripped = whitespace_text.lstrip()
rstripped = whitespace_text.rstrip()

print(f"Left padded: '{left_pad}'")
print(f"Right padded: '{right_pad}'")
print(f"Centered: '{center_pad}'")
print(f"Translated: {translated}")
```

Slide 8: String Encoding and Decoding

Understanding string encoding and decoding is crucial for handling text data across different character encodings. Python provides robust support for various encodings, ensuring proper text handling in international contexts.

```python
# String encoding and decoding
text = "Hello, 世界"  # Unicode string

# Encoding to different formats
utf8_encoded = text.encode('utf-8')
utf16_encoded = text.encode('utf-16')
ascii_encoded = text.encode('ascii', errors='replace')

# Decoding back to string
utf8_decoded = utf8_encoded.decode('utf-8')
utf16_decoded = utf16_encoded.decode('utf-16')

print(f"Original: {text}")
print(f"UTF-8 encoded: {utf8_encoded}")
print(f"UTF-16 encoded: {utf16_encoded}")
print(f"Decoded back: {utf8_decoded}")
print(f"ASCII (replaced): {ascii_encoded}")
```

Slide 9: String Memory Management

Understanding string memory management in Python reveals how the interpreter handles string objects internally through string interning and the memory optimization techniques that improve performance in string-heavy applications.

```python
# String interning demonstration
str1 = "python"
str2 = "python"
str3 = "".join(['p', 'y', 't', 'h', 'o', 'n'])

# Identity comparison
is_same_object = str1 is str2
is_same_constructed = str1 is str3

# Memory address inspection
id_str1 = id(str1)
id_str2 = id(str2)
id_str3 = id(str3)

print(f"str1 is str2: {is_same_object}")  # True
print(f"str1 is str3: {is_same_constructed}")  # False
print(f"Memory addresses: {id_str1}, {id_str2}, {id_str3}")
```

Slide 10: Real-world Application: Text Analysis

This implementation demonstrates practical string manipulation in a text analysis context, including word frequency counting, sentence parsing, and basic text statistics calculation.

```python
def analyze_text(text):
    # Preprocess text
    cleaned_text = text.lower().strip()
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    words = cleaned_text.split()
    
    # Word frequency analysis
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Calculate statistics
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sentence_length = len(words) / len(sentences)
    
    return {
        'total_words': len(words),
        'unique_words': len(word_freq),
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'word_frequency': dict(sorted(word_freq.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:5])
    }

# Example usage
sample_text = """Python is a versatile programming language. 
It supports multiple programming paradigms. 
Python's simplicity makes it popular."""

results = analyze_text(sample_text)
for key, value in results.items():
    print(f"{key}: {value}")
```

Slide 11: String Regular Expression Integration

Regular expressions combined with string operations provide powerful text pattern matching and manipulation capabilities, essential for advanced text processing tasks.

```python
import re

def advanced_string_processing(text):
    # Pattern matching
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    
    # Find all matches
    emails = re.findall(email_pattern, text)
    urls = re.findall(url_pattern, text)
    
    # Replace patterns
    censored = re.sub(r'\b\w{4,}\b', 
                     lambda m: m.group(0)[0] + '*' * (len(m.group(0))-2) + 
                             m.group(0)[-1], 
                     text)
    
    # Split on multiple delimiters
    words = re.split(r'[;,\s]\s*', text)
    
    return {
        'emails': emails,
        'urls': urls,
        'censored_text': censored,
        'word_list': [w for w in words if w]
    }

# Example usage
sample_text = """Contact us at support@example.com or visit 
https://example.com. Alternative email: info@example.com"""

results = advanced_string_processing(sample_text)
for key, value in results.items():
    print(f"{key}:\n{value}\n")
```

Slide 12: String Performance Optimization

Understanding string performance characteristics enables developers to write more efficient code. This implementation demonstrates various techniques for optimizing string operations in performance-critical applications.

```python
import timeit
import sys

def performance_comparison():
    # String concatenation methods
    def concat_plus():
        result = ""
        for i in range(1000):
            result += str(i)
        return result

    def concat_join():
        return ''.join(str(i) for i in range(1000))
    
    def concat_list():
        parts = []
        for i in range(1000):
            parts.append(str(i))
        return ''.join(parts)

    # Measure execution time
    times = {
        'plus': timeit.timeit(concat_plus, number=100),
        'join': timeit.timeit(concat_join, number=100),
        'list': timeit.timeit(concat_list, number=100)
    }
    
    # Memory usage for different string operations
    base_str = "x" * 1000
    memory_usage = {
        'original': sys.getsizeof(base_str),
        'sliced': sys.getsizeof(base_str[10:900]),
        'multiplied': sys.getsizeof(base_str * 2)
    }
    
    return {'execution_times': times, 'memory_usage': memory_usage}

results = performance_comparison()
print("Execution times (seconds):")
for method, time in results['execution_times'].items():
    print(f"{method}: {time:.4f}")
print("\nMemory usage (bytes):")
for op, size in results['memory_usage'].items():
    print(f"{op}: {size}")
```

Slide 13: Real-world Application: Text Preprocessing Pipeline

This implementation showcases a comprehensive text preprocessing pipeline commonly used in natural language processing tasks, demonstrating practical string manipulation techniques.

```python
import re
from typing import List, Dict

class TextPreprocessor:
    def __init__(self):
        self.contractions = {
            "won't": "will not",
            "can't": "cannot",
            "I'm": "I am",
            "it's": "it is"
        }
    
    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def remove_special_chars(self, text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        return ' '.join(text.split())
    
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def process_text(self, text: str) -> Dict:
        # Pipeline execution
        expanded = self.expand_contractions(text)
        cleaned = self.remove_special_chars(expanded)
        normalized = self.normalize_whitespace(cleaned)
        tokens = self.tokenize(normalized)
        
        return {
            'original': text,
            'processed': normalized,
            'tokens': tokens,
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens))
        }

# Example usage
preprocessor = TextPreprocessor()
sample_text = """I'm really excited about Python! It's won't 
                let you down, can't wait to learn more..."""

results = preprocessor.process_text(sample_text)
for key, value in results.items():
    print(f"{key}: {value}")
```

Slide 14: Additional Resources

*   String Methods Documentation
    *   [https://docs.python.org/3/library/stdtypes.html#string-methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
*   String Processing Tutorial
    *   [https://realpython.com/python-strings/](https://realpython.com/python-strings/)
*   Advanced String Manipulation Techniques
    *   [https://www.python.org/dev/peps/pep-0292/](https://www.python.org/dev/peps/pep-0292/)
*   Performance Optimization Guidelines
    *   [https://wiki.python.org/moin/PythonSpeed/PerformanceTips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
*   String Encoding and Unicode
    *   [https://docs.python.org/3/howto/unicode.html](https://docs.python.org/3/howto/unicode.html)

