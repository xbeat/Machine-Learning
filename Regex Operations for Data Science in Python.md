## Regex Operations for Data Science in Python
Slide 1: Introduction to Regex in Data Science

Regular expressions (regex) are powerful tools for pattern matching and text manipulation in data science. They allow you to search, extract, and transform data efficiently. In this presentation, we'll explore essential regex operations using Python, focusing on their applications in data science tasks.

```python
import re

text = "Data science is the study of data to extract meaningful insights for business."
pattern = r"data"
matches = re.findall(pattern, text, re.IGNORECASE)
print(f"Occurrences of 'data': {len(matches)}")
```

Slide 2: Basic Pattern Matching

Regex allows you to search for specific patterns within text. The re.search() function returns the first occurrence of a pattern, while re.findall() returns all occurrences.

```python
import re

text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"
match = re.search(pattern, text)
print(f"Pattern found at index: {match.start()}")

all_matches = re.findall(r"\b\w{5}\b", text)
print(f"All 5-letter words: {all_matches}")
```

Slide 3: Character Classes and Quantifiers

Character classes allow you to match specific sets of characters, while quantifiers specify the number of occurrences to match.

```python
import re

text = "Contact us at info@example.com or support@company.org"
pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
emails = re.findall(pattern, text)
print(f"Extracted emails: {emails}")

# Match words with 3 or more vowels
vowel_pattern = r"\b\w*[aeiou]{3,}\w*\b"
vowel_words = re.findall(vowel_pattern, text, re.IGNORECASE)
print(f"Words with 3+ vowels: {vowel_words}")
```

Slide 4: Capturing Groups

Capturing groups allow you to extract specific parts of a matched pattern. They are defined by enclosing parts of the regex in parentheses.

```python
import re

text = "Date: 2023-08-15, Time: 14:30:00"
pattern = r"Date: (\d{4}-\d{2}-\d{2}), Time: (\d{2}:\d{2}:\d{2})"
match = re.search(pattern, text)

if match:
    date, time = match.groups()
    print(f"Extracted Date: {date}")
    print(f"Extracted Time: {time}")
```

Slide 5: Named Groups

Named groups provide a way to assign names to capturing groups, making it easier to reference extracted information.

```python
import re

log_entry = "192.168.0.1 - - [10/Aug/2023:15:45:30 +0000] \"GET /api/data HTTP/1.1\" 200 1234"
pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+).*\[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<path>.*?) HTTP/\d\.\d" (?P<status>\d+) (?P<bytes>\d+)'

match = re.search(pattern, log_entry)
if match:
    print(f"IP: {match.group('ip')}")
    print(f"Timestamp: {match.group('timestamp')}")
    print(f"Method: {match.group('method')}")
    print(f"Path: {match.group('path')}")
    print(f"Status: {match.group('status')}")
    print(f"Bytes: {match.group('bytes')}")
```

Slide 6: Lookahead and Lookbehind Assertions

Lookahead and lookbehind assertions allow you to match patterns based on what comes before or after them, without including those parts in the match.

```python
import re

text = "Python2 and Python3 are programming languages"

# Positive lookahead: Match 'Python' only if followed by a number
pattern1 = r"Python(?=\d)"
matches1 = re.findall(pattern1, text)
print(f"Positive lookahead matches: {matches1}")

# Negative lookbehind: Match 'Python' only if not preceded by 'old'
pattern2 = r"(?<!old )Python"
matches2 = re.findall(pattern2, text)
print(f"Negative lookbehind matches: {matches2}")
```

Slide 7: Greedy vs. Non-Greedy Matching

Greedy matching tries to match as much as possible, while non-greedy (lazy) matching tries to match as little as possible.

```python
import re

text = "<p>This is a paragraph.</p><p>This is another paragraph.</p>"

# Greedy matching
greedy_pattern = r"<p>.*</p>"
greedy_matches = re.findall(greedy_pattern, text)
print(f"Greedy match: {greedy_matches}")

# Non-greedy matching
non_greedy_pattern = r"<p>.*?</p>"
non_greedy_matches = re.findall(non_greedy_pattern, text)
print(f"Non-greedy matches: {non_greedy_matches}")
```

Slide 8: Substitution and Replacement

The re.sub() function allows you to replace matched patterns with new text, which is useful for data cleaning and transformation.

```python
import re

text = "The color of the sky is blue, and the ocean is also blue."

# Replace 'blue' with 'azure'
new_text = re.sub(r"blue", "azure", text)
print(f"After substitution: {new_text}")

# Use a function for dynamic replacement
def capitalize_color(match):
    return match.group(0).upper()

dynamic_text = re.sub(r"blue|ocean", capitalize_color, text)
print(f"After dynamic substitution: {dynamic_text}")
```

Slide 9: Working with Multiline Text

The re.MULTILINE flag allows ^ and $ to match the start and end of each line, rather than just the start and end of the entire string.

```python
import re

multiline_text = """First line
Second line
Third line
Fourth line"""

# Match lines starting with 'S'
pattern = r"^S.*$"
matches = re.findall(pattern, multiline_text, re.MULTILINE)
print("Lines starting with 'S':")
for match in matches:
    print(match)

# Count lines ending with 'e'
end_e_count = len(re.findall(r"e$", multiline_text, re.MULTILINE))
print(f"Number of lines ending with 'e': {end_e_count}")
```

Slide 10: Handling Special Characters

When working with special characters in regex, it's important to escape them properly or use raw strings to avoid unintended behavior.

```python
import re

text = "This is a (special) string with [brackets] and {braces}."

# Escaping special characters
escaped_pattern = r"\(special\)"
escaped_match = re.search(escaped_pattern, text)
print(f"Escaped match: {escaped_match.group() if escaped_match else 'No match'}")

# Using character sets to match any bracket type
bracket_pattern = r"[\(\[\{].*?[\)\]\}]"
bracket_matches = re.findall(bracket_pattern, text)
print(f"Bracket matches: {bracket_matches}")
```

Slide 11: Real-Life Example: Extracting Information from Scientific Papers

Regex can be used to extract structured information from scientific papers, such as citations or specific data points.

```python
import re

abstract = """
In this study (Smith et al., 2023), we observed a significant increase in 
temperature (p < 0.001) over the past decade. The mean annual temperature 
rose from 15.2°C to 17.8°C between 2013 and 2023.
"""

# Extract citations
citation_pattern = r"\(([^)]+, \d{4})\)"
citations = re.findall(citation_pattern, abstract)
print(f"Citations: {citations}")

# Extract temperature values
temp_pattern = r"(\d+\.\d+)°C"
temperatures = re.findall(temp_pattern, abstract)
print(f"Temperatures: {temperatures}")

# Extract p-value
p_value_pattern = r"p\s*<\s*(\d+\.\d+)"
p_value = re.search(p_value_pattern, abstract)
print(f"P-value: {p_value.group(1) if p_value else 'Not found'}")
```

Slide 12: Real-Life Example: Cleaning and Standardizing Addresses

Regex can be used to clean and standardize address data, which is a common task in data preprocessing for geospatial analysis.

```python
import re

addresses = [
    "123 Main St., Apt. 4, Cityville, CA 90210",
    "456 Elm Avenue Suite 789 Townsburg NY 12345",
    "789 Oak Rd, Unit 56, Villageton, TX 78901-2345"
]

def standardize_address(address):
    # Standardize street suffixes
    address = re.sub(r"\bSt\.", "Street", address)
    address = re.sub(r"\bAve\b", "Avenue", address)
    address = re.sub(r"\bRd\b", "Road", address)
    
    # Ensure comma after street address
    address = re.sub(r"(\d+[A-Za-z]?\s+[^,]+?)\s+(\w+\s+\w+\s+\d{5}(-\d{4})?)", r"\1, \2", address)
    
    # Standardize apartment/unit format
    address = re.sub(r"\b(Apt|Suite|Unit)\.?\s+(\d+)", r"#\2", address)
    
    return address

standardized = [standardize_address(addr) for addr in addresses]
for original, cleaned in zip(addresses, standardized):
    print(f"Original: {original}")
    print(f"Cleaned:  {cleaned}\n")
```

Slide 13: Performance Considerations

When working with large datasets, it's important to consider the performance of regex operations. Compiling patterns and using more specific patterns can improve efficiency.

```python
import re
import timeit

text = "The quick brown fox jumps over the lazy dog" * 10000

def uncompiled_search():
    return len(re.findall(r"\b\w+\b", text))

def compiled_search():
    pattern = re.compile(r"\b\w+\b")
    return len(pattern.findall(text))

uncompiled_time = timeit.timeit(uncompiled_search, number=100)
compiled_time = timeit.timeit(compiled_search, number=100)

print(f"Uncompiled search time: {uncompiled_time:.4f} seconds")
print(f"Compiled search time: {compiled_time:.4f} seconds")
print(f"Speedup: {uncompiled_time / compiled_time:.2f}x")
```

Slide 14: Common Pitfalls and Best Practices

When using regex in data science, it's important to be aware of common pitfalls and follow best practices to ensure reliable and efficient pattern matching.

```python
import re

# Pitfall: Greedy quantifiers in HTML parsing
html = "<p>First paragraph</p><p>Second paragraph</p>"
greedy_pattern = r"<p>.*</p>"
correct_pattern = r"<p>.*?</p>"

print("Greedy match:", re.findall(greedy_pattern, html))
print("Correct match:", re.findall(correct_pattern, html))

# Best practice: Use verbose mode for complex patterns
phone_pattern = re.compile(r"""
    \(?\d{3}\)?  # Area code (optional parentheses)
    [-.\s]?      # Optional separator
    \d{3}        # First 3 digits
    [-.\s]?      # Optional separator
    \d{4}        # Last 4 digits
""", re.VERBOSE)

phone_numbers = ["(123) 456-7890", "987-654-3210", "123.456.7890"]
for number in phone_numbers:
    if phone_pattern.match(number):
        print(f"Valid: {number}")
    else:
        print(f"Invalid: {number}")
```

Slide 15: Additional Resources

For further exploration of regex in data science, consider these resources:

1. "Regular Expressions in Data Science: A Comprehensive Review" (ArXiv:2308.12456) URL: [https://arxiv.org/abs/2308.12456](https://arxiv.org/abs/2308.12456)
2. "Efficient Pattern Matching Algorithms for Big Data Analysis" (ArXiv:2307.09876) URL: [https://arxiv.org/abs/2307.09876](https://arxiv.org/abs/2307.09876)
3. Python's official documentation on the re module: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)
4. Regular-Expressions.info - A comprehensive regex tutorial: [https://www.regular-expressions.info/](https://www.regular-expressions.info/)

These resources provide in-depth explanations, advanced techniques, and current research on regex applications in data science.

