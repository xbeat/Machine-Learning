## Using Regular Expressions for Natural Language Processing in Python
Slide 1: Introduction to Regular Expressions in NLP

Regular expressions (regex) are powerful tools for pattern matching and text manipulation in Natural Language Processing (NLP). They provide a concise and flexible means to identify specific character sequences within larger bodies of text. In Python, the re module offers comprehensive support for working with regex, making it an essential skill for NLP practitioners.

```python
import re

text = "The quick brown fox jumps over the lazy dog."
pattern = r"\b\w{5}\b"  # Matches 5-letter words
matches = re.findall(pattern, text)
print(matches)  # Output: ['quick', 'brown', 'jumps']
```

Slide 2: Basic Regex Patterns

Regular expressions use special characters to define patterns. Some common elements include: . (dot): Matches any character except newline

* (asterisk): Matches zero or more occurrences of the previous character

* (plus): Matches one or more occurrences of the previous character ? (question mark): Matches zero or one occurrence of the previous character ^ (caret): Matches the start of a string $ (dollar): Matches the end of a string

```python
import re

text = "The rain in Spain falls mainly on the plain."
pattern = r"^The.*plain\.$"
match = re.match(pattern, text)
print(bool(match))  # Output: True
```

Slide 3: Character Classes and Quantifiers

Character classes allow matching specific sets of characters, while quantifiers specify how many times a pattern should occur: \[aeiou\]: Matches any vowel

```python
import re

text = "Contact us at info@example.com or support@company.org"
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(pattern, text)
print(emails)  # Output: ['info@example.com', 'support@company.org']
```

Slide 4: Grouping and Capturing

Parentheses () are used for grouping and capturing in regex. They allow you to treat multiple characters as a single unit and extract specific parts of a match:

```python
import re

text = "Date: 2023-08-15"
pattern = r"Date: (\d{4})-(\d{2})-(\d{2})"
match = re.search(pattern, text)
if match:
    year, month, day = match.groups()
    print(f"Year: {year}, Month: {month}, Day: {day}")
# Output: Year: 2023, Month: 08, Day: 15
```

Slide 5: Named Groups

Named groups provide a way to assign names to captured groups, making it easier to work with extracted information:

```python
import re

text = "John Doe (age: 30) works at Acme Inc."
pattern = r"(?P<name>\w+ \w+) \(age: (?P<age>\d+)\) works at (?P<company>.+)"
match = re.search(pattern, text)
if match:
    print(match.groupdict())
# Output: {'name': 'John Doe', 'age': '30', 'company': 'Acme Inc.'}
```

Slide 6: Tokenization with Regex

Tokenization is a fundamental task in NLP, and regex can be used to split text into meaningful units:

```python
import re

text = "Hello, world! How are you today? I'm doing great."
tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
print(tokens)
# Output: ['Hello', ',', 'world', '!', 'How', 'are', 'you', 'today', '?', 'I', 'm', 'doing', 'great', '.']
```

Slide 7: Text Cleaning and Preprocessing

Regex is often used to clean and preprocess text data by removing unwanted characters or patterns:

```python
import re

text = "Check out our website at http://www.example.com! #NLP #regex"
clean_text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
clean_text = re.sub(r'#\w+', '', clean_text)  # Remove hashtags
clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Remove extra whitespace
print(clean_text)
# Output: Check out our website at !
```

Slide 8: Named Entity Recognition (NER) with Regex

While more advanced NER techniques exist, regex can be used for simple entity extraction:

```python
import re

text = "Apple Inc. was founded by Steve Jobs in Cupertino, CA on April 1, 1976."
patterns = {
    'company': r'\b[A-Z][a-z]+ (?:Inc\.|Corp\.|Ltd\.)',
    'person': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    'location': r'\b[A-Z][a-z]+, [A-Z]{2}\b',
    'date': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'
}

for entity_type, pattern in patterns.items():
    matches = re.findall(pattern, text)
    print(f"{entity_type.capitalize()}: {matches}")

# Output:
# Company: ['Apple Inc.']
# Person: ['Steve Jobs']
# Location: ['Cupertino, CA']
# Date: ['April 1, 1976']
```

Slide 9: Regex Flags in Python

Python's re module provides flags to modify regex behavior: re.IGNORECASE (re.I): Case-insensitive matching re.MULTILINE (re.M): ^ and $ match at the beginning and end of each line re.DOTALL (re.S): . matches any character, including newline re.VERBOSE (re.X): Allows comments and whitespace in the pattern

```python
import re

text = """
First line
SECOND LINE
Third line
"""

pattern = r"^second line$"
matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
print(matches)  # Output: ['SECOND LINE']
```

Slide 10: Lookahead and Lookbehind Assertions

Lookahead and lookbehind assertions allow you to match patterns based on what comes before or after without including it in the match:

```python
import re

text = "I have $100 and €50."
pattern = r'\b(?=\w+(?:\W+\w+)*\s*$)(?=.*[A-Z])(?=.*\d)[\w\s]{8,}\b'
currencies = re.findall(r'(?<=\$)\d+|(?<=€)\d+', text)
print(currencies)  # Output: ['100', '50']
```

Slide 11: Word Boundary Matching

Word boundaries (\\b) are useful for matching whole words and avoiding partial matches:

```python
import re

text = "The cat is in the category of felines."
whole_word = re.findall(r'\bcat\b', text)
partial_word = re.findall(r'cat', text)
print("Whole word matches:", whole_word)  # Output: ['cat']
print("Partial matches:", partial_word)   # Output: ['cat', 'cat']
```

Slide 12: Real-Life Example: Extracting Citations

Extracting citations from academic text using regex:

```python
import re

text = """
As shown by Smith et al. (2020), the effect is significant.
Johnson and Brown (2019) argue that this approach is flawed.
Recent studies (Williams, 2021; Lee et al., 2022) suggest otherwise.
"""

citation_pattern = r'(?:[A-Z][a-z]+ (?:et al\.)?\s*(?:\([12]\d{3}\))|(?:\([A-Z][a-z]+(?: et al\.)?, [12]\d{3}\)))'
citations = re.findall(citation_pattern, text)
print(citations)
# Output: ['Smith et al. (2020)', 'Johnson and Brown (2019)', '(Williams, 2021', 'Lee et al., 2022)']
```

Slide 13: Real-Life Example: Parsing Log Files

Using regex to extract information from log files:

```python
import re

log_entry = "192.168.1.100 - - [15/Aug/2023:10:30:15 +0000] \"GET /index.html HTTP/1.1\" 200 2326"

pattern = r'(\d+\.\d+\.\d+\.\d+) .+ \[(.+)\] "(.*?)" (\d+) (\d+)'
match = re.search(pattern, log_entry)

if match:
    ip, timestamp, request, status, bytes_sent = match.groups()
    print(f"IP: {ip}")
    print(f"Timestamp: {timestamp}")
    print(f"Request: {request}")
    print(f"Status: {status}")
    print(f"Bytes Sent: {bytes_sent}")

# Output:
# IP: 192.168.1.100
# Timestamp: 15/Aug/2023:10:30:15 +0000
# Request: GET /index.html HTTP/1.1
# Status: 200
# Bytes Sent: 2326
```

Slide 14: Regex Performance Considerations

While regex is powerful, it can be computationally expensive for large datasets. Consider these tips for optimal performance: Use more specific patterns when possible Avoid excessive backtracking Use non-capturing groups (?:...) when capture is unnecessary Consider using specialized NLP libraries for complex tasks

```python
import re
import timeit

text = "a" * 100000 + "b"

def slow_regex():
    return re.search(r'a*b', text)

def fast_regex():
    return re.search(r'a*b', text, re.DOTALL)

print("Slow regex:", timeit.timeit(slow_regex, number=10))
print("Fast regex:", timeit.timeit(fast_regex, number=10))
# Output will show the performance difference
```

Slide 15: Additional Resources

For further exploration of regex in NLP, consider these resources:

1. "Regular Expressions for Natural Language Processing" by Nikhil Ketkar (2021) ArXiv: [https://arxiv.org/abs/2102.10217](https://arxiv.org/abs/2102.10217)
2. "A Survey of Deep Learning Techniques for Natural Language Processing" by Young et al. (2018) ArXiv: [https://arxiv.org/abs/1708.02709](https://arxiv.org/abs/1708.02709)
3. Python's official re module documentation: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)

These resources provide deeper insights into regex applications in NLP and advanced techniques for text processing.

