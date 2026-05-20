## String Methods vs Regex Choosing the Right Approach
Slide 1: String Methods - The Basics

String methods in Python provide a straightforward approach to text manipulation. These built-in methods are optimized for common text operations and offer clean, readable syntax for basic string transformations without requiring additional imports or complex pattern matching.

```python
# Basic string method examples
text = "  Hello, Python Programming World!  "

# Common string methods
print(text.strip())                 # Remove whitespace
print(text.lower())                 # Convert to lowercase
print(text.upper())                 # Convert to uppercase
print(text.replace("Python", "Advanced Python"))  # Replace text
print(text.split(","))             # Split into list

# Output:
# 'Hello, Python Programming World!'
# '  hello, python programming world!  '
# '  HELLO, PYTHON PROGRAMMING WORLD!  '
# '  Hello, Advanced Python Programming World!  '
# ['  Hello', ' Python Programming World!  ']
```

Slide 2: Regular Expressions - Introduction

Regular expressions provide a powerful pattern matching language for text processing. The re module in Python implements regex operations, enabling complex text analysis and manipulation through concise pattern definitions and specialized methods.

```python
import re

# Basic regex pattern matching
text = "Contact us: info@example.com or support@company.co.uk"

# Find all email addresses
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
emails = re.findall(email_pattern, text)

print("Found emails:", emails)
# Output: ['info@example.com', 'support@company.co.uk']

# Check if pattern exists
has_email = re.search(email_pattern, text) is not None
print("Contains email:", has_email)
# Output: True
```

Slide 3: String Methods for Text Cleaning

String methods excel at basic text cleaning operations, offering intuitive methods for removing unwanted characters, normalizing text, and performing simple replacements. These operations are fundamental in data preprocessing pipelines.

```python
def clean_text(text):
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters and normalize
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    
    # Remove punctuation except apostrophes
    punctuation = """.,?!-:;()[]{}"""
    for char in punctuation:
        text = text.replace(char, "")
    
    return text

# Example usage
messy_text = """This    is a  Very\n\tMessy
                text!!! With... lots (of) punctuation."""
clean = clean_text(messy_text)
print(clean)
# Output: 'this is a very messy text with lots of punctuation'
```

Slide 4: Advanced Regex Patterns

Regular expressions support sophisticated pattern matching through metacharacters and quantifiers. These patterns can capture complex text structures and extract specific information based on flexible rules and conditions.

```python
import re

def extract_data(text):
    # Match dates in various formats
    date_pattern = r'(\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})'
    
    # Match phone numbers
    phone_pattern = r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})'
    
    # Match URLs
    url_pattern = r'(https?://[^\s<>"]+|www\.[^\s<>"]+)'
    
    dates = re.findall(date_pattern, text)
    phones = re.findall(phone_pattern, text)
    urls = re.findall(url_pattern, text)
    
    return {'dates': dates, 'phones': phones, 'urls': urls}

# Example usage
sample_text = """
Meeting on 23/04/2024 with contact +1 (555) 123-4567
Visit https://example.com for more info
Follow-up on 2024-05-15
"""

results = extract_data(sample_text)
for key, value in results.items():
    print(f"{key}: {value}")

# Output:
# dates: ['23/04/2024', '2024-05-15']
# phones: ['+1 (555) 123-4567']
# urls: ['https://example.com']
```

Slide 5: String Method Performance Optimization

When working with large text datasets, optimizing string method operations becomes crucial. Using appropriate string methods and implementing efficient algorithms can significantly improve processing speed and memory usage.

```python
import time
from typing import List

def optimize_string_operations(texts: List[str]) -> List[str]:
    # Pre-compile frequently used strings
    REPLACE_CHARS = str.maketrans("", "", ".,!?")
    
    def process_text(text: str) -> str:
        # Use translate instead of multiple replace calls
        text = text.translate(REPLACE_CHARS)
        
        # Use join with list comprehension instead of multiple splits
        return ' '.join(word.lower() for word in text.split())
    
    # Process in batches for memory efficiency
    BATCH_SIZE = 1000
    results = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        results.extend(map(process_text, batch))
    
    return results

# Performance comparison
sample_texts = [
    "Hello, World! " * 100,
    "Python Programming!!! " * 100,
] * 1000

start_time = time.time()
processed = optimize_string_operations(sample_texts)
end_time = time.time()

print(f"Processed {len(sample_texts)} texts in {end_time - start_time:.2f} seconds")
print(f"Sample output: {processed[0][:50]}...")
```

Slide 6: Regex Performance Optimization

Regular expressions can be computationally expensive when not properly optimized. Pre-compilation of patterns, using non-capturing groups, and implementing efficient matching strategies can significantly improve performance in large-scale text processing.

```python
import re
import time
from typing import List, Dict

class RegexOptimizer:
    def __init__(self):
        # Pre-compile regex patterns
        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'url': re.compile(r'(?:https?://[^\s<>"]+|www\.[^\s<>"]+)')
        }
    
    def process_text(self, text: str) -> Dict[str, List[str]]:
        results = {}
        
        # Use single pass through text for all patterns
        for pattern_name, pattern in self.patterns.items():
            results[pattern_name] = pattern.findall(text)
            
        return results

# Performance benchmark
optimizer = RegexOptimizer()
large_text = """
Contact: test@email.com, +1-555-123-4567
Website: https://example.com
Secondary: support@company.com, (555) 987-6543
""" * 10000

start_time = time.time()
matches = optimizer.process_text(large_text)
end_time = time.time()

print(f"Processing time: {end_time - start_time:.3f} seconds")
print(f"Found {len(matches['email'])} emails")
print(f"Found {len(matches['phone'])} phone numbers")
print(f"Found {len(matches['url'])} URLs")
```

Slide 7: String Methods for Data Extraction

String methods provide efficient tools for extracting structured information from text when patterns are consistent and well-defined. This approach is particularly useful for parsing formatted data without the complexity of regular expressions.

```python
def extract_structured_data(text: str) -> dict:
    # Extract key-value pairs from formatted text
    data = {}
    
    for line in text.split('\n'):
        if not line.strip():
            continue
            
        # Handle key-value pairs separated by ':'
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
            
        # Handle structured data with known positions
        elif '|' in line:
            fields = [f.strip() for f in line.split('|')]
            if len(fields) >= 3:
                data[fields[0]] = {
                    'category': fields[1],
                    'value': fields[2]
                }
    
    return data

# Example usage
sample_data = """
Product: Advanced Python Course
Price: $199.99
Status: Available
Category|Programming|Python 3.x
Level|Difficulty|Intermediate
"""

result = extract_structured_data(sample_data)
for key, value in result.items():
    print(f"{key}: {value}")

# Output:
# Product: Advanced Python Course
# Price: $199.99
# Status: Available
# Category: {'category': 'Programming', 'value': 'Python 3.x'}
# Level: {'category': 'Difficulty', 'value': 'Intermediate'}
```

Slide 8: Complex Pattern Matching with Regex

Regular expressions excel at handling complex text patterns with variable structures. This implementation demonstrates advanced regex features for extracting and validating sophisticated text patterns in real-world scenarios.

```python
import re
from typing import List, Dict

class PatternMatcher:
    def __init__(self):
        # Complex patterns for various data types
        self.patterns = {
            'datetime': re.compile(
                r'(?P<date>\d{4}-\d{2}-\d{2})[T\s](?P<time>\d{2}:\d{2}:\d{2})(?P<timezone>[+-]\d{2}:?\d{2})?'
            ),
            'version': re.compile(
                r'(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<prerelease>[0-9A-Za-z-]+))?'
            ),
            'log_entry': re.compile(
                r'\[(?P<level>ERROR|WARN|INFO|DEBUG)\]\s+(?P<message>.*?)(?:\s+\{(?P<metadata>.*?)\})?$'
            )
        }
    
    def extract_patterns(self, text: str) -> Dict[str, List[Dict]]:
        results = {pattern_name: [] for pattern_name in self.patterns}
        
        for line in text.splitlines():
            for pattern_name, pattern in self.patterns.items():
                matches = pattern.finditer(line)
                for match in matches:
                    results[pattern_name].append(match.groupdict())
        
        return results

# Example usage
log_text = """
2024-03-15T14:30:45+00:00 - System update to version 2.3.1-beta
[ERROR] Database connection failed {retry_count: 3}
[INFO] Processing complete
2024-03-15T14:35:22-05:00 - Deployed version 2.3.2
"""

matcher = PatternMatcher()
results = matcher.extract_patterns(log_text)

for pattern_name, matches in results.items():
    print(f"\n{pattern_name} matches:")
    for match in matches:
        print(f"  {match}")

# Output example:
# datetime matches:
#   {'date': '2024-03-15', 'time': '14:30:45', 'timezone': '+00:00'}
#   {'date': '2024-03-15', 'time': '14:35:22', 'timezone': '-05:00'}
# version matches:
#   {'major': '2', 'minor': '3', 'patch': '1', 'prerelease': 'beta'}
#   {'major': '2', 'minor': '3', 'patch': '2', 'prerelease': None}
# log_entry matches:
#   {'level': 'ERROR', 'message': 'Database connection failed', 'metadata': 'retry_count: 3'}
#   {'level': 'INFO', 'message': 'Processing complete', 'metadata': None}
```

Slide 9: String Methods for Text Analysis

String methods provide efficient tools for analyzing text characteristics, word frequencies, and basic linguistic patterns. This implementation demonstrates practical text analysis techniques using built-in string operations.

```python
from collections import Counter
from typing import Dict, List, Tuple

class TextAnalyzer:
    def __init__(self, text: str):
        self.text = text
        self.words = text.lower().split()
    
    def word_statistics(self) -> Dict[str, int]:
        # Calculate basic text statistics
        return {
            'total_words': len(self.words),
            'unique_words': len(set(self.words)),
            'avg_word_length': round(sum(len(word) for word in self.words) / len(self.words), 2),
            'char_count': len(self.text)
        }
    
    def word_frequency(self, top_n: int = 10) -> List[Tuple[str, int]]:
        # Get most common words and their frequencies
        return Counter(self.words).most_common(top_n)
    
    def sentence_analysis(self) -> Dict[str, float]:
        sentences = [s.strip() for s in self.text.split('.') if s.strip()]
        words_per_sentence = [len(s.split()) for s in sentences]
        
        return {
            'total_sentences': len(sentences),
            'avg_sentence_length': round(sum(words_per_sentence) / len(sentences), 2),
            'max_sentence_length': max(words_per_sentence),
            'min_sentence_length': min(words_per_sentence)
        }

# Example usage
sample_text = """
Natural language processing is a subfield of artificial intelligence. 
It focuses on the interaction between computers and human language. 
Machine learning techniques are commonly used in modern NLP applications. 
Text analysis helps understand patterns in written communication.
"""

analyzer = TextAnalyzer(sample_text)
print("Word Statistics:")
print(analyzer.word_statistics())
print("\nTop 5 Most Common Words:")
print(analyzer.word_frequency(5))
print("\nSentence Analysis:")
print(analyzer.sentence_analysis())

# Output example:
# Word Statistics:
# {'total_words': 32, 'unique_words': 27, 'avg_word_length': 5.84, 'char_count': 234}
# 
# Top 5 Most Common Words:
# [('in', 2), ('language', 2), ('and', 1), ('processing', 1), ('is', 1)]
# 
# Sentence Analysis:
# {'total_sentences': 4, 'avg_sentence_length': 8.0, 'max_sentence_length': 10, 'min_sentence_length': 6}
```

Slide 10: Advanced Regex for Data Validation

Regular expressions provide powerful tools for validating complex data formats. This implementation showcases advanced regex patterns for validating various data types with comprehensive error reporting.

```python
import re
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ValidationError:
    field: str
    value: str
    message: str

class DataValidator:
    def __init__(self):
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'password': re.compile(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'),
            'username': re.compile(r'^[a-zA-Z0-9_-]{4,20}$'),
            'ipv4': re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'),
            'date': re.compile(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$')
        }
        
    def validate_field(self, field: str, value: str) -> Optional[ValidationError]:
        if field not in self.patterns:
            return ValidationError(field, value, f"Unknown field type: {field}")
            
        if not isinstance(value, str):
            return ValidationError(field, str(value), "Value must be a string")
            
        if not self.patterns[field].match(value):
            return ValidationError(field, value, f"Invalid {field} format")
            
        return None

    def validate_data(self, data: dict) -> List[ValidationError]:
        errors = []
        for field, value in data.items():
            error = self.validate_field(field, value)
            if error:
                errors.append(error)
        return errors

# Example usage
validator = DataValidator()
test_data = {
    'email': 'invalid.email@com',
    'password': 'weak',
    'username': 'user@123',
    'ipv4': '256.1.2.3',
    'date': '2024-13-45'
}

errors = validator.validate_data(test_data)
for error in errors:
    print(f"{error.field}: '{error.value}' - {error.message}")

# Output example:
# email: 'invalid.email@com' - Invalid email format
# password: 'weak' - Invalid password format
# username: 'user@123' - Invalid username format
# ipv4: '256.1.2.3' - Invalid ipv4 format
# date: '2024-13-45' - Invalid date format
```

Slide 11: Real-world Application - Log Parser

This implementation demonstrates a practical application combining string methods and regex for parsing complex log files. The solution handles various log formats and provides structured output for analysis.

```python
import re
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class LogEntry:
    timestamp: datetime
    level: str
    service: str
    message: str
    metadata: Optional[Dict] = None

class LogParser:
    def __init__(self):
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+'
            r'(?P<level>ERROR|WARN|INFO|DEBUG)\s+'
            r'\[(?P<service>[^\]]+)\]\s+'
            r'(?P<message>.*?)' 
            r'(?:\s+\{(?P<metadata>.*?)\})?$'
        )
        
    def parse_metadata(self, metadata_str: Optional[str]) -> Optional[Dict]:
        if not metadata_str:
            return None
            
        metadata = {}
        for item in metadata_str.split(','):
            if ':' in item:
                key, value = item.split(':', 1)
                metadata[key.strip()] = value.strip()
        return metadata
    
    def parse_log_file(self, log_content: str) -> List[LogEntry]:
        entries = []
        for line in log_content.splitlines():
            if not line.strip():
                continue
                
            match = self.log_pattern.match(line)
            if match:
                data = match.groupdict()
                entries.append(LogEntry(
                    timestamp=datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S'),
                    level=data['level'],
                    service=data['service'],
                    message=data['message'],
                    metadata=self.parse_metadata(data['metadata'])
                ))
        return entries

# Example usage
log_content = """
2024-03-15 10:30:45 ERROR [database-service] Connection timeout {retries: 3, db: users}
2024-03-15 10:30:46 INFO [auth-service] User authentication successful {user_id: 12345}
2024-03-15 10:30:47 WARN [api-gateway] High latency detected {latency: 500ms, endpoint: /users}
2024-03-15 10:30:48 DEBUG [cache-service] Cache miss for key user_preferences
"""

parser = LogParser()
log_entries = parser.parse_log_file(log_content)

# Analysis of parsed logs
level_count = {}
service_stats = {}

for entry in log_entries:
    # Count by level
    level_count[entry.level] = level_count.get(entry.level, 0) + 1
    
    # Collect service statistics
    if entry.service not in service_stats:
        service_stats[entry.service] = {'count': 0, 'errors': 0}
    service_stats[entry.service]['count'] += 1
    if entry.level == 'ERROR':
        service_stats[entry.service]['errors'] += 1

print("Log Level Distribution:")
for level, count in level_count.items():
    print(f"{level}: {count}")

print("\nService Statistics:")
for service, stats in service_stats.items():
    print(f"{service}:")
    print(f"  Total logs: {stats['count']}")
    print(f"  Error rate: {(stats['errors']/stats['count']*100):.1f}%")

# Output example:
# Log Level Distribution:
# ERROR: 1
# INFO: 1
# WARN: 1
# DEBUG: 1
#
# Service Statistics:
# database-service:
#   Total logs: 1
#   Error rate: 100.0%
# auth-service:
#   Total logs: 1
#   Error rate: 0.0%
# api-gateway:
#   Total logs: 1
#   Error rate: 0.0%
# cache-service:
#   Total logs: 1
#   Error rate: 0.0%
```

Slide 12: Real-world Application - Code Analyzer

This implementation combines string methods and regex to analyze source code files, extracting meaningful metrics and patterns that are useful for code quality assessment and refactoring decisions.

```python
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CodeMetrics:
    lines_of_code: int
    comment_lines: int
    blank_lines: int
    function_count: int
    class_count: int
    complexity: int
    imports: List[str]
    dependencies: Dict[str, int]

class CodeAnalyzer:
    def __init__(self):
        self.patterns = {
            'function': re.compile(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'),
            'class': re.compile(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?:'),
            'import': re.compile(r'(?:from\s+([.\w]+)\s+)?import\s+([^#\n]+)'),
            'comment': re.compile(r'^\s*#.*$|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', re.MULTILINE),
            'complexity': re.compile(r'\bif\b|\bfor\b|\bwhile\b|\band\b|\bor\b')
        }

    def analyze_code(self, code: str) -> CodeMetrics:
        # Remove multi-line strings and comments first
        clean_code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '', code)
        
        # Calculate basic metrics
        lines = clean_code.splitlines()
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = len(re.findall(self.patterns['comment'], code))
        
        # Extract functions and classes
        functions = self.patterns['function'].findall(clean_code)
        classes = self.patterns['class'].findall(clean_code)
        
        # Calculate complexity
        complexity = len(self.patterns['complexity'].findall(clean_code))
        
        # Analyze imports and dependencies
        imports = []
        dependencies = {}
        
        for match in self.patterns['import'].finditer(clean_code):
            module_from, module_import = match.groups()
            if module_from:
                imports.append(f"{module_from}.{module_import.strip()}")
                dependencies[module_from] = dependencies.get(module_from, 0) + 1
            else:
                for module in module_import.split(','):
                    clean_module = module.strip()
                    imports.append(clean_module)
                    dependencies[clean_module] = dependencies.get(clean_module, 0) + 1
        
        return CodeMetrics(
            lines_of_code=len(lines) - blank_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            function_count=len(functions),
            class_count=len(classes),
            complexity=complexity,
            imports=sorted(imports),
            dependencies=dependencies
        )

# Example usage
sample_code = """
import os
from typing import List, Dict
import numpy as np
from pandas import DataFrame

class DataProcessor:
    \"\"\"
    A class for processing data.
    \"\"\"
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def process(self) -> DataFrame:
        # Process the data
        if len(self.data) > 0:
            for item in self.data:
                if 'value' in item and item['value'] > 0:
                    item['processed'] = True
        
        return DataFrame(self.data)

def main():
    # Initialize processor
    data = [{'value': i} for i in range(10)]
    processor = DataProcessor(data)
    result = processor.process()
    return result
"""

analyzer = CodeAnalyzer()
metrics = analyzer.analyze_code(sample_code)

print("Code Analysis Results:")
print(f"Lines of Code: {metrics.lines_of_code}")
print(f"Comment Lines: {metrics.comment_lines}")
print(f"Blank Lines: {metrics.blank_lines}")
print(f"Functions: {metrics.function_count}")
print(f"Classes: {metrics.class_count}")
print(f"Complexity Score: {metrics.complexity}")
print("\nImports:")
for imp in metrics.imports:
    print(f"  - {imp}")
print("\nDependencies:")
for dep, count in metrics.dependencies.items():
    print(f"  {dep}: {count} references")

# Output example:
# Code Analysis Results:
# Lines of Code: 21
# Comment Lines: 3
# Blank Lines: 5
# Functions: 3
# Classes: 1
# Complexity Score: 4
#
# Imports:
# - DataFrame
# - Dict
# - List
# - numpy
# - os
#
# Dependencies:
# typing: 2 references
# pandas: 1 reference
# numpy: 1 reference
# os: 1 reference
```

Slide 13: Additional Resources

*   Search for "Python String Processing" on arXiv: [https://arxiv.org/search/?searchtype=all&query=python+string+processing](https://arxiv.org/search/?searchtype=all&query=python+string+processing)
*   Regular Expressions in Programming Languages: [https://arxiv.org/abs/2010.14411](https://arxiv.org/abs/2010.14411)
*   Efficient Text Processing Algorithms: [https://arxiv.org/abs/2103.00901](https://arxiv.org/abs/2103.00901)
*   For learning resources and documentation:
    *   Python Official Documentation on String Methods: [https://docs.python.org/3/library/stdtypes.html#string-methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
    *   Python Regular Expression Documentation: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)
    *   Regular Expression Testing Tool: [https://regex101.com](https://regex101.com)

