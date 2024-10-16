## Simplify Python Code with defaultdict
Slide 1: Introduction to defaultdict

The defaultdict class is a powerful tool in Python's collections module that simplifies handling of dictionaries with default values. It automatically initializes new keys with a default value, reducing the need for explicit key checks.

```python
from collections import defaultdict

# Regular dictionary
regular_dict = {}
# Raises KeyError if key doesn't exist
# print(regular_dict['non_existent_key'])

# defaultdict
default_dict = defaultdict(int)
# Returns 0 for non-existent keys
print(default_dict['non_existent_key'])  # Output: 0
```

Slide 2: Creating a defaultdict

A defaultdict is created by specifying a default factory function. This function is called when a key is accessed that doesn't exist in the dictionary.

```python
from collections import defaultdict

# Create a defaultdict with int as the default factory
int_dict = defaultdict(int)

# Create a defaultdict with list as the default factory
list_dict = defaultdict(list)

# Create a defaultdict with a custom default factory
def custom_default():
    return "Not found"

custom_dict = defaultdict(custom_default)

print(int_dict['a'])    # Output: 0
print(list_dict['b'])   # Output: []
print(custom_dict['c']) # Output: Not found
```

Slide 3: Counting with defaultdict

One common use of defaultdict is for counting occurrences. It simplifies the code by eliminating the need to check if a key exists before incrementing its value.

```python
from collections import defaultdict

# Count word occurrences in a sentence
sentence = "the quick brown fox jumps over the lazy dog"
word_count = defaultdict(int)

for word in sentence.split():
    word_count[word] += 1

print(word_count)
# Output: defaultdict(<class 'int'>, {'the': 2, 'quick': 1, 'brown': 1, 'fox': 1, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1})
```

Slide 4: Grouping with defaultdict

defaultdict is excellent for grouping related items. It automatically creates a list for each new key, allowing you to append items without checking if the key exists.

```python
from collections import defaultdict

# Group students by their grade
students = [
    ('Alice', 'A'),
    ('Bob', 'B'),
    ('Charlie', 'A'),
    ('David', 'C'),
    ('Eve', 'B')
]

grade_groups = defaultdict(list)

for name, grade in students:
    grade_groups[grade].append(name)

print(grade_groups)
# Output: defaultdict(<class 'list'>, {'A': ['Alice', 'Charlie'], 'B': ['Bob', 'Eve'], 'C': ['David']})
```

Slide 5: Nested defaultdicts

defaultdicts can be nested to create more complex data structures. This is particularly useful for representing hierarchical or multi-dimensional data.

```python
from collections import defaultdict

# Create a nested defaultdict for storing city populations by country and state
populations = defaultdict(lambda: defaultdict(int))

# Add some data
populations['USA']['California'] = 39.5  # million
populations['USA']['Texas'] = 29.1  # million
populations['Canada']['Ontario'] = 14.7  # million

# Access data, even for non-existent keys
print(populations['USA']['California'])  # Output: 39.5
print(populations['Australia']['Sydney'])  # Output: 0

print(dict(populations))  # Convert to regular dict for display
# Output: {'USA': {'California': 39.5, 'Texas': 29.1}, 'Canada': {'Ontario': 14.7}, 'Australia': {'Sydney': 0}}
```

Slide 6: Default Factory Functions

The default factory function can be any callable that returns a value. This allows for great flexibility in how default values are created.

```python
from collections import defaultdict
import random

# defaultdict with a lambda function
random_dict = defaultdict(lambda: random.randint(1, 10))

print(random_dict['a'])  # Output: Random integer between 1 and 10
print(random_dict['b'])  # Output: Another random integer between 1 and 10

# defaultdict with a class method
class DefaultValue:
    counter = 0
    
    @classmethod
    def get_next(cls):
        cls.counter += 1
        return f"Value_{cls.counter}"

sequence_dict = defaultdict(DefaultValue.get_next)

print(sequence_dict['x'])  # Output: Value_1
print(sequence_dict['y'])  # Output: Value_2
print(sequence_dict['x'])  # Output: Value_1 (existing key, not incremented)
```

Slide 7: Real-Life Example: Word Frequency Analysis

Let's use defaultdict to analyze word frequencies in a text, a common task in natural language processing.

```python
from collections import defaultdict
import re

def word_frequency(text):
    # Create a defaultdict to store word frequencies
    freq = defaultdict(int)
    
    # Convert to lowercase and split into words
    words = re.findall(r'\w+', text.lower())
    
    # Count the frequency of each word
    for word in words:
        freq[word] += 1
    
    return freq

# Example usage
sample_text = """
Python is a programming language that lets you work quickly and integrate systems more effectively.
Python is powerful... and fast; plays well with others; runs everywhere; is friendly & easy to learn; is Open.
"""

result = word_frequency(sample_text)

# Print the 5 most common words
print(sorted(result.items(), key=lambda x: x[1], reverse=True)[:5])
# Output: [('is', 3), ('and', 2), ('python', 2), ('a', 1), ('programming', 1)]
```

Slide 8: Real-Life Example: Building a Simple Graph

defaultdict can be used to represent graph structures efficiently. Let's create a simple undirected graph and perform some basic operations.

```python
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(set)
    
    def add_edge(self, u, v):
        self.graph[u].add(v)
        self.graph[v].add(u)
    
    def remove_edge(self, u, v):
        self.graph[u].discard(v)
        self.graph[v].discard(u)
    
    def has_edge(self, u, v):
        return v in self.graph[u]
    
    def neighbors(self, u):
        return self.graph[u]

# Create a graph
g = Graph()
g.add_edge(1, 2)
g.add_edge(1, 3)
g.add_edge(2, 4)

print(g.has_edge(1, 2))  # Output: True
print(g.neighbors(1))    # Output: {2, 3}
g.remove_edge(1, 2)
print(g.has_edge(1, 2))  # Output: False
```

Slide 9: Performance Considerations

While defaultdict offers convenience, it's important to understand its performance characteristics compared to regular dictionaries.

```python
from collections import defaultdict
import timeit

def regular_dict_count():
    d = {}
    for i in range(1000):
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d

def defaultdict_count():
    d = defaultdict(int)
    for i in range(1000):
        d[i] += 1
    return d

# Measure execution time
regular_time = timeit.timeit(regular_dict_count, number=1000)
defaultdict_time = timeit.timeit(defaultdict_count, number=1000)

print(f"Regular dict: {regular_time:.6f} seconds")
print(f"defaultdict: {defaultdict_time:.6f} seconds")
# Output may vary, but defaultdict is generally faster
# Sample output:
# Regular dict: 0.234567 seconds
# defaultdict: 0.123456 seconds
```

Slide 10: Handling Missing Keys

defaultdict changes how missing keys are handled. This can be both an advantage and a potential pitfall.

```python
from collections import defaultdict

# Regular dictionary
regular_dict = {}
try:
    print(regular_dict['missing'])
except KeyError:
    print("KeyError raised for missing key in regular dict")

# defaultdict
default_dict = defaultdict(int)
print(default_dict['missing'])  # No KeyError, returns 0

# This behavior can lead to unexpected results if not handled carefully
print('missing' in default_dict)  # Output: True
print(list(default_dict.keys()))  # Output: ['missing']

# To avoid creating keys unintentionally, use .get() method
print(default_dict.get('another_missing'))  # Output: None
print('another_missing' in default_dict)  # Output: False
```

Slide 11: Customizing defaultdict Behavior

You can subclass defaultdict to add or modify its behavior. This allows for more complex default value logic.

```python
from collections import defaultdict

class LimitedDefaultDict(defaultdict):
    def __init__(self, default_factory=None, limit=10):
        super().__init__(default_factory)
        self.limit = limit
        self.call_count = 0

    def __missing__(self, key):
        if self.call_count >= self.limit:
            raise ValueError("Default factory call limit exceeded")
        self.call_count += 1
        return super().__missing__(key)

# Usage
limited_dict = LimitedDefaultDict(int, limit=3)

print(limited_dict['a'])  # Output: 0
print(limited_dict['b'])  # Output: 0
print(limited_dict['c'])  # Output: 0

try:
    print(limited_dict['d'])
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: Default factory call limit exceeded
```

Slide 12: Combining defaultdict with Other Data Structures

defaultdict can be combined with other data structures to create more complex and efficient solutions.

```python
from collections import defaultdict, deque

class RecentCounter:
    def __init__(self):
        self.requests = defaultdict(deque)

    def ping(self, t: int, name: str) -> int:
        self.requests[name].append(t)
        while self.requests[name] and self.requests[name][0] < t - 3000:
            self.requests[name].popleft()
        return len(self.requests[name])

# Usage
counter = RecentCounter()
print(counter.ping(1, "user1"))    # Output: 1
print(counter.ping(100, "user1"))  # Output: 2
print(counter.ping(3001, "user1")) # Output: 3
print(counter.ping(3002, "user1")) # Output: 3 (request at t=1 is now outside the 3000ms window)
```

Slide 13: Best Practices and Common Pitfalls

When using defaultdict, it's important to be aware of some best practices and common pitfalls to ensure efficient and correct usage.

```python
from collections import defaultdict

# Best Practice: Use type hints for clarity
from typing import DefaultDict, List
user_posts: DefaultDict[str, List[str]] = defaultdict(list)

# Pitfall: Modifying the default factory after creation
d = defaultdict(list)
d['key'].append(1)
d.default_factory = set  # This doesn't affect existing values
print(d['key'])  # Output: [1]
print(d['new_key'])  # Output: set()

# Best Practice: Use defaultdict.() for shallow copies
original = defaultdict(int, {'a': 1, 'b': 2})
 = original.()  # Correctly copies the default_factory

# Pitfall: Forgetting that accessing a key creates it
d = defaultdict(int)
if 'key' in d:  # This will always be False for new keys
    print("Key exists")
print(d)  # Output: defaultdict(<class 'int'>, {'key': 0})

# Best Practice: Use .get() or .setdefault() to avoid unintended key creation
print(d.get('another_key'))  # Output: None
print(d)  # The dictionary remains unchanged
```

Slide 14: Additional Resources

For further exploration of defaultdict and related topics, consider these resources:

1. Python Official Documentation on defaultdict: [https://docs.python.org/3/library/collections.html#collections.defaultdict](https://docs.python.org/3/library/collections.html#collections.defaultdict)
2. "Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin This book includes detailed discussions on using defaultdict and other Python features effectively.
3. "Python Cookbook" by David Beazley and Brian K. Jones This resource provides practical recipes for solving common programming problems, including uses of defaultdict.
4. "Fluent Python" by Luciano Ramalho This book offers in-depth coverage of Python's data structures, including defaultdict.

Note: While these resources are widely recognized in the Python community, always verify the latest editions and information.

