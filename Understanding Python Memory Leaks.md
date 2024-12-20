## Understanding Python Memory Leaks
Slide 1: Memory Leaks in Python

Memory leaks in Python are indeed possible, despite the presence of a garbage collector. While Python's automatic memory management helps prevent many common memory issues, it doesn't guarantee complete immunity from leaks. Long-running applications are particularly susceptible to these problems. Let's explore why memory leaks occur and how to address them.

Slide 2: Reference Cycles

Reference cycles occur when objects reference each other, creating a loop that prevents the garbage collector from freeing memory. This is one of the primary causes of memory leaks in Python.

Slide 3: Source Code for Reference Cycles

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# Create a circular reference
node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1

# These objects will not be garbage collected
# even when they go out of scope
```

Slide 4: Global Variables and Caching

Improper use of global variables or caching mechanisms can lead to memory leaks by holding onto references longer than necessary. This is especially problematic in long-running applications or scripts.

Slide 5: Source Code for Global Variables and Caching

```python
cache = {}

def expensive_operation(key):
    if key not in cache:
        # Simulate expensive operation
        result = sum(range(key * 1000000))
        cache[key] = result
    return cache[key]

# This cache will grow indefinitely as new keys are added
for i in range(1000):
    expensive_operation(i)

print(f"Cache size: {len(cache)}")
```

Slide 6: Detecting Memory Leaks

Python provides built-in tools to help detect and diagnose memory leaks. The `tracemalloc` module is particularly useful for tracking memory allocations and identifying potential issues.

Slide 7: Source Code for Detecting Memory Leaks

```python
import tracemalloc
import time

tracemalloc.start()

# Simulate a memory leak
leaky_list = []
for _ in range(1000000):
    leaky_list.append(object())

# Get memory snapshot
snapshot = tracemalloc.take_snapshot()

# Print top 10 memory consumers
print("Top 10 memory consumers:")
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)

tracemalloc.stop()
```

Slide 8: Results for: Detecting Memory Leaks

```
Top 10 memory consumers:
<frozen importlib._bootstrap>:219: size=4855 KiB, count=39328, average=126 B
<unknown>:0: size=865 KiB, count=1, average=865 KiB
/path/to/script.py:7: size=76.3 MiB, count=1000000, average=80 B
/usr/lib/python3.x/tracemalloc.py:491: size=4855 KiB, count=39328, average=126 B
...
```

Slide 9: Fixing Memory Leaks

To fix memory leaks, focus on breaking reference cycles, limiting the scope of variables, and implementing proper cleanup mechanisms. Let's look at some strategies to address common issues.

Slide 10: Source Code for Fixing Memory Leaks

```python
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

def create_cycle():
    node1 = Node(1)
    node2 = Node(2)
    node1.next = weakref.ref(node2)  # Use weak reference
    node2.next = weakref.ref(node1)  # Use weak reference
    return node1, node2

# Create nodes
n1, n2 = create_cycle()

# When n1 and n2 go out of scope, they can be garbage collected
del n1, n2
```

Slide 11: Real-Life Example: Web Scraper

Consider a web scraper that downloads and processes web pages. Without proper memory management, it could lead to significant memory leaks over time.

Slide 12: Source Code for Web Scraper

```python
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Process the soup object
    # ...
    return soup

# Problematic implementation
scraped_data = []
urls = ["http://example.com"] * 1000  # 1000 identical URLs for demonstration

for url in urls:
    scraped_data.append(scrape_website(url))

# Memory usage grows with each iteration
print(f"Number of stored pages: {len(scraped_data)}")
```

Slide 13: Improved Web Scraper

Let's improve our web scraper to avoid memory leaks by processing data immediately and releasing resources.

Slide 14: Source Code for Improved Web Scraper

```python
import requests
from bs4 import BeautifulSoup

def scrape_and_process(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Process the soup object immediately
    title = soup.title.string if soup.title else "No title"
    # Return only necessary data
    return title

# Improved implementation
processed_data = []
urls = ["http://example.com"] * 1000  # 1000 identical URLs for demonstration

for url in urls:
    processed_data.append(scrape_and_process(url))

# Memory usage is significantly reduced
print(f"Number of processed titles: {len(processed_data)}")
```

Slide 15: Additional Resources

For more information on memory management and leak detection in Python, consider exploring these resources:

1.  Python's official documentation on the garbage collector: [https://docs.python.org/3/library/gc.html](https://docs.python.org/3/library/gc.html)
2.  The tracemalloc module: [https://docs.python.org/3/library/tracemalloc.html](https://docs.python.org/3/library/tracemalloc.html)
3.  "Hunting memory leaks in Python" by Victor Stinner: [https://arxiv.org/abs/1808.03022](https://arxiv.org/abs/1808.03022)

These resources provide in-depth information on Python's memory management system and advanced techniques for identifying and resolving memory leaks.

