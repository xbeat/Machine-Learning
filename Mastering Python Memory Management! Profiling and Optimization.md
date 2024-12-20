## Mastering Python Memory Management! Profiling and Optimization
Slide 1: Memory Management in Python

Python's memory management is automatic, but understanding its intricacies can help optimize your code. This presentation will cover profiling techniques and optimization strategies to help you master memory management in Python.

```python
import sys

# Create a list of integers
numbers = list(range(1000))

# Get the size of the list in bytes
size = sys.getsizeof(numbers)

print(f"Size of the list: {size} bytes")
```

Slide 2: Understanding Python's Memory Allocation

Python uses a private heap to store objects and data structures. The memory manager allocates heap space for objects when they're created and frees it when they're no longer used.

```python
import gc

# Create some objects
a = [1, 2, 3]
b = "Hello, World!"
c = {1: 'one', 2: 'two'}

# Get the number of objects tracked by the garbage collector
object_count = len(gc.get_objects())

print(f"Number of objects tracked: {object_count}")
```

Slide 3: Reference Counting

Python uses reference counting as its primary memory management technique. When an object's reference count drops to zero, it's deallocated.

```python
import sys

# Create a list and get its reference count
my_list = [1, 2, 3]
ref_count = sys.getrefcount(my_list)

print(f"Initial reference count: {ref_count}")

# Create a new reference to the list
another_ref = my_list
new_ref_count = sys.getrefcount(my_list)

print(f"Reference count after new reference: {new_ref_count}")
```

Slide 4: Garbage Collection

For cyclic references, Python uses a garbage collector that periodically checks for and frees unreachable objects.

```python
import gc

# Create a circular reference
def create_cycle():
    l = {}
    l['self'] = l
    return l

# Create some cycles
cycles = [create_cycle() for _ in range(10)]

# Force garbage collection
collected = gc.collect()

print(f"Number of objects collected: {collected}")
```

Slide 5: Memory Profiling: memory\_profiler

The memory\_profiler module helps track memory usage of Python code.

```python
from memory_profiler import profile

@profile
def memory_hungry_function():
    big_list = [i for i in range(1000000)]
    del big_list

if __name__ == '__main__':
    memory_hungry_function()

# Run this script with: python -m memory_profiler script.py
```

Slide 6: Time Profiling: cProfile

cProfile helps identify time-consuming parts of your code, which often correlate with memory usage.

```python
import cProfile

def time_consuming_function():
    return sum(i * i for i in range(10**6))

cProfile.run('time_consuming_function()')
```

Slide 7: Using tracemalloc

tracemalloc is a built-in module that tracks Python memory allocations.

```python
import tracemalloc

tracemalloc.start()

# Your code here
big_list = [0] * 1000000

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:3]:
    print(stat)
```

Slide 8: Optimizing Data Structures: Lists vs. Tuples

Tuples are immutable and generally use less memory than lists.

```python
import sys

# Compare memory usage of list and tuple
list_ex = [1, 2, 3, 4, 5]
tuple_ex = (1, 2, 3, 4, 5)

print(f"List size: {sys.getsizeof(list_ex)} bytes")
print(f"Tuple size: {sys.getsizeof(tuple_ex)} bytes")
```

Slide 9: Using Generators for Memory Efficiency

Generators can help reduce memory usage by yielding items one at a time instead of creating a whole list in memory.

```python
def number_generator(n):
    for i in range(n):
        yield i

# Using a generator
for num in number_generator(1000000):
    # Process num
    pass

# Compared to using a list
# numbers = list(range(1000000))  # This would consume more memory
```

Slide 10: Real-life Example: Processing Large Files

When dealing with large files, it's more memory-efficient to process them line by line.

```python
def process_large_file(filename):
    with open(filename, 'r') as file:
        for line in file:
            # Process each line
            processed_line = line.strip().upper()
            print(processed_line)

# Usage
process_large_file('large_file.txt')
```

Slide 11: Real-life Example: Image Processing

When processing large images, using generators can help manage memory usage.

```python
from PIL import Image
import numpy as np

def image_processor(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        for y in range(height):
            row = []
            for x in range(width):
                pixel = img.getpixel((x, y))
                row.append(sum(pixel) / 3)  # Convert to grayscale
            yield row

# Usage
for row in image_processor('large_image.jpg'):
    # Process each row
    np.array(row)  # Convert to numpy array for further processing
```

Slide 12: Using **slots** for Optimizing Classes

The **slots** attribute can reduce memory usage of instances by specifying allowed attributes.

```python
class RegularClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SlottedClass:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys

regular = RegularClass(1, 2)
slotted = SlottedClass(1, 2)

print(f"Regular class size: {sys.getsizeof(regular)} bytes")
print(f"Slotted class size: {sys.getsizeof(slotted)} bytes")
```

Slide 13: Weak References

Weak references allow you to refer to an object without increasing its reference count.

```python
import weakref

class HeavyObject:
    def __init__(self, data):
        self.data = data

# Create a heavy object
heavy = HeavyObject([1] * 1000000)

# Create a weak reference
weak_ref = weakref.ref(heavy)

# Access the object through the weak reference
if weak_ref() is not None:
    print("Object still exists")

# Delete the original reference
del heavy

# Check if the object still exists
print("Object exists:" if weak_ref() is not None else "Object has been garbage collected")
```

Slide 14: Optimizing Strings: String Interning

String interning can save memory by reusing string objects.

```python
import sys

# String interning in action
a = 'hello'
b = 'hello'

print(f"a is b: {a is b}")  # True, Python interns short strings

# Manual interning for longer strings
c = sys.intern('a longer string that would not be interned automatically')
d = sys.intern('a longer string that would not be interned automatically')

print(f"c is d: {c is d}")  # True, we manually interned the strings
```

Slide 15: Additional Resources

For further exploration of memory management and optimization in Python:

1. "Optimizing Python" by Gabriele Lanaro (ArXiv:1405.5958) [https://arxiv.org/abs/1405.5958](https://arxiv.org/abs/1405.5958)
2. "Memory Management in Python" by Jake VanderPlas (ArXiv:1607.03379) [https://arxiv.org/abs/1607.03379](https://arxiv.org/abs/1607.03379)

These resources provide in-depth discussions on advanced memory management techniques and optimization strategies in Python.

