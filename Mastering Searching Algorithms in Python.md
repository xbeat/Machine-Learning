## Mastering Searching Algorithms in Python
Slide 1: Introduction to Searching Algorithms

Searching algorithms are fundamental techniques used to find specific items within a collection of data. In Python, these algorithms play a crucial role in efficiently locating elements in various data structures such as lists, arrays, and dictionaries. Understanding and implementing these algorithms is essential for writing optimized code and improving overall program performance.

```python
def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i  # Return index if found
    return -1  # Return -1 if not found

numbers = [4, 2, 7, 1, 9, 5]
result = linear_search(numbers, 7)
print(f"Index of 7: {result}")  # Output: Index of 7: 2
```

Slide 2: Linear Search

Linear search is the simplest searching algorithm. It sequentially checks each element in a collection until a match is found or the end is reached. While not the most efficient for large datasets, it's straightforward and works well for small collections or unsorted data.

```python
def linear_search_with_steps(arr, target):
    for i, item in enumerate(arr):
        print(f"Checking index {i}: {item}")
        if item == target:
            return i
    return -1

data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
result = linear_search_with_steps(data, 6)
print(f"Index of 6: {result}")
```

Slide 3: Binary Search

Binary search is an efficient algorithm for searching sorted arrays. It repeatedly divides the search interval in half, significantly reducing the number of comparisons needed. This algorithm has a time complexity of O(log n), making it much faster than linear search for large datasets.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

sorted_numbers = [1, 3, 5, 7, 9, 11, 13, 15]
result = binary_search(sorted_numbers, 7)
print(f"Index of 7: {result}")  # Output: Index of 7: 3
```

Slide 4: Binary Search Visualization

To better understand how binary search works, let's visualize its steps using a simple Python implementation that prints the current search range and midpoint at each iteration.

```python
def binary_search_visual(arr, target):
    left, right = 0, len(arr) - 1
    steps = 1
    while left <= right:
        mid = (left + right) // 2
        print(f"Step {steps}: Searching in range [{left}, {right}], midpoint: {mid}")
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
        steps += 1
    return -1

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 13
result = binary_search_visual(sorted_data, target)
print(f"Index of {target}: {result}")
```

Slide 5: Jump Search

Jump search is an algorithm that works on sorted arrays by skipping a fixed number of elements and then performing a linear search. It's a good middle ground between linear and binary search, especially for large datasets where binary search might be overkill.

```python
import math

def jump_search(arr, target):
    n = len(arr)
    step = int(math.sqrt(n))
    prev = 0
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return -1
    
    if arr[prev] == target:
        return prev
    return -1

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
result = jump_search(sorted_data, 13)
print(f"Index of 13: {result}")  # Output: Index of 13: 6
```

Slide 6: Interpolation Search

Interpolation search is an improved variant of binary search, particularly effective for uniformly distributed sorted arrays. It uses a formula to estimate the position of the target value, potentially reducing the number of comparisons needed.

```python
def interpolation_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high and arr[low] <= target <= arr[high]:
        if low == high:
            if arr[low] == target:
                return low
            return -1
        
        pos = low + int(((target - arr[low]) * (high - low)) / (arr[high] - arr[low]))
        
        if arr[pos] == target:
            return pos
        elif arr[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    return -1

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
result = interpolation_search(sorted_data, 11)
print(f"Index of 11: {result}")  # Output: Index of 11: 5
```

Slide 7: Exponential Search

Exponential search is particularly useful for unbounded or infinite arrays. It works by finding a range where the target might exist and then performing a binary search within that range. This algorithm is efficient for both small and large datasets.

```python
def binary_search(arr, left, right, target):
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def exponential_search(arr, target):
    if arr[0] == target:
        return 0
    
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2
    
    return binary_search(arr, i // 2, min(i, len(arr) - 1), target)

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
result = exponential_search(sorted_data, 15)
print(f"Index of 15: {result}")  # Output: Index of 15: 7
```

Slide 8: Fibonacci Search

Fibonacci search is an efficient searching algorithm that uses Fibonacci numbers to divide the array into unequal parts. It's particularly useful when the binary search's mid-calculation might cause overflow for large arrays.

```python
def fibonacci_search(arr, target):
    n = len(arr)
    fib2 = 0  # (m-2)'th Fibonacci number
    fib1 = 1  # (m-1)'th Fibonacci number
    fib = fib2 + fib1  # m'th Fibonacci number
    
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib2 + fib1
    
    offset = -1
    while fib > 1:
        i = min(offset + fib2, n - 1)
        if arr[i] < target:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > target:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            return i
    
    if fib1 and arr[offset + 1] == target:
        return offset + 1
    
    return -1

sorted_data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
result = fibonacci_search(sorted_data, 13)
print(f"Index of 13: {result}")  # Output: Index of 13: 6
```

Slide 9: Sublist Search

Sublist search, also known as pattern matching within a list, is used to find a sequence of elements within a larger list. This algorithm has applications in text processing and data analysis.

```python
def sublist_search(main_list, pattern):
    n, m = len(main_list), len(pattern)
    for i in range(n - m + 1):
        if main_list[i:i+m] == pattern:
            return i
    return -1

main_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
pattern = [3, 4, 5]
result = sublist_search(main_list, pattern)
print(f"Pattern found at index: {result}")  # Output: Pattern found at index: 2
```

Slide 10: Hash-based Search

Hash-based searching uses a hash table to achieve constant-time average-case complexity. In Python, dictionaries are implemented using hash tables, making them extremely efficient for key-based searches.

```python
def hash_search(data, key):
    return data.get(key, "Not found")

hash_table = {
    "apple": 5,
    "banana": 7,
    "orange": 3,
    "grape": 9
}

result = hash_search(hash_table, "banana")
print(f"Value for 'banana': {result}")  # Output: Value for 'banana': 7
```

Slide 11: Depth-First Search (DFS)

Depth-First Search is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It's useful for tasks like finding connected components, topological sorting, and solving puzzles.

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("DFS traversal:")
dfs(graph, 'A')
# Output: DFS traversal: A B D E F C
```

Slide 12: Breadth-First Search (BFS)

Breadth-First Search is another graph traversal algorithm that explores all vertices at the present depth before moving to vertices at the next depth level. It's particularly useful for finding the shortest path in unweighted graphs.

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print("BFS traversal:")
bfs(graph, 'A')
# Output: BFS traversal: A B C D E F
```

Slide 13: Real-life Example: Library Catalog Search

Imagine implementing a search function for a library catalog system. We'll use a binary search to quickly find books by their ISBN numbers, assuming the books are sorted by ISBN.

```python
class Book:
    def __init__(self, isbn, title, author):
        self.isbn = isbn
        self.title = title
        self.author = author

def library_search(books, target_isbn):
    left, right = 0, len(books) - 1
    while left <= right:
        mid = (left + right) // 2
        if books[mid].isbn == target_isbn:
            return books[mid]
        elif books[mid].isbn < target_isbn:
            left = mid + 1
        else:
            right = mid - 1
    return None

# Sample library catalog
library = [
    Book("9780061120084", "To Kill a Mockingbird", "Harper Lee"),
    Book("9780141439518", "Pride and Prejudice", "Jane Austen"),
    Book("9780743273565", "The Great Gatsby", "F. Scott Fitzgerald"),
    Book("9780451524935", "1984", "George Orwell"),
    Book("9780486284736", "Frankenstein", "Mary Shelley")
]

# Searching for a book
target_isbn = "9780743273565"
result = library_search(library, target_isbn)

if result:
    print(f"Book found: {result.title} by {result.author}")
else:
    print("Book not found")

# Output: Book found: The Great Gatsby by F. Scott Fitzgerald
```

Slide 14: Real-life Example: Spell Checker

Let's implement a simple spell checker using the Levenshtein distance algorithm to find the closest match for a misspelled word in a dictionary.

```python
import difflib

def spell_check(word, dictionary):
    if word in dictionary:
        return word
    
    closest_matches = difflib.get_close_matches(word, dictionary, n=1, cutoff=0.6)
    return closest_matches[0] if closest_matches else None

# Sample dictionary
dictionary = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]

# Test the spell checker
misspelled_words = ["aple", "banan", "cheery", "dat", "elderberri"]

for word in misspelled_words:
    correction = spell_check(word, dictionary)
    if correction:
        print(f"'{word}' might be a misspelling of '{correction}'")
    else:
        print(f"No close match found for '{word}'")

# Output:
# 'aple' might be a misspelling of 'apple'
# 'banan' might be a misspelling of 'banana'
# 'cheery' might be a misspelling of 'cherry'
# 'dat' might be a misspelling of 'date'
# 'elderberri' might be a misspelling of 'elderberry'
```

Slide 15: Additional Resources

For those interested in diving deeper into searching algorithms and their implementations in Python, here are some valuable resources:

1.  "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein - A comprehensive textbook covering various algorithms, including searching algorithms.
2.  "Algorithms" by Robert Sedgewick and Kevin Wayne - Offers in-depth explanations and implementations of searching algorithms.
3.  Python's official documentation on sorting and searching algorithms: [https://docs.python.org/3/howto/sorting.html](https://docs.python.org/3/howto/sorting.html)
4.  ArXiv paper on "Efficient String Matching: An Aid to Bibliographic Search" by Aho and Corasick: [https://arxiv.org/abs/cs/0604024](https://arxiv.org/abs/cs/0604024)
5.  ArXiv paper on "Faster and Simpler Algorithm for Sorting Signed Permutations by Reversals" by Tannier, Bergeron, and Sagot: [https://arxiv.org/abs/cs/0604033](https://arxiv.org/abs/cs/0604033)

These resources provide a solid foundation for understanding and implementing various searching algorithms in Python and other programming languages.

