## Mastering Big O Notation for Efficient Algorithms
Slide 1: Understanding O(1) Complexity

Constant time operations maintain consistent execution time regardless of input size. In Python, accessing dictionary elements and array indices exemplifies O(1) complexity since these operations take the same amount of time whether working with 10 or 10 million elements.

```python
def constant_time_operations(arr, key):
    # O(1) - Array index access
    first_element = arr[0]
    
    # O(1) - Dictionary operations
    cache = {"key": "value"}
    cache[key] = "new_value"
    
    # O(1) - Set operations
    number_set = {1, 2, 3}
    exists = 1 in number_set
    
    return first_element, cache[key], exists

# Example usage
arr = [1, 2, 3, 4, 5]
result = constant_time_operations(arr, "key")
print(f"Results: {result}")  # Output: Results: (1, 'new_value', True)
```

Slide 2: Linear Search with O(n) Complexity

Linear time complexity represents algorithms where execution time grows proportionally with input size. Searching through an unsorted array exemplifies O(n) complexity as each element must be examined exactly once to find a target value.

```python
def linear_search(arr, target):
    comparisons = 0
    for i, num in enumerate(arr):
        comparisons += 1
        if num == target:
            return i, comparisons
    return -1, comparisons

# Example with timing
import time

arr = list(range(1000000))
target = 999999

start_time = time.time()
index, comparisons = linear_search(arr, target)
end_time = time.time()

print(f"Found at index: {index}")
print(f"Comparisons made: {comparisons}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

Slide 3: Binary Search and O(log n)

Logarithmic time complexity represents algorithms that repeatedly divide the problem size by a constant factor. Binary search exemplifies this by halving the search space in each iteration, making it significantly more efficient than linear search for large sorted datasets.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    steps = 0
    
    while left <= right:
        steps += 1
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid, steps
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1, steps

# Example usage with timing
import time

sorted_arr = list(range(1000000))
target = 999999

start_time = time.time()
index, steps = binary_search(sorted_arr, target)
end_time = time.time()

print(f"Found at index: {index}")
print(f"Steps taken: {steps}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

Slide 4: Understanding O(n²) with Bubble Sort

The quadratic time complexity is often seen in nested iterations over the input. Bubble Sort demonstrates O(n²) complexity by comparing and swapping adjacent elements repeatedly, making it inefficient for large datasets but simple to implement and understand.

```python
def bubble_sort(arr):
    n = len(arr)
    comparisons = 0
    swaps = 0
    
    for i in range(n):
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                
    return arr, comparisons, swaps

# Example with performance metrics
import random
import time

data = random.sample(range(1000), 100)
start_time = time.time()
sorted_arr, comparisons, swaps = bubble_sort(data.copy())
end_time = time.time()

print(f"Time taken: {end_time - start_time:.6f} seconds")
print(f"Comparisons: {comparisons}")
print(f"Swaps: {swaps}")
```

Slide 5: Matrix Multiplication with O(n³)

Cubic time complexity occurs in algorithms with three nested loops. Traditional matrix multiplication demonstrates O(n³) complexity through iterating over rows and columns while accumulating products, showing how computational complexity grows rapidly with input size.

```python
def matrix_multiply(A, B):
    n = len(A)
    result = [[0 for _ in range(n)] for _ in range(n)]
    operations = 0
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                operations += 1
                result[i][j] += A[i][k] * B[k][j]
                
    return result, operations

# Example usage with timing
import time

def generate_matrix(n):
    return [[i + j for j in range(n)] for i in range(n)]

n = 100
A = generate_matrix(n)
B = generate_matrix(n)

start_time = time.time()
result, ops = matrix_multiply(A, B)
end_time = time.time()

print(f"Matrix size: {n}x{n}")
print(f"Operations: {ops}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

Slide 6: Merge Sort and O(n log n)

Merge sort exemplifies the efficiency of divide-and-conquer algorithms with O(n log n) complexity. By recursively dividing the array and merging sorted subarrays, it achieves optimal performance for comparison-based sorting, making it suitable for large datasets.

```python
def merge_sort(arr):
    operations = {'comparisons': 0, 'merges': 0}
    
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            operations['comparisons'] += 1
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        operations['merges'] += 1
        return result
    
    if len(arr) <= 1:
        return arr, operations
        
    mid = len(arr) // 2
    left, _ = merge_sort(arr[:mid])
    right, _ = merge_sort(arr[mid:])
    
    return merge(left, right), operations

# Example usage with performance metrics
import random
import time

data = random.sample(range(10000), 1000)
start_time = time.time()
sorted_arr, stats = merge_sort(data)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.6f} seconds")
print(f"Comparisons: {stats['comparisons']}")
print(f"Merge operations: {stats['merges']}")
```

Slide 7: Fibonacci and O(2^n)

Exponential time complexity manifests in naive recursive implementations of problems like Fibonacci sequence calculation. This implementation demonstrates how runtime grows exponentially, making it impractical for even moderately large inputs.

```python
def fibonacci_exponential(n, calls=None):
    if calls is None:
        calls = {'count': 0}
    calls['count'] += 1
    
    if n <= 1:
        return n, calls
    
    fib1, _ = fibonacci_exponential(n-1, calls)
    fib2, _ = fibonacci_exponential(n-2, calls)
    return fib1 + fib2, calls

# Compare with dynamic programming approach
def fibonacci_linear(n):
    if n <= 1:
        return n, 1
    
    operations = 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        operations += 1
        a, b = b, a + b
    return b, operations

# Performance comparison
import time

n = 30
print("Exponential approach:")
start = time.time()
result, stats = fibonacci_exponential(n)
print(f"Result: {result}")
print(f"Function calls: {stats['count']}")
print(f"Time: {time.time() - start:.6f} seconds")

print("\nLinear approach:")
start = time.time()
result, ops = fibonacci_linear(n)
print(f"Result: {result}")
print(f"Operations: {ops}")
print(f"Time: {time.time() - start:.6f} seconds")
```

Slide 8: Factorial Time Complexity O(n!)

Factorial complexity represents algorithms whose runtime grows with the factorial of the input size. The generation of all possible permutations demonstrates this extreme growth rate, making it practical only for very small inputs.

```python
def generate_permutations(arr):
    operations = {'count': 0}
    
    def permute(elements, current=[]):
        operations['count'] += 1
        if not elements:
            return [current]
        
        perms = []
        for i, e in enumerate(elements):
            remaining = elements[:i] + elements[i+1:]
            perms.extend(permute(remaining, current + [e]))
        return perms
    
    result = permute(arr)
    return result, operations

# Example with performance measurement
import time

# Test with small input due to factorial growth
test_array = list(range(8))
start_time = time.time()
perms, stats = generate_permutations(test_array)
end_time = time.time()

print(f"Input size: {len(test_array)}")
print(f"Number of permutations: {len(perms)}")
print(f"Operations performed: {stats['count']}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

Slide 9: Square Root Time O(sqrt(n))

Square root time complexity appears in algorithms that leverage mathematical properties to reduce the search space. The Sieve of Eratosthenes for finding prime numbers demonstrates this complexity by optimizing divisibility checks.

```python
def sieve_of_eratosthenes(n):
    operations = 0
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            operations += 1
            for j in range(i * i, n + 1, i):
                sieve[j] = False
                operations += 1
    
    primes = [i for i in range(n + 1) if sieve[i]]
    return primes, operations

# Example usage with performance metrics
import time

n = 1000000
start_time = time.time()
primes, ops = sieve_of_eratosthenes(n)
end_time = time.time()

print(f"Found {len(primes)} prime numbers up to {n}")
print(f"Operations performed: {ops}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

Slide 10: Real-World Application - Data Search Engine

A practical implementation combining multiple time complexities to create a simple search engine. This example demonstrates how different complexity patterns interact in real applications through indexing and searching operations.

```python
class SearchEngine:
    def __init__(self):
        self.index = {}  # O(1) access time
        self.documents = {}
        self.operations = {'indexing': 0, 'searching': 0}
    
    def add_document(self, doc_id, content):
        # O(n) where n is words in content
        words = content.lower().split()
        self.documents[doc_id] = content
        
        for position, word in enumerate(words):
            if word not in self.index:
                self.index[word] = {}
            if doc_id not in self.index[word]:
                self.index[word][doc_id] = []
            self.index[word][doc_id].append(position)
            self.operations['indexing'] += 1
    
    def search(self, query):
        # O(m * log n) where m is query words and n is matching documents
        words = query.lower().split()
        results = {}
        
        for word in words:
            if word in self.index:
                for doc_id in self.index[word]:
                    results[doc_id] = results.get(doc_id, 0) + len(self.index[word][doc_id])
                    self.operations['searching'] += 1
        
        return sorted(results.items(), key=lambda x: x[1], reverse=True)

# Example usage with performance metrics
import time

# Create sample documents
documents = {
    1: "The quick brown fox jumps over the lazy dog",
    2: "A quick brown cat sleeps on the mat",
    3: "The lazy dog runs away from the quick fox"
}

# Initialize and test search engine
engine = SearchEngine()
start_time = time.time()

# Index documents
for doc_id, content in documents.items():
    engine.add_document(doc_id, content)

# Perform searches
queries = ["quick", "lazy dog", "brown fox"]
results = {}

for query in queries:
    results[query] = engine.search(query)

end_time = time.time()

print(f"Indexing operations: {engine.operations['indexing']}")
print(f"Search operations: {engine.operations['searching']}")
print(f"Total time: {end_time - start_time:.6f} seconds")

# Display search results
for query, matches in results.items():
    print(f"\nResults for '{query}':")
    for doc_id, relevance in matches:
        print(f"Document {doc_id}: {documents[doc_id]} (relevance: {relevance})")
```

Slide 11: Optimizing Database Queries with Indexes

This example demonstrates how proper indexing can reduce query complexity from O(n) to O(log n) in database-like operations, showing practical application of big O notation in data management.

```python
class OptimizedDatabase:
    def __init__(self):
        self.data = []
        self.index = {}
        self.operations = {'insert': 0, 'search': 0}
    
    def insert(self, key, value):
        # O(log n) insertion with index maintenance
        position = len(self.data)
        self.data.append((key, value))
        
        # Update index (B-tree simulation)
        if key in self.index:
            self.index[key].append(position)
        else:
            self.index[key] = [position]
        
        self.operations['insert'] += 1
        
    def search(self, key):
        # O(1) for index lookup, O(log n) for sorted results
        self.operations['search'] += 1
        
        if key in self.index:
            return [(self.data[pos][1], pos) for pos in self.index[key]]
        return []
    
    def range_search(self, start_key, end_key):
        # O(k log n) where k is the number of keys in range
        results = []
        for key in sorted(self.index.keys()):
            if start_key <= key <= end_key:
                results.extend(self.search(key))
                self.operations['search'] += 1
        return sorted(results, key=lambda x: x[1])

# Performance testing
import random
import time

db = OptimizedDatabase()
test_data = [(i, f"value_{i}") for i in range(1000)]
random.shuffle(test_data)

# Insert test
start_time = time.time()
for key, value in test_data:
    db.insert(key, value)
insert_time = time.time() - start_time

# Search test
start_time = time.time()
searches = [random.randint(0, 999) for _ in range(100)]
for key in searches:
    results = db.search(key)
search_time = time.time() - start_time

# Range search test
start_time = time.time()
range_results = db.range_search(100, 200)
range_time = time.time() - start_time

print(f"Insert operations: {db.operations['insert']}")
print(f"Search operations: {db.operations['search']}")
print(f"Insert time: {insert_time:.6f} seconds")
print(f"Search time: {search_time:.6f} seconds")
print(f"Range search time: {range_time:.6f} seconds")
```

Slide 12: Graph Traversal Complexities

Understanding time complexities in graph algorithms is crucial for network analysis and pathfinding. This implementation demonstrates both Depth-First Search (DFS) and Breadth-First Search (BFS) with their respective complexities of O(V + E).

```python
from collections import defaultdict, deque
import time

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.metrics = {'operations': 0, 'visited_nodes': 0}
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
    
    def dfs(self, start):
        visited = set()
        path = []
        
        def dfs_util(vertex):
            self.metrics['operations'] += 1
            visited.add(vertex)
            path.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_util(neighbor)
        
        dfs_util(start)
        self.metrics['visited_nodes'] = len(visited)
        return path
    
    def bfs(self, start):
        visited = set([start])
        queue = deque([start])
        path = []
        
        while queue:
            self.metrics['operations'] += 1
            vertex = queue.popleft()
            path.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        self.metrics['visited_nodes'] = len(visited)
        return path

# Example usage with performance comparison
def create_test_graph():
    g = Graph()
    # Creating a sample graph
    edges = [
        (0, 1), (0, 2), (1, 2), (2, 0),
        (2, 3), (3, 3), (1, 3), (3, 4),
        (4, 5), (5, 6), (6, 4)
    ]
    for u, v in edges:
        g.add_edge(u, v)
    return g

# Performance testing
graph = create_test_graph()

# DFS Test
start_time = time.time()
dfs_path = graph.dfs(0)
dfs_time = time.time() - start_time
dfs_ops = graph.metrics['operations']
dfs_visited = graph.metrics['visited_nodes']

# Reset metrics
graph.metrics = {'operations': 0, 'visited_nodes': 0}

# BFS Test
start_time = time.time()
bfs_path = graph.bfs(0)
bfs_time = time.time() - start_time
bfs_ops = graph.metrics['operations']
bfs_visited = graph.metrics['visited_nodes']

print("DFS Results:")
print(f"Path: {dfs_path}")
print(f"Operations: {dfs_ops}")
print(f"Nodes visited: {dfs_visited}")
print(f"Time: {dfs_time:.6f} seconds\n")

print("BFS Results:")
print(f"Path: {bfs_path}")
print(f"Operations: {bfs_ops}")
print(f"Nodes visited: {bfs_visited}")
print(f"Time: {bfs_time:.6f} seconds")
```

Slide 13: Space Complexity Analysis

Demonstrating how different data structures and algorithms affect memory usage, this implementation compares space complexities from O(1) to O(n²) with practical memory measurements.

```python
import sys
import time
import numpy as np

class SpaceComplexityDemo:
    def __init__(self):
        self.metrics = {'memory_usage': [], 'creation_time': []}
    
    def constant_space(self, n):
        # O(1) space complexity
        start_time = time.time()
        x = 0
        for i in range(n):
            x += i
        return sys.getsizeof(x), time.time() - start_time
    
    def linear_space(self, n):
        # O(n) space complexity
        start_time = time.time()
        arr = list(range(n))
        return sys.getsizeof(arr) + sum(sys.getsizeof(x) for x in arr), time.time() - start_time
    
    def quadratic_space(self, n):
        # O(n²) space complexity
        start_time = time.time()
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        return sys.getsizeof(matrix) + sum(sys.getsizeof(row) for row in matrix), time.time() - start_time
    
    def analyze_complexity(self, sizes):
        results = {'constant': [], 'linear': [], 'quadratic': []}
        
        for n in sizes:
            # Measure each complexity
            const_mem, const_time = self.constant_space(n)
            lin_mem, lin_time = self.linear_space(n)
            quad_mem, quad_time = self.quadratic_space(n)
            
            # Store results
            results['constant'].append((const_mem, const_time))
            results['linear'].append((lin_mem, lin_time))
            results['quadratic'].append((quad_mem, quad_time))
        
        return results

# Run analysis
demo = SpaceComplexityDemo()
test_sizes = [100, 1000, 10000]
results = demo.analyze_complexity(test_sizes)

# Print results
for complexity, data in results.items():
    print(f"\n{complexity.capitalize()} Space Complexity:")
    for i, size in enumerate(test_sizes):
        memory, time = data[i]
        print(f"Size {size}:")
        print(f"Memory usage: {memory} bytes")
        print(f"Creation time: {time:.6f} seconds")
```

Slide 14: Additional Resources

*   Research paper on algorithm complexity analysis: [https://arxiv.org/abs/1902.05100](https://arxiv.org/abs/1902.05100)
*   Survey of space-time complexity tradeoffs: [https://arxiv.org/abs/1909.07395](https://arxiv.org/abs/1909.07395)
*   Practical applications of Big O in machine learning: [https://arxiv.org/abs/2007.12823](https://arxiv.org/abs/2007.12823)
*   Optimization techniques for algorithm design: [https://www.sciencedirect.com/journal/algorithmica](https://www.sciencedirect.com/journal/algorithmica)
*   Modern approaches to complexity analysis: [https://dl.acm.org/doi/10.1145/algorithms-complexity](https://dl.acm.org/doi/10.1145/algorithms-complexity)
*   Google Scholar search suggestions:
    *   "Big O notation practical applications"
    *   "Algorithm complexity analysis modern techniques"
    *   "Space-time complexity tradeoffs in data structures"

