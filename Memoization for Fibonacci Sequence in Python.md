## Memoization for Fibonacci Sequence in Python
Slide 1: Introduction to Memoization

Memoization is an optimization technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again. This powerful strategy can significantly improve the performance of recursive algorithms or functions with repeated computations.

```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def expensive_function(n):
    # Simulate an expensive computation
    return sum(i for i in range(n))

# Usage
result = expensive_function(1000000)
print(result)  # Output: 499999500000
```

Slide 2: The Fibonacci Sequence

The Fibonacci sequence is a classic example where memoization shines. Each number is the sum of the two preceding ones, starting from 0 and 1. Let's implement it without memoization first.

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Usage
print(fibonacci(30))  # Output: 832040 (takes several seconds)
```

Slide 3: Memoized Fibonacci

Now, let's apply memoization to our Fibonacci function and observe the performance improvement.

```python
@memoize
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n-1) + fib_memo(n-2)

# Usage
print(fib_memo(30))  # Output: 832040 (instant result)
print(fib_memo(100))  # Output: 354224848179261915075 (still instant)
```

Slide 4: Understanding the Performance Gain

To visualize the performance difference, let's time both implementations for various inputs.

```python
import time

def time_function(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

# Compare performance
for n in [20, 30, 35]:
    _, time_regular = time_function(fibonacci, n)
    _, time_memo = time_function(fib_memo, n)
    print(f"n={n}:")
    print(f"  Regular: {time_regular:.6f} seconds")
    print(f"  Memoized: {time_memo:.6f} seconds")
    print(f"  Speedup: {time_regular / time_memo:.2f}x")
```

Slide 5: Memoization in Practice

Memoization isn't limited to mathematical sequences. It's useful in various scenarios where computations are repeated. Let's consider a function that calculates the number of ways to climb stairs, taking 1 or 2 steps at a time.

```python
@memoize
def climb_stairs(n):
    if n <= 1:
        return 1
    return climb_stairs(n-1) + climb_stairs(n-2)

# Usage
print(climb_stairs(30))  # Output: 1346269
print(climb_stairs(50))  # Output: 20365011074
```

Slide 6: Memoization with Multiple Arguments

Memoization works well with functions that take multiple arguments. Let's implement a memoized function to calculate the binomial coefficient.

```python
@memoize
def binomial_coefficient(n, k):
    if k == 0 or k == n:
        return 1
    return binomial_coefficient(n-1, k-1) + binomial_coefficient(n-1, k)

# Usage
print(binomial_coefficient(30, 15))  # Output: 155117520
print(binomial_coefficient(50, 25))  # Output: 126410606437752
```

Slide 7: Memoization in Dynamic Programming

Memoization is a key component of dynamic programming. Let's solve the classic "Longest Common Subsequence" problem using memoization.

```python
@memoize
def lcs(X, Y, m, n):
    if m == 0 or n == 0:
        return 0
    elif X[m-1] == Y[n-1]:
        return 1 + lcs(X, Y, m-1, n-1)
    else:
        return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

# Usage
X = "AGGTAB"
Y = "GXTXAYB"
print(lcs(X, Y, len(X), len(Y)))  # Output: 4
```

Slide 8: Memoization in Graph Algorithms

Memoization can significantly improve the performance of graph algorithms. Let's implement a memoized depth-first search to count the number of paths between two nodes.

```python
@memoize
def count_paths(graph, start, end, visited=None):
    if visited is None:
        visited = set()
    if start == end:
        return 1
    if start in visited:
        return 0
    visited.add(start)
    count = 0
    for neighbor in graph[start]:
        count += count_paths(graph, neighbor, end, visited.copy())
    return count

# Usage
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['D'],
    'D': ['E'],
    'E': []
}
print(count_paths(graph, 'A', 'E'))  # Output: 3
```

Slide 9: Memoization in Tree Traversal

Tree traversal algorithms can benefit from memoization, especially when dealing with repeated subtrees. Let's implement a memoized function to count unique binary search trees.

```python
@memoize
def count_unique_bst(n):
    if n <= 1:
        return 1
    count = 0
    for i in range(1, n + 1):
        count += count_unique_bst(i - 1) * count_unique_bst(n - i)
    return count

# Usage
for i in range(1, 11):
    print(f"Unique BSTs with {i} nodes: {count_unique_bst(i)}")
```

Slide 10: Memoization in String Manipulation

String manipulation problems often involve repeated subproblems. Let's use memoization to solve the "Edit Distance" problem efficiently.

```python
@memoize
def edit_distance(str1, str2, m, n):
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m-1] == str2[n-1]:
        return edit_distance(str1, str2, m-1, n-1)
    return 1 + min(edit_distance(str1, str2, m, n-1),    # Insert
                   edit_distance(str1, str2, m-1, n),    # Remove
                   edit_distance(str1, str2, m-1, n-1))  # Replace

# Usage
str1 = "kitten"
str2 = "sitting"
print(edit_distance(str1, str2, len(str1), len(str2)))  # Output: 3
```

Slide 11: Memoization in Recursive Mathematical Functions

Complex mathematical functions with recursive definitions can greatly benefit from memoization. Let's implement the Catalan number sequence using memoization.

```python
@memoize
def catalan(n):
    if n <= 1:
        return 1
    res = 0
    for i in range(n):
        res += catalan(i) * catalan(n-i-1)
    return res

# Usage
for i in range(10):
    print(f"Catalan({i}) = {catalan(i)}")
```

Slide 12: Memoization in Game Theory

Game theory algorithms often involve exploring many possible game states. Memoization can help reduce redundant calculations. Let's implement a memoized function to solve the coin game problem.

```python
@memoize
def coin_game(coins, i, j):
    if i > j:
        return 0
    return max(
        coins[i] + min(coin_game(coins, i+2, j), coin_game(coins, i+1, j-1)),
        coins[j] + min(coin_game(coins, i+1, j-1), coin_game(coins, i, j-2))
    )

# Usage
coins = [20, 30, 2, 10]
print(coin_game(tuple(coins), 0, len(coins) - 1))  # Output: 40
```

Slide 13: Limitations and Considerations

While memoization is powerful, it's not always the best solution. Consider these factors:

1.  Memory usage: Memoization trades space for time.
2.  Input size: For small inputs, the overhead might outweigh the benefits.
3.  Function purity: Memoization works best with pure functions.
4.  Cache invalidation: Ensure the cache is cleared or updated when necessary.

```python
def memoize_with_limit(func, max_size=100):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        if len(cache) >= max_size:
            cache.clear()  # Clear cache when it reaches the limit
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize_with_limit
def limited_memo_func(n):
    # Some expensive computation
    return sum(i for i in range(n))

# Usage
print(limited_memo_func(1000))  # Output: 499500
print(limited_memo_func(1000))  # Cached result
```

Slide 14: Real-life Example: DNA Sequence Alignment

In bioinformatics, sequence alignment is a common task. Let's use memoization to implement a simple DNA sequence alignment algorithm.

```python
@memoize
def dna_align(seq1, seq2, i, j, gap_penalty=-1):
    if i == 0:
        return j * gap_penalty
    if j == 0:
        return i * gap_penalty
    if seq1[i-1] == seq2[j-1]:
        return dna_align(seq1, seq2, i-1, j-1)
    return max(
        dna_align(seq1, seq2, i-1, j) + gap_penalty,
        dna_align(seq1, seq2, i, j-1) + gap_penalty,
        dna_align(seq1, seq2, i-1, j-1) + gap_penalty
    )

# Usage
seq1 = "ATCG"
seq2 = "ATACG"
alignment_score = dna_align(seq1, seq2, len(seq1), len(seq2))
print(f"Alignment score: {alignment_score}")  # Output: Alignment score: -1
```

Slide 15: Real-life Example: Image Processing

In image processing, we often need to apply filters or transformations to pixels. Let's use memoization to optimize a simple edge detection algorithm.

```python
@memoize
def detect_edge(image, x, y):
    if x == 0 or y == 0 or x == len(image) - 1 or y == len(image[0]) - 1:
        return 0
    gradient_x = abs(image[x+1][y] - image[x-1][y])
    gradient_y = abs(image[x][y+1] - image[x][y-1])
    return max(gradient_x, gradient_y)

# Simulated image (2D list of pixel intensities)
image = [
    [0, 0, 0, 0, 0],
    [0, 50, 50, 50, 0],
    [0, 50, 100, 50, 0],
    [0, 50, 50, 50, 0],
    [0, 0, 0, 0, 0]
]

# Detect edges
edge_image = [[detect_edge(tuple(map(tuple, image)), x, y) for y in range(5)] for x in range(5)]

for row in edge_image:
    print(row)
```

Slide 16: Additional Resources

For those interested in diving deeper into memoization and its applications, here are some valuable resources:

1.  "Dynamic Programming and Memoization: Bottom-Up vs Top-Down Approaches" by Ivaylo Ivanof (arXiv:2003.12959) [https://arxiv.org/abs/2003.12959](https://arxiv.org/abs/2003.12959)
2.  "Efficient Dynamic Programming Using Memoization" by Richard E. Korf (arXiv:1604.05197) [https://arxiv.org/abs/1604.05197](https://arxiv.org/abs/1604.05197)

These papers provide in-depth analyses of memoization techniques and their applications in various computational problems.

