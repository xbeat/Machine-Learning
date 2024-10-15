## Data Structures and Algorithms for AI Enthusiasts
Slide 1: Introduction to Data Structures and Algorithms for AI

Data Structures and Algorithms (DSA) form the backbone of efficient AI systems. They enable us to organize, store, and process data effectively, which is crucial for developing intelligent algorithms. This presentation will explore key DSA concepts relevant to AI, with a focus on Python implementations.

```python
# A simple example demonstrating the importance of DSA in AI
import time

def linear_search(arr, target):
    for i, num in enumerate(arr):
        if num == target:
            return i
    return -1

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

# Generate a large sorted array
data = list(range(1000000))

# Measure time for linear search
start = time.time()
linear_search(data, 999999)
linear_time = time.time() - start

# Measure time for binary search
start = time.time()
binary_search(data, 999999)
binary_time = time.time() - start

print(f"Linear search time: {linear_time:.6f} seconds")
print(f"Binary search time: {binary_time:.6f} seconds")
print(f"Binary search is {linear_time / binary_time:.2f}x faster")
```

Slide 2: Arrays and Lists in Python

Arrays and lists are fundamental data structures in Python. They store collections of elements and provide efficient access to individual items. In Python, lists are more commonly used and offer greater flexibility.

```python
# Creating and manipulating lists in Python
fruits = ["apple", "banana", "cherry"]
print(f"Original list: {fruits}")

# Adding elements
fruits.append("date")
fruits.insert(1, "blueberry")
print(f"After adding elements: {fruits}")

# Accessing elements
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")

# Slicing
print(f"First three fruits: {fruits[:3]}")

# List comprehension
uppercase_fruits = [fruit.upper() for fruit in fruits]
print(f"Uppercase fruits: {uppercase_fruits}")

# Sorting
fruits.sort()
print(f"Sorted fruits: {fruits}")
```

Slide 3: Linked Lists

Linked lists are linear data structures where elements are stored in nodes. Each node contains data and a reference to the next node. They are useful when frequent insertions and deletions are required.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def display(self):
        elements = []
        current = self.head
        while current:
            elements.append(current.data)
            current = current.next
        return elements

# Create and manipulate a linked list
ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(f"Linked list: {ll.display()}")
```

Slide 4: Stacks and Queues

Stacks and queues are linear data structures that follow specific orders for adding and removing elements. Stacks follow Last-In-First-Out (LIFO), while queues follow First-In-First-Out (FIFO).

```python
from collections import deque

# Stack implementation using a list
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
print(f"Stack: {stack}")
print(f"Popped item: {stack.pop()}")
print(f"Stack after pop: {stack}")

# Queue implementation using deque
queue = deque()
queue.append(1)
queue.append(2)
queue.append(3)
print(f"Queue: {queue}")
print(f"Dequeued item: {queue.popleft()}")
print(f"Queue after dequeue: {queue}")
```

Slide 5: Trees and Binary Search Trees

Trees are hierarchical data structures with a root node and child nodes. Binary Search Trees (BST) are special trees where each node has at most two children, and the left subtree contains smaller values while the right subtree contains larger values.

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        self.root = self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if node is None:
            return TreeNode(value)
        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        else:
            node.right = self._insert_recursive(node.right, value)
        return node

    def inorder_traversal(self):
        return self._inorder_recursive(self.root)

    def _inorder_recursive(self, node):
        if node is None:
            return []
        return (self._inorder_recursive(node.left) +
                [node.value] +
                self._inorder_recursive(node.right))

# Create and manipulate a BST
bst = BinarySearchTree()
for value in [5, 3, 7, 1, 4, 6, 8]:
    bst.insert(value)
print(f"BST inorder traversal: {bst.inorder_traversal()}")
```

Slide 6: Graphs

Graphs are versatile data structures consisting of vertices (nodes) and edges (connections between nodes). They are essential in AI for representing complex relationships and solving problems like pathfinding and social network analysis.

```python
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        self.graph[u].append(v)
        self.graph[v].append(u)

    def bfs(self, start):
        visited = set()
        queue = [start]
        visited.add(start)
        while queue:
            vertex = queue.pop(0)
            print(vertex, end=" ")
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

# Create and traverse a graph
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)
g.add_edge(3, 3)

print("BFS traversal starting from vertex 2:")
g.bfs(2)
```

Slide 7: Hash Tables

Hash tables provide fast insertion, deletion, and lookup operations. They use a hash function to map keys to array indices, enabling efficient data retrieval. In Python, dictionaries are implemented as hash tables.

```python
class SimpleHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        for item in self.table[index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[index].append([key, value])

    def get(self, key):
        index = self._hash(key)
        for item in self.table[index]:
            if item[0] == key:
                return item[1]
        raise KeyError(key)

    def remove(self, key):
        index = self._hash(key)
        for i, item in enumerate(self.table[index]):
            if item[0] == key:
                del self.table[index][i]
                return
        raise KeyError(key)

# Use the SimpleHashTable
ht = SimpleHashTable(10)
ht.insert("apple", 5)
ht.insert("banana", 7)
ht.insert("cherry", 3)

print(f"Value of 'banana': {ht.get('banana')}")
ht.remove("apple")
print("'apple' removed from the hash table")
```

Slide 8: Sorting Algorithms

Sorting algorithms arrange data in a specific order, which is crucial for efficient searching and data processing. Common sorting algorithms include Bubble Sort, Merge Sort, and Quick Sort.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Test sorting algorithms
arr = [64, 34, 25, 12, 22, 11, 90]
print(f"Original array: {arr}")
print(f"Bubble sorted: {bubble_sort(arr.())}")
print(f"Merge sorted: {merge_sort(arr)}")
```

Slide 9: Searching Algorithms

Searching algorithms find specific elements in a dataset. Linear search and binary search are common techniques, with binary search being more efficient for sorted data.

```python
def linear_search(arr, target):
    for i, num in enumerate(arr):
        if num == target:
            return i
    return -1

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

# Test searching algorithms
arr = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
print(f"Array: {arr}")
print(f"Linear search for {target}: {linear_search(arr, target)}")
print(f"Binary search for {target}: {binary_search(arr, target)}")
```

Slide 10: Dynamic Programming

Dynamic programming is an algorithmic paradigm that solves complex problems by breaking them down into simpler subproblems. It's particularly useful in optimization problems and can significantly improve efficiency.

```python
def fibonacci_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

def longest_common_subsequence(X, Y):
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]

# Test dynamic programming algorithms
print(f"10th Fibonacci number: {fibonacci_dp(10)}")
X, Y = "AGGTAB", "GXTXAYB"
print(f"Length of Longest Common Subsequence: {longest_common_subsequence(X, Y)}")
```

Slide 11: Greedy Algorithms

Greedy algorithms make locally optimal choices at each step, aiming to find a global optimum. They are often used for optimization problems but may not always yield the best solution.

```python
def fractional_knapsack(values, weights, capacity):
    items = sorted(zip(values, weights),
                   key=lambda x: x[0]/x[1], reverse=True)
    total_value = 0
    for value, weight in items:
        if capacity >= weight:
            total_value += value
            capacity -= weight
        else:
            total_value += value * (capacity / weight)
            break
    return total_value

# Test greedy algorithm
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
max_value = fractional_knapsack(values, weights, capacity)
print(f"Maximum value in Knapsack: {max_value}")
```

Slide 12: Real-life Example: Recommendation Systems

Recommendation systems are a common application of DSA in AI. They use various algorithms to suggest items to users based on their preferences and behavior.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simple collaborative filtering for movie recommendations
user_ratings = np.array([
    [4, 3, 0, 5, 0],
    [5, 0, 4, 0, 2],
    [3, 1, 2, 4, 1],
    [0, 0, 0, 2, 0],
    [1, 0, 3, 0, 0]
])

def recommend_movies(user_id, user_ratings, n_recommendations=2):
    user_similarities = cosine_similarity(user_ratings)
    user_sim_scores = user_similarities[user_id]
    similar_users = user_sim_scores.argsort()[::-1][1:]
    
    recommendations = []
    for movie in range(user_ratings.shape[1]):
        if user_ratings[user_id][movie] == 0:  # User hasn't rated this movie
            weighted_sum = sum(user_sim_scores[u] * user_ratings[u][movie]
                               for u in similar_users if user_ratings[u][movie] > 0)
            total_similarity = sum(user_sim_scores[u]
                                   for u in similar_users if user_ratings[u][movie] > 0)
            if total_similarity > 0:
                recommendations.append((movie, weighted_sum / total_similarity))
    
    return sorted(recommendations, key=lambda x: x[1], reverse=True)[:n_recommendations]

# Get recommendations for user 0
user_id = 0
recommendations = recommend_movies(user_id, user_ratings)
print(f"Recommended movies for user {user_id}:")
for movie, score in recommendations:
    print(f"Movie {movie}: Predicted rating {score:.2f}")
```

Slide 13: Real-life Example: Graph-based Social Network Analysis

Social network analysis is another application of DSA in AI, particularly using graph algorithms to analyze relationships and influence in networks.

```python
import networkx as nx

# Create a sample social network
G = nx.Graph()
G.add_edges_from([
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)
])

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Find the most influential node based on degree centrality
most_influential = max(degree_centrality, key=degree_centrality.get)

print(f"Most influential node: {most_influential}")
print(f"Degree centrality: {degree_centrality[most_influential]:.4f}")
print(f"Betweenness centrality: {betweenness_centrality[most_influential]:.4f}")
print(f"Closeness centrality: {closeness_centrality[most_influential]:.4f}")

# Find communities using the Louvain method
communities = nx.community.louvain_communities(G)
print(f"Number of communities: {len(communities)}")
print("Communities:", communities)
```

Slide 14: Algorithms for Natural Language Processing

Natural Language Processing (NLP) is a crucial area in AI that heavily relies on efficient data structures and algorithms. Here's an example of implementing a simple tokenizer and a bag-of-words model.

```python
import re
from collections import Counter

def tokenize(text):
    # Convert to lowercase and split into words
    words = re.findall(r'\w+', text.lower())
    return words

def create_bow(documents):
    # Create a vocabulary from all documents
    vocabulary = set(word for doc in documents for word in tokenize(doc))
    
    # Create bag-of-words for each document
    bow = []
    for doc in documents:
        word_counts = Counter(tokenize(doc))
        doc_bow = [word_counts.get(word, 0) for word in vocabulary]
        bow.append(doc_bow)
    
    return list(vocabulary), bow

# Example usage
documents = [
    "Natural language processing is fascinating",
    "Machine learning algorithms are powerful",
    "Data structures are essential for efficient algorithms"
]

vocabulary, bow = create_bow(documents)

print("Vocabulary:", vocabulary)
print("Bag-of-Words representations:")
for i, doc_bow in enumerate(bow):
    print(f"Document {i + 1}:", doc_bow)
```

Slide 15: Additional Resources

For those interested in diving deeper into Data Structures and Algorithms for AI, here are some valuable resources:

1. "Algorithm Design for AI" by Hutter et al. (2021) - ArXiv:2102.01868 URL: [https://arxiv.org/abs/2102.01868](https://arxiv.org/abs/2102.01868)
2. "Graph Neural Networks: A Review of Methods and Applications" by Zhou et al. (2018) - ArXiv:1812.08434 URL: [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
3. "Efficient Transformers: A Survey" by Tay et al. (2020) - ArXiv:2009.06732 URL: [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)

These papers provide in-depth discussions on advanced algorithms and data structures used in modern AI systems. Remember to verify the information and check for updates, as the field of AI is rapidly evolving.

