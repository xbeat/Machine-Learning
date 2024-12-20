## Heap Data Structure in Python

Slide 1: Introduction to Heaps

Introduction to Heaps

Heaps are tree-based data structures that satisfy the heap property: for a max heap, the value at each node is greater than or equal to the values of its children, and for a min heap, the value at each node is less than or equal to the values of its children. Heaps are commonly used to implement priority queues and are widely used in algorithms like Dijkstra's shortest path algorithm and the heap sort algorithm.

```python
import heapq

# Creating a min heap
minHeap = []
heapq.heappush(minHeap, 5)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 8)
heapq.heappush(minHeap, 1)

print(minHeap)  # Output: [1, 2, 8, 5]

# Creating a max heap
maxHeap = [-x for x in minHeap]
heapq.heapify(maxHeap)

print(maxHeap)  # Output: [-8, -5, -1, -2]
```

Slide 2: Heap Operations

Heap Operations

Heaps support two main operations: `heappush` and `heappop`. The `heappush` operation adds an element to the heap, and the `heappop` operation removes and returns the root element (the minimum or maximum element, depending on the heap type).

```python
import heapq

# Creating a min heap
minHeap = []

# Pushing elements to the heap
heapq.heappush(minHeap, 5)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 8)
heapq.heappush(minHeap, 1)

print(minHeap)  # Output: [1, 2, 8, 5]

# Popping elements from the heap
smallest = heapq.heappop(minHeap)
print(smallest)  # Output: 1
print(minHeap)   # Output: [2, 5, 8]
```

Slide 3: Building a Heap

Building a Heap

The `heapify` function is used to convert a regular list into a heap data structure in-place. This operation has a time complexity of O(n), where n is the number of elements in the list.

```python
import heapq

# Creating a list
numbers = [5, 2, 8, 1, 9, 3]

# Converting the list to a min heap
heapq.heapify(numbers)

print(numbers)  # Output: [1, 2, 3, 5, 9, 8]
```

Slide 4: Heap Sort Algorithm

Heap Sort Algorithm

Heap sort is a comparison-based sorting algorithm that works by first creating a max heap from the input data. It then repeatedly swaps the root element (the maximum value) with the last element of the heap and reduces the heap size by one. After n swaps, the data is sorted in ascending order.

```python
import heapq

def heapsort(nums):
    heap = nums.copy()
    heapq.heapify(heap)
    sorted_nums = []

    while heap:
        largest = heapq.heappop(heap)
        sorted_nums.append(largest)

    return sorted_nums[::-1]

# Example usage
unsorted_nums = [5, 2, 8, 1, 9, 3]
sorted_nums = heapsort(unsorted_nums)
print(sorted_nums)  # Output: [1, 2, 3, 5, 8, 9]
```

Slide 5: Heap Implementation

Heap Implementation

Python's `heapq` module provides a heap implementation using a list as the underlying data structure. However, you can also implement a heap from scratch using a tree data structure. Here's an example of a binary heap implementation:

```python
class BinaryHeap:
    def __init__(self):
        self.heap = []

    def push(self, value):
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            return None

        root = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        self._sift_down(0)

        return root

    def _sift_up(self, index):
        parent = (index - 1) // 2
        if parent >= 0 and self.heap[parent] > self.heap[index]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            self._sift_up(parent)

    def _sift_down(self, index):
        left_child = 2 * index + 1
        right_child = 2 * index + 2

        if left_child >= len(self.heap):
            return

        min_index = index
        if self.heap[left_child] < self.heap[min_index]:
            min_index = left_child

        if right_child < len(self.heap) and self.heap[right_child] < self.heap[min_index]:
            min_index = right_child

        if min_index != index:
            self.heap[index], self.heap[min_index] = self.heap[min_index], self.heap[index]
            self._sift_down(min_index)
```

Slide 6: Heap Applications

Heap Applications

Heaps have various applications in computer science and algorithms, including:

1. Priority Queues: Heaps are commonly used to implement priority queues, where elements are sorted based on their priority.
2. Graph Algorithms: Dijkstra's algorithm for finding the shortest path in a weighted graph uses a min heap to efficiently track the shortest distances.
3. Huffman Coding: Huffman coding, used for data compression, uses a min heap to construct an optimal prefix code.
4. Heap Sort: The heap sort algorithm is an efficient comparison-based sorting algorithm that uses a max heap to sort elements in ascending order.

```python
# Example: Using a min heap as a priority queue
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def pop(self):
        return heapq.heappop(self.heap)[1]

# Example usage
pq = PriorityQueue()
pq.push("Task 1", 2)
pq.push("Task 2", 1)
pq.push("Task 3", 3)

print(pq.pop())  # Output: Task 2
print(pq.pop())  # Output: Task 1
print(pq.pop())  # Output: Task 3
```

Slide 7: Heap Complexity

Heap Complexity

The time complexity of heap operations depends on the height of the heap, which is logarithmic in the number of elements (n). The main operations and their time complexities are:

* `heappush`: O(log n)
* `heappop`: O(log n)
* `heapify`: O(n)

Space complexity: The space complexity of a heap is O(n), where n is the number of elements in the heap, as it requires storing all the elements in memory.

```python
import heapq
import time
import random

# Generate a list of random numbers
nums = [random.randint(1, 1000000) for _ in range(100000)]

# Time the heapify operation
start_time = time.time()
heapq.heapify(nums)
end_time = time.time()
print(f"Heapify took {end_time - start_time:.6f} seconds")

# Time the heappush and heappop operations
heap = []
start_time = time.time()
for num in nums:
    heapq.heappush(heap, num)
end_time = time.time()
print(f"Pushing {len(nums)} elements took {end_time - start_time:.6f} seconds")

start_time = time.time()
while heap:
    heapq.heappop(heap)
end_time = time.time()
print(f"Popping {len(nums)} elements took {end_time - start_time:.6f} seconds")
```

This example generates a list of random numbers and measures the time taken for the `heapify`, `heappush`, and `heappop` operations. The time taken for each operation is printed, demonstrating the efficiency of heap operations for large datasets.

Slide 8: Heap Interview Questions

Heap Interview Questions

Heaps are a common topic in coding interviews, and here are a few examples of heap-related questions:

1. Merge k Sorted Lists: Given k sorted linked lists, merge them into a single sorted linked list using heaps.
2. Top K Frequent Elements: Given a non-empty array of integers, find the top k most frequent elements using heaps.
3. Sliding Window Maximum: Given an array of integers and a window size, find the maximum value in each sliding window using heaps.
4. Kth Largest Element in a Stream: Given a stream of integers, find the kth largest element at any given time using heaps.

```python
# Example: Merge k Sorted Lists
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    heap = []
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(heap, (lists[i].val, i, lists[i]))

    dummy = ListNode()
    curr = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

Slide 9: Heap Applications in Real-World

Heap Applications in Real-World

Heaps have several real-world applications across various domains, including:

1. Operating Systems: Heaps are used in process and job scheduling, managing memory allocation, and implementing priority-based scheduling algorithms.
2. Data Compression: Huffman coding, a popular data compression algorithm, uses min heaps to construct an optimal prefix code.
3. Network Routing: Network routing protocols, such as Dijkstra's algorithm, use min heaps to find the shortest path between nodes.
4. Machine Learning: Heaps are used in various machine learning algorithms, such as k-nearest neighbors and k-means clustering.
5. Event Management: Priority queues implemented using heaps are useful in event management systems, where events are processed based on their priority.

```python
# Example: Dijkstra's Algorithm for Shortest Path
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
```

Slide 10: Heap vs. Other Data Structures

Heap vs. Other Data Structures

Heaps differ from other data structures in terms of their properties and use cases. Here's a comparison with some common data structures:

1. Arrays: Arrays provide constant-time access to elements, but inserting or removing elements can be expensive, as it may require shifting all subsequent elements.
2. Linked Lists: Linked lists allow efficient insertion and removal of elements, but searching for an element can be slow, as it requires traversing the entire list.
3. Binary Search Trees (BSTs): BSTs provide efficient search, insertion, and deletion operations, but they don't guarantee logarithmic time complexity for all operations, and they can become unbalanced in the worst case.
4. Heaps: Heaps are designed to efficiently retrieve and remove the minimum (or maximum) element, making them well-suited for implementing priority queues. However, they don't support efficient search or access to arbitrary elements.

```python
# Example: Comparison of time complexities
import time
import random

def heapsort(nums):
    heap = nums.copy()
    heapq.heapify(heap)
    sorted_nums = []

    while heap:
        largest = heapq.heappop(heap)
        sorted_nums.append(largest)

    return sorted_nums[::-1]

def bubblesort(nums):
    sorted_nums = nums.copy()
    n = len(sorted_nums)

    for i in range(n):
        for j in range(0, n - i - 1):
            if sorted_nums[j] > sorted_nums[j + 1]:
                sorted_nums[j], sorted_nums[j + 1] = sorted_nums[j + 1], sorted_nums[j]

    return sorted_nums

# Generate a list of random numbers
nums = [random.randint(1, 1000000) for _ in range(100000)]

start_time = time.time()
heapsort(nums)
end_time = time.time()
print(f"Heapsort took {end_time - start_time:.6f} seconds")

start_time = time.time()
bubblesort(nums)
end_time = time.time()
print(f"Bubblesort took {end_time - start_time:.6f} seconds")
```

This example compares the time taken by heapsort (which uses a heap) and bubblesort (a simple sorting algorithm) for sorting a large list of random numbers, highlighting the efficiency advantage of heaps for certain operations.

Slide 11: Heap in Python's Standard Library

Heap in Python's Standard Library

Python's standard library provides the `heapq` module, which implements a min heap using a list as the underlying data structure. The `heapq` module offers several functions for working with heaps, including:

* `heapq.heappush(heap, item)`: Pushes an item onto the heap.
* `heapq.heappop(heap)`: Pops and returns the smallest item from the heap.
* `heapq.heapify(x)`: Transforms the list `x` into a heap in-place.
* `heapq.heapreplace(heap, item)`: Pops and returns the smallest item from the heap, and pushes the new `item` onto the heap.
* `heapq.nsmallest(n, iterable, key=None)`: Returns a list with the `n` smallest elements from the iterable.
* `heapq.nlargest(n, iterable, key=None)`: Returns a list with the `n` largest elements from the iterable.

```python
import heapq

# Creating a min heap
minHeap = []
heapq.heappush(minHeap, 5)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 8)
heapq.heappush(minHeap, 1)

print(minHeap)  # Output: [1, 2, 8, 5]

# Popping elements from the heap
smallest = heapq.heappop(minHeap)
print(smallest)  # Output: 1
print(minHeap)   # Output: [2, 5, 8]

# Finding the largest 3 elements
numbers = [5, 2, 8, 1, 9, 3]
largest_three = heapq.nlargest(3, numbers)
print(largest_three)  # Output: [9, 8, 5]
```

The `heapq` module provides a convenient and efficient way to work with heaps in Python, making it easier to implement algorithms and data structures that require priority queues or sorted data structures.

Slide 12: Heap vs. Priority Queue

Heap vs. Priority Queue

Heaps and priority queues are closely related concepts, but they are not the same. A priority queue is an abstract data type that provides the following operations:

* `enqueue(item, priority)`: Adds an item to the queue with a given priority.
* `dequeue()`: Removes and returns the item with the highest priority from the queue.
* `peek()`: Returns the item with the highest priority without removing it from the queue.

A heap is a specific data structure that can be used to implement a priority queue efficiently. In Python, the `heapq` module provides an implementation of a priority queue using a min heap.

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def enqueue(self, item, priority):
        heapq.heappush(self.heap, (priority, item))

    def dequeue(self):
        return heapq.heappop(self.heap)[1]

    def peek(self):
        return self.heap[0][1]

# Example usage
pq = PriorityQueue()
pq.enqueue("Task 1", 2)
pq.enqueue("Task 2", 1)
pq.enqueue("Task 3", 3)

print(pq.dequeue())  # Output: Task 2
print(pq.peek())     # Output: Task 1
```

In this example, the `PriorityQueue` class uses a min heap implemented by the `heapq` module to store the items and their priorities. The `enqueue` method adds an item to the heap with a given priority, the `dequeue` method removes and returns the item with the highest priority, and the `peek` method returns the item with the highest priority without removing it from the heap.

Slide 13: Heap Visualization

Heap Visualization

Heaps can be visualized as binary trees, with the root node representing the minimum (or maximum) value, and the children nodes satisfying the heap property. Here's an example of a min heap visualized as a binary tree:

```
                 1
              /     \
             2       8
           /   \    /
          5    3   9
```

In this visualization, the root node (1) represents the minimum value in the heap. The left child (2) is smaller than the right child (8), and the heap property is maintained throughout the tree.

While the code examples in this presentation use Python's `heapq` module, which represents heaps as lists, visualizing heaps as binary trees can help in understanding their structure and properties.

Slide 14: Additional Resources

Additional Resources

For further learning and exploration of heaps and related topics, here are some additional resources:

1. "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein (CLRS) - Chapter 6: Heaps
   * ArXiv link: [https://arxiv.org/abs/0909.2368](https://arxiv.org/abs/0909.2368)
2. "The Algorithm Design Manual" by Steven S. Skiena - Chapter 5: Heaps and Priority Queues
   * ArXiv link: [https://arxiv.org/abs/1905.03858](https://arxiv.org/abs/1905.03858)
3. "Mastering Algorithms with C" by Kyle Loudon - Chapter 5: Heaps
   * Reference: Loudon, K. (1999). Mastering Algorithms with C. O'Reilly Media.
4. "Data Structures and Algorithms in Python" by Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
   * ArXiv link: [https://arxiv.org/abs/1805.10555](https://arxiv.org/abs/1805.10555)

These resources provide in-depth explanations, examples, and exercises related to heaps and their applications, helping you further solidify your understanding of this important data structure.

