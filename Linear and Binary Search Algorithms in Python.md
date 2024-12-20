## Linear and Binary Search Algorithms in Python
Slide 1: Introduction to Linear Search

Linear search is a simple search algorithm used to find an element in a list or array. It works by iterating through the elements one by one, comparing each element with the target value until a match is found or the end of the list is reached.

Code:

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Example usage
my_list = [5, 2, 8, 1, 9]
target = 8
index = linear_search(my_list, target)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 2: Time Complexity of Linear Search

The time complexity of linear search is O(n), where n is the size of the list or array. In the worst case scenario, where the target element is not present or is the last element in the list, the algorithm needs to iterate through all elements, resulting in a linear time complexity.

Code:

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Example usage with a large list
import random

large_list = [random.randint(1, 1000) for _ in range(10000)]
target = 999
index = linear_search(large_list, target)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 3: Introduction to Binary Search

Binary search is an efficient search algorithm used to find an element in a sorted list or array. It works by repeatedly dividing the search interval in half until the target value is found or the search interval becomes empty.

Code:

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage
sorted_list = [1, 3, 5, 7, 9]
target = 5
index = binary_search(sorted_list, target)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 4: Time Complexity of Binary Search

The time complexity of binary search is O(log n), where n is the size of the list or array. This is because the algorithm halves the search interval in each iteration, resulting in a logarithmic time complexity.

Code:

```python
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage with a large sorted list
import random

large_sorted_list = sorted([random.randint(1, 1000000) for _ in range(100000)])
target = 999999
index = binary_search(large_sorted_list, target)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 5: Comparison of Linear Search and Binary Search

While linear search can be used for both sorted and unsorted lists, binary search requires the list to be sorted. Binary search is generally more efficient than linear search, especially for large lists or arrays, due to its logarithmic time complexity.

Code:

```python
import time

# Linear search
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Binary search
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# Example usage
unsorted_list = [random.randint(1, 1000000) for _ in range(100000)]
sorted_list = sorted(unsorted_list)
target = 999999

start_time = time.time()
linear_index = linear_search(unsorted_list, target)
linear_time = time.time() - start_time

start_time = time.time()
binary_index = binary_search(sorted_list, target)
binary_time = time.time() - start_time

print(f"Linear search time: {linear_time:.6f} seconds")
print(f"Binary search time: {binary_time:.6f} seconds")
```

Slide 6: When to Use Linear Search or Binary Search

Linear search is suitable when the list is small or unsorted, or when the time complexity is not a significant concern. Binary search, on the other hand, is more efficient for large sorted lists or arrays, where the time complexity becomes crucial.

Code:

```python
# Linear search example
unsorted_list = [5, 2, 8, 1, 9]
target = 8
index = linear_search(unsorted_list, target)
if index != -1:
    print(f"Target value {target} found at index {index} using linear search")
else:
    print(f"Target value {target} not found in the list using linear search")

# Binary search example
sorted_list = [1, 3, 5, 7, 9]
target = 5
index = binary_search(sorted_list, target)
if index != -1:
    print(f"Target value {target} found at index {index} using binary search")
else:
    print(f"Target value {target} not found in the list using binary search")
```

Slide 7: Recursive Implementation of Linear Search

Linear search can also be implemented using recursion, where the function calls itself with a smaller portion of the list or array until the target value is found or the base case is reached.

Code:

```python
def recursive_linear_search(arr, target, index=0):
    if index >= len(arr):
        return -1
    if arr[index] == target:
        return index
    return recursive_linear_search(arr, target, index + 1)

# Example usage
my_list = [5, 2, 8, 1, 9]
target = 8
index = recursive_linear_search(my_list, target)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 8: Recursive Implementation of Binary Search

Binary search can also be implemented using recursion, where the function divides the search interval in half and calls itself with the appropriate sub-interval until the target value is found or the base case is reached.

Code:

```python
def recursive_binary_search(arr, target, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return recursive_binary_search(arr, target, mid + 1, high)
    else:
        return recursive_binary_search(arr, target, low, mid - 1)

# Example usage
sorted_list = [1, 3, 5, 7, 9]
target = 5
index = recursive_binary_search(sorted_list, target)
if index != -1:
    print(f"Target value {target} found at index {index} using recursive binary search")
else:
    print(f"Target value {target} not found in the list using recursive binary search")
```

Slide 9: Searching in a Rotated Sorted Array

In some cases, you may need to search for an element in a rotated sorted array, where the array is divided into two sorted subarrays. This problem can be solved using a modified version of binary search.

Code:

```python
def search_rotated_sorted_array(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] >= arr[left]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

# Example usage
rotated_sorted_list = [4, 5, 6, 7, 0, 1, 2]
target = 0
index = search_rotated_sorted_array(rotated_sorted_list, target)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 10: Searching in a Nearly Sorted Array

In some situations, you may need to search for an element in a nearly sorted array, where each element is at most k positions away from its sorted position. This problem can be solved using a modified version of binary search, which considers the potential displacement of elements.

Code:

```python
def search_nearly_sorted_array(arr, target, k):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = max(left, mid - k)
        else:
            right = min(right, mid + k)
    return -1

# Example usage
nearly_sorted_list = [3, 2, 10, 4, 40]
target = 4
k = 2
index = search_nearly_sorted_array(nearly_sorted_list, target, k)
if index != -1:
    print(f"Target value {target} found at index {index}")
else:
    print(f"Target value {target} not found in the list")
```

Slide 11: Searching in a 2D Sorted Matrix

Binary search can also be extended to search for an element in a 2D sorted matrix, where the rows and columns are sorted in ascending order.

Code:

```python
def search_2d_sorted_matrix(matrix, target):
    rows = len(matrix)
    cols = len(matrix[0])
    left = 0
    right = rows * cols - 1
    while left <= right:
        mid = (left + right) // 2
        row = mid // cols
        col = mid % cols
        if matrix[row][col] == target:
            return (row, col)
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1
    return (-1, -1)

# Example usage
sorted_matrix = [
    [1, 4, 7, 11, 15],
    [2, 5, 8, 12, 19],
    [3, 6, 9, 16, 22],
    [10, 13, 14, 17, 24],
    [18, 21, 23, 26, 30]
]
target = 5
row, col = search_2d_sorted_matrix(sorted_matrix, target)
if row != -1 and col != -1:
    print(f"Target value {target} found at ({row}, {col})")
else:
    print(f"Target value {target} not found in the matrix")
```

Slide 12: Searching in a Sorted List with Duplicates

When searching for an element in a sorted list or array that contains duplicates, binary search can be modified to find the first or last occurrence of the target value.

Code:

```python
def find_first_occurrence(arr, target):
    left = 0
    right = len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

def find_last_occurrence(arr, target):
    left = 0
    right = len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            left = mid + 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Example usage
sorted_list_with_duplicates = [1, 2, 3, 3, 3, 4, 5, 5, 5, 5]
target = 3
first_occurrence = find_first_occurrence(sorted_list_with_duplicates, target)
last_occurrence = find_last_occurrence(sorted_list_with_duplicates, target)
print(f"First occurrence of {target}: {first_occurrence}")
print(f"Last occurrence of {target}: {last_occurrence}")
```

Slide 13: Additional Resources

For further reading and exploration, you can refer to the following resources:

* "Introduction to Algorithms" by Cormen, Leiserson, Rivest, and Stein (CLRS)
* "Algorithms" by Robert Sedgewick and Kevin Wayne
* arxiv.org/abs/1208.3180 - "A Unified View on Searching in Sorted and Partially Sorted Arrays"

Slide 14: Additional Resources (continued)

Here are some additional resources for further study:

* arxiv.org/abs/1701.03705 - "Nearly Sorted Linear Search"
* arxiv.org/abs/1512.00517 - "An Optimal Algorithm for the k-Sorted Problem"

