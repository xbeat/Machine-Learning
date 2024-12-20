## Counting Sort Algorithm in Python
Slide 1: Introduction to Counting Sort

Counting Sort is a non-comparative sorting algorithm that operates in O(n+k) time complexity, where n is the number of elements and k is the range of input. This algorithm is particularly efficient when dealing with integers or strings with a limited range of possible values.

```python
def counting_sort(arr):
    # Find the range of input
    max_val = max(arr)
    min_val = min(arr)
    range_of_values = max_val - min_val + 1

    # Initialize count array and output array
    count = [0] * range_of_values
    output = [0] * len(arr)

    # Count occurrences of each element
    for num in arr:
        count[num - min_val] += 1

    # Calculate cumulative count
    for i in range(1, len(count)):
        count[i] += count[i-1]

    # Build the output array
    for num in reversed(arr):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1

    return output

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
sorted_arr = counting_sort(arr)
print(f"Original array: {arr}")
print(f"Sorted array: {sorted_arr}")
```

Slide 2: How Counting Sort Works

Counting Sort works by counting the occurrences of each element in the input array and using this information to determine the correct position of each element in the sorted output. This approach allows us to sort the array without comparing elements directly.

```python
import matplotlib.pyplot as plt

def visualize_counting_sort(arr):
    max_val = max(arr)
    min_val = min(arr)
    range_of_values = max_val - min_val + 1

    count = [0] * range_of_values
    for num in arr:
        count[num - min_val] += 1

    plt.figure(figsize=(12, 6))
    plt.bar(range(min_val, max_val + 1), count)
    plt.title("Counting Sort: Element Frequency")
    plt.xlabel("Element Value")
    plt.ylabel("Frequency")
    plt.show()

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
visualize_counting_sort(arr)
```

Slide 3: Step 1: Finding the Range

The first step in Counting Sort is to determine the range of input values. This information is crucial for creating the count array, which will store the frequency of each element.

```python
def find_range(arr):
    max_val = max(arr)
    min_val = min(arr)
    range_of_values = max_val - min_val + 1
    return min_val, max_val, range_of_values

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
min_val, max_val, range_of_values = find_range(arr)
print(f"Minimum value: {min_val}")
print(f"Maximum value: {max_val}")
print(f"Range of values: {range_of_values}")
```

Slide 4: Step 2: Counting Occurrences

In this step, we count the occurrences of each element in the input array. We use the count array to store these frequencies, with each index representing an element value.

```python
def count_occurrences(arr, min_val, range_of_values):
    count = [0] * range_of_values
    for num in arr:
        count[num - min_val] += 1
    return count

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
min_val, max_val, range_of_values = find_range(arr)
count = count_occurrences(arr, min_val, range_of_values)
print(f"Count array: {count}")
```

Slide 5: Step 3: Calculating Cumulative Count

We transform the count array into a cumulative count array. This step helps us determine the correct position of each element in the sorted output.

```python
def calculate_cumulative_count(count):
    for i in range(1, len(count)):
        count[i] += count[i-1]
    return count

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
min_val, max_val, range_of_values = find_range(arr)
count = count_occurrences(arr, min_val, range_of_values)
cumulative_count = calculate_cumulative_count(count.())
print(f"Original count array: {count}")
print(f"Cumulative count array: {cumulative_count}")
```

Slide 6: Step 4: Building the Sorted Output

Using the cumulative count array, we can now place each element in its correct position in the sorted output array. We iterate through the input array in reverse order to maintain stability.

```python
def build_sorted_output(arr, count, min_val):
    output = [0] * len(arr)
    for num in reversed(arr):
        output[count[num - min_val] - 1] = num
        count[num - min_val] -= 1
    return output

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
min_val, max_val, range_of_values = find_range(arr)
count = count_occurrences(arr, min_val, range_of_values)
cumulative_count = calculate_cumulative_count(count.())
sorted_arr = build_sorted_output(arr, cumulative_count, min_val)
print(f"Original array: {arr}")
print(f"Sorted array: {sorted_arr}")
```

Slide 7: Time and Space Complexity

Counting Sort has a time complexity of O(n+k), where n is the number of elements and k is the range of input values. The space complexity is O(n+k) as well, due to the additional arrays used in the sorting process.

```python
import time
import random

def measure_time_complexity(n, k):
    arr = [random.randint(0, k-1) for _ in range(n)]
    
    start_time = time.time()
    sorted_arr = counting_sort(arr)
    end_time = time.time()
    
    return end_time - start_time

# Example: Measure time for different input sizes
sizes = [1000, 10000, 100000, 1000000]
k = 1000  # Range of values

for n in sizes:
    execution_time = measure_time_complexity(n, k)
    print(f"n = {n}: {execution_time:.6f} seconds")
```

Slide 8: Advantages of Counting Sort

Counting Sort offers several advantages, including its linear time complexity for a fixed range of input values and its stability, which preserves the relative order of equal elements. It is particularly efficient for sorting integers or strings with a limited range of possible values.

```python
def demonstrate_stability(arr):
    # Create a list of tuples (value, index)
    indexed_arr = list(enumerate(arr))
    
    # Sort the list using a stable sort (e.g., Timsort)
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1])
    
    # Extract the sorted values
    sorted_arr = [x[1] for x in sorted_indexed_arr]
    
    return sorted_arr, sorted_indexed_arr

# Example usage
arr = [4, 2, 2, 8, 3, 3, 1]
sorted_arr, sorted_indexed_arr = demonstrate_stability(arr)
print(f"Original array: {arr}")
print(f"Sorted array: {sorted_arr}")
print(f"Sorted array with original indices: {sorted_indexed_arr}")
```

Slide 9: Limitations of Counting Sort

Despite its efficiency, Counting Sort has some limitations. It is not suitable for sorting large ranges of values or floating-point numbers, as it requires additional memory proportional to the range of input values.

```python
import sys

def calculate_memory_usage(arr, k):
    input_size = sys.getsizeof(arr)
    count_size = sys.getsizeof([0] * k)
    output_size = sys.getsizeof([0] * len(arr))
    
    total_size = input_size + count_size + output_size
    return total_size

# Example: Compare memory usage for different ranges
arr = [random.randint(0, 999) for _ in range(1000)]
ranges = [1000, 10000, 100000, 1000000]

for k in ranges:
    memory_usage = calculate_memory_usage(arr, k)
    print(f"Range {k}: {memory_usage} bytes")
```

Slide 10: Counting Sort for Strings

Counting Sort can be adapted to sort strings by considering each character's ASCII value. This approach is particularly useful for sorting strings of equal length.

```python
def counting_sort_strings(arr):
    # Find the maximum length of strings
    max_len = max(len(s) for s in arr)
    
    # Pad shorter strings with spaces
    padded_arr = [s.ljust(max_len) for s in arr]
    
    # Sort strings character by character, starting from the rightmost
    for i in range(max_len - 1, -1, -1):
        # Count occurrences of each character
        count = [0] * 256
        for s in padded_arr:
            count[ord(s[i])] += 1
        
        # Calculate cumulative count
        for j in range(1, 256):
            count[j] += count[j-1]
        
        # Build the output array
        output = [''] * len(arr)
        for s in reversed(padded_arr):
            output[count[ord(s[i])] - 1] = s
            count[ord(s[i])] -= 1
        
        padded_arr = output
    
    # Remove padding and return the sorted array
    return [s.strip() for s in padded_arr]

# Example usage
arr = ["apple", "banana", "cherry", "date", "elderberry"]
sorted_arr = counting_sort_strings(arr)
print(f"Original array: {arr}")
print(f"Sorted array: {sorted_arr}")
```

Slide 11: Real-Life Example: Sorting Student Grades

Counting Sort can be efficiently used to sort student grades, which typically have a limited range (e.g., 0-100). This example demonstrates how to sort grades and calculate class statistics.

```python
import random

def sort_student_grades(grades):
    max_grade = 100
    count = [0] * (max_grade + 1)
    
    for grade in grades:
        count[grade] += 1
    
    sorted_grades = []
    for grade, freq in enumerate(count):
        sorted_grades.extend([grade] * freq)
    
    return sorted_grades

def calculate_statistics(grades):
    total = sum(grades)
    avg = total / len(grades)
    median = grades[len(grades) // 2]
    return avg, median

# Generate random grades for 30 students
grades = [random.randint(0, 100) for _ in range(30)]

sorted_grades = sort_student_grades(grades)
avg, median = calculate_statistics(sorted_grades)

print(f"Original grades: {grades}")
print(f"Sorted grades: {sorted_grades}")
print(f"Class average: {avg:.2f}")
print(f"Median grade: {median}")
```

Slide 12: Real-Life Example: Organizing Books by ISBN

Libraries and bookstores can use Counting Sort to efficiently organize books by their ISBN (International Standard Book Number). This example demonstrates sorting books by the last digit of their ISBN.

```python
import random

class Book:
    def __init__(self, title, isbn):
        self.title = title
        self.isbn = isbn
    
    def __repr__(self):
        return f"{self.title} (ISBN: {self.isbn})"

def sort_books_by_isbn(books):
    # Use the last digit of ISBN for sorting
    count = [0] * 10
    
    for book in books:
        count[int(book.isbn[-1])] += 1
    
    for i in range(1, 10):
        count[i] += count[i-1]
    
    output = [None] * len(books)
    for book in reversed(books):
        index = int(book.isbn[-1])
        output[count[index] - 1] = book
        count[index] -= 1
    
    return output

# Generate random books
titles = ["The Great Gatsby", "To Kill a Mockingbird", "1984", "Pride and Prejudice", "The Catcher in the Rye"]
books = [Book(title, f"978-0-{random.randint(100000, 999999)}-{random.randint(0, 9)}") for title in titles]

sorted_books = sort_books_by_isbn(books)

print("Original book list:")
for book in books:
    print(book)

print("\nSorted book list:")
for book in sorted_books:
    print(book)
```

Slide 13: Optimizing Counting Sort for Sparse Data

When dealing with sparse data (where the range of values is much larger than the number of elements), we can optimize Counting Sort to reduce memory usage and improve performance.

```python
def optimized_counting_sort(arr):
    # Create a dictionary to store non-zero counts
    count = {}
    
    # Count occurrences of each element
    for num in arr:
        count[num] = count.get(num, 0) + 1
    
    # Sort the unique elements
    sorted_keys = sorted(count.keys())
    
    # Build the sorted output
    output = []
    for key in sorted_keys:
        output.extend([key] * count[key])
    
    return output

# Example usage with sparse data
sparse_arr = [1000, 10, 10000, 10, 1, 100000, 1000]
sorted_sparse_arr = optimized_counting_sort(sparse_arr)

print(f"Original sparse array: {sparse_arr}")
print(f"Sorted sparse array: {sorted_sparse_arr}")
```

Slide 14: Comparison with Other Sorting Algorithms

While Counting Sort is efficient for certain types of data, it's important to understand how it compares to other sorting algorithms in different scenarios. This visualization compares Counting Sort with Quick Sort and Merge Sort for various input sizes and ranges.

```python
import time
import random
import matplotlib.pyplot as plt

def measure_sorting_time(sort_func, arr):
    start_time = time.time()
    sort_func(arr.())
    return time.time() - start_time

def compare_sorting_algorithms(sizes, k):
    counting_times = []
    quick_times = []
    merge_times = []

    for n in sizes:
        arr = [random.randint(0, k-1) for _ in range(n)]
        
        counting_times.append(measure_sorting_time(counting_sort, arr))
        quick_times.append(measure_sorting_time(quick_sort, arr))
        merge_times.append(measure_sorting_time(merge_sort, arr))

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, counting_times, label='Counting Sort')
    plt.plot(sizes, quick_times, label='Quick Sort')
    plt.plot(sizes, merge_times, label='Merge Sort')
    plt.xlabel('Input Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Sorting Algorithm Performance Comparison')
    plt.legend()
    plt.show()

# Example usage
sizes = [1000, 5000, 10000, 50000, 100000]
k = 1000  # Range of values
compare_sorting_algorithms(sizes, k)
```

Slide 15: Additional Resources

For those interested in delving deeper into Counting Sort and other sorting algorithms, here are some valuable resources:

1. "Algorithms" by Robert Sedgewick and Kevin Wayne - A comprehensive book covering various algorithms, including Counting Sort.
2. "Introduction to Algorithms" by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein - A classic textbook that provides in-depth analysis of sorting algorithms.
3. ArXiv paper: "A Survey of Sorting Algorithms" by Fuad M. M. Zayer (arXiv:2008.05895) - This paper provides a comprehensive overview of various sorting algorithms, including Counting Sort.
4. Online platforms like Coursera, edX, and MIT OpenCourseWare offer courses on algorithms and data structures that cover sorting algorithms in detail.
5. Visualization tools such as VisuAlgo ([https://visualgo.net/en/sorting](https://visualgo.net/en/sorting)) can help in understanding the step-by-step process of different sorting algorithms, including Counting Sort.

Remember to verify the accuracy and relevance of these resources, as they may have been updated since my last knowledge update.

