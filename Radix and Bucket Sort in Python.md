## Radix and Bucket Sort Slideshow in Python
Slide 1: Introduction to Sorting Algorithms

Sorting algorithms are fundamental in computer science, organizing data for efficient retrieval and processing. This presentation focuses on two efficient sorting algorithms: Radix Sort and Bucket Sort. Both are non-comparative sorting algorithms that can achieve linear time complexity under certain conditions.

```python
def demonstrate_sorting_importance():
    unsorted_list = [64, 34, 25, 12, 22, 11, 90]
    sorted_list = sorted(unsorted_list)
    print(f"Unsorted: {unsorted_list}")
    print(f"Sorted: {sorted_list}")
    
    # Binary search on sorted list
    import bisect
    index = bisect.bisect_left(sorted_list, 25)
    print(f"25 found at index: {index}")

demonstrate_sorting_importance()
```

Slide 2: Radix Sort Overview

Radix Sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping the keys by individual digits sharing the same significant position and value. It processes the digits from least significant to most significant, making it efficient for fixed-length integer keys.

```python
def radix_sort(arr):
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= 10
    return arr

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]
```

Slide 3: Radix Sort Step-by-Step

Let's walk through the Radix Sort algorithm step-by-step using a small example. We'll sort the list \[170, 45, 75, 90, 802, 24, 2, 66\] using Radix Sort.

```python
def visualize_radix_sort(arr):
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        print(f"\nSorting by {exp}'s place:")
        counting_sort(arr, exp)
        print(arr)
        exp *= 10
    return arr

numbers = [170, 45, 75, 90, 802, 24, 2, 66]
print("Original array:", numbers)
visualize_radix_sort(numbers)
```

Slide 4: Radix Sort Time Complexity

Radix Sort has a time complexity of O(d \* (n + k)), where d is the number of digits in the maximum number, n is the number of elements, and k is the range of values (usually 10 for decimal numbers). This makes it efficient for sorting large numbers of integers with a fixed number of digits.

```python
import time
import random

def time_radix_sort(n):
    arr = [random.randint(0, 10**6) for _ in range(n)]
    start_time = time.time()
    radix_sort(arr)
    end_time = time.time()
    return end_time - start_time

sizes = [10**3, 10**4, 10**5, 10**6]
for size in sizes:
    elapsed_time = time_radix_sort(size)
    print(f"Time taken for {size} elements: {elapsed_time:.6f} seconds")
```

Slide 5: Radix Sort Applications

Radix Sort is particularly useful in scenarios where we need to sort large numbers of integers with a fixed number of digits. Some real-world applications include:

1. Sorting dates and times
2. Organizing large databases of numeric IDs
3. Sorting IP addresses

```python
def sort_ip_addresses(ip_list):
    def ip_to_int(ip):
        return sum(int(octet) << (8 * i) for i, octet in enumerate(reversed(ip.split('.'))))
    
    def int_to_ip(num):
        return '.'.join(str(num >> (8 * i) & 255) for i in range(3, -1, -1))
    
    int_ips = [ip_to_int(ip) for ip in ip_list]
    sorted_int_ips = radix_sort(int_ips)
    return [int_to_ip(num) for num in sorted_int_ips]

ip_addresses = ["192.168.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1"]
print("Sorted IP addresses:", sort_ip_addresses(ip_addresses))
```

Slide 6: Bucket Sort Overview

Bucket Sort is a distribution-based sorting algorithm that distributes elements into a number of buckets, then sorts these buckets individually. It works well when input is uniformly distributed over a range.

```python
def bucket_sort(arr):
    # Find minimum and maximum values
    min_val, max_val = min(arr), max(arr)
    
    # Create buckets
    bucket_range = (max_val - min_val) / len(arr)
    buckets = [[] for _ in range(len(arr) + 1)]
    
    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        buckets[index].append(num)
    
    # Sort individual buckets
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))
    
    return sorted_arr
```

Slide 7: Bucket Sort Step-by-Step

Let's visualize the Bucket Sort process using a small example. We'll sort the list \[0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51\] using Bucket Sort.

```python
def visualize_bucket_sort(arr):
    print("Original array:", arr)
    
    # Find minimum and maximum values
    min_val, max_val = min(arr), max(arr)
    
    # Create buckets
    bucket_range = (max_val - min_val) / len(arr)
    buckets = [[] for _ in range(len(arr) + 1)]
    
    # Distribute elements into buckets
    for num in arr:
        index = int((num - min_val) / bucket_range)
        buckets[index].append(num)
    
    print("\nBuckets after distribution:")
    for i, bucket in enumerate(buckets):
        if bucket:
            print(f"Bucket {i}: {bucket}")
    
    # Sort individual buckets
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(sorted(bucket))
    
    print("\nFinal sorted array:", sorted_arr)
    return sorted_arr

numbers = [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]
visualize_bucket_sort(numbers)
```

Slide 8: Bucket Sort Time Complexity

Bucket Sort has an average-case time complexity of O(n + k), where n is the number of elements and k is the number of buckets. In the worst case, when all elements are placed in a single bucket, it degrades to O(n^2) if using a comparison sort for that bucket.

```python
import time
import random

def time_bucket_sort(n):
    arr = [random.random() for _ in range(n)]
    start_time = time.time()
    bucket_sort(arr)
    end_time = time.time()
    return end_time - start_time

sizes = [10**3, 10**4, 10**5, 10**6]
for size in sizes:
    elapsed_time = time_bucket_sort(size)
    print(f"Time taken for {size} elements: {elapsed_time:.6f} seconds")
```

Slide 9: Bucket Sort Applications

Bucket Sort is particularly useful when input is uniformly distributed over a range. Some real-world applications include:

1. Sorting student grades within a fixed range (e.g., 0-100)
2. Organizing data in histograms or frequency distributions
3. External sorting of large datasets that don't fit in memory

```python
def sort_student_grades(grades):
    def bucket_sort_grades(arr):
        buckets = [[] for _ in range(101)]  # 0 to 100 inclusive
        for grade in arr:
            buckets[grade].append(grade)
        return [grade for bucket in buckets for grade in bucket]
    
    return bucket_sort_grades(grades)

student_grades = [85, 92, 78, 54, 100, 87, 93, 76, 89, 95]
print("Sorted student grades:", sort_student_grades(student_grades))
```

Slide 10: Comparison of Radix Sort and Bucket Sort

While both Radix Sort and Bucket Sort are distribution-based sorting algorithms, they have different strengths and use cases. Radix Sort is typically used for integers, while Bucket Sort works well with floating-point numbers uniformly distributed over a range.

```python
import random
import time

def compare_sorts(n):
    # Generate random integers for Radix Sort
    int_arr = [random.randint(0, 10**6) for _ in range(n)]
    
    # Generate random floats for Bucket Sort
    float_arr = [random.random() for _ in range(n)]
    
    # Time Radix Sort
    start_time = time.time()
    radix_sort(int_arr.())
    radix_time = time.time() - start_time
    
    # Time Bucket Sort
    start_time = time.time()
    bucket_sort(float_arr.())
    bucket_time = time.time() - start_time
    
    print(f"For {n} elements:")
    print(f"Radix Sort time: {radix_time:.6f} seconds")
    print(f"Bucket Sort time: {bucket_time:.6f} seconds")

compare_sorts(10**5)
```

Slide 11: When to Use Radix Sort

Radix Sort is particularly effective when:

1. Sorting integers or strings with fixed-length keys
2. The range of possible key values is known and reasonably small
3. Memory usage is not a significant constraint

```python
def sort_fixed_length_strings(strings):
    def string_to_int(s):
        return sum(ord(c) << (8 * i) for i, c in enumerate(s))
    
    def int_to_string(n, length):
        return ''.join(chr((n >> (8 * i)) & 255) for i in range(length))
    
    length = len(strings[0])
    int_strings = [string_to_int(s) for s in strings]
    sorted_ints = radix_sort(int_strings)
    return [int_to_string(n, length) for n in sorted_ints]

words = ["cat", "dog", "bat", "ant", "pig", "owl"]
print("Sorted words:", sort_fixed_length_strings(words))
```

Slide 12: When to Use Bucket Sort

Bucket Sort is most effective when:

1. Input is uniformly distributed over a known range
2. The number of buckets can be chosen to balance between memory usage and performance
3. Additional memory usage is acceptable for improved average-case performance

```python
def analyze_data_distribution(data):
    min_val, max_val = min(data), max(data)
    bucket_range = (max_val - min_val) / 10
    buckets = [0] * 10
    
    for value in data:
        index = min(9, int((value - min_val) / bucket_range))
        buckets[index] += 1
    
    for i, count in enumerate(buckets):
        print(f"Bucket {i}: {count} items")
    
    return buckets

# Generate non-uniform data
data = [random.gauss(50, 15) for _ in range(1000)]
print("Data distribution analysis:")
analyze_data_distribution(data)
```

Slide 13: Optimizing Radix and Bucket Sort

Both Radix Sort and Bucket Sort can be optimized for specific use cases:

1. For Radix Sort, using a larger base (e.g., 256 instead of 10) can reduce the number of passes
2. For Bucket Sort, adaptive bucket sizes based on data distribution can improve performance
3. Hybrid approaches, combining distribution-based sorting with comparison-based sorting for small subarrays, can be effective

```python
def optimized_radix_sort(arr, base=256):
    def counting_sort(arr, exp):
        n = len(arr)
        output = [0] * n
        count = [0] * base
        
        for i in range(n):
            index = arr[i] // exp
            count[index % base] += 1
        
        for i in range(1, base):
            count[i] += count[i - 1]
        
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % base] - 1] = arr[i]
            count[index % base] -= 1
            i -= 1
        
        for i in range(n):
            arr[i] = output[i]
    
    max_num = max(arr)
    exp = 1
    while max_num // exp > 0:
        counting_sort(arr, exp)
        exp *= base
    return arr

# Test the optimized Radix Sort
test_arr = [random.randint(0, 10**6) for _ in range(10**5)]
start_time = time.time()
optimized_radix_sort(test_arr)
print(f"Optimized Radix Sort time: {time.time() - start_time:.6f} seconds")
```

Slide 14: Conclusion and Best Practices

When choosing between Radix Sort and Bucket Sort, consider:

1. Data characteristics (integers vs. floating-point, range, distribution)
2. Memory constraints
3. Performance requirements

Both algorithms can offer significant performance improvements over comparison-based sorts in specific scenarios. Always profile your specific use case to determine the most effective approach.

```python
def choose_sort_algorithm(data):
    if all(isinstance(x, int) for x in data):
        return "Radix Sort"
    elif max(data) - min(data) <= len(data):
        return "Bucket Sort"
    else:
        return "Consider alternative sorting algorithms"

# Example usage
integer_data = [random.randint(0, 1000) for _ in range(1000)]
float_data = [random.random() for _ in range(1000)]
large_range_data = [random.uniform(0, 10**6) for _ in range(1000)]

print("For integer data:", choose_sort_algorithm(integer_data))
print("For uniformly distributed float data:", choose_sort_algorithm(float_data))
print("For large range data:", choose_sort_algorithm(large_range_data))
```

Slide 15: Additional Resources

For further exploration of Radix Sort, Bucket Sort, and other sorting algorithms, consider the following resources:

1. "Algorithms" by Robert Sedgewick and Kevin Wayne (Princeton University)
2. "Introduction to Algorithms" by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (MIT Press)
3. ArXiv paper: "A Survey of Sorting Algorithms" by Debasis Ganguly (arXiv:1410.5256)

These resources provide in-depth analysis and comparisons of various sorting algorithms, including Radix Sort and Bucket Sort, as well as their theoretical foundations and practical implementations.

