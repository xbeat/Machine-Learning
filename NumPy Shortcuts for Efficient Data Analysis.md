## NumPy Shortcuts for Efficient Data Analysis
Slide 1: NumPy Essentials: Your Shortcut to Efficient Data Analysis

NumPy is a fundamental library for scientific computing in Python. This guide will walk you through essential NumPy commands and operations, helping you streamline your data analysis workflow. Let's dive in with some practical examples and code snippets.

```python
import numpy as np

# Create a simple array
arr = np.array([1, 2, 3, 4, 5])
print(arr)
# Output: [1 2 3 4 5]
```

Slide 2: Array Creation: Building Blocks of NumPy

NumPy offers various methods to create arrays. We'll explore some common techniques for array creation, including using lists, ranges, and special functions.

```python
# Create an array from a list
list_array = np.array([1, 2, 3, 4, 5])

# Create an array with a range of values
range_array = np.arange(0, 10, 2)

# Create an array of ones
ones_array = np.ones((3, 3))

print("List array:", list_array)
print("Range array:", range_array)
print("Ones array:\n", ones_array)
```

Slide 3: Array Attributes: Understanding Your Data

NumPy arrays have several attributes that provide useful information about their structure and contents. Let's explore some key attributes.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Shape:", arr.shape)
print("Dimensions:", arr.ndim)
print("Data type:", arr.dtype)
print("Size:", arr.size)

# Output:
# Shape: (2, 3)
# Dimensions: 2
# Data type: int64
# Size: 6
```

Slide 4: Indexing and Slicing: Accessing Array Elements

Efficient data manipulation often requires accessing specific elements or subsets of an array. NumPy provides powerful indexing and slicing capabilities.

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Indexing
print("Element at (1, 2):", arr[1, 2])

# Slicing
print("First two rows, last two columns:\n", arr[:2, 2:])

# Boolean indexing
mask = arr > 5
print("Elements greater than 5:\n", arr[mask])
```

Slide 5: Array Manipulation: Reshaping and Stacking

NumPy offers various functions to manipulate array shapes and combine multiple arrays. These operations are crucial for data preprocessing and feature engineering.

```python
# Reshape an array
arr = np.arange(12)
reshaped = arr.reshape((3, 4))
print("Reshaped array:\n", reshaped)

# Stack arrays vertically
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vertical_stack = np.vstack((a, b))
print("Vertical stack:\n", vertical_stack)

# Stack arrays horizontally
horizontal_stack = np.hstack((a, b))
print("Horizontal stack:", horizontal_stack)
```

Slide 6: Arithmetic Operations: Vector and Matrix Math

NumPy simplifies vector and matrix operations, allowing element-wise calculations and broadcasting for arrays of different shapes.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
print("Addition:", a + b)
print("Multiplication:", a * b)

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix + vector:\n", matrix + a)

# Matrix multiplication
result = np.dot(matrix, a)
print("Matrix-vector product:", result)
```

Slide 7: Statistical Operations: Descriptive Statistics

NumPy provides a wide range of statistical functions to analyze your data quickly and efficiently.

```python
data = np.array([14, 23, 32, 41, 50, 59])

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard deviation:", np.std(data))
print("Variance:", np.var(data))
print("Min and Max:", np.min(data), np.max(data))

# Output:
# Mean: 36.5
# Median: 36.5
# Standard deviation: 16.19
# Variance: 262.25
# Min and Max: 14 59
```

Slide 8: Linear Algebra: Matrix Operations

NumPy's linear algebra module provides powerful tools for matrix operations, eigenvalue calculations, and solving linear systems.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print("Matrix multiplication:\n", C)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Solve linear system Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:", x)
```

Slide 9: Broadcasting: Efficient Array Operations

Broadcasting is a powerful NumPy feature that allows operations between arrays of different shapes. It can significantly simplify your code and improve performance.

```python
# Broadcasting example
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])

result = a + b
print("Broadcasting result:\n", result)

# Without broadcasting, we'd need to do:
# result = np.array([a + b[i] for i in range(3)])

# Output:
# Broadcasting result:
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]
```

Slide 10: Random Number Generation: Simulations and Sampling

NumPy's random module provides various functions for generating random numbers, which are crucial for simulations, statistical analysis, and machine learning.

```python
# Set a seed for reproducibility
np.random.seed(42)

# Generate random integers
random_ints = np.random.randint(1, 11, size=5)
print("Random integers:", random_ints)

# Generate random floats
random_floats = np.random.random(5)
print("Random floats:", random_floats)

# Generate numbers from a normal distribution
normal_dist = np.random.normal(loc=0, scale=1, size=5)
print("Normal distribution:", normal_dist)
```

Slide 11: Real-life Example: Image Processing

NumPy is extensively used in image processing. Let's create a simple example of image manipulation using NumPy.

```python
# Create a simple 5x5 grayscale image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Rotate the image by 90 degrees
rotated = np.rot90(image)

print("Original image:\n", image)
print("\nRotated image:\n", rotated)

# Apply a simple filter (e.g., edge detection)
filter = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

filtered = np.zeros_like(image)
for i in range(1, 4):
    for j in range(1, 4):
        filtered[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * filter)

print("\nFiltered image (edge detection):\n", filtered)
```

Slide 12: Real-life Example: Time Series Analysis

NumPy is also valuable for time series analysis. Let's create a simple example of analyzing temperature data.

```python
# Generate synthetic temperature data
days = np.arange(1, 31)
temperatures = 20 + 5 * np.sin(days * 2 * np.pi / 30) + np.random.normal(0, 1, 30)

# Calculate moving average
window_size = 3
moving_avg = np.convolve(temperatures, np.ones(window_size), 'valid') / window_size

print("First 10 days temperatures:", temperatures[:10])
print("Moving average (first 8 days):", moving_avg[:8])

# Find days with temperature above average
above_avg = days[temperatures > np.mean(temperatures)]
print("Days with above-average temperature:", above_avg)

# Calculate temperature range
temp_range = np.ptp(temperatures)
print(f"Temperature range: {temp_range:.2f}")
```

Slide 13: NumPy Performance: Vectorization vs. Loops

One of NumPy's key advantages is its ability to perform vectorized operations, which are much faster than traditional Python loops. Let's compare the performance.

```python
import time

# Create a large array
arr = np.random.random(1000000)

# Using a loop
start_time = time.time()
result_loop = [x**2 for x in arr]
loop_time = time.time() - start_time

# Using NumPy vectorization
start_time = time.time()
result_numpy = arr**2
numpy_time = time.time() - start_time

print(f"Loop time: {loop_time:.6f} seconds")
print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"NumPy is {loop_time/numpy_time:.2f}x faster")
```

Slide 14: Additional Resources

For those looking to deepen their understanding of NumPy and its applications in data science, here are some valuable resources:

1.  NumPy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
2.  "From Python to Numpy" by Nicolas P. Rougier: [https://www.labri.fr/perso/nrougier/from-python-to-numpy/](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
3.  "NumPy: A guide to NumPy" by Travis E. Oliphant: [https://web.mit.edu/dvp/Public/numpybook.pdf](https://web.mit.edu/dvp/Public/numpybook.pdf)
4.  ArXiv paper: "Array programming with NumPy" (2020): [https://arxiv.org/abs/2006.10256](https://arxiv.org/abs/2006.10256)

These resources provide in-depth explanations, advanced techniques, and real-world applications of NumPy in scientific computing and data analysis.

