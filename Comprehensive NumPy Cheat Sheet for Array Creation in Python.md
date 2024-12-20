## Comprehensive NumPy Cheat Sheet for Array Creation in Python
Slide 1: Introduction to NumPy Arrays

NumPy is a powerful library for numerical computing in Python. At its core are NumPy arrays, which are efficient, multi-dimensional containers for homogeneous data. These arrays form the foundation for many scientific and mathematical operations in Python.

```python
import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D array:", arr_1d)

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array:\n", arr_2d)

# Output:
# 1D array: [1 2 3 4 5]
# 2D array:
# [[1 2 3]
#  [4 5 6]]
```

Slide 2: Array Creation Functions

NumPy provides various functions to create arrays with specific properties. These functions are essential for initializing data structures efficiently.

```python
import numpy as np

# Create an array of zeros
zeros_arr = np.zeros((3, 4))
print("Zeros array:\n", zeros_arr)

# Create an array of ones
ones_arr = np.ones((2, 3))
print("Ones array:\n", ones_arr)

# Create an identity matrix
identity_matrix = np.eye(3)
print("Identity matrix:\n", identity_matrix)

# Output:
# Zeros array:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
# Ones array:
# [[1. 1. 1.]
#  [1. 1. 1.]]
# Identity matrix:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

Slide 3: Array Ranges and Sequences

NumPy offers functions to create arrays with evenly spaced values, which are useful for generating sequences and ranges.

```python
import numpy as np

# Create an array with a range of values
range_arr = np.arange(0, 10, 2)
print("Range array:", range_arr)

# Create an array with evenly spaced values
linspace_arr = np.linspace(0, 1, 5)
print("Linspace array:", linspace_arr)

# Create a logarithmically spaced array
logspace_arr = np.logspace(0, 2, 5)
print("Logspace array:", logspace_arr)

# Output:
# Range array: [0 2 4 6 8]
# Linspace array: [0.   0.25 0.5  0.75 1.  ]
# Logspace array: [  1.           3.16227766  10.          31.6227766  100.        ]
```

Slide 4: Reshaping and Transposing Arrays

NumPy allows easy manipulation of array shapes and dimensions, enabling efficient data restructuring.

```python
import numpy as np

# Create a 1D array
arr = np.arange(12)
print("Original array:", arr)

# Reshape the array to 2D
reshaped_arr = arr.reshape(3, 4)
print("Reshaped array:\n", reshaped_arr)

# Transpose the 2D array
transposed_arr = reshaped_arr.T
print("Transposed array:\n", transposed_arr)

# Output:
# Original array: [ 0  1  2  3  4  5  6  7  8  9 10 11]
# Reshaped array:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
# Transposed array:
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
```

Slide 5: Array Indexing and Slicing

Efficient data access and manipulation in NumPy arrays is achieved through indexing and slicing operations.

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Original array:\n", arr)

# Indexing: Get a single element
print("Element at (1, 2):", arr[1, 2])

# Slicing: Get a sub-array
print("Slice (first 2 rows, last 2 columns):\n", arr[:2, 2:])

# Boolean indexing
mask = arr > 5
print("Elements greater than 5:\n", arr[mask])

# Output:
# Original array:
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# Element at (1, 2): 7
# Slice (first 2 rows, last 2 columns):
# [[3 4]
#  [7 8]]
# Elements greater than 5:
# [ 6  7  8  9 10 11 12]
```

Slide 6: Basic Array Operations

NumPy provides efficient element-wise operations on arrays, simplifying mathematical computations.

```python
import numpy as np

# Create two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition
print("Addition:", a + b)

# Multiplication
print("Multiplication:", a * b)

# Exponentiation
print("Exponentiation:", a ** 2)

# Dot product
print("Dot product:", np.dot(a, b))

# Output:
# Addition: [5 7 9]
# Multiplication: [ 4 10 18]
# Exponentiation: [1 4 9]
# Dot product: 32
```

Slide 7: Array Broadcasting

Broadcasting allows NumPy to perform operations on arrays with different shapes, expanding smaller arrays to match larger ones.

```python
import numpy as np

# Create a 2D array and a 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

# Broadcasting: Add 1D array to each row of 2D array
result = arr_2d + arr_1d
print("Result of broadcasting:")
print(result)

# Broadcasting with scalar
scalar_result = arr_2d * 2
print("Result of scalar broadcasting:")
print(scalar_result)

# Output:
# Result of broadcasting:
# [[11 22 33]
#  [14 25 36]]
# Result of scalar broadcasting:
# [[ 2  4  6]
#  [ 8 10 12]]
```

Slide 8: Array Aggregation Functions

NumPy provides various functions for performing aggregate operations on arrays, such as computing sums, means, and extrema.

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original array:\n", arr)

# Sum of all elements
print("Sum of all elements:", np.sum(arr))

# Mean of all elements
print("Mean of all elements:", np.mean(arr))

# Maximum and minimum values
print("Maximum value:", np.max(arr))
print("Minimum value:", np.min(arr))

# Sum along axis 0 (columns)
print("Sum along columns:", np.sum(arr, axis=0))

# Mean along axis 1 (rows)
print("Mean along rows:", np.mean(arr, axis=1))

# Output:
# Original array:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
# Sum of all elements: 45
# Mean of all elements: 5.0
# Maximum value: 9
# Minimum value: 1
# Sum along columns: [12 15 18]
# Mean along rows: [2. 5. 8.]
```

Slide 9: Array Sorting and Searching

NumPy provides efficient functions for sorting arrays and searching for specific elements or conditions.

```python
import numpy as np

# Create an unsorted array
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
print("Original array:", arr)

# Sort the array
sorted_arr = np.sort(arr)
print("Sorted array:", sorted_arr)

# Find indices that would sort the array
sort_indices = np.argsort(arr)
print("Indices that would sort the array:", sort_indices)

# Find unique elements
unique_elements = np.unique(arr)
print("Unique elements:", unique_elements)

# Search for a value
value_to_search = 5
indices = np.where(arr == value_to_search)
print(f"Indices where {value_to_search} is found:", indices[0])

# Output:
# Original array: [3 1 4 1 5 9 2 6 5 3 5]
# Sorted array: [1 1 2 3 3 4 5 5 5 6 9]
# Indices that would sort the array: [1 3 6 0 9 2 4 8 10 7 5]
# Unique elements: [1 2 3 4 5 6 9]
# Indices where 5 is found: [4 8 10]
```

Slide 10: Array Concatenation and Splitting

NumPy allows for easy combination and division of arrays along specified axes.

```python
import numpy as np

# Create two arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Concatenate arrays vertically
vertical_concat = np.concatenate((arr1, arr2), axis=0)
print("Vertical concatenation:\n", vertical_concat)

# Concatenate arrays horizontally
horizontal_concat = np.concatenate((arr1, arr2), axis=1)
print("Horizontal concatenation:\n", horizontal_concat)

# Split an array vertically
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
vertical_split = np.split(arr, 3, axis=0)
print("Vertical split:")
for i, sub_arr in enumerate(vertical_split):
    print(f"Sub-array {i}:\n", sub_arr)

# Output:
# Vertical concatenation:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# Horizontal concatenation:
# [[1 2 5 6]
#  [3 4 7 8]]
# Vertical split:
# Sub-array 0:
#  [[1 2 3 4]]
# Sub-array 1:
#  [[5 6 7 8]]
# Sub-array 2:
#  [[ 9 10 11 12]]
```

Slide 11: Array Random Sampling

NumPy's random module provides functions for generating random numbers and sampling from various probability distributions.

```python
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random integers
random_ints = np.random.randint(0, 10, size=5)
print("Random integers:", random_ints)

# Generate random floats
random_floats = np.random.random(5)
print("Random floats:", random_floats)

# Sample from a normal distribution
normal_samples = np.random.normal(loc=0, scale=1, size=5)
print("Samples from normal distribution:", normal_samples)

# Shuffle an array
arr = np.arange(10)
np.random.shuffle(arr)
print("Shuffled array:", arr)

# Output:
# Random integers: [6 3 7 4 6]
# Random floats: [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
# Samples from normal distribution: [ 0.60276338 -0.54488318  0.43788141  0.88053932  1.46564877]
# Shuffled array: [2 8 4 9 1 6 7 3 0 5]
```

Slide 12: Real-life Example: Image Processing

NumPy is extensively used in image processing tasks. Here's an example of how to load an image, convert it to grayscale, and apply a simple filter.

```python
import numpy as np
from PIL import Image

# Load an image (assuming 'image.jpg' exists in the current directory)
img = np.array(Image.open('image.jpg'))
print("Original image shape:", img.shape)

# Convert to grayscale
gray_img = np.mean(img, axis=2).astype(np.uint8)
print("Grayscale image shape:", gray_img.shape)

# Apply a simple blur filter
kernel = np.ones((5, 5)) / 25  # 5x5 averaging filter
blurred = np.zeros_like(gray_img)
for i in range(2, gray_img.shape[0] - 2):
    for j in range(2, gray_img.shape[1] - 2):
        blurred[i, j] = np.sum(gray_img[i-2:i+3, j-2:j+3] * kernel)

print("Blurred image shape:", blurred.shape)

# Save the processed images
Image.fromarray(gray_img).save('grayscale.jpg')
Image.fromarray(blurred).save('blurred.jpg')

# Note: This code assumes you have an 'image.jpg' file in your working directory
# and the necessary permissions to read and write files.
```

Slide 13: Real-life Example: Data Analysis

NumPy is crucial for data analysis tasks. Here's an example of analyzing temperature data for a city over a year.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic temperature data for a year (365 days)
temperatures = np.random.normal(loc=15, scale=10, size=365)  # Mean 15°C, std dev 10°C

# Calculate basic statistics
avg_temp = np.mean(temperatures)
max_temp = np.max(temperatures)
min_temp = np.min(temperatures)

print(f"Average temperature: {avg_temp:.2f}°C")
print(f"Maximum temperature: {max_temp:.2f}°C")
print(f"Minimum temperature: {min_temp:.2f}°C")

# Find days above 25°C (hot days)
hot_days = np.sum(temperatures > 25)
print(f"Number of hot days (>25°C): {hot_days}")

# Calculate moving average (7-day window)
moving_avg = np.convolve(temperatures, np.ones(7), 'valid') / 7

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(temperatures, label='Daily Temperature')
plt.plot(np.arange(3, 362), moving_avg, label='7-day Moving Average', color='red')
plt.xlabel('Day of the Year')
plt.ylabel('Temperature (°C)')
plt.title('Yearly Temperature Analysis')
plt.legend()
plt.grid(True)
plt.savefig('temperature_analysis.png')
plt.close()

print("Temperature analysis plot saved as 'temperature_analysis.png'")

# Note: This code generates a plot file. Ensure you have write permissions
# in your working directory.
```

Slide 14: Additional Resources

For further exploration of NumPy and its applications in scientific computing, consider the following resources:

1. NumPy official documentation: [https://numpy.org/doc/](https://numpy.org/doc/) This comprehensive guide covers all aspects of NumPy, from basic to advanced topics.
2. "From Python to NumPy" by Nicolas P. Rougier Available at: [https://www.labri.fr/perso/nrougier/from-python-to-numpy/](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) This free online book provides an in-depth look at NumPy's capabilities and optimizations.
3. "Python Data Science Handbook" by Jake VanderPlas This book includes extensive coverage of NumPy and its integration with other data science tools.
4. SciPy Lecture Notes Available at: [https://scipy-lectures.org/](https://scipy-lectures.org/) These lecture notes cover NumPy along with other scientific Python libraries.
5. NumPy tutorials on Real Python Available at: [https://realpython.com/tutorials/numpy/](https://realpython.com/tutorials/numpy/) A collection of practical tutorials covering various aspects of NumPy.
6. ArXiv paper: "Array programming with NumPy" by Harris et al. (2020) ArXiv URL: [https://arxiv.org/abs/2006.10256](https://arxiv.org/abs/2006.10256) This paper provides insights into NumPy's design and its impact on scientific computing.

These resources offer a mix of official documentation, books, tutorials, and academic papers to deepen your understanding of NumPy and its applications in scientific computing and data analysis.

