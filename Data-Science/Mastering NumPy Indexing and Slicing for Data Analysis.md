## Mastering NumPy Indexing and Slicing for Data Analysis
Slide 1: Introduction to NumPy Indexing and Slicing

NumPy, a fundamental library for scientific computing in Python, offers powerful tools for data manipulation. Indexing and slicing are key techniques that allow efficient access and modification of array elements. These operations form the foundation for advanced data analysis and processing in NumPy.

```python
import numpy as np

# Create a sample 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Original array:")
print(arr)
```

Slide 2: Basic Indexing in NumPy

NumPy arrays support integer indexing similar to Python lists. However, NumPy extends this concept to multiple dimensions, allowing precise element selection in multi-dimensional arrays.

```python
# Accessing elements using integer indexing
print("Element at index [1, 2]:", arr[1, 2])
print("First row:", arr[0])
print("Last column:", arr[:, -1])
```

Slide 3: Slicing in NumPy

Slicing in NumPy allows extracting subarrays by specifying start, stop, and step values for each dimension. This powerful feature enables efficient data subset selection and manipulation.

```python
# Slicing examples
print("First two rows, all columns:")
print(arr[:2, :])

print("All rows, last two columns:")
print(arr[:, 1:])

print("Every other element in the first row:")
print(arr[0, ::2])
```

Slide 4: Advanced Indexing: Boolean Indexing

Boolean indexing uses a boolean array to select elements that satisfy specific conditions. This technique is particularly useful for filtering data based on complex criteria.

```python
# Boolean indexing
mask = arr > 5
print("Elements greater than 5:")
print(arr[mask])

# Combining conditions
complex_mask = (arr > 3) & (arr < 8)
print("Elements between 3 and 8:")
print(arr[complex_mask])
```

Slide 5: Advanced Indexing: Integer Array Indexing

Integer array indexing allows selecting elements using arrays of indices. This technique enables complex element selection and rearrangement operations.

```python
# Integer array indexing
row_indices = np.array([0, 1, 2])
col_indices = np.array([2, 1, 0])
print("Selected elements:")
print(arr[row_indices, col_indices])

# Selecting specific elements
print("Elements at (0,0), (1,1), and (2,2):")
print(arr[np.arange(3), np.arange(3)])
```

Slide 6: Modifying Array Elements

NumPy's indexing and slicing capabilities also allow for efficient array modification. Elements can be updated individually or in groups using various indexing techniques.

```python
# Modifying elements
arr[1, 1] = 10
arr[:, 2] = [30, 60, 90]
print("Modified array:")
print(arr)

# Broadcasting with boolean indexing
arr[arr < 30] *= 2
print("Array after doubling elements < 30:")
print(arr)
```

Slide 7: Real-Life Example: Image Processing

NumPy's indexing and slicing are extensively used in image processing. Let's demonstrate a simple image cropping operation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a sample 8x8 grayscale image
image = np.random.randint(0, 256, (8, 8))

# Crop the image
cropped = image[2:6, 2:6]

# Display original and cropped images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title("Original Image")
ax2.imshow(cropped, cmap='gray')
ax2.set_title("Cropped Image")
plt.show()
```

Slide 8: Fancy Indexing

Fancy indexing allows selecting or modifying subsets of an array in a flexible manner using integer arrays or boolean masks. This technique is particularly useful for complex data manipulation tasks.

```python
# Create a sample array
arr = np.arange(16).reshape(4, 4)

# Select specific rows and columns
rows = np.array([0, 2, 3])
cols = np.array([1, 2])
selected = arr[rows[:, np.newaxis], cols]

print("Original array:")
print(arr)
print("\nSelected sub-array:")
print(selected)
```

Slide 9: Masking and Filtering

Masking allows for conditional selection of array elements based on their values or other criteria. This technique is crucial for data cleaning and preprocessing.

```python
# Create a sample array
data = np.random.randn(5, 5)

# Create a mask for positive values
mask = data > 0

# Apply the mask
filtered_data = data[mask]

print("Original data:")
print(data)
print("\nFiltered data (positive values only):")
print(filtered_data)
```

Slide 10: Slicing with Step Size

NumPy allows specifying a step size when slicing, enabling selection of every nth element. This is useful for downsampling or selecting specific patterns in data.

```python
# Create a sample array
arr = np.arange(20)

# Select every third element
every_third = arr[::3]

# Reverse the array
reversed_arr = arr[::-1]

print("Original array:", arr)
print("Every third element:", every_third)
print("Reversed array:", reversed_arr)
```

Slide 11: Multidimensional Slicing

NumPy's slicing capabilities extend seamlessly to multidimensional arrays, allowing for complex data extraction and manipulation in higher dimensions.

```python
# Create a 3D array
arr_3d = np.arange(27).reshape(3, 3, 3)

# Extract a 2D slice
slice_2d = arr_3d[1, :, :]

# Extract a 1D slice
slice_1d = arr_3d[1, 1, :]

print("3D array:")
print(arr_3d)
print("\n2D slice:")
print(slice_2d)
print("\n1D slice:")
print(slice_1d)
```

Slide 12: Real-Life Example: Time Series Analysis

NumPy's indexing and slicing are invaluable in time series analysis. Let's demonstrate how to select specific time periods from a dataset.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
dates = np.arange('2023-01-01', '2024-01-01', dtype='datetime64[D]')
values = np.cumsum(np.random.randn(365))

# Select data for Q2 2023
q2_mask = (dates >= '2023-04-01') & (dates < '2023-07-01')
q2_dates = dates[q2_mask]
q2_values = values[q2_mask]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(dates, values, label='Full Year')
plt.plot(q2_dates, q2_values, label='Q2 2023')
plt.title('Time Series Data Analysis')
plt.legend()
plt.show()
```

Slide 13: Performance Considerations

NumPy's indexing and slicing operations are highly optimized for performance. However, certain practices can significantly impact efficiency, especially when dealing with large datasets.

```python
import numpy as np
import timeit

# Create a large array
large_arr = np.random.rand(1000000)

# Compare performance of different indexing methods
def method1():
    return large_arr[large_arr > 0.5]

def method2():
    mask = large_arr > 0.5
    return large_arr[mask]

time1 = timeit.timeit(method1, number=100)
time2 = timeit.timeit(method2, number=100)

print(f"Method 1 time: {time1:.6f} seconds")
print(f"Method 2 time: {time2:.6f} seconds")
```

Slide 14: Advanced Indexing: Combining Techniques

NumPy allows combining different indexing and slicing techniques for complex data manipulation. This flexibility is crucial for advanced data analysis tasks.

```python
# Create a sample 3D array
arr_3d = np.arange(27).reshape(3, 3, 3)

# Combine boolean and integer indexing
mask = arr_3d > 10
selected = arr_3d[mask][:5]

# Combine slicing and fancy indexing
complex_slice = arr_3d[1:, [0, 2], ::2]

print("Selected elements:", selected)
print("\nComplex slice:")
print(complex_slice)
```

Slide 15: Additional Resources

For further exploration of NumPy indexing and slicing:

1. NumPy Official Documentation: [https://numpy.org/doc/stable/user/basics.indexing.html](https://numpy.org/doc/stable/user/basics.indexing.html)
2. "A Visual Intro to NumPy and Data Representation" by Jay Alammar: [https://jalammar.github.io/visual-numpy/](https://jalammar.github.io/visual-numpy/)
3. "NumPy: Creating and Manipulating Numerical Data" chapter in "Python Data Science Handbook" by Jake VanderPlas: [https://arxiv.org/abs/1607.01719](https://arxiv.org/abs/1607.01719)

These resources provide in-depth explanations and additional examples to enhance your understanding of NumPy's powerful indexing and slicing capabilities.

