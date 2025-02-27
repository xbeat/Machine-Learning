## NumPy Fundamentals for TikTok

Slide 1 (Introduction): NumPy Fundamentals Discover the power of NumPy, the essential library for scientific computing in Python.

Slide 2: What is NumPy? NumPy is a powerful library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a vast collection of high-level mathematical functions to operate on these arrays. Code Example:

```python
import numpy as np
```

Slide 3: NumPy arrays can be created from Python lists or using special functions. Code Example:

```python
# From a Python list
a = np.array([1, 2, 3, 4])
# Output: [1 2 3 4]

# Using special functions
b = np.zeros((2, 3))  # Create an array of zeros
# Output: [[0. 0. 0.]
#          [0. 0. 0.]]
c = np.ones((3, 2))   # Create an array of ones
# Output: [[1. 1.]
#          [1. 1.]
#          [1. 1.]]
d = np.random.rand(2, 2)  # Create an array of random values
# Output will be different each time
```
Slide 4: Creating numpy arrays of random values and sizes and other similar functions
```python
random_array = np.random.randint(0,10,size=(2, 3)) # Creates a random array of size 2x3 and values ranging from 0 to 10
print(random_array)

random_array = np.random.rand(1,10) # Gives an random array of desired size 1x10 and random values
print(random_array) #Ex) [[0.94200759 0.78812449 0.22936564 0.30757089 0.28442394 0.49795422
# 0.38306793 0.85061234 0.91607897 0.54049652]]

random_int = np.random.randint( 1,100)
print(random_int)  # Output will be a random integer between 1 and 99

random_array = np.random.randn(2, 3) # Outputs a random array of normal distribution and desired size 
print(random_array)

# Generate random numbers from a uniform distribution between 0 and 1
random_uniform = np.random.uniform(0, 1)
print(random_uniform)  # Output will be a random number between 0 and 1

sample = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
print(sample)  # Output will be a random sample from the list without repetition
np.mean(sample)

# Generate a random permutation of elements in an array
permutation = np.random.permutation([1, 2, 3, 4, 5])
print(permutation)  # Output will be a randomly permuted version of the list

# Generate a random number from a binomial distribution (number of trials=10, probability of success=0.5)
random_binomial = np.random.binomial(10, 0.5)
print(random_binomial)

arr = np.linspace(0, 1, 5)  # 5 numbers between 0 and 1
print(arr)  # Output = [0.   0.25 0.5  0.75 1.  ]

``` 
Slide 5: Matrix operations with numpy 
```python
# Addition
matrix1 = np.array([[1, 2], 
                    [3, 4]])
matrix2 = np.array([[5, 6], 
                    [7, 8]])
result_add = np.add(matrix1, matrix2)  # Or simply: matrix1 + matrix2
# Output = [[ 6  8], [10 12]]

#Matrix multiplication
matrix3 = np.dot(matrix1 , matrix2)  # Or simply: matrix1 @ matrix2
# Output = [[19 22],
#            [43 50]]

# Inverse of a matrix
matrix = np.array([[1, 2],
                   [3, 4]])
inverse = np.linalg.inv(matrix) # its determinant should be non zero

#You can add or subtract a scalar or a 1D array (row/column vector) to/from a matrix using broadcasting.
scalar = 10
result_scalar_add = matrix1 + scalar
#Output = [[11 12]
#          [13 14]]

result_scalar_multiply = matrix1 * scalar
# Output = [[ 2  4]
#           [ 6  8]]


# Transpose of a matrix
matrix4 = np.array([[1, 2],
                    [3, 4],
                   [5, 6]])
transpose_matrix = matrix.T
#Original Matrix:
#[[1 2]
# [3 4]
# [5 6]]

#Transpose of the Matrix:
#[[1 3 5]
# [2 4 6]]
```

Slide 6: Array Indexing and Slicing NumPy arrays can be indexed and sliced like Python lists, but with more flexibility. Code Example:

```python
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[1, 2])  # Output: 6
print(a[:, 1])  # Output: [2 5 8]
print(a[1:, :2])  # Output: [[4 5]
                  #          [7 8]]
```

Slide 7: Array Operations NumPy provides a wide range of mathematical operations that can be applied to arrays elementwise or across entire arrays. Code Example:

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b   # Elementwise addition
# Output: [5 7 9]
d = a * b   # Elementwise multiplication
# Output: [4 10 18]
e = np.sum(a)  # Sum of all elements in the array
# Output: 6
```

Slide 8: Broadcasting NumPy's broadcasting feature allows arithmetic operations between arrays with different shapes. Code Example:

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

c = a + b  # Broadcasting: b is "stretched" to match a's shape
# Output: [[11 22 33]
#          [14 25 36]]

d = a*b # Broadcasting allows b to be automatically "stretched" to match the dimensions of a
# Output = [[ 10  40  90],[ 40 100 180]]


```

Slide 9: Array Reshaping NumPy arrays can be reshaped to different dimensions without changing their data. Code Example:

```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)  # Reshape to a 2x3 array
# Output: [[1 2 3]
#          [4 5 6]]
c = a.reshape(3, 2)  # Reshape to a 3x2 array
# Output: [[1 2]
#          [3 4]
#          [5 6]]
```

Slide 10: Array Concatenation NumPy provides functions to concatenate arrays along different axes. Code Example:

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

c = np.concatenate((a, b), axis=0)  # Concatenate along rows
# Output: [[1 2]
#          [3 4]
#          [5 6]
#          [7 8]]
d = np.concatenate((a, b), axis=1)  # Concatenate along columns
# Output: [[1 2 5 6]
#          [3 4 7 8]]
```

Slide 11: Conditions and Boolean Arrays NumPy allows you to apply conditions and create boolean arrays for advanced indexing and filtering. Code Example:

```python
a = np.array([1, 2, 3, 4, 5])
condition = a > 2
# Output: [False False  True  True  True]
b = a[condition]  # Filter elements greater than 2
# Output: [3 4 5]
```

Slide 12: Mathematical Functions NumPy provides a wide range of mathematical functions to perform various operations on arrays. Code Example:

```python
a = np.array([1, 2, 3, 4])
b = np.sin(a)  # Compute sine values
# Output: [ 0.84147098  0.90929743  0.14112001 -0.7568025 ]
c = np.exp(a)  # Compute exponential values
# Output: [2.71828183e+00 7.38905610e+00 2.00855369e+01 5.45981500e+01]
d = np.sqrt(a)  # Compute square root values
# Output: [1.         1.41421356 1.73205081 2.        ]
```

Slide 13: Loading and Saving Arrays NumPy provides functions to load and save arrays from/to disk in various formats. Code Example:

```python
# Save an array to a binary file
a = np.array([1, 2, 3, 4])
np.save('data.npy', a)

# Load an array from a binary file
b = np.load('data.npy')
# Output: [1 2 3 4]
```

Slide 14: NumPy and Data Analysis NumPy seamlessly integrates with other data analysis libraries like Pandas and Matplotlib, making it an essential tool for scientific computing and data analysis in Python. Code Example:

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
x = data['x'].values  # Convert to NumPy array
y = data['y'].values

plt.plot(x, y)
plt.show()
# Displays a line plot using the NumPy arrays x and y
```
 Slide 15: Indexing rows, columns, sub-matrices:
 ```python
m =  np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
m[0,2] # Outputs 7 (item index (1,3) for mathematiicans)

m[[0,2]] # Will retrieve the first and third rows:
# Output = [[1, 2, 3],
#           [7, 8, 9]]

m = [: , [1,2]] # Will retrive second and third columns:
#Output = [[2, 3],
#          [4, 5],
#          [8, 9]]

m[[0,2]][:,[1,2]] #Outputs a sub-matrix with first and third rows and second and third columns:
# Output = [[2, 3],
#           [8, 9]] 
```
 
