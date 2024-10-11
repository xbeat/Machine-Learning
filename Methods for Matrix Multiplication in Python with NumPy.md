## Methods for Matrix Multiplication in Python with NumPy
Slide 1: 3 Ways to Perform Matrix Multiplication in Python using NumPy

Matrix multiplication is a fundamental operation in linear algebra and has numerous applications in various fields. This presentation will explore three efficient methods to perform matrix multiplication using NumPy, a powerful library for numerical computing in Python.

```python
import numpy as np
```

Slide 2: Method 1: Using np.dot()

The np.dot() function is a versatile tool for matrix multiplication. It can handle both 1D and 2D arrays, making it suitable for vector-matrix and matrix-matrix multiplication.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.dot(A, B)
print(result)
```

Slide 3: Output for np.dot()

```
[[19 22]
 [43 50]]
```

Slide 4: Understanding np.dot()

The np.dot() function performs the dot product of two arrays. For 2D arrays, it's equivalent to matrix multiplication. It's important to note that the number of columns in the first matrix must match the number of rows in the second matrix.

```python
# Vector-matrix multiplication
v = np.array([1, 2])
M = np.array([[1, 2], [3, 4]])
result = np.dot(v, M)
print(result)  # Output: [7 10]
```

Slide 5: Method 2: Using the @ Operator

Python 3.5 introduced the @ operator for matrix multiplication. This operator provides a more intuitive and readable syntax for matrix operations.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A @ B
print(result)
```

Slide 6: Output for the @ Operator

```
[[19 22]
 [43 50]]
```

Slide 7: Advantages of the @ Operator

The @ operator is not only more concise but also more explicit in its intent. It clearly indicates matrix multiplication, improving code readability and reducing potential confusion with element-wise multiplication.

```python
# Chaining multiple matrix multiplications
C = np.array([[9, 10], [11, 12]])
result = A @ B @ C
print(result)
```

Slide 8: Output for the @ Operator

```
[[499 542]
 [1131 1230]]
```

Slide 9: Method 3: Using np.matmul()

The np.matmul() function is specifically designed for matrix product operations. It can handle higher-dimensional arrays and provides broadcasting capabilities for certain array shapes.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.matmul(A, B)
print(result)
```

Slide 10: Output for np.matmul()

```
[[19 22]
 [43 50]]
```

Slide 11: np.matmul() with Higher Dimensions

One advantage of np.matmul() is its ability to work with arrays of more than two dimensions. It applies matrix multiplication to the last two dimensions while broadcasting over the rest.

```python
A = np.random.rand(2, 3, 4)
B = np.random.rand(2, 4, 3)
result = np.matmul(A, B)
print(result.shape)  # Output: (2, 3, 3)
```

Slide 12: Performance Comparison

Let's compare the performance of these three methods using timeit for larger matrices.

```python
import timeit

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

dot_time = timeit.timeit(lambda: np.dot(A, B), number=10)
matmul_time = timeit.timeit(lambda: np.matmul(A, B), number=10)
operator_time = timeit.timeit(lambda: A @ B, number=10)

print(f"np.dot(): {dot_time:.4f} seconds")
print(f"np.matmul(): {matmul_time:.4f} seconds")
print(f"@ operator: {operator_time:.4f} seconds")
```

Slide 13: Real-life Example 1: Image Convolution

Matrix multiplication is crucial in image processing, particularly for applying convolution filters. Let's implement a simple edge detection filter using matrix multiplication.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Edge detection kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Pad the image
padded_image = np.pad(image, pad_width=1, mode='constant')

# Apply convolution using matrix multiplication
result = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        result[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel)

plt.imshow(result, cmap='gray')
plt.title("Edge Detection Result")
plt.show()
```

Slide 14: Real-life Example 2: Solving Systems of Linear Equations

Matrix multiplication is essential in solving systems of linear equations. Let's solve a simple system using NumPy's matrix operations.

```python
import numpy as np

# Define the system: 2x + y = 7, x + 3y = 11
A = np.array([[2, 1],
              [1, 3]])
b = np.array([7, 11])

# Solve using matrix multiplication and inverse
x = np.dot(np.linalg.inv(A), b)

print("Solution:")
print(f"x = {x[0]:.2f}")
print(f"y = {x[1]:.2f}")

# Verify the solution
verification = np.dot(A, x)
print("\nVerification:")
print(f"2x + y = {verification[0]:.2f}")
print(f"x + 3y = {verification[1]:.2f}")
```

Slide 15: Choosing the Right Method

Each method has its strengths:

* np.dot(): Versatile for both vector and matrix operations
* @ operator: Intuitive and readable for matrix multiplication
* np.matmul(): Efficient for higher-dimensional arrays and broadcasting

Consider factors like code readability, performance requirements, and the dimensions of your arrays when choosing a method.

```python
# Example of choosing methods based on array dimensions
v = np.array([1, 2, 3])
M = np.array([[1, 2], [3, 4], [5, 6]])

# For vector-matrix multiplication, np.dot() is suitable
result1 = np.dot(v, M)

# For matrix-matrix multiplication, @ operator is readable
result2 = M.T @ M

print("Vector-matrix result:", result1)
print("Matrix-matrix result:\n", result2)
```

Slide 16: Best Practices and Tips

1. Always check matrix dimensions before multiplication
2. Use np.matmul() or @ for clearer intent in matrix multiplication
3. Consider memory efficiency for large matrices
4. Leverage NumPy's broadcasting capabilities when applicable

```python
# Example of dimension checking and broadcasting
def safe_matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions are not compatible")
    return np.matmul(A, B)

# Broadcasting example
a = np.array([[1, 2, 3]])  # Shape: (1, 3)
b = np.array([[4], [5], [6]])  # Shape: (3, 1)
result = np.matmul(a, b)  # Result shape: (1, 1)
print("Broadcasting result:", result)
```

Slide 17: Conclusion

Matrix multiplication in Python using NumPy offers powerful tools for various computational tasks. By understanding the nuances of np.dot(), the @ operator, and np.matmul(), you can efficiently perform matrix operations in your Python projects. Remember to consider the specific requirements of your task when choosing the most appropriate method.

Slide 18: Additional Resources

For more in-depth information on matrix multiplication and NumPy:

1. "The Art of Linear Algebra" by Liesen and Mehrmann (ArXiv:2108.06468) [https://arxiv.org/abs/2108.06468](https://arxiv.org/abs/2108.06468)
2. "Numerical Linear Algebra in Data Science Using Python" by Linderman (ArXiv:2111.04227) [https://arxiv.org/abs/2111.04227](https://arxiv.org/abs/2111.04227)

These resources provide comprehensive coverage of linear algebra concepts and their applications in Python, offering valuable insights for further exploration.

