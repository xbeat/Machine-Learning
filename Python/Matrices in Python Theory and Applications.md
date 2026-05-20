## Matrices in Python: Theory and Applications

Slide 1: Introduction to Matrices

Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are fundamental in linear algebra and have widespread applications in various fields.

```python
import numpy as np

# Creating a 2x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(matrix)
print(f"Shape: {matrix.shape}")
```

Slide 2: Matrix Operations - Addition and Subtraction

Matrix addition and subtraction are performed element-wise between matrices of the same dimensions.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

addition = A + B
subtraction = A - B

print("Addition:")
print(addition)
print("\nSubtraction:")
print(subtraction)
```

Slide 3: Matrix Multiplication

Matrix multiplication involves dot products of rows and columns. The number of columns in the first matrix must equal the number of rows in the second.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

multiplication = np.dot(A, B)
print("Matrix Multiplication:")
print(multiplication)
```

Slide 4: Transpose of a Matrix

The transpose of a matrix flips it over its diagonal, switching rows and columns.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])

A_transpose = A.T
print("Original matrix:")
print(A)
print("\nTransposed matrix:")
print(A_transpose)
```

Slide 5: Identity Matrix

An identity matrix is a square matrix with 1s on the main diagonal and 0s elsewhere. It's crucial in matrix operations.

```python
n = 3
identity = np.eye(n)
print(f"{n}x{n} Identity Matrix:")
print(identity)
```

Slide 6: Determinant of a Matrix

The determinant is a scalar value that provides information about a square matrix's properties.

```python
A = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(A)
print("Matrix A:")
print(A)
print(f"\nDeterminant of A: {det_A}")
```

Slide 7: Inverse of a Matrix

The inverse of a matrix A is another matrix that, when multiplied by A, results in the identity matrix.

```python
A = np.array([[1, 2], [3, 4]])
A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print("\nInverse of A:")
print(A_inv)
print("\nVerification (A * A_inv):")
print(np.dot(A, A_inv))
```

Slide 8: Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are important in understanding linear transformations and solving systems of differential equations.

```python
A = np.array([[4, -2], [1, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
```

Slide 9: Solving Systems of Linear Equations

Matrices can efficiently represent and solve systems of linear equations.

```python
# Solving Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 5])

x = np.linalg.solve(A, b)
print("Solution x:")
print(x)
print("\nVerification (Ax):")
print(np.dot(A, x))
```

Slide 10: Matrix Decomposition - LU Decomposition

LU decomposition factors a matrix into lower and upper triangular matrices, useful for solving linear systems and computing determinants.

```python
from scipy.linalg import lu

A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])

P, L, U = lu(A)
print("Original matrix A:")
print(A)
print("\nLower triangular matrix L:")
print(L)
print("\nUpper triangular matrix U:")
print(U)
```

Slide 11: Singular Value Decomposition (SVD)

SVD is a factorization technique with applications in data compression, dimensionality reduction, and recommendation systems.

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, Vt = np.linalg.svd(A)

print("Original matrix A:")
print(A)
print("\nLeft singular vectors U:")
print(U)
print("\nSingular values s:")
print(s)
print("\nRight singular vectors V^T:")
print(Vt)
```

Slide 12: Real-life Example 1: Image Compression

Matrices are used in image processing for compression using SVD.

```python
import matplotlib.pyplot as plt

# Load an image and convert to grayscale
img = plt.imread('example_image.jpg')
gray_img = np.mean(img, axis=2)

U, s, Vt = np.linalg.svd(gray_img)

# Reconstruct image with fewer singular values
k = 50
compressed_img = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(gray_img, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(compressed_img, cmap='gray'), plt.title(f'Compressed (k={k})')
plt.show()
```

Slide 13: Real-life Example 2: PageRank Algorithm

The PageRank algorithm, used by search engines, relies on matrix operations to rank web pages.

```python
def pagerank(adjacency_matrix, damping_factor=0.85, epsilon=1e-8):
    n = adjacency_matrix.shape[0]
    out_degree = np.sum(adjacency_matrix, axis=1)
    
    # Normalize the adjacency matrix
    M = adjacency_matrix / out_degree[:, np.newaxis]
    
    # Initialize PageRank vector
    r = np.ones(n) / n
    
    while True:
        r_new = (1 - damping_factor) / n + damping_factor * M.T @ r
        if np.sum(np.abs(r_new - r)) < epsilon:
            return r_new
        r = r_new

# Example usage
A = np.array([[0, 1, 1, 0],
              [0, 0, 1, 1],
              [1, 0, 0, 1],
              [0, 0, 1, 0]])

pagerank_scores = pagerank(A)
print("PageRank scores:")
print(pagerank_scores)
```

Slide 14: Additional Resources

For further exploration of Matrix Theory and its applications:

1. "Matrix Computations" by Gene H. Golub and Charles F. Van Loan ArXiv link: [https://arxiv.org/abs/1607.00687](https://arxiv.org/abs/1607.00687)
2. "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III
3. "Applied Linear Algebra" by Peter J. Olver and Chehrzad Shakiban ArXiv link: [https://arxiv.org/abs/1609.05636](https://arxiv.org/abs/1609.05636)

These resources provide in-depth coverage of matrix theory, algorithms, and applications in various fields.

