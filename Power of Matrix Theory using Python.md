## Power of Matrix Theory using Python
Slide 1: Introduction to Matrix Theory

Matrix theory is a fundamental branch of mathematics with wide-ranging applications in various fields. It provides a powerful framework for solving complex problems in linear algebra, computer graphics, quantum mechanics, and more. This presentation will explore key concepts and practical applications of matrix theory using Python.

```python
import numpy as np

# Create a 2x2 matrix
A = np.array([[1, 2],
              [3, 4]])

print("Matrix A:")
print(A)

# Basic matrix operations
print("\nTranspose of A:")
print(A.T)

print("\nDeterminant of A:")
print(np.linalg.det(A))
```

Slide 2: Matrix Operations: Addition and Subtraction

Matrix addition and subtraction are fundamental operations performed element-wise. These operations are only defined for matrices of the same dimensions. Let's explore how to perform these operations using NumPy.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Addition
print("\nA + B:")
print(A + B)

# Subtraction
print("\nA - B:")
print(A - B)
```

Slide 3: Matrix Multiplication

Matrix multiplication is a crucial operation in matrix theory. Unlike addition and subtraction, multiplication is not commutative (A \* B ≠ B \* A). The number of columns in the first matrix must equal the number of rows in the second matrix.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix multiplication
C = np.dot(A, B)
print("\nA * B:")
print(C)

# Note: A * B ≠ B * A
D = np.dot(B, A)
print("\nB * A:")
print(D)
```

Slide 4: Matrix Transposition

The transpose of a matrix is obtained by interchanging its rows and columns. Transposition is a fundamental operation in matrix theory and is often used in various mathematical and computational problems.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original matrix A:")
print(A)

# Transpose the matrix
A_transposed = A.T

print("\nTransposed matrix A:")
print(A_transposed)

# Verify properties
print("\nIs (A^T)^T = A?", np.array_equal(A_transposed.T, A))
```

Slide 5: Determinants

The determinant is a scalar value that can be computed from a square matrix. It has many important applications, including solving systems of linear equations and finding inverse matrices. Let's calculate determinants using NumPy.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Matrix A:")
print(A)
print("Determinant of A:", np.linalg.det(A))

print("\nMatrix B:")
print(B)
print("Determinant of B:", np.linalg.det(B))

# Create a singular matrix
C = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
print("\nSingular Matrix C:")
print(C)
print("Determinant of C:", np.linalg.det(C))
```

Slide 6: Inverse Matrices

The inverse of a square matrix A, denoted as A^(-1), is a matrix that when multiplied with A, results in the identity matrix. Not all matrices have inverses; only non-singular matrices (determinant ≠ 0) are invertible.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

print("Matrix A:")
print(A)

# Calculate the inverse
A_inv = np.linalg.inv(A)

print("\nInverse of A:")
print(A_inv)

# Verify: A * A^(-1) = I
I = np.dot(A, A_inv)
print("\nA * A^(-1):")
print(np.round(I, decimals=10))  # Round to avoid floating-point errors

# Try inverting a singular matrix
B = np.array([[1, 2], [2, 4]])
try:
    B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print("\nError inverting singular matrix:", str(e))
```

Slide 7: Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental concepts in matrix theory. An eigenvector of a square matrix A is a non-zero vector v that, when multiplied by A, yields a scalar multiple of itself. This scalar is called an eigenvalue.

```python
import numpy as np

A = np.array([[4, -2], [1, 1]])

print("Matrix A:")
print(A)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Verify Av = λv for the first eigenpair
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0]

print("\nVerification:")
print("Av =", np.dot(A, v1))
print("λv =", lambda1 * v1)
```

Slide 8: Matrix Decomposition: LU Decomposition

Matrix decomposition is a powerful technique in matrix theory. LU decomposition factors a matrix into the product of a lower triangular matrix (L) and an upper triangular matrix (U). This decomposition is useful for solving linear systems and calculating determinants.

```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])

print("Matrix A:")
print(A)

# Perform LU decomposition
P, L, U = lu(A)

print("\nLower triangular matrix L:")
print(L)

print("\nUpper triangular matrix U:")
print(U)

# Verify A = P * L * U
print("\nVerification:")
print("A =\n", A)
print("P * L * U =\n", np.dot(P, np.dot(L, U)))
```

Slide 9: Solving Systems of Linear Equations

One of the most important applications of matrix theory is solving systems of linear equations. We can use matrix operations to solve these systems efficiently.

```python
import numpy as np

# System of equations:
# 2x + y = 8
# -3x + y = -11

A = np.array([[2, 1],
              [-3, 1]])
b = np.array([8, -11])

print("Matrix A:")
print(A)
print("\nVector b:")
print(b)

# Solve the system
x = np.linalg.solve(A, b)

print("\nSolution x:")
print(x)

# Verify the solution
print("\nVerification:")
print("Ax =", np.dot(A, x))
print("b  =", b)
```

Slide 10: Matrix Rank

The rank of a matrix is the dimension of the vector space spanned by its columns (or rows). It's a measure of the "nondegenerateness" of the system of linear equations represented by the matrix.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])

print("Matrix A:")
print(A)
print("Rank of A:", np.linalg.matrix_rank(A))

print("\nMatrix B:")
print(B)
print("Rank of B:", np.linalg.matrix_rank(B))

# Create a visualization of the column space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.quiver(0, 0, 0, *A.T, length=0.1, normalize=True)
ax1.set_title("Column Space of A")

ax2 = fig.add_subplot(122, projection='3d')
ax2.quiver(0, 0, 0, *B.T, length=0.1, normalize=True)
ax2.set_title("Column Space of B")

plt.tight_layout()
plt.show()
```

Slide 11: Matrix Norms

Matrix norms provide a way to measure the "size" of a matrix. They are useful in analyzing the stability of numerical algorithms and in error analysis. Let's explore some common matrix norms.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

print("Matrix A:")
print(A)

# Frobenius norm
frob_norm = np.linalg.norm(A, 'fro')
print("\nFrobenius norm:", frob_norm)

# Spectral norm (2-norm)
spectral_norm = np.linalg.norm(A, 2)
print("Spectral norm:", spectral_norm)

# 1-norm (maximum absolute column sum)
one_norm = np.linalg.norm(A, 1)
print("1-norm:", one_norm)

# Infinity norm (maximum absolute row sum)
inf_norm = np.linalg.norm(A, np.inf)
print("Infinity norm:", inf_norm)
```

Slide 12: Real-life Example: Image Compression

Matrix theory plays a crucial role in image compression techniques. One such method is the Singular Value Decomposition (SVD), which can be used to approximate images with fewer data points.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 100x100 grayscale image
image = np.zeros((100, 100))
image[25:75, 25:75] = 1  # White square in the middle

# Perform SVD
U, s, Vt = np.linalg.svd(image)

# Reconstruct the image using different numbers of singular values
def reconstruct(U, s, Vt, k):
    return np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(Vt[:k, :])

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 1].imshow(reconstruct(U, s, Vt, 5), cmap='gray')
axs[0, 1].set_title('k = 5')
axs[1, 0].imshow(reconstruct(U, s, Vt, 10), cmap='gray')
axs[1, 0].set_title('k = 10')
axs[1, 1].imshow(reconstruct(U, s, Vt, 20), cmap='gray')
axs[1, 1].set_title('k = 20')

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Markov Chains

Markov chains are mathematical systems that transition from one state to another according to certain probabilistic rules. They are represented using stochastic matrices and have applications in various fields, including physics, biology, and computer science.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the transition matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

# Initial state
x0 = np.array([1, 0, 0])

# Simulate the Markov chain
steps = 20
X = np.zeros((steps, 3))
X[0] = x0

for i in range(1, steps):
    X[i] = np.dot(X[i-1], P)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], label='State 0')
plt.plot(X[:, 1], label='State 1')
plt.plot(X[:, 2], label='State 2')
plt.xlabel('Steps')
plt.ylabel('Probability')
plt.title('Markov Chain State Probabilities')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real
stationary /= stationary.sum()

print("Stationary distribution:", stationary.flatten())
```

Slide 14: Additional Resources

For those interested in delving deeper into matrix theory and its applications, here are some valuable resources:

1. "Matrix Analysis" by Roger A. Horn and Charles R. Johnson ([https://arxiv.org/abs/1907.09263](https://arxiv.org/abs/1907.09263))
2. "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III
3. "Introduction to Linear Algebra" by Gilbert Strang (MIT OpenCourseWare)
4. "The Matrix Cookbook" by Kaare Brandt Petersen and Michael Syskind Pedersen ([https://arxiv.org/abs/2111.11176](https://arxiv.org/abs/2111.11176))

These resources provide in-depth explanations, proofs, and advanced applications of matrix theory concepts.

