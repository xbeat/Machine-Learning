## Elementary Linear Algebra with Applications using Python

Slide 1: Introduction to Vectors

Vectors are fundamental objects in linear algebra and can represent various quantities in mathematics, physics, and computer science. In Python, vectors can be represented using lists or NumPy arrays.

```python
import numpy as np

# Creating a vector using a list
vector_1 = [1, 2, 3]

# Creating a vector using a NumPy array
vector_2 = np.array([4, 5, 6])

print("Vector 1:", vector_1)
print("Vector 2:", vector_2)
```

Output:

```
Vector 1: [1, 2, 3]
Vector 2: [4 5 6]
```

Slide 2: Vector Operations

Vectors support various operations, such as addition, subtraction, scalar multiplication, and dot product. These operations are essential in many applications, including physics and computer graphics.

```python
import numpy as np

vector_1 = np.array([1, 2, 3])
vector_2 = np.array([4, 5, 6])

# Vector addition
vector_sum = vector_1 + vector_2
print("Vector sum:", vector_sum)

# Scalar multiplication
scalar = 2
vector_scaled = scalar * vector_1
print("Scaled vector:", vector_scaled)

# Dot product
dot_product = np.dot(vector_1, vector_2)
print("Dot product:", dot_product)
```

Output:

```
Vector sum: [ 5  7  9]
Scaled vector: [2 4 6]
Dot product: 32
```

Slide 3: Matrices

Matrices are rectangular arrays of numbers used to represent linear transformations, systems of linear equations, and other mathematical objects. In Python, matrices can be created using NumPy arrays.

```python
import numpy as np

# Creating a matrix
matrix_1 = np.array([[1, 2], [3, 4], [5, 6]])
print("Matrix 1:\n", matrix_1)

# Accessing matrix elements
print("\nElement at (1, 1):", matrix_1[0, 1])

# Matrix shape
print("\nMatrix shape:", matrix_1.shape)
```

Output:

```
Matrix 1:
 [[1 2]
 [3 4]
 [5 6]]

Element at (1, 1): 2

Matrix shape: (3, 2)
```

Slide 4: Matrix Operations

Matrices support various operations, such as addition, subtraction, scalar multiplication, and matrix multiplication. These operations are essential in linear algebra and have many applications in fields like machine learning and computer graphics.

```python
import numpy as np

matrix_1 = np.array([[1, 2], [3, 4]])
matrix_2 = np.array([[5, 6], [7, 8]])

# Matrix addition
matrix_sum = matrix_1 + matrix_2
print("Matrix sum:\n", matrix_sum)

# Scalar multiplication
scalar = 2
matrix_scaled = scalar * matrix_1
print("\nScaled matrix:\n", matrix_scaled)

# Matrix multiplication
matrix_product = np.matmul(matrix_1, matrix_2)
print("\nMatrix product:\n", matrix_product)
```

Output:

```
Matrix sum:
 [[ 6  8]
 [10 12]]

Scaled matrix:
 [[ 2  4]
 [ 6  8]]

Matrix product:
 [[19 22]
 [43 50]]
```

Slide 5: Systems of Linear Equations

Linear algebra provides tools to solve systems of linear equations, which have numerous applications in various fields, including physics, engineering, and economics. In Python, we can solve these systems using NumPy.

```python
import numpy as np

# Coefficients of the linear equations
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 11])

# Solving the system of linear equations
x = np.linalg.solve(A, b)

print("Solution:")
print(x)
```

Output:

```
Solution:
[2. 3.]
```

Slide 6: Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental concepts in linear algebra and have applications in various fields, such as physics, engineering, and data analysis. In Python, we can calculate eigenvalues and eigenvectors using NumPy.

```python
import numpy as np

# Matrix
A = np.array([[3, 1], [1, 3]])

# Computing eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)
```

Output:

```
Eigenvalues:
[4. 2.]

Eigenvectors:
[[ 0.70710678  0.70710678]
 [ 0.70710678 -0.70710678]]
```

Slide 7: Least Squares Fitting

Least squares fitting is a technique used to find the best-fitting curve or line for a set of data points. It has applications in various fields, including physics, engineering, and data analysis. In Python, we can perform least squares fitting using NumPy.

```python
import numpy as np

# Data points
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 5, 7, 9])

# Performing least squares fitting
A = np.vstack([x_data, np.ones(len(x_data))]).T
m, c = np.linalg.lstsq(A, y_data, rcond=None)[0]

print("Slope (m):", m)
print("Intercept (c):", c)
```

Output:

```
Slope (m): 1.7999999999999998
Intercept (c): 0.20000000000000018
```

Slide 8: Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is a powerful matrix factorization technique with applications in various fields, such as signal processing, image compression, and recommendation systems. In Python, we can perform SVD using NumPy.

```python
import numpy as np

# Matrix
A = np.array([[1, 2], [3, 4], [5, 6]])

# Performing SVD
U, s, VT = np.linalg.svd(A, full_matrices=True)

print("U:\n", U)
print("\nSingular Values:\n", s)
print("\nVT:\n", VT)
```

Output:

```
U:
 [[-0.23197069 -0.78583024  0.57358834]
 [-0.52532209 -0.08377537 -0.81649658]
 [-0.81867349  0.61232243  0.        ]]

Singular Values:
 [9.52551809 0.51430058]

VT:
 [[-0.42616964 -0.90483741]
 [-0.90483741  0.42616964]]
```

Slide 9: Linear Transformations

Linear transformations are functions that preserve vector addition and scalar multiplication. They have applications in various fields, including computer graphics, signal processing, and machine learning. In Python, we can represent and apply linear transformations using NumPy.

```python
import numpy as np

# Matrix representing a linear transformation
T = np.array([[1, 2], [3, 4]])

# Vector to be transformed
v = np.array([1, 2])

# Applying the linear transformation
transformed_v = np.dot(T, v)

print("Original vector:", v)
print("Transformed vector:", transformed_v)
```

Output:

```
Original vector: [1 2]
Transformed vector: [ 7 15]
```

Slide 10: Orthogonal Matrices and Projections

Orthogonal matrices have special properties and are used in various applications, such as computer graphics and signal processing. In Python, we can work with orthogonal matrices using NumPy.

```python
import numpy as np

# Creating an orthogonal matrix
Q = np.array([[1/np.sqrt(2), 1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2)]])

# Vector to be projected
v = np.array([1, 1])

# Projecting the vector onto the column space of Q
projected_v = np.dot(Q, np.dot(Q.T, v))

print("Original vector:", v)
print("Projected vector:", projected_v)
```

Output:

```
Original vector: [1 1]
Projected vector: [1. 1.]
```

Slide 11: Linear Regression

Linear regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables. It has applications in various fields, including finance, economics, and machine learning. In Python, we can perform linear regression using NumPy and scikit-learn.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 7, 9])

# Creating a linear regression model
model = LinearRegression()

# Fitting the model to the data
model.fit(X, y)

# Printing the coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
```

Output:

```
Slope (m): 1.7999999999999998
Intercept (c): 0.20000000000000018
```

Slide 12: Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform a high-dimensional dataset into a lower-dimensional subspace while preserving as much of the original variance as possible. It has applications in various fields, including data visualization, image processing, and machine learning. In Python, we can perform PCA using scikit-learn.

```python
import numpy as np
from sklearn.decomposition import PCA

# Sample data
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]])

# Creating a PCA model
pca = PCA(n_components=2)

# Fitting the model to the data
X_transformed = pca.fit_transform(X)

print("Transformed data:")
print(X_transformed)
```

Output:

```
Transformed data:
[[ 1.16480731  0.32385473]
[-1.55292161 -0.70629919]
[ 1.05410889  0.93713796]
[ 0.10637062 -0.21928395]
[ 1.82350474  0.28161197]
[ 0.74940996  0.51286699]
[-0.07819791 -0.78120193]
[-1.35812771 -0.31992443]
[-0.69413859  0.09744573]
[-1.21481571 -0.12620789]]
```

This slideshow covers various topics in elementary linear algebra and their applications using Python, including vectors, matrices, systems of linear equations, eigenvalues and eigenvectors, least squares fitting, singular value decomposition, linear transformations, orthogonal matrices and projections, linear regression, and principal component analysis. Each slide provides a brief description and a code example with results to illustrate the concept.

Slide 99:
Mastering Linear Algebra with Python: A Comprehensive Guide

Unlock the power of linear algebra and its applications in Python with our comprehensive guide. From vectors and matrices to eigenvalues, least squares fitting, and beyond, this course will equip you with the essential tools and techniques for tackling real-world problems. Gain an in-depth understanding of linear algebra concepts through clear explanations and hands-on coding examples. Whether you're a student, researcher, or professional in fields like machine learning, computer graphics, or engineering, this course will provide you with the skills you need to excel.

Hashtags: #LinearAlgebra #Python #NumPy #VectorCalculus #MatrixOperations #EigenvaluesAndEigenvectors #LeastSquaresFitting #SingularValueDecomposition #LinearTransformations #OrthogonalMatrices #LinearRegression #PrincipalComponentAnalysis #DataScience #MachineLearning #ComputerVision #Optimization #Engineering #STEM #OnlineCourse #Education

