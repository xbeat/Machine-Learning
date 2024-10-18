## Linear Algebra and Affine Projective Geometry
Slide 1: Vector Spaces in Linear Algebra

Vector spaces form the foundation of linear algebra. They are sets of vectors that can be added together and multiplied by scalars. Let's explore a simple vector space using Python.

```python
import numpy as np

# Define two vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
v_sum = v1 + v2
print("Vector sum:", v_sum)

# Scalar multiplication
scalar = 2
v_scaled = scalar * v1
print("Scaled vector:", v_scaled)

# Check if vectors are in the same space
def are_in_same_space(v1, v2):
    return len(v1) == len(v2)

print("Vectors in same space:", are_in_same_space(v1, v2))
```

Slide 2: Matrices in Linear Algebra

Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are fundamental in representing and solving systems of linear equations.

```python
import numpy as np

# Create a matrix
A = np.array([[1, 2], [3, 4]])
print("Matrix A:\n", A)

# Matrix multiplication
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
print("Matrix multiplication result:\n", C)

# Matrix transpose
A_transpose = A.T
print("Transpose of A:\n", A_transpose)

# Matrix determinant
det_A = np.linalg.det(A)
print("Determinant of A:", det_A)
```

Slide 3: Linear Maps

Linear maps are functions between vector spaces that preserve vector addition and scalar multiplication. They can be represented by matrices.

```python
import numpy as np

def linear_map(matrix, vector):
    return np.dot(matrix, vector)

# Define a linear map as a matrix
A = np.array([[2, 1], [1, 3]])

# Apply the linear map to a vector
v = np.array([1, 2])
result = linear_map(A, v)

print("Original vector:", v)
print("Result of linear map:", result)

# Verify linearity properties
scalar = 2
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Property 1: f(av) = af(v)
prop1_left = linear_map(A, scalar * v1)
prop1_right = scalar * linear_map(A, v1)
print("f(av) == af(v):", np.allclose(prop1_left, prop1_right))

# Property 2: f(u + v) = f(u) + f(v)
prop2_left = linear_map(A, v1 + v2)
prop2_right = linear_map(A, v1) + linear_map(A, v2)
print("f(u + v) == f(u) + f(v):", np.allclose(prop2_left, prop2_right))
```

Slide 4: Affine Geometry Basics

Affine geometry extends vector spaces by including translations. An affine space consists of points and free vectors, allowing for operations like translation and barycentric combinations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define points in 2D affine space
A = np.array([1, 1])
B = np.array([4, 5])

# Translation vector
v = np.array([2, 3])

# Translate point A by vector v
C = A + v

# Plot the points and vector
plt.figure(figsize=(8, 6))
plt.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], c=['r', 'g', 'b'])
plt.annotate('A', A)
plt.annotate('B', B)
plt.annotate('C', C)
plt.arrow(A[0], A[1], v[0], v[1], head_width=0.2, head_length=0.3, fc='k', ec='k')
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.grid(True)
plt.title('Affine Transformation: Translation')
plt.show()
```

Slide 5: Projective Geometry Fundamentals

Projective geometry extends affine geometry by adding points at infinity. It's crucial in computer vision and graphics for handling perspective transformations.

```python
import numpy as np
import matplotlib.pyplot as plt

def to_homogeneous(points):
    return np.column_stack((points, np.ones(len(points))))

def from_homogeneous(points):
    return points[:, :-1] / points[:, -1:]

# Define points in 2D
points = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])

# Convert to homogeneous coordinates
homogeneous_points = to_homogeneous(points)

# Define a projective transformation matrix
H = np.array([[1, -0.5, 1],
              [0.5, 1, 0.5],
              [0.1, 0.1, 1]])

# Apply the transformation
transformed_points = np.dot(homogeneous_points, H.T)

# Convert back to 2D
result_points = from_homogeneous(transformed_points)

# Plotting
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(points[:, 0], points[:, 1])
plt.title('Original Points')
plt.grid(True)

plt.subplot(122)
plt.scatter(result_points[:, 0], result_points[:, 1])
plt.title('Transformed Points')
plt.grid(True)

plt.tight_layout()
plt.show()
```

Slide 6: Bilinear Forms in Geometry

Bilinear forms are functions of two variables that are linear in each variable separately. They're essential in defining inner products and quadratic forms.

```python
import numpy as np

def bilinear_form(x, y, A):
    return np.dot(np.dot(x, A), y)

# Define a symmetric matrix A for the bilinear form
A = np.array([[2, 1], [1, 3]])

# Define two vectors
x = np.array([1, 2])
y = np.array([3, 4])

# Compute the bilinear form
result = bilinear_form(x, y, A)

print(f"Bilinear form B(x, y) = x^T A y = {result}")

# Verify symmetry property: B(x, y) = B(y, x)
result_symmetric = bilinear_form(y, x, A)
print(f"B(y, x) = {result_symmetric}")
print(f"Symmetry property holds: {np.isclose(result, result_symmetric)}")

# Verify linearity in first argument
z = np.array([5, 6])
left_side = bilinear_form(x + z, y, A)
right_side = bilinear_form(x, y, A) + bilinear_form(z, y, A)
print(f"Linearity in first argument holds: {np.isclose(left_side, right_side)}")
```

Slide 7: Polynomials in Algebra

Polynomials are expressions consisting of variables and coefficients. They play a crucial role in various areas of mathematics and computer science.

```python
import numpy as np
import matplotlib.pyplot as plt

def polynomial(x, coeffs):
    return sum(coeff * x**power for power, coeff in enumerate(coeffs))

# Define a polynomial: 2x^3 - 3x^2 + 4x - 1
coeffs = [-1, 4, -3, 2]

# Generate x values
x = np.linspace(-2, 2, 100)

# Compute y values
y = polynomial(x, coeffs)

# Plot the polynomial
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Polynomial: 2x^3 - 3x^2 + 4x - 1')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()

# Find roots of the polynomial
roots = np.roots(coeffs[::-1])
print("Roots of the polynomial:", roots)
```

Slide 8: Ideals in Algebra

An ideal is a special subset of a ring that absorbs multiplication by ring elements. They're crucial in abstract algebra and algebraic geometry.

```python
class Polynomial:
    def __init__(self, coeffs):
        self.coeffs = coeffs
    
    def __add__(self, other):
        max_degree = max(len(self.coeffs), len(other.coeffs))
        new_coeffs = [0] * max_degree
        for i in range(max_degree):
            if i < len(self.coeffs):
                new_coeffs[i] += self.coeffs[i]
            if i < len(other.coeffs):
                new_coeffs[i] += other.coeffs[i]
        return Polynomial(new_coeffs)
    
    def __mul__(self, other):
        new_coeffs = [0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i in range(len(self.coeffs)):
            for j in range(len(other.coeffs)):
                new_coeffs[i+j] += self.coeffs[i] * other.coeffs[j]
        return Polynomial(new_coeffs)
    
    def __repr__(self):
        return f"Polynomial({self.coeffs})"

# Define polynomials
f = Polynomial([1, 0, 1])  # x^2 + 1
g = Polynomial([1, 1])     # x + 1

# Generate ideal elements
ideal_element1 = f * Polynomial([2, 3])  # 2f + 3xf
ideal_element2 = g * Polynomial([1, 1, 1])  # g + xg + x^2g

print("Ideal element 1:", ideal_element1)
print("Ideal element 2:", ideal_element2)

# Verify closure under addition
sum_element = ideal_element1 + ideal_element2
print("Sum of ideal elements:", sum_element)
```

Slide 9: Tensor Algebra Basics

Tensor algebra extends vector algebra to multilinear maps. It's fundamental in physics, engineering, and machine learning for representing complex data relationships.

```python
import numpy as np

# Define two vectors
v1 = np.array([1, 2])
v2 = np.array([3, 4])

# Tensor product (outer product) of two vectors
tensor_product = np.outer(v1, v2)
print("Tensor product of v1 and v2:\n", tensor_product)

# Define a matrix (2nd order tensor)
A = np.array([[1, 2], [3, 4]])

# Tensor contraction (trace of a matrix)
contraction = np.trace(A)
print("Contraction of A:", contraction)

# Tensor addition
B = np.array([[5, 6], [7, 8]])
tensor_sum = A + B
print("Tensor sum of A and B:\n", tensor_sum)

# Tensor multiplication (matrix multiplication)
tensor_mult = np.dot(A, B)
print("Tensor multiplication of A and B:\n", tensor_mult)
```

Slide 10: Metric Spaces in Topology

A metric space is a set where a notion of distance between elements is defined. It's fundamental in analysis and topology.

```python
import numpy as np
import matplotlib.pyplot as plt

class MetricSpace:
    def distance(self, x, y):
        raise NotImplementedError("Subclass must implement abstract method")

class EuclideanSpace(MetricSpace):
    def distance(self, x, y):
        return np.linalg.norm(np.array(x) - np.array(y))

class ManhattanSpace(MetricSpace):
    def distance(self, x, y):
        return np.sum(np.abs(np.array(x) - np.array(y)))

# Create points
points = [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0)]

# Calculate distances
euclidean = EuclideanSpace()
manhattan = ManhattanSpace()

for i in range(len(points)):
    for j in range(i+1, len(points)):
        print(f"Distance between {points[i]} and {points[j]}:")
        print(f"  Euclidean: {euclidean.distance(points[i], points[j]):.2f}")
        print(f"  Manhattan: {manhattan.distance(points[i], points[j]):.2f}")

# Visualize points
plt.figure(figsize=(10, 5))
x, y = zip(*points)
plt.scatter(x, y)
for i, point in enumerate(points):
    plt.annotate(f'P{i}', (point[0], point[1]))
plt.title('Points in 2D Space')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
```

Slide 11: Derivatives in Differential Calculus

Derivatives measure the rate of change of a function. They're crucial in optimization, physics, and many areas of applied mathematics.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 4

def df(x):
    return 2*x - 4

# Generate x values
x = np.linspace(0, 6, 100)

# Compute y values for f(x) and f'(x)
y = f(x)
dy = df(x)

# Plot f(x) and f'(x)
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='f(x) = x^2 - 4x + 4')
plt.plot(x, dy, label="f'(x) = 2x - 4")
plt.title('Function and its Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Mark the minimum point
min_x = 2  # The minimum occurs at x = 2
plt.plot(min_x, f(min_x), 'ro', markersize=10)
plt.annotate('Minimum', xy=(min_x, f(min_x)), xytext=(min_x+0.5, f(min_x)+1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# Numerical approximation of derivative
h = 0.0001
x0 = 2
numerical_derivative = (f(x0 + h) - f(x0)) / h
print(f"Numerical derivative at x = {x0}: {numerical_derivative}")
print(f"Analytical derivative at x = {x0}: {df(x0)}")
```

Slide 12: Extrema in Optimization Theory

Finding extrema (minima and maxima) of functions is central to optimization theory, crucial in machine learning and data analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f(x):
    return x**4 - 4*x**2 + 2

x = np.linspace(-3, 3, 200)
y = f(x)

result = minimize_scalar(f)
min_x, min_y = result.x, result.fun

plt.figure(figsize=(12, 6))
plt.plot(x, y, label='f(x) = x^4 - 4x^2 + 2')
plt.scatter([min_x], [min_y], color='red', s=100, label='Global Minimum')
plt.title('Function with Global Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

print(f"Global minimum: f({min_x:.4f}) = {min_y:.4f}")
```

Slide 13: Linear Optimization

Linear optimization, or linear programming, involves optimizing a linear objective function subject to linear constraints. It's widely used in resource allocation and decision-making problems.

```python
import numpy as np
from scipy.optimize import linprog

# Objective function coefficients
c = [-1, -2]  # Maximize 1x + 2y (equivalent to minimizing -1x - 2y)

# Inequality constraints (Ax <= b)
A = [[2, 1],  # 2x + y <= 20
     [1, 3],  # x + 3y <= 30
     [-1, 0],  # -x <= 0 (i.e., x >= 0)
     [0, -1]]  # -y <= 0 (i.e., y >= 0)
b = [20, 30, 0, 0]

# Solve the linear programming problem
result = linprog(c, A_ub=A, b_ub=b)

print("Optimal solution:")
print(f"x = {result.x[0]:.2f}")
print(f"y = {result.x[1]:.2f}")
print(f"Optimal value = {-result.fun:.2f}")  # Negate because we maximized
```

Slide 14: Nonlinear Optimization

Nonlinear optimization deals with problems where the objective function or constraints are nonlinear. These problems are common in machine learning, engineering, and economics.

```python
import numpy as np
from scipy.optimize import minimize

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def constraint(x):
    return x[0]**2 + x[1]**2 - 5

# Initial guess
x0 = [0, 0]

# Define constraints
cons = {'type': 'ineq', 'fun': constraint}

# Solve the nonlinear optimization problem
result = minimize(objective, x0, method='SLSQP', constraints=cons)

print("Optimal solution:")
print(f"x = {result.x[0]:.4f}")
print(f"y = {result.x[1]:.4f}")
print(f"Optimal value = {result.fun:.4f}")
```

Slide 15: Support Vector Machines in Machine Learning

Support Vector Machines (SVM) are powerful classifiers that find the hyperplane maximizing the margin between classes. They utilize concepts from linear algebra and optimization.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Load iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # Use sepal length and petal length
y = iris.target

# Create SVM classifier
svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(X, y)

# Create mesh to plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict for each point in the mesh
Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.title('SVM Decision Boundary on Iris Dataset')
plt.show()
```

Slide 16: Additional Resources

For further exploration of the topics covered in this presentation, consider the following resources:

1. "Introduction to Linear Algebra" by Gilbert Strang (MIT OpenCourseWare)
2. "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe
3. "Pattern Recognition and Machine Learning" by Christopher Bishop

ArXiv papers for advanced topics:

* "A Tutorial on Support Vector Machines for Pattern Recognition" by Christopher J.C. Burges ArXiv: [https://arxiv.org/abs/cs/9805019](https://arxiv.org/abs/cs/9805019)
* "An Introduction to Tensor Calculus" by Taha Sochi ArXiv: [https://arxiv.org/abs/1603.01660](https://arxiv.org/abs/1603.01660)

These resources provide in-depth coverage of the mathematical foundations crucial for advanced machine learning and data science.

