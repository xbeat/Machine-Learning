## Understanding Vectors in Linear Algebra with Python
Slide 1: Introduction to Vectors

Vectors are fundamental elements in linear algebra, representing quantities with both magnitude and direction. They form the backbone of many mathematical and computational applications, from physics simulations to machine learning algorithms.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 2D vector
vector = np.array([3, 4])

# Plot the vector
plt.figure(figsize=(8, 6))
plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='r')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid()
plt.axis('equal')
plt.title('A 2D Vector')
plt.show()
```

Slide 2: Vector Operations: Addition and Subtraction

Vector addition and subtraction are performed element-wise, maintaining the dimensionality of the original vectors. These operations are crucial for combining or finding differences between vector quantities.

```python
import numpy as np

# Define two vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
v_sum = v1 + v2
print("Sum:", v_sum)

# Vector subtraction
v_diff = v1 - v2
print("Difference:", v_diff)
```

Slide 3: Scalar Multiplication

Scalar multiplication involves multiplying a vector by a scalar (a single number). This operation changes the magnitude of the vector while preserving its direction (unless the scalar is negative).

```python
import numpy as np

# Define a vector and a scalar
v = np.array([1, 2, 3])
scalar = 2

# Scalar multiplication
v_scaled = scalar * v
print("Original vector:", v)
print("Scaled vector:", v_scaled)

# Visualize the effect
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Original')
plt.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1, color='r', label='Scaled')
plt.legend()
plt.axis('equal')
plt.grid()
plt.title('Scalar Multiplication')
plt.show()
```

Slide 4: Dot Product

The dot product is a crucial operation that returns a scalar value representing the projection of one vector onto another. It's widely used in calculating angles between vectors and in many machine learning algorithms.

```python
import numpy as np

# Define two vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Calculate dot product
dot_product = np.dot(v1, v2)
print("Dot product:", dot_product)

# Calculate angle between vectors
angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
print("Angle between vectors (in radians):", angle)
```

Slide 5: Cross Product

The cross product is an operation on two vectors in three-dimensional space, resulting in a vector perpendicular to both input vectors. It's extensively used in physics and computer graphics.

```python
import numpy as np

# Define two 3D vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Calculate cross product
cross_product = np.cross(v1, v2)
print("Cross product:", cross_product)

# Visualize the vectors and their cross product
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2')
ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2], color='g', label='v1 x v2')

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.legend()
ax.set_title('Cross Product Visualization')
plt.show()
```

Slide 6: Vector Normalization

Normalization is the process of scaling a vector to have a magnitude of 1, creating a unit vector. This is often used to standardize vector directions or to prepare data for machine learning algorithms.

```python
import numpy as np

# Define a vector
v = np.array([3, 4])

# Calculate the magnitude (length) of the vector
magnitude = np.linalg.norm(v)
print("Magnitude:", magnitude)

# Normalize the vector
v_normalized = v / magnitude
print("Normalized vector:", v_normalized)

# Verify that the normalized vector has a magnitude of 1
print("Magnitude of normalized vector:", np.linalg.norm(v_normalized))

# Visualize original and normalized vectors
plt.figure(figsize=(8, 6))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Original')
plt.quiver(0, 0, v_normalized[0], v_normalized[1], angles='xy', scale_units='xy', scale=1, color='r', label='Normalized')
plt.legend()
plt.axis('equal')
plt.grid()
plt.title('Vector Normalization')
plt.show()
```

Slide 7: Vector Projection

Vector projection is the process of finding the component of one vector that is parallel to another vector. This concept is crucial in many applications, including computer graphics and signal processing.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define two vectors
v = np.array([3, 4])
u = np.array([1, 2])

# Calculate the projection of v onto u
proj_v_on_u = (np.dot(v, u) / np.dot(u, u)) * u

# Visualize the vectors and projection
plt.figure(figsize=(8, 6))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='v')
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='r', label='u')
plt.quiver(0, 0, proj_v_on_u[0], proj_v_on_u[1], angles='xy', scale_units='xy', scale=1, color='g', label='proj_v_on_u')
plt.legend()
plt.axis('equal')
plt.grid()
plt.title('Vector Projection')
plt.show()

print("Projection of v onto u:", proj_v_on_u)
```

Slide 8: Linear Independence and Span

Linear independence is a key concept in understanding vector spaces. A set of vectors is linearly independent if none of them can be expressed as a linear combination of the others. The span of a set of vectors is the set of all possible linear combinations of those vectors.

```python
import numpy as np

# Define a set of vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])

# Check for linear independence
matrix = np.column_stack((v1, v2, v3))
rank = np.linalg.matrix_rank(matrix)

print("Rank of the matrix:", rank)
print("Number of vectors:", matrix.shape[1])

if rank == matrix.shape[1]:
    print("The vectors are linearly independent.")
else:
    print("The vectors are linearly dependent.")

# Visualize the span of v1 and v2
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='b', label='v2')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Span of v1 and v2')
plt.show()
```

Slide 9: Basis and Dimension

A basis is a set of linearly independent vectors that span a vector space. The number of vectors in a basis determines the dimension of the space. Understanding these concepts is crucial for working with vector spaces and linear transformations.

```python
import numpy as np

# Define a set of vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

# Create a matrix from these vectors
matrix = np.column_stack((v1, v2, v3))

# Check if the vectors form a basis
rank = np.linalg.matrix_rank(matrix)
dimension = matrix.shape[1]

print("Rank of the matrix:", rank)
print("Number of vectors:", dimension)

if rank == dimension:
    print("These vectors form a basis.")
    print("The dimension of the space is:", dimension)
else:
    print("These vectors do not form a basis.")

# Visualize the basis vectors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='r', label='v1')
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='g', label='v2')
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='b', label='v3')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.legend()
ax.set_title('Basis Vectors of 3D Space')
plt.show()
```

Slide 10: Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental concepts in linear algebra, particularly important in understanding linear transformations. An eigenvector of a linear transformation is a vector that, when the transformation is applied, changes only by a scalar factor (the eigenvalue).

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix
A = np.array([[3, 1],
              [1, 2]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)

# Visualize the eigenvectors
plt.figure(figsize=(8, 8))
for i in range(2):
    plt.quiver(0, 0, eigenvectors[0, i], eigenvectors[1, i], 
               angles='xy', scale_units='xy', scale=1, 
               color=['r', 'b'][i], label=f'Eigenvector {i+1}')

# Plot some random vectors and their transformations
for _ in range(10):
    v = np.random.randn(2)
    v_transformed = A @ v
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.3)
    plt.quiver(0, 0, v_transformed[0], v_transformed[1], angles='xy', scale_units='xy', scale=1, color='green', alpha=0.3)

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.legend()
plt.title('Eigenvectors and Linear Transformation')
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 11: Vector Spaces and Subspaces

A vector space is a set of vectors that is closed under vector addition and scalar multiplication. A subspace is a subset of a vector space that is itself a vector space. Understanding these concepts is crucial for working with more advanced topics in linear algebra.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2D vector space
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)

# Define a subspace (in this case, a line through the origin)
m = 2  # slope of the line
subspace_y = m * x

# Visualize the vector space and subspace
plt.figure(figsize=(10, 8))
plt.plot(X, Y, 'bo', alpha=0.1, label='Vector Space')
plt.plot(x, subspace_y, 'r-', linewidth=2, label='Subspace')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.legend()
plt.title('2D Vector Space and a Subspace')
plt.grid(True)
plt.axis('equal')
plt.show()

# Verify that the subspace is closed under vector addition and scalar multiplication
v1 = np.array([1, m])
v2 = np.array([2, 2*m])
scalar = 3

sum_vector = v1 + v2
scaled_vector = scalar * v1

print("v1:", v1)
print("v2:", v2)
print("v1 + v2:", sum_vector)
print("3 * v1:", scaled_vector)
print("All these vectors lie on the subspace (line).")
```

Slide 12: Real-life Example: Image Processing

Vectors play a crucial role in image processing. Each pixel in an image can be represented as a vector of color intensities. We can use vector operations to manipulate images, such as adjusting brightness or applying filters.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 grayscale image
image = np.array([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.2, 0.3, 0.4, 0.5, 0.6],
    [0.3, 0.4, 0.5, 0.6, 0.7],
    [0.4, 0.5, 0.6, 0.7, 0.8],
    [0.5, 0.6, 0.7, 0.8, 0.9]
])

# Display the original image
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# Increase brightness (scalar multiplication)
bright_image = 1.5 * image
plt.subplot(132)
plt.imshow(bright_image, cmap='gray')
plt.title('Increased Brightness')

# Apply a simple filter (vector addition)
filter_kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9
filtered_image = np.zeros_like(image)
for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        filtered_image[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * filter_kernel)

plt.subplot(133)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Physics Simulation

Vectors are fundamental in physics for representing quantities like position, velocity, and force. Here's a simple simulation of projectile motion using vectors.

```python
import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
initial_position = np.array([0, 0])  # m
initial_velocity = np.array([10, 15])  # m/s
gravity = np.array([0, -9.8])  # m/s^2
time_step = 0.1  # s
total_time = 3  # s

# Simulation
time = np.arange(0, total_time, time_step)
positions = []
position = initial_position

for t in time:
    positions.append(position)
    velocity = initial_velocity + gravity * t
    position = initial_position + initial_velocity * t + 0.5 * gravity * t**2

positions = np.array(positions)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(positions[:, 0], positions[:, 1])
plt.title('Projectile Motion')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Distance (m)')
plt.grid(True)
plt.axis('equal')
plt.show()

print(f"Final position: {positions[-1]} m")
print(f"Distance traveled: {np.linalg.norm(positions[-1] - initial_position):.2f} m")
```

Slide 14: Vector Applications in Machine Learning

Vectors are essential in machine learning, particularly in representing features and implementing algorithms. Here's a simple example of using vectors for k-Nearest Neighbors (k-NN) classification.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 points with 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple classification rule

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.7, s=50)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', marker='s', s=50, linewidth=2, edgecolor='black')
plt.title('k-NN Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(f"Accuracy: {knn.score(X_test, y_test):.2f}")
```

Slide 15: Additional Resources

For further exploration of vectors in linear algebra and their applications:

1. ArXiv paper: "A Survey of Computational Linear Algebra Methods" by Demmel et al. ([https://arxiv.org/abs/2002.12890](https://arxiv.org/abs/2002.12890))
2. ArXiv paper: "Vector Representations of Graphs: A Survey" by Khosla et al. ([https://arxiv.org/abs/2101.00218](https://arxiv.org/abs/2101.00218))
3. Online courses on linear algebra and its applications in machine learning and data science
4. Python libraries for linear algebra: NumPy, SciPy, and scikit-learn documentation
5. Textbooks on linear algebra and its applications in computer science and engineering

