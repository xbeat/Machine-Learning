## Orthogonal Vectors in Python
Slide 1: Introduction to Orthogonal Vectors

Orthogonal vectors are fundamental concepts in linear algebra and geometry. They are vectors that are perpendicular to each other, forming a right angle. In this presentation, we'll explore orthogonal vectors, their properties, and applications using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(v1, v2):
    plt.quiver([0, 0], [0, 0], [v1[0], v2[0]], [v1[1], v2[1]], scale=1, scale_units='xy', angles='xy')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid()
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()

v1 = np.array([1, 0])
v2 = np.array([0, 1])
plot_vectors(v1, v2)
```

Slide 2: Definition of Orthogonal Vectors

Two vectors are orthogonal if their dot product equals zero. This means they are perpendicular to each other in the geometric sense. Orthogonality is a crucial concept in many areas of mathematics and physics.

```python
def are_orthogonal(v1, v2):
    return np.dot(v1, v2) == 0

v1 = np.array([1, 0])
v2 = np.array([0, 1])

print(f"Are v1 and v2 orthogonal? {are_orthogonal(v1, v2)}")
```

Slide 3: Properties of Orthogonal Vectors

Orthogonal vectors have several important properties. They form a right angle, their dot product is zero, and they are linearly independent. These properties make orthogonal vectors useful in various applications, such as coordinate systems and basis vectors.

```python
def vector_properties(v1, v2):
    dot_product = np.dot(v1, v2)
    angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_degrees = np.degrees(angle)
    return dot_product, angle_degrees

v1 = np.array([1, 0])
v2 = np.array([0, 1])

dot_product, angle = vector_properties(v1, v2)
print(f"Dot product: {dot_product}")
print(f"Angle between vectors: {angle} degrees")
```

Slide 4: Orthogonalization Process

The Gram-Schmidt process is a method to convert a set of linearly independent vectors into a set of orthogonal vectors. This process is crucial in many mathematical and computational applications, such as QR decomposition and solving systems of linear equations.

```python
def gram_schmidt(vectors):
    orthogonalized = []
    for v in vectors:
        w = v - sum(np.dot(v, u) * u for u in orthogonalized)
        orthogonalized.append(w / np.linalg.norm(w))
    return np.array(orthogonalized)

vectors = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])]
orthogonal_vectors = gram_schmidt(vectors)
print("Orthogonalized vectors:")
print(orthogonal_vectors)
```

Slide 5: Verifying Orthogonality

After applying the Gram-Schmidt process, it's important to verify that the resulting vectors are indeed orthogonal. We can do this by computing the dot product between each pair of vectors and checking if it's close to zero (accounting for floating-point precision).

```python
def verify_orthogonality(vectors):
    n = len(vectors)
    for i in range(n):
        for j in range(i+1, n):
            dot_product = np.dot(vectors[i], vectors[j])
            print(f"Dot product of vector {i+1} and vector {j+1}: {dot_product:.2e}")

verify_orthogonality(orthogonal_vectors)
```

Slide 6: Applications in Linear Algebra

Orthogonal vectors play a crucial role in linear algebra. They are used in creating orthonormal bases, which are essential for many matrix decompositions and transformations. Let's create an orthonormal basis from our orthogonal vectors.

```python
def create_orthonormal_basis(vectors):
    return np.array([v / np.linalg.norm(v) for v in vectors])

orthonormal_basis = create_orthonormal_basis(orthogonal_vectors)
print("Orthonormal basis:")
print(orthonormal_basis)

# Verify that the basis vectors are unit vectors
for i, v in enumerate(orthonormal_basis):
    print(f"Norm of vector {i+1}: {np.linalg.norm(v):.6f}")
```

Slide 7: Orthogonal Projections

Orthogonal projections are a fundamental application of orthogonal vectors. They allow us to decompose a vector into components parallel and perpendicular to another vector. This is useful in many areas, including computer graphics and signal processing.

```python
def orthogonal_projection(v, u):
    return (np.dot(v, u) / np.dot(u, u)) * u

v = np.array([3, 2])
u = np.array([1, 0])

proj = orthogonal_projection(v, u)
print(f"Projection of v onto u: {proj}")

# Visualize the projection
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='v')
plt.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='r', label='proj')
plt.quiver(proj[0], proj[1], v[0]-proj[0], v[1]-proj[1], angles='xy', scale_units='xy', scale=1, color='g', label='orthogonal component')
plt.legend()
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.show()
```

Slide 8: Orthogonal Matrices

An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors. These matrices have special properties that make them useful in various applications, such as rotations in 3D space.

```python
def is_orthogonal_matrix(A):
    n = A.shape[0]
    identity = np.eye(n)
    return np.allclose(np.dot(A, A.T), identity) and np.allclose(np.dot(A.T, A), identity)

# Create a 2D rotation matrix
theta = np.pi / 4  # 45-degree rotation
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

print("Rotation matrix:")
print(rotation_matrix)
print(f"Is orthogonal: {is_orthogonal_matrix(rotation_matrix)}")
```

Slide 9: Orthogonal Vectors in Signal Processing

In signal processing, orthogonal vectors are used to represent signals as a combination of basis functions. The Fourier transform, which decomposes a signal into sinusoidal components, is a prime example of this application.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(t):
    return np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t)

t = np.linspace(0, 1, 1000)
signal = generate_signal(t)

plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title("Composite Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Perform Fourier Transform
frequencies = np.fft.fftfreq(len(t), t[1] - t[0])
fft_values = np.fft.fft(signal)

plt.figure(figsize=(10, 4))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_values)[:len(frequencies)//2])
plt.title("Frequency Spectrum")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
```

Slide 10: Orthogonal Vectors in Machine Learning

In machine learning, orthogonal vectors are used in various techniques, such as Principal Component Analysis (PCA) for dimensionality reduction. PCA finds orthogonal vectors (principal components) that capture the most variance in the data.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot original data and principal components
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Original Data")
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.title("Data after PCA")

for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    comp = comp * var  # scale component by its variance explanation power
    plt.arrow(0, 0, comp[0], comp[1], color='r', alpha=0.8, width=0.02,
              head_width=0.05, head_length=0.05, label=f'PC{i+1}')

plt.legend()
plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 11: Orthogonal Vectors in Computer Graphics

In computer graphics, orthogonal vectors are essential for defining coordinate systems, camera orientations, and transformations. They are particularly useful in creating viewing matrices for 3D scenes.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_orthonormal_basis(forward):
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.array([0, 1, 0]), forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward

# Define camera position and target
camera_pos = np.array([1, 2, 3])
target_pos = np.array([0, 0, 0])

# Create orthonormal basis
forward = target_pos - camera_pos
right, up, forward = create_orthonormal_basis(forward)

# Plot the camera coordinate system
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], right[0], right[1], right[2], color='r', length=0.5, label='Right')
ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], up[0], up[1], up[2], color='g', length=0.5, label='Up')
ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2], forward[0], forward[1], forward[2], color='b', length=0.5, label='Forward')

ax.set_xlim(-1, 2)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Camera Coordinate System')

plt.show()
```

Slide 12: Orthogonal Vectors in Quantum Mechanics

In quantum mechanics, orthogonal vectors represent mutually exclusive quantum states. The concept of orthogonality is crucial in understanding superposition and measurement in quantum systems.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_bloch_sphere(state_vector):
    theta = np.arccos(np.abs(state_vector[1]))
    phi = np.angle(state_vector[0]) - np.angle(state_vector[1])
    
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    w = np.cos(theta)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Bloch sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
    
    # Plot state vector
    ax.quiver(0, 0, 0, u, v, w, color='r', linewidth=3, arrow_length_ratio=0.15)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Bloch Sphere Representation")
    plt.show()

# Define two orthogonal quantum states
state_0 = np.array([1, 0], dtype=complex)
state_1 = np.array([0, 1], dtype=complex)

print("Dot product of states:", np.dot(state_0, state_1))

# Plot state_0 on Bloch sphere
plot_bloch_sphere(state_0)
```

Slide 13: Conclusion and Real-world Applications

Orthogonal vectors are not just abstract mathematical concepts but have numerous practical applications in various fields:

1. Signal Processing: Used in noise reduction, data compression, and signal separation.
2. Computer Graphics: Essential for 3D transformations, camera positioning, and rendering.
3. Machine Learning: Employed in dimensionality reduction techniques like PCA and feature extraction.
4. Quantum Computing: Fundamental in representing quantum states and quantum operations.
5. Control Systems: Used in designing stable control systems and optimizing system performance.

Understanding and applying orthogonal vectors is crucial for advancing technology and solving complex problems in these fields.

Slide 14: Additional Resources

For those interested in delving deeper into orthogonal vectors and their applications, here are some valuable resources:

1. ArXiv paper on applications of orthogonal vectors in quantum computing: "Quantum Computation with Orthogonal Matrix-Vector Multiplication" ArXiv URL: [https://arxiv.org/abs/2006.08924](https://arxiv.org/abs/2006.08924)
2. ArXiv paper on orthogonal vectors in machine learning: "Orthogonal Machine Learning: Power and Limitations" ArXiv URL: [https://arxiv.org/abs/1905.01665](https://arxiv.org/abs/1905.01665)
3. ArXiv paper on orthogonal vectors in signal processing: "Orthogonal Vector Approach for Synthesis of Multi-Beam Directional Modulation Transmitters" ArXiv URL: [https://arxiv.org/abs/1502.00387](https://arxiv.org/abs/1502.00387)

These papers provide in-depth discussions and advanced applications of orthogonal vectors in various fields.

