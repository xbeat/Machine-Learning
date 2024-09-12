## Principal Geodesic Analysis with Python
Slide 1: Introduction to Principal Geodesic Analysis (PGA)

Principal Geodesic Analysis is an extension of Principal Component Analysis (PCA) for data lying on a curved manifold. It aims to find the principal directions of variation in the data while respecting the geometry of the manifold.

```python
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere

# Create a hypersphere
sphere = Hypersphere(dim=2)

# Generate random points on the sphere
n_points = 100
data = sphere.random_uniform(n_points)

# Visualize the data
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
ax.set_title('Random points on a 2-sphere')
plt.show()
```

Slide 2: Manifolds and Tangent Spaces

In PGA, we work with data on manifolds, which are curved spaces that locally resemble Euclidean space. The tangent space at a point on the manifold is a flat approximation of the manifold at that point.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_surface(u, v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z

# Generate sphere surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x, y, z = sphere_surface(u[:, np.newaxis], v[np.newaxis, :])

# Plot sphere and tangent plane
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, alpha=0.3)

# Tangent plane at (1, 0, 0)
xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
zz = np.zeros_like(xx)
ax.plot_surface(1 + xx, yy, zz, alpha=0.5, color='r')

ax.set_title('Sphere with Tangent Plane')
plt.show()
```

Slide 3: Exponential and Logarithmic Maps

The exponential map projects points from the tangent space onto the manifold, while the logarithmic map does the opposite. These maps are crucial for moving between the manifold and its tangent spaces.

```python
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=2)

# Define a point on the sphere
base_point = np.array([1.0, 0.0, 0.0])

# Define a tangent vector
tangent_vec = np.array([0.0, 0.5, 0.5])

# Compute the exponential map
exp_point = sphere.exp(tangent_vec, base_point)

# Visualize
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)

# Plot the points
ax.scatter(*base_point, color='r', s=100, label='Base point')
ax.scatter(*exp_point, color='g', s=100, label='Exp map result')

# Plot the tangent vector
ax.quiver(*base_point, *tangent_vec, color='b', length=1, label='Tangent vector')

ax.set_title('Exponential Map on Sphere')
ax.legend()
plt.show()
```

Slide 4: Geodesics on Manifolds

Geodesics are the shortest paths between two points on a manifold. In PGA, we use geodesics to define the principal directions of variation in the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere

sphere = Hypersphere(dim=2)

# Define two points on the sphere
point_a = np.array([1.0, 0.0, 0.0])
point_b = np.array([0.0, 1.0, 0.0])

# Compute the geodesic
t = np.linspace(0, 1, 100)
geodesic = sphere.geodesic(initial_point=point_a, end_point=point_b)
points = geodesic(t)

# Visualize
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)

# Plot the geodesic
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r', linewidth=2)

# Plot the points
ax.scatter(*point_a, color='g', s=100, label='Point A')
ax.scatter(*point_b, color='b', s=100, label='Point B')

ax.set_title('Geodesic on Sphere')
ax.legend()
plt.show()
```

Slide 5: Fréchet Mean

The Fréchet mean is a generalization of the arithmetic mean for data on a manifold. It minimizes the sum of squared geodesic distances to all data points.

```python
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

sphere = Hypersphere(dim=2)

# Generate random points on the sphere
n_points = 100
data = sphere.random_uniform(n_points)

# Compute Fréchet mean
frechet_mean = FrechetMean(metric=sphere.metric)
frechet_mean.fit(data)

mean = frechet_mean.estimate_

print(f"Fréchet mean: {mean}")
print(f"Is the mean on the sphere? {np.allclose(np.linalg.norm(mean), 1.0)}")
```

Slide 6: Principal Geodesic Analysis Algorithm

PGA extends PCA to manifolds by performing the following steps:

1. Compute the Fréchet mean of the data
2. Map data points to the tangent space at the mean using the log map
3. Perform PCA in the tangent space
4. Map the principal components back to the manifold using the exp map

```python
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean

def pga(data, n_components):
    sphere = Hypersphere(dim=data.shape[1] - 1)
    
    # Step 1: Compute Fréchet mean
    frechet_mean = FrechetMean(metric=sphere.metric)
    frechet_mean.fit(data)
    mean = frechet_mean.estimate_
    
    # Step 2: Map data to tangent space
    tangent_data = sphere.log(data, base_point=mean)
    
    # Step 3: Perform PCA in tangent space
    cov = np.cov(tangent_data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    principal_directions = eigenvectors[:, :n_components]
    
    return mean, principal_directions, eigenvalues[:n_components]

# Example usage
n_points = 100
sphere = Hypersphere(dim=2)
data = sphere.random_uniform(n_points)

mean, principal_directions, eigenvalues = pga(data, n_components=2)
print("Mean:", mean)
print("Principal directions:")
print(principal_directions)
print("Eigenvalues:", eigenvalues)
```

Slide 7: Visualization of PGA Results

Let's visualize the results of PGA on the sphere to better understand how it captures the principal directions of variation in the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere

def visualize_pga_results(data, mean, principal_directions):
    sphere = Hypersphere(dim=2)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
    
    # Plot data points
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b', alpha=0.5)
    
    # Plot mean
    ax.scatter(*mean, color='r', s=100, label='Fréchet mean')
    
    # Plot principal directions
    for i, direction in enumerate(principal_directions.T):
        geodesic = sphere.geodesic(initial_point=mean, initial_tangent_vec=direction)
        t = np.linspace(-1, 1, 100)
        points = geodesic(t)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color='g', linewidth=2, label=f'PD {i+1}')
    
    ax.set_title('PGA Results on Sphere')
    ax.legend()
    plt.show()

# Generate data and perform PGA
n_points = 100
sphere = Hypersphere(dim=2)
data = sphere.random_uniform(n_points)

mean, principal_directions, _ = pga(data, n_components=2)

# Visualize results
visualize_pga_results(data, mean, principal_directions)
```

Slide 8: Comparison with PCA

PGA respects the geometry of the manifold, while PCA assumes a flat Euclidean space. This difference is crucial when working with data that naturally lies on a curved manifold.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate data on a hemisphere
n_points = 1000
theta = np.random.uniform(0, np.pi/2, n_points)
phi = np.random.uniform(0, 2*np.pi, n_points)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)
data = np.column_stack((x, y, z))

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Perform PGA
_, principal_directions, _ = pga(data, n_components=2)

# Visualize results
fig = plt.figure(figsize=(15, 5))

# Original data
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(x, y, z)
ax1.set_title('Original Data')

# PCA result
ax2 = fig.add_subplot(132)
ax2.scatter(pca_result[:, 0], pca_result[:, 1])
ax2.set_title('PCA Result')

# PGA result
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(x, y, z)
for direction in principal_directions.T:
    ax3.quiver(0, 0, 0, *direction, color='r', length=1)
ax3.set_title('PGA Result')

plt.tight_layout()
plt.show()
```

Slide 9: Applications of PGA: Shape Analysis

PGA is widely used in shape analysis, where shapes are represented as points on a manifold. It allows us to capture the main modes of variation in a set of shapes.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

def generate_ellipse(a, b, rotation):
    t = np.linspace(0, 2*np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)
    points = np.column_stack((x, y, np.zeros_like(x)))
    return np.dot(points, rotation.T)

# Generate a set of ellipses with varying parameters
n_shapes = 50
shapes = []
for _ in range(n_shapes):
    a = np.random.uniform(0.5, 1.5)
    b = np.random.uniform(0.5, 1.5)
    rotation = special_ortho_group.rvs(3)
    shapes.append(generate_ellipse(a, b, rotation))

# Perform PGA on the shapes
mean, principal_directions, _ = pga(np.array(shapes).reshape(n_shapes, -1), n_components=2)

# Visualize the mean shape and principal modes of variation
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(mean[::3], mean[1::3])
axs[0].set_title('Mean Shape')

for i, direction in enumerate(principal_directions.T):
    variation = mean + direction.reshape(-1, 3)
    axs[i+1].plot(variation[::3], variation[1::3])
    axs[i+1].set_title(f'Principal Mode {i+1}')

for ax in axs:
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()
```

Slide 10: Applications of PGA: Motion Analysis

PGA can be applied to analyze human motion data, where each pose is represented as a point on a high-dimensional manifold. This technique helps in understanding the principal modes of variation in motion sequences.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_walking_pose(t, amplitude=1, frequency=1):
    left_leg = np.sin(2 * np.pi * frequency * t) * amplitude
    right_leg = -np.sin(2 * np.pi * frequency * t) * amplitude
    left_arm = -np.sin(2 * np.pi * frequency * t) * amplitude * 0.5
    right_arm = np.sin(2 * np.pi * frequency * t) * amplitude * 0.5
    return np.array([left_leg, right_leg, left_arm, right_arm])

n_frames = 100
t = np.linspace(0, 1, n_frames)
motion_data = np.array([generate_walking_pose(ti) for ti in t])

mean, principal_directions, _ = pga(motion_data, n_components=2)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(t, motion_data)
axs[0].set_title('Original Motion Data')

axs[1].plot(t, mean.reshape(-1, 4))
axs[1].set_title('Mean Motion')

for i, direction in enumerate(principal_directions.T):
    variation = mean + direction.reshape(-1, 4)
    axs[2].plot(t, variation, label=f'Mode {i+1}')
axs[2].set_title('Principal Modes of Variation')
axs[2].legend()

plt.tight_layout()
plt.show()
```

Slide 11: PGA for Dimensionality Reduction

PGA can be used for dimensionality reduction on manifolds, similar to how PCA is used in Euclidean spaces. This is particularly useful for visualizing high-dimensional data that lies on a manifold.

```python
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.hypersphere import Hypersphere

def pga_projection(data, mean, principal_directions, n_components):
    sphere = Hypersphere(dim=data.shape[1] - 1)
    tangent_data = sphere.log(data, base_point=mean)
    projected_data = np.dot(tangent_data, principal_directions[:, :n_components])
    return projected_data

# Generate data on a 3-sphere
n_points = 1000
sphere = Hypersphere(dim=3)
data = sphere.random_uniform(n_points)

# Perform PGA
mean, principal_directions, _ = pga(data, n_components=2)

# Project data onto the first two principal geodesics
projected_data = pga_projection(data, mean, principal_directions, 2)

# Visualize the results
plt.figure(figsize=(10, 10))
plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.5)
plt.title('PGA Projection of 3-sphere Data')
plt.xlabel('First Principal Geodesic')
plt.ylabel('Second Principal Geodesic')
plt.show()
```

Slide 12: PGA for Data Generation

PGA can be used to generate new data points that follow the distribution of the original data on the manifold. This is useful for tasks such as data augmentation or synthesis.

```python
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere

def generate_pga_samples(mean, principal_directions, eigenvalues, n_samples):
    sphere = Hypersphere(dim=mean.shape[0] - 1)
    
    # Generate random coefficients
    coeffs = np.random.randn(n_samples, len(eigenvalues)) * np.sqrt(eigenvalues)
    
    # Generate tangent vectors
    tangent_vecs = np.dot(coeffs, principal_directions.T)
    
    # Map tangent vectors to the manifold
    samples = sphere.exp(tangent_vecs, base_point=mean)
    
    return samples

# Generate data on a 2-sphere
n_points = 1000
sphere = Hypersphere(dim=2)
data = sphere.random_uniform(n_points)

# Perform PGA
mean, principal_directions, eigenvalues = pga(data, n_components=2)

# Generate new samples
n_new_samples = 500
new_samples = generate_pga_samples(mean, principal_directions, eigenvalues, n_new_samples)

# Visualize original data and new samples
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.3, label='Original Data')
ax.scatter(new_samples[:, 0], new_samples[:, 1], new_samples[:, 2], color='r', alpha=0.5, label='Generated Samples')

ax.set_title('Original Data and PGA-Generated Samples on 2-sphere')
ax.legend()
plt.show()
```

Slide 13: Challenges and Limitations of PGA

While PGA is a powerful tool for analyzing data on manifolds, it faces some challenges and limitations:

1. Computational complexity: PGA can be computationally expensive, especially for high-dimensional manifolds or large datasets.
2. Choice of manifold: The results of PGA depend on the chosen manifold structure, which may not always be obvious for complex data.
3. Non-convexity: The optimization problem in PGA is generally non-convex, which can lead to local optima.
4. Interpretability: The principal geodesics may be harder to interpret than principal components in standard PCA.

```python
# Pseudocode for illustrating computational complexity
def pga_complexity(n_samples, manifold_dim, n_components):
    # Compute Fréchet mean
    mean_complexity = O(n_samples * manifold_dim)
    
    # Map data to tangent space
    tangent_map_complexity = O(n_samples * manifold_dim)
    
    # Compute covariance matrix
    cov_complexity = O(n_samples * manifold_dim**2)
    
    # Eigendecomposition
    eigen_complexity = O(manifold_dim**3)
    
    total_complexity = mean_complexity + tangent_map_complexity + cov_complexity + eigen_complexity
    return total_complexity

# Example usage
n_samples = 1000
manifold_dim = 100
n_components = 10

complexity = pga_complexity(n_samples, manifold_dim, n_components)
print(f"Computational complexity: {complexity}")
```

Slide 14: Real-life Example: Analyzing Brain Connectivity

PGA can be applied to analyze brain connectivity patterns, where each brain's connectivity matrix is treated as a point on a manifold of symmetric positive definite matrices.

```python
import numpy as np
import matplotlib.pyplot as plt
from geomstats.geometry.spd_matrices import SPDMatrices

def generate_connectivity_matrix(n_regions):
    # Generate a random symmetric positive definite matrix
    A = np.random.randn(n_regions, n_regions)
    return np.dot(A, A.T)

# Generate synthetic brain connectivity data
n_subjects = 100
n_regions = 10
connectivity_data = np.array([generate_connectivity_matrix(n_regions) for _ in range(n_subjects)])

# Perform PGA on the connectivity data
spd_space = SPDMatrices(n_regions)
mean, principal_directions, _ = pga(connectivity_data, n_components=2)

# Visualize the mean connectivity and principal modes of variation
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(mean, cmap='coolwarm')
axs[0].set_title('Mean Connectivity')

for i in range(2):
    variation = mean + principal_directions[:, i].reshape(n_regions, n_regions)
    axs[i+1].imshow(variation, cmap='coolwarm')
    axs[i+1].set_title(f'Principal Mode {i+1}')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into Principal Geodesic Analysis and its applications, here are some valuable resources:

1. Fletcher, P. T., Lu, C., Pizer, S. M., & Joshi, S. (2004). Principal geodesic analysis for the study of nonlinear statistics of shape. IEEE transactions on medical imaging, 23(8), 995-1005. ArXiv link: [https://arxiv.org/abs/cs/0401040](https://arxiv.org/abs/cs/0401040)
2. Sommer, S., Lauze, F., & Nielsen, M. (2014). Optimization over geodesics for exact principal geodesic analysis. Advances in Computational Mathematics, 40(2), 283-313. ArXiv link: [https://arxiv.org/abs/1008.1902](https://arxiv.org/abs/1008.1902)
3. Pennec, X. (2006). Intrinsic statistics on Riemannian manifolds: Basic tools for geometric measurements. Journal of Mathematical Imaging and Vision, 25(1), 127-154. ArXiv link: [https://arxiv.org/abs/math/0401420](https://arxiv.org/abs/math/0401420)

These papers provide in-depth explanations of PGA theory, algorithms, and applications in various fields such as medical imaging and computer vision.

