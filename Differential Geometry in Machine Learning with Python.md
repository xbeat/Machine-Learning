## Differential Geometry in Machine Learning with Python
Slide 1: Introduction to Differential Geometry in Machine Learning and AI

Differential geometry provides a powerful framework for understanding and optimizing machine learning algorithms. It allows us to analyze the geometric properties of high-dimensional data spaces and optimization landscapes. In this presentation, we'll explore key concepts and their applications in ML and AI, with practical Python examples.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_manifold(f, u_range, v_range):
    u = np.linspace(*u_range, 100)
    v = np.linspace(*v_range, 100)
    U, V = np.meshgrid(u, v)
    X, Y, Z = f(U, V)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.show()

# Example: Plotting a sphere
f = lambda u, v: (np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v))
plot_manifold(f, (0, 2*np.pi), (0, np.pi))
```

Slide 2: Manifolds and Data Representation

In machine learning, data often lies on or near a lower-dimensional manifold embedded in a high-dimensional space. Understanding these manifolds can help in dimensionality reduction, feature extraction, and model design. Let's visualize a simple manifold: a Swiss roll dataset.

```python
from sklearn.datasets import make_swiss_roll

# Generate Swiss roll dataset
X, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

fig = plt.figure(figsize=(12, 6))

# 3D scatter plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis')
ax1.set_title("3D Swiss Roll")

# 2D scatter plot (first two principal components)
ax2 = fig.add_subplot(122)
ax2.scatter(X[:, 0], X[:, 1], c=color, cmap='viridis')
ax2.set_title("2D Projection")

plt.tight_layout()
plt.show()
```

Slide 3: Tangent Spaces and Gradients

The tangent space at a point on a manifold is crucial for understanding local geometry and defining gradients. In machine learning, gradients guide optimization algorithms like gradient descent. Let's implement a simple gradient descent algorithm on a 2D function.

```python
def f(x, y):
    return x**2 + y**2

def grad_f(x, y):
    return np.array([2*x, 2*y])

def gradient_descent(f, grad_f, start, lr=0.1, n_iter=100):
    path = [start]
    for _ in range(n_iter):
        grad = grad_f(*path[-1])
        new_point = path[-1] - lr * grad
        path.append(new_point)
    return np.array(path)

start = np.array([1.0, 1.0])
path = gradient_descent(f, grad_f, start)

x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=20)
plt.colorbar(label='f(x, y)')
plt.plot(path[:, 0], path[:, 1], 'ro-')
plt.title("Gradient Descent on f(x, y) = x^2 + y^2")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 4: Riemannian Metrics and Distance

Riemannian metrics define how to measure distances and angles on a manifold. In machine learning, these metrics can be used to design algorithms that respect the geometry of the data space. Let's implement a simple example of computing geodesic distances on a sphere.

```python
def sphere_metric(theta, phi):
    return np.array([[1, 0], [0, np.sin(theta)**2]])

def geodesic_distance(p1, p2):
    theta1, phi1 = p1
    theta2, phi2 = p2
    
    # Great circle distance
    return np.arccos(np.sin(theta1)*np.sin(theta2) + 
                     np.cos(theta1)*np.cos(theta2)*np.cos(phi1-phi2))

# Example points on a sphere
p1 = (np.pi/4, np.pi/4)
p2 = (np.pi/2, np.pi/2)

print(f"Geodesic distance: {geodesic_distance(p1, p2)}")

# Visualize the points on a sphere
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, alpha=0.3)
ax.scatter(*p1, c='r', s=100)
ax.scatter(*p2, c='b', s=100)
ax.set_title("Points on a Sphere")
plt.show()
```

Slide 5: Parallel Transport and Vector Fields

Parallel transport is a way to move vectors along curves on a manifold while preserving their properties. In machine learning, this concept is useful for analyzing how gradients change across the parameter space. Let's visualize a simple vector field on a manifold.

```python
def vector_field(x, y):
    return -y, x

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
U, V = vector_field(X, Y)

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V)
plt.title("Vector Field: (-y, x)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
```

Slide 6: Curvature and Its Impact on Optimization

Curvature measures how a manifold deviates from being flat. In machine learning, the curvature of the loss landscape affects optimization dynamics. Let's visualize the curvature of a simple 2D function and its impact on optimization.

```python
def f(x, y):
    return x**2 - y**2

def grad_f(x, y):
    return np.array([2*x, -2*y])

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.contour(X, Y, Z, levels=20)
plt.colorbar(label='f(x, y)')
plt.title("Contour Plot of f(x, y) = x^2 - y^2")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(122)
start_points = [(-1, -1), (1, 1), (1, -1), (-1, 1)]
colors = ['r', 'g', 'b', 'c']

for start, color in zip(start_points, colors):
    path = gradient_descent(f, grad_f, np.array(start), lr=0.1, n_iter=20)
    plt.plot(path[:, 0], path[:, 1], color+'-o', label=f"Start: {start}")

plt.contour(X, Y, Z, levels=20)
plt.colorbar(label='f(x, y)')
plt.title("Gradient Descent Paths")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 7: Geodesics and Natural Gradient Descent

Geodesics are curves that locally minimize distance on a manifold. In optimization, following geodesics can lead to more efficient algorithms like natural gradient descent. Let's implement a simple version of natural gradient descent.

```python
def f(theta):
    return -np.cos(theta[0]) * np.cos(theta[1])

def grad_f(theta):
    return np.array([np.sin(theta[0]) * np.cos(theta[1]),
                     np.cos(theta[0]) * np.sin(theta[1])])

def fisher_information_matrix(theta):
    return np.array([[np.cos(theta[0])**2, 0],
                     [0, np.cos(theta[1])**2]])

def natural_gradient_descent(f, grad_f, fim, start, lr=0.1, n_iter=100):
    path = [start]
    for _ in range(n_iter):
        grad = grad_f(path[-1])
        F_inv = np.linalg.inv(fim(path[-1]))
        natural_grad = F_inv @ grad
        new_point = path[-1] - lr * natural_grad
        path.append(new_point)
    return np.array(path)

start = np.array([np.pi/4, np.pi/4])
path_ngd = natural_gradient_descent(f, grad_f, fisher_information_matrix, start)
path_gd = gradient_descent(f, grad_f, start)

theta1 = np.linspace(0, 2*np.pi, 100)
theta2 = np.linspace(0, 2*np.pi, 100)
Theta1, Theta2 = np.meshgrid(theta1, theta2)
Z = f([Theta1, Theta2])

plt.figure(figsize=(10, 8))
plt.contour(Theta1, Theta2, Z, levels=20)
plt.colorbar(label='f(θ)')
plt.plot(path_ngd[:, 0], path_ngd[:, 1], 'r-o', label='Natural Gradient Descent')
plt.plot(path_gd[:, 0], path_gd[:, 1], 'b-o', label='Gradient Descent')
plt.title("Natural Gradient Descent vs Gradient Descent")
plt.xlabel("θ1")
plt.ylabel("θ2")
plt.legend()
plt.show()
```

Slide 8: Lie Groups and Symmetries in Neural Networks

Lie groups are continuous symmetry groups that play a role in understanding invariances in neural networks. Let's implement a simple example of a rotation-invariant convolutional layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationInvariantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        # Original convolution
        out1 = self.conv(x)
        
        # Rotate input by 90 degrees
        x_rot90 = torch.rot90(x, 1, [2, 3])
        out2 = self.conv(x_rot90)
        out2 = torch.rot90(out2, -1, [2, 3])
        
        # Rotate input by 180 degrees
        x_rot180 = torch.rot90(x, 2, [2, 3])
        out3 = self.conv(x_rot180)
        out3 = torch.rot90(out3, -2, [2, 3])
        
        # Rotate input by 270 degrees
        x_rot270 = torch.rot90(x, 3, [2, 3])
        out4 = self.conv(x_rot270)
        out4 = torch.rot90(out4, -3, [2, 3])
        
        # Average the outputs
        return (out1 + out2 + out3 + out4) / 4

# Example usage
conv = RotationInvariantConv2d(3, 16, 3)
input_tensor = torch.randn(1, 3, 32, 32)
output = conv(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

Slide 9: Differential Forms and Information Geometry

Differential forms provide a coordinate-independent way to describe geometric quantities. In machine learning, they're useful for understanding information geometry and natural gradient methods. Let's implement a simple example of computing the Fisher information matrix using differential forms.

```python
import sympy as sp

# Define symbolic variables
theta1, theta2 = sp.symbols('theta1 theta2')

# Define a probability distribution
p = sp.exp(-(theta1**2 + theta2**2) / 2) / (2 * sp.pi)

# Compute log-likelihood
log_p = sp.log(p)

# Compute Fisher information matrix
fim = sp.Matrix([
    [sp.diff(log_p, theta1, 2), sp.diff(log_p, theta1, theta2)],
    [sp.diff(log_p, theta2, theta1), sp.diff(log_p, theta2, 2)]
])

print("Fisher Information Matrix:")
sp.pprint(fim)

# Evaluate FIM at a specific point
fim_eval = fim.subs({theta1: 0, theta2: 0})
print("\nFIM evaluated at (0, 0):")
sp.pprint(fim_eval)
```

Slide 10: Connections and Parallel Transport in Neural Networks

Connections define how to compare vectors at different points on a manifold. In neural networks, they can be used to design weight-sharing schemes that respect the geometry of the problem. Let's implement a simple example of parallel transport along a curve.

```python
import numpy as np
import matplotlib.pyplot as plt

def parallel_transport_circle(v0, theta):
    """Parallel transport vector v0 along a circle by angle theta."""
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ v0

# Initial vector
v0 = np.array([1, 0])

# Generate points along a circle
theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

plt.figure(figsize=(10, 10))
plt.plot(x, y, 'k-')

# Plot parallel transported vectors
for t in np.linspace(0, 2*np.pi, 8):
    v = parallel_transport_circle(v0, t)
    plt.arrow(np.cos(t), np.sin(t), v[0]*0.2, v[1]*0.2, 
              head_width=0.05, head_length=0.1, fc='r', ec='r')

plt.title("Parallel Transport of a Vector Along a Circle")
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 11: Holonomy and Weight Initialization

Holonomy measures how parallel transport around a closed loop can change a vector. This concept inspires novel weight initialization schemes for neural networks. Let's implement a holonomy-aware weight initialization.

```python
import torch
import torch.nn as nn
import numpy as np

def holonomy_init(tensor):
    dim = tensor.size(0)
    U, _, V = torch.linalg.svd(torch.randn(dim, dim))
    holonomy_matrix = U @ V.T
    with torch.no_grad():
        tensor._(holonomy_matrix)

class HolonomyLinear(nn.Linear):
    def reset_parameters(self):
        holonomy_init(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

# Example usage
layer = HolonomyLinear(10, 10)
input_tensor = torch.randn(5, 10)
output = layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Weight matrix condition number: {np.linalg.cond(layer.weight.detach().numpy())}")
```

Slide 12: Lie Derivatives and Invariant Features

Lie derivatives describe how scalar fields change along vector fields. In machine learning, they can be used to design invariant features. Let's implement a simple example of computing Lie derivatives for a rotation-invariant feature.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2

def rotation_vector_field(x, y):
    return -y, x

def lie_derivative(f, vf, x, y):
    fx = 2*x
    fy = 2*y
    return fx * vf(x, y)[0] + fy * vf(x, y)[1]

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

Z = f(X, Y)
L_Z = lie_derivative(f, rotation_vector_field, X, Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

c1 = ax1.contourf(X, Y, Z)
ax1.set_title("Original Function")
fig.colorbar(c1, ax=ax1)

c2 = ax2.contourf(X, Y, L_Z)
ax2.set_title("Lie Derivative along Rotation Field")
fig.colorbar(c2, ax=ax2)

plt.show()
```

Slide 13: Ricci Flow and Manifold Learning

Ricci flow is a process that evolves a manifold's metric to smooth out its curvature. In machine learning, it can be used for manifold learning and dimensionality reduction. Let's implement a simple version of Ricci flow on a graph.

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def ricci_flow_step(G, eta=0.01):
    for (u, v) in G.edges():
        weight = G[u][v]['weight']
        new_weight = weight * (1 - eta * (G.degree(u) + G.degree(v)))
        G[u][v]['weight'] = max(0.01, new_weight)  # Prevent negative weights

def visualize_graph(G, title):
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for (u, v) in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, width=weights)
    plt.title(title)
    plt.axis('off')

# Create a sample graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])
for (u, v) in G.edges():
    G[u][v]['weight'] = 1.0

plt.figure(figsize=(15, 5))

plt.subplot(131)
visualize_graph(G, "Original Graph")

# Apply Ricci flow
for _ in range(50):
    ricci_flow_step(G)

plt.subplot(132)
visualize_graph(G, "After Ricci Flow")

plt.tight_layout()
plt.show()
```

Slide 14: Conclusion and Future Directions

We've explored various concepts from differential geometry and their applications in machine learning and AI. These tools provide powerful ways to analyze and design algorithms that respect the geometric structure of data and parameter spaces. Future research directions include:

1. Developing more efficient natural gradient methods for large-scale optimization
2. Designing neural network architectures with built-in geometric invariances
3. Applying differential geometric techniques to reinforcement learning and causal inference
4. Exploring connections between information geometry and quantum computing for machine learning

As the field progresses, we can expect differential geometry to play an increasingly important role in advancing the theoretical foundations and practical applications of machine learning and AI.

Slide 15: Additional Resources

For those interested in diving deeper into differential geometry in machine learning and AI, here are some valuable resources:

1. "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" by Bronstein et al. (2021) - ArXiv: [https://arxiv.org/abs/2104.13478](https://arxiv.org/abs/2104.13478)
2. "Information Geometry and Its Applications" by Amari (2016) - A comprehensive book on the subject.
3. "Riemannian Geometry and Geometric Analysis" by Jost (2017) - For a rigorous mathematical treatment of differential geometry.
4. "Natural Gradient Works Efficiently in Learning" by Amari (1998) - ArXiv: [https://arxiv.org/abs/1301.3584](https://arxiv.org/abs/1301.3584)
5. "A Differential Geometric Approach to Automated Variational Inference" by Regier et al. (2017) - ArXiv: [https://arxiv.org/abs/1710.07283](https://arxiv.org/abs/1710.07283)

These resources provide a mix of theoretical foundations and practical applications, suitable for readers with varying levels of expertise in the field.

