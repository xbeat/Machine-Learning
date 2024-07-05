## Harmonic Function Theory with Python

Slide 1: Introduction to Harmonic Function Theory

Harmonic functions are solutions to Laplace's equation and play a crucial role in many areas of mathematics and physics. They are smooth functions with fascinating properties and applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def laplacian(f, x, y, h=1e-5):
    return (f(x+h,y) + f(x-h,y) + f(x,y+h) + f(x,y-h) - 4*f(x,y)) / (h**2)

def is_harmonic(f, x, y, tolerance=1e-6):
    return abs(laplacian(f, x, y)) < tolerance

# Example harmonic function
def harmonic_example(x, y):
    return x**2 - y**2

# Test the function
x, y = 1.0, 2.0
print(f"Is harmonic at ({x}, {y}): {is_harmonic(harmonic_example, x, y)}")
```

Slide 2: Properties of Harmonic Functions

Harmonic functions possess several important properties, including the mean value property and the maximum principle. These properties make them useful in various fields.

```python
def mean_value_property(f, x, y, r, n=1000):
    theta = np.linspace(0, 2*np.pi, n)
    circle_x = x + r * np.cos(theta)
    circle_y = y + r * np.sin(theta)
    return np.mean([f(cx, cy) for cx, cy in zip(circle_x, circle_y)])

def check_mean_value_property(f, x, y, r):
    center_value = f(x, y)
    mean_value = mean_value_property(f, x, y, r)
    return np.isclose(center_value, mean_value)

# Test mean value property
x, y, r = 0.5, 0.5, 0.1
print(f"Satisfies mean value property: {check_mean_value_property(harmonic_example, x, y, r)}")
```

Slide 3: Laplace's Equation

Laplace's equation is a second-order partial differential equation that defines harmonic functions. Understanding this equation is crucial for grasping harmonic function theory.

```python
def laplace_equation(f, x, y, h=1e-5):
    return (f(x+h,y) + f(x-h,y) + f(x,y+h) + f(x,y-h) - 4*f(x,y)) / (h**2) == 0

# Visualize Laplace's equation
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = harmonic_example(X, Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Function value')
plt.title("Contour plot of a harmonic function")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 4: Complex Analysis and Harmonic Functions

In complex analysis, the real and imaginary parts of analytic functions are harmonic. This connection provides powerful tools for studying harmonic functions.

```python
import cmath

def complex_harmonic(z):
    return z.real**2 - z.imag**2 + 1j * (2 * z.real * z.imag)

# Visualize real and imaginary parts
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = complex_harmonic(X + 1j*Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
im1 = ax1.contourf(X, Y, Z.real, levels=20, cmap='viridis')
ax1.set_title("Real part")
plt.colorbar(im1, ax=ax1)
im2 = ax2.contourf(X, Y, Z.imag, levels=20, cmap='viridis')
ax2.set_title("Imaginary part")
plt.colorbar(im2, ax=ax2)
plt.show()
```

Slide 5: Dirichlet Problem

The Dirichlet problem involves finding a harmonic function in a domain that satisfies given boundary conditions. It has applications in physics and engineering.

```python
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

def solve_dirichlet(n, boundary_func):
    h = 1.0 / (n + 1)
    A = sparse.lil_matrix((n**2, n**2))
    b = np.zeros(n**2)

    for i in range(n):
        for j in range(n):
            k = i * n + j
            A[k, k] = 4
            if i > 0: A[k, k-n] = -1
            if i < n-1: A[k, k+n] = -1
            if j > 0: A[k, k-1] = -1
            if j < n-1: A[k, k+1] = -1

            if i == 0: b[k] += boundary_func(0, (j+1)*h)
            if i == n-1: b[k] += boundary_func(1, (j+1)*h)
            if j == 0: b[k] += boundary_func((i+1)*h, 0)
            if j == n-1: b[k] += boundary_func((i+1)*h, 1)

    A = A.tocsr()
    u = spsolve(A, b)
    return u.reshape((n, n))

# Example boundary condition
def boundary_func(x, y):
    return np.sin(np.pi * x) * np.sinh(np.pi * y)

# Solve and visualize
n = 50
u = solve_dirichlet(n, boundary_func)
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, u, levels=20, cmap='viridis')
plt.colorbar(label='Function value')
plt.title("Solution to Dirichlet problem")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 6: Green's Functions

Green's functions are fundamental solutions to Laplace's equation and are used to solve boundary value problems. They provide a powerful method for finding harmonic functions.

```python
def green_function(x, y, x0, y0):
    return -1 / (2 * np.pi) * np.log(np.sqrt((x-x0)**2 + (y-y0)**2))

# Visualize Green's function
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
x0, y0 = 0, 0
Z = green_function(X, Y, x0, y0)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Function value')
plt.title("Green's function")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x0, y0, 'ro', markersize=10)
plt.show()
```

Slide 7: Harmonic Conjugates

For a given harmonic function u(x, y), its harmonic conjugate v(x, y) forms an analytic function f(z) = u(x, y) + i\*v(x, y). This concept links harmonic functions to complex analysis.

```python
def harmonic_conjugate(u, x, y, h=1e-5):
    ux = (u(x+h, y) - u(x-h, y)) / (2*h)
    uy = (u(x, y+h) - u(x, y-h)) / (2*h)
    return -uy, ux

def u(x, y):
    return np.exp(x) * np.cos(y)

# Calculate and visualize harmonic conjugate
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
U = u(X, Y)
vx, vy = harmonic_conjugate(u, X, Y)
V = np.cumsum(vx, axis=1) * (x[1] - x[0]) + np.cumsum(vy, axis=0) * (y[1] - y[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
im1 = ax1.contourf(X, Y, U, levels=20, cmap='viridis')
ax1.set_title("Harmonic function u(x,y)")
plt.colorbar(im1, ax=ax1)
im2 = ax2.contourf(X, Y, V, levels=20, cmap='viridis')
ax2.set_title("Harmonic conjugate v(x,y)")
plt.colorbar(im2, ax=ax2)
plt.show()
```

Slide 8: Poisson's Equation

Poisson's equation is a generalization of Laplace's equation that includes a source term. It's widely used in physics to describe phenomena like electrostatics and gravity.

```python
def poisson_solver(f, n, boundary_func):
    h = 1.0 / (n + 1)
    A = sparse.lil_matrix((n**2, n**2))
    b = np.zeros(n**2)

    for i in range(n):
        for j in range(n):
            k = i * n + j
            A[k, k] = 4
            if i > 0: A[k, k-n] = -1
            if i < n-1: A[k, k+n] = -1
            if j > 0: A[k, k-1] = -1
            if j < n-1: A[k, k+1] = -1

            b[k] = h**2 * f((i+1)*h, (j+1)*h)

            if i == 0: b[k] += boundary_func(0, (j+1)*h)
            if i == n-1: b[k] += boundary_func(1, (j+1)*h)
            if j == 0: b[k] += boundary_func((i+1)*h, 0)
            if j == n-1: b[k] += boundary_func((i+1)*h, 1)

    A = A.tocsr()
    u = spsolve(A, b)
    return u.reshape((n, n))

# Example source term and boundary condition
def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def boundary_func(x, y):
    return 0

# Solve and visualize
n = 50
u = poisson_solver(f, n, boundary_func)
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, u, levels=20, cmap='viridis')
plt.colorbar(label='Function value')
plt.title("Solution to Poisson's equation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 9: Harmonic Functions in Higher Dimensions

Harmonic functions can be extended to higher dimensions, maintaining their properties and applications. This extension is crucial for many physical problems.

```python
from mpl_toolkits.mplot3d import Axes3D

def harmonic_3d(x, y, z):
    return x**2 + y**2 - 2*z**2

# Visualize 3D harmonic function
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
z = np.linspace(-2, 2, 20)
X, Y, Z = np.meshgrid(x, y, z)
U = harmonic_3d(X, Y, Z)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X, Y, Z, c=U, cmap='viridis')
plt.colorbar(scatter)
ax.set_title("3D Harmonic Function")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

Slide 10: Harmonic Functions in Polar Coordinates

Harmonic functions can be expressed in polar coordinates, which is useful for problems with radial symmetry. This representation leads to important solutions like the fundamental solution of Laplace's equation.

```python
def harmonic_polar(r, theta):
    return r**2 * np.cos(2*theta)

# Visualize harmonic function in polar coordinates
r = np.linspace(0, 2, 100)
theta = np.linspace(0, 2*np.pi, 100)
R, Theta = np.meshgrid(r, theta)
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = harmonic_polar(R, Theta)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
plt.colorbar(surf)
ax.set_title("Harmonic Function in Polar Coordinates")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

Slide 11: Spherical Harmonics

Spherical harmonics are the angular portion of the solution to Laplace's equation in spherical coordinates. They have important applications in quantum mechanics and computer graphics.

```python
from scipy.special import sph_harm

def plot_spherical_harmonic(l, m):
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    Y = sph_harm(m, l, phi, theta).real
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(Y), alpha=0.8)
    ax.set_title(f"Spherical Harmonic Y_{l}^{m}")
    plt.show()

# Example: Plot Y_3^2
plot_spherical_harmonic(3, 2)
```

Slide 12: Applications in Physics - Heat Conduction

Harmonic functions play a crucial role in various physical phenomena, including heat conduction. Let's simulate a 2D heat equation, which is closely related to Laplace's equation in steady-state.

```python
def heat_equation_2d(u0, dx, dy, dt, t_final):
    nx, ny = u0.shape
    nt = int(t_final / dt)
    u = u0.()

    for _ in range(nt):
        u_old = u.()
        u[1:-1, 1:-1] = u_old[1:-1, 1:-1] + dt * (
            (u_old[2:, 1:-1] - 2*u_old[1:-1, 1:-1] + u_old[:-2, 1:-1]) / dx**2 +
            (u_old[1:-1, 2:] - 2*u_old[1:-1, 1:-1] + u_old[1:-1, :-2]) / dy**2
        )
    return u

# Initial conditions
nx, ny = 50, 50
u0 = np.zeros((nx, ny))
u0[10:20, 10:20] = 100  # Heat source

# Simulation parameters
dx = dy = 0.1
dt = 0.0001
t_final = 0.1

# Run simulation
u_final = heat_equation_2d(u0, dx, dy, dt, t_final)

# Visualize results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(u0, cmap='hot')
plt.title("Initial Temperature")
plt.colorbar()

plt.subplot(122)
plt.imshow(u_final, cmap='hot')
plt.title(f"Temperature at t = {t_final}")
plt.colorbar()

plt.tight_layout()
plt.show()
```

Slide 13: Harmonic Functions in Machine Learning

Harmonic functions have found applications in machine learning, particularly in semi-supervised learning and graph-based methods. Let's implement a simple example of harmonic function label propagation.

```python
import networkx as nx

def harmonic_function_label_propagation(G, labeled_nodes, max_iter=1000, tol=1e-6):
    # Initialize labels
    labels = {node: 0 for node in G.nodes()}
    labels.update(labeled_nodes)
    
    for _ in range(max_iter):
        diff = 0
        for node in G.nodes():
            if node not in labeled_nodes:
                neighbors = list(G.neighbors(node))
                new_label = sum(labels[n] for n in neighbors) / len(neighbors)
                diff += abs(labels[node] - new_label)
                labels[node] = new_label
        
        if diff < tol:
            break
    
    return labels

# Create a sample graph
G = nx.grid_2d_graph(5, 5)
pos = {(x, y): (y, -x) for x, y in G.nodes()}

# Set labeled nodes
labeled_nodes = {(0, 0): 1, (4, 4): 0}

# Run label propagation
result = harmonic_function_label_propagation(G, labeled_nodes)

# Visualize the result
plt.figure(figsize=(10, 8))
nx.draw(G, pos, node_color=[result[node] for node in G.nodes()],
        cmap=plt.cm.RdYlBu, with_labels=True, node_size=500)
plt.title("Harmonic Function Label Propagation")
plt.show()
```

Slide 14: Harmonic Functions in Computer Graphics

In computer graphics, harmonic functions are used for various tasks such as mesh smoothing, texture mapping, and shape interpolation. Let's implement a simple mesh smoothing algorithm using harmonic functions.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def laplacian_smoothing(vertices, faces, num_iterations=10):
    for _ in range(num_iterations):
        new_vertices = np.zeros_like(vertices)
        count = np.zeros(len(vertices))
        
        for face in faces:
            for i in range(3):
                new_vertices[face[i]] += vertices[face[(i+1)%3]] + vertices[face[(i+2)%3]]
                count[face[i]] += 2
        
        new_vertices /= count[:, np.newaxis]
        vertices = new_vertices
    
    return vertices

# Create a simple mesh (a bumpy plane)
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)
Z = 0.3 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)

vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
faces = []
for i in range(19):
    for j in range(19):
        faces.append([i*20+j, i*20+j+1, (i+1)*20+j])
        faces.append([i*20+j+1, (i+1)*20+j+1, (i+1)*20+j])

# Apply Laplacian smoothing
smoothed_vertices = laplacian_smoothing(vertices, faces)

# Visualize the results
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=faces, cmap='viridis')
ax1.set_title("Original Mesh")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_trisurf(smoothed_vertices[:,0], smoothed_vertices[:,1], smoothed_vertices[:,2], triangles=faces, cmap='viridis')
ax2.set_title("Smoothed Mesh")

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of Harmonic Function Theory, consider the following resources:

1. ArXiv paper: "Harmonic Function Theory and Applications" URL: [https://arxiv.org/abs/1901.04945](https://arxiv.org/abs/1901.04945)
2. ArXiv paper: "Discrete Harmonic Functions" URL: [https://arxiv.org/abs/math/0110024](https://arxiv.org/abs/math/0110024)
3. ArXiv paper: "Harmonic functions on graphs and manifolds" URL: [https://arxiv.org/abs/1804.04644](https://arxiv.org/abs/1804.04644)

These papers provide in-depth discussions on various aspects of harmonic function theory and its applications in different fields of mathematics and physics.

Remember to verify these links, as ArXiv URLs may change over time. For the most up-to-date information, visit the ArXiv.org website and search for papers on harmonic function theory.

