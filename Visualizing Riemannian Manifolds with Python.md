## Visualizing Riemannian Manifolds with Python

Slide 1: Introduction to Riemannian Manifolds

Riemannian manifolds are fundamental structures in differential geometry, combining the concepts of smooth manifolds with inner product spaces. They provide a framework for studying curved spaces and generalizing concepts from Euclidean geometry.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_manifold(u, v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
U, V = np.meshgrid(u, v)
X, Y, Z = sphere_manifold(U, V)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.title("Sphere: A Simple Riemannian Manifold")
plt.show()
```

Slide 2: Smooth Manifolds

A smooth manifold is a topological space that locally resembles Euclidean space and allows for calculus to be performed. It forms the underlying structure of a Riemannian manifold.

```python
import sympy as sp

def chart_transition(x, y):
    # Example of a chart transition function
    u = sp.log(x**2 + y**2)
    v = sp.atan2(y, x)
    return u, v

x, y = sp.symbols('x y')
u, v = chart_transition(x, y)

print("Chart transition function:")
print(f"u = {u}")
print(f"v = {v}")

# Compute the Jacobian
J = sp.Matrix([[sp.diff(u, x), sp.diff(u, y)],
               [sp.diff(v, x), sp.diff(v, y)]])

print("\nJacobian matrix:")
print(J)
```

Slide 3: Tangent Spaces and Vectors

The tangent space at a point on a manifold is the vector space containing all possible directions in which one can tangentially pass through the point. Tangent vectors represent velocities of curves passing through the point.

```python
import numpy as np
import matplotlib.pyplot as plt

def tangent_vector(t):
    # Parametric curve on a manifold
    x = np.cos(t)
    y = np.sin(t)
    return x, y

def tangent_line(t0, direction, scale=1):
    x0, y0 = tangent_vector(t0)
    dx, dy = direction
    return [x0, x0 + scale*dx], [y0, y0 + scale*dy]

t = np.linspace(0, 2*np.pi, 100)
x, y = tangent_vector(t)

plt.figure(figsize=(8, 8))
plt.plot(x, y)

t0 = np.pi/4
direction = [-np.sin(t0), np.cos(t0)]
tx, ty = tangent_line(t0, direction)

plt.plot(tx, ty, 'r-', linewidth=2)
plt.plot(tx[0], ty[0], 'ro')

plt.title("Tangent Vector on a Circle")
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 4: Riemannian Metric

A Riemannian metric is a smooth assignment of inner products to each tangent space, allowing us to measure distances and angles on the manifold.

```python
import numpy as np

def riemannian_metric(x, y):
    # Example metric for a 2D manifold
    return np.array([[1 + x**2, x*y],
                     [x*y, 1 + y**2]])

def inner_product(v1, v2, g):
    return v1.T @ g @ v2

# Example usage
x, y = 1, 2
g = riemannian_metric(x, y)

v1 = np.array([1, 0])
v2 = np.array([0, 1])

ip = inner_product(v1, v2, g)
print(f"Inner product of v1 and v2 at point ({x}, {y}): {ip}")

# Compute length of a vector
length = np.sqrt(inner_product(v1, v1, g))
print(f"Length of v1 at point ({x}, {y}): {length}")
```

Slide 5: Geodesics

Geodesics are curves on a Riemannian manifold that locally minimize the distance between points. They generalize the concept of straight lines to curved spaces.

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def geodesic_equation(state, t, a, b):
    x, y, dx, dy = state
    d2x = -a * (dx**2 + dy**2) / (1 + a*x**2 + b*y**2)
    d2y = -b * (dx**2 + dy**2) / (1 + a*x**2 + b*y**2)
    return [dx, dy, d2x, d2y]

a, b = 1, 1  # Parameters for the metric
initial_state = [0, 0, 1, 1]  # Initial position and velocity
t = np.linspace(0, 10, 1000)

solution = odeint(geodesic_equation, initial_state, t, args=(a, b))

plt.figure(figsize=(10, 8))
plt.plot(solution[:, 0], solution[:, 1])
plt.title("Geodesic on a 2D Riemannian Manifold")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
```

Slide 6: Curvature

Curvature measures how a Riemannian manifold deviates from being flat. The Riemann curvature tensor encodes this information and is crucial for understanding the geometry of the manifold.

```python
import sympy as sp

def riemann_tensor_2d(g):
    x, y = sp.symbols('x y')
    R = sp.zeros(2, 2, 2, 2)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    term1 = sp.diff(g[i, l], y) + sp.diff(g[j, l], x) - sp.diff(g[i, j], y)
                    term2 = sp.diff(g[i, k], x) + sp.diff(g[j, k], y) - sp.diff(g[k, l], x)
                    R[i, j, k, l] = 0.5 * (term1 - term2)
    
    return R

# Example metric
x, y = sp.symbols('x y')
g = sp.Matrix([[1 + x**2, x*y],
               [x*y, 1 + y**2]])

R = riemann_tensor_2d(g)

print("Riemann curvature tensor components:")
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                if R[i, j, k, l] != 0:
                    print(f"R_{i}{j}{k}{l} = {R[i, j, k, l]}")
```

Slide 7: Parallel Transport

Parallel transport is a way of moving vectors along curves on a manifold while preserving their length and angle with the curve. It's fundamental for comparing vectors at different points.

```python
import numpy as np
import matplotlib.pyplot as plt

def parallel_transport_circle(theta, v0):
    # Parallel transport on a unit circle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ v0

theta = np.linspace(0, 2*np.pi, 100)
v0 = np.array([1, 0])  # Initial vector

x = np.cos(theta)
y = np.sin(theta)

plt.figure(figsize=(10, 10))
plt.plot(x, y, 'b-')

for t in np.linspace(0, 2*np.pi, 8):
    v = parallel_transport_circle(t, v0)
    plt.arrow(np.cos(t), np.sin(t), v[0]*0.2, v[1]*0.2, 
              head_width=0.05, head_length=0.08, fc='r', ec='r')

plt.title("Parallel Transport on a Circle")
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 8: Covariant Derivative

The covariant derivative extends the notion of directional derivatives to Riemannian manifolds, taking into account the manifold's curvature.

```python
import sympy as sp

def christoffel_symbols(g):
    x, y = sp.symbols('x y')
    gamma = sp.zeros(2, 2, 2)
    g_inv = g.inv()
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                gamma[i, j, k] = 0
                for l in range(2):
                    gamma[i, j, k] += 0.5 * g_inv[i, l] * (sp.diff(g[l, j], sp.symbols[k]) + 
                                                           sp.diff(g[l, k], sp.symbols[j]) - 
                                                           sp.diff(g[j, k], sp.symbols[l]))
    return gamma

# Example metric
x, y = sp.symbols('x y')
g = sp.Matrix([[1 + x**2, x*y],
               [x*y, 1 + y**2]])

gamma = christoffel_symbols(g)

print("Christoffel symbols:")
for i in range(2):
    for j in range(2):
        for k in range(2):
            if gamma[i, j, k] != 0:
                print(f"Γ_{i}{j}{k} = {gamma[i, j, k]}")
```

Slide 9: Riemannian Volume Form

The Riemannian volume form allows us to integrate functions over Riemannian manifolds, generalizing the concept of volume integrals from Euclidean space.

```python
import sympy as sp
import numpy as np

def riemannian_volume_form(g):
    return sp.sqrt(g.det())

# Example: volume form on a sphere
theta, phi = sp.symbols('theta phi')
g = sp.Matrix([[1, 0],
               [0, sp.sin(theta)**2]])

dV = riemannian_volume_form(g)
print(f"Volume form on a sphere: {dV}")

# Numerical integration example
def sphere_volume(R):
    return 4 * np.pi * R**3 / 3

def numerical_sphere_volume(R, n=1000):
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2*np.pi, n)
    dtheta = np.pi / n
    dphi = 2 * np.pi / n
    
    volume = 0
    for t in theta:
        for p in phi:
            volume += R**2 * np.sin(t) * dtheta * dphi
    
    return R * volume

R = 1  # radius
exact_vol = sphere_volume(R)
num_vol = numerical_sphere_volume(R)

print(f"Exact volume: {exact_vol}")
print(f"Numerical volume: {num_vol}")
print(f"Relative error: {abs(exact_vol - num_vol) / exact_vol:.6f}")
```

Slide 10: Isometries and Killing Vector Fields

Isometries are distance-preserving maps between Riemannian manifolds. Killing vector fields generate one-parameter groups of isometries and are important in studying symmetries of manifolds.

```python
import sympy as sp

def killing_equation(X, g):
    x, y = sp.symbols('x y')
    eq = sp.zeros(2, 2)
    
    for i in range(2):
        for j in range(2):
            eq[i, j] = X[0] * sp.diff(g[i, j], x) + X[1] * sp.diff(g[i, j], y)
            eq[i, j] += sp.diff(X[i], sp.symbols[j]) + sp.diff(X[j], sp.symbols[i])
    
    return eq

# Example: Killing vector fields on a sphere
theta, phi = sp.symbols('theta phi')
g = sp.Matrix([[1, 0],
               [0, sp.sin(theta)**2]])

# Rotational symmetry around z-axis
X = sp.Matrix([0, 1])

eq = killing_equation(X, g)
print("Killing equation for rotation around z-axis:")
print(eq)

# Verify if X is a Killing vector field
is_killing = eq.equals(sp.zeros(2, 2))
print(f"Is X a Killing vector field? {is_killing}")
```

Slide 11: Einstein Field Equations

The Einstein field equations describe the fundamental interaction of gravitation as a result of spacetime being curved by matter and energy. They form the core of general relativity, which can be formulated using Riemannian geometry.

```python
import sympy as sp

def einstein_tensor(g):
    x, y = sp.symbols('x y')
    R = sp.zeros(2, 2)  # Ricci tensor
    G = sp.zeros(2, 2)  # Einstein tensor
    
    # Compute Ricci tensor (simplified 2D version)
    for i in range(2):
        for j in range(2):
            R[i, j] = sp.diff(sp.diff(g[i, j], x), x) + sp.diff(sp.diff(g[i, j], y), y)
    
    # Compute scalar curvature
    R_scalar = sum(g.inv()[i, j] * R[i, j] for i in range(2) for j in range(2))
    
    # Compute Einstein tensor
    for i in range(2):
        for j in range(2):
            G[i, j] = R[i, j] - 0.5 * g[i, j] * R_scalar
    
    return G

# Example metric (simplified 2D spacetime)
t, r = sp.symbols('t r')
g = sp.Matrix([[-1, 0],
               [0, 1 / (1 - 2/r)]])

G = einstein_tensor(g)
print("Einstein tensor components:")
sp.pprint(G)
```

Slide 12: Applications in Machine Learning

Riemannian geometry has found applications in machine learning, particularly in areas like manifold learning, dimensionality reduction, and optimization on manifolds.

```python
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt

# Generate Swiss roll dataset
n_samples = 1500
noise = 0.05
X, color = manifold.make_swiss_roll(n_samples, noise=noise)

# Apply Isomap for dimensionality reduction
n_neighbors = 10
n_components = 2
isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
X_iso = isomap.fit_transform(X)

# Plot results
fig = plt.figure(figsize=(15, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.viridis)
ax1.set_title("Original Swiss Roll in 3D")

ax2 = fig.add_subplot(122)
ax2.scatter(X_iso[:, 0], X_iso[:, 1], c=color, cmap=plt.cm.viridis)
ax2.set_title("Swiss Roll unrolled by Isomap")

plt.show()
```

Slide 13: Riemannian Optimization

Optimization on Riemannian manifolds is crucial for many machine learning tasks, such as matrix completion and dimensionality reduction. Here's a simple example of gradient descent on a sphere.

```python
import numpy as np
import matplotlib.pyplot as plt

def proj_sphere(x):
    return x / np.linalg.norm(x)

def f(x):
    return x[0]**2 + 2*x[1]**2 + 3*x[2]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1], 6*x[2]])

def riemannian_gradient_descent(x0, learning_rate, num_iterations):
    x = x0
    path = [x]
    for _ in range(num_iterations):
        grad = grad_f(x)
        riem_grad = grad - np.dot(grad, x) * x
        x = proj_sphere(x - learning_rate * riem_grad)
        path.append(x)
    return np.array(path)

x0 = proj_sphere(np.array([1.0, 1.0, 1.0]))
learning_rate = 0.1
num_iterations = 50

path = riemannian_gradient_descent(x0, learning_rate, num_iterations)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)
ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r.-')
ax.set_title("Riemannian Gradient Descent on a Sphere")
plt.show()
```

Slide 14: Connections to Physics

Riemannian geometry is fundamental to general relativity and other areas of physics. Here's a simple visualization of spacetime curvature near a massive object.

```python
import numpy as np
import matplotlib.pyplot as plt

def schwarzschild_metric(r, M=1):
    return 1 - 2*M/r

def plot_curved_spacetime(M=1, r_max=10):
    r = np.linspace(2*M + 0.1, r_max, 1000)
    g_tt = schwarzschild_metric(r, M)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(r, g_tt)
    ax.set_xlabel('r')
    ax.set_ylabel('g_tt')
    ax.set_title('Schwarzschild Metric Component g_tt')
    ax.grid(True)
    
    # Add visual representation of spacetime curvature
    for y in np.linspace(0.2, 1, 5):
        ax.plot(r, y - 0.2*(1-g_tt), 'k--', alpha=0.3)
    
    plt.show()

plot_curved_spacetime(M=1, r_max=10)
```

Slide 15: Additional Resources

For those interested in diving deeper into Riemannian geometry and its applications, here are some valuable resources:

1. ArXiv.org papers:
   * "A Survey of Riemannian Geometry for Machine Learning" by Nikhil Ghosh et al. (2020) ArXiv link: [https://arxiv.org/abs/2004.06992](https://arxiv.org/abs/2004.06992)
   * "Riemannian Geometry for Machine Learning" by Suvrit Sra and Reshad Hosseini (2015) ArXiv link: [https://arxiv.org/abs/1511.07428](https://arxiv.org/abs/1511.07428)
2. Online courses:
   * Coursera: "Differential Geometry" by National Research University Higher School of Economics
   * edX: "General Relativity" by MIT
3. Textbooks:
   * "Introduction to Riemannian Manifolds" by John M. Lee
   * "Riemannian Geometry: A Modern Introduction" by Isaac Chavel
4. Software libraries:
   * Geomstats: Python package for computations and statistics on manifolds
   * PyManopt: Python toolbox for optimization on Riemannian manifolds

These resources provide a mix of theoretical foundations and practical applications of Riemannian geometry in various fields.

