## Differential Geometry Visualizations with Python

Slide 1: Introduction to Differential Geometry

Differential geometry is a mathematical discipline that uses techniques of differential calculus, integral calculus, linear algebra, and multilinear algebra to study problems in geometry. It deals with smooth shapes and spaces, known as manifolds.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere(u, v):
    r = 1
    x = r * np.cos(u) * np.sin(v)
    y = r * np.sin(u) * np.sin(v)
    z = r * np.cos(v)
    return x, y, z

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)
x, y, z = sphere(u, v)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='b', alpha=0.3)
plt.show()
```

Slide 2: Curves in Space

A curve in space is a continuous function from an interval of real numbers to a space. In differential geometry, we study smooth curves, which have continuous derivatives up to some desired order.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def helix(t):
    x = np.cos(t)
    y = np.sin(t)
    z = t
    return x, y, z

t = np.linspace(0, 10 * np.pi, 1000)
x, y, z = helix(t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
plt.show()
```

Slide 3: Tangent Vectors and the Tangent Space

The tangent space at a point on a manifold is the vector space containing all possible directions in which one can tangentially pass through the point. Tangent vectors are elements of this space.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def tangent_line(x0, x):
    y0 = f(x0)
    slope = 2 * x0
    return slope * (x - x0) + y0

x = np.linspace(-2, 2, 100)
y = f(x)

x0 = 1
y0 = f(x0)

plt.plot(x, y, label='f(x) = x^2')
plt.plot(x, tangent_line(x0, x), '--', label=f'Tangent at x={x0}')
plt.plot(x0, y0, 'ro')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Curvature of Plane Curves

Curvature measures how quickly a curve changes direction. For a plane curve, it's defined as the rate of change of the tangent vector with respect to arc length.

```python
import numpy as np
import matplotlib.pyplot as plt

def circle(t, r=1):
    return r * np.cos(t), r * np.sin(t)

def curvature(t, r=1):
    return 1 / r

t = np.linspace(0, 2*np.pi, 100)
x, y = circle(t)
k = curvature(t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x, y)
ax1.set_aspect('equal')
ax1.set_title('Circle')

ax2.plot(t, k)
ax2.set_title('Curvature')

plt.tight_layout()
plt.show()
```

Slide 5: Frenet-Serret Frame

The Frenet-Serret frame is a moving reference frame of three orthonormal vectors used to describe a curve locally at each point. It consists of the tangent, normal, and binormal unit vectors.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def helix(t):
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2 * np.pi)
    return np.array([x, y, z])

def frenet_frame(t):
    r = helix(t)
    T = np.array([-np.sin(t), np.cos(t), 1 / (2 * np.pi)])
    T /= np.linalg.norm(T)
    B = np.array([-np.cos(t), -np.sin(t), 0])
    N = np.cross(B, T)
    return r, T, N, B

t = np.linspace(0, 4 * np.pi, 100)
points = np.array([helix(ti) for ti in t])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(*points.T)

for ti in np.linspace(0, 4 * np.pi, 10):
    r, T, N, B = frenet_frame(ti)
    ax.quiver(*r, *T, color='r', length=0.5)
    ax.quiver(*r, *N, color='g', length=0.5)
    ax.quiver(*r, *B, color='b', length=0.5)

plt.show()
```

Slide 6: Surfaces and Parametrizations

A surface is a two-dimensional manifold embedded in three-dimensional space. Parametrizations allow us to describe surfaces using two parameters.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def torus(u, v, R=2, r=1):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return x, y, z

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)
x, y, z = torus(u, v)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
plt.show()
```

Slide 7: First Fundamental Form

The first fundamental form is a quadratic form that allows measurements on a surface (such as length and angle) to be determined without reference to the ambient space.

```python
import numpy as np
import sympy as sp

# Define symbolic variables
u, v = sp.symbols('u v')

# Define a parametrization of a sphere
x = sp.sin(u) * sp.cos(v)
y = sp.sin(u) * sp.sin(v)
z = sp.cos(u)

# Compute partial derivatives
xu = sp.diff(x, u)
xv = sp.diff(x, v)
yu = sp.diff(y, u)
yv = sp.diff(y, v)
zu = sp.diff(z, u)
zv = sp.diff(z, v)

# Compute coefficients of the first fundamental form
E = xu**2 + yu**2 + zu**2
F = xu*xv + yu*yv + zu*zv
G = xv**2 + yv**2 + zv**2

print("First Fundamental Form:")
print(f"E = {E}")
print(f"F = {F}")
print(f"G = {G}")
```

Slide 8: Gaussian Curvature

Gaussian curvature is an intrinsic measure of curvature that depends on how the surface is curved, without reference to an embedding space.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_curvature(x, y):
    return -2 / (1 + x**2 + y**2)**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = gaussian_curvature(X, Y)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Gaussian Curvature')

ax2 = fig.add_subplot(122)
im = ax2.imshow(Z, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
plt.colorbar(im)
ax2.set_title('Gaussian Curvature (Top View)')

plt.tight_layout()
plt.show()
```

Slide 9: Mean Curvature

Mean curvature is an extrinsic measure of curvature that depends on the embedding of the surface in three-dimensional space. It's the average of the principal curvatures.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mean_curvature(x, y):
    return 2 * (1 + x**2 + y**2) / (1 + x**2 + y**2)**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = mean_curvature(X, Y)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Mean Curvature')

ax2 = fig.add_subplot(122)
im = ax2.imshow(Z, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
plt.colorbar(im)
ax2.set_title('Mean Curvature (Top View)')

plt.tight_layout()
plt.show()
```

Slide 10: Geodesics

Geodesics are curves on a surface that locally minimize the distance between points. They are the generalization of straight lines to curved spaces.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere(u, v):
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    return x, y, z

def geodesic(t):
    return np.pi/2, t

u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)
X, Y, Z = sphere(U, V)

t = np.linspace(0, 2*np.pi, 100)
u_geo, v_geo = geodesic(t)
X_geo, Y_geo, Z_geo = sphere(u_geo, v_geo)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.3)
ax.plot(X_geo, Y_geo, Z_geo, color='r', linewidth=3)
plt.show()
```

Slide 11: Parallel Transport

Parallel transport is a way of transporting geometric data along a curve on a manifold. It preserves the inner product of vectors and is used to compare vectors at different points on the manifold.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere(u, v):
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    return x, y, z

def parallel_transport(t, v0):
    u = np.pi/2
    x = np.cos(t)
    y = np.sin(t)
    z = 0
    T = np.array([-y, x, 0])
    N = np.array([0, 0, 1])
    B = np.cross(T, N)
    return x*v0[0]*T + y*v0[0]*B + v0[1]*N

u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)
X, Y, Z = sphere(U, V)

t = np.linspace(0, 2*np.pi, 20)
v0 = np.array([1, 0])
vectors = [parallel_transport(ti, v0) for ti in t]
points = [sphere(np.pi/2, ti) for ti in t]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, alpha=0.3)

for point, vector in zip(points, vectors):
    ax.quiver(*point, *vector, color='r', length=0.2)

plt.show()
```

Slide 12: Riemannian Manifolds

A Riemannian manifold is a smooth manifold equipped with an inner product on the tangent space at each point, which varies smoothly from point to point. This inner product is called the Riemannian metric.

```python
import numpy as np
import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y')

# Define a Riemannian metric on R^2
g11 = 1 + x**2
g12 = g21 = x*y
g22 = 1 + y**2

# Create the metric tensor
g = sp.Matrix([[g11, g12], [g21, g22]])

# Compute the determinant and inverse of the metric
det_g = g.det()
g_inv = g.inv()

print("Metric tensor:")
print(g)
print("\nDeterminant of metric:")
print(det_g)
print("\nInverse metric:")
print(g_inv)

# Compute Christoffel symbols (first kind)
christoffel = [[[sp.diff(g[i,j], x_k) + sp.diff(g[i,k], x_j) - sp.diff(g[j,k], x_i) 
                 for k in range(2)] for j in range(2)] for i in range(2)]

print("\nChristoffel symbols (first kind):")
for i in range(2):
    for j in range(2):
        for k in range(2):
            print(f"Γ_{i+1}{j+1}{k+1} = {christoffel[i][j][k]}")
```

Slide 13: Applications in Physics and Engineering

Differential geometry has numerous applications in physics and engineering, including:

1. General Relativity: Describes gravity as the curvature of spacetime.
2. Fluid Dynamics: Models the flow of fluids on curved surfaces.
3. Computer Graphics: Used in 3D modeling and animation.
4. Robotics: Helps in path planning and control of robotic arms.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def schwarzschild_metric(r, theta):
    Rs = 1  # Schwarzschild radius
    g00 = -(1 - Rs / r)
    g11 = 1 / (1 - Rs / r)
    g22 = r**2
    g33 = r**2 * np.sin(theta)**2
    return np.diag([g00, g11, g22, g33])

r = np.linspace(1.1, 5, 100)
theta = np.linspace(0, np.pi, 100)
R, THETA = np.meshgrid(r, theta)

g00 = schwarzschild_metric(R, THETA)[0, 0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(R * np.sin(THETA), R * np.cos(THETA), g00, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('g00')
ax.set_title('Schwarzschild Metric Component g00')
plt.colorbar(surf)
plt.show()
```

Slide 14: Lie Groups and Lie Algebras

Lie groups are smooth manifolds that are also groups, with smooth group operations. Lie algebras are the tangent spaces at the identity of Lie groups, equipped with a bracket operation.

```python
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def plot_so2():
    thetas = np.linspace(0, 2*np.pi, 100)
    x = np.cos(thetas)
    y = np.sin(thetas)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot SO(2)
    ax1.plot(x, y)
    ax1.set_aspect('equal')
    ax1.set_title('SO(2) - Special Orthogonal Group')
    ax1.set_xlabel('cos(θ)')
    ax1.set_ylabel('sin(θ)')
    
    # Plot action on a point
    point = np.array([1, 0])
    rotated_points = np.array([rotation_matrix(theta) @ point for theta in thetas])
    
    ax2.plot(rotated_points[:, 0], rotated_points[:, 1])
    ax2.set_aspect('equal')
    ax2.set_title('Action of SO(2) on (1, 0)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()

plot_so2()
```

Slide 15: Connections and Covariant Derivatives

Connections provide a way to compare vectors at different points on a manifold. The covariant derivative is a generalization of the directional derivative to manifolds, using a connection.

```python
import sympy as sp

# Define symbolic variables and metric
x, y = sp.symbols('x y')
g = sp.Matrix([[1 + x**2, x*y], [x*y, 1 + y**2]])

# Compute Christoffel symbols
def christoffel(g, coord):
    n = g.shape[0]
    gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    g_inv = g.inv()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                gamma[i][j][k] = sum(g_inv[i, l] * (g[l, j].diff(coord[k]) + 
                                                    g[l, k].diff(coord[j]) - 
                                                    g[j, k].diff(coord[l])) 
                                     for l in range(n)) / 2
    return gamma

gamma = christoffel(g, [x, y])

print("Christoffel symbols:")
for i in range(2):
    for j in range(2):
        for k in range(2):
            print(f"Γ^{i}_{j}{k} = {gamma[i][j][k]}")
```

Slide 16: Additional Resources

For further exploration of Differential Geometry, consider these resources:

1. ArXiv.org: "Introduction to Differential Geometry" by John M. Lee URL: [https://arxiv.org/abs/math/9805030](https://arxiv.org/abs/math/9805030)
2. ArXiv.org: "Notes on Differential Geometry and Lie Groups" by Jean Gallier URL: [https://arxiv.org/abs/1805.01462](https://arxiv.org/abs/1805.01462)
3. ArXiv.org: "Differential Geometry of Curves and Surfaces" by Manfredo P. do Carmo URL: [https://arxiv.org/abs/math/0406005](https://arxiv.org/abs/math/0406005)

These resources provide in-depth coverage of the topics we've introduced in this slideshow. Remember to verify the availability and content of these papers, as ArXiv listings may change over time.

