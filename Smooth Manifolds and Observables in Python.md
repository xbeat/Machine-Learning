## Smooth Manifolds and Observables in Python
Slide 1: Introduction to Smooth Manifolds

Smooth manifolds are fundamental objects in differential geometry, generalizing the concept of curves and surfaces to higher dimensions. They provide a framework for studying geometric structures that locally resemble Euclidean space.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_coordinates(u, v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)

x, y, z = sphere_coordinates(u, v)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_title('Sphere: Example of a 2D Smooth Manifold')
plt.show()
```

Slide 2: Local Coordinates and Charts

A smooth manifold is equipped with local coordinate systems called charts. These charts allow us to describe the manifold locally using familiar Euclidean coordinates.

```python
import numpy as np
import matplotlib.pyplot as plt

def stereographic_projection(x, y, z):
    u = x / (1 - z)
    v = y / (1 - z)
    return u, v

theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

u, v = stereographic_projection(x, y, z)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot_surface(x, y, z, cmap='viridis')
ax1.set_title('Sphere')
ax2.plot(u, v, 'b.', alpha=0.1)
ax2.set_title('Stereographic Projection (Chart)')
plt.show()
```

Slide 3: Tangent Spaces and Vectors

Tangent spaces are crucial in understanding the local structure of smooth manifolds. They represent the space of all possible directions in which one can move on the manifold at a given point.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_point(theta, phi):
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

def tangent_vectors(theta, phi):
    p = sphere_point(theta, phi)
    v1 = np.array([
        -np.sin(theta),
        np.cos(theta),
        0
    ])
    v2 = np.array([
        np.cos(theta) * np.cos(phi),
        np.sin(theta) * np.cos(phi),
        -np.sin(phi)
    ])
    return p, v1, v2

theta, phi = np.pi/4, np.pi/3
p, v1, v2 = tangent_vectors(theta, phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='b', alpha=0.2)
ax.quiver(*p, *v1, color='r', length=0.2)
ax.quiver(*p, *v2, color='g', length=0.2)
ax.set_title('Tangent Vectors on a Sphere')
plt.show()
```

Slide 4: Smooth Functions on Manifolds

Smooth functions on manifolds are essential for defining various geometric concepts. These functions should be differentiable when composed with chart maps.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_to_plane(theta, phi):
    return theta, phi

def height_function(theta, phi):
    return np.cos(phi)

theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

h = height_function(theta, phi)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, facecolors=plt.cm.viridis(h), alpha=0.7)
ax1.set_title('Height Function on Sphere')

ax2 = fig.add_subplot(122)
im = ax2.imshow(h, extent=[0, 2*np.pi, 0, np.pi], origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Height Function in Chart Coordinates')
ax2.set_xlabel('θ')
ax2.set_ylabel('φ')
plt.colorbar(im)

plt.show()
```

Slide 5: Differentiable Maps and Pushforwards

Differentiable maps between manifolds induce linear maps between their tangent spaces, called pushforwards. These are crucial for understanding how geometric structures transform under maps.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_to_cylinder(theta, phi):
    return theta, np.cos(phi)

def pushforward(theta, phi):
    return np.array([[1, 0], [0, -np.sin(phi)]])

theta = np.linspace(0, 2*np.pi, 20)
phi = np.linspace(0, np.pi, 10)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

u, v = sphere_to_cylinder(theta, phi)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, alpha=0.7)
ax1.set_title('Sphere')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(u, v, np.zeros_like(u), alpha=0.7)
ax2.set_title('Cylinder')

for i in range(5):
    for j in range(5):
        t, p = theta[i,j], phi[i,j]
        pf = pushforward(t, p)
        v1, v2 = pf @ np.eye(2)
        ax1.quiver(x[i,j], y[i,j], z[i,j], *v1, 0, color='r', length=0.1)
        ax1.quiver(x[i,j], y[i,j], z[i,j], *v2, 0, color='g', length=0.1)
        ax2.quiver(u[i,j], v[i,j], 0, *v1, 0, color='r', length=0.1)
        ax2.quiver(u[i,j], v[i,j], 0, *v2, 0, color='g', length=0.1)

plt.show()
```

Slide 6: Differential Forms

Differential forms are antisymmetric, multilinear maps on tangent spaces. They provide a coordinate-independent way to integrate over manifolds.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_coordinates(u, v):
    return np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])

def area_form(u, v):
    return np.sin(v)

u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 20)
u, v = np.meshgrid(u, v)

x, y, z = sphere_coordinates(u, v)
omega = area_form(u, v)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(omega), alpha=0.7)
ax.set_title('Area Form on Sphere')

plt.colorbar(surf, ax=ax, label='Magnitude of Area Form')
plt.show()

# Compute the total area of the sphere
total_area = np.sum(omega) * (2*np.pi/30) * (np.pi/20)
print(f"Computed area of the sphere: {total_area:.4f}")
print(f"Actual area of unit sphere: {4*np.pi:.4f}")
```

Slide 7: Vector Fields and Flows

Vector fields assign a tangent vector to each point on a manifold. They generate flows, which are one-parameter families of diffeomorphisms.

```python
import numpy as np
import matplotlib.pyplot as plt

def vector_field(x, y):
    return -y, x

def flow(x0, y0, t):
    return x0 * np.cos(t) - y0 * np.sin(t), x0 * np.sin(t) + y0 * np.cos(t)

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

U, V = vector_field(X, Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.quiver(X, Y, U, V)
ax1.set_title('Vector Field')

for x0, y0 in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
    t = np.linspace(0, 2*np.pi, 100)
    x, y = flow(x0, y0, t)
    ax2.plot(x, y)

ax2.set_title('Flow of the Vector Field')
plt.show()
```

Slide 8: Lie Groups and Lie Algebras

Lie groups are smooth manifolds with a compatible group structure. Their tangent space at the identity forms a Lie algebra, which captures the local structure of the group.

```python
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def exp_map(A):
    return np.linalg.matrix_power(np.eye(2) + A/1000, 1000)

theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(circle_x, circle_y)
ax1.set_title('SO(2) Lie Group')
ax1.set_aspect('equal')

# Lie algebra elements
X = np.array([[0, -1], [1, 0]])
Y = np.array([[0, -2], [2, 0]])

t = np.linspace(0, 1, 100)
exp_tX = np.array([exp_map(t_i * X) for t_i in t])
exp_tY = np.array([exp_map(t_i * Y) for t_i in t])

ax2.plot(exp_tX[:, 0, 0], exp_tX[:, 1, 0], label='exp(tX)')
ax2.plot(exp_tY[:, 0, 0], exp_tY[:, 1, 0], label='exp(tY)')
ax2.set_title('Exponential Map from Lie Algebra to Lie Group')
ax2.legend()
ax2.set_aspect('equal')

plt.show()
```

Slide 9: Riemannian Metrics

Riemannian metrics define a notion of distance and angle on manifolds, allowing us to measure lengths, areas, and volumes.

```python
import numpy as np
import matplotlib.pyplot as plt

def metric_tensor(u, v):
    return np.array([[1, 0], [0, np.sin(u)**2]])

def geodesic(u0, v0, du, dv, t):
    u = u0 + du * t
    v = v0 + dv * t / np.sin(u0)
    return u, v

u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.7)
ax1.set_title('Sphere with Geodesics')

# Plot some geodesics
for u0, v0, du, dv in [(np.pi/4, 0, 0, 1), (np.pi/2, 0, 1, 1), (np.pi/4, np.pi/2, 1, 0)]:
    t = np.linspace(0, 2*np.pi, 100)
    u, v = geodesic(u0, v0, du, dv, t)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax1.plot(x, y, z, color='r')

ax2 = fig.add_subplot(122)
im = ax2.imshow(metric_tensor(U, V)[1,1], extent=[0, 2*np.pi, 0, np.pi], 
                origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Metric Tensor Component g_φφ')
ax2.set_xlabel('φ')
ax2.set_ylabel('θ')
plt.colorbar(im)

plt.show()
```

Slide 10: Connections and Parallel Transport

Connections provide a way to compare tangent vectors at different points on a manifold, enabling the concept of parallel transport.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_point(theta, phi):
    return np.array([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)])

def parallel_transport(theta, phi, v, t):
    # Simplified parallel transport along a great circle
    rotation = np.array([[np.cos(t), -np.sin(t)],
                         [np.sin(t), np.cos(t)]])
    return rotation @ v

# Generate points on a great circle
theta = np.linspace(0, 2*np.pi, 100)
phi = np.pi/2  # Equator
points = np.array([sphere_point(t, phi) for t in theta])

# Initial vector to transport
v0 = np.array([0, 1])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Plot the great circle and transported vectors
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r')
for i in range(0, len(theta), 10):
    p = points[i]
    v = parallel_transport(theta[i], phi, v0, theta[i])
    ax.quiver(p[0], p[1], p[2], v[0], v[1], 0, color='g', length=0.1)

ax.set_title('Parallel Transport on a Sphere')
plt.show()
```

Slide 11: Curvature

Curvature measures how a manifold deviates from being flat. It can be expressed through the Riemann curvature tensor, which quantifies the failure of parallel transport to be path-independent.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_curvature(u, v):
    # Gaussian curvature of a sphere is constant
    return np.ones_like(u)

def sectional_curvature(u, v):
    # Sectional curvature of a sphere is also constant
    return np.ones_like(u)

u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

K = gaussian_curvature(U, V)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(K), alpha=0.7)
ax1.set_title('Sphere colored by Gaussian Curvature')

ax2 = fig.add_subplot(122)
im = ax2.imshow(K, extent=[0, 2*np.pi, 0, np.pi], origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Gaussian Curvature Map')
ax2.set_xlabel('φ')
ax2.set_ylabel('θ')
plt.colorbar(im)

plt.show()
```

Slide 12: Geodesics

Geodesics are curves that locally minimize the distance between points on a manifold. They generalize the concept of straight lines to curved spaces.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def geodesic_equation(y, t, a, b):
    theta, phi, dtheta, dphi = y
    d2theta = 2 * np.tan(phi) * dtheta * dphi
    d2phi = -np.sin(phi) * np.cos(phi) * dtheta**2
    return [dtheta, dphi, d2theta, d2phi]

def solve_geodesic(theta0, phi0, dtheta0, dphi0, t):
    y0 = [theta0, phi0, dtheta0, dphi0]
    solution = odeint(geodesic_equation, y0, t, args=(0, 0))
    return solution[:, 0], solution[:, 1]

t = np.linspace(0, 10, 1000)
theta, phi = solve_geodesic(0, np.pi/2, 1, 0, t)

X = np.sin(phi) * np.cos(theta)
Y = np.sin(phi) * np.sin(theta)
Z = np.cos(phi)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Plot the geodesic
ax.plot(X, Y, Z, color='r', linewidth=2)
ax.set_title('Geodesic on a Sphere')

plt.show()
```

Slide 13: Real-life Example: Earth's Surface

Earth's surface can be approximated as a smooth manifold. Understanding its geometry is crucial for navigation, cartography, and geophysics.

```python
import numpy as np
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Example: Distance between New York and Tokyo
ny_lat, ny_lon = 40.7128, -74.0060
tokyo_lat, tokyo_lon = 35.6762, 139.6503

distance = haversine_distance(ny_lat, ny_lon, tokyo_lat, tokyo_lon)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Earth
phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2*np.pi, 100)
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones_like(theta))

ax.plot_surface(x, y, z, color='b', alpha=0.3)

# Plot New York and Tokyo
ny = np.array([np.cos(np.radians(ny_lat)) * np.cos(np.radians(ny_lon)),
               np.cos(np.radians(ny_lat)) * np.sin(np.radians(ny_lon)),
               np.sin(np.radians(ny_lat))])
tokyo = np.array([np.cos(np.radians(tokyo_lat)) * np.cos(np.radians(tokyo_lon)),
                  np.cos(np.radians(tokyo_lat)) * np.sin(np.radians(tokyo_lon)),
                  np.sin(np.radians(tokyo_lat))])

ax.scatter(*ny, color='r', s=50, label='New York')
ax.scatter(*tokyo, color='g', s=50, label='Tokyo')

# Plot the geodesic
t = np.linspace(0, 1, 100)
path = np.outer(1-t, ny) + np.outer(t, tokyo)
path /= np.linalg.norm(path, axis=1)[:, np.newaxis]
ax.plot(*path.T, color='r', linewidth=2)

ax.set_title(f'Geodesic on Earth: NY to Tokyo (Distance: {distance:.2f} km)')
ax.legend()

plt.show()
```

Slide 14: Real-life Example: General Relativity

Einstein's theory of General Relativity describes gravity as the curvature of a 4-dimensional spacetime manifold. This example illustrates a simplified model of spacetime curvature.

```python
import numpy as np
import matplotlib.pyplot as plt

def schwarzschild_metric(r, M=1):
    c = 1  # Speed of light
    Rs = 2 * M * c**2  # Schwarzschild radius
    g00 = -(1 - Rs/r)
    g11 = 1 / (1 - Rs/r)
    g22 = r**2
    g33 = r**2 * np.sin(np.pi/4)**2  # Fixed θ = π/4 for simplicity
    return np.diag([g00, g11, g22, g33])

r = np.linspace(2.1, 10, 100)  # Start slightly outside event horizon
metric_components = np.array([schwarzschild_metric(ri) for ri in r])

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Schwarzschild Metric Components")

for i in range(4):
    row = i // 2
    col = i % 2
    ax[row, col].plot(r, metric_components[:, i, i])
    ax[row, col].set_xlabel('r')
    ax[row, col].set_ylabel(f'g{i}{i}')
    ax[row, col].set_title(f'Metric Component g{i}{i}')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into smooth manifolds and related topics, here are some valuable resources:

1. "Introduction to Smooth Manifolds" by John M. Lee ArXiv: [https://arxiv.org/abs/math/9940009](https://arxiv.org/abs/math/9940009)
2. "Differential Geometry of Curves and Surfaces" by Manfredo P. do Carmo (Not available on ArXiv, but widely used textbook)
3. "Notes on Differential Geometry and Lie Groups" by Jean Gallier ArXiv: [https://arxiv.org/abs/0805.0287](https://arxiv.org/abs/0805.0287)
4. "Riemannian Geometry: A Modern Introduction" by Isaac Chavel ArXiv: [https://arxiv.org/abs/math/0306138](https://arxiv.org/abs/math/0306138)

These resources provide a more in-depth exploration of the concepts covered in this presentation, offering rigorous mathematical treatments and advanced topics in differential geometry.

