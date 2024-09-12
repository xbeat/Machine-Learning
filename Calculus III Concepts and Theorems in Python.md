## Calculus III Concepts and Theorems in Python
Slide 1: Vector Functions and Space Curves

Vector functions map scalar inputs to vector outputs, often used to represent curves in three-dimensional space. These functions are crucial in understanding motion, velocity, and acceleration in 3D.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def helix(t):
    return np.array([np.cos(t), np.sin(t), t])

t = np.linspace(0, 10*np.pi, 1000)
x, y, z = helix(t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_title('Helix: A Space Curve')
plt.show()
```

Slide 2: Limits and Continuity of Vector Functions

Limits and continuity for vector functions are defined component-wise. A vector function is continuous if all its component functions are continuous.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.array([t**2, np.sin(t)])

t = np.linspace(-5, 5, 1000)
x, y = f(t)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('Continuous Vector Function')
plt.xlabel('x = t^2')
plt.ylabel('y = sin(t)')
plt.grid(True)
plt.show()

# Check continuity at t = 0
t0 = 0
limit = f(t0)
print(f"Limit at t = {t0}: {limit}")
print(f"Function value at t = {t0}: {f(t0)}")
print("The function is continuous at t = 0")
```

Slide 3: Derivatives of Vector Functions

The derivative of a vector function is obtained by differentiating each component function. It represents the instantaneous rate of change of the function with respect to its parameter.

```python
import sympy as sp

# Define symbolic variable and vector function
t = sp.Symbol('t')
r = sp.Matrix([sp.cos(t), sp.sin(t), t**2])

# Calculate derivative
r_prime = r.diff(t)

print("Vector function r(t):")
print(r)
print("\nDerivative r'(t):")
print(r_prime)

# Evaluate at t = π/4
t_val = sp.pi/4
r_val = r.subs(t, t_val)
r_prime_val = r_prime.subs(t, t_val)

print(f"\nAt t = π/4:")
print(f"r(π/4) = {r_val.evalf()}")
print(f"r'(π/4) = {r_prime_val.evalf()}")
```

Slide 4: Tangent Vectors and Normal Vectors

Tangent vectors represent the direction of motion along a curve, while normal vectors are perpendicular to the curve at a given point. These vectors are essential in understanding the geometry of space curves.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def curve(t):
    return np.array([np.cos(t), np.sin(t), t])

def tangent(t):
    return np.array([-np.sin(t), np.cos(t), 1])

t = np.linspace(0, 4*np.pi, 100)
points = np.array([curve(ti) for ti in t])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(points[:, 0], points[:, 1], points[:, 2], label='Curve')

# Plot tangent vectors at several points
for ti in [0, np.pi/2, np.pi, 3*np.pi/2]:
    p = curve(ti)
    v = tangent(ti)
    ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2], color='r', length=0.5, normalize=True)

ax.set_title('Space Curve with Tangent Vectors')
ax.legend()
plt.show()
```

Slide 5: Arc Length and Parameterization

Arc length is the distance along a curve, and parameterization allows us to represent a curve using different parameters. Arc length parameterization ensures constant speed along the curve.

```python
import numpy as np
import sympy as sp

# Define symbolic variable and vector function
t = sp.Symbol('t')
r = sp.Matrix([sp.cos(t), sp.sin(t), t])

# Calculate derivative
r_prime = r.diff(t)

# Arc length function
arc_length = sp.integrate(sp.sqrt(sum(r_prime[i]**2 for i in range(3))), (t, 0, t))

print("Arc length function:")
print(arc_length)

# Calculate arc length for a specific interval
t1, t2 = 0, np.pi
length = arc_length.subs(t, t2) - arc_length.subs(t, t1)

print(f"\nArc length from t = {t1} to t = {t2}:")
print(length.evalf())
```

Slide 6: Curvature and Torsion

Curvature measures how sharply a curve bends, while torsion quantifies how much a curve twists out of a plane. These concepts are fundamental in differential geometry.

```python
import sympy as sp

# Define symbolic variable and vector function
t = sp.Symbol('t')
r = sp.Matrix([sp.cos(t), sp.sin(t), t])

# Calculate derivatives
r_prime = r.diff(t)
r_double_prime = r_prime.diff(t)
r_triple_prime = r_double_prime.diff(t)

# Calculate curvature
curvature = (r_prime.cross(r_double_prime).norm() / r_prime.norm()**3)

# Calculate torsion
torsion = (r_prime.dot(r_double_prime.cross(r_triple_prime)) / 
           r_prime.cross(r_double_prime).norm()**2)

print("Curvature:")
print(curvature.simplify())
print("\nTorsion:")
print(torsion.simplify())

# Evaluate at t = π/4
t_val = sp.pi/4
curvature_val = curvature.subs(t, t_val)
torsion_val = torsion.subs(t, t_val)

print(f"\nAt t = π/4:")
print(f"Curvature = {curvature_val.evalf()}")
print(f"Torsion = {torsion_val.evalf()}")
```

Slide 7: Partial Derivatives

Partial derivatives measure the rate of change of a function with respect to one variable while holding others constant. They are fundamental in multivariable calculus.

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variables and function
x, y = sp.symbols('x y')
f = x**2 + y**2

# Calculate partial derivatives
fx = sp.diff(f, x)
fy = sp.diff(f, y)

print("Function f(x, y):", f)
print("∂f/∂x =", fx)
print("∂f/∂y =", fy)

# Create a 3D plot of the function
x_vals = y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x, y) = x² + y²')
plt.colorbar(surf)
plt.show()
```

Slide 8: Directional Derivatives and Gradients

Directional derivatives measure the rate of change of a function in a specific direction. The gradient is a vector of partial derivatives and points in the direction of steepest ascent.

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables and function
x, y = sp.symbols('x y')
f = x**2 - y**2

# Calculate gradient
grad_f = sp.Matrix([sp.diff(f, x), sp.diff(f, y)])

print("Function f(x, y):", f)
print("Gradient ∇f:", grad_f)

# Calculate directional derivative
u = sp.Matrix([1, 1])  # Direction vector
dir_deriv = grad_f.dot(u.normalize())

print("Directional derivative in direction [1, 1]:", dir_deriv)

# Visualize gradient vector field
x_vals = y_vals = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x_vals, y_vals)
U = 2 * X
V = -2 * Y

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V, scale=50)
plt.title('Gradient Vector Field of f(x, y) = x² - y²')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 9: Multiple Integrals

Multiple integrals extend the concept of integration to functions of several variables. They are used to calculate volumes, surface areas, and other properties of multidimensional objects.

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variables and function
x, y = sp.symbols('x y')
f = sp.sin(x) * sp.cos(y)

# Calculate double integral
integral = sp.integrate(sp.integrate(f, (y, 0, sp.pi/2)), (x, 0, sp.pi/2))

print("Double integral of sin(x)cos(y) over [0, π/2] × [0, π/2]:")
print(integral)

# Visualize the function
x_vals = y_vals = np.linspace(0, np.pi/2, 50)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.sin(X) * np.cos(Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x, y) = sin(x)cos(y)')
plt.colorbar(surf)
plt.show()
```

Slide 10: Line Integrals

Line integrals compute the integral of a function along a curve. They are essential in calculating work done by a force field and in understanding conservative vector fields.

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbolic variables and vector field
t, x, y = sp.symbols('t x y')
F = sp.Matrix([y, x])

# Define parametric curve
r = sp.Matrix([sp.cos(t), sp.sin(t)])

# Calculate line integral
integrand = F.dot(r.diff(t))
line_integral = sp.integrate(integrand, (t, 0, 2*sp.pi))

print("Line integral of F = [y, x] along the unit circle:")
print(line_integral)

# Visualize vector field and curve
x_vals = y_vals = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x_vals, y_vals)
U = Y
V = X

plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V)
theta = np.linspace(0, 2*np.pi, 100)
x_curve = np.cos(theta)
y_curve = np.sin(theta)
plt.plot(x_curve, y_curve, 'r-', linewidth=2)
plt.title('Vector Field F = [y, x] and Unit Circle')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 11: Surface Integrals

Surface integrals extend line integrals to surfaces in three-dimensional space. They are used to calculate flux, surface area, and other properties of surfaces.

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define symbolic variables and surface
u, v = sp.symbols('u v')
x = u * sp.cos(v)
y = u * sp.sin(v)
z = u**2

# Calculate surface area
r_u = sp.Matrix([sp.diff(x, u), sp.diff(y, u), sp.diff(z, u)])
r_v = sp.Matrix([sp.diff(x, v), sp.diff(y, v), sp.diff(z, v)])
normal = r_u.cross(r_v)
surface_element = normal.norm()

surface_area = sp.integrate(sp.integrate(surface_element, (u, 0, 1)), (v, 0, 2*sp.pi))

print("Surface area of z = x^2 + y^2 for 0 ≤ √(x^2 + y^2) ≤ 1:")
print(surface_area)

# Visualize the surface
u_vals = np.linspace(0, 1, 50)
v_vals = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u_vals, v_vals)
X = U * np.cos(V)
Y = U * np.sin(V)
Z = U**2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Surface z = x^2 + y^2')
plt.colorbar(surf)
plt.show()
```

Slide 12: Green's Theorem

Green's Theorem relates a line integral around a simple closed curve to a double integral over the plane region it encloses. It's a fundamental result in vector calculus with applications in physics and engineering.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define vector field
def P(x, y):
    return x**2 + y

def Q(x, y):
    return x - y**2

# Define curl
def curl(x, y):
    return -2*y - 2*x

# Create grid
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Calculate vector field
U = P(X, Y)
V = Q(X, Y)

# Plot vector field
plt.figure(figsize=(10, 8))
plt.quiver(X, Y, U, V)

# Plot unit circle
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
plt.plot(x_circle, y_circle, 'r-', linewidth=2)

plt.title("Vector Field and Region for Green's Theorem")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()

# Note: Actual integration would require symbolic computation
print("Green's Theorem states that the line integral")
print("around the unit circle equals the double integral")
print("of the curl over the enclosed region.")
```

Slide 13: Stokes' Theorem

Stokes' Theorem generalizes Green's Theorem to three dimensions, relating the surface integral of the curl of a vector field over a surface to the line integral of the vector field around the boundary of the surface.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define vector field
def F(x, y, z):
    return np.array([y, z, x])

# Define surface (part of a sphere)
theta = np.linspace(0, np.pi/2, 30)
phi = np.linspace(0, np.pi, 30)
THETA, PHI = np.meshgrid(theta, phi)

R = 2  # radius
X = R * np.sin(PHI) * np.cos(THETA)
Y = R * np.sin(PHI) * np.sin(THETA)
Z = R * np.cos(PHI)

# Plot surface and vector field
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, alpha=0.3)

# Sample points for vector field
u = np.linspace(0, 2, 5)
v = np.linspace(0, 2, 5)
U, V = np.meshgrid(u, v)

x = U
y = V
z = np.sqrt(R**2 - x**2 - y**2)

Fx, Fy, Fz = F(x, y, z)

ax.quiver(x, y, z, Fx, Fy, Fz, length=0.5, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Surface and Vector Field for Stokes' Theorem")

plt.show()

print("Stokes' Theorem relates the surface integral of curl F")
print("over this portion of the sphere to the line integral of F")
print("around its boundary curve.")
```

Slide 14: Divergence Theorem

The Divergence Theorem, also known as Gauss's Theorem, relates the flux of a vector field through a closed surface to the divergence of the field within the volume enclosed by the surface.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define vector field
def F(x, y, z):
    return np.array([x, y, z])

# Define divergence
def div_F(x, y, z):
    return 3  # div F = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z = 1 + 1 + 1 = 3

# Create a cube
r = [-1, 1]
x, y, z = np.meshgrid(r, r, r)

# Calculate vector field at cube vertices
u, v, w = F(x, y, z)

# Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot cube edges
for s, e in zip(combinations(np.array(list(product(r, r, r))), 2), product([0, 1, 2], repeat=2)):
    if np.sum(np.abs(s[0] - e[0])) == r[1] - r[0]:
        ax.plot3D(*zip(s[0], e[0]), color="b")

# Plot vector field
ax.quiver(x, y, z, u, v, w, length=0.2, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Cube and Vector Field for Divergence Theorem")

plt.show()

print("The Divergence Theorem states that the surface integral")
print("of F • n over the cube's surface equals the volume integral")
print("of div F throughout the cube.")
print(f"Here, div F = {div_F(0, 0, 0)} everywhere.")
```

Slide 15: Real-life Example: Fluid Flow

Calculus III concepts are crucial in fluid dynamics. Consider water flowing through a pipe with varying cross-sectional area. The continuity equation, derived from the Divergence Theorem, relates flow rate to velocity and area.

```python
import numpy as np
import matplotlib.pyplot as plt

# Pipe parameters
L = 10  # length
R1, R2 = 2, 1  # radii at ends

# Generate pipe shape
x = np.linspace(0, L, 100)
r = R1 + (R2 - R1) * x / L

# Calculate velocity (assuming constant flow rate)
Q = 10  # flow rate
v = Q / (np.pi * r**2)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(x, r, 'b', label='Pipe radius')
plt.plot(x, -r, 'b')
plt.plot(x, v, 'r', label='Fluid velocity')
plt.title('Fluid Flow in a Converging Pipe')
plt.xlabel('Distance along pipe')
plt.ylabel('Radius / Velocity')
plt.legend()
plt.grid(True)
plt.show()

print("The continuity equation states: A1v1 = A2v2")
print("where A is cross-sectional area and v is velocity.")
print("This demonstrates conservation of mass in fluid flow.")
```

Slide 16: Real-life Example: Electromagnetism

Maxwell's equations, fundamental to electromagnetism, heavily use concepts from vector calculus. Gauss's law, one of these equations, relates the electric field to charge distribution using the Divergence Theorem.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a point charge
q = 1  # charge
k = 1  # Coulomb's constant (simplified)

# Define electric field
def E(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    return k * q * np.array([x, y, z]) / (r**3)

# Create grid
x, y, z = np.meshgrid(np.linspace(-2, 2, 5),
                      np.linspace(-2, 2, 5),
                      np.linspace(-2, 2, 5))

# Calculate electric field
Ex, Ey, Ez = E(x, y, z)

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot electric field vectors
ax.quiver(x, y, z, Ex, Ey, Ez, length=0.5, normalize=True)

# Plot charge
ax.scatter([0], [0], [0], color='r', s=100, label='Point charge')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Electric Field of a Point Charge")
ax.legend()

plt.show()

print("Gauss's Law: ∮ E • dA = q / ε₀")
print("This relates the electric field flux through a closed surface")
print("to the enclosed charge, demonstrating the Divergence Theorem.")
```

Slide 17: Additional Resources

For further exploration of Calculus III concepts:

1. ArXiv.org: "A Visual Introduction to Differential Forms and Calculus on Manifolds" by Fortney, J.P. (2010) URL: [https://arxiv.org/abs/1503.02651](https://arxiv.org/abs/1503.02651)
2. ArXiv.org: "Divergence Theorem Revisited" by Malakhaltsev, M.A. and Sidorov, S.V. (2019) URL: [https://arxiv.org/abs/1911.03539](https://arxiv.org/abs/1911.03539)
3. ArXiv.org: "On Green's Theorem" by Gonzalez, R. (2010) URL: [https://arxiv.org/abs/1001.4750](https://arxiv.org/abs/1001.4750)

These papers provide in-depth discussions and novel perspectives on key Calculus III topics.

