## Finite Element Analysis with Python:
Slide 1: Introduction to Finite Element Analysis (FEA)

Finite Element Analysis is a numerical method for solving complex engineering problems. It involves dividing a large problem into smaller, simpler parts called finite elements. Let's visualize a simple mesh generation:

```python
import numpy as np
import matplotlib.pyplot as plt

def create_mesh(x_range, y_range, num_elements):
    x = np.linspace(x_range[0], x_range[1], num_elements + 1)
    y = np.linspace(y_range[0], y_range[1], num_elements + 1)
    return np.meshgrid(x, y)

x, y = create_mesh((0, 1), (0, 1), 10)
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b')
plt.plot(x.T, y.T, 'b')
plt.title('Simple 2D Mesh for FEA')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

Slide 2: Setting Up the FEA Environment

Before diving into FEA, we need to set up our Python environment with necessary libraries:

```python
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Example: Check versions
print(f"NumPy version: {np.__version__}")
print(f"SciPy version: {linalg.__version__}")
print(f"Matplotlib version: {plt.__version__}")
```

Slide 3: Defining the Problem Domain

Let's consider a simple 1D heat conduction problem. We'll define the domain and boundary conditions:

```python
# Problem parameters
L = 1.0  # Length of the domain
k = 1.0  # Thermal conductivity
T_left = 100.0  # Temperature at left boundary
T_right = 0.0  # Temperature at right boundary

# Discretization
n_elements = 10
element_length = L / n_elements

# Node coordinates
x = np.linspace(0, L, n_elements + 1)
```

Slide 4: Element Stiffness Matrix

For our 1D heat conduction problem, let's create the element stiffness matrix:

```python
def element_stiffness_matrix(k, l):
    return k / l * np.array([[1, -1], [-1, 1]])

K_e = element_stiffness_matrix(k, element_length)
print("Element Stiffness Matrix:")
print(K_e)
```

Slide 5: Global Stiffness Matrix Assembly

Now, we'll assemble the global stiffness matrix:

```python
def assemble_global_stiffness(n_elements, K_e):
    K = np.zeros((n_elements + 1, n_elements + 1))
    for i in range(n_elements):
        K[i:i+2, i:i+2] += K_e
    return K

K = assemble_global_stiffness(n_elements, K_e)
print("Global Stiffness Matrix:")
print(K)
```

Slide 6: Applying Boundary Conditions

Let's apply the Dirichlet boundary conditions to our problem:

```python
def apply_boundary_conditions(K, T_left, T_right):
    K_bc = K.()
    K_bc[0, :] = 0
    K_bc[0, 0] = 1
    K_bc[-1, :] = 0
    K_bc[-1, -1] = 1
    
    F = np.zeros(K.shape[0])
    F[0] = T_left
    F[-1] = T_right
    
    return K_bc, F

K_bc, F = apply_boundary_conditions(K, T_left, T_right)
```

Slide 7: Solving the System

Now we can solve the system of equations:

```python
def solve_system(K, F):
    return linalg.solve(K, F)

T = solve_system(K_bc, F)
print("Temperature distribution:")
print(T)
```

Slide 8: Visualizing the Results

Let's plot the temperature distribution:

```python
plt.figure(figsize=(10, 6))
plt.plot(x, T, 'ro-')
plt.title('Temperature Distribution')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
```

Slide 9: Real-life Example 1 - Beam Deflection

Let's consider a simple beam deflection problem:

```python
def beam_deflection(L, E, I, w):
    x = np.linspace(0, L, 100)
    y = w * x**2 * (L - x)**2 / (24 * E * I)
    return x, y

L = 5  # Length of beam (m)
E = 200e9  # Young's modulus (Pa)
I = 3e-5  # Moment of inertia (m^4)
w = 10000  # Distributed load (N/m)

x, y = beam_deflection(L, E, I, w)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Beam Deflection')
plt.xlabel('Position along beam (m)')
plt.ylabel('Deflection (m)')
plt.grid(True)
plt.show()
```

Slide 10: Element Types in FEA

Different element types are used for various problems. Let's visualize some common 2D elements:

```python
def plot_element(vertices, element_type):
    plt.figure(figsize=(4, 4))
    plt.plot(vertices[:, 0], vertices[:, 1], 'ro-')
    plt.title(f'{element_type} Element')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Triangle element
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
plot_element(triangle, 'Triangle')

# Quadrilateral element
quad = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
plot_element(quad, 'Quadrilateral')
```

Slide 11: Nonlinear FEA

Nonlinear FEA is used for problems with material nonlinearity, geometric nonlinearity, or both. Here's a simple example of a nonlinear spring:

```python
def nonlinear_spring(k, x):
    return k * x + 0.1 * k * x**3

x = np.linspace(-2, 2, 100)
F = nonlinear_spring(1000, x)

plt.figure(figsize=(8, 6))
plt.plot(x, F)
plt.title('Nonlinear Spring Force-Displacement Curve')
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.grid(True)
plt.show()
```

Slide 12: Real-life Example 2 - Stress Analysis

Let's simulate a simple stress analysis for a plate with a hole:

```python
import numpy as np
import matplotlib.pyplot as plt

def stress_concentration(r, a, sigma):
    theta = np.linspace(0, 2*np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    sigma_xx = sigma * (1 - (a**2 / r**2) * (3/2 * np.cos(2*theta) + np.cos(4*theta)) + 
                        (3*a**4 / (2*r**4)) * np.cos(4*theta))
    
    return x, y, sigma_xx

r = 1.0  # Radius of the plate
a = 0.2  # Radius of the hole
sigma = 100  # Applied stress

x, y, sigma_xx = stress_concentration(r, a, sigma)

plt.figure(figsize=(10, 8))
sc = plt.scatter(x, y, c=sigma_xx, cmap='jet')
plt.colorbar(sc, label='Stress')
plt.title('Stress Distribution Around a Circular Hole')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()
```

Slide 13: Mesh Refinement

Mesh refinement is crucial for accurate FEA results. Let's visualize different mesh densities:

```python
def create_refined_mesh(x_range, y_range, num_elements):
    x = np.linspace(x_range[0], x_range[1], num_elements + 1)
    y = np.linspace(y_range[0], y_range[1], num_elements + 1)
    return np.meshgrid(x, y)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
mesh_sizes = [5, 10, 20]

for i, size in enumerate(mesh_sizes):
    x, y = create_refined_mesh((0, 1), (0, 1), size)
    axs[i].plot(x, y, 'b')
    axs[i].plot(x.T, y.T, 'b')
    axs[i].set_title(f'Mesh with {size}x{size} elements')
    axs[i].set_aspect('equal')

plt.tight_layout()
plt.show()
```

Slide 14: Error Estimation and Convergence

Error estimation is important in FEA. Let's simulate convergence for our 1D heat conduction problem:

```python
def analytical_solution(x, L, T_left, T_right):
    return T_left + (T_right - T_left) * x / L

def compute_error(numerical, analytical):
    return np.max(np.abs(numerical - analytical))

errors = []
element_counts = [5, 10, 20, 40, 80]

for n in element_counts:
    x = np.linspace(0, L, n + 1)
    K = assemble_global_stiffness(n, element_stiffness_matrix(k, L/n))
    K_bc, F = apply_boundary_conditions(K, T_left, T_right)
    T_num = solve_system(K_bc, F)
    T_ana = analytical_solution(x, L, T_left, T_right)
    errors.append(compute_error(T_num, T_ana))

plt.figure(figsize=(10, 6))
plt.loglog(element_counts, errors, 'bo-')
plt.title('Convergence of FEA Solution')
plt.xlabel('Number of Elements')
plt.ylabel('Maximum Error')
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For further reading on Finite Element Analysis using Python, consider the following resources:

1. "A Tutorial on the Implementation of Finite Element Methods using Python" by A. Sridhar et al. (arXiv:2005.07742)
2. "Finite Element Analysis with Python" by R. Jaiman (arXiv:1910.04772)
3. "Numerical Methods in Engineering with Python 3" by J. Kiusalaas (Cambridge University Press)

These resources provide in-depth explanations and more advanced examples of FEA implementation in Python.

