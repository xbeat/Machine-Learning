## Jacobian Matrix Slideshow with Python Examples
Slide 1:

Introduction to Jacobian Matrices

A Jacobian matrix is a fundamental concept in multivariable calculus and linear algebra. It represents the best linear approximation of a differentiable function near a given point. The Jacobian matrix contains all first-order partial derivatives of a vector-valued function.

```python
import numpy as np

def f(x, y):
    return np.array([x**2 + y, x*y + y**2])

def jacobian(x, y):
    return np.array([
        [2*x, 1],
        [y, x + 2*y]
    ])

# Example point
x, y = 1, 2
J = jacobian(x, y)
print(f"Jacobian at (1, 2):\n{J}")
```

Slide 2:

Defining the Jacobian Matrix

The Jacobian matrix J of a function f: ℝⁿ → ℝᵐ is an m×n matrix of all first-order partial derivatives. For a function f(x₁, ..., xₙ) = (f₁, ..., fₘ), the Jacobian is:

J = \[∂fᵢ/∂xⱼ\]

```python
import sympy as sp

# Define symbolic variables
x, y = sp.symbols('x y')

# Define a vector-valued function
f = sp.Matrix([x**2 + y, sp.sin(x) + y**2])

# Calculate the Jacobian matrix
J = f.jacobian([x, y])

print("Symbolic Jacobian:")
sp.pprint(J)
```

Slide 3:

Computing the Jacobian Matrix

To compute the Jacobian matrix, we calculate partial derivatives of each component function with respect to each variable. This process can be done symbolically or numerically.

```python
import numpy as np
from scipy.misc import derivative

def f(x):
    return np.array([x[0]**2 + x[1], np.sin(x[0]) + x[1]**2])

def numerical_jacobian(f, x, h=1e-5):
    n = len(x)
    jac = np.zeros((n, n))
    for i in range(n):
        def fi(t):
            xt = x.()
            xt[i] = t
            return f(xt)[i]
        for j in range(n):
            jac[i, j] = derivative(fi, x[j], dx=h)
    return jac

x = np.array([1.0, 2.0])
J = numerical_jacobian(f, x)
print(f"Numerical Jacobian at {x}:\n{J}")
```

Slide 4:

Jacobian Matrix Properties

The Jacobian matrix has several important properties:

1. Dimension: For a function f: ℝⁿ → ℝᵐ, the Jacobian is an m×n matrix.
2. Invertibility: If n = m and the Jacobian is invertible at a point, the function is locally invertible near that point.
3. Determinant: The determinant of the Jacobian represents the factor by which the function scales volumes.

```python
import numpy as np

def f(x, y, z):
    return np.array([x**2 + y + z, x*y + y**2 + z, x + y + z**2])

def jacobian(x, y, z):
    return np.array([
        [2*x, 1, 1],
        [y, 2*y + x, 1],
        [1, 1, 2*z]
    ])

# Example point
x, y, z = 1, 2, 3
J = jacobian(x, y, z)
print(f"Jacobian at (1, 2, 3):\n{J}")
print(f"Determinant: {np.linalg.det(J)}")
print(f"Invertible: {np.linalg.det(J) != 0}")
```

Slide 5:

Jacobian in Coordinate Transformations

The Jacobian matrix plays a crucial role in coordinate transformations. It helps us understand how areas or volumes change when we switch between different coordinate systems.

```python
import numpy as np

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

def jacobian_polar_to_cartesian(r, theta):
    return np.array([
        [np.cos(theta), -r * np.sin(theta)],
        [np.sin(theta), r * np.cos(theta)]
    ])

# Example transformation
r, theta = 2, np.pi/4
J = jacobian_polar_to_cartesian(r, theta)
print(f"Jacobian of polar to Cartesian at (r={r}, θ={theta}):\n{J}")
print(f"Determinant (represents area scaling): {np.linalg.det(J)}")
```

Slide 6:

Jacobian in Optimization

The Jacobian matrix is essential in optimization algorithms, particularly in gradient-based methods for multivariate functions. It's used to compute the direction of steepest ascent or descent.

```python
import numpy as np

def f(x):
    return x[0]**2 + 2*x[1]**2

def gradient(x):
    return np.array([2*x[0], 4*x[1]])

def gradient_descent(f, gradient, x0, learning_rate=0.1, num_iterations=100):
    x = x0
    for _ in range(num_iterations):
        x = x - learning_rate * gradient(x)
    return x

x0 = np.array([1.0, 1.0])
result = gradient_descent(f, gradient, x0)
print(f"Optimized point: {result}")
print(f"Optimized value: {f(result)}")
```

Slide 7:

Jacobian in Newton's Method

Newton's method for finding roots of multivariate functions uses the Jacobian matrix. It's an iterative method that approximates the function with its linear approximation at each step.

```python
import numpy as np

def f(x):
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        x[0] * x[1] - 1
    ])

def jacobian(x):
    return np.array([
        [2*x[0], 2*x[1]],
        [x[1], x[0]]
    ])

def newton_method(f, jacobian, x0, num_iterations=10):
    x = x0
    for _ in range(num_iterations):
        J_inv = np.linalg.inv(jacobian(x))
        x = x - np.dot(J_inv, f(x))
    return x

x0 = np.array([1.0, 1.0])
result = newton_method(f, jacobian, x0)
print(f"Root found: {result}")
print(f"Function value at root: {f(result)}")
```

Slide 8:

Jacobian in Sensitivity Analysis

The Jacobian matrix is used in sensitivity analysis to understand how small changes in input variables affect the output of a system. This is crucial in many engineering and scientific applications.

```python
import numpy as np

def system(x, y):
    return np.array([
        x**2 + y**2,
        x * y
    ])

def jacobian(x, y):
    return np.array([
        [2*x, 2*y],
        [y, x]
    ])

# Compute sensitivity at a point
x, y = 1, 2
J = jacobian(x, y)
print(f"Jacobian (sensitivity matrix) at (1, 2):\n{J}")

# Interpret sensitivity
dx, dy = 0.1, 0.1
dF = np.dot(J, np.array([dx, dy]))
print(f"Estimated change in output for dx={dx}, dy={dy}: {dF}")
```

Slide 9:

Jacobian in Robotics

In robotics, the Jacobian matrix relates joint velocities to end-effector velocities. It's crucial for motion planning and control of robotic arms.

```python
import numpy as np

def forward_kinematics(theta1, theta2):
    # Simple 2-link planar robot
    l1, l2 = 1, 1  # link lengths
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])

def jacobian(theta1, theta2):
    l1, l2 = 1, 1
    return np.array([
        [-l1*np.sin(theta1) - l2*np.sin(theta1+theta2), -l2*np.sin(theta1+theta2)],
        [l1*np.cos(theta1) + l2*np.cos(theta1+theta2), l2*np.cos(theta1+theta2)]
    ])

# Example configuration
theta1, theta2 = np.pi/4, np.pi/3
J = jacobian(theta1, theta2)
print(f"Jacobian for robot arm at θ1={theta1:.2f}, θ2={theta2:.2f}:\n{J}")

# Compute end-effector velocity for given joint velocities
dtheta1, dtheta2 = 0.1, 0.2
dX = np.dot(J, np.array([dtheta1, dtheta2]))
print(f"End-effector velocity: {dX}")
```

Slide 10:

Jacobian in Fluid Dynamics

In fluid dynamics, the Jacobian matrix appears in the study of flow fields and transformations between different coordinate systems. It's particularly useful in analyzing complex fluid flows.

```python
import numpy as np

def velocity_field(x, y, z):
    # Example velocity field (e.g., for a vortex)
    u = -y
    v = x
    w = np.sin(z)
    return np.array([u, v, w])

def jacobian(x, y, z):
    return np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, np.cos(z)]
    ])

# Analyze flow at a point
x, y, z = 1, 2, np.pi/4
J = jacobian(x, y, z)
print(f"Jacobian of velocity field at (1, 2, π/4):\n{J}")

# Compute vorticity (curl of velocity field)
vorticity = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
print(f"Vorticity: {vorticity}")
```

Slide 11:

Jacobian in Machine Learning

In machine learning, particularly in neural networks, the Jacobian matrix is used in backpropagation to compute gradients. It's essential for training models using gradient-based optimization methods.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neural_network(x, W1, W2):
    # Simple 2-layer neural network
    h = sigmoid(np.dot(W1, x))
    y = sigmoid(np.dot(W2, h))
    return y

def jacobian_nn(x, W1, W2):
    h = sigmoid(np.dot(W1, x))
    y = sigmoid(np.dot(W2, h))
    
    # Compute Jacobian with respect to input x
    dh_dx = h * (1 - h) * W1
    dy_dh = y * (1 - y) * W2
    J = np.dot(dy_dh, dh_dx)
    
    return J

# Example network and input
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
W2 = np.array([[0.5, 0.6]])
x = np.array([1, 2])

J = jacobian_nn(x, W1, W2)
print(f"Jacobian of neural network output with respect to input:\n{J}")
```

Slide 12:

Jacobian in Image Processing

In image processing, the Jacobian matrix is used in various transformations and analysis techniques. It's particularly useful in image registration and warping.

```python
import numpy as np
import matplotlib.pyplot as plt

def image_warp(x, y):
    # Example warping function
    u = x + 0.1 * np.sin(2 * np.pi * y)
    v = y + 0.1 * np.sin(2 * np.pi * x)
    return u, v

def jacobian_warp(x, y):
    return np.array([
        [1 + 0.2 * np.pi * np.cos(2 * np.pi * y), 0.2 * np.pi * np.cos(2 * np.pi * x)],
        [0.2 * np.pi * np.cos(2 * np.pi * y), 1 + 0.2 * np.pi * np.cos(2 * np.pi * x)]
    ])

# Create a sample image
x, y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
image = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Compute warped coordinates
u, v = image_warp(x, y)

# Compute Jacobian determinant
J_det = np.abs(np.linalg.det(jacobian_warp(x, y)))

# Plot original and warped images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(image, cmap='gray')
ax2.contour(u, v, colors='r', linewidths=0.5)
ax2.set_title('Warped Grid')
ax3.imshow(J_det, cmap='viridis')
ax3.set_title('Jacobian Determinant')
plt.tight_layout()
plt.show()
```

Slide 13:

Real-Life Example: Stress Analysis in Materials Science

In materials science, the Jacobian matrix is used to analyze stress and strain relationships in materials. It helps engineers understand how materials deform under various loads.

```python
import numpy as np

def stress_strain_relation(strain):
    E = 200e9  # Young's modulus for steel (Pa)
    nu = 0.3   # Poisson's ratio for steel
    
    D = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])
    
    return np.dot(D, strain)

def jacobian_stress_strain(strain):
    E = 200e9
    nu = 0.3
    return E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ])

strain = np.array([0.001, 0.0005, 0.0002])
stress = stress_strain_relation(strain)
J = jacobian_stress_strain(strain)

print(f"Strain: {strain}")
print(f"Stress: {stress}")
print(f"Jacobian (stiffness matrix):\n{J}")
```

Slide 14:

Real-Life Example: Chemical Reaction Kinetics

The Jacobian matrix is crucial in analyzing the dynamics of chemical reaction systems. It helps in understanding the stability of reaction networks and predicting system behavior.

```python
import numpy as np

def reaction_rates(concentrations, k):
    A, B, C = concentrations
    return np.array([
        -k[0]*A*B + k[1]*C,    # dA/dt
        -k[0]*A*B + k[1]*C,    # dB/dt
        k[0]*A*B - k[1]*C      # dC/dt
    ])

def jacobian_reaction(concentrations, k):
    A, B, C = concentrations
    return np.array([
        [-k[0]*B, -k[0]*A, k[1]],
        [-k[0]*B, -k[0]*A, k[1]],
        [k[0]*B,  k[0]*A, -k[1]]
    ])

k = [0.1, 0.05]  # Rate constants
concentrations = np.array([1.0, 1.0, 0.0])  # Initial [A], [B], [C]

rates = reaction_rates(concentrations, k)
J = jacobian_reaction(concentrations, k)

print("Reaction rates:")
print(rates)
print("\nJacobian matrix:")
print(J)
```

Slide 15:

Additional Resources

For further exploration of Jacobian matrices and their applications, consider these resources:

1. "Multivariable Calculus and Differential Geometry" by Hubbard and Hubbard (ArXiv:1609.07077)
2. "Numerical Methods for Unconstrained Optimization and Nonlinear Equations" by Dennis and Schnabel (ArXiv:1803.06673)
3. "An Introduction to Sensitivity Analysis" by Saltelli et al. (ArXiv:1101.5242)

These papers provide in-depth discussions on the theory and applications of Jacobian matrices in various fields of mathematics and science.

