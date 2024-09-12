## Exploring Lie Groups and Lie Algebras with Python

Slide 1: Introduction to Lie Groups

Lie groups are continuous symmetry groups that play a crucial role in physics and mathematics. They are named after Sophus Lie, who studied their properties in the late 19th century.

```python
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

# Example: Rotation group SO(2)
thetas = np.linspace(0, 2*np.pi, 100)
x, y = np.cos(thetas), np.sin(thetas)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.title("SO(2) - Special Orthogonal Group in 2D")
plt.axis('equal')
plt.show()
```

Slide 2: Lie Group Example - Special Unitary Group SU(2)

SU(2) is a Lie group of 2×2 unitary matrices with determinant 1. It is closely related to rotations in 3D space and plays a crucial role in quantum mechanics.

```python
import numpy as np

def su2_matrix(a, b):
    return np.array([[a, -np.conj(b)],
                     [b, np.conj(a)]])

# Example: Generate a random SU(2) matrix
a = np.random.normal() + 1j * np.random.normal()
b = np.random.normal() + 1j * np.random.normal()
norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
a, b = a/norm, b/norm

matrix = su2_matrix(a, b)
print("Random SU(2) matrix:")
print(matrix)
print("Determinant:", np.linalg.det(matrix))
```

Slide 3: Lie Algebras - The Tangent Space of Lie Groups

Lie algebras are vector spaces associated with Lie groups, representing their infinitesimal generators. They capture the local structure of the group near the identity element.

```python
import numpy as np

# Pauli matrices - generators of SU(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

def commutator(A, B):
    return A @ B - B @ A

print("Commutator [σx, σy]:")
print(commutator(sigma_x, sigma_y))
print("2i * σz:")
print(2j * sigma_z)
```

Slide 4: Exponential Map - Connecting Lie Algebras to Lie Groups

The exponential map connects elements of the Lie algebra to elements of the Lie group. It's a crucial tool for understanding the relationship between these structures.

```python
import numpy as np
from scipy.linalg import expm

def rotation_generator(axis):
    if axis == 'x':
        return np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    elif axis == 'y':
        return np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    elif axis == 'z':
        return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

# Example: Generate a rotation around the z-axis
theta = np.pi/4  # 45-degree rotation
R_z = expm(theta * rotation_generator('z'))

print("Rotation matrix around z-axis:")
print(R_z)
```

Slide 5: Representations - Making Lie Groups Act on Vector Spaces

Representations allow us to study how Lie groups act on vector spaces. They are essential for applications in physics and other fields.

```python
import numpy as np

def represent_su2(a, b):
    return np.array([[a, -np.conj(b)],
                     [b, np.conj(a)]])

# Example: Action of SU(2) on a 2D vector
vector = np.array([1, 0])
a, b = 1/np.sqrt(2), 1j/np.sqrt(2)
su2_element = represent_su2(a, b)

transformed_vector = su2_element @ vector

print("Original vector:", vector)
print("Transformed vector:", transformed_vector)
```

Slide 6: Adjoint Representation - Groups Acting on Their Lie Algebras

The adjoint representation describes how a Lie group acts on its own Lie algebra through conjugation. It's a fundamental tool for understanding the structure of Lie groups.

```python
import numpy as np

def adjoint_action(g, X):
    return g @ X @ np.linalg.inv(g)

# Example: Adjoint action of SO(3) on its Lie algebra
def so3_element(theta, axis):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

X = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # Element of so(3)
g = so3_element(np.pi/4, 'z')  # Rotation around z-axis

print("Adjoint action result:")
print(adjoint_action(g, X))
```

Slide 7: Lie Group Homomorphisms and Isomorphisms

Homomorphisms and isomorphisms are structure-preserving maps between Lie groups. They help us understand relationships between different groups.

```python
import numpy as np

def su2_to_so3(a, b):
    """Map from SU(2) to SO(3)"""
    x, y, z = np.real(a), np.imag(a), np.real(b)
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z, 2*x*z + 2*y],
        [2*x*y + 2*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x],
        [2*x*z - 2*y, 2*y*z + 2*x, 1 - 2*x**2 - 2*y**2]
    ])

# Example: Map an SU(2) element to SO(3)
a, b = 1/np.sqrt(2), 1j/np.sqrt(2)
so3_matrix = su2_to_so3(a, b)

print("Mapped SO(3) matrix:")
print(so3_matrix)
print("Determinant:", np.linalg.det(so3_matrix))
```

Slide 8: Lie Group Actions on Manifolds

Lie groups often act on manifolds, leading to concepts like orbits and stabilizers. This is crucial in areas like differential geometry and theoretical physics.

```python
import numpy as np
import matplotlib.pyplot as plt

def rotate_point(theta, point):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R @ point

# Example: SO(2) action on a point in R^2
point = np.array([1, 0])
thetas = np.linspace(0, 2*np.pi, 100)
orbit = np.array([rotate_point(theta, point) for theta in thetas])

plt.figure(figsize=(8, 8))
plt.plot(orbit[:, 0], orbit[:, 1])
plt.scatter(*point, color='red', s=100, label='Original point')
plt.title("SO(2) Action on a Point in R^2")
plt.axis('equal')
plt.legend()
plt.show()
```

Slide 9: Lie Algebra Brackets and the Jacobi Identity

The Lie bracket is a crucial operation in Lie algebras, satisfying properties like bilinearity, antisymmetry, and the Jacobi identity.

```python
import numpy as np

def lie_bracket(X, Y):
    return X @ Y - Y @ X

def check_jacobi(X, Y, Z):
    return (lie_bracket(X, lie_bracket(Y, Z)) +
            lie_bracket(Y, lie_bracket(Z, X)) +
            lie_bracket(Z, lie_bracket(X, Y)))

# Example: Check Jacobi identity for so(3)
X = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
Y = np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
Z = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

jacobi_result = check_jacobi(X, Y, Z)
print("Jacobi identity result:")
print(jacobi_result)
print("Jacobi identity satisfied:", np.allclose(jacobi_result, 0))
```

Slide 10: Character Theory and Representation Decomposition

Characters are powerful tools for studying representations, allowing us to decompose them into irreducible components.

```python
import numpy as np

def character(representation):
    return np.trace(representation)

# Example: Characters of SU(2) representations
def su2_rep_1(theta):
    return np.array([[np.exp(1j*theta/2), 0],
                     [0, np.exp(-1j*theta/2)]])

def su2_rep_2(theta):
    return np.array([[np.exp(1j*theta), 0, 0],
                     [0, 1, 0],
                     [0, 0, np.exp(-1j*theta)]])

theta = np.pi/4
chi_1 = character(su2_rep_1(theta))
chi_2 = character(su2_rep_2(theta))

print("Character of 2D representation:", chi_1)
print("Character of 3D representation:", chi_2)
```

Slide 11: Lie Groups in Physics - Symmetries and Conservation Laws

Lie groups are fundamental in physics, connecting symmetries to conservation laws through Noether's theorem.

```python
import sympy as sp

# Example: Conservation of angular momentum from rotational symmetry
t, m, r, theta = sp.symbols('t m r theta')
L = sp.Function('L')(t, r, theta)

# Lagrangian for a particle in polar coordinates
L = 0.5 * m * (r**2 + (r * sp.diff(theta, t))**2)

# Euler-Lagrange equation for theta
EL_theta = sp.diff(L, theta) - sp.diff(sp.diff(L, sp.diff(theta, t)), t)

# Angular momentum
p_theta = sp.diff(L, sp.diff(theta, t))

print("Euler-Lagrange equation for theta:")
print(EL_theta)
print("\nAngular momentum:")
print(p_theta)
print("\nTime derivative of angular momentum:")
print(sp.diff(p_theta, t).simplify())
```

Slide 12: Representation Theory in Quantum Mechanics

Representation theory is crucial in quantum mechanics, particularly in understanding angular momentum and spin.

```python
import numpy as np

# Pauli matrices - generators of SU(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Example: Spin-1/2 particle in a magnetic field
def hamiltonian(B_x, B_y, B_z):
    return -0.5 * (B_x * sigma_x + B_y * sigma_y + B_z * sigma_z)

B = np.array([1, 0, 1])  # Magnetic field
H = hamiltonian(*B)

eigenvalues, eigenvectors = np.linalg.eigh(H)

print("Hamiltonian:")
print(H)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
```

Slide 13: Lie Groups in Machine Learning - Invariant Neural Networks

Lie group theory can be applied to create neural networks with built-in symmetries, improving generalization and data efficiency.

```python
import torch
import torch.nn as nn

class SO2EquivariantConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        # Regular convolution
        y = self.conv(x)
        
        # Rotate input by 90 degrees
        x_rot = torch.rot90(x, k=1, dims=[2, 3])
        y_rot = self.conv(x_rot)
        y_rot = torch.rot90(y_rot, k=-1, dims=[2, 3])
        
        # Average the results
        return 0.5 * (y + y_rot)

# Example usage
conv = SO2EquivariantConv(3, 16)
input_tensor = torch.randn(1, 3, 32, 32)
output = conv(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
```

Slide 14: Additional Resources

For further exploration of Lie Groups, Lie Algebras, and Representations, consider the following ArXiv papers:

1. "An Introduction to Lie Groups for Physicists" by John Baez ArXiv: [https://arxiv.org/abs/math-ph/0202046](https://arxiv.org/abs/math-ph/0202046)
2. "Representation Theory of Lie Groups and Lie Algebras" by Anthony W. Knapp ArXiv: [https://arxiv.org/abs/math/0402395](https://arxiv.org/abs/math/0402395)
3. "Lie Groups for 2D and 3D Transformations" by Ethan Eade ArXiv: [https://arxiv.org/abs/1812.01572](https://arxiv.org/abs/1812.01572)

These resources provide in-depth discussions and advanced topics in the field.

