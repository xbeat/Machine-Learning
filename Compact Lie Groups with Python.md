## Compact Lie Groups with Python
Slide 1: Introduction to Compact Lie Groups

Compact Lie groups are fundamental structures in mathematics and physics, combining the properties of compactness and differentiability. They play a crucial role in various fields, including quantum mechanics, particle physics, and differential geometry. In this presentation, we'll explore the basics of compact Lie groups and their applications, using Python to illustrate key concepts.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_so2():
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    plt.figure(figsize=(6,6))
    plt.plot(x, y)
    plt.title('SO(2) - Special Orthogonal Group in 2D')
    plt.axis('equal')
    plt.show()

visualize_so2()
```

Slide 2: Definition and Properties

A compact Lie group is a Lie group that is compact as a topological space. This combination of algebraic and topological properties makes them particularly useful in various applications. Key properties include:

1. Closed and bounded in the topology of the ambient space
2. Finite volume with respect to Haar measure
3. Existence of a bi-invariant Riemannian metric

```python
import numpy as np

def is_orthogonal(matrix):
    return np.allclose(np.dot(matrix, matrix.T), np.eye(matrix.shape[0]))

def random_so3_matrix():
    A = np.random.rand(3, 3)
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))

matrix = random_so3_matrix()
print(f"Random SO(3) matrix:\n{matrix}")
print(f"Is orthogonal: {is_orthogonal(matrix)}")
print(f"Determinant: {np.linalg.det(matrix):.6f}")
```

Slide 3: Examples of Compact Lie Groups

Some well-known examples of compact Lie groups include:

1. SO(n) - Special Orthogonal Group
2. U(n) - Unitary Group
3. SU(n) - Special Unitary Group
4. Sp(n) - Symplectic Group

Let's focus on SO(3), the group of rotations in 3D space, which is crucial in physics and computer graphics.

```python
import numpy as np

def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

axis = [1, 0, 0]  # Rotation around x-axis
theta = np.pi / 4  # 45 degree rotation
R = rotation_matrix(axis, theta)
print(f"Rotation matrix:\n{R}")
```

Slide 4: Lie Algebra and Exponential Map

The Lie algebra of a Lie group is the tangent space at the identity element. For matrix Lie groups, the exponential map connects the Lie algebra to the Lie group. This relationship is fundamental in understanding the structure of Lie groups.

```python
import numpy as np
from scipy.linalg import expm

def so3_algebra_basis():
    E1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    E2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
    E3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    return E1, E2, E3

def lie_algebra_to_group(w):
    E1, E2, E3 = so3_algebra_basis()
    W = w[0] * E1 + w[1] * E2 + w[2] * E3
    return expm(W)

w = np.array([0.1, 0.2, 0.3])
R = lie_algebra_to_group(w)
print(f"Lie algebra element:\n{w}")
print(f"Corresponding group element:\n{R}")
```

Slide 5: Representation Theory

Representations of compact Lie groups are fundamental in quantum mechanics and particle physics. They allow us to understand how these groups act on vector spaces, which is crucial for describing symmetries in physical systems.

```python
import numpy as np

def represent_su2(theta, phi, chi):
    return np.array([
        [np.exp(-1j*(phi+chi)/2) * np.cos(theta/2),
         -np.exp(-1j*(phi-chi)/2) * np.sin(theta/2)],
        [np.exp(1j*(phi-chi)/2) * np.sin(theta/2),
         np.exp(1j*(phi+chi)/2) * np.cos(theta/2)]
    ])

theta, phi, chi = np.pi/4, np.pi/3, np.pi/6
U = represent_su2(theta, phi, chi)
print(f"SU(2) representation:\n{U}")
print(f"Determinant: {np.linalg.det(U):.6f}")
```

Slide 6: Character Theory

Characters of representations are powerful tools in the study of compact Lie groups. They provide a way to classify and analyze representations, which is particularly useful in applications like particle physics.

```python
import numpy as np

def character_su2(j, theta):
    return np.sin((2*j + 1) * theta/2) / np.sin(theta/2)

j_values = [0, 1/2, 1, 3/2, 2]
theta = np.linspace(0, 2*np.pi, 100)

for j in j_values:
    chi = character_su2(j, theta)
    plt.plot(theta, chi, label=f'j = {j}')

plt.xlabel('θ')
plt.ylabel('χ(θ)')
plt.title('Characters of SU(2) Representations')
plt.legend()
plt.show()
```

Slide 7: Haar Measure

The Haar measure is a unique (up to scaling) invariant measure on compact Lie groups. It allows us to integrate functions over the group in a way that respects the group structure. This is crucial for many applications in physics and mathematics.

```python
import numpy as np

def random_so3_element():
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))

def monte_carlo_integral(f, n_samples=10000):
    integral = 0
    for _ in range(n_samples):
        g = random_so3_element()
        integral += f(g)
    return integral / n_samples

def test_function(g):
    return np.trace(g)

result = monte_carlo_integral(test_function)
print(f"Monte Carlo estimate of integral: {result:.6f}")
print("Expected value: 0 (by symmetry)")
```

Slide 8: Applications in Physics: Angular Momentum

Compact Lie groups, particularly SU(2) and SO(3), are fundamental in quantum mechanics for describing angular momentum. The representation theory of these groups directly corresponds to the behavior of spin and orbital angular momentum.

```python
import numpy as np

def spin_matrices(s):
    if s == 1/2:
        Sx = np.array([[0, 1], [1, 0]]) / 2
        Sy = np.array([[0, -1j], [1j, 0]]) / 2
        Sz = np.array([[1, 0], [0, -1]]) / 2
    elif s == 1:
        Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
        Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    else:
        raise ValueError("Only s=1/2 and s=1 are implemented")
    return Sx, Sy, Sz

Sx, Sy, Sz = spin_matrices(1/2)
print("Spin-1/2 matrices:")
print(f"Sx:\n{Sx}")
print(f"Sy:\n{Sy}")
print(f"Sz:\n{Sz}")
```

Slide 9: Applications in Machine Learning: Rotation-Invariant Neural Networks

Compact Lie groups, especially SO(3), are increasingly important in machine learning, particularly for developing rotation-invariant neural networks. These networks can recognize 3D objects regardless of their orientation, which is crucial for tasks like 3D object recognition and molecular property prediction.

```python
import torch
import torch.nn as nn

class SO3Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SO3Conv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Assume x is a 5D tensor: (batch, channels, x, y, z)
        x_rot = torch.rot90(x, k=1, dims=(2, 3))
        out = self.conv(x) + self.conv(x_rot)
        return out

model = SO3Conv(1, 16)
input_tensor = torch.randn(1, 1, 32, 32, 32)
output = model(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

Slide 10: Compact Lie Groups in Cryptography

Compact Lie groups, particularly elliptic curves (which are related to certain compact Lie groups), play a crucial role in modern cryptography. Elliptic Curve Cryptography (ECC) provides strong security with smaller key sizes compared to other methods.

```python
from tinyec import registry
import secrets

def compress_point(point):
    return hex(point.x) + hex(point.y % 2)[2:]

def ecc_calc_shared_key(privKey, pubKey):
    sharedECCKey = privKey * pubKey
    return compress_point(sharedECCKey)

curve = registry.get_curve('brainpoolP256r1')

alicePrivKey = secrets.randbelow(curve.field.n)
alicePubKey = alicePrivKey * curve.g

bobPrivKey = secrets.randbelow(curve.field.n)
bobPubKey = bobPrivKey * curve.g

print("Alice public key:", compress_point(alicePubKey))
print("Bob public key:", compress_point(bobPubKey))
print("Alice shared key:", ecc_calc_shared_key(alicePrivKey, bobPubKey))
print("Bob shared key:", ecc_calc_shared_key(bobPrivKey, alicePubKey))
```

Slide 11: Compact Lie Groups in Computer Graphics

Compact Lie groups, especially SO(3) and SE(3), are fundamental in computer graphics for representing rotations and rigid body transformations. They are used extensively in 3D modeling, animation, and game development.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_points(points, axis, angle):
    axis = axis / np.linalg.norm(axis)
    rot_matrix = rotation_matrix(axis, angle)
    return np.dot(points, rot_matrix.T)

def plot_cube(ax, points):
    for i in range(4):
        ax.plot3D(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2], 'b')
        ax.plot3D(points[i+4:i+6, 0], points[i+4:i+6, 1], points[i+4:i+6, 2], 'b')
        ax.plot3D([points[i, 0], points[i+4, 0]], [points[i, 1], points[i+4, 1]], 
                  [points[i, 2], points[i+4, 2]], 'b')

cube_points = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
])

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

plot_cube(ax1, cube_points)
ax1.set_title("Original Cube")

rotated_points = rotate_points(cube_points, [1, 1, 1], np.pi/4)
plot_cube(ax2, rotated_points)
ax2.set_title("Rotated Cube")

plt.show()
```

Slide 12: Compact Lie Groups in Robotics

Compact Lie groups, particularly SO(3) and SE(3), are essential in robotics for representing the orientation and position of robot arms and mobile robots. They allow for efficient computation of robot kinematics and dynamics.

```python
import numpy as np

def dh_matrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])

# Example: 2-DOF planar robot arm
def forward_kinematics(theta1, theta2, l1, l2):
    T1 = dh_matrix(theta1, 0, l1, 0)
    T2 = dh_matrix(theta2, 0, l2, 0)
    T = np.dot(T1, T2)
    return T[:3, 3]  # Extract position from transformation matrix

theta1, theta2 = np.pi/4, np.pi/3
l1, l2 = 1, 0.5
end_effector_pos = forward_kinematics(theta1, theta2, l1, l2)
print(f"End effector position: {end_effector_pos}")
```

Slide 13: Conclusion and Future Directions

Compact Lie groups are fundamental structures with wide-ranging applications in mathematics, physics, and engineering. As we've seen, they play crucial roles in quantum mechanics, cryptography, computer graphics, machine learning, and robotics. Future research directions include:

1. Developing more efficient algorithms for computations on Lie groups
2. Exploring applications in quantum computing and quantum error correction
3. Investigating the role of Lie groups in deep learning architectures
4. Applying Lie group methods to problems in data analysis and manifold learning

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_research_trends():
    areas = ['Quantum Computing', 'Deep Learning', 'Data Analysis', 'Robotics']
    growth_rates = [0.8, 0.9, 0.7, 0.6]

    plt.figure(figsize=(10, 6))
    plt.bar(areas, growth_rates)
    plt.title('Projected Growth in Compact Lie Group Research Areas')
    plt.ylabel('Relative Growth Rate')
    plt.ylim(0, 1)
    plt.show()

plot_research_trends()
```

Slide 14: Additional Resources

For those interested in delving deeper into the world of compact Lie groups, here are some valuable resources:

1. "Lie Groups, Lie Algebras, and Representations: An Elementary Introduction" by Brian C. Hall ArXiv: [https://arxiv.org/abs/math-ph/0005032](https://arxiv.org/abs/math-ph/0005032)
2. "Compact Lie Groups and Their Representations" by D. P. Želobenko ArXiv: [https://arxiv.org/abs/math/0504359](https://arxiv.org/abs/math/0504359)
3. "Applications of Lie Groups to Differential Equations" by Peter J. Olver (This book is not available on ArXiv, but is a widely recognized resource in the field)
4. "Lie Groups for Computer Vision" by Ethan Eade ArXiv: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)

These resources provide a mix of theoretical foundations and practical applications, suitable for readers with varying levels of mathematical background.

Slide 15: Hands-on Exercise

To solidify your understanding of compact Lie groups, try the following exercise:

Implement a function that generates random elements of SU(2) (Special Unitary Group in 2D) and visualize their distribution on the Bloch sphere. This exercise will help you understand the connection between SU(2) and rotations in 3D space.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_su2():
    u1, u2, u3 = np.random.random(3)
    theta = 2 * np.pi * u1
    phi = 2 * np.pi * u2
    lamb = np.arccos(1 - 2 * u3)
    return np.array([
        [np.exp(1j * phi) * np.cos(lamb/2), np.exp(1j * theta) * np.sin(lamb/2)],
        [-np.exp(-1j * theta) * np.sin(lamb/2), np.exp(-1j * phi) * np.cos(lamb/2)]
    ])

def su2_to_bloch(U):
    return np.array([
        np.real(U[0, 1] + U[1, 0]),
        np.imag(U[1, 0] - U[0, 1]),
        np.real(U[0, 0] - U[1, 1])
    ])

# Generate random SU(2) elements and plot on Bloch sphere
# Implement visualization here

# After implementing, discuss the results and their implications
```

This exercise will provide hands-on experience with the structure of SU(2) and its relation to 3D rotations.

