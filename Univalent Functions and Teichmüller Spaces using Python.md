## Univalent Functions and Teichmüller Spaces using Python
Slide 1: Introduction to Univalent Functions

Univalent functions are a fundamental concept in complex analysis, characterized by their one-to-one property. A function f(z) is univalent in a domain D if it never takes the same value twice in D. In other words, for any two distinct points z1 and z2 in D, f(z1) ≠ f(z2).

```python
import matplotlib.pyplot as plt
import numpy as np

def is_univalent(f, domain):
    values = set()
    for z in domain:
        fz = f(z)
        if fz in values:
            return False
        values.add(fz)
    return True

# Example: f(z) = z^2 is not univalent on the entire complex plane
f = lambda z: z**2
domain = [complex(x, y) for x in np.linspace(-1, 1, 10) for y in np.linspace(-1, 1, 10)]

print(f"Is f(z) = z^2 univalent? {is_univalent(f, domain)}")
```

Slide 2: Properties of Univalent Functions

Univalent functions possess several important properties:

1. They are injective (one-to-one) in their domain.
2. They preserve the topology of the domain.
3. They have a non-vanishing derivative in their domain.

Let's visualize the behavior of a univalent function:

```python
import numpy as np
import matplotlib.pyplot as plt

def univalent_function(z):
    return z + 1/z

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

W = univalent_function(Z)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.title("Original Domain")
plt.contourf(X, Y, np.abs(Z), cmap='viridis')
plt.colorbar(label='|z|')

plt.subplot(122)
plt.title("Transformed Domain")
plt.contourf(W.real, W.imag, np.abs(W), cmap='viridis')
plt.colorbar(label='|f(z)|')

plt.tight_layout()
plt.show()
```

Slide 3: Koebe Function: A Canonical Example

The Koebe function, k(z) = z / (1-z)^2, is a crucial example in the theory of univalent functions. It maps the unit disk onto the complex plane minus a slit along the negative real axis from -1/4 to -∞.

```python
import numpy as np
import matplotlib.pyplot as plt

def koebe(z):
    return z / (1 - z)**2

theta = np.linspace(0, 2*np.pi, 1000)
z = np.exp(1j * theta)
w = koebe(z)

plt.figure(figsize=(10, 5))
plt.plot(w.real, w.imag)
plt.title("Image of the unit circle under the Koebe function")
plt.xlabel("Re(w)")
plt.ylabel("Im(w)")
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 4: Schwarz Lemma and Its Applications

The Schwarz Lemma is a powerful tool in complex analysis, especially for studying univalent functions. It states that for an analytic function f(z) mapping the unit disk to itself with f(0) = 0, we have |f(z)| ≤ |z| for all z in the unit disk.

```python
import numpy as np
import matplotlib.pyplot as plt

def schwarz_example(z):
    return 0.5 * z * (3 - z**2) / (3 - z**2 / 2)

z = np.linspace(-1, 1, 1000)
w = schwarz_example(z)

plt.figure(figsize=(8, 8))
circle = plt.Circle((0, 0), 1, fill=False)
plt.gca().add_artist(circle)
plt.plot(z, w, label='f(z)')
plt.plot(z, z, '--', label='|z|')
plt.plot(z, -z, '--')
plt.legend()
plt.title("Function satisfying Schwarz Lemma")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 5: Area Theorem and Coefficient Bounds

The Area Theorem provides important bounds on the coefficients of univalent functions. For a univalent function f(z) = z + a2z^2 + a3z^3 + ... in the unit disk, we have ∑(n|an|^2) ≤ 1.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_coefficients(n):
    coeffs = np.random.rand(n)
    norm = np.sqrt(np.sum(np.arange(2, n+1) * coeffs**2))
    return coeffs / norm

def plot_coefficient_bounds(n):
    coeffs = generate_coefficients(n)
    plt.figure(figsize=(10, 6))
    plt.bar(range(2, n+1), coeffs)
    plt.title(f"Coefficients satisfying Area Theorem (n={n})")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient value")
    plt.show()
    
    print(f"Sum of n|an|^2: {np.sum(np.arange(2, n+1) * coeffs**2)}")

plot_coefficient_bounds(10)
```

Slide 6: Bieberbach Conjecture and de Branges' Theorem

The Bieberbach Conjecture, now proven and known as de Branges' Theorem, states that for a univalent function f(z) = z + a2z^2 + a3z^3 + ... in the unit disk, |an| ≤ n for all n ≥ 2. This was a major open problem in complex analysis for over 70 years.

```python
import numpy as np
import matplotlib.pyplot as plt

def koebe_coefficients(n):
    return [k / (k+1) for k in range(1, n+1)]

def plot_koebe_coefficients(n):
    coeffs = koebe_coefficients(n)
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n+1), coeffs)
    plt.plot(range(1, n+1), range(1, n+1), 'r--', label='|an| = n')
    plt.title(f"Koebe function coefficients (n={n})")
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.show()

plot_koebe_coefficients(10)
```

Slide 7: Introduction to Teichmüller Spaces

Teichmüller spaces are complex manifolds that parametrize certain geometric structures on surfaces. They play a crucial role in the study of Riemann surfaces and quasiconformal mappings.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def teichmuller_embedding(tau, n):
    x = np.real(tau)
    y = np.imag(tau)
    z = np.abs(tau - 1j) * np.abs(tau + 1j) * np.abs(tau - 1)
    return x, y, z

tau = [complex(x, y) for x in np.linspace(-2, 2, 100) for y in np.linspace(0, 2, 100)]
X, Y, Z = zip(*[teichmuller_embedding(t, 1) for t in tau])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=1)
ax.set_xlabel('Re(τ)')
ax.set_ylabel('Im(τ)')
ax.set_zlabel('Embedding')
plt.title("Embedding of the Teichmüller space of once-punctured tori")
plt.show()
```

Slide 8: Quasiconformal Mappings and Teichmüller Theory

Quasiconformal mappings are a generalization of conformal mappings, allowing for a controlled amount of distortion. They are fundamental in Teichmüller theory, providing a way to relate different complex structures on a surface.

```python
import numpy as np
import matplotlib.pyplot as plt

def quasiconformal_grid(K, n=20):
    x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    z = x + 1j*y
    w = z * np.abs(z)**(K-1)
    return x, y, w.real, w.imag

K_values = [1, 1.5, 2]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, K in zip(axes, K_values):
    x, y, u, v = quasiconformal_grid(K)
    ax.plot(u, v, 'b', x, y, 'r', linewidth=0.5)
    ax.plot(u.T, v.T, 'b', x.T, y.T, 'r', linewidth=0.5)
    ax.set_title(f'K = {K}')
    ax.set_aspect('equal')
    ax.axis('off')

plt.suptitle("Quasiconformal mappings with different dilatations")
plt.tight_layout()
plt.show()
```

Slide 9: Moduli Spaces and Teichmüller Spaces

Moduli spaces are quotients of Teichmüller spaces by the action of the mapping class group. They represent the space of all complex structures on a surface up to isomorphism.

```python
import numpy as np
import matplotlib.pyplot as plt

def fundamental_domain():
    x = np.linspace(-0.5, 0.5, 1000)
    y = np.sqrt(1 - x**2)
    return x, y

def modular_group_action(z, n):
    return [(z + k) / (n*z + n*k + 1) for k in range(-n, n+1)]

x, y = fundamental_domain()
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'k-')
plt.plot(x, -y, 'k-')
plt.plot([-0.5, 0.5], [0, 0], 'k-')

for n in range(1, 4):
    for z in [0.5 + 0.5j, -0.5 + 0.5j]:
        orbit = modular_group_action(z, n)
        x_orbit, y_orbit = zip(*[(z.real, z.imag) for z in orbit])
        plt.scatter(x_orbit, y_orbit, label=f'n={n}')

plt.legend()
plt.title("Action of modular group on the upper half-plane")
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.axis('equal')
plt.xlim(-2, 2)
plt.ylim(0, 2)
plt.grid(True)
plt.show()
```

Slide 10: Beltrami Differentials and Quadratic Differentials

Beltrami differentials and quadratic differentials are important objects in Teichmüller theory. Beltrami differentials represent infinitesimal deformations of complex structures, while quadratic differentials are dual to Beltrami differentials.

```python
import numpy as np
import matplotlib.pyplot as plt

def beltrami_differential(z, mu):
    return mu * np.conj(z) / z

def quadratic_differential(z):
    return 1 / z**2

z = np.linspace(-2, 2, 100) + 1j * np.linspace(-2, 2, 100)[:, np.newaxis]
mu = 0.5

beltrami = beltrami_differential(z, mu)
quadratic = quadratic_differential(z)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

im1 = ax1.imshow(np.abs(beltrami), extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
ax1.set_title("Magnitude of Beltrami differential")
fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow(np.abs(quadratic), extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
ax2.set_title("Magnitude of quadratic differential")
fig.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
```

Slide 11: Teichmüller Metric and Extremal Quasiconformal Mappings

The Teichmüller metric is a natural metric on Teichmüller space, defined using extremal quasiconformal mappings. It measures the "distance" between different complex structures on a surface.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

def teichmuller_distance(K):
    return 0.5 * np.log(K)

def extremal_dilatation(t):
    return np.exp(2*t)

t = np.linspace(0, 2, 100)
K = extremal_dilatation(t)
d = teichmuller_distance(K)

plt.figure(figsize=(10, 6))
plt.plot(t, d)
plt.title("Teichmüller distance as a function of extremal dilatation")
plt.xlabel("log K / 2")
plt.ylabel("Teichmüller distance")
plt.grid(True)
plt.show()
```

Slide 12: Applications: Conformal Welding and Slit Mappings

Conformal welding and slit mappings are important applications of Teichmüller theory in complex analysis. They involve gluing together or cutting Riemann surfaces along prescribed curves.

```python
import numpy as np
import matplotlib.pyplot as plt

def slit_mapping(z, a):
    return z + a**2 / z

a = 0.5
theta = np.linspace(0, 2*np.pi, 1000)
z = np.exp(1j * theta)
w = slit_mapping(z, a)

plt.figure(figsize=(10, 8))
plt.plot(w.real, w.imag)
plt.title("Slit mapping of the unit circle")
plt.xlabel("Re(w)")
plt.ylabel("Im(w)")
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 13: Real-life Example: Shape Analysis in Computer Vision

Teichmüller spaces find applications in computer vision for shape analysis. By representing shapes as points in Teichmüller space, we can compare and classify shapes in a geometrically meaningful way.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_shape(t, noise_level=0.05):
    x = np.cos(t) + 0.5 * np.cos(2*t) + noise_level * np.random.randn(len(t))
    y = np.sin(t) + 0.5 * np.sin(2*t) + noise_level * np.random.randn(len(t))
    return x, y

def compute_shape_descriptor(x, y):
    # Simplified shape descriptor using curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return np.mean(curvature), np.std(curvature)

t = np.linspace(0, 2*np.pi, 100)
shapes = [generate_shape(t) for _ in range(5)]
descriptors = [compute_shape_descriptor(x, y) for x, y in shapes]

plt.figure(figsize=(12, 6))
for i, (x, y) in enumerate(shapes):
    plt.subplot(1, 5, i+1)
    plt.plot(x, y)
    plt.title(f"Shape {i+1}")
    plt.axis('equal')

plt.tight_layout()
plt.show()

print("Shape descriptors:")
for i, (mean_curv, std_curv) in enumerate(descriptors):
    print(f"Shape {i+1}: Mean curvature = {mean_curv:.3f}, Std curvature = {std_curv:.3f}")
```

Slide 14: Real-life Example: Mesh Parameterization in Computer Graphics

Teichmüller theory is used in computer graphics for mesh parameterization, which involves mapping 3D surfaces onto 2D domains while minimizing distortion. This is crucial for texture mapping and surface analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_torus(R, r, n=50, m=100):
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, 2*np.pi, m)
    theta, phi = np.meshgrid(theta, phi)
    
    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    return x, y, z

def parameterize_torus(x, y, z):
    u = np.arctan2(y, x)
    v = np.arctan2(z, np.sqrt(x**2 + y**2) - (R + r))
    return u, v

R, r = 3, 1  # Major and minor radii of the torus
x, y, z = generate_torus(R, r)
u, v = parameterize_torus(x, y, z)

fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x, y, z, cmap='viridis')
ax1.set_title("3D Torus")

ax2 = fig.add_subplot(132)
im = ax2.pcolormesh(u, v, z, cmap='viridis')
ax2.set_title("Parameterized Torus")
ax2.set_xlabel("u")
ax2.set_ylabel("v")
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Univalent Functions and Teichmüller Spaces, here are some valuable resources:

1. "Univalent Functions and Teichmüller Spaces" by Kurt Strebel (Springer, 1984)
2. "Quasiconformal Mappings and Riemann Surfaces" by Olli Lehto (ArXiv:1006.2824)
3. "An Introduction to Teichmüller Spaces" by Yoichi Imayoshi and Masahiko Taniguchi (Springer, 1992)
4. "Handbook of Teichmüller Theory" edited by Athanase Papadopoulos (European Mathematical Society, 2007)

These resources provide comprehensive coverage of the topics discussed in this presentation and offer more advanced material for further study.

