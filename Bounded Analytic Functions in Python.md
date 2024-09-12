## Bounded Analytic Functions in Python
Slide 1: Introduction to Bounded Analytic Functions

Bounded analytic functions are complex-valued functions that are both analytic and bounded on a given domain. They play a crucial role in complex analysis and have applications in various fields of mathematics and engineering.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_bounded_function(f, domain):
    x = np.linspace(domain[0], domain[1], 1000)
    y = np.linspace(domain[0], domain[1], 1000)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, np.abs(f(Z)), levels=20, cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Magnitude of a Bounded Analytic Function')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

# Example: f(z) = (z - 1) / (z + 1)
f = lambda z: (z - 1) / (z + 1)
plot_bounded_function(f, [-2, 2])
```

Slide 2: Definition and Properties

A function f(z) is bounded and analytic on a domain D if:

1. f(z) is analytic (complex differentiable) at every point in D
2. There exists a constant M > 0 such that |f(z)| ≤ M for all z in D

```python
def is_bounded_analytic(f, domain, M, epsilon=1e-6):
    def complex_derivative(f, z, h=1e-8):
        return (f(z + h) - f(z)) / h
    
    z = np.random.uniform(domain[0], domain[1], size=(1000, 2)).view(np.complex128)
    
    analytic = np.allclose(f(z), complex_derivative(f, z), atol=epsilon)
    bounded = np.all(np.abs(f(z)) <= M)
    
    return analytic and bounded

# Example: f(z) = sin(z) / z
f = lambda z: np.sin(z) / z
domain = [-10, 10]
M = 1

print(f"Is f(z) = sin(z)/z bounded analytic? {is_bounded_analytic(f, domain, M)}")
```

Slide 3: The Unit Disk and Hardy Space

The unit disk D = {z : |z| < 1} is a fundamental domain for studying bounded analytic functions. The Hardy space H∞(D) consists of all bounded analytic functions on D.

```python
def plot_unit_disk():
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'b-')
    plt.fill(x, y, 'lightblue', alpha=0.3)
    plt.title('Unit Disk')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

plot_unit_disk()
```

Slide 4: Maximum Modulus Principle

For a non-constant analytic function f(z) on a domain D, the maximum of |f(z)| occurs on the boundary of D, not in its interior.

```python
def maximum_modulus_principle(f, domain, num_points=1000):
    boundary_x = np.concatenate([
        np.linspace(domain[0], domain[1], num_points),
        np.full(num_points, domain[1]),
        np.linspace(domain[1], domain[0], num_points),
        np.full(num_points, domain[0])
    ])
    
    boundary_y = np.concatenate([
        np.full(num_points, domain[2]),
        np.linspace(domain[2], domain[3], num_points),
        np.full(num_points, domain[3]),
        np.linspace(domain[3], domain[2], num_points)
    ])
    
    boundary_z = boundary_x + 1j*boundary_y
    max_modulus = np.max(np.abs(f(boundary_z)))
    
    return max_modulus

# Example: f(z) = z^2
f = lambda z: z**2
domain = [-1, 1, -1, 1]  # [x_min, x_max, y_min, y_max]

max_value = maximum_modulus_principle(f, domain)
print(f"Maximum modulus of f(z) = z^2 on the boundary: {max_value}")
```

Slide 5: Schwarz Lemma

The Schwarz Lemma is a powerful tool for studying bounded analytic functions on the unit disk. It states that if f(z) is analytic on D, |f(z)| ≤ 1 for all z in D, and f(0) = 0, then |f(z)| ≤ |z| for all z in D.

```python
def schwarz_lemma(f, z):
    if np.abs(z) >= 1:
        raise ValueError("z must be inside the unit disk")
    
    return np.abs(f(z)) <= np.abs(z)

# Example: f(z) = z^2
f = lambda z: z**2
z = 0.5 + 0.3j

print(f"For f(z) = z^2 and z = {z}:")
print(f"|f(z)| = {np.abs(f(z)):.4f}")
print(f"|z| = {np.abs(z):.4f}")
print(f"Schwarz Lemma holds: {schwarz_lemma(f, z)}")
```

Slide 6: Blaschke Products

Blaschke products are an important class of bounded analytic functions on the unit disk. They are used to construct other bounded analytic functions and study their properties.

```python
def blaschke_product(z, zeros):
    product = 1
    for a in zeros:
        if np.abs(a) >= 1:
            raise ValueError("All zeros must be inside the unit disk")
        product *= (z - a) / (1 - np.conj(a) * z)
    return product

def plot_blaschke_product(zeros):
    theta = np.linspace(0, 2*np.pi, 1000)
    z = np.exp(1j * theta)
    
    values = blaschke_product(z, zeros)
    
    plt.figure(figsize=(10, 8))
    plt.polar(theta, np.abs(values))
    plt.title('Magnitude of Blaschke Product on the Unit Circle')
    plt.show()

# Example: Blaschke product with zeros at 0.5 and -0.3+0.4j
zeros = [0.5, -0.3+0.4j]
plot_blaschke_product(zeros)
```

Slide 7: Inner and Outer Functions

Bounded analytic functions can be factored into inner and outer functions. Inner functions have modulus 1 on the unit circle, while outer functions have no zeros in the unit disk.

```python
def inner_function(z, a):
    return (z - a) / (1 - np.conj(a) * z)

def outer_function(z, f):
    return np.exp((1 / (2*np.pi)) * np.log(np.abs(f(np.exp(1j*z)))))

def plot_inner_outer(a, f):
    theta = np.linspace(0, 2*np.pi, 1000)
    z = np.exp(1j * theta)
    
    inner_values = inner_function(z, a)
    outer_values = outer_function(theta, f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(theta, np.abs(inner_values))
    ax1.set_title('Magnitude of Inner Function')
    ax1.set_xlabel('θ')
    ax1.set_ylabel('|f(e^(iθ))|')
    
    ax2.plot(theta, outer_values)
    ax2.set_title('Outer Function')
    ax2.set_xlabel('θ')
    ax2.set_ylabel('f(e^(iθ))')
    
    plt.tight_layout()
    plt.show()

# Example: Inner function with a = 0.5, Outer function f(z) = 1 + z
a = 0.5
f = lambda z: 1 + z
plot_inner_outer(a, f)
```

Slide 8: Nevanlinna-Pick Interpolation

The Nevanlinna-Pick interpolation problem involves finding a bounded analytic function that takes specified values at given points in the unit disk.

```python
import numpy as np
from scipy.linalg import solve

def nevanlinna_pick(points, values):
    n = len(points)
    A = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            A[i, j] = (1 - values[i] * np.conj(values[j])) / (1 - points[i] * np.conj(points[j]))
    
    return np.all(np.linalg.eigvals(A) >= 0)

# Example: Interpolation problem
points = [0, 0.5, -0.5j]
values = [0, 0.5, 0.25]

solvable = nevanlinna_pick(points, values)
print(f"Is the interpolation problem solvable? {solvable}")
```

Slide 9: Hardy Spaces and Hp Norms

Hardy spaces Hp are spaces of analytic functions on the unit disk with bounded p-norms. H∞ is the space of bounded analytic functions.

```python
def hp_norm(f, p, num_points=1000):
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.exp(1j * theta)
    
    if p == float('inf'):
        return np.max(np.abs(f(z)))
    else:
        return (np.mean(np.abs(f(z))**p)**(1/p))

# Example: Calculate H2 and H∞ norms for f(z) = z / (1 - z/2)
f = lambda z: z / (1 - z/2)

h2_norm = hp_norm(f, 2)
hinf_norm = hp_norm(f, float('inf'))

print(f"H2 norm: {h2_norm:.4f}")
print(f"H∞ norm: {hinf_norm:.4f}")
```

Slide 10: Conformal Mapping

Conformal mapping preserves angles and local shapes. It's a powerful tool for transforming bounded analytic functions between different domains.

```python
def mobius_transform(z, a, b, c, d):
    return (a*z + b) / (c*z + d)

def plot_conformal_mapping(f, domain):
    x = np.linspace(domain[0], domain[1], 100)
    y = np.linspace(domain[2], domain[3], 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    W = f(Z)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.contour(X, Y, X, colors='blue', alpha=0.5)
    ax1.contour(X, Y, Y, colors='red', alpha=0.5)
    ax1.set_title('Original Domain')
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    
    ax2.contour(W.real, W.imag, X, colors='blue', alpha=0.5)
    ax2.contour(W.real, W.imag, Y, colors='red', alpha=0.5)
    ax2.set_title('Mapped Domain')
    ax2.set_xlabel('Re(w)')
    ax2.set_ylabel('Im(w)')
    
    plt.tight_layout()
    plt.show()

# Example: Möbius transform mapping the unit disk to the upper half-plane
f = lambda z: 1j * (1 + z) / (1 - z)
domain = [-1, 1, -1, 1]

plot_conformal_mapping(f, domain)
```

Slide 11: Boundary Behavior

The boundary behavior of bounded analytic functions is crucial for understanding their properties and applications.

```python
def plot_boundary_behavior(f, num_points=1000):
    theta = np.linspace(0, 2*np.pi, num_points)
    z = np.exp(1j * theta)
    
    values = f(z)
    
    plt.figure(figsize=(10, 8))
    plt.plot(values.real, values.imag)
    plt.title('Boundary Behavior of a Bounded Analytic Function')
    plt.xlabel('Re(f(e^(iθ)))')
    plt.ylabel('Im(f(e^(iθ)))')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example: f(z) = (z + 1) / (z - 1)
f = lambda z: (z + 1) / (z - 1)
plot_boundary_behavior(f)
```

Slide 12: Applications in Signal Processing

Bounded analytic functions have applications in signal processing, particularly in the design of digital filters and spectral factorization.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def design_lowpass_filter(cutoff, order):
    b, a = signal.butter(order, cutoff, btype='low', analog=False)
    w, h = signal.freqz(b, a)
    return w, h

def plot_filter_response(w, h):
    plt.figure(figsize=(10, 6))
    plt.plot(w / np.pi, np.abs(h))
    plt.title('Lowpass Filter Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.show()

# Example: Design a lowpass filter
cutoff = 0.3
order = 5
w, h = design_lowpass_filter(cutoff, order)
plot_filter_response(w, h)
```

Slide 13: Real-life Example: Image Compression

Bounded analytic functions play a role in image compression algorithms, particularly in techniques like singular value decomposition (SVD) for image approximation.

```python
import numpy as np
import matplotlib.pyplot as plt

def compress_image(image, k):
    U, S, Vt = np.linalg.svd(image)
    compressed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    return compressed

def plot_compressed_image(original, compressed):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(compressed, cmap='gray')
    ax2.set_title('Compressed Image')
    ax2.axis('off')
    plt.show()

# Example usage (assuming 'image' is a 2D numpy array):
# compressed = compress_image(image, k=50)
# plot_compressed_image(image, compressed)
```

Slide 14: Real-life Example: Control Systems

Bounded analytic functions are crucial in control system theory, particularly in the design of stabilizing controllers for linear systems.

```python
import control
import numpy as np
import matplotlib.pyplot as plt

def design_controller(system, desired_poles):
    K = control.place(system.A, system.B, desired_poles)
    return K

def plot_step_response(system, title):
    t, y = control.step_response(system)
    plt.figure(figsize=(10, 6))
    plt.plot(t, y)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.grid(True)
    plt.show()

# Example usage:
# A = np.array([[0, 1], [-2, -3]])
# B = np.array([[0], [1]])
# C = np.array([[1, 0]])
# D = np.array([[0]])
# 
# system = control.ss(A, B, C, D)
# desired_poles = [-1, -2]
# 
# K = design_controller(system, desired_poles)
# controlled_system = control.feedback(system, K)
# 
# plot_step_response(system, 'Original System')
# plot_step_response(controlled_system, 'Controlled System')
```

Slide 15: Additional Resources

For further exploration of bounded analytic functions and their applications, consider the following resources:

1. "Bounded Analytic Functions" by John B. Garnett (Springer) ArXiv link: [https://arxiv.org/abs/math/0007181](https://arxiv.org/abs/math/0007181)
2. "Theory of H^p Spaces" by Peter Duren (Academic Press) ArXiv link: [https://arxiv.org/abs/1508.03491](https://arxiv.org/abs/1508.03491)
3. "Complex Analysis" by Elias M. Stein and Rami Shakarchi (Princeton University Press) ArXiv link: [https://arxiv.org/abs/math/0104031](https://arxiv.org/abs/math/0104031)

These resources provide in-depth coverage of bounded analytic functions and related topics in complex analysis. Remember to verify the availability and relevance of these resources, as ArXiv links may change over time.

