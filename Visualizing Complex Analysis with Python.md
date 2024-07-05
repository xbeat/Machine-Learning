## Visualizing Complex Analysis with Python

Slide 1: Introduction to Complex Analysis

Complex analysis is a branch of mathematics that studies functions of complex numbers. It has wide-ranging applications in physics, engineering, and other fields of mathematics.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_plane(z):
    return z.real, z.imag

z = 2 + 3j
x, y = complex_plane(z)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.title('Complex Number on Complex Plane')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)
plt.show()
```

Slide 2: Complex Numbers

Complex numbers are numbers of the form a + bi, where a and b are real numbers and i is the imaginary unit (i² = -1).

```python
class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
    
    def __str__(self):
        return f"{self.real} + {self.imag}i"

z1 = ComplexNumber(3, 4)
print(f"Complex number: {z1}")
```

Slide 3: Basic Operations on Complex Numbers

Complex numbers support addition, subtraction, multiplication, and division.

```python
def add_complex(z1, z2):
    return ComplexNumber(z1.real + z2.real, z1.imag + z2.imag)

def multiply_complex(z1, z2):
    real = z1.real * z2.real - z1.imag * z2.imag
    imag = z1.real * z2.imag + z1.imag * z2.real
    return ComplexNumber(real, imag)

z1 = ComplexNumber(1, 2)
z2 = ComplexNumber(3, 4)
print(f"Sum: {add_complex(z1, z2)}")
print(f"Product: {multiply_complex(z1, z2)}")
```

Slide 4: Polar Form of Complex Numbers

Complex numbers can be represented in polar form (r, θ), where r is the magnitude and θ is the argument.

```python
import cmath

def polar_form(z):
    r = abs(z)
    theta = cmath.phase(z)
    return r, theta

z = 1 + 1j
r, theta = polar_form(z)
print(f"Polar form: r = {r:.2f}, θ = {theta:.2f} radians")
```

Slide 5: Euler's Formula

Euler's formula establishes a fundamental relationship between trigonometric functions and the complex exponential function.

```python
import numpy as np

def euler_formula(x):
    return np.cos(x) + 1j * np.sin(x)

x = np.pi / 4
result = euler_formula(x)
print(f"e^(i*π/4) = {result:.4f}")
```

Slide 6: Complex Functions

Complex functions map complex numbers to complex numbers. They can be visualized using domain coloring.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_function(z):
    return z**2 + 2*z + 1

x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

W = complex_function(Z)
plt.imshow(np.angle(W), cmap='hsv', extent=[-2, 2, -2, 2])
plt.title('Domain Coloring of f(z) = z² + 2z + 1')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.colorbar(label='Argument')
plt.show()
```

Slide 7: Analytic Functions

Analytic functions are complex functions that are differentiable at every point in their domain.

```python
def is_analytic(f, z, h=1e-6):
    dz = h + 1j*h
    df_dx = (f(z + h) - f(z)) / h
    df_dy = (f(z + 1j*h) - f(z)) / (1j*h)
    return np.isclose(df_dx, df_dy.conjugate())

def f(z):
    return z**2

z = 1 + 1j
print(f"Is f(z) = z² analytic at {z}? {is_analytic(f, z)}")
```

Slide 8: Cauchy-Riemann Equations

The Cauchy-Riemann equations are necessary conditions for a complex function to be analytic.

```python
def cauchy_riemann(u, v, x, y, h=1e-6):
    du_dx = (u(x + h, y) - u(x, y)) / h
    du_dy = (u(x, y + h) - u(x, y)) / h
    dv_dx = (v(x + h, y) - v(x, y)) / h
    dv_dy = (v(x, y + h) - v(x, y)) / h
    return np.isclose(du_dx, dv_dy) and np.isclose(du_dy, -dv_dx)

def u(x, y):
    return x**2 - y**2

def v(x, y):
    return 2*x*y

x, y = 1, 1
print(f"Do u and v satisfy C-R equations at ({x}, {y})? {cauchy_riemann(u, v, x, y)}")
```

Slide 9: Complex Integration

Complex integration involves integrating complex functions along paths in the complex plane.

```python
import scipy.integrate as integrate

def integrand(t):
    return np.exp(1j * t)

def path(t):
    return np.cos(t) + 1j * np.sin(t)

result, _ = integrate.quad(lambda t: integrand(t) * np.abs(path(t)), 0, 2*np.pi)
print(f"∫exp(iz)dz along the unit circle: {result:.4f}")
```

Slide 10: Cauchy's Integral Formula

Cauchy's Integral Formula relates the values of an analytic function inside a closed contour to the values on the contour.

```python
def cauchy_integral(f, z0, radius, n_points=1000):
    theta = np.linspace(0, 2*np.pi, n_points)
    z = z0 + radius * np.exp(1j * theta)
    integrand = f(z) / (z - z0)
    integral = np.trapz(integrand, theta) * 1j / (2 * np.pi)
    return integral

def f(z):
    return 1 / (z**2 + 1)

z0 = 0 + 0j
radius = 2
result = cauchy_integral(f, z0, radius)
print(f"f({z0}) by Cauchy's formula: {result:.4f}")
print(f"f({z0}) directly: {f(z0):.4f}")
```

Slide 11: Residue Theorem

The residue theorem is a powerful tool for evaluating complex integrals.

```python
def residue(f, z0, order=1):
    def coefficient(n):
        return np.polyder(lambda z: f(z + z0), n)(0) / np.math.factorial(n)
    return coefficient(order - 1)

def f(z):
    return 1 / (z**2 + 1)

z0 = 1j
res = residue(f, z0)
print(f"Residue of 1/(z²+1) at z = i: {res:.4f}")
```

Slide 12: Laurent Series

Laurent series extend power series to include negative powers, useful for analyzing functions near singularities.

```python
from sympy import Symbol, series, oo

def laurent_series(f, z0, n):
    z = Symbol('z')
    return series(f(z), (z, z0), n=n)

def f(z):
    return 1 / (z**2 + 1)

z0 = 0
n = 5
laurent = laurent_series(f, z0, n)
print(f"Laurent series of 1/(z²+1) around z = 0:")
print(laurent)
```

Slide 13: Applications in Signal Processing

Complex analysis is crucial in signal processing, particularly in Fourier analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform(t, signal):
    return np.fft.fft(signal)

t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

ft = fourier_transform(t, signal)
freq = np.fft.fftfreq(len(t), t[1] - t[0])

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(t, signal)
plt.title('Original Signal')
plt.subplot(212)
plt.plot(freq, np.abs(ft))
plt.title('Frequency Spectrum')
plt.xlim(0, 30)
plt.show()
```

Slide 14: Applications in Fluid Dynamics

Complex analysis is used in fluid dynamics to model 2D fluid flow.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_potential(z, strength):
    return strength * np.log(z)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

W = complex_potential(Z, 1)
phi = W.real
psi = W.imag

plt.figure(figsize=(10, 8))
plt.contour(X, Y, phi, colors='blue', levels=20)
plt.contour(X, Y, psi, colors='red', levels=20)
plt.title('Flow Around a Point Vortex')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
```

Slide 15: Additional Resources

For further study in complex analysis, consider these resources:

1. ArXiv.org paper: "A Visual Introduction to Complex Analysis" by Elias Wegert URL: [https://arxiv.org/abs/1208.3282](https://arxiv.org/abs/1208.3282)
2. ArXiv.org paper: "Complex Analysis: A Visual and Interactive Introduction" by Juan Carlos Ponce Campuzano URL: [https://arxiv.org/abs/1905.01239](https://arxiv.org/abs/1905.01239)

These papers provide visual and interactive approaches to learning complex analysis, which can complement the coding examples we've explored in this presentation.

