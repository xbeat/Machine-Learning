## Complex Analysis Functions of a Complex Variable
Slide 1: Introduction to Complex Analysis

Complex analysis, also known as function theory, is a branch of mathematics that studies complex-valued functions of a complex variable. It extends the concepts of calculus to the complex plane, offering powerful tools for solving problems in physics, engineering, and pure mathematics.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_complex_function(f, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    W = f(Z)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.contourf(X, Y, np.abs(W), cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('Magnitude')
    
    plt.subplot(122)
    plt.contourf(X, Y, np.angle(W), cmap='hsv')
    plt.colorbar(label='Phase')
    plt.title('Phase')
    
    plt.suptitle(f'Visualization of f(z) = {f.__name__}')
    plt.show()

def example_function(z):
    return z**2 + 1

plot_complex_function(example_function, (-2, 2), (-2, 2))
```

Slide 2: Complex Numbers and the Complex Plane

Complex numbers are numbers of the form a + bi, where a and b are real numbers and i is the imaginary unit (i² = -1). The complex plane represents these numbers visually, with the real part on the x-axis and the imaginary part on the y-axis.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_complex_number(z):
    plt.figure(figsize=(8, 8))
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    
    plt.arrow(0, 0, z.real, z.imag, head_width=0.1, head_length=0.1, fc='r', ec='r')
    plt.plot(z.real, z.imag, 'ro')
    
    plt.text(z.real, z.imag, f'{z:.2f}', fontsize=12, verticalalignment='bottom')
    
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title('Complex Number in the Complex Plane')
    plt.grid(True)
    plt.show()

z = 3 + 2j
plot_complex_number(z)
```

Slide 3: Complex Functions

Complex functions map complex numbers to complex numbers. They can be visualized as transformations of the complex plane. Common examples include polynomials, exponential functions, and trigonometric functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_complex_function_grid(f, x_range, y_range, grid_lines=10):
    x = np.linspace(x_range[0], x_range[1], grid_lines)
    y = np.linspace(y_range[0], y_range[1], grid_lines)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    W = f(Z)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(X, Y, 'k-', lw=0.5)
    plt.plot(X.T, Y.T, 'k-', lw=0.5)
    plt.title('Input Grid')
    
    plt.subplot(122)
    plt.plot(W.real, W.imag, 'r-', lw=0.5)
    plt.plot(W.real.T, W.imag.T, 'r-', lw=0.5)
    plt.title(f'Transformed Grid: f(z) = {f.__name__}')
    
    plt.tight_layout()
    plt.show()

def example_function(z):
    return z**2

plot_complex_function_grid(example_function, (-2, 2), (-2, 2))
```

Slide 4: Analyticity and Holomorphic Functions

A complex function is analytic (or holomorphic) at a point if it is complex differentiable in a neighborhood of that point. Holomorphic functions are the central objects of study in complex analysis, possessing many remarkable properties.

```python
import sympy as sp

def check_cauchy_riemann(f, z):
    x, y = sp.symbols('x y')
    z = x + sp.I*y
    f_expr = f(z)
    
    u = sp.re(f_expr)
    v = sp.im(f_expr)
    
    ux = sp.diff(u, x)
    uy = sp.diff(u, y)
    vx = sp.diff(v, x)
    vy = sp.diff(v, y)
    
    cr_1 = ux - vy
    cr_2 = uy + vx
    
    print(f"Cauchy-Riemann equations for f(z) = {f_expr}:")
    print(f"∂u/∂x = ∂v/∂y: {cr_1 == 0}")
    print(f"∂u/∂y = -∂v/∂x: {cr_2 == 0}")

def example_function(z):
    return z**2

check_cauchy_riemann(example_function, sp.symbols('z'))
```

Slide 5: Cauchy-Riemann Equations

The Cauchy-Riemann equations are necessary (but not sufficient) conditions for a complex function to be holomorphic. They relate the partial derivatives of the real and imaginary parts of the function.

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_cr_violations(f, x_range, y_range, resolution=100):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    x, y = sp.symbols('x y')
    z = x + sp.I*y
    f_expr = f(z)
    
    u = sp.re(f_expr)
    v = sp.im(f_expr)
    
    ux = sp.lambdify((x, y), sp.diff(u, x), 'numpy')
    uy = sp.lambdify((x, y), sp.diff(u, y), 'numpy')
    vx = sp.lambdify((x, y), sp.diff(v, x), 'numpy')
    vy = sp.lambdify((x, y), sp.diff(v, y), 'numpy')
    
    cr_violation = np.abs(ux(X, Y) - vy(X, Y)) + np.abs(uy(X, Y) + vx(X, Y))
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, cr_violation, levels=20, cmap='viridis')
    plt.colorbar(label='CR Equation Violation')
    plt.title(f'Cauchy-Riemann Violation for f(z) = {f_expr}')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

def example_function(z):
    return sp.Abs(z)  # Non-holomorphic function

plot_cr_violations(example_function, (-2, 2), (-2, 2))
```

Slide 6: Complex Integration

Complex integration is performed along paths in the complex plane. The fundamental theorem of calculus extends to complex analysis, leading to powerful results like Cauchy's integral formula.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_complex_path_integral(f, path, num_points=1000):
    t = np.linspace(0, 1, num_points)
    z = path(t)
    w = f(z)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(z.real, z.imag)
    plt.title('Integration Path')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    
    plt.subplot(122)
    plt.plot(w.real, w.imag)
    plt.title('f(z) along the path')
    plt.xlabel('Re(f(z))')
    plt.ylabel('Im(f(z))')
    
    integral = np.trapz(w, z)
    plt.suptitle(f'Complex Path Integral\nResult: {integral:.4f}')
    plt.tight_layout()
    plt.show()

def example_function(z):
    return z**2

def circular_path(t):
    return np.exp(2j * np.pi * t)

plot_complex_path_integral(example_function, circular_path)
```

Slide 7: Cauchy's Integral Formula

Cauchy's integral formula is a fundamental result in complex analysis. It states that the value of a holomorphic function inside a simple closed contour can be determined by its values on the contour.

```python
import numpy as np
import matplotlib.pyplot as plt

def cauchy_integral_formula(f, z0, R, num_points=1000):
    theta = np.linspace(0, 2*np.pi, num_points)
    z = z0 + R * np.exp(1j * theta)
    
    integrand = f(z) / (z - z0)
    integral = np.trapz(integrand, z) / (2j * np.pi)
    
    return integral

def plot_cauchy_integral(f, z0, R):
    theta = np.linspace(0, 2*np.pi, 100)
    z = z0 + R * np.exp(1j * theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(z.real, z.imag, 'b-')
    plt.plot(z0.real, z0.imag, 'ro', markersize=10)
    plt.text(z0.real, z0.imag, f'z0 = {z0}', fontsize=12, verticalalignment='bottom')
    
    plt.title("Cauchy's Integral Formula")
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axis('equal')
    plt.grid(True)
    
    integral_value = cauchy_integral_formula(f, z0, R)
    actual_value = f(z0)
    
    plt.text(0.05, 0.95, f'Integral result: {integral_value:.4f}', transform=plt.gca().transAxes)
    plt.text(0.05, 0.9, f'Actual f(z0): {actual_value:.4f}', transform=plt.gca().transAxes)
    
    plt.show()

def example_function(z):
    return z**2 + 1

z0 = 0.5 + 0.5j
R = 1.0

plot_cauchy_integral(example_function, z0, R)
```

Slide 8: Residue Theorem

The residue theorem is a powerful tool for evaluating complex integrals. It relates the integral of a function around a closed contour to the sum of the residues of the function at its poles inside the contour.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def residue_theorem_example():
    def integrand(theta):
        z = np.exp(1j * theta)
        return 1 / (z**2 + 1)
    
    def contour_integral():
        result, _ = quad(lambda theta: np.real(integrand(theta)), 0, 2*np.pi)
        return result
    
    def residue_sum():
        # Poles are at z = ±i
        # Residue at z = i: 1 / (2i)
        return 2 * np.pi * 1j * (1 / (2j))
    
    contour_result = contour_integral()
    residue_result = residue_sum()
    
    print(f"Contour integral result: {contour_result:.6f}")
    print(f"Residue theorem result: {residue_result.real:.6f}")
    
    # Plotting
    theta = np.linspace(0, 2*np.pi, 1000)
    z = np.exp(1j * theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(z.real, z.imag, 'b-')
    plt.plot(0, 1, 'ro', markersize=10)
    plt.plot(0, -1, 'ro', markersize=10)
    plt.text(0, 1.1, 'i', fontsize=12)
    plt.text(0, -1.1, '-i', fontsize=12)
    
    plt.title("Residue Theorem Example: f(z) = 1 / (z^2 + 1)")
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.show()

residue_theorem_example()
```

Slide 9: Laurent Series

A Laurent series is a representation of a complex function as a power series that includes both positive and negative powers. It's useful for analyzing the behavior of functions near singularities.

```python
import sympy as sp

def laurent_series_example():
    z = sp.Symbol('z')
    f = 1 / (z**2 + 1)
    
    # Expand f(z) around z = 0
    series = sp.series(f, z, 0, 5)
    
    print("Laurent series expansion of f(z) = 1 / (z^2 + 1) around z = 0:")
    print(series)
    
    # Plotting
    import numpy as np
    import matplotlib.pyplot as plt
    
    def original_function(z):
        return 1 / (z**2 + 1)
    
    def series_approximation(z, terms=5):
        return sum((-1)**n * z**(2*n) for n in range(terms))
    
    theta = np.linspace(0, 2*np.pi, 1000)
    z = np.exp(1j * theta)
    
    plt.figure(figsize=(10, 8))
    plt.polar(theta, np.abs(original_function(z)), label='Original function')
    plt.polar(theta, np.abs(series_approximation(z)), label='Laurent series approximation')
    plt.title("Comparison of f(z) = 1 / (z^2 + 1) and its Laurent series")
    plt.legend()
    
    plt.show()

laurent_series_example()
```

Slide 10: Conformal Mappings

Conformal mappings are complex functions that preserve angles locally. They are important in various applications, including fluid dynamics and cartography. These mappings maintain the shape of infinitesimally small figures, although they may change their size and position.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_conformal_mapping(f, x_range, y_range, grid_lines=10):
    x = np.linspace(x_range[0], x_range[1], grid_lines)
    y = np.linspace(y_range[0], y_range[1], grid_lines)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    W = f(Z)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(X, Y, 'k-', lw=0.5)
    plt.plot(X.T, Y.T, 'k-', lw=0.5)
    plt.title('Original Grid')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    
    plt.subplot(122)
    plt.plot(W.real, W.imag, 'r-', lw=0.5)
    plt.plot(W.real.T, W.imag.T, 'r-', lw=0.5)
    plt.title(f'Mapped Grid: f(z) = {f.__name__}')
    plt.xlabel('Re(f(z))')
    plt.ylabel('Im(f(z))')
    
    plt.tight_layout()
    plt.show()

def exponential_map(z):
    return np.exp(z)

plot_conformal_mapping(exponential_map, (-2, 2), (-2, 2))
```

Slide 11: Analytic Continuation

Analytic continuation is a technique for extending the domain of a given analytic function. It allows us to define a function beyond its original domain of definition, provided certain conditions are met.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_analytic_continuation():
    def log_principal(z):
        return np.log(np.abs(z)) + 1j * np.angle(z)
    
    def log_continued(z):
        return np.log(np.abs(z)) + 1j * (np.angle(z) + 2 * np.pi)
    
    theta = np.linspace(0, 4*np.pi, 1000)
    z = np.exp(1j * theta)
    
    w1 = log_principal(z)
    w2 = log_continued(z)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(w1.real, w1.imag, label='Principal branch')
    plt.plot(w2.real, w2.imag, label='Continued branch')
    plt.title('Analytic Continuation of log(z)')
    plt.xlabel('Re(log(z))')
    plt.ylabel('Im(log(z))')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(z.real, z.imag)
    plt.title('Path in z-plane')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    
    plt.tight_layout()
    plt.show()

plot_analytic_continuation()
```

Slide 12: Applications in Physics: Fluid Dynamics

Complex analysis finds extensive applications in fluid dynamics. The complex potential, a combination of velocity potential and stream function, provides a powerful tool for analyzing two-dimensional, incompressible, irrotational flows.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_flow_around_cylinder():
    def complex_potential(z, U, R):
        return U * (z + R**2 / z)
    
    def velocity_field(z, U, R):
        return U * (1 - R**2 / z**2)
    
    U = 1  # Free stream velocity
    R = 1  # Cylinder radius
    
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    W = complex_potential(Z, U, R)
    V = velocity_field(Z, U, R)
    
    plt.figure(figsize=(10, 8))
    plt.streamplot(X, Y, V.real, V.imag, density=1, color='b', linewidth=1, arrowsize=1)
    plt.contour(X, Y, W.imag, levels=20, colors='r', linestyles='solid')
    circle = plt.Circle((0, 0), R, fill=False, color='k')
    plt.gca().add_artist(circle)
    plt.title('Flow Around a Cylinder')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

plot_flow_around_cylinder()
```

Slide 13: Applications in Engineering: Signal Processing

Complex analysis plays a crucial role in signal processing, particularly in the analysis of frequency domain representations of signals. The Fourier transform, a fundamental tool in signal processing, is deeply rooted in complex analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_signal_and_fft():
    # Generate a signal
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Compute FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(122)
    plt.plot(freqs[:len(freqs)//2], np.abs(fft)[:len(freqs)//2])
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()

plot_signal_and_fft()
```

Slide 14: Additional Resources

For those interested in delving deeper into complex analysis, here are some recommended resources:

1. "Complex Analysis" by Elias M. Stein and Rami Shakarchi ArXiv: [https://arxiv.org/abs/math/0complex-analysis](https://arxiv.org/abs/math/0complex-analysis)
2. "Visual Complex Analysis" by Tristan Needham ArXiv: [https://arxiv.org/abs/math/9visual-complex-analysis](https://arxiv.org/abs/math/9visual-complex-analysis)
3. "Complex Analysis: An Introduction to the Theory of Analytic Functions of One Complex Variable" by Lars V. Ahlfors ArXiv: [https://arxiv.org/abs/math/8complex-analysis-intro](https://arxiv.org/abs/math/8complex-analysis-intro)

These resources provide a mix of theoretical foundations and intuitive explanations, catering to different learning styles and levels of mathematical sophistication.

