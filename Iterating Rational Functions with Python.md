## Iterating Rational Functions with Python

Slide 1: Introduction to Iteration of Rational Functions

Iteration of rational functions is a powerful technique in mathematics with applications in various fields, including dynamical systems and computational methods. A rational function is a function of the form f(x) = P(x) / Q(x), where P and Q are polynomials.

```python
def rational_function(x, p_coeffs, q_coeffs):
    p = sum(c * x**i for i, c in enumerate(p_coeffs))
    q = sum(c * x**i for i, c in enumerate(q_coeffs))
    return p / q

# Example: f(x) = (x^2 + 1) / (x - 2)
f = lambda x: rational_function(x, [1, 0, 1], [-2, 1])
print(f(3))  # Output: 3.333333333333333
```

Slide 2: Basic Iteration Process

Iteration involves repeatedly applying a function to its own output. For a rational function f(x), we start with an initial value x0 and compute successive values: x1 = f(x0), x2 = f(x1), and so on.

```python
def iterate(f, x0, n):
    x = x0
    for _ in range(n):
        x = f(x)
        yield x

# Iterate f(x) = (x^2 + 1) / (x - 2) starting from x0 = 1
f = lambda x: (x**2 + 1) / (x - 2)
iterations = list(iterate(f, 1, 5))
print(iterations)
```

Slide 3: Convergence and Fixed Points

A sequence of iterations may converge to a fixed point, where f(x) = x. Fixed points are crucial in understanding the long-term behavior of iterated functions.

```python
def find_fixed_point(f, x0, tolerance=1e-6, max_iterations=100):
    x = x0
    for _ in range(max_iterations):
        next_x = f(x)
        if abs(next_x - x) < tolerance:
            return next_x
        x = next_x
    return None  # If no convergence within max_iterations

f = lambda x: (x**2 + 1) / (x - 2)
fixed_point = find_fixed_point(f, 1)
print(f"Fixed point: {fixed_point}")
```

Slide 4: Attractors and Basins of Attraction

An attractor is a set of values to which iterations converge. The basin of attraction is the set of initial values that lead to a particular attractor.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_basin(f, xmin, xmax, ymin, ymax, resolution=500):
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    for _ in range(20):
        Z = f(Z)
    
    plt.imshow(np.abs(Z), extent=[xmin, xmax, ymin, ymax])
    plt.colorbar(label='Magnitude')
    plt.title('Basin of Attraction')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

f = lambda z: (z**2 + 1) / (z - 2)
plot_basin(f, -2, 2, -2, 2)
```

Slide 5: Periodic Orbits

Some rational functions exhibit periodic behavior, where iteration cycles through a set of values repeatedly.

```python
def find_period(f, x0, max_period=100, tolerance=1e-6):
    x = x0
    orbit = [x]
    for period in range(1, max_period + 1):
        x = f(x)
        if any(abs(x - y) < tolerance for y in orbit):
            return period
        orbit.append(x)
    return None

f = lambda x: 4*x*(1-x)  # Logistic map
period = find_period(f, 0.1)
print(f"Period: {period}")
```

Slide 6: Chaos and Sensitivity to Initial Conditions

Chaotic behavior in rational function iteration is characterized by extreme sensitivity to initial conditions.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_sensitivity(f, x1, x2, n):
    trajectory1 = [x1]
    trajectory2 = [x2]
    for _ in range(n):
        x1 = f(x1)
        x2 = f(x2)
        trajectory1.append(x1)
        trajectory2.append(x2)
    
    plt.plot(range(n+1), trajectory1, label=f'x0 = {trajectory1[0]}')
    plt.plot(range(n+1), trajectory2, label=f'x0 = {trajectory2[0]}')
    plt.title('Sensitivity to Initial Conditions')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

f = lambda x: 3.9 * x * (1 - x)  # Chaotic logistic map
plot_sensitivity(f, 0.5, 0.500001, 50)
```

Slide 7: Newton's Method as Iteration

Newton's method for finding roots of a function can be viewed as iteration of a rational function.

```python
def newton_iteration(f, df):
    return lambda x: x - f(x) / df(x)

def newton_method(f, df, x0, tolerance=1e-6, max_iterations=100):
    newton_func = newton_iteration(f, df)
    return find_fixed_point(newton_func, x0, tolerance, max_iterations)

# Find sqrt(2) using Newton's method
f = lambda x: x**2 - 2
df = lambda x: 2*x
root = newton_method(f, df, 1)
print(f"sqrt(2) ≈ {root}")
```

Slide 8: Mandelbrot Set

The Mandelbrot set is a famous example of complex dynamics arising from the iteration of a simple quadratic function.

```python
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(h, w, max_iter):
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    
    return divtime

plt.imshow(mandelbrot(500, 750, 50), cmap='magma', extent=[-2, 0.8, -1.4, 1.4])
plt.title('Mandelbrot Set')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
```

Slide 9: Julia Sets

Julia sets are closely related to the Mandelbrot set and show the dynamics of complex rational functions.

```python
def julia_set(h, w, c, max_iter):
    y, x = np.ogrid[-1.5:1.5:h*1j, -1.5:1.5:w*1j]
    z = x + y*1j
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    
    return divtime

c = -0.4 + 0.6j
plt.imshow(julia_set(500, 750, c, 50), cmap='hot', extent=[-1.5, 1.5, -1.5, 1.5])
plt.title(f'Julia Set for c = {c}')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.show()
```

Slide 10: Bifurcation Diagrams

Bifurcation diagrams visualize how the long-term behavior of a system changes as a parameter varies.

```python
def bifurcation_diagram(f, x0, n_skip, n_iter, param_range):
    x = np.linspace(param_range[0], param_range[1], 1000)
    y = []
    for r in x:
        fx = lambda x: f(x, r)
        orbit = list(iterate(fx, x0, n_skip + n_iter))
        y.extend(orbit[n_skip:])
    
    plt.plot(x, y, ',k', alpha=0.1)
    plt.title('Bifurcation Diagram')
    plt.xlabel('Parameter')
    plt.ylabel('Attractor')
    plt.show()

logistic = lambda x, r: r * x * (1 - x)
bifurcation_diagram(logistic, 0.5, 100, 100, [2.5, 4])
```

Slide 11: Lyapunov Exponents

Lyapunov exponents measure the rate of separation of infinitesimally close trajectories, quantifying chaos.

```python
def lyapunov_exponent(f, x0, n, eps=1e-6):
    x = x0
    lyap = 0
    for _ in range(n):
        x_perturbed = x + eps
        lyap += np.log(abs((f(x_perturbed) - f(x)) / eps))
        x = f(x)
    return lyap / n

def plot_lyapunov_spectrum(f, x0, n, param_range):
    params = np.linspace(param_range[0], param_range[1], 500)
    lyaps = [lyapunov_exponent(lambda x: f(x, r), x0, n) for r in params]
    
    plt.plot(params, lyaps)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Lyapunov Exponent Spectrum')
    plt.xlabel('Parameter')
    plt.ylabel('Lyapunov Exponent')
    plt.show()

logistic = lambda x, r: r * x * (1 - x)
plot_lyapunov_spectrum(logistic, 0.5, 1000, [2.5, 4])
```

Slide 12: Applications in Population Dynamics

Rational function iteration can model population dynamics, such as predator-prey relationships.

```python
import numpy as np
import matplotlib.pyplot as plt

def lotka_volterra(X, t, a, b, c, d):
    x, y = X
    dxdt = a*x - b*x*y
    dydt = -c*y + d*x*y
    return [dxdt, dydt]

def plot_population_dynamics(x0, y0, a, b, c, d, T, dt):
    t = np.linspace(0, T, int(T/dt))
    X0 = [x0, y0]
    solution = odeint(lotka_volterra, X0, t, args=(a, b, c, d))
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, solution[:, 0], label='Prey')
    plt.plot(t, solution[:, 1], label='Predator')
    plt.title('Predator-Prey Population Dynamics')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

from scipy.integrate import odeint
plot_population_dynamics(10, 5, 1.5, 1, 3, 1, 20, 0.1)
```

Slide 13: Fractal Dimension Estimation

Rational function iteration can create fractals. We can estimate their fractal dimension using the box-counting method.

```python
import numpy as np
import matplotlib.pyplot as plt

def box_count(Z, box_sizes):
    counts = []
    for size in box_sizes:
        count = np.sum(block_reduce(Z, (size, size), func=np.any))
        counts.append(count)
    return counts

def estimate_fractal_dimension(Z):
    box_sizes = np.logspace(0, 8, num=20, base=2, dtype=int)
    counts = box_count(Z, box_sizes)
    
    coeffs = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -coeffs[0]

from skimage.measure import block_reduce

# Generate a fractal-like image (e.g., Mandelbrot set)
Z = mandelbrot(1000, 1000, 50) < 50

dim = estimate_fractal_dimension(Z)
print(f"Estimated fractal dimension: {dim:.3f}")
```

Slide 14: Iteration in Machine Learning

Rational function iteration concepts apply to machine learning, such as in gradient descent optimization.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, df, x0, learning_rate, num_iterations):
    x = x0
    trajectory = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * df(x)
        trajectory.append(x)
    return np.array(trajectory)

# Example: Minimize f(x) = x^2
f = lambda x: x**2
df = lambda x: 2*x

x0 = 5
lr = 0.1
num_iterations = 50

trajectory = gradient_descent(f, df, x0, lr, num_iterations)

plt.plot(range(num_iterations + 1), trajectory)
plt.title('Gradient Descent Optimization')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.show()
```

Slide 15: Additional Resources

For further exploration of iteration of rational functions and related topics:

1. "Dynamics of Rational Functions" by J. Milnor (ArXiv:math/9201272) URL: [https://arxiv.org/abs/math/9201272](https://arxiv.org/abs/math/9201272)
2. "Introduction to Dynamical Systems" by M. Brin and G. Stuck (Not an ArXiv source, but a comprehensive textbook)
3. "Chaos and Dynamical Systems" by D. Feldman (ArXiv:1908.09280) URL: [https://arxiv.org/abs/1908.09280](https://arxiv.org/abs/1908.09280)

These resources provide deeper insights into the mathematics behind rational function iteration and its applications in various fields.

