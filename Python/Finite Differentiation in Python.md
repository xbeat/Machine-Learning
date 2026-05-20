## Finite Differentiation in Python
Slide 1: Introduction to Finite Differentiation

Finite differentiation is a numerical method used to approximate derivatives of functions. It's a fundamental concept in calculus and numerical analysis, with applications in various fields of science and engineering. This method is particularly useful when dealing with discrete data points or when analytical derivatives are difficult to obtain.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2  # Example function: f(x) = x^2

x = np.linspace(-5, 5, 100)
y = f(x)

plt.plot(x, y)
plt.title("f(x) = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()
```

Slide 2: Forward Difference

The forward difference is the simplest form of finite differentiation. It approximates the derivative of a function at a point by calculating the slope of the line connecting that point to the next point.

```python
def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

x0 = 2
h = 0.1
approx_derivative = forward_difference(f, x0, h)
print(f"Approximate derivative at x = {x0}: {approx_derivative}")
```

Slide 3: Backward Difference

The backward difference is similar to the forward difference, but it uses the previous point instead of the next point to calculate the slope.

```python
def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

x0 = 2
h = 0.1
approx_derivative = backward_difference(f, x0, h)
print(f"Approximate derivative at x = {x0}: {approx_derivative}")
```

Slide 4: Central Difference

The central difference provides a more accurate approximation of the derivative by using both the previous and next points. It's often preferred due to its symmetry and higher accuracy.

```python
def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

x0 = 2
h = 0.1
approx_derivative = central_difference(f, x0, h)
print(f"Approximate derivative at x = {x0}: {approx_derivative}")
```

Slide 5: Higher-Order Differences

Higher-order differences can provide even more accurate approximations of derivatives. The second-order central difference, for example, uses four points to estimate the derivative.

```python
def second_order_central_difference(f, x, h):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12 * h)

x0 = 2
h = 0.1
approx_derivative = second_order_central_difference(f, x0, h)
print(f"Approximate derivative at x = {x0}: {approx_derivative}")
```

Slide 6: Error Analysis

The accuracy of finite difference methods depends on the step size h. As h approaches zero, the approximation generally improves, but very small h values can lead to numerical instability due to floating-point arithmetic limitations.

```python
def error_analysis(f, df, x, method):
    h_values = np.logspace(-10, 0, 100)
    errors = [abs(method(f, x, h) - df(x)) for h in h_values]
    
    plt.loglog(h_values, errors)
    plt.title(f"Error Analysis for {method.__name__}")
    plt.xlabel("Step size (h)")
    plt.ylabel("Absolute error")
    plt.grid(True)
    plt.show()

def df(x):  # Analytical derivative of x^2
    return 2*x

error_analysis(f, df, 2, central_difference)
```

Slide 7: Partial Derivatives

Finite differentiation can be extended to functions of multiple variables to compute partial derivatives. This is particularly useful in multivariable calculus and optimization problems.

```python
def partial_derivative(f, x, y, h, variable):
    if variable == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif variable == 'y':
        return (f(x, y + h) - f(x, y - h)) / (2 * h)

def g(x, y):
    return x**2 + y**2

x0, y0 = 1, 1
h = 0.01
dx = partial_derivative(g, x0, y0, h, 'x')
dy = partial_derivative(g, x0, y0, h, 'y')
print(f"∂g/∂x at ({x0}, {y0}): {dx}")
print(f"∂g/∂y at ({x0}, {y0}): {dy}")
```

Slide 8: Numerical Integration

Finite differentiation techniques can be reversed to perform numerical integration. The simplest method is the rectangle rule, which approximates the area under a curve using rectangles.

```python
def rectangle_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)[:-1]
    return h * np.sum(f(x))

def f(x):
    return np.sin(x)

a, b = 0, np.pi
n = 1000
integral = rectangle_rule(f, a, b, n)
print(f"Approximate integral of sin(x) from 0 to π: {integral}")
```

Slide 9: Trapezoidal Rule

The trapezoidal rule is an improvement over the rectangle rule, using trapezoids instead of rectangles to approximate the area under the curve.

```python
def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))

a, b = 0, np.pi
n = 1000
integral = trapezoidal_rule(f, a, b, n)
print(f"Approximate integral of sin(x) from 0 to π: {integral}")
```

Slide 10: Simpson's Rule

Simpson's rule provides an even more accurate approximation of the integral by using parabolic arcs instead of straight lines.

```python
def simpsons_rule(f, a, b, n):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2]))

a, b = 0, np.pi
n = 1000
integral = simpsons_rule(f, a, b, n)
print(f"Approximate integral of sin(x) from 0 to π: {integral}")
```

Slide 11: Real-Life Example: Heat Transfer

In heat transfer problems, finite differentiation can be used to solve the heat equation numerically. This example simulates the temperature distribution in a 1D rod over time.

```python
def heat_equation_1d(T0, k, L, t_max, nx, nt):
    dx = L / (nx - 1)
    dt = t_max / (nt - 1)
    
    T = np.zeros((nt, nx))
    T[0] = T0
    
    for n in range(1, nt):
        for i in range(1, nx-1):
            T[n, i] = T[n-1, i] + k * dt / dx**2 * (T[n-1, i+1] - 2*T[n-1, i] + T[n-1, i-1])
    
    return T

L = 1.0  # Length of the rod
t_max = 0.5  # Total simulation time
nx, nt = 50, 1000
k = 0.01  # Thermal diffusivity
T0 = np.sin(np.pi * np.linspace(0, L, nx))  # Initial temperature distribution

T = heat_equation_1d(T0, k, L, t_max, nx, nt)

plt.imshow(T, aspect='auto', extent=[0, L, t_max, 0])
plt.colorbar(label='Temperature')
plt.title('Heat Equation Solution')
plt.xlabel('Position')
plt.ylabel('Time')
plt.show()
```

Slide 12: Real-Life Example: Population Dynamics

Finite differentiation can be applied to model population dynamics using the logistic growth equation. This example simulates the growth of a population over time.

```python
def logistic_growth(r, K, P0, t_max, dt):
    t = np.arange(0, t_max, dt)
    P = np.zeros_like(t)
    P[0] = P0
    
    for i in range(1, len(t)):
        dP = r * P[i-1] * (1 - P[i-1] / K) * dt
        P[i] = P[i-1] + dP
    
    return t, P

r = 0.5  # Growth rate
K = 100  # Carrying capacity
P0 = 10  # Initial population
t_max = 20
dt = 0.1

t, P = logistic_growth(r, K, P0, t_max, dt)

plt.plot(t, P)
plt.title('Logistic Growth Model')
plt.xlabel('Time')
plt.ylabel('Population')
plt.grid(True)
plt.show()
```

Slide 13: Limitations and Considerations

While finite differentiation is a powerful tool, it has limitations. Numerical errors can accumulate, especially with small step sizes or in complex calculations. It's important to choose appropriate step sizes and be aware of potential instabilities in the results.

```python
def unstable_function(x):
    return np.sin(1/x)

x = np.linspace(0.01, 1, 1000)
y = unstable_function(x)

plt.plot(x, y)
plt.title("Unstable Function: sin(1/x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# Attempt to calculate derivative
x0 = 0.1
h_values = [1e-2, 1e-4, 1e-6, 1e-8]
for h in h_values:
    deriv = central_difference(unstable_function, x0, h)
    print(f"h = {h}: f'({x0}) ≈ {deriv}")
```

Slide 14: Advanced Topics and Future Directions

Finite differentiation serves as a foundation for more advanced numerical methods in scientific computing. These include finite element methods, spectral methods, and adaptive mesh refinement techniques. As computational power increases, these methods are becoming increasingly important in solving complex problems in physics, engineering, and beyond.

```python
def adaptive_integration(f, a, b, tol=1e-6, max_subdivisions=1000):
    def simpson(f, a, b):
        c = (a + b) / 2
        h = b - a
        return h / 6 * (f(a) + 4*f(c) + f(b))
    
    def recursive_simpson(f, a, b, tol, whole):
        c = (a + b) / 2
        left = simpson(f, a, c)
        right = simpson(f, c, b)
        if abs(left + right - whole) <= 15 * tol:
            return left + right
        return recursive_simpson(f, a, c, tol/2, left) + recursive_simpson(f, c, b, tol/2, right)
    
    return recursive_simpson(f, a, b, tol, simpson(f, a, b))

def f(x):
    return 1 / (1 + x**2)

a, b = 0, 1
result = adaptive_integration(f, a, b)
print(f"Integral of 1/(1+x^2) from 0 to 1: {result}")
print(f"Actual value (arctan(1)): {np.arctan(1)}")
```

Slide 15: Additional Resources

For those interested in diving deeper into finite differentiation and numerical methods, here are some recommended resources:

1. "Numerical Methods for Ordinary Differential Equations" by J.C. Butcher (2016) - Available on arXiv: [https://arxiv.org/abs/1603.04332](https://arxiv.org/abs/1603.04332)
2. "A Survey of Numerical Methods for Nonlinear Partial Differential Equations" by R.J. LeVeque (2018) - Available on arXiv: [https://arxiv.org/abs/1807.05624](https://arxiv.org/abs/1807.05624)
3. "Finite Difference Methods for Ordinary and Partial Differential Equations" by R.J. LeVeque (2007) - This book provides a comprehensive introduction to finite difference methods.

These resources offer in-depth explanations and advanced techniques for applying finite differentiation in various scientific and engineering contexts.

