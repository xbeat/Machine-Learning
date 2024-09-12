## Numerical Methods Optimization and Mathematical Modeling in Python

Slide 1: Introduction to Numerical Methods

Numerical methods are techniques used to approximate solutions to mathematical problems using numerical approximations and iterative methods. These methods are essential when analytical solutions are impossible or impractical to obtain.

Code:

```python
import numpy as np

# Example: Approximating the value of pi using the Leibniz series
def leibniz_pi(n):
    pi = 0
    for i in range(n):
        pi += ((-1)**i) / (2*i + 1)
    return 4 * pi

# Approximating pi with 1000000 terms
approx_pi = leibniz_pi(1000000)
print(f"Approximation of pi: {approx_pi}")
```

Slide 2: Numerical Integration

Numerical integration is the process of approximating the value of a definite integral using numerical techniques, such as the Trapezoidal Rule or Simpson's Rule.

Code:

```python
import numpy as np

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    s = y[0] + y[-1]
    for i in range(1, n):
        s += 2 * y[i]
    return h * s / 2

# Example: Integrating x^2 from 0 to 1
def f(x):
    return x**2

result = trapezoidal(f, 0, 1, 100)
print(f"Integral of x^2 from 0 to 1: {result}")
```

Slide 3: Root-Finding Methods

Root-finding methods are numerical techniques used to find the roots (or zeros) of a given function. Common methods include the Bisection Method, Newton-Raphson Method, and Secant Method.

Code:

```python
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative is zero")
        x = x - fx / dfx
    raise ValueError("Maximum iterations reached")

# Example: Finding the root of x^3 - x - 1
f = lambda x: x**3 - x - 1
df = lambda x: 3*x**2 - 1

root = newton_raphson(f, df, 1.5)
print(f"Root of x^3 - x - 1: {root}")
```

Slide 4: Optimization Methods

Optimization methods are used to find the minimum or maximum of a given function, subject to constraints. Common methods include Gradient Descent, Newton's Method, and Sequential Quadratic Programming (SQP).

Code:

```python
import numpy as np

def gradient_descent(f, grad, x0, lr=0.01, tol=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        gradx = grad(x)
        x = x - lr * gradx
        if np.linalg.norm(gradx) < tol:
            return x
    raise ValueError("Maximum iterations reached")

# Example: Minimizing f(x, y) = x^2 + y^2
f = lambda x: x[0]**2 + x[1]**2
grad = lambda x: np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
min_point = gradient_descent(f, grad, x0)
print(f"Minimum point: {min_point}")
```

Slide 5: Interpolation Methods

Interpolation methods are used to construct a function that passes through a set of given data points. Common methods include Polynomial Interpolation (Lagrange and Newton forms) and Spline Interpolation.

Code:

```python
import numpy as np

def lagrange_interpolation(x, y, x_new):
    n = len(x)
    y_new = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_new - x[j]) / (x[i] - x[j])
        y_new += term
    return y_new

# Example: Interpolating f(x) = sin(x) at x = 0, pi/4, pi/2
x = [0, np.pi/4, np.pi/2]
y = [np.sin(xi) for xi in x]
x_new = np.pi/3

y_interp = lagrange_interpolation(x, y, x_new)
print(f"Interpolated value at x = {x_new}: {y_interp}")
```

Slide 6: Numerical Differentiation

Numerical differentiation is the process of approximating the derivative of a function using numerical techniques, such as finite differences or automatic differentiation.

Code:

```python
import numpy as np

def finite_difference(f, x, h=1e-6):
    return (f(x + h) - f(x)) / h

# Example: Approximating the derivative of sin(x) at x = pi/4
x = np.pi/4
h = 1e-6
dfdx = finite_difference(np.sin, x, h)

print(f"Approximate derivative of sin(x) at x = {x}: {dfdx}")
print(f"Exact derivative: {np.cos(x)}")
```

Slide 7: Numerical Solution of Ordinary Differential Equations (ODEs)

Numerical methods for solving ODEs involve approximating the solution at discrete points by using techniques such as the Euler method, Runge-Kutta methods, or finite difference methods.

Code:

```python
import numpy as np

def euler(f, x0, y0, xn, n):
    x = np.linspace(x0, xn, n+1)
    y = np.zeros(n+1)
    y[0] = y0
    h = (xn - x0) / n
    for i in range(n):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return x, y

# Example: Solving y' = y, y(0) = 1
def f(x, y):
    return y

x0, y0 = 0, 1
xn = 2
n = 10

x, y = euler(f, x0, y0, xn, n)
print(f"Solution at x = {xn}: {y[-1]}")
print(f"Exact solution: {np.exp(xn)}")
```

Slide 8: Finite Difference Methods

Finite difference methods are numerical techniques used to approximate solutions to partial differential equations (PDEs) by discretizing the problem domain and approximating the derivatives using finite differences.

Code:

```python
import numpy as np

def laplace_2d(n, m, f, boundary_conditions):
    u = np.zeros((n+2, m+2))
    u[1:-1, 1:-1] = f
    u = apply_boundary_conditions(u, boundary_conditions)

    iteration = 0
    max_iter = 1000
    tolerance = 1e-6
    while iteration < max_iter:
        u_old = u.copy()
        for i in range(1, n+1):
            for j in range(1, m+1):
                u[i, j] = (u_old[i-1, j] + u_old[i+1, j] + u_old[i, j-1] + u_old[i, j+1]) / 4
        diff = np.max(np.abs(u - u_old))
        if diff < tolerance:
            break
        iteration += 1

    return u[1:-1, 1:-1]

# Example: Solving Laplace's equation with Dirichlet boundary conditions
n, m = 10, 10
f = np.zeros((n, m))
boundary_conditions = lambda u: np.pad(u, 1, mode='constant', constant_values=1)

solution = laplace_2d(n, m, f, boundary_conditions)
print(solution)
```

Slide 9: Finite Element Methods

Finite element methods (FEM) are numerical techniques used to approximate solutions to partial differential equations (PDEs) by dividing the problem domain into small, finite elements and approximating the solution over each element.

Code:

```python
import numpy as np
from scipy.integrate import quad

def fem_1d_poisson(n, f, boundary_conditions):
    x = np.linspace(0, 1, n+1)
    h = 1 / n
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)

    for i in range(n-1):
        A[i, i] = 2 / h
        if i > 0:
            A[i, i-1] = -1 / h
        if i < n-2:
            A[i, i+1] = -1 / h
        b[i] = quad(f, x[i], x[i+1])[0]

    u = np.zeros(n+1)
    u[0], u[-1] = boundary_conditions
    u_interior = np.linalg.solve(A, b)
    u[1:-1] = u_interior

    return x, u

# Example: Solving -u''(x) = 1, u(0) = 0, u(1) = 0
f = lambda x: 1
boundary_conditions = (0, 0)
n = 10

x, u = fem_1d_poisson(n, f, boundary_conditions)
print(f"Solution: {u}")
```

Slide 10: Mathematical Modeling

Mathematical modeling is the process of creating a mathematical representation of a real-world system or phenomenon. This involves identifying the key variables, formulating equations or constraints, and using numerical methods to solve the resulting models.

Code:

```python
import numpy as np

# Example: Modeling population growth with logistic equation
def logistic_growth(r, K, N0, t):
    return K / (1 + (K/N0 - 1) * np.exp(-r*t))

# Parameters
r = 0.5  # Growth rate
K = 1000  # Carrying capacity
N0 = 100  # Initial population
t = np.linspace(0, 20, 100)  # Time range

# Solve the model
N = logistic_growth(r, K, N0, t)

# Plot the solution
import matplotlib.pyplot as plt
plt.plot(t, N)
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Logistic Growth Model')
plt.show()
```

Slide 11: Optimization Methods in Practice

Optimization methods are widely used in various fields, such as machine learning, engineering, finance, and operations research. This slide demonstrates the application of optimization methods in a practical scenario.

Code:

```python
import numpy as np
from scipy.optimize import minimize

# Example: Portfolio optimization
def portfolio_return(weights, returns):
    return np.sum(weights * returns)

def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_optimization(returns, cov_matrix, risk_tolerance):
    n = len(returns)
    weights = np.random.random(n)
    weights /= np.sum(weights)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    initial_risk = portfolio_risk(weights, cov_matrix)
    if initial_risk <= risk_tolerance:
        return weights

    def neg_return(weights):
        return -portfolio_return(weights, returns)

    res = minimize(neg_return, weights, method='SLSQP',
                   constraints=constraints, bounds=bounds)

    return res.x

# Example data
returns = np.array([0.08, 0.12, 0.05])
cov_matrix = np.array([[0.01, 0.005, 0.002],
                       [0.005, 0.04, 0.003],
                       [0.002, 0.003, 0.02]])
risk_tolerance = 0.1

optimal_weights = portfolio_optimization(returns, cov_matrix, risk_tolerance)
print(f"Optimal portfolio weights: {optimal_weights}")
```

Slide 12: Numerical Methods in Scientific Computing

Numerical methods play a crucial role in various scientific computing applications, such as computational fluid dynamics, computational chemistry, and astrophysics. This slide showcases the use of numerical methods in a scientific computing scenario.

Code:

```python
import numpy as np
from scipy.integrate import odeint

# Example: Modeling the motion of a pendulum
def pendulum(y, t, L, g):
    theta, omega = y
    dydt = [omega, -g/L * np.sin(theta)]
    return dydt

# Parameters
L = 1.0  # Length of the pendulum
g = 9.81  # Acceleration due to gravity
y0 = [np.pi/4, 0]  # Initial conditions: theta = pi/4, omega = 0

# Time range
t = np.linspace(0, 10, 1000)

# Solve the ODE
sol = odeint(pendulum, y0, t, args=(L, g))

# Plot the solution
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(t, sol[:, 0], label='theta')
plt.plot(t, sol[:, 1], label='omega')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Pendulum Motion')
plt.legend()
plt.show()
```

Slide 13: Additional Resources

For those interested in further exploring numerical methods, optimization, and mathematical modeling, here are some additional resources:

* Numerical Analysis by Richard L. Burden and J. Douglas Faires (Book)
* Numerical Optimization by Jorge Nocedal and Stephen J. Wright (Book)
* Mathematical Modeling by Mark M. Meerschaert (Book)
* Numerical Methods for Engineers by Steven C. Chapra (Book)

From ArXiv.org:

* "A Survey of Numerical Methods for High-Dimensional Optimization Problems" ([https://arxiv.org/abs/2003.06151](https://arxiv.org/abs/2003.06151))
* "Mathematical Modeling of Infectious Disease Dynamics" ([https://arxiv.org/abs/2004.08709](https://arxiv.org/abs/2004.08709))

