## Introduction to Nonlinear Analysis with Python
Slide 1: Introduction to Nonlinear Analysis

Nonlinear analysis is a branch of mathematics that deals with systems and equations that are not linear. It's crucial in various fields, including physics, engineering, and economics. This slideshow will introduce key concepts in nonlinear analysis, focusing on optima and equilibria, with Python examples to illustrate these ideas.

```python
import numpy as np
import matplotlib.pyplot as plt

def nonlinear_function(x):
    return np.sin(x) + 0.1 * x**2

x = np.linspace(-10, 10, 1000)
y = nonlinear_function(x)

plt.plot(x, y)
plt.title('Example of a Nonlinear Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 2: Local and Global Optima

In nonlinear analysis, we often seek to find the optimal points of a function. Local optima are the best solutions within a neighboring set of candidate solutions, while global optima are the best solutions among all possible solutions. Identifying these points is crucial in optimization problems.

```python
import numpy as np
from scipy.optimize import minimize_scalar

def objective_function(x):
    return (x - 2) * x * (x + 2)**2

result = minimize_scalar(objective_function)

print(f"Global minimum: x = {result.x:.4f}, f(x) = {result.fun:.4f}")

x = np.linspace(-3, 3, 1000)
y = objective_function(x)

plt.plot(x, y)
plt.plot(result.x, result.fun, 'ro', label='Global Minimum')
plt.title('Function with Multiple Local Optima')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Gradient Descent

Gradient descent is a first-order iterative optimization algorithm used to find a local minimum of a differentiable function. It takes steps proportional to the negative of the gradient of the function at the current point.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 4*x**2 + 2

def df(x):
    return 4*x**3 - 8*x

def gradient_descent(start, learn_rate, num_iterations):
    x = start
    x_history = [x]
    
    for _ in range(num_iterations):
        x = x - learn_rate * df(x)
        x_history.append(x)
    
    return x, x_history

x_min, x_history = gradient_descent(start=2, learn_rate=0.1, num_iterations=20)

x = np.linspace(-2, 2, 100)
plt.plot(x, f(x), 'b-', label='f(x)')
plt.plot(x_history, [f(x) for x in x_history], 'ro-', label='Gradient Descent')
plt.title('Gradient Descent Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Local minimum found at x = {x_min:.4f}")
```

Slide 4: Newton's Method

Newton's method is a root-finding algorithm that produces successively better approximations to the roots of a real-valued function. It can be used to find local maxima and minima of functions by finding the roots of its first derivative.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

def newton_method(x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = x - f(x) / df(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

root = newton_method(1)
print(f"Root found at x = {root:.6f}")

x = np.linspace(1, 2, 100)
plt.plot(x, f(x), label='f(x)')
plt.plot(x, np.zeros_like(x), 'k--')
plt.plot(root, f(root), 'ro', label='Root')
plt.title("Newton's Method for Root Finding")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: Fixed-Point Iteration

Fixed-point iteration is a method of computing fixed points of a function. It's used in various areas of nonlinear analysis, including solving nonlinear equations and finding equilibrium points in dynamical systems.

```python
import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return np.cos(x)

def fixed_point_iteration(g, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

fixed_point = fixed_point_iteration(g, 0)
print(f"Fixed point found at x = {fixed_point:.6f}")

x = np.linspace(0, np.pi, 100)
plt.plot(x, g(x), label='g(x)')
plt.plot(x, x, 'k--', label='y = x')
plt.plot(fixed_point, g(fixed_point), 'ro', label='Fixed Point')
plt.title('Fixed-Point Iteration')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Bifurcation Analysis

Bifurcation analysis studies how the qualitative behavior of a system changes as a parameter varies. It's crucial in understanding the stability and dynamics of nonlinear systems.

```python
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    return r * x * (1 - x)

def bifurcation_diagram(r_min, r_max, num_r, num_iterations, num_discard):
    r_range = np.linspace(r_min, r_max, num_r)
    x = np.ones(num_r) * 0.5
    
    results = []
    for r in r_range:
        for _ in range(num_discard):
            x = logistic_map(r, x)
        for _ in range(num_iterations):
            x = logistic_map(r, x)
            results.append((r, x))
    
    return np.array(results)

data = bifurcation_diagram(2.5, 4.0, 1000, 100, 100)

plt.figure(figsize=(12, 8))
plt.plot(data[:, 0], data[:, 1], ',k', alpha=0.1, markersize=0.1)
plt.title('Bifurcation Diagram of the Logistic Map')
plt.xlabel('r')
plt.ylabel('x')
plt.show()
```

Slide 7: Lyapunov Exponents

Lyapunov exponents measure the rate at which nearby trajectories in a dynamical system diverge or converge. They are crucial for understanding chaos and stability in nonlinear systems.

```python
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x):
    return r * x * (1 - x)

def lyapunov_exponent(r, num_iterations=1000, num_discard=100):
    x = 0.5
    lyap = 0
    
    for _ in range(num_discard):
        x = logistic_map(r, x)
    
    for _ in range(num_iterations):
        x = logistic_map(r, x)
        lyap += np.log(abs(r * (1 - 2*x)))
    
    return lyap / num_iterations

r_range = np.linspace(2.5, 4.0, 1000)
lyap_exponents = [lyapunov_exponent(r) for r in r_range]

plt.figure(figsize=(12, 6))
plt.plot(r_range, lyap_exponents)
plt.title('Lyapunov Exponent of the Logistic Map')
plt.xlabel('r')
plt.ylabel('Lyapunov Exponent')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)
plt.show()
```

Slide 8: Chaos and Strange Attractors

Chaos is a phenomenon where small changes in initial conditions lead to drastically different outcomes. Strange attractors are geometric shapes that characterize the long-term behavior of chaotic systems.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(xyz, s=10, r=28, b=2.667):
    x, y, z = xyz
    dx = s * (y - x)
    dy = r * x - y - x * z
    dz = x * y - b * z
    return np.array([dx, dy, dz])

def simulate_lorenz(num_steps=10000, dt=0.01):
    xyz = np.zeros((num_steps, 3))
    xyz[0] = np.random.rand(3)
    
    for i in range(1, num_steps):
        dxyz = lorenz_system(xyz[i-1])
        xyz[i] = xyz[i-1] + dxyz * dt
    
    return xyz

lorenz_data = simulate_lorenz()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2], lw=0.5)
ax.set_title('Lorenz Attractor')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

Slide 9: Optimization with Constraints

Constrained optimization is the process of optimizing an objective function with respect to some variables in the presence of constraints on those variables. This is a common problem in many fields, including economics and engineering.

```python
import numpy as np
from scipy.optimize import minimize

def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2

def constraint(x):
    return x[0] - 2*x[1] + 2

cons = {'type': 'ineq', 'fun': constraint}
bounds = ((0, None), (0, None))
x0 = [0, 0]

result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

print("Optimal solution:", result.x)
print("Optimal value:", result.fun)

import matplotlib.pyplot as plt

x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)
Z = (X - 1)**2 + (Y - 2.5)**2

plt.contour(X, Y, Z, levels=20)
plt.plot(x, x/2 - 1, 'r--', label='Constraint')
plt.plot(result.x[0], result.x[1], 'ro', label='Optimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Constrained Optimization')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 10: Variational Methods

Variational methods are techniques for finding extrema of functionals. They are widely used in physics, particularly in quantum mechanics and classical mechanics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def euler_lagrange(y, x, args):
    dy, y = y
    L, dLdy, dLddy = args
    d2y = (dLdy(x, y, dy) - L(x, y, dy)) / dLddy(x, y, dy)
    return [d2y, dy]

def L(x, y, dy):
    return 0.5 * dy**2 - 0.5 * y**2

def dLdy(x, y, dy):
    return -y

def dLddy(x, y, dy):
    return dy

x = np.linspace(0, 10, 100)
y0 = [0, 1]  # Initial conditions: y(0) = 0, y'(0) = 1

solution = odeint(euler_lagrange, y0, x, args=((L, dLdy, dLddy),))

plt.plot(x, solution[:, 1])
plt.title('Solution to the Euler-Lagrange Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 11: Dynamical Systems and Phase Portraits

Dynamical systems theory studies the long-term behavior of evolving systems. Phase portraits are graphical representations of trajectories of a dynamical system in the phase plane.

```python
import numpy as np
import matplotlib.pyplot as plt

def vector_field(X, Y):
    U = Y
    V = -np.sin(X) - 0.5*Y
    return U, V

x = np.linspace(-2*np.pi, 2*np.pi, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

U, V = vector_field(X, Y)

plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, U, V, density=1, linewidth=1, arrowsize=1, arrowstyle='->')
plt.title('Phase Portrait of a Nonlinear Pendulum')
plt.xlabel('θ')
plt.ylabel('dθ/dt')
plt.grid(True)
plt.show()
```

Slide 12: Numerical Integration of ODEs

Numerical methods for solving ordinary differential equations (ODEs) are essential tools in nonlinear analysis, especially when analytical solutions are not available.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def van_der_pol(y, t, mu):
    dy1 = y[1]
    dy2 = mu * (1 - y[0]**2) * y[1] - y[0]
    return [dy1, dy2]

mu = 1.0
y0 = [2, 0]
t = np.linspace(0, 20, 1000)

solution = odeint(van_der_pol, y0, t, args=(mu,))

plt.figure(figsize=(10, 6))
plt.plot(t, solution[:, 0], label='x(t)')
plt.plot(t, solution[:, 1], label='y(t)')
plt.title('Van der Pol Oscillator')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 13: Nonlinear Least Squares

Nonlinear least squares is a form of least squares analysis used to fit a set of m observations with a model that is nonlinear in n unknown parameters. This method is widely used in data fitting.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exponential_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate synthetic data
x_data = np.linspace(0, 10, 100)
y_true = exponential_model(x_data, 2.5, 0.5, 1.0)
y_noise = 0.2 * np.random.normal(size=x_data.size)
y_data = y_true + y_noise

# Fit the model
popt, _ = curve_fit(exponential_model, x_data, y_data)

# Plot the results
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, exponential_model(x_data, *popt), 'r-', label='Fitted Curve')
plt.legend()
plt.title('Nonlinear Least Squares Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f"Fitted parameters: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")
```

Slide 14: Stability Analysis

Stability analysis is crucial in nonlinear dynamics to understand how systems behave near equilibrium points. It helps predict whether small perturbations will grow or decay over time.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def system(X, t):
    x, y = X
    dx = y
    dy = -np.sin(x)
    return [dx, dy]

def plot_phase_portrait(ax, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 20)
    y = np.linspace(y_range[0], y_range[1], 20)
    X, Y = np.meshgrid(x, y)
    U = Y
    V = -np.sin(X)
    ax.streamplot(X, Y, U, V, density=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Phase Portrait')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Stable equilibrium point
plot_phase_portrait(ax1, [-np.pi, np.pi], [-2, 2])
ax1.plot(0, 0, 'ro', markersize=10)
ax1.set_title('Stable Equilibrium (0, 0)')

# Unstable equilibrium point
plot_phase_portrait(ax2, [0, 2*np.pi], [-2, 2])
ax2.plot(np.pi, 0, 'ro', markersize=10)
ax2.set_title('Unstable Equilibrium (π, 0)')

plt.tight_layout()
plt.show()
```

Slide 15: Continuation Methods

Continuation methods are numerical techniques for computing solution curves of parametrized nonlinear equations. They are particularly useful for studying how solutions change as parameters vary.

```python
import numpy as np
import matplotlib.pyplot as plt

def cubic(x, r):
    return x**3 - x + r

def continuation_method(r_start, r_end, num_points):
    r_values = np.linspace(r_start, r_end, num_points)
    x_values = []
    
    x = 0  # Initial guess
    for r in r_values:
        # Newton's method to find the root
        for _ in range(10):
            fx = cubic(x, r)
            if abs(fx) < 1e-6:
                break
            dfx = 3 * x**2 - 1
            x = x - fx / dfx
        x_values.append(x)
    
    return r_values, x_values

r_values, x_values = continuation_method(-1, 1, 200)

plt.figure(figsize=(10, 6))
plt.plot(r_values, x_values)
plt.title('Continuation Method for x³ - x + r = 0')
plt.xlabel('r')
plt.ylabel('x')
plt.grid(True)
plt.show()
```

Slide 16: Additional Resources

For those interested in diving deeper into nonlinear analysis, here are some valuable resources:

1. "Nonlinear Dynamics and Chaos" by Steven H. Strogatz ArXiv: [https://arxiv.org/abs/chao-dyn/9506001](https://arxiv.org/abs/chao-dyn/9506001)
2. "An Introduction to Dynamical Systems" by D. K. Arrowsmith and C. M. Place
3. "Numerical Methods for Scientists and Engineers" by R. W. Hamming
4. "Nonlinear Analysis" by Zdzisław Denkowski et al. ArXiv: [https://arxiv.org/abs/math/0111213](https://arxiv.org/abs/math/0111213)
5. "Applied Nonlinear Control" by Jean-Jacques E. Slotine and Weiping Li

These resources provide a mix of theoretical foundations and practical applications in nonlinear analysis, suitable for beginners and intermediate learners alike.

