## Visualizing Partial Differential Equations with Python
Slide 1: Introduction to Partial Differential Equations (PDEs)

Partial Differential Equations (PDEs) are equations involving functions of multiple variables and their partial derivatives. They are essential in modeling various physical phenomena.

```python
import sympy as sp

# Define variables and function
x, y, t = sp.symbols('x y t')
u = sp.Function('u')(x, y, t)

# Example of a PDE: Heat equation
heat_equation = sp.Eq(sp.diff(u, t), sp.diff(u, x, x) + sp.diff(u, y, y))
print(f"Heat Equation: {heat_equation}")
```

Slide 2: Types of PDEs

PDEs are classified based on their order and linearity. Common types include:

1. First-order PDEs
2. Second-order PDEs (e.g., elliptic, parabolic, hyperbolic)

```python
# First-order PDE example: Transport equation
transport_equation = sp.Eq(sp.diff(u, t) + sp.diff(u, x), 0)
print(f"Transport Equation: {transport_equation}")

# Second-order PDE example: Wave equation
wave_equation = sp.Eq(sp.diff(u, t, t), sp.diff(u, x, x))
print(f"Wave Equation: {wave_equation}")
```

Slide 3: Elliptic PDEs

Elliptic PDEs describe steady-state phenomena. A classic example is Laplace's equation.

```python
import numpy as np
import matplotlib.pyplot as plt

def laplace_equation(nx, ny, iterations):
    u = np.zeros((ny, nx))
    u[0, :] = 100  # Top boundary condition
    
    for _ in range(iterations):
        u[1:-1, 1:-1] = 0.25 * (u[1:-1, 2:] + u[1:-1, :-2] + u[2:, 1:-1] + u[:-2, 1:-1])
    
    return u

nx, ny = 50, 50
u = laplace_equation(nx, ny, 1000)
plt.imshow(u, cmap='hot')
plt.colorbar()
plt.title("Solution to Laplace's Equation")
plt.show()
```

Slide 4: Parabolic PDEs

Parabolic PDEs model diffusion processes. The heat equation is a prime example.

```python
def heat_equation_1d(nx, nt, alpha):
    dx = 1 / (nx - 1)
    dt = 0.5 * dx**2 / alpha
    u = np.zeros(nx)
    u[0] = 100  # Left boundary condition
    
    for _ in range(nt):
        u[1:-1] += alpha * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    
    return u

nx, nt = 50, 1000
alpha = 0.01
u = heat_equation_1d(nx, nt, alpha)
plt.plot(np.linspace(0, 1, nx), u)
plt.title("1D Heat Equation Solution")
plt.xlabel("Position")
plt.ylabel("Temperature")
plt.show()
```

Slide 5: Hyperbolic PDEs

Hyperbolic PDEs describe wave-like phenomena. The wave equation is a classic example.

```python
def wave_equation_1d(nx, nt, c):
    dx = 1 / (nx - 1)
    dt = 0.5 * dx / c
    u = np.zeros((3, nx))
    u[0, nx//2] = 1  # Initial pulse
    
    for n in range(1, nt):
        u[2, 1:-1] = 2*u[1, 1:-1] - u[0, 1:-1] + c**2 * dt**2 / dx**2 * (u[1, 2:] - 2*u[1, 1:-1] + u[1, :-2])
        u[0], u[1] = u[1], u[2]
    
    return u[1]

nx, nt = 100, 200
c = 1
u = wave_equation_1d(nx, nt, c)
plt.plot(np.linspace(0, 1, nx), u)
plt.title("1D Wave Equation Solution")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.show()
```

Slide 6: Finite Difference Method

The finite difference method approximates derivatives using discrete differences.

```python
def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return np.sin(x)

x = np.pi / 4
h = 0.001
approx_derivative = central_difference(f, x, h)
exact_derivative = np.cos(x)

print(f"Approximate derivative: {approx_derivative}")
print(f"Exact derivative: {exact_derivative}")
print(f"Error: {abs(approx_derivative - exact_derivative)}")
```

Slide 7: Finite Element Method

The finite element method divides the domain into smaller elements and approximates the solution using basis functions.

```python
from scipy.integrate import quad

def basis_function(x, xi, h):
    if xi - h <= x <= xi:
        return (x - (xi - h)) / h
    elif xi < x <= xi + h:
        return ((xi + h) - x) / h
    else:
        return 0

def assemble_system(n, f):
    h = 1 / n
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)
    
    for i in range(n-1):
        for j in range(n-1):
            xi, xj = (i+1)*h, (j+1)*h
            A[i, j] = quad(lambda x: basis_function(x, xi, h) * basis_function(x, xj, h), 0, 1)[0]
        b[i] = quad(lambda x: f(x) * basis_function(x, (i+1)*h, h), 0, 1)[0]
    
    return A, b

# Example usage
n = 10
f = lambda x: np.sin(np.pi * x)
A, b = assemble_system(n, f)
u = np.linalg.solve(A, b)

x = np.linspace(0, 1, 100)
plt.plot(x, f(x), label='Exact')
plt.plot(np.linspace(h, 1-h, n-1), u, 'o-', label='FEM')
plt.legend()
plt.title("Finite Element Method Solution")
plt.show()
```

Slide 8: Method of Characteristics

The method of characteristics solves first-order PDEs by finding characteristic curves.

```python
def method_of_characteristics(u0, a, T, nx, nt):
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, nx))
    
    u[0, :] = u0(x)
    for i in range(1, nt):
        u[i, :] = u[0, (x - a * t[i]) % 1]
    
    return u, x, t

def initial_condition(x):
    return np.sin(2 * np.pi * x)

a = 1
T = 1
nx, nt = 100, 50
u, x, t = method_of_characteristics(initial_condition, a, T, nx, nt)

plt.imshow(u, extent=[0, 1, 0, T], aspect='auto', origin='lower')
plt.colorbar()
plt.title("Solution using Method of Characteristics")
plt.xlabel("x")
plt.ylabel("t")
plt.show()
```

Slide 9: Spectral Methods

Spectral methods use Fourier series or other orthogonal functions to approximate solutions.

```python
def spectral_method(u0, N, T, nt):
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    t = np.linspace(0, T, nt)
    u = np.zeros((nt, N))
    
    u[0] = u0(x)
    u_hat = np.fft.fft(u[0])
    k = np.fft.fftfreq(N, 2*np.pi/N)
    
    for i in range(1, nt):
        u_hat *= np.exp(-1j * k**2 * t[i])
        u[i] = np.real(np.fft.ifft(u_hat))
    
    return u, x, t

def initial_condition(x):
    return np.exp(-(x - np.pi)**2)

N = 128
T = 0.1
nt = 50
u, x, t = spectral_method(initial_condition, N, T, nt)

plt.imshow(u, extent=[0, 2*np.pi, 0, T], aspect='auto', origin='lower')
plt.colorbar()
plt.title("Solution using Spectral Method")
plt.xlabel("x")
plt.ylabel("t")
plt.show()
```

Slide 10: Numerical Stability and Convergence

Numerical stability and convergence are crucial for accurate PDE solutions.

```python
def von_neumann_stability(scheme, dx, dt, k):
    return abs(scheme(k * dx, dt))

def ftcs_scheme(kdx, dt):
    return 1 - 2 * np.sin(kdx/2)**2

dx = 0.1
dt = 0.001
k_values = np.linspace(0, np.pi/dx, 100)
stability = [von_neumann_stability(ftcs_scheme, dx, dt, k) for k in k_values]

plt.plot(k_values, stability)
plt.axhline(y=1, color='r', linestyle='--')
plt.title("Von Neumann Stability Analysis")
plt.xlabel("k*dx")
plt.ylabel("|G(k)|")
plt.show()
```

Slide 11: Boundary Value Problems

Boundary value problems involve solving PDEs with specified boundary conditions.

```python
def solve_bvp(a, b, c, f, ya, yb, N):
    h = (b - a) / (N - 1)
    x = np.linspace(a, b, N)
    
    A = np.zeros((N, N))
    B = np.zeros(N)
    
    A[0, 0] = 1
    B[0] = ya
    A[-1, -1] = 1
    B[-1] = yb
    
    for i in range(1, N-1):
        A[i, i-1] = 1 / h**2
        A[i, i] = -2 / h**2 + c(x[i])
        A[i, i+1] = 1 / h**2
        B[i] = f(x[i])
    
    y = np.linalg.solve(A, B)
    return x, y

a, b = 0, 1
c = lambda x: 0
f = lambda x: np.exp(-x)
ya, yb = 0, 1
N = 100

x, y = solve_bvp(a, b, c, f, ya, yb, N)
plt.plot(x, y)
plt.title("Solution to Boundary Value Problem")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 12: Initial Value Problems

Initial value problems involve solving PDEs with specified initial conditions.

```python
def solve_ivp(f, y0, t_span, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(1, len(t)):
        k1 = dt * f(t[i-1], y[i-1])
        k2 = dt * f(t[i-1] + dt/2, y[i-1] + k1/2)
        k3 = dt * f(t[i-1] + dt/2, y[i-1] + k2/2)
        k4 = dt * f(t[i-1] + dt, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, y

def f(t, y):
    return np.array([y[1], -np.sin(y[0])])

y0 = [np.pi/2, 0]
t_span = [0, 10]
dt = 0.01

t, y = solve_ivp(f, y0, t_span, dt)
plt.plot(t, y[:, 0])
plt.title("Solution to Initial Value Problem")
plt.xlabel("t")
plt.ylabel("y")
plt.show()
```

Slide 13: Real-life Example: Heat Conduction in a Rod

Let's model heat conduction in a metal rod using the heat equation.

```python
def heat_conduction_rod(L, T, nx, nt, alpha, T_left, T_right):
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    
    u = np.zeros((nt, nx))
    u[0, :] = 0  # Initial temperature
    u[:, 0] = T_left  # Left boundary condition
    u[:, -1] = T_right  # Right boundary condition
    
    for j in range(1, nt):
        for i in range(1, nx-1):
            u[j, i] = u[j-1, i] + alpha * dt / dx**2 * (u[j-1, i+1] - 2*u[j-1, i] + u[j-1, i-1])
    
    return u, x, t

L = 1  # Length of the rod
T = 0.5  # Total time
nx, nt = 50, 1000
alpha = 0.01  # Thermal diffusivity
T_left, T_right = 100, 0  # Boundary temperatures

u, x, t = heat_conduction_rod(L, T, nx, nt, alpha, T_left, T_right)

plt.imshow(u, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='hot')
plt.colorbar(label='Temperature')
plt.title("Heat Conduction in a Rod")
plt.xlabel("Position (m)")
plt.ylabel("Time (s)")
plt.show()
```

Slide 14: Real-life Example: Wave Propagation in a String

Let's model wave propagation in a vibrating string using the wave equation.

```python
def wave_propagation_string(L, T, nx, nt, c):
    dx = L / (nx - 1)
    dt = T / (nt - 1)
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    
    u = np.zeros((nt, nx))
    u[0, :] = np.sin(np.pi * x / L)  # Initial displacement
    u[1, :] = u[0, :]  # Initial velocity is zero
    
    r = c * dt / dx
    for j in range(1, nt-1):
        for i in range(1, nx-1):
            u[j+1, i] = 2 * u[j, i] - u[j-1, i] + r**2 * (u[j, i+1] - 2*u[j, i] + u[j, i-1])
    
    return u, x, t

L = 1  # Length of the string
T = 2  # Total time
nx, nt = 100, 200
c = 1  # Wave speed

u, x, t = wave_propagation_string(L, T, nx, nt, c)

plt.imshow(u, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Displacement')
plt.title("Wave Propagation in a String")
plt.xlabel("Position (m)")
plt.ylabel("Time (s)")
plt.show()
```

Slide 15: Numerical Methods for Nonlinear PDEs

Nonlinear PDEs often require specialized numerical methods. Let's look at the Burgers' equation as an example.

```python
def burgers_equation(nx, nt, Lx, Lt, nu):
    dx = Lx / (nx - 1)
    dt = Lt / (nt - 1)
    x = np.linspace(0, Lx, nx)
    t = np.linspace(0, Lt, nt)
    
    u = np.zeros((nt, nx))
    u[0, :] = np.sin(2 * np.pi * x / Lx)  # Initial condition
    
    for n in range(nt - 1):
        u[n+1, 0] = u[n+1, -1] = 0  # Boundary conditions
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] - u[n, i] * dt / dx * (u[n, i] - u[n, i-1]) + \
                        nu * dt / dx**2 * (u[n, i+1] - 2 * u[n, i] + u[n, i-1])
    
    return u, x, t

nx, nt = 100, 500
Lx, Lt = 2, 1
nu = 0.01  # Viscosity

u, x, t = burgers_equation(nx, nt, Lx, Lt, nu)

plt.imshow(u, extent=[0, Lx, 0, Lt], aspect='auto', origin='lower', cmap='coolwarm')
plt.colorbar(label='Velocity')
plt.title("Solution of Burgers' Equation")
plt.xlabel("Position")
plt.ylabel("Time")
plt.show()
```

Slide 16: Additional Resources

For further study on Partial Differential Equations, consider these resources:

1. ArXiv.org papers:
   * "Numerical Methods for Partial Differential Equations" by J. S. Hesthaven (arXiv:1803.07454)
   * "Machine Learning for Partial Differential Equations" by M. Raissi et al. (arXiv:1706.02242)
2. Online courses:
   * MIT OpenCourseWare: "Partial Differential Equations" (18.152)
   * Coursera: "Numerical Methods for Partial Differential Equations" by The Hong Kong University of Science and Technology
3. Textbooks:
   * "Partial Differential Equations for Scientists and Engineers" by Stanley J. Farlow
   * "Numerical Solution of Partial Differential Equations" by K. W. Morton and D. F. Mayers

Remember to verify the accuracy and relevance of these resources, as they may have been updated or changed since my last knowledge update.

