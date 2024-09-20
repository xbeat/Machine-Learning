## Weakly Differentiable Functions using Python
Slide 1: Introduction to Weakly Differentiable Functions

Weakly differentiable functions are a generalization of differentiable functions in the context of functional analysis and partial differential equations. These functions may not be differentiable in the classical sense but possess a weak derivative that satisfies certain integral conditions. This concept is crucial in the study of Sobolev spaces and the weak formulation of PDEs.

```python
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x):
    return np.where(x >= 0, 1, 0)

x = np.linspace(-2, 2, 1000)
y = heaviside(x)

plt.plot(x, y)
plt.title("Heaviside Step Function")
plt.xlabel("x")
plt.ylabel("H(x)")
plt.grid(True)
plt.show()
```

Slide 2: Classical vs. Weak Derivatives

Classical derivatives require functions to be smooth and continuous. However, many physical phenomena are described by functions with discontinuities. Weak derivatives extend the concept of differentiation to these functions, allowing us to solve a broader class of problems in physics and engineering.

```python
import numpy as np
import matplotlib.pyplot as plt

def classical_derivative(x):
    return np.where(x != 0, 0, np.inf)

def weak_derivative(x):
    return np.zeros_like(x)

x = np.linspace(-2, 2, 1000)
y_classical = classical_derivative(x)
y_weak = weak_derivative(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y_classical)
plt.title("Classical Derivative of Heaviside")
plt.ylim(-1, 1)  # Limiting y-axis for visibility
plt.subplot(1, 2, 2)
plt.plot(x, y_weak)
plt.title("Weak Derivative of Heaviside")
plt.tight_layout()
plt.show()
```

Slide 3: Definition of Weak Derivatives

A function u is weakly differentiable if there exists a function v such that for all smooth test functions φ with compact support:

∫ u(x) φ'(x) dx = -∫ v(x) φ(x) dx

Here, v is called the weak derivative of u. This definition allows us to extend the notion of derivatives to functions that may have discontinuities or lack smoothness at certain points.

```python
import numpy as np
from scipy.integrate import quad

def u(x):  # Heaviside function
    return np.heaviside(x, 1)

def v(x):  # Weak derivative of Heaviside
    return 0

def test_function(x):
    return np.exp(-x**2)  # Gaussian as a test function

def test_function_derivative(x):
    return -2 * x * np.exp(-x**2)

left_integral, _ = quad(lambda x: u(x) * test_function_derivative(x), -np.inf, np.inf)
right_integral, _ = quad(lambda x: -v(x) * test_function(x), -np.inf, np.inf)

print(f"Left integral: {left_integral:.6f}")
print(f"Right integral: {right_integral:.6f}")
print(f"Difference: {abs(left_integral - right_integral):.6f}")
```

Slide 4: Sobolev Spaces

Sobolev spaces are vector spaces of functions with weak derivatives up to a certain order. These spaces are crucial in the study of partial differential equations. The most common Sobolev space is H¹(Ω), which consists of L² functions with weak first derivatives also in L².

```python
import numpy as np
from scipy.integrate import simps

def function_in_H1(x):
    return np.abs(x)  # |x| is in H¹

def weak_derivative(x):
    return np.sign(x)

x = np.linspace(-1, 1, 1000)
f = function_in_H1(x)
df = weak_derivative(x)

L2_norm = np.sqrt(simps(f**2, x))
H1_seminorm = np.sqrt(simps(df**2, x))
H1_norm = np.sqrt(L2_norm**2 + H1_seminorm**2)

print(f"L² norm: {L2_norm:.6f}")
print(f"H¹ seminorm: {H1_seminorm:.6f}")
print(f"H¹ norm: {H1_norm:.6f}")
```

Slide 5: Weak Formulation of PDEs

The weak formulation of partial differential equations is a powerful tool that allows us to solve problems involving discontinuous functions or irregular domains. It replaces the strong form of the PDE with an integral equation that holds for all test functions in a suitable function space.

```python
import numpy as np
from scipy.integrate import quad

def weak_poisson_solver(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    
    A = np.zeros((n-1, n-1))
    b = np.zeros(n-1)
    
    for i in range(n-1):
        A[i, i] = 2 / h
        if i > 0:
            A[i, i-1] = -1 / h
        if i < n-2:
            A[i, i+1] = -1 / h
        
        b[i] = quad(f, x[i], x[i+2])[0]
    
    u = np.linalg.solve(A, b)
    return np.concatenate(([0], u, [0]))

# Example: -u'' = 1 on (0,1) with u(0) = u(1) = 0
f = lambda x: 1
x = np.linspace(0, 1, 101)
u = weak_poisson_solver(f, 0, 1, 100)

import matplotlib.pyplot as plt
plt.plot(x, u)
plt.title("Solution to -u'' = 1 with u(0) = u(1) = 0")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.show()
```

Slide 6: Distributions and Generalized Functions

Distributions, or generalized functions, provide a rigorous framework for dealing with objects like the Dirac delta function. These objects are not functions in the classical sense but can be understood as continuous linear functionals on a space of test functions. Weakly differentiable functions can be viewed as a special case of distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def dirac_delta_approximation(x, epsilon):
    return 1 / (np.pi * epsilon * (1 + (x/epsilon)**2))

x = np.linspace(-1, 1, 1000)
epsilon_values = [0.1, 0.05, 0.01]

plt.figure(figsize=(10, 6))
for epsilon in epsilon_values:
    y = dirac_delta_approximation(x, epsilon)
    plt.plot(x, y, label=f'ε = {epsilon}')

plt.title("Approximations of the Dirac Delta Function")
plt.xlabel("x")
plt.ylabel("δ(x)")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 7: Numerical Methods for Weakly Differentiable Functions

Finite Element Method (FEM) is a powerful numerical technique for solving PDEs, especially suitable for problems involving weakly differentiable functions. FEM discretizes the domain into small elements and approximates the solution using piecewise polynomial functions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def fem_1d(n, f):
    h = 1.0 / n
    x = np.linspace(0, 1, n+1)
    
    # Assemble stiffness matrix and load vector
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(n-1, n-1)) / h**2
    b = f(x[1:-1]) * h
    
    # Solve the system
    u = np.zeros(n+1)
    u[1:-1] = spsolve(A, b)
    
    return x, u

# Example: -u'' = 1 on (0,1) with u(0) = u(1) = 0
f = lambda x: np.ones_like(x)
x, u = fem_1d(100, f)

plt.plot(x, u)
plt.title("FEM Solution to -u'' = 1 with u(0) = u(1) = 0")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.show()
```

Slide 8: Real-life Example: Heat Conduction in a Rod

Consider a rod with varying thermal conductivity. The heat equation describing this system involves weakly differentiable functions due to potential discontinuities in the material properties. We can solve this problem using the weak formulation and FEM.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def heat_conduction_fem(n, k, f):
    h = 1.0 / n
    x = np.linspace(0, 1, n+1)
    
    # Assemble stiffness matrix and load vector
    K = np.zeros((n-1, n-1))
    for i in range(n-1):
        K[i, i] = (k(x[i]) + k(x[i+1])) / h
        if i > 0:
            K[i, i-1] = -k(x[i]) / h
        if i < n-2:
            K[i, i+1] = -k(x[i+1]) / h
    
    b = f(x[1:-1]) * h
    
    # Solve the system
    u = np.zeros(n+1)
    u[1:-1] = spsolve(K, b)
    
    return x, u

# Example: -(k(x)u')' = 1 on (0,1) with u(0) = u(1) = 0
# k(x) = 1 for x < 0.5, k(x) = 2 for x >= 0.5
k = lambda x: np.where(x < 0.5, 1, 2)
f = lambda x: np.ones_like(x)

x, u = heat_conduction_fem(100, k, f)

plt.plot(x, u)
plt.title("Heat Distribution in a Rod with Varying Conductivity")
plt.xlabel("Position")
plt.ylabel("Temperature")
plt.grid(True)
plt.show()
```

Slide 9: Real-life Example: Elastic Beam with Variable Stiffness

Consider an elastic beam with variable stiffness subjected to a distributed load. The beam's deflection is described by a fourth-order differential equation, which can be solved using weak formulation and FEM. This example demonstrates the application of weakly differentiable functions in structural mechanics.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def beam_deflection_fem(n, EI, q):
    h = 1.0 / n
    x = np.linspace(0, 1, n+1)
    
    # Assemble stiffness matrix and load vector
    K = np.zeros((n-3, n-3))
    for i in range(n-3):
        K[i, i] = (EI(x[i]) + EI(x[i+1]) + EI(x[i+2]) + EI(x[i+3])) / h**3
        if i > 0:
            K[i, i-1] = -(EI(x[i]) + EI(x[i+1]) + EI(x[i+2])) / h**3
        if i < n-4:
            K[i, i+1] = -(EI(x[i+1]) + EI(x[i+2]) + EI(x[i+3])) / h**3
        if i > 1:
            K[i, i-2] = EI(x[i+1]) / h**3
        if i < n-5:
            K[i, i+2] = EI(x[i+2]) / h**3
    
    b = q(x[2:-2]) * h
    
    # Solve the system
    w = np.zeros(n+1)
    w[2:-2] = spsolve(K, b)
    
    return x, w

# Example: (EI(x)w'''')' = q on (0,1) with w(0) = w'(0) = w(1) = w'(1) = 0
# EI(x) = 1 for x < 0.5, EI(x) = 2 for x >= 0.5
EI = lambda x: np.where(x < 0.5, 1, 2)
q = lambda x: np.ones_like(x)

x, w = beam_deflection_fem(100, EI, q)

plt.plot(x, w)
plt.title("Deflection of a Beam with Variable Stiffness")
plt.xlabel("Position")
plt.ylabel("Deflection")
plt.grid(True)
plt.show()
```

Slide 10: Weak Derivatives and Regularization

Regularization techniques often involve weakly differentiable functions. For instance, Total Variation (TV) regularization uses the L¹ norm of the gradient, which is well-defined for weakly differentiable functions. This approach is particularly useful in image processing for edge-preserving denoising.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def tv_denoising(image, lambda_, iterations):
    u = image.()
    
    for _ in range(iterations):
        grad_x = np.roll(u, -1, axis=1) - u
        grad_y = np.roll(u, -1, axis=0) - u
        
        div = np.roll(grad_x, 1, axis=1) + np.roll(grad_y, 1, axis=0)
        
        u = u + lambda_ * div / (1 + lambda_ * np.sqrt(grad_x**2 + grad_y**2))
    
    return u

# Create a noisy image
np.random.seed(0)
image = np.zeros((100, 100))
image[25:75, 25:75] = 1
noisy_image = image + 0.2 * np.random.randn(*image.shape)

# Apply TV denoising
denoised_image = tv_denoising(noisy_image, lambda_=0.1, iterations=100)

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title("Original Image")
ax2.imshow(noisy_image, cmap='gray')
ax2.set_title("Noisy Image")
ax3.imshow(denoised_image, cmap='gray')
ax3.set_title("Denoised Image (TV)")
plt.tight_layout()
plt.show()
```

Slide 11: Weak Derivatives in Machine Learning

Weakly differentiable functions play a crucial role in machine learning, particularly in the design of activation functions and loss functions. The Rectified Linear Unit (ReLU) activation function is weakly differentiable at x=0, allowing for efficient gradient-based optimization in neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

x = np.linspace(-2, 2, 1000)
y_relu = relu(x)
y_relu_derivative = relu_derivative(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y_relu)
plt.title("ReLU Function")
plt.subplot(1, 2, 2)
plt.plot(x, y_relu_derivative)
plt.title("ReLU Derivative")
plt.tight_layout()
plt.show()
```

Slide 12: Applications in Signal Processing

Weakly differentiable functions are essential in signal processing, particularly in wavelet analysis. Wavelets, such as the Haar wavelet, are often weakly differentiable and provide a powerful tool for multi-resolution analysis of signals.

```python
import numpy as np
import matplotlib.pyplot as plt

def haar_wavelet(x):
    return np.where((x >= 0) & (x < 0.5), 1,
                    np.where((x >= 0.5) & (x < 1), -1, 0))

x = np.linspace(-0.5, 1.5, 1000)
y = haar_wavelet(x)

plt.plot(x, y)
plt.title("Haar Wavelet")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.grid(True)
plt.show()
```

Slide 13: Weak Derivatives in Variational Problems

Variational problems often involve minimizing functionals of weakly differentiable functions. The calculus of variations provides tools to find solutions to these problems, which have applications in physics, engineering, and image processing.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def euler_lagrange(y, x, p, q, f):
    y1, y2 = y
    dydx = [y2, -p(x)*y2 - q(x)*y1 + f(x)]
    return dydx

# Example: y'' + y = x on (0,π) with y(0) = y(π) = 0
p = lambda x: 0
q = lambda x: 1
f = lambda x: x

x = np.linspace(0, np.pi, 100)
y0 = [0, 1]  # Initial conditions: y(0) = 0, y'(0) = 1

sol = odeint(euler_lagrange, y0, x, args=(p, q, f))

plt.plot(x, sol[:, 0])
plt.title("Solution to y'' + y = x")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.show()
```

Slide 14: Challenges and Future Directions

The study of weakly differentiable functions continues to evolve, with ongoing research in areas such as:

1. Numerical methods for solving PDEs with low regularity solutions
2. Applications in machine learning and deep neural networks
3. Extension to functions of bounded variation and other function spaces
4. Development of more efficient algorithms for problems involving weak derivatives

These advancements promise to expand the applicability of weakly differentiable functions in various fields of science and engineering.

Slide 15: Additional Resources

For those interested in delving deeper into the topic of weakly differentiable functions, here are some valuable resources:

1. Evans, L. C. (2010). Partial Differential Equations. American Mathematical Society. ArXiv: [https://arxiv.org/abs/math/0606721](https://arxiv.org/abs/math/0606721)
2. Brezis, H. (2010). Functional Analysis, Sobolev Spaces and Partial Differential Equations. Springer. ArXiv: [https://arxiv.org/abs/math/0601350](https://arxiv.org/abs/math/0601350)
3. Leoni, G. (2017). A First Course in Sobolev Spaces. American Mathematical Society. ArXiv: [https://arxiv.org/abs/1603.05033](https://arxiv.org/abs/1603.05033)

These resources provide a comprehensive foundation for understanding weakly differentiable functions and their applications in various fields of mathematics and engineering.



