## Elementary Functional Analysis with Python

Slide 1: Introduction to Elementary Functional Analysis

Functional analysis is a branch of mathematics that studies vector spaces and the linear operators acting on them. It combines concepts from linear algebra, topology, and analysis.

```python
import numpy as np
from typing import Callable

def linear_operator(T: Callable, x: np.ndarray) -> np.ndarray:
    """
    Applies a linear operator T to vector x
    """
    return T(x)

# Example linear operator (matrix multiplication)
A = np.array([[1, 2], [3, 4]])
T = lambda x: A @ x

x = np.array([1, 2])
result = linear_operator(T, x)
print(f"T(x) = {result}")
```

Slide 2: Normed Vector Spaces

A normed vector space is a vector space equipped with a norm, which measures the "length" of vectors.

```python
import numpy as np

def euclidean_norm(x: np.ndarray) -> float:
    """
    Computes the Euclidean norm of a vector
    """
    return np.sqrt(np.sum(x**2))

x = np.array([3, 4])
norm_x = euclidean_norm(x)
print(f"||x|| = {norm_x}")

# Verifying norm properties
y = np.array([1, 2])
alpha = 2

print(f"||x + y|| ≤ ||x|| + ||y||: {euclidean_norm(x + y) <= euclidean_norm(x) + euclidean_norm(y)}")
print(f"||αx|| = |α| * ||x||: {np.isclose(euclidean_norm(alpha * x), abs(alpha) * euclidean_norm(x))}")
```

Slide 3: Banach Spaces

A Banach space is a complete normed vector space. Completeness means that every Cauchy sequence converges to a point in the space.

```python
import numpy as np

def is_cauchy(sequence: list, epsilon: float = 1e-6) -> bool:
    """
    Checks if a sequence is Cauchy
    """
    for i in range(len(sequence)):
        for j in range(i + 1, len(sequence)):
            if abs(sequence[i] - sequence[j]) > epsilon:
                return False
    return True

# Example: convergent sequence in R (a Banach space)
sequence = [1 + 1/n for n in range(1, 1001)]
print(f"Is the sequence Cauchy? {is_cauchy(sequence)}")
print(f"Limit of the sequence: {sequence[-1]}")
```

Slide 4: Inner Product Spaces

An inner product space is a vector space equipped with an inner product, which allows us to define angles and orthogonality between vectors.

```python
import numpy as np

def inner_product(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the inner product of two vectors
    """
    return np.dot(x, y)

def angle_between(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the angle between two vectors
    """
    cos_theta = inner_product(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

x = np.array([1, 0])
y = np.array([0, 1])

print(f"Inner product of x and y: {inner_product(x, y)}")
print(f"Angle between x and y: {np.degrees(angle_between(x, y))} degrees")
```

Slide 5: Hilbert Spaces

A Hilbert space is a complete inner product space. It generalizes the notion of Euclidean space and provides the setting for much of quantum mechanics.

```python
import numpy as np

def gram_schmidt(vectors: list) -> list:
    """
    Performs Gram-Schmidt orthogonalization
    """
    orthogonalized = []
    for v in vectors:
        w = v - sum(np.dot(v, u) * u for u in orthogonalized)
        if not np.allclose(w, 0):
            orthogonalized.append(w / np.linalg.norm(w))
    return orthogonalized

# Example: orthogonalizing vectors in R^3
vectors = [np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1])]
orthonormal_basis = gram_schmidt(vectors)

print("Orthonormal basis:")
for v in orthonormal_basis:
    print(v)
```

Slide 6: Bounded Linear Operators

A bounded linear operator is a linear operator between normed vector spaces that is continuous.

```python
import numpy as np

def is_bounded_linear_operator(T: callable, domain: np.ndarray, codomain: np.ndarray) -> bool:
    """
    Checks if T is a bounded linear operator
    """
    # Check linearity
    x, y = np.random.rand(domain.shape[1]), np.random.rand(domain.shape[1])
    alpha, beta = np.random.rand(), np.random.rand()
    linearity = np.allclose(T(alpha*x + beta*y), alpha*T(x) + beta*T(y))
    
    # Check boundedness
    norm_T = np.max([np.linalg.norm(T(x)) / np.linalg.norm(x) for x in domain])
    boundedness = np.isfinite(norm_T)
    
    return linearity and boundedness

# Example: matrix multiplication as a bounded linear operator
A = np.array([[1, 2], [3, 4]])
T = lambda x: A @ x
domain = np.random.rand(100, 2)
codomain = np.random.rand(100, 2)

print(f"Is T a bounded linear operator? {is_bounded_linear_operator(T, domain, codomain)}")
```

Slide 7: Spectral Theory

Spectral theory studies the properties of linear operators through their eigenvalues and eigenvectors.

```python
import numpy as np

def power_method(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> tuple:
    """
    Implements the power method to find the dominant eigenvalue and eigenvector
    """
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for _ in range(max_iter):
        x_new = A @ x
        lambda_new = np.dot(x, A @ x)
        
        if np.linalg.norm(x_new - lambda_new * x) < tol:
            return lambda_new, x_new / np.linalg.norm(x_new)
        
        x = x_new / np.linalg.norm(x_new)
    
    raise ValueError("Power method did not converge")

# Example: finding the dominant eigenvalue and eigenvector
A = np.array([[4, -1], [2, 1]])
eigenvalue, eigenvector = power_method(A)

print(f"Dominant eigenvalue: {eigenvalue}")
print(f"Corresponding eigenvector: {eigenvector}")
```

Slide 8: Compact Operators

Compact operators are linear operators that map bounded sets to relatively compact sets. They have properties similar to finite-dimensional operators.

```python
import numpy as np

def is_compact_operator(T: callable, domain: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Checks if T is a compact operator (simplified version)
    """
    # Generate a sequence in the unit ball
    sequence = [x / np.linalg.norm(x) for x in domain]
    
    # Apply T to the sequence
    T_sequence = [T(x) for x in sequence]
    
    # Check if the image has a convergent subsequence
    for i in range(len(T_sequence)):
        for j in range(i + 1, len(T_sequence)):
            if np.linalg.norm(T_sequence[i] - T_sequence[j]) < epsilon:
                return True
    
    return False

# Example: integral operator (which is compact)
def integral_operator(f: callable) -> callable:
    return lambda x: np.array([np.trapz([f(t) * np.sin(x[0] * t) for t in np.linspace(0, 1, 100)], np.linspace(0, 1, 100))])

T = integral_operator(lambda t: t**2)
domain = np.random.rand(100, 1)

print(f"Is T a compact operator? {is_compact_operator(T, domain)}")
```

Slide 9: Fredholm Theory

Fredholm theory deals with integral equations and provides a framework for solving certain types of operator equations.

```python
import numpy as np
from scipy.integrate import quad

def fredholm_equation(kernel: callable, f: callable, lambda_: float, a: float, b: float) -> callable:
    """
    Solves the Fredholm equation of the second kind:
    phi(x) = f(x) + lambda * integral(K(x,t) * phi(t), t=a..b)
    """
    def phi(x):
        integral, _ = quad(lambda t: kernel(x, t) * phi(t), a, b)
        return f(x) + lambda_ * integral
    
    return phi

# Example: solving a simple Fredholm equation
kernel = lambda x, t: np.exp(-(x-t)**2)
f = lambda x: np.sin(np.pi * x)
lambda_ = 0.5
a, b = 0, 1

solution = fredholm_equation(kernel, f, lambda_, a, b)

# Evaluate the solution at some points
x_values = np.linspace(a, b, 10)
y_values = [solution(x) for x in x_values]

print("Solution of the Fredholm equation:")
for x, y in zip(x_values, y_values):
    print(f"phi({x:.2f}) = {y:.4f}")
```

Slide 10: Functional Analysis in Quantum Mechanics

Functional analysis provides the mathematical framework for quantum mechanics, where states are represented as vectors in Hilbert spaces.

```python
import numpy as np

def expectation_value(operator: np.ndarray, state: np.ndarray) -> float:
    """
    Computes the expectation value of an operator for a given state
    """
    return np.real(np.dot(state.conj(), np.dot(operator, state)))

# Example: computing expectation value of energy for a particle in a box
def energy_operator(n: int) -> np.ndarray:
    """
    Energy operator for a particle in a box
    """
    return np.diag([(np.pi * k / L)**2 for k in range(1, n+1)])

L = 1  # Box length
n = 5  # Number of basis states

H = energy_operator(n)
psi = np.random.rand(n) + 1j * np.random.rand(n)
psi = psi / np.linalg.norm(psi)  # Normalize the state

E = expectation_value(H, psi)
print(f"Expected energy: {E}")
```

Slide 11: Functional Analysis in Signal Processing

Functional analysis is crucial in signal processing, particularly in the study of Fourier transforms and filter design.

```python
import numpy as np
import matplotlib.pyplot as plt

def fourier_series(f: callable, T: float, n: int) -> callable:
    """
    Computes the Fourier series approximation of a function
    """
    def a(k):
        return 2/T * np.trapz([f(t) * np.cos(2*np.pi*k*t/T) for t in np.linspace(0, T, 1000)], np.linspace(0, T, 1000))
    
    def b(k):
        return 2/T * np.trapz([f(t) * np.sin(2*np.pi*k*t/T) for t in np.linspace(0, T, 1000)], np.linspace(0, T, 1000))
    
    return lambda t: a(0)/2 + sum(a(k)*np.cos(2*np.pi*k*t/T) + b(k)*np.sin(2*np.pi*k*t/T) for k in range(1, n+1))

# Example: approximating a square wave
T = 2
f = lambda t: 1 if t % T < T/2 else -1
f_approx = fourier_series(f, T, 10)

t = np.linspace(0, 2*T, 1000)
plt.plot(t, [f(ti) for ti in t], label='Original')
plt.plot(t, [f_approx(ti) for ti in t], label='Approximation')
plt.legend()
plt.title('Fourier Series Approximation of Square Wave')
plt.show()
```

Slide 12: Functional Analysis in Optimization

Optimization problems in infinite-dimensional spaces are studied using functional analysis, with applications in control theory and machine learning.

```python
import numpy as np
from scipy.optimize import minimize_scalar

def functional_optimization(J: callable, a: float, b: float) -> tuple:
    """
    Minimizes a functional J over the interval [a, b]
    """
    result = minimize_scalar(J, bounds=(a, b), method='bounded')
    return result.x, result.fun

# Example: finding the function that minimizes the integral of (f'(x)^2 + f(x)^2) dx
def J(alpha):
    f = lambda x: alpha * np.exp(-x)
    integrand = lambda x: (f(x)**2 + (alpha * np.exp(-x))**2)
    integral, _ = np.quad(integrand, 0, np.inf)
    return integral

a, b = 0, 10
optimal_alpha, min_value = functional_optimization(J, a, b)

print(f"Optimal alpha: {optimal_alpha}")
print(f"Minimum value of the functional: {min_value}")
```

Slide 13: Functional Analysis in Partial Differential Equations

Functional analysis provides tools for studying the existence, uniqueness, and properties of solutions to partial differential equations.

```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solve_heat_equation(u0: callable, T: float, L: float, Nx: int, Nt: int) -> np.ndarray:
    """
    Solves the 1D heat equation using implicit finite differences
    """
    dx = L / (Nx - 1)
    dt = T / Nt
    r = dt / (dx**2)
    
    # Set up the tridiagonal matrix
    main_diag = np.ones(Nx) * (1 + 2*r)
    off_diag = np.ones(Nx-1) * (-r)
    A = diags([main_diag, off_diag, off_diag], [0, -1, 1]).tocsr()
    
    # Initial condition
    u = np.array([u0(x) for x in np.linspace(0, L, Nx)])
    
    # Time stepping
    for _ in range(Nt):
        u = spsolve(A, u)
    
    return u

# Example: solving the heat equation
L, T = 1, 0.1
Nx, Nt = 50, 1000
u0 = lambda x: np.sin(np.pi * x / L)

solution = solve_heat_equation(u0, T, L, Nx, Nt)

x = np.linspace(0, L, Nx)
plt.plot(x, solution)
plt.title(f'Solution of the Heat Equation at t = {T}')
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.show()
```

Slide 14: Functional Analysis in Machine Learning

Functional analysis concepts are fundamental in understanding and developing machine learning algorithms, particularly in kernel methods and neural networks.

```python
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

def rbf_kernel_svm(X: np.ndarray, y: np.ndarray, gamma: float = 1.0, n_components: int = 100) -> callable:
    """
    Trains an approximate RBF kernel SVM using random Fourier features
    """
    rbf_feature = RBFSampler(gamma=gamma, n_components=n_components, random_state=1)
    clf = make_pipeline(rbf_feature, SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(X, y)
    return clf.predict

# Example: binary classification with RBF kernel SVM
np.random.seed(0)
X = np.random.randn(200, 2)
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)

predict = rbf_kernel_svm(X, y)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.title('RBF Kernel SVM Decision Boundary')
plt.show()
```

Slide 15: Additional Resources

For further study in Elementary Functional Analysis, consider the following resources:

1. ArXiv: "Functional Analysis: An Elementary Introduction" by Markus Haase URL: [https://arxiv.org/abs/1204.2551](https://arxiv.org/abs/1204.2551)
2. ArXiv: "A Concise Course in Functional Analysis" by Xue-Feng Wang URL: [https://arxiv.org/abs/2106.02064](https://arxiv.org/abs/2106.02064)
3. ArXiv: "Functional Analysis and Operator Theory" by Gilles Pisier URL: [https://arxiv.org/abs/1101.4845](https://arxiv.org/abs/1101.4845)

These papers provide in-depth coverage of the topics discussed in this presentation and can serve as a starting point for more advanced study in functional analysis.

