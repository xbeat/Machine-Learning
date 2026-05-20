## Visualizing Jordan Canonical Form in Python
Slide 1: Introduction to Jordan Canonical Form

Jordan Canonical Form is a powerful concept in linear algebra that allows us to represent a matrix in a simplified, block-diagonal structure. This form is particularly useful for understanding the behavior of linear transformations and solving systems of differential equations.

```python
import numpy as np

def is_diagonalizable(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return np.linalg.matrix_rank(eigenvectors) == matrix.shape[0]

# Example matrix
A = np.array([[3, 1], [0, 3]])
print(f"Is A diagonalizable? {is_diagonalizable(A)}")
```

Slide 2: Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are crucial for understanding Jordan Canonical Form. An eigenvector is a non-zero vector that, when a linear transformation is applied, changes only by a scalar factor (the eigenvalue).

```python
import numpy as np

# Define a matrix
A = np.array([[4, -2], [1, 1]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```

Slide 3: Algebraic and Geometric Multiplicity

The algebraic multiplicity of an eigenvalue is its multiplicity as a root of the characteristic polynomial. The geometric multiplicity is the dimension of the eigenspace associated with that eigenvalue.

```python
import numpy as np

def algebraic_multiplicity(matrix, eigenvalue):
    char_poly = np.poly(matrix)
    roots = np.roots(char_poly)
    return np.sum(np.isclose(roots, eigenvalue))

def geometric_multiplicity(matrix, eigenvalue):
    eigenspace = null_space(matrix - eigenvalue * np.eye(matrix.shape[0]))
    return eigenspace.shape[1]

def null_space(A):
    u, s, vh = np.linalg.svd(A)
    return vh[np.sum(s > 1e-10):].T

# Example matrix
A = np.array([[3, 1, 0], [0, 3, 0], [0, 0, 2]])
eigenvalue = 3

print(f"Algebraic multiplicity: {algebraic_multiplicity(A, eigenvalue)}")
print(f"Geometric multiplicity: {geometric_multiplicity(A, eigenvalue)}")
```

Slide 4: Generalized Eigenvectors

When the algebraic multiplicity of an eigenvalue is greater than its geometric multiplicity, we need generalized eigenvectors to form a Jordan chain. These vectors satisfy (A - λI)^k v = 0 for some k > 1.

```python
import numpy as np

def generalized_eigenvector(A, eigenvalue, eigenvector, k):
    n = A.shape[0]
    I = np.eye(n)
    B = A - eigenvalue * I
    v = eigenvector
    for _ in range(k-1):
        v = np.linalg.solve(B, v)
    return v

# Example matrix
A = np.array([[3, 1, 0], [0, 3, 0], [0, 0, 2]])
eigenvalue = 3
eigenvector = np.array([1, 0, 0])

gen_eigenvector = generalized_eigenvector(A, eigenvalue, eigenvector, 2)
print("Generalized eigenvector:", gen_eigenvector)
```

Slide 5: Jordan Blocks

A Jordan block is a square matrix with an eigenvalue λ on the main diagonal, 1's on the superdiagonal, and 0's elsewhere. The size of the Jordan block corresponds to the length of the Jordan chain for that eigenvalue.

```python
import numpy as np

def jordan_block(eigenvalue, size):
    block = np.zeros((size, size))
    np.fill_diagonal(block, eigenvalue)
    np.fill_diagonal(block[:, 1:], 1)
    return block

# Create a 3x3 Jordan block with eigenvalue 2
J = jordan_block(2, 3)
print("3x3 Jordan block with eigenvalue 2:")
print(J)
```

Slide 6: Constructing the Jordan Canonical Form

To construct the Jordan Canonical Form, we arrange Jordan blocks for each eigenvalue along the diagonal of a matrix. The sizes of these blocks are determined by the algebraic and geometric multiplicities of the eigenvalues.

```python
import numpy as np

def jordan_canonical_form(A):
    eigenvalues, _ = np.linalg.eig(A)
    unique_eigenvalues = np.unique(eigenvalues)
    
    jordan_form = np.zeros_like(A)
    current_index = 0
    
    for eigenvalue in unique_eigenvalues:
        alg_mult = np.sum(np.isclose(eigenvalues, eigenvalue))
        geo_mult = np.linalg.matrix_rank(A - eigenvalue * np.eye(A.shape[0]))
        
        for size in range(alg_mult, geo_mult, -1):
            block = jordan_block(eigenvalue, size)
            end_index = current_index + size
            jordan_form[current_index:end_index, current_index:end_index] = block
            current_index = end_index
    
    return jordan_form

# Example matrix
A = np.array([[3, 1, 0], [0, 3, 0], [0, 0, 2]])
J = jordan_canonical_form(A)
print("Jordan Canonical Form:")
print(J)
```

Slide 7: Change of Basis Matrix

To find the Jordan Canonical Form, we need a change of basis matrix P such that P^(-1)AP = J, where J is the Jordan Canonical Form. The columns of P are the generalized eigenvectors.

```python
import numpy as np

def change_of_basis_matrix(A, J):
    n = A.shape[0]
    P = np.zeros((n, n))
    
    for i in range(n):
        if i > 0 and np.isclose(J[i, i], J[i-1, i-1]) and np.isclose(J[i, i-1], 1):
            P[:, i] = (A - J[i, i] * np.eye(n)) @ P[:, i-1]
        else:
            eigenvalue = J[i, i]
            P[:, i] = null_space(A - eigenvalue * np.eye(n))[:, 0]
    
    return P

def null_space(A):
    u, s, vh = np.linalg.svd(A)
    return vh[np.sum(s > 1e-10):].T

# Example matrix
A = np.array([[3, 1, 0], [0, 3, 0], [0, 0, 2]])
J = jordan_canonical_form(A)
P = change_of_basis_matrix(A, J)

print("Change of basis matrix P:")
print(P)
print("\nVerification: P^(-1)AP = J")
print(np.allclose(np.linalg.inv(P) @ A @ P, J))
```

Slide 8: Properties of Jordan Canonical Form

Jordan Canonical Form has several important properties: it's unique up to the order of Jordan blocks, it preserves eigenvalues, and it simplifies matrix powers and exponentials.

```python
import numpy as np

def matrix_power(A, n):
    return np.linalg.matrix_power(A, n)

def matrix_exponential(A):
    return np.linalg.expm(A)

# Example matrix
A = np.array([[3, 1, 0], [0, 3, 0], [0, 0, 2]])
J = jordan_canonical_form(A)
P = change_of_basis_matrix(A, J)

# Compute A^5 and e^A
A_power_5 = matrix_power(A, 5)
A_exp = matrix_exponential(A)

# Compute J^5 and e^J
J_power_5 = matrix_power(J, 5)
J_exp = matrix_exponential(J)

print("A^5 == P * J^5 * P^(-1):")
print(np.allclose(A_power_5, P @ J_power_5 @ np.linalg.inv(P)))

print("\ne^A == P * e^J * P^(-1):")
print(np.allclose(A_exp, P @ J_exp @ np.linalg.inv(P)))
```

Slide 9: Applications in Differential Equations

Jordan Canonical Form is particularly useful for solving systems of linear differential equations. It allows us to decouple the system and solve each component independently.

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def system_ode(y, t, A):
    return A @ y

def plot_solution(t, y):
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[:, 0], label='y1')
    plt.plot(t, y[:, 1], label='y2')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.title('Solution of the ODE system')
    plt.grid(True)
    plt.show()

# Example system: dy/dt = Ay
A = np.array([[3, 1], [0, 3]])
y0 = np.array([1, 1])
t = np.linspace(0, 1, 100)

# Solve the system
solution = odeint(system_ode, y0, t, args=(A,))

# Plot the solution
plot_solution(t, solution)
```

Slide 10: Real-life Example: Vibration Analysis

In mechanical engineering, Jordan Canonical Form can be used to analyze complex vibration systems. Consider a system of coupled oscillators, where the motion of each oscillator affects the others.

```python
import numpy as np
import matplotlib.pyplot as plt

def coupled_oscillators(t, y, m, k):
    x1, v1, x2, v2 = y
    dx1dt = v1
    dv1dt = (-2*k*x1 + k*x2) / m
    dx2dt = v2
    dv2dt = (k*x1 - 2*k*x2) / m
    return [dx1dt, dv1dt, dx2dt, dv2dt]

# System parameters
m = 1.0  # mass
k = 10.0  # spring constant

# Initial conditions
y0 = [1.0, 0.0, -1.0, 0.0]

# Time points
t = np.linspace(0, 10, 1000)

# Solve ODE
solution = odeint(coupled_oscillators, y0, t, args=(m, k))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, solution[:, 0], label='Oscillator 1')
plt.plot(t, solution[:, 2], label='Oscillator 2')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.title('Coupled Oscillators')
plt.grid(True)
plt.show()
```

Slide 11: Real-life Example: Population Dynamics

In ecology, Jordan Canonical Form can be applied to study population dynamics in complex ecosystems. Let's model a simple predator-prey system using the Lotka-Volterra equations.

```python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lotka_volterra(state, t, a, b, c, d):
    x, y = state
    dxdt = a*x - b*x*y
    dydt = -c*y + d*x*y
    return [dxdt, dydt]

# Parameters
a, b, c, d = 1, 0.1, 1.5, 0.075

# Initial conditions
x0 = 10  # initial prey population
y0 = 5   # initial predator population
state0 = [x0, y0]

# Time points
t = np.linspace(0, 100, 1000)

# Solve ODE
solution = odeint(lotka_volterra, state0, t, args=(a, b, c, d))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, solution[:, 0], label='Prey')
plt.plot(t, solution[:, 1], label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Predator-Prey Population Dynamics')
plt.grid(True)
plt.show()
```

Slide 12: Limitations and Considerations

While Jordan Canonical Form is powerful, it has some limitations. Numerical stability can be an issue, especially for large matrices or those with close eigenvalues. In practice, the Schur decomposition is often preferred for numerical computations.

```python
import numpy as np

def compare_jordan_schur(A):
    # Jordan Canonical Form
    J = jordan_canonical_form(A)
    
    # Schur decomposition
    T, U = np.linalg.schur(A)
    
    print("Jordan Canonical Form:")
    print(J)
    print("\nSchur decomposition:")
    print(T)
    
    # Compare numerical stability
    jordan_error = np.linalg.norm(A - np.linalg.inv(P) @ J @ P)
    schur_error = np.linalg.norm(A - U @ T @ U.T)
    
    print(f"\nJordan Form Error: {jordan_error}")
    print(f"Schur Decomposition Error: {schur_error}")

# Example matrix
A = np.array([[1, 1e-8], [0, 1]])
compare_jordan_schur(A)
```

Slide 13: Conclusion and Future Directions

Jordan Canonical Form is a fundamental concept in linear algebra with applications in various fields. As we've seen, it provides a powerful tool for analyzing linear systems and solving differential equations. Future research directions include developing more numerically stable algorithms for computing Jordan Canonical Form and extending the concept to infinite-dimensional spaces.

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_jordan_structure(J):
    G = nx.DiGraph()
    n = J.shape[0]
    for i in range(n):
        G.add_node(i, eigenvalue=f"{J[i,i]:.2f}")
        if i > 0 and J[i, i-1] == 1:
            G.add_edge(i-1, i)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrows=True)
    nx.draw_networkx_labels(G, pos, {i: f"λ={d['eigenvalue']}" for i, d in G.nodes(data=True)})
    plt.title("Jordan Structure Visualization")
    plt.axis('off')
    plt.show()

# Example Jordan Canonical Form
J = np.array([[2, 1, 0, 0],
              [0, 2, 0, 0],
              [0, 0, 3, 1],
              [0, 0, 0, 3]])

visualize_jordan_structure(J)
```

Slide 14: Additional Resources

For those interested in delving deeper into Jordan Canonical Form and its applications, here are some recommended resources:

1. "Matrix Analysis" by Roger A. Horn and Charles R. Johnson ArXiv link: [https://arxiv.org/abs/math/0605769](https://arxiv.org/abs/math/0605769)
2. "Linear Algebra and Its Applications" by Gilbert Strang ArXiv link: [https://arxiv.org/abs/1807.05155](https://arxiv.org/abs/1807.05155)
3. "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III ArXiv link: https

