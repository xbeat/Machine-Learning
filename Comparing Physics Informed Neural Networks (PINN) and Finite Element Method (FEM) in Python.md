## Comparing Physics Informed Neural Networks (PINN) and Finite Element Method (FEM) in Python
Slide 1: Introduction to Physics Informed Neural Networks (PINN) and Finite Element Method (FEM)

Physics Informed Neural Networks (PINN) and Finite Element Method (FEM) are two powerful approaches for solving complex physical problems. While FEM has been a cornerstone of computational physics for decades, PINNs represent a newer, machine learning-based approach. This slideshow will explore both methods, their strengths, and their applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_comparison():
    x = np.linspace(0, 10, 100)
    y_fem = np.sin(x) + np.random.normal(0, 0.1, 100)
    y_pinn = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_fem, label='FEM (with noise)')
    plt.plot(x, y_pinn, label='PINN')
    plt.legend()
    plt.title('Comparison of FEM and PINN Solutions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_comparison()
```

Slide 2: Fundamentals of Finite Element Method (FEM)

The Finite Element Method is a numerical technique for solving partial differential equations by dividing the domain into smaller, simpler parts called finite elements. It approximates complex equations with simpler ones, solving them over many small subdomains to obtain a global solution.

```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

def solve_1d_heat_equation(nx, nt, L, T, alpha):
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / (dx**2)
    
    # Create the tridiagonal matrix
    diagonals = [1, -2, 1]
    offsets = [-1, 0, 1]
    A = diags(diagonals, offsets, shape=(nx, nx))
    A = np.eye(nx) - r * A
    
    # Initial condition
    u = np.zeros(nx)
    u[0] = 0
    u[-1] = 1
    
    for _ in range(nt):
        u = spsolve(A, u)
    
    return u

# Solve the 1D heat equation
nx, nt = 50, 1000
L, T, alpha = 1.0, 0.5, 0.01
u = solve_1d_heat_equation(nx, nt, L, T, alpha)

plt.plot(np.linspace(0, L, nx), u)
plt.title('1D Heat Equation Solution using FEM')
plt.xlabel('Position')
plt.ylabel('Temperature')
plt.show()
```

Slide 3: Key Components of FEM

FEM involves several key steps: discretization of the domain, selection of interpolation functions, formulation of the system of equations, and solving the resulting system. The method's flexibility allows it to handle complex geometries and boundary conditions effectively.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_mesh(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_mesh(X, Y):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, np.zeros_like(X))
    ax.set_title('2D Mesh for FEM')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Create and plot a 2D mesh
X, Y = create_mesh(10, 10)
plot_mesh(X, Y)
```

Slide 4: Advantages of FEM

FEM excels in handling complex geometries, non-linear problems, and multiphysics simulations. It provides high accuracy for structural analysis, fluid dynamics, and heat transfer problems. FEM's maturity means it has extensive software support and a large community of users.

```python
import numpy as np
import matplotlib.pyplot as plt

def fem_beam_deflection(L, E, I, w, n):
    # L: length, E: Young's modulus, I: moment of inertia
    # w: distributed load, n: number of elements
    x = np.linspace(0, L, n+1)
    h = L / n
    
    K = np.zeros((n+1, n+1))
    F = np.zeros(n+1)
    
    # Assemble stiffness matrix and force vector
    for i in range(n):
        K[i:i+2, i:i+2] += E*I/h**3 * np.array([[12, 6*h], [6*h, 4*h**2]])
        F[i:i+2] += w*h/2 * np.array([1, h/6])
    
    # Apply boundary conditions
    K = K[1:-1, 1:-1]
    F = F[1:-1]
    
    # Solve for deflections
    u = np.linalg.solve(K, F)
    u = np.insert(u, [0, n], [0, 0])
    
    return x, u

# Example usage
L, E, I, w = 10, 200e9, 1e-4, 10000
x, u = fem_beam_deflection(L, E, I, w, 100)

plt.plot(x, u)
plt.title('Beam Deflection using FEM')
plt.xlabel('Position along beam')
plt.ylabel('Deflection')
plt.grid(True)
plt.show()
```

Slide 5: Limitations of FEM

Despite its strengths, FEM can be computationally expensive for large-scale problems or those requiring frequent remeshing. It may struggle with singularities, and the quality of results depends heavily on mesh quality and element choice. FEM also requires explicit knowledge of the governing equations.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_mesh_quality_data(n_elements, quality_range):
    mesh_qualities = np.random.uniform(*quality_range, n_elements)
    computation_times = 100 * np.exp(-5 * mesh_qualities) + np.random.normal(0, 5, n_elements)
    return mesh_qualities, computation_times

# Generate data
n_elements = 1000
mesh_qualities, computation_times = generate_mesh_quality_data(n_elements, (0.5, 1.0))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(mesh_qualities, computation_times, alpha=0.5)
plt.title('Effect of Mesh Quality on Computation Time')
plt.xlabel('Mesh Quality (higher is better)')
plt.ylabel('Computation Time (seconds)')
plt.show()
```

Slide 6: Introduction to Physics Informed Neural Networks (PINN)

Physics Informed Neural Networks combine the flexibility of neural networks with the constraints of physical laws. PINNs learn to solve differential equations by minimizing both the data mismatch and the residual of the governing physical equations.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class SimplePINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

# Create a simple PINN
model = SimplePINN()

# Plot the architecture
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
```

Slide 7: PINN Architecture and Training

PINNs typically use deep neural networks with multiple hidden layers. The loss function includes both data fitting terms and physics-based constraints. Training involves backpropagation through the entire computational graph, including the physics constraints.

```python
import tensorflow as tf
import numpy as np

def pinn_loss(model, x, y_true, pde_weight=1.0):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        dy_dx = tape.gradient(y_pred, x)
    
    # Data loss
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # PDE loss (example: dy/dx = y)
    pde_loss = tf.reduce_mean(tf.square(dy_dx - y_pred))
    
    return mse_loss + pde_weight * pde_loss

# Example usage
model = SimplePINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

x = tf.linspace(0, 1, 100)[:, None]
y_true = tf.exp(x)  # True solution to dy/dx = y

for _ in range(1000):
    with tf.GradientTape() as tape:
        loss = pinn_loss(model, x, y_true)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label='True')
plt.plot(x, model(x), label='PINN')
plt.legend()
plt.title('PINN Solution vs True Solution')
plt.show()
```

Slide 8: Advantages of PINNs

PINNs excel in scenarios with limited data or complex geometries. They can handle inverse problems and parameter identification naturally. PINNs are mesh-free, making them suitable for high-dimensional problems and those with moving boundaries.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_limited_data(func, x_range, num_points):
    x = np.random.uniform(*x_range, num_points)
    y = func(x) + np.random.normal(0, 0.1, num_points)
    return x, y

def true_function(x):
    return np.sin(x) * np.exp(-0.1 * x)

def pinn_prediction(x):
    # Simulated PINN prediction
    return true_function(x) + np.random.normal(0, 0.05, len(x))

# Generate limited data
x_data, y_data = generate_limited_data(true_function, (0, 10), 20)

# Generate points for plotting
x_plot = np.linspace(0, 10, 200)
y_true = true_function(x_plot)
y_pinn = pinn_prediction(x_plot)

plt.figure(figsize=(12, 6))
plt.scatter(x_data, y_data, label='Limited Data', color='red')
plt.plot(x_plot, y_true, label='True Function', color='blue')
plt.plot(x_plot, y_pinn, label='PINN Prediction', color='green', linestyle='--')
plt.title('PINN Performance with Limited Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

Slide 9: Limitations of PINNs

PINNs can be challenging to train, especially for highly nonlinear problems. They may struggle with enforcing hard constraints and boundary conditions. The choice of network architecture and loss function weights can significantly impact performance.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_pinn_training(epochs, initial_error):
    error = initial_error
    errors = [error]
    
    for _ in range(epochs):
        decrease = np.random.uniform(0, 0.1)
        fluctuation = np.random.normal(0, 0.05)
        error = max(0, error - decrease + fluctuation)
        errors.append(error)
    
    return errors

epochs = 1000
initial_error = 1.0

errors = simulate_pinn_training(epochs, initial_error)

plt.figure(figsize=(12, 6))
plt.plot(range(epochs + 1), errors)
plt.title('Simulated PINN Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 10: Comparison of FEM and PINN

FEM and PINN have distinct strengths and weaknesses. FEM is well-established and reliable for a wide range of problems, while PINNs offer flexibility and potential for solving complex, high-dimensional problems with limited data. The choice between them depends on the specific problem and available resources.

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_methods(problem_complexity, data_availability):
    fem_score = 10 - problem_complexity + 0.5 * data_availability
    pinn_score = 5 + 0.5 * problem_complexity - 0.3 * data_availability
    return fem_score, pinn_score

complexities = np.linspace(1, 10, 50)
data_avail = 5  # Fixed data availability

fem_scores = []
pinn_scores = []

for complexity in complexities:
    fem, pinn = compare_methods(complexity, data_avail)
    fem_scores.append(fem)
    pinn_scores.append(pinn)

plt.figure(figsize=(12, 6))
plt.plot(complexities, fem_scores, label='FEM')
plt.plot(complexities, pinn_scores, label='PINN')
plt.title('FEM vs PINN Performance with Increasing Problem Complexity')
plt.xlabel('Problem Complexity')
plt.ylabel('Performance Score')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Real-Life Example: Heat Transfer in a Cooling Fin

Heat transfer in a cooling fin is a practical problem that both FEM and PINN can solve. FEM discretizes the fin into elements, while PINN learns the temperature distribution as a continuous function. Let's compare their approaches.

```python
import numpy as np
import matplotlib.pyplot as plt

def fem_fin_heat(L, k, h, T_base, T_inf, n):
    dx = L / n
    x = np.linspace(0, L, n+1)
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    
    for i in range(1, n):
        A[i, i-1:i+2] = [1, -2 - (h*dx**2)/(k), 1]
        b[i] = -(h*dx**2*T_inf)/k
    
    A[0, 0] = 1
    b[0] = T_base
    A[-1, -1] = 1
    A[-1, -2] = -1
    
    T = np.linalg.solve(A, b)
    return x, T

def pinn_fin_heat(x, L, k, h, T_base, T_inf):
    m = np.sqrt(h / (k * 0.01))
    return (T_base - T_inf) * np.cosh(m*(L-x)) / np.cosh(m*L) + T_inf

# Parameters
L, k, h = 0.1, 200, 100
T_base, T_inf = 100, 25

x_fem, T_fem = fem_fin_heat(L, k, h, T_base, T_inf, 100)
x_pinn = np.linspace(0, L, 100)
T_pinn = pinn_fin_heat(x_pinn, L, k, h, T_base, T_inf)

plt.figure(figsize=(10, 6))
plt.plot(x_fem, T_fem, label='FEM')
plt.plot(x_pinn, T_pinn, label='PINN', linestyle='--')
plt.title('Temperature Distribution in a Cooling Fin')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Real-Life Example: Fluid Flow Around an Airfoil

Analyzing fluid flow around an airfoil is crucial in aerodynamics. FEM and PINN can both model this complex problem, each with its unique approach.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_airfoil(n_points):
    theta = np.linspace(0, 2*np.pi, n_points)
    x = np.cos(theta) + 0.1 * np.cos(2*theta)
    y = 0.2 * np.sin(theta)
    return x, y

def simulate_flow(x, y, angle_of_attack):
    u = np.cos(angle_of_attack) * (1 - (y**2 + (x-0.5)**2) / ((x-0.5)**2 + y**2))
    v = -np.sin(angle_of_attack) * (1 + (y**2 + (x-0.5)**2) / ((x-0.5)**2 + y**2))
    return u, v

# Generate airfoil shape
airfoil_x, airfoil_y = generate_airfoil(100)

# Create grid
x = np.linspace(-1, 2, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

# Simulate flow
angle_of_attack = np.radians(10)
U, V = simulate_flow(X, Y, angle_of_attack)

# Plot
plt.figure(figsize=(12, 8))
plt.streamplot(X, Y, U, V, density=1, color='blue', arrowsize=1)
plt.plot(airfoil_x, airfoil_y, 'k-', linewidth=2)
plt.title('Simulated Fluid Flow Around an Airfoil')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 13: Hybrid Approaches: Combining FEM and PINN

Recent research explores hybrid approaches that combine the strengths of FEM and PINN. These methods aim to leverage FEM's reliability with PINN's flexibility, potentially offering superior performance for complex problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def fem_solution(x, problem_complexity):
    return np.sin(problem_complexity * x) + 0.1 * np.random.randn(len(x))

def pinn_solution(x, problem_complexity):
    return np.sin(problem_complexity * x) + 0.05 * np.cos(5 * x)

def hybrid_solution(x, problem_complexity, alpha):
    fem = fem_solution(x, problem_complexity)
    pinn = pinn_solution(x, problem_complexity)
    return alpha * fem + (1 - alpha) * pinn

x = np.linspace(0, 1, 100)
problem_complexity = 5

fem = fem_solution(x, problem_complexity)
pinn = pinn_solution(x, problem_complexity)
hybrid = hybrid_solution(x, problem_complexity, 0.7)

plt.figure(figsize=(12, 6))
plt.plot(x, fem, label='FEM', alpha=0.7)
plt.plot(x, pinn, label='PINN', alpha=0.7)
plt.plot(x, hybrid, label='Hybrid', linewidth=2)
plt.title('Comparison of FEM, PINN, and Hybrid Solutions')
plt.xlabel('x')
plt.ylabel('Solution')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 14: Future Directions and Challenges

The field of computational physics is evolving rapidly, with both FEM and PINN contributing to advancements. Future research may focus on improving PINN's reliability, enhancing FEM's efficiency for large-scale problems, and developing more sophisticated hybrid methods.

```python
import numpy as np
import matplotlib.pyplot as plt

def project_growth(initial_value, growth_rate, years):
    return initial_value * (1 + growth_rate) ** years

years = np.arange(2024, 2034)
fem_growth = project_growth(100, 0.05, years - 2024)
pinn_growth = project_growth(20, 0.25, years - 2024)
hybrid_growth = project_growth(10, 0.35, years - 2024)

plt.figure(figsize=(12, 6))
plt.plot(years, fem_growth, label='FEM', marker='o')
plt.plot(years, pinn_growth, label='PINN', marker='s')
plt.plot(years, hybrid_growth, label='Hybrid Methods', marker='^')
plt.title('Projected Growth in Research Publications')
plt.xlabel('Year')
plt.ylabel('Number of Publications (Normalized)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into FEM and PINN, here are some valuable resources:

1. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" by M. Raissi, P. Perdikaris, and G.E. Karniadakis (2019). Available at: [https://arxiv.org/abs/1711.10561](https://arxiv.org/abs/1711.10561)
2. "A comprehensive review of deep learning applications in hydrology and water resources" by Q. Shen (2018). Available at: [https://arxiv.org/abs/1807.05099](https://arxiv.org/abs/1807.05099)
3. "Neural Ordinary Differential Equations" by R.T.Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud (2018). Available at: [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)

These papers provide in-depth discussions on the theory and applications of PINNs and related techniques in various scientific domains.

