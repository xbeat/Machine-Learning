## Second Order Optimization in Python
Slide 1: Introduction to Second Order Optimization

Second order optimization methods utilize information about the curvature of the objective function to improve convergence rates and find better local minima. These methods are particularly useful for problems with complex loss landscapes or ill-conditioned gradients.

```python
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x**4 - 4*x**2 + 2

x = np.linspace(-3, 3, 100)
y = objective_function(x)

plt.plot(x, y)
plt.title('Example of a Non-Convex Objective Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 2: First Order vs. Second Order Methods

First order methods use gradient information to update parameters, while second order methods incorporate the Hessian matrix or its approximations. This additional information allows for more intelligent step sizes and directions.

```python
def gradient_descent(x, learning_rate, num_iterations):
    for _ in range(num_iterations):
        gradient = 4*x**3 - 8*x
        x = x - learning_rate * gradient
    return x

def newton_method(x, num_iterations):
    for _ in range(num_iterations):
        gradient = 4*x**3 - 8*x
        hessian = 12*x**2 - 8
        x = x - gradient / hessian
    return x

x0 = 2.0
print(f"Gradient Descent: {gradient_descent(x0, 0.01, 100)}")
print(f"Newton's Method: {newton_method(x0, 10)}")
```

Slide 3: The Hessian Matrix

The Hessian matrix contains second-order partial derivatives of the objective function. It provides information about the function's local curvature, which can be used to determine step sizes and directions more effectively.

```python
import numpy as np

def hessian(x, y):
    return np.array([
        [2*x*y, x**2 + y**2],
        [x**2 + y**2, 2*x*y]
    ])

x, y = 1, 2
H = hessian(x, y)
print("Hessian matrix:")
print(H)

eigenvalues, eigenvectors = np.linalg.eig(H)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```

Slide 4: Newton's Method

Newton's method uses both the gradient and the Hessian to update parameters. It can converge quadratically near the optimum but may be computationally expensive for high-dimensional problems.

```python
import numpy as np

def newton_optimization(f, grad, hess, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        g = grad(x)
        H = hess(x)
        delta = np.linalg.solve(H, -g)
        x_new = x + delta
        if np.linalg.norm(delta) < tol:
            return x_new, i
        x = x_new
    return x, max_iter

# Example usage
def f(x): return x[0]**2 + 2*x[1]**2
def grad(x): return np.array([2*x[0], 4*x[1]])
def hess(x): return np.array([[2, 0], [0, 4]])

x0 = np.array([1.0, 1.0])
result, iterations = newton_optimization(f, grad, hess, x0)
print(f"Optimum found at {result} after {iterations} iterations")
```

Slide 5: Quasi-Newton Methods: BFGS

BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a popular quasi-Newton method that approximates the Hessian matrix using gradient information. It's more memory-efficient than Newton's method for large-scale problems.

```python
import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                     200*(x[1] - x[0]**2)])

x0 = np.array([0, 0])
result = minimize(rosenbrock, x0, method='BFGS', jac=rosenbrock_grad)

print("Optimization result:")
print(f"x = {result.x}")
print(f"f(x) = {result.fun}")
print(f"Iterations: {result.nit}")
```

Slide 6: Limited-memory BFGS (L-BFGS)

L-BFGS is a memory-efficient variant of BFGS that stores only a limited history of past updates. It's particularly useful for high-dimensional optimization problems where storing the full Hessian approximation is impractical.

```python
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2 + (x[0] + x[1] - 2)**2

def gradient(x):
    return np.array([
        2*x[0] + 2*(x[0] + x[1] - 2),
        2*x[1] + 2*(x[0] + x[1] - 2)
    ])

x0 = np.array([0, 0])
result = minimize(objective, x0, method='L-BFGS-B', jac=gradient)

print("L-BFGS-B Optimization result:")
print(f"x = {result.x}")
print(f"f(x) = {result.fun}")
print(f"Iterations: {result.nit}")
```

Slide 7: Trust Region Methods

Trust region methods define a region around the current point where a quadratic approximation of the objective function is trusted. They can handle non-convex functions and ill-conditioned problems more robustly than line search methods.

```python
from scipy.optimize import minimize

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_grad(x):
    return np.array([
        4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7),
        2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    ])

x0 = np.array([0, 0])
result = minimize(himmelblau, x0, method='trust-ncg', jac=himmelblau_grad)

print("Trust Region Optimization result:")
print(f"x = {result.x}")
print(f"f(x) = {result.fun}")
print(f"Iterations: {result.nit}")
```

Slide 8: Conjugate Gradient Method

The Conjugate Gradient method is an iterative algorithm that can be used for both linear system solving and optimization. It's particularly efficient for large-scale problems with sparse Hessian matrices.

```python
import numpy as np
from scipy.optimize import minimize

def quadratic(x):
    return 0.5 * x.dot(A.dot(x)) - b.dot(x)

def quadratic_grad(x):
    return A.dot(x) - b

n = 1000
A = np.random.rand(n, n)
A = A.T.dot(A)  # Make A positive definite
b = np.random.rand(n)

x0 = np.zeros(n)
result = minimize(quadratic, x0, method='CG', jac=quadratic_grad)

print("Conjugate Gradient Optimization result:")
print(f"Objective value: {result.fun}")
print(f"Iterations: {result.nit}")
```

Slide 9: Levenberg-Marquardt Algorithm

The Levenberg-Marquardt algorithm is a popular method for solving non-linear least squares problems. It interpolates between the Gauss-Newton algorithm and gradient descent, making it more robust than Gauss-Newton.

```python
from scipy.optimize import least_squares
import numpy as np

def model(x, t):
    return x[0] * np.exp(-x[1] * t)

def residuals(x, t, y):
    return model(x, t) - y

# Generate synthetic data
t_data = np.linspace(0, 10, 100)
x_true = [2.5, 0.5]
y_data = model(x_true, t_data) + 0.1 * np.random.randn(len(t_data))

x0 = [1.0, 0.1]  # Initial guess
res = least_squares(residuals, x0, args=(t_data, y_data), method='lm')

print("Levenberg-Marquardt Optimization result:")
print(f"Estimated parameters: {res.x}")
print(f"Cost: {res.cost}")
print(f"Iterations: {res.nfev}")
```

Slide 10: Adaptive Moment Estimation (Adam)

Adam is a popular optimization algorithm that combines ideas from RMSprop and momentum. It adapts the learning rate for each parameter, making it effective for problems with sparse gradients or noisy data.

```python
import numpy as np

def adam(grad, x, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=1000):
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x

# Example usage
def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

x0 = np.array([-1.0, 2.0])
result = adam(rosenbrock_grad, x0)
print(f"Optimum found at {result}")
```

Slide 11: Natural Gradient Descent

Natural Gradient Descent uses the Fisher information matrix to adapt the gradient update. This method is invariant to reparameterization of the model and can be particularly effective for optimizing neural networks.

```python
import numpy as np

def natural_gradient_descent(f, grad, fisher_info, x0, learning_rate=0.1, iterations=100):
    x = x0
    for _ in range(iterations):
        g = grad(x)
        F = fisher_info(x)
        natural_grad = np.linalg.solve(F, g)
        x = x - learning_rate * natural_grad
    return x

# Example: Gaussian distribution parameter estimation
def neg_log_likelihood(theta):
    mu, log_sigma = theta
    return 0.5 * (np.log(2 * np.pi) + 2 * log_sigma + ((data - mu) / np.exp(log_sigma))**2).sum()

def grad_neg_log_likelihood(theta):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    d_mu = ((mu - data) / sigma**2).sum()
    d_log_sigma = (1 - ((data - mu) / sigma)**2).sum()
    return np.array([d_mu, d_log_sigma])

def fisher_info(theta):
    _, log_sigma = theta
    n = len(data)
    sigma2 = np.exp(2 * log_sigma)
    return np.array([[n / sigma2, 0], [0, 2 * n]])

# Generate some data
np.random.seed(0)
data = np.random.normal(0, 1, 1000)

# Optimize
theta0 = np.array([1.0, 0.0])  # Initial guess
result = natural_gradient_descent(neg_log_likelihood, grad_neg_log_likelihood, fisher_info, theta0)
print(f"Estimated parameters: mu = {result[0]:.4f}, sigma = {np.exp(result[1]):.4f}")
```

Slide 12: Real-life Example: Image Classification with Second Order Optimization

In this example, we'll use a second-order optimization method (L-BFGS) to train a simple neural network for image classification on the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(100,), solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
```

Slide 13: Real-life Example: Hyperparameter Optimization

In this example, we'll use Bayesian optimization, which incorporates second-order information, to optimize hyperparameters for a machine learning model.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Load dataset
X, y = load_iris(return_X_y=True)

# Define the search space
space = [Real(1e-6, 1e+1, name='C'),
         Real(1e-6, 1e+1, name='gamma')]

# Define the objective function
@use_named_args(space)
def objective(**params):
    model = SVC(**params)
    return -np.mean(cross_val_score(model, X, y, cv=5, n_jobs=-1))

# Perform Bayesian optimization
result = gp_minimize(objective, space, n_calls=50, random_state=42)

# Print results
print("Best hyperparameters:")
print(f"C: {result.x[0]:.6f}")
print(f"gamma: {result.x[1]:.6f}")
print(f"Best cross-validation score: {-result.fun:.4f}")
```

Slide 14: Future Directions and Advanced Topics

Second-order optimization methods continue to evolve, with ongoing research focusing on improving their efficiency and applicability to large-scale machine learning problems. Some promising areas of development include:

1. Stochastic second-order methods for deep learning
2. Distributed and parallel implementations of second-order algorithms
3. Incorporation of curvature information in reinforcement learning
4. Adaptive preconditioning techniques for ill-conditioned problems

These advancements aim to address the computational challenges of second-order methods while maintaining their superior convergence properties. As hardware capabilities increase and algorithms improve, we can expect to see wider adoption of these powerful optimization techniques in various domains of machine learning and scientific computing.

Slide 15: Future Directions and Advanced Topics

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_optimization_landscape():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1-X)**2 + 100*(Y-X**2)**2  # Rosenbrock function
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, np.log(Z), levels=np.logspace(0, 5, 35))
    plt.colorbar(label='log(f(x, y))')
    plt.title('Optimization Landscape: Rosenbrock Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_optimization_landscape()
```

This visualization demonstrates the complex landscape that optimization algorithms navigate, highlighting the importance of advanced techniques in finding global optima efficiently.


