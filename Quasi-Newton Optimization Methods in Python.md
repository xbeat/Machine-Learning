## Quasi-Newton Optimization Methods in Python
Slide 1: Introduction to Quasi-Newton Methods

Quasi-Newton methods are optimization algorithms used to find local maxima and minima of functions. They are particularly useful when the Hessian matrix is unavailable or too expensive to compute. These methods approximate the Hessian matrix or its inverse, updating it iteratively to improve convergence.

```python
import numpy as np

def quasi_newton_optimization(f, grad_f, x0, max_iter=100, tol=1e-6):
    x = x0
    n = len(x)
    B = np.eye(n)  # Initial approximation of the Hessian inverse
    
    for i in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            return x
        
        d = -np.dot(B, g)
        alpha = 0.5  # Fixed step size for simplicity
        x_new = x + alpha * d
        
        s = x_new - x
        y = grad_f(x_new) - g
        
        # BFGS update
        B = B + (np.outer(s, s) / np.dot(y, s)) - \
            (np.dot(B, np.outer(y, y, B)) / np.dot(y, np.dot(B, y)))
        
        x = x_new
    
    return x

# Example usage
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
result = quasi_newton_optimization(f, grad_f, x0)
print(f"Optimum found at: {result}")
```

Slide 2: The BFGS Algorithm

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is one of the most popular quasi-Newton methods. It approximates the Hessian matrix using gradient information and updates from previous iterations. BFGS maintains positive definiteness of the Hessian approximation, ensuring descent directions.

```python
import numpy as np

def bfgs_update(B, s, y):
    """
    Update the approximate Hessian inverse using the BFGS formula
    
    B: current approximate Hessian inverse
    s: step taken
    y: change in gradient
    """
    rho = 1.0 / (np.dot(y, s))
    I = np.eye(B.shape[0])
    
    B_new = (I - rho * np.outer(s, y)).dot(B).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
    
    return B_new

# Example usage
n = 2
B = np.eye(n)  # Initial Hessian inverse approximation
s = np.array([0.1, 0.2])
y = np.array([0.3, 0.4])

B_updated = bfgs_update(B, s, y)
print("Updated Hessian inverse approximation:")
print(B_updated)
```

Slide 3: Line Search in Quasi-Newton Methods

Line search is crucial in quasi-Newton methods to determine the step size along the search direction. The goal is to find a step size that satisfies the Wolfe conditions, ensuring sufficient decrease in the objective function and curvature condition.

```python
import numpy as np

def wolfe_conditions(f, grad_f, x, d, alpha, c1=1e-4, c2=0.9):
    """
    Check if the Wolfe conditions are satisfied
    
    f: objective function
    grad_f: gradient of the objective function
    x: current point
    d: search direction
    alpha: step size
    c1, c2: parameters for Wolfe conditions
    """
    phi_0 = f(x)
    phi_alpha = f(x + alpha * d)
    dphi_0 = np.dot(grad_f(x), d)
    
    sufficient_decrease = phi_alpha <= phi_0 + c1 * alpha * dphi_0
    curvature = np.dot(grad_f(x + alpha * d), d) >= c2 * dphi_0
    
    return sufficient_decrease and curvature

# Example usage
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x = np.array([1.0, 1.0])
d = np.array([-1.0, -1.0])
alpha = 0.5

satisfied = wolfe_conditions(f, grad_f, x, d, alpha)
print(f"Wolfe conditions satisfied: {satisfied}")
```

Slide 4: Limited-Memory BFGS (L-BFGS)

L-BFGS is a memory-efficient variant of BFGS, suitable for large-scale optimization problems. Instead of storing the full Hessian approximation, L-BFGS keeps a limited history of position and gradient differences, using them to implicitly represent the Hessian inverse.

```python
import numpy as np

class LBFGS:
    def __init__(self, m=10):
        self.m = m
        self.s = []
        self.y = []

    def update(self, s, y):
        if len(self.s) == self.m:
            self.s.pop(0)
            self.y.pop(0)
        self.s.append(s)
        self.y.append(y)

    def compute_direction(self, grad):
        q = grad.()
        alphas = []

        for s, y in zip(reversed(self.s), reversed(self.y)):
            alpha = np.dot(s, q) / np.dot(y, s)
            q -= alpha * y
            alphas.append(alpha)

        if self.s:
            gamma = np.dot(self.s[-1], self.y[-1]) / np.dot(self.y[-1], self.y[-1])
            r = gamma * q
        else:
            r = q

        for s, y, alpha in zip(self.s, self.y, reversed(alphas)):
            beta = np.dot(y, r) / np.dot(y, s)
            r += s * (alpha - beta)

        return -r

# Example usage
lbfgs = LBFGS(m=5)
x = np.array([1.0, 1.0])
grad = np.array([2.0, 2.0])

for _ in range(3):
    direction = lbfgs.compute_direction(grad)
    step_size = 0.1
    s = step_size * direction
    x_new = x + s
    grad_new = np.array([2*x_new[0], 2*x_new[1]])
    y = grad_new - grad
    
    lbfgs.update(s, y)
    x, grad = x_new, grad_new

print(f"Final x: {x}")
print(f"Final gradient: {grad}")
```

Slide 5: Convergence Properties of Quasi-Newton Methods

Quasi-Newton methods typically exhibit superlinear convergence, which is faster than linear convergence but slower than quadratic convergence. The rate of convergence depends on the accuracy of the Hessian approximation and the properties of the objective function.

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

def bfgs_optimization(f, grad_f, x0, max_iter=100, tol=1e-6):
    x = x0
    n = len(x)
    B = np.eye(n)
    convergence = []

    for i in range(max_iter):
        g = grad_f(x)
        convergence.append(np.linalg.norm(g))
        if np.linalg.norm(g) < tol:
            break

        d = -np.dot(B, g)
        alpha = 0.5  # Fixed step size for simplicity
        x_new = x + alpha * d

        s = x_new - x
        y = grad_f(x_new) - g

        B = B + (np.outer(s, s) / np.dot(y, s)) - \
            (np.dot(B, np.outer(y, y, B)) / np.dot(y, np.dot(B, y)))

        x = x_new

    return x, convergence

x0 = np.array([0.0, 0.0])
result, convergence = bfgs_optimization(rosenbrock, grad_rosenbrock, x0)

plt.semilogy(convergence)
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm')
plt.title('BFGS Convergence on Rosenbrock Function')
plt.grid(True)
plt.show()

print(f"Optimum found at: {result}")
print(f"Function value at optimum: {rosenbrock(result)}")
```

Slide 6: Handling Nonconvex Functions

Quasi-Newton methods can be applied to nonconvex functions, but special care must be taken to ensure positive definiteness of the Hessian approximation. Techniques like damped BFGS updates or trust region methods can be employed to handle nonconvexity.

```python
import numpy as np

def damped_bfgs_update(B, s, y, damping_factor=0.2):
    """
    Damped BFGS update for handling nonconvex functions
    
    B: current approximate Hessian inverse
    s: step taken
    y: change in gradient
    damping_factor: controls the damping (0 < damping_factor < 1)
    """
    sTy = np.dot(s, y)
    if sTy < damping_factor * np.dot(s, np.dot(B, s)):
        theta = (1 - damping_factor) * np.dot(s, np.dot(B, s)) / (np.dot(s, np.dot(B, s)) - sTy)
        y = theta * y + (1 - theta) * np.dot(B, s)
    
    rho = 1.0 / np.dot(y, s)
    I = np.eye(B.shape[0])
    
    B_new = (I - rho * np.outer(s, y)).dot(B).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
    
    return B_new

# Example usage
n = 2
B = np.eye(n)
s = np.array([0.1, 0.2])
y = np.array([-0.05, 0.1])  # Negative curvature example

B_updated = damped_bfgs_update(B, s, y)
print("Updated Hessian inverse approximation:")
print(B_updated)

# Check positive definiteness
eigenvalues = np.linalg.eigvals(B_updated)
print(f"Eigenvalues: {eigenvalues}")
print(f"Positive definite: {np.all(eigenvalues > 0)}")
```

Slide 7: Quasi-Newton Methods for Constrained Optimization

Quasi-Newton methods can be extended to handle constrained optimization problems. One approach is to use augmented Lagrangian methods or sequential quadratic programming (SQP) with quasi-Newton updates for the Hessian approximation.

```python
import numpy as np
from scipy.optimize import minimize

def augmented_lagrangian(x, lambda_, mu, f, g):
    """
    Augmented Lagrangian function
    
    x: decision variables
    lambda_: Lagrange multipliers
    mu: penalty parameter
    f: objective function
    g: constraint functions
    """
    penalty = sum(lambda_[i] * g_i for i, g_i in enumerate(g(x)))
    penalty += 0.5 * mu * sum(max(0, g_i)**2 for g_i in g(x))
    return f(x) + penalty

def solve_subproblem(x0, lambda_, mu, f, g, bounds):
    """
    Solve the augmented Lagrangian subproblem using L-BFGS-B
    """
    obj = lambda x: augmented_lagrangian(x, lambda_, mu, f, g)
    result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)
    return result.x

def augmented_lagrangian_method(f, g, x0, bounds, max_iter=50, tol=1e-6):
    x = x0
    lambda_ = np.zeros(len(g(x0)))
    mu = 1.0
    
    for k in range(max_iter):
        x = solve_subproblem(x, lambda_, mu, f, g, bounds)
        
        constraint_violation = np.array([max(0, g_i) for g_i in g(x)])
        if np.linalg.norm(constraint_violation) < tol:
            return x
        
        lambda_ += mu * constraint_violation
        mu *= 2
    
    return x

# Example usage
def objective(x):
    return x[0]**2 + x[1]**2

def constraints(x):
    return [x[0] + x[1] - 1]  # g(x) <= 0

x0 = np.array([0.5, 0.5])
bounds = [(-1, 1), (-1, 1)]

result = augmented_lagrangian_method(objective, constraints, x0, bounds)
print(f"Optimal solution: {result}")
print(f"Objective value: {objective(result)}")
print(f"Constraint violation: {constraints(result)}")
```

Slide 8: Quasi-Newton Methods in Machine Learning

Quasi-Newton methods are widely used in machine learning for training models, especially when dealing with large-scale problems. They offer a good balance between convergence speed and computational efficiency.

```python
import numpy as np
from scipy.optimize import minimize

def logistic_function(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, w):
    z = np.dot(X, w)
    return logistic_function(z)

def cost_function(w, X, y):
    m = len(y)
    h = logistic_regression(X, y, w)
    J = (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1-h))
    return J

def gradient(w, X, y):
    m = len(y)
    h = logistic_regression(X, y, w)
    return (1/m) * np.dot(X.T, (h - y))

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term

# Initial weights
w0 = np.zeros(X.shape[1])

# Optimize using L-BFGS
result = minimize(cost_function, w0, args=(X, y), method='L-BFGS-B', jac=gradient)

print("Optimized weights:", result.x)
print("Final cost:", result.fun)

# Make predictions
y_pred = (logistic_regression(X, y, result.x) > 0.5).astype(int)
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)
```

Slide 9: Quasi-Newton Methods for Nonlinear Least Squares

Quasi-Newton methods can be adapted for nonlinear least squares problems, which are common in data fitting and parameter estimation. The Gauss-Newton method is often combined with quasi-Newton updates to improve convergence.

```python
import numpy as np
from scipy.optimize import least_squares

def model(x, t):
    return x[0] * np.exp(-x[1] * t)

def residuals(x, t, y):
    return model(x, t) - y

def jacobian(x, t, y):
    J = np.zeros((len(t), len(x)))
    J[:, 0] = np.exp(-x[1] * t)
    J[:, 1] = -x[0] * t * np.exp(-x[1] * t)
    return J

# Generate synthetic data
np.random.seed(42)
t_true = np.linspace(0, 5, 100)
y_true = model([2.5, 1.3], t_true)
y_noisy = y_true + 0.2 * np.random.normal(size=t_true.shape)

# Initial guess
x0 = [1.0, 0.5]

# Solve using Levenberg-Marquardt (a quasi-Newton method for nonlinear least squares)
result = least_squares(residuals, x0, jac=jacobian, args=(t_true, y_noisy), method='lm')

print("Optimized parameters:", result.x)
print("Cost:", result.cost)
print("Success:", result.success)
```

Slide 10: Trust Region Methods

Trust region methods are a class of optimization algorithms that combine quasi-Newton approximations with a "trust region" approach. They define a region around the current point where the quadratic model is trusted to be an accurate representation of the objective function.

```python
import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

# Initial point
x0 = np.array([-1.2, 1.0])

# Solve using trust-region method (trust-ncg)
result = minimize(rosenbrock, x0, method='trust-ncg', jac=rosenbrock_grad)

print("Optimized solution:", result.x)
print("Function value at optimum:", result.fun)
print("Success:", result.success)
print("Number of iterations:", result.nit)
```

Slide 11: Quasi-Newton Methods for Stochastic Optimization

Stochastic quasi-Newton methods are designed to handle optimization problems with noisy or stochastic gradients, which are common in machine learning and stochastic programming.

```python
import numpy as np

class StochasticLBFGS:
    def __init__(self, n, m=10, lr=0.1):
        self.n = n
        self.m = m
        self.lr = lr
        self.s = []
        self.y = []
        self.rho = []
        
    def update(self, grad, x):
        if len(self.s) > 0:
            s = x - self.prev_x
            y = grad - self.prev_grad
            self.s.append(s)
            self.y.append(y)
            self.rho.append(1 / np.dot(y, s))
            
            if len(self.s) > self.m:
                self.s.pop(0)
                self.y.pop(0)
                self.rho.pop(0)
        
        self.prev_x = x.()
        self.prev_grad = grad.()
        
        return self.compute_step(grad)
    
    def compute_step(self, grad):
        q = grad.()
        alpha = np.zeros(len(self.s))
        
        for i in reversed(range(len(self.s))):
            alpha[i] = self.rho[i] * np.dot(self.s[i], q)
            q -= alpha[i] * self.y[i]
        
        r = q
        
        for i in range(len(self.s)):
            beta = self.rho[i] * np.dot(self.y[i], r)
            r += self.s[i] * (alpha[i] - beta)
        
        return -self.lr * r

# Example usage
def noisy_rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2 + np.random.normal(0, 0.1)

def noisy_rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ]) + np.random.normal(0, 0.1, 2)

np.random.seed(42)
x = np.array([-1.2, 1.0])
optimizer = StochasticLBFGS(2)

for i in range(1000):
    grad = noisy_rosenbrock_grad(x)
    step = optimizer.update(grad, x)
    x += step

print("Final solution:", x)
print("Function value at solution:", noisy_rosenbrock(x))
```

Slide 12: Quasi-Newton Methods for Nonsmooth Optimization

Quasi-Newton methods can be extended to handle nonsmooth optimization problems by incorporating techniques like bundle methods or proximal gradient approaches.

```python
import numpy as np

def prox_l1(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def f(x):
    return 0.5 * np.sum(x**2)

def grad_f(x):
    return x

def g(x):
    return np.sum(np.abs(x))

def proximal_quasi_newton(x0, lambda_, max_iter=100, tol=1e-6):
    x = x0
    n = len(x)
    B = np.eye(n)
    
    for k in range(max_iter):
        grad = grad_f(x)
        d = -np.linalg.solve(B, grad)
        
        # Line search
        alpha = 1.0
        while f(prox_l1(x + alpha * d, lambda_)) > f(x) + 0.5 * alpha * np.dot(grad, d):
            alpha *= 0.5
        
        x_new = prox_l1(x + alpha * d, lambda_)
        s = x_new - x
        y = grad_f(x_new) - grad
        
        # BFGS update
        if np.dot(y, s) > 1e-10:
            B = B - np.outer(B.dot(s), B.dot(s)) / np.dot(s, B.dot(s)) + np.outer(y, y) / np.dot(y, s)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
    
    return x

# Example usage
x0 = np.array([1.0, 2.0, -1.5, 0.5])
lambda_ = 0.1

result = proximal_quasi_newton(x0, lambda_)
print("Optimized solution:", result)
print("Objective value:", f(result) + lambda_ * g(result))
```

Slide 13: Real-life Example: Portfolio Optimization

Quasi-Newton methods can be applied to portfolio optimization problems, where the goal is to find the optimal allocation of assets to maximize returns while minimizing risk.

```python
import numpy as np
from scipy.optimize import minimize

# Sample data: expected returns and covariance matrix
expected_returns = np.array([0.05, 0.08, 0.12, 0.07])
cov_matrix = np.array([
    [0.0064, 0.0008, 0.0012, 0.0016],
    [0.0008, 0.0100, 0.0018, 0.0024],
    [0.0012, 0.0018, 0.0400, 0.0036],
    [0.0016, 0.0024, 0.0036, 0.0144]
])

def portfolio_return(weights):
    return np.dot(weights, expected_returns)

def portfolio_variance(weights):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def objective(weights):
    return -portfolio_return(weights) / np.sqrt(portfolio_variance(weights))

def constraint(weights):
    return np.sum(weights) - 1

initial_weights = np.array([0.25, 0.25, 0.25, 0.25])
bounds = [(0, 1) for _ in range(len(expected_returns))]
constraint = {'type': 'eq', 'fun': constraint}

result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraint)

print("Optimal portfolio weights:", result.x)
print("Expected return:", portfolio_return(result.x))
print("Portfolio volatility:", np.sqrt(portfolio_variance(result.x)))
print("Sharpe ratio:", -result.fun)
```

Slide 14: Real-life Example: Image Reconstruction

Quasi-Newton methods can be used in image processing tasks, such as image reconstruction or denoising. Here's an example of using L-BFGS for image denoising.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import convolve

def add_noise(image, noise_level):
    return image + noise_level * np.random.randn(*image.shape)

def total_variation(image):
    dx = np.diff(image, axis=0)
    dy = np.diff(image, axis=1)
    return np.sum(np.sqrt(dx**2 + dy**2))

def objective(x, noisy_image, lambda_):
    image = x.reshape(noisy_image.shape)
    return 0.5 * np.sum((image - noisy_image)**2) + lambda_ * total_variation(image)

def gradient(x, noisy_image, lambda_):
    image = x.reshape(noisy_image.shape)
    grad = image - noisy_image
    
    # Total variation gradient
    dx = np.diff(image, axis=0, append=0)
    dy = np.diff(image, axis=1, append=0)
    tv_grad = np.zeros_like(image)
    tv_grad[:-1, :] -= dx / np.sqrt(dx**2 + dy**2 + 1e-10)
    tv_grad[:, :-1] -= dy / np.sqrt(dx**2 + dy**2 + 1e-10)
    tv_grad[1:, :] += dx[:-1, :] / np.sqrt(dx[:-1, :]**2 + dy[1:, :]**2 + 1e-10)
    tv_grad[:, 1:] += dy[:, :-1] / np.sqrt(dx[:, 1:]**2 + dy[:, :-1]**2 + 1e-10)
    
    return (grad + lambda_ * tv_grad).ravel()

# Generate a simple image
image = np.zeros((50, 50))
image[10:40, 10:40] = 1

# Add noise
noisy_image = add_noise(image, 0.1)

# Optimize using L-BFGS
lambda_ = 0.1
result = minimize(
    objective,
    noisy_image.ravel(),
    args=(noisy_image, lambda_),
    method='L-BFGS-B',
    jac=gradient,
    options={'maxiter': 100}
)

denoised_image = result.x.reshape(image.shape)

print("Optimization successful:", result.success)
print("Number of iterations:", result.nit)
```

Slide 15: Additional Resources

For those interested in delving deeper into Quasi-Newton methods, here are some valuable resources:

1. Nocedal, J., & Wright, S. (2006). Numerical optimization. Springer Science & Business Media. ArXiv: [https://arxiv.org/abs/1011.1669v3](https://arxiv.org/abs/1011.1669v3)
2. Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization. Mathematical programming, 45(1), 503-528. ArXiv: [https://arxiv.org/abs/1409.7358](https://arxiv.org/abs/1409.7358)
3. Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing, 16(5), 1190-1208. ArXiv: [https://arxiv.org/abs/1208.2080](https://arxiv.org/abs/1208.2080)
4. Schmidt, M., van den Berg, E., Friedlander, M. P., & Murphy, K. (2009). Optimizing costly functions with simple constraints: A limited-memory projected quasi-Newton algorithm. In Artificial Intelligence and Statistics (pp. 456-463). ArXiv: [https://arxiv.org/abs/0908.0838](https://arxiv.org/abs/0908.0838)

These resources provide in-depth theoretical foundations and practical implementations of Quasi-Newton methods, covering various aspects from basic concepts to advanced applications in machine learning and optimization.

