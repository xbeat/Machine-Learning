## Optimization Algorithms Beyond Gradient Descent
Slide 1: Newton's Method - Beyond Gradient Descent

Newton's method is an advanced optimization algorithm that utilizes second-order derivatives to find optimal parameters more efficiently than gradient descent. It approximates the objective function locally using a quadratic function and finds its minimum analytically.

```python
# Implementation of Newton's Method Optimization
import numpy as np

def newton_optimize(f, df, d2f, x0, tol=1e-6, max_iter=100):
    x = x0
    
    for i in range(max_iter):
        # Calculate gradient and Hessian
        grad = df(x)
        hess = d2f(x)
        
        # Newton step
        delta = -np.linalg.solve(hess, grad)
        x_new = x + delta
        
        # Check convergence
        if np.linalg.norm(delta) < tol:
            return x_new, i
        x = x_new
    
    return x, max_iter
```

Slide 2: Mathematical Foundation of Newton's Method

The core principle behind Newton's method lies in the second-order Taylor expansion of the objective function around the current point. This leads to more accurate local approximations and faster convergence compared to first-order methods.

```python
# Mathematical representation in code format
"""
$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x$$

$$\Delta x = -H(x)^{-1} \nabla f(x)$$

Where:
- f(x) is the objective function
- ∇f(x) is the gradient
- H(x) is the Hessian matrix
"""
```

Slide 3: Simple Quadratic Optimization Example

Let's implement Newton's method for optimizing a simple quadratic function to demonstrate its rapid convergence properties and compare it with traditional gradient descent approaches in terms of iterations required.

```python
import numpy as np
import matplotlib.pyplot as plt

def quadratic(x):
    return x[0]**2 + 2*x[1]**2

def grad_quadratic(x):
    return np.array([2*x[0], 4*x[1]])

def hess_quadratic(x):
    return np.array([[2, 0], [0, 4]])

# Initial point
x0 = np.array([1.0, 1.0])

# Optimize
result, iterations = newton_optimize(quadratic, grad_quadratic, 
                                  hess_quadratic, x0)

print(f"Optimum found at: {result}")
print(f"Iterations needed: {iterations}")
```

Slide 4: Line Search Enhancement

Line search methods improve Newton's method by adaptively selecting step sizes, ensuring better convergence properties and preventing overshooting in regions where the quadratic approximation might be poor.

```python
def newton_with_line_search(f, df, d2f, x0, alpha=0.5, beta=0.8, 
                          max_iter=100):
    x = x0
    
    for i in range(max_iter):
        grad = df(x)
        hess = d2f(x)
        
        # Compute Newton direction
        delta = -np.linalg.solve(hess, grad)
        
        # Backtracking line search
        t = 1.0
        while f(x + t*delta) > f(x) + alpha*t*grad.dot(delta):
            t *= beta
            
        x = x + t*delta
        
        if np.linalg.norm(grad) < 1e-6:
            return x, i
    
    return x, max_iter
```

Slide 5: Handling Non-Positive Definite Hessians

Real-world optimization problems often involve non-positive definite Hessian matrices, requiring modification to ensure Newton's method remains stable and convergent throughout the optimization process.

```python
def modified_newton(f, df, d2f, x0, tol=1e-6, max_iter=100):
    x = x0
    
    for i in range(max_iter):
        grad = df(x)
        hess = d2f(x)
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvals(hess))
        if min_eig < 0:
            hess += (-min_eig + 0.1) * np.eye(len(x0))
            
        delta = -np.linalg.solve(hess, grad)
        x = x + delta
        
        if np.linalg.norm(grad) < tol:
            return x, i
            
    return x, max_iter
```

Slide 6: Real-world Application - Portfolio Optimization

Newton's method excels in portfolio optimization problems where we need to find optimal asset weights that minimize risk while maximizing expected returns, considering both the covariance matrix and expected returns vector.

```python
import numpy as np
from scipy.optimize import minimize

def portfolio_objective(weights, returns, cov_matrix, risk_aversion=1):
    portfolio_return = np.sum(returns * weights)
    portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
    return -portfolio_return + risk_aversion * portfolio_risk

# Sample data
n_assets = 4
returns = np.array([0.1, 0.15, 0.12, 0.09])
cov_matrix = np.array([[0.04, 0.02, 0.01, 0.02],
                       [0.02, 0.05, 0.02, 0.01],
                       [0.01, 0.02, 0.03, 0.015],
                       [0.02, 0.01, 0.015, 0.035]])

# Constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(n_assets))

# Initial weights
initial_weights = np.array([1/n_assets] * n_assets)

# Optimize using Newton-CG method
result = minimize(portfolio_objective, initial_weights,
                 args=(returns, cov_matrix),
                 method='Newton-CG',
                 jac=True,
                 constraints=constraints,
                 bounds=bounds)

print("Optimal portfolio weights:", result.x)
```

Slide 7: Solving Nonlinear Least Squares Problems

Newton's method is particularly effective for solving nonlinear least squares problems, where the objective function is a sum of squared residuals. This makes it ideal for curve fitting and parameter estimation tasks.

```python
def nonlinear_least_squares(x_data, y_data, model, params0):
    def residuals(params):
        return y_data - model(x_data, params)
        
    def objective(params):
        r = residuals(params)
        return 0.5 * np.sum(r**2)
        
    def jacobian(params):
        eps = 1e-8
        jac = np.zeros((len(params), len(x_data)))
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            jac[i] = (model(x_data, params_plus) - 
                     model(x_data, params)) / eps
        return -jac.T
        
    # Newton iterations
    params = params0
    for _ in range(50):
        r = residuals(params)
        J = jacobian(params)
        H = J.T @ J
        g = J.T @ r
        params = params - np.linalg.solve(H, g)
        
    return params
```

Slide 8: Trust Region Methods

Trust region methods enhance Newton's method by constraining the optimization step within a region where the quadratic approximation is trusted to be accurate, providing better convergence guarantees for difficult problems.

```python
def trust_region_newton(f, df, d2f, x0, radius=1.0, eta=0.1):
    x = x0
    n = len(x0)
    
    def solve_trust_region_subproblem(g, H, radius):
        # Solve the trust region subproblem using Steihaug-CG method
        p = np.zeros(n)
        r = -g
        d = r.copy()
        
        for _ in range(n):
            Hd = H @ d
            dHd = d @ Hd
            
            if dHd <= 0:
                # Find the boundary solution
                a = d @ d
                b = 2 * (p @ d)
                c = (p @ p) - radius**2
                tau = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                return p + tau * d
                
            alpha = (r @ r) / dHd
            p_new = p + alpha * d
            
            if np.linalg.norm(p_new) >= radius:
                # Find the boundary solution
                a = d @ d
                b = 2 * (p @ d)
                c = (p @ p) - radius**2
                tau = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
                return p + tau * d
                
            r_new = r - alpha * Hd
            beta = (r_new @ r_new) / (r @ r)
            d = r_new + beta * d
            p = p_new
            r = r_new
            
        return p
    
    for _ in range(100):
        g = df(x)
        H = d2f(x)
        
        # Solve trust region subproblem
        p = solve_trust_region_subproblem(g, H, radius)
        
        # Compute actual vs predicted reduction
        actual_red = f(x) - f(x + p)
        pred_red = -(g @ p + 0.5 * p @ H @ p)
        
        rho = actual_red / pred_red
        
        # Update trust region radius
        if rho < 0.25:
            radius *= 0.25
        elif rho > 0.75 and np.linalg.norm(p) == radius:
            radius = min(2.0 * radius, 10.0)
            
        # Update point
        if rho > eta:
            x = x + p
            
        if np.linalg.norm(g) < 1e-6:
            break
            
    return x
```

Slide 9: Quasi-Newton Methods Implementation

Quasi-Newton methods approximate the Hessian matrix using gradient information, reducing computational cost while maintaining superlinear convergence. The BFGS method is one of the most successful variants.

```python
def bfgs_optimize(f, df, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    x = x0
    H = np.eye(n)  # Initial Hessian approximation
    
    for i in range(max_iter):
        g = df(x)
        if np.linalg.norm(g) < tol:
            break
            
        # Search direction
        p = -H @ g
        
        # Line search
        alpha = 1.0
        while f(x + alpha*p) > f(x) + 0.1*alpha*g@p:
            alpha *= 0.5
            
        # Update position
        x_new = x + alpha*p
        
        # BFGS update
        s = x_new - x
        y = df(x_new) - g
        
        rho = 1.0 / (y@s)
        H = (np.eye(n) - rho*np.outer(s, y)) @ H @ \
            (np.eye(n) - rho*np.outer(y, s)) + rho*np.outer(s, s)
            
        x = x_new
        
    return x, i+1

# Example usage
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

x0 = np.array([-1.0, 1.0])
result, iterations = bfgs_optimize(rosenbrock, rosenbrock_grad, x0)
print(f"Minimum found at {result} after {iterations} iterations")
```

Slide 10: Results for Portfolio Optimization

Demonstrating the practical application of Newton's method in the context of the portfolio optimization problem from Slide 6.

```python
# Extended results analysis
def analyze_portfolio(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns * weights)
    portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe_ratio = portfolio_return / portfolio_risk
    
    print("Portfolio Analysis:")
    print("-----------------")
    print(f"Expected Return: {portfolio_return:.4f}")
    print(f"Portfolio Risk: {portfolio_risk:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print("\nAsset Weights:")
    for i, w in enumerate(weights):
        print(f"Asset {i+1}: {w:.4f}")
        
# Sample execution with realistic data
returns = np.array([0.12, 0.15, 0.10, 0.13])
cov_matrix = np.array([
    [0.040, 0.012, 0.015, 0.010],
    [0.012, 0.035, 0.010, 0.014],
    [0.015, 0.010, 0.045, 0.012],
    [0.010, 0.014, 0.012, 0.030]
])

result = minimize(portfolio_objective, 
                 np.array([0.25, 0.25, 0.25, 0.25]),
                 args=(returns, cov_matrix),
                 method='Newton-CG',
                 jac=True)

analyze_portfolio(result.x, returns, cov_matrix)
```

Slide 11: Benchmarking Against Gradient Descent

A comprehensive comparison between Newton's Method and Gradient Descent showcasing convergence speed, computational complexity, and accuracy across different optimization scenarios.

```python
import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_optimizers(f, df, d2f, x0, true_minimum):
    # Newton's Method
    start_time = time.time()
    newton_path = []
    x = x0.copy()
    
    for i in range(100):
        newton_path.append(x.copy())
        grad = df(x)
        hess = d2f(x)
        delta = -np.linalg.solve(hess, grad)
        x += delta
        if np.linalg.norm(delta) < 1e-6:
            break
    
    newton_time = time.time() - start_time
    newton_error = np.linalg.norm(x - true_minimum)
    
    # Gradient Descent
    start_time = time.time()
    gd_path = []
    x = x0.copy()
    learning_rate = 0.1
    
    for i in range(1000):
        gd_path.append(x.copy())
        grad = df(x)
        x -= learning_rate * grad
        if np.linalg.norm(grad) < 1e-6:
            break
    
    gd_time = time.time() - start_time
    gd_error = np.linalg.norm(x - true_minimum)
    
    return {
        'newton': {
            'path': np.array(newton_path),
            'time': newton_time,
            'error': newton_error,
            'iterations': len(newton_path)
        },
        'gradient_descent': {
            'path': np.array(gd_path),
            'time': gd_time,
            'error': gd_error,
            'iterations': len(gd_path)
        }
    }

# Example usage with quadratic function
def quad_function(x):
    return x[0]**2 + 2*x[1]**2

def quad_gradient(x):
    return np.array([2*x[0], 4*x[1]])

def quad_hessian(x):
    return np.array([[2, 0], [0, 4]])

# Run benchmark
x0 = np.array([2.0, 2.0])
true_min = np.array([0.0, 0.0])
results = benchmark_optimizers(quad_function, quad_gradient, 
                             quad_hessian, x0, true_min)

print("Benchmark Results:")
print("Newton's Method:")
print(f"Time: {results['newton']['time']:.6f} seconds")
print(f"Error: {results['newton']['error']:.6f}")
print(f"Iterations: {results['newton']['iterations']}")
print("\nGradient Descent:")
print(f"Time: {results['gradient_descent']['time']:.6f} seconds")
print(f"Error: {results['gradient_descent']['error']:.6f}")
print(f"Iterations: {results['gradient_descent']['iterations']}")
```

Slide 12: Advanced Applications in Neural Networks

Newton's method can be adapted for training neural networks, particularly in scenarios where second-order information can significantly improve convergence and generalization performance.

```python
class NewtonNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.params = np.concatenate([self.W1.flatten(), 
                                    self.W2.flatten()])
        
    def forward(self, X):
        Z1 = X @ self.W1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2
        return Z1, A1, Z2
        
    def loss(self, X, y):
        _, _, Z2 = self.forward(X)
        return 0.5 * np.mean((Z2 - y) ** 2)
        
    def compute_gradients(self, X, y):
        Z1, A1, Z2 = self.forward(X)
        m = X.shape[0]
        
        dZ2 = (Z2 - y) / m
        dW2 = A1.T @ dZ2
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - np.tanh(Z1)**2)
        dW1 = X.T @ dZ1
        
        return np.concatenate([dW1.flatten(), dW2.flatten()])
        
    def compute_hessian(self, X, y, eps=1e-4):
        n_params = len(self.params)
        H = np.zeros((n_params, n_params))
        grad = self.compute_gradients(X, y)
        
        for i in range(n_params):
            params_plus = self.params.copy()
            params_plus[i] += eps
            self.params = params_plus
            grad_plus = self.compute_gradients(X, y)
            
            H[:, i] = (grad_plus - grad) / eps
            self.params = params_plus - eps
            
        return (H + H.T) / 2  # Ensure symmetry
        
    def newton_step(self, X, y):
        grad = self.compute_gradients(X, y)
        hess = self.compute_hessian(X, y)
        
        # Add regularization to ensure positive definiteness
        hess += 1e-4 * np.eye(len(self.params))
        
        delta = np.linalg.solve(hess, grad)
        self.params -= delta
        
        # Reshape parameters back to weights
        split_idx = self.W1.size
        self.W1 = self.params[:split_idx].reshape(self.W1.shape)
        self.W2 = self.params[split_idx:].reshape(self.W2.shape)
```

Slide 13: Additional Resources

*   "A Comprehensive Study of Newton-Type Methods in Machine Learning"
    *   Search on Google Scholar: "Newton methods machine learning optimization comprehensive review"
*   "Trust Region Methods for Large-Scale Optimization"
    *   [https://arxiv.org/abs/1804.06218](https://arxiv.org/abs/1804.06218)
*   "Quasi-Newton Methods for Deep Learning: Forget the Past, Just Sample"
    *   [https://arxiv.org/abs/1901.09997](https://arxiv.org/abs/1901.09997)
*   "Second-Order Optimization for Neural Networks"
    *   Search on Google Scholar: "second order optimization neural networks survey"
*   "On the Convergence of Newton-Type Methods in Deep Learning"
    *   Search "Newton methods convergence deep learning" on academic repositories

