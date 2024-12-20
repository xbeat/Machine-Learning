## Navigating Local Minima and Saddle Points in Deep Learning
Slide 1: Understanding Local Minima and Saddle Points

Neural networks' loss landscapes are highly complex multidimensional surfaces. Understanding the topology of these surfaces helps grasp optimization challenges. Let's visualize a simple 2D case to demonstrate local minima versus saddle points.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_surface():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a surface with both local minima and saddle points
    Z = X**2 - Y**2 + 2*np.sin(X) + 2*np.cos(Y)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    
    plt.colorbar(surf)
    plt.title('Loss Landscape with Saddle Points')
    plt.show()

plot_surface()
```

Slide 2: Gradient Analysis Near Critical Points

Critical points in loss landscapes can be characterized by analyzing the eigenvalues of the Hessian matrix. At saddle points, the Hessian has both positive and negative eigenvalues, while local minima have all positive eigenvalues.

```python
import numpy as np
from scipy.linalg import eigh

def analyze_critical_point(x, y):
    # Example Hessian computation for f(x,y) = x^2 - y^2
    hessian = np.array([[2, 0],
                        [0, -2]])
    
    eigenvalues, eigenvectors = eigh(hessian)
    
    print(f"Eigenvalues at point ({x}, {y}):")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)
    
    if np.all(eigenvalues > 0):
        return "Local minimum"
    elif np.all(eigenvalues < 0):
        return "Local maximum"
    else:
        return "Saddle point"

# Analyze a saddle point at origin
point_type = analyze_critical_point(0, 0)
print(f"\nPoint type: {point_type}")
```

Slide 3: Implementing Gradient Descent with Momentum

Momentum helps overcome saddle points by accumulating velocity in directions of consistent gradient, enabling faster convergence and better escape from flat regions in the loss landscape.

```python
import numpy as np

class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return params + self.velocity

# Example usage
def saddle_function(x):
    return x[0]**2 - x[1]**2

def gradient(x):
    return np.array([2*x[0], -2*x[1]])

optimizer = MomentumOptimizer()
position = np.array([0.5, 0.5])

for _ in range(100):
    grad = gradient(position)
    position = optimizer.update(position, grad)
    if _ % 20 == 0:
        print(f"Position: {position}, Loss: {saddle_function(position)}")
```

Slide 4: Visualizing Optimization Trajectories

We'll create a visualization tool to track how different optimizers behave around saddle points, comparing standard gradient descent against momentum-based approaches in a challenging loss landscape.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_loss_surface():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 - Y**2  # Classic saddle surface
    return X, Y, Z

def visualize_trajectory(trajectory, X, Y, Z):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    
    # Plot trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            [trajectory[i, 0]**2 - trajectory[i, 1]**2 for i in range(len(trajectory))],
            'r.-', linewidth=2, label='Optimization path')
    
    plt.title('Optimizer Trajectory Around Saddle Point')
    plt.legend()
    plt.show()

# Generate and visualize example trajectory
X, Y, Z = create_loss_surface()
trajectory = [(0.5, 0.5)]  # Starting point
visualize_trajectory(trajectory, X, Y, Z)
```

Slide 5: Implementing Hessian-Free Optimization

Hessian-free optimization provides an efficient way to exploit second-order curvature information without explicitly computing the full Hessian matrix, particularly useful for escaping saddle points.

```python
import numpy as np
from scipy.sparse.linalg import cg

class HessianFreeOptimizer:
    def __init__(self, learning_rate=0.1, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def hessian_vector_product(self, params, vector, function):
        epsilon = 1e-6
        gradient_plus = self.compute_gradient(params + epsilon * vector, function)
        gradient_minus = self.compute_gradient(params - epsilon * vector, function)
        return (gradient_plus - gradient_minus) / (2 * epsilon)
    
    def compute_gradient(self, params, function):
        epsilon = 1e-7
        gradient = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            params_minus = params.copy()
            params_minus[i] -= epsilon
            gradient[i] = (function(params_plus) - function(params_minus)) / (2 * epsilon)
        return gradient

    def optimize(self, initial_params, loss_function):
        params = initial_params.copy()
        
        for iteration in range(self.max_iterations):
            gradient = self.compute_gradient(params, loss_function)
            
            # Solve Hx = -g using conjugate gradient
            def hvp_wrapper(v):
                return self.hessian_vector_product(params, v, loss_function)
            
            search_direction, _ = cg(hvp_wrapper, -gradient)
            
            # Update parameters
            params += self.learning_rate * search_direction
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss_function(params)}")
                
        return params

# Example usage
def test_function(x):
    return x[0]**2 - x[1]**2 + 0.1*x[0]*x[1]

optimizer = HessianFreeOptimizer()
initial_params = np.array([1.0, 1.0])
optimized_params = optimizer.optimize(initial_params, test_function)
```

Slide 6: Adaptive Learning Rate Methods

Adaptive learning rate methods like Adam combine the benefits of momentum with per-parameter learning rate adaptation, making them particularly effective at navigating saddle points in high-dimensional loss landscapes.

```python
import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, gradients):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage with saddle point function
def saddle_function(x):
    return x[0]**2 - x[1]**2 + 0.1*x[0]*x[1]

def compute_gradients(x):
    return np.array([2*x[0] + 0.1*x[1], -2*x[1] + 0.1*x[0]])

# Initialize optimizer and parameters
optimizer = AdamOptimizer()
params = np.array([1.0, 1.0])

# Training loop
for i in range(100):
    grads = compute_gradients(params)
    params = optimizer.update(params, grads)
    
    if i % 20 == 0:
        loss = saddle_function(params)
        print(f"Step {i}: Loss = {loss:.6f}, Position = {params}")
```

Slide 7: Analyzing Loss Landscape Curvature

Understanding the curvature of the loss landscape through eigenvalue analysis helps identify saddle points and determine appropriate optimization strategies for different regions.

```python
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def compute_hessian(x, y, delta=1e-5):
    """Compute Hessian matrix numerically for f(x,y) = x^2 - y^2 + 0.1xy"""
    def f(x, y):
        return x**2 - y**2 + 0.1*x*y
    
    # Second derivatives
    dxx = (f(x + delta, y) - 2*f(x, y) + f(x - delta, y)) / delta**2
    dyy = (f(x, y + delta) - 2*f(x, y) + f(x, y - delta)) / delta**2
    
    # Mixed derivative
    dxy = ((f(x + delta, y + delta) - f(x + delta, y - delta)) -
           (f(x - delta, y + delta) - f(x - delta, y - delta))) / (4*delta**2)
    
    return np.array([[dxx, dxy],
                    [dxy, dyy]])

def analyze_curvature(x_range, y_range, points=20):
    X, Y = np.meshgrid(np.linspace(*x_range, points), np.linspace(*y_range, points))
    min_eigenvals = np.zeros_like(X)
    max_eigenvals = np.zeros_like(X)
    
    for i in range(points):
        for j in range(points):
            H = compute_hessian(X[i,j], Y[i,j])
            eigenvals = eigh(H, eigvals_only=True)
            min_eigenvals[i,j] = min(eigenvals)
            max_eigenvals[i,j] = max(eigenvals)
    
    # Plot curvature analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = ax1.imshow(min_eigenvals, extent=[*x_range, *y_range], origin='lower')
    ax1.set_title('Minimum Eigenvalue')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(max_eigenvals, extent=[*x_range, *y_range], origin='lower')
    ax2.set_title('Maximum Eigenvalue')
    plt.colorbar(im2, ax=ax2)
    
    plt.show()

# Analyze curvature in region around origin
analyze_curvature((-2, 2), (-2, 2))
```

Slide 8: Implementing Trust Region Methods

Trust region methods provide robust optimization by constraining parameter updates based on the local approximation's reliability, particularly effective when dealing with saddle points.

```python
import numpy as np
from scipy.optimize import minimize

class TrustRegionOptimizer:
    def __init__(self, initial_radius=1.0, max_radius=2.0):
        self.radius = initial_radius
        self.max_radius = max_radius
        self.min_radius = 1e-4
        self.eta = 0.1  # Acceptance threshold
        
    def quadratic_model(self, params, gradient, hessian, step):
        """Compute quadratic approximation at current point"""
        return (gradient.dot(step) + 
                0.5 * step.dot(hessian.dot(step)))
    
    def solve_trust_region_subproblem(self, gradient, hessian, radius):
        """Solve the trust region subproblem using dogleg method"""
        n = len(gradient)
        
        # Compute Cauchy point
        g_norm = np.linalg.norm(gradient)
        if g_norm == 0:
            return np.zeros_like(gradient)
            
        try:
            # Try to compute Newton step
            newton_step = np.linalg.solve(hessian, -gradient)
            newton_norm = np.linalg.norm(newton_step)
            
            if newton_norm <= radius:
                return newton_step
                
        except np.linalg.LinAlgError:
            pass
        
        # Fall back to scaled steepest descent
        return -radius * gradient / g_norm
    
    def optimize(self, initial_params, objective_func, max_iterations=100):
        params = initial_params.copy()
        
        for iteration in range(max_iterations):
            # Compute gradient and Hessian
            gradient = compute_gradient(params, objective_func)
            hessian = compute_hessian_matrix(params, objective_func)
            
            # Solve trust region subproblem
            step = self.solve_trust_region_subproblem(gradient, hessian, self.radius)
            
            # Compute actual and predicted reduction
            actual_reduction = (objective_func(params) - 
                              objective_func(params + step))
            predicted_reduction = -self.quadratic_model(params, gradient, 
                                                      hessian, step)
            
            # Update trust region radius
            if predicted_reduction <= 0:
                self.radius *= 0.25
            else:
                ratio = actual_reduction / predicted_reduction
                if ratio < 0.25:
                    self.radius *= 0.5
                elif ratio > 0.75 and np.linalg.norm(step) == self.radius:
                    self.radius = min(2.0 * self.radius, self.max_radius)
            
            # Update parameters if improvement
            if actual_reduction > 0:
                params = params + step
            
            if np.linalg.norm(gradient) < 1e-6:
                break
                
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {objective_func(params)}")
        
        return params

# Helper functions for gradient and Hessian computation
def compute_gradient(params, func, eps=1e-8):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (func(params_plus) - func(params_minus)) / (2 * eps)
    return grad

def compute_hessian_matrix(params, func, eps=1e-8):
    n = len(params)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            params_pp = params.copy()
            params_pp[i] += eps
            params_pp[j] += eps
            
            params_pm = params.copy()
            params_pm[i] += eps
            params_pm[j] -= eps
            
            params_mp = params.copy()
            params_mp[i] -= eps
            params_mp[j] += eps
            
            params_mm = params.copy()
            params_mm[i] -= eps
            params_mm[j] -= eps
            
            hessian[i,j] = ((func(params_pp) - func(params_pm) - 
                            func(params_mp) + func(params_mm)) / 
                           (4 * eps * eps))
    return hessian
```

Slide 9: Eigenvalue Analysis and Saddle Point Detection

Understanding the nature of critical points requires analyzing the eigenvalue spectrum of the Hessian matrix. This implementation provides tools for detecting and characterizing saddle points in neural network loss landscapes.

```python
import numpy as np
from scipy.sparse.linalg import eigsh

class SaddlePointDetector:
    def __init__(self, threshold=1e-6):
        self.threshold = threshold
        
    def compute_hessian_eigenvalues(self, params, loss_fn, k=10):
        """Compute the k largest and smallest eigenvalues of the Hessian."""
        n = len(params)
        
        def hessian_vector_product(v):
            eps = 1e-6
            gradient_plus = self.compute_gradient(params + eps * v, loss_fn)
            gradient_minus = self.compute_gradient(params - eps * v, loss_fn)
            return (gradient_plus - gradient_minus) / (2 * eps)
        
        # Create linear operator for Hessian
        hessian_op = scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=hessian_vector_product
        )
        
        # Compute largest eigenvalues
        largest_eigenvals, _ = eigsh(hessian_op, k=k, which='LA')
        # Compute smallest eigenvalues
        smallest_eigenvals, _ = eigsh(hessian_op, k=k, which='SA')
        
        return np.sort(np.concatenate([smallest_eigenvals, largest_eigenvals]))
    
    def is_saddle_point(self, eigenvalues):
        """Determine if point is a saddle point based on eigenvalue spectrum."""
        positive_eigvals = np.sum(eigenvalues > self.threshold)
        negative_eigvals = np.sum(eigenvalues < -self.threshold)
        
        return positive_eigvals > 0 and negative_eigvals > 0
    
    def characterize_critical_point(self, params, loss_fn):
        """Analyze the nature of a critical point."""
        eigenvalues = self.compute_hessian_eigenvalues(params, loss_fn)
        
        if self.is_saddle_point(eigenvalues):
            point_type = "Saddle Point"
        elif np.all(eigenvalues > -self.threshold):
            point_type = "Local Minimum"
        elif np.all(eigenvalues < self.threshold):
            point_type = "Local Maximum"
        else:
            point_type = "Degenerate Critical Point"
            
        return {
            'type': point_type,
            'eigenvalues': eigenvalues,
            'smallest_eigenvalue': np.min(eigenvalues),
            'largest_eigenvalue': np.max(eigenvalues),
            'condition_number': abs(np.max(eigenvalues) / np.min(eigenvalues))
        }

# Example usage
def test_loss_function(x):
    return x[0]**2 - x[1]**2 + 0.1*x[0]*x[1]**3

detector = SaddlePointDetector()
test_point = np.array([0.1, 0.1])
analysis = detector.characterize_critical_point(test_point, test_loss_function)

print(f"Critical Point Analysis:")
for key, value in analysis.items():
    print(f"{key}: {value}")
```

Slide 10: Second-Order Optimization with Curvature Information

This implementation demonstrates how to leverage curvature information for more effective optimization around saddle points, using a modified Newton method with eigenvalue-based regularization.

```python
import numpy as np
from scipy.linalg import eigh

class CurvatureAwareOptimizer:
    def __init__(self, learning_rate=0.1, damping=1e-4):
        self.learning_rate = learning_rate
        self.damping = damping
        
    def modify_hessian(self, H):
        """Modify Hessian to ensure positive definiteness near saddle points."""
        eigenvals, eigenvecs = eigh(H)
        modified_eigenvals = np.maximum(np.abs(eigenvals), self.damping)
        return eigenvecs @ np.diag(modified_eigenvals) @ eigenvecs.T
    
    def compute_update(self, params, gradient, hessian):
        """Compute parameter update using modified curvature information."""
        modified_H = self.modify_hessian(hessian)
        try:
            update = np.linalg.solve(modified_H, -gradient)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent if numerical issues occur
            update = -gradient
        
        return self.learning_rate * update
    
    def optimize(self, initial_params, objective_fn, max_iterations=100):
        params = initial_params.copy()
        trajectory = [params.copy()]
        
        for iteration in range(max_iterations):
            gradient = compute_gradient(params, objective_fn)
            hessian = compute_hessian_matrix(params, objective_fn)
            
            # Compute update step
            update = self.compute_update(params, gradient, hessian)
            params = params + update
            
            # Store trajectory
            trajectory.append(params.copy())
            
            # Logging
            if iteration % 10 == 0:
                loss = objective_fn(params)
                grad_norm = np.linalg.norm(gradient)
                print(f"Iteration {iteration}:")
                print(f"  Loss: {loss:.6f}")
                print(f"  Gradient norm: {grad_norm:.6f}")
                
            # Convergence check
            if np.linalg.norm(gradient) < 1e-6:
                break
                
        return params, np.array(trajectory)

# Example usage with visualization
def visualize_optimization_path(trajectory, objective_fn):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i,j] = objective_fn(np.array([X[i,j], Y[i,j]]))
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20)
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:,0], trajectory[:,1], 'r.-', label='Optimization path')
    plt.colorbar(label='Loss')
    plt.legend()
    plt.title('Optimization Trajectory in Loss Landscape')
    plt.show()

# Test optimization
def test_function(x):
    return x[0]**2 - x[1]**2 + 0.1*x[0]*x[1]**3

optimizer = CurvatureAwareOptimizer()
initial_point = np.array([1.0, 1.0])
final_params, trajectory = optimizer.optimize(initial_point, test_function)
visualize_optimization_path(trajectory, test_function)
```

Slide 11: Implementing Natural Gradient Descent

Natural gradient descent provides a principled way to handle optimization in the parameter manifold, making it particularly effective at escaping saddle points by considering the geometric structure of the parameter space.

```python
import numpy as np
from scipy.linalg import solve

class NaturalGradientOptimizer:
    def __init__(self, learning_rate=0.1, damping=1e-4):
        self.learning_rate = learning_rate
        self.damping = damping
        
    def compute_fisher_matrix(self, params, model_output, true_output):
        """Compute Fisher Information Matrix approximation."""
        batch_size = len(true_output)
        jacobian = self.compute_jacobian(params, model_output)
        fisher = np.zeros((params.shape[0], params.shape[0]))
        
        for i in range(batch_size):
            j = jacobian[i]
            fisher += np.outer(j, j)
            
        fisher /= batch_size
        # Add damping for numerical stability
        fisher += self.damping * np.eye(fisher.shape[0])
        return fisher
    
    def compute_natural_gradient(self, gradient, fisher):
        """Compute natural gradient using Fisher matrix."""
        return solve(fisher, gradient, assume_a='pos')
    
    def update(self, params, gradient, fisher):
        """Update parameters using natural gradient."""
        natural_grad = self.compute_natural_gradient(gradient, fisher)
        return params - self.learning_rate * natural_grad

    def optimize(self, initial_params, loss_fn, max_iterations=100):
        params = initial_params.copy()
        
        for iteration in range(max_iterations):
            # Compute model output and gradients
            output = self.forward_pass(params)
            gradient = self.compute_gradient(params, loss_fn)
            
            # Compute Fisher matrix and update
            fisher = self.compute_fisher_matrix(params, output, target_output)
            params = self.update(params, gradient, fisher)
            
            if iteration % 10 == 0:
                loss = loss_fn(params)
                print(f"Iteration {iteration}, Loss: {loss:.6f}")
                
            if np.linalg.norm(gradient) < 1e-6:
                break
                
        return params

    def forward_pass(self, params):
        """Simple neural network forward pass for demonstration."""
        # Implement your network architecture here
        x = input_data  # Assuming this is available
        h = np.tanh(x @ params[:half] + b1)
        y = h @ params[half:] + b2
        return y
    
    def compute_jacobian(self, params, output):
        """Compute Jacobian matrix of the model output with respect to parameters."""
        n_samples = output.shape[0]
        n_params = params.shape[0]
        jacobian = np.zeros((n_samples, n_params))
        
        epsilon = 1e-6
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += epsilon
            output_plus = self.forward_pass(params_plus)
            
            params_minus = params.copy()
            params_minus[i] -= epsilon
            output_minus = self.forward_pass(params_minus)
            
            jacobian[:, i] = (output_plus - output_minus).ravel() / (2 * epsilon)
            
        return jacobian

# Example usage
def simple_loss(params):
    return np.sum(params**2) - params[0]*params[1]

# Initialize optimizer
optimizer = NaturalGradientOptimizer()
initial_params = np.random.randn(10)  # 10-dimensional parameter space
optimized_params = optimizer.optimize(initial_params, simple_loss)
```

Slide 12: Analyzing Saddle Point Escape Times

This implementation provides tools to analyze how different optimizers perform in escaping saddle points, measuring the time and trajectory characteristics of various optimization methods.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class SaddlePointAnalyzer:
    def __init__(self, optimizers, test_problems):
        self.optimizers = optimizers
        self.test_problems = test_problems
        self.results = defaultdict(dict)
        
    def measure_escape_time(self, optimizer, problem, max_iterations=1000):
        """Measure iterations needed to escape saddle point region."""
        params = problem['initial_point']
        trajectory = [params.copy()]
        escape_threshold = problem.get('escape_threshold', 1e-2)
        
        for iteration in range(max_iterations):
            gradient = problem['gradient'](params)
            params = optimizer.update(params, gradient)
            trajectory.append(params.copy())
            
            # Check if escaped saddle point region
            if np.linalg.norm(params - problem['saddle_point']) > escape_threshold:
                return {
                    'iterations': iteration + 1,
                    'trajectory': np.array(trajectory),
                    'escaped': True
                }
                
        return {
            'iterations': max_iterations,
            'trajectory': np.array(trajectory),
            'escaped': False
        }
    
    def run_analysis(self):
        """Run analysis for all optimizers and test problems."""
        for opt_name, optimizer in self.optimizers.items():
            for prob_name, problem in self.test_problems.items():
                result = self.measure_escape_time(optimizer, problem)
                self.results[opt_name][prob_name] = result
                
    def visualize_results(self):
        """Visualize escape trajectories and performance metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot escape times
        opt_names = list(self.optimizers.keys())
        prob_names = list(self.test_problems.keys())
        
        escape_times = np.zeros((len(opt_names), len(prob_names)))
        for i, opt_name in enumerate(opt_names):
            for j, prob_name in enumerate(prob_names):
                escape_times[i, j] = self.results[opt_name][prob_name]['iterations']
        
        plt.subplot(121)
        plt.imshow(escape_times, aspect='auto')
        plt.xticks(range(len(prob_names)), prob_names, rotation=45)
        plt.yticks(range(len(opt_names)), opt_names)
        plt.colorbar(label='Iterations to escape')
        plt.title('Saddle Point Escape Performance')
        
        # Plot example trajectory
        plt.subplot(122)
        example_traj = self.results[opt_names[0]][prob_names[0]]['trajectory']
        plt.plot(example_traj[:, 0], example_traj[:, 1], 'b.-')
        plt.scatter(*self.test_problems[prob_names[0]]['saddle_point'], 
                   color='red', label='Saddle point')
        plt.title(f'Example Trajectory: {opt_names[0]}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
test_problems = {
    'simple_saddle': {
        'initial_point': np.array([0.1, 0.1]),
        'saddle_point': np.array([0.0, 0.0]),
        'gradient': lambda x: np.array([2*x[0], -2*x[1]]),
        'escape_threshold': 0.5
    }
}

optimizers = {
    'sgd': GradientDescent(learning_rate=0.1),
    'momentum': MomentumOptimizer(learning_rate=0.1, momentum=0.9),
    'adam': AdamOptimizer(learning_rate=0.1)
}

analyzer = SaddlePointAnalyzer(optimizers, test_problems)
analyzer.run_analysis()
analyzer.visualize_results()
```

Slide 13: Implementing Stochastic Weight Averaging

Stochastic Weight Averaging (SWA) provides a robust approach to finding better solutions by averaging multiple points along the optimization trajectory, helping to escape and avoid saddle points.

```python
import numpy as np
from copy import deepcopy

class SWAOptimizer:
    def __init__(self, base_optimizer, swa_start=10, swa_freq=5):
        self.base_optimizer = base_optimizer
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_model = None
        self.n_averaged = 0
        
    def should_average(self, iteration):
        """Determine if we should update SWA model at current iteration."""
        return (iteration >= self.swa_start and 
                (iteration - self.swa_start) % self.swa_freq == 0)
    
    def update_swa_model(self, current_params):
        """Update the averaged model parameters."""
        if self.swa_model is None:
            self.swa_model = current_params.copy()
        else:
            self.n_averaged += 1
            alpha = 1.0 / (self.n_averaged + 1)
            self.swa_model = (1.0 - alpha) * self.swa_model + alpha * current_params
    
    def optimize(self, initial_params, loss_fn, max_iterations=1000):
        """Optimize using SWA."""
        params = initial_params.copy()
        trajectory = []
        swa_trajectory = []
        
        for iteration in range(max_iterations):
            # Regular optimization step
            gradient = compute_gradient(params, loss_fn)
            params = self.base_optimizer.update(params, gradient)
            trajectory.append(params.copy())
            
            # SWA update
            if self.should_average(iteration):
                self.update_swa_model(params)
                swa_trajectory.append(self.swa_model.copy())
            
            # Logging
            if iteration % 10 == 0:
                base_loss = loss_fn(params)
                swa_loss = loss_fn(self.swa_model) if self.swa_model is not None else None
                print(f"Iteration {iteration}:")
                print(f"  Base loss: {base_loss:.6f}")
                if swa_loss is not None:
                    print(f"  SWA loss: {swa_loss:.6f}")
        
        return {
            'final_params': params,
            'swa_params': self.swa_model,
            'trajectory': np.array(trajectory),
            'swa_trajectory': np.array(swa_trajectory)
        }

    def visualize_optimization(self, results, loss_fn):
        """Visualize optimization trajectories."""
        plt.figure(figsize=(15, 5))
        
        # Plot trajectories
        plt.subplot(121)
        plt.plot(results['trajectory'][:,0], results['trajectory'][:,1], 
                'b.-', alpha=0.5, label='SGD path')
        plt.plot(results['swa_trajectory'][:,0], results['swa_trajectory'][:,1], 
                'r.-', linewidth=2, label='SWA path')
        plt.legend()
        plt.title('Optimization Trajectories')
        
        # Plot loss landscape
        plt.subplot(122)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i,j] = loss_fn(np.array([X[i,j], Y[i,j]]))
        
        plt.contour(X, Y, Z, levels=20)
        plt.colorbar(label='Loss')
        plt.scatter(results['final_params'][0], results['final_params'][1], 
                   color='blue', label='Final SGD')
        plt.scatter(results['swa_params'][0], results['swa_params'][1], 
                   color='red', label='Final SWA')
        plt.legend()
        plt.title('Loss Landscape')
        
        plt.tight_layout()
        plt.show()

# Example usage
def test_loss_function(x):
    """Test function with multiple local minima and saddle points."""
    return (x[0]**2 - x[1]**2 + 
            0.1 * np.sin(5*x[0]) + 
            0.1 * np.cos(5*x[1]))

# Initialize optimizers
base_optimizer = MomentumOptimizer(learning_rate=0.1, momentum=0.9)
swa_optimizer = SWAOptimizer(base_optimizer, swa_start=50, swa_freq=5)

# Run optimization
initial_params = np.array([1.0, 1.0])
results = swa_optimizer.optimize(initial_params, test_loss_function)

# Visualize results
swa_optimizer.visualize_optimization(results, test_loss_function)
```

Slide 14: Additional Resources

\[List of relevant papers with complete URLs\]

*   "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization" [https://arxiv.org/abs/1406.2572](https://arxiv.org/abs/1406.2572)
*   "Deep Learning without Poor Local Minima" [https://arxiv.org/abs/1605.07110](https://arxiv.org/abs/1605.07110)
*   "On the Geometry of Gradient Descent for Deep Linear Neural Networks" [https://arxiv.org/abs/1710.00779](https://arxiv.org/abs/1710.00779)
*   "The Loss Surfaces of Multilayer Networks" [https://arxiv.org/abs/1412.0233](https://arxiv.org/abs/1412.0233)
*   "Gradient Descent Can Take Exponential Time to Escape Saddle Points" [https://arxiv.org/abs/1705.10412](https://arxiv.org/abs/1705.10412)
*   "How to Escape Saddle Points Efficiently" [https://arxiv.org/abs/1703.00887](https://arxiv.org/abs/1703.00887)

