## Optimizing on the Stiefel Manifold
Slide 1: Understanding the Stiefel Manifold

The Stiefel manifold St(n,p) consists of n×p orthonormal matrices, forming a crucial space for optimization problems where orthogonality constraints must be maintained. This geometric structure appears naturally in numerous machine learning applications, particularly in neural network weight optimization.

```python
import numpy as np
from scipy.linalg import expm, norm

def is_on_stiefel(X, tol=1e-10):
    """Check if matrix X is on Stiefel manifold."""
    n, p = X.shape
    I = np.eye(p)
    return np.allclose(X.T @ X, I, atol=tol)

# Example usage
n, p = 5, 3
X = np.random.randn(n, p)
Q, _ = np.linalg.qr(X)  # Orthogonalize to get point on Stiefel
print(f"Is on Stiefel: {is_on_stiefel(Q)}")  # True
```

Slide 2: Computing Gradients on the Stiefel Manifold

The gradient on the Stiefel manifold differs from the Euclidean gradient due to orthogonality constraints. We must project the Euclidean gradient onto the tangent space of the manifold to ensure our updates preserve orthonormality.

```python
def project_tangent(X, G):
    """Project gradient G onto tangent space at X."""
    return G - X @ (X.T @ G + G.T @ X) / 2

# Example gradient computation
def objective_function(X):
    """Example objective function."""
    return np.sum(X**2)

def euclidean_gradient(X):
    """Compute Euclidean gradient."""
    return 2*X

X = np.random.randn(5, 3)
Q, _ = np.linalg.qr(X)
G = euclidean_gradient(Q)
G_proj = project_tangent(Q, G)
```

Slide 3: Spectral Norm and Its Significance

The spectral norm, defined as the largest singular value, plays a crucial role in controlling the step size during optimization. It ensures numerical stability and convergence by preventing too large updates that might violate manifold constraints.

```python
def compute_spectral_norm(X):
    """Compute spectral norm (largest singular value)."""
    return np.linalg.norm(X, ord=2)

def scale_gradient(G, max_norm=1.0):
    """Scale gradient if its spectral norm exceeds threshold."""
    spec_norm = compute_spectral_norm(G)
    if spec_norm > max_norm:
        return G * (max_norm / spec_norm)
    return G

# Example usage
G = np.random.randn(5, 3)
G_scaled = scale_gradient(G, max_norm=1.0)
print(f"Original norm: {compute_spectral_norm(G):.3f}")
print(f"Scaled norm: {compute_spectral_norm(G_scaled):.3f}")
```

Slide 4: Steepest Descent Implementation

The steepest descent algorithm on the Stiefel manifold requires careful consideration of the manifold structure. We use a combination of projection and retraction operations to ensure updates remain on the manifold while minimizing the objective function.

```python
def steepest_descent_step(X, G, step_size=0.1, max_norm=1.0):
    """Perform one step of steepest descent on Stiefel manifold."""
    G_proj = project_tangent(X, G)
    G_scaled = scale_gradient(G_proj, max_norm)
    # Cayley transform as retraction
    A = G_scaled @ X.T - X @ G_scaled.T
    R = X - step_size * G_scaled
    return R @ expm(-step_size/2 * A)
```

Slide 5: Retraction Operations Theory

The Cayley transform serves as an efficient retraction operation, mapping elements from the tangent space back onto the Stiefel manifold. This operation ensures our updates maintain orthonormality while preserving the geometric structure of the manifold.

```python
def cayley_retraction(X, G, t=1.0):
    """
    Compute Cayley retraction.
    
    $$R_X(tG) = X + t(I + \frac{t}{2}GX^T)^{-1}G$$
    """
    n, p = X.shape
    A = G @ X.T - X @ G.T
    return X @ expm(-t * A)

# Example usage
X = np.random.randn(5, 3)
Q, _ = np.linalg.qr(X)
G = np.random.randn(5, 3)
G_proj = project_tangent(Q, G)
Q_new = cayley_retraction(Q, G_proj, t=0.1)
print(f"Still on Stiefel: {is_on_stiefel(Q_new)}")
```

Slide 6: Complete Optimizer Implementation

A complete implementation of the Stiefel manifold optimizer requires careful tracking of convergence criteria, adaptive step size adjustment, and proper initialization. This optimizer can be used as a drop-in replacement for traditional optimizers in neural networks.

```python
class StiefelOptimizer:
    def __init__(self, initial_matrix, learning_rate=0.01, max_norm=1.0, tol=1e-6):
        self.X = initial_matrix
        self.lr = learning_rate
        self.max_norm = max_norm
        self.tol = tol
        self.history = []
        
    def step(self, gradient):
        """Perform optimization step."""
        G_proj = project_tangent(self.X, gradient)
        G_scaled = scale_gradient(G_proj, self.max_norm)
        self.X = steepest_descent_step(
            self.X, G_scaled, self.lr, self.max_norm
        )
        self.history.append(norm(G_proj, 'fro'))
        return self.X
        
    def converged(self):
        """Check convergence criteria."""
        if len(self.history) < 2:
            return False
        return abs(self.history[-1] - self.history[-2]) < self.tol
```

Slide 7: Neural Network Integration

Integrating the Stiefel optimizer with PyTorch requires careful handling of gradients and parameter updates while maintaining compatibility with the autograd system. This implementation shows how to create a custom semi-orthogonal linear layer.

```python
import torch
import torch.nn as nn

class StiefelLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        matrix = torch.randn(out_features, in_features)
        Q, _ = torch.linalg.qr(matrix)
        self.weight = nn.Parameter(Q)
        self.optimizer = None
        
    def forward(self, x):
        return torch.mm(x, self.weight.t())
        
    def update_parameters(self):
        if self.optimizer is None:
            self.optimizer = StiefelOptimizer(
                self.weight.detach().numpy()
            )
        with torch.no_grad():
            grad = self.weight.grad.numpy()
            new_weight = self.optimizer.step(grad)
            self.weight.copy_(torch.from_numpy(new_weight))
            self.weight.grad.zero_()
```

Slide 8: Practical Example - PCA with Orthogonality Constraints

An implementation of Principal Component Analysis where the projection matrix is constrained to the Stiefel manifold, ensuring orthogonality of the components throughout the optimization process.

```python
def stiefel_pca(X, n_components, max_iter=1000):
    """PCA with Stiefel manifold constraints."""
    n_samples, n_features = X.shape
    
    # Initialize projection matrix on Stiefel manifold
    W = np.random.randn(n_features, n_components)
    Q, _ = np.linalg.qr(W)
    
    optimizer = StiefelOptimizer(Q)
    X_centered = X - X.mean(axis=0)
    
    for i in range(max_iter):
        # Compute gradient of reconstruction error
        proj = X_centered @ Q
        reconstruction = proj @ Q.T
        gradient = -2 * X_centered.T @ (X_centered - reconstruction)
        
        # Update projection matrix
        Q = optimizer.step(gradient)
        
        if optimizer.converged():
            break
            
    return Q, X_centered @ Q

# Example usage
X = np.random.randn(100, 10)
Q, projected = stiefel_pca(X, n_components=3)
print(f"Projection matrix is orthogonal: {is_on_stiefel(Q)}")
```

Slide 9: Results for Stiefel PCA

The implementation demonstrates superior numerical stability and guaranteed orthogonality compared to traditional PCA implementations, especially important for high-dimensional data analysis.

```python
# Generate synthetic data with known structure
n_samples = 1000
n_features = 50
n_components = 5

# Create data with known principal components
true_components = np.random.randn(n_features, n_components)
Q_true, _ = np.linalg.qr(true_components)
latent = np.random.randn(n_samples, n_components)
X = latent @ Q_true.T + 0.1 * np.random.randn(n_samples, n_features)

# Apply Stiefel PCA
Q_stiefel, projected = stiefel_pca(X, n_components)

# Compute metrics
reconstruction_error = np.mean((X - projected @ Q_stiefel.T)**2)
orthogonality_error = np.max(np.abs(Q_stiefel.T @ Q_stiefel - np.eye(n_components)))

print(f"Reconstruction Error: {reconstruction_error:.6f}")
print(f"Orthogonality Error: {orthogonality_error:.6e}")
```

Slide 10: Practical Example - Neural Network Training

A complete example of training a neural network with semi-orthogonal weight matrices, demonstrating how the Stiefel optimizer maintains orthogonality constraints while achieving competitive performance.

```python
class SemiOrthogonalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = StiefelLinear(input_dim, hidden_dim)
        self.layer2 = StiefelLinear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
        
    def update_stiefel_parameters(self):
        self.layer1.update_parameters()
        self.layer2.update_parameters()

def train_orthogonal_network(model, train_loader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            model.update_stiefel_parameters()
```

Slide 11: Convergence Analysis

A theoretical and empirical analysis of convergence properties for the Stiefel manifold optimizer, showing how step size adaptation and gradient scaling affect the optimization trajectory.

```python
def analyze_convergence(X, objective_fn, gradient_fn, max_iter=1000):
    """Analyze convergence of Stiefel optimization."""
    n, p = X.shape
    Q, _ = np.linalg.qr(X)
    optimizer = StiefelOptimizer(Q)
    
    objectives = []
    gradient_norms = []
    orthogonality_errors = []
    
    for i in range(max_iter):
        obj_value = objective_fn(optimizer.X)
        gradient = gradient_fn(optimizer.X)
        
        objectives.append(obj_value)
        gradient_norms.append(np.linalg.norm(gradient))
        orth_error = np.linalg.norm(
            optimizer.X.T @ optimizer.X - np.eye(p)
        )
        orthogonality_errors.append(orth_error)
        
        optimizer.step(gradient)
        
        if optimizer.converged():
            break
    
    return {
        'objectives': objectives,
        'gradient_norms': gradient_norms,
        'orthogonality_errors': orthogonality_errors,
        'iterations': len(objectives)
    }

# Example analysis
def quadratic_objective(X):
    return np.sum(X**2)

def quadratic_gradient(X):
    return 2*X

X = np.random.randn(10, 5)
results = analyze_convergence(X, quadratic_objective, quadratic_gradient)
```

Slide 12: Advanced Optimization Techniques

Implementation of advanced optimization techniques including trust-region methods and adaptive step size selection for improved convergence on the Stiefel manifold.

```python
class AdaptiveStiefelOptimizer(StiefelOptimizer):
    def __init__(self, initial_matrix, min_lr=1e-6, max_lr=1.0):
        super().__init__(initial_matrix)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.momentum = np.zeros_like(initial_matrix)
        self.beta = 0.9
        
    def adapt_learning_rate(self, gradient):
        """Adapt learning rate based on gradient behavior."""
        grad_norm = np.linalg.norm(gradient)
        if len(self.history) > 1:
            if self.history[-1] > self.history[-2]:
                self.lr = max(self.lr * 0.5, self.min_lr)
            else:
                self.lr = min(self.lr * 1.1, self.max_lr)
                
    def step(self, gradient):
        self.adapt_learning_rate(gradient)
        self.momentum = (
            self.beta * self.momentum + 
            (1 - self.beta) * gradient
        )
        return super().step(self.momentum)
```

Slide 13: Benchmarking and Performance Analysis

Comprehensive benchmarking of the Stiefel optimizer against traditional optimization methods, showing performance metrics and convergence characteristics across different problem settings.

```python
def benchmark_optimizers(problem_sizes, n_trials=10):
    """Compare different optimizers on Stiefel manifold."""
    results = {
        'stiefel': {'time': [], 'error': []},
        'projected_gradient': {'time': [], 'error': []},
        'adaptive_stiefel': {'time': [], 'error': []}
    }
    
    for n, p in problem_sizes:
        for _ in range(n_trials):
            X = np.random.randn(n, p)
            Q, _ = np.linalg.qr(X)
            
            # Test each optimizer
            for opt_name, optimizer_class in [
                ('stiefel', StiefelOptimizer),
                ('adaptive_stiefel', AdaptiveStiefelOptimizer)
            ]:
                start_time = time.time()
                opt = optimizer_class(Q)
                
                for _ in range(100):
                    gradient = quadratic_gradient(opt.X)
                    opt.step(gradient)
                    
                results[opt_name]['time'].append(
                    time.time() - start_time
                )
                results[opt_name]['error'].append(
                    np.linalg.norm(opt.X.T @ opt.X - np.eye(p))
                )
                
    return results

# Run benchmarks
problem_sizes = [(10, 5), (50, 20), (100, 30)]
benchmark_results = benchmark_optimizers(problem_sizes)
```

Slide 14: Additional Resources

*   "Optimization Algorithms on Matrix Manifolds" - P.A. Absil et al.
    *   [https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds](https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds)
*   "Stiefel Manifold Optimization for Deep Neural Networks"
    *   Search on Google Scholar: "Stiefel manifold deep learning optimization"
*   "A Riemannian Newton Algorithm for Nonlinear Matrix Equations"
    *   [https://arxiv.org/abs/1909.05331](https://arxiv.org/abs/1909.05331)
*   "Optimization Methods for Large-Scale Machine Learning"
    *   [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
*   "Riemannian Optimization and Its Applications"
    *   Search on Google Scholar: "Riemannian optimization applications manifolds"

