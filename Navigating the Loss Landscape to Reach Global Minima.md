## Navigating the Loss Landscape to Reach Global Minima
Slide 1: Understanding Loss Landscape and Global Minima

The loss landscape in machine learning represents the relationship between model parameters and the error function. Global minima represent optimal parameter values where the loss function reaches its lowest possible value across the entire parameter space.

```python
import numpy as np
import matplotlib.pyplot as plt

def loss_landscape(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2  # Rosenbrock function

# Create mesh grid for visualization
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = loss_landscape(X, Y)

# Plotting loss landscape
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
plt.colorbar(label='Loss')
plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('Loss Landscape Visualization')
plt.show()
```

Slide 2: Implementing Basic SGD Optimizer

Stochastic Gradient Descent implementation showcases the fundamental optimization process, where parameters are updated iteratively based on the gradient of the loss function computed on random batches of data.

```python
class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, gradients):
        """
        params: Current model parameters
        gradients: Computed gradients for each parameter
        """
        updated_params = {}
        for param_name, param_value in params.items():
            updated_params[param_name] = param_value - self.learning_rate * gradients[param_name]
        return updated_params

# Example usage
params = {'w1': np.array([0.1, 0.2]), 'b1': np.array([0.01])}
gradients = {'w1': np.array([0.01, 0.02]), 'b1': np.array([0.001])}

optimizer = SGDOptimizer(learning_rate=0.1)
updated_params = optimizer.update(params, gradients)
print("Updated parameters:", updated_params)
```

Slide 3: Enhanced SGD with Momentum

Momentum helps accelerate SGD in the relevant direction and dampens oscillations by adding a fraction of the previous update to the current update, improving convergence speed and stability.

```python
class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params, gradients):
        if not self.velocity:
            self.velocity = {k: np.zeros_like(v) for k, v in params.items()}
        
        updated_params = {}
        for param_name in params:
            # Update velocity
            self.velocity[param_name] = (self.momentum * self.velocity[param_name] - 
                                       self.learning_rate * gradients[param_name])
            # Update parameters
            updated_params[param_name] = params[param_name] + self.velocity[param_name]
        
        return updated_params
```

Slide 4: Local Minima Detection

Understanding when SGD gets trapped in local minima is crucial for optimization. This implementation demonstrates how to detect potential local minima by monitoring gradient magnitudes and loss improvements.

```python
def detect_local_minimum(loss_history, gradient_history, window_size=10, 
                        threshold=1e-5):
    """
    Detects if optimization is stuck in a local minimum
    """
    if len(loss_history) < window_size:
        return False
    
    recent_losses = loss_history[-window_size:]
    recent_grads = gradient_history[-window_size:]
    
    loss_improvement = abs(recent_losses[-1] - recent_losses[0])
    avg_gradient_magnitude = np.mean([np.linalg.norm(g) for g in recent_grads])
    
    return loss_improvement < threshold and avg_gradient_magnitude < threshold
```

Slide 5: Learning Rate Scheduling Implementation

Adaptive learning rate scheduling helps optimize convergence by adjusting the learning rate during training. This implementation shows common scheduling strategies including step decay and exponential decay.

```python
class LearningRateScheduler:
    def __init__(self, initial_lr=0.1):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10):
        """Step decay schedule"""
        self.current_lr = self.initial_lr * np.power(drop_rate, 
                                                    np.floor((1+epoch)/epochs_drop))
        return self.current_lr
    
    def exp_decay(self, epoch, decay_rate=0.95):
        """Exponential decay schedule"""
        self.current_lr = self.initial_lr * np.exp(-decay_rate*epoch)
        return self.current_lr
    
    def cosine_decay(self, epoch, total_epochs):
        """Cosine annealing schedule"""
        self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return self.current_lr

# Example usage
scheduler = LearningRateScheduler(initial_lr=0.1)
for epoch in range(10):
    lr = scheduler.step_decay(epoch)
    print(f"Epoch {epoch}, Learning Rate: {lr:.4f}")
```

Slide 6: Mini-batch SGD Implementation

Mini-batch processing improves training stability and computational efficiency by updating model parameters using small random subsets of the training data instead of individual samples.

```python
def create_mini_batches(X, y, batch_size=32, shuffle=True):
    m = X.shape[0]
    mini_batches = []
    
    # Shuffle data
    if shuffle:
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
    else:
        X_shuffled = X
        y_shuffled = y
    
    # Create mini-batches
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        mini_batch_X = X_shuffled[k * batch_size:(k + 1) * batch_size]
        mini_batch_y = y_shuffled[k * batch_size:(k + 1) * batch_size]
        mini_batches.append((mini_batch_X, mini_batch_y))
    
    # Handle the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        mini_batch_X = X_shuffled[num_complete_batches * batch_size:]
        mini_batch_y = y_shuffled[num_complete_batches * batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_y))
    
    return mini_batches
```

Slide 7: Gradient Computation and Error Analysis

Accurate gradient computation and error analysis are essential for understanding model performance and optimization progress during training.

```python
def compute_gradients_and_error(model, X_batch, y_batch, loss_fn):
    """
    Computes gradients and error metrics for a mini-batch
    """
    predictions = model.forward(X_batch)
    loss = loss_fn(predictions, y_batch)
    
    # Compute gradients using autograd or manual backpropagation
    gradients = {}
    for param_name, param in model.parameters.items():
        gradients[param_name] = np.zeros_like(param)
        # Gradient computation logic here
        
    # Calculate error metrics
    mse = np.mean((predictions - y_batch) ** 2)
    mae = np.mean(np.abs(predictions - y_batch))
    
    return {
        'gradients': gradients,
        'loss': loss,
        'mse': mse,
        'mae': mae
    }
```

Slide 8: Real-world Example: Linear Regression with SGD

Implementation of linear regression using SGD optimization on a real dataset, demonstrating practical application of the concepts including data preprocessing and model training.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, n_features=20):
        self.lr = learning_rate
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def update_parameters(self, gradients):
        self.weights -= self.lr * gradients['weights']
        self.bias -= self.lr * gradients['bias']
    
    def train(self, X, y, epochs=100, batch_size=32):
        losses = []
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X, y, batch_size)
            epoch_loss = 0
            
            for X_batch, y_batch in mini_batches:
                y_pred = self.forward(X_batch)
                
                # Compute gradients
                grad_weights = -2/batch_size * X_batch.T.dot(y_batch - y_pred)
                grad_bias = -2/batch_size * np.sum(y_batch - y_pred)
                
                # Update parameters
                self.update_parameters({
                    'weights': grad_weights,
                    'bias': grad_bias
                })
                
                # Track loss
                batch_loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += batch_loss
                
            losses.append(epoch_loss / len(mini_batches))
            
        return losses
```

Slide 9: Implementing Adaptive Learning Rate Methods

Adaptive learning rate methods like AdaGrad and RMSprop automatically adjust learning rates for each parameter based on historical gradient information, improving convergence in complex loss landscapes.

```python
class AdaptiveOptimizer:
    def __init__(self, learning_rate=0.01, epsilon=1e-8, method='adagrad'):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.method = method
        self.cache = {}
        self.beta = 0.9  # For RMSprop
        
    def update(self, params, gradients):
        if not self.cache:
            self.cache = {k: np.zeros_like(v) for k, v in params.items()}
        
        updated_params = {}
        for param_name, param in params.items():
            if self.method == 'adagrad':
                # Accumulate squared gradients
                self.cache[param_name] += gradients[param_name]**2
                adapted_lr = self.lr / (np.sqrt(self.cache[param_name] + self.epsilon))
                
            elif self.method == 'rmsprop':
                # Exponentially decaying average of squared gradients
                self.cache[param_name] = (self.beta * self.cache[param_name] + 
                                        (1 - self.beta) * gradients[param_name]**2)
                adapted_lr = self.lr / (np.sqrt(self.cache[param_name] + self.epsilon))
            
            updated_params[param_name] = param - adapted_lr * gradients[param_name]
            
        return updated_params

# Example usage
optimizer = AdaptiveOptimizer(method='rmsprop')
params = {'w': np.array([0.1, 0.2]), 'b': np.array([0.01])}
grads = {'w': np.array([0.01, 0.02]), 'b': np.array([0.001])}
updated = optimizer.update(params, grads)
```

Slide 10: Visualization of Optimization Trajectories

Understanding optimization paths helps diagnose convergence issues. This implementation visualizes the trajectory of different optimization algorithms in the loss landscape.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_optimization_trajectory(optimizer_path, loss_landscape_fn, bounds):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh for loss landscape
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = loss_landscape_fn(X, Y)
    
    # Plot loss landscape
    ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    
    # Plot optimization trajectory
    path = np.array(optimizer_path)
    ax.plot(path[:, 0], path[:, 1], 
            [loss_landscape_fn(x, y) for x, y in path],
            'r.-', linewidth=2, label='Optimization Path')
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Loss')
    plt.title('Optimization Trajectory in Loss Landscape')
    plt.legend()
    plt.show()

# Example usage
def simple_loss(x, y):
    return x**2 + y**2  # Simple quadratic loss

path = [(1.5, 1.5), (1.0, 1.0), (0.5, 0.5), (0.1, 0.1)]
plot_optimization_trajectory(path, simple_loss, [-2, 2])
```

Slide 11: Real-world Example: Neural Network with SGD

Implementation of a simple neural network trained with SGD on the MNIST dataset, showcasing practical application in deep learning.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * 0.01,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * 0.01,
            'b2': np.zeros(output_size)
        }
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.cache = {}
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = self.relu(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = self.softmax(self.cache['Z2'])
        return self.cache['A2']
    
    def backward(self, X, y, m):
        grads = {}
        delta3 = self.cache['A2']
        delta3[range(m), y] -= 1
        
        grads['W2'] = np.dot(self.cache['A1'].T, delta3) / m
        grads['b2'] = np.sum(delta3, axis=0) / m
        
        delta2 = np.dot(delta3, self.params['W2'].T) * (self.cache['Z1'] > 0)
        grads['W1'] = np.dot(X.T, delta2) / m
        grads['b1'] = np.sum(delta2, axis=0) / m
        
        return grads
```

Slide 12: Advanced Gradient Descent Techniques

Advanced optimization techniques like Nesterov momentum and gradient clipping help improve training stability and convergence speed by modifying the traditional SGD update rule.

```python
class AdvancedSGD:
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=True, 
                 clip_threshold=1.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.clip_threshold = clip_threshold
        self.velocity = {}
    
    def clip_gradients(self, gradients):
        """Implement gradient clipping to prevent exploding gradients"""
        grad_norm = np.sqrt(sum(np.sum(np.square(g)) for g in gradients.values()))
        scaling_factor = min(1.0, self.clip_threshold / (grad_norm + 1e-8))
        return {k: v * scaling_factor for k, v in gradients.items()}
    
    def update(self, params, gradients):
        if not self.velocity:
            self.velocity = {k: np.zeros_like(v) for k, v in params.items()}
        
        # Clip gradients
        gradients = self.clip_gradients(gradients)
        
        updated_params = {}
        for param_name in params:
            if self.nesterov:
                # Nesterov momentum update
                self.velocity[param_name] = (self.momentum * self.velocity[param_name] - 
                                          self.lr * gradients[param_name])
                updated_params[param_name] = (params[param_name] + 
                                           self.momentum * self.velocity[param_name] - 
                                           self.lr * gradients[param_name])
            else:
                # Standard momentum update
                self.velocity[param_name] = (self.momentum * self.velocity[param_name] - 
                                          self.lr * gradients[param_name])
                updated_params[param_name] = (params[param_name] + 
                                           self.velocity[param_name])
        
        return updated_params
```

Slide 13: Mathematical Foundations of SGD

The theoretical underpinnings of SGD involve concepts from optimization theory and statistical learning. This implementation demonstrates the mathematical principles in code.

```python
def sgd_mathematical_components():
    """
    Demonstrates mathematical components of SGD using LaTeX notation
    """
    # Gradient descent update rule
    latex_gradient_descent = """
    # Gradient Descent Update Rule:
    $$θ_{t+1} = θ_t - η∇J(θ_t)$$
    
    # Stochastic Gradient Descent Update Rule:
    $$θ_{t+1} = θ_t - η∇J_i(θ_t)$$
    
    # Momentum Update Rule:
    $$v_{t+1} = μv_t - η∇J(θ_t)$$
    $$θ_{t+1} = θ_t + v_{t+1}$$
    
    # Nesterov Momentum:
    $$v_{t+1} = μv_t - η∇J(θ_t + μv_t)$$
    $$θ_{t+1} = θ_t + v_{t+1}$$
    """
    
    # Learning rate decay
    def learning_rate_decay(initial_lr, epoch, decay_rate):
        """
        $$η_t = η_0 / (1 + kt)$$
        where k is the decay rate and t is the epoch number
        """
        return initial_lr / (1 + decay_rate * epoch)
    
    return latex_gradient_descent

print(sgd_mathematical_components())
```

Slide 14: Additional Resources

*   "On the Convergence of Adam and Beyond" - ArXiv URL: [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   "Why Momentum Really Works" - ArXiv URL: [https://arxiv.org/abs/1705.03071](https://arxiv.org/abs/1705.03071)
*   "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" - ArXiv URL: [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
*   For practical implementations and tutorials, search for:
    *   "Deep Learning Optimization Algorithms"
    *   "PyTorch SGD Implementation"
    *   "TensorFlow Optimizers Guide"

