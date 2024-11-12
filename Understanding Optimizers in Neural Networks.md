## Understanding Optimizers in Neural Networks
Slide 1: Understanding Basic Gradient Descent

Gradient descent forms the foundation of optimization in neural networks. It iteratively adjusts parameters by computing gradients of the loss function with respect to model parameters, moving in the direction that reduces the loss most rapidly.

```python
import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update(self, params, grads):
        # Simple update rule: params = params - learning_rate * gradients
        return params - self.learning_rate * grads

# Example usage
params = np.array([1.0, 2.0])
grads = np.array([0.1, 0.2])
optimizer = GradientDescent(learning_rate=0.1)
new_params = optimizer.update(params, grads)
print(f"Original params: {params}")
print(f"Updated params: {new_params}")
```

Slide 2: Implementing Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) processes one sample at a time, making it more memory-efficient and often faster than batch gradient descent. It introduces randomness that can help escape local minima.

```python
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def compute_gradients(self, X, y, w, b):
        m = X.shape[0]
        y_pred = X.dot(w) + b
        dw = (1/m) * X.T.dot(y_pred - y)
        db = (1/m) * np.sum(y_pred - y)
        return dw, db
    
    def step(self, w, b, dw, db):
        w -= self.learning_rate * dw
        b -= self.learning_rate * db
        return w, b

# Example usage
X = np.random.randn(100, 2)
y = np.random.randn(100)
w = np.zeros(2)
b = 0
sgd = SGD(learning_rate=0.01)
dw, db = sgd.compute_gradients(X, y, w, b)
w_new, b_new = sgd.step(w, b, dw, db)
```

Slide 3: Momentum Optimizer Implementation

The momentum optimizer addresses oscillation issues in standard gradient descent by accumulating a velocity vector in directions of persistent reduction in the objective, enabling faster convergence and better handling of noisy gradients.

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
            
        self.velocity = self.momentum * self.velocity - self.learning_rate * grads
        return params + self.velocity

# Example usage
optimizer = MomentumOptimizer()
params = np.array([1.0, -1.0])
grads = np.array([0.1, -0.1])
for _ in range(3):
    params = optimizer.update(params, grads)
    print(f"Updated params: {params}")
```

Slide 4: Adaptive Gradient (AdaGrad) Implementation

AdaGrad adapts learning rates to parameters, performing smaller updates for frequently updated parameters and larger updates for infrequent ones. This makes it particularly effective for sparse data and different scales of features.

```python
class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accumulated_grads = None
        
    def update(self, params, grads):
        if self.accumulated_grads is None:
            self.accumulated_grads = np.zeros_like(params)
            
        self.accumulated_grads += np.square(grads)
        adjusted_grads = grads / (np.sqrt(self.accumulated_grads) + self.epsilon)
        return params - self.learning_rate * adjusted_grads

# Example usage
optimizer = AdaGrad()
params = np.array([1.0, 2.0])
grads = np.array([0.1, 0.2])
for _ in range(3):
    params = optimizer.update(params, grads)
    print(f"Step result: {params}")
```

Slide 5: RMSprop Optimizer

RMSprop improves upon AdaGrad by using an exponentially decaying average of squared gradients, preventing the learning rate from decreasing too quickly and enabling better navigation of non-convex functions.

```python
class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
        
    def update(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)
            
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * np.square(grads)
        adjusted_grads = grads / (np.sqrt(self.cache) + self.epsilon)
        return params - self.learning_rate * adjusted_grads

# Example usage
optimizer = RMSprop()
params = np.array([0.5, -0.5])
grads = np.array([0.2, -0.1])
for _ in range(3):
    params = optimizer.update(params, grads)
    print(f"Updated parameters: {params}")
```

Slide 6: Adam Optimizer Implementation

Adam combines ideas from RMSprop and momentum, maintaining both first-order moments (mean of gradients) and second-order moments (uncentered variance). This makes it particularly effective for deep learning tasks with noisy or sparse gradients.

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage
optimizer = Adam()
params = np.array([1.0, 2.0])
grads = np.array([0.1, 0.2])
for i in range(3):
    params = optimizer.update(params, grads)
    print(f"Step {i+1} parameters: {params}")
```

Slide 7: Real-World Application - Linear Regression with Different Optimizers

This implementation demonstrates how different optimizers perform on a practical linear regression problem, showing the convergence characteristics and optimization behavior of each method.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Base class for linear regression
class LinearRegression:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.weights = np.random.randn(2, 1)  # [w, b]
        self.loss_history = []
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.weights)
    
    def compute_loss(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
    
    def compute_gradients(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        error = self.predict(X) - y
        return 2/m * X_b.T.dot(error)
    
    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            grads = self.compute_gradients(X, y)
            self.weights = self.optimizer.update(self.weights, grads)
            self.loss_history.append(self.compute_loss(X, y))

# Train models with different optimizers
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Momentum': MomentumOptimizer(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.01)
}

models = {name: LinearRegression(opt) for name, opt in optimizers.items()}
for name, model in models.items():
    model.fit(X, y)
    print(f"{name} final loss: {model.loss_history[-1]:.4f}")
```

Slide 8: Visualizing Optimizer Performance

Understanding how different optimizers converge is crucial for selecting the right optimization strategy. This visualization compares the convergence paths of various optimizers on our linear regression problem.

```python
plt.figure(figsize=(12, 6))
for name, model in models.items():
    plt.plot(model.loss_history, label=name)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Convergence Comparison of Different Optimizers')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Print final weights for each model
for name, model in models.items():
    print(f"\n{name} final weights:")
    print(f"w = {model.weights[1][0]:.4f}")
    print(f"b = {model.weights[0][0]:.4f}")
```

Slide 9: Implementing Learning Rate Scheduling

Learning rate scheduling can significantly improve optimizer performance by adjusting the learning rate during training. This implementation shows common scheduling strategies including step decay and exponential decay.

```python
class LearningRateScheduler:
    def __init__(self, initial_lr=0.1):
        self.initial_lr = initial_lr
        
    def step_decay(self, epoch, drop=0.5, epochs_drop=10):
        """Step decay schedule"""
        return self.initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    
    def exp_decay(self, epoch, k=0.1):
        """Exponential decay schedule"""
        return self.initial_lr * np.exp(-k * epoch)
    
    def cosine_decay(self, epoch, total_epochs):
        """Cosine annealing schedule"""
        return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

# Example usage with Adam optimizer
class AdamWithScheduler(Adam):
    def __init__(self, scheduler, **kwargs):
        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.epoch = 0
        
    def update(self, params, grads):
        self.learning_rate = self.scheduler.step_decay(self.epoch)
        self.epoch += 1
        return super().update(params, grads)

# Test different scheduling strategies
scheduler = LearningRateScheduler(initial_lr=0.1)
epochs = range(100)
lr_schedules = {
    'Step Decay': [scheduler.step_decay(e) for e in epochs],
    'Exp Decay': [scheduler.exp_decay(e) for e in epochs],
    'Cosine Decay': [scheduler.cosine_decay(e, 100) for e in epochs]
}

# Plot learning rate schedules
plt.figure(figsize=(10, 5))
for name, lr_schedule in lr_schedules.items():
    plt.plot(epochs, lr_schedule, label=name)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Scheduling Strategies')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 10: AdaFactor Optimizer Implementation

AdaFactor is a memory-efficient variant of Adam that uses factored second moment estimation to reduce memory usage, making it particularly suitable for training large models with limited computational resources.

```python
class AdaFactor:
    def __init__(self, learning_rate=0.01, beta2=0.999, epsilon1=1e-30, epsilon2=1e-3):
        self.learning_rate = learning_rate
        self.beta2 = beta2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if len(params.shape) < 2:
            # Fall back to Adam-style update for 1D params
            return self._update_1d(params, grads)
            
        if self.v is None:
            self.v = {
                'row': np.zeros(params.shape[0]),
                'col': np.zeros(params.shape[1])
            }
        
        self.t += 1
        decay_rate = 1.0 - self.beta2
        
        # Update row and column factors
        row_square = np.mean(np.square(grads), axis=1)
        col_square = np.mean(np.square(grads), axis=0)
        
        self.v['row'] = self.beta2 * self.v['row'] + decay_rate * row_square
        self.v['col'] = self.beta2 * self.v['col'] + decay_rate * col_square
        
        # Compute scaling factors
        row_scaling = 1.0 / (np.sqrt(self.v['row'] + self.epsilon1))
        col_scaling = 1.0 / (np.sqrt(self.v['col'] + self.epsilon1))
        
        # Compute update
        scaled_grads = grads * np.outer(row_scaling, col_scaling)
        update = self.learning_rate * scaled_grads
        
        return params - update
    
    def _update_1d(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        update = self.learning_rate * grads / (np.sqrt(self.v) + self.epsilon2)
        return params - update

# Example usage
optimizer = AdaFactor()
params = np.random.randn(4, 3)
grads = np.random.randn(4, 3)
updated_params = optimizer.update(params, grads)
print("Original params shape:", params.shape)
print("Updated params shape:", updated_params.shape)
```

Slide 11: Implementing NAdam (Nesterov-accelerated Adaptive Moment Estimation)

NAdam combines Adam with Nesterov momentum, providing better convergence properties by looking ahead when computing gradients. This implementation shows how to incorporate Nesterov momentum into the adaptive learning framework.

```python
class NAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Update momentum and velocity
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # NAdam update incorporating Nesterov momentum
        m_bar = (self.beta1 * m_hat + (1 - self.beta1) * grads) / (1 - self.beta1**self.t)
        
        # Final update
        update = self.learning_rate * m_bar / (np.sqrt(v_hat) + self.epsilon)
        return params - update

# Comparison example
optimizers = {
    'Adam': Adam(),
    'NAdam': NAdam()
}

# Test convergence on a simple quadratic function
def quadratic_function(x):
    return x**2

def quadratic_gradient(x):
    return 2*x

# Compare convergence
starting_point = np.array([2.0])
n_steps = 10

for name, opt in optimizers.items():
    x = starting_point.copy()
    trajectory = [x[0]]
    
    for _ in range(n_steps):
        grad = quadratic_gradient(x)
        x = opt.update(x, grad)
        trajectory.append(x[0])
    
    print(f"\n{name} optimization trajectory:")
    print(trajectory)
```

Slide 12: Real-World Application - Neural Network Training with Multiple Optimizers

This implementation demonstrates how different optimizers perform on a practical neural network classification task, including training metrics and convergence analysis.

```python
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, optimizer):
        self.weights = {
            'W1': np.random.randn(input_size, hidden_size) * 0.01,
            'b1': np.zeros((1, hidden_size)),
            'W2': np.random.randn(hidden_size, output_size) * 0.01,
            'b2': np.zeros((1, output_size))
        }
        self.optimizer = optimizer
        self.loss_history = []
        
    def forward(self, X):
        self.Z1 = np.dot(X, self.weights['W1']) + self.weights['b1']
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights['W2']) + self.weights['b2']
        self.A2 = self._softmax(self.Z2)
        return self.A2
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, X, y, output):
        m = X.shape[0]
        one_hot_y = np.zeros((m, self.weights['W2'].shape[1]))
        one_hot_y[np.arange(m), y] = 1
        
        dZ2 = output - one_hot_y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dZ1 = np.dot(dZ2, self.weights['W2'].T) * (1 - np.power(self.A1, 2))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def train(self, X, y, epochs=100, batch_size=32):
        m = X.shape[0]
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                # Forward pass
                output = self.forward(batch_X)
                
                # Backward pass
                gradients = self.backward(batch_X, batch_y, output)
                
                # Update weights using optimizer
                for key in self.weights:
                    self.weights[key] = self.optimizer.update(
                        self.weights[key], 
                        gradients[key]
                    )
                
            # Calculate loss
            output = self.forward(X)
            loss = -np.mean(np.log(output[np.arange(m), y] + 1e-8))
            self.loss_history.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Generate synthetic classification data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = np.random.randint(0, 3, 1000)

# Train with different optimizers
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Adam': Adam(),
    'NAdam': NAdam()
}

models = {}
for name, opt in optimizers.items():
    print(f"\nTraining with {name}")
    model = SimpleNN(20, 64, 3, opt)
    model.train(X, y, epochs=50)
    models[name] = model
```

Slide 13: Benchmarking Optimizer Performance

A comprehensive comparison of different optimizers' performance metrics, including convergence speed, final loss values, and computational efficiency across different types of neural network architectures.

```python
import time
from collections import defaultdict

class OptimizerBenchmark:
    def __init__(self, optimizers, problem_sizes):
        self.optimizers = optimizers
        self.problem_sizes = problem_sizes
        self.results = defaultdict(dict)
        
    def run_benchmark(self, epochs=50):
        for size in self.problem_sizes:
            print(f"\nBenchmarking problem size: {size}")
            # Generate synthetic data
            X = np.random.randn(1000, size)
            y = np.random.randint(0, 3, 1000)
            
            for opt_name, optimizer in self.optimizers.items():
                print(f"Testing {opt_name}...")
                
                # Initialize model
                model = SimpleNN(size, size*2, 3, optimizer)
                
                # Measure training time
                start_time = time.time()
                model.train(X, y, epochs=epochs)
                training_time = time.time() - start_time
                
                # Store results
                self.results[size][opt_name] = {
                    'final_loss': model.loss_history[-1],
                    'training_time': training_time,
                    'convergence_speed': len([l for l in model.loss_history 
                                           if l <= 1.1 * min(model.loss_history)]),
                    'loss_history': model.loss_history
                }
                
    def print_results(self):
        for size in self.problem_sizes:
            print(f"\nResults for problem size {size}:")
            print("-" * 50)
            for opt_name, metrics in self.results[size].items():
                print(f"\n{opt_name}:")
                print(f"Final loss: {metrics['final_loss']:.4f}")
                print(f"Training time: {metrics['training_time']:.2f} seconds")
                print(f"Epochs to converge: {metrics['convergence_speed']}")

# Run benchmark
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Momentum': MomentumOptimizer(learning_rate=0.01),
    'Adam': Adam(),
    'NAdam': NAdam(),
    'AdaFactor': AdaFactor()
}

problem_sizes = [10, 50, 100]
benchmark = OptimizerBenchmark(optimizers, problem_sizes)
benchmark.run_benchmark()
benchmark.print_results()

# Visualize results
plt.figure(figsize=(15, 5))
for i, size in enumerate(problem_sizes, 1):
    plt.subplot(1, 3, i)
    for opt_name, metrics in benchmark.results[size].items():
        plt.plot(metrics['loss_history'], label=opt_name)
    plt.title(f'Loss Convergence (Size={size})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
plt.tight_layout()
plt.show()
```

Slide 14: Advanced Learning Rate Scheduling with Warmup

Implementation of advanced learning rate scheduling techniques including linear warmup and cosine decay with restarts, which have shown improved performance in training deep neural networks.

```python
class AdvancedLRScheduler:
    def __init__(self, initial_lr=0.001, min_lr=1e-6):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
    def linear_warmup(self, current_step, warmup_steps):
        """Linear warmup followed by constant learning rate"""
        if current_step < warmup_steps:
            return self.initial_lr * (current_step / warmup_steps)
        return self.initial_lr
    
    def cosine_decay_restart(self, current_step, first_decay_steps, 
                            t_mul=2.0, m_mul=0.5):
        """Cosine decay with warm restarts"""
        first_decay_steps = max(1, first_decay_steps)
        alpha = self.min_lr / self.initial_lr
        
        completed_fraction = current_step / first_decay_steps
        
        # Calculate restart cycle
        cycle = np.floor(np.log2(completed_fraction * (t_mul - 1.0) + 1.0))
        current_period = first_decay_steps * t_mul ** (cycle - 1)
        current_step_in_period = current_step - (
            first_decay_steps * (t_mul ** cycle - 1.0) / (t_mul - 1.0)
        )
        
        fraction_in_period = current_step_in_period / current_period
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * fraction_in_period))
        decayed = (1 - alpha) * cosine_decay + alpha
        return self.initial_lr * decayed * (m_mul ** cycle)

# Example usage with Adam optimizer
class AdamWithAdvancedScheduler(Adam):
    def __init__(self, scheduler, warmup_steps=1000, **kwargs):
        super().__init__(**kwargs)
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
    def update(self, params, grads):
        self.step_count += 1
        # Get learning rate from scheduler
        self.learning_rate = self.scheduler.cosine_decay_restart(
            self.step_count, 
            first_decay_steps=5000
        )
        return super().update(params, grads)

# Demonstration
scheduler = AdvancedLRScheduler(initial_lr=0.1)
steps = np.arange(20000)
lr_schedule = [scheduler.cosine_decay_restart(step, first_decay_steps=5000) 
               for step in steps]

plt.figure(figsize=(12, 4))
plt.plot(steps, lr_schedule)
plt.title('Cosine Decay with Warm Restarts')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

*   The Theory Behind Adaptive Learning Rate Methods
    *   [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   An Overview of Gradient Descent Optimization Algorithms
    *   [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   AdaFactor: Adaptive Learning Rates with Sublinear Memory Cost
    *   [https://arxiv.org/abs/1804.04235](https://arxiv.org/abs/1804.04235)
*   On the Convergence of Adam and Beyond
    *   [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   A Method for Solving the Convex Programming Problem with Convergence Rate O(1/k^2)
    *   Search on Google: "Nesterov accelerated gradient method original paper"
*   Understanding Adaptive Methods in Deep Learning
    *   Search: "adaptive optimization methods deep learning survey"

