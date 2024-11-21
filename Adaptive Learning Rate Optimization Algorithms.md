## Adaptive Learning Rate Optimization Algorithms
Slide 1: Introduction to Adaptive Learning Rate Algorithms

Adaptive learning rate optimization algorithms automatically adjust the learning rate parameter during training to achieve better convergence. These methods maintain individual learning rates for each parameter, modifying them based on historical gradient information.

```python
# Base class for adaptive optimizers
class AdaptiveOptimizer:
    def __init__(self, params, learning_rate=0.01):
        self.params = params
        self.lr = learning_rate
        self.iterations = 0
        
    def step(self):
        self.iterations += 1
        raise NotImplementedError
```

Slide 2: AdaGrad Implementation

AdaGrad adapts learning rates by scaling them inversely proportional to the square root of the sum of all past squared gradients. This allows the algorithm to handle sparse data efficiently while automatically decreasing learning rates over time.

```python
import numpy as np

class AdaGrad(AdaptiveOptimizer):
    def __init__(self, params, learning_rate=0.01, eps=1e-8):
        super().__init__(params, learning_rate)
        self.eps = eps
        self.squared_gradients = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, gradients):
        for param_name in self.params:
            self.squared_gradients[param_name] += gradients[param_name] ** 2
            adjusted_lr = self.lr / (np.sqrt(self.squared_gradients[param_name]) + self.eps)
            self.params[param_name] -= adjusted_lr * gradients[param_name]
```

Slide 3: RMSprop Implementation

RMSprop addresses AdaGrad's rapidly diminishing learning rates by using an exponentially decaying average of squared gradients. This modification allows the algorithm to maintain steady learning rates for non-convex problems.

```python
class RMSprop(AdaptiveOptimizer):
    def __init__(self, params, learning_rate=0.01, decay_rate=0.9, eps=1e-8):
        super().__init__(params, learning_rate)
        self.decay_rate = decay_rate
        self.eps = eps
        self.squared_gradients = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, gradients):
        for param_name in self.params:
            self.squared_gradients[param_name] = (
                self.decay_rate * self.squared_gradients[param_name] + 
                (1 - self.decay_rate) * gradients[param_name]**2
            )
            adjusted_lr = self.lr / (np.sqrt(self.squared_gradients[param_name]) + self.eps)
            self.params[param_name] -= adjusted_lr * gradients[param_name]
```

Slide 4: Adam Implementation Part 1

Adam combines ideas from RMSprop and momentum optimization, maintaining both first and second moment estimates of the gradients. This dual averaging provides more reliable parameter updates and faster convergence.

```python
class Adam(AdaptiveOptimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # First moment
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # Second moment
```

Slide 5: Adam Implementation Part 2: Step Method

The step method in Adam implements bias correction for both first and second moments, ensuring unbiased estimates throughout training, particularly in the early iterations where bias correction is most critical.

```python
    def step(self, gradients):
        for param_name in self.params:
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradients[param_name]
            # Update biased second moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * gradients[param_name]**2
            
            # Bias correction
            m_corrected = self.m[param_name] / (1 - self.beta1 ** (self.iterations + 1))
            v_corrected = self.v[param_name] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Update parameters
            self.params[param_name] -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)
```

Slide 6: Real-world Example: Linear Regression with Adam

Implementation of linear regression using Adam optimizer demonstrates practical application in a simple machine learning context. The example includes data generation, model training, and performance monitoring.

```python
import numpy as np
np.random.seed(42)

# Generate synthetic data
X = np.random.randn(1000, 5)
true_weights = np.array([1., 2., -0.5, 3., -2.])
y = np.dot(X, true_weights) + np.random.randn(1000) * 0.1

# Initialize model parameters
params = {'weights': np.random.randn(5) * 0.1}
optimizer = Adam(params)

# Training loop
losses = []
for epoch in range(100):
    # Forward pass
    y_pred = np.dot(X, params['weights'])
    loss = np.mean((y_pred - y)**2)
    losses.append(loss)
    
    # Backward pass
    gradients = {'weights': 2 * np.dot(X.T, (y_pred - y)) / len(y)}
    optimizer.step(gradients)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
```

Slide 7: AdamW Implementation

AdamW modifies the original Adam optimizer by implementing proper weight decay regularization, decoupling it from the gradient update. This modification helps prevent the adaptive learning rate from interfering with the regularization effect.

```python
class AdamW(AdaptiveOptimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        
    def step(self, gradients):
        for param_name in self.params:
            # Apply weight decay
            self.params[param_name] -= self.lr * self.weight_decay * self.params[param_name]
            
            # Update moments
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradients[param_name]
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * gradients[param_name]**2
            
            # Bias correction
            m_corrected = self.m[param_name] / (1 - self.beta1 ** (self.iterations + 1))
            v_corrected = self.v[param_name] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Update parameters
            self.params[param_name] -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)
```

Slide 8: Nadam (Nesterov-accelerated Adaptive Moment Estimation)

Nadam combines Adam with Nesterov momentum, providing stronger convergence properties. It applies the momentum update to the gradient before computing the gradient, resulting in more accurate parameter updates.

```python
class Nadam(AdaptiveOptimizer):
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, gradients):
        for param_name in self.params:
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradients[param_name]
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * gradients[param_name]**2
            
            m_corrected = self.m[param_name] / (1 - self.beta1 ** (self.iterations + 1))
            v_corrected = self.v[param_name] / (1 - self.beta2 ** (self.iterations + 1))
            
            # Nesterov momentum update
            m_nesterov = (self.beta1 * m_corrected + (1 - self.beta1) * gradients[param_name]) / (1 - self.beta1)
            
            self.params[param_name] -= self.lr * m_nesterov / (np.sqrt(v_corrected) + self.eps)
```

Slide 9: Real-world Example: Neural Network Training

A practical implementation of a neural network trained with various adaptive optimizers, demonstrating their effectiveness in deep learning applications with non-convex optimization problems.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.params = {
            'W1': np.random.randn(input_size, hidden_size) * 0.01,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, output_size) * 0.01,
            'b2': np.zeros(output_size)
        }
        
    def forward(self, X):
        z1 = np.dot(X, self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, self.params['W2']) + self.params['b2']
        return z2, (a1, z1)
    
    def train(self, X, y, optimizer, epochs=100, batch_size=32):
        losses = []
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                # Forward pass
                output, cache = self.forward(X_batch)
                loss = np.mean((output - y_batch)**2)
                
                # Backward pass
                gradients = self.backward(X_batch, y_batch, cache)
                optimizer.step(gradients)
                
            losses.append(loss)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return losses
```

Slide 10: Learning Rate Scheduling

Learning rate scheduling dynamically adjusts the learning rate during training according to predefined rules or adaptive criteria, helping optimize convergence and prevent overshooting of optimal parameters.

```python
class LearningRateScheduler:
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
        
    def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10.0):
        """Step decay schedule"""
        return self.initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
    
    def exponential_decay(self, epoch, decay_rate=0.95):
        """Exponential decay schedule"""
        return self.initial_lr * np.power(decay_rate, epoch)
    
    def cosine_decay(self, epoch, total_epochs):
        """Cosine annealing schedule"""
        return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))

# Usage example
scheduler = LearningRateScheduler(initial_lr=0.1)
for epoch in range(100):
    current_lr = scheduler.cosine_decay(epoch, total_epochs=100)
    # Update optimizer's learning rate
    optimizer.lr = current_lr
```

Slide 11: Adaptive Learning Rate Comparison

This implementation provides a comprehensive comparison between different adaptive optimization algorithms on a common problem, showing their convergence characteristics and performance differences under controlled conditions.

```python
import numpy as np
from time import time

def compare_optimizers(X, y, optimizers, epochs=100):
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        start_time = time()
        model = NeuralNetwork(X.shape[1], 64, y.shape[1])
        
        # Training history
        losses = []
        for epoch in range(epochs):
            output, cache = model.forward(X)
            loss = np.mean((output - y)**2)
            losses.append(loss)
            
            gradients = model.backward(X, y, cache)
            optimizer.step(gradients)
        
        results[opt_name] = {
            'losses': losses,
            'time': time() - start_time,
            'final_loss': losses[-1]
        }
    
    return results

# Example usage
optimizers = {
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001),
    'AdaGrad': AdaGrad(learning_rate=0.01),
    'AdamW': AdamW(learning_rate=0.001),
    'Nadam': Nadam(learning_rate=0.001)
}

comparison_results = compare_optimizers(X_train, y_train, optimizers)
```

Slide 12: Gradient Clipping Implementation

Gradient clipping prevents exploding gradients by scaling down gradient norms that exceed a threshold, essential for training deep networks with adaptive optimizers. This implementation shows both norm and value clipping approaches.

```python
class GradientClipper:
    def __init__(self, threshold=1.0):
        self.threshold = threshold
    
    def clip_by_norm(self, gradients):
        """Clips gradients based on global norm"""
        total_norm = np.sqrt(sum(
            np.sum(np.square(g)) for g in gradients.values()
        ))
        
        clip_coef = self.threshold / (total_norm + 1e-6)
        if clip_coef < 1:
            for k in gradients:
                gradients[k] = gradients[k] * clip_coef
        return gradients
    
    def clip_by_value(self, gradients):
        """Clips gradient values element-wise"""
        for k in gradients:
            np.clip(gradients[k], 
                   -self.threshold, 
                   self.threshold, 
                   out=gradients[k])
        return gradients

# Integration with optimizer
class AdaptiveOptimizerWithClipping(AdaptiveOptimizer):
    def __init__(self, params, learning_rate=0.001, clip_threshold=1.0):
        super().__init__(params, learning_rate)
        self.clipper = GradientClipper(clip_threshold)
    
    def step(self, gradients):
        # Clip gradients before applying updates
        clipped_gradients = self.clipper.clip_by_norm(gradients)
        # Continue with normal optimization step
        super().step(clipped_gradients)
```

Slide 13: Learning Rate Warm-up Strategy

Learning rate warm-up gradually increases the learning rate from a small initial value, helping stabilize training in the early stages. This implementation provides various warm-up schedules for adaptive optimizers.

```python
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.initial_lr = optimizer.lr
    
    def linear_warmup(self, epoch):
        """Linear learning rate warm-up"""
        if epoch >= self.warmup_epochs:
            return self.target_lr
        return self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
    
    def exponential_warmup(self, epoch):
        """Exponential learning rate warm-up"""
        if epoch >= self.warmup_epochs:
            return self.target_lr
        return self.initial_lr * (self.target_lr / self.initial_lr) ** (epoch / self.warmup_epochs)
    
    def step(self, epoch):
        self.optimizer.lr = self.linear_warmup(epoch)

# Usage example
optimizer = Adam(params, learning_rate=1e-6)
scheduler = WarmupScheduler(optimizer, warmup_epochs=5, target_lr=0.001)

for epoch in range(num_epochs):
    scheduler.step(epoch)
    # Training loop continues...
```

Slide 14: Additional Resources

*   "Adam: A Method for Stochastic Optimization"
    *   [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   "On the Convergence of Adam and Beyond"
    *   [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   "Decoupled Weight Decay Regularization"
    *   [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
*   "Adaptive Learning Rate Methods: A Survey"
    *   search on Google Scholar for recent surveys on adaptive optimization methods
*   "An Overview of Gradient Descent Optimization Algorithms"
    *   [https://ruder.io/optimizing-gradient-descent/](https://ruder.io/optimizing-gradient-descent/)

