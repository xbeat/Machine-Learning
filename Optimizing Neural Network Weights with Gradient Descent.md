## Optimizing Neural Network Weights with Gradient Descent
Slide 1: Gradient Descent Fundamentals

Gradient descent forms the backbone of neural network optimization by iteratively adjusting weights to minimize the loss function. The process involves computing partial derivatives with respect to each weight and updating them in the direction that reduces the error.

```python
import numpy as np

def gradient_descent(X, y, weights, learning_rate=0.01, epochs=100):
    for _ in range(epochs):
        # Forward pass
        prediction = np.dot(X, weights)
        
        # Compute gradients
        error = prediction - y
        gradients = np.dot(X.T, error) / len(X)
        
        # Update weights
        weights -= learning_rate * gradients
    return weights

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([5, 7, 9])
weights = np.random.randn(2)
optimized_weights = gradient_descent(X, y, weights)
```

Slide 2: Stochastic Gradient Descent Implementation

Stochastic Gradient Descent (SGD) processes one sample at a time, making it computationally efficient and helping escape local minima. This implementation includes mini-batch processing and momentum for improved convergence.

```python
def sgd_optimizer(X, y, weights, learning_rate=0.01, batch_size=32, momentum=0.9):
    velocity = np.zeros_like(weights)
    indices = np.arange(len(X))
    
    # Mini-batch processing
    np.random.shuffle(indices)
    for i in range(0, len(X), batch_size):
        batch_idx = indices[i:i + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        # Compute gradients
        prediction = np.dot(X_batch, weights)
        error = prediction - y_batch
        gradients = np.dot(X_batch.T, error) / len(X_batch)
        
        # Apply momentum
        velocity = momentum * velocity - learning_rate * gradients
        weights += velocity
    
    return weights
```

Slide 3: Adaptive Learning Rates - Adam Optimizer

Adam combines the benefits of RMSprop and momentum, adapting learning rates for each parameter. This implementation showcases the complete Adam optimization algorithm with bias correction terms.

```python
def adam_optimizer(params, gradients, state, learning_rate=0.001, 
                  beta1=0.9, beta2=0.999, epsilon=1e-8):
    if 'm' not in state:
        state['m'] = np.zeros_like(params)
        state['v'] = np.zeros_like(params)
        state['t'] = 0
    
    state['t'] += 1
    state['m'] = beta1 * state['m'] + (1 - beta1) * gradients
    state['v'] = beta2 * state['v'] + (1 - beta2) * np.square(gradients)
    
    # Bias correction
    m_hat = state['m'] / (1 - beta1**state['t'])
    v_hat = state['v'] / (1 - beta2**state['t'])
    
    # Update parameters
    params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return params, state
```

Slide 4: Weight Initialization Strategies

Proper weight initialization prevents vanishing/exploding gradients and ensures effective training. This implementation demonstrates various initialization methods including Xavier/Glorot and He initialization.

```python
def initialize_weights(layer_dims, initialization='he'):
    weights = {}
    
    for l in range(1, len(layer_dims)):
        if initialization == 'xavier':
            # Xavier/Glorot initialization
            weights[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * \
                              np.sqrt(2.0 / (layer_dims[l] + layer_dims[l-1]))
        elif initialization == 'he':
            # He initialization
            weights[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * \
                              np.sqrt(2.0 / layer_dims[l-1])
        
        weights[f'b{l}'] = np.zeros((layer_dims[l], 1))
    
    return weights
```

Slide 5: Backpropagation Through Time (BPTT)

BPTT is essential for training recurrent neural networks by unrolling the network through time steps and computing gradients. This implementation shows the core mechanism of gradient flow through time.

```python
def bptt(inputs, targets, hidden_state, weights, sequence_length):
    # Forward pass storage
    states = {'h': [np.zeros_like(hidden_state)]}
    losses = []
    
    # Forward pass through time
    for t in range(sequence_length):
        # Current hidden state
        h_t = np.tanh(np.dot(weights['Wx'], inputs[t]) + 
                      np.dot(weights['Wh'], states['h'][t]))
        states['h'].append(h_t)
        
        # Compute loss
        output = np.dot(weights['Wy'], h_t)
        losses.append((output - targets[t])**2)
    
    # Backward pass
    dWx = np.zeros_like(weights['Wx'])
    dWh = np.zeros_like(weights['Wh'])
    dWy = np.zeros_like(weights['Wy'])
    
    for t in reversed(range(sequence_length)):
        # Gradient computation through time
        dh = (1 - states['h'][t+1]**2) * np.dot(weights['Wh'].T, dh)
        dWx += np.dot(dh, inputs[t].T)
        dWh += np.dot(dh, states['h'][t].T)
    
    return {'Wx': dWx, 'Wh': dWh, 'Wy': dWy}, np.mean(losses)
```

Slide 6: Learning Rate Scheduling

Learning rate scheduling dynamically adjusts the learning rate during training to improve convergence and prevent oscillations. This implementation showcases various scheduling strategies including step decay and cosine annealing.

```python
class LRScheduler:
    def __init__(self, initial_lr=0.1):
        self.initial_lr = initial_lr
        
    def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10):
        """Step decay schedule"""
        lr = self.initial_lr * np.power(drop_rate, np.floor(epoch/epochs_drop))
        return lr
        
    def cosine_annealing(self, epoch, total_epochs, eta_min=0):
        """Cosine annealing schedule"""
        lr = eta_min + 0.5 * (self.initial_lr - eta_min) * \
             (1 + np.cos(np.pi * epoch / total_epochs))
        return lr

# Example usage
scheduler = LRScheduler(initial_lr=0.1)
for epoch in range(100):
    current_lr = scheduler.cosine_annealing(epoch, total_epochs=100)
    # Use current_lr in optimizer
```

Slide 7: Weight Regularization Techniques

Regularization prevents overfitting by adding penalty terms to the loss function. This implementation demonstrates L1, L2, and elastic net regularization methods for neural network training.

```python
def compute_regularization(weights, lambda_l1=0.01, lambda_l2=0.01):
    # L1 regularization
    l1_reg = lambda_l1 * sum(np.abs(w).sum() for w in weights.values())
    
    # L2 regularization
    l2_reg = lambda_l2 * sum(np.square(w).sum() for w in weights.values())
    
    # Elastic net combines L1 and L2
    elastic_net = l1_reg + l2_reg
    
    # Compute gradients for regularization
    reg_gradients = {}
    for key, w in weights.items():
        reg_gradients[key] = lambda_l1 * np.sign(w) + 2 * lambda_l2 * w
    
    return elastic_net, reg_gradients
```

Slide 8: Batch Normalization Implementation

Batch normalization stabilizes training by normalizing layer inputs, reducing internal covariate shift. This implementation shows forward and backward passes with running statistics tracking.

```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                              self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                             self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        
        return out
```

Slide 9: Dropout Implementation

Dropout is a powerful regularization technique that randomly deactivates neurons during training. This implementation shows both training and inference phases with proper scaling.

```python
class Dropout:
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate
        self.mask = None
        
    def forward(self, x, training=True):
        if training:
            # Generate dropout mask
            self.mask = (np.random.rand(*x.shape) > self.drop_rate) / \
                       (1 - self.drop_rate)
            return x * self.mask
        return x
    
    def backward(self, grad_output):
        return grad_output * self.mask

# Example usage
layer = Dropout(drop_rate=0.3)
x = np.random.randn(100, 50)  # batch_size=100, features=50
training_output = layer.forward(x, training=True)
inference_output = layer.forward(x, training=False)
```

Slide 10: Momentum-based Weight Updates

Momentum optimization accelerates gradient descent by accumulating past gradients, helping overcome local minima and saddle points. This implementation demonstrates classical momentum and Nesterov accelerated gradient.

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}
        
    def update(self, params, gradients):
        if not self.velocities:
            for key in params:
                self.velocities[key] = np.zeros_like(params[key])
        
        for key in params:
            if self.nesterov:
                # Nesterov momentum update
                self.velocities[key] = self.momentum * self.velocities[key] - \
                                     self.learning_rate * gradients[key]
                params[key] += self.momentum * self.velocities[key] - \
                              self.learning_rate * gradients[key]
            else:
                # Classical momentum update
                self.velocities[key] = self.momentum * self.velocities[key] - \
                                     self.learning_rate * gradients[key]
                params[key] += self.velocities[key]
        
        return params
```

Slide 11: Weight Pruning and Network Compression

Network pruning reduces model size by removing less important connections while maintaining performance. This implementation shows magnitude-based pruning with gradual sparsification.

```python
def prune_network(weights, prune_ratio=0.3):
    pruned_weights = {}
    for layer_name, w in weights.items():
        # Calculate threshold for pruning
        threshold = np.percentile(np.abs(w), prune_ratio * 100)
        
        # Create binary mask for weights
        mask = np.abs(w) > threshold
        
        # Apply mask to weights
        pruned_weights[layer_name] = w * mask
        
        # Calculate sparsity
        sparsity = 1.0 - np.count_nonzero(mask) / mask.size
        print(f"Layer {layer_name} sparsity: {sparsity:.2%}")
    
    return pruned_weights

class GradualPruning:
    def __init__(self, initial_sparsity=0.0, final_sparsity=0.9, 
                 begin_step=0, end_step=100):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        
    def compute_sparsity(self, current_step):
        if current_step < self.begin_step:
            return self.initial_sparsity
        if current_step >= self.end_step:
            return self.final_sparsity
            
        # Linear sparsity schedule
        slope = (self.final_sparsity - self.initial_sparsity) / \
               (self.end_step - self.begin_step)
        current_sparsity = slope * (current_step - self.begin_step) + \
                          self.initial_sparsity
        return current_sparsity
```

Slide 12: Distributed Weight Updates

Distributed training enables parallel weight updates across multiple devices. This implementation shows parameter averaging and gradient accumulation for distributed scenarios.

```python
class DistributedOptimizer:
    def __init__(self, num_workers=4, update_frequency=16):
        self.num_workers = num_workers
        self.update_frequency = update_frequency
        self.accumulated_gradients = None
        self.step_count = 0
        
    def accumulate_gradients(self, worker_gradients):
        """Accumulate gradients from different workers"""
        if self.accumulated_gradients is None:
            self.accumulated_gradients = {k: np.zeros_like(v) 
                                        for k, v in worker_gradients.items()}
        
        # Add worker gradients to accumulator
        for key in worker_gradients:
            self.accumulated_gradients[key] += worker_gradients[key]
            
        self.step_count += 1
        
        # Return True if it's time to update weights
        return self.step_count >= self.update_frequency
        
    def compute_updates(self):
        """Average accumulated gradients and compute updates"""
        if self.accumulated_gradients is None:
            return None
            
        # Average gradients across steps and workers
        averaged_gradients = {
            k: v / (self.step_count * self.num_workers)
            for k, v in self.accumulated_gradients.items()
        }
        
        # Reset accumulators
        self.accumulated_gradients = None
        self.step_count = 0
        
        return averaged_gradients
```

Slide 13: Adaptive Weight Clipping

Adaptive weight clipping prevents extreme weight values while maintaining network stability. This implementation includes dynamic threshold computation and gradient rescaling mechanisms.

```python
class AdaptiveClipper:
    def __init__(self, init_clip_value=1.0, adaptation_rate=0.01):
        self.clip_value = init_clip_value
        self.adaptation_rate = adaptation_rate
        self.running_max = 0.0
        
    def clip_gradients(self, gradients):
        clipped_grads = {}
        grad_norms = []
        
        # Compute gradient norms
        for key, grad in gradients.items():
            grad_norm = np.sqrt(np.sum(np.square(grad)))
            grad_norms.append(grad_norm)
            
        # Update running max
        current_max = np.max(grad_norms)
        self.running_max = (1 - self.adaptation_rate) * self.running_max + \
                          self.adaptation_rate * current_max
                          
        # Adjust clip value
        self.clip_value = min(self.clip_value, self.running_max)
        
        # Clip gradients
        for key, grad in gradients.items():
            grad_norm = np.sqrt(np.sum(np.square(grad)))
            if grad_norm > self.clip_value:
                clipped_grads[key] = grad * (self.clip_value / grad_norm)
            else:
                clipped_grads[key] = grad
                
        return clipped_grads
```

Slide 14: Layer-wise Adaptive Learning Rates

Layer-wise adaptation of learning rates improves training stability by considering the different scales of gradients across layers. This implementation shows per-layer learning rate adjustment.

```python
class LayerAdaptiveLR:
    def __init__(self, base_lr=0.01, beta=0.999):
        self.base_lr = base_lr
        self.beta = beta
        self.layer_stats = {}
        
    def compute_lr(self, layer_name, gradient):
        if layer_name not in self.layer_stats:
            self.layer_stats[layer_name] = {
                'square_avg': np.zeros_like(gradient),
                'step': 0
            }
        
        stats = self.layer_stats[layer_name]
        stats['step'] += 1
        
        # Update running average of squared gradients
        stats['square_avg'] = self.beta * stats['square_avg'] + \
                             (1 - self.beta) * np.square(gradient)
        
        # Bias correction
        square_avg_corrected = stats['square_avg'] / \
                              (1 - self.beta ** stats['step'])
        
        # Compute adaptive learning rate
        adaptive_lr = self.base_lr / (np.sqrt(square_avg_corrected) + 1e-8)
        
        return adaptive_lr
        
    def apply_gradients(self, params, gradients):
        updates = {}
        for name in params:
            lr = self.compute_lr(name, gradients[name])
            updates[name] = params[name] - lr * gradients[name]
        return updates
```

Slide 15: Additional Resources

*   Original Adam Optimizer Paper
    *   [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   Batch Normalization: Accelerating Deep Network Training
    *   [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
*   Deep Learning Best Practices: Weight Initialization
    *   [https://arxiv.org/abs/1704.08863](https://arxiv.org/abs/1704.08863)
*   Network Pruning Research
    *   [https://arxiv.org/abs/1506.02626](https://arxiv.org/abs/1506.02626)
*   Distributed Training Strategies
    *   [https://arxiv.org/abs/1706.02677](https://arxiv.org/abs/1706.02677)
*   Adaptive Learning Rate Methods Review
    *   For detailed review, search "Adaptive Learning Rate Methods for Deep Learning" on Google Scholar

