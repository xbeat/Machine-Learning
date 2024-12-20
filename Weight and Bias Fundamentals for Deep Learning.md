## Weight and Bias Fundamentals for Deep Learning
Slide 1: Understanding Neural Network Weight Initialization

Neural network weights and biases are critical parameters that determine how input signals are transformed through the network. Proper initialization of these parameters is crucial for successful model training, as it affects gradient flow and convergence speed during backpropagation.

```python
import numpy as np

def initialize_weights(input_size, hidden_size):
    # Xavier/Glorot initialization for better gradient flow
    weights = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    biases = np.zeros(hidden_size)
    return weights, biases

# Example usage
input_dim, hidden_dim = 784, 256
W, b = initialize_weights(input_dim, hidden_dim)
print(f"Weight matrix shape: {W.shape}, mean: {W.mean():.4f}, std: {W.std():.4f}")
print(f"Bias vector shape: {b.shape}, mean: {b.mean():.4f}, std: {b.std():.4f}")
```

Slide 2: Weight Update Mechanisms in Neural Networks

During training, weights and biases are updated using gradients computed through backpropagation. The update process involves calculating partial derivatives with respect to each parameter and applying optimization algorithms to minimize the loss function.

```python
def update_parameters(weights, biases, gradients, learning_rate=0.01):
    """
    Basic gradient descent update for neural network parameters
    """
    # Unpack gradients
    dW, db = gradients
    
    # Update weights and biases
    weights = weights - learning_rate * dW
    biases = biases - learning_rate * db
    
    return weights, biases

# Example gradients
dW = np.random.randn(784, 256) * 0.01
db = np.random.randn(256) * 0.01

# Perform update
W_updated, b_updated = update_parameters(W, b, (dW, db))
print(f"Weight update magnitude: {np.mean(np.abs(W - W_updated)):.6f}")
```

Slide 3: Bias Terms and Their Impact

Bias terms in neural networks act as learnable offsets that allow the activation functions to shift their output distributions. They provide flexibility in fitting the underlying data distribution by allowing neurons to activate even when input features are zero.

```python
def add_bias_activation(inputs, weights, biases, activation='relu'):
    """
    Demonstrates the role of bias in neural network computations
    """
    # Linear transformation with bias
    z = np.dot(inputs, weights) + biases
    
    # Apply activation function
    if activation == 'relu':
        output = np.maximum(0, z)
    elif activation == 'sigmoid':
        output = 1 / (1 + np.exp(-z))
    
    return output, z

# Example with and without bias
x = np.random.randn(1, 784)
out_with_bias, z_with_bias = add_bias_activation(x, W, b)
out_without_bias, z_without_bias = add_bias_activation(x, W, np.zeros_like(b))

print(f"Activation with bias: {np.mean(out_with_bias):.4f}")
print(f"Activation without bias: {np.mean(out_without_bias):.4f}")
```

Slide 4: Mathematical Foundations of Weights and Biases

The mathematical relationship between inputs, weights, and biases forms the foundation of neural networks. Understanding these relationships is crucial for implementing effective learning algorithms and optimization strategies.

```python
# Mathematical formulas in LaTeX notation
"""
Forward propagation:
$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g(z^{[l]})$$

Backpropagation:
$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial W^{[l]}}$$
$$\frac{\partial L}{\partial b^{[l]}} = \frac{\partial L}{\partial z^{[l]}}$$

Weight update:
$$W^{[l]} = W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$$
$$b^{[l]} = b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}$$
"""
```

Slide 5: Implementation of Weight Initialization Techniques

Multiple weight initialization strategies have been developed to address various challenges in deep learning. Each technique aims to maintain proper gradient flow and prevent issues like vanishing or exploding gradients during training.

```python
def initialize_layer_weights(input_size, output_size, method='glorot'):
    if method == 'glorot':
        # Glorot/Xavier initialization
        limit = np.sqrt(6 / (input_size + output_size))
        weights = np.random.uniform(-limit, limit, (input_size, output_size))
    elif method == 'he':
        # He initialization for ReLU networks
        std = np.sqrt(2 / input_size)
        weights = np.random.randn(input_size, output_size) * std
    elif method == 'lecun':
        # LeCun initialization
        std = np.sqrt(1 / input_size)
        weights = np.random.randn(input_size, output_size) * std
        
    biases = np.zeros(output_size)
    return weights, biases

# Compare different initializations
methods = ['glorot', 'he', 'lecun']
for method in methods:
    W, b = initialize_layer_weights(784, 256, method)
    print(f"{method} initialization - Weight std: {W.std():.4f}")
```

Slide 6: Advanced Weight Update with Momentum

Momentum helps accelerate gradient descent in the relevant direction and dampens oscillations. This technique maintains a moving average of past gradients to update weights more effectively, particularly useful for navigating through areas of high curvature.

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_W = None
        self.velocity_b = None
    
    def update(self, weights, biases, gradients):
        dW, db = gradients
        
        # Initialize velocity if first update
        if self.velocity_W is None:
            self.velocity_W = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(biases)
        
        # Update velocities
        self.velocity_W = self.momentum * self.velocity_W - self.learning_rate * dW
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * db
        
        # Update parameters
        weights = weights + self.velocity_W
        biases = biases + self.velocity_b
        
        return weights, biases

# Example usage
optimizer = MomentumOptimizer()
for _ in range(3):  # Simulate few updates
    dW = np.random.randn(784, 256) * 0.01
    db = np.random.randn(256) * 0.01
    W, b = optimizer.update(W, b, (dW, db))
    print(f"Update step - Weight velocity mean: {optimizer.velocity_W.mean():.6f}")
```

Slide 7: Weight Regularization Techniques

Regularization prevents overfitting by adding penalties on weights during training. L1 and L2 regularization are common techniques that encourage smaller weight values, leading to simpler models with better generalization capabilities.

```python
def compute_regularization(weights, lambda_param=0.01, method='l2'):
    """
    Compute regularization loss and gradients
    """
    if method == 'l2':
        # L2 regularization
        reg_loss = 0.5 * lambda_param * np.sum(weights ** 2)
        reg_grad = lambda_param * weights
    elif method == 'l1':
        # L1 regularization
        reg_loss = lambda_param * np.sum(np.abs(weights))
        reg_grad = lambda_param * np.sign(weights)
    
    return reg_loss, reg_grad

# Compare regularization methods
W = np.random.randn(784, 256) * 0.1
methods = ['l1', 'l2']
for method in methods:
    loss, grad = compute_regularization(W, method=method)
    print(f"{method} regularization - Loss: {loss:.4f}, Gradient mean: {grad.mean():.4f}")
```

Slide 8: Weight Pruning and Sparsification

Weight pruning reduces model complexity by identifying and removing less important weights. This technique is crucial for model compression and can improve inference speed while maintaining performance within acceptable bounds.

```python
def prune_weights(weights, threshold_percentile=90):
    """
    Prune weights below certain magnitude threshold
    """
    # Calculate magnitude threshold
    threshold = np.percentile(np.abs(weights), threshold_percentile)
    
    # Create binary mask
    mask = np.abs(weights) >= threshold
    
    # Apply mask to weights
    pruned_weights = weights * mask
    
    # Calculate sparsity
    sparsity = 1.0 - np.count_nonzero(mask) / mask.size
    
    return pruned_weights, sparsity, mask

# Example usage
W_dense = np.random.randn(784, 256) * 0.1
thresholds = [50, 70, 90]
for thresh in thresholds:
    W_pruned, sparsity, _ = prune_weights(W_dense, thresh)
    print(f"Threshold {thresh}% - Sparsity: {sparsity:.4f}")
```

Slide 9: Adaptive Learning Rate for Weights

Adaptive learning rate methods automatically adjust the learning rate for each weight based on historical gradient information. This leads to more efficient training by optimizing the update step size for different parameters.

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None
        self.t = 0
        
    def update(self, weights, biases, gradients):
        self.t += 1
        dW, db = gradients
        
        # Initialize moments if first update
        if self.m_W is None:
            self.m_W = np.zeros_like(weights)
            self.v_W = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)
        
        # Update moments for weights
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
        
        # Update moments for biases
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        # Update parameters
        weights = weights - self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        biases = biases - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return weights, biases

# Example usage
optimizer = AdamOptimizer()
for i in range(3):
    dW = np.random.randn(784, 256) * 0.01
    db = np.random.randn(256) * 0.01
    W, b = optimizer.update(W, b, (dW, db))
    print(f"Step {i+1} - Weight update mean: {np.mean(np.abs(dW)):.6f}")
```

Slide 10: Real-world Implementation: MNIST Classifier

A complete implementation of a neural network for MNIST digit classification demonstrating the practical application of weights and biases in a real-world scenario, including initialization, training, and evaluation phases.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class MNISTClassifier:
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        # Initialize weights using He initialization
        self.W1, self.b1 = self._init_weights(input_size, hidden_size)
        self.W2, self.b2 = self._init_weights(hidden_size, output_size)
        self.optimizer = AdamOptimizer()
        
    def _init_weights(self, input_size, output_size):
        W = np.random.randn(input_size, output_size) * np.sqrt(2./input_size)
        b = np.zeros(output_size)
        return W, b
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y, batch_size):
        dz2 = self.probs
        dz2[range(batch_size), y] -= 1
        dz2 /= batch_size
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)  # ReLU derivative
        
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        return [(dW1, db1), (dW2, db2)]
    
    def train_step(self, X_batch, y_batch):
        batch_size = len(X_batch)
        
        # Forward pass
        probs = self.forward(X_batch)
        
        # Backward pass
        gradients = self.backward(X_batch, y_batch, batch_size)
        
        # Update parameters
        self.W1, self.b1 = self.optimizer.update(self.W1, self.b1, gradients[0])
        self.W2, self.b2 = self.optimizer.update(self.W2, self.b2, gradients[1])

# Load and preprocess MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values
y = y.astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training example
model = MNISTClassifier()
batch_size = 128
n_batches = len(X_train) // batch_size

# Train for one epoch
for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]
    model.train_step(X_batch, y_batch)
```

Slide 11: Results for MNIST Classifier Implementation

The performance metrics and analysis of the MNIST classifier implementation, demonstrating the effects of proper weight initialization and optimization.

```python
def evaluate_model(model, X, y):
    probs = model.forward(X)
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy, predictions

# Evaluate on test set
test_accuracy, test_predictions = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Weight statistics
print("\nWeight Statistics:")
print(f"Layer 1 - Mean: {model.W1.mean():.4f}, Std: {model.W1.std():.4f}")
print(f"Layer 2 - Mean: {model.W2.mean():.4f}, Std: {model.W2.std():.4f}")

# Bias statistics
print("\nBias Statistics:")
print(f"Layer 1 - Mean: {model.b1.mean():.4f}, Std: {model.b1.std():.4f}")
print(f"Layer 2 - Mean: {model.b2.mean():.4f}, Std: {model.b2.std():.4f}")
```

Slide 12: Weight Visualization Techniques

Visualizing weights and their distributions provides insights into the learning process and helps identify potential issues in the network's parameter space.

```python
import matplotlib.pyplot as plt

def visualize_weights(model):
    plt.figure(figsize=(15, 5))
    
    # Plot weight distributions
    plt.subplot(121)
    plt.hist(model.W1.ravel(), bins=50, alpha=0.5, label='Layer 1')
    plt.hist(model.W2.ravel(), bins=50, alpha=0.5, label='Layer 2')
    plt.title('Weight Distributions')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot first layer weights as images
    plt.subplot(122)
    w_img = model.W1[:, :25].reshape(5, 5, 28, 28)
    w_img = w_img.transpose(0, 2, 1, 3).reshape(140, 140)
    plt.imshow(w_img, cmap='viridis')
    plt.title('First Layer Weight Patterns')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize weights
visualize_weights(model)
```

Slide 13: Dynamic Weight Pruning Strategy

A sophisticated approach to weight pruning that dynamically adjusts the sparsification threshold during training, maintaining model performance while progressively reducing parameter count through structured and unstructured pruning techniques.

```python
class DynamicPruningOptimizer:
    def __init__(self, initial_sparsity=0.0, target_sparsity=0.9, prune_steps=10):
        self.current_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.prune_steps = prune_steps
        self.step_size = (target_sparsity - initial_sparsity) / prune_steps
        self.mask = None
        self.step_count = 0
        
    def calculate_threshold(self, weights):
        """Calculate threshold for current sparsity level"""
        return np.percentile(np.abs(weights), 
                           self.current_sparsity * 100)
    
    def update_mask(self, weights):
        """Update binary mask based on current sparsity"""
        threshold = self.calculate_threshold(weights)
        return np.abs(weights) >= threshold
    
    def apply_pruning(self, weights, gradients):
        self.step_count += 1
        
        # Update sparsity level
        if self.step_count % (self.prune_steps // 10) == 0:
            self.current_sparsity = min(
                self.current_sparsity + self.step_size,
                self.target_sparsity
            )
        
        # Initialize or update mask
        if self.mask is None:
            self.mask = np.ones_like(weights)
        else:
            self.mask = self.update_mask(weights)
        
        # Apply mask to weights and gradients
        pruned_weights = weights * self.mask
        pruned_gradients = gradients * self.mask
        
        return pruned_weights, pruned_gradients, self.mask

# Example usage with monitoring
pruning_optimizer = DynamicPruningOptimizer()
W = np.random.randn(784, 256) * 0.1
history = []

for step in range(20):
    # Simulate gradient update
    dW = np.random.randn(784, 256) * 0.01
    
    # Apply pruning
    W_pruned, dW_pruned, mask = pruning_optimizer.apply_pruning(W, dW)
    
    # Update weights (simplified)
    W = W_pruned - 0.01 * dW_pruned
    
    # Record statistics
    sparsity = 1.0 - np.count_nonzero(mask) / mask.size
    history.append({
        'step': step,
        'sparsity': sparsity,
        'mean_weight': np.mean(np.abs(W_pruned))
    })
    
    if step % 5 == 0:
        print(f"Step {step}: Sparsity = {sparsity:.4f}, "
              f"Mean Weight = {np.mean(np.abs(W_pruned)):.4f}")
```

Slide 14: Advanced Bias Initialization Techniques

Understanding and implementing sophisticated bias initialization strategies can significantly impact model convergence and final performance, especially in deep architectures with complex activation functions.

```python
def advanced_bias_initialization(layer_sizes, activation_types):
    """
    Initialize biases based on activation function characteristics
    and network architecture
    """
    biases = []
    for i, (input_size, output_size) in enumerate(zip(layer_sizes[:-1], 
                                                     layer_sizes[1:])):
        if activation_types[i] == 'relu':
            # He initialization-based bias for ReLU
            bias = np.random.randn(output_size) * np.sqrt(2.0/input_size)
        elif activation_types[i] == 'sigmoid':
            # Compensate for sigmoid's saturation regions
            bias = np.zeros(output_size) + 0.1
        elif activation_types[i] == 'tanh':
            # Tanh-specific initialization
            limit = np.sqrt(6.0 / (input_size + output_size))
            bias = np.random.uniform(-limit, limit, output_size)
        else:  # Linear or others
            bias = np.zeros(output_size)
        biases.append(bias)
    return biases

# Example architecture
layer_sizes = [784, 256, 128, 10]
activation_types = ['relu', 'relu', 'sigmoid']

# Initialize biases
biases = advanced_bias_initialization(layer_sizes, activation_types)

# Analysis of initialization
for i, bias in enumerate(biases):
    print(f"Layer {i+1} ({activation_types[i]}):")
    print(f"  Mean: {bias.mean():.4f}")
    print(f"  Std: {bias.std():.4f}")
    print(f"  Range: [{bias.min():.4f}, {bias.max():.4f}]")
```

Slide 15: Additional Resources

*   "Weight Initialization in Deep Neural Networks: A Survey"
    *   [https://arxiv.org/abs/2007.07587](https://arxiv.org/abs/2007.07587)
*   "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    *   [https://proceedings.mlr.press/v9/glorot10a.html](https://proceedings.mlr.press/v9/glorot10a.html)
*   "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    *   [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
*   "An Empirical Study of Weight Initialization and Pruning in Deep Neural Networks"
    *   [https://arxiv.org/abs/2010.07150](https://arxiv.org/abs/2010.07150)
*   "Fixup Initialization: Residual Learning Without Normalization"
    *   [https://arxiv.org/abs/1901.09321](https://arxiv.org/abs/1901.09321)

