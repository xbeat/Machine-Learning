## Initializing Weights in Deep Learning
Slide 1: Xavier Weight Initialization

Xavier initialization is a crucial method for deep neural networks that helps maintain consistent variance of activations across layers. It sets initial weights by drawing from a distribution with zero mean and variance scaled by the number of input and output connections.

```python
import numpy as np

def xavier_init(n_inputs, n_outputs):
    # Calculate variance based on Xavier/Glorot formula
    variance = 2.0 / (n_inputs + n_outputs)
    stddev = np.sqrt(variance)
    # Initialize weights from normal distribution
    weights = np.random.normal(0, stddev, (n_inputs, n_outputs))
    return weights

# Example usage
layer_weights = xavier_init(784, 256)
print(f"Weight Statistics:\nMean: {np.mean(layer_weights):.6f}")
print(f"Variance: {np.var(layer_weights):.6f}")
```

Slide 2: He (Kaiming) Initialization

He initialization adapts Xavier's approach specifically for ReLU activation functions, accounting for the fact that ReLU sets negative values to zero. It uses a scaling factor of sqrt(2/n) where n is the number of input connections.

```python
def he_init(n_inputs, n_outputs):
    # Calculate standard deviation based on He formula
    stddev = np.sqrt(2.0 / n_inputs)
    # Initialize weights from normal distribution
    weights = np.random.normal(0, stddev, (n_inputs, n_outputs))
    return weights

# Example usage
layer_weights = he_init(784, 256)
print(f"Weight Statistics:\nMean: {np.mean(layer_weights):.6f}")
print(f"Variance: {np.var(layer_weights):.6f}")
```

Slide 3: Uniform Distribution Initialization

Uniform initialization distributes weights uniformly within a specified range. This method provides an alternative to normal distribution-based approaches, particularly useful when dealing with specific architectural requirements or activation functions.

```python
def uniform_init(n_inputs, n_outputs, scale=0.05):
    weights = np.random.uniform(
        low=-scale,
        high=scale,
        size=(n_inputs, n_outputs)
    )
    return weights

# Example usage
layer_weights = uniform_init(784, 256)
print(f"Weight Statistics:\nMean: {np.mean(layer_weights):.6f}")
print(f"Variance: {np.var(layer_weights):.6f}")
```

Slide 4: Lecun Initialization

LeCun initialization is designed for networks using tanh activation functions. It helps maintain the variance of the activations and gradients across layers, preventing vanishing or exploding gradients during training.

```python
def lecun_init(n_inputs, n_outputs):
    # Calculate standard deviation based on LeCun formula
    stddev = np.sqrt(1.0 / n_inputs)
    # Initialize weights from normal distribution
    weights = np.random.normal(0, stddev, (n_inputs, n_outputs))
    return weights

# Example usage
layer_weights = lecun_init(784, 256)
print(f"Weight Statistics:\nMean: {np.mean(layer_weights):.6f}")
print(f"Variance: {np.var(layer_weights):.6f}")
```

Slide 5: Orthogonal Initialization

Orthogonal initialization creates weight matrices with orthogonal vectors, which can help with gradient flow in deep networks. This method is particularly useful for RNNs and helps maintain consistent gradients during backpropagation.

```python
def orthogonal_init(n_inputs, n_outputs):
    # Generate random matrix
    random_matrix = np.random.randn(n_inputs, n_outputs)
    # Compute QR factorization
    q, r = np.linalg.qr(random_matrix)
    # Make orthogonal matrix with desired shape
    weights = q if n_inputs >= n_outputs else q.T
    return weights[:n_inputs, :n_outputs]

# Example usage
layer_weights = orthogonal_init(784, 256)
print(f"Weight Statistics:\nMean: {np.mean(layer_weights):.6f}")
print(f"Variance: {np.var(layer_weights):.6f}")
```

Slide 6: Custom Weight Distribution Implementation

Creating custom weight distributions allows for precise control over the initialization process. This implementation demonstrates how to create a flexible initialization scheme that can accommodate different statistical requirements.

```python
def custom_weight_init(n_inputs, n_outputs, distribution='normal', 
                      mean=0.0, scale=0.01, bounds=None):
    if distribution == 'normal':
        weights = np.random.normal(mean, scale, (n_inputs, n_outputs))
    elif distribution == 'uniform':
        if bounds is None:
            bounds = (-scale, scale)
        weights = np.random.uniform(bounds[0], bounds[1], (n_inputs, n_outputs))
    elif distribution == 'truncated_normal':
        weights = np.clip(
            np.random.normal(mean, scale, (n_inputs, n_outputs)),
            -2*scale, 2*scale
        )
    return weights

# Example usage
layer_weights = custom_weight_init(784, 256, 'truncated_normal', scale=0.02)
print(f"Weight Statistics:\nMean: {np.mean(layer_weights):.6f}")
print(f"Variance: {np.var(layer_weights):.6f}")
```

Slide 7: Real-world Implementation - MNIST Classification

A practical implementation showing weight initialization impact on MNIST classification using a simple neural network. This example demonstrates the effect of different initialization methods on model convergence and performance.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, init_method='xavier'):
        self.init_methods = {
            'xavier': lambda n_in, n_out: xavier_init(n_in, n_out),
            'he': lambda n_in, n_out: he_init(n_in, n_out)
        }
        
        # Initialize weights using selected method
        self.W1 = self.init_methods[init_method](input_size, hidden_size)
        self.W2 = self.init_methods[init_method](hidden_size, output_size)
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

# Example usage
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = NeuralNetwork(784, 256, 10, init_method='he')
probs = model.forward(X_train[:1])
print(f"Prediction probabilities shape: {probs.shape}")
print(f"First sample predictions:\n{probs[0]}")
```

Slide 8: Results for MNIST Classification

The comparison of different initialization methods shows significant impact on model convergence and final accuracy. Here we analyze the statistical properties of the network's initial state.

```python
def analyze_initialization(model):
    # Analyze weight distributions
    w1_stats = {
        'mean': np.mean(model.W1),
        'std': np.std(model.W1),
        'max': np.max(np.abs(model.W1)),
        'activation_variance': np.var(np.dot(X_train[:1000], model.W1))
    }
    
    print("Layer 1 Weight Statistics:")
    for key, value in w1_stats.items():
        print(f"{key}: {value:.6f}")
    
    # Analyze activation statistics
    activations = model.forward(X_train[:1000])
    act_stats = {
        'mean': np.mean(activations),
        'std': np.std(activations),
        'dead_neurons': np.mean(activations == 0) * 100
    }
    
    print("\nActivation Statistics:")
    for key, value in act_stats.items():
        print(f"{key}: {value:.6f}")

# Compare different initializations
models = {
    'xavier': NeuralNetwork(784, 256, 10, 'xavier'),
    'he': NeuralNetwork(784, 256, 10, 'he')
}

for name, model in models.items():
    print(f"\nAnalyzing {name} initialization:")
    analyze_initialization(model)
```

Slide 9: Deep Network Implementation with Multiple Initialization Options

A comprehensive implementation of a deep neural network that supports various initialization methods and demonstrates their impact on training dynamics and final model performance.

```python
class DeepNetwork:
    def __init__(self, layer_sizes, init_method='he', activation='relu'):
        self.layers = []
        self.activations = []
        
        for i in range(len(layer_sizes) - 1):
            layer = {
                'W': self._initialize_weights(
                    layer_sizes[i], 
                    layer_sizes[i+1], 
                    init_method,
                    activation
                ),
                'b': np.zeros(layer_sizes[i+1])
            }
            self.layers.append(layer)
    
    def _initialize_weights(self, n_in, n_out, method, activation):
        if method == 'he':
            scale = np.sqrt(2.0 / n_in)
        elif method == 'xavier':
            scale = np.sqrt(2.0 / (n_in + n_out))
        elif method == 'lecun':
            scale = np.sqrt(1.0 / n_in)
        else:
            raise ValueError(f"Unknown initialization method: {method}")
            
        return np.random.normal(0, scale, (n_in, n_out))
    
    def forward(self, X):
        self.activations = [X]
        current_input = X
        
        for layer in self.layers[:-1]:
            z = np.dot(current_input, layer['W']) + layer['b']
            current_input = np.maximum(0, z)  # ReLU
            self.activations.append(current_input)
        
        # Output layer
        final_layer = self.layers[-1]
        z = np.dot(current_input, final_layer['W']) + final_layer['b']
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs

# Example usage
model = DeepNetwork([784, 512, 256, 128, 10], init_method='he')
sample_output = model.forward(X_train[:1])
print(f"Network output shape: {sample_output.shape}")
```

Slide 10: Initialization Impact Analysis

This implementation provides tools for analyzing the impact of different initialization methods on gradient flow and activation patterns throughout the network during training.

```python
def analyze_gradient_flow(model, X_batch):
    activations = []
    gradients = []
    current_input = X_batch
    
    # Forward pass with gradient tracking
    for layer in model.layers:
        z = np.dot(current_input, layer['W']) + layer['b']
        activation = np.maximum(0, z)
        activations.append(activation)
        current_input = activation
        
        # Store gradients
        if z.size > 0:
            grad_w = np.dot(current_input.T, activation)
            gradients.append(np.abs(grad_w).mean())
    
    return {
        'activation_means': [np.mean(act) for act in activations],
        'activation_vars': [np.var(act) for act in activations],
        'gradient_means': gradients,
        'dead_neurons': [np.mean(act == 0) * 100 for act in activations]
    }

# Example analysis
model_he = DeepNetwork([784, 512, 256, 128, 10], init_method='he')
model_xavier = DeepNetwork([784, 512, 256, 128, 10], init_method='xavier')

batch_size = 1000
X_batch = X_train[:batch_size]

print("He Initialization Analysis:")
he_stats = analyze_gradient_flow(model_he, X_batch)
for key, values in he_stats.items():
    print(f"\n{key}:")
    print([f"{v:.6f}" for v in values])

print("\nXavier Initialization Analysis:")
xavier_stats = analyze_gradient_flow(model_xavier, X_batch)
for key, values in xavier_stats.items():
    print(f"\n{key}:")
    print([f"{v:.6f}" for v in values])
```

Slide 11: Training Loop Implementation with Weight Analysis

A comprehensive training implementation that monitors weight statistics and gradient flow during training, demonstrating the long-term effects of different initialization strategies.

```python
def train_network(model, X_train, y_train, epochs=10, batch_size=128, learning_rate=0.01):
    n_samples = X_train.shape[0]
    training_stats = {
        'loss_history': [],
        'weight_stats': [],
        'gradient_stats': []
    }
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            
            # Forward pass
            layer_outputs = []
            current_input = X_batch
            
            for layer in model.layers:
                z = np.dot(current_input, layer['W']) + layer['b']
                activation = np.maximum(0, z)
                layer_outputs.append(activation)
                current_input = activation
            
            # Compute loss
            loss = -np.mean(np.log(current_input[range(len(y_batch)), y_batch]))
            
            # Record statistics
            training_stats['loss_history'].append(loss)
            training_stats['weight_stats'].append([
                np.mean(layer['W']) for layer in model.layers
            ])
            
            # Backpropagation and update (simplified)
            error = current_input
            error[range(len(y_batch)), y_batch] -= 1
            error /= batch_size
            
            for j in reversed(range(len(model.layers))):
                layer = model.layers[j]
                input_activation = X_batch if j == 0 else layer_outputs[j-1]
                
                # Compute gradients
                dW = np.dot(input_activation.T, error)
                db = np.sum(error, axis=0)
                
                # Record gradient statistics
                training_stats['gradient_stats'].append(np.mean(np.abs(dW)))
                
                # Update weights
                layer['W'] -= learning_rate * dW
                layer['b'] -= learning_rate * db
                
                # Propagate error
                if j > 0:
                    error = np.dot(error, layer['W'].T)
                    error[layer_outputs[j-1] <= 0] = 0  # ReLU gradient
        
        # Print epoch statistics
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    
    return training_stats

# Example usage
model = DeepNetwork([784, 512, 256, 128, 10], init_method='he')
stats = train_network(model, X_train[:1000], y_train[:1000].astype(int))

# Analysis of training statistics
print("\nFinal Training Statistics:")
print(f"Final Loss: {stats['loss_history'][-1]:.4f}")
print("Weight Mean Evolution:")
for layer_idx, layer_means in enumerate(zip(*stats['weight_stats'])):
    print(f"Layer {layer_idx + 1}: {layer_means[-1]:.6f}")
```

Slide 12: Full-Scale Implementation with Cyclical Learning Rate

This implementation combines optimal weight initialization with cyclical learning rates to achieve faster convergence and better final performance on complex datasets.

```python
class AdvancedTrainer:
    def __init__(self, model, init_method='he'):
        self.model = model
        self.init_method = init_method
        self.training_history = {
            'loss': [], 'accuracy': [],
            'weight_dist': [], 'gradient_norm': []
        }
    
    def cyclical_learning_rate(self, iteration, base_lr=0.001, max_lr=0.1, step_size=2000):
        cycle = np.floor(1 + iteration / (2 * step_size))
        x = np.abs(iteration/step_size - 2 * cycle + 1)
        return base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    
    def train_epoch(self, X, y, batch_size=128):
        n_batches = len(X) // batch_size
        total_loss = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Get cyclical learning rate
            lr = self.cyclical_learning_rate(self.global_step)
            
            # Forward pass and loss computation
            output = self.model.forward(X_batch)
            loss = -np.mean(np.log(output[range(batch_size), y_batch]))
            
            # Record statistics
            self.training_history['loss'].append(loss)
            self.training_history['weight_dist'].append(
                [np.std(layer['W']) for layer in self.model.layers]
            )
            
            # Update weights using computed gradients
            self._update_weights(X_batch, y_batch, lr)
            
            total_loss += loss
            self.global_step += 1
        
        return total_loss / n_batches
    
    def _update_weights(self, X_batch, y_batch, learning_rate):
        # Implementation of weight updates with gradient clipping
        gradients = self._compute_gradients(X_batch, y_batch)
        for layer_idx, layer in enumerate(self.model.layers):
            # Clip gradients
            grad_norm = np.linalg.norm(gradients[layer_idx]['W'])
            clip_norm = 1.0
            if grad_norm > clip_norm:
                gradients[layer_idx]['W'] *= clip_norm / grad_norm
            
            # Update weights
            layer['W'] -= learning_rate * gradients[layer_idx]['W']
            layer['b'] -= learning_rate * gradients[layer_idx]['b']

# Example usage
trainer = AdvancedTrainer(
    DeepNetwork([784, 512, 256, 128, 10], init_method='he')
)
trainer.global_step = 0
loss = trainer.train_epoch(X_train[:1000], y_train[:1000].astype(int))
print(f"Training loss: {loss:.4f}")
```

Slide 13: Additional Resources

*   "Understanding the difficulty of training deep feedforward neural networks" - Xavier Initialization [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
*   "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" - He Initialization [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
*   "Random walk initialization for training very deep feedforward networks" - Network Initialization Analysis [https://arxiv.org/abs/1412.6558](https://arxiv.org/abs/1412.6558)
*   "All you need is a good init" - Comprehensive Study of Initialization Methods [https://arxiv.org/abs/1511.06422](https://arxiv.org/abs/1511.06422)
*   "Fixup Initialization: Residual Learning Without Normalization" - Modern Initialization Techniques [https://arxiv.org/abs/1901.09321](https://arxiv.org/abs/1901.09321)

