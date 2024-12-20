## Activation Functions in Neural Networks
Slide 1: Understanding Activation Functions Fundamentals

Activation functions serve as crucial non-linear transformations in neural networks, enabling them to learn complex patterns by introducing non-linearity between layers. They determine whether neurons should fire based on input signals, effectively controlling information flow through the network.

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
# Visualization
x = np.linspace(-5, 5, 100)
acts = ActivationFunctions()

plt.figure(figsize=(12, 4))
plt.plot(x, acts.relu(x), label='ReLU')
plt.plot(x, acts.sigmoid(x), label='Sigmoid')
plt.plot(x, acts.tanh(x), label='Tanh')
plt.grid(True)
plt.legend()
plt.title('Common Activation Functions')
plt.show()
```

Slide 2: Mathematical Foundations of Activation Functions

Understanding the mathematical properties of activation functions is essential for choosing the right one for specific neural network architectures and training requirements. These functions transform input signals into output signals based on specific mathematical formulas.

```python
# Mathematical representations of activation functions
"""
ReLU:
$$f(x) = \max(0, x)$$

Sigmoid:
$$f(x) = \frac{1}{1 + e^{-x}}$$

Tanh:
$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Leaky ReLU:
$$f(x) = \max(\alpha x, x)$$, where \alpha is a small constant
"""

class AdvancedActivations:
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

Slide 3: Implementing ReLU from Scratch

The Rectified Linear Unit (ReLU) is the most widely used activation function due to its computational efficiency and effectiveness in reducing the vanishing gradient problem. This implementation shows both forward and backward propagation mechanics.

```python
class ReLU:
    def __init__(self):
        self.cache = None
    
    def forward(self, input_data):
        self.cache = input_data
        return np.maximum(0, input_data)
    
    def backward(self, dout):
        x = self.cache
        dx = dout * (x > 0)
        return dx

# Example usage
relu = ReLU()
input_data = np.array([-2, -1, 0, 1, 2])
output = relu.forward(input_data)
gradient = relu.backward(np.ones_like(input_data))
print(f"Input: {input_data}")
print(f"Forward: {output}")
print(f"Backward: {gradient}")
```

Slide 4: Advanced Sigmoid Implementation

Sigmoid activation functions are particularly useful for binary classification problems as they squash input values between 0 and 1. This implementation includes numerical stability considerations and gradient computation.

```python
class Sigmoid:
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        # Clip values for numerical stability
        x_clipped = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x_clipped))
        return self.output
    
    def backward(self, dout):
        # Compute gradient using chain rule
        return dout * self.output * (1 - self.output)

# Numerical stability demonstration
sigmoid = Sigmoid()
extreme_values = np.array([-1000, -1, 0, 1, 1000])
stable_output = sigmoid.forward(extreme_values)
gradients = sigmoid.backward(np.ones_like(extreme_values))

print(f"Extreme inputs: {extreme_values}")
print(f"Stable outputs: {stable_output}")
print(f"Gradients: {gradients}")
```

Slide 5: Implementing Batch Normalization with Activation

Batch normalization is often used in conjunction with activation functions to stabilize and accelerate neural network training. This implementation demonstrates the integration of batch normalization with ReLU activation.

```python
class BatchNormReLU:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.gamma = np.ones(num_features)
        
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize and scale
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        
        # Apply ReLU
        return np.maximum(0, out)
```

Slide 6: Custom Activation Function Implementation

Creating custom activation functions allows neural networks to adapt to specific problem domains. This implementation demonstrates how to create and use a custom activation function with its corresponding derivative for backpropagation.

```python
class CustomActivation:
    def __init__(self, alpha=0.5, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.input_cache = None
    
    def forward(self, x):
        """Custom activation: f(x) = alpha * x^2 if x > 0 else beta * ln(1 + e^x)"""
        self.input_cache = x
        positive_part = self.alpha * np.square(x) * (x > 0)
        negative_part = self.beta * np.log(1 + np.exp(x)) * (x <= 0)
        return positive_part + negative_part
    
    def backward(self, dout):
        x = self.input_cache
        positive_grad = 2 * self.alpha * x * (x > 0)
        negative_grad = self.beta * (np.exp(x)/(1 + np.exp(x))) * (x <= 0)
        return dout * (positive_grad + negative_grad)

# Example usage and visualization
x = np.linspace(-5, 5, 100)
custom_act = CustomActivation()
y = custom_act.forward(x)
dy = custom_act.backward(np.ones_like(x))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y, label='Forward')
plt.title('Custom Activation')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, dy, label='Gradient')
plt.title('Gradient')
plt.grid(True)
plt.legend()
plt.show()
```

Slide 7: Real-world Application - Image Classification

This implementation demonstrates how different activation functions affect the performance of a convolutional neural network for image classification, including proper initialization and regularization techniques.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ConvLayer:
    def __init__(self, input_shape, num_filters, kernel_size, activation='relu'):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        # Initialize weights using He initialization
        self.weights = np.random.randn(
            num_filters, 
            input_shape[0], 
            kernel_size, 
            kernel_size
        ) * np.sqrt(2.0 / (kernel_size * kernel_size * input_shape[0]))
        
        self.bias = np.zeros(num_filters)
        
        # Activation function selection
        if activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
        elif activation == 'leaky_relu':
            self.activation = lambda x: np.where(x > 0, x, 0.01 * x)
        else:
            self.activation = lambda x: x
            
    def forward(self, x):
        batch_size = x.shape[0]
        output_shape = self.get_output_shape(x.shape)
        output = np.zeros((batch_size, self.num_filters, *output_shape[2:]))
        
        for i in range(batch_size):
            for f in range(self.num_filters):
                for c in range(self.input_shape[0]):
                    output[i, f] += self.convolve(x[i, c], self.weights[f, c])
                output[i, f] += self.bias[f]
                
        return self.activation(output)
    
    def convolve(self, x, kernel):
        output_shape = self.get_output_shape(x.shape)
        output = np.zeros(output_shape[2:])
        
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(
                    x[i:i+self.kernel_size, j:j+self.kernel_size] * kernel
                )
        return output
    
    def get_output_shape(self, input_shape):
        if len(input_shape) == 4:
            _, channels, height, width = input_shape
        else:
            height, width = input_shape
        
        output_height = height - self.kernel_size + 1
        output_width = width - self.kernel_size + 1
        
        return (1, self.num_filters, output_height, output_width)

# Example usage with MNIST digits
digits = load_digits()
X = digits.images.reshape((-1, 1, 8, 8))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and test different activation functions
activations = ['relu', 'leaky_relu']
results = {}

for act in activations:
    conv_layer = ConvLayer(
        input_shape=(1, 8, 8),
        num_filters=16,
        kernel_size=3,
        activation=act
    )
    
    # Forward pass
    output = conv_layer.forward(X_test)
    results[act] = output.shape

print("Output shapes for different activations:")
for act, shape in results.items():
    print(f"{act}: {shape}")
```

Slide 8: Performance Analysis of Activation Functions

Different activation functions can significantly impact model convergence and final performance. This implementation provides tools for analyzing and comparing various activation functions in terms of gradient flow and training dynamics.

```python
class ActivationAnalyzer:
    def __init__(self, activation_functions):
        self.activation_functions = activation_functions
        self.gradient_history = {name: [] for name in activation_functions.keys()}
        self.activation_history = {name: [] for name in activation_functions.keys()}
    
    def analyze_gradient_flow(self, input_range, num_points=1000):
        x = np.linspace(input_range[0], input_range[1], num_points)
        
        for name, func in self.activation_functions.items():
            # Forward pass
            activation = func(x)
            # Approximate gradient
            dx = x[1] - x[0]
            gradient = np.gradient(activation, dx)
            
            self.activation_history[name].append(activation)
            self.gradient_history[name].append(gradient)
            
        return self.generate_analysis_report()
    
    def generate_analysis_report(self):
        report = {}
        for name in self.activation_functions.keys():
            gradients = np.concatenate(self.gradient_history[name])
            activations = np.concatenate(self.activation_history[name])
            
            report[name] = {
                'mean_gradient': np.mean(np.abs(gradients)),
                'max_gradient': np.max(np.abs(gradients)),
                'vanishing_gradient_ratio': np.mean(np.abs(gradients) < 0.01),
                'activation_range': (np.min(activations), np.max(activations))
            }
        return report

# Example usage
activation_functions = {
    'relu': lambda x: np.maximum(0, x),
    'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'tanh': lambda x: np.tanh(x)
}

analyzer = ActivationAnalyzer(activation_functions)
report = analyzer.analyze_gradient_flow((-5, 5))

# Visualize results
plt.figure(figsize=(15, 5))
for name, metrics in report.items():
    print(f"\nAnalysis for {name}:")
    print(f"Mean gradient magnitude: {metrics['mean_gradient']:.4f}")
    print(f"Maximum gradient: {metrics['max_gradient']:.4f}")
    print(f"Vanishing gradient ratio: {metrics['vanishing_gradient_ratio']:.4f}")
    print(f"Activation range: {metrics['activation_range']}")
```

Slide 9: Adaptive Activation Functions

Adaptive activation functions dynamically adjust their parameters during training to optimize network performance. This implementation showcases a trainable activation function that learns its optimal shape through backpropagation.

```python
class AdaptiveActivation:
    def __init__(self, size, init_alpha=0.2, init_beta=1.0):
        self.alpha = np.full(size, init_alpha)
        self.beta = np.full(size, init_beta)
        self.x_cache = None
        self.alpha_grad = np.zeros_like(self.alpha)
        self.beta_grad = np.zeros_like(self.beta)
        
    def forward(self, x):
        """Parametric activation: f(x) = alpha * x * sigmoid(beta * x)"""
        self.x_cache = x
        return self.alpha * x * (1 / (1 + np.exp(-self.beta * x)))
    
    def backward(self, grad_output):
        x = self.x_cache
        sigmoid_val = 1 / (1 + np.exp(-self.beta * x))
        
        # Gradient with respect to input
        dx = grad_output * self.alpha * (
            sigmoid_val + x * self.beta * sigmoid_val * (1 - sigmoid_val)
        )
        
        # Gradient with respect to parameters
        self.alpha_grad = grad_output * x * sigmoid_val
        self.beta_grad = grad_output * self.alpha * x * x * sigmoid_val * (1 - sigmoid_val)
        
        return dx, self.alpha_grad, self.beta_grad

# Training demonstration
import torch.optim as optim

class AdaptiveNetwork:
    def __init__(self, input_size, hidden_size):
        self.weights = np.random.randn(input_size, hidden_size) * 0.01
        self.activation = AdaptiveActivation(hidden_size)
        
    def train_step(self, x, y, learning_rate=0.01):
        # Forward pass
        hidden = x @ self.weights
        output = self.activation.forward(hidden)
        
        # Backward pass
        grad_output = 2 * (output - y)  # MSE loss derivative
        dx, dalpha, dbeta = self.activation.backward(grad_output)
        
        # Update parameters
        self.weights -= learning_rate * (x.T @ dx)
        self.activation.alpha -= learning_rate * np.mean(dalpha, axis=0)
        self.activation.beta -= learning_rate * np.mean(dbeta, axis=0)
        
        return np.mean((output - y) ** 2)

# Example usage
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.sin(X[:, 0]) + np.cos(X[:, 1])  # Non-linear target

model = AdaptiveNetwork(10, 1)
losses = []

for epoch in range(100):
    loss = model.train_step(X, y)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

Slide 10: Specialized Activation Functions for Deep Networks

Deep networks often require carefully designed activation functions to maintain gradient flow through many layers. This implementation demonstrates specialized activation functions optimized for very deep architectures.

```python
class DeepActivations:
    @staticmethod
    def swish(x, beta=1.0):
        """Swish activation: x * sigmoid(beta * x)"""
        return x * (1 / (1 + np.exp(-beta * x)))
    
    @staticmethod
    def mish(x):
        """Mish activation: x * tanh(softplus(x))"""
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    @staticmethod
    def snake(x, freq=1.0):
        """Snake activation: x + sin²(freq * x)"""
        return x + np.square(np.sin(freq * x))
    
    @staticmethod
    def gelu(x):
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))

class DeepNetworkLayer:
    def __init__(self, input_size, output_size, activation='swish'):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros(output_size)
        
        activations = {
            'swish': DeepActivations.swish,
            'mish': DeepActivations.mish,
            'snake': DeepActivations.snake,
            'gelu': DeepActivations.gelu
        }
        self.activation = activations.get(activation, DeepActivations.swish)
        
    def forward(self, x):
        z = x @ self.weights + self.bias
        return self.activation(z)

# Performance comparison
x = np.linspace(-5, 5, 1000)
acts = DeepActivations()

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.plot(x, acts.swish(x), label='Swish')
plt.plot(x, acts.mish(x), label='Mish')
plt.plot(x, acts.snake(x), label='Snake')
plt.plot(x, acts.gelu(x), label='GELU')
plt.grid(True)
plt.legend()
plt.title('Advanced Activation Functions')

# Gradient analysis
dx = 0.001
gradients = {
    'Swish': np.gradient(acts.swish(x), dx),
    'Mish': np.gradient(acts.mish(x), dx),
    'Snake': np.gradient(acts.snake(x), dx),
    'GELU': np.gradient(acts.gelu(x), dx)
}

plt.subplot(122)
for name, grad in gradients.items():
    plt.plot(x, grad, label=f'{name} gradient')
plt.grid(True)
plt.legend()
plt.title('Gradient Behavior')
plt.show()
```

Slide 11: Activation Functions for Recurrent Neural Networks

Recurrent neural networks require specialized activation functions to handle temporal dependencies and prevent gradient issues over long sequences. This implementation focuses on activation functions optimized for RNNs.

```python
class RNNActivations:
    def __init__(self):
        self.states = {}
        
    def hard_tanh(self, x, min_val=-1, max_val=1):
        """Computationally efficient bounded activation"""
        return np.minimum(np.maximum(x, min_val), max_val)
    
    def time_aware_activation(self, x, t, decay_rate=0.1):
        """Time-dependent activation function"""
        time_factor = np.exp(-decay_rate * t)
        return np.tanh(x) * time_factor
    
    def gated_activation(self, x, h_prev):
        """Gated activation with previous state consideration"""
        z = 1 / (1 + np.exp(-(x + h_prev)))  # Update gate
        return z * np.tanh(x) + (1 - z) * h_prev

class RNNLayer:
    def __init__(self, input_size, hidden_size):
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros(hidden_size)
        self.activation = RNNActivations()
        
    def forward(self, x, h_prev, t):
        """Forward pass with time-aware activation"""
        h_raw = np.dot(x, self.Wxh) + np.dot(h_prev, self.Whh) + self.bh
        h_next = self.activation.time_aware_activation(h_raw, t)
        return h_next

# Example usage with sequence data
sequence_length = 100
input_size = 10
hidden_size = 20
batch_size = 32

# Generate sample sequential data
X = np.random.randn(batch_size, sequence_length, input_size)
rnn = RNNLayer(input_size, hidden_size)

# Process sequence
hidden_states = []
h_t = np.zeros((batch_size, hidden_size))

for t in range(sequence_length):
    h_t = rnn.forward(X[:, t, :], h_t, t)
    hidden_states.append(h_t)

# Analyze activation behavior
hidden_states = np.array(hidden_states)
activation_stats = {
    'mean': np.mean(hidden_states, axis=(0, 1)),
    'std': np.std(hidden_states, axis=(0, 1)),
    'max': np.max(hidden_states, axis=(0, 1)),
    'min': np.min(hidden_states, axis=(0, 1))
}

print("Activation Statistics across time steps:")
for metric, value in activation_stats.items():
    print(f"{metric.capitalize()}: {np.mean(value):.4f}")
```

Slide 12: Self-Attention Activation Mechanisms

This implementation demonstrates how activation functions can be integrated with self-attention mechanisms to create more context-aware neural networks.

```python
class AttentionActivation:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.Wq = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wk = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wv = np.random.randn(hidden_size, hidden_size) * 0.01
        
    def attention_weights(self, query, key):
        """Compute attention weights with activation"""
        attention = np.dot(query, key.T)
        attention = attention / np.sqrt(self.hidden_size)
        # Activated attention weights
        return np.exp(attention) / np.sum(np.exp(attention), axis=-1, keepdims=True)
    
    def forward(self, x):
        """Forward pass with self-attention and activation"""
        batch_size = x.shape[0]
        
        # Compute Q, K, V with non-linear transformations
        Q = np.tanh(np.dot(x, self.Wq))
        K = np.tanh(np.dot(x, self.Wk))
        V = np.relu(np.dot(x, self.Wv))
        
        # Compute attention weights
        attention_weights = self.attention_weights(Q, K)
        
        # Apply attention and final activation
        output = np.dot(attention_weights, V)
        return self.gelu_activation(output)
    
    def gelu_activation(self, x):
        """GELU activation for final output"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Example usage
batch_size = 16
sequence_length = 10
hidden_size = 32

# Sample input
x = np.random.randn(batch_size, sequence_length, hidden_size)
attention_layer = AttentionActivation(hidden_size)

# Process with attention activation
output = attention_layer.forward(x)

# Analyze attention patterns
attention_patterns = attention_layer.attention_weights(
    np.mean(x, axis=0),
    np.mean(x, axis=0)
)

plt.figure(figsize=(8, 6))
plt.imshow(attention_patterns, cmap='viridis')
plt.colorbar()
plt.title('Attention Activation Patterns')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()
```

Slide 13: Hardware-Optimized Activation Functions

Modern neural networks must consider hardware constraints and computational efficiency. This implementation demonstrates activation functions specifically designed for optimal hardware performance and reduced memory usage.

```python
import numpy as np
from numba import jit
import time

class HardwareOptimizedActivations:
    @staticmethod
    @jit(nopython=True)
    def fast_relu(x):
        """Hardware-optimized ReLU using Numba"""
        return np.maximum(0, x)
    
    @staticmethod
    @jit(nopython=True)
    def quantized_sigmoid(x, bits=8):
        """Quantized sigmoid for reduced memory footprint"""
        x_quant = np.clip(x * (2**bits), -(2**bits), 2**bits - 1)
        x_quant = np.round(x_quant)
        return x_quant / (2**bits)
    
    @staticmethod
    @jit(nopython=True)
    def approximate_tanh(x):
        """Fast approximation of tanh using piece-wise linear function"""
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        
        result = np.where(abs_x <= 1,
                         x,
                         sign_x * (1 + 0.25 * (abs_x - 1)))
        return np.clip(result, -1, 1)

# Performance benchmark
class ActivationBenchmark:
    def __init__(self, size=(1000, 1000)):
        self.input_data = np.random.randn(*size)
        self.activations = HardwareOptimizedActivations()
    
    def benchmark_activation(self, activation_fn, iterations=100):
        start_time = time.time()
        
        for _ in range(iterations):
            _ = activation_fn(self.input_data)
            
        end_time = time.time()
        return (end_time - start_time) / iterations

# Run benchmarks
benchmark = ActivationBenchmark()
results = {
    'Fast ReLU': benchmark.benchmark_activation(
        HardwareOptimizedActivations.fast_relu
    ),
    'Quantized Sigmoid': benchmark.benchmark_activation(
        lambda x: HardwareOptimizedActivations.quantized_sigmoid(x, bits=8)
    ),
    'Approximate Tanh': benchmark.benchmark_activation(
        HardwareOptimizedActivations.approximate_tanh
    )
}

# Print results
print("Average execution time per iteration (seconds):")
for name, time_taken in results.items():
    print(f"{name}: {time_taken:.6f}")

# Memory usage analysis
def analyze_memory_usage(activation_fn, input_data):
    result = activation_fn(input_data)
    return {
        'Input bytes': input_data.nbytes,
        'Output bytes': result.nbytes,
        'Dtype': result.dtype
    }

memory_analysis = {
    name: analyze_memory_usage(
        lambda x: getattr(HardwareOptimizedActivations, fn)(x),
        benchmark.input_data
    )
    for name, fn in [
        ('fast_relu', 'fast_relu'),
        ('quantized_sigmoid', 'quantized_sigmoid'),
        ('approximate_tanh', 'approximate_tanh')
    ]
}

print("\nMemory Usage Analysis:")
for name, stats in memory_analysis.items():
    print(f"\n{name}:")
    for metric, value in stats.items():
        print(f"  {metric}: {value}")
```

Slide 14: Activation Functions for Generative Models

Generative models require specialized activation functions that can handle both positive and negative values while maintaining stable gradients during training. This implementation focuses on activation functions tailored for generative architectures.

```python
class GenerativeActivations:
    def __init__(self):
        self.cache = {}
    
    def softplus(self, x, beta=1):
        """Smoothed ReLU variant for stable gradients"""
        return (1/beta) * np.log(1 + np.exp(beta * x))
    
    def scaled_tanh(self, x, scale=1.7159):
        """Scaled tanh for improved training dynamics"""
        return scale * np.tanh(2/3 * x)
    
    def prelu(self, x, alpha=0.25):
        """Parametric ReLU for learned negative slopes"""
        return np.where(x > 0, x, alpha * x)
    
    def gaussian_glu(self, x, mean=0, std=1):
        """Gaussian-gated linear unit"""
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        gaussian_gate = np.exp(-0.5 * ((a - mean)/std)**2)
        return gaussian_gate * b

class GenerativeLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.02
        self.b = np.zeros(output_size)
        self.activations = GenerativeActivations()
        
    def forward(self, x, activation='scaled_tanh'):
        z = np.dot(x, self.W) + self.b
        activation_fn = getattr(self.activations, activation)
        return activation_fn(z)

# Demonstration with sample data
batch_size = 64
latent_dim = 100
hidden_dim = 256

# Generate random latent vectors
z = np.random.randn(batch_size, latent_dim)

# Create and test generative layer
gen_layer = GenerativeLayer(latent_dim, hidden_dim)

# Test different activation functions
activation_outputs = {
    'softplus': gen_layer.forward(z, 'softplus'),
    'scaled_tanh': gen_layer.forward(z, 'scaled_tanh'),
    'prelu': gen_layer.forward(z, 'prelu'),
    'gaussian_glu': gen_layer.forward(
        np.concatenate([z, z], axis=-1), 
        'gaussian_glu'
    )
}

# Analyze activation statistics
for name, output in activation_outputs.items():
    stats = {
        'mean': np.mean(output),
        'std': np.std(output),
        'min': np.min(output),
        'max': np.max(output)
    }
    print(f"\n{name} Statistics:")
    for metric, value in stats.items():
        print(f"  {metric}: {value:.4f}")
```

Slide 15: Additional Resources

1.  "Neural Networks and Deep Learning: A Comprehensive Guide to Activation Functions" [https://arxiv.org/abs/2004.06632](https://arxiv.org/abs/2004.06632)
2.  "Activation Functions in Deep Learning: A Comprehensive Survey" [https://arxiv.org/abs/2011.08098](https://arxiv.org/abs/2011.08098)
3.  "Hardware-Aware Training for Efficient Neural Network Design" [https://arxiv.org/abs/1911.03894](https://arxiv.org/abs/1911.03894)
4.  "Advances in Activation Functions for Neural Network Architectures" [https://arxiv.org/abs/2009.04759](https://arxiv.org/abs/2009.04759)
5.  "Self-Attention with Functional Time Representation Learning" [https://arxiv.org/abs/1911.09063](https://arxiv.org/abs/1911.09063)

