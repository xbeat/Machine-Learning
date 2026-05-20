## Activation Functions Understanding in Neural Networks
Slide 1: Introduction to Activation Functions

Neural networks utilize activation functions to transform input signals into output activations, introducing essential non-linearity that enables the network to learn complex patterns. This implementation demonstrates how to create basic activation functions from scratch using NumPy, providing a foundation for understanding their behavior.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    """Rectified Linear Unit activation function"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Hyperbolic tangent activation function"""
    return np.tanh(x)

# Generate sample data
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(10, 6))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.grid(True)
plt.legend()
plt.title('Common Activation Functions')
plt.show()
```

Slide 2: ReLU Implementation and Derivatives

The Rectified Linear Unit (ReLU) activation function has become the default choice for deep neural networks due to its computational efficiency and effectiveness in preventing vanishing gradients. This implementation shows both forward and backward passes.

```python
import numpy as np

class ReLU:
    def __init__(self):
        self.cache = None
    
    def forward(self, x):
        """Forward pass of ReLU"""
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        """Backward pass of ReLU"""
        x = self.cache
        dx = dout * (x > 0)
        return dx

# Example usage
relu = ReLU()
x = np.array([-2, -1, 0, 1, 2])
forward_output = relu.forward(x)
backward_output = relu.backward(np.ones_like(x))

print(f"Input: {x}")
print(f"Forward output: {forward_output}")
print(f"Backward output: {backward_output}")
```

Slide 3: Mathematical Foundations of Activation Functions

Understanding the mathematical expressions behind activation functions is crucial for implementing and optimizing neural networks. These formulas represent the core transformations applied to neuron inputs.

```python
"""
Common Activation Functions and their Derivatives:

ReLU:
$$f(x) = max(0, x)$$
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

Sigmoid:
$$f(x) = \frac{1}{1 + e^{-x}}$$
$$f'(x) = f(x)(1 - f(x))$$

Tanh:
$$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
$$f'(x) = 1 - f(x)^2$$

Leaky ReLU:
$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x \leq 0 \end{cases}$$
"""
```

Slide 4: Advanced Activation Functions Implementation

Implementation of advanced activation functions including Leaky ReLU, ELU, and SELU, which address various limitations of traditional activation functions and provide improved learning characteristics.

```python
import numpy as np

class AdvancedActivations:
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU with configurable slope"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def elu(x, alpha=1.0):
        """Exponential Linear Unit"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def selu(x):
        """Scaled Exponential Linear Unit"""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Test the implementations
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
activations = AdvancedActivations()

print("Input:", x)
print("Leaky ReLU:", activations.leaky_relu(x))
print("ELU:", activations.elu(x))
print("SELU:", activations.selu(x))
```

Slide 5: Gradient Flow Analysis

Understanding gradient flow through different activation functions is crucial for training deep networks effectively. This implementation provides tools to analyze and visualize gradient propagation through multiple network layers.

```python
import numpy as np
import matplotlib.pyplot as plt

class GradientAnalyzer:
    def __init__(self, activation_fn, derivative_fn):
        self.activation_fn = activation_fn
        self.derivative_fn = derivative_fn
    
    def analyze_gradient_flow(self, x, num_layers=10):
        gradients = []
        activations = [x]
        
        # Forward pass
        for _ in range(num_layers):
            x = self.activation_fn(x)
            activations.append(x)
        
        # Backward pass
        gradient = np.ones_like(x)
        for activation in reversed(activations):
            gradient *= self.derivative_fn(activation)
            gradients.append(np.mean(np.abs(gradient)))
            
        return np.array(gradients)

# Define activation functions and their derivatives
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

# Analysis
analyzer = GradientAnalyzer(sigmoid, sigmoid_derivative)
x = np.random.randn(1000) * 2
gradients = analyzer.analyze_gradient_flow(x)

plt.plot(gradients)
plt.title('Gradient Magnitude vs Layer Depth')
plt.xlabel('Layer')
plt.ylabel('Average Gradient Magnitude')
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 6: Real-world Application: Image Classification

Implementing a convolutional neural network with various activation functions for image classification, demonstrating the impact of activation choice on model performance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, activation='relu'):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dynamic activation function selection
        self.activation = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }[activation]
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Create models with different activations
activations = ['relu', 'leaky_relu', 'elu', 'selu']
models = {act: ConvNet(activation=act) for act in activations}

# Example usage
input_tensor = torch.randn(1, 3, 32, 32)
for act, model in models.items():
    output = model(input_tensor)
    print(f"{act.upper()} output shape:", output.shape)
```

Slide 7: Performance Metrics and Visualization

A comprehensive implementation for analyzing and visualizing the performance characteristics of different activation functions, including convergence speed, accuracy, and gradient behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class ActivationAnalyzer:
    def __init__(self):
        self.metrics = {}
        
    def collect_metrics(self, activation_name, loss_history, accuracy_history, 
                       gradient_norms, training_time):
        self.metrics[activation_name] = {
            'loss': loss_history,
            'accuracy': accuracy_history,
            'gradient_norms': gradient_norms,
            'training_time': training_time
        }
    
    def visualize_comparison(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        for name, data in self.metrics.items():
            axes[0, 0].plot(data['loss'], label=name)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy curves
        for name, data in self.metrics.items():
            axes[0, 1].plot(data['accuracy'], label=name)
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Gradient norms
        for name, data in self.metrics.items():
            axes[1, 0].plot(data['gradient_norms'], label=name)
        axes[1, 0].set_title('Gradient Norms')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('L2 Norm')
        axes[1, 0].legend()
        
        # Training times
        times = [data['training_time'] for data in self.metrics.values()]
        names = list(self.metrics.keys())
        axes[1, 1].bar(names, times)
        axes[1, 1].set_title('Training Time')
        axes[1, 1].set_ylabel('Seconds')
        
        plt.tight_layout()
        plt.show()

# Example usage
analyzer = ActivationAnalyzer()
# Simulate metrics for different activations
for activation in ['ReLU', 'Leaky ReLU', 'ELU']:
    analyzer.collect_metrics(
        activation,
        loss_history=np.random.exponential(scale=0.5, size=100)[::-1],
        accuracy_history=1 - np.random.exponential(scale=0.3, size=100)[::-1],
        gradient_norms=np.random.exponential(scale=0.1, size=100)[::-1],
        training_time=np.random.uniform(100, 200)
    )

analyzer.visualize_comparison()
```

Slide 8: Specialized Activation Functions for Deep Learning

Advanced activation functions designed specifically for deep learning applications, including GELU (Gaussian Error Linear Unit) and Swish, which have shown superior performance in transformer architectures and deep networks.

```python
import numpy as np
import torch
import torch.nn as nn

class AdvancedActivations(nn.Module):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def gelu(x):
        """Gaussian Error Linear Unit"""
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    @staticmethod
    def swish(x, beta=1.0):
        """Swish activation function"""
        return x * torch.sigmoid(beta * x)
    
    @staticmethod
    def mish(x):
        """Mish activation function"""
        return x * torch.tanh(torch.nn.functional.softplus(x))

# Test implementation
x = torch.linspace(-5, 5, 100)
activations = AdvancedActivations()

# Generate comparison plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(x.numpy(), activations.gelu(x).numpy(), label='GELU')
plt.plot(x.numpy(), activations.swish(x).numpy(), label='Swish')
plt.plot(x.numpy(), activations.mish(x).numpy(), label='Mish')
plt.grid(True)
plt.legend()
plt.title('Advanced Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
```

Slide 9: Custom Activation Function Implementation

Implementing a custom parameterized activation function with learnable parameters, demonstrating how to create adaptive activation functions that can be optimized during training.

```python
import torch
import torch.nn as nn

class ParametricActivation(nn.Module):
    def __init__(self, num_parameters=3, init_values=None):
        super().__init__()
        if init_values is None:
            init_values = torch.ones(num_parameters)
        self.parameters = nn.Parameter(init_values)
        
    def forward(self, x):
        """
        Custom activation: f(x) = a*x + b*sin(c*x)
        where a, b, c are learnable parameters
        """
        return (self.parameters[0] * x + 
                self.parameters[1] * torch.sin(self.parameters[2] * x))

class CustomActivationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.custom_activation = ParametricActivation()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.custom_activation(x)
        x = self.layer2(x)
        return x

# Example usage
model = CustomActivationNetwork(10, 20, 2)
x = torch.randn(32, 10)  # Batch of 32 samples
output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Learnable activation parameters:", model.custom_activation.parameters.data)
```

Slide 10: Gradient Analysis Tools

A comprehensive toolkit for analyzing gradient flow through different activation functions, helping identify potential training issues like vanishing or exploding gradients.

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GradientAnalysisTools:
    @staticmethod
    def compute_gradient_stats(model, criterion, input_data, target):
        """Compute gradient statistics for each layer"""
        model.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        
        gradient_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats = {
                    'mean': float(param.grad.abs().mean()),
                    'std': float(param.grad.std()),
                    'max': float(param.grad.abs().max()),
                    'min': float(param.grad.abs().min())
                }
                gradient_stats[name] = grad_stats
        
        return gradient_stats
    
    @staticmethod
    def visualize_gradients(gradient_stats):
        """Visualize gradient statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics = ['mean', 'std', 'max', 'min']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx//2, idx%2]
            values = [stats[metric] for stats in gradient_stats.values()]
            ax.bar(range(len(values)), values)
            ax.set_title(f'Gradient {metric.capitalize()}')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(gradient_stats.keys(), rotation=45)
        
        plt.tight_layout()
        return fig

# Example usage
input_size = 784
hidden_size = 256
output_size = 10

# Create a simple network for testing
model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
)

# Generate sample data
batch_size = 32
x = torch.randn(batch_size, input_size)
y = torch.randint(0, output_size, (batch_size,))

# Analyze gradients
analyzer = GradientAnalysisTools()
stats = analyzer.compute_gradient_stats(model, nn.CrossEntropyLoss(), x, y)
analyzer.visualize_gradients(stats)
plt.show()
```

Slide 11: Real-world Application: Natural Language Processing

Implementation of a simple neural network for sentiment analysis, demonstrating the impact of different activation functions on text classification performance using word embeddings.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, activation='relu'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Dynamic activation selection
        self.activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = self.activation_map[activation]
        
    def forward(self, text, lengths):
        embedded = self.embedding(text)
        # Average word embeddings
        pooled = torch.mean(embedded, dim=1)
        
        x = self.activation(self.fc1(pooled))
        x = self.activation(self.fc2(x))
        return self.fc3(x)

# Example usage and training setup
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2

# Create models with different activations
models = {
    activation: SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, activation)
    for activation in ['relu', 'gelu', 'tanh', 'sigmoid']
}

# Example training loop function
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_text, lengths, labels in data_loader:
        optimizer.zero_grad()
        predictions = model(batch_text, lengths)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = predictions.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(data_loader), correct / total

# Example batch for demonstration
batch_size = 32
seq_length = 50
sample_batch = torch.randint(0, vocab_size, (batch_size, seq_length))
sample_lengths = torch.randint(10, seq_length, (batch_size,))
sample_labels = torch.randint(0, 2, (batch_size,))

# Test each model
for name, model in models.items():
    output = model(sample_batch, sample_lengths)
    print(f"{name.upper()} output shape:", output.shape)
```

Slide 12: Activation Function Visualization Toolkit

A comprehensive visualization toolkit for analyzing activation function behaviors, including gradient flow, activation distributions, and layer-wise activations during network training.

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class ActivationVisualizer:
    def __init__(self):
        self.activation_functions = {
            'relu': torch.nn.ReLU(),
            'leaky_relu': torch.nn.LeakyReLU(),
            'elu': torch.nn.ELU(),
            'gelu': torch.nn.GELU(),
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh
        }
    
    def plot_activation_landscape(self, x_range=(-5, 5), num_points=1000):
        x = torch.linspace(x_range[0], x_range[1], num_points)
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Activation functions
        ax1 = fig.add_subplot(gs[0, :])
        for name, func in self.activation_functions.items():
            y = func(x)
            ax1.plot(x.numpy(), y.numpy(), label=name)
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Activation Functions')
        
        # Derivatives
        ax2 = fig.add_subplot(gs[1, 0])
        for name, func in self.activation_functions.items():
            x.requires_grad_(True)
            y = func(x)
            y.sum().backward()
            ax2.plot(x.detach().numpy(), x.grad.numpy(), label=f'{name} gradient')
            x.grad.zero_()
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Gradients')
        
        # Activation distributions
        ax3 = fig.add_subplot(gs[1, 1])
        sample_input = torch.randn(10000)
        for name, func in self.activation_functions.items():
            output = func(sample_input)
            ax3.hist(output.numpy(), bins=50, alpha=0.3, label=name)
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Output Distributions')
        
        plt.tight_layout()
        return fig

# Create and use the visualizer
visualizer = ActivationVisualizer()
fig = visualizer.plot_activation_landscape()
plt.show()

# Add distribution analysis
def analyze_activation_statistics(activation_name, func, input_tensor):
    output = func(input_tensor)
    stats = {
        'mean': float(output.mean()),
        'std': float(output.std()),
        'min': float(output.min()),
        'max': float(output.max()),
        'zero_fraction': float((output == 0).float().mean())
    }
    print(f"\nStatistics for {activation_name}:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

# Analyze each activation function
input_tensor = torch.randn(10000)
for name, func in visualizer.activation_functions.items():
    analyze_activation_statistics(name, func, input_tensor)
```

Slide 13: Implementing Self-Adaptive Activation Functions

This advanced implementation demonstrates how to create activation functions that can adapt their behavior during training, potentially leading to better model performance by learning optimal activation shapes for specific tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveActivation(nn.Module):
    def __init__(self, num_parameters=3, init_type='identity'):
        super().__init__()
        self.init_type = init_type
        
        # Initialize learnable parameters
        if init_type == 'identity':
            self.a = nn.Parameter(torch.ones(num_parameters))
            self.b = nn.Parameter(torch.zeros(num_parameters))
        else:
            self.a = nn.Parameter(torch.randn(num_parameters))
            self.b = nn.Parameter(torch.randn(num_parameters))
        
    def forward(self, x):
        # Combine multiple activation functions with learnable weights
        return (self.a[0] * F.relu(x) +
                self.a[1] * torch.tanh(x) +
                self.a[2] * torch.sigmoid(x) +
                self.b[0])
                
class AdaptiveNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.adaptive_activation1 = AdaptiveActivation()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.adaptive_activation2 = AdaptiveActivation()
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.adaptive_activation1(self.layer1(x))
        x = self.adaptive_activation2(self.layer2(x))
        return self.layer3(x)

# Training and visualization
def train_adaptive_network(model, epochs=100):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # Generate synthetic data
    X = torch.randn(1000, 10)
    y = torch.sum(X**2, dim=1, keepdim=True)
    
    history = {'loss': [], 'activation_params': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        history['activation_params'].append([
            model.adaptive_activation1.a.detach().clone(),
            model.adaptive_activation1.b.detach().clone()
        ])
    
    return history

# Example usage
model = AdaptiveNetwork(10, 20, 1)
history = train_adaptive_network(model)

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
params = torch.stack([p[0] for p in history['activation_params']])
for i in range(params.shape[1]):
    plt.plot(params[:, i], label=f'Parameter {i+1}')
plt.title('Activation Parameters Evolution')
plt.xlabel('Epoch')
plt.ylabel('Parameter Value')
plt.legend()
plt.tight_layout()
plt.show()
```

Slide 14: Performance Benchmarking Suite

A comprehensive suite for comparing different activation functions across various neural network architectures, providing detailed performance metrics and visualization tools.

```python
import torch
import torch.nn as nn
import time
from collections import defaultdict
import matplotlib.pyplot as plt

class BenchmarkSuite:
    def __init__(self):
        self.activation_functions = {
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU(),
            'ELU': nn.ELU(),
            'GELU': nn.GELU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid()
        }
        self.results = defaultdict(dict)
        
    def create_model(self, activation, architecture):
        layers = []
        for i in range(len(architecture)-1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture)-2:
                layers.append(activation)
        return nn.Sequential(*layers)
    
    def benchmark_forward_pass(self, model, input_size, batch_size, num_iterations=1000):
        x = torch.randn(batch_size, input_size)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        return (time.time() - start_time) / num_iterations
    
    def benchmark_backward_pass(self, model, input_size, batch_size, num_iterations=1000):
        x = torch.randn(batch_size, input_size)
        criterion = nn.MSELoss()
        y = torch.randn(batch_size, model[-1].out_features)
        
        times = []
        for _ in range(num_iterations):
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            start_time = time.time()
            loss.backward()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start_time)
            
        return sum(times) / len(times)
    
    def run_benchmarks(self, architecture=[784, 256, 128, 10], batch_size=32):
        for name, activation in self.activation_functions.items():
            model = self.create_model(activation, architecture)
            
            # Measure forward pass time
            forward_time = self.benchmark_forward_pass(
                model, architecture[0], batch_size)
            
            # Measure backward pass time
            backward_time = self.benchmark_backward_pass(
                model, architecture[0], batch_size)
            
            self.results[name] = {
                'forward_time': forward_time,
                'backward_time': backward_time,
                'total_time': forward_time + backward_time
            }
    
    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot timing results
        names = list(self.results.keys())
        forward_times = [r['forward_time'] for r in self.results.values()]
        backward_times = [r['backward_time'] for r in self.results.values()]
        
        x = range(len(names))
        width = 0.35
        
        ax1.bar(x, forward_times, width, label='Forward')
        ax1.bar([i + width for i in x], backward_times, width, label='Backward')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Computational Performance')
        ax1.set_xticks([i + width/2 for i in x])
        ax1.set_xticklabels(names, rotation=45)
        ax1.legend()
        
        # Plot total time comparison
        total_times = [r['total_time'] for r in self.results.values()]
        ax2.bar(names, total_times)
        ax2.set_ylabel('Total Time (seconds)')
        ax2.set_title('Overall Performance')
        ax2.set_xticklabels(names, rotation=45)
        
        plt.tight_layout()
        plt.show()

# Run benchmarks
suite = BenchmarkSuite()
suite.run_benchmarks()
suite.plot_results()
```

Slide 15: Additional Resources

*   Recent Advances in Neural Network Activation Functions
    *   [https://arxiv.org/abs/2104.02921](https://arxiv.org/abs/2104.02921)
    *   [https://arxiv.org/abs/2009.04759](https://arxiv.org/abs/2009.04759)
    *   [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
*   Adaptive Activation Functions
    *   [https://arxiv.org/abs/2002.07424](https://arxiv.org/abs/2002.07424)
    *   [https://arxiv.org/abs/1906.01170](https://arxiv.org/abs/1906.01170)
*   Performance Analysis and Benchmarking
    *   [https://arxiv.org/abs/2010.09458](https://arxiv.org/abs/2010.09458)
    *   [https://arxiv.org/abs/1801.09403](https://arxiv.org/abs/1801.09403)
*   Suggested searches:
    *   "Deep learning activation functions comparison"
    *   "Neural network activation function optimization"
    *   "Modern activation functions for deep learning"
    *   "Self-adaptive neural network architectures"

