## Types of Activation Functions
Slide 1: Sigmoid Activation Function

The sigmoid function transforms input values into outputs between 0 and 1, making it useful for binary classification problems and probability outputs. It features smooth gradients and was historically popular, though it suffers from vanishing gradient problems at extremes.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Implementing sigmoid function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# Generate sample data points
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

# Plotting sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.grid(True)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')

# Mathematical formula in LaTeX
print("Sigmoid Formula: $$f(x) = \\frac{1}{1 + e^{-x}}$$")

# Example outputs
print(f"sigmoid(0) = {sigmoid(0)}")    # Should output ~0.5
print(f"sigmoid(2) = {sigmoid(2)}")    # Should output ~0.88
print(f"sigmoid(-2) = {sigmoid(-2)}")  # Should output ~0.12
```

Slide 2: ReLU (Rectified Linear Unit)

ReLU has become the most widely used activation function in deep learning due to its computational efficiency and ability to mitigate the vanishing gradient problem. It outputs the input directly if positive, and zero otherwise.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    # ReLU function: f(x) = max(0, x)
    return np.maximum(0, x)

# Generate sample data
x = np.linspace(-10, 10, 100)
y = relu(x)

# Plotting ReLU function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('ReLU Activation Function')
plt.grid(True)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')

# Mathematical formula in LaTeX
print("ReLU Formula: $$f(x) = max(0, x)$$")

# Example outputs
print(f"relu(5) = {relu(5)}")     # Should output 5
print(f"relu(-3) = {relu(-3)}")   # Should output 0
print(f"relu(0) = {relu(0)}")     # Should output 0
```

Slide 3: Leaky ReLU

Leaky ReLU addresses the "dying ReLU" problem by allowing small negative values to pass through, using a small slope coefficient for negative inputs instead of zero, helping maintain some gradient flow for negative values.

```python
import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    # Leaky ReLU: f(x) = x if x > 0, else alpha * x
    return np.where(x > 0, x, alpha * x)

# Generate sample data
x = np.linspace(-10, 10, 100)
y = leaky_relu(x)

# Plotting Leaky ReLU
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Leaky ReLU Activation Function')
plt.grid(True)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')

# Mathematical formula in LaTeX
print("Leaky ReLU Formula: $$f(x) = \\max(\\alpha x, x)$$")

# Example outputs
print(f"leaky_relu(5) = {leaky_relu(5)}")     # Should output 5
print(f"leaky_relu(-3) = {leaky_relu(-3)}")   # Should output -0.03
```

Slide 4: Hyperbolic Tangent (tanh)

The hyperbolic tangent function maps inputs to outputs between -1 and 1, providing stronger gradients compared to sigmoid. It's often preferred in hidden layers of neural networks due to its zero-centered nature and bounded output range.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    # tanh function: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    return np.tanh(x)

# Generate sample data
x = np.linspace(-10, 10, 100)
y = tanh(x)

# Plotting tanh function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Tanh Activation Function')
plt.grid(True)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')

# Mathematical formula in LaTeX
print("Tanh Formula: $$f(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}$$")

# Example outputs
print(f"tanh(0) = {tanh(0)}")    # Should output 0
print(f"tanh(2) = {tanh(2)}")    # Should output ~0.96
print(f"tanh(-2) = {tanh(-2)}")  # Should output ~-0.96
```

Slide 5: Softmax Activation Function

Softmax transforms a vector of real numbers into a probability distribution, making it essential for multi-class classification problems. It ensures all outputs are between 0 and 1 and sum to 1, representing class probabilities.

```python
import numpy as np

def softmax(x):
    # Softmax function: f(x_i) = exp(x_i) / sum(exp(x_j))
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example with a batch of logits
logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

# Mathematical formula in LaTeX
print("Softmax Formula: $$f(x_i) = \\frac{e^{x_i}}{\\sum_{j=1}^K e^{x_j}}$$")

# Example outputs
print("Input logits:", logits)
print("Softmax probabilities:", probs)
print("Sum of probabilities:", np.sum(probs))  # Should be 1.0
```

Slide 6: Parameterized ReLU (PReLU)

PReLU extends Leaky ReLU by making the negative slope coefficient learnable during training. This adaptive approach allows the network to determine the optimal negative slope for each neuron, potentially improving model performance.

```python
import numpy as np
import torch
import torch.nn as nn

class PReLU(nn.Module):
    def __init__(self, num_parameters=1):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.ones(num_parameters) * 0.25)
    
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)

# Example usage
prelu = PReLU()
x = torch.randn(5)
output = prelu(x)

print("Input:", x.numpy())
print("Output:", output.detach().numpy())
print("Learned alpha:", prelu.alpha.item())

# Mathematical formula in LaTeX
print("PReLU Formula: $$f(x) = \\begin{cases} x, & \\text{if } x > 0 \\\\ \\alpha x, & \\text{if } x \\leq 0 \\end{cases}$$")
```

Slide 7: ELU (Exponential Linear Unit)

ELU provides smooth negative values unlike ReLU, helping prevent the dead neuron problem while maintaining the benefits of ReLU. It uses an exponential function for negative values and identity for positive values.

```python
import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1.0):
    # ELU function: f(x) = x if x > 0, else alpha * (exp(x) - 1)
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Generate sample data
x = np.linspace(-5, 5, 100)
y = elu(x)

# Plot ELU function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('ELU Activation Function')
plt.grid(True)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')

# Mathematical formula in LaTeX
print("ELU Formula: $$f(x) = \\begin{cases} x, & \\text{if } x > 0 \\\\ \\alpha(e^x - 1), & \\text{if } x \\leq 0 \\end{cases}$$")

print(f"elu(2) = {elu(2)}")      # Should output 2
print(f"elu(-2) = {elu(-2)}")    # Should output ~ -0.86
```

Slide 8: GELU (Gaussian Error Linear Unit)

GELU has gained popularity in modern architectures like BERT and GPT. It approximates the product of input with the cumulative distribution function of the standard normal distribution, providing smooth transitions.

```python
import numpy as np
import matplotlib.pyplot as plt

def gelu(x):
    # GELU approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# Generate sample data
x = np.linspace(-5, 5, 100)
y = gelu(x)

# Plot GELU function
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('GELU Activation Function')
plt.grid(True)
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')

# Mathematical formula in LaTeX
print("GELU Formula: $$f(x) = x \\cdot \\Phi(x)$$")
print("where Φ(x) is the cumulative distribution function of the standard normal distribution")

print(f"gelu(2) = {gelu(2)}")    # Should output ~1.96
print(f"gelu(-2) = {gelu(-2)}")  # Should output ~-0.04
```

Slide 9: Comparing Activation Functions Performance

This slide demonstrates a practical comparison of different activation functions in a neural network for the MNIST dataset, measuring training time, accuracy, and convergence speed for each activation function.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import time

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model(activation):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Test different activations
activations = ['relu', 'tanh', 'sigmoid', 'elu']
results = {}

for activation in activations:
    start_time = time.time()
    model = create_model(activation)
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)
    training_time = time.time() - start_time
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[activation] = {
        'training_time': training_time,
        'test_accuracy': test_acc,
        'final_loss': history.history['loss'][-1]
    }

# Print results
for activation, metrics in results.items():
    print(f"\nActivation: {activation}")
    print(f"Training Time: {metrics['training_time']:.2f}s")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Final Loss: {metrics['final_loss']:.4f}")
```

Slide 10: Custom Activation Function Implementation

Understanding how to implement custom activation functions is crucial for advanced deep learning applications. This example shows how to create and use a custom activation function in TensorFlow/Keras.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class CustomActivation(Layer):
    def __init__(self, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        
    def call(self, x):
        # Custom activation: combination of ReLU and tanh
        # f(x) = ReLU(x) + 0.5 * tanh(x)
        return tf.nn.relu(x) + 0.5 * tf.tanh(x)

# Test the custom activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(784,)),
    CustomActivation(),
    tf.keras.layers.Dense(32),
    CustomActivation(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Mathematical formula in LaTeX
print("Custom Activation Formula: $$f(x) = max(0,x) + 0.5 \\tanh(x)$$")

# Test with sample input
test_input = np.random.randn(1, 784)
output = model(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output summary: min={output.numpy().min():.4f}, max={output.numpy().max():.4f}")
```

Slide 11: Activation Function Visualization Tools

Creating comprehensive visualization tools helps in understanding the behavior of different activation functions, their derivatives, and how they handle different input ranges and gradients.

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationVisualizer:
    def __init__(self):
        self.functions = {
            'relu': lambda x: np.maximum(0, x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'tanh': lambda x: np.tanh(x)
        }
        
    def plot_activation_and_derivative(self, func_name, x_range=(-5, 5)):
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = self.functions[func_name](x)
        
        # Approximate derivative using central differences
        dx = x[1] - x[0]
        dy = np.gradient(y, dx)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot activation function
        ax1.plot(x, y)
        ax1.set_title(f'{func_name} Function')
        ax1.grid(True)
        
        # Plot derivative
        ax2.plot(x, dy)
        ax2.set_title(f'{func_name} Derivative')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

# Example usage
visualizer = ActivationVisualizer()
for func_name in visualizer.functions.keys():
    visualizer.plot_activation_and_derivative(func_name)
    plt.show()
```

Slide 12: Real-world Application: Image Classification

Demonstrating the impact of different activation functions on a real image classification task using a convolutional neural network with the CIFAR-10 dataset to compare performance metrics.

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def create_cnn_model(activation):
    model = Sequential([
        Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=activation),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Test different activations
activations = ['relu', 'elu', 'tanh']
performance_metrics = {}

for activation in activations:
    model = create_cnn_model(activation)
    history = model.fit(x_train[:1000], y_train[:1000], 
                       epochs=5, 
                       validation_split=0.2,
                       verbose=0)
    
    test_loss, test_acc = model.evaluate(x_test[:200], y_test[:200], verbose=0)
    performance_metrics[activation] = {
        'final_accuracy': test_acc,
        'final_loss': test_loss,
        'training_history': history.history
    }

print("\nPerformance Comparison:")
for activation, metrics in performance_metrics.items():
    print(f"\n{activation.upper()}:")
    print(f"Test Accuracy: {metrics['final_accuracy']:.4f}")
    print(f"Test Loss: {metrics['final_loss']:.4f}")
```

Slide 13: Results Analysis

Comparing the performance metrics and convergence characteristics of different activation functions from the previous implementation to understand their practical implications in deep learning models.

```python
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metrics(performance_metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for activation, metrics in performance_metrics.items():
        history = metrics['training_history']
        ax1.plot(history['accuracy'], label=f'{activation}')
        ax2.plot(history['loss'], label=f'{activation}')
    
    ax1.set_title('Training Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Training Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Create summary table
results_df = pd.DataFrame({
    activation: {
        'Final Accuracy': metrics['final_accuracy'],
        'Final Loss': metrics['final_loss']
    }
    for activation, metrics in performance_metrics.items()
}).round(4)

print("\nSummary Results Table:")
print(results_df)

# Plot training metrics
plot_training_metrics(performance_metrics)
```

Slide 14: Additional Resources

*   Empirical Evaluation of Rectified Activations in Convolutional Network [https://arxiv.org/abs/1505.00853](https://arxiv.org/abs/1505.00853)
*   Gaussian Error Linear Units (GELUs) [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)
*   Deep Sparse Rectifier Neural Networks [https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf](https://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)
*   SEARCHING FOR ACTIVATION FUNCTIONS [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)
*   The Search for Better Activation Functions for Deep Neural Networks [https://www.google.com/search?q=activation+functions+deep+learning+research+papers](https://www.google.com/search?q=activation+functions+deep+learning+research+papers)

