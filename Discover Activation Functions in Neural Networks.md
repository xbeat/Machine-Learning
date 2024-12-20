## Discover Activation Functions in Neural Networks
Slide 1: ReLU (Rectified Linear Unit)

The Rectified Linear Unit (ReLU) is one of the most widely used activation functions in modern neural networks. It transforms input by outputting zero for negative values and maintaining the input value for positive inputs, effectively introducing non-linearity while avoiding vanishing gradient problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    # ReLU implementation: max(0,x)
    return np.maximum(0, x)

# Generate sample data
x = np.linspace(-10, 10, 200)
y = relu(x)

# Plot ReLU function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ReLU', color='blue')
plt.grid(True)
plt.title('ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Example usage
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output_values = relu(input_values)
print(f"Input: {input_values}")
print(f"Output: {output_values}")
```

Slide 2: Sigmoid Activation Function

The sigmoid activation function maps input values to a range between 0 and 1, making it particularly useful for binary classification problems. It introduces non-linearity and produces smooth, differentiable outputs suitable for gradient-based optimization.

```python
def sigmoid(x):
    # Sigmoid implementation: 1/(1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# Generate sample data
x = np.linspace(-10, 10, 200)
y = sigmoid(x)

# Plot sigmoid function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid', color='red')
plt.grid(True)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Mathematical formula (LaTeX)
print("Sigmoid Formula:")
print("```")
print("$$\sigma(x) = \frac{1}{1 + e^{-x}}$$")
print("```")

# Example usage
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output_values = sigmoid(input_values)
print(f"Input: {input_values}")
print(f"Output: {output_values}")
```

Slide 3: Hyperbolic Tangent (tanh)

The hyperbolic tangent activation function maps inputs to values between -1 and 1, making it zero-centered unlike the sigmoid function. This property often leads to faster convergence in training deep neural networks and helps mitigate the vanishing gradient problem.

```python
def tanh(x):
    # tanh implementation using numpy
    return np.tanh(x)

# Generate sample data
x = np.linspace(-10, 10, 200)
y = tanh(x)

# Plot tanh function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='tanh', color='green')
plt.grid(True)
plt.title('Hyperbolic Tangent Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Mathematical formula (LaTeX)
print("Tanh Formula:")
print("```")
print("$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$")
print("```")

# Example usage
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output_values = tanh(input_values)
print(f"Input: {input_values}")
print(f"Output: {output_values}")
```

Slide 4: Leaky ReLU Implementation

Leaky ReLU addresses the "dying ReLU" problem by allowing a small negative slope instead of completely zeroing out negative values. This modification helps maintain some gradient flow for negative inputs, potentially improving learning in deep neural networks.

```python
def leaky_relu(x, alpha=0.01):
    # Leaky ReLU implementation
    return np.where(x > 0, x, alpha * x)

# Generate sample data
x = np.linspace(-10, 10, 200)
y = leaky_relu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Leaky ReLU', color='purple')
plt.grid(True)
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Example usage with different alpha values
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output_values_01 = leaky_relu(input_values, alpha=0.01)
output_values_05 = leaky_relu(input_values, alpha=0.05)
print(f"Input: {input_values}")
print(f"Output (α=0.01): {output_values_01}")
print(f"Output (α=0.05): {output_values_05}")
```

Slide 5: ELU (Exponential Linear Unit)

The Exponential Linear Unit combines the benefits of ReLU with smooth negative values, helping to push mean unit activations closer to zero. This self-normalizing property can speed up learning and improve gradient flow through the network.

```python
def elu(x, alpha=1.0):
    # ELU implementation
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Generate sample data
x = np.linspace(-10, 10, 200)
y = elu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='ELU', color='orange')
plt.grid(True)
plt.title('ELU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Mathematical formula (LaTeX)
print("ELU Formula:")
print("```")
print("$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$")
print("```")

# Example usage
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output_values = elu(input_values)
print(f"Input: {input_values}")
print(f"Output: {output_values}")
```

Slide 6: SELU (Scaled Exponential Linear Unit)

SELU is a self-normalizing variant of ELU that maintains consistent mean and variance across network layers. It automatically ensures a normalized distribution of activations, eliminating the need for external normalization techniques in deep neural networks.

```python
def selu(x, alpha=1.6732632423543772848170429916717, 
         scale=1.0507009873554804934193349852946):
    # SELU implementation with standard parameters
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Generate sample data
x = np.linspace(-10, 10, 200)
y = selu(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='SELU', color='blue')
plt.grid(True)
plt.title('SELU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Example with comparison to standard activation
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
selu_outputs = selu(input_values)
print(f"Input: {input_values}")
print(f"SELU Output: {selu_outputs}")
```

Slide 7: Softmax Implementation for Multi-class Classification

The Softmax activation function converts a vector of real numbers into a probability distribution. It's essential for multi-class classification tasks, ensuring outputs sum to 1 and can be interpreted as class probabilities.

```python
def softmax(x):
    # Softmax implementation with numerical stability
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Example with multi-class scenario
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)

print("Softmax Formula:")
print("```")
print("$$\sigma(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$$")
print("```")

# Demonstration with multiple examples
test_cases = [
    [1.0, 2.0, 3.0],
    [0.1, 0.2, 0.3],
    [10.0, -10.0, 0.0]
]

for case in test_cases:
    input_vector = np.array(case)
    output_probs = softmax(input_vector)
    print(f"\nInput: {input_vector}")
    print(f"Probabilities: {output_probs}")
    print(f"Sum of probabilities: {np.sum(output_probs):.10f}")
```

Slide 8: Practical Implementation: Binary Classification with Sigmoid

This implementation demonstrates a complete binary classification task using the sigmoid activation function, including data preprocessing, model training, and evaluation metrics.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

class BinaryClassifier:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute gradients
            dZ = predictions - y
            dw = np.dot(X.T, dZ) / len(y)
            db = np.mean(dZ)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = BinaryClassifier(input_size=20)
model.train(X_train, y_train, learning_rate=0.1, epochs=100)

# Evaluate
train_preds = (model.forward(X_train) > 0.5).astype(int)
test_preds = (model.forward(X_test) > 0.5).astype(int)

print(f"Training Accuracy: {accuracy_score(y_train, train_preds):.4f}")
print(f"Testing Accuracy: {accuracy_score(y_test, test_preds):.4f}")
```

Slide 9: Mish Activation Function

Mish is a self-regularizing non-monotonic activation function that allows for better gradient flow. It combines the benefits of ReLU with smoother optimization characteristics, often leading to better performance in deep networks.

```python
def mish(x):
    # Mish implementation
    return x * np.tanh(np.log(1 + np.exp(x)))

# Generate visualization data
x = np.linspace(-10, 10, 200)
y = mish(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Mish', color='green')
plt.grid(True)
plt.title('Mish Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

# Mathematical formula (LaTeX)
print("Mish Formula:")
print("```")
print("$$f(x) = x \cdot \tanh(\ln(1 + e^x))$$")
print("```")

# Example outputs
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output_values = mish(input_values)
print(f"Input: {input_values}")
print(f"Output: {output_values}")
```

Slide 10: Multi-Layer Neural Network with Multiple Activation Functions

This implementation demonstrates how different activation functions can be combined in a deep neural network architecture, showcasing their interaction and impact on network performance.

```python
import numpy as np

class MultiLayerNetwork:
    def __init__(self, layer_sizes, activations):
        self.weights = []
        self.biases = []
        self.activations = activations
        
        # Initialize weights and biases
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.random.randn(layer_sizes[i+1]))
    
    def activation(self, x, name):
        if name == 'relu':
            return np.maximum(0, x)
        elif name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif name == 'tanh':
            return np.tanh(x)
        
    def forward(self, X):
        current_input = X
        layer_outputs = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = self.activation(z, self.activations[i])
            layer_outputs.append(current_input)
            
        return layer_outputs

# Example usage
layer_sizes = [10, 20, 15, 1]  # Network architecture
activations = ['relu', 'tanh', 'sigmoid']  # Activation functions per layer

# Create and test network
network = MultiLayerNetwork(layer_sizes, activations)
test_input = np.random.randn(5, 10)  # 5 samples, 10 features
layer_outputs = network.forward(test_input)

for i, output in enumerate(layer_outputs):
    print(f"Layer {i} output shape: {output.shape}")
```

Slide 11: Swish Activation Function

Swish, developed by Google Brain researchers, is a self-gated activation function that closely resembles ReLU but with smoother derivatives, leading to better gradient flow during backpropagation and faster learning in deep networks.

```python
def swish(x, beta=1.0):
    # Swish implementation
    return x * sigmoid(beta * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate visualization data
x = np.linspace(-10, 10, 200)
y = swish(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Swish', color='purple')
plt.grid(True)
plt.title('Swish Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.show()

print("Swish Formula:")
print("```")
print("$$f(x) = x \cdot \sigma(\beta x)$$")
print("```")

# Compare different beta values
betas = [0.5, 1.0, 2.0]
input_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

for beta in betas:
    output_values = swish(input_values, beta)
    print(f"\nBeta = {beta}")
    print(f"Input: {input_values}")
    print(f"Output: {output_values}")
```

Slide 12: PReLU (Parametric ReLU)

PReLU extends Leaky ReLU by making the negative slope a learnable parameter, allowing the network to adaptively determine the optimal negative slope during training, potentially leading to improved model performance.

```python
class PReLU:
    def __init__(self, num_parameters=1):
        self.alpha = np.random.randn(num_parameters) * 0.01
        
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x, grad_output):
        grad_alpha = np.sum(grad_output * np.where(x > 0, 0, x))
        grad_input = grad_output * np.where(x > 0, 1, self.alpha)
        return grad_input, grad_alpha

# Example usage
prelu = PReLU()
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output = prelu.forward(x)

print("PReLU Formula:")
print("```")
print("$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$")
print("```")

print(f"Input: {x}")
print(f"Output: {output}")
print(f"Learned alpha: {prelu.alpha}")
```

Slide 13: Additional Resources

*   ArXiv paper on Deep Learning with Different Activations:
    *   [https://arxiv.org/abs/1801.03344](https://arxiv.org/abs/1801.03344)
    *   "Comprehensive Analysis of Modern Activation Functions"
*   Research on Self-Normalizing Neural Networks:
    *   [https://arxiv.org/abs/1706.02515](https://arxiv.org/abs/1706.02515)
    *   "Self-Normalizing Neural Networks with SELU"
*   Novel Activation Functions Survey:
    *   [https://arxiv.org/abs/2010.09458](https://arxiv.org/abs/2010.09458)
    *   "A Survey of the Recent Architectures of Deep Convolutional Neural Networks"
*   Activation Functions in Deep Learning:
    *   Google Scholar: "Activation Functions in Deep Learning: A Comprehensive Survey"
    *   IEEE: "Comparative Analysis of Activation Functions in Neural Networks"

