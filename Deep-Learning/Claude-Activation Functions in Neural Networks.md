## Response:
Slide 1: Introduction to Activation Functions

Neural networks transform input data through layers of interconnected nodes, with activation functions determining the output of each neuron. These mathematical functions introduce non-linearity, enabling neural networks to learn and approximate complex patterns that simple linear models cannot capture.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation(function, x_range=(-5, 5), num_points=1000):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = function(x)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.grid(True)
    plt.title(function.__name__)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    return plt
```

Slide 2: The Sigmoid Activation Function

The sigmoid function, also known as the logistic function, maps any input value to a probability-like output between 0 and 1. Its smooth, S-shaped curve makes it particularly useful for binary classification problems and gates in advanced architectures.

```python
def sigmoid(x):
    """
    Compute the sigmoid of x
    Formula: sigmoid(x) = 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-x))

# Mathematical representation (for documentation)
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = sigmoid(x)
print(f"Input: {x}")
print(f"Output: {result}")
```

Slide 3: ReLU (Rectified Linear Unit)

ReLU has become the most widely used activation function in deep learning due to its computational efficiency and effectiveness in preventing vanishing gradients. It outputs the input directly if positive, and zero otherwise, introducing non-linearity while maintaining simplicity.

```python
def relu(x):
    """
    Compute the ReLU of x
    Formula: relu(x) = max(0, x)
    """
    return np.maximum(0, x)

# Mathematical representation
# $$f(x) = \max(0, x)$$

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = relu(x)
print(f"Input: {x}")
print(f"Output: {result}")
```

Slide 4: Leaky ReLU Implementation

Leaky ReLU addresses the "dying ReLU" problem by allowing a small gradient when the input is negative. This modification helps prevent neurons from becoming permanently inactive during training, improving model robustness and learning capacity.

```python
def leaky_relu(x, alpha=0.01):
    """
    Compute the Leaky ReLU of x
    Formula: f(x) = max(αx, x) where α is typically 0.01
    """
    return np.where(x > 0, x, alpha * x)

# Mathematical representation
# $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases}$$

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = leaky_relu(x)
print(f"Input: {x}")
print(f"Output: {result}")
```

Slide 5: Hyperbolic Tangent (tanh)

The hyperbolic tangent function maps inputs to outputs between -1 and 1, making it zero-centered unlike the sigmoid. This property often leads to faster convergence in training deep networks, especially when dealing with normalized input data.

```python
def tanh(x):
    """
    Compute the hyperbolic tangent of x
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """
    return np.tanh(x)

# Mathematical representation
# $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = tanh(x)
print(f"Input: {x}")
print(f"Output: {result}")
```

\[Continuing with the remaining slides...\]

## Response:
Slide 6: Softmax Function

The Softmax function converts a vector of real numbers into a probability distribution. It's commonly used in multi-class classification problems as the final layer activation, ensuring outputs sum to 1 and can be interpreted as class probabilities.

```python
def softmax(x):
    """
    Compute softmax values for vector x
    Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j in range(len(x))
    """
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Mathematical representation
# $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

# Example usage
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
print(f"Input logits: {logits}")
print(f"Output probabilities: {probabilities}")
print(f"Sum of probabilities: {np.sum(probabilities)}")
```

Slide 7: Custom Activation Function Implementation

Creating custom activation functions allows for specialized network behavior. This example implements a combination of ReLU and tanh, demonstrating how to create hybrid activation functions that can capture different aspects of both functions.

```python
def custom_activation(x, threshold=2.0):
    """
    Custom activation function combining ReLU and tanh
    Uses ReLU for inputs below threshold and tanh for inputs above
    """
    return np.where(np.abs(x) <= threshold,
                   relu(x),
                   np.tanh(x))

# Example usage
x = np.linspace(-5, 5, 10)
result = custom_activation(x)
print(f"Input: {x}")
print(f"Output: {result}")

# Visualization
plt.figure(figsize=(10, 6))
x_plot = np.linspace(-5, 5, 1000)
plt.plot(x_plot, custom_activation(x_plot), label='Custom')
plt.plot(x_plot, relu(x_plot), '--', label='ReLU')
plt.plot(x_plot, np.tanh(x_plot), ':', label='tanh')
plt.grid(True)
plt.legend()
plt.title('Custom Activation Function')
```

Slide 8: Implementing Derivatives for Backpropagation

Understanding and implementing activation function derivatives is crucial for backpropagation. These implementations show how to compute gradients for common activation functions, essential for training neural networks from scratch.

```python
class ActivationFunctions:
    @staticmethod
    def sigmoid_derivative(x):
        """
        Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        """
        sx = sigmoid(x)
        return sx * (1 - sx)
    
    @staticmethod
    def relu_derivative(x):
        """
        Derivative of ReLU: 1 if x > 0 else 0
        """
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def tanh_derivative(x):
        """
        Derivative of tanh: 1 - tanh^2(x)
        """
        return 1 - np.tanh(x)**2

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
af = ActivationFunctions()

print("Derivatives at test points:")
print(f"Sigmoid derivative: {af.sigmoid_derivative(x)}")
print(f"ReLU derivative: {af.relu_derivative(x)}")
print(f"Tanh derivative: {af.tanh_derivative(x)}")
```

Slide 9: Real-world Example - Binary Classification with Custom Activation

This implementation demonstrates a complete binary classification problem using a custom neural network with configurable activation functions, showing how different activations affect model performance.

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, activation_fn=relu):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self.activation = activation_fn

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# Train model with different activations
model_relu = SimpleNeuralNetwork(2, 4, activation_fn=relu)
model_tanh = SimpleNeuralNetwork(2, 4, activation_fn=tanh)
```

\[Continuing with the remaining slides...\]

## Response:
Slide 10: Results for Binary Classification Example

The comparative analysis of different activation functions in the binary classification task demonstrates their impact on model convergence and final performance. This implementation shows actual metrics and visualization of decision boundaries.

```python
def train_and_evaluate(model, X, y, epochs=1000, learning_rate=0.01):
    losses = []
    for _ in range(epochs):
        # Forward pass
        output = model.forward(X)
        loss = -np.mean(y * np.log(output + 1e-8) + 
                       (1-y) * np.log(1-output + 1e-8))
        losses.append(loss)
        
        # Backward pass (simplified for demonstration)
        error = output - y
        model.W2 -= learning_rate * np.dot(model.a1.T, error)
        model.W1 -= learning_rate * np.dot(X.T, 
                    np.dot(error, model.W2.T) * (model.a1 > 0))
    
    # Evaluate
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    return accuracy, losses

# Train and evaluate both models
relu_acc, relu_losses = train_and_evaluate(model_relu, X, y)
tanh_acc, tanh_losses = train_and_evaluate(model_tanh, X, y)

print(f"ReLU Model Accuracy: {relu_acc:.4f}")
print(f"Tanh Model Accuracy: {tanh_acc:.4f}")

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(relu_losses, label='ReLU')
plt.plot(tanh_losses, label='Tanh')
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
```

Slide 11: Advanced Activation Functions - GELU Implementation

The Gaussian Error Linear Unit (GELU) has gained popularity in modern architectures like BERT and GPT. It provides a smooth approximation that combines properties of ReLU with probabilistic behavior.

```python
def gelu(x):
    """
    Gaussian Error Linear Unit
    Formula: x * Φ(x) where Φ is the cumulative distribution function
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * 
                                 (x + 0.044715 * x**3)))

# Mathematical representation
# $$\text{GELU}(x) = x \cdot \Phi(x)$$

# Example usage and visualization
x = np.linspace(-4, 4, 1000)
y_gelu = gelu(x)
y_relu = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_gelu, label='GELU')
plt.plot(x, y_relu, '--', label='ReLU')
plt.grid(True)
plt.legend()
plt.title('GELU vs ReLU Activation')
plt.xlabel('Input')
plt.ylabel('Output')
```

Slide 12: Adaptive Activation Functions

Adaptive activation functions learn their parameters during training, allowing the network to optimize its non-linearity for specific tasks. This implementation shows a parametric ReLU with learnable slope for negative values.

```python
class PReLU:
    def __init__(self, size, alpha_init=0.25):
        self.alpha = np.full(size, alpha_init)
        self.gradients = np.zeros_like(self.alpha)
    
    def forward(self, x):
        self.input = x
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, grad_output):
        grad_input = np.where(self.input > 0, 1, self.alpha)
        self.gradients = np.sum(
            np.where(self.input <= 0, self.input * grad_output, 0),
            axis=0
        )
        return grad_input * grad_output

# Example usage
prelu = PReLU(size=1)
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
output = prelu.forward(x)
print(f"Input: {x}")
print(f"Output with learned alpha={prelu.alpha[0]}: {output}")
```

Slide 13: Additional Resources

*   "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    *   Search on Google Scholar: "He et al. ICCV 2015 PReLU"
*   "GELU Activation Function"
    *   [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)
*   "Searching for Activation Functions"
    *   [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)
*   "Beta-RELU: A Learnable Activation Function"
    *   Search on Google Scholar: "Beta-RELU activation function neural networks"
*   "Activation Functions in Deep Learning: A Comprehensive Survey"
    *   Search for latest surveys on ArXiv about activation functions

