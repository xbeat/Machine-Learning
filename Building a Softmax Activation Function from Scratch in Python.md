## Building a Softmax Activation Function from Scratch in Python
Slide 1: Understanding the Softmax Function

The Softmax function is a crucial component in many machine learning models, particularly in multi-class classification problems. It transforms a vector of real numbers into a probability distribution, ensuring that the sum of all output probabilities equals 1.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example input
x = np.array([2.0, 1.0, 0.1])
probabilities = softmax(x)

plt.bar(range(len(x)), probabilities)
plt.title('Softmax Output')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.show()

print(f"Input: {x}")
print(f"Softmax output: {probabilities}")
print(f"Sum of probabilities: {np.sum(probabilities)}")
```

Slide 2: Mathematical Foundation of Softmax

The Softmax function is defined as:

σ(z)i = exp(zi) / Σj exp(zj)

Where z is the input vector and i ranges over the vector's components.

```python
import sympy as sp

# Define symbols
z1, z2, z3 = sp.symbols('z1 z2 z3')
z = sp.Matrix([z1, z2, z3])

# Define softmax function symbolically
softmax = sp.exp(z) / sp.sum(sp.exp(z))

# Print the symbolic representation
print("Symbolic Softmax:")
sp.pprint(softmax)

# Evaluate for specific values
values = {z1: 2, z2: 1, z3: 0.1}
result = softmax.subs(values)

print("\nNumerical result:")
sp.pprint(result.evalf())
```

Slide 3: Implementing Softmax from Scratch

Let's implement the Softmax function step by step:

```python
import numpy as np

def softmax(x):
    # Step 1: Subtract max(x) for numerical stability
    x_shifted = x - np.max(x)
    
    # Step 2: Calculate exponentials
    exp_x = np.exp(x_shifted)
    
    # Step 3: Calculate sum of exponentials
    sum_exp_x = np.sum(exp_x)
    
    # Step 4: Divide each exponential by the sum
    softmax_x = exp_x / sum_exp_x
    
    return softmax_x

# Test the function
x = np.array([2.0, 1.0, 0.1])
result = softmax(x)
print(f"Input: {x}")
print(f"Softmax output: {result}")
print(f"Sum of probabilities: {np.sum(result)}")
```

Slide 4: Numerical Stability in Softmax

The Softmax function can suffer from numerical instability due to large exponents. We address this by subtracting the maximum value from each input.

```python
import numpy as np

def unstable_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def stable_softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

# Test with large values
x_large = np.array([1000, 2000, 3000])

print("Unstable Softmax:")
print(unstable_softmax(x_large))

print("\nStable Softmax:")
print(stable_softmax(x_large))
```

Slide 5: Softmax Derivative

Understanding the derivative of the Softmax function is crucial for implementing backpropagation in neural networks.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

# Test the derivative
x = np.array([2.0, 1.0, 0.1])
derivative = softmax_derivative(x)

print(f"Input: {x}")
print(f"Softmax derivative: {derivative}")
```

Slide 6: Vectorized Implementation

For efficiency, we can implement Softmax using vectorized operations, which is especially useful for batch processing.

```python
import numpy as np

def vectorized_softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

# Test with a batch of inputs
X = np.array([[2.0, 1.0, 0.1],
              [3.0, 2.0, 1.0],
              [1.0, 2.0, 3.0]])

result = vectorized_softmax(X)
print("Input batch:")
print(X)
print("\nSoftmax output:")
print(result)
print("\nSum of probabilities for each row:")
print(np.sum(result, axis=1))
```

Slide 7: Real-Life Example: Image Classification

In image classification tasks, Softmax is often used as the final layer to convert raw scores into class probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Simulated raw scores from a CNN for an image
class_scores = np.array([2.1, 1.8, 0.7, 0.9, 1.5])
class_names = ['Dog', 'Cat', 'Bird', 'Fish', 'Rabbit']

probabilities = softmax(class_scores)

plt.figure(figsize=(10, 6))
plt.bar(class_names, probabilities)
plt.title('Class Probabilities for Image Classification')
plt.ylabel('Probability')
plt.ylim(0, 1)
for i, v in enumerate(probabilities):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.show()

print("Class probabilities:")
for name, prob in zip(class_names, probabilities):
    print(f"{name}: {prob:.4f}")
```

Slide 8: Real-Life Example: Natural Language Processing

In sentiment analysis, Softmax can be used to classify text into different sentiment categories.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Simulated raw scores from a sentiment analysis model
sentiment_scores = np.array([1.2, 0.8, -0.5])
sentiment_labels = ['Positive', 'Neutral', 'Negative']

probabilities = softmax(sentiment_scores)

print("Sentiment Analysis Results:")
for label, prob in zip(sentiment_labels, probabilities):
    print(f"{label}: {prob:.4f}")

predicted_sentiment = sentiment_labels[np.argmax(probabilities)]
print(f"\nPredicted Sentiment: {predicted_sentiment}")
```

Slide 9: Softmax vs. Sigmoid

While Softmax is used for multi-class classification, Sigmoid is used for binary classification. Let's compare them:

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
softmax_output = softmax(np.column_stack((x, np.zeros_like(x))))[:, 0]
sigmoid_output = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, softmax_output, label='Softmax')
plt.plot(x, sigmoid_output, label='Sigmoid')
plt.title('Softmax vs Sigmoid')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 10: Temperature in Softmax

The temperature parameter in Softmax can control the "sharpness" of the probability distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax_with_temperature(x, temperature):
    exp_x = np.exp(x / temperature)
    return exp_x / exp_x.sum()

x = np.array([2.0, 1.0, 0.1])
temperatures = [0.1, 1.0, 10.0]

plt.figure(figsize=(12, 4))
for i, temp in enumerate(temperatures):
    probs = softmax_with_temperature(x, temp)
    plt.subplot(1, 3, i+1)
    plt.bar(range(len(x)), probs)
    plt.title(f'Temperature: {temp}')
    plt.ylim(0, 1)

plt.tight_layout()
plt.show()
```

Slide 11: Softmax in Neural Networks

Implementing Softmax as the final layer in a simple neural network for multi-class classification:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = softmax(self.z2)
        return self.output

# Example usage
input_size, hidden_size, output_size = 4, 5, 3
nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)

X = np.random.randn(2, input_size)  # 2 samples
output = nn.forward(X)

print("Network output (probabilities):")
print(output)
print("\nSum of probabilities for each sample:")
print(np.sum(output, axis=1))
```

Slide 12: Cross-Entropy Loss with Softmax

The cross-entropy loss is commonly used with Softmax in classification tasks. Let's implement it:

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / n_samples
    return loss

# Example usage
logits = np.array([[2.0, 1.0, 0.1],
                   [0.8, 1.2, 3.0]])
y_true = np.array([[1, 0, 0],
                   [0, 0, 1]])

y_pred = softmax(logits)
loss = cross_entropy_loss(y_true, y_pred)

print("Predicted probabilities:")
print(y_pred)
print("\nCross-entropy loss:", loss)
```

Slide 13: Softmax in PyTorch

For comparison, let's see how Softmax is implemented in a popular deep learning framework like PyTorch:

```python
import torch
import torch.nn as nn

# Define a simple neural network with Softmax
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Create an instance of the network
input_size, hidden_size, num_classes = 4, 10, 3
model = SimpleNet(input_size, hidden_size, num_classes)

# Generate some random input
x = torch.randn(2, input_size)

# Forward pass
output = model(x)

print("Network output (probabilities):")
print(output)
print("\nSum of probabilities for each sample:")
print(torch.sum(output, dim=1))
```

Slide 14: Additional Resources

For a deeper understanding of Softmax and its applications in machine learning:

1. "Softmax Classification" by Stanford CS231n: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)
2. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima": [https://arxiv.org/abs/1609.04836](https://arxiv.org/abs/1609.04836)
3. "Rethinking the Inception Architecture for Computer Vision": [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)

These papers provide insights into the use of Softmax in various deep learning architectures and techniques for improving its performance.

