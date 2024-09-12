## Building a Sigmoid Activation Function from Scratch in Python
Slide 1: The Sigmoid Function: An Introduction

The sigmoid function is a crucial component in neural networks, particularly in binary classification problems. It maps any input value to a number between 0 and 1, making it ideal for representing probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()
```

Slide 2: Mathematical Definition of Sigmoid

The sigmoid function is defined mathematically as σ(x) = 1 / (1 + e^(-x)). This formula is the foundation for our implementation in Python. Let's break down the components and implement them step by step.

```python
import math

def sigmoid_step_by_step(x):
    # Step 1: Calculate e^(-x)
    exp_neg_x = math.exp(-x)
    
    # Step 2: Add 1 to the result
    denominator = 1 + exp_neg_x
    
    # Step 3: Calculate the reciprocal
    result = 1 / denominator
    
    return result

# Test the function
print(sigmoid_step_by_step(0))  # Should output approximately 0.5
print(sigmoid_step_by_step(2))  # Should output approximately 0.88
```

Slide 3: Implementing Sigmoid from Scratch

Let's implement the sigmoid function without using any specialized libraries. We'll use only basic math operations available in Python.

```python
def custom_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

# Test the function
print(custom_sigmoid(0))   # Should output 0.5
print(custom_sigmoid(2))   # Should output approximately 0.88
print(custom_sigmoid(-2))  # Should output approximately 0.12
```

Slide 4: Sigmoid Function Properties

The sigmoid function has several important properties:

1. Output range: (0, 1)
2. Symmetry: σ(-x) = 1 - σ(x)
3. Derivative: σ'(x) = σ(x) \* (1 - σ(x))

Let's visualize these properties:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-6, 6, 200)
y = sigmoid(x)
y_derivative = y * (1 - y)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Sigmoid')
plt.plot(x, y_derivative, label='Derivative')
plt.plot(x, 1 - sigmoid(-x), '--', label='1 - sigmoid(-x)')
plt.title('Sigmoid Function and Its Properties')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: Implementing the Sigmoid Derivative

The derivative of the sigmoid function is crucial for backpropagation in neural networks. Let's implement it from scratch.

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sig_x = sigmoid(x)
    return sig_x * (1 - sig_x)

# Test the derivative function
x_values = [-2, -1, 0, 1, 2]
for x in x_values:
    print(f"x: {x}, sigmoid'(x): {sigmoid_derivative(x):.4f}")
```

Slide 6: Numerical Stability in Sigmoid Implementation

When implementing sigmoid, we need to be careful about numerical stability, especially for large negative inputs. Let's improve our implementation to handle this.

```python
import numpy as np

def stable_sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

# Test with extreme values
print(stable_sigmoid(100))   # Should be close to 1
print(stable_sigmoid(-100))  # Should be close to 0
print(stable_sigmoid(0))     # Should be 0.5
```

Slide 7: Vectorized Sigmoid Implementation

For efficiency, especially when working with neural networks, we often need to apply sigmoid to arrays of values. Let's implement a vectorized version using NumPy.

```python
import numpy as np

def vectorized_sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

# Test with an array of values
x = np.array([-5, -2, 0, 2, 5])
print("Input:", x)
print("Sigmoid output:", vectorized_sigmoid(x))
```

Slide 8: Real-Life Example: Binary Classification

Let's use our sigmoid function in a simple binary classification problem. We'll classify points as inside or outside a circle.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples):
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 < 1).astype(int)
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate data
X, y = generate_data(1000)

# Simple model: distance from origin
distances = np.sum(X**2, axis=1)
probabilities = sigmoid(-distances + 1)

plt.scatter(X[:, 0], X[:, 1], c=probabilities, cmap='coolwarm')
plt.colorbar(label='Probability of being inside the circle')
plt.title('Binary Classification with Sigmoid')
plt.show()
```

Slide 9: Implementing Sigmoid in a Simple Neuron

Let's implement a single neuron using our sigmoid function. This neuron will take two inputs and produce one output.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuron:
    def __init__(self):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        return sigmoid(np.dot(inputs, self.weights) + self.bias)

# Test the neuron
neuron = SimpleNeuron()
input_data = np.array([0.5, 0.3])
output = neuron.forward(input_data)
print(f"Input: {input_data}")
print(f"Output: {output:.4f}")
```

Slide 10: Visualizing Decision Boundary with Sigmoid

Let's visualize how sigmoid creates a decision boundary in a 2D space. We'll use our simple neuron for this purpose.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNeuron:
    def __init__(self):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        return sigmoid(np.dot(inputs, self.weights) + self.bias)

neuron = SimpleNeuron()

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

Z = np.array([neuron.forward(np.array([x1, x2])) 
              for x1, x2 in zip(X1.flatten(), X2.flatten())])
Z = Z.reshape(X1.shape)

plt.contourf(X1, X2, Z, levels=20, cmap='coolwarm')
plt.colorbar(label='Neuron Output')
plt.title('Decision Boundary of a Simple Neuron with Sigmoid')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.show()
```

Slide 11: Sigmoid vs. Other Activation Functions

While sigmoid is widely used, it's important to understand its limitations and compare it with other activation functions. Let's visualize sigmoid alongside ReLU and tanh.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 200)

plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, tanh(x), label='Tanh')
plt.title('Comparison of Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Handling Overflow in Sigmoid

When dealing with large inputs, we might encounter overflow errors. Let's implement a version of sigmoid that can handle very large positive or negative inputs.

```python
import numpy as np

def safe_sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

# Test with extreme values
print(safe_sigmoid(1000))    # Should be very close to 1
print(safe_sigmoid(-1000))   # Should be very close to 0
print(safe_sigmoid(0))       # Should be 0.5

# Vectorized version
def vectorized_safe_sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

# Test vectorized version
x = np.array([-1000, -10, 0, 10, 1000])
print(vectorized_safe_sigmoid(x))
```

Slide 13: Real-Life Example: Image Classification

Let's use our sigmoid function in a simple image classification task. We'll create a tiny neural network to classify handwritten digits from the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Load data
digits = load_digits()
X = digits.data / 16.0  # Normalize
y = np.eye(10)[digits.target]  # One-hot encode

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define network
input_size = 64
hidden_size = 32
output_size = 10

# Initialize weights
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Training loop (simplified, no optimization)
for _ in range(1000):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    # Backward pass (simplified)
    dz2 = a2 - y_train
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    # Update weights
    W1 -= 0.1 * dW1
    b1 -= 0.1 * db1
    W2 -= 0.1 * dW2
    b2 -= 0.1 * db2

# Test accuracy
z1 = np.dot(X_test, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
predictions = np.argmax(sigmoid(z2), axis=1)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test accuracy: {accuracy:.2f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into the mathematics and applications of the sigmoid function in neural networks, here are some valuable resources:

1. "Efficient BackProp" by Yann LeCun et al. (1998) - This paper discusses various practical techniques for improving the performance of neural networks, including the use of sigmoid and other activation functions. Available at: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
2. "Understanding the difficulty of training deep feedforward neural networks" by Xavier Glorot and Yoshua Bengio (2010) - This paper provides insights into the challenges of training deep neural networks and discusses the impact of activation functions like sigmoid. Available at: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
3. "Rectified Linear Units Improve Restricted Boltzmann Machines" by Vinod Nair and Geoffrey Hinton (2010) - While this paper focuses on ReLU, it provides a good comparison with sigmoid and helps understand why alternatives to sigmoid were developed. Available at: [https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

These resources offer a more in-depth look at the sigmoid function and its role in the broader context of neural network architectures and training techniques.

