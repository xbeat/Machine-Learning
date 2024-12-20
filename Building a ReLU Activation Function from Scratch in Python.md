## Building a ReLU Activation Function from Scratch in Python
Slide 1: Introduction to ReLU Activation Function

The Rectified Linear Unit (ReLU) is a popular activation function in neural networks. It's simple yet effective, helping to solve the vanishing gradient problem. In this presentation, we'll build a ReLU function from scratch in Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 100)
y = relu(x)

plt.plot(x, y)
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
```

Slide 2: Understanding ReLU

ReLU is defined as f(x) = max(0, x). It returns 0 for negative inputs and the input itself for positive values. This non-linearity allows neural networks to learn complex patterns.

```python
def relu_explained(x):
    if x > 0:
        return x
    else:
        return 0

# Test the function
print(relu_explained(-5))  # Output: 0
print(relu_explained(3))   # Output: 3
```

Slide 3: Implementing ReLU

Let's implement ReLU using NumPy for efficient array operations. This vectorized implementation allows us to apply ReLU to entire arrays at once.

```python
import numpy as np

def relu_numpy(x):
    return np.maximum(0, x)

# Test the function
input_array = np.array([-2, -1, 0, 1, 2])
output_array = relu_numpy(input_array)
print(output_array)  # Output: [0 0 0 1 2]
```

Slide 4: Visualizing ReLU

Visualization helps understand how ReLU transforms inputs. Let's create a plot to see the ReLU function in action.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-10, 10, 200)
y = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b', label='ReLU')
plt.plot(x, x, 'r--', label='y = x')
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: ReLU Derivative

The derivative of ReLU is important for backpropagation. It's 1 for positive inputs and 0 for negative inputs. At x=0, it's undefined, but we typically use 0 or 1.

```python
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

x = np.linspace(-10, 10, 200)
y = relu_derivative(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'g', label='ReLU Derivative')
plt.axhline(y=0, color='k', linestyle='-')
plt.axvline(x=0, color='k', linestyle='-')
plt.title('ReLU Derivative')
plt.xlabel('x')
plt.ylabel("ReLU'(x)")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: ReLU in a Simple Neural Network

Let's implement ReLU in a basic neural network with one hidden layer. This example demonstrates how ReLU is used in practice.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def simple_network(input_data, weights1, weights2):
    hidden = np.dot(input_data, weights1)
    hidden_activated = relu(hidden)
    output = np.dot(hidden_activated, weights2)
    return output

# Example usage
input_data = np.array([1, 2, 3])
weights1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
weights2 = np.array([0.1, 0.2, 0.3])

result = simple_network(input_data, weights1, weights2)
print("Network output:", result)
```

Slide 7: Comparing ReLU with Other Activation Functions

ReLU is one of many activation functions. Let's compare it with the sigmoid and tanh functions to understand its advantages.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 200)
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y_relu, 'b', label='ReLU')
plt.plot(x, y_sigmoid, 'r', label='Sigmoid')
plt.plot(x, y_tanh, 'g', label='Tanh')
plt.title('Activation Functions Comparison')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Handling the Dying ReLU Problem

The "dying ReLU" problem occurs when neurons become inactive and only output 0. Let's implement Leaky ReLU to address this issue.

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

x = np.linspace(-10, 10, 200)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_relu, 'b', label='ReLU')
plt.plot(x, y_leaky_relu, 'r', label='Leaky ReLU')
plt.title('ReLU vs Leaky ReLU')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: ReLU in Convolutional Neural Networks

ReLU is commonly used in Convolutional Neural Networks (CNNs). Let's apply ReLU to a simple 2D convolution operation.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def conv2d(input_data, kernel):
    h, w = input_data.shape
    kh, kw = kernel.shape
    output = np.zeros((h-kh+1, w-kw+1))
    
    for i in range(h-kh+1):
        for j in range(w-kw+1):
            output[i, j] = np.sum(input_data[i:i+kh, j:j+kw] * kernel)
    
    return relu(output)

# Example usage
input_data = np.random.rand(5, 5)
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

result = conv2d(input_data, kernel)
print("Convolution with ReLU result:\n", result)
```

Slide 10: ReLU in Recurrent Neural Networks

ReLU can also be used in Recurrent Neural Networks (RNNs). Here's a simple implementation of an RNN cell with ReLU activation.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def rnn_cell(x, h_prev, W_xh, W_hh, b_h):
    h_next = relu(np.dot(x, W_xh) + np.dot(h_prev, W_hh) + b_h)
    return h_next

# Example usage
x = np.array([1, 2, 3])  # Input
h_prev = np.zeros(4)  # Previous hidden state
W_xh = np.random.randn(3, 4)  # Input to hidden weights
W_hh = np.random.randn(4, 4)  # Hidden to hidden weights
b_h = np.zeros(4)  # Hidden bias

h_next = rnn_cell(x, h_prev, W_xh, W_hh, b_h)
print("Next hidden state:", h_next)
```

Slide 11: ReLU and Gradient Descent

ReLU's simple derivative makes it efficient for gradient descent. Let's implement a basic gradient descent step using ReLU.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def gradient_descent_step(x, y, w, b, learning_rate):
    z = np.dot(x, w) + b
    a = relu(z)
    error = a - y
    
    dw = np.dot(x.T, error * relu_derivative(z))
    db = np.sum(error * relu_derivative(z))
    
    w -= learning_rate * dw
    b -= learning_rate * db
    
    return w, b

# Example usage
x = np.array([[1, 2], [3, 4]])
y = np.array([1, 0])
w = np.random.randn(2)
b = 0
learning_rate = 0.01

w, b = gradient_descent_step(x, y, w, b, learning_rate)
print("Updated weights:", w)
print("Updated bias:", b)
```

Slide 12: ReLU in Multi-Layer Perceptrons

Let's implement a Multi-Layer Perceptron (MLP) with ReLU activation in the hidden layers and a softmax output layer.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mlp_forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return A2

# Example usage
X = np.random.randn(10, 5)  # 10 samples, 5 features
W1 = np.random.randn(5, 8)  # 5 input features, 8 hidden neurons
b1 = np.zeros(8)
W2 = np.random.randn(8, 3)  # 8 hidden neurons, 3 output classes
b2 = np.zeros(3)

output = mlp_forward(X, W1, b1, W2, b2)
print("MLP output (probabilities):\n", output)
```

Slide 13: Real-Life Example: Image Classification

ReLU is widely used in image classification tasks. Let's create a simple convolutional layer with ReLU for image processing.

```python
import numpy as np
from scipy import signal

def relu(x):
    return np.maximum(0, x)

def convolve2d(image, kernel):
    return signal.convolve2d(image, kernel, mode='valid')

def conv_layer(image, kernels):
    feature_maps = []
    for kernel in kernels:
        feature_map = convolve2d(image, kernel)
        feature_map = relu(feature_map)
        feature_maps.append(feature_map)
    return np.array(feature_maps)

# Example usage
image = np.random.rand(28, 28)  # 28x28 grayscale image
kernels = np.random.randn(3, 3, 3)  # 3 3x3 kernels

output = conv_layer(image, kernels)
print("Output shape:", output.shape)
print("First feature map:\n", output[0])
```

Slide 14: Real-Life Example: Natural Language Processing

ReLU is also used in NLP tasks. Here's a simple implementation of a word embedding layer with ReLU activation.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def word_embedding(word_indices, embedding_matrix):
    return embedding_matrix[word_indices]

def process_sentence(sentence, vocab, embedding_matrix, hidden_weights, hidden_bias):
    word_indices = [vocab.get(word, 0) for word in sentence.lower().split()]
    embeddings = word_embedding(word_indices, embedding_matrix)
    hidden = np.dot(embeddings, hidden_weights) + hidden_bias
    activated = relu(hidden)
    return np.mean(activated, axis=0)

# Example usage
vocab = {"the": 1, "quick": 2, "brown": 3, "fox": 4, "jumps": 5, "over": 6, "lazy": 7, "dog": 8}
embedding_matrix = np.random.randn(len(vocab) + 1, 50)  # +1 for unknown words
hidden_weights = np.random.randn(50, 20)
hidden_bias = np.zeros(20)

sentence = "The quick brown fox jumps over the lazy dog"
result = process_sentence(sentence, vocab, embedding_matrix, hidden_weights, hidden_bias)
print("Sentence embedding:", result)
```

Slide 15: Additional Resources

For more in-depth understanding of ReLU and its variants:

1. "Rectified Linear Units Improve Restricted Boltzmann Machines" by Nair and Hinton (2010) ArXiv: [https://arxiv.org/abs/1003.0894](https://arxiv.org/abs/1003.0894)
2. "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" by He et al. (2015) ArXiv: [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
3. "Empirical Evaluation of Rectified Activations in Convolutional Network" by Xu et al. (2015) ArXiv: [https://arxiv.org/abs/1505.00853](https://arxiv.org/abs/1505.00853)

These papers provide detailed analyses and comparisons of ReLU with other activation functions, as well as introduce variants like Parametric ReLU and Exponential Linear Units (ELU).

