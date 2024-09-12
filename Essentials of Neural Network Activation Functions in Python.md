## Essentials of Neural Network Activation Functions in Python
Slide 1: Introduction to Neural Network Activation Functions

Activation functions are crucial components in neural networks, determining the output of a neuron given an input or set of inputs. They introduce non-linearity into the network, allowing it to learn complex patterns and relationships in data. This presentation will explore various activation functions, their characteristics, and implementations in Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation(func, x_range=(-5, 5), num_points=1000):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    plt.plot(x, y)
    plt.title(f"{func.__name__} Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.show()
```

Slide 2: Step Function

The step function is one of the simplest activation functions. It outputs 0 for inputs less than 0, and 1 for inputs greater than or equal to 0. While rarely used in modern neural networks due to its binary nature, it's important to understand as a foundation for more complex functions.

```python
def step_function(x):
    return np.where(x >= 0, 1, 0)

plot_activation(step_function)
```

Slide 3: Linear Activation Function

The linear activation function, f(x) = x, simply returns the input value. It's primarily used in the output layer for regression problems. However, using only linear activations in a network is equivalent to a single-layer network, limiting its ability to learn complex patterns.

```python
def linear_function(x):
    return x

plot_activation(linear_function)
```

Slide 4: Sigmoid Function

The sigmoid function, also known as the logistic function, maps input values to a range between 0 and 1. It's commonly used in the output layer for binary classification problems and in recurrent neural networks.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plot_activation(sigmoid)
```

Slide 5: Hyperbolic Tangent (tanh) Function

The tanh function is similar to the sigmoid but maps inputs to a range between -1 and 1. It's often preferred over sigmoid in hidden layers because its output is zero-centered, which can help with the vanishing gradient problem.

```python
def tanh(x):
    return np.tanh(x)

plot_activation(tanh)
```

Slide 6: Rectified Linear Unit (ReLU)

ReLU is one of the most widely used activation functions in modern neural networks. It outputs the input directly if it's positive, and 0 otherwise. ReLU helps alleviate the vanishing gradient problem and allows for faster training of deep neural networks.

```python
def relu(x):
    return np.maximum(0, x)

plot_activation(relu)
```

Slide 7: Leaky ReLU

Leaky ReLU is a variant of ReLU that allows a small, non-zero gradient when the input is negative. This helps prevent "dying ReLU" problem, where neurons can get stuck in a state where they never activate.

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

plot_activation(leaky_relu)
```

Slide 8: Parametric ReLU (PReLU)

PReLU is similar to Leaky ReLU, but the slope for negative inputs is learned during training rather than being fixed. This allows the network to adapt the activation function to the specific problem.

```python
class PReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, x, self.alpha * x)

prelu = PReLU()
plot_activation(prelu)
```

Slide 9: Exponential Linear Unit (ELU)

ELU is another variant of ReLU that allows negative values, helping to push mean unit activations closer to zero. This can lead to faster learning and better generalization.

```python
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

plot_activation(elu)
```

Slide 10: Softmax Function

The softmax function is commonly used in the output layer for multi-class classification problems. It transforms a vector of real numbers into a probability distribution, where the sum of all probabilities equals 1.

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example usage
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)
print(f"Input scores: {scores}")
print(f"Softmax probabilities: {probabilities}")
```

Slide 11: Swish Activation Function

Swish is a relatively new activation function introduced by Google researchers. It's defined as f(x) = x \* sigmoid(x) and has been shown to outperform ReLU in some deep learning tasks.

```python
def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

plot_activation(swish)
```

Slide 12: Real-life Example: Image Classification

In image classification tasks, different activation functions are often used in different parts of the network. Here's a simple example of a convolutional neural network for image classification using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
model = create_cnn_model((28, 28, 1), 10)  # For MNIST dataset
model.summary()
```

Slide 13: Real-life Example: Sentiment Analysis

In natural language processing tasks like sentiment analysis, different activation functions can be used in recurrent neural networks. Here's an example using a simple LSTM network for sentiment classification:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_lstm_model(vocab_size, embedding_dim, max_length):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(64, activation='tanh', recurrent_activation='sigmoid'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Example usage
model = create_lstm_model(vocab_size=10000, embedding_dim=100, max_length=200)
model.summary()
```

Slide 14: Choosing the Right Activation Function

Selecting the appropriate activation function depends on various factors:

1. Problem type (regression, binary classification, multi-class classification)
2. Network architecture (feedforward, convolutional, recurrent)
3. Desired properties (non-linearity, range of outputs, gradient behavior)
4. Computational efficiency

Experiment with different activation functions and combinations to find the best performance for your specific task.

```python
# Example of comparing different activation functions
activations = [relu, leaky_relu, elu, swish]
x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(12, 8))
for func in activations:
    y = func(x)
    plt.plot(x, y, label=func.__name__)

plt.title("Comparison of Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For more in-depth information on neural network activation functions, consider exploring these research papers:

1. "Rectified Linear Units Improve Restricted Boltzmann Machines" by Nair and Hinton (2010) ArXiv: [https://arxiv.org/abs/1003.0894](https://arxiv.org/abs/1003.0894)
2. "Empirical Evaluation of Rectified Activations in Convolutional Network" by Xu et al. (2015) ArXiv: [https://arxiv.org/abs/1505.00853](https://arxiv.org/abs/1505.00853)
3. "Searching for Activation Functions" by Ramachandran et al. (2017) ArXiv: [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)

These papers provide valuable insights into the development and comparison of various activation functions in deep learning.

