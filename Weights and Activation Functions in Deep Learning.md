## Weights and Activation Functions in Deep Learning
Slide 1: Introduction to Weights and Activation Functions

Weights and activation functions are fundamental components of neural networks in deep learning. Weights determine the strength of connections between neurons, while activation functions introduce non-linearity, enabling the network to learn complex patterns. Together, they form the basis for the network's ability to approximate various functions and make predictions.

Slide 2: Source Code for Introduction to Weights and Activation Functions

```python
import random

# Simple neuron class
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return max(0, weighted_sum)  # ReLU activation function

# Create a neuron with 3 inputs
neuron = Neuron(3)
inputs = [0.5, 0.3, 0.7]

output = neuron.activate(inputs)
print(f"Neuron output: {output}")
```

Slide 3: Weights in Neural Networks

Weights are numerical parameters that determine the strength of connections between neurons. They are adjusted during training to minimize the difference between predicted and actual outputs. Positive weights strengthen connections, while negative weights weaken them. The magnitude of a weight indicates its importance in the network's decision-making process.

Slide 4: Source Code for Weights in Neural Networks

```python
import random

def initialize_weights(num_inputs, num_neurons):
    return [[random.uniform(-1, 1) for _ in range(num_inputs)] for _ in range(num_neurons)]

def apply_weights(inputs, weights):
    return [sum(i * w for i, w in zip(inputs, neuron_weights)) for neuron_weights in weights]

# Example usage
num_inputs = 3
num_neurons = 2
inputs = [0.5, 0.3, 0.7]

weights = initialize_weights(num_inputs, num_neurons)
weighted_sums = apply_weights(inputs, weights)

print("Weights:")
for i, neuron_weights in enumerate(weights):
    print(f"Neuron {i + 1}: {neuron_weights}")
print(f"\nWeighted sums: {weighted_sums}")
```

Slide 5: Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. They determine whether a neuron should be activated based on its input. Common activation functions include ReLU, Sigmoid, and Tanh. Each has unique properties that make them suitable for different types of problems and network architectures.

Slide 6: Source Code for Activation Functions

```python
import math

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

# Example usage
inputs = [-2, -1, 0, 1, 2]

print("ReLU:")
print([relu(x) for x in inputs])

print("\nSigmoid:")
print([sigmoid(x) for x in inputs])

print("\nTanh:")
print([tanh(x) for x in inputs])
```

Slide 7: ReLU (Rectified Linear Unit)

ReLU is a popular activation function that outputs the input if it's positive, and zero otherwise. It's computationally efficient and helps mitigate the vanishing gradient problem. However, it can suffer from the "dying ReLU" issue, where neurons become inactive and stop learning.

Slide 8: Source Code for ReLU (Rectified Linear Unit)

```python
import matplotlib.pyplot as plt

def relu(x):
    return max(0, x)

x = list(range(-10, 11))
y = [relu(i) for i in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Example of ReLU in a neuron
class ReLUNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return relu(weighted_sum)

neuron = ReLUNeuron([0.5, -0.6, 0.8], 0.1)
print(f"ReLU Neuron output: {neuron.forward([1, 2, 3])}")
```

Slide 9: Sigmoid Function

The sigmoid function maps input values to a range between 0 and 1. It's useful for binary classification problems and in the output layer of multi-class classification tasks. However, it can suffer from vanishing gradients for very large or small inputs, potentially slowing down learning.

Slide 10: Source Code for Sigmoid Function

```python
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = list(range(-10, 11))
y = [sigmoid(i) for i in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Example of Sigmoid in a neuron
class SigmoidNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return sigmoid(weighted_sum)

neuron = SigmoidNeuron([0.5, -0.6, 0.8], 0.1)
print(f"Sigmoid Neuron output: {neuron.forward([1, 2, 3])}")
```

Slide 11: Tanh (Hyperbolic Tangent) Function

The tanh function is similar to the sigmoid but maps inputs to a range between -1 and 1. It's zero-centered, which can help in certain scenarios. Like sigmoid, it can also suffer from vanishing gradients but is generally preferred over sigmoid in hidden layers due to its symmetric nature.

Slide 12: Source Code for Tanh (Hyperbolic Tangent) Function

```python
import math
import matplotlib.pyplot as plt

def tanh(x):
    return math.tanh(x)

x = list(range(-10, 11))
y = [tanh(i) for i in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Example of Tanh in a neuron
class TanhNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return tanh(weighted_sum)

neuron = TanhNeuron([0.5, -0.6, 0.8], 0.1)
print(f"Tanh Neuron output: {neuron.forward([1, 2, 3])}")
```

Slide 13: Real-life Example: Image Classification

In image classification tasks, convolutional neural networks (CNNs) use weights and activation functions to process and classify images. Weights in convolutional layers act as filters to detect features, while activation functions introduce non-linearity, enabling the network to learn complex patterns in images.

Slide 14: Source Code for Real-life Example: Image Classification

```python
import random

class SimpleConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size):
        self.weights = [[[random.uniform(-1, 1) for _ in range(kernel_size)]
                         for _ in range(kernel_size)]
                        for _ in range(output_channels)]
        self.bias = [random.uniform(-1, 1) for _ in range(output_channels)]

    def convolve(self, input_image):
        # Simplified convolution operation
        output = [[0 for _ in range(len(input_image[0]))] for _ in range(len(input_image))]
        for i in range(len(input_image)):
            for j in range(len(input_image[0])):
                output[i][j] = sum(self.weights[0][0]) * input_image[i][j] + self.bias[0]
        return output

    def relu_activate(self, convolved_image):
        return [[max(0, pixel) for pixel in row] for row in convolved_image]

# Example usage
input_image = [[random.random() for _ in range(5)] for _ in range(5)]
conv_layer = SimpleConvLayer(1, 1, 3)

convolved = conv_layer.convolve(input_image)
activated = conv_layer.relu_activate(convolved)

print("Input Image:")
for row in input_image:
    print(row)
print("\nActivated Output:")
for row in activated:
    print(row)
```

Slide 15: Real-life Example: Natural Language Processing

In natural language processing tasks, such as sentiment analysis or language translation, recurrent neural networks (RNNs) or transformers use weights and activation functions to process sequential data. Weights capture relationships between words or tokens, while activation functions help the network learn complex language patterns.

Slide 16: Source Code for Real-life Example: Natural Language Processing

```python
import math
import random

class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.Wx = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.Wh = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b = [random.uniform(-1, 1) for _ in range(hidden_size)]

    def forward(self, x, h_prev):
        # x: input vector, h_prev: previous hidden state
        h_new = [0] * self.hidden_size
        for i in range(self.hidden_size):
            h_new[i] = sum(self.Wx[i][j] * x[j] for j in range(len(x))) + \
                       sum(self.Wh[i][j] * h_prev[j] for j in range(self.hidden_size)) + \
                       self.b[i]
            h_new[i] = math.tanh(h_new[i])  # Apply tanh activation
        return h_new

# Example usage
input_size = 5
hidden_size = 3
rnn_cell = SimpleRNNCell(input_size, hidden_size)

# Simulate processing a sequence of 3 words
sequence = [[random.random() for _ in range(input_size)] for _ in range(3)]
h = [0] * hidden_size  # Initial hidden state

print("Processing sequence:")
for i, word_vector in enumerate(sequence):
    h = rnn_cell.forward(word_vector, h)
    print(f"Step {i + 1}, Hidden state: {h}")
```

Slide 17: Additional Resources

For more in-depth information on weights and activation functions in deep learning, consider exploring these resources:

1.  "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2.  "Understanding Activation Functions in Deep Learning" (arXiv:1907.03452)
3.  "Visualizing and Understanding Convolutional Networks" (arXiv:1311.2901)
4.  "On the difficulty of training Recurrent Neural Networks" (arXiv:1211.5063)

These resources provide comprehensive insights into the theoretical foundations and practical applications of weights and activation functions in various deep learning architectures.

