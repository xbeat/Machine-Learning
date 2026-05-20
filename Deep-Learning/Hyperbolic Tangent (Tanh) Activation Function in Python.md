## Hyperbolic Tangent (Tanh) Activation Function in Python

Slide 1: Introduction to Hyperbolic Tangent (Tanh) Activation Function

The Hyperbolic Tangent (Tanh) is a popular activation function in neural networks. It's an S-shaped curve that maps input values to output values between -1 and 1. Tanh is often preferred over the logistic sigmoid function because it's zero-centered, which can help in certain neural network architectures.

```python
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
y = tanh(x)

plt.plot(x, y)
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
```

Slide 2: Mathematical Definition of Tanh

The Tanh function is defined mathematically as the ratio of the hyperbolic sine to the hyperbolic cosine. It can also be expressed in terms of exponentials:

tanh(x) = (e^x - e^-x) / (e^x + e^-x)

```python

def tanh_manual(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = 2.0
result = tanh_manual(x)
print(f"tanh({x}) = {result}")
```

Slide 3: Properties of Tanh

Tanh has several important properties that make it useful in neural networks. It's differentiable, antisymmetric around the origin, and has a range of (-1, 1). The function approaches but never quite reaches its asymptotes.

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = np.tanh(x)

plt.plot(x, y, label='tanh(x)')
plt.axhline(y=1, color='r', linestyle='--', label='y = 1')
plt.axhline(y=-1, color='r', linestyle='--', label='y = -1')
plt.axvline(x=0, color='g', linestyle='--', label='x = 0')
plt.legend()
plt.title('Properties of Tanh')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)
plt.show()
```

Slide 4: Derivative of Tanh

The derivative of the Tanh function is important for backpropagation in neural networks. It's given by:

d/dx tanh(x) = 1 - tanh^2(x)

```python
import matplotlib.pyplot as plt

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

x = np.linspace(-5, 5, 100)
y = tanh_derivative(x)

plt.plot(x, y)
plt.title('Derivative of Tanh')
plt.xlabel('x')
plt.ylabel("tanh'(x)")
plt.grid(True)
plt.show()
```

Slide 5: Implementing Tanh in Python

Python's NumPy library provides an efficient implementation of the Tanh function. Here's how to use it:

```python

# Single value
x = 2.0
result = np.tanh(x)
print(f"tanh({x}) = {result}")

# Array of values
x_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result_array = np.tanh(x_array)
print("Input array:", x_array)
print("Tanh output:", result_array)
```

Slide 6: Tanh in Neural Networks

In neural networks, Tanh is often used as an activation function in hidden layers. It helps introduce non-linearity into the model, allowing it to learn complex patterns.

```python

class NeuronWithTanh:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, inputs):
        z = np.dot(self.weights, inputs) + self.bias
        return np.tanh(z)

# Example usage
neuron = NeuronWithTanh(3)
inputs = np.array([0.5, -0.2, 0.1])
output = neuron.activate(inputs)
print(f"Neuron output: {output}")
```

Slide 7: Comparing Tanh with Other Activation Functions

Tanh is often compared to other activation functions like ReLU and Sigmoid. Each has its strengths and use cases. Tanh is particularly useful when you need negative outputs.

```python
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.legend()
plt.title('Comparison of Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 8: Vanishing Gradient Problem

One challenge with Tanh is the vanishing gradient problem. For inputs with large absolute values, the gradient becomes very small, which can slow down learning.

```python
import matplotlib.pyplot as plt

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

x = np.linspace(-10, 10, 1000)
y = tanh_derivative(x)

plt.plot(x, y)
plt.title('Tanh Derivative: Vanishing Gradient')
plt.xlabel('x')
plt.ylabel("tanh'(x)")
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()
```

Slide 9: Tanh in Practice: Image Classification

In image classification tasks, Tanh can be used in convolutional neural networks. Here's a simple example using TensorFlow:

```python

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

Slide 10: Tanh in Practice: Natural Language Processing

Tanh is also used in recurrent neural networks for tasks like sentiment analysis. Here's a simple RNN using PyTorch:

```python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Example usage
model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)
input_seq = torch.randn(1, 5, 10)  # (batch_size, seq_length, input_size)
output = model(input_seq)
print("Output shape:", output.shape)
```

Slide 11: Tanh and Gradient Clipping

To mitigate the vanishing gradient problem, gradient clipping is often used alongside Tanh. This technique prevents gradients from becoming too large during backpropagation.

```python
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Tanh(),
    nn.Linear(20, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # ... forward pass and loss calculation ...
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

Slide 12: Initializing Weights with Tanh

When using Tanh activation, proper weight initialization is crucial. The Xavier/Glorot initialization is often used to ensure that the activations and gradients have similar variances.

```python

class TanhNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TanhNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x

model = TanhNetwork(10, 20, 1)
print(model)
```

Slide 13: Real-life Example: Robotics Control

In robotics, Tanh is often used in control systems. For example, in a robot arm controller, Tanh can map joint angles to motor outputs:

```python

class RobotArmController:
    def __init__(self, num_joints):
        self.weights = np.random.randn(num_joints)
    
    def compute_motor_output(self, joint_angles):
        weighted_sum = np.dot(self.weights, joint_angles)
        return np.tanh(weighted_sum)

# Simulate robot arm control
controller = RobotArmController(num_joints=3)
joint_angles = np.array([0.5, -0.2, 0.8])
motor_output = controller.compute_motor_output(joint_angles)
print(f"Motor output: {motor_output}")
```

Slide 14: Real-life Example: Audio Signal Processing

In audio signal processing, Tanh can be used for soft clipping, which introduces harmonic distortion in a more musical way than hard clipping:

```python
import matplotlib.pyplot as plt

def generate_sine_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * freq * t)

def soft_clip(signal, drive):
    return np.tanh(drive * signal) / np.tanh(drive)

# Generate a sine wave
signal = generate_sine_wave(freq=440, duration=0.01, sample_rate=44100)

# Apply soft clipping
clipped_signal = soft_clip(signal, drive=5)

plt.figure(figsize=(10, 6))
plt.plot(signal, label='Original')
plt.plot(clipped_signal, label='Soft Clipped')
plt.title('Soft Clipping with Tanh')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For more in-depth information on the Hyperbolic Tangent function and its applications in machine learning, consider the following resources:

1. "Understanding Activation Functions in Neural Networks" by Avinash Sharma V (arXiv:1608.08225)
2. "Efficient BackProp" by Yann LeCun et al. (Neural Networks: Tricks of the Trade, 1998)
3. "On the importance of initialization and momentum in deep learning" by Ilya Sutskever et al. (ICML 2013)

These papers provide comprehensive insights into activation functions, including Tanh, and their role in neural network training and performance.


