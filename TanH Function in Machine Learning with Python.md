## TanH Function in Machine Learning with Python
Slide 1: Introduction to TanH Function

The TanH (Hyperbolic Tangent) function is a crucial activation function in neural networks and machine learning. It maps input values to output values between -1 and 1, making it useful for classification tasks and hidden layers in neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
y = tanh(x)

plt.plot(x, y)
plt.title('TanH Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()
```

Slide 2: Mathematical Definition of TanH

The TanH function is defined as the ratio of hyperbolic sine to hyperbolic cosine. It can be expressed in terms of exponential functions:

tanh(x) = (e^x - e^-x) / (e^x + e^-x)

```python
import numpy as np

def tanh_manual(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = 2.0
result = tanh_manual(x)
print(f"TanH of {x} is approximately {result:.4f}")
```

Slide 3: Properties of TanH Function

TanH is a smooth, continuous function with a range between -1 and 1. It's symmetric around the origin and has a steeper gradient compared to the sigmoid function, which can lead to faster learning in some cases.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
y = tanh(x)

plt.plot(x, y)
plt.title('TanH Function Properties')
plt.xlabel('Input')
plt.ylabel('Output')
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(0.5, 0.9, 'Range: (-1, 1)', transform=plt.gca().transAxes)
plt.text(0.5, 0.8, 'Symmetric around origin', transform=plt.gca().transAxes)
plt.grid(True)
plt.show()
```

Slide 4: Derivative of TanH Function

The derivative of the TanH function is important for backpropagation in neural networks. It's given by:

d/dx tanh(x) = 1 - tanh^2(x)

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x)**2

x = np.linspace(-5, 5, 100)
y_tanh = tanh(x)
y_derivative = tanh_derivative(x)

plt.plot(x, y_tanh, label='TanH')
plt.plot(x, y_derivative, label='TanH Derivative')
plt.title('TanH and its Derivative')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: TanH in Neural Networks

TanH is commonly used as an activation function in hidden layers of neural networks. It helps introduce non-linearity and can handle negative values better than ReLU.

```python
import numpy as np

class NeuronWithTanH:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.tanh(linear_output)

# Example usage
neuron = NeuronWithTanH(weights=np.array([0.5, -0.5, 0.3]), bias=0.1)
input_data = np.array([0.5, 1.0, -0.5])
output = neuron.forward(input_data)
print(f"Neuron output: {output:.4f}")
```

Slide 6: Advantages of TanH

TanH offers several benefits in machine learning models:

1. Zero-centered output, which can help in subsequent layers.
2. Stronger gradients compared to sigmoid, potentially leading to faster learning.
3. Ability to map negative inputs to negative outputs, preserving negative information.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-5, 5, 100)
y_tanh = tanh(x)
y_sigmoid = sigmoid(x)

plt.plot(x, y_tanh, label='TanH')
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('TanH vs Sigmoid')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 7: Disadvantages of TanH

Despite its advantages, TanH has some limitations:

1. Vanishing gradient problem for very large or small inputs.
2. Computationally more expensive than simpler functions like ReLU.
3. Still suffers from saturation in extreme regions.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x)**2

x = np.linspace(-10, 10, 1000)
y_tanh = tanh(x)
y_derivative = tanh_derivative(x)

plt.plot(x, y_tanh, label='TanH')
plt.plot(x, y_derivative, label='TanH Derivative')
plt.title('TanH Saturation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.text(0, -0.9, 'Vanishing gradient\nin extreme regions', ha='center')
plt.show()
```

Slide 8: Implementing TanH in PyTorch

PyTorch, a popular deep learning framework, provides a built-in TanH function. Here's how to use it in a simple neural network:

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.tanh(self.hidden(x))
        x = self.output(x)
        return x

# Example usage
model = SimpleNN(input_size=10, hidden_size=5, output_size=2)
input_tensor = torch.randn(1, 10)
output = model(input_tensor)
print("Output shape:", output.shape)
```

Slide 9: TanH in TensorFlow

TensorFlow, another popular deep learning library, also provides a TanH activation function. Here's an example of using TanH in a TensorFlow model:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Example usage
input_data = tf.random.normal((100, 10))
output = model(input_data)
print("Output shape:", output.shape)
```

Slide 10: Comparing TanH with Other Activation Functions

It's important to understand how TanH compares to other common activation functions like ReLU and Sigmoid:

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-5, 5, 100)
y_tanh = tanh(x)
y_relu = relu(x)
y_sigmoid = sigmoid(x)

plt.plot(x, y_tanh, label='TanH')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Activation Functions Comparison')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Real-life Example: Image Classification

TanH can be used in image classification tasks. Here's a simple example using the MNIST dataset:

```python
import tensorflow as tf

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create a model with TanH activation
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Slide 12: Real-life Example: Sentiment Analysis

TanH can also be effective in natural language processing tasks like sentiment analysis:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this movie", "This film is terrible", "Great acting and plot"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

# Create and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded, labels, epochs=10, verbose=0)

# Test the model
test_text = ["This movie is amazing"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_seq, maxlen=10, padding='post', truncating='post')
prediction = model.predict(test_padded)
print(f"Sentiment prediction: {'Positive' if prediction > 0.5 else 'Negative'}")
```

Slide 13: Choosing TanH: When and Why

TanH is particularly useful in certain scenarios:

1. When dealing with data centered around zero.
2. In recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks.
3. When negative inputs should be treated differently from positive ones.
4. As an alternative to sigmoid in hidden layers to mitigate the vanishing gradient problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation(func, name):
    x = np.linspace(-5, 5, 100)
    y = func(x)
    plt.plot(x, y, label=name)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plot_activation(tanh, 'TanH')
plot_activation(sigmoid, 'Sigmoid')

plt.title('TanH vs Sigmoid: Zero-Centered Output')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()
```

Slide 14: Practical Tips for Using TanH

When implementing TanH in your models, consider these tips:

1. Initialize weights properly to avoid saturation.
2. Use TanH in combination with other activation functions.
3. Monitor for vanishing gradients, especially in deep networks.
4. Consider scaling inputs to the \[-1, 1\] range for optimal TanH performance.

```python
import tensorflow as tf

# Example of a model combining TanH with ReLU
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='tanh', input_shape=(10,),
                          kernel_initializer='he_uniform'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Example of input scaling
x = tf.random.normal((100, 10))
x_scaled = 2 * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) - 1

print("Original input range:", tf.reduce_min(x).numpy(), "to", tf.reduce_max(x).numpy())
print("Scaled input range:", tf.reduce_min(x_scaled).numpy(), "to", tf.reduce_max(x_scaled).numpy())
```

Slide 15: Additional Resources

For further exploration of the TanH function and its applications in machine learning:

1. "Understanding Activation Functions in Neural Networks" (ArXiv:1907.03452) URL: [https://arxiv.org/abs/1907.03452](https://arxiv.org/abs/1907.03452)
2. "Efficient BackProp" by Yann LeCun et al. (Neural Networks: Tricks of the Trade) ArXiv reference: cs/9804002
3. "On the importance of initialization and momentum in deep learning" (ArXiv:1301.3691) URL: [https://arxiv.org/abs/1301.3691](https://arxiv.org/abs/1301.3691)

These resources provide in-depth discussions on activation functions, including TanH, and their role in neural network performance and training dynamics.

