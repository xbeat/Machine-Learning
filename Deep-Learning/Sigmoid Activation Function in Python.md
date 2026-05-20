## Sigmoid Activation Function in Python
Slide 1: Introduction to Sigmoid Activation Function

The sigmoid activation function is a fundamental component in neural networks and machine learning. It maps input values to an output range between 0 and 1, making it useful for binary classification problems and as a smooth, differentiable alternative to step functions.

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

Slide 2: Mathematical Definition

The sigmoid function is defined as σ(x) = 1 / (1 + e^(-x)), where e is the base of natural logarithms. This formula produces an S-shaped curve that asymptotically approaches 0 for large negative values and 1 for large positive values.

```python
import sympy as sp

x = sp.Symbol('x')
sigmoid = 1 / (1 + sp.exp(-x))

print(f"Sigmoid function: σ(x) = {sigmoid}")
print(f"Limit as x approaches positive infinity: {sp.limit(sigmoid, x, sp.oo)}")
print(f"Limit as x approaches negative infinity: {sp.limit(sigmoid, x, -sp.oo)}")
```

Slide 3: Properties of Sigmoid Function

The sigmoid function has several important properties: it's continuous and differentiable, its output is always between 0 and 1, and it has a characteristic S-shape. These properties make it useful for modeling probabilities and gradual transitions.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)
dy = sigmoid_derivative(x)

plt.plot(x, y, label='Sigmoid')
plt.plot(x, dy, label='Derivative')
plt.title('Sigmoid and Its Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Implementing Sigmoid in Python

Here's a simple implementation of the sigmoid function in Python, along with its derivative. The derivative is useful for backpropagation in neural networks.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Example usage
x = np.array([-1, 0, 1])
print(f"Sigmoid of {x}: {sigmoid(x)}")
print(f"Sigmoid derivative of {x}: {sigmoid_derivative(x)}")
```

Slide 5: Sigmoid in Neural Networks

In neural networks, the sigmoid function is often used as an activation function in the output layer for binary classification problems. It maps the network's output to a probability between 0 and 1.

```python
import numpy as np

class SimpleNeuron:
    def __init__(self):
        self.weight = np.random.randn()
        self.bias = np.random.randn()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        return self.sigmoid(self.weight * x + self.bias)

neuron = SimpleNeuron()
input_value = 0.5
output = neuron.forward(input_value)
print(f"Input: {input_value}, Output: {output}")
```

Slide 6: Vanishing Gradient Problem

One limitation of the sigmoid function is the vanishing gradient problem. For inputs with large absolute values, the gradient becomes very small, which can slow down learning in deep neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.linspace(-10, 10, 1000)
y = sigmoid_derivative(x)

plt.plot(x, y)
plt.title('Sigmoid Derivative')
plt.xlabel('x')
plt.ylabel('sigmoid\'(x)')
plt.ylim(0, 0.3)
plt.grid(True)
plt.show()
```

Slide 7: Comparing Sigmoid to Other Activation Functions

While sigmoid has its uses, other activation functions like ReLU (Rectified Linear Unit) and tanh are often preferred in hidden layers of neural networks due to their properties.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, tanh(x), label='Tanh')
plt.title('Activation Functions Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Sigmoid in Logistic Regression

Logistic regression, a fundamental algorithm in machine learning, uses the sigmoid function to model the probability of binary outcomes. It's widely used in various fields for classification tasks.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np

# Generate a random dataset
X, y = make_classification(n_samples=1000, n_features=1, n_classes=2, n_clusters_per_class=1)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Generate points for plotting
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_proba = model.predict_proba(X_plot)[:, 1]

plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_plot, y_proba, color='red')
plt.title('Logistic Regression with Sigmoid')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.show()
```

Slide 9: Real-Life Example: Image Classification

In image classification tasks, sigmoid functions are often used in the output layer of convolutional neural networks (CNNs) for binary classification problems, such as determining if an image contains a specific object or not.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple CNN model for binary classification
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
```

Slide 10: Real-Life Example: Natural Language Processing

In natural language processing tasks, sigmoid functions can be used in sentiment analysis models to classify text as positive or negative. Here's a simple example using a basic neural network:

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = ["I love this movie", "This was awful", "Great performance", "Terrible experience"]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Tokenize the text
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=16, input_length=10),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

Slide 11: Implementing Sigmoid from Scratch

To better understand the sigmoid function, let's implement it from scratch using only basic Python operations. This implementation avoids potential overflow issues for large negative inputs.

```python
import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

# Test the function
test_values = [-1000, -10, -1, 0, 1, 10, 1000]
for value in test_values:
    print(f"sigmoid({value}) = {sigmoid(value)}")
```

Slide 12: Visualizing Sigmoid's Effect on Data

Let's visualize how the sigmoid function transforms data. We'll generate some random data and apply the sigmoid function to it, then plot both the original and transformed data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42)
data = np.random.randn(1000)

# Apply sigmoid function
sigmoid_data = 1 / (1 + np.exp(-data))

# Plot histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, alpha=0.7)
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.hist(sigmoid_data, bins=30, alpha=0.7)
plt.title('Data after Sigmoid')
plt.tight_layout()
plt.show()
```

Slide 13: Sigmoid in Multi-Class Classification

While sigmoid is typically used for binary classification, it can be extended to multi-class problems using the softmax function, which is a generalization of the sigmoid for multiple classes.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0)

# Example with 3 classes
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)

print("Input scores:", scores)
print("Output probabilities:", probabilities)
print("Sum of probabilities:", np.sum(probabilities))
```

Slide 14: Sigmoid vs. Logit: Inverse Operations

The logit function is the inverse of the sigmoid function. Understanding this relationship can be useful in various machine learning contexts, such as interpreting logistic regression coefficients.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logit(p):
    return np.log(p / (1 - p))

x = np.linspace(0.01, 0.99, 100)
y = logit(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(y))
plt.title('sigmoid(logit(x))')
plt.xlabel('x')
plt.ylabel('y')
plt.subplot(1, 2, 2)
plt.plot(sigmoid(y), y)
plt.title('logit(sigmoid(x))')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of the sigmoid function and its applications in machine learning:

1. "Understanding the Difficulty of Training Deep Feedforward Neural Networks" by Xavier Glorot and Yoshua Bengio (2010) ArXiv: [https://arxiv.org/abs/1001.3218](https://arxiv.org/abs/1001.3218)
2. "Efficient BackProp" by Yann LeCun, Leon Bottou, Genevieve B. Orr, and Klaus-Robert Müller (1998) ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
3. "Rectified Linear Units Improve Restricted Boltzmann Machines" by Vinod Nair and Geoffrey E. Hinton (2010) ICML Proceedings: [https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)

These resources provide deeper insights into activation functions, their properties, and their impact on neural network training.

