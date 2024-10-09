## Activation Functions The Driving Force of Neural Networks
Slide 1: Activation Functions: The Spark of Neural Networks

Activation functions are a crucial component in neural networks, acting as the non-linear transformation that allows these networks to learn complex patterns. They introduce non-linearity into the network, enabling it to approximate any function and solve non-trivial problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_function(func, x_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    plt.plot(x, y)
    plt.title(func.__name__)
    plt.grid(True)
    plt.show()

# We'll use this to visualize different activation functions
```

Slide 2: The Linear Neuron Problem

Without activation functions, neural networks would be limited to linear transformations, regardless of their depth. This would make them no more powerful than a single layer perceptron.

```python
def linear_neuron(x):
    return 2 * x + 1

plot_function(linear_neuron, (-5, 5))
```

This graph shows a linear function, which is what we'd get without activation functions. No matter how many layers we stack, we'd still only be able to represent linear relationships.

Slide 3: Introducing Non-linearity

Activation functions introduce non-linearity into the network, allowing it to learn and represent complex, non-linear relationships in the data.

```python
def relu(x):
    return np.maximum(0, x)

plot_function(relu, (-5, 5))
```

This graph shows the ReLU (Rectified Linear Unit) activation function, a popular choice that introduces non-linearity while being computationally efficient.

Slide 4: Sigmoid Activation Function

The sigmoid function was one of the earliest activation functions used in neural networks. It squashes the input into a range between 0 and 1.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

plot_function(sigmoid, (-10, 10))
```

The sigmoid function is smooth and differentiable, making it suitable for gradient-based optimization methods. However, it suffers from the vanishing gradient problem for very large or small inputs.

Slide 5: Hyperbolic Tangent (tanh) Activation

The tanh function is similar to sigmoid but maps inputs to the range \[-1, 1\]. It's often preferred over sigmoid as it's zero-centered.

```python
def tanh(x):
    return np.tanh(x)

plot_function(tanh, (-5, 5))
```

Tanh addresses some issues of sigmoid, but still suffers from the vanishing gradient problem at extreme values.

Slide 6: ReLU (Rectified Linear Unit)

ReLU has become the most widely used activation function due to its simplicity and effectiveness in deep networks.

```python
def relu(x):
    return np.maximum(0, x)

plot_function(relu, (-5, 5))
```

ReLU is computationally efficient and helps mitigate the vanishing gradient problem. However, it can suffer from the "dying ReLU" problem where neurons can get stuck in an inactive state.

Slide 7: Leaky ReLU

Leaky ReLU is a variant of ReLU that allows a small, non-zero gradient when the input is negative.

```python
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

plot_function(leaky_relu, (-5, 5))
```

Leaky ReLU helps prevent the dying ReLU problem by allowing a small gradient for negative inputs.

Slide 8: Softmax Activation

Softmax is commonly used in the output layer of multi-class classification problems. It converts a vector of numbers into a probability distribution.

```python
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Example usage
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)
print(f"Scores: {scores}")
print(f"Probabilities: {probabilities}")
```

This code demonstrates how softmax converts raw scores into probabilities that sum to 1.

Slide 9: Activation Functions and Gradients

The choice of activation function affects the gradients flowing through the network during backpropagation.

```python
def plot_function_and_derivative(func, x_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = func(x)
    dy = np.gradient(y, x)
    
    plt.plot(x, y, label='Function')
    plt.plot(x, dy, label='Derivative')
    plt.title(f"{func.__name__} and its derivative")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_function_and_derivative(sigmoid, (-10, 10))
```

This graph shows the sigmoid function and its derivative. Notice how the gradient becomes very small for large positive or negative inputs, leading to the vanishing gradient problem.

Slide 10: Activation Functions in Practice

Let's implement a simple neural network with different activation functions to see how they affect learning.

```python
import tensorflow as tf

def create_model(activation):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation, input_shape=(784,)),
        tf.keras.layers.Dense(64, activation=activation),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0

# Train models with different activation functions
activations = ['relu', 'sigmoid', 'tanh']
histories = {}

for activation in activations:
    model = create_model(activation)
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, verbose=0)
    histories[activation] = history.history['val_accuracy'][-1]

print("Validation accuracies:")
for activation, accuracy in histories.items():
    print(f"{activation}: {accuracy:.4f}")
```

This code trains simple neural networks on the MNIST dataset using different activation functions, allowing us to compare their performance.

Slide 11: Real-life Example: Image Classification

Activation functions play a crucial role in image classification tasks. Let's consider a convolutional neural network (CNN) for classifying images of animals.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Assuming we have a dataset of animal images
# (x_train, y_train), (x_test, y_test) = load_animal_dataset()

# model = create_cnn_model()
# history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

In this CNN, ReLU activation is used in the convolutional and dense layers to introduce non-linearity, while softmax is used in the output layer for multi-class classification.

Slide 12: Real-life Example: Natural Language Processing

Activation functions are also crucial in natural language processing tasks. Here's an example of a simple sentiment analysis model:

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential

def create_sentiment_model(vocab_size, embedding_dim, max_length):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
# vocab_size = 10000
# embedding_dim = 16
# max_length = 100
# model = create_sentiment_model(vocab_size, embedding_dim, max_length)
# model.summary()
```

In this sentiment analysis model, we use ReLU activation in the dense layer and sigmoid activation in the output layer for binary classification.

Slide 13: Choosing the Right Activation Function

The choice of activation function depends on various factors:

1. Problem type (regression, binary classification, multi-class classification)
2. Network architecture
3. Desired properties (e.g., range of output values)
4. Computational efficiency

There's no one-size-fits-all solution, and experimentation is often necessary to find the best activation function for a specific task.

```python
def experiment_with_activations(x_train, y_train, x_test, y_test, activations):
    results = {}
    for activation in activations:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=activation, input_shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[activation] = test_acc
    return results

# Example usage:
# activations = ['relu', 'tanh', 'sigmoid', 'elu']
# results = experiment_with_activations(x_train, y_train, x_test, y_test, activations)
# for activation, accuracy in results.items():
#     print(f"{activation}: {accuracy:.4f}")
```

This function allows you to experiment with different activation functions on a given dataset, helping you choose the best one for your specific problem.

Slide 14: Advanced Activation Functions

Research in neural networks has led to the development of more advanced activation functions:

1. Swish: f(x) = x \* sigmoid(x)
2. GELU (Gaussian Error Linear Unit): Smoother approximation of ReLU
3. Mish: A self-regularized non-monotonic activation function

```python
def swish(x, beta=1.0):
    return x * tf.sigmoid(beta * x)

def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def mish(x):
    return x * tf.tanh(tf.math.softplus(x))

x = tf.linspace(-5, 5, 100)
plt.figure(figsize=(12, 4))
plt.plot(x, swish(x), label='Swish')
plt.plot(x, gelu(x), label='GELU')
plt.plot(x, mish(x), label='Mish')
plt.legend()
plt.title('Advanced Activation Functions')
plt.grid(True)
plt.show()
```

This code plots these advanced activation functions, showcasing their unique properties.

Slide 15: Additional Resources

For more in-depth information on activation functions and their role in neural networks, consider exploring these resources:

1. "Understanding Activation Functions in Neural Networks" by Avinash Sharma V (arXiv:1709.04626) [https://arxiv.org/abs/1709.04626](https://arxiv.org/abs/1709.04626)
2. "Activation Functions: Comparison of Trends in Practice and Research for Deep Learning" by Chigozie Enyinna Nwankpa et al. (arXiv:1811.03378) [https://arxiv.org/abs/1811.03378](https://arxiv.org/abs/1811.03378)
3. "Mish: A Self Regularized Non-Monotonic Activation Function" by Diganta Misra (arXiv:1908.08681) [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)

These papers provide comprehensive overviews and comparisons of various activation functions, as well as insights into recent developments in the field.

