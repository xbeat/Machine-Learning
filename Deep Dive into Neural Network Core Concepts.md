## Deep Dive into Neural Network Core Concepts
Slide 1: Introduction to Neural Networks

Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) organized in layers, designed to recognize patterns and solve complex problems. This slideshow will explore the fundamental concepts and mathematical foundations of neural networks, focusing on the chain rule, gradient descent, and backpropagation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple neural network architecture
input_layer = 3
hidden_layer = 4
output_layer = 2

# Initialize random weights
W1 = np.random.randn(input_layer, hidden_layer)
W2 = np.random.randn(hidden_layer, output_layer)

# Visualize the network
plt.figure(figsize=(10, 6))
for i in range(input_layer):
    for j in range(hidden_layer):
        plt.plot([0, 1], [i, j], 'b-')
        for k in range(output_layer):
            plt.plot([1, 2], [j, k], 'r-')

plt.title("Simple Neural Network Architecture")
plt.axis('off')
plt.show()
```

Slide 2: Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Common activation functions include sigmoid, tanh, and ReLU (Rectified Linear Unit). These functions determine whether a neuron should be activated based on its input.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.plot(x, relu(x), label='ReLU')
plt.legend()
plt.title("Common Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
```

Slide 3: Forward Propagation

Forward propagation is the process of passing input data through the neural network to generate predictions. It involves matrix multiplication of inputs with weights, followed by the application of activation functions at each layer.

```python
import numpy as np

def forward_propagation(X, W1, W2):
    # Hidden layer
    Z1 = np.dot(X, W1)
    A1 = np.maximum(0, Z1)  # ReLU activation
    
    # Output layer
    Z2 = np.dot(A1, W2)
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
    
    return A2

# Example usage
X = np.array([[0.1, 0.2, 0.3]])
W1 = np.random.randn(3, 4)
W2 = np.random.randn(4, 2)

output = forward_propagation(X, W1, W2)
print("Network output:", output)
```

Slide 4: The Chain Rule

The chain rule is a fundamental concept in calculus that allows us to compute the derivative of composite functions. In neural networks, it's crucial for backpropagation, enabling the calculation of gradients through multiple layers.

```python
import sympy as sp

# Define variables
x, y, z = sp.symbols('x y z')

# Define composite function
f = sp.exp(x**2 + y**2)
g = sp.sin(z)
h = f * g

# Calculate partial derivatives using the chain rule
dh_dx = sp.diff(h, x)
dh_dy = sp.diff(h, y)
dh_dz = sp.diff(h, z)

print("∂h/∂x =", dh_dx)
print("∂h/∂y =", dh_dy)
print("∂h/∂z =", dh_dz)
```

Slide 5: Gradient Descent

Gradient descent is an optimization algorithm used to minimize the loss function in neural networks. It iteratively adjusts the network's parameters (weights and biases) in the direction of steepest descent of the loss function.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5*np.sin(x)

def df(x):
    return 2*x + 5*np.cos(x)

def gradient_descent(start, learn_rate, num_iterations):
    x = start
    x_history = [x]
    
    for _ in range(num_iterations):
        x = x - learn_rate * df(x)
        x_history.append(x)
    
    return x, x_history

x_min, x_history = gradient_descent(start=3, learn_rate=0.1, num_iterations=50)

x = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', label='f(x)')
plt.plot(x_history, [f(x) for x in x_history], 'ro-', label='Gradient descent path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Gradient Descent Optimization')
plt.show()

print(f"Minimum found at x = {x_min:.2f}")
```

Slide 6: Backpropagation

Backpropagation is an algorithm that efficiently computes gradients in neural networks. It applies the chain rule to propagate the error backward through the network, updating weights to minimize the loss function.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Example usage
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(X, y)

for _ in range(1500):
    nn.feedforward()
    nn.backprop()

print("Output after training:")
print(nn.output)
```

Slide 7: Loss Functions

Loss functions measure the difference between predicted and actual outputs. They guide the learning process by quantifying the network's performance. Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy for classification tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_true = np.array([0, 1, 1, 0])
y_pred = np.linspace(0, 1, 100)

mse_loss = [mse(y_true, np.full_like(y_true, p)) for p in y_pred]
ce_loss = [cross_entropy(y_true, np.full_like(y_true, p)) for p in y_pred]

plt.figure(figsize=(10, 6))
plt.plot(y_pred, mse_loss, label='MSE')
plt.plot(y_pred, ce_loss, label='Cross-Entropy')
plt.xlabel('Predicted Value')
plt.ylabel('Loss')
plt.title('Comparison of Loss Functions')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Regularization Techniques

Regularization helps prevent overfitting in neural networks by adding a penalty term to the loss function. Common techniques include L1 (Lasso) and L2 (Ridge) regularization, which encourage simpler models with smaller weights.

```python
import numpy as np
import matplotlib.pyplot as plt

def model(X, weights):
    return np.dot(X, weights)

def loss(X, y, weights, lambda_reg, reg_type):
    predictions = model(X, weights)
    mse = np.mean((y - predictions) ** 2)
    if reg_type == 'L1':
        reg_term = lambda_reg * np.sum(np.abs(weights))
    elif reg_type == 'L2':
        reg_term = lambda_reg * np.sum(weights ** 2)
    else:
        reg_term = 0
    return mse + reg_term

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 2

weights = np.linspace(-10, 10, 100)
losses_no_reg = [loss(X, y, w, 0, None) for w in weights]
losses_l1 = [loss(X, y, w, 0.1, 'L1') for w in weights]
losses_l2 = [loss(X, y, w, 0.1, 'L2') for w in weights]

plt.figure(figsize=(10, 6))
plt.plot(weights, losses_no_reg, label='No Regularization')
plt.plot(weights, losses_l1, label='L1 Regularization')
plt.plot(weights, losses_l2, label='L2 Regularization')
plt.xlabel('Weight')
plt.ylabel('Loss')
plt.title('Effect of Regularization on Loss')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: Optimizers

Optimizers are algorithms that adjust the learning process to improve convergence and performance. Popular optimizers include Stochastic Gradient Descent (SGD), Adam, and RMSprop. They adapt the learning rate and momentum to navigate the loss landscape efficiently.

```python
import numpy as np
import matplotlib.pyplot as plt

def optimize(optimizer, init_params, iterations):
    params = init_params.()
    path = [params.()]
    
    for _ in range(iterations):
        gradient = 2 * params
        params = optimizer(params, gradient)
        path.append(params.())
    
    return np.array(path)

def sgd(params, gradient, learning_rate=0.1):
    return params - learning_rate * gradient

def momentum(params, gradient, v=None, learning_rate=0.1, beta=0.9):
    if v is None:
        v = np.zeros_like(params)
    v = beta * v + (1 - beta) * gradient
    return params - learning_rate * v

def adam(params, gradient, m=None, v=None, t=0, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    if m is None:
        m = np.zeros_like(params)
        v = np.zeros_like(params)
    t += 1
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

init_params = np.array([3.0, 3.0])
iterations = 50

sgd_path = optimize(sgd, init_params, iterations)
momentum_path = optimize(lambda p, g: momentum(p, g), init_params, iterations)
adam_path = optimize(lambda p, g: adam(p, g), init_params, iterations)

plt.figure(figsize=(10, 8))
plt.contour(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100), 
            lambda x, y: x**2 + y**2, levels=30)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'r.-', label='SGD')
plt.plot(momentum_path[:, 0], momentum_path[:, 1], 'g.-', label='Momentum')
plt.plot(adam_path[:, 0], adam_path[:, 1], 'b.-', label='Adam')
plt.legend()
plt.title('Optimizer Comparison')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 10: Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to extract features, pooling layers to reduce dimensionality, and fully connected layers for classification or regression tasks.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def convolve2d(image, kernel):
    return signal.convolve2d(image, kernel, mode='valid')

# Sample image
image = np.random.rand(8, 8)

# Edge detection kernel
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

# Apply convolution
conv_result = convolve2d(image, edge_kernel)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(edge_kernel, cmap='gray')
ax2.set_title('Edge Detection Kernel')
ax2.axis('off')

ax3.imshow(conv_result, cmap='gray')
ax3.set_title('Convolution Result')
ax3.axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: Recurrent Neural Networks (RNNs)

RNNs are designed to process sequential data by maintaining an internal state (memory). They are well-suited for tasks like natural language processing, time series analysis, and speech recognition. Long Short-Term Memory (LSTM) networks are a popular variant of RNNs.

```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleRNN:
    def __init__(self, hidden_size, input_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        hidden_states = []
        for x in inputs:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hidden_states.append(h)
        return hidden_states

# Example usage
rnn = SimpleRNN(hidden_size=4, input_size=3)
inputs = [np.random.randn(3, 1) for _ in range(5)]
hidden_states = rnn.forward(inputs)

plt.figure(figsize=(10, 6))
for i, h in enumerate(hidden_states):
    plt.plot(h.flatten(), label=f'Time step {i+1}')
plt.title('RNN Hidden States Over Time')
plt.xlabel('Hidden Unit')
plt.ylabel('Activation')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Transfer Learning

Transfer learning is a technique where a pre-trained model is used as a starting point for a new task. This approach can significantly reduce training time and improve performance, especially when working with limited data.

```python
import numpy as np
import matplotlib.pyplot as plt

def pretrained_model(x):
    # Simulated pre-trained model
    return np.sin(x) + 0.1 * np.random.randn(*x.shape)

def transfer_learn(x, y, pretrained_model, epochs=1000, lr=0.01):
    # Initialize weights
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    losses = []
    for _ in range(epochs):
        # Forward pass
        y_pred = w * pretrained_model(x) + b
        
        # Compute loss
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)
        
        # Backward pass
        dw = -2 * np.mean((y - y_pred) * pretrained_model(x))
        db = -2 * np.mean(y - y_pred)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, b, losses

# Generate data
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * np.sin(x) + 0.5 + 0.2 * np.random.randn(*x.shape)

# Apply transfer learning
w, b, losses = transfer_learn(x, y, pretrained_model)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(x, y, 'b.', label='True data')
plt.plot(x, pretrained_model(x), 'g-', label='Pre-trained model')
plt.plot(x, w * pretrained_model(x) + b, 'r-', label='Transfer learned model')
plt.legend()
plt.title('Transfer Learning Results')

plt.subplot(122)
plt.plot(losses)
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```

Slide 13: Generative Adversarial Networks (GANs)

GANs consist of two neural networks, a generator and a discriminator, competing against each other. The generator creates fake data, while the discriminator tries to distinguish between real and fake data. This adversarial process leads to the generation of highly realistic synthetic data.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_real_samples(n):
    # Generate samples from a normal distribution
    return np.random.normal(loc=4, scale=1.5, size=(n, 1))

def generate_fake_samples(generator, n):
    # Generate random noise as input to the generator
    noise = np.random.normal(0, 1, (n, 1))
    # Use the generator to produce fake samples
    return generator(noise)

def generator(x):
    # Simple generator function
    return 2 * x + 3

def discriminator(x):
    # Simple discriminator function
    return 1 / (1 + np.exp(-(x - 4)))

# Generate samples
n_samples = 1000
real_samples = generate_real_samples(n_samples)
fake_samples = generate_fake_samples(generator, n_samples)

# Evaluate samples with discriminator
real_predictions = discriminator(real_samples)
fake_predictions = discriminator(fake_samples)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(real_samples, bins=50, alpha=0.7, label='Real data')
plt.hist(fake_samples, bins=50, alpha=0.7, label='Fake data')
plt.legend()
plt.title('Distribution of Real and Fake Samples')

plt.subplot(122)
plt.scatter(real_samples, real_predictions, alpha=0.5, label='Real')
plt.scatter(fake_samples, fake_predictions, alpha=0.5, label='Fake')
plt.legend()
plt.title('Discriminator Predictions')
plt.xlabel('Sample Value')
plt.ylabel('Probability of being real')

plt.tight_layout()
plt.show()
```

Slide 14: Real-life Example: Image Classification

Image classification is a common application of neural networks. In this example, we'll use a simple CNN to classify handwritten digits from the MNIST dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# Build the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_split=0.2, batch_size=128, verbose=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 15: Real-life Example: Natural Language Processing

Neural networks are widely used in natural language processing tasks. This example demonstrates a simple sentiment analysis model using a recurrent neural network (LSTM) to classify movie reviews as positive or negative.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the IMDB dataset
max_features = 10000
maxlen = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform length
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build the LSTM model
model = Sequential([
    Embedding(max_features, 128),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 16: Additional Resources

For those interested in diving deeper into neural networks and deep learning, here are some valuable resources:

1. ArXiv.org: A repository of research papers on various topics in machine learning and artificial intelligence. Example: "Attention Is All You Need" by Vaswani et al. (2017) ArXiv URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Deep Learning Book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville Available online at: [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/)
3. Stanford CS231n: Convolutional Neural Networks for Visual Recognition Course materials available at: [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)
4. FastAI: Practical Deep Learning for Coders Free online course: [https://course.fast.ai/](https://course.fast.ai/)
5. TensorFlow and PyTorch documentation for implementing neural networks

These resources provide in-depth explanations, tutorials, and practical examples to further your understanding of neural networks and their applications.
