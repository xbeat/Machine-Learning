## Explaining Back Propagation with Python
Slide 1: Introduction to Back Propagation

Back propagation is a fundamental algorithm in training neural networks. It's the backbone of how these networks learn from data, adjusting their weights to minimize errors and improve predictions. This slideshow will explore the mathematics behind back propagation and demonstrate its implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Simple neural network architecture
input_layer = 2
hidden_layer = 3
output_layer = 1

# Initialize random weights
hidden_weights = np.random.uniform(size=(input_layer, hidden_layer))
output_weights = np.random.uniform(size=(hidden_layer, output_layer))

print("Initial hidden weights:\n", hidden_weights)
print("Initial output weights:\n", output_weights)
```

Slide 2: The Forward Pass

In the forward pass, input data propagates through the network, layer by layer. Each neuron receives inputs, applies weights, sums the results, and passes them through an activation function. This process continues until the output layer is reached.

```python
def forward_pass(X):
    hidden_layer_activation = sigmoid(np.dot(X, hidden_weights))
    output = sigmoid(np.dot(hidden_layer_activation, output_weights))
    return hidden_layer_activation, output

# Example input
X = np.array([[0.1, 0.2]])
hidden_activation, output = forward_pass(X)

print("Hidden layer activation:", hidden_activation)
print("Output:", output)
```

Slide 3: The Loss Function

The loss function quantifies the difference between the network's predictions and the actual target values. Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy for classification tasks. Here, we'll use MSE.

```python
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example target
y_true = np.array([[0.3]])
loss = mean_squared_error(y_true, output)

print("Loss:", loss)
```

Slide 4: The Backward Pass: Output Layer

The backward pass starts at the output layer. We calculate the error and its gradient with respect to the output layer's weights. This gradient tells us how to adjust the weights to reduce the error.

```python
output_error = y_true - output
output_delta = output_error * sigmoid_derivative(output)

print("Output error:", output_error)
print("Output delta:", output_delta)
```

Slide 5: The Backward Pass: Hidden Layer

We then propagate the error backward to the hidden layer. This involves calculating how much each hidden neuron contributed to the output error, and how the hidden layer weights should be adjusted.

```python
hidden_error = np.dot(output_delta, output_weights.T)
hidden_delta = hidden_error * sigmoid_derivative(hidden_activation)

print("Hidden error:", hidden_error)
print("Hidden delta:", hidden_delta)
```

Slide 6: Weight Updates

Using the calculated deltas, we update the weights of both the output and hidden layers. The learning rate determines the size of these updates.

```python
learning_rate = 0.1

output_weights += learning_rate * np.dot(hidden_activation.T, output_delta)
hidden_weights += learning_rate * np.dot(X.T, hidden_delta)

print("Updated output weights:\n", output_weights)
print("Updated hidden weights:\n", hidden_weights)
```

Slide 7: The Complete Training Loop

We combine all these steps into a complete training loop. This process repeats for multiple epochs, gradually improving the network's performance.

```python
def train(X, y, epochs=1000):
    for _ in range(epochs):
        hidden_activation, output = forward_pass(X)
        
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = np.dot(output_delta, output_weights.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_activation)
        
        output_weights += learning_rate * np.dot(hidden_activation.T, output_delta)
        hidden_weights += learning_rate * np.dot(X.T, hidden_delta)
    
    return output

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

final_output = train(X, y)
print("Final predictions:\n", final_output)
```

Slide 8: Visualizing the Learning Process

To better understand how the network learns, we can visualize the loss over time during training.

```python
def train_with_history(X, y, epochs=1000):
    loss_history = []
    for _ in range(epochs):
        hidden_activation, output = forward_pass(X)
        loss = mean_squared_error(y, output)
        loss_history.append(loss)
        
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = np.dot(output_delta, output_weights.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_activation)
        
        output_weights += learning_rate * np.dot(hidden_activation.T, output_delta)
        hidden_weights += learning_rate * np.dot(X.T, hidden_delta)
    
    return loss_history

loss_history = train_with_history(X, y)

plt.plot(loss_history)
plt.title('Loss over time')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.show()
```

Slide 9: The Chain Rule in Back Propagation

Back propagation relies heavily on the chain rule from calculus. This rule allows us to compute the gradient of the loss with respect to each weight in the network, even in deep architectures.

```python
def chain_rule_example(x):
    # f(x) = (x^2 + 1)^3
    # f'(x) = 3(x^2 + 1)^2 * 2x
    
    inner_function = x**2 + 1
    outer_function = inner_function**3
    
    inner_derivative = 2*x
    outer_derivative = 3 * inner_function**2
    
    total_derivative = outer_derivative * inner_derivative
    
    return outer_function, total_derivative

x = 2
result, derivative = chain_rule_example(x)
print(f"f({x}) = {result}")
print(f"f'({x}) = {derivative}")
```

Slide 10: Gradient Descent Optimization

Back propagation is typically used in conjunction with gradient descent optimization. This algorithm iteratively adjusts the weights in the direction that minimizes the loss function.

```python
def gradient_descent(start, gradient, learn_rate, n_iter):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        vector += diff
    return vector

# Example: finding the minimum of f(x) = x^2 + 5
def gradient(x):
    return 2 * x

minimum = gradient_descent(start=10.0, gradient=gradient, learn_rate=0.1, n_iter=100)
print(f"The minimum occurs at x = {minimum}")
```

Slide 11: Handling Vanishing and Exploding Gradients

In deep networks, gradients can become very small (vanishing) or very large (exploding) as they propagate backward. This can lead to slow learning or instability. Techniques like careful weight initialization and using activation functions like ReLU can help mitigate these issues.

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Compare gradients for deep networks with sigmoid vs ReLU
def compare_gradients(depth):
    x = np.linspace(-5, 5, 100)
    
    sigmoid_grad = sigmoid_derivative(x)
    for _ in range(depth - 1):
        sigmoid_grad = np.dot(sigmoid_grad, sigmoid_derivative(x))
    
    relu_grad = relu_derivative(x)
    for _ in range(depth - 1):
        relu_grad = np.dot(relu_grad, relu_derivative(x))
    
    plt.plot(x, sigmoid_grad, label='Sigmoid')
    plt.plot(x, relu_grad, label='ReLU')
    plt.title(f'Gradients in {depth}-layer network')
    plt.legend()
    plt.show()

compare_gradients(10)  # Compare gradients in a 10-layer network
```

Slide 12: Real-life Example: Image Classification

Back propagation is crucial in training convolutional neural networks (CNNs) for image classification tasks. Here's a simplified example using the MNIST dataset.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=0)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

Slide 13: Real-life Example: Natural Language Processing

Back propagation is also essential in training recurrent neural networks (RNNs) for natural language processing tasks like sentiment analysis. Here's a simplified example using IMDB movie reviews.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load and preprocess the IMDB dataset
max_features = 10000
maxlen = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Create a simple RNN model
model = Sequential([
    Embedding(max_features, 128),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=0)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into the mathematics and implementation of back propagation, here are some valuable resources:

1. "Gradient-Based Learning Applied to Document Recognition" by LeCun et al. (1998) - A seminal paper on convolutional neural networks and back propagation. ArXiv: [https://arxiv.org/abs/1998.01365](https://arxiv.org/abs/1998.01365)
2. "Understanding the difficulty of training deep feedforward neural networks" by Glorot and Bengio (2010) - Explores weight initialization techniques to improve back propagation. ArXiv: [https://arxiv.org/abs/1008.3024](https://arxiv.org/abs/1008.3024)
3. "Deep Learning" by Goodfellow, Bengio, and Courville - A comprehensive textbook covering back propagation and other deep learning concepts.
4. "Neural Networks and Deep Learning" by Michael Nielsen - A free online book with interactive visualizations of back propagation.
5. "Backpropagation Algorithm" by Andrew Ng - A detailed lecture series on Coursera explaining the mathematics behind back propagation.

