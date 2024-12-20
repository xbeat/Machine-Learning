## Simple Neural Network Implementation in Python
Slide 1: Introduction to Neural Networks

Neural networks are computational models inspired by the human brain. They consist of interconnected nodes (neurons) that process and transmit information. In this presentation, we'll build a neural network from scratch using Python.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

# Create a simple neural network
nn = NeuralNetwork(2, 3, 1)
print("Input to Hidden Weights:")
print(nn.W1)
print("\nHidden to Output Weights:")
print(nn.W2)
```

Slide 2: Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. We'll implement the sigmoid function and its derivative.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Test the sigmoid function
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.grid(True)
plt.show()
```

Slide 3: Forward Propagation

Forward propagation is the process of passing input data through the network to generate predictions. We'll implement this in our NeuralNetwork class.

```python
class NeuralNetwork:
    # ... (previous code)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

# Test forward propagation
nn = NeuralNetwork(2, 3, 1)
X = np.array([[0.5, 0.1]])
output = nn.forward(X)
print("Network output:", output)
```

Slide 4: Loss Function

The loss function measures how well our network is performing. We'll use the mean squared error (MSE) loss function.

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Test the MSE loss function
y_true = np.array([[1], [0], [1]])
y_pred = np.array([[0.9], [0.1], [0.8]])

loss = mse_loss(y_true, y_pred)
print("MSE Loss:", loss)
```

Slide 5: Backpropagation

Backpropagation is the algorithm used to calculate gradients and update weights. We'll implement this in our NeuralNetwork class.

```python
class NeuralNetwork:
    # ... (previous code)

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # Output layer
        dZ2 = self.a2 - y
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Initialize a neural network and perform one backward pass
nn = NeuralNetwork(2, 3, 1)
X = np.array([[0.5, 0.1]])
y = np.array([[1]])
nn.forward(X)
nn.backward(X, y, learning_rate=0.1)
```

Slide 6: Training Loop

Now we'll combine forward propagation, loss calculation, and backpropagation into a training loop.

```python
class NeuralNetwork:
    # ... (previous code)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Compute loss
            loss = mse_loss(y, output)

            # Backpropagation
            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Train the neural network on a simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR function

nn = NeuralNetwork(2, 4, 1)
nn.train(X, y, epochs=1000, learning_rate=0.1)
```

Slide 7: Making Predictions

After training, we can use our neural network to make predictions on new data.

```python
class NeuralNetwork:
    # ... (previous code)

    def predict(self, X):
        return self.forward(X)

# Make predictions using the trained network
nn = NeuralNetwork(2, 4, 1)
nn.train(X, y, epochs=5000, learning_rate=0.1)

test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.predict(test_data)

for input_data, prediction in zip(test_data, predictions):
    print(f"Input: {input_data}, Prediction: {prediction[0]:.4f}")
```

Slide 8: Visualizing the Decision Boundary

Let's visualize how our neural network separates the input space for the XOR problem.

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("XOR Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, nn)
```

Slide 9: Adding Regularization

To prevent overfitting, we can add L2 regularization to our neural network.

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01):
        # ... (previous initialization code)
        self.lambda_reg = lambda_reg

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # ... (previous backward code)

        # Add L2 regularization terms
        dW2 += (self.lambda_reg / m) * self.W2
        dW1 += (self.lambda_reg / m) * self.W1

        # ... (previous weight update code)

# Train the network with regularization
nn_reg = NeuralNetwork(2, 4, 1, lambda_reg=0.1)
nn_reg.train(X, y, epochs=5000, learning_rate=0.1)

plot_decision_boundary(X, y, nn_reg)
```

Slide 10: Mini-batch Gradient Descent

To improve training efficiency, we can implement mini-batch gradient descent.

```python
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size

    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, y_mini))

    return mini_batches

class NeuralNetwork:
    # ... (previous code)

    def train(self, X, y, epochs, learning_rate, batch_size):
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X, y, batch_size)
            
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch
                self.forward(X_mini)
                self.backward(X_mini, y_mini, learning_rate)
            
            if epoch % 100 == 0:
                loss = mse_loss(y, self.predict(X))
                print(f"Epoch {epoch}, Loss: {loss}")

# Train the network using mini-batch gradient descent
nn_mini_batch = NeuralNetwork(2, 4, 1)
nn_mini_batch.train(X, y, epochs=5000, learning_rate=0.1, batch_size=2)
```

Slide 11: Adding Momentum

Momentum can help accelerate gradient descent and dampen oscillations. Let's implement momentum in our neural network.

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01, momentum=0.9):
        # ... (previous initialization code)
        self.momentum = momentum
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # ... (previous gradient calculation code)

        # Update velocities
        self.v_W2 = self.momentum * self.v_W2 + learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 + learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 + learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 + learning_rate * db1

        # Update weights and biases
        self.W2 -= self.v_W2
        self.b2 -= self.v_b2
        self.W1 -= self.v_W1
        self.b1 -= self.v_b1

# Train the network with momentum
nn_momentum = NeuralNetwork(2, 4, 1, momentum=0.9)
nn_momentum.train(X, y, epochs=5000, learning_rate=0.1, batch_size=2)
```

Slide 12: Real-life Example: Handwritten Digit Recognition

Let's apply our neural network to recognize handwritten digits using the MNIST dataset.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoded vectors
def to_one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_onehot = to_one_hot(y_train, 10)
y_test_onehot = to_one_hot(y_test, 10)

# Create and train the neural network
nn_digits = NeuralNetwork(64, 32, 10, lambda_reg=0.01, momentum=0.9)
nn_digits.train(X_train, y_train_onehot, epochs=1000, learning_rate=0.1, batch_size=32)

# Make predictions
y_pred = nn_digits.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"True: {y_test[i]}, Pred: {y_pred_classes[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 13: Real-life Example: Image Classification

Let's use our neural network for a simple image classification task using the CIFAR-10 dataset.

```python
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

y_train_onehot = to_one_hot(y_train.flatten(), 10)
y_test_onehot = to_one_hot(y_test.flatten(), 10)

# Create and train the neural network
nn_cifar = NeuralNetwork(3072, 128, 10, lambda_reg=0.01, momentum=0.9)
nn_cifar.train(X_train[:5000], y_train_onehot[:5000], epochs=100, learning_rate=0.01, batch_size=64)

# Make predictions
y_pred = nn_cifar.predict(X_test[:1000])
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test[:1000], y_pred_classes)
print(f"Accuracy: {accuracy:.4f}")

# Visualize some predictions
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, ax in enumerate(axes.flat):
    if i < 20:
        ax.imshow(X_test[i].reshape(32, 32, 3))
        ax.set_title(f"True: {class_names[y_test[i][0]]}\nPred: {class_names[y_pred_classes[i]]}")
        ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 14: Improving Model Performance

To enhance our neural network's performance, we can implement additional techniques:

1. Add more hidden layers
2. Use different activation functions (e.g., ReLU)
3. Implement dropout regularization
4. Apply batch normalization
5. Use adaptive learning rate methods (e.g., Adam optimizer)

Here's a pseudocode example of how to implement some of these improvements:

```python
class ImprovedNeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', dropout_rate=0.5):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append({
                'W': np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i-1]),
                'b': np.zeros((1, layer_sizes[i])),
                'activation': activation,
                'dropout_rate': dropout_rate
            })

    def forward(self, X, training=True):
        self.activations = [X]
        for layer in self.layers:
            Z = np.dot(self.activations[-1], layer['W']) + layer['b']
            A = self.activate(Z, layer['activation'])
            if training:
                A = self.apply_dropout(A, layer['dropout_rate'])
            self.activations.append(A)
        return self.activations[-1]

    def activate(self, Z, activation):
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        # Add more activation functions as needed

    def apply_dropout(self, A, dropout_rate):
        mask = np.random.rand(*A.shape) > dropout_rate
        return (A * mask) / (1 - dropout_rate)

    # Implement backward pass, parameter updates, and training loop
```

Slide 15: Additional Resources

For further exploration of neural networks and deep learning, consider these resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ArXiv: [https://arxiv.org/abs/1606.01781](https://arxiv.org/abs/1606.01781)
2. "Neural Networks and Deep Learning" by Michael Nielsen (Free online book)
3. "Efficient BackProp" by Yann LeCun et al. ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
4. "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
5. Stanford CS231n: Convolutional Neural Networks for Visual Recognition (Course materials available online)

These resources provide in-depth explanations and advanced techniques for neural network implementation and optimization.