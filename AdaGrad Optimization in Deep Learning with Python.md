## AdaGrad Optimization in Deep Learning with Python
Slide 1: Introduction to AdaGrad

AdaGrad (Adaptive Gradient Algorithm) is an optimization algorithm used in deep learning to automatically adapt the learning rate for each parameter. It's particularly effective for sparse data and can help accelerate convergence in neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def adagrad(gradient, x, learning_rate, epsilon=1e-8):
    state = np.zeros_like(x)
    state += gradient**2
    x -= learning_rate * gradient / (np.sqrt(state) + epsilon)
    return x, state

# Example usage
x = np.array([0.0, 0.0])
learning_rate = 0.1
epsilon = 1e-8

for _ in range(1000):
    gradient = np.array([2*x[0], 2*x[1]])  # Example gradient
    x, _ = adagrad(gradient, x, learning_rate, epsilon)

print(f"Final x: {x}")
```

Slide 2: The AdaGrad Algorithm

AdaGrad maintains a per-parameter learning rate, which is adapted based on the historical gradients for that parameter. This allows it to take larger steps for infrequent parameters and smaller steps for frequent ones.

```python
def adagrad_optimizer(params, learning_rate=0.01, epsilon=1e-8):
    gradients = [np.random.randn(*p.shape) for p in params]  # Example gradients
    states = [np.zeros_like(p) for p in params]
    
    for param, grad, state in zip(params, gradients, states):
        state += grad**2
        param -= learning_rate * grad / (np.sqrt(state) + epsilon)
    
    return params, states

# Initialize parameters
params = [np.random.randn(3, 3), np.random.randn(3, 1)]

# Optimize
for _ in range(100):
    params, _ = adagrad_optimizer(params)

print("Optimized parameters:")
for p in params:
    print(p)
```

Slide 3: AdaGrad's Learning Rate Adaptation

AdaGrad's key feature is its ability to adapt the learning rate for each parameter individually. This is achieved by dividing the learning rate by the square root of the sum of squared historical gradients.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_rate_adaptation():
    gradients = np.random.randn(1000)
    learning_rate = 0.1
    epsilon = 1e-8
    
    adapted_lr = np.zeros_like(gradients)
    sum_squares = 0
    
    for i, grad in enumerate(gradients):
        sum_squares += grad**2
        adapted_lr[i] = learning_rate / (np.sqrt(sum_squares) + epsilon)
    
    plt.plot(adapted_lr)
    plt.title("AdaGrad Learning Rate Adaptation")
    plt.xlabel("Iteration")
    plt.ylabel("Adapted Learning Rate")
    plt.show()

plot_learning_rate_adaptation()
```

Slide 4: Implementing AdaGrad in a Neural Network

Let's implement AdaGrad in a simple neural network for binary classification. We'll use the popular iris dataset and focus on distinguishing between two classes.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X = iris.data[:100]  # First two classes
y = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network with AdaGrad
class NeuralNetworkAdaGrad:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # AdaGrad states
        self.W1_state = np.zeros_like(self.W1)
        self.b1_state = np.zeros_like(self.b1)
        self.W2_state = np.zeros_like(self.W2)
        self.b2_state = np.zeros_like(self.b2)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        delta2 = output - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        delta1 = np.dot(delta2, self.W2.T) * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, learning_rate=0.01, epsilon=1e-8):
        # AdaGrad update
        self.W1_state += dW1**2
        self.W1 -= learning_rate * dW1 / (np.sqrt(self.W1_state) + epsilon)
        
        self.b1_state += db1**2
        self.b1 -= learning_rate * db1 / (np.sqrt(self.b1_state) + epsilon)
        
        self.W2_state += dW2**2
        self.W2 -= learning_rate * dW2 / (np.sqrt(self.W2_state) + epsilon)
        
        self.b2_state += db2**2
        self.b2 -= learning_rate * db2 / (np.sqrt(self.b2_state) + epsilon)
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for _ in range(epochs):
            output = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(X, y, output)
            self.update_params(dW1, db1, dW2, db2, learning_rate)
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Train and evaluate
nn = NeuralNetworkAdaGrad(4, 5, 1)
nn.train(X_train, y_train)

train_accuracy = np.mean(nn.predict(X_train) == y_train)
test_accuracy = np.mean(nn.predict(X_test) == y_test)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")
```

Slide 5: Advantages of AdaGrad

AdaGrad's main advantage is its ability to handle different features with varying frequencies. It gives more weight to rare features and less weight to common ones, which is particularly useful in natural language processing and computer vision tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_sgd_adagrad():
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(1000) * 0.1
    
    # Initialize parameters
    w_sgd = np.zeros(2)
    w_adagrad = np.zeros(2)
    lr = 0.1
    epsilon = 1e-8
    
    # AdaGrad state
    adagrad_state = np.zeros(2)
    
    # Training loop
    n_iterations = 100
    sgd_loss = []
    adagrad_loss = []
    
    for _ in range(n_iterations):
        # Compute gradients
        grad = -2 * np.mean((y - np.dot(X, w_sgd))[:, np.newaxis] * X, axis=0)
        
        # Update SGD
        w_sgd -= lr * grad
        sgd_loss.append(np.mean((y - np.dot(X, w_sgd))**2))
        
        # Update AdaGrad
        adagrad_state += grad**2
        w_adagrad -= lr * grad / (np.sqrt(adagrad_state) + epsilon)
        adagrad_loss.append(np.mean((y - np.dot(X, w_adagrad))**2))
    
    # Plot results
    plt.plot(sgd_loss, label='SGD')
    plt.plot(adagrad_loss, label='AdaGrad')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('SGD vs AdaGrad Convergence')
    plt.legend()
    plt.show()

compare_sgd_adagrad()
```

Slide 6: AdaGrad in Natural Language Processing

AdaGrad is particularly useful in NLP tasks due to the sparse nature of text data. Let's implement a simple sentiment analysis model using AdaGrad optimization.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Sample data
texts = [
    "I love this movie", "Great film", "Awesome acting",
    "Terrible movie", "Waste of time", "Disappointing"
]
labels = [1, 1, 1, 0, 0, 0]

# Preprocess data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LogisticRegressionAdaGrad:
    def __init__(self, input_dim):
        self.w = np.zeros(input_dim)
        self.b = 0
        self.ada_w = np.zeros(input_dim)
        self.ada_b = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y, learning_rate=0.1, epochs=1000, epsilon=1e-8):
        for _ in range(epochs):
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)
            
            dw = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.sum(y_pred - y) / len(y)
            
            self.ada_w += dw**2
            self.ada_b += db**2
            
            self.w -= learning_rate * dw / (np.sqrt(self.ada_w) + epsilon)
            self.b -= learning_rate * db / (np.sqrt(self.ada_b) + epsilon)
    
    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.w) + self.b) > 0.5).astype(int)

# Train and evaluate
model = LogisticRegressionAdaGrad(X_train.shape[1])
model.train(X_train, y_train)

train_accuracy = np.mean(model.predict(X_train) == y_train)
test_accuracy = np.mean(model.predict(X_test) == y_test)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# Example prediction
new_text = ["This movie was amazing"]
new_features = vectorizer.transform(new_text).toarray()
prediction = model.predict(new_features)
print(f"Prediction for '{new_text[0]}': {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Slide 7: AdaGrad in Computer Vision

AdaGrad can be beneficial in computer vision tasks, especially when dealing with large-scale image datasets. Let's implement a simple convolutional neural network with AdaGrad optimization for image classification using the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class SimpleConvNet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Initialize weights
        self.W1 = np.random.randn(3, 3, 1, 16) / np.sqrt(3*3)
        self.b1 = np.zeros((16, 1))
        self.W2 = np.random.randn(16 * 6 * 6, num_classes) / np.sqrt(16 * 6 * 6)
        self.b2 = np.zeros((num_classes, 1))
        
        # AdaGrad states
        self.W1_state = np.zeros_like(self.W1)
        self.b1_state = np.zeros_like(self.b1)
        self.W2_state = np.zeros_like(self.W2)
        self.b2_state = np.zeros_like(self.b2)
    
    def forward(self, X):
        # Simplified forward pass
        self.X = X.reshape(-1, 8, 8, 1)
        self.conv1 = self.convolution(self.X, self.W1, self.b1)
        self.relu1 = np.maximum(0, self.conv1)
        self.flat = self.relu1.reshape(X.shape[0], -1)
        self.output = np.dot(self.flat, self.W2) + self.b2.T
        return self.softmax(self.output)
    
    def convolution(self, X, W, b):
        # Simplified convolution operation
        return np.sum(X[..., np.newaxis] * W, axis=(1, 2, 3)) + b.T
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, X, y):
        # Simplified backward pass
        m = X.shape[0]
        y_one_hot = np.eye(self.num_classes)[y]
        
        dout = self.output - y_one_hot
        dW2 = np.dot(self.flat.T, dout) / m
        db2 = np.sum(dout, axis=0, keepdims=True).T / m
        
        dflat = np.dot(dout, self.W2.T)
        dconv1 = dflat.reshape(self.relu1.shape) * (self.relu1 > 0)
        dW1 = np.sum(self.X[..., np.newaxis] * dconv1[:, np.newaxis, np.newaxis, :], axis=0) / m
        db1 = np.sum(dconv1, axis=(0, 1, 2), keepdims=True).T / m
        
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, learning_rate=0.01, epsilon=1e-8):
        # AdaGrad update
        self.W1_state += dW1**2
        self.W1 -= learning_rate * dW1 / (np.sqrt(self.W1_state) + epsilon)
        
        self.b1_state += db1**2
        self.b1 -= learning_rate * db1 / (np.sqrt(self.b1_state) + epsilon)
        
        self.W2_state += dW2**2
        self.W2 -= learning_rate * dW2 / (np.sqrt(self.W2_state) + epsilon)
        
        self.b2_state += db2**2
        self.b2 -= learning_rate * db2 / (np.sqrt(self.b2_state) + epsilon)
    
    def train(self, X, y, epochs=100, batch_size=32, learning_rate=0.01):
        for _ in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                self.forward(X_batch)
                dW1, db1, dW2, db2 = self.backward(X_batch, y_batch)
                self.update_params(dW1, db1, dW2, db2, learning_rate)
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Train and evaluate
model = SimpleConvNet((8, 8, 1), 10)
model.train(X_train, y_train)

train_accuracy = np.mean(model.predict(X_train) == y_train)
test_accuracy = np.mean(model.predict(X_test) == y_test)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")
```

Slide 8: AdaGrad vs. Other Optimization Algorithms

AdaGrad is one of several adaptive learning rate methods. Let's compare its performance with standard Stochastic Gradient Descent (SGD) and Adam optimizer on a simple regression task.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=1000):
    X = np.random.randn(n_samples, 1)
    y = 2 * X + 1 + np.random.randn(n_samples, 1) * 0.1
    return X, y

def sgd(X, y, learning_rate=0.01, epochs=100):
    w, b = np.random.randn(1), np.random.randn(1)
    losses = []
    for _ in range(epochs):
        y_pred = X * w + b
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)
        
        dw = np.mean(2 * X * (y_pred - y))
        db = np.mean(2 * (y_pred - y))
        
        w -= learning_rate * dw
        b -= learning_rate * db
    return losses

def adagrad(X, y, learning_rate=0.1, epochs=100, epsilon=1e-8):
    w, b = np.random.randn(1), np.random.randn(1)
    w_state, b_state = 0, 0
    losses = []
    for _ in range(epochs):
        y_pred = X * w + b
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)
        
        dw = np.mean(2 * X * (y_pred - y))
        db = np.mean(2 * (y_pred - y))
        
        w_state += dw**2
        b_state += db**2
        
        w -= learning_rate * dw / (np.sqrt(w_state) + epsilon)
        b -= learning_rate * db / (np.sqrt(b_state) + epsilon)
    return losses

def adam(X, y, learning_rate=0.001, epochs=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    w, b = np.random.randn(1), np.random.randn(1)
    m_w, v_w, m_b, v_b = 0, 0, 0, 0
    losses = []
    for t in range(1, epochs + 1):
        y_pred = X * w + b
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)
        
        dw = np.mean(2 * X * (y_pred - y))
        db = np.mean(2 * (y_pred - y))
        
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * dw**2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db**2
        
        m_w_hat = m_w / (1 - beta1**t)
        v_w_hat = v_w / (1 - beta2**t)
        m_b_hat = m_b / (1 - beta1**t)
        v_b_hat = v_b / (1 - beta2**t)
        
        w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
        b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    return losses

X, y = generate_data()

sgd_losses = sgd(X, y)
adagrad_losses = adagrad(X, y)
adam_losses = adam(X, y)

plt.plot(sgd_losses, label='SGD')
plt.plot(adagrad_losses, label='AdaGrad')
plt.plot(adam_losses, label='Adam')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Optimization Algorithms Comparison')
plt.legend()
plt.show()
```

Slide 9: AdaGrad's Limitations

While AdaGrad has several advantages, it also has some limitations:

1. Accumulation of squared gradients: The denominator grows continuously, which can cause the learning rate to shrink and eventually become infinitesimally small.
2. Uniform learning rate decay: AdaGrad applies the same learning rate decay to all parameters, which may not be optimal for all problems.
3. Sensitivity to initial learning rate: The algorithm's performance can be sensitive to the choice of the initial learning rate.

```python
import numpy as np
import matplotlib.pyplot as plt

def adagrad_learning_rate_decay(iterations=1000, initial_lr=0.1, epsilon=1e-8):
    gradients = np.random.randn(iterations)
    learning_rates = np.zeros(iterations)
    accumulated_grad = 0
    
    for i in range(iterations):
        accumulated_grad += gradients[i]**2
        learning_rates[i] = initial_lr / (np.sqrt(accumulated_grad) + epsilon)
    
    plt.plot(learning_rates)
    plt.title('AdaGrad Learning Rate Decay')
    plt.xlabel('Iterations')
    plt.ylabel('Effective Learning Rate')
    plt.yscale('log')
    plt.show()

adagrad_learning_rate_decay()
```

Slide 10: AdaGrad in Practice: Image Classification

Let's apply AdaGrad to a real-world image classification task using a simple convolutional neural network on the CIFAR-10 dataset.

```python
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class SimpleCNN:
    def __init__(self):
        self.W1 = np.random.randn(3, 3, 3, 32) * 0.1
        self.b1 = np.zeros((32, 1))
        self.W2 = np.random.randn(3, 3, 32, 64) * 0.1
        self.b2 = np.zeros((64, 1))
        self.W3 = np.random.randn(64 * 8 * 8, 10) * 0.1
        self.b3 = np.zeros((10, 1))
        
        # AdaGrad states
        self.W1_state = np.zeros_like(self.W1)
        self.b1_state = np.zeros_like(self.b1)
        self.W2_state = np.zeros_like(self.W2)
        self.b2_state = np.zeros_like(self.b2)
        self.W3_state = np.zeros_like(self.W3)
        self.b3_state = np.zeros_like(self.b3)
    
    def forward(self, X):
        # Simplified forward pass
        self.conv1 = self.convolution(X, self.W1, self.b1)
        self.relu1 = np.maximum(0, self.conv1)
        self.pool1 = self.max_pool(self.relu1)
        
        self.conv2 = self.convolution(self.pool1, self.W2, self.b2)
        self.relu2 = np.maximum(0, self.conv2)
        self.pool2 = self.max_pool(self.relu2)
        
        self.flat = self.pool2.reshape(X.shape[0], -1)
        self.output = np.dot(self.flat, self.W3) + self.b3.T
        return self.softmax(self.output)
    
    def convolution(self, X, W, b):
        # Simplified convolution operation
        return np.sum(X[..., np.newaxis] * W, axis=(1, 2, 3)) + b.T
    
    def max_pool(self, X, pool_size=2):
        # Simplified max pooling operation
        return X[:, ::pool_size, ::pool_size, :]
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, X, y):
        # Simplified backward pass
        m = X.shape[0]
        
        dout = self.output - y
        dW3 = np.dot(self.flat.T, dout) / m
        db3 = np.sum(dout, axis=0, keepdims=True).T / m
        
        dflat = np.dot(dout, self.W3.T)
        dpool2 = dflat.reshape(self.pool2.shape)
        
        # Simplified gradients for conv2, pool1, and conv1
        dW2 = np.random.randn(*self.W2.shape) # Placeholder
        db2 = np.random.randn(*self.b2.shape) # Placeholder
        dW1 = np.random.randn(*self.W1.shape) # Placeholder
        db1 = np.random.randn(*self.b1.shape) # Placeholder
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, learning_rate=0.01, epsilon=1e-8):
        # AdaGrad update
        self.W1_state += dW1**2
        self.W1 -= learning_rate * dW1 / (np.sqrt(self.W1_state) + epsilon)
        self.b1_state += db1**2
        self.b1 -= learning_rate * db1 / (np.sqrt(self.b1_state) + epsilon)
        
        self.W2_state += dW2**2
        self.W2 -= learning_rate * dW2 / (np.sqrt(self.W2_state) + epsilon)
        self.b2_state += db2**2
        self.b2 -= learning_rate * db2 / (np.sqrt(self.b2_state) + epsilon)
        
        self.W3_state += dW3**2
        self.W3 -= learning_rate * dW3 / (np.sqrt(self.W3_state) + epsilon)
        self.b3_state += db3**2
        self.b3 -= learning_rate * db3 / (np.sqrt(self.b3_state) + epsilon)
    
    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01):
        for _ in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                self.forward(X_batch)
                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)
                self.update_params(dW1, db1, dW2, db2, dW3, db3, learning_rate)

# Train and evaluate
model = SimpleCNN()
model.train(X_train[:1000], y_train[:1000], epochs=5)

# Evaluate
y_pred = model.forward(X_test[:100])
accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test[:100], axis=1))
print(f"Test accuracy: {accuracy:.2f}")
```

Slide 11: AdaGrad in Reinforcement Learning

AdaGrad can also be applied to reinforcement learning tasks. Let's implement a simple Q-learning agent with AdaGrad for the CartPole environment.

```python
import numpy as np
import gym

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = np.zeros((state_size, action_size))
        self.adagrad_state = np.zeros_like(self.Q)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        target = reward + (1 - done) * self.gamma * np.max(self.Q[next_state])
        error = target - self.Q[state, action]
        
        self.adagrad_state[state, action] += error**2
        self.Q[state, action] += self.learning_rate * error / (np.sqrt(self.adagrad_state[state, action]) + 1e-8)

# Training loop
env = gym.make('CartPole-v1')
agent = QLearningAgent(env.observation_space.n, env.action_space.n)

n_episodes = 1000
for episode in range(n_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
```

Slide 12: AdaGrad in Natural Language Processing

AdaGrad is particularly useful in NLP tasks due to the sparse nature of text data. Let's implement a simple word embedding model using AdaGrad optimization.

```python
import numpy as np
from collections import defaultdict

class SimpleWordEmbedding:
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.adagrad_state = np.zeros_like(self.embeddings)
    
    def forward(self, word_idx):
        return self.embeddings[word_idx]
    
    def backward(self, word_idx, grad):
        self.adagrad_state[word_idx] += grad**2
        self.embeddings[word_idx] -= self.learning_rate * grad / (np.sqrt(self.adagrad_state[word_idx]) + 1e-8)

# Example usage
vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
vocab_size = len(vocab)
embedding_dim = 5

model = SimpleWordEmbedding(vocab_size, embedding_dim)

# Simulate training
for _ in range(1000):
    for word in vocab:
        word_idx = vocab.index(word)
        context_words = [vocab[i] for i in range(len(vocab)) if i != word_idx]
        
        for context_word in context_words:
            context_idx = vocab.index(context_word)
            
            # Simplified forward pass
            target = model.forward(word_idx)
            context = model.forward(context_idx)
            
            # Simplified backward pass
            grad = target - context
            model.backward(word_idx, grad)
            model.backward(context_idx, -grad)

# Print final embeddings
for word in vocab:
    print(f"{word}: {model.forward(vocab.index(word))}")
```

Slide 13: AdaGrad: Pros and Cons

Pros:

1. Adapts learning rates for each parameter
2. Performs well on sparse data
3. Eliminates the need for manual learning rate tuning

Cons:

1. Accumulation of squared gradients can lead to premature stopping
2. May not perform well in non-convex settings
3. Uniform learning rate decay might not be optimal for all problems

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_adagrad_behavior():
    iterations = 1000
    params = np.random.randn(2)
    adagrad_state = np.zeros_like(params)
    learning_rate = 0.1
    epsilon = 1e-8
    
    param_history = []
    
    for _ in range(iterations):
        gradient = np.random.randn(2)  # Simulated gradient
        adagrad_state += gradient**2
        params -= learning_rate * gradient / (np.sqrt(adagrad_state) + epsilon)
        param_history.append(params.())
    
    param_history = np.array(param_history)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(param_history[:, 0], label='Parameter 1')
    plt.plot(param_history[:, 1], label='Parameter 2')
    plt.title('Parameter Updates')
    plt.xlabel('Iterations')
    plt.ylabel('Parameter Value')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rate / (np.sqrt(np.cumsum(np.random.randn(iterations)**2)) + epsilon))
    plt.title('Effective Learning Rate')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.show()

plot_adagrad_behavior()
```

Slide 14: Additional Resources

For those interested in diving deeper into AdaGrad and adaptive optimization methods, here are some valuable resources:

1. Original AdaGrad paper: "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by Duchi et al. (2011) ArXiv link: [https://arxiv.org/abs/1101.0390](https://arxiv.org/abs/1101.0390)
2. "An overview of gradient descent optimization algorithms" by Sebastian Ruder (2016) ArXiv link: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Adaptive Methods for Nonconvex Optimization" by Ward et al. (2019) ArXiv link: [https://arxiv.org/abs/1905.08175](https://arxiv.org/abs/1905.08175)
4. "Advances in Optimizing Recurrent Networks" by Yoshua Bengio et al. (2012) ArXiv link: [https://arxiv.org/abs/1212.0901](https://arxiv.org/abs/1212.0901)

These papers provide in-depth analysis and comparisons of AdaGrad with other optimization algorithms, as well as its applications in various machine learning tasks.

