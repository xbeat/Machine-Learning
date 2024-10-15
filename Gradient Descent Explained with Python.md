## Gradient Descent Explained with Python
Slide 1: Introduction to Gradient Descent

Gradient Descent is a fundamental optimization algorithm used in machine learning and deep learning. It's designed to find the minimum of a function by iteratively moving in the direction of steepest descent. This algorithm is particularly useful for training models with large datasets.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 5, 100)
y = f(x)

plt.plot(x, y)
plt.title('f(x) = x^2 + 5x + 10')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 2: The Concept of Gradient

The gradient is a vector of partial derivatives that points in the direction of the steepest increase of a function. In gradient descent, we move in the opposite direction of the gradient to find the minimum.

```python
def gradient(x):
    return 2*x + 5  # Derivative of f(x) = x^2 + 5x + 10

x = np.linspace(-10, 5, 100)
grad = gradient(x)

plt.plot(x, grad)
plt.title('Gradient of f(x)')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.grid(True)
plt.show()
```

Slide 3: Basic Gradient Descent Algorithm

The basic gradient descent algorithm iteratively updates the parameters by subtracting the gradient multiplied by a learning rate. This process continues until convergence or a maximum number of iterations is reached.

```python
def gradient_descent(start, learn_rate, num_iterations):
    x = start
    for i in range(num_iterations):
        grad = gradient(x)
        x = x - learn_rate * grad
    return x

minimum = gradient_descent(start=5, learn_rate=0.1, num_iterations=100)
print(f"The minimum occurs at x = {minimum:.2f}")
```

Slide 4: Learning Rate

The learning rate determines the size of the steps taken during gradient descent. A large learning rate may overshoot the minimum, while a small one may result in slow convergence.

```python
def plot_gd_path(start, learn_rate, num_iterations):
    x = start
    path = [x]
    for _ in range(num_iterations):
        grad = gradient(x)
        x = x - learn_rate * grad
        path.append(x)
    return path

x = np.linspace(-10, 5, 100)
y = f(x)

plt.figure(figsize=(12, 4))
for lr in [0.01, 0.1, 0.5]:
    path = plot_gd_path(start=5, learn_rate=lr, num_iterations=20)
    plt.plot(path, f(np.array(path)), label=f'LR = {lr}')

plt.plot(x, y, 'r--')
plt.legend()
plt.title('Gradient Descent Paths with Different Learning Rates')
plt.show()
```

Slide 5: Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) uses a single random data point to compute the gradient at each iteration. This can lead to faster convergence and help escape local minima.

```python
def stochastic_gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            prediction = np.dot(xi, theta)
            theta = theta - learning_rate * (1/m) * xi.T.dot((prediction - yi))
    
    return theta

# Example usage:
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
theta = stochastic_gradient_descent(X, y, learning_rate=0.01, num_iterations=1000)
print("Optimized parameters:", theta)
```

Slide 6: Mini-Batch Gradient Descent

Mini-Batch Gradient Descent combines aspects of both batch and stochastic gradient descent. It uses a small random subset of the data to compute the gradient at each iteration.

```python
def mini_batch_gradient_descent(X, y, batch_size, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        indices = np.random.randint(m, size=batch_size)
        X_batch = X[indices]
        y_batch = y[indices]
        prediction = np.dot(X_batch, theta)
        theta = theta - learning_rate * (1/batch_size) * X_batch.T.dot((prediction - y_batch))
    
    return theta

# Example usage:
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([1, 2, 3, 4, 5])
theta = mini_batch_gradient_descent(X, y, batch_size=2, learning_rate=0.01, num_iterations=1000)
print("Optimized parameters:", theta)
```

Slide 7: Momentum

Momentum is a technique used to accelerate gradient descent by adding a fraction of the previous update to the current one. This helps to dampen oscillations and converge faster.

```python
def gradient_descent_with_momentum(start, learn_rate, momentum, num_iterations):
    x = start
    velocity = 0
    path = [x]
    
    for _ in range(num_iterations):
        grad = gradient(x)
        velocity = momentum * velocity - learn_rate * grad
        x = x + velocity
        path.append(x)
    
    return path

x = np.linspace(-10, 5, 100)
y = f(x)

plt.figure(figsize=(12, 4))
path_no_momentum = plot_gd_path(start=5, learn_rate=0.1, num_iterations=20)
path_with_momentum = gradient_descent_with_momentum(start=5, learn_rate=0.1, momentum=0.9, num_iterations=20)

plt.plot(x, y, 'r--')
plt.plot(path_no_momentum, f(np.array(path_no_momentum)), label='No Momentum')
plt.plot(path_with_momentum, f(np.array(path_with_momentum)), label='With Momentum')
plt.legend()
plt.title('Gradient Descent With and Without Momentum')
plt.show()
```

Slide 8: Adaptive Learning Rates

Adaptive learning rate methods automatically adjust the learning rate during training. Popular algorithms include AdaGrad, RMSprop, and Adam.

```python
def adam_optimizer(start, learn_rate, beta1, beta2, epsilon, num_iterations):
    x = start
    m = 0
    v = 0
    path = [x]
    
    for t in range(1, num_iterations + 1):
        grad = gradient(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - learn_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x)
    
    return path

x = np.linspace(-10, 5, 100)
y = f(x)

plt.figure(figsize=(12, 4))
path_gd = plot_gd_path(start=5, learn_rate=0.1, num_iterations=50)
path_adam = adam_optimizer(start=5, learn_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=50)

plt.plot(x, y, 'r--')
plt.plot(path_gd, f(np.array(path_gd)), label='Standard GD')
plt.plot(path_adam, f(np.array(path_adam)), label='Adam')
plt.legend()
plt.title('Comparison of Standard Gradient Descent and Adam')
plt.show()
```

Slide 9: Gradient Descent for Linear Regression

Gradient descent is commonly used to optimize the parameters of a linear regression model. Let's implement this from scratch.

```python
def linear_regression_gd(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        prediction = np.dot(X, theta)
        error = prediction - y
        gradient = (1/m) * X.T.dot(error)
        theta = theta - learning_rate * gradient
    
    return theta

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Run gradient descent
theta = linear_regression_gd(X_b, y, learning_rate=0.01, num_iterations=1000)

# Plot results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='r')
plt.title('Linear Regression with Gradient Descent')
plt.show()

print("Optimized parameters:", theta.flatten())
```

Slide 10: Gradient Descent for Logistic Regression

Gradient descent can also be used to optimize logistic regression models for binary classification problems.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - learning_rate * gradient
    
    return theta

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Run gradient descent
theta = logistic_regression_gd(X_b, y, learning_rate=0.1, num_iterations=1000)

# Plot decision boundary
x1 = np.linspace(-3, 3, 100)
x2 = -(theta[0] + theta[1] * x1) / theta[2]

plt.scatter(X[y==0, 0], X[y==0, 1], c='b', label='Class 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='r', label='Class 1')
plt.plot(x1, x2, 'g-', label='Decision Boundary')
plt.legend()
plt.title('Logistic Regression with Gradient Descent')
plt.show()

print("Optimized parameters:", theta)
```

Slide 11: Gradient Descent in Neural Networks

In neural networks, gradient descent is used to update the weights and biases during backpropagation. Here's a simple example of a neural network trained with gradient descent.

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def neural_network(X, y, hidden_size, learning_rate, num_iterations):
    input_size, output_size = X.shape[1], 1
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    
    for _ in range(num_iterations):
        # Forward pass
        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = sigmoid(z2)
        
        # Backward pass
        dz2 = y_pred - y
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
y = (np.sum(X**2, axis=1) > 2).astype(int).reshape(-1, 1)

# Train neural network
W1, b1, W2, b2 = neural_network(X, y, hidden_size=5, learning_rate=0.1, num_iterations=10000)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
X_grid = np.c_[xx.ravel(), yy.ravel()]
z1 = relu(np.dot(X_grid, W1) + b1)
z2 = sigmoid(np.dot(z1, W2) + b2)
Z = z2.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], c='b', label='Class 0')
plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], c='r', label='Class 1')
plt.legend()
plt.title('Neural Network Decision Boundary')
plt.show()
```

Slide 12: Real-Life Example: Image Classification

Gradient descent is crucial in training convolutional neural networks (CNNs) for image classification tasks. Here's a simplified example using a subset of the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load a subset of MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values
X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, solver='sgd', learning_rate_init=0.1, random_state=42)
model.partial_fit(X_train, y_train, classes=np.unique(y))

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Make a prediction
sample_image = X_test[0].reshape(1, -1)
prediction = model.predict(sample_image)
print(f"Predicted digit: {prediction[0]}")
```

Slide 13: Real-Life Example: Natural Language Processing

Gradient descent is also used in training models for natural language processing tasks, such as sentiment analysis. Here's a simple example using a basic neural network.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Sample dataset
texts = [
    "I love this product", "Great service", "Terrible experience",
    "Not worth the money", "Highly recommended", "Waste of time"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 for positive, 0 for negative

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, solver='sgd', learning_rate_init=0.01, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Make a prediction
new_text = ["This product exceeded my expectations"]
new_vector = vectorizer.transform(new_text).toarray()
prediction = model.predict(new_vector)
print(f"Sentiment: {'Positive' if prediction[0] == 1 else 'Negative'}")
```

Slide 14: Challenges and Limitations of Gradient Descent

Gradient descent, while powerful, faces several challenges:

1. Local Minima: The algorithm may get stuck in local minima, especially in non-convex optimization problems.
2. Saddle Points: In high-dimensional spaces, saddle points can slow down convergence.
3. Choosing Learning Rate: Selecting an appropriate learning rate is crucial and often requires careful tuning.
4. Scaling: Different features may have different scales, affecting the optimization process.

To address these issues, various techniques have been developed, such as momentum, adaptive learning rates, and second-order optimization methods.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2 + np.sin(5*x) + np.sin(5*y)

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20)
plt.colorbar(label='f(x, y)')
plt.title('Contour Plot of a Function with Multiple Local Minima')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into gradient descent and optimization algorithms, here are some valuable resources:

1. "Optimization Methods for Large-Scale Machine Learning" by Bottou et al. (2018) ArXiv: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "An Overview of Gradient Descent Optimization Algorithms" by Ruder (2016) ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by Duchi et al. (2011) ArXiv: [https://arxiv.org/abs/1101.3618](https://arxiv.org/abs/1101.3618)

These papers provide in-depth discussions on various aspects of gradient descent and its applications in machine learning.

