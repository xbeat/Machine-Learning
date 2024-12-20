## Introduction to Gradient Descent in Python
Slide 1: Introduction to Gradient Descent

Gradient descent is a fundamental optimization algorithm in machine learning and deep learning. It's used to minimize a cost function by iteratively moving in the direction of steepest descent. This method is crucial for training various models, including neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title("Cost Function")
plt.xlabel("x")
plt.ylabel("Cost")
plt.show()
```

Slide 2: The Gradient

The gradient is a vector of partial derivatives that points in the direction of the steepest increase of a function. In gradient descent, we move in the opposite direction to minimize the function.

```python
def gradient(x):
    return 2 * x

x = np.linspace(-10, 10, 100)
grad = gradient(x)

plt.plot(x, grad)
plt.title("Gradient of Cost Function")
plt.xlabel("x")
plt.ylabel("Gradient")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 3: The Learning Rate

The learning rate determines the size of the steps we take during gradient descent. It's a crucial hyperparameter that affects the convergence of the algorithm.

```python
def gradient_descent_step(x, learning_rate):
    return x - learning_rate * gradient(x)

x = 5
learning_rates = [0.1, 0.01, 0.001]

for lr in learning_rates:
    x_new = gradient_descent_step(x, lr)
    print(f"Learning rate: {lr}, New x: {x_new}")
```

Slide 4: Iterative Process

Gradient descent is an iterative process. We repeatedly calculate the gradient and update our parameters until we reach a minimum or a specified number of iterations.

```python
def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    x_history = [x]
    
    for _ in range(num_iterations):
        x = gradient_descent_step(x, learning_rate)
        x_history.append(x)
    
    return x, x_history

final_x, x_history = gradient_descent(5, 0.1, 20)
print(f"Final x: {final_x}")

plt.plot(range(len(x_history)), x_history)
plt.title("Convergence of Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.show()
```

Slide 5: Mathematical Formulation

The gradient descent update rule can be expressed mathematically (LaTex) as:

$x\_{n+1} = x\_n - \\alpha \\nabla f(x\_n)$

Where:

* $x\_n$ is the current point
* $\\alpha$ is the learning rate
* $\\nabla f(x\_n)$ is the gradient of the function at $x\_n$

```python
# Visualizing the mathematical formulation
x = np.linspace(-10, 10, 100)
y = cost_function(x)
grad = gradient(x)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title("Cost Function")
plt.xlabel("x")
plt.ylabel("Cost")

plt.subplot(1, 2, 2)
plt.plot(x, grad)
plt.title("Gradient")
plt.xlabel("x")
plt.ylabel("Gradient")
plt.tight_layout()
plt.show()
```

Slide 6: Batch Gradient Descent

Batch gradient descent uses the entire dataset to compute the gradient in each iteration. It's computationally expensive for large datasets but guaranteed to converge to the global minimum for convex problems.

```python
def batch_gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        h = X.dot(theta)
        gradient = (1/m) * X.T.dot(h - y)
        theta -= learning_rate * gradient
    
    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
theta = batch_gradient_descent(X, y, 0.01, 1000)
print("Optimized theta:", theta)
```

Slide 7: Stochastic Gradient Descent (SGD)

SGD updates the parameters using only one training example at a time. It's faster and requires less memory, but the path to the minimum is noisier.

```python
def stochastic_gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradient
    
    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
theta = stochastic_gradient_descent(X, y, 0.01, 1000)
print("Optimized theta:", theta)
```

Slide 8: Mini-Batch Gradient Descent

Mini-batch gradient descent is a compromise between batch and stochastic gradient descent. It updates parameters using a small random subset of the training data.

```python
def mini_batch_gradient_descent(X, y, learning_rate, num_iterations, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            Xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradient = Xi.T.dot(Xi.dot(theta) - yi) / batch_size
            theta -= learning_rate * gradient
    
    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([1, 2, 3, 4, 5])
theta = mini_batch_gradient_descent(X, y, 0.01, 1000, 2)
print("Optimized theta:", theta)
```

Slide 9: Gradient Descent in Multiple Dimensions

In practice, we often deal with multidimensional problems. The concept remains the same, but we update multiple parameters simultaneously.

```python
def multidim_cost_function(x, y):
    return x**2 + y**2

def multidim_gradient(x, y):
    return np.array([2*x, 2*y])

def multidim_gradient_descent(start_x, start_y, learning_rate, num_iterations):
    point = np.array([start_x, start_y])
    path = [point]
    
    for _ in range(num_iterations):
        grad = multidim_gradient(point[0], point[1])
        point = point - learning_rate * grad
        path.append(point)
    
    return np.array(path)

path = multidim_gradient_descent(5, 5, 0.1, 50)

x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = multidim_cost_function(X, Y)

plt.contour(X, Y, Z, levels=50)
plt.colorbar(label='Cost')
plt.plot(path[:, 0], path[:, 1], 'ro-')
plt.title("Gradient Descent in 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 10: Momentum

Momentum is a method that helps accelerate gradient descent in the relevant direction and dampens oscillations. It does this by adding a fraction of the update vector of the past time step to the current update vector.

```python
def gradient_descent_with_momentum(start_x, learning_rate, momentum, num_iterations):
    x = start_x
    v = 0
    x_history = [x]
    
    for _ in range(num_iterations):
        grad = gradient(x)
        v = momentum * v - learning_rate * grad
        x = x + v
        x_history.append(x)
    
    return x, x_history

final_x, x_history = gradient_descent_with_momentum(5, 0.1, 0.9, 20)
print(f"Final x: {final_x}")

plt.plot(range(len(x_history)), x_history)
plt.title("Gradient Descent with Momentum")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.show()
```

Slide 11: Adaptive Learning Rates

Adaptive learning rate methods adjust the learning rate for each parameter. One popular method is AdaGrad (Adaptive Gradient Algorithm).

```python
def adagrad(start_x, learning_rate, num_iterations, epsilon=1e-8):
    x = start_x
    sum_grad_squared = 0
    x_history = [x]
    
    for _ in range(num_iterations):
        grad = gradient(x)
        sum_grad_squared += grad**2
        x = x - (learning_rate / (np.sqrt(sum_grad_squared) + epsilon)) * grad
        x_history.append(x)
    
    return x, x_history

final_x, x_history = adagrad(5, 1.0, 20)
print(f"Final x: {final_x}")

plt.plot(range(len(x_history)), x_history)
plt.title("AdaGrad")
plt.xlabel("Iteration")
plt.ylabel("x")
plt.show()
```

Slide 12: Real-Life Example: Linear Regression

Gradient descent is commonly used in linear regression to find the best-fitting line for a set of data points.

```python
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]
theta = np.random.randn(2, 1)

learning_rate = 0.1
n_iterations = 1000

for iteration in range(n_iterations):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("Theta found by gradient descent:", theta)

plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='r')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 13: Real-Life Example: Image Classification

Gradient descent is crucial in training neural networks for tasks like image classification. Here's a simplified example using a small neural network.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def neural_network(X, W1, W2):
    z1 = X.dot(W1)
    a1 = sigmoid(z1)
    z2 = a1.dot(W2)
    return sigmoid(z2)

def loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate dummy data
np.random.seed(0)
X = np.random.randn(100, 10)
y = np.random.randint(2, size=(100, 1))

# Initialize weights
W1 = np.random.randn(10, 5)
W2 = np.random.randn(5, 1)

learning_rate = 0.1
n_iterations = 1000

losses = []

for _ in range(n_iterations):
    # Forward pass
    y_pred = neural_network(X, W1, W2)
    
    # Compute loss
    current_loss = loss(y, y_pred)
    losses.append(current_loss)
    
    # Backward pass (simplified)
    d_W2 = X.T.dot(y_pred - y) / len(y)
    d_W1 = X.T.dot((y_pred - y).dot(W2.T) * y_pred * (1 - y_pred)) / len(y)
    
    # Update weights
    W1 -= learning_rate * d_W1
    W2 -= learning_rate * d_W2

plt.plot(losses)
plt.title("Loss over iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
```

Slide 14: Additional Resources

For a deeper understanding of gradient descent and its applications, consider exploring these peer-reviewed articles from arXiv:

1. "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder arXiv:1609.04747 \[cs.LG\] [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "Gradient Descent Revisited via an Adaptive Online Learning Rate" by Yann N. Dauphin et al. arXiv:1403.5782 \[cs.LG\] [https://arxiv.org/abs/1403.5782](https://arxiv.org/abs/1403.5782)
3. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by John Duchi et al. arXiv:1403.5782 \[cs.LG\] [https://arxiv.org/abs/1011.1768](https://arxiv.org/abs/1011.1768)

These resources provide in-depth analysis and advanced techniques related to gradient descent in machine learning and optimization.

