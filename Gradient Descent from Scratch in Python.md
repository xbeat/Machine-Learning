## Gradient Descent from Scratch in Python
Slide 1: Introduction to Gradient Descent

Gradient descent is a fundamental optimization algorithm used in machine learning to minimize the cost function of a model. It iteratively adjusts the model's parameters in the direction of steepest descent of the cost function.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title('Cost Function')
plt.xlabel('x')
plt.ylabel('Cost')
plt.show()
```

Slide 2: The Gradient

The gradient is a vector of partial derivatives that points in the direction of steepest ascent. In gradient descent, we move in the opposite direction to minimize the cost function.

```python
def gradient(x):
    return 2*x + 5

x = np.linspace(-10, 10, 100)
grad = gradient(x)

plt.plot(x, grad)
plt.title('Gradient of Cost Function')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 3: Basic Gradient Descent Algorithm

The algorithm updates the parameters iteratively by subtracting the product of the learning rate and the gradient from the current parameter values.

```python
def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    for i in range(num_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x:.4f}, cost = {cost_function(x):.4f}")
    return x

optimal_x = gradient_descent(start_x=5, learning_rate=0.1, num_iterations=20)
print(f"Optimal x: {optimal_x:.4f}")
```

Slide 4: Learning Rate

The learning rate determines the step size at each iteration. A too large learning rate may overshoot the minimum, while a too small one may result in slow convergence.

```python
learning_rates = [0.01, 0.1, 0.5]
start_x = 5
iterations = 50

for lr in learning_rates:
    x = start_x
    xs = [x]
    for _ in range(iterations):
        x = x - lr * gradient(x)
        xs.append(x)
    
    plt.plot(range(iterations+1), xs, label=f'LR = {lr}')

plt.legend()
plt.title('Effect of Learning Rate')
plt.xlabel('Iterations')
plt.ylabel('x')
plt.show()
```

Slide 5: Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) computes the gradient using a single random sample from the dataset, making it faster and able to escape local minima more easily.

```python
import random

def stochastic_gradient_descent(data, labels, learning_rate, num_iterations):
    w, b = 0, 0
    for _ in range(num_iterations):
        idx = random.randint(0, len(data)-1)
        x, y = data[idx], labels[idx]
        y_pred = w * x + b
        error = y_pred - y
        w -= learning_rate * error * x
        b -= learning_rate * error
    return w, b

# Example usage
data = [1, 2, 3, 4, 5]
labels = [2, 4, 6, 8, 10]
w, b = stochastic_gradient_descent(data, labels, 0.01, 1000)
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")
```

Slide 6: Mini-batch Gradient Descent

Mini-batch gradient descent combines the advantages of both batch and stochastic gradient descent by using a small random subset of the data for each update.

```python
def mini_batch_gradient_descent(data, labels, batch_size, learning_rate, num_iterations):
    w, b = 0, 0
    for _ in range(num_iterations):
        batch_indices = np.random.choice(len(data), batch_size, replace=False)
        x_batch = [data[i] for i in batch_indices]
        y_batch = [labels[i] for i in batch_indices]
        
        grad_w, grad_b = 0, 0
        for x, y in zip(x_batch, y_batch):
            y_pred = w * x + b
            error = y_pred - y
            grad_w += error * x
            grad_b += error
        
        w -= learning_rate * grad_w / batch_size
        b -= learning_rate * grad_b / batch_size
    
    return w, b

# Example usage
data = [1, 2, 3, 4, 5]
labels = [2, 4, 6, 8, 10]
w, b = mini_batch_gradient_descent(data, labels, batch_size=2, learning_rate=0.01, num_iterations=1000)
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")
```

Slide 7: Momentum

Momentum helps accelerate gradient descent in the relevant direction and dampens oscillations. It does this by adding a fraction of the previous update to the current one.

```python
def momentum_gradient_descent(start_x, learning_rate, momentum, num_iterations):
    x = start_x
    velocity = 0
    for i in range(num_iterations):
        grad = gradient(x)
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
        print(f"Iteration {i+1}: x = {x:.4f}, cost = {cost_function(x):.4f}")
    return x

optimal_x = momentum_gradient_descent(start_x=5, learning_rate=0.1, momentum=0.9, num_iterations=20)
print(f"Optimal x: {optimal_x:.4f}")
```

Slide 8: Adaptive Learning Rates

Adaptive learning rate methods adjust the learning rate for each parameter. One popular method is AdaGrad, which adapts the learning rate to the parameters, performing smaller updates for frequently occurring features.

```python
def adagrad(start_x, learning_rate, num_iterations):
    x = start_x
    sum_squared_gradients = 0
    epsilon = 1e-8  # Small value to avoid division by zero
    
    for i in range(num_iterations):
        grad = gradient(x)
        sum_squared_gradients += grad**2
        adjusted_learning_rate = learning_rate / (np.sqrt(sum_squared_gradients) + epsilon)
        x = x - adjusted_learning_rate * grad
        print(f"Iteration {i+1}: x = {x:.4f}, cost = {cost_function(x):.4f}")
    
    return x

optimal_x = adagrad(start_x=5, learning_rate=1, num_iterations=20)
print(f"Optimal x: {optimal_x:.4f}")
```

Slide 9: Gradient Descent for Multivariable Functions

In practice, we often deal with functions of multiple variables. Gradient descent can be extended to work with these functions by computing partial derivatives for each variable.

```python
def multivariable_cost(x, y):
    return x**2 + y**2

def multivariable_gradient(x, y):
    return np.array([2*x, 2*y])

def multivariable_gradient_descent(start_x, start_y, learning_rate, num_iterations):
    point = np.array([start_x, start_y])
    
    for i in range(num_iterations):
        grad = multivariable_gradient(point[0], point[1])
        point = point - learning_rate * grad
        cost = multivariable_cost(point[0], point[1])
        print(f"Iteration {i+1}: x = {point[0]:.4f}, y = {point[1]:.4f}, cost = {cost:.4f}")
    
    return point

optimal_point = multivariable_gradient_descent(start_x=5, start_y=5, learning_rate=0.1, num_iterations=20)
print(f"Optimal point: x = {optimal_point[0]:.4f}, y = {optimal_point[1]:.4f}")
```

Slide 10: Visualizing Gradient Descent

Visualizing the path of gradient descent can help understand how the algorithm converges to the minimum. Let's create a contour plot and show the optimization path.

```python
def plot_gradient_descent(start_x, start_y, learning_rate, num_iterations):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = multivariable_cost(X, Y)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50)
    
    point = np.array([start_x, start_y])
    path = [point]
    
    for _ in range(num_iterations):
        grad = multivariable_gradient(point[0], point[1])
        point = point - learning_rate * grad
        path.append(point)
    
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'ro-')
    plt.title('Gradient Descent Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_gradient_descent(start_x=8, start_y=8, learning_rate=0.1, num_iterations=50)
```

Slide 11: Real-life Example: Linear Regression

Gradient descent is commonly used in linear regression to find the best-fitting line for a set of data points.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient descent for linear regression
def linear_regression_gradient_descent(X, y, learning_rate, num_iterations):
    m = len(y)
    theta = np.random.randn(2, 1)
    
    for _ in range(num_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
    
    return theta

X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
theta = linear_regression_gradient_descent(X_b, y, learning_rate=0.01, num_iterations=1000)

# Plot the results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='r')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Estimated parameters: intercept = {theta[0][0]:.4f}, slope = {theta[1][0]:.4f}")
```

Slide 12: Real-life Example: Image Classification

Gradient descent is crucial in training neural networks for image classification tasks. Let's use a simple example with the MNIST dataset.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) / 255.0
X_test = X_test.reshape(10000, 784) / 255.0

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model with SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 13: Challenges and Considerations

Gradient descent, while powerful, faces challenges such as getting stuck in local minima, slow convergence for ill-conditioned problems, and the need for careful hyperparameter tuning. Advanced variants like Adam and RMSprop address some of these issues.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_function(x):
    return np.sin(x) + 0.1 * x**2

x = np.linspace(-10, 10, 1000)
y = complex_function(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.title('Complex Function with Multiple Local Minima')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into gradient descent and optimization algorithms, here are some recommended resources:

1. "Optimization for Machine Learning" by Suvrit Sra, Sebastian Nowozin, and Stephen J. Wright (MIT Press)
2. "Gradient Descent Revisited: A New Perspective Based on Path-Following" by Bin Shi et al. (arXiv:2008.11266)
3. "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder (arXiv:1609.04747)

These papers can be found on ArXiv.org by searching for their respective arXiv IDs.

