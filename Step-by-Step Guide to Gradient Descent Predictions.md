## Step-by-Step Guide to Gradient Descent Predictions
Slide 1: Introduction to Gradient Descent

Gradient Descent is a fundamental optimization algorithm in machine learning used to minimize a cost function and improve model performance. It iteratively adjusts model parameters to find the optimal solution. This process transforms initial random points in parameter space into powerful predictions.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title('Cost Function')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.show()
```

Slide 2: The Starting Point

Training begins by initializing parameters (weights and biases) randomly. These parameters represent a point in high-dimensional space, corresponding to a specific model configuration and error value.

```python
import numpy as np

# Initialize random parameters
np.random.seed(42)
initial_params = np.random.randn(5)

print("Initial parameters:", initial_params)
```

Slide 3: Objective - Finding the Minimum

The goal of gradient descent is to find the point where the model error (cost function) is minimized, known as the global minimum. This is achieved by iteratively moving towards lower error regions.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title('Cost Function with Global Minimum')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.plot(-2.5, cost_function(-2.5), 'ro', label='Global Minimum')
plt.legend()
plt.show()
```

Slide 4: Step 1: Computing the Gradient

At each point, we calculate the gradient, which represents the direction of steepest ascent. Since we aim to minimize error, we move in the opposite direction of the gradient.

```python
def gradient(x):
    return 2*x + 5

x = 2
grad = gradient(x)
print(f"Gradient at x = {x}: {grad}")

# Visualize gradient
x = np.linspace(-10, 10, 100)
y = cost_function(x)
plt.plot(x, y)
plt.quiver(2, cost_function(2), -1, -gradient(2), scale=20, color='r')
plt.title('Gradient at a Point')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.show()
```

Slide 5: Step 2: Updating the Point

We adjust the parameters by taking a step in the opposite direction of the gradient. The step size is controlled by the learning rate.

```python
def gradient_descent_step(x, learning_rate):
    return x - learning_rate * gradient(x)

x = 2
learning_rate = 0.1
new_x = gradient_descent_step(x, learning_rate)

print(f"Old x: {x}")
print(f"New x: {new_x}")
print(f"Cost reduction: {cost_function(x) - cost_function(new_x)}")
```

Slide 6: Step 3: Repeat Until Convergence

The process repeats iteratively, with the model walking through parameter space, updating its position with each step and gradually reducing the error.

```python
def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    history = [x]
    for _ in range(num_iterations):
        x = gradient_descent_step(x, learning_rate)
        history.append(x)
    return x, history

final_x, history = gradient_descent(5, 0.1, 20)

print(f"Final x: {final_x}")
print(f"Final cost: {cost_function(final_x)}")

# Plot optimization path
x = np.linspace(-10, 10, 100)
y = cost_function(x)
plt.plot(x, y)
plt.plot(history, [cost_function(x) for x in history], 'ro-')
plt.title('Gradient Descent Optimization Path')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.show()
```

Slide 7: Batch Gradient Descent

Batch Gradient Descent uses the full dataset for each step, providing a stable but potentially slow descent towards the minimum.

```python
import numpy as np

def batch_gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        h = np.dot(X, theta)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
    
    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
theta = batch_gradient_descent(X, y, 0.01, 1000)
print("Optimized parameters:", theta)
```

Slide 8: Stochastic Gradient Descent (SGD)

SGD updates parameters after each data point, making it faster but noisier compared to batch gradient descent.

```python
import numpy as np

def stochastic_gradient_descent(X, y, learning_rate, num_epochs):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_epochs):
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
print("Optimized parameters:", theta)
```

Slide 9: Mini-batch Gradient Descent

Mini-batch Gradient Descent combines elements of both batch and stochastic methods, balancing speed and precision.

```python
import numpy as np

def mini_batch_gradient_descent(X, y, learning_rate, num_epochs, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_epochs):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        
        for i in range(0, m, batch_size):
            xi = X[i:i+batch_size]
            yi = y[i:i+batch_size]
            gradient = xi.T.dot(xi.dot(theta) - yi) / batch_size
            theta -= learning_rate * gradient
    
    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
y = np.array([1, 2, 3, 4, 5, 6])
theta = mini_batch_gradient_descent(X, y, 0.01, 1000, 2)
print("Optimized parameters:", theta)
```

Slide 10: Learning Rate

The learning rate is a crucial hyperparameter that controls the step size in parameter updates. Too large a learning rate can cause divergence, while too small a rate leads to slow convergence.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    history = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * (2*x + 5)
        history.append(x)
    return history

x = np.linspace(-10, 10, 100)
y = x**2 + 5*x + 10

plt.figure(figsize=(12, 4))
for lr in [0.01, 0.1, 0.5]:
    history = gradient_descent(8, lr, 20)
    plt.plot(history, [i**2 + 5*i + 10 for i in history], 'o-', label=f'LR = {lr}')

plt.plot(x, y, 'r--')
plt.title('Effect of Learning Rate on Convergence')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.legend()
plt.show()
```

Slide 11: Momentum

Momentum is a technique that helps accelerate gradient descent in the relevant direction and dampens oscillations.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_momentum(start_x, learning_rate, momentum, num_iterations):
    x = start_x
    velocity = 0
    history = [x]
    for _ in range(num_iterations):
        gradient = 2*x + 5
        velocity = momentum * velocity - learning_rate * gradient
        x += velocity
        history.append(x)
    return history

x = np.linspace(-10, 10, 100)
y = x**2 + 5*x + 10

plt.figure(figsize=(12, 4))
history_standard = gradient_descent(8, 0.1, 20)
history_momentum = gradient_descent_momentum(8, 0.1, 0.9, 20)

plt.plot(history_standard, [i**2 + 5*i + 10 for i in history_standard], 'o-', label='Standard GD')
plt.plot(history_momentum, [i**2 + 5*i + 10 for i in history_momentum], 'o-', label='GD with Momentum')
plt.plot(x, y, 'r--')
plt.title('Standard Gradient Descent vs Gradient Descent with Momentum')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.legend()
plt.show()
```

Slide 12: Real-life Example: Linear Regression

Gradient descent is commonly used in linear regression to find the best-fitting line for a set of data points.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient descent for linear regression
X_b = np.c_[np.ones((100, 1)), X]  # add bias term
theta = np.random.randn(2, 1)

learning_rate = 0.1
n_iterations = 1000
m = 100

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("Final parameters:", theta.ravel())

# Plot results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='r')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 13: Real-life Example: Image Classification

Gradient descent is crucial in training neural networks for image classification tasks.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define simple neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, weights):
    return sigmoid(np.dot(X, weights))

def backward(X, y, output):
    return np.dot(X.T, (output - y)) / len(y)

# Train network
np.random.seed(42)
n_features = X_train.shape[1]
n_classes = 10
weights = np.random.randn(n_features, n_classes)

learning_rate = 0.01
n_iterations = 1000

for _ in range(n_iterations):
    output = forward(X_train, weights)
    gradient = backward(X_train, np.eye(n_classes)[y_train], output)
    weights -= learning_rate * gradient

# Evaluate
predictions = np.argmax(forward(X_test, weights), axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Visualize a prediction
sample_index = np.random.randint(len(X_test))
sample_image = X_test[sample_index].reshape(8, 8)
sample_prediction = predictions[sample_index]

plt.imshow(sample_image, cmap='gray')
plt.title(f"Prediction: {sample_prediction}")
plt.axis('off')
plt.show()
```

Slide 14: Conclusion and Additional Resources

Gradient descent is a powerful optimization technique that enables machine learning models to learn from data and make accurate predictions. Its variants, such as SGD and mini-batch gradient descent, offer flexibility in balancing computational efficiency and convergence stability.

For further exploration of gradient descent and its applications in machine learning, consider the following resources:

1.  "Gradient Descent Revisited" by S. Ruder (2016), arXiv:1609.04747 URL: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2.  "An Overview of Gradient Descent Optimization Algorithms" by S. Ruder (2017), arXiv:1609.04747v2 URL: [https://arxiv.org/abs/1609.04747v2](https://arxiv.org/abs/1609.04747v2)

