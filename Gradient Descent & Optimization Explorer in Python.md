## Gradient Descent & Optimization Explorer in Python
Slide 1: Introduction to Gradient Descent

Gradient Descent is a fundamental optimization algorithm used in machine learning to minimize the cost function of a model. It iteratively adjusts parameters to find the minimum of a function.

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 5, 100)
y = f(x)

plt.plot(x, y)
plt.title('Simple Quadratic Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 2: The Concept of Gradients

Gradients represent the direction of steepest increase in a function. In optimization, we move in the opposite direction of the gradient to find the minimum.

```python
def gradient(x):
    return 2*x + 5  # Derivative of f(x) = x^2 + 5x + 10

x = np.linspace(-10, 5, 100)
grad = gradient(x)

plt.plot(x, grad)
plt.title('Gradient of the Quadratic Function')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)
plt.show()
```

Slide 3: Basic Gradient Descent Algorithm

The algorithm updates the parameters iteratively by subtracting the gradient multiplied by a learning rate.

```python
def gradient_descent(start, learn_rate, num_iter):
    x = start
    for i in range(num_iter):
        grad = gradient(x)
        x = x - learn_rate * grad
    return x

minimum = gradient_descent(start=5, learn_rate=0.1, num_iter=50)
print(f"The minimum occurs at x = {minimum:.2f}")
```

Slide 4: Visualizing Gradient Descent

Let's visualize how gradient descent navigates the function landscape to find the minimum.

```python
def visualize_descent(start, learn_rate, num_iter):
    x = start
    path = [x]
    for i in range(num_iter):
        grad = gradient(x)
        x = x - learn_rate * grad
        path.append(x)
    return path

path = visualize_descent(start=5, learn_rate=0.1, num_iter=20)

x = np.linspace(-10, 5, 100)
plt.plot(x, f(x))
plt.plot(path, [f(x) for x in path], 'ro-')
plt.title('Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 5: Learning Rate Importance

The learning rate determines the step size at each iteration. It's crucial for convergence and efficiency.

```python
def compare_learning_rates():
    rates = [0.01, 0.1, 0.5]
    for rate in rates:
        path = visualize_descent(start=5, learn_rate=rate, num_iter=20)
        plt.plot(path, [f(x) for x in path], 'o-', label=f'LR = {rate}')
    
    x = np.linspace(-10, 5, 100)
    plt.plot(x, f(x), 'k--', label='Function')
    plt.title('Effect of Learning Rate')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

compare_learning_rates()
```

Slide 6: Stochastic Gradient Descent (SGD)

SGD approximates the true gradient using a subset of data, making it faster and more suitable for large datasets.

```python
import random

def stochastic_gradient_descent(data, start, learn_rate, num_iter):
    x = start
    for i in range(num_iter):
        random_point = random.choice(data)
        grad = gradient(random_point)
        x = x - learn_rate * grad
    return x

# Simulating a dataset
data = np.random.uniform(-5, 5, 1000)
minimum_sgd = stochastic_gradient_descent(data, start=5, learn_rate=0.1, num_iter=1000)
print(f"The minimum found by SGD occurs at x = {minimum_sgd:.2f}")
```

Slide 7: Mini-batch Gradient Descent

Mini-batch GD strikes a balance between full-batch and stochastic GD, using small random subsets of data.

```python
def mini_batch_gradient_descent(data, start, learn_rate, num_iter, batch_size):
    x = start
    for i in range(num_iter):
        batch = random.sample(list(data), batch_size)
        grad = np.mean([gradient(point) for point in batch])
        x = x - learn_rate * grad
    return x

minimum_mbgd = mini_batch_gradient_descent(data, start=5, learn_rate=0.1, num_iter=1000, batch_size=32)
print(f"The minimum found by Mini-batch GD occurs at x = {minimum_mbgd:.2f}")
```

Slide 8: Momentum in Gradient Descent

Momentum helps accelerate GD in relevant directions and dampens oscillations.

```python
def momentum_gradient_descent(start, learn_rate, num_iter, momentum):
    x = start
    velocity = 0
    path = [x]
    for i in range(num_iter):
        grad = gradient(x)
        velocity = momentum * velocity - learn_rate * grad
        x = x + velocity
        path.append(x)
    return path

path_momentum = momentum_gradient_descent(start=5, learn_rate=0.01, num_iter=100, momentum=0.9)

x = np.linspace(-10, 5, 100)
plt.plot(x, f(x), 'k--', label='Function')
plt.plot(path_momentum, [f(x) for x in path_momentum], 'r-', label='Momentum GD')
plt.title('Gradient Descent with Momentum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: Adaptive Learning Rates: AdaGrad

AdaGrad adapts the learning rate to the parameters, performing smaller updates for frequent features.

```python
def adagrad(start, learn_rate, num_iter, epsilon=1e-8):
    x = start
    sum_squared_grad = 0
    path = [x]
    for i in range(num_iter):
        grad = gradient(x)
        sum_squared_grad += grad ** 2
        adjusted_learn_rate = learn_rate / (np.sqrt(sum_squared_grad) + epsilon)
        x = x - adjusted_learn_rate * grad
        path.append(x)
    return path

path_adagrad = adagrad(start=5, learn_rate=1, num_iter=100)

x = np.linspace(-10, 5, 100)
plt.plot(x, f(x), 'k--', label='Function')
plt.plot(path_adagrad, [f(x) for x in path_adagrad], 'g-', label='AdaGrad')
plt.title('AdaGrad Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 10: RMSprop: Addressing AdaGrad's Radically Diminishing Learning Rates

RMSprop divides the learning rate by an exponentially decaying average of squared gradients.

```python
def rmsprop(start, learn_rate, num_iter, decay_rate=0.9, epsilon=1e-8):
    x = start
    avg_squared_grad = 0
    path = [x]
    for i in range(num_iter):
        grad = gradient(x)
        avg_squared_grad = decay_rate * avg_squared_grad + (1 - decay_rate) * grad ** 2
        adjusted_learn_rate = learn_rate / (np.sqrt(avg_squared_grad) + epsilon)
        x = x - adjusted_learn_rate * grad
        path.append(x)
    return path

path_rmsprop = rmsprop(start=5, learn_rate=0.1, num_iter=100)

x = np.linspace(-10, 5, 100)
plt.plot(x, f(x), 'k--', label='Function')
plt.plot(path_rmsprop, [f(x) for x in path_rmsprop], 'b-', label='RMSprop')
plt.title('RMSprop Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Adam: Adaptive Moment Estimation

Adam combines ideas from momentum and RMSprop, adapting the learning rate for each parameter.

```python
def adam(start, learn_rate, num_iter, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = start
    m = 0  # First moment estimate
    v = 0  # Second moment estimate
    path = [x]
    for i in range(1, num_iter + 1):
        grad = gradient(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** i)
        v_hat = v / (1 - beta2 ** i)
        x = x - learn_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x)
    return path

path_adam = adam(start=5, learn_rate=0.1, num_iter=100)

x = np.linspace(-10, 5, 100)
plt.plot(x, f(x), 'k--', label='Function')
plt.plot(path_adam, [f(x) for x in path_adam], 'm-', label='Adam')
plt.title('Adam Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Comparison of Optimization Algorithms

Let's compare the convergence of different optimization algorithms we've explored.

```python
def compare_optimizers():
    x = np.linspace(-10, 5, 100)
    plt.plot(x, f(x), 'k--', label='Function')
    
    optimizers = {
        'GD': visualize_descent(5, 0.1, 100),
        'Momentum': momentum_gradient_descent(5, 0.01, 100, 0.9),
        'AdaGrad': adagrad(5, 1, 100),
        'RMSprop': rmsprop(5, 0.1, 100),
        'Adam': adam(5, 0.1, 100)
    }
    
    for name, path in optimizers.items():
        plt.plot([p for p in path], [f(p) for p in path], '-', label=name)
    
    plt.title('Comparison of Optimization Algorithms')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

compare_optimizers()
```

Slide 13: Real-life Example: Image Brightness Adjustment

Let's use gradient descent to automatically adjust the brightness of an image.

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load an image and convert to grayscale
img = Image.open('example_image.jpg').convert('L')
img_array = np.array(img)

# Define the cost function (mean squared error from target brightness)
def cost(brightness, target=128):
    return np.mean((img_array + brightness - target) ** 2)

# Gradient of the cost function
def gradient(brightness, target=128):
    return 2 * np.mean(img_array + brightness - target)

# Gradient descent
brightness = 0
learning_rate = 0.1
iterations = 100

for i in range(iterations):
    grad = gradient(brightness)
    brightness -= learning_rate * grad

print(f"Optimal brightness adjustment: {brightness:.2f}")

# Apply the adjustment and display the result
adjusted_img = np.clip(img_array + brightness, 0, 255).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(adjusted_img, cmap='gray')
plt.title('Adjusted Image')
plt.show()
```

Slide 14: Real-life Example: Weather Prediction Model

Let's use gradient descent to train a simple linear regression model for predicting temperature based on humidity.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic weather data
np.random.seed(42)
humidity = np.random.uniform(30, 100, 100)
temperature = 0.5 * humidity + 10 + np.random.normal(0, 5, 100)

# Define the model and loss function
def predict(humidity, w, b):
    return w * humidity + b

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient descent
w, b = 0, 0
learning_rate = 0.0001
iterations = 1000

for i in range(iterations):
    y_pred = predict(humidity, w, b)
    dw = -2 * np.mean(humidity * (temperature - y_pred))
    db = -2 * np.mean(temperature - y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db

print(f"Learned model: Temperature = {w:.2f} * Humidity + {b:.2f}")

# Visualize the results
plt.scatter(humidity, temperature, alpha=0.5)
plt.plot(humidity, predict(humidity, w, b), 'r')
plt.xlabel('Humidity (%)')
plt.ylabel('Temperature (°C)')
plt.title('Weather Prediction Model')
plt.show()
```

Slide 15: Additional Resources

For a deeper dive into gradient descent and optimization algorithms, consider exploring these peer-reviewed articles:

1. "An overview of gradient descent optimization algorithms" by Sebastian Ruder (2017) ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by Duchi et al. (2011) ArXiv: [https://arxiv.org/abs/1101.3618](https://arxiv.org/abs/1101.3618)
3. "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014) ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

These resources provide comprehensive explanations and mathematical foundations of the algorithms discussed in this presentation.

