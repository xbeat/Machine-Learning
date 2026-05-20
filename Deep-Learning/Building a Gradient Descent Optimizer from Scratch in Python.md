## Building a Gradient Descent Optimizer from Scratch in Python
Slide 1: Introduction to Gradient Descent

Gradient descent is a fundamental optimization algorithm in machine learning. It's used to minimize a cost function by iteratively moving in the direction of steepest descent. In this presentation, we'll build a gradient descent optimizer from scratch in Python.

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

The gradient is the vector of partial derivatives of a function. For a single-variable function, it's simply the derivative. We'll implement a function to calculate the gradient numerically.

```python
def gradient(func, x, epsilon=1e-7):
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)

x_test = 2
grad = gradient(cost_function, x_test)
print(f"Gradient at x = {x_test}: {grad}")
```

Slide 3: Basic Gradient Descent Algorithm

Here's a simple implementation of gradient descent. We'll use a fixed learning rate and a maximum number of iterations.

```python
def gradient_descent(func, initial_x, learning_rate, num_iterations):
    x = initial_x
    for _ in range(num_iterations):
        grad = gradient(func, x)
        x = x - learning_rate * grad
    return x

result = gradient_descent(cost_function, 5, 0.1, 100)
print(f"Minimum found at x = {result}")
```

Slide 4: Visualizing Gradient Descent

Let's visualize how gradient descent works by plotting the path it takes.

```python
def visualize_gradient_descent(func, initial_x, learning_rate, num_iterations):
    x = initial_x
    path = [x]
    for _ in range(num_iterations):
        grad = gradient(func, x)
        x = x - learning_rate * grad
        path.append(x)
    
    x_range = np.linspace(-10, 10, 100)
    plt.plot(x_range, func(x_range), label='Cost Function')
    plt.plot(path, func(np.array(path)), 'ro-', label='Gradient Descent Path')
    plt.title('Gradient Descent Visualization')
    plt.xlabel('x')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

visualize_gradient_descent(cost_function, 5, 0.1, 50)
```

Slide 5: Adaptive Learning Rate

A fixed learning rate may not always be optimal. Let's implement an adaptive learning rate that decreases over time.

```python
def adaptive_gradient_descent(func, initial_x, initial_lr, num_iterations):
    x = initial_x
    lr = initial_lr
    for i in range(num_iterations):
        grad = gradient(func, x)
        lr = initial_lr / (1 + i / 50)  # Decrease learning rate over time
        x = x - lr * grad
    return x

result = adaptive_gradient_descent(cost_function, 5, 0.5, 100)
print(f"Minimum found at x = {result}")
```

Slide 6: Momentum

Momentum helps accelerate gradient descent by adding a fraction of the previous update to the current one. This can help overcome local minima and speed up convergence.

```python
def momentum_gradient_descent(func, initial_x, learning_rate, momentum, num_iterations):
    x = initial_x
    velocity = 0
    for _ in range(num_iterations):
        grad = gradient(func, x)
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
    return x

result = momentum_gradient_descent(cost_function, 5, 0.1, 0.9, 100)
print(f"Minimum found at x = {result}")
```

Slide 7: Nesterov Accelerated Gradient (NAG)

NAG is an improvement over standard momentum. It calculates the gradient at the "looked-ahead" position, which can provide better convergence.

```python
def nag_gradient_descent(func, initial_x, learning_rate, momentum, num_iterations):
    x = initial_x
    velocity = 0
    for _ in range(num_iterations):
        x_ahead = x + momentum * velocity
        grad = gradient(func, x_ahead)
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
    return x

result = nag_gradient_descent(cost_function, 5, 0.1, 0.9, 100)
print(f"Minimum found at x = {result}")
```

Slide 8: AdaGrad (Adaptive Gradient)

AdaGrad adapts the learning rate to the parameters, performing smaller updates for frequently occurring features and larger updates for infrequent ones.

```python
def adagrad(func, initial_x, learning_rate, num_iterations, epsilon=1e-8):
    x = initial_x
    accumulated_grad = 0
    for _ in range(num_iterations):
        grad = gradient(func, x)
        accumulated_grad += grad ** 2
        x = x - (learning_rate / (np.sqrt(accumulated_grad) + epsilon)) * grad
    return x

result = adagrad(cost_function, 5, 0.5, 100)
print(f"Minimum found at x = {result}")
```

Slide 9: RMSProp

RMSProp is an extension of AdaGrad that uses a moving average of squared gradients to normalize the gradient.

```python
def rmsprop(func, initial_x, learning_rate, decay_rate, num_iterations, epsilon=1e-8):
    x = initial_x
    moving_avg = 0
    for _ in range(num_iterations):
        grad = gradient(func, x)
        moving_avg = decay_rate * moving_avg + (1 - decay_rate) * grad ** 2
        x = x - (learning_rate / (np.sqrt(moving_avg) + epsilon)) * grad
    return x

result = rmsprop(cost_function, 5, 0.01, 0.9, 100)
print(f"Minimum found at x = {result}")
```

Slide 10: Adam (Adaptive Moment Estimation)

Adam combines ideas from RMSProp and momentum methods, maintaining both a moving average of past gradients and a moving average of past squared gradients.

```python
def adam(func, initial_x, learning_rate, beta1, beta2, num_iterations, epsilon=1e-8):
    x = initial_x
    m = 0  # First moment estimate
    v = 0  # Second moment estimate
    for t in range(1, num_iterations + 1):
        grad = gradient(func, x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x

result = adam(cost_function, 5, 0.01, 0.9, 0.999, 100)
print(f"Minimum found at x = {result}")
```

Slide 11: Real-Life Example: Image Denoising

Let's apply gradient descent to denoise an image. We'll create a noisy image and use gradient descent to recover the original.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple image
x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
original_image = np.sin(5 * x) * np.cos(5 * y)

# Add noise
noisy_image = original_image + 0.5 * np.random.randn(*original_image.shape)

def denoise_cost(image):
    return np.sum((image - noisy_image) ** 2) + 0.1 * np.sum(np.abs(np.gradient(image)))

def denoise_gradient(image):
    return 2 * (image - noisy_image) + 0.1 * np.sign(np.gradient(image))

def denoise_gd(initial_image, learning_rate, num_iterations):
    image = initial_image.()
    for _ in range(num_iterations):
        grad = denoise_gradient(image)
        image -= learning_rate * grad
    return image

denoised_image = denoise_gd(noisy_image, 0.1, 100)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(noisy_image, cmap='gray')
axs[1].set_title('Noisy Image')
axs[2].imshow(denoised_image, cmap='gray')
axs[2].set_title('Denoised Image')
plt.show()
```

Slide 12: Real-Life Example: Curve Fitting

Let's use gradient descent to fit a polynomial curve to a set of data points.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some noisy data
x = np.linspace(0, 10, 100)
y = 2*x**2 - 5*x + 3 + np.random.randn(100) * 10

def polynomial(x, params):
    return params[0] * x**2 + params[1] * x + params[2]

def cost(params):
    return np.mean((polynomial(x, params) - y)**2)

def gradient(params):
    y_pred = polynomial(x, params)
    grad = np.zeros(3)
    grad[0] = np.mean(2 * (y_pred - y) * x**2)
    grad[1] = np.mean(2 * (y_pred - y) * x)
    grad[2] = np.mean(2 * (y_pred - y))
    return grad

def fit_curve(learning_rate, num_iterations):
    params = np.random.randn(3)
    for _ in range(num_iterations):
        params -= learning_rate * gradient(params)
    return params

fitted_params = fit_curve(0.0001, 1000)

plt.scatter(x, y, label='Data')
plt.plot(x, polynomial(x, fitted_params), 'r', label='Fitted Curve')
plt.legend()
plt.show()

print(f"Fitted parameters: a={fitted_params[0]:.2f}, b={fitted_params[1]:.2f}, c={fitted_params[2]:.2f}")
```

Slide 13: Comparison of Optimizers

Let's compare the performance of different optimizers on our original cost function.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

optimizers = [
    ('Gradient Descent', gradient_descent),
    ('Momentum', momentum_gradient_descent),
    ('NAG', nag_gradient_descent),
    ('AdaGrad', adagrad),
    ('RMSProp', rmsprop),
    ('Adam', adam)
]

plt.figure(figsize=(12, 8))
x_range = np.linspace(-10, 10, 100)
plt.plot(x_range, cost_function(x_range), 'k--', label='Cost Function')

for name, optimizer in optimizers:
    if name in ['Gradient Descent', 'Momentum', 'NAG']:
        result = optimizer(cost_function, 5, 0.1, 0.9, 100)
    elif name == 'AdaGrad':
        result = optimizer(cost_function, 5, 0.5, 100)
    elif name == 'RMSProp':
        result = optimizer(cost_function, 5, 0.01, 0.9, 100)
    else:  # Adam
        result = optimizer(cost_function, 5, 0.01, 0.9, 0.999, 100)
    
    plt.plot(result, cost_function(result), 'o', label=name)

plt.legend()
plt.title('Comparison of Optimizers')
plt.xlabel('x')
plt.ylabel('Cost')
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into gradient descent and optimization algorithms, here are some valuable resources:

1. "An overview of gradient descent optimization algorithms" by Sebastian Ruder ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "Optimization Methods for Large-Scale Machine Learning" by Léon Bottou, Frank E. Curtis, and Jorge Nocedal ArXiv: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
3. "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

These papers provide in-depth explanations of various optimization algorithms and their applications in machine learning.

