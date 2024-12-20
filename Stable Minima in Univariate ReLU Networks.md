## Stable Minima in Univariate ReLU Networks
Slide 1: Understanding Stable Minima in Univariate ReLU Networks

Stable minima play a crucial role in the generalization capabilities of neural networks. In this presentation, we'll explore how stable minima cannot overfit in univariate ReLU networks and how large step sizes contribute to generalization. We'll use Python to illustrate these concepts and provide practical examples.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def plot_relu():
    x = np.linspace(-5, 5, 100)
    y = relu(x)
    plt.plot(x, y)
    plt.title("ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True)
    plt.show()

plot_relu()
```

Slide 2: Univariate ReLU Networks: A Simple Model

Univariate ReLU networks are simple neural networks with one input, one output, and ReLU activation functions. These networks serve as a good starting point for understanding the behavior of more complex models. Let's implement a basic univariate ReLU network in Python.

```python
class UnivariateReLUNetwork:
    def __init__(self, num_neurons):
        self.weights = np.random.randn(num_neurons)
        self.biases = np.random.randn(num_neurons)
        
    def forward(self, x):
        activations = relu(x * self.weights + self.biases)
        return np.sum(activations)

# Example usage
network = UnivariateReLUNetwork(3)
x = np.linspace(-5, 5, 100)
y = [network.forward(xi) for xi in x]

plt.plot(x, y)
plt.title("Univariate ReLU Network Output")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
```

Slide 3: Stable Minima: Definition and Importance

Stable minima are local minima of the loss function that are robust to small perturbations in the network parameters. These minima are crucial for generalization because they represent solutions that are less likely to overfit the training data. Let's visualize a simple loss landscape to understand stable minima better.

```python
def loss_landscape(w, b):
    return np.sin(5 * w) * np.cos(5 * b) + w**2 + b**2

w = np.linspace(-2, 2, 100)
b = np.linspace(-2, 2, 100)
W, B = np.meshgrid(w, b)
L = loss_landscape(W, B)

plt.contourf(W, B, L, levels=20)
plt.colorbar(label='Loss')
plt.title("Loss Landscape")
plt.xlabel("Weight")
plt.ylabel("Bias")
plt.show()
```

Slide 4: Overfitting in Neural Networks

Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, leading to poor generalization on unseen data. In the context of univariate ReLU networks, we'll explore how stable minima help prevent overfitting.

```python
def generate_data(n_samples, noise_level=0.1):
    X = np.linspace(-5, 5, n_samples)
    y = np.sin(X) + np.random.normal(0, noise_level, n_samples)
    return X, y

def plot_overfitting():
    X, y = generate_data(50)
    X_test = np.linspace(-5, 5, 200)
    
    # Overfitted model
    coeffs_overfit = np.polyfit(X, y, 15)
    y_overfit = np.polyval(coeffs_overfit, X_test)
    
    # Well-fitted model
    coeffs_good = np.polyfit(X, y, 3)
    y_good = np.polyval(coeffs_good, X_test)
    
    plt.scatter(X, y, label='Data')
    plt.plot(X_test, y_overfit, label='Overfitted')
    plt.plot(X_test, y_good, label='Well-fitted')
    plt.legend()
    plt.title("Overfitting vs. Good Fit")
    plt.show()

plot_overfitting()
```

Slide 5: Large Step Sizes and Generalization

Large step sizes in gradient descent can help the optimization process escape narrow local minima and find wider, more stable minima. This contributes to better generalization. Let's implement a simple gradient descent algorithm with different step sizes to illustrate this concept.

```python
def gradient_descent(f, grad_f, x0, learning_rate, num_iterations):
    x = x0
    trajectory = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * grad_f(x)
        trajectory.append(x)
    return np.array(trajectory)

def f(x):
    return x**4 - 4*x**2 + x

def grad_f(x):
    return 4*x**3 - 8*x + 1

x = np.linspace(-2.5, 2.5, 100)
plt.plot(x, f(x), label='f(x)')

for lr in [0.01, 0.1, 0.5]:
    trajectory = gradient_descent(f, grad_f, 2, lr, 50)
    plt.plot(trajectory, f(trajectory), 'o-', label=f'LR = {lr}')

plt.legend()
plt.title("Gradient Descent with Different Step Sizes")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

Slide 6: ReLU Networks and Piecewise Linear Functions

Univariate ReLU networks represent piecewise linear functions. This property is key to understanding their behavior and generalization capabilities. Let's visualize how a univariate ReLU network approximates a non-linear function.

```python
def target_function(x):
    return np.sin(x) + 0.5 * x

class UnivariateReLUNetwork:
    def __init__(self, num_neurons):
        self.weights = np.random.randn(num_neurons)
        self.biases = np.random.randn(num_neurons)
    
    def forward(self, x):
        return np.sum(relu(x * self.weights + self.biases))

network = UnivariateReLUNetwork(10)
x = np.linspace(-5, 5, 200)
y_target = target_function(x)
y_network = np.array([network.forward(xi) for xi in x])

plt.plot(x, y_target, label='Target Function')
plt.plot(x, y_network, label='ReLU Network')
plt.legend()
plt.title("ReLU Network Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
```

Slide 7: Loss Landscape and Optimization

The loss landscape of univariate ReLU networks has interesting properties that affect optimization and generalization. Let's visualize a simplified loss landscape and how different optimization trajectories might look.

```python
def simplified_loss(w1, w2):
    return (w1**2 - 1)**2 + (w2**2 - 1)**2 + 0.1 * (w1 + w2)**2

w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1, w2)
L = simplified_loss(W1, W2)

plt.contourf(W1, W2, L, levels=20, cmap='viridis')
plt.colorbar(label='Loss')

# Simulated optimization trajectories
trajectory1 = np.random.randn(20, 2)
trajectory2 = np.random.randn(20, 2)

plt.plot(trajectory1[:, 0], trajectory1[:, 1], 'r-', label='Trajectory 1')
plt.plot(trajectory2[:, 0], trajectory2[:, 1], 'w-', label='Trajectory 2')

plt.title("Simplified Loss Landscape and Optimization Trajectories")
plt.xlabel("Weight 1")
plt.ylabel("Weight 2")
plt.legend()
plt.show()
```

Slide 8: Generalization Bounds for ReLU Networks

Generalization bounds provide theoretical guarantees on the performance of a model on unseen data. For univariate ReLU networks, these bounds are related to the network's complexity and the stability of its minima. Let's simulate the generalization performance of networks with different complexities.

```python
def generate_data(n_samples):
    X = np.linspace(-5, 5, n_samples)
    y = np.sin(X) + np.random.normal(0, 0.1, n_samples)
    return X, y

def train_network(network, X, y, epochs=1000, lr=0.01):
    for _ in range(epochs):
        y_pred = np.array([network.forward(xi) for xi in X])
        loss = np.mean((y_pred - y)**2)
        grad = 2 * (y_pred - y)
        network.weights -= lr * grad * X[:, np.newaxis]
        network.biases -= lr * grad

X_train, y_train = generate_data(100)
X_test, y_test = generate_data(1000)

test_errors = []
neuron_counts = [1, 2, 5, 10, 20, 50]

for neurons in neuron_counts:
    network = UnivariateReLUNetwork(neurons)
    train_network(network, X_train, y_train)
    y_pred = np.array([network.forward(xi) for xi in X_test])
    test_error = np.mean((y_pred - y_test)**2)
    test_errors.append(test_error)

plt.plot(neuron_counts, test_errors, 'o-')
plt.title("Test Error vs. Network Complexity")
plt.xlabel("Number of Neurons")
plt.ylabel("Test Error")
plt.xscale('log')
plt.show()
```

Slide 9: Stable Minima and Parameter Sensitivity

Stable minima are characterized by low sensitivity to small perturbations in the network parameters. This property contributes to better generalization. Let's visualize how small changes in weights affect the network's output for stable and unstable minima.

```python
def perturb_network(network, perturbation_scale):
    perturbed_network = UnivariateReLUNetwork(len(network.weights))
    perturbed_network.weights = network.weights + np.random.normal(0, perturbation_scale, len(network.weights))
    perturbed_network.biases = network.biases + np.random.normal(0, perturbation_scale, len(network.biases))
    return perturbed_network

def plot_sensitivity(network, perturbation_scale):
    x = np.linspace(-5, 5, 100)
    y_original = np.array([network.forward(xi) for xi in x])
    
    plt.plot(x, y_original, label='Original')
    
    for _ in range(5):
        perturbed_net = perturb_network(network, perturbation_scale)
        y_perturbed = np.array([perturbed_net.forward(xi) for xi in x])
        plt.plot(x, y_perturbed, alpha=0.5)
    
    plt.title(f"Network Sensitivity (Perturbation Scale: {perturbation_scale})")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()

stable_network = UnivariateReLUNetwork(10)
unstable_network = UnivariateReLUNetwork(50)

plot_sensitivity(stable_network, 0.1)
plot_sensitivity(unstable_network, 0.1)
```

Slide 10: Large Step Sizes and Escaping Local Minima

Large step sizes in gradient descent can help the optimization process escape narrow local minima, leading to more stable solutions. Let's visualize this phenomenon using a toy optimization problem.

```python
def toy_loss(x):
    return np.sin(5 * x) + 0.1 * x**2

def toy_grad(x):
    return 5 * np.cos(5 * x) + 0.2 * x

def optimize(loss_fn, grad_fn, x0, learning_rate, num_iterations):
    x = x0
    trajectory = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * grad_fn(x)
        trajectory.append(x)
    return np.array(trajectory)

x = np.linspace(-2, 2, 200)
plt.plot(x, toy_loss(x), label='Loss Function')

for lr in [0.01, 0.1, 0.5]:
    trajectory = optimize(toy_loss, toy_grad, 1.5, lr, 50)
    plt.plot(trajectory, toy_loss(trajectory), 'o-', label=f'LR = {lr}')

plt.legend()
plt.title("Optimization with Different Step Sizes")
plt.xlabel("x")
plt.ylabel("Loss")
plt.show()
```

Slide 11: Practical Example: Binary Classification

Let's apply our understanding of stable minima and generalization to a practical binary classification problem using a univariate ReLU network.

```python
def generate_binary_data(n_samples):
    X = np.random.uniform(-5, 5, n_samples)
    y = (np.sin(X) > 0).astype(int)
    return X, y

def binary_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-10) + (1 - y_true) * np.log(1 - y_pred + 1e-10))

def train_binary_network(network, X, y, epochs=1000, lr=0.01):
    for _ in range(epochs):
        y_pred = np.array([network.forward(xi) for xi in X])
        y_pred = 1 / (1 + np.exp(-y_pred))  # Sigmoid activation
        loss = binary_loss(y, y_pred)
        grad = (y_pred - y) / len(y)
        network.weights -= lr * grad * X[:, np.newaxis]
        network.biases -= lr * grad

X_train, y_train = generate_binary_data(100)
X_test, y_test = generate_binary_data(1000)

network = UnivariateReLUNetwork(10)
train_binary_network(network, X_train, y_train)

X_plot = np.linspace(-5, 5, 200)
y_plot = np.array([network.forward(xi) for xi in X_plot])
y_plot = 1 / (1 + np.exp(-y_plot))

plt.scatter(X_train, y_train, c=y_train, cmap='coolwarm', alpha=0.5)
plt.plot(X_plot, y_plot, 'g-', label='Decision Boundary')
plt.title("Binary Classification with Univariate ReLU Network")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 12: Real-life Example: Stock Price Prediction

Let's apply our understanding of stable minima and generalization to a real-life example of stock price prediction using a univariate ReLU network. We'll use historical stock data to predict future prices.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_stock_data(n_samples):
    t = np.linspace(0, 100, n_samples)
    price = 100 + 10 * np.sin(t/10) + t + np.random.normal(0, 5, n_samples)
    return t, price

def prepare_data(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

t, price = generate_stock_data(1000)
X, y = prepare_data(price, lookback=10)

plt.figure(figsize=(12, 6))
plt.plot(t, price)
plt.title("Simulated Stock Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# Train the ReLU network (implementation details omitted for brevity)
# Predict future prices
# Plot predictions vs actual prices
```

Slide 13: Limitations and Future Directions

While stable minima in univariate ReLU networks provide insights into generalization, there are limitations and areas for future research:

1. Extension to multivariate inputs and deep networks
2. Consideration of different activation functions
3. Impact of regularization techniques on stable minima
4. Connection to other generalization theories

Researchers continue to explore these areas to develop a more comprehensive understanding of neural network generalization.

```python
def visualize_future_research():
    topics = ['Multivariate', 'Activation Functions', 'Regularization', 'Generalization Theories']
    importance = [0.8, 0.6, 0.7, 0.9]
    
    plt.figure(figsize=(10, 6))
    plt.bar(topics, importance)
    plt.title("Future Research Directions")
    plt.xlabel("Topics")
    plt.ylabel("Relative Importance")
    plt.ylim(0, 1)
    plt.show()

visualize_future_research()
```

Slide 14: Additional Resources

For those interested in diving deeper into the topic of stable minima and generalization in neural networks, here are some valuable resources:

1. ArXiv paper: "On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization" by Sanjeev Arora et al. ([https://arxiv.org/abs/1802.06509](https://arxiv.org/abs/1802.06509))
2. ArXiv paper: "Gradient Descent Finds Global Minima of Deep Neural Networks" by Simon S. Du et al. ([https://arxiv.org/abs/1811.03804](https://arxiv.org/abs/1811.03804))
3. ArXiv paper: "A Convergence Theory for Deep Learning via Over-Parameterization" by Zeyuan Allen-Zhu et al. ([https://arxiv.org/abs/1811.03962](https://arxiv.org/abs/1811.03962))

These papers provide theoretical foundations and empirical studies related to the concepts discussed in this presentation.

