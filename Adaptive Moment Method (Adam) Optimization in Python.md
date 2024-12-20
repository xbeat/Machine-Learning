## Adaptive Moment Method (Adam) Optimization in Python

Slide 1: Introduction to Adaptive Moment Method (Adam)

The Adaptive Moment Method, commonly known as Adam, is an optimization algorithm for stochastic gradient descent. It combines the benefits of two other extensions of stochastic gradient descent: Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). Adam is designed to efficiently handle sparse gradients and noisy data, making it particularly useful for large datasets and high-dimensional parameter spaces.

```python
import math

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)
        
        self.t += 1
        
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            params[i] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
        
        return params
```

Slide 2: Adam Algorithm Explained

The Adam algorithm maintains two moving averages: the first moment (mean) and the second moment (uncentered variance) of the gradients. These moving averages are used to adapt the learning rate for each parameter. The algorithm also includes bias correction terms to counteract the initialization bias towards zero.

```python
def adam_step(params, grads, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for i in range(len(params)):
        # Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
        
        # Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m[i] / (1 - beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v[i] / (1 - beta2 ** t)
        
        # Update parameters
        params[i] -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)
    
    return params, m, v

# Example usage
params = [0, 0]
grads = [0.1, 0.2]
m = [0, 0]
v = [0, 0]
t = 1

params, m, v = adam_step(params, grads, m, v, t)
print(f"Updated parameters: {params}")
```

Slide 3: Advantages of Adam

Adam combines the strengths of AdaGrad and RMSProp, making it well-suited for a wide range of optimization problems. It adapts the learning rate for each parameter individually, which is particularly beneficial for sparse gradients and noisy data. Adam also incorporates momentum, which helps accelerate convergence and reduce oscillations in the optimization process.

```python
import random

def objective_function(x, y):
    return x**2 + y**2

def compute_gradients(x, y):
    return 2*x, 2*y

def adam_optimization(iterations=1000):
    x, y = random.uniform(-10, 10), random.uniform(-10, 10)
    adam = Adam()
    
    for i in range(iterations):
        grad_x, grad_y = compute_gradients(x, y)
        x, y = adam.update([x, y], [grad_x, grad_y])
        
        if i % 100 == 0:
            print(f"Iteration {i}: x={x:.4f}, y={y:.4f}, f(x,y)={objective_function(x, y):.4f}")
    
    return x, y

optimal_x, optimal_y = adam_optimization()
print(f"Optimal solution: x={optimal_x:.4f}, y={optimal_y:.4f}")
```

Slide 4: Hyperparameters in Adam

Adam has several hyperparameters that can be tuned to optimize its performance:

1.  Learning rate (α): Controls the step size during optimization.
2.  β1 and β2: Exponential decay rates for moment estimates.
3.  ε: A small constant for numerical stability.

These hyperparameters usually have good default values, but fine-tuning them can improve convergence for specific problems.

```python
class AdamTuner:
    def __init__(self, objective_function, parameter_ranges):
        self.objective_function = objective_function
        self.parameter_ranges = parameter_ranges
    
    def tune(self, num_trials=100, iterations=1000):
        best_params = None
        best_score = float('inf')
        
        for _ in range(num_trials):
            params = [random.uniform(r[0], r[1]) for r in self.parameter_ranges]
            adam = Adam(*params)
            x, y = random.uniform(-10, 10), random.uniform(-10, 10)
            
            for _ in range(iterations):
                grad_x, grad_y = compute_gradients(x, y)
                x, y = adam.update([x, y], [grad_x, grad_y])
            
            score = self.objective_function(x, y)
            if score < best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score

# Example usage
tuner = AdamTuner(objective_function, [(0.0001, 0.01), (0.8, 0.999), (0.8, 0.999), (1e-8, 1e-6)])
best_params, best_score = tuner.tune()
print(f"Best hyperparameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 5: Implementing Adam from Scratch

Let's implement the Adam optimizer from scratch using only built-in Python functions. This implementation will help us understand the inner workings of the algorithm without relying on external libraries.

```python
import math
import random

class AdamFromScratch:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if not self.m:
            self.m = {param: 0 for param in params}
            self.v = {param: 0 for param in params}

        for param, grad in zip(params, grads):
            # Update biased first moment estimate
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[param] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)

        return params

# Example usage
def objective_function(x, y):
    return x**2 + y**2

def optimize():
    adam = AdamFromScratch()
    params = {'x': random.uniform(-10, 10), 'y': random.uniform(-10, 10)}
    
    for i in range(1000):
        grads = [2 * params['x'], 2 * params['y']]
        params = adam.update(params, grads)
        
        if i % 100 == 0:
            print(f"Iteration {i}: x={params['x']:.4f}, y={params['y']:.4f}, f(x,y)={objective_function(params['x'], params['y']):.4f}")

    return params

final_params = optimize()
print(f"Final solution: x={final_params['x']:.4f}, y={final_params['y']:.4f}")
```

Slide 6: Visualizing Adam Optimization

To better understand how Adam works, let's create a visualization of the optimization process for a simple 2D function. We'll use Python's built-in plotting library, matplotlib, to create this visualization.

```python
import math
import random
import matplotlib.pyplot as plt

def plot_contour(ax, func, xrange, yrange, levels):
    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    ax.contour(X, Y, Z, levels=levels)

def visualize_adam_optimization():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def objective_function(x, y):
        return x**2 + y**2
    
    plot_contour(ax, objective_function, (-5, 5), (-5, 5), 20)
    
    adam = AdamFromScratch()
    x, y = random.uniform(-5, 5), random.uniform(-5, 5)
    trajectory = [(x, y)]
    
    for _ in range(50):
        grad_x, grad_y = 2*x, 2*y
        params = adam.update({'x': x, 'y': y}, [grad_x, grad_y])
        x, y = params['x'], params['y']
        trajectory.append((x, y))
    
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Adam Optimization Trajectory')
    plt.show()

visualize_adam_optimization()
```

Slide 7: Adam vs. Standard Gradient Descent

Let's compare the performance of Adam with standard gradient descent on a challenging optimization problem. We'll use the Rosenbrock function, which has a global minimum inside a long, narrow, parabolic valley.

```python
import math
import random
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def gradient_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return dx, dy

def optimize(optimizer, iterations=1000):
    x, y = random.uniform(-2, 2), random.uniform(-2, 2)
    trajectory = [(x, y)]
    
    for _ in range(iterations):
        grad_x, grad_y = gradient_rosenbrock(x, y)
        params = optimizer.update({'x': x, 'y': y}, [grad_x, grad_y])
        x, y = params['x'], params['y']
        trajectory.append((x, y))
    
    return trajectory

def plot_optimization_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    for ax in (ax1, ax2):
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    adam_trajectory = np.array(optimize(AdamFromScratch()))
    sgd_trajectory = np.array(optimize(AdamFromScratch(learning_rate=0.001, beta1=0, beta2=0)))
    
    ax1.plot(adam_trajectory[:, 0], adam_trajectory[:, 1], 'ro-', markersize=3)
    ax1.set_title('Adam Optimization')
    
    ax2.plot(sgd_trajectory[:, 0], sgd_trajectory[:, 1], 'bo-', markersize=3)
    ax2.set_title('Standard Gradient Descent')
    
    plt.show()

plot_optimization_comparison()
```

Slide 8: Adaptive Learning Rates in Adam

One of the key features of Adam is its adaptive learning rates for each parameter. Let's visualize how these learning rates change during optimization for a simple problem.

```python
import math
import random
import matplotlib.pyplot as plt

def quadratic(x):
    return x**2

def gradient_quadratic(x):
    return 2 * x

class AdamWithHistory(AdamFromScratch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate_history = []

    def update(self, params, grads):
        result = super().update(params, grads)
        self.learning_rate_history.append(self.learning_rate / (math.sqrt(self.v['x']) + self.epsilon))
        return result

def optimize_with_history():
    adam = AdamWithHistory()
    x = random.uniform(-10, 10)
    trajectory = [x]
    
    for _ in range(100):
        grad = gradient_quadratic(x)
        params = adam.update({'x': x}, [grad])
        x = params['x']
        trajectory.append(x)
    
    return trajectory, adam.learning_rate_history

def plot_adaptive_learning_rates():
    trajectory, learning_rates = optimize_with_history()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    ax1.plot(trajectory)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('x')
    ax1.set_title('Parameter Value over Time')
    
    ax2.plot(learning_rates)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Effective Learning Rate')
    ax2.set_title('Adaptive Learning Rate over Time')
    
    plt.tight_layout()
    plt.show()

plot_adaptive_learning_rates()
```

Slide 9: Handling Sparse Gradients with Adam

Adam is particularly effective in handling sparse gradients, which are common in many machine learning problems, especially in natural language processing. Let's simulate a sparse gradient scenario and see how Adam performs.

```python
import random
import math

def sparse_quadratic(x):
    return sum(xi**2 for xi in x)

def sparse_gradient(x):
    return [2 * xi if random.random() < 0.1 else 0 for xi in x]

def optimize_sparse(optimizer, dim=100, iterations=1000):
    x = [random.uniform(-1, 1) for _ in range(dim)]
    losses = []

    for _ in range(iterations):
        grad = sparse_gradient(x)
        params = {i: xi for i, xi in enumerate(x)}
        updated_params = optimizer.update(params, grad)
        x = [updated_params[i] for i in range(dim)]
        losses.append(sparse_quadratic(x))

    return losses

adam_optimizer = AdamFromScratch()
sgd_optimizer = AdamFromScratch(learning_rate=0.01, beta1=0, beta2=0)

adam_losses = optimize_sparse(adam_optimizer)
sgd_losses = optimize_sparse(sgd_optimizer)

# Plotting code would go here, but we'll skip it due to complexity constraints
```

Slide 10: Adam in Neural Networks

Adam is widely used in training neural networks. Let's implement a simple neural network using Adam as the optimizer. We'll create a basic feedforward neural network for binary classification.

```python
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]
        self.optimizer = AdamFromScratch()

    def forward(self, x):
        h = [sigmoid(sum(w*xi for w, xi in zip(row, x)) + b) for row, b in zip(self.w1, self.b1)]
        y = [sigmoid(sum(w*hi for w, hi in zip(row, h)) + b) for row, b in zip(self.w2, self.b2)]
        return y

    def train(self, x, y_true):
        # Forward pass
        h = [sigmoid(sum(w*xi for w, xi in zip(row, x)) + b) for row, b in zip(self.w1, self.b1)]
        y_pred = [sigmoid(sum(w*hi for w, hi in zip(row, h)) + b) for row, b in zip(self.w2, self.b2)]

        # Backward pass (simplified)
        d_y = [yi - yi_true for yi, yi_true in zip(y_pred, y_true)]
        d_w2 = [[d * hi for hi in h] for d in d_y]
        d_b2 = d_y
        d_h = [sum(d * w for d, w in zip(d_y, col)) for col in zip(*self.w2)]
        d_w1 = [[d * xi * hi * (1 - hi) for xi in x] for d, hi in zip(d_h, h)]
        d_b1 = [d * hi * (1 - hi) for d, hi in zip(d_h, h)]

        # Update weights and biases using Adam
        params = {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}
        grads = {'w1': d_w1, 'b1': d_b1, 'w2': d_w2, 'b2': d_b2}
        updated_params = self.optimizer.update(params, grads)

        self.w1, self.b1, self.w2, self.b2 = updated_params['w1'], updated_params['b1'], updated_params['w2'], updated_params['b2']

# Usage example would go here
```

Slide 11: Real-Life Example: Image Classification

Let's consider a simplified image classification task using a convolutional neural network (CNN) optimized with Adam. We'll use a basic CNN architecture for classifying handwritten digits.

```python
# Note: This is a simplified pseudocode representation
class CNN:
    def __init__(self):
        self.conv1 = ConvLayer(1, 32, 3, 3)
        self.conv2 = ConvLayer(32, 64, 3, 3)
        self.fc1 = FullyConnectedLayer(64 * 5 * 5, 128)
        self.fc2 = FullyConnectedLayer(128, 10)
        self.optimizer = AdamFromScratch()

    def forward(self, x):
        x = self.conv1(x)
        x = relu(x)
        x = max_pool(x, 2, 2)
        x = self.conv2(x)
        x = relu(x)
        x = max_pool(x, 2, 2)
        x = flatten(x)
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        return softmax(x)

    def train(self, x, y_true):
        y_pred = self.forward(x)
        loss = cross_entropy(y_pred, y_true)
        grads = compute_gradients(loss, self.parameters())
        self.optimizer.update(self.parameters(), grads)

# Training loop
cnn = CNN()
for epoch in range(num_epochs):
    for batch in data_loader:
        x, y_true = batch
        cnn.train(x, y_true)
```

Slide 12: Real-Life Example: Natural Language Processing

Adam is widely used in natural language processing tasks. Let's consider a simplified example of training a recurrent neural network (RNN) for sentiment analysis using Adam as the optimizer.

```python
# Note: This is a simplified pseudocode representation
class RNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn = RNNCell(embedding_dim, hidden_dim)
        self.fc = FullyConnectedLayer(hidden_dim, output_dim)
        self.optimizer = AdamFromScratch()

    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.init_hidden()
        for t in range(len(x)):
            output, hidden = self.rnn(embedded[t], hidden)
        return self.fc(output)

    def train(self, x, y_true):
        y_pred = self.forward(x)
        loss = binary_cross_entropy(y_pred, y_true)
        grads = compute_gradients(loss, self.parameters())
        self.optimizer.update(self.parameters(), grads)

# Training loop
rnn = RNN(vocab_size=10000, embedding_dim=100, hidden_dim=256, output_dim=1)
for epoch in range(num_epochs):
    for batch in data_loader:
        x, y_true = batch
        rnn.train(x, y_true)
```

Slide 13: Implementing Adam with Momentum

Adam incorporates momentum to accelerate training and reduce oscillations. Let's implement a version of Adam that explicitly shows the momentum calculations.

```python
import math

class AdamWithMomentum:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            self.m = {param: 0 for param in params}
            self.v = {param: 0 for param in params}
        
        self.t += 1
        
        for param in params:
            # Momentum update
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grads[param]
            
            # RMSprop update
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grads[param] ** 2)
            
            # Bias correction
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[param] -= self.learning_rate * m_hat / (math.sqrt(v_hat) + self.epsilon)
        
        return params

# Usage example
optimizer = AdamWithMomentum()
params = {'w': 0, 'b': 0}
grads = {'w': 0.1, 'b': 0.01}
updated_params = optimizer.update(params, grads)
```

Slide 14: Additional Resources

For those interested in diving deeper into the Adaptive Moment Method (Adam) and related optimization techniques, here are some valuable resources:

1.  Original Adam paper: Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980. Available at: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
2.  AdamW: Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. arXiv:1711.05101. Available at: [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
3.  Adam with AMSGrad: Reddi, S. J., Kale, S., & Kumar, S. (2018). On the Convergence of Adam and Beyond. arXiv:1904.09237. Available at: [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)

These papers provide in-depth explanations of the Adam algorithm, its variants, and improvements, offering valuable insights into the theoretical foundations and practical applications of adaptive optimization methods in machine learning.

