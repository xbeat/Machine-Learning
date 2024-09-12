## Building a RMSprop Optimizer from Scratch in Python
Slide 1: Introduction to RMSprop Optimizer

RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimization algorithm designed to address the diminishing learning rates in AdaGrad. It was proposed by Geoffrey Hinton in 2012 and has since become a popular choice for training neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_rate(iterations, learning_rates):
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, learning_rates)
    plt.title('RMSprop Learning Rate Adaptation')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.show()

# Simulated learning rate adaptation
iterations = np.arange(1, 1001)
learning_rates = 0.01 / np.sqrt(1 + 0.1 * iterations)

plot_learning_rate(iterations, learning_rates)
```

Slide 2: The Problem with Fixed Learning Rates

Fixed learning rates can lead to slow convergence or oscillations in the optimization process. RMSprop addresses this by adapting the learning rate for each parameter based on the historical gradient information.

```python
import numpy as np
import matplotlib.pyplot as plt

def optimize_fixed_lr(learning_rate, iterations):
    x = 5
    trajectory = [x]
    for _ in range(iterations):
        gradient = 2 * x
        x = x - learning_rate * gradient
        trajectory.append(x)
    return trajectory

trajectories = {
    'High LR': optimize_fixed_lr(0.1, 50),
    'Low LR': optimize_fixed_lr(0.01, 50)
}

plt.figure(figsize=(10, 6))
for label, traj in trajectories.items():
    plt.plot(traj, label=label)
plt.title('Optimization with Fixed Learning Rates')
plt.xlabel('Iterations')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()
```

Slide 3: RMSprop Algorithm Overview

RMSprop maintains a moving average of squared gradients for each parameter. It then uses this average to normalize the gradients, which allows the learning rate to be adapted for each parameter individually.

```python
def rmsprop_update(param, grad, cache, learning_rate, decay_rate=0.9, epsilon=1e-8):
    cache = decay_rate * cache + (1 - decay_rate) * grad**2
    update = learning_rate * grad / (np.sqrt(cache) + epsilon)
    param -= update
    return param, cache

# Example usage
param = 5.0
grad = 2.0
cache = 0.0
learning_rate = 0.01

for _ in range(5):
    param, cache = rmsprop_update(param, grad, cache, learning_rate)
    print(f"Parameter: {param:.4f}, Cache: {cache:.4f}")
```

Slide 4: Implementing the RMSprop Optimizer Class

We'll create a Python class for the RMSprop optimizer, which will store the optimization parameters and implement the update rule.

```python
import numpy as np

class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        for param_name in params:
            if param_name not in self.cache:
                self.cache[param_name] = np.zeros_like(params[param_name])

            self.cache[param_name] = self.decay_rate * self.cache[param_name] + \
                                     (1 - self.decay_rate) * np.square(grads[param_name])
            
            params[param_name] -= self.learning_rate * grads[param_name] / \
                                  (np.sqrt(self.cache[param_name]) + self.epsilon)

        return params

# Example usage
optimizer = RMSprop()
params = {'w': np.array([1.0, 2.0, 3.0]), 'b': np.array([0.1])}
grads = {'w': np.array([0.1, 0.2, 0.3]), 'b': np.array([0.01])}

updated_params = optimizer.update(params, grads)
print("Updated parameters:", updated_params)
```

Slide 5: Understanding the Decay Rate

The decay rate in RMSprop controls how much the algorithm "remembers" about past gradients. A higher decay rate gives more weight to recent gradients, while a lower rate considers a longer history.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decay_rate_effect(decay_rates, iterations):
    plt.figure(figsize=(12, 6))
    for decay_rate in decay_rates:
        cache = np.zeros(iterations)
        for i in range(1, iterations):
            cache[i] = decay_rate * cache[i-1] + (1 - decay_rate) * 1
        plt.plot(cache, label=f'Decay rate: {decay_rate}')
    
    plt.title('Effect of Decay Rate on Cache Values')
    plt.xlabel('Iterations')
    plt.ylabel('Cache Value')
    plt.legend()
    plt.show()

decay_rates = [0.9, 0.99, 0.999]
plot_decay_rate_effect(decay_rates, 100)
```

Slide 6: Handling Different Parameter Types

Our RMSprop implementation should be able to handle different types of parameters, such as weights and biases, which may have different shapes.

```python
class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, params, grads):
        for param_name, param_value in params.items():
            grad_value = grads[param_name]
            
            if param_name not in self.cache:
                self.cache[param_name] = np.zeros_like(param_value)

            self.cache[param_name] = self.decay_rate * self.cache[param_name] + \
                                     (1 - self.decay_rate) * np.square(grad_value)
            
            params[param_name] -= self.learning_rate * grad_value / \
                                  (np.sqrt(self.cache[param_name]) + self.epsilon)

        return params

# Example with different parameter shapes
optimizer = RMSprop()
params = {
    'w1': np.random.randn(3, 2),
    'b1': np.zeros(2),
    'w2': np.random.randn(2, 1),
    'b2': np.zeros(1)
}
grads = {
    'w1': np.random.randn(3, 2),
    'b1': np.random.randn(2),
    'w2': np.random.randn(2, 1),
    'b2': np.random.randn(1)
}

updated_params = optimizer.update(params, grads)
for param_name, param_value in updated_params.items():
    print(f"{param_name} shape: {param_value.shape}")
```

Slide 7: Comparing RMSprop with Gradient Descent

Let's compare the performance of RMSprop with standard gradient descent on a simple optimization problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x**2

def gradient(x):
    return 2*x

def optimize(optimizer, initial_x, iterations):
    x = initial_x
    trajectory = [x]
    for _ in range(iterations):
        grad = gradient(x)
        x = optimizer.update({'x': np.array([x])}, {'x': np.array([grad])})['x'][0]
        trajectory.append(x)
    return trajectory

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for param_name in params:
            params[param_name] -= self.learning_rate * grads[param_name]
        return params

initial_x = 5.0
iterations = 50

gd_optimizer = GradientDescent(learning_rate=0.1)
rmsprop_optimizer = RMSprop(learning_rate=0.1)

gd_trajectory = optimize(gd_optimizer, initial_x, iterations)
rmsprop_trajectory = optimize(rmsprop_optimizer, initial_x, iterations)

plt.figure(figsize=(10, 6))
plt.plot(gd_trajectory, label='Gradient Descent')
plt.plot(rmsprop_trajectory, label='RMSprop')
plt.title('Optimization Comparison: Gradient Descent vs RMSprop')
plt.xlabel('Iterations')
plt.ylabel('Parameter Value')
plt.legend()
plt.show()
```

Slide 8: RMSprop in Neural Network Training

Let's implement a simple neural network and train it using our RMSprop optimizer.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.dz2 = output - y
        self.dw2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        self.dz1 = np.dot(self.dz2, self.w2.T) * sigmoid_derivative(self.a1)
        self.dw1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0)

    def get_params(self):
        return {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}

    def set_params(self, params):
        self.w1, self.b1, self.w2, self.b2 = params['w1'], params['b1'], params['w2'], params['b2']

    def get_grads(self):
        return {'w1': self.dw1, 'b1': self.db1, 'w2': self.dw2, 'b2': self.db2}

# Training example
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(3, 4, 1)
optimizer = RMSprop(learning_rate=0.1)

for _ in range(10000):
    output = nn.forward(X)
    nn.backward(X, y, output)
    params = optimizer.update(nn.get_params(), nn.get_grads())
    nn.set_params(params)

print("Final output:", nn.forward(X))
```

Slide 9: Visualizing RMSprop Optimization

Let's visualize how RMSprop optimizes a 2D function compared to standard gradient descent.

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def optimize_2d(optimizer, initial_point, iterations):
    point = initial_point
    trajectory = [point]
    for _ in range(iterations):
        grad = rosenbrock_gradient(point[0], point[1])
        point = optimizer.update({'p': point}, {'p': grad})['p']
        trajectory.append(point)
    return np.array(trajectory)

initial_point = np.array([-1.5, 2.5])
iterations = 1000

gd_optimizer = GradientDescent(learning_rate=0.001)
rmsprop_optimizer = RMSprop(learning_rate=0.01)

gd_trajectory = optimize_2d(gd_optimizer, initial_point, iterations)
rmsprop_trajectory = optimize_2d(rmsprop_optimizer, initial_point, iterations)

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
plt.colorbar(label='Rosenbrock function value')
plt.plot(gd_trajectory[:, 0], gd_trajectory[:, 1], 'r-', label='Gradient Descent')
plt.plot(rmsprop_trajectory[:, 0], rmsprop_trajectory[:, 1], 'g-', label='RMSprop')
plt.legend()
plt.title('Optimization Trajectories')

plt.subplot(122)
plt.semilogy(np.arange(iterations+1), rosenbrock(gd_trajectory[:, 0], gd_trajectory[:, 1]), 'r-', label='Gradient Descent')
plt.semilogy(np.arange(iterations+1), rosenbrock(rmsprop_trajectory[:, 0], rmsprop_trajectory[:, 1]), 'g-', label='RMSprop')
plt.legend()
plt.title('Convergence Comparison')
plt.xlabel('Iterations')
plt.ylabel('Rosenbrock function value (log scale)')

plt.tight_layout()
plt.show()
```

Slide 10: Real-life Example: Image Classification

Let's use our RMSprop optimizer to train a simple neural network for image classification on the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32')
y = y.astype('int')

# Normalize and split the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, output):
        n_samples = X.shape[0]
        delta3 = output
        delta3[range(n_samples), y] -= 1
        delta3 /= n_samples
        
        dw2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = np.dot(delta3, self.w2.T)
        delta2[self.a1 <= 0] = 0
        
        dw1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        return {'w1': dw1, 'b1': db1, 'w2': dw2, 'b2': db2}

# Training loop and results visualization would follow here
```

Slide 11: Training the Neural Network with RMSprop

Now let's train our simple neural network using the RMSprop optimizer we built earlier.

```python
def train(nn, X_train, y_train, optimizer, epochs, batch_size):
    n_samples = X_train.shape[0]
    losses = []

    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Forward pass
            probs = nn.forward(X_batch)
            
            # Compute loss
            correct_logprobs = -np.log(probs[range(len(y_batch)), y_batch])
            loss = np.sum(correct_logprobs) / len(y_batch)
            losses.append(loss)
            
            # Backward pass
            grads = nn.backward(X_batch, y_batch, probs)
            
            # Update parameters
            params = {'w1': nn.w1, 'b1': nn.b1, 'w2': nn.w2, 'b2': nn.b2}
            updated_params = optimizer.update(params, grads)
            nn.w1, nn.b1, nn.w2, nn.b2 = updated_params['w1'], updated_params['b1'], updated_params['w2'], updated_params['b2']
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

# Initialize and train the network
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # 10 digits

nn = SimpleNN(input_size, hidden_size, output_size)
optimizer = RMSprop(learning_rate=0.001)

train(nn, X_train, y_train, optimizer, epochs=100, batch_size=128)
```

Slide 12: Evaluating the Trained Model

After training, let's evaluate our model's performance on the test set.

```python
def predict(nn, X):
    probs = nn.forward(X)
    return np.argmax(probs, axis=1)

def accuracy(predictions, labels):
    return np.mean(predictions == labels)

# Make predictions on test set
y_pred = predict(nn, X_test)

# Calculate and print accuracy
acc = accuracy(y_pred, y_test)
print(f"Test accuracy: {acc:.2f}")

# Visualize some predictions
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 13: RMSprop vs Other Optimizers

Let's compare RMSprop with other popular optimizers like SGD and Adam on a simple 2D optimization problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        for param in params:
            params[param] -= self.learning_rate * grads[param]
        return params

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        for param in params:
            if param not in self.m:
                self.m[param] = np.zeros_like(params[param])
                self.v[param] = np.zeros_like(params[param])
            
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grads[param]
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grads[param]**2)
            
            m_hat = self.m[param] / (1 - self.beta1**self.t)
            v_hat = self.v[param] / (1 - self.beta2**self.t)
            
            params[param] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params

def optimize(optimizer, start_point, steps):
    x, y = start_point
    path = [start_point]
    for _ in range(steps):
        grad = rosenbrock_grad(x, y)
        params = optimizer.update({'p': np.array([x, y])}, {'p': grad})
        x, y = params['p']
        path.append((x, y))
    return np.array(path)

# Run optimizations
start = (-1.5, 2.5)
steps = 1000

optimizers = {
    'SGD': SGD(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.01)
}

paths = {name: optimize(opt, start, steps) for name, opt in optimizers.items()}

# Plot results
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(12, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), norm=LogNorm(), cmap='viridis')
for name, path in paths.items():
    plt.plot(path[:, 0], path[:, 1], label=name, linewidth=2)
plt.plot(*start, 'ro', label='Start')
plt.legend()
plt.title('Optimizer Comparison on Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='z')
plt.show()
```

Slide 14: Real-life Example: Natural Language Processing

Let's use RMSprop to train a simple recurrent neural network for sentiment analysis on movie reviews.

```python
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load movie reviews dataset
reviews = load_files(r'path_to_movie_reviews_dataset')
X, y = reviews.data, reviews.target

# Preprocess text data
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X).toarray()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = {0: h}

        for t, x in enumerate(inputs):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[t + 1] = h

        y = np.dot(self.Why, h) + self.by
        p = np.exp(y) / np.sum(np.exp(y))
        return p

    def backward(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)
        
        d_Why = np.dot(d_y, self.last_hs[n].T)
        d_by = d_y

        d_h = np.dot(self.Why.T, d_y)
        d_Wxh, d_Whh, d_bh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.bh)

        for t in reversed(range(n)):
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
            d_bh += temp
            d_Wxh += np.dot(temp, self.last_inputs[t].T)
            d_Whh += np.dot(temp, self.last_hs[t].T)
            d_h = np.dot(self.Whh.T, temp)

        for d_param in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d_param, -1, 1, out=d_param)

        self.Wxh -= learn_rate * d_Wxh
        self.Whh -= learn_rate * d_Whh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by

# Training loop and results visualization would follow here
```

Slide 15: Additional Resources

For those interested in diving deeper into RMSprop and other optimization algorithms, here are some valuable resources:

1. Original RMSprop Lecture Slides by Geoffrey Hinton: [https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture\_slides\_lec6.pdf](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
2. "An overview of gradient descent optimization algorithms" by Sebastian Ruder: ArXiv link: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by Duchi et al.: ArXiv link: [https://arxiv.org/abs/1101.3618](https://arxiv.org/abs/1101.3618)

These resources provide in-depth explanations and comparisons of various optimization algorithms, including RMSprop, helping to build a stronger understanding of their strengths and use cases in different scenarios.

