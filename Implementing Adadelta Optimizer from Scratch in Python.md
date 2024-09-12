## Implementing Adadelta Optimizer from Scratch in Python
Slide 1: Introduction to Adadelta Optimizer

Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w. Let's implement this optimizer from scratch in Python.

```python
import numpy as np

class Adadelta:
    def __init__(self, params, rho=0.95, eps=1e-6):
        self.params = params
        self.rho = rho
        self.eps = eps
        self.E_g2 = {p: np.zeros_like(p.data) for p in self.params}
        self.E_dx2 = {p: np.zeros_like(p.data) for p in self.params}

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            
            # Update running average of gradient squares
            self.E_g2[p] = self.rho * self.E_g2[p] + (1 - self.rho) * p.grad**2
            
            # Compute update
            RMS_g = np.sqrt(self.E_g2[p] + self.eps)
            RMS_dx = np.sqrt(self.E_dx2[p] + self.eps)
            dx = -(RMS_dx / RMS_g) * p.grad
            
            # Update running average of parameter updates
            self.E_dx2[p] = self.rho * self.E_dx2[p] + (1 - self.rho) * dx**2
            
            # Apply update
            p.data += dx
```

Slide 2: Understanding Adadelta Parameters

The Adadelta optimizer has two main parameters: rho and epsilon. Rho is the decay factor for the running averages, typically set to 0.95. Epsilon is a small constant added for numerical stability, usually around 1e-6. Let's create a function to visualize how these parameters affect the optimizer's behavior.

```python
import matplotlib.pyplot as plt

def plot_adadelta_behavior(rho_values, eps_values):
    iterations = 100
    x = np.linspace(-5, 5, 100)
    
    fig, axs = plt.subplots(len(rho_values), len(eps_values), figsize=(15, 15))
    
    for i, rho in enumerate(rho_values):
        for j, eps in enumerate(eps_values):
            optimizer = Adadelta([x], rho=rho, eps=eps)
            
            trajectory = []
            for _ in range(iterations):
                optimizer.step()
                trajectory.append(x.())
            
            axs[i, j].plot(range(iterations), trajectory)
            axs[i, j].set_title(f'rho={rho}, eps={eps}')
    
    plt.tight_layout()
    plt.show()

rho_values = [0.9, 0.95, 0.99]
eps_values = [1e-8, 1e-6, 1e-4]
plot_adadelta_behavior(rho_values, eps_values)
```

Slide 3: Implementing the Update Rule

The core of Adadelta lies in its update rule. We'll break down the implementation step by step, focusing on the mathematical operations involved.

```python
def adadelta_update(param, grad, E_g2, E_dx2, rho, eps):
    # Update running average of gradient squares
    E_g2 = rho * E_g2 + (1 - rho) * grad**2
    
    # Compute RMS of gradients and updates
    RMS_g = np.sqrt(E_g2 + eps)
    RMS_dx = np.sqrt(E_dx2 + eps)
    
    # Compute update
    dx = -(RMS_dx / RMS_g) * grad
    
    # Update running average of parameter updates
    E_dx2 = rho * E_dx2 + (1 - rho) * dx**2
    
    # Apply update
    param += dx
    
    return param, E_g2, E_dx2

# Example usage
param = np.array([1.0])
grad = np.array([0.1])
E_g2 = np.zeros_like(param)
E_dx2 = np.zeros_like(param)
rho = 0.95
eps = 1e-6

for _ in range(10):
    param, E_g2, E_dx2 = adadelta_update(param, grad, E_g2, E_dx2, rho, eps)
    print(f"Parameter: {param}, Gradient: {grad}")
```

Slide 4: Handling Multiple Parameters

In practice, we often need to optimize multiple parameters simultaneously. Let's extend our Adadelta implementation to handle this scenario.

```python
class MultiParamAdadelta:
    def __init__(self, params, rho=0.95, eps=1e-6):
        self.params = params
        self.rho = rho
        self.eps = eps
        self.E_g2 = {id(p): np.zeros_like(p) for p in self.params}
        self.E_dx2 = {id(p): np.zeros_like(p) for p in self.params}

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            param_id = id(param)
            
            # Update running average of gradient squares
            self.E_g2[param_id] = self.rho * self.E_g2[param_id] + (1 - self.rho) * grad**2
            
            # Compute RMS of gradients and updates
            RMS_g = np.sqrt(self.E_g2[param_id] + self.eps)
            RMS_dx = np.sqrt(self.E_dx2[param_id] + self.eps)
            
            # Compute update
            dx = -(RMS_dx / RMS_g) * grad
            
            # Update running average of parameter updates
            self.E_dx2[param_id] = self.rho * self.E_dx2[param_id] + (1 - self.rho) * dx**2
            
            # Apply update
            param += dx

# Example usage
params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
grads = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
optimizer = MultiParamAdadelta(params)

for _ in range(10):
    optimizer.step(grads)
    print(f"Parameters: {params}")
```

Slide 5: Comparing Adadelta with Other Optimizers

To understand the advantages of Adadelta, let's compare it with other popular optimizers like SGD and Adam on a simple optimization problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def optimize(optimizer, initial_params, num_iterations):
    params = initial_params.()
    trajectory = [params.()]
    
    for _ in range(num_iterations):
        grad = rosenbrock_grad(*params)
        optimizer.step([params], [grad])
        trajectory.append(params.())
    
    return np.array(trajectory)

# Initialize optimizers
sgd = lambda params: MultiParamAdadelta(params, rho=0)  # SGD is Adadelta with rho=0
adadelta = lambda params: MultiParamAdadelta(params)
adam = lambda params: MultiParamAdadelta(params, rho=0.9)  # Simplified Adam

initial_params = np.array([-1.0, 2.0])
num_iterations = 1000

trajectories = {
    'SGD': optimize(sgd([initial_params]), initial_params, num_iterations),
    'Adadelta': optimize(adadelta([initial_params]), initial_params, num_iterations),
    'Adam': optimize(adam([initial_params]), initial_params, num_iterations)
}

# Plot results
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(12, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
for name, traj in trajectories.items():
    plt.plot(traj[:, 0], traj[:, 1], label=name)
plt.legend()
plt.title('Optimizer Comparison on Rosenbrock Function')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 6: Implementing Adadelta with Momentum

Adadelta can be further improved by incorporating momentum. Let's implement this variant and compare its performance with the standard Adadelta.

```python
class AdadeltaWithMomentum:
    def __init__(self, params, rho=0.95, eps=1e-6, momentum=0.9):
        self.params = params
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.E_g2 = {id(p): np.zeros_like(p) for p in self.params}
        self.E_dx2 = {id(p): np.zeros_like(p) for p in self.params}
        self.v = {id(p): np.zeros_like(p) for p in self.params}

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            param_id = id(param)
            
            # Update running average of gradient squares
            self.E_g2[param_id] = self.rho * self.E_g2[param_id] + (1 - self.rho) * grad**2
            
            # Compute RMS of gradients and updates
            RMS_g = np.sqrt(self.E_g2[param_id] + self.eps)
            RMS_dx = np.sqrt(self.E_dx2[param_id] + self.eps)
            
            # Compute update
            dx = -(RMS_dx / RMS_g) * grad
            
            # Apply momentum
            self.v[param_id] = self.momentum * self.v[param_id] + dx
            
            # Update running average of parameter updates
            self.E_dx2[param_id] = self.rho * self.E_dx2[param_id] + (1 - self.rho) * self.v[param_id]**2
            
            # Apply update
            param += self.v[param_id]

# Compare standard Adadelta with Adadelta + Momentum
adadelta = lambda params: MultiParamAdadelta(params)
adadelta_momentum = lambda params: AdadeltaWithMomentum(params)

trajectories = {
    'Adadelta': optimize(adadelta([initial_params]), initial_params, num_iterations),
    'Adadelta + Momentum': optimize(adadelta_momentum([initial_params]), initial_params, num_iterations)
}

# Plot results (similar to previous slide)
```

Slide 7: Adaptive Learning Rate in Adadelta

One of Adadelta's key features is its adaptive learning rate. Let's visualize how the effective learning rate changes during optimization.

```python
class AdadeltaWithLearningRateTracking(MultiParamAdadelta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rates = []

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            param_id = id(param)
            
            self.E_g2[param_id] = self.rho * self.E_g2[param_id] + (1 - self.rho) * grad**2
            
            RMS_g = np.sqrt(self.E_g2[param_id] + self.eps)
            RMS_dx = np.sqrt(self.E_dx2[param_id] + self.eps)
            
            effective_lr = RMS_dx / RMS_g
            self.learning_rates.append(np.mean(effective_lr))
            
            dx = -effective_lr * grad
            
            self.E_dx2[param_id] = self.rho * self.E_dx2[param_id] + (1 - self.rho) * dx**2
            
            param += dx

# Optimize using Adadelta with learning rate tracking
adadelta_tracker = AdadeltaWithLearningRateTracking([initial_params])
trajectory = optimize(adadelta_tracker, initial_params, num_iterations)

# Plot learning rate over time
plt.figure(figsize=(10, 6))
plt.plot(adadelta_tracker.learning_rates)
plt.title('Adadelta Effective Learning Rate over Time')
plt.xlabel('Iteration')
plt.ylabel('Effective Learning Rate')
plt.yscale('log')
plt.show()
```

Slide 8: Adadelta for Neural Network Training

Let's apply our Adadelta implementation to train a simple neural network for binary classification.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dz2 = y_pred - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        return [dW1, db1, dW2, db2]

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)

# Initialize model and optimizer
model = SimpleNN(2, 5, 1)
optimizer = MultiParamAdadelta([model.W1, model.b1, model.W2, model.b2])

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_pred = model.forward(X)
    loss = binary_cross_entropy(y, y_pred)
    
    # Backward pass
    grads = model.backward(X, y, y_pred)
    
    # Update parameters
    optimizer.step(grads)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Evaluate model
y_pred_final = model.forward(X)
accuracy = np.mean((y_pred_final > 0.5) == y)
print(f"Final Accuracy: {accuracy}")
```

Slide 9: Visualizing Adadelta's Impact on Neural Network Training

To better understand how Adadelta affects neural network training, let's compare it with standard SGD and visualize the loss curves.

```python
import matplotlib.pyplot as plt

def train_model(model, optimizer, X, y, epochs):
    losses = []
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = binary_cross_entropy(y, y_pred)
        grads = model.backward(X, y, y_pred)
        optimizer.step(grads)
        losses.append(loss)
    return losses

# Initialize models and optimizers
model_sgd = SimpleNN(2, 5, 1)
model_adadelta = SimpleNN(2, 5, 1)

sgd = MultiParamAdadelta([model_sgd.W1, model_sgd.b1, model_sgd.W2, model_sgd.b2], rho=0)
adadelta = MultiParamAdadelta([model_adadelta.W1, model_adadelta.b1, model_adadelta.W2, model_adadelta.b2])

# Train models
epochs = 1000
losses_sgd = train_model(model_sgd, sgd, X, y, epochs)
losses_adadelta = train_model(model_adadelta, adadelta, X, y, epochs)

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(losses_sgd, label='SGD')
plt.plot(losses_adadelta, label='Adadelta')
plt.title('Training Loss: SGD vs Adadelta')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 10: Real-life Example: Image Classification

Let's apply our Adadelta implementation to a more practical scenario: image classification using a convolutional neural network (CNN) on the MNIST dataset.

```python
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0
y = y.astype('int')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleCNN:
    def __init__(self):
        self.conv1 = np.random.randn(32, 1, 3, 3) * 0.01
        self.conv2 = np.random.randn(64, 32, 3, 3) * 0.01
        self.fc1 = np.random.randn(64 * 5 * 5, 128) * 0.01
        self.fc2 = np.random.randn(128, 10) * 0.01
    
    def forward(self, X):
        # Implement forward pass (convolutions, pooling, fully connected layers)
        pass
    
    def backward(self, X, y, y_pred):
        # Implement backward pass
        pass

# Initialize model and optimizer
model = SimpleCNN()
optimizer = MultiParamAdadelta([model.conv1, model.conv2, model.fc1, model.fc2])

# Training loop (pseudocode)
for epoch in range(num_epochs):
    for batch in get_batches(X_train, y_train):
        y_pred = model.forward(batch)
        loss = compute_loss(y_pred, batch_labels)
        grads = model.backward(batch, batch_labels, y_pred)
        optimizer.step(grads)
    
    # Evaluate on validation set
    val_accuracy = evaluate(model, X_val, y_val)
    print(f"Epoch {epoch}, Validation Accuracy: {val_accuracy}")

# Final evaluation on test set
test_accuracy = evaluate(model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
```

Slide 11: Adadelta vs. Other Adaptive Optimizers

Let's compare Adadelta with other popular adaptive optimizers like Adam and RMSprop on a challenging optimization problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

def rastrigin_grad(x, y):
    dx = 2*x + 20*np.pi*np.sin(2*np.pi*x)
    dy = 2*y + 20*np.pi*np.sin(2*np.pi*y)
    return np.array([dx, dy])

class Optimizer:
    def __init__(self, params):
        self.params = params
    
    def step(self, grads):
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
    
    def step(self, grads):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Implement RMSprop and Adadelta classes similarly

def optimize(optimizer_class, initial_params, num_iterations):
    params = initial_params.()
    optimizer = optimizer_class([params])
    trajectory = [params.()]
    
    for _ in range(num_iterations):
        grad = rastrigin_grad(*params)
        optimizer.step([grad])
        trajectory.append(params.())
    
    return np.array(trajectory)

# Run optimization with different optimizers
initial_params = np.array([4.0, 4.0])
num_iterations = 1000

trajectories = {
    'Adam': optimize(Adam, initial_params, num_iterations),
    'RMSprop': optimize(RMSprop, initial_params, num_iterations),
    'Adadelta': optimize(Adadelta, initial_params, num_iterations)
}

# Plot results
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)

plt.figure(figsize=(12, 8))
plt.contour(X, Y, Z, levels=np.logspace(0, 3, 20))
for name, traj in trajectories.items():
    plt.plot(traj[:, 0], traj[:, 1], label=name)
plt.legend()
plt.title('Optimizer Comparison on Rastrigin Function')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 12: Adadelta in Natural Language Processing

Adadelta can be effectively used in natural language processing tasks. Let's implement a simple word embedding model using Adadelta.

```python
import numpy as np
from collections import defaultdict

class WordEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def forward(self, word_indices):
        return self.embeddings[word_indices]
    
    def backward(self, word_indices, grad):
        self.embeddings[word_indices] -= grad

def skipgram(center_word, context_words, negative_samples, embeddings):
    # Implement skip-gram model with negative sampling
    pass

# Prepare data
corpus = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
word_to_idx = {word: i for i, word in enumerate(set(corpus))}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# Initialize model and optimizer
vocab_size = len(word_to_idx)
embedding_dim = 50
model = WordEmbedding(vocab_size, embedding_dim)
optimizer = Adadelta([model.embeddings])

# Training loop (pseudocode)
for epoch in range(num_epochs):
    for i in range(len(corpus)):
        center_word = word_to_idx[corpus[i]]
        context_words = [word_to_idx[corpus[j]] for j in range(max(0, i-2), min(len(corpus), i+3)) if j != i]
        negative_samples = sample_negative_words(vocab_size, 5)
        
        loss, grads = skipgram(center_word, context_words, negative_samples, model)
        optimizer.step([grads])
    
    print(f"Epoch {epoch}, Loss: {loss}")

# Visualize word embeddings (e.g., using t-SNE)
```

Slide 13: Adadelta for Reinforcement Learning

Adadelta can also be applied to reinforcement learning tasks. Let's implement a simple Q-learning agent using Adadelta for policy optimization.

```python
import numpy as np
import gym

class QNetwork:
    def __init__(self, state_dim, action_dim):
        self.W = np.random.randn(state_dim, action_dim) * 0.01
        self.b = np.zeros(action_dim)
    
    def forward(self, state):
        return np.dot(state, self.W) + self.b
    
    def backward(self, state, action, td_error):
        dW = np.outer(state, td_error)
        db = td_error
        return dW, db

class AdadeltaAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=0.1):
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = Adadelta([self.q_network.W, self.q_network.b])
        self.gamma = gamma
        self.epsilon = epsilon
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_network.b.shape[0])
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_network.forward(state)[action]
        next_q = np.max(self.q_network.forward(next_state))
        target_q = reward + self.gamma * next_q * (1 - done)
        td_error = target_q - current_q
        
        dW, db = self.q_network.backward(state, action, td_error)
        self.optimizer.step([dW, db])

# Training loop (pseudocode)
env = gym.make('CartPole-v1')
agent = AdadeltaAgent(4, 2)  # 4 state dimensions, 2 actions for CartPole

for episode in range(1000):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode {episode}, Total Reward: {total_reward}")
```

Slide 14: Conclusion and Future Directions

Adadelta has proven to be a versatile optimizer across various domains of machine learning, including neural networks, natural language processing, and reinforcement learning. Its adaptive learning rate and momentum-like behavior make it particularly suitable for tasks with sparse gradients or when the optimal learning rate is difficult to determine.

Key advantages of Adadelta:

1. No manual learning rate tuning required
2. Robustness to different problem types and architectures
3. Ability to handle sparse gradients effectively

Future research directions for Adadelta and adaptive optimizers:

1. Theoretical analysis of convergence properties
2. Hybrid approaches combining Adadelta with other optimization techniques
3. Application to emerging areas such as meta-learning and federated learning

```python
# Pseudocode for a potential future direction: Adadelta with layer-wise adaptivity

class LayerwiseAdadelta:
    def __init__(self, params, rho=0.95, eps=1e-6):
        self.params = params
        self.rho = rho
        self.eps = eps
        self.E_g2 = {id(p): np.zeros_like(p) for p in self.params}
        self.E_dx2 = {id(p): np.zeros_like(p) for p in self.params}
        self.layer_lr = {id(p): 1.0 for p in self.params}

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            param_id = id(param)
            
            # Update running averages
            self.E_g2[param_id] = self.rho * self.E_g2[param_id] + (1 - self.rho) * grad**2
            RMS_g = np.sqrt(self.E_g2[param_id] + self.eps)
            RMS_dx = np.sqrt(self.E_dx2[param_id] + self.eps)
            
            # Compute update
            dx = -(RMS_dx / RMS_g) * grad
            
            # Apply layer-wise learning rate
            dx *= self.layer_lr[param_id]
            
            # Update parameter
            param += dx
            
            # Update E_dx2
            self.E_dx2[param_id] = self.rho * self.E_dx2[param_id] + (1 - self.rho) * dx**2
            
            # Adjust layer-wise learning rate based on gradient statistics
            self.layer_lr[param_id] *= np.mean(RMS_dx / RMS_g)

# Usage example
optimizer = LayerwiseAdadelta(model.parameters())
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        grads = compute_gradients(loss)
        optimizer.step(grads)
```

Slide 15: Additional Resources

For those interested in diving deeper into Adadelta and adaptive optimization methods, here are some valuable resources:

1. Original Adadelta paper: Zeiler, M. D. (2012). "ADADELTA: An Adaptive Learning Rate Method" arXiv:1212.5701 \[cs.LG\] [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
2. Comparative study of adaptive optimizers: Ruder, S. (2016). "An overview of gradient descent optimization algorithms" arXiv:1609.04747 \[cs.LG\] [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. Theoretical analysis of adaptive methods: Wilson, A. C., et al. (2017). "The Marginal Value of Adaptive Gradient Methods in Machine Learning" arXiv:1705.08292 \[stat.ML\] [https://arxiv.org/abs/1705.08292](https://arxiv.org/abs/1705.08292)
4. Adaptive methods in deep learning: Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" MIT Press [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/)
5. Implementation guides and tutorials:
   * PyTorch documentation on optimizers
   * TensorFlow documentation on optimizers
   * Keras documentation on optimizers

These resources provide a comprehensive understanding of Adadelta and its place in the broader context of optimization algorithms for machine learning.

