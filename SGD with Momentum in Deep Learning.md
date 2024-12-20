## SGD with Momentum in Deep Learning
Slide 1: Introduction to SGD with Momentum

Stochastic Gradient Descent (SGD) with Momentum is an optimization algorithm used in deep learning to accelerate convergence and reduce oscillations during training. It introduces a velocity term that accumulates past gradients, allowing the optimizer to move faster in consistent directions and dampen oscillations in inconsistent directions.

```python
import numpy as np
import matplotlib.pyplot as plt

def sgd_momentum(gradient, x, v, learning_rate, momentum):
    v = momentum * v - learning_rate * gradient(x)
    x = x + v
    return x, v

# Example function to optimize
def f(x):
    return x**2

def gradient(x):
    return 2*x

x = np.linspace(-10, 10, 100)
plt.plot(x, f(x))
plt.title("Function to optimize: f(x) = x^2")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

Slide 2: The Momentum Algorithm

The momentum algorithm maintains a velocity vector that accumulates the gradients of past steps. This vector is used to update the parameters, allowing the optimizer to gain "momentum" in directions of consistent gradient descent.

```python
def momentum_update(params, velocity, grads, learning_rate, momentum):
    for param, vel, grad in zip(params, velocity, grads):
        vel = momentum * vel - learning_rate * grad
        param += vel
    return params, velocity

# Example usage
params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
velocity = [np.zeros_like(param) for param in params]
grads = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

learning_rate = 0.01
momentum = 0.9

new_params, new_velocity = momentum_update(params, velocity, grads, learning_rate, momentum)
print("Updated params:", new_params)
print("Updated velocity:", new_velocity)
```

Slide 3: Advantages of SGD with Momentum

SGD with Momentum offers several benefits over vanilla SGD. It accelerates convergence, especially in scenarios with high curvature or small but consistent gradients. The algorithm also helps in reducing oscillations in the optimization process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create model and optimizer
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop (simplified)
for epoch in range(100):
    # Forward pass
    output = model(torch.randn(1, 10))
    loss = output.sum()

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final model parameters:", list(model.parameters()))
```

Slide 4: Hyperparameters in SGD with Momentum

Two key hyperparameters in SGD with Momentum are the learning rate and the momentum coefficient. The learning rate controls the step size of parameter updates, while the momentum coefficient determines the contribution of past gradients.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x)

def train_model(lr, momentum):
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(torch.randn(10, 5))
        loss = criterion(output, torch.randn(10, 1))
        loss.backward()
        optimizer.step()

    return loss.item()

# Compare different hyperparameters
lrs = [0.01, 0.1, 1.0]
momentums = [0.0, 0.5, 0.9]

for lr in lrs:
    for momentum in momentums:
        final_loss = train_model(lr, momentum)
        print(f"LR: {lr}, Momentum: {momentum}, Final Loss: {final_loss:.4f}")
```

Slide 5: Implementing SGD with Momentum from Scratch

To gain a deeper understanding, let's implement SGD with Momentum from scratch using NumPy. This implementation will help illustrate the core concepts behind the algorithm.

```python
import numpy as np

class SGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            param += self.velocity[i]

        return params

# Example usage
params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
grads = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9)
updated_params = optimizer.update(params, grads)

print("Updated params:", updated_params)
```

Slide 6: Visualizing SGD with Momentum

To better understand how SGD with Momentum works, let's visualize its behavior on a 2D contour plot. We'll compare it with vanilla SGD to see the difference in optimization paths.

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def gradient(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(12, 5))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
plt.colorbar(label='Rosenbrock function value')

# SGD
x_sgd, y_sgd = 1.5, 1.5
path_sgd = [[x_sgd, y_sgd]]
lr = 0.001

for _ in range(100):
    grad = gradient(x_sgd, y_sgd)
    x_sgd -= lr * grad[0]
    y_sgd -= lr * grad[1]
    path_sgd.append([x_sgd, y_sgd])

# SGD with Momentum
x_mom, y_mom = 1.5, 1.5
path_mom = [[x_mom, y_mom]]
v_x, v_y = 0, 0
momentum = 0.9

for _ in range(100):
    grad = gradient(x_mom, y_mom)
    v_x = momentum * v_x - lr * grad[0]
    v_y = momentum * v_y - lr * grad[1]
    x_mom += v_x
    y_mom += v_y
    path_mom.append([x_mom, y_mom])

plt.plot(*zip(*path_sgd), 'r-', label='SGD')
plt.plot(*zip(*path_mom), 'g-', label='SGD with Momentum')
plt.legend()
plt.title('Optimization Path: SGD vs SGD with Momentum')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Slide 7: Real-Life Example: Image Classification

Let's apply SGD with Momentum to a real-life example of image classification using a convolutional neural network (CNN) on the CIFAR-10 dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

# Load and preprocess CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')
```

Slide 8: Comparing SGD with Momentum to Other Optimizers

Let's compare the performance of SGD with Momentum to other popular optimizers like Adam and RMSprop on a simple regression task.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data
X = torch.linspace(-10, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + torch.randn(X.shape) * 2

# Define model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(optimizer_class, **kwargs):
    model = LinearRegression()
    optimizer = optimizer_class(model.parameters(), **kwargs)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# Train with different optimizers
sgd_losses = train_model(optim.SGD, lr=0.01, momentum=0.9)
adam_losses = train_model(optim.Adam, lr=0.01)
rmsprop_losses = train_model(optim.RMSprop, lr=0.01)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sgd_losses, label='SGD with Momentum')
plt.plot(adam_losses, label='Adam')
plt.plot(rmsprop_losses, label='RMSprop')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.show()
```

Slide 9: Adaptive Learning Rate in SGD with Momentum

While SGD with Momentum uses a fixed learning rate, we can implement an adaptive learning rate to further improve convergence. Let's modify our SGD with Momentum implementation to include a simple learning rate decay.

```python
import numpy as np
import matplotlib.pyplot as plt

class AdaptiveSGDMomentum:
    def __init__(self, initial_lr=0.01, momentum=0.9, decay_rate=0.95):
        self.lr = initial_lr
        self.momentum = momentum
        self.decay_rate = decay_rate
        self.velocity = None
        self.t = 0

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]

        self.t += 1
        self.lr *= self.decay_rate

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
            param += self.velocity[i]

        return params

# Example usage
def quadratic(x):
    return x**2

def gradient(x):
    return 2*x

x = np.linspace(-5, 5, 100)
y = quadratic(x)

optimizer = AdaptiveSGDMomentum(initial_lr=0.1, momentum=0.9, decay_rate=0.99)
current_x = 4.0
xs = [current_x]

for _ in range(50):
    grad = gradient(current_x)
    current_x = optimizer.update([current_x], [grad])[0]
    xs.append(current_x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = x^2')
plt.plot(xs, quadratic(np.array(xs)), 'ro-', label='Optimization path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('SGD with Momentum and Adaptive Learning Rate')
plt.legend()
plt.show()
```

Slide 10: Handling Sparse Gradients

In some cases, such as natural language processing tasks, we may encounter sparse gradients. Let's modify our SGD with Momentum implementation to handle sparse updates more efficiently.

```python
import numpy as np

class SparseSGDMomentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for param_id, (param, grad) in enumerate(zip(params, grads)):
            if param_id not in self.velocity:
                self.velocity[param_id] = np.zeros_like(param)

            mask = grad != 0
            self.velocity[param_id][mask] = (
                self.momentum * self.velocity[param_id][mask] -
                self.learning_rate * grad[mask]
            )
            param[mask] += self.velocity[param_id][mask]

        return params

# Example usage
params = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
grads = [np.array([0.1, 0.0, 0.3, 0.0, 0.5])]

optimizer = SparseSGDMomentum()
updated_params = optimizer.update(params, grads)

print("Updated params:", updated_params)
```

Slide 11: Real-Life Example: Natural Language Processing

Let's apply SGD with Momentum to a simple sentiment analysis task using a recurrent neural network (RNN) on movie reviews.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Define the RNN model
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return torch.sigmoid(output)

# Prepare data
tokenizer = get_tokenizer("basic_english")
train_iter = IMDB(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Initialize model and optimizer
model = SentimentRNN(len(vocab), embedding_dim=100, hidden_dim=64)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCELoss()

# Training loop (simplified)
for epoch in range(5):
    for label, text in train_iter:
        optimizer.zero_grad()
        tokenized = torch.tensor([vocab[token] for token in tokenizer(text)])
        output = model(tokenized.unsqueeze(0))
        loss = criterion(output, torch.tensor([[float(label)]]))
        loss.backward()
        optimizer.step()

print("Training completed")
```

Slide 12: Nesterov Accelerated Gradient (NAG)

Nesterov Accelerated Gradient is a variation of SGD with Momentum that provides even faster convergence in some cases. Let's implement NAG and compare it with standard SGD with Momentum.

```python
import numpy as np
import matplotlib.pyplot as plt

def sgd_momentum(gradient, x, v, learning_rate, momentum):
    v = momentum * v - learning_rate * gradient(x)
    x = x + v
    return x, v

def nesterov_momentum(gradient, x, v, learning_rate, momentum):
    v_prev = v
    v = momentum * v - learning_rate * gradient(x + momentum * v)
    x = x + v + momentum * (v - v_prev)
    return x, v

def quadratic(x):
    return x**2

def gradient(x):
    return 2*x

x = np.linspace(-5, 5, 100)
y = quadratic(x)

x_sgd, v_sgd = 4.0, 0.0
x_nag, v_nag = 4.0, 0.0
xs_sgd, xs_nag = [x_sgd], [x_nag]

for _ in range(20):
    x_sgd, v_sgd = sgd_momentum(gradient, x_sgd, v_sgd, 0.1, 0.9)
    x_nag, v_nag = nesterov_momentum(gradient, x_nag, v_nag, 0.1, 0.9)
    xs_sgd.append(x_sgd)
    xs_nag.append(x_nag)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='f(x) = x^2')
plt.plot(xs_sgd, quadratic(np.array(xs_sgd)), 'ro-', label='SGD with Momentum')
plt.plot(xs_nag, quadratic(np.array(xs_nag)), 'go-', label='Nesterov Momentum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('SGD with Momentum vs Nesterov Momentum')
plt.legend()
plt.show()
```

Slide 13: Practical Tips for Using SGD with Momentum

When using SGD with Momentum in practice, consider these tips:

1. Start with a small learning rate (e.g., 0.01) and adjust as needed.
2. Use a momentum value between 0.9 and 0.99.
3. Monitor the loss during training to detect convergence issues.
4. Consider learning rate schedules for better convergence.
5. Experiment with Nesterov Momentum for potentially faster convergence.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
for epoch in range(100):
    # Forward pass and loss calculation
    output = model(torch.randn(1, 10))
    loss = output.sum()

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update learning rate
    scheduler.step()

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.4f}")

print("Training completed")
```

Slide 14: Additional Resources

For further reading on SGD with Momentum and related optimization techniques, consider these resources:

1. "On the importance of initialization and momentum in deep learning" by Sutskever et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014) ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. "An overview of gradient descent optimization algorithms" by Sebastian Ruder (2016) ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)

These papers provide in-depth discussions on optimization algorithms in deep learning, including SGD with Momentum and its variants.

