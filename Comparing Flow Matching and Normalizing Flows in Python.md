## Comparing Flow Matching and Normalizing Flows in Python
Slide 1: Introduction to Flow Matching and Normalizing Flows

Flow Matching and Normalizing Flows are two powerful techniques in the field of generative modeling. Both methods aim to transform simple probability distributions into more complex ones, enabling the generation of diverse and realistic data. This slideshow explores the common ground between these approaches, highlighting their shared principles and applications.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple demonstration of transforming a distribution
np.random.seed(42)
simple_dist = np.random.normal(0, 1, 1000)
complex_dist = np.exp(simple_dist) + np.sin(simple_dist)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(simple_dist, bins=30, density=True)
plt.title("Simple Distribution")
plt.subplot(122)
plt.hist(complex_dist, bins=30, density=True)
plt.title("Transformed Distribution")
plt.tight_layout()
plt.show()
```

Slide 2: Bijective Transformations

Both Flow Matching and Normalizing Flows rely on bijective transformations. These are invertible functions that map each point in the input space to a unique point in the output space, and vice versa. This property allows for efficient sampling and density estimation.

```python
import torch
import torch.nn as nn

class BijectiveTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * torch.exp(self.weight) + self.bias

    def inverse(self, y):
        return (y - self.bias) * torch.exp(-self.weight)

# Example usage
transform = BijectiveTransform()
x = torch.randn(5)
y = transform(x)
x_recovered = transform.inverse(y)
print(f"Original x: {x}")
print(f"Transformed y: {y}")
print(f"Recovered x: {x_recovered}")
```

Slide 3: Change of Variables Formula

The change of variables formula is a fundamental concept shared by Flow Matching and Normalizing Flows. It allows us to compute the probability density of the transformed distribution given the density of the original distribution and the Jacobian of the transformation.

```python
import torch

def change_of_variables(x, y, transform):
    # Compute the Jacobian
    jac = torch.autograd.functional.jacobian(transform, x)
    log_det_jac = torch.log(torch.abs(jac))
    
    # Assuming standard normal as base distribution
    log_prob_x = -0.5 * x**2 - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    
    # Change of variables formula
    log_prob_y = log_prob_x - log_det_jac
    
    return log_prob_y

# Example usage
x = torch.randn(1, requires_grad=True)
transform = lambda x: torch.exp(x)
y = transform(x)
log_prob_y = change_of_variables(x, y, transform)
print(f"Log probability of y: {log_prob_y.item()}")
```

Slide 4: Compositional Nature

Both Flow Matching and Normalizing Flows benefit from their compositional nature. Multiple simple transformations can be combined to create more expressive models, allowing for the representation of complex distributions.

```python
import torch
import torch.nn as nn

class ComposedFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        log_det_sum = 0
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_sum += log_det
        return x, log_det_sum

    def inverse(self, y):
        for flow in reversed(self.flows):
            y = flow.inverse(y)
        return y

# Example usage
class SimpleFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        y = x * torch.exp(self.a) + self.b
        log_det = self.a
        return y, log_det

    def inverse(self, y):
        return (y - self.b) * torch.exp(-self.a)

flow = ComposedFlow([SimpleFlow() for _ in range(3)])
x = torch.randn(5)
y, log_det = flow(x)
x_recovered = flow.inverse(y)
print(f"Original x: {x}")
print(f"Transformed y: {y}")
print(f"Log determinant: {log_det}")
print(f"Recovered x: {x_recovered}")
```

Slide 5: Continuous-Time Formulation

Both Flow Matching and Normalizing Flows can be formulated in continuous time, leading to more flexible and expressive models. This perspective allows for the use of ordinary differential equations (ODEs) to describe the transformation process.

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ContinuousTimeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t, x):
        return self.net(x)

# Example usage
flow = ContinuousTimeFlow()
x0 = torch.randn(10, 1)
t = torch.linspace(0, 1, 100)
x = odeint(flow, x0, t)

plt.figure(figsize=(10, 5))
plt.plot(t, x.squeeze().T)
plt.title("Continuous-Time Flow Trajectories")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```

Slide 6: Training Objectives

Flow Matching and Normalizing Flows share similar training objectives. Both methods aim to maximize the likelihood of the observed data under the transformed distribution. This is typically achieved by minimizing the negative log-likelihood.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        y = x * torch.exp(self.a) + self.b
        log_det = self.a
        return y, log_det

def train_flow(flow, data, num_epochs=1000):
    optimizer = optim.Adam(flow.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y, log_det = flow(data)
        
        # Assuming standard normal as base distribution
        log_prob = -0.5 * data**2 - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        loss = -torch.mean(log_prob - log_det)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Example usage
data = torch.randn(1000, 1)
flow = SimpleFlow()
train_flow(flow, data)
```

Slide 7: Invertibility and Sampling

A key advantage shared by Flow Matching and Normalizing Flows is their invertibility, which allows for efficient sampling from the learned distribution. This property makes these models particularly useful for generating new data points.

```python
import torch
import torch.nn as nn

class InvertibleFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * torch.exp(self.a) + self.b

    def inverse(self, y):
        return (y - self.b) * torch.exp(-self.a)

# Example usage
flow = InvertibleFlow()
x = torch.randn(1000)
y = flow(x)

# Sampling
z = torch.randn(1000)
samples = flow.inverse(z)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(y.detach().numpy(), bins=30, density=True)
plt.title("Transformed Distribution")
plt.subplot(122)
plt.hist(samples.detach().numpy(), bins=30, density=True)
plt.title("Sampled Distribution")
plt.tight_layout()
plt.show()
```

Slide 8: Handling Multi-Modal Distributions

Both Flow Matching and Normalizing Flows excel at modeling multi-modal distributions, which are common in real-world data. By learning complex transformations, these methods can capture multiple modes in the data distribution.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MultiModalFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        y = x + self.net(x)
        log_det = torch.log(torch.abs(1 + self.net(x).diff()))
        return y, log_det

# Generate multi-modal data
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-2, 0.5, 1000),
    np.random.normal(2, 0.5, 1000)
])

# Train the flow
flow = MultiModalFlow()
optimizer = optim.Adam(flow.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    x = torch.FloatTensor(data).unsqueeze(1)
    y, log_det = flow(x)
    
    log_prob = -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    loss = -torch.mean(log_prob - log_det)
    
    loss.backward()
    optimizer.step()

# Visualize results
x = torch.linspace(-5, 5, 1000).unsqueeze(1)
y, _ = flow(x)

plt.figure(figsize=(10, 5))
plt.hist(data, bins=50, density=True, alpha=0.5, label="Original")
plt.hist(y.detach().numpy(), bins=50, density=True, alpha=0.5, label="Transformed")
plt.legend()
plt.title("Multi-Modal Distribution Modeling")
plt.show()
```

Slide 9: Latent Space Manipulation

Flow Matching and Normalizing Flows allow for meaningful latent space manipulation. By transforming data into a structured latent space, we can perform operations like interpolation and attribute manipulation.

```python
import torch
import torch.nn as nn

class LatentFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return x + self.net(x)

    def inverse(self, y):
        x = y
        for _ in range(10):  # Fixed-point iteration
            x = y - self.net(x)
        return x

# Example usage
flow = LatentFlow()
x1 = torch.tensor([1.0, 0.0])
x2 = torch.tensor([0.0, 1.0])

# Interpolation in latent space
alphas = torch.linspace(0, 1, 10)
interpolations = []

for alpha in alphas:
    z = alpha * x1 + (1 - alpha) * x2
    interpolations.append(flow.inverse(z))

interpolations = torch.stack(interpolations)

plt.figure(figsize=(10, 5))
plt.scatter(interpolations[:, 0], interpolations[:, 1])
plt.plot(interpolations[:, 0], interpolations[:, 1])
plt.title("Latent Space Interpolation")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

Slide 10: Real-Life Example: Image Generation

Flow Matching and Normalizing Flows can be applied to image generation tasks. By learning a bijective mapping between a simple distribution and the complex distribution of natural images, we can generate diverse and realistic images.

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

class ImageFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)

    def inverse(self, y):
        x = y
        for _ in range(10):  # Fixed-point iteration
            x = y - self.net(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist = MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

# Train the flow (simplified for demonstration)
flow = ImageFlow()
optimizer = torch.optim.Adam(flow.parameters(), lr=0.001)

for epoch in range(5):  # Train for 5 epochs
    for batch, _ in dataloader:
        optimizer.zero_grad()
        y, _ = flow(batch)
        loss = torch.mean(y**2)  # Simplified loss
        loss.backward()
        optimizer.step()

# Generate samples
z = torch.randn(16, 1, 28, 28)
samples = flow.inverse(z)

# Visualize samples
grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.title("Generated MNIST Samples")
plt.axis('off')
plt.show()
```

Slide 11: Real-Life Example: Density Estimation

Flow Matching and Normalizing Flows excel at density estimation tasks. They can learn complex probability distributions from data, allowing for accurate modeling of various phenomena in fields like physics, biology, and social sciences.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons

class DensityFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        y = x + self.net(x)
        return y

# Generate moon dataset
X, _ = make_moons(n_samples=1000, noise=0.05, random_state=42)
data = torch.FloatTensor(
```

## Response:
Slide 11: Real-Life Example: Density Estimation

Flow Matching and Normalizing Flows excel at density estimation tasks. They can learn complex probability distributions from data, allowing for accurate modeling of various phenomena in fields like physics, biology, and social sciences.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons

class DensityFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return x + self.net(x)

# Generate moon dataset
X, _ = make_moons(n_samples=1000, noise=0.05, random_state=42)
data = torch.FloatTensor(X)

# Train the flow
flow = DensityFlow()
optimizer = optim.Adam(flow.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    y = flow(data)
    log_det = torch.log(torch.abs(torch.det(torch.eye(2) + torch.autograd.functional.jacobian(flow, data))))
    loss = torch.mean(0.5 * torch.sum(y**2, dim=1) - log_det)
    loss.backward()
    optimizer.step()

# Visualize the learned density
x = torch.linspace(-3, 3, 100)
y = torch.linspace(-3, 3, 100)
xx, yy = torch.meshgrid(x, y)
xy = torch.stack([xx.flatten(), yy.flatten()], dim=1)

with torch.no_grad():
    log_prob = -0.5 * torch.sum(flow(xy)**2, dim=1)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, log_prob.reshape(100, 100), levels=50)
plt.scatter(data[:, 0], data[:, 1], color='red', alpha=0.5)
plt.title("Learned Density Estimation")
plt.show()
```

Slide 12: Handling High-Dimensional Data

Both Flow Matching and Normalizing Flows are capable of handling high-dimensional data efficiently. This makes them suitable for tasks involving complex data types such as images, audio, or high-dimensional scientific measurements.

```python
import torch
import torch.nn as nn

class HighDimFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(3)
        ])

    def forward(self, x):
        log_det_sum = 0
        for layer in self.layers:
            x = x + torch.tanh(layer(x))
            log_det_sum += torch.log(torch.abs(1 + torch.sum(layer.weight * (1 - torch.tanh(layer(x))**2), dim=1)))
        return x, log_det_sum

# Example usage
dim = 100
flow = HighDimFlow(dim)
x = torch.randn(1000, dim)
y, log_det = flow(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Log determinant shape: {log_det.shape}")
```

Slide 13: Conditional Flows

Flow Matching and Normalizing Flows can be extended to conditional settings, allowing the generation or transformation of data based on additional input conditions. This is particularly useful for tasks like style transfer or domain adaptation.

```python
import torch
import torch.nn as nn

class ConditionalFlow(nn.Module):
    def __init__(self, data_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, data_dim)
        )

    def forward(self, x, cond):
        input_concat = torch.cat([x, cond], dim=1)
        return x + self.net(input_concat)

# Example usage
data_dim, cond_dim = 10, 5
flow = ConditionalFlow(data_dim, cond_dim)

x = torch.randn(100, data_dim)
cond = torch.randn(100, cond_dim)

y = flow(x, cond)

print(f"Input shape: {x.shape}")
print(f"Condition shape: {cond.shape}")
print(f"Output shape: {y.shape}")
```

Slide 14: Regularization Techniques

Both Flow Matching and Normalizing Flows benefit from various regularization techniques to improve training stability and generalization. These may include weight decay, spectral normalization, or custom regularization terms in the loss function.

```python
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class RegularizedFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(dim, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, dim))
        )

    def forward(self, x):
        return x + self.net(x)

# Training loop with regularization
def train_flow(flow, data, num_epochs=1000):
    optimizer = torch.optim.Adam(flow.parameters(), lr=0.001, weight_decay=1e-5)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y = flow(data)
        
        # Negative log-likelihood loss
        nll_loss = 0.5 * torch.sum(y**2, dim=1).mean()
        
        # Jacobian regularization
        jac = torch.autograd.functional.jacobian(flow, data)
        jac_frob_norm = torch.norm(jac, p='fro')
        reg_loss = 1e-4 * jac_frob_norm
        
        loss = nll_loss + reg_loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Example usage
dim = 10
flow = RegularizedFlow(dim)
data = torch.randn(1000, dim)
train_flow(flow, data)
```

Slide 15: Additional Resources

For those interested in diving deeper into Flow Matching and Normalizing Flows, here are some valuable resources:

1. "Normalizing Flows for Probabilistic Modeling and Inference" by Papamakarios et al. (2019) ArXiv: [https://arxiv.org/abs/1912.02762](https://arxiv.org/abs/1912.02762)
2. "Flow Matching for Generative Modeling" by Lipman et al. (2022) ArXiv: [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
3. "Continuous Normalizing Flows" by Chen et al. (2018) ArXiv: [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
4. "Neural Ordinary Differential Equations" by Chen et al. (2018) ArXiv: [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)

These papers provide in-depth explanations of the theory behind these methods and their various applications in machine learning and statistical modeling.

