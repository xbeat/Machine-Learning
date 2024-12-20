## Introducing Adaptive Neural Networks with Stochastic Synapses in Python
Slide 1: Introduction to Adaptive Neural Networks with Stochastic Synapses

Adaptive Neural Networks with Stochastic Synapses are an advanced form of artificial neural networks that incorporate randomness into their synaptic connections. This approach aims to improve generalization and robustness in machine learning models. In this presentation, we'll explore the concept, implementation, and applications of these networks using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def stochastic_synapse(input_signal, noise_level=0.1):
    noise = np.random.normal(0, noise_level, input_signal.shape)
    return input_signal + noise

# Example usage
input_signal = np.linspace(0, 10, 100)
noisy_signal = stochastic_synapse(input_signal)

plt.plot(input_signal, label='Original')
plt.plot(noisy_signal, label='Stochastic')
plt.legend()
plt.title('Stochastic Synapse Effect')
plt.show()
```

Slide 2: Stochastic Synapses: The Basics

Stochastic synapses introduce controlled randomness into the neural network's connections. This randomness can help prevent overfitting and improve the network's ability to generalize. The key idea is to add noise to the synaptic weights during training and potentially during inference.

```python
import torch
import torch.nn as nn

class StochasticLinear(nn.Module):
    def __init__(self, in_features, out_features, noise_level=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.noise_level = noise_level

    def forward(self, x):
        weight_noise = torch.randn_like(self.linear.weight) * self.noise_level
        noisy_weight = self.linear.weight + weight_noise
        return nn.functional.linear(x, noisy_weight, self.linear.bias)

# Example usage
layer = StochasticLinear(10, 5)
input_tensor = torch.randn(1, 10)
output = layer(input_tensor)
print(output)
```

Slide 3: Implementing a Simple Adaptive Neural Network

Let's implement a basic adaptive neural network with stochastic synapses using PyTorch. This network will adapt its architecture based on the input data.

```python
import torch
import torch.nn as nn

class AdaptiveStochasticNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, noise_level=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.noise_level = noise_level
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(StochasticLinear(prev_size, size, noise_level))
            self.layers.append(nn.ReLU())
            prev_size = size
        self.layers.append(StochasticLinear(prev_size, output_size, noise_level))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
model = AdaptiveStochasticNN(input_size=10, hidden_sizes=[20, 15], output_size=5)
input_tensor = torch.randn(1, 10)
output = model(input_tensor)
print(output)
```

Slide 4: Training the Adaptive Neural Network

Now that we have our adaptive neural network with stochastic synapses, let's train it on a simple dataset. We'll use the MNIST dataset for this example.

```python
import torch
import torch.optim as optim
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model and optimizer
model = AdaptiveStochasticNN(input_size=784, hidden_sizes=[128, 64], output_size=10)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

print("Training completed!")
```

Slide 5: Evaluating the Adaptive Neural Network

After training our adaptive neural network with stochastic synapses, it's important to evaluate its performance. We'll use the MNIST test dataset for this purpose.

```python
import torch
from torchvision import datasets, transforms

# Load MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 784)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Visualize some predictions
import matplotlib.pyplot as plt

def plot_predictions(model, data, target, num_samples=5):
    model.eval()
    with torch.no_grad():
        data = data.view(-1, 784)
        outputs = model(data[:num_samples])
        _, predicted = torch.max(outputs.data, 1)
    
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axs[i].imshow(data[i].view(28, 28), cmap='gray')
        axs[i].set_title(f'Pred: {predicted[i]}, True: {target[i]}')
        axs[i].axis('off')
    plt.show()

# Get a batch of test data
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Plot predictions
plot_predictions(model, images, labels)
```

Slide 6: Adaptive Architecture

One of the key features of adaptive neural networks is their ability to modify their architecture based on the input data or task requirements. Let's implement a simple adaptive mechanism that adds or removes hidden layers based on the model's performance.

```python
class AdaptiveStochasticNN(nn.Module):
    def __init__(self, input_size, initial_hidden_sizes, output_size, noise_level=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = initial_hidden_sizes
        self.output_size = output_size
        self.noise_level = noise_level
        self.build_network()
    
    def build_network(self):
        self.layers = nn.ModuleList()
        prev_size = self.input_size
        for size in self.hidden_sizes:
            self.layers.append(StochasticLinear(prev_size, size, self.noise_level))
            self.layers.append(nn.ReLU())
            prev_size = size
        self.layers.append(StochasticLinear(prev_size, self.output_size, self.noise_level))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def adapt_architecture(self, performance, threshold=0.9):
        if performance < threshold:
            # Add a new hidden layer
            new_size = self.hidden_sizes[-1] // 2
            self.hidden_sizes.append(new_size)
        elif len(self.hidden_sizes) > 1:
            # Remove a hidden layer if performance is good
            self.hidden_sizes.pop()
        self.build_network()

# Example usage
model = AdaptiveStochasticNN(input_size=784, initial_hidden_sizes=[128, 64], output_size=10)
print(f"Initial architecture: {model.hidden_sizes}")

# Simulate adaptation based on performance
performances = [0.85, 0.92, 0.88, 0.95]
for i, perf in enumerate(performances):
    model.adapt_architecture(perf)
    print(f"Performance: {perf:.2f}, New architecture: {model.hidden_sizes}")
```

Slide 7: Stochastic Gradient Descent with Momentum

To further enhance our adaptive neural network, let's implement Stochastic Gradient Descent (SGD) with momentum. This optimization technique can help accelerate convergence and reduce oscillations during training.

```python
import torch
import torch.optim as optim

class StochasticSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, noise_level=0.01):
        defaults = dict(lr=lr, momentum=momentum, noise_level=noise_level)
        super(StochasticSGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # Add noise to the gradient
                noise = torch.randn_like(d_p) * group['noise_level']
                d_p.add_(noise)
                
                if 'momentum_buffer' not in self.state[p]:
                    buf = self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = self.state[p]['momentum_buffer']
                
                buf.mul_(group['momentum']).add_(d_p)
                p.data.add_(-group['lr'], buf)

# Example usage
model = AdaptiveStochasticNN(input_size=784, initial_hidden_sizes=[128, 64], output_size=10)
optimizer = StochasticSGD(model.parameters(), lr=0.01, momentum=0.9, noise_level=0.01)

# Training loop (simplified)
for epoch in range(5):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

Slide 8: Visualizing the Stochastic Synapses

To better understand the effect of stochastic synapses, let's create a visualization of the weight distributions before and after training.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_weight_distributions(model, layer_index=0):
    initial_weights = model.layers[layer_index].linear.weight.detach().numpy().flatten()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(initial_weights, bins=50, alpha=0.7)
    plt.title('Initial Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    # Train the model (simplified)
    optimizer = StochasticSGD(model.parameters(), lr=0.01, momentum=0.9, noise_level=0.01)
    for _ in range(100):
        for data, target in train_loader:
            data = data.view(-1, 784)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    trained_weights = model.layers[layer_index].linear.weight.detach().numpy().flatten()
    
    plt.subplot(1, 2, 2)
    plt.hist(trained_weights, bins=50, alpha=0.7)
    plt.title('Trained Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example usage
model = AdaptiveStochasticNN(input_size=784, initial_hidden_sizes=[128, 64], output_size=10)
plot_weight_distributions(model)
```

Slide 9: Real-Life Example: Image Denoising

Let's apply our adaptive neural network with stochastic synapses to a real-life problem: image denoising. We'll create a simple autoencoder structure to remove noise from images.

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            StochasticLinear(784, 256),
            nn.ReLU(),
            StochasticLinear(256, 128),
            nn.ReLU(),
            StochasticLinear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            StochasticLinear(64, 128),
            nn.ReLU(),
            StochasticLinear(128, 256),
            nn.ReLU(),
            StochasticLinear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load and preprocess an image
image = Image.open("sample_image.jpg").convert('L')
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
original_tensor = transform(image).view(1, -1)

# Add noise to the image
noisy_tensor = original_tensor + 0.3 * torch.randn_like(original_tensor)
noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)

# Train the model (simplified)
model = DenoisingAutoencoder()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

for _ in range(1000):
    optimizer.zero_grad()
    output = model(noisy_tensor)
    loss = criterion(output, original_tensor)
    loss.backward()
    optimizer.step()

# Denoise the image
with torch.no_grad():
    denoised_tensor = model(noisy_tensor)

# Visualize results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(original_tensor.view(28, 28), cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(noisy_tensor.view(28, 28), cmap='gray')
axs[1].set_title('Noisy')
axs[2].imshow(denoised_tensor.view(28, 28), cmap='gray')
axs[2].set_title('Denoised')
plt.show()
```

Slide 10: Adaptive Learning Rate

Implementing an adaptive learning rate can significantly improve the training process of our stochastic neural network. We'll use a simple adaptive learning rate technique based on the loss trend.

```python
class AdaptiveLearningRate:
    def __init__(self, initial_lr=0.01, patience=10, factor=0.5, min_lr=1e-6):
        self.lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.counter = 0

    def update(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self.counter = 0
        return self.lr

# Example usage
model = AdaptiveStochasticNN(input_size=784, initial_hidden_sizes=[128, 64], output_size=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
adaptive_lr = AdaptiveLearningRate()

for epoch in range(10):
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    new_lr = adaptive_lr.update(avg_loss)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Learning Rate: {new_lr:.6f}")
```

Slide 11: Dropout in Stochastic Neural Networks

Dropout is a regularization technique that can be combined with stochastic synapses to further improve the model's generalization. Let's implement a stochastic dropout layer.

```python
class StochasticDropout(nn.Module):
    def __init__(self, p=0.5, noise_level=0.1):
        super().__init__()
        self.p = p
        self.noise_level = noise_level

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            noise = torch.randn_like(x) * self.noise_level
            return (x * mask + noise) / (1 - self.p)
        return x

class StochasticNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.extend([
                StochasticLinear(prev_size, size),
                nn.ReLU(),
                StochasticDropout()
            ])
            prev_size = size
        layers.append(StochasticLinear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Example usage
model = StochasticNN(input_size=784, hidden_sizes=[256, 128], output_size=10)
x = torch.randn(32, 784)
output = model(x)
print(output.shape)
```

Slide 12: Bayesian Perspective on Stochastic Neural Networks

Stochastic neural networks can be viewed from a Bayesian perspective, where the network weights are treated as random variables. This approach allows us to quantify uncertainty in predictions.

```python
import torch.distributions as dist

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior distributions
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        
    def forward(self, x):
        weight = dist.Normal(self.weight_mu, torch.exp(self.weight_sigma)).rsample()
        bias = dist.Normal(self.bias_mu, torch.exp(self.bias_sigma)).rsample()
        return nn.functional.linear(x, weight, bias)

# Example usage
bayesian_layer = BayesianLinear(10, 5)
x = torch.randn(3, 10)
output = bayesian_layer(x)
print(output.shape)
```

Slide 13: Analyzing Uncertainty in Stochastic Neural Networks

One advantage of stochastic neural networks is their ability to provide uncertainty estimates. Let's implement a function to analyze prediction uncertainty using Monte Carlo sampling.

```python
def predict_with_uncertainty(model, x, num_samples=100):
    model.eval()
    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            output = model(x)
            predictions.append(output)
    
    # Calculate mean and variance of predictions
    pred_mean = torch.stack(predictions).mean(dim=0)
    pred_var = torch.stack(predictions).var(dim=0)
    
    return pred_mean, pred_var

# Example usage
model = StochasticNN(input_size=784, hidden_sizes=[256, 128], output_size=10)
x = torch.randn(1, 784)  # Single input sample
mean, variance = predict_with_uncertainty(model, x)

print("Prediction mean:", mean)
print("Prediction variance:", variance)

plt.figure(figsize=(10, 5))
plt.bar(range(10), mean[0].numpy(), yerr=torch.sqrt(variance[0]).numpy(), capsize=5)
plt.title("Prediction with Uncertainty")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.show()
```

Slide 14: Real-Life Example: Handwritten Digit Recognition with Uncertainty

Let's apply our stochastic neural network to the task of handwritten digit recognition, demonstrating how it can provide uncertainty estimates for its predictions.

```python
import torchvision.datasets as datasets

# Load MNIST dataset
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)

# Define and train the model (assuming it's already trained)
model = StochasticNN(input_size=784, hidden_sizes=[256, 128], output_size=10)

# Function to plot digit with prediction and uncertainty
def plot_prediction_with_uncertainty(image, pred_mean, pred_var):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classes = range(10)
    plt.bar(classes, pred_mean, yerr=torch.sqrt(pred_var), capsize=5)
    plt.title("Prediction with Uncertainty")
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Get a random test sample
data, true_label = next(iter(test_loader))
data = data.view(1, -1)

# Make prediction with uncertainty
pred_mean, pred_var = predict_with_uncertainty(model, data)

# Plot results
plot_prediction_with_uncertainty(data.view(28, 28), pred_mean[0], pred_var[0])
print(f"True label: {true_label.item()}")
print(f"Predicted label: {pred_mean.argmax().item()}")
```

Slide 15: Additional Resources

For those interested in diving deeper into adaptive neural networks with stochastic synapses, here are some valuable resources:

1. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Srivastava et al. (2014) ArXiv: [https://arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
2. "Weight Uncertainty in Neural Networks" by Blundell et al. (2015) ArXiv: [https://arxiv.org/abs/1505.05424](https://arxiv.org/abs/1505.05424)
3. "Bayesian Learning for Neural Networks" by Neal (1996) Book available at: [https://www.springer.com/gp/book/9780387947242](https://www.springer.com/gp/book/9780387947242)
4. "A Comprehensive Introduction to Different Types of Convolutions in Deep Learning" by Zhang (2019) ArXiv: [https://arxiv.org/abs/1905.03193](https://arxiv.org/abs/1905.03193)

These resources provide in-depth explanations and theoretical foundations for the concepts we've explored in this presentation. They can help you further understand and implement advanced techniques in stochastic neural networks and adaptive architectures.

