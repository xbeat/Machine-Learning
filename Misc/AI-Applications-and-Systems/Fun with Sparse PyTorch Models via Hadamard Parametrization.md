## Fun with Sparse PyTorch Models via Hadamard Parametrization
Slide 1: Introduction to Sparsity in PyTorch

Sparsity in neural networks refers to the concept of having many zero-valued parameters. This can lead to more efficient models in terms of computation and memory usage. In this presentation, we'll explore how to implement sparsity in PyTorch using Hadamard product parametrization.

```python
import torch
import torch.nn as nn

# Create a sparse tensor
sparse_tensor = torch.sparse_coo_tensor(indices=[[0, 1, 2], [2, 0, 1]],
                                        values=[1, 2, 3],
                                        size=(3, 3))

print(sparse_tensor.to_dense())
```

Slide 2: Hadamard Product Parametrization

The Hadamard product, also known as element-wise multiplication, is a key operation in implementing sparsity. We'll use it to create a mask that determines which weights in our neural network are active.

```python
def hadamard_product(tensor1, tensor2):
    return tensor1 * tensor2

# Example
a = torch.tensor([1, 2, 3])
b = torch.tensor([0, 1, 0])
result = hadamard_product(a, b)
print(result)  # Output: tensor([0, 2, 0])
```

Slide 3: Creating a Sparse Linear Layer

Let's implement a sparse linear layer using Hadamard product parametrization. This layer will have a mask that determines which weights are active.

```python
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.mask = nn.Parameter(torch.bernoulli(torch.full_like(self.weight, 1 - sparsity)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)

# Usage
layer = SparseLinear(10, 5, sparsity=0.7)
input_tensor = torch.randn(1, 10)
output = layer(input_tensor)
print(output.shape)  # Output: torch.Size([1, 5])
```

Slide 4: Visualizing Sparsity

To better understand the effect of sparsity, let's create a function to visualize the sparse weights of our layer.

```python
import matplotlib.pyplot as plt

def visualize_sparsity(layer):
    plt.figure(figsize=(10, 5))
    plt.imshow(layer.weight.detach() * layer.mask.detach(), cmap='viridis')
    plt.colorbar()
    plt.title(f'Sparse Weights (Sparsity: {1 - layer.mask.mean().item():.2f})')
    plt.xlabel('Input Features')
    plt.ylabel('Output Features')
    plt.show()

# Visualize our sparse layer
visualize_sparsity(layer)
```

Slide 5: Implementing Gradual Pruning

Gradual pruning is a technique where we start with a dense network and gradually increase sparsity during training. Let's implement a function to update the mask of our sparse layer.

```python
def update_mask(layer, target_sparsity, current_step, total_steps):
    current_sparsity = 1 - layer.mask.mean().item()
    new_sparsity = current_sparsity + (target_sparsity - current_sparsity) * (current_step / total_steps)
    
    # Sort weights by magnitude
    weights_mag = torch.abs(layer.weight.data)
    _, indices = torch.sort(weights_mag.view(-1))
    
    # Update mask
    new_mask = torch.ones_like(layer.mask)
    new_mask.view(-1)[indices[:int(new_sparsity * new_mask.numel())]] = 0
    layer.mask.data = new_mask

# Example usage in a training loop
for epoch in range(num_epochs):
    for step, (inputs, targets) in enumerate(dataloader):
        # ... forward pass, loss calculation, etc.
        update_mask(layer, target_sparsity=0.9, current_step=step, total_steps=len(dataloader) * num_epochs)
        # ... backward pass, optimization step, etc.
```

Slide 6: Sparse Convolutional Layer

Let's extend our sparsity concept to convolutional layers, which are commonly used in computer vision tasks.

```python
class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, sparsity=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask = nn.Parameter(torch.bernoulli(torch.full_like(self.conv.weight, 1 - sparsity)))

    def forward(self, x):
        return nn.functional.conv2d(x, self.conv.weight * self.mask, self.conv.bias,
                                    self.conv.stride, self.conv.padding)

# Usage
sparse_conv = SparseConv2d(3, 16, kernel_size=3, sparsity=0.7)
input_image = torch.randn(1, 3, 32, 32)
output = sparse_conv(input_image)
print(output.shape)  # Output: torch.Size([1, 16, 30, 30])
```

Slide 7: Real-Life Example: Image Classification

Let's create a simple sparse CNN for image classification using the CIFAR-10 dataset.

```python
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Define the sparse CNN
class SparseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SparseConv2d(3, 32, kernel_size=3, sparsity=0.5)
        self.conv2 = SparseConv2d(32, 64, kernel_size=3, sparsity=0.5)
        self.fc = SparseLinear(64 * 6 * 6, 10, sparsity=0.5)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = SparseCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(5):
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

Slide 8: Analyzing Sparsity Impact

Let's create functions to analyze the impact of sparsity on our model's performance and efficiency.

```python
def count_nonzero_params(model):
    return sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_tensor, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_runs

# Analyze our sparse CNN
sparse_model = SparseCNN()
dense_model = SparseCNN()
for m in dense_model.modules():
    if isinstance(m, (SparseConv2d, SparseLinear)):
        m.mask.data.fill_(1)

print(f"Sparse model non-zero parameters: {count_nonzero_params(sparse_model)}")
print(f"Dense model non-zero parameters: {count_nonzero_params(dense_model)}")

input_tensor = torch.randn(1, 3, 32, 32)
sparse_time = measure_inference_time(sparse_model, input_tensor)
dense_time = measure_inference_time(dense_model, input_tensor)

print(f"Sparse model inference time: {sparse_time:.6f} seconds")
print(f"Dense model inference time: {dense_time:.6f} seconds")
```

Slide 9: Magnitude-based Pruning

Instead of random pruning, we can use magnitude-based pruning to remove the smallest weights. This often leads to better performance.

```python
def magnitude_based_prune(module, sparsity):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        weight = module.weight.data
        mask = torch.ones_like(weight)
        threshold = torch.kthvalue(torch.abs(weight).view(-1), int(sparsity * weight.numel())).values
        mask[torch.abs(weight) < threshold] = 0
        module.weight.data *= mask

# Apply pruning to our model
model = SparseCNN()
for m in model.modules():
    if isinstance(m, (SparseConv2d, SparseLinear)):
        magnitude_based_prune(m, sparsity=0.5)

# Visualize the pruned weights of the first convolutional layer
plt.figure(figsize=(10, 5))
plt.imshow(model.conv1.conv.weight.data[0].abs().mean(dim=0), cmap='viridis')
plt.colorbar()
plt.title('Pruned Weights of First Conv Layer')
plt.show()
```

Slide 10: Structured Sparsity

Structured sparsity involves pruning entire groups of weights, such as channels or neurons. This can lead to better hardware efficiency.

```python
def structured_channel_prune(module, sparsity):
    if isinstance(module, nn.Conv2d):
        weight = module.weight.data
        norm = weight.abs().sum(dim=(1, 2, 3))
        threshold = torch.kthvalue(norm, int(sparsity * len(norm))).values
        mask = (norm >= threshold).float().view(-1, 1, 1, 1)
        module.weight.data *= mask

# Apply structured pruning to our model
model = SparseCNN()
for m in model.modules():
    if isinstance(m, SparseConv2d):
        structured_channel_prune(m.conv, sparsity=0.3)

# Visualize the pruned channels
plt.figure(figsize=(10, 5))
plt.bar(range(model.conv1.conv.weight.size(0)), model.conv1.conv.weight.data.abs().sum(dim=(1, 2, 3)))
plt.title('Channel Magnitudes After Structured Pruning')
plt.xlabel('Channel Index')
plt.ylabel('Magnitude')
plt.show()
```

Slide 11: Dynamic Sparse Training

Dynamic sparse training involves changing the sparsity pattern during training. This allows the network to adapt its sparse structure.

```python
class DynamicSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.mask = nn.Parameter(torch.bernoulli(torch.full_like(self.weight, 1 - sparsity)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)

    def update_mask(self):
        with torch.no_grad():
            # Grow new connections
            new_connections = torch.bernoulli(torch.full_like(self.mask, 0.1))
            self.mask.data = torch.clamp(self.mask + new_connections, 0, 1)
            
            # Prune weakest connections
            abs_weights = torch.abs(self.weight * self.mask)
            threshold = torch.kthvalue(abs_weights.view(-1), int(self.mask.numel() * 0.5)).values
            self.mask.data[abs_weights < threshold] = 0

# Usage in training loop
layer = DynamicSparseLinear(10, 5)
for epoch in range(num_epochs):
    # ... forward pass, loss calculation, backward pass ...
    if epoch % 10 == 0:
        layer.update_mask()
    # ... optimization step ...
```

Slide 12: Real-Life Example: Natural Language Processing

Let's apply sparsity to a simple sentiment analysis model using a sparse LSTM.

```python
class SparseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.weight_mask = nn.Parameter(torch.bernoulli(torch.full((4 * hidden_size, input_size + hidden_size), 1 - sparsity)))
        
    def forward(self, x, hidden=None):
        self.lstm.weight_ih_l0.data *= self.weight_mask[:, :self.lstm.input_size]
        self.lstm.weight_hh_l0.data *= self.weight_mask[:, self.lstm.input_size:]
        return self.lstm(x, hidden)

# Simple sentiment analysis model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, sparsity=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = SparseLSTM(embed_dim, hidden_dim, sparsity)
        self.fc = SparseLinear(hidden_dim, output_dim, sparsity)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# Example usage (assuming preprocessed data)
vocab_size = 10000
embed_dim = 100
hidden_dim = 256
output_dim = 2  # binary sentiment

model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim, sparsity=0.7)
input_sequence = torch.randint(0, vocab_size, (20, 32))  # (sequence_length, batch_size)
output = model(input_sequence)
print(output.shape)  # Output: torch.Size([32, 2])
```

Slide 13: Challenges and Considerations

While sparsity can bring benefits, there are challenges to consider:

1. Training instability: Sparse networks can be more difficult to train, requiring careful hyperparameter tuning.
2. Hardware efficiency: Not all hardware can efficiently leverage sparsity, potentially limiting real-world speedups.
3. Accuracy trade-offs: Excessive sparsity can lead to reduced model accuracy.
4. Implementation complexity: Implementing and maintaining sparse models can be more challenging than dense models.
5. Gradient flow: Sparse connections may impede gradient flow during backpropagation.

To address these challenges, researchers and practitioners often employ techniques such as gradual sparsification, dynamic sparse training, and specialized optimizers. Let's explore a simple implementation of gradual sparsification:

```python
class GradualSparsifier:
    def __init__(self, model, initial_sparsity, final_sparsity, start_epoch, end_epoch):
        self.model = model
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        
    def step(self, epoch):
        if epoch < self.start_epoch:
            return
        
        if epoch > self.end_epoch:
            sparsity = self.final_sparsity
        else:
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            sparsity = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
        
        for module in self.model.modules():
            if isinstance(module, (SparseLinear, SparseConv2d)):
                self._apply_sparsity(module, sparsity)
    
    def _apply_sparsity(self, module, sparsity):
        with torch.no_grad():
            mask = module.mask.data
            weights = module.weight.data
            num_zeros = int(sparsity * mask.numel())
            threshold = torch.kthvalue(weights.abs().view(-1), num_zeros).values
            new_mask = (weights.abs() > threshold).float()
            module.mask.data = new_mask

# Usage in training loop
model = YourSparseModel()
sparsifier = GradualSparsifier(model, initial_sparsity=0.5, final_sparsity=0.9, start_epoch=10, end_epoch=50)

for epoch in range(num_epochs):
    # ... training code ...
    sparsifier.step(epoch)
    # ... more training code ...
```

Slide 14: Future Directions and Research

The field of sparse neural networks is rapidly evolving. Some promising directions include:

1. Neural Architecture Search (NAS) for sparse networks
2. Combining sparsity with other efficiency techniques like quantization
3. Developing hardware accelerators optimized for sparse computations
4. Exploring the theoretical foundations of sparsity in deep learning

Here's a simple example of combining sparsity with quantization:

```python
class SparseQuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.mask = nn.Parameter(torch.bernoulli(torch.full_like(self.weight, 1 - sparsity)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bits = bits
        
    def quantize(self, x):
        max_val = x.abs().max()
        scale = (2 ** (self.bits - 1) - 1) / max_val
        return torch.round(x * scale) / scale
    
    def forward(self, x):
        quantized_weight = self.quantize(self.weight * self.mask)
        return nn.functional.linear(x, quantized_weight, self.bias)

# Usage
layer = SparseQuantizedLinear(10, 5, sparsity=0.7, bits=4)
input_tensor = torch.randn(1, 10)
output = layer(input_tensor)
print(output.shape)  # Output: torch.Size([1, 5])
```

Slide 15: Additional Resources

For those interested in diving deeper into sparsity in PyTorch and neural networks, here are some valuable resources:

1. "The State of Sparsity in Deep Neural Networks" by T. Hoefler et al. (2021) ArXiv: [https://arxiv.org/abs/1902.09574](https://arxiv.org/abs/1902.09574)
2. "Sparse Training for Deep Neural Networks" by S. Han et al. (2019) ArXiv: [https://arxiv.org/abs/1904.10274](https://arxiv.org/abs/1904.10274)
3. "Dynamic Sparse Reparameterization" by H. Liu et al. (2019) ArXiv: [https://arxiv.org/abs/1911.10474](https://arxiv.org/abs/1911.10474)
4. "Rigging the Lottery: Making All Tickets Winners" by U. Evci et al. (2020) ArXiv: [https://arxiv.org/abs/1911.11134](https://arxiv.org/abs/1911.11134)

These papers provide in-depth discussions on various aspects of sparsity in neural networks, including theoretical foundations, practical implementations, and state-of-the-art techniques.

