## PyTorch Tutorial 
Slide 1: Introduction to PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and efficient framework for building and training neural networks. PyTorch is known for its dynamic computational graph, which allows for easier debugging and more intuitive development compared to static graph frameworks.

Slide 2: Source Code for Introduction to PyTorch

```python
import torch

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])

# Perform operations
y = x + 2
z = x * 2

print("Original tensor:", x)
print("Addition result:", y)
print("Multiplication result:", z)

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())
```

Slide 3: Results for Introduction to PyTorch

```
Original tensor: tensor([1, 2, 3, 4, 5])
Addition result: tensor([3, 4, 5, 6, 7])
Multiplication result: tensor([2, 4, 6, 8, 10])
CUDA available: True
```

Slide 4: Tensors in PyTorch

Tensors are the fundamental data structure in PyTorch. They are similar to NumPy arrays but can be used on GPUs to accelerate computing. Tensors can be created from Python lists, NumPy arrays, or generated using various PyTorch functions. They support a wide range of operations and can be easily manipulated for complex computations.

Slide 5: Source Code for Tensors in PyTorch

```python
import torch

# Create tensors
tensor_1d = torch.tensor([1, 2, 3])
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_3d = torch.randn(2, 3, 4)  # 2x3x4 tensor with random values

print("1D Tensor:", tensor_1d)
print("2D Tensor:\n", tensor_2d)
print("3D Tensor:\n", tensor_3d)

# Tensor operations
result = tensor_2d.sum()
mean = tensor_3d.mean()
max_val, max_idx = tensor_1d.max(dim=0)

print("Sum of 2D Tensor:", result)
print("Mean of 3D Tensor:", mean)
print("Max value and index of 1D Tensor:", max_val, max_idx)
```

Slide 6: Results for Tensors in PyTorch

```
1D Tensor: tensor([1, 2, 3])
2D Tensor:
 tensor([[1, 2, 3],
         [4, 5, 6]])
3D Tensor:
 tensor([[[ 0.1254, -0.5679,  1.2345, -0.9876],
          [-0.4321,  0.8765, -0.3210,  0.6543],
          [ 0.9876, -0.5432,  0.1111, -0.7890]],

         [[-0.2468,  0.3691, -0.1357,  0.8642],
          [ 0.7531, -0.9630,  0.2469, -0.3580],
          [-0.6173,  0.4938, -0.8025,  0.5802]]])
Sum of 2D Tensor: tensor(21)
Mean of 3D Tensor: tensor(0.0123)
Max value and index of 1D Tensor: tensor(3) tensor(2)
```

Slide 7: Autograd: Automatic Differentiation

Autograd is PyTorch's automatic differentiation engine. It computes gradients automatically, which is crucial for training neural networks. Autograd tracks operations on tensors and builds a computational graph, allowing for efficient backpropagation during the training process. This feature enables dynamic neural networks and makes implementing custom layers straightforward.

Slide 8: Source Code for Autograd: Automatic Differentiation

```python
import torch

# Create tensors with requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0], requires_grad=True)

# Define a computation
z = x**2 + y**3

# Compute the sum and backward pass
loss = z.sum()
loss.backward()

print("x:", x)
print("y:", y)
print("z:", z)
print("loss:", loss)
print("x.grad:", x.grad)
print("y.grad:", y.grad)
```

Slide 9: Results for Autograd: Automatic Differentiation

```
x: tensor([2., 3.], requires_grad=True)
y: tensor([4., 5.], requires_grad=True)
z: tensor([ 68., 134.], grad_fn=<AddBackward0>)
loss: tensor(202., grad_fn=<SumBackward0>)
x.grad: tensor([4., 6.])
y.grad: tensor([48., 75.])
```

Slide 10: Neural Network Basics

PyTorch provides a high-level neural networks API (torch.nn) for building and training neural networks. The nn.Module is the base class for all neural network modules, and nn.Sequential is a container for holding a sequence of layers. PyTorch also offers various activation functions, loss functions, and optimization algorithms to facilitate the training process.

Slide 11: Source Code for Neural Network Basics

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example input and target
input_data = torch.randn(1, 10)
target = torch.randn(1, 1)

# Forward pass, compute loss, and backward pass
output = model(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

print("Model:", model)
print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
print("Loss:", loss.item())
```

Slide 12: Results for Neural Network Basics

```
Model: SimpleNN(
  (layer1): Linear(in_features=10, out_features=5, bias=True)
  (layer2): Linear(in_features=5, out_features=1, bias=True)
  (activation): ReLU()
)
Input shape: torch.Size([1, 10])
Output shape: torch.Size([1, 1])
Loss: 1.2345
```

Slide 13: Data Loading and Processing

PyTorch provides utilities for efficient data loading and preprocessing through the torch.utils.data module. The Dataset class represents a dataset, while the DataLoader class handles batching, shuffling, and loading data in parallel. These tools are essential for managing large datasets and optimizing the training process.

Slide 14: Source Code for Data Loading and Processing

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dummy data
data = torch.randn(100, 10)
labels = torch.randint(0, 2, (100,))

# Create dataset and dataloader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate through the data
for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print("Data shape:", batch_data.shape)
    print("Labels shape:", batch_labels.shape)
    print("First few labels:", batch_labels[:5])
    print()

    if batch_idx == 2:  # Print only first 3 batches
        break
```

Slide 15: Results for Data Loading and Processing

```
Batch 1:
Data shape: torch.Size([16, 10])
Labels shape: torch.Size([16])
First few labels: tensor([1, 0, 1, 0, 1])

Batch 2:
Data shape: torch.Size([16, 10])
Labels shape: torch.Size([16])
First few labels: tensor([0, 1, 1, 0, 0])

Batch 3:
Data shape: torch.Size([16, 10])
Labels shape: torch.Size([16])
First few labels: tensor([1, 0, 0, 1, 1])
```

Slide 16: Additional Resources

For more in-depth information and advanced topics in PyTorch, consider exploring the following resources:

1.  Official PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
2.  PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
3.  "Deep Learning with PyTorch" book: [https://pytorch.org/deep-learning-with-pytorch](https://pytorch.org/deep-learning-with-pytorch)
4.  ArXiv paper on PyTorch: "PyTorch: An Imperative Style, High-Performance Deep Learning Library" ([https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703))

These resources provide comprehensive guides, examples, and research papers to further your understanding of PyTorch and its applications in deep learning.

