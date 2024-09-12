## Creating and Manipulating Tensors with PyTorch
Slide 1: Introduction to Tensors in PyTorch

Tensors are the fundamental data structure in PyTorch, representing multi-dimensional arrays. They are similar to NumPy arrays but can be used on GPUs for accelerated computing. Let's create a simple tensor and explore its properties.

```python
import torch

# Create a 2x3 tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}")

# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
# Shape: torch.Size([2, 3])
# Datatype: torch.int64
# Device: cpu
```

Slide 2: Creating Tensors from Python Lists

PyTorch allows us to create tensors from Python lists easily. We can specify the datatype and device during creation.

```python
import torch

# From a list
list_tensor = torch.tensor([1, 2, 3, 4, 5])
print(list_tensor)

# Specifying datatype and device
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cpu')
print(float_tensor)

# Output:
# tensor([1, 2, 3, 4, 5])
# tensor([1., 2., 3.])
```

Slide 3: Tensor Initialization Functions

PyTorch provides various functions to initialize tensors with specific values or patterns.

```python
import torch

# Zeros tensor
zeros = torch.zeros(3, 4)
print("Zeros tensor:", zeros)

# Ones tensor
ones = torch.ones(2, 3)
print("Ones tensor:", ones)

# Random tensor
random = torch.rand(2, 2)
print("Random tensor:", random)

# Output:
# Zeros tensor: tensor([[0., 0., 0., 0.],
#                       [0., 0., 0., 0.],
#                       [0., 0., 0., 0.]])
# Ones tensor: tensor([[1., 1., 1.],
#                      [1., 1., 1.]])
# Random tensor: tensor([[0.1234, 0.5678],
#                        [0.9012, 0.3456]])
```

Slide 4: Tensor Operations: Addition and Multiplication

Tensors support element-wise operations, making mathematical computations straightforward.

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
sum_tensor = a + b
print("Sum:", sum_tensor)

# Multiplication
prod_tensor = a * b
print("Product:", prod_tensor)

# Matrix multiplication
mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[5, 6], [7, 8]])
matmul = torch.matmul(mat1, mat2)
print("Matrix multiplication:", matmul)

# Output:
# Sum: tensor([5, 7, 9])
# Product: tensor([ 4, 10, 18])
# Matrix multiplication: tensor([[19, 22],
#                                [43, 50]])
```

Slide 5: Reshaping Tensors

Reshaping allows us to change the dimensions of a tensor while preserving its data.

```python
import torch

original = torch.tensor([1, 2, 3, 4, 5, 6])
print("Original:", original)

# Reshape to 2x3
reshaped = original.reshape(2, 3)
print("Reshaped to 2x3:", reshaped)

# Reshape to 3x2
reshaped_again = original.reshape(3, 2)
print("Reshaped to 3x2:", reshaped_again)

# Output:
# Original: tensor([1, 2, 3, 4, 5, 6])
# Reshaped to 2x3: tensor([[1, 2, 3],
#                          [4, 5, 6]])
# Reshaped to 3x2: tensor([[1, 2],
#                          [3, 4],
#                          [5, 6]])
```

Slide 6: Indexing and Slicing Tensors

Accessing specific elements or subsets of tensors is crucial for data manipulation.

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original tensor:", tensor)

# Indexing
print("Element at (1, 2):", tensor[1, 2])

# Slicing
print("First row:", tensor[0, :])
print("Second column:", tensor[:, 1])
print("Submatrix:", tensor[1:, 1:])

# Output:
# Original tensor: tensor([[1, 2, 3],
#                          [4, 5, 6],
#                          [7, 8, 9]])
# Element at (1, 2): tensor(6)
# First row: tensor([1, 2, 3])
# Second column: tensor([2, 5, 8])
# Submatrix: tensor([[5, 6],
#                    [8, 9]])
```

Slide 7: Tensor Data Types and Type Casting

PyTorch supports various data types, and we can convert between them using type casting.

```python
import torch

# Integer tensor
int_tensor = torch.tensor([1, 2, 3])
print("Integer tensor:", int_tensor, int_tensor.dtype)

# Float tensor
float_tensor = int_tensor.float()
print("Float tensor:", float_tensor, float_tensor.dtype)

# Double precision tensor
double_tensor = int_tensor.double()
print("Double tensor:", double_tensor, double_tensor.dtype)

# Boolean tensor
bool_tensor = int_tensor > 1
print("Boolean tensor:", bool_tensor, bool_tensor.dtype)

# Output:
# Integer tensor: tensor([1, 2, 3]) torch.int64
# Float tensor: tensor([1., 2., 3.]) torch.float32
# Double tensor: tensor([1., 2., 3.]) torch.float64
# Boolean tensor: tensor([False,  True,  True]) torch.bool
```

Slide 8: Tensor Operations on GPU

PyTorch allows seamless transition between CPU and GPU computations.

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Create a tensor on CPU
cpu_tensor = torch.tensor([1, 2, 3, 4, 5])
print("CPU tensor:", cpu_tensor)

# Move tensor to GPU (if available)
gpu_tensor = cpu_tensor.to(device)
print("GPU tensor:", gpu_tensor)

# Perform operation on GPU
result = gpu_tensor * 2
print("Result on GPU:", result)

# Move result back to CPU
cpu_result = result.to("cpu")
print("Result on CPU:", cpu_result)

# Output (if CUDA is available):
# CUDA is available. Using GPU.
# CPU tensor: tensor([1, 2, 3, 4, 5])
# GPU tensor: tensor([1, 2, 3, 4, 5], device='cuda:0')
# Result on GPU: tensor([2, 4, 6, 8, 10], device='cuda:0')
# Result on CPU: tensor([ 2,  4,  6,  8, 10])
```

Slide 9: Broadcasting in PyTorch

Broadcasting allows operations between tensors of different shapes, automatically expanding smaller tensors to match larger ones.

```python
import torch

# Create tensors of different shapes
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1], [2], [3]])

# Broadcasting addition
c = a + b
print("Result of broadcasting:")
print(c)

# Explicit broadcasting
d = a.unsqueeze(0) + b.unsqueeze(1)
print("Result of explicit broadcasting:")
print(d)

# Output:
# Result of broadcasting:
# tensor([[2, 3, 4],
#         [3, 4, 5],
#         [4, 5, 6]])
# Result of explicit broadcasting:
# tensor([[2, 3, 4],
#         [3, 4, 5],
#         [4, 5, 6]])
```

Slide 10: Tensor Reduction Operations

Reduction operations allow us to compute aggregate values across tensor dimensions.

```python
import torch

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original tensor:")
print(tensor)

# Sum of all elements
print("Sum of all elements:", tensor.sum())

# Mean of all elements
print("Mean of all elements:", tensor.mean())

# Sum along rows (dim=1)
print("Sum along rows:", tensor.sum(dim=1))

# Mean along columns (dim=0)
print("Mean along columns:", tensor.mean(dim=0))

# Output:
# Original tensor:
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])
# Sum of all elements: tensor(45)
# Mean of all elements: tensor(5.)
# Sum along rows: tensor([ 6, 15, 24])
# Mean along columns: tensor([4., 5., 6.])
```

Slide 11: Tensor Concatenation and Stacking

Combining multiple tensors is a common operation in data preprocessing and model architecture design.

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate along rows (dim=0)
c = torch.cat((a, b), dim=0)
print("Concatenated along rows:")
print(c)

# Concatenate along columns (dim=1)
d = torch.cat((a, b), dim=1)
print("Concatenated along columns:")
print(d)

# Stack tensors
e = torch.stack((a, b))
print("Stacked tensors:")
print(e)

# Output:
# Concatenated along rows:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])
# Concatenated along columns:
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
# Stacked tensors:
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])
```

Slide 12: Real-life Example: Image Processing

Let's use tensors to perform simple image processing tasks, such as grayscale conversion and edge detection.

```python
import torch
import matplotlib.pyplot as plt

# Create a simple 5x5 color image tensor
image = torch.tensor([
    [[255, 0, 0], [255, 0, 0], [255, 255, 255], [0, 255, 0], [0, 255, 0]],
    [[255, 0, 0], [255, 0, 0], [255, 255, 255], [0, 255, 0], [0, 255, 0]],
    [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
    [[0, 0, 255], [0, 0, 255], [255, 255, 255], [255, 255, 0], [255, 255, 0]],
    [[0, 0, 255], [0, 0, 255], [255, 255, 255], [255, 255, 0], [255, 255, 0]]
], dtype=torch.float32) / 255.0

# Convert to grayscale
grayscale = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]

# Simple edge detection using gradient
gradient_x = grayscale[:, 1:] - grayscale[:, :-1]
gradient_y = grayscale[1:, :] - grayscale[:-1, :]
edges = torch.sqrt(gradient_x[:, :-1]**2 + gradient_y[:-1, :]**2)

# Plot results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image.numpy())
axs[0].set_title("Original")
axs[1].imshow(grayscale.numpy(), cmap='gray')
axs[1].set_title("Grayscale")
axs[2].imshow(edges.numpy(), cmap='gray')
axs[2].set_title("Edges")
plt.show()
```

Slide 13: Real-life Example: Time Series Analysis

Using tensors for time series analysis, we'll create a simple moving average calculation.

```python
import torch
import matplotlib.pyplot as plt

# Generate a random time series data
time_series = torch.randn(100) * 10 + 50  # Mean 50, std 10

# Calculate 5-day moving average
window_size = 5
weights = torch.ones(window_size) / window_size
moving_avg = torch.conv1d(time_series.view(1, 1, -1), weights.view(1, 1, -1), padding=window_size//2).squeeze()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_series.numpy(), label='Original Data')
plt.plot(moving_avg.numpy(), label='5-day Moving Average')
plt.legend()
plt.title('Time Series with Moving Average')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

print("Original data shape:", time_series.shape)
print("Moving average shape:", moving_avg.shape)

# Output:
# Original data shape: torch.Size([100])
# Moving average shape: torch.Size([100])
```

Slide 14: Additional Resources

For further exploration of PyTorch and tensor operations, consider the following resources:

1. PyTorch official documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
2. "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann
3. ArXiv paper: "PyTorch: An Imperative Style, High-Performance Deep Learning Library" by Adam Paszke et al. ([https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703))
4. PyTorch tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
5. "Dive into Deep Learning" interactive book: [https://d2l.ai/](https://d2l.ai/)

These resources provide in-depth explanations, tutorials, and advanced techniques for working with tensors and building deep learning models using PyTorch.

