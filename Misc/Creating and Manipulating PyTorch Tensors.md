## Creating and Manipulating PyTorch Tensors
Slide 1: Introduction to PyTorch Tensors

PyTorch is a powerful deep learning library that uses tensors as its fundamental data structure. Tensors are multi-dimensional arrays that can represent various types of data, from scalars to complex matrices. In this presentation, we'll explore how to create and manipulate tensors using PyTorch.

```python
import torch

# Create a simple 2D tensor
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_2d)

# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
```

Slide 2: Creating Tensors from Python Lists

One of the most straightforward ways to create a tensor is by converting a Python list. This method allows you to initialize tensors with specific values.

```python
# Create a 1D tensor from a list
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(tensor_1d)

# Create a 3D tensor from nested lists
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor_3d)

# Output:
# tensor([1, 2, 3, 4, 5])
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])
```

Slide 3: Generating Tensors with PyTorch Functions

PyTorch provides various functions to create tensors with specific patterns or random values. These functions are useful for initializing weights in neural networks or creating data for experiments.

```python
# Create a tensor of ones
ones_tensor = torch.ones(3, 4)
print("Ones tensor:")
print(ones_tensor)

# Create a tensor of zeros
zeros_tensor = torch.zeros(2, 3)
print("\nZeros tensor:")
print(zeros_tensor)

# Create a tensor with random values
random_tensor = torch.rand(2, 2)
print("\nRandom tensor:")
print(random_tensor)

# Output:
# Ones tensor:
# tensor([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]])
#
# Zeros tensor:
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
#
# Random tensor:
# tensor([[0.1234, 0.5678],
#         [0.9012, 0.3456]])
```

Slide 4: Tensor Attributes and Properties

Understanding tensor attributes is crucial for working with PyTorch. These properties provide information about the tensor's shape, data type, and device (CPU or GPU).

```python
tensor = torch.randn(3, 4, 5)

print(f"Shape: {tensor.shape}")
print(f"Dimensions: {tensor.dim()}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")

# Reshape the tensor
reshaped_tensor = tensor.reshape(6, 10)
print(f"\nReshaped tensor shape: {reshaped_tensor.shape}")

# Output:
# Shape: torch.Size([3, 4, 5])
# Dimensions: 3
# Data type: torch.float32
# Device: cpu
#
# Reshaped tensor shape: torch.Size([6, 10])
```

Slide 5: Indexing and Slicing Tensors

Accessing specific elements or subsets of tensors is similar to working with NumPy arrays. PyTorch supports various indexing and slicing operations.

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access a single element
print(f"Element at [1, 2]: {tensor[1, 2]}")

# Slice a row
print(f"Second row: {tensor[1, :]}")

# Slice a column
print(f"Third column: {tensor[:, 2]}")

# Advanced indexing
indices = torch.tensor([0, 2])
print(f"First and third rows: {tensor[indices]}")

# Output:
# Element at [1, 2]: 6
# Second row: tensor([4, 5, 6])
# Third column: tensor([3, 6, 9])
# First and third rows: tensor([[1, 2, 3],
#                               [7, 8, 9]])
```

Slide 6: Tensor Operations: Addition and Subtraction

PyTorch provides element-wise operations for tensors. These operations are fundamental for various mathematical computations in deep learning.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
sum_tensor = a + b
print(f"Sum: {sum_tensor}")

# Subtraction
diff_tensor = b - a
print(f"Difference: {diff_tensor}")

# In-place addition
a.add_(b)
print(f"In-place addition result: {a}")

# Output:
# Sum: tensor([5, 7, 9])
# Difference: tensor([3, 3, 3])
# In-place addition result: tensor([5, 7, 9])
```

Slide 7: Tensor Operations: Multiplication and Division

Element-wise multiplication and division are essential operations in tensor manipulation. PyTorch also supports matrix multiplication for 2D tensors.

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Element-wise multiplication
prod_tensor = a * b
print(f"Element-wise product: {prod_tensor}")

# Element-wise division
div_tensor = b / a
print(f"Element-wise division: {div_tensor}")

# Matrix multiplication
mat1 = torch.tensor([[1, 2], [3, 4]])
mat2 = torch.tensor([[5, 6], [7, 8]])
mat_prod = torch.matmul(mat1, mat2)
print(f"Matrix product:\n{mat_prod}")

# Output:
# Element-wise product: tensor([ 4, 10, 18])
# Element-wise division: tensor([4.0000, 2.5000, 2.0000])
# Matrix product:
# tensor([[19, 22],
#         [43, 50]])
```

Slide 8: Broadcasting in PyTorch

Broadcasting allows operations between tensors of different shapes. PyTorch automatically expands smaller tensors to match the shape of larger ones, enabling efficient computations.

```python
# Create tensors of different shapes
a = torch.tensor([1, 2, 3])
b = torch.tensor([[1], [2], [3]])

# Broadcasting addition
result = a + b
print("Broadcasting result:")
print(result)

# Broadcasting with scalar
scalar = 2
scalar_result = a * scalar
print("\nScalar multiplication result:")
print(scalar_result)

# Output:
# Broadcasting result:
# tensor([[2, 3, 4],
#         [3, 4, 5],
#         [4, 5, 6]])
#
# Scalar multiplication result:
# tensor([2, 4, 6])
```

Slide 9: Tensor Reduction Operations

Reduction operations allow you to compute aggregate values across tensor dimensions. Common reduction operations include sum, mean, max, and min.

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Sum of all elements
total_sum = tensor.sum()
print(f"Sum of all elements: {total_sum}")

# Mean along rows
row_means = tensor.mean(dim=1)
print(f"Mean of each row: {row_means}")

# Max along columns
col_max, col_max_indices = tensor.max(dim=0)
print(f"Max of each column: {col_max}")
print(f"Indices of max values: {col_max_indices}")

# Output:
# Sum of all elements: 45
# Mean of each row: tensor([2., 5., 8.])
# Max of each column: tensor([7, 8, 9])
# Indices of max values: tensor([2, 2, 2])
```

Slide 10: Tensor Concatenation and Stacking

Combining multiple tensors is a common operation in deep learning, especially when working with batches of data or merging features.

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Concatenate tensors along rows
cat_rows = torch.cat((a, b), dim=0)
print("Concatenated along rows:")
print(cat_rows)

# Stack tensors to create a new dimension
stacked = torch.stack((a, b))
print("\nStacked tensors:")
print(stacked)

# Output:
# Concatenated along rows:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])
#
# Stacked tensors:
# tensor([[[1, 2],
#          [3, 4]],
#         [[5, 6],
#          [7, 8]]])
```

Slide 11: Tensor Transposition and Permutation

Changing the order of dimensions in a tensor is useful for various data processing tasks and model architectures.

```python
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Transpose a 2D tensor
transposed = tensor.t()
print("Transposed tensor:")
print(transposed)

# Permute dimensions of a 3D tensor
tensor_3d = torch.randn(2, 3, 4)
permuted = tensor_3d.permute(2, 0, 1)
print(f"\nOriginal shape: {tensor_3d.shape}")
print(f"Permuted shape: {permuted.shape}")

# Output:
# Transposed tensor:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])
#
# Original shape: torch.Size([2, 3, 4])
# Permuted shape: torch.Size([4, 2, 3])
```

Slide 12: Real-life Example: Image Processing with PyTorch

PyTorch tensors are commonly used in image processing tasks. Let's demonstrate how to load an image, convert it to a tensor, and apply a simple transformation.

```python
import torch
from torchvision import transforms
from PIL import Image

# Load an image
image = Image.open("example_image.jpg")

# Define a transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformation to convert the image to a tensor
image_tensor = transform(image)

print(f"Image tensor shape: {image_tensor.shape}")
print(f"Image tensor data type: {image_tensor.dtype}")

# Reverse the normalization for visualization
unnormalized = image_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
               torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

# Convert back to PIL Image for display
restored_image = transforms.ToPILImage()(unnormalized)
restored_image.show()

# Output:
# Image tensor shape: torch.Size([3, 224, 224])
# Image tensor data type: torch.float32
```

Slide 13: Real-life Example: Time Series Analysis with PyTorch

PyTorch tensors can be used for time series analysis. Let's create a simple example of generating a synthetic time series and calculating moving averages.

```python
import torch
import matplotlib.pyplot as plt

# Generate synthetic time series data
time = torch.arange(0, 100, 0.1)
amplitude = torch.sin(time) + 0.1 * torch.randn(time.size(0))

# Calculate moving average
window_size = 50
weights = torch.ones(window_size) / window_size
moving_avg = torch.conv1d(amplitude.view(1, 1, -1), weights.view(1, 1, -1), padding=window_size//2).squeeze()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time, amplitude, label='Original')
plt.plot(time, moving_avg, label='Moving Average')
plt.legend()
plt.title('Time Series Analysis with PyTorch')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

print(f"Time series shape: {amplitude.shape}")
print(f"Moving average shape: {moving_avg.shape}")

# Output:
# Time series shape: torch.Size([1000])
# Moving average shape: torch.Size([1000])
```

Slide 14: GPU Acceleration with PyTorch Tensors

One of PyTorch's strengths is its seamless GPU support. Moving tensors to GPU can significantly speed up computations, especially for large-scale operations.

```python
import torch
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a large tensor
tensor = torch.randn(10000, 10000)

# CPU computation
start_time = time.time()
result_cpu = torch.matmul(tensor, tensor.t())
cpu_time = time.time() - start_time
print(f"CPU computation time: {cpu_time:.4f} seconds")

# GPU computation (if available)
if torch.cuda.is_available():
    tensor_gpu = tensor.to(device)
    start_time = time.time()
    result_gpu = torch.matmul(tensor_gpu, tensor_gpu.t())
    torch.cuda.synchronize()  # Wait for GPU computation to finish
    gpu_time = time.time() - start_time
    print(f"GPU computation time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

# Output (results may vary):
# Using device: cuda
# CPU computation time: 2.3456 seconds
# GPU computation time: 0.1234 seconds
# Speedup: 19.01x
```

Slide 15: Additional Resources

For more in-depth information on PyTorch tensors and their applications, consider exploring the following resources:

1.  PyTorch official documentation: [https://pytorch.org/docs/stable/torch.html](https://pytorch.org/docs/stable/torch.html)
2.  "PyTorch: An Imperative Style, High-Performance Deep Learning Library" by Paszke et al. (2019): [https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)
3.  "Deep Learning with PyTorch" by Stevens et al. (2020): [https://pytorch.org/deep-learning-with-pytorch-book](https://pytorch.org/deep-learning-with-pytorch-book)

These resources provide comprehensive guides, tutorials, and research papers that can help you master PyTorch tensor operations and apply them to various machine learning tasks.

