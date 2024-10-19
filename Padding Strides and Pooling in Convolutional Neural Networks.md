## Padding Strides and Pooling in Convolutional Neural Networks

Slide 1: Introduction to CNNs and Feature Maps

Convolutional Neural Networks (CNNs) are a class of deep learning models particularly effective for image processing tasks. In CNNs, the size of the output feature map is determined by the size of the input feature map, the size of the kernel, and the stride. Let's explore these concepts with a simple example.

```python
import numpy as np

# Create a simple 5x5 input feature map
input_feature_map = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

# Define a 3x3 kernel
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Function to perform convolution
def convolve(input_map, kernel):
    output_size = input_map.shape[0] - kernel.shape[0] + 1
    output = np.zeros((output_size, output_size))
    for i in range(output_size):
        for j in range(output_size):
            output[i, j] = np.sum(input_map[i:i+3, j:j+3] * kernel)
    return output

# Perform convolution
output_feature_map = convolve(input_feature_map, kernel)
print("Output feature map:")
print(output_feature_map)
```

Slide 2: Understanding Padding in CNNs

Padding is a technique used to preserve the spatial dimensions of the input feature map. It involves adding extra pixels around the border of the input image. This allows CNNs to process edge pixels more effectively. Let's implement zero padding and observe its effect.

```python
def add_zero_padding(input_map, pad_size):
    padded = np.pad(input_map, pad_size, mode='constant')
    return padded

# Add zero padding to our input feature map
padded_input = add_zero_padding(input_feature_map, 1)

print("Original input:")
print(input_feature_map)
print("\nPadded input:")
print(padded_input)

# Perform convolution on padded input
output_with_padding = convolve(padded_input, kernel)
print("\nOutput with padding:")
print(output_with_padding)
```

Slide 3: Exploring Strides in CNNs

Strides determine the step size of the convolution filter as it moves across the input image. A stride of one means the filter moves one pixel at a time, while higher strides result in smaller output dimensions. Let's implement convolution with different strides.

```python
def convolve_with_stride(input_map, kernel, stride):
    output_size = (input_map.shape[0] - kernel.shape[0]) // stride + 1
    output = np.zeros((output_size, output_size))
    for i in range(0, input_map.shape[0] - kernel.shape[0] + 1, stride):
        for j in range(0, input_map.shape[1] - kernel.shape[1] + 1, stride):
            output[i//stride, j//stride] = np.sum(input_map[i:i+3, j:j+3] * kernel)
    return output

# Perform convolution with stride 1 and 2
output_stride_1 = convolve_with_stride(input_feature_map, kernel, 1)
output_stride_2 = convolve_with_stride(input_feature_map, kernel, 2)

print("Output with stride 1:")
print(output_stride_1)
print("\nOutput with stride 2:")
print(output_stride_2)
```

Slide 4: Pooling in CNNs

Pooling reduces the spatial dimensions of the input, aiding in dimensionality reduction, feature extraction, and controlling overfitting. Let's implement max pooling and average pooling.

```python
def max_pooling(input_map, pool_size):
    output_size = input_map.shape[0] // pool_size
    output = np.zeros((output_size, output_size))
    for i in range(0, input_map.shape[0], pool_size):
        for j in range(0, input_map.shape[1], pool_size):
            output[i//pool_size, j//pool_size] = np.max(input_map[i:i+pool_size, j:j+pool_size])
    return output

def avg_pooling(input_map, pool_size):
    output_size = input_map.shape[0] // pool_size
    output = np.zeros((output_size, output_size))
    for i in range(0, input_map.shape[0], pool_size):
        for j in range(0, input_map.shape[1], pool_size):
            output[i//pool_size, j//pool_size] = np.mean(input_map[i:i+pool_size, j:j+pool_size])
    return output

# Apply max pooling and average pooling
max_pooled = max_pooling(input_feature_map, 2)
avg_pooled = avg_pooling(input_feature_map, 2)

print("Original input:")
print(input_feature_map)
print("\nMax pooled output:")
print(max_pooled)
print("\nAverage pooled output:")
print(avg_pooled)
```

Slide 5: Calculating Output Dimensions

Understanding how to calculate the output dimensions of a convolutional layer is crucial. Let's create a function to compute these dimensions based on input size, kernel size, padding, and stride.

```python
def calculate_output_size(input_size, kernel_size, padding, stride):
    return (input_size - kernel_size + 2 * padding) // stride + 1

# Example calculations
input_size = 28
kernel_sizes = [3, 5, 7]
paddings = [0, 1, 2]
strides = [1, 2, 3]

for k in kernel_sizes:
    for p in paddings:
        for s in strides:
            output_size = calculate_output_size(input_size, k, p, s)
            print(f"Input: {input_size}, Kernel: {k}, Padding: {p}, Stride: {s} -> Output: {output_size}")
```

Slide 6: Visualizing Convolution Operation

To better understand how convolution works, let's create a visual representation of the process using ASCII art.

```python
def visualize_convolution(input_map, kernel, stride=1):
    input_size = len(input_map)
    kernel_size = len(kernel)
    output_size = (input_size - kernel_size) // stride + 1
    
    for i in range(0, input_size - kernel_size + 1, stride):
        for j in range(0, input_size - kernel_size + 1, stride):
            print(f"Step: ({i}, {j})")
            for x in range(input_size):
                for y in range(input_size):
                    if i <= x < i + kernel_size and j <= y < j + kernel_size:
                        print(f"[{input_map[x][y]:2d}]", end="")
                    else:
                        print(f" {input_map[x][y]:2d} ", end="")
                print()
            print()

# Example usage
input_map = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
]
kernel = [
    [1, 0],
    [0, 1]
]

visualize_convolution(input_map, kernel, stride=1)
```

Slide 7: Real-life Example: Edge Detection

Edge detection is a fundamental image processing technique often used in CNNs. Let's implement a simple edge detection algorithm using convolution.

```python
import numpy as np
from PIL import Image

def edge_detection(image_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # Define edge detection kernels
    horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Apply convolution
    horizontal_edges = convolve(image_array, horizontal_kernel)
    vertical_edges = convolve(image_array, vertical_kernel)

    # Combine edges
    edges = np.sqrt(horizontal_edges**2 + vertical_edges**2)
    edges = (edges / edges.max() * 255).astype(np.uint8)

    return Image.fromarray(edges)

# Example usage (you need to provide an image path)
# edge_image = edge_detection('path_to_your_image.jpg')
# edge_image.show()
```

Slide 8: Real-life Example: Image Blurring

Image blurring is another common operation in image processing that can be achieved using convolution. Let's implement a simple Gaussian blur.

```python
import numpy as np
from PIL import Image

def gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def blur_image(image_path, kernel_size=5, sigma=1):
    # Load image
    image = Image.open(image_path)
    image_array = np.array(image)

    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply convolution to each color channel
    blurred = np.zeros_like(image_array)
    for i in range(3):  # Assuming RGB image
        blurred[:,:,i] = convolve(image_array[:,:,i], kernel)

    return Image.fromarray(blurred.astype(np.uint8))

# Example usage (you need to provide an image path)
# blurred_image = blur_image('path_to_your_image.jpg')
# blurred_image.show()
```

Slide 9: Understanding the Impact of Stride

Let's visualize how different stride values affect the output of a convolution operation.

```python
import matplotlib.pyplot as plt

def visualize_stride_impact(input_size=8, kernel_size=3, strides=[1, 2, 3]):
    input_map = np.arange(1, input_size**2 + 1).reshape((input_size, input_size))
    kernel = np.ones((kernel_size, kernel_size))
    
    fig, axes = plt.subplots(1, len(strides), figsize=(15, 5))
    fig.suptitle("Impact of Stride on Convolution Output")
    
    for i, stride in enumerate(strides):
        output = convolve_with_stride(input_map, kernel, stride)
        axes[i].imshow(output, cmap='viridis')
        axes[i].set_title(f"Stride = {stride}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_stride_impact()
```

Slide 10: Padding Techniques

There are various padding techniques used in CNNs. Let's implement and visualize some common padding methods.

```python
def pad_image(image, pad_width, mode='constant'):
    if mode == 'constant':
        return np.pad(image, pad_width, mode='constant')
    elif mode == 'reflect':
        return np.pad(image, pad_width, mode='reflect')
    elif mode == 'symmetric':
        return np.pad(image, pad_width, mode='symmetric')
    else:
        raise ValueError("Unsupported padding mode")

def visualize_padding(image_size=5, pad_width=2):
    image = np.arange(1, image_size**2 + 1).reshape((image_size, image_size))
    
    padding_modes = ['constant', 'reflect', 'symmetric']
    fig, axes = plt.subplots(1, len(padding_modes), figsize=(15, 5))
    fig.suptitle("Different Padding Techniques")
    
    for i, mode in enumerate(padding_modes):
        padded = pad_image(image, pad_width, mode)
        axes[i].imshow(padded, cmap='viridis')
        axes[i].set_title(f"{mode.capitalize()} Padding")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_padding()
```

Slide 11: Pooling Visualization

Let's create a visual representation of how max pooling and average pooling affect an input feature map.

```python
def visualize_pooling(input_size=6, pool_size=2):
    input_map = np.random.randint(0, 10, (input_size, input_size))
    
    max_pooled = max_pooling(input_map, pool_size)
    avg_pooled = avg_pooling(input_map, pool_size)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pooling Visualization")
    
    axes[0].imshow(input_map, cmap='viridis')
    axes[0].set_title("Original Input")
    axes[0].axis('off')
    
    axes[1].imshow(max_pooled, cmap='viridis')
    axes[1].set_title("Max Pooling")
    axes[1].axis('off')
    
    axes[2].imshow(avg_pooled, cmap='viridis')
    axes[2].set_title("Average Pooling")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_pooling()
```

Slide 12: Receptive Field Calculation

The receptive field is the region in the input space that a particular CNN feature is looking at. Let's create a function to calculate the receptive field size for a given layer in a CNN.

```python
def calculate_receptive_field(layer_configs):
    n = len(layer_configs)
    receptive_field = 1
    for i in range(n-1, -1, -1):
        layer = layer_configs[i]
        receptive_field = layer['stride'] * receptive_field + (layer['kernel_size'] - layer['stride'])
    return receptive_field

# Example usage
layer_configs = [
    {'kernel_size': 3, 'stride': 1},
    {'kernel_size': 3, 'stride': 2},
    {'kernel_size': 3, 'stride': 1},
    {'kernel_size': 3, 'stride': 2},
]

receptive_field = calculate_receptive_field(layer_configs)
print(f"Receptive field size: {receptive_field}")
```

Slide 13: Impact of Kernel Size

The kernel size in a convolutional layer affects the amount of context captured from the input. Let's visualize how different kernel sizes impact the convolution output.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_kernel_impact(input_size=8, kernel_sizes=[3, 5, 7]):
    input_map = np.random.randint(0, 10, (input_size, input_size))
    
    fig, axes = plt.subplots(1, len(kernel_sizes) + 1, figsize=(20, 5))
    fig.suptitle("Impact of Kernel Size on Convolution Output")
    
    axes[0].imshow(input_map, cmap='viridis')
    axes[0].set_title("Input")
    axes[0].axis('off')
    
    for i, k_size in enumerate(kernel_sizes):
        kernel = np.ones((k_size, k_size)) / (k_size ** 2)
        output = convolve(input_map, kernel)
        axes[i+1].imshow(output, cmap='viridis')
        axes[i+1].set_title(f"Kernel Size: {k_size}x{k_size}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_kernel_impact()
```

Slide 14: Dilated Convolutions

Dilated convolutions, also known as atrous convolutions, expand the receptive field without increasing the number of parameters. Let's implement and visualize a dilated convolution.

```python
def dilated_convolve(input_map, kernel, dilation_rate):
    input_height, input_width = input_map.shape
    kernel_size = kernel.shape[0]
    output_size = input_height - (kernel_size - 1) * dilation_rate
    
    output = np.zeros((output_size, output_size))
    
    for i in range(output_size):
        for j in range(output_size):
            region = input_map[i:i+kernel_size*dilation_rate:dilation_rate, 
                               j:j+kernel_size*dilation_rate:dilation_rate]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example usage
input_map = np.random.randint(0, 10, (10, 10))
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
dilated_output = dilated_convolve(input_map, kernel, dilation_rate=2)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(input_map, cmap='viridis')
plt.title("Input")
plt.subplot(122)
plt.imshow(dilated_output, cmap='viridis')
plt.title("Dilated Convolution Output")
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of CNNs, padding, strides, and pooling, consider these peer-reviewed resources:

1.  "A guide to convolution arithmetic for deep learning" by Vincent Dumoulin and Francesco Visin (2016) ArXiv: [https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)
2.  "Rethinking Atrous Convolution for Semantic Image Segmentation" by Liang-Chieh Chen et al. (2017) ArXiv: [https://arxiv.org/abs/1706.05587](https://arxiv.org/abs/1706.05587)
3.  "Network In Network" by Min Lin et al. (2013) ArXiv: [https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400)

These papers provide in-depth discussions on various aspects of CNNs and their components.

