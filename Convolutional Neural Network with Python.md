## Convolutional Neural Network with Python
Slide 1: Introduction to Convolutional Operations

Convolutional operations are fundamental building blocks in neural networks, particularly in computer vision tasks. They enable the network to learn spatial hierarchies of features, making them highly effective for image processing and recognition tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Define a 3x3 kernel for edge detection
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Display the image and kernel
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(kernel, cmap='gray')
ax2.set_title('Edge Detection Kernel')
plt.show()
```

Slide 2: Basic Convolution Operation

A convolution operation slides a kernel (small matrix) over an input image, performing element-wise multiplication and summing the results. This process extracts features from the input data.

```python
def convolve2d(image, kernel):
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate output dimensions
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform convolution
    for i in range(o_height):
        for j in range(o_width):
            output[i, j] = np.sum(image[i:i+k_height, j:j+k_width] * kernel)
    
    return output

# Apply convolution
result = convolve2d(image, kernel)

# Display the result
plt.imshow(result, cmap='gray')
plt.title('Convolution Result')
plt.colorbar()
plt.show()
```

Slide 3: Padding in Convolutions

Padding adds extra pixels around the input image to control the output size and preserve information at the edges. Common types include 'valid' (no padding) and 'same' (output size equals input size).

```python
def pad_image(image, pad_width, mode='constant'):
    return np.pad(image, pad_width, mode=mode)

# Add padding to our image
padded_image = pad_image(image, pad_width=1)

# Display original and padded images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(padded_image, cmap='gray')
ax2.set_title('Padded Image')
plt.show()

# Apply convolution to padded image
padded_result = convolve2d(padded_image, kernel)

# Display the result
plt.imshow(padded_result, cmap='gray')
plt.title('Convolution Result with Padding')
plt.colorbar()
plt.show()
```

Slide 4: Strided Convolutions

Strided convolutions control how the kernel moves across the input. A stride of 1 moves the kernel one pixel at a time, while larger strides skip pixels, reducing the output size.

```python
def strided_convolve2d(image, kernel, stride=1):
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate output dimensions
    o_height = (i_height - k_height) // stride + 1
    o_width = (i_width - k_width) // stride + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform strided convolution
    for i in range(0, o_height):
        for j in range(0, o_width):
            output[i, j] = np.sum(
                image[i*stride:i*stride+k_height, j*stride:j*stride+k_width] * kernel
            )
    
    return output

# Apply strided convolution
strided_result = strided_convolve2d(image, kernel, stride=2)

# Display the result
plt.imshow(strided_result, cmap='gray')
plt.title('Strided Convolution Result (stride=2)')
plt.colorbar()
plt.show()
```

Slide 5: Multiple Input Channels

Real-world images often have multiple channels (e.g., RGB). Convolutions can be applied to multi-channel inputs by using 3D kernels that span all input channels.

```python
# Create a simple 5x5x3 RGB image
rgb_image = np.zeros((5, 5, 3))
rgb_image[1:4, 1:4, 0] = 1  # Red square
rgb_image[2, 2, 1] = 1  # Green center

# Define a 3x3x3 kernel for edge detection in RGB
rgb_kernel = np.array([
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
])

def convolve3d(image, kernel):
    # Get dimensions
    i_height, i_width, i_channels = image.shape
    k_height, k_width, k_channels = kernel.shape
    
    # Calculate output dimensions
    o_height = i_height - k_height + 1
    o_width = i_width - k_width + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform 3D convolution
    for i in range(o_height):
        for j in range(o_width):
            output[i, j] = np.sum(image[i:i+k_height, j:j+k_width, :] * kernel)
    
    return output

# Apply 3D convolution
rgb_result = convolve3d(rgb_image, rgb_kernel)

# Display the original image and result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(rgb_image)
ax1.set_title('Original RGB Image')
ax2.imshow(rgb_result, cmap='gray')
ax2.set_title('3D Convolution Result')
plt.show()
```

Slide 6: Multiple Output Channels

Convolutional layers often use multiple kernels to create multiple output channels, each capturing different features of the input.

```python
def multi_kernel_convolve2d(image, kernels):
    return np.array([convolve2d(image, kernel) for kernel in kernels])

# Define multiple kernels
kernels = [
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),  # Edge detection
    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),      # Sharpen
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9       # Blur
]

# Apply multi-kernel convolution
multi_result = multi_kernel_convolve2d(image, kernels)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (ax, res) in enumerate(zip(axes, multi_result)):
    ax.imshow(res, cmap='gray')
    ax.set_title(f'Kernel {i+1} Result')
plt.show()
```

Slide 7: Activation Functions in Convolutional Layers

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common choices include ReLU, sigmoid, and tanh.

```python
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Apply convolution and different activation functions
conv_result = convolve2d(image, kernel)

relu_result = relu(conv_result)
sigmoid_result = sigmoid(conv_result)
tanh_result = tanh(conv_result)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(conv_result, cmap='gray')
axes[0, 0].set_title('Convolution Result')
axes[0, 1].imshow(relu_result, cmap='gray')
axes[0, 1].set_title('ReLU Activation')
axes[1, 0].imshow(sigmoid_result, cmap='gray')
axes[1, 0].set_title('Sigmoid Activation')
axes[1, 1].imshow(tanh_result, cmap='gray')
axes[1, 1].set_title('Tanh Activation')
plt.tight_layout()
plt.show()
```

Slide 8: Pooling Operations

Pooling reduces the spatial dimensions of the data, helping to control overfitting and reduce computational cost. Common types include max pooling and average pooling.

```python
def pool2d(image, pool_size, mode='max'):
    # Get dimensions
    i_height, i_width = image.shape
    p_height, p_width = pool_size
    
    # Calculate output dimensions
    o_height = i_height // p_height
    o_width = i_width // p_width
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform pooling
    for i in range(o_height):
        for j in range(o_width):
            if mode == 'max':
                output[i, j] = np.max(image[i*p_height:(i+1)*p_height, 
                                            j*p_width:(j+1)*p_width])
            elif mode == 'avg':
                output[i, j] = np.mean(image[i*p_height:(i+1)*p_height, 
                                             j*p_width:(j+1)*p_width])
    
    return output

# Apply max and average pooling
max_pooled = pool2d(image, (2, 2), mode='max')
avg_pooled = pool2d(image, (2, 2), mode='avg')

# Display results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(max_pooled, cmap='gray')
ax2.set_title('Max Pooling')
ax3.imshow(avg_pooled, cmap='gray')
ax3.set_title('Average Pooling')
plt.show()
```

Slide 9: Implementing a Simple Convolutional Layer

A convolutional layer combines convolution, activation, and often pooling operations. Here's a basic implementation of a convolutional layer with ReLU activation.

```python
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and biases
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.biases = np.zeros(out_channels)
    
    def forward(self, input):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height + 2*self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input if necessary
        if self.padding > 0:
            input = np.pad(input, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                input_slice = input[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for k in range(self.out_channels):
                    output[:, k, i, j] = np.sum(input_slice * self.weights[k], axis=(1,2,3)) + self.biases[k]
        
        # Apply ReLU activation
        return np.maximum(0, output)

# Create a sample input
input = np.random.randn(1, 3, 32, 32)  # 1 image, 3 channels, 32x32 pixels

# Create and apply a convolutional layer
conv_layer = ConvLayer(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
output = conv_layer.forward(input)

print(f"Input shape: {input.shape}")
print(f"Output shape: {output.shape}")

# Visualize one of the output channels
plt.imshow(output[0, 0], cmap='gray')
plt.title('First Channel of Convolutional Layer Output')
plt.colorbar()
plt.show()
```

Slide 10: Real-life Example: Edge Detection in Images

Edge detection is a fundamental operation in computer vision, used in applications like object recognition and image segmentation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Create a sample image
image = np.zeros((100, 100))
image[20:80, 20:80] = 1  # White square on black background

# Define edge detection kernels
horizontal_kernel = np.array([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]])

vertical_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

# Apply edge detection
horizontal_edges = convolve2d(image, horizontal_kernel, mode='same', boundary='symm')
vertical_edges = convolve2d(image, vertical_kernel, mode='same', boundary='symm')
edges = np.sqrt(horizontal_edges**2 + vertical_edges**2)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(horizontal_edges, cmap='gray')
axes[0, 1].set_title('Horizontal Edges')
axes[1, 0].imshow(vertical_edges, cmap='gray')
axes[1, 0].set_title('Vertical Edges')
axes[1, 1].imshow(edges, cmap='gray')
axes[1, 1].set_title('Combined Edges')
plt.tight_layout()
plt.show()
```

Slide 11: Real-life Example: Image Blurring

Image blurring is commonly used for noise reduction and creating special effects in image processing.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Create a noisy image
np.random.seed(0)
image = np.random.rand(100, 100)

# Define a Gaussian blur kernel
kernel_size = 5
sigma = 1.0
x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
gaussian_kernel = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

# Apply blurring
blurred_image = convolve2d(image, gaussian_kernel, mode='same', boundary='symm')

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Noisy Image')
ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Blurred Image')
plt.tight_layout()
plt.show()
```

Slide 12: Dilated Convolutions

Dilated convolutions expand the receptive field without increasing the number of parameters, useful for capturing multi-scale context in tasks like semantic segmentation.

```python
import numpy as np
import matplotlib.pyplot as plt

def dilated_convolve2d(image, kernel, dilation_rate):
    # Get dimensions
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate effective kernel size
    eff_k_height = k_height + (k_height - 1) * (dilation_rate - 1)
    eff_k_width = k_width + (k_width - 1) * (dilation_rate - 1)
    
    # Calculate output dimensions
    o_height = i_height - eff_k_height + 1
    o_width = i_width - eff_k_width + 1
    
    # Initialize output
    output = np.zeros((o_height, o_width))
    
    # Perform dilated convolution
    for i in range(o_height):
        for j in range(o_width):
            for m in range(k_height):
                for n in range(k_width):
                    output[i, j] += image[i + m * dilation_rate, j + n * dilation_rate] * kernel[m, n]
    
    return output

# Create a sample image and kernel
image = np.zeros((15, 15))
image[5:10, 5:10] = 1
kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

# Apply dilated convolutions with different dilation rates
conv_normal = dilated_convolve2d(image, kernel, dilation_rate=1)
conv_dilated_2 = dilated_convolve2d(image, kernel, dilation_rate=2)
conv_dilated_3 = dilated_convolve2d(image, kernel, dilation_rate=3)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(conv_normal, cmap='gray')
axes[0, 1].set_title('Normal Convolution')
axes[1, 0].imshow(conv_dilated_2, cmap='gray')
axes[1, 0].set_title('Dilated Convolution (rate=2)')
axes[1, 1].imshow(conv_dilated_3, cmap='gray')
axes[1, 1].set_title('Dilated Convolution (rate=3)')
plt.tight_layout()
plt.show()
```

Slide 13: Depthwise Separable Convolutions

Depthwise separable convolutions factorize a standard convolution into a depthwise convolution followed by a pointwise convolution, reducing computational cost.

```python
import numpy as np
import matplotlib.pyplot as plt

def depthwise_conv(input, kernels):
    depth = input.shape[2]
    output = np.zeros_like(input)
    for d in range(depth):
        output[:,:,d] = convolve2d(input[:,:,d], kernels[d], mode='same')
    return output

def pointwise_conv(input, kernel):
    depth = input.shape[2]
    output = np.sum(input * kernel, axis=2)
    return output

# Create a sample input
input = np.random.rand(10, 10, 3)

# Create depthwise kernels and pointwise kernel
depthwise_kernels = np.random.rand(3, 3, 3)
pointwise_kernel = np.random.rand(1, 1, 3)

# Apply depthwise separable convolution
depthwise_output = depthwise_conv(input, depthwise_kernels)
final_output = pointwise_conv(depthwise_output, pointwise_kernel)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(input[:,:,0], cmap='gray')
axes[0].set_title('Input (Channel 0)')
axes[1].imshow(depthwise_output[:,:,0], cmap='gray')
axes[1].set_title('Depthwise Output (Channel 0)')
axes[2].imshow(final_output, cmap='gray')
axes[2].set_title('Final Output')
plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For more in-depth information on convolutional neural networks and their applications, consider exploring these resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2. "Convolutional Neural Networks for Visual Recognition" course by Stanford University (CS231n)
3. ArXiv paper: "A Survey of the Recent Architectures of Deep Convolutional Neural Networks" by Asifullah Khan et al. (arXiv:1901.06032)
4. ArXiv paper: "A guide to convolution arithmetic for deep learning" by Vincent Dumoulin and Francesco Visin (arXiv:1603.07285)
5. TensorFlow and PyTorch documentation on convolutional layers and operations

These resources provide comprehensive coverage of convolutional neural networks, from basic concepts to advanced architectures and applications.

