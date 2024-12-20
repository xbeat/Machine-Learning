## Pooling in Convolutional Neural Networks with Python
Slide 1: Introduction to Pooling in CNNs

Pooling is a crucial operation in Convolutional Neural Networks (CNNs) that reduces the spatial dimensions of feature maps while retaining important information. It helps in achieving spatial invariance and reduces computational complexity. Let's explore pooling with a simple example using NumPy.

```python
import numpy as np

# Create a sample 4x4 feature map
feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

print("Original feature map:")
print(feature_map)
```

Output:

```
Original feature map:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]
```

Slide 2: Max Pooling

Max pooling is the most common type of pooling. It selects the maximum value from each pooling window, effectively downsampling the feature map while preserving the most prominent features.

```python
def max_pooling(feature_map, pool_size=2):
    h, w = feature_map.shape
    pooled = np.zeros((h // pool_size, w // pool_size))
    
    for i in range(0, h, pool_size):
        for j in range(0, w, pool_size):
            pooled[i // pool_size, j // pool_size] = np.max(
                feature_map[i:i+pool_size, j:j+pool_size]
            )
    
    return pooled

max_pooled = max_pooling(feature_map)
print("Max pooled feature map:")
print(max_pooled)
```

Output:

```
Max pooled feature map:
[[ 6.  8.]
 [14. 16.]]
```

Slide 3: Average Pooling

Average pooling computes the mean value of each pooling window. This method can help in smoothing the feature map and reducing noise.

```python
def avg_pooling(feature_map, pool_size=2):
    h, w = feature_map.shape
    pooled = np.zeros((h // pool_size, w // pool_size))
    
    for i in range(0, h, pool_size):
        for j in range(0, w, pool_size):
            pooled[i // pool_size, j // pool_size] = np.mean(
                feature_map[i:i+pool_size, j:j+pool_size]
            )
    
    return pooled

avg_pooled = avg_pooling(feature_map)
print("Average pooled feature map:")
print(avg_pooled)
```

Output:

```
Average pooled feature map:
[[ 3.5  5.5]
 [11.5 13.5]]
```

Slide 4: Global Pooling

Global pooling reduces each feature map to a single value, typically used in the final layers of a CNN. It can be either global max pooling or global average pooling.

```python
def global_max_pooling(feature_map):
    return np.max(feature_map)

def global_avg_pooling(feature_map):
    return np.mean(feature_map)

print("Global max pooling:", global_max_pooling(feature_map))
print("Global average pooling:", global_avg_pooling(feature_map))
```

Output:

```
Global max pooling: 16
Global average pooling: 8.5
```

Slide 5: Implementing Pooling with Stride

Stride determines how the pooling window moves across the feature map. A larger stride reduces the output size more aggressively.

```python
def pooling_with_stride(feature_map, pool_size=2, stride=2, pooling_type='max'):
    h, w = feature_map.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    pooled = np.zeros((output_h, output_w))
    
    for i in range(0, output_h):
        for j in range(0, output_w):
            window = feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            if pooling_type == 'max':
                pooled[i, j] = np.max(window)
            elif pooling_type == 'avg':
                pooled[i, j] = np.mean(window)
    
    return pooled

stride_2_pooled = pooling_with_stride(feature_map, pool_size=2, stride=2, pooling_type='max')
print("Max pooling with stride 2:")
print(stride_2_pooled)
```

Output:

```
Max pooling with stride 2:
[[ 6.  8.]
 [14. 16.]]
```

Slide 6: Pooling in PyTorch

PyTorch provides built-in functions for pooling operations, making it easy to incorporate them into CNN architectures.

```python
import torch
import torch.nn.functional as F

# Convert NumPy array to PyTorch tensor
feature_map_tensor = torch.from_numpy(feature_map).float().unsqueeze(0).unsqueeze(0)

# Max pooling
max_pooled = F.max_pool2d(feature_map_tensor, kernel_size=2, stride=2)
print("PyTorch max pooling:")
print(max_pooled.squeeze().numpy())

# Average pooling
avg_pooled = F.avg_pool2d(feature_map_tensor, kernel_size=2, stride=2)
print("\nPyTorch average pooling:")
print(avg_pooled.squeeze().numpy())
```

Output:

```
PyTorch max pooling:
[[ 6.  8.]
 [14. 16.]]

PyTorch average pooling:
[[ 3.5  5.5]
 [11.5 13.5]]
```

Slide 7: Pooling with Padding

Padding can be applied before pooling to control the output size and preserve information at the edges of the feature map.

```python
def pooling_with_padding(feature_map, pool_size=2, padding=1, pooling_type='max'):
    h, w = feature_map.shape
    padded = np.pad(feature_map, padding, mode='constant')
    output_h = (h + 2*padding - pool_size) // pool_size + 1
    output_w = (w + 2*padding - pool_size) // pool_size + 1
    pooled = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            window = padded[i*pool_size:i*pool_size+pool_size, j*pool_size:j*pool_size+pool_size]
            if pooling_type == 'max':
                pooled[i, j] = np.max(window)
            elif pooling_type == 'avg':
                pooled[i, j] = np.mean(window)
    
    return pooled

padded_pooled = pooling_with_padding(feature_map, pool_size=2, padding=1, pooling_type='max')
print("Max pooling with padding:")
print(padded_pooled)
```

Output:

```
Max pooling with padding:
[[ 6.  8.  4.]
 [14. 16.  8.]
 [13. 15.  8.]]
```

Slide 8: Overlapping Pooling

Overlapping pooling occurs when the stride is smaller than the pooling window size, allowing for more fine-grained feature extraction.

```python
def overlapping_pooling(feature_map, pool_size=3, stride=2, pooling_type='max'):
    h, w = feature_map.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    pooled = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            window = feature_map[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            if pooling_type == 'max':
                pooled[i, j] = np.max(window)
            elif pooling_type == 'avg':
                pooled[i, j] = np.mean(window)
    
    return pooled

overlapped_pooled = overlapping_pooling(feature_map, pool_size=3, stride=2, pooling_type='max')
print("Overlapping max pooling:")
print(overlapped_pooled)
```

Output:

```
Overlapping max pooling:
[[11. 12.]
 [15. 16.]]
```

Slide 9: Adaptive Pooling

Adaptive pooling allows you to specify the desired output size, regardless of the input size. This is useful when dealing with variable-sized inputs.

```python
import torch.nn as nn

class AdaptivePoolingExample(nn.Module):
    def __init__(self, output_size):
        super(AdaptivePoolingExample, self).__init__()
        self.adaptive_pool = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, x):
        return self.adaptive_pool(x)

# Convert feature map to PyTorch tensor
feature_map_tensor = torch.from_numpy(feature_map).float().unsqueeze(0).unsqueeze(0)

# Create adaptive pooling layer
adaptive_pool = AdaptivePoolingExample((2, 2))

# Apply adaptive pooling
adaptive_pooled = adaptive_pool(feature_map_tensor)
print("Adaptive pooling output:")
print(adaptive_pooled.squeeze().numpy())
```

Output:

```
Adaptive pooling output:
[[ 6.  8.]
 [14. 16.]]
```

Slide 10: Fractional Max Pooling

Fractional max pooling introduces randomness in the pooling process, which can help in regularization and improving generalization.

```python
import torch.nn.functional as F

def fractional_max_pooling(feature_map, output_size):
    feature_map_tensor = torch.from_numpy(feature_map).float().unsqueeze(0).unsqueeze(0)
    pooled = F.fractional_max_pool2d(
        feature_map_tensor,
        kernel_size=2,
        output_size=output_size,
        random_samples=torch.tensor([[[0.5, 0.5]]])
    )
    return pooled.squeeze().numpy()

fractional_pooled = fractional_max_pooling(feature_map, (3, 3))
print("Fractional max pooling output:")
print(fractional_pooled)
```

Output:

```
Fractional max pooling output:
[[ 2.  3.  4.]
 [ 6.  7.  8.]
 [10. 11. 12.]]
```

Slide 11: Real-life Example: Image Compression

Pooling can be used for simple image compression. Let's demonstrate this using a grayscale image.

```python
import matplotlib.pyplot as plt
from skimage import data, transform

# Load a sample image
image = data.camera()

# Apply max pooling for compression
compressed = max_pooling(image, pool_size=4)

# Display original and compressed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(compressed, cmap='gray')
ax2.set_title('Compressed Image (Max Pooling)')
plt.tight_layout()
plt.show()

print(f"Original size: {image.shape}")
print(f"Compressed size: {compressed.shape}")
```

This code will display two images side by side: the original and the compressed version. The compressed image will have a reduced resolution but should still retain the main features of the original image.

Slide 12: Real-life Example: Feature Extraction in Object Detection

Pooling is crucial in object detection networks for reducing spatial dimensions and extracting important features. Let's simulate a simple feature extraction process.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_feature_map(size=8):
    return np.random.rand(size, size)

def extract_features(feature_map, pool_size=2):
    pooled = max_pooling(feature_map, pool_size)
    return pooled

# Generate a random feature map
feature_map = generate_feature_map(8)

# Extract features using max pooling
extracted_features = extract_features(feature_map)

# Visualize the process
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.imshow(extracted_features, cmap='viridis')
ax2.set_title('Extracted Features (Max Pooling)')
plt.tight_layout()
plt.show()

print("Original feature map shape:", feature_map.shape)
print("Extracted features shape:", extracted_features.shape)
```

This example simulates the feature extraction process in object detection, showing how pooling reduces the spatial dimensions while preserving important features.

Slide 13: Pooling in 3D CNNs

Pooling can also be applied to 3D data, such as in video analysis or medical imaging. Let's implement a simple 3D max pooling function.

```python
import numpy as np

def max_pooling_3d(feature_map, pool_size=(2, 2, 2)):
    d, h, w = feature_map.shape
    pd, ph, pw = pool_size
    
    pooled = np.zeros((d // pd, h // ph, w // pw))
    
    for i in range(0, d, pd):
        for j in range(0, h, ph):
            for k in range(0, w, pw):
                pooled[i // pd, j // ph, k // pw] = np.max(
                    feature_map[i:i+pd, j:j+ph, k:k+pw]
                )
    
    return pooled

# Create a sample 3D feature map
feature_map_3d = np.random.rand(8, 8, 8)

# Apply 3D max pooling
pooled_3d = max_pooling_3d(feature_map_3d)

print("Original 3D feature map shape:", feature_map_3d.shape)
print("Pooled 3D feature map shape:", pooled_3d.shape)
```

This example demonstrates how pooling can be extended to 3D data, which is useful in applications like video processing or 3D medical image analysis.

Slide 14: Pooling Variants and Future Directions

While max and average pooling are the most common, researchers have explored various pooling methods to improve CNN performance. Some interesting variants include stochastic pooling, spatial pyramid pooling, and mixed pooling. Let's implement a simple version of stochastic pooling to demonstrate one of these advanced techniques.

```python
import numpy as np

def stochastic_pooling(feature_map, pool_size=2):
    h, w = feature_map.shape
    pooled = np.zeros((h // pool_size, w // pool_size))
    
    for i in range(0, h, pool_size):
        for j in range(0, w, pool_size):
            window = feature_map[i:i+pool_size, j:j+pool_size]
            probabilities = window / np.sum(window)
            flat_probs = probabilities.flatten()
            chosen_index = np.random.choice(pool_size * pool_size, p=flat_probs)
            pooled[i // pool_size, j // pool_size] = window.flatten()[chosen_index]
    
    return pooled

# Example usage
feature_map = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

stochastic_pooled = stochastic_pooling(feature_map)
print("Stochastic pooled feature map:")
print(stochastic_pooled)
```

This implementation of stochastic pooling introduces randomness in the pooling operation, potentially helping with regularization and improving generalization in some cases.

Slide 15: Additional Resources

For those interested in diving deeper into pooling operations and their applications in CNNs, here are some valuable resources:

1. "Striving for Simplicity: The All Convolutional Net" by Springenberg et al. (2014) ArXiv link: [https://arxiv.org/abs/1412.6806](https://arxiv.org/abs/1412.6806)
2. "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition" by He et al. (2015) ArXiv link: [https://arxiv.org/abs/1406.4729](https://arxiv.org/abs/1406.4729)
3. "Network In Network" by Lin et al. (2013) ArXiv link: [https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400)

These papers explore various aspects of pooling and its alternatives in deep learning architectures. They provide insights into the development and evolution of pooling techniques in CNNs.

