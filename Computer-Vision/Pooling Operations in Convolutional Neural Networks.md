## Pooling Operations in Convolutional Neural Networks
Slide 1: Introduction to Pooling in Convolutional Neural Networks

Pooling is a crucial operation in Convolutional Neural Networks (CNNs) that reduces the spatial dimensions of feature maps while retaining important information. It helps in achieving spatial invariance and reduces computational complexity.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_pooling(input_matrix, pool_size):
    plt.imshow(input_matrix, cmap='viridis')
    plt.title("Input Matrix")
    plt.colorbar()
    plt.show()
    
    rows, cols = input_matrix.shape
    for i in range(0, rows, pool_size):
        for j in range(0, cols, pool_size):
            plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), pool_size, pool_size, 
                                fill=False, edgecolor='red', linewidth=2))
    
    plt.imshow(input_matrix, cmap='viridis')
    plt.title("Pooling Regions")
    plt.colorbar()
    plt.show()

input_matrix = np.random.rand(6, 6)
visualize_pooling(input_matrix, pool_size=2)
```

Slide 2: Max Pooling

Max pooling is the most common type of pooling operation. It selects the maximum value from each pooling region, effectively reducing the spatial dimensions while preserving the most prominent features.

```python
import numpy as np

def max_pooling(input_matrix, pool_size):
    rows, cols = input_matrix.shape
    pooled_rows = rows // pool_size
    pooled_cols = cols // pool_size
    
    output = np.zeros((pooled_rows, pooled_cols))
    
    for i in range(pooled_rows):
        for j in range(pooled_cols):
            pool = input_matrix[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(pool)
    
    return output

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

pooled = max_pooling(input_matrix, pool_size=2)
print("Input matrix:")
print(input_matrix)
print("\nPooled matrix:")
print(pooled)
```

Slide 3: Average Pooling

Average pooling computes the mean value of each pooling region. This operation smooths the feature maps and can be useful for certain types of data or architectures.

```python
import numpy as np

def average_pooling(input_matrix, pool_size):
    rows, cols = input_matrix.shape
    pooled_rows = rows // pool_size
    pooled_cols = cols // pool_size
    
    output = np.zeros((pooled_rows, pooled_cols))
    
    for i in range(pooled_rows):
        for j in range(pooled_cols):
            pool = input_matrix[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            output[i, j] = np.mean(pool)
    
    return output

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

pooled = average_pooling(input_matrix, pool_size=2)
print("Input matrix:")
print(input_matrix)
print("\nPooled matrix:")
print(pooled)
```

Slide 4: Global Pooling

Global pooling reduces each feature map to a single value, typically used in the final layers of a CNN. It can be either global max pooling or global average pooling.

```python
import numpy as np

def global_max_pooling(input_matrix):
    return np.max(input_matrix)

def global_average_pooling(input_matrix):
    return np.mean(input_matrix)

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

global_max = global_max_pooling(input_matrix)
global_avg = global_average_pooling(input_matrix)

print("Input matrix:")
print(input_matrix)
print(f"\nGlobal Max Pooling: {global_max}")
print(f"Global Average Pooling: {global_avg}")
```

Slide 5: Pooling with Stride

Stride in pooling determines the step size between pooling regions. It affects the output size and can be used to control the amount of downsampling.

```python
import numpy as np

def pooling_with_stride(input_matrix, pool_size, stride, pooling_type='max'):
    rows, cols = input_matrix.shape
    pooled_rows = (rows - pool_size) // stride + 1
    pooled_cols = (cols - pool_size) // stride + 1
    
    output = np.zeros((pooled_rows, pooled_cols))
    
    for i in range(pooled_rows):
        for j in range(pooled_cols):
            pool = input_matrix[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            if pooling_type == 'max':
                output[i, j] = np.max(pool)
            elif pooling_type == 'average':
                output[i, j] = np.mean(pool)
    
    return output

input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

pooled = pooling_with_stride(input_matrix, pool_size=2, stride=1, pooling_type='max')
print("Input matrix:")
print(input_matrix)
print("\nPooled matrix (Max pooling with stride 1):")
print(pooled)
```

Slide 6: Pooling in PyTorch

PyTorch provides built-in functions for various pooling operations, making it easy to incorporate them into neural network architectures.

```python
import torch
import torch.nn as nn

# Create a random input tensor
input_tensor = torch.randn(1, 1, 6, 6)

# Max Pooling
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
max_pooled = max_pool(input_tensor)

# Average Pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
avg_pooled = avg_pool(input_tensor)

# Global Average Pooling
global_avg_pool = nn.AdaptiveAvgPool2d(1)
global_avg_pooled = global_avg_pool(input_tensor)

print("Input tensor shape:", input_tensor.shape)
print("Max pooled shape:", max_pooled.shape)
print("Average pooled shape:", avg_pooled.shape)
print("Global average pooled shape:", global_avg_pooled.shape)
```

Slide 7: Pooling in TensorFlow/Keras

TensorFlow and Keras also provide easy-to-use pooling layers for building CNNs.

```python
import tensorflow as tf

# Create a random input tensor
input_tensor = tf.random.normal([1, 6, 6, 1])

# Max Pooling
max_pooled = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(input_tensor)

# Average Pooling
avg_pooled = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(input_tensor)

# Global Average Pooling
global_avg_pooled = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)

print("Input tensor shape:", input_tensor.shape)
print("Max pooled shape:", max_pooled.shape)
print("Average pooled shape:", avg_pooled.shape)
print("Global average pooled shape:", global_avg_pooled.shape)
```

Slide 8: Pooling and Feature Maps

Pooling operations help in reducing the spatial dimensions of feature maps while preserving important information. This visualization demonstrates how pooling affects feature maps.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_feature_map(size):
    return np.random.rand(size, size)

def visualize_pooling_effect(feature_map, pool_size):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(feature_map, cmap='viridis')
    ax1.set_title("Original Feature Map")
    ax1.axis('off')
    
    pooled = max_pooling(feature_map, pool_size)
    ax2.imshow(pooled, cmap='viridis')
    ax2.set_title(f"After Max Pooling (size={pool_size})")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

feature_map = create_feature_map(8)
visualize_pooling_effect(feature_map, pool_size=2)
```

Slide 9: Overlapping vs. Non-overlapping Pooling

Pooling can be performed with or without overlap between pooling regions. Overlapping pooling can sometimes lead to better performance but at the cost of increased computation.

```python
import numpy as np
import matplotlib.pyplot as plt

def pooling_with_overlap(input_matrix, pool_size, stride, pooling_type='max'):
    rows, cols = input_matrix.shape
    pooled_rows = (rows - pool_size) // stride + 1
    pooled_cols = (cols - pool_size) // stride + 1
    
    output = np.zeros((pooled_rows, pooled_cols))
    
    for i in range(pooled_rows):
        for j in range(pooled_cols):
            pool = input_matrix[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            if pooling_type == 'max':
                output[i, j] = np.max(pool)
            elif pooling_type == 'average':
                output[i, j] = np.mean(pool)
    
    return output

input_matrix = np.random.rand(6, 6)

non_overlapping = pooling_with_overlap(input_matrix, pool_size=2, stride=2)
overlapping = pooling_with_overlap(input_matrix, pool_size=3, stride=2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(input_matrix, cmap='viridis')
ax1.set_title("Original")
ax1.axis('off')

ax2.imshow(non_overlapping, cmap='viridis')
ax2.set_title("Non-overlapping (2x2, stride 2)")
ax2.axis('off')

ax3.imshow(overlapping, cmap='viridis')
ax3.set_title("Overlapping (3x3, stride 2)")
ax3.axis('off')

plt.tight_layout()
plt.show()
```

Slide 10: Pooling and Translation Invariance

Pooling operations contribute to translation invariance in CNNs, making the network less sensitive to small spatial shifts in the input.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_shifted_images(base_image, shifts):
    return [np.roll(base_image, shift, axis=(0, 1)) for shift in shifts]

def apply_pooling_to_images(images, pool_size):
    return [max_pooling(img, pool_size) for img in images]

base_image = np.zeros((8, 8))
base_image[2:6, 2:6] = 1

shifts = [(0, 0), (1, 1), (-1, -1)]
shifted_images = create_shifted_images(base_image, shifts)
pooled_images = apply_pooling_to_images(shifted_images, pool_size=2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (original, pooled) in enumerate(zip(shifted_images, pooled_images)):
    axes[0, i].imshow(original, cmap='gray')
    axes[0, i].set_title(f"Original (shift={shifts[i]})")
    axes[0, i].axis('off')
    
    axes[1, i].imshow(pooled, cmap='gray')
    axes[1, i].set_title(f"Pooled (shift={shifts[i]})")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: Pooling and Hierarchical Feature Learning

Pooling helps in creating a hierarchical representation of features in CNNs, allowing the network to capture increasingly abstract patterns as we go deeper.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_hierarchical_features(size, levels):
    features = []
    for i in range(levels):
        feature = np.zeros((size, size))
        feature[size//(2**i):size//(2**(i-1)), size//(2**i):size//(2**(i-1))] = 1
        features.append(feature)
    return features

def visualize_hierarchical_features(features):
    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))
    
    for i, feature in enumerate(features):
        axes[i].imshow(feature, cmap='gray')
        axes[i].set_title(f"Level {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

hierarchical_features = create_hierarchical_features(size=32, levels=3)
visualize_hierarchical_features(hierarchical_features)
```

Slide 12: Real-life Example: Image Classification

In image classification tasks, pooling helps reduce the spatial dimensions of feature maps, allowing the network to focus on the most important features while reducing computational complexity. Here's a simplified example of how pooling is used in a Convolutional Neural Network (CNN) for image classification:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

# Assume we have an image loaded as 'input_tensor'
model = SimpleCNN()
output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Real-life Example: Facial Recognition

Pooling operations play a crucial role in facial recognition systems by helping to extract key facial features while maintaining some spatial invariance. This allows the network to recognize faces even with slight variations in position or expression.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_facial_features(size):
    face = np.zeros((size, size))
    # Simulate eyes
    face[size//4, size//3] = 1
    face[size//4, 2*size//3] = 1
    # Simulate nose
    face[size//2, size//2] = 1
    # Simulate mouth
    face[3*size//4, size//3:2*size//3] = 1
    return face

def apply_pooling_to_face(face, pool_size):
    pooled_face = max_pooling(face, pool_size)
    return pooled_face

face = simulate_facial_features(20)
pooled_face = apply_pooling_to_face(face, 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(face, cmap='gray')
ax1.set_title("Original Face Features")
ax1.axis('off')
ax2.imshow(pooled_face, cmap='gray')
ax2.set_title("Pooled Face Features")
ax2.axis('off')
plt.show()
```

Slide 14: Pooling in Non-Image Data

While pooling is commonly associated with image processing, it can also be applied to non-image data, such as time series or text data, to extract important features and reduce dimensionality.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_time_series(length):
    return np.sin(np.linspace(0, 4*np.pi, length)) + np.random.normal(0, 0.1, length)

def pool_time_series(data, pool_size):
    return np.max(data.reshape(-1, pool_size), axis=1)

time_series = generate_time_series(100)
pooled_series = pool_time_series(time_series, 5)

plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Original')
plt.plot(np.arange(0, 100, 5), pooled_series, label='Pooled')
plt.legend()
plt.title('Time Series Data with Max Pooling')
plt.show()
```

Slide 15: Additional Resources

For further exploration of pooling operations in Convolutional Neural Networks, consider these resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2. "Convolutional Neural Networks for Visual Recognition" course by Stanford University (CS231n)
3. ArXiv paper: "Network In Network" by Min Lin, Qiang Chen, and Shuicheng Yan ([https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400))
4. ArXiv paper: "Striving for Simplicity: The All Convolutional Net" by Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller ([https://arxiv.org/abs/1412.6806](https://arxiv.org/abs/1412.6806))

These resources provide in-depth discussions on pooling operations, their variants, and their role in modern deep learning architectures.

