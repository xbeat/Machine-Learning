## Pooling Layers in Convolutional Neural Networks
Slide 1: Introduction to Pooling Layers in CNNs

Pooling layers are fundamental components in Convolutional Neural Networks (CNNs) that reduce the spatial dimensions of feature maps. They help in decreasing computational complexity, controlling overfitting, and achieving spatial invariance. In this presentation, we'll explore various types of pooling, their implementation, and their impact on CNN architecture.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a sample feature map
feature_map = np.random.rand(6, 6)

# Display the feature map
plt.imshow(feature_map, cmap='viridis')
plt.title('Sample Feature Map')
plt.colorbar()
plt.show()
```

Slide 2: Max Pooling

Max pooling is the most common type of pooling operation. It selects the maximum value from a defined region in the input feature map. This operation helps in retaining the most prominent features while reducing spatial dimensions.

```python
def max_pool(input_map, pool_size=2, stride=2):
    h, w = input_map.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            output[i//stride, j//stride] = np.max(input_map[i:i+pool_size, j:j+pool_size])
    
    return output

# Apply max pooling to the sample feature map
pooled_map = max_pool(feature_map)

# Display the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.imshow(pooled_map, cmap='viridis')
ax2.set_title('Max Pooled Feature Map')
plt.show()
```

Slide 3: Average Pooling

Average pooling computes the average value of a defined region in the input feature map. This operation helps in smoothing the features and can be useful in scenarios where we want to preserve more context information.

```python
def avg_pool(input_map, pool_size=2, stride=2):
    h, w = input_map.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            output[i//stride, j//stride] = np.mean(input_map[i:i+pool_size, j:j+pool_size])
    
    return output

# Apply average pooling to the sample feature map
avg_pooled_map = avg_pool(feature_map)

# Display the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.imshow(avg_pooled_map, cmap='viridis')
ax2.set_title('Average Pooled Feature Map')
plt.show()
```

Slide 4: Global Pooling

Global pooling operates on the entire feature map, reducing it to a single value. This is often used in the final layers of a CNN to create a fixed-size output regardless of the input dimensions. Global max pooling and global average pooling are two common variants.

```python
def global_max_pool(input_map):
    return np.max(input_map)

def global_avg_pool(input_map):
    return np.mean(input_map)

# Apply global pooling to the sample feature map
global_max = global_max_pool(feature_map)
global_avg = global_avg_pool(feature_map)

print(f"Global Max Pooling Result: {global_max:.4f}")
print(f"Global Average Pooling Result: {global_avg:.4f}")

# Visualize the global pooling concept
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.imshow([[global_max]], cmap='viridis')
ax2.set_title('Global Max Pooling')
ax3.imshow([[global_avg]], cmap='viridis')
ax3.set_title('Global Average Pooling')
plt.show()
```

Slide 5: Stochastic Pooling

Stochastic pooling is a probabilistic approach to pooling. It selects a value from the pooling region based on a probability distribution derived from the values in that region. This method can help in regularizing the network and reducing overfitting.

```python
def stochastic_pool(input_map, pool_size=2, stride=2):
    h, w = input_map.shape
    output_h = (h - pool_size) // stride + 1
    output_w = (w - pool_size) // stride + 1
    output = np.zeros((output_h, output_w))
    
    for i in range(0, h - pool_size + 1, stride):
        for j in range(0, w - pool_size + 1, stride):
            pool = input_map[i:i+pool_size, j:j+pool_size]
            probabilities = pool / np.sum(pool)
            output[i//stride, j//stride] = np.random.choice(pool.flatten(), p=probabilities.flatten())
    
    return output

# Apply stochastic pooling to the sample feature map
stochastic_pooled_map = stochastic_pool(feature_map)

# Display the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.imshow(stochastic_pooled_map, cmap='viridis')
ax2.set_title('Stochastic Pooled Feature Map')
plt.show()
```

Slide 6: Spatial Pyramid Pooling

Spatial Pyramid Pooling (SPP) is a technique that allows CNNs to handle inputs of variable sizes. It divides the feature map into a fixed number of bins at different scales, performing pooling in each bin. This results in a fixed-size output regardless of the input dimensions.

```python
def spatial_pyramid_pool(input_map, levels=[4, 2, 1]):
    results = []
    for level in levels:
        h_step = input_map.shape[0] // level
        w_step = input_map.shape[1] // level
        
        for i in range(level):
            for j in range(level):
                h_start, h_end = i * h_step, (i + 1) * h_step
                w_start, w_end = j * w_step, (j + 1) * w_step
                results.append(np.max(input_map[h_start:h_end, w_start:w_end]))
    
    return np.array(results)

# Apply spatial pyramid pooling to the sample feature map
spp_result = spatial_pyramid_pool(feature_map)

print(f"SPP output shape: {spp_result.shape}")
print(f"SPP output: {spp_result}")

# Visualize the SPP concept
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.bar(range(len(spp_result)), spp_result)
ax2.set_title('Spatial Pyramid Pooling Output')
ax2.set_xlabel('Bin Index')
ax2.set_ylabel('Pooled Value')
plt.tight_layout()
plt.show()
```

Slide 7: Mixed Pooling

Mixed pooling combines different pooling operations to leverage their respective advantages. For example, we can use a combination of max pooling and average pooling, either by alternating between them in different layers or by using a weighted sum of their outputs.

```python
def mixed_pool(input_map, pool_size=2, stride=2, alpha=0.5):
    max_pooled = max_pool(input_map, pool_size, stride)
    avg_pooled = avg_pool(input_map, pool_size, stride)
    return alpha * max_pooled + (1 - alpha) * avg_pooled

# Apply mixed pooling to the sample feature map
mixed_pooled_map = mixed_pool(feature_map)

# Display the results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
ax1.imshow(feature_map, cmap='viridis')
ax1.set_title('Original Feature Map')
ax2.imshow(max_pool(feature_map), cmap='viridis')
ax2.set_title('Max Pooled')
ax3.imshow(avg_pool(feature_map), cmap='viridis')
ax3.set_title('Average Pooled')
ax4.imshow(mixed_pooled_map, cmap='viridis')
ax4.set_title('Mixed Pooled')
plt.show()
```

Slide 8: Pooling with PyTorch

PyTorch provides built-in functions for various pooling operations, making it easy to incorporate them into CNN architectures. Here's an example of how to use max pooling and average pooling in PyTorch:

```python
import torch
import torch.nn as nn

# Create a sample input tensor
input_tensor = torch.randn(1, 1, 6, 6)

# Define pooling layers
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Apply pooling operations
max_pooled = max_pool(input_tensor)
avg_pooled = avg_pool(input_tensor)

print("Input shape:", input_tensor.shape)
print("Max pooled shape:", max_pooled.shape)
print("Avg pooled shape:", avg_pooled.shape)

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(input_tensor.squeeze().numpy(), cmap='viridis')
ax1.set_title('Original Input')
ax2.imshow(max_pooled.squeeze().numpy(), cmap='viridis')
ax2.set_title('Max Pooled')
ax3.imshow(avg_pooled.squeeze().numpy(), cmap='viridis')
ax3.set_title('Avg Pooled')
plt.show()
```

Slide 9: Real-life Example: Image Classification

In image classification tasks, pooling layers play a crucial role in reducing the spatial dimensions of feature maps and capturing important features. Let's implement a simple CNN with pooling layers for classifying handwritten digits using the MNIST dataset:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}")

print("Training finished!")
```

Slide 10: Real-life Example: Image Segmentation

Pooling layers are essential in image segmentation tasks, where we need to maintain spatial information while reducing dimensions. Let's implement a simple U-Net architecture with pooling and upsampling layers for image segmentation:

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 1024)
        self.up1 = self.up(1024, 512)
        self.up2 = self.up(512, 256)
        self.up3 = self.up(256, 128)
        self.up4 = self.up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_ch, out_ch)
        )

    def up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            self.double_conv(in_ch, out_ch)
        )

# Create a sample input
input_image = torch.randn(1, 3, 256, 256)

# Initialize the model
model = UNet(n_channels=3, n_classes=2)

# Run the model
output = model(input_image)

print(f"Input shape: {input_image.shape}")
print(f"Output shape: {output.shape}")
```

Slide 11: Pooling in Residual Networks

Residual Networks (ResNets) use pooling layers strategically to reduce spatial dimensions while maintaining the network's depth. Here's a simplified implementation of a ResNet block with pooling:

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create a sample input
input_image = torch.randn(1, 3, 224, 224)

# Initialize the model
model = SimpleResNet()

# Run the model
output = model(input_image)

print(f"Input shape: {input_image.shape}")
print(f"Output shape: {output.shape}")
```

Slide 12: Pooling in Generative Adversarial Networks (GANs)

In GANs, pooling layers are often used in the discriminator network to downsample the input image. Here's a simple example of a discriminator using pooling layers:

```python
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes it 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# Create a sample input
channels_img = 3
features_d = 64
input_image = torch.randn(1, channels_img, 64, 64)

# Initialize the model
model = Discriminator(channels_img, features_d)

# Run the model
output = model(input_image)

print(f"Input shape: {input_image.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Pooling in Attention Mechanisms

Pooling can be used in attention mechanisms to aggregate information from different parts of the input. Here's a simple example of an attention mechanism using pooling:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPool(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionPool, self).__init__()
        self.query = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.key = nn.Conv2d(in_features, out_features, kernel_size=1)
        self.value = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = F.softmax(torch.matmul(q.view(q.size(0), q.size(1), -1).transpose(1, 2),
                                      k.view(k.size(0), k.size(1), -1)), dim=-1)
        
        out = torch.matmul(v.view(v.size(0), v.size(1), -1), attn.transpose(1, 2))
        out = out.view(x.size())
        
        return out

# Create a sample input
input_tensor = torch.randn(1, 64, 16, 16)

# Initialize the model
model = AttentionPool(64, 64)

# Run the model
output = model(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

Slide 14: Additional Resources

For more in-depth information on pooling layers in CNNs, consider exploring these resources:

1. "Striving for Simplicity: The All Convolutional Net" by Springenberg et al. (2014) ArXiv: [https://arxiv.org/abs/1412.6806](https://arxiv.org/abs/1412.6806)
2. "Network In Network" by Lin et al. (2013) ArXiv: [https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400)
3. "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition" by He et al. (2015) ArXiv: [https://arxiv.org/abs/1406.4729](https://arxiv.org/abs/1406.4729)

These papers provide valuable insights into various pooling techniques and their applications in deep learning architectures.

