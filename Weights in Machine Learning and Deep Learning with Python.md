## Weights in Machine Learning and Deep Learning with Python
Slide 1: Introduction to Weights in Machine Learning

Weights are fundamental components in machine learning models, serving as the parameters that the model learns during training. They determine the strength of connections between inputs and outputs, playing a crucial role in the model's decision-making process.

```python
import numpy as np

# Simple neural network with one input and one output
class SimpleNeuralNetwork:
    def __init__(self):
        self.weight = np.random.rand()
        self.bias = np.random.rand()

    def forward(self, x):
        return x * self.weight + self.bias

# Initialize and use the network
model = SimpleNeuralNetwork()
input_value = 2
output = model.forward(input_value)
print(f"Input: {input_value}, Output: {output:.2f}")
```

Slide 2: Weight Initialization

Proper weight initialization is crucial for effective model training. Random initialization helps break symmetry and allows the network to learn diverse features. Here's an example of weight initialization in PyTorch:

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
        # He initialization for ReLU activation
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')

# Create a network with 10 input features, 20 hidden neurons, and 2 output classes
model = NeuralNetwork(10, 20, 2)
print(model.layer1.weight[:5, :5])  # Print a subset of weights
```

Slide 3: Weight Updates during Training

During training, weights are updated based on the calculated gradients and chosen optimization algorithm. Here's a simple example of weight updates using stochastic gradient descent (SGD):

```python
import numpy as np

def update_weights(weights, gradients, learning_rate):
    return weights - learning_rate * gradients

# Example usage
weights = np.array([0.5, -0.3, 0.8])
gradients = np.array([0.1, -0.2, 0.05])
learning_rate = 0.01

new_weights = update_weights(weights, gradients, learning_rate)
print("Original weights:", weights)
print("Updated weights:", new_weights)
```

Slide 4: Weight Regularization

Regularization techniques help prevent overfitting by adding a penalty term to the loss function based on the weights. L1 and L2 regularization are common methods. Here's an example using L2 regularization in PyTorch:

```python
import torch
import torch.nn as nn

class RegularizedModel(nn.Module):
    def __init__(self):
        super(RegularizedModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = RegularizedModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# In the training loop:
for epoch in range(num_epochs):
    # ... (forward pass and loss calculation)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Slide 5: Weight Visualization

Visualizing weights can provide insights into what features the model has learned. Here's an example of visualizing weights in a convolutional neural network:

```python
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3)

    def forward(self, x):
        return self.conv(x)

model = SimpleCNN()

# Visualize the weights
weights = model.conv.weight.data.numpy()
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(16):
    ax = axs[i//4, i%4]
    ax.imshow(weights[i, 0], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 6: Weight Pruning

Weight pruning is a technique to reduce model size and improve efficiency by removing less important weights. Here's a simple example of weight pruning:

```python
import torch
import torch.nn as nn

def prune_weights(model, threshold):
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask = torch.abs(param.data) > threshold
            param.data *= mask

# Example usage
model = nn.Linear(10, 5)
print("Before pruning:")
print(model.weight.data)

prune_weights(model, threshold=0.1)
print("\nAfter pruning:")
print(model.weight.data)
```

Slide 7: Weight Quantization

Weight quantization reduces the precision of weights to decrease model size and improve inference speed. Here's an example using PyTorch's quantization features:

```python
import torch
import torch.nn as nn
import torch.quantization

class QuantizableLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(QuantizableLinear, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super(QuantizableLinear, self).forward(x)
        x = self.dequant(x)
        return x

model = QuantizableLinear(10, 5)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# ... (after calibration)
torch.quantization.convert(model, inplace=True)

print(model.weight())
```

Slide 8: Weight Transfer in Transfer Learning

Transfer learning leverages pre-trained weights to improve performance on new tasks. Here's an example using a pre-trained ResNet model:

```python
import torch
import torchvision.models as models

# Load pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, 10)  # 10 classes in the new task

# Only train the new layer
optimizer = torch.optim.Adam(resnet.fc.parameters())

# Use the model for fine-tuning
# ... (training loop)
```

Slide 9: Weight Sharing in Siamese Networks

Siamese networks use weight sharing to process paired inputs. Here's an example of a simple Siamese network:

```python
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 62 * 62, 128)

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

model = SiameseNetwork()
```

Slide 10: Weight Importance in Feature Attribution

Analyzing weight importance helps understand which features contribute most to the model's decisions. Here's an example using a simple method:

```python
import numpy as np
import matplotlib.pyplot as plt

def feature_importance(weights):
    importance = np.abs(weights).sum(axis=0)
    return importance / importance.sum()

# Example weights for a linear model with 5 features
weights = np.array([-0.5, 1.2, 0.8, -0.3, 2.1])

importance = feature_importance(weights)
plt.bar(range(len(importance)), importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Based on Weights')
plt.show()
```

Slide 11: Gradient Flow and Weight Updates

Understanding gradient flow helps diagnose training issues. Here's an example of tracking gradient statistics:

```python
import torch
import torch.nn as nn

class GradientTracker(nn.Module):
    def __init__(self, model):
        super(GradientTracker, self).__init__()
        self.model = model
        self.gradient_stats = {}

    def forward(self, x):
        return self.model(x)

    def track_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.gradient_stats:
                    self.gradient_stats[name] = []
                self.gradient_stats[name].append({
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'max': param.grad.max().item(),
                    'min': param.grad.min().item()
                })

# Usage in training loop
model = nn.Linear(10, 1)
tracker = GradientTracker(model)

# After backward pass
loss.backward()
tracker.track_gradients()
```

Slide 12: Weight Normalization

Weight normalization is a technique to improve the stability of neural network training. Here's an example implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = F.normalize(self.weight, dim=1)
        return F.linear(input, weight, self.bias)

# Usage
layer = WeightNormalizedLinear(10, 5)
input = torch.randn(20, 10)
output = layer(input)
print(output.shape)
```

Slide 13: Real-Life Example: Image Classification

Let's consider an image classification task using a convolutional neural network. We'll focus on the weights in different layers:

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

# Load CIFAR10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop (simplified)
for epoch in range(5):
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Analyze weights after training
print("Conv1 weight shape:", model.conv1.weight.shape)
print("Conv2 weight shape:", model.conv2.weight.shape)
print("FC weight shape:", model.fc.weight.shape)
```

Slide 14: Real-Life Example: Natural Language Processing

Let's explore weights in a simple recurrent neural network for sentiment analysis:

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Example usage
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # Binary sentiment (positive/negative)

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim)

# Analyze weights
print("Embedding weight shape:", model.embedding.weight.shape)
print("RNN weight shape:", model.rnn.weight_ih_l0.shape)
print("FC weight shape:", model.fc.weight.shape)

# Example forward pass
input_sequence = torch.randint(0, vocab_size, (1, 20))  # Batch size 1, sequence length 20
output = model(input_sequence)
print("Output shape:", output.shape)
```

Slide 15: Additional Resources

For more in-depth information on weights in machine learning and deep learning, consider exploring the following resources:

1. "Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming" by Chirag Goyal (arXiv:2004.10640)
2. "Visualizing and Understanding Convolutional Networks" by Matthew D. Zeiler and Rob Fergus (arXiv:1311.2901)
3. "Deep Residual Learning for Image Recognition" by Kaiming He et al. (arXiv:1512.03385)
4. "Attention Is All You Need" by Ashish Vaswani et al. (arXiv:1706.03762)

These papers provide valuable insights into weight initialization, visualization, and their role in various neural network architectures.

