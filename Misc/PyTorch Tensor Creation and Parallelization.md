## PyTorch Tensor Creation and Parallelization

Slide 1: Tensor Creation 

In PyTorch, tensors are the fundamental data structures used for computations. Here's how to create a tensor from a Python list or array. Code Example:

```python
import torch

# Create a tensor from a list
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])

# Create a tensor from a numpy array
import numpy as np
numpy_array = np.array([[1, 2], [3, 4]])
tensor_from_numpy = torch.from_numpy(numpy_array)
```

Slide 2: Tensor Operations 

PyTorch provides a rich set of operations for tensor manipulation and computation. Code Example:

```python
import torch

# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Tensor addition
z = x + y
print(z)  # Output: tensor([5, 7, 9])

# Matrix multiplication
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[1, 1], [1, 1]])
z = torch.matmul(x, y)
print(z)  # Output: tensor([[ 3,  3], [11, 11]])
```

Slide 3: Autograd 

PyTorch's autograd feature enables automatic differentiation, making it easy to compute gradients for training neural networks. Code Example:

```python
import torch

# Create a tensor with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)

# Perform operations
y = x ** 2
z = 2 * y + 3

# Compute gradients
z.backward()
print(x.grad)  # Output: tensor(8.)
```

Slide 4: Linear Regression 

Building a simple linear regression model in PyTorch. Code Example:

```python
import torch
import torch.nn as nn

# Define the model
model = nn.Linear(1, 1)  # 1 input feature, 1 output

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    inputs = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[2.0], [4.0], [6.0]])

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Learned weight: {model.weight.item()} Learned bias: {model.bias.item()}')
```

Slide 5: Convolutional Neural Networks 

Building a simple convolutional neural network (CNN) for image classification. Code Example:

```python
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels, 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Flatten and feed into fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

Slide 6: Recurrent Neural Networks 

Building a simple recurrent neural network (RNN) for sequence modeling. Code Example:

```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = out[:, -1, :]  # Get the last output for each sequence
        out = self.fc(out)
        return out, hidden

# Example usage
rnn = RNN(10, 20, 5)  # 10 input features, 20 hidden units, 5 output classes
inputs = torch.randn(32, 10, 10)  # Batch size 32, sequence length 10, 10 input features
hidden = torch.zeros(1, 32, 20)  # Initial hidden state
outputs, hidden = rnn(inputs, hidden)
```

Slide 7: Data Loaders 

PyTorch provides data loaders for efficient batching and loading of data during training and evaluation. Code Example:

```python
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and data loader
dataset = MyDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for batch_data, batch_labels in dataloader:
    # Train or evaluate with batch_data and batch_labels
    pass
```

Slide 8: Transfer Learning 

PyTorch allows for efficient transfer learning by leveraging pre-trained models on large datasets. Code Example:

```python
import torchvision.models as models

# Load a pre-trained model
model = models.resnet18(pretrained=True)

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer with a new one
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Fine-tune the model on your dataset
# ...
```

Slide 9: Deployment 

PyTorch models can be exported and deployed for inference on various platforms. Code Example:

```python
import torch.onnx

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))

# Export to ONNX format
dummy_input = torch.randn(1, 3, 224, 224)  # Example input size
torch.onnx.export(model, dummy_input, 'model.onnx')

# Deploy the ONNX model on various platforms (e.g., mobile, web, cloud)
```

Slide 10: Distributed Training 

PyTorch supports distributed training for scaling model training across multiple GPUs or machines. Code Example:

```python
import torch.distributed as dist

# Initialize the process group
dist.init_process_group(backend='nccl', init_method='...')

# Create your model, optimizer, and data loader
model = MyModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_loader = ...

# Scatter the model across processes
model = nn.parallel.DistributedDataParallel
```

Slide 11: Tensorboard Integration 

PyTorch integrates well with TensorBoard, allowing you to visualize training metrics, model graphs, and more. Code Example:

```python
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# Log scalar values
for epoch in range(100):
    # Training loop...
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

# Log model graph
writer.add_graph(model, input_to_model)

# Flush and close the writer
writer.flush()
writer.close()
```

Slide 12: PyTorch Lightning 

PyTorch Lightning is a lightweight PyTorch wrapper that simplifies the training loop and provides additional features like built-in loggers, callbacks, and more. Code Example:

```python
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

model = LitModel()
trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(model, dataloader)
```

Slide 13: Quantization 

PyTorch supports quantization for optimizing models for deployment on resource-constrained devices. Code Example:

```python
import torch.quantization

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)

# Evaluate or deploy the quantized model
# ...
```

Slide 14: Pruning 

PyTorch provides tools for pruning models, removing unnecessary weights to reduce model size and improve inference speed. Code Example:

```python
import torch.nn.utils.prune as prune

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))

# Prune the model
prune.l1_unstructured(model.conv1, 'weight', 0.5)
prune.remove(model.conv1, 'weight')

# Fine-tune the pruned model
# ...
```

This covers a wide range of PyTorch features and examples, from tensor operations and autograd to model architectures, data loaders, transfer learning, deployment, distributed training, visualization, quantization, and pruning.

## Meta
Here's a title, description, and hashtags for a TikTok series on PyTorch, with an institutional tone:

"Mastering PyTorch: The Deep Learning Toolbox"

Unlock the power of PyTorch, the cutting-edge deep learning framework from the AI research community. This comprehensive video series takes you on a guided tour through PyTorch's essential features, from tensor operations and automatic differentiation to building and training state-of-the-art neural network architectures. Whether you're a seasoned data scientist or just starting your deep learning journey, this institutional-level content will equip you with the skills to tackle complex AI problems and push the boundaries of modern machine learning. Join us as we explore the depth of PyTorch and its applications in computer vision, natural language processing, and beyond.

Hashtags: #PyTorch #DeepLearning #AI #NeuralNetworks #TensorOperations #Autograd #ConvolutionalNeuralNetworks #RecurrentNeuralNetworks #TransferLearning #ModelDeployment #DistributedTraining #Quantization #Pruning #MachineLearning #DataScience #Research #InstitutionalLearning

