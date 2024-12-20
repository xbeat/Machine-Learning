## PyTorch for Deep Learning Practical Code Examples
Slide 1: PyTorch - The Modern Deep Learning Framework

PyTorch has emerged as a leading framework for deep learning research and production, offering dynamic computational graphs, native Python integration, and extensive ecosystem support. Its intuitive design philosophy emphasizes code readability and debugging capabilities.

```python
import torch
import torch.nn as nn

# Basic neural network in PyTorch
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Create model instance
model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
```

Slide 2: PyTorch Tensor Operations

PyTorch's tensor operations form the foundation of all deep learning computations, providing GPU acceleration and automatic differentiation capabilities essential for training neural networks.

```python
# Create tensors and perform operations
x = torch.randn(3, 3)
y = torch.ones(3, 3)

# Basic operations
z = x + y  # Addition
w = torch.matmul(x, y)  # Matrix multiplication
grad_tensor = torch.autograd.grad(w.sum(), x)  # Automatic differentiation

print(f"Original tensor:\n{x}\n")
print(f"Matrix multiplication result:\n{w}")
```

Slide 3: Deep Learning Model Architecture

Understanding model architecture design principles is crucial for implementing effective neural networks. This example demonstrates a modern convolutional neural network implementation.

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
```

Slide 4: Training Loop Implementation

The training loop represents the core mechanism for model optimization, incorporating forward passes, loss calculation, backpropagation, and parameter updates in a systematic manner.

```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
```

Slide 5: Data Preprocessing Pipeline

```python
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Custom dataset implementation
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = self.load_data(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

Slide 6: Loss Functions and Optimization Algorithms

PyTorch provides a comprehensive suite of loss functions and optimizers for training neural networks. Understanding their mathematical foundations and implementation details is crucial for effective model training.

```python
# Common loss functions and optimizers
criterion = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
custom_loss = lambda y_pred, y_true: torch.mean((y_pred - y_true)**2)

# Multiple optimizer implementations
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```

Slide 7: Transfer Learning Implementation

Transfer learning leverages pre-trained models to achieve superior performance on new tasks with limited data. This implementation demonstrates fine-tuning a pre-trained ResNet model.

```python
import torchvision.models as models

def create_transfer_model(num_classes):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    # Freeze backbone layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify final layer for new task
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model
```

Slide 8: Attention Mechanism

The attention mechanism has revolutionized deep learning architectures. This implementation shows a basic self-attention module that can be integrated into various neural network architectures.

```python
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Shape: (batch_size, seq_len, dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
```

Slide 9: Model Evaluation and Metrics

A robust evaluation pipeline is essential for assessing model performance. This implementation includes various metrics and visualization tools for model analysis.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

Slide 10: Time Series Analysis with LSTM

```python
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                           num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Model instantiation and training setup
model = LSTMPredictor(input_dim=10, hidden_dim=64, num_layers=2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()
```

Slide 11: Custom Dataset for Image Classification

The implementation of custom datasets is crucial for handling specialized data formats and preprocessing requirements in deep learning applications. This example demonstrates a complete image classification pipeline.

```python
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Usage example
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

dataset = ImageClassificationDataset(
    image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'],
    labels=[0, 1],
    transform=transform
)
```

Slide 12: Model Regularization Techniques

Understanding and implementing regularization techniques is essential for preventing overfitting and improving model generalization. This implementation showcases various regularization methods.

```python
class RegularizedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(RegularizedNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # L2 regularization is applied through weight decay in optimizer
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# Training with regularization
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)
```

Slide 13: Advanced Model Architectures - Transformer

The Transformer architecture has become fundamental in modern deep learning. This implementation shows a basic Transformer encoder block with multi-head attention.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Deep Residual Learning for Image Recognition" - [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
*   "Adam: A Method for Stochastic Optimization" - [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" - [https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
*   For more PyTorch implementations and tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

