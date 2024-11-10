## Mastering Deep Learning with PyTorch
Slide 1: PyTorch Fundamentals - Tensors and Operations

PyTorch's fundamental building block is the tensor, a multi-dimensional array optimized for deep learning computations. Understanding tensor operations, including creation, manipulation, and mathematical transformations, is essential for developing neural networks efficiently using PyTorch's dynamic computation capabilities.

```python
import torch

# Creating tensors from different sources
tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
random_tensor = torch.rand(3, 3)
zeros_tensor = torch.zeros(2, 4)

# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
addition = a + b
multiplication = a * b

# Moving tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_tensor = tensor_from_list.to(device)

print(f"Original tensor:\n{tensor_from_list}")
print(f"Random tensor:\n{random_tensor}")
print(f"Addition result:\n{addition}")
print(f"Multiplication result:\n{multiplication}")
```

Slide 2: Building Neural Networks with torch.nn

PyTorch's nn module provides the building blocks for creating neural network architectures. This implementation demonstrates a basic feed-forward neural network using nn.Module as the base class, incorporating layers, activation functions, and forward pass definition.

```python
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model instance
model = FeedForwardNet(input_size=784, hidden_size=256, output_size=10)
print(model)

# Example forward pass
sample_input = torch.randn(1, 784)
output = model(sample_input)
print(f"Output shape: {output.shape}")
```

Slide 3: Implementing a Convolutional Neural Network

The implementation of a CNN architecture demonstrates the power of convolutional layers for feature extraction in image processing tasks. This network combines convolutional layers with pooling operations and fully connected layers for effective image classification.

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model and test forward pass
cnn_model = ConvNet()
sample_image = torch.randn(1, 3, 32, 32)
output = cnn_model(sample_image)
print(f"CNN output shape: {output.shape}")
```

Slide 4: Custom Dataset Creation

Creating custom datasets in PyTorch requires implementing the Dataset class with three essential methods: **init**, **len**, and **getitem**. This implementation shows how to build a dataset for handling image data with corresponding labels.

```python
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CustomImageDataset(Dataset):
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

# Example usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Sample data
sample_paths = ['image1.jpg', 'image2.jpg']
sample_labels = [0, 1]
dataset = CustomImageDataset(sample_paths, sample_labels, transform)
```

Slide 5: Training Loop Implementation

A comprehensive training loop implementation showcasing the essential components of model training in PyTorch, including forward pass, loss calculation, backpropagation, and optimization steps with proper device handling and progress tracking.

```python
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, '
                      f'Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

# Example usage
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train for 5 epochs
train_model(model, train_loader, criterion, optimizer, device, epochs=5)
```

Slide 6: Transfer Learning with Pre-trained Models

Transfer learning leverages pre-trained models to achieve superior performance on new tasks with limited data. This implementation demonstrates how to modify and fine-tune a pre-trained ResNet model for custom classification tasks while preserving learned features.

```python
import torchvision.models as models
from torch.optim import lr_scheduler

def create_transfer_model(num_classes):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model

# Create and prepare model
model = create_transfer_model(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print(f"Modified final layers: {model.fc}")
```

Slide 7: Implementing Attention Mechanism

The attention mechanism has revolutionized deep learning by enabling models to focus on relevant parts of input data. This implementation shows a scaled dot-product attention mechanism, fundamental to transformer architectures.

```python
class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.scaling = float(self.head_dim) ** -0.5
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return output, attention

# Example usage
attention = AttentionLayer(embed_dim=512, num_heads=8)
x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10, embed_dim=512
output, attention_weights = attention(x, x, x)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

Slide 8: Custom Loss Function Implementation

Creating custom loss functions allows for specialized training objectives beyond standard losses. This implementation demonstrates a focal loss function commonly used in object detection tasks to address class imbalance problems.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Example usage
criterion = FocalLoss(gamma=2)
outputs = torch.randn(10, 5)  # 10 samples, 5 classes
targets = torch.randint(0, 5, (10,))  # Random labels between 0 and 4
loss = criterion(outputs, targets)
print(f"Focal Loss: {loss.item():.4f}")
```

Slide 9: Implementing Gradient Clipping

Gradient clipping is a crucial technique for preventing exploding gradients in deep networks, particularly in RNNs and transformers. This implementation shows how to properly implement gradient clipping during training.

```python
def train_with_gradient_clipping(model, train_loader, criterion, optimizer, 
                               max_grad_norm, device, epochs):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Compute total gradient norm
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) 
                           for p in model.parameters() if p.grad is not None]),
                2
            )
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Gradient norm: {total_norm:.4f}')

# Example usage
max_grad_norm = 1.0
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_with_gradient_clipping(model, train_loader, criterion, optimizer,
                           max_grad_norm, device, epochs=1)
```

Slide 10: Real-world Image Classification Implementation

A complete implementation of an image classification system using PyTorch, demonstrating data preprocessing, model architecture, training pipeline, and evaluation metrics for a practical computer vision task.

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report

class ImageClassifier:
    def __init__(self, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        accuracy = 100. * correct / total
        return running_loss / len(train_loader), accuracy
    
    def evaluate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        return classification_report(all_labels, all_preds)

# Example usage
classifier = ImageClassifier(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    loss, acc = classifier.train_epoch(train_loader, criterion, optimizer)
    print(f'Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%')

# Evaluation
eval_results = classifier.evaluate(val_loader)
print("\nEvaluation Results:")
print(eval_results)
```

Slide 11: Sequence-to-Sequence Learning with LSTM

Implementation of a sequence-to-sequence model using LSTM networks for tasks like machine translation or text summarization, showing encoder-decoder architecture with attention mechanism.

```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size, 
                 embedding_dim, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        
        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                                  batch_first=True, bidirectional=True)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim + hidden_size*2, hidden_size, 
                                  num_layers, batch_first=True)
        
        # Attention
        self.attention = nn.Linear(hidden_size*3, 1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_vocab_size)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = tgt.shape[1]
        
        # Encoder
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded_src)
        
        # Initialize decoder input
        decoder_input = tgt[:, 0:1]
        decoder_hidden = hidden[-1].unsqueeze(0)
        decoder_cell = cell[-1].unsqueeze(0)
        
        outputs = []
        for t in range(1, max_len):
            # Decoder step
            embedded_tgt = self.decoder_embedding(decoder_input)
            
            # Attention
            attention_weights = F.softmax(
                self.attention(
                    torch.cat([encoder_outputs, 
                             decoder_hidden[-1:].repeat(1, src.shape[1], 1)], 
                             dim=2)
                ), dim=1)
            context = torch.bmm(attention_weights.transpose(1, 2), 
                              encoder_outputs)
            
            # Decoder input with context
            lstm_input = torch.cat([embedded_tgt, context], dim=2)
            decoder_output, (decoder_hidden, decoder_cell) = \
                self.decoder_lstm(lstm_input, (decoder_hidden, decoder_cell))
            
            output = self.output_layer(decoder_output)
            outputs.append(output)
            
            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                decoder_input = tgt[:, t:t+1]
            else:
                decoder_input = output.argmax(2)
        
        return torch.cat(outputs, dim=1)

# Model initialization and usage example
model = Seq2SeqLSTM(input_vocab_size=1000, output_vocab_size=1000,
                    hidden_size=256, embedding_dim=128)
src = torch.randint(0, 1000, (32, 20))  # batch_size=32, seq_len=20
tgt = torch.randint(0, 1000, (32, 15))  # batch_size=32, seq_len=15
output = model(src, tgt)
print(f"Output shape: {output.shape}")
```

Slide 12: Data Parallel Training Implementation

Implementing distributed training across multiple GPUs using PyTorch's DataParallel and DistributedDataParallel, enabling efficient scaling of deep learning models for faster training on large datasets.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ParallelTrainer:
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
        
    def train_parallel(self, rank, train_loader, criterion, optimizer, epochs):
        setup(rank, self.world_size)
        
        # Move model to device and wrap with DDP
        torch.cuda.set_device(rank)
        self.model = self.model.to(rank)
        ddp_model = DistributedDataParallel(self.model, device_ids=[rank])
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(rank), target.to(rank)
                
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if rank == 0 and batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')
        
        cleanup()

# Example usage
def main():
    world_size = torch.cuda.device_count()
    model = YourModel()
    trainer = ParallelTrainer(model, world_size)
    
    mp.spawn(
        trainer.train_parallel,
        args=(train_loader, criterion, optimizer, num_epochs),
        nprocs=world_size
    )

if __name__ == "__main__":
    main()
```

Slide 13: Advanced Model Validation and Metrics

Implementation of comprehensive model validation techniques including k-fold cross-validation, confusion matrix analysis, and various performance metrics for deep learning model evaluation.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

class ModelValidator:
    def __init__(self, model_class, num_folds=5):
        self.model_class = model_class
        self.num_folds = num_folds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def k_fold_validation(self, dataset, batch_size):
        kfold = KFold(n_splits=self.num_folds, shuffle=True)
        results = []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(dataset, batch_size=batch_size, 
                                    sampler=train_subsampler)
            val_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=val_subsampler)
            
            # Initialize model for this fold
            model = self.model_class().to(self.device)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Train and evaluate
            fold_results = self.train_and_evaluate(
                model, train_loader, val_loader, 
                criterion, optimizer
            )
            results.append(fold_results)
            
        return self.aggregate_results(results)
    
    def calculate_metrics(self, true_labels, predictions, probabilities):
        # Confusion Matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # ROC Curve and AUC
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        
        # Precision, Recall, F1
        precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1])
        recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def aggregate_results(self, results):
        metrics = {}
        for metric in results[0].keys():
            values = [r[metric] for r in results]
            metrics[f'{metric}_mean'] = np.mean(values)
            metrics[f'{metric}_std'] = np.std(values)
        return metrics

# Example usage
validator = ModelValidator(YourModelClass, num_folds=5)
results = validator.k_fold_validation(dataset, batch_size=32)
print("\nValidation Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

Slide 14: Additional Resources

*   Transfer Learning in Deep Neural Networks: [https://arxiv.org/abs/1411.1792](https://arxiv.org/abs/1411.1792)
*   Deep Learning Model Compression: [https://arxiv.org/abs/1710.09282](https://arxiv.org/abs/1710.09282)
*   Attention Mechanisms in Neural Networks: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   Efficient Training of Deep Neural Networks: [https://arxiv.org/abs/1812.06162](https://arxiv.org/abs/1812.06162)
*   PyTorch Implementation Best Practices: [https://arxiv.org/abs/2006.14050](https://arxiv.org/abs/2006.14050)

