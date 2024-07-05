## 2-Stage Backpropagation in Python
Slide 1: Introduction to 2-Stage Backpropagation

2-Stage Backpropagation is an advanced technique for training neural networks that aims to improve convergence and generalization. It involves splitting the network into two parts and training them separately before fine-tuning the entire network.

```python
import torch
import torch.nn as nn

class TwoStageNetwork(nn.Module):
    def __init__(self):
        super(TwoStageNetwork, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.stage2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return x
```

Slide 2: Stage 1: Feature Extraction

The first stage focuses on learning meaningful features from the input data. This stage typically consists of convolutional layers or dense layers that extract hierarchical representations.

```python
def train_stage1(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for inputs, _ in dataloader:
            optimizer.zero_grad()
            features = model.stage1(inputs)
            loss = criterion(features, features.detach())  # Self-supervised loss
            loss.backward()
            optimizer.step()
```

Slide 3: Stage 2: Classification

The second stage takes the learned features from Stage 1 and performs the final classification task. This stage is typically composed of fully connected layers.

```python
def train_stage2(model, dataloader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            features = model.stage1(inputs).detach()
            outputs = model.stage2(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

Slide 4: Freezing Stage 1 Parameters

During Stage 2 training, we freeze the parameters of Stage 1 to preserve the learned features and focus on optimizing the classification layers.

```python
def freeze_stage1(model):
    for param in model.stage1.parameters():
        param.requires_grad = False

def unfreeze_stage1(model):
    for param in model.stage1.parameters():
        param.requires_grad = True
```

Slide 5: Fine-tuning the Entire Network

After training both stages separately, we fine-tune the entire network end-to-end to optimize the overall performance.

```python
def finetune(model, dataloader, optimizer, criterion, epochs):
    unfreeze_stage1(model)
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

Slide 6: Learning Rate Scheduling

Implementing learning rate scheduling can help improve convergence during the different stages of training. We'll use a step learning rate scheduler as an example.

```python
from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(epochs):
    train_epoch(model, dataloader, optimizer, criterion)
    scheduler.step()
```

Slide 7: Data Augmentation for Stage 1

Applying data augmentation techniques during Stage 1 training can help learn more robust features. Here's an example using torchvision transforms.

```python
from torchvision import transforms

stage1_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

stage1_dataset = CustomDataset(transform=stage1_transforms)
stage1_dataloader = DataLoader(stage1_dataset, batch_size=32, shuffle=True)
```

Slide 8: Implementing Gradient Clipping

Gradient clipping can help stabilize training, especially during the fine-tuning phase. Let's implement it in our training loop.

```python
def train_with_gradient_clipping(model, dataloader, optimizer, criterion, clip_value):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
```

Slide 9: Early Stopping

Implementing early stopping can prevent overfitting during the training process. We'll create a simple early stopping mechanism.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
```

Slide 10: Visualizing Feature Maps

Visualizing feature maps can provide insights into what the network has learned during Stage 1. Let's create a function to visualize feature maps.

```python
import matplotlib.pyplot as plt

def visualize_feature_maps(model, input_image):
    model.eval()
    with torch.no_grad():
        features = model.stage1(input_image.unsqueeze(0))
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < features.shape[1]:
            ax.imshow(features[0, i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
```

Slide 11: Monitoring Training Progress

Keeping track of training progress is crucial. Let's implement a simple progress monitoring system using TensorBoard.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

def train_with_monitoring(model, dataloader, optimizer, criterion, epoch):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 99:
            writer.add_scalar('training loss',
                              running_loss / 100,
                              epoch * len(dataloader) + i)
            running_loss = 0.0
```

Slide 12: Transfer Learning with 2-Stage Backpropagation

2-Stage Backpropagation can be effectively combined with transfer learning. Let's modify our network to use a pre-trained model for Stage 1.

```python
import torchvision.models as models

class TransferTwoStageNetwork(nn.Module):
    def __init__(self, num_classes):
        super(TransferTwoStageNetwork, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.stage1 = nn.Sequential(*list(resnet.children())[:-1])
        self.stage2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = x.view(x.size(0), -1)
        x = self.stage2(x)
        return x
```

Slide 13: Handling Imbalanced Datasets

When dealing with imbalanced datasets, we can modify the loss function to address class imbalance. Let's implement a weighted cross-entropy loss.

```python
def calculate_class_weights(dataset):
    class_counts = torch.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1
    return 1.0 / class_counts

class_weights = calculate_class_weights(train_dataset)
criterion = nn.CrossEntropyLoss(weight=class_weights)

def train_with_weighted_loss(model, dataloader, optimizer, criterion):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

Slide 14: Evaluating Model Performance

After training, it's important to evaluate the model's performance on a separate test set. Let's create an evaluation function.

```python
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    average_loss = total_loss / len(test_loader)
    
    return accuracy, average_loss
```

Slide 15: Additional Resources

To further explore 2-Stage Backpropagation and related techniques, consider the following resources:

1. "Deep Learning" by Goodfellow, Bengio, and Courville - Available at: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
2. "Curriculum Learning" by Bengio et al. (2009) - ArXiv: [https://arxiv.org/abs/0904.2425](https://arxiv.org/abs/0904.2425)
3. "Progressive Neural Networks" by Rusu et al. (2016) - ArXiv: [https://arxiv.org/abs/1606.04671](https://arxiv.org/abs/1606.04671)
4. "An Overview of Multi-Task Learning in Deep Neural Networks" by Ruder (2017) - ArXiv: [https://arxiv.org/abs/1706.05098](https://arxiv.org/abs/1706.05098)

