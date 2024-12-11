## Advantages of Transfer Learning in Python
Slide 1: Introduction to Transfer Learning

Transfer learning leverages knowledge gained from solving one problem and applies it to a different but related problem. This powerful approach enables models to learn from limited data by utilizing pre-trained weights, significantly reducing training time and computational resources while improving model performance.

```python
import torch
import torchvision.models as models
import torch.nn as nn

# Load a pre-trained ResNet model
base_model = models.resnet50(pretrained=True)

# Freeze the pre-trained layers
for param in base_model.parameters():
    param.requires_grad = False

# Modify the final layer for new task
num_features = base_model.fc.in_features
num_classes = 10  # New task classes
base_model.fc = nn.Linear(num_features, num_classes)
```

Slide 2: Feature Extraction Implementation

Feature extraction is a fundamental transfer learning technique where we use the pre-trained model as a fixed feature extractor. The convolutional layers act as feature extractors while only training the new classification layers for the target task.

```python
import torch.optim as optim

# Define optimizer only for the new layers
optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)

# Training loop example
def train_step(model, data, labels):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Slide 3: Fine-tuning Strategy

Fine-tuning involves unfreezing some layers of the pre-trained model and training them with a lower learning rate. This allows the model to adapt its learned features to the new task while preserving useful low-level features from the source domain.

```python
# Unfreeze last few layers
for param in base_model.layer4.parameters():
    param.requires_grad = True

# Use different learning rates
optimizer = optim.Adam([
    {'params': base_model.fc.parameters(), 'lr': 1e-3},
    {'params': base_model.layer4.parameters(), 'lr': 1e-4}
])
```

Slide 4: Domain Adaptation Mathematics

The mathematical foundation of transfer learning involves minimizing the domain discrepancy between source and target distributions. This slide presents key formulas used in domain adaptation techniques.

```python
# Mathematical formulas for domain adaptation
"""
Domain Divergence:
$$d_{\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) = 2\sup_{h \in \mathcal{H}}|\mathbb{E}_{\mathcal{D}_S}[h(x)] - \mathbb{E}_{\mathcal{D}_T}[h(x)]|$$

Transfer Risk Bound:
$$\epsilon_T(h) \leq \epsilon_S(h) + \frac{1}{2}d_{\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) + \lambda$$

where:
$$\lambda = \min_{h \in \mathcal{H}}[\epsilon_S(h) + \epsilon_T(h)]$$
"""
```

Slide 5: Transfer Learning with Computer Vision

Transfer learning is extensively used in computer vision tasks. This implementation demonstrates how to use a pre-trained ResNet model for a custom image classification task with proper data preprocessing.

```python
from torchvision import transforms
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Custom dataset class example
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
```

Slide 6: Advanced Fine-tuning Techniques

Progressive layer unfreezing is an advanced fine-tuning strategy that gradually unfreezes layers from top to bottom. This approach maintains stability during transfer learning while allowing the model to adapt more effectively to the target domain.

```python
class ProgressiveUnfreezing:
    def __init__(self, model, epochs_per_layer=3):
        self.model = model
        self.epochs_per_layer = epochs_per_layer
        
    def unfreeze_layer(self, layer_name):
        for name, param in self.model.named_parameters():
            if layer_name in name:
                param.requires_grad = True
                
    def training_schedule(self):
        layers = ['layer4', 'layer3', 'layer2', 'layer1']
        for epoch, layer in enumerate(layers):
            if epoch % self.epochs_per_layer == 0:
                self.unfreeze_layer(layer)
                print(f"Unfreezing {layer}")
```

Slide 7: Loss Function Engineering

Transfer learning effectiveness can be improved by designing appropriate loss functions that balance knowledge transfer and task-specific learning. This implementation shows a custom loss function combining cross-entropy and knowledge distillation.

```python
class TransferLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets):
        # Standard cross-entropy loss
        ce = self.ce_loss(student_logits, targets)
        
        # Knowledge distillation loss
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (self.temperature ** 2)
        
        # Combined loss
        return (1 - self.alpha) * ce + self.alpha * kd_loss
```

Slide 8: Real-world Example - Medical Image Classification

Transfer learning application in medical image classification demonstrates its practical value in domains with limited labeled data. This implementation shows a complete pipeline for adapting a pre-trained model to medical image analysis.

```python
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

class MedicalImageClassifier:
    def __init__(self, num_classes, pretrained_model='resnet50'):
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / len(dataloader), 100. * correct / total
```

Slide 9: Source Code for Medical Image Classification Results

```python
# Training and evaluation setup
def train_medical_classifier():
    model = MedicalImageClassifier(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.model.fc.parameters(), 'lr': 1e-3},
        {'params': model.model.layer4.parameters(), 'lr': 1e-4}
    ])
    
    # Sample training results
    epochs = 10
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("Training Results:")
    print("================")
    for epoch in range(epochs):
        train_loss, train_acc = model.train_epoch(train_loader, criterion, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    return results, model

# Example output:
"""
Training Results:
================
Epoch 1/10:
Train Loss: 0.8245, Train Acc: 74.32%
Epoch 2/10:
Train Loss: 0.5632, Train Acc: 82.15%
...
"""
```

Slide 10: Feature Map Visualization

Understanding how transfer learning affects internal representations is crucial. This implementation provides tools to visualize feature maps at different layers, helping to understand what knowledge is being transferred and adapted.

```python
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class FeatureMapVisualizer:
    def __init__(self, model):
        self.model = model
        self.features = {}
        
    def hook_layer(self, layer_name):
        def hook(model, input, output):
            self.features[layer_name] = output.detach()
        return hook
    
    def visualize_feature_maps(self, image, layer_name):
        # Register hook
        layer = dict([*self.model.named_modules()])[layer_name]
        layer.register_forward_hook(self.hook_layer(layer_name))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image.unsqueeze(0))
        
        # Get feature maps
        feature_maps = self.features[layer_name][0]
        
        # Plot first 16 feature maps
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for idx, ax in enumerate(axes.flat):
            if idx < feature_maps.shape[0]:
                ax.imshow(feature_maps[idx].cpu().numpy())
            ax.axis('off')
        
        return fig
```

Slide 11: Gradient-based Feature Analysis

This implementation introduces techniques for analyzing which features are most important during transfer learning using gradient-based methods, helping to understand the transfer learning process at a deeper level.

```python
class GradientAnalyzer:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        
    def save_gradients(self, grad):
        self.gradients = grad
        
    def analyze_feature_importance(self, image, target_class):
        # Register hook for gradients
        image.requires_grad_()
        
        # Forward pass
        output = self.model(image.unsqueeze(0))
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get gradients
        gradients = image.grad.data.abs()
        
        # Average across channels
        importance_map = gradients.mean(dim=1).squeeze()
        
        return importance_map.cpu().numpy()
```

Slide 12: Domain Adaptation Implementation

Domain adaptation is a crucial aspect of transfer learning when source and target domains differ significantly. This implementation shows how to measure and minimize domain discrepancy using Maximum Mean Discrepancy (MMD).

```python
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), 
                                         int(total.size(0)), 
                                         int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), 
                                         int(total.size(0)), 
                                         int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) 
                         for i in range(self.kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) 
                     for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss
```

Slide 13: Multi-task Transfer Learning

Multi-task transfer learning enables simultaneous knowledge transfer across multiple related tasks. This approach leverages shared representations while maintaining task-specific components, leading to improved performance across all target tasks.

```python
class MultiTaskTransferNet(nn.Module):
    def __init__(self, num_tasks, shared_dim=512, task_dim=256):
        super().__init__()
        # Shared feature extractor (pre-trained)
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Task-specific layers
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, shared_dim),
                nn.ReLU(),
                nn.Linear(shared_dim, task_dim),
                nn.ReLU(),
                nn.Linear(task_dim, 1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        shared_features = self.backbone(x).squeeze()
        return [head(shared_features) for head in self.task_heads]
```

Slide 14: Meta-Learning for Transfer

Meta-learning enhances transfer learning by learning how to adapt quickly to new tasks. This implementation demonstrates Model-Agnostic Meta-Learning (MAML) for efficient transfer learning across multiple domains.

```python
class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        
    def adapt(self, support_data, support_labels):
        theta = {name: param.clone() for name, param in self.model.named_parameters()}
        grads = torch.autograd.grad(
            self.loss_fn(self.model(support_data), support_labels),
            self.model.parameters()
        )
        
        # Inner loop update
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            theta[name] = param - self.inner_lr * grad
            
        return theta
    
    def meta_train_step(self, task_batch):
        meta_loss = 0
        for support, query in task_batch:
            # Adapt to task
            adapted_params = self.adapt(support['x'], support['y'])
            
            # Compute loss with adapted parameters
            with torch.set_grad_enabled(True):
                query_loss = self.loss_fn(
                    self.forward_with_params(query['x'], adapted_params),
                    query['y']
                )
                meta_loss += query_loss
                
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() / len(task_batch)
```

Slide 15: Additional Resources

*   Learning Transferable Features with Deep Adaptation Networks
    *   [https://arxiv.org/abs/1502.02791](https://arxiv.org/abs/1502.02791)
*   Deep Transfer Learning with Joint Adaptation Networks
    *   [https://arxiv.org/abs/1605.06636](https://arxiv.org/abs/1605.06636)
*   A Survey on Transfer Learning
    *   [https://arxiv.org/abs/1808.01974](https://arxiv.org/abs/1808.01974)
*   Meta-Learning: Learning to Learn Fast
    *   [https://arxiv.org/abs/1810.03548](https://arxiv.org/abs/1810.03548)
*   Progressive Neural Networks for Transfer Learning
    *   [https://arxiv.org/abs/1606.04671](https://arxiv.org/abs/1606.04671)

