## Differences Between Pretraining Finetuning and Transfer Learning
Slide 1: Introduction to Pretraining

Pretraining is a fundamental technique in machine learning where a model is trained on a large dataset to learn general features and patterns before being adapted for specific tasks. This initial training phase creates a robust foundation of learned representations.

```python
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Example of loading a pretrained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Pretraining objective: Masked Language Modeling
text = "The cat sits on the [MASK]."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Get hidden states
hidden_states = outputs.last_hidden_state
print(f"Hidden state shape: {hidden_states.shape}")
# Output: Hidden state shape: torch.Size([1, 8, 768])
```

Slide 2: Implementing a Simple Pretraining Task

We'll implement a basic autoencoder for pretraining on MNIST data, demonstrating how to create a model that learns useful feature representations through unsupervised learning.

```python
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Setup training data
transform = transforms.ToTensor()
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

Slide 3: Pretraining Implementation

Here we implement the training loop for our autoencoder, showing how the model learns to reconstruct input data and extract meaningful features during the pretraining phase.

```python
def pretrain_autoencoder(model, train_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, data.view(-1, 784))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

# Train the model
model = Autoencoder()
pretrain_autoencoder(model, train_loader)
```

Slide 4: Fine-tuning Fundamentals

Fine-tuning involves taking a pretrained model and adapting it to a specific task by updating its parameters using a smaller, task-specific dataset while maintaining the knowledge learned during pretraining.

```python
class FineTunedClassifier(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        # Freeze pretrained encoder weights
        self.encoder = pretrained_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Add new classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # 10 classes for MNIST
        )
    
    def forward(self, x):
        x = x.view(-1, 784)
        features = self.encoder(x)
        return self.classifier(features)
```

Slide 5: Implementing Fine-tuning

This implementation demonstrates how to fine-tune a pretrained model for a specific classification task, showing the process of adapting learned features to new objectives.

```python
def finetune_classifier(pretrained_model, train_loader, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FineTunedClassifier(pretrained_model.encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.2f}%')

# Fine-tune the model
finetune_classifier(model, train_loader)
```

Slide 6: Transfer Learning Architecture

Transfer learning leverages knowledge from one domain to improve learning in another domain. This implementation shows how to adapt a pretrained convolutional neural network for a new image classification task.

```python
import torchvision.models as models
import torch.nn as nn

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load pretrained ResNet
        self.resnet = models.resnet18(pretrained=True)
        
        # Freeze all layers except the last few
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
            
        # Modify final layer for new task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)
```

Slide 7: Domain Adaptation Implementation

Domain adaptation is a specific case of transfer learning where we adapt a model trained on one domain to perform well on a different but related domain while preserving learned features.

```python
class DomainAdaptation(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Binary domain classification
        )
        self.task_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        reverse_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        task_output = self.task_classifier(features)
        return task_output, domain_output

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
```

Slide 8: Progressive Fine-tuning Strategy

Progressive fine-tuning gradually unfreezes and trains deeper layers of a pretrained model, allowing for better adaptation while maintaining learned features. This implementation shows the layer-by-layer unfreezing process.

```python
class ProgressiveFinetuning:
    def __init__(self, model, num_epochs_per_layer=3):
        self.model = model
        self.num_epochs_per_layer = num_epochs_per_layer
        
    def unfreeze_layer(self, layer):
        for param in layer.parameters():
            param.requires_grad = True
            
    def train_layer(self, layer, train_loader, criterion, optimizer):
        for epoch in range(self.num_epochs_per_layer):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
    def progressive_finetune(self, train_loader):
        layers = list(self.model.children())
        criterion = nn.CrossEntropyLoss()
        
        # Start with last layer and move backwards
        for i in range(len(layers)-1, -1, -1):
            self.unfreeze_layer(layers[i])
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=0.001 * (0.1 ** i)  # Decrease learning rate for earlier layers
            )
            self.train_layer(layers[i], train_loader, criterion, optimizer)

# Usage example
model = TransferLearningModel()
progressive_trainer = ProgressiveFinetuning(model)
progressive_trainer.progressive_finetune(train_loader)
```

Slide 9: Knowledge Distillation Implementation

Knowledge distillation transfers knowledge from a large teacher model to a smaller student model. This implementation shows how to train a compact model using the outputs of a pretrained larger model.

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, student_logits, teacher_logits, targets, alpha=0.5):
        # Softmax with temperature
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=1)
        soft_pred = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # Distillation loss
        distillation_loss = self.kl_div(soft_pred, soft_targets) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        return alpha * distillation_loss + (1 - alpha) * ce_loss

def train_with_distillation(teacher_model, student_model, train_loader, epochs=10):
    criterion = DistillationLoss()
    optimizer = torch.optim.Adam(student_model.parameters())
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            student_logits = student_model(inputs)
            loss = criterion(student_logits, teacher_logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

Slide 10: Real-world Example - Medical Image Classification

This implementation demonstrates a complete transfer learning pipeline for medical image classification, including data preprocessing, model adaptation, and evaluation metrics commonly used in healthcare applications.

```python
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = load_dicom_image(self.image_paths[idx])  # Custom DICOM loader
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class MedicalImageClassifier:
    def __init__(self, num_classes):
        self.model = models.densenet121(pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def train_model(self, train_loader, val_loader, epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        best_auc = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels.float())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_auc = self.evaluate(val_loader)
            scheduler.step(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}')
    
    def evaluate(self, data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                outputs = torch.sigmoid(self.model(images))
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return roc_auc_score(all_labels, all_preds)
```

Slide 11: Real-world Example - Natural Language Processing Transfer

This implementation shows how to fine-tune a pretrained BERT model for a specific NLP task, including text preprocessing and evaluation metrics.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                 max_length=max_length, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class BERTClassifier:
    def __init__(self, num_labels, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        
    def train_model(self, train_texts, train_labels, val_texts, val_labels, epochs=3):
        train_dataset = TextClassificationDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = TextClassificationDataset(val_texts, val_labels, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps=0,
                                                  num_training_steps=total_steps)
        
        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
            
            # Validation
            metrics = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}:')
            print(f'Validation Metrics: {metrics}')
    
    def evaluate(self, data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
```

Slide 12: Evaluating Transfer Learning Performance

Transfer learning effectiveness can be measured through various metrics and visualization techniques. This implementation demonstrates how to evaluate and compare different transfer learning approaches.

```python
class TransferLearningEvaluator:
    def __init__(self):
        self.metrics_history = {
            'baseline': {},
            'transfer': {},
            'finetuned': {}
        }
    
    def evaluate_model(self, model, test_loader, model_type):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        predictions = []
        true_labels = []
        feature_vectors = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                outputs = model(data)
                
                # Store predictions and features
                if isinstance(outputs, tuple):
                    features, logits = outputs
                    feature_vectors.extend(features.cpu().numpy())
                else:
                    logits = outputs
                
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.numpy())
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(true_labels, predictions),
            'feature_vectors': np.array(feature_vectors) if feature_vectors else None
        }
        
        self.metrics_history[model_type] = results
        return results
    
    def visualize_feature_space(self, reduction_method='tsne'):
        plt.figure(figsize=(15, 5))
        
        for idx, (model_type, results) in enumerate(self.metrics_history.items()):
            if results.get('feature_vectors') is not None:
                features = results['feature_vectors']
                
                if reduction_method == 'tsne':
                    reducer = TSNE(n_components=2, random_state=42)
                else:
                    reducer = PCA(n_components=2)
                
                reduced_features = reducer.fit_transform(features)
                
                plt.subplot(1, 3, idx + 1)
                plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=true_labels)
                plt.title(f'{model_type} Feature Space')
                plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, training_history):
        plt.figure(figsize=(12, 4))
        
        for model_type, history in training_history.items():
            plt.plot(history['val_accuracy'], label=f'{model_type} Validation')
            plt.plot(history['train_accuracy'], label=f'{model_type} Training')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves Comparison')
        plt.legend()
        plt.show()
```

Slide 13: Advanced Pretraining Techniques

Implementation of advanced pretraining techniques including curriculum learning and multi-task pretraining for improved model generalization.

```python
class CurriculumPretrainer:
    def __init__(self, model, num_difficulty_levels=3):
        self.model = model
        self.num_difficulty_levels = num_difficulty_levels
        self.difficulty_schedulers = {
            'linear': lambda epoch, max_epochs: min(epoch / max_epochs * self.num_difficulty_levels, 
                                                  self.num_difficulty_levels - 1),
            'step': lambda epoch, max_epochs: int(epoch / (max_epochs/self.num_difficulty_levels))
        }
    
    def get_curriculum_batch(self, data, labels, current_difficulty):
        # Example difficulty metrics: sample length, complexity, etc.
        difficulties = self.calculate_sample_difficulties(data)
        mask = difficulties <= current_difficulty
        return data[mask], labels[mask]
    
    def train_with_curriculum(self, train_loader, epochs, scheduler_type='linear'):
        scheduler = self.difficulty_schedulers[scheduler_type]
        optimizer = torch.optim.Adam(self.model.parameters())
        
        for epoch in range(epochs):
            current_difficulty = scheduler(epoch, epochs)
            
            for batch_data, batch_labels in train_loader:
                # Filter batch based on current difficulty
                curr_data, curr_labels = self.get_curriculum_batch(
                    batch_data, batch_labels, current_difficulty)
                
                if len(curr_data) == 0:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(curr_data)
                loss = self.calculate_loss(outputs, curr_labels)
                loss.backward()
                optimizer.step()

class MultiTaskPretrainer:
    def __init__(self, shared_encoder, task_heads):
        self.shared_encoder = shared_encoder
        self.task_heads = nn.ModuleDict(task_heads)
        self.task_weights = nn.Parameter(torch.ones(len(task_heads)))
        
    def forward(self, x, task=None):
        features = self.shared_encoder(x)
        
        if task is not None:
            return self.task_heads[task](features)
        
        return {task: head(features) for task, head in self.task_heads.items()}
    
    def train_step(self, batch, optimizer):
        total_loss = 0
        
        for task, (data, labels) in batch.items():
            outputs = self(data, task)
            loss = self.task_specific_loss(outputs, labels, task)
            weighted_loss = self.task_weights[task] * loss
            total_loss += weighted_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        
        # Update task weights using gradient norm scaling
        with torch.no_grad():
            grads = [p.grad.norm(p=2) for p in self.shared_encoder.parameters()]
            grad_norms = torch.stack(grads)
            self.task_weights.data = F.softmax(grad_norms, dim=0)
        
        optimizer.step()
        return total_loss.item()
```

Slide 14: Additional Resources

*   ArXiv Papers:
*   "A Survey of Transfer Learning" - [https://arxiv.org/abs/1808.01974](https://arxiv.org/abs/1808.01974)
*   "Pre-trained Models for Natural Language Processing" - [https://arxiv.org/abs/2003.08271](https://arxiv.org/abs/2003.08271)
*   "How transferable are features in deep neural networks?" - [https://arxiv.org/abs/1411.1792](https://arxiv.org/abs/1411.1792)
*   "Curriculum Learning for Natural Language Understanding" - [https://arxiv.org/abs/2010.12582](https://arxiv.org/abs/2010.12582)
*   "Multi-Task Learning Using Uncertainty to Weigh Losses" - [https://arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115)
*   Recommended Search Terms:
*   "Recent advances in transfer learning techniques"
*   "Pretraining strategies for deep learning"
*   "Fine-tuning best practices for transformers"
*   "Domain adaptation methods in machine learning"

