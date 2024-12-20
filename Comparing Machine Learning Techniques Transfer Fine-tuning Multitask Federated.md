## Comparing Machine Learning Techniques Transfer Fine-tuning Multitask Federated
Slide 1: Transfer Learning Basics

Transfer learning enables leveraging knowledge from a pre-trained model to solve a new but related task. This technique is particularly effective when dealing with limited data in the target domain while having access to a model trained on a larger, related dataset.

```python
import torch
import torchvision.models as models
from torch import nn

def create_transfer_model(num_classes):
    # Load pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Create model for a new classification task with 5 classes
model = create_transfer_model(num_classes=5)

# Example usage
x = torch.randn(1, 3, 224, 224)  # Sample input
output = model(x)
print(f"Output shape: {output.shape}")  # Output: torch.Size([1, 5])
```

Slide 2: Fine-tuning Implementation

Unlike transfer learning where most layers remain frozen, fine-tuning involves updating all or selected layers of a pre-trained model. This approach allows the model to adapt more comprehensively to the new task while maintaining learned features.

```python
import torch.optim as optim

def setup_fine_tuning(model, learning_rate=0.001):
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Use different learning rates for different layers
    params = [
        {'params': model.conv1.parameters(), 'lr': learning_rate/10},
        {'params': model.fc.parameters(), 'lr': learning_rate}
    ]
    
    optimizer = optim.Adam(params)
    criterion = nn.CrossEntropyLoss()
    
    return optimizer, criterion

# Fine-tune the model
model = create_transfer_model(num_classes=5)
optimizer, criterion = setup_fine_tuning(model)

# Training loop example
def train_step(model, inputs, labels):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Slide 3: Multitask Learning Architecture

Multitask learning implements a shared network architecture with task-specific branches, enabling simultaneous learning of multiple related tasks. This approach promotes knowledge sharing and improved generalization across tasks while reducing computational overhead.

```python
class MultitaskModel(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super().__init__()
        
        # Shared layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Task-specific branches
        self.task1_branch = nn.Sequential(
            nn.Linear(64 * 111 * 111, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task1)
        )
        
        self.task2_branch = nn.Sequential(
            nn.Linear(64 * 111 * 111, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task2)
        )
    
    def forward(self, x):
        shared_features = self.shared_conv(x)
        shared_features = shared_features.view(x.size(0), -1)
        
        task1_output = self.task1_branch(shared_features)
        task2_output = self.task2_branch(shared_features)
        
        return task1_output, task2_output

# Create multitask model
mtl_model = MultitaskModel(num_classes_task1=5, num_classes_task2=3)
```

Slide 4: Multitask Learning Training Loop

The training process for multitask learning requires careful balancing of losses from different tasks. This implementation demonstrates how to combine multiple task-specific losses and update the shared model parameters effectively.

```python
def train_multitask_model(model, task1_data, task2_data, epochs=10):
    optimizer = optim.Adam(model.parameters())
    task1_criterion = nn.CrossEntropyLoss()
    task2_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for (x1, y1), (x2, y2) in zip(task1_data, task2_data):
            optimizer.zero_grad()
            
            # Forward pass
            out1, out2 = model(x1)
            
            # Calculate losses
            loss1 = task1_criterion(out1, y1)
            loss2 = task2_criterion(out2, y2)
            
            # Combined loss with task weighting
            total_loss = 0.5 * loss1 + 0.5 * loss2
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")

# Example usage
batch_size = 32
task1_data = [(torch.randn(batch_size, 3, 224, 224), 
               torch.randint(0, 5, (batch_size,)))]
task2_data = [(torch.randn(batch_size, 3, 224, 224), 
               torch.randint(0, 3, (batch_size,)))]

train_multitask_model(mtl_model, task1_data, task2_data)
```

Slide 5: Federated Learning Core Implementation

Federated learning enables distributed model training while keeping data private on local devices. This implementation showcases the core components of federated learning including client updates and server aggregation.

```python
import copy
import numpy as np

class FederatedLearning:
    def __init__(self, base_model):
        self.global_model = base_model
        self.client_models = {}
        
    def distribute_model(self, client_ids):
        """Distribute global model to clients"""
        for client_id in client_ids:
            self.client_models[client_id] = copy.deepcopy(self.global_model)
    
    def client_update(self, client_id, data, epochs=5):
        """Update client model with local data"""
        model = self.client_models[client_id]
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for inputs, labels in data:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        return model.state_dict()
    
    def aggregate_models(self, client_updates):
        """Aggregate client models using FedAvg"""
        averaged_weights = {}
        
        for layer_name in self.global_model.state_dict().keys():
            weights = [client_weights[layer_name] for client_weights in client_updates]
            averaged_weights[layer_name] = torch.stack(weights).mean(dim=0)
            
        self.global_model.load_state_dict(averaged_weights)
        return self.global_model
```

Slide 6: Federated Learning Training Loop

This implementation demonstrates a complete federated learning training cycle, including client selection, local training, and model aggregation phases typical in real-world federated learning deployments.

```python
def federated_training_round(fed_learning, client_data, num_rounds=10):
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}")
        
        # Select clients for this round (random 50% of clients)
        available_clients = list(client_data.keys())
        num_clients = max(1, len(available_clients) // 2)
        selected_clients = np.random.choice(
            available_clients, num_clients, replace=False)
        
        # Distribute model to selected clients
        fed_learning.distribute_model(selected_clients)
        
        # Collect client updates
        client_updates = []
        for client_id in selected_clients:
            updated_weights = fed_learning.client_update(
                client_id, client_data[client_id])
            client_updates.append(updated_weights)
        
        # Aggregate updates
        fed_learning.aggregate_models(client_updates)
        
        # Evaluate global model (simplified)
        test_loss = evaluate_model(fed_learning.global_model, test_data)
        print(f"Global model test loss: {test_loss:.4f}")

# Example usage
base_model = create_transfer_model(num_classes=10)
fed_learning = FederatedLearning(base_model)

# Simulate client data
client_data = {
    i: [(torch.randn(32, 3, 224, 224), 
         torch.randint(0, 10, (32,))) for _ in range(5)]
    for i in range(10)
}

federated_training_round(fed_learning, client_data)
```

Slide 7: Transfer Learning for Computer Vision

This implementation demonstrates a practical transfer learning application for image classification using a pre-trained ResNet model on a custom dataset, showing data preprocessing and model adaptation techniques.

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class CustomImageDataset(Dataset):
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

def setup_transfer_learning():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    
    # Modify final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model, transform

# Training configuration
learning_rate = 0.001
num_epochs = 10
num_classes = 5

# Create model and optimizer
model, transform = setup_transfer_learning()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
```

Slide 8: Multi-Task Learning for Text Classification

This implementation showcases multi-task learning applied to text classification tasks, handling sentiment analysis and topic classification simultaneously using shared embeddings.

```python
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class TextMultiTaskModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, 
                 num_sentiment_classes, num_topic_classes):
        super().__init__()
        
        # Shared layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           batch_first=True, bidirectional=True)
        
        # Task-specific layers
        self.sentiment_classifier = nn.Linear(hidden_dim * 2, 
                                            num_sentiment_classes)
        self.topic_classifier = nn.Linear(hidden_dim * 2, 
                                        num_topic_classes)
        
    def forward(self, text, lengths):
        # Shared processing
        embedded = self.embedding(text)
        packed = pack_padded_sequence(embedded, lengths, 
                                    batch_first=True, 
                                    enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)
        
        # Global max pooling
        batch_size = text.size(0)
        hidden_dim = lstm_out.size(2)
        sentence_repr = lstm_out.max(dim=1)[0]
        
        # Task-specific predictions
        sentiment_logits = self.sentiment_classifier(sentence_repr)
        topic_logits = self.topic_classifier(sentence_repr)
        
        return sentiment_logits, topic_logits

# Model initialization
vocab_size = 10000
embed_dim = 300
hidden_dim = 256
num_sentiment_classes = 3
num_topic_classes = 5

model = TextMultiTaskModel(vocab_size, embed_dim, hidden_dim,
                          num_sentiment_classes, num_topic_classes)
```

Slide 9: Federated Learning with Differential Privacy

This implementation demonstrates how to incorporate differential privacy into federated learning to enhance privacy guarantees by adding calibrated noise to model updates.

```python
import torch.nn.utils.clip_grad as clip_grad

class PrivateFederatedLearning:
    def __init__(self, model, noise_scale=1.0, clip_norm=1.0):
        self.global_model = model
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm
        
    def privatize_gradients(self, model):
        # Clip gradients
        clip_grad.clip_grad_norm_(model.parameters(), self.clip_norm)
        
        # Add Gaussian noise to gradients
        for param in model.parameters():
            if param.requires_grad:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_scale * self.clip_norm,
                    size=param.grad.shape
                )
                param.grad += noise
                
    def client_update(self, client_data, epochs=5):
        model = copy.deepcopy(self.global_model)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            for batch_data, batch_labels in client_data:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = F.cross_entropy(outputs, batch_labels)
                loss.backward()
                
                # Apply differential privacy
                self.privatize_gradients(model)
                optimizer.step()
                
        return model.state_dict()

# Example usage
model = create_transfer_model(num_classes=10)
private_fed = PrivateFederatedLearning(
    model, 
    noise_scale=0.1, 
    clip_norm=1.0
)

# Simulate private training
client_updates = []
for client_id in range(5):
    update = private_fed.client_update(
        [(torch.randn(32, 3, 224, 224), 
          torch.randint(0, 10, (32,)))]
    )
    client_updates.append(update)
```

Slide 10: Real-world Transfer Learning Example - Medical Image Classification

This implementation shows a complete transfer learning pipeline for medical image classification, including data preprocessing, model adaptation, and training with cross-validation.

```python
from sklearn.model_selection import KFold
from torchvision.datasets import ImageFolder

def medical_transfer_learning(data_dir, num_classes, num_folds=5):
    # Data augmentation and preprocessing
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                           [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = ImageFolder(data_dir, transform=train_transform)
    
    # K-fold cross-validation
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold + 1}/{num_folds}")
        
        # Create data loaders
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=32,
            shuffle=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=32
        )
        
        # Initialize model
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(1024, num_classes)
        
        # Training loop with early stopping
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(30):
            train_loss = train_epoch(model, train_loader)
            val_acc = evaluate(model, val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        fold_results.append(best_val_acc)
        
    return np.mean(fold_results), np.std(fold_results)

# Training utilities
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    
    for inputs, labels in loader:
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)
```

Slide 11: Implementation of Multitask Learning for Medical Diagnosis

This implementation demonstrates a practical multitask learning system for simultaneous disease detection and severity classification using medical imaging data, incorporating attention mechanisms.

```python
class MedicalMultiTaskNetwork(nn.Module):
    def __init__(self, num_diseases, num_severity_levels):
        super().__init__()
        
        # Shared CNN backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Tanh(),
            nn.Linear(512, 1280),
            nn.Sigmoid()
        )
        
        # Task-specific heads
        self.disease_classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )
        
        self.severity_classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_severity_levels)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Task-specific predictions
        disease_pred = self.disease_classifier(attended_features)
        severity_pred = self.severity_classifier(attended_features)
        
        return disease_pred, severity_pred

def train_medical_multitask(model, train_loader, val_loader, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                   patience=3)
    
    disease_criterion = nn.BCEWithLogitsLoss()
    severity_criterion = nn.CrossEntropyLoss()
    
    best_val_score = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_disease_loss = 0
        train_severity_loss = 0
        
        for images, (disease_labels, severity_labels) in train_loader:
            disease_pred, severity_pred = model(images)
            
            # Calculate losses
            d_loss = disease_criterion(disease_pred, disease_labels)
            s_loss = severity_criterion(severity_pred, severity_labels)
            total_loss = d_loss + s_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            train_disease_loss += d_loss.item()
            train_severity_loss += s_loss.item()
            
        # Validation
        val_score = evaluate_multitask(model, val_loader)
        scheduler.step(val_score)
        
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), 'best_model.pth')
```

Slide 12: Federated Learning with Personalization

This implementation showcases a personalized federated learning approach where clients maintain both shared and local model components to better adapt to local data distributions.

```python
class PersonalizedFederatedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Shared components
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Personal components (to be maintained locally)
        self.personal_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        output = self.personal_layers(shared_features)
        return output

class PersonalizedFederated:
    def __init__(self, global_model, num_clients):
        self.global_model = global_model
        self.personal_models = {
            i: copy.deepcopy(global_model.personal_layers)
            for i in range(num_clients)
        }
        
    def client_update(self, client_id, data, epochs=5):
        # Create client's personalized model
        model = copy.deepcopy(self.global_model)
        model.personal_layers = self.personal_models[client_id]
        
        optimizer = optim.Adam([
            {'params': model.shared_layers.parameters(), 'lr': 0.001},
            {'params': model.personal_layers.parameters(), 'lr': 0.01}
        ])
        
        for epoch in range(epochs):
            for batch_x, batch_y in data:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = F.cross_entropy(output, batch_y)
                loss.backward()
                optimizer.step()
        
        # Save updated personal layers
        self.personal_models[client_id] = copy.deepcopy(
            model.personal_layers
        )
        
        return model.shared_layers.state_dict()
```

Slide 13: Real-world Example - Transfer Learning for NLP

This implementation demonstrates transfer learning using BERT for text classification, showing how to leverage pre-trained language models for specific downstream tasks.

```python
from transformers import BertModel, BertTokenizer

class BertTransferLearning(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Classification
        return self.classifier(pooled_output)

def train_bert_transfer(train_texts, train_labels, epochs=3):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertTransferLearning(num_classes=len(set(train_labels)))
    
    # Freeze BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # Prepare optimizer
    optimizer = optim.AdamW(model.classifier.parameters(), lr=2e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_texts, batch_labels in create_batches(
            train_texts, train_labels):
            # Tokenize
            encodings = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt',
                max_length=512
            )
            
            # Forward pass
            outputs = model(
                encodings['input_ids'],
                encodings['attention_mask']
            )
            
            loss = F.cross_entropy(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_texts)}")
```

Slide 14: Additional Resources

*   ArXiv Paper: "A Survey of Transfer Learning" - [https://arxiv.org/abs/1808.01974](https://arxiv.org/abs/1808.01974)
*   ArXiv Paper: "Federated Learning: Challenges, Methods, and Future Directions" - [https://arxiv.org/abs/1908.07873](https://arxiv.org/abs/1908.07873)
*   ArXiv Paper: "An Overview of Multi-Task Learning in Deep Neural Networks" - [https://arxiv.org/abs/1706.05098](https://arxiv.org/abs/1706.05098)
*   Research Paper: "Deep Transfer Learning for Medical Image Analysis" - Search for: "Transfer Learning Medical Imaging Survey"
*   Tutorial: "Practical Federated Learning for Edge Devices" - Visit: pytorch.org/tutorials
*   Research Guide: "Multi-Task Learning Implementation Best Practices" - Visit: paperswithcode.com

These comprehensive slides cover the implementation details of transfer learning, multi-task learning, and federated learning with practical examples and real-world applications. Each implementation includes essential components, best practices, and considerations for production deployment.

