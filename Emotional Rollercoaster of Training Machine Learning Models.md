## Emotional Rollercoaster of Training Machine Learning Models
Slide 1: Early Stopping Implementation

Early stopping is a regularization technique that prevents overfitting by monitoring the model's performance on a validation set during training. When the validation performance stops improving or begins to degrade, training is halted to preserve the model's generalization ability.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0
```

Slide 2: Implementing Model Training with Early Stopping

This implementation demonstrates a complete training loop with early stopping integration using PyTorch, showcasing how to monitor validation loss and stop training when the model begins to overfit.

```python
import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                output = model(X)
                val_loss += criterion(output, y).item()
        
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
```

Slide 3: Custom Neural Network Architecture

The implementation defines a flexible neural network architecture with dropout layers for regularization and batch normalization to stabilize training, essential components when dealing with overfitting prevention.

```python
class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example instantiation
model = CustomNN(input_size=10, hidden_sizes=[64, 32], output_size=1)
```

Slide 4: Learning Rate Scheduler

A learning rate scheduler dynamically adjusts the learning rate during training, helping prevent overfitting by reducing the learning rate when the validation loss plateaus.

```python
class CustomLRScheduler:
    def __init__(self, optimizer, factor=0.5, patience=3, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.counter = 0
        self.best_loss = None
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self._adjust_learning_rate()
                self.counter = 0
        else:
            self.best_loss = val_loss
            self.counter = 0
    
    def _adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
```

Slide 5: Cross-Validation Implementation

Cross-validation helps assess model performance and prevent overfitting by evaluating the model on different data splits. This implementation includes a k-fold cross-validation strategy with early stopping.

```python
def cross_validate(model_class, X, y, k_folds=5):
    fold_size = len(X) // k_folds
    scores = []
    
    for fold in range(k_folds):
        # Create fold indices
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        
        # Split data
        X_train = torch.cat([X[:val_start], X[val_end:]])
        y_train = torch.cat([y[:val_start], y[val_end:]])
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        
        # Initialize and train model
        model = model_class()
        train_with_early_stopping(model, 
                                (X_train, y_train),
                                (X_val, y_val))
        
        # Evaluate
        with torch.no_grad():
            val_pred = model(X_val)
            score = nn.MSELoss()(val_pred, y_val)
        scores.append(score.item())
    
    return np.mean(scores), np.std(scores)
```

Slide 6: Model Performance Visualization

A comprehensive visualization system to track model performance during training, helping identify overfitting patterns by plotting training and validation metrics over time.

```python
import matplotlib.pyplot as plt
import numpy as np

class PerformanceVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        
    def update(self, train_loss, val_loss, accuracy=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if accuracy is not None:
            self.accuracies.append(accuracy)
    
    def plot(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 4))
        
        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy subplot if available
        if self.accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, self.accuracies, 'g-', label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
```

Slide 7: Real-World Example - Credit Card Fraud Detection

Implementation of a fraud detection model with early stopping and regularization techniques, demonstrating practical application in handling imbalanced datasets.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def prepare_fraud_detection_data(df):
    # Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)
    
    # Create weighted sampler for imbalanced data
    weights = torch.FloatTensor([
        1.0 / (len(y) - sum(y)) if label == 0 
        else 1.0 / sum(y) for label in y
    ])
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights)
    )
    
    # Create dataset and loader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        sampler=sampler
    )
    
    return loader, scaler
```

Slide 8: Fraud Detection Model Architecture

A specialized neural network architecture designed for fraud detection, incorporating dropout and batch normalization layers with considerations for class imbalance.

```python
class FraudDetectionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)
```

Slide 9: Custom Loss Function for Imbalanced Data

Implementation of a weighted binary cross-entropy loss function that handles class imbalance by applying different weights to positive and negative samples.

```python
class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred, target):
        if self.pos_weight is None:
            self.pos_weight = torch.tensor(
                [target.shape[0] / torch.sum(target)]
            )
            
        loss = -(self.pos_weight * target * torch.log(pred + 1e-7) + 
                (1 - target) * torch.log(1 - pred + 1e-7))
        return loss.mean()

# Usage example
criterion = WeightedBinaryCrossEntropy()
```

Slide 10: Performance Metrics Implementation

A comprehensive set of evaluation metrics specifically designed for imbalanced classification problems, including precision, recall, F1-score, and ROC curves.

```python
class PerformanceMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    def update(self, predictions, targets):
        pred_labels = (predictions >= self.threshold).float()
        self.true_positives += torch.sum((pred_labels == 1) & (targets == 1))
        self.false_positives += torch.sum((pred_labels == 1) & (targets == 0))
        self.true_negatives += torch.sum((pred_labels == 0) & (targets == 0))
        self.false_negatives += torch.sum((pred_labels == 0) & (targets == 1))
    
    def get_metrics(self):
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (self.true_positives + self.true_negatives) / \
                  (self.true_positives + self.true_negatives + 
                   self.false_positives + self.false_negatives)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'accuracy': accuracy.item()
        }
```

Slide 11: Results Visualization and Analysis

A comprehensive implementation for visualizing model results including ROC curves, confusion matrices, and prediction distribution analysis to better understand model performance.

```python
class ResultsAnalyzer:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.predictions = []
        self.true_labels = []
        
    def collect_predictions(self):
        self.model.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                outputs = self.model(X)
                self.predictions.extend(outputs.numpy())
                self.true_labels.extend(y.numpy())
                
        self.predictions = np.array(self.predictions)
        self.true_labels = np.array(self.true_labels)
    
    def plot_roc_curve(self):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(self.true_labels, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
```

Slide 12: Advanced Learning Rate Management

Implementation of a cyclical learning rate scheduler with warmup period and cosine annealing, which helps prevent local minima and improves model convergence.

```python
class CyclicalLRScheduler:
    def __init__(self, optimizer, base_lr, max_lr, step_size, warmup_steps=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.warmup_steps = warmup_steps
        self.cycle = 0
        self.step_count = 0
        
    def step(self):
        if self.step_count < self.warmup_steps:
            lr = self.base_lr + (self.max_lr - self.base_lr) * \
                 (self.step_count / self.warmup_steps)
        else:
            cycle_progress = (self.step_count - self.warmup_steps) % (2 * self.step_size)
            if cycle_progress < self.step_size:
                # Increasing phase
                lr = self.base_lr + (self.max_lr - self.base_lr) * \
                     (cycle_progress / self.step_size)
            else:
                # Decreasing phase
                cycle_progress -= self.step_size
                lr = self.max_lr - (self.max_lr - self.base_lr) * \
                     (cycle_progress / self.step_size)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.step_count += 1
        return lr
```

Slide 13: Training Pipeline Integration

A complete training pipeline that integrates all previous components including early stopping, metrics tracking, and learning rate management.

```python
class TrainingPipeline:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stopping = EarlyStopping(patience=5)
        self.lr_scheduler = CyclicalLRScheduler(
            optimizer, base_lr=1e-4, max_lr=1e-2, 
            step_size=100, warmup_steps=50
        )
        self.metrics = PerformanceMetrics()
        self.visualizer = PerformanceVisualizer()
        
    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for X, y in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.lr_scheduler.step()
        return train_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        self.metrics.reset()
        with torch.no_grad():
            for X, y in self.val_loader:
                output = self.model(X)
                val_loss += self.criterion(output, y).item()
                self.metrics.update(output, y)
        return val_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            metrics = self.metrics.get_metrics()
            
            self.visualizer.update(train_loss, val_loss, metrics['accuracy'])
            self.early_stopping(val_loss)
            
            if self.early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        self.visualizer.plot()
```

Slide 14: Additional Resources

*   Learning Rate Scheduling Strategies - [https://arxiv.org/abs/1506.01186](https://arxiv.org/abs/1506.01186)
*   Early Stopping Methods Survey - [https://arxiv.org/abs/1905.12666](https://arxiv.org/abs/1905.12666)
*   Neural Networks Regularization Techniques - [https://arxiv.org/abs/1801.09060](https://arxiv.org/abs/1801.09060)
*   Cyclical Learning Rates for Training Neural Networks - [https://arxiv.org/abs/1506.01186](https://arxiv.org/abs/1506.01186)
*   Batch Normalization: Accelerating Deep Network Training - [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

