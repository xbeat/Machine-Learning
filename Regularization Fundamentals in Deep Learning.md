## Regularization Fundamentals in Deep Learning
Slide 1: Understanding L2 Regularization Implementation

L2 regularization adds a penalty term to the loss function proportional to the squared magnitude of weights. This helps prevent overfitting by constraining the model's capacity and ensuring weights don't grow too large during training, leading to better generalization.

```python
import numpy as np

class NeuralNetworkWithL2:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.lambda_reg = lambda_reg
        
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1)
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2)
        return self.softmax(self.z2)
    
    def loss_with_l2(self, y_true, y_pred):
        ce_loss = -np.mean(y_true * np.log(y_pred + 1e-10))
        l2_loss = (self.lambda_reg/2) * (np.sum(self.weights1**2) + np.sum(self.weights2**2))
        return ce_loss + l2_loss
```

Slide 2: Implementing L1 Regularization

L1 regularization uses absolute values of weights instead of squared values, promoting sparsity in the model by driving some weights exactly to zero. This helps in feature selection and creates simpler, more interpretable models.

```python
class NeuralNetworkWithL1:
    def __init__(self, input_size, hidden_size, output_size, lambda_reg=0.01):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.lambda_reg = lambda_reg
    
    def loss_with_l1(self, y_true, y_pred):
        ce_loss = -np.mean(y_true * np.log(y_pred + 1e-10))
        l1_loss = self.lambda_reg * (np.sum(np.abs(self.weights1)) + 
                                    np.sum(np.abs(self.weights2)))
        return ce_loss + l1_loss
    
    def gradient_with_l1(self):
        l1_grad1 = self.lambda_reg * np.sign(self.weights1)
        l1_grad2 = self.lambda_reg * np.sign(self.weights2)
        return l1_grad1, l1_grad2
```

Slide 3: Implementing Dropout

Dropout randomly deactivates neurons during training, preventing co-adaptation and creating an implicit ensemble of multiple neural networks. This technique significantly reduces overfitting and improves model generalization.

```python
class DropoutLayer:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
    
    def forward(self, inputs, training=True):
        if not training:
            return inputs
            
        self.mask = np.random.binomial(1, 1-self.dropout_rate, 
                                     size=inputs.shape) / (1-self.dropout_rate)
        return inputs * self.mask
    
    def backward(self, gradient):
        return gradient * self.mask
```

Slide 4: Data Augmentation Implementation

Data augmentation artificially expands the training dataset by applying various transformations to existing samples. This technique helps the model learn invariant features and improves generalization by exposing it to different variations of the input.

```python
import cv2
import numpy as np

class ImageAugmenter:
    def __init__(self, rotation_range=20, zoom_range=0.15):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
    
    def augment(self, image):
        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        height, width = image.shape[:2]
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (width, height))
        
        # Random zoom
        scale = np.random.uniform(1-self.zoom_range, 1+self.zoom_range)
        M = cv2.getRotationMatrix2D((width/2, height/2), 0, scale)
        zoomed = cv2.warpAffine(rotated, M, (width, height))
        
        return zoomed
```

Slide 5: Early Stopping Implementation

Early stopping monitors the validation loss during training and stops when it starts to increase, preventing overfitting. This implementation includes patience and minimum delta parameters for robust stopping criteria.

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop
```

Slide 6: Batch Normalization Implementation

Batch normalization stabilizes training by normalizing layer inputs, reducing internal covariate shift. This implementation includes both training and inference modes, with running statistics for test-time normalization.

```python
class BatchNormalization:
    def __init__(self, input_dim, epsilon=1e-8, momentum=0.9):
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        self.epsilon = epsilon
        self.momentum = momentum
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean + 
                               (1 - self.momentum) * mean)
            self.running_var = (self.momentum * self.running_var + 
                              (1 - self.momentum) * var)
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta
```

Slide 7: Elastic Net Regularization

Elastic Net combines L1 and L2 regularization, providing both feature selection and weight magnitude control. This implementation allows fine-tuning of the balance between L1 and L2 penalties through the l1\_ratio parameter.

```python
class ElasticNetRegularization:
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def compute_regularization(self, weights):
        l1_term = self.alpha * self.l1_ratio * np.sum(np.abs(weights))
        l2_term = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(weights**2)
        return l1_term + l2_term
        
    def compute_gradient(self, weights):
        l1_grad = self.alpha * self.l1_ratio * np.sign(weights)
        l2_grad = self.alpha * (1 - self.l1_ratio) * weights
        return l1_grad + l2_grad
```

Slide 8: Real-world Example: Image Classification with Regularization

This complete example demonstrates the application of multiple regularization techniques in a CNN for image classification, including dropout, batch normalization, and L2 regularization.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(RegularizedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

Slide 9: Training Pipeline with Multiple Regularization Techniques

Implementation of a comprehensive training pipeline incorporating various regularization methods, including learning rate scheduling and gradient clipping for stable training.

```python
def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    early_stopping = EarlyStopping(patience=10)
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add L2 regularization
            l2_lambda = 0.01
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation phase
        val_loss = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
```

Slide 10: Performance Metrics Implementation

A comprehensive implementation of various metrics to evaluate model performance and overfitting detection. This includes training-validation loss comparison and regularization effectiveness measurements.

```python
class RegularizationMetrics:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.weight_norms = []
        
    def compute_metrics(self, model, train_loss, val_loss):
        l1_norm = sum(p.abs().sum().item() for p in model.parameters())
        l2_norm = sum(p.pow(2.0).sum().item() for p in model.parameters())
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.weight_norms.append((l1_norm, l2_norm))
        
        overfitting_score = val_loss / train_loss
        regularization_effect = l2_norm / (l1_norm + 1e-10)
        
        return {
            'overfitting_score': overfitting_score,
            'regularization_effect': regularization_effect,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm
        }
```

Slide 11: Advanced Dropout Variations

Implementation of advanced dropout techniques including Spatial Dropout and Variational Dropout, which provide more sophisticated regularization for specific neural network architectures.

```python
class AdvancedDropout:
    class SpatialDropout2D(nn.Module):
        def __init__(self, drop_prob):
            super().__init__()
            self.drop_prob = drop_prob
            
        def forward(self, x):
            if not self.training or self.drop_prob == 0:
                return x
            
            # Spatial dropout maintains channel coherence
            mask = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], 1, 1) * 
                                 (1 - self.drop_prob)).to(x.device)
            mask = mask.expand_as(x)
            return mask * x / (1 - self.drop_prob)
    
    class VariationalDropout(nn.Module):
        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha
            
        def forward(self, x):
            if not self.training:
                return x
                
            batch_size = x.size(0)
            eps = torch.randn_like(x)
            mask = torch.exp(0.5 * torch.log(self.alpha + 1e-8))
            return x + mask * eps
```

Slide 12: Real-world Example: NLP Model with Combined Regularization

Implementation of a text classification model incorporating multiple regularization techniques, demonstrating their combined effect in natural language processing tasks.

```python
class RegularizedTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        
        self.transformer_block = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.MultiheadAttention(embed_dim, num_heads),
            nn.Dropout(0.1),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Weight initialization with regularization in mind
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
```

Slide 13: Regularization Results Visualization

Implementation of visualization tools to analyze the effect of different regularization techniques on model performance and weight distributions.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class RegularizationVisualizer:
    def plot_regularization_effects(self, metrics):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training vs Validation Loss
        axes[0,0].plot(metrics['train_losses'], label='Train')
        axes[0,0].plot(metrics['val_losses'], label='Validation')
        axes[0,0].set_title('Loss Curves')
        axes[0,0].legend()
        
        # Weight Distribution
        sns.histplot(metrics['weight_values'], ax=axes[0,1])
        axes[0,1].set_title('Weight Distribution')
        
        # L1/L2 Norm Evolution
        axes[1,0].plot(metrics['l1_norms'], label='L1 Norm')
        axes[1,0].plot(metrics['l2_norms'], label='L2 Norm')
        axes[1,0].set_title('Weight Norm Evolution')
        axes[1,0].legend()
        
        # Overfitting Score
        axes[1,1].plot(metrics['overfitting_scores'])
        axes[1,1].set_title('Overfitting Score')
        
        plt.tight_layout()
        return fig
```

Slide 14: Additional Resources

*   arXiv:1412.6980 - "Adam: A Method for Stochastic Optimization" [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   arXiv:1502.03167 - "Batch Normalization: Accelerating Deep Network Training" [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
*   arXiv:1207.0580 - "Improving Neural Networks by Preventing Co-adaptation of Feature Detectors" [https://arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580)
*   Search terms for further research:
    *   "Deep Learning Regularization Techniques"
    *   "Modern Regularization Methods in Neural Networks"
    *   "Adaptive Regularization Methods"

