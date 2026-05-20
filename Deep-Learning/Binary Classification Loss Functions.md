## Binary Classification Loss Functions
Slide 1: Binary Cross-Entropy Loss Function

Binary cross-entropy loss, also known as log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. It quantifies the difference between predicted probability distributions for binary classification tasks by calculating how uncertain our prediction is compared to the true label.

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    # Avoid numerical instability by clipping predictions
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate binary cross entropy
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

# Example usage
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
loss = binary_cross_entropy(y_true, y_pred)
print(f"Binary Cross-Entropy Loss: {loss:.4f}")
```

Slide 2: Binary Cross-Entropy Loss Implementation with PyTorch

A practical implementation of binary cross-entropy loss using PyTorch, demonstrating its application in a neural network for binary classification. The example includes gradient calculation and backpropagation, essential components of model training.

```python
import torch
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Create synthetic data
X = torch.randn(100, 5)
y = torch.randint(0, 2, (100, 1)).float()

# Initialize model and loss
model = BinaryClassifier(5)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

Slide 3: Custom Binary Cross-Entropy with Regularization

Binary cross-entropy loss with L2 regularization helps prevent overfitting by penalizing large weights. This implementation demonstrates how to combine the standard binary cross-entropy loss with a regularization term for improved model generalization.

```python
import numpy as np

class RegularizedBCELoss:
    def __init__(self, lambda_reg=0.01):
        self.lambda_reg = lambda_reg
    
    def __call__(self, y_true, y_pred, weights):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate BCE loss
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add L2 regularization term
        l2_reg = 0.5 * self.lambda_reg * np.sum(weights ** 2)
        
        return bce + l2_reg

# Example usage
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.3])
weights = np.array([0.5, -0.3, 0.2, 0.1])

loss_fn = RegularizedBCELoss(lambda_reg=0.01)
total_loss = loss_fn(y_true, y_pred, weights)
print(f"Regularized BCE Loss: {total_loss:.4f}")
```

Slide 4: Focal Loss Implementation

Focal Loss is a modified form of cross-entropy that addresses class imbalance by down-weighting easy examples and focusing training on hard negatives. This implementation provides a robust solution for handling imbalanced datasets in binary classification.

```python
import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Calculate focal loss
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    
    focal = -alpha_t * np.power(1 - pt, gamma) * np.log(pt)
    return np.mean(focal)

# Example with imbalanced dataset
y_true = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
y_pred = np.array([0.9, 0.1, 0.2, 0.1, 0.3, 0.8, 0.2, 0.1, 0.2, 0.1])

loss = focal_loss(y_true, y_pred)
print(f"Focal Loss: {loss:.4f}")
```

Slide 5: Weighted Binary Cross-Entropy

Weighted binary cross-entropy addresses class imbalance by assigning different weights to positive and negative classes. This approach is particularly useful when dealing with datasets where one class significantly outnumbers the other, helping prevent bias towards the majority class.

```python
import numpy as np
import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, y_pred, y_true):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        
        # Calculate weighted loss
        loss = -(self.pos_weight * y_true * torch.log(y_pred) + 
                (1 - y_true) * torch.log(1 - y_pred))
        return torch.mean(loss)

# Example usage
pos_weight = 2.0  # Weight for positive class
criterion = WeightedBCELoss(pos_weight)

# Sample data
y_true = torch.tensor([1., 0., 1., 0., 1.])
y_pred = torch.tensor([0.8, 0.2, 0.7, 0.1, 0.9])

loss = criterion(y_pred, y_true)
print(f"Weighted BCE Loss: {loss.item():.4f}")
```

Slide 6: Real-world Example: Credit Card Fraud Detection

A comprehensive implementation of binary classification for credit card fraud detection, demonstrating data preprocessing, model creation, and evaluation using binary cross-entropy loss. This example showcases handling imbalanced financial transaction data.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class FraudDetector(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Simulate credit card transaction data
np.random.seed(42)
n_samples = 10000
n_features = 30

# Generate synthetic data with imbalance (1% fraud)
X = np.random.randn(n_samples, n_features)
y = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01])

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)

# Initialize model and training components
model = FraudDetector(n_features)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
batch_size = 32

for epoch in range(n_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
```

Slide 7: Results Analysis for Credit Card Fraud Detection

This slide presents the evaluation metrics and performance analysis of the fraud detection model implemented in the previous slide, including precision, recall, and F1-score calculations.

```python
from sklearn.metrics import classification_report, roc_auc_score
import torch

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate ROC-AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {auc_score:.4f}")

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Example confusion matrix visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

Slide 8: Hinge Loss Implementation

Hinge loss, commonly used in Support Vector Machines, provides an alternative to binary cross-entropy for binary classification. This implementation demonstrates its application and compares its characteristics with BCE loss.

```python
import numpy as np

def hinge_loss(y_true, y_pred):
    # Convert binary labels to {-1, 1}
    y_true = 2 * y_true - 1
    
    # Calculate hinge loss
    loss = np.maximum(0, 1 - y_true * y_pred)
    return np.mean(loss)

class HingeLossClassifier:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, epochs=100):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(epochs):
            for idx in range(n_samples):
                condition = y[idx] * (np.dot(X[idx], self.weights) + self.bias)
                
                if condition < 1:
                    self.weights += self.learning_rate * y[idx] * X[idx]
                    self.bias += self.learning_rate * y[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Example usage
X = np.random.randn(100, 2)
y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])

classifier = HingeLossClassifier()
classifier.fit(X, y)

# Calculate loss
y_pred = np.dot(X, classifier.weights) + classifier.bias
loss = hinge_loss(y, y_pred)
print(f"Hinge Loss: {loss:.4f}")
```

Slide 9: Log-Cosh Loss Implementation

Log-cosh loss serves as a smooth approximation to the absolute error and is particularly useful for binary classification when you want the benefits of L2 loss without its extreme sensitivity to outliers. It combines the best properties of L1 and L2 losses.

```python
import numpy as np
import torch
import torch.nn as nn

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true):
        def log_cosh(x):
            return x + torch.log(1 + torch.exp(-2 * x)) - np.log(2)
        
        return torch.mean(log_cosh(y_pred - y_true))

# Example implementation
class BinaryClassifierLogCosh(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layer(x)

# Training example
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000, 1)).float()

model = BinaryClassifierLogCosh(10)
criterion = LogCoshLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Single training iteration
optimizer.zero_grad()
output = model(X)
loss = criterion(output, y)
loss.backward()
optimizer.step()

print(f"Log-Cosh Loss: {loss.item():.4f}")
```

Slide 10: Real-world Example: Customer Churn Prediction

A comprehensive implementation of binary classification for predicting customer churn, demonstrating the application of different loss functions and handling class imbalance in a business context.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

class ChurnPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Generate synthetic churn data
def generate_churn_data(n_samples=10000):
    np.random.seed(42)
    features = {
        'usage_minutes': np.random.normal(600, 200, n_samples),
        'contract_length': np.random.choice([1, 12, 24], n_samples),
        'monthly_charges': np.random.normal(70, 30, n_samples),
        'customer_service_calls': np.random.poisson(2, n_samples)
    }
    df = pd.DataFrame(features)
    
    # Generate churn probability based on features
    churn_prob = 1 / (1 + np.exp(-(
        -2 
        + 0.005 * (df['usage_minutes'] - 600)
        - 0.1 * df['contract_length']
        + 0.03 * (df['monthly_charges'] - 70)
        + 0.2 * df['customer_service_calls']
    )))
    
    df['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    return df

# Prepare data
df = generate_churn_data()
X = df.drop('churn', axis=1).values
y = df['churn'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)

# Initialize model and training components
model = ChurnPredictor(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
batch_size = 64
n_epochs = 50

for epoch in range(n_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
```

Slide 11: Evaluation Metrics for Churn Prediction

This slide demonstrates comprehensive evaluation metrics for the churn prediction model, including ROC curves, precision-recall curves, and business-specific metrics.

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

def evaluate_churn_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).numpy()
        
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC curve
    ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    # Precision-Recall curve
    ax2.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate business metrics
    threshold = 0.5
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Customer retention rate
    retention_rate = 1 - np.mean(y_pred)
    print(f"Predicted Customer Retention Rate: {retention_rate:.2%}")
    
    # Cost savings analysis (assuming $500 retention cost per customer)
    true_positives = np.sum((y_test == 1) & (y_pred == 1))
    potential_savings = true_positives * 500
    print(f"Potential Cost Savings: ${potential_savings:,.2f}")

# Evaluate the model
evaluate_churn_model(model, X_test, y_test)
```

Slide 12: Advanced Binary Classification Loss Functions

A comparative analysis of sophisticated loss functions for binary classification, including Jaccard/IoU loss and Tversky loss. These loss functions offer unique properties for handling specific types of classification problems and imbalanced datasets.

```python
import torch
import torch.nn as nn
import numpy as np

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred) - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1 - y_true) * y_pred)
        fn = torch.sum(y_true * (1 - y_pred))
        
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return 1 - tversky

# Example usage
y_true = torch.tensor([1., 0., 1., 1., 0.])
y_pred = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.2])

jaccard_loss = JaccardLoss()
tversky_loss = TverskyLoss()

print(f"Jaccard Loss: {jaccard_loss(y_pred, y_true):.4f}")
print(f"Tversky Loss: {tversky_loss(y_pred, y_true):.4f}")
```

Slide 13: Loss Function Comparison Framework

A comprehensive framework for comparing different binary classification loss functions, allowing practitioners to make informed decisions about which loss function best suits their specific use case.

```python
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class LossComparisonFramework:
    def __init__(self, loss_functions):
        self.loss_functions = loss_functions
        self.history = {name: {'loss': [], 'accuracy': [], 'f1': []} 
                       for name in loss_functions.keys()}
    
    def evaluate_losses(self, y_true, y_pred_proba, threshold=0.5):
        results = {}
        y_pred = (y_pred_proba >= threshold).astype(np.float32)
        
        for name, loss_fn in self.loss_functions.items():
            # Calculate loss
            loss_value = loss_fn(
                torch.tensor(y_pred_proba), 
                torch.tensor(y_true)
            ).item()
            
            # Calculate metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            results[name] = {
                'loss': loss_value,
                'accuracy': acc,
                'f1': f1
            }
        
        return results

    def plot_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy comparison
        for name in self.loss_functions.keys():
            axes[0].plot(self.history[name]['accuracy'], label=name)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Plot F1 score comparison
        for name in self.loss_functions.keys():
            axes[1].plot(self.history[name]['f1'], label=name)
        axes[1].set_title('F1 Score Comparison')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
loss_functions = {
    'BCE': nn.BCELoss(),
    'Jaccard': JaccardLoss(),
    'Tversky': TverskyLoss()
}

# Generate synthetic imbalanced dataset
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 10)
y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])

# Initialize framework
framework = LossComparisonFramework(loss_functions)

# Simulate predictions
y_pred_proba = np.random.rand(n_samples)
results = framework.evaluate_losses(y, y_pred_proba)

# Print results
for name, metrics in results.items():
    print(f"\n{name} Loss Function:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
```

Slide 14: Additional Resources

*   ArXiv paper: "Focal Loss for Dense Object Detection" - [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)
*   ArXiv paper: "The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks" - [https://arxiv.org/abs/1705.08790](https://arxiv.org/abs/1705.08790)
*   ArXiv paper: "Tversky loss function for image segmentation using 3D fully convolutional deep networks" - [https://arxiv.org/abs/1706.05721](https://arxiv.org/abs/1706.05721)
*   Suggested search terms for additional resources:
    *   "Binary classification loss functions comparison"
    *   "Deep learning loss functions for imbalanced datasets"
    *   "Advanced loss functions for neural networks"

