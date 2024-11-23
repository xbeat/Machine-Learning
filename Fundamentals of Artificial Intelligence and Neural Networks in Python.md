## Fundamentals of Artificial Intelligence and Neural Networks in Python
Slide 1: Introduction to Artificial Neural Networks

Artificial Neural Networks are computational models inspired by biological neural networks, composed of interconnected processing nodes that transform input data through layers to produce meaningful outputs. These networks form the foundation of modern deep learning systems.

```python
# Basic structure of a neural network
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize network architecture
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, self.output_size))
```

Slide 2: Mathematical Foundations of Neural Networks

Understanding the mathematical principles behind neural networks is crucial for implementing effective solutions. The fundamental operations involve matrix multiplication, activation functions, and gradient-based optimization techniques.

```python
# Mathematical foundations in code
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(X, W1, b1, W2, b2):
    # Mathematical representation of forward propagation
    """
    Mathematical equations:
    $$Z1 = XW1 + b1$$
    $$A1 = sigmoid(Z1)$$
    $$Z2 = A1W2 + b2$$
    $$output = sigmoid(Z2)$$
    """
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = sigmoid(Z2)
    return output, A1
```

Slide 3: Activation Functions in Neural Networks

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common functions include ReLU, sigmoid, and tanh, each serving specific purposes in different network architectures.

```python
import numpy as np

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
```

Slide 4: Implementing Backpropagation

Backpropagation is the cornerstone of neural network training, allowing the network to learn by adjusting weights based on calculated error gradients. The process involves computing partial derivatives through the chain rule.

```python
def backward_propagation(X, y, output, A1, W1, W2):
    m = X.shape[0]
    
    # Compute gradients
    dZ2 = output - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2
```

Slide 5: Basic Neural Network Implementation

This implementation demonstrates a complete neural network class with training capabilities, incorporating both forward and backward propagation mechanisms for binary classification tasks.

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_init(input_size, hidden_size, output_size)
        self.learning_rate = learning_rate
    
    def weights_init(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
```

Slide 6: Training the Neural Network

The training process involves iteratively updating network parameters through epochs, minimizing the loss function while monitoring convergence. This implementation includes batch processing and learning rate optimization.

```python
def train(self, X, y, epochs=1000):
    for epoch in range(epochs):
        # Forward propagation
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        
        # Calculate loss
        loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        
        # Backpropagation
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2) / X.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
        
        dZ1 = np.dot(dZ2, self.W2.T) * (A1 * (1 - A1))
        dW1 = np.dot(X.T, dZ1) / X.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        
        # Update parameters
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
```

Slide 7: Introduction to PyTorch Framework

PyTorch provides a powerful framework for building and training neural networks, offering automatic differentiation and GPU acceleration capabilities while maintaining a pythonic interface for deep learning development.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
```

Slide 8: Data Preprocessing for Neural Networks

Data preprocessing is crucial for neural network performance, involving normalization, feature scaling, and categorical encoding. This implementation demonstrates essential preprocessing techniques for real-world datasets.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data):
    # Numerical features scaling
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Categorical encoding
    le = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    return data, scaler, le

# Example usage
def prepare_dataset():
    X = pd.read_csv('dataset.csv')
    X_processed, scaler, le = preprocess_data(X)
    return X_processed, scaler, le
```

Slide 9: Real-world Example: Credit Card Fraud Detection

Implementation of a neural network for detecting fraudulent credit card transactions, demonstrating practical application of deep learning in financial security systems.

```python
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class FraudDetectionModel(nn.Module):
    def __init__(self):
        super(FraudDetectionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
```

Slide 10: Training Process for Fraud Detection

The training implementation includes batch processing, loss calculation, and model evaluation metrics specific to fraud detection, incorporating class imbalance handling techniques.

```python
def train_fraud_model(model, train_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
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

Slide 11: Advanced Neural Network Architectures

Advanced architectures incorporate residual connections, attention mechanisms, and layer normalization to handle complex patterns and dependencies in data. This implementation showcases modern architectural components.

```python
class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedNeuralNetwork, self).__init__()
        
        # Residual block
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h = self.network(x)
        h = h + self.residual_block(h)  # Residual connection
        return self.output_layer(h)
```

Slide 12: Real-world Example: Time Series Prediction

A practical implementation of neural networks for time series forecasting, incorporating temporal dependencies and sequential data processing for financial market prediction.

```python
class TimeSeriesPredictor(nn.Module):
    def __init__(self, seq_length, n_features, hidden_size=64):
        super(TimeSeriesPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        prediction = self.regressor(last_time_step)
        return prediction
```

Slide 13: Performance Metrics and Model Evaluation

Comprehensive evaluation metrics implementation for assessing neural network performance across different tasks, including classification and regression metrics.

```python
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error

class ModelEvaluator:
    @staticmethod
    def classification_metrics(y_true, y_pred, threshold=0.5):
        y_pred_binary = (y_pred > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='binary'
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': np.mean(y_true == y_pred_binary)
        }
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
```

Slide 14: Additional Resources

*   "Deep Learning Book" - Ian Goodfellow et al. ([https://www.deeplearningbook.org](https://www.deeplearningbook.org))
*   "Neural Networks and Deep Learning" tutorial series ([https://arxiv.org/abs/1709.02664](https://arxiv.org/abs/1709.02664))
*   "A Comprehensive Survey of Deep Learning Approaches" ([https://arxiv.org/abs/2004.03705](https://arxiv.org/abs/2004.03705))
*   "Efficient BackProp" - Yann LeCun et al. ([http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf))
*   "Deep Learning: A Practitioner's Approach" (Search on Google Scholar)
*   PyTorch documentation and tutorials ([https://pytorch.org/tutorials](https://pytorch.org/tutorials))

