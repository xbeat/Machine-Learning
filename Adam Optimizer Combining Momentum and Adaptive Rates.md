## Adam Optimizer Combining Momentum and Adaptive Rates
Slide 1: Introduction to Adam Optimizer

Adam (Adaptive Moment Estimation) optimizer combines the benefits of momentum-based and adaptive learning rate methods, specifically integrating RMSprop and momentum. It maintains per-parameter learning rates that are adapted based on estimates of first and second moments of the gradients.

```python
# Mathematical representation of Adam update rules:
"""
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}}\hat{m}_t$$
"""
```

Slide 2: Basic Adam Implementation

A fundamental implementation of Adam optimizer showcasing its core mechanisms and parameter updates while maintaining both momentum and adaptive learning rate components.

```python
import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
```

Slide 3: Linear Regression with Adam

The implementation demonstrates Adam's effectiveness in a practical scenario using linear regression, showing how it adapts learning rates and handles different scales of features.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)
y = y.reshape(-1, 1)

# Linear regression model with Adam optimizer
class LinearRegressionAdam:
    def __init__(self, n_features):
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = np.zeros((1, 1))
        self.optimizer = Adam()
        
    def forward(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_gradients(self, X, y, y_pred):
        m = X.shape[0]
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        return dw, db
    
    def train_step(self, X, y):
        y_pred = self.forward(X)
        dw, db = self.compute_gradients(X, y, y_pred)
        self.weights = self.optimizer.update(self.weights, dw)
        self.bias = self.optimizer.update(self.bias, db)
        return np.mean((y_pred - y) ** 2)
```

Slide 4: Training Loop and Visualization

```python
import matplotlib.pyplot as plt

# Training process
model = LinearRegressionAdam(X.shape[1])
losses = []

for epoch in range(100):
    loss = model.train_step(X, y)
    losses.append(loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Visualization of training progress
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 5: Neural Network Implementation with Adam

An implementation of a neural network using Adam optimizer, demonstrating its effectiveness in training deep learning models with multiple layers.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        self.optimizers_w = []
        self.optimizers_b = []
        
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
            self.optimizers_w.append(Adam())
            self.optimizers_b.append(Adam())
```

Slide 6: Neural Network Forward and Backward Pass

```python
def forward(self, X):
    self.activations = [X]
    for i in range(len(self.weights)):
        Z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
        A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        self.activations.append(A)
    return self.activations[-1]

def backward(self, X, y, learning_rate):
    m = X.shape[0]
    delta = self.activations[-1] - y
    
    for i in range(len(self.weights) - 1, -1, -1):
        dW = np.dot(self.activations[i].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        
        if i > 0:
            delta = np.dot(delta, self.weights[i].T) * (self.activations[i] * (1 - self.activations[i]))
        
        self.weights[i] = self.optimizers_w[i].update(self.weights[i], dW)
        self.biases[i] = self.optimizers_b[i].update(self.biases[i], db)
```

Slide 7: Practical Example - Binary Classification

The implementation demonstrates Adam's effectiveness in a real-world binary classification problem using the Wisconsin Breast Cancer dataset.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = NeuralNetwork([30, 16, 8, 1])
```

Slide 8: Training and Evaluation Code

```python
def train_model(model, X_train, y_train, epochs=100):
    losses = []
    for epoch in range(epochs):
        predictions = model.forward(X_train)
        loss = -np.mean(y_train * np.log(predictions + 1e-8) + 
                       (1-y_train) * np.log(1-predictions + 1e-8))
        model.backward(X_train, y_train, 0.001)
        losses.append(loss)
        
        if epoch % 10 == 0:
            accuracy = np.mean((predictions > 0.5) == y_train)
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return losses

# Train the model
losses = train_model(model, X_train, y_train)

# Evaluate on test set
test_predictions = model.forward(X_test)
test_accuracy = np.mean((test_predictions > 0.5) == y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

Slide 9: Performance Visualization and Metrics

Code for visualizing training progress and computing key performance metrics for binary classification using Adam optimizer.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_metrics(model, X_train, y_train, X_test, y_test):
    # Generate predictions
    train_pred = model.forward(X_train)
    test_pred = model.forward(X_test)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, test_pred)
    pr_auc = auc(recall, precision)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    
    ax2.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

Slide 10: Implementing Learning Rate Decay

Learning rate decay helps Adam optimizer achieve better convergence by gradually reducing the learning rate during training, preventing oscillations in later epochs.

```python
class AdamWithDecay(Adam):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, decay_rate=0.95):
        super().__init__(learning_rate, beta1, beta2, epsilon)
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate
        
    def update(self, params, grads):
        # Decay learning rate
        self.learning_rate = self.initial_learning_rate * \
                            (self.decay_rate ** (self.t / 1000))
        
        return super().update(params, grads)

# Example usage
optimizer = AdamWithDecay(learning_rate=0.001, decay_rate=0.95)
```

Slide 11: Comparison with Other Optimizers

Implementation comparing Adam against SGD and RMSprop on a complex optimization problem to demonstrate its advantages.

```python
import numpy as np
from typing import Callable, Dict

class OptimizerComparison:
    def __init__(self, optimizers: Dict[str, Callable]):
        self.optimizers = optimizers
        self.histories = {name: [] for name in optimizers.keys()}
    
    def rosenbrock(self, x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def compare(self, iterations=1000):
        starting_point = np.array([-1.0, 1.0])
        for name, optimizer in self.optimizers.items():
            params = starting_point.copy()
            for _ in range(iterations):
                # Compute gradients
                x, y = params
                dx = -2*(1-x) + 200*(y-x**2)*(-2*x)
                dy = 200*(y-x**2)
                grads = np.array([dx, dy])
                
                # Update parameters
                params = optimizer.update(params, grads)
                self.histories[name].append(self.rosenbrock(*params))
```

Slide 12: Real-world Example: Image Classification

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
```

Slide 13: Training Results Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_results(histories: Dict[str, list]):
    plt.figure(figsize=(12, 6))
    for name, history in histories.items():
        plt.plot(history, label=name)
    
    plt.title('Optimizer Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Add convergence speed analysis
    convergence_steps = {
        name: np.argmin(history) 
        for name, history in histories.items()
    }
    
    for name, steps in convergence_steps.items():
        print(f"{name} converged in {steps} steps")
```

Slide 14: Additional Resources

*   "Adam: A Method for Stochastic Optimization" - [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   "On the Convergence of Adam and Beyond" - [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   "Adaptive Gradient Methods with Dynamic Bound of Learning Rate" - [https://arxiv.org/abs/1902.09843](https://arxiv.org/abs/1902.09843)
*   "Decoupled Weight Decay Regularization" - [https://www.google.com/search?q=Decoupled+Weight+Decay+Regularization+paper](https://www.google.com/search?q=Decoupled+Weight+Decay+Regularization+paper)
*   For implementation details and best practices: [https://www.tensorflow.org/api\_docs/python/tf/keras/optimizers/Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

