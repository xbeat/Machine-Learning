## Regression and Classification Loss Functions Cheat Sheet
Slide 1: Mean Bias Error (MBE) - Fundamental Regression Loss

Mean Bias Error provides insights into the systematic over or underestimation in regression models by calculating the average difference between predicted and actual values. While rarely used as a primary training objective, it serves as a crucial diagnostic tool for model bias assessment.

```python
import numpy as np

def mean_bias_error(y_true, y_pred):
    """
    Calculate Mean Bias Error
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    Returns:
        float: MBE score
    """
    mbe = np.mean(y_pred - y_true)
    return mbe

# Example usage
y_true = np.array([2.5, 3.0, 4.5, 5.0])
y_pred = np.array([2.7, 3.2, 4.3, 5.1])
mbe = mean_bias_error(y_true, y_pred)
print(f"Mean Bias Error: {mbe:.4f}")  # Output: Mean Bias Error: 0.0750
```

Slide 2: Mean Absolute Error (MAE) Implementation

Mean Absolute Error provides a robust measure of prediction accuracy by calculating the average magnitude of errors without considering their direction. This implementation includes both numpy and pure Python approaches for educational purposes.

```python
import numpy as np

def mae_numpy(y_true, y_pred):
    """
    Calculate MAE using numpy
    """
    return np.mean(np.abs(y_true - y_pred))

def mae_pure_python(y_true, y_pred):
    """
    Calculate MAE using pure Python
    """
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

# Example usage with synthetic data
np.random.seed(42)
y_true = np.random.normal(0, 1, 1000)
y_pred = y_true + np.random.normal(0, 0.5, 1000)

print(f"MAE (numpy): {mae_numpy(y_true, y_pred):.4f}")
print(f"MAE (pure): {mae_pure_python(y_true, y_pred):.4f}")
```

Slide 3: Mean Squared Error (MSE) with Gradient Computation

Mean Squared Error penalizes larger errors more heavily by squaring the differences. This implementation includes gradient computation, essential for understanding how MSE works in gradient-based optimization algorithms.

```python
import numpy as np

class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_true, y_pred):
        """
        Forward pass for MSE computation
        """
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.square(y_pred - y_true))
    
    def gradient(self):
        """
        Compute gradient of MSE with respect to predictions
        """
        return 2 * (self.y_pred - self.y_true) / len(self.y_true)

# Example usage
mse_loss = MSELoss()
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 2.2, 2.8, 4.1])

loss = mse_loss.forward(y_true, y_pred)
grad = mse_loss.gradient()

print(f"MSE Loss: {loss:.4f}")
print(f"Gradient: {grad}")
```

Slide 4: Root Mean Squared Error (RMSE) with Scikit-learn Integration

Root Mean Squared Error provides an intuitive error metric in the same units as the target variable. This implementation demonstrates both custom implementation and integration with scikit-learn's metrics.

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def custom_rmse(y_true, y_pred):
    """
    Custom RMSE implementation
    """
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

# Generate synthetic regression data
np.random.seed(42)
X = np.random.normal(0, 1, (100, 1))
y_true = 2 * X.squeeze() + np.random.normal(0, 0.5, 100)
y_pred = 1.9 * X.squeeze() + np.random.normal(0, 0.3, 100)

# Calculate RMSE using both methods
custom_score = custom_rmse(y_true, y_pred)
sklearn_score = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Custom RMSE: {custom_score:.4f}")
print(f"Sklearn RMSE: {sklearn_score:.4f}")
```

Slide 5: Huber Loss Implementation

Huber Loss combines the best properties of MSE and MAE by using quadratic loss for small errors and linear loss for large ones. This implementation includes the delta parameter that controls the transition point.

```python
import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate Huber Loss
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        delta: Threshold parameter
    """
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * np.abs(error) - 0.5 * np.square(delta)
    
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

# Example with different delta values
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([0.8, 2.5, 2.7, 4.2])

for delta in [0.5, 1.0, 2.0]:
    loss = huber_loss(y_true, y_pred, delta)
    print(f"Huber Loss (delta={delta}): {loss:.4f}")
```

Slide 6: Log-Cosh Loss Function

Log-cosh loss approximates Huber Loss but is twice differentiable everywhere, making it particularly suitable for gradient-based optimization. It combines the robustness of Huber loss with smoother derivatives, beneficial for neural network training.

```python
import numpy as np

def log_cosh_loss(y_true, y_pred):
    """
    Calculate Log-Cosh Loss
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    Returns:
        float: Log-cosh loss value
    """
    error = y_pred - y_true
    return np.mean(np.log(np.cosh(error)))

def log_cosh_gradient(y_true, y_pred):
    """
    Calculate gradient of log-cosh loss
    """
    error = y_pred - y_true
    return np.tanh(error)

# Example usage with synthetic data
np.random.seed(42)
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.3, 2.8, 4.2, 4.8])

loss = log_cosh_loss(y_true, y_pred)
gradient = log_cosh_gradient(y_true, y_pred)

print(f"Log-Cosh Loss: {loss:.4f}")
print(f"Gradient: {gradient}")
```

Slide 7: Binary Cross-Entropy Loss Implementation

Binary Cross-Entropy Loss is fundamental for binary classification tasks, measuring the divergence between predicted probabilities and true binary labels. This implementation includes numerical stability through clipping.

```python
import numpy as np

class BinaryCrossEntropy:
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon
    
    def __call__(self, y_true, y_pred):
        """
        Calculate Binary Cross-Entropy Loss
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities [0, 1]
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        bce = -np.mean(
            y_true * np.log(y_pred) + 
            (1 - y_true) * np.log(1 - y_pred)
        )
        return bce
    
    def gradient(self, y_true, y_pred):
        """
        Calculate gradient of BCE loss
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

# Example usage
bce_loss = BinaryCrossEntropy()
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7])

loss = bce_loss(y_true, y_pred)
grad = bce_loss.gradient(y_true, y_pred)

print(f"BCE Loss: {loss:.4f}")
print(f"Gradient first 3 samples: {grad[:3]}")
```

Slide 8: Hinge Loss with SVM Implementation

Hinge Loss is crucial for Support Vector Machine training, creating a margin-based classifier. This implementation includes both the basic hinge loss and a complete SVM classifier using gradient descent.

```python
import numpy as np

class SVMWithHingeLoss:
    def __init__(self, learning_rate=0.01, lambda_param=0.01):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
    
    def hinge_loss(self, y_true, y_pred):
        """
        Calculate Hinge Loss
        """
        return np.maximum(0, 1 - y_true * y_pred)
    
    def fit(self, X, y, epochs=100):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(epochs):
            for idx, x_i in enumerate(X):
                y_i = y[idx]
                condition = y_i * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_i * x_i)
                    self.b -= self.lr * y_i

# Example usage
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

svm = SVMWithHingeLoss()
svm.fit(X, y)

# Make predictions
y_pred = np.sign(np.dot(X, svm.w) + svm.b)
accuracy = np.mean(y == y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 9: Categorical Cross-Entropy Implementation

Categorical Cross-Entropy extends binary cross-entropy to multi-class scenarios. This implementation includes numerical stability measures and supports both one-hot encoded and sparse label formats.

```python
import numpy as np

def categorical_crossentropy(y_true, y_pred, epsilon=1e-15):
    """
    Calculate Categorical Cross-Entropy Loss
    Args:
        y_true: One-hot encoded true labels
        y_pred: Predicted probabilities
        epsilon: Small constant for numerical stability
    """
    # Clip predictions
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # If labels are sparse, convert to one-hot
    if len(y_true.shape) == 1:
        num_classes = y_pred.shape[1]
        y_true = np.eye(num_classes)[y_true]
    
    # Calculate cross-entropy
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example usage with synthetic data
num_classes = 3
num_samples = 100

# Generate random probabilities
y_pred = np.random.random((num_samples, num_classes))
y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)  # Normalize to get valid probabilities

# Generate random true labels
y_true = np.random.randint(0, num_classes, num_samples)

# Calculate loss with both sparse and one-hot labels
sparse_loss = categorical_crossentropy(y_true, y_pred)
one_hot_true = np.eye(num_classes)[y_true]
one_hot_loss = categorical_crossentropy(one_hot_true, y_pred)

print(f"Sparse Label Loss: {sparse_loss:.4f}")
print(f"One-Hot Label Loss: {one_hot_loss:.4f}")
```

Slide 10: KL Divergence Loss Implementation

Kullback-Leibler Divergence quantifies the difference between probability distributions. This implementation demonstrates both symmetric and asymmetric KL divergence calculations, commonly used in variational autoencoders and knowledge distillation.

```python
import numpy as np

def kl_divergence(p, q, epsilon=1e-15):
    """
    Calculate KL Divergence between two probability distributions
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small constant for numerical stability
    """
    # Ensure valid probability distributions
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    return np.sum(p * np.log(p / q))

def symmetric_kl_divergence(p, q, epsilon=1e-15):
    """
    Calculate Symmetric KL Divergence (Jensen-Shannon Divergence)
    """
    return 0.5 * (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon))

# Example usage
# Generate two different probability distributions
p = np.array([0.2, 0.5, 0.3])
q = np.array([0.1, 0.4, 0.5])

kl_div = kl_divergence(p, q)
symmetric_kl = symmetric_kl_divergence(p, q)

print(f"KL Divergence (P||Q): {kl_div:.4f}")
print(f"Symmetric KL Divergence: {symmetric_kl:.4f}")
```

Slide 11: Real-world Example - House Price Prediction

This comprehensive example demonstrates the application of multiple regression loss functions on a real estate dataset, comparing their performance and characteristics in a practical scenario.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MultiLossRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def initialize_parameters(self, n_features):
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
    def compute_losses(self, y_true, y_pred):
        mse = np.mean(np.square(y_true - y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        return {'mse': mse, 'mae': mae, 'rmse': rmse}
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)
        
        history = {'mse': [], 'mae': [], 'rmse': []}
        
        for _ in range(self.epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients (MSE loss)
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Track metrics
            losses = self.compute_losses(y, y_pred)
            for k, v in losses.items():
                history[k].append(v)
                
        return history

# Generate synthetic house price data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 3)  # Features: size, bedrooms, location
true_weights = np.array([100000, 50000, 75000])
true_bias = 200000
noise = np.random.normal(0, 25000, n_samples)
y = np.dot(X, true_weights) + true_bias + noise

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model and evaluate
model = MultiLossRegressor(learning_rate=0.01, epochs=100)
history = model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = np.dot(X_test_scaled, model.weights) + model.bias
final_losses = model.compute_losses(y_test, y_pred)

print("Final Test Metrics:")
for metric, value in final_losses.items():
    print(f"{metric.upper()}: ${value:,.2f}")
```

Slide 12: Real-world Example - Multi-class Classification with Multiple Losses

This example implements a neural network classifier that compares different classification loss functions on the MNIST-like dataset, showcasing practical differences between BCE, CCE, and Hinge Loss.

```python
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class MultiLossClassifier:
    def __init__(self, input_size, num_classes):
        self.W = np.random.randn(input_size, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        self.num_classes = num_classes
        
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        scores = np.dot(X, self.W) + self.b
        probs = self.softmax(scores)
        return probs
    
    def compute_losses(self, y_true, y_pred):
        # Cross-entropy loss
        ce_loss = -np.mean(np.sum(y_true * np.log(np.clip(y_pred, 1e-15, 1.0)), axis=1))
        
        # Hinge loss (one-vs-all)
        margins = y_pred - y_pred[range(len(y_true)), y_true.argmax(axis=1)].reshape(-1, 1) + 1.0
        margins[range(len(y_true)), y_true.argmax(axis=1)] = 0
        hinge_loss = np.mean(np.sum(np.maximum(0, margins), axis=1))
        
        return {'cross_entropy': ce_loss, 'hinge': hinge_loss}

# Generate synthetic multi-class data
np.random.seed(42)
n_samples = 1000
n_features = 20
n_classes = 5

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)
lb = LabelBinarizer()
y_one_hot = lb.fit_transform(y)

# Train and evaluate
clf = MultiLossClassifier(n_features, n_classes)
probs = clf.forward(X)
losses = clf.compute_losses(y_one_hot, probs)

print("Training Losses:")
for loss_name, loss_value in losses.items():
    print(f"{loss_name}: {loss_value:.4f}")
```

Slide 13: Additional Resources

*   Understanding Deep Learning Optimization and Loss Functions
    *   [https://arxiv.org/abs/1908.01958](https://arxiv.org/abs/1908.01958)
*   A Comparative Study of Loss Functions for Deep Learning
    *   [https://arxiv.org/abs/2009.01827](https://arxiv.org/abs/2009.01827)
*   On Loss Functions for Deep Neural Networks in Classification
    *   [https://arxiv.org/abs/1702.05659](https://arxiv.org/abs/1702.05659)
*   Implementing Advanced Loss Functions in Neural Networks
    *   [https://dl.acm.org/doi/10.1145/3992424](https://dl.acm.org/doi/10.1145/3992424)
*   Empirical Loss Function Analysis for Regression and Classification
    *   [https://www.sciencedirect.com/science/article/pii/S0893608019303181](https://www.sciencedirect.com/science/article/pii/S0893608019303181)

