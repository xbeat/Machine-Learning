## Role of Cost Functions in Machine Learning
Slide 1: Fundamentals of Cost Functions

A cost function, also known as loss function, measures the difference between predicted and actual values in machine learning models. It quantifies the error in predictions and guides the optimization process during model training by providing a scalar value representing prediction accuracy.

```python
import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE) cost function
    Args:
        y_true: actual values
        y_pred: predicted values
    Returns:
        mse: mean squared error value
    """
    mse = np.mean(np.square(y_true - y_pred))
    return mse

# Example usage
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.2, 1.9, 3.1, 3.8])
cost = mean_squared_error(y_true, y_pred)
print(f"MSE Cost: {cost:.4f}")  # Output: MSE Cost: 0.0275
```

Slide 2: Mathematical Foundations of Cost Functions

Understanding the mathematical principles behind cost functions is crucial for implementing and optimizing machine learning models effectively. Common cost functions include Mean Squared Error, Cross-Entropy, and Hinge Loss, each serving specific purposes in different scenarios.

```python
# Common Cost Function Formulas in LaTeX notation
"""
Mean Squared Error:
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$

Binary Cross-Entropy:
$$BCE = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y_i}) + (1-y_i)\log(1-\hat{y_i})]$$

Hinge Loss:
$$L = \max(0, 1 - y\hat{y})$$
"""
```

Slide 3: Cross-Entropy Loss Implementation

Cross-entropy loss is particularly useful for classification problems, measuring the difference between predicted probability distributions and actual class labels. This implementation includes numerical stability through clipping to prevent logarithm of zero.

```python
def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """
    Implement binary cross-entropy loss with numerical stability
    Args:
        y_true: actual labels (0 or 1)
        y_pred: predicted probabilities
        epsilon: small value to prevent log(0)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.95])
loss = cross_entropy_loss(y_true, y_pred)
print(f"Cross-Entropy Loss: {loss:.4f}")  # Output: Cross-Entropy Loss: 0.1520
```

Slide 4: Gradient Descent Optimization

Gradient descent optimizes cost functions by iteratively adjusting model parameters in the direction that minimizes the loss. This process involves calculating partial derivatives of the cost function with respect to each parameter.

```python
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """
    Implement gradient descent for linear regression
    Args:
        X: input features
        y: target values
        learning_rate: step size for parameter updates
        epochs: number of training iterations
    """
    m = len(y)
    theta = np.zeros(X.shape[1])
    costs = []
    
    for _ in range(epochs):
        # Compute predictions
        y_pred = np.dot(X, theta)
        
        # Compute gradients
        gradients = (1/m) * np.dot(X.T, (y_pred - y))
        
        # Update parameters
        theta -= learning_rate * gradients
        
        # Compute and store cost
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        costs.append(cost)
    
    return theta, costs

# Example usage
X = np.random.randn(100, 2)
y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(100)*0.1
theta, costs = gradient_descent(X, y)
print(f"Final parameters: {theta}")
print(f"Final cost: {costs[-1]:.4f}")
```

Slide 5: Regularized Cost Functions

Regularization prevents overfitting by adding penalty terms to the cost function, controlling model complexity. L1 (Lasso) and L2 (Ridge) regularization modify the basic cost function by incorporating parameter magnitude penalties with different characteristics.

```python
def regularized_cost_function(y_true, y_pred, weights, lambda_l1=0.01, lambda_l2=0.01):
    """
    Implement cost function with both L1 and L2 regularization
    Args:
        y_true: actual values
        y_pred: predicted values
        weights: model parameters
        lambda_l1: L1 regularization strength
        lambda_l2: L2 regularization strength
    """
    n_samples = len(y_true)
    mse = np.mean(np.square(y_true - y_pred))
    l1_penalty = lambda_l1 * np.sum(np.abs(weights))
    l2_penalty = lambda_l2 * np.sum(np.square(weights))
    
    total_cost = mse + l1_penalty + l2_penalty
    return total_cost, mse, l1_penalty, l2_penalty

# Example usage
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.2, 1.9, 3.1, 3.8])
weights = np.array([0.5, -0.3, 0.8])

total_cost, mse, l1, l2 = regularized_cost_function(y_true, y_pred, weights)
print(f"Total Cost: {total_cost:.4f}")
print(f"MSE: {mse:.4f}")
print(f"L1 Penalty: {l1:.4f}")
print(f"L2 Penalty: {l2:.4f}")
```

Slide 6: Custom Cost Function Design

Creating custom cost functions allows for specialized optimization objectives tailored to specific machine learning tasks. This implementation demonstrates how to design a custom cost function with asymmetric penalties for over and under-prediction.

```python
class CustomCostFunction:
    def __init__(self, under_prediction_weight=1.5, over_prediction_weight=1.0):
        """
        Initialize custom cost function with asymmetric penalties
        Args:
            under_prediction_weight: penalty multiplier for under-predictions
            over_prediction_weight: penalty multiplier for over-predictions
        """
        self.under_pred_weight = under_prediction_weight
        self.over_pred_weight = over_prediction_weight
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute asymmetric cost based on prediction errors
        """
        errors = y_pred - y_true
        under_pred_mask = errors < 0
        over_pred_mask = errors >= 0
        
        under_pred_cost = np.sum(np.square(errors[under_pred_mask])) * self.under_pred_weight
        over_pred_cost = np.sum(np.square(errors[over_pred_mask])) * self.over_pred_weight
        
        return (under_pred_cost + over_pred_cost) / len(y_true)

# Example usage
custom_cost = CustomCostFunction(under_prediction_weight=1.5, over_prediction_weight=1.0)
y_true = np.array([10, 20, 30, 40])
y_pred = np.array([9, 21, 28, 41])
cost = custom_cost.compute_cost(y_true, y_pred)
print(f"Custom Cost: {cost:.4f}")
```

Slide 7: Real-world Application: Stock Price Prediction

This practical implementation demonstrates cost function usage in a real-world stock price prediction model, incorporating both traditional MSE and custom financial metrics for model evaluation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def financial_cost_function(y_true, y_pred, direction_weight=0.3):
    """
    Custom cost function for financial predictions combining MSE and direction accuracy
    """
    # Calculate MSE component
    mse = np.mean(np.square(y_true - y_pred))
    
    # Calculate direction prediction accuracy
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    direction_accuracy = np.mean(true_direction == pred_direction)
    
    # Combine metrics
    total_cost = (1 - direction_weight) * mse - direction_weight * direction_accuracy
    return total_cost, mse, direction_accuracy

# Generate sample stock data
np.random.seed(42)
n_samples = 1000
dates = pd.date_range(start='2023-01-01', periods=n_samples)
stock_prices = np.cumsum(np.random.randn(n_samples)) + 100

# Prepare features and target
X = np.column_stack([np.arange(n_samples), 
                     np.sin(np.arange(n_samples)/10),
                     np.cos(np.arange(n_samples)/10)])
y = stock_prices

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train simple model and evaluate
weights = np.linalg.inv(X_train_scaled.T @ X_train_scaled) @ X_train_scaled.T @ y_train
y_pred = X_test_scaled @ weights

# Evaluate with custom financial cost function
cost, mse, dir_acc = financial_cost_function(y_test, y_pred)
print(f"Total Cost: {cost:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Direction Accuracy: {dir_acc:.4f}")
```

Slide 8: Dynamic Cost Function Adaptation

The concept of dynamic cost functions involves adjusting the loss calculation based on training progress or data characteristics. This implementation shows how to create an adaptive cost function that changes its behavior during training.

```python
class AdaptiveCostFunction:
    def __init__(self, initial_temp=1.0, decay_rate=0.995):
        self.temperature = initial_temp
        self.decay_rate = decay_rate
        self.iteration = 0
        
    def compute_cost(self, y_true, y_pred):
        """
        Compute cost with temperature-dependent behavior
        """
        # Update temperature
        self.temperature *= self.decay_rate
        self.iteration += 1
        
        # Compute base error
        squared_errors = np.square(y_true - y_pred)
        
        # Apply temperature-scaled focusing
        focused_errors = squared_errors * np.exp(squared_errors / self.temperature)
        
        return np.mean(focused_errors), self.temperature

# Example usage
adaptive_cost = AdaptiveCostFunction()
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.2, 1.9, 3.1, 3.8])

for epoch in range(5):
    cost, temp = adaptive_cost.compute_cost(y_true, y_pred)
    print(f"Epoch {epoch}: Cost = {cost:.4f}, Temperature = {temp:.4f}")
```

Slide 9: Huber Loss Implementation

Huber loss combines the best properties of MSE and MAE by being quadratic for small errors and linear for large errors. This implementation provides robustness against outliers while maintaining MSE's advantages for smaller errors.

```python
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Implement Huber loss function
    Args:
        y_true: actual values
        y_pred: predicted values
        delta: threshold for switching between MSE and MAE
    """
    errors = np.abs(y_true - y_pred)
    quadratic_mask = errors <= delta
    linear_mask = errors > delta
    
    quadratic_loss = 0.5 * np.square(errors[quadratic_mask])
    linear_loss = delta * errors[linear_mask] - 0.5 * (delta ** 2)
    
    return np.concatenate([quadratic_loss, linear_loss]).mean()

# Example with outliers
np.random.seed(42)
y_true = np.array([1, 2, 3, 4, 100])  # 100 is an outlier
y_pred = np.array([1.1, 2.1, 2.9, 4.2, 5.0])

mse_loss = np.mean(np.square(y_true - y_pred))
mae_loss = np.mean(np.abs(y_true - y_pred))
hub_loss = huber_loss(y_true, y_pred, delta=1.0)

print(f"MSE Loss: {mse_loss:.4f}")
print(f"MAE Loss: {mae_loss:.4f}")
print(f"Huber Loss: {hub_loss:.4f}")
```

Slide 10: Focal Loss for Imbalanced Classification

Focal Loss addresses class imbalance by down-weighting well-classified examples and focusing on hard, misclassified examples. This implementation is particularly useful for datasets with severe class imbalance.

```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Implement Focal Loss for binary classification
    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted probabilities
        gamma: focusing parameter
        alpha: class balance parameter
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross entropy
    cross_entropy = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    # Compute focal weights
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weights = alpha_t * np.power(1 - p_t, gamma)
    
    return np.mean(focal_weights * cross_entropy)

# Example with imbalanced dataset
n_samples = 1000
n_positive = 50  # Only 5% positive samples
y_true = np.zeros(n_samples)
y_true[:n_positive] = 1
np.random.shuffle(y_true)

# Simulated predictions
y_pred = np.random.beta(2, 5, n_samples)  # Biased predictions
focal = focal_loss(y_true, y_pred)
ce = cross_entropy_loss(y_true, y_pred)

print(f"Focal Loss: {focal:.4f}")
print(f"Cross-Entropy Loss: {ce:.4f}")
```

Slide 11: Distance-Based Cost Functions

Distance-based cost functions measure the similarity or dissimilarity between predictions and ground truth using various distance metrics. This implementation showcases different distance measures for specialized applications.

```python
class DistanceBasedCost:
    def __init__(self):
        self.metrics = {
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'cosine': self._cosine_distance
        }
    
    def _euclidean_distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y), axis=1))
    
    def _manhattan_distance(self, x, y):
        return np.sum(np.abs(x - y), axis=1)
    
    def _cosine_distance(self, x, y):
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        dot_product = np.sum(x * y, axis=1)
        return 1 - dot_product / (x_norm * y_norm + 1e-15)
    
    def compute_cost(self, y_true, y_pred, metric='euclidean'):
        """
        Compute distance-based cost using specified metric
        """
        if metric not in self.metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        
        distance = self.metrics[metric](y_true, y_pred)
        return np.mean(distance)

# Example usage
dist_cost = DistanceBasedCost()
y_true = np.random.randn(100, 3)  # 100 samples, 3 features
y_pred = np.random.randn(100, 3)

for metric in ['euclidean', 'manhattan', 'cosine']:
    cost = dist_cost.compute_cost(y_true, y_pred, metric=metric)
    print(f"{metric.capitalize()} Cost: {cost:.4f}")
```

Slide 12: Real-world Application: Image Reconstruction Loss

Image reconstruction tasks require specialized cost functions that capture both pixel-wise differences and structural similarity. This implementation demonstrates a comprehensive image reconstruction loss combining multiple components.

```python
import numpy as np
from scipy.ndimage import gaussian_filter

class ImageReconstructionLoss:
    def __init__(self, alpha=0.84, kernel_size=11, sigma=1.5):
        """
        Initialize image reconstruction loss
        Args:
            alpha: weight for SSIM component
            kernel_size: size of Gaussian kernel for SSIM
            sigma: standard deviation for Gaussian kernel
        """
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def _ssim(self, img1, img2):
        """Structural Similarity Index"""
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        mu1 = gaussian_filter(img1, self.sigma)
        mu2 = gaussian_filter(img2, self.sigma)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = gaussian_filter(img1 ** 2, self.sigma) - mu1_sq
        sigma2_sq = gaussian_filter(img2 ** 2, self.sigma) - mu2_sq
        sigma12 = gaussian_filter(img1 * img2, self.sigma) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return np.mean(ssim_map)
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute combined reconstruction loss
        """
        # L1 loss component
        mae = np.mean(np.abs(y_true - y_pred))
        
        # SSIM loss component
        ssim_loss = 1 - self._ssim(y_true, y_pred)
        
        # Combined loss
        total_loss = self.alpha * ssim_loss + (1 - self.alpha) * mae
        return total_loss, mae, ssim_loss

# Example usage
img_size = 64
y_true = np.random.rand(img_size, img_size)  # Original image
y_pred = y_true + 0.1 * np.random.randn(img_size, img_size)  # Noisy reconstruction

loss_fn = ImageReconstructionLoss()
total_loss, mae, ssim_loss = loss_fn.compute_loss(y_true, y_pred)

print(f"Total Loss: {total_loss:.4f}")
print(f"MAE: {mae:.4f}")
print(f"SSIM Loss: {ssim_loss:.4f}")
```

Slide 13: Perceptual Loss Implementation

Perceptual loss leverages pre-trained neural network features to compute differences in higher-level representations rather than pixel-space differences, providing more semantically meaningful error metrics for various generation tasks.

```python
import numpy as np

class PerceptualLoss:
    def __init__(self, feature_weights=None):
        """
        Initialize perceptual loss calculator
        Args:
            feature_weights: weights for different feature levels
        """
        self.feature_weights = feature_weights or {
            'layer1': 1.0,
            'layer2': 0.75,
            'layer3': 0.5,
            'layer4': 0.25
        }
    
    def _extract_features(self, x, layer):
        """
        Simulate feature extraction from different network layers
        In practice, this would use a real pre-trained network
        """
        # Simplified feature extraction simulation
        if layer == 'layer1':
            return np.mean(x.reshape(x.shape[0], -1, 16), axis=2)
        elif layer == 'layer2':
            return np.mean(x.reshape(x.shape[0], -1, 8), axis=2)
        elif layer == 'layer3':
            return np.mean(x.reshape(x.shape[0], -1, 4), axis=2)
        else:  # layer4
            return np.mean(x.reshape(x.shape[0], -1, 2), axis=2)
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute weighted perceptual loss across feature layers
        """
        total_loss = 0
        layer_losses = {}
        
        for layer, weight in self.feature_weights.items():
            true_features = self._extract_features(y_true, layer)
            pred_features = self._extract_features(y_pred, layer)
            
            layer_loss = np.mean(np.square(true_features - pred_features))
            weighted_loss = weight * layer_loss
            
            layer_losses[layer] = layer_loss
            total_loss += weighted_loss
            
        return total_loss, layer_losses

# Example usage
batch_size, height, width = 4, 32, 32
y_true = np.random.rand(batch_size, height, width)
y_pred = y_true + 0.1 * np.random.randn(batch_size, height, width)

perceptual_loss = PerceptualLoss()
total_loss, layer_losses = perceptual_loss.compute_loss(y_true, y_pred)

print(f"Total Perceptual Loss: {total_loss:.4f}")
for layer, loss in layer_losses.items():
    print(f"{layer} Loss: {loss:.4f}")
```

Slide 14: Additional Resources

*   "Deep Learning Book - Chapter 8: Optimization for Training Deep Models" - [https://www.deeplearningbook.org/contents/optimization.html](https://www.deeplearningbook.org/contents/optimization.html)
*   "On Loss Functions for Deep Neural Networks in Classification" - [https://arxiv.org/abs/2011.05827](https://arxiv.org/abs/2011.05827)
*   "Focal Loss for Dense Object Detection" - [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)
*   "Understanding Deep Learning Requires Rethinking Generalization" - [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)
*   "Learning Loss Functions for Semi-supervised Learning via Discriminative Adversarial Networks" - [https://arxiv.org/abs/1707.02198](https://arxiv.org/abs/1707.02198)
*   Search keywords for further research:
    *   "Novel loss functions deep learning"
    *   "Adaptive cost functions machine learning"
    *   "Custom loss functions for specific domains"

