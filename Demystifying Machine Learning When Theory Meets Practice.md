## Demystifying Machine Learning When Theory Meets Practice
Slide 1: Understanding Vectors in Machine Learning

Linear algebra's foundation in ML starts with vectors. Vectors represent features, parameters, or data points in n-dimensional space, forming the basic building blocks for more complex operations like matrix multiplication and tensor operations used in neural networks.

```python
import numpy as np

# Creating feature vectors
feature_vector = np.array([1.2, 3.4, 2.1, 0.8])

# Basic vector operations
magnitude = np.linalg.norm(feature_vector)
normalized = feature_vector / magnitude

# Example: Cosine similarity between two vectors
vector2 = np.array([0.9, 3.1, 2.4, 1.0])
cos_sim = np.dot(feature_vector, vector2) / (np.linalg.norm(feature_vector) * np.linalg.norm(vector2))

print(f"Original vector: {feature_vector}")
print(f"Magnitude: {magnitude:.2f}")
print(f"Normalized vector: {normalized}")
print(f"Cosine similarity: {cos_sim:.4f}")
```

Slide 2: Matrix Operations Fundamentals

Matrices are essential for representing datasets, where each row typically represents a sample and each column represents a feature. Understanding matrix operations is crucial for implementing linear transformations and neural network layers.

```python
import numpy as np

# Create a sample dataset matrix (3 samples, 4 features)
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

# Weight matrix for transformation (4 features to 2 outputs)
W = np.array([[0.1, 0.2],
              [0.3, 0.4],
              [0.5, 0.6],
              [0.7, 0.8]])

# Matrix multiplication (linear transformation)
output = np.dot(X, W)

print("Input shape:", X.shape)
print("Weight shape:", W.shape)
print("Output shape:", output.shape)
print("\nTransformed output:\n", output)
```

Slide 3: Probability Fundamentals in ML

Probability theory underlies many ML concepts, from loss functions to probabilistic models. Understanding probability distributions and maximum likelihood estimation helps in designing better models and interpreting their outputs.

```python
import numpy as np
from scipy import stats

# Generate synthetic data from normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Maximum Likelihood Estimation
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=1)

# Calculate log-likelihood
log_likelihood = np.sum(stats.norm.logpdf(data, mu_mle, sigma_mle))

print(f"MLE Mean: {mu_mle:.4f}")
print(f"MLE Standard Deviation: {sigma_mle:.4f}")
print(f"Log-likelihood: {log_likelihood:.4f}")
```

Slide 4: Gradient Descent Implementation

Gradient descent is the cornerstone of modern ML optimization. This implementation demonstrates how calculus principles translate into practical optimization algorithms for finding the minimum of a function.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(start, gradient, learning_rate, n_iterations):
    path = [start]
    position = start
    
    for _ in range(n_iterations):
        gradient_val = gradient(position)
        position = position - learning_rate * gradient_val
        path.append(position)
    
    return np.array(path)

# Example: Finding minimum of f(x) = x^2 + 2
gradient = lambda x: 2*x  # derivative of x^2
start = 2.0
learning_rate = 0.1
n_iterations = 20

path = gradient_descent(start, gradient, learning_rate, n_iterations)

print("Optimization path:")
for i, pos in enumerate(path):
    print(f"Iteration {i}: x = {pos:.4f}, f(x) = {pos**2 + 2:.4f}")
```

Slide 5: Principal Component Analysis from Scratch

Principal Component Analysis (PCA) demonstrates the practical application of eigendecomposition in dimensionality reduction. This implementation shows how linear algebra concepts translate into data transformation techniques.

```python
import numpy as np

def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_transformed = np.dot(X_centered, components)
    
    return X_transformed, components, eigenvalues

# Example usage
np.random.seed(42)
X = np.random.randn(100, 5)
X_transformed, components, eigenvalues = pca_from_scratch(X, n_components=2)

print("Original data shape:", X.shape)
print("Transformed data shape:", X_transformed.shape)
print("Explained variance ratio:", eigenvalues[:2] / sum(eigenvalues))
```

Slide 6: Implementing Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) forms the theoretical foundation for many ML loss functions. This implementation demonstrates how probability theory connects to practical model optimization through the likelihood function.

```python
import numpy as np
from scipy.optimize import minimize

def negative_log_likelihood(params, data):
    mu, sigma = params
    return -np.sum(stats.norm.logpdf(data, mu, sigma))

# Generate synthetic data
np.random.seed(42)
true_mu, true_sigma = 2.0, 1.5
data = np.random.normal(true_mu, true_sigma, 1000)

# Find MLE parameters
initial_guess = [0, 1]
result = minimize(negative_log_likelihood, initial_guess, args=(data,), method='Nelder-Mead')

print(f"True parameters: mu={true_mu}, sigma={true_sigma}")
print(f"MLE estimates: mu={result.x[0]:.4f}, sigma={result.x[1]:.4f}")
print(f"Negative log-likelihood: {result.fun:.4f}")
```

Slide 7: Implementing Linear Regression with Gradient Descent

Understanding the mathematics behind linear regression helps in grasping more complex ML models. This implementation shows how calculus and linear algebra combine in a practical optimization scenario.

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.random.randn(100, 3)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)*0.1

model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print("True weights: [3, 2, -1]")
print(f"Estimated weights: {model.weights}")
print(f"Estimated bias: {model.bias:.4f}")
```

Slide 8: Advanced Matrix Operations for Deep Learning

Understanding matrix calculus is crucial for implementing backpropagation in neural networks. This implementation demonstrates key matrix operations used in deep learning frameworks.

```python
import numpy as np

def relu(X):
    return np.maximum(0, X)

def relu_derivative(X):
    return (X > 0).astype(float)

def forward_pass(X, W1, b1, W2, b2):
    # First layer
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    # Second layer
    Z2 = np.dot(A1, W2) + b2
    A2 = Z2  # Linear activation for regression
    
    cache = (Z1, A1, Z2, A2)
    return cache

def backward_pass(X, y, cache, W1, W2):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache
    
    # Output layer gradients
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0)
    
    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0)
    
    return dW1, db1, dW2, db2

# Example initialization
X = np.random.randn(100, 10)
y = np.random.randn(100, 1)
W1 = np.random.randn(10, 5) * 0.01
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 1) * 0.01
b2 = np.zeros((1, 1))

# One step of forward and backward propagation
cache = forward_pass(X, W1, b1, W2, b2)
gradients = backward_pass(X, y, cache, W1, W2)

print("Shape of gradients:")
for i, grad in enumerate(gradients):
    print(f"Gradient {i+1} shape: {grad.shape}")
```

Slide 9: Real-World Example - Credit Card Fraud Detection

This implementation demonstrates how mathematical concepts translate into a practical machine learning solution for detecting fraudulent transactions using probabilistic approaches.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import pandas as pd

class FraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mean_normal = None
        self.cov_normal = None
        
    def fit(self, X, y):
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate parameters for normal transactions
        normal_data = X_scaled[y == 0]
        self.mean_normal = np.mean(normal_data, axis=0)
        self.cov_normal = np.cov(normal_data.T)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        
        # Calculate Mahalanobis distance
        diff = X_scaled - self.mean_normal
        inv_covmat = np.linalg.inv(self.cov_normal)
        left_term = np.dot(diff, inv_covmat)
        mahalanobis = np.sqrt(np.sum(left_term * diff, axis=1))
        
        # Convert to probability using softmax
        return 1 / (1 + np.exp(-mahalanobis + np.median(mahalanobis)))

# Example usage with synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Generate synthetic data
X_normal = np.random.normal(0, 1, (950, n_features))
X_fraud = np.random.normal(2, 2, (50, n_features))
X = np.vstack([X_normal, X_fraud])
y = np.hstack([np.zeros(950), np.ones(50)])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
detector = FraudDetector()
detector.fit(X_train, y_train)
y_pred_proba = detector.predict_proba(X_test)

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

print("Model Performance Metrics:")
print(f"Number of thresholds: {len(thresholds)}")
print(f"Max precision: {np.max(precision):.4f}")
print(f"Max recall: {np.max(recall):.4f}")
```

Slide 10: Results for Credit Card Fraud Detection

This slide presents the detailed analysis and visualization of the fraud detection model's performance, demonstrating how mathematical concepts translate into measurable outcomes.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_model_performance(y_true, y_pred_proba):
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    
    # ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Distribution of probabilities
    plt.subplot(1, 2, 2)
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, label='Fraud', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    
    plt.tight_layout()
    return plt

# Calculate and display metrics
precision = len(y_test[y_pred_proba > 0.5]) / len(y_test)
recall = sum(y_test[y_pred_proba > 0.5]) / sum(y_test)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Visualize results
plt = plot_model_performance(y_test, y_pred_proba)
plt.show()
```

Slide 11: Implementing Bayesian Parameter Estimation

Bayesian methods provide a robust framework for uncertainty estimation in ML models. This implementation shows how probability theory connects with parameter estimation.

```python
import numpy as np
from scipy import stats

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.mean = None    # Posterior mean
        self.precision = None  # Posterior precision
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Calculate posterior precision
        self.precision = self.alpha * np.eye(n_features) + self.beta * X.T @ X
        
        # Calculate posterior mean
        self.mean = self.beta * np.linalg.solve(self.precision, X.T @ y)
        
    def predict(self, X, return_std=False):
        y_mean = X @ self.mean
        
        if return_std:
            # Calculate predictive variance
            y_var = 1/self.beta + np.sum(X @ np.linalg.solve(self.precision, X.T), axis=0)
            return y_mean, np.sqrt(y_var)
        return y_mean

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 2)
true_weights = np.array([1.5, -0.8])
y = X @ true_weights + np.random.normal(0, 0.1, size=100)

# Fit model and make predictions
model = BayesianLinearRegression(alpha=0.1, beta=10.0)
model.fit(X, y)

# Make predictions with uncertainty
X_test = np.random.randn(10, 2)
y_pred, y_std = model.predict(X_test, return_std=True)

print("True weights:", true_weights)
print("Estimated weights (posterior mean):", model.mean)
print("\nPredictions with uncertainty:")
for i in range(len(y_pred)):
    print(f"Prediction {i+1}: {y_pred[i]:.3f} ± {2*y_std[i]:.3f}")
```

Slide 12: Implementing Information Theory Concepts

Information theory provides the mathematical foundation for many ML concepts, including cross-entropy loss and KL divergence. This implementation demonstrates practical applications.

```python
import numpy as np

class InformationTheoryMetrics:
    @staticmethod
    def entropy(p):
        """Calculate Shannon entropy of a probability distribution."""
        p = np.array(p)
        p = p[p > 0]  # Remove zero probabilities
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def kl_divergence(p, q):
        """Calculate Kullback-Leibler divergence between two distributions."""
        p = np.array(p)
        q = np.array(q)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        q = q + epsilon
        return np.sum(p * np.log2(p / q))
    
    @staticmethod
    def cross_entropy(p, q):
        """Calculate cross-entropy between true distribution p and predicted q."""
        return -np.sum(p * np.log2(q + 1e-10))
    
    @staticmethod
    def mutual_information(joint_prob):
        """Calculate mutual information from joint probability distribution."""
        p_x = np.sum(joint_prob, axis=1)
        p_y = np.sum(joint_prob, axis=0)
        
        h_x = InformationTheoryMetrics.entropy(p_x)
        h_y = InformationTheoryMetrics.entropy(p_y)
        h_xy = InformationTheoryMetrics.entropy(joint_prob.flatten())
        
        return h_x + h_y - h_xy

# Example usage
# True and predicted probability distributions
p = np.array([0.3, 0.4, 0.3])
q = np.array([0.25, 0.45, 0.3])

# Joint probability distribution
joint_prob = np.array([[0.2, 0.1],
                      [0.1, 0.6]])

metrics = InformationTheoryMetrics()

print(f"Entropy of p: {metrics.entropy(p):.4f}")
print(f"KL divergence (P||Q): {metrics.kl_divergence(p, q):.4f}")
print(f"Cross-entropy: {metrics.cross_entropy(p, q):.4f}")
print(f"Mutual Information: {metrics.mutual_information(joint_prob):.4f}")
```

Slide 13: Advanced Optimization Techniques

This implementation showcases advanced optimization methods commonly used in machine learning, demonstrating the practical application of calculus concepts in optimization algorithms.

```python
import numpy as np
from typing import Callable, Tuple

class AdvancedOptimizer:
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def adam(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
            
        self.t += 1
        
        # Update biased first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

def optimize_function(func: Callable, 
                     grad_func: Callable, 
                     initial_params: np.ndarray,
                     n_iterations: int = 1000) -> Tuple[np.ndarray, list]:
    
    optimizer = AdvancedOptimizer()
    params = initial_params.copy()
    loss_history = []
    
    for _ in range(n_iterations):
        # Calculate gradients
        grads = grad_func(params)
        
        # Update parameters using Adam
        params = optimizer.adam(params, grads)
        
        # Record loss
        loss = func(params)
        loss_history.append(loss)
    
    return params, loss_history

# Example: Optimize Rosenbrock function
def rosenbrock(x: np.ndarray) -> float:
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dx1 = 200*(x[1] - x[0]**2)
    return np.array([dx0, dx1])

# Optimize
initial_guess = np.array([-1.0, 1.0])
optimal_params, loss_history = optimize_function(
    rosenbrock, 
    rosenbrock_gradient, 
    initial_guess
)

print(f"Initial parameters: {initial_guess}")
print(f"Optimal parameters: {optimal_params}")
print(f"Final loss: {rosenbrock(optimal_params):.8f}")
```

Slide 14: Real-World Example - Time Series Forecasting

This implementation demonstrates how mathematical concepts in linear algebra and probability theory apply to time series analysis and forecasting.

```python
import numpy as np
from scipy import stats
from typing import Tuple, List

class TimeSeriesForecaster:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.weights = None
        self.bias = None
        
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
    
    def fit(self, data: np.ndarray, learning_rate: float = 0.01, 
            epochs: int = 100) -> List[float]:
        """Fit the model using gradient descent"""
        X, y = self.create_sequences(data)
        self.weights = np.random.randn(self.window_size) * 0.01
        self.bias = 0.0
        loss_history = []
        
        for _ in range(epochs):
            # Forward pass
            predictions = np.dot(X, self.weights) + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            loss_history.append(loss)
            
            # Compute gradients
            dw = 2 * np.dot(X.T, (predictions - y)) / len(y)
            db = 2 * np.mean(predictions - y)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
        return loss_history
    
    def predict(self, sequence: np.ndarray, n_steps: int) -> np.ndarray:
        """Make multi-step predictions"""
        predictions = []
        curr_sequence = sequence[-self.window_size:].copy()
        
        for _ in range(n_steps):
            # Make single prediction
            next_pred = np.dot(curr_sequence, self.weights) + self.bias
            predictions.append(next_pred)
            
            # Update sequence
            curr_sequence = np.roll(curr_sequence, -1)
            curr_sequence[-1] = next_pred
            
        return np.array(predictions)

# Generate synthetic time series data
np.random.seed(42)
t = np.linspace(0, 4*np.pi, 200)
y = np.sin(t) + np.random.normal(0, 0.1, len(t))

# Train model
forecaster = TimeSeriesForecaster(window_size=20)
loss_history = forecaster.fit(y, learning_rate=0.01, epochs=200)

# Make predictions
initial_sequence = y[-20:]
predictions = forecaster.predict(initial_sequence, n_steps=50)

print("Model Performance:")
print(f"Final training loss: {loss_history[-1]:.6f}")
print(f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
```

Slide 15: Additional Resources

*   Building Neural Networks from Scratch
    *   arXiv:2006.14901 \[cs.LG\]
    *   [https://arxiv.org/abs/2006.14901](https://arxiv.org/abs/2006.14901)
*   Mathematical Foundations of Machine Learning
    *   arXiv:1908.09492 \[cs.LG\]
    *   [https://arxiv.org/abs/1908.09492](https://arxiv.org/abs/1908.09492)
*   Advanced Optimization Methods in Deep Learning
    *   arXiv:1912.05671 \[cs.LG\]
    *   [https://arxiv.org/abs/1912.05671](https://arxiv.org/abs/1912.05671)
*   Probabilistic Machine Learning: Theory and Algorithms
    *   [https://probml.github.io/pml-book/](https://probml.github.io/pml-book/)
*   Information Theory in Machine Learning
    *   [https://www.microsoft.com/en-us/research/publication/information-theory-machine-learning/](https://www.microsoft.com/en-us/research/publication/information-theory-machine-learning/)

