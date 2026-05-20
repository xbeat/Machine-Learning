## Stochastic Gradient Descent Outperforms Batch Gradient Descent
Slide 1: Introduction to Gradient Descent Optimization

Gradient descent optimization serves as the backbone of machine learning model training, where the algorithm iteratively adjusts parameters to minimize the loss function. The mathematical foundation relies on computing partial derivatives to determine the steepest descent direction.

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    # Initialize parameters
    m, n = X.shape
    theta = np.zeros(n)
    
    # Gradient descent implementation
    for _ in range(epochs):
        # Compute predictions
        h = np.dot(X, theta)
        # Compute gradients
        gradient = (1/m) * np.dot(X.T, (h - y))
        # Update parameters
        theta -= learning_rate * gradient
        
    return theta

# Example usage
X = np.random.randn(100, 3)
y = 2*X[:,0] + 3*X[:,1] - X[:,2] + np.random.randn(100)*0.1
theta = gradient_descent(X, y)
print(f"Optimized parameters: {theta}")
```

Slide 2: Mathematical Foundations of SGD vs BGD

Stochastic Gradient Descent differs fundamentally from Batch Gradient Descent in its update rule and convergence properties. The mathematical distinction lies in the computation of the gradient estimate using individual samples versus the entire dataset.

```python
# Mathematical representation of BGD vs SGD update rules
"""
BGD Update Rule:
$$θ_{t+1} = θ_t - η \frac{1}{N} \sum_{i=1}^N \nabla L(x_i, y_i, θ_t)$$

SGD Update Rule:
$$θ_{t+1} = θ_t - η \nabla L(x_i, y_i, θ_t)$$

where:
η: learning rate
θ: model parameters
L: loss function
N: total number of samples
"""
```

Slide 3: Implementing Basic SGD Optimizer

Stochastic Gradient Descent implementation requires careful handling of individual samples and proper shuffling mechanisms. This implementation demonstrates the core concepts of SGD including sample-wise updates and random sample selection.

```python
import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Single sample updates
            for i in range(m):
                gradient = X_shuffled[i] * (np.dot(X_shuffled[i], theta) - y_shuffled[i])
                theta -= self.learning_rate * gradient
                
        return theta

# Example usage
X = np.random.randn(1000, 5)
y = np.sum(X * np.array([1, 2, 3, -1, 0.5]), axis=1)
optimizer = SGDOptimizer()
theta = optimizer.optimize(X, y)
print(f"Final parameters: {theta}")
```

Slide 4: Batch Gradient Descent Implementation

This implementation showcases the traditional batch gradient descent approach, processing the entire dataset in each iteration. The code demonstrates the key difference in update mechanism compared to SGD.

```python
class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        
    def compute_loss(self, X, y, theta):
        predictions = np.dot(X, theta)
        return np.mean((predictions - y) ** 2)
    
    def optimize(self, X, y, max_iterations=1000):
        m, n = X.shape
        theta = np.zeros(n)
        prev_loss = float('inf')
        
        for iteration in range(max_iterations):
            # Compute gradients using entire dataset
            predictions = np.dot(X, theta)
            gradients = (2/m) * np.dot(X.T, (predictions - y))
            
            # Update parameters
            theta -= self.learning_rate * gradients
            
            # Check convergence
            current_loss = self.compute_loss(X, y, theta)
            if abs(prev_loss - current_loss) < self.tolerance:
                break
                
            prev_loss = current_loss
            
        return theta, current_loss

# Demonstration
X = np.random.randn(1000, 3)
true_theta = np.array([2, -1, 0.5])
y = np.dot(X, true_theta) + np.random.randn(1000) * 0.1

bgd = BatchGradientDescent()
estimated_theta, final_loss = bgd.optimize(X, y)
print(f"True parameters: {true_theta}")
print(f"Estimated parameters: {estimated_theta}")
print(f"Final loss: {final_loss}")
```

Slide 5: Mini-batch Gradient Descent Implementation

Mini-batch gradient descent combines the benefits of both SGD and BGD by processing small batches of data. This approach provides a balance between computation efficiency and convergence stability, making it particularly effective for large datasets.

```python
class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def create_mini_batches(self, X, y):
        m = X.shape[0]
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        mini_batches = []
        num_batches = m // self.batch_size
        
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            mini_batches.append((
                X_shuffled[start_idx:end_idx],
                y_shuffled[start_idx:end_idx]
            ))
            
        return mini_batches
    
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(X, y)
            
            for X_batch, y_batch in mini_batches:
                gradients = np.dot(X_batch.T, 
                                 (np.dot(X_batch, theta) - y_batch))
                theta -= self.learning_rate * gradients / len(X_batch)
                
        return theta

# Example usage
X = np.random.randn(10000, 4)
true_theta = np.array([1, -0.5, 0.25, 2])
y = np.dot(X, true_theta) + np.random.randn(10000) * 0.1

mbgd = MiniBatchGradientDescent()
estimated_theta = mbgd.optimize(X, y)
print(f"True parameters: {true_theta}")
print(f"Estimated parameters: {estimated_theta}")
```

Slide 6: Momentum-based SGD Implementation

Momentum helps accelerate SGD in the relevant direction and dampens oscillations. This implementation demonstrates how momentum can be incorporated into the SGD optimizer to improve convergence speed and stability.

```python
class MomentumSGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        velocity = np.zeros(n)
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                gradient = X_shuffled[i] * (
                    np.dot(X_shuffled[i], theta) - y_shuffled[i]
                )
                
                # Update velocity and parameters using momentum
                velocity = self.momentum * velocity - self.learning_rate * gradient
                theta += velocity
                
        return theta

# Example with momentum
X = np.random.randn(5000, 3)
true_theta = np.array([1.5, -0.8, 0.6])
y = np.dot(X, true_theta) + np.random.randn(5000) * 0.1

momentum_sgd = MomentumSGD()
estimated_theta = momentum_sgd.optimize(X, y)
print(f"True parameters: {true_theta}")
print(f"Estimated parameters: {estimated_theta}")
```

Slide 7: Learning Rate Scheduling

Learning rate scheduling is crucial for optimizing convergence. This implementation shows how to adapt the learning rate during training using various scheduling strategies to improve model performance.

```python
class AdaptiveSGD:
    def __init__(self, initial_lr=0.1, decay_rate=0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        
    def get_learning_rate(self, epoch):
        # Implement different scheduling strategies
        
        # Step decay
        step_lr = self.initial_lr * (self.decay_rate ** epoch)
        
        # Exponential decay
        exp_lr = self.initial_lr * np.exp(-self.decay_rate * epoch)
        
        # 1/t decay
        inv_lr = self.initial_lr / (1 + self.decay_rate * epoch)
        
        return step_lr  # Choose one strategy
    
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        
        for epoch in range(epochs):
            current_lr = self.get_learning_rate(epoch)
            indices = np.random.permutation(m)
            
            for i in indices:
                gradient = X[i] * (np.dot(X[i], theta) - y[i])
                theta -= current_lr * gradient
                
        return theta

# Example usage
X = np.random.randn(1000, 4)
true_theta = np.array([1, -1, 0.5, 2])
y = np.dot(X, true_theta) + np.random.randn(1000) * 0.1

adaptive_sgd = AdaptiveSGD()
estimated_theta = adaptive_sgd.optimize(X, y)
print(f"Results with learning rate scheduling:")
print(f"True parameters: {true_theta}")
print(f"Estimated parameters: {estimated_theta}")
```

Slide 8: Implementing Gradient Descent with Early Stopping

Early stopping is a crucial regularization technique that prevents overfitting by monitoring validation performance. This implementation demonstrates how to incorporate early stopping into the gradient descent optimization process.

```python
class GradientDescentWithEarlyStopping:
    def __init__(self, learning_rate=0.01, patience=5):
        self.learning_rate = learning_rate
        self.patience = patience
        
    def compute_loss(self, X, y, theta):
        predictions = np.dot(X, theta)
        return np.mean((predictions - y) ** 2)
    
    def optimize(self, X_train, y_train, X_val, y_val):
        m, n = X_train.shape
        theta = np.zeros(n)
        best_val_loss = float('inf')
        patience_counter = 0
        best_theta = None
        
        while patience_counter < self.patience:
            # Training step
            indices = np.random.permutation(m)
            for i in indices:
                gradient = X_train[i] * (
                    np.dot(X_train[i], theta) - y_train[i]
                )
                theta -= self.learning_rate * gradient
            
            # Validation step
            val_loss = self.compute_loss(X_val, y_val, theta)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_theta = theta.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
        return best_theta, best_val_loss

# Example usage
from sklearn.model_selection import train_test_split

X = np.random.randn(2000, 3)
true_theta = np.array([1, -0.5, 2])
y = np.dot(X, true_theta) + np.random.randn(2000) * 0.1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

gd_early = GradientDescentWithEarlyStopping()
best_theta, final_val_loss = gd_early.optimize(X_train, y_train, X_val, y_val)
print(f"Best parameters: {best_theta}")
print(f"Final validation loss: {final_val_loss}")
```

Slide 9: Implementing Adaptive Learning Rates with AdaGrad

AdaGrad adapts the learning rate for each parameter based on historical gradients. This implementation shows how to incorporate parameter-specific learning rates for improved optimization.

```python
class AdaGradOptimizer:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        accumulated_gradients = np.zeros(n)
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                gradient = X_shuffled[i] * (
                    np.dot(X_shuffled[i], theta) - y_shuffled[i]
                )
                
                # Update accumulated gradients
                accumulated_gradients += gradient ** 2
                
                # Compute adaptive learning rates
                adaptive_lr = self.learning_rate / (
                    np.sqrt(accumulated_gradients + self.epsilon)
                )
                
                # Update parameters
                theta -= adaptive_lr * gradient
                
        return theta

# Example usage
X = np.random.randn(1500, 4)
true_theta = np.array([1.5, -0.8, 0.6, 2.0])
y = np.dot(X, true_theta) + np.random.randn(1500) * 0.1

adagrad = AdaGradOptimizer()
estimated_theta = adagrad.optimize(X, y)
print(f"True parameters: {true_theta}")
print(f"AdaGrad estimated parameters: {estimated_theta}")
```

Slide 10: Real-world Application - Linear Regression with SGD

This implementation demonstrates a complete real-world example of using SGD for linear regression, including data preprocessing, model training, and performance evaluation.

```python
class SGDLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def standardize_data(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)
    
    def fit(self, X, y):
        # Standardize features
        X_std = self.standardize_data(X)
        m, n = X_std.shape
        self.theta = np.zeros(n)
        
        # Training history
        self.loss_history = []
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            epoch_loss = 0
            
            for i in indices:
                prediction = np.dot(X_std[i], self.theta)
                loss = prediction - y[i]
                gradient = X_std[i] * loss
                self.theta -= self.learning_rate * gradient
                epoch_loss += loss ** 2
                
            self.loss_history.append(epoch_loss / m)
            
    def predict(self, X):
        X_std = (X - self.mean) / (self.std + 1e-8)
        return np.dot(X_std, self.theta)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)

# Real-world example with California housing dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load and prepare data
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)

# Train model
model = SGDLinearRegression(learning_rate=0.001, epochs=50)
model.fit(X_train, y_train)

# Evaluate performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score: {test_score:.4f}")
```

Slide 11: Implementing Nesterov Accelerated Gradient (NAG)

Nesterov Accelerated Gradient enhances momentum-based optimization by looking ahead in the direction of the accumulated gradient. This implementation shows how to incorporate NAG into the SGD framework for faster convergence.

```python
class NesterovSGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        velocity = np.zeros(n)
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(m):
                # Look ahead position
                theta_ahead = theta + self.momentum * velocity
                
                # Compute gradient at the look-ahead position
                gradient = X_shuffled[i] * (
                    np.dot(X_shuffled[i], theta_ahead) - y_shuffled[i]
                )
                
                # Update velocity and parameters
                velocity = self.momentum * velocity - self.learning_rate * gradient
                theta += velocity
                
        return theta, velocity

# Performance comparison example
np.random.seed(42)
X = np.random.randn(2000, 5)
true_theta = np.array([1.0, -0.5, 0.25, 2.0, -1.5])
y = np.dot(X, true_theta) + np.random.randn(2000) * 0.1

nag = NesterovSGD()
final_theta, final_velocity = nag.optimize(X, y)

mse = np.mean((np.dot(X, final_theta) - y) ** 2)
print(f"True parameters: {true_theta}")
print(f"NAG estimated parameters: {final_theta}")
print(f"Final MSE: {mse:.6f}")
```

Slide 12: RMSprop Optimization Implementation

RMSprop addresses the diminishing learning rates in AdaGrad by using an exponentially decaying average of squared gradients. This implementation demonstrates how to effectively use RMSprop for gradient descent optimization.

```python
class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        
    def optimize(self, X, y, epochs=100):
        m, n = X.shape
        theta = np.zeros(n)
        cache = np.zeros(n)
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            epoch_loss = 0
            
            for i in indices:
                # Compute gradient
                gradient = X[i] * (np.dot(X[i], theta) - y[i])
                
                # Update cache
                cache = self.decay_rate * cache + (1 - self.decay_rate) * gradient**2
                
                # Update parameters
                theta -= (self.learning_rate / np.sqrt(cache + self.epsilon)) * gradient
                
                # Calculate loss
                prediction = np.dot(X[i], theta)
                epoch_loss += (prediction - y[i])**2
                
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {epoch_loss/m:.6f}")
                
        return theta

# Example usage with comparison
X = np.random.randn(1500, 3)
true_theta = np.array([1.0, 2.0, -0.5])
y = np.dot(X, true_theta) + np.random.randn(1500) * 0.1

rmsprop = RMSpropOptimizer()
estimated_theta = rmsprop.optimize(X, y)

# Compute final MSE
final_predictions = np.dot(X, estimated_theta)
mse = np.mean((y - final_predictions)**2)

print("\nResults:")
print(f"True parameters: {true_theta}")
print(f"RMSprop estimated parameters: {estimated_theta}")
print(f"Final MSE: {mse:.6f}")
```

Slide 13: Convergence Analysis Implementation

This implementation provides tools for analyzing the convergence behavior of different gradient descent variants, helping understand their comparative performance characteristics.

```python
class ConvergenceAnalyzer:
    def __init__(self, optimizers_dict):
        self.optimizers = optimizers_dict
        self.convergence_history = {}
        
    def analyze_convergence(self, X, y, epochs=100):
        for name, optimizer in self.optimizers.items():
            m, n = X.shape
            theta = np.zeros(n)
            loss_history = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                predictions = np.dot(X, theta)
                loss = np.mean((predictions - y)**2)
                loss_history.append(loss)
                
                # Update parameters using optimizer
                theta = optimizer.optimize(X, y, epochs=1)
                
            self.convergence_history[name] = loss_history
            
        return self.convergence_history

# Example usage
import matplotlib.pyplot as plt

# Create synthetic dataset
X = np.random.randn(1000, 4)
true_theta = np.array([1.0, -0.5, 2.0, -1.5])
y = np.dot(X, true_theta) + np.random.randn(1000) * 0.1

# Initialize optimizers
optimizers = {
    'SGD': SGDOptimizer(learning_rate=0.01),
    'Momentum': MomentumSGD(learning_rate=0.01),
    'RMSprop': RMSpropOptimizer(learning_rate=0.001),
    'NAG': NesterovSGD(learning_rate=0.01)
}

# Analyze convergence
analyzer = ConvergenceAnalyzer(optimizers)
convergence_results = analyzer.analyze_convergence(X, y)

# Plot results
plt.figure(figsize=(10, 6))
for name, history in convergence_results.items():
    plt.plot(history, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Convergence Comparison of Different Optimizers')
plt.grid(True)
```

Slide 14: Additional Resources

*   "On the importance of initialization and momentum in deep learning" ([https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701))
*   "Adam: A Method for Stochastic Optimization" ([https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980))
*   "An Overview of Gradient Descent Optimization Algorithms" ([https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747))
*   "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" ([https://arxiv.org/abs/1011.1669](https://arxiv.org/abs/1011.1669))
*   "ADADELTA: An Adaptive Learning Rate Method" ([https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701))

