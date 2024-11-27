## Exploring the Perceptron Architecture and Training
Slide 1: The Perceptron Architecture

The perceptron is a fundamental building block of neural networks, representing a binary classifier that maps input features to a binary output through a weighted sum followed by an activation function. It serves as the foundation for understanding more complex neural architectures.

```python
import numpy as np

class Perceptron:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        
    def activation(self, x):
        # Step activation function
        return 1 if x > 0 else -1
    
    def forward(self, x):
        # Calculate weighted sum and apply activation
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)
```

Slide 2: Mathematical Foundation

The perceptron's decision boundary is defined by a linear equation that separates the input space into two regions. The mathematical representation helps understand how the perceptron makes classification decisions.

```python
# Mathematical representation of perceptron's decision boundary
"""
$$z = \sum_{i=1}^n w_i x_i + b$$
$$y = \begin{cases} 
1 & \text{if } z > 0 \\
-1 & \text{if } z \leq 0
\end{cases}$$

Where:
- z is the weighted sum
- w_i are the weights
- x_i are the input features
- b is the bias term
- y is the output prediction
"""
```

Slide 3: Perceptron Training Algorithm

The perceptron learning algorithm iteratively updates weights and bias based on misclassified points. This implementation demonstrates the complete training process including error calculation and weight updates.

```python
def train(self, X, y, learning_rate=0.1, epochs=100):
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            prediction = self.forward(xi)
            if prediction != yi:
                # Update weights and bias
                self.weights += learning_rate * yi * xi
                self.bias += learning_rate * yi
                errors += 1
        if errors == 0:
            break
```

Slide 4: Real-world Example - Binary Classification

Here we implement a complete binary classification task using the perceptron on a real dataset, demonstrating data preprocessing, training, and evaluation phases for a practical problem.

```python
# Generate sample dataset
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Create and train perceptron
model = Perceptron(input_size=2)
model.train(X, y, learning_rate=0.01, epochs=100)

# Evaluate accuracy
def accuracy(model, X, y):
    predictions = [model.forward(xi) for xi in X]
    return np.mean(predictions == y)

print(f"Accuracy: {accuracy(model, X, y):.2f}")
```

Slide 5: Implementing the Perceptron Trick

The perceptron trick involves adjusting the decision boundary incrementally for each misclassified point. This implementation shows the geometric interpretation and practical application of the weight update rule.

```python
def perceptron_trick(self, x, y, learning_rate):
    """
    x: input feature vector
    y: true label (-1 or 1)
    learning_rate: step size for updates
    """
    prediction = self.forward(x)
    if prediction != y:
        # Move decision boundary towards correct classification
        adjustment = learning_rate * y
        self.weights += adjustment * x
        self.bias += adjustment
        return True  # indicating an update was made
    return False
```

Slide 6: Visualization of Decision Boundary

A crucial aspect of understanding perceptron behavior is visualizing its decision boundary. This implementation creates a comprehensive visualization showing how the decision boundary evolves during training.

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 8))
    
    # Create grid points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Get predictions for grid points
    Z = np.array([model.forward([x1, x2]) 
                 for x1, x2 in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title("Perceptron Decision Boundary")
    plt.show()
```

Slide 7: Batch Training Implementation

Batch training processes multiple samples before updating weights, offering more stable convergence. This implementation shows how to perform batch updates efficiently.

```python
def batch_train(self, X, y, batch_size=32, learning_rate=0.1, epochs=100):
    n_samples = len(X)
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        total_error = 0
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Calculate batch updates
            batch_predictions = np.array([self.forward(xi) for xi in batch_X])
            batch_errors = batch_y - batch_predictions
            
            # Update weights and bias
            self.weights += learning_rate * np.dot(batch_X.T, batch_errors) / batch_size
            self.bias += learning_rate * np.mean(batch_errors)
            
            total_error += np.sum(np.abs(batch_errors))
            
        if total_error == 0:
            print(f"Converged at epoch {epoch}")
            break
```

Slide 8: Early Stopping Implementation

Early stopping prevents overfitting by monitoring performance on a validation set. This implementation demonstrates how to implement this crucial technique for perceptron training.

```python
def train_with_early_stopping(self, X_train, y_train, X_val, y_val, 
                            learning_rate=0.1, epochs=100, patience=5):
    best_val_accuracy = 0
    patience_counter = 0
    best_weights = None
    best_bias = None
    
    for epoch in range(epochs):
        # Train on training data
        self.train(X_train, y_train, learning_rate, epochs=1)
        
        # Evaluate on validation set
        val_accuracy = accuracy(self, X_val, y_val)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = self.weights.copy()
            best_bias = self.bias
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            self.weights = best_weights
            self.bias = best_bias
            break
            
    return best_val_accuracy
```

Slide 9: Learning Rate Scheduling

Adaptive learning rates can improve convergence. This implementation shows dynamic learning rate adjustment based on training progress.

```python
def exponential_decay_lr(initial_lr, epoch, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)

def train_with_lr_scheduling(self, X, y, initial_lr=0.1, epochs=100):
    for epoch in range(epochs):
        current_lr = exponential_decay_lr(initial_lr, epoch)
        errors = 0
        
        for xi, yi in zip(X, y):
            prediction = self.forward(xi)
            if prediction != yi:
                # Update with decayed learning rate
                self.weights += current_lr * yi * xi
                self.bias += current_lr * yi
                errors += 1
                
        print(f"Epoch {epoch}, LR: {current_lr:.4f}, Errors: {errors}")
        if errors == 0:
            break
```

Slide 10: Multi-class Perceptron Implementation

Extending the perceptron to handle multiple classes requires maintaining separate weight vectors for each class. This implementation shows how to create a multi-class perceptron using the one-vs-all approach.

```python
class MultiClassPerceptron:
    def __init__(self, input_size, num_classes):
        self.perceptrons = [Perceptron(input_size) for _ in range(num_classes)]
        
    def predict(self, x):
        # Get scores from all perceptrons
        scores = [p.forward(x) for p in self.perceptrons]
        # Return class with highest score
        return np.argmax(scores)
    
    def train(self, X, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # Convert to one-vs-all format
                true_labels = [-1] * len(self.perceptrons)
                true_labels[yi] = 1
                
                # Train each perceptron
                for idx, perceptron in enumerate(self.perceptrons):
                    if perceptron.perceptron_trick(xi, true_labels[idx], learning_rate):
                        errors += 1
                        
            if errors == 0:
                break
```

Slide 11: Regularized Perceptron

Implementing regularization helps prevent overfitting by penalizing large weights. This implementation shows L2 regularization in the perceptron algorithm.

```python
class RegularizedPerceptron:
    def __init__(self, input_size, lambda_reg=0.01):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lambda_reg = lambda_reg
        
    def train(self, X, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.forward(xi)
                if prediction != yi:
                    # Update with regularization
                    self.weights = (1 - learning_rate * self.lambda_reg) * self.weights + \
                                 learning_rate * yi * xi
                    self.bias += learning_rate * yi
                    errors += 1
            
            # Apply regularization to weights
            self.weights *= (1 - self.lambda_reg)
            
            if errors == 0:
                break
```

Slide 12: Real-world Application - Iris Dataset Classification

Implementing the perceptron for a practical multi-class classification problem using the famous Iris dataset, demonstrating complete workflow from data preprocessing to evaluation.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multi-class perceptron
model = MultiClassPerceptron(input_size=4, num_classes=3)
model.train(X_train, y_train, learning_rate=0.01, epochs=100)

# Evaluate
def evaluate(model, X, y):
    predictions = [model.predict(xi) for xi in X]
    accuracy = np.mean(predictions == y)
    return accuracy

print(f"Test Accuracy: {evaluate(model, X_test, y_test):.2f}")
```

Slide 13: Additional Resources

*   "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" - Original perceptron paper [https://psycnet.apa.org/record/1959-09865-001](https://psycnet.apa.org/record/1959-09865-001)
*   "A Tutorial on Support Vector Machines and the Perceptron Algorithm" [https://www.cs.cmu.edu/~avrim/ML07/lect0118.pdf](https://www.cs.cmu.edu/~avrim/ML07/lect0118.pdf)
*   "Learning Internal Representations by Error Propagation" - Historical development from perceptron to modern neural networks [https://web.stanford.edu/class/psych209a/ReadingsByDate/02\_01/PDPVolI.pdf](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_01/PDPVolI.pdf)
*   "Perceptron Learning with Random Projection" - Modern developments [https://www.jmlr.org/papers/v8/arriaga07a.html](https://www.jmlr.org/papers/v8/arriaga07a.html)

