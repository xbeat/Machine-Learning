## Introduction to Supervised and Unsupervised Machine Learning
Slide 1: Introduction to Supervised Learning Classification

Supervised learning classification involves training models to categorize data into predefined classes using labeled examples. This fundamental approach requires preparing training data where each instance has a known target class, enabling the model to learn decision boundaries for future predictions.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

class SimpleClassifier:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, epochs=1000, lr=0.01):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(epochs):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Backward pass
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= lr * dw
            self.bias -= lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(linear_pred)
        return [1 if p > 0.5 else 0 for p in predictions]

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
clf = SimpleClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
```

Slide 2: Supervised Learning Regression Implementation

Regression in supervised learning predicts continuous values by learning patterns from labeled training data. This implementation demonstrates a basic linear regression model built from scratch, incorporating gradient descent optimization for parameter learning.

```python
import numpy as np
from sklearn.datasets import make_regression

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(epochs):
            # Make predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Train model
reg = LinearRegression()
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)
mse = np.mean((y - predictions) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
```

Slide 3: K-Means Clustering from Scratch

K-means clustering is an unsupervised learning algorithm that partitions data into K distinct clusters. This implementation showcases the core algorithm components including centroid initialization, assignment, and update steps using NumPy operations.

```python
import numpy as np
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.k)])
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Generate synthetic clustering data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)

# Train and predict
kmeans = KMeansClustering(k=3)
kmeans.fit(X)
labels = kmeans.predict(X)
```

Slide 4: Neural Network Forward Propagation Implementation

Neural network forward propagation involves computing sequential layer activations to transform input data into predictions. This implementation demonstrates a basic feedforward neural network with customizable architecture and activation functions.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        current_activation = X
        activations = [X]
        
        # Forward propagate through layers
        for i in range(len(self.weights) - 1):
            z = np.dot(current_activation, self.weights[i]) + self.biases[i]
            current_activation = self.relu(z)
            activations.append(current_activation)
        
        # Output layer with softmax
        z_final = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        output = self.softmax(z_final)
        activations.append(output)
        
        return activations

# Example usage
nn = NeuralNetwork([784, 128, 64, 10])  # MNIST-like architecture
X = np.random.randn(32, 784)  # 32 samples of 784 features
activations = nn.forward(X)
predictions = np.argmax(activations[-1], axis=1)
```

Slide 5: Support Vector Machine Implementation

Support Vector Machines find optimal hyperplanes to separate data classes by maximizing the margin between classes. This implementation uses the Sequential Minimal Optimization (SMO) algorithm for binary classification.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class SimpleSVM:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.alpha = None
        self.b = None
        self.support_vectors = None
        
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)
    
    def rbf_kernel(self, x1, x2, gamma=0.1):
        norm = np.linalg.norm(x1[:, np.newaxis] - x2, axis=2)
        return np.exp(-gamma * norm ** 2)
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        if self.kernel == 'linear':
            K = self.linear_kernel(X, X)
        else:
            K = self.rbf_kernel(X, X)
        
        # SMO Algorithm
        for _ in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            
            for i in range(n_samples):
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                
                eta = 2.0 * K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue
                    
                alpha_i_old = self.alpha[i]
                alpha_j_old = self.alpha[j]
                
                # Calculate bounds
                if y[i] != y[j]:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                else:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                if L == H:
                    continue
                
                # Update alpha
                self.alpha[j] = alpha_j_old - (y[j] * (self.decision_function(X[i]) - y[i] - 
                               self.decision_function(X[j]) + y[j])) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
            
            # Check convergence
            if np.allclose(self.alpha, alpha_prev):
                break
                
        # Store support vectors
        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.alpha = self.alpha[sv]
        self.sv_y = y[sv]
    
    def decision_function(self, X):
        if self.kernel == 'linear':
            kernel = self.linear_kernel(X, self.support_vectors)
        else:
            kernel = self.rbf_kernel(X, self.support_vectors)
        return np.dot(kernel, self.alpha * self.sv_y) + self.b
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

# Example usage
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

svm = SimpleSVM(C=1.0)
svm.fit(X, y)
predictions = svm.predict(X)
```

Slide 6: Decision Tree Implementation

The decision tree algorithm recursively partitions data based on feature thresholds to create a tree-like model for classification or regression. This implementation showcases binary classification with information gain splitting criterion.

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        return (self.entropy(parent) - 
                weight_left * self.entropy(left_child) - 
                weight_right * self.entropy(right_child))
    
    def split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], X[right_mask], 
                y[left_mask], y[right_mask])
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                gain = self.information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_classes == 1 or n_samples < 2):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Create child nodes
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        
        return Node(best_feature, best_threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

# Example usage
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

tree = DecisionTree(max_depth=5)
tree.fit(X, y)
predictions = tree.predict(X)
```

Slide 7: Ensemble Learning - Random Forest Implementation

Random Forest combines multiple decision trees to create a robust classifier that reduces overfitting through bagging and random feature selection. This implementation demonstrates the core concepts of ensemble learning with parallel tree construction.

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _build_tree(self, X, y):
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features
        )
        X_sample, y_sample = self.bootstrap_sample(X, y)
        tree.fit(X_sample, y_sample)
        return tree
    
    def fit(self, X, y):
        self.n_features = self.n_features or int(np.sqrt(X.shape[1]))
        
        # Parallel tree construction
        with ThreadPoolExecutor() as executor:
            self.trees = list(executor.map(
                lambda _: self._build_tree(X, y),
                range(self.n_trees)
            ))
    
    def predict(self, X):
        # Get predictions from all trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.trees
        ])
        
        # Return majority vote
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=tree_predictions
        )

# Example usage with synthetic data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000, 
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

rf = RandomForest(n_trees=100, max_depth=10)
rf.fit(X, y)
predictions = rf.predict(X)
```

Slide 8: Gradient Boosting Implementation

Gradient Boosting builds an ensemble of weak learners sequentially, where each new model tries to correct the errors of previous models. This implementation shows binary classification with decision trees as base learners.

```python
class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        # Initialize predictions with zeros
        F = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            # Calculate negative gradient (residuals)
            p = self.sigmoid(F)
            residuals = y - p
            
            # Fit a tree to the residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update model
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            
            self.trees.append(tree)
    
    def predict_proba(self, X):
        # Initialize predictions
        F = np.zeros(len(X))
        
        # Add up predictions from all trees
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        
        return self.sigmoid(F)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Example usage
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    random_state=42
)

gb = GradientBoosting(n_estimators=100, learning_rate=0.1)
gb.fit(X, y)
predictions = gb.predict(X)
probabilities = gb.predict_proba(X)

print(f"Accuracy: {np.mean(predictions == y):.4f}")
```

Slide 9: Neural Network Backpropagation Implementation

Backpropagation is the core algorithm for training neural networks by computing gradients of the loss function with respect to weights. This implementation demonstrates the complete training process with gradient descent optimization.

```python
import numpy as np

class NeuralNetworkWithBackprop:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        self.gradients = {'weights': [], 'biases': []}
        
        # Initialize weights and biases
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1)
            self.biases.append(np.random.randn(1, layer_sizes[i+1]) * 0.1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
    def forward_pass(self, X):
        activations = [X]
        layer_inputs = []
        
        current = X
        for w, b in zip(self.weights, self.biases):
            layer_input = np.dot(current, w) + b
            layer_inputs.append(layer_input)
            current = self.relu(layer_input)
            activations.append(current)
        
        # Softmax for output layer
        exp_scores = np.exp(layer_inputs[-1])
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        activations[-1] = probs
        
        return activations, layer_inputs
    
    def backward_pass(self, X, y, activations, layer_inputs):
        m = X.shape[0]
        deltas = []
        
        # Output layer error
        delta = activations[-1] - y
        deltas.append(delta)
        
        # Hidden layers
        for i in range(len(self.weights)-1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(layer_inputs[i-1])
            deltas.append(delta)
        deltas = deltas[::-1]
        
        # Compute gradients
        self.gradients['weights'] = []
        self.gradients['biases'] = []
        
        for i in range(len(self.weights)):
            weight_grad = np.dot(activations[i].T, deltas[i]) / m
            bias_grad = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            self.gradients['weights'].append(weight_grad)
            self.gradients['biases'].append(bias_grad)
    
    def train(self, X, y, epochs=100, learning_rate=0.01):
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, layer_inputs = self.forward_pass(X)
            
            # Backward pass
            self.backward_pass(X, y, activations, layer_inputs)
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * self.gradients['weights'][i]
                self.biases[i] -= learning_rate * self.gradients['biases'][i]
            
            # Calculate loss
            loss = self.cross_entropy_loss(y, activations[-1])
            losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses

# Example usage
X = np.random.randn(100, 20)  # 100 samples, 20 features
y = np.eye(3)[np.random.randint(0, 3, 100)]  # One-hot encoded labels for 3 classes

nn = NeuralNetworkWithBackprop([20, 16, 8, 3])  # 20 input, 16 and 8 hidden, 3 output
losses = nn.train(X, y, epochs=100, learning_rate=0.01)
```

Slide 10: Principal Component Analysis Implementation

Principal Component Analysis (PCA) reduces data dimensionality by projecting it onto principal components that capture maximum variance. This implementation shows the complete PCA algorithm with eigendecomposition.

```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues[:self.n_components] / total_var
        
        return self
    
    def transform(self, X):
        # Project data onto principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        # Project data back to original space
        return np.dot(X_transformed, self.components.T) + self.mean

# Example usage with synthetic data
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=1000, n_features=50, centers=5, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
pca.fit(X)
X_transformed = pca.transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_transformed.shape)
print("Explained variance ratio:", pca.explained_variance_ratio)
```

Slide 11: Validation and Cross-Validation Implementation

Robust model validation is crucial for assessing generalization performance. This implementation demonstrates various validation techniques including k-fold cross-validation and stratified sampling.

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelValidator:
    def __init__(self, model, n_splits=5, random_state=42):
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        
    def train_validation_split(self, X, y, test_size=0.2):
        np.random.seed(self.random_state)
        indices = np.random.permutation(len(X))
        test_size = int(len(X) * test_size)
        
        train_idx = indices[test_size:]
        val_idx = indices[:test_size]
        
        return (X[train_idx], X[val_idx], 
                y[train_idx], y[val_idx])
    
    def cross_validate(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, 
                            shuffle=True, 
                            random_state=self.random_state)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone model for each fold
            model = clone(self.model)
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_val, y_pred, average='weighted'))
            metrics['f1'].append(f1_score(y_val, y_pred, average='weighted'))
        
        # Calculate mean and std for each metric
        self.cv_results = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for metric, scores in metrics.items()
        }
        
        return self.cv_results
    
    def print_results(self):
        print("\nCross-validation results:")
        for metric, stats in self.cv_results.items():
            print(f"{metric.capitalize()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")

# Example usage
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data
X, y = make_classification(n_samples=1000, 
                         n_features=20,
                         n_informative=15,
                         n_redundant=5,
                         random_state=42)

# Initialize validator with a model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
validator = ModelValidator(rf, n_splits=5)

# Perform cross-validation
results = validator.cross_validate(X, y)
validator.print_results()
```

Slide 12: Feature Selection and Engineering Implementation

Feature selection optimizes model performance by identifying the most relevant features while reducing dimensionality. This implementation demonstrates multiple feature selection techniques including mutual information and recursive feature elimination.

```python
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

class FeatureSelector:
    def __init__(self):
        self.feature_scores = {}
        self.selected_features = None
        
    def mutual_information(self, X, y):
        mi_scores = mutual_info_classif(X, y)
        self.feature_scores['mutual_info'] = {
            i: score for i, score in enumerate(mi_scores)
        }
        return mi_scores
    
    def chi_square_test(self, X, y, bins=10):
        chi_scores = []
        p_values = []
        
        for feature in range(X.shape[1]):
            # Discretize continuous features
            x_binned = np.histogram(X[:, feature], bins=bins)[0]
            contingency_table = np.histogram2d(X[:, feature], y, bins=bins)[0]
            
            # Calculate chi-square
            chi2, p_val, _, _ = chi2_contingency(contingency_table)
            chi_scores.append(chi2)
            p_values.append(p_val)
        
        self.feature_scores['chi_square'] = {
            'scores': chi_scores,
            'p_values': p_values
        }
        return chi_scores, p_values
    
    def recursive_feature_elimination(self, X, y, model, n_features_to_select=None):
        if n_features_to_select is None:
            n_features_to_select = X.shape[1] // 2
            
        n_features = X.shape[1]
        feature_ranks = np.zeros(n_features)
        remaining_features = list(range(n_features))
        
        for rank in range(n_features - n_features_to_select):
            # Fit model with current features
            model.fit(X[:, remaining_features], y)
            
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                raise ValueError("Model must have feature_importances_ or coef_ attribute")
            
            # Remove least important feature
            least_important = np.argmin(importances)
            feature_ranks[remaining_features[least_important]] = rank + 1
            remaining_features.pop(least_important)
        
        # Assign top rank to remaining features
        feature_ranks[remaining_features] = n_features
        self.feature_scores['rfe'] = feature_ranks
        return feature_ranks
    
    def select_features(self, X, y, method='mutual_info', threshold=0.05):
        if method == 'mutual_info':
            scores = self.mutual_information(X, y)
            selected = scores > threshold
        elif method == 'chi_square':
            _, p_values = self.chi_square_test(X, y)
            selected = np.array(p_values) < threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.selected_features = selected
        return X[:, selected]

# Example usage
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=5,
    random_state=42
)

# Initialize selector and model
selector = FeatureSelector()
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply different selection methods
mi_scores = selector.mutual_information(X, y)
chi_scores, p_values = selector.chi_square_test(X, y)
rfe_ranks = selector.recursive_feature_elimination(X, y, rf, n_features_to_select=10)

# Select features using mutual information
X_selected = selector.select_features(X, y, method='mutual_info', threshold=0.1)
print(f"Original features: {X.shape[1]}")
print(f"Selected features: {X_selected.shape[1]}")
```

Slide 13: Deep Learning Model Pipeline Implementation

This implementation demonstrates a complete deep learning pipeline including data preprocessing, model architecture definition, training loop optimization, and evaluation metrics tracking.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        sample = self.X[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class DeepLearningPipeline:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            *[layer for hidden_size in hidden_sizes[1:] for layer in (
                nn.Linear(hidden_sizes[i-1], hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            )],
            nn.Linear(hidden_sizes[-1], output_size)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Update history
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['val_loss'].append(val_loss / len(val_loader))
            self.history['train_acc'].append(train_correct / train_total)
            self.history['val_acc'].append(val_correct / val_total)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Train Acc: {train_correct/train_total:.4f}")
                print(f"Val Loss: {val_loss/len(val_loader):.4f}, "
                      f"Val Acc: {val_correct/val_total:.4f}")
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.numpy())
                
        return np.array(predictions)

# Example usage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Create data loaders
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize and train model
pipeline = DeepLearningPipeline(
    input_size=20,
    hidden_sizes=[64, 32],
    output_size=3
)

pipeline.train(train_loader, val_loader, epochs=100)
predictions = pipeline.predict(test_loader)
```

Slide 14: Additional Resources

*   "Deep Learning for Computer Vision: A Comprehensive Review"
    *   [https://arxiv.org/abs/2001.06523](https://arxiv.org/abs/2001.06523)
*   "A Survey of Deep Learning Techniques for Neural Networks"
    *   [https://arxiv.org/abs/1910.03151](https://arxiv.org/abs/1910.03151)
*   "Machine Learning: A Review of Classification Techniques"
    *   [https://arxiv.org/abs/2110.01556](https://arxiv.org/abs/2110.01556)
*   "Recent Advances in Deep Learning: An Overview"
    *   [https://arxiv.org/abs/1905.05697](https://arxiv.org/abs/1905.05697)
*   Suggested searches:
    *   "Deep Learning fundamentals and implementations"
    *   "Machine Learning algorithms from scratch"
    *   "Neural Network architecture design patterns"
    *   "Best practices in ML model validation"

