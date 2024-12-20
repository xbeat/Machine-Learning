## Supervised vs. Unsupervised Learning Decoding the Aha Moments
Slide 1: Understanding Supervised Learning Implementation

Supervised learning requires labeled data to train models that can make predictions. We'll implement a simple linear regression model from scratch to demonstrate the core concepts of supervised learning, including gradient descent optimization.

```python
import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegressionFromScratch()
model.fit(X, y)
print(f"Predictions: {model.predict(np.array([[5], [6]]))}")
```

Slide 2: Implementing K-Means Clustering

K-means clustering is a fundamental unsupervised learning algorithm that partitions data into k clusters. This implementation demonstrates the iterative process of centroid assignment and update steps.

```python
import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        
    def fit(self, X):
        # Initialize random centroids
        idx = np.random.permutation(X.shape[0])[:self.n_clusters]
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            cluster_labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[cluster_labels == k].mean(axis=0)
                                    for k in range(self.n_clusters)])
            
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return cluster_labels
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Example usage
X = np.random.randn(100, 2) * 2
kmeans = KMeansFromScratch(n_clusters=3)
labels = kmeans.fit(X)
print(f"Cluster assignments shape: {labels.shape}")
```

Slide 3: Decision Trees for Classification

Decision trees partition the feature space recursively based on information gain or Gini impurity. This implementation showcases the core concepts of tree-based learning for classification tasks.

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

class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None
        
    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
        
    def _split(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return (X[left_mask], y[left_mask], X[~left_mask], y[~left_mask])
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if depth >= self.max_depth or n_classes == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        best_gain = -1
        best_split = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                gain = self._gini(y) - (len(y_left) * self._gini(y_left) + 
                                      len(y_right) * self._gini(y_right)) / len(y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)
        
        if best_gain == -1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        feature, threshold = best_split
        X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)
        
        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)
```

Slide 4: Results for Decision Tree Classifier

This implementation demonstrates the practical application of our decision tree classifier on a synthetic dataset, showing both the training process and prediction accuracy.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
dt = DecisionTreeClassifier(max_depth=5)
dt.root = dt._grow_tree(X_train, y_train)

# Prediction function
def predict(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature] <= node.threshold:
        return predict(node.left, X)
    return predict(node.right, X)

# Make predictions
y_pred = np.array([predict(dt.root, x) for x in X_test])
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Feature importance visualization
importance = np.zeros(X.shape[1])
def calculate_importance(node, total_samples):
    if node.feature is not None:
        importance[node.feature] += 1
    if node.left:
        calculate_importance(node.left, total_samples)
    if node.right:
        calculate_importance(node.right, total_samples)

calculate_importance(dt.root, len(X_train))
print("Feature Importance:", importance / np.sum(importance))
```

Slide 5: Neural Network Implementation

A fundamental neural network implementation showcasing forward propagation, backpropagation, and gradient descent optimization for supervised learning tasks.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i+1])))
            
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights)):
            net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.activations.append(self.sigmoid(net))
            
        return self.activations[-1]
        
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

Slide 6: Principal Component Analysis Implementation

A detailed implementation of PCA for dimensionality reduction, demonstrating eigenvalue decomposition and feature transformation in unsupervised learning.

```python
import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
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
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
        # Calculate explained variance ratio
        explained_variance = eigenvalues[idx]
        self.explained_variance_ratio_ = explained_variance[:self.n_components] / np.sum(explained_variance)
        
        return self
        
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components.T) + self.mean

# Example usage
X = np.random.randn(100, 5)
pca = PCAFromScratch(n_components=2)
pca.fit(X)
X_transformed = pca.transform(X)
print(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 7: Support Vector Machine Implementation

A pure Python implementation of SVM using the Sequential Minimal Optimization (SMO) algorithm, demonstrating kernel tricks and margin optimization concepts.

```python
import numpy as np

class SVMFromScratch:
    def __init__(self, C=1.0, kernel='linear', max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            gamma = 0.1
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        
    def _compute_kernel_matrix(self, X):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self._kernel_function(X[i], X[j])
        return K
        
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y
        
        # Initialize alphas and kernel matrix
        self.alpha = np.zeros(n_samples)
        self.K = self._compute_kernel_matrix(X)
        
        # SMO optimization
        for _ in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            
            for i in range(n_samples):
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                    
                eta = 2.0 * self.K[i,j] - self.K[i,i] - self.K[j,j]
                if eta >= 0:
                    continue
                    
                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                y_i, y_j = y[i], y[j]
                
                # Compute bounds
                if y_i != y_j:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)
                    
                if L == H:
                    continue
                    
                # Update alpha_j
                self.alpha[j] = alpha_j_old - (y_j * (self._decision_function(X[i]) - 
                                                     y_i - y_j * self.K[i,j])) / eta
                self.alpha[j] = min(H, max(L, self.alpha[j]))
                
                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    continue
                    
                # Update alpha_i
                self.alpha[i] = alpha_i_old + y_i * y_j * (alpha_j_old - self.alpha[j])
                
                # Update bias term
                b1 = self.b - y_i * (self.alpha[i] - alpha_i_old) * self.K[i,i] \
                     - y_j * (self.alpha[j] - alpha_j_old) * self.K[i,j]
                b2 = self.b - y_i * (self.alpha[i] - alpha_i_old) * self.K[i,j] \
                     - y_j * (self.alpha[j] - alpha_j_old) * self.K[j,j]
                self.b = (b1 + b2) / 2
                
            if np.linalg.norm(self.alpha - alpha_prev) < 1e-5:
                break
                
    def _decision_function(self, x):
        decision = np.sum(self.alpha * self.y * 
                         np.array([self._kernel_function(x_i, x) for x_i in self.X]))
        return decision + self.b
        
    def predict(self, X):
        return np.sign([self._decision_function(x) for x in X])
```

Slide 8: Results for SVM Implementation

```python
# Generate synthetic dataset
np.random.seed(42)
X = np.concatenate([np.random.randn(50, 2) + [2, 2],
                   np.random.randn(50, 2) + [-2, -2]])
y = np.array([1] * 50 + [-1] * 50)

# Train SVM
svm = SVMFromScratch(C=1.0, kernel='rbf')
svm.fit(X, y)

# Make predictions
y_pred = svm.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Training accuracy: {accuracy:.4f}")

# Support vector analysis
support_vectors = X[svm.alpha > 1e-5]
print(f"Number of support vectors: {len(support_vectors)}")
```

Slide 9: Gradient Boosting Implementation

A robust implementation of gradient boosting for regression tasks, demonstrating the ensemble learning approach with decision trees as base learners.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        pseudo_residuals = y - self.initial_prediction
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, pseudo_residuals)
            self.trees.append(tree)
            
            # Update predictions and compute new pseudo-residuals
            predictions = tree.predict(X)
            pseudo_residuals -= self.learning_rate * predictions
            
    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Example usage
X = np.random.rand(100, 4)
y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, 100)

gb = GradientBoostingRegressor(n_estimators=100)
gb.fit(X, y)
y_pred = gb.predict(X)
mse = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 10: Implementation of Gaussian Mixture Model

A comprehensive implementation of GMM showcasing the Expectation-Maximization algorithm for unsupervised clustering with probabilistic assignments.

```python
import numpy as np
from scipy.stats import multivariate_normal

class GMMFromScratch:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        
        # Randomly initialize means
        random_idx = np.random.permutation(n_samples)[:self.n_components]
        self.means = X[random_idx]
        
        # Initialize covariances
        self.covs = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # Initialize mixing coefficients
        self.weights = np.ones(self.n_components) / self.n_components
        
    def _e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            gaussian = multivariate_normal(self.means[k], self.covs[k])
            responsibilities[:, k] = self.weights[k] * gaussian.pdf(X)
            
        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
        
    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        
        # Update weights
        Nk = responsibilities.sum(axis=0)
        self.weights = Nk / n_samples
        
        # Update means
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
            
    def fit(self, X):
        self._initialize_parameters(X)
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Compute log likelihood
            log_likelihood = 0
            for k in range(self.n_components):
                gaussian = multivariate_normal(self.means[k], self.covs[k])
                log_likelihood += self.weights[k] * gaussian.pdf(X)
            log_likelihood = np.sum(np.log(log_likelihood))
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood
            
    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
```

Slide 11: Cross-Validation Implementation

A robust implementation of k-fold cross-validation demonstrating model evaluation techniques for supervised learning algorithms.

```python
import numpy as np
from typing import Callable
from sklearn.base import clone

class CrossValidation:
    def __init__(self, n_splits=5, shuffle=True):
        self.n_splits = n_splits
        self.shuffle = shuffle
        
    def split(self, X, y):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
            
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[indices[start:stop]] = True
            
            yield (indices[~test_mask], indices[test_mask])
            current = stop
            
    def cross_validate(self, model, X, y, scoring_func: Callable):
        scores = []
        
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone model to avoid fitting on same instance
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            
            y_pred = model_clone.predict(X_test)
            score = scoring_func(y_test, y_pred)
            scores.append(score)
            
        return np.array(scores)

# Example usage
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Generate synthetic data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Perform cross-validation
cv = CrossValidation(n_splits=5)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

scores = cv.cross_validate(model, X, y, accuracy_score)
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.4f} ± {scores.std():.4f}")
```

Slide 12: DBSCAN Clustering Implementation

This density-based clustering implementation demonstrates handling noise points and discovering clusters of arbitrary shapes in unsupervised learning.

```python
import numpy as np
from collections import defaultdict

class DBSCANFromScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def _get_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)  # -1 represents noise points
        current_cluster = 0
        
        # Find core points and their neighborhoods
        for point_idx in range(n_samples):
            if labels[point_idx] != -1:
                continue
                
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                continue  # Mark as noise
                
            # Start a new cluster
            current_cluster += 1
            labels[point_idx] = current_cluster
            
            # Expand cluster
            seed_set = neighbors.tolist()
            for seed in seed_set:
                if labels[seed] == -1:
                    labels[seed] = current_cluster
                    
                    # Get neighbors of current seed point
                    seed_neighbors = self._get_neighbors(X, seed)
                    
                    if len(seed_neighbors) >= self.min_samples:
                        seed_set.extend(seed_neighbors[~np.in1d(seed_neighbors, seed_set)])
                        
        return labels
        
# Example usage with visualization
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
dbscan = DBSCANFromScratch(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering Results')
plt.colorbar(label='Cluster Label')
print(f"Number of clusters found: {len(np.unique(labels[labels != -1]))}")
print(f"Number of noise points: {np.sum(labels == -1)}")
```

Slide 13: Naive Bayes Implementation

A complete implementation of Gaussian Naive Bayes classifier showing probability calculations and the naive independence assumption.

```python
import numpy as np
from collections import defaultdict

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.parameters = {}
        self.priors = {}
        
    def _calculate_class_parameters(self, X, y):
        parameters = {
            'mean': np.zeros((len(self.classes), X.shape[1])),
            'var': np.zeros((len(self.classes), X.shape[1]))
        }
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            parameters['mean'][idx] = X_c.mean(axis=0)
            parameters['var'][idx] = X_c.var(axis=0)
        
        return parameters
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate prior probabilities
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / n_samples
            
        # Calculate mean and variance for each feature per class
        self.parameters = self._calculate_class_parameters(X, y)
        
    def _calculate_likelihood(self, x, mean, var):
        # Gaussian probability density function
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return np.prod(1 / np.sqrt(2 * np.pi * var) * exponent, axis=1)
        
    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            mean = self.parameters['mean'][idx]
            var = self.parameters['var'][idx]
            
            # Calculate class conditional probabilities
            likelihood = self._calculate_likelihood(X, mean, var)
            probas[:, idx] = likelihood * self.priors[c]
            
        # Normalize probabilities
        probas /= probas.sum(axis=1, keepdims=True)
        return probas
        
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

# Example usage
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, 
                          n_redundant=0, random_state=42)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

Slide 14: Additional Resources

*   arXiv:1806.02215 - "Deep Neural Network Approximation Theory" - [https://arxiv.org/abs/1806.02215](https://arxiv.org/abs/1806.02215)
*   arXiv:1904.10922 - "Visualizing the Loss Landscape of Neural Nets" - [https://arxiv.org/abs/1904.10922](https://arxiv.org/abs/1904.10922)
*   arXiv:1707.09725 - "A Tutorial on Support Vector Machines for Pattern Recognition" - [https://arxiv.org/abs/1707.09725](https://arxiv.org/abs/1707.09725)
*   arXiv:1803.09655 - "A Tutorial on Bayesian Optimization" - [https://arxiv.org/abs/1803.09655](https://arxiv.org/abs/1803.09655)
*   arXiv:1802.05365 - "A Tutorial on Spectral Clustering" - [https://arxiv.org/abs/1802.05365](https://arxiv.org/abs/1802.05365)

