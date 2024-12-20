## Understanding the 3 Main Types of Machine Learning
Slide 1: Supervised Learning - Linear Regression Implementation

Linear regression serves as a foundational supervised learning algorithm that models the relationship between dependent and independent variables. This implementation demonstrates how to create a simple linear regression model from scratch using numpy, focusing on the mathematical principles behind the algorithm.

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
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
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
```

Slide 2: Unsupervised Learning - K-Means Clustering

K-means clustering algorithm partitions n observations into k clusters by iteratively updating cluster centroids. This implementation shows how to create a k-means clustering algorithm from scratch, including centroid initialization and cluster assignment.

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign clusters
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
X = np.random.randn(300, 2) * 2
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit(X)
```

Slide 3: Reinforcement Learning - Q-Learning Algorithm

Q-Learning is a model-free reinforcement learning algorithm that learns to make optimal decisions by maintaining a Q-table of state-action pairs. This implementation demonstrates a simple Q-learning agent in a discrete environment.

```python
import numpy as np

class QLearningAgent:
    def __init__(self, states, actions, learning_rate=0.1, discount=0.95, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning formula
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# Example usage
n_states = 10
n_actions = 4
agent = QLearningAgent(n_states, n_actions)

# Training loop example
state = 0
for _ in range(1000):
    action = agent.choose_action(state)
    next_state = min(state + action, n_states - 1)  # Simple environment
    reward = 1 if next_state == n_states - 1 else 0
    
    agent.learn(state, action, reward, next_state)
    state = next_state if next_state != n_states - 1 else 0
```

Slide 4: Neural Network Implementation from Scratch

Neural networks form the backbone of deep learning, consisting of interconnected layers of neurons. This implementation shows how to create a basic feedforward neural network with backpropagation using only numpy.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = [np.random.randn(y, x) * 0.01 
                       for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.zeros((y, 1)) for y in layers[1:]]
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.activations = [X]
        for w, b in zip(self.weights, self.biases):
            net = np.dot(w, self.activations[-1]) + b
            self.activations.append(self.sigmoid(net))
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate):
        m = X.shape[1]
        delta = self.activations[-1] - y
        
        for l in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(delta, self.activations[l].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * \
                        self.sigmoid_derivative(self.activations[l])
            
            self.weights[l] -= learning_rate * dW
            self.biases[l] -= learning_rate * db

# Example usage
nn = NeuralNetwork([2, 4, 1])
X = np.random.randn(2, 100)
y = np.array([int(x1 > x2) for x1, x2 in X.T]).reshape(1, -1)

for _ in range(1000):
    output = nn.forward(X)
    nn.backward(X, y, 0.1)
```

Slide 5: Support Vector Machine Implementation

Support Vector Machines find the optimal hyperplane that separates classes by maximizing the margin between them. This implementation demonstrates a simplified SVM using the Sequential Minimal Optimization (SMO) algorithm for binary classification.

```python
import numpy as np

class SVM:
    def __init__(self, C=1.0, max_iter=100):
        self.C = C
        self.max_iter = max_iter
        
    def kernel(self, x1, x2):
        return np.dot(x1, x2)  # Linear kernel
        
    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y
        
        # Initialize alphas and bias
        self.alphas = np.zeros(self.n_samples)
        self.b = 0
        
        # SMO Algorithm
        for _ in range(self.max_iter):
            alpha_pairs_changed = 0
            for i in range(self.n_samples):
                Ei = self._decision_function(X[i]) - y[i]
                
                if (y[i] * Ei < -0.001 and self.alphas[i] < self.C) or \
                   (y[i] * Ei > 0.001 and self.alphas[i] > 0):
                    
                    j = np.random.randint(0, self.n_samples)
                    while j == i:
                        j = np.random.randint(0, self.n_samples)
                        
                    Ej = self._decision_function(X[j]) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * self.kernel(X[i], X[j]) - \
                          self.kernel(X[i], X[i]) - \
                          self.kernel(X[j], X[j])
                    
                    if eta >= 0:
                        continue
                    
                    # Update alpha j
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update threshold b
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * \
                         self.kernel(X[i], X[i]) - \
                         y[j] * (self.alphas[j] - alpha_j_old) * \
                         self.kernel(X[i], X[j])
                    
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * \
                         self.kernel(X[i], X[j]) - \
                         y[j] * (self.alphas[j] - alpha_j_old) * \
                         self.kernel(X[j], X[j])
                    
                    self.b = (b1 + b2) / 2
                    alpha_pairs_changed += 1
                    
            if alpha_pairs_changed == 0:
                break
                
    def _decision_function(self, X):
        return np.sum(self.alphas * self.y * \
               np.apply_along_axis(lambda x: self.kernel(x, X), 1, self.X)) + self.b
    
    def predict(self, X):
        return np.sign([self._decision_function(x) for x in X])

# Example usage
X = np.random.randn(100, 2)
y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])

svm = SVM(C=1.0)
svm.fit(X, y)
predictions = svm.predict(X)
```

Slide 6: Decision Tree Implementation

Decision trees are versatile machine learning algorithms that make predictions by learning simple decision rules from data. This implementation shows how to build a decision tree classifier from scratch with information gain splitting criterion.

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, 
                 right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None
        
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
        
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(y[left_mask]), len(y[right_mask])
        e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        return parent_entropy - child_entropy
        
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or n_samples < 2:
            leaf_value = max(Counter(y).items(), key=lambda x: x[1])[0]
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        # Create child splits
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

# Example usage
X = np.random.randn(100, 2)
y = np.array([0 if x[0] + x[1] > 0 else 1 for x in X])

tree = DecisionTree(max_depth=5)
tree.fit(X, y)
predictions = tree.predict(X)
```

Slide 7: Gradient Boosting Implementation

Gradient Boosting combines multiple weak learners into a strong predictor by iteratively fitting new models to the residuals of previous predictions. This implementation shows a basic gradient boosting regressor using decision trees.

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
        
        # Initialize predictions with mean value
        F = np.full_like(y, self.initial_prediction, dtype=np.float64)
        
        for _ in range(self.n_estimators):
            # Calculate negative gradients (residuals)
            residuals = y - F
            
            # Fit a new tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            
            self.trees.append(tree)
            
    def predict(self, X):
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction, 
                            dtype=np.float64)
        
        # Add predictions from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions

# Example usage
np.random.seed(42)
X = np.random.randn(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X, y)
predictions = gb.predict(X)

# Calculate MSE
mse = np.mean((predictions - y) ** 2)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 8: Random Forest Implementation

Random Forest is an ensemble learning method that constructs multiple decision trees and outputs the mean prediction of the individual trees. This implementation demonstrates how to build a random forest classifier from scratch.

```python
import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, 
                 n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _get_random_features(self, n_features):
        feature_idxs = np.random.choice(self.n_features_total, 
                                      size=n_features, replace=False)
        return feature_idxs
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features_total = X.shape[1]
        if self.n_features is None:
            self.n_features = int(np.sqrt(self.n_features_total))
            
        # Create trees
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split)
            
            # Get bootstrap samples
            X_sample, y_sample = self._bootstrap_samples(X, y)
            
            # Get random feature subset
            feature_idxs = self._get_random_features(self.n_features)
            
            # Train tree on bootstrap samples with random features
            tree.fit(X_sample[:, feature_idxs], y_sample)
            self.trees.append((tree, feature_idxs))
            
    def predict(self, X):
        tree_predictions = []
        for tree, feature_idxs in self.trees:
            prediction = tree.predict(X[:, feature_idxs])
            tree_predictions.append(prediction)
            
        # Take majority vote
        tree_predictions = np.array(tree_predictions).T
        predictions = [Counter(pred).most_common(1)[0][0] 
                      for pred in tree_predictions]
        return np.array(predictions)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or \
           n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # Find best split
        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # Create child splits
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(best_feat, best_thresh, left, right)
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

# Example usage
X = np.random.randn(100, 4)
y = np.array([0 if np.sum(x) > 0 else 1 for x in X])

rf = RandomForestClassifier(n_trees=10, max_depth=5)
rf.fit(X, y)
predictions = rf.predict(X)
```

Slide 9: K-Nearest Neighbors Implementation

K-Nearest Neighbors is a simple yet powerful algorithm that makes predictions based on the majority class or average value of the k closest training examples. This implementation shows both classification and regression capabilities.

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, weighted=True):
        self.k = k
        self.weighted = weighted
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _get_neighbors(self, x):
        # Calculate distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) 
                    for x_train in self.X_train]
        
        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get corresponding distances
        k_distances = np.array(distances)[k_indices]
        
        # Get labels of k-nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        
        return k_nearest_labels, k_distances
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Get k nearest neighbors
        k_labels, k_distances = self._get_neighbors(x)
        
        # For regression
        if isinstance(self.y_train[0], (int, float, np.integer, np.floating)):
            if self.weighted:
                # Avoid division by zero
                weights = 1 / (k_distances + 1e-10)
                return np.sum(k_labels * weights) / np.sum(weights)
            return np.mean(k_labels)
        
        # For classification
        if self.weighted:
            # Weight votes by inverse distance
            weights = 1 / (k_distances + 1e-10)
            weighted_votes = {}
            for label, weight in zip(k_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            return max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        # Majority voting
        counter = Counter(k_labels)
        return counter.most_common(1)[0][0]

# Example usage - Classification
X = np.random.randn(100, 2)
y = np.array([0 if x[0] + x[1] > 0 else 1 for x in X])

knn_clf = KNN(k=3, weighted=True)
knn_clf.fit(X, y)
clf_predictions = knn_clf.predict(X[:5])

# Example usage - Regression
y_reg = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1

knn_reg = KNN(k=3, weighted=True)
knn_reg.fit(X, y_reg)
reg_predictions = knn_reg.predict(X[:5])

print("Classification predictions:", clf_predictions)
print("Regression predictions:", reg_predictions)
```

Slide 10: Principal Component Analysis Implementation

PCA is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. This implementation shows how to compute PCA from scratch using eigendecomposition.

```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        
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
        
        # Store explained variance
        self.explained_variance = eigenvalues
        
        # Store first n_components eigenvectors
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        self.components = eigenvectors[:, :self.n_components]
        
    def transform(self, X):
        # Center the data
        X_centered = X - self.mean
        
        # Project data onto principal components
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X):
        # Project back to original space
        return np.dot(X, self.components.T) + self.mean
    
    def get_explained_variance_ratio(self):
        return self.explained_variance / np.sum(self.explained_variance)

# Example usage
np.random.seed(42)
X = np.random.randn(100, 5)

# Add some correlation
X[:, 1] = X[:, 0] * 2 + np.random.randn(100) * 0.1
X[:, 2] = X[:, 0] * -0.5 + X[:, 1] * 0.8 + np.random.randn(100) * 0.1

pca = PCA(n_components=2)
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)

# Get explained variance ratio
explained_variance_ratio = pca.get_explained_variance_ratio()

print("Explained variance ratio:", explained_variance_ratio[:2])
print("Transformed data shape:", X_transformed.shape)

# Reconstruct original data
X_reconstructed = pca.inverse_transform(X_transformed)
reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print("Reconstruction error:", reconstruction_error)
```

Slide 11: Neural Network with Backpropagation - Mathematical Foundation

The mathematical foundation of neural networks relies on forward propagation for predictions and backpropagation for learning. This implementation demonstrates the core mathematical concepts using matrix operations and chain rule.

```python
import numpy as np

class NeuralNetworkMath:
    def __init__(self, layer_sizes):
        """
        Mathematical implementation showing detailed computations
        layer_sizes: list of integers representing neurons per layer
        """
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Initialize weights and biases with He initialization
            self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 
                              np.sqrt(2.0/layer_sizes[i]))
            self.biases.append(np.random.randn(layer_sizes[i+1], 1))
            
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def cost_derivative(self, output_activations, y):
        """
        Cost function derivative for MSE
        $$\frac{\partial C}{\partial a} = (a - y)$$
        """
        return output_activations - y
    
    def feedforward(self, a):
        """
        Forward propagation with mathematical notation
        $$a^{l+1} = \sigma(w^l a^l + b^l)$$
        """
        self.zs = []  # List to store all z vectors
        self.activations = [a]  # List to store all activations
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            self.zs.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
        return a
    
    def backprop(self, x, y):
        """
        Backpropagation algorithm implementation
        Returns gradients for weights and biases
        
        Key equations:
        $$\delta^L = \nabla_a C \odot \sigma'(z^L)$$
        $$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$
        $$\frac{\partial C}{\partial w^l} = \delta^l (a^{l-1})^T$$
        $$\frac{\partial C}{\partial b^l} = \delta^l$$
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Forward pass
        output = self.feedforward(x)
        
        # Backward pass
        # Compute delta for output layer
        delta = self.cost_derivative(output, y) * \
                self.sigmoid_prime(self.zs[-1])
        
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].T)
        
        # Compute delta for hidden layers
        for l in range(2, len(self.weights) + 1):
            delta = np.dot(self.weights[-l+1].T, delta) * \
                    self.sigmoid_prime(self.zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activations[-l-1].T)
            
        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Update weights and biases using mini-batch gradient descent
        $$w^l \rightarrow w^l - \frac{\eta}{m} \sum \frac{\partial C}{\partial w^l}$$
        $$b^l \rightarrow b^l - \frac{\eta}{m} \sum \frac{\partial C}{\partial b^l}$$
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        
        self.weights = [w - (learning_rate/len(mini_batch)) * nw 
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate/len(mini_batch)) * nb 
                      for b, nb in zip(self.biases, nabla_b)]

# Example usage
# Create network with 2 inputs, 3 hidden neurons, and 1 output
nn = NeuralNetworkMath([2, 3, 1])

# Training data: XOR problem
X = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
y = np.array([[[0]], [[1]], [[1]], [[0]]])

# Train for a few epochs
for epoch in range(1000):
    for i in range(len(X)):
        nn.update_mini_batch([(X[i], y[i])], learning_rate=0.1)

# Test the network
for x in X:
    prediction = nn.feedforward(x)
    print(f"Input: {x.T}, Prediction: {prediction.T}")
```

Slide 12: Implementation of Convolutional Neural Network Operations

Convolutional Neural Networks are specialized for processing grid-like data. This implementation shows the fundamental operations of convolution and pooling layers from scratch without using deep learning frameworks.

```python
import numpy as np

class CNNOperations:
    @staticmethod
    def conv2d(input_volume, kernel, stride=1, padding=0):
        """
        Implements 2D convolution operation
        
        Parameters:
        - input_volume: shape (height, width, channels)
        - kernel: shape (kernel_height, kernel_width, in_channels, out_channels)
        """
        if padding > 0:
            input_volume = np.pad(
                input_volume,
                ((padding, padding), (padding, padding), (0, 0)),
                mode='constant'
            )
        
        h_in, w_in, c_in = input_volume.shape
        k_h, k_w, _, c_out = kernel.shape
        
        # Calculate output dimensions
        h_out = (h_in - k_h) // stride + 1
        w_out = (w_in - k_w) // stride + 1
        
        # Initialize output volume
        output = np.zeros((h_out, w_out, c_out))
        
        # Perform convolution
        for h in range(h_out):
            for w in range(w_out):
                h_start = h * stride
                w_start = w * stride
                
                # Extract patch from input volume
                patch = input_volume[
                    h_start:h_start+k_h,
                    w_start:w_start+k_w,
                    :
                ]
                
                # Compute convolution for all output channels
                for c in range(c_out):
                    output[h, w, c] = np.sum(
                        patch * kernel[:, :, :, c]
                    )
        
        return output
    
    @staticmethod
    def max_pooling2d(input_volume, pool_size=2, stride=2):
        """
        Implements 2D max pooling operation
        
        Parameters:
        - input_volume: shape (height, width, channels)
        - pool_size: size of pooling window
        - stride: stride of pooling operation
        """
        h_in, w_in, c = input_volume.shape
        
        # Calculate output dimensions
        h_out = (h_in - pool_size) // stride + 1
        w_out = (w_in - pool_size) // stride + 1
        
        # Initialize output volume
        output = np.zeros((h_out, w_out, c))
        
        # Perform max pooling
        for h in range(h_out):
            for w in range(w_out):
                h_start = h * stride
                w_start = w * stride
                
                # Extract patch and compute max for each channel
                patch = input_volume[
                    h_start:h_start+pool_size,
                    w_start:w_start+pool_size,
                    :
                ]
                output[h, w, :] = np.max(np.max(patch, axis=0), axis=0)
        
        return output
    
    @staticmethod
    def relu(x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
# Create sample input volume (6x6 image with 3 channels)
input_volume = np.random.randn(6, 6, 3)

# Create sample kernels (3x3 kernels, 3 input channels, 2 output channels)
kernels = np.random.randn(3, 3, 3, 2)

# Demonstrate convolution operation
conv_output = CNNOperations.conv2d(input_volume, kernels, stride=1, padding=1)
print("Convolution output shape:", conv_output.shape)

# Apply ReLU activation
relu_output = CNNOperations.relu(conv_output)
print("ReLU output shape:", relu_output.shape)

# Apply max pooling
pooling_output = CNNOperations.max_pooling2d(relu_output, pool_size=2, stride=2)
print("Max pooling output shape:", pooling_output.shape)

# Flatten and apply softmax
flattened = pooling_output.reshape(-1)
softmax_output = CNNOperations.softmax(flattened)
print("Softmax output shape:", softmax_output.shape)
```

Slide 13: Autoencoders - Dimensionality Reduction and Feature Learning

Autoencoders are neural networks that learn to compress and reconstruct data. This implementation shows a simple autoencoder with customizable architecture for unsupervised feature learning.

```python
import numpy as np

class Autoencoder:
    def __init__(self, input_dim, encoding_dims, learning_rate=0.01):
        """
        Parameters:
        - input_dim: dimension of input data
        - encoding_dims: list of dimensions for encoder layers
        - learning_rate: learning rate for gradient descent
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Encoder weights
        prev_dim = input_dim
        for dim in encoding_dims:
            self.weights.append(
                np.random.randn(prev_dim, dim) * np.sqrt(2.0/prev_dim)
            )
            self.biases.append(np.zeros((dim, 1)))
            prev_dim = dim
            
        # Decoder weights (mirror of encoder)
        for i in range(len(encoding_dims)-2, -1, -1):
            dim = encoding_dims[i]
            self.weights.append(
                np.random.randn(prev_dim, dim) * np.sqrt(2.0/prev_dim)
            )
            self.biases.append(np.zeros((dim, 1)))
            prev_dim = dim
            
        # Output layer
        self.weights.append(
            np.random.randn(prev_dim, input_dim) * np.sqrt(2.0/prev_dim)
        )
        self.biases.append(np.zeros((input_dim, 1)))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, x):
        """Forward pass through the autoencoder"""
        self.activations = [x]
        self.zs = []
        
        # Forward propagation
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w.T, a) + b
            self.zs.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
            
        return a
    
    def backward(self, x, output):
        """Backward pass to compute gradients"""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Compute output error
        delta = (output - x) * self.sigmoid_derivative(self.zs[-1])
        
        # Backpropagate error
        for l in range(len(self.weights)):
            nabla_b[-l-1] = delta
            nabla_w[-l-1] = np.dot(self.activations[-l-2], delta.T)
            if l < len(self.weights) - 1:
                delta = np.dot(self.weights[-l-1], delta) * \
                        self.sigmoid_derivative(self.zs[-l-2])
                
        return nabla_w, nabla_b
    
    def train_step(self, x):
        """Perform one training step"""
        # Forward pass
        output = self.forward(x)
        
        # Backward pass
        nabla_w, nabla_b = self.backward(x, output)
        
        # Update weights and biases
        self.weights = [w - self.learning_rate * nw 
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - self.learning_rate * nb 
                      for b, nb in zip(self.biases, nabla_b)]
        
        # Return reconstruction error
        return np.mean((output - x) ** 2)

# Example usage
# Create synthetic data
data_dim = 10
n_samples = 1000
data = np.random.randn(data_dim, n_samples)

# Create autoencoder
autoencoder = Autoencoder(
    input_dim=data_dim,
    encoding_dims=[8, 4, 2]  # Compress to 2 dimensions
)

# Train autoencoder
n_epochs = 100
for epoch in range(n_epochs):
    total_error = 0
    for i in range(n_samples):
        x = data[:, i:i+1]
        error = autoencoder.train_step(x)
        total_error += error
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Average Error: {total_error/n_samples:.4f}")

# Encode some data
sample_data = data[:, :5]
encoded_data = autoencoder.forward(sample_data)
print("\nOriginal data shape:", sample_data.shape)
print("Encoded data shape:", encoded_data.shape)
```

Slide 14: Natural Language Processing Basic Implementations

Natural Language Processing involves various techniques for processing and analyzing text data. This implementation shows fundamental NLP operations including tokenization, TF-IDF, and basic text classification.

```python
import numpy as np
from collections import Counter, defaultdict
import re

class NLPToolkit:
    def __init__(self):
        self.vocab = set()
        self.word2idx = {}
        self.idx2word = {}
        self.idf = {}
        
    def tokenize(self, text):
        """Basic tokenization"""
        # Convert to lowercase and split on non-word characters
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def build_vocabulary(self, texts):
        """Build vocabulary from list of texts"""
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
            
        # Create vocabulary (words appearing at least twice)
        self.vocab = {word for word, count in word_counts.items() 
                     if count >= 2}
        
        # Create word-to-index mappings
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def compute_tf(self, text):
        """Compute term frequency"""
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        # Normalize by document length
        total_terms = len(tokens)
        return {term: count/total_terms for term, count in tf.items()}
    
    def compute_idf(self, texts):
        """Compute inverse document frequency"""
        doc_count = len(texts)
        term_doc_count = defaultdict(int)
        
        for text in texts:
            # Count each term only once per document
            terms = set(self.tokenize(text))
            for term in terms:
                term_doc_count[term] += 1
        
        # Compute IDF
        self.idf = {term: np.log(doc_count/(count + 1)) + 1
                   for term, count in term_doc_count.items()}
    
    def compute_tfidf(self, text):
        """Compute TF-IDF for a document"""
        tf = self.compute_tf(text)
        return {term: tf_val * self.idf.get(term, 0)
                for term, tf_val in tf.items()}
    
    def text_to_bow(self, text):
        """Convert text to bag-of-words vector"""
        tokens = self.tokenize(text)
        bow = np.zeros(len(self.vocab))
        for token in tokens:
            if token in self.word2idx:
                bow[self.word2idx[token]] += 1
        return bow
    
    def text_to_tfidf_vector(self, text):
        """Convert text to TF-IDF vector"""
        tfidf = self.compute_tfidf(text)
        vector = np.zeros(len(self.vocab))
        for term, value in tfidf.items():
            if term in self.word2idx:
                vector[self.word2idx[term]] = value
        return vector

class NaiveBayesClassifier:
    def __init__(self, nlp_toolkit):
        self.nlp = nlp_toolkit
        self.class_probs = {}
        self.word_probs = defaultdict(dict)
        
    def train(self, texts, labels):
        """Train Naive Bayes classifier"""
        # Compute class probabilities
        label_counts = Counter(labels)
        total_docs = len(labels)
        self.class_probs = {label: count/total_docs 
                           for label, count in label_counts.items()}
        
        # Compute word probabilities per class
        word_counts = defaultdict(Counter)
        for text, label in zip(texts, labels):
            tokens = self.nlp.tokenize(text)
            word_counts[label].update(tokens)
            
        # Compute word probabilities with Laplace smoothing
        vocab_size = len(self.nlp.vocab)
        for label in self.class_probs:
            total_words = sum(word_counts[label].values())
            for word in self.nlp.vocab:
                count = word_counts[label][word]
                # Add-one smoothing
                prob = (count + 1) / (total_words + vocab_size)
                self.word_probs[label][word] = prob
    
    def predict(self, text):
        """Predict class for text"""
        tokens = self.nlp.tokenize(text)
        scores = {}
        
        for label in self.class_probs:
            # Start with log of class probability
            score = np.log(self.class_probs[label])
            
            # Add log probabilities of words
            for token in tokens:
                if token in self.nlp.vocab:
                    score += np.log(self.word_probs[label][token])
            
            scores[label] = score
            
        # Return label with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

# Example usage
# Sample texts and labels
texts = [
    "machine learning is fascinating",
    "deep neural networks are powerful",
    "natural language processing with python",
    "statistical analysis and data science",
    "artificial intelligence and robotics"
]
labels = ["ML", "DL", "NLP", "Stats", "AI"]

# Initialize NLP toolkit
nlp = NLPToolkit()
nlp.build_vocabulary(texts)
nlp.compute_idf(texts)

# Create and train classifier
classifier = NaiveBayesClassifier(nlp)
classifier.train(texts, labels)

# Test classification
test_text = "learning neural networks with python"
prediction = classifier.predict(test_text)
print(f"Predicted class for '{test_text}': {prediction}")

# Show TF-IDF vector for a document
tfidf_vector = nlp.text_to_tfidf_vector(test_text)
print("\nTF-IDF vector shape:", tfidf_vector.shape)
print("Non-zero terms:")
for idx, value in enumerate(tfidf_vector):
    if value > 0:
        print(f"{nlp.idx2word[idx]}: {value:.4f}")
```

Slide 15: Additional Resources

*   "Deep Learning" by Goodfellow, Bengio, and Courville
    *   [https://arxiv.org/abs/1521.00561](https://arxiv.org/abs/1521.00561)
*   "Machine Learning: A Probabilistic Perspective" by Kevin Murphy
    *   Search for: "Murphy ML Probabilistic Perspective PDF"
*   "Natural Language Processing with Deep Learning"
    *   [https://arxiv.org/abs/1703.03091](https://arxiv.org/abs/1703.03091)
*   "A Tutorial on Support Vector Machines for Pattern Recognition"
    *   [https://arxiv.org/abs/1412.3321](https://arxiv.org/abs/1412.3321)
*   "Random Forests in Machine Learning"
    *   Search for: "Breiman Random Forests Statistics PDF"
*   "An Introduction to Statistical Learning"
    *   [www.statlearning.com](http://www.statlearning.com)

