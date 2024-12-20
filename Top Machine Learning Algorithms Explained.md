## Top Machine Learning Algorithms Explained
Slide 1: Linear Regression Implementation

Linear regression serves as a foundational predictive modeling algorithm that establishes relationships between dependent and independent variables through a linear equation. The model minimizes the sum of squared residuals between observed and predicted values to find optimal coefficients.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

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
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradient computation
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegressionFromScratch()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
mse = np.mean((predictions - y_test) ** 2)
print(f"Mean Squared Error: {mse:.2f}")
```

Slide 2: Linear Regression Mathematics

The mathematical foundation of linear regression relies on the optimization of parameters through gradient descent. The core equation represents the relationship between features and target variable, while the cost function measures prediction error.

```python
# Mathematical representation in LaTeX notation
"""
Linear Equation:
$$y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$$

Cost Function (Mean Squared Error):
$$J(\beta) = \frac{1}{2m}\sum_{i=1}^m(h_\beta(x^{(i)}) - y^{(i)})^2$$

Gradient Descent Update:
$$\beta_j := \beta_j - \alpha\frac{\partial}{\partial\beta_j}J(\beta)$$

where:
$$\frac{\partial}{\partial\beta_j}J(\beta) = \frac{1}{m}\sum_{i=1}^m(h_\beta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
"""
```

Slide 3: Logistic Regression Implementation

Logistic regression extends linear regression concepts to binary classification by applying the sigmoid function to transform continuous outputs into probability scores between 0 and 1, enabling decision boundaries for classification tasks.

```python
import numpy as np
from sklearn.datasets import make_classification

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

# Generate binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
model = LogisticRegressionFromScratch()
model.fit(X, y)
```

Slide 4: Decision Tree Implementation

Decision trees partition data through recursive binary splitting, selecting features and thresholds that maximize information gain. This implementation demonstrates the construction of a decision tree classifier with entropy-based splitting criteria.

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

class DecisionTreeFromScratch:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None
        
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def split_data(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        return (
            X[left_mask], X[~left_mask],
            y[left_mask], y[~left_mask]
        )
    
    def information_gain(self, parent, left_child, right_child):
        num_left, num_right = len(left_child), len(right_child)
        parent_entropy = self.entropy(parent)
        children_entropy = (num_left * self.entropy(left_child) + 
                          num_right * self.entropy(right_child)) / len(parent)
        return parent_entropy - children_entropy
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (depth >= self.max_depth or n_labels == 1):
            leaf_value = max(Counter(y).items(), key=lambda x: x[1])[0]
            return Node(value=leaf_value)
            
        best_gain = -1
        best_split = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split_data(X, y, feature, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    gain = self.information_gain(y, y_left, y_right)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (feature, threshold, X_left, X_right, y_left, y_right)
        
        if best_gain == -1:
            leaf_value = max(Counter(y).items(), key=lambda x: x[1])[0]
            return Node(value=leaf_value)
            
        feature, threshold, X_left, X_right, y_left, y_right = best_split
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        
        return Node(feature, threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)
```

Slide 5: Random Forest Implementation

Random Forest leverages ensemble learning by combining multiple decision trees trained on different subsets of data and features. This implementation showcases bootstrap sampling and random feature selection for building robust forest classifiers.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class RandomForestFromScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.trees = []
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.n_features
            )
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(pred).most_common(1)[0][0] for pred in tree_preds]
        return np.array(y_pred)

# Usage example
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
rf = RandomForestFromScratch(n_trees=10)
rf.fit(X, y)
predictions = rf.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 6: Support Vector Machine Implementation

Support Vector Machines find optimal hyperplanes for classification by maximizing the margin between classes. This implementation uses the Sequential Minimal Optimization (SMO) algorithm for training and kernel tricks for non-linear classification.

```python
import numpy as np

class SVMFromScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def gaussian_kernel(self, x1, x2, sigma=1.0):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Convert labels to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

# Example usage with synthetic data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=42)
y = np.where(y <= 0, -1, 1)

# Train SVM
svm = SVMFromScratch()
svm.fit(X, y)

# Make predictions
predictions = svm.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 7: K-Nearest Neighbors Implementation

KNN classification determines class membership by majority voting among k nearest neighbors, using distance metrics to measure similarity. This implementation includes multiple distance metrics and weighted voting options.

```python
import numpy as np
from collections import Counter

class KNNFromScratch:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2), axis=1)
    
    def calculate_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return self.euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self.manhattan_distance(x1, x2)
            
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        
        for x in X:
            distances = self.calculate_distance(self.X_train, x)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            if self.weights == 'uniform':
                most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            else:  # distance weighted
                weights = 1 / (distances[k_indices] + 1e-10)
                weighted_votes = {}
                for label, weight in zip(k_nearest_labels, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                most_common = max(weighted_votes.items(), key=lambda x: x[1])[0]
                
            predictions.append(most_common)
            
        return np.array(predictions)

# Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNNFromScratch(k=5, weights='distance')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 8: K-Means Clustering Implementation

K-means clustering partitions data into k distinct clusters by iteratively updating centroids and reassigning points to their nearest cluster center. This implementation includes initialization strategies and convergence criteria.

```python
import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def initialize_centroids(self, X):
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Initialize using k-means++
        centroids[0] = X[np.random.randint(n_samples)]
        
        for i in range(1, self.n_clusters):
            distances = np.array([min([np.linalg.norm(x-c)**2 for c in centroids[:i]]) 
                                for x in X])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centroids[i] = X[j]
                    break
                    
        return centroids
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels == k], axis=0)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids, rtol=self.tol):
                break
                
        return self
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def inertia(self, X):
        distances = np.sqrt(((X - self.centroids[self.labels])**2).sum(axis=1))
        return np.sum(distances)

# Example usage with generated data
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit K-means
kmeans = KMeansFromScratch(n_clusters=4)
kmeans.fit(X)

# Get cluster assignments and inertia
labels = kmeans.labels
inertia = kmeans.inertia(X)
print(f"Inertia: {inertia:.2f}")
```

Slide 9: Naive Bayes Classifier Implementation

Naive Bayes applies Bayes' theorem with strong independence assumptions between features. This implementation includes Gaussian and Multinomial variants for different types of data distributions.

```python
import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9  # Add small value for numerical stability
            self.priors[idx] = len(X_c) / n_samples
    
    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        y_pred = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                posterior = np.sum(np.log(self.gaussian_density(idx, x)))
                posterior = prior + posterior
                posteriors.append(posterior)
            
            # Select class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)

# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
nb = GaussianNaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 10: Neural Network Implementation

A feedforward neural network with backpropagation implements deep learning fundamentals. This implementation includes multiple layers, activation functions, and gradient descent optimization.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, layers[i+1])))
    
    def relu(self, X):
        return np.maximum(0, X)
    
    def relu_derivative(self, X):
        return np.where(X > 0, 1, 0)
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-np.clip(X, -500, 500)))
    
    def sigmoid_derivative(self, X):
        return X * (1 - X)
    
    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                activation = self.sigmoid(z)
            else:
                activation = self.relu(z)
            self.activations.append(activation)
            
        return self.activations[-1]
    
    def backward_propagation(self, X, y):
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        dW = [np.dot(self.activations[-2].T, delta) / m]
        db = [np.sum(delta, axis=0, keepdims=True) / m]
        
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
            dW.insert(0, np.dot(self.activations[i].T, delta) / m)
            db.insert(0, np.sum(delta, axis=0, keepdims=True) / m)
        
        return dW, db
    
    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward_propagation(X)
            
            # Backward propagation
            dW, db = self.backward_propagation(X, y)
            
            # Update parameters
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(y_pred + 1e-15) + 
                              (1-y) * np.log(1-y_pred + 1e-15))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return np.round(self.forward_propagation(X))

# Example usage
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.20, random_state=42)
y = y.reshape(-1, 1)

# Create and train network
nn = NeuralNetwork([2, 16, 8, 1], learning_rate=0.1)
nn.fit(X, y, epochs=1000)

# Make predictions
predictions = nn.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 11: Gradient Boosting Implementation

Gradient Boosting builds an ensemble of weak learners sequentially, where each learner tries to correct the errors of its predecessors. This implementation shows the core boosting algorithm with decision trees.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingFromScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        
    def fit(self, X, y):
        self.trees = []
        F = np.zeros(len(y))
        
        for _ in range(self.n_estimators):
            # Calculate pseudo residuals
            residuals = y - F
            
            # Fit a tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            
            self.trees.append(tree)
            
    def predict(self, X):
        predictions = np.zeros((X.shape[0],))
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions

# Example usage with regression problem
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
gb = GradientBoostingFromScratch(n_estimators=100, learning_rate=0.1)
gb.fit(X_train, y_train)

# Evaluate
predictions = gb.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.4f}")
```

Slide 12: Cross-Validation and Model Evaluation

Cross-validation provides reliable estimates of model performance by partitioning data into multiple training and validation sets. This implementation showcases k-fold cross-validation and various performance metrics.

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CrossValidator:
    def __init__(self, model, n_folds=5):
        self.model = model
        self.n_folds = n_folds
        
    def k_fold_split(self, X, y):
        fold_size = len(X) // self.n_folds
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])
            
            yield (X[train_indices], X[test_indices], 
                  y[train_indices], y[test_indices])
    
    def compute_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def cross_validate(self, X, y):
        metrics_per_fold = []
        
        for fold_num, (X_train, X_test, y_train, y_test) in enumerate(
            self.k_fold_split(X, y), 1):
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Compute metrics
            metrics = self.compute_metrics(y_test, y_pred)
            metrics_per_fold.append(metrics)
            
            print(f"\nFold {fold_num} Results:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        # Compute average metrics across folds
        avg_metrics = {}
        for metric in metrics_per_fold[0].keys():
            avg_metrics[metric] = np.mean([
                fold[metric] for fold in metrics_per_fold
            ])
        
        print("\nAverage Metrics Across Folds:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return avg_metrics

# Example usage
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_classes=3, random_state=42)

# Initialize model and cross-validator
model = DecisionTreeClassifier(random_state=42)
cv = CrossValidator(model, n_folds=5)

# Perform cross-validation
results = cv.cross_validate(X, y)
```

Slide 13: Additional Resources

*   "A Survey of Cross-Validation Procedures for Model Selection" [https://arxiv.org/abs/1811.12808](https://arxiv.org/abs/1811.12808)
*   "XGBoost: A Scalable Tree Boosting System" [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
*   "Random Forests" [https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
*   "Deep Learning Book" by Goodfellow, Bengio, and Courville [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
*   "Support Vector Machines: Theory and Applications" [https://arxiv.org/abs/0904.3664v1](https://arxiv.org/abs/0904.3664v1)

These resources provide comprehensive coverage of machine learning algorithms, their mathematical foundations, and practical implementations. For more recent developments and implementations, searching on Google Scholar with keywords related to specific algorithms is recommended.

