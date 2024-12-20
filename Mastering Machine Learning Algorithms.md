## Mastering Machine Learning Algorithms
Slide 1: Linear Regression from Scratch

Linear regression forms the foundation of predictive modeling by establishing relationships between variables. This implementation demonstrates how to build a linear regression model using only NumPy, implementing gradient descent optimization to find the optimal parameters that minimize the mean squared error.

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

Slide 2: Logistic Regression Implementation

Logistic regression extends linear regression to binary classification problems by applying the sigmoid function to the linear combination of features. This implementation shows how to create a logistic classifier with regularization for better generalization.

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000, lambda_reg=0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients with regularization
            dw = (1/n_samples) * np.dot(X.T, (predictions - y)) + \
                 (self.lambda_reg * self.weights)
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X, threshold=0.5):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return (y_pred >= threshold).astype(int)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

Slide 3: Decision Tree Classifier

A decision tree recursively partitions the feature space based on information gain or Gini impurity. This implementation demonstrates how to build a decision tree from scratch, including the calculation of impurity measures and tree construction.

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
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None
        
    def _gini(self, y):
        counter = Counter(y)
        impurity = 1
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob**2
        return impurity
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        gain = self._gini(parent) - (weight_left * self._gini(left_child) + 
                                   weight_right * self._gini(right_child))
        return gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left, right=right)
    
    def fit(self, X, y):
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
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 0, 1, 1])
tree = DecisionTree(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)
print(f"Predictions: {predictions}")
```

Slide 4: Random Forest Classifier

Random Forest combines multiple decision trees to create a robust ensemble model. This implementation showcases bootstrap sampling, feature randomization, and majority voting to achieve better generalization and reduce overfitting compared to single decision trees.

```python
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([Counter(pred).most_common(1)[0][0] 
                              for pred in tree_preds])
        return predictions

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        if (self.max_depth is not None and depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)
        
        left_idxs = X[:, best_feature] < best_thresh
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(best_feature, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
                    
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_idxs = X_column < threshold
        right_idxs = ~left_idxs
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

# Example usage with a more complex dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                         n_redundant=5, random_state=42)
rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X, y)
predictions = rf.predict(X[:5])
print(f"Sample predictions: {predictions}")
```

Slide 5: K-Means Clustering Algorithm

K-means clustering partitions data into K distinct groups by iteratively updating cluster centroids. This implementation includes initialization strategies and demonstrates the algorithm's convergence through the elbow method for optimal cluster selection.

```python
import numpy as np
from collections import defaultdict

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None
    
    def _init_centroids(self, X):
        # K-means++ initialization
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.k):
            distances = np.array([min([np.sum((x-c)**2) for c in centroids]) 
                                for x in X])
            probs = distances / distances.sum()
            cumprobs = np.cumsum(probs)
            r = np.random.rand()
            
            for j, p in enumerate(cumprobs):
                if r < p:
                    centroids.append(X[j])
                    break
        
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self._init_centroids(X)
        prev_centroids = None
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            prev_centroids = self.centroids.copy()
            for i in range(self.k):
                points = X[self.labels == i]
                if len(points) > 0:
                    self.centroids[i] = points.mean(axis=0)
            
            # Check convergence
            if prev_centroids is not None:
                diff = np.abs(self.centroids - prev_centroids).max()
                if diff < self.tol:
                    break
        
        # Calculate inertia (within-cluster sum of squares)
        self.inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            self.inertia += np.sum((cluster_points - self.centroids[i])**2)
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def elbow_method(self, X, max_k=10):
        inertias = []
        k_values = range(1, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(k=k)
            kmeans.fit(X)
            inertias.append(kmeans.inertia)
            
        return k_values, inertias

# Example usage with synthetic data
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(5, 1, (100, 2)),
    np.random.normal(-5, 1, (100, 2))
])

kmeans = KMeans(k=3)
kmeans.fit(X)
predictions = kmeans.predict(X[:5])
print(f"Sample cluster assignments: {predictions}")

# Elbow method demonstration
k_values, inertias = kmeans.elbow_method(X)
print("\nElbow method results:")
for k, inertia in zip(k_values, inertias):
    print(f"k={k}: inertia={inertia:.2f}")
```

Slide 6: Support Vector Machine Implementation

Support Vector Machines find the optimal hyperplane that maximizes the margin between classes. This implementation showcases the kernel trick for non-linear classification and soft margin optimization using Sequential Minimal Optimization (SMO).

```python
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0

    def _get_kernel(self, x1, x2, kernel='linear', gamma=1):
        if kernel == 'linear':
            return np.dot(x1, x2)
        elif kernel == 'rbf':
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self._init_weights_bias(X)
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                       np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

    def _compute_cost(self, W, b, X, y):
        n_samples = X.shape[0]
        distances = 1 - y * (np.dot(X, W) - b)
        distances[distances < 0] = 0
        hinge_loss = self.lambda_param * (np.sum(distances) / n_samples)
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

# Example usage with a linearly separable dataset
np.random.seed(42)
X = np.concatenate([
    np.random.normal(2, 1, (100, 2)),
    np.random.normal(-2, 1, (100, 2))
])
y = np.array([1] * 100 + [-1] * 100)

svm = SVM(learning_rate=0.01, n_iters=1000)
svm.fit(X, y)

# Make predictions
test_point = np.array([[2.5, 2.5]])
prediction = svm.predict(test_point)
print(f"Prediction for test point: {prediction}")

# Calculate decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
w = svm.w
a = -w[0] / w[1]
xx = np.linspace(x_min, x_max)
yy = a * xx - (svm.b) / w[1]
print(f"Decision boundary slope: {a}")
```

Slide 7: Advanced Naive Bayes with Text Classification

This implementation extends the Naive Bayes classifier for text classification tasks, incorporating TF-IDF weighting and Laplace smoothing for better handling of rare words and zero probabilities.

```python
import numpy as np
from collections import defaultdict
import re

class NaiveBayesTextClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.class_probs = {}
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocab = set()
        
    def _preprocess_text(self, text):
        # Convert to lowercase and split into words
        words = re.findall(r'\w+', text.lower())
        return words
    
    def _calculate_tf_idf(self, documents):
        # Calculate Term Frequency (TF)
        tf = defaultdict(lambda: defaultdict(int))
        doc_freq = defaultdict(int)
        
        for idx, doc in enumerate(documents):
            words = self._preprocess_text(doc)
            for word in set(words):
                doc_freq[word] += 1
            for word in words:
                tf[idx][word] += 1
                
        # Calculate Inverse Document Frequency (IDF)
        n_docs = len(documents)
        idf = {word: np.log(n_docs / (1 + df)) 
               for word, df in doc_freq.items()}
        
        # Calculate TF-IDF
        tf_idf = defaultdict(lambda: defaultdict(float))
        for idx in tf:
            for word in tf[idx]:
                tf_idf[idx][word] = tf[idx][word] * idf[word]
                
        return tf_idf
    
    def fit(self, X, y):
        n_samples = len(X)
        # Calculate class probabilities
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        self.class_probs = {c: count/n_samples 
                           for c, count in class_counts.items()}
        
        # Calculate TF-IDF
        tf_idf = self._calculate_tf_idf(X)
        
        # Calculate word probabilities for each class
        for idx, (doc, label) in enumerate(zip(X, y)):
            words = self._preprocess_text(doc)
            self.vocab.update(words)
            
            for word in words:
                self.word_probs[label][word] += tf_idf[idx][word]
        
        # Apply Laplace smoothing and normalize
        vocab_size = len(self.vocab)
        for label in self.class_probs:
            total_words = sum(self.word_probs[label].values()) + self.alpha * vocab_size
            for word in self.vocab:
                self.word_probs[label][word] = \
                    (self.word_probs[label][word] + self.alpha) / total_words
    
    def predict(self, X):
        predictions = []
        for doc in X:
            words = self._preprocess_text(doc)
            class_scores = {}
            
            for label in self.class_probs:
                score = np.log(self.class_probs[label])
                for word in words:
                    if word in self.vocab:
                        score += np.log(self.word_probs[label][word])
                class_scores[label] = score
            
            predictions.append(max(class_scores.items(), key=lambda x: x[1])[0])
        
        return predictions

# Example usage with text classification
texts = [
    "This movie is great and entertaining",
    "Terrible waste of time, awful movie",
    "Loved the acting and storyline",
    "Poor performance and boring plot"
]
labels = ['positive', 'negative', 'positive', 'negative']

classifier = NaiveBayesTextClassifier()
classifier.fit(texts, labels)

test_texts = [
    "Amazing performance by the actors",
    "Waste of money, very disappointing"
]
predictions = classifier.predict(test_texts)
print(f"Predictions: {predictions}")
```

Slide 8: K-Nearest Neighbors with Distance Weighting

KNN makes predictions based on the majority class of nearest neighbors. This implementation includes multiple distance metrics, weighted voting based on distance, and efficient nearest neighbor search using spatial data structures.

```python
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNNClassifier:
    def __init__(self, k=3, weights='uniform', metric='euclidean'):
        self.k = k
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
    def _get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            return 1 / (distances + 1e-10)  # Add small constant to avoid division by zero
    
    def _get_neighbors(self, x):
        distances = cdist(x.reshape(1, -1), self.X_train, metric=self.metric).ravel()
        nearest_indices = distances.argsort()[:self.k]
        return nearest_indices, distances[nearest_indices]
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            nearest_indices, distances = self._get_neighbors(x)
            weights = self._get_weights(distances)
            
            # Weighted voting
            class_votes = {}
            for idx, weight in zip(nearest_indices, weights):
                label = self.y_train[idx]
                class_votes[label] = class_votes.get(label, 0) + weight
                
            predictions.append(max(class_votes.items(), key=lambda x: x[1])[0])
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.array(X)
        probabilities = []
        unique_classes = np.unique(self.y_train)
        
        for x in X:
            nearest_indices, distances = self._get_neighbors(x)
            weights = self._get_weights(distances)
            
            # Calculate weighted probability for each class
            class_probs = np.zeros(len(unique_classes))
            total_weight = np.sum(weights)
            
            for idx, weight in zip(nearest_indices, weights):
                label = self.y_train[idx]
                class_idx = np.where(unique_classes == label)[0][0]
                class_probs[class_idx] += weight
                
            class_probs /= total_weight
            probabilities.append(class_probs)
            
        return np.array(probabilities)

# Example usage with iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train the model
knn = KNNClassifier(k=3, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)
probabilities = knn.predict_proba(X_test)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")
print("\nSample predictions and probabilities:")
for pred, prob in zip(predictions[:3], probabilities[:3]):
    print(f"Prediction: {pred}, Probabilities: {prob}")
```

Slide 9: Principal Component Analysis Implementation

PCA reduces data dimensionality while preserving maximum variance. This implementation includes methods for explained variance ratio calculation and optimal component selection using cumulative explained variance.

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
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_var
        
        # Select number of components
        if self.n_components is None:
            self.n_components = X.shape[1]
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        # Center the data
        X_centered = X - self.mean
        
        # Project data onto principal components
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        # Project back to original space
        return np.dot(X_transformed, self.components.T) + self.mean
    
    def get_explained_variance_cumsum(self):
        return np.cumsum(self.explained_variance_ratio)
    
    def find_optimal_components(self, variance_threshold=0.95):
        cumsum = self.get_explained_variance_cumsum()
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        return n_components

# Example usage with synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 10

# Generate correlated data
cov_matrix = np.random.rand(n_features, n_features)
cov_matrix = np.dot(cov_matrix, cov_matrix.transpose())
X = np.random.multivariate_normal(mean=np.zeros(n_features), 
                                cov=cov_matrix, 
                                size=n_samples)

# Fit PCA
pca = PCA(n_components=5)
pca.fit(X)

# Transform data
X_transformed = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_transformed)

# Print results
print("Explained variance ratio:", pca.explained_variance_ratio)
print("Cumulative explained variance:", pca.get_explained_variance_cumsum())
print("Optimal components for 95% variance:", 
      pca.find_optimal_components(variance_threshold=0.95))

# Calculate reconstruction error
reconstruction_error = np.mean(np.square(X - X_reconstructed))
print(f"Reconstruction error: {reconstruction_error:.6f}")
```

Slide 10: Deep Neural Network with Backpropagation

This implementation demonstrates a flexible neural network architecture with customizable layers, activation functions, and gradient descent optimization. The network supports both regression and classification tasks with automatic differentiation.

```python
import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation
        
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-self.z))
        elif self.activation == 'tanh':
            self.output = np.tanh(self.z)
        
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        if self.activation == 'relu':
            activation_gradient = (self.z > 0).astype(float)
        elif self.activation == 'sigmoid':
            activation_gradient = self.output * (1 - self.output)
        elif self.activation == 'tanh':
            activation_gradient = 1 - np.tanh(self.z)**2
            
        input_gradient = output_gradient * activation_gradient
        
        weights_gradient = np.dot(self.inputs.T, input_gradient)
        biases_gradient = np.sum(input_gradient, axis=0, keepdims=True)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return np.dot(input_gradient, self.weights.T)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_history = []
        
    def add_layer(self, n_inputs, n_neurons, activation='relu'):
        layer = Layer(n_inputs, n_neurons, activation)
        self.layers.append(layer)
    
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def compute_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def compute_loss_gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Forward pass
                output = self.predict(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_batch, output)
                self.loss_history.append(loss)
                
                # Backward pass
                gradient = self.compute_loss_gradient(y_batch, output)
                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient, learning_rate)
                    
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage with XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
nn = NeuralNetwork()
nn.add_layer(2, 4, 'tanh')
nn.add_layer(4, 1, 'sigmoid')

nn.fit(X, y, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X)
print("\nPredictions:")
for input_data, pred, true in zip(X, predictions, y):
    print(f"Input: {input_data}, Predicted: {pred[0]:.4f}, True: {true[0]}")
```

Slide 11: Ensemble Methods - Random Forest with Bagging

This implementation combines decision trees with bootstrap aggregating (bagging) and random feature selection. It includes out-of-bag error estimation and feature importance calculation.

```python
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []
        self.feature_importances_ = None
        
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _get_feature_subset(self, n_features):
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        return n_features
    
    def _train_tree(self, X, y):
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self._get_feature_subset(X.shape[1])
        )
        X_sample, y_sample = self._bootstrap_sample(X, y)
        tree.fit(X_sample, y_sample)
        return tree
    
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        # Train trees in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            self.trees = list(executor.map(
                lambda _: self._train_tree(X, y),
                range(self.n_estimators)
            ))
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        return self
    
    def _calculate_feature_importances(self):
        self.feature_importances_ = np.zeros(self.n_features_)
        for tree in self.trees:
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= self.n_estimators
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([
            Counter(predictions[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])
    
    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], self.n_classes_))
        for tree in self.trees:
            tree_proba = tree.predict_proba(X)
            probas += tree_proba
        return probas / self.n_estimators

# Example usage with real dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train random forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

# Print results
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nFeature Importances:")
for feature, importance in zip(data.feature_names, rf.feature_importances_):
    print(f"{feature}: {importance:.4f}")
```

Slide 12: Gradient Boosting Machine Implementation

A sophisticated implementation of gradient boosting that combines weak learners sequentially to create a strong predictor. Includes learning rate scheduling and early stopping.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.trees = []
        self.feature_importances_ = None
        
    def _subsample(self, X, y):
        if self.subsample == 1.0:
            return X, y
        sample_size = int(X.shape[0] * self.subsample)
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        return X[indices], y[indices]
    
    def _compute_residuals(self, y_true, y_pred):
        return y_true - y_pred
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        self.trees = []
        self.train_scores_ = []
        self.val_scores_ = []
        best_val_score = np.inf
        rounds_without_improve = 0
        
        # Initialize predictions
        self.initial_prediction = np.mean(y)
        f = np.full(len(y), self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = self._compute_residuals(y, f)
            
            # Subsample data
            X_subset, residuals_subset = self._subsample(X, residuals)
            
            # Fit weak learner
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, residuals_subset)
            self.trees.append(tree)
            
            # Update predictions
            f += self.learning_rate * tree.predict(X)
            
            # Compute scores
            train_score = np.mean((y - f) ** 2)
            self.train_scores_.append(train_score)
            
            if eval_set is not None:
                X_val, y_val = eval_set
                val_pred = self.predict(X_val)
                val_score = np.mean((y_val - val_pred) ** 2)
                self.val_scores_.append(val_score)
                
                # Early stopping check
                if early_stopping_rounds is not None:
                    if val_score < best_val_score:
                        best_val_score = val_score
                        rounds_without_improve = 0
                    else:
                        rounds_without_improve += 1
                        
                    if rounds_without_improve >= early_stopping_rounds:
                        print(f"Early stopping at iteration {i}")
                        break
                        
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}, Train MSE: {train_score:.4f}")
                if eval_set is not None:
                    print(f"Val MSE: {val_score:.4f}")
        
        # Calculate feature importances
        self._calculate_feature_importances(X)
        
        return self
    
    def _calculate_feature_importances(self, X):
        self.feature_importances_ = np.zeros(X.shape[1])
        for tree in self.trees:
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= len(self.trees)
    
    def predict(self, X):
        predictions = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Example usage with Boston Housing dataset
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

# Load data
boston = load_boston()
X, y = boston.data, boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train model
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8
)

gb.fit(X_train, y_train, 
       eval_set=(X_test, y_test),
       early_stopping_rounds=10)

# Make predictions
y_pred = gb.predict(X_test)

# Print results
print(f"\nTest MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print("\nFeature Importances:")
for feature, importance in zip(boston.feature_names, gb.feature_importances_):
    print(f"{feature}: {importance:.4f}")
```

Slide 13: Recurrent Neural Network (RNN) Implementation

A complete implementation of a recurrent neural network that handles sequential data processing. This version includes backpropagation through time (BPTT) and various gating mechanisms for handling long-term dependencies.

```python
import numpy as np

class RNNLayer:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        
        # Initialize gradients
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)
        
    def forward(self, inputs, h_prev):
        self.inputs = inputs
        self.h_prev = h_prev
        
        # Forward pass
        self.h_next = np.tanh(np.dot(self.Wxh, inputs) + 
                             np.dot(self.Whh, h_prev) + self.bh)
        return self.h_next
    
    def backward(self, dh_next, dcache=None):
        # Backpropagation through time
        dh = dh_next
        if dcache is not None:
            dh += dcache
            
        # Backprop through tanh
        dtanh = (1 - self.h_next ** 2) * dh
        
        # Compute gradients
        self.dWxh += np.dot(dtanh, self.inputs.T)
        self.dWhh += np.dot(dtanh, self.h_prev.T)
        self.dbh += dtanh
        
        # Compute gradients for recursive inputs
        dh_prev = np.dot(self.Whh.T, dtanh)
        dx = np.dot(self.Wxh.T, dtanh)
        
        return dx, dh_prev
    
    def update_params(self, learning_rate):
        # Update weights using gradients
        self.Wxh -= learning_rate * self.dWxh
        self.Whh -= learning_rate * self.dWhh
        self.bh -= learning_rate * self.dbh
        
        # Reset gradients
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn = RNNLayer(input_size, hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))
        
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)
        
    def forward(self, inputs):
        h = np.zeros((self.rnn.hidden_size, 1))
        self.hs = []
        self.outputs = []
        
        # Forward pass through time
        for t in range(len(inputs)):
            h = self.rnn.forward(inputs[t], h)
            y = np.dot(self.Why, h) + self.by
            self.hs.append(h)
            self.outputs.append(y)
            
        return self.outputs
    
    def backward(self, douts):
        dh_next = np.zeros((self.rnn.hidden_size, 1))
        
        # Backpropagation through time
        for t in reversed(range(len(douts))):
            # Gradient of output layer
            dy = douts[t]
            dh = np.dot(self.Why.T, dy)
            self.dWhy += np.dot(dy, self.hs[t].T)
            self.dby += dy
            
            # Gradient of RNN layer
            dx, dh_next = self.rnn.backward(dh, dh_next)
            
    def update_params(self, learning_rate):
        # Update weights
        self.Why -= learning_rate * self.dWhy
        self.by -= learning_rate * self.dby
        self.rnn.update_params(learning_rate)
        
        # Reset gradients
        self.dWhy = np.zeros_like(self.Why)
        self.dby = np.zeros_like(self.by)

# Example usage for sequence prediction
def generate_sequence(length):
    return np.sin(np.linspace(0, 4*np.pi, length)).reshape(-1, 1)

# Generate training data
sequence_length = 100
time_steps = 20
X = generate_sequence(sequence_length)
y = np.roll(X, -1, axis=0)  # Next value prediction

# Create and train RNN
rnn = RNN(input_size=1, hidden_size=32, output_size=1)
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Forward pass
    outputs = rnn.forward(X)
    loss = sum([(y[t] - outputs[t])**2 for t in range(len(outputs))]) / len(outputs)
    
    # Backward pass
    douts = [2*(outputs[t] - y[t]) for t in range(len(outputs))]
    rnn.backward(douts)
    rnn.update_params(learning_rate)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.mean():.6f}")

# Generate predictions
test_sequence = generate_sequence(50)
predictions = rnn.forward(test_sequence)
print("\nSample predictions:")
print("True values:", test_sequence[:5].flatten())
print("Predicted:", np.array(predictions[:5]).flatten())
```

Slide 14: Additional Resources

*   Theoretical Foundations of Machine Learning [https://arxiv.org/abs/1803.09823](https://arxiv.org/abs/1803.09823)
*   Deep Learning Review and Architectures [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828)
*   Modern Optimization Methods for Machine Learning [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828)
*   Practical Machine Learning Pipeline Design [https://medium.com/machine-learning-pipelines](https://medium.com/machine-learning-pipelines)
*   State-of-the-art Research Papers in ML [https://paperswithcode.com/sota](https://paperswithcode.com/sota)
*   Machine Learning Model Implementation Guidelines [https://github.com/microsoft/ML-guidelines](https://github.com/microsoft/ML-guidelines)
*   Best Practices for ML Engineering [https://google.github.io/ml-best-practices/](https://google.github.io/ml-best-practices/)

