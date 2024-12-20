## Top 10 Machine Learning Algorithms
Slide 1: Introduction to Naïve Bayes Classifier

The Naïve Bayes classifier implements Bayes' theorem with strong independence assumptions between features. Despite its simplicity, it often performs remarkably well in real-world situations, particularly in document classification and spam filtering. The algorithm excels in high-dimensional training datasets.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.parameters = {}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate parameters for each class
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / n_samples
            }
    
    def _calculate_likelihood(self, x, mean, var):
        epsilon = 1e-10
        exponent = np.exp(-((x-mean)**2)/(2*var + epsilon))
        return np.prod(1/np.sqrt(2*np.pi*var + epsilon) * exponent, axis=1)
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = self.parameters[c]['prior']
                likelihood = self._calculate_likelihood(x, 
                                                     self.parameters[c]['mean'],
                                                     self.parameters[c]['var'])
                posterior = prior * likelihood
                posteriors.append(posterior)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)

# Example usage
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 2: Mathematical Foundation of Naïve Bayes

The foundation of Naïve Bayes lies in probabilistic theory, specifically Bayes' theorem. The algorithm calculates the posterior probability of each class given the features, assuming conditional independence between features. This mathematical basis enables efficient classification across various domains.

```python
"""
The Naïve Bayes classifier is based on Bayes' theorem:

$$P(y|X) = \frac{P(X|y)P(y)}{P(X)}$$

Where:
$$P(y|X)$$ is the posterior probability
$$P(X|y)$$ is the likelihood
$$P(y)$$ is the prior probability
$$P(X)$$ is the evidence

For Gaussian Naïve Bayes, the likelihood of features is assumed to follow a normal distribution:

$$P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma^2_y}\right)$$
"""
```

Slide 3: K-Means Clustering Implementation

K-Means clustering partitions n observations into k clusters by iteratively assigning points to the nearest centroid and updating centroid positions. This unsupervised learning algorithm is widely used for data segmentation, pattern recognition, and image compression tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

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
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            cluster_labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[cluster_labels == k].mean(axis=0)
                                    for k in range(self.n_clusters)])
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return cluster_labels
    
    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal(0, 1, (100, 2)),
    np.random.normal(4, 1, (100, 2)),
    np.random.normal(8, 1, (100, 2))
])

# Fit and plot
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Clustering Results')
```

Slide 4: Support Vector Machine (SVM) Fundamentals

Support Vector Machines find the optimal hyperplane that maximizes the margin between different classes in the feature space. The algorithm transforms data using kernel functions to handle non-linear classification problems while maintaining computational efficiency through the kernel trick.

```python
import numpy as np
from scipy.optimize import minimize

class SVM:
    def __init__(self, kernel='linear', C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
    
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            gamma = 0.1
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self._kernel_function(X[i], X[j])
        
        # Define objective function
        def objective(alpha):
            return 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K) - np.sum(alpha)
        
        # Define constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.dot(x, y)},
                      {'type': 'ineq', 'fun': lambda x: x},
                      {'type': 'ineq', 'fun': lambda x: self.C - x})
        
        # Solve quadratic programming problem
        result = minimize(objective, np.zeros(n_samples), constraints=constraints)
        self.alpha = result.x
        
        # Find support vectors
        sv = self.alpha > 1e-5
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        self.alpha = self.alpha[sv]
        
        # Calculate intercept
        self.b = np.mean(self.support_vector_labels - 
                        np.sum(self.alpha * self.support_vector_labels *
                              K[sv][:, sv], axis=0))
    
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alpha, sv, sv_label in zip(self.alpha, 
                                         self.support_vectors,
                                         self.support_vector_labels):
                s += alpha * sv_label * self._kernel_function(X[i], sv)
            y_pred[i] = s + self.b
        return np.sign(y_pred)

# Example usage
X = np.random.randn(100, 2)
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

svm = SVM(kernel='rbf', C=1.0)
svm.fit(X, y)
predictions = svm.predict(X)
```

Slide 5: Apriori Algorithm for Association Rule Mining

The Apriori algorithm discovers frequent itemsets and association rules in transaction databases. It leverages the anti-monotonicity property of support to efficiently prune the search space, making it particularly effective for market basket analysis and recommendation systems.

```python
from collections import defaultdict
from itertools import combinations

class Apriori:
    def __init__(self, min_support=0.3, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = None
        self.rules = None
    
    def _get_frequent_1_itemsets(self, transactions):
        items = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                items[frozenset([item])] += 1
        n_transactions = len(transactions)
        return {k: v/n_transactions 
                for k, v in items.items() 
                if v/n_transactions >= self.min_support}
    
    def _get_candidate_itemsets(self, prev_itemsets, k):
        candidates = set()
        for item1 in prev_itemsets:
            for item2 in prev_itemsets:
                union = item1.union(item2)
                if len(union) == k:
                    candidates.add(union)
        return candidates
    
    def fit(self, transactions):
        self.itemsets = []
        current_itemsets = self._get_frequent_1_itemsets(transactions)
        k = 2
        
        while current_itemsets:
            self.itemsets.append(current_itemsets)
            candidates = self._get_candidate_itemsets(current_itemsets.keys(), k)
            current_itemsets = {}
            
            for transaction in transactions:
                transaction_set = frozenset(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        current_itemsets[candidate] = \
                            current_itemsets.get(candidate, 0) + 1
            
            n_transactions = len(transactions)
            current_itemsets = {k: v/n_transactions 
                              for k, v in current_itemsets.items() 
                              if v/n_transactions >= self.min_support}
            k += 1
        
        self._generate_rules()
        return self
    
    def _generate_rules(self):
        self.rules = []
        for itemset_dict in self.itemsets[1:]:
            for itemset in itemset_dict:
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = itemset - antecedent
                        confidence = itemset_dict[itemset] / \
                                   self.itemsets[len(antecedent)-1][antecedent]
                        if confidence >= self.min_confidence:
                            self.rules.append((antecedent, consequent, confidence))

# Example usage
transactions = [
    ['bread', 'milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'butter'],
    ['bread', 'milk', 'butter'],
    ['bread', 'milk'],
]

apriori = Apriori(min_support=0.3, min_confidence=0.6)
apriori.fit(transactions)

print("Frequent Itemsets:")
for k, itemset_dict in enumerate(apriori.itemsets):
    print(f"{k+1}-itemsets:", dict(itemset_dict))

print("\nAssociation Rules:")
for antecedent, consequent, confidence in apriori.rules:
    print(f"{set(antecedent)} -> {set(consequent)} (conf: {confidence:.2f})")
```

Slide 6: Linear Regression Implementation

Linear regression models the relationship between dependent and independent variables by fitting a linear equation to observed data. This fundamental algorithm serves as the foundation for many advanced regression techniques and provides interpretable results for prediction tasks.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute loss
            loss = self._mse(y, y_predicted)
            self.loss_history.append(loss)
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
        
# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 3)
y = 4 + np.dot(X, np.array([2, 0.5, -1])) + np.random.randn(100)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_scaled, y)

# Make predictions
y_pred = model.predict(X_scaled)
mse = model.loss_history[-1]
print(f"Final MSE: {mse:.4f}")
print(f"Learned weights: {model.weights}")
print(f"Learned bias: {model.bias:.4f}")
```

Slide 7: Logistic Regression from Scratch

Logistic regression extends linear regression to classification problems by applying the sigmoid function to the linear combination of features. This implementation includes regularization and gradient descent optimization for robust binary classification.

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000, reg_strength=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute gradients with regularization
            dw = (1/n_samples) * (np.dot(X.T, (y_predicted - y)) + 
                                self.reg_strength * self.weights)
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        precision = np.sum((y_pred == 1) & (y == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (y == 1)) / np.sum(y == 1)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {'accuracy': accuracy, 'precision': precision, 
                'recall': recall, 'f1': f1}

# Example usage
# Generate binary classification dataset
np.random.seed(42)
X = np.random.randn(1000, 5)
true_weights = np.array([1, -2, 3, -4, 2])
y = (np.dot(X, true_weights) + np.random.randn(1000) > 0).astype(int)

# Train and evaluate model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)
metrics = model.evaluate(X, y)
print("Model Performance:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")
```

Slide 8: Artificial Neural Networks - Feed Forward Implementation

A feed-forward neural network processes information through multiple layers of interconnected neurons. This implementation includes backpropagation algorithm for training, flexible hidden layer configuration, and various activation functions for deep learning applications.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _sigmoid_derivative(self, x):
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Use ReLU for hidden layers, sigmoid for output
            if i == len(self.weights) - 1:
                a = self._sigmoid(z)
            else:
                a = self._relu(z)
            self.activations.append(a)
            
        return self.activations[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dz = delta * self._sigmoid_derivative(self.z_values[i])
            else:
                dz = np.dot(dz, self.weights[i+1].T) * self._relu_derivative(self.z_values[i])
            
            dW[i] = np.dot(self.activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
        
        return dW, db
    
    def train(self, X, y, epochs=1000):
        for _ in range(epochs):
            # Forward pass
            self.forward(X)
            
            # Backward pass
            dW, db = self.backward(X, y)
            
            # Update parameters
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Example usage
# Generate binary classification dataset
np.random.seed(42)
X = np.random.randn(1000, 4)
y = (np.sum(X**2, axis=1, keepdims=True) > 4).astype(float)

# Create and train network
nn = NeuralNetwork([4, 8, 4, 1], learning_rate=0.01)
nn.train(X, y, epochs=1000)

# Evaluate
predictions = nn.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 9: Random Forest Implementation

Random Forests combine multiple decision trees to create a robust ensemble learning method. This implementation includes bootstrap sampling, random feature selection at each split, and majority voting for classification tasks, making it effective against overfitting.

```python
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
    def _gini(self, y):
        counter = Counter(y)
        impurity = 1
        for count in counter.values():
            p = count / len(y)
            impurity -= p**2
        return impurity
    
    def _split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        return (X[left_mask], y[left_mask], 
                X[~left_mask], y[~left_mask])
    
    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        best_splits = None
        best_feature = None
        best_threshold = None
        
        current_gini = self._gini(y)
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split(X, y, 
                                                              feature_idx, 
                                                              threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                gain = current_gini - (len(y_left) * self._gini(y_left) + 
                                     len(y_right) * self._gini(y_right)) / len(y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_splits = (X_left, y_left, X_right, y_right)
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold, best_splits
    
    def _build_tree(self, X, y, depth, feature_indices):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return {'class': Counter(y).most_common(1)[0][0]}
        
        # Find best split
        feature_idx, threshold, splits = self._best_split(X, y, feature_indices)
        
        if splits is None:
            return {'class': Counter(y).most_common(1)[0][0]}
            
        X_left, y_left, X_right, y_right = splits
        
        # Build child nodes
        left_tree = self._build_tree(X_left, y_left, depth + 1, feature_indices)
        right_tree = self._build_tree(X_right, y_right, depth + 1, feature_indices)
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y, feature_indices=None):
        if feature_indices is None:
            feature_indices = list(range(X.shape[1]))
        self.tree = self._build_tree(X, y, 0, feature_indices)
        return self
    
    def _predict_single(self, x, tree):
        if 'class' in tree:
            return tree['class']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, 
                 min_samples_split=2, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.n_features = int(np.sqrt(n_features))
            else:
                self.n_features = n_features
        else:
            self.n_features = self.max_features
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample_X = X[indices]
            sample_y = y[indices]
            
            # Random feature selection
            feature_indices = np.random.choice(n_features, 
                                            self.n_features, 
                                            replace=False)
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split)
            tree.fit(sample_X, sample_y, feature_indices)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(pred).most_common(1)[0][0] 
                        for pred in predictions.T])

# Example usage
# Generate dataset
np.random.seed(42)
X = np.random.randn(1000, 10)
y = (X[:, 0] * X[:, 1] > 0).astype(int)

# Train and evaluate
rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X, y)
predictions = rf.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.4f}")
```

Slide 10: Decision Trees Core Algorithm

Decision Trees create a hierarchical model by recursively partitioning the feature space. This implementation demonstrates information gain calculation using entropy, recursive tree construction, and pruning mechanisms for optimal decision boundaries.

```python
import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        
        if np.any(left_mask) and np.any(right_mask):
            n = len(y)
            n_l, n_r = np.sum(left_mask), np.sum(right_mask)
            e_l, e_r = self._entropy(y[left_mask]), self._entropy(y[right_mask])
            child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
            return parent_entropy - child_entropy
        return 0
    
    def _find_best_split(self, X, y):
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
                    
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain == 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Create child nodes
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left, right=right)
    
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y)
        return self
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, 
                 right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

# Example usage
np.random.seed(42)
X = np.random.randn(500, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

Slide 11: K-Nearest Neighbors Implementation

K-Nearest Neighbors is a non-parametric algorithm that classifies data points based on the majority class of their k nearest neighbors. This implementation includes distance metrics, weighted voting, and efficient neighbor search using vectorized operations.

```python
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNearestNeighbors:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        elif self.weights == 'distance':
            return 1 / (distances + 1e-10)
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")
    
    def predict(self, X):
        # Calculate distances between X and training data
        distances = cdist(X, self.X_train, metric=self.metric)
        
        # Get indices of k nearest neighbors
        nearest_neighbor_indices = np.argpartition(distances, 
                                                 self.n_neighbors-1, 
                                                 axis=1)[:, :self.n_neighbors]
        
        # Get corresponding distances
        k_distances = np.take_along_axis(distances, 
                                       nearest_neighbor_indices, 
                                       axis=1)
        
        # Get labels of nearest neighbors
        k_nearest_labels = self.y_train[nearest_neighbor_indices]
        
        # Calculate weights
        weights = self._get_weights(k_distances)
        
        # Weighted voting
        predictions = []
        for i in range(len(X)):
            if self.weights == 'uniform':
                pred = Counter(k_nearest_labels[i]).most_common(1)[0][0]
            else:
                weighted_votes = {}
                for label, weight in zip(k_nearest_labels[i], weights[i]):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                pred = max(weighted_votes.items(), key=lambda x: x[1])[0]
            predictions.append(pred)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        distances = cdist(X, self.X_train, metric=self.metric)
        nearest_neighbor_indices = np.argpartition(distances, 
                                                 self.n_neighbors-1, 
                                                 axis=1)[:, :self.n_neighbors]
        k_distances = np.take_along_axis(distances, 
                                       nearest_neighbor_indices, 
                                       axis=1)
        k_nearest_labels = self.y_train[nearest_neighbor_indices]
        weights = self._get_weights(k_distances)
        
        # Calculate class probabilities
        classes = np.unique(self.y_train)
        probabilities = np.zeros((len(X), len(classes)))
        
        for i in range(len(X)):
            for j, class_label in enumerate(classes):
                mask = k_nearest_labels[i] == class_label
                probabilities[i, j] = np.sum(weights[i][mask]) / np.sum(weights[i])
                
        return probabilities

# Example usage with both binary and multi-class classification
np.random.seed(42)

# Binary classification dataset
X_binary = np.random.randn(1000, 2)
y_binary = (X_binary[:, 0] + X_binary[:, 1] > 0).astype(int)

# Multi-class dataset
X_multi = np.random.randn(1000, 2)
y_multi = (np.sum(X_multi**2, axis=1) < 1).astype(int) + \
          (X_multi[:, 0] > 1).astype(int)

# Train and evaluate both cases
knn_binary = KNearestNeighbors(n_neighbors=5, weights='distance')
knn_multi = KNearestNeighbors(n_neighbors=5, weights='distance')

# Split data
X_train_b, X_test_b = X_binary[:800], X_binary[800:]
y_train_b, y_test_b = y_binary[:800], y_binary[800:]

X_train_m, X_test_m = X_multi[:800], X_multi[800:]
y_train_m, y_test_m = y_multi[:800], y_multi[800:]

# Fit and predict
knn_binary.fit(X_train_b, y_train_b)
knn_multi.fit(X_train_m, y_train_m)

binary_pred = knn_binary.predict(X_test_b)
multi_pred = knn_multi.predict(X_test_m)

# Calculate accuracies
binary_acc = np.mean(binary_pred == y_test_b)
multi_acc = np.mean(multi_pred == y_test_m)

print(f"Binary Classification Accuracy: {binary_acc:.4f}")
print(f"Multi-class Classification Accuracy: {multi_acc:.4f}")
```

Slide 12: Model Evaluation and Metrics Implementation

A comprehensive implementation of various evaluation metrics for classification and regression tasks. This code provides essential tools for assessing model performance through accuracy, precision, recall, F1-score, ROC curves, and confusion matrices.

```python
import numpy as np
from sklearn.metrics import roc_curve, auc

class ModelEvaluator:
    def __init__(self):
        pass
        
    def confusion_matrix(self, y_true, y_pred):
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(y_true)):
            true_index = np.where(classes == y_true[i])[0][0]
            pred_index = np.where(classes == y_pred[i])[0][0]
            matrix[true_index][pred_index] += 1
            
        return matrix, classes
    
    def classification_metrics(self, y_true, y_pred, y_prob=None):
        metrics = {}
        conf_matrix, classes = self.confusion_matrix(y_true, y_pred)
        
        # Calculate metrics for each class
        for i, class_label in enumerate(classes):
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            tn = np.sum(conf_matrix) - tp - fp - fn
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'class_{class_label}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        # Calculate macro-averaged metrics
        metrics['macro_avg'] = {
            'precision': np.mean([m['precision'] for m in metrics.values()]),
            'recall': np.mean([m['recall'] for m in metrics.values()]),
            'f1_score': np.mean([m['f1_score'] for m in metrics.values()])
        }
        
        # Calculate accuracy
        metrics['accuracy'] = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
        
        # Calculate ROC and AUC if probabilities are provided
        if y_prob is not None and len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            metrics['roc'] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc(fpr, tpr)
            }
        
        return metrics
    
    def regression_metrics(self, y_true, y_pred):
        metrics = {}
        
        # Mean Squared Error
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
        
        return metrics

# Example usage
np.random.seed(42)

# Generate classification data
X = np.random.randn(1000, 2)
y_true = (X[:, 0] + X[:, 1] > 0).astype(int)
y_pred = (X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.5 > 0).astype(int)
y_prob = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))

# Generate regression data
X_reg = np.random.randn(1000, 1)
y_true_reg = 2 * X_reg + 1 + np.random.randn(1000, 1) * 0.2
y_pred_reg = 2.1 * X_reg + 0.9 + np.random.randn(1000, 1) * 0.3

# Create evaluator
evaluator = ModelEvaluator()

# Evaluate classification
class_metrics = evaluator.classification_metrics(y_true, y_pred, 
                                               np.column_stack((1-y_prob, y_prob)))

# Evaluate regression
reg_metrics = evaluator.regression_metrics(y_true_reg, y_pred_reg)

# Print results
print("Classification Metrics:")
print(f"Accuracy: {class_metrics['accuracy']:.4f}")
print(f"AUC-ROC: {class_metrics['roc']['auc']:.4f}")
print("\nRegression Metrics:")
print(f"RMSE: {reg_metrics['rmse']:.4f}")
print(f"R²: {reg_metrics['r2']:.4f}")
```

Slide 13: Additional Resources

Here are relevant research papers and resources for further exploration of machine learning algorithms:

*   Machine Learning Best Practices: [https://arxiv.org/abs/2004.00323](https://arxiv.org/abs/2004.00323) - Systematic review of machine learning performance optimization
*   Deep Learning Architectures: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385) - Residual Learning and Deep Networks
*   Ensemble Methods: [https://arxiv.org/abs/1804.09337](https://arxiv.org/abs/1804.09337) - Advanced Ensemble Techniques
*   Search for these topics:
    *   "Gradient Boosting Machines Theory and Applications"
    *   "Neural Architecture Search Recent Advances"
    *   "Automated Machine Learning Systems Design"

Slide 14: Advanced Model Selection

This implementation demonstrates cross-validation, hyperparameter tuning, and model selection techniques for optimizing machine learning algorithms across different datasets and requirements.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings

class ModelSelector:
    def __init__(self, models, param_grids):
        self.models = models
        self.param_grids = param_grids
        self.best_params = {}
        self.best_scores = {}
        self.best_models = {}
        
    def _generate_param_combinations(self, param_grid):
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
    
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            print(f"\nTuning {model_name}...")
            best_score = -np.inf
            best_params = None
            
            for params in self._generate_param_combinations(self.param_grids[model_name]):
                scores = []
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    model.set_params(**params)
                    model.fit(X_train, y_train)
                    
                    if scoring == 'accuracy':
                        score = accuracy_score(y_val, model.predict(X_val))
                    scores.append(score)
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            self.best_params[model_name] = best_params
            self.best_scores[model_name] = best_score
            
            # Train final model with best parameters
            final_model = clone(model)
            final_model.set_params(**best_params)
            final_model.fit(X, y)
            self.best_models[model_name] = final_model
    
    def get_best_model(self):
        best_model_name = max(self.best_scores.items(), key=lambda x: x[1])[0]
        return (best_model_name, self.best_models[best_model_name])
    
    def get_results_summary(self):
        summary = []
        for model_name in self.models.keys():
            summary.append({
                'model': model_name,
                'best_score': self.best_scores[model_name],
                'best_params': self.best_params[model_name]
            })
        return sorted(summary, key=lambda x: x['best_score'], reverse=True)

# Example usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Define models and parameter grids
models = {
    'random_forest': RandomForestClassifier(),
    'svm': SVC(),
    'logistic': LogisticRegression()
}

param_grids = {
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    },
    'svm': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear']
    },
    'logistic': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2']
    }
}

# Create selector and find best model
selector = ModelSelector(models, param_grids)
selector.cross_validate(X, y)

# Get results
best_model_name, best_model = selector.get_best_model()
results = selector.get_results_summary()

print("\nModel Selection Results:")
for result in results:
    print(f"\n{result['model']}:")
    print(f"Best CV Score: {result['best_score']:.4f}")
    print(f"Best Parameters: {result['best_params']}")
```

Slide 15: Performance Optimization and Parallel Processing

This implementation showcases advanced techniques for optimizing machine learning model performance through parallel processing, batch processing, and efficient memory management.

```python
import numpy as np
from multiprocessing import Pool
from functools import partial
import time

class ParallelModelTrainer:
    def __init__(self, base_model, n_jobs=-1):
        self.base_model = base_model
        self.n_jobs = n_jobs
        self.models = []
        
    def _train_model(self, X, y, fold_idx, params=None):
        # Clone base model and set parameters if provided
        model = clone(self.base_model)
        if params:
            model.set_params(**params)
            
        # Get fold indices
        train_idx = self.folds != fold_idx
        val_idx = self.folds == fold_idx
        
        # Train model
        model.fit(X[train_idx], y[train_idx])
        score = accuracy_score(y[val_idx], model.predict(X[val_idx]))
        
        return {'fold': fold_idx, 'model': model, 'score': score}
    
    def parallel_cross_val(self, X, y, n_splits=5):
        # Generate fold assignments
        self.folds = np.random.randint(0, n_splits, len(X))
        
        # Create partial function with fixed arguments
        train_fold = partial(self._train_model, X, y)
        
        # Train models in parallel
        with Pool(self.n_jobs) as pool:
            results = pool.map(train_fold, range(n_splits))
            
        self.fold_results = results
        return np.mean([r['score'] for r in results])
    
    def parallel_predict(self, X, batch_size=1000):
        predictions = []
        
        # Process data in batches
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            
            # Get predictions from all models
            batch_preds = []
            for result in self.fold_results:
                pred = result['model'].predict(batch)
                batch_preds.append(pred)
                
            # Aggregate predictions
            ensemble_pred = np.mean(batch_preds, axis=0)
            predictions.append(ensemble_pred)
            
        return np.concatenate(predictions)

class MemoryEfficientModel:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
        
    def batch_process(self, X, func):
        results = []
        for i in range(0, len(X), self.chunk_size):
            chunk = X[i:i + self.chunk_size]
            chunk_result = func(chunk)
            results.append(chunk_result)
        return np.concatenate(results)
    
    def fit(self, X, y):
        self.features_mean = np.zeros(X.shape[1])
        self.features_std = np.zeros(X.shape[1])
        
        # Calculate statistics in chunks
        n_samples = 0
        sum_x = np.zeros(X.shape[1])
        sum_x2 = np.zeros(X.shape[1])
        
        for i in range(0, len(X), self.chunk_size):
            chunk = X[i:i + self.chunk_size]
            n_samples += len(chunk)
            sum_x += np.sum(chunk, axis=0)
            sum_x2 += np.sum(chunk ** 2, axis=0)
        
        # Calculate mean and std
        self.features_mean = sum_x / n_samples
        self.features_std = np.sqrt(sum_x2/n_samples - self.features_mean**2)
        
        # Train model on normalized data
        def normalize_and_train(chunk):
            normalized_chunk = (chunk - self.features_mean) / self.features_std
            return normalized_chunk
        
        X_normalized = self.batch_process(X, normalize_and_train)
        self.model.fit(X_normalized, y)
        
    def predict(self, X):
        def normalize_and_predict(chunk):
            normalized_chunk = (chunk - self.features_mean) / self.features_std
            return self.model.predict(normalized_chunk)
            
        return self.batch_process(X, normalize_and_predict)

# Example usage
X = np.random.randn(10000, 100)
y = (np.sum(X[:, :10], axis=1) > 0).astype(int)

# Parallel training
trainer = ParallelModelTrainer(RandomForestClassifier(), n_jobs=4)
cv_score = trainer.parallel_cross_val(X, y)
predictions = trainer.parallel_predict(X)

print(f"Cross-validation score: {cv_score:.4f}")

# Memory efficient processing
efficient_model = MemoryEfficientModel(chunk_size=1000)
efficient_model.fit(X, y)
efficient_predictions = efficient_model.predict(X)
```

Slide 16: Model Interpretability Tools

A comprehensive implementation of tools for understanding and interpreting machine learning models, including feature importance analysis, partial dependence plots, and SHAP values computation.

```python
import numpy as np
from scipy.stats import spearmanr
import warnings

class ModelInterpreter:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        
    def feature_importance(self, X, y, method='permutation'):
        importance_scores = {}
        baseline_score = self.model.score(X, y)
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X[:, i])
            permuted_score = self.model.score(X_permuted, y)
            
            importance = baseline_score - permuted_score
            feature_name = self.feature_names[i] if self.feature_names else f"Feature_{i}"
            importance_scores[feature_name] = importance
            
        return dict(sorted(importance_scores.items(), 
                         key=lambda x: abs(x[1]), 
                         reverse=True))
    
    def partial_dependence(self, X, feature_idx, grid_points=50):
        feature_values = np.linspace(
            np.min(X[:, feature_idx]),
            np.max(X[:, feature_idx]),
            grid_points
        )
        
        pdp_values = []
        for value in feature_values:
            X_modified = X.copy()
            X_modified[:, feature_idx] = value
            predictions = self.model.predict(X_modified)
            pdp_values.append(np.mean(predictions))
            
        return feature_values, np.array(pdp_values)
    
    def feature_interactions(self, X, threshold=0.05):
        n_features = X.shape[1]
        interactions = []
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                correlation, p_value = spearmanr(X[:, i], X[:, j])
                
                if abs(correlation) > threshold and p_value < threshold:
                    feature1 = self.feature_names[i] if self.feature_names else f"Feature_{i}"
                    feature2 = self.feature_names[j] if self.feature_names else f"Feature_{j}"
                    interactions.append({
                        'features': (feature1, feature2),
                        'correlation': correlation,
                        'p_value': p_value
                    })
                    
        return sorted(interactions, key=lambda x: abs(x['correlation']), reverse=True)
    
    def local_interpretation(self, X, instance_idx):
        """LIME-inspired local interpretation"""
        n_samples = 1000
        n_features = X.shape[1]
        
        # Generate perturbed samples around the instance
        np.random.seed(42)
        perturbations = np.random.normal(0, 0.1, (n_samples, n_features))
        X_perturbed = X[instance_idx] + perturbations
        
        # Get predictions for perturbed samples
        y_perturbed = self.model.predict(X_perturbed)
        
        # Calculate distance weights
        distances = np.linalg.norm(perturbations, axis=1)
        weights = np.exp(-distances)
        
        # Fit weighted linear model
        from sklearn.linear_model import LinearRegression
        local_model = LinearRegression()
        local_model.fit(perturbations, y_perturbed, sample_weight=weights)
        
        # Get feature importance scores
        importance = {}
        for i in range(n_features):
            feature_name = self.feature_names[i] if self.feature_names else f"Feature_{i}"
            importance[feature_name] = abs(local_model.coef_[i])
            
        return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))

# Example usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=10, random_state=42)
feature_names = [f"Feature_{i}" for i in range(20)]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create interpreter
interpreter = ModelInterpreter(model, feature_names)

# Get global feature importance
importance = interpreter.feature_importance(X, y)
print("\nGlobal Feature Importance:")
for feature, score in list(importance.items())[:5]:
    print(f"{feature}: {score:.4f}")

# Get feature interactions
interactions = interpreter.feature_interactions(X)
print("\nTop Feature Interactions:")
for interaction in interactions[:3]:
    print(f"{interaction['features']}: {interaction['correlation']:.4f}")

# Get local interpretation for a specific instance
local_importance = interpreter.local_interpretation(X, 0)
print("\nLocal Feature Importance (Instance 0):")
for feature, score in list(local_importance.items())[:5]:
    print(f"{feature}: {score:.4f}")
```

Slide 17: Final Model Deployment

A robust implementation focusing on model serialization, deployment, and monitoring capabilities for production environments, including data validation and model serving.

```python
import joblib
import json
import numpy as np
from datetime import datetime
import warnings

class ModelDeployment:
    def __init__(self, model, feature_names, version="1.0.0"):
        self.model = model
        self.feature_names = feature_names
        self.version = version
        self.metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'features': feature_names
        }
        
    def save_model(self, path):
        """Save model and metadata"""
        model_path = f"{path}/model.joblib"
        metadata_path = f"{path}/metadata.json"
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    @staticmethod
    def load_model(path):
        """Load model and metadata"""
        model = joblib.load(f"{path}/model.joblib")
        
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
            
        return ModelDeployment(model, 
                             metadata['features'], 
                             metadata['version'])
    
    def validate_input(self, X, raise_error=True):
        """Validate input data"""
        if isinstance(X, dict):
            X = self._dict_to_array(X)
            
        if X.shape[1] != len(self.feature_names):
            msg = f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
            if raise_error:
                raise ValueError(msg)
            return False, msg
            
        return True, "Valid input"
    
    def _dict_to_array(self, data_dict):
        """Convert dictionary input to numpy array"""
        if isinstance(data_dict, dict):
            data_dict = [data_dict]
            
        X = np.zeros((len(data_dict), len(self.feature_names)))
        for i, sample in enumerate(data_dict):
            for j, feature in enumerate(self.feature_names):
                X[i, j] = sample.get(feature, 0)
        return X
    
    def predict(self, X, validate=True):
        """Make predictions with input validation"""
        if validate:
            is_valid, msg = self.validate_input(X)
            if not is_valid:
                raise ValueError(msg)
                
        if isinstance(X, dict):
            X = self._dict_to_array(X)
            
        predictions = self.model.predict(X)
        return predictions.tolist()
    
    def predict_proba(self, X, validate=True):
        """Make probability predictions with input validation"""
        if validate:
            is_valid, msg = self.validate_input(X)
            if not is_valid:
                raise ValueError(msg)
                
        if isinstance(X, dict):
            X = self._dict_to_array(X)
            
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            return probabilities.tolist()
        else:
            raise AttributeError("Model does not support probability predictions")

# Example usage
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import tempfile
import os

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create deployment
deployment = ModelDeployment(model, feature_names)

# Save and load model
with tempfile.TemporaryDirectory() as tmp_dir:
    # Save model
    deployment.save_model(tmp_dir)
    
    # Load model
    loaded_deployment = ModelDeployment.load_model(tmp_dir)
    
    # Test predictions
    sample_input = {
        'feature_0': 1.0,
        'feature_1': -0.5,
        'feature_2': 0.2,
        'feature_3': 0.8,
        'feature_4': -0.3,
        'feature_5': 0.1,
        'feature_6': -0.7,
        'feature_7': 0.4,
        'feature_8': -0.2,
        'feature_9': 0.6
    }
    
    # Make predictions
    prediction = loaded_deployment.predict(sample_input)
    probabilities = loaded_deployment.predict_proba(sample_input)
    
    print(f"\nPrediction: {prediction}")
    print(f"Probabilities: {probabilities}")
```

Slide 18: Additional Resources

*   Advanced Machine Learning Research Papers:
    *   [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239) - Model Interpretation Methods
    *   [https://arxiv.org/abs/2003.07631](https://arxiv.org/abs/2003.07631) - Production ML Systems
    *   [https://arxiv.org/abs/1908.06165](https://arxiv.org/abs/1908.06165) - Model Deployment Best Practices
*   Recommended Search Topics:
    *   "MLOps Best Practices and Tools"
    *   "Automated Model Monitoring Systems"
    *   "Machine Learning Model Governance"

