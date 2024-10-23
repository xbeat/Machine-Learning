## Building a Machine Learning Model

Slide 1: Data Preparation and Loading 

Advanced data preprocessing techniques including handling missing values, feature scaling, and categorical encoding using pandas and scikit-learn libraries for preparing machine learning datasets.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load and preprocess sample dataset
data = pd.read_csv('sample_data.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'income', 'credit_score']
data_imputed[numerical_cols] = scaler.fit_transform(data_imputed[numerical_cols])

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['occupation', 'education']
for col in categorical_cols:
    data_imputed[col] = le.fit_transform(data_imputed[col])

# Output example
print("Processed dataset shape:", data_imputed.shape)
print("\nFirst 5 rows:\n", data_imputed.head())
```

Slide 2: Linear Regression Implementation

Implementing linear regression from scratch using gradient descent optimization, demonstrating the fundamental concepts of model training and prediction.

```python
import numpy as np

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
        
        for _ in range(self.iterations):
            # Linear model
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(100)

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("First 5 predictions:", predictions[:5])
```

Slide 3: Decision Trees from Scratch

Understanding decision tree construction through implementation of a basic decision tree classifier with information gain splitting criterion.

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
        
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
        
    def _information_gain(self, parent, left_child, right_child):
        num_left = len(left_child)
        num_right = len(right_child)
        
        if num_left == 0 or num_right == 0:
            return 0
            
        parent_entropy = self._entropy(parent)
        left_entropy = self._entropy(left_child)
        right_entropy = self._entropy(right_child)
        
        child_entropy = (num_left * left_entropy + num_right * right_entropy) / len(parent)
        return parent_entropy - child_entropy
        
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
        
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        if (self.max_depth <= depth or n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
            
        feature, threshold = self._best_split(X, y)
        
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
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)
print("Accuracy:", np.mean(predictions == y))
```

Slide 4: Neural Network Architecture

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
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, delta) / m
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
                
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(predictions - y))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage
X = np.random.randn(100, 2)
y = np.array([(x[0] + x[1] > 0) for x in X]).reshape(-1, 1)

nn = NeuralNetwork([2, 4, 1])
nn.train(X, y, epochs=1000, learning_rate=0.1)
```

Slide 5: Real-world Example - House Price Prediction

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load housing dataset
df = pd.read_csv('housing.csv')

# Feature engineering
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X = df[features]
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom Linear Regression implementation
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradient descent
            dw = (1/X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1/X.shape[0]) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train and evaluate
model = CustomLinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

Slide 6: K-Means Clustering Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        
    def fit(self, X):
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            for k in range(self.n_clusters):
                if sum(self.labels == k) > 0:
                    self.centroids[k] = X[self.labels == k].mean(axis=0)
                    
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
                
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

# Fit and predict
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x')
plt.savefig('kmeans_clustering.png')
plt.close()
```

Slide 7: Support Vector Machine Implementation

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Generate sample data
X = np.random.randn(100, 2)
y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and evaluate
svm = SVM()
svm.fit(X_scaled, y)
predictions = svm.predict(X_scaled)

accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
```

Slide 8: Random Forest Implementation

A simplified random forest classifier implementation focusing on the ensemble of decision trees with bootstrap sampling and majority voting for predictions.

```python
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(pred).most_common(1)[0][0] 
                        for pred in predictions.T])

# Example usage
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

rf = RandomForest(n_trees=5, max_depth=3)
rf.fit(X, y)
predictions = rf.predict(X)
accuracy = np.mean(predictions == y)
print(f"Random Forest Accuracy: {accuracy:.2f}")
```

Slide 9: Gradient Boosting Implementation

Basic implementation of gradient boosting for regression tasks using decision trees as base learners and squared error loss function.

```python
import numpy as np

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        
    def fit(self, X, y):
        # Initialize prediction with zeros
        self.initial_guess = np.mean(y)
        f = np.full_like(y, self.initial_guess, dtype=float)
        
        for _ in range(self.n_estimators):
            residuals = y - f
            tree = DecisionTree(max_depth=3)
            tree.fit(X, residuals)
            
            predictions = tree.predict(X)
            f += self.learning_rate * predictions
            self.trees.append(tree)
            
    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_guess)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Example usage
X = np.random.randn(100, 2)
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100)

gb = GradientBoosting(n_estimators=50)
gb.fit(X, y)
predictions = gb.predict(X)
mse = np.mean((predictions - y)**2)
print(f"Mean Squared Error: {mse:.2f}")
```

Slide 10: Cross-Validation Implementation

Implementation of k-fold cross-validation to assess model performance and prevent overfitting.

```python
import numpy as np

def cross_validation(X, y, model, k_folds=5):
    fold_size = len(X) // k_folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    scores = []
    
    for i in range(k_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = np.mean(predictions == y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# Example usage
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3)

mean_score, std_score = cross_validation(X, y, model)
print(f"CV Score: {mean_score:.2f} (+/- {std_score:.2f})")
```

Slide 11: Feature Selection Implementation

Implementation of feature selection using correlation analysis and recursive feature elimination.

```python
import numpy as np
from sklearn.base import clone

def select_features(X, y, threshold=0.5):
    correlations = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) 
                            for i in range(X.shape[1])])
    
    selected_features = correlations > threshold
    return selected_features, correlations

def recursive_feature_elimination(X, y, model, n_features_to_select=3):
    n_features = X.shape[1]
    selected_features = np.ones(n_features, dtype=bool)
    feature_ranks = np.zeros(n_features)
    
    while sum(selected_features) > n_features_to_select:
        model_clone = clone(model)
        model_clone.fit(X[:, selected_features], y)
        
        # Get feature importance
        importance = abs(model_clone.coef_[0])
        least_important = np.argmin(importance)
        
        current_selected = np.where(selected_features)[0]
        selected_features[current_selected[least_important]] = False
        
    return selected_features

# Example usage
X = np.random.randn(100, 5)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)

selected, correlations = select_features(X, y, threshold=0.3)
print("Correlation-based selection:", selected)
print("Feature correlations:", correlations)
```

Slide 12: Model Evaluation Metrics

Implementation of common evaluation metrics for classification and regression tasks.

```python
import numpy as np

def classification_metrics(y_true, y_pred):
    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def regression_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2
    }

# Example usage
y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)
print("Classification metrics:", classification_metrics(y_true, y_pred))

y_true_reg = np.random.randn(100)
y_pred_reg = y_true_reg + np.random.randn(100) * 0.1
print("Regression metrics:", regression_metrics(y_true_reg, y_pred_reg))
```

Slide 13: Additional Resources

1.  "A Tutorial on Support Vector Machines for Pattern Recognition" [https://arxiv.org/abs/1303.5779](https://arxiv.org/abs/1303.5779)
2.  "Random Forests in Machine Learning" [https://arxiv.org/abs/1011.1669](https://arxiv.org/abs/1011.1669)
3.  "XGBoost: A Scalable Tree Boosting System" [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
4.  "Deep Learning: A Comprehensive Survey" [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828)
5.  "An Introduction to Statistical Learning" [https://arxiv.org/abs/2103.05155](https://arxiv.org/abs/2103.05155)

