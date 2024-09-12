## Building an Isolation Forest Algorithm in Python
Slide 1: Introduction to Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm used for anomaly detection. It works on the principle that anomalies are rare and different, making them easier to isolate than normal points. This algorithm is particularly effective for high-dimensional datasets and can handle large-scale problems efficiently.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate sample data
X = np.random.randn(1000, 2)
X[0] = [3, 3]  # Add an anomaly

# Create and fit the model
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# Predict anomalies
y_pred = clf.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.colorbar()
plt.title("Isolation Forest Anomaly Detection")
plt.show()
```

Slide 2: Basic Concepts of Isolation Forest

The Isolation Forest algorithm builds an ensemble of isolation trees for a given dataset. Each tree is constructed by recursively partitioning the data space until each data point is isolated. Anomalies require fewer partitions to be isolated, resulting in shorter paths from the root to the leaf nodes in the isolation trees.

```python
import numpy as np

class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None

    def fit(self, X, current_height=0):
        if current_height >= self.height_limit or len(X) <= 1:
            return Node(size=len(X))
        
        feature = np.random.randint(X.shape[1])
        split_value = np.random.uniform(X[:, feature].min(), X[:, feature].max())
        
        left_indices = X[:, feature] < split_value
        right_indices = ~left_indices
        
        return Node(
            feature=feature,
            split_value=split_value,
            left=self.fit(X[left_indices], current_height + 1),
            right=self.fit(X[right_indices], current_height + 1)
        )

class Node:
    def __init__(self, feature=None, split_value=None, left=None, right=None, size=0):
        self.feature = feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size

# Usage
X = np.random.randn(100, 2)
tree = IsolationTree(height_limit=10)
root = tree.fit(X)
```

Slide 3: Building an Isolation Tree

An Isolation Tree is constructed by randomly selecting a feature and a split value within the feature's range. This process continues recursively for the resulting subspaces until a stopping criterion is met, such as reaching a maximum tree height or having only one data point in a subspace.

```python
def build_isolation_tree(X, height_limit, current_height=0):
    if current_height >= height_limit or len(X) <= 1:
        return Node(size=len(X))
    
    feature = np.random.randint(X.shape[1])
    split_value = np.random.uniform(X[:, feature].min(), X[:, feature].max())
    
    left_indices = X[:, feature] < split_value
    right_indices = ~left_indices
    
    return Node(
        feature=feature,
        split_value=split_value,
        left=build_isolation_tree(X[left_indices], height_limit, current_height + 1),
        right=build_isolation_tree(X[right_indices], height_limit, current_height + 1)
    )

# Usage
X = np.random.randn(100, 2)
root = build_isolation_tree(X, height_limit=10)
```

Slide 4: Anomaly Score Calculation

The anomaly score for a data point is calculated based on its average path length across multiple isolation trees. Shorter path lengths indicate a higher likelihood of being an anomaly. The score is normalized using the average path length of unsuccessful search in a Binary Search Tree.

```python
import numpy as np

def calculate_average_path_length(n):
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)

def path_length(x, node, current_height=0):
    if node.left is None and node.right is None:
        return current_height
    
    if x[node.feature] < node.split_value:
        return path_length(x, node.left, current_height + 1)
    else:
        return path_length(x, node.right, current_height + 1)

def anomaly_score(x, tree, n):
    path_len = path_length(x, tree.root)
    return 2 ** (-path_len / calculate_average_path_length(n))

# Usage
X = np.random.randn(100, 2)
tree = IsolationTree(height_limit=10)
root = tree.fit(X)
scores = [anomaly_score(x, tree, len(X)) for x in X]
```

Slide 5: Implementing the Isolation Forest

The Isolation Forest algorithm combines multiple Isolation Trees to create a robust anomaly detection system. We'll implement a basic version of the algorithm, including methods for fitting the model and predicting anomalies.

```python
import numpy as np

class IsolationForest:
    def __init__(self, n_trees=100, max_samples='auto', contamination=0.1):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees = []
        self.threshold = None

    def fit(self, X):
        n_samples = X.shape[0]
        
        if self.max_samples == 'auto':
            self.max_samples = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            self.max_samples = int(self.max_samples * n_samples)
        
        height_limit = int(np.ceil(np.log2(self.max_samples)))
        
        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            tree = IsolationTree(height_limit)
            tree.root = tree.fit(X[indices])
            self.trees.append(tree)
        
        return self

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= self.threshold, 1, -1)

    def decision_function(self, X):
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            scores[i] = np.mean([anomaly_score(x, tree, self.max_samples) for tree in self.trees])
        
        if self.threshold is None:
            self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        return scores

# Usage
X = np.random.randn(1000, 2)
X[0] = [3, 3]  # Add an anomaly
clf = IsolationForest()
clf.fit(X)
y_pred = clf.predict(X)
```

Slide 6: Visualizing Isolation Forest Results

To better understand how the Isolation Forest algorithm works, we can visualize its results. We'll create a plot showing normal points and detected anomalies, along with the decision boundary.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 2)
X[0:10] = np.random.uniform(low=-4, high=4, size=(10, 2))  # Add some anomalies

# Fit Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(X)

# Predict anomalies
y_pred = clf.predict(X)

# Create a mesh grid
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the results
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
b1 = plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='white', s=20, edgecolor='k')
b2 = plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', s=40, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2], ["normal points", "anomalies"])
plt.title("Isolation Forest Anomaly Detection")
plt.show()
```

Slide 7: Handling High-Dimensional Data

One of the advantages of Isolation Forest is its ability to handle high-dimensional data efficiently. Let's implement a function to generate high-dimensional data with anomalies and visualize the results using dimensionality reduction.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def generate_high_dimensional_data(n_samples=1000, n_features=50, n_anomalies=10):
    X = np.random.randn(n_samples, n_features)
    anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, n_features))
    X[:n_anomalies] = anomalies
    return X

# Generate and process high-dimensional data
X = generate_high_dimensional_data()
clf = IsolationForest(contamination=0.01, random_state=42)
y_pred = clf.fit_predict(X)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[y_pred == 1, 0], X_pca[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X_pca[y_pred == -1, 0], X_pca[y_pred == -1, 1], c='red', label='Anomaly')
plt.legend()
plt.title("Isolation Forest on High-Dimensional Data (PCA Visualization)")
plt.show()

print(f"Number of detected anomalies: {sum(y_pred == -1)}")
```

Slide 8: Comparing Isolation Forest with Other Anomaly Detection Methods

To understand the strengths of Isolation Forest, let's compare it with other popular anomaly detection methods, such as Local Outlier Factor (LOF) and One-Class SVM, using a synthetic dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 2)
X[0:20] = np.random.uniform(low=-4, high=4, size=(20, 2))  # Add anomalies

# Initialize and fit models
isolation_forest = IsolationForest(contamination=0.02, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02)
one_class_svm = OneClassSVM(nu=0.02, kernel="rbf", gamma=0.1)

# Predict anomalies
y_pred_if = isolation_forest.fit_predict(X)
y_pred_lof = lof.fit_predict(X)
y_pred_ocsvm = one_class_svm.fit_predict(X)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X[y_pred_if == 1, 0], X[y_pred_if == 1, 1], c='blue', label='Normal')
ax1.scatter(X[y_pred_if == -1, 0], X[y_pred_if == -1, 1], c='red', label='Anomaly')
ax1.set_title("Isolation Forest")
ax1.legend()

ax2.scatter(X[y_pred_lof == 1, 0], X[y_pred_lof == 1, 1], c='blue', label='Normal')
ax2.scatter(X[y_pred_lof == -1, 0], X[y_pred_lof == -1, 1], c='red', label='Anomaly')
ax2.set_title("Local Outlier Factor")
ax2.legend()

ax3.scatter(X[y_pred_ocsvm == 1, 0], X[y_pred_ocsvm == 1, 1], c='blue', label='Normal')
ax3.scatter(X[y_pred_ocsvm == -1, 0], X[y_pred_ocsvm == -1, 1], c='red', label='Anomaly')
ax3.set_title("One-Class SVM")
ax3.legend()

plt.tight_layout()
plt.show()

print(f"Isolation Forest detected {sum(y_pred_if == -1)} anomalies")
print(f"Local Outlier Factor detected {sum(y_pred_lof == -1)} anomalies")
print(f"One-Class SVM detected {sum(y_pred_ocsvm == -1)} anomalies")
```

Slide 9: Optimizing Isolation Forest Parameters

The performance of Isolation Forest can be improved by tuning its parameters. Let's explore how different parameters affect the algorithm's performance using a grid search approach.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Generate synthetic data with known anomalies
np.random.seed(42)
X = np.random.randn(1000, 2)
y_true = np.ones(1000)
X[0:50] = np.random.uniform(low=-4, high=4, size=(50, 2))
y_true[0:50] = -1

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': [0.1, 0.5, 1.0],
    'contamination': [0.01, 0.05, 0.1],
    'max_features': [0.5, 0.8, 1.0]
}

# Create custom scorer
scorer = make_scorer(f1_score, pos_label=-1)

# Perform grid search
grid_search = GridSearchCV(
    IsolationForest(random_state=42),
    param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X, y_true)

print("Best parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)

# Use best parameters for final model
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X)
```

Slide 10: Real-life Example: Network Intrusion Detection

Isolation Forest can be applied to detect network intrusions by identifying anomalous network traffic patterns. This example demonstrates how to use Isolation Forest for this purpose using a simplified dataset.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Simulated network traffic data
# Features: packet size, inter-arrival time, protocol type (encoded)
np.random.seed(42)
normal_traffic = np.column_stack([
    np.random.normal(1000, 200, 1000),  # packet size
    np.random.exponential(0.1, 1000),   # inter-arrival time
    np.random.choice([0, 1, 2], 1000)   # protocol type
])

# Simulated intrusion attempts
intrusions = np.column_stack([
    np.random.normal(500, 100, 50),    # smaller packets
    np.random.exponential(0.01, 50),   # faster inter-arrival time
    np.random.choice([0, 1, 2], 50)    # protocol type
])

# Combine normal traffic and intrusions
X = np.vstack([normal_traffic, intrusions])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
y_pred = clf.fit_predict(X_scaled)

print(f"Number of detected anomalies: {sum(y_pred == -1)}")
print(f"Percentage of traffic flagged as anomalous: {sum(y_pred == -1) / len(y_pred):.2%}")
```

Slide 11: Real-life Example: Sensor Data Anomaly Detection

Isolation Forest can be used to detect anomalies in sensor data, which is crucial for maintaining industrial equipment. This example shows how to apply Isolation Forest to sensor readings from a hypothetical machine.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate simulated sensor data
np.random.seed(42)
n_samples = 1000
time = np.arange(n_samples)
temperature = 20 + 5 * np.sin(2 * np.pi * time / 100) + np.random.normal(0, 0.5, n_samples)
vibration = 0.5 + 0.1 * np.sin(2 * np.pi * time / 50) + np.random.normal(0, 0.05, n_samples)

# Introduce anomalies
anomaly_indices = [200, 500, 800]
temperature[anomaly_indices] += np.random.uniform(5, 10, len(anomaly_indices))
vibration[anomaly_indices] += np.random.uniform(0.5, 1, len(anomaly_indices))

# Combine features
X = np.column_stack([temperature, vibration])

# Apply Isolation Forest
clf = IsolationForest(contamination=0.01, random_state=42)
y_pred = clf.fit_predict(X)

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(time, temperature, label='Temperature')
plt.scatter(time[y_pred == -1], temperature[y_pred == -1], color='red', label='Anomaly')
plt.legend()
plt.title('Temperature Readings with Detected Anomalies')

plt.subplot(212)
plt.plot(time, vibration, label='Vibration')
plt.scatter(time[y_pred == -1], vibration[y_pred == -1], color='red', label='Anomaly')
plt.legend()
plt.title('Vibration Readings with Detected Anomalies')

plt.tight_layout()
plt.show()

print(f"Number of detected anomalies: {sum(y_pred == -1)}")
```

Slide 12: Handling Categorical Features in Isolation Forest

Isolation Forest primarily works with numerical data. When dealing with categorical features, we need to preprocess the data. This example demonstrates how to handle categorical features using one-hot encoding before applying Isolation Forest.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

# Create a sample dataset with mixed numerical and categorical features
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'temperature': np.random.normal(25, 5, n_samples),
    'pressure': np.random.normal(100, 10, n_samples),
    'machine_type': np.random.choice(['A', 'B', 'C'], n_samples),
    'operator': np.random.choice(['X', 'Y', 'Z'], n_samples)
})

# Introduce some anomalies
data.loc[0:10, 'temperature'] += 20
data.loc[11:20, 'pressure'] -= 50

# Separate numerical and categorical columns
numerical_cols = ['temperature', 'pressure']
categorical_cols = ['machine_type', 'operator']

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[categorical_cols])

# Combine numerical and encoded categorical features
X = np.hstack([data[numerical_cols].values, encoded_categorical])

# Apply Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
y_pred = clf.fit_predict(X)

# Add predictions to the original dataframe
data['anomaly'] = y_pred

print("Sample of detected anomalies:")
print(data[data['anomaly'] == -1].head())
print(f"\nNumber of detected anomalies: {sum(y_pred == -1)}")
```

Slide 13: Evaluating Isolation Forest Performance

To assess the performance of Isolation Forest, we can use metrics such as precision, recall, and F1-score. This example demonstrates how to evaluate the algorithm's performance using a synthetic dataset with known anomalies.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

# Generate synthetic data with known anomalies
np.random.seed(42)
n_samples = 1000
n_anomalies = 50

X = np.random.randn(n_samples, 2)
y_true = np.ones(n_samples)

# Introduce anomalies
X[:n_anomalies] = np.random.uniform(low=-4, high=4, size=(n_anomalies, 2))
y_true[:n_anomalies] = -1

# Apply Isolation Forest
clf = IsolationForest(contamination=n_anomalies/n_samples, random_state=42)
y_pred = clf.fit_predict(X)

# Calculate performance metrics
precision = precision_score(y_true, y_pred, pos_label=-1)
recall = recall_score(y_true, y_pred, pos_label=-1)
f1 = f1_score(y_true, y_pred, pos_label=-1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal (Predicted)')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomaly (Predicted)')
plt.scatter(X[y_true == -1, 0], X[y_true == -1, 1], c='green', marker='x', s=200, label='True Anomaly')
plt.legend()
plt.title("Isolation Forest Performance Visualization")
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into Isolation Forest and anomaly detection techniques, here are some valuable resources:

1. Original Isolation Forest paper: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE. ArXiv: [https://arxiv.org/abs/1811.02141](https://arxiv.org/abs/1811.02141)
2. Extended Isolation Forest: Hariri, S., Carrasco Kind, M., & Brunner, R. J. (2019). Extended Isolation Forest. IEEE Transactions on Knowledge and Data Engineering, 33(4), 1479-1489. ArXiv: [https://arxiv.org/abs/1811.02141](https://arxiv.org/abs/1811.02141)
3. Comparison of Anomaly Detection Algorithms: Goldstein, M., & Uchida, S. (2016). A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data. PloS one, 11(4), e0152173. ArXiv: [https://arxiv.org/abs/1603.04240](https://arxiv.org/abs/1603.04240)

These resources provide in-depth explanations of the Isolation Forest algorithm, its extensions, and comparisons with other anomaly detection methods.

