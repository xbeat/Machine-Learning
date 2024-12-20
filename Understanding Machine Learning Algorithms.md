## Understanding Machine Learning Algorithms
Slide 1: Introduction to Supervised Learning - Linear Regression

Linear regression serves as a foundational supervised learning algorithm for predicting continuous values. It establishes relationships between independent and dependent variables by fitting a linear equation to observed data points, minimizing the sum of squared residuals between predicted and actual values.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.randn(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 2

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Coefficient: {model.coef_[0][0]:.2f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
```

Slide 2: Mathematical Foundation of Linear Regression

The mathematical foundation of linear regression is built upon the optimization of parameters through minimizing the cost function. This involves calculating the sum of squared differences between predicted and actual values, leading to the optimal line of best fit.

```python
# Mathematical representation of Linear Regression
"""
Cost Function (Mean Squared Error):
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

Hypothesis Function:
$$h_\theta(x) = \theta_0 + \theta_1x$$

Gradient Descent Update:
$$\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$
"""

# Implementation from scratch
class LinearRegressionScratch:
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
            
            # Gradient descent
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

Slide 3: Logistic Regression Implementation

Logistic regression extends linear regression concepts to classification problems by applying a sigmoid function to the linear combination of features. This transformation allows the model to output probabilities between 0 and 1 for binary classification tasks.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create custom logistic regression
class LogisticRegressionCustom:
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
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        return np.round(y_pred)

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

Slide 4: Decision Trees from Scratch

Decision trees create a flowchart-like structure by recursively splitting the dataset based on feature values. This implementation demonstrates the fundamental concepts of decision tree construction using information gain and entropy calculations.

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
        
    def entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
        
    def split_node(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
        
    def information_gain(self, parent, left_child, right_child):
        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)
        return self.entropy(parent) - (left_weight * self.entropy(left_child) + 
                                     right_weight * self.entropy(right_child))
```

Slide 5: Results for Decision Tree Implementation

The decision tree implementation showcases practical classification results on a real dataset. This slide demonstrates the performance metrics and visualization of the decision boundaries created by our custom implementation.

```python
# Continuing from previous Decision Tree implementation
def build_tree(self, X, y, depth=0):
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # Stopping criteria
    if (self.max_depth <= depth or n_classes == 1):
        leaf_value = max(Counter(y).items(), key=lambda x: x[1])[0]
        return Node(value=leaf_value)
    
    # Find best split
    best_feature = None
    best_threshold = None
    best_gain = -1
    
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = self.split_node(X, y, feature, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
                
            gain = self.information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    # Create child nodes
    X_left, X_right, y_left, y_right = self.split_node(X, y, best_feature, best_threshold)
    left_child = self.build_tree(X_left, y_left, depth + 1)
    right_child = self.build_tree(X_right, y_right, depth + 1)
    
    return Node(best_feature, best_threshold, left_child, right_child)

# Test the implementation
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X, y)
predictions = tree.predict(X)
accuracy = np.sum(predictions == y) / len(y)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
```

Slide 6: K-Nearest Neighbors Implementation

K-Nearest Neighbors (KNN) operates by finding the k closest training examples in the feature space and making predictions based on majority voting for classification or averaging for regression. This implementation demonstrates distance calculations and neighbor selection.

```python
import numpy as np
from collections import Counter
from sklearn.metrics import euclidean_distances

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate distances
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            # Get k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and split data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# Train and evaluate
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"KNN Accuracy: {accuracy:.4f}")
```

Slide 7: Random Forest Ensemble Learning

Random Forests combine multiple decision trees to create a robust ensemble model. This implementation shows how to create a forest of trees with bootstrapped samples and random feature selection for improved generalization.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class CustomRandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]
        
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            # Get bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)
            # Train tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.array([
            np.bincount(tree_preds[:, sample]).argmax()
            for sample in range(tree_preds.shape[1])
        ])

# Example usage with iris dataset
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
rf = CustomRandomForest(n_trees=100, max_depth=5)
rf.fit(X, y)
predictions = rf.predict(X)
accuracy = np.mean(predictions == y)
print(f"Random Forest Accuracy: {accuracy:.4f}")
```

Slide 8: Neural Networks Fundamentals

Neural networks process information through interconnected layers of artificial neurons. This implementation demonstrates a basic feedforward neural network with backpropagation using numpy, showing the essential components of deep learning.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers=[3, 4, 1], learning_rate=0.1):
        self.layers = layers
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]))
            self.biases.append(np.random.randn(1, layers[i+1]))
    
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
    
    def backward(self, X, y, output):
        error = y - output
        delta = error * self.sigmoid_derivative(output)
        
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] += self.lr * np.dot(self.activations[i].T, delta)
            self.biases[i] += self.lr * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])

# Example usage
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([3, 4, 1])
for epoch in range(1000):
    output = nn.forward(X)
    nn.backward(X, y, output)
    
print("Final predictions:")
print(nn.forward(X))
```

Slide 9: Unsupervised Learning - K-Means Clustering

K-means clustering partitions data into k distinct groups by iteratively updating cluster centroids. This implementation shows the complete clustering process including centroid initialization and convergence checking.

```python
import numpy as np
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        
    def initialize_centroids(self, X):
        n_samples, n_features = X.shape
        indices = np.random.permutation(n_samples)[:self.k]
        return X[indices]
        
    def compute_distances(self, X, centroids):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        return distances
        
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to clusters
            distances = self.compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.mean(cluster_points, axis=0)
            
            # Check convergence
            if np.all(np.abs(old_centroids - self.centroids) < self.tol):
                break
                
        return self.labels
    
# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X)

# Plot results
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.show()
```

Slide 10: Hierarchical Clustering Implementation

Hierarchical clustering creates a tree of clusters by iteratively merging or splitting groups based on distance metrics. This implementation demonstrates agglomerative clustering with different linkage methods and dendrogram visualization.

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
    def compute_distance_matrix(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
                distances[j, i] = distances[i, j]
        return distances
    
    def merge_clusters(self, distances, clusters):
        n = len(clusters)
        min_dist = float('inf')
        merge_i, merge_j = 0, 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.linkage == 'single':
                    dist = np.min(distances[np.ix_(clusters[i], clusters[j])])
                elif self.linkage == 'complete':
                    dist = np.max(distances[np.ix_(clusters[i], clusters[j])])
                elif self.linkage == 'average':
                    dist = np.mean(distances[np.ix_(clusters[i], clusters[j])])
                    
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j
                    
        return merge_i, merge_j, min_dist
    
    def fit(self, X):
        n_samples = X.shape[0]
        distances = self.compute_distance_matrix(X)
        clusters = [[i] for i in range(n_samples)]
        
        self.linkage_matrix = []
        current_cluster = n_samples
        
        while len(clusters) > self.n_clusters:
            i, j, dist = self.merge_clusters(distances, clusters)
            new_cluster = clusters[i] + clusters[j]
            
            # Record merge for dendrogram
            self.linkage_matrix.append([clusters[i][0], clusters[j][0], 
                                      dist, len(new_cluster)])
            
            clusters.pop(max(i, j))
            clusters.pop(min(i, j))
            clusters.append(new_cluster)
            current_cluster += 1
            
        self.labels_ = np.zeros(n_samples)
        for i, cluster in enumerate(clusters):
            self.labels_[cluster] = i
            
        return self.labels_
    
    def plot_dendrogram(self):
        plt.figure(figsize=(10, 7))
        dendrogram(np.array(self.linkage_matrix))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()

# Example usage
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
hc = HierarchicalClustering(n_clusters=3)
labels = hc.fit(X)
hc.plot_dendrogram()
```

Slide 11: Principal Component Analysis

PCA reduces data dimensionality while preserving maximum variance. This implementation shows the complete PCA process including covariance matrix computation, eigenvalue decomposition, and projection.

```python
import numpy as np

class PCA:
    def __init__(self, n_components=2):
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
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        return self
        
    def transform(self, X):
        # Project data onto principal components
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
        
    def explained_variance_ratio(self):
        return self.explained_variance / np.sum(self.explained_variance)

# Example usage with visualization
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=iris.target)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio())
```

Slide 12: Semi-Supervised Learning with Label Propagation

Label propagation utilizes both labeled and unlabeled data by propagating labels through a similarity graph. This implementation demonstrates the iterative process of label spreading using a Gaussian kernel.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

class LabelPropagation:
    def __init__(self, kernel='rbf', gamma=1, max_iter=1000, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        
    def build_graph(self, X):
        if self.kernel == 'rbf':
            distances = squareform(pdist(X, 'sqeuclidean'))
            W = np.exp(-self.gamma * distances)
            return W
        
    def fit(self, X, y_partial):
        n_samples = X.shape[0]
        W = self.build_graph(X)
        
        # Normalize graph
        D = np.diag(np.sum(W, axis=1))
        D_inv = np.linalg.inv(D)
        T = np.dot(D_inv, W)
        
        # Initialize labels
        Y = np.zeros((n_samples, len(np.unique(y_partial[y_partial != -1]))))
        for i, label in enumerate(y_partial):
            if label != -1:
                Y[i, label] = 1
                
        Y_static = Y.copy()
        
        # Iterate until convergence
        for _ in range(self.max_iter):
            Y_old = Y.copy()
            Y = np.dot(T, Y)
            Y[y_partial != -1] = Y_static[y_partial != -1]
            
            # Check convergence
            if np.abs(Y - Y_old).sum() < self.tol:
                break
                
        self.label_distributions_ = Y
        self.labels_ = np.argmax(Y, axis=1)
        return self

# Example usage
np.random.seed(42)
X = np.random.randn(200, 2)
y_true = np.concatenate([np.zeros(100), np.ones(100)])
labeled_mask = np.random.choice(200, 20, replace=False)
y_partial = np.full(200, -1)
y_partial[labeled_mask] = y_true[labeled_mask]

model = LabelPropagation()
model.fit(X, y_partial)

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y_true)
plt.title('True Labels')
plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.title('Propagated Labels')
plt.show()
```

Slide 13: Deep Q-Network Implementation

Deep Q-Networks combine Q-learning with deep neural networks for reinforcement learning. This implementation shows a basic DQN agent for discrete action spaces with experience replay and target network.

```python
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state

        target = self.model.predict(states)
        target_next = self.target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target[i][action] = reward
            else:
                target[i][action] = reward + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Example usage (requires gym environment)
"""
import gym
env = gym.make('CartPole-v1')
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

for episode in range(100):
    state = env.reset()
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if len(agent.memory) > 32:
            agent.replay(32)
        if done:
            break
    if episode % 10 == 0:
        agent.update_target_model()
"""
```

Slide 14: Additional Resources

*   "Deep Learning" - arXiv:1404.7828 ([https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828))
*   "Reinforcement Learning: An Introduction" - [http://incompleteideas.net/book/RLbook2020.pdf](http://incompleteideas.net/book/RLbook2020.pdf)
*   "A Tutorial on Support Vector Machines for Pattern Recognition" - [https://www.microsoft.com/en-us/research/publication/tutorial-support-vector-machines-pattern-recognition/](https://www.microsoft.com/en-us/research/publication/tutorial-support-vector-machines-pattern-recognition/)
*   "Random Forests" - [https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
*   "Playing Atari with Deep Reinforcement Learning" - arXiv:1312.5602 ([https://arxiv.org/abs/1312.5602](https://arxiv.org/abs/1312.5602))

