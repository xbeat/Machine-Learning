## Exploring Distance Metrics in K-Nearest Neighbors
Slide 1: Understanding Distance Metrics in KNN

In K-Nearest Neighbors, distance metrics play a crucial role in determining the similarity between data points. The most commonly used distance metric is Euclidean distance, which calculates the straight-line distance between two points in n-dimensional space using the Pythagorean theorem.

```python
import numpy as np

def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1, point2: Arrays of coordinates in n-dimensional space
    Returns:
        float: Euclidean distance between points
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Example usage
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
distance = euclidean_distance(p1, p2)
print(f"Euclidean distance: {distance:.2f}")  # Output: 5.20
```

Slide 2: Manhattan Distance Implementation

Manhattan distance, also known as L1 or city block distance, measures the sum of absolute differences between coordinates. This metric is particularly useful when dealing with grid-like features or when diagonal movement between points is not permitted in the feature space.

```python
def manhattan_distance(point1, point2):
    """
    Calculate Manhattan distance between two points
    
    Args:
        point1, point2: Arrays of coordinates in n-dimensional space
    Returns:
        float: Manhattan distance between points
    """
    return np.sum(np.abs(point1 - point2))

# Example usage
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
distance = manhattan_distance(p1, p2)
print(f"Manhattan distance: {distance}")  # Output: 9
```

Slide 3: Minkowski Distance - The Generalized Metric

The Minkowski distance is a generalization of both Euclidean and Manhattan distances, controlled by the parameter p. When p=2, it becomes Euclidean distance; when p=1, it becomes Manhattan distance; as p approaches infinity, it becomes Chebyshev distance.

```python
def minkowski_distance(point1, point2, p):
    """
    Calculate Minkowski distance between two points
    
    Args:
        point1, point2: Arrays of coordinates in n-dimensional space
        p: Order of the Minkowski distance
    Returns:
        float: Minkowski distance between points
    """
    return np.power(np.sum(np.power(np.abs(point1 - point2), p)), 1/p)

# Example usage
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
for p in [1, 2, 3]:
    distance = minkowski_distance(p1, p2, p)
    print(f"Minkowski distance (p={p}): {distance:.2f}")
```

Slide 4: Cosine Similarity for High-Dimensional Data

Cosine similarity measures the cosine of the angle between two vectors, making it particularly useful for high-dimensional data where the magnitude of the vectors is less important than their direction. This metric is commonly used in text analysis and recommendation systems.

```python
def cosine_similarity(point1, point2):
    """
    Calculate cosine similarity between two vectors
    
    Args:
        point1, point2: Arrays representing vectors
    Returns:
        float: Cosine similarity between vectors
    """
    dot_product = np.dot(point1, point2)
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2)
    return dot_product / (norm1 * norm2)

# Example usage
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
similarity = cosine_similarity(p1, p2)
print(f"Cosine similarity: {similarity:.4f}")  # Output: 0.9746
```

Slide 5: Custom KNN Implementation with Multiple Distance Metrics

```python
class KNN:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': lambda x, y: 1 - cosine_similarity(x, y)
        }
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.distance_functions[self.metric](x, x_train) 
                        for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            predictions.append(max(set(k_nearest_labels), 
                                key=k_nearest_labels.count))
        return np.array(predictions)
```

Slide 6: Real-world Example - Iris Classification

Understanding how different distance metrics perform on a real dataset helps in choosing the appropriate metric for specific problems. We'll implement KNN with various distance metrics on the iris dataset and compare their performance.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different metrics
metrics = ['euclidean', 'manhattan', 'cosine']
results = {}

for metric in metrics:
    knn = KNN(k=3, metric=metric)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    results[metric] = accuracy_score(y_test, y_pred)
```

Slide 7: Results for Iris Classification

The implementation results demonstrate the performance variations across different distance metrics when applied to the Iris dataset. Each metric's accuracy provides insights into their effectiveness for this specific classification problem.

```python
# Display results from previous slide
for metric, accuracy in results.items():
    print(f"{metric.capitalize()} Distance Accuracy: {accuracy:.4f}")
    
# Visualize decision boundaries
import matplotlib.pyplot as plt

def plot_decision_boundaries(X, y, metric):
    knn = KNN(k=3, metric=metric)
    knn.fit(X, y)
    
    # Create meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict for each point in meshgrid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(f'Decision Boundaries with {metric} Distance')
    plt.show()
```

Slide 8: Handling Missing Values in KNN

When dealing with real-world datasets, missing values pose a significant challenge. This implementation demonstrates how to handle missing values using mean imputation while maintaining the integrity of distance calculations.

```python
def weighted_distance(point1, point2, weights=None):
    """
    Calculate weighted distance considering missing values
    
    Args:
        point1, point2: Arrays with possible np.nan values
        weights: Optional array of feature weights
    Returns:
        float: Weighted distance between points
    """
    if weights is None:
        weights = np.ones(len(point1))
        
    # Create masks for non-missing values
    valid_mask = ~(np.isnan(point1) | np.isnan(point2))
    
    if not np.any(valid_mask):
        return np.inf
    
    # Calculate distance only for non-missing values
    valid_diff = (point1[valid_mask] - point2[valid_mask]) ** 2
    valid_weights = weights[valid_mask]
    
    # Normalize by number of valid features
    return np.sqrt(np.sum(valid_diff * valid_weights) / np.sum(valid_weights))

# Example with missing values
p1 = np.array([1, np.nan, 3, 4])
p2 = np.array([2, 3, np.nan, 5])
distance = weighted_distance(p1, p2)
print(f"Distance with missing values: {distance:.2f}")
```

Slide 9: Feature Scaling Impact on Distance Metrics

Feature scaling significantly affects distance-based algorithms. This implementation demonstrates different scaling techniques and their impact on distance calculations, essential for achieving optimal KNN performance.

```python
def scale_features(X, method='standard'):
    """
    Scale features using different methods
    
    Args:
        X: Input features array
        method: 'standard', 'minmax', or 'robust'
    Returns:
        scaled_X: Scaled features
        scaler: Fitted scaler object
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    
    scaled_X = scaler.fit_transform(X)
    return scaled_X, scaler

# Compare distances before and after scaling
original_dist = euclidean_distance(X_train[0], X_train[1])
scaled_dist = euclidean_distance(X_train_scaled[0], X_train_scaled[1])

print(f"Original distance: {original_dist:.2f}")
print(f"Scaled distance: {scaled_dist:.2f}")
```

Slide 10: Implementing KNN with Distance Weighting

Distance weighting in KNN assigns higher importance to closer neighbors, often improving classification accuracy. This implementation includes both uniform and distance-weighted voting schemes.

```python
class WeightedKNN:
    def __init__(self, k=3, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
    
    def _calculate_weights(self, distances):
        """Calculate weights based on distances"""
        if self.weights == 'uniform':
            return np.ones(len(distances))
        elif self.weights == 'distance':
            # Avoid division by zero
            return 1 / (distances + 1e-10)
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([self.distance_functions[self.metric](x, x_train) 
                                for x_train in self.X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_distances = distances[k_indices]
            k_weights = self._calculate_weights(k_distances)
            
            # Weighted voting
            k_nearest_labels = self.y_train[k_indices]
            unique_labels = np.unique(k_nearest_labels)
            weighted_votes = {label: np.sum(k_weights[k_nearest_labels == label])
                            for label in unique_labels}
            predictions.append(max(weighted_votes.items(), key=lambda x: x[1])[0])
        
        return np.array(predictions)
```

Slide 11: Real-world Example - Text Document Classification

Text classification presents unique challenges for distance metrics due to high dimensionality. This implementation shows how to apply KNN with cosine similarity for document classification using TF-IDF vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups

# Prepare text data
categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Transform documents to TF-IDF vectors
X_tfidf = vectorizer.fit_transform(newsgroups.data).toarray()
y_text = newsgroups.target

# Split data
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_tfidf, y_text, test_size=0.2, random_state=42
)

# Apply KNN with cosine similarity
text_knn = KNN(k=5, metric='cosine')
text_knn.fit(X_train_text, y_train_text)
text_pred = text_knn.predict(X_test_text)

print(f"Text classification accuracy: {accuracy_score(y_test_text, text_pred):.4f}")
```

Slide 12: Optimizing KNN with Ball Tree Data Structure

Ball Tree structure significantly improves KNN query time complexity from O(nd) to O(n log n) for high-dimensional data, making it especially useful for large datasets where traditional KNN becomes computationally expensive.

```python
class BallTreeNode:
    def __init__(self, points, labels=None):
        self.points = points
        self.labels = labels
        self.left = None
        self.right = None
        self.center = np.mean(points, axis=0)
        self.radius = max([euclidean_distance(p, self.center) for p in points])

def build_ball_tree(points, labels, leaf_size=10):
    """
    Build a Ball Tree for efficient nearest neighbor search
    
    Args:
        points: Array of data points
        labels: Array of corresponding labels
        leaf_size: Maximum number of points in leaf nodes
    Returns:
        BallTreeNode: Root node of the Ball Tree
    """
    node = BallTreeNode(points, labels)
    
    if len(points) <= leaf_size:
        return node
        
    # Find dimension with maximum variance
    dim = np.argmax(np.var(points, axis=0))
    median = np.median(points[:, dim])
    
    # Split points
    left_mask = points[:, dim] < median
    node.left = build_ball_tree(points[left_mask], labels[left_mask], leaf_size)
    node.right = build_ball_tree(points[~left_mask], labels[~left_mask], leaf_size)
    
    return node
```

Slide 13: Performance Analysis of Different Distance Metrics

A comprehensive analysis of different distance metrics' performance characteristics helps in selecting the most appropriate metric for specific use cases. This implementation provides timing and accuracy comparisons.

```python
import time
from sklearn.metrics import classification_report

def benchmark_metrics(X_train, X_test, y_train, y_test, k=3):
    """
    Benchmark different distance metrics for KNN
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        k: Number of neighbors
    Returns:
        dict: Performance metrics for each distance measure
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}
    
    for metric in metrics:
        start_time = time.time()
        knn = KNN(k=k, metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        end_time = time.time()
        
        results[metric] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'time': end_time - start_time,
            'report': classification_report(y_test, y_pred)
        }
        
        print(f"\nMetric: {metric}")
        print(f"Accuracy: {results[metric]['accuracy']:.4f}")
        print(f"Time: {results[metric]['time']:.4f} seconds")
        print("\nClassification Report:")
        print(results[metric]['report'])
    
    return results

# Run benchmark
benchmark_results = benchmark_metrics(X_train_scaled, X_test_scaled, 
                                   y_train, y_test)
```

Slide 14: Additional Resources

*   Search on Google Scholar for recent papers about KNN distance metrics optimization
*   Research directions:
    *   "A Survey of Binary Similarity and Distance Measures" - [https://www.jyotirmayeemahapatra.com/binary-distance-measures.html](https://www.jyotirmayeemahapatra.com/binary-distance-measures.html)
    *   "Distance Metric Learning: A Comprehensive Survey" - [https://www.cs.cmu.edu/~liuy/distlearn.html](https://www.cs.cmu.edu/~liuy/distlearn.html)
    *   "Adaptive Distance Metrics for Nearest Neighbor Classification" on arXiv
*   Recommended tools and libraries:
    *   scikit-learn KNN implementation documentation
    *   FAISS library for efficient similarity search
    *   Annoy library for approximate nearest neighbor search

