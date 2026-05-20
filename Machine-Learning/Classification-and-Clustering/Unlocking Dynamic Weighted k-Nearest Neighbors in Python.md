## Unlocking Dynamic Weighted k-Nearest Neighbors in Python
Slide 1: Introduction to k-Nearest Neighbours (KNN)

KNN is a simple yet powerful machine learning algorithm used for classification and regression tasks. It works by finding the k closest data points to a given query point and making predictions based on their labels or values.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Make a prediction
new_point = np.array([[2.5, 3.5]])
prediction = knn.predict(new_point)
print(f"Predicted class: {prediction[0]}")
```

Slide 2: How KNN Works

KNN operates by calculating the distance between a query point and all other points in the dataset. It then selects the k nearest neighbors and uses their labels to make a prediction, either through majority voting (for classification) or averaging (for regression).

```python
import numpy as np
from scipy.spatial.distance import euclidean

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [euclidean(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
        predictions.append(prediction)
    return predictions

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[2.5, 3.5]])

result = knn_predict(X_train, y_train, X_test, k=3)
print(f"Predicted class: {result[0]}")
```

Slide 3: Limitations of Traditional KNN

Traditional KNN has several limitations, including sensitivity to the choice of k, equal weighting of all neighbors, and vulnerability to the curse of dimensionality. These issues can lead to suboptimal performance in certain scenarios.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Generate a high-dimensional dataset
np.random.seed(42)
X = np.random.rand(100, 50)
y = np.random.randint(0, 2, 100)

# Evaluate KNN performance for different k values
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    print(f"k={k}, Average accuracy: {scores.mean():.3f}")
```

Slide 4: Introducing DynamicWeightedKNN

DynamicWeightedKNN is an enhanced version of the traditional KNN algorithm that addresses some of its limitations. It dynamically assigns weights to neighbors based on their distance from the query point and adapts the number of neighbors considered for each prediction.

Slide 5: Introducing DynamicWeightedKNN

```python
import numpy as np
from scipy.spatial.distance import euclidean

class DynamicWeightedKNN:
    def __init__(self, max_k):
        self.max_k = max_k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [euclidean(x, x_train) for x_train in self.X_train]
            sorted_indices = np.argsort(distances)
            
            # Dynamic k selection
            k = min(self.max_k, len(self.X_train))
            while k > 1 and distances[sorted_indices[k-1]] > 2 * distances[sorted_indices[0]]:
                k -= 1
            
            # Weight calculation
            weights = 1 / (np.array(distances[sorted_indices[:k]]) ** 2 + 1e-8)
            
            # Weighted voting
            class_votes = {}
            for i in range(k):
                vote = self.y_train[sorted_indices[i]]
                class_votes[vote] = class_votes.get(vote, 0) + weights[i]
            
            prediction = max(class_votes, key=class_votes.get)
            predictions.append(prediction)
        
        return np.array(predictions)

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[2.5, 3.5]])

dwknn = DynamicWeightedKNN(max_k=3)
dwknn.fit(X_train, y_train)
result = dwknn.predict(X_test)
print(f"Predicted class: {result[0]}")
```

Slide 6: Dynamic K Selection

DynamicWeightedKNN adaptively selects the number of neighbors to consider for each prediction. This approach helps to mitigate the impact of choosing a fixed k value and can improve performance in regions with varying data density.

```python
import numpy as np
import matplotlib.pyplot as plt

def dynamic_k_selection(distances, max_k):
    k = min(max_k, len(distances))
    while k > 1 and distances[k-1] > 2 * distances[0]:
        k -= 1
    return k

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 2)
distances = np.sort(np.random.rand(100))

# Plot dynamic k selection
max_k = 10
k_values = [dynamic_k_selection(distances[:i+1], max_k) for i in range(len(distances))]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(distances)), distances, alpha=0.5, label='Data points')
plt.plot(k_values, 'r-', label='Dynamic k')
plt.xlabel('Number of neighbors')
plt.ylabel('Distance')
plt.title('Dynamic K Selection')
plt.legend()
plt.ylim(0, 1)
plt.show()
```

Slide 7: Distance-based Weighting

DynamicWeightedKNN assigns weights to neighbors based on their distance from the query point. This approach gives more importance to closer neighbors, potentially improving the algorithm's performance.

```python
import numpy as np
import matplotlib.pyplot as plt

def distance_weight(distance):
    return 1 / (distance ** 2 + 1e-8)

# Generate sample distances
distances = np.linspace(0.1, 2, 100)

# Calculate weights
weights = distance_weight(distances)

# Plot distance-based weights
plt.figure(figsize=(10, 6))
plt.plot(distances, weights)
plt.xlabel('Distance')
plt.ylabel('Weight')
plt.title('Distance-based Weighting')
plt.grid(True)
plt.show()
```

Slide 8: Implementing DynamicWeightedKNN

Let's implement the core functionality of DynamicWeightedKNN, including dynamic k selection and distance-based weighting for classification tasks.

Slide 9: Implementing DynamicWeightedKNN

```python
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict

class DynamicWeightedKNN:
    def __init__(self, max_k):
        self.max_k = max_k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _get_neighbors(self, x):
        distances = [euclidean(x, x_train) for x_train in self.X_train]
        sorted_indices = np.argsort(distances)
        
        k = min(self.max_k, len(self.X_train))
        while k > 1 and distances[sorted_indices[k-1]] > 2 * distances[sorted_indices[0]]:
            k -= 1
        
        return sorted_indices[:k], distances[sorted_indices[:k]]
    
    def _weight_function(self, distance):
        return 1 / (distance ** 2 + 1e-8)
    
    def predict(self, X):
        predictions = []
        for x in X:
            neighbors, distances = self._get_neighbors(x)
            weights = self._weight_function(np.array(distances))
            
            class_votes = defaultdict(float)
            for i, neighbor in enumerate(neighbors):
                class_votes[self.y_train[neighbor]] += weights[i]
            
            prediction = max(class_votes, key=class_votes.get)
            predictions.append(prediction)
        
        return np.array(predictions)

# Example usage
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_train = np.array([0, 0, 1, 1, 1])
X_test = np.array([[2.5, 3.5], [4.5, 5.5]])

dwknn = DynamicWeightedKNN(max_k=3)
dwknn.fit(X_train, y_train)
predictions = dwknn.predict(X_test)
print(f"Predictions: {predictions}")
```

Slide 10: Comparing Traditional KNN and DynamicWeightedKNN

Let's compare the performance of traditional KNN and DynamicWeightedKNN on a simple dataset to highlight the differences between the two approaches.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a sample dataset
np.random.seed(42)
X = np.random.rand(1000, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Traditional KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# DynamicWeightedKNN
dwknn = DynamicWeightedKNN(max_k=5)
dwknn.fit(X_train, y_train)
dwknn_pred = dwknn.predict(X_test)
dwknn_accuracy = accuracy_score(y_test, dwknn_pred)

print(f"Traditional KNN accuracy: {knn_accuracy:.4f}")
print(f"DynamicWeightedKNN accuracy: {dwknn_accuracy:.4f}")
```

Slide 11: Visualizing Decision Boundaries

To better understand the differences between traditional KNN and DynamicWeightedKNN, let's visualize their decision boundaries on a 2D dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Generate a sample dataset
np.random.seed(42)
X = np.random.rand(200, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Traditional KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
plot_decision_boundary(X, y, knn, "Traditional KNN Decision Boundary")

# DynamicWeightedKNN
dwknn = DynamicWeightedKNN(max_k=5)
dwknn.fit(X, y)
plot_decision_boundary(X, y, dwknn, "DynamicWeightedKNN Decision Boundary")
```

Slide 12: Handling Imbalanced Datasets

DynamicWeightedKNN can be particularly useful for handling imbalanced datasets, where traditional KNN might struggle. Let's compare their performance on an imbalanced dataset.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1],
                           n_features=2, n_redundant=0, n_informative=2,
                           random_state=42, n_clusters_per_class=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Traditional KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_f1 = f1_score(y_test, knn_pred)

# DynamicWeightedKNN
dwknn = DynamicWeightedKNN(max_k=5)
dwknn.fit(X_train, y_train)
dwknn_pred = dwknn.predict(X_test)
dwknn_f1 = f1_score(y_test, dwknn_pred)

print(f"Traditional KNN F1-score: {knn_f1:.4f}")
print(f"DynamicWeightedKNN F1-score: {dwknn_f1:.4f}")
```

Slide 13: Real-life Example: Handwritten Digit Recognition

Let's apply DynamicWeightedKNN to a real-world problem: handwritten digit recognition using the MNIST dataset.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate DynamicWeightedKNN
dwknn = DynamicWeightedKNN(max_k=5)
dwknn.fit(X_train_scaled, y_train)
dwknn_pred = dwknn.predict(X_test_scaled)
dwknn_accuracy = accuracy_score(y_test, dwknn_pred)

print(f"DynamicWeightedKNN accuracy: {dwknn_accuracy:.4f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {dwknn_pred[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 14: Real-life Example: Iris Flower Classification

Let's use DynamicWeightedKNN for classifying iris flowers, a classic machine learning problem.

Slide 15: Real-life Example: Iris Flower Classification

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate DynamicWeightedKNN
dwknn = DynamicWeightedKNN(max_k=5)
dwknn.fit(X_train, y_train)
dwknn_pred = dwknn.predict(X_test)
dwknn_accuracy = accuracy_score(y_test, dwknn_pred)

print(f"DynamicWeightedKNN accuracy: {dwknn_accuracy:.4f}")

# Visualize decision regions (for two features)
feature1, feature2 = 0, 1  # Sepal length and sepal width
X_train_2d = X_train[:, [feature1, feature2]]
X_test_2d = X_test[:, [feature1, feature2]]

dwknn_2d = DynamicWeightedKNN(max_k=5)
dwknn_2d.fit(X_train_2d, y_train)

x_min, x_max = X[:, feature1].min() - 0.5, X[:, feature1].max() + 0.5
y_min, y_max = X[:, feature2].min() - 0.5, X[:, feature2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = dwknn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.xlabel(iris.feature_names[feature1])
plt.ylabel(iris.feature_names[feature2])
plt.title("DynamicWeightedKNN Decision Regions for Iris Classification")
plt.show()
```

Slide 16: Advantages and Limitations of DynamicWeightedKNN

DynamicWeightedKNN offers several advantages over traditional KNN, including adaptive neighbor selection and distance-based weighting. However, it also has some limitations to consider.

Advantages:

1. Adaptive to local data density
2. Reduced sensitivity to the choice of k
3. Improved handling of imbalanced datasets
4. Better performance in regions with varying data distribution

Slide 17: Advantages and Limitations of DynamicWeightedKNN

Limitations:

1. Increased computational complexity
2. May still struggle with high-dimensional data
3. Requires careful tuning of the max\_k parameter
4. Not suitable for very large datasets due to memory requirements

```python
# Pseudocode for comparing computational complexity
def traditional_knn_complexity(n_samples, n_features, k):
    return n_samples * n_features  # Distance calculation
           + n_samples * log(n_samples)  # Sorting
           + k  # Selecting k neighbors

def dynamic_weighted_knn_complexity(n_samples, n_features, max_k):
    return n_samples * n_features  # Distance calculation
           + n_samples * log(n_samples)  # Sorting
           + max_k  # Dynamic k selection
           + max_k  # Weight calculation
           + max_k  # Weighted voting

# Example usage
n_samples, n_features, k, max_k = 1000, 10, 5, 10
traditional_complexity = traditional_knn_complexity(n_samples, n_features, k)
dynamic_complexity = dynamic_weighted_knn_complexity(n_samples, n_features, max_k)

print(f"Traditional KNN complexity: {traditional_complexity}")
print(f"DynamicWeightedKNN complexity: {dynamic_complexity}")
```

Slide 18: Conclusion and Future Directions

DynamicWeightedKNN offers a promising improvement over traditional KNN by addressing some of its limitations. However, there is still room for further research and development in this area.

Potential future directions:

1. Exploring different weighting functions
2. Incorporating feature importance in the distance calculation
3. Developing efficient implementations for large-scale datasets
4. Combining DynamicWeightedKNN with other machine learning techniques
5. Investigating the theoretical properties of the algorithm

Slide 19: Conclusion and Future Directions

```python
# Pseudocode for a potential future improvement: Feature importance weighting
class FeatureWeightedDynamicKNN(DynamicWeightedKNN):
    def __init__(self, max_k, feature_weights=None):
        super().__init__(max_k)
        self.feature_weights = feature_weights
    
    def fit(self, X, y):
        super().fit(X, y)
        if self.feature_weights is None:
            # Calculate feature importance using some method
            self.feature_weights = calculate_feature_importance(X, y)
    
    def _weighted_distance(self, x1, x2):
        return np.sqrt(np.sum(self.feature_weights * (x1 - x2)**2))
    
    def _get_neighbors(self, x):
        distances = [self._weighted_distance(x, x_train) for x_train in self.X_train]
        # Rest of the method remains the same
        ...

# Example usage
fw_dwknn = FeatureWeightedDynamicKNN(max_k=5)
fw_dwknn.fit(X_train, y_train)
fw_dwknn_pred = fw_dwknn.predict(X_test)
```

Slide 20: Additional Resources

For those interested in diving deeper into the world of k-Nearest Neighbors and its variants, here are some valuable resources:

1. ArXiv paper on weighted k-Nearest Neighbors: "Weighted k-Nearest Neighbor Classification on Feature Projections" by Gönen et al. URL: [https://arxiv.org/abs/1112.0741](https://arxiv.org/abs/1112.0741)
2. ArXiv paper on adaptive k-Nearest Neighbors: "Adaptive k-Nearest Neighbor Classification Using Local Distance Functions" by Wang et al. URL: [https://arxiv.org/abs/1706.08561](https://arxiv.org/abs/1706.08561)
3. ArXiv paper on large-scale k-Nearest Neighbors: "Fast k-Nearest Neighbour Search via Dynamic Continuous Indexing" by Li et al. URL: [https://arxiv.org/abs/1512.00442](https://arxiv.org/abs/1512.00442)

These papers provide in-depth discussions on various improvements and extensions to the traditional k-Nearest Neighbors algorithm, which can inspire further enhancements to the DynamicWeightedKNN approach presented in this slideshow.

