## Exploring K-Nearest Neighbors (KNN) in Python
Slide 1: Introduction to K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple yet powerful machine learning algorithm used for both classification and regression tasks. It operates on the principle that similar data points tend to have similar outcomes. KNN makes predictions for a new data point by examining the 'k' closest neighbors in the feature space and determining the majority class or average value among those neighbors.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)

# Create and train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Make a prediction for a new point
new_point = np.array([[0, 0]])
prediction = knn.predict(new_point)
print(f"Predicted class for new point: {prediction[0]}")
```

Slide 2: The Concept of Distance in KNN

In KNN, the notion of 'nearness' is typically determined by a distance metric. The most common distance metric is Euclidean distance, but others like Manhattan or Minkowski distance can also be used. The choice of distance metric can significantly impact the algorithm's performance and should be selected based on the nature of the data.

```python
import numpy as np
from scipy.spatial.distance import euclidean, manhattan, minkowski

point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])

euclidean_dist = euclidean(point1, point2)
manhattan_dist = manhattan(point1, point2)
minkowski_dist = minkowski(point1, point2, p=3)

print(f"Euclidean distance: {euclidean_dist}")
print(f"Manhattan distance: {manhattan_dist}")
print(f"Minkowski distance (p=3): {minkowski_dist}")
```

Slide 3: Choosing the Right 'k' Value

The 'k' in KNN represents the number of nearest neighbors to consider when making a prediction. A small 'k' can lead to overfitting, while a large 'k' might result in underfitting. The optimal 'k' value is often determined through cross-validation, balancing the trade-off between bias and variance.

```python
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Test different k values
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    cv_scores.append(scores.mean())

# Find the best k
best_k = k_values[np.argmax(cv_scores)]
print(f"Best k value: {best_k}")
```

Slide 4: KNN for Classification

In classification tasks, KNN predicts the class of a new data point by taking a majority vote among its k-nearest neighbors. This makes KNN particularly effective for multi-class problems where decision boundaries are irregular.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split the iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy:.2f}")
```

Slide 5: KNN for Regression

In regression tasks, KNN predicts the value of a new data point by averaging the values of its k-nearest neighbors. This approach is useful for problems where the relationship between features and the target variable is non-linear or complex.

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate a regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the KNN regressor
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = knn_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
```

Slide 6: Feature Scaling in KNN

KNN is sensitive to the scale of features. Features with larger scales can dominate the distance calculations, leading to poor performance. Standardization or normalization of features is often necessary to ensure all features contribute equally to the distance computation.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split the iris dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate KNN with scaled features
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with scaled features: {accuracy:.2f}")
```

Slide 7: Handling Imbalanced Data in KNN

Imbalanced datasets can pose challenges for KNN, as the algorithm may be biased towards the majority class. Techniques like oversampling, undersampling, or using class weights can help address this issue and improve the algorithm's performance on minority classes.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# Generate an imbalanced dataset
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train KNN on balanced data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_balanced, y_train_balanced)

# Evaluate the model
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
```

Slide 8: KNN for Dimensionality Reduction

KNN can be used for dimensionality reduction through techniques like Local Linear Embedding (LLE). LLE attempts to preserve the local structure of the data while projecting it into a lower-dimensional space, making it useful for visualization and as a preprocessing step for other algorithms.

```python
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply LLE
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_lle = lle.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_lle[:, 0], X_lle[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('LLE projection of the digits dataset')
plt.show()
```

Slide 9: KNN for Anomaly Detection

KNN can be adapted for anomaly detection by calculating the average distance to the k-nearest neighbors for each point. Points with significantly larger average distances are considered potential anomalies. This approach is particularly useful in scenarios where normal behavior is well-defined but anomalies are diverse and unpredictable.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs

# Generate a dataset with outliers
X, _ = make_blobs(n_samples=300, centers=1, random_state=42)
outliers = np.random.uniform(low=-15, high=15, size=(15, 2))
X = np.vstack([X, outliers])

# Fit KNN and calculate distances
k = 5
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X)
distances, _ = nn.kneighbors(X)

# Calculate average distance to k-nearest neighbors
avg_distances = np.mean(distances, axis=1)

# Identify potential anomalies
threshold = np.percentile(avg_distances, 95)
anomalies = X[avg_distances > threshold]

print(f"Number of potential anomalies: {len(anomalies)}")
```

Slide 10: KNN for Image Classification

KNN can be applied to image classification tasks by treating each pixel as a feature. While not as sophisticated as modern deep learning approaches, KNN can still perform reasonably well on simple image datasets and serves as a good baseline for comparison.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize a misclassified example
misclassified = X_test[y_test != y_pred]
misclassified_true = y_test[y_test != y_pred]
misclassified_pred = y_pred[y_test != y_pred]

plt.imshow(misclassified[0].reshape(8, 8), cmap='gray')
plt.title(f"True: {misclassified_true[0]}, Predicted: {misclassified_pred[0]}")
plt.show()
```

Slide 11: KNN for Recommender Systems

KNN can be used to build simple recommender systems by finding similar users or items based on their features or past interactions. This approach, known as collaborative filtering, can provide personalized recommendations without requiring deep understanding of the items' content.

```python
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Sample user-item interaction data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'item_id': [1, 2, 3, 2, 4, 1, 2, 5, 1, 5],
    'rating': [5, 4, 3, 5, 2, 4, 3, 1, 5, 2]
}
df = pd.DataFrame(data)

# Create user-item matrix
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

# Find similar users for user 1
distances, indices = knn.kneighbors(user_item_matrix.iloc[0:1], n_neighbors=3)

print("Similar users for user 1:")
for i, index in enumerate(indices[0]):
    print(f"User {user_item_matrix.index[index]}, Distance: {distances[0][i]:.2f}")
```

Slide 12: Limitations and Considerations of KNN

KNN, while simple and effective, has several limitations. It can be computationally expensive for large datasets, as it requires calculating distances to all training points for each prediction. The algorithm is also sensitive to irrelevant features and the curse of dimensionality. Additionally, KNN lacks a true training phase, making it a lazy learner that can be slow during prediction time.

```python
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification

# Generate datasets of increasing size
sizes = [1000, 10000, 100000]
training_times = []
prediction_times = []

for size in sizes:
    X, y = make_classification(n_samples=size, n_features=10, random_state=42)
    X_test = np.random.rand(100, 10)
    
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    training_times.append(time.time() - start_time)
    
    start_time = time.time()
    knn.predict(X_test)
    prediction_times.append(time.time() - start_time)

for i, size in enumerate(sizes):
    print(f"Dataset size: {size}")
    print(f"Training time: {training_times[i]:.4f} seconds")
    print(f"Prediction time: {prediction_times[i]:.4f} seconds")
    print()
```

Slide 13: Optimizing KNN Performance

To address some of KNN's limitations, various optimization techniques can be employed. These include using spatial data structures like KD-trees or Ball-trees for faster neighbor searches, applying dimensionality reduction techniques before KNN, and using approximate nearest neighbor algorithms for very large datasets.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

# Generate a large dataset
X, y = make_classification(n_samples=100000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare brute force vs. KD-tree
algorithms = ['brute', 'kd_tree']
for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=3, algorithm=algo)
    
    start_time = time.time()
    knn.fit(X_train, y_train)
    fit_time = time.time() - start_time
    
    start_time = time.time()
    knn.predict(X_test)
    predict_time = time.time() - start_time
    
    print(f"Algorithm: {algo}")
    print(f"Fit time: {fit_time:.4f} seconds")
    print(f"Predict time: {predict_time:.4f} seconds")
    print()
```

Slide 14: KNN in Real-World Applications

KNN finds applications in various real-world scenarios due to its simplicity and effectiveness. Two common use cases include:

1. Credit Scoring: Banks use KNN to assess credit risk by comparing new applicants to similar existing customers based on financial history, income, and other relevant factors.
2. Recommendation Systems: E-commerce platforms employ KNN to suggest products to users based on the purchasing history of similar customers, enhancing user experience and potentially increasing sales.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Simulated customer data (age, income, credit score)
customers = np.array([
    [25, 50000, 700],
    [35, 70000, 750],
    [45, 60000, 800],
    [55, 80000, 820],
    [30, 55000, 680]
])

# New applicant
new_applicant = np.array([[40, 65000, 730]])

# Find 3 nearest neighbors
nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
nn.fit(customers)
distances, indices = nn.kneighbors(new_applicant)

print("Indices of 3 most similar customers:", indices[0])
print("Distances to 3 most similar customers:", distances[0])
```

Slide 15: Future Directions and Advanced KNN Techniques

While KNN is a classic algorithm, research continues to enhance its capabilities and address its limitations. Some advanced techniques and future directions include:

1. Adaptive KNN: Dynamically adjusting the 'k' value based on local data density.
2. Fuzzy KNN: Incorporating fuzzy set theory to handle uncertainty in class assignments.
3. KNN with Deep Learning: Combining KNN with neural networks for feature learning and improved performance.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class AdaptiveKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k_min=1, k_max=20):
        self.k_min = k_min
        self.k_max = k_max

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        distances = euclidean_distances(X, self.X_)
        y_pred = []
        for dist in distances:
            k = self._adaptive_k(dist)
            k_nearest = np.argsort(dist)[:k]
            y_pred.append(np.argmax(np.bincount(self.y_[k_nearest])))
        return np.array(y_pred)

    def _adaptive_k(self, distances):
        # Simple adaptive k selection based on local density
        median_dist = np.median(distances)
        k = int(self.k_min + (self.k_max - self.k_min) * (1 - np.exp(-median_dist)))
        return max(self.k_min, min(k, self.k_max))

# Usage example (commented out to avoid execution)
# adaptive_knn = AdaptiveKNN(k_min=1, k_max=20)
# adaptive_knn.fit(X_train, y_train)
# y_pred = adaptive_knn.predict(X_test)
```

Slide 16: Additional Resources

For those interested in diving deeper into KNN and its applications, the following resources provide valuable insights:

1. "A Comprehensive Survey of Neighborhood-based Recommendation Methods" by Ning et al. (2015) ArXiv: [https://arxiv.org/abs/1505.01861](https://arxiv.org/abs/1505.01861)
2. "Nearest Neighbor Methods in Learning and Vision: Theory and Practice" by Shakhnarovich et al. (2006) MIT Press (not available on ArXiv)
3. "An Experimental Investigation of K-Nearest Neighbor as an Alternative to Maximum Likelihood Discriminant Analysis" by Edelbrock (1979) Educational and Psychological Measurement (not available on ArXiv)

These resources offer a mix of theoretical foundations and practical applications of KNN, providing a comprehensive understanding of the algorithm and its variants.

