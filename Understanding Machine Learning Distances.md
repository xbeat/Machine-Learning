## Understanding Machine Learning Distances
Slide 1: Euclidean Distance Implementation

The Euclidean distance represents the straight-line distance between two points in n-dimensional space. It's calculated as the square root of the sum of squared differences between corresponding elements. This metric is fundamental in k-means clustering and KNN algorithms.

```python
import numpy as np

def euclidean_distance(x1, x2):
    # Convert inputs to numpy arrays for vectorized operations
    x1, x2 = np.array(x1), np.array(x2)
    
    # Calculate squared differences and sum
    squared_diff = np.sum((x1 - x2) ** 2)
    
    # Return square root of sum
    return np.sqrt(squared_diff)

# Example usage
point1 = [1, 2, 3]
point2 = [4, 5, 6]
distance = euclidean_distance(point1, point2)
print(f"Euclidean distance between {point1} and {point2}: {distance:.2f}")
# Output: Euclidean distance between [1, 2, 3] and [4, 5, 6]: 5.20
```

Slide 2: Manhattan Distance Implementation

Manhattan distance, also known as L1 distance or city block distance, measures the sum of absolute differences between coordinates. This metric is particularly useful when movement is restricted to specific paths, like in grid-based systems.

```python
import numpy as np

def manhattan_distance(x1, x2):
    # Convert inputs to numpy arrays
    x1, x2 = np.array(x1), np.array(x2)
    
    # Calculate absolute differences and sum
    return np.sum(np.abs(x1 - x2))

# Example usage
point1 = [1, 2, 3]
point2 = [4, 5, 6]
distance = manhattan_distance(point1, point2)
print(f"Manhattan distance between {point1} and {point2}: {distance}")
# Output: Manhattan distance between [1, 2, 3] and [4, 5, 6]: 9
```

Slide 3: Minkowski Distance

The Minkowski distance is a generalized metric that encompasses both Euclidean (p=2) and Manhattan (p=1) distances. It provides flexibility in controlling the influence of individual differences through the parameter p, making it adaptable to various applications.

```python
def minkowski_distance(x1, x2, p):
    # Convert inputs to numpy arrays
    x1, x2 = np.array(x1), np.array(x2)
    
    # Calculate the p-th power of absolute differences
    diff = np.abs(x1 - x2) ** p
    
    # Return the p-th root of the sum
    return np.sum(diff) ** (1/p)

# Example usage with different p values
point1 = [1, 2, 3]
point2 = [4, 5, 6]
p_values = [1, 2, 3]

for p in p_values:
    dist = minkowski_distance(point1, point2, p)
    print(f"Minkowski distance (p={p}): {dist:.2f}")
```

Slide 4: Cosine Similarity Implementation

Cosine similarity measures the cosine of the angle between two non-zero vectors, making it invariant to vector magnitudes. This property makes it particularly useful in text analysis and recommendation systems where absolute values are less important than directional relationships.

```python
def cosine_similarity(x1, x2):
    # Convert inputs to numpy arrays
    x1, x2 = np.array(x1), np.array(x2)
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    
    # Return cosine similarity
    return dot_product / (norm_x1 * norm_x2)

# Example usage
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
similarity = cosine_similarity(vector1, vector2)
print(f"Cosine similarity: {similarity:.4f}")
# Convert to distance
cosine_distance = 1 - similarity
print(f"Cosine distance: {cosine_distance:.4f}")
```

Slide 5: Mahalanobis Distance

The Mahalanobis distance accounts for correlations between variables by incorporating the covariance matrix. This makes it particularly useful in detecting outliers and pattern recognition where features are correlated and have different scales.

```python
def mahalanobis_distance(x, data):
    # Calculate mean vector and covariance matrix
    mean = np.mean(data, axis=0)
    covariance_matrix = np.cov(data.T)
    
    # Calculate difference from mean
    diff = x - mean
    
    # Calculate Mahalanobis distance
    inv_covariance = np.linalg.inv(covariance_matrix)
    distance = np.sqrt(diff.dot(inv_covariance).dot(diff))
    
    return distance

# Example usage
data = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=1000
)
point = np.array([2, 2])
distance = mahalanobis_distance(point, data)
print(f"Mahalanobis distance: {distance:.4f}")
```

Slide 6: Hamming Distance for Binary Features

Hamming distance measures the number of positions at which two sequences differ, commonly used in information theory and binary feature comparison. This implementation demonstrates both binary array and string comparison methods.

```python
def hamming_distance(x1, x2):
    if len(x1) != len(x2):
        raise ValueError("Sequences must have equal length")
    
    # For binary arrays
    if isinstance(x1[0], (int, bool)):
        return sum(x1_i != x2_i for x1_i, x2_i in zip(x1, x2))
    # For strings
    return sum(c1 != c2 for c1, c2 in zip(x1, x2))

# Example usage
binary_seq1 = [1, 0, 1, 1, 0]
binary_seq2 = [1, 1, 0, 1, 0]
string1 = "hello"
string2 = "world"

print(f"Hamming distance (binary): {hamming_distance(binary_seq1, binary_seq2)}")
print(f"Hamming distance (string): {hamming_distance(string1, string2)}")
```

Slide 7: Haversine Distance Implementation

The Haversine formula calculates the great-circle distance between two points on a sphere, making it essential for geospatial applications. This implementation provides accurate distance calculations between latitude/longitude coordinates.

```python
import math

def haversine_distance(lat1, lon1, lat2, lon2, radius=6371):
    # Convert latitude and longitude to radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    # Haversine formula components
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Calculate distance
    return radius * c

# Example: Distance between New York and London
ny_lat, ny_lon = 40.7128, -74.0060
london_lat, london_lon = 51.5074, -0.1278

distance = haversine_distance(ny_lat, ny_lon, london_lat, london_lon)
print(f"Distance between New York and London: {distance:.2f} km")
```

Slide 8: K-Nearest Neighbors Implementation with Distance Metrics

Here's a practical implementation of KNN that allows for different distance metrics. This versatile implementation demonstrates how distance metrics affect classification results in real-world scenarios.

```python
class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': lambda x1, x2: 1 - cosine_similarity(x1, x2)
        }
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = [self.distance_functions[self.distance_metric](x, x_train) 
                        for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            prediction = max(set(k_nearest_labels), 
                           key=list(k_nearest_labels).count)
            predictions.append(prediction)
        
        return np.array(predictions)

# Example usage with sklearn dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate with different metrics
for metric in ['euclidean', 'manhattan', 'cosine']:
    knn = KNNClassifier(k=3, distance_metric=metric)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = sum(predictions == y_test) / len(y_test)
    print(f"Accuracy with {metric} distance: {accuracy:.4f}")
```

Slide 9: Real-world Application: Document Similarity Analysis

This implementation demonstrates document similarity analysis using TF-IDF vectorization and various distance metrics, commonly used in information retrieval and document clustering applications.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def document_similarity_analysis(documents):
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents).toarray()
    
    n_docs = len(documents)
    similarity_matrix = np.zeros((n_docs, n_docs))
    
    # Calculate similarity using multiple metrics
    for i in range(n_docs):
        for j in range(i, n_docs):
            euclidean = euclidean_distance(tfidf_matrix[i], tfidf_matrix[j])
            cosine = 1 - cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
            manhattan = manhattan_distance(tfidf_matrix[i], tfidf_matrix[j])
            
            similarity_matrix[i,j] = cosine  # Use cosine as default
            similarity_matrix[j,i] = similarity_matrix[i,j]
    
    return similarity_matrix

# Example usage
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks for pattern recognition",
    "Artificial intelligence revolutionizes technology",
    "Pattern recognition is crucial in computer vision"
]

similarities = document_similarity_analysis(documents)
print("Document Similarity Matrix:")
print(similarities)
```

Slide 10: Distance-based Anomaly Detection

Implementation of a distance-based anomaly detection system using multiple distance metrics to identify outliers in multivariate data, essential for fraud detection and system monitoring.

```python
class DistanceBasedAnomalyDetector:
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        
    def fit(self, X):
        self.X_train = np.array(X)
        self.mean = np.mean(X, axis=0)
        self.cov = np.cov(X.T)
        
    def detect_anomalies(self, X):
        X = np.array(X)
        anomalies = []
        
        for x in X:
            # Calculate different distance metrics
            euclidean_dist = euclidean_distance(x, self.mean)
            mahalanobis_dist = mahalanobis_distance(x, self.X_train)
            
            # Combine metrics for robust detection
            combined_score = (euclidean_dist + mahalanobis_dist) / 2
            
            anomalies.append(combined_score > self.threshold)
            
        return np.array(anomalies)

# Example usage with synthetic data
np.random.seed(42)
normal_data = np.random.normal(0, 1, (100, 2))
anomaly_data = np.random.normal(4, 1, (10, 2))
test_data = np.vstack([normal_data, anomaly_data])

detector = DistanceBasedAnomalyDetector(threshold=3.0)
detector.fit(normal_data)
results = detector.detect_anomalies(test_data)
print(f"Number of anomalies detected: {sum(results)}")
```

Slide 11: Clustering Quality Metrics

Implementation of various distance-based metrics to evaluate clustering quality, including silhouette score and Davies-Bouldin index, essential for assessing clustering algorithm performance.

```python
def clustering_quality_metrics(X, labels):
    def calculate_centroid(cluster_points):
        return np.mean(cluster_points, axis=0)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    centroids = []
    
    # Calculate centroids for each cluster
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroids.append(calculate_centroid(cluster_points))
    
    centroids = np.array(centroids)
    
    # Calculate average distances within clusters
    intra_cluster_distances = []
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        distances = [euclidean_distance(point, centroids[i]) 
                    for point in cluster_points]
        intra_cluster_distances.append(np.mean(distances))
    
    # Calculate inter-cluster distances
    inter_cluster_distances = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            distance = euclidean_distance(centroids[i], centroids[j])
            inter_cluster_distances.append(distance)
    
    metrics = {
        'avg_intra_cluster_distance': np.mean(intra_cluster_distances),
        'avg_inter_cluster_distance': np.mean(inter_cluster_distances),
        'davies_bouldin_score': np.mean(intra_cluster_distances) / 
                               np.mean(inter_cluster_distances)
    }
    
    return metrics

# Example usage with k-means clustering
from sklearn.cluster import KMeans

# Generate sample data
X = np.random.randn(200, 2)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate quality metrics
metrics = clustering_quality_metrics(X, labels)
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

Slide 12: Distance Metrics for Time Series Analysis

Implementation of specialized distance metrics for time series data, including Dynamic Time Warping (DTW) and Longest Common Subsequence (LCSS), essential for sequence analysis and pattern matching.

```python
def dtw_distance(sequence1, sequence2):
    n, m = len(sequence1), len(sequence2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill the DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(sequence1[i-1] - sequence2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
    
    return dtw_matrix[n, m]

def lcss_distance(sequence1, sequence2, epsilon=1.0):
    n, m = len(sequence1), len(sequence2)
    lcss_matrix = np.zeros((n + 1, m + 1))
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(sequence1[i-1] - sequence2[j-1]) < epsilon:
                lcss_matrix[i, j] = lcss_matrix[i-1, j-1] + 1
            else:
                lcss_matrix[i, j] = max(
                    lcss_matrix[i-1, j],
                    lcss_matrix[i, j-1]
                )
    
    return 1 - (lcss_matrix[n, m] / max(n, m))

# Example usage with synthetic time series
import numpy as np

# Generate sample time series
t = np.linspace(0, 10, 100)
series1 = np.sin(t)
series2 = np.sin(t + 0.5)

dtw_dist = dtw_distance(series1, series2)
lcss_dist = lcss_distance(series1, series2)

print(f"DTW distance: {dtw_dist:.4f}")
print(f"LCSS distance: {lcss_dist:.4f}")
```

Slide 13: Distance-Based Feature Selection

Implementation of a distance-based feature selection method that uses various distance metrics to identify the most discriminative features in a dataset.

```python
class DistanceBasedFeatureSelector:
    def __init__(self, n_features=5, metric='euclidean'):
        self.n_features = n_features
        self.metric = metric
        self.distance_functions = {
            'euclidean': euclidean_distance,
            'manhattan': manhattan_distance,
            'cosine': lambda x, y: 1 - cosine_similarity(x, y)
        }
    
    def calculate_feature_importance(self, X, y):
        n_samples, n_features = X.shape
        feature_scores = np.zeros(n_features)
        
        for feature_idx in range(n_features):
            within_class_distances = []
            between_class_distances = []
            
            for class_label in np.unique(y):
                class_samples = X[y == class_label][:, feature_idx]
                other_samples = X[y != class_label][:, feature_idx]
                
                # Calculate within-class distances
                for i in range(len(class_samples)):
                    for j in range(i + 1, len(class_samples)):
                        dist = self.distance_functions[self.metric](
                            [class_samples[i]], [class_samples[j]])
                        within_class_distances.append(dist)
                
                # Calculate between-class distances
                for i in range(len(class_samples)):
                    for j in range(len(other_samples)):
                        dist = self.distance_functions[self.metric](
                            [class_samples[i]], [other_samples[j]])
                        between_class_distances.append(dist)
            
            # Fisher-like criterion
            within_mean = np.mean(within_class_distances) if within_class_distances else 0
            between_mean = np.mean(between_class_distances) if between_class_distances else 0
            
            if within_mean == 0:
                feature_scores[feature_idx] = float('inf')
            else:
                feature_scores[feature_idx] = between_mean / within_mean
        
        return feature_scores
    
    def select_features(self, X, y):
        feature_scores = self.calculate_feature_importance(X, y)
        selected_indices = np.argsort(feature_scores)[-self.n_features:]
        return selected_indices

# Example usage
from sklearn.datasets import make_classification

# Generate sample dataset
X, y = make_classification(n_samples=100, n_features=20, 
                         n_informative=10, random_state=42)

# Select features
selector = DistanceBasedFeatureSelector(n_features=5)
selected_features = selector.select_features(X, y)
print(f"Selected feature indices: {selected_features}")
```

Slide 14: Additional Resources

*   ArXiv paper on distance metrics in machine learning: [https://arxiv.org/abs/1812.05944](https://arxiv.org/abs/1812.05944)
*   Survey of distance measures for time series analysis: [https://arxiv.org/abs/2201.12072](https://arxiv.org/abs/2201.12072)
*   Comprehensive review of similarity measures: [https://arxiv.org/abs/1903.00693](https://arxiv.org/abs/1903.00693)
*   Distance-based feature selection methods: [http://www.jmlr.org/papers/volume20/18-403/18-403.pdf](http://www.jmlr.org/papers/volume20/18-403/18-403.pdf)
*   Statistical distances for machine learning: [https://cs.cornell.edu/research/papers/statistical-distances.pdf](https://cs.cornell.edu/research/papers/statistical-distances.pdf)

