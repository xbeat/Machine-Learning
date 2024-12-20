## Unsupervised Learning Techniques for Clustering
Slide 1: Introduction to Unsupervised Learning and Clustering

Clustering is a fundamental unsupervised learning technique that groups similar data points together based on their intrinsic characteristics. The algorithm identifies patterns and structures within unlabeled data by measuring similarities between observations using distance metrics.

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate random data points
np.random.seed(42)
X = np.random.randn(300, 2)  # 300 points in 2D space

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           marker='x', color='red', s=200, label='Centroids')
plt.title('K-Means Clustering Example')
plt.legend()
plt.show()
```

Slide 2: K-Means Algorithm Implementation from Scratch

The K-means algorithm iteratively assigns data points to the nearest centroid and updates centroid positions. This implementation demonstrates the core mechanics of the algorithm without using external libraries, showcasing the mathematical foundations of clustering.

```python
import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            # Calculate distances to centroids
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # Assign points to nearest centroid
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # Check convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
        
        return self.labels

# Example usage
X = np.random.randn(100, 2)
kmeans = KMeansFromScratch(n_clusters=3)
labels = kmeans.fit(X)
```

Slide 3: Hierarchical Clustering Implementation

Hierarchical clustering builds a tree of clusters by recursively merging or splitting groups. This approach provides a dendrogram visualization showing the hierarchical relationship between clusters at different distance thresholds.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(50, 2)

# Compute linkage matrix
linkage_matrix = linkage(X, method='ward')

# Create dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Implementation of agglomerative clustering
def compute_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distances[i,j] = distances[j,i] = np.sqrt(np.sum((X[i] - X[j])**2))
    return distances
```

Slide 4: DBSCAN Clustering Implementation

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters based on density, capable of discovering clusters of arbitrary shapes and identifying noise points in the dataset.

```python
from sklearn.cluster import DBSCAN
import numpy as np

# Generate sample data with varying densities
np.random.seed(42)
centers = [[1, 1], [-1, -1], [1, -1]]
X = np.concatenate([
    np.random.randn(100, 2) * 0.3 + center 
    for center in centers
])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('DBSCAN Clustering Results')
plt.show()
```

Slide 5: Gaussian Mixture Models

Gaussian Mixture Models represent a probabilistic approach to clustering, modeling data as a mixture of several Gaussian distributions. Each cluster is characterized by its mean, covariance, and mixing coefficient.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Generate data from multiple Gaussian distributions
np.random.seed(42)
n_samples = 300

# Create mixture of 3 Gaussians
X = np.concatenate([
    np.random.normal(0, 1, (n_samples, 2)),
    np.random.normal(4, 1.5, (n_samples, 2)),
    np.random.normal(-4, 0.5, (n_samples, 2))
])

# Fit GMM
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')
plt.show()
```

Slide 6: Spectral Clustering Theory and Implementation

Spectral clustering leverages eigenvalues of the similarity matrix to perform dimensionality reduction before clustering. This technique excels at identifying complex, non-spherical cluster shapes by transforming the data into a spectral embedding space.

```python
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.neighbors import kneighbors_graph

# Generate non-linear data
t = np.linspace(0, 2*np.pi, 200)
X = np.column_stack([
    np.concatenate([np.cos(t), 0.5*np.cos(t) + 0.5]),
    np.concatenate([np.sin(t), 1.5*np.sin(t) + 0.5])
])

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, 
                             affinity='nearest_neighbors',
                             random_state=42)
labels = spectral.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Clustering Results')
plt.show()
```

Slide 7: Cluster Validation Metrics

When evaluating clustering algorithms, several metrics help assess the quality of cluster assignments. The Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index provide quantitative measures of clustering performance.

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from sklearn.cluster import KMeans

def evaluate_clustering(X, labels):
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels)
    }
    return metrics

# Generate sample data
X = np.random.randn(300, 2)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Calculate metrics
metrics = evaluate_clustering(X, labels)
for metric, score in metrics.items():
    print(f"{metric}: {score:.3f}")
```

Slide 8: Real-World Application: Customer Segmentation

Application of clustering algorithms to segment customers based on their purchasing behavior and demographics. This implementation demonstrates data preprocessing, feature scaling, and analysis of resulting segments.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample customer data
data = {
    'customer_id': range(1000),
    'age': np.random.normal(45, 15, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'spending_score': np.random.normal(50, 25, 1000),
    'frequency': np.random.poisson(10, 1000)
}
df = pd.DataFrame(data)

# Preprocess data
scaler = StandardScaler()
features = ['age', 'income', 'spending_score', 'frequency']
X_scaled = scaler.fit_transform(df[features])

# Apply clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Segment'] = kmeans.fit_predict(X_scaled)

# Analyze segments
segment_analysis = df.groupby('Segment')[features].mean()
print(segment_analysis)
```

Slide 9: Implementation of Advanced Distance Metrics

Distance metrics play a crucial role in clustering algorithms. This implementation showcases various distance measures including Euclidean, Manhattan, Cosine similarity, and Mahalanobis distance for robust cluster analysis.

```python
import numpy as np
from scipy.spatial.distance import cdist

class DistanceMetrics:
    @staticmethod
    def euclidean(X, Y):
        return cdist(X, Y, metric='euclidean')
    
    @staticmethod
    def manhattan(X, Y):
        return cdist(X, Y, metric='cityblock')
    
    @staticmethod
    def cosine_similarity(X, Y):
        return cdist(X, Y, metric='cosine')
    
    @staticmethod
    def mahalanobis(X, Y):
        covariance = np.cov(X.T)
        inv_covariance = np.linalg.inv(covariance)
        return cdist(X, Y, metric='mahalanobis', VI=inv_covariance)

# Example usage
X = np.random.randn(100, 2)
Y = np.random.randn(50, 2)

metrics = DistanceMetrics()
distances = {
    'euclidean': metrics.euclidean(X, Y),
    'manhattan': metrics.manhattan(X, Y),
    'cosine': metrics.cosine_similarity(X, Y),
    'mahalanobis': metrics.mahalanobis(X, Y)
}
```

Slide 10: Mini-batch K-means Implementation

Mini-batch K-means reduces computational complexity by using small random batches of data points in each iteration, making it suitable for large-scale clustering tasks while maintaining good convergence properties.

```python
import numpy as np

class MiniBatchKMeans:
    def __init__(self, n_clusters=3, batch_size=100, max_iters=100):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iters = max_iters
        
    def fit(self, X):
        n_samples = X.shape[0]
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            # Sample mini-batch
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            batch = X[batch_indices]
            
            # Assign clusters
            distances = np.sqrt(((batch - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Update centroids using batch
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    self.centroids[k] = np.mean(batch[labels == k], axis=0)
                    
        return self

# Example usage
X = np.random.randn(1000, 2)
mbk = MiniBatchKMeans(n_clusters=3)
mbk.fit(X)
```

Slide 11: Real-World Application: Image Segmentation using Clustering

Implementing clustering for image segmentation demonstrates practical application in computer vision. This technique reduces an image's color space to a specified number of clusters, effectively segmenting the image into distinct regions.

```python
import numpy as np
from sklearn.cluster import KMeans
import cv2

def segment_image(image_path, n_clusters=5):
    # Read and reshape image
    image = cv2.imread(image_path)
    pixels = image.reshape(-1, 3)
    
    # Apply clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Replace pixels with centroids
    segmented = kmeans.cluster_centers_[labels].reshape(image.shape)
    
    return segmented.astype(np.uint8)

# Example usage
image_path = 'sample_image.jpg'
segmented_image = segment_image(image_path)
cv2.imwrite('segmented_output.jpg', segmented_image)
```

Slide 12: Time Series Clustering Implementation

Time series clustering groups similar temporal sequences using Dynamic Time Warping (DTW) distance metric, particularly useful for analyzing sequential data patterns and identifying similar temporal behaviors.

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from dtaidistance import dtw
from dtaidistance import dtw_ndim

class TimeSeriesClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        
    def dtw_distance_matrix(self, sequences):
        n = len(sequences)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distance = dtw.distance(sequences[i], sequences[j])
                distances[i,j] = distances[j,i] = distance
                
        return distances
    
    def fit_predict(self, sequences):
        # Compute DTW distance matrix
        distances = self.dtw_distance_matrix(sequences)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances[np.triu_indices(len(distances), k=1)],
                               method='complete')
        
        # Extract clusters
        labels = fcluster(linkage_matrix, 
                         t=self.n_clusters, 
                         criterion='maxclust')
        
        return labels

# Generate sample time series data
n_series = 50
length = 100
t = np.linspace(0, 2*np.pi, length)
series = np.vstack([np.sin(t + np.random.normal(0, 0.1, length)) for _ in range(n_series)])

# Cluster time series
tsc = TimeSeriesClustering(n_clusters=3)
labels = tsc.fit_predict(series)
```

Slide 13: Ensemble Clustering Methods

Ensemble clustering combines multiple clustering solutions to create a more robust and stable final clustering. This implementation demonstrates consensus clustering using various base clustering algorithms.

```python
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import numpy as np
from scipy.stats import mode

class EnsembleClustering:
    def __init__(self, n_clusters=3, n_estimators=5):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.estimators = [
            KMeans(n_clusters=n_clusters),
            SpectralClustering(n_clusters=n_clusters),
            DBSCAN(eps=0.3, min_samples=5)
        ]
    
    def fit_predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.fit_predict(X)
            
        # Consensus voting
        final_labels, _ = mode(predictions, axis=1)
        return final_labels.ravel()

# Example usage
X = np.random.randn(200, 2)
ensemble = EnsembleClustering()
consensus_labels = ensemble.fit_predict(X)
```

Slide 14: Additional Resources

*   Arxiv Paper: "A Survey of Clustering Methods for Unsupervised Learning" - [https://arxiv.org/abs/2201.03146](https://arxiv.org/abs/2201.03146)
*   Research on Deep Clustering Techniques - [https://arxiv.org/abs/1801.07648](https://arxiv.org/abs/1801.07648)
*   Comparative Analysis of Clustering Algorithms - [https://arxiv.org/abs/2004.03149](https://arxiv.org/abs/2004.03149)
*   For implementation details and tutorials, search for:
    *   Scikit-learn clustering documentation
    *   Papers with Code - Clustering section
    *   Google Scholar: "Advanced Clustering Algorithms"

