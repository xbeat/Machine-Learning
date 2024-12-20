## 6 Types of Clustering Algorithms Beyond K-Means
Slide 1: Centroid-Based Clustering Overview

K-means clustering revolutionized data partitioning by iteratively assigning data points to their nearest centroids and updating centroid positions. This fundamental approach minimizes within-cluster variance through an expectation-maximization process, making it computationally efficient for large datasets.

```python
import numpy as np
from sklearn.datasets import make_blobs

def kmeans_scratch(X, k, max_iters=100):
    # Randomly initialize centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)
labels, centroids = kmeans_scratch(X, k=4)
```

Slide 2: Hierarchical Connectivity-Based Clustering

Agglomerative clustering builds a hierarchy of clusters by progressively merging the closest pairs of clusters. This method excels in revealing underlying hierarchical relationships and produces a dendrogram visualization showing the merging sequence and distances.

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from collections import defaultdict

def agglomerative_clustering(X, n_clusters):
    # Initialize each point as a cluster
    clusters = {i: [X[i]] for i in range(len(X))}
    history = []
    
    while len(clusters) > n_clusters:
        min_dist = float('inf')
        merge_pair = None
        
        # Find closest clusters
        for i in clusters:
            for j in clusters:
                if i < j:
                    dist = np.min([
                        np.linalg.norm(p1 - p2) 
                        for p1 in clusters[i] 
                        for p2 in clusters[j]
                    ])
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)
        
        # Merge clusters
        i, j = merge_pair
        clusters[i].extend(clusters[j])
        del clusters[j]
        history.append((i, j, min_dist))
    
    return clusters, history

# Example usage
X = np.random.rand(10, 2)
clusters, history = agglomerative_clustering(X, n_clusters=3)
```

Slide 3: Density-Based Spatial Clustering

DBSCAN identifies clusters by finding areas of high density separated by areas of low density. It excels at discovering clusters of arbitrary shapes and automatically identifying noise points, making it robust for real-world applications with non-spherical cluster patterns.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def dbscan(X, eps, min_samples):
    # Initialize labels as unvisited (-1)
    labels = np.full(len(X), -1)
    cluster_id = 0
    
    # Find neighbors for all points
    nbrs = NearestNeighbors(radius=eps).fit(X)
    neighbors = nbrs.radius_neighbors(X)[1]
    
    def expand_cluster(point_idx, neighbors_idx):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors_idx):
            nb = neighbors_idx[i]
            if labels[nb] == -1:
                labels[nb] = cluster_id
                new_neighbors = neighbors[nb]
                if len(new_neighbors) >= min_samples:
                    neighbors_idx = np.concatenate([neighbors_idx, new_neighbors])
            i += 1
    
    # Iterate through each point
    for point_idx in range(len(X)):
        if labels[point_idx] != -1:
            continue
            
        neighbors_idx = neighbors[point_idx]
        if len(neighbors_idx) < min_samples:
            labels[point_idx] = -2  # Noise
        else:
            expand_cluster(point_idx, neighbors_idx)
            cluster_id += 1
    
    return labels

# Example usage
X = np.random.rand(100, 2)
labels = dbscan(X, eps=0.3, min_samples=5)
```

Slide 4: Graph-Based Spectral Clustering

Spectral clustering leverages eigenvalues of the graph Laplacian matrix to perform dimensionality reduction before clustering. This technique excels at capturing complex cluster shapes by transforming the data into a space where cluster separation becomes more pronounced.

```python
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize

def spectral_clustering(X, n_clusters):
    # Compute affinity matrix
    distances = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
    sigma = np.median(distances)
    affinity = np.exp(-distances ** 2 / (2. * sigma ** 2))
    
    # Compute normalized Laplacian
    D = np.diag(np.sum(affinity, axis=1))
    L = D - affinity
    D_norm = np.diag(1.0 / np.sqrt(np.sum(affinity, axis=1)))
    L_norm = D_norm @ L @ D_norm
    
    # Compute eigenvectors
    eigenvalues, eigenvectors = eigsh(L_norm, k=n_clusters, which='SM')
    embedding = normalize(eigenvectors)
    
    # Apply k-means to the embedding
    _, labels = kmeans_scratch(embedding, n_clusters)
    return labels

# Example usage with synthetic data
X = np.random.rand(100, 2)
labels = spectral_clustering(X, n_clusters=3)
```

Slide 5: Distribution-Based Gaussian Mixture Models

Gaussian Mixture Models assume data points are generated from a mixture of multiple Gaussian distributions. This probabilistic approach enables soft clustering assignments and captures cluster shapes through covariance matrices, providing uncertainty estimates in cluster assignments.

```python
import numpy as np
from scipy.stats import multivariate_normal

def gmm_clustering(X, n_components, max_iters=100):
    n_samples, n_features = X.shape
    
    # Initialize parameters
    weights = np.ones(n_components) / n_components
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covs = [np.eye(n_features) for _ in range(n_components)]
    
    for _ in range(max_iters):
        # E-step: Calculate responsibilities
        resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            resp[:, k] = weights[k] * multivariate_normal.pdf(X, means[k], covs[k])
        resp /= resp.sum(axis=1, keepdims=True)
        
        # M-step: Update parameters
        Nk = resp.sum(axis=0)
        weights = Nk / n_samples
        means = np.array([np.sum(resp[:, k:k+1] * X, axis=0) / Nk[k] for k in range(n_components)])
        
        for k in range(n_components):
            diff = X - means[k]
            covs[k] = (resp[:, k:k+1].T @ (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])) / Nk[k]
    
    return resp.argmax(axis=1), resp

# Example usage
X = np.random.randn(200, 2)
labels, probabilities = gmm_clustering(X, n_components=3)
```

Slide 6: Compression-Based Clustering with Autoencoders

Neural network autoencoders perform clustering in latent space after compressing high-dimensional data. This approach combines dimensionality reduction with clustering, enabling the discovery of complex patterns in the compressed representation of the data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ClusteringAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, latent_dim))
        torch.nn.init.xavier_uniform_(self.cluster_layer)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        # Calculate cluster assignments
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer.unsqueeze(0), 2), 2
        ))
        q = (q.t() / torch.sum(q, 1)).t()
        
        return x_recon, q, z

# Example usage
input_dim = 784  # e.g., for MNIST
model = ClusteringAutoencoder(input_dim=input_dim, latent_dim=10, n_clusters=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
```

Slide 7: Real-World Application - Customer Segmentation

Customer segmentation analysis using multiple clustering approaches demonstrates the practical application of these algorithms. This implementation processes customer transaction data to identify distinct behavioral patterns and market segments for targeted marketing strategies.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def customer_segmentation(data_path):
    # Load and preprocess customer data
    df = pd.read_csv(data_path)
    features = ['recency', 'frequency', 'monetary_value']
    X = df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dimensionality reduction
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
    
    # Apply multiple clustering algorithms
    results = {}
    
    # K-means clustering
    kmeans_labels, _ = kmeans_scratch(X_reduced, k=4)
    results['kmeans'] = kmeans_labels
    
    # DBSCAN clustering
    dbscan_labels = dbscan(X_reduced, eps=0.3, min_samples=5)
    results['dbscan'] = dbscan_labels
    
    # GMM clustering
    gmm_labels, _ = gmm_clustering(X_reduced, n_components=4)
    results['gmm'] = gmm_labels
    
    return results, X_reduced

# Example usage
results, reduced_data = customer_segmentation('customer_data.csv')
```

Slide 8: Results Analysis for Customer Segmentation

```python
def analyze_clustering_results(results, data, original_features):
    analysis = {}
    
    for method, labels in results.items():
        # Calculate cluster statistics
        n_clusters = len(np.unique(labels[labels >= 0]))
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        # Calculate silhouette score
        silhouette = silhouette_score(data, labels)
        
        # Calculate cluster centroids
        centroids = []
        for i in range(n_clusters):
            cluster_data = data[labels == i]
            centroids.append(np.mean(cluster_data, axis=0))
        
        analysis[method] = {
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'silhouette_score': silhouette,
            'centroids': centroids
        }
    
    return analysis

# Calculate and display results
analysis_results = analyze_clustering_results(
    results, 
    reduced_data,
    ['recency', 'frequency', 'monetary_value']
)

print("Clustering Analysis Results:")
for method, metrics in analysis_results.items():
    print(f"\n{method.upper()} Results:")
    print(f"Number of clusters: {metrics['n_clusters']}")
    print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    print("Cluster sizes:", metrics['cluster_sizes'].tolist())
```

Slide 9: Advanced Applications - Image Segmentation

Image segmentation using clustering algorithms demonstrates their versatility in computer vision tasks. This implementation combines spectral clustering with image processing techniques to identify distinct regions in images.

```python
import cv2
import numpy as np
from sklearn.feature_extraction import image
from scipy.sparse.linalg import eigsh

def image_segmentation(image_path, n_segments=5):
    # Load and preprocess image
    img = cv2.imread(image_path)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Extract features
    height, width = img.shape[:2]
    grid_x, grid_y = np.mgrid[0:height:1, 0:width:1]
    
    # Combine spatial and color features
    features = np.concatenate([
        grid_x.reshape(-1, 1) / height,
        grid_y.reshape(-1, 1) / width,
        img_lab.reshape(-1, 3) / 255
    ], axis=1)
    
    # Apply spectral clustering
    affinity = image.img_to_graph(features.reshape((height, width, -1)))
    labels = spectral_clustering(features, n_segments)
    
    # Reshape labels to image dimensions
    segmented = labels.reshape(height, width)
    
    return segmented, img

# Example usage
segmented_image, original_image = image_segmentation('sample_image.jpg')
```

Slide 10: Time Series Clustering Implementation

Time series clustering requires specialized distance metrics and preprocessing steps to handle temporal dependencies. This implementation showcases Dynamic Time Warping (DTW) distance combined with density-based clustering for temporal pattern discovery.

```python
import numpy as np
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

def time_series_clustering(sequences, n_clusters=3, window_size=10):
    def dtw_distance_matrix(sequences):
        n = len(sequences)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distance, _ = fastdtw(sequences[i], sequences[j], window=window_size)
                distance_matrix[i,j] = distance
                distance_matrix[j,i] = distance
        
        return distance_matrix
    
    # Calculate DTW distance matrix
    distances = dtw_distance_matrix(sequences)
    
    # Apply spectral clustering on DTW distances
    adjacency = np.exp(-distances / distances.std())
    D = np.diag(adjacency.sum(axis=1))
    L = D - adjacency
    
    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    indices = np.argsort(eigenvalues)[:n_clusters]
    
    # Apply k-means on eigenvectors
    embedding = eigenvectors[:, indices]
    labels, _ = kmeans_scratch(embedding, n_clusters)
    
    return labels

# Generate sample time series data
def generate_sample_sequences(n_sequences=100, length=50):
    sequences = []
    for _ in range(n_sequences):
        # Generate different patterns
        pattern = np.random.choice(['sine', 'linear', 'exp'])
        t = np.linspace(0, 10, length)
        
        if pattern == 'sine':
            seq = np.sin(t + np.random.random())
        elif pattern == 'linear':
            seq = t * np.random.random()
        else:
            seq = np.exp(t/10 * np.random.random())
            
        sequences.append(seq + np.random.normal(0, 0.1, length))
    
    return sequences

# Example usage
sequences = generate_sample_sequences()
labels = time_series_clustering(sequences)
```

Slide 11: Performance Metrics Implementation

Comprehensive evaluation of clustering algorithms requires multiple metrics to assess different aspects of cluster quality. This implementation provides a suite of internal and external validation measures for clustering performance assessment.

```python
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, adjusted_rand_score

class ClusteringMetrics:
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels
        self.n_clusters = len(np.unique(labels[labels >= 0]))
        
    def calculate_all_metrics(self):
        metrics = {}
        
        # Silhouette Score
        metrics['silhouette'] = silhouette_score(self.X, self.labels)
        
        # Davies-Bouldin Index
        metrics['davies_bouldin'] = self._davies_bouldin_index()
        
        # Calinski-Harabasz Index
        metrics['calinski_harabasz'] = self._calinski_harabasz_index()
        
        # Dunn Index
        metrics['dunn'] = self._dunn_index()
        
        return metrics
        
    def _davies_bouldin_index(self):
        centroids = np.array([self.X[self.labels == i].mean(axis=0) 
                            for i in range(self.n_clusters)])
        
        # Calculate cluster dispersions
        dispersions = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            cluster_points = self.X[self.labels == i]
            dispersions[i] = np.mean(cdist(cluster_points, [centroids[i]]))
        
        # Calculate Davies-Bouldin Index
        db_index = 0
        for i in range(self.n_clusters):
            max_ratio = 0
            for j in range(self.n_clusters):
                if i != j:
                    ratio = (dispersions[i] + dispersions[j]) / \
                            np.linalg.norm(centroids[i] - centroids[j])
                    max_ratio = max(max_ratio, ratio)
            db_index += max_ratio
            
        return db_index / self.n_clusters
    
    def _calinski_harabasz_index(self):
        labels_unique = np.unique(self.labels)
        mean = np.mean(self.X, axis=0)
        
        bg_ss = 0
        wg_ss = 0
        
        for k in labels_unique:
            cluster_k = self.X[self.labels == k]
            centroid_k = cluster_k.mean(axis=0)
            
            bg_ss += len(cluster_k) * np.sum((centroid_k - mean) ** 2)
            wg_ss += np.sum((cluster_k - centroid_k) ** 2)
        
        return (bg_ss * (len(self.X) - self.n_clusters)) / \
               (wg_ss * (self.n_clusters - 1))

# Example usage
X = np.random.rand(100, 2)
labels = kmeans_scratch(X, k=3)[0]
metrics = ClusteringMetrics(X, labels)
results = metrics.calculate_all_metrics()
```

Slide 12: Online Clustering Implementation

Online clustering algorithms process data points sequentially, making them suitable for streaming data applications. This implementation demonstrates an online variant of k-means that updates cluster centroids incrementally as new data arrives.

```python
import numpy as np
from collections import defaultdict

class OnlineKMeans:
    def __init__(self, n_clusters, learning_rate=0.1):
        self.n_clusters = n_clusters
        self.learning_rate = learning_rate
        self.centroids = None
        self.counts = np.zeros(n_clusters)
        
    def partial_fit(self, X):
        # Initialize centroids with first n_clusters points
        if self.centroids is None:
            self.centroids = X[:self.n_clusters].copy()
            self.counts[:len(X)] += 1
            return self
        
        for x in X:
            # Find nearest centroid
            distances = np.linalg.norm(self.centroids - x, axis=1)
            nearest_centroid = np.argmin(distances)
            
            # Update centroid
            self.counts[nearest_centroid] += 1
            lr = self.learning_rate / self.counts[nearest_centroid]
            self.centroids[nearest_centroid] += lr * (x - self.centroids[nearest_centroid])
        
        return self
    
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Example with streaming data
def generate_stream(n_samples=1000, n_features=2):
    for _ in range(n_samples):
        # Generate point from one of three Gaussians
        cluster = np.random.choice(3)
        centers = np.array([[0, 0], [5, 5], [-5, 5]])
        point = centers[cluster] + np.random.randn(n_features)
        yield point

# Process streaming data
online_kmeans = OnlineKMeans(n_clusters=3)
stream_buffer = []

for i, point in enumerate(generate_stream()):
    stream_buffer.append(point)
    
    # Process in mini-batches
    if len(stream_buffer) == 10:
        online_kmeans.partial_fit(np.array(stream_buffer))
        stream_buffer = []

# Final predictions
X_test = np.random.randn(100, 2)
predictions = online_kmeans.predict(X_test)
```

Slide 13: Ensemble Clustering Implementation

Ensemble clustering combines multiple clustering solutions to create a more robust and stable final clustering. This implementation uses a consensus matrix approach to merge different clustering results.

```python
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class EnsembleClustering(BaseEstimator, ClusterMixin):
    def __init__(self, base_clusterers, n_clusters):
        self.base_clusterers = base_clusterers
        self.n_clusters = n_clusters
        
    def fit_predict(self, X):
        # Generate base clustering solutions
        base_labels = np.zeros((len(X), len(self.base_clusterers)))
        
        for i, clusterer in enumerate(self.base_clusterers):
            base_labels[:, i] = clusterer(X, self.n_clusters)[0]
            
        # Build consensus matrix
        consensus_matrix = np.zeros((len(X), len(X)))
        
        for labels in base_labels.T:
            # Update co-association matrix
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    if labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1
                        
        consensus_matrix /= len(self.base_clusterers)
        
        # Final clustering on consensus matrix
        final_labels = spectral_clustering(
            1 - consensus_matrix, 
            self.n_clusters
        )
        
        return final_labels

# Example usage
def create_ensemble():
    clusterers = [
        lambda X, k: kmeans_scratch(X, k),
        lambda X, k: spectral_clustering(X, k),
        lambda X, k: (dbscan(X, eps=0.3, min_samples=5), None)
    ]
    return clusterers

X = np.random.rand(100, 2)
ensemble = EnsembleClustering(create_ensemble(), n_clusters=3)
ensemble_labels = ensemble.fit_predict(X)
```

Slide 14: Advanced Density-Based Clustering Implementation

HDBSCAN extends traditional density-based clustering by creating a hierarchy of clusters using mutual reachability distance. This implementation shows the core algorithm components including distance calculation and cluster extraction.

```python
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

class HDBSCAN:
    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        
    def fit_predict(self, X):
        # Calculate core distances
        tree = KDTree(X)
        core_distances = tree.query(X, k=self.min_samples)[0][:, -1]
        
        # Compute mutual reachability distance
        distances = cdist(X, X)
        mutual_reach_dist = np.maximum(
            distances,
            np.maximum.outer(core_distances, core_distances)
        )
        
        # Build minimum spanning tree
        mst = self._compute_mst(mutual_reach_dist)
        
        # Extract hierarchy
        hierarchy = self._extract_hierarchy(mst)
        
        # Extract clusters
        labels = self._extract_clusters(hierarchy, len(X))
        
        return labels
    
    def _compute_mst(self, distances):
        n_points = len(distances)
        mst = np.zeros((n_points - 1, 3))
        
        # Prim's algorithm
        vertices = np.zeros(n_points, dtype=bool)
        vertices[0] = True
        edges = distances[0]
        
        for i in range(n_points - 1):
            min_idx = np.argmin(edges[~vertices])
            vertex_idx = np.arange(n_points)[~vertices][min_idx]
            
            vertices[vertex_idx] = True
            mst[i] = [i, vertex_idx, edges[vertex_idx]]
            
            mask = ~vertices
            edges[mask] = np.minimum(edges[mask], distances[vertex_idx][mask])
            
        return mst
        
    def _extract_hierarchy(self, mst):
        # Sort edges by weight
        sorted_edges = mst[np.argsort(mst[:, 2])]
        
        # Build hierarchy levels
        hierarchy = []
        current_component = np.arange(len(mst) + 1)
        
        for edge in sorted_edges:
            a, b = int(edge[0]), int(edge[1])
            size_a = np.sum(current_component == current_component[a])
            size_b = np.sum(current_component == current_component[b])
            
            if size_a >= self.min_cluster_size and size_b >= self.min_cluster_size:
                hierarchy.append({
                    'level': edge[2],
                    'component_a': current_component[a],
                    'component_b': current_component[b],
                    'size': size_a + size_b
                })
            
            # Merge components
            current_component[current_component == current_component[b]] = current_component[a]
            
        return hierarchy
    
    def _extract_clusters(self, hierarchy, n_points):
        if not hierarchy:
            return np.zeros(n_points, dtype=int)
            
        # Extract stable clusters
        stability = np.zeros(n_points)
        labels = np.arange(n_points)
        
        for level in hierarchy:
            mask = labels == level['component_b']
            labels[mask] = level['component_a']
            stability[labels == level['component_a']] += level['size'] * (1/level['level'])
        
        # Assign final cluster labels
        unique_labels = np.unique(labels)
        final_labels = np.zeros(n_points, dtype=int)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.sum(mask) >= self.min_cluster_size:
                final_labels[mask] = i + 1
                
        return final_labels - 1  # Convert to 0-based indexing

# Example usage
X = np.random.randn(100, 2)
clusterer = HDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(X)
```

Slide 15: Additional Resources

*   Comprehensive Survey: "A Comprehensive Survey of Clustering Algorithms" - [https://arxiv.org/abs/2009.00147](https://arxiv.org/abs/2009.00147)
*   Deep Clustering: "Deep Clustering: A Comprehensive Survey" - [https://arxiv.org/abs/2002.01333](https://arxiv.org/abs/2002.01333)
*   Ensemble Clustering: "Ensemble Clustering: A Review" - [https://arxiv.org/abs/1704.02280](https://arxiv.org/abs/1704.02280)
*   Time Series Clustering: "Time Series Clustering - A Decade Review" - [https://arxiv.org/abs/1910.09864](https://arxiv.org/abs/1910.09864)
*   Online Clustering: "Online Clustering Algorithms: A Systematic Review" - Search "online clustering algorithms review" on Google Scholar
*   Density-Based Clustering: "Density-Based Clustering: A Review" - [https://arxiv.org/abs/1801.07648](https://arxiv.org/abs/1801.07648)

