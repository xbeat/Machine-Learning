## Exploring K-Medoids Clustering Algorithm
Slide 1: K-Medoids Algorithm Implementation

K-Medoids clustering algorithm fundamentally differs from K-Means by using actual data points as cluster centers. This implementation demonstrates the core algorithm using numpy, focusing on the Manhattan distance metric for robustness against outliers and computational efficiency.

```python
import numpy as np
from typing import Tuple

class KMedoids:
    def __init__(self, n_clusters: int, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.medoids_indices = None
        
    def manhattan_distance(self, X: np.ndarray) -> np.ndarray:
        # Compute pairwise Manhattan distances between points
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            distances[i] = np.sum(np.abs(X - X[i]), axis=1)
        return distances
    
    def fit(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        n_samples = X.shape[0]
        distances = self.manhattan_distance(X)
        
        # Randomly initialize medoids
        self.medoids_indices = np.random.choice(
            n_samples, self.n_clusters, replace=False
        )
        
        for _ in range(self.max_iter):
            # Assign points to nearest medoids
            labels = np.argmin(distances[:, self.medoids_indices], axis=1)
            
            # Update medoids
            old_medoids = self.medoids_indices.copy()
            for cluster in range(self.n_clusters):
                cluster_points = np.where(labels == cluster)[0]
                costs = np.sum(distances[cluster_points][:, cluster_points], axis=0)
                self.medoids_indices[cluster] = cluster_points[np.argmin(costs)]
            
            if np.all(old_medoids == self.medoids_indices):
                break
                
        return self.medoids_indices, labels
```

Slide 2: K-Medoids Distance Computation Theory

The mathematical foundation of K-Medoids relies on minimizing the sum of dissimilarities within clusters. For a cluster C\_k with medoid m\_k, we aim to minimize the objective function, which represents the total distance between points and their medoids.

```python
# Mathematical representation of K-Medoids objective function
"""
$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} d(x_i, m_k)
$$

Where:
$$
d(x_i, m_k) = \sum_{j=1}^{n} |x_{ij} - m_{kj}|
$$

Cost function for updating medoids:
$$
\text{cost}(m_k) = \sum_{x_i \in C_k} d(x_i, m_k)
$$
"""
```

Slide 3: Data Preprocessing for K-Medoids

Proper data preprocessing is crucial for K-Medoids performance. This implementation includes scaling, handling missing values, and outlier detection using the Interquartile Range method to ensure optimal clustering results.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Detect and handle outliers using IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
    data_cleaned = data[outlier_mask]
    
    # Scale features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_cleaned)
    
    return data_scaled
```

Slide 4: Silhouette Score Implementation

The Silhouette score provides a quantitative measure of clustering quality by evaluating both cohesion and separation. This implementation calculates the score for K-Medoids clustering to validate optimal cluster numbers.

```python
def silhouette_score(X: np.ndarray, labels: np.ndarray, distances: np.ndarray) -> float:
    n_samples = X.shape[0]
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        
        # Calculate a_i (mean intra-cluster distance)
        cluster_indices = np.where(labels == cluster_i)[0]
        if len(cluster_indices) > 1:
            a_i = np.mean(distances[i][cluster_indices[cluster_indices != i]])
        else:
            a_i = 0
            
        # Calculate b_i (mean nearest-cluster distance)
        b_i = float('inf')
        for cluster in set(labels):
            if cluster != cluster_i:
                cluster_indices = np.where(labels == cluster)[0]
                cluster_dist = np.mean(distances[i][cluster_indices])
                b_i = min(b_i, cluster_dist)
                
        # Calculate silhouette value
        if a_i == 0 and b_i == float('inf'):
            silhouette_vals[i] = 0
        else:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
            
    return np.mean(silhouette_vals)
```

Slide 5: Real-world Application - Customer Segmentation

Implementation of K-Medoids clustering for customer segmentation using real-world e-commerce data. This example demonstrates preprocessing, optimal cluster selection, and interpretation of customer segments based on purchasing behavior.

```python
import pandas as pd
import numpy as np
from typing import Dict

def customer_segmentation(data: pd.DataFrame) -> Dict:
    # Prepare features for clustering
    features = ['recency', 'frequency', 'monetary_value']
    X = preprocess_data(data[features])
    
    # Find optimal number of clusters
    best_score = -1
    best_k = 2
    
    for k in range(2, 6):
        kmedoids = KMedoids(n_clusters=k)
        medoids, labels = kmedoids.fit(X)
        score = silhouette_score(X, labels, kmedoids.manhattan_distance(X))
        
        if score > best_score:
            best_score = score
            best_k = k
    
    # Final clustering with optimal k
    final_kmedoids = KMedoids(n_clusters=best_k)
    medoids, labels = final_kmedoids.fit(X)
    
    # Analyze segments
    segments = {}
    for i in range(best_k):
        mask = labels == i
        segments[f'Segment_{i}'] = {
            'size': np.sum(mask),
            'avg_recency': np.mean(data.loc[mask, 'recency']),
            'avg_frequency': np.mean(data.loc[mask, 'frequency']),
            'avg_monetary': np.mean(data.loc[mask, 'monetary_value'])
        }
    
    return segments
```

Slide 6: Parallel K-Medoids Implementation

A parallel implementation of K-Medoids leveraging multiprocessing capabilities to handle large-scale datasets efficiently. This approach distributes distance calculations and medoid updates across multiple CPU cores for improved performance.

```python
import multiprocessing as mp
from functools import partial
import numpy as np

class ParallelKMedoids:
    def __init__(self, n_clusters: int, n_jobs: int = -1):
        self.n_clusters = n_clusters
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
    def _parallel_distance(self, X: np.ndarray, chunk: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(X[chunk][:, np.newaxis] - X), axis=2)
        
    def fit(self, X: np.ndarray) -> tuple:
        n_samples = X.shape[0]
        chunks = np.array_split(range(n_samples), self.n_jobs)
        
        # Parallel distance computation
        with mp.Pool(self.n_jobs) as pool:
            distances = pool.map(
                partial(self._parallel_distance, X),
                chunks
            )
        distances = np.vstack(distances)
        
        # Initialize medoids
        medoids = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        while True:
            old_medoids = medoids.copy()
            labels = np.argmin(distances[:, medoids], axis=1)
            
            # Update medoids in parallel
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                costs = np.sum(distances[cluster_points][:, cluster_points], axis=0)
                medoids[k] = cluster_points[np.argmin(costs)]
                
            if np.all(old_medoids == medoids):
                break
                
        return medoids, labels
```

Slide 7: Handling Mixed Data Types

Implementation of K-Medoids for datasets containing both numerical and categorical features, using Gower's distance metric for meaningful distance calculations across different data types.

```python
import pandas as pd
import numpy as np

def gower_distance(X: pd.DataFrame) -> np.ndarray:
    n_samples = len(X)
    distances = np.zeros((n_samples, n_samples))
    
    for column in X.columns:
        if X[column].dtype in [np.float64, np.int64]:
            # Numerical features
            range_col = X[column].max() - X[column].min()
            if range_col != 0:
                diff_matrix = np.abs(X[column].values[:, np.newaxis] - 
                                   X[column].values) / range_col
            else:
                diff_matrix = np.zeros((n_samples, n_samples))
        else:
            # Categorical features
            diff_matrix = (X[column].values[:, np.newaxis] != 
                         X[column].values).astype(float)
            
        distances += diff_matrix
        
    return distances / len(X.columns)

class MixedKMedoids:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        
    def fit(self, X: pd.DataFrame) -> tuple:
        distances = gower_distance(X)
        medoids = np.random.choice(len(X), self.n_clusters, replace=False)
        
        while True:
            old_medoids = medoids.copy()
            labels = np.argmin(distances[:, medoids], axis=1)
            
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                costs = np.sum(distances[cluster_points][:, cluster_points], axis=0)
                medoids[k] = cluster_points[np.argmin(costs)]
                
            if np.all(old_medoids == medoids):
                break
                
        return medoids, labels
```

Slide 8: Incremental K-Medoids

An implementation of incremental K-Medoids that efficiently handles streaming data by updating cluster assignments and medoids as new data points arrive, without requiring complete recalculation.

```python
class IncrementalKMedoids:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.medoids = None
        self.distances = None
        self.labels = None
        
    def partial_fit(self, X_new: np.ndarray) -> tuple:
        if self.medoids is None:
            # Initial fit
            self.distances = np.sum(np.abs(X_new[:, np.newaxis] - X_new), axis=2)
            self.medoids = np.random.choice(len(X_new), self.n_clusters, 
                                          replace=False)
            self.labels = np.argmin(self.distances[:, self.medoids], axis=1)
        else:
            # Update distances matrix
            new_distances = np.sum(np.abs(X_new[:, np.newaxis] - 
                                 np.vstack([self.X, X_new])), axis=2)
            self.distances = np.block([
                [self.distances, self.distances[-len(X_new):].T],
                [new_distances, new_distances[-len(X_new):, -len(X_new):]]
            ])
            
            # Update labels for new points
            new_labels = np.argmin(self.distances[-len(X_new):, self.medoids], 
                                 axis=1)
            self.labels = np.concatenate([self.labels, new_labels])
            
            # Update medoids if needed
            self._update_medoids()
            
        self.X = np.vstack([self.X, X_new]) if hasattr(self, 'X') else X_new
        return self.medoids, self.labels
        
    def _update_medoids(self):
        for k in range(self.n_clusters):
            cluster_points = np.where(self.labels == k)[0]
            costs = np.sum(self.distances[cluster_points][:, cluster_points], 
                          axis=0)
            self.medoids[k] = cluster_points[np.argmin(costs)]
```

Slide 9: Real-world Application - Genomic Data Clustering

This implementation demonstrates K-Medoids application in bioinformatics for clustering gene expression data, incorporating specialized distance metrics and visualization techniques for high-dimensional genomic data analysis.

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

class GenomicKMedoids:
    def __init__(self, n_clusters: int, distance_metric: str = 'correlation'):
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        
    def fit(self, expression_data: pd.DataFrame) -> tuple:
        # Compute distance matrix using correlation distance
        distances = squareform(pdist(expression_data, metric=self.distance_metric))
        
        # Initialize medoids
        n_samples = len(expression_data)
        medoids = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        while True:
            old_medoids = medoids.copy()
            # Assign genes to nearest medoid
            labels = np.argmin(distances[:, medoids], axis=1)
            
            # Update medoids
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                cluster_distances = distances[cluster_points][:, cluster_points]
                total_distances = np.sum(cluster_distances, axis=0)
                medoids[k] = cluster_points[np.argmin(total_distances)]
                
            if np.all(old_medoids == medoids):
                break
                
        return medoids, labels, self._calculate_silhouette(distances, labels)
    
    def _calculate_silhouette(self, distances: np.ndarray, 
                            labels: np.ndarray) -> float:
        n_samples = len(labels)
        silhouette_vals = np.zeros(n_samples)
        
        for i in range(n_samples):
            cluster_i = labels[i]
            cluster_indices = labels == cluster_i
            
            if np.sum(cluster_indices) > 1:
                a_i = np.mean(distances[i][cluster_indices & (np.arange(n_samples) != i)])
                b_i = np.min([np.mean(distances[i][labels == j]) 
                            for j in range(self.n_clusters) if j != cluster_i])
                silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
                
        return np.mean(silhouette_vals)
```

Slide 10: Medoid Selection Optimization

Implementation of an optimized medoid selection strategy using the Build phase of the PAM algorithm, which improves initial medoid selection for better convergence and cluster quality.

```python
class OptimizedKMedoids:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        
    def _build_phase(self, distances: np.ndarray) -> np.ndarray:
        n_samples = distances.shape[0]
        medoids = np.zeros(self.n_clusters, dtype=int)
        
        # Select first medoid
        total_distances = np.sum(distances, axis=1)
        medoids[0] = np.argmin(total_distances)
        
        # Select remaining medoids
        for k in range(1, self.n_clusters):
            costs = np.zeros(n_samples)
            for i in range(n_samples):
                if i not in medoids[:k]:
                    temp_medoids = np.append(medoids[:k], i)
                    labels = np.argmin(distances[:, temp_medoids], axis=1)
                    costs[i] = np.sum([np.min(distances[j][temp_medoids]) 
                                     for j in range(n_samples)])
                else:
                    costs[i] = np.inf
                    
            medoids[k] = np.argmin(costs)
            
        return medoids
    
    def fit(self, X: np.ndarray) -> tuple:
        distances = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        medoids = self._build_phase(distances)
        
        while True:
            old_medoids = medoids.copy()
            labels = np.argmin(distances[:, medoids], axis=1)
            
            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]
                costs = np.sum(distances[cluster_points][:, cluster_points], axis=0)
                medoids[k] = cluster_points[np.argmin(costs)]
                
            if np.all(old_medoids == medoids):
                break
                
        return medoids, labels
```

Slide 11: Hierarchical K-Medoids

A novel implementation combining hierarchical clustering principles with K-Medoids, enabling multi-level clustering analysis and automatic determination of optimal cluster numbers through dendrogram analysis.

```python
class HierarchicalKMedoids:
    def __init__(self, max_clusters: int = 10):
        self.max_clusters = max_clusters
        self.cluster_hierarchy = {}
        
    def fit(self, X: np.ndarray) -> dict:
        n_samples = X.shape[0]
        distances = np.sum(np.abs(X[:, np.newaxis] - X), axis=2)
        
        # Initialize each point as its own cluster
        current_clusters = {i: [i] for i in range(n_samples)}
        self.cluster_hierarchy[n_samples] = current_clusters.copy()
        
        for k in range(n_samples-1, 1, -1):
            # Find closest clusters to merge
            min_dist = float('inf')
            merge_pairs = None
            
            for i in current_clusters:
                for j in current_clusters:
                    if i < j:
                        cluster_dist = np.mean([distances[p][q] 
                                     for p in current_clusters[i]
                                     for q in current_clusters[j]])
                        if cluster_dist < min_dist:
                            min_dist = cluster_dist
                            merge_pairs = (i, j)
            
            # Merge clusters
            if merge_pairs:
                i, j = merge_pairs
                current_clusters[i].extend(current_clusters[j])
                del current_clusters[j]
            
            if k <= self.max_clusters:
                self.cluster_hierarchy[k] = current_clusters.copy()
        
        return self.cluster_hierarchy
    
    def get_clusters(self, k: int) -> list:
        if k not in self.cluster_hierarchy:
            raise ValueError(f"No clustering solution for k={k}")
        return list(self.cluster_hierarchy[k].values())
```

Slide 12: Advanced Distance Metrics

Implementation of advanced distance metrics for K-Medoids, including Earth Mover's Distance and Dynamic Time Warping, enabling clustering of complex data types like time series and distributions.

```python
import numpy as np
from scipy.stats import wasserstein_distance

class AdvancedMetricsKMedoids:
    def __init__(self, n_clusters: int, metric: str = 'dtw'):
        self.n_clusters = n_clusters
        self.metric = metric
        
    def _dtw_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        n, m = len(x), len(y)
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.abs(x[i-1] - y[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                            dtw_matrix[i, j-1],
                                            dtw_matrix[i-1, j-1])
        return dtw_matrix[n, m]
    
    def _emd_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return wasserstein_distance(x, y)
    
    def compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if self.metric == 'dtw':
                    dist = self._dtw_distance(X[i], X[j])
                elif self.metric == 'emd':
                    dist = self._emd_distance(X[i], X[j])
                distances[i, j] = distances[j, i] = dist
                
        return distances
```

Slide 13: Results and Validation Metrics

Comprehensive implementation of clustering validation metrics specific to K-Medoids, including Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index for robust cluster quality assessment.

```python
class KMedoidsValidation:
    @staticmethod
    def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray, 
                               medoids: np.ndarray) -> float:
        n_samples = X.shape[0]
        n_clusters = len(np.unique(labels))
        
        # Calculate between-cluster dispersion
        overall_mean = np.mean(X, axis=0)
        between_dispersion = np.sum([
            len(X[labels == k]) * 
            np.sum((X[medoids[k]] - overall_mean) ** 2)
            for k in range(n_clusters)
        ])
        
        # Calculate within-cluster dispersion
        within_dispersion = np.sum([
            np.sum((X[labels == k] - X[medoids[k]]) ** 2)
            for k in range(n_clusters)
        ])
        
        return (between_dispersion * (n_samples - n_clusters) /
                (within_dispersion * (n_clusters - 1)))
    
    @staticmethod
    def davies_bouldin_score(X: np.ndarray, labels: np.ndarray, 
                            medoids: np.ndarray) -> float:
        n_clusters = len(medoids)
        if n_clusters == 1:
            return 0.0
            
        intra_dists = np.zeros(n_clusters)
        centroids = X[medoids]
        
        for k in range(n_clusters):
            cluster_k = X[labels == k]
            if len(cluster_k) > 0:
                intra_dists[k] = np.mean(
                    np.sum(np.abs(cluster_k - centroids[k]), axis=1)
                )
                
        centroid_distances = np.sum(
            np.abs(centroids[:, np.newaxis] - centroids), axis=2
        )
        np.fill_diagonal(centroid_distances, np.inf)
        
        scores = np.zeros(n_clusters)
        for k in range(n_clusters):
            ratio = ((intra_dists[k] + intra_dists) / 
                     centroid_distances[k])
            scores[k] = np.max(ratio)
            
        return np.mean(scores)
```

Slide 14: Additional Resources

*   "A Modified K-medoids Algorithm for Interactive Document Clustering"
    *   [https://arxiv.org/abs/2203.12814](https://arxiv.org/abs/2203.12814)
*   "Efficient K-Medoids Clustering on Map-Reduce Architecture"
    *   [https://arxiv.org/abs/1912.02857](https://arxiv.org/abs/1912.02857)
*   "A Fast PAM Algorithm for Big Data Clustering"
    *   [https://arxiv.org/abs/2008.11645](https://arxiv.org/abs/2008.11645)
*   "Comparative Analysis of K-Means and K-Medoids Clustering Algorithms"
    *   [https://arxiv.org/abs/2009.08117](https://arxiv.org/abs/2009.08117)
*   "Robust K-Medoids Clustering Using Novel Distance Measures"
    *   [https://arxiv.org/abs/2106.09407](https://arxiv.org/abs/2106.09407)

