## K-Means Clustering Algorithm Explained
Slide 1: K-Means Implementation from Scratch

The fundamental k-means clustering algorithm implements iterative refinement to partition n observations into k clusters. Each cluster is represented by the mean of its points, called the centroid. This implementation demonstrates the core algorithm without external libraries.

```python
import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            for i in range(self.k):
                self.centroids[i] = X[self.labels == i].mean(axis=0)
                
            # Check convergence
            if np.all(old_centroids == self.centroids):
                break
                
        return self.labels

# Example usage
X = np.random.randn(100, 2)  # Generate random 2D data
kmeans = KMeans(k=3)
labels = kmeans.fit(X)
```

Slide 2: Mathematical Foundations of K-Means

The k-means algorithm minimizes the within-cluster sum of squares (WCSS) through an objective function. This slide presents the mathematical formulation and demonstrates how to compute the objective function value.

```python
def compute_wcss(X, labels, centroids):
    """
    Mathematical formulation of K-means objective:
    '''
    $$WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$
    where:
    - k is number of clusters
    - Ci is the set of points in cluster i
    - μi is the centroid of cluster i
    """
    wcss = 0
    for i in range(len(np.unique(labels))):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

# Example usage
X = np.random.randn(100, 2)
kmeans = KMeans(k=3)
labels = kmeans.fit(X)
wcss = compute_wcss(X, labels, kmeans.centroids)
print(f"Within-cluster sum of squares: {wcss:.2f}")
```

Slide 3: Elbow Method Implementation

The elbow method helps determine the optimal number of clusters by plotting the WCSS against different k values. The "elbow" point represents diminishing returns in cluster compactness as k increases.

```python
import matplotlib.pyplot as plt

def plot_elbow_curve(X, k_range):
    wcss_values = []
    for k in k_range:
        kmeans = KMeans(k=k)
        labels = kmeans.fit(X)
        wcss = compute_wcss(X, labels, kmeans.centroids)
        wcss_values.append(wcss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss_values, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Example usage
    X = np.random.randn(200, 2)
    k_range = range(1, 11)
    plot_elbow_curve(X, k_range)
    plt.show()
```

Slide 4: Silhouette Analysis Implementation

Silhouette analysis measures how similar an object is to its own cluster compared to other clusters. The silhouette score ranges from -1 to 1, where higher values indicate better-defined clusters and optimal cluster separation.

```python
def silhouette_score(X, labels, centroids):
    def calculate_distance(point, cluster_points):
        return np.mean(np.sqrt(np.sum((cluster_points - point)**2, axis=1)))
    
    silhouette_scores = []
    for i, point in enumerate(X):
        # Find points in same cluster
        current_cluster = labels[i]
        cluster_points = X[labels == current_cluster]
        
        # Calculate a (average distance to points in same cluster)
        a = calculate_distance(point, cluster_points)
        
        # Calculate b (minimum average distance to points in different cluster)
        b = float('inf')
        for cluster in range(len(centroids)):
            if cluster != current_cluster:
                other_cluster_points = X[labels == cluster]
                avg_distance = calculate_distance(point, other_cluster_points)
                b = min(b, avg_distance)
        
        # Calculate silhouette score for this point
        silhouette = (b - a) / max(a, b)
        silhouette_scores.append(silhouette)
    
    return np.mean(silhouette_scores)

# Example usage
X = np.random.randn(100, 2)
kmeans = KMeans(k=3)
labels = kmeans.fit(X)
score = silhouette_score(X, labels, kmeans.centroids)
print(f"Silhouette Score: {score:.3f}")
```

Slide 5: K-Means++ Initialization

K-means++ initialization improves the standard k-means by selecting initial centroids that are far apart, leading to better convergence and final clustering results. This implementation demonstrates the probabilistic selection process.

```python
def kmeans_plus_plus_init(X, k):
    n_samples = X.shape[0]
    centroids = [X[np.random.randint(n_samples)]]
    
    for _ in range(1, k):
        # Calculate distances from points to nearest centroid
        distances = np.array([min([np.sum((x-c)**2) for c in centroids]) 
                            for x in X])
        
        # Choose next centroid with probability proportional to distance squared
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(X[j])
                break
    
    return np.array(centroids)

class KMeansPlusPlus(KMeans):
    def fit(self, X):
        self.centroids = kmeans_plus_plus_init(X, self.k)
        return super().fit(X)

# Example usage
X = np.random.randn(100, 2)
kmeans_pp = KMeansPlusPlus(k=3)
labels = kmeans_pp.fit(X)
```

Slide 6: Real-World Application - Customer Segmentation

The following implementation demonstrates customer segmentation using k-means clustering on customer purchase data. The example includes data preprocessing, scaling, and visualization of customer segments.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generate sample customer data
np.random.seed(42)
n_customers = 1000
customer_data = {
    'annual_income': np.random.normal(50000, 15000, n_customers),
    'spending_score': np.random.normal(50, 25, n_customers),
    'purchase_frequency': np.random.normal(10, 5, n_customers)
}
df = pd.DataFrame(customer_data)

# Preprocess and scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply k-means clustering
kmeans = KMeansPlusPlus(k=4)
labels = kmeans.fit(X_scaled)

# Visualize results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['annual_income'], df['spending_score'], 
                     c=labels, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.colorbar(scatter, label='Cluster')
plt.show()
```

Slide 7: Handling High-Dimensional Data

When dealing with high-dimensional data, k-means requires additional considerations for distance calculations and visualization. This implementation includes dimensionality reduction using PCA before clustering and visualization tools.

```python
import numpy as np
from sklearn.decomposition import PCA

class HighDimensionalKMeans:
    def __init__(self, k=3, n_components=2):
        self.k = k
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.kmeans = KMeansPlusPlus(k=k)
        
    def fit_transform(self, X):
        # Reduce dimensionality
        X_reduced = self.pca.fit_transform(X)
        
        # Apply k-means
        self.labels = self.kmeans.fit(X_reduced)
        
        # Store explained variance
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        return X_reduced, self.labels

    def plot_clusters(self, X_reduced):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                            c=self.labels, cmap='viridis')
        plt.xlabel(f'PC1 ({self.explained_variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({self.explained_variance_ratio[1]:.2%} variance)')
        plt.colorbar(scatter)
        plt.title('High-Dimensional Data Clusters')
        
# Example with 10-dimensional data
X = np.random.randn(500, 10)
hd_kmeans = HighDimensionalKMeans(k=4)
X_reduced, labels = hd_kmeans.fit_transform(X)
hd_kmeans.plot_clusters(X_reduced)
```

Slide 8: Mini-Batch K-Means Implementation

Mini-batch k-means processes subsets of the data in each iteration, making it more memory-efficient for large datasets. This implementation includes batch processing and incremental centroid updates.

```python
class MiniBatchKMeans:
    def __init__(self, k=3, batch_size=100, max_iters=100):
        self.k = k
        self.batch_size = batch_size
        self.max_iters = max_iters
        
    def fit(self, X):
        n_samples = X.shape[0]
        # Initialize centroids using k-means++
        self.centroids = kmeans_plus_plus_init(X, self.k)
        
        for iteration in range(self.max_iters):
            # Sample mini-batch
            indices = np.random.choice(n_samples, self.batch_size)
            batch = X[indices]
            
            # Assign samples to nearest centroids
            distances = np.sqrt(((batch - self.centroids[:, np.newaxis])**2).sum(axis=2))
            batch_labels = np.argmin(distances, axis=0)
            
            # Update centroids using moving average
            for i in range(self.k):
                batch_cluster = batch[batch_labels == i]
                if len(batch_cluster) > 0:
                    learning_rate = 1.0 / (iteration + 1)
                    self.centroids[i] = (1 - learning_rate) * self.centroids[i] + \
                                      learning_rate * batch_cluster.mean(axis=0)
        
        # Final assignment for all points
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        self.labels = np.argmin(distances, axis=0)
        return self.labels

# Example usage with large dataset
X_large = np.random.randn(10000, 2)
mini_batch = MiniBatchKMeans(k=5, batch_size=500)
labels = mini_batch.fit(X_large)
```

Slide 9: Cluster Validation Metrics

A comprehensive set of metrics to validate cluster quality, including Calinski-Harabasz Index and Davies-Bouldin Index, helps in assessing clustering performance beyond the silhouette score.

```python
def cluster_validation_metrics(X, labels, centroids):
    def calinski_harabasz_index(X, labels, centroids):
        n_samples = X.shape[0]
        n_clusters = len(centroids)
        
        # Between-cluster dispersion
        overall_mean = np.mean(X, axis=0)
        between_cluster_ss = sum(
            len(X[labels == i]) * np.sum((centroid - overall_mean) ** 2)
            for i, centroid in enumerate(centroids)
        )
        
        # Within-cluster dispersion
        within_cluster_ss = sum(
            np.sum((X[labels == i] - centroids[i]) ** 2)
            for i in range(n_clusters)
        )
        
        return (between_cluster_ss / (n_clusters - 1)) / \
               (within_cluster_ss / (n_samples - n_clusters))
    
    def davies_bouldin_index(X, labels, centroids):
        n_clusters = len(centroids)
        cluster_distances = np.zeros((n_clusters, n_clusters))
        cluster_dispersions = np.zeros(n_clusters)
        
        # Calculate cluster dispersions
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                cluster_dispersions[i] = np.mean(
                    np.linalg.norm(cluster_points - centroids[i], axis=1)
                )
        
        # Calculate davies-bouldin score
        score = 0
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    ratio = (cluster_dispersions[i] + cluster_dispersions[j]) / \
                            np.linalg.norm(centroids[i] - centroids[j])
                    max_ratio = max(max_ratio, ratio)
            score += max_ratio
        
        return score / n_clusters

    ch_index = calinski_harabasz_index(X, labels, centroids)
    db_index = davies_bouldin_index(X, labels, centroids)
    
    return {
        'calinski_harabasz_index': ch_index,
        'davies_bouldin_index': db_index
    }

# Example usage
X = np.random.randn(500, 2)
kmeans = KMeans(k=4)
labels = kmeans.fit(X)
metrics = cluster_validation_metrics(X, labels, kmeans.centroids)
print(f"Validation Metrics:\n{metrics}")
```

Slide 10: Online K-Means for Streaming Data

This implementation handles streaming data by updating clusters incrementally as new data points arrive. The algorithm maintains running statistics and adapts centroids in real-time without storing all historical data.

```python
class OnlineKMeans:
    def __init__(self, k=3):
        self.k = k
        self.n_samples = 0
        self.centroids = None
        self.counts = None
        
    def partial_fit(self, X):
        # Initialize centroids with first k points if not initialized
        if self.centroids is None:
            self.centroids = np.zeros((self.k, X.shape[1]))
            self.counts = np.zeros(self.k)
            for i in range(min(self.k, len(X))):
                self.centroids[i] = X[i]
                self.counts[i] = 1
                self.n_samples += 1
            return self
        
        # Process each new point
        for x in X:
            self.n_samples += 1
            # Find nearest centroid
            distances = np.sum((self.centroids - x) ** 2, axis=1)
            nearest_centroid = np.argmin(distances)
            
            # Update centroid
            self.counts[nearest_centroid] += 1
            lr = 1.0 / self.counts[nearest_centroid]
            self.centroids[nearest_centroid] += lr * (x - self.centroids[nearest_centroid])
        
        return self

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

# Example with streaming data
stream_kmeans = OnlineKMeans(k=3)

# Simulate data stream
for _ in range(5):
    batch = np.random.randn(100, 2)  # New batch of data
    stream_kmeans.partial_fit(batch)
    
    # Get predictions for current batch
    labels = stream_kmeans.predict(batch)
    
    # Plot current state
    plt.figure(figsize=(8, 6))
    plt.scatter(batch[:, 0], batch[:, 1], c=labels, cmap='viridis')
    plt.scatter(stream_kmeans.centroids[:, 0], stream_kmeans.centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    plt.title(f'Online K-Means after {stream_kmeans.n_samples} samples')
    plt.show()
```

Slide 11: Weighted K-Means Implementation

Weighted k-means assigns different importance to data points during clustering, useful when certain observations are more significant or reliable than others.

```python
class WeightedKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
    
    def fit(self, X, weights):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        weights = np.array(weights).reshape(-1, 1)
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids using weights
            for i in range(self.k):
                mask = (self.labels == i)
                if np.any(mask):
                    weighted_sum = np.sum(X[mask] * weights[mask], axis=0)
                    weight_sum = np.sum(weights[mask])
                    self.centroids[i] = weighted_sum / weight_sum
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
                
        return self.labels

# Example usage with weighted points
X = np.random.randn(200, 2)
# Generate weights based on distance from origin
weights = 1 / (1 + np.sqrt(np.sum(X**2, axis=1)))
weighted_kmeans = WeightedKMeans(k=3)
labels = weighted_kmeans.fit(X, weights)

# Visualize results with point sizes proportional to weights
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=weights*1000, cmap='viridis', alpha=0.6)
plt.scatter(weighted_kmeans.centroids[:, 0], weighted_kmeans.centroids[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title('Weighted K-Means Clustering')
plt.show()
```

Slide 12: Robust K-Means with Median Centers

This implementation uses medians instead of means for centroid calculation, making the algorithm more robust to outliers and non-spherical cluster shapes. The median-based approach provides better stability in the presence of noise.

```python
class RobustKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        
    def fit(self, X):
        # Initialize centroids using k-means++
        self.centroids = kmeans_plus_plus_init(X, self.k)
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids using median
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = np.median(cluster_points, axis=0)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
                
        return self.labels

# Example with outliers
np.random.seed(42)
# Generate core clusters
X = np.vstack([
    np.random.randn(100, 2),
    np.random.randn(100, 2) + [4, 4],
    np.random.randn(100, 2) + [-4, 4]
])
# Add outliers
outliers = np.random.uniform(-10, 10, (20, 2))
X = np.vstack([X, outliers])

# Compare regular and robust k-means
regular_kmeans = KMeans(k=3)
robust_kmeans = RobustKMeans(k=3)

regular_labels = regular_kmeans.fit(X)
robust_labels = robust_kmeans.fit(X)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X[:, 0], X[:, 1], c=regular_labels, cmap='viridis')
ax1.scatter(regular_kmeans.centroids[:, 0], regular_kmeans.centroids[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
ax1.set_title('Regular K-Means')

ax2.scatter(X[:, 0], X[:, 1], c=robust_labels, cmap='viridis')
ax2.scatter(robust_kmeans.centroids[:, 0], robust_kmeans.centroids[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
ax2.set_title('Robust K-Means')

plt.show()
```

Slide 13: Real-World Application - Image Segmentation

This implementation demonstrates k-means clustering for image segmentation, converting an image into a specified number of dominant colors. The example includes color space conversion and pixel clustering.

```python
import numpy as np
from PIL import Image

class ImageSegmentation:
    def __init__(self, k=5):
        self.k = k
        self.kmeans = KMeansPlusPlus(k=k)
        
    def segment_image(self, image_array):
        # Reshape image to 2D array of pixels
        pixels = image_array.reshape(-1, 3)
        
        # Apply k-means clustering
        labels = self.kmeans.fit(pixels)
        
        # Replace each pixel with its centroid color
        segmented_pixels = self.kmeans.centroids[labels]
        
        # Reshape back to original image dimensions
        return segmented_pixels.reshape(image_array.shape)

    @staticmethod
    def load_and_process_image(image_path):
        # Load image and convert to numpy array
        img = Image.open(image_path)
        img_array = np.array(img) / 255.0  # Normalize to [0,1]
        return img_array
    
    @staticmethod
    def save_segmented_image(segmented_array, output_path):
        # Convert back to 0-255 range and save
        img_array = (segmented_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_path)

# Example usage
def segment_image_example(image_path, k=5):
    segmenter = ImageSegmentation(k=k)
    
    # Load and process image
    img_array = segmenter.load_and_process_image(image_path)
    
    # Perform segmentation
    segmented_array = segmenter.segment_image(img_array)
    
    # Display original and segmented images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(img_array)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(segmented_array)
    ax2.set_title(f'Segmented Image (k={k})')
    ax2.axis('off')
    
    plt.show()

# Example call (assuming image path exists)
# segment_image_example('example_image.jpg', k=5)
```

Slide 14: Parallel K-Means Implementation

This implementation leverages multiprocessing to parallelize the computation of distances and cluster assignments, significantly improving performance for large datasets while maintaining clustering quality.

```python
import multiprocessing as mp
from functools import partial

class ParallelKMeans:
    def __init__(self, k=3, max_iters=100, n_jobs=-1):
        self.k = k
        self.max_iters = max_iters
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
    def _assign_clusters(self, X_chunk, centroids):
        distances = np.sqrt(((X_chunk - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def fit(self, X):
        # Initialize centroids using k-means++
        self.centroids = kmeans_plus_plus_init(X, self.k)
        
        # Split data into chunks for parallel processing
        chunk_size = len(X) // self.n_jobs
        chunks = [X[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()
            
            # Parallel assignment of clusters
            with mp.Pool(self.n_jobs) as pool:
                assign_func = partial(self._assign_clusters, centroids=self.centroids)
                labels_chunks = pool.map(assign_func, chunks)
            
            # Combine labels from all chunks
            self.labels = np.concatenate(labels_chunks)
            
            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
                
        return self.labels

# Performance comparison example
def compare_performance(X, k=3):
    # Standard K-means
    start_time = time.time()
    kmeans = KMeans(k=k)
    kmeans.fit(X)
    standard_time = time.time() - start_time
    
    # Parallel K-means
    start_time = time.time()
    parallel_kmeans = ParallelKMeans(k=k)
    parallel_kmeans.fit(X)
    parallel_time = time.time() - start_time
    
    print(f"Standard K-means time: {standard_time:.2f} seconds")
    print(f"Parallel K-means time: {parallel_time:.2f} seconds")
    print(f"Speedup: {standard_time/parallel_time:.2f}x")

# Example with large dataset
X_large = np.random.randn(100000, 10)
compare_performance(X_large, k=5)
```

Slide 15: Additional Resources

The following ArXiv papers provide comprehensive insights into k-means clustering algorithms, optimizations, and applications:

*   [https://arxiv.org/abs/1503.00900](https://arxiv.org/abs/1503.00900) - "Mini-batch k-means clustering of streaming and evolving data"
*   [https://arxiv.org/abs/1908.04664](https://arxiv.org/abs/1908.04664) - "A Survey of Clustering With Deep Learning: From the Perspective of Network Architecture"
*   [https://arxiv.org/abs/2002.11645](https://arxiv.org/abs/2002.11645) - "Accelerated k-means clustering algorithm using dimensionality reduction and parallelization"
*   [https://arxiv.org/abs/1912.00643](https://arxiv.org/abs/1912.00643) - "A Comprehensive Survey of Clustering Algorithms: State-of-the-Art Machine Learning Applications, Taxonomy, Challenges, and Future Research Prospects"
*   [https://arxiv.org/abs/1902.04938](https://arxiv.org/abs/1902.04938) - "Clustering with Deep Learning: Taxonomy and New Methods"

Slide 16: Results and Performance Metrics

```python
def generate_performance_report(X, algorithms):
    results = {}
    for name, algo in algorithms.items():
        # Time execution
        start_time = time.time()
        labels = algo.fit(X)
        execution_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'execution_time': execution_time,
            'inertia': compute_wcss(X, labels, algo.centroids),
            'silhouette': silhouette_score(X, labels, algo.centroids)
        }
        
        # Add validation metrics
        metrics.update(cluster_validation_metrics(X, labels, algo.centroids))
        results[name] = metrics
    
    # Format and display results
    print("Performance Comparison:")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    return results

# Example usage
X = np.random.randn(1000, 2)
algorithms = {
    'Standard K-means': KMeans(k=3),
    'K-means++': KMeansPlusPlus(k=3),
    'Mini-batch K-means': MiniBatchKMeans(k=3),
    'Robust K-means': RobustKMeans(k=3)
}

performance_results = generate_performance_report(X, algorithms)
```

