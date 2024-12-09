## K-means & K-modes for numerical data
Slide 1: K-means Clustering Foundation

The K-means algorithm partitions n observations into k clusters by minimizing within-cluster variances. Each cluster is represented by its centroid, which is the mean of all points assigned to that cluster. The algorithm iteratively assigns points and updates centroids until convergence.

```python
import numpy as np
from sklearn.datasets import make_blobs

def kmeans_from_scratch(X, k, max_iters=100):
    # Randomly initialize centroids
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check convergence
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
labels, centroids = kmeans_from_scratch(X, k=4)
```

Slide 2: Mathematical Foundation of K-means

K-means optimization objective is to minimize the within-cluster sum of squares (WCSS), also known as inertia. This mathematical foundation helps understand the algorithm's behavior and limitations in clustering tasks.

```python
# Mathematical representation of K-means objective function
"""
$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$

Where:
- J is the objective function to minimize
- k is the number of clusters
- x represents a data point
- C_i is the i-th cluster
- μ_i is the centroid of cluster i
"""

def calculate_wcss(X, labels, centroids):
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss
```

Slide 3: Elbow Method Implementation

The elbow method determines the optimal number of clusters by plotting WCSS against different k values. The "elbow point" represents the best trade-off between cluster compactness and number of clusters.

```python
import matplotlib.pyplot as plt

def elbow_method(X, max_k=10):
    wcss_values = []
    k_values = range(1, max_k + 1)
    
    for k in k_values:
        labels, centroids = kmeans_from_scratch(X, k)
        wcss = calculate_wcss(X, labels, centroids)
        wcss_values.append(wcss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.title('Elbow Method')
    plt.grid(True)
    return wcss_values
```

Slide 4: K-modes Implementation

K-modes extends clustering to categorical data by replacing means with modes and using a different dissimilarity measure. This implementation handles categorical variables efficiently while maintaining the clustering concept.

```python
import pandas as pd
from collections import Counter

def calculate_modes(data, labels, k):
    modes = []
    for i in range(k):
        cluster_data = data[labels == i]
        cluster_modes = []
        for column in cluster_data.T:
            mode = Counter(column).most_common(1)[0][0]
            cluster_modes.append(mode)
        modes.append(cluster_modes)
    return np.array(modes)

def kmodes_from_scratch(data, k, max_iters=100):
    n_samples, n_features = data.shape
    
    # Initialize centroids
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Calculate dissimilarity matrix
        dissim_matrix = np.zeros((k, n_samples))
        for i, centroid in enumerate(centroids):
            dissim_matrix[i] = np.sum(data != centroid, axis=1)
        
        # Assign points to nearest centroid
        labels = np.argmin(dissim_matrix, axis=0)
        
        # Update modes
        new_centroids = calculate_modes(data, labels, k)
        
        if np.array_equal(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids
```

Slide 5: Data Preprocessing for K-means and K-modes

Data preprocessing is crucial for clustering performance. For K-means, numerical features require standardization, while K-modes needs categorical encoding. This implementation demonstrates proper preprocessing techniques for both algorithms.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_mixed_data(df, numerical_cols, categorical_cols):
    # Preprocess numerical data for K-means
    scaler = StandardScaler()
    df_num = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), 
                         columns=numerical_cols)
    
    # Preprocess categorical data for K-modes
    df_cat = df[categorical_cols].copy()
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_cat[col] = label_encoders[col].fit_transform(df_cat[col])
    
    return df_num, df_cat, label_encoders

# Example usage
data = pd.DataFrame({
    'age': [25, 35, 45, 20, 30],
    'income': [30000, 45000, 60000, 25000, 40000],
    'education': ['HS', 'BS', 'MS', 'PhD', 'BS'],
    'occupation': ['A', 'B', 'A', 'C', 'B']
})

num_cols = ['age', 'income']
cat_cols = ['education', 'occupation']
df_num, df_cat, encoders = preprocess_mixed_data(data, num_cols, cat_cols)
```

Slide 6: Silhouette Analysis Implementation

Silhouette analysis provides a quantitative way to evaluate clustering quality by measuring how similar an object is to its own cluster compared to other clusters, helping validate the optimal number of clusters.

```python
import numpy as np

def silhouette_score_from_scratch(X, labels):
    n_samples = len(X)
    n_clusters = len(np.unique(labels))
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Calculate a (mean intra-cluster distance)
        cluster_i = labels[i]
        a_i = np.mean([np.linalg.norm(X[i] - X[j]) 
                      for j in range(n_samples) 
                      if labels[j] == cluster_i and i != j])
        
        # Calculate b (mean nearest-cluster distance)
        b_i = float('inf')
        for cluster in range(n_clusters):
            if cluster != cluster_i:
                cluster_dist = np.mean([np.linalg.norm(X[i] - X[j]) 
                                      for j in range(n_samples) 
                                      if labels[j] == cluster])
                b_i = min(b_i, cluster_dist)
        
        # Calculate silhouette score
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
    
    return np.mean(silhouette_vals)

# Example usage
X, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)
labels, _ = kmeans_from_scratch(X, k=4)
silhouette_avg = silhouette_score_from_scratch(X, labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")
```

Slide 7: Real-world Application: Customer Segmentation

Customer segmentation using K-means helps identify distinct customer groups based on their behavior patterns. This implementation demonstrates clustering customers based on their purchase history and demographic data.

```python
# Generate sample customer data
np.random.seed(42)
n_customers = 1000

customer_data = pd.DataFrame({
    'annual_income': np.random.normal(50000, 20000, n_customers),
    'spending_score': np.random.normal(50, 25, n_customers),
    'age': np.random.normal(35, 12, n_customers),
    'loyalty_years': np.random.normal(5, 3, n_customers)
})

def customer_segmentation(data, n_clusters=5):
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Apply K-means
    labels, centroids = kmeans_from_scratch(scaled_data, k=n_clusters)
    
    # Analyze segments
    data['Segment'] = labels
    segment_stats = data.groupby('Segment').agg({
        'annual_income': 'mean',
        'spending_score': 'mean',
        'age': 'mean',
        'loyalty_years': 'mean',
    }).round(2)
    
    return segment_stats, labels

segment_stats, customer_segments = customer_segmentation(customer_data)
print("\nCustomer Segments Statistics:")
print(segment_stats)
```

Slide 8: Real-world Application: Image Color Quantization

K-means clustering can effectively reduce the number of colors in an image while maintaining visual quality. This implementation demonstrates color quantization by treating each pixel as a point in RGB color space.

```python
import cv2

def color_quantization(image_path, k=8):
    # Read and reshape image
    image = cv2.imread(image_path)
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Apply K-means to colors
    labels, centroids = kmeans_from_scratch(pixels, k)
    
    # Reconstruct image with reduced colors
    quantized_pixels = centroids[labels].astype(np.uint8)
    quantized_image = quantized_pixels.reshape(image.shape)
    
    return quantized_image

def calculate_compression_ratio(original, quantized):
    original_colors = len(np.unique(original.reshape(-1, 3), axis=0))
    quantized_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
    return original_colors / quantized_colors

# Example usage
original_image = cv2.imread('sample_image.jpg')
quantized_image = color_quantization(original_image, k=16)
ratio = calculate_compression_ratio(original_image, quantized_image)
print(f"Compression ratio: {ratio:.2f}x")
```

Slide 9: Handling Empty Clusters

Empty clusters can occur during K-means iteration when no points are assigned to a centroid. This implementation provides a robust solution by reinitializing empty clusters from the farthest points.

```python
def kmeans_with_empty_cluster_handling(X, k, max_iters=100):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        # Assign points to clusters
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Handle empty clusters
        for i in range(k):
            if np.sum(labels == i) == 0:
                # Find the point farthest from its assigned centroid
                current_distances = np.min(distances, axis=0)
                farthest_point_idx = np.argmax(current_distances)
                centroids[i] = X[farthest_point_idx]
                
                # Reassign points
                distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
                labels = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids
```

Slide 10: Parallel K-means Implementation

Parallel implementation of K-means leverages multi-core processing to handle large datasets efficiently. This implementation uses Python's multiprocessing to parallelize distance calculations and centroid updates.

```python
from multiprocessing import Pool
import multiprocessing as mp

def parallel_distance_calculation(args):
    X, centroid = args
    return np.sqrt(((X - centroid)**2).sum(axis=1))

def parallel_kmeans(X, k, n_jobs=None, max_iters=100):
    if n_jobs is None:
        n_jobs = mp.cpu_count()
    
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    with Pool(processes=n_jobs) as pool:
        for _ in range(max_iters):
            # Parallel distance calculation
            distance_args = [(X, centroid) for centroid in centroids]
            distances = np.array(pool.map(parallel_distance_calculation, distance_args))
            
            # Assign points to clusters
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            if np.all(centroids == new_centroids):
                break
                
            centroids = new_centroids
    
    return labels, centroids

# Example usage
X, _ = make_blobs(n_samples=10000, centers=8, random_state=42)
labels, centroids = parallel_kmeans(X, k=8)
```

Slide 11: K-means++ Initialization

K-means++ improves cluster initialization by selecting centroids that are far apart from each other, leading to better convergence and final clustering results compared to random initialization.

```python
def kmeans_plus_plus_init(X, k):
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    
    # Choose first centroid randomly
    centroids[0] = X[np.random.randint(n_samples)]
    
    # Choose remaining centroids
    for c in range(1, k):
        # Calculate distances to closest centroid for each point
        distances = np.min([np.sum((X - cent) ** 2, axis=1) 
                          for cent in centroids[:c]], axis=0)
        
        # Calculate probabilities proportional to squared distances
        probs = distances / distances.sum()
        
        # Choose next centroid using weighted probabilities
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        centroids[c] = X[np.searchsorted(cumulative_probs, r)]
    
    return centroids

def kmeans_plus_plus(X, k, max_iters=100):
    # Initialize centroids using k-means++
    centroids = kmeans_plus_plus_init(X, k)
    
    for _ in range(max_iters):
        # Standard k-means iterations
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
            
        centroids = new_centroids
    
    return labels, centroids

# Comparison with random initialization
X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)
labels_random, _ = kmeans_from_scratch(X, k=5)
labels_plus_plus, _ = kmeans_plus_plus(X, k=5)
```

Slide 12: Mini-batch K-means Implementation

Mini-batch K-means reduces computational cost by using small random batches of data in each iteration, making it suitable for large datasets while maintaining good clustering quality.

```python
def mini_batch_kmeans(X, k, batch_size=100, max_iters=100):
    n_samples, n_features = X.shape
    centroids = kmeans_plus_plus_init(X, k)  # Using k-means++ initialization
    
    for _ in range(max_iters):
        # Randomly sample batch
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        batch = X[batch_indices]
        
        # Assign points to clusters
        distances = np.sqrt(((batch - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        
        # Update centroids using learning rate
        learning_rate = 1.0 / (_ + 1)  # Decreasing learning rate
        for i in range(k):
            cluster_points = batch[labels == i]
            if len(cluster_points) > 0:
                centroid_update = cluster_points.mean(axis=0)
                centroids[i] = (1 - learning_rate) * centroids[i] + learning_rate * centroid_update
    
    # Final assignment for all points
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    
    return labels, centroids

# Example with timing comparison
from time import time

X = np.random.randn(10000, 10)
start = time()
_ = kmeans_from_scratch(X, k=5)
print(f"Standard K-means time: {time() - start:.2f}s")

start = time()
_ = mini_batch_kmeans(X, k=5)
print(f"Mini-batch K-means time: {time() - start:.2f}s")
```

Slide 13: Additional Resources

*   "k-means++: The Advantages of Careful Seeding"
    *   [https://arxiv.org/pdf/0701164v1.pdf](https://arxiv.org/pdf/0701164v1.pdf)
*   "Web-scale k-means clustering"
    *   [https://research.google/pubs/pub37242/](https://research.google/pubs/pub37242/)
*   "A Comparative Study of Efficient Initialization Methods for the K-Means Clustering Algorithm"
    *   [https://arxiv.org/abs/1209.1960](https://arxiv.org/abs/1209.1960)
*   "Mini-batch k-means: Scalable Mini-batch Algorithms for Kernel k-means"
    *   [https://research.yahoo.com/publications/6948/mini-batch-k-means-scalable-mini-batch-algorithms](https://research.yahoo.com/publications/6948/mini-batch-k-means-scalable-mini-batch-algorithms)
*   "Scalable K-Means++"
    *   Search on Google Scholar for "Bahman Bahmani, Benjamin Moseley, Andrea Vattani, Ravi Kumar, and Sergei Vassilvitskii. Scalable k-means++"

