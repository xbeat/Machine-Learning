## Hierarchical Clustering in Python
Slide 1: Introduction to Hierarchical Clustering

Hierarchical clustering is an unsupervised machine learning technique used to group similar data points into clusters. Unlike other clustering methods, it creates a hierarchy of clusters, allowing for a more detailed understanding of data structure at different levels of granularity. This method is particularly useful when the number of clusters is unknown or when you want to explore the data at multiple scales.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Sample Data for Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Slide 2: Types of Hierarchical Clustering

There are two main types of hierarchical clustering: agglomerative and divisive. Agglomerative clustering starts with each data point as a separate cluster and iteratively merges the closest clusters. Divisive clustering, on the other hand, starts with all data points in one cluster and recursively splits them into smaller clusters. We'll focus on agglomerative clustering as it's more commonly used.

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogram of Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
```

Slide 3: Distance Metrics

The choice of distance metric is crucial in hierarchical clustering. Common metrics include Euclidean distance, Manhattan distance, and cosine similarity. The Euclidean distance is the most widely used and is defined as the straight-line distance between two points in n-dimensional space.

```python
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Example usage
p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
distance = euclidean_distance(p1, p2)
print(f"Euclidean distance between {p1} and {p2}: {distance:.2f}")
```

Slide 4: Linkage Methods

Linkage methods determine how the distance between clusters is calculated. Common methods include single linkage (minimum distance), complete linkage (maximum distance), and average linkage (average distance). Ward's method, which minimizes the variance within clusters, is also popular.

```python
def single_linkage(cluster1, cluster2):
    return np.min([euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2])

def complete_linkage(cluster1, cluster2):
    return np.max([euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2])

def average_linkage(cluster1, cluster2):
    distances = [euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2]
    return np.mean(distances)

# Example usage
cluster1 = np.array([[1, 2], [2, 3]])
cluster2 = np.array([[4, 5], [5, 6]])
print(f"Single linkage: {single_linkage(cluster1, cluster2):.2f}")
print(f"Complete linkage: {complete_linkage(cluster1, cluster2):.2f}")
print(f"Average linkage: {average_linkage(cluster1, cluster2):.2f}")
```

Slide 5: Agglomerative Clustering Algorithm

The agglomerative clustering algorithm follows these steps:

1.  Start with each data point as a separate cluster
2.  Calculate the distances between all pairs of clusters
3.  Merge the two closest clusters
4.  Update the distances between the new cluster and all other clusters
5.  Repeat steps 3-4 until only one cluster remains or a stopping criterion is met

```python
def agglomerative_clustering(X, n_clusters, linkage_func):
    clusters = [[point] for point in X]
    while len(clusters) > n_clusters:
        min_distance = float('inf')
        merge_indices = (0, 0)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = linkage_func(clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)
        
        i, j = merge_indices
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    return clusters

# Example usage
X_small = X[:10]  # Use a small subset for demonstration
result = agglomerative_clustering(X_small, n_clusters=3, linkage_func=single_linkage)
print(f"Number of clusters: {len(result)}")
print(f"Cluster sizes: {[len(cluster) for cluster in result]}")
```

Slide 6: Dendrogram Visualization

A dendrogram is a tree-like diagram that shows the hierarchical relationship between clusters. The height of each branch represents the distance at which clusters are merged. Dendrograms are useful for visualizing the clustering process and determining an appropriate number of clusters.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogram of Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
```

Slide 7: Determining the Number of Clusters

Choosing the optimal number of clusters is a critical step in hierarchical clustering. Common methods include:

1.  Visual inspection of the dendrogram
2.  Elbow method: plotting the within-cluster sum of squares against the number of clusters
3.  Silhouette analysis: measuring how similar an object is to its own cluster compared to other clusters

```python
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

def evaluate_clusters(X, Z, max_clusters):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    plt.plot(range(2, max_clusters + 1), silhouette_scores)
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

# Example usage
evaluate_clusters(X, Z, max_clusters=10)
```

Slide 8: Implementing Hierarchical Clustering from Scratch

Let's implement a basic version of hierarchical clustering using the single linkage method. This implementation will help us understand the core concepts of the algorithm.

```python
import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def single_linkage(cluster1, cluster2):
    return min(euclidean_distance(p1, p2) for p1 in cluster1 for p2 in cluster2)

def hierarchical_clustering(X, n_clusters):
    clusters = [[point] for point in X]
    while len(clusters) > n_clusters:
        min_dist = float('inf')
        merge_pair = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = single_linkage(clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)
        i, j = merge_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    return clusters

# Example usage
np.random.seed(42)
X = np.random.rand(20, 2)
result = hierarchical_clustering(X, n_clusters=3)
print(f"Number of clusters: {len(result)}")
print(f"Cluster sizes: {[len(cluster) for cluster in result]}")
```

Slide 9: Visualizing Hierarchical Clustering Results

After performing hierarchical clustering, it's important to visualize the results to gain insights into the data structure. We'll create a scatter plot of the data points, coloring them based on their cluster assignments.

```python
import matplotlib.pyplot as plt

def plot_clusters(X, clusters):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.figure(figsize=(10, 8))
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
    plt.title("Hierarchical Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

# Example usage
np.random.seed(42)
X = np.random.rand(100, 2)
clusters = hierarchical_clustering(X, n_clusters=4)
plot_clusters(X, clusters)
```

Slide 10: Real-Life Example: Document Clustering

Hierarchical clustering can be used for document clustering, which is useful in organizing large collections of text documents. We'll use a simple example with TF-IDF vectors to represent documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outfoxes a lazy fox",
    "The lazy dog sleeps all day",
    "The quick rabbit runs away from the fox",
    "A brown fox chases a quick rabbit"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Perform hierarchical clustering
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=documents, leaf_rotation=90, leaf_font_size=8)
plt.title("Document Clustering Dendrogram")
plt.xlabel("Documents")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Image Segmentation

Hierarchical clustering can be applied to image segmentation, which involves partitioning an image into multiple segments or objects. We'll use a simple example with color-based segmentation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def segment_image(image, n_segments):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape((-1, 3))
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=n_segments)
    labels = clustering.fit_predict(pixels)
    
    # Create the segmented image
    segmented = labels.reshape(image.shape[:2])
    return segmented

# Generate a simple synthetic image
image = np.zeros((100, 100, 3), dtype=np.uint8)
image[:50, :50] = [255, 0, 0]  # Red square
image[50:, 50:] = [0, 255, 0]  # Green square
image[:50, 50:] = [0, 0, 255]  # Blue square

# Segment the image
segmented = segment_image(image, n_segments=3)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image)
ax1.set_title("Original Image")
ax2.imshow(segmented, cmap='viridis')
ax2.set_title("Segmented Image")
plt.show()
```

Slide 12: Advantages and Disadvantages of Hierarchical Clustering

Advantages:

1.  Provides a hierarchical representation of data
2.  No need to specify the number of clusters in advance
3.  Suitable for datasets with hierarchical structures

Disadvantages:

1.  Computationally expensive (O(n^3) time complexity)
2.  Sensitive to outliers
3.  Difficult to handle large datasets due to memory constraints

```python
import numpy as np
import time

def measure_runtime(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

# Generate datasets of different sizes
sizes = [100, 500, 1000, 2000]
runtimes = []

for size in sizes:
    X = np.random.rand(size, 2)
    runtime = measure_runtime(linkage, X, method='ward')
    runtimes.append(runtime)

# Plot runtime vs dataset size
plt.figure(figsize=(10, 6))
plt.plot(sizes, runtimes, marker='o')
plt.title("Runtime of Hierarchical Clustering vs Dataset Size")
plt.xlabel("Number of Data Points")
plt.ylabel("Runtime (seconds)")
plt.show()
```

Slide 13: Comparison with Other Clustering Algorithms

Hierarchical clustering has unique characteristics compared to other popular clustering algorithms like K-means and DBSCAN. Let's compare their performance on a simple dataset.

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate a dataset
X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

# Apply different clustering algorithms
hc = AgglomerativeClustering(n_clusters=2)
kmeans = KMeans(n_clusters=2, random_state=42)
dbscan = DBSCAN(eps=0.3, min_samples=5)

hc_labels = hc.fit_predict(X)
kmeans_labels = kmeans.fit_predict(X)
dbscan_labels = dbscan.fit_predict(X)

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(X[:, 0], X[:, 1], c=hc_labels, cmap='viridis')
ax1.set_title("Hierarchical Clustering")

ax2.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
ax2.set_title("K-means Clustering")

ax3.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
ax3.set_title("DBSCAN Clustering")

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into hierarchical clustering, here are some valuable resources:

1.  "A Survey of Hierarchical Clustering Algorithms" by Olatz Arbelaitz et al. (2013) ArXiv URL: [https://arxiv.org/abs/1305.6827](https://arxiv.org/abs/1305.6827)
2.  "Hierarchical Clustering Algorithms" by F. Murtagh and P. Contreras (2011) ArXiv URL: [https://arxiv.org/abs/1105.0121](https://arxiv.org/abs/1105.0121)
3.  "Comparison of Hierarchical Cluster Analysis Methods by Cophenetic Correlation" by Sacha Epskamp et al. (2018) ArXiv URL: [https://arxiv.org/abs/1809.10903](https://arxiv.org/abs/1809.10903)

These papers provide in-depth discussions on various aspects of hierarchical clustering,

