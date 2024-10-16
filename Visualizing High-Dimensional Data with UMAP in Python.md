## Visualizing High-Dimensional Data with UMAP in Python
Slide 1: Introduction to UMAP

Uniform Manifold Approximation and Projection (UMAP) is a dimensionality reduction technique used for visualizing high-dimensional data. It preserves both local and global structure, making it effective for various data types.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

# Generate sample high-dimensional data
data = np.random.rand(1000, 50)

# Create UMAP object and fit_transform data
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(data)

# Plot the result
plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
plt.title('UMAP Projection of Random Data')
plt.show()
```

Slide 2: UMAP Algorithm Overview

UMAP constructs a high-dimensional graph representation of the data and then optimizes a low-dimensional graph to be as structurally similar as possible.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def simplified_umap_graph(data, n_neighbors=15):
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Compute graph weights (simplified)
    weights = np.exp(-distances**2)
    
    return indices, weights

# Example usage
data = np.random.rand(100, 10)
indices, weights = simplified_umap_graph(data)
print("Neighbor indices shape:", indices.shape)
print("Weights shape:", weights.shape)
```

Slide 3: Graph Construction

UMAP begins by constructing a weighted k-neighbor graph. Each point is connected to its k nearest neighbors, with edge weights based on the distance between points.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt

def create_knn_graph(data, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    G = nx.Graph()
    for i in range(len(data)):
        for j, dist in zip(indices[i], distances[i]):
            G.add_edge(i, j, weight=np.exp(-dist))
    
    return G

# Example usage
data = np.random.rand(20, 2)
graph = create_knn_graph(data)

pos = {i: data[i] for i in range(len(data))}
nx.draw(graph, pos, node_size=50, with_labels=True)
plt.title('K-Nearest Neighbor Graph')
plt.show()
```

Slide 4: Fuzzy Topological Representation

UMAP creates a fuzzy topological representation of the high-dimensional data. This step involves computing local fuzzy simplicial set representations for each point.

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def fuzzy_simplicial_set(data, n_neighbors=15, metric='euclidean'):
    distances = squareform(pdist(data, metric=metric))
    knn_distances = np.partition(distances, n_neighbors, axis=1)[:, :n_neighbors]
    sigma = np.mean(knn_distances[:, -1])
    
    adjacency = np.exp(-distances**2 / sigma**2)
    np.fill_diagonal(adjacency, 0)
    
    return adjacency

# Example usage
data = np.random.rand(100, 10)
adjacency = fuzzy_simplicial_set(data)
plt.imshow(adjacency, cmap='viridis')
plt.colorbar()
plt.title('Fuzzy Simplicial Set Representation')
plt.show()
```

Slide 5: Low-Dimensional Embedding Initialization

UMAP initializes the low-dimensional embedding, often using spectral embedding techniques or random initialization.

```python
import numpy as np
from sklearn.manifold import spectral_embedding

def initialize_embedding(graph, n_components=2, method='spectral'):
    if method == 'spectral':
        return spectral_embedding(graph, n_components=n_components)
    elif method == 'random':
        return np.random.uniform(low=-10, high=10, size=(graph.shape[0], n_components))

# Example usage
n_samples = 1000
high_dim_data = np.random.rand(n_samples, 50)
graph = fuzzy_simplicial_set(high_dim_data)  # Using function from previous slide

embedding = initialize_embedding(graph, method='spectral')
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title('Initial Low-Dimensional Embedding')
plt.show()
```

Slide 6: Optimization Process

UMAP optimizes the low-dimensional representation to minimize the cross-entropy between the high and low-dimensional fuzzy topological representations.

```python
import numpy as np
import numba

@numba.jit(nopython=True)
def optimize_embedding(embedding, graph, n_epochs=200, learning_rate=1.0):
    for epoch in range(n_epochs):
        for i in range(embedding.shape[0]):
            for j in range(embedding.shape[0]):
                if i != j:
                    d_high = graph[i, j]
                    d_low = np.linalg.norm(embedding[i] - embedding[j])
                    grad = 2 * (d_high - d_low) * (embedding[i] - embedding[j])
                    embedding[i] += learning_rate * grad
                    embedding[j] -= learning_rate * grad
    return embedding

# Example usage (continuing from previous slide)
optimized_embedding = optimize_embedding(embedding, graph)
plt.scatter(optimized_embedding[:, 0], optimized_embedding[:, 1], alpha=0.5)
plt.title('Optimized Low-Dimensional Embedding')
plt.show()
```

Slide 7: UMAP Parameters

Key parameters in UMAP include n\_neighbors, min\_dist, and n\_components. These parameters control the trade-off between preserving local and global structure.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

def plot_umap_variations(data, n_neighbors_list, min_dist_list):
    fig, axs = plt.subplots(len(n_neighbors_list), len(min_dist_list), figsize=(15, 15))
    for i, n_neighbors in enumerate(n_neighbors_list):
        for j, min_dist in enumerate(min_dist_list):
            reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            embedding = reducer.fit_transform(data)
            axs[i, j].scatter(embedding[:, 0], embedding[:, 1], s=5)
            axs[i, j].set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
    plt.tight_layout()
    plt.show()

# Example usage
data = np.random.rand(1000, 50)
plot_umap_variations(data, [5, 15, 30], [0.1, 0.5, 1.0])
```

Slide 8: Comparison with Other Dimensionality Reduction Techniques

UMAP often outperforms t-SNE in preserving global structure and is computationally more efficient for large datasets.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

def compare_dim_reduction(data):
    umap_embedding = umap.UMAP().fit_transform(data)
    tsne_embedding = TSNE().fit_transform(data)
    pca_embedding = PCA(n_components=2).fit_transform(data)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.scatter(umap_embedding[:, 0], umap_embedding[:, 1], s=5)
    ax1.set_title('UMAP')
    ax2.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], s=5)
    ax2.set_title('t-SNE')
    ax3.scatter(pca_embedding[:, 0], pca_embedding[:, 1], s=5)
    ax3.set_title('PCA')
    plt.tight_layout()
    plt.show()

# Example usage
data = np.random.rand(1000, 50)
compare_dim_reduction(data)
```

Slide 9: Supervised and Semi-Supervised UMAP

UMAP can incorporate label information for supervised or semi-supervised dimensionality reduction, improving separation between classes.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate synthetic data with labels
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_clusters_per_class=1)

# Unsupervised UMAP
umap_unsupervised = umap.UMAP().fit_transform(X)

# Supervised UMAP
umap_supervised = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, target_metric='categorical').fit_transform(X, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.scatter(umap_unsupervised[:, 0], umap_unsupervised[:, 1], c=y, cmap='viridis')
ax1.set_title('Unsupervised UMAP')
ax2.scatter(umap_supervised[:, 0], umap_supervised[:, 1], c=y, cmap='viridis')
ax2.set_title('Supervised UMAP')
plt.tight_layout()
plt.show()
```

Slide 10: UMAP for Clustering

UMAP can be used as a preprocessing step for clustering algorithms, often improving cluster separation and visualization.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic clustered data
X, y_true = make_blobs(n_samples=1000, centers=5, cluster_std=0.5, random_state=42)

# Apply UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(X)

# Perform K-means clustering on the embedding
kmeans = KMeans(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(embedding)

# Visualize results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('Original Data')
plt.subplot(122)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, cmap='viridis')
plt.title('UMAP + K-means Clustering')
plt.tight_layout()
plt.show()
```

Slide 11: UMAP for Anomaly Detection

UMAP can help identify anomalies by projecting data into a lower-dimensional space where outliers become more apparent.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

# Generate normal data and anomalies
X_normal, _ = make_blobs(n_samples=1000, centers=1, cluster_std=0.5, random_state=42)
X_anomalies = np.random.uniform(low=-10, high=10, size=(50, 2))
X = np.vstack([X_normal, X_anomalies])

# Apply UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(X)

# Perform anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_pred = iso_forest.fit_predict(embedding)

# Visualize results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Original Data')
plt.subplot(122)
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, cmap='viridis')
plt.title('UMAP + Isolation Forest')
plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example: Image Data Visualization

UMAP can be used to visualize high-dimensional image data, such as the MNIST dataset of handwritten digits.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Subsample for faster computation
n_samples = 5000
subset = np.random.choice(X.shape[0], n_samples, replace=False)
X_subset, y_subset = X[subset], y[subset]

# Apply UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(X_subset)

# Visualize results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_subset.astype(int), cmap='tab10', s=5)
plt.colorbar(scatter)
plt.title('UMAP visualization of MNIST digits')
plt.show()
```

Slide 13: Real-Life Example: Gene Expression Analysis

UMAP is widely used in bioinformatics for visualizing high-dimensional gene expression data and identifying cell types in single-cell RNA sequencing experiments.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Simulate gene expression data
n_samples = 1000
n_features = 1000
n_cell_types = 5

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_cell_types, cluster_std=0.5, random_state=42)

# Apply UMAP
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2)
embedding = reducer.fit_transform(X)

# Visualize results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Set1', s=5)
plt.colorbar(scatter)
plt.title('UMAP visualization of simulated gene expression data')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
```

Slide 14: Additional Resources

For more information on UMAP, refer to the following resources:

1. Original UMAP paper: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. ArXiv:1802.03426 \[Cs, Stat\]. [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
2. UMAP documentation: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
3. Comparison of UMAP with t-SNE: Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell transc

## Response:
undefined

