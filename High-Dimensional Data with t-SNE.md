## High-Dimensional Data with t-SNE
Slide 1: t-SNE Fundamentals

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a powerful dimensionality reduction technique that excels at preserving local structure in high-dimensional data by modeling similar samples as nearby points and dissimilar samples as distant points in the lower-dimensional space.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate sample high-dimensional data
np.random.seed(42)
X = np.random.randn(1000, 50)  # 1000 samples, 50 dimensions

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.xlabel('First Component')
plt.ylabel('Second Component')
```

Slide 2: Understanding t-SNE Probability Distribution

The core concept of t-SNE involves converting high-dimensional Euclidean distances into conditional probabilities that represent similarities, using a Student t-distribution in the low-dimensional space to avoid the crowding problem.

```python
def compute_pairwise_affinities(X, perplexity=30.0, sigma=1.0):
    """
    Compute pairwise affinities between points using Gaussian kernel
    """
    distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    P = np.exp(-distances / (2 * sigma ** 2))
    np.fill_diagonal(P, 0)
    P = P / np.sum(P)
    return P
```

Slide 3: Perplexity Parameter in t-SNE

The perplexity parameter in t-SNE controls the balance between local and global structure preservation, effectively determining the number of nearest neighbors considered. Higher values preserve more global structure while lower values focus on local patterns.

```python
def tsne_multiple_perplexities(X, perplexities=[5, 30, 50]):
    """
    Compare t-SNE with different perplexity values
    """
    fig, axes = plt.subplots(1, len(perplexities), figsize=(15, 5))
    
    for idx, perp in enumerate(perplexities):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        X_embedded = tsne.fit_transform(X)
        
        axes[idx].scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.6)
        axes[idx].set_title(f'Perplexity: {perp}')
```

Slide 4: Optimization Process

The t-SNE algorithm uses gradient descent to minimize the Kullback-Leibler divergence between the probability distributions in high-dimensional and low-dimensional spaces, making it computationally intensive but effective.

```python
def compute_gradient(P, Q, Y, n_components=2):
    """
    Compute the gradient of the t-SNE objective function
    """
    pq_diff = P - Q  # Difference between probability matrices
    grad = np.zeros_like(Y)
    
    for i in range(Y.shape[0]):
        diff = Y[i] - Y
        dist = 1 / (1 + np.sum(diff ** 2, axis=1))
        grad[i] = 4 * np.sum(np.expand_dims(pq_diff[i] * dist, 1) * diff, axis=0)
    
    return grad
```

Slide 5: Real-world Example: MNIST Visualization

This implementation demonstrates t-SNE's effectiveness in visualizing high-dimensional image data using the MNIST dataset, showing how it clusters similar digits together while maintaining local structure.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# Load and preprocess MNIST data
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

# Visualize with colors for different digits
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                     c=digits.target, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of MNIST digits')
```

Slide 6: Implementation from Scratch

Understanding the core mathematics of t-SNE requires implementing the algorithm from scratch. This implementation focuses on the fundamental probability computations and gradient descent optimization process.

```python
def tsne_from_scratch(X, n_components=2, perplexity=30.0, n_iter=1000):
    """
    Implementation of t-SNE from scratch
    """
    n_samples = X.shape[0]
    
    # Initialize low-dimensional representation
    np.random.seed(42)
    Y = np.random.randn(n_samples, n_components) * 0.0001
    
    # Compute high-dimensional pairwise similarities
    distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    P = np.exp(-distances / (2 * perplexity ** 2))
    np.fill_diagonal(P, 0)
    P = (P + P.T) / (2 * n_samples)
    P = np.maximum(P, 1e-12)
    
    return Y
```

Slide 7: Gradient Descent Implementation for t-SNE

The optimization process in t-SNE requires careful implementation of gradient descent with momentum to avoid local minima and ensure convergence to a good solution.

```python
def optimize_tsne(P, Y, n_iter=1000, learning_rate=200, momentum=0.5):
    """
    Gradient descent with momentum for t-SNE optimization
    """
    Y_prev = Y.copy()
    Y_incs = np.zeros_like(Y)
    
    for iteration in range(n_iter):
        # Compute low-dimensional affinities
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradients
        grad = 4 * (P - Q)[:, :, np.newaxis] * (Y[:, np.newaxis, :] - Y[np.newaxis, :, :])
        grad = np.sum(grad, axis=1)
        
        # Update Y with momentum
        Y_incs = momentum * Y_incs - learning_rate * grad
        Y = Y + Y_incs
        
        # Early exaggeration
        if iteration == 100:
            P = P / 4
            
    return Y
```

Slide 8: Real-world Example: Gene Expression Analysis

t-SNE is particularly useful in bioinformatics for visualizing high-dimensional gene expression data, helping identify patterns and clusters in complex biological datasets.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Simulate gene expression data
np.random.seed(42)
n_genes = 1000
n_samples = 100
gene_expression = np.random.normal(loc=0, scale=1, size=(n_samples, n_genes))

# Add some structure (simulated cell types)
cell_types = np.repeat(['TypeA', 'TypeB', 'TypeC'], n_samples // 3)
for i, cell_type in enumerate(['TypeA', 'TypeB', 'TypeC']):
    idx = np.where(cell_types == cell_type)[0]
    gene_expression[idx, :100] += i * 2

# Normalize and apply t-SNE
scaled_data = StandardScaler().fit_transform(gene_expression)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding = tsne.fit_transform(scaled_data)

# Visualize results
plt.figure(figsize=(10, 8))
for cell_type in np.unique(cell_types):
    mask = cell_types == cell_type
    plt.scatter(embedding[mask, 0], embedding[mask, 1], label=cell_type, alpha=0.7)
plt.legend()
plt.title('t-SNE visualization of gene expression data')
```

Slide 9: Barnes-Hut Optimization

Barnes-Hut approximation significantly reduces t-SNE's computational complexity from O(n²) to O(n log n) by using a tree-based algorithm for computing similarities.

```python
def build_vptree(X):
    """
    Implement Vantage Point Tree for efficient nearest neighbor search
    """
    class VPTree:
        def __init__(self, point, left=None, right=None, threshold=0):
            self.point = point
            self.left = left
            self.right = right
            self.threshold = threshold
    
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def build_tree(points):
        if len(points) == 0:
            return None
        
        # Choose random vantage point
        vp_idx = np.random.randint(len(points))
        vp = points[vp_idx]
        
        if len(points) == 1:
            return VPTree(vp)
        
        # Compute distances to vantage point
        distances = [distance(vp, p) for p in points]
        median_dist = np.median(distances)
        
        # Split points
        left_points = points[distances <= median_dist]
        right_points = points[distances > median_dist]
        
        return VPTree(vp,
                     build_tree(left_points),
                     build_tree(right_points),
                     median_dist)
    
    return build_tree(X)
```

Slide 10: Early Exaggeration Technique

Early exaggeration multiplies the original probabilities by a factor in the initial phase of optimization, helping to form well-separated clusters by increasing the attractive forces between similar points.

```python
def apply_early_exaggeration(X, exaggeration_factor=12, early_iter=250):
    """
    Implement early exaggeration for better cluster separation
    """
    n_samples = X.shape[0]
    P = compute_pairwise_affinities(X)
    
    # Initialize optimization variables
    Y = np.random.randn(n_samples, 2) * 0.0001
    gains = np.ones((n_samples, 2))
    update = np.zeros((n_samples, 2))
    
    # Apply early exaggeration
    P = P * exaggeration_factor
    
    for iter in range(early_iter):
        # Compute Q distribution and gradients
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + sum_Y.reshape((-1, 1)) + sum_Y - 2 * np.dot(Y, Y.T))
        num[range(n_samples), range(n_samples)] = 0
        Q = num / np.sum(num)
        grad = 4 * (P - Q) @ Y
        
        # Update Y with momentum
        gains = (gains + 0.2) * ((grad > 0) != (update > 0)) + gains * 0.8
        update = 0.8 * update - 200 * gains * grad
        Y = Y + update
        
        # Zero-center the solution
        Y = Y - np.mean(Y, axis=0)
        
    return Y, P / exaggeration_factor
```

Slide 11: Handling Different Distance Metrics

t-SNE can be adapted to work with various distance metrics beyond Euclidean distance, making it suitable for different types of data structures and similarity measures.

```python
def custom_distance_tsne(X, metric='cosine', n_components=2):
    """
    Implement t-SNE with custom distance metrics
    """
    from scipy.spatial.distance import cdist
    
    def compute_custom_affinities(X, metric, perplexity=30.0):
        distances = cdist(X, X, metric=metric)
        P = np.exp(-distances ** 2)
        np.fill_diagonal(P, 0)
        P = P / np.sum(P)
        return P
    
    # Initialize parameters
    n_samples = X.shape[0]
    Y = np.random.randn(n_samples, n_components) * 0.0001
    
    # Compute affinities with custom metric
    P = compute_custom_affinities(X, metric)
    
    # Example usage with different metrics
    metrics = ['euclidean', 'cosine', 'manhattan']
    results = {}
    
    for metric in metrics:
        tsne = TSNE(n_components=2, metric=metric)
        results[metric] = tsne.fit_transform(X)
        
    return results
```

Slide 12: Visualization Techniques for t-SNE Results

Advanced visualization techniques help interpret t-SNE results by incorporating additional information such as cluster density, uncertainty, and temporal evolution of the embedding.

```python
def advanced_tsne_visualization(X_embedded, labels=None, uncertainties=None):
    """
    Create advanced visualization for t-SNE results
    """
    import seaborn as sns
    from scipy.stats import gaussian_kde
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Basic scatter plot with labels
    scatter = axes[0,0].scatter(X_embedded[:, 0], X_embedded[:, 1],
                               c=labels if labels is not None else 'blue',
                               alpha=0.6)
    axes[0,0].set_title('Basic t-SNE Plot')
    
    # Density estimation
    xy = np.vstack([X_embedded[:,0], X_embedded[:,1]])
    z = gaussian_kde(xy)(xy)
    
    # Density-colored scatter plot
    density_scatter = axes[0,1].scatter(X_embedded[:, 0], X_embedded[:, 1],
                                      c=z, cmap='viridis')
    plt.colorbar(density_scatter, ax=axes[0,1])
    axes[0,1].set_title('Density-based Visualization')
    
    # Contour plot
    x = np.linspace(X_embedded[:,0].min(), X_embedded[:,0].max(), 100)
    y = np.linspace(X_embedded[:,1].min(), X_embedded[:,1].max(), 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = gaussian_kde(xy)(positions)
    Z = Z.reshape(X.shape)
    
    axes[1,0].contour(X, Y, Z, levels=15)
    axes[1,0].set_title('Contour Plot of Density')
    
    # Uncertainty visualization if provided
    if uncertainties is not None:
        uncertainty_scatter = axes[1,1].scatter(X_embedded[:, 0], X_embedded[:, 1],
                                              c=uncertainties, cmap='RdYlBu_r')
        plt.colorbar(uncertainty_scatter, ax=axes[1,1])
        axes[1,1].set_title('Uncertainty Visualization')
    
    plt.tight_layout()
    return fig
```

Slide 13: Additional Resources

*   "Visualizing Data using t-SNE"
*   [https://arxiv.org/abs/1008.4309](https://arxiv.org/abs/1008.4309)
*   "How to Use t-SNE Effectively"
*   [https://arxiv.org/abs/1612.03628](https://arxiv.org/abs/1612.03628)
*   "Accelerating t-SNE using Tree-Based Algorithms"
*   [https://arxiv.org/abs/1401.7957](https://arxiv.org/abs/1401.7957)
*   "Understanding t-SNE: Parameter Selection and Optimization"
*   [https://arxiv.org/abs/1712.09005](https://arxiv.org/abs/1712.09005)
*   "A Theoretical Analysis of t-SNE"
*   [https://arxiv.org/abs/1902.05969](https://arxiv.org/abs/1902.05969)

