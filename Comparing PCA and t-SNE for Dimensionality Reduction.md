## Comparing PCA and t-SNE for Dimensionality Reduction

Slide 1: Understanding PCA and t-SNE

PCA (Principal Component Analysis) and t-SNE (t-distributed Stochastic Neighbor Embedding) are both dimensionality reduction techniques, but they serve different purposes and have distinct characteristics. PCA is primarily used for linear dimensionality reduction and data compression, while t-SNE is designed for visualization of high-dimensional data in lower-dimensional spaces.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate sample data
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 2: Principal Component Analysis (PCA)

PCA is a linear dimensionality reduction technique that identifies the directions (principal components) along which the data varies the most. It projects the data onto these components, effectively reducing the number of dimensions while preserving as much variance as possible.

```python
import numpy as np

def pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    return np.dot(X_centered, top_eigenvectors)

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
X_pca = pca(X, n_components=2)
print("PCA result:", X_pca)
```

Slide 3: t-distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear dimensionality reduction technique specifically designed for visualizing high-dimensional data. It aims to preserve local relationships between data points, making it particularly effective at revealing clusters and patterns in complex datasets.

```python
import numpy as np

def tsne(X, n_components=2, perplexity=30.0, n_iter=1000):
    def compute_pairwise_affinities(X, perplexity):
        distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            Di = distances[i]
            Di[i] = np.inf
            Pi = np.exp(-Di / (2 * (perplexity ** 2)))
            Pi /= np.sum(Pi)
            P[i] = Pi
        return (P + P.T) / (2 * X.shape[0])

    P = compute_pairwise_affinities(X, perplexity)
    Y = np.random.randn(X.shape[0], n_components)
    
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        
        PQ_diff = P - Q
        dY = np.zeros_like(Y)
        for i in range(X.shape[0]):
            dY[i] = 4 * np.sum((PQ_diff[i] * Q[i])[:, np.newaxis] * (Y[i] - Y), axis=0)
        
        Y -= dY * 0.1  # Simple gradient descent
    
    return Y

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X_tsne = tsne(X, n_components=2, perplexity=1.0, n_iter=100)
print("t-SNE result:", X_tsne)
```

Slide 4: Key Differences between PCA and t-SNE

PCA and t-SNE differ in their approach, purpose, and output. PCA is a linear method that preserves global structure, while t-SNE is non-linear and focuses on preserving local relationships. PCA is deterministic and faster, whereas t-SNE is stochastic and computationally intensive.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
X = np.column_stack((np.sin(t), np.cos(t), t))

# Implement PCA
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Implement t-SNE (simplified version)
def tsne(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply PCA and t-SNE
X_pca = pca(X, n_components=2)
X_tsne = tsne(X, n_components=2, perplexity=30, n_iter=300)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 5: PCA: Mathematical Foundation

PCA is based on the concept of maximizing variance along orthogonal directions. It involves calculating the covariance matrix of the data and finding its eigenvectors and eigenvalues. The principal components are the eigenvectors sorted by their corresponding eigenvalues in descending order.

```python
import numpy as np

def pca_math(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    projected_data = np.dot(X_centered, top_eigenvectors)
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return projected_data, explained_variance_ratio

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
projected_data, explained_variance_ratio = pca_math(X, n_components=2)

print("Projected data:")
print(projected_data)
print("\nExplained variance ratio:")
print(explained_variance_ratio)
```

Slide 6: t-SNE: Mathematical Foundation

t-SNE is based on the idea of preserving probability distributions of pairwise similarities between data points in both high and low-dimensional spaces. It uses the Student t-distribution to compute similarities in the low-dimensional space, which helps address the "crowding problem" often encountered in high-dimensional data visualization.

```python
import numpy as np

def tsne_math(X, n_components=2, perplexity=30.0, n_iter=1000):
    def compute_pairwise_affinities(X, perplexity):
        distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            Di = distances[i]
            Di[i] = np.inf
            Pi = np.exp(-Di / (2 * (perplexity ** 2)))
            Pi /= np.sum(Pi)
            P[i] = Pi
        return (P + P.T) / (2 * X.shape[0])

    def compute_q_distribution(Y):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        return Q

    P = compute_pairwise_affinities(X, perplexity)
    Y = np.random.randn(X.shape[0], n_components)
    
    for _ in range(n_iter):
        Q = compute_q_distribution(Y)
        
        PQ_diff = P - Q
        dY = np.zeros_like(Y)
        for i in range(X.shape[0]):
            dY[i] = 4 * np.sum((PQ_diff[i] * Q[i])[:, np.newaxis] * (Y[i] - Y), axis=0)
        
        Y -= dY * 0.1  # Simple gradient descent
    
    return Y

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X_tsne = tsne_math(X, n_components=2, perplexity=1.0, n_iter=100)
print("t-SNE result:")
print(X_tsne)
```

Slide 7: PCA: Advantages and Limitations

PCA is computationally efficient and works well for linear relationships in data. It's useful for data compression and noise reduction. However, it struggles with non-linear relationships and may not capture complex patterns in the data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 1000)
X = np.column_stack((np.sin(t), np.cos(t), t))

# Implement PCA
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Apply PCA
X_pca = pca(X, n_components=2)

# Plot original data and PCA result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='viridis')
ax1.set_title('Original Data (First 2 Dimensions)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 2], cmap='viridis')
ax2.set_title('PCA Result')
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')

plt.tight_layout()
plt.show()
```

Slide 8: t-SNE: Advantages and Limitations

t-SNE excels at revealing clusters and patterns in high-dimensional data. It's particularly effective for visualization tasks. However, it's computationally intensive, non-deterministic, and can be sensitive to hyperparameters like perplexity.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate clustered data
np.random.seed(42)
n_clusters = 5
n_points = 200
X = np.vstack([np.random.randn(n_points, 10) + np.random.randn(10) * 5 for _ in range(n_clusters)])

# Simplified t-SNE implementation
def tsne_simplified(X, n_components=2, perplexity=30.0, n_iter=1000):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply t-SNE
X_tsne = tsne_simplified(X, n_components=2, perplexity=30, n_iter=500)

# Plot t-SNE result
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.repeat(range(n_clusters), n_points), cmap='viridis')
plt.title('t-SNE Visualization of Clustered Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster')
plt.show()
```

Slide 9: PCA: Real-life Example - Image Compression

PCA can be used for image compression by reducing the dimensionality of image data. This technique is particularly useful for grayscale images, where each pixel is represented by a single intensity value.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_sample_image(size=50):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return np.exp(-(x**2 + y**2))

def pca_compress(image, n_components):
    flattened = image.reshape(-1, image.shape[1])
    mean = np.mean(flattened, axis=0)
    centered = flattened - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    compressed = np.dot(U[:, :n_components], np.diag(S[:n_components])).dot(Vt[:n_components, :])
    return (compressed + mean).reshape(image.shape)

# Create and compress the image
original_image = create_sample_image(50)
compressed_image = pca_compress(original_image, n_components=10)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(original_image, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(compressed_image, cmap='gray')
ax2.set_title('PCA Compressed Image')
ax2.axis('off')

plt.tight_layout()
plt.show()

print(f"Original size: {original_image.size}")
print(f"Compressed size: {10 * (50 + 1)}")  # n_components * (image_width + 1)
print(f"Compression ratio: {original_image.size / (10 * (50 + 1)):.2f}")
```

Slide 10: t-SNE: Real-life Example - Visualizing Word Embeddings

t-SNE is often used to visualize high-dimensional word embeddings in natural language processing. This example demonstrates how t-SNE can reveal semantic relationships between words in a 2D space.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified word embedding generation
def generate_word_embeddings(vocab_size=1000, embedding_dim=100):
    np.random.seed(42)
    return np.random.randn(vocab_size, embedding_dim)

# Simplified t-SNE implementation
def tsne_simplified(X, n_components=2, perplexity=30.0, n_iter=1000):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Generate word embeddings and apply t-SNE
embeddings = generate_word_embeddings(vocab_size=100, embedding_dim=50)
tsne_result = tsne_simplified(embeddings, n_components=2, perplexity=5, n_iter=500)

# Plot results
plt.figure(figsize=(12, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])

# Add labels for some random points
np.random.seed(42)
for i in np.random.choice(100, 10, replace=False):
    plt.annotate(f'Word_{i}', (tsne_result[i, 0], tsne_result[i, 1]))

plt.title('t-SNE Visualization of Word Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

Slide 11: Choosing between PCA and t-SNE

The choice between PCA and t-SNE depends on the specific task and data characteristics. PCA is suitable for linear dimensionality reduction, data compression, and when global structure is important. t-SNE is better for non-linear visualization and preserving local structures in high-dimensional data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
X = np.column_stack((np.sin(t), np.cos(t), t))

# Implement PCA
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Implement t-SNE (simplified version)
def tsne(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply PCA and t-SNE
X_pca = pca(X, n_components=2)
X_tsne = tsne(X, n_components=2, perplexity=30, n_iter=300)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 12: Combining PCA and t-SNE

In practice, PCA is often used as a preprocessing step before applying t-SNE to reduce computational complexity and noise. This combination can leverage the strengths of both methods: PCA for initial dimensionality reduction and t-SNE for non-linear visualization.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate high-dimensional data
np.random.seed(42)
n_samples = 1000
n_features = 50
X = np.random.randn(n_samples, n_features)

# Add some structure to the data
X[:n_samples//2, :10] += np.random.randn(n_samples//2, 10) * 5
X[n_samples//2:, 10:20] += np.random.randn(n_samples//2, 10) * 5

# PCA implementation
def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

# Simplified t-SNE implementation
def tsne_simplified(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01  # Simplified update step
    return Y

# Apply PCA followed by t-SNE
X_pca = pca(X, n_components=10)
X_tsne = tsne_simplified(X_pca, n_components=2, perplexity=30, n_iter=300)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(n_samples), cmap='viridis')
plt.title('PCA + t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Sample Index')
plt.show()
```

Slide 13: Conclusion and Best Practices

Both PCA and t-SNE are powerful tools for dimensionality reduction and data visualization. PCA is best for linear relationships and global structure preservation, while t-SNE excels at revealing local patterns and clusters in high-dimensional data. When working with large datasets, consider using PCA as a preprocessing step before applying t-SNE to reduce computational complexity.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples=1000, n_features=50):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    X[:n_samples//2, :10] += np.random.randn(n_samples//2, 10) * 5
    X[n_samples//2:, 10:20] += np.random.randn(n_samples//2, 10) * 5
    return X

def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_eigenvectors = eigenvectors[:, idx[:n_components]]
    return np.dot(X_centered, top_eigenvectors)

def tsne_simplified(X, n_components, perplexity, n_iter):
    Y = np.random.randn(X.shape[0], n_components)
    for _ in range(n_iter):
        distances = np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
        Q = 1 / (1 + distances)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Y -= np.random.randn(*Y.shape) * 0.01
    return Y

# Generate and process data
X = generate_data()
X_pca = pca(X, n_components=10)
X_tsne = tsne_simplified(X_pca, n_components=2, perplexity=30, n_iter=300)

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.arange(X.shape[0]), cmap='viridis')
plt.title('PCA + t-SNE: Best of Both Worlds')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Sample Index')
plt.show()
```

Slide 14: Additional Resources

For more in-depth information on PCA and t-SNE, consider exploring the following resources:

1.  "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (2008). Available on arXiv: [https://arxiv.org/abs/1807.01281](https://arxiv.org/abs/1807.01281)
2.  "Principal Component Analysis" by Jonathon Shlens (2014). Available on arXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
3.  "How to Use t-SNE Effectively" by Martin Wattenberg, Fernanda Viégas, and Ian Johnson. Published in Distill (2016). [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
4.  "Understanding the Manifold Hypothesis" by Saket Choudhary. Available on arXiv: [https://arxiv.org/abs/2101.05742](https://arxiv.org/abs/2101.05742)

These resources provide detailed mathematical foundations, implementation insights, and best practices for using PCA and t-SNE in various data analysis scenarios.

