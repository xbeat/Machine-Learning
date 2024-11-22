## Dimensionality Reduction Algorithms in Python
Slide 1: Introduction to Principal Component Analysis (PCA)

Principal Component Analysis stands as the foundational dimensionality reduction technique, transforming high-dimensional data into lower dimensions while preserving maximum variance. It works by finding orthogonal axes (principal components) that capture the most significant patterns in the data.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 10)  # 100 samples, 10 features

# Initialize and fit PCA
pca = PCA()
X_transformed = pca.fit_transform(X)

# Calculate explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Mathematical representation (not rendered):
# $$C = \frac{1}{n} X^T X$$
# $$\lambda v = Cv$$

print(f"Explained variance ratio: {explained_variance}")
```

Slide 2: t-SNE Implementation

t-Distributed Stochastic Neighbor Embedding excels at preserving local structure in high-dimensional data by modeling probability distributions of pairwise similarities between points in both high and low-dimensional spaces.

```python
from sklearn.manifold import TSNE
import numpy as np

# Generate high-dimensional data
X = np.random.randn(1000, 50)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
X_tsne = tsne.fit_transform(X)

# Visualize results
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
```

Slide 3: UMAP Overview

Uniform Manifold Approximation and Projection combines the theoretical foundations of manifold learning with computational efficiency, making it particularly effective for large-scale datasets while preserving both local and global structure.

```python
import umap
import numpy as np
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X = digits.data

# Apply UMAP
reducer = umap.UMAP(n_neighbors=15, 
                   min_dist=0.1,
                   n_components=2,
                   random_state=42)
X_umap = reducer.fit_transform(X)

# Visualization
plt.scatter(X_umap[:, 0], X_umap[:, 1], 
           c=digits.target, cmap='Spectral')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
```

Slide 4: Autoencoder Implementation

Autoencoders provide a neural network-based approach to dimensionality reduction, learning a compressed representation of the input data through an encoding-decoding process that minimizes reconstruction error.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define autoencoder architecture
input_dim = 784  # Example: MNIST dimensions
encoding_dim = 32

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Create and compile model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
```

Slide 5: Kernel PCA Implementation

Kernel PCA extends traditional PCA by using kernel methods to capture nonlinear relationships in the data, enabling dimensionality reduction for datasets with complex, nonlinear patterns.

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# Generate nonlinear data
X, y = make_circles(n_samples=400, factor=0.3, noise=0.05)

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Visualization
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original Data')
plt.subplot(122)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title('Kernel PCA Transformation')
```

Slide 6: Real-World Application - Gene Expression Analysis

Gene expression datasets typically contain thousands of features (genes) with relatively few samples. This implementation demonstrates dimensionality reduction for visualizing complex biological data relationships.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Simulate gene expression data
np.random.seed(42)
n_genes = 1000
n_samples = 100
gene_expr = np.random.normal(0, 1, (n_samples, n_genes))

# Preprocessing
scaler = StandardScaler()
gene_expr_scaled = scaler.fit_transform(gene_expr)

# Apply PCA
pca = PCA(n_components=2)
gene_expr_pca = pca.fit_transform(gene_expr_scaled)

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(gene_expr_pca[:, 0], gene_expr_pca[:, 1])
plt.title('Gene Expression PCA')
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 7: Linear Discriminant Analysis (LDA)

LDA performs dimensionality reduction while maximizing class separability, making it particularly effective for supervised learning tasks where maintaining class distinction is crucial.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

# Generate classified data
X, y = make_classification(n_samples=1000, n_features=20, 
                         n_informative=15, n_redundant=5,
                         n_classes=3, random_state=42)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualization
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('LDA Transformation')
```

Slide 8: Locally Linear Embedding (LLE)

LLE preserves the local geometry of high-dimensional data by reconstructing each point from its neighbors, providing an effective nonlinear dimensionality reduction technique for manifold learning.

```python
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import make_swiss_roll

# Generate swiss roll dataset
X, color = make_swiss_roll(n_samples=1000, random_state=42)

# Apply LLE
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                            method='modified', random_state=42)
X_lle = lle.fit_transform(X)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 2], c=color)
plt.title('Original Swiss Roll')
plt.subplot(122)
plt.scatter(X_lle[:, 0], X_lle[:, 1], c=color)
plt.title('LLE Transformation')
```

Slide 9: Isomap Implementation

Isomap extends traditional MDS by replacing Euclidean distances with geodesic distances, effectively capturing the intrinsic geometry of nonlinear manifolds in the data.

```python
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

# Generate complex nonlinear data
t = np.pi * np.linspace(0, 1, 1000)
X = np.column_stack([
    np.sin(2*t), np.cos(3*t), 
    np.sin(4*t), np.cos(5*t)
])

# Apply Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_iso = isomap.fit_transform(X)

# Scale results for visualization
scaler = MinMaxScaler()
X_iso_scaled = scaler.fit_transform(X_iso)

plt.scatter(X_iso_scaled[:, 0], X_iso_scaled[:, 1], c=t)
plt.colorbar(label='Position in original curve')
plt.title('Isomap Embedding')
```

Slide 10: Factor Analysis

Factor Analysis uncovers latent variables that explain correlations among observed variables, providing interpretable dimensionality reduction particularly useful in psychological and social sciences.

```python
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# Generate correlated data
n_samples = 1000
n_features = 10
X = np.random.multivariate_normal(
    mean=np.zeros(n_features),
    cov=np.eye(n_features) + 0.3*np.ones((n_features, n_features)),
    size=n_samples
)

# Apply Factor Analysis
fa = FactorAnalysis(n_components=3, random_state=42)
X_fa = fa.fit_transform(X)

# Print factor loadings
print("Factor loadings:")
print(fa.components_.T)
```

Slide 11: Real-World Application - Image Dimensionality Reduction

This implementation demonstrates dimensionality reduction on image data, commonly used in computer vision tasks for feature extraction and visualization of high-dimensional image datasets.

```python
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                     c=y, cmap='Spectral')
plt.colorbar(scatter)
plt.title('t-SNE visualization of digits dataset')
print(f"Original shape: {X.shape}, Reduced shape: {X_tsne.shape}")
```

Slide 12: Sparse PCA Implementation

Sparse PCA introduces sparsity constraints to traditional PCA, producing principal components with fewer non-zero coefficients, which enhances interpretability and feature selection capabilities.

```python
from sklearn.decomposition import SparsePCA
import numpy as np

# Generate synthetic sparse data
n_samples, n_features = 100, 50
rng = np.random.RandomState(42)
data = rng.randn(n_samples, n_features)

# Apply Sparse PCA
spca = SparsePCA(n_components=5, alpha=1, random_state=42)
X_sparse = spca.fit_transform(data)

# Analyze sparsity
components_sparsity = np.mean(spca.components_ == 0)
print(f"Components sparsity: {components_sparsity:.2%}")

# Visualize first two components
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(spca.components_[0])
plt.title('First Sparse Component')
plt.subplot(122)
plt.plot(spca.components_[1])
plt.title('Second Sparse Component')
```

Slide 13: Incremental PCA for Large Datasets

Incremental PCA enables dimensionality reduction on datasets too large to fit in memory by processing data in batches, making it suitable for big data applications.

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Generate large dataset simulation
def data_generator(n_batches, batch_size, n_features):
    for _ in range(n_batches):
        yield np.random.randn(batch_size, n_features)

# Initialize Incremental PCA
ipca = IncrementalPCA(n_components=10)

# Process data in batches
n_batches = 10
batch_size = 100
n_features = 50

for batch in data_generator(n_batches, batch_size, n_features):
    ipca.partial_fit(batch)

# Show explained variance ratio
print("Explained variance ratio:", ipca.explained_variance_ratio_)
print("Total explained variance:", sum(ipca.explained_variance_ratio_))
```

Slide 14: Performance Comparison of Reduction Methods

This implementation compares various dimensionality reduction techniques in terms of computation time and reconstruction error on a standardized dataset.

```python
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from time import time
import numpy as np

# Generate dataset
X = np.random.randn(1000, 50)

# Compare methods
methods = {
    'PCA': PCA(n_components=2),
    'MDS': MDS(n_components=2),
    't-SNE': TSNE(n_components=2)
}

results = {}
for name, method in methods.items():
    start_time = time()
    transformed = method.fit_transform(X)
    results[name] = {
        'time': time() - start_time,
        'shape': transformed.shape
    }

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"Time: {metrics['time']:.2f}s")
    print(f"Output shape: {metrics['shape']}\n")
```

Slide 15: Additional Resources

*   Overview of dimensionality reduction techniques:
    *   [https://arxiv.org/abs/2106.04716](https://arxiv.org/abs/2106.04716)
*   Modern advances in autoencoder architectures:
    *   [https://arxiv.org/abs/2003.05991](https://arxiv.org/abs/2003.05991)
*   Comparative analysis of manifold learning methods:
    *   [https://arxiv.org/abs/2009.01796](https://arxiv.org/abs/2009.01796)
*   Deep learning approaches to dimensionality reduction:
    *   [https://arxiv.org/abs/2102.07559](https://arxiv.org/abs/2102.07559)
*   Theoretical foundations of t-SNE and UMAP:
    *   [https://www.google.com/search?q=theoretical+foundations+of+tsne+and+umap+paper](https://www.google.com/search?q=theoretical+foundations+of+tsne+and+umap+paper)

