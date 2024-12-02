## Dimensionality Reduction Techniques in Python
Slide 1: Principal Component Analysis (PCA)

Principal Component Analysis is a fundamental dimensionality reduction technique that transforms high-dimensional data into lower dimensions while preserving maximum variance. It works by identifying orthogonal directions (principal components) that capture the most significant variations in the data.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)

# Initialize and fit PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance ratio: {np.cumsum(pca.explained_variance_ratio_)}")
```

Slide 2: Mathematical Foundation of PCA

The mathematical foundation of PCA revolves around eigendecomposition of the covariance matrix. The principal components are eigenvectors corresponding to the largest eigenvalues of the covariance matrix.

```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    
    # Project data
    return np.dot(X_centered, components)
```

Slide 3: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique that emphasizes the preservation of local structure in the data. It's particularly effective for visualizing high-dimensional data by maintaining the relative distances between points.

```python
from sklearn.manifold import TSNE
import seaborn as sns

def apply_tsne(X, perplexity=30, n_components=2):
    tsne = TSNE(n_components=n_components, 
                perplexity=perplexity, 
                random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1])
    plt.title('t-SNE visualization')
    plt.show()
```

Slide 4: UMAP (Uniform Manifold Approximation and Projection)

UMAP is a modern dimensionality reduction algorithm that combines mathematical foundations from manifold learning and topological data analysis. It often provides better preservation of global structure than t-SNE while maintaining computational efficiency.

```python
import umap
import pandas as pd

def apply_umap(X, n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       random_state=42)
    X_umap = reducer.fit_transform(X)
    
    # Create DataFrame for visualization
    df_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(df_umap['UMAP1'], df_umap['UMAP2'], alpha=0.6)
    plt.title('UMAP projection')
    plt.show()
```

Slide 5: Autoencoder for Dimensionality Reduction

Autoencoders provide a neural network-based approach to dimensionality reduction, learning a compressed representation of the input data through an encoding-decoding process that minimizes reconstruction error.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def create_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Full autoencoder
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
```

Slide 6: Real-world Application - Image Dimensionality Reduction

This example demonstrates dimensionality reduction on the MNIST dataset, comparing PCA, t-SNE, and UMAP approaches for visualizing high-dimensional image data in a 2D space for pattern recognition tasks.

```python
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
digits = load_digits()
X = digits.data
y = digits.target
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting
plt.figure(figsize=(12, 4))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA of MNIST digits')
plt.colorbar()
plt.show()
```

Slide 7: Kernel PCA Implementation

Kernel PCA extends traditional PCA by using kernel methods to perform dimensionality reduction in an implicit feature space, making it capable of capturing nonlinear patterns in the data through different kernel functions.

```python
from sklearn.decomposition import KernelPCA
import numpy as np

def apply_kernel_pca(X, n_components=2, kernel='rbf'):
    # Initialize and fit KernelPCA
    kpca = KernelPCA(n_components=n_components,
                     kernel=kernel,
                     random_state=42)
    X_kpca = kpca.fit_transform(X)
    
    # Compute explained variance (approximated)
    explained_var = np.var(X_kpca, axis=0)
    explained_var_ratio = explained_var / np.sum(explained_var)
    
    return X_kpca, explained_var_ratio
```

Slide 8: Locally Linear Embedding (LLE)

LLE is a manifold learning technique that preserves the local geometry of the data by reconstructing each point from its neighbors, making it particularly effective for data that lies on a nonlinear manifold.

```python
from sklearn.manifold import LocallyLinearEmbedding

def apply_lle(X, n_neighbors=10, n_components=2):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                n_components=n_components,
                                random_state=42)
    X_lle = lle.fit_transform(X)
    
    # Reconstruction error
    error = lle.reconstruction_error_
    print(f"LLE Reconstruction error: {error}")
    
    return X_lle
```

Slide 9: Factor Analysis Implementation

Factor Analysis assumes that observed variables can be modeled as linear combinations of unobserved latent factors plus error terms, providing a probabilistic approach to dimensionality reduction.

```python
from sklearn.decomposition import FactorAnalysis
import numpy as np

def apply_factor_analysis(X, n_components=2):
    # Initialize and fit Factor Analysis
    fa = FactorAnalysis(n_components=n_components,
                       random_state=42)
    X_fa = fa.fit_transform(X)
    
    # Get components and noise variances
    components = fa.components_
    noise_variance = fa.noise_variance_
    
    return X_fa, components, noise_variance
```

Slide 10: Multidimensional Scaling (MDS)

MDS aims to preserve the pairwise distances between points in the high-dimensional space when projecting to lower dimensions, offering both metric and non-metric variants for different types of distance preservation.

```python
from sklearn.manifold import MDS
import numpy as np

def apply_mds(X, n_components=2, metric=True):
    # Initialize and fit MDS
    mds = MDS(n_components=n_components,
              metric=metric,
              random_state=42)
    X_mds = mds.fit_transform(X)
    
    # Compute stress (goodness of fit)
    stress = mds.stress_
    print(f"MDS Stress: {stress}")
    
    return X_mds
```

Slide 11: Comparative Analysis with Synthetic Data

This implementation creates synthetic data with known structure to compare the effectiveness of different dimensionality reduction techniques, providing metrics for quantitative evaluation of their performance.

```python
from sklearn.datasets import make_swiss_roll
import numpy as np
from sklearn.metrics import trustworthiness

def compare_reduction_methods(n_samples=1000):
    # Generate swiss roll dataset
    X, color = make_swiss_roll(n_samples, random_state=42)
    
    # Apply different methods
    methods = {
        'PCA': PCA(n_components=2),
        'tSNE': TSNE(n_components=2, random_state=42),
        'UMAP': umap.UMAP(random_state=42),
        'MDS': MDS(n_components=2, random_state=42)
    }
    
    results = {}
    for name, method in methods.items():
        X_reduced = method.fit_transform(X)
        # Calculate trustworthiness
        trust = trustworthiness(X, X_reduced, n_neighbors=10)
        results[name] = {'embedding': X_reduced, 'trust': trust}
        print(f"{name} trustworthiness: {trust:.4f}")
    
    return results
```

Slide 12: Real-world Application - Gene Expression Analysis

In this practical application, we analyze high-dimensional gene expression data, demonstrating how dimensionality reduction can reveal hidden patterns in biological datasets.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def analyze_gene_expression(expression_matrix):
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expression_matrix)
    
    # Apply PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_pca = pca.fit_transform(X_scaled)
    
    # Apply UMAP for visualization
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X_pca)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    return X_umap, explained_var, cumulative_var
```

Slide 13: Mathematical Formulations in Dimensionality Reduction

A comprehensive overview of the mathematical foundations underlying various dimensionality reduction techniques, presented with their core equations and theoretical basis.

```python
# Mathematical formulations for different techniques

# PCA objective function
"""
$$\arg\max_{W} \frac{1}{n}\sum_{i=1}^n (x_i^T W)^T (x_i^T W)$$
"""

# t-SNE probability computation
"""
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$
"""

# UMAP fuzzy topological representation
"""
$$\mu_{Z}(x) = \exp(-\frac{d(x,Z)}{\rho_0})$$
"""

# Autoencoder loss function
"""
$$L(x,x') = ||x - x'||^2 + \lambda \sum_{l=1}^{L} ||W^{(l)}||_F^2$$
"""
```

Slide 14: Additional Resources

*   ArXiv: "A Survey of Dimensionality Reduction Techniques" - [https://arxiv.org/abs/2007.07844](https://arxiv.org/abs/2007.07844)
*   ArXiv: "Understanding UMAP" - [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
*   ArXiv: "Visualizing Data using t-SNE" - [https://arxiv.org/abs/1807.01882](https://arxiv.org/abs/1807.01882)
*   General Resource: [https://scikit-learn.org/stable/modules/manifold.html](https://scikit-learn.org/stable/modules/manifold.html)
*   Tutorial Collection: [https://towardsdatascience.com/dimensionality-reduction-techniques-comparison-573cd6b357cb](https://towardsdatascience.com/dimensionality-reduction-techniques-comparison-573cd6b357cb)

