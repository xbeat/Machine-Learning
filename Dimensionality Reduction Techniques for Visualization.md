## Dimensionality Reduction Techniques for Visualization
Slide 1: Principal Component Analysis (PCA)

Principal Component Analysis is a fundamental dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. It works by finding orthogonal directions, called principal components, along which the data shows the highest variation.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def pca_from_scratch(X, n_components):
    # Standardize the data
    X_std = StandardScaler().fit_transform(X)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_std.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components eigenvectors
    selected_vectors = eigenvectors[:, :n_components]
    
    # Project data onto new space
    return np.dot(X_std, selected_vectors)

# Example usage
X = np.random.rand(100, 5)  # 100 samples, 5 features
X_reduced = pca_from_scratch(X, n_components=2)
print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")
```

Slide 2: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction algorithm that emphasizes the preservation of local structure in the data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of low-dimensional embedding and high-dimensional data.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_visualization(X, perplexity=30, n_iter=1000):
    # Initialize t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, 
                n_iter=n_iter, random_state=42)
    
    # Fit and transform the data
    X_tsne = tsne.fit_transform(X)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    return X_tsne

# Example usage
X = np.random.rand(1000, 50)  # 1000 samples, 50 features
X_embedded = tsne_visualization(X)
```

Slide 3: UMAP (Uniform Manifold Approximation and Projection)

UMAP is a dimension reduction technique that preserves both local and global structure of the data. It constructs a topological representation using local manifold approximations and fuzzy topological structures, then optimizes the low-dimensional embedding using cross-entropy minimization.

```python
import umap
import numpy as np
import matplotlib.pyplot as plt

def umap_reduction(X, n_neighbors=15, min_dist=0.1):
    # Initialize UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                       min_dist=min_dist,
                       random_state=42)
    
    # Fit and transform data
    X_umap = reducer.fit_transform(X)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.5)
    plt.title('UMAP Projection')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    return X_umap

# Example usage
X = np.random.rand(1000, 30)  # 1000 samples, 30 features
X_reduced = umap_reduction(X)
```

Slide 4: Autoencoder Dimensionality Reduction

Autoencoders are neural networks that learn compressed representations of data through an encoding-decoding process. The encoder reduces dimensionality while the decoder reconstructs the original data, making them powerful for nonlinear dimensionality reduction.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim*2, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(encoding_dim*2, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Example usage
input_dim = 784  # e.g., for MNIST
encoding_dim = 32
autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
```

Slide 5: Kernel PCA Implementation

Kernel PCA extends traditional PCA by using kernel methods to perform nonlinear dimensionality reduction. It implicitly maps data into a higher-dimensional feature space where linear PCA is performed, allowing the capture of nonlinear patterns in the original data space.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def kernel_function(X1, X2, kernel='rbf', gamma=1.0):
    if kernel == 'rbf':
        # Radial basis function kernel
        pairwise_sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
                           np.sum(X2**2, axis=1) - \
                           2 * np.dot(X1, X2.T)
        return np.exp(-gamma * pairwise_sq_dists)
    
def kernel_pca(X, n_components, kernel='rbf', gamma=1.0):
    # Center the data
    X_std = StandardScaler().fit_transform(X)
    
    # Compute kernel matrix
    K = kernel_function(X_std, X_std, kernel, gamma)
    
    # Center kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    
    # Sort eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    return np.dot(K_centered, eigenvectors[:, :n_components])

# Example usage
X = np.random.rand(100, 10)
X_kpca = kernel_pca(X, n_components=2)
```

Slide 6: Locally Linear Embedding (LLE)

LLE preserves the local geometry of high-dimensional data by representing each point as a weighted linear combination of its neighbors. These local relationships are then used to find a low-dimensional embedding that preserves these neighborhoods.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def lle_reduction(X, n_components=2, n_neighbors=10):
    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Compute weights for each point
    weights = np.zeros((X.shape[0], n_neighbors))
    for i in range(X.shape[0]):
        Xi = X[i]
        Xi_neighbors = X[indices[i]]
        
        # Local covariance
        C = np.dot(Xi_neighbors - Xi, (Xi_neighbors - Xi).T)
        C += np.eye(n_neighbors) * 1e-3  # Regularization
        
        # Solve for weights
        w = np.linalg.solve(C, np.ones(n_neighbors))
        weights[i] = w / np.sum(w)
    
    # Construct sparse matrix M
    M = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j, idx in enumerate(indices[i]):
            M[i, idx] = weights[i, j]
    
    # Compute embedding
    I = np.eye(X.shape[0])
    M = (I - M).T.dot(I - M)
    
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    return eigenvectors[:, 1:n_components+1]

# Example usage
X = np.random.rand(200, 20)
X_embedded = lle_reduction(X, n_components=2)
```

Slide 7: Real-World Application: Gene Expression Analysis

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def analyze_gene_expression(expression_matrix):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expression_matrix)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_
    
    # Create results DataFrame
    pca_df = pd.DataFrame(data=pca_result, 
                         columns=['PC1', 'PC2'])
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2')
    plt.title('Gene Expression PCA')
    
    return pca_df, variance_explained

# Example usage with synthetic data
n_genes = 1000
n_samples = 100
expression_data = np.random.normal(0, 1, (n_samples, n_genes))
results, var_explained = analyze_gene_expression(expression_data)
print(f"Variance explained: {var_explained}")
```

Slide 8: Results Analysis for Gene Expression

This slide demonstrates statistical analysis and visualization of the gene expression PCA results, including variance explained ratios, loading scores, and feature importance metrics for biological interpretation.

```python
def analyze_pca_results(pca, feature_names, pca_results):
    # Calculate loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    # Get top contributing features
    n_top = 10
    loading_scores = pd.DataFrame()
    for pc in ['PC1', 'PC2']:
        sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
        loading_scores[f'{pc}_feature'] = sorted_loadings.index[:n_top]
        loading_scores[f'{pc}_score'] = sorted_loadings.values[:n_top]
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Scree plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    # Biplot
    plt.subplot(1, 2, 2)
    plt.scatter(pca_results[:, 0], pca_results[:, 1], alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    return loading_scores

# Example usage
feature_names = [f'Gene_{i}' for i in range(1000)]
loading_analysis = analyze_pca_results(pca, feature_names, pca_result)
print("Top contributing genes:\n", loading_analysis)
```

Slide 9: Dimensionality Reduction for Image Data

A practical implementation showing how dimensionality reduction techniques can be applied to image data for visualization and compression, demonstrating the balance between data reduction and information preservation.

```python
import cv2
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def reduce_image_dimensionality(image_path, n_components=50):
    # Read and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_normalized = img.astype(float) / 255.0
    
    # Apply PCA to each channel
    pca = PCA(n_components=n_components)
    img_transformed = pca.fit_transform(img_normalized)
    img_reconstructed = pca.inverse_transform(img_transformed)
    
    # Calculate compression ratio and error
    original_size = img.shape[0] * img.shape[1]
    compressed_size = n_components * (img.shape[0] + img.shape[1])
    compression_ratio = original_size / compressed_size
    mse = np.mean((img_normalized - img_reconstructed) ** 2)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_normalized, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(img_reconstructed, cmap='gray')
    axes[1].set_title(f'Reconstructed ({n_components} components)')
    
    return compression_ratio, mse, img_reconstructed

# Example usage
compression_ratio, mse, reconstructed = reduce_image_dimensionality(
    'sample_image.jpg', n_components=50)
print(f"Compression ratio: {compression_ratio:.2f}")
print(f"MSE: {mse:.6f}")
```

Slide 10: Implementation of Multidimensional Scaling (MDS)

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def mds_reduction(X, n_components=2, max_iter=300):
    # Calculate pairwise distances
    distances = squareform(pdist(X))
    
    # Initialize random coordinates
    Y = np.random.rand(X.shape[0], n_components)
    
    for iter in range(max_iter):
        # Calculate pairwise distances in embedded space
        embedded_distances = squareform(pdist(Y))
        
        # Compute stress
        stress = np.sum((distances - embedded_distances) ** 2)
        
        # Gradient descent
        gradient = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i != j:
                    diff = Y[i] - Y[j]
                    gradient[i] += 2 * (embedded_distances[i,j] - 
                                      distances[i,j]) * diff / (embedded_distances[i,j] + 1e-7)
        
        # Update coordinates
        Y -= 0.001 * gradient
        
        if iter % 50 == 0:
            print(f"Iteration {iter}, Stress: {stress:.4f}")
    
    return Y

# Example usage
X = np.random.rand(100, 20)
X_mds = mds_reduction(X)
```

Slide 11: Isomap Implementation

Isomap extends MDS by attempting to preserve geodesic distances along the manifold rather than straight-line Euclidean distances between points. This implementation demonstrates the core algorithm including neighborhood graph construction and geodesic distance computation.

```python
import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph

def isomap_reduction(X, n_components=2, n_neighbors=5):
    # Build neighborhood graph
    connectivity = kneighbors_graph(
        X, n_neighbors=n_neighbors, mode='distance'
    ).toarray()
    
    # Compute geodesic distances
    geodesic_distances = shortest_path(connectivity)
    
    # Center the distance matrix
    n = geodesic_distances.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(geodesic_distances ** 2).dot(H)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute embedding
    embedding = eigenvectors[:, :n_components] * np.sqrt(eigenvalues[:n_components])
    return embedding

# Example usage
X = np.random.rand(200, 10)
X_isomap = isomap_reduction(X)
```

Slide 12: Real-World Application: Document Embedding

This implementation demonstrates dimensionality reduction for text documents using TF-IDF vectorization followed by UMAP reduction, enabling visualization of document similarities and clusters.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import matplotlib.pyplot as plt

def reduce_document_dimensions(documents, n_components=2):
    # Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        stop_words='english'
    )
    X = vectorizer.fit_transform(documents)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        metric='cosine',
        n_neighbors=15,
        min_dist=0.1
    )
    
    embedding = reducer.fit_transform(X.toarray())
    
    # Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
    plt.title('Document Embedding Visualization')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    return embedding, vectorizer.get_feature_names_out()

# Example usage
documents = [
    "Machine learning is fascinating",
    "Deep neural networks excel at pattern recognition",
    "Natural language processing advances rapidly"
]
embedding, features = reduce_document_dimensions(documents)
```

Slide 13: Additional Resources

*   Dimensionality Reduction for Visualization: [https://arxiv.org/abs/1708.08992](https://arxiv.org/abs/1708.08992)
*   Comparative Study of Dimension Reduction Methods: [https://arxiv.org/abs/2012.04176](https://arxiv.org/abs/2012.04176)
*   Modern Manifold Learning Techniques: [https://arxiv.org/abs/2011.01307](https://arxiv.org/abs/2011.01307)
*   Recommended search terms for more resources:
    *   "manifold learning algorithms comparison"
    *   "nonlinear dimensionality reduction techniques"
    *   "visualization high dimensional data methods"

