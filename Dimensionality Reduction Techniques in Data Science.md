## Dimensionality Reduction Techniques in Data Science
Slide 1: Introduction to Dimensionality Reduction

Dimensionality reduction is a crucial technique in data science for simplifying complex datasets while preserving essential information. It helps in visualizing high-dimensional data, reducing computational costs, and mitigating the curse of dimensionality.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a 3D dataset
X, _ = make_blobs(n_samples=300, n_features=3, centers=3, random_state=42)

# Visualize the 3D data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('3D Dataset')
plt.show()
```

Slide 2: Principal Component Analysis (PCA)

PCA is a widely used linear dimensionality reduction technique. It identifies the principal components (directions of maximum variance) in the data and projects the data onto these components.

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the reduced 2D data
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: 3D to 2D Reduction')
plt.show()

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

Slide 3: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique that is particularly effective for visualizing high-dimensional data. It preserves local structure and can reveal clusters in the data.

```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the t-SNE result
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE: 3D to 2D Reduction')
plt.show()
```

Slide 4: UMAP (Uniform Manifold Approximation and Projection)

UMAP is another nonlinear dimensionality reduction technique that aims to preserve both local and global structure in the data. It's often faster than t-SNE and can handle larger datasets.

```python
import umap

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize the UMAP result
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.xlabel('UMAP feature 1')
plt.ylabel('UMAP feature 2')
plt.title('UMAP: 3D to 2D Reduction')
plt.show()
```

Slide 5: Autoencoders for Dimensionality Reduction

Autoencoders are neural networks that can learn compact representations of data. They consist of an encoder that compresses the input and a decoder that reconstructs it.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the autoencoder model
input_dim = X.shape[1]
encoding_dim = 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='linear')(encoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Encode the data
X_encoded = encoder.predict(X)

# Visualize the encoded data
plt.figure(figsize=(10, 8))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1])
plt.xlabel('Encoded feature 1')
plt.ylabel('Encoded feature 2')
plt.title('Autoencoder: 3D to 2D Reduction')
plt.show()
```

Slide 6: Truncated SVD (Singular Value Decomposition)

Truncated SVD is a linear dimensionality reduction technique similar to PCA but can be applied to sparse matrices. It's particularly useful for text data and in algorithms like LSA (Latent Semantic Analysis).

```python
from sklearn.decomposition import TruncatedSVD

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Visualize the SVD result
plt.figure(figsize=(10, 8))
plt.scatter(X_svd[:, 0], X_svd[:, 1])
plt.xlabel('SVD component 1')
plt.ylabel('SVD component 2')
plt.title('Truncated SVD: 3D to 2D Reduction')
plt.show()

# Print explained variance ratio
print(f"Explained variance ratio: {svd.explained_variance_ratio_}")
```

Slide 7: Factor Analysis

Factor Analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors.

```python
from sklearn.decomposition import FactorAnalysis

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X)

# Visualize the Factor Analysis result
plt.figure(figsize=(10, 8))
plt.scatter(X_fa[:, 0], X_fa[:, 1])
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Factor Analysis: 3D to 2D Reduction')
plt.show()
```

Slide 8: Multidimensional Scaling (MDS)

MDS is a technique that represents high-dimensional data in a lower-dimensional space while preserving the distances between data points as much as possible.

```python
from sklearn.manifold import MDS

# Apply MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)

# Visualize the MDS result
plt.figure(figsize=(10, 8))
plt.scatter(X_mds[:, 0], X_mds[:, 1])
plt.xlabel('MDS dimension 1')
plt.ylabel('MDS dimension 2')
plt.title('MDS: 3D to 2D Reduction')
plt.show()
```

Slide 9: Isomap

Isomap is a nonlinear dimensionality reduction technique that preserves geodesic distances between data points. It's particularly useful for data that lies on a low-dimensional manifold within a high-dimensional space.

```python
from sklearn.manifold import Isomap

# Apply Isomap
isomap = Isomap(n_components=2, n_neighbors=5)
X_isomap = isomap.fit_transform(X)

# Visualize the Isomap result
plt.figure(figsize=(10, 8))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1])
plt.xlabel('Isomap dimension 1')
plt.ylabel('Isomap dimension 2')
plt.title('Isomap: 3D to 2D Reduction')
plt.show()
```

Slide 10: Locally Linear Embedding (LLE)

LLE is another nonlinear dimensionality reduction technique that preserves local properties of the data. It works by representing each data point as a linear combination of its neighbors.

```python
from sklearn.manifold import LocallyLinearEmbedding

# Apply LLE
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_lle = lle.fit_transform(X)

# Visualize the LLE result
plt.figure(figsize=(10, 8))
plt.scatter(X_lle[:, 0], X_lle[:, 1])
plt.xlabel('LLE dimension 1')
plt.ylabel('LLE dimension 2')
plt.title('LLE: 3D to 2D Reduction')
plt.show()
```

Slide 11: Kernel PCA

Kernel PCA is a nonlinear extension of PCA that uses kernel methods to find principal components in a high-dimensional feature space.

```python
from sklearn.decomposition import KernelPCA

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
X_kpca = kpca.fit_transform(X)

# Visualize the Kernel PCA result
plt.figure(figsize=(10, 8))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1])
plt.xlabel('Kernel PCA dimension 1')
plt.ylabel('Kernel PCA dimension 2')
plt.title('Kernel PCA: 3D to 2D Reduction')
plt.show()
```

Slide 12: Real-life Example: Image Compression

Dimensionality reduction techniques can be used for image compression. Here's an example using PCA to compress a grayscale image:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import data

# Load a sample image
image = data.camera()

# Reshape the image
X = image.reshape(-1, image.shape[1])

# Apply PCA
pca = PCA(0.95)  # Keep 95% of variance
X_compressed = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_compressed)

# Reshape back to image
reconstructed_image = X_reconstructed.reshape(image.shape)

# Visualize original and reconstructed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(reconstructed_image, cmap='gray')
ax2.set_title('Reconstructed Image')
plt.show()

print(f"Original shape: {X.shape}, Compressed shape: {X_compressed.shape}")
print(f"Compression ratio: {X.size / X_compressed.size:.2f}")
```

Slide 13: Real-life Example: Text Document Clustering

Dimensionality reduction is crucial in text analysis for reducing the high-dimensional space of word features. Here's an example using TruncatedSVD (LSA) for document clustering:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Neural networks are used in deep learning",
    "Natural language processing deals with text data",
    "Computer vision focuses on image and video analysis",
    "Reinforcement learning is used in game AI"
]

# Convert documents to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply TruncatedSVD (LSA)
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

# Cluster the documents
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_lsa)

# Visualize the clusters
plt.figure(figsize=(10, 8))
for i, doc in enumerate(documents):
    plt.scatter(X_lsa[i, 0], X_lsa[i, 1], c=clusters[i], cmap='viridis')
    plt.annotate(f"Doc {i+1}", (X_lsa[i, 0], X_lsa[i, 1]))
plt.xlabel('LSA component 1')
plt.ylabel('LSA component 2')
plt.title('Document Clustering using LSA')
plt.colorbar(ticks=range(2), label='Cluster')
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into dimensionality reduction techniques, here are some valuable resources:

1. "Dimensionality Reduction: A Comparative Review" by L.J.P. van der Maaten, E.O. Postma, and H.J. van den Herik (2008) ArXiv: [https://arxiv.org/abs/1403.2210](https://arxiv.org/abs/1403.2210)
2. "A Global Geometric Framework for Nonlinear Dimensionality Reduction" by J.B. Tenenbaum, V. de Silva, and J.C. Langford (2000) Science, Vol. 290, Issue 5500, pp. 2319-2323 DOI: 10.1126/science.290.5500.2319
3. "Visualizing Data using t-SNE" by L.J.P. van der Maaten and G.E. Hinton (2008) Journal of Machine Learning Research, Vol. 9, pp. 2579-2605 [https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

These resources provide in-depth explanations and mathematical foundations of various dimensionality reduction techniques.

