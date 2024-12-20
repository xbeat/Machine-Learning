## Unsupervised Dimensionality Reduction in Python
Slide 1: Unsupervised Dimensionality Reduction

Unsupervised dimensionality reduction is a technique used to reduce the number of features in a dataset while preserving its essential structure. This process is crucial for handling high-dimensional data, improving computational efficiency, and facilitating visualization. In this presentation, we'll explore various methods and their implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate a sample dataset
X, _ = make_blobs(n_samples=300, n_features=3, centers=4, random_state=42)

# Visualize the 3D data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
plt.title("3D Dataset")
plt.show()
```

Slide 2: Principal Component Analysis (PCA)

PCA is one of the most popular dimensionality reduction techniques. It identifies the principal components of the data, which are the directions of maximum variance. By projecting the data onto these components, we can reduce its dimensionality while retaining most of its information.

```python
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA-reduced Dataset")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 3: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique that is particularly effective for visualizing high-dimensional data. It preserves local structure, making it useful for exploring clusters and patterns in the data.

```python
from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE-reduced Dataset")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.show()
```

Slide 4: UMAP (Uniform Manifold Approximation and Projection)

UMAP is a more recent dimensionality reduction technique that aims to preserve both local and global structure. It's often faster than t-SNE and can handle larger datasets more efficiently.

```python
import umap

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("UMAP-reduced Dataset")
plt.xlabel("UMAP Feature 1")
plt.ylabel("UMAP Feature 2")
plt.show()
```

Slide 5: Autoencoder for Dimensionality Reduction

Autoencoders are neural networks that can be used for dimensionality reduction. They consist of an encoder that compresses the data and a decoder that reconstructs it. The compressed representation in the middle layer can be used as a lower-dimensional representation of the data.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the autoencoder model
input_dim = X.shape[1]
encoding_dim = 2

input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = models.Model(input_layer, decoded)
encoder = models.Model(input_layer, encoded)

# Compile and train the model
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Use the encoder to get the reduced representation
X_encoded = encoder.predict(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1])
plt.title("Autoencoder-reduced Dataset")
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.show()
```

Slide 6: Kernel PCA

Kernel PCA is an extension of PCA that can capture nonlinear relationships in the data. It uses the kernel trick to implicitly map the data to a higher-dimensional space before applying PCA.

```python
from sklearn.decomposition import KernelPCA

# Apply Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1])
plt.title("Kernel PCA-reduced Dataset")
plt.xlabel("KPCA Feature 1")
plt.ylabel("KPCA Feature 2")
plt.show()
```

Slide 7: Truncated SVD (LSA)

Truncated SVD, also known as LSA (Latent Semantic Analysis) in text processing, is another linear dimensionality reduction technique. It's particularly useful for sparse matrices and can be faster than PCA for certain types of data.

```python
from sklearn.decomposition import TruncatedSVD

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_svd[:, 0], X_svd[:, 1])
plt.title("Truncated SVD-reduced Dataset")
plt.xlabel("SVD Feature 1")
plt.ylabel("SVD Feature 2")
plt.show()

# Print explained variance ratio
print("Explained variance ratio:", svd.explained_variance_ratio_)
```

Slide 8: Multidimensional Scaling (MDS)

MDS is a technique that preserves pairwise distances between data points in the lower-dimensional space. It can be used for both linear and nonlinear dimensionality reduction.

```python
from sklearn.manifold import MDS

# Apply MDS
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_mds[:, 0], X_mds[:, 1])
plt.title("MDS-reduced Dataset")
plt.xlabel("MDS Feature 1")
plt.ylabel("MDS Feature 2")
plt.show()
```

Slide 9: Isomap

Isomap is a nonlinear dimensionality reduction technique that attempts to preserve geodesic distances between data points. It's particularly useful for data that lies on a low-dimensional manifold embedded in a higher-dimensional space.

```python
from sklearn.manifold import Isomap

# Apply Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1])
plt.title("Isomap-reduced Dataset")
plt.xlabel("Isomap Feature 1")
plt.ylabel("Isomap Feature 2")
plt.show()
```

Slide 10: Factor Analysis

Factor Analysis is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors.

```python
from sklearn.decomposition import FactorAnalysis

# Apply Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X)

# Visualize the reduced data
plt.figure(figsize=(10, 8))
plt.scatter(X_fa[:, 0], X_fa[:, 1])
plt.title("Factor Analysis-reduced Dataset")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()
```

Slide 11: Real-life Example: Image Compression

One practical application of dimensionality reduction is image compression. We can use PCA to reduce the dimensionality of an image while preserving its main features.

```python
from sklearn.decomposition import PCA
import matplotlib.image as mpimg

# Load and preprocess the image
img = mpimg.imread('example_image.jpg')
img_gray = np.mean(img, axis=2)

# Reshape the image
img_reshaped = img_gray.reshape(-1, img_gray.shape[1])

# Apply PCA
pca = PCA(n_components=50)
img_compressed = pca.fit_transform(img_reshaped)

# Reconstruct the image
img_reconstructed = pca.inverse_transform(img_compressed)
img_reconstructed = img_reconstructed.reshape(img_gray.shape)

# Display original and reconstructed images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(img_gray, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(img_reconstructed, cmap='gray')
ax2.set_title('Reconstructed Image')
plt.show()

# Print compression ratio
original_size = img_gray.size
compressed_size = img_compressed.size
print(f"Compression ratio: {original_size / compressed_size:.2f}")
```

Slide 12: Real-life Example: Text Document Clustering

Another practical application of dimensionality reduction is in text analysis. We can use techniques like LSA (Truncated SVD) to reduce the dimensionality of text data for tasks such as document clustering.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outfoxes a lazy canine",
    "The fast and brown fox jumps over the dog",
    "Pythons are non-venomous snakes found in Asia, Africa and Australia",
    "Anacondas are large, non-venomous snakes found in South America",
    "Python is also a popular programming language"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply LSA
lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

# Cluster the documents
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_lsa)

# Visualize the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_lsa[:, 0], X_lsa[:, 1], c=clusters)
plt.title("Document Clusters")
plt.xlabel("LSA Feature 1")
plt.ylabel("LSA Feature 2")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# Print documents with their cluster assignments
for doc, cluster in zip(documents, clusters):
    print(f"Cluster {cluster}: {doc}")
```

Slide 13: Choosing the Right Dimensionality Reduction Technique

Selecting the appropriate dimensionality reduction method depends on various factors:

1. Data characteristics: Linear vs. nonlinear relationships
2. Computational resources: Some methods are more computationally intensive
3. Interpretability: PCA components are often more interpretable than t-SNE or UMAP embeddings
4. Preservation of global vs. local structure
5. Scalability to large datasets

Consider these factors and experiment with different techniques to find the best approach for your specific problem.

Slide 14: Choosing the Right Dimensionality Reduction Technique

```python
import time
from sklearn.datasets import make_swiss_roll

# Generate Swiss Roll dataset
X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# List of dimensionality reduction techniques
techniques = [
    ('PCA', PCA(n_components=2)),
    ('t-SNE', TSNE(n_components=2, random_state=42)),
    ('UMAP', umap.UMAP(random_state=42)),
    ('Isomap', Isomap(n_components=2)),
    ('MDS', MDS(n_components=2, random_state=42))
]

# Apply each technique and measure time
results = []
for name, technique in techniques:
    start_time = time.time()
    X_reduced = technique.fit_transform(X)
    end_time = time.time()
    results.append((name, X_reduced, end_time - start_time))

# Visualize results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, (name, X_reduced, runtime) in enumerate(results):
    axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1])
    axes[i].set_title(f"{name}\nRuntime: {runtime:.2f}s")

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into unsupervised dimensionality reduction, here are some valuable resources:

1. "Dimensionality Reduction: A Comparative Review" by L.J.P. van der Maaten et al. ArXiv: [https://arxiv.org/abs/0904.3367](https://arxiv.org/abs/0904.3367)
2. "How to Use t-SNE Effectively" by Martin Wattenberg et al. ArXiv: [https://arxiv.org/abs/1610.02831](https://arxiv.org/abs/1610.02831)
3. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" by Leland McInnes et al. ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

These papers provide in-depth discussions on various dimensionality reduction techniques, their properties, and applications.

