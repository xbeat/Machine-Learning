## Dimensionality Reduction in Python
Slide 1: Introduction to Dimensionality Reduction

Dimensionality reduction is a crucial technique in data science and machine learning, used to simplify complex datasets while preserving essential information. It helps in visualizing high-dimensional data, reducing computational complexity, and mitigating the curse of dimensionality.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample high-dimensional data
np.random.seed(42)
X = np.random.randn(100, 50)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("2D Representation of 50D Data")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
```

Slide 2: Principal Component Analysis (PCA)

PCA is one of the most popular dimensionality reduction techniques. It works by identifying the principal components, which are the directions of maximum variance in the data. These components are orthogonal to each other and capture the most important patterns in the dataset.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))
```

Slide 3: t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique that is particularly effective for visualizing high-dimensional data. It works by minimizing the divergence between two distributions: one that measures pairwise similarities in the high-dimensional space and another in the low-dimensional space.

```python
from sklearn.manifold import TSNE
import seaborn as sns

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=iris.target, palette='deep')
plt.title("t-SNE visualization of Iris dataset")
plt.show()
```

Slide 4: UMAP (Uniform Manifold Approximation and Projection)

UMAP is another powerful nonlinear dimensionality reduction technique. It's based on manifold learning techniques and topological data analysis. UMAP often provides better preservation of global structure than t-SNE while maintaining computational efficiency.

```python
import umap

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize the result
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=iris.target, palette='deep')
plt.title("UMAP visualization of Iris dataset")
plt.show()
```

Slide 5: Autoencoders for Dimensionality Reduction

Autoencoders are neural networks that can be used for dimensionality reduction. They consist of an encoder that compresses the input data and a decoder that reconstructs it. The bottleneck layer in the middle represents the reduced-dimensional space.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the autoencoder architecture
input_dim = X.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Use the encoder to get the reduced representation
X_encoded = encoder.predict(X)
```

Slide 6: Feature Selection vs. Feature Extraction

Dimensionality reduction can be achieved through feature selection or feature extraction. Feature selection involves choosing a subset of the original features, while feature extraction creates new features by combining the original ones. PCA is an example of feature extraction, while methods like Lasso can be used for feature selection.

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Feature selection using ANOVA F-value
selector = SelectKBest(f_classif, k=2)
X_selected = selector.fit_transform(X, iris.target)

# Print selected feature indices
print("Selected feature indices:", selector.get_support(indices=True))

# Visualize selected features
plt.scatter(X_selected[:, 0], X_selected[:, 1], c=iris.target)
plt.title("Selected Features")
plt.show()
```

Slide 7: Curse of Dimensionality

The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces. As the number of dimensions increases, the volume of the space increases so fast that the available data become sparse, making statistical analysis challenging.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(dim, num_points=1000):
    return np.random.random((num_points, dim))

def calculate_pairwise_distances(points):
    return np.linalg.norm(points[:, np.newaxis] - points, axis=2)

dims = range(1, 101, 10)
avg_distances = []

for dim in dims:
    points = generate_random_points(dim)
    distances = calculate_pairwise_distances(points)
    avg_distances.append(np.mean(distances))

plt.plot(dims, avg_distances)
plt.xlabel("Number of Dimensions")
plt.ylabel("Average Pairwise Distance")
plt.title("Effect of Dimensionality on Average Pairwise Distance")
plt.show()
```

Slide 8: Manifold Learning

Manifold learning is based on the assumption that high-dimensional data often lies on or near a lower-dimensional manifold. Techniques like Isomap and Locally Linear Embedding (LLE) try to discover this underlying manifold structure.

```python
from sklearn.manifold import Isomap, LocallyLinearEmbedding

# Apply Isomap
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)

# Apply LLE
lle = LocallyLinearEmbedding(n_components=2)
X_lle = lle.fit_transform(X)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(X_isomap[:, 0], X_isomap[:, 1], c=iris.target)
ax1.set_title("Isomap")

ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=iris.target)
ax2.set_title("Locally Linear Embedding")

plt.show()
```

Slide 9: Truncated SVD (LSA)

Truncated SVD, also known as Latent Semantic Analysis (LSA) in text processing, is a linear dimensionality reduction technique. It's particularly useful for sparse matrices and is often applied in text mining and natural language processing.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
texts = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outfoxes a lazy fox",
    "The lazy fox is quickly outfoxed by the dog"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# Visualize the result
plt.scatter(X_svd[:, 0], X_svd[:, 1])
plt.title("Truncated SVD on Text Data")
for i, text in enumerate(texts):
    plt.annotate(f"Text {i+1}", (X_svd[i, 0], X_svd[i, 1]))
plt.show()
```

Slide 10: Real-Life Example: Image Compression

Dimensionality reduction can be used for image compression. By applying PCA to image data, we can retain the most important features while reducing file size.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import data

# Load sample image
image = data.camera()

# Reshape image to 2D array
X = image.reshape(-1, image.shape[1])

# Apply PCA with different numbers of components
n_components = [10, 50, 100, 200]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for ax, n in zip(axes.ravel(), n_components):
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Reshape back to image
    img_reconstructed = X_reconstructed.reshape(image.shape)
    
    ax.imshow(img_reconstructed, cmap='gray')
    ax.set_title(f"{n} components")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example: Anomaly Detection

Dimensionality reduction can be used for anomaly detection by identifying data points that deviate significantly from the reduced representation.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate normal data and anomalies
np.random.seed(42)
normal_data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)
anomalies = np.random.multivariate_normal(mean=[3, 3], cov=[[1, 0.5], [0.5, 1]], size=20)
X = np.vstack((normal_data, anomalies))

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Reconstruct the data
X_reconstructed = pca.inverse_transform(X_pca)

# Calculate reconstruction error
mse = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=mse, cmap='viridis')
plt.colorbar(label='Reconstruction Error')
plt.title("Anomaly Detection using PCA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Slide 12: Choosing the Right Dimensionality Reduction Technique

Selecting the appropriate dimensionality reduction method depends on various factors such as the nature of your data, the desired output dimensionality, and the specific requirements of your task. Consider factors like linearity vs. nonlinearity, computational efficiency, and interpretability when making your choice.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Generate sample data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 50)  # 50-dimensional data

# Apply different dimensionality reduction techniques
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
umap_reducer = umap.UMAP(random_state=42)

X_pca = pca.fit_transform(X)
X_tsne = tsne.fit_transform(X)
X_umap = umap_reducer.fit_transform(X)

# Visualize results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X_pca[:, 0], X_pca[:, 1])
ax1.set_title("PCA")

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1])
ax2.set_title("t-SNE")

ax3.scatter(X_umap[:, 0], X_umap[:, 1])
ax3.set_title("UMAP")

plt.tight_layout()
plt.show()
```

Slide 13: Evaluating Dimensionality Reduction

Assessing the quality of dimensionality reduction is crucial. Common evaluation metrics include explained variance ratio, reconstruction error, and the preservation of pairwise distances or local structure.

```python
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

def evaluate_dim_reduction(X_original, X_reduced):
    # Calculate pairwise distances in original and reduced space
    dist_original = pairwise_distances(X_original)
    dist_reduced = pairwise_distances(X_reduced)
    
    # Flatten distance matrices
    dist_original_flat = dist_original[np.triu_indices(dist_original.shape[0], k=1)]
    dist_reduced_flat = dist_reduced[np.triu_indices(dist_reduced.shape[0], k=1)]
    
    # Calculate Spearman correlation
    correlation, _ = spearmanr(dist_original_flat, dist_reduced_flat)
    
    return correlation

# Evaluate PCA, t-SNE, and UMAP
pca_score = evaluate_dim_reduction(X, X_pca)
tsne_score = evaluate_dim_reduction(X, X_tsne)
umap_score = evaluate_dim_reduction(X, X_umap)

print(f"PCA distance preservation: {pca_score:.4f}")
print(f"t-SNE distance preservation: {tsne_score:.4f}")
print(f"UMAP distance preservation: {umap_score:.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into dimensionality reduction techniques and their applications, the following resources are recommended:

1. "Dimensionality Reduction: A Comparative Review" by L.J.P. van der Maaten, E.O. Postma, and H.J. van den Herik (2008) ArXiv: [https://arxiv.org/abs/0904.3841](https://arxiv.org/abs/0904.3841)
2. "Visualizing Data using t-SNE" by L.J.P. van der Maaten and G.E. Hinton (2008) Journal of Machine Learning Research
3. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" by L. McInnes, J. Healy, and J. Melville (2018) ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)

These papers provide in-depth discussions of various dimensionality reduction techniques, their mathematical foundations, and practical applications.

