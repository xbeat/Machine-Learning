## Dimensionality Reduction Methods for Preserving Data Variance
Slide 1: Principal Component Analysis (PCA) for Dimensionality Reduction

PCA is a powerful technique for reducing dimensionality while preserving the maximum variance in the data. It works by identifying the principal components, which are orthogonal vectors that capture the most significant patterns in the dataset.

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 5)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot results
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Reduced Data')
plt.show()
```

Slide 2: Explained Variance Ratio

The explained variance ratio helps us understand how much information is retained after dimensionality reduction. It represents the proportion of variance explained by each principal component.

```python
# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

print(f"Total variance explained: {sum(explained_variance_ratio):.2f}")
```

Slide 3: Selecting the Number of Components

Choosing the right number of components is crucial. We can use the cumulative explained variance ratio to determine how many components to keep while preserving a desired amount of information.

```python
# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()

# Find the number of components needed to explain 95% of variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components needed to explain 95% of variance: {n_components_95}")
```

Slide 4: Reconstructing Data from Reduced Dimensions

After reducing dimensionality, we can reconstruct the original data to assess the quality of our reduction. This process helps us understand how much information is lost during dimensionality reduction.

```python
# Reduce dimensionality and reconstruct
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate reconstruction error
reconstruction_error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
print(f"Mean reconstruction error: {reconstruction_error:.4f}")

# Visualize original vs reconstructed data (first two dimensions)
plt.scatter(X[:, 0], X[:, 1], label='Original', alpha=0.5)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], label='Reconstructed', alpha=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()
```

Slide 5: Real-Life Example: Image Compression

PCA can be used for image compression by reducing the dimensionality of image data. This example demonstrates how PCA can compress a grayscale image while preserving its main features.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load digit dataset
digits = load_digits()
X = digits.data
y = digits.target

# Select a single digit image
digit_image = X[0].reshape(8, 8)

# Apply PCA with different numbers of components
n_components_list = [2, 5, 10, 20, 30, 40]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Image Reconstruction with Different Numbers of Components')

for i, n_comp in enumerate(n_components_list):
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)
    
    reconstructed_image = X_reconstructed[0].reshape(8, 8)
    
    ax = axes[i // 3, i % 3]
    ax.imshow(reconstructed_image, cmap='gray')
    ax.set_title(f'{n_comp} Components')
    ax.axis('off')

plt.tight_layout()
plt.show()
```

Slide 6: Kernel PCA for Non-Linear Dimensionality Reduction

Kernel PCA extends PCA to handle non-linear relationships in the data by projecting the original features into a higher-dimensional space using a kernel function.

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Generate non-linear data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Apply Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Original Data')
plt.subplot(122)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title('Kernel PCA Transformed Data')
plt.tight_layout()
plt.show()
```

Slide 7: Incremental PCA for Large Datasets

When dealing with large datasets that don't fit into memory, Incremental PCA allows us to perform dimensionality reduction by processing the data in batches.

```python
from sklearn.decomposition import IncrementalPCA
import numpy as np

# Generate a large dataset
n_samples, n_features = 10000, 100
np.random.seed(42)
X = np.random.randn(n_samples, n_features)

# Apply Incremental PCA
batch_size = 1000
ipca = IncrementalPCA(n_components=10, batch_size=batch_size)

for i in range(0, n_samples, batch_size):
    ipca.partial_fit(X[i:i+batch_size])

# Transform the data
X_reduced = ipca.transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Explained variance ratio: {ipca.explained_variance_ratio_.sum():.2f}")
```

Slide 8: Sparse PCA for Feature Selection

Sparse PCA combines the benefits of PCA with feature selection by enforcing sparsity in the principal components, leading to more interpretable results.

```python
from sklearn.decomposition import SparsePCA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
n_samples, n_features = 100, 20
X = np.random.randn(n_samples, n_features)

# Apply Sparse PCA
spca = SparsePCA(n_components=5, alpha=1, random_state=42)
X_sparse = spca.fit_transform(X)

# Visualize component sparsity
plt.figure(figsize=(12, 6))
plt.imshow(spca.components_.T, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Sparse PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Original Feature')
plt.tight_layout()
plt.show()

print(f"Sparsity ratio: {np.sum(spca.components_ == 0) / spca.components_.size:.2f}")
```

Slide 9: Real-Life Example: Text Document Analysis

PCA can be used to analyze and visualize relationships between text documents by reducing the dimensionality of their term-frequency representations.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample documents
documents = [
    "Machine learning is a subfield of artificial intelligence",
    "Natural language processing deals with text and speech",
    "Deep learning uses neural networks with many layers",
    "Computer vision focuses on image and video analysis",
    "Reinforcement learning is about decision making and rewards"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (X_pca[i, 0], X_pca[i, 1]))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Text Documents')
plt.tight_layout()
plt.show()
```

Slide 10: Truncated SVD (LSA) for Sparse Data

Truncated SVD, also known as Latent Semantic Analysis (LSA) in text processing, is particularly useful for dimensionality reduction of sparse data, such as TF-IDF matrices.

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Sample documents (reusing from previous slide)
documents = [
    "Machine learning is a subfield of artificial intelligence",
    "Natural language processing deals with text and speech",
    "Deep learning uses neural networks with many layers",
    "Computer vision focuses on image and video analysis",
    "Reinforcement learning is about decision making and rewards"
]

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(X_svd[:, 0], X_svd[:, 1])
for i, doc in enumerate(documents):
    plt.annotate(f"Doc {i+1}", (X_svd[i, 0], X_svd[i, 1]))
plt.xlabel('First SVD Component')
plt.ylabel('Second SVD Component')
plt.title('Truncated SVD of Text Documents')
plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.2f}")
```

Slide 11: t-SNE for Non-Linear Dimensionality Reduction and Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a powerful technique for visualizing high-dimensional data in 2D or 3D space while preserving local structure.

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Digits Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.show()
```

Slide 12: UMAP for Fast Non-Linear Dimensionality Reduction

UMAP (Uniform Manifold Approximation and Projection) is a more recent algorithm that offers faster computation and better preservation of global structure compared to t-SNE.

```python
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('UMAP Visualization of Digits Dataset')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.tight_layout()
plt.show()
```

Slide 13: Comparison of Dimensionality Reduction Techniques

Different dimensionality reduction techniques have varying strengths and weaknesses. This slide compares PCA, t-SNE, and UMAP on the same dataset to highlight their differences.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply different dimensionality reduction techniques
pca = PCA(n_components=2, random_state=42)
tsne = TSNE(n_components=2, random_state=42)
umap_reducer = umap.UMAP(random_state=42)

X_pca = pca.fit_transform(X)
X_tsne = tsne.fit_transform(X)
X_umap = umap_reducer.fit_transform(X)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
techniques = [('PCA', X_pca), ('t-SNE', X_tsne), ('UMAP', X_umap)]

for ax, (name, data) in zip(axes, techniques):
    scatter = ax.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis')
    ax.set_title(f'{name} Visualization')
    ax.set_xlabel(f'{name} Component 1')
    ax.set_ylabel(f'{name} Component 2')

plt.colorbar(scatter, ax=axes[-1])
plt.tight_layout()
plt.show()
```

Slide 14: Choosing the Right Dimensionality Reduction Technique

Selecting the appropriate dimensionality reduction method depends on various factors such as dataset size, sparsity, desired output dimensions, and the need for interpretability or visualization. Here's a guide to help you choose:

1. PCA: Use when linear relationships are sufficient, and you need interpretable components or fast computation.
2. Kernel PCA: Apply for non-linear relationships when you can afford higher computational cost.
3. Incremental PCA: Choose for large datasets that don't fit in memory.
4. Sparse PCA: Opt for feature selection and interpretability in high-dimensional spaces.
5. Truncated SVD (LSA): Prefer for sparse data, especially in text processing.
6. t-SNE: Use for visualization of high-dimensional data in 2D or 3D, preserving local structure.
7. UMAP: Select for faster computation and better preservation of global structure compared to t-SNE.

Slide 15: Choosing the Right Dimensionality Reduction Technique

Selecting the appropriate dimensionality reduction method depends on various factors such as dataset size, sparsity, desired output dimensions, and the need for interpretability or visualization. This flowchart helps guide the decision-making process:

```python
import networkx as nx
import matplotlib.pyplot as plt

def create_decision_flowchart():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Start", "Linear?"),
        ("Linear?", "PCA"),
        ("Linear?", "Non-linear"),
        ("PCA", "Large dataset?"),
        ("Large dataset?", "Incremental PCA"),
        ("Large dataset?", "Standard PCA"),
        ("Non-linear", "Visualization?"),
        ("Visualization?", "t-SNE/UMAP"),
        ("Visualization?", "Kernel PCA"),
        ("Kernel PCA", "Sparse data?"),
        ("Sparse data?", "Truncated SVD"),
        ("Sparse data?", "Standard Kernel PCA")
    ])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=8, arrows=True)
    
    edge_labels = {("Start", "Linear?"): "Start",
                   ("Linear?", "PCA"): "Yes",
                   ("Linear?", "Non-linear"): "No",
                   ("PCA", "Large dataset?"): "",
                   ("Large dataset?", "Incremental PCA"): "Yes",
                   ("Large dataset?", "Standard PCA"): "No",
                   ("Non-linear", "Visualization?"): "",
                   ("Visualization?", "t-SNE/UMAP"): "Yes",
                   ("Visualization?", "Kernel PCA"): "No",
                   ("Kernel PCA", "Sparse data?"): "",
                   ("Sparse data?", "Truncated SVD"): "Yes",
                   ("Sparse data?", "Standard Kernel PCA"): "No"}
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.axis('off')
    plt.title("Dimensionality Reduction Technique Selection Flowchart")
    plt.tight_layout()
    plt.show()

create_decision_flowchart()
```

This flowchart provides a visual guide for choosing the most suitable dimensionality reduction technique based on your data characteristics and requirements.

Slide 16: Evaluation Metrics for Dimensionality Reduction

To assess the quality of dimensionality reduction, we can use various metrics. Here's a code example demonstrating two common evaluation metrics: reconstruction error and trustworthiness.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import numpy as np

# Load the digits dataset
digits = load_digits()
X = digits.data

# Apply PCA
n_components = 20
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

# Calculate reconstruction error
reconstruction_error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=1))
print(f"Reconstruction Error: {reconstruction_error:.4f}")

# Calculate trustworthiness
trust_score = trustworthiness(X, X_reduced)
print(f"Trustworthiness Score: {trust_score:.4f}")

# Plot cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.show()
```

These metrics help quantify how well the reduced-dimensional representation preserves the structure and information of the original data.

Slide 17: Additional Resources

For further exploration of dimensionality reduction techniques and their applications, consider the following resources:

1. "A Survey of Dimensionality Reduction Techniques" by L. van der Maaten et al. (2009) ArXiv: [https://arxiv.org/abs/0904.3664](https://arxiv.org/abs/0904.3664)
2. "Visualizing Data using t-SNE" by L. van der Maaten and G. Hinton (2008) Journal of Machine Learning Research
3. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" by L. McInnes et al. (2018) ArXiv: [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
4. "A Tutorial on Principal Component Analysis" by J. Shlens (2014) ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)

These resources provide in-depth discussions on various dimensionality reduction methods, their theoretical foundations, and practical applications.
