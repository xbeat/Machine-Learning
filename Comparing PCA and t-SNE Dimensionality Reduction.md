## Comparing PCA and t-SNE Dimensionality Reduction

Slide 1: Key Differences Between PCA and t-SNE

Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) are two popular dimensionality reduction techniques used in data science and machine learning. While both aim to reduce the dimensionality of high-dimensional data, they differ significantly in their approach and applications. This presentation will explore the key differences between PCA and t-SNE, providing insights into when to use each method.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate sample data
np.random.seed(42)
data = np.random.randn(1000, 50)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
ax1.set_title('PCA')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
ax2.set_title('t-SNE')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 2: Linear vs. Non-linear

PCA is a linear dimensionality reduction technique that assumes relationships between variables are linear. It works by finding the directions of maximum variance in the data and projecting the data onto these directions. In contrast, t-SNE is a non-linear technique that can capture complex, non-linear relationships in the data. This makes t-SNE better suited for revealing intricate data structures that might be missed by linear methods like PCA.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate non-linear data (Swiss roll dataset)
n_points = 1000
X = np.zeros((n_points, 3))
t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_points))
X[:, 0] = t * np.cos(t)
X[:, 1] = 21 * np.random.rand(n_points)
X[:, 2] = t * np.sin(t)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.scatter(X[:, 0], X[:, 2], c=t, cmap='viridis')
ax1.set_title('Original Swiss Roll')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
ax2.set_title('PCA')

ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
ax3.set_title('t-SNE')

plt.tight_layout()
plt.show()
```

Slide 3: Global vs. Local Structure

PCA focuses on preserving the global structure of the data by maximizing the variance along each principal component. This approach is effective for capturing overall trends and patterns in the dataset. On the other hand, t-SNE prioritizes the preservation of local relationships, keeping similar data points close together in the reduced space. This local focus allows t-SNE to reveal clusters and local patterns that might be obscured in a global analysis.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate clustered data
X, y = make_blobs(n_samples=1000, n_features=50, centers=5, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
ax1.set_title('PCA (Global Structure)')
ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
ax2.set_title('t-SNE (Local Structure)')
ax2.set_xlabel('First t-SNE Component')
ax2.set_ylabel('Second t-SNE Component')

plt.tight_layout()
plt.show()
```

Slide 4: Deterministic vs. Stochastic

PCA is a deterministic algorithm, meaning it always produces the same result for a given dataset. This property makes PCA results reproducible and consistent across multiple runs. In contrast, t-SNE is a stochastic algorithm that involves randomness in its optimization process. As a result, t-SNE can produce slightly different results each time it's run on the same data, even with the same random seed.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate sample data
np.random.seed(42)
X = np.random.randn(500, 50)

# Apply PCA multiple times
pca_results = []
for _ in range(3):
    pca = PCA(n_components=2)
    pca_results.append(pca.fit_transform(X))

# Apply t-SNE multiple times
tsne_results = []
for _ in range(3):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results.append(tsne.fit_transform(X))

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, (pca_result, tsne_result) in enumerate(zip(pca_results, tsne_results)):
    axes[0, i].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    axes[0, i].set_title(f'PCA Run {i+1}')
    
    axes[1, i].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
    axes[1, i].set_title(f't-SNE Run {i+1}')

plt.tight_layout()
plt.show()
```

Slide 5: Interpretability

PCA offers straightforward, interpretable results. Each principal component is a linear combination of the original features, allowing us to understand which features contribute most to the variance in the data. This interpretability makes PCA useful for feature selection and understanding the underlying structure of the data. t-SNE, while excellent for visualization, is harder to interpret. The resulting components don't have a clear relationship to the original features, making t-SNE primarily useful for data exploration and visualization rather than feature interpretation.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA results
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.title('PCA of Iris Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Plot feature contributions
plt.subplot(122)
components = pca.components_.T
plt.bar(feature_names, components[:, 0], alpha=0.5, label='PC1')
plt.bar(feature_names, components[:, 1], alpha=0.5, label='PC2')
plt.title('Feature Contributions to Principal Components')
plt.xlabel('Features')
plt.ylabel('Contribution')
plt.legend()
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 6: Computational Cost

PCA is computationally efficient and scales well to large datasets. Its time complexity is O(min(n^2d, nd^2)), where n is the number of samples and d is the number of features. This efficiency makes PCA suitable for high-dimensional data reduction tasks. In contrast, t-SNE is computationally expensive, especially for larger datasets. Its time complexity is O(n^2), which can become prohibitive for very large datasets. As a result, t-SNE is often applied to smaller datasets or used as a final visualization step after initial dimensionality reduction with other methods.

```python
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compare_computation_time(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    
    # Measure PCA computation time
    start_time = time.time()
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    pca_time = time.time() - start_time
    
    # Measure t-SNE computation time
    start_time = time.time()
    tsne = TSNE(n_components=2)
    tsne.fit_transform(X)
    tsne_time = time.time() - start_time
    
    return pca_time, tsne_time

# Compare computation times for different dataset sizes
sizes = [100, 500, 1000, 2000]
pca_times = []
tsne_times = []

for size in sizes:
    pca_time, tsne_time = compare_computation_time(size, 50)
    pca_times.append(pca_time)
    tsne_times.append(tsne_time)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sizes, pca_times, marker='o', label='PCA')
plt.plot(sizes, tsne_times, marker='o', label='t-SNE')
plt.xlabel('Number of Samples')
plt.ylabel('Computation Time (seconds)')
plt.title('PCA vs t-SNE Computation Time')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 7: When to Use PCA

PCA is particularly useful when you need a simple, interpretable reduction method for high-dimensional data. It's ideal for preparing data for machine learning models that require linear transformations or when you want to reduce noise by eliminating less important features. PCA is also valuable for exploratory data analysis, helping to identify the most significant variables in your dataset.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('PCA of Digits Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Add some digit images to the plot
for i in range(10):
    idx = np.where(y == i)[0][0]
    plt.annotate(str(i), X_pca[idx], xytext=(5, 5), textcoords='offset points')
    plt.imshow(digits.images[idx], cmap='binary', extent=(X_pca[idx, 0]-10, X_pca[idx, 0]+10, 
                                                          X_pca[idx, 1]-10, X_pca[idx, 1]+10))

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 8: When to Use t-SNE

t-SNE is particularly effective when you're looking to visualize high-dimensional data in 2D or 3D while maintaining local data clusters. It's ideal for exploring datasets with complex, non-linear relationships that PCA might miss. t-SNE is also useful for clustering and exploring the intrinsic structure of the data, especially when local relationships are more important than global structure.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE of Digits Dataset')
plt.xlabel('First t-SNE Component')
plt.ylabel('Second t-SNE Component')

# Add some digit images to the plot
for i in range(10):
    idx = np.where(y == i)[0][0]
    plt.annotate(str(i), X_tsne[idx], xytext=(5, 5), textcoords='offset points')
    plt.imshow(digits.images[idx], cmap='binary', extent=(X_tsne[idx, 0]-10, X_tsne[idx, 0]+10, 
                                                          X_tsne[idx, 1]-10, X_tsne[idx, 1]+10))

plt.tight_layout()
plt.show()
```

Slide 9: Real-Life Example: Image Processing

In image processing, PCA can be used for tasks like image compression and feature extraction. For instance, we can use PCA to reduce the dimensionality of image data while preserving the most important visual information. This technique is particularly useful in facial recognition systems, where PCA is often referred to as the "eigenface" method.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
y = faces.target

# Apply PCA
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X)

# Reconstruct faces using different numbers of components
n_reconstructions = [10, 50, 100, 150]
fig, axs = plt.subplots(5, len(n_reconstructions), figsize=(15, 12))

for i, n in enumerate(n_reconstructions):
    reconst_img = pca.inverse_transform(pca.transform(X[0].reshape(1, -1))[:, :n])
    axs[0, i].imshow(reconst_img.reshape(faces.images[0].shape), cmap='gray')
    axs[0, i].set_title(f'{n} components')
    axs[0, i].axis('off')

# Show original image
axs[0, 0].imshow(X[0].reshape(faces.images[0].shape), cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

plt.tight_layout()
plt.show()

print("Explained variance ratio sum:", sum(pca.explained_variance_ratio_))
```

Slide 10: Real-Life Example: Genomic Data Analysis

In genomics, t-SNE is often used to visualize high-dimensional gene expression data. It can reveal clusters of genes with similar expression patterns or groups of samples with similar genetic profiles. This application is crucial in understanding complex biological systems and identifying potential biomarkers for diseases.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# Simulate gene expression data
n_samples = 1000
n_features = 100
n_clusters = 5

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of Simulated Gene Expression Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

Slide 11: Combining PCA and t-SNE

In practice, it's often beneficial to combine PCA and t-SNE, especially when dealing with very high-dimensional data. PCA can be used as a preprocessing step to reduce the dimensionality of the data before applying t-SNE. This approach can significantly speed up the t-SNE computation while still preserving important structures in the data.

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

# Apply PCA as a preprocessing step
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Apply t-SNE on PCA-reduced data
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.astype(int), cmap='tab10', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE Visualization of MNIST (PCA preprocessed)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

Slide 12: Hyperparameter Tuning in t-SNE

While PCA has few hyperparameters, t-SNE has several that can significantly affect its output. The most important ones are perplexity and the number of iterations. Perplexity balances local and global aspects of the data, while the number of iterations affects how well the algorithm optimizes the embedding. It's crucial to experiment with these parameters to find the best visualization for your data.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Define different perplexity values
perplexities = [5, 30, 50, 100]

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    
    axs[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.5)
    axs[i].set_title(f'Perplexity: {perplexity}')
    axs[i].set_xlabel('t-SNE 1')
    axs[i].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

Slide 13: Limitations and Considerations

While PCA and t-SNE are powerful tools, they have limitations. PCA assumes linear relationships and may miss important non-linear structures. t-SNE can sometimes create misleading visualizations, especially when the perplexity is not well-tuned. Both methods can struggle with very high-dimensional data. It's important to understand these limitations and use these techniques as part of a broader analytical approach, rather than relying on them exclusively.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Generate S-curve dataset
X, color = make_s_curve(n_samples=1000, random_state=42)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X[:, 0], X[:, 2], c=color, cmap='viridis')
ax1.set_title('Original S-curve')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis')
ax2.set_title('PCA')

ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='viridis')
ax3.set_title('t-SNE')

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into PCA and t-SNE, here are some valuable resources:

1. "Dimensionality Reduction: A Comparative Review" by L.J.P. van der Maaten, E.O. Postma, and H.J. van den Herik (ArXiv:0904.3383)
2. "Visualizing Data using t-SNE" by Laurens van der Maaten and Geoffrey Hinton (Journal of Machine Learning Research, 2008)
3. "A Tutorial on Principal Component Analysis" by Jonathon Shlens (ArXiv:1404.1100)
4. "How to Use t-SNE Effectively" by Martin Wattenberg, Fernanda Viégas, and Ian Johnson (Distill, 2016)

These papers provide in-depth explanations of the algorithms, their mathematical foundations, and best practices for their application in various domains.


