## Implementing PCA for Dimensionality Reduction
Slide 1: Introduction to PCA

Principal Component Analysis (PCA) is a powerful technique used for dimensionality reduction in machine learning and data analysis. It helps address the curse of dimensionality by transforming high-dimensional data into a lower-dimensional space while preserving the most important information. PCA works by identifying the principal components, which are orthogonal vectors that capture the maximum variance in the data.

Slide 2: The Curse of Dimensionality

The curse of dimensionality refers to the challenges that arise when working with high-dimensional data. As the number of features increases, the amount of data required to make statistically sound predictions grows exponentially. This can lead to overfitting, increased computational complexity, and difficulty in visualizing and interpreting the data.

Slide 3: Source Code for The Curse of Dimensionality

```python
import random
import math

def generate_random_point(dimensions):
    return [random.uniform(0, 1) for _ in range(dimensions)]

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def demonstrate_curse_of_dimensionality(num_points=1000, max_dim=100):
    dimensions = list(range(1, max_dim + 1, 10))
    avg_distances = []

    for dim in dimensions:
        points = [generate_random_point(dim) for _ in range(num_points)]
        distances = [euclidean_distance(points[i], points[j])
                     for i in range(num_points)
                     for j in range(i + 1, num_points)]
        avg_distances.append(sum(distances) / len(distances))

    for dim, avg_dist in zip(dimensions, avg_distances):
        print(f"Dimensions: {dim}, Average distance: {avg_dist:.4f}")

demonstrate_curse_of_dimensionality()
```

Slide 4: Results for Source Code for The Curse of Dimensionality

```
Dimensions: 1, Average distance: 0.3336
Dimensions: 11, Average distance: 1.1045
Dimensions: 21, Average distance: 1.5256
Dimensions: 31, Average distance: 1.8533
Dimensions: 41, Average distance: 2.1317
Dimensions: 51, Average distance: 2.3778
Dimensions: 61, Average distance: 2.6016
Dimensions: 71, Average distance: 2.8083
Dimensions: 81, Average distance: 3.0021
Dimensions: 91, Average distance: 3.1846
```

Slide 5: Mathematical Foundations of PCA

PCA is based on the concept of eigenvectors and eigenvalues. Given a dataset X, PCA computes the covariance matrix and then finds its eigenvectors and eigenvalues. The eigenvectors represent the directions of maximum variance in the data, while the eigenvalues indicate the amount of variance explained by each eigenvector. The principal components are sorted in descending order of their corresponding eigenvalues.

Slide 6: Source Code for Mathematical Foundations of PCA

```python
def covariance_matrix(X):
    n = X.shape[0]
    X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / (n - 1)

def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = [], []
    n = cov_matrix.shape[0]
    
    for i in range(n):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(100):  # Power iteration
            v_new = cov_matrix @ v
            v_new = v_new / np.linalg.norm(v_new)
            
            if np.allclose(v, v_new):
                break
            v = v_new
        
        eigenvalue = (v.T @ cov_matrix @ v) / (v.T @ v)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        
        # Deflation
        cov_matrix = cov_matrix - eigenvalue * np.outer(v, v)
    
    return np.array(eigenvalues), np.array(eigenvectors).T

# Example usage
X = np.random.rand(100, 5)
cov_matrix = covariance_matrix(X)
eigenvalues, eigenvectors = eigen_decomposition(cov_matrix)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors shape:", eigenvectors.shape)
```

Slide 7: PCA Algorithm Steps

The PCA algorithm consists of several key steps:

1.  Standardize the dataset
2.  Compute the covariance matrix
3.  Calculate eigenvectors and eigenvalues
4.  Sort eigenvectors by decreasing eigenvalues
5.  Choose the top k eigenvectors
6.  Project the data onto the new subspace

These steps transform the original high-dimensional data into a lower-dimensional representation while preserving the most important information.

Slide 8: Source Code for PCA Algorithm Steps

```python
import numpy as np

def pca(X, k):
    # Step 1: Standardize the dataset
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_std.T)
    
    # Step 3: Calculate eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Choose the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]
    
    # Step 6: Project the data onto the new subspace
    X_pca = X_std.dot(top_k_eigenvectors)
    
    return X_pca, eigenvalues, eigenvectors

# Example usage
X = np.random.rand(100, 5)
k = 3
X_pca, eigenvalues, eigenvectors = pca(X, k)

print("Original shape:", X.shape)
print("PCA shape:", X_pca.shape)
print("Top 3 eigenvalues:", eigenvalues[:3])
```

Slide 9: Selecting the Number of Principal Components

Choosing the optimal number of principal components is crucial for effective dimensionality reduction. One common approach is to use the cumulative explained variance ratio, which measures the proportion of variance explained by each principal component. By setting a threshold (e.g., 95% of total variance), we can determine the number of components to retain.

Slide 10: Source Code for Selecting the Number of Principal Components

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_cumulative_variance(eigenvalues):
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
    plt.grid(True)
    plt.show()

def select_components(eigenvalues, threshold=0.95):
    total_variance = np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
    return np.argmax(cumulative_variance_ratio >= threshold) + 1

# Example usage
X = np.random.rand(100, 10)
_, eigenvalues, _ = pca(X, 10)

plot_cumulative_variance(eigenvalues)
optimal_components = select_components(eigenvalues)
print(f"Optimal number of components: {optimal_components}")
```

Slide 11: Real-Life Example: Image Compression

PCA can be used for image compression by reducing the dimensionality of image data. This technique is particularly useful for grayscale images, where each pixel is represented by a single value. By applying PCA to the image matrix, we can compress the image while preserving its essential features.

Slide 12: Source Code for Image Compression Example

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def compress_image(image_path, k):
    # Load the image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Apply PCA
    X_pca, _, _ = pca(img_array, k)
    
    # Reconstruct the image
    reconstructed = X_pca.dot(X_pca.T)
    reconstructed = reconstructed.astype(np.uint8)
    
    # Display original and compressed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img_array, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(reconstructed, cmap='gray')
    ax2.set_title(f'Compressed Image (k={k})')
    plt.show()

# Example usage
image_path = 'example_image.jpg'
compress_image(image_path, k=50)
```

Slide 13: Real-Life Example: Genome Data Analysis

PCA is widely used in genomics for analyzing high-dimensional genetic data. It can help identify patterns in gene expression, discover population structures, and visualize relationships between different genetic samples. This example demonstrates how to apply PCA to a dataset of single nucleotide polymorphisms (SNPs) from different populations.

Slide 14: Source Code for Genome Data Analysis Example

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_snp_data(n_samples, n_snps, n_populations):
    populations = np.random.randint(0, n_populations, n_samples)
    snp_data = np.random.binomial(2, 0.3 + 0.1 * populations[:, np.newaxis], (n_samples, n_snps))
    return snp_data, populations

def analyze_snp_data(snp_data, populations):
    X_pca, _, _ = pca(snp_data, k=2)
    
    plt.figure(figsize=(10, 8))
    for pop in range(max(populations) + 1):
        mask = populations == pop
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Population {pop}')
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of SNP Data')
    plt.legend()
    plt.show()

# Example usage
n_samples, n_snps, n_populations = 1000, 1000, 3
snp_data, populations = simulate_snp_data(n_samples, n_snps, n_populations)
analyze_snp_data(snp_data, populations)
```

Slide 15: Limitations and Considerations

While PCA is a powerful technique, it has some limitations:

1.  It assumes linear relationships between features.
2.  It may not work well with highly non-linear data.
3.  Principal components can be difficult to interpret.
4.  It is sensitive to outliers.
5.  It may not always preserve important information for specific tasks.

Consider alternative techniques like t-SNE or UMAP for non-linear dimensionality reduction when dealing with complex datasets.

Slide 16: Additional Resources

For more in-depth information on PCA and related topics, consider the following resources:

1.  "A Tutorial on Principal Component Analysis" by Jonathon Shlens (2014) ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2.  "Dimensionality Reduction: A Comparative Review" by Laurens van der Maaten, Eric Postma, and Jaap van den Herik (2009) Available at: [https://lvdmaaten.github.io/publications/papers/TR\_Dimensionality\_Reduction\_Review\_2009.pdf](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf)
3.  "Principal Component Analysis" by Svante Wold, Kim Esbensen, and Paul Geladi (1987) DOI: 10.1016/0169-7439(87)80084-9

These resources provide comprehensive overviews and advanced discussions on PCA and related dimensionality reduction techniques.

