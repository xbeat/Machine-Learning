## Principal Component Analysis  in Python
Slide 1: Introduction to Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible. It identifies the directions (principal components) along which the data varies the most.

```python
import numpy as np
from matplotlib import pyplot as plt

# Generate sample 2D data
np.random.seed(42)
data = np.random.randn(100, 2)
data = data @ [[2, 1], [1, 3]]  # Create correlation

plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.axis('equal')
plt.title('Sample 2D Data with Correlation')
```

Slide 2: Mathematical Foundation

PCA is based on finding eigenvectors and eigenvalues of the covariance matrix. The eigenvectors determine the directions of maximum variance, while eigenvalues represent the amount of variance explained by each direction. The mathematical formula for PCA is:

X' = X - μ C = (1/n) X'ᵀX' Cv = λv

Where X is the data matrix, μ is the mean, C is the covariance matrix, λ are eigenvalues, and v are eigenvectors.

Slide 3: Code for Mathematical Foundation

```python
def calculate_pca_components(data):
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Calculate covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors, mean
```

Slide 4: Data Preprocessing

Before applying PCA, data must be preprocessed by centering (subtracting the mean) and optionally scaling. Scaling is crucial when features have different units or variances.

Slide 5: Code for Data Preprocessing

```python
def preprocess_data(X):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Scale the data
    X_scaled = X_centered / np.std(X_centered, axis=0)
    
    return X_scaled
```

Slide 6: Explained Variance

The proportion of variance explained by each principal component helps determine how many components to retain. This is calculated by dividing each eigenvalue by the sum of all eigenvalues.

Slide 7: Code for Explained Variance

```python
def calculate_explained_variance(eigenvalues):
    # Calculate proportion of variance explained
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    return explained_variance_ratio, cumulative_variance
```

Slide 8: Real-life Example - Image Compression

PCA can be used for image compression by reducing the dimensionality of image data while preserving important features.

Slide 9: Code for Image Compression Example

```python
def compress_image(image, n_components):
    # Reshape image to 2D array
    h, w = image.shape
    X = image.reshape(h, w)
    
    # Apply PCA
    eigenvalues, eigenvectors, mean = calculate_pca_components(X)
    
    # Project data onto principal components
    projected = (X - mean) @ eigenvectors[:, :n_components]
    
    # Reconstruct image
    reconstructed = projected @ eigenvectors[:, :n_components].T + mean
    
    return reconstructed.reshape(h, w)
```

Slide 10: Real-life Example - Color Analysis

PCA can analyze color distributions in images, useful for computer vision applications like object detection and scene understanding.

Slide 11: Code for Color Analysis Example

```python
def analyze_colors(image):
    # Reshape image to 2D array (pixels x RGB)
    pixels = image.reshape(-1, 3)
    
    # Apply PCA
    eigenvalues, eigenvectors, mean = calculate_pca_components(pixels)
    
    # Project colors onto principal components
    projected_colors = (pixels - mean) @ eigenvectors
    
    return projected_colors, eigenvectors, mean
```

Slide 12: Implementation from Scratch

Here's a complete implementation of PCA without using specialized libraries, useful for understanding the underlying mechanics.

Slide 13: Code for Implementation from Scratch

```python
def pca_from_scratch(X):
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Calculate covariance matrix
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvects = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvects = eigenvects[:, idx]
    
    return eigenvals, eigenvects, X_mean
```

Slide 14: Additional Resources

For a deeper understanding of PCA, refer to these peer-reviewed papers:

1.  "A Tutorial on Principal Component Analysis" - Jonathon Shlens ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2.  "Principal Component Analysis: A Review and Recent Developments" - Ian Jolliffe & Jorge Cadima ArXiv: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)

