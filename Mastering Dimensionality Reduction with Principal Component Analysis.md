## Mastering Dimensionality Reduction with Principal Component Analysis
Slide 1: Introduction to Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a new coordinate system where features are uncorrelated. The first principal component captures the direction of maximum variance, with subsequent components orthogonal to previous ones.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)  # 100 samples, 4 features

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate covariance matrix
cov_matrix = np.cov(X_scaled.T)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues)
print("Explained variance ratio:", eigenvalues / np.sum(eigenvalues))
```

Slide 2: Mathematical Foundation of PCA

The mathematical foundation of PCA involves finding the eigenvectors and eigenvalues of the covariance matrix. These eigenvectors represent the principal components, while eigenvalues indicate the amount of variance explained by each component.

```python
# Mathematical formulation in LaTeX notation
$$
\text{Covariance Matrix} = \Sigma = \frac{1}{n-1}X^TX
$$

$$
\text{Eigendecomposition}: \Sigma v = \lambda v
$$

$$
\text{Transformed Data} = X W
$$

# where X is the centered data matrix
# W is the matrix of eigenvectors
# λ represents eigenvalues
```

Slide 3: Implementing PCA from Scratch

```python
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Select top n_components
    components = eigenvecs[:, :n_components]
    
    # Project data
    X_transformed = X_centered @ components
    
    return X_transformed, components, eigenvals

# Example usage
X_transformed, components, eigenvals = pca_from_scratch(X_scaled, 2)
print("Transformed shape:", X_transformed.shape)
```

Slide 4: Variance Explained and Component Selection

Understanding the variance explained by each principal component is crucial for determining the optimal number of components to retain. This analysis helps balance dimensionality reduction with information preservation.

```python
def plot_explained_variance(eigenvalues):
    import matplotlib.pyplot as plt
    
    # Calculate cumulative explained variance ratio
    total_var = np.sum(eigenvalues)
    cum_var_ratio = np.cumsum(eigenvalues) / total_var
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), cum_var_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    
    return cum_var_ratio

# Example usage
explained_variance_ratio = plot_explained_variance(eigenvalues)
print("Explained variance ratios:", explained_variance_ratio)
```

Slide 5: PCA with Scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer

# Load real-world dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Initialize and fit PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Transformed shape:", X_pca.shape)
print("Components retained:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 6: Real-World Example - Image Compression

Principal Component Analysis can be effectively used for image compression by reducing the dimensionality of image data while preserving essential visual information. This example demonstrates compression of a grayscale image.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

def compress_image(image_array, n_components):
    # Standardize pixel values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(image_array)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Reconstruct image
    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    
    return X_reconstructed, pca.explained_variance_ratio_

# Example usage
img = np.random.rand(100, 100)  # Example grayscale image
compressed_img, var_ratio = compress_image(img, n_components=20)

print(f"Original size: {img.size}")
print(f"Compressed size: {compressed_img.size}")
print(f"Compression ratio: {img.size/compressed_img.size:.2f}")
```

Slide 7: Feature Selection Using PCA

PCA can identify which original features contribute most significantly to the principal components, enabling informed feature selection decisions in high-dimensional datasets.

```python
def analyze_feature_importance(pca_model, feature_names):
    # Get absolute value of component loadings
    loadings = np.abs(pca_model.components_)
    
    # Calculate feature importance scores
    importance = np.sum(loadings, axis=0)
    importance = importance / np.sum(importance)
    
    # Create feature importance dictionary
    feature_importance = dict(zip(feature_names, importance))
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    return sorted_features

# Example with breast cancer dataset
pca = PCA()
pca.fit(X)
important_features = analyze_feature_importance(pca, data.feature_names)
print("Top 5 most important features:")
for feature, importance in important_features[:5]:
    print(f"{feature}: {importance:.3f}")
```

Slide 8: Incremental PCA for Large Datasets

When dealing with datasets too large to fit in memory, Incremental PCA allows processing data in batches while maintaining mathematical equivalence to standard PCA.

```python
from sklearn.decomposition import IncrementalPCA

def incremental_pca_processing(data_generator, n_components, batch_size):
    # Initialize incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Process data in batches
    for batch in data_generator:
        ipca.partial_fit(batch)
    
    return ipca

# Example with simulated data stream
def generate_batches(n_batches, batch_size, n_features):
    for _ in range(n_batches):
        yield np.random.randn(batch_size, n_features)

# Process batches
ipca = incremental_pca_processing(
    generate_batches(10, 1000, 50),
    n_components=10,
    batch_size=1000
)

print("Number of components:", ipca.n_components_)
print("Explained variance ratio:", ipca.explained_variance_ratio_)
```

Slide 9: PCA for Anomaly Detection

PCA can be used for anomaly detection by identifying data points that have high reconstruction error when projected onto and back from the principal component space.

```python
def detect_anomalies(X, n_components, threshold_percentile=95):
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Reconstruct data
    X_reconstructed = pca.inverse_transform(X_pca)
    
    # Calculate reconstruction error
    reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
    
    # Set threshold
    threshold = np.percentile(reconstruction_error, threshold_percentile)
    
    # Identify anomalies
    anomalies = reconstruction_error > threshold
    
    return anomalies, reconstruction_error

# Example usage
X = np.random.randn(1000, 10)
X[0] = X[0] * 10  # Create an obvious anomaly

anomalies, errors = detect_anomalies(X, n_components=5)
print("Number of anomalies detected:", np.sum(anomalies))
print("Reconstruction error for first sample:", errors[0])
```

Slide 10: PCA for Time Series Analysis

PCA can extract meaningful patterns from multivariate time series data by identifying the principal temporal components that explain the most variance across different time series channels.

```python
def analyze_time_series_pca(time_series_data, n_components):
    # Standardize the time series
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(time_series_data)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    # Calculate component contributions
    reconstructed = pca.inverse_transform(components)
    reconstruction_error = np.mean((X_scaled - reconstructed) ** 2)
    
    return components, pca.explained_variance_ratio_, reconstruction_error

# Generate example multivariate time series
np.random.seed(42)
t = np.linspace(0, 10, 1000)
signals = np.column_stack([
    np.sin(t),
    np.sin(2*t),
    np.sin(3*t),
    np.random.normal(0, 0.1, len(t))
])

components, var_ratio, error = analyze_time_series_pca(signals, 2)
print(f"Explained variance ratios: {var_ratio}")
print(f"Reconstruction error: {error}")
```

Slide 11: Kernel PCA Implementation

Kernel PCA extends traditional PCA to handle nonlinear relationships in data by projecting the data into a higher-dimensional feature space using the kernel trick.

```python
from sklearn.preprocessing import KernelCenterer
from scipy.linalg import eigh

def kernel_pca(X, n_components, kernel='rbf', gamma=1.0):
    def rbf_kernel(X, Y=None):
        if Y is None:
            Y = X
        return np.exp(-gamma * np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
    
    # Compute kernel matrix
    K = rbf_kernel(X)
    
    # Center kernel matrix
    centerer = KernelCenterer()
    K_centered = centerer.fit_transform(K)
    
    # Eigendecomposition
    eigenvals, eigenvecs = eigh(K_centered)
    
    # Sort eigenvectors in descending order
    indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[indices]
    eigenvecs = eigenvecs[:, indices]
    
    # Select top components
    return eigenvecs[:, :n_components] * np.sqrt(eigenvals[:n_components])

# Example usage with nonlinear data
X = np.vstack([
    np.random.randn(100, 2) * 0.5,
    np.random.randn(100, 2) * 2.0 + 2
])

X_kpca = kernel_pca(X, n_components=2, gamma=2.0)
print("Transformed shape:", X_kpca.shape)
```

Slide 12: PCA for Noise Reduction

PCA can effectively remove noise from data by reconstructing signals using only the most significant principal components, filtering out components that likely represent noise.

```python
def denoise_with_pca(X, n_components):
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_denoised = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_denoised)
    
    # Calculate noise reduction metrics
    noise_reduction = np.mean((X - X_reconstructed) ** 2)
    signal_retention = np.sum(pca.explained_variance_ratio_)
    
    return X_reconstructed, noise_reduction, signal_retention

# Generate noisy data
clean_signal = np.sin(np.linspace(0, 10, 1000))
noise = np.random.normal(0, 0.2, 1000)
noisy_signal = clean_signal + noise

# Reshape for PCA
X = noisy_signal.reshape(-1, 10)
X_denoised, noise_red, signal_ret = denoise_with_pca(X, n_components=3)

print(f"Noise reduction: {noise_red:.4f}")
print(f"Signal retention: {signal_ret:.4f}")
```

Slide 13: Additional Resources

*   "Principal Component Analysis in Linear Algebra and Implications for Data Analysis"
    *   [https://arxiv.org/abs/2108.03247](https://arxiv.org/abs/2108.03247)
*   "A Tutorial on Principal Component Analysis with Applications in R"
    *   [https://arxiv.org/abs/2009.10835](https://arxiv.org/abs/2009.10835)
*   "Robust Principal Component Analysis: A Survey and Recent Developments"
    *   [https://arxiv.org/abs/1705.10403](https://arxiv.org/abs/1705.10403)
*   "Incremental Principal Component Analysis: A Comprehensive Survey"
    *   For detailed implementation strategies, search "Incremental PCA implementations" on Google Scholar
*   "Kernel Principal Component Analysis and its Applications in Face Recognition"
    *   Available through IEEE Digital Library or Google Scholar search

