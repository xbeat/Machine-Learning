## Dimensionality Reduction with Principal Component Analysis
Slide 1: Introduction to Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they explain in the data. This mathematical technique identifies patterns by finding directions of maximum variance.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot variance explained
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()
```

Slide 2: Mathematical Foundation of PCA

PCA involves calculating the covariance matrix of standardized features and finding its eigenvalues and eigenvectors. The eigenvectors represent the directions of maximum variance, while eigenvalues indicate the amount of variance explained by each component.

```python
# Mathematical implementation of PCA from scratch
def pca_from_scratch(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    return X_centered @ eigenvectors[:, :n_components]
```

Slide 3: Variance Explained and Component Selection

Understanding how much variance each principal component explains is crucial for determining the optimal number of components to retain. The explained variance ratio helps identify the trade-off between dimensionality reduction and information preservation.

```python
def analyze_variance_explained(X, threshold=0.95):
    # Apply PCA
    pca = PCA()
    pca.fit(X)
    
    # Calculate cumulative variance explained
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components needed for threshold
    n_components = np.argmax(cumsum >= threshold) + 1
    
    print(f"Components needed for {threshold*100}% variance: {n_components}")
    return pca.explained_variance_ratio_, n_components
```

Slide 4: Real-world Example - Image Compression

PCA can be effectively used for image compression by reducing the dimensionality of image data while preserving essential features. This example demonstrates how to compress and reconstruct a grayscale image using PCA.

```python
from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with different numbers of components
n_components_list = [10, 20, 30, 40]
reconstructed_images = []

for n in n_components_list:
    pca = PCA(n_components=n)
    X_transformed = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstructed_images.append(X_reconstructed)

# Plot original vs reconstructed images
fig, axes = plt.subplots(5, 4, figsize=(12, 15))
for i, n_comp in enumerate(n_components_list):
    axes[0, i].imshow(X[0].reshape(8, 8), cmap='gray')
    axes[0, i].set_title(f'Original')
    
    axes[1, i].imshow(reconstructed_images[i][0].reshape(8, 8), cmap='gray')
    axes[1, i].set_title(f'{n_comp} components')
plt.show()
```

Slide 5: Feature Correlation Analysis

Before applying PCA, it's essential to understand the correlation structure of your features. High correlations between features often indicate redundancy that PCA can effectively address.

```python
def analyze_feature_correlations(X, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix
```

Slide 6: PCA for Anomaly Detection

PCA can be used effectively for anomaly detection by identifying data points that have high reconstruction error when projected onto and back from the principal component space. This approach helps identify outliers in high-dimensional datasets.

```python
def pca_anomaly_detection(X, n_components=2, threshold=3):
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # Calculate reconstruction error
    reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
    
    # Identify anomalies using standard deviation
    mean_error = np.mean(reconstruction_error)
    std_error = np.std(reconstruction_error)
    anomalies = reconstruction_error > mean_error + threshold * std_error
    
    return anomalies, reconstruction_error

# Example usage
np.random.seed(42)
X_normal = np.random.randn(100, 10)
X_anomaly = np.random.randn(5, 10) * 5
X_combined = np.vstack([X_normal, X_anomaly])

anomalies, errors = pca_anomaly_detection(X_combined)
print(f"Number of anomalies detected: {np.sum(anomalies)}")
```

Slide 7: Incremental PCA Implementation

For large datasets that don't fit in memory, Incremental PCA allows processing data in batches while maintaining mathematical equivalence to regular PCA. This implementation demonstrates how to perform PCA on chunks of data.

```python
from sklearn.decomposition import IncrementalPCA

def incremental_pca_processing(data_generator, batch_size=1000, n_components=2):
    # Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Process data in batches
    for batch in data_generator:
        ipca.partial_fit(batch)
    
    return ipca

# Example data generator
def generate_batches(n_samples=10000, n_features=50, batch_size=1000):
    for i in range(0, n_samples, batch_size):
        yield np.random.randn(min(batch_size, n_samples - i), n_features)

# Process data incrementally
ipca = incremental_pca_processing(generate_batches())
print(f"Explained variance ratios: {ipca.explained_variance_ratio_}")
```

Slide 8: Real-world Example - Financial Data Analysis

In this example, we apply PCA to a financial dataset containing daily returns of multiple stocks to identify common factors driving market movements and reduce the dimensionality for portfolio optimization.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate sample financial data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
n_stocks = 30
stock_returns = pd.DataFrame(
    np.random.randn(len(dates), n_stocks) * 0.02,  # 2% daily volatility
    index=dates,
    columns=[f'Stock_{i}' for i in range(n_stocks)]
)

def analyze_market_factors(returns_df, n_factors=3):
    # Standardize returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_df)
    
    # Apply PCA
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(returns_scaled)
    
    # Create factors DataFrame
    factors_df = pd.DataFrame(
        factors,
        index=returns_df.index,
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    
    return factors_df, pca.explained_variance_ratio_

factors_df, explained_var = analyze_market_factors(stock_returns)
print(f"Variance explained by factors: {explained_var * 100:.2f}%")
```

Slide 9: Source Code for Visualization Components

This implementation provides comprehensive visualization tools for PCA results, including scree plots, biplot visualization, and component loading analysis.

```python
def create_pca_visualizations(X, feature_names=None):
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Fit PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Scree plot
    plt.subplot(131)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'bo-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    
    # Cumulative variance plot
    plt.subplot(132)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_), 'ro-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    # Component loadings
    plt.subplot(133)
    loadings = pca.components_[:2].T
    plt.scatter(loadings[:, 0], loadings[:, 1])
    for i, txt in enumerate(feature_names):
        plt.annotate(txt, (loadings[i, 0], loadings[i, 1]))
    plt.title('PCA Loading Plot')
    plt.xlabel('PC1 Loadings')
    plt.ylabel('PC2 Loadings')
    
    plt.tight_layout()
    return fig
```

Slide 10: PCA for Time Series Dimensionality Reduction

PCA can effectively reduce the dimensionality of multivariate time series data while preserving temporal patterns. This implementation shows how to apply PCA to time series data with sliding windows for feature extraction.

```python
def time_series_pca(data, window_size=10, n_components=2):
    # Create sliding windows
    n_samples = data.shape[0] - window_size + 1
    windows = np.zeros((n_samples, window_size * data.shape[1]))
    
    for i in range(n_samples):
        windows[i] = data[i:i+window_size].flatten()
    
    # Apply PCA to windowed data
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(windows)
    
    # Return transformed data and explained variance
    return transformed_data, pca.explained_variance_ratio_

# Example usage with synthetic time series
n_timesteps = 1000
n_features = 5
time_series = np.random.randn(n_timesteps, n_features)
transformed, var_ratio = time_series_pca(time_series)
print(f"Explained variance ratios: {var_ratio}")
```

Slide 11: Robust PCA Implementation

Robust PCA is less sensitive to outliers than standard PCA, decomposing the data matrix into low-rank and sparse components. This implementation uses an iterative approach to perform robust PCA.

```python
def robust_pca(X, max_iter=100, tol=1e-7):
    # Initialize variables
    n, m = X.shape
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    mu = np.prod(X.shape) / (4 * np.linalg.norm(X, ord=1))
    
    for iter_num in range(max_iter):
        # Update Low-rank component
        U, sig, Vt = np.linalg.svd(X - S + (1/mu) * Y, full_matrices=False)
        sig = np.maximum(sig - 1/mu, 0)
        L_new = U @ np.diag(sig) @ Vt
        
        # Update Sparse component
        S_new = np.sign(X - L_new + (1/mu) * Y) * \
                np.maximum(np.abs(X - L_new + (1/mu) * Y) - 1/mu, 0)
        
        # Update dual variable
        Y = Y + mu * (X - L_new - S_new)
        
        # Check convergence
        if np.linalg.norm(L_new - L) / np.linalg.norm(L) < tol:
            break
            
        L = L_new
        S = S_new
    
    return L, S

# Example usage
n_samples = 100
n_features = 50
X = np.random.randn(n_samples, n_features)
# Add sparse noise
X[np.random.choice(n_samples, 10), np.random.choice(n_features, 5)] = 10

L, S = robust_pca(X)
print(f"Low-rank matrix shape: {L.shape}")
print(f"Sparse matrix shape: {S.shape}")
```

Slide 12: Kernel PCA Implementation

Kernel PCA extends PCA to handle nonlinear relationships in data by implicitly mapping the data to a higher-dimensional space using the kernel trick. This implementation demonstrates different kernel functions.

```python
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

def kernel_pca(X, n_components=2, kernel='rbf', gamma=1.0, degree=3):
    # Compute kernel matrix
    if kernel == 'rbf':
        K = rbf_kernel(X, gamma=gamma)
    elif kernel == 'polynomial':
        K = polynomial_kernel(X, degree=degree)
    else:
        raise ValueError("Unsupported kernel type")
    
    # Center kernel matrix
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(K_centered)
    
    # Sort eigenvectors in descending order
    indices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[indices]
    eigenvecs = eigenvecs[:, indices]
    
    # Select top components and normalize
    return eigenvecs[:, :n_components] * np.sqrt(1/eigenvals[:n_components])

# Example with nonlinear data
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=100, noise=0.1)
X_kpca = kernel_pca(X, kernel='rbf', gamma=2)
```

Slide 13: Real-world Example - Text Document Analysis

PCA can be applied to text data after converting documents to term frequency-inverse document frequency (TF-IDF) vectors. This implementation shows how to reduce dimensionality of document vectors while preserving semantic relationships.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def analyze_text_documents(documents, n_components=2):
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(documents)
    
    # Apply PCA (using TruncatedSVD for sparse matrices)
    pca = TruncatedSVD(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Get feature importance
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=vectorizer.get_feature_names_out()
    )
    
    return X_reduced, feature_importance, pca.explained_variance_ratio_

# Example usage
documents = [
    "machine learning applications in healthcare",
    "deep learning neural networks research",
    "healthcare data analysis methods",
    "artificial intelligence in medicine"
]

reduced_docs, features, var_ratio = analyze_text_documents(documents)
print(f"Explained variance ratio: {var_ratio}")
```

Slide 14: Cross-Validation for PCA Component Selection

This implementation demonstrates how to use cross-validation to select the optimal number of principal components, measuring the reconstruction error on held-out data.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cv_component_selection(X, max_components=None, n_splits=5):
    if max_components is None:
        max_components = X.shape[1]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    reconstruction_errors = []
    
    for n_comp in range(1, max_components + 1):
        fold_errors = []
        
        for train_idx, val_idx in kf.split(X):
            # Fit PCA on training data
            pca = PCA(n_components=n_comp)
            pca.fit(X[train_idx])
            
            # Transform and reconstruct validation data
            X_val_transformed = pca.transform(X[val_idx])
            X_val_reconstructed = pca.inverse_transform(X_val_transformed)
            
            # Calculate reconstruction error
            error = mean_squared_error(X[val_idx], X_val_reconstructed)
            fold_errors.append(error)
        
        reconstruction_errors.append(np.mean(fold_errors))
    
    # Plot reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_components + 1), reconstruction_errors, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Reconstruction Error')
    plt.title('Cross-Validation Reconstruction Error vs. Components')
    
    return reconstruction_errors

# Example usage
X = np.random.randn(100, 10)
errors = cv_component_selection(X)
optimal_components = np.argmin(errors) + 1
print(f"Optimal number of components: {optimal_components}")
```

Slide 15: Additional Resources

*   arXiv Papers for Deep Dive into PCA:
    *   "A Tutorial on Principal Component Analysis" - [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
    *   "Robust Principal Component Analysis?" - [https://arxiv.org/abs/0912.3599](https://arxiv.org/abs/0912.3599)
    *   "Random Projections for k-means Clustering" - [https://arxiv.org/abs/1412.2486](https://arxiv.org/abs/1412.2486)
    *   "Online Principal Component Analysis in High Dimension" - [https://arxiv.org/abs/1307.0032](https://arxiv.org/abs/1307.0032)
*   Recommended Search Keywords:
    *   "PCA dimensionality reduction techniques"
    *   "Kernel PCA implementations"
    *   "Robust PCA algorithms"
    *   "Incremental PCA for large datasets"
    *   "Cross-validation in PCA"

