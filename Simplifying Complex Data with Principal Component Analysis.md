## Simplifying Complex Data with Principal Component Analysis
Slide 1: Understanding PCA Mathematics

Principal Component Analysis relies on fundamental linear algebra concepts to transform data. The core mathematical foundation involves computing the covariance matrix, eigenvalues, and eigenvectors to identify directions of maximum variance in the dataset.

```python
import numpy as np

def explain_pca_math(X):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    print("Covariance Matrix:")
    print(cov_matrix)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Mathematical formulation in comments:
    # $$Σ = \frac{1}{n-1}X^TX$$
    # $$Σv = λv$$
    
    return eigenvalues, eigenvectors

# Example usage
X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
eigenvals, eigenvecs = explain_pca_math(X)
print("\nEigenvalues:", eigenvals)
print("Eigenvectors:\n", eigenvecs)
```

Slide 2: Implementing PCA from Scratch

This implementation demonstrates the core PCA algorithm without using specialized libraries. We'll create a class that handles data standardization, component calculation, and dimensionality reduction through matrix operations.

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.var = None
        
    def fit(self, X):
        # Standardize the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Calculate covariance matrix
        cov_matrix = np.cov(X.T)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components
        self.components = eigenvectors[:, :self.n_components]
        
        return self
        
    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components)
```

Slide 3: Data Preprocessing for PCA

Before applying PCA, data must be properly preprocessed to ensure optimal results. This includes handling missing values, scaling features, and removing outliers that could skew the principal components.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_for_pca(data):
    # Handle missing values
    data = pd.DataFrame(data).fillna(method='ffill')
    
    # Remove outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

# Example usage
data = np.random.randn(100, 5)
processed_data, scaler = preprocess_for_pca(data)
print("Original shape:", data.shape)
print("Processed shape:", processed_data.shape)
```

Slide 4: Variance Explained Analysis

Understanding how much variance each principal component explains is crucial for determining the optimal number of components to retain. This implementation calculates and visualizes the explained variance ratio.

```python
import matplotlib.pyplot as plt

def analyze_variance_explained(X):
    # Calculate PCA
    pca = PCA()
    pca.fit(X)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
             cumulative_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs Number of Components')
    plt.grid(True)
    
    return explained_variance_ratio, cumulative_variance_ratio

# Example usage with synthetic data
X = np.random.randn(200, 10)
var_ratio, cum_var_ratio = analyze_variance_explained(X)
print("Explained variance ratio:", var_ratio)
print("Cumulative explained variance ratio:", cum_var_ratio)
```

Slide 5: Real-world Application - Image Compression

PCA effectively reduces image dimensionality while preserving essential features. This implementation demonstrates image compression by retaining only the most significant principal components to reconstruct the original image.

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compress_image(image_path, n_components):
    # Load and convert image to grayscale
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    img_transformed = pca.fit_transform(img_data)
    img_reconstructed = pca.inverse_transform(img_transformed)
    
    # Calculate compression ratio
    original_size = img_data.size * img_data.itemsize
    compressed_size = (img_transformed.size + pca.components_.size) * img_data.itemsize
    compression_ratio = original_size / compressed_size
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_data, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(img_reconstructed, cmap='gray')
    ax2.set_title(f'Compressed Image\nRatio: {compression_ratio:.2f}:1')
    
    return compression_ratio, img_reconstructed
```

Slide 6: Dimensionality Reduction for High-Dimensional Data

In this implementation, we handle high-dimensional datasets by efficiently selecting optimal components while maintaining computational performance through batch processing and memory optimization.

```python
class EfficientPCA:
    def __init__(self, n_components, batch_size=1000):
        self.n_components = n_components
        self.batch_size = batch_size
        self.components_ = None
        
    def fit_transform(self, X):
        n_samples, n_features = X.shape
        
        # Initialize variables for incremental mean and covariance
        mean = np.zeros(n_features)
        cov = np.zeros((n_features, n_features))
        
        # Calculate mean and covariance in batches
        for i in range(0, n_samples, self.batch_size):
            batch = X[i:min(i + self.batch_size, n_samples)]
            mean += np.sum(batch, axis=0)
            cov += np.dot(batch.T, batch)
        
        mean /= n_samples
        cov = (cov / n_samples) - np.outer(mean, mean)
        
        # Compute eigenvectors for largest eigenvalues
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        indices = np.argsort(eigenvals)[::-1][:self.n_components]
        
        self.components_ = eigenvecs[:, indices]
        return np.dot(X - mean, self.components_)
```

Slide 7: Implementing Kernel PCA

Kernel PCA extends traditional PCA to handle nonlinear relationships in data by implicitly mapping the input space to a higher-dimensional feature space using the kernel trick.

```python
class KernelPCA:
    def __init__(self, n_components, kernel='rbf', gamma=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        
    def rbf_kernel(self, X, Y):
        # Compute RBF kernel matrix
        # $$K(x,y) = exp(-γ||x-y||^2)$$
        dist_matrix = np.sum(X**2, axis=1)[:, np.newaxis] + \
                     np.sum(Y**2, axis=1) - \
                     2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * dist_matrix)
    
    def fit_transform(self, X):
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.rbf_kernel(X, X)
        
        # Center kernel matrix
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - np.dot(one_n, K) - \
                    np.dot(K, one_n) + \
                    np.dot(np.dot(one_n, K), one_n)
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(K_centered)
        indices = np.argsort(eigenvals)[::-1][:self.n_components]
        
        return eigenvecs[:, indices] * np.sqrt(eigenvals[indices])
```

Slide 8: Incremental PCA Implementation

Incremental PCA allows processing of large datasets that don't fit in memory by updating the components incrementally, making it memory-efficient for big data applications.

```python
class IncrementalPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.n_samples_seen_ = 0
        
    def partial_fit(self, X):
        n_samples, n_features = X.shape
        
        if self.components_ is None:
            self.components_ = np.zeros((self.n_components, n_features))
            self.mean_ = np.zeros(n_features)
            
        # Update mean
        old_mean = self.mean_.copy()
        self.mean_ = (self.n_samples_seen_ * self.mean_ + X.sum(axis=0)) / \
                    (self.n_samples_seen_ + n_samples)
        
        # Update components
        X_centered = X - self.mean_
        cov = np.dot(X_centered.T, X_centered) / n_samples
        
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        indices = np.argsort(eigenvals)[::-1][:self.n_components]
        self.components_ = eigenvecs[:, indices].T
        
        self.n_samples_seen_ += n_samples
        return self
```

Slide 9: Sparse PCA Implementation

Sparse PCA produces sparse loadings by imposing L1 penalty, making the principal components more interpretable while maintaining good reconstruction properties for high-dimensional data analysis.

```python
import numpy as np
from sklearn.linear_model import Lasso

class SparsePCA:
    def __init__(self, n_components, alpha=1.0, max_iter=100):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.components_ = None
        
    def fit_transform(self, X):
        n_samples, n_features = X.shape
        
        # Initialize components randomly
        self.components_ = np.random.randn(self.n_components, n_features)
        
        # Alternate optimization
        for _ in range(self.max_iter):
            # Fix components, update scores
            scores = np.dot(X, self.components_.T)
            
            # Fix scores, update components
            for comp_idx in range(self.n_components):
                lasso = Lasso(alpha=self.alpha)
                residual = X - np.dot(scores[:, :comp_idx], 
                                    self.components_[:comp_idx])
                lasso.fit(scores[:, comp_idx:comp_idx+1], residual)
                self.components_[comp_idx] = lasso.coef_
                
        return np.dot(X, self.components_.T)
```

Slide 10: Real-world Application - Stock Market Analysis

This implementation demonstrates PCA application in financial data analysis, reducing the dimensionality of multiple stock price time series to identify common market factors.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def analyze_stock_market(prices_df, n_components=3):
    # Compute returns
    returns = prices_df.pct_change().dropna()
    
    # Standardize returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled_returns)
    
    # Create component DataFrame
    component_df = pd.DataFrame(
        components,
        index=returns.index,
        columns=[f'Market Factor {i+1}' for i in range(n_components)]
    )
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    
    # Calculate factor correlations
    correlations = pd.DataFrame(
        pca.components_,
        columns=returns.columns,
        index=[f'Factor {i+1}' for i in range(n_components)]
    )
    
    return component_df, explained_var, correlations

# Example usage:
# prices_df = pd.DataFrame of stock prices
# factors, var_explained, cors = analyze_stock_market(prices_df)
```

Slide 11: Robust PCA Implementation

Robust PCA decomposes a matrix into low-rank and sparse components, making it resistant to outliers and noise in the data while preserving the main structural patterns.

```python
class RobustPCA:
    def __init__(self, max_iter=1000, tol=1e-7):
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, X):
        m, n = X.shape
        lambda_param = 1 / np.sqrt(max(m, n))
        
        # Initialize variables
        S = np.zeros_like(X)
        Y = np.zeros_like(X)
        mu = 1.25 / np.linalg.norm(X, 2)
        mu_bar = mu * 1e7
        rho = 1.5
        
        for i in range(self.max_iter):
            # Update low-rank component
            U, sigma, Vt = np.linalg.svd(
                X - S + (1/mu) * Y, 
                full_matrices=False
            )
            sigma = np.maximum(sigma - mu, 0)
            L = np.dot(U * sigma, Vt)
            
            # Update sparse component
            temp = X - L + (1/mu) * Y
            S = np.sign(temp) * np.maximum(
                np.abs(temp) - lambda_param/mu, 0
            )
            
            # Update dual variable
            Z = X - L - S
            Y = Y + mu * Z
            mu = min(rho * mu, mu_bar)
            
            # Check convergence
            err = np.linalg.norm(Z, 'fro') / np.linalg.norm(X, 'fro')
            if err < self.tol:
                break
                
        return L, S  # Low-rank and Sparse components
```

Slide 12: PCA for Anomaly Detection

This implementation uses PCA for detecting anomalies by measuring the reconstruction error between original data points and their projections, identifying outliers that deviate significantly from the principal subspace.

```python
class PCAAnomaly:
    def __init__(self, n_components, threshold_percentile=95):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.pca = None
        self.threshold = None
        
    def fit(self, X):
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        X_transformed = self.pca.fit_transform(X)
        
        # Reconstruct data
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.sum(
            (X - X_reconstructed) ** 2, 
            axis=1
        )
        
        # Set threshold based on percentile
        self.threshold = np.percentile(
            reconstruction_errors, 
            self.threshold_percentile
        )
        
    def predict(self, X):
        # Transform and reconstruct
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        
        # Calculate reconstruction errors
        errors = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Classify anomalies
        return errors > self.threshold

# Example usage
X = np.random.randn(1000, 10)
detector = PCAAnomaly(n_components=5)
detector.fit(X)
anomalies = detector.predict(X)
```

Slide 13: PCA with Missing Values

This implementation handles datasets with missing values using an iterative imputation approach combined with PCA, allowing for robust dimensionality reduction even with incomplete data.

```python
class PCAMV:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, X):
        # Create mask for missing values
        missing_mask = np.isnan(X)
        
        # Initial mean imputation
        X_imputed = X.copy()
        column_means = np.nanmean(X, axis=0)
        for col in range(X.shape[1]):
            mask = missing_mask[:, col]
            X_imputed[mask, col] = column_means[col]
            
        prev_X = np.zeros_like(X)
        
        for iteration in range(self.max_iter):
            # Perform PCA on current imputation
            pca = PCA(n_components=self.n_components)
            X_transformed = pca.fit_transform(X_imputed)
            X_reconstructed = pca.inverse_transform(X_transformed)
            
            # Update only missing values
            X_imputed[missing_mask] = X_reconstructed[missing_mask]
            
            # Check convergence
            diff = np.mean((X_imputed - prev_X) ** 2)
            if diff < self.tol:
                break
                
            prev_X = X_imputed.copy()
            
        return X_transformed, X_imputed

# Example usage
X = np.random.randn(100, 10)
X[np.random.random(X.shape) < 0.1] = np.nan  # Add missing values
pcamv = PCAMV(n_components=5)
X_transformed, X_imputed = pcamv.fit_transform(X)
```

Slide 14: PCA for Time Series Dimensionality Reduction

This implementation applies PCA to time series data, incorporating sliding windows to capture temporal patterns and reduce the dimensionality of multivariate time series while preserving temporal dependencies.

```python
class TimeSeriesPCA:
    def __init__(self, n_components, window_size):
        self.n_components = n_components
        self.window_size = window_size
        self.pca = None
        
    def create_windows(self, X):
        n_samples = X.shape[0] - self.window_size + 1
        windows = np.zeros((n_samples, X.shape[1] * self.window_size))
        
        for i in range(n_samples):
            windows[i] = X[i:i+self.window_size].flatten()
            
        return windows
    
    def fit_transform(self, X):
        # Create windowed version of data
        X_windowed = self.create_windows(X)
        
        # Apply PCA
        self.pca = PCA(n_components=self.n_components)
        X_transformed = self.pca.fit_transform(X_windowed)
        
        # Calculate explained variance per component
        explained_variance = self.pca.explained_variance_ratio_
        
        return X_transformed, explained_variance
        
# Example usage
ts_data = np.random.randn(1000, 5)  # 5 time series
ts_pca = TimeSeriesPCA(n_components=3, window_size=10)
transformed_data, variance = ts_pca.fit_transform(ts_data)
```

Slide 15: Additional Resources

1.  "A Tutorial on Principal Component Analysis" - [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
2.  "Robust Principal Component Analysis?" - [https://arxiv.org/abs/0912.3599](https://arxiv.org/abs/0912.3599)
3.  "Online Principal Component Analysis in High Dimension: Tracking Data Streams" - [https://arxiv.org/abs/1607.03463](https://arxiv.org/abs/1607.03463)
4.  "Random Projections for k-means Clustering" - [https://arxiv.org/abs/1412.2486](https://arxiv.org/abs/1412.2486)
5.  "Large-scale Machine Learning with Stochastic Gradient Descent" - [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)

