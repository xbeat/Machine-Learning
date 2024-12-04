## Packing Your Data's Suitcase An Introduction to PCA
Slide 1: Principal Component Analysis Fundamentals

Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional space while preserving maximum variance. It accomplishes this by identifying orthogonal directions, called principal components, along which the data varies most significantly.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 4)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate covariance matrix
covariance_matrix = np.cov(X_scaled.T)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:", eigenvalues)
print("Explained variance ratio:", eigenvalues / np.sum(eigenvalues))
```

Slide 2: Mathematical Foundation of PCA

The mathematical basis of PCA involves finding the eigenvectors and eigenvalues of the covariance matrix. These eigenvectors represent the principal components, while eigenvalues indicate the amount of variance explained by each component.

```python
# Mathematical formulation in LaTeX (not rendered)
"""
$$
\Sigma = \frac{1}{n-1}X^TX
$$

$$
\Sigma v = \lambda v
$$

Where:
$$\Sigma$$ is the covariance matrix
$$\lambda$$ represents eigenvalues
$$v$$ represents eigenvectors
"""

def calculate_pca_manually(X):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    return eigenvals[::-1], eigenvecs[:, ::-1]
```

Slide 3: Implementing PCA from Scratch

Understanding the core implementation of PCA helps grasp its underlying mechanics. This implementation demonstrates the step-by-step process of performing PCA without using pre-built libraries, focusing on the fundamental mathematical operations.

```python
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx][:, :self.n_components]
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
```

Slide 4: Data Preprocessing for PCA

Proper data preprocessing is crucial for PCA effectiveness. This involves handling missing values, scaling features, and ensuring data quality. Standardization is particularly important as PCA is sensitive to the scale of input features.

```python
def preprocess_for_pca(X, handle_missing=True, standardize=True):
    # Handle missing values if specified
    if handle_missing:
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    
    # Standardize features if specified
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, scaler if standardize else None

# Example usage
X = np.array([[1, 2, np.nan], [4, 5, 6], [7, 8, 9]])
X_processed, scaler = preprocess_for_pca(X)
print("Processed data:\n", X_processed)
```

Slide 5: Variance Explained and Component Selection

Determining the optimal number of principal components is crucial for effective dimensionality reduction. This process involves analyzing the cumulative explained variance ratio and selecting components that capture a sufficient amount of data variance.

```python
import matplotlib.pyplot as plt

def analyze_variance_explained(X, plot=True):
    # Perform PCA
    pca = PCAFromScratch(n_components=X.shape[1])
    pca.fit(X)
    
    # Calculate explained variance ratio
    eigenvalues = np.linalg.eigvalsh(np.cov(X.T))[::-1]
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
                cumulative_variance_ratio, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance vs. Number of Components')
        plt.grid(True)
        plt.show()
    
    return explained_variance_ratio, cumulative_variance_ratio
```

Slide 6: PCA for Image Compression

PCA can be effectively used for image compression by reducing the dimensionality of image data while preserving essential visual information. This implementation demonstrates how to compress and reconstruct grayscale images using PCA.

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compress_image_pca(image_path, n_components):
    # Load and convert image to grayscale
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    # Apply PCA
    pca = PCAFromScratch(n_components=n_components)
    compressed = pca.fit_transform(img_array)
    reconstructed = np.dot(compressed, pca.components.T) + pca.mean
    
    # Calculate compression ratio
    original_size = img_array.size
    compressed_size = compressed.size + pca.components.size
    compression_ratio = original_size / compressed_size
    
    return reconstructed, compression_ratio

# Example usage
reconstructed, ratio = compress_image_pca('sample_image.jpg', 50)
print(f"Compression ratio: {ratio:.2f}")
```

Slide 7: Real-World Application: Stock Market Analysis

PCA helps identify underlying patterns in financial market data by reducing the dimensionality of multiple correlated stock prices to a smaller set of factors that drive market movements.

```python
import pandas as pd
import yfinance as yf

def analyze_stock_market_pca():
    # Download stock data
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
    data = pd.DataFrame()
    
    for ticker in tickers:
        stock = yf.download(ticker, start='2020-01-01', end='2023-12-31')
        data[ticker] = stock['Close']
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Apply PCA
    pca = PCAFromScratch(n_components=2)
    principal_components = pca.fit_transform(returns)
    
    return principal_components, returns

# Example visualization
components, returns = analyze_stock_market_pca()
plt.scatter(components[:, 0], components[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Stock Market PCA Analysis')
```

Slide 8: PCA for Anomaly Detection

Principal Component Analysis can be used effectively for anomaly detection by identifying data points that deviate significantly from the principal components in the transformed space.

```python
def detect_anomalies_pca(X, threshold=3):
    # Standardize data
    X_scaled, _ = preprocess_for_pca(X)
    
    # Apply PCA
    pca = PCAFromScratch(n_components=2)
    X_transformed = pca.fit_transform(X_scaled)
    
    # Reconstruct data
    X_reconstructed = np.dot(X_transformed, pca.components.T) + pca.mean
    
    # Calculate reconstruction error
    reconstruction_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
    
    # Identify anomalies
    anomalies = reconstruction_error > (np.mean(reconstruction_error) + 
                                      threshold * np.std(reconstruction_error))
    
    return anomalies, reconstruction_error

# Test with random data containing anomalies
X = np.random.randn(1000, 10)
X[0:10] = X[0:10] * 5  # Create anomalies
anomalies, errors = detect_anomalies_pca(X)
print(f"Number of anomalies detected: {np.sum(anomalies)}")
```

Slide 9: Incremental PCA Implementation

For large datasets that don't fit in memory, Incremental PCA provides a solution by processing data in batches while maintaining the same mathematical properties as standard PCA.

```python
class IncrementalPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.n_samples_seen = 0
        
    def partial_fit(self, X):
        if self.mean is None:
            self.mean = np.zeros(X.shape[1])
            self.components = np.zeros((self.n_components, X.shape[1]))
        
        # Update mean and covariance incrementally
        n_samples = X.shape[0]
        self.n_samples_seen += n_samples
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = self.mean + (np.mean(X, axis=0) - self.mean) * (
            n_samples / self.n_samples_seen)
        
        # Update components using online SVD
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        
        return self
```

Slide 10: Kernel PCA Implementation

Kernel PCA extends traditional PCA to handle nonlinear relationships in data by implicitly mapping the data to a higher-dimensional feature space using the kernel trick, enabling nonlinear dimensionality reduction.

```python
from scipy.spatial.distance import pdist, squareform

class KernelPCA:
    def __init__(self, n_components, kernel='rbf', gamma=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.alphas = None
        self.X_fit = None
        
    def compute_kernel_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == 'rbf':
            distances = pdist(X, metric='sqeuclidean')
            K = squareform(distances)
            K = np.exp(-self.gamma * K)
        return K
    
    def fit_transform(self, X):
        n_samples = X.shape[0]
        K = self.compute_kernel_matrix(X)
        
        # Center kernel matrix
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(K_centered)
        indices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[indices]
        eigenvecs = eigenvecs[:, indices]
        
        # Store results
        self.alphas = eigenvecs[:, :self.n_components]
        self.X_fit = X
        
        return eigenvecs[:, :self.n_components] * np.sqrt(eigenvals[:self.n_components])

# Example usage
X = np.random.randn(100, 5)
kpca = KernelPCA(n_components=2, gamma=0.1)
X_transformed = kpca.fit_transform(X)
```

Slide 11: PCA for Time Series Dimensionality Reduction

Time series data often contains multiple correlated features that can be effectively reduced using PCA while preserving temporal patterns and trends for more efficient analysis and forecasting.

```python
def reduce_timeseries_dimensions(timeseries_data, n_components=2, window_size=10):
    # Create windowed segments
    n_samples = len(timeseries_data) - window_size + 1
    windows = np.zeros((n_samples, window_size))
    
    for i in range(n_samples):
        windows[i] = timeseries_data[i:i+window_size]
    
    # Apply PCA
    pca = PCAFromScratch(n_components=n_components)
    reduced_data = pca.fit_transform(windows)
    
    # Reconstruct the signal
    reconstructed = pca.inverse_transform(reduced_data)
    
    return reduced_data, reconstructed, pca.components

# Generate sample time series
t = np.linspace(0, 10, 1000)
signal = np.sin(t) + 0.5 * np.sin(5*t) + np.random.normal(0, 0.1, len(t))

reduced, reconstructed, components = reduce_timeseries_dimensions(signal)
print(f"Compression ratio: {signal.size / reduced.size:.2f}")
```

Slide 12: Robust PCA Implementation

Robust PCA decomposes a matrix into low-rank and sparse components, making it resistant to outliers and useful for applications like background subtraction in video processing.

```python
def robust_pca(X, lambda_=1.0, max_iter=100, tol=1e-7):
    n, m = X.shape
    L = np.zeros((n, m))
    S = np.zeros((n, m))
    Y = np.zeros((n, m))
    mu = 1.25/np.linalg.norm(X, 2)
    mu_bar = mu * 1e7
    rho = 1.5
    
    for i in range(max_iter):
        # Update L
        U, sigma, Vt = np.linalg.svd(X - S + (1/mu) * Y, full_matrices=False)
        sigma = np.maximum(sigma - mu, 0)
        L_new = U.dot(np.diag(sigma)).dot(Vt)
        
        # Update S
        temp = X - L_new + (1/mu) * Y
        S_new = np.sign(temp) * np.maximum(np.abs(temp) - lambda_/mu, 0)
        
        # Update Y
        Y = Y + mu * (X - L_new - S_new)
        
        # Update mu
        mu = min(rho * mu, mu_bar)
        
        # Check convergence
        if (np.linalg.norm(L_new - L, 'fro') / np.linalg.norm(L, 'fro') < tol and
            np.linalg.norm(S_new - S, 'fro') / np.linalg.norm(S, 'fro') < tol):
            break
            
        L = L_new
        S = S_new
        
    return L, S

# Example usage
X = np.random.randn(50, 50)
X[0:5, 0:5] = 10  # Add some outliers
L, S = robust_pca(X)
```

Slide 13: Additional Resources

*   Principal Component Analysis: A Review and Recent Developments [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
*   A Tutorial on Principal Component Analysis [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
*   Kernel PCA for Feature Extraction and De-noising [https://arxiv.org/abs/1207.3538](https://arxiv.org/abs/1207.3538)
*   Robust Principal Component Analysis: Exact Recovery of Corrupted Low-Rank Matrices [https://arxiv.org/abs/0912.3599](https://arxiv.org/abs/0912.3599)
*   Online Principal Component Analysis in High Dimension Search on Google Scholar for recent papers on Online PCA implementations

