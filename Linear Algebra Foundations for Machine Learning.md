## Linear Algebra Foundations for Machine Learning
Slide 1: Vectors and Matrices in Machine Learning

Linear algebra operations form the backbone of machine learning algorithms. Understanding vector and matrix manipulations is crucial as they represent features and transformations. The dot product between vectors encapsulates the similarity between data points in high-dimensional spaces.

```python
import numpy as np

# Creating feature vectors and weight matrix
features = np.array([2.5, 3.0, -1.2, 0.8])  # Feature vector
weights = np.array([[0.1, 0.2, 0.3],
                   [0.4, 0.5, 0.6],
                   [-0.1, -0.2, -0.3],
                   [0.7, 0.8, 0.9]])  # Weight matrix

# Linear transformation: projecting features to new space
transformed = np.dot(features, weights)
print(f"Original dimension: {features.shape}")
print(f"Transformed dimension: {transformed.shape}")
print(f"Transformed features: {transformed}")
```

Slide 2: Linear Independence and Feature Selection

Linear independence in feature vectors ensures that each feature contributes unique information to the model. This concept directly relates to feature selection and dimensionality reduction, helping eliminate redundant or correlated features that might impact model performance.

```python
import numpy as np
from sklearn.decomposition import PCA

# Generate correlated features
n_samples = 1000
x1 = np.random.normal(0, 1, n_samples)
x2 = 2 * x1 + np.random.normal(0, 0.1, n_samples)  # Highly correlated with x1
x3 = np.random.normal(0, 1, n_samples)  # Independent feature

# Create feature matrix
X = np.column_stack([x1, x2, x3])

# Check correlation matrix
correlation_matrix = np.corrcoef(X.T)
print("Correlation Matrix:")
print(correlation_matrix)

# Apply PCA to check linear independence
pca = PCA()
pca.fit(X)
print("\nExplained variance ratio:", pca.explained_variance_ratio_)
```

Slide 3: Matrix Decomposition in ML

Matrix decomposition techniques like Singular Value Decomposition (SVD) are fundamental in dimensionality reduction, recommendation systems, and feature extraction. SVD decomposes a matrix into three components, revealing underlying patterns in high-dimensional data.

```python
import numpy as np
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=100, n_features=5, random_state=42)

# Perform SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Reconstruct data using top k components
k = 3
X_reconstructed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Calculate reconstruction error
error = np.linalg.norm(X - X_reconstructed, 'fro')
print(f"Original shape: {X.shape}")
print(f"Reconstruction error: {error:.4f}")
print(f"Singular values: {S}")
```

Slide 4: Eigenvalues in Principal Component Analysis

Eigenvalues and eigenvectors play a crucial role in PCA, determining the directions of maximum variance in the data. The eigenvalues represent the amount of variance explained by each principal component, helping in dimensionality reduction decisions.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate correlated data
n_samples = 200
x = np.random.normal(0, 1, n_samples)
y = 2*x + np.random.normal(0, 0.5, n_samples)
data = np.column_stack([x, y])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Calculate covariance matrix
cov_matrix = np.cov(data_scaled.T)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Covariance Matrix:")
print(cov_matrix)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
```

Slide 5: Implementing Linear Regression Using Linear Algebra

The mathematical foundation of linear regression can be expressed through linear algebra operations. The closed-form solution involves matrix operations to find optimal weights that minimize the sum of squared residuals.

```python
import numpy as np

class LinearRegressionLA:
    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]
        # Closed form solution: w = (X^T X)^(-1) X^T y
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights

# Example usage
X = np.random.rand(100, 1)
y = 3*X.reshape(-1) + 2 + np.random.normal(0, 0.1, 100)

model = LinearRegressionLA()
model.fit(X, y)
print("Weights (bias, slope):", model.weights)
```

Slide 6: Vector Spaces and Neural Network Activations

Neural network activations can be viewed as transformations between vector spaces. Each layer maps inputs to a new vector space, with activation functions introducing non-linearity. Understanding these transformations helps in network architecture design.

```python
import numpy as np

class NeuralLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
        
    def forward(self, X):
        # Linear transformation
        linear_output = np.dot(X, self.weights) + self.bias
        # Non-linear activation (ReLU)
        activated_output = np.maximum(0, linear_output)
        return activated_output

# Example usage
X = np.random.randn(4, 3)  # 4 samples, 3 features
layer1 = NeuralLayer(3, 5)  # Transform to 5-dimensional space
layer2 = NeuralLayer(5, 2)  # Transform to 2-dimensional space

# Forward pass
hidden = layer1.forward(X)
output = layer2.forward(hidden)

print(f"Input shape: {X.shape}")
print(f"Hidden layer shape: {hidden.shape}")
print(f"Output shape: {output.shape}")
```

Slide 7: Matrix Calculus in Gradient Descent

Matrix calculus forms the foundation of gradient-based optimization in machine learning. The gradient of a loss function with respect to weight matrices determines the direction of steepest descent for parameter updates.

```python
import numpy as np

def compute_gradient(X, y, weights):
    # Compute predictions
    predictions = X @ weights
    
    # Compute error
    error = predictions - y
    
    # Compute gradients using matrix calculus
    # dL/dW = X^T * (predictions - y) / n
    gradient = (X.T @ error) / len(y)
    
    return gradient

# Example gradient descent implementation
def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    weights = np.zeros(X.shape[1])
    
    for epoch in range(epochs):
        gradient = compute_gradient(X, y, weights)
        weights -= learning_rate * gradient
        
        # Compute loss
        loss = np.mean((X @ weights - y) ** 2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return weights

# Generate sample data
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + np.random.normal(0, 0.1, 100)

# Train model
learned_weights = gradient_descent(X, y)
print("True weights:", true_weights)
print("Learned weights:", learned_weights)
```

Slide 8: Orthogonality and Feature Engineering

Orthogonality in feature vectors ensures maximum information capture with minimal redundancy. This concept guides feature engineering practices and helps in creating more effective input representations for machine learning models.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_orthogonal_features(X):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute Gram-Schmidt orthogonalization
    Q = np.zeros_like(X_scaled)
    Q[:, 0] = X_scaled[:, 0] / np.linalg.norm(X_scaled[:, 0])
    
    for i in range(1, X_scaled.shape[1]):
        v = X_scaled[:, i]
        # Subtract projections onto previous vectors
        for j in range(i):
            proj = np.dot(v, Q[:, j]) * Q[:, j]
            v = v - proj
        # Normalize
        Q[:, i] = v / np.linalg.norm(v)
    
    return Q

# Example usage
X = np.random.randn(100, 3)  # Create random features
X_orthogonal = create_orthogonal_features(X)

# Verify orthogonality
correlation = np.corrcoef(X_orthogonal.T)
print("Correlation matrix of orthogonal features:")
print(correlation)

# Verify dot products are close to 0 (orthogonal) or 1 (same vector)
dot_products = X_orthogonal.T @ X_orthogonal
print("\nDot products between features:")
print(np.round(dot_products, 4))
```

Slide 9: Eigendecomposition in Covariance Matrices

Eigendecomposition of covariance matrices reveals the principal directions of variation in data. This fundamental concept underlies numerous dimensionality reduction and feature extraction techniques used in machine learning.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate correlated 2D data
n_points = 1000
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
                           
original_data = np.random.randn(n_points, 2)
data = original_data @ rotation_matrix * np.array([2, 0.5])

# Compute covariance matrix
covariance_matrix = np.cov(data.T)

# Compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Sort by eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Covariance matrix:\n", covariance_matrix)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Plot data and eigenvectors
plt.figure(figsize=(10, 10))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
for i in range(2):
    plt.arrow(0, 0, 
              eigenvectors[0, i] * eigenvalues[i],
              eigenvectors[1, i] * eigenvalues[i],
              head_width=0.1, head_length=0.1, fc='r', ec='r')
plt.axis('equal')
plt.grid(True)
```

Slide 10: Linear Transformations in Convolutional Neural Networks

Convolutional layers perform linear transformations through kernel operations. Each convolution operation can be represented as a matrix multiplication, where the kernel defines a linear transformation applied to local regions of the input.

```python
import numpy as np

def im2col(input_data, kernel_height, kernel_width, stride=1):
    # Convert input data into column form for efficient convolution
    N, C, H, W = input_data.shape
    out_h = (H - kernel_height) // stride + 1
    out_w = (W - kernel_width) // stride + 1

    col = np.zeros((N, C, kernel_height, kernel_width, out_h, out_w))
    for y in range(kernel_height):
        y_max = y + stride * out_h
        for x in range(kernel_width):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

# Example usage
# Create sample image data (N, C, H, W)
image = np.random.randn(1, 3, 32, 32)
kernel_size = 3
stride = 1

# Convert image to column format
col = im2col(image, kernel_size, kernel_size, stride)

# Create random kernel weights
num_filters = 16
weights = np.random.randn(num_filters, 3 * kernel_size * kernel_size)

# Perform convolution as matrix multiplication
output = np.dot(col, weights.T)
print(f"Input shape: {image.shape}")
print(f"Column shape: {col.shape}")
print(f"Output shape: {output.shape}")
```

Slide 11: Matrix Factorization for Recommendation Systems

Matrix factorization decomposes the user-item interaction matrix into lower-dimensional latent factors. This technique reveals hidden patterns in user preferences and item characteristics, enabling accurate recommendations.

```python
import numpy as np

class MatrixFactorization:
    def __init__(self, R, K, alpha=0.001, beta=0.02, iterations=100):
        self.R = R  # rating matrix
        self.K = K  # latent features
        self.alpha = alpha  # learning rate
        self.beta = beta    # regularization parameter
        self.iterations = iterations
        self.users, self.items = R.shape
        
    def train(self):
        # Initialize user and item latent feature matrices
        self.P = np.random.normal(scale=1./self.K, size=(self.users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.items, self.K))
        
        # Optimization loop
        for step in range(self.iterations):
            for u in range(self.users):
                for i in range(self.items):
                    if self.R[u, i] > 0:  # Only for observed ratings
                        # Compute error
                        e_ui = self.R[u, i] - np.dot(self.P[u, :], self.Q[i, :].T)
                        
                        # Update latent features
                        for k in range(self.K):
                            self.P[u, k] += self.alpha * (2 * e_ui * self.Q[i, k] - self.beta * self.P[u, k])
                            self.Q[i, k] += self.alpha * (2 * e_ui * self.P[u, k] - self.beta * self.Q[i, k])
            
            # Compute total error
            error = self.compute_error()
            if step % 10 == 0:
                print(f"Iteration {step}: error = {error:.4f}")
                
    def compute_error(self):
        error = 0
        for u in range(self.users):
            for i in range(self.items):
                if self.R[u, i] > 0:
                    error += pow(self.R[u, i] - np.dot(self.P[u, :], self.Q[i, :].T), 2)
        return np.sqrt(error)
    
    def predict(self, u, i):
        return np.dot(self.P[u, :], self.Q[i, :].T)

# Example usage
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
])

mf = MatrixFactorization(R, K=2)
mf.train()

# Make predictions
print("\nPredicted Rating Matrix:")
predicted_matrix = mf.P @ mf.Q.T
print(np.round(predicted_matrix, 2))
```

Slide 12: Basis Transformation in Feature Spaces

Kernel methods implicitly transform data into higher-dimensional spaces through basis transformations. This enables linear algorithms to capture non-linear patterns by operating in the transformed feature space.

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def polynomial_kernel(X, Y, degree=2):
    """Compute the polynomial kernel between X and Y."""
    return (1 + np.dot(X, Y.T)) ** degree

def gaussian_kernel(X, Y, gamma=1.0):
    """Compute the RBF (Gaussian) kernel between X and Y."""
    return rbf_kernel(X, Y, gamma=gamma)

# Generate non-linear separable data
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.where((X[:, 0]**2 + X[:, 1]**2) < 1, -1, 1)

# Compute kernel matrices
K_poly = polynomial_kernel(X, X)
K_rbf = gaussian_kernel(X, X)

# Compute centered kernel matrices
n = len(X)
I = np.eye(n)
ones = np.ones((n, n)) / n
K_poly_centered = K_poly - np.dot(ones, K_poly) - np.dot(K_poly, ones) + np.dot(np.dot(ones, K_poly), ones)
K_rbf_centered = K_rbf - np.dot(ones, K_rbf) - np.dot(K_rbf, ones) + np.dot(np.dot(ones, K_rbf), ones)

print("Original data shape:", X.shape)
print("Polynomial kernel matrix shape:", K_poly.shape)
print("RBF kernel matrix shape:", K_rbf.shape)
print("\nFirst 5x5 elements of polynomial kernel matrix:")
print(np.round(K_poly[:5, :5], 3))
```

Slide 13: Implementing Singular Value Decomposition

SVD is a fundamental matrix factorization technique that decomposes a matrix into three components. This implementation demonstrates the practical application of SVD in dimensionality reduction and data compression.

```python
import numpy as np

def manual_svd(X, k=None):
    # Compute eigendecomposition of X^T X
    XtX = X.T @ X
    eigenvalues, eigenvectors = np.linalg.eigh(XtX)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute singular values and right singular vectors
    singular_values = np.sqrt(np.abs(eigenvalues))
    V = eigenvectors
    
    # Compute left singular vectors
    U = np.zeros((X.shape[0], len(singular_values)))
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:  # Numerical stability
            U[:, i] = (X @ V[:, i]) / singular_values[i]
    
    # Truncate to k components if specified
    if k is not None:
        U = U[:, :k]
        singular_values = singular_values[:k]
        V = V[:, :k]
    
    return U, singular_values, V.T

# Example usage with image compression
def compress_image(image_matrix, k):
    U, s, Vt = manual_svd(image_matrix, k)
    compressed = U @ np.diag(s) @ Vt
    return compressed

# Generate sample image-like data
image = np.random.rand(100, 100)

# Compress using different numbers of components
k_values = [5, 10, 20]
for k in k_values:
    compressed = compress_image(image, k)
    compression_ratio = k * (image.shape[0] + image.shape[1]) / (image.shape[0] * image.shape[1])
    error = np.linalg.norm(image - compressed, 'fro') / np.linalg.norm(image, 'fro')
    
    print(f"\nCompression with k={k}:")
    print(f"Compression ratio: {compression_ratio:.2%}")
    print(f"Relative error: {error:.4f}")
```

Slide 14: Gram-Schmidt Process Implementation

The Gram-Schmidt process creates an orthogonal basis from a set of linearly independent vectors. This implementation shows how to construct orthonormal feature representations for machine learning algorithms.

```python
import numpy as np

def gram_schmidt(vectors):
    """
    Implements the Gram-Schmidt orthogonalization process.
    Input: matrix where each column is a vector
    Output: orthonormal basis
    """
    n = vectors.shape[1]
    orthonormal_basis = np.zeros_like(vectors)
    
    for i in range(n):
        # Take the current vector
        v = vectors[:, i].copy()
        
        # Subtract projections onto previous vectors
        for j in range(i):
            proj = np.dot(vectors[:, i], orthonormal_basis[:, j])
            v = v - proj * orthonormal_basis[:, j]
        
        # Normalize the vector
        norm = np.linalg.norm(v)
        if norm > 1e-10:  # Check for numerical stability
            orthonormal_basis[:, i] = v / norm
        else:
            raise ValueError(f"Vector {i} is linearly dependent")
    
    return orthonormal_basis

# Example usage
# Generate random linearly independent vectors
n_dim = 4
n_vectors = 3
vectors = np.random.randn(n_dim, n_vectors)

# Apply Gram-Schmidt process
orthonormal_basis = gram_schmidt(vectors)

# Verify orthonormality
gram_matrix = orthonormal_basis.T @ orthonormal_basis
expected_gram = np.eye(n_vectors)

print("Original vectors shape:", vectors.shape)
print("\nOrthonormal basis shape:", orthonormal_basis.shape)
print("\nGram matrix (should be identity):")
print(np.round(gram_matrix, 6))
print("\nMaximum deviation from identity:", np.max(np.abs(gram_matrix - expected_gram)))
```

Slide 15: Additional Resources

1.  [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366) - "The Matrix Calculus You Need For Deep Learning"
2.  [https://arxiv.org/abs/1802.01528](https://arxiv.org/abs/1802.01528) - "Mathematics of Deep Learning"
3.  [https://arxiv.org/abs/2108.10601](https://arxiv.org/abs/2108.10601) - "A Theoretical Analysis of Feature Learning in Deep Linear Networks"
4.  [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100) - "Deep Learning and the Information Bottleneck Principle"
5.  [https://arxiv.org/abs/2007.12338](https://arxiv.org/abs/2007.12338) - "On the Optimization of Deep Networks: Implicit Acceleration by Overparameterization"

