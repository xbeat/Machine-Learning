## Linear Algebra Fundamentals for Machine Learning
Slide 1: Vector Operations Fundamentals

Linear algebra operations form the backbone of machine learning algorithms. Understanding vector operations is crucial for implementing efficient ML models, particularly in neural networks and optimization algorithms.

```python
import numpy as np

# Define two vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
addition = v1 + v2  # Output: [5 7 9]

# Scalar multiplication
scalar_mult = 2 * v1  # Output: [2 4 6]

# Dot product
dot_product = np.dot(v1, v2)  # Output: 32

# Vector magnitude (L2 norm)
magnitude = np.linalg.norm(v1)  # Output: 3.7416573867739413
```

Slide 2: Matrix Operations Implementation

Matrix operations are essential for data transformation and feature engineering in ML pipelines. Understanding these operations helps in implementing neural network layers and optimization algorithms effectively.

```python
import numpy as np

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print(f"Matrix multiplication:\n{C}")

# Matrix transpose
A_transpose = A.T
print(f"Transpose:\n{A_transpose}")

# Matrix inverse
A_inv = np.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# Determinant
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")
```

Slide 3: Eigenvalues and Eigenvectors

Eigendecomposition is fundamental in dimensionality reduction techniques like PCA. Understanding eigenvectors helps in identifying principal components and feature importance in data analysis.

```python
import numpy as np

# Create a symmetric matrix
A = np.array([[4, -2], 
              [-2, 3]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Verify eigenvalue equation: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"\nVerification for eigenvalue {lam}:")
    print("Av =", np.dot(A, v))
    print("λv =", lam * v)
```

Slide 4: Principal Component Analysis Implementation

PCA is a crucial dimensionality reduction technique in ML. This implementation shows how to use eigendecomposition for feature extraction and data visualization.

```python
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Center the data
X_centered = X - np.mean(X, axis=0)

# Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project data onto first two principal components
X_pca = X_centered.dot(eigenvectors[:, :2])

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Iris Dataset - First Two Principal Components')
```

Slide 5: Singular Value Decomposition (SVD)

SVD decomposes a matrix into three matrices, making it valuable for dimensionality reduction and matrix approximation in ML applications. It's particularly useful in recommendation systems and image compression.

```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Perform SVD
U, s, VT = np.linalg.svd(A)

# Reconstruct original matrix
S = np.zeros_like(A, dtype=float)
np.fill_diagonal(S, s)
A_reconstructed = U.dot(S).dot(VT)

print("Original matrix:\n", A)
print("\nSingular values:", s)
print("\nReconstructed matrix:\n", A_reconstructed)
```

Slide 6: Matrix Factorization for Recommender Systems

Matrix factorization techniques are fundamental in building recommender systems. This implementation demonstrates how to decompose a user-item interaction matrix for making predictions.

```python
import numpy as np

# Create user-item rating matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4]
])

class MatrixFactorization:
    def __init__(self, R, K, alpha=0.001, beta=0.02, iterations=100):
        self.R = R
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.P = np.random.normal(scale=1./self.K, size=(R.shape[0], K))
        self.Q = np.random.normal(scale=1./self.K, size=(R.shape[1], K))
    
    def train(self):
        for _ in range(self.iterations):
            for i in range(self.R.shape[0]):
                for j in range(self.R.shape[1]):
                    if self.R[i, j] > 0:
                        eij = self.R[i, j] - np.dot(self.P[i,:], self.Q[j,:].T)
                        self.P[i,:] += self.alpha * (2 * eij * self.Q[j,:] - self.beta * self.P[i,:])
                        self.Q[j,:] += self.alpha * (2 * eij * self.P[i,:] - self.beta * self.Q[j,:])
    
    def predict(self):
        return np.dot(self.P, self.Q.T)

# Train model
mf = MatrixFactorization(ratings, K=2)
mf.train()
predicted_ratings = mf.predict()
print("Predicted ratings:\n", predicted_ratings)
```

Slide 7: Linear Transformations Visualization

Understanding linear transformations is crucial for neural network architecture design and feature engineering. This implementation visualizes common linear transformations in 2D space.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_transformation(A):
    # Create grid of points
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    
    # Stack coordinates
    coords = np.stack([X.flatten(), Y.flatten()])
    
    # Apply transformation
    transformed = A @ coords
    
    # Plot original and transformed points
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.scatter(coords[0], coords[1], c='blue', alpha=0.5)
    plt.grid(True)
    plt.title('Original Space')
    
    plt.subplot(122)
    plt.scatter(transformed[0], transformed[1], c='red', alpha=0.5)
    plt.grid(True)
    plt.title('Transformed Space')
    
    plt.tight_layout()

# Example transformations
rotation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                    [np.sin(np.pi/4), np.cos(np.pi/4)]])
scaling = np.array([[2, 0],
                   [0, 0.5]])

plot_transformation(rotation)
plt.suptitle('Rotation Transformation')
plt.show()

plot_transformation(scaling)
plt.suptitle('Scaling Transformation')
plt.show()
```

Slide 8: Gram-Schmidt Orthogonalization

Gram-Schmidt process creates an orthogonal basis from a set of linearly independent vectors, crucial for QR decomposition and solving linear systems in numerical optimization.

```python
import numpy as np

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v.copy()
        for u in basis:
            w = w - np.dot(v, u) / np.dot(u, u) * u
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)

# Example vectors
vectors = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
])

orthonormal_basis = gram_schmidt(vectors)
print("Orthonormal basis:\n", orthonormal_basis)

# Verify orthogonality
print("\nVerification (should be identity matrix):")
print(orthonormal_basis @ orthonormal_basis.T)
```

Slide 9: Linear Regression Using Matrix Operations

Implementing linear regression using matrix operations demonstrates the practical application of linear algebra in machine learning for solving optimization problems.

```python
import numpy as np

class LinearRegressionMatrix:
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones(X.shape[0]), X]
        # Calculate weights using normal equation
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Train model
model = LinearRegressionMatrix()
model.fit(X, y)

print("Weights:", model.weights.flatten())
print("True coefficients: [4, 3]")

# Calculate R-squared
y_pred = model.predict(X)
r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
print(f"R-squared: {r2}")
```

Slide 10: QR Decomposition Implementation

QR decomposition is essential for solving linear systems and eigenvalue problems. This implementation shows how to decompose a matrix into orthogonal and upper triangular components.

```python
import numpy as np

def qr_decomposition(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R

# Example matrix
A = np.array([[12, -51, 4],
              [6, 167, -68],
              [-4, 24, -41]])

Q, R = qr_decomposition(A)

print("Q matrix:\n", Q)
print("\nR matrix:\n", R)
print("\nVerification (should be close to original):\n", Q @ R)
print("\nOriginal matrix:\n", A)
```

Slide 11: Neural Network Layer Implementation

Linear algebra operations form the foundation of neural network layers. This implementation demonstrates forward and backward propagation using matrix operations for a simple dense layer.

```python
import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias
    
    def backward(self, grad_output, learning_rate=0.01):
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_output, self.weights.T)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_inputs

# Example usage
layer = DenseLayer(3, 2)
inputs = np.random.randn(4, 3)
outputs = layer.forward(inputs)
grad = np.random.randn(*outputs.shape)
gradients = layer.backward(grad)

print("Input shape:", inputs.shape)
print("Output shape:", outputs.shape)
print("Gradient shape:", gradients.shape)
```

Slide 12: Eigenface Implementation for Face Recognition

Eigenfaces demonstrate practical application of eigendecomposition in computer vision and face recognition systems using PCA concepts.

```python
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
n_samples, n_features = X.shape

# Center the data
X_centered = X - X.mean(axis=0)

# Compute PCA
n_components = 150
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_centered)

# Reconstruct faces
X_reconstructed = pca.inverse_transform(X_pca)

def plot_eigenfaces(pca, n_eigenfaces=5):
    eigenfaces = pca.components_[:n_eigenfaces]
    eigenvalues = pca.explained_variance_[:n_eigenfaces]
    
    for i, (eigenface, eigenvalue) in enumerate(zip(eigenfaces, eigenvalues)):
        print(f"Eigenface {i+1}, Explained variance: {eigenvalue:.2f}")
        print("Shape:", eigenface.shape)

plot_eigenfaces(pca)
print(f"\nTotal variance explained: {pca.explained_variance_ratio_.sum():.3f}")
```

Slide 13: Additional Resources

*   Foundations of Linear Algebra in Machine Learning [https://arxiv.org/abs/2008.01891](https://arxiv.org/abs/2008.01891)
*   Efficient Implementation of Matrix Operations in Deep Learning [https://arxiv.org/abs/1906.01911](https://arxiv.org/abs/1906.01911)
*   Applications of SVD in Recommendation Systems [https://arxiv.org/abs/1907.06178](https://arxiv.org/abs/1907.06178)
*   Matrix Calculus for Deep Learning [https://www.google.com/search?q=matrix+calculus+deep+learning](https://www.google.com/search?q=matrix+calculus+deep+learning)
*   Linear Algebra and Optimization for Machine Learning [https://www.google.com/search?q=linear+algebra+optimization+machine+learning](https://www.google.com/search?q=linear+algebra+optimization+machine+learning)

