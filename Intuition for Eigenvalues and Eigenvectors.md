## Intuition for Eigenvalues and Eigenvectors
Slide 1: Linear Transformations Fundamentals

Linear transformations represent mappings between vector spaces that preserve vector addition and scalar multiplication. In machine learning, understanding these transformations is crucial as they form the foundation for understanding how neural networks manipulate data through weight matrices.

```python
import numpy as np
import matplotlib.pyplot as plt

def linear_transform(matrix, vector):
    """Applies linear transformation to a vector"""
    return np.dot(matrix, vector)

# Example transformation matrix (rotation + scaling)
A = np.array([[2, -1],
              [1, 2]])

# Original vector
v = np.array([1, 1])

# Apply transformation
transformed_v = linear_transform(A, v)

print(f"Original vector: {v}")
print(f"Transformed vector: {transformed_v}")

# Visualization
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Original')
plt.quiver(0, 0, transformed_v[0], transformed_v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Transformed')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
```

Slide 2: Matrix Operations and Determinants

The determinant of a matrix provides crucial information about the transformation it represents, including whether it preserves, inverts, or collapses space. A zero determinant indicates a transformation that collapses space into a lower dimension.

```python
def calculate_determinant_2d(matrix):
    """
    Calculate determinant and visualize area change for 2D matrix
    Returns determinant and prints geometric interpretation
    """
    det = np.linalg.det(matrix)
    interpretation = "expands" if abs(det) > 1 else "contracts" if 0 < abs(det) < 1 else "preserves" if abs(det) == 1 else "collapses"
    
    print(f"Determinant: {det:.2f}")
    print(f"This transformation {interpretation} space")
    
    return det

# Example matrices
matrices = {
    "Rotation": np.array([[0, -1], [1, 0]]),
    "Scaling": np.array([[2, 0], [0, 2]]),
    "Singular": np.array([[1, 1], [2, 2]])
}

for name, matrix in matrices.items():
    print(f"\n{name} Matrix:")
    calculate_determinant_2d(matrix)
```

Slide 3: Understanding Eigenvalues

Eigenvalues represent the scaling factor applied to vectors during linear transformation along special directions. These directions, represented by eigenvectors, maintain their orientation after transformation, being only scaled by their corresponding eigenvalues.

```python
def find_eigenvalues(matrix):
    """
    Calculate and interpret eigenvalues of a matrix
    """
    eigenvalues = np.linalg.eigvals(matrix)
    
    print("Matrix:")
    print(matrix)
    print("\nEigenvalues:", eigenvalues)
    
    # Characteristic equation in code
    print("\nVerifying characteristic equation det(A - λI) = 0:")
    λ = eigenvalues[0]  # Using first eigenvalue
    I = np.eye(matrix.shape[0])
    det = np.linalg.det(matrix - λ * I)
    print(f"det(A - {λ:.2f}I) = {det:.2e}")

# Example matrix
A = np.array([[3, -2],
              [1, 4]])

find_eigenvalues(A)
```

Slide 4: Computing Eigenvectors

Computing eigenvectors involves solving the homogeneous system (A - λI)v = 0 for each eigenvalue λ. These vectors reveal the principal directions along which the linear transformation acts purely as scaling, fundamental to understanding system behavior.

```python
def compute_eigenvectors(matrix):
    """
    Compute and visualize eigenvectors and their transformations
    """
    eigenvals, eigenvecs = np.linalg.eig(matrix)
    
    print("Matrix A:")
    print(matrix)
    print("\nEigenvalues:", eigenvals)
    print("\nEigenvectors:")
    for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
        print(f"\nEigenvector {i+1}:", vec)
        # Verify Av = λv
        Av = np.dot(matrix, vec)
        λv = val * vec
        print(f"Av = {Av}")
        print(f"λv = {λv}")
        print(f"Verification error: {np.allclose(Av, λv)}")

# Example matrix
A = np.array([[4, -1],
              [2, 1]])

compute_eigenvectors(A)

# Visualization
plt.figure(figsize=(10, 10))
for i, vec in enumerate(eigenvecs.T):
    plt.quiver(0, 0, vec[0], vec[1], angles='xy', scale_units='xy', 
               scale=1, color=['b', 'r'][i], label=f'Eigenvector {i+1}')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.title('Eigenvectors Visualization')
plt.show()
```

Slide 5: Eigendecomposition in Machine Learning

Eigendecomposition is crucial in machine learning for dimensionality reduction, principal component analysis, and understanding covariance matrices. It decomposes a matrix into its fundamental components, revealing the primary directions of variation in data.

```python
def demonstrate_eigendecomposition(matrix):
    """
    Demonstrate matrix eigendecomposition and reconstruction
    A = PDP^(-1) where P contains eigenvectors and D is diagonal matrix of eigenvalues
    """
    eigenvals, eigenvecs = np.linalg.eig(matrix)
    
    # Create diagonal matrix D
    D = np.diag(eigenvals)
    
    # Eigenvector matrix P
    P = eigenvecs
    
    # Compute P^(-1)
    P_inv = np.linalg.inv(P)
    
    # Reconstruct original matrix
    A_reconstructed = P @ D @ P_inv
    
    print("Original Matrix:")
    print(matrix)
    print("\nDiagonal Matrix (D):")
    print(D)
    print("\nEigenvector Matrix (P):")
    print(P)
    print("\nReconstructed Matrix:")
    print(A_reconstructed)
    print("\nReconstruction Error:")
    print(np.allclose(matrix, A_reconstructed))

# Example symmetric matrix
A = np.array([[3, 1],
              [1, 3]])

demonstrate_eigendecomposition(A)
```

Slide 6: Real-world Application: PCA Implementation

Principal Component Analysis uses eigendecomposition to identify the principal directions of variation in data. This implementation demonstrates how eigenvalues and eigenvectors are used to reduce dimensionality while preserving maximum variance.

```python
def implement_pca(X, n_components):
    """
    Implement PCA from scratch using eigendecomposition
    
    Parameters:
    X : array-like of shape (n_samples, n_features)
    n_components : int, number of components to keep
    
    Returns:
    X_transformed : array-like of shape (n_samples, n_components)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Select top n_components
    components = eigenvecs[:, :n_components]
    
    # Transform the data
    X_transformed = X_centered @ components
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvals[:n_components] / np.sum(eigenvals)
    
    return X_transformed, explained_variance_ratio

# Generate example data
np.random.seed(42)
X = np.random.randn(100, 4)  # 100 samples, 4 features

# Apply PCA
X_pca, exp_var_ratio = implement_pca(X, n_components=2)

print("Original data shape:", X.shape)
print("Transformed data shape:", X_pca.shape)
print("Explained variance ratio:", exp_var_ratio)

# Visualize results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('Data in Principal Component Space')
plt.subplot(122)
plt.bar(range(len(exp_var_ratio)), exp_var_ratio)
plt.title('Explained Variance Ratio')
plt.show()
```

Slide 7: Power Iteration Method

The power iteration method is an essential algorithm for computing the dominant eigenvalue and eigenvector of a matrix. This technique is particularly useful in large-scale applications where computing all eigenvalues would be computationally expensive.

```python
def power_iteration(matrix, num_iterations=100, tolerance=1e-10):
    """
    Implements power iteration method to find dominant eigenvalue and eigenvector
    
    Parameters:
        matrix: square numpy array
        num_iterations: maximum number of iterations
        tolerance: convergence threshold
    """
    n = matrix.shape[0]
    
    # Initialize random vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    # Initialize eigenvalue
    eigenvalue = 0
    
    for i in range(num_iterations):
        # Store previous eigenvector for convergence check
        v_prev = v
        
        # Power iteration step
        v_new = np.dot(matrix, v)
        
        # Normalize
        v = v_new / np.linalg.norm(v_new)
        
        # Compute Rayleigh quotient (eigenvalue estimate)
        eigenvalue = np.dot(np.dot(v, matrix), v)
        
        # Check convergence
        if np.allclose(v, v_prev, rtol=tolerance) or np.allclose(v, -v_prev, rtol=tolerance):
            break
    
    return eigenvalue, v

# Example usage
A = np.array([[4, -1],
              [2, 1]])

eigenvalue, eigenvector = power_iteration(A)
print(f"Dominant eigenvalue: {eigenvalue:.6f}")
print(f"Dominant eigenvector: {eigenvector}")

# Verify result
true_eigenvals, true_eigenvecs = np.linalg.eig(A)
print(f"\nTrue eigenvalues: {true_eigenvals}")
print(f"True eigenvectors:\n{true_eigenvecs}")
```

Slide 8: Eigenfaces for Face Recognition

Eigenfaces represent a practical application of eigenvalue decomposition in computer vision. This implementation demonstrates how eigenvectors of face image covariance matrices can be used for dimensionality reduction and face recognition.

```python
def compute_eigenfaces(faces_matrix, n_components=10):
    """
    Compute eigenfaces from a matrix of flattened face images
    
    Parameters:
        faces_matrix: shape (n_samples, n_pixels)
        n_components: number of eigenfaces to compute
    """
    # Center the data
    mean_face = np.mean(faces_matrix, axis=0)
    centered_faces = faces_matrix - mean_face
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered_faces.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenfaces = eigenvecs[:, idx]
    
    # Select top components
    eigenfaces = eigenfaces[:, :n_components]
    
    # Project faces onto eigenface space
    weights = centered_faces @ eigenfaces
    
    return eigenfaces, weights, mean_face

# Generate synthetic face data for demonstration
n_samples, n_features = 100, 400  # 20x20 pixel images
synthetic_faces = np.random.randn(n_samples, n_features)

# Compute eigenfaces
eigenfaces, weights, mean_face = compute_eigenfaces(synthetic_faces, n_components=5)

# Visualize first eigenface
plt.figure(figsize=(8, 8))
plt.imshow(eigenfaces[:, 0].reshape(20, 20), cmap='gray')
plt.title('First Eigenface')
plt.axis('off')
plt.show()

print(f"Original dimension: {n_features}")
print(f"Reduced dimension: {weights.shape[1]}")
```

Slide 9: QR Decomposition for Eigenvalue Computation

QR decomposition provides an iterative method for computing all eigenvalues and eigenvectors of a matrix. This technique is particularly useful for symmetric matrices and forms the basis for many numerical algorithms in machine learning.

```python
def qr_iteration(matrix, max_iter=100, tolerance=1e-10):
    """
    Compute eigenvalues using QR iteration
    
    Parameters:
        matrix: square numpy array
        max_iter: maximum number of iterations
        tolerance: convergence threshold
    """
    A = matrix.copy()
    n = A.shape[0]
    
    # Store eigenvalues
    eigenvals_history = []
    
    for i in range(max_iter):
        # Store previous diagonal for convergence check
        prev_diag = np.diag(A)
        
        # QR decomposition
        Q, R = np.linalg.qr(A)
        
        # Update A
        A = R @ Q
        
        # Get current eigenvalue estimates
        current_eigenvals = np.diag(A)
        eigenvals_history.append(current_eigenvals)
        
        # Check convergence
        if np.allclose(current_eigenvals, prev_diag, rtol=tolerance):
            break
    
    return current_eigenvals, eigenvals_history

# Example usage
A = np.array([[4, -1, 0],
              [-1, 3, -1],
              [0, -1, 4]])

eigenvals, history = qr_iteration(A)
print("Computed eigenvalues:", eigenvals)
print("True eigenvalues:", np.linalg.eigvals(A))

# Visualize convergence
plt.figure(figsize=(10, 6))
history_array = np.array(history)
for i in range(A.shape[0]):
    plt.plot(history_array[:, i], label=f'Eigenvalue {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Eigenvalue Estimate')
plt.legend()
plt.title('QR Iteration Convergence')
plt.grid(True)
plt.show()
```

Slide 10: Spectral Clustering Implementation

Spectral clustering utilizes eigenvalues and eigenvectors of the graph Laplacian matrix to perform dimensionality reduction before clustering. This technique is particularly effective for non-linearly separable data.

```python
def spectral_clustering(X, n_clusters=2):
    """
    Implement spectral clustering from scratch
    
    Parameters:
        X: array-like of shape (n_samples, n_features)
        n_clusters: number of clusters
    """
    # Compute similarity matrix (using RBF kernel)
    gamma = 1.0
    distances = np.square(scipy.spatial.distance.pdist(X)).reshape(-1, 1)
    W = np.exp(-gamma * distances)
    W = scipy.spatial.distance.squareform(W)
    
    # Compute degree matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Compute normalized Laplacian
    L = D - W
    L_norm = np.linalg.inv(np.sqrt(D)) @ L @ np.linalg.inv(np.sqrt(D))
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(L_norm)
    
    # Select k smallest eigenvalues (excluding zero)
    idx = np.argsort(eigenvals)[1:n_clusters+1]
    k_smallest_eigenvecs = eigenvecs[:, idx]
    
    # Normalize rows
    k_smallest_eigenvecs = k_smallest_eigenvecs / np.linalg.norm(k_smallest_eigenvecs, axis=1, keepdims=True)
    
    # Cluster using k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(k_smallest_eigenvecs)
    
    return clusters

# Generate example data
np.random.seed(42)
n_samples = 300
X = np.concatenate([
    np.random.randn(n_samples//2, 2) * 0.5 + np.array([2, 2]),
    np.random.randn(n_samples//2, 2) * 0.5 + np.array([-2, -2])
])

# Apply spectral clustering
clusters = spectral_clustering(X, n_clusters=2)

# Visualize results
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title('Spectral Clustering Results')
plt.colorbar()
plt.show()
```

Slide 11: SVD and Eigenvalue Relationship

Singular Value Decomposition (SVD) establishes a fundamental connection with eigenvalues through the relationship between singular values and eigenvalues. Understanding this connection is crucial for various machine learning applications.

```python
def demonstrate_svd_eigenvalue_relationship(matrix):
    """
    Demonstrate the relationship between SVD and eigenvalues
    
    Parameters:
        matrix: numpy array of shape (m, n)
    """
    # Compute SVD
    U, s, Vh = np.linalg.svd(matrix)
    
    # Compute eigenvalues of A^T A and AA^T
    ATA = matrix.T @ matrix
    AAT = matrix @ matrix.T
    
    eigenvals_ATA = np.linalg.eigvals(ATA)
    eigenvals_AAT = np.linalg.eigvals(AAT)
    
    # Sort eigenvalues
    eigenvals_ATA = np.sort(eigenvals_ATA)[::-1]
    eigenvals_AAT = np.sort(eigenvals_AAT)[::-1]
    
    print("Singular values:", s)
    print("\nSquared singular values:", s**2)
    print("\nEigenvalues of A^T A:", eigenvals_ATA.real)
    print("\nEigenvalues of AA^T:", eigenvals_AAT.real)
    
    # Verify relationship: singular values are sqrt of eigenvalues
    print("\nVerification: singular values ≈ sqrt(eigenvalues of A^T A)")
    print("Maximum difference:", np.max(np.abs(s - np.sqrt(eigenvals_ATA.real))))

# Example usage
A = np.array([[1, 2, 0],
              [2, 3, 4],
              [0, 4, 5]])

demonstrate_svd_eigenvalue_relationship(A)
```

Slide 12: Eigenvalue Perturbation Analysis

Eigenvalue perturbation analysis is crucial for understanding the stability of machine learning algorithms. This implementation demonstrates how small changes in a matrix affect its eigenvalues and eigenvectors.

```python
def analyze_eigenvalue_perturbation(matrix, perturbation_scale=0.1):
    """
    Analyze the effect of perturbations on eigenvalues
    
    Parameters:
        matrix: original matrix
        perturbation_scale: scale of random perturbation
    """
    # Original eigenvalues
    orig_eigenvals = np.linalg.eigvals(matrix)
    
    # Generate multiple perturbations
    n_perturbations = 100
    perturbed_eigenvals = []
    
    for i in range(n_perturbations):
        # Random perturbation
        perturbation = np.random.randn(*matrix.shape) * perturbation_scale
        perturbed_matrix = matrix + perturbation
        
        # Compute new eigenvalues
        new_eigenvals = np.linalg.eigvals(perturbed_matrix)
        perturbed_eigenvals.append(new_eigenvals)
    
    perturbed_eigenvals = np.array(perturbed_eigenvals)
    
    # Analyze changes
    mean_change = np.mean(np.abs(perturbed_eigenvals - orig_eigenvals), axis=0)
    std_change = np.std(np.abs(perturbed_eigenvals - orig_eigenvals), axis=0)
    
    return orig_eigenvals, mean_change, std_change, perturbed_eigenvals

# Example usage
A = np.array([[3, -1],
              [-1, 3]])

orig_vals, mean_ch, std_ch, pert_vals = analyze_eigenvalue_perturbation(A)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(pert_vals[:, 0].real, bins=30, alpha=0.5, label='First eigenvalue')
plt.hist(pert_vals[:, 1].real, bins=30, alpha=0.5, label='Second eigenvalue')
plt.axvline(orig_vals[0].real, color='r', linestyle='--')
plt.axvline(orig_vals[1].real, color='g', linestyle='--')
plt.legend()
plt.title('Distribution of Perturbed Eigenvalues')

plt.subplot(122)
plt.errorbar(range(len(mean_ch)), mean_ch, yerr=std_ch, fmt='o')
plt.title('Mean Change in Eigenvalues with Error Bars')
plt.show()
```

Slide 13: Tensor Eigenvalues in Deep Learning

Tensor eigenvalues extend classical matrix eigenvalue theory to higher-order tensors, crucial for understanding deep neural network behavior. This implementation demonstrates computation and analysis of tensor eigenvalues using power iteration.

```python
def tensor_eigenvalue_computation(tensor, max_iter=100, tolerance=1e-6):
    """
    Compute dominant eigenvalue of a symmetric 3rd-order tensor
    using power iteration method
    
    Parameters:
        tensor: 3D numpy array of shape (n, n, n)
        max_iter: maximum iterations
        tolerance: convergence threshold
    """
    n = tensor.shape[0]
    
    # Initialize random vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    eigenvalue = 0
    for i in range(max_iter):
        v_old = v.copy()
        
        # Tensor contraction
        Tv = np.zeros(n)
        for j in range(n):
            for k in range(n):
                Tv += tensor[:, j, k] * v[j] * v[k]
        
        # Update eigenvector estimate
        v = Tv / np.linalg.norm(Tv)
        
        # Update eigenvalue estimate
        eigenvalue = np.sum(tensor * v.reshape(-1, 1, 1) * 
                          v.reshape(1, -1, 1) * v.reshape(1, 1, -1))
        
        # Check convergence
        if np.linalg.norm(v - v_old) < tolerance:
            break
            
    return eigenvalue, v

# Example usage with a small symmetric tensor
n = 3
tensor = np.random.rand(n, n, n)
# Make tensor symmetric
tensor = (tensor + tensor.transpose(1, 0, 2) + 
         tensor.transpose(2, 1, 0)) / 3

eigenvalue, eigenvector = tensor_eigenvalue_computation(tensor)

print("Tensor shape:", tensor.shape)
print("Dominant eigenvalue:", eigenvalue)
print("Corresponding eigenvector:", eigenvector)

# Verify tensor eigenvalue equation
Tv = np.zeros(n)
for j in range(n):
    for k in range(n):
        Tv += tensor[:, j, k] * eigenvector[j] * eigenvector[k]
print("\nVerification:")
print("Tv:", Tv)
print("λv:", eigenvalue * eigenvector)
print("Relative error:", np.linalg.norm(Tv - eigenvalue * eigenvector))
```

Slide 14: Eigenvalue Applications in Graph Neural Networks

Graph Neural Networks utilize eigenvalues of the graph Laplacian to perform spectral convolutions. This implementation demonstrates how eigendecomposition aids in graph signal processing and feature learning.

```python
def graph_convolution_layer(adjacency_matrix, node_features, n_filters=16):
    """
    Implement graph convolution using eigendecomposition
    
    Parameters:
        adjacency_matrix: normalized adjacency matrix
        node_features: node feature matrix
        n_filters: number of output filters
    """
    # Compute normalized Laplacian
    D = np.diag(np.sum(adjacency_matrix, axis=1))
    L = D - adjacency_matrix
    L_norm = np.linalg.inv(np.sqrt(D)) @ L @ np.linalg.inv(np.sqrt(D))
    
    # Eigendecomposition of Laplacian
    eigenvals, eigenvecs = np.linalg.eigh(L_norm)
    
    # Initialize random filter parameters
    theta = np.random.randn(len(eigenvals), n_filters)
    
    # Spectral convolution
    def chebyshev_polynomial(x, k):
        if k == 0:
            return 1
        elif k == 1:
            return x
        return 2 * x * chebyshev_polynomial(x, k-1) - chebyshev_polynomial(x, k-2)
    
    # Apply spectral filters
    conv_output = []
    for k in range(n_filters):
        spectral_filter = np.zeros_like(eigenvals)
        for i in range(len(eigenvals)):
            spectral_filter += theta[i, k] * chebyshev_polynomial(eigenvals, i)
            
        filtered_signal = eigenvecs @ np.diag(spectral_filter) @ eigenvecs.T @ node_features
        conv_output.append(filtered_signal)
    
    return np.stack(conv_output, axis=-1)

# Example usage
n_nodes = 20
n_features = 8

# Generate random graph
adjacency = np.random.randint(0, 2, (n_nodes, n_nodes))
adjacency = (adjacency + adjacency.T) // 2
np.fill_diagonal(adjacency, 1)

# Generate random node features
features = np.random.randn(n_nodes, n_features)

# Apply graph convolution
output = graph_convolution_layer(adjacency, features)

print("Input shape:", features.shape)
print("Output shape:", output.shape)

# Visualize first filter response
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(features, aspect='auto')
plt.title('Input Features')
plt.colorbar()
plt.subplot(122)
plt.imshow(output[:, :, 0], aspect='auto')
plt.title('First Filter Response')
plt.colorbar()
plt.show()
```

Slide 15: Additional Resources

*   Eigenvalue Algorithms in Recommender Systems
    *   [https://arxiv.org/abs/1407.5107](https://arxiv.org/abs/1407.5107)
*   Deep Learning and the Information Bottleneck Principle
    *   [https://arxiv.org/abs/1503.02406](https://arxiv.org/abs/1503.02406)
*   Spectral Networks and Deep Locally Connected Networks on Graphs
    *   [https://arxiv.org/abs/1312.6203](https://arxiv.org/abs/1312.6203)
*   On the Stability of Deep Neural Networks
    *   [https://arxiv.org/abs/1412.6557](https://arxiv.org/abs/1412.6557)
*   Tutorial on Spectral Clustering and Modern Applications
    *   [https://www.google.com/search?q=tutorial+spectral+clustering+modern+applications](https://www.google.com/search?q=tutorial+spectral+clustering+modern+applications)

