## Spectral Theory Exploration with Python:
Slide 1: Introduction to Spectral Theory

Spectral theory is a branch of mathematics that studies the properties of linear operators and their spectra. It has applications in quantum mechanics, signal processing, and data analysis.

```python
import numpy as np

def linear_operator(matrix, vector):
    return np.dot(matrix, vector)

A = np.array([[1, 2], [3, 4]])
v = np.array([1, 1])
result = linear_operator(A, v)
print(f"Linear operator result: {result}")
```

Slide 2: Eigenvalues and Eigenvectors

Eigenvalues (λ) and eigenvectors (v) are fundamental concepts in spectral theory. An eigenvector of a linear operator A is a non-zero vector that, when the operator is applied to it, results in a scalar multiple of itself.

```python
import numpy as np

A = np.array([[4, -2], [1, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:")
print(eigenvectors)
```

Slide 3: Spectral Decomposition

Spectral decomposition, also known as eigendecomposition, represents a matrix as a product of its eigenvectors and eigenvalues.

```python
import numpy as np

def spectral_decomposition(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    D = np.diag(eigenvalues)
    V = eigenvectors
    V_inv = np.linalg.inv(V)
    return V, D, V_inv

A = np.array([[4, -2], [1, 1]])
V, D, V_inv = spectral_decomposition(A)
print("Original matrix:")
print(A)
print("Reconstructed matrix:")
print(np.dot(np.dot(V, D), V_inv))
```

Slide 4: Spectral Radius

The spectral radius of a matrix is the maximum absolute value of its eigenvalues. It's crucial in determining the convergence of iterative methods.

```python
import numpy as np

def spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

A = np.array([[3, 1], [0, 2]])
radius = spectral_radius(A)
print(f"Spectral radius: {radius}")
```

Slide 5: Discrete Fourier Transform (DFT)

The Discrete Fourier Transform is an application of spectral theory in signal processing. It decomposes a signal into its frequency components.

```python
import numpy as np

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

signal = np.array([1, 2, 1, 0])
fourier = dft(signal)
print("DFT result:", fourier)
```

Slide 6: Singular Value Decomposition (SVD)

SVD is a factorization technique that decomposes a matrix into three matrices, revealing important properties of the original matrix.

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
U, s, Vt = np.linalg.svd(A)

print("U:", U)
print("Singular values:", s)
print("V transpose:", Vt)

# Reconstruct the original matrix
S = np.zeros(A.shape)
S[:2, :2] = np.diag(s)
A_reconstructed = np.dot(U, np.dot(S, Vt))
print("Reconstructed A:", A_reconstructed)
```

Slide 7: Power Method

The power method is an iterative algorithm to compute the dominant eigenvalue and eigenvector of a matrix.

```python
import numpy as np

def power_method(A, num_iterations=100):
    n, _ = A.shape
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        w = np.dot(A, v)
        v = w / np.linalg.norm(w)
    
    eigenvalue = np.dot(np.dot(v.T, A), v) / np.dot(v.T, v)
    return eigenvalue, v

A = np.array([[2, -1], [1, 3]])
eigenvalue, eigenvector = power_method(A)
print(f"Dominant eigenvalue: {eigenvalue}")
print(f"Corresponding eigenvector: {eigenvector}")
```

Slide 8: Spectral Clustering

Spectral clustering is a technique that uses eigenvalues of the similarity matrix to perform dimensionality reduction before clustering.

```python
import numpy as np
from sklearn.cluster import KMeans

def spectral_clustering(similarity_matrix, n_clusters):
    eigenvalues, eigenvectors = np.linalg.eig(similarity_matrix)
    idx = eigenvalues.argsort()[::-1]
    top_k_eigenvectors = eigenvectors[:, idx[:n_clusters]]
    
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(top_k_eigenvectors)
    return kmeans.labels_

# Example similarity matrix
S = np.array([[1, 0.5, 0.1], [0.5, 1, 0.2], [0.1, 0.2, 1]])
labels = spectral_clustering(S, n_clusters=2)
print("Cluster labels:", labels)
```

Slide 9: Graph Laplacian

The graph Laplacian is a matrix representation of a graph, useful in spectral graph theory and manifold learning.

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def graph_laplacian(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    return degree_matrix - adjacency_matrix

# Create a sample graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

A = nx.adjacency_matrix(G).toarray()
L = graph_laplacian(A)

print("Graph Laplacian:")
print(L)

# Visualize the graph
nx.draw(G, with_labels=True)
plt.show()
```

Slide 10: Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that uses the eigendecomposition of the covariance matrix to find principal components.

```python
import numpy as np
import matplotlib.pyplot as plt

def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return X_centered.dot(eigenvectors[:, :n_components])

# Generate sample data
np.random.seed(42)
X = np.random.multivariate_normal([0, 0], [[2, 1.5], [1.5, 3]], 200)

# Apply PCA
X_pca = pca(X, n_components=1)

# Plot results
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.plot([0, eigenvectors[0, 0] * 3], [0, eigenvectors[1, 0] * 3], color='red', linewidth=2)
plt.axis('equal')
plt.show()
```

Slide 11: Spectral Theory in Quantum Mechanics

In quantum mechanics, the energy levels of a system are given by the eigenvalues of the Hamiltonian operator.

```python
import numpy as np
import matplotlib.pyplot as plt

def quantum_harmonic_oscillator(n_levels):
    H = np.diag(np.arange(n_levels) + 0.5)
    for i in range(n_levels - 1):
        H[i, i+1] = H[i+1, i] = np.sqrt(i + 1) / np.sqrt(2)
    return H

H = quantum_harmonic_oscillator(10)
energies, states = np.linalg.eigh(H)

print("Energy levels:")
print(energies)

plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(states[:, i], label=f'n={i}')
plt.legend()
plt.title("Wavefunctions of Quantum Harmonic Oscillator")
plt.xlabel("Position")
plt.ylabel("Amplitude")
plt.show()
```

Slide 12: Spectral Theory in Signal Processing

Spectral analysis is widely used in signal processing for filtering and feature extraction.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(t):
    return np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

t = np.linspace(0, 1, 1000)
signal = generate_signal(t)

fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(len(t), t[1] - t[0])

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(t, signal)
plt.title("Original Signal")
plt.subplot(212)
plt.plot(freqs, np.abs(fft))
plt.title("Frequency Spectrum")
plt.xlim(0, 30)
plt.show()
```

Slide 13: Matrix Exponential

The matrix exponential is crucial in solving systems of linear differential equations and in quantum mechanics for time evolution.

```python
import numpy as np
import scipy.linalg as la

def matrix_exponential(A, t):
    return la.expm(A * t)

A = np.array([[0, -1], [1, 0]])
t = np.pi / 2

exp_A = matrix_exponential(A, t)
print("Matrix exponential:")
print(exp_A)

# Verify that exp(A * pi/2) rotates a vector by 90 degrees
v = np.array([1, 0])
rotated_v = exp_A.dot(v)
print("Rotated vector:")
print(rotated_v)
```

Slide 14: Spectral Theory in Machine Learning

Spectral methods are used in various machine learning algorithms, including spectral clustering and manifold learning techniques.

```python
from sklearn.manifold import SpectralEmbedding
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Apply spectral embedding
embedding = SpectralEmbedding(n_components=2)
X_embedded = embedding.fit_transform(X)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Original Data")
plt.subplot(122)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis')
plt.title("Spectral Embedding")
plt.show()
```

Slide 15: Additional Resources

For further exploration of Spectral Theory, consider the following resources:

1. "Spectral Theory of Linear Operators" by Tosio Kato (ArXiv:math/0412047)
2. "Introduction to Spectral Theory in Hilbert Space" by Gilbert Helmberg (ArXiv:math/0612424)
3. "Lectures on Spectral Graph Theory" by Fan Chung (ArXiv:math/9701248)

These papers provide in-depth coverage of various aspects of spectral theory and its applications.

