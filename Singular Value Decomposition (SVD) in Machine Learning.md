## Singular Value Decomposition (SVD) in Machine Learning
Slide 1: Introduction to Singular Value Decomposition (SVD)

Singular Value Decomposition is a fundamental technique in linear algebra with widespread applications in machine learning and AI. It decomposes a matrix into three matrices, revealing important properties of the original matrix. This decomposition is crucial for dimensionality reduction, feature extraction, and noise reduction in various ML tasks.

```python
import numpy as np

# Create a sample matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Perform SVD
U, S, Vt = np.linalg.svd(A)

print("Original matrix A:")
print(A)
print("\nLeft singular vectors (U):")
print(U)
print("\nSingular values (S):")
print(S)
print("\nRight singular vectors transposed (Vt):")
print(Vt)
```

Slide 2: Mathematical Foundations of SVD

SVD decomposes a matrix A into the product of three matrices: A = USV^T, where U and V are orthogonal matrices, and S is a diagonal matrix containing singular values. This decomposition reveals the matrix's rank, nullspace, and range, which are essential for understanding its properties and behavior in various applications.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 2D matrix for visualization
A = np.array([[3, 2], [2, 3]])

# Perform SVD
U, S, Vt = np.linalg.svd(A)

# Visualize the transformation
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)
xy = np.column_stack([X.ravel(), Y.ravel()])

transformed = xy.dot(A)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(xy[:, 0], xy[:, 1], c='b', alpha=0.5)
plt.title("Original Space")
plt.subplot(122)
plt.scatter(transformed[:, 0], transformed[:, 1], c='r', alpha=0.5)
plt.title("Transformed Space")
plt.tight_layout()
plt.show()
```

Slide 3: SVD for Dimensionality Reduction

One of the most powerful applications of SVD is dimensionality reduction. By selecting the top k singular values and their corresponding singular vectors, we can create a low-rank approximation of the original matrix. This technique is the foundation of Principal Component Analysis (PCA) and is widely used in data compression and feature selection.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = 0.5 * X[:, 0] + X[:, 1] * 0.1

# Perform SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Project data onto first principal component
X_reduced = X.dot(Vt.T[:, :1])

# Plot original and reduced data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original 2D Data")
plt.subplot(122)
plt.scatter(X_reduced, np.zeros_like(X_reduced))
plt.title("Reduced 1D Data")
plt.tight_layout()
plt.show()
```

Slide 4: SVD for Image Compression

Image compression is a practical application of SVD. By keeping only the most significant singular values and their corresponding vectors, we can reconstruct an approximation of the original image with reduced file size. This technique is particularly useful for grayscale images, where each pixel is represented by a single value.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load sample image
image = data.camera()

# Perform SVD
U, S, Vt = np.linalg.svd(image, full_matrices=False)

# Function to reconstruct image with k components
def reconstruct(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Reconstruct image with different numbers of components
k_values = [5, 20, 50]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original")
for i, k in enumerate(k_values):
    row, col = (i + 1) // 2, (i + 1) % 2
    reconstructed = reconstruct(U, S, Vt, k)
    axes[row, col].imshow(reconstructed, cmap='gray')
    axes[row, col].set_title(f"k = {k}")
plt.tight_layout()
plt.show()
```

Slide 5: SVD for Collaborative Filtering

Collaborative filtering is a popular technique in recommendation systems. SVD can be used to factorize the user-item interaction matrix, revealing latent features that explain user preferences and item characteristics. This low-rank approximation helps in predicting user ratings for unseen items.

```python
import numpy as np
import pandas as pd

# Create a sample user-item rating matrix
ratings = np.array([
    [4, 3, 0, 5, 0],
    [5, 0, 4, 0, 2],
    [3, 1, 2, 4, 1],
    [0, 0, 0, 2, 0],
    [1, 0, 3, 4, 5]
])

# Perform SVD
U, S, Vt = np.linalg.svd(ratings)

# Choose number of latent factors
k = 2

# Reconstruct the rating matrix with k factors
reconstructed_ratings = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

print("Original ratings:")
print(pd.DataFrame(ratings))
print("\nReconstructed ratings:")
print(pd.DataFrame(reconstructed_ratings))
```

Slide 6: SVD for Text Analysis and Topic Modeling

In natural language processing, SVD is used for text analysis and topic modeling. By applying SVD to a term-document matrix, we can uncover latent semantic structures in the text. This technique, known as Latent Semantic Analysis (LSA), is useful for document clustering, information retrieval, and identifying related terms.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Sample documents
documents = [
    "The cat and the dog",
    "The dog chased the cat",
    "The bird flew over the cat and the dog"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Perform SVD (LSA)
svd = TruncatedSVD(n_components=2)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Print results
print("TF-IDF Matrix:")
print(tfidf_matrix.toarray())
print("\nLSA Matrix:")
print(lsa_matrix)
print("\nTop terms for each topic:")
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(svd.components_):
    top_terms = terms[comp.argsort()[-3:][::-1]]
    print(f"Topic {i + 1}: {', '.join(top_terms)}")
```

Slide 7: SVD for Noise Reduction in Signals

SVD can be effectively used for noise reduction in signals. By decomposing a noisy signal and reconstructing it using only the most significant singular values, we can filter out high-frequency noise while preserving the important features of the original signal.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a noisy signal
t = np.linspace(0, 10, 1000)
clean_signal = np.sin(t) + 0.5 * np.sin(3 * t)
noise = np.random.normal(0, 0.2, t.shape)
noisy_signal = clean_signal + noise

# Construct Hankel matrix from the signal
N = len(noisy_signal)
L = N // 2
H = np.array([noisy_signal[i:i+L] for i in range(N-L+1)])

# Perform SVD
U, S, Vt = np.linalg.svd(H, full_matrices=False)

# Reconstruct signal using top k singular values
k = 2
S_filtered = np.diag(S[:k])
H_filtered = U[:, :k] @ S_filtered @ Vt[:k, :]
filtered_signal = np.array([np.mean(H_filtered.diagonal(i)) for i in range(-H_filtered.shape[0]+1, H_filtered.shape[1])])

# Plot results
plt.figure(figsize=(12, 8))
plt.plot(t, clean_signal, label='Clean Signal')
plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.5)
plt.plot(t, filtered_signal, label='Filtered Signal', linewidth=2)
plt.legend()
plt.title('SVD-based Noise Reduction')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()
```

Slide 8: SVD for Matrix Completion

Matrix completion is the task of filling in missing entries in a partially observed matrix. SVD plays a crucial role in this process by finding a low-rank approximation of the incomplete matrix. This technique is widely used in collaborative filtering and data imputation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a low-rank matrix
np.random.seed(42)
A = np.random.rand(10, 5) @ np.random.rand(5, 10)

# Add missing values
mask = np.random.rand(*A.shape) < 0.3
A_incomplete = A.()
A_incomplete[mask] = np.nan

# Function to perform matrix completion
def complete_matrix(X, rank, max_iter=100, tol=1e-5):
    mask = ~np.isnan(X)
    X_filled = np.where(mask, X, 0)
    
    for _ in range(max_iter):
        U, S, Vt = np.linalg.svd(X_filled, full_matrices=False)
        X_low_rank = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        X_new = np.where(mask, X, X_low_rank)
        
        if np.linalg.norm(X_new - X_filled) < tol:
            break
        X_filled = X_new
    
    return X_filled

# Complete the matrix
A_completed = complete_matrix(A_incomplete, rank=5)

# Visualize results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(A, cmap='viridis')
ax1.set_title('Original Matrix')
ax2.imshow(A_incomplete, cmap='viridis')
ax2.set_title('Incomplete Matrix')
ax3.imshow(A_completed, cmap='viridis')
ax3.set_title('Completed Matrix')
plt.tight_layout()
plt.show()

print(f"Relative error: {np.linalg.norm(A - A_completed) / np.linalg.norm(A):.4f}")
```

Slide 9: SVD for Solving Linear Systems

SVD can be used to solve systems of linear equations, especially when the coefficient matrix is ill-conditioned or singular. This technique, known as the pseudoinverse method, provides a numerically stable solution even when traditional methods like Gaussian elimination fail.

```python
import numpy as np

# Create a system of linear equations: Ax = b
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([14, 32, 50])

# Compute the SVD of A
U, S, Vt = np.linalg.svd(A)

# Compute the pseudoinverse of A
S_inv = np.zeros_like(A, dtype=float)
S_inv[:S.shape[0], :S.shape[0]] = np.diag(1 / S)
A_pseudo = Vt.T @ S_inv.T @ U.T

# Solve the system using the pseudoinverse
x = A_pseudo @ b

print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nSolution x:")
print(x)
print("\nVerification (Ax):")
print(A @ x)
```

Slide 10: SVD for Principal Component Analysis (PCA)

Principal Component Analysis is a widely used technique for dimensionality reduction and feature extraction. SVD provides an efficient way to compute the principal components of a dataset, allowing us to identify the most important directions of variation in the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Center the data
X_centered = X - X.mean(axis=0)

# Perform SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Project data onto first two principal components
X_pca = X_centered @ Vt.T[:, :2]

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original Data")
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Projection")

# Plot principal components
for i, (comp, var) in enumerate(zip(Vt[:2], S[:2])):
    plt.arrow(0, 0, comp[0]*var, comp[1]*var, color='r', alpha=0.8, 
              width=0.03, head_width=0.1, head_length=0.1)
    plt.text(comp[0]*var*1.2, comp[1]*var*1.2, f"PC{i+1}", color='r')

plt.tight_layout()
plt.show()

print("Explained variance ratio:")
print(S**2 / np.sum(S**2))
```

Slide 11: SVD for Image Denoising

Image denoising is another important application of SVD in image processing. By decomposing a noisy image and reconstructing it using only the most significant singular values, we can effectively remove noise while preserving important image features.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util

# Load and add noise to the image
image = data.camera()
noisy_image = util.random_noise(image, mode='gaussian', var=0.05)

# Perform SVD on the noisy image
U, S, Vt = np.linalg.svd(noisy_image, full_matrices=False)

# Function to reconstruct image with k components
def denoise(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Denoise image with different numbers of components
k_values = [10, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(noisy_image, cmap='gray')
axes[0, 0].set_title("Noisy Image")
for i, k in enumerate(k_values):
    row,
```

## Response:
Slide 11: SVD for Image Denoising

Image denoising is another important application of SVD in image processing. By decomposing a noisy image and reconstructing it using only the most significant singular values, we can effectively remove noise while preserving important image features.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util

# Load and add noise to the image
image = data.camera()
noisy_image = util.random_noise(image, mode='gaussian', var=0.05)

# Perform SVD on the noisy image
U, S, Vt = np.linalg.svd(noisy_image, full_matrices=False)

# Function to reconstruct image with k components
def denoise(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Denoise image with different numbers of components
k_values = [10, 50, 100]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(noisy_image, cmap='gray')
axes[0, 0].set_title("Noisy Image")
for i, k in enumerate(k_values):
    row, col = (i + 1) // 2, (i + 1) % 2
    denoised = denoise(U, S, Vt, k)
    axes[row, col].imshow(denoised, cmap='gray')
    axes[row, col].set_title(f"Denoised (k = {k})")
plt.tight_layout()
plt.show()
```

Slide 12: SVD for Data Compression

SVD is widely used in data compression, allowing us to represent high-dimensional data with fewer dimensions while preserving its essential structure. This technique is particularly useful in areas such as image and audio processing, where large amounts of data need to be stored or transmitted efficiently.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample 2D dataset
np.random.seed(42)
X = np.random.randn(1000, 2)
X[:, 1] = 3 * X[:, 0] + 2 * X[:, 1]

# Perform SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Compress data by keeping only the first singular value
k = 1
X_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Calculate compression ratio
original_size = X.size * X.itemsize
compressed_size = (U[:, :k].size + S[:k].size + Vt[:k, :].size) * X.itemsize
compression_ratio = original_size / compressed_size

# Plot original and compressed data
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Original Data")
plt.subplot(122)
plt.scatter(X_compressed[:, 0], X_compressed[:, 1], alpha=0.5)
plt.title(f"Compressed Data (Ratio: {compression_ratio:.2f})")
plt.tight_layout()
plt.show()
```

Slide 13: SVD for Anomaly Detection

SVD can be employed for anomaly detection in multivariate data. By projecting data onto the subspace defined by the principal components, we can identify data points that deviate significantly from the norm. This technique is useful in various domains, including network security and fault detection in industrial systems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate sample data with outliers
X, _ = make_blobs(n_samples=300, centers=1, random_state=42)
X = np.vstack([X, np.array([[10, 10], [-8, -8]])])  # Add outliers

# Perform SVD
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)

# Project data onto first two principal components
X_proj = (X - X.mean(axis=0)) @ Vt.T[:, :2]

# Calculate reconstruction error
X_recon = X_proj @ Vt[:2, :] + X.mean(axis=0)
recon_error = np.sum((X - X_recon) ** 2, axis=1)

# Identify anomalies (points with high reconstruction error)
threshold = np.percentile(recon_error, 97.5)
anomalies = recon_error > threshold

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=anomalies, cmap='coolwarm')
plt.title("Original Data")
plt.subplot(122)
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=anomalies, cmap='coolwarm')
plt.title("Projected Data")
plt.colorbar(label='Anomaly')
plt.tight_layout()
plt.show()
```

Slide 14: Real-life Example: Document Similarity

SVD can be used to measure document similarity in natural language processing tasks. By applying SVD to a term-document matrix, we can represent documents in a lower-dimensional space and compute their similarity using cosine similarity.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast red fox leaps above a sleeping hound",
    "Python is a popular programming language",
    "Machine learning is a subset of artificial intelligence"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Perform SVD
U, S, Vt = np.linalg.svd(tfidf_matrix.toarray(), full_matrices=False)

# Choose number of dimensions for low-rank approximation
k = 2

# Project documents into low-dimensional space
doc_vectors = U[:, :k] @ np.diag(S[:k])

# Compute pairwise cosine similarities
similarities = cosine_similarity(doc_vectors)

print("Document Similarity Matrix:")
print(similarities)

# Find most similar pair of documents
max_sim = np.max(similarities - np.eye(len(documents)))
max_idx = np.unravel_index(np.argmax(similarities - np.eye(len(documents))), similarities.shape)

print(f"\nMost similar documents: {max_idx[0]} and {max_idx[1]}")
print(f"Similarity score: {max_sim:.4f}")
```

Slide 15: Real-life Example: Image Recognition

SVD plays a crucial role in various image recognition tasks, including facial recognition. By applying SVD to a dataset of face images, we can extract the most important features (eigenfaces) and use them for classification or recognition tasks.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
n_samples, n_features = X.shape

# Perform SVD
U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)

# Plot first 16 eigenfaces
n_components = 16
eigenfaces = Vt[:n_components].reshape((n_components, 50, 37))

fig, axes = plt.subplots(4, 4, figsize=(10, 10),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f"Eigenface {i+1}")

plt.tight_layout()
plt.show()

# Project a sample face onto eigenface space
sample_face = X[0] - X.mean(axis=0)
weights = sample_face @ Vt[:n_components].T

# Reconstruct the face using different numbers of components
k_values = [5, 10, 50, 100]
fig, axes = plt.subplots(1, len(k_values)+1, figsize=(15, 3),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

axes[0].imshow(X[0].reshape(50, 37), cmap='gray')
axes[0].set_title("Original")

for i, k in enumerate(k_values):
    reconstructed = (weights[:k] @ Vt[:k, :]) + X.mean(axis=0)
    axes[i+1].imshow(reconstructed.reshape(50, 37), cmap='gray')
    axes[i+1].set_title(f"k = {k}")

plt.tight_layout()
plt.show()
```

Slide 16: Additional Resources

For those interested in diving deeper into Singular Value Decomposition and its applications in machine learning and AI, here are some valuable resources:

1. "Matrix Computations" by Gene H. Golub and Charles F. Van Loan - A comprehensive reference on matrix algorithms, including SVD.
2. "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III - Provides an in-depth treatment of SVD and related techniques.
3. ArXiv paper: "Singular Value Decomposition and Principal Component Analysis" by Jonathon Shlens URL: [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
4. ArXiv paper: "A Survey of Matrix Factorization Techniques for Recommendation Systems" by Yehuda Koren, Robert Bell, and Chris Volinsky URL: [https://arxiv.org/abs/0911.3421](https://arxiv.org/abs/0911.3421)
5. Online course: "Singular Value Decomposition (SVD) tutorial" on Coursera, part of the "Mathematics for Machine Learning" specialization.

These resources provide a mix of theoretical foundations and practical applications of SVD in various domains of machine learning and artificial intelligence.

