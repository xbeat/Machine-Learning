## Matrix Factorization in Machine Learning
Slide 1: Matrix Factorization in Machine Learning and AI

Matrix factorization is a fundamental technique in machine learning and AI that decomposes a matrix into the product of two or more matrices. This process is crucial for dimensionality reduction, feature extraction, and collaborative filtering. In this presentation, we'll explore the concept, its applications, and implementation using Python.

```python
import numpy as np

# Create a sample matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Perform Singular Value Decomposition (SVD)
U, S, V = np.linalg.svd(matrix)

# Reconstruct the original matrix
reconstructed = np.dot(U, np.dot(np.diag(S), V))

print("Original matrix:\n", matrix)
print("\nReconstructed matrix:\n", reconstructed)
```

Slide 2: Types of Matrix Factorization

There are several types of matrix factorization techniques, each with its own properties and use cases. Some common types include Singular Value Decomposition (SVD), Non-negative Matrix Factorization (NMF), and QR Decomposition. These methods differ in their constraints and the resulting factorized matrices.

```python
import numpy as np
from scipy.linalg import qr
from sklearn.decomposition import NMF

# Create a sample matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# SVD
U, S, V = np.linalg.svd(matrix)

# QR Decomposition
Q, R = qr(matrix)

# NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(matrix)
H = model.components_

print("SVD - U:\n", U)
print("QR - Q:\n", Q)
print("NMF - W:\n", W)
```

Slide 3: Singular Value Decomposition (SVD)

SVD is one of the most widely used matrix factorization techniques. It decomposes a matrix A into the product of three matrices: U, Σ, and V^T. U and V are orthogonal matrices, and Σ is a diagonal matrix containing singular values. SVD is particularly useful for dimensionality reduction and noise reduction in data.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a sample matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Perform SVD
U, S, V = np.linalg.svd(matrix)

# Plot singular values
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(S) + 1), S)
plt.title('Singular Values')
plt.xlabel('Component')
plt.ylabel('Singular Value')
plt.show()

# Reconstruct the matrix using different numbers of singular values
for k in range(1, len(S) + 1):
    reconstructed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
    print(f"Reconstruction with {k} singular values:\n", reconstructed)
```

Slide 4: Non-negative Matrix Factorization (NMF)

NMF is a matrix factorization method that constrains the factors to be non-negative. This property makes it particularly useful for applications where negative values don't make sense, such as in image processing or topic modeling. NMF decomposes a matrix A into two non-negative matrices W and H, such that A ≈ WH.

```python
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# Create a sample non-negative matrix
matrix = np.abs(np.random.randn(10, 5))

# Apply NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(matrix)
H = model.components_

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(matrix, aspect='auto', cmap='viridis')
ax1.set_title('Original Matrix')

ax2.imshow(W, aspect='auto', cmap='viridis')
ax2.set_title('W Matrix')

ax3.imshow(H, aspect='auto', cmap='viridis')
ax3.set_title('H Matrix')

plt.tight_layout()
plt.show()

print("Original matrix shape:", matrix.shape)
print("W matrix shape:", W.shape)
print("H matrix shape:", H.shape)
```

Slide 5: Dimensionality Reduction with SVD

One of the primary applications of matrix factorization is dimensionality reduction. By keeping only the most significant singular values and their corresponding vectors, we can approximate the original matrix with fewer dimensions. This technique is often used in data compression and feature extraction.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a high-dimensional dataset
np.random.seed(42)
X = np.random.randn(100, 50)

# Perform SVD
U, S, V = np.linalg.svd(X, full_matrices=False)

# Calculate cumulative explained variance ratio
explained_variance_ratio = np.cumsum(S**2) / np.sum(S**2)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Reduce dimensionality
k = 10  # Number of components to keep
X_reduced = np.dot(U[:, :k], np.diag(S[:k]))

print("Original data shape:", X.shape)
print("Reduced data shape:", X_reduced.shape)
```

Slide 6: Collaborative Filtering with Matrix Factorization

Matrix factorization is widely used in recommender systems for collaborative filtering. It can be used to predict user-item interactions by decomposing the user-item interaction matrix into user and item latent factor matrices. This approach helps in discovering latent features that explain observed user-item interactions.

```python
import numpy as np
from sklearn.metrics import mean_squared_error

# Create a sample user-item interaction matrix
user_item_matrix = np.array([
    [4, 3, 0, 5, 0],
    [5, 0, 4, 0, 2],
    [3, 1, 2, 4, 1],
    [0, 0, 0, 2, 0],
    [1, 0, 3, 4, 0]
])

# Perform matrix factorization
U, S, V = np.linalg.svd(user_item_matrix)

# Choose the number of latent factors
k = 2

# Reconstruct the matrix using k latent factors
user_factors = U[:, :k]
item_factors = V[:k, :]
reconstructed_matrix = np.dot(user_factors, np.dot(np.diag(S[:k]), item_factors))

# Calculate RMSE
mask = user_item_matrix != 0
rmse = np.sqrt(mean_squared_error(user_item_matrix[mask], reconstructed_matrix[mask]))

print("Original matrix:\n", user_item_matrix)
print("\nReconstructed matrix:\n", reconstructed_matrix)
print(f"\nRMSE: {rmse:.4f}")
```

Slide 7: Image Compression using SVD

Matrix factorization can be applied to image compression by treating an image as a matrix of pixel values. By using SVD and keeping only the most significant singular values, we can compress the image while maintaining most of its visual information.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Load a sample image
image = data.camera()

# Perform SVD on the image
U, S, V = np.linalg.svd(image, full_matrices=False)

# Function to reconstruct image with k singular values
def reconstruct_image(U, S, V, k):
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))

# Reconstruct images with different numbers of singular values
k_values = [5, 20, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')

for i, k in enumerate(k_values):
    row, col = (i + 1) // 3, (i + 1) % 3
    reconstructed = reconstruct_image(U, S, V, k)
    axes[row, col].imshow(reconstructed, cmap='gray')
    axes[row, col].set_title(f'k = {k}')

plt.tight_layout()
plt.show()

# Print compression ratios
original_size = image.shape[0] * image.shape[1]
for k in k_values:
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio for k={k}: {compression_ratio:.2f}")
```

Slide 8: Topic Modeling with NMF

Non-negative Matrix Factorization is particularly useful for topic modeling in natural language processing. By applying NMF to a document-term matrix, we can discover latent topics in a collection of documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

# Sample documents
documents = [
    "The sky is blue and beautiful.",
    "Love this blue and calm sky!",
    "The quick brown fox jumps over the lazy dog.",
    "A king's breakfast has sausages, ham, bacon, eggs, toast and beans",
    "I love green eggs, ham, sausages and bacon!",
    "The brown fox is quick and the blue dog is lazy!",
    "The sky is very blue and the sky is very beautiful today",
    "The dog is lazy but the brown fox is quick!"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(documents)

# Apply NMF
n_topics = 3
nmf_model = NMF(n_components=n_topics, random_state=42)
topic_matrix = nmf_model.fit_transform(tfidf_matrix)

# Get the top words for each topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(nmf_model.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# Print document-topic distribution
for doc_idx, doc_topics in enumerate(topic_matrix):
    print(f"\nDocument {doc_idx + 1} topic distribution:")
    for topic_idx, weight in enumerate(doc_topics):
        print(f"Topic {topic_idx + 1}: {weight:.4f}")
```

Slide 9: Matrix Completion

Matrix completion is a technique used to fill in missing values in a matrix. It's particularly useful in recommender systems where we often have sparse user-item interaction matrices. Matrix factorization can be used to predict the missing values.

```python
import numpy as np
from sklearn.impute import SimpleImputer

# Create a sample matrix with missing values
matrix = np.array([
    [4, np.nan, 2, 5],
    [np.nan, 3, np.nan, 1],
    [6, 2, np.nan, 4],
    [1, np.nan, 5, 3]
])

# Perform matrix completion using mean imputation
imputer = SimpleImputer(strategy='mean')
completed_matrix = imputer.fit_transform(matrix)

print("Original matrix with missing values:\n", matrix)
print("\nCompleted matrix:\n", completed_matrix)

# Perform SVD on the completed matrix
U, S, V = np.linalg.svd(completed_matrix)

# Reconstruct the matrix using a reduced number of singular values
k = 2
reconstructed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))

print("\nReconstructed matrix:\n", reconstructed)

# Calculate RMSE for non-missing values
mask = ~np.isnan(matrix)
rmse = np.sqrt(np.mean((matrix[mask] - reconstructed[mask])**2))
print(f"\nRMSE: {rmse:.4f}")
```

Slide 10: Eigenface Representation

Matrix factorization can be used in facial recognition systems to create a set of eigenfaces. These eigenfaces represent the principal components of the variation in facial images and can be used for efficient face recognition and reconstruction.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# Load face dataset
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = faces.data
n_samples, n_features = X.shape

# Perform PCA (which uses SVD internally)
n_components = 150
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

# Plot the first few eigenfaces
eigenfaces = pca.components_.reshape((n_components, faces.images[0].shape[0], faces.images[0].shape[1]))

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(eigenfaces[i], cmap=plt.cm.gray)
    plt.title(f"Eigenface {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Reconstruct a face using different numbers of components
original_face = X[0].reshape(faces.images[0].shape)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(original_face, cmap=plt.cm.gray)
plt.title("Original Face")
plt.axis('off')

for i, n in enumerate([10, 50, 100, 150]):
    reconstructed = pca.inverse_transform(pca.transform([X[0]])[:, :n])
    plt.subplot(2, 3, i + 2)
    plt.imshow(reconstructed.reshape(faces.images[0].shape), cmap=plt.cm.gray)
    plt.title(f"Reconstructed ({n} components)")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

Slide 11: Latent Semantic Analysis (LSA)

Latent Semantic Analysis is a technique used in natural language processing to analyze relationships between documents and terms. It uses SVD to reduce the dimensionality of the term-document matrix, revealing latent semantic structures in the text data.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Sample documents
documents = [
    "The cat and the dog",
    "The dog chased the cat",
    "The cat climbed the tree",
    "Dogs like to play fetch",
    "Cats enjoy sleeping in the sun"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Perform LSA
n_components = 2
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Print results
print("Document-topic matrix:")
print(lsa_matrix)

print("\nTop terms for each topic:")
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(lsa.components_):
    top_terms = [terms[j] for j in comp.argsort()[:-6:-1]]
    print(f"Topic {i + 1}: {', '.join(top_terms)}")
```

Slide 12: Tensor Factorization

While matrix factorization deals with two-dimensional data, tensor factorization extends this concept to higher-dimensional data. Tensor factorization techniques, such as CANDECOMP/PARAFAC (CP) decomposition, are used in various applications including signal processing and recommender systems.

```python
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

# Create a sample 3D tensor
tensor = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]]
])

# Perform CP decomposition
rank = 2
factors = parafac(tensor, rank=rank)

# Reconstruct the tensor
reconstructed_tensor = tl.cp_to_tensor(factors)

print("Original tensor:")
print(tensor)
print("\nReconstructed tensor:")
print(reconstructed_tensor)

# Calculate reconstruction error
error = np.linalg.norm(tensor - reconstructed_tensor)
print(f"\nReconstruction error: {error:.4f}")
```

Slide 13: Real-life Example: Document Clustering

Matrix factorization can be used for document clustering, helping to group similar documents together. This technique is widely used in information retrieval and text mining applications.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast fox leaps above a sleepy canine",
    "Python is a popular programming language",
    "Coding in Python is fun and productive",
    "Machine learning algorithms process data",
    "Data science involves statistical analysis"
]

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Perform NMF
n_components = 2
nmf = NMF(n_components=n_components, random_state=42)
nmf_features = nmf.fit_transform(tfidf_matrix)

# Cluster documents using K-means
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(nmf_features)

# Print results
for i, doc in enumerate(documents):
    print(f"Document {i + 1} (Cluster {clusters[i]}): {doc}")

print("\nTop terms for each component:")
terms = vectorizer.get_feature_names_out()
for i, comp in enumerate(nmf.components_):
    top_terms = [terms[j] for j in comp.argsort()[:-6:-1]]
    print(f"Component {i + 1}: {', '.join(top_terms)}")
```

Slide 14: Real-life Example: Image Denoising

Matrix factorization techniques, particularly SVD, can be used for image denoising. By decomposing a noisy image and reconstructing it using only the most significant components, we can reduce noise while preserving important features.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util

# Load and add noise to an image
image = data.camera()
noisy_image = util.random_noise(image, mode='gaussian', var=0.1)

# Perform SVD
U, S, V = np.linalg.svd(noisy_image, full_matrices=False)

# Function to reconstruct image with k singular values
def reconstruct_image(U, S, V, k):
    return np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))

# Reconstruct image with different numbers of components
k_values = [10, 50, 100, 200]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(noisy_image, cmap='gray')
axes[0, 1].set_title('Noisy Image')

for i, k in enumerate(k_values):
    row, col = (i + 2) // 3, (i + 2) % 3
    denoised = reconstruct_image(U, S, V, k)
    axes[row, col].imshow(denoised, cmap='gray')
    axes[row, col].set_title(f'Denoised (k = {k})')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into matrix factorization and its applications in machine learning and AI, here are some valuable resources:

1. "Matrix Factorization Techniques for Recommender Systems" by Yehuda Koren et al. (2009) ArXiv: [https://arxiv.org/abs/0911.3421](https://arxiv.org/abs/0911.3421)
2. "Probabilistic Matrix Factorization" by Ruslan Salakhutdinov and Andriy Mnih (2007) NIPS Proceedings: [https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html](https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html)
3. "Tensor Decompositions and Applications" by Tamara G. Kolda and Brett W. Bader (2009) SIAM Review: [https://epubs.siam.org/doi/10.1137/07070111X](https://epubs.siam.org/doi/10.1137/07070111X)

These resources provide in-depth explanations and advanced applications of matrix factorization techniques in various domains of machine learning and artificial intelligence.

