## Describing Location with Latitude and Longitude Vectors
Slide 1: Vector Spaces Fundamentals

A vector space implementation focusing on basic operations like addition, scalar multiplication, and dot products. This forms the foundation for understanding high-dimensional spaces used in AI applications and machine learning models.

```python
import numpy as np

class VectorSpace:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        
    def create_vector(self, components):
        if len(components) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} components")
        return np.array(components)
    
    def add_vectors(self, v1, v2):
        return v1 + v2
    
    def scalar_multiply(self, scalar, vector):
        return scalar * vector
    
    def dot_product(self, v1, v2):
        return np.dot(v1, v2)

# Example usage
vs = VectorSpace(3)
v1 = vs.create_vector([1, 2, 3])
v2 = vs.create_vector([4, 5, 6])
print(f"Vector addition: {vs.add_vectors(v1, v2)}")
print(f"Scalar multiplication: {vs.scalar_multiply(2, v1)}")
print(f"Dot product: {vs.dot_product(v1, v2)}")
```

Slide 2: High-Dimensional Data Representation

Understanding how to represent and manipulate high-dimensional data is crucial in AI. This implementation demonstrates creating and handling vectors in arbitrary dimensions, with methods for basic statistical analysis.

```python
class HighDimensionalSpace:
    def __init__(self, data_matrix):
        self.data = np.array(data_matrix)
        self.dimensions = self.data.shape[1]
        
    def compute_centroid(self):
        return np.mean(self.data, axis=0)
    
    def compute_covariance(self):
        return np.cov(self.data.T)
    
    def euclidean_distance(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

# Example with 5D data
data = np.random.randn(100, 5)  # 100 samples, 5 dimensions
hds = HighDimensionalSpace(data)
print(f"Centroid: {hds.compute_centroid()}")
print(f"Covariance matrix shape: {hds.compute_covariance().shape}")
```

Slide 3: Dimensionality Reduction with PCA

Principal Component Analysis is a fundamental technique for reducing high-dimensional data while preserving important relationships. This implementation shows how to perform PCA from scratch using eigenvector decomposition.

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Store first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

# Example usage
X = np.random.randn(100, 10)  # 100 samples, 10 dimensions
pca = PCA(n_components=3)
pca.fit(X)
X_reduced = pca.transform(X)
print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")
```

Slide 4: Distance Metrics in High Dimensions

Understanding different distance metrics is crucial when working with high-dimensional spaces. This implementation showcases various distance measures commonly used in machine learning and their behavior in high dimensions.

```python
class DistanceMetrics:
    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    def manhattan_distance(x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    @staticmethod
    def cosine_similarity(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    @staticmethod
    def mahalanobis_distance(x, data):
        mean = np.mean(data, axis=0)
        cov_inv = np.linalg.inv(np.cov(data.T))
        diff = x - mean
        return np.sqrt(diff.dot(cov_inv).dot(diff))

# Example usage
x1 = np.random.randn(100)  # 100-dimensional vector
x2 = np.random.randn(100)
data = np.random.randn(1000, 100)  # 1000 samples, 100 dimensions

dm = DistanceMetrics()
print(f"Euclidean distance: {dm.euclidean_distance(x1, x2):.4f}")
print(f"Manhattan distance: {dm.manhattan_distance(x1, x2):.4f}")
print(f"Cosine similarity: {dm.cosine_similarity(x1, x2):.4f}")
print(f"Mahalanobis distance: {dm.mahalanobis_distance(x1, data):.4f}")
```

Slide 5: The Curse of Dimensionality

Implementation demonstrating how distance metrics behave differently in high dimensions, illustrating the curse of dimensionality through practical examples and visualizations of distance distributions.

```python
import matplotlib.pyplot as plt

class DimensionalityEffects:
    def __init__(self, max_dim=1000):
        self.max_dim = max_dim
        
    def analyze_distance_distribution(self, n_points=1000):
        dimensions = [2, 10, 100, 500, self.max_dim]
        distances = []
        
        for dim in dimensions:
            # Generate random points in dim-dimensional space
            points = np.random.normal(0, 1, (n_points, dim))
            
            # Calculate pairwise distances
            dist = []
            for i in range(n_points-1):
                for j in range(i+1, n_points):
                    d = np.sqrt(np.sum((points[i] - points[j])**2))
                    dist.append(d/np.sqrt(dim))  # Normalize by sqrt(dim)
            distances.append(dist)
        
        # Plot distributions
        plt.figure(figsize=(10, 6))
        plt.boxplot(distances, labels=dimensions)
        plt.xlabel('Dimensions')
        plt.ylabel('Normalized Distance')
        plt.title('Distance Distribution vs Dimensionality')
        return plt

# Example usage
de = DimensionalityEffects()
plt = de.analyze_distance_distribution()
plt.savefig('dimensionality_effect.png')
print("Analysis complete. Check dimensionality_effect.png for visualization")
```

Slide 6: Vector Space Transformations

Implementation of common vector space transformations used in machine learning, including linear transformations, normalization, and standardization for high-dimensional data.

```python
class VectorTransformations:
    def __init__(self):
        self.scale_params = {}
        
    def linear_transform(self, X, A):
        """Apply linear transformation AX"""
        return np.dot(X, A)
    
    def normalize(self, X, norm='l2'):
        """Normalize vectors"""
        if norm == 'l2':
            return X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        elif norm == 'l1':
            return X / np.sum(np.abs(X), axis=1)[:, np.newaxis]
    
    def standardize(self, X, fit=True):
        """Standardize features to zero mean and unit variance"""
        if fit:
            self.scale_params['mean'] = np.mean(X, axis=0)
            self.scale_params['std'] = np.std(X, axis=0)
            
        return (X - self.scale_params['mean']) / self.scale_params['std']

# Example usage
X = np.random.randn(100, 5)  # 100 samples, 5 dimensions
A = np.random.randn(5, 3)    # Transform to 3 dimensions

vt = VectorTransformations()
X_transformed = vt.linear_transform(X, A)
X_normalized = vt.normalize(X)
X_standardized = vt.standardize(X)

print(f"Original shape: {X.shape}")
print(f"Transformed shape: {X_transformed.shape}")
print(f"Normalized shape: {X_normalized.shape}")
print(f"Standardized shape: {X_standardized.shape}")
```

Slide 7: Feature Embeddings in High Dimensions

This implementation demonstrates how to create and manipulate feature embeddings, a crucial concept in AI where discrete features are mapped to continuous vectors in high-dimensional space.

```python
class FeatureEmbedding:
    def __init__(self, vocab_size, embedding_dim):
        # Initialize random embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) / np.sqrt(embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def embed_sequence(self, sequence):
        return self.embeddings[sequence]
    
    def cosine_similarity_matrix(self):
        # Compute pairwise cosine similarities
        norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized = self.embeddings / norm
        return normalized @ normalized.T
    
    def find_nearest_neighbors(self, vector, k=5):
        similarities = self.embeddings @ vector / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vector)
        )
        return np.argsort(similarities)[-k:][::-1]

# Example usage
vocab_size, embedding_dim = 1000, 50
embedder = FeatureEmbedding(vocab_size, embedding_dim)

# Example sequence
sequence = np.array([1, 4, 2, 7])
embedded_sequence = embedder.embed_sequence(sequence)
print(f"Embedded sequence shape: {embedded_sequence.shape}")

# Find nearest neighbors for a random vector
query_vector = np.random.randn(embedding_dim)
neighbors = embedder.find_nearest_neighbors(query_vector)
print(f"Nearest neighbors: {neighbors}")
```

Slide 8: High-Dimensional Clustering

Implementation of clustering algorithms specifically adapted for high-dimensional spaces, including optimizations to handle the curse of dimensionality.

```python
class HighDimensionalClustering:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        
    def initialize_centroids(self, X):
        # K-means++ initialization
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            distances = np.min([np.sum((X - c) ** 2, axis=1) 
                              for c in centroids], axis=0)
            probabilities = distances / distances.sum()
            next_centroid = X[np.random.choice(n_samples, p=probabilities)]
            centroids.append(next_centroid)
            
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iter):
            # Assign points to nearest centroids
            distances = np.array([np.sum((X - c) ** 2, axis=1) 
                                for c in self.centroids])
            labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return labels

# Example usage
X = np.random.randn(1000, 50)  # 1000 samples in 50 dimensions
clusterer = HighDimensionalClustering(n_clusters=5)
labels = clusterer.fit(X)
print(f"Cluster sizes: {np.bincount(labels)}")
```

Slide 9: Vector Space Visualization Techniques

Implementation of dimensionality reduction techniques for visualizing high-dimensional vector spaces, including t-SNE and UMAP-inspired approaches.

```python
class VectorSpaceVisualizer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        
    def compute_pairwise_affinities(self, X, perplexity=30.0):
        distances = np.sum((X[:, np.newaxis, :] - X) ** 2, axis=2)
        P = np.zeros((X.shape[0], X.shape[0]))
        
        for i in range(X.shape[0]):
            distances_i = distances[i]
            sigma = self._binary_search_sigma(distances_i, perplexity)
            P[i] = np.exp(-distances_i / (2 * sigma ** 2))
            P[i, i] = 0
            P[i] = P[i] / np.sum(P[i])
            
        return (P + P.T) / (2 * X.shape[0])
    
    def _binary_search_sigma(self, distances, perplexity, tol=1e-5, max_iter=50):
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0
        
        for _ in range(max_iter):
            prob = np.exp(-distances * beta)
            prob[prob == np.inf] = 0
            prob /= np.sum(prob)
            
            entropy = -np.sum(prob * np.log2(prob + 1e-7))
            perp = 2 ** entropy
            
            if np.abs(perp - perplexity) < tol:
                break
                
            if perp > perplexity:
                beta_min = beta
                beta = (beta + beta_max) / 2 if beta_max != np.inf else beta * 2
            else:
                beta_max = beta
                beta = (beta + beta_min) / 2 if beta_min != -np.inf else beta / 2
                
        return 1 / np.sqrt(beta)

# Example usage
X = np.random.randn(500, 50)  # 500 samples in 50 dimensions
visualizer = VectorSpaceVisualizer()
affinities = visualizer.compute_pairwise_affinities(X)
print(f"Affinity matrix shape: {affinities.shape}")
```

Slide 10: Neural Network Embeddings

Implementation of a neural network that learns to create meaningful embeddings in high-dimensional space, demonstrating how networks can learn optimal representations of data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

class EmbeddingTrainer:
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        self.model = EmbeddingNetwork(input_dim, embedding_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def train_step(self, batch):
        self.optimizer.zero_grad()
        embedding, reconstruction = self.model(batch)
        loss = self.criterion(reconstruction, batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), embedding

# Example usage
input_dim, embedding_dim, hidden_dim = 100, 20, 50
trainer = EmbeddingTrainer(input_dim, embedding_dim, hidden_dim)

# Generate sample data
X = torch.randn(1000, input_dim)
loss, embeddings = trainer.train_step(X)
print(f"Training loss: {loss:.4f}")
print(f"Embedding shape: {embeddings.shape}")
```

Slide 11: Manifold Learning in Vector Spaces

Implementation of manifold learning techniques to discover the underlying structure of high-dimensional data, focusing on locally linear embeddings and geodesic distance computation.

```python
class ManifoldLearning:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        
    def compute_adjacency_matrix(self, X):
        n_samples = X.shape[0]
        adjacency = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            distances = np.sum((X - X[i]) ** 2, axis=1)
            nearest = np.argsort(distances)[1:self.n_neighbors + 1]
            adjacency[i, nearest] = 1
            adjacency[nearest, i] = 1
            
        return adjacency
    
    def compute_geodesic_distances(self, adjacency):
        n_samples = adjacency.shape[0]
        geodesic = np.full((n_samples, n_samples), np.inf)
        
        # Initialize with direct connections
        geodesic[adjacency == 1] = 1
        np.fill_diagonal(geodesic, 0)
        
        # Floyd-Warshall algorithm
        for k in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    if geodesic[i, j] > geodesic[i, k] + geodesic[k, j]:
                        geodesic[i, j] = geodesic[i, k] + geodesic[k, j]
                        
        return geodesic
    
    def fit_transform(self, X):
        adjacency = self.compute_adjacency_matrix(X)
        geodesic = self.compute_geodesic_distances(adjacency)
        return geodesic

# Example usage
X = np.random.randn(200, 50)  # 200 samples in 50 dimensions
manifold = ManifoldLearning()
distances = manifold.fit_transform(X)
print(f"Geodesic distances shape: {distances.shape}")
print(f"Average geodesic distance: {np.mean(distances[distances != np.inf]):.4f}")
```

Slide 12: Real-World Application - Text Embeddings

Implementation of a practical text embedding system that converts documents into high-dimensional vectors, demonstrating how to process and analyze textual data in vector space.

```python
from collections import Counter
import numpy as np
from sklearn.preprocessing import normalize

class TextVectorizer:
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.embeddings = None
        self.word_counts = Counter()
        
    def fit(self, documents):
        # Build vocabulary
        words = [word for doc in documents 
                for word in doc.lower().split()]
        self.word_counts.update(words)
        
        # Keep most frequent words
        vocab = [w for w, c in self.word_counts.most_common(10000)]
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        
        # Initialize random embeddings
        self.embeddings = np.random.randn(len(vocab), self.embedding_dim)
        self.embeddings = normalize(self.embeddings)
        
    def transform(self, document):
        words = document.lower().split()
        indices = [self.word_to_idx[w] for w in words 
                  if w in self.word_to_idx]
        
        if not indices:
            return np.zeros(self.embedding_dim)
            
        # Average word embeddings
        doc_vector = np.mean(self.embeddings[indices], axis=0)
        return normalize(doc_vector.reshape(1, -1))[0]
    
    def document_similarity(self, doc1, doc2):
        vec1 = self.transform(doc1)
        vec2 = self.transform(doc2)
        return np.dot(vec1, vec2)

# Example usage
documents = [
    "artificial intelligence and machine learning",
    "deep learning neural networks",
    "natural language processing",
    "computer vision and image recognition"
]

vectorizer = TextVectorizer()
vectorizer.fit(documents)

doc1 = "AI and deep learning"
doc2 = "machine learning algorithms"
similarity = vectorizer.document_similarity(doc1, doc2)
print(f"Document similarity: {similarity:.4f}")
```

Slide 13: Real-World Application - Image Feature Extraction

Implementation of a feature extraction system for images, demonstrating how to convert visual data into high-dimensional vectors suitable for machine learning tasks.

```python
import numpy as np
from scipy.signal import convolve2d

class ImageFeatureExtractor:
    def __init__(self, n_features=256):
        self.n_features = n_features
        self.filters = self._initialize_filters()
        
    def _initialize_filters(self):
        # Create Gabor filters at different orientations
        filters = []
        for theta in np.linspace(0, np.pi, 8):
            kernel = np.zeros((7, 7))
            for x in range(7):
                for y in range(7):
                    x0 = x - 3
                    y0 = y - 3
                    x1 = x0 * np.cos(theta) - y0 * np.sin(theta)
                    y1 = x0 * np.sin(theta) + y0 * np.cos(theta)
                    kernel[x, y] = np.exp(-(x1**2 + y1**2)/4) * np.cos(2*np.pi*x1/4)
            filters.append(kernel)
        return filters
    
    def extract_features(self, image):
        if len(image.shape) == 3:
            # Convert to grayscale if colored
            image = np.mean(image, axis=2)
            
        features = []
        for filter_kernel in self.filters:
            # Apply filter
            filtered = convolve2d(image, filter_kernel, mode='valid')
            
            # Extract statistical features
            features.extend([
                np.mean(filtered),
                np.std(filtered),
                np.percentile(filtered, 10),
                np.percentile(filtered, 90)
            ])
            
        # Add histogram features
        hist, _ = np.histogram(image, bins=32, range=(0, 255))
        features.extend(hist / np.sum(hist))
        
        return np.array(features)

# Example usage
# Generate sample image (64x64 random pattern)
image = np.random.randint(0, 256, (64, 64))
extractor = ImageFeatureExtractor()
features = extractor.extract_features(image)
print(f"Extracted feature vector shape: {features.shape}")
print(f"Feature statistics - Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")
```

Slide 14: Additional Resources

*   ArXiv: "Dimensionality Reduction: A Comparative Review"
    *   [https://arxiv.org/abs/0904.1430](https://arxiv.org/abs/0904.1430)
*   ArXiv: "Understanding High-Dimensional Spaces"
    *   [https://arxiv.org/abs/1806.09337](https://arxiv.org/abs/1806.09337)
*   ArXiv: "A Tutorial on Principal Component Analysis"
    *   [https://arxiv.org/abs/1404.1100](https://arxiv.org/abs/1404.1100)
*   Suggested searches:
    *   "Vector Space Models in NLP"
    *   "Manifold Learning Techniques"
    *   "High-Dimensional Data Visualization Methods"

