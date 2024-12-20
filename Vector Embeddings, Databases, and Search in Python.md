## Vector Embeddings, Databases, and Search in Python
Slide 1: Introduction to Vector Embeddings

Vector embeddings are numerical representations of data in a high-dimensional space. They capture semantic relationships between objects, making them crucial for various machine learning tasks.

```python
import numpy as np

# Create word embeddings
word_embeddings = {
    "king": np.array([0.5, 0.7, 0.2]),
    "queen": np.array([0.45, 0.75, 0.25]),
    "man": np.array([0.4, 0.3, 0.1]),
    "woman": np.array([0.35, 0.35, 0.15])
}

# Demonstrate vector operations
king_vector = word_embeddings["king"]
queen_vector = word_embeddings["queen"]
man_vector = word_embeddings["man"]

result = queen_vector - king_vector + man_vector

print("Resulting vector:", result)
print("Closest to 'woman':", np.allclose(result, word_embeddings["woman"], atol=0.1))
```

Slide 2: Creating Vector Embeddings

Vector embeddings can be created using various techniques, such as Word2Vec for text or CNN features for images. Here's a simple example using TensorFlow to create embeddings for text data.

```python
import tensorflow as tf

# Sample vocabulary
vocab = ["apple", "banana", "cherry", "date", "elderberry"]

# Create a lookup table
vocab_layer = tf.keras.layers.StringLookup(vocabulary=vocab)

# Create an embedding layer
embed_layer = tf.keras.layers.Embedding(input_dim=len(vocab) + 1, output_dim=5)

# Convert words to embeddings
words = tf.constant(["apple", "banana", "cherry"])
indices = vocab_layer(words)
embeddings = embed_layer(indices)

print("Word embeddings:")
print(embeddings.numpy())
```

Slide 3: Vector Database Basics

A vector database is designed to store and efficiently retrieve vector embeddings. It allows for similarity searches and nearest neighbor queries in high-dimensional spaces.

```python
import numpy as np
from scipy.spatial.distance import cosine

class SimpleVectorDB:
    def __init__(self):
        self.vectors = {}
    
    def add_vector(self, key, vector):
        self.vectors[key] = vector
    
    def search(self, query_vector, top_k=1):
        results = []
        for key, vector in self.vectors.items():
            similarity = 1 - cosine(query_vector, vector)
            results.append((key, similarity))
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

# Usage
db = SimpleVectorDB()
db.add_vector("cat", np.array([0.2, 0.5, 0.1]))
db.add_vector("dog", np.array([0.3, 0.4, 0.2]))
db.add_vector("fish", np.array([0.1, 0.1, 0.8]))

query = np.array([0.25, 0.45, 0.15])
results = db.search(query, top_k=2)
print("Search results:", results)
```

Slide 4: Indexing in Vector Databases

Efficient indexing is crucial for vector databases to handle large-scale data. Common indexing techniques include LSH (Locality-Sensitive Hashing) and tree-based methods like KD-trees.

```python
from sklearn.neighbors import KDTree
import numpy as np

# Sample data
vectors = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Build KD-tree
tree = KDTree(vectors)

# Query
query = np.array([[3, 4, 5]])
distances, indices = tree.query(query, k=2)

print("Nearest neighbors indices:", indices)
print("Distances:", distances)
```

Slide 5: Vector Search Algorithms

Vector search algorithms find similar vectors in high-dimensional spaces. Popular methods include k-Nearest Neighbors (k-NN) and Approximate Nearest Neighbors (ANN).

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Create and fit the model
nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
nn.fit(X)

# Query point
query = np.array([[3.5, 4.5]])

# Find nearest neighbors
distances, indices = nn.kneighbors(query)

print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
```

Slide 6: Dimensionality Reduction for Vector Embeddings

High-dimensional embeddings can be reduced to lower dimensions for visualization or efficiency. Common techniques include PCA and t-SNE.

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample high-dimensional data
np.random.seed(42)
X = np.random.rand(100, 50)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize the reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title("PCA-reduced Vector Embeddings")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
```

Slide 7: Cosine Similarity for Vector Comparison

Cosine similarity is a common metric for comparing vector embeddings, especially in text analysis and recommendation systems.

```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Example vectors
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
vec3 = np.array([-1, -2, -3])

print("Similarity between vec1 and vec2:", cosine_similarity(vec1, vec2))
print("Similarity between vec1 and vec3:", cosine_similarity(vec1, vec3))
```

Slide 8: Vector Quantization

Vector quantization reduces the storage requirements of vector embeddings by mapping them to a finite set of representative vectors.

```python
import numpy as np
from sklearn.cluster import KMeans

# Generate sample embeddings
np.random.seed(42)
embeddings = np.random.rand(1000, 10)

# Perform vector quantization using K-means
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(embeddings)

# Quantize vectors
quantized_embeddings = kmeans.cluster_centers_[kmeans.labels_]

print("Original shape:", embeddings.shape)
print("Quantized shape:", quantized_embeddings.shape)
print("Compression ratio:", embeddings.size / (quantized_embeddings.size + kmeans.cluster_centers_.size))
```

Slide 9: Handling Out-of-Vocabulary Words

When working with text embeddings, it's crucial to handle out-of-vocabulary (OOV) words. One approach is to use a special OOV token or average nearby word vectors.

```python
import numpy as np

class SimpleEmbedding:
    def __init__(self, vocab, embedding_dim):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embeddings = {word: np.random.randn(embedding_dim) for word in vocab}
        self.oov_vector = np.zeros(embedding_dim)
    
    def get_vector(self, word):
        return self.embeddings.get(word, self.oov_vector)
    
    def sentence_to_vector(self, sentence):
        words = sentence.lower().split()
        vectors = [self.get_vector(word) for word in words]
        return np.mean(vectors, axis=0)

# Usage
vocab = ["hello", "world", "python", "vector", "embedding"]
embed = SimpleEmbedding(vocab, embedding_dim=5)

print(embed.sentence_to_vector("Hello world"))
print(embed.sentence_to_vector("Python is awesome"))  # 'awesome' is OOV
```

Slide 10: Real-life Example: Document Similarity

Vector embeddings can be used to find similar documents in a large corpus, which is useful for content recommendation systems.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast orange fox leaps above a sleepy canine",
    "Python is a popular programming language",
    "Machine learning is a subset of artificial intelligence"
]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute pairwise cosine similarities
similarities = cosine_similarity(tfidf_matrix)

# Find the most similar document to the first one
most_similar = similarities[0].argsort()[-2]
print(f"Most similar to document 1: Document {most_similar + 1}")
print(f"Similarity score: {similarities[0][most_similar]:.2f}")
```

Slide 11: Real-life Example: Image Search

Vector embeddings can be used for image search by encoding images into vectors and finding the nearest neighbors.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Simulate image embeddings (in practice, these would come from a CNN)
np.random.seed(42)
num_images = 1000
embedding_dim = 128
image_embeddings = np.random.randn(num_images, embedding_dim)

# Create an efficient search index
nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nn.fit(image_embeddings)

# Simulate a query image
query_embedding = np.random.randn(1, embedding_dim)

# Find similar images
distances, indices = nn.kneighbors(query_embedding)

print("Indices of most similar images:", indices[0])
print("Distances to most similar images:", distances[0])
```

Slide 12: Challenges in Vector Search

Vector search faces challenges such as the "curse of dimensionality" and scalability issues. Techniques like approximate nearest neighbor search help address these problems.

```python
import numpy as np
from annoy import AnnoyIndex

# Generate sample high-dimensional data
num_vectors = 100000
dim = 100
vectors = np.random.randn(num_vectors, dim)

# Build Annoy index
index = AnnoyIndex(dim, 'angular')
for i, v in enumerate(vectors):
    index.add_item(i, v)

index.build(10)  # 10 trees for better accuracy

# Query
query_vector = np.random.randn(dim)
num_neighbors = 5

# Approximate nearest neighbor search
approx_nns = index.get_nns_by_vector(query_vector, num_neighbors)

print("Approximate nearest neighbors:", approx_nns)

# For comparison, calculate exact nearest neighbors
exact_nns = np.argsort(np.linalg.norm(vectors - query_vector, axis=1))[:num_neighbors]

print("Exact nearest neighbors:", exact_nns)
```

Slide 13: Future Directions in Vector Embeddings and Search

Emerging trends include multimodal embeddings, dynamic embeddings, and quantum-inspired algorithms for vector search.

```python
import numpy as np
from scipy.stats import unitary_group

def quantum_inspired_search(database, query, num_samples=10):
    # Simulate quantum-inspired sampling
    unitary = unitary_group.rvs(len(database))
    sampled_indices = np.random.choice(len(database), num_samples, replace=False)
    
    # Perform search on sampled subset
    similarities = np.dot(database[sampled_indices], query)
    best_match = sampled_indices[np.argmax(similarities)]
    
    return best_match

# Example usage
database = np.random.randn(1000, 50)  # 1000 vectors of dimension 50
query = np.random.randn(50)

result = quantum_inspired_search(database, query)
print(f"Best match index: {result}")
```

Slide 14: Additional Resources

For further exploration of vector embeddings, vector databases, and vector search, consider these peer-reviewed articles:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "Billion-scale similarity search with GPUs" by Johnson et al. (2017) ArXiv: [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)
3. "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms" by Aumüller et al. (2017) ArXiv: [https://arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614)

These papers provide in-depth insights into the techniques and applications of vector embeddings and search algorithms.

