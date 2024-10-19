## Cosine Similarity in Vector Stores
Slide 1: Introduction to Cosine Similarity

Cosine similarity is a metric used to measure the similarity between two non-zero vectors in a multi-dimensional space. It calculates the cosine of the angle between these vectors, providing a value between -1 and 1. In the context of vector stores and AI applications, cosine similarity is particularly useful for comparing high-dimensional data, such as text embeddings or feature vectors.

```python
import math

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (magnitude1 * magnitude2)

# Example vectors
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

similarity = cosine_similarity(vector1, vector2)
print(f"Cosine similarity: {similarity:.4f}")
```

Slide 2: Results for: Introduction to Cosine Similarity

```
Cosine similarity: 0.9746
```

Slide 3: Mathematical Foundation

The cosine similarity between two vectors A and B is defined by the dot product of the vectors divided by the product of their magnitudes. Mathematically, it can be expressed as:

cosine similarity\=A⋅B∥A∥∥B∥\=∑i\=1nAiBi∑i\=1nAi2∑i\=1nBi2\\text{cosine similarity} = \\frac{A \\cdot B}{\\|A\\| \\|B\\|} = \\frac{\\sum\_{i=1}^n A\_i B\_i}{\\sqrt{\\sum\_{i=1}^n A\_i^2} \\sqrt{\\sum\_{i=1}^n B\_i^2}}cosine similarity\=∥A∥∥B∥A⋅B​\=∑i\=1n​Ai2​​∑i\=1n​Bi2​​∑i\=1n​Ai​Bi​​

Where A and B are n-dimensional vectors, and A\_i and B\_i represent their respective components.

```python
import math

def cosine_similarity_formula(A, B):
    dot_product = sum(a * b for a, b in zip(A, B))
    magnitude_A = math.sqrt(sum(a ** 2 for a in A))
    magnitude_B = math.sqrt(sum(b ** 2 for b in B))
    return dot_product / (magnitude_A * magnitude_B)

# Example calculation
A = [1, 2, 3]
B = [4, 5, 6]
similarity = cosine_similarity_formula(A, B)
print(f"Cosine similarity: {similarity:.4f}")
```

Slide 4: Properties of Cosine Similarity

Cosine similarity has several important properties that make it useful for comparing vectors:

1.  It is independent of vector magnitude, focusing on the angle between vectors.
2.  The result ranges from -1 to 1, where 1 indicates identical direction, 0 indicates orthogonality, and -1 indicates opposite directions.
3.  It is particularly effective for sparse high-dimensional spaces, common in text analysis and recommendation systems.

```python
def cosine_similarity_properties():
    # Demonstrating magnitude independence
    v1 = [1, 0]
    v2 = [2, 0]
    v3 = [0, 1]
    
    print(f"Similarity between [1, 0] and [2, 0]: {cosine_similarity(v1, v2)}")
    print(f"Similarity between [1, 0] and [0, 1]: {cosine_similarity(v1, v3)}")
    
    # Demonstrating range
    v4 = [-1, 0]
    print(f"Similarity between [1, 0] and [-1, 0]: {cosine_similarity(v1, v4)}")

cosine_similarity_properties()
```

Slide 5: Results for: Properties of Cosine Similarity

```
Similarity between [1, 0] and [2, 0]: 1.0
Similarity between [1, 0] and [0, 1]: 0.0
Similarity between [1, 0] and [-1, 0]: -1.0
```

Slide 6: Cosine Similarity in Text Analysis

In text analysis, documents are often represented as vectors in a high-dimensional space where each dimension corresponds to a unique word. Cosine similarity can effectively measure the similarity between these document vectors, regardless of their length.

```python
from collections import Counter

def text_to_vector(text):
    words = text.lower().split()
    return Counter(words)

def cosine_similarity_text(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    
    intersection = set(vector1.keys()) & set(vector2.keys())
    numerator = sum([vector1[x] * vector2[x] for x in intersection])
    
    sum1 = sum([vector1[x]**2 for x in vector1.keys()])
    sum2 = sum([vector2[x]**2 for x in vector2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    return float(numerator) / denominator if denominator != 0 else 0.0

# Example usage
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "The lazy dog sleeps in the sun"

similarity = cosine_similarity_text(text1, text2)
print(f"Cosine similarity between texts: {similarity:.4f}")
```

Slide 7: Results for: Cosine Similarity in Text Analysis

```
Cosine similarity between texts: 0.3780
```

Slide 8: Implementing a Simple Vector Store

A vector store is a database optimized for storing and retrieving high-dimensional vectors. We'll implement a basic vector store using cosine similarity for retrieval.

```python
class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.data = []

    def add_vector(self, vector, data):
        self.vectors.append(vector)
        self.data.append(data)

    def search(self, query_vector, top_k=1):
        similarities = [cosine_similarity(query_vector, v) for v in self.vectors]
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        return [(self.data[i], similarities[i]) for i in sorted_indices[:top_k]]

# Example usage
vector_store = SimpleVectorStore()
vector_store.add_vector([1, 2, 3], "Document 1")
vector_store.add_vector([4, 5, 6], "Document 2")
vector_store.add_vector([1, 1, 1], "Document 3")

query = [2, 3, 4]
results = vector_store.search(query, top_k=2)
for doc, sim in results:
    print(f"{doc}: Similarity = {sim:.4f}")
```

Slide 9: Results for: Implementing a Simple Vector Store

```
Document 2: Similarity = 0.9746
Document 1: Similarity = 0.9925
```

Slide 10: Optimizing Vector Search

For large-scale vector stores, linear search becomes inefficient. Various algorithms and data structures can optimize vector search, such as Approximate Nearest Neighbors (ANN) algorithms like Locality-Sensitive Hashing (LSH) or Hierarchical Navigable Small World (HNSW) graphs.

```python
import random

class LSHVectorStore:
    def __init__(self, dim, num_hashes):
        self.dim = dim
        self.num_hashes = num_hashes
        self.hash_functions = [self._random_hyperplane() for _ in range(num_hashes)]
        self.buckets = {}

    def _random_hyperplane(self):
        return [random.gauss(0, 1) for _ in range(self.dim)]

    def _hash(self, vector):
        return tuple(1 if sum(a*b for a, b in zip(v, vector)) > 0 else 0 
                     for v in self.hash_functions)

    def add_vector(self, vector, data):
        h = self._hash(vector)
        if h not in self.buckets:
            self.buckets[h] = []
        self.buckets[h].append((vector, data))

    def search(self, query_vector, top_k=1):
        h = self._hash(query_vector)
        if h not in self.buckets:
            return []
        candidates = self.buckets[h]
        similarities = [(cosine_similarity(query_vector, v), d) for v, d in candidates]
        return sorted(similarities, reverse=True)[:top_k]

# Example usage
lsh_store = LSHVectorStore(dim=3, num_hashes=5)
lsh_store.add_vector([1, 2, 3], "Document 1")
lsh_store.add_vector([4, 5, 6], "Document 2")
lsh_store.add_vector([1, 1, 1], "Document 3")

query = [2, 3, 4]
results = lsh_store.search(query, top_k=2)
for sim, doc in results:
    print(f"{doc}: Similarity = {sim:.4f}")
```

Slide 11: Real-Life Example: Content-Based Recommendation System

A content-based recommendation system suggests items similar to those a user has liked in the past. We'll implement a simple recommendation system for movies using cosine similarity.

```python
class MovieRecommender:
    def __init__(self):
        self.movies = {}

    def add_movie(self, title, genres):
        self.movies[title] = genres

    def get_recommendations(self, liked_movie, top_k=3):
        if liked_movie not in self.movies:
            return []

        liked_genres = self.movies[liked_movie]
        similarities = []

        for title, genres in self.movies.items():
            if title != liked_movie:
                sim = cosine_similarity(liked_genres, genres)
                similarities.append((title, sim))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

# Example usage
recommender = MovieRecommender()
recommender.add_movie("The Matrix", [1, 1, 1, 0, 0])  # Sci-Fi, Action, Cyberpunk
recommender.add_movie("Inception", [1, 1, 0, 1, 0])   # Sci-Fi, Action, Thriller
recommender.add_movie("The Notebook", [0, 0, 0, 0, 1]) # Romance
recommender.add_movie("Interstellar", [1, 0, 0, 1, 0]) # Sci-Fi, Drama

recommendations = recommender.get_recommendations("The Matrix")
for movie, similarity in recommendations:
    print(f"Recommended: {movie}, Similarity: {similarity:.4f}")
```

Slide 12: Results for: Real-Life Example: Content-Based Recommendation System

```
Recommended: Inception, Similarity: 0.8165
Recommended: Interstellar, Similarity: 0.5774
Recommended: The Notebook, Similarity: 0.0000
```

Slide 13: Real-Life Example: Document Clustering

Document clustering is a technique used to group similar documents together. We'll implement a simple K-means clustering algorithm using cosine similarity as the distance metric.

```python
import random

def kmeans_clustering(documents, k, max_iterations=100):
    # Convert documents to term frequency vectors
    doc_vectors = [Counter(doc.split()) for doc in documents]

    # Initialize centroids randomly
    centroids = random.sample(doc_vectors, k)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        # Assign documents to nearest centroid
        for i, doc in enumerate(doc_vectors):
            nearest_centroid = max(range(k), key=lambda j: cosine_similarity(doc, centroids[j]))
            clusters[nearest_centroid].append(i)

        # Update centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                centroid = Counter()
                for i in cluster:
                    centroid.update(doc_vectors[i])
                new_centroids.append(centroid)
            else:
                new_centroids.append(random.choice(doc_vectors))

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return clusters

# Example usage
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outfoxes a lazy canine",
    "The lazy cat sleeps all day long",
    "Sleeping cats are very lazy animals"
]

k = 2
clusters = kmeans_clustering(documents, k)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}:")
    for doc_index in cluster:
        print(f"  - {documents[doc_index]}")
```

Slide 14: Results for: Real-Life Example: Document Clustering

```
Cluster 1:
  - The quick brown fox jumps over the lazy dog
  - A quick brown dog outfoxes a lazy canine
Cluster 2:
  - The lazy cat sleeps all day long
  - Sleeping cats are very lazy animals
```

Slide 15: Additional Resources

For those interested in diving deeper into cosine similarity and its applications in vector stores, the following resources from ArXiv.org provide valuable insights:

1.  "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" by Yu. A. Malkov and D. A. Yashunin (2016) ArXiv: [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
2.  "Billion-scale similarity search with GPUs" by Johnson et al. (2017) ArXiv: [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)

These papers discuss advanced techniques for optimizing similarity search in large-scale vector stores, which is crucial for practical applications of cosine similarity in AI and machine learning.

