## Embeddings Re-ranking and Vector Databases in Python
Slide 1: Introduction to Embeddings

Embeddings are dense vector representations of data that capture semantic meaning. They're crucial for various machine learning tasks, especially in natural language processing. Let's create a simple word embedding using Python and the Gensim library.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [["cat", "dog", "pet"], ["car", "drive", "vehicle"]]

# Train the model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get the embedding for a word
cat_embedding = model.wv['cat']
print(f"Embedding for 'cat': {cat_embedding[:5]}...")  # Show first 5 dimensions

# Find similar words
similar_words = model.wv.most_similar('cat', topn=3)
print(f"Words similar to 'cat': {similar_words}")
```

Slide 2: Understanding Word Embeddings

Word embeddings represent words in a continuous vector space. Words with similar meanings are closer in this space. Let's visualize this using t-SNE dimensionality reduction.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get word vectors
words = list(model.wv.key_to_index.keys())
word_vectors = [model.wv[word] for word in words]

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(word_vectors)

# Plot the results
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.annotate(word, (x, y))
plt.title("Word Embeddings Visualization")
plt.show()
```

Slide 3: Document Embeddings

Document embeddings represent entire documents as vectors. We can create them by averaging word embeddings or using more sophisticated methods like Doc2Vec.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Prepare the data
documents = [
    "The cat sat on the mat",
    "The dog chased the cat",
    "The mat was comfortable"
]
tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(documents)]

# Train the model
model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(tagged_docs)
model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

# Get document vector
doc_vector = model.infer_vector("The cat and dog played".split())
print(f"Document vector: {doc_vector[:5]}...")  # Show first 5 dimensions
```

Slide 4: Real-Life Example: Sentiment Analysis

Embeddings can significantly improve sentiment analysis tasks. Let's use pre-trained GloVe embeddings for a simple sentiment classifier.

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Sample data
texts = ["This movie was great", "I hated this book", "The food was delicious"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=20)

# Load pre-trained GloVe embeddings (assuming you have the file)
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((10000, 100))
for word, i in tokenizer.word_index.items():
    if i < 10000:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Create the model
model = Sequential([
    Embedding(10000, 100, weights=[embedding_matrix], input_length=20, trainable=False),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, np.array(labels), epochs=10, batch_size=32)
```

Slide 5: Introduction to Re-ranking

Re-ranking is the process of improving the order of search results or recommendations. It often involves a two-stage approach: initial retrieval followed by a more sophisticated ranking model.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initial set of documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "The lazy cat sleeps all day",
    "Quick foxes and lazy dogs are common in stories"
]

# Query
query = "quick brown fox"

# Initial ranking using TF-IDF and cosine similarity
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

# Calculate similarities
similarities = cosine_similarity(query_vector, doc_vectors).flatten()

# Sort documents by similarity
initial_ranking = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

print("Initial ranking:")
for idx, score in initial_ranking:
    print(f"{documents[idx]}: {score}")
```

Slide 6: Implementing a Simple Re-ranker

Let's implement a simple re-ranker that considers both the initial ranking and the presence of specific keywords.

```python
def rerank(initial_ranking, documents, keywords):
    reranked = []
    for idx, score in initial_ranking:
        doc = documents[idx]
        # Boost score if document contains keywords
        keyword_boost = sum(2 if keyword in doc.lower() else 0 for keyword in keywords)
        new_score = score + keyword_boost * 0.1  # Adjust the impact of keyword presence
        reranked.append((idx, new_score))
    return sorted(reranked, key=lambda x: x[1], reverse=True)

# Keywords to boost
keywords = ["quick", "fox"]

# Apply re-ranking
reranked = rerank(initial_ranking, documents, keywords)

print("\nRe-ranked results:")
for idx, score in reranked:
    print(f"{documents[idx]}: {score}")
```

Slide 7: Real-Life Example: Product Search Re-ranking

In e-commerce, re-ranking can improve product search results by considering factors like user preferences and product popularity.

```python
import random

# Simulated product data
products = [
    {"id": 1, "name": "Laptop", "category": "Electronics", "price": 1000, "rating": 4.5, "sales": 500},
    {"id": 2, "name": "Smartphone", "category": "Electronics", "price": 800, "rating": 4.2, "sales": 1000},
    {"id": 3, "name": "Headphones", "category": "Electronics", "price": 200, "rating": 4.0, "sales": 2000},
    {"id": 4, "name": "Tablet", "category": "Electronics", "price": 600, "rating": 4.3, "sales": 750},
]

# Initial ranking based on simple text match
query = "electronic device"
initial_ranking = sorted(products, key=lambda x: random.random())  # Simulating initial retrieval

# Re-ranking function
def rerank_products(products, user_preferences):
    return sorted(products, key=lambda x: (
        x['rating'] * 0.4 +  # Consider product rating
        (1000 - x['price']) / 1000 * 0.3 +  # Lower price is better
        x['sales'] / 2000 * 0.2 +  # Consider popularity
        (1 if x['category'] in user_preferences['preferred_categories'] else 0) * 0.1  # User category preference
    ), reverse=True)

# Simulated user preferences
user_preferences = {
    "preferred_categories": ["Electronics"],
    "price_sensitivity": "medium"
}

# Apply re-ranking
reranked_products = rerank_products(initial_ranking, user_preferences)

print("Re-ranked product search results:")
for product in reranked_products:
    print(f"{product['name']} - Price: ${product['price']}, Rating: {product['rating']}, Sales: {product['sales']}")
```

Slide 8: Introduction to Vector Databases

Vector databases are specialized systems designed to store and efficiently query high-dimensional vectors, such as embeddings. They're crucial for similarity search in large datasets.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simulating a simple vector database
class SimpleVectorDB:
    def __init__(self):
        self.vectors = []
        self.metadata = []
    
    def add(self, vector, metadata):
        self.vectors.append(vector)
        self.metadata.append(metadata)
    
    def search(self, query_vector, k=5):
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [(self.metadata[i], similarities[i]) for i in top_k_indices]

# Create and populate the database
db = SimpleVectorDB()
for i in range(1000):
    vector = np.random.rand(100)  # 100-dimensional vector
    metadata = f"Item {i}"
    db.add(vector, metadata)

# Perform a search
query = np.random.rand(100)
results = db.search(query, k=5)

print("Search results:")
for metadata, similarity in results:
    print(f"{metadata}: Similarity = {similarity:.4f}")
```

Slide 9: Indexing in Vector Databases

Efficient indexing is crucial for fast similarity search in vector databases. Let's implement a simple version of the Inverted File Index (IVF).

```python
import numpy as np
from sklearn.cluster import KMeans

class IVFIndex:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.inverted_lists = [[] for _ in range(n_clusters)]
    
    def build(self, vectors):
        self.kmeans.fit(vectors)
        for i, vector in enumerate(vectors):
            cluster = self.kmeans.predict([vector])[0]
            self.inverted_lists[cluster].append((i, vector))
    
    def search(self, query, k=5):
        cluster = self.kmeans.predict([query])[0]
        candidates = self.inverted_lists[cluster]
        similarities = [(i, np.dot(query, v) / (np.linalg.norm(query) * np.linalg.norm(v))) 
                        for i, v in candidates]
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

# Create and index vectors
vectors = np.random.rand(10000, 100)
index = IVFIndex(n_clusters=100)
index.build(vectors)

# Perform a search
query = np.random.rand(100)
results = index.search(query, k=5)

print("Search results:")
for i, similarity in results:
    print(f"Vector {i}: Similarity = {similarity:.4f}")
```

Slide 10: Vector Databases in Practice: FAISS

Facebook AI Similarity Search (FAISS) is a popular library for efficient similarity search. Let's use it to index and search vectors.

```python
import numpy as np
import faiss

# Generate sample data
d = 64  # Dimension of the vectors
nb = 100000  # Number of vectors in the database
nq = 10  # Number of queries

np.random.seed(1234)  # For reproducibility
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Build the index
index = faiss.IndexFlatL2(d)  # Use L2 distance
index.add(xb)  # Add vectors to the index

# Perform the search
k = 4  # Number of nearest neighbors
distances, indices = index.search(xq, k)

print(f"Searching for {nq} vectors in a database of {nb} vectors")
print("\nSearch results:")
for i in range(nq):
    print(f"Query {i}:")
    for j in range(k):
        print(f"  Neighbor {j}: Index {indices[i][j]}, Distance: {distances[i][j]:.4f}")
```

Slide 11: Combining Embeddings, Re-ranking, and Vector DBs

Let's create a simple document search system that uses embeddings, a vector database, and re-ranking.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a quick fox",
    "The lazy cat sleeps all day",
    "Quick foxes and lazy dogs are common in stories"
]

# Create embeddings
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(documents).toarray().astype('float32')

# Create FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Search function with re-ranking
def search(query, k=2):
    # Get query embedding
    query_embedding = vectorizer.transform([query]).toarray().astype('float32')
    
    # Perform initial search
    distances, indices = index.search(query_embedding, k)
    
    # Re-ranking: boost documents containing exact query terms
    query_terms = set(query.lower().split())
    reranked = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]
        score = 1 / (1 + distances[0][i])  # Convert distance to similarity score
        
        # Boost score for exact matches
        matches = sum(term in doc.lower() for term in query_terms)
        boosted_score = score * (1 + 0.1 * matches)
        
        reranked.append((idx, boosted_score))
    
    # Sort by boosted score
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked

# Perform a search
query = "quick animal"
results = search(query)

print(f"Search results for '{query}':")
for idx, score in results:
    print(f"Document: {documents[idx]}")
    print(f"Score: {score:.4f}\n")
```

Slide 12: Optimizing Vector Database Performance

To improve vector database performance, we can use techniques like product quantization. Let's implement this using FAISS.

```python
import numpy as np
import faiss

# Generate sample data
d = 64  # Dimension of the vectors
nb = 100000  # Number of vectors in the database
nq = 10  # Number of queries

np.random.seed(1234)  # For reproducibility
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# Create an index with product quantization
nlist = 100  # Number of clusters
m = 8  # Number of subquantizers
bits = 8  # Number of bits per subquantizer

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

# Train and add vectors
index.train(xb)
index.add(xb)

# Search
k = 4  # Number of nearest neighbors
index.nprobe = 10  # Number of clusters to visit during search
distances, indices = index.search(xq, k)

print("Search results:")
for i in range(nq):
    print(f"Query {i}:")
    for j in range(k):
        print(f"  Neighbor {j}: Index {indices[i][j]}, Distance: {distances[i][j]:.4f}")
```

Slide 13: Scalability in Vector Databases

As data grows, scalability becomes crucial. Distributed vector databases can help handle large-scale datasets efficiently.

```python
import numpy as np
from multiprocessing import Pool

class DistributedVectorDB:
    def __init__(self, num_shards):
        self.num_shards = num_shards
        self.shards = [[] for _ in range(num_shards)]
    
    def add(self, vector):
        shard_id = hash(tuple(vector)) % self.num_shards
        self.shards[shard_id].append(vector)
    
    def search(self, query, k=5):
        with Pool(self.num_shards) as p:
            results = p.starmap(self._search_shard, 
                                [(shard, query, k) for shard in self.shards])
        
        # Merge and sort results
        all_results = [item for sublist in results for item in sublist]
        return sorted(all_results, key=lambda x: x[1], reverse=True)[:k]
    
    @staticmethod
    def _search_shard(shard, query, k):
        return sorted([(i, np.dot(query, v)) for i, v in enumerate(shard)], 
                      key=lambda x: x[1], reverse=True)[:k]

# Usage example
db = DistributedVectorDB(num_shards=4)

# Add vectors
for _ in range(10000):
    db.add(np.random.rand(100))

# Search
query = np.random.rand(100)
results = db.search(query, k=5)

print("Search results:")
for idx, similarity in results:
    print(f"Vector {idx}: Similarity = {similarity:.4f}")
```

Slide 14: Continuous Learning in Vector Databases

Vector databases can benefit from continuous learning to adapt to changing data distributions. Here's a simple example of how this might work:

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class AdaptiveVectorDB:
    def __init__(self, dim, num_clusters):
        self.kmeans = MiniBatchKMeans(n_clusters=num_clusters)
        self.vectors = []
        self.cluster_centers = np.zeros((num_clusters, dim))
    
    def add(self, vector):
        self.vectors.append(vector)
        if len(self.vectors) % 1000 == 0:  # Update every 1000 vectors
            self._update_clusters()
    
    def _update_clusters(self):
        self.kmeans.partial_fit(self.vectors)
        self.cluster_centers = self.kmeans.cluster_centers_
    
    def search(self, query, k=5):
        distances = np.linalg.norm(self.cluster_centers - query, axis=1)
        nearest_cluster = np.argmin(distances)
        
        # Search within the nearest cluster
        cluster_vectors = [v for v, c in zip(self.vectors, self.kmeans.labels_) 
                           if c == nearest_cluster]
        similarities = [np.dot(query, v) / (np.linalg.norm(query) * np.linalg.norm(v)) 
                        for v in cluster_vectors]
        
        return sorted(zip(similarities, cluster_vectors), reverse=True)[:k]

# Usage
db = AdaptiveVectorDB(dim=100, num_clusters=10)

# Add vectors and perform search
for _ in range(5000):
    db.add(np.random.rand(100))

query = np.random.rand(100)
results = db.search(query)

print("Search results:")
for similarity, vector in results:
    print(f"Similarity: {similarity:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into embeddings, re-ranking, and vector databases, here are some valuable resources:

1. "Efficient Estimation of Word Representations in Vector Space" by Mikolov et al. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Global Vectors for Word Representation" by Pennington et al. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "Billion-scale similarity search with GPUs" by Johnson et al. (2017) ArXiv: [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734)
4. "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms" by Aumüller et al. (2017) ArXiv: [https://arxiv.org/abs/1807.05614](https://arxiv.org/abs/1807.05614)
5. "Learning to Rank: From Pairwise Approach to Listwise Approach" by Cao et al. (2007) ArXiv: [https://arxiv.org/abs/0803.2460](https://arxiv.org/abs/0803.2460)

These papers provide in-depth insights into the theories and practical applications of the concepts we've explored in this presentation.

