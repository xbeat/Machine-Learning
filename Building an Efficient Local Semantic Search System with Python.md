## Building an Efficient Local Semantic Search System with Python
Slide 1: Introduction to Local Semantic Search

Semantic search goes beyond keyword matching, understanding the intent and contextual meaning of a query. A local semantic search system processes and retrieves information from a local dataset, providing relevant results based on semantic similarity. This slideshow will guide you through building an efficient local semantic search system using Python.

```python
# Illustration of semantic search vs. keyword search
def keyword_search(query, documents):
    return [doc for doc in documents if query.lower() in doc.lower()]

def semantic_search(query, documents, model):
    query_embedding = model.encode(query)
    document_embeddings = [model.encode(doc) for doc in documents]
    similarities = [cosine_similarity([query_embedding], [doc_emb])[0][0] for doc_emb in document_embeddings]
    return [doc for _, doc in sorted(zip(similarities, documents), reverse=True)]

# Example usage
documents = ["The quick brown fox jumps over the lazy dog", "A fast orange fox leaps above a sleepy canine"]
query = "speedy animal jumps"

print("Keyword search results:", keyword_search(query, documents))
print("Semantic search results:", semantic_search(query, documents, sentence_transformer_model))
```

Slide 2: Setting Up the Environment

To build our local semantic search system, we'll use Python with some essential libraries. Let's set up our environment and install the required packages.

```python
# Install required packages
!pip install sentence-transformers faiss-cpu numpy

# Import necessary libraries
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Environment setup complete. Model loaded:", model)
```

Slide 3: Data Preparation

Before we can perform semantic search, we need to prepare our data. This involves loading the documents and creating embeddings for each document using our sentence transformer model.

```python
# Sample documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing deals with the interaction between computers and humans using natural language.",
    "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
]

# Create embeddings for documents
document_embeddings = model.encode(documents)

print("Document embeddings shape:", document_embeddings.shape)
print("First document embedding:", document_embeddings[0][:5])  # Show first 5 values
```

Slide 4: Building the FAISS Index

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. We'll use it to create an index for our document embeddings, allowing for fast retrieval.

```python
# Create a FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add document embeddings to the index
index.add(document_embeddings.astype('float32'))

print("FAISS index created with", index.ntotal, "vectors of dimension", dimension)
```

Slide 5: Implementing the Search Function

Now that we have our index, let's implement the search function. This function will take a query, convert it to an embedding, and find the most similar documents in our index.

```python
def semantic_search(query, index, model, documents, k=2):
    # Convert query to embedding
    query_embedding = model.encode([query])
    
    # Perform the search
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Return the results
    results = [
        (documents[i], float(distances[0][j]))
        for j, i in enumerate(indices[0])
    ]
    return results

# Example usage
query = "What is machine learning?"
results = semantic_search(query, index, model, documents)

print("Query:", query)
for doc, score in results:
    print(f"Score: {score:.4f}, Document: {doc}")
```

Slide 6: Improving Search Relevance

To improve search relevance, we can implement techniques like TF-IDF weighting or BM25 scoring. Let's implement TF-IDF weighting to give more importance to rare terms in our corpus.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

# Function to combine TF-IDF weights with semantic similarity
def improved_semantic_search(query, index, model, documents, tfidf, tfidf_matrix, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    query_tfidf = tfidf.transform([query])
    tfidf_scores = [query_tfidf.dot(tfidf_matrix[i].T).toarray()[0][0] for i in indices[0]]
    
    combined_scores = 1 / (1 + np.array(distances[0])) * np.array(tfidf_scores)
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    return [(documents[indices[0][i]], float(combined_scores[i])) for i in sorted_indices]

# Example usage
query = "What is machine learning?"
results = improved_semantic_search(query, index, model, documents, tfidf, tfidf_matrix)

print("Query:", query)
for doc, score in results:
    print(f"Score: {score:.4f}, Document: {doc}")
```

Slide 7: Handling Large Document Collections

When dealing with large document collections, we need to consider memory efficiency and search speed. Let's implement a chunking strategy to handle large datasets.

```python
def process_large_dataset(documents, chunk_size=1000):
    index = None
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i+chunk_size]
        chunk_embeddings = model.encode(chunk)
        
        if index is None:
            dimension = chunk_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
        
        index.add(chunk_embeddings.astype('float32'))
        
    return index

# Example usage with a larger dataset
large_documents = ["Document " + str(i) for i in range(10000)]
large_index = process_large_dataset(large_documents)

print("Large index created with", large_index.ntotal, "vectors")
```

Slide 8: Real-Life Example: Recipe Search Engine

Let's create a semantic search engine for recipes. This example demonstrates how our local semantic search system can be applied to a practical use case.

```python
recipes = [
    "Spaghetti Carbonara: pasta, eggs, cheese, bacon",
    "Chicken Stir Fry: chicken, vegetables, soy sauce",
    "Vegetable Soup: carrots, celery, onions, broth",
    "Chocolate Chip Cookies: flour, sugar, butter, chocolate chips",
    "Greek Salad: tomatoes, cucumbers, olives, feta cheese"
]

recipe_index = faiss.IndexFlatL2(model.encode(recipes[0]).shape[0])
recipe_index.add(model.encode(recipes).astype('float32'))

def search_recipes(query):
    results = semantic_search(query, recipe_index, model, recipes, k=2)
    print(f"Query: {query}")
    for recipe, score in results:
        print(f"Score: {score:.4f}, Recipe: {recipe}")

# Example searches
search_recipes("pasta dish")
search_recipes("healthy vegetarian meal")
```

Slide 9: Real-Life Example: Customer Support Chatbot

Another practical application of our local semantic search system is a customer support chatbot that can quickly find relevant answers to user queries.

```python
support_documents = [
    "To reset your password, click on the 'Forgot Password' link on the login page.",
    "Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "We offer free shipping on orders over $50.",
    "To track your order, log in to your account and go to the 'Order History' section.",
    "Our customer support team is available 24/7 via email and chat."
]

support_index = faiss.IndexFlatL2(model.encode(support_documents[0]).shape[0])
support_index.add(model.encode(support_documents).astype('float32'))

def chatbot_response(query):
    results = semantic_search(query, support_index, model, support_documents, k=1)
    return results[0][0]

# Example usage
print(chatbot_response("How do I change my password?"))
print(chatbot_response("What's your shipping policy?"))
```

Slide 10: Handling Multilingual Queries

To make our search system more versatile, let's extend it to handle multilingual queries using a multilingual model.

```python
from sentence_transformers import SentenceTransformer

# Load a multilingual model
multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

multilingual_documents = [
    "The weather is nice today.",
    "El tiempo está agradable hoy.",
    "Das Wetter ist heute schön.",
    "Le temps est beau aujourd'hui.",
    "今日の天気はいいです。"
]

multi_index = faiss.IndexFlatL2(multilingual_model.encode(multilingual_documents[0]).shape[0])
multi_index.add(multilingual_model.encode(multilingual_documents).astype('float32'))

def multilingual_search(query):
    results = semantic_search(query, multi_index, multilingual_model, multilingual_documents, k=2)
    print(f"Query: {query}")
    for doc, score in results:
        print(f"Score: {score:.4f}, Document: {doc}")

# Example searches in different languages
multilingual_search("How's the weather?")
multilingual_search("Wie ist das Wetter?")
multilingual_search("Quel temps fait-il?")
```

Slide 11: Optimizing Search Performance

To optimize search performance, especially for large datasets, we can use approximate nearest neighbor search techniques provided by FAISS.

```python
import time

# Create a large dataset for demonstration
large_documents = [f"Document {i}: This is a sample text for document number {i}." for i in range(100000)]

# Function to measure search time
def measure_search_time(index, query, k=10):
    start_time = time.time()
    distances, _ = index.search(model.encode([query]).astype('float32'), k)
    end_time = time.time()
    return end_time - start_time

# Create and compare different index types
dimension = model.encode(large_documents[0]).shape[0]

# Flat index (exact search)
flat_index = faiss.IndexFlatL2(dimension)

# IVF index (approximate search)
nlist = 100
quantizer = faiss.IndexFlatL2(dimension)
ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Train and add vectors to both indexes
vectors = model.encode(large_documents).astype('float32')
flat_index.add(vectors)
ivf_index.train(vectors)
ivf_index.add(vectors)

# Compare search times
query = "Sample query for performance testing"
flat_time = measure_search_time(flat_index, query)
ivf_time = measure_search_time(ivf_index, query)

print(f"Flat index search time: {flat_time:.4f} seconds")
print(f"IVF index search time: {ivf_time:.4f} seconds")
print(f"Speed improvement: {flat_time / ivf_time:.2f}x")
```

Slide 12: Implementing Semantic Clustering

Semantic clustering can help organize large document collections into meaningful groups. Let's implement k-means clustering on our document embeddings.

```python
from sklearn.cluster import KMeans

# Sample documents for clustering
cluster_documents = [
    "Python is a programming language",
    "Java is used for enterprise applications",
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing analyzes text",
    "Computer vision works with images",
    "JavaScript is used for web development",
    "Ruby on Rails is a web framework",
    "TensorFlow is a machine learning library",
    "PyTorch is used for deep learning"
]

# Create embeddings
cluster_embeddings = model.encode(cluster_documents)

# Perform k-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(cluster_embeddings)

# Display clustering results
for i, (doc, label) in enumerate(zip(cluster_documents, cluster_labels)):
    print(f"Document {i}: Cluster {label} - {doc}")

# Function to find the most representative document for each cluster
def get_cluster_representatives(embeddings, labels, documents):
    centroids = kmeans.cluster_centers_
    representatives = []
    for i in range(n_clusters):
        distances = np.linalg.norm(embeddings[labels == i] - centroids[i], axis=1)
        rep_index = np.argmin(distances)
        representatives.append(documents[np.where(labels == i)[0][rep_index]])
    return representatives

cluster_reps = get_cluster_representatives(cluster_embeddings, cluster_labels, cluster_documents)
print("\nCluster Representatives:")
for i, rep in enumerate(cluster_reps):
    print(f"Cluster {i}: {rep}")
```

Slide 13: Implementing Incremental Updates

In real-world applications, we often need to update our search index with new documents. Let's implement a function to add new documents to our existing index.

```python
def add_new_documents(index, model, existing_documents, new_documents):
    # Encode new documents
    new_embeddings = model.encode(new_documents)
    
    # Add new embeddings to the index
    index.add(new_embeddings.astype('float32'))
    
    # Update the list of documents
    updated_documents = existing_documents + new_documents
    
    return index, updated_documents

# Example usage
existing_index = faiss.IndexFlatL2(model.encode(documents[0]).shape[0])
existing_index.add(model.encode(documents).astype('float32'))

new_docs = [
    "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment.",
    "Quantum computing is the use of quantum phenomena such as superposition and entanglement to perform computation."
]

updated_index, updated_documents = add_new_documents(existing_index, model, documents, new_docs)

print(f"Original index size: {len(documents)}")
print(f"Updated index size: {len(updated_documents)}")

# Test search with a new document
query = "What is quantum computing?"
results = semantic_search(query, updated_index, model, updated_documents, k=1)
print(f"\nQuery: {query}")
print(f"Top result: {results[0][0]}")
```

Slide 14: Additional Resources

For further exploration of semantic search and related topics, consider the following resources:

1. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs" by Yu. A. Malkov and D. A. Yashunin (2018) ArXiv: [https://arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Jacob Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" by Nils Reimers and Iryna Gurevych (2019) ArXiv: [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
4. "

