## Retrieval Techniques in RAG Sparse and Dense Vector Tools
Slide 1: Retrieval Techniques in RAG: Sparse and Dense Vector Tools

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. This presentation focuses on two key retrieval techniques: sparse and dense vector tools, implemented in Python.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Example corpus
corpus = [
    "Sparse vectors in RAG",
    "Dense vectors for retrieval",
    "Combining sparse and dense techniques"
]

# Sparse vectorization (TF-IDF)
tfidf = TfidfVectorizer()
sparse_vectors = tfidf.fit_transform(corpus)

# Dense vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')
dense_vectors = model.encode(corpus)

print("Sparse vectors shape:", sparse_vectors.shape)
print("Dense vectors shape:", dense_vectors.shape)
```

Slide 2: Sparse Vector Representation: TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a popular sparse vector representation. It captures the importance of words in a document relative to a collection of documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "TF-IDF captures word importance",
    "Sparse vectors are efficient for large vocabularies",
    "TF-IDF is widely used in information retrieval"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()
print("First 5 features:", feature_names[:5])
```

Slide 3: TF-IDF Calculation

TF-IDF is calculated by multiplying term frequency (TF) with inverse document frequency (IDF). Let's break down the calculation for a single term.

```python
import numpy as np

def tf(term, doc):
    return doc.count(term) / len(doc.split())

def idf(term, docs):
    n_docs_with_term = sum(1 for doc in docs if term in doc)
    return np.log((len(docs) + 1) / (n_docs_with_term + 1)) + 1

def tfidf(term, doc, docs):
    return tf(term, doc) * idf(term, docs)

documents = [
    "This is a sample document",
    "Another example document",
    "Third document for demonstration"
]

term = "document"
doc = documents[0]

tf_value = tf(term, doc)
idf_value = idf(term, documents)
tfidf_value = tfidf(term, doc, documents)

print(f"TF: {tf_value:.4f}")
print(f"IDF: {idf_value:.4f}")
print(f"TF-IDF: {tfidf_value:.4f}")
```

Slide 4: Dense Vector Representation: Word Embeddings

Dense vector representations, like word embeddings, capture semantic relationships between words in a low-dimensional space. Popular models include Word2Vec, GloVe, and FastText.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    ["dense", "vectors", "capture", "semantic", "relationships"],
    ["word", "embeddings", "are", "useful", "for", "many", "nlp", "tasks"],
    ["vector", "representations", "in", "low", "dimensional", "space"]
]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
word_vector = model.wv['dense']

print("Vector shape:", word_vector.shape)
print("First 5 dimensions:", word_vector[:5])

# Find similar words
similar_words = model.wv.most_similar('vector', topn=3)
print("Words similar to 'vector':", similar_words)
```

Slide 5: Sentence Embeddings with Transformers

For document retrieval, we often need to represent entire sentences or paragraphs. Transformer-based models like BERT can generate dense vectors for longer text.

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample sentences
sentences = [
    "Sentence embeddings represent whole sentences.",
    "They capture context and meaning effectively.",
    "Transformer models excel at generating these embeddings."
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Embeddings shape:", embeddings.shape)
print("First sentence embedding (first 5 dimensions):", embeddings[0][:5])

# Calculate similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity between first two sentences: {similarity:.4f}")
```

Slide 6: Sparse Retrieval: Inverted Index

An inverted index is a data structure used for efficient retrieval in sparse vector spaces. It maps terms to the documents containing them.

```python
from collections import defaultdict

def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(documents):
        for term in doc.split():
            inverted_index[term].append(doc_id)
    return inverted_index

documents = [
    "sparse vectors in information retrieval",
    "efficient search using inverted index",
    "vector space model for document ranking"
]

index = build_inverted_index(documents)

# Query the index
query = "vectors in retrieval"
matching_docs = set()
for term in query.split():
    matching_docs.update(index.get(term, []))

print("Matching document IDs:", matching_docs)
```

Slide 7: Dense Retrieval: Approximate Nearest Neighbors

For dense vector retrieval, we often use approximate nearest neighbor (ANN) algorithms like Hierarchical Navigable Small World (HNSW) graphs.

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Generate random dense vectors
np.random.seed(42)
num_vectors = 1000
vector_dim = 128
vectors = np.random.rand(num_vectors, vector_dim)

# Build ANN index
ann_index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
ann_index.fit(vectors)

# Query vector
query_vector = np.random.rand(1, vector_dim)

# Find nearest neighbors
distances, indices = ann_index.kneighbors(query_vector)

print("Nearest neighbor indices:", indices[0])
print("Distances:", distances[0])
```

Slide 8: Hybrid Retrieval: Combining Sparse and Dense

Hybrid retrieval combines the strengths of both sparse and dense techniques. We can use ensemble methods or learn to rank approaches.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

documents = [
    "Hybrid retrieval combines sparse and dense techniques",
    "Ensemble methods improve search results",
    "Learning to rank optimizes retrieval performance"
]

# Sparse retrieval (TF-IDF)
tfidf = TfidfVectorizer()
sparse_vectors = tfidf.fit_transform(documents)

# Dense retrieval (Sentence Transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')
dense_vectors = model.encode(documents)

# Hybrid scoring function
def hybrid_score(query, doc_id, alpha=0.5):
    sparse_score = sparse_vectors[doc_id].dot(tfidf.transform([query]).T).toarray()[0][0]
    dense_score = np.dot(dense_vectors[doc_id], model.encode([query])[0])
    return alpha * sparse_score + (1 - alpha) * dense_score

query = "combining retrieval techniques"
scores = [hybrid_score(query, i) for i in range(len(documents))]

print("Hybrid retrieval scores:", scores)
print("Best matching document:", documents[np.argmax(scores)])
```

Slide 9: Real-Life Example: Document Search Engine

Implementing a simple document search engine using both sparse and dense retrieval techniques.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

class DocumentSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.tfidf = TfidfVectorizer()
        self.sparse_vectors = self.tfidf.fit_transform(documents)
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dense_vectors = self.dense_model.encode(documents)
    
    def search(self, query, k=3, alpha=0.5):
        sparse_query = self.tfidf.transform([query])
        dense_query = self.dense_model.encode([query])
        
        sparse_scores = self.sparse_vectors.dot(sparse_query.T).toarray().flatten()
        dense_scores = np.dot(self.dense_vectors, dense_query.T).flatten()
        
        hybrid_scores = alpha * sparse_scores + (1 - alpha) * dense_scores
        top_k = np.argsort(hybrid_scores)[::-1][:k]
        
        return [(self.documents[i], hybrid_scores[i]) for i in top_k]

# Example usage
documents = [
    "Python is a popular programming language",
    "Machine learning models require large datasets",
    "Natural language processing analyzes text data",
    "Deep learning architectures include neural networks",
    "Data scientists use various statistical techniques"
]

search_engine = DocumentSearchEngine(documents)
results = search_engine.search("programming languages for data science")

for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Slide 10: Real-Life Example: Recommendation System

Using dense vector representations to build a simple content-based recommendation system for articles.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class ArticleRecommender:
    def __init__(self, articles):
        self.articles = articles
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(articles)
    
    def recommend(self, user_history, n=3):
        user_embedding = self.model.encode(user_history)
        similarities = np.dot(self.embeddings, user_embedding.T).flatten()
        top_n = np.argsort(similarities)[::-1][:n]
        return [(self.articles[i], similarities[i]) for i in top_n]

# Example usage
articles = [
    "The impact of artificial intelligence on modern society",
    "Exploring the wonders of the deep ocean",
    "Advancements in renewable energy technologies",
    "The role of genetics in personalized medicine",
    "Space exploration: Past achievements and future goals"
]

recommender = ArticleRecommender(articles)
user_history = "I'm interested in technology and its effects on our world"

recommendations = recommender.recommend(user_history)

print("Recommended articles:")
for article, similarity in recommendations:
    print(f"Similarity: {similarity:.4f} - {article}")
```

Slide 11: Challenges in RAG Retrieval

Retrieval in RAG systems faces several challenges, including handling out-of-distribution queries, dealing with large-scale datasets, and maintaining up-to-date information.

```python
import numpy as np
from sklearn.preprocessing import normalize

class RAGRetriever:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = normalize(embeddings)  # Normalize for cosine similarity
    
    def retrieve(self, query_embedding, k=3, threshold=0.5):
        query_embedding = normalize(query_embedding.reshape(1, -1))
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_k = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k:
            if similarities[idx] >= threshold:
                results.append((self.documents[idx], similarities[idx]))
            else:
                break  # Stop if similarity is below threshold
        
        return results if results else [("No relevant documents found", 0)]

# Example usage
documents = [
    "Artificial intelligence and machine learning",
    "Climate change and global warming",
    "Quantum computing and cryptography",
    "Renewable energy sources and sustainability"
]
embeddings = np.random.rand(len(documents), 128)  # Simulated embeddings

retriever = RAGRetriever(documents, embeddings)
query_embedding = np.random.rand(128)  # Simulated query embedding

results = retriever.retrieve(query_embedding, threshold=0.7)

for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Slide 12: Evaluating Retrieval Performance

Assessing the quality of retrieval is crucial for RAG systems. Common metrics include precision, recall, and mean average precision (MAP).

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

def mean_average_precision(relevant_docs, retrieved_docs, k=10):
    if not relevant_docs:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            num_hits += 1
            score += num_hits / (i + 1)
    
    return score / len(relevant_docs)

# Example evaluation
relevant_docs = set([1, 3, 5, 7])
retrieved_docs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

precision = precision_score(
    [1 if doc in relevant_docs else 0 for doc in retrieved_docs],
    [1 if doc in relevant_docs else 0 for doc in range(1, 11)]
)

recall = recall_score(
    [1 if doc in relevant_docs else 0 for doc in retrieved_docs],
    [1 if doc in relevant_docs else 0 for doc in range(1, 11)]
)

map_score = mean_average_precision(relevant_docs, retrieved_docs)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Mean Average Precision: {map_score:.4f}")
```

Slide 13: Future Directions in RAG Retrieval

RAG retrieval continues to evolve with advancements in neural information retrieval, few-shot learning, and multi-modal retrieval techniques. These innovations aim to improve retrieval accuracy and efficiency in diverse scenarios.

```python
import numpy as np
from scipy.special import softmax

class AdvancedRetriever:
    def __init__(self, documents, text_embeddings, image_embeddings):
        self.documents = documents
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
    
    def multi_modal_retrieve(self, text_query, image_query, k=3):
        text_sim = np.dot(self.text_embeddings, text_query)
        image_sim = np.dot(self.image_embeddings, image_query)
        
        combined_sim = 0.7 * text_sim + 0.3 * image_sim
        top_k = np.argsort(combined_sim)[::-1][:k]
        
        return [(self.documents[i], combined_sim[i]) for i in top_k]

# Simulated data
documents = ["Doc1", "Doc2", "Doc3", "Doc4"]
text_embeddings = np.random.rand(4, 128)
image_embeddings = np.random.rand(4, 256)

retriever = AdvancedRetriever(documents, text_embeddings, image_embeddings)
text_query = np.random.rand(128)
image_query = np.random.rand(256)

results = retriever.multi_modal_retrieve(text_query, image_query)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Slide 14: Ethical Considerations in RAG Retrieval

As RAG systems become more prevalent, it's crucial to address ethical concerns such as bias in retrieval, privacy preservation, and the potential for misinformation amplification.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class EthicalRetriever:
    def __init__(self, documents, embeddings, sensitive_attributes):
        self.documents = documents
        self.embeddings = embeddings
        self.sensitive_attributes = sensitive_attributes
        self.scaler = StandardScaler()
        self.normalized_embeddings = self.scaler.fit_transform(embeddings)
    
    def fair_retrieve(self, query_embedding, k=3):
        query_embedding = self.scaler.transform(query_embedding.reshape(1, -1))
        similarities = np.dot(self.normalized_embeddings, query_embedding.T).flatten()
        
        # Apply fairness constraint
        fairness_scores = np.mean(self.sensitive_attributes, axis=1)
        adjusted_similarities = similarities * (1 - fairness_scores)
        
        top_k = np.argsort(adjusted_similarities)[::-1][:k]
        return [(self.documents[i], adjusted_similarities[i]) for i in top_k]

# Simulated data
documents = ["Doc1", "Doc2", "Doc3", "Doc4"]
embeddings = np.random.rand(4, 128)
sensitive_attributes = np.random.randint(0, 2, size=(4, 3))  # Binary attributes

retriever = EthicalRetriever(documents, embeddings, sensitive_attributes)
query_embedding = np.random.rand(1, 128)

results = retriever.fair_retrieve(query_embedding)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Slide 15: Additional Resources

For further exploration of RAG and vector retrieval techniques, consider the following resources:

1. "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020) ArXiv: [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
2. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020) ArXiv: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
3. "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020) ArXiv: [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
4. "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018) Available at: [https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

These papers provide in-depth discussions on various aspects of retrieval techniques in the context of language models and question-answering systems.

