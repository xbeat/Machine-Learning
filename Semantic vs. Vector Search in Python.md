## Semantic vs. Vector Search in Python
Slide 1: Understanding Semantic Search Fundamentals

Semantic search focuses on understanding the meaning and intent behind search queries rather than exact keyword matching. It leverages natural language processing techniques to comprehend context, synonyms, and relationships between words to deliver more relevant search results.

```python
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np

def semantic_similarity(query, document):
    # Tokenize query and document
    query_tokens = word_tokenize(query.lower())
    doc_tokens = word_tokenize(document.lower())
    
    similarities = []
    for q_token in query_tokens:
        token_sims = []
        q_synsets = wordnet.synsets(q_token)
        
        if not q_synsets:
            continue
            
        for d_token in doc_tokens:
            d_synsets = wordnet.synsets(d_token)
            if not d_synsets:
                continue
            
            # Calculate similarity between synsets
            sim = max([q_syn.path_similarity(d_syn) or 0 
                      for q_syn in q_synsets 
                      for d_syn in d_synsets])
            token_sims.append(sim)
            
        if token_sims:
            similarities.append(max(token_sims))
    
    return np.mean(similarities) if similarities else 0.0

# Example usage
query = "fast car"
document = "speedy automobile racing"
score = semantic_similarity(query, document)
print(f"Semantic similarity score: {score:.3f}")  # Output: ~0.8
```

Slide 2: Vector Search Architecture

Vector search transforms text into high-dimensional numerical vectors, enabling efficient similarity comparisons using distance metrics. This approach maps semantic relationships into geometric space, where closer vectors represent more similar concepts.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine

def vector_search_engine():
    # Sample document collection
    documents = [
        "machine learning algorithms",
        "deep neural networks",
        "artificial intelligence systems",
        "data science applications"
    ]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform documents to vectors
    doc_vectors = vectorizer.fit_transform(documents)
    
    def search(query):
        # Transform query to vector
        query_vector = vectorizer.transform([query])
        
        # Calculate cosine similarity with all documents
        similarities = []
        for doc_vector in doc_vectors:
            similarity = 1 - cosine(query_vector.toarray()[0], 
                                  doc_vector.toarray()[0])
            similarities.append(similarity)
        
        # Return ranked results
        ranked_results = sorted(zip(similarities, documents), 
                              reverse=True)
        return ranked_results
    
    return search

# Initialize search engine
search_engine = vector_search_engine()

# Example search
results = search_engine("neural networks")
for score, doc in results:
    print(f"Score: {score:.3f} - Document: {doc}")
```

Slide 3: Semantic Search with BERT Embeddings

Building upon transformer architectures, BERT embeddings capture deep contextual relationships in text, producing rich semantic representations that significantly outperform traditional word-based approaches in understanding meaning.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class BERTSemanticSearch:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
    def get_embeddings(self, text):
        # Tokenize and convert to tensor
        inputs = self.tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding as sentence representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def semantic_search(self, query, documents):
        query_embedding = self.get_embeddings(query)
        doc_embeddings = [self.get_embeddings(doc) for doc in documents]
        
        # Calculate cosine similarities
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = np.dot(query_embedding[0], doc_emb[0]) / \
                        (np.linalg.norm(query_embedding[0]) * \
                         np.linalg.norm(doc_emb[0]))
            similarities.append(similarity)
        
        return similarities

# Example usage
searcher = BERTSemanticSearch()
documents = [
    "Advanced machine learning techniques",
    "Natural language processing systems",
    "Computer vision applications"
]

query = "AI algorithms"
scores = searcher.semantic_search(query, documents)
for score, doc in zip(scores, documents):
    print(f"Score: {score:.3f} - Document: {doc}")
```

Slide 4: Implementing Approximate Nearest Neighbors (ANN)

Approximate Nearest Neighbors optimization enables efficient vector similarity search in high-dimensional spaces by trading exact precision for dramatic speed improvements. This technique is crucial for scaling semantic search to large datasets.

```python
import numpy as np
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer

class ANNSearchEngine:
    def __init__(self, vector_size=100, n_trees=10):
        self.vector_size = vector_size
        self.index = AnnoyIndex(vector_size, 'angular')
        self.n_trees = n_trees
        self.vectorizer = TfidfVectorizer(max_features=vector_size)
        self.documents = []
        
    def add_documents(self, documents):
        self.documents = documents
        vectors = self.vectorizer.fit_transform(documents).toarray()
        
        # Build ANN index
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)
            
        self.index.build(self.n_trees)
        
    def search(self, query, n_results=5):
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        nearest_ids = self.index.get_nns_by_vector(
            query_vector, n_results, include_distances=True
        )
        
        results = [
            (dist, self.documents[idx]) 
            for idx, dist in zip(*nearest_ids)
        ]
        return results

# Example usage
engine = ANNSearchEngine()
docs = [
    "quantum computing advances",
    "machine learning applications",
    "database optimization techniques",
    "artificial neural networks",
    "distributed systems architecture"
]

engine.add_documents(docs)
results = engine.search("ML and AI", n_results=3)
for dist, doc in results:
    print(f"Distance: {dist:.3f} - Document: {doc}")
```

Slide 5: Dense Passage Retrieval Implementation

Dense Passage Retrieval represents a modern approach to semantic search by using neural networks to encode both queries and documents into a shared dense vector space, enabling more nuanced semantic matching.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        # Mean pooling over sequence length
        pooled = torch.mean(embedded, dim=1)
        return self.encoder(pooled)

class DPRSearchEngine:
    def __init__(self, vocab_size):
        self.query_encoder = DenseEncoder(vocab_size)
        self.passage_encoder = DenseEncoder(vocab_size)
        
    def encode_query(self, query_tokens):
        with torch.no_grad():
            return self.query_encoder(query_tokens)
            
    def encode_passage(self, passage_tokens):
        with torch.no_grad():
            return self.passage_encoder(passage_tokens)
            
    def similarity(self, query_vec, passage_vec):
        return F.cosine_similarity(query_vec, passage_vec)

# Training loop example
def train_step(query_batch, positive_passages, negative_passages, 
               optimizer, temperature=0.1):
    q_vecs = query_encoder(query_batch)
    pos_vecs = passage_encoder(positive_passages)
    neg_vecs = passage_encoder(negative_passages)
    
    # Compute similarities
    pos_scores = F.cosine_similarity(q_vecs, pos_vecs)
    neg_scores = F.cosine_similarity(q_vecs, neg_vecs)
    
    # Contrastive loss
    logits = torch.cat([pos_scores, neg_scores]) / temperature
    labels = torch.zeros(len(logits))
    labels[:len(pos_scores)] = 1
    
    loss = F.cross_entropy(logits, labels)
    return loss
```

Slide 6: Hybrid Search Architecture

Hybrid search combines the strengths of both semantic and keyword-based approaches to achieve optimal search results. This implementation demonstrates how to merge BM25 keyword scores with semantic similarity scores.

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

class HybridSearchEngine:
    def __init__(self, semantic_weight=0.7):
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1 - semantic_weight
        self.scaler = MinMaxScaler()
        self.bert_search = BERTSemanticSearch()  # From previous slide
        
    def preprocess_documents(self, documents):
        # Tokenize documents for BM25
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.documents = documents
        
    def search(self, query, top_k=5):
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get semantic scores
        semantic_scores = self.bert_search.semantic_search(
            query, self.documents
        )
        
        # Normalize scores
        bm25_scores = self.scaler.fit_transform(
            bm25_scores.reshape(-1, 1)
        ).flatten()
        semantic_scores = self.scaler.fit_transform(
            np.array(semantic_scores).reshape(-1, 1)
        ).flatten()
        
        # Combine scores
        final_scores = (self.semantic_weight * semantic_scores + 
                       self.keyword_weight * bm25_scores)
        
        # Get top results
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results = [(final_scores[i], self.documents[i]) 
                  for i in top_indices]
        
        return results

# Example usage
documents = [
    "advanced machine learning techniques",
    "neural network architectures",
    "deep learning applications",
    "artificial intelligence systems",
    "computer vision algorithms"
]

engine = HybridSearchEngine()
engine.preprocess_documents(documents)
results = engine.search("AI and neural networks")

for score, doc in results:
    print(f"Score: {score:.3f} - Document: {doc}")
```

Slide 7: Evaluation Metrics for Search Systems

Modern search systems require comprehensive evaluation using multiple metrics to assess both relevance and ranking quality. This implementation demonstrates key evaluation metrics including Mean Average Precision (MAP), Normalized Discounted Cumulative Gain (NDCG), and Mean Reciprocal Rank (MRR).

```python
import numpy as np
from typing import List, Dict

class SearchEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def calculate_precision_at_k(self, 
                               relevant_docs: set, 
                               retrieved_docs: List, 
                               k: int) -> float:
        """Calculate precision@k"""
        retrieved_k = retrieved_docs[:k]
        relevant_retrieved = len(
            set(retrieved_k).intersection(relevant_docs)
        )
        return relevant_retrieved / k if k > 0 else 0.0
    
    def calculate_ndcg(self, 
                      relevance_scores: Dict, 
                      retrieved_docs: List, 
                      k: int) -> float:
        """Calculate NDCG@k"""
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, doc in enumerate(retrieved_docs[:k]):
            rel = relevance_scores.get(doc, 0)
            dcg += (2 ** rel - 1) / np.log2(i + 2)
        
        # Calculate IDCG
        ideal_scores = sorted(
            relevance_scores.values(), reverse=True
        )[:k]
        for i, rel in enumerate(ideal_scores):
            idcg += (2 ** rel - 1) / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, 
                     relevant_docs: set, 
                     retrieved_docs: List) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate(self, 
                relevant_docs: set, 
                retrieved_docs: List, 
                relevance_scores: Dict = None) -> Dict:
        """Evaluate search results using multiple metrics"""
        if relevance_scores is None:
            relevance_scores = {doc: 1 for doc in relevant_docs}
            
        metrics = {
            'precision@1': self.calculate_precision_at_k(
                relevant_docs, retrieved_docs, 1
            ),
            'precision@5': self.calculate_precision_at_k(
                relevant_docs, retrieved_docs, 5
            ),
            'ndcg@5': self.calculate_ndcg(
                relevance_scores, retrieved_docs, 5
            ),
            'mrr': self.calculate_mrr(
                relevant_docs, retrieved_docs
            )
        }
        
        return metrics

# Example usage
evaluator = SearchEvaluator()

# Sample data
retrieved_docs = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
relevant_docs = {'doc1', 'doc3', 'doc5'}
relevance_scores = {
    'doc1': 3,
    'doc2': 0,
    'doc3': 2,
    'doc4': 0,
    'doc5': 1
}

results = evaluator.evaluate(
    relevant_docs, 
    retrieved_docs, 
    relevance_scores
)

for metric, score in results.items():
    print(f"{metric}: {score:.3f}")
```

Slide 8: Cross-Encoder Reranking System

Cross-encoders provide superior accuracy in semantic matching by directly comparing query-document pairs, making them ideal for reranking initial search results, though computationally intensive for large-scale retrieval.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
    def rerank(self, 
               query: str, 
               documents: List[str], 
               top_k: int = None) -> List[Tuple[float, str]]:
        # Prepare input pairs
        pairs = [[query, doc] for doc in documents]
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        # Get relevance scores
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1)
            scores = torch.sigmoid(scores).cpu().numpy()
        
        # Sort documents by score
        ranked_results = sorted(
            zip(scores, documents), 
            reverse=True
        )
        
        if top_k:
            ranked_results = ranked_results[:top_k]
            
        return ranked_results

# Example usage
reranker = CrossEncoderReranker()

query = "deep learning architectures"
documents = [
    "Neural network designs for machine learning",
    "Database management systems",
    "Advanced deep learning model architectures",
    "Web development frameworks",
    "Transformer models in deep learning"
]

reranked_results = reranker.rerank(query, documents, top_k=3)

print("\nReranked results:")
for score, doc in reranked_results:
    print(f"Score: {score:.3f} - {doc}")
```

Slide 9: Elastic Search Integration with Semantic Search

Elastic Search provides powerful text search capabilities that can be enhanced with semantic search functionality. This implementation demonstrates how to combine Elasticsearch's inverted index with dense vector search for hybrid retrieval.

```python
from elasticsearch import Elasticsearch
import numpy as np
from sentence_transformers import SentenceTransformer

class ElasticSemanticSearch:
    def __init__(self, index_name="semantic_search"):
        self.es = Elasticsearch()
        self.index_name = index_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_index(self):
        settings = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        self.es.indices.create(
            index=self.index_name, 
            body=settings
        )
        
    def index_documents(self, documents):
        for i, doc in enumerate(documents):
            vector = self.model.encode(doc)
            doc_with_vector = {
                'text': doc,
                'vector': vector.tolist()
            }
            self.es.index(
                index=self.index_name,
                id=i,
                body=doc_with_vector
            )
        self.es.indices.refresh(index=self.index_name)
        
    def search(self, query, k=5):
        query_vector = self.model.encode(query)
        
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vector.tolist()}
                }
            }
        }
        
        response = self.es.search(
            index=self.index_name,
            body={
                "size": k,
                "query": script_query,
                "_source": ["text"]
            }
        )
        
        return [
            (hit["_score"], hit["_source"]["text"]) 
            for hit in response["hits"]["hits"]
        ]

# Example usage
searcher = ElasticSemanticSearch()
documents = [
    "Advanced machine learning techniques",
    "Natural language processing methods",
    "Deep neural network architectures",
    "Computer vision applications",
    "Reinforcement learning algorithms"
]

# Index documents
searcher.create_index()
searcher.index_documents(documents)

# Perform search
results = searcher.search("AI and neural networks", k=3)
for score, text in results:
    print(f"Score: {score:.3f} - {text}")
```

Slide 10: Query Expansion Using Word Embeddings

Query expansion enhances search accuracy by including semantically related terms in the original query. This implementation uses word embeddings to identify and add relevant terms to expand the search context.

```python
import gensim.downloader as api
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import List, Set

class QueryExpander:
    def __init__(self, 
                 model_name: str = 'word2vec-google-news-300'):
        self.word_vectors = api.load(model_name)
        self.stop_words = set(stopwords.words('english'))
        
    def get_similar_terms(self, 
                         word: str, 
                         n_terms: int = 3) -> List[str]:
        try:
            similar_terms = self.word_vectors.most_similar(
                word, topn=n_terms
            )
            return [term for term, _ in similar_terms]
        except KeyError:
            return []
    
    def expand_query(self, 
                    query: str, 
                    n_terms_per_word: int = 2,
                    min_word_len: int = 3) -> str:
        # Tokenize and filter query
        tokens = word_tokenize(query.lower())
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words 
            and len(token) >= min_word_len
        ]
        
        # Expand each token
        expanded_terms = set(filtered_tokens)
        for token in filtered_tokens:
            similar_terms = self.get_similar_terms(
                token, n_terms_per_word
            )
            expanded_terms.update(similar_terms)
        
        # Combine original and expanded terms
        expanded_query = ' '.join(list(expanded_terms))
        return expanded_query
    
    def expand_queries_batch(self, 
                           queries: List[str]) -> List[str]:
        return [self.expand_query(q) for q in queries]

# Example usage
expander = QueryExpander()

# Single query expansion
query = "machine learning algorithms"
expanded_query = expander.expand_query(query)
print(f"Original query: {query}")
print(f"Expanded query: {expanded_query}")

# Batch query expansion
queries = [
    "neural networks",
    "computer vision",
    "natural language processing"
]
expanded_queries = expander.expand_queries_batch(queries)
for original, expanded in zip(queries, expanded_queries):
    print(f"\nOriginal: {original}")
    print(f"Expanded: {expanded}")
```

Slide 11: Real-time Search Results Aggregation

Real-time search aggregation combines results from multiple search backends while maintaining high performance. This implementation demonstrates how to merge and rank results from different search engines with weighted scoring.

```python
import asyncio
from typing import List, Dict, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class MultiSearchAggregator:
    def __init__(self, search_weights: Dict[str, float] = None):
        self.search_weights = search_weights or {
            'semantic': 0.4,
            'keyword': 0.3,
            'vector': 0.3
        }
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    async def search_engine_a(self, query: str) -> List[Tuple[str, float]]:
        # Simulate semantic search
        await asyncio.sleep(0.1)  # Simulated latency
        return [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]
    
    async def search_engine_b(self, query: str) -> List[Tuple[str, float]]:
        # Simulate keyword search
        await asyncio.sleep(0.05)  # Simulated latency
        return [("doc2", 0.8), ("doc3", 0.6), ("doc4", 0.4)]
    
    async def search_engine_c(self, query: str) -> List[Tuple[str, float]]:
        # Simulate vector search
        await asyncio.sleep(0.15)  # Simulated latency
        return [("doc1", 0.85), ("doc4", 0.75), ("doc2", 0.65)]
    
    def normalize_scores(self, 
                        results: List[Tuple[str, float]]) -> Dict[str, float]:
        if not results:
            return {}
        scores = np.array([score for _, score in results])
        min_score, max_score = scores.min(), scores.max()
        range_score = max_score - min_score
        
        normalized = {}
        for doc, score in results:
            if range_score > 0:
                normalized[doc] = (score - min_score) / range_score
            else:
                normalized[doc] = 1.0
        return normalized
    
    async def aggregate_search_results(self, 
                                     query: str, 
                                     top_k: int = 5) -> List[Tuple[str, float]]:
        # Execute searches concurrently
        searches = [
            self.search_engine_a(query),
            self.search_engine_b(query),
            self.search_engine_c(query)
        ]
        
        results = await asyncio.gather(*searches)
        
        # Normalize and combine scores
        combined_scores = {}
        for engine_results, (engine, weight) in zip(
            results, self.search_weights.items()):
            normalized = self.normalize_scores(engine_results)
            for doc, score in normalized.items():
                if doc not in combined_scores:
                    combined_scores[doc] = 0
                combined_scores[doc] += score * weight
        
        # Sort and return top results
        ranked_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return ranked_results[:top_k]

# Example usage
async def main():
    aggregator = MultiSearchAggregator()
    query = "machine learning optimization"
    results = await aggregator.aggregate_search_results(query)
    
    print(f"Results for query: {query}")
    for doc_id, score in results:
        print(f"Document: {doc_id}, Score: {score:.3f}")

# Run the example
import asyncio
asyncio.run(main())
```

Slide 12: Contextual Search Personalization

Contextual search personalization adapts search results based on user behavior and preferences. This implementation demonstrates how to incorporate user context and historical interactions into search rankings.

```python
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

class ContextualSearchPersonalizer:
    def __init__(self):
        self.user_profiles = defaultdict(lambda: {
            'interests': defaultdict(float),
            'click_history': defaultdict(int),
            'search_history': []
        })
        
    def update_user_profile(self, 
                          user_id: str, 
                          clicked_doc: str, 
                          doc_topics: Dict[str, float]):
        # Update click history
        self.user_profiles[user_id]['click_history'][clicked_doc] += 1
        
        # Update interest scores
        for topic, score in doc_topics.items():
            current_score = self.user_profiles[user_id]['interests'][topic]
            self.user_profiles[user_id]['interests'][topic] = \
                0.9 * current_score + 0.1 * score
    
    def get_personalization_score(self, 
                                user_id: str, 
                                doc_id: str, 
                                doc_topics: Dict[str, float]) -> float:
        if user_id not in self.user_profiles:
            return 1.0
            
        profile = self.user_profiles[user_id]
        
        # Calculate topic similarity
        topic_score = 0.0
        for topic, score in doc_topics.items():
            topic_score += score * profile['interests'].get(topic, 0)
            
        # Calculate historical interaction score
        history_score = np.log1p(
            profile['click_history'].get(doc_id, 0)
        )
        
        # Combine scores
        final_score = (0.7 * topic_score + 0.3 * history_score)
        return final_score
    
    def personalize_results(self, 
                          user_id: str, 
                          search_results: List[Tuple[str, float]], 
                          doc_topics: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        if not search_results:
            return []
            
        personalized_scores = []
        for doc_id, base_score in search_results:
            personalization_score = self.get_personalization_score(
                user_id, doc_id, doc_topics.get(doc_id, {})
            )
            final_score = 0.7 * base_score + 0.3 * personalization_score
            personalized_scores.append((doc_id, final_score))
            
        return sorted(personalized_scores, 
                     key=lambda x: x[1], 
                     reverse=True)

# Example usage
personalizer = ContextualSearchPersonalizer()

# Sample data
user_id = "user123"
doc_topics = {
    "doc1": {"machine_learning": 0.8, "neural_networks": 0.6},
    "doc2": {"data_science": 0.7, "statistics": 0.5},
    "doc3": {"deep_learning": 0.9, "computer_vision": 0.4}
}

search_results = [
    ("doc1", 0.9),
    ("doc2", 0.7),
    ("doc3", 0.5)
]

# Update user profile with some interactions
personalizer.update_user_profile(
    user_id, "doc1", doc_topics["doc1"]
)

# Get personalized results
personalized_results = personalizer.personalize_results(
    user_id, search_results, doc_topics
)

print("Personalized Search Results:")
for doc_id, score in personalized_results:
    print(f"Document: {doc_id}, Score: {score:.3f}")
```

Slide 13: Performance Optimization with Caching

Search performance optimization through intelligent caching strategies is crucial for large-scale systems. This implementation demonstrates a multi-level cache system with TTL and LRU eviction policies for query results and vector embeddings.

```python
from collections import OrderedDict
from typing import Any, Optional
import time
import numpy as np

class MultiLevelCache:
    def __init__(self, 
                 max_size: int = 1000, 
                 ttl: int = 3600):
        self.query_cache = TTLCache(max_size, ttl)
        self.embedding_cache = LRUCache(max_size)
        self.result_cache = TTLCache(max_size, ttl // 2)
        
    def get_query_cache(self, query: str) -> Optional[Any]:
        return self.query_cache.get(query)
        
    def set_query_cache(self, 
                       query: str, 
                       results: Any) -> None:
        self.query_cache.set(query, results)
        
    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        return self.embedding_cache.get(key)
        
    def set_embedding(self, 
                     key: str, 
                     embedding: np.ndarray) -> None:
        self.embedding_cache.set(key, embedding)
        
class TTLCache:
    def __init__(self, max_size: int, ttl: int):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        
    def _evict_expired(self) -> None:
        current_time = time.time()
        expired_keys = [
            k for k, ts in self.timestamps.items() 
            if current_time - ts > self.ttl
        ]
        for k in expired_keys:
            self.cache.pop(k, None)
            self.timestamps.pop(k, None)
            
    def get(self, key: str) -> Optional[Any]:
        self._evict_expired()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        self._evict_expired()
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            self.cache.pop(oldest)
            self.timestamps.pop(oldest)
            
        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.cache.move_to_end(key)
        
class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = OrderedDict()
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = value
        self.cache.move_to_end(key)

# Example usage
cache = MultiLevelCache(max_size=1000, ttl=3600)

# Cache query results
query = "machine learning algorithms"
results = ["doc1", "doc2", "doc3"]
cache.set_query_cache(query, results)

# Cache embeddings
doc_id = "doc1"
embedding = np.random.rand(768)
cache.set_embedding(doc_id, embedding)

# Retrieve from cache
cached_results = cache.get_query_cache(query)
cached_embedding = cache.get_embedding(doc_id)

print(f"Cached results found: {cached_results is not None}")
print(f"Cached embedding found: {cached_embedding is not None}")
```

Slide 14: Additional Resources

*   Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
    *   [https://arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
*   Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
    *   [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
*   Xiong, L., et al. (2020). Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval
    *   [https://arxiv.org/abs/2007.00808](https://arxiv.org/abs/2007.00808)
*   Lin, J., et al. (2021). Pretrained Transformers for Text Ranking: BERT and Beyond
    *   [https://arxiv.org/abs/2010.06467](https://arxiv.org/abs/2010.06467)
*   Suggested searches for implementation details:
    *   "Dense Passage Retrieval implementation"
    *   "Semantic search architecture best practices"
    *   "Vector similarity search optimization"
    *   "Hybrid search systems design"

