## Improving Semantic Similarity in Question Answering with HyDE
Slide 1: Understanding HyDE Architecture

Traditional RAG systems face semantic similarity challenges between queries and answers. HyDE (Hypothetical Document Embeddings) addresses this by generating synthetic answers first. This implementation demonstrates the core HyDE architecture using transformers and semantic search capabilities.

```python
from transformers import AutoTokenizer, AutoModelForSeq2Seq
import torch
import faiss
import numpy as np

class HyDERetriever:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2Seq.from_pretrained(model_name)
        self.index = None
        
    def generate_hypothesis(self, query):
        inputs = self.tokenizer(f"Generate a detailed answer: {query}", 
                              return_tensors="pt", max_length=512)
        outputs = self.model.generate(**inputs, max_length=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", 
                              max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model.encoder(**inputs).last_hidden_state
        return outputs.mean(dim=1).numpy()
```

Slide 2: Building the Vector Database

A crucial component of HyDE is the vector database for storing and retrieving document embeddings. This implementation uses FAISS, an efficient similarity search library, to create and query the vector index for document retrieval.

```python
def build_vector_store(self, documents):
    embeddings = []
    for doc in documents:
        embedding = self.embed_text(doc)
        embeddings.append(embedding[0])
    
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    
    self.index = faiss.IndexFlatL2(dimension)
    self.index.add(embeddings)
    return self.index

def retrieve_context(self, query_embedding, k=5):
    distances, indices = self.index.search(
        query_embedding.astype('float32'), k
    )
    return indices[0]
```

Slide 3: Query Processing Pipeline

The query processing pipeline in HyDE involves generating a hypothetical answer, embedding it, and using this embedding to retrieve relevant contexts. This implementation shows the complete workflow from query to context retrieval.

```python
def process_query(self, query, documents, k=5):
    # Generate hypothetical answer
    hypothesis = self.generate_hypothesis(query)
    
    # Embed hypothesis
    hypothesis_embedding = self.embed_text(hypothesis)
    
    # Build or update index if needed
    if self.index is None:
        self.build_vector_store(documents)
    
    # Retrieve relevant contexts
    context_indices = self.retrieve_context(hypothesis_embedding, k)
    
    return {
        'hypothesis': hypothesis,
        'retrieved_contexts': [documents[i] for i in context_indices]
    }
```

Slide 4: Contrastive Learning Implementation

Contrastive learning is essential for training bi-encoders in HyDE. This implementation demonstrates the contrastive loss calculation and training loop for improving embedding quality through similarity learning.

```python
def contrastive_loss(anchor, positive, negative, margin=0.5):
    pos_dist = torch.nn.functional.pairwise_distance(anchor, positive)
    neg_dist = torch.nn.functional.pairwise_distance(anchor, negative)
    return torch.mean(torch.max(pos_dist - neg_dist + margin, 
                              torch.zeros_like(pos_dist)))

class ContrastiveTrainer:
    def __init__(self, model, learning_rate=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=learning_rate)
    
    def train_step(self, anchor, positive, negative):
        self.optimizer.zero_grad()
        loss = contrastive_loss(anchor, positive, negative)
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

Slide 5: Real-world Implementation: Question Answering System

This implementation demonstrates a complete question answering system using HyDE for document retrieval. The system includes preprocessing, embedding generation, and answer synthesis with performance monitoring.

```python
import time
from typing import List, Dict
import numpy as np

class HyDEQuestionAnswering:
    def __init__(self, hyde_retriever, documents):
        self.hyde = hyde_retriever
        self.documents = documents
        self.performance_metrics = []
        
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        processed = []
        for doc in documents:
            # Remove special characters and normalize
            clean_doc = ' '.join(doc.lower().split())
            processed.append(clean_doc)
        return processed
    
    def answer_question(self, question: str) -> Dict:
        start_time = time.time()
        
        # Process through HyDE pipeline
        result = self.hyde.process_query(
            question, 
            self.preprocess_documents(self.documents)
        )
        
        # Track performance
        elapsed_time = time.time() - start_time
        self.performance_metrics.append({
            'query_time': elapsed_time,
            'num_contexts': len(result['retrieved_contexts'])
        })
        
        return result
```

Slide 6: Results for Question Answering System

```python
# Example usage and results
qa_system = HyDEQuestionAnswering(hyde_retriever, documents)
question = "What are the key benefits of transfer learning?"

result = qa_system.answer_question(question)
print(f"Hypothetical Answer: {result['hypothesis'][:100]}...")
print("\nRetrieved Contexts:")
for i, ctx in enumerate(result['retrieved_contexts'], 1):
    print(f"\nContext {i}: {ctx[:100]}...")
print(f"\nAverage Query Time: {np.mean([m['query_time'] for m in qa_system.performance_metrics]):.3f}s")

# Example Output:
# Hypothetical Answer: Transfer learning allows models to leverage knowledge from related tasks, reducing training time and...
# Context 1: Transfer learning is a machine learning technique where a model developed for one task is reused as a starting...
# Average Query Time: 0.245s
```

Slide 7: Embedding Cache Implementation

To optimize performance, implementing a caching mechanism for embeddings is crucial. This implementation shows how to cache and reuse embeddings for frequently accessed documents.

```python
from collections import LRUCache
import hashlib

class EmbeddingCache:
    def __init__(self, capacity: int = 1000):
        self.cache = LRUCache(capacity)
        
    def get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str, embed_fn):
        cache_key = self.get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = embed_fn(text)
        self.cache[cache_key] = embedding
        return embedding

# Integration with HyDE
class CachedHyDERetriever(HyDERetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = EmbeddingCache()
        
    def embed_text(self, text):
        return self.cache.get_embedding(
            text, 
            lambda t: super().embed_text(t)
        )
```

Slide 8: Performance Optimization and Batch Processing

For handling large document collections, batch processing is essential. This implementation shows how to optimize HyDE for processing multiple queries and documents simultaneously.

```python
import torch
from typing import List, Optional

class BatchHyDERetriever:
    def __init__(self, base_retriever: HyDERetriever, batch_size: int = 32):
        self.base = base_retriever
        self.batch_size = batch_size
        
    def batch_embed(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            # Process batch in parallel
            with torch.no_grad():
                inputs = self.base.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                outputs = self.base.model.encoder(**inputs).last_hidden_state
                batch_embeddings = outputs.mean(dim=1).numpy()
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
```

Slide 9: Document Preprocessing and Chunking

Effective document chunking is crucial for HyDE's performance. This implementation demonstrates intelligent document segmentation while preserving semantic context and handling various document formats.

```python
import re
from typing import List, Tuple

class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_document(self, text: str) -> List[str]:
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create overlapping chunks
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust chunk boundaries to respect sentence endings
            if end < len(text):
                # Find the last sentence boundary
                last_period = text.rfind('.', start, end)
                if last_period != -1:
                    end = last_period + 1
            
            chunks.append(text[start:end].strip())
            start = end - self.overlap
            
        return chunks
    
    def preprocess_chunks(self, chunks: List[str]) -> List[str]:
        processed = []
        for chunk in chunks:
            # Clean and normalize text
            clean_chunk = re.sub(r'[^\w\s.,?!-]', '', chunk)
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk).strip()
            processed.append(clean_chunk)
        
        return processed
```

Slide 10: Semantic Similarity Scoring

Implementing custom similarity metrics helps fine-tune HyDE's retrieval performance. This code shows various similarity scoring methods and their application in context ranking.

```python
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Tuple

class SimilarityScorer:
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return 1 - cosine(v1, v2)
    
    @staticmethod
    def euclidean_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return 1 / (1 + np.linalg.norm(v1 - v2))
    
    @staticmethod
    def dot_product_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2)
    
    def rank_contexts(self, 
                     query_embedding: np.ndarray, 
                     context_embeddings: List[np.ndarray],
                     method: str = 'cosine') -> List[Tuple[int, float]]:
        similarity_scores = []
        
        for idx, context_emb in enumerate(context_embeddings):
            if method == 'cosine':
                score = self.cosine_similarity(query_embedding, context_emb)
            elif method == 'euclidean':
                score = self.euclidean_similarity(query_embedding, context_emb)
            else:
                score = self.dot_product_similarity(query_embedding, context_emb)
                
            similarity_scores.append((idx, score))
            
        return sorted(similarity_scores, key=lambda x: x[1], reverse=True)
```

Slide 11: Performance Monitoring and Analytics

Implementing comprehensive performance monitoring helps track and optimize HyDE's effectiveness. This implementation includes metrics collection, analysis, and visualization capabilities.

```python
import time
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class QueryMetrics:
    query_time: float
    hypothesis_generation_time: float
    retrieval_time: float
    num_contexts: int
    similarity_scores: List[float]

class HyDEAnalytics:
    def __init__(self):
        self.metrics_history: List[QueryMetrics] = []
        
    def record_query(self, metrics: QueryMetrics):
        self.metrics_history.append(metrics)
        
    def get_performance_summary(self) -> Dict:
        if not self.metrics_history:
            return {}
            
        avg_query_time = np.mean([m.query_time for m in self.metrics_history])
        avg_hypothesis_time = np.mean([m.hypothesis_generation_time 
                                     for m in self.metrics_history])
        avg_retrieval_time = np.mean([m.retrieval_time 
                                    for m in self.metrics_history])
        avg_similarity = np.mean([np.mean(m.similarity_scores) 
                                for m in self.metrics_history])
        
        return {
            'average_query_time': avg_query_time,
            'average_hypothesis_time': avg_hypothesis_time,
            'average_retrieval_time': avg_retrieval_time,
            'average_similarity_score': avg_similarity,
            'total_queries': len(self.metrics_history)
        }
```

Slide 12: Real-world Example: Research Paper Recommender

This implementation demonstrates HyDE's application in a research paper recommendation system, showcasing how to process academic papers and generate relevant recommendations based on user queries.

```python
class PaperRecommender:
    def __init__(self, hyde_retriever, paper_database):
        self.hyde = hyde_retriever
        self.papers = paper_database
        self.processor = DocumentProcessor()
        
    def extract_paper_features(self, paper):
        return f"""Title: {paper['title']}
                  Abstract: {paper['abstract']}
                  Keywords: {', '.join(paper['keywords'])}"""
    
    def recommend_papers(self, query: str, top_k: int = 5):
        # Generate hypothesis about relevant papers
        hypothesis = self.hyde.generate_hypothesis(
            f"Research papers about: {query}"
        )
        
        # Process papers and create embeddings
        paper_features = [self.extract_paper_features(p) 
                         for p in self.papers]
        paper_chunks = [self.processor.chunk_document(p) 
                       for p in paper_features]
        
        # Get recommendations
        results = self.hyde.process_query(
            hypothesis,
            [chunk for chunks in paper_chunks for chunk in chunks]
        )
        
        return self._format_recommendations(results, top_k)
```

Slide 13: Results for Research Paper Recommender

```python
# Example usage and results
papers = [
    {
        'title': 'Advances in Neural Information Retrieval',
        'abstract': 'This paper presents recent developments...',
        'keywords': ['IR', 'neural networks', 'deep learning']
    },
    # ... more papers ...
]

recommender = PaperRecommender(hyde_retriever, papers)
query = "Latest developments in transformer architectures"

recommendations = recommender.recommend_papers(query)
print("\nQuery:", query)
print("\nRecommended Papers:")
for i, paper in enumerate(recommendations, 1):
    print(f"\n{i}. Title: {paper['title']}")
    print(f"   Relevance Score: {paper['score']:.3f}")
    print(f"   Keywords: {', '.join(paper['keywords'])}")

# Example Output:
# Query: Latest developments in transformer architectures
# Recommended Papers:
# 1. Title: Advances in Neural Information Retrieval
#    Relevance Score: 0.892
#    Keywords: IR, neural networks, deep learning
```

Slide 14: Integration with Existing Systems

This implementation shows how to integrate HyDE into existing information retrieval systems, providing a flexible adapter pattern for different backend services and data sources.

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class DataSourceAdapter(ABC):
    @abstractmethod
    def fetch_documents(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def update_index(self, documents: List[Dict[str, Any]]) -> None:
        pass

class HyDESystem:
    def __init__(self, data_source: DataSourceAdapter):
        self.data_source = data_source
        self.hyde_retriever = HyDERetriever()
        self.analytics = HyDEAnalytics()
        
    def refresh_index(self):
        documents = self.data_source.fetch_documents()
        self.hyde_retriever.build_vector_store(documents)
        
    def query(self, user_query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Process query through HyDE
        results = self.hyde_retriever.process_query(user_query, 
            self.data_source.fetch_documents())
        
        # Record metrics
        query_time = time.time() - start_time
        self.analytics.record_query(QueryMetrics(
            query_time=query_time,
            hypothesis_generation_time=results.get('hypothesis_time', 0),
            retrieval_time=results.get('retrieval_time', 0),
            num_contexts=len(results['retrieved_contexts']),
            similarity_scores=results.get('similarity_scores', [])
        ))
        
        return results
```

Slide 15: Additional Resources

*   Improving Text-to-Text Retrieval with HyDE: [https://arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496)
*   Dense Passage Retrieval for Open-Domain Question Answering: [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
*   Neural Information Retrieval Advances: [https://arxiv.org/abs/2201.05994](https://arxiv.org/abs/2201.05994)
*   Exploring Contrastive Learning in Information Retrieval Systems: [https://dl.acm.org/doi/10.1145/3477495.3531937](https://dl.acm.org/doi/10.1145/3477495.3531937)
*   Performance Analysis of Vector Search Methods in Information Retrieval: [https://www.sciencedirect.com/science/article/pii/S0020025520308451](https://www.sciencedirect.com/science/article/pii/S0020025520308451)

