## Low-Latency Boost for Context in RAG Pipelines Rerankers
Slide 1: Understanding RAG Pipeline Fundamentals

A Retrieval-Augmented Generation (RAG) pipeline combines vector similarity search with reranking to improve context retrieval. The base implementation uses FAISS for efficient similarity search and transformers for embedding generation, forming the foundation for later reranking enhancements.

```python
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class BaseRAGPipeline:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.index = None
        
    def encode_text(self, texts):
        # Tokenize and encode texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token embeddings
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def build_index(self, documents):
        embeddings = self.encode_text(documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        return embeddings

# Example usage
documents = ["Document 1 content", "Document 2 content"]
rag = BaseRAGPipeline()
embeddings = rag.build_index(documents)
```

Slide 2: Implementing Cross-Encoder Reranking

Cross-encoder models directly compare query-document pairs for more nuanced relevance assessment. This implementation uses the cross-encoder/ms-marco-MiniLM-L-6-v2 model, which is specifically trained for reranking tasks and provides more accurate semantic matching.

```python
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

class RerankerModule:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(pairs, padding=True, truncation=True,
                              return_tensors='pt', max_length=512)
        
        with torch.no_grad():
            scores = self.model(**inputs).logits
            scores = F.softmax(scores, dim=1)[:, 1].numpy()
        
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices], scores[ranked_indices]

# Example usage
reranker = RerankerModule()
query = "What is machine learning?"
ranked_docs, scores = reranker.rerank(query, documents)
```

Slide 3: Hybrid Retrieval System

A hybrid retrieval system combines both embedding-based similarity search and cross-encoder reranking to leverage the strengths of both approaches. The initial retrieval uses fast vector similarity, while reranking provides deep semantic understanding for final document selection.

```python
class HybridRetrieval:
    def __init__(self, base_k=100, rerank_k=10):
        self.rag_pipeline = BaseRAGPipeline()
        self.reranker = RerankerModule()
        self.base_k = base_k
        self.rerank_k = rerank_k
        self.documents = None
        
    def index_documents(self, documents):
        self.documents = documents
        self.embeddings = self.rag_pipeline.build_index(documents)
        
    def search(self, query):
        # Initial retrieval using vector similarity
        query_embedding = self.rag_pipeline.encode_text([query])
        D, I = self.rag_pipeline.index.search(
            query_embedding.astype('float32'), self.base_k)
        
        # Get candidate documents
        candidates = [self.documents[i] for i in I[0]]
        
        # Rerank candidates
        reranked_docs, scores = self.reranker.rerank(
            query, candidates, self.rerank_k)
        
        return reranked_docs, scores

# Example usage
hybrid = HybridRetrieval()
hybrid.index_documents(documents)
results, scores = hybrid.search("What is machine learning?")
```

Slide 4: Document Chunking and Preprocessing

Effective document chunking is crucial for RAG pipelines as it impacts context relevance. This implementation uses a sliding window approach with overlap to maintain contextual continuity while creating manageable chunks for embedding and reranking processes.

```python
class DocumentPreprocessor:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_document(self, text):
        words = text.split()
        chunks = []
        chunk_metadata = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if len(chunk.split()) >= self.chunk_size // 2:  # Avoid tiny chunks
                chunks.append(chunk)
                chunk_metadata.append({
                    'start_idx': i,
                    'end_idx': i + self.chunk_size,
                    'length': len(chunk.split())
                })
        return chunks, chunk_metadata
    
    def process_documents(self, documents):
        processed_chunks = []
        all_metadata = []
        
        for doc_id, doc in enumerate(documents):
            chunks, metadata = self.chunk_document(doc)
            processed_chunks.extend(chunks)
            
            # Add document reference to metadata
            for m in metadata:
                m['doc_id'] = doc_id
            all_metadata.extend(metadata)
            
        return processed_chunks, all_metadata

# Example usage
preprocessor = DocumentPreprocessor()
docs = ["Long document text...", "Another long document..."]
chunks, metadata = preprocessor.process_documents(docs)
```

Slide 5: Semantic Search Enhancement

Enhancing semantic search capabilities by implementing a custom similarity metric that combines lexical and semantic matching. This approach helps capture both exact matches and conceptual similarities, improving retrieval quality.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np

class EnhancedSemanticSearch:
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Weight for combining lexical and semantic scores
        self.tfidf = TfidfVectorizer()
        self.rag_pipeline = BaseRAGPipeline()
        
    def fit(self, documents):
        # Compute TF-IDF vectors
        self.tfidf_matrix = self.tfidf.fit_transform(documents)
        # Compute semantic embeddings
        self.semantic_embeddings = self.rag_pipeline.encode_text(documents)
        self.documents = documents
        
    def hybrid_similarity(self, query):
        # Lexical similarity using TF-IDF
        query_tfidf = self.tfidf.transform([query])
        lexical_scores = (query_tfidf @ self.tfidf_matrix.T).toarray()[0]
        
        # Semantic similarity using embeddings
        query_embedding = self.rag_pipeline.encode_text([query])
        semantic_scores = np.array([
            1 - cosine(query_embedding[0], doc_emb) 
            for doc_emb in self.semantic_embeddings
        ])
        
        # Combine scores
        combined_scores = (
            self.alpha * lexical_scores + 
            (1 - self.alpha) * semantic_scores
        )
        return combined_scores
    
    def search(self, query, top_k=5):
        scores = self.hybrid_similarity(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices], scores[top_indices]

# Example usage
enhanced_search = EnhancedSemanticSearch()
enhanced_search.fit(chunks)
results, scores = enhanced_search.search("query text", top_k=5)
```

Slide 6: Query Intent Classification

Implementing query intent classification to dynamically adjust reranking strategies. This component analyzes the query type to determine whether to emphasize semantic similarity, factual matching, or comparative analysis.

```python
from transformers import pipeline
import re

class QueryIntentClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        self.intent_patterns = {
            'comparison': r'compare|versus|vs|difference',
            'factual': r'what is|who is|when|where|how many',
            'explanation': r'why|how does|explain|describe'
        }
        
    def classify_intent(self, query):
        # Rule-based pattern matching
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query.lower()):
                return intent, 1.0
                
        # Zero-shot classification for uncertain cases
        candidate_labels = ["comparison", "factual", "explanation"]
        result = self.classifier(
            query, 
            candidate_labels,
            multi_label=False
        )
        return result['labels'][0], result['scores'][0]
    
    def get_reranking_weights(self, query):
        intent, confidence = self.classify_intent(query)
        weights = {
            'comparison': {'semantic': 0.7, 'lexical': 0.3},
            'factual': {'semantic': 0.4, 'lexical': 0.6},
            'explanation': {'semantic': 0.6, 'lexical': 0.4}
        }
        return weights.get(intent, {'semantic': 0.5, 'lexical': 0.5})

# Example usage
intent_classifier = QueryIntentClassifier()
query = "Compare transformer architectures with RNNs"
intent, confidence = intent_classifier.classify_intent(query)
weights = intent_classifier.get_reranking_weights(query)
```

Slide 7: Dynamic Context Window Optimization

Implementing adaptive context window sizing based on query complexity and document characteristics. This approach dynamically adjusts chunk sizes and overlap to optimize context retention while maintaining computational efficiency.

```python
class DynamicContextOptimizer:
    def __init__(self, min_size=256, max_size=1024):
        self.min_size = min_size
        self.max_size = max_size
        self.query_analyzer = QueryIntentClassifier()
        
    def compute_complexity_score(self, text):
        # Compute linguistic complexity metrics
        sentence_count = len(text.split('.'))
        word_count = len(text.split())
        avg_word_length = sum(len(w) for w in text.split()) / word_count
        
        # Normalize and combine metrics
        complexity = (
            0.3 * (sentence_count / 10) +
            0.3 * (word_count / 200) +
            0.4 * (avg_word_length / 6)
        )
        return min(1.0, complexity)
    
    def optimize_window(self, query, document):
        intent, confidence = self.query_analyzer.classify_intent(query)
        doc_complexity = self.compute_complexity_score(document)
        
        # Adjust window size based on complexity and intent
        base_size = self.min_size + (self.max_size - self.min_size) * doc_complexity
        
        window_params = {
            'comparison': {'size': int(base_size * 1.2), 'overlap': 0.3},
            'explanation': {'size': int(base_size * 1.5), 'overlap': 0.4},
            'factual': {'size': int(base_size * 0.8), 'overlap': 0.2}
        }
        
        params = window_params.get(intent, {'size': int(base_size), 'overlap': 0.25})
        return min(params['size'], self.max_size), params['overlap']

# Example usage
optimizer = DynamicContextOptimizer()
query = "Explain the differences between attention mechanisms"
document = "Long technical document..."
window_size, overlap = optimizer.optimize_window(query, document)

print(f"Optimized window size: {window_size}")
print(f"Optimized overlap: {overlap}")
```

Slide 8: Contextual Relevance Scoring

Implementing a sophisticated scoring mechanism that considers both local and global context relevance. This system weights different aspects of relevance based on query characteristics and document structure.

```python
class ContextualRelevanceScorer:
    def __init__(self):
        self.semantic_model = BaseRAGPipeline()
        self.intent_classifier = QueryIntentClassifier()
        
    def compute_local_coherence(self, chunk, neighbors):
        chunk_emb = self.semantic_model.encode_text([chunk])[0]
        neighbor_embs = self.semantic_model.encode_text(neighbors)
        
        # Calculate cosine similarity with neighboring chunks
        coherence_scores = [
            1 - cosine(chunk_emb, n_emb) 
            for n_emb in neighbor_embs
        ]
        return np.mean(coherence_scores)
    
    def compute_relevance_score(self, query, chunk, context):
        query_intent, _ = self.intent_classifier.classify_intent(query)
        
        # Compute different relevance components
        semantic_score = 1 - cosine(
            self.semantic_model.encode_text([query])[0],
            self.semantic_model.encode_text([chunk])[0]
        )
        
        local_coherence = self.compute_local_coherence(
            chunk, 
            context['neighbors']
        )
        
        # Position bias for different intents
        position_weights = {
            'comparison': 0.15,
            'explanation': 0.25,
            'factual': 0.1
        }
        position_weight = position_weights.get(query_intent, 0.2)
        position_score = 1 - (context['position'] / context['total_chunks'])
        
        # Combine scores with intent-specific weights
        weights = {
            'comparison': {'semantic': 0.5, 'coherence': 0.3, 'position': 0.2},
            'explanation': {'semantic': 0.4, 'coherence': 0.4, 'position': 0.2},
            'factual': {'semantic': 0.6, 'coherence': 0.2, 'position': 0.2}
        }
        
        w = weights.get(query_intent, 
                       {'semantic': 0.5, 'coherence': 0.3, 'position': 0.2})
        
        final_score = (
            w['semantic'] * semantic_score +
            w['coherence'] * local_coherence +
            w['position'] * position_score
        )
        
        return final_score, {
            'semantic': semantic_score,
            'coherence': local_coherence,
            'position': position_score
        }

# Example usage
scorer = ContextualRelevanceScorer()
query = "Compare different attention mechanisms"
chunk = "Technical description of attention..."
context = {
    'neighbors': ["Previous chunk...", "Next chunk..."],
    'position': 2,
    'total_chunks': 10
}
score, components = scorer.compute_relevance_score(query, chunk, context)
```

Slide 9: Real-world Implementation: Scientific Paper Analysis

Implementing a complete RAG pipeline for analyzing scientific papers, demonstrating the practical application of reranking in a real-world scenario. This implementation handles PDF extraction, section identification, and hierarchical context retention.

```python
import fitz  # PyMuPDF
from typing import List, Dict
import re

class ScientificPaperRAG:
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.semantic_search = EnhancedSemanticSearch()
        self.reranker = RerankerModule()
        self.sections = {}
        
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, str]:
        doc = fitz.open(pdf_path)
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': ''
        }
        
        current_section = None
        section_pattern = re.compile(
            r'^(?:abstract|introduction|method|results?|conclusion)',
            re.IGNORECASE
        )
        
        for page in doc:
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                if section_pattern.match(line.strip()):
                    current_section = line.strip().lower()
                    if 'method' in current_section:
                        current_section = 'methodology'
                elif current_section and current_section in sections:
                    sections[current_section] += line + '\n'
        
        return sections
    
    def process_paper(self, pdf_path: str):
        # Extract and process paper content
        self.sections = self.extract_pdf_content(pdf_path)
        
        # Process each section with appropriate context
        processed_sections = {}
        for section, content in self.sections.items():
            chunks, metadata = self.preprocessor.chunk_document(content)
            processed_sections[section] = {
                'chunks': chunks,
                'metadata': metadata
            }
        
        # Build section-specific indices
        for section in processed_sections:
            self.semantic_search.fit(processed_sections[section]['chunks'])
            
        return processed_sections
    
    def query_paper(self, query: str, top_k: int = 3) -> List[Dict]:
        results = []
        
        # Query each section and rerank combined results
        for section, content in self.sections.items():
            section_results, scores = self.semantic_search.search(
                query, 
                top_k=top_k
            )
            
            for res, score in zip(section_results, scores):
                results.append({
                    'section': section,
                    'content': res,
                    'base_score': score
                })
        
        # Rerank combined results
        contents = [r['content'] for r in results]
        reranked_scores = self.reranker.rerank(query, contents)[1]
        
        # Combine original and reranked scores
        for i, (result, rerank_score) in enumerate(zip(results, reranked_scores)):
            result['final_score'] = 0.4 * result['base_score'] + 0.6 * rerank_score
            
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        return results[:top_k]

# Example usage
paper_rag = ScientificPaperRAG()
processed_paper = paper_rag.process_paper("example_paper.pdf")
results = paper_rag.query_paper(
    "What are the main findings regarding attention mechanisms?"
)

for result in results:
    print(f"\nSection: {result['section']}")
    print(f"Score: {result['final_score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
```

Slide 10: Performance Metrics and Evaluation

Creating a comprehensive evaluation framework to assess the effectiveness of reranking in the RAG pipeline, measuring both quantitative metrics and qualitative improvements in context relevance.

```python
from sklearn.metrics import ndcg_score, precision_recall_curve
import numpy as np
from typing import List, Dict

class RAGEvaluator:
    def __init__(self):
        self.metrics = {}
        self.query_logs = []
        
    def calculate_ndcg(self, 
                      relevance_scores: np.ndarray, 
                      predictions: np.ndarray,
                      k: int = 5) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain
        """
        return ndcg_score(
            [relevance_scores], 
            [predictions], 
            k=k
        )
    
    def semantic_similarity_score(self,
                                query: str,
                                retrieved_text: str,
                                model) -> float:
        """
        Calculate semantic similarity between query and retrieved text
        """
        query_embedding = model.encode_text([query])[0]
        text_embedding = model.encode_text([retrieved_text])[0]
        return 1 - cosine(query_embedding, text_embedding)
    
    def evaluate_pipeline(self,
                         pipeline,
                         test_queries: List[Dict],
                         ground_truth: Dict) -> Dict:
        results = {
            'ndcg': [],
            'precision': [],
            'semantic_similarity': [],
            'response_time': []
        }
        
        for query_item in test_queries:
            query = query_item['query']
            relevant_docs = ground_truth[query]
            
            # Measure response time
            start_time = time.time()
            retrieved_docs = pipeline.query_paper(query, top_k=5)
            response_time = time.time() - start_time
            
            # Calculate metrics
            pred_scores = [doc['final_score'] for doc in retrieved_docs]
            true_scores = [1.0 if doc['content'] in relevant_docs else 0.0 
                          for doc in retrieved_docs]
            
            results['ndcg'].append(
                self.calculate_ndcg(
                    np.array([true_scores]),
                    np.array([pred_scores])
                )
            )
            
            # Calculate semantic similarity
            sim_scores = [
                self.semantic_similarity_score(
                    query, 
                    doc['content'],
                    pipeline.semantic_search.rag_pipeline
                )
                for doc in retrieved_docs
            ]
            results['semantic_similarity'].append(np.mean(sim_scores))
            results['response_time'].append(response_time)
            
            # Log query details
            self.query_logs.append({
                'query': query,
                'retrieved_docs': retrieved_docs,
                'metrics': {
                    'ndcg': results['ndcg'][-1],
                    'semantic_similarity': results['semantic_similarity'][-1],
                    'response_time': response_time
                }
            })
        
        # Aggregate results
        final_metrics = {
            'mean_ndcg': np.mean(results['ndcg']),
            'mean_semantic_similarity': np.mean(results['semantic_similarity']),
            'mean_response_time': np.mean(results['response_time']),
            'std_response_time': np.std(results['response_time'])
        }
        
        return final_metrics

# Example usage
evaluator = RAGEvaluator()
test_queries = [
    {'query': 'Explain the attention mechanism'},
    {'query': 'Compare transformer architectures'},
    # ... more test queries
]

ground_truth = {
    'Explain the attention mechanism': ['relevant_doc_1', 'relevant_doc_2'],
    'Compare transformer architectures': ['relevant_doc_3', 'relevant_doc_4'],
    # ... ground truth for other queries
}

metrics = evaluator.evaluate_pipeline(paper_rag, test_queries, ground_truth)
print("Evaluation Results:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
```

Slide 11: Real-time Adaptive Reranking

Implementing an adaptive reranking system that adjusts its parameters based on real-time feedback and query patterns. This system optimizes the balance between embedding-based retrieval and cross-encoder reranking dynamically.

```python
class AdaptiveReranker:
    def __init__(self):
        self.base_ranker = BaseRAGPipeline()
        self.cross_encoder = RerankerModule()
        self.performance_history = []
        self.adaptation_window = 100
        self.current_weights = {'embedding': 0.5, 'cross_encoder': 0.5}
        
    def update_weights(self, query_success_rate: float):
        """Dynamically adjust weights based on performance"""
        if len(self.performance_history) >= self.adaptation_window:
            recent_performance = np.mean(self.performance_history[-self.adaptation_window:])
            
            # Adjust weights based on moving average
            if recent_performance < query_success_rate:
                self.current_weights['cross_encoder'] = min(
                    0.8, 
                    self.current_weights['cross_encoder'] + 0.05
                )
            else:
                self.current_weights['cross_encoder'] = max(
                    0.2, 
                    self.current_weights['cross_encoder'] - 0.05
                )
                
            self.current_weights['embedding'] = 1 - self.current_weights['cross_encoder']
            
    def adaptive_rerank(self, query: str, candidates: List[str], 
                       feedback: Optional[float] = None) -> List[Tuple[str, float]]:
        # Get base embeddings scores
        embedding_scores = self.base_ranker.encode_text([query])
        
        # Get cross-encoder scores
        cross_encoder_scores = self.cross_encoder.rerank(query, candidates)[1]
        
        # Combine scores using current weights
        final_scores = (
            self.current_weights['embedding'] * embedding_scores +
            self.current_weights['cross_encoder'] * cross_encoder_scores
        )
        
        # Update weights if feedback provided
        if feedback is not None:
            self.performance_history.append(feedback)
            self.update_weights(feedback)
            
        # Return reranked results
        ranked_indices = np.argsort(final_scores)[::-1]
        return [(candidates[i], final_scores[i]) for i in ranked_indices]

    def get_performance_stats(self):
        if not self.performance_history:
            return None
            
        return {
            'avg_performance': np.mean(self.performance_history),
            'current_weights': self.current_weights,
            'performance_trend': np.gradient(
                self.performance_history[-min(100, len(self.performance_history)):]
            ).mean()
        }

# Example usage
adaptive_reranker = AdaptiveReranker()

# Simulate queries and feedback
queries = [
    "Explain attention mechanisms",
    "Compare transformer architectures",
    "Describe BERT's pretraining"
]

candidates = [
    "Detailed explanation of attention...",
    "Transformer architecture overview...",
    "BERT pretraining process..."
]

for query in queries:
    # Get reranked results
    results = adaptive_reranker.adaptive_rerank(
        query,
        candidates,
        feedback=0.8  # Simulated feedback score
    )
    
    # Print results and current performance stats
    print(f"\nQuery: {query}")
    for doc, score in results:
        print(f"Score: {score:.3f} - {doc[:50]}...")
        
    stats = adaptive_reranker.get_performance_stats()
    print("\nPerformance Stats:")
    print(f"Current Weights: {stats['current_weights']}")
    print(f"Performance Trend: {stats['performance_trend']:.3f}")
```

Slide 12: Results Analysis and Visualization

Implementing comprehensive visualization tools for analyzing reranker performance and understanding the impact of different components in the RAG pipeline.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

class RAGVisualizer:
    def __init__(self):
        plt.style.use('seaborn')
        self.metrics_history = []
        
    def log_metrics(self, metrics: Dict):
        self.metrics_history.append(metrics)
        
    def plot_performance_over_time(self):
        df = pd.DataFrame(self.metrics_history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot NDCG and Semantic Similarity
        ax1.plot(df['mean_ndcg'], label='NDCG', marker='o')
        ax1.plot(df['mean_semantic_similarity'], 
                label='Semantic Similarity', 
                marker='s')
        ax1.set_title('Retrieval Quality Metrics Over Time')
        ax1.set_ylabel('Score')
        ax1.legend()
        
        # Plot Response Time
        ax2.plot(df['mean_response_time'], 
                label='Response Time', 
                color='green', 
                marker='d')
        ax2.fill_between(
            range(len(df)),
            df['mean_response_time'] - df['std_response_time'],
            df['mean_response_time'] + df['std_response_time'],
            alpha=0.3,
            color='green'
        )
        ax2.set_title('Response Time Distribution')
        ax2.set_ylabel('Seconds')
        ax2.set_xlabel('Query Batch')
        
        plt.tight_layout()
        return fig
    
    def plot_reranking_impact(self, 
                            before_scores: List[float],
                            after_scores: List[float],
                            labels: List[str]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(before_scores))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], 
               before_scores, 
               width, 
               label='Before Reranking')
        ax.bar([i + width/2 for i in x], 
               after_scores, 
               width, 
               label='After Reranking')
        
        ax.set_ylabel('Relevance Score')
        ax.set_title('Impact of Reranking on Document Relevance')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_performance_report(self) -> Dict:
        df = pd.DataFrame(self.metrics_history)
        
        report = {
            'overall_metrics': {
                'mean_ndcg': df['mean_ndcg'].mean(),
                'mean_semantic_similarity': df['mean_semantic_similarity'].mean(),
                'mean_response_time': df['mean_response_time'].mean()
            },
            'improvement_trends': {
                'ndcg_trend': df['mean_ndcg'].pct_change().mean(),
                'semantic_similarity_trend': df['mean_semantic_similarity'].pct_change().mean(),
                'response_time_trend': df['mean_response_time'].pct_change().mean()
            }
        }
        
        return report

# Example usage
visualizer = RAGVisualizer()

# Log some example metrics
for _ in range(10):
    metrics = {
        'mean_ndcg': np.random.uniform(0.7, 0.9),
        'mean_semantic_similarity': np.random.uniform(0.6, 0.8),
        'mean_response_time': np.random.uniform(0.1, 0.3),
        'std_response_time': np.random.uniform(0.02, 0.05)
    }
    visualizer.log_metrics(metrics)

# Generate visualizations
performance_fig = visualizer.plot_performance_over_time()
performance_fig.savefig('performance_over_time.png')

# Generate and print report
report = visualizer.generate_performance_report()
print("\nPerformance Report:")
print(json.dumps(report, indent=2))
```

Slide 13: Advanced Cross-Encoder Optimization

Implementing sophisticated optimization techniques for cross-encoder reranking, including attention pruning and dynamic batch sizing to improve computational efficiency while maintaining accuracy.

```python
from torch.cuda import amp
import torch.nn as nn
from transformers import AutoConfig
import torch

class OptimizedCrossEncoder:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.scaler = amp.GradScaler()
        self.attention_threshold = 0.1
        
    def prune_attention(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Prune attention weights below threshold"""
        mask = attention_weights > self.attention_threshold
        return attention_weights * mask
        
    def optimize_batch_size(self, sequence_length: int) -> int:
        """Dynamically adjust batch size based on sequence length"""
        base_tokens = 512
        base_batch = 32
        return max(1, int(base_batch * (base_tokens / sequence_length)))
        
    def rerank_optimized(self, 
                        query: str, 
                        documents: List[str], 
                        top_k: int = 5) -> Tuple[List[str], np.ndarray]:
        # Prepare inputs
        pairs = [[query, doc] for doc in documents]
        max_length = max(len(p[0]) + len(p[1]) for p in pairs)
        batch_size = self.optimize_batch_size(max_length)
        
        scores = []
        with torch.cuda.amp.autocast():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                with torch.no_grad():
                    attention_mask = inputs['attention_mask']
                    outputs = self.model(**inputs)
                    
                    # Get attention weights and prune
                    attention_weights = outputs.attentions[-1]
                    pruned_attention = self.prune_attention(attention_weights)
                    
                    # Calculate final scores
                    logits = outputs.logits
                    batch_scores = F.softmax(logits, dim=1)[:, 1]
                    scores.extend(batch_scores.cpu().numpy())
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in top_indices], np.array(scores)[top_indices]
    
    def compute_attention_statistics(self, 
                                  attention_weights: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about attention patterns"""
        return {
            'mean_attention': attention_weights.mean().item(),
            'sparsity': (attention_weights < self.attention_threshold).float().mean().item(),
            'max_attention': attention_weights.max().item()
        }

# Example usage
optimizer = OptimizedCrossEncoder()

# Test data
query = "What are the key components of attention mechanisms?"
documents = [
    "Attention mechanisms consist of query, key, and value matrices...",
    "The transformer architecture revolutionized NLP...",
    "BERT uses bidirectional attention for pretraining..."
]

# Perform optimized reranking
reranked_docs, scores = optimizer.rerank_optimized(query, documents)

# Print results
for doc, score in zip(reranked_docs, scores):
    print(f"\nScore: {score:.3f}")
    print(f"Document: {doc[:100]}...")
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "REALM: Retrieval-Augmented Language Model Pre-Training" - [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
*   "Dense Passage Retrieval for Open-Domain Question Answering" - [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
*   "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" - [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
*   "Improving Document Reranking with Cross-Encoder-Based Relevance Feedback" - [https://arxiv.org/abs/2203.08855](https://arxiv.org/abs/2203.08855)

