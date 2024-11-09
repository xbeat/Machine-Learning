## Improving RAG Pipelines with Low-Latency Rerankers
Slide 1: Understanding RAG Pipeline Components

A Retrieval Augmented Generation (RAG) pipeline combines vector database retrieval with language model generation. The core components include document chunking, embedding generation, and semantic search to retrieve relevant context before generating responses.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class RAGPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize sentence transformer for embeddings
        self.embed_model = SentenceTransformer(model_name)
        self.doc_store = {}  # Simple in-memory document store
        
    def chunk_document(self, text: str, chunk_size: int = 512) -> List[str]:
        # Basic text chunking implementation
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        # Generate embeddings for text chunks
        return self.embed_model.encode(chunks)
    
    def add_document(self, doc_id: str, text: str):
        chunks = self.chunk_document(text)
        embeddings = self.embed_chunks(chunks)
        self.doc_store[doc_id] = {
            'chunks': chunks,
            'embeddings': embeddings
        }
```

Slide 2: Implementing Cross-Encoder Reranking

Cross-encoder models directly compare query-passage pairs to compute relevance scores, offering more nuanced semantic understanding than bi-encoders. This implementation shows how to integrate a cross-encoder reranker with the basic RAG pipeline.

```python
from sentence_transformers import CrossEncoder
import torch
from typing import List, Tuple

class RerankerPipeline(RAGPipeline):
    def __init__(self, 
                 embed_model: str = 'all-MiniLM-L6-v2',
                 reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        super().__init__(embed_model)
        self.reranker = CrossEncoder(reranker_model)
        
    def retrieve_and_rerank(self, 
                           query: str, 
                           k_retrieve: int = 10,
                           k_rerank: int = 3) -> List[Tuple[str, float]]:
        # Get query embedding
        query_embedding = self.embed_model.encode(query)
        
        # First-stage retrieval using embeddings
        candidates = []
        for doc_id, doc_data in self.doc_store.items():
            similarities = np.dot(doc_data['embeddings'], query_embedding)
            top_k_idx = np.argsort(similarities)[-k_retrieve:]
            for idx in top_k_idx:
                candidates.append((doc_data['chunks'][idx], similarities[idx]))
        
        # Rerank candidates using cross-encoder
        pairs = [[query, candidate[0]] for candidate in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort by reranker scores and return top k
        reranked = sorted(zip(candidates, rerank_scores), 
                         key=lambda x: x[1], reverse=True)
        return reranked[:k_rerank]
```

Slide 3: Semantic Search Implementation

The semantic search component uses cosine similarity to find relevant documents based on embedding similarity. This implementation includes optimization techniques for efficient similarity computation using numpy operations.

```python
def semantic_search(self, 
                   query: str,
                   top_k: int = 5) -> List[Dict[str, any]]:
    # Compute query embedding
    query_embedding = self.embed_model.encode(query)
    
    # Initialize results container
    results = []
    
    # Compute similarities with all stored embeddings
    for doc_id, doc_data in self.doc_store.items():
        # Normalize embeddings for cosine similarity
        doc_embeddings = doc_data['embeddings']
        doc_embeddings_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1)[:, np.newaxis]
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarities
        similarities = np.dot(doc_embeddings_norm, query_embedding_norm)
        
        # Get top k chunks from this document
        top_indices = np.argsort(similarities)[-top_k:]
        
        for idx in top_indices:
            results.append({
                'doc_id': doc_id,
                'chunk': doc_data['chunks'][idx],
                'score': float(similarities[idx])
            })
    
    # Sort all results and return top k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]
```

Slide 4: Hybrid Retrieval Strategy

Hybrid retrieval combines sparse retrieval (BM25) with dense retrieval (embeddings) to leverage both lexical and semantic matching. This approach helps capture both exact keyword matches and semantic relationships between queries and documents.

```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple

class HybridRetriever:
    def __init__(self, dense_weight: float = 0.5):
        self.bm25 = None
        self.dense_weight = dense_weight
        self.sparse_weight = 1.0 - dense_weight
        
    def fit(self, documents: List[str]):
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def hybrid_search(self, 
                     query: str,
                     dense_scores: np.ndarray,
                     top_k: int = 5) -> List[Tuple[int, float]]:
        # Get BM25 scores
        tokenized_query = query.lower().split()
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize scores
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        
        # Combine scores
        final_scores = (self.sparse_weight * sparse_scores + 
                       self.dense_weight * dense_scores)
        
        # Get top k results
        top_indices = np.argsort(final_scores)[-top_k:][::-1]
        return [(idx, final_scores[idx]) for idx in top_indices]
```

Slide 5: Cross-Encoder Optimization

Cross-encoders can be computationally expensive when dealing with large candidate sets. This implementation uses batch processing and early stopping to optimize reranking performance while maintaining quality.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

class OptimizedReranker:
    def __init__(self, 
                 model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        self.device = device
        
    def rerank_batched(self, 
                      query: str,
                      candidates: List[str],
                      max_candidates: int = 1000) -> List[Tuple[str, float]]:
        # Truncate candidates if needed
        candidates = candidates[:max_candidates]
        
        # Prepare input pairs
        pairs = [[query, doc] for doc in candidates]
        
        # Create batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch)
            all_scores.extend(scores)
            
        # Sort and return results
        scored_candidates = list(zip(candidates, all_scores))
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
```

Slide 6: Contextual Chunk Generation

Effective document chunking considers semantic boundaries and contextual overlap to maintain coherence and improve retrieval quality. This implementation uses sliding windows with overlap and semantic segmentation.

```python
from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class ContextualChunker:
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 128,
                 min_chunk_size: int = 256):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
    def create_chunks(self, text: str) -> List[str]:
        # Split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep last sentences for overlap
                    overlap_tokens = 0
                    overlap_chunk = []
                    for sent in reversed(current_chunk):
                        sent_len = len(sent.split())
                        if overlap_tokens + sent_len <= self.overlap:
                            overlap_chunk.insert(0, sent)
                            overlap_tokens += sent_len
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_tokens
                    
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk if it meets minimum size
        if current_length >= self.min_chunk_size:
            chunks.append(' '.join(current_chunk))
            
        return chunks
```

Slide 7: Evaluation Metrics Implementation

This implementation provides comprehensive evaluation metrics for RAG systems, including relevance scoring, semantic similarity, and answer correctness metrics to assess the quality of retrieved contexts and generated responses.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from typing import List, Dict, Tuple

class RAGEvaluator:
    def __init__(self, embed_model):
        self.embed_model = embed_model
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_contexts: List[str],
                          ground_truth: str) -> Dict[str, float]:
        # Compute embeddings
        query_embedding = self.embed_model.encode(query)
        context_embeddings = self.embed_model.encode(retrieved_contexts)
        truth_embedding = self.embed_model.encode(ground_truth)
        
        # Calculate semantic similarity scores
        semantic_scores = cosine_similarity(
            context_embeddings, 
            truth_embedding.reshape(1, -1)
        ).flatten()
        
        # Calculate ROUGE scores
        rouge_scores = [
            self.rouge_scorer.score(ground_truth, context)
            for context in retrieved_contexts
        ]
        
        return {
            'semantic_similarity': semantic_scores.mean(),
            'rouge1_f1': np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
            'rouge2_f1': np.mean([s['rouge2'].fmeasure for s in rouge_scores]),
            'rougeL_f1': np.mean([s['rougeL'].fmeasure for s in rouge_scores])
        }
```

Slide 8: Context Window Analysis

Understanding the optimal context window size is crucial for RAG performance. This implementation analyzes different window sizes and their impact on relevance scores using a sliding window approach.

```python
class ContextWindowAnalyzer:
    def __init__(self, 
                 window_sizes: List[int] = [256, 512, 1024],
                 stride: int = 128):
        self.window_sizes = window_sizes
        self.stride = stride
        
    def analyze_windows(self, 
                       text: str,
                       query: str,
                       reranker) -> Dict[int, List[float]]:
        words = text.split()
        results = {size: [] for size in self.window_sizes}
        
        for window_size in self.window_sizes:
            # Generate windows with different sizes
            windows = []
            for i in range(0, len(words) - window_size + 1, self.stride):
                window = ' '.join(words[i:i + window_size])
                windows.append(window)
            
            # Score windows using reranker
            scores = reranker.rerank_batched(query, windows)
            results[window_size] = [score for _, score in scores]
            
        # Calculate statistics for each window size
        stats = {}
        for size, scores in results.items():
            stats[size] = {
                'mean_score': np.mean(scores),
                'max_score': np.max(scores),
                'std_score': np.std(scores)
            }
            
        return stats
```

Slide 9: Source Code for Context Window Analysis Results

The implementation provides visualization and analysis of context window performance across different sizes and queries.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def visualize_window_analysis(stats: Dict[int, Dict[str, float]]):
    # Prepare data for plotting
    window_sizes = list(stats.keys())
    mean_scores = [s['mean_score'] for s in stats.values()]
    std_scores = [s['std_score'] for s in stats.values()]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(window_sizes, mean_scores, yerr=std_scores, 
                fmt='o-', capsize=5)
    plt.xlabel('Window Size (tokens)')
    plt.ylabel('Mean Relevance Score')
    plt.title('Context Window Size Analysis')
    
    # Add score distribution violin plot
    plt.figure(figsize=(10, 6))
    plot_data = []
    for size, scores in stats.items():
        plot_data.extend([(size, score) for score in scores])
    
    sns.violinplot(data=plot_data, x='Window Size', y='Score')
    plt.title('Score Distribution by Window Size')
    
    return {
        'optimal_size': window_sizes[np.argmax(mean_scores)],
        'score_stability': np.mean(std_scores),
        'size_performance': dict(zip(window_sizes, mean_scores))
    }
```

Slide 10: Advanced Query Preprocessing

Query preprocessing significantly impacts retrieval quality. This implementation includes query expansion, entity recognition, and semantic decomposition to enhance retrieval effectiveness for complex queries.

```python
from nltk import pos_tag, word_tokenize
import spacy
from typing import List, Dict, Set

class QueryPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.important_pos = {'NOUN', 'VERB', 'ADJ'}
        
    def process_query(self, query: str) -> Dict[str, any]:
        # Process with spaCy
        doc = self.nlp(query)
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract key terms based on POS
        tokens = word_tokenize(query)
        pos_tags = pos_tag(tokens)
        key_terms = [word for word, pos in pos_tags 
                    if pos.startswith(('NN', 'VB', 'JJ'))]
        
        # Decompose complex queries
        clauses = [sent.text for sent in doc.sents]
        
        return {
            'original': query,
            'entities': entities,
            'key_terms': key_terms,
            'clauses': clauses,
            'processed': ' '.join(key_terms)
        }
        
    def expand_query(self, 
                    query_info: Dict[str, any],
                    top_k: int = 3) -> List[str]:
        expanded_queries = []
        base_query = query_info['processed']
        
        # Entity-focused expansion
        for entity, label in query_info['entities']:
            expanded = f"{base_query} {entity}"
            expanded_queries.append(expanded)
            
        # Key terms combination
        key_terms = query_info['key_terms']
        for i in range(min(len(key_terms), top_k)):
            terms_subset = key_terms[:i+1]
            expanded = f"{base_query} {' '.join(terms_subset)}"
            expanded_queries.append(expanded)
            
        return list(set(expanded_queries))
```

Slide 11: Real-world Implementation: Question Answering System

This implementation demonstrates a complete question answering system using RAG with reranking, including preprocessing, retrieval, and answer generation with performance metrics.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
import torch
from typing import List, Dict, Tuple

class RAGQuestionAnswering:
    def __init__(self,
                 retriever: RerankerPipeline,
                 model_name: str = 't5-base',
                 max_length: int = 512):
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
        self.max_length = max_length
        
    def answer_question(self, 
                       question: str,
                       k_contexts: int = 3) -> Dict[str, any]:
        # Retrieve relevant contexts
        contexts = self.retriever.retrieve_and_rerank(
            question,
            k_retrieve=10,
            k_rerank=k_contexts
        )
        
        # Prepare input for generation
        context_text = " [SEP] ".join([c[0] for c in contexts])
        input_text = f"question: {question} context: {context_text}"
        
        # Generate answer
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'answer': answer,
            'contexts': contexts,
            'confidence': float(torch.mean(outputs[1]).item())
        }
```

Slide 12: Real-world Implementation: Performance Analysis

This implementation provides comprehensive performance monitoring and analysis for the RAG pipeline, tracking latency, relevance metrics, and system resource utilization across different query types.

```python
import time
import psutil
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class PerformanceMetrics:
    latency: float
    memory_usage: float
    cpu_usage: float
    retrieval_time: float
    rerank_time: float
    generation_time: float
    
class RAGPerformanceAnalyzer:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.metrics_history = []
        
    def measure_performance(self, 
                          query: str,
                          warm_up: bool = True) -> Dict[str, float]:
        if warm_up:
            # Warm up run
            _ = self.rag_system.answer_question(query)
        
        # Measure actual performance
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()
        
        # Track individual component times
        retrieve_start = time.time()
        contexts = self.rag_system.retriever.retrieve_and_rerank(query)
        retrieve_time = time.time() - retrieve_start
        
        rerank_start = time.time()
        reranked_contexts = self.rag_system.retriever.rerank_candidates(contexts)
        rerank_time = time.time() - rerank_start
        
        generate_start = time.time()
        answer = self.rag_system.answer_question(query, reranked_contexts)
        generate_time = time.time() - generate_start
        
        # Calculate final metrics
        total_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            latency=total_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=end_cpu - start_cpu,
            retrieval_time=retrieve_time,
            rerank_time=rerank_time,
            generation_time=generate_time
        )
        
        self.metrics_history.append(metrics)
        return metrics

    def analyze_performance_trends(self) -> Dict[str, float]:
        metrics_array = np.array([
            [m.latency, m.memory_usage, m.cpu_usage, 
             m.retrieval_time, m.rerank_time, m.generation_time]
            for m in self.metrics_history
        ])
        
        return {
            'avg_latency': np.mean(metrics_array[:, 0]),
            'latency_std': np.std(metrics_array[:, 0]),
            'memory_usage_avg': np.mean(metrics_array[:, 1]),
            'cpu_usage_avg': np.mean(metrics_array[:, 2]),
            'component_times': {
                'retrieval': np.mean(metrics_array[:, 3]),
                'rerank': np.mean(metrics_array[:, 4]),
                'generation': np.mean(metrics_array[:, 5])
            }
        }
```

Slide 13: Optimized Token Management

Efficient token management is crucial for performance in RAG systems. This implementation provides optimized token handling with dynamic batching and context window adjustment.

```python
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TokenStats:
    total_tokens: int
    context_tokens: int
    question_tokens: int
    padding_tokens: int

class TokenManager:
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 2048,
                 target_batch_size: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_batch_size = target_batch_size
        
    def optimize_context_windows(self, 
                               contexts: List[str],
                               question: str) -> Tuple[List[str], TokenStats]:
        # Tokenize question
        question_tokens = self.tokenizer(
            question,
            add_special_tokens=False
        ).input_ids
        question_length = len(question_tokens)
        
        # Calculate available context length
        available_length = self.max_length - question_length - 3  # Special tokens
        
        # Tokenize and truncate contexts
        optimized_contexts = []
        total_tokens = 0
        padding_tokens = 0
        
        for context in contexts:
            context_tokens = self.tokenizer(
                context,
                add_special_tokens=False
            ).input_ids
            
            if len(context_tokens) > available_length:
                # Truncate while maintaining sentence boundaries
                truncated = self.truncate_to_sentence(
                    context,
                    available_length
                )
                context_tokens = self.tokenizer(
                    truncated,
                    add_special_tokens=False
                ).input_ids
            
            optimized_contexts.append(context)
            total_tokens += len(context_tokens)
            
            # Calculate padding needed for batch alignment
            batch_padding = (self.target_batch_size - 
                           (len(context_tokens) % self.target_batch_size))
            padding_tokens += batch_padding
            
        return optimized_contexts, TokenStats(
            total_tokens=total_tokens + question_length,
            context_tokens=total_tokens,
            question_tokens=question_length,
            padding_tokens=padding_tokens
        )
    
    def truncate_to_sentence(self, 
                           text: str,
                           max_tokens: int) -> str:
        sentences = text.split('.')
        tokenized_sentences = [
            self.tokenizer(sent, add_special_tokens=False).input_ids
            for sent in sentences
        ]
        
        total_tokens = 0
        keep_sentences = []
        
        for i, tokens in enumerate(tokenized_sentences):
            if total_tokens + len(tokens) <= max_tokens:
                keep_sentences.append(sentences[i])
                total_tokens += len(tokens)
            else:
                break
                
        return '.'.join(keep_sentences)
```

Slide 14: Results Analysis Dashboard

This implementation provides a comprehensive analysis dashboard for evaluating RAG pipeline performance, including detailed metrics visualization and performance comparisons across different configurations.

```python
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any

class RAGAnalysisDashboard:
    def __init__(self):
        self.metrics_data = []
        self.config_history = []
        
    def add_experiment_results(self,
                             config: Dict[str, Any],
                             metrics: Dict[str, float],
                             query_results: List[Dict[str, Any]]):
        experiment_data = {
            'config': config,
            'metrics': metrics,
            'results': query_results,
            'timestamp': pd.Timestamp.now()
        }
        self.metrics_data.append(experiment_data)
        
    def generate_performance_report(self) -> Dict[str, Any]:
        df = pd.DataFrame([
            {
                'latency': exp['metrics']['latency'],
                'accuracy': exp['metrics']['accuracy'],
                'retrieval_precision': exp['metrics']['retrieval_precision'],
                'config_type': exp['config']['type']
            }
            for exp in self.metrics_data
        ])
        
        performance_metrics = {
            'latency_stats': {
                'mean': df['latency'].mean(),
                'std': df['latency'].std(),
                'p95': df['latency'].quantile(0.95)
            },
            'accuracy_stats': {
                'mean': df['accuracy'].mean(),
                'std': df['accuracy'].std(),
                'by_config': df.groupby('config_type')['accuracy'].mean().to_dict()
            },
            'retrieval_stats': {
                'mean': df['retrieval_precision'].mean(),
                'by_config': df.groupby('config_type')['retrieval_precision'].mean().to_dict()
            }
        }
        
        # Create visualization
        fig = go.Figure()
        
        # Add latency trace
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['latency'],
            name='Latency',
            line=dict(color='blue')
        ))
        
        # Add accuracy trace
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['accuracy'],
            name='Accuracy',
            line=dict(color='green'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='RAG Pipeline Performance Over Time',
            xaxis_title='Experiment Number',
            yaxis_title='Latency (s)',
            yaxis2=dict(
                title='Accuracy',
                overlaying='y',
                side='right'
            )
        )
        
        return {
            'metrics': performance_metrics,
            'visualization': fig,
            'summary': self.generate_summary_insights(df)
        }
    
    def generate_summary_insights(self, df: pd.DataFrame) -> List[str]:
        insights = []
        
        # Performance trends
        latency_trend = df['latency'].diff().mean()
        accuracy_trend = df['accuracy'].diff().mean()
        
        if latency_trend < 0:
            insights.append(f"Latency improving by {abs(latency_trend):.3f}s per experiment")
        else:
            insights.append(f"Latency increasing by {latency_trend:.3f}s per experiment")
            
        if accuracy_trend > 0:
            insights.append(f"Accuracy improving by {accuracy_trend:.3f} per experiment")
        else:
            insights.append(f"Accuracy decreasing by {abs(accuracy_trend):.3f} per experiment")
            
        return insights
```

Slide 15: Additional Resources

*   arXiv:2304.03442 - "Retrieval-Augmented Generation for Large Language Models: A Survey"
*   arXiv:2312.05934 - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
*   arXiv:2309.07158 - "Cross-Encoder Reranking for Dense Retrieval: A Deep Dive"
*   arXiv:2312.12693 - "RAGMatch: Retrieval-Augmented Generation for Large-Scale Entity Matching"
*   arXiv:2310.03025 - "Improving Reranking by Learning to Score Initial Retrieval"
*   arXiv:2312.09044 - "RAG vs Fine-tuning: Pipeline, Challenges and Optimizations"

