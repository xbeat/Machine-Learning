## Retrieval Augmented Generation (RAG) in Large Language Models
Slide 1: RAG Components Implementation

A comprehensive implementation of core RAG system components including document loading, text chunking, and embedding generation using modern Python libraries. This foundation demonstrates the essential building blocks for creating a retrieval-augmented generation system.

```python
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import torch

class RAGComponents:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.chunk_size = 512
        self.overlap = 50
        
    def chunk_text(self, text: str) -> List[str]:
        # Implement sliding window chunking
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        
        for text in texts:
            # Tokenize and get model outputs
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token embedding as text representation    
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding)
            
        return np.vstack(embeddings)

# Usage example
rag = RAGComponents()
text = "Long document text here..."
chunks = rag.chunk_text(text)
embeddings = rag.generate_embeddings(chunks)
print(f"Generated {len(chunks)} chunks with embedding shape: {embeddings.shape}")
```

Slide 2: Vector Database Integration

Implementing a vector database interface for storing and retrieving document embeddings efficiently using FAISS. This component enables fast similarity search and nearest neighbor retrieval for matching user queries with relevant context.

```python
import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Document:
    text: str
    embedding: np.ndarray
    metadata: dict

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        
    def add_documents(self, docs: List[Document]):
        embeddings = np.vstack([doc.embedding for doc in docs])
        self.index.add(embeddings)
        self.documents.extend(docs)
        
    def similarity_search(self, 
                         query_embedding: np.ndarray, 
                         k: int = 5) -> List[Tuple[Document, float]]:
        # Perform k-NN search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), k
        )
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            results.append((self.documents[i], float(dist)))
            
        return results

# Example usage
store = VectorStore(embedding_dim=384)  # MiniLM embedding size
docs = [Document(text="Example text", 
                 embedding=np.random.randn(384), 
                 metadata={"source": "doc1"})]
store.add_documents(docs)
```

Slide 3: Query Processing Pipeline

The query processing pipeline handles user input transformation, leveraging the embedding model to convert natural language queries into vector representations for semantic search. This component ensures effective matching between queries and stored documents.

```python
import torch
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict

class QueryProcessor:
    def __init__(self, rag_components, vector_store):
        self.rag = rag_components
        self.store = vector_store
        
    def process_query(self, query: str, top_k: int = 3) -> List[SearchResult]:
        # Generate query embedding
        query_embedding = self.rag.generate_embeddings([query])[0]
        
        # Perform similarity search
        results = self.store.similarity_search(query_embedding, k=top_k)
        
        # Format results
        search_results = []
        for doc, score in results:
            search_results.append(
                SearchResult(
                    text=doc.text,
                    score=score,
                    metadata=doc.metadata
                )
            )
            
        return search_results

# Example usage
processor = QueryProcessor(rag_components=rag, vector_store=store)
results = processor.process_query("Example query text")
for result in results:
    print(f"Score: {result.score:.4f}, Text: {result.text[:100]}")
```

Slide 4: Context Augmentation and Prompt Engineering

Advanced prompt engineering techniques for RAG systems, implementing dynamic context injection and prompt templates. This component orchestrates how retrieved context is combined with user queries to generate optimal LLM inputs.

```python
from string import Template
from typing import List, Dict, Optional

class PromptManager:
    def __init__(self):
        self.templates = {
            'default': Template(
                """Context information:
                $context
                
                Given the context above, please answer the following question:
                $query
                
                Answer:"""
            ),
            'analytical': Template(
                """Analyze the following information:
                $context
                
                Question for analysis:
                $query
                
                Detailed analysis:"""
            )
        }
        
    def construct_prompt(self,
                        query: str,
                        contexts: List[str],
                        template_key: str = 'default',
                        max_context_length: int = 2000) -> str:
        
        # Format and truncate context
        formatted_context = '\n\n'.join([
            f"[{i+1}] {ctx[:max_context_length]}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Generate final prompt
        return self.templates[template_key].substitute(
            context=formatted_context,
            query=query
        )
    
# Usage example
prompt_manager = PromptManager()
contexts = ["First relevant context", "Second relevant context"]
prompt = prompt_manager.construct_prompt(
    query="What are the key points?",
    contexts=contexts,
    template_key='analytical'
)
print(prompt)
```

Slide 5: LLM Integration Layer

Implementation of the LLM integration layer that handles communication with different language models, manages response streaming, and implements retry logic for robust operation in production environments.

```python
import openai
from typing import Generator, Optional
import backoff
import asyncio

class LLMInterface:
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 max_retries: int = 3,
                 temperature: float = 0.7):
        self.model = model_name
        self.max_retries = max_retries
        self.temperature = temperature
        
    @backoff.on_exception(backoff.expo, 
                         (openai.error.RateLimitError,
                          openai.error.ServiceUnavailableError),
                         max_tries=3)
    async def generate_response(self,
                              prompt: str,
                              stream: bool = True) -> Generator[str, None, None]:
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                stream=stream
            )
            
            if stream:
                async for chunk in response:
                    if chunk and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content
                
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")

# Usage example
async def main():
    llm = LLMInterface()
    async for chunk in llm.generate_response("Explain RAG systems"):
        print(chunk, end='', flush=True)

# Run the example
asyncio.run(main())
```

Slide 6: Document Preprocessing Pipeline

Comprehensive document preprocessing pipeline that handles multiple file formats, performs cleaning, and implements advanced text extraction techniques for optimal chunking and embedding generation.

```python
import PyPDF2
import docx
import re
from typing import List, Dict, Union
from pathlib import Path

class DocumentPreprocessor:
    def __init__(self, clean_text: bool = True):
        self.clean_text = clean_text
        
    def load_document(self, file_path: Union[str, Path]) -> str:
        file_path = Path(file_path)
        
        if file_path.suffix == '.pdf':
            return self._extract_pdf(file_path)
        elif file_path.suffix == '.docx':
            return self._extract_docx(file_path)
        elif file_path.suffix == '.txt':
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_pdf(self, file_path: Path) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return self._clean_text(text) if self.clean_text else text
    
    def _extract_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return self._clean_text(text) if self.clean_text else text
    
    def _extract_txt(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return self._clean_text(text) if self.clean_text else text
    
    def _clean_text(self, text: str) -> str:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Normalize line endings
        text = text.replace('\n', ' ').strip()
        return text

# Usage example
preprocessor = DocumentPreprocessor()
text = preprocessor.load_document("example.pdf")
print(f"Extracted and cleaned text length: {len(text)}")
```

Slide 7: Advanced Chunking Strategies

Implementation of sophisticated text chunking algorithms that preserve semantic coherence and maintain context boundaries. This component uses multiple strategies including sentence-based, paragraph-based, and semantic-based chunking methods.

```python
import spacy
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

class AdvancedChunker:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf = TfidfVectorizer()
        
    def semantic_chunks(self, 
                       text: str, 
                       target_size: int = 512,
                       overlap: int = 50) -> List[str]:
        # Get sentence boundaries
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Calculate semantic similarity between sentences
        tfidf_matrix = self.tfidf.fit_transform(sentences)
        similarities = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.nlp(sentence))
            
            if current_size + sentence_tokens > target_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Add overlap from previous chunk
                    current_chunk = current_chunk[-overlap:]
                    current_size = sum(len(self.nlp(s)) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def paragraph_chunks(self, 
                        text: str, 
                        min_size: int = 200) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(self.nlp(para))
            
            if current_size + para_size > min_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

# Usage example
chunker = AdvancedChunker()
text = """Long document text with multiple paragraphs..."""
semantic_chunks = chunker.semantic_chunks(text)
paragraph_chunks = chunker.paragraph_chunks(text)
print(f"Generated {len(semantic_chunks)} semantic chunks")
print(f"Generated {len(paragraph_chunks)} paragraph chunks")
```

Slide 8: Embedding Cache and Management

Implementation of an efficient embedding cache system to prevent redundant embedding generation and optimize memory usage in production RAG systems. Includes LRU cache and persistence mechanisms.

```python
from functools import lru_cache
import pickle
from pathlib import Path
import hashlib
from typing import Dict, Optional
import numpy as np

class EmbeddingCache:
    def __init__(self, 
                 cache_dir: str = "./embedding_cache",
                 max_cache_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()
        
    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, 
                     text: str, 
                     generate_fn) -> np.ndarray:
        text_hash = self._get_hash(text)
        
        # Check in-memory cache
        if text_hash in self.cache:
            return self.cache[text_hash]
            
        # Check disk cache
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
                self.cache[text_hash] = embedding
                return embedding
                
        # Generate new embedding
        embedding = generate_fn(text)
        
        # Update cache
        self.cache[text_hash] = embedding
        
        # Persist to disk
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
            
        # Manage cache size
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()
            
        return embedding
    
    def _cleanup_cache(self):
        # Remove oldest entries
        remove_count = len(self.cache) - self.max_cache_size
        for key in list(self.cache.keys())[:remove_count]:
            del self.cache[key]
            
    def _load_cache(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            with open(cache_file, 'rb') as f:
                text_hash = cache_file.stem
                self.cache[text_hash] = pickle.load(f)

# Usage example
def dummy_embedding_generator(text: str) -> np.ndarray:
    return np.random.randn(384)  # Simulated embedding generation

cache = EmbeddingCache()
text = "Example text for embedding"
embedding = cache.get_embedding(text, dummy_embedding_generator)
print(f"Generated/retrieved embedding shape: {embedding.shape}")
```

Slide 9: Real-world RAG Implementation Example - Document QA System

A complete implementation of a document question-answering system using RAG architecture. This example demonstrates the integration of all previously discussed components into a production-ready system.

```python
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class QAResponse:
    answer: str
    contexts: List[str]
    confidence: float

class DocumentQASystem:
    def __init__(self,
                 embedding_cache: EmbeddingCache,
                 vector_store: VectorStore,
                 llm_interface: LLMInterface,
                 chunk_size: int = 512):
        self.embedding_cache = embedding_cache
        self.vector_store = vector_store
        self.llm = llm_interface
        self.chunker = AdvancedChunker()
        self.prompt_manager = PromptManager()
        
    async def process_document(self, file_path: str):
        # Load and preprocess document
        preprocessor = DocumentPreprocessor()
        text = preprocessor.load_document(file_path)
        
        # Generate chunks
        chunks = self.chunker.semantic_chunks(text, 
                                            target_size=self.chunk_size)
        
        # Generate and store embeddings
        for chunk in chunks:
            embedding = self.embedding_cache.get_embedding(
                chunk,
                lambda x: self.llm.generate_embedding(x)
            )
            self.vector_store.add_documents([
                Document(text=chunk, 
                        embedding=embedding,
                        metadata={"source": file_path})
            ])
    
    async def answer_question(self, 
                            question: str,
                            top_k: int = 3) -> QAResponse:
        # Generate question embedding
        question_embedding = self.embedding_cache.get_embedding(
            question,
            lambda x: self.llm.generate_embedding(x)
        )
        
        # Retrieve relevant contexts
        results = self.vector_store.similarity_search(
            question_embedding,
            k=top_k
        )
        
        contexts = [doc.text for doc, _ in results]
        scores = [score for _, score in results]
        
        # Generate prompt
        prompt = self.prompt_manager.construct_prompt(
            query=question,
            contexts=contexts
        )
        
        # Generate answer
        answer_stream = self.llm.generate_response(prompt, stream=False)
        answer = await anext(answer_stream)
        
        return QAResponse(
            answer=answer,
            contexts=contexts,
            confidence=float(np.mean(scores))
        )

# Usage example
async def main():
    # Initialize components
    cache = EmbeddingCache()
    store = VectorStore(embedding_dim=384)
    llm = LLMInterface()
    
    # Create QA system
    qa_system = DocumentQASystem(cache, store, llm)
    
    # Process documents
    await qa_system.process_document("example_doc.pdf")
    
    # Ask questions
    response = await qa_system.answer_question(
        "What are the key findings in the document?"
    )
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print("\nRelevant contexts:")
    for ctx in response.contexts:
        print(f"- {ctx[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 10: Performance Optimization - Re-ranking Implementation

Implementation of a sophisticated re-ranking system that improves retrieval accuracy by applying multiple scoring methods and cross-attention between query and documents.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict

class CrossAttentionReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    @torch.no_grad()
    def rerank(self,
              query: str,
              documents: List[str],
              top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        
        # Prepare inputs
        pairs = [(query, doc) for doc in documents]
        inputs = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Generate cross-attention scores
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        scores = F.softmax(torch.matmul(
            embeddings, embeddings.transpose(0, 1)
        ), dim=1)
        
        # Sort by score
        doc_scores = [(doc, score.item()) 
                     for doc, score in zip(documents, scores[:, 0])]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            doc_scores = doc_scores[:top_k]
            
        return doc_scores

class HybridReranker:
    def __init__(self):
        self.cross_attention = CrossAttentionReranker()
        
    def rerank_with_metadata(self,
                           query: str,
                           documents: List[Dict],
                           weights: Dict[str, float] = None) -> List[Dict]:
        if weights is None:
            weights = {
                "vector_score": 0.3,
                "cross_attention": 0.5,
                "metadata_score": 0.2
            }
            
        # Get cross-attention scores
        texts = [doc["text"] for doc in documents]
        cross_scores = self.cross_attention.rerank(query, texts)
        
        # Combine scores
        final_scores = []
        for doc, cross_score in zip(documents, cross_scores):
            score = (
                weights["vector_score"] * doc["vector_score"] +
                weights["cross_attention"] * cross_score[1] +
                weights["metadata_score"] * self._calculate_metadata_score(doc)
            )
            final_scores.append((doc, score))
            
        # Sort and return
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in final_scores]
    
    def _calculate_metadata_score(self, doc: Dict) -> float:
        # Custom metadata scoring logic
        score = 0.0
        if doc.get("freshness_date"):
            # Boost recent documents
            days_old = (datetime.now() - doc["freshness_date"]).days
            score += max(0, 1 - (days_old / 365))
        return score

# Usage example
reranker = HybridReranker()
documents = [
    {"text": "First document", "vector_score": 0.8},
    {"text": "Second document", "vector_score": 0.7}
]
reranked_docs = reranker.rerank_with_metadata(
    "example query",
    documents
)
```

Slide 11: Real-time Monitoring and Analytics System

Implementation of a comprehensive monitoring system for RAG applications that tracks performance metrics, latency, and relevance scores in real-time, enabling continuous optimization and quality assurance.

```python
import time
from datetime import datetime
import numpy as np
from typing import Dict, List, Any
from collections import deque
import json

class RAGMonitor:
    def __init__(self, window_size: int = 1000):
        self.metrics = {
            'latency': deque(maxlen=window_size),
            'relevance_scores': deque(maxlen=window_size),
            'token_usage': deque(maxlen=window_size),
            'cache_hits': deque(maxlen=window_size),
            'error_rates': deque(maxlen=window_size)
        }
        self.query_log = deque(maxlen=window_size)
        
    def log_query(self, 
                  query: str,
                  response: Dict[str, Any],
                  metrics: Dict[str, float]):
        timestamp = datetime.now().isoformat()
        
        # Record metrics
        self.metrics['latency'].append(metrics.get('latency', 0))
        self.metrics['relevance_scores'].append(
            metrics.get('relevance_score', 0)
        )
        self.metrics['token_usage'].append(metrics.get('tokens_used', 0))
        self.metrics['cache_hits'].append(metrics.get('cache_hit', False))
        self.metrics['error_rates'].append(metrics.get('error', False))
        
        # Log query details
        query_record = {
            'timestamp': timestamp,
            'query': query,
            'response_length': len(response.get('answer', '')),
            'num_contexts': len(response.get('contexts', [])),
            'metrics': metrics
        }
        self.query_log.append(query_record)
        
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'avg_latency': np.mean(self.metrics['latency']),
            'p95_latency': np.percentile(self.metrics['latency'], 95),
            'avg_relevance': np.mean(self.metrics['relevance_scores']),
            'cache_hit_rate': np.mean(self.metrics['cache_hits']),
            'error_rate': np.mean(self.metrics['error_rates']),
            'total_tokens': sum(self.metrics['token_usage']),
            'queries_processed': len(self.query_log)
        }
        return stats
    
    def export_logs(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(list(self.query_log), f, indent=2)

class RAGQualityAnalyzer:
    def __init__(self, monitor: RAGMonitor):
        self.monitor = monitor
        
    def analyze_quality(self) -> Dict[str, Any]:
        logs = list(self.monitor.query_log)
        
        # Analyze response consistency
        response_lengths = [log['response_length'] for log in logs]
        context_counts = [log['num_contexts'] for log in logs]
        
        analysis = {
            'response_length': {
                'mean': np.mean(response_lengths),
                'std': np.std(response_lengths),
                'min': min(response_lengths),
                'max': max(response_lengths)
            },
            'context_usage': {
                'mean': np.mean(context_counts),
                'std': np.std(context_counts)
            },
            'performance_metrics': self.monitor.get_statistics()
        }
        
        return analysis

# Usage example
async def monitored_qa_system():
    monitor = RAGMonitor()
    qa_system = DocumentQASystem(...)
    
    query = "What are the key concepts?"
    start_time = time.time()
    
    try:
        response = await qa_system.answer_question(query)
        latency = time.time() - start_time
        
        metrics = {
            'latency': latency,
            'relevance_score': response.confidence,
            'tokens_used': len(response.answer.split()),
            'cache_hit': True,  # Determine from cache status
            'error': False
        }
        
        monitor.log_query(query, 
                         {'answer': response.answer, 
                          'contexts': response.contexts},
                         metrics)
        
        # Analyze quality periodically
        analyzer = RAGQualityAnalyzer(monitor)
        quality_report = analyzer.analyze_quality()
        print(f"Quality Report: {quality_report}")
        
    except Exception as e:
        metrics = {
            'latency': time.time() - start_time,
            'error': True
        }
        monitor.log_query(query, {}, metrics)
        raise e

# Run the monitored system
await monitored_qa_system()
```

Slide 12: Retrieval Strategy Optimization

Implementation of advanced retrieval strategies including hybrid search, semantic clustering, and dynamic context window adjustment based on query complexity and document structure.

```python
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional
import numpy as np

class AdvancedRetriever:
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_model: Any,
                 n_clusters: int = 5):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.kmeans = None
        
    def build_semantic_clusters(self):
        # Get all embeddings
        embeddings = np.array([
            doc.embedding for doc in self.vector_store.documents
        ])
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(embeddings)
        
        # Assign clusters to documents
        for doc, cluster_id in zip(
            self.vector_store.documents,
            self.kmeans.labels_
        ):
            doc.metadata['cluster_id'] = int(cluster_id)
            
    def retrieve(self,
                query: str,
                strategy: str = 'hybrid',
                top_k: int = 5) -> List[Document]:
        if strategy == 'hybrid':
            return self._hybrid_search(query, top_k)
        elif strategy == 'cluster':
            return self._cluster_based_search(query, top_k)
        else:
            return self._semantic_search(query, top_k)
            
    def _hybrid_search(self,
                      query: str,
                      top_k: int) -> List[Document]:
        # Get semantic search results
        semantic_results = self._semantic_search(query, top_k * 2)
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine results with weights
        combined_results = {}
        for doc, score in semantic_results:
            combined_results[doc.id] = {
                'doc': doc,
                'score': 0.7 * score
            }
            
        for doc, score in keyword_results:
            if doc.id in combined_results:
                combined_results[doc.id]['score'] += 0.3 * score
            else:
                combined_results[doc.id] = {
                    'doc': doc,
                    'score': 0.3 * score
                }
        
        # Sort and return top_k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        return [item['doc'] for item in sorted_results[:top_k]]
    
    def _cluster_based_search(self,
                            query: str,
                            top_k: int) -> List[Document]:
        if self.kmeans is None:
            self.build_semantic_clusters()
            
        # Get query embedding and find nearest cluster
        query_embedding = self.embedding_model.encode(query)
        cluster_id = self.kmeans.predict([query_embedding])[0]
        
        # Get documents from the same cluster
        cluster_docs = [
            doc for doc in self.vector_store.documents
            if doc.metadata.get('cluster_id') == cluster_id
        ]
        
        # Perform semantic search within cluster
        results = self.vector_store.similarity_search(
            query_embedding,
            k=top_k,
            filter_fn=lambda doc: doc in cluster_docs
        )
        
        return [doc for doc, _ in results]

# Usage example
retriever = AdvancedRetriever(vector_store, embedding_model)
retriever.build_semantic_clusters()

results = retriever.retrieve(
    "Example query",
    strategy='hybrid',
    top_k=5
)
```

Slide 13: Evaluation and Testing Framework

Implementation of a comprehensive evaluation framework for RAG systems that measures retrieval accuracy, answer quality, and system performance through automated test suites and human feedback integration.

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

@dataclass
class EvaluationMetrics:
    retrieval_precision: float
    answer_relevance: float
    answer_completeness: float
    response_time: float
    rouge_scores: Dict[str, float]

class RAGEvaluator:
    def __init__(self, 
                 rag_system: Any,
                 embedding_model: Any):
        self.rag_system = rag_system
        self.embedding_model = embedding_model
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    async def evaluate_system(self,
                            test_cases: List[Dict],
                            metrics_callback: Optional[Callable] = None
                            ) -> Dict[str, float]:
        results = []
        
        for test_case in test_cases:
            metrics = await self.evaluate_single_case(test_case)
            results.append(metrics)
            
            if metrics_callback:
                metrics_callback(metrics)
                
        # Aggregate results
        avg_metrics = self._aggregate_metrics(results)
        return avg_metrics
    
    async def evaluate_single_case(self,
                                 test_case: Dict) -> EvaluationMetrics:
        query = test_case['query']
        ground_truth = test_case['ground_truth']
        relevant_contexts = test_case.get('relevant_contexts', [])
        
        # Measure response time
        start_time = time.time()
        response = await self.rag_system.answer_question(query)
        response_time = time.time() - start_time
        
        # Calculate metrics
        retrieval_precision = self._calculate_retrieval_precision(
            response.contexts,
            relevant_contexts
        )
        
        answer_relevance = self._calculate_semantic_similarity(
            response.answer,
            ground_truth
        )
        
        answer_completeness = self._calculate_completeness(
            response.answer,
            ground_truth
        )
        
        rouge_scores = self._calculate_rouge_scores(
            response.answer,
            ground_truth
        )
        
        return EvaluationMetrics(
            retrieval_precision=retrieval_precision,
            answer_relevance=answer_relevance,
            answer_completeness=answer_completeness,
            response_time=response_time,
            rouge_scores=rouge_scores
        )
    
    def _calculate_retrieval_precision(self,
                                    retrieved_contexts: List[str],
                                    relevant_contexts: List[str]) -> float:
        retrieved_embeddings = self.embedding_model.encode(retrieved_contexts)
        relevant_embeddings = self.embedding_model.encode(relevant_contexts)
        
        # Calculate similarity matrix
        similarities = cosine_similarity(
            retrieved_embeddings,
            relevant_embeddings
        )
        
        # Count matches above threshold
        matches = (similarities > 0.8).sum()
        precision = matches / len(retrieved_contexts)
        
        return float(precision)
    
    def _calculate_semantic_similarity(self,
                                    response: str,
                                    ground_truth: str) -> float:
        response_embedding = self.embedding_model.encode([response])[0]
        truth_embedding = self.embedding_model.encode([ground_truth])[0]
        
        similarity = cosine_similarity(
            response_embedding.reshape(1, -1),
            truth_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def _calculate_completeness(self,
                              response: str,
                              ground_truth: str) -> float:
        # Use ROUGE-L as a proxy for completeness
        scores = self.rouge_scorer.score(ground_truth, response)
        return float(scores['rougeL'].fmeasure)
    
    def _calculate_rouge_scores(self,
                              response: str,
                              ground_truth: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(ground_truth, response)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def _aggregate_metrics(self,
                         metrics_list: List[EvaluationMetrics]
                         ) -> Dict[str, float]:
        aggregated = {
            'avg_retrieval_precision': np.mean(
                [m.retrieval_precision for m in metrics_list]
            ),
            'avg_answer_relevance': np.mean(
                [m.answer_relevance for m in metrics_list]
            ),
            'avg_answer_completeness': np.mean(
                [m.answer_completeness for m in metrics_list]
            ),
            'avg_response_time': np.mean(
                [m.response_time for m in metrics_list]
            ),
            'avg_rouge_scores': {
                metric: np.mean([m.rouge_scores[metric] 
                               for m in metrics_list])
                for metric in ['rouge1', 'rouge2', 'rougeL']
            }
        }
        return aggregated

# Usage example
test_cases = [
    {
        'query': 'What is RAG?',
        'ground_truth': 'RAG is a method that combines retrieval...',
        'relevant_contexts': [
            'RAG systems are designed to...',
            'The key components of RAG include...'
        ]
    }
]

evaluator = RAGEvaluator(rag_system, embedding_model)
results = await evaluator.evaluate_system(
    test_cases,
    metrics_callback=lambda m: print(f"Case metrics: {m}")
)
print(f"Overall system performance: {results}")
```

Slide 14: Additional Resources

Relevant research papers on RAG systems and implementations:

1.  [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2.  [https://arxiv.org/abs/2312.05934](https://arxiv.org/abs/2312.05934) - "A Survey on Retrieval-Augmented Generation for Large Language Models"
3.  [https://arxiv.org/abs/2310.03025](https://arxiv.org/abs/2310.03025) - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
4.  [https://arxiv.org/abs/2309.07158](https://arxiv.org/abs/2309.07158) - "Advanced RAG Techniques: A Guide to Better LLM RAG Systems"
5.  [https://arxiv.org/abs/2312.17276](https://arxiv.org/abs/2312.17276) - "Context-faithful Prompting for Large Language Models"

