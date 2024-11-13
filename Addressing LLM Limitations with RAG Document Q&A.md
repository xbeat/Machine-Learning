## Addressing LLM Limitations with RAG Document Q&A
Slide 1: Introduction to RAG Architecture

Retrieval Augmented Generation (RAG) is a powerful technique that combines document retrieval with language model generation. The architecture consists of an indexing phase where documents are processed and stored, a retrieval phase that finds relevant content, and a generation phase that produces natural language responses.

```python
# Basic RAG Pipeline Implementation
import chromadb
from langchain import OpenAI, LLMChain
from langchain.embeddings import OpenAIEmbeddings

class RAGPipeline:
    def __init__(self, api_key):
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("documents")
        
    def index_documents(self, documents):
        # Convert documents to embeddings and store
        embeddings = self.embeddings.embed_documents(documents)
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        
    def retrieve(self, query, k=3):
        # Retrieve relevant documents
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results['documents'][0]
```

Slide 2: Document Preprocessing

Document preprocessing is crucial for effective RAG systems. This involves cleaning text, splitting documents into manageable chunks, and removing irrelevant information. The chunk size should be optimized based on the model's context window and retrieval requirements.

```python
from typing import List
import re
import numpy as np

class DocumentPreprocessor:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def clean_text(self, text: str) -> str:
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s.]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def chunk_document(self, document: str) -> List[str]:
        # Split document into overlapping chunks
        words = document.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if len(chunk) > 0:
                chunks.append(chunk)
                
        return chunks

# Example usage
processor = DocumentPreprocessor()
document = "Long document text here..."
cleaned_text = processor.clean_text(document)
chunks = processor.chunk_document(cleaned_text)
print(f"Generated {len(chunks)} chunks")
```

Slide 3: Vector Store Implementation

Vector stores are essential components in RAG systems for efficient similarity search. They index document embeddings and enable fast retrieval of relevant content during query time using approximate nearest neighbor search algorithms.

```python
import faiss
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Document:
    id: str
    content: str
    embedding: np.ndarray

class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: Dict[int, Document] = {}
        
    def add_documents(self, documents: List[Document]):
        embeddings = np.vstack([doc.embedding for doc in documents])
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store document mapping
        start_id = len(self.documents)
        for i, doc in enumerate(documents):
            self.documents[start_id + i] = doc
            
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Document]:
        # Ensure query embedding is 2D
        query_embedding = query_embedding.reshape(1, -1)
        
        # Perform similarity search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return matched documents
        return [self.documents[idx] for idx in indices[0]]
```

Slide 4: Embedding Generation

Document embedding generation is a critical step in RAG systems that transforms text into dense vector representations. These embeddings capture semantic meaning and enable efficient similarity search during retrieval operations.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def generate_embedding(self, text: str) -> np.ndarray:
        # Tokenize and prepare input
        inputs = self.tokenizer(text, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy()
    
    def batch_generate(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.generate_embedding(batch)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

# Example usage
generator = EmbeddingGenerator()
text = "Sample document for embedding generation"
embedding = generator.generate_embedding(text)
print(f"Generated embedding shape: {embedding.shape}")
```

Slide 5: Retriever Implementation

The retriever component is responsible for finding the most relevant documents given a query. It uses similarity metrics to rank documents and implements various strategies like re-ranking and filtering to improve result quality.

```python
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class RetrievedDocument:
    content: str
    score: float

class RAGRetriever:
    def __init__(self, vector_store, embedding_generator, top_k: int = 3):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        
    def retrieve(self, query: str) -> List[RetrievedDocument]:
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Retrieve candidates from vector store
        candidates = self.vector_store.search(query_embedding, k=self.top_k * 2)
        
        # Rerank using cosine similarity
        scores = cosine_similarity(
            query_embedding,
            np.vstack([doc.embedding for doc in candidates])
        )[0]
        
        # Sort and filter results
        ranked_results = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )[:self.top_k]
        
        return [
            RetrievedDocument(doc.content, score)
            for doc, score in ranked_results
        ]
        
    def retrieve_with_filter(self, query: str, 
                           min_score: float = 0.7) -> List[RetrievedDocument]:
        results = self.retrieve(query)
        return [doc for doc in results if doc.score >= min_score]

# Example usage
retriever = RAGRetriever(vector_store, embedding_generator)
results = retriever.retrieve("Sample query")
for i, doc in enumerate(results):
    print(f"Result {i+1} - Score: {doc.score:.3f}")
```

Slide 6: Context Window Management

Managing context windows effectively is crucial for optimal RAG performance. This implementation demonstrates how to dynamically adjust document chunks based on token limits and ensure retrieved content fits within model constraints.

```python
import tiktoken
from typing import List, Dict

class ContextManager:
    def __init__(self, model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 4096):
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def fit_to_context(self, documents: List[RetrievedDocument],
                      query: str,
                      system_prompt: str) -> List[RetrievedDocument]:
        # Calculate fixed token costs
        query_tokens = self.count_tokens(query)
        prompt_tokens = self.count_tokens(system_prompt)
        reserved_tokens = 500  # For response generation
        
        available_tokens = (self.max_tokens - query_tokens - 
                          prompt_tokens - reserved_tokens)
        
        fitted_docs = []
        current_tokens = 0
        
        for doc in documents:
            doc_tokens = self.count_tokens(doc.content)
            if current_tokens + doc_tokens <= available_tokens:
                fitted_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break
                
        return fitted_docs

# Example usage
context_manager = ContextManager()
fitted_docs = context_manager.fit_to_context(
    retrieved_docs,
    "What is machine learning?",
    "You are a helpful AI assistant."
)
```

Slide 7: Query Generation and Processing

Query processing involves transforming user questions into effective search queries. This implementation includes query expansion, decomposition for complex questions, and handling of different query types to improve retrieval accuracy.

```python
from typing import List, Set
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')

class QueryProcessor:
    def __init__(self):
        self.stop_words = set(['is', 'the', 'a', 'an', 'and', 'or', 'but'])
        
    def expand_query(self, query: str) -> str:
        tokens = word_tokenize(query.lower())
        expanded_terms = set()
        
        for token in tokens:
            if token not in self.stop_words:
                # Add original term
                expanded_terms.add(token)
                # Add synonyms
                synsets = wordnet.synsets(token)
                for syn in synsets[:2]:  # Limit to top 2 synsets
                    for lemma in syn.lemmas():
                        expanded_terms.add(lemma.name())
                        
        return ' '.join(expanded_terms)
    
    def decompose_complex_query(self, query: str) -> List[str]:
        # Split complex queries into sub-queries
        if '?' in query:
            sub_queries = [q.strip() + '?' for q in query.split('?') if q.strip()]
        else:
            sub_queries = [query]
            
        return sub_queries
    
    def process_query(self, query: str, 
                     expand: bool = True) -> List[str]:
        sub_queries = self.decompose_complex_query(query)
        
        if expand:
            return [self.expand_query(q) for q in sub_queries]
        return sub_queries

# Example usage
processor = QueryProcessor()
query = "What are the main types of neural networks and their applications?"
processed_queries = processor.process_query(query)
print(f"Original query: {query}")
print(f"Processed queries: {processed_queries}")
```

Slide 8: Response Generation

The generation phase combines retrieved contexts with the original query to produce coherent and accurate responses. This implementation includes prompt engineering and response formatting techniques.

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class GenerationContext:
    query: str
    retrieved_docs: List[RetrievedDocument]
    system_prompt: str

class ResponseGenerator:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def format_context(self, context: GenerationContext) -> str:
        # Combine retrieved documents into context
        doc_contexts = []
        for i, doc in enumerate(context.retrieved_docs, 1):
            doc_contexts.append(f"[Document {i}]: {doc.content}")
            
        formatted_prompt = f"""
{context.system_prompt}

Relevant Information:
{'\n'.join(doc_contexts)}

User Query: {context.query}

Based on the provided information, please answer the query. 
If the information is insufficient, please state so clearly.
"""
        return formatted_prompt
    
    def generate_response(self, context: GenerationContext,
                         max_tokens: int = 1000) -> str:
        formatted_prompt = self.format_context(context)
        
        response = self.llm.generate(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return self.post_process_response(response)
    
    def post_process_response(self, response: str) -> str:
        # Clean up response formatting
        response = response.strip()
        # Remove redundant citations if present
        response = re.sub(r'\[\d+\]', '', response)
        return response

# Example usage
generator = ResponseGenerator(llm_client)
context = GenerationContext(
    query="Explain deep learning",
    retrieved_docs=retrieved_docs,
    system_prompt="You are an AI expert providing clear explanations."
)
response = generator.generate_response(context)
print(response)
```

Slide 9: Real-world Implementation - Document Q&A System

This implementation demonstrates a complete RAG-based document question-answering system. The system processes PDF documents, indexes their content, and answers user queries by combining relevant document sections with language model capabilities.

```python
import fitz  # PyMuPDF
from pathlib import Path
import numpy as np
from typing import Dict, List

class DocumentQASystem:
    def __init__(self, embedding_dim: int = 384):
        self.preprocessor = DocumentPreprocessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore(embedding_dim)
        self.query_processor = QueryProcessor()
        self.context_manager = ContextManager()
        
    def load_pdf(self, pdf_path: str) -> List[str]:
        doc = fitz.open(pdf_path)
        text_chunks = []
        
        for page in doc:
            text = page.get_text()
            # Clean and chunk the text
            cleaned_text = self.preprocessor.clean_text(text)
            chunks = self.preprocessor.chunk_document(cleaned_text)
            text_chunks.extend(chunks)
            
        return text_chunks
    
    def index_document(self, file_path: str):
        # Load and process document
        chunks = self.load_pdf(file_path)
        
        # Generate embeddings
        embeddings = self.embedding_generator.batch_generate(chunks)
        
        # Create document objects
        documents = [
            Document(
                id=f"doc_{i}",
                content=chunk,
                embedding=embedding
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
    def answer_query(self, query: str) -> str:
        # Process and expand query
        processed_query = self.query_processor.process_query(query)[0]
        
        # Generate query embedding and retrieve relevant docs
        query_embedding = self.embedding_generator.generate_embedding(processed_query)
        retrieved_docs = self.vector_store.search(query_embedding)
        
        # Fit documents to context window
        fitted_docs = self.context_manager.fit_to_context(
            retrieved_docs,
            query,
            "You are a helpful assistant providing accurate answers based on the documents."
        )
        
        # Generate response
        generation_context = GenerationContext(
            query=query,
            retrieved_docs=fitted_docs,
            system_prompt="Answer based on the provided document contexts."
        )
        
        response = self.response_generator.generate_response(generation_context)
        return response

# Example usage
qa_system = DocumentQASystem()
qa_system.index_document("technical_document.pdf")
answer = qa_system.answer_query("What are the main findings in the document?")
print(f"Answer: {answer}")
```

Slide 10: Performance Metrics and Evaluation

A comprehensive evaluation framework for RAG systems that measures retrieval accuracy, response quality, and system performance. This implementation includes standard metrics and custom evaluation approaches.

```python
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Dict, Tuple
import time
import numpy as np

class RAGEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_retrieval(self, 
                          relevant_docs: List[str],
                          retrieved_docs: List[str],
                          k: int = None) -> Dict[str, float]:
        if k:
            retrieved_docs = retrieved_docs[:k]
            
        # Calculate precision, recall, and F1
        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)
        
        true_positives = len(relevant_set.intersection(retrieved_set))
        
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate_response_quality(self,
                                generated_responses: List[str],
                                ground_truth: List[str],
                                rouge_evaluator) -> Dict[str, float]:
        # Calculate ROUGE scores
        rouge_scores = rouge_evaluator.compute(
            predictions=generated_responses,
            references=ground_truth,
            use_agregator=True
        )
        
        return {
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL']
        }
    
    def evaluate_system_performance(self,
                                  queries: List[str],
                                  rag_system) -> Dict[str, float]:
        latencies = []
        
        for query in queries:
            start_time = time.time()
            _ = rag_system.answer_query(query)
            latency = time.time() - start_time
            latencies.append(latency)
            
        return {
            "mean_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99)
        }

# Example usage
evaluator = RAGEvaluator()
retrieval_metrics = evaluator.evaluate_retrieval(
    relevant_docs=["doc1", "doc2"],
    retrieved_docs=["doc1", "doc3"]
)
print(f"Retrieval Metrics: {retrieval_metrics}")
```

Slide 11: Optimization and Caching

Implementation of advanced caching strategies and query optimization techniques to improve RAG system performance. This includes embedding cache, document cache, and query result caching mechanisms.

```python
from functools import lru_cache
import hashlib
import time
from typing import Dict, Any, Optional
import json

class RAGCache:
    def __init__(self, cache_size: int = 1000):
        self.embedding_cache = {}
        self.query_cache = {}
        self.cache_size = cache_size
        
    def get_cache_key(self, text: str) -> str:
        # Generate deterministic cache key
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        cache_key = self.get_cache_key(text)
        return self.embedding_cache.get(cache_key)
    
    def set_embedding(self, text: str, embedding: np.ndarray):
        cache_key = self.get_cache_key(text)
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        self.embedding_cache[cache_key] = embedding
        
    def get_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        cache_key = self.get_cache_key(query)
        cached_result = self.query_cache.get(cache_key)
        
        if cached_result:
            current_time = time.time()
            if current_time - cached_result['timestamp'] < 3600:  # 1 hour TTL
                return cached_result['result']
            else:
                del self.query_cache[cache_key]
        return None
    
    def set_query_result(self, query: str, result: Dict[str, Any]):
        cache_key = self.get_cache_key(query)
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

class OptimizedRAGSystem:
    def __init__(self):
        self.cache = RAGCache()
        self.batch_size = 32
        
    def batch_process_documents(self, documents: List[str]) -> List[np.ndarray]:
        embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_embeddings = []
            
            for doc in batch:
                cached_embedding = self.cache.get_embedding(doc)
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                else:
                    embedding = self.embedding_generator.generate_embedding(doc)
                    self.cache.set_embedding(doc, embedding)
                    batch_embeddings.append(embedding)
                    
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def optimized_query(self, query: str) -> Dict[str, Any]:
        # Check cache first
        cached_result = self.cache.get_query_result(query)
        if cached_result:
            return cached_result
            
        # Process query and cache result
        result = self.process_query(query)
        self.cache.set_query_result(query, result)
        return result

# Example usage
optimized_system = OptimizedRAGSystem()
result = optimized_system.optimized_query("What is machine learning?")
print(f"Query result: {result}")
```

Slide 12: Error Handling and Monitoring

Robust error handling and monitoring implementation for RAG systems, including detailed logging, error recovery mechanisms, and system health monitoring.

```python
import logging
from datetime import datetime
import traceback
from typing import Optional, Dict, Any

class RAGMonitor:
    def __init__(self):
        self.logger = self._setup_logger()
        self.metrics = {
            'queries_processed': 0,
            'errors': 0,
            'avg_latency': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('RAGMonitor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('rag_system.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_query(self, query: str, 
                  result: Optional[Dict[str, Any]], 
                  error: Optional[Exception] = None):
        self.metrics['queries_processed'] += 1
        
        if error:
            self.metrics['errors'] += 1
            self.logger.error(f"Query error: {query}")
            self.logger.error(traceback.format_exc())
        else:
            self.logger.info(f"Successful query: {query}")
            
    def log_cache_event(self, hit: bool):
        if hit:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1
            
    def update_latency(self, latency: float):
        current_avg = self.metrics['avg_latency']
        n = self.metrics['queries_processed']
        self.metrics['avg_latency'] = (
            (current_avg * (n - 1) + latency) / n
        )
        
    def get_system_health(self) -> Dict[str, Any]:
        return {
            'total_queries': self.metrics['queries_processed'],
            'error_rate': self.metrics['errors'] / max(1, self.metrics['queries_processed']),
            'avg_latency': self.metrics['avg_latency'],
            'cache_hit_rate': self.metrics['cache_hits'] / 
                max(1, (self.metrics['cache_hits'] + self.metrics['cache_misses']))
        }

# Example usage
monitor = RAGMonitor()
try:
    start_time = time.time()
    result = rag_system.process_query("Sample query")
    latency = time.time() - start_time
    
    monitor.log_query("Sample query", result)
    monitor.update_latency(latency)
except Exception as e:
    monitor.log_query("Sample query", None, error=e)

health_metrics = monitor.get_system_health()
print(f"System Health Metrics: {health_metrics}")
```

Slide 13: Advanced Document Processing and Chunking

Implementation of sophisticated document processing strategies including semantic chunking, overlap management, and metadata preservation. This approach ensures optimal context preservation for retrieval tasks.

```python
import spacy
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    semantic_score: float
    start_idx: int
    end_idx: int

class SemanticDocumentProcessor:
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 512):
        self.nlp = spacy.load("en_core_web_sm")
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
    def get_semantic_boundaries(self, text: str) -> List[int]:
        doc = self.nlp(text)
        boundaries = []
        
        for sent in doc.sents:
            # Score sentence boundaries based on semantic completeness
            score = self._calculate_semantic_score(sent)
            if score > 0.7:  # Threshold for semantic completeness
                boundaries.append(sent.end_char)
                
        return boundaries
    
    def _calculate_semantic_score(self, sent) -> float:
        # Heuristic scoring based on linguistic features
        has_subject = any(token.dep_ == "nsubj" for token in sent)
        has_verb = any(token.pos_ == "VERB" for token in sent)
        proper_punctuation = sent.text.strip()[-1] in {'.', '!', '?'}
        
        score = (
            0.4 * has_subject +
            0.4 * has_verb +
            0.2 * proper_punctuation
        )
        return score
    
    def create_chunks(self, 
                     text: str, 
                     metadata: Dict[str, Any]) -> List[DocumentChunk]:
        boundaries = self.get_semantic_boundaries(text)
        chunks = []
        current_start = 0
        
        for boundary in boundaries:
            if boundary - current_start >= self.min_chunk_size:
                chunk_text = text[current_start:boundary].strip()
                
                if len(chunk_text) <= self.max_chunk_size:
                    semantic_score = self._calculate_chunk_score(chunk_text)
                    
                    chunk = DocumentChunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            'position': len(chunks),
                            'boundary_type': 'semantic'
                        },
                        semantic_score=semantic_score,
                        start_idx=current_start,
                        end_idx=boundary
                    )
                    chunks.append(chunk)
                    current_start = boundary
                    
        # Handle remaining text
        if current_start < len(text):
            remaining_text = text[current_start:].strip()
            if remaining_text:
                semantic_score = self._calculate_chunk_score(remaining_text)
                chunks.append(DocumentChunk(
                    text=remaining_text,
                    metadata={
                        **metadata,
                        'position': len(chunks),
                        'boundary_type': 'final'
                    },
                    semantic_score=semantic_score,
                    start_idx=current_start,
                    end_idx=len(text)
                ))
                
        return chunks
    
    def _calculate_chunk_score(self, text: str) -> float:
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0.0
            
        # Calculate average sentence score
        scores = [self._calculate_semantic_score(sent) for sent in sentences]
        return sum(scores) / len(scores)

# Example usage
processor = SemanticDocumentProcessor()
document_text = """
Long document text with multiple paragraphs and sentences...
"""
metadata = {
    'source': 'technical_paper.pdf',
    'date': '2024-01-01'
}

chunks = processor.create_chunks(document_text, metadata)
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}:")
    print(f"Score: {chunk.semantic_score:.2f}")
    print(f"Text: {chunk.text[:100]}...")
    print(f"Metadata: {chunk.metadata}")
```

Slide 14: Real-world Implementation - Technical Documentation Search

This implementation showcases a complete RAG system specifically designed for searching and answering questions from technical documentation, including handling of code snippets and technical terminology.

````python
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class TechnicalDocument:
    content: str
    code_blocks: List[str]
    terminology: Dict[str, str]
    metadata: Dict[str, Any]

class TechnicalRAGSystem:
    def __init__(self):
        self.code_pattern = re.compile(r'```[\s\S]*?```')
        self.term_pattern = re.compile(r'`([^`]+)`')
        self.preprocessor = SemanticDocumentProcessor()
        
    def parse_technical_document(self, content: str) -> TechnicalDocument:
        # Extract code blocks
        code_blocks = self.code_pattern.findall(content)
        clean_content = self.code_pattern.sub('[CODE_BLOCK]', content)
        
        # Extract technical terms
        terms = self.term_pattern.findall(clean_content)
        terminology = {}
        
        # Process technical terms
        for term in terms:
            # Generate term definition using embedding similarity
            definition = self._get_term_definition(term)
            terminology[term] = definition
            
        return TechnicalDocument(
            content=clean_content,
            code_blocks=[block.strip('`') for block in code_blocks],
            terminology=terminology,
            metadata={'type': 'technical', 'term_count': len(terminology)}
        )
    
    def _get_term_definition(self, term: str) -> str:
        # Simplified term definition lookup
        return f"Technical definition for {term}"
    
    def process_technical_query(self, 
                              query: str,
                              doc: TechnicalDocument) -> Dict[str, Any]:
        # Check if query is about code
        is_code_query = any(keyword in query.lower() 
                           for keyword in ['code', 'implementation', 'example'])
        
        # Check if query is about terminology
        is_term_query = any(term.lower() in query.lower() 
                           for term in doc.terminology.keys())
        
        if is_code_query:
            return self._handle_code_query(query, doc)
        elif is_term_query:
            return self._handle_term_query(query, doc)
        else:
            return self._handle_general_query(query, doc)
    
    def _handle_code_query(self, 
                          query: str,
                          doc: TechnicalDocument) -> Dict[str, Any]:
        relevant_blocks = []
        for block in doc.code_blocks:
            # Calculate relevance score for code block
            score = self._calculate_code_relevance(query, block)
            if score > 0.5:
                relevant_blocks.append({
                    'code': block,
                    'relevance': score
                })
                
        return {
            'type': 'code_response',
            'blocks': sorted(relevant_blocks, 
                           key=lambda x: x['relevance'], 
                           reverse=True)
        }
    
    def _handle_term_query(self, 
                          query: str,
                          doc: TechnicalDocument) -> Dict[str, Any]:
        relevant_terms = {}
        for term, definition in doc.terminology.items():
            if term.lower() in query.lower():
                relevant_terms[term] = definition
                
        return {
            'type': 'terminology_response',
            'terms': relevant_terms
        }
    
    def _handle_general_query(self, 
                            query: str,
                            doc: TechnicalDocument) -> Dict[str, Any]:
        chunks = self.preprocessor.create_chunks(
            doc.content,
            metadata=doc.metadata
        )
        
        relevant_chunks = []
        for chunk in chunks:
            # Calculate relevance score for content chunk
            score = self._calculate_content_relevance(query, chunk.text)
            if score > 0.6:
                relevant_chunks.append({
                    'content': chunk.text,
                    'relevance': score
                })
                
        return {
            'type': 'general_response',
            'chunks': sorted(relevant_chunks,
                           key=lambda x: x['relevance'],
                           reverse=True)
        }
    
    def _calculate_code_relevance(self, query: str, code: str) -> float:
        # Implement code similarity scoring
        return 0.8  # Simplified score
        
    def _calculate_content_relevance(self, query: str, content: str) -> float:
        # Implement content similarity scoring
        return 0.7  # Simplified score

# Example usage
tech_rag = TechnicalRAGSystem()
content = """
# Technical Documentation
Here's an example implementation:
```python
def example():
    return "Hello World"
````

The `example` function demonstrates basic syntax. """

doc = tech\_rag.parse\_technical\_document(content) result = tech\_rag.process\_technical\_query( "Show me the example code", doc ) print(f"Query result: {result}")

```

Slide 15: Additional Resources

* ArXiv Papers on RAG Systems:
  - "Retrieval-Augmented Generation for Large Language Models: A Survey" - https://arxiv.org/abs/2312.10997
  - "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" - https://arxiv.org/abs/2310.11511
  - "Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models" - https://arxiv.org/abs/2311.09210

* Recommended Search Terms:
  - "RAG optimization techniques"
  - "Vector store implementations"
  - "Semantic chunking strategies"
  - "Document retrieval systems"

* Additional Learning Resources:
  - LangChain Documentation
  - ChromaDB GitHub Repository
  - Semantic Search Implementation Guides
```

