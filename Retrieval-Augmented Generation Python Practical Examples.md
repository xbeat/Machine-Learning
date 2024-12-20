## Retrieval-Augmented Generation Python Practical Examples
Slide 1: Introduction to RAG Architecture

Retrieval-Augmented Generation (RAG) combines traditional language models with an external knowledge base to generate more accurate and factual responses. The architecture consists of two main components: a retriever that fetches relevant documents and a generator that produces the final output.

```python
import numpy as np
from typing import List, Dict

class RAGArchitecture:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def generate_response(self, query: str) -> str:
        # Retrieve relevant documents
        documents = self.retriever.get_relevant_docs(query)
        
        # Augment query with retrieved information
        augmented_query = self._augment_query(query, documents)
        
        # Generate response using the augmented context
        response = self.generator.generate(augmented_query)
        
        return response

    def _augment_query(self, query: str, docs: List[Dict]) -> str:
        context = "\n".join([doc["content"] for doc in docs])
        return f"Query: {query}\nContext: {context}"
```

Slide 2: Vector Store Implementation

A vector store is essential for efficient similarity search in RAG systems. This implementation uses FAISS for storing and retrieving document embeddings based on cosine similarity.

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
    
    def add_documents(self, texts: List[str]):
        # Encode documents to vectors
        embeddings = self.encoder.encode(texts)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Add to FAISS index
        self.index.add(normalized_embeddings.astype('float32'))
        self.documents.extend(texts)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        query_embedding = self.encoder.encode([query])
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # Perform similarity search
        scores, indices = self.index.search(normalized_query.astype('float32'), k)
        return [self.documents[i] for i in indices[0]]
```

Slide 3: Document Chunking Strategy

Effective document chunking is crucial for RAG systems to maintain context and improve retrieval accuracy. This implementation provides various chunking strategies including overlap and semantic boundaries.

```python
from typing import List, Optional
import nltk
nltk.download('punkt')

class DocumentChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, text: str, strategy: str = 'fixed') -> List[str]:
        if strategy == 'fixed':
            return self._fixed_chunks(text)
        elif strategy == 'semantic':
            return self._semantic_chunks(text)
        
    def _fixed_chunks(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap
        return chunks
    
    def _semantic_chunks(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
```

Slide 4: Query Expansion and Reformulation

Query expansion enhances retrieval effectiveness by reformulating the original query to include related terms and context. This implementation uses multiple strategies including synonym expansion and contextual reformulation.

```python
from typing import List
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

class QueryExpander:
    def __init__(self):
        self.history = []
        
    def expand_query(self, query: str, method: str = 'synonym') -> str:
        if method == 'synonym':
            expanded = self._synonym_expansion(query)
        elif method == 'contextual':
            expanded = self._contextual_expansion(query)
        
        self.history.append({'original': query, 'expanded': expanded})
        return expanded
    
    def _synonym_expansion(self, query: str) -> str:
        words = query.split()
        expanded_words = []
        
        for word in words:
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
            
            expanded_words.append(word)
            if synonyms:
                expanded_words.extend(synonyms[:2])  # Add top 2 synonyms
                
        return ' '.join(expanded_words)
    
    def _contextual_expansion(self, query: str) -> str:
        # Add query-specific context based on previous interactions
        context = self._get_context_from_history()
        return f"{query} {context}"
    
    def _get_context_from_history(self) -> str:
        if not self.history:
            return ""
        return " ".join([h['original'] for h in self.history[-2:]])
```

Slide 5: Reranking Implementation

Document reranking improves retrieval quality by applying a more sophisticated scoring mechanism to the initial retrieved documents. This implementation uses cross-attention and semantic similarity scoring.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DocumentReranker:
    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[dict]:
        pairs = [[query, doc] for doc in documents]
        features = self.tokenizer.batch_encode_plus(
            pairs,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze()
            
        ranked_results = []
        for doc, score in zip(documents, scores):
            ranked_results.append({
                'document': doc,
                'score': float(score)
            })
            
        # Sort by score in descending order
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        return ranked_results[:top_k]
```

Slide 6: Knowledge Base Management

Efficient knowledge base management is crucial for RAG systems. This implementation provides functionality for document indexing, updating, and maintenance of the knowledge base.

```python
import hashlib
from datetime import datetime
from typing import Dict, List, Optional

class KnowledgeBase:
    def __init__(self):
        self.documents = {}
        self.metadata = {}
        self.vector_store = None
        
    def add_document(self, content: str, metadata: Optional[Dict] = None) -> str:
        # Generate unique document ID
        doc_id = self._generate_doc_id(content)
        
        # Store document and metadata
        self.documents[doc_id] = content
        self.metadata[doc_id] = {
            'timestamp': datetime.now().isoformat(),
            'length': len(content),
            'hash': hashlib.sha256(content.encode()).hexdigest(),
            **metadata if metadata else {}
        }
        
        # Update vector store if initialized
        if self.vector_store:
            self.vector_store.add_documents([content])
            
        return doc_id
    
    def _generate_doc_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_document(self, doc_id: str) -> Dict:
        return {
            'content': self.documents.get(doc_id),
            'metadata': self.metadata.get(doc_id)
        }
    
    def update_document(self, doc_id: str, new_content: str) -> bool:
        if doc_id in self.documents:
            old_content = self.documents[doc_id]
            self.documents[doc_id] = new_content
            self.metadata[doc_id]['updated_at'] = datetime.now().isoformat()
            self.metadata[doc_id]['previous_hash'] = hashlib.sha256(old_content.encode()).hexdigest()
            return True
        return False
```

Slide 7: Context Window Management

Efficient management of context windows is critical for RAG systems to handle large documents while maintaining relevance. This implementation provides dynamic window sizing and context selection based on query relevance.

```python
class ContextWindowManager:
    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.tokenizer = None  # Initialize with your preferred tokenizer
        
    def create_windows(self, text: str, query: str) -> List[str]:
        windows = []
        tokens = self._tokenize(text)
        
        # Calculate optimal window size based on query
        window_size = self._calculate_window_size(query)
        
        # Create overlapping windows
        for i in range(0, len(tokens), window_size // 2):
            window = tokens[i:i + window_size]
            if window:
                windows.append(self._detokenize(window))
                
        return self._rank_windows(windows, query)
    
    def _calculate_window_size(self, query: str) -> int:
        query_tokens = len(self._tokenize(query))
        # Dynamic window sizing based on query length
        return min(self.max_tokens, max(512, query_tokens * 4))
    
    def _rank_windows(self, windows: List[str], query: str) -> List[str]:
        scores = []
        for window in windows:
            score = self._calculate_relevance(window, query)
            scores.append((score, window))
        
        return [w for _, w in sorted(scores, reverse=True)]
```

Slide 8: Response Generation Pipeline

The response generation pipeline combines retrieved context with the base language model to produce accurate and contextually relevant responses. This implementation includes multiple generation strategies and output verification.

```python
class RAGResponseGenerator:
    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def generate_response(self, query: str, contexts: List[str]) -> Dict:
        # Prepare prompt with retrieved contexts
        prompt = self._construct_prompt(query, contexts)
        
        # Generate multiple candidates
        candidates = self._generate_candidates(prompt)
        
        # Verify and select best response
        best_response = self._verify_and_select(candidates, contexts)
        
        return {
            'response': best_response,
            'contexts_used': contexts,
            'confidence_score': self._calculate_confidence(best_response, contexts)
        }
    
    def _construct_prompt(self, query: str, contexts: List[str]) -> str:
        formatted_contexts = "\n".join([f"Context {i+1}: {ctx}" 
                                      for i, ctx in enumerate(contexts)])
        return f"""
        Query: {query}
        Retrieved Information:
        {formatted_contexts}
        Response:"""
    
    def _generate_candidates(self, prompt: str, num_candidates: int = 3) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            num_return_sequences=num_candidates,
            num_beams=num_candidates * 2,
            temperature=0.7,
            do_sample=True
        )
        
        return [self.tokenizer.decode(output, skip_special_tokens=True) 
                for output in outputs]
```

Slide 9: Source Code for Response Generation Pipeline

```python
    def _verify_and_select(self, candidates: List[str], 
                          contexts: List[str]) -> str:
        verified_responses = []
        
        for response in candidates:
            # Check factual consistency
            if self._check_factual_consistency(response, contexts):
                # Calculate response quality score
                score = self._calculate_quality_score(response)
                verified_responses.append((score, response))
        
        # Return highest scored response
        return max(verified_responses, key=lambda x: x[0])[1]
    
    def _check_factual_consistency(self, response: str, 
                                 contexts: List[str]) -> bool:
        # Extract key facts from response
        response_facts = self._extract_facts(response)
        
        # Check if facts are supported by contexts
        supported_facts = 0
        for fact in response_facts:
            if any(self._fact_is_supported(fact, ctx) for ctx in contexts):
                supported_facts += 1
                
        # Require at least 80% fact support
        return supported_facts / len(response_facts) >= 0.8 if response_facts else True
    
    def _calculate_confidence(self, response: str, contexts: List[str]) -> float:
        # Implement confidence scoring based on:
        # 1. Semantic similarity with contexts
        # 2. Factual consistency score
        # 3. Response coherence
        semantic_score = self._calculate_semantic_similarity(response, contexts)
        factual_score = self._check_factual_consistency(response, contexts)
        coherence_score = self._calculate_coherence(response)
        
        return (semantic_score + float(factual_score) + coherence_score) / 3
```

Slide 10: Cache Management System

Implementing an efficient caching system for RAG reduces latency and computational costs by storing frequently accessed documents and query results. This implementation includes TTL-based expiration and LRU eviction strategies.

```python
from collections import OrderedDict
from time import time
from typing import Any, Optional

class RAGCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
            
        if self._is_expired(key):
            self._remove(key)
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            self._remove(key)
            
        # Evict least recently used if full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
            
        self.cache[key] = value
        self.timestamps[key] = time()
        self.cache.move_to_end(key)
        
    def _is_expired(self, key: str) -> bool:
        return time() - self.timestamps[key] > self.ttl
        
    def _remove(self, key: str) -> None:
        del self.cache[key]
        del self.timestamps[key]
        
    def _evict_lru(self) -> None:
        if self.cache:
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
```

Slide 11: Real-world Implementation: Scientific Paper Q&A System

This implementation demonstrates a complete RAG system for answering questions about scientific papers, including PDF processing, citation handling, and specialized scientific context understanding.

```python
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple

class ScientificRAG:
    def __init__(self, vector_store, model, tokenizer):
        self.vector_store = vector_store
        self.model = model
        self.tokenizer = tokenizer
        self.citation_pattern = r'\[([\d,\s]+)\]'
        
    def process_paper(self, pdf_path: str) -> Dict[str, Any]:
        # Extract text and maintain structure
        doc = fitz.open(pdf_path)
        sections = self._extract_sections(doc)
        
        # Process citations and references
        citations = self._extract_citations(sections)
        
        # Create embeddings for sections
        section_embeddings = self._embed_sections(sections)
        
        return {
            'sections': sections,
            'citations': citations,
            'embeddings': section_embeddings
        }
        
    def answer_question(self, question: str, paper_data: Dict) -> Dict:
        # Retrieve relevant sections
        relevant_sections = self._retrieve_relevant_sections(
            question, 
            paper_data['sections'],
            paper_data['embeddings']
        )
        
        # Generate answer with citations
        answer = self._generate_scientific_answer(
            question,
            relevant_sections,
            paper_data['citations']
        )
        
        return {
            'answer': answer,
            'supporting_sections': relevant_sections,
            'confidence': self._calculate_scientific_confidence(answer)
        }
```

Slide 12: Real-world Implementation: Source Code for Scientific Paper Q&A System

```python
def _extract_sections(self, doc) -> List[Dict]:
    sections = []
    current_section = {'title': '', 'content': '', 'page': 0}
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        lines = text.split('\n')
        
        for line in lines:
            # Detect section headers using formatting and position
            if self._is_section_header(line, page):
                if current_section['content']:
                    sections.append(current_section)
                current_section = {
                    'title': line.strip(),
                    'content': '',
                    'page': page_num + 1
                }
            else:
                current_section['content'] += line + '\n'
    
    if current_section['content']:
        sections.append(current_section)
    return sections

def _generate_scientific_answer(self, question: str, 
                              relevant_sections: List[Dict],
                              citations: Dict) -> str:
    # Prepare context with scientific formatting
    context = self._format_scientific_context(relevant_sections)
    
    # Generate initial answer
    answer = self._generate_base_response(question, context)
    
    # Add citations
    answer = self._add_citations(answer, citations)
    
    # Validate scientific accuracy
    if not self._validate_scientific_claims(answer, relevant_sections):
        answer = self._regenerate_with_constraints(question, context)
    
    return answer

def _validate_scientific_claims(self, answer: str, 
                              sections: List[Dict]) -> bool:
    claims = self._extract_scientific_claims(answer)
    
    for claim in claims:
        support_found = False
        for section in sections:
            if self._find_claim_support(claim, section['content']):
                support_found = True
                break
        if not support_found:
            return False
    return True

def _calculate_scientific_confidence(self, answer: str) -> float:
    # Calculate confidence based on:
    # 1. Citation density
    citation_score = self._calculate_citation_density(answer)
    
    # 2. Technical term accuracy
    term_score = self._validate_technical_terms(answer)
    
    # 3. Logical consistency
    logic_score = self._check_logical_consistency(answer)
    
    weights = [0.4, 0.3, 0.3]
    return (citation_score * weights[0] + 
            term_score * weights[1] + 
            logic_score * weights[2])
```

Slide 13: Performance Metrics and Evaluation

A comprehensive evaluation framework for RAG systems must assess both retrieval accuracy and generation quality. This implementation provides metrics and testing protocols for system evaluation.

```python
from sklearn.metrics import precision_recall_curve
from rouge_score import rouge_scorer
import numpy as np

class RAGEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def evaluate_system(self, test_queries: List[Dict],
                       system_outputs: List[Dict]) -> Dict:
        metrics = {
            'retrieval': self._evaluate_retrieval(test_queries, system_outputs),
            'generation': self._evaluate_generation(test_queries, system_outputs),
            'end2end': self._evaluate_end2end(test_queries, system_outputs)
        }
        
        return self._aggregate_metrics(metrics)
        
    def _evaluate_retrieval(self, queries: List[Dict],
                           outputs: List[Dict]) -> Dict:
        precisions = []
        recalls = []
        
        for query, output in zip(queries, outputs):
            retrieved_docs = set(output['retrieved_documents'])
            relevant_docs = set(query['relevant_documents'])
            
            precision = len(retrieved_docs & relevant_docs) / len(retrieved_docs)
            recall = len(retrieved_docs & relevant_docs) / len(relevant_docs)
            
            precisions.append(precision)
            recalls.append(recall)
            
        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': 2 * (np.mean(precisions) * np.mean(recalls)) / 
                  (np.mean(precisions) + np.mean(recalls))
        }
```

Slide 14: Additional Resources

*   Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks:
    *   [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
*   Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection:
    *   [https://arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511)
*   Atlas: Few-shot Learning with Retrieval Augmented Language Models:
    *   [https://arxiv.org/abs/2208.03299](https://arxiv.org/abs/2208.03299)
*   REPLUG: Retrieval-Augmented Black-Box Language Models:
    *   [https://arxiv.org/abs/2301.12652](https://arxiv.org/abs/2301.12652)
*   A Survey on Retrieval-Augmented Text Generation:
    *   [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)

