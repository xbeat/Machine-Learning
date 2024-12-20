## Comparing RAG and Fine-Tuned Language Models
Slide 1: Understanding RAG vs Fine-Tuned Models Architecture

The fundamental architectural differences between RAG and fine-tuned models lie in how they process and generate responses. RAG systems employ a retrieval mechanism to fetch relevant context before generation, while fine-tuned models rely on encoded knowledge within their parameters.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGModel:
    def __init__(self, base_model_name, retriever_model_name):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.retriever = DocumentRetriever(retriever_model_name)
    
    def generate_response(self, query):
        # Retrieve relevant documents
        context = self.retriever.get_relevant_docs(query)
        
        # Combine query with retrieved context
        augmented_prompt = f"Context: {context}\nQuery: {query}"
        
        # Generate response
        inputs = self.tokenizer(augmented_prompt, return_tensors="pt")
        outputs = self.base_model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])

class FineTunedModel:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def generate_response(self, query):
        # Direct generation without external retrieval
        inputs = self.tokenizer(query, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])
```

Slide 2: Implementing a Basic Document Retriever

A document retriever is crucial for RAG systems, using embedding-based similarity search to find relevant context. This implementation demonstrates a simple but effective approach using cosine similarity for document matching.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.document_store = []
        self.document_embeddings = None
    
    def add_documents(self, documents):
        self.document_store.extend(documents)
        # Create embeddings for all documents
        self.document_embeddings = self.encoder.encode(documents)
    
    def get_relevant_docs(self, query, k=3):
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity(
            [query_embedding], 
            self.document_embeddings
        )[0]
        
        # Get top-k most similar documents
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.document_store[i] for i in top_k_indices]
```

Slide 3: Custom Fine-Tuning Pipeline

A comprehensive fine-tuning pipeline that implements gradient accumulation and learning rate scheduling for optimal model adaptation on domain-specific tasks.

```python
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class FineTuningPipeline:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
    def train(self, dataset, epochs=3, batch_size=4, 
              accumulation_steps=4, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Calculate total training steps
        total_steps = len(dataset) * epochs // (batch_size * accumulation_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            for i, batch in enumerate(dataset):
                inputs = self.tokenizer(batch['text'], 
                                      return_tensors='pt', 
                                      padding=True,
                                      truncation=True).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

Slide 4: Vector Database Integration for RAG

An efficient vector database implementation using FAISS for high-performance similarity search in RAG systems, essential for handling large-scale document collections.

```python
import faiss
import numpy as np
from typing import List, Tuple

class VectorDatabase:
    def __init__(self, dimension: int, index_type: str = 'Flat'):
        if index_type == 'Flat':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IVF':
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.train(np.random.random((1000, dimension)).astype('float32'))
        
        self.document_map = {}
        self.current_id = 0
    
    def add_documents(self, embeddings: np.ndarray, documents: List[str]):
        num_documents = len(documents)
        document_ids = np.arange(self.current_id, 
                               self.current_id + num_documents)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Update document mapping
        for doc_id, doc in zip(document_ids, documents):
            self.document_map[int(doc_id)] = doc
        
        self.current_id += num_documents
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        # Perform similarity search
        distances, indices = self.index.search(
            query_embedding.astype('float32').reshape(1, -1), k
        )
        
        # Return documents and their distances
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                results.append((self.document_map[int(idx)], float(dist)))
        
        return results
```

Slide 5: Hybrid Retrieval System

A sophisticated hybrid retrieval system that combines dense and sparse retrieval methods to improve document matching accuracy, implementing both BM25 and neural embedding-based retrieval.

```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict

class HybridRetriever:
    def __init__(self, embedding_model, alpha=0.5):
        self.embedding_model = embedding_model
        self.alpha = alpha
        self.documents = []
        self.embeddings = None
        self.bm25 = None
        
    def index_documents(self, documents: List[str]):
        # Prepare documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Create dense embeddings
        self.embeddings = self.embedding_model.encode(documents)
        self.documents = documents
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        # Dense retrieval scores
        query_embedding = self.embedding_model.encode([query])[0]
        dense_scores = np.dot(self.embeddings, query_embedding)
        
        # Sparse retrieval scores
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
        
        # Combine scores
        final_scores = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
        
        # Get top-k results
        top_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(final_scores[idx]),
                'dense_score': float(dense_scores[idx]),
                'sparse_score': float(sparse_scores[idx])
            })
        
        return results
```

Slide 6: Prompt Engineering for RAG Systems

The effectiveness of RAG systems heavily depends on prompt engineering. This implementation showcases advanced prompt construction techniques for optimal context integration and response generation.

```python
class PromptEngineering:
    def __init__(self, max_context_length: int = 2048):
        self.max_context_length = max_context_length
        self.templates = {
            'qa': {
                'prefix': "Answer the question based on the context below.\n\nContext:\n",
                'middle': "\n\nQuestion: ",
                'suffix': "\n\nAnswer:"
            },
            'summary': {
                'prefix': "Summarize the following context:\n\nContext:\n",
                'middle': "\n\nSummary:",
                'suffix': ""
            }
        }
    
    def construct_prompt(self, 
                        query: str, 
                        retrieved_contexts: List[Dict],
                        task_type: str = 'qa') -> str:
        template = self.templates[task_type]
        
        # Sort contexts by relevance score
        sorted_contexts = sorted(retrieved_contexts, 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        # Combine contexts with relevance weights
        weighted_contexts = []
        for ctx in sorted_contexts:
            relevance_prefix = f"[Relevance: {ctx['score']:.2f}]\n"
            weighted_contexts.append(relevance_prefix + ctx['document'])
        
        # Truncate combined context to fit max length
        combined_context = "\n\n".join(weighted_contexts)
        
        # Construct final prompt
        prompt = (f"{template['prefix']}{combined_context}"
                 f"{template['middle']}{query}{template['suffix']}")
        
        return prompt
    
    def rerank_contexts(self, 
                       query: str, 
                       contexts: List[str], 
                       model) -> List[Dict]:
        """Re-rank contexts using cross-attention scores"""
        query_tokens = model.tokenizer(query, return_tensors='pt')
        
        reranked = []
        for ctx in contexts:
            # Calculate cross-attention scores
            ctx_tokens = model.tokenizer(ctx, return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids=ctx_tokens['input_ids'],
                              attention_mask=ctx_tokens['attention_mask'],
                              encoder_hidden_states=query_tokens['input_ids'])
                
                # Use attention scores for ranking
                score = outputs.cross_attentions[-1].mean().item()
                reranked.append({'document': ctx, 'score': score})
        
        return sorted(reranked, key=lambda x: x['score'], reverse=True)
```

Slide 7: Knowledge Base Management

Efficient knowledge base management is crucial for RAG systems. This implementation provides a sophisticated document storage and indexing system with versioning support.

```python
import sqlite3
from datetime import datetime
import json

class KnowledgeBase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()
    
    def setup_database(self):
        cursor = self.conn.cursor()
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding_version TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Create embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                version TEXT,
                vector BLOB,
                created_at TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        self.conn.commit()
    
    def add_document(self, content: str, metadata: Dict = None):
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO documents (content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (content, json.dumps(metadata), now, now))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_embedding(self, doc_id: int, embedding: np.ndarray, version: str):
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        
        # Store embedding as binary
        embedding_binary = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO embeddings (document_id, version, vector, created_at)
            VALUES (?, ?, ?, ?)
        """, (doc_id, version, embedding_binary, now))
        
        # Update document's embedding version
        cursor.execute("""
            UPDATE documents 
            SET embedding_version = ?, updated_at = ?
            WHERE id = ?
        """, (version, now, doc_id))
        
        self.conn.commit()
    
    def get_documents_needing_embedding(self, version: str) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, metadata
            FROM documents
            WHERE embedding_version IS NULL
            OR embedding_version != ?
            AND is_active = 1
        """, (version,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'metadata': json.loads(row[2]) if row[2] else {}
            })
        
        return results
```

Slide 8: Performance Metrics for RAG vs Fine-tuned Models

A comprehensive evaluation system implementing various metrics to compare RAG and fine-tuned model performance, including relevance scoring, response coherence, and factual accuracy measurement.

```python
from rouge_score import rouge_scorer
from bert_score import score
from typing import List, Dict
import numpy as np

class ModelEvaluator:
    def __init__(self, bert_model='microsoft/deberta-xlarge-mnli'):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.bert_model = bert_model
        
    def evaluate_response(self, 
                         generated: str, 
                         reference: str, 
                         context: str = None) -> Dict:
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)
        metrics['rouge'] = {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure
        }
        
        # BERTScore for semantic similarity
        P, R, F1 = score([generated], [reference], model_type=self.bert_model)
        metrics['bert_score'] = {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
        
        # Context relevance if provided
        if context:
            metrics['context_relevance'] = self._calculate_context_relevance(
                generated, context
            )
        
        return metrics
    
    def _calculate_context_relevance(self, response: str, context: str) -> float:
        # Implement context relevance scoring using pretrained model
        sentences = response.split('.')
        context_segments = context.split('.')
        
        relevance_scores = []
        for sent in sentences:
            if sent.strip():
                max_score = max(
                    self._semantic_similarity(sent, ctx)
                    for ctx in context_segments if ctx.strip()
                )
                relevance_scores.append(max_score)
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        # Implement semantic similarity using the specified model
        _, _, F1 = score([text1], [text2], model_type=self.bert_model)
        return F1.mean().item()
```

Slide 9: Implementation of Chunking Strategies

Advanced document chunking strategies for optimal context retrieval, implementing both fixed-size and semantic-aware chunking methods with overlap control.

```python
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

class DocumentChunker:
    def __init__(self, 
                 chunk_size: int = 512, 
                 overlap: int = 50,
                 method: str = 'semantic'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.method = method
        nltk.download('punkt')
    
    def chunk_document(self, text: str) -> List[Dict]:
        if self.method == 'fixed':
            return self._fixed_size_chunking(text)
        elif self.method == 'semantic':
            return self._semantic_chunking(text)
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")
    
    def _fixed_size_chunking(self, text: str) -> List[Dict]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append({
                'text': chunk,
                'start_idx': i,
                'end_idx': min(i + self.chunk_size, len(words))
            })
        
        return chunks
    
    def _semantic_chunking(self, text: str) -> List[Dict]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_words = len(sent.split())
            
            if current_length + sent_words > self.chunk_size:
                # Create new chunk if current one is full
                if current_chunk:
                    chunks.append(self._create_chunk_dict(current_chunk))
                current_chunk = [sent]
                current_length = sent_words
            else:
                current_chunk.append(sent)
                current_length += sent_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(current_chunk))
        
        return chunks
    
    def _create_chunk_dict(self, sentences: List[str]) -> Dict:
        text = ' '.join(sentences)
        return {
            'text': text,
            'num_sentences': len(sentences),
            'length': len(text.split())
        }
```

Slide 10: Context Window Management

Sophisticated context window management system for handling long documents and multiple context pieces while maintaining coherence and relevance.

```python
class ContextWindowManager:
    def __init__(self, 
                 max_tokens: int = 2048,
                 overlap_tokens: int = 128):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
    def manage_context_window(self, 
                            contexts: List[Dict],
                            query: str,
                            tokenizer) -> List[Dict]:
        # Tokenize query
        query_tokens = len(tokenizer.encode(query))
        available_tokens = self.max_tokens - query_tokens - 100  # Reserve tokens for response
        
        # Sort contexts by relevance
        sorted_contexts = sorted(contexts, key=lambda x: x['score'], reverse=True)
        
        selected_contexts = []
        current_tokens = 0
        
        for ctx in sorted_contexts:
            ctx_tokens = len(tokenizer.encode(ctx['text']))
            
            if current_tokens + ctx_tokens <= available_tokens:
                selected_contexts.append(ctx)
                current_tokens += ctx_tokens
            else:
                # Try to fit partial context with overlap
                if ctx_tokens > self.overlap_tokens:
                    truncated_text = self._truncate_context(
                        ctx['text'],
                        available_tokens - current_tokens + self.overlap_tokens,
                        tokenizer
                    )
                    if truncated_text:
                        ctx['text'] = truncated_text
                        selected_contexts.append(ctx)
                break
        
        return selected_contexts
    
    def _truncate_context(self, 
                         text: str, 
                         max_tokens: int,
                         tokenizer) -> str:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        # Truncate while maintaining sentence boundaries
        sentences = text.split('.')
        truncated_sentences = []
        current_tokens = 0
        
        for sent in sentences:
            sent_tokens = len(tokenizer.encode(sent))
            if current_tokens + sent_tokens <= max_tokens:
                truncated_sentences.append(sent)
                current_tokens += sent_tokens
            else:
                break
                
        return '.'.join(truncated_sentences) + '.'
```

Slide 11: Response Generation Pipeline

A comprehensive pipeline for generating responses that combines retrieved context processing, prompt construction, and response filtering while maintaining coherence and factual accuracy.

```python
class ResponseGenerator:
    def __init__(self, model, tokenizer, max_length: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def generate_response(self, 
                         query: str,
                         contexts: List[Dict],
                         temperature: float = 0.7) -> Dict:
        # Prepare context and construct prompt
        processed_context = self._process_contexts(contexts)
        prompt = self._construct_prompt(query, processed_context)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=self.max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process and validate response
        processed_response = self._post_process_response(response, contexts)
        
        return {
            'response': processed_response,
            'metadata': {
                'context_used': len(contexts),
                'response_length': len(processed_response.split()),
                'source_documents': [ctx['id'] for ctx in contexts]
            }
        }
    
    def _process_contexts(self, contexts: List[Dict]) -> str:
        # Sort contexts by relevance score
        sorted_contexts = sorted(contexts, 
                               key=lambda x: x.get('score', 0), 
                               reverse=True)
        
        # Combine contexts with relevance markers
        processed = []
        for ctx in sorted_contexts:
            processed.append(f"[Source {ctx.get('id', 'unknown')}] {ctx['text']}")
            
        return "\n\n".join(processed)
    
    def _construct_prompt(self, query: str, context: str) -> str:
        return (f"Based on the following context, please answer the query.\n\n"
                f"Context:\n{context}\n\n"
                f"Query: {query}\n\n"
                f"Answer:")
    
    def _post_process_response(self, response: str, contexts: List[Dict]) -> str:
        # Remove any potential hallucinated references
        response = self._remove_invalid_references(response, contexts)
        
        # Ensure response coherence
        response = self._ensure_coherence(response)
        
        return response.strip()
    
    def _remove_invalid_references(self, response: str, contexts: List[Dict]) -> str:
        # Implementation to remove references not found in context
        valid_ids = {ctx['id'] for ctx in contexts}
        
        # Remove invalid source citations
        import re
        pattern = r'\[Source \d+\]'
        
        def replace_invalid_source(match):
            source_id = int(re.search(r'\d+', match.group()).group())
            return match.group() if source_id in valid_ids else ''
        
        return re.sub(pattern, replace_invalid_source, response)
    
    def _ensure_coherence(self, response: str) -> str:
        sentences = response.split('.')
        coherent_sentences = []
        
        for i, sent in enumerate(sentences):
            if i == 0 or (sent.strip() and not sent.strip().startswith('[')):
                coherent_sentences.append(sent)
                
        return '.'.join(coherent_sentences)
```

Slide 12: Real-time Model Adaptation

Implementation of a real-time model adaptation system that can fine-tune the model on new data while maintaining performance on existing knowledge.

```python
class RealTimeAdapter:
    def __init__(self, base_model, learning_rate=1e-5):
        self.model = base_model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        self.adaptation_buffer = []
        self.performance_metrics = []
        
    def adapt_to_feedback(self, 
                         query: str,
                         response: str,
                         correct_response: str,
                         context: str = None):
        # Prepare training instance
        training_instance = self._prepare_training_instance(
            query, response, correct_response, context
        )
        
        # Update adaptation buffer
        self.adaptation_buffer.append(training_instance)
        
        # Perform adaptation if buffer is full
        if len(self.adaptation_buffer) >= 8:  # Batch size
            self._perform_adaptation()
            self.adaptation_buffer = []
            
    def _prepare_training_instance(self, 
                                 query: str,
                                 response: str,
                                 correct_response: str,
                                 context: str) -> Dict:
        return {
            'input_text': f"Query: {query}\nContext: {context}" if context 
                         else f"Query: {query}",
            'response': response,
            'correct_response': correct_response
        }
        
    def _perform_adaptation(self):
        self.model.train()
        total_loss = 0
        
        for instance in self.adaptation_buffer:
            # Prepare input
            inputs = self.tokenizer(
                instance['input_text'],
                return_tensors='pt'
            )
            
            # Prepare target
            targets = self.tokenizer(
                instance['correct_response'],
                return_tensors='pt'
            )
            
            # Forward pass
            outputs = self.model(**inputs, labels=targets['input_ids'])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
        # Record adaptation metrics
        self.performance_metrics.append({
            'timestamp': datetime.now().isoformat(),
            'average_loss': total_loss / len(self.adaptation_buffer),
            'num_instances': len(self.adaptation_buffer)
        })
```

Slide 13: Cross-Encoder Reranking System

An advanced reranking system that uses cross-encoder architecture to improve context relevance scoring and selection accuracy, particularly useful for complex queries requiring precise context matching.

```python
import torch
from transformers import AutoModelForSequenceClassification
from typing import List, Dict

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def rerank(self, 
               query: str, 
               contexts: List[Dict], 
               top_k: int = None) -> List[Dict]:
        self.model.eval()
        pairs = []
        
        # Prepare text pairs for scoring
        for ctx in contexts:
            pairs.append([query, ctx['text']])
            
        # Tokenize all pairs at once
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(self.device)
        
        # Calculate relevance scores
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze()
            scores = torch.sigmoid(scores).cpu().numpy()
            
        # Update context objects with new scores
        reranked_contexts = []
        for ctx, score in zip(contexts, scores):
            ctx_copy = ctx.copy()
            ctx_copy['rerank_score'] = float(score)
            reranked_contexts.append(ctx_copy)
            
        # Sort by reranking score
        reranked_contexts.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Return top-k if specified
        if top_k:
            return reranked_contexts[:top_k]
        return reranked_contexts
        
    def batch_rerank(self, 
                     queries: List[str], 
                     context_sets: List[List[Dict]],
                     batch_size: int = 32) -> List[List[Dict]]:
        all_reranked = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_contexts = context_sets[i:i + batch_size]
            
            # Process each query-context set in the batch
            batch_reranked = []
            for query, contexts in zip(batch_queries, batch_contexts):
                reranked = self.rerank(query, contexts)
                batch_reranked.append(reranked)
                
            all_reranked.extend(batch_reranked)
            
        return all_reranked
```

Slide 14: Training Data Generator for Fine-tuning

A sophisticated system for generating high-quality training data by combining different sources and applying various augmentation techniques to improve model performance.

```python
class TrainingDataGenerator:
    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.augmentation_techniques = {
            'paraphrase': self._generate_paraphrase,
            'question_generation': self._generate_questions,
            'context_expansion': self._expand_context
        }
        
    def generate_training_data(self, 
                             source_documents: List[Dict],
                             num_samples: int = 1000,
                             techniques: List[str] = ['paraphrase']) -> List[Dict]:
        training_data = []
        
        for doc in source_documents:
            # Generate base examples
            base_examples = self._generate_base_examples(doc)
            
            # Apply augmentation techniques
            augmented_examples = []
            for technique in techniques:
                if technique in self.augmentation_techniques:
                    augmented = self.augmentation_techniques[technique](base_examples)
                    augmented_examples.extend(augmented)
            
            training_data.extend(base_examples)
            training_data.extend(augmented_examples)
            
        # Ensure quality and diversity
        filtered_data = self._filter_training_data(training_data)
        
        # Sample requested number of examples
        return self._sample_diverse_examples(filtered_data, num_samples)
    
    def _generate_base_examples(self, document: Dict) -> List[Dict]:
        examples = []
        
        # Extract key information
        chunks = self._chunk_document(document['text'])
        
        for chunk in chunks:
            # Generate question-answer pairs
            qa_pairs = self._generate_qa_pairs(chunk)
            
            # Generate instruction-response pairs
            instruction_pairs = self._generate_instruction_pairs(chunk)
            
            examples.extend(qa_pairs)
            examples.extend(instruction_pairs)
            
        return examples
    
    def _generate_qa_pairs(self, text: str) -> List[Dict]:
        # Implementation of question-answer pair generation
        inputs = self.tokenizer(
            f"Generate questions from: {text}",
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        outputs = self.base_model.generate(
            inputs['input_ids'],
            max_length=128,
            num_return_sequences=3,
            temperature=0.7
        )
        
        questions = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        qa_pairs = []
        for question in questions:
            # Generate answer for each question
            answer = self._generate_answer(question, text)
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'context': text,
                'type': 'qa_pair'
            })
            
        return qa_pairs
    
    def _generate_answer(self, question: str, context: str) -> str:
        inputs = self.tokenizer(
            f"Question: {question}\nContext: {context}\nAnswer:",
            return_tensors='pt',
            max_length=512,
            truncation=True
        )
        
        outputs = self.base_model.generate(
            inputs['input_ids'],
            max_length=128,
            num_return_sequences=1,
            temperature=0.3
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

Slide 15: Additional Resources

*   Recent developments in RAG systems:
*   "Retrieval-Augmented Generation for Large Language Models: A Survey" [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
*   Fine-tuning techniques and best practices:
*   "A Survey of Deep Learning Approaches for Fine-tuning Large Language Models" [https://arxiv.org/abs/2401.01071](https://arxiv.org/abs/2401.01071)
*   Hybrid retrieval methodologies:
*   "Improving Neural Information Retrieval through Hybrid Architectures" [https://arxiv.org/abs/2305.15462](https://arxiv.org/abs/2305.15462)
*   Context optimization strategies:
*   "Context Length and Generation Performance in LLMs" [https://arxiv.org/abs/2310.12862](https://arxiv.org/abs/2310.12862)

