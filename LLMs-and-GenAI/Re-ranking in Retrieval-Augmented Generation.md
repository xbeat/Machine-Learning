## Re-ranking in Retrieval-Augmented Generation
Slide 1: Initial Document Retrieval with BM25

BM25 serves as a foundational retrieval algorithm in RAG systems, implementing a probabilistic scoring function that ranks documents based on query term appearances. This implementation demonstrates a basic BM25 retriever using pure Python for educational purposes.

```python
from math import log
from collections import Counter
import re

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = Counter()
        self.idf = {}
        self.doc_vectors = []
        
    def fit(self, documents):
        # Preprocess and tokenize documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Calculate document frequencies
        for doc in tokenized_docs:
            terms = set(doc)
            for term in terms:
                self.doc_freqs[term] += 1
        
        # Calculate IDF scores
        N = len(documents)
        self.idf = {
            term: log((N - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freqs.items()
        }
        
    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

# Example usage:
docs = [
    "RAG systems enhance LLM responses with external knowledge",
    "Document retrieval is crucial for accurate information access",
    "BM25 algorithm ranks documents based on relevance scores"
]

retriever = BM25Retriever()
retriever.fit(docs)
```

Slide 2: BM25 Scoring Implementation

The scoring mechanism of BM25 combines term frequency, inverse document frequency, and document length normalization. This implementation shows how to calculate relevance scores for query-document pairs.

```python
class BM25Retriever:  # Continuing from previous slide
    def _score_doc(self, query_terms, doc_terms, doc_len):
        score = 0
        term_freqs = Counter(doc_terms)
        
        for term in query_terms:
            if term not in self.idf:
                continue
                
            tf = term_freqs[term]
            idf = self.idf[term]
            
            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)
            
        return score
    
    def retrieve(self, query, top_k=3):
        query_terms = self._tokenize(query)
        scores = []
        
        for idx, doc_terms in enumerate(self.doc_vectors):
            score = self._score_doc(query_terms, doc_terms, self.doc_lengths[idx])
            scores.append((idx, score))
            
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# Example usage:
query = "relevance ranking algorithms"
top_docs = retriever.retrieve(query)
print(f"Top documents for query '{query}':")
for idx, score in top_docs:
    print(f"Doc {idx}: Score = {score:.4f}")
```

Slide 3: Dense Retrieval with Sentence Transformers

Dense retrieval leverages neural networks to encode queries and documents into dense vector representations. This implementation uses the sentence-transformers library to create semantic embeddings.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class DenseRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.documents = None
        
    def index_documents(self, documents: List[str]):
        self.documents = documents
        self.doc_embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_tensor=True
        )
```

Slide 4: Dense Retrieval Search Implementation

This implementation demonstrates the similarity search mechanism in dense retrieval, using cosine similarity to rank documents based on their semantic proximity to the query embedding.

```python
class DenseRetriever:  # Continuing from previous slide
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        # Encode query to dense vector
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cos_scores = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(cos_scores)[::-1][:top_k]
        results = [(idx, cos_scores[idx]) for idx in top_indices]
        return results

# Example usage:
docs = [
    "Dense retrieval uses neural networks for document encoding",
    "Semantic search improves traditional keyword matching",
    "Vector representations capture contextual meanings"
]

retriever = DenseRetriever()
retriever.index_documents(docs)
results = retriever.search("semantic document representations", top_k=2)
print(f"Search results: {results}")
```

Slide 5: Cross-Encoder Re-Ranking Model

Cross-encoders provide more accurate relevance assessment by processing query-document pairs together. This implementation uses the transformers library to build a re-ranking model.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
    def score_pair(self, query: str, document: str) -> float:
        inputs = self.tokenizer(
            query,
            document,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            scores = self.model(**inputs).logits
            return scores.item()
```

Slide 6: Re-Ranking Implementation

The re-ranking process takes initial retrieval results and refines them using the cross-encoder model. This implementation shows how to combine both retrieval stages effectively.

```python
def rerank_documents(
    initial_results: List[Tuple[str, float]],
    query: str,
    reranker: CrossEncoderReranker,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    
    # Score all documents using cross-encoder
    scored_docs = []
    for doc, initial_score in initial_results:
        cross_encoder_score = reranker.score_pair(query, doc)
        scored_docs.append((doc, cross_encoder_score))
    
    # Sort by cross-encoder scores
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return reranked_docs[:top_k]

# Example usage:
initial_results = [
    ("Document A discusses neural networks", 0.85),
    ("Document B covers machine learning", 0.75),
    ("Document C explains deep learning", 0.70)
]

reranker = CrossEncoderReranker()
query = "neural network architectures"
reranked_results = rerank_documents(initial_results, query, reranker)
print(f"Reranked results: {reranked_results}")
```

Slide 7: Complete RAG Pipeline Implementation

This comprehensive implementation combines document retrieval, re-ranking, and generation into a single pipeline, demonstrating how different components work together in a RAG system.

```python
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGPipeline:
    def __init__(
        self,
        retriever: DenseRetriever,
        reranker: CrossEncoderReranker,
        generator_model: str = 'gpt2'
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = AutoModelForCausalLM.from_pretrained(generator_model)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model)
        
    def generate_response(
        self,
        query: str,
        initial_k: int = 10,
        rerank_k: int = 3
    ) -> str:
        # Initial retrieval
        initial_results = self.retriever.search(query, top_k=initial_k)
        
        # Re-ranking
        reranked_docs = rerank_documents(
            initial_results, query, self.reranker, top_k=rerank_k
        )
```

Slide 8: RAG Pipeline Generation Component

The generation component processes retrieved and re-ranked documents along with the query to produce contextually relevant responses. This implementation shows the integration with a language model.

```python
class RAGPipeline:  # Continuing from previous slide
    def _prepare_context(self, query: str, documents: List[str]) -> str:
        # Combine documents into context
        context = " ".join(documents)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        return prompt
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.generator_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.generator_tokenizer.eos_token_id
            )
            
        response = self.generator_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        return response.replace(prompt, "").strip()

# Example usage:
rag = RAGPipeline(retriever, reranker)
response = rag.generate_response("How do neural networks learn?")
print(f"Generated response: {response}")
```

Slide 9: Document Preprocessing for RAG

Effective document preprocessing is crucial for RAG systems. This implementation demonstrates key preprocessing steps including text cleaning, chunking, and metadata extraction.

```python
import re
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    metadata: Dict
    chunk_id: int

class DocumentPreprocessor:
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def clean_text(self, text: str) -> str:
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s\.]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def chunk_document(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk = DocumentChunk(
                text=chunk_text,
                metadata=metadata,
                chunk_id=len(chunks)
            )
            chunks.append(chunk)
            
        return chunks
```

Slide 10: Real-world Example: Research Paper QA System

This implementation demonstrates a complete RAG system for answering questions about research papers, including PDF parsing and specialized preprocessing.

```python
import fitz  # PyMuPDF
from typing import List, Dict

class ResearchPaperRAG:
    def __init__(self):
        self.preprocessor = DocumentPreprocessor()
        self.retriever = DenseRetriever()
        self.reranker = CrossEncoderReranker()
        
    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        # Extract text from PDF
        doc = fitz.open(pdf_path)
        paper_text = ""
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "pages": len(doc)
        }
        
        for page in doc:
            paper_text += page.get_text()
            
        # Chunk the document
        chunks = self.preprocessor.chunk_document(paper_text, metadata)
        return chunks
        
    def index_paper(self, pdf_path: str):
        chunks = self.process_pdf(pdf_path)
        texts = [chunk.text for chunk in chunks]
        self.retriever.index_documents(texts)
        return len(chunks)
```

Slide 11: Research Paper QA System Query Processing

Implementation of the query processing component for the research paper QA system, including specialized handling of academic queries and citation tracking.

```python
class ResearchPaperRAG:  # Continuing from previous slide
    def answer_question(self, question: str) -> Dict:
        # Get initial candidates
        retrieved_chunks = self.retriever.search(question, top_k=5)
        
        # Re-rank with academic context awareness
        reranked_results = []
        for chunk_id, score in retrieved_chunks:
            chunk = self.chunks[chunk_id]
            cross_score = self.reranker.score_pair(
                question,
                f"{chunk.metadata['title']}: {chunk.text}"
            )
            reranked_results.append((chunk, cross_score))
            
        # Sort and format response
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        top_chunk = reranked_results[0][0]
        
        return {
            "answer": self._generate_answer(question, top_chunk),
            "source": {
                "title": top_chunk.metadata["title"],
                "author": top_chunk.metadata["author"],
                "chunk_id": top_chunk.chunk_id
            },
            "confidence": reranked_results[0][1]
        }

# Example usage:
rag = ResearchPaperRAG()
rag.index_paper("research_paper.pdf")
result = rag.answer_question("What are the main findings of this paper?")
print(f"Answer: {result['answer']}\nSource: {result['source']}")
```

Slide 12: Performance Metrics Implementation

This implementation shows how to evaluate RAG system performance using standard metrics including Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG), and precision@k.

```python
import numpy as np
from typing import List, Dict

class RAGEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def calculate_mrr(self, rankings: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        for ranking in rankings:
            # Add 1 since ranking is 0-based
            rr = 1.0 / (ranking + 1) if ranking >= 0 else 0
            reciprocal_ranks.append(rr)
        return np.mean(reciprocal_ranks)
    
    def calculate_ndcg(
        self,
        relevance_scores: List[float],
        retrieved_scores: List[float],
        k: int = None
    ) -> float:
        """Calculate NDCG@k"""
        if k is not None:
            relevance_scores = relevance_scores[:k]
            retrieved_scores = retrieved_scores[:k]
            
        dcg = np.sum(
            retrieved_scores / np.log2(np.arange(2, len(retrieved_scores) + 2))
        )
        idcg = np.sum(
            sorted(relevance_scores, reverse=True) /
            np.log2(np.arange(2, len(relevance_scores) + 2))
        )
        return dcg / idcg if idcg > 0 else 0.0

# Example usage:
evaluator = RAGEvaluator()
rankings = [0, 2, 1]  # First result at rank 0, third at rank 2, etc.
mrr = evaluator.calculate_mrr(rankings)
print(f"MRR: {mrr:.3f}")
```

Slide 13: Additional Resources

*   Improving Language Understanding by Generative Pre-Training [https://cdn.openai.com/research-covers/language-unsupervised/language\_understanding\_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
*   REALM: Retrieval-Augmented Language Model Pre-Training [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
*   Dense Passage Retrieval for Open-Domain Question Answering [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
*   ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
*   Cross-Encoder Architectures for Neural Re-Ranking [https://www.google.com/search?q=Cross-Encoder+Architectures+for+Neural+Re-Ranking+research+paper](https://www.google.com/search?q=Cross-Encoder+Architectures+for+Neural+Re-Ranking+research+paper)

