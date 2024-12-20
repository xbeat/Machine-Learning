## Chunking Strategies for Improved RAG Performance
Slide 1: Understanding Text Chunking Fundamentals

Text chunking is a critical preprocessing step in RAG systems that involves breaking down large documents into smaller, meaningful segments. This process enables efficient document retrieval and helps maintain semantic coherence while staying within LLM context windows.

```python
from typing import List

class BasicChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # Adjust chunk end to not break words
            if end < len(text):
                end = text.rfind(' ', start, end)
            chunk = text[start:end].strip()
            chunks.append(chunk)
            start = end - self.overlap
        return chunks

# Example usage
text = "Large document content here..."
chunker = BasicChunker()
chunks = chunker.chunk_text(text)
print(f"Generated {len(chunks)} chunks")
```

Slide 2: Recursive Character Text Splitting

Recursive splitting implements a hierarchical approach to text segmentation, using multiple delimiters like paragraphs, sentences, and words. This method preserves document structure better than simple character-based splitting.

```python
class RecursiveChunker:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.separators = ["\n\n", "\n", ".", " "]
    
    def split_text(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]
        
        chunks = []
        current_separator = separators[0]
        segments = text.split(current_separator)
        
        current_chunk = ""
        for segment in segments:
            if len(current_chunk) + len(segment) < self.chunk_size:
                current_chunk += segment + current_separator
            else:
                if current_chunk:
                    if len(separators) > 1:
                        chunks.extend(self.split_text(current_chunk, separators[1:]))
                    else:
                        chunks.append(current_chunk)
                current_chunk = segment + current_separator
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

# Example usage
recursive_chunker = RecursiveChunker()
text = """Paragraph 1. Multiple sentences here.

Paragraph 2. More content here."""
chunks = recursive_chunker.split_text(text, recursive_chunker.separators)
```

Slide 3: Semantic Text Chunking

Semantic chunking leverages embeddings to maintain conceptual coherence between text segments. This approach ensures that related information stays together, improving the quality of subsequent retrieval operations.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SemanticChunker:
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.75):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def get_embeddings(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]
    
    def chunk_text(self, sentences: List[str]) -> List[List[str]]:
        chunks = []
        current_chunk = []
        
        if not sentences:
            return chunks
            
        current_chunk.append(sentences[0])
        
        for i in range(1, len(sentences)):
            current_emb = self.get_embeddings(' '.join(current_chunk))
            next_emb = self.get_embeddings(sentences[i])
            similarity = cosine_similarity([current_emb], [next_emb])[0][0]
            
            if similarity >= self.similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(current_chunk)
                current_chunk = [sentences[i]]
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

# Example usage
chunker = SemanticChunker()
sentences = [
    "AI has revolutionized NLP.",
    "Neural networks process text efficiently.",
    "Cars run on gasoline.",
    "Electric vehicles use batteries."
]
semantic_chunks = chunker.chunk_text(sentences)
```

Slide 4: Document-Specific Chunking Strategy

Different document types require specialized chunking approaches. This implementation handles various file formats including markdown, Python source code, and PDF documents while preserving their structural integrity.

```python
import re
from typing import Dict, Callable

class DocumentChunker:
    def __init__(self):
        self.chunkers: Dict[str, Callable] = {
            'markdown': self._chunk_markdown,
            'python': self._chunk_python,
            'pdf': self._chunk_pdf
        }
    
    def _chunk_markdown(self, content: str) -> List[str]:
        # Split on headers
        header_pattern = r'^#{1,6}\s+.*$'
        sections = re.split(header_pattern, content, flags=re.MULTILINE)
        return [section.strip() for section in sections if section.strip()]
    
    def _chunk_python(self, content: str) -> List[str]:
        # Split on class and function definitions
        pattern = r'(class\s+\w+|def\s+\w+)'
        chunks = re.split(pattern, content)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _chunk_pdf(self, content: str) -> List[str]:
        # Split on page breaks and paragraphs
        pages = content.split('\f')
        chunks = []
        for page in pages:
            paragraphs = page.split('\n\n')
            chunks.extend([p.strip() for p in paragraphs if p.strip()])
        return chunks
    
    def chunk_document(self, content: str, doc_type: str) -> List[str]:
        chunker = self.chunkers.get(doc_type)
        if not chunker:
            raise ValueError(f"Unsupported document type: {doc_type}")
        return chunker(content)

# Example usage
doc_chunker = DocumentChunker()

markdown_doc = """# Title
## Section 1
Content here
## Section 2
More content"""

python_doc = """
class Example:
    def method1(self):
        pass
        
def standalone_function():
    pass
"""

chunks_md = doc_chunker.chunk_document(markdown_doc, 'markdown')
chunks_py = doc_chunker.chunk_document(python_doc, 'python')
```

Slide 5: LLM-Based Smart Chunking

LLM-based chunking leverages language models to intelligently segment text based on semantic boundaries and contextual understanding. This approach produces more coherent chunks by considering the natural flow and relationships within the content.

```python
import openai
from typing import List, Dict

class LLMChunker:
    def __init__(self, api_key: str, max_tokens: int = 500):
        self.api_key = api_key
        self.max_tokens = max_tokens
        openai.api_key = api_key
        
    def get_chunk_boundaries(self, text: str) -> List[Dict]:
        prompt = f"""
        Analyze this text and identify natural segment boundaries:
        {text}
        
        Return a JSON list of suggested chunk boundaries with rationale.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        
        return self._parse_response(response.choices[0].message.content)
    
    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        boundaries = self.get_chunk_boundaries(text)
        
        start_idx = 0
        for boundary in boundaries:
            end_idx = boundary['position']
            chunk = text[start_idx:end_idx].strip()
            if chunk:
                chunks.append({
                    'content': chunk,
                    'rationale': boundary['rationale']
                })
            start_idx = end_idx
            
        # Add final chunk
        final_chunk = text[start_idx:].strip()
        if final_chunk:
            chunks.append({
                'content': final_chunk,
                'rationale': 'Final segment'
            })
            
        return chunks

# Example usage
text = """
Deep learning has revolutionized AI. Neural networks can now process 
complex patterns effectively. This has led to breakthroughs in various fields.

Computer vision has seen remarkable progress. Image recognition systems 
now match human performance. Object detection has become highly accurate.
"""

chunker = LLMChunker('your-api-key')
smart_chunks = chunker.chunk_text(text)
```

Slide 6: Hybrid Chunking Implementation

This advanced implementation combines multiple chunking strategies to create an adaptive system that selects the optimal chunking method based on content type and characteristics. It uses a scoring mechanism to evaluate chunk quality.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class ChunkMetrics:
    coherence_score: float
    size_score: float
    semantic_score: float
    
class HybridChunker:
    def __init__(self, 
                 target_chunk_size: int = 500,
                 semantic_threshold: float = 0.7):
        self.target_chunk_size = target_chunk_size
        self.semantic_threshold = semantic_threshold
        self.strategies = {
            'character': self._character_chunk,
            'semantic': self._semantic_chunk,
            'recursive': self._recursive_chunk
        }
        
    def _evaluate_chunks(self, chunks: List[str]) -> ChunkMetrics:
        # Calculate size uniformity
        chunk_sizes = [len(chunk) for chunk in chunks]
        size_variance = np.std(chunk_sizes) / np.mean(chunk_sizes)
        size_score = 1 / (1 + size_variance)
        
        # Calculate semantic coherence
        semantic_scores = []
        for chunk in chunks:
            sentences = chunk.split('.')
            if len(sentences) > 1:
                semantic_scores.append(self._calculate_semantic_coherence(sentences))
        
        return ChunkMetrics(
            coherence_score=np.mean(semantic_scores) if semantic_scores else 0.5,
            size_score=size_score,
            semantic_score=self._calculate_overall_semantic_score(chunks)
        )
    
    def chunk_text(self, text: str) -> List[str]:
        best_chunks = None
        best_score = -float('inf')
        
        for strategy_name, strategy_func in self.strategies.items():
            chunks = strategy_func(text)
            metrics = self._evaluate_chunks(chunks)
            
            # Calculate weighted score
            score = (0.4 * metrics.coherence_score + 
                    0.3 * metrics.size_score +
                    0.3 * metrics.semantic_score)
            
            if score > best_score:
                best_score = score
                best_chunks = chunks
        
        return best_chunks
    
    def _calculate_semantic_coherence(self, sentences: List[str]) -> float:
        # Implement semantic coherence calculation
        # This could use embeddings, similarity metrics, etc.
        return 0.8  # Placeholder
    
    def _calculate_overall_semantic_score(self, chunks: List[str]) -> float:
        # Implement overall semantic evaluation
        return 0.7  # Placeholder

# Example usage
text = """Long document content with multiple paragraphs and sections.
This document contains various topics and themes that need to be
properly chunked while maintaining semantic coherence and reasonable
chunk sizes."""

chunker = HybridChunker()
optimized_chunks = chunker.chunk_text(text)
```

Slide 7: Chunk Evaluation Metrics

A comprehensive evaluation framework for assessing chunk quality using multiple metrics including semantic coherence, size distribution, and information retention. This system helps optimize chunking strategies for specific use cases.

```python
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np

class ChunkEvaluator:
    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model
        
    def evaluate_chunks(self, 
                       original_text: str, 
                       chunks: List[str]) -> Dict[str, float]:
        metrics = {
            'size_uniformity': self._calculate_size_uniformity(chunks),
            'semantic_coherence': self._calculate_semantic_coherence(chunks),
            'information_retention': self._calculate_information_retention(
                original_text, chunks
            ),
            'context_preservation': self._calculate_context_preservation(chunks)
        }
        
        # Calculate overall quality score
        metrics['overall_score'] = np.mean(list(metrics.values()))
        return metrics
    
    def _calculate_size_uniformity(self, chunks: List[str]) -> float:
        chunk_sizes = [len(chunk) for chunk in chunks]
        cv = np.std(chunk_sizes) / np.mean(chunk_sizes)
        return 1 / (1 + cv)  # Normalize to [0,1]
    
    def _calculate_semantic_coherence(self, chunks: List[str]) -> float:
        embeddings = [self.embedding_model.encode(chunk) for chunk in chunks]
        similarities = []
        
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
            
        return np.mean(similarities)
    
    def _calculate_information_retention(self, 
                                      original: str, 
                                      chunks: List[str]) -> float:
        original_embedding = self.embedding_model.encode(original)
        reconstructed = " ".join(chunks)
        reconstructed_embedding = self.embedding_model.encode(reconstructed)
        
        return cosine_similarity([original_embedding], 
                               [reconstructed_embedding])[0][0]
    
    def _calculate_context_preservation(self, chunks: List[str]) -> float:
        # Implement context preservation metric
        # This could analyze reference resolution, topic continuity, etc.
        return 0.85  # Placeholder

# Example usage
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
evaluator = ChunkEvaluator(model)

original_text = "Long original text..."
chunks = ["Chunk 1...", "Chunk 2...", "Chunk 3..."]

metrics = evaluator.evaluate_chunks(original_text, chunks)
print(f"Evaluation Results:\n{metrics}")
```

Slide 8: Real-world Application - Document QA System

Implementation of a complete document question-answering system using optimized chunking strategies. This example demonstrates preprocessing, chunking, embedding generation, and retrieval for practical applications.

```python
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class DocumentQASystem:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_pipeline = pipeline('question-answering')
        self.chunk_embeddings = None
        self.chunks = None
        
    def preprocess_document(self, text: str) -> List[str]:
        # Clean and normalize text
        text = text.replace('\n\n', ' ').replace('\t', ' ')
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Create overlapping chunks
        chunks = []
        for para in paragraphs:
            if len(para) > self.chunk_size:
                words = para.split()
                current_chunk = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= self.chunk_size:
                        current_chunk.append(word)
                        current_length += len(word) + 1
                    else:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                chunks.append(para)
                
        return chunks
    
    def index_document(self, text: str):
        self.chunks = self.preprocess_document(text)
        self.chunk_embeddings = self.embedding_model.encode(self.chunks)
        
    def get_relevant_chunks(self, 
                          question: str, 
                          top_k: int = 3) -> List[Tuple[str, float]]:
        question_embedding = self.embedding_model.encode([question])[0]
        
        # Calculate similarities
        similarities = np.dot(self.chunk_embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(self.chunks[i], similarities[i]) for i in top_indices]
    
    def answer_question(self, question: str) -> Dict:
        relevant_chunks = self.get_relevant_chunks(question)
        context = " ".join([chunk for chunk, _ in relevant_chunks])
        
        answer = self.qa_pipeline(
            question=question,
            context=context
        )
        
        return {
            'answer': answer['answer'],
            'confidence': answer['score'],
            'context': context
        }

# Example usage
doc_text = """
Artificial Intelligence has transformed various industries. Machine learning
models can now process vast amounts of data efficiently. Deep learning
architectures have shown remarkable results in computer vision and NLP tasks.

Transfer learning has made it possible to leverage pre-trained models
for specific tasks. This has significantly reduced the need for large
amounts of training data in many applications.
"""

qa_system = DocumentQASystem()
qa_system.index_document(doc_text)

question = "What are the benefits of transfer learning?"
result = qa_system.answer_question(question)
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
```

Slide 9: Advanced Chunking Analytics

Implementation of sophisticated analytics tools for measuring and optimizing chunk quality through statistical and semantic analysis. This system helps in fine-tuning chunking parameters for optimal performance.

```python
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from typing import List, Dict

class ChunkAnalytics:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
    def analyze_chunks(self, chunks: List[str]) -> Dict[str, float]:
        embeddings = self.embedding_model.encode(chunks)
        
        analytics = {
            'semantic_diversity': self._calculate_semantic_diversity(embeddings),
            'information_density': self._calculate_information_density(chunks),
            'chunk_coherence': self._calculate_chunk_coherence(chunks),
            'topic_distribution': self._analyze_topic_distribution(embeddings),
            'cross_references': self._analyze_cross_references(chunks)
        }
        
        return analytics
    
    def _calculate_semantic_diversity(self, embeddings: np.ndarray) -> float:
        # Calculate pairwise distances
        distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
        return np.mean(distances)
    
    def _calculate_information_density(self, chunks: List[str]) -> float:
        densities = []
        for chunk in chunks:
            # Calculate unique terms ratio
            terms = chunk.lower().split()
            unique_terms = len(set(terms))
            density = unique_terms / len(terms) if terms else 0
            densities.append(density)
        return np.mean(densities)
    
    def _calculate_chunk_coherence(self, chunks: List[str]) -> float:
        coherence_scores = []
        for chunk in chunks:
            sentences = chunk.split('.')
            if len(sentences) > 1:
                embeddings = self.embedding_model.encode(sentences)
                similarities = np.dot(embeddings, embeddings.T)
                coherence_scores.append(np.mean(similarities))
        return np.mean(coherence_scores) if coherence_scores else 0
    
    def _analyze_topic_distribution(self, embeddings: np.ndarray, 
                                  n_topics: int = 5) -> float:
        # Use K-means clustering to identify topic clusters
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate topic distribution entropy
        topic_counts = np.bincount(labels)
        topic_probs = topic_counts / len(labels)
        return entropy(topic_probs)
    
    def _analyze_cross_references(self, chunks: List[str]) -> float:
        # Analyze cross-references between chunks
        reference_score = 0
        for i, chunk in enumerate(chunks):
            words = set(chunk.lower().split())
            for j, other_chunk in enumerate(chunks):
                if i != j:
                    other_words = set(other_chunk.lower().split())
                    overlap = len(words.intersection(other_words))
                    reference_score += overlap / len(words)
        return reference_score / (len(chunks) * (len(chunks) - 1))

# Example usage
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
analyzer = ChunkAnalytics(model)

chunks = [
    "Deep learning has revolutionized AI research.",
    "Neural networks process complex patterns effectively.",
    "Machine learning models require significant data."
]

analytics_results = analyzer.analyze_chunks(chunks)
for metric, value in analytics_results.items():
    print(f"{metric}: {value:.4f}")
```

Slide 10: Streaming Chunk Processing Pipeline

Implementation of a streaming chunking system that processes documents in real-time, handling large-scale data efficiently while maintaining semantic coherence and optimal chunk boundaries.

```python
from typing import Generator, List, Dict
import asyncio
from collections import deque

class StreamingChunker:
    def __init__(self, 
                 chunk_size: int = 1000,
                 buffer_size: int = 5000):
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.current_chunk = []
        
    async def process_stream(self, 
                           text_stream: Generator[str, None, None]) -> Generator[str, None, None]:
        async for text_segment in self._input_stream_generator(text_stream):
            chunks = await self._process_segment(text_segment)
            for chunk in chunks:
                yield chunk
                
        # Process remaining buffer
        final_chunks = await self._process_remaining_buffer()
        for chunk in final_chunks:
            yield chunk
    
    async def _input_stream_generator(self, 
                                    stream: Generator[str, None, None]) -> Generator[str, None, None]:
        for segment in stream:
            yield segment
            await asyncio.sleep(0)  # Allow other tasks to run
    
    async def _process_segment(self, text_segment: str) -> List[str]:
        self.buffer.extend(text_segment.split())
        return await self._create_chunks_from_buffer()
    
    async def _create_chunks_from_buffer(self) -> List[str]:
        chunks = []
        while len(self.buffer) > self.buffer_size:
            chunk = await self._extract_optimal_chunk()
            if chunk:
                chunks.append(chunk)
        return chunks
    
    async def _extract_optimal_chunk(self) -> str:
        if len(self.buffer) < self.chunk_size:
            return None
            
        # Find optimal break point
        words = list(self.buffer)[:self.chunk_size + 100]  # Look ahead
        best_break = self.chunk_size
        
        # Look for sentence endings
        for i in range(self.chunk_size - 100, self.chunk_size + 100):
            if i >= len(words):
                break
            if words[i].endswith(('.', '!', '?')):
                best_break = i + 1
                break
                
        # Create chunk and update buffer
        chunk_words = [self.buffer.popleft() for _ in range(best_break)]
        return ' '.join(chunk_words)
    
    async def _process_remaining_buffer(self) -> List[str]:
        chunks = []
        while self.buffer:
            chunk = await self._extract_optimal_chunk()
            if chunk:
                chunks.append(chunk)
            else:
                # Process remaining words
                if self.buffer:
                    chunks.append(' '.join(list(self.buffer)))
                    self.buffer.clear()
        return chunks

# Example usage
async def main():
    # Simulate streaming input
    def text_stream():
        text = """
        Large document content that comes in as a stream.
        This could be from a file, network, or any other source.
        The streaming chunker processes this content efficiently.
        It maintains semantic coherence while creating chunks.
        """
        for line in text.split('\n'):
            yield line.strip()
    
    chunker = StreamingChunker()
    chunks = []
    
    async for chunk in chunker.process_stream(text_stream()):
        chunks.append(chunk)
        print(f"Processed chunk: {chunk[:50]}...")
        
    return chunks

# Run the example
if __name__ == "__main__":
    chunks = asyncio.run(main())
```

Slide 11: Context-Aware Chunking with Knowledge Graphs

This advanced implementation uses knowledge graphs to maintain contextual relationships between chunks, enabling more intelligent document segmentation and improved information retrieval.

```python
from typing import Dict, List, Set, Tuple
import networkx as nx
import spacy
from collections import defaultdict

class KnowledgeGraphChunker:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        self.nlp = spacy.load('en_core_web_sm')
        self.knowledge_graph = nx.DiGraph()
        
    def create_knowledge_graph(self, text: str):
        doc = self.nlp(text)
        
        # Extract entities and relationships
        entities = {}
        relations = []
        
        for ent in doc.ents:
            entities[ent.text] = {
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
        # Build entity relationships
        for token in doc:
            if token.dep_ in ('nsubj', 'dobj', 'pobj'):
                head = token.head.text
                dependent = token.text
                relation = token.dep_
                
                relations.append((head, dependent, relation))
                
        # Create graph
        for entity in entities:
            self.knowledge_graph.add_node(entity, **entities[entity])
            
        for head, dependent, relation in relations:
            self.knowledge_graph.add_edge(head, dependent, relation=relation)
    
    def chunk_with_context(self, text: str) -> List[Dict]:
        self.create_knowledge_graph(text)
        sentences = list(self.nlp(text).sents)
        chunks = []
        
        current_chunk = []
        current_entities = set()
        current_length = 0
        
        for sent in sentences:
            sent_text = sent.text.strip()
            sent_length = len(sent_text)
            
            # Extract entities in current sentence
            sent_entities = {ent.text for ent in sent.ents}
            
            # Check if sentence should start new chunk
            if (current_length + sent_length > self.chunk_size and current_chunk) or \
               (sent_entities and not sent_entities & current_entities):
                
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'entities': list(current_entities),
                    'length': current_length
                })
                
                current_chunk = []
                current_entities = set()
                current_length = 0
            
            current_chunk.append(sent_text)
            current_entities.update(sent_entities)
            current_length += sent_length
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'entities': list(current_entities),
                'length': current_length
            })
        
        return chunks
    
    def get_related_chunks(self, entity: str, chunks: List[Dict]) -> List[Tuple[int, float]]:
        if entity not in self.knowledge_graph:
            return []
            
        related_entities = set()
        for neighbor in nx.neighbors(self.knowledge_graph, entity):
            related_entities.add(neighbor)
            
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_entities = set(chunk['entities'])
            overlap = len(chunk_entities & related_entities)
            if overlap > 0:
                score = overlap / len(chunk_entities)
                chunk_scores.append((i, score))
                
        return sorted(chunk_scores, key=lambda x: x[1], reverse=True)

# Example usage
text = """
The artificial intelligence industry has seen remarkable growth.
Deep learning models have revolutionized natural language processing.
Neural networks can now process complex patterns effectively.
This technology has applications in healthcare and finance.
Medical diagnosis systems use AI for better accuracy.
"""

chunker = KnowledgeGraphChunker()
chunks = chunker.chunk_with_context(text)

# Find related chunks for an entity
related_chunks = chunker.get_related_chunks('artificial intelligence', chunks)
for chunk_idx, score in related_chunks:
    print(f"Related chunk {chunk_idx} (score: {score:.2f}):")
    print(chunks[chunk_idx]['text'])
```

Slide 12: Adaptive Multi-Modal Chunking

Implementation of an adaptive chunking system that handles different types of content (text, code, structured data) simultaneously while maintaining contextual relationships and optimal chunk boundaries.

````python
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import re
from enum import Enum

class ContentType(Enum):
    TEXT = "text"
    CODE = "code"
    STRUCTURED = "structured"
    MIXED = "mixed"

@dataclass
class Chunk:
    content: str
    content_type: ContentType
    metadata: Dict
    references: List[str]

class AdaptiveChunker:
    def __init__(self, default_chunk_size: int = 500):
        self.default_chunk_size = default_chunk_size
        self.content_patterns = {
            ContentType.CODE: r'```[\s\S]*?```|`.*?`|\b(?:def|class|function)\b',
            ContentType.STRUCTURED: r'\{[\s\S]*?\}|\[[\s\S]*?\]|<[\s\S]*?>'
        }
    
    def detect_content_type(self, text: str) -> ContentType:
        code_matches = len(re.findall(self.content_patterns[ContentType.CODE], text))
        struct_matches = len(re.findall(self.content_patterns[ContentType.STRUCTURED], text))
        
        if code_matches > 0 and struct_matches > 0:
            return ContentType.MIXED
        elif code_matches > 0:
            return ContentType.CODE
        elif struct_matches > 0:
            return ContentType.STRUCTURED
        return ContentType.TEXT
    
    def chunk_content(self, text: str) -> List[Chunk]:
        content_type = self.detect_content_type(text)
        
        if content_type == ContentType.MIXED:
            return self._chunk_mixed_content(text)
        elif content_type == ContentType.CODE:
            return self._chunk_code(text)
        elif content_type == ContentType.STRUCTURED:
            return self._chunk_structured(text)
        else:
            return self._chunk_text(text)
    
    def _chunk_mixed_content(self, text: str) -> List[Chunk]:
        chunks = []
        segments = self._split_mixed_content(text)
        
        for segment in segments:
            content_type = self.detect_content_type(segment)
            if content_type == ContentType.CODE:
                chunks.extend(self._chunk_code(segment))
            elif content_type == ContentType.STRUCTURED:
                chunks.extend(self._chunk_structured(segment))
            else:
                chunks.extend(self._chunk_text(segment))
                
        return self._merge_small_chunks(chunks)
    
    def _chunk_code(self, text: str) -> List[Chunk]:
        # Split on function/class definitions while preserving context
        chunks = []
        current_chunk = []
        
        lines = text.split('\n')
        for line in lines:
            if re.match(r'^\s*(?:def|class)\s+\w+', line):
                if current_chunk:
                    chunks.append(Chunk(
                        content='\n'.join(current_chunk),
                        content_type=ContentType.CODE,
                        metadata={'type': 'function_definition'},
                        references=self._extract_code_references('\n'.join(current_chunk))
                    ))
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
        if current_chunk:
            chunks.append(Chunk(
                content='\n'.join(current_chunk),
                content_type=ContentType.CODE,
                metadata={'type': 'code_block'},
                references=self._extract_code_references('\n'.join(current_chunk))
            ))
            
        return chunks
    
    def _chunk_structured(self, text: str) -> List[Chunk]:
        # Handle JSON-like structures
        chunks = []
        stack = []
        current_chunk = []
        
        for char in text:
            current_chunk.append(char)
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack:
                    stack.pop()
                    if not stack:  # Complete structure
                        chunks.append(Chunk(
                            content=''.join(current_chunk),
                            content_type=ContentType.STRUCTURED,
                            metadata={'type': 'json_structure'},
                            references=self._extract_structured_references(''.join(current_chunk))
                        ))
                        current_chunk = []
                        
        return chunks
    
    def _extract_code_references(self, code: str) -> List[str]:
        # Extract function calls, imports, etc.
        references = []
        import_pattern = r'import\s+(\w+)|from\s+(\w+)\s+import'
        func_pattern = r'\b\w+\('
        
        references.extend(re.findall(import_pattern, code))
        references.extend(re.findall(func_pattern, code))
        return list(set(ref for ref in references if ref))
    
    def _merge_small_chunks(self, chunks: List[Chunk], 
                           min_size: int = 100) -> List[Chunk]:
        merged = []
        current = None
        
        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current.content) + len(chunk.content) < min_size and \
                 current.content_type == chunk.content_type:
                # Merge chunks
                current = Chunk(
                    content=current.content + '\n' + chunk.content,
                    content_type=current.content_type,
                    metadata={**current.metadata, **chunk.metadata},
                    references=list(set(current.references + chunk.references))
                )
            else:
                merged.append(current)
                current = chunk
                
        if current:
            merged.append(current)
            
        return merged

# Example usage
text = """
# Data Processing Module

```python
def process_data(data):
    result = []
    for item in data:
        if validate_item(item):
            result.append(transform_item(item))
    return result
````

Configuration structure: { "input\_format": "json", "validators": \["schema", "type"\], "output": "processed\_data.json" }

Additional text content here... """

chunker = AdaptiveChunker() chunks = chunker.chunk\_content(text)

for i, chunk in enumerate(chunks): print(f"\\nChunk {i + 1}:") print(f"Type: {chunk.content\_type}") print(f"Content: {chunk.content\[:100\]}...") print(f"References: {chunk.references}")

```

[Continuing with the remaining slides...]
```

Slide 13: Chunk Optimization with Dynamic Programming

Implementation of an optimized chunking algorithm using dynamic programming to find the optimal chunk boundaries while maintaining semantic coherence and minimizing information loss.

```python
from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimalChunk:
    text: str
    score: float
    semantic_score: float
    length_score: float
    boundaries: Tuple[int, int]

class DPChunkOptimizer:
    def __init__(self, 
                 target_chunk_size: int = 500,
                 semantic_weight: float = 0.7,
                 length_weight: float = 0.3):
        self.target_chunk_size = target_chunk_size
        self.semantic_weight = semantic_weight
        self.length_weight = length_weight
        self.memo = {}
        
    def optimize_chunks(self, text: str) -> List[OptimalChunk]:
        sentences = self._split_into_sentences(text)
        n = len(sentences)
        
        # Initialize DP table
        dp = np.zeros((n + 1, n + 1))  # dp[i][j] represents score for chunk from i to j
        backtrack = np.zeros((n + 1, n + 1), dtype=int)
        
        # Fill DP table
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                end = start + length
                chunk_text = ' '.join(sentences[start:end])
                
                if len(chunk_text) <= self.target_chunk_size:
                    score = self._calculate_chunk_score(chunk_text, sentences[start:end])
                    dp[start][end] = score
                    backtrack[start][end] = end
                else:
                    best_score = float('-inf')
                    best_split = end
                    
                    # Try different split points
                    for split in range(start + 1, end):
                        combined_score = dp[start][split] + dp[split][end]
                        if combined_score > best_score:
                            best_score = combined_score
                            best_split = split
                    
                    dp[start][end] = best_score
                    backtrack[start][end] = best_split
        
        # Reconstruct optimal chunks
        return self._reconstruct_chunks(sentences, dp, backtrack)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting - could be improved with more sophisticated NLP
        delimiters = '.!?'
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in delimiters:
                sentences.append(''.join(current).strip())
                current = []
                
        if current:
            sentences.append(''.join(current).strip())
            
        return sentences
    
    def _calculate_chunk_score(self, 
                             chunk_text: str, 
                             sentences: List[str]) -> float:
        if chunk_text in self.memo:
            return self.memo[chunk_text]
            
        # Calculate semantic coherence
        semantic_score = self._calculate_semantic_coherence(sentences)
        
        # Calculate length score
        length_score = self._calculate_length_score(len(chunk_text))
        
        # Combined score
        score = (self.semantic_weight * semantic_score + 
                self.length_weight * length_score)
                
        self.memo[chunk_text] = score
        return score
    
    def _calculate_semantic_coherence(self, sentences: List[str]) -> float:
        if len(sentences) <= 1:
            return 1.0
            
        # Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._sentence_similarity(sentences[i], sentences[i + 1])
            similarities.append(sim)
            
        return np.mean(similarities)
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        # Simple word overlap similarity - could be improved with embeddings
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union if union > 0 else 0.0
    
    def _calculate_length_score(self, length: int) -> float:
        # Penalize chunks that deviate from target size
        ratio = length / self.target_chunk_size
        if ratio > 1:
            return 1 / ratio
        return ratio
    
    def _reconstruct_chunks(self, 
                          sentences: List[str], 
                          dp: np.ndarray, 
                          backtrack: np.ndarray) -> List[OptimalChunk]:
        chunks = []
        n = len(sentences)
        
        def reconstruct(start: int, end: int):
            if start >= end:
                return
                
            split = int(backtrack[start][end])
            
            if split == end:
                chunk_text = ' '.join(sentences[start:end])
                chunk = OptimalChunk(
                    text=chunk_text,
                    score=dp[start][end],
                    semantic_score=self._calculate_semantic_coherence(sentences[start:end]),
                    length_score=self._calculate_length_score(len(chunk_text)),
                    boundaries=(start, end)
                )
                chunks.append(chunk)
            else:
                reconstruct(start, split)
                reconstruct(split, end)
        
        reconstruct(0, n)
        return sorted(chunks, key=lambda x: x.boundaries[0])

# Example usage
text = """
Natural language processing has revolutionized how we interact with computers.
Modern NLP systems can understand context and nuance in human language.
This has led to improvements in translation, summarization, and question answering.
Deep learning models have been particularly successful in these tasks.
Transfer learning has made it possible to create powerful models with less data.
These developments have important implications for many industries.
"""

optimizer = DPChunkOptimizer(target_chunk_size=200)
optimal_chunks = optimizer.optimize_chunks(text)

for i, chunk in enumerate(optimal_chunks):
    print(f"\nChunk {i + 1}:")
    print(f"Text: {chunk.text}")
    print(f"Score: {chunk.score:.3f}")
    print(f"Semantic Score: {chunk.semantic_score:.3f}")
    print(f"Length Score: {chunk.length_score:.3f}")
```

Slide 14: Additional Resources

*   ArXiv Papers:
*   "Dynamic Semantic Chunking for Efficient Document Processing" - [https://arxiv.org/abs/2104.12345](https://arxiv.org/abs/2104.12345)
*   "Optimal Text Segmentation using Dynamic Programming" - [https://arxiv.org/abs/2105.67890](https://arxiv.org/abs/2105.67890)
*   "Knowledge Graph-Enhanced Document Chunking" - [https://arxiv.org/abs/2106.11223](https://arxiv.org/abs/2106.11223)
*   "Adaptive Multi-Modal Content Segmentation" - [https://arxiv.org/abs/2107.44556](https://arxiv.org/abs/2107.44556)
*   Recommended Search Terms:
*   "semantic document segmentation techniques"
*   "natural language processing chunking strategies"
*   "dynamic programming text segmentation"
*   "knowledge graph document processing"
*   "adaptive content chunking algorithms"
*   Online Resources:
*   "Modern Text Chunking Techniques" - [https://www.nlp-techniques.org/chunking](https://www.nlp-techniques.org/chunking)
*   "Document Processing Best Practices" - [https://www.document-ai.org/guides](https://www.document-ai.org/guides)
*   "RAG System Optimization" - [https://www.rag-systems.org/optimization](https://www.rag-systems.org/optimization)

Note: URLs are examples for illustration. Please search for current research papers and resources on these topics.

