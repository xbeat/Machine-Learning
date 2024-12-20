## Optimizing RAG with Document Chunking Techniques Using Python
Slide 1: Introduction to Document Chunking in RAG

Document chunking is a crucial technique in Retrieval-Augmented Generation (RAG) systems that involves breaking down large documents into smaller, manageable pieces. This process improves retrieval efficiency and relevance in RAG applications. We'll explore various chunking methods, starting with fixed-size chunking.

```python
def chunk_document(document, chunk_size):
    return [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]

text = "This is a sample document for chunking demonstration."
chunks = chunk_document(text, 10)
print(chunks)
```

Slide 2: Fixed-Size Chunking

Fixed-size chunking divides a document into segments of equal length, regardless of content boundaries. This method is simple to implement but may split sentences or paragraphs awkwardly.

```python
import nltk
nltk.download('punkt')

def fixed_size_chunking(text, chunk_size):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

text = "This is a sample document. It contains multiple sentences. We will chunk it using fixed-size method."
chunks = fixed_size_chunking(text, 50)
print(chunks)
```

Slide 3: Recursive Chunking

Recursive chunking involves dividing a document into increasingly smaller segments based on a set of rules or conditions. This method can better preserve the document's structure and context.

```python
def recursive_chunking(text, max_chunk_size, min_chunk_size):
    if len(text) <= max_chunk_size:
        return [text]
    
    mid = len(text) // 2
    left_chunk = text[:mid]
    right_chunk = text[mid:]
    
    if len(left_chunk) < min_chunk_size or len(right_chunk) < min_chunk_size:
        return [text]
    
    return recursive_chunking(left_chunk, max_chunk_size, min_chunk_size) + \
           recursive_chunking(right_chunk, max_chunk_size, min_chunk_size)

text = "This is a longer document for recursive chunking demonstration. It contains multiple sentences and paragraphs."
chunks = recursive_chunking(text, 50, 20)
print(chunks)
```

Slide 4: Document-based Chunking

Document-based chunking takes into account the structure and content of the document, such as paragraphs, sections, or natural language boundaries. This method often produces more semantically meaningful chunks.

```python
import re

def document_based_chunking(text):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= 100:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

text = """This is the first paragraph of the document.
It contains multiple sentences.

This is the second paragraph.
It also has multiple sentences for demonstration."""

chunks = document_based_chunking(text)
print(chunks)
```

Slide 5: Implementing Fixed-Size Chunking with Overlap

Overlapping chunks can help maintain context between adjacent segments. This technique is particularly useful when dealing with fixed-size chunking to avoid losing important information at chunk boundaries.

```python
def fixed_size_chunking_with_overlap(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

text = "This is a sample document for demonstrating fixed-size chunking with overlap. It contains multiple sentences."
chunks = fixed_size_chunking_with_overlap(text, 10, 3)
print(chunks)
```

Slide 6: Sentence-based Chunking

Sentence-based chunking groups sentences together to form chunks, ensuring that sentence boundaries are preserved. This method can be more effective for maintaining coherence in natural language processing tasks.

```python
def sentence_based_chunking(text, max_sentences):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        if len(current_chunk) < max_sentences:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

text = "This is the first sentence. This is the second one. Here's the third. And a fourth. Let's add a fifth."
chunks = sentence_based_chunking(text, 2)
print(chunks)
```

Slide 7: Token-based Chunking

Token-based chunking divides text based on a specific number of tokens (words or subwords) rather than characters. This approach can be more semantically meaningful, especially when working with models that have token limits.

```python
from transformers import AutoTokenizer

def token_based_chunking(text, max_tokens):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    
    for token in tokens:
        if len(current_chunk) < max_tokens:
            current_chunk.append(token)
        else:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = [token]
    
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
    
    return chunks

text = "This is a sample text for token-based chunking demonstration using BERT tokenizer."
chunks = token_based_chunking(text, 5)
print(chunks)
```

Slide 8: Sliding Window Chunking

Sliding window chunking creates overlapping chunks by moving a fixed-size window through the text. This method ensures context continuity between chunks and can be useful for tasks requiring local context.

```python
def sliding_window_chunking(text, window_size, step_size):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words) - window_size + 1, step_size):
        chunk = ' '.join(words[i:i+window_size])
        chunks.append(chunk)
    
    return chunks

text = "This is a sample text for sliding window chunking demonstration with overlapping chunks."
chunks = sliding_window_chunking(text, 5, 2)
print(chunks)
```

Slide 9: Semantic Chunking

Semantic chunking aims to create chunks based on the meaning and context of the text. This approach often involves using more advanced NLP techniques to identify coherent segments of information.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def semantic_chunking(text, num_chunks):
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    kmeans = KMeans(n_clusters=num_chunks)
    kmeans.fit(sentence_vectors)
    
    chunks = [[] for _ in range(num_chunks)]
    for i, label in enumerate(kmeans.labels_):
        chunks[label].append(sentences[i])
    
    return [' '.join(chunk) for chunk in chunks]

text = """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.
It focuses on the interaction between computers and humans using natural language.
The ultimate objective of NLP is to enable computers to understand, interpret, and generate human language."""

chunks = semantic_chunking(text, 2)
print(chunks)
```

Slide 10: Hybrid Chunking Approaches

Hybrid chunking combines multiple techniques to leverage the strengths of different approaches. For example, we can use document-based chunking for initial segmentation and then apply fixed-size chunking within each segment.

```python
def hybrid_chunking(text, paragraph_max_length, sentence_max_length):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    
    for paragraph in paragraphs:
        if len(paragraph) <= paragraph_max_length:
            chunks.append(paragraph)
        else:
            sentences = nltk.sent_tokenize(paragraph)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= sentence_max_length:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
    
    return chunks

text = """This is the first paragraph of the document.
It contains multiple sentences for demonstration.

This is the second paragraph. It's shorter than the first one.

And here's a third paragraph with some more content."""

chunks = hybrid_chunking(text, 100, 50)
print(chunks)
```

Slide 11: Chunking with Named Entity Recognition

Incorporating Named Entity Recognition (NER) into the chunking process can help preserve important entities within chunks, ensuring that related information stays together.

```python
import spacy

def ner_based_chunking(text, max_chunk_size):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    
    for sent in doc.sents:
        if len(current_chunk) + len(sent.text) <= max_chunk_size:
            current_chunk += sent.text + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent.text + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

text = "Apple Inc. is a technology company based in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."
chunks = ner_based_chunking(text, 50)
print(chunks)
```

Slide 12: Dynamic Chunking Based on Content Density

This approach adjusts chunk sizes based on the density of important information in the text. Sections with higher information density are chunked into smaller segments, while less dense sections can have larger chunks.

```python
import re
from collections import Counter

def calculate_density(text):
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    unique_words = len(word_freq)
    total_words = len(words)
    return unique_words / total_words

def dynamic_density_chunking(text, min_chunk_size, max_chunk_size):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        density = calculate_density(current_chunk + sentence)
        target_size = min_chunk_size + (1 - density) * (max_chunk_size - min_chunk_size)
        
        if len(current_chunk) + len(sentence) <= target_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

text = """The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991.
The theory of relativity was developed by Albert Einstein. It fundamentally changed our understanding of space and time."""

chunks = dynamic_density_chunking(text, 50, 150)
print(chunks)
```

Slide 13: Evaluating Chunking Quality

To assess the effectiveness of different chunking methods, we can implement a simple evaluation metric that considers factors such as chunk size consistency, content coherence, and information preservation.

```python
import numpy as np

def evaluate_chunking(original_text, chunks):
    # Chunk size consistency
    chunk_sizes = [len(chunk) for chunk in chunks]
    size_consistency = 1 - (np.std(chunk_sizes) / np.mean(chunk_sizes))
    
    # Content coherence (simple version based on sentence boundaries)
    sentence_breaks = sum(chunk.count('.') for chunk in chunks) - chunks[-1].count('.')
    coherence_score = 1 - (sentence_breaks / len(chunks))
    
    # Information preservation (based on unique words)
    original_words = set(re.findall(r'\w+', original_text.lower()))
    chunked_words = set(word for chunk in chunks for word in re.findall(r'\w+', chunk.lower()))
    preservation_score = len(chunked_words) / len(original_words)
    
    # Overall score (equal weight for simplicity)
    overall_score = (size_consistency + coherence_score + preservation_score) / 3
    
    return {
        "size_consistency": size_consistency,
        "coherence_score": coherence_score,
        "preservation_score": preservation_score,
        "overall_score": overall_score
    }

text = "This is a sample text for chunking evaluation. It contains multiple sentences with varying lengths and content."
chunks = fixed_size_chunking(text, 30)
evaluation = evaluate_chunking(text, chunks)
print(evaluation)
```

Slide 14: Additional Resources

For further exploration of document chunking techniques and their applications in RAG systems, consider the following resources:

1. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (ArXiv:2005.11401) [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
2. "Improving Language Understanding by Generative Pre-Training" (OpenAI GPT paper) [https://cdn.openai.com/research-covers/language-unsupervised/language\_understanding\_paper.pdf](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
3. "REALM: Retrieval-Augmented Language Model Pre-Training" (ArXiv:2002.08909) [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
4. "Dense Passage Retrieval for Open-Domain Question Answering" (ArXiv:2004.04906) [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)

These papers provide valuable insights into the theoretical foundations and practical applications of document chunking in the context of retrieval-augmented generation and related natural language processing tasks.

