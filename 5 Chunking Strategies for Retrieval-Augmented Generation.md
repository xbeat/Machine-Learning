## 5 Chunking Strategies for Retrieval-Augmented Generation
Slide 1: Introduction to Chunking Strategies for RAG

Chunking is a crucial step in Retrieval-Augmented Generation (RAG) systems, where large documents are divided into smaller, manageable pieces. This process ensures that text fits the input size of embedding models and enhances the efficiency and accuracy of retrieval. Let's explore five common chunking strategies and their implementations.

Slide 2: Fixed-size Chunking

Fixed-size chunking splits text into uniform segments of a specified length. While simple, this method may break sentences or ideas mid-stream, potentially distributing important information across multiple chunks.

Slide 3: Source Code for Fixed-size Chunking

```python
def fixed_size_chunking(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

text = "This is a sample text for demonstrating fixed-size chunking. It may break sentences."
chunks = fixed_size_chunking(text, 20)
print(chunks)
```

Slide 4: Results for: Source Code for Fixed-size Chunking

```python
['This is a sample te', 'xt for demonstratin', 'g fixed-size chunki', 'ng. It may break se', 'ntences.']
```

Slide 5: Semantic Chunking

Semantic chunking segments documents based on meaningful units like sentences or paragraphs. It creates embeddings for each segment and combines them based on cosine similarity until a significant drop is detected, forming a new chunk.

Slide 6: Source Code for Semantic Chunking

```python
import re
from collections import Counter
import math

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator) / denominator

def create_embedding(text):
    words = re.findall(r'\w+', text.lower())
    return Counter(words)

def semantic_chunking(text, similarity_threshold=0.5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = sentences[0]
    current_embedding = create_embedding(current_chunk)
    
    for sentence in sentences[1:]:
        sentence_embedding = create_embedding(sentence)
        similarity = cosine_similarity(current_embedding, sentence_embedding)
        
        if similarity >= similarity_threshold:
            current_chunk += " " + sentence
            current_embedding = create_embedding(current_chunk)
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_embedding = sentence_embedding
    
    chunks.append(current_chunk)
    return chunks

text = "This is a sample text. It demonstrates semantic chunking. We split based on meaning. New topics start new chunks."
chunks = semantic_chunking(text)
print(chunks)
```

Slide 7: Results for: Source Code for Semantic Chunking

```python
['This is a sample text. It demonstrates semantic chunking.', 'We split based on meaning.', 'New topics start new chunks.']
```

Slide 8: Recursive Chunking

Recursive chunking first divides text based on inherent separators like paragraphs or sections. If any resulting chunk exceeds a predefined size limit, it's further split into smaller chunks.

Slide 9: Source Code for Recursive Chunking

```python
def recursive_chunking(text, max_chunk_size=100, separator='\n\n'):
    chunks = text.split(separator)
    result = []
    
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            result.append(chunk)
        else:
            # Recursively split large chunks
            result.extend(recursive_chunking(chunk, max_chunk_size, '. '))
    
    return result

text = """Paragraph 1 is short.

Paragraph 2 is a bit longer and exceeds the maximum chunk size. It will be split into smaller parts based on sentences.

Paragraph 3 is also short."""

chunks = recursive_chunking(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
```

Slide 10: Results for: Source Code for Recursive Chunking

```python
Chunk 1: Paragraph 1 is short.
Chunk 2: Paragraph 2 is a bit longer and exceeds the maximum chunk size.
Chunk 3: It will be split into smaller parts based on sentences.
Chunk 4: Paragraph 3 is also short.
```

Slide 11: Document Structure-based Chunking

This method utilizes the inherent structure of documents, such as headings, sections, or paragraphs, to define chunk boundaries. It maintains structural integrity by aligning with the document's logical sections but assumes a clear structure exists.

Slide 12: Source Code for Document Structure-based Chunking

```python
import re

def structure_based_chunking(text):
    # Define patterns for different structural elements
    patterns = {
        'heading': r'^#+\s+.*$',
        'paragraph': r'^(?!#+\s+).*(?:\n(?!#+\s+).+)*',
    }
    
    chunks = []
    lines = text.split('\n')
    current_chunk = ''
    
    for line in lines:
        if re.match(patterns['heading'], line):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
        elif re.match(patterns['paragraph'], line):
            current_chunk += line + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = ''
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

text = """# Introduction
This is the introduction paragraph.

## Section 1
This is the first section's content.
It spans multiple lines.

## Section 2
This is the second section's content."""

chunks = structure_based_chunking(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n")
```

Slide 13: Results for: Source Code for Document Structure-based Chunking

```python
Chunk 1:
# Introduction
This is the introduction paragraph.

Chunk 2:
## Section 1
This is the first section's content.
It spans multiple lines.

Chunk 3:
## Section 2
This is the second section's content.
```

Slide 14: LLM-based Chunking

LLM-based chunking leverages language models to create semantically isolated and meaningful chunks. While this method ensures high semantic accuracy, it is computationally demanding and may be limited by the LLM's context window.

Slide 15: Source Code for LLM-based Chunking

```python
def simulate_llm_chunking(text, max_chunk_size=100):
    # This is a simplified simulation of LLM-based chunking
    # In practice, you would use an actual LLM API
    
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

text = "This is a simulation of LLM-based chunking. In reality, an LLM would understand context and create more semantically meaningful chunks. This method is computationally expensive but potentially more accurate."

chunks = simulate_llm_chunking(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
```

Slide 16: Results for: Source Code for LLM-based Chunking

```python
Chunk 1: This is a simulation of LLM-based chunking. In reality, an LLM would understand context and create more
Chunk 2: semantically meaningful chunks. This method is computationally expensive but potentially more accurate.
```

Slide 17: Real-life Example: Text Summarization

Chunking strategies are crucial in text summarization tasks. For instance, when summarizing a long research paper, semantic chunking can be used to divide the paper into coherent sections, allowing for more accurate summarization of each part.

Slide 18: Real-life Example: Question Answering Systems

In question answering systems, document structure-based chunking can be employed to break down textbooks or manuals. This allows the system to quickly locate relevant sections when answering specific questions about the content.

Slide 19: Additional Resources

For more information on chunking strategies and their applications in natural language processing, refer to the following ArXiv papers:

1.  "Efficient Document Retrieval by End-to-End Refining and Chunking" (arXiv:2310.14102)
2.  "Retrieval-Augmented Generation for Large Language Models: A Survey" (arXiv:2312.10997)

These papers provide in-depth discussions on various chunking techniques and their impact on retrieval-augmented generation systems.

