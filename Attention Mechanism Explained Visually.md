## Attention Mechanism Explained Visually
Slide 1: Attention Mechanism Fundamentals

The attention mechanism revolutionizes sequence processing by enabling models to focus on relevant parts of input data dynamically. It calculates importance scores between elements, allowing the model to weigh different parts of the input differently when producing outputs.

```python
import numpy as np

def simple_attention(query, keys, values):
    # Calculate attention scores using dot product
    scores = np.dot(query, keys.T)
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores))
    
    # Weighted sum of values
    output = np.dot(weights, values)
    return output, weights

# Example usage
query = np.array([0.1, 0.2, 0.3])
keys = np.array([[0.4, 0.5, 0.6],
                 [0.7, 0.8, 0.9]])
values = np.array([[1.0, 1.1],
                  [1.2, 1.3]])

output, attention_weights = simple_attention(query, keys, values)
print(f"Attention weights: {attention_weights}")
print(f"Output: {output}")
```

Slide 2: Self-Attention Mathematics

Self-attention computation involves three main components: queries, keys, and values. The attention weights are computed using scaled dot-product attention, followed by softmax normalization to ensure weights sum to 1.

```python
# Mathematical formula for attention mechanism
"""
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
$$d_k$$ is the dimension of keys
$$Q$$ represents queries
$$K$$ represents keys
$$V$$ represents values
"""
```

Slide 3: Implementing Scaled Dot-Product Attention

This implementation demonstrates the core mathematics behind scaled dot-product attention, including the scaling factor that prevents gradient issues in deeper networks. The scaling factor helps maintain stable gradients during training.

```python
import numpy as np

def scaled_dot_product_attention(queries, keys, values, mask=None):
    # Get dimensionality of keys
    d_k = keys.shape[-1]
    
    # Compute scaled attention scores
    attention_scores = np.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        attention_scores += (mask * -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    
    # Compute output as weighted sum of values
    output = np.matmul(attention_weights, values)
    
    return output, attention_weights
```

Slide 4: Multi-Head Attention Architecture

Multi-head attention allows the model to attend to information from different representation subspaces simultaneously. Each head learns different aspects of the input sequence, enabling richer feature extraction and better model performance.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = np.random.randn(d_model, d_model)
        self.wk = np.random.randn(d_model, d_model)
        self.wv = np.random.randn(d_model, d_model)
        
        self.dense = np.random.randn(d_model, d_model)
```

Slide 5: Multi-Head Attention Implementation

```python
def split_heads(self, x, batch_size):
    x = np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return np.transpose(x, (0, 2, 1, 3))

def call(self, queries, keys, values, mask=None):
    batch_size = queries.shape[0]
    
    # Linear layers
    q = np.dot(queries, self.wq)
    k = np.dot(keys, self.wk)
    v = np.dot(values, self.wv)
    
    # Split heads
    q = self.split_heads(q, batch_size)
    k = self.split_heads(k, batch_size)
    v = self.split_heads(v, batch_size)
    
    # Scaled dot-product attention
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    # Reshape and apply final linear layer
    output = np.dot(scaled_attention, self.dense)
    
    return output, attention_weights
```

Slide 6: Positional Encoding

Positional encoding adds information about token positions in the sequence, enabling the attention mechanism to consider sequential order. This is crucial since attention operations are inherently position-independent.

```python
def get_positional_encoding(sequence_length, d_model):
    angles = np.arange(sequence_length)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    
    # Apply sin to even indices
    sines = np.sin(angles[:, 0::2])
    
    # Apply cos to odd indices
    cosines = np.cos(angles[:, 1::2])
    
    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    
    return pos_encoding
```

Slide 7: Real-World Example - Text Classification

Implementing attention mechanism for sentiment analysis using a custom dataset. This implementation showcases how attention weights help identify important words in sentences for classification tasks.

```python
import numpy as np
from sklearn.model_selection import train_test_split

class TextClassificationAttention:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim)
        self.attention_weights = np.random.randn(embedding_dim, 1)
        
    def forward(self, input_sequence):
        # Convert input tokens to embeddings
        embeddings = self.embedding_matrix[input_sequence]
        
        # Calculate attention scores
        attention_scores = np.tanh(np.dot(embeddings, self.attention_weights))
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
        
        # Apply attention weights
        context_vector = np.sum(embeddings * attention_weights, axis=1)
        return context_vector, attention_weights

# Example usage
classifier = TextClassificationAttention(vocab_size=10000, embedding_dim=100)
sample_sequence = np.array([1, 45, 232, 876, 23])
context, weights = classifier.forward(sample_sequence)
```

Slide 8: Attention for Machine Translation

A practical implementation of attention mechanism for neural machine translation, demonstrating how attention helps align words between source and target languages during translation.

```python
class TranslationAttention:
    def __init__(self, source_vocab_size, target_vocab_size, hidden_dim):
        self.encoder_embedding = np.random.randn(source_vocab_size, hidden_dim)
        self.decoder_embedding = np.random.randn(target_vocab_size, hidden_dim)
        self.attention_matrix = np.random.randn(hidden_dim, hidden_dim)
        
    def compute_attention(self, encoder_states, decoder_state):
        # Project decoder state
        projected_decoder = np.dot(decoder_state, self.attention_matrix)
        
        # Calculate alignment scores
        alignment_scores = np.dot(encoder_states, projected_decoder.T)
        
        # Normalize scores
        attention_weights = np.exp(alignment_scores) / np.sum(np.exp(alignment_scores))
        
        # Calculate context vector
        context = np.dot(attention_weights.T, encoder_states)
        return context, attention_weights

# Example translation input
source_sentence = np.array([12, 45, 67, 89, 34])
target_state = np.random.randn(1, 100)
translator = TranslationAttention(10000, 8000, 100)
```

Slide 9: Self-Attention in Transformers

Understanding self-attention implementation in transformer architecture, which forms the foundation for modern language models. This code demonstrates the parallel computation of attention across all positions.

```python
class TransformerSelfAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Initialize projection matrices
        self.q_proj = np.random.randn(d_model, d_model)
        self.k_proj = np.random.randn(d_model, d_model)
        self.v_proj = np.random.randn(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project inputs to Q, K, V
        Q = np.dot(x, self.q_proj).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = np.dot(x, self.k_proj).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = np.dot(x, self.v_proj).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Apply attention to values
        attended_values = np.matmul(attention_weights, V)
        return attended_values, attention_weights
```

Slide 10: Performance Metrics Implementation

This implementation shows how to evaluate attention-based models using various metrics, including attention weight analysis and prediction accuracy.

```python
def evaluate_attention_model(true_labels, predictions, attention_weights):
    # Calculate classification metrics
    accuracy = np.mean(predictions == true_labels)
    
    # Analyze attention distribution
    attention_entropy = -np.sum(
        attention_weights * np.log(attention_weights + 1e-9),
        axis=-1
    )
    
    # Calculate attention concentration
    attention_concentration = np.max(attention_weights, axis=-1)
    
    return {
        'accuracy': accuracy,
        'attention_entropy': attention_entropy.mean(),
        'attention_concentration': attention_concentration.mean()
    }

# Example usage
true_labels = np.array([1, 0, 1, 1, 0])
predictions = np.array([1, 0, 1, 0, 0])
attention_weights = np.random.rand(5, 10)  # 5 samples, 10 attention weights each
attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

metrics = evaluate_attention_model(true_labels, predictions, attention_weights)
print(f"Model Performance Metrics:\n{metrics}")
```

Slide 11: Attention Visualization Tools

Implementing visualization tools for attention weights helps understand model behavior and debug attention patterns. This implementation provides functions to generate attention heatmaps and analyze cross-attention patterns.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention_weights(attention_weights, source_tokens, target_tokens=None):
    plt.figure(figsize=(10, 8))
    if target_tokens is None:
        # Self-attention visualization
        plt.imshow(attention_weights, cmap='viridis')
        plt.xticks(range(len(source_tokens)), source_tokens, rotation=45)
        plt.yticks(range(len(source_tokens)), source_tokens)
    else:
        # Cross-attention visualization
        plt.imshow(attention_weights, cmap='viridis')
        plt.xticks(range(len(source_tokens)), source_tokens, rotation=45)
        plt.yticks(range(len(target_tokens)), target_tokens)
    
    plt.colorbar()
    
    # Example usage
    source = ["The", "cat", "sat", "on", "mat"]
    attention_matrix = np.random.rand(5, 5)
    attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
    
    visualize_attention_weights(attention_matrix, source)
    plt.title("Self-Attention Visualization")
```

Slide 12: Optimizing Attention Computation

Advanced implementation focusing on memory-efficient attention computation, particularly useful for processing long sequences. This implementation uses chunked attention to reduce memory requirements.

```python
def chunked_attention(queries, keys, values, chunk_size=128):
    batch_size, seq_len, dim = queries.shape
    outputs = np.zeros((batch_size, seq_len, dim))
    
    for i in range(0, seq_len, chunk_size):
        chunk_end = min(i + chunk_size, seq_len)
        
        # Process attention in chunks
        q_chunk = queries[:, i:chunk_end, :]
        
        # Calculate attention scores for current chunk
        scores = np.matmul(q_chunk, keys.transpose(0, 2, 1))
        scores = scores / np.sqrt(dim)
        
        # Apply softmax
        attention_weights = np.exp(scores)
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Compute chunk output
        chunk_output = np.matmul(attention_weights, values)
        outputs[:, i:chunk_end, :] = chunk_output
    
    return outputs

# Example usage
batch_size, seq_len, dim = 2, 512, 64
queries = np.random.randn(batch_size, seq_len, dim)
keys = np.random.randn(batch_size, seq_len, dim)
values = np.random.randn(batch_size, seq_len, dim)

efficient_output = chunked_attention(queries, keys, values)
```

Slide 13: Real-World Application - Document Summarization

Implementation of an attention-based document summarization system that identifies and extracts key sentences from long documents using hierarchical attention.

```python
class DocumentSummarizer:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.word_attention = np.random.randn(embedding_dim, hidden_dim)
        self.sentence_attention = np.random.randn(hidden_dim, 1)
        self.embedding = np.random.randn(vocab_size, embedding_dim)
    
    def compute_importance_scores(self, document):
        # document shape: (num_sentences, words_per_sentence)
        word_embeddings = self.embedding[document]
        
        # Word-level attention
        word_scores = np.tanh(np.dot(word_embeddings, self.word_attention))
        word_weights = np.exp(word_scores) / np.sum(np.exp(word_scores), axis=-1, keepdims=True)
        
        # Sentence representations
        sentence_vectors = np.sum(word_embeddings * word_weights[..., np.newaxis], axis=1)
        
        # Sentence-level attention
        sentence_scores = np.tanh(np.dot(sentence_vectors, self.sentence_attention))
        sentence_weights = np.exp(sentence_scores) / np.sum(np.exp(sentence_scores))
        
        return sentence_weights, word_weights

# Example usage
document = np.random.randint(0, 1000, size=(5, 20))  # 5 sentences, 20 words each
summarizer = DocumentSummarizer(vocab_size=1000, embedding_dim=100, hidden_dim=50)
sentence_importance, word_importance = summarizer.compute_importance_scores(document)
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Neural Machine Translation by Jointly Learning to Align and Translate" - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
*   "Effective Approaches to Attention-based Neural Machine Translation" - [https://arxiv.org/abs/1508.04025](https://arxiv.org/abs/1508.04025)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" - [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044)
*   Search keywords: "attention mechanism deep learning", "transformer architecture", "self-attention neural networks"

