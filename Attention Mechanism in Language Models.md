## Attention Mechanism in Language Models
Slide 1: Understanding Attention Mathematics

The attention mechanism calculates weighted importance scores between input elements using queries, keys, and values. These weights determine how much focus each element should receive, enabling the model to identify relevant relationships in sequences.

```python
import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implementation of scaled dot-product attention
    
    Parameters:
        query: shape (batch_size, seq_len_q, depth)
        key: shape (batch_size, seq_len_k, depth)
        value: shape (batch_size, seq_len_v, depth)
        mask: Optional mask for padding
    """
    # Calculate attention scores
    matmul_qk = np.matmul(query, key.transpose(-2, -1))
    depth = key.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(depth)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
query = np.random.rand(1, 3, 4)  # (batch_size, seq_len_q, depth)
key = np.random.rand(1, 3, 4)    # (batch_size, seq_len_k, depth)
value = np.random.rand(1, 3, 4)  # (batch_size, seq_len_v, depth)

output, weights = scaled_dot_product_attention(query, key, value)
```

Slide 2: Self-Attention Implementation

Self-attention allows each position in a sequence to attend to all positions, creating a rich understanding of context. This implementation demonstrates the core mathematics behind self-attention using tensor operations in Python.

```python
class SelfAttention:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.W_query = np.random.randn(embedding_dim, embedding_dim)
        self.W_key = np.random.randn(embedding_dim, embedding_dim)
        self.W_value = np.random.randn(embedding_dim, embedding_dim)
        
    def forward(self, X):
        """
        X: Input tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Project input to Q, K, V
        Q = np.dot(X, self.W_query)
        K = np.dot(X, self.W_key)
        V = np.dot(X, self.W_value)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        scores = scores / np.sqrt(self.embedding_dim)
        
        # Apply softmax
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Get weighted sum
        output = np.matmul(attention_weights, V)
        return output

# Example usage
batch_size, seq_len, embedding_dim = 2, 4, 8
X = np.random.randn(batch_size, seq_len, embedding_dim)
attention = SelfAttention(embedding_dim)
output = attention.forward(X)
```

Slide 3: Multi-Head Attention

Multi-head attention extends single-head attention by allowing the model to focus on different aspects of the input simultaneously. Each head learns distinct representation subspaces, enhancing the model's ability to capture various types of relationships.

```python
class MultiHeadAttention:
    def __init__(self, embedding_dim, num_heads):
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim
        
        # Initialize weights for each head
        self.W_query = np.random.randn(num_heads, embedding_dim, self.head_dim)
        self.W_key = np.random.randn(num_heads, embedding_dim, self.head_dim)
        self.W_value = np.random.randn(num_heads, embedding_dim, self.head_dim)
        self.W_output = np.random.randn(num_heads * self.head_dim, embedding_dim)
        
    def forward(self, X):
        batch_size, seq_len = X.shape[0], X.shape[1]
        
        # Project input for each head
        Q = np.stack([np.dot(X, W_q) for W_q in self.W_query])
        K = np.stack([np.dot(X, W_k) for W_k in self.W_key])
        V = np.stack([np.dot(X, W_v) for W_v in self.W_value])
        
        # Compute attention scores for each head
        scores = np.matmul(Q, np.transpose(K, (0, 1, 3, 2)))
        scaled_scores = scores / np.sqrt(self.head_dim)
        
        # Apply softmax
        attention_weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=-1, keepdims=True)
        
        # Compute output for each head
        head_outputs = np.matmul(attention_weights, V)
        
        # Concatenate and project heads
        concat_output = head_outputs.transpose(1, 2, 0, 3).reshape(batch_size, seq_len, -1)
        final_output = np.dot(concat_output, self.W_output)
        
        return final_output

# Example usage
batch_size, seq_len, embedding_dim = 2, 4, 8
num_heads = 2
X = np.random.randn(batch_size, seq_len, embedding_dim)
mha = MultiHeadAttention(embedding_dim, num_heads)
output = mha.forward(X)
```

Slide 4: Position-Aware Attention

Position-aware attention incorporates positional information into the attention mechanism, enabling the model to understand sequence order. This implementation demonstrates how positional encodings are combined with input embeddings.

```python
import numpy as np

def positional_encoding(seq_length, d_model):
    """
    Generate positional encodings for attention mechanism
    seq_length: length of input sequence
    d_model: dimension of the model
    """
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

class PositionAwareAttention:
    def __init__(self, d_model, seq_length):
        self.d_model = d_model
        self.pos_encoding = positional_encoding(seq_length, d_model)
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
    
    def forward(self, X):
        # Add positional encoding to input
        X_pos = X + self.pos_encoding
        
        # Compute Q, K, V with positional information
        Q = np.dot(X_pos, self.W_q)
        K = np.dot(X_pos, self.W_k)
        V = np.dot(X_pos, self.W_v)
        
        # Compute attention scores
        scores = np.matmul(Q, K.T) / np.sqrt(self.d_model)
        attention = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        return np.matmul(attention, V)

# Example usage
seq_length, d_model = 10, 16
X = np.random.randn(seq_length, d_model)
pos_attention = PositionAwareAttention(d_model, seq_length)
output = pos_attention.forward(X)
```

Slide 5: Masked Self-Attention

Masked self-attention prevents positions from attending to subsequent positions, crucial for autoregressive tasks. This implementation shows how masking is applied during the attention computation process.

```python
class MaskedSelfAttention:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
    
    def create_mask(self, seq_length):
        """Create causal mask to prevent attending to future tokens"""
        mask = np.triu(np.ones((seq_length, seq_length)), k=1)
        return mask * -1e9
    
    def forward(self, X):
        seq_length = X.shape[1]
        mask = self.create_mask(seq_length)
        
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Apply mask before softmax
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.embedding_dim)
        scores += mask
        
        attention_weights = np.exp(scores) / (np.sum(np.exp(scores), axis=-1, keepdims=True) + 1e-9)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights

# Example usage
batch_size, seq_length, embedding_dim = 2, 8, 16
X = np.random.randn(batch_size, seq_length, embedding_dim)
masked_attention = MaskedSelfAttention(embedding_dim)
output, weights = masked_attention.forward(X)

# Visualize attention pattern
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(weights[0], cmap='viridis')
plt.colorbar()
plt.title('Masked Attention Pattern')
plt.close()
```

Slide 6: Relative Position Attention

Relative position attention considers the relative distances between tokens rather than absolute positions. This implementation demonstrates how to compute attention scores using relative position representations.

```python
class RelativePositionAttention:
    def __init__(self, embedding_dim, max_relative_position=32):
        self.embedding_dim = embedding_dim
        self.max_relative_position = max_relative_position
        
        # Initialize relative position embeddings
        self.relative_embeddings = np.random.randn(
            2 * max_relative_position + 1,
            embedding_dim
        )
    
    def _get_relative_positions(self, length):
        """Generate matrix of relative positions between all pairs of positions"""
        range_vec = np.arange(length)
        range_mat = range_vec[:, None] - range_vec[None, :]
        
        # Clip relative positions to max_relative_position
        range_mat = np.clip(
            range_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift values to be non-negative
        return range_mat + self.max_relative_position
    
    def forward(self, X):
        seq_length = X.shape[1]
        relative_positions = self._get_relative_positions(seq_length)
        
        # Get relative position embeddings for each position pair
        relative_pos_embeddings = self.relative_embeddings[relative_positions]
        
        # Compute attention scores with relative position information
        Q = np.dot(X, np.random.randn(self.embedding_dim, self.embedding_dim))
        K = np.dot(X, np.random.randn(self.embedding_dim, self.embedding_dim))
        V = np.dot(X, np.random.randn(self.embedding_dim, self.embedding_dim))
        
        # Combine content-based and position-based attention
        content_scores = np.matmul(Q, K.transpose(0, 2, 1))
        position_scores = np.matmul(Q, relative_pos_embeddings.transpose(0, 2, 1))
        
        scores = (content_scores + position_scores) / np.sqrt(self.embedding_dim)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        return np.matmul(attention_weights, V)

# Example usage
batch_size, seq_length, embedding_dim = 2, 16, 32
X = np.random.randn(batch_size, seq_length, embedding_dim)
rel_attention = RelativePositionAttention(embedding_dim)
output = rel_attention.forward(X)
```

Slide 7: Attention Visualization Tools

Implementation of visualization tools for attention mechanisms helps understand how the model processes information. This code provides functions to visualize attention weights and patterns across different heads and layers.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self):
        self.figsize = (12, 8)
        
    def plot_attention_weights(self, attention_weights, tokens=None):
        """
        Visualize attention weights with optional token labels
        attention_weights: shape (seq_len, seq_len)
        tokens: optional list of token labels
        """
        plt.figure(figsize=self.figsize)
        
        if tokens is None:
            tokens = [f'Token_{i}' for i in range(attention_weights.shape[0])]
            
        sns.heatmap(attention_weights, 
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap='viridis',
                    annot=True,
                    fmt='.2f')
        
        plt.title('Attention Weights Visualization')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        return plt.gcf()
    
    def plot_multi_head_attention(self, attention_weights, num_heads):
        """
        Visualize attention patterns across multiple heads
        attention_weights: shape (num_heads, seq_len, seq_len)
        """
        fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx in range(num_heads):
            sns.heatmap(attention_weights[idx], 
                       cmap='viridis',
                       ax=axes[idx],
                       cbar=False)
            axes[idx].set_title(f'Head {idx+1}')
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            
        plt.tight_layout()
        return fig

# Example usage
def generate_sample_attention():
    seq_len = 8
    num_heads = 4
    
    # Generate sample attention weights
    single_head = np.random.uniform(0, 1, (seq_len, seq_len))
    single_head = single_head / single_head.sum(axis=-1, keepdims=True)
    
    multi_head = np.random.uniform(0, 1, (num_heads, seq_len, seq_len))
    multi_head = multi_head / multi_head.sum(axis=-1, keepdims=True)[..., None]
    
    return single_head, multi_head

# Create visualizations
visualizer = AttentionVisualizer()
single_head_weights, multi_head_weights = generate_sample_attention()

# Sample tokens for visualization
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.', '<END>']

# Plot single head attention
single_head_fig = visualizer.plot_attention_weights(
    single_head_weights, 
    tokens=tokens
)

# Plot multi-head attention
multi_head_fig = visualizer.plot_multi_head_attention(
    multi_head_weights, 
    num_heads=4
)
```

Slide 8: Efficient Attention Implementation

This implementation focuses on memory-efficient attention computation using chunked processing and sparse attention patterns, suitable for handling long sequences with limited computational resources.

```python
class EfficientAttention:
    def __init__(self, embedding_dim, chunk_size=128):
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
    
    def chunked_attention(self, Q, K, V):
        """
        Compute attention scores in chunks to save memory
        """
        seq_len = Q.shape[1]
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        output_chunks = []
        
        for i in range(num_chunks):
            chunk_start = i * self.chunk_size
            chunk_end = min((i + 1) * self.chunk_size, seq_len)
            
            # Process current chunk
            Q_chunk = Q[:, chunk_start:chunk_end, :]
            
            chunk_scores = []
            chunk_weights = []
            
            for j in range(num_chunks):
                k_start = j * self.chunk_size
                k_end = min((j + 1) * self.chunk_size, seq_len)
                
                K_chunk = K[:, k_start:k_end, :]
                V_chunk = V[:, k_start:k_end, :]
                
                # Compute attention scores for current chunks
                scores = np.matmul(Q_chunk, K_chunk.transpose(0, 2, 1))
                scores = scores / np.sqrt(self.embedding_dim)
                chunk_scores.append(scores)
            
            # Concatenate and compute softmax across all chunks
            scores_cat = np.concatenate(chunk_scores, axis=-1)
            attention_weights = np.exp(scores_cat)
            attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
            
            # Compute weighted sum for current chunk
            output_chunk = np.zeros_like(Q_chunk)
            
            for j in range(num_chunks):
                k_start = j * self.chunk_size
                k_end = min((j + 1) * self.chunk_size, seq_len)
                
                V_chunk = V[:, k_start:k_end, :]
                weights_chunk = attention_weights[:, :, k_start:k_end]
                
                output_chunk += np.matmul(weights_chunk, V_chunk)
            
            output_chunks.append(output_chunk)
        
        # Concatenate all chunks
        return np.concatenate(output_chunks, axis=1)
    
    def forward(self, X):
        batch_size = X.shape[0]
        
        # Project inputs
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Compute attention in chunks
        output = self.chunked_attention(Q, K, V)
        
        return output

# Example usage
batch_size, seq_length, embedding_dim = 2, 512, 64
X = np.random.randn(batch_size, seq_length, embedding_dim)
efficient_attention = EfficientAttention(embedding_dim)
output = efficient_attention.forward(X)
```

Slide 9: Cross-Attention Implementation

Cross-attention enables interaction between two different sequences, essential for tasks like translation. This implementation shows how to compute attention between encoder and decoder sequences efficiently.

```python
class CrossAttention:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        # Initialize weights for query, key, and value transformations
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
        
    def forward(self, decoder_state, encoder_output, encoder_mask=None):
        """
        decoder_state: shape (batch_size, target_seq_len, embedding_dim)
        encoder_output: shape (batch_size, source_seq_len, embedding_dim)
        encoder_mask: shape (batch_size, target_seq_len, source_seq_len)
        """
        # Project decoder state to queries
        Q = np.dot(decoder_state, self.W_q)
        
        # Project encoder output to keys and values
        K = np.dot(encoder_output, self.W_k)
        V = np.dot(encoder_output, self.W_v)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1))
        scaled_scores = scores / np.sqrt(self.embedding_dim)
        
        # Apply encoder mask if provided
        if encoder_mask is not None:
            scaled_scores += (encoder_mask * -1e9)
        
        # Compute attention weights
        attention_weights = np.exp(scaled_scores)
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Compute weighted sum of values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights

# Example usage with mask creation
def create_padding_mask(seq_length, valid_lengths):
    """Create mask for padded sequences"""
    mask = np.zeros((len(valid_lengths), seq_length))
    for i, length in enumerate(valid_lengths):
        mask[i, length:] = 1
    return mask

# Test the implementation
batch_size = 2
source_seq_len = 10
target_seq_len = 8
embedding_dim = 16

# Sample data
decoder_state = np.random.randn(batch_size, target_seq_len, embedding_dim)
encoder_output = np.random.randn(batch_size, source_seq_len, embedding_dim)
valid_lengths = [8, 7]  # Example of variable sequence lengths

# Create padding mask
padding_mask = create_padding_mask(source_seq_len, valid_lengths)
encoder_mask = np.expand_dims(padding_mask, 1)  # Add broadcast dimension

# Initialize and apply cross-attention
cross_attention = CrossAttention(embedding_dim)
output, weights = cross_attention.forward(decoder_state, encoder_output, encoder_mask)

# Print attention pattern for first sequence
print("Attention pattern shape:", weights.shape)
print("Output shape:", output.shape)
```

Slide 10: Memory-Efficient Attention with Linear Complexity

This implementation demonstrates how to reduce the quadratic memory complexity of standard attention to linear complexity using kernel approximation techniques.

```python
class LinearAttention:
    def __init__(self, embedding_dim, num_features=256):
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
        
    def kernel_feature_map(self, x):
        """
        Random Fourier feature approximation for the softmax kernel
        """
        projection = np.random.normal(0, 1, (self.embedding_dim, self.num_features))
        features = np.cos(np.dot(x, projection) / np.sqrt(self.num_features))
        return features
    
    def forward(self, X):
        batch_size, seq_length = X.shape[0], X.shape[1]
        
        # Project inputs
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Apply kernel feature map
        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)
        
        # Compute linear attention
        KV = np.matmul(K_features.transpose(0, 2, 1), V)
        QKV = np.matmul(Q_features, KV)
        
        # Normalize
        normalizer = np.matmul(Q_features, 
                              np.sum(K_features, axis=1, keepdims=True).transpose(0, 2, 1))
        output = QKV / (normalizer + 1e-9)
        
        return output

# Benchmark comparison with standard attention
def benchmark_attention(seq_length, embedding_dim):
    X = np.random.randn(1, seq_length, embedding_dim)
    
    # Linear attention
    linear_attention = LinearAttention(embedding_dim)
    start_time = time.time()
    linear_output = linear_attention.forward(X)
    linear_time = time.time() - start_time
    
    # Memory usage estimation
    linear_memory = seq_length * embedding_dim * 4  # Approximate memory in bytes
    
    print(f"Sequence length: {seq_length}")
    print(f"Linear attention time: {linear_time:.4f}s")
    print(f"Approximate memory usage: {linear_memory/1024/1024:.2f}MB")

# Test with different sequence lengths
for seq_length in [1000, 2000, 4000]:
    print(f"\nBenchmarking with sequence length {seq_length}")
    benchmark_attention(seq_length, embedding_dim=64)
```

Slide 11: Attention with Sparse Transformers

This implementation demonstrates sparse attention patterns that reduce computational complexity while maintaining model performance through structured sparsity in the attention mechanism.

```python
import numpy as np
from scipy.sparse import csr_matrix

class SparseAttention:
    def __init__(self, embedding_dim, sparsity_factor=4):
        self.embedding_dim = embedding_dim
        self.sparsity_factor = sparsity_factor
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
    
    def create_sparse_mask(self, seq_length):
        """
        Creates a strided sparse attention pattern
        Only attend to every sparsity_factor-th position
        """
        mask = np.zeros((seq_length, seq_length))
        for i in range(seq_length):
            # Local attention window
            start_idx = max(0, i - self.sparsity_factor)
            end_idx = min(seq_length, i + self.sparsity_factor + 1)
            mask[i, start_idx:end_idx] = 1
            
            # Strided attention
            stride_indices = np.arange(0, seq_length, self.sparsity_factor)
            mask[i, stride_indices] = 1
        
        return csr_matrix(mask)
    
    def forward(self, X):
        batch_size, seq_length = X.shape[0], X.shape[1]
        
        # Create sparse attention mask
        sparse_mask = self.create_sparse_mask(seq_length)
        
        # Project inputs
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        outputs = []
        for b in range(batch_size):
            # Compute sparse attention scores
            scores = np.zeros((seq_length, seq_length))
            Q_b, K_b = Q[b], K[b]
            
            # Only compute attention for non-zero elements in mask
            for i in range(seq_length):
                mask_row = sparse_mask[i].indices
                scores[i, mask_row] = np.dot(Q_b[i], K_b[mask_row].T)
            
            # Scale scores
            scores = scores / np.sqrt(self.embedding_dim)
            
            # Apply sparse softmax
            exp_scores = np.exp(scores) * sparse_mask.toarray()
            attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
            
            # Compute output
            output = np.dot(attention_weights, V[b])
            outputs.append(output)
        
        return np.stack(outputs)

# Example usage and performance analysis
def analyze_sparse_attention():
    seq_length = 1024
    embedding_dim = 64
    batch_size = 2
    
    # Initialize data
    X = np.random.randn(batch_size, seq_length, embedding_dim)
    
    # Create and test sparse attention
    sparse_attn = SparseAttention(embedding_dim)
    output = sparse_attn.forward(X)
    
    # Analyze sparsity
    mask = sparse_attn.create_sparse_mask(seq_length)
    sparsity = 1.0 - (mask.nnz / (seq_length * seq_length))
    
    print(f"Sequence length: {seq_length}")
    print(f"Attention matrix sparsity: {sparsity:.2%}")
    print(f"Output shape: {output.shape}")
    
    return output, sparsity

# Run analysis
output, sparsity = analyze_sparse_attention()
```

Slide 12: Real-world Application - Document QA System

Implementation of an attention-based document question-answering system that processes long documents efficiently by using hierarchical attention mechanisms.

```python
class DocumentQASystem:
    def __init__(self, embedding_dim, max_doc_length=1000):
        self.embedding_dim = embedding_dim
        self.max_doc_length = max_doc_length
        
        # Initialize document and question encoders
        self.doc_attention = MultiHeadAttention(embedding_dim, num_heads=8)
        self.cross_attention = CrossAttention(embedding_dim)
        
        # Document chunking parameters
        self.chunk_size = 200
        self.stride = 100
    
    def chunk_document(self, document_embeddings):
        """Split document into overlapping chunks"""
        chunks = []
        for i in range(0, len(document_embeddings), self.stride):
            chunk = document_embeddings[i:i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                # Pad last chunk if necessary
                pad_length = self.chunk_size - len(chunk)
                chunk = np.pad(chunk, ((0, pad_length), (0, 0)))
            chunks.append(chunk)
        return np.stack(chunks)
    
    def forward(self, document_embeddings, question_embeddings):
        # Chunk document for efficient processing
        doc_chunks = self.chunk_document(document_embeddings)
        batch_size, num_chunks = doc_chunks.shape[0], doc_chunks.shape[1]
        
        # Process document chunks
        chunk_encodings = []
        for i in range(num_chunks):
            chunk_encoding = self.doc_attention.forward(doc_chunks[:, i])
            chunk_encodings.append(chunk_encoding)
        
        # Combine chunk encodings
        doc_encoding = np.mean(chunk_encodings, axis=0)
        
        # Cross-attention between question and document
        answer_logits, attention_weights = self.cross_attention.forward(
            question_embeddings,
            doc_encoding
        )
        
        return answer_logits, attention_weights

# Example usage
def test_qa_system():
    # Setup parameters
    embedding_dim = 256
    doc_length = 800
    question_length = 20
    
    # Create sample inputs
    document = np.random.randn(1, doc_length, embedding_dim)
    question = np.random.randn(1, question_length, embedding_dim)
    
    # Initialize and run QA system
    qa_system = DocumentQASystem(embedding_dim)
    answer_logits, attention_weights = qa_system.forward(document, question)
    
    print(f"Document shape: {document.shape}")
    print(f"Question shape: {question.shape}")
    print(f"Answer logits shape: {answer_logits.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

# Run test
test_qa_system()
```

Slide 13: Attention with Local and Global Context

This implementation combines local and global attention patterns to capture both fine-grained and high-level relationships in sequences efficiently.

```python
class LocalGlobalAttention:
    def __init__(self, embedding_dim, local_window=32):
        self.embedding_dim = embedding_dim
        self.local_window = local_window
        self.W_q = np.random.randn(embedding_dim, embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim)
        
        # Global tokens initialization
        self.num_global_tokens = 8
        self.global_tokens = np.random.randn(1, self.num_global_tokens, embedding_dim)
        
    def compute_local_attention(self, Q, K, V):
        batch_size, seq_length = Q.shape[0], Q.shape[1]
        
        # Create local attention mask
        local_mask = np.zeros((seq_length, seq_length))
        for i in range(seq_length):
            window_start = max(0, i - self.local_window)
            window_end = min(seq_length, i + self.local_window + 1)
            local_mask[i, window_start:window_end] = 1
            
        # Compute local attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.embedding_dim)
        scores = scores * local_mask + (-1e9) * (1 - local_mask)
        
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        local_output = np.matmul(weights, V)
        
        return local_output
    
    def compute_global_attention(self, X, global_tokens):
        # Update global tokens through attention
        Q_global = np.dot(global_tokens, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        scores = np.matmul(Q_global, K.transpose(0, 2, 1)) / np.sqrt(self.embedding_dim)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        updated_global = np.matmul(weights, V)
        return updated_global
    
    def forward(self, X):
        batch_size = X.shape[0]
        global_tokens = np.repeat(self.global_tokens, batch_size, axis=0)
        
        # Project inputs
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Compute local attention
        local_output = self.compute_local_attention(Q, K, V)
        
        # Update global tokens
        updated_global = self.compute_global_attention(X, global_tokens)
        
        # Combine local and global attention
        Q_combined = np.dot(X, self.W_q)
        global_scores = np.matmul(Q_combined, updated_global.transpose(0, 2, 1))
        global_weights = np.exp(global_scores) / np.sum(np.exp(global_scores), axis=-1, keepdims=True)
        global_context = np.matmul(global_weights, updated_global)
        
        # Final output combines local and global information
        output = local_output + global_context
        
        return output, (local_output, global_context)

# Example usage and testing
def test_local_global_attention():
    embedding_dim = 64
    seq_length = 256
    batch_size = 2
    
    # Create test input
    X = np.random.randn(batch_size, seq_length, embedding_dim)
    
    # Initialize and apply attention
    lg_attention = LocalGlobalAttention(embedding_dim)
    output, (local_out, global_ctx) = lg_attention.forward(X)
    
    # Analyze results
    print("Input shape:", X.shape)
    print("Output shape:", output.shape)
    print("Local output shape:", local_out.shape)
    print("Global context shape:", global_ctx.shape)
    
    # Compute attention statistics
    local_influence = np.mean(np.abs(local_out))
    global_influence = np.mean(np.abs(global_ctx))
    
    print(f"\nAttention Statistics:")
    print(f"Average local influence: {local_influence:.4f}")
    print(f"Average global influence: {global_influence:.4f}")
    print(f"Local/Global ratio: {local_influence/global_influence:.4f}")

# Run test
test_local_global_attention()
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Efficient Transformers: A Survey" - [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
*   "Longformer: The Long-Document Transformer" - [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
*   "Reformer: The Efficient Transformer" - [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)
*   "Linear Transformers Are Secretly Fast Weight Memory Systems" - [https://arxiv.org/abs/2102.11174](https://arxiv.org/abs/2102.11174)
*   "Sparse Sinkhorn Attention" - [https://arxiv.org/abs/2002.11296](https://arxiv.org/abs/2002.11296)

