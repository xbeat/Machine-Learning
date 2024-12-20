## Attention Mechanism in Deep Learning
Slide 1: Understanding Attention Mechanism Fundamentals

The attention mechanism revolutionizes how neural networks process sequential data by implementing a dynamic weighting system. It enables models to selectively focus on different parts of the input sequence when generating each element of the output sequence, similar to how humans pay attention to specific details.

```python
import numpy as np

def attention_score(query, key):
    """
    Implements basic attention scoring mechanism
    query: shape (query_len, d_k)
    key: shape (key_len, d_k)
    """
    # Compute dot product attention
    scores = np.dot(query, key.T)
    
    # Scale by sqrt(d_k) to prevent exploding gradients
    d_k = query.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)
    
    # Apply softmax for probability distribution
    attention_weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=-1, keepdims=True)
    
    return attention_weights

# Example usage
query = np.random.randn(4, 8)  # 4 queries, dimension 8
key = np.random.randn(6, 8)    # 6 keys, dimension 8
weights = attention_score(query, key)
print("Attention Weights Shape:", weights.shape)
print("Sample Weights:\n", weights[0])  # Weights for first query
```

Slide 2: Implementing Self-Attention Layer

Self-attention allows a sequence to attend to itself, capturing relationships between all positions. This implementation demonstrates the core mathematical operations behind self-attention, including the query, key, and value transformations that form the foundation of modern attention mechanisms.

```python
class SelfAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        # Initialize transformation matrices
        self.W_q = np.random.randn(embed_dim, embed_dim)
        self.W_k = np.random.randn(embed_dim, embed_dim)
        self.W_v = np.random.randn(embed_dim, embed_dim)
    
    def forward(self, X):
        """
        X: Input sequence (batch_size, seq_len, embed_dim)
        """
        # Generate Q, K, V matrices
        Q = np.dot(X, self.W_q)  # Query
        K = np.dot(X, self.W_k)  # Key
        V = np.dot(X, self.W_v)  # Value
        
        # Compute attention scores
        scores = np.dot(Q, K.transpose(0, 2, 1))
        scores = scores / np.sqrt(self.embed_dim)
        
        # Apply softmax
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Compute weighted sum
        output = np.dot(attention_weights, V)
        return output, attention_weights

# Example usage
batch_size, seq_len, embed_dim = 2, 4, 8
X = np.random.randn(batch_size, seq_len, embed_dim)
attention = SelfAttention(embed_dim)
output, weights = attention.forward(X)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
```

Slide 3: Multi-Head Attention Implementation

Multi-head attention extends single-head attention by allowing the model to jointly attend to information from different representation subspaces. This parallel processing of attention enables the model to capture various types of relationships within the data simultaneously.

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize weights for each head
        self.W_q = [np.random.randn(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.W_k = [np.random.randn(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.W_v = [np.random.randn(embed_dim, self.head_dim) for _ in range(num_heads)]
        self.W_o = np.random.randn(embed_dim, embed_dim)
    
    def attention(self, Q, K, V):
        scores = np.dot(Q, K.T) / np.sqrt(self.head_dim)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        return np.dot(weights, V)
    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        
        # Process each attention head
        head_outputs = []
        for i in range(self.num_heads):
            Q = np.dot(X, self.W_q[i])
            K = np.dot(X, self.W_k[i])
            V = np.dot(X, self.W_v[i])
            head_output = self.attention(Q, K, V)
            head_outputs.append(head_output)
        
        # Concatenate and project
        multi_head_output = np.concatenate(head_outputs, axis=-1)
        final_output = np.dot(multi_head_output, self.W_o)
        return final_output

# Example usage
embed_dim, num_heads = 512, 8
mha = MultiHeadAttention(embed_dim, num_heads)
x = np.random.randn(1, 10, embed_dim)  # Batch size 1, sequence length 10
output = mha.forward(x)
print("Multi-head attention output shape:", output.shape)
```

Slide 4: Scaled Dot-Product Attention Mathematics

The mathematical foundation of attention mechanisms relies on the scaled dot-product operation, which prevents gradient issues in deep networks. The scaling factor is crucial as it helps maintain stable gradients during training, especially with large dimension sizes.

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Implements the mathematical formula:
    Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    """
    d_k = query.shape[-1]
    
    # Compute attention scores
    attention_logits = np.matmul(query, np.transpose(key, (0, 2, 1)))
    attention_logits = attention_logits / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        attention_logits += (mask * -1e9)
    
    # Softmax normalization
    attention_weights = np.exp(attention_logits) / np.sum(np.exp(attention_logits), axis=-1, keepdims=True)
    
    # Compute output
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

# Example with dimensions
batch_size, num_heads, seq_length, depth = 2, 4, 6, 8
q = np.random.random((batch_size, num_heads, seq_length, depth))
k = np.random.random((batch_size, num_heads, seq_length, depth))
v = np.random.random((batch_size, num_heads, seq_length, depth))

output, weights = scaled_dot_product_attention(q, k, v)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 5: Positional Encoding Implementation

Positional encoding adds information about the position of tokens in a sequence, enabling the attention mechanism to consider sequential order. This implementation demonstrates both sinusoidal and learned positional encodings, crucial for sequence processing.

```python
import numpy as np

def sinusoidal_positional_encoding(position, d_model):
    """
    Implements the sinusoidal positional encoding formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    position_enc = np.zeros((position, d_model))
    
    for pos in range(position):
        for i in range(0, d_model, 2):
            div_term = np.exp(-(i * 2.0 * np.log(10000.0) / d_model))
            position_enc[pos, i] = np.sin(pos * div_term)
            if i + 1 < d_model:
                position_enc[pos, i + 1] = np.cos(pos * div_term)
    
    return position_enc

class LearnedPositionalEncoding:
    def __init__(self, max_position, d_model):
        self.encoding = np.random.randn(max_position, d_model) * 0.1
    
    def __call__(self, position):
        return self.encoding[position]

# Example usage
seq_length, d_model = 100, 512

# Sinusoidal encoding
sin_pos_encoding = sinusoidal_positional_encoding(seq_length, d_model)
print("Sinusoidal Positional Encoding shape:", sin_pos_encoding.shape)

# Learned encoding
learned_pos_encoding = LearnedPositionalEncoding(seq_length, d_model)
sample_position = 5
print("Learned Positional Encoding for position 5:", 
      learned_pos_encoding(sample_position).shape)

# Visualize encoding patterns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.imshow(sin_pos_encoding[:20, :20], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('First 20x20 of Positional Encoding Matrix')
plt.show()
```

Slide 6: Attention Masking for Sequence Processing

Masking is essential in attention mechanisms to prevent positions from attending to subsequent positions in training. This implementation shows how to create and apply different types of attention masks, including padding masks and causal masks.

```python
import numpy as np

def create_padding_mask(seq_length, valid_length):
    """
    Creates a mask for padding tokens in sequences
    """
    mask = np.zeros((seq_length, seq_length))
    mask[:, valid_length:] = -np.inf
    return mask

def create_causal_mask(seq_length):
    """
    Creates a mask for causal attention (cannot look at future tokens)
    """
    mask = np.triu(np.ones((seq_length, seq_length)) * -np.inf, k=1)
    return mask

def apply_attention_mask(scores, mask):
    """
    Applies the mask to attention scores
    """
    masked_scores = scores + mask
    return masked_scores

# Example usage
seq_length = 6
valid_length = 4

# Create masks
padding_mask = create_padding_mask(seq_length, valid_length)
causal_mask = create_causal_mask(seq_length)

# Generate sample attention scores
attention_scores = np.random.randn(seq_length, seq_length)

# Apply masks
masked_padding = apply_attention_mask(attention_scores, padding_mask)
masked_causal = apply_attention_mask(attention_scores, causal_mask)

print("Original Attention Scores:\n", attention_scores)
print("\nPadding Masked Scores:\n", masked_padding)
print("\nCausal Masked Scores:\n", masked_causal)

# Visualize masks
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.imshow(padding_mask, cmap='viridis')
ax1.set_title('Padding Mask')

ax2.imshow(causal_mask, cmap='viridis')
ax2.set_title('Causal Mask')

plt.show()
```

Slide 7: Implementing Cross-Attention Mechanism

Cross-attention enables a model to attend to information from different sequences, crucial for tasks like translation or question-answering. This implementation shows how to create a cross-attention layer that processes queries from one sequence while using keys and values from another.

```python
class CrossAttention:
    def __init__(self, query_dim, key_dim, value_dim, output_dim):
        self.W_q = np.random.randn(query_dim, output_dim) * 0.1
        self.W_k = np.random.randn(key_dim, output_dim) * 0.1
        self.W_v = np.random.randn(value_dim, output_dim) * 0.1
        self.output_dim = output_dim
    
    def forward(self, query_seq, key_value_seq):
        """
        query_seq: shape (batch_size, query_len, query_dim)
        key_value_seq: shape (batch_size, kv_len, key_dim)
        """
        # Project inputs
        Q = np.dot(query_seq, self.W_q)
        K = np.dot(key_value_seq, self.W_k)
        V = np.dot(key_value_seq, self.W_v)
        
        # Compute attention scores
        scores = np.dot(Q, K.transpose(0, 2, 1))
        scaled_scores = scores / np.sqrt(self.output_dim)
        
        # Apply softmax
        attention_weights = np.exp(scaled_scores) / np.sum(
            np.exp(scaled_scores), axis=-1, keepdims=True
        )
        
        # Compute weighted sum
        output = np.dot(attention_weights, V)
        return output, attention_weights

# Example usage
batch_size = 2
query_len, kv_len = 5, 8
query_dim, key_dim, value_dim = 64, 64, 64
output_dim = 32

# Create sample sequences
query_seq = np.random.randn(batch_size, query_len, query_dim)
key_value_seq = np.random.randn(batch_size, kv_len, key_dim)

# Initialize and apply cross-attention
cross_attention = CrossAttention(query_dim, key_dim, value_dim, output_dim)
output, weights = cross_attention.forward(query_seq, key_value_seq)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
```

Slide 8: Attention Visualization Tools

Understanding attention patterns is crucial for model interpretation. This implementation provides tools to visualize attention weights and analyze how the model attends to different parts of the input sequence.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self):
        self.attention_maps = []
    
    def plot_attention_weights(self, weights, tokens_source=None, tokens_target=None):
        """
        Visualizes attention weights between source and target sequences
        weights: shape (target_len, source_len)
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            weights,
            xticklabels=tokens_source if tokens_source else 'auto',
            yticklabels=tokens_target if tokens_target else 'auto',
            cmap='viridis',
            annot=True,
            fmt='.2f'
        )
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Tokens')
        plt.title('Attention Weights Visualization')
        return plt.gcf()
    
    def analyze_attention_patterns(self, weights):
        """
        Analyzes attention patterns and returns statistics
        """
        avg_attention = np.mean(weights, axis=0)
        max_attention = np.max(weights, axis=0)
        entropy = -np.sum(weights * np.log(weights + 1e-9), axis=1)
        
        return {
            'average_attention': avg_attention,
            'max_attention': max_attention,
            'attention_entropy': entropy
        }

# Example usage
source_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
target_tokens = ['Le', 'chat', 'est', 'assis', 'sur', 'le', 'tapis']

# Generate sample attention weights
sample_weights = np.random.rand(len(target_tokens), len(source_tokens))
sample_weights = sample_weights / sample_weights.sum(axis=1, keepdims=True)

# Create visualizer and plot
visualizer = AttentionVisualizer()
attention_fig = visualizer.plot_attention_weights(
    sample_weights,
    source_tokens,
    target_tokens
)

# Analyze patterns
patterns = visualizer.analyze_attention_patterns(sample_weights)
print("\nAttention Analysis:")
print("Average attention per source token:", patterns['average_attention'])
print("Maximum attention per source token:", patterns['max_attention'])
print("Attention entropy per target token:", patterns['attention_entropy'])
```

Slide 9: Implementing Attention with PyTorch

This implementation demonstrates a production-ready attention mechanism using PyTorch, including forward and backward propagation. The implementation includes optimizations for memory efficiency and numerical stability.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Linear layers for Q, K, V projections
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim
        )
        output = self.out_linear(context)
        
        return output, attention_weights

# Example usage
hidden_dim = 512
num_heads = 8
seq_length = 10
batch_size = 4

model = EfficientAttention(hidden_dim, num_heads)
x = torch.randn(batch_size, seq_length, hidden_dim)
mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

output, weights = model(x, x, x, mask)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 10: Real-World Application - Neural Machine Translation

This implementation shows how attention mechanisms are used in neural machine translation, including preprocessing, model implementation, and inference with attention visualization.

```python
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        
        self.encoder = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(embed_dim + hidden_dim*2, hidden_dim, batch_first=True)
        
        self.attention = nn.Linear(hidden_dim*3, 1)
        self.output_layer = nn.Linear(hidden_dim, tgt_vocab_size)
        
    def encode(self, src, src_lengths):
        embedded = self.encoder_embedding(src)
        packed = pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.encoder(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell
    
    def attend(self, decoder_state, encoder_outputs):
        attention_inputs = torch.cat([
            decoder_state.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1),
            encoder_outputs
        ], dim=-1)
        
        scores = self.attention(attention_inputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), weights
    
    def decode_step(self, encoder_outputs, decoder_input, decoder_state):
        embedded = self.decoder_embedding(decoder_input)
        context, weights = self.attend(decoder_state[0][-1], encoder_outputs)
        
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, (hidden, cell) = self.decoder(lstm_input, decoder_state)
        
        predictions = self.output_layer(output.squeeze(1))
        return predictions, (hidden, cell), weights

    def forward(self, src, tgt, src_lengths):
        encoder_outputs, hidden, cell = self.encode(src, src_lengths)
        
        decoder_state = (hidden, cell)
        outputs = []
        attentions = []
        
        for t in range(tgt.size(1) - 1):
            decoder_input = tgt[:, t:t+1]
            predictions, decoder_state, weights = self.decode_step(
                encoder_outputs, decoder_input, decoder_state
            )
            outputs.append(predictions)
            attentions.append(weights)
            
        return torch.stack(outputs, dim=1), torch.stack(attentions, dim=1)

# Example usage with toy data
src_vocab_size = 1000
tgt_vocab_size = 1000
embed_dim = 256
hidden_dim = 512

model = AttentionNMT(src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim)

# Sample batch
batch_size = 4
src_seq_len = 10
tgt_seq_len = 12

src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
src_lengths = torch.randint(5, src_seq_len+1, (batch_size,))

outputs, attentions = model(src, tgt, src_lengths)
print(f"Outputs shape: {outputs.shape}")
print(f"Attention weights shape: {attentions.shape}")
```

Slide 11: Real-World Application - Document Classification with Hierarchical Attention

This implementation demonstrates a hierarchical attention network for document classification, which applies attention at both word and sentence levels, commonly used for sentiment analysis and text categorization.

```python
import torch
import torch.nn as nn

class HierarchicalAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, word_hidden_dim, 
                 sent_hidden_dim, num_classes):
        super().__init__()
        # Word level
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.word_gru = nn.GRU(embed_dim, word_hidden_dim, bidirectional=True, 
                              batch_first=True)
        self.word_attention = nn.Linear(2*word_hidden_dim, 2*word_hidden_dim)
        self.word_context = nn.Parameter(torch.randn(2*word_hidden_dim))
        
        # Sentence level
        self.sent_gru = nn.GRU(2*word_hidden_dim, sent_hidden_dim, 
                              bidirectional=True, batch_first=True)
        self.sent_attention = nn.Linear(2*sent_hidden_dim, 2*sent_hidden_dim)
        self.sent_context = nn.Parameter(torch.randn(2*sent_hidden_dim))
        
        # Classification
        self.fc = nn.Linear(2*sent_hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def word_level_attention(self, word_hidden, word_mask):
        """Apply attention at word level"""
        word_att = torch.tanh(self.word_attention(word_hidden))
        word_att = torch.matmul(word_att, self.word_context)
        
        if word_mask is not None:
            word_att = word_att.masked_fill(word_mask == 0, float('-inf'))
        
        word_att_weights = torch.softmax(word_att, dim=1)
        word_att_out = torch.bmm(word_att_weights.unsqueeze(1), word_hidden)
        return word_att_out.squeeze(1), word_att_weights
    
    def sentence_level_attention(self, sent_hidden, sent_mask):
        """Apply attention at sentence level"""
        sent_att = torch.tanh(self.sent_attention(sent_hidden))
        sent_att = torch.matmul(sent_att, self.sent_context)
        
        if sent_mask is not None:
            sent_att = sent_att.masked_fill(sent_mask == 0, float('-inf'))
        
        sent_att_weights = torch.softmax(sent_att, dim=1)
        sent_att_out = torch.bmm(sent_att_weights.unsqueeze(1), sent_hidden)
        return sent_att_out.squeeze(1), sent_att_weights
    
    def forward(self, documents, word_mask=None, sent_mask=None):
        """
        documents: (batch_size, num_sentences, num_words)
        word_mask: (batch_size, num_sentences, num_words)
        sent_mask: (batch_size, num_sentences)
        """
        batch_size, num_sentences, num_words = documents.size()
        
        # Process each sentence
        sentence_vectors = []
        word_attention_weights = []
        
        for i in range(num_sentences):
            # Word embedding
            words = documents[:, i, :]
            word_embed = self.word_embedding(words)
            
            # Word encoder
            word_hidden, _ = self.word_gru(word_embed)
            
            # Word attention
            curr_word_mask = word_mask[:, i, :] if word_mask is not None else None
            sent_vector, word_weights = self.word_level_attention(
                word_hidden, curr_word_mask)
            
            sentence_vectors.append(sent_vector)
            word_attention_weights.append(word_weights)
        
        # Stack sentence vectors
        sentence_vectors = torch.stack(sentence_vectors, dim=1)
        
        # Sentence encoder
        sent_hidden, _ = self.sent_gru(sentence_vectors)
        
        # Sentence attention
        doc_vector, sent_attention_weights = self.sentence_level_attention(
            sent_hidden, sent_mask)
        
        # Classification
        doc_vector = self.dropout(doc_vector)
        output = self.fc(doc_vector)
        
        return output, word_attention_weights, sent_attention_weights

# Example usage
vocab_size = 10000
embed_dim = 200
word_hidden_dim = 100
sent_hidden_dim = 100
num_classes = 5
batch_size = 16
num_sentences = 10
num_words = 50

model = HierarchicalAttention(vocab_size, embed_dim, word_hidden_dim, 
                            sent_hidden_dim, num_classes)

# Sample batch
documents = torch.randint(0, vocab_size, (batch_size, num_sentences, num_words))
word_mask = torch.ones(batch_size, num_sentences, num_words)
sent_mask = torch.ones(batch_size, num_sentences)

output, word_weights, sent_weights = model(documents, word_mask, sent_mask)
print(f"Output shape: {output.shape}")
print(f"Word attention weights shape: {word_weights[0].shape}")
print(f"Sentence attention weights shape: {sent_weights.shape}")
```

Slide 12: Transformer Block Implementation with Advanced Attention

This implementation shows a complete transformer block including multi-head attention, layer normalization, and feed-forward networks with advanced features like relative positional encoding and gradient checkpointing.

```python
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1,
                 use_relative_pos=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_relative_pos = use_relative_pos
        
        # Multi-head attention
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Relative positional encoding
        if use_relative_pos:
            self.rel_pos_embed = nn.Parameter(
                torch.randn(2 * hidden_dim - 1, self.head_dim))
            
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def relative_position_to_absolute(self, x):
        """Convert relative position representation to absolute"""
        batch_size, num_heads, seq_length, _ = x.size()
        
        # Pad for shifting
        col_pad = torch.zeros(batch_size, num_heads, seq_length, 1,
                            device=x.device)
        x = torch.cat([x, col_pad], dim=-1)
        
        flat_x = x.view(batch_size, num_heads, seq_length * 2 - 1)
        flat_pad = torch.zeros(batch_size, num_heads, seq_length - 1,
                             device=x.device)
        
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=-1)
        
        # Reshape and slice out the padded elements
        final_x = flat_x_padded.view(batch_size, num_heads, seq_length + 1,
                                   seq_length)
        final_x = final_x[:, :, :seq_length, :]
        
        return final_x
    
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        
        # Multi-head attention
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads,
                                 self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads,
                                 self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads,
                                 self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add relative positional encoding
        if self.use_relative_pos:
            rel_pos_bias = self.compute_relative_position_bias(seq_length)
            scores = scores + rel_pos_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.hidden_dim)
        out = self.out_linear(context)
        
        # Residual connection and layer normalization
        x = self.norm1(x + self.dropout(out))
        
        # Feed-forward network
        ff_out = self.ff(x)
        
        # Residual connection and layer normalization
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attention_weights
    
    def compute_relative_position_bias(self, seq_length):
        """Compute relative positional bias"""
        if not self.use_relative_pos:
            return 0
            
        positions = torch.arange(seq_length, device=self.rel_pos_embed.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions += seq_length - 1  # Shift to all positive indices
        
        bias = self.rel_pos_embed[relative_positions]
        return bias.unsqueeze(0).unsqueeze(0)

# Example usage
hidden_dim = 512
num_heads = 8
ff_dim = 2048
seq_length = 100
batch_size = 16

transformer = TransformerBlock(hidden_dim, num_heads, ff_dim)
x = torch.randn(batch_size, seq_length, hidden_dim)
mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

output, attention = transformer(x, mask)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention.shape}")
```

Slide 13: Advanced Attention Regularization and Loss Functions

This implementation demonstrates advanced techniques for regularizing attention mechanisms and specialized loss functions that improve attention learning, including entropy regularization and sparse attention penalties.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRegularizer:
    def __init__(self, entropy_weight=0.1, sparsity_weight=0.1, coverage_weight=0.1):
        self.entropy_weight = entropy_weight
        self.sparsity_weight = sparsity_weight
        self.coverage_weight = coverage_weight
    
    def entropy_loss(self, attention_weights):
        """
        Encourages attention to be more focused or dispersed
        Lower entropy = more focused attention
        """
        eps = 1e-7
        entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
        return entropy.mean()
    
    def sparsity_loss(self, attention_weights):
        """
        Encourages sparse attention distributions
        Uses L1 regularization on attention weights
        """
        return torch.norm(attention_weights, p=1, dim=-1).mean()
    
    def coverage_loss(self, attention_weights, coverage_vector):
        """
        Prevents over-attention and under-attention to input tokens
        """
        coverage_vector = coverage_vector + attention_weights
        penalty = torch.min(coverage_vector, attention_weights)
        return penalty.sum(dim=-1).mean()

class RegularizedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.regularizer = AttentionRegularizer()
        self.coverage_vector = None
    
    def forward(self, query, key, value, mask=None, return_regularization=True):
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, 
                                    self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, 
                                   self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, 
                                    self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Initialize coverage vector if None
        if self.coverage_vector is None:
            self.coverage_vector = torch.zeros_like(attention_weights)
        
        # Calculate regularization losses
        reg_losses = {
            'entropy': self.regularizer.entropy_loss(attention_weights),
            'sparsity': self.regularizer.sparsity_loss(attention_weights),
            'coverage': self.regularizer.coverage_loss(
                attention_weights, self.coverage_vector
            )
        }
        
        # Update coverage vector
        self.coverage_vector = self.coverage_vector + attention_weights
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_dim)
        output = self.out_linear(context)
        
        if return_regularization:
            return output, attention_weights, reg_losses
        return output, attention_weights

class RegularizedAttentionLoss(nn.Module):
    def __init__(self, base_criterion=nn.CrossEntropyLoss()):
        super().__init__()
        self.base_criterion = base_criterion
        
    def forward(self, outputs, targets, reg_losses):
        base_loss = self.base_criterion(outputs, targets)
        
        # Combine regularization losses
        total_reg_loss = (
            reg_losses['entropy'] * 0.1 +
            reg_losses['sparsity'] * 0.1 +
            reg_losses['coverage'] * 0.1
        )
        
        return base_loss + total_reg_loss

# Example usage
hidden_dim = 512
num_heads = 8
batch_size = 16
seq_length = 20
num_classes = 10

model = RegularizedAttention(hidden_dim, num_heads)
criterion = RegularizedAttentionLoss()

# Sample data
x = torch.randn(batch_size, seq_length, hidden_dim)
targets = torch.randint(0, num_classes, (batch_size,))
mask = torch.ones(batch_size, num_heads, seq_length, seq_length)

# Forward pass
output, attention_weights, reg_losses = model(x, x, x, mask)

# Calculate loss
logits = torch.randn(batch_size, num_classes)  # Simulated classification logits
loss = criterion(logits, targets, reg_losses)

print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Regularization losses: {reg_losses}")
print(f"Total loss: {loss.item()}")
```

Slide 14: Additional Resources

1.  "Attention Is All You Need" - Original Transformer paper [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "On Layer Normalization in the Transformer Architecture" [https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)
3.  "Self-Attention with Relative Position Representations" [https://arxiv.org/abs/1803.02155](https://arxiv.org/abs/1803.02155)
4.  "Longformer: The Long-Document Transformer" [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
5.  "Synthesizer: Rethinking Self-Attention in Transformer Models" [https://arxiv.org/abs/2005.00743](https://arxiv.org/abs/2005.00743)
6.  "Reformer: The Efficient Transformer" [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)
7.  "Sparse Transformer: Concentrated Attention Through Constrained Matrix Factorization" [https://arxiv.org/abs/1912.11637](https://arxiv.org/abs/1912.11637)

