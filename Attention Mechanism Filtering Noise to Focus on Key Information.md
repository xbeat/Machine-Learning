## Attention Mechanism Filtering Noise to Focus on Key Information
Slide 1: Understanding Attention Mechanism Components

The attention mechanism consists of three fundamental components: queries, keys, and values. These elements work together to create weighted relationships between different parts of the input sequence, enabling the model to focus on relevant information dynamically during processing.

```python
import numpy as np

class AttentionComponents:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        # Initialize weight matrices for Q, K, V transformations
        self.W_query = np.random.randn(hidden_dim, hidden_dim)
        self.W_key = np.random.randn(hidden_dim, hidden_dim)
        self.W_value = np.random.randn(hidden_dim, hidden_dim)
    
    def transform_input(self, x):
        # Transform input into Q, K, V representations
        Q = np.dot(x, self.W_query)
        K = np.dot(x, self.W_key)
        V = np.dot(x, self.W_value)
        return Q, K, V

# Example usage
hidden_dim = 4
attention = AttentionComponents(hidden_dim)
input_sequence = np.random.randn(3, hidden_dim)  # Batch size 3
Q, K, V = attention.transform_input(input_sequence)
print(f"Query shape: {Q.shape}")
print(f"Key shape: {K.shape}")
print(f"Value shape: {V.shape}")
```

Slide 2: Mathematical Foundation of Attention

The core attention mechanism is defined by the scaled dot-product attention formula. This mathematical operation determines how much focus to place on different parts of the input sequence by computing compatibility scores between queries and keys.

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    $$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$
    """
    # Calculate attention scores
    d_k = K.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Apply attention weights to values
    output = np.dot(attention_weights, V)
    return output, attention_weights

# Example usage
Q = np.random.randn(4, 8)  # 4 queries of dimension 8
K = np.random.randn(6, 8)  # 6 keys of dimension 8
V = np.random.randn(6, 8)  # 6 values of dimension 8

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 3: Multi-Head Attention Implementation

Multi-head attention allows the model to jointly attend to information from different representation subspaces, enabling the capture of various types of relationships between elements in the sequence simultaneously.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Calculate attention
        scaled_attention, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = np.dot(scaled_attention, self.W_o)
        return output, attention_weights
```

Slide 4: Position-wise Feed-Forward Networks

The position-wise feed-forward network is a crucial component that processes each position's output from the attention layer independently, applying the same transformation to each position in the sequence.

```python
class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff)
        self.w2 = np.random.randn(d_ff, d_model)
        
    def forward(self, x):
        # First linear transformation with ReLU
        intermediate = np.maximum(0, np.dot(x, self.w1))
        # Second linear transformation
        return np.dot(intermediate, self.w2)

# Example usage
d_model, d_ff = 512, 2048
ff_network = PositionwiseFeedForward(d_model, d_ff)
sample_input = np.random.randn(32, 10, d_model)  # batch_size=32, seq_len=10
output = ff_network.forward(sample_input)
print(f"Feed-forward output shape: {output.shape}")
```

Slide 5: Positional Encoding Design

Positional encodings are essential for introducing sequence order information since attention mechanisms are inherently permutation-invariant. The encoding uses sine and cosine functions of different frequencies to create unique position-dependent patterns.

```python
def get_positional_encoding(seq_len, d_model):
    """
    $$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$$
    $$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$$
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# Example usage
seq_length, d_model = 100, 512
positional_encoding = get_positional_encoding(seq_length, d_model)
print(f"Positional encoding shape: {positional_encoding.shape}")
```

Slide 6: Self-Attention Implementation

Self-attention allows each position in the sequence to attend to all positions in the same sequence, creating a rich representation that captures both local and global dependencies.

```python
class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
        
    def forward(self, x):
        # Self-attention uses the same input for Q, K, V
        energy = np.dot(x, x.transpose(0, 2, 1)) / self.scale
        attention_weights = np.exp(energy) / np.sum(np.exp(energy), axis=-1, keepdims=True)
        
        # Apply attention weights to input
        output = np.dot(attention_weights, x)
        return output, attention_weights

# Example usage
batch_size, seq_len, d_model = 32, 10, 64
input_sequence = np.random.randn(batch_size, seq_len, d_model)
self_attention = SelfAttention(d_model)
output, weights = self_attention.forward(input_sequence)
print(f"Self-attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 7: Attention Mask Implementation

Masking is crucial for preventing attention to certain positions, such as padding tokens or future positions in autoregressive models, ensuring the model maintains causal relationships in sequential data.

```python
def create_attention_mask(sequence_length, padding_mask=None):
    # Create causal mask (lower triangular)
    causal_mask = np.tril(np.ones((sequence_length, sequence_length)))
    
    if padding_mask is not None:
        # Combine with padding mask
        padding_mask = padding_mask[:, np.newaxis, np.newaxis, :]
        causal_mask = causal_mask * padding_mask
    
    return causal_mask

def masked_attention(Q, K, V, mask):
    scores = np.dot(Q, K.transpose(-2, -1)) / np.sqrt(K.shape[-1])
    scores = scores.masked_fill(mask == 0, -1e9)
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.dot(weights, V)

# Example usage
seq_len = 10
mask = create_attention_mask(seq_len)
print("Attention mask shape:", mask.shape)
print("Sample mask pattern:\n", mask[0])
```

Slide 8: Attention Layer Normalization

Layer normalization is essential for stabilizing the learning process in attention-based models by normalizing the inputs across the feature dimension, helping to prevent internal covariate shift.

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-12):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example usage
batch_size, seq_len, d_model = 32, 10, 512
layer_norm = LayerNorm(d_model)
sample_input = np.random.randn(batch_size, seq_len, d_model)
normalized_output = layer_norm.forward(sample_input)
print(f"Normalized output shape: {normalized_output.shape}")
```

Slide 9: Real-world Example - Document Classification

The attention mechanism can be effectively used for document classification by learning to focus on relevant parts of the text. This implementation demonstrates a complete pipeline for processing and classifying text documents.

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DocumentClassifier:
    def __init__(self, vocab_size, embedding_dim, num_classes):
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.attention = SelfAttention(embedding_dim)
        self.classifier = np.random.randn(embedding_dim, num_classes) * 0.01
        
    def forward(self, x):
        # Convert token indices to embeddings
        embedded = self.embedding[x]
        
        # Apply self-attention
        attended, weights = self.attention.forward(embedded)
        
        # Global average pooling
        doc_vector = np.mean(attended, axis=1)
        
        # Classification layer
        logits = np.dot(doc_vector, self.classifier)
        return logits, weights

# Example usage
vocab_size, embedding_dim, num_classes = 10000, 256, 5
classifier = DocumentClassifier(vocab_size, embedding_dim, num_classes)

# Simulate document input (batch_size=16, seq_len=100)
sample_docs = np.random.randint(0, vocab_size, (16, 100))
predictions, attention_weights = classifier.forward(sample_docs)
print(f"Predictions shape: {predictions.shape}")
```

Slide 10: Results Analysis for Document Classification

Performance metrics and visualization of the attention mechanism's behavior in the document classification task, showing how the model attends to different parts of the input text.

```python
def analyze_attention_results(attention_weights, tokens, predictions, true_labels):
    # Calculate classification metrics
    accuracy = np.mean(np.argmax(predictions, axis=1) == true_labels)
    
    # Analyze attention distribution
    avg_attention = np.mean(attention_weights, axis=0)
    max_attention_tokens = np.argsort(avg_attention, axis=-1)[-5:]
    
    print(f"Classification Accuracy: {accuracy:.4f}")
    print("\nMost attended tokens:")
    for idx in max_attention_tokens:
        print(f"Token {idx}: Weight {avg_attention[idx]:.4f}")
    
    return {
        'accuracy': accuracy,
        'attention_stats': {
            'mean': np.mean(attention_weights),
            'std': np.std(attention_weights),
            'max_weight': np.max(attention_weights)
        }
    }

# Example usage
sample_tokens = np.arange(100)
true_labels = np.random.randint(0, num_classes, 16)
results = analyze_attention_results(attention_weights, sample_tokens, predictions, true_labels)
```

Slide 11: Cross-Attention Implementation

Cross-attention enables the model to attend to information from different sequences, crucial for tasks like machine translation where the model needs to align source and target sequences.

```python
class CrossAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.w_query = np.random.randn(d_model, d_model)
        self.w_key = np.random.randn(d_model, d_model)
        self.w_value = np.random.randn(d_model, d_model)
        
    def forward(self, query_seq, key_value_seq):
        # Transform inputs
        Q = np.dot(query_seq, self.w_query)
        K = np.dot(key_value_seq, self.w_key)
        V = np.dot(key_value_seq, self.w_value)
        
        # Scaled dot-product attention
        scores = np.dot(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        output = np.dot(attention_weights, V)
        return output, attention_weights

# Example usage
d_model = 256
cross_attention = CrossAttention(d_model)
query_sequence = np.random.randn(8, 10, d_model)  # target sequence
key_value_sequence = np.random.randn(8, 15, d_model)  # source sequence
output, cross_attention_weights = cross_attention.forward(query_sequence, key_value_sequence)
print(f"Cross-attention output shape: {output.shape}")
```

Slide 12: Attention with Relative Position Encoding

Relative positional encoding enhances the standard attention mechanism by explicitly modeling the relationships between positions, allowing the model to better capture local patterns and dependencies.

```python
class RelativePositionAttention:
    def __init__(self, d_model, max_relative_position):
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        # Initialize relative position embeddings
        self.relative_embeddings = np.random.randn(
            2 * max_relative_position + 1,
            d_model
        )
    
    def _get_relative_positions(self, length):
        range_vec = np.arange(length)
        distance_mat = range_vec[None, :] - range_vec[:, None]
        distance_mat_clipped = np.clip(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        return distance_mat_clipped + self.max_relative_position
    
    def forward(self, x):
        seq_length = x.shape[1]
        relative_positions = self._get_relative_positions(seq_length)
        relative_position_embeddings = self.relative_embeddings[relative_positions]
        
        # Compute attention scores with relative positions
        scores = np.dot(x, x.transpose(0, 2, 1)) + np.sum(
            x[:, :, None, :] * relative_position_embeddings, axis=-1
        )
        
        scores = scores / np.sqrt(self.d_model)
        attention_weights = np.exp(scores) / np.sum(
            np.exp(scores), axis=-1, keepdims=True
        )
        
        output = np.dot(attention_weights, x)
        return output, attention_weights

# Example usage
d_model, max_relative_position = 64, 10
rel_pos_attention = RelativePositionAttention(d_model, max_relative_position)
sample_input = np.random.randn(4, 20, d_model)
output, weights = rel_pos_attention.forward(sample_input)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 13: Attention Training Pipeline

A complete training pipeline implementation for attention-based models, including gradient computation, parameter updates, and monitoring of training progress.

```python
class AttentionTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_history = []
    
    def compute_loss(self, predictions, targets):
        # Cross-entropy loss
        exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
        softmax_preds = exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
        loss = -np.mean(np.log(softmax_preds[np.arange(len(targets)), targets]))
        return loss
    
    def train_step(self, batch_x, batch_y):
        # Forward pass
        predictions, attention_weights = self.model.forward(batch_x)
        loss = self.compute_loss(predictions, batch_y)
        
        # Backward pass (simplified gradient update)
        grad_scale = self.learning_rate / len(batch_x)
        self.model.update_parameters(grad_scale)
        
        self.loss_history.append(loss)
        return loss, attention_weights
    
    def train(self, train_data, num_epochs, batch_size):
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Batch processing
            for i in range(0, len(train_data), batch_size):
                batch_x = train_data[i:i+batch_size]
                batch_y = train_data[i:i+batch_size]
                
                loss, _ = self.train_step(batch_x, batch_y)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Example usage
trainer = AttentionTrainer(model=DocumentClassifier(10000, 256, 5))
sample_data = np.random.randint(0, 10000, (1000, 100))
trainer.train(sample_data, num_epochs=5, batch_size=32)
```

Slide 14: Additional Resources

*   Attention Is All You Need (Original Transformer Paper): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   Self-Attention with Relative Position Representations: [https://arxiv.org/abs/1803.02155](https://arxiv.org/abs/1803.02155)
*   Effective Approaches to Attention-based Neural Machine Translation: [https://arxiv.org/abs/1508.04025](https://arxiv.org/abs/1508.04025)
*   For practical implementations: [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)
*   For attention visualization tools: [https://github.com/jessevig/bertviz](https://github.com/jessevig/bertviz)

