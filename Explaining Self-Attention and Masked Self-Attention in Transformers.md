## Explaining Self-Attention and Masked Self-Attention in Transformers
Slide 1: Understanding Self-Attention Mechanism

Self-attention enables a model to weigh the importance of different words in a sequence by computing attention scores between all pairs of words. This fundamental mechanism allows the model to capture long-range dependencies and contextual relationships within the input sequence.

```python
import numpy as np

def self_attention(query, key, value):
    # Calculate attention scores using scaled dot-product
    scores = np.dot(query, key.T) / np.sqrt(key.shape[1])
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute weighted sum of values
    attention_output = np.dot(weights, value)
    return attention_output, weights

# Example usage
sequence_length, d_model = 4, 8
query = np.random.randn(sequence_length, d_model)
key = np.random.randn(sequence_length, d_model)
value = np.random.randn(sequence_length, d_model)

output, attention_weights = self_attention(query, key, value)
```

Slide 2: Mathematical Foundations of Self-Attention

The self-attention mechanism computes attention scores through a scaled dot-product operation between queries and keys, followed by softmax normalization. This mathematical foundation ensures numerical stability and effective gradient flow during training.

```python
"""
Key equations for self-attention:

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

$$softmax(x_i) = \frac{exp(x_i)}{\sum_j exp(x_j)}$$

$$d_k = \text{dimension of key vectors}$$
"""

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose()) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask * -1e9
        
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.matmul(attention_weights, V)
    return output, attention_weights
```

Slide 3: Multi-Head Attention Implementation

Multi-head attention splits the input into multiple heads, allowing the model to attend to different aspects of the input simultaneously. This parallel processing enables the model to capture various types of relationships within the sequence.

```python
import numpy as np

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
        
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)
    
    def __call__(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear transformations
        q = np.dot(query, self.wq)
        k = np.dot(key, self.wk)
        v = np.dot(value, self.wv)
        
        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape output
        output = scaled_attention.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = np.dot(output, self.dense)
        return output
```

Slide 4: Masked Self-Attention Implementation

The masked self-attention mechanism prevents the model from attending to future tokens during training, which is crucial for autoregressive tasks. This implementation demonstrates how to create and apply attention masks.

```python
def create_padding_mask(sequence):
    # Create mask for padding tokens (zeros)
    return (sequence == 0).astype(np.float32)[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    # Create mask to prevent attention to future tokens
    mask = 1 - np.triu(np.ones((size, size)), k=1)
    return mask.astype(np.float32)

def masked_self_attention(query, key, value, mask=None):
    d_k = key.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores += (mask * -1e9)
    
    # Apply softmax and compute weighted sum
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
seq_len = 5
d_model = 8
look_ahead_mask = create_look_ahead_mask(seq_len)
sequence = np.random.randn(1, seq_len, d_model)
masked_output, weights = masked_self_attention(sequence, sequence, sequence, look_ahead_mask)
```

Slide 5: Positional Encoding in Transformers

Positional encodings inject sequential information into the self-attention mechanism since it has no inherent notion of order. These encodings use sinusoidal functions to create unique position-dependent patterns that the model can learn to interpret.

```python
def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angles

    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    # Apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return pos_encoding

# Example usage
sequence_length = 50
d_model = 512
pos_encoding = positional_encoding(sequence_length, d_model)
print(f"Positional encoding shape: {pos_encoding.shape}")
```

Slide 6: Layer Normalization Implementation

Layer normalization is crucial for transformer architectures, stabilizing training by normalizing activations across features. This implementation shows the forward pass computation with learned scale and shift parameters.

```python
class LayerNormalization:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        
    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Initialize parameters if not already done
        if self.gamma is None:
            self.gamma = np.ones_like(x[0])
            self.beta = np.zeros_like(x[0])
        
        # Normalize and scale
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta

# Example usage
batch_size, seq_length, features = 2, 10, 512
x = np.random.randn(batch_size, seq_length, features)
layer_norm = LayerNormalization()
normalized_x = layer_norm(x)
```

Slide 7: Feed-Forward Neural Network in Transformers

The feed-forward network in transformers consists of two linear transformations with a ReLU activation in between. This component processes each position independently, adding non-linearity to the model's representations.

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def __call__(self, x):
        # First linear transformation
        output = np.dot(x, self.W1) + self.b1
        # ReLU activation
        output = self.relu(output)
        # Second linear transformation
        output = np.dot(output, self.W2) + self.b2
        return output

# Example usage
d_model, d_ff = 512, 2048
ff_layer = FeedForward(d_model, d_ff)
x = np.random.randn(2, 10, d_model)  # (batch_size, seq_length, d_model)
output = ff_layer(x)
```

Slide 8: Encoder Block Implementation

The encoder block combines self-attention, feed-forward networks, and layer normalization in a specific arrangement. This implementation shows the full forward pass through a single encoder layer.

```python
class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout_rate = dropout_rate
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)
    
    def __call__(self, x, training=True, mask=None):
        # Multi-head attention
        attn_output = self.mha(x, x, x, mask)
        if training:
            attn_output = self.dropout(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        if training:
            ffn_output = self.dropout(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# Example usage
d_model, num_heads, d_ff = 512, 8, 2048
encoder = EncoderBlock(d_model, num_heads, d_ff)
x = np.random.randn(2, 10, d_model)
output = encoder(x)
```

Slide 9: Real-World Application - Neural Machine Translation

This implementation demonstrates a complete neural machine translation system using transformers. The example includes data preprocessing, tokenization, and the core translation pipeline using self-attention mechanisms.

```python
import numpy as np
from collections import Counter

class NMTTransformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8):
        self.d_model = d_model
        self.embedding = np.random.randn(src_vocab_size, d_model) / np.sqrt(d_model)
        self.encoder = EncoderBlock(d_model, num_heads, d_model * 4)
        self.decoder = DecoderBlock(d_model, num_heads, d_model * 4)
        self.final_layer = np.random.randn(d_model, tgt_vocab_size) / np.sqrt(d_model)
        
    def preprocess_text(self, text):
        # Tokenization and vocabulary building
        tokens = text.lower().split()
        vocab = Counter(tokens)
        return [vocab.get(token, 0) for token in tokens]
    
    def translate(self, src_sentence, max_length=50):
        # Preprocess source sentence
        src_tokens = self.preprocess_text(src_sentence)
        src_embedded = np.dot(src_tokens, self.embedding)
        
        # Encode source sequence
        encoder_output = self.encoder(src_embedded)
        
        # Decode step by step
        output_sequence = []
        for _ in range(max_length):
            decoder_output = self.decoder(
                np.array(output_sequence),
                encoder_output
            )
            prediction = np.dot(decoder_output[-1], self.final_layer)
            next_token = np.argmax(prediction)
            
            if next_token == self.eos_token:
                break
                
            output_sequence.append(next_token)
            
        return output_sequence

# Example usage
src_vocab_size, tgt_vocab_size = 10000, 8000
nmt_model = NMTTransformer(src_vocab_size, tgt_vocab_size)
translation = nmt_model.translate("Hello world")
```

Slide 10: Real-World Application - Document Classification

This implementation shows how self-attention can be used for document classification, including preprocessing, attention-based feature extraction, and classification layers.

```python
class DocumentClassifier:
    def __init__(self, vocab_size, num_classes, d_model=256, num_heads=4):
        self.embedding = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.classifier = np.random.randn(d_model, num_classes) / np.sqrt(d_model)
        
    def preprocess_document(self, text):
        # Convert text to lowercase and split into words
        words = text.lower().split()
        # Create word indices (simplified)
        word_to_idx = {word: idx for idx, word in enumerate(set(words))}
        return [word_to_idx.get(word, 0) for word in words]
    
    def forward(self, document):
        # Convert document to embeddings
        token_ids = self.preprocess_document(document)
        embeddings = np.take(self.embedding, token_ids, axis=0)
        
        # Apply self-attention
        attended_features, attention_weights = self.attention(
            embeddings, embeddings, embeddings
        )
        
        # Global average pooling
        doc_embedding = np.mean(attended_features, axis=1)
        
        # Classification
        logits = np.dot(doc_embedding, self.classifier)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        return probs, attention_weights

# Example usage
classifier = DocumentClassifier(vocab_size=5000, num_classes=3)
sample_doc = "This is a sample document for classification"
predictions, attention_map = classifier.forward(sample_doc)
```

Slide 11: Attention Visualization Implementation

This implementation provides tools for visualizing attention patterns in transformer models, helping understand how the model attends to different parts of the input sequence.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, title="Attention Weights"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        annot=True,
        fmt='.2f'
    )
    plt.title(title)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    
    # Example usage demonstration
    sample_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    sample_attention = np.random.rand(len(sample_tokens), len(sample_tokens))
    sample_attention = sample_attention / sample_attention.sum(axis=-1, keepdims=True)
    
    visualize_attention(sample_attention, sample_tokens)
    return plt.gcf()  # Return figure for downstream use
```

Slide 12: Advanced Training Techniques for Transformers

This implementation showcases advanced training techniques including gradient accumulation, learning rate scheduling, and label smoothing, which are crucial for training transformer models effectively.

```python
class TransformerTrainer:
    def __init__(self, model, learning_rate=0.0001, warmup_steps=4000):
        self.model = model
        self.initial_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def get_learning_rate(self):
        # Implement learning rate schedule
        arg1 = np.reciprocal(np.sqrt(self.step))
        arg2 = self.step * (self.warmup_steps ** -1.5)
        return self.initial_lr * min(arg1, arg2)
    
    def label_smoothing(self, labels, smoothing=0.1):
        confidence = 1.0 - smoothing
        smoothed_labels = np.zeros_like(labels)
        smoothed_labels += smoothing / labels.shape[-1]
        smoothed_labels[labels == 1] = confidence
        return smoothed_labels
    
    def train_step(self, batch, accumulation_steps=4):
        total_loss = 0
        gradients = []
        
        for mini_batch in np.array_split(batch, accumulation_steps):
            # Forward pass
            outputs = self.model(mini_batch)
            loss = self.compute_loss(outputs, mini_batch['labels'])
            
            # Backward pass (simplified)
            grads = self.compute_gradients(loss)
            gradients.append(grads)
            total_loss += loss
        
        # Average gradients and update
        avg_gradients = [np.mean(grad, axis=0) for grad in zip(*gradients)]
        self.apply_gradients(avg_gradients)
        
        self.step += 1
        return total_loss / accumulation_steps

    def compute_loss(self, logits, labels, label_smoothing=0.1):
        smoothed_labels = self.label_smoothing(labels, label_smoothing)
        return -np.sum(smoothed_labels * np.log(logits + 1e-9))
```

Slide 13: Performance Optimization and Memory Management

This implementation focuses on optimizing transformer performance through efficient memory management and computational optimizations, including attention score caching and gradient checkpointing.

```python
class OptimizedTransformer:
    def __init__(self, d_model, num_heads, use_cache=True):
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_cache = use_cache
        self.attention_cache = {}
        
    def optimize_attention(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Cache key for this input
        cache_key = hash(key.tobytes())
        
        if self.use_cache and cache_key in self.attention_cache:
            attention_weights = self.attention_cache[cache_key]
        else:
            # Compute attention scores
            scores = np.matmul(query, key.transpose(0, 2, 1))
            scores = scores / np.sqrt(self.d_model)
            
            if mask is not None:
                scores += (mask * -1e9)
            
            # Apply softmax
            attention_weights = np.exp(scores) / np.sum(
                np.exp(scores), axis=-1, keepdims=True
            )
            
            if self.use_cache:
                self.attention_cache[cache_key] = attention_weights
        
        # Compute output
        output = np.matmul(attention_weights, value)
        
        return output, attention_weights
    
    def gradient_checkpointing(self, layer_outputs, save_interval=2):
        # Save only every save_interval activations
        checkpoints = {}
        for i, output in enumerate(layer_outputs):
            if i % save_interval == 0:
                checkpoints[i] = output
        return checkpoints
    
    def clear_cache(self):
        self.attention_cache.clear()

# Example usage
optimizer = OptimizedTransformer(d_model=512, num_heads=8)
q = np.random.randn(2, 10, 512)
k = np.random.randn(2, 10, 512)
v = np.random.randn(2, 10, 512)
output, weights = optimizer.optimize_attention(q, k, v)
```

Slide 14: Additional Resources

1.  "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3.  "Deep Residual Learning for Self-Attention" - [https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768)
4.  "Reformer: The Efficient Transformer" - [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)
5.  "Training Tips for the Transformer Model" - [https://arxiv.org/abs/1804.00247](https://arxiv.org/abs/1804.00247)

