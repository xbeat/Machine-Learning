## Self-Attention vs. Cross-Attention in Deep Learning
Slide 1: Understanding Self-Attention Implementation

Self-attention allows tokens in a sequence to weigh the importance of other tokens when encoding meaning. This fundamental mechanism enables models to capture dependencies between different positions in the sequence through trainable query, key, and value matrices.

```python
import numpy as np

class SelfAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        
    def scaled_dot_product(self, Q, K, V, mask=None):
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = np.softmax(scores, axis=-1)
        return np.matmul(attention, V)
```

Slide 2: Self-Attention Forward Pass

The forward pass of self-attention involves computing query, key, and value matrices through linear transformations, then applying scaled dot-product attention. This implementation shows how to process input sequences through the attention mechanism.

```python
def forward(self, x, mask=None):
    batch_size, seq_length, _ = x.shape
    
    # Linear transformations
    Q = np.matmul(x, self.W_q)
    K = np.matmul(x, self.W_k)
    V = np.matmul(x, self.W_v)
    
    # Reshape for multi-head attention
    Q = Q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
    K = K.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
    V = V.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
    
    # Compute attention
    output = self.scaled_dot_product(Q, K, V, mask)
    
    # Reshape back
    output = output.reshape(batch_size, seq_length, self.d_model)
    return output
```

Slide 3: Implementing Cross-Attention

Cross-attention differs from self-attention by operating on two different sequences. This implementation shows how to create relationships between source and target sequences, commonly used in sequence-to-sequence models like machine translation.

```python
class CrossAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Separate weights for encoder and decoder
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
    
    def forward(self, encoder_output, decoder_input):
        # Generate Q from decoder, K and V from encoder
        Q = np.matmul(decoder_input, self.W_q)
        K = np.matmul(encoder_output, self.W_k)
        V = np.matmul(encoder_output, self.W_v)
        
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = np.softmax(scores, axis=-1)
        return np.matmul(attention, V)
```

Slide 4: Mathematical Foundations of Attention

The attention mechanism can be expressed mathematically through matrix operations. The key equation for scaled dot-product attention forms the basis for both self and cross-attention implementations.

```python
"""
The fundamental attention equation:
$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Where:
$$d_k$$ is the dimension of the key vectors
$$Q$$ is the query matrix
$$K$$ is the key matrix
$$V$$ is the value matrix
"""

def attention_formula(Q, K, V, d_k):
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    weights = np.softmax(scores, axis=-1)
    return np.matmul(weights, V)
```

Slide 5: Implementing Attention Masking

Masking is crucial in self-attention to prevent information leakage from future tokens. This implementation demonstrates how to create and apply attention masks for autoregressive models.

```python
def create_attention_mask(seq_length):
    # Create causal mask (lower triangular)
    mask = np.tril(np.ones((seq_length, seq_length)))
    return mask

def masked_attention(Q, K, V, mask):
    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.shape[-1])
    
    # Apply mask by setting masked positions to -inf
    scores = np.where(mask == 0, float('-inf'), scores)
    weights = np.softmax(scores, axis=-1)
    
    return np.matmul(weights, V)
```

Slide 6: Real-World Application: Neural Machine Translation

Neural Machine Translation represents a perfect use case for both self and cross-attention mechanisms. The encoder processes the source language using self-attention, while the decoder uses both self and cross-attention for translation.

```python
class NMTAttention:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512):
        self.encoder_emb = np.random.randn(src_vocab_size, d_model)
        self.decoder_emb = np.random.randn(tgt_vocab_size, d_model)
        self.self_attention = SelfAttention(d_model, num_heads=8)
        self.cross_attention = CrossAttention(d_model, num_heads=8)
        
    def translate(self, source_seq, target_seq):
        # Embed source sequence
        src_embedded = np.matmul(source_seq, self.encoder_emb)
        # Apply self-attention to source
        encoder_output = self.self_attention.forward(src_embedded)
        
        # Embed target sequence
        tgt_embedded = np.matmul(target_seq, self.decoder_emb)
        # Apply self-attention with masking
        decoder_output = self.self_attention.forward(
            tgt_embedded, 
            mask=create_attention_mask(target_seq.shape[1])
        )
        
        # Apply cross-attention
        final_output = self.cross_attention.forward(encoder_output, decoder_output)
        return final_output
```

Slide 7: Multi-Head Attention Implementation

Multi-head attention allows the model to jointly attend to information from different representation subspaces. This implementation shows how to split attention into multiple heads and process them in parallel.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Initialize weights for all heads
        self.W_q = np.random.randn(num_heads, d_model, self.head_dim)
        self.W_k = np.random.randn(num_heads, d_model, self.head_dim)
        self.W_v = np.random.randn(num_heads, d_model, self.head_dim)
        self.W_o = np.random.randn(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        
        # Project input for each head
        Q = np.stack([np.matmul(x, w_q) for w_q in self.W_q])
        K = np.stack([np.matmul(x, w_k) for w_k in self.W_k])
        V = np.stack([np.matmul(x, w_v) for w_v in self.W_v])
        
        # Compute attention for each head
        head_outputs = []
        for i in range(self.num_heads):
            scores = np.matmul(Q[i], K[i].transpose(-2, -1))
            attention = np.softmax(scores / np.sqrt(self.head_dim), axis=-1)
            head_output = np.matmul(attention, V[i])
            head_outputs.append(head_output)
        
        # Concatenate and project heads
        multi_head_output = np.concatenate(head_outputs, axis=-1)
        return np.matmul(multi_head_output, self.W_o)
```

Slide 8: Attention Visualization

Understanding attention weights through visualization helps debug and interpret model behavior. This implementation provides tools to extract and visualize attention patterns.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, source_tokens, target_tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap='viridis',
        annot=True,
        fmt='.2f'
    )
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title('Attention Weights Visualization')
    
    # Example usage
    source = ['Hello', 'world', '!']
    target = ['Hola', 'mundo', '!']
    weights = np.random.rand(len(target), len(source))
    weights = weights / weights.sum(axis=1, keepdims=True)
    visualize_attention(weights, source, target)
```

Slide 9: Positional Encoding for Attention

Attention mechanisms need position information since they are permutation-invariant. This implementation shows how to add sinusoidal positional encodings to input embeddings.

```python
def positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

class PositionalEncoder:
    def __init__(self, d_model, max_seq_length=5000):
        self.pos_encoding = positional_encoding(max_seq_length, d_model)
        
    def forward(self, x):
        seq_length = x.shape[1]
        return x + self.pos_encoding[:seq_length, :]
```

Slide 10: Relative Position Self-Attention

Relative position self-attention enhances the standard attention mechanism by explicitly modeling relative positions between tokens. This implementation shows how to incorporate relative position information into the attention computation.

```python
class RelativePositionSelfAttention:
    def __init__(self, d_model, num_heads, max_relative_position=32):
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        
        # Initialize relative position embeddings
        self.relative_positions_embeddings = np.random.randn(
            2 * max_relative_position + 1,
            d_model // num_heads
        )
    
    def _relative_position_bucket(self, relative_position):
        ret = np.zeros_like(relative_position)
        positive_pos = relative_position >= 0
        neg = -relative_position
        
        # Positive positions
        ret[positive_pos] = np.minimum(
            relative_position[positive_pos],
            self.max_relative_position
        )
        # Negative positions
        ret[~positive_pos] = -np.minimum(
            neg[~positive_pos],
            self.max_relative_position
        )
        return ret + self.max_relative_position
    
    def forward(self, q, k, v):
        seq_length = q.shape[1]
        positions = np.arange(seq_length)
        relative_positions = positions[:, None] - positions[None, :]
        relative_positions_bucket = self._relative_position_bucket(relative_positions)
        
        rel_embeddings = self.relative_positions_embeddings[relative_positions_bucket]
        
        # Include relative position information in attention computation
        scores = np.matmul(q, k.transpose(-2, -1))
        relative_scores = np.matmul(q, rel_embeddings.transpose(-2, -1))
        
        attention = np.softmax((scores + relative_scores) / np.sqrt(self.d_model), axis=-1)
        return np.matmul(attention, v)
```

Slide 11: Real-World Example: Document Classification

This implementation demonstrates how to use self-attention for document classification, showing preprocessing, attention mechanism, and classification head working together.

```python
class DocumentClassifier:
    def __init__(self, vocab_size, d_model, num_classes):
        self.embedding = np.random.randn(vocab_size, d_model)
        self.attention = SelfAttention(d_model, num_heads=8)
        self.classifier = np.random.randn(d_model, num_classes)
        
    def preprocess(self, text, max_length=512):
        # Simplified tokenization for demonstration
        tokens = text.lower().split()
        token_ids = [hash(token) % self.embedding.shape[0] for token in tokens]
        
        # Padding or truncation
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))
        
        return np.array(token_ids)
    
    def forward(self, text):
        # Preprocess and embed text
        token_ids = self.preprocess(text)
        embedded = np.matmul(
            np.eye(len(token_ids))[token_ids],
            self.embedding
        )
        
        # Apply self-attention
        attended = self.attention.forward(embedded[None, :, :])[0]
        
        # Pool attention outputs (mean pooling)
        doc_embedding = np.mean(attended, axis=0)
        
        # Classify
        logits = np.matmul(doc_embedding, self.classifier)
        return np.softmax(logits)

# Example usage
classifier = DocumentClassifier(vocab_size=10000, d_model=256, num_classes=3)
text = "This is a sample document for classification."
probabilities = classifier.forward(text)
print(f"Class probabilities: {probabilities}")
```

Slide 12: Attention with Memory Mechanism

Memory-augmented attention extends traditional attention by incorporating a persistent memory bank, allowing the model to access information beyond the current sequence.

```python
class MemoryAugmentedAttention:
    def __init__(self, d_model, num_heads, memory_size=128):
        self.d_model = d_model
        self.num_heads = num_heads
        self.memory_size = memory_size
        
        # Initialize persistent memory
        self.memory = np.random.randn(memory_size, d_model)
        self.memory_key = np.random.randn(d_model, d_model)
        self.memory_value = np.random.randn(d_model, d_model)
        
    def forward(self, queries, keys, values):
        # Transform memory into keys and values
        memory_k = np.matmul(self.memory, self.memory_key)
        memory_v = np.matmul(self.memory, self.memory_value)
        
        # Concatenate memory with input keys and values
        extended_keys = np.concatenate([keys, memory_k], axis=0)
        extended_values = np.concatenate([values, memory_v], axis=0)
        
        # Compute attention scores
        scores = np.matmul(queries, extended_keys.transpose(-2, -1))
        scores = scores / np.sqrt(self.d_model)
        attention = np.softmax(scores, axis=-1)
        
        # Combine with values
        output = np.matmul(attention, extended_values)
        return output
```

Slide 13: Attention Performance Metrics

Quantitative evaluation of attention mechanisms requires specific metrics to measure both attention quality and computational efficiency. This implementation provides tools for performance assessment.

```python
class AttentionMetrics:
    def __init__(self):
        self.attention_entropy_scores = []
        self.coverage_scores = []
        self.computation_times = []
    
    def compute_attention_entropy(self, attention_weights):
        # Compute entropy of attention distribution
        epsilon = 1e-10  # Prevent log(0)
        entropy = -np.sum(
            attention_weights * np.log(attention_weights + epsilon),
            axis=-1
        )
        return np.mean(entropy)
    
    def compute_coverage(self, attention_weights, threshold=0.1):
        # Measure how many source tokens receive significant attention
        significant_attention = (attention_weights > threshold).sum(axis=-1)
        return np.mean(significant_attention)
    
    def evaluate_attention(self, model, test_data):
        import time
        results = {}
        
        for batch in test_data:
            start_time = time.time()
            attention_weights = model.get_attention_weights(batch)
            
            # Record metrics
            self.attention_entropy_scores.append(
                self.compute_attention_entropy(attention_weights)
            )
            self.coverage_scores.append(
                self.compute_coverage(attention_weights)
            )
            self.computation_times.append(time.time() - start_time)
        
        # Aggregate results
        results['mean_entropy'] = np.mean(self.attention_entropy_scores)
        results['mean_coverage'] = np.mean(self.coverage_scores)
        results['avg_computation_time'] = np.mean(self.computation_times)
        
        return results

# Example usage
metrics = AttentionMetrics()
test_weights = np.random.rand(10, 8, 8)  # Batch x Target x Source
entropy = metrics.compute_attention_entropy(test_weights)
coverage = metrics.compute_coverage(test_weights)
print(f"Attention Entropy: {entropy:.4f}")
print(f"Coverage Score: {coverage:.4f}")
```

Slide 14: Results Visualization and Analysis

A comprehensive visualization suite for analyzing attention patterns and model behavior across different attention mechanisms and tasks.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionAnalyzer:
    def __init__(self):
        self.attention_maps = []
        
    def plot_attention_patterns(self, attention_weights, tokens, title):
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis'
        )
        plt.title(title)
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Tokens')
        
    def compare_attention_types(self, self_attention_weights, cross_attention_weights):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sns.heatmap(self_attention_weights, ax=ax1, cmap='coolwarm')
        ax1.set_title('Self-Attention Pattern')
        
        sns.heatmap(cross_attention_weights, ax=ax2, cmap='coolwarm')
        ax2.set_title('Cross-Attention Pattern')
        
        plt.tight_layout()
        
    def plot_attention_stats(self, metrics_over_time):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        ax1.plot(metrics_over_time['entropy'])
        ax1.set_title('Attention Entropy Over Time')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Entropy')
        
        ax2.plot(metrics_over_time['coverage'])
        ax2.set_title('Attention Coverage Over Time')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Coverage Score')
```

Slide 15: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Self-Attention with Relative Position Representations" - [https://arxiv.org/abs/1803.02155](https://arxiv.org/abs/1803.02155)
*   "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" - [https://arxiv.org/abs/1905.09418](https://arxiv.org/abs/1905.09418)
*   "Memory Augmented Self-Attention for Enhanced Representation Learning" - [https://arxiv.org/abs/2006.12195](https://arxiv.org/abs/2006.12195)
*   "Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View" - [https://arxiv.org/abs/2206.04481](https://arxiv.org/abs/2206.04481)

