## Attention Mechanism in Transformer Models
Slide 1: Attention Mechanism Fundamentals

The attention mechanism forms the core of transformer architectures by computing relevance scores between input elements. It enables models to dynamically focus on different parts of the input sequence by learning weights that represent the importance of relationships between tokens.

```python
import numpy as np

def attention_scores(query, key, value):
    # Calculate attention scores using scaled dot-product attention
    # query, key, value shapes: (sequence_length, embedding_dim)
    
    d_k = query.shape[-1]
    scores = np.dot(query, key.T) / np.sqrt(d_k)  # Scale by sqrt(d_k)
    attention_weights = softmax(scores)  # Apply softmax
    output = np.dot(attention_weights, value)
    
    return output, attention_weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example usage
seq_len, d_model = 4, 8
query = np.random.randn(seq_len, d_model)
key = np.random.randn(seq_len, d_model)
value = np.random.randn(seq_len, d_model)

output, weights = attention_scores(query, key, value)
print(f"Attention Weights Shape: {weights.shape}")
print(f"Output Shape: {output.shape}")
```

Slide 2: Multi-Head Attention Implementation

Multi-head attention allows the model to jointly attend to information from different representation subspaces, enabling the capture of various types of relationships between tokens. Each head learns distinct attention patterns, enriching the model's understanding.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices for Q, K, V projections
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, query, key, value):
        # Linear projections
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        attention_weights = softmax(scores)
        attention_output = np.matmul(attention_weights, V)
        
        # Combine heads and apply final linear transformation
        output = np.dot(attention_output.transpose(0, 2, 1, 3).reshape(-1, self.d_model), 
                       self.W_o)
        return output, attention_weights
```

Slide 3: Positional Encoding Design

Positional encoding injects information about token positions into the sequence since the attention mechanism is inherently position-agnostic. Using sinusoidal functions creates unique patterns for each position while maintaining relative distance relationships.

```python
def positional_encoding(max_seq_length, d_model):
    position = np.arange(max_seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((max_seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    # Add batch dimension
    pos_encoding = pos_encoding[np.newaxis, :, :]
    return pos_encoding

# Example usage
max_seq_length, d_model = 100, 512
pe = positional_encoding(max_seq_length, d_model)
print(f"Positional Encoding Shape: {pe.shape}")

# Visualize first few dimensions
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(pe[0, :, 4:8])
plt.legend([f'dim_{i}' for i in range(4, 8)])
plt.title('Positional Encoding Patterns')
```

Slide 4: Layer Normalization Implementation

Layer normalization stabilizes neural network training by normalizing activations across features. In transformers, it's applied after attention and feed-forward layers, ensuring consistent scale of activations and improving gradient flow throughout the network.

```python
class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps

    def forward(self, x):
        # Calculate mean and variance along last axis
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        
        # Normalize and scale
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta

# Example usage
layer_norm = LayerNorm(512)
sample_activations = np.random.randn(32, 10, 512)  # (batch, seq_len, features)
normalized_output = layer_norm.forward(sample_activations)
print(f"Output shape: {normalized_output.shape}")
print(f"Mean: {np.mean(normalized_output):.6f}")
print(f"Std: {np.std(normalized_output):.6f}")
```

Slide 5: Masked Self-Attention Implementation

Masked self-attention prevents the decoder from attending to future tokens during training, maintaining the autoregressive property essential for sequence generation. This is achieved by applying a mask to the attention scores before softmax.

```python
def masked_self_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply mask (if provided)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Apply softmax
    attention_weights = softmax(scores)
    
    # Compute output
    output = np.matmul(attention_weights, value)
    return output, attention_weights

# Create causal mask for decoder self-attention
def create_causal_mask(size):
    mask = np.triu(np.ones((size, size)), k=1).astype('uint8')
    return mask == 0

# Example usage
seq_len, d_model = 8, 64
query = np.random.randn(1, seq_len, d_model)
key = value = query
mask = create_causal_mask(seq_len)

output, weights = masked_self_attention(query, key, value, mask)
print(f"Attention mask shape: {mask.shape}")
print(f"Output shape: {output.shape}")
```

Slide 6: Cross-Attention Mechanism

Cross-attention enables the decoder to focus on relevant parts of the encoder's output while generating each token. This mechanism forms a crucial bridge between encoder and decoder, allowing the model to incorporate source sequence information during generation.

```python
class CrossAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize projection matrices
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
        
    def forward(self, decoder_state, encoder_output):
        # Project queries from decoder state
        Q = np.dot(decoder_state, self.W_q)
        # Project keys and values from encoder output
        K = np.dot(encoder_output, self.W_k)
        V = np.dot(encoder_output, self.W_v)
        
        # Compute scaled dot-product attention
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attention_weights = softmax(scores)
        
        # Apply attention weights to values
        context_vector = np.matmul(attention_weights, V)
        output = np.dot(context_vector, self.W_o)
        
        return output, attention_weights

# Example usage
batch_size, decoder_len, encoder_len = 2, 4, 6
d_model = 64

decoder_state = np.random.randn(batch_size, decoder_len, d_model)
encoder_output = np.random.randn(batch_size, encoder_len, d_model)

cross_attention = CrossAttention(d_model, num_heads=8)
output, weights = cross_attention.forward(decoder_state, encoder_output)
print(f"Cross-attention output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 7: Feed-Forward Neural Network Layer

The feed-forward network in transformer architectures processes each position independently, applying two linear transformations with a ReLU activation in between. This component allows the model to introduce non-linearity and transform the attention mechanism outputs.

```python
class FeedForward:
    def __init__(self, d_model, d_ff=2048):
        self.w1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.w2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # First linear transformation
        hidden = np.dot(x, self.w1) + self.b1
        # ReLU activation
        hidden = self.relu(hidden)
        # Second linear transformation
        output = np.dot(hidden, self.w2) + self.b2
        return output

# Example usage
ff_layer = FeedForward(512, 2048)
sample_input = np.random.randn(32, 10, 512)  # (batch, seq_len, d_model)
output = ff_layer.forward(sample_input)
print(f"Feed-forward output shape: {output.shape}")
```

Slide 8: Complete Encoder Layer Implementation

The encoder layer combines multi-head attention with feed-forward processing, employing residual connections and layer normalization. This structure allows for deep architectures while maintaining stable training dynamics.

```python
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
        
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)
    
    def forward(self, x, mask=None):
        # Self attention block
        attention_output, _ = self.self_attention.forward(x, x, x)
        attention_output = self.dropout(attention_output)
        x = self.norm1.forward(x + attention_output)  # Add & Norm
        
        # Feed forward block
        ff_output = self.feed_forward.forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2.forward(x + ff_output)  # Add & Norm
        
        return x

# Example usage
encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
sample_input = np.random.randn(32, 10, 512)
output = encoder_layer.forward(sample_input)
print(f"Encoder layer output shape: {output.shape}")
```

Slide 9: Decoder Layer with Masked Attention

The decoder layer incorporates three sub-layers: masked self-attention, cross-attention, and feed-forward processing. The masking ensures autoregressive generation while cross-attention enables contextual understanding from the encoder.

```python
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, encoder_output, look_ahead_mask, padding_mask):
        # Masked self-attention
        attn1, _ = self.self_attention.forward(x, x, x)
        attn1 = self.dropout(attn1)
        out1 = self.norm1.forward(x + attn1)
        
        # Cross-attention
        attn2, _ = self.cross_attention.forward(out1, encoder_output, encoder_output)
        attn2 = self.dropout(attn2)
        out2 = self.norm2.forward(out1 + attn2)
        
        # Feed forward
        ffn_output = self.feed_forward.forward(out2)
        ffn_output = self.dropout(ffn_output)
        out3 = self.norm3.forward(out2 + ffn_output)
        
        return out3
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)

# Example usage
decoder_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048)
decoder_input = np.random.randn(32, 10, 512)
encoder_output = np.random.randn(32, 15, 512)
look_ahead_mask = create_causal_mask(10)
padding_mask = None

output = decoder_layer.forward(decoder_input, encoder_output, look_ahead_mask, padding_mask)
print(f"Decoder layer output shape: {output.shape}")
```

Slide 10: Attention Visualization Implementation

Attention visualization helps understand how the model processes relationships between tokens. This implementation creates heatmaps of attention weights, providing insights into the model's decision-making process during sequence processing.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens_in, tokens_out=None):
    if tokens_out is None:
        tokens_out = tokens_in
    
    # Create figure and axes
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights,
                xticklabels=tokens_in,
                yticklabels=tokens_out,
                cmap='viridis',
                annot=True,
                fmt='.2f')
    
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title('Attention Weights Visualization')
    return plt

# Example usage with sample data
sequence = ["The", "quick", "brown", "fox", "jumps"]
attn_weights = np.random.rand(len(sequence), len(sequence))
attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

fig = visualize_attention(attn_weights, sequence)
plt.close()  # Close to prevent display in notebook

print(f"Generated attention visualization for {len(sequence)} tokens")
```

Slide 11: Real-world Application - Machine Translation

Implementation of a simplified transformer-based translation system showcasing the practical application of attention mechanisms in sequence-to-sequence tasks. This example includes data preprocessing and model training components.

```python
class TranslatorModel:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512):
        self.encoder = EncoderLayer(d_model, num_heads=8, d_ff=2048)
        self.decoder = DecoderLayer(d_model, num_heads=8, d_ff=2048)
        self.src_embed = np.random.randn(src_vocab_size, d_model)
        self.tgt_embed = np.random.randn(tgt_vocab_size, d_model)
        self.pos_encoding = positional_encoding(1000, d_model)
        
    def encode(self, src_tokens):
        # Embed source tokens and add positional encoding
        src_embedded = self.src_embed[src_tokens]
        src_embedded += self.pos_encoding[:, :src_embedded.shape[1], :]
        return self.encoder.forward(src_embedded)
    
    def decode(self, tgt_tokens, encoder_output):
        # Embed target tokens and add positional encoding
        tgt_embedded = self.tgt_embed[tgt_tokens]
        tgt_embedded += self.pos_encoding[:, :tgt_embedded.shape[1], :]
        
        # Create causal mask
        mask = create_causal_mask(tgt_tokens.shape[1])
        
        return self.decoder.forward(tgt_embedded, encoder_output, mask, None)

# Example usage
src_vocab_size, tgt_vocab_size = 5000, 6000
translator = TranslatorModel(src_vocab_size, tgt_vocab_size)

# Simulate translation
src_sentence = np.array([[1, 4, 2, 3, 0]])  # Dummy token indices
tgt_sentence = np.array([[1, 5, 3, 0]])     # Dummy token indices

encoder_output = translator.encode(src_sentence)
decoder_output = translator.decode(tgt_sentence, encoder_output)

print(f"Encoder output shape: {encoder_output.shape}")
print(f"Decoder output shape: {decoder_output.shape}")
```

Slide 12: Results Analysis for Translation Model

A comprehensive evaluation of the translation model's performance, including attention pattern analysis and quality metrics calculation to assess the effectiveness of the attention mechanism.

```python
def analyze_translation_results(model, src_text, pred_text, attn_weights):
    # Calculate BLEU score (simplified version)
    def calculate_bleu(reference, hypothesis):
        matches = sum(r == h for r, h in zip(reference, hypothesis))
        return matches / len(hypothesis) if len(hypothesis) > 0 else 0
    
    # Analyze attention patterns
    def analyze_attention_patterns(attn_weights):
        avg_attention = np.mean(attn_weights, axis=0)
        max_attention = np.max(attn_weights, axis=0)
        entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-9), axis=-1)
        return avg_attention, max_attention, entropy

    # Example metrics calculation
    bleu_score = calculate_bleu(src_text, pred_text)
    avg_attn, max_attn, entropy = analyze_attention_patterns(attn_weights)
    
    print(f"Translation Quality Metrics:")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Average Attention Score: {np.mean(avg_attn):.4f}")
    print(f"Attention Entropy: {np.mean(entropy):.4f}")
    
    return {
        'bleu': bleu_score,
        'avg_attention': avg_attn,
        'attention_entropy': entropy
    }

# Example usage with dummy data
src_text = [1, 2, 3, 4, 5]
pred_text = [1, 2, 3, 4]
dummy_attn_weights = np.random.rand(4, 5)  # (tgt_len, src_len)
dummy_attn_weights = dummy_attn_weights / dummy_attn_weights.sum(axis=-1, keepdims=True)

metrics = analyze_translation_results(None, src_text, pred_text, dummy_attn_weights)
```

Slide 13: Real-world Application - Text Summarization

Implementing attention-based text summarization demonstrates how transformer architectures can identify and extract key information from longer sequences to generate concise summaries while maintaining semantic coherence.

```python
class SummarizationModel:
    def __init__(self, vocab_size, d_model=512, max_length=1024):
        self.encoder = EncoderLayer(d_model, num_heads=8, d_ff=2048)
        self.decoder = DecoderLayer(d_model, num_heads=8, d_ff=2048)
        self.embeddings = np.random.randn(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.output_projection = np.random.randn(d_model, vocab_size)
        
    def summarize(self, input_ids, max_summary_length=150):
        # Encode input sequence
        embedded_input = self.embeddings[input_ids]
        embedded_input += self.pos_encoding[:, :input_ids.shape[1], :]
        encoder_output = self.encoder.forward(embedded_input)
        
        # Iterative decoding
        summary_ids = [self.get_start_token()]
        for i in range(max_summary_length):
            decoder_input = np.array([summary_ids])
            decoder_embedding = self.embeddings[decoder_input]
            decoder_output = self.decoder.forward(
                decoder_embedding,
                encoder_output,
                create_causal_mask(len(summary_ids)),
                None
            )
            
            # Project to vocabulary
            logits = np.dot(decoder_output[:, -1], self.output_projection)
            next_token = np.argmax(logits, axis=-1)
            
            if next_token == self.get_end_token():
                break
                
            summary_ids.append(next_token)
            
        return np.array(summary_ids)
    
    def get_start_token(self):
        return 1  # Assuming 1 is START token
        
    def get_end_token(self):
        return 2  # Assuming 2 is END token

# Example usage
vocab_size = 10000
summarizer = SummarizationModel(vocab_size)

# Sample input text (as token ids)
input_text = np.array([[45, 67, 89, 123, 456, 789, 234, 567, 890]])
summary = summarizer.summarize(input_text)

print(f"Input sequence length: {input_text.shape[1]}")
print(f"Generated summary length: {len(summary)}")
```

Slide 14: Benchmarking Attention Performance

Analysis of computational complexity and performance characteristics of different attention mechanisms, including implementation of optimization techniques for improved efficiency in real-world applications.

```python
class AttentionBenchmark:
    def __init__(self):
        self.timing_results = {}
        
    def benchmark_attention_variants(self, sequence_lengths, d_model=512, num_heads=8):
        results = {
            'standard': [],
            'linear': [],
            'sparse': []
        }
        
        for seq_len in sequence_lengths:
            # Generate random input data
            query = np.random.randn(1, seq_len, d_model)
            key = value = query
            
            # Benchmark standard attention
            start_time = time.time()
            self._standard_attention(query, key, value)
            results['standard'].append(time.time() - start_time)
            
            # Benchmark linear attention
            start_time = time.time()
            self._linear_attention(query, key, value)
            results['linear'].append(time.time() - start_time)
            
            # Benchmark sparse attention
            start_time = time.time()
            self._sparse_attention(query, key, value, block_size=64)
            results['sparse'].append(time.time() - start_time)
            
        return results
    
    def _standard_attention(self, q, k, v):
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        weights = softmax(scores)
        return np.matmul(weights, v)
    
    def _linear_attention(self, q, k, v):
        # Simplified linear attention implementation
        q_prime = np.exp(q)
        k_prime = np.exp(k)
        kv = np.matmul(k_prime.transpose(-2, -1), v)
        denom = np.sum(k_prime, axis=-2)
        return np.matmul(q_prime, kv) / denom[..., None]
    
    def _sparse_attention(self, q, k, v, block_size):
        # Blocked sparse attention implementation
        seq_len = q.shape[1]
        num_blocks = seq_len // block_size
        
        output = np.zeros_like(q)
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            block_q = q[:, start_idx:end_idx]
            block_output = self._standard_attention(block_q, k, v)
            output[:, start_idx:end_idx] = block_output
            
        return output

# Example usage
benchmark = AttentionBenchmark()
sequence_lengths = [128, 256, 512, 1024]
results = benchmark.benchmark_attention_variants(sequence_lengths)

print("Performance Results (seconds):")
for variant, timings in results.items():
    print(f"\n{variant.capitalize()} Attention:")
    for seq_len, timing in zip(sequence_lengths, timings):
        print(f"Sequence length {seq_len}: {timing:.4f}s")
```

Slide 15: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Efficient Transformers: A Survey" - [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
*   "Longformer: The Long-Document Transformer" - [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
*   "Reformer: The Efficient Transformer" - [https://arxiv.org/abs/2001.04451](https://arxiv.org/abs/2001.04451)
*   "Linformer: Self-Attention with Linear Complexity" - [https://arxiv.org/abs/2006.04768](https://arxiv.org/abs/2006.04768)

