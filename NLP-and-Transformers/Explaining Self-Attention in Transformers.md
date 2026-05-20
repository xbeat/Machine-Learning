## Explaining Self-Attention in Transformers
Slide 1: Self-Attention in Transformers

Self-attention is a key mechanism in Transformer models, allowing the model to weigh the importance of different parts of the input sequence when processing each element. It enables the model to capture contextual relationships within the data.

```python
import numpy as np

def self_attention(query, key, value):
    # Compute attention scores
    scores = np.dot(query, key.T) / np.sqrt(key.shape[1])
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute weighted sum of values
    output = np.dot(weights, value)
    
    return output

# Example usage
seq_len, d_model = 4, 8
query = np.random.randn(seq_len, d_model)
key = np.random.randn(seq_len, d_model)
value = np.random.randn(seq_len, d_model)

attention_output = self_attention(query, key, value)
print("Attention output shape:", attention_output.shape)
```

Slide 2: Components of Self-Attention

Self-attention involves three main components: queries, keys, and values. These are derived from the input sequence through linear transformations. The interaction between these components determines how information is aggregated across the sequence.

```python
import numpy as np

class SelfAttention:
    def __init__(self, d_model):
        self.d_model = d_model
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)

    def forward(self, X):
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        return Q, K, V

# Example usage
d_model = 64
seq_len = 10
X = np.random.randn(seq_len, d_model)

attention = SelfAttention(d_model)
Q, K, V = attention.forward(X)

print("Query shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)
```

Slide 3: Attention Scores and Weights

Attention scores are computed by comparing queries with keys. These scores are then normalized using softmax to obtain attention weights. The weights determine how much each value contributes to the final output for a given position.

```python
import numpy as np

def compute_attention_weights(Q, K):
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(K.shape[1])
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    return weights

# Example usage
seq_len, d_model = 5, 8
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)

weights = compute_attention_weights(Q, K)
print("Attention weights shape:", weights.shape)
print("Attention weights (first row):", weights[0])
```

Slide 4: Scaled Dot-Product Attention

Scaled dot-product attention is the core operation in self-attention. It involves computing attention scores, applying softmax, and then using the resulting weights to aggregate values. The scaling factor (√d\_k) helps stabilize gradients during training.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = K.shape[1]
    
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute weighted sum of values
    output = np.dot(weights, V)
    
    return output, weights

# Example usage
seq_len, d_model = 4, 8
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
```

Slide 5: Multi-Head Attention

Multi-head attention extends self-attention by applying multiple attention operations in parallel. This allows the model to capture different types of relationships within the data. The outputs from different heads are concatenated and linearly transformed to produce the final result.

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, X):
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        output, _ = scaled_dot_product_attention(Q, K, V)
        output = output.transpose(0, 2, 1, 3).reshape(-1, seq_len, self.d_model)
        
        return np.dot(output, self.W_o)

# Example usage
d_model, num_heads = 64, 8
seq_len = 10
X = np.random.randn(1, seq_len, d_model)

mha = MultiHeadAttention(d_model, num_heads)
output = mha.forward(X)

print("Multi-head attention output shape:", output.shape)
```

Slide 6: Masked Self-Attention

Masked self-attention is a variant used in decoder layers to prevent the model from attending to future positions. It applies a mask to the attention scores before softmax, effectively setting the weights for future positions to zero.

```python
import numpy as np

def masked_self_attention(Q, K, V, mask):
    d_k = K.shape[-1]
    
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply mask
    scores = np.where(mask == 0, -1e9, scores)
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Compute weighted sum of values
    output = np.dot(weights, V)
    
    return output, weights

# Example usage
seq_len, d_model = 5, 8
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

# Create a lower triangular mask
mask = np.tril(np.ones((seq_len, seq_len)))

output, weights = masked_self_attention(Q, K, V, mask)
print("Masked self-attention output shape:", output.shape)
print("Masked attention weights:\n", weights)
```

Slide 7: Positional Encoding

Positional encoding is crucial in Transformers to provide information about the relative or absolute position of tokens in the sequence. It's usually added to the input embeddings before self-attention layers.

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    
    return pe

# Example usage
max_len, d_model = 100, 64
pe = positional_encoding(max_len, d_model)

plt.figure(figsize=(10, 6))
plt.imshow(pe, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.show()
```

Slide 8: Self-Attention in Encoder Layer

In a Transformer encoder layer, self-attention is followed by a feed-forward neural network. Layer normalization and residual connections are applied after each sub-layer to facilitate training and information flow.

```python
import numpy as np

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x):
        # Self-attention sub-layer
        attn_output = self.self_attention.forward(x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward sub-layer
        ffn_output = self.ffn.forward(x)
        x = self.layer_norm2(x + ffn_output)
        
        return x

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)

    def forward(self, x):
        return np.dot(np.maximum(0, np.dot(x, self.W1)), self.W2)

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Example usage
d_model, num_heads, d_ff = 64, 8, 256
seq_len = 10
x = np.random.randn(1, seq_len, d_model)

encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
output = encoder_layer.forward(x)

print("Encoder layer output shape:", output.shape)
```

Slide 9: Self-Attention in Decoder Layer

The decoder layer in a Transformer includes masked self-attention, cross-attention (attending to encoder outputs), and a feed-forward network. The masked self-attention prevents the decoder from looking at future positions during training.

```python
import numpy as np

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, mask):
        # Masked self-attention sub-layer
        attn_output = self.masked_self_attention.forward(x, mask=mask)
        x = self.layer_norm1(x + attn_output)
        
        # Cross-attention sub-layer
        cross_attn_output = self.cross_attention.forward(x, encoder_output)
        x = self.layer_norm2(x + cross_attn_output)
        
        # Feed-forward sub-layer
        ffn_output = self.ffn.forward(x)
        x = self.layer_norm3(x + ffn_output)
        
        return x

# Example usage
d_model, num_heads, d_ff = 64, 8, 256
seq_len = 10
x = np.random.randn(1, seq_len, d_model)
encoder_output = np.random.randn(1, seq_len, d_model)
mask = np.tril(np.ones((seq_len, seq_len)))

decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
output = decoder_layer.forward(x, encoder_output, mask)

print("Decoder layer output shape:", output.shape)
```

Slide 10: Real-Life Example: Language Translation

Self-attention in Transformers has revolutionized machine translation. It allows the model to capture long-range dependencies and context, resulting in more accurate and fluent translations.

```python
import numpy as np

def translate(input_sequence, encoder, decoder, max_len):
    encoder_output = encoder(input_sequence)
    
    output_sequence = [START_TOKEN]
    for _ in range(max_len):
        decoder_input = np.array(output_sequence)
        decoder_output = decoder(decoder_input, encoder_output)
        
        next_token = np.argmax(decoder_output[-1])
        output_sequence.append(next_token)
        
        if next_token == END_TOKEN:
            break
    
    return output_sequence[1:-1]  # Remove start and end tokens

# Example usage (simplified)
input_sequence = np.array([1, 2, 3, 4, 5])  # Tokenized input sentence
encoder = Encoder()
decoder = Decoder()
translated_sequence = translate(input_sequence, encoder, decoder, max_len=50)

print("Input sequence:", input_sequence)
print("Translated sequence:", translated_sequence)
```

Slide 11: Real-Life Example: Text Summarization

Self-attention mechanisms in Transformers are particularly effective for text summarization tasks. They can identify and focus on the most important parts of a long document to generate concise and informative summaries.

```python
import numpy as np

def summarize(document, model, max_summary_len):
    document_embedding = model.encode(document)
    
    summary = [START_TOKEN]
    for _ in range(max_summary_len):
        summary_embedding = model.encode(summary)
        
        attention_output = model.self_attention(summary_embedding, document_embedding)
        next_token = model.generate_next_token(attention_output)
        
        summary.append(next_token)
        
        if next_token == END_TOKEN:
            break
    
    return model.decode(summary[1:-1])  # Remove start and end tokens

# Example usage (simplified)
document = "Long document text..."
model = SummarizationModel()
max_summary_len = 100
summary = summarize(document, model, max_summary_len)

print("Original document:", document)
print("Generated summary:", summary)
```

Slide 12: Self-Attention in Computer Vision

While initially designed for natural language processing, self-attention has found applications in computer vision tasks. It allows models to focus on relevant parts of an image, improving performance in tasks like image classification and object detection.

```python
import numpy as np

def image_self_attention(image, num_patches):
    # Split image into patches
    patches = split_image_into_patches(image, num_patches)
    
    # Compute self-attention on patches
    attention_weights = compute_attention_weights(patches)
    
    # Apply attention weights to patches
    attended_patches = np.sum(patches[:, np.newaxis] * attention_weights, axis=2)
    
    # Reconstruct image from attended patches
    attended_image = reconstruct_image_from_patches(attended_patches)
    
    return attended_image

def split_image_into_patches(image, num_patches):
    # Implementation details omitted for brevity
    return patches

def compute_attention_weights(patches):
    # Implementation details omitted for brevity
    return attention_weights

def reconstruct_image_from_patches(patches):
    # Implementation details omitted for brevity
    return reconstructed_image

# Example usage
image = np.random.rand(224, 224, 3)  # Random RGB image
num_patches = 16
attended_image = image_self_attention(image, num_patches)

print("Original image shape:", image.shape)
print("Attended image shape:", attended_image.shape)
```

Slide 13: Attention Visualization

Visualizing attention weights can provide insights into how the model processes information. This can be particularly useful for interpreting model decisions and debugging.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(text, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')

    ax.set_xticks(np.arange(len(text)))
    ax.set_yticks(np.arange(len(text)))
    ax.set_xticklabels(text)
    ax.set_yticklabels(text)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(text)):
        for j in range(len(text)):
            text = ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title("Attention Weights")
    fig.tight_layout()
    plt.colorbar(im)
    plt.show()

# Example usage
text = ["I", "love", "machine", "learning"]
attention_weights = np.random.rand(len(text), len(text))
visualize_attention(text, attention_weights)
```

Slide 14: Efficiency Improvements in Self-Attention

Recent research has focused on improving the efficiency of self-attention, especially for long sequences. Techniques like sparse attention and linear attention aim to reduce the quadratic complexity of standard self-attention.

```python
import numpy as np

def linear_attention(Q, K, V):
    Q_sum = np.sum(Q, axis=1, keepdims=True)
    K_sum = np.sum(K, axis=1, keepdims=True)
    
    KV = np.dot(K.T, V)
    
    attention = np.dot(Q, KV)
    normalizer = np.dot(Q_sum, K_sum.T)
    
    return attention / normalizer

# Example usage
seq_len, d_model = 1000, 64
Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output = linear_attention(Q, K, V)
print("Linear attention output shape:", output.shape)
```

Slide 15: Additional Resources

For more in-depth information on self-attention and Transformers, consider exploring these resources:

1. "Attention Is All You Need" paper (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "Efficient Transformers: A Survey" (Tay et al., 2020): [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
3. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020): [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

These papers provide comprehensive overviews of self-attention mechanisms, Transformer architectures, and their applications in various domains.

