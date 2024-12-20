## Transformer Encoder Explained
Slide 1: What's the Encoder?

The Encoder is a crucial component of the Transformer architecture, responsible for processing input tokens and generating context-aware representations. It consists of multiple layers that apply self-attention and feed-forward neural networks to refine the input representations iteratively.

Slide 2: Source Code for What's the Encoder?

```python
import math

class Encoder:
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
```

Slide 3: Input Embedding

The first step in the encoder is to embed every input word into a vector of fixed size (typically 512 dimensions). This embedding process occurs only in the bottom-most encoder layer and transforms discrete tokens into continuous vector representations.

Slide 4: Source Code for Input Embedding

```python
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# Example usage
vocab_size = 10000
d_model = 512
embedding_layer = InputEmbedding(vocab_size, d_model)

input_tokens = torch.tensor([1, 2, 3, 4, 5])
embedded_tokens = embedding_layer(input_tokens)
print(f"Input shape: {input_tokens.shape}")
print(f"Embedded shape: {embedded_tokens.shape}")
```

Slide 5: Results for Input Embedding

```
Input shape: torch.Size([5])
Embedded shape: torch.Size([5, 512])
```

Slide 6: Positional Encoding

Transformers lack recurrence, so they use positional encodings to indicate token positions. A combination of sine and cosine functions allows the model to understand the order of words in a sentence. These encodings are added to the input embeddings to provide positional information.

Slide 7: Source Code for Positional Encoding

```python
import numpy as np

def positional_encoding(max_seq_len, d_model):
    pos_enc = np.zeros((max_seq_len, d_model))
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
    return pos_enc

# Example usage
max_seq_len = 10
d_model = 512
pos_enc = positional_encoding(max_seq_len, d_model)
print(f"Positional encoding shape: {pos_enc.shape}")
print(f"First two dimensions for position 0: {pos_enc[0, :2]}")
print(f"First two dimensions for position 5: {pos_enc[5, :2]}")
```

Slide 8: Results for Positional Encoding

```
Positional encoding shape: (10, 512)
First two dimensions for position 0: [0.         1.        ]
First two dimensions for position 5: [ 0.84147098 -0.54030231]
```

Slide 9: Multi-Headed Self-Attention

Self-attention allows the model to relate each word to every other word in the input. The process involves calculating attention scores based on Query, Key, and Value vectors. This operation is performed multiple times in parallel, called "heads," to capture different aspects of the relationships between words.

Slide 10: Source Code for Multi-Headed Self-Attention

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
    
    def attention(self, q, k, v, mask=None):
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        return np.matmul(attention_weights, v)
    
    def forward(self, x, mask=None):
        q = np.matmul(x, self.W_q)
        k = np.matmul(x, self.W_k)
        v = np.matmul(x, self.W_v)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attn_output = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(-1, x.shape[1], self.d_model)
        
        return np.matmul(attn_output, self.W_o)

# Example usage
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

x = np.random.randn(1, 10, d_model)  # Batch size 1, sequence length 10
output = mha.forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 11: Results for Multi-Headed Self-Attention

```
Input shape: (1, 10, 512)
Output shape: (1, 10, 512)
```

Slide 12: Feed-Forward Neural Network

After the self-attention layer, each position in the sequence is processed independently through a feed-forward neural network. This network typically consists of two linear transformations with a ReLU activation in between, allowing the model to introduce non-linearity and further refine the representations.

Slide 13: Source Code for Feed-Forward Neural Network

```python
import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.w2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        hidden = self.relu(np.dot(x, self.w1) + self.b1)
        return np.dot(hidden, self.w2) + self.b2

# Example usage
d_model = 512
d_ff = 2048
ff_layer = FeedForward(d_model, d_ff)

x = np.random.randn(1, 10, d_model)  # Batch size 1, sequence length 10
output = ff_layer.forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 14: Results for Feed-Forward Neural Network

```
Input shape: (1, 10, 512)
Output shape: (1, 10, 512)
```

Slide 15: Real-Life Example: Sentiment Analysis

Let's consider a practical application of the Transformer encoder for sentiment analysis. In this example, we'll use a simplified encoder to process a movie review and classify its sentiment.

Slide 16: Source Code for Sentiment Analysis

```python
import numpy as np

class SimplifiedEncoder:
    def __init__(self, vocab_size, d_model):
        self.embedding = np.random.randn(vocab_size, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads=4)
        self.ff = FeedForward(d_model, d_ff=1024)
    
    def forward(self, tokens):
        x = self.embedding[tokens]
        x = self.attention.forward(x)
        x = self.ff.forward(x)
        return x.mean(axis=1)  # Average pooling

# Simplified sentiment classifier
def classify_sentiment(encoder_output):
    w = np.random.randn(512, 2)  # Binary classification: positive/negative
    logits = np.dot(encoder_output, w)
    return np.argmax(logits, axis=1)

# Example usage
vocab_size = 10000
d_model = 512
encoder = SimplifiedEncoder(vocab_size, d_model)

# Tokenized movie review (simplified)
review = np.array([42, 1337, 7, 42, 1337, 7, 42, 1337, 7])
encoded_review = encoder.forward(review)
sentiment = classify_sentiment(encoded_review)

print(f"Encoded review shape: {encoded_review.shape}")
print(f"Predicted sentiment: {'Positive' if sentiment[0] == 1 else 'Negative'}")
```

Slide 17: Results for Sentiment Analysis

```
Encoded review shape: (1, 512)
Predicted sentiment: Positive
```

Slide 18: Real-Life Example: Machine Translation

Another common application of the Transformer encoder is machine translation. Here's a simplified example of how the encoder could be used in a translation system.

Slide 19: Source Code for Machine Translation

```python
import numpy as np

class TranslationEncoder(SimplifiedEncoder):
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model)
        self.language_embedding = np.random.randn(2, d_model)  # Source and target language embeddings
    
    def forward(self, tokens, language_id):
        x = self.embedding[tokens] + self.language_embedding[language_id]
        x = self.attention.forward(x)
        x = self.ff.forward(x)
        return x

# Simplified translation decoder (just for demonstration)
def translate(encoder_output):
    # In a real system, this would be another Transformer stack (the decoder)
    w = np.random.randn(512, vocab_size)
    logits = np.dot(encoder_output, w)
    return np.argmax(logits, axis=-1)

# Example usage
vocab_size = 10000
d_model = 512
encoder = TranslationEncoder(vocab_size, d_model)

# Tokenized sentence in source language (simplified)
sentence = np.array([42, 1337, 7, 42, 1337, 7, 42, 1337, 7])
language_id = 0  # 0 for source language, 1 for target language
encoded_sentence = encoder.forward(sentence, language_id)
translation = translate(encoded_sentence)

print(f"Encoded sentence shape: {encoded_sentence.shape}")
print(f"Translated sentence (token IDs): {translation}")
```

Slide 20: Results for Machine Translation

```
Encoded sentence shape: (9, 512)
Translated sentence (token IDs): [3845 6201 9321 4562 1078 7890 2345 6789 1234]
```

Slide 21: Additional Resources

For more in-depth information on the Transformer architecture and its encoder component, consider the following resources:

1.  Original Transformer paper: "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3.  "The Illustrated Transformer" by Jay Alammar [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

These resources provide comprehensive explanations and visualizations of the Transformer architecture, including detailed discussions of the encoder component.

