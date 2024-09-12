## Transformer Encoder Explained with Python
Slide 1: Introduction to Transformers: The Encoder

The Transformer architecture, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing. At its core is the encoder, which processes input sequences and captures their contextual relationships. Let's explore the encoder's components using Python.

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
    
    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

# Example usage
encoder = TransformerEncoder(d_model=512, nhead=8, num_layers=6)
input_seq = torch.rand(10, 32, 512)  # (seq_len, batch_size, d_model)
output = encoder(input_seq)
print(output.shape)  # torch.Size([10, 32, 512])
```

Slide 2: Input Embedding

The first step in the encoder is converting input tokens into dense vector representations. This is done through an embedding layer, which maps each token to a high-dimensional vector.

```python
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.embedding(x) * (self.d_model ** 0.5)

# Example usage
vocab_size = 10000
d_model = 512
embedding_layer = InputEmbedding(vocab_size, d_model)
input_ids = torch.randint(0, vocab_size, (32, 10))  # (batch_size, seq_len)
embedded = embedding_layer(input_ids)
print(embedded.shape)  # torch.Size([32, 10, 512])
```

Slide 3: Positional Encoding

To inject information about token positions, we add positional encodings to the embedded inputs. These encodings use sine and cosine functions of different frequencies.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Example usage
d_model = 512
pos_encoder = PositionalEncoding(d_model)
input_seq = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
encoded = pos_encoder(input_seq)
print(encoded.shape)  # torch.Size([10, 32, 512])
```

Slide 4: Multi-Head Attention: Query, Key, Value Projections

The core of the Transformer is the multi-head attention mechanism. It starts by projecting the input into query, key, and value vectors for each attention head.

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        return q, k, v

# Example usage
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)
x = torch.rand(32, 10, d_model)  # (batch_size, seq_len, d_model)
q, k, v = mha(x, x, x)
print(q.shape, k.shape, v.shape)  # All: torch.Size([32, 8, 10, 64])
```

Slide 5: Multi-Head Attention: Scaled Dot-Product Attention

After projecting inputs, we compute attention scores using scaled dot-product attention. This determines how much focus to put on different parts of the input sequence.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
q = torch.rand(32, 8, 10, 64)  # (batch_size, num_heads, seq_len, head_dim)
k = torch.rand(32, 8, 10, 64)
v = torch.rand(32, 8, 10, 64)
output, weights = scaled_dot_product_attention(q, k, v)
print(output.shape, weights.shape)
# output: torch.Size([32, 8, 10, 64])
# weights: torch.Size([32, 8, 10, 10])
```

Slide 6: Multi-Head Attention: Combining Heads

After computing attention for each head, we concatenate the results and project them back to the original dimension.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(attn_output)

# Example usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.rand(32, 10, 512)  # (batch_size, seq_len, d_model)
output = mha(x, x, x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 7: Feed-Forward Network

After multi-head attention, each position in the sequence is processed independently through a feed-forward network, consisting of two linear transformations with a ReLU activation in between.

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Example usage
d_model = 512
d_ff = 2048
ff_layer = FeedForward(d_model, d_ff)
x = torch.rand(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = ff_layer(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 8: Layer Normalization

Layer normalization is applied after each sub-layer (multi-head attention and feed-forward network) to stabilize the activations and improve training dynamics.

```python
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
x = torch.rand(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = encoder_layer(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 9: Residual Connections

Residual connections are used around each sub-layer to facilitate gradient flow during training and enable the network to learn deeper representations.

```python
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual connection for self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Residual connection for feed-forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

# Example usage
encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)
x = torch.rand(32, 10, 512)  # (batch_size, seq_len, d_model)
output = encoder_layer(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 10: Stacking Encoder Layers

The complete encoder consists of multiple stacked encoder layers. Each layer processes the output of the previous layer, allowing the model to learn increasingly complex representations.

```python
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers)
x = torch.rand(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = encoder(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 11: Real-Life Example: Text Classification

Let's use our Transformer encoder for a text classification task, such as sentiment analysis on movie reviews.

```python
import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
num_classes = 2  # Positive or negative sentiment

model = SentimentClassifier(vocab_size, d_model, num_heads, d_ff, num_layers, num_classes)
input_ids = torch.randint(0, vocab_size, (32, 100))  # (batch_size, seq_len)
output = model(input_ids)
print(output.shape)  # torch.Size([32, 2])
```

Slide 12: Real-Life Example: Named Entity Recognition

Another application of the Transformer encoder is Named Entity Recognition (NER), where we identify and classify named entities in text.

```python
import torch
import torch.nn as nn

class NERModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_entities):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers)
        self.entity_classifier = nn.Linear(d_model, num_entities)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.entity_classifier(x)

# Example usage
vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
num_entities = 9  # e.g., O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

model = NERModel(vocab_size, d_model, num_heads, d_ff, num_layers, num_entities)
input_ids = torch.randint(0, vocab_size, (32, 50))  # (batch_size, seq_len)
output = model(input_ids)
print(output.shape)  # torch.Size([32, 50, 9])
```

Slide 13: Attention Visualization

Visualizing attention weights can provide insights into how the model processes input sequences. Here's a simple example of how to extract and visualize attention weights from a single head.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='YlGnBu')
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

# Assuming we have attention weights and tokens
attention_weights = torch.rand(10, 10)  # (seq_len, seq_len)
tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

visualize_attention(attention_weights.numpy(), tokens)
```

Slide 14: Handling Variable Length Sequences

In practice, input sequences often have different lengths. We can use padding and masking to handle this variability in the Transformer encoder.

```python
import torch
import torch.nn.functional as F

def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

class TransformerEncoderWithMask(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
encoder = TransformerEncoderWithMask(d_model, num_heads, d_ff, num_layers)

# Simulating variable length sequences
seq_lengths = [7, 5, 8, 6]
max_len = max(seq_lengths)
batch_size = len(seq_lengths)

# Create padded input
x = torch.rand(batch_size, max_len, d_model)
for i, length in enumerate(seq_lengths):
    x[i, length:] = 0  # Pad with zeros

# Create mask
mask = torch.ones(batch_size, 1, 1, max_len)
for i, length in enumerate(seq_lengths):
    mask[i, :, :, length:] = 0

output = encoder(x, mask)
print(output.shape)  # torch.Size([4, 8, 512])
```

Slide 15: Additional Resources

For a deeper understanding of Transformers and their applications, consider exploring these resources:

1. "Attention Is All You Need" paper (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "The Illustrated Transformer" by Jay Alammar: [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)

These resources provide in-depth explanations and visualizations of the Transformer architecture and its variants, helping to solidify your understanding of this powerful model.

