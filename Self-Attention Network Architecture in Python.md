## Self-Attention Network Architecture in Python
Slide 1: Introduction to Self-Attention Networks

Self-Attention Networks are a fundamental component of modern deep learning architectures, particularly in natural language processing. They allow models to weigh the importance of different parts of the input data dynamically, leading to more effective learning of complex relationships.

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        queries = self.queries(query).reshape(N, query_len, self.heads, self.head_dim)
        keys = self.keys(key).reshape(N, key_len, self.heads, self.head_dim)
        values = self.values(value).reshape(N, value_len, self.heads, self.head_dim)

        # Perform self-attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)
```

Slide 2: Key Components of Self-Attention

Self-Attention consists of three main components: Queries, Keys, and Values. These are created by transforming the input through linear layers. The dot product between Queries and Keys determines the attention weights, which are then applied to the Values.

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        
        out = torch.matmul(attention_weights, V)
        return out

# Example usage
embed_size = 512
x = torch.randn(10, 20, embed_size)  # batch_size=10, seq_len=20
self_attention = SelfAttention(embed_size)
output = self_attention(x)
print(output.shape)  # torch.Size([10, 20, 512])
```

Slide 3: Scaled Dot-Product Attention

Scaled Dot-Product Attention is the core operation in self-attention. It computes the dot product of the query with all keys, scales them, and applies a softmax function to obtain the weights on the values.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights

# Example usage
q = torch.randn(2, 4, 8)  # (batch_size, num_queries, d_model)
k = torch.randn(2, 5, 8)  # (batch_size, num_keys, d_model)
v = torch.randn(2, 5, 8)  # (batch_size, num_values, d_model)

output, weights = scaled_dot_product_attention(q, k, v)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)

# Output:
# Output shape: torch.Size([2, 4, 8])
# Attention weights shape: torch.Size([2, 4, 5])
```

Slide 4: Multi-Head Attention

Multi-Head Attention allows the model to jointly attend to information from different representation subspaces. It consists of several attention layers running in parallel, each with its own learned projection.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)

# Example usage
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = mha(x, x, x)
print("Output shape:", output.shape)  # torch.Size([32, 10, 512])
```

Slide 5: Position-wise Feed-Forward Networks

Position-wise Feed-Forward Networks are applied to each position separately and identically. They consist of two linear transformations with a ReLU activation in between.

```python
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Example usage
d_model = 512
d_ff = 2048
ff = PositionwiseFeedForward(d_model, d_ff)
x = torch.randn(32, 10, d_model)  # (batch_size, seq_len, d_model)
output = ff(x)
print("Output shape:", output.shape)  # torch.Size([32, 10, 512])
```

Slide 6: Positional Encoding

Positional Encoding is crucial in self-attention networks to provide information about the relative or absolute position of the tokens in the sequence. Here's an implementation of sinusoidal positional encoding.

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

# Example usage
seq_len = 100
d_model = 512
pe = positional_encoding(seq_len, d_model)
print("Positional Encoding shape:", pe.shape)  # torch.Size([100, 512])

# Visualize the positional encoding
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.imshow(pe.numpy())
plt.title("Positional Encoding")
plt.xlabel("Encoding dimension")
plt.ylabel("Sequence position")
plt.colorbar()
plt.show()
```

Slide 7: Layer Normalization

Layer Normalization is a technique used to normalize the inputs across the features. It's applied after the self-attention and feed-forward layers to stabilize the learning process.

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Example usage
features = 512
ln = LayerNorm(features)
x = torch.randn(32, 10, features)  # (batch_size, seq_len, features)
output = ln(x)
print("Output shape:", output.shape)  # torch.Size([32, 10, 512])

# Verify normalization
print("Mean:", output.mean().item())  # Should be close to 0
print("Std:", output.std().item())   # Should be close to 1
```

Slide 8: Residual Connections

Residual connections are used to help the network learn identity functions and mitigate the vanishing gradient problem. They are typically applied around the self-attention and feed-forward layers.

```python
import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Example usage
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
    
    def forward(self, x):
        return self.attn(x, x, x)[0]

d_model = 512
dropout = 0.1
residual = ResidualConnection(d_model, dropout)
self_attn = SelfAttention(d_model)

x = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = residual(x, self_attn)
print("Output shape:", output.shape)  # torch.Size([10, 32, 512])
```

Slide 9: Encoder Layer

The Encoder Layer combines all the components we've discussed so far: self-attention, feed-forward network, layer normalization, and residual connections.

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
encoder_layer = EncoderLayer(d_model, num_heads, d_ff)

x = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = encoder_layer(x)
print("Output shape:", output.shape)  # torch.Size([10, 32, 512])
```

Slide 10: Full Encoder

The full Encoder consists of multiple Encoder Layers stacked on top of each other. Here's an implementation of a complete Encoder.

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Example usage
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
encoder = Encoder(d_model, num_heads, d_ff, num_layers)

x = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = encoder(x)
print("Output shape:", output.shape)  # torch.Size([10, 32, 512])
```

Slide 11: Self-Attention Visualization

Visualizing attention weights provides insights into what the model focuses on. Here's a simple implementation to visualize attention weights:

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                           ha="center", va="center", color="black")
    
    ax.set_title("Self-Attention Visualization")
    fig.tight_layout()
    plt.show()

# Example usage
tokens = ["The", "cat", "sat", "on", "the", "mat"]
attention_weights = np.random.rand(len(tokens), len(tokens))
visualize_attention(attention_weights, tokens)
```

Slide 12: Real-Life Example - Text Classification

Self-attention networks can be used for various NLP tasks. Here's a simple text classification model using self-attention:

```python
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_heads):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads)
        self.fc = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        attended, _ = self.self_attention(embedded, embedded, embedded)
        pooled = torch.mean(attended, dim=0)
        return self.fc(pooled)

# Example usage
vocab_size = 10000
embed_size = 256
num_classes = 5
num_heads = 8

model = TextClassifier(vocab_size, embed_size, num_classes, num_heads)
input_seq = torch.randint(0, vocab_size, (20, 32))  # (seq_len, batch_size)
output = model(input_seq)
print("Output shape:", output.shape)  # torch.Size([32, 5])
```

Slide 13: Real-Life Example - Machine Translation

Self-attention networks are fundamental in machine translation. Here's a simplified encoder-decoder architecture for translation:

```python
import torch
import torch.nn as nn

class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers):
        super(Translator, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output)
        return self.generator(dec_output)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
    
    def forward(self, x, enc_output):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, enc_output)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        return self.norm(x + attn_output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, enc_output):
        self_attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self_attn_output)
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output)
        return self.norm2(x + cross_attn_output)

# Example usage
src_vocab_size = 10000
tgt_vocab_size = 8000
d_model = 512
num_heads = 8
num_layers = 6

model = Translator(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers)
src = torch.randint(0, src_vocab_size, (20, 32))  # (seq_len, batch_size)
tgt = torch.randint(0, tgt_vocab_size, (15, 32))  # (seq_len, batch_size)
output = model(src, tgt)
print("Output shape:", output.shape)  # torch.Size([15, 32, 8000])
```

Slide 14: Additional Resources

For further exploration of Self-Attention Network Architecture, consider these resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Transformers: State-of-the-Art Natural Language Processing" by Wolf et al. (2020) ArXiv: [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771)

These papers provide in-depth explanations and implementations of self-attention networks and their applications in various natural language processing tasks.

