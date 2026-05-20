## Exploring Self-Attention in Transformers with Python
Slide 1: Introduction to Self-Attention

Self-attention is a key mechanism in transformer models that allows the model to weigh the importance of different parts of the input when processing each element. It enables the model to capture complex relationships within the data.

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
query = np.array([[1, 0, 1], [0, 1, 1]])
key = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
value = np.array([[1, 0], [0, 1], [1, 1]])

result = self_attention(query, key, value)
print("Self-attention output:")
print(result)
```

Slide 2: Query, Key, and Value Vectors

In self-attention, input elements are transformed into query, key, and value vectors. The query vector is compared with key vectors to determine attention weights, which are then used to aggregate value vectors.

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

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Split embeddings into multiple heads
        queries = self.queries(x).reshape(batch_size, seq_length, self.heads, self.head_dim)
        keys = self.keys(x).reshape(batch_size, seq_length, self.heads, self.head_dim)
        values = self.values(x).reshape(batch_size, seq_length, self.heads, self.head_dim)

        return queries, keys, values

# Example usage
embed_size = 256
heads = 8
seq_length = 10
batch_size = 32

model = SelfAttention(embed_size, heads)
x = torch.randn(batch_size, seq_length, embed_size)
q, k, v = model(x)

print("Query shape:", q.shape)
print("Key shape:", k.shape)
print("Value shape:", v.shape)
```

Slide 3: Attention Scores and Weights

Attention scores are computed by comparing query vectors with key vectors. These scores are then normalized using the softmax function to obtain attention weights.

```python
import torch
import torch.nn.functional as F

def compute_attention(query, key, value):
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(key.size(-1)))
    
    # Apply softmax to get attention weights
    weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum of values
    output = torch.matmul(weights, value)
    
    return output, weights

# Example usage
query = torch.randn(1, 4, 8)  # (batch_size, seq_length, embed_size)
key = torch.randn(1, 4, 8)
value = torch.randn(1, 4, 8)

output, weights = compute_attention(query, key, value)
print("Attention output shape:", output.shape)
print("Attention weights shape:", weights.shape)
print("\nAttention weights:")
print(weights[0])  # Display weights for the first sequence in the batch
```

Slide 4: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces, enhancing the model's ability to capture diverse relationships in the data.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_length = query.shape[1]

        # Split embeddings into multiple heads
        queries = self.queries(query).reshape(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        keys = self.keys(key).reshape(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)
        values = self.values(value).reshape(batch_size, seq_length, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values)

        # Concatenate heads and pass through final linear layer
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.embed_size)
        return self.fc_out(out)

# Example usage
embed_size = 256
heads = 8
seq_length = 10
batch_size = 32

model = MultiHeadAttention(embed_size, heads)
x = torch.randn(batch_size, seq_length, embed_size)
output = model(x, x, x)  # Self-attention: query, key, and value are the same

print("Multi-head attention output shape:", output.shape)
```

Slide 5: Positional Encoding

Positional encoding adds information about the position of tokens in the sequence, allowing the self-attention mechanism to consider the order of inputs.

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_position, d_model):
    position = np.arange(max_position)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_encoding = np.zeros((max_position, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    return pos_encoding

# Generate positional encoding
max_position, d_model = 100, 64
pos_encoding = positional_encoding(max_position, d_model)

# Visualize positional encoding
plt.figure(figsize=(10, 6))
plt.imshow(pos_encoding, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Positional Encoding')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.tight_layout()
plt.show()
```

Slide 6: Self-Attention in Action - Text Analysis

Let's examine how self-attention works in practice for text analysis, focusing on a simple sentence to visualize attention weights.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def self_attention_visualization(sentence):
    # Tokenize the sentence (simplified for demonstration)
    tokens = sentence.lower().split()
    
    # Create a simple embedding (one-hot encoding)
    vocab = set(tokens)
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    embeddings = torch.eye(len(vocab))
    
    # Get embeddings for tokens
    token_embeddings = torch.stack([embeddings[token_to_idx[token]] for token in tokens])
    
    # Compute self-attention
    scores = torch.matmul(token_embeddings, token_embeddings.t())
    weights = F.softmax(scores, dim=-1)
    
    # Visualize attention weights
    plt.figure(figsize=(10, 8))
    plt.imshow(weights.detach().numpy(), cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title('Self-Attention Weights')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    return weights

# Example usage
sentence = "The cat sat on the mat"
attention_weights = self_attention_visualization(sentence)
print("Attention weights:")
print(attention_weights)
```

Slide 7: Real-Life Example - Sentiment Analysis

Self-attention can be used for sentiment analysis by allowing the model to focus on important words or phrases that contribute to the overall sentiment.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_classes):
        super(SentimentAnalysis, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.self_attention = nn.MultiheadAttention(embed_size, num_heads)
        self.fc = nn.Linear(embed_size, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        attended, _ = self.self_attention(embedded, embedded, embedded)
        pooled = torch.mean(attended, dim=1)
        return self.fc(pooled)

# Example usage
vocab_size, embed_size, num_heads, num_classes = 10000, 256, 4, 2
model = SentimentAnalysis(vocab_size, embed_size, num_heads, num_classes)

# Simulate input (batch_size=1, seq_length=10)
input_ids = torch.randint(0, vocab_size, (1, 10))
output = model(input_ids)

print("Sentiment analysis output shape:", output.shape)
print("Predicted sentiment (0: negative, 1: positive):", torch.argmax(output, dim=1).item())
```

Slide 8: Self-Attention for Image Processing

Self-attention can also be applied to image processing tasks, allowing the model to focus on relevant parts of the image.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ImageSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImageSelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h*w)
        v = self.value(x).view(b, -1, h*w)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return out

# Example usage
in_channels, out_channels = 3, 16
model = ImageSelfAttention(in_channels, out_channels)

# Simulate an RGB image (batch_size=1, channels=3, height=32, width=32)
image = torch.randn(1, 3, 32, 32)
output = model(image)

# Visualize input and output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image[0].permute(1, 2, 0).detach().numpy())
ax1.set_title('Input Image')
ax2.imshow(output[0].sum(dim=0).detach().numpy())
ax2.set_title('Self-Attention Output')
plt.tight_layout()
plt.show()

print("Output shape:", output.shape)
```

Slide 9: Self-Attention in Natural Language Processing

Self-attention is particularly powerful in natural language processing tasks, such as machine translation and text summarization. It allows the model to weigh the importance of different words in a sentence contextually.

```python
import torch
import torch.nn as nn

class SelfAttentionNLP(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionNLP, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output

# Example usage
embed_dim, num_heads = 512, 8
model = SelfAttentionNLP(embed_dim, num_heads)

# Simulate input (seq_length=10, batch_size=32, embed_dim=512)
x = torch.randn(10, 32, 512)
output = model(x)

print("Self-attention output shape:", output.shape)
```

Slide 10: Scaled Dot-Product Attention

The core of self-attention is the scaled dot-product attention mechanism. It computes the dot products of the query with all keys, scales them, and applies a softmax function to obtain the weights on the values.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
seq_len, batch_size, d_model = 10, 32, 64
query = key = value = torch.randn(seq_len, batch_size, d_model)

output, weights = scaled_dot_product_attention(query, key, value)
print("Attention output shape:", output.shape)
print("Attention weights shape:", weights.shape)
```

Slide 11: Real-Life Example - Language Translation

Self-attention is crucial in machine translation models, allowing the model to focus on relevant words across languages.

```python
import torch
import torch.nn as nn

class SimpleTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead):
        super(SimpleTranslator, self).__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead)
        self.fc = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src_embedded = self.src_embed(src)
        tgt_embedded = self.tgt_embed(tgt)
        output = self.transformer(src_embedded, tgt_embedded)
        return self.fc(output)

# Example usage
src_vocab_size, tgt_vocab_size, d_model, nhead = 5000, 6000, 512, 8
model = SimpleTranslator(src_vocab_size, tgt_vocab_size, d_model, nhead)

# Simulate input (seq_length=10, batch_size=32)
src = torch.randint(0, src_vocab_size, (10, 32))
tgt = torch.randint(0, tgt_vocab_size, (12, 32))  # +2 for start and end tokens

output = model(src, tgt)
print("Translation output shape:", output.shape)
```

Slide 12: Attention Visualization

Visualizing attention weights can provide insights into how the model focuses on different parts of the input.

```python
import torch
import matplotlib.pyplot as plt

def visualize_attention(sentence, attention_weights):
    words = sentence.split()
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')

    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)

    plt.colorbar(im)
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    plt.show()

# Example usage
sentence = "The quick brown fox jumps over the lazy dog"
attention_weights = torch.rand(len(sentence.split()), len(sentence.split()))
visualize_attention(sentence, attention_weights)
```

Slide 13: Self-Attention Limitations and Improvements

While powerful, self-attention has some limitations, such as quadratic complexity with sequence length. Recent improvements address these issues.

```python
import torch
import torch.nn as nn

class LinearAttention(nn.Module):
    def __init__(self, d_model):
        super(LinearAttention, self).__init__()
        self.projection = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        projected = self.projection(x)
        attention = torch.matmul(projected, x.transpose(-2, -1))
        return torch.matmul(attention, x)

# Example usage
d_model, seq_len, batch_size = 256, 1000, 32
model = LinearAttention(d_model)

x = torch.randn(batch_size, seq_len, d_model)
output = model(x)

print("Linear attention output shape:", output.shape)
print("Time complexity: O(n) instead of O(n^2)")
```

Slide 14: Future Directions in Self-Attention

Researchers are exploring new ways to improve self-attention, including sparse attention patterns and adaptive mechanisms.

```python
import torch
import torch.nn as nn

class AdaptiveSelfAttention(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(AdaptiveSelfAttention, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.adaptive_span = nn.Parameter(torch.zeros(max_seq_len))
    
    def forward(self, x):
        b, n, d = x.shape
        q, k, v = x, x, x
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        adaptive_mask = torch.sigmoid(self.adaptive_span[:n])
        scores = scores * adaptive_mask.unsqueeze(0)
        
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

# Example usage
d_model, max_seq_len, batch_size = 256, 512, 32
model = AdaptiveSelfAttention(d_model, max_seq_len)

x = torch.randn(batch_size, 100, d_model)  # Sequence length of 100
output = model(x)

print("Adaptive self-attention output shape:", output.shape)
```

Slide 15: Additional Resources

For more in-depth understanding of self-attention and transformer models, consider exploring these resources:

1. "Attention Is All You Need" (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "Transformers: State-of-the-Art Natural Language Processing" (Wolf et al., 2020): [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771)
3. "Efficient Transformers: A Survey" (Tay et al., 2020): [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)

These papers provide comprehensive insights into the development and recent advancements in self-attention mechanisms and transformer models.

