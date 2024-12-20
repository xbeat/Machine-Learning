## Explaining RoPE Positional Embeddings in Python
Slide 1: Introduction to RoPE (Rotary Positional Embeddings)

RoPE is a technique used in large language models to encode positional information into token embeddings. It allows models to understand the relative positions of tokens in a sequence without using separate positional encodings.

```python
import torch
import math

def rope_embed(x, dim, base=10000):
    device = x.device
    half_dim = dim // 2
    emb = torch.tensor([[i / (base ** (2 * (i // 2) / dim)) for i in range(dim)]], device=device)
    emb = emb.repeat(x.shape[1], 1)
    return torch.cat([x * emb.sin(), x * emb.cos()], dim=-1)

# Example usage
seq_len, d_model = 10, 64
x = torch.randn(1, seq_len, d_model)
embedded = rope_embed(x, d_model)
print(f"Input shape: {x.shape}, Output shape: {embedded.shape}")
```

Slide 2: How RoPE Works

RoPE applies a rotation to each element of the input embedding based on its position. This rotation is performed in 2D space, treating pairs of elements as coordinates.

```python
def apply_rope(x, cos, sin, position):
    x_rot = torch.zeros_like(x)
    x_rot[:, :, 0::2] = x[:, :, 0::2] * cos - x[:, :, 1::2] * sin
    x_rot[:, :, 1::2] = x[:, :, 1::2] * cos + x[:, :, 0::2] * sin
    return x_rot

# Generate rotation matrices
seq_len, d_model = 10, 64
position = torch.arange(seq_len).unsqueeze(1)
freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
theta = position * freqs
cos, sin = torch.cos(theta), torch.sin(theta)

# Apply RoPE
x = torch.randn(1, seq_len, d_model)
x_rotated = apply_rope(x, cos.unsqueeze(0), sin.unsqueeze(0), position)
print(f"Original shape: {x.shape}, Rotated shape: {x_rotated.shape}")
```

Slide 3: Advantages of RoPE

RoPE offers several benefits over traditional positional encodings, including better extrapolation to longer sequences and improved performance on tasks requiring precise positional information.

```python
import matplotlib.pyplot as plt

def visualize_rope_extrapolation(max_len=1000, d_model=64):
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    position = torch.arange(max_len).unsqueeze(1)
    theta = position * freqs
    
    plt.figure(figsize=(12, 6))
    plt.plot(position[:, 0], torch.sin(theta[:, 0]), label='sin(θ)')
    plt.plot(position[:, 0], torch.cos(theta[:, 0]), label='cos(θ)')
    plt.title('RoPE Extrapolation')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

visualize_rope_extrapolation()
```

Slide 4: Implementing RoPE in a Transformer Layer

Let's see how to integrate RoPE into a simplified Transformer layer, focusing on the self-attention mechanism.

```python
import torch.nn as nn
import torch.nn.functional as F

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE to q and k
        q, k = apply_rope_to_qk(q, k)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# Placeholder for apply_rope_to_qk function
def apply_rope_to_qk(q, k):
    # Implementation of RoPE for q and k
    return q, k

# Example usage
mha = RoPEMultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)
output = mha(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

Slide 5: RoPE vs. Absolute Positional Encodings

RoPE differs from absolute positional encodings by encoding relative positions implicitly. This allows for better generalization to unseen sequence lengths.

```python
import numpy as np
import matplotlib.pyplot as plt

def absolute_positional_encoding(max_len, d_model):
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def rope_encoding(max_len, d_model):
    position = np.arange(max_len)[:, np.newaxis]
    div_term = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
    theta = position * div_term
    rope = np.zeros((max_len, d_model))
    rope[:, 0::2] = np.sin(theta)
    rope[:, 1::2] = np.cos(theta)
    return rope

max_len, d_model = 100, 64
abs_pe = absolute_positional_encoding(max_len, d_model)
rope_pe = rope_encoding(max_len, d_model)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(abs_pe, aspect='auto', cmap='viridis')
plt.title('Absolute Positional Encoding')
plt.subplot(1, 2, 2)
plt.imshow(rope_pe, aspect='auto', cmap='viridis')
plt.title('RoPE Encoding')
plt.tight_layout()
plt.show()
```

Slide 6: RoPE in Attention Mechanism

RoPE modifies the attention mechanism by rotating query and key vectors. This rotation encodes relative positional information directly into the attention computation.

```python
import torch
import torch.nn.functional as F

def rope_attention(q, k, v, scale=None):
    B, H, T, C = q.shape
    scale = scale or (1.0 / C**0.5)
    
    # Apply RoPE to q and k
    q, k = apply_rope_to_qk(q, k)
    
    # Compute attention scores
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    
    # Compute output
    out = attn @ v
    return out

# Placeholder for apply_rope_to_qk function
def apply_rope_to_qk(q, k):
    # Implementation of RoPE for q and k
    return q, k

# Example usage
B, H, T, C = 2, 8, 100, 64
q = torch.randn(B, H, T, C)
k = torch.randn(B, H, T, C)
v = torch.randn(B, H, T, C)

output = rope_attention(q, k, v)
print(f"Input shape: {q.shape}, Output shape: {output.shape}")
```

Slide 7: RoPE and Relative Attention

RoPE implicitly encodes relative positions, making it similar to relative attention mechanisms. However, RoPE achieves this without explicitly computing pairwise position differences.

```python
import torch
import torch.nn as nn

class RelativePositionAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * d_model - 1, num_heads))
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute relative position bias
        pos_bias = self.rel_pos_bias[T-1:2*T-1].unsqueeze(0).unsqueeze(0)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / self.head_dim**0.5)
        attn = attn + pos_bias
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# Example usage
rel_attn = RelativePositionAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)
output = rel_attn(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

Slide 8: RoPE and Sequence Length Extrapolation

One of RoPE's key advantages is its ability to extrapolate to longer sequences than seen during training. Let's visualize this property.

```python
import torch
import matplotlib.pyplot as plt

def rope_embed(x, dim, base=10000):
    device = x.device
    half_dim = dim // 2
    emb = torch.tensor([[i / (base ** (2 * (i // 2) / dim)) for i in range(dim)]], device=device)
    emb = emb.repeat(x.shape[1], 1)
    return torch.cat([x * emb.sin(), x * emb.cos()], dim=-1)

def visualize_rope_extrapolation(train_len=512, eval_len=2048, d_model=64):
    x_train = torch.randn(1, train_len, d_model)
    x_eval = torch.randn(1, eval_len, d_model)
    
    emb_train = rope_embed(x_train, d_model)
    emb_eval = rope_embed(x_eval, d_model)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(train_len), emb_train[0, :, 0], label='Training', color='blue')
    plt.plot(range(eval_len), emb_eval[0, :, 0], label='Evaluation', color='red')
    plt.axvline(x=train_len, color='green', linestyle='--', label='Train Length')
    plt.title('RoPE Extrapolation')
    plt.xlabel('Sequence Position')
    plt.ylabel('Embedding Value')
    plt.legend()
    plt.show()

visualize_rope_extrapolation()
```

Slide 9: RoPE in Multi-Layer Transformers

In multi-layer Transformers, RoPE is typically applied at each layer. This allows the model to maintain positional information throughout its depth.

```python
import torch
import torch.nn as nn

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class RoPETransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
model = RoPETransformer(num_layers=6, d_model=512, num_heads=8, d_ff=2048)
x = torch.randn(2, 100, 512)
output = model(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

Slide 10: RoPE and Attention Patterns

RoPE influences attention patterns by encoding relative positions. Let's visualize how this affects attention weights.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def rope_attention_pattern(seq_len, d_model, num_heads):
    q = torch.randn(1, num_heads, seq_len, d_model // num_heads)
    k = torch.randn(1, num_heads, seq_len, d_model // num_heads)
    
    # Apply simplified RoPE
    position = torch.arange(seq_len).unsqueeze(1)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model // num_heads, 2).float() / (d_model // num_heads)))
    theta = position * freqs
    
    # Rotate q and k (simplified for visualization)
    q_rot = q * torch.cos(theta) + torch.roll(q, shifts=1, dims=-1) * torch.sin(theta)
    k_rot = k * torch.cos(theta) + torch.roll(k, shifts=1, dims=-1) * torch.sin(theta)
    
    # Compute attention weights
    attn_weights = F.softmax(torch.matmul(q_rot, k_rot.transpose(-2, -1)) / (d_model // num_heads)**0.5, dim=-1)
    
    return attn_weights.squeeze(0).mean(0)

# Visualize attention pattern
seq_len, d_model, num_heads = 50, 64, 1
attn_pattern = rope_attention_pattern(seq_len, d_model, num_heads)

plt.figure(figsize=(10, 8))
plt.imshow(attn_pattern, cmap='viridis')
plt.colorbar()
plt.title('RoPE Attention Pattern')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()
```

Slide 11: RoPE in Language Generation

RoPE enhances language generation tasks by providing better context understanding. Let's implement a simple language model using RoPE.

```python
import torch
import torch.nn as nn

class RoPELanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = RoPETransformer(num_layers, d_model, num_heads, d_model * 4)
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.transformer(x)
        return self.output_layer(x)

# Example usage
vocab_size, d_model, num_heads, num_layers = 10000, 512, 8, 6
model = RoPELanguageModel(vocab_size, d_model, num_heads, num_layers)

# Simulate input tokens
input_tokens = torch.randint(0, vocab_size, (1, 50))
output = model(input_tokens)
print(f"Input shape: {input_tokens.shape}, Output shape: {output.shape}")
```

Slide 12: RoPE and Long-Range Dependencies

RoPE helps models capture long-range dependencies more effectively. Let's visualize how attention scores change with distance in a RoPE model.

```python
import torch
import matplotlib.pyplot as plt

def rope_attention_decay(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    theta = position * freqs
    
    # Simulate attention scores for a fixed query at position 0
    query_pos = 0
    key_pos = torch.arange(seq_len)
    relative_pos = key_pos - query_pos
    
    # Simplified attention score calculation
    attention_scores = torch.cos(theta[query_pos] - theta)[:, 0]
    
    return key_pos.numpy(), attention_scores.numpy()

# Visualize attention decay
seq_len, d_model = 1000, 64
positions, scores = rope_attention_decay(seq_len, d_model)

plt.figure(figsize=(12, 6))
plt.plot(positions, scores)
plt.title('RoPE Attention Score Decay')
plt.xlabel('Key Position')
plt.ylabel('Attention Score')
plt.show()
```

Slide 13: RoPE in Real-World Applications

RoPE has been successfully applied in various natural language processing tasks. Let's look at a simplified example of using RoPE for text classification.

```python
import torch
import torch.nn as nn

class RoPETextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = RoPETransformer(num_layers, d_model, num_heads, d_model * 4)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        # Use the [CLS] token representation for classification
        return self.classifier(x[:, 0, :])

# Example usage
vocab_size, d_model, num_heads, num_layers, num_classes = 10000, 512, 8, 6, 5
model = RoPETextClassifier(vocab_size, d_model, num_heads, num_layers, num_classes)

# Simulate input tokens for a batch of sentences
batch_size, seq_len = 32, 128
input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_tokens)
print(f"Input shape: {input_tokens.shape}, Output shape: {output.shape}")
```

Slide 14: RoPE vs. Other Positional Encoding Methods

RoPE offers advantages over other positional encoding methods. Let's compare RoPE with sinusoidal positional encoding and learned positional embeddings.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def rope_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    theta = position * div_term
    return torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)

def sinusoidal_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Learned positional embeddings
learned_pe = nn.Embedding(1000, d_model)

seq_len, d_model = 100, 64
rope = rope_encoding(seq_len, d_model)
sinusoidal = sinusoidal_encoding(seq_len, d_model)
learned = learned_pe(torch.arange(seq_len))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(rope, aspect='auto', cmap='viridis')
plt.title('RoPE')
plt.subplot(1, 3, 2)
plt.imshow(sinusoidal, aspect='auto', cmap='viridis')
plt.title('Sinusoidal')
plt.subplot(1, 3, 3)
plt.imshow(learned.detach(), aspect='auto', cmap='viridis')
plt.title('Learned')
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For more information on RoPE and its applications in language models, consider the following resources:

1. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021) ArXiv: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
2. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (Dai et al., 2019) ArXiv: [https://arxiv.org/abs/1901.02860](https://arxiv.org/abs/1901.02860)
3. "Attention Is All You Need" (Vaswani et al., 2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

These papers provide comprehensive insights into positional encodings and their role in transformer architectures.

