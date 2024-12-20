## Rotary Positional Embeddings (RoPE) in Python
Slide 1: Introduction to Rotary Positional Embeddings (RoPE)

Rotary Positional Embeddings (RoPE) is an innovative technique in natural language processing that addresses the challenge of incorporating positional information into transformer models. RoPE offers a more efficient and effective alternative to traditional positional encoding methods, enabling models to better understand the relative positions of tokens in a sequence.

```python
Copyimport torch
import math

def rope(x, dim):
    device = x.device
    d = x.shape[-1] // 2
    positions = torch.arange(x.shape[1], device=device).unsqueeze(1)
    freqs = torch.exp(torch.arange(0, d, 2, device=device) * -(math.log(10000.0) / d))
    theta = positions * freqs
    rot_emb = torch.cat([theta.cos(), theta.sin()], dim=-1)
    return (x * rot_emb.unsqueeze(0)).reshape(*x.shape[:-1], -1, 2).flatten(start_dim=-2)

# Example usage
x = torch.randn(1, 10, 64)  # Batch size 1, sequence length 10, embedding dim 64
rotated_x = rope(x, dim=64)
print(rotated_x.shape)  # Output: torch.Size([1, 10, 64])
```

Slide 2: The Need for Positional Information

In transformer models, self-attention mechanisms operate on unordered sets of vectors. However, the order of words in a sentence is crucial for understanding its meaning. Positional embeddings provide this essential sequential information to the model.

```python
Copyimport torch
import torch.nn as nn

class TransformerWithoutPosition(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=8)
    
    def forward(self, x):
        return self.transformer(self.embedding(x))

# Example usage
vocab_size, d_model = 1000, 512
model = TransformerWithoutPosition(vocab_size, d_model)
input_ids = torch.randint(0, vocab_size, (1, 20))  # Batch size 1, sequence length 20
output = model(input_ids)
print(output.shape)  # Output: torch.Size([1, 20, 512])
```

Slide 3: Traditional Positional Encodings

Before RoPE, models like the original Transformer used sinusoidal positional encodings or learned positional embeddings. These methods add absolute position information to token embeddings.

```python
Copyimport torch
import math

def sinusoidal_positional_encoding(max_seq_len, d_model):
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Example usage
max_seq_len, d_model = 100, 512
pe = sinusoidal_positional_encoding(max_seq_len, d_model)
print(pe.shape)  # Output: torch.Size([100, 512])

# Visualize the first few dimensions
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(pe[:, :20])
plt.title("Sinusoidal Positional Encodings")
plt.xlabel("Sequence Position")
plt.ylabel("Encoding Value")
plt.show()
```

Slide 4: Limitations of Traditional Positional Encodings

Traditional positional encodings have limitations, such as difficulty in extrapolating to longer sequences and potential interference with token embeddings. RoPE addresses these issues by encoding relative positions directly into the attention computation.

```python
Copyimport torch
import torch.nn as nn

class TransformerWithAbsolutePosition(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead=8)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.transformer(self.embedding(x) + self.pos_encoding(positions))

# Example usage
vocab_size, d_model, max_seq_len = 1000, 512, 100
model = TransformerWithAbsolutePosition(vocab_size, d_model, max_seq_len)
input_ids = torch.randint(0, vocab_size, (1, 20))  # Batch size 1, sequence length 20
output = model(input_ids)
print(output.shape)  # Output: torch.Size([1, 20, 512])

# Attempt to process a longer sequence
long_input = torch.randint(0, vocab_size, (1, 150))  # Sequence length > max_seq_len
try:
    output = model(long_input)
except IndexError as e:
    print(f"Error processing longer sequence: {e}")
```

Slide 5: Introduction to RoPE

RoPE introduces a novel approach to positional encoding by applying a rotation to the token embeddings. This rotation is based on the token's position and frequency, allowing the model to capture relative positional information efficiently.

```python
Copyimport torch
import math

def rope_rotate(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)

def rope_embedding(x, dim):
    device = x.device
    seq_len = x.shape[1]
    d = dim // 2
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, device=device) * -(math.log(10000.0) / d))
    theta = position * div_term
    cos = theta.cos().repeat_interleave(2, dim=-1)
    sin = theta.sin().repeat_interleave(2, dim=-1)
    return rope_rotate(x, cos, sin)

# Example usage
x = torch.randn(1, 10, 64)  # Batch size 1, sequence length 10, embedding dim 64
rotated_x = rope_embedding(x, dim=64)
print(rotated_x.shape)  # Output: torch.Size([1, 10, 64])
```

Slide 6: Mathematical Foundation of RoPE

RoPE is based on complex number rotations. For a token at position m, its embedding e is rotated by θ = mω, where ω is a frequency vector. This rotation preserves the inner product between tokens while encoding their relative positions.

```python
Copyimport torch
import math

def rope_rotate_complex(x, theta):
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
    rotation = torch.polar(torch.ones_like(x_complex), theta)
    return torch.view_as_real(x_complex * rotation).flatten(-2)

def rope_embedding_complex(x, dim):
    device = x.device
    seq_len = x.shape[1]
    d = dim // 2
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, device=device) * -(math.log(10000.0) / d))
    theta = position * div_term
    return rope_rotate_complex(x, theta)

# Example usage
x = torch.randn(1, 10, 64)  # Batch size 1, sequence length 10, embedding dim 64
rotated_x = rope_embedding_complex(x, dim=64)
print(rotated_x.shape)  # Output: torch.Size([1, 10, 64])
```

Slide 7: RoPE in Attention Mechanism

RoPE modifies the attention mechanism by rotating the query and key vectors before computing attention scores. This allows the model to consider relative positions without explicitly adding positional encodings.

```python
Copyimport torch
import torch.nn as nn
import math

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q_rotated = rope_embedding(q, dim=self.head_dim)
        k_rotated = rope_embedding(k, dim=self.head_dim)
        
        attention_scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, v).view(batch_size, seq_len, -1)
        return self.out_proj(output)

# Example usage
d_model, num_heads = 512, 8
mha = RoPEMultiHeadAttention(d_model, num_heads)
x = torch.randn(1, 20, d_model)  # Batch size 1, sequence length 20
output = mha(x)
print(output.shape)  # Output: torch.Size([1, 20, 512])
```

Slide 8: Advantages of RoPE

RoPE offers several advantages over traditional positional encodings:

1. It naturally handles variable-length sequences without a fixed maximum length.
2. It preserves the original token embedding space, allowing for better generalization.
3. It efficiently encodes relative positions, which is crucial for many NLP tasks.
4. It can be easily integrated into existing transformer architectures with minimal changes.

```python
Copyimport torch
import torch.nn as nn

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
    
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Example usage
d_model, nhead = 512, 8
layer = RoPETransformerLayer(d_model, nhead)
x = torch.randn(1, 20, d_model)  # Batch size 1, sequence length 20
output = layer(x)
print(output.shape)  # Output: torch.Size([1, 20, 512])

# Process a longer sequence
long_x = torch.randn(1, 100, d_model)  # Sequence length 100
long_output = layer(long_x)
print(long_output.shape)  # Output: torch.Size([1, 100, 512])
```

Slide 9: RoPE vs. Absolute Positional Encodings

RoPE differs from absolute positional encodings by focusing on relative positions. This approach allows for better generalization to unseen sequence lengths and maintains the semantic meaning of token embeddings.

```python
Copyimport torch
import torch.nn as nn
import math

def absolute_positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

class AbsolutePositionTransformer(nn.Module):
    def __init__(self, d_model, nhead, max_len):
        super().__init__()
        self.pos_encoding = absolute_positional_encoding(max_len, d_model)
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead)
    
    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        return self.transformer(x)

class RoPETransformer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.transformer = RoPETransformerLayer(d_model, nhead)
    
    def forward(self, x):
        return self.transformer(x)

# Compare the two models
d_model, nhead, max_len = 512, 8, 1000
abs_model = AbsolutePositionTransformer(d_model, nhead, max_len)
rope_model = RoPETransformer(d_model, nhead)

x_short = torch.randn(1, 20, d_model)
x_long = torch.randn(1, 1500, d_model)

print("Short sequence:")
print("Absolute:", abs_model(x_short).shape)
print("RoPE:", rope_model(x_short).shape)

print("\nLong sequence:")
try:
    print("Absolute:", abs_model(x_long).shape)
except RuntimeError as e:
    print("Absolute: Error -", str(e))
print("RoPE:", rope_model(x_long).shape)
```

Slide 10: Implementing RoPE in PyTorch

Here's a step-by-step implementation of RoPE in PyTorch, demonstrating how to integrate it into a transformer model:

```python
Copyimport torch
import torch.nn as nn
import math

class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def rope_rotate(self, x, cos, sin):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([-x2 * sin + x1 * cos, x1 * sin + x2 * cos], dim=-1)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Generate RoPE embeddings
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.head_dim, 2, device=x.device) * -(math.log(10000.0) / self.head_dim))
        theta = position * div_term
        cos = theta.cos().repeat_interleave(2, dim=-1)
        sin = theta.sin().repeat_interleave(2, dim=-1)
        
        # Apply RoPE to queries and keys
        q_rotated = self.rope_rotate(q, cos, sin)
        k_rotated = self.rope_rotate(k, cos, sin)
        
        # Compute attention and output
        attn = (q_rotated @ k_rotated.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

# Example usage
d_model, num_heads = 512, 8
attention = RoPEAttention(d_model, num_heads)
x = torch.randn(2, 100, d_model)  # Batch size 2, sequence length 100
output = attention(x)
print(output.shape)  # Output: torch.Size([2, 100, 512])
```

Slide 11: RoPE in Language Model Fine-tuning

RoPE can significantly improve the performance of language models during fine-tuning tasks. It allows the model to better capture positional relationships in the input sequence, leading to improved results on various NLP tasks.

```python
Copyimport torch
import torch.nn as nn

class RoPELanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([RoPEAttention(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = self.norm(x)
        return self.fc(x)

# Example usage for fine-tuning
vocab_size, d_model, num_heads, num_layers = 30000, 512, 8, 6
model = RoPELanguageModel(vocab_size, d_model, num_heads, num_layers)

# Simulated fine-tuning loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    # Simulated batch
    input_ids = torch.randint(0, vocab_size, (32, 128))  # Batch size 32, sequence length 128
    labels = torch.randint(0, vocab_size, (32, 128))
    
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Slide 12: Real-life Example: Text Summarization with RoPE

Let's explore how RoPE can be applied to a text summarization task, improving the model's ability to capture long-range dependencies and produce coherent summaries.

```python
Copyimport torch
import torch.nn as nn

class RoPESummarizer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder = RoPELanguageModel(vocab_size, d_model, num_heads, num_layers)
        self.decoder = RoPELanguageModel(vocab_size, d_model, num_heads, num_layers)
        self.summary_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt)
        return self.summary_proj(encoder_output + decoder_output)

# Example usage
vocab_size, d_model, num_heads, num_layers = 30000, 512, 8, 6
summarizer = RoPESummarizer(vocab_size, d_model, num_heads, num_layers)

# Simulated summarization task
src_text = torch.randint(0, vocab_size, (1, 500))  # Source text
tgt_text = torch.randint(0, vocab_size, (1, 100))  # Target summary

summary_logits = summarizer(src_text, tgt_text)
print("Summary logits shape:", summary_logits.shape)  # Output: torch.Size([1, 100, 30000])

# In practice, you would use these logits to generate the summary text
```

Slide 13: Real-life Example: Question Answering with RoPE

RoPE can enhance question answering models by helping them better understand the relative positions of words in both the question and the context passage.

```python
Copyimport torch
import torch.nn as nn

class RoPEQuestionAnswering(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder = RoPELanguageModel(vocab_size, d_model, num_heads, num_layers)
        self.question_proj = nn.Linear(d_model, d_model)
        self.context_proj = nn.Linear(d_model, d_model)
        self.span_predictor = nn.Linear(d_model, 2)  # Start and end positions
    
    def forward(self, question, context):
        q_embed = self.encoder(question)
        c_embed = self.encoder(context)
        
        q_proj = self.question_proj(q_embed)
        c_proj = self.context_proj(c_embed)
        
        # Compute attention between question and context
        attention = torch.matmul(q_proj, c_proj.transpose(-2, -1))
        attention_weights = torch.softmax(attention, dim=-1)
        
        # Weighted sum of context embeddings
        weighted_context = torch.matmul(attention_weights, c_embed)
        
        # Predict answer span
        span_logits = self.span_predictor(weighted_context)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

# Example usage
vocab_size, d_model, num_heads, num_layers = 30000, 512, 8, 6
qa_model = RoPEQuestionAnswering(vocab_size, d_model, num_heads, num_layers)

# Simulated question answering task
question = torch.randint(0, vocab_size, (1, 20))  # Question text
context = torch.randint(0, vocab_size, (1, 200))  # Context passage

start_logits, end_logits = qa_model(question, context)
print("Start logits shape:", start_logits.shape)  # Output: torch.Size([1, 200])
print("End logits shape:", end_logits.shape)    # Output: torch.Size([1, 200])

# In practice, you would use these logits to select the answer span from the context
```

Slide 14: Additional Resources

For those interested in diving deeper into Rotary Positional Embeddings, here are some valuable resources:

1. Original RoPE paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al. (2021) ArXiv link: [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
2. "Transformer Language Models without Positional Encodings Still Learn Positional Information" by Haviv et al. (2022) ArXiv link: [https://arxiv.org/abs/2203.16634](https://arxiv.org/abs/2203.16634)
3. "InstructGPT: Training language models to follow instructions with human feedback" by Ouyang et al. (2022) ArXiv link: [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

These papers provide in-depth discussions on the theory and applications of RoPE and related positional encoding techniques in transformer models.

