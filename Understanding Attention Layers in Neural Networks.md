## Understanding Attention Layers in Neural Networks
Slide 1: Understanding Attention Layers in Neural Networks

Attention layers are a crucial component of modern neural network architectures, particularly in natural language processing and computer vision tasks. They allow models to focus on specific parts of the input data, improving performance and interpretability. Let's explore the internals of attention layers and how they work.

```python
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, encoder_outputs):
        attention_weights = torch.softmax(self.attention(encoder_outputs), dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context_vector, attention_weights

# Example usage
hidden_size = 256
sequence_length = 10
batch_size = 32

attention = SimpleAttention(hidden_size)
encoder_outputs = torch.randn(batch_size, sequence_length, hidden_size)
context, weights = attention(encoder_outputs)

print("Context vector shape:", context.shape)
print("Attention weights shape:", weights.shape)
```

Slide 2: The Attention Mechanism

The attention mechanism computes a weighted sum of input elements, where the weights are determined by the relevance of each element to the task at hand. This allows the model to focus on the most important parts of the input.

```python
import torch
import torch.nn.functional as F

def attention_mechanism(query, key, value):
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale scores
    scale = torch.sqrt(torch.tensor(key.shape[-1], dtype=torch.float32))
    scaled_scores = scores / scale
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
query = torch.randn(8, 16, 64)  # (batch_size, num_queries, d_model)
key = torch.randn(8, 20, 64)    # (batch_size, seq_length, d_model)
value = torch.randn(8, 20, 64)  # (batch_size, seq_length, d_model)

output, weights = attention_mechanism(query, key, value)
print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
```

Slide 3: Self-Attention: Relating Different Positions

Self-attention allows the model to relate different positions of a single sequence. It's a key component in transformer architectures and helps capture long-range dependencies in the input data.

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
        
    def forward(self, x):
        N = x.shape[0]
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]
        
        # Split embedding into self.heads pieces
        queries = self.queries(x).reshape(N, query_len, self.heads, self.head_dim)
        keys = self.keys(x).reshape(N, key_len, self.heads, self.head_dim)
        values = self.values(x).reshape(N, value_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# Example usage
embed_size = 256
heads = 8
seq_length = 10
batch_size = 32

self_attention = SelfAttention(embed_size, heads)
x = torch.randn(batch_size, seq_length, embed_size)
output = self_attention(x)

print("Output shape:", output.shape)
```

Slide 4: Multi-Head Attention: Parallel Attention Layers

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It improves the model's ability to focus on different positions and capture various aspects of the input.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Example usage
d_model = 512
num_heads = 8
seq_length = 10
batch_size = 32

multi_head_attention = MultiHeadAttention(d_model, num_heads)
q = k = v = torch.randn(batch_size, seq_length, d_model)
output = multi_head_attention(q, k, v)

print("Output shape:", output.shape)
```

Slide 5: Positional Encoding: Adding Sequence Information

Attention layers don't inherently capture the order of elements in a sequence. Positional encoding adds information about the position of each element, allowing the model to leverage sequence order.

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding

# Example usage
seq_len = 100
d_model = 512

pos_encoding = positional_encoding(seq_len, d_model)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.imshow(pos_encoding.numpy())
plt.xlabel('Embedding Dimension')
plt.ylabel('Sequence Position')
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.show()

print("Positional encoding shape:", pos_encoding.shape)
```

Slide 6: Masked Attention: Preventing Information Leakage

In tasks like language modeling, we need to prevent the model from attending to future tokens. Masked attention solves this by applying a mask to the attention scores, effectively hiding future information.

```python
import torch
import torch.nn.functional as F

def masked_attention(query, key, value, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage: Creating a causal mask for a sequence
seq_length = 10
mask = torch.tril(torch.ones(seq_length, seq_length))

query = key = value = torch.randn(1, seq_length, 64)
output, weights = masked_attention(query, key, value, mask)

print("Output shape:", output.shape)
print("Attention weights:")
print(weights.squeeze().numpy())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(weights.squeeze().numpy(), cmap='viridis')
plt.colorbar()
plt.title('Masked Attention Weights')
plt.xlabel('Key/Value position')
plt.ylabel('Query position')
plt.show()
```

Slide 7: Attention Visualization: Understanding Model Focus

Visualizing attention weights can provide insights into what the model is focusing on. This can be particularly useful for debugging and interpreting model behavior.

```python
import torch
import matplotlib.pyplot as plt

def visualize_attention(text, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')
    
    ax.set_xticks(range(len(text)))
    ax.set_yticks(range(len(text)))
    ax.set_xticklabels(text, rotation=90)
    ax.set_yticklabels(text)
    
    plt.colorbar(im)
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    plt.show()

# Example: Simulating attention weights for a sentence
text = "The quick brown fox jumps over the lazy dog".split()
seq_len = len(text)
fake_attention_weights = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)

visualize_attention(text, fake_attention_weights.numpy())

print("Attention visualization complete.")
```

Slide 8: Efficient Attention: Linear Attention Mechanisms

As sequence lengths grow, the quadratic complexity of standard attention becomes prohibitive. Linear attention mechanisms aim to reduce this complexity while maintaining performance.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        context = torch.einsum('bhnd,bhne->bhde', k, v)
        out = torch.einsum('bhnd,bhde->bhne', q, context)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

# Example usage
dim = 512
seq_len = 1000
batch_size = 32

linear_attention = LinearAttention(dim)
x = torch.randn(batch_size, seq_len, dim)
output = linear_attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

Slide 9: Attention in Computer Vision: Visual Attention

Attention mechanisms have been successfully applied to computer vision tasks, allowing models to focus on relevant parts of an image.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class VisualAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VisualAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return out, attention

# Example usage
in_channels = 64
out_channels = 32
height, width = 28, 28
batch_size = 1

visual_attention = VisualAttention(in_channels, out_channels)
x = torch.randn(batch_size, in_channels, height, width)
output, attention = visual_attention(x)

# Visualize attention
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(x[0, 0].detach().numpy(), cmap='gray')
plt.title('Input Image (Channel 0)')
plt.subplot(122)
plt.imshow(attention[0].sum(0).view(height, width).detach().numpy())
plt.title('Attention Map')
plt.colorbar()
plt.show()

print("Output shape:", output.shape)
```

Slide 10: Attention in Natural Language Processing: BERT

BERT (Bidirectional Encoder Representations from Transformers) uses self-attention to create contextual word embeddings. Let's implement a simplified version of BERT's attention mechanism.

```python
import torch
import torch.nn as nn

class BERTAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(BERTAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

# Example usage
hidden_size = 768
num_attention_heads = 12
dropout_prob = 0.1
seq_length = 128
batch_size = 32

bert_attention = BERTAttention(hidden_size, num_attention_heads, dropout_prob)
hidden_states = torch.randn(batch_size, seq_length, hidden_size)
output = bert_attention(hidden_states)

print("Output shape:", output.shape)
```

Slide 11: Attention in Machine Translation: Transformer Architecture

The Transformer architecture, which relies heavily on attention mechanisms, has revolutionized machine translation. Let's implement a simplified version of the Transformer's encoder.

```python
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Example usage
d_model = 512
nhead = 8
dim_feedforward = 2048
dropout = 0.1
seq_length = 30
batch_size = 64

encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
src = torch.randn(seq_length, batch_size, d_model)
output = encoder_layer(src)

print("Output shape:", output.shape)
```

Slide 12: Real-life Example: Sentiment Analysis with Attention

Let's implement a simple sentiment analysis model using attention to focus on important words in a review.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentAnalysis, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        
        attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        
        return torch.sigmoid(self.fc(context_vector))

# Example usage
vocab_size = 10000
embed_dim = 100
hidden_dim = 128
seq_length = 50
batch_size = 16

model = SentimentAnalysis(vocab_size, embed_dim, hidden_dim)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
output = model(input_ids)

print("Output shape:", output.shape)
print("Sample predictions:", output[:5].squeeze().tolist())
```

Slide 13: Real-life Example: Image Captioning with Attention

Image captioning combines computer vision and natural language processing. Let's implement a simple attention-based image captioning model.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class ImageCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(ImageCaptioner, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, embed_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, images, captions):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features).unsqueeze(1)
        
        embeddings = self.embed(captions)
        hiddens, _ = self.lstm(embeddings)
        
        attention_weights = torch.bmm(hiddens, features.transpose(1, 2))
        attention_weights = F.softmax(attention_weights, dim=2)
        context = torch.bmm(attention_weights, features)
        
        output = self.fc(hiddens + context)
        return output
    
# Example usage
embed_size = 256
hidden_size = 512
vocab_size = 5000
batch_size = 32
seq_length = 20

model = ImageCaptioner(embed_size, hidden_size, vocab_size)
images = torch.randn(batch_size, 3, 224, 224)
captions = torch.randint(0, vocab_size, (batch_size, seq_length))
output = model(images, captions)

print("Output shape:", output.shape)
```

Slide 14: Additional Resources

For those interested in delving deeper into attention mechanisms and their applications, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention" by Xu et al. (2015) ArXiv: [https://arxiv.org/abs/1502.03044](https://arxiv.org/abs/1502.03044)
4. "Efficient Transformers: A Survey" by Tay et al. (2020) ArXiv: [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)

These papers provide in-depth explanations of various attention mechanisms and their applications in different domains of machine learning.

