## Drawing Transformer Networks from Scratch in Python

Slide 1: Introduction to Transformer Networks

The Transformer architecture, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing. This slideshow will guide you through building a Transformer from scratch using Python.

```python
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        # We'll fill this in as we progress
```

Slide 2: Embedding Layer

The embedding layer converts input tokens into dense vector representations. It also adds positional encoding to incorporate sequence order information.

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)
```

Slide 3: Positional Encoding

Positional encoding adds information about the position of tokens in the sequence. We use sine and cosine functions of different frequencies.

```python
def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_encoding = torch.zeros(seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding
```

Slide 4: Multi-Head Attention Mechanism

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It's a key component of the Transformer.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
```

Slide 5: Scaled Dot-Product Attention

The core of the attention mechanism is the scaled dot-product attention. It computes the compatibility between queries and keys, then applies softmax and multiplies with values.

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights
```

Slide 6: Multi-Head Attention Forward Pass

The forward pass of multi-head attention splits the input into multiple heads, applies attention, and concatenates the results.

```python
def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    
    Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    x, attention = scaled_dot_product_attention(Q, K, V, mask)
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    return self.W_o(x)
```

Slide 7: Feed-Forward Network

The feed-forward network is applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

Slide 8: Layer Normalization

Layer normalization helps stabilize the learning process. It's applied after each sub-layer in the encoder and decoder.

```python
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
```

Slide 9: Encoder Layer

The encoder layer consists of multi-head attention followed by a feed-forward network, with residual connections and layer normalization.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

Slide 10: Decoder Layer

The decoder layer is similar to the encoder but includes an additional multi-head attention layer that attends to the encoder output.

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
```

Slide 11: Encoder

The encoder consists of a stack of identical layers. It processes the input sequence and produces representations for the decoder.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(5000, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x) + self.pos_encoding[:x.size(1), :].to(x.device)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

Slide 12: Decoder

The decoder also consists of a stack of identical layers. It takes both the encoder output and the target sequence as input.

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(5000, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x) + self.pos_encoding[:x.size(1), :].to(x.device)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
```

Slide 13: Complete Transformer Model

Now we can put all the pieces together to create the complete Transformer model.

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.linear(dec_output)
```

Slide 14: Creating Masks

Masks are crucial for the Transformer. They prevent attending to padding tokens and future tokens in the target sequence.

```python
def create_padding_mask(seq):
    return (seq != 0).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).float()
    return mask == 0
```

Slide 15: Initializing and Using the Transformer

Finally, we can initialize our Transformer model and use it for a simple forward pass.

```python
# Initialize the model
src_vocab_size = 10000
tgt_vocab_size = 10000
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers)

# Example forward pass
src = torch.randint(1, src_vocab_size, (64, 20))  # (batch_size, seq_len)
tgt = torch.randint(1, tgt_vocab_size, (64, 20))  # (batch_size, seq_len)

src_mask = create_padding_mask(src)
tgt_mask = create_look_ahead_mask(tgt.size(1))

output = model(src, tgt, src_mask, tgt_mask)
print(output.shape)  # Should be (64, 20, tgt_vocab_size)
```

Slide 16: Additional Resources

For a deeper understanding of Transformer networks, consider these resources:

1. "Attention Is All You Need" paper: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "The Annotated Transformer" blog post: [http://nlp.seas.harvard.edu/2018/04/03/attention.html](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. "Transformers from Scratch" tutorial: [https://e2eml.school/transformers.html](https://e2eml.school/transformers.html)

These resources provide in-depth explanations and implementations of Transformer networks.

