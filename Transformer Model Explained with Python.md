## Transformer Model Explained with Python

Slide 1: Introduction to Transformer Models

The Transformer model, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing. It's a neural network architecture that uses self-attention mechanisms to process sequential data, eliminating the need for recurrent or convolutional layers.

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer(embedded, embedded)
        return self.fc(output)

# Example usage
vocab_size, d_model, nhead, num_encoder_layers = 10000, 512, 8, 6
model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers)
```

Slide 2: Self-Attention Mechanism

The core of the Transformer model is the self-attention mechanism. It allows the model to weigh the importance of different parts of the input sequence when processing each element, enabling it to capture long-range dependencies effectively.

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
```

Slide 3: Positional Encoding

Transformers don't inherently understand the order of input elements. Positional encoding is used to inject information about the position of tokens in the sequence, allowing the model to leverage sequential information.

```python
import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# Example usage
seq_len, d_model = 100, 512
pos_encoding = positional_encoding(seq_len, d_model)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Sequence Position')
plt.colorbar()
plt.title("Positional Encoding")
plt.show()
```

Slide 4: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, enhancing the model's ability to capture various aspects of the input.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
```

Slide 5: Feedforward Neural Network

The feedforward neural network in a Transformer consists of two linear transformations with a ReLU activation in between. It processes each position separately and identically, allowing the model to introduce non-linearity and increase its representational power.

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Example usage
embed_size, ff_hidden_dim = 512, 2048
ff_layer = FeedForward(embed_size, ff_hidden_dim)

# Test with random input
x = torch.randn(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
output = ff_layer(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 6: Encoder Layer

The Encoder layer is a fundamental building block of the Transformer model. It consists of a multi-head self-attention mechanism followed by a feedforward neural network, with layer normalization and residual connections.

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, embed_size * forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Example usage
embed_size, heads, dropout, forward_expansion = 512, 8, 0.1, 4
encoder_layer = EncoderLayer(embed_size, heads, dropout, forward_expansion)

# Test with random input
x = torch.randn(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
mask = torch.ones(32, 1, 1, 10)  # (batch_size, 1, 1, seq_len)
output = encoder_layer(x, x, x, mask)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 7: Decoder Layer

The Decoder layer is similar to the Encoder layer but includes an additional multi-head attention sublayer that attends to the output of the Encoder stack. This allows the Decoder to focus on relevant parts of the input sequence.

```python
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = EncoderLayer(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

# Example usage
embed_size, heads, dropout, forward_expansion = 512, 8, 0.1, 4
decoder_layer = DecoderLayer(embed_size, heads, dropout, forward_expansion)

# Test with random input
x = torch.randn(32, 10, embed_size)  # (batch_size, trg_seq_len, embed_size)
encoder_out = torch.randn(32, 15, embed_size)  # (batch_size, src_seq_len, embed_size)
src_mask = torch.ones(32, 1, 1, 15)  # (batch_size, 1, 1, src_seq_len)
trg_mask = torch.tril(torch.ones(10, 10)).unsqueeze(0).unsqueeze(0)  # (1, 1, trg_seq_len, trg_seq_len)
output = decoder_layer(x, encoder_out, encoder_out, src_mask, trg_mask)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 8: Complete Transformer Model

The complete Transformer model combines the Encoder and Decoder stacks, along with embedding layers and a final linear layer for output generation.

```python
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        device="cpu",
        max_length=100
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# Example usage
src_vocab_size, trg_vocab_size = 10000, 10000
src_pad_idx, trg_pad_idx = 0, 0
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)

# Test with random input
src = torch.randint(1, src_vocab_size, (32, 15))  # (batch_size, src_seq_len)
trg = torch.randint(1, trg_vocab_size, (32, 10))  # (batch_size, trg_seq_len)
output = model(src, trg)
print(f"Input shapes: src {src.shape}, trg {trg.shape}")
print(f"Output shape: {output.shape}")
```

Slide 9: Training the Transformer

Training a Transformer model involves defining a loss function, optimizer, and implementing the training loop. Here's a simplified example of how to train a Transformer for a sequence-to-sequence task.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume we have defined our Transformer model as 'model'
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train_step(src, trg):
    model.train()
    optimizer.zero_grad()
    
    output = model(src, trg[:, :-1])
    output = output.reshape(-1, output.shape[2])
    trg = trg[:, 1:].reshape(-1)
    
    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:  # Assume we have a dataloader
        src, trg = batch
        loss = train_step(src, trg)
        total_loss += loss
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
```

Slide 10: Inference with Transformer

After training, we can use the Transformer model for inference. This process typically involves generating output tokens one at a time until an end-of-sequence token is produced or a maximum length is reached.

```python
def greedy_decode(model, src, max_len, start_symbol):
    src_mask = model.make_src_mask(src)
    enc_src = model.encoder(src, src_mask)
    
    trg_indexes = [start_symbol]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(model.device)
        trg_mask = model.make_trg_mask(trg_tensor)
        
        output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        
        if pred_token == trg_pad_idx:
            break
    
    return trg_indexes

# Example usage
src_sentence = torch.LongTensor([[1, 5, 6, 4, 3, 9, 5, 2]]).to(model.device)
translated_sentence = greedy_decode(model, src_sentence, max_len=50, start_symbol=2)
print(translated_sentence)
```

Slide 11: Attention Visualization

Visualizing attention weights can provide insights into how the model focuses on different parts of the input when generating each output token. Here's a simple example of how to extract and visualize attention weights.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention, src_words, trg_words):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(attention, xticklabels=src_words, yticklabels=trg_words, ax=ax)
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    plt.title('Attention Weights')
    plt.show()

# Assuming we have a method to extract attention weights from the model
attention_weights = model.get_attention_weights(src, trg)

src_words = ['Hello', 'how', 'are', 'you', '?']
trg_words = ['Hola', 'cómo', 'estás', '?']

visualize_attention(attention_weights[0], src_words, trg_words)
```

Slide 12: Real-life Example: Machine Translation

One common application of Transformer models is machine translation. Here's a simplified example of how to use a pre-trained Transformer for translating English to Spanish.

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example usage
english_text = "Hello, how are you today?"
spanish_translation = translate(english_text)
print(f"English: {english_text}")
print(f"Spanish: {spanish_translation}")
```

Slide 13: Real-life Example: Text Summarization

Another application of Transformer models is text summarization. Here's a simple example using a pre-trained model for abstractive summarization.

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
The Transformer model has revolutionized natural language processing tasks. 
It uses self-attention mechanisms to process sequential data without the need 
for recurrent or convolutional layers. This architecture has led to significant 
improvements in various NLP tasks, including translation, summarization, and 
question answering. The model's ability to capture long-range dependencies and 
its parallelizable nature make it highly effective and efficient.
"""

summary = summarizer(text, max_length=50, min_length=10, do_sample=False)
print(summary[0]['summary_text'])
```

Slide 14: Additional Resources

For those interested in diving deeper into Transformer models, here are some valuable resources:

1.  Original Transformer paper: "Attention Is All You Need" by Vaswani et al. (2017) ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "The Illustrated Transformer" by Jay Alammar A visual guide to understanding Transformers
3.  "The Annotated Transformer" by Harvard NLP An annotated implementation of the Transformer model
4.  "Transformers from Scratch" tutorial series by Peter Bloem A comprehensive guide to implementing Transformers from the ground up

These resources provide in-depth explanations, visualizations, and implementations to enhance your understanding of Transformer models.

