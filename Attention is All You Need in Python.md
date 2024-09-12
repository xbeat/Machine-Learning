## Attention is All You Need in Python
Slide 1: The Transformer Architecture

The Transformer architecture, introduced in the paper "Attention is All You Need" by Vaswani et al., revolutionized natural language processing. This model relies solely on attention mechanisms, eschewing recurrence and convolutions. It has become the foundation for many state-of-the-art language models.

```python
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt):
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)
        output = self.transformer(src_embed, tgt_embed)
        return self.linear(output)
```

Slide 2: Self-Attention Mechanism

The core of the Transformer is the self-attention mechanism. It allows the model to weigh the importance of different words in a sentence when processing each word. This mechanism enables the model to capture long-range dependencies effectively.

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

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

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

Slide 3: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It improves the model's ability to focus on different positions and capture various aspects of the input.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

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

Slide 4: Positional Encoding

Since the Transformer doesn't use recurrence or convolution, it needs a way to capture the order of words in a sequence. Positional encoding adds information about the position of each word to its embedding.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
```

Slide 5: Feed-Forward Networks

Each layer in the Transformer contains a fully connected feed-forward network. This network consists of two linear transformations with a ReLU activation in between, allowing the model to introduce non-linearity and process the attention output.

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

Slide 6: Layer Normalization

Layer normalization is applied after each sub-layer in the Transformer. It helps stabilize the learning process and reduces the training time by normalizing the inputs across the features.

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

Slide 7: Encoder Layer

The encoder layer is a fundamental building block of the Transformer's encoder. It consists of a multi-head self-attention mechanism followed by a position-wise feed-forward network, with layer normalization and residual connections.

```python
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

Slide 8: Decoder Layer

The decoder layer is similar to the encoder layer but includes an additional multi-head attention layer that attends to the encoder's output. This allows the decoder to focus on relevant parts of the input sequence.

```python
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.cross_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
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

Slide 9: Masked Attention

In the decoder, we use masked attention to prevent the model from attending to future tokens during training. This ensures that the prediction for a given position only depends on known outputs at previous positions.

```python
import torch

def create_mask(seq):
    seq_len = seq.size(1)
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
```

Slide 10: Training the Transformer

Training a Transformer involves defining a loss function (typically cross-entropy for language tasks) and using an optimizer like Adam. The model is trained on large datasets, often using techniques like learning rate warmup and label smoothing.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_data, epochs, lr):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    model.train()
    for epoch in range(epochs):
        for batch in train_data:
            src, tgt = batch
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.contiguous().view(-1))
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

Slide 11: Inference with the Transformer

During inference, we use beam search or greedy decoding to generate sequences. The model generates one token at a time, using the previously generated tokens as input for the next prediction.

```python
import torch

def greedy_decode(model, src, max_len, start_symbol):
    memory = model.encode(src)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
    return ys
```

Slide 12: Real-Life Example: Machine Translation

One common application of the Transformer is machine translation. Here's a simple example of how to use a pre-trained Transformer model for translating English to French:

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

english_text = "Hello, how are you?"
french_translation = translate(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")
```

Slide 13: Real-Life Example: Text Summarization

Another practical application of the Transformer is text summarization. Here's an example using a pre-trained model:

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Transformer model has revolutionized natural language processing. 
Introduced in the paper "Attention is All You Need", it relies solely on attention mechanisms, 
eschewing recurrence and convolutions. This architecture has become the foundation for many 
state-of-the-art language models and has been applied to various tasks such as translation, 
summarization, and question answering.
"""

summary = summarizer(article, max_length=50, min_length=10, do_sample=False)
print(summary[0]['summary_text'])
```

Slide 14: Limitations and Future Directions

While the Transformer has been incredibly successful, it has limitations. The quadratic complexity in sequence length can be problematic for very long sequences. Recent research has focused on developing more efficient variants like the Reformer and Linformer. Future directions include exploring ways to incorporate external knowledge and improving the model's ability to handle multimodal inputs.

```python
import math

def attention_complexity(seq_length):
    return seq_length ** 2

def linear_attention_complexity(seq_length):
    return seq_length

seq_lengths = [100, 1000, 10000]
for length in seq_lengths:
    print(f"Sequence length: {length}")
    print(f"Standard attention complexity: {attention_complexity(length)}")
    print(f"Linear attention complexity: {linear_attention_complexity(length)}")
    print()
```

Slide 15: Transformer Variants

Researchers have proposed various modifications to the original Transformer architecture to address its limitations or improve its performance in specific tasks. Some notable variants include BERT, GPT, T5, and ELECTRA.

```python
class TransformerVariant:
    def __init__(self, name, key_features):
        self.name = name
        self.key_features = key_features

variants = [
    TransformerVariant("BERT", ["Bidirectional", "Pre-training on masked language modeling"]),
    TransformerVariant("GPT", ["Unidirectional (left-to-right)", "Generative pre-training"]),
    TransformerVariant("T5", ["Text-to-Text framework", "Unified approach to NLP tasks"]),
    TransformerVariant("ELECTRA", ["Replaced token detection", "More sample-efficient pre-training"])
]

for variant in variants:
    print(f"{variant.name}:")
    for feature in variant.key_features:
        print(f"- {feature}")
    print()
```

Slide 16: Additional Resources

For those interested in delving deeper into the Transformer architecture and its applications, here are some valuable resources:

1. Original paper: "Attention Is All You Need" by Vaswani et al. (2017) ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "The Illustrated Transformer" by Jay Alammar A visual and intuitive explanation of the Transformer architecture
3. "Transformers from Scratch" by Peter Bloem A detailed tutorial on implementing Transformers
4. "The Annotated Transformer" by Harvard NLP An annotated implementation of the Transformer in PyTorch

These resources provide a mix of theoretical foundations and practical implementations to further your understanding of the Transformer architecture and its impact on natural language processing.

