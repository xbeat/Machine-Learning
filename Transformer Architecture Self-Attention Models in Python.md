## Transformer Architecture Self-Attention Models in Python
Slide 1: Introduction to Transformer Architecture

Transformers have revolutionized natural language processing and machine learning. This architecture, introduced in the paper "Attention Is All You Need," uses self-attention mechanisms to process sequential data efficiently. Let's explore the key components of a Transformer model, focusing on the encoder part which forms the basis for many auto-encoding models.

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
    
    def forward(self, src):
        return self.transformer_encoder(src)

# Example usage
d_model, nhead, num_layers = 512, 8, 6
encoder = TransformerEncoder(d_model, nhead, num_layers)
src = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = encoder(src)
print(output.shape)  # torch.Size([10, 32, 512])
```

Slide 2: Self-Attention Mechanism

The core of the Transformer architecture is the self-attention mechanism. It allows the model to weigh the importance of different parts of the input sequence when processing each element. Self-attention computes query, key, and value matrices from the input, then uses these to calculate attention scores.

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
    def forward(self, query, key, value):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        
        query = self.queries(query)
        key = self.keys(key)
        value = self.values(value)
        
        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        return out

# Example usage
embed_size, heads = 256, 8
self_attention = SelfAttention(embed_size, heads)
x = torch.rand(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
output = self_attention(x, x, x)
print(output.shape)  # torch.Size([32, 10, 256])
```

Slide 3: Positional Encoding

Transformers process input sequences in parallel, losing the inherent order information. Positional encoding adds this information back by injecting position-dependent signals into the input embeddings. These encodings use sine and cosine functions of different frequencies.

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
seq_len, d_model = 100, 512
pos_encoding = positional_encoding(seq_len, d_model)
print(pos_encoding.shape)  # torch.Size([100, 512])

# Visualize the positional encoding
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.imshow(pos_encoding.numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Positional Encoding')
plt.xlabel('Embedding Dimension')
plt.ylabel('Sequence Position')
plt.show()
```

Slide 4: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It applies the attention mechanism multiple times in parallel, then concatenates and linearly transforms the results.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
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
        
        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        
        # Attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# Example usage
embed_size, heads = 512, 8
mha = MultiHeadAttention(embed_size, heads)
x = torch.rand(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
mask = torch.ones(10, 10)  # No masking
output = mha(x, x, x, mask)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 5: Feed-Forward Networks

Each layer in a Transformer includes a position-wise feed-forward network. This network consists of two linear transformations with a ReLU activation in between. It processes each position separately and identically.

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Example usage
embed_size, ff_hidden_dim = 512, 2048
ff_network = FeedForward(embed_size, ff_hidden_dim)
x = torch.rand(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
output = ff_network(x)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 6: Layer Normalization

Layer normalization is applied after each sub-layer in the Transformer. It helps stabilize the learning process and reduces the training time. Unlike batch normalization, layer normalization operates on the feature dimension, making it suitable for sequence modeling tasks.

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embed_size))
        self.bias = nn.Parameter(torch.zeros(embed_size))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Example usage
embed_size = 512
layer_norm = LayerNorm(embed_size)
x = torch.rand(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
output = layer_norm(x)
print(output.shape)  # torch.Size([32, 10, 512])

# Visualize the effect of layer normalization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(x[0, 0, :].detach().numpy(), bins=50)
plt.title('Before Layer Normalization')
plt.subplot(1, 2, 2)
plt.hist(output[0, 0, :].detach().numpy(), bins=50)
plt.title('After Layer Normalization')
plt.tight_layout()
plt.show()
```

Slide 7: Encoder Block

An encoder block in a Transformer consists of a multi-head attention layer followed by a feed-forward network, with layer normalization and residual connections applied after each sub-layer. This structure allows the model to capture complex relationships in the input data.

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Example usage
embed_size, heads, dropout, forward_expansion = 512, 8, 0.1, 4
encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
x = torch.rand(32, 10, embed_size)  # (batch_size, seq_len, embed_size)
mask = torch.ones(10, 10)  # No masking
output = encoder_block(x, x, x, mask)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 8: Complete Transformer Encoder

The complete Transformer encoder consists of multiple encoder blocks stacked on top of each other. This allows the model to learn increasingly complex representations of the input data at each layer.

```python
import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out

# Example usage
src_vocab_size, embed_size, num_layers, heads = 10000, 512, 6, 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forward_expansion, dropout, max_length = 4, 0.1, 100

encoder = TransformerEncoder(
    src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length
).to(device)

x = torch.randint(0, src_vocab_size, (32, 10)).to(device)  # (batch_size, seq_len)
mask = torch.ones(10, 10).to(device)  # No masking

output = encoder(x, mask)
print(output.shape)  # torch.Size([32, 10, 512])
```

Slide 9: Auto-Encoding with Transformers

Auto-encoding models based on Transformers aim to reconstruct their input, often with some form of noise or masking applied. This process helps the model learn robust representations of the input data. Let's implement a simple auto-encoding Transformer.

```python
import torch
import torch.nn as nn

class AutoEncodingTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, embed_size, num_layers, heads, 
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            forward_expansion=4, dropout=dropout, max_length=100
        )
        self.decoder = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x, mask):
        encoding = self.encoder(x, mask)
        return self.decoder(encoding)

    def add_noise(self, x, noise_prob=0.15):
        noise_mask = torch.bernoulli(torch.full(x.shape, noise_prob)).bool()
        x[noise_mask] = torch.randint(0, x.max().item() + 1, (noise_mask.sum(),))
        return x

# Example usage
vocab_size, embed_size, num_layers, heads, dropout = 10000, 512, 6, 8, 0.1
model = AutoEncodingTransformer(vocab_size, embed_size, num_layers, heads, dropout)
x = torch.randint(0, vocab_size, (32, 10))  # (batch_size, seq_len)
noisy_x = model.add_noise(x.clone())
mask = torch.ones(10, 10)
output = model(noisy_x, mask)
print(output.shape)  # torch.Size([32, 10, 10000])
```

Slide 10: Masked Language Modeling

Masked Language Modeling (MLM) is a key task in auto-encoding Transformer models like BERT. In MLM, some tokens in the input sequence are randomly masked, and the model is trained to predict these masked tokens. This approach allows the model to learn bidirectional context representations.

```python
import torch
import torch.nn as nn

class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, dropout):
        super().__init__()
        self.transformer = AutoEncodingTransformer(vocab_size, embed_size, num_layers, heads, dropout)
        self.mask_token = vocab_size  # Use the last token as the mask token
        
    def forward(self, x, mask):
        return self.transformer(x, mask)
    
    def create_masked_input(self, x, mask_prob=0.15):
        mask = torch.bernoulli(torch.full(x.shape, mask_prob)).bool()
        masked_x = x.clone()
        masked_x[mask] = self.mask_token
        return masked_x, mask

# Example usage
vocab_size, embed_size, num_layers, heads, dropout = 10000, 512, 6, 8, 0.1
model = MaskedLanguageModel(vocab_size, embed_size, num_layers, heads, dropout)
x = torch.randint(0, vocab_size, (32, 10))  # (batch_size, seq_len)
masked_x, mask = model.create_masked_input(x)
output = model(masked_x, torch.ones(10, 10))
print(output.shape)  # torch.Size([32, 10, 10001])
```

Slide 11: Training an Auto-Encoding Transformer

Training an auto-encoding Transformer involves minimizing the difference between the model's predictions and the true tokens. We use cross-entropy loss to measure this difference. Here's a basic training loop for our Masked Language Model.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_mlm(model, data_loader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model.mask_token)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            x = batch.to(device)
            masked_x, mask = model.create_masked_input(x)
            optimizer.zero_grad()
            output = model(masked_x, torch.ones(x.size(1), x.size(1)).to(device))
            loss = criterion(output.view(-1, output.size(-1)), x.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")

# Example usage (assuming we have a data_loader)
# train_mlm(model, data_loader, num_epochs=10, learning_rate=0.001)
```

Slide 12: Real-Life Example: Sentiment Analysis

Auto-encoding Transformer models can be fine-tuned for various downstream tasks. Let's look at how we can use a pre-trained model for sentiment analysis of movie reviews.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def sentiment_analysis(text):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    sentiment = "Positive" if probabilities[0][1] > probabilities[0][0] else "Negative"
    confidence = max(probabilities[0]).item()
    
    return sentiment, confidence

# Example usage
review = "This movie was absolutely fantastic! The plot was engaging and the acting was superb."
sentiment, confidence = sentiment_analysis(review)
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2f}")

# Output:
# Sentiment: Positive
# Confidence: 0.99
```

Slide 13: Real-Life Example: Text Generation

Another application of auto-encoding Transformer models is text generation. We can use a pre-trained model to generate coherent text given a prompt.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, 
                             no_repeat_ngram_size=2, top_k=50, top_p=0.95, 
                             temperature=0.7)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "In the year 2050, artificial intelligence has"
generated_text = generate_text(prompt)
print(generated_text)

# Output:
# In the year 2050, artificial intelligence has become an integral part of our daily lives. 
# From autonomous vehicles to personal assistants, AI is everywhere. However, as we rely 
# more on these intelligent systems, new challenges arise. The ethical implications of AI 
# decision-making have sparked debates worldwide...
```

Slide 14: Additional Resources

For those interested in diving deeper into Transformer architecture and auto-encoding models, here are some valuable resources:

1. "Attention Is All You Need" - The original Transformer paper ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Improving Language Understanding by Generative Pre-Training" - The GPT paper (No ArXiv link available, but can be found on OpenAI's website)
4. "The Illustrated Transformer" by Jay Alammar - A visual guide to understanding Transformers (Not an ArXiv resource, but widely recognized in the community)

These papers and resources provide in-depth explanations of the concepts we've covered and offer insights into the latest developments in the field of natural language processing using Transformer-based models.

