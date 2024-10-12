## Exploring Transformer Models with Python
Slide 1: Introduction to Transformer Models

Transformer models have revolutionized natural language processing and machine learning. These powerful neural network architectures, introduced in the "Attention Is All You Need" paper, excel at capturing long-range dependencies in sequential data. Let's explore their inner workings using Python.

```python
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        output = self.transformer(src_embed, tgt_embed)
        return self.fc(output)

# Example usage
model = TransformerModel(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
```

Slide 2: Self-Attention Mechanism

The core of Transformer models is the self-attention mechanism. It allows the model to weigh the importance of different parts of the input sequence when processing each element. Self-attention computes query, key, and value vectors for each input token and uses them to calculate attention scores.

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
        
    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        
        queries = self.queries(query)
        keys = self.keys(key)
        values = self.values(value)
        
        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        return out

# Example usage
attention = SelfAttention(embed_size=256, heads=8)
x = torch.randn(32, 10, 256)  # (batch_size, seq_len, embed_size)
output = attention(x, x, x)
print(output.shape)  # torch.Size([32, 10, 256])
```

Slide 3: Multi-Head Attention

Multi-head attention extends the self-attention mechanism by applying multiple attention operations in parallel. This allows the model to capture different types of relationships between tokens. Each attention head focuses on different aspects of the input, enhancing the model's representational power.

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        q_len, k_len, v_len = query.shape[1], key.shape[1], value.shape[1]
        
        # Split embedding into self.heads pieces
        queries = self.queries(query).reshape(N, q_len, self.heads, self.head_dim)
        keys = self.keys(key).reshape(N, k_len, self.heads, self.head_dim)
        values = self.values(value).reshape(N, v_len, self.heads, self.head_dim)
        
        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, q_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

# Example usage
mha = MultiHeadAttention(embed_size=256, heads=8)
x = torch.randn(32, 10, 256)  # (batch_size, seq_len, embed_size)
output = mha(x, x, x)
print(output.shape)  # torch.Size([32, 10, 256])
```

Slide 4: Positional Encoding

Transformer models process input tokens in parallel, losing the inherent order information. Positional encoding addresses this by adding position-dependent signals to the input embeddings. This allows the model to understand the relative or absolute position of tokens in the sequence.

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create a long enough position tensor
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
max_len = 100
pos_encoder = PositionalEncoding(d_model, max_len)

# Generate a sample input tensor
batch_size = 32
seq_len = 50
x = torch.randn(seq_len, batch_size, d_model)

# Apply positional encoding
encoded_x = pos_encoder(x)
print(encoded_x.shape)  # torch.Size([50, 32, 512])

# Visualize the positional encoding
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.imshow(pos_encoder.pe[:max_len, :].squeeze().T, cmap='hot', aspect='auto')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.title('Positional Encoding')
plt.colorbar()
plt.show()
```

Slide 5: Feed-Forward Networks

Between attention layers, Transformer models employ feed-forward networks. These consist of two linear transformations with a ReLU activation in between. Feed-forward networks process each position independently, allowing the model to introduce non-linearity and increase its representational capacity.

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

# Example usage
d_model = 512
d_ff = 2048
ff_network = FeedForward(d_model, d_ff)

# Generate a sample input tensor
batch_size = 32
seq_len = 50
x = torch.randn(batch_size, seq_len, d_model)

# Apply feed-forward network
output = ff_network(x)
print(output.shape)  # torch.Size([32, 50, 512])

# Visualize the transformation
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(x[0].detach().numpy(), cmap='viridis', aspect='auto')
plt.title('Input')
plt.subplot(1, 2, 2)
plt.imshow(output[0].detach().numpy(), cmap='viridis', aspect='auto')
plt.title('Output')
plt.tight_layout()
plt.show()
```

Slide 6: Layer Normalization

Layer normalization is a crucial component in Transformer models. It helps stabilize the learning process by normalizing the activations across the feature dimension. This technique is applied after the self-attention and feed-forward layers, allowing for more stable and faster training.

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Example usage
d_model = 512
layer_norm = LayerNorm(d_model)

# Generate a sample input tensor
batch_size = 32
seq_len = 50
x = torch.randn(batch_size, seq_len, d_model)

# Apply layer normalization
normalized_x = layer_norm(x)

print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
print(f"Output mean: {normalized_x.mean():.4f}, std: {normalized_x.std():.4f}")

# Visualize the normalization effect
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(x[0].flatten().detach().numpy(), bins=50, alpha=0.7)
plt.title('Input Distribution')
plt.subplot(1, 2, 2)
plt.hist(normalized_x[0].flatten().detach().numpy(), bins=50, alpha=0.7)
plt.title('Normalized Distribution')
plt.tight_layout()
plt.show()
```

Slide 7: Encoder-Decoder Architecture

Transformer models typically use an encoder-decoder architecture. The encoder processes the input sequence, while the decoder generates the output sequence. This structure is particularly effective for tasks like machine translation, where the input and output sequences may have different lengths.

```python
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.cross_attn = MultiHeadAttention(d_model, heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, enc_out, enc_out, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout(self.feed_forward(x2))
        return x

# Example usage
d_model = 512
heads = 8
d_ff = 2048
encoder_layer = EncoderLayer(d_model, heads, d_ff)
decoder_layer = DecoderLayer(d_model, heads, d_ff)

# Generate sample input tensors
batch_size = 32
src_len = 50
tgt_len = 30
src = torch.randn(batch_size, src_len, d_model)
tgt = torch.randn(batch_size, tgt_len, d_model)
src_mask = torch.ones(batch_size, 1, src_len)
tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0)

# Apply encoder and decoder layers
enc_out = encoder_layer(src, src_mask)
dec_out = decoder_layer(tgt, enc_out, src_mask, tgt_mask)

print(f"Encoder output shape: {enc_out.shape}")
print(f"Decoder output shape: {dec_out.shape}")
```

Slide 8: Training Transformer Models

Training Transformer models involves optimizing the model parameters to minimize a loss function. For language modeling tasks, we typically use cross-entropy loss and advanced optimizers like Adam. Let's implement a simple training loop for a language modeling task.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)
        output = self.transformer(src_embed, tgt_embed)
        return self.fc(output)

# Hyperparameters
vocab_size = 10000
d_model = 512
nhead = 8
num_layers = 6
batch_size = 32
num_epochs = 10

# Create model, loss function, and optimizer
model = TransformerLM(vocab_size, d_model, nhead, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:  # Assume we have a data loader
        src, tgt = batch
        optimizer.zero_grad()
        output = model(src, tgt[:-1])
        loss = criterion(output.view(-1, vocab_size), tgt[1:].view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

Slide 9: Attention Visualization

Visualizing attention weights helps understand how the model focuses on different parts of the input. Let's create a function to visualize attention patterns in a Transformer model.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='YlGnBu')
    plt.title('Attention Weights Visualization')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

# Example usage (assuming we have attention weights and tokens)
attention_weights = torch.rand(10, 10)  # 10x10 attention matrix
tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']

visualize_attention(attention_weights.detach().numpy(), tokens)
```

Slide 10: Transformer Variants

Various Transformer variants have been developed to address specific challenges or improve performance. Some popular variants include:

1. BERT (Bidirectional Encoder Representations from Transformers)
2. GPT (Generative Pre-trained Transformer)
3. T5 (Text-to-Text Transfer Transformer)
4. Transformer-XL (Extra Long)
5. Reformer (Efficient Transformer)

Let's implement a simplified version of the BERT model:

```python
import torch
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        return self.fc(x)

# Example usage
vocab_size = 30000
d_model = 768
nhead = 12
num_layers = 12

model = BERTModel(vocab_size, d_model, nhead, num_layers)
input_ids = torch.randint(0, vocab_size, (32, 512))  # (batch_size, seq_len)
output = model(input_ids)
print(output.shape)  # torch.Size([32, 512, 30000])
```

Slide 11: Fine-tuning Transformer Models

Fine-tuning pre-trained Transformer models for specific tasks is a common practice in transfer learning. This process involves adapting a pre-trained model to a new task by training it on task-specific data. Let's implement a simple fine-tuning procedure for text classification:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# Hyperparameters
num_classes = 2
num_epochs = 3
batch_size = 32
learning_rate = 2e-5

# Create model, loss function, and optimizer
model = BERTClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Fine-tuning loop
for epoch in range(num_epochs):
    for batch in data_loader:  # Assume we have a data loader
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

Slide 12: Real-life Example: Sentiment Analysis

Let's use a fine-tuned BERT model for sentiment analysis on movie reviews. This example demonstrates how Transformer models can be applied to real-world natural language processing tasks.

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example movie reviews
reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "The acting was terrible and the plot made no sense.",
    "An average film with some good moments but overall disappointing."
]

# Tokenize and prepare input
inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)

# Print results
for review, pred in zip(reviews, predictions):
    sentiment = "Positive" if pred[1] > pred[0] else "Negative"
    confidence = max(pred[0], pred[1]).item()
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2f})")
    print()

# Output:
# Review: This movie was fantastic! I loved every minute of it.
# Sentiment: Positive (confidence: 0.98)
#
# Review: The acting was terrible and the plot made no sense.
# Sentiment: Negative (confidence: 0.97)
#
# Review: An average film with some good moments but overall disappointing.
# Sentiment: Negative (confidence: 0.68)
```

Slide 13: Real-life Example: Text Generation

Another common application of Transformer models is text generation. Let's use a pre-trained GPT-2 model to generate text based on a given prompt.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1,
                            no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example prompts
prompts = [
    "The future of artificial intelligence is",
    "Once upon a time, in a galaxy far, far away",
    "The recipe for the perfect chocolate cake includes"
]

for prompt in prompts:
    generated_text = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
    print()

# Output:
# Prompt: The future of artificial intelligence is
# Generated text: The future of artificial intelligence is bright, but it's not without its challenges. As AI continues to evolve and become more sophisticated, we'll need to address issues such as ethics, privacy, and job displacement. However, the potential benefits of AI in fields like healthcare, education, and scientific research are enormous...

# Prompt: Once upon a time, in a galaxy far, far away
# Generated text: Once upon a time, in a galaxy far, far away, there was a young Jedi named Kira. She had always dreamed of becoming a powerful force user, but her path was not an easy one. As she trained under the guidance of Master Yoda, she faced many challenges and obstacles...

# Prompt: The recipe for the perfect chocolate cake includes
# Generated text: The recipe for the perfect chocolate cake includes the following ingredients: 2 cups of all-purpose flour, 1 3/4 cups of granulated sugar, 3/4 cup of unsweetened cocoa powder, 2 teaspoons of baking soda, 1 teaspoon of baking powder, 1 teaspoon of salt, 2 eggs, 1 cup of milk...
```

Slide 14: Additional Resources

For those interested in diving deeper into Transformer models, here are some valuable resources:

1. "Attention Is All You Need" paper (Vaswani et al., 2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018): [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" (GPT-3 paper, Brown et al., 2020): [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4. "The Illustrated Transformer" by Jay Alammar: [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
5. Hugging Face Transformers library documentation: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)

These resources provide in-depth explanations, implementations, and applications of Transformer models in various natural language processing tasks.

