## Guide to Self-Attention Network Architecture in Python
Slide 1: Introduction to Self-Attention Networks

Self-attention networks are a fundamental component of modern deep learning architectures, particularly in natural language processing. They allow models to weigh the importance of different parts of the input when processing sequences, enabling more effective learning of long-range dependencies.

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
        # Implementation details in the following slides
        pass
```

Slide 2: Self-Attention Mechanism

The self-attention mechanism computes attention scores between all pairs of positions in a sequence. It uses queries, keys, and values derived from the input to determine how much focus to place on other parts of the input when encoding a particular element.

```python
def forward(self, x):
    N, seq_length, _ = x.shape
    
    # Split embeddings into multiple heads
    queries = self.queries(x).reshape(N, seq_length, self.heads, self.head_dim)
    keys = self.keys(x).reshape(N, seq_length, self.heads, self.head_dim)
    values = self.values(x).reshape(N, seq_length, self.heads, self.head_dim)
    
    # Compute attention scores
    energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
    attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
    
    # Apply attention to values
    out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
    out = out.reshape(N, seq_length, self.embed_size)
    
    return self.fc_out(out)
```

Slide 3: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It improves the model's ability to focus on different positions and capture various aspects of the input sequence.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.attention = SelfAttention(embed_size, heads)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        out = self.attention(x)
        return self.fc_out(out)
```

Slide 4: Positional Encoding

Positional encoding is crucial in self-attention networks as the attention mechanism itself is permutation-invariant. It adds information about the position of tokens in the sequence, allowing the model to leverage sequential information.

```python
import numpy as np

def positional_encoding(seq_length, embed_size):
    pos_encoding = np.zeros((seq_length, embed_size))
    positions = np.arange(0, seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(10000.0) / embed_size))
    
    pos_encoding[:, 0::2] = np.sin(positions * div_term)
    pos_encoding[:, 1::2] = np.cos(positions * div_term)
    
    return torch.FloatTensor(pos_encoding)

# Usage
seq_length, embed_size = 100, 512
pos_encoding = positional_encoding(seq_length, embed_size)
```

Slide 5: Feed-Forward Networks

Feed-forward networks are used in conjunction with self-attention layers to introduce non-linearity and increase the model's capacity to learn complex patterns. They typically consist of two linear transformations with a ReLU activation in between.

```python
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Usage
embed_size, ff_hidden_dim = 512, 2048
ff_layer = FeedForward(embed_size, ff_hidden_dim)
```

Slide 6: Layer Normalization

Layer normalization is applied after each sub-layer in the self-attention network. It helps stabilize the learning process and reduces the training time by normalizing the inputs across the features.

```python
class LayerNorm(nn.Module):
    def __init__(self, embed_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

# Usage
layer_norm = LayerNorm(embed_size)
normalized_output = layer_norm(attention_output)
```

Slide 7: Transformer Encoder Layer

The Transformer encoder layer combines self-attention, feed-forward networks, and layer normalization. It's the building block for many modern NLP architectures.

```python
class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = LayerNorm(embed_size)
        self.ff = FeedForward(embed_size, ff_hidden_dim)
        self.norm2 = LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Usage
encoder_layer = EncoderLayer(embed_size=512, heads=8, ff_hidden_dim=2048)
encoded_output = encoder_layer(input_sequence)
```

Slide 8: Masked Self-Attention

Masked self-attention is used in decoder architectures to prevent the model from attending to future tokens during training. It's crucial for tasks like language generation where we want to predict the next token based only on previous tokens.

```python
def create_mask(seq_length):
    mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0)
    return mask

class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MaskedSelfAttention, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

    def forward(self, x):
        seq_length = x.size(1)
        mask = create_mask(seq_length).to(x.device)
        return self.attention(x, mask)

# Usage
masked_attention = MaskedSelfAttention(embed_size=512, heads=8)
masked_output = masked_attention(input_sequence)
```

Slide 9: Real-life Example: Sentiment Analysis

Self-attention networks can be used for sentiment analysis tasks, where the model needs to understand the context and relationships between words to determine the overall sentiment of a text.

```python
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_classes):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_layer = EncoderLayer(embed_size, num_heads, ff_hidden_dim=embed_size*4)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder_layer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# Usage
vocab_size, embed_size, num_heads, num_classes = 10000, 256, 4, 2
model = SentimentClassifier(vocab_size, embed_size, num_heads, num_classes)
input_ids = torch.randint(0, vocab_size, (32, 50))  # Batch of 32 sequences, each 50 tokens long
output = model(input_ids)
print(output.shape)  # torch.Size([32, 2])
```

Slide 10: Real-life Example: Named Entity Recognition

Self-attention networks are effective for named entity recognition tasks, where the model needs to identify and classify named entities in text by understanding the context of each word.

```python
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_entities):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_layer = EncoderLayer(embed_size, num_heads, ff_hidden_dim=embed_size*4)
        self.fc = nn.Linear(embed_size, num_entities)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder_layer(x)
        return self.fc(x)

# Usage
vocab_size, embed_size, num_heads, num_entities = 10000, 256, 4, 10
model = NERModel(vocab_size, embed_size, num_heads, num_entities)
input_ids = torch.randint(0, vocab_size, (32, 50))  # Batch of 32 sequences, each 50 tokens long
output = model(input_ids)
print(output.shape)  # torch.Size([32, 50, 10])
```

Slide 11: Attention Visualization

Visualizing attention weights can provide insights into how the model is processing the input. This can be particularly useful for debugging and understanding model behavior.

```python
import matplotlib.pyplot as plt

def visualize_attention(attention_weights, tokens):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')

    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text = ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title("Attention Weights")
    fig.tight_layout()
    plt.show()

# Usage
tokens = ["The", "cat", "sat", "on", "the", "mat"]
attention_weights = torch.rand(len(tokens), len(tokens))
visualize_attention(attention_weights.numpy(), tokens)
```

Slide 12: Training Techniques

Training self-attention networks often involves techniques like learning rate warmup and gradient clipping to stabilize the learning process and prevent gradient explosions.

```python
import torch.optim as optim

def train_model(model, train_loader, num_epochs, warmup_steps=4000):
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss()

    def lr_lambda(step):
        return min((step + 1) ** -0.5, step * (warmup_steps ** -1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels = batch
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

# Usage
# Assuming 'model' and 'train_loader' are defined
train_model(model, train_loader, num_epochs=10)
```

Slide 13: Evaluation and Inference

When using self-attention networks for inference, it's important to consider techniques like beam search for sequence generation tasks or thresholding for classification tasks.

```python
def generate_sequence(model, start_token, max_length=50, temperature=1.0):
    model.eval()
    current_seq = [start_token]
    
    with torch.no_grad():
        for _ in range(max_length - 1):
            input_seq = torch.tensor(current_seq).unsqueeze(0)
            logits = model(input_seq)
            next_token_logits = logits[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).item()
            
            if next_token == end_token:
                break
            
            current_seq.append(next_token)
    
    return current_seq

# Usage
start_token, end_token = 1, 2  # Assuming 1 is start token and 2 is end token
generated_sequence = generate_sequence(model, start_token)
print("Generated sequence:", generated_sequence)
```

Slide 14: Performance Optimization

Optimizing self-attention networks for performance is crucial, especially for large-scale applications. Techniques like mixed-precision training and model parallelism can significantly speed up training and inference.

```python
import torch.cuda.amp as amp

def train_with_mixed_precision(model, train_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels = batch

            with amp.autocast():
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

# Usage
# Assuming 'model' and 'train_loader' are defined
train_with_mixed_precision(model, train_loader, num_epochs=10)
```

Slide 15: Additional Resources

For further exploration of self-attention networks and transformer architectures, consider the following resources:

1. "Attention Is All You Need" (Vaswani et al., 2017): The original paper introducing the transformer architecture. Available at: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "

