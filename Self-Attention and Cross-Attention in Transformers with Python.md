## Self-Attention and Cross-Attention in Transformers with Python
Slide 1: Introduction to Self-Attention & Cross-Attention

Self-attention and cross-attention are fundamental mechanisms in transformer architectures. They allow models to weigh the importance of different parts of the input sequence when processing each element. This slideshow will explore these concepts with practical Python examples.

```python
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        # Implement attention mechanism
        pass
```

Slide 2: Self-Attention: The Basics

Self-attention allows a sequence to attend to itself, capturing relationships between different positions within the same sequence. It's a key component in understanding context and dependencies in sequential data.

```python
def self_attention(x):
    # x: input tensor of shape (batch_size, seq_len, embed_dim)
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention to values
    return torch.matmul(attn_weights, v)
```

Slide 3: Visualizing Self-Attention

Let's create a simple visualization of self-attention weights to better understand how it works.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attn_weights, tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights.detach().numpy(), annot=True, cmap='coolwarm', xticklabels=tokens, yticklabels=tokens)
    plt.title('Self-Attention Weights')
    plt.xlabel('Key/Value Tokens')
    plt.ylabel('Query Tokens')
    plt.show()

# Example usage
tokens = ['I', 'love', 'natural', 'language', 'processing']
attn_weights = torch.rand(5, 5)  # Random weights for demonstration
visualize_attention(attn_weights, tokens)
```

Slide 4: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces, enhancing the model's ability to capture various aspects of the input.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(context)
```

Slide 5: Cross-Attention: Bridging Sequences

Cross-attention allows a model to attend to a different sequence than the one being processed. This is crucial in tasks like machine translation, where the model needs to align source and target sequences.

```python
def cross_attention(query, key, value):
    # query: tensor of shape (batch_size, query_len, embed_dim)
    # key, value: tensors of shape (batch_size, key_len, embed_dim)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)

# Example usage
query = torch.rand(1, 5, 64)  # 5 query tokens
key = value = torch.rand(1, 7, 64)  # 7 key/value tokens
output = cross_attention(query, key, value)
print(f"Cross-attention output shape: {output.shape}")
```

Slide 6: Real-Life Example: Text Summarization

Text summarization is a practical application of self-attention and cross-attention. The model attends to important parts of the input text to generate a concise summary.

```python
class SummarizationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.output = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, source, target):
        src_embed = self.embedding(source)
        tgt_embed = self.embedding(target)
        
        # Self-attention on source
        src_attended = self.self_attention(src_embed, src_embed, src_embed)
        
        # Cross-attention between target and source
        cross_attended = self.cross_attention(tgt_embed, src_attended, src_attended)
        
        # Feed-forward
        output = self.feed_forward(cross_attended)
        
        return self.output(output)

# Example usage
vocab_size, embed_dim, num_heads = 10000, 256, 8
model = SummarizationModel(vocab_size, embed_dim, num_heads)
source = torch.randint(0, vocab_size, (1, 100))  # 100 source tokens
target = torch.randint(0, vocab_size, (1, 20))   # 20 target tokens
output = model(source, target)
print(f"Summarization model output shape: {output.shape}")
```

Slide 7: Masked Self-Attention for Language Modeling

In language modeling tasks, we use masked self-attention to prevent the model from attending to future tokens during training. This ensures the model only uses past and present information to predict the next token.

```python
def masked_self_attention(x):
    seq_len = x.size(1)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    
    return torch.matmul(attn_weights, v)

# Example usage
x = torch.rand(1, 10, 64)  # Batch size 1, 10 tokens, 64 dimensions
output = masked_self_attention(x)
print(f"Masked self-attention output shape: {output.shape}")
```

Slide 8: Positional Encoding

Transformers don't inherently understand token positions. Positional encoding adds this information to the input embeddings, allowing the model to leverage sequence order.

```python
import math

def positional_encoding(seq_len, embed_dim):
    pos = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
    pe = torch.zeros(seq_len, embed_dim)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe

# Visualize positional encoding
seq_len, embed_dim = 100, 64
pe = positional_encoding(seq_len, embed_dim)
plt.figure(figsize=(10, 8))
sns.heatmap(pe.detach().numpy(), cmap='coolwarm')
plt.title('Positional Encoding')
plt.xlabel('Embedding Dimension')
plt.ylabel('Sequence Position')
plt.show()
```

Slide 9: Attention Scaling

Attention scaling is crucial for maintaining stable gradients, especially for large input sequences. We divide the dot product by the square root of the embedding dimension.

```python
def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value)

# Compare scaled vs. unscaled attention
q = k = v = torch.rand(1, 8, 10, 64)  # 8 heads, 10 tokens, 64 dimensions

unscaled_output = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1), v)
scaled_output = scaled_dot_product_attention(q, k, v)

print(f"Unscaled max value: {unscaled_output.max().item():.4f}")
print(f"Scaled max value: {scaled_output.max().item():.4f}")
```

Slide 10: Real-Life Example: Named Entity Recognition

Named Entity Recognition (NER) is another practical application of self-attention. The model attends to contextual information to classify words as entities like person names, organizations, or locations.

```python
class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.self_attention(x, x, x)
        x = self.feed_forward(x)
        return self.classifier(x)

# Example usage
vocab_size, embed_dim, num_heads, num_classes = 10000, 256, 8, 9  # 9 NER classes
model = NERModel(vocab_size, embed_dim, num_heads, num_classes)
input_sequence = torch.randint(0, vocab_size, (1, 50))  # Batch size 1, 50 tokens
output = model(input_sequence)
print(f"NER model output shape: {output.shape}")
```

Slide 11: Attention Visualization for NER

Let's visualize how the attention mechanism focuses on different parts of the input for Named Entity Recognition.

```python
def visualize_ner_attention(sentence, attention_weights, entities):
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(attention_weights, cmap='YlOrRd')
    
    ax.set_xticks(np.arange(len(sentence)))
    ax.set_yticks(np.arange(len(sentence)))
    ax.set_xticklabels(sentence)
    ax.set_yticklabels(sentence)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(sentence)):
        for j in range(len(sentence)):
            text = ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                           ha="center", va="center", color="black")
    
    for i, word in enumerate(sentence):
        if entities[i] != 'O':
            ax.get_xticklabels()[i].set_color('red')
            ax.get_xticklabels()[i].set_fontweight('bold')
    
    ax.set_title("Attention Weights for Named Entity Recognition")
    fig.tight_layout()
    plt.show()

# Example usage
sentence = ["John", "works", "at", "Google", "in", "New", "York"]
entities = ["B-PER", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC"]
attention_weights = torch.rand(len(sentence), len(sentence))
visualize_ner_attention(sentence, attention_weights, entities)
```

Slide 12: Efficient Attention: Linear Attention

As sequence lengths grow, the quadratic complexity of standard attention becomes problematic. Linear attention offers a more efficient alternative for long sequences.

```python
def linear_attention(query, key, value):
    q = torch.nn.functional.elu(query) + 1
    k = torch.nn.functional.elu(key) + 1
    v = value
    
    k_sum = k.sum(dim=-2)
    kv = torch.matmul(k.transpose(-2, -1), v)
    
    z = 1 / (torch.matmul(q, k_sum.unsqueeze(-1)))
    return torch.matmul(q, kv) * z

# Compare standard and linear attention
seq_len = 1000
d_model = 64
q = k = v = torch.rand(1, seq_len, d_model)

%timeit scaled_dot_product_attention(q, k, v)
%timeit linear_attention(q, k, v)

print("Standard attention shape:", scaled_dot_product_attention(q, k, v).shape)
print("Linear attention shape:", linear_attention(q, k, v).shape)
```

Slide 13: Attention in Vision Transformers

Vision Transformers (ViT) apply the attention mechanism to image patches, demonstrating the versatility of attention beyond text data.

```python
class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Example usage
embed_dim, num_heads = 768, 12
vit_block = VisionTransformerBlock(embed_dim, num_heads)
image_patches = torch.rand(1, 196, embed_dim)  # 14x14 patches from 224x224 image
output = vit_block(image_patches)
print(f"Vision Transformer block output shape: {output.shape}")
```

Slide 14: Attention Mechanisms in Natural Language Processing

Attention mechanisms have revolutionized natural language processing tasks. Let's explore a simple sentiment analysis model using self-attention.

```python
class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, 2)  # Binary classification
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.attention(x, x, x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

# Example usage
vocab_size, embed_dim, num_heads = 10000, 256, 8
model = SentimentAnalysisModel(vocab_size, embed_dim, num_heads)
input_ids = torch.randint(0, vocab_size, (1, 50))  # Batch size 1, 50 tokens
output = model(input_ids)
print(f"Sentiment analysis output shape: {output.shape}")
```

Slide 15: Additional Resources

For those interested in diving deeper into self-attention and cross-attention in transformers, here are some valuable resources:

1. "Attention Is All You Need" (Vaswani et al., 2017) - The original transformer paper: ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018): ArXiv link: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020) - Vision Transformers: ArXiv link: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

These papers provide in-depth explanations of the concepts we've covered and their applications in various domains.

