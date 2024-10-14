## Explaining Self-Attention vs. Cross-Attention in Python

Slide 1: Introduction to Self-Attention and Cross-Attention

Self-attention and cross-attention are fundamental mechanisms in modern deep learning architectures, particularly in transformer models. These techniques have revolutionized natural language processing and have found applications in various domains. In this presentation, we'll explore the differences between self-attention and cross-attention, their implementations, and their practical applications.

```python
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, is_self_attention=True):
        super().__init__()
        self.is_self_attention = is_self_attention
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        q = self.query(x)
        if self.is_self_attention:
            k, v = self.key(x), self.value(x)
        else:
            k, v = self.key(context), self.value(context)
        return torch.matmul(q, k.transpose(-2, -1)) * v
```

Slide 2: Self-Attention Mechanism

Self-attention allows a model to weigh the importance of different parts of the input sequence when processing each element. It enables the model to capture long-range dependencies and contextual information within the same sequence.

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
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
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)
```

Slide 3: Self-Attention in Action

Let's visualize how self-attention works on a simple sequence of words:

```python
import numpy as np
import matplotlib.pyplot as plt

def self_attention(query, key, value):
    scores = np.dot(query, key.T) / np.sqrt(query.shape[-1])
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.dot(attention_weights, value)

# Example sequence
sequence = ["The", "cat", "sat", "on", "the", "mat"]
embed_dim = 4

# Generate random embeddings for simplicity
embeddings = np.random.randn(len(sequence), embed_dim)

attention_output = self_attention(embeddings, embeddings, embeddings)

plt.figure(figsize=(10, 8))
plt.imshow(attention_output, cmap='viridis')
plt.title("Self-Attention Output")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence Position")
plt.colorbar()
plt.xticks(range(embed_dim))
plt.yticks(range(len(sequence)), sequence)
plt.show()
```

Slide 4: Cross-Attention Mechanism

Cross-attention, unlike self-attention, operates on two different sequences. It allows the model to focus on relevant parts of one sequence (typically called the "context") while processing another sequence (the "query").

```python
class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
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
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return self.fc_out(out)
```

Slide 5: Cross-Attention in Action

Let's visualize how cross-attention works between two sequences:

```python
import numpy as np
import matplotlib.pyplot as plt

def cross_attention(query, key, value):
    scores = np.dot(query, key.T) / np.sqrt(query.shape[-1])
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.dot(attention_weights, value)

# Example sequences
context = ["The", "quick", "brown", "fox"]
query = ["A", "lazy", "dog"]
embed_dim = 4

# Generate random embeddings for simplicity
context_embeddings = np.random.randn(len(context), embed_dim)
query_embeddings = np.random.randn(len(query), embed_dim)

attention_output = cross_attention(query_embeddings, context_embeddings, context_embeddings)

plt.figure(figsize=(10, 8))
plt.imshow(attention_output, cmap='viridis')
plt.title("Cross-Attention Output")
plt.xlabel("Embedding Dimension")
plt.ylabel("Query Sequence Position")
plt.colorbar()
plt.xticks(range(embed_dim))
plt.yticks(range(len(query)), query)
plt.show()
```

Slide 6: Key Differences

The main difference between self-attention and cross-attention lies in the input sequences they operate on. Self-attention processes a single sequence, allowing each element to attend to every other element within the same sequence. Cross-attention, on the other hand, involves two distinct sequences, enabling elements from one sequence to attend to elements in the other sequence.

```python
import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1)

    def forward(self, x):
        return self.attention(x, x, x)[0]

class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1)

    def forward(self, query, context):
        return self.attention(query, context, context)[0]

# Example usage
embed_dim = 64
seq_len = 10
batch_size = 2

x = torch.randn(seq_len, batch_size, embed_dim)
context = torch.randn(seq_len, batch_size, embed_dim)

self_attn = SelfAttentionLayer(embed_dim)
cross_attn = CrossAttentionLayer(embed_dim)

self_output = self_attn(x)
cross_output = cross_attn(x, context)

print("Self-attention output shape:", self_output.shape)
print("Cross-attention output shape:", cross_output.shape)
```

Slide 7: Applications of Self-Attention

Self-attention has found numerous applications in natural language processing tasks. One prominent example is sentiment analysis, where the model needs to understand the context and relationships between words in a sentence to determine the overall sentiment.

```python
import torch
import torch.nn as nn

class SentimentAnalyzer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, embed_dim)
        attn_output, _ = self.self_attention(x, x, x)
        pooled = torch.mean(attn_output, dim=0)  # Average pooling
        return self.fc(pooled)

# Example usage
vocab_size = 10000
embed_dim = 256
num_heads = 8
num_classes = 3  # Negative, Neutral, Positive
model = SentimentAnalyzer(vocab_size, embed_dim, num_heads, num_classes)

# Dummy input (batch_size=2, seq_len=10)
input_ids = torch.randint(0, vocab_size, (2, 10))
output = model(input_ids)
print("Sentiment prediction shape:", output.shape)
```

Slide 8: Applications of Cross-Attention

Cross-attention is particularly useful in sequence-to-sequence tasks, such as machine translation, where the model needs to align words or phrases between the source and target languages.

```python
import torch
import torch.nn as nn

class Translator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        src_embed = self.src_embedding(src).permute(1, 0, 2)
        tgt_embed = self.tgt_embedding(tgt).permute(1, 0, 2)
        attn_output, _ = self.cross_attention(tgt_embed, src_embed, src_embed)
        return self.fc(attn_output.permute(1, 0, 2))

# Example usage
src_vocab_size = 10000
tgt_vocab_size = 8000
embed_dim = 256
num_heads = 8
model = Translator(src_vocab_size, tgt_vocab_size, embed_dim, num_heads)

# Dummy input (batch_size=2, seq_len=10)
src_ids = torch.randint(0, src_vocab_size, (2, 10))
tgt_ids = torch.randint(0, tgt_vocab_size, (2, 8))
output = model(src_ids, tgt_ids)
print("Translation output shape:", output.shape)
```

Slide 9: Self-Attention in Image Processing

Self-attention can also be applied to image processing tasks, such as image classification or segmentation. In this context, each pixel or patch in the image can attend to other relevant parts of the image.

```python
import torch
import torch.nn as nn

class ImageSelfAttention(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x):
        b, c, h, w = x.size()
        x = self.conv(x)
        x = x.flatten(2).permute(2, 0, 1)  # (h*w, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.permute(1, 2, 0).view(b, -1, h, w)

# Example usage
in_channels = 3
embed_dim = 64
num_heads = 8
model = ImageSelfAttention(in_channels, embed_dim, num_heads)

# Dummy input (batch_size=1, channels=3, height=32, width=32)
input_image = torch.randn(1, 3, 32, 32)
output = model(input_image)
print("Image self-attention output shape:", output.shape)
```

Slide 10: Cross-Attention in Multi-Modal Learning

Cross-attention is particularly useful in multi-modal learning tasks, where information from different modalities (e.g., text and images) needs to be combined. A common application is image captioning, where the model generates text descriptions based on input images.

```python
import torch
import torch.nn as nn

class ImageCaptioner(nn.Module):
    def __init__(self, image_embed_dim, text_embed_dim, vocab_size, num_heads):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.image_proj = nn.Linear(64, image_embed_dim)
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.cross_attention = nn.MultiheadAttention(text_embed_dim, num_heads)
        self.fc = nn.Linear(text_embed_dim, vocab_size)

    def forward(self, image, text):
        image_features = self.image_encoder(image).squeeze(-1).squeeze(-1)
        image_embed = self.image_proj(image_features).unsqueeze(0)
        text_embed = self.text_embedding(text).permute(1, 0, 2)
        attn_output, _ = self.cross_attention(text_embed, image_embed, image_embed)
        return self.fc(attn_output.permute(1, 0, 2))

# Example usage
image_embed_dim = 256
text_embed_dim = 256
vocab_size = 10000
num_heads = 8
model = ImageCaptioner(image_embed_dim, text_embed_dim, vocab_size, num_heads)

# Dummy input (batch_size=2, channels=3, height=224, width=224)
image = torch.randn(2, 3, 224, 224)
text = torch.randint(0, vocab_size, (2, 10))  # (batch_size, seq_len)

output = model(image, text)
print("Image captioning output shape:", output.shape)
```

Slide 11: Self-Attention vs Cross-Attention: Performance Comparison

Let's compare the computational performance of self-attention and cross-attention operations:

```python
import torch
import torch.nn as nn
import time

def time_attention(attention_fn, *args):
    start_time = time.time()
    for _ in range(100):
        _ = attention_fn(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / 100

# Set up attention modules
embed_dim = 256
num_heads = 8
self_attn = nn.MultiheadAttention(embed_dim, num_heads).cuda()
cross_attn = nn.MultiheadAttention(embed_dim, num_heads).cuda()

# Generate input data
seq_len = 100
batch_size = 32
x = torch.randn(seq_len, batch_size, embed_dim).cuda()
y = torch.randn(seq_len, batch_size, embed_dim).cuda()

# Measure execution time
self_attn_time = time_attention(self_attn, x, x, x)
cross_attn_time = time_attention(cross_attn, x, y, y)

print(f"Self-attention average time: {self_attn_time:.6f} seconds")
print(f"Cross-attention average time: {cross_attn_time:.6f} seconds")
```

Slide 12: Real-Life Example: Text Summarization

Text summarization is a practical application that can benefit from both self-attention and cross-attention mechanisms. Here's a simplified example of how these attention mechanisms can be used in a summarization model:

```python
import torch
import torch.nn as nn

class SummarizationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, source, summary):
        src_embed = self.embedding(source).permute(1, 0, 2)
        sum_embed = self.embedding(summary).permute(1, 0, 2)

        # Self-attention on source text
        src_self_attn, _ = self.self_attention(src_embed, src_embed, src_embed)

        # Cross-attention between summary and source
        cross_attn, _ = self.cross_attention(sum_embed, src_self_attn, src_self_attn)

        return self.fc(cross_attn.permute(1, 0, 2))

# Example usage
vocab_size = 10000
embed_dim = 256
num_heads = 8
model = SummarizationModel(vocab_size, embed_dim, num_heads)

# Dummy input (batch_size=2, seq_len=100 for source, seq_len=20 for summary)
source = torch.randint(0, vocab_size, (2, 100))
summary = torch.randint(0, vocab_size, (2, 20))

output = model(source, summary)
print("Summarization output shape:", output.shape)
```

Slide 13: Real-Life Example: Question Answering

Question answering systems can leverage both self-attention and cross-attention to process the context and question, and generate accurate answers. Here's a simplified example:

```python
import torch
import torch.nn as nn

class QuestionAnsweringModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.question_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, 2)  # Start and end position

    def forward(self, context, question):
        ctx_embed = self.embedding(context).permute(1, 0, 2)
        q_embed = self.embedding(question).permute(1, 0, 2)

        # Self-attention on context and question
        ctx_self_attn, _ = self.context_self_attn(ctx_embed, ctx_embed, ctx_embed)
        q_self_attn, _ = self.question_self_attn(q_embed, q_embed, q_embed)

        # Cross-attention between question and context
        cross_attn, _ = self.cross_attn(q_self_attn, ctx_self_attn, ctx_self_attn)

        # Predict start and end positions
        logits = self.fc(cross_attn.mean(dim=0))
        return logits

# Example usage
vocab_size = 10000
embed_dim = 256
num_heads = 8
model = QuestionAnsweringModel(vocab_size, embed_dim, num_heads)

# Dummy input (batch_size=2, context_len=200, question_len=20)
context = torch.randint(0, vocab_size, (2, 200))
question = torch.randint(0, vocab_size, (2, 20))

output = model(context, question)
print("Question Answering output shape:", output.shape)
```

Slide 14: Conclusion and Future Directions

Self-attention and cross-attention mechanisms have revolutionized various fields in machine learning, particularly natural language processing and computer vision. Self-attention allows models to capture long-range dependencies within a single sequence, while cross-attention enables the integration of information from different sources or modalities.

As we've seen through various examples, these attention mechanisms can be applied to a wide range of tasks, from sentiment analysis and machine translation to image captioning and question answering. The flexibility and power of these techniques have led to significant improvements in model performance across many domains.

Slide 16: Conclusion and Future Directions

Future research directions in attention mechanisms include:

1. Developing more efficient attention algorithms to reduce computational complexity
2. Exploring novel attention architectures for specific tasks or domains
3. Investigating the interpretability of attention weights to gain insights into model decision-making
4. Combining attention mechanisms with other deep learning techniques for enhanced performance

As the field continues to evolve, we can expect to see even more innovative applications and improvements in attention-based models, further advancing the state-of-the-art in artificial intelligence and machine learning.

Slide 17: Additional Resources

For those interested in diving deeper into self-attention and cross-attention mechanisms, here are some valuable resources:

1. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020) ArXiv: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
4. "End-to-End Object Detection with Transformers" by Carion et al. (2020) ArXiv: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)

These papers provide in-depth explanations of various attention mechanisms and their applications in different domains, serving as excellent starting points for further exploration of the topic.

