## Transformer Self-Attention Mechanism in Python
Slide 1: Introduction to Transformers and Self-Attention

The Transformer architecture, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing. At its core lies the self-attention mechanism, which allows the model to weigh the importance of different words in a sentence when processing each word. This slideshow will explore the self-attention mechanism and its implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(sentence, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(attention_weights)
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(sentence, rotation=45)
    ax.set_yticklabels(sentence)
    plt.show()

sentence = ["The", "cat", "sat", "on", "the", "mat"]
attention_weights = np.random.rand(len(sentence), len(sentence))
visualize_attention(sentence, attention_weights)
```

Slide 2: Self-Attention: The Basics

Self-attention allows each word in a sequence to attend to all other words, including itself. It computes a weighted sum of all words' representations, where the weights are determined by how relevant each word is to the current word being processed.

```python
import torch
import torch.nn.functional as F

def self_attention(query, key, value):
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # Scale scores
    d_k = query.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
seq_len, d_model = 4, 8
query = key = value = torch.rand(1, seq_len, d_model)
output, attention_weights = self_attention(query, key, value)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
```

Slide 3: Query, Key, and Value Vectors

In self-attention, we project each input word into three vectors: query, key, and value. These projections allow the model to capture different aspects of the input for attention computation.

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project input to Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        return Q, K, V

# Example usage
d_model, num_heads, seq_len = 512, 8, 10
attention = SelfAttention(d_model, num_heads)
x = torch.rand(1, seq_len, d_model)
Q, K, V = attention(x)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
```

Slide 4: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces. It involves running the attention mechanism multiple times in parallel and concatenating the results.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q, K, V = self.attention(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute weighted sum
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and apply final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.attention.d_model)
        output = self.linear(attention_output)
        
        return output

# Example usage
mha = MultiHeadAttention(d_model, num_heads)
output = mha(x)
print("Multi-head attention output shape:", output.shape)
```

Slide 5: Positional Encoding

Transformers don't have a built-in notion of word order. Positional encoding adds information about the position of each word in the sequence, allowing the model to use word order information.

```python
def positional_encoding(max_seq_len, d_model):
    position = torch.arange(max_seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    pos_encoding = torch.zeros(max_seq_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    return pos_encoding

# Example usage
max_seq_len, d_model = 100, 512
pos_encoding = positional_encoding(max_seq_len, d_model)

plt.figure(figsize=(10, 6))
plt.pcolormesh(pos_encoding.numpy(), cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title("Positional Encoding")
plt.show()
```

Slide 6: Masked Self-Attention

In tasks like language modeling, we need to prevent the model from attending to future tokens. Masked self-attention applies a mask to the attention scores, setting future positions to negative infinity before the softmax operation.

```python
def masked_self_attention(query, key, value, mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# Example usage
seq_len, d_model = 4, 8
query = key = value = torch.rand(1, seq_len, d_model)
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
output, attention_weights = masked_self_attention(query, key, value, mask)

print("Masked attention weights:")
print(attention_weights.squeeze().numpy())
```

Slide 7: Feed-Forward Networks

After the self-attention layer, Transformers use a position-wise feed-forward network. This network applies the same fully connected layer to each position separately and identically.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Example usage
d_model, d_ff = 512, 2048
ff = FeedForward(d_model, d_ff)
x = torch.rand(1, 10, d_model)
output = ff(x)
print("Feed-forward output shape:", output.shape)
```

Slide 8: Layer Normalization

Layer normalization is applied after each sub-layer in the Transformer. It helps stabilize the activations, making the network easier to train.

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

# Example usage
features = 512
ln = LayerNorm(features)
x = torch.rand(1, 10, features)
output = ln(x)
print("Layer norm output shape:", output.shape)
```

Slide 9: Encoder Layer

An encoder layer in a Transformer combines multi-head attention, feed-forward networks, and layer normalization.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Example usage
d_model, num_heads, d_ff = 512, 8, 2048
encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
x = torch.rand(1, 10, d_model)
output = encoder_layer(x)
print("Encoder layer output shape:", output.shape)
```

Slide 10: Decoder Layer

A decoder layer is similar to an encoder layer but includes an additional multi-head attention layer that attends to the encoder's output.

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
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        self_attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(x)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Example usage
d_model, num_heads, d_ff = 512, 8, 2048
decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
x = torch.rand(1, 10, d_model)
encoder_output = torch.rand(1, 15, d_model)
src_mask = torch.ones(1, 15, 15)
tgt_mask = torch.tril(torch.ones(1, 10, 10))
output = decoder_layer(x, encoder_output, src_mask, tgt_mask)
print("Decoder layer output shape:", output.shape)
```

Slide 11: Real-Life Example: Text Summarization

Text summarization is a common application of Transformers. Here's a simplified example using the T5 model for abstractive summarization.

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text, model, tokenizer, max_length=150):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example text
text = """
The Transformer architecture has revolutionized natural language processing tasks. 
It uses self-attention mechanisms to process input sequences, allowing it to capture 
long-range dependencies more effectively than traditional recurrent neural networks. 
Transformers have been successfully applied to various tasks, including translation, 
summarization, and question answering.
"""

summary = summarize_text(text, model, tokenizer)
print("Summary:", summary)
```

Slide 12: Real-Life Example: Named Entity Recognition

Named Entity Recognition (NER) is another task where Transformers excel. Here's a simple example using a pre-trained BERT model for NER.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def perform_ner(text, model, tokenizer):
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    results = ner_pipeline(text)
    return results

# Load pre-trained model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Example text
text = "Apple Inc. is headquartered in Cupertino, California."

entities = perform_ner(text, model, tokenizer)
print("Named Entities:")
for entity in entities:
    print(f"{entity['word']} - {entity['entity_group']}")
```

Slide 13: Challenges and Limitations

While Transformers have achieved remarkable success, they face challenges:

1. Computational complexity: The self-attention mechanism has quadratic complexity with respect to sequence length, limiting its use on very long sequences.
2. Memory requirements: Transformers need to store attention weights for all pairs of tokens, which can be memory-intensive for long sequences.
3. Lack of inherent positional understanding: Unlike RNNs, Transformers rely on additional positional encodings to understand token order.
4. Training data requirements: Large Transformer models often require vast amounts of training data to perform well.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_complexity():
    seq_lengths = np.arange(1, 1001)
    complexity = seq_lengths ** 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, complexity)
    plt.xlabel('Sequence Length')
    plt.ylabel('Computational Complexity')
    plt.title('Quadratic Complexity of Self-Attention')
    plt.show()

plot_complexity()
```

Slide 14: Future Directions

Researchers are actively working on addressing Transformer limitations and expanding their capabilities:

1. Efficient attention mechanisms: Techniques like sparse attention and linear attention aim to reduce computational complexity.
2. Long-range Transformers: Models like Longformer and Big Bird extend Transformer's ability to handle longer sequences.
3. Parameter-efficient fine-tuning: Methods like adapter tuning and prompt tuning allow for more efficient adaptation of pre-trained models.
4. Multimodal Transformers: Extending Transformers to handle multiple modalities simultaneously, such as text and images.

```python
def sparse_attention(query, key, value, sparsity_factor=0.1):
    scores = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Keep only top-k values
    k = int(scores.size(-1) * sparsity_factor)
    top_k_scores, _ = torch.topk(scores, k, dim=-1)
    threshold = top_k_scores[..., -1, None]
    mask = scores < threshold
    
    attention_weights = torch.where(mask, torch.full_like(scores, float('-inf')), scores)
    attention_weights = F.softmax(attention_weights, dim=-1)
    
    output = torch.matmul(attention_weights, value)
    return output

# Example usage
seq_len, d_model = 100, 64
query = key = value = torch.rand(1, seq_len, d_model)
output = sparse_attention(query, key, value)
print("Sparse attention output shape:", output.shape)
```

Slide 15: Additional Resources

For those interested in diving deeper into Transformer architectures and self-attention mechanisms, here are some valuable resources:

1. "Attention Is All You Need" - The original Transformer paper: ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": ArXiv link: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Efficient Transformers: A Survey": ArXiv link: [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)
4. "A Survey of Transformers": ArXiv link: [https://arxiv.org/abs/2106.04554](https://arxiv.org/abs/2106.04554)

These papers provide in-depth explanations of Transformer architectures, their applications, and recent advancements in the field.

