## Explaining Self-Attention and Masked Self-Attention in Deep Learning
Slide 1: Introduction to Attention Mechanisms

Attention mechanisms have revolutionized deep learning, particularly in sequence-to-sequence tasks and natural language processing. They allow models to focus on relevant parts of input data, improving performance on tasks involving long-range dependencies.

```python
import torch
import torch.nn as nn

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(0)
        h = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attention(torch.cat((h, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return nn.functional.softmax(attention, dim=1)
```

Slide 2: Types of Attention Mechanisms

There are several types of attention mechanisms, including:

1. Self-Attention: Relates different positions within a single sequence.
2. Cross-Attention: Relates positions from one sequence to another.
3. Multi-Head Attention: Applies attention multiple times in parallel.
4. Scaled Dot-Product Attention: A more efficient variation used in Transformers.

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights
```

Slide 3: Self-Attention Basics

Self-attention, also known as intra-attention, relates different positions of a single sequence to compute a representation of the same sequence. It allows the model to attend to different parts of the input when processing each element.

```python
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

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        q_len, k_len, v_len = query.shape[1], key.shape[1], value.shape[1]

        # Split embedding into self.heads pieces
        q = self.queries(query).reshape(N, q_len, self.heads, self.head_dim)
        k = self.keys(key).reshape(N, k_len, self.heads, self.head_dim)
        v = self.values(value).reshape(N, v_len, self.heads, self.head_dim)

        # Perform scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            N, q_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out
```

Slide 4: Step-by-Step Explanation of Self-Attention

1. Input Transformation: Convert input vectors into query, key, and value vectors.
2. Score Calculation: Compute attention scores between all pairs of words.
3. Softmax: Apply softmax to normalize the scores.
4. Value Weighting: Multiply each value vector by its softmaxed score.
5. Sum: Sum up the weighted value vectors to produce the output.

```python
def self_attention_step_by_step(input_seq):
    # Assume input_seq is a tensor of shape (seq_len, embed_dim)
    seq_len, embed_dim = input_seq.shape
    
    # 1. Input Transformation
    W_q = torch.randn(embed_dim, embed_dim)
    W_k = torch.randn(embed_dim, embed_dim)
    W_v = torch.randn(embed_dim, embed_dim)
    
    Q = torch.matmul(input_seq, W_q)
    K = torch.matmul(input_seq, W_k)
    V = torch.matmul(input_seq, W_v)
    
    # 2. Score Calculation
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 3. Softmax
    attention_weights = nn.functional.softmax(scores / (embed_dim ** 0.5), dim=-1)
    
    # 4 & 5. Value Weighting and Sum
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Example usage
input_seq = torch.randn(10, 512)  # 10 words, 512-dimensional embeddings
output, weights = self_attention_step_by_step(input_seq)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 5: Why Self-Attention Works

Self-attention is effective because it:

1. Captures long-range dependencies directly.
2. Is computationally efficient, allowing for parallelization.
3. Provides interpretable attention weights.
4. Can handle variable-length sequences.
5. Learns to focus on relevant parts of the input.

```python
import matplotlib.pyplot as plt

def visualize_attention(sentence, attention_weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(attention_weights)
    
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    
    ax.set_xticklabels(sentence, rotation=45)
    ax.set_yticklabels(sentence)
    
    plt.show()

# Example usage
sentence = ["The", "cat", "sat", "on", "the", "mat"]
fake_attention_weights = torch.rand(len(sentence), len(sentence))
visualize_attention(sentence, fake_attention_weights)
```

Slide 6: Masked Self-Attention

Masked self-attention is a variant used in decoders to prevent the model from attending to future tokens during training. It ensures that predictions for a given position can depend only on known outputs at earlier positions.

```python
def masked_self_attention(input_seq, mask=None):
    seq_len, embed_dim = input_seq.shape
    
    Q = input_seq
    K = input_seq
    V = input_seq
    
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = nn.functional.softmax(scores / (embed_dim ** 0.5), dim=-1)
    
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Create a lower triangular mask
seq_len = 5
mask = torch.tril(torch.ones(seq_len, seq_len))

input_seq = torch.randn(seq_len, 512)
output, weights = masked_self_attention(input_seq, mask)

print(f"Mask shape: {mask.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 7: Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. It improves the performance of the attention layer by running multiple attention operations in parallel.

```python
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

Slide 8: Transformer Architecture

Transformers use self-attention as their core building block. The architecture consists of an encoder and a decoder, each composed of multiple layers of multi-head attention and feedforward neural networks.

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
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

        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
```

Slide 9: Positional Encoding

Transformers use positional encoding to inject information about the position of tokens in the sequence. This is necessary because self-attention operations are permutation-invariant.

```python
import numpy as np

def get_positional_encoding(max_seq_len, embed_dim):
    positional_encoding = np.zeros((max_seq_len, embed_dim))
    for pos in range(max_seq_len):
        for i in range(0, embed_dim, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/embed_dim)))
            positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/embed_dim)))
    
    return torch.FloatTensor(positional_encoding)

# Example usage
max_seq_len, embed_dim = 100, 512
pos_encoding = get_positional_encoding(max_seq_len, embed_dim)

plt.figure(figsize=(10, 10))
plt.imshow(pos_encoding, cmap='hot', aspect='auto')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence Position")
plt.show()
```

Slide 10: Real-Life Example: Machine Translation

Machine translation is a common application of Transformers. The model translates text from one language to another by encoding the source language and generating the target language using self-attention mechanisms.

```python
from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, src_lang="en", tgt_lang="fr"):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    return translated_text

# Example usage
english_text = "Hello, how are you doing today?"
french_translation = translate_text(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")
```

Slide 11: Real-Life Example: Text Summarization

Text summarization is another application where Transformers excel. The model reads the input text and generates a concise summary using self-attention to focus on the most important parts of the text.

```python
from transformers import pipeline

def summarize_text(text, max_length=150, min_length=50):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example usage
long_text = """
Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels. These activities release greenhouse gases into the atmosphere, trapping heat and causing the Earth's average temperature to rise. The consequences of climate change are far-reaching and include more frequent and severe weather events, rising sea levels, and disruptions to ecosystems and biodiversity. To address this global challenge, countries and organizations worldwide are working on strategies to reduce greenhouse gas emissions, transition to renewable energy sources, and adapt to the changing climate.
"""

summary = summarize_text(long_text)
print(f"Original text length: {len(long_text)}")
print(f"Summary length: {len(summary)}")
print(f"Summary: {summary}")
```

Slide 12: PyTorch Implementation of Self-Attention

Here's a concise PyTorch implementation of self-attention in about 30 lines of code:

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = embed_size ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        return out

# Example usage
embed_size = 512
seq_len = 10
batch_size = 2

x = torch.randn(batch_size, seq_len, embed_size)
self_attention = SelfAttention(embed_size)
output = self_attention(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Advantages and Limitations of Self-Attention

Advantages:

1. Captures long-range dependencies effectively
2. Parallelizable, allowing for efficient computation
3. Provides interpretable attention weights
4. Handles variable-length sequences well

Limitations:

1. Quadratic complexity with sequence length
2. May struggle with very long sequences
3. Requires careful tuning of hyperparameters

Slide 14: Advantages and Limitations of Self-Attention

```python
def self_attention_complexity(seq_length):
    time_complexity = seq_length ** 2
    space_complexity = seq_length ** 2
    return time_complexity, space_complexity

seq_lengths = [10, 100, 1000, 10000]
for length in seq_lengths:
    time, space = self_attention_complexity(length)
    print(f"Sequence length: {length}")
    print(f"Time complexity: O({time})")
    print(f"Space complexity: O({space})")
    print()
```

Slide 15: Future Directions and Improvements

Researchers are actively working on improving self-attention mechanisms:

1. Efficient attention: Reducing computational complexity
2. Sparse attention: Attending to only a subset of tokens
3. Adaptive attention spans: Dynamically adjusting attention range
4. Hierarchical attention: Capturing structure at multiple levels

Slide 16: Future Directions and Improvements

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_attention_improvements():
    seq_lengths = np.arange(1, 1001)
    full_attention = seq_lengths ** 2
    linear_attention = seq_lengths
    log_attention = np.log(seq_lengths) * seq_lengths

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, full_attention, label='Full Attention')
    plt.plot(seq_lengths, linear_attention, label='Linear Attention')
    plt.plot(seq_lengths, log_attention, label='Log-Linear Attention')
    plt.xlabel('Sequence Length')
    plt.ylabel('Computational Complexity')
    plt.title('Comparison of Attention Mechanisms')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

plot_attention_improvements()
```

Slide 17: Additional Resources

For those interested in diving deeper into self-attention and Transformers, here are some valuable resources:

1. "Attention Is All You Need" paper: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Transformers: State-of-the-Art Natural Language Processing": [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771)
4. "Efficient Transformers: A Survey": [https://arxiv.org/abs/2009.06732](https://arxiv.org/abs/2009.06732)

These papers provide in-depth explanations of the concepts discussed in this presentation and offer insights into the latest developments in the field.

