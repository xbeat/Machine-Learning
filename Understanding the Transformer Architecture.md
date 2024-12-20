## Understanding the Transformer Architecture
Slide 1: Introduction to Transformer Architecture

The Transformer architecture, introduced in the "Attention Is All You Need" paper, revolutionized natural language processing. It relies on self-attention mechanisms to process sequential data without recurrence or convolution. This architecture forms the basis for models like BERT and GPT.

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

# Example usage
src_vocab_size, tgt_vocab_size, d_model, nhead = 1000, 1000, 512, 8
num_encoder_layers, num_decoder_layers = 6, 6
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)
```

Slide 2: The Encoder

The encoder in a Transformer processes the input sequence. It consists of multiple identical layers, each containing two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network. The encoder's output is then passed to the decoder.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, src):
        return self.encoder(src)

# Example usage
d_model, nhead, num_layers = 512, 8, 6
encoder = TransformerEncoder(d_model, nhead, num_layers)
src = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
encoder_output = encoder(src)
print(f"Encoder output shape: {encoder_output.shape}")
```

Slide 3: The Decoder

The decoder generates the output sequence. It has a similar structure to the encoder but includes an additional sub-layer that performs multi-head attention over the encoder's output. The decoder uses masked self-attention to prevent looking at future tokens during training.

```python
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    
    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)

# Example usage
d_model, nhead, num_layers = 512, 8, 6
decoder = TransformerDecoder(d_model, nhead, num_layers)
tgt = torch.rand(20, 32, d_model)  # (seq_len, batch_size, d_model)
memory = torch.rand(10, 32, d_model)  # Encoder output
decoder_output = decoder(tgt, memory)
print(f"Decoder output shape: {decoder_output.shape}")
```

Slide 4: Position Embedding

Position embeddings provide information about the relative or absolute position of tokens in the sequence. They are added to the input embeddings to retain positional information, which is crucial since the Transformer doesn't inherently consider token order.

```python
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

# Example usage
d_model = 512
pos_encoder = PositionalEncoding(d_model)
x = torch.rand(100, 32, d_model)  # (seq_len, batch_size, d_model)
encoded = pos_encoder(x)
print(f"Positionally encoded shape: {encoded.shape}")
```

Slide 5: Encoder Block

An encoder block consists of a multi-head self-attention layer followed by a position-wise feed-forward network. Each sub-layer is wrapped with a residual connection and layer normalization.

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

# Example usage
d_model, nhead = 512, 8
encoder_block = EncoderBlock(d_model, nhead)
src = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = encoder_block(src)
print(f"Encoder block output shape: {output.shape}")
```

Slide 6: Self-Attention Layer

The self-attention layer allows the model to weigh the importance of different parts of the input sequence when processing each token. It computes attention scores between all pairs of tokens in the sequence.

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return attn_output

# Example usage
d_model, nhead = 512, 8
self_attn = SelfAttention(d_model, nhead)
x = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = self_attn(x)
print(f"Self-attention output shape: {output.shape}")
```

Slide 7: Layer Normalization

Layer normalization helps stabilize the learning process by normalizing the inputs across the features. It's applied after each sub-layer in the Transformer, following the residual connection.

```python
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

# Example usage
features = 512
layer_norm = LayerNorm(features)
x = torch.rand(10, 32, features)
normalized = layer_norm(x)
print(f"Normalized output shape: {normalized.shape}")
```

Slide 8: Position-wise Feed-Forward Network

The position-wise feed-forward network is applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between.

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))

# Example usage
d_model, d_ff = 512, 2048
ff_network = PositionWiseFeedForward(d_model, d_ff)
x = torch.rand(10, 32, d_model)  # (seq_len, batch_size, d_model)
output = ff_network(x)
print(f"Feed-forward network output shape: {output.shape}")
```

Slide 9: Decoder Block

A decoder block is similar to an encoder block but includes an additional cross-attention layer that attends to the encoder's output. It also uses masked self-attention to prevent looking at future tokens during training.

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# Example usage
d_model, nhead = 512, 8
decoder_block = DecoderBlock(d_model, nhead)
tgt = torch.rand(20, 32, d_model)  # (seq_len, batch_size, d_model)
memory = torch.rand(10, 32, d_model)  # Encoder output
output = decoder_block(tgt, memory)
print(f"Decoder block output shape: {output.shape}")
```

Slide 10: Cross-Attention Layer

The cross-attention layer in the decoder allows it to focus on relevant parts of the input sequence. It computes attention between the decoder's representations and the encoder's output.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)
        return attn_output

# Example usage
d_model, nhead = 512, 8
cross_attn = CrossAttention(d_model, nhead)
query = torch.rand(20, 32, d_model)  # (seq_len, batch_size, d_model)
key = value = torch.rand(10, 32, d_model)  # Encoder output
output = cross_attn(query, key, value)
print(f"Cross-attention output shape: {output.shape}")
```

Slide 11: Predicting Head

The predicting head is the final layer of the Transformer, typically a linear layer that projects the decoder's output to the target vocabulary size. It's used to generate the final output probabilities for each token in the sequence.

```python
class PredictingHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(PredictingHead, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

# Example usage
d_model, vocab_size = 512, 10000
predicting_head = PredictingHead(d_model, vocab_size)
x = torch.rand(20, 32, d_model)  # (seq_len, batch_size, d_model)
output = predicting_head(x)
print(f"Predicting head output shape: {output.shape}")
```

Slide 12: Real-Life Example: Machine Translation

Machine translation is a common application of Transformer models. Here's a simplified example of how to use a pre-trained Transformer for translation:

```python
from transformers import MarianMTModel, MarianTokenizer

# Load pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-de'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate a sentence
src_text = "Hello, how are you?"
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

print(f"Source: {src_text}")
print(f"Translation: {tgt_text[0]}")

# Output:
# Source: Hello, how are you?
# Translation: Hallo, wie geht es Ihnen?
```

Slide 13: Real-Life Example: Text Summarization

Text summarization is another task where Transformers excel. Here's a simple example using a pre-trained model:

```python
from transformers import pipeline

# Load pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Text to summarize
text = """
The Transformer model has revolutionized natural language processing tasks. 
It uses self-attention mechanisms to process input sequences, allowing it to capture 
long-range dependencies more effectively than previous architectures. 
Transformers have been successfully applied to various tasks, including 
translation, summarization, and question answering.
"""

# Generate summary
summary = summarizer(text, max_length=50, min_length=10, do_sample=False)

print("Original text:")
print(text)
print("\nSummary:")
print(summary[0]['summary_text'])

# Output:
# Original text:
# The Transformer model has revolutionized natural language processing tasks...
# 
# Summary:
# Transformer model uses self-attention mechanisms to process input sequences. 
# It captures long-range dependencies more effectively than previous architectures.
```

Slide 14: Transformer Scaling and Variants

Transformer models have been scaled to billions of parameters and adapted for various tasks. Some notable variants include:

1. BERT (Bidirectional Encoder Representations from Transformers)
2. GPT (Generative Pre-trained Transformer)
3. T5 (Text-to-Text Transfer Transformer)

These models have pushed the boundaries of NLP, achieving state-of-the-art results on numerous benchmarks.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a pre-trained BERT model for sequence classification
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input
text = "Transformers are powerful language models."
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
predicted_class = torch.argmax(outputs.logits).item()
print(f"Predicted class: {predicted_class}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Transformer architectures, the following resources are recommended:

1. "Attention Is All You Need" paper (Vaswani et al., 2017) ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018) ArXiv link: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3. "Language Models are Few-Shot Learners" (Brown et al., 2020) ArXiv link: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

These papers provide in-depth explanations of the Transformer architecture and its variants, offering valuable insights into their design and capabilities.
