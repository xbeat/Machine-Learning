## Transformers and Attention Mechanisms in Large Language Models
Slide 1: Understanding Attention Mechanism Fundamentals

The attention mechanism enables neural networks to selectively focus on specific parts of the input sequence when generating output. This fundamental concept allows models to assign different importance weights to different elements, dramatically improving performance in sequence-to-sequence tasks.

```python
import numpy as np

def attention_score(query, key):
    # Calculate raw attention scores using dot product
    scores = np.dot(query, key.T)
    # Apply softmax to get probability distribution
    scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    return scores

# Example usage
query = np.random.randn(1, 64)  # Query vector
key = np.random.randn(10, 64)   # Key matrix
attention_weights = attention_score(query, key)
print(f"Attention weights shape: {attention_weights.shape}")
print(f"Sum of weights: {np.sum(attention_weights)}")  # Should be close to 1
```

Slide 2: Self-Attention Implementation

Self-attention allows each position in a sequence to attend to all positions in the same sequence. This mechanism is crucial for capturing long-range dependencies and understanding contextual relationships within the input data.

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        return output, attention
```

Slide 3: Mathematical Foundation of Attention

The attention mechanism computes a weighted sum of values, where weights are determined by compatibility between queries and keys. The mathematical formulation provides the theoretical foundation for implementing attention in neural networks.

```python
# Mathematical formulation of attention
"""
$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Where:
$$Q$$ = Query matrix
$$K$$ = Key matrix
$$V$$ = Value matrix
$$d_k$$ = Dimension of keys
$$softmax(x_i) = \frac{exp(x_i)}{\sum_j exp(x_j)}$$
"""

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

Slide 4: Multi-Head Attention Architecture

Multi-head attention allows the model to jointly attend to information from different representation subspaces. This enables the model to capture various aspects of the relationships between elements in the input sequence.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.split_heads(self.q_linear(q), batch_size)
        k = self.split_heads(self.k_linear(k), batch_size)
        v = self.split_heads(self.v_linear(v), batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        return self.out(concat_attention)
```

Slide 5: Positional Encoding in Transformers

Positional encoding is crucial for transformers to understand the sequential order of input elements since attention mechanisms are position-agnostic. The encoding uses sine and cosine functions of different frequencies to represent position information.

```python
import torch
import numpy as np

def positional_encoding(max_seq_length, d_model):
    position = torch.arange(max_seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros((max_seq_length, d_model))
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)
    
    return pos_encoding

# Example usage
max_length = 100
embedding_dim = 512
pos_enc = positional_encoding(max_length, embedding_dim)
print(f"Positional encoding shape: {pos_enc.shape}")
```

Slide 6: Implementing TransformerPreLN Architecture

The TransformerPreLN variant applies layer normalization before the self-attention and feed-forward layers, providing better training stability and faster convergence compared to the original transformer architecture.

```python
class TransformerPreLNLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-LN architecture
        norm_x = self.norm1(x)
        x = x + self.dropout(self.mha(norm_x, norm_x, norm_x))
        
        norm_x = self.norm2(x)
        x = x + self.dropout(self.ff(norm_x))
        return x
```

Slide 7: Cross-Attention Implementation

Cross-attention enables the model to attend to elements from different sequences, crucial for tasks like machine translation where the decoder must attend to the encoder's output while generating the translation.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, enc_output, mask=None):
        # x: decoder input
        # enc_output: encoder output
        norm_x = self.norm(x)
        attention_output = self.mha(
            q=norm_x,
            k=enc_output,
            v=enc_output,
            mask=mask
        )
        return x + self.dropout(attention_output)

# Example dimensions
batch_size, seq_len, d_model = 32, 50, 512
decoder_input = torch.randn(batch_size, seq_len, d_model)
encoder_output = torch.randn(batch_size, seq_len, d_model)
```

Slide 8: Vision Transformer Implementation

Vision Transformers (ViT) adapt the transformer architecture for image processing by splitting images into patches and treating them as sequence elements. This implementation shows the core components of the ViT architecture.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, h', w')
        x = x.flatten(2)        # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (batch_size, num_patches, embed_dim)
        return x

class ViTEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        
    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x
```

Slide 9: Translation Model Implementation

This implementation demonstrates a complete sequence-to-sequence translation model using transformers, incorporating both encoder and decoder components with practical attention mechanisms for language translation tasks.

```python
class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = positional_encoding(1000, d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerPreLNLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerPreLNLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def encode(self, src):
        x = self.encoder_embedding(src)
        x = x + self.pos_encoding[:x.size(1)].to(x.device)
        
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def decode(self, tgt, enc_output):
        x = self.decoder_embedding(tgt)
        x = x + self.pos_encoding[:x.size(1)].to(x.device)
        
        for layer, cross_attn in zip(self.decoder_layers, 
                                   self.cross_attention_layers):
            x = layer(x)
            x = cross_attn(x, enc_output)
            
        return self.output_layer(x)
```

Slide 10: Attention Visualization Implementation

Understanding attention patterns is crucial for model interpretation. This implementation provides tools to visualize attention weights and patterns in transformer models.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_weights(attention_weights, src_tokens, tgt_tokens=None):
    plt.figure(figsize=(10, 8))
    if tgt_tokens is None:
        # Self-attention visualization
        sns.heatmap(attention_weights, 
                    xticklabels=src_tokens,
                    yticklabels=src_tokens,
                    cmap='viridis')
        plt.title('Self-Attention Weights')
    else:
        # Cross-attention visualization
        sns.heatmap(attention_weights,
                    xticklabels=src_tokens,
                    yticklabels=tgt_tokens,
                    cmap='viridis')
        plt.title('Cross-Attention Weights')
    
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    return plt.gcf()

# Example usage
src_tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
tgt_tokens = ['Le', 'chat', 'est', 'assis', 'sur', 'le', 'tapis']
attention_matrix = torch.rand(len(tgt_tokens), len(src_tokens))
fig = plot_attention_weights(attention_matrix.numpy(), src_tokens, tgt_tokens)
```

Slide 11: Training Pipeline Implementation

A comprehensive training pipeline for transformer models, including loss calculation, optimization, and training loop with proper gradient handling and learning rate scheduling.

```python
class TransformerTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def train_step(self, src, tgt):
        self.model.train()
        self.optimizer.zero_grad()
        
        enc_output = self.model.encode(src)
        output = self.model.decode(tgt[:, :-1], enc_output)
        
        loss = self.criterion(
            output.contiguous().view(-1, output.size(-1)),
            tgt[:, 1:].contiguous().view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            loss = self.train_step(src, tgt)
            total_loss += loss
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
                
        return total_loss / len(dataloader)
```

Slide 12: Real-world Application: Neural Machine Translation

This implementation demonstrates a complete neural machine translation system using transformers, including data preprocessing, training, and inference pipelines for practical translation tasks.

```python
class NMTSystem:
    def __init__(self, src_vocab, tgt_vocab, model_dim=512):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.model = TranslationTransformer(
            len(src_vocab), len(tgt_vocab),
            d_model=model_dim,
            num_heads=8,
            num_layers=6
        )
        
    def preprocess(self, text, vocab):
        # Tokenize and convert to indices
        tokens = text.lower().split()
        return torch.tensor([vocab.get(token, vocab['<unk>']) 
                           for token in tokens])
    
    def translate(self, src_text, max_length=50):
        self.model.eval()
        with torch.no_grad():
            # Preprocess source text
            src = self.preprocess(src_text, self.src_vocab)
            src = src.unsqueeze(0)
            
            # Generate translation
            enc_output = self.model.encode(src)
            tgt = torch.tensor([[self.tgt_vocab['<start>']]])
            
            for _ in range(max_length):
                out = self.model.decode(tgt, enc_output)
                pred = out[:, -1:].argmax(-1)
                tgt = torch.cat([tgt, pred], dim=1)
                
                if pred.item() == self.tgt_vocab['<end>']:
                    break
            
            # Convert indices back to text
            result = []
            for idx in tgt[0][1:]:
                token = list(self.tgt_vocab.keys())[list(self.tgt_vocab.values()).index(idx.item())]
                if token == '<end>':
                    break
                result.append(token)
                
            return ' '.join(result)

# Example usage
src_text = "The weather is beautiful today."
translator = NMTSystem(src_vocab, tgt_vocab)
translation = translator.translate(src_text)
print(f"Source: {src_text}")
print(f"Translation: {translation}")
```

Slide 13: Real-world Application: Document Classification

Implementation of a transformer-based document classification system, showing how attention mechanisms can be applied to long-form text analysis and categorization.

```python
class DocumentClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=512, max_length=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_length, d_model)
        
        self.transformer_layers = nn.ModuleList([
            TransformerPreLNLayer(d_model, num_heads=8, d_ff=d_model * 4)
            for _ in range(4)
        ])
        
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_length)
        x = self.embedding(x)
        x = x + self.pos_encoding[:x.size(1)].to(x.device)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global pooling
        pooled = self.pool(x.mean(dim=1))
        return self.classifier(pooled)

# Training loop example
def train_classifier(model, train_loader, optimizer, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" - [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
*   "Layer Normalization" - [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
*   "Neural Machine Translation by Jointly Learning to Align and Translate" - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

