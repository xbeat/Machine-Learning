## Transformer Encoder Processing Input Tokens
Slide 1: Input Embeddings in Transformers

The embedding layer transforms input tokens into dense vector representations of dimension d\_model (typically 512). This crucial first step maps discrete tokens into a continuous vector space where semantic relationships can be captured effectively.

```python
import numpy as np
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * np.sqrt(self.d_model)

# Example usage
vocab_size, d_model = 5000, 512
embed = TokenEmbedding(vocab_size, d_model)
x = torch.tensor([[1, 2, 3, 4, 5]])  # Batch of token indices
output = embed(x)  # Shape: [1, 5, 512]
```

Slide 2: Positional Encodings Implementation

Positional encodings enable the Transformer to understand token positions without recurrence or convolution. Using sinusoidal functions creates unique position-dependent patterns that maintain relative positional information through linear combinations.

```python
def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return torch.FloatTensor(pos_encoding)

# Example
seq_len, d_model = 100, 512
pos_encoding = get_positional_encoding(seq_len, d_model)
```

Slide 3: Multi-Head Self-Attention Implementation

The multi-head attention mechanism allows the model to jointly attend to information from different representation subspaces, enabling the model to capture various types of dependencies between tokens simultaneously.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        x = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x)
```

Slide 4: Feed-Forward Neural Network Layer

The position-wise feed-forward network applies two linear transformations with a ReLU activation in between, allowing the model to process the attention output and introduce non-linearity into the representation.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)
        return x
```

Slide 5: Layer Normalization Implementation

Layer normalization stabilizes the learning process by normalizing the inputs across the features. Unlike batch normalization, it's independent of batch size and particularly effective for sequence modeling tasks.

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
```

Slide 6: Complete Encoder Layer

The encoder layer combines multi-head attention, feed-forward networks, and normalization layers with residual connections. This architecture enables effective processing of input sequences through parallel attention mechanisms.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and normalization
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and normalization
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

Slide 7: Complete Transformer Encoder

The complete encoder stacks multiple encoder layers, processing input embeddings through repeated rounds of self-attention and feed-forward transformations to create rich contextual representations.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(get_positional_encoding(max_seq_len, d_model))
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoding[:seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        return x
```

Slide 8: Attention Score Calculation

The attention mechanism computes compatibility scores between queries and keys, scaling them to maintain stable gradients during training. These scores determine how much focus each position places on other positions.

```python
def attention_scores(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get probabilities
    attention_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    return torch.matmul(attention_weights, value), attention_weights
```

Slide 9: Practical Example - Text Classification

Implementation of a complete text classification system using the Transformer encoder, demonstrating real-world application for sentiment analysis on movie reviews.

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, num_classes):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len
        )
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        # Encode the input sequence
        encoded = self.encoder(x, mask)
        # Use [CLS] token representation for classification
        return self.classifier(encoded[:, 0, :])

# Example usage
model = TextClassifier(
    vocab_size=30000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    max_seq_len=512,
    num_classes=2
)
```

Slide 10: Training Pipeline Implementation

A comprehensive training pipeline for the Transformer-based classifier, including data preprocessing, batching, and optimization setup with learning rate scheduling.

```python
def train_transformer(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.WarmupCosineSchedule(
        optimizer, warmup_steps=4000, t_total=epochs * len(train_loader)
    )
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
```

Slide 11: Attention Visualization Implementation

This implementation provides tools to visualize attention patterns in the Transformer encoder, helping understand how the model attends to different parts of the input sequence during processing.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, layer_num):
    """
    Visualizes attention weights for a specific layer
    attention_weights: tensor of shape [num_heads, seq_len, seq_len]
    tokens: list of input tokens
    """
    num_heads = attention_weights.shape[0]
    fig, axs = plt.subplots(2, num_heads//2, figsize=(15, 8))
    
    for idx, ax in enumerate(axs.flat):
        sns.heatmap(
            attention_weights[idx].cpu().detach(),
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cmap='viridis'
        )
        ax.set_title(f'Head {idx+1}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.suptitle(f'Layer {layer_num} Attention Patterns')
    return fig

# Example usage
def get_attention_maps(model, input_ids):
    attention_maps = []
    def hook_fn(module, input, output):
        attention_maps.append(output[1])  # Store attention weights
    
    hooks = []
    for layer in model.encoder.layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
    
    for hook in hooks:
        hook.remove()
    
    return attention_maps
```

Slide 12: Real-world Application - Document Classification

Complete implementation of a document classification system using the Transformer encoder, including data preprocessing and model evaluation metrics.

```python
class DocumentClassifier:
    def __init__(self, model_config):
        self.tokenizer = self.get_tokenizer()
        self.model = TextClassifier(**model_config)
        
    def preprocess_document(self, text):
        # Tokenize and prepare input
        tokens = self.tokenizer.encode(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens
    
    def classify_document(self, text):
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocess_document(text)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=-1)
            return probs.numpy()
    
    def evaluate_model(self, test_loader):
        metrics = {
            'accuracy': 0,
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['precision'] = precision_score(all_labels, all_preds, average=None)
        metrics['recall'] = recall_score(all_labels, all_preds, average=None)
        metrics['f1'] = f1_score(all_labels, all_preds, average=None)
        
        return metrics
```

Slide 13: Advanced Model Configuration and Mathematical Foundations

The mathematical foundations and configurations that drive the Transformer encoder's performance, including key formulas and hyperparameter relationships.

```python
class ModelConfig:
    def __init__(self):
        # Model architecture
        self.d_model = 512  # Embedding dimension
        self.n_heads = 8    # Number of attention heads
        self.d_k = self.d_model // self.n_heads  # Dimension per head
        
        # Key formulas (as comments for reference)
        """
        Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
        
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        
        LayerNorm(x) = γ * (x - μ)/sqrt(σ^2 + ε) + β
        
        PositionalEncoding(pos, 2i) = sin(pos/10000^(2i/d_model))
        PositionalEncoding(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        """
        
        # Learning rate schedule
        def get_lr(step, warmup_steps=4000):
            return self.d_model ** (-0.5) * min(
                step ** (-0.5),
                step * warmup_steps ** (-1.5)
            )
        
        self.learning_rate_schedule = get_lr
```

Slide 14: Additional Resources

arXiv Papers:

*   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) - "Attention Is All You Need"
*   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) - "BERT: Pre-training of Deep Bidirectional Transformers"
*   [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) - "Understanding Attention in Transformers"
*   [https://arxiv.org/abs/1907.00235](https://arxiv.org/abs/1907.00235) - "On Layer Normalization in the Transformer Architecture"
*   [https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745) - "On the Relationship between Self-Attention and Convolutional Layers"

