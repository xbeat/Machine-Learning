## Understanding the Transformer Architecture of Large Language Models
Slide 1: Transformer Architecture Implementation

The transformer architecture forms the backbone of modern LLMs, utilizing self-attention mechanisms to process sequential data. This implementation demonstrates the core components including multi-head attention, positional encoding, and feed-forward networks in a clean, modular approach.

```python
import numpy as np
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_length, self.d_model)
        return self.output_proj(out)
```

Slide 2: Positional Encoding

Positional encoding enables transformers to understand sequence order by injecting position information into input embeddings. This implementation shows both sinusoidal and learnable position encodings, with the former being parameter-free and the latter learned during training.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # Calculate sinusoidal position encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]
```

Slide 3: Tokenization and Vocabulary

Building an efficient tokenizer is crucial for LLM performance. This implementation demonstrates a subword tokenization approach using Byte-Pair Encoding (BPE), which balances vocabulary size and token meaningfulness.

```python
from collections import defaultdict
import re

class SubwordTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        
    def train(self, texts):
        # Initialize character vocabulary
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[' '.join(list(word)) + ' </w>'] += 1
                
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - 256  # Reserve space for bytes
        for i in range(num_merges):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = i
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            
    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
```

Slide 4: Feed-Forward Network Implementation

The feed-forward network in transformers processes token representations independently at each position. This implementation shows the standard architecture with two linear transformations and a ReLU activation, including dropout for regularization.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # First linear transformation with expansion
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear transformation with projection
        return self.linear2(x)

# Example usage
ff_network = FeedForward(d_model=512, d_ff=2048)
sample_input = torch.randn(32, 100, 512)  # (batch_size, seq_len, d_model)
output = ff_network(sample_input)
```

Slide 5: Layer Normalization in Transformers

Layer normalization stabilizes training by normalizing activations across feature dimensions. This implementation shows the transformer-specific layer norm with learned affine parameters and numerical stability considerations.

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # Calculate mean and variance along last dimension
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        
        # Normalize and scale
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# Example usage
layer_norm = LayerNorm(d_model=512)
sample_input = torch.randn(32, 100, 512)
normalized_output = layer_norm(sample_input)
```

Slide 6: Transformer Encoder Block

The encoder block combines multi-head attention, feed-forward networks, and normalization layers. This implementation demonstrates the complete architecture with residual connections and proper ordering of operations.

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff_network = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection
        attention_output = self.self_attention(x, mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.ff_network(x)
        x = x + self.dropout(ff_output)
        return self.norm2(x)
```

Slide 7: Attention Visualization

Understanding attention patterns is crucial for model interpretability. This implementation provides tools to visualize attention weights and analyze how the model attends to different input tokens.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, save_path=None):
    """
    Visualizes attention weights between tokens
    
    Args:
        attention_weights: tensor of shape (num_heads, seq_len, seq_len)
        tokens: list of input tokens
        save_path: optional path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    
    # Average attention weights across heads
    avg_weights = attention_weights.mean(dim=0).detach().numpy()
    
    # Create heatmap
    sns.heatmap(avg_weights, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='viridis',
                annot=True,
                fmt='.2f')
    
    plt.title('Attention Weights Visualization')
    plt.xlabel('Target Tokens')
    plt.ylabel('Source Tokens')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
tokens = ['The', 'cat', 'sat', 'on', 'mat']
fake_attention = torch.rand(8, 5, 5)  # (num_heads, seq_len, seq_len)
visualize_attention(fake_attention, tokens)
```

Slide 8: Embedding Layer Implementation

The embedding layer converts input tokens into continuous vector representations. This implementation includes learned token embeddings with optional weight tying and proper scaling to maintain appropriate magnitude of activations.

```python
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        # Initialize embeddings with Xavier uniform distribution
        nn.init.xavier_uniform_(self.token_embedding.weight)
        
    def forward(self, x):
        # Scale embeddings by sqrt(d_model)
        token_embeddings = self.token_embedding(x) * np.sqrt(self.d_model)
        # Add positional encoding
        embeddings = self.position_encoding(token_embeddings)
        return self.dropout(embeddings)

# Example usage
embedding_layer = TransformerEmbedding(vocab_size=30000, d_model=512, max_seq_length=512)
sample_input = torch.randint(0, 30000, (32, 100))  # (batch_size, seq_len)
embedded_output = embedding_layer(sample_input)
```

Slide 9: Training Loop Implementation

The training loop orchestrates the model training process with gradient accumulation, learning rate scheduling, and proper handling of padding tokens in loss calculation.

```python
def train_transformer(model, train_dataloader, optimizer, scheduler, num_epochs, pad_idx):
    model.train()
    total_loss = 0
    accumulation_steps = 4  # Gradient accumulation steps
    
    for epoch in range(num_epochs):
        for batch_idx, (src, tgt) in enumerate(train_dataloader):
            # Create padding mask
            pad_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
            
            # Forward pass
            outputs = model(src, mask=pad_mask)
            loss = calculate_loss(outputs, tgt, pad_idx)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
def calculate_loss(outputs, targets, pad_idx):
    # Mask padded positions in loss calculation
    mask = (targets != pad_idx)
    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                          targets.view(-1), 
                          ignore_index=pad_idx,
                          reduction='sum')
    return loss / mask.sum()
```

Slide 10: Inference and Generation

This implementation shows the generation process using beam search and sampling strategies, handling both autoregressive generation and parallel decoding where applicable.

```python
class BeamSearchGenerator:
    def __init__(self, model, tokenizer, beam_size=4, max_length=100):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length
        
    @torch.no_grad()
    def generate(self, input_ids, temperature=1.0):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize beam search
        scores = torch.zeros(batch_size, self.beam_size, device=device)
        sequences = input_ids.repeat(1, self.beam_size, 1)
        
        for step in range(self.max_length):
            # Get model predictions
            outputs = self.model(sequences)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Calculate log probabilities
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Get top-k next tokens and their scores
            vocab_size = next_token_scores.size(-1)
            top_k_scores, top_k_tokens = next_token_scores.topk(
                self.beam_size, dim=-1)
            
            # Update sequences and scores
            sequences = torch.cat([sequences, top_k_tokens.unsqueeze(-1)], dim=-1)
            scores += top_k_scores
            
            # Check for completion
            if self._is_generation_done(sequences):
                break
                
        return self._get_best_sequences(sequences, scores)
    
    def _is_generation_done(self, sequences):
        # Check if all sequences have end token or reached max length
        return (sequences[:, :, -1] == self.tokenizer.eos_token_id).all()
    
    def _get_best_sequences(self, sequences, scores):
        # Return sequences with highest scores
        best_scores, best_idx = scores.max(dim=-1)
        return sequences[torch.arange(sequences.size(0)), best_idx]
```

Slide 11: Attention Mechanism Implementation

A detailed implementation of scaled dot-product attention with masking support, demonstrating the core mathematical operations that enable contextual understanding in transformer models.

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        # Calculate attention scores
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Calculate output
        # Shape: (batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attn, v)
        
        return output, attn

# Example usage with shapes
batch_size, num_heads = 32, 8
seq_len, d_k = 100, 64

q = torch.randn(batch_size, num_heads, seq_len, d_k)
k = torch.randn(batch_size, num_heads, seq_len, d_k)
v = torch.randn(batch_size, num_heads, seq_len, d_k)

attention = ScaledDotProductAttention(temperature=np.sqrt(d_k))
output, attention_weights = attention(q, k, v)
```

Slide 12: Preprocessing Pipeline

Implementation of a robust preprocessing pipeline for transformer models, including text cleaning, tokenization, and dynamic batching with proper padding and attention mask generation.

```python
class TransformerPreprocessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def preprocess_batch(self, texts):
        # Clean and normalize texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Tokenize and pad sequences
        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create attention masks
        attention_mask = encoded['attention_mask']
        
        # Create position IDs
        position_ids = torch.arange(0, encoded['input_ids'].size(1))
        position_ids = position_ids.unsqueeze(0).expand_as(encoded['input_ids'])
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    
    def _clean_text(self, text):
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lowercase
        return text.lower()

# Example usage
preprocessor = TransformerPreprocessor(tokenizer)
texts = [
    "Hello, how are you?",
    "Natural language processing is fascinating!"
]
batch = preprocessor.preprocess_batch(texts)
```

Slide 13: Loss Functions and Metrics

Implementation of specialized loss functions and evaluation metrics for transformer models, including label smoothing and perplexity calculation.

```python
class TransformerLoss:
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        
    def label_smoothed_nll_loss(self, pred, target):
        """
        Implements label smoothed cross entropy loss
        """
        # Create smoothed targets
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 2))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate loss with padding mask
        padding_mask = target.ne(self.pad_idx)
        losses = -torch.sum(smooth_target * F.log_softmax(pred, dim=-1), dim=-1)
        loss = losses.masked_select(padding_mask).mean()
        
        return loss
    
    def calculate_perplexity(self, loss):
        """
        Calculate perplexity from cross entropy loss
        """
        return torch.exp(loss)
    
    def sequence_accuracy(self, pred, target):
        """
        Calculate sequence-level accuracy
        """
        pred_tokens = pred.argmax(dim=-1)
        correct = (pred_tokens == target).float()
        # Mask out padding tokens
        mask = target.ne(self.pad_idx)
        accuracy = (correct * mask).sum() / mask.sum()
        return accuracy

# Example usage
criterion = TransformerLoss(vocab_size=30000, pad_idx=0)
pred = torch.randn(32, 100, 30000)  # (batch_size, seq_len, vocab_size)
target = torch.randint(0, 30000, (32, 100))  # (batch_size, seq_len)
loss = criterion.label_smoothed_nll_loss(pred, target)
perplexity = criterion.calculate_perplexity(loss)
accuracy = criterion.sequence_accuracy(pred, target)
```

Slide 14: Real-World Example - Text Classification

Implementation of a complete text classification system using transformers, demonstrating data preprocessing, model training, and evaluation on a sentiment analysis task.

```python
class TextClassificationTransformer:
    def __init__(self, vocab_size, num_classes, d_model=512):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_length=512)
        self.encoder_layer = TransformerEncoderBlock(d_model, num_heads=8, d_ff=2048)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        # Embed input tokens
        x = self.embedding(x)
        # Apply transformer encoder
        x = self.encoder_layer(x, mask)
        # Pool sequence representations
        x = x.mean(dim=1)  # Global average pooling
        # Classify
        return self.classifier(x)

# Training implementation
def train_classifier():
    # Initialize model and training components
    model = TextClassificationTransformer(vocab_size=30000, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Training loop with metrics
    best_accuracy = 0
    for epoch in range(10):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (texts, labels) in enumerate(train_dataloader):
            # Forward pass
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        accuracy = 100. * correct / total
        print(f'Epoch {epoch}, Accuracy: {accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step()
```

Slide 15: Results Display for Text Classification

Comprehensive evaluation results from the text classification model, including confusion matrix, precision-recall curves, and error analysis.

```python
def evaluate_classifier(model, test_dataloader):
    model.eval()
    predictions = []
    true_labels = []
    confidence_scores = []
    
    with torch.no_grad():
        for texts, labels in test_dataloader:
            outputs = model(texts)
            probs = F.softmax(outputs, dim=1)
            
            predictions.extend(probs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidence_scores.extend(probs.max(dim=1)[0].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Print results
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'true_labels': true_labels,
        'confidence_scores': confidence_scores
    }
```

Slide 16: Additional Resources

*   Large Language Models and Where to Find Them: A Study of Language Model Scaling • [https://arxiv.org/abs/2402.12366](https://arxiv.org/abs/2402.12366)
*   On the Effectiveness of Natural Language Model Compression • [https://arxiv.org/abs/2308.14256](https://arxiv.org/abs/2308.14256)
*   The Impact of Positional Encoding on Length Generalization in Transformers • [https://arxiv.org/abs/2305.19466](https://arxiv.org/abs/2305.19466)
*   A Survey of Length Extrapolation in Large Language Models • [https://arxiv.org/abs/2402.01257](https://arxiv.org/abs/2402.01257)
*   Scaling Laws and Interpretability of Learning from Repeated Data • [https://arxiv.org/abs/2301.07388](https://arxiv.org/abs/2301.07388)

