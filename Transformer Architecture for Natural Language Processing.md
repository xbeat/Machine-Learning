## Transformer Architecture for Natural Language Processing
Slide 1: Introduction to Transformer Architecture

The Transformer architecture revolutionized natural language processing by introducing self-attention mechanisms, enabling parallel processing of sequential data and capturing long-range dependencies more effectively than traditional RNNs and LSTMs. This fundamental shift in approach has become the backbone of modern language models.

```python
import torch
import torch.nn as nn
import math

class TransformerConfig:
    def __init__(self):
        self.hidden_size = 512
        self.num_attention_heads = 8
        self.dropout = 0.1
        self.max_position_embeddings = 512
        self.vocab_size = 30000
        self.num_hidden_layers = 6
        
# Example usage
config = TransformerConfig()
print(f"Hidden size: {config.hidden_size}")
print(f"Number of attention heads: {config.num_attention_heads}")
```

Slide 2: Self-Attention Mechanism Core

Self-attention allows each element in a sequence to attend to all other elements, computing attention scores through query, key, and value matrices. This mechanism forms the foundation of the transformer's ability to process contextual relationships.

```python
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers for Q, K, V transformations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
```

Slide 3: Attention Score Calculation

The attention mechanism computes scores by taking the dot product of queries and keys, scaling by the square root of the dimension, and applying softmax to obtain attention weights. These weights determine how much each token influences the final representation.

```python
def attention_scores(query, key, value, mask=None):
    # Shape: (batch_size, num_heads, seq_len, head_dim)
    attention_weights = torch.matmul(query, key.transpose(-1, -2))
    
    # Scale attention scores
    d_k = query.size(-1)
    attention_weights = attention_weights / math.sqrt(d_k)
    
    if mask is not None:
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
    
    # Apply softmax
    attention_probs = nn.Softmax(dim=-1)(attention_weights)
    
    # Calculate weighted sum
    context_layer = torch.matmul(attention_probs, value)
    return context_layer, attention_probs
```

Slide 4: Positional Encoding Implementation

Positional encodings inject sequential information into the model using sinusoidal functions, allowing the transformer to understand token positions despite processing them in parallel. This mathematical approach ensures unique position representations.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculate positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer to save memory
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

Slide 5: Multi-Head Attention

Multi-head attention enables the model to attend to information from different representation subspaces simultaneously, allowing it to capture various types of relationships between tokens in parallel processing streams.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        # Compute self attention
        self_outputs, attention_probs = self.self_attention(
            hidden_states, hidden_states, hidden_states, attention_mask
        )
        
        # Project outputs
        attention_output = self.output(self_outputs)
        attention_output = self.dropout(attention_output)
        
        return attention_output, attention_probs
```

Slide 6: Feed-Forward Neural Network

The feed-forward network in each transformer layer processes each position independently, applying two linear transformations with a ReLU activation in between. This component allows the model to incorporate non-linear transformations of the attention outputs.

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```

Slide 7: Layer Normalization Implementation

Layer normalization stabilizes the learning process by normalizing the activations across the features. It's applied before each major component in the transformer architecture, following the pre-norm design pattern.

```python
class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.eps = 1e-12
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta
```

Slide 8: Complete Transformer Layer

A transformer layer combines self-attention, feed-forward networks, and layer normalization in a specific arrangement. This implementation shows how these components work together to process input sequences.

```python
class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = LayerNorm(config)
        self.ffn = FeedForward(config)
        self.layer_norm2 = LayerNorm(config)
        
    def forward(self, hidden_states, attention_mask=None):
        # Self attention block
        attention_output, attention_probs = self.attention(
            self.layer_norm1(hidden_states),
            attention_mask
        )
        hidden_states = hidden_states + attention_output
        
        # Feed-forward block
        ffn_output = self.ffn(self.layer_norm2(hidden_states))
        hidden_states = hidden_states + ffn_output
        
        return hidden_states, attention_probs
```

Slide 9: Token and Position Embeddings

The embedding layer combines token embeddings with positional encodings to create input representations that capture both semantic meaning and sequential position information for the transformer model.

```python
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = PositionalEncoding(config.hidden_size, config.max_position_embeddings)
        self.layer_norm = LayerNorm(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.position_embeddings(embeddings)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

Slide 10: Complete Transformer Encoder

The transformer encoder stacks multiple transformer layers to process input sequences. This implementation shows how to combine all previously defined components into a complete encoder architecture.

```python
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        all_attention_probs = []
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            all_attention_probs.append(attention_probs)
            
        return hidden_states, all_attention_probs
```

Slide 11: Training Configuration and Loss Function

The transformer model requires careful configuration of hyperparameters and an appropriate loss function for training. This implementation demonstrates setting up training parameters and implementing masked language modeling loss.

```python
class TransformerTrainingConfig:
    def __init__(self):
        self.learning_rate = 1e-4
        self.warmup_steps = 10000
        self.max_grad_norm = 1.0
        self.weight_decay = 0.01
        
class TransformerForMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TransformerEncoder(config)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids, labels=None, attention_mask=None):
        hidden_states, _ = self.transformer(input_ids, attention_mask)
        prediction_scores = self.mlm_head(hidden_states)
        
        if labels is not None:
            loss = self.loss_fn(prediction_scores.view(-1, prediction_scores.size(-1)), 
                              labels.view(-1))
            return loss, prediction_scores
        return prediction_scores
```

Slide 12: Practical Implementation with Real Data

This implementation shows how to process real text data and train a transformer model for masked language modeling using a practical dataset preprocessing pipeline.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length',
                                 max_length=max_length, return_tensors='pt')
        self.labels = self.encodings['input_ids'].clone()
        
    def __len__(self):
        return len(self.encodings['input_ids'])
        
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

# Example usage
texts = ["Example text for training", "Another example text"]
tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

Slide 13: Training Loop Implementation

A comprehensive training loop implementation showcasing the complete process of training a transformer model, including gradient computation, optimization, and learning rate scheduling.

```python
def train_transformer(model, dataloader, config, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step / config.warmup_steps, 1.0)
    )
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            loss, _ = model(input_ids, labels=labels, 
                          attention_mask=attention_mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                         config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - Original Transformer Paper [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Language Models are Few-Shot Learners" (GPT-3) [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
*   "An Empirical Study of Transformer Architecture Components" [https://arxiv.org/abs/2203.03014](https://arxiv.org/abs/2203.03014)
*   For additional research and implementation details:
    *   Google AI Blog: [https://ai.googleblog.com](https://ai.googleblog.com)
    *   Papers With Code: [https://paperswithcode.com/method/transformer](https://paperswithcode.com/method/transformer)
    *   Hugging Face Documentation: [https://huggingface.co/docs](https://huggingface.co/docs)

