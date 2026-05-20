## Transformer Architecture for Efficient LLM Processing
Slide 1: Self-Attention Mechanism Implementation

The self-attention mechanism calculates attention scores between all pairs of input tokens, enabling the model to weigh the importance of different parts of the input sequence dynamically. This implementation demonstrates the core mathematical operations behind self-attention computation.

```python
import numpy as np

def self_attention(query, key, value, mask=None):
    # query, key, value shapes: (batch_size, seq_len, d_model)
    d_k = query.shape[-1]
    
    # Compute attention scores
    attention_scores = np.matmul(query, key.transpose(-2, -1))
    attention_scores = attention_scores / np.sqrt(d_k)
    
    if mask is not None:
        attention_scores += (mask * -1e9)
    
    # Apply softmax
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)
    
    # Compute weighted sum
    output = np.matmul(attention_weights, value)
    return output, attention_weights

# Example usage
seq_len, d_model = 4, 8
query = np.random.randn(1, seq_len, d_model)
key = np.random.randn(1, seq_len, d_model)
value = np.random.randn(1, seq_len, d_model)

output, weights = self_attention(query, key, value)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

Slide 2: Positional Encoding Generation

Positional encodings inject sequential information into the self-attention mechanism, which is otherwise position-invariant. This implementation creates sinusoidal position embeddings as described in the original Transformer paper.

```python
def positional_encoding(max_seq_length, d_model):
    position = np.arange(max_seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((max_seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return pos_encoding

# Generate and visualize positional encodings
max_seq_length, d_model = 100, 512
pos_enc = positional_encoding(max_seq_length, d_model)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.pcolormesh(pos_enc[:30, :30], cmap='RdBu')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.show()
```

Slide 3: Multi-Head Attention Implementation

Multi-head attention allows the model to jointly attend to information from different representation subspaces, enhancing the model's ability to capture various aspects of the input sequence simultaneously.

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.w_q = np.random.randn(d_model, d_model)
        self.w_k = np.random.randn(d_model, d_model)
        self.w_v = np.random.randn(d_model, d_model)
        self.w_o = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = np.matmul(query, self.w_q)
        K = np.matmul(key, self.w_k)
        V = np.matmul(value, self.w_v)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        attention_output, _ = self.attention(Q, K, V, mask)
        
        # Reshape and apply final linear layer
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = np.matmul(attention_output, self.w_o)
        
        return output
```

Slide 4: Transformer Layer Implementation

The transformer layer combines multi-head attention with position-wise feed-forward networks, layer normalization, and residual connections to create a powerful building block for sequence processing.

```python
class TransformerLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn_w1 = np.random.randn(d_model, d_ff)
        self.ffn_w2 = np.random.randn(d_ff, d_model)
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.mha.forward(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Position-wise feed-forward network
        ffn_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        
        return x
    
    def feed_forward(self, x):
        hidden = np.maximum(0, np.matmul(x, self.ffn_w1))  # ReLU
        output = np.matmul(hidden, self.ffn_w2)
        return output
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
        return x * mask / (1 - self.dropout_rate)
```

Slide 5: Tokenization and Vocabulary Implementation

Tokenization transforms raw text into numerical sequences that can be processed by the transformer. This implementation showcases a basic subword tokenizer using byte-pair encoding (BPE) principles for handling out-of-vocabulary words.

```python
class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
    def train(self, texts):
        # Count word frequencies
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Build vocabulary with most common words
        vocab_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab_words = vocab_words[:self.vocab_size - len(self.special_tokens)]
        
        # Create vocabulary mappings
        self.vocab = {**self.special_tokens}
        for i, (word, _) in enumerate(vocab_words):
            self.vocab[word] = i + len(self.special_tokens)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text):
        tokens = []
        for word in text.split():
            tokens.append(self.vocab.get(word, self.vocab['<UNK>']))
        return tokens
    
    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '<UNK>') for token in tokens])

# Example usage
texts = [
    "the quick brown fox jumps over the lazy dog",
    "hello world machine learning transformer model"
]
tokenizer = SimpleTokenizer(vocab_size=100)
tokenizer.train(texts)

# Test encoding and decoding
sample_text = "the quick fox"
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print(f"Original: {sample_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

Slide 6: Training Loss Implementation

The transformer's training process relies on calculating cross-entropy loss for next-token prediction. This implementation shows how to compute the loss while handling padding tokens correctly.

```python
import numpy as np

def compute_loss(logits, targets, pad_token_id=0):
    """
    Compute cross-entropy loss with padding token handling
    
    Args:
        logits: shape (batch_size, seq_len, vocab_size)
        targets: shape (batch_size, seq_len)
        pad_token_id: ID of padding token to ignore
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Create padding mask
    pad_mask = (targets != pad_token_id).astype(np.float32)
    
    # Convert targets to one-hot
    targets_one_hot = np.zeros((batch_size, seq_len, vocab_size))
    for i in range(batch_size):
        for j in range(seq_len):
            if targets[i, j] != pad_token_id:
                targets_one_hot[i, j, targets[i, j]] = 1
    
    # Compute cross-entropy loss
    log_probs = log_softmax(logits, axis=-1)
    loss = -np.sum(targets_one_hot * log_probs, axis=-1)
    
    # Apply padding mask
    masked_loss = loss * pad_mask
    
    # Average loss over non-padding tokens
    total_tokens = np.sum(pad_mask)
    loss = np.sum(masked_loss) / (total_tokens + 1e-8)
    
    return loss

def log_softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return x - max_x - np.log(np.sum(exp_x, axis=axis, keepdims=True))

# Example usage
batch_size, seq_len, vocab_size = 2, 5, 1000
logits = np.random.randn(batch_size, seq_len, vocab_size)
targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
targets[0, -1] = 0  # Add padding token

loss = compute_loss(logits, targets)
print(f"Training loss: {loss:.4f}")
```

Slide 7: Scaled Dot-Product Attention Implementation

The scaled dot-product attention is the fundamental building block of the transformer architecture, computing attention weights while accounting for sequence length through scaling.

```python
def scaled_dot_product_attention(query, key, value, mask=None, scale=True):
    """
    Implements scaled dot-product attention mechanism
    
    Args:
        query: shape (..., seq_len_q, depth)
        key: shape (..., seq_len_k, depth)
        value: shape (..., seq_len_v, depth)
        mask: shape (seq_len_q, seq_len_k)
        scale: whether to scale attention scores
    """
    # Compute attention scores
    matmul_qk = np.matmul(query, key.transpose(-2, -1))
    
    # Scale matmul_qk
    depth = query.shape[-1]
    if scale:
        matmul_qk = matmul_qk / np.sqrt(depth)
    
    if mask is not None:
        matmul_qk += (mask * -1e9)
    
    # Compute attention weights
    attention_weights = np.exp(matmul_qk) / np.sum(np.exp(matmul_qk), axis=-1, keepdims=True)
    
    # Apply attention weights to values
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
seq_len_q = 4
seq_len_k = 4
depth = 8

query = np.random.randn(1, seq_len_q, depth)
key = np.random.randn(1, seq_len_k, depth)
value = np.random.randn(1, seq_len_k, depth)

# Create causal mask for autoregressive attention
mask = np.triu(np.ones((seq_len_q, seq_len_k)), k=1)

output, attention = scaled_dot_product_attention(query, key, value, mask)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention.shape}")
```

Slide 8: Layer Normalization Implementation

Layer normalization stabilizes training by normalizing activations across the feature dimension. This implementation shows the forward and backward passes of layer normalization with learnable parameters.

```python
class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps
        
    def forward(self, x):
        # Calculate mean and variance along last dimension
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # Scale and shift
        out = self.gamma * self.x_norm + self.beta
        return out
    
    def backward(self, grad_out):
        # Compute gradients for gamma and beta
        grad_gamma = np.sum(grad_out * self.x_norm, axis=(0, 1))
        grad_beta = np.sum(grad_out, axis=(0, 1))
        
        # Compute gradient for input
        N = grad_out.shape[-1]
        grad_x = (1. / N) * self.gamma * (self.var + self.eps) ** (-1./2.) * (
            N * grad_out
            - np.sum(grad_out, axis=-1, keepdims=True)
            - self.x_norm * np.sum(grad_out * self.x_norm, axis=-1, keepdims=True)
        )
        return grad_x, grad_gamma, grad_beta

# Example usage
batch_size, seq_len, features = 2, 5, 512
x = np.random.randn(batch_size, seq_len, features)
layer_norm = LayerNorm(features)

# Forward pass
normalized = layer_norm.forward(x)
print(f"Input mean: {np.mean(x):.4f}, std: {np.std(x):.4f}")
print(f"Output mean: {np.mean(normalized):.4f}, std: {np.std(normalized):.4f}")
```

Slide 9: Transformer Training Loop Implementation

This implementation demonstrates a complete training loop for a transformer model, including gradient computation, parameter updates, and learning rate scheduling with warmup steps.

```python
class TransformerTrainer:
    def __init__(self, model, learning_rate=0.0001, warmup_steps=4000):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def learning_rate_schedule(self):
        # Implement learning rate warmup and decay
        arg1 = self.step ** (-0.5)
        arg2 = self.step * (self.warmup_steps ** -1.5)
        return self.learning_rate * min(arg1, arg2)
    
    def train_step(self, batch, targets):
        self.step += 1
        current_lr = self.learning_rate_schedule()
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(batch)
            loss = compute_loss(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        for param in self.model.parameters():
            param.data -= current_lr * param.grad
            param.grad.zero_()
        
        return loss.item()

    def train_epoch(self, data_loader):
        total_loss = 0
        num_batches = 0
        
        for batch, targets in data_loader:
            loss = self.train_step(batch, targets)
            total_loss += loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Batch {num_batches}, Loss: {loss:.4f}, LR: {self.learning_rate_schedule():.6f}")
        
        return total_loss / num_batches

# Example usage
batch_size = 32
vocab_size = 30000
max_seq_length = 512

model = TransformerModel(vocab_size=vocab_size, 
                        d_model=512,
                        num_heads=8,
                        num_layers=6)

trainer = TransformerTrainer(model)
print("Training started...")
```

Slide 10: Beam Search Decoding Implementation

Beam search improves text generation quality by maintaining multiple hypotheses during decoding. This implementation shows how to perform beam search with length normalization and early stopping.

```python
class BeamSearch:
    def __init__(self, model, beam_width=4, max_length=50, length_penalty=0.6):
        self.model = model
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
    
    def score_normalize(self, score, length):
        """Apply length normalization to the score"""
        return score / (length ** self.length_penalty)
    
    def decode(self, input_ids):
        # Initialize beam with start token
        beams = [(0, [self.model.bos_token_id])]
        completed_beams = []
        
        for step in range(self.max_length):
            candidates = []
            
            for score, sequence in beams:
                if sequence[-1] == self.model.eos_token_id:
                    completed_beams.append((score, sequence))
                    continue
                
                # Get model predictions
                with torch.no_grad():
                    logits = self.model(torch.tensor([sequence]))
                    probs = torch.softmax(logits[0, -1], dim=-1)
                    
                # Get top k predictions
                top_k_probs, top_k_ids = torch.topk(probs, self.beam_width)
                
                for prob, token_id in zip(top_k_probs, top_k_ids):
                    new_score = score + torch.log(prob).item()
                    new_sequence = sequence + [token_id.item()]
                    candidates.append((new_score, new_sequence))
            
            # Select top beams for next step
            candidates = sorted(candidates, 
                             key=lambda x: self.score_normalize(x[0], len(x[1])), 
                             reverse=True)
            beams = candidates[:self.beam_width]
            
            # Early stopping if all beams are completed
            if len(completed_beams) >= self.beam_width:
                break
        
        # Add remaining beams to completed list
        completed_beams.extend(beams)
        
        # Sort and return best sequence
        completed_beams = sorted(completed_beams,
                               key=lambda x: self.score_normalize(x[0], len(x[1])),
                               reverse=True)
        
        return completed_beams[0][1]

# Example usage
model = load_pretrained_model()  # Placeholder for loading model
beam_search = BeamSearch(model)

input_text = "Translate to French: Hello, how are you?"
input_ids = tokenize(input_text)  # Placeholder for tokenization
generated_ids = beam_search.decode(input_ids)
generated_text = detokenize(generated_ids)  # Placeholder for detokenization
print(f"Generated text: {generated_text}")
```

Slide 11: Attention Visualization Implementation

This implementation provides tools to visualize attention patterns in transformer models, helping understand how the model processes different parts of the input sequence.

```python
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_attention_maps(self, text):
        # Tokenize input
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        
        # Get model attention weights
        with torch.no_grad():
            outputs = self.model(tokens, output_attentions=True)
        
        # Extract attention weights from all layers and heads
        attention_maps = outputs.attentions
        
        return attention_maps, self.tokenizer.convert_ids_to_tokens(tokens[0])
    
    def plot_attention_head(self, attention_weights, tokens, layer, head):
        """Plot attention weights for a specific layer and head"""
        plt.figure(figsize=(10, 10))
        attention = attention_weights[layer][0, head].numpy()
        
        sns.heatmap(attention,
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap='viridis',
                    annot=True,
                    fmt='.2f')
        
        plt.title(f'Attention weights: Layer {layer}, Head {head}')
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_attention_summary(self, text):
        """Plot average attention across all layers and heads"""
        attention_maps, tokens = self.get_attention_maps(text)
        
        # Average attention across layers and heads
        avg_attention = torch.mean(torch.stack(attention_maps), dim=(0,1))[0].numpy()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_attention,
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap='viridis')
        
        plt.title('Average attention across all layers and heads')
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage
model = load_pretrained_model()  # Placeholder for loading model
tokenizer = load_tokenizer()     # Placeholder for loading tokenizer
visualizer = AttentionVisualizer(model, tokenizer)

text = "The transformer model processes text efficiently."
visualizer.plot_attention_summary(text)
```

Slide 12: Custom Learning Rate Scheduler Implementation

This implementation showcases a custom learning rate scheduler with warmup and cosine decay, essential for stable transformer training and optimal convergence.

```python
import numpy as np
import matplotlib.pyplot as plt

class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000, max_steps=100000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        # Implement the paper's learning rate schedule
        step = self.current_step
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        # Add cosine decay after warmup
        if step > self.warmup_steps:
            decay = 0.5 * (1 + np.cos(
                np.pi * (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            ))
            return (self.d_model ** -0.5) * min(arg1, arg2) * decay
        
        return (self.d_model ** -0.5) * min(arg1, arg2)
    
    def plot_schedule(self):
        """Visualize the learning rate schedule"""
        lrs = []
        steps = list(range(1, self.max_steps))
        
        for step in steps:
            self.current_step = step
            lrs.append(self.get_lr())
        
        plt.figure(figsize=(10, 5))
        plt.plot(steps, lrs)
        plt.axvline(x=self.warmup_steps, color='r', linestyle='--', 
                   label='End of Warmup')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Transformer Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
class DummyOptimizer:
    def __init__(self):
        self.param_groups = [{'lr': 0.0}]

optimizer = DummyOptimizer()
scheduler = TransformerLRScheduler(
    optimizer=optimizer,
    d_model=512,
    warmup_steps=4000,
    max_steps=100000
)

# Visualize the learning rate schedule
scheduler.plot_schedule()
```

Slide 13: Transformer Inference with Cache Implementation

This implementation demonstrates how to cache key and value tensors during autoregressive generation to improve inference speed by avoiding redundant computations.

```python
class CachedTransformerDecoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = num_layers
        self.cache = {}
        self.initialize_cache()
        
    def initialize_cache(self):
        """Initialize empty cache for keys and values"""
        self.cache = {
            f"layer_{i}": {
                "self_attention": {"keys": [], "values": []},
                "cross_attention": {"keys": [], "values": []}
            } for i in range(self.layers)
        }
    
    def update_cache(self, layer_idx, attn_type, key, value):
        """Update cache with new key-value pairs"""
        cache_entry = self.cache[f"layer_{layer_idx}"][attn_type]
        cache_entry["keys"].append(key)
        cache_entry["values"].append(value)
    
    def get_cached_attention(self, query, layer_idx, attn_type):
        """Compute attention using cached keys and values"""
        cache_entry = self.cache[f"layer_{layer_idx}"][attn_type]
        
        if not cache_entry["keys"]:
            return None
        
        # Concatenate cached keys and values
        keys = np.concatenate(cache_entry["keys"], axis=1)
        values = np.concatenate(cache_entry["values"], axis=1)
        
        # Compute attention with cached KV
        attention_output, _ = scaled_dot_product_attention(
            query=query,
            key=keys,
            value=values
        )
        
        return attention_output
    
    def forward_step(self, x, encoder_output=None):
        """Single step forward pass with caching"""
        batch_size, seq_len = x.shape[0], 1  # Only process one token at a time
        
        for i in range(self.layers):
            # Self attention
            query = self.get_query_projection(x)
            key = self.get_key_projection(x)
            value = self.get_value_projection(x)
            
            # Update cache
            self.update_cache(i, "self_attention", key, value)
            
            # Compute attention with cache
            attn_output = self.get_cached_attention(
                query, i, "self_attention"
            )
            
            x = x + attn_output
            x = self.layer_norm1(x)
            
            # Cross attention (if encoder output is provided)
            if encoder_output is not None:
                # Similar process for cross attention...
                pass
            
            # Feed forward
            x = x + self.feed_forward(x)
            x = self.layer_norm2(x)
        
        return x

# Example usage
model = CachedTransformerDecoder(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6
)

# Generate sequence token by token
input_ids = [1]  # Start token
max_length = 50

for _ in range(max_length):
    # Forward pass with cache
    x = np.array([input_ids])
    output = model.forward_step(x)
    
    # Get next token
    next_token = get_next_token(output)  # Placeholder function
    input_ids.append(next_token)
    
    if next_token == end_token_id:
        break
```

Slide 14: Additional Resources

*   Attention Is All You Need (Original Transformer Paper)
    *   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   BERT: Pre-training of Deep Bidirectional Transformers
    *   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   GPT: Improving Language Understanding by Generative Pre-Training
    *   [https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language\_understanding\_paper.pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
*   Layer Normalization in Transformer Models
    *   [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
*   Synthesizer: Rethinking Self-Attention in Transformer Models
    *   [https://arxiv.org/abs/2005.00743](https://arxiv.org/abs/2005.00743)

