## Introduction to Large Language Models
Slide 1: Understanding LLM Tokenization

Tokenization is the fundamental process of converting raw text into numerical tokens that language models can process. This implementation demonstrates a basic tokenizer using character-level encoding, similar to what more sophisticated models use as their foundation.

```python
class SimpleTokenizer:
    def __init__(self):
        # Create vocabulary from printable ASCII characters
        self.char_to_id = {chr(i): i-32 for i in range(32, 127)}
        self.id_to_char = {i-32: chr(i) for i in range(32, 127)}
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text):
        return [self.char_to_id[char] for char in text if char in self.char_to_id]
    
    def decode(self, tokens):
        return ''.join([self.id_to_char[token] for token in tokens])

# Example usage
tokenizer = SimpleTokenizer()
text = "Hello, LLM!"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)

print(f"Original: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
```

Slide 2: Implementing Byte-Pair Encoding

Byte-Pair Encoding (BPE) is a crucial tokenization algorithm used in modern LLMs that iteratively merges the most frequent pairs of bytes or characters to form new tokens, optimizing the vocabulary for common subwords.

```python
from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = None
    
    def get_stats(self, words):
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out
    
    def train(self, text):
        # Initialize with characters
        word_freqs = defaultdict(int)
        for word in text.split():
            word = ' '.join(list(word)) + ' </w>'
            word_freqs[word] += 1
        
        vocab = dict(word_freqs)
        num_merges = self.vocab_size - len(vocab)
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges[best] = i
```

Slide 3: Attention Mechanism Implementation

Attention mechanisms allow models to weigh the importance of different input tokens when generating each output token. This implementation shows the core mathematical operations behind self-attention, including query, key, and value transformations.

```python
import numpy as np

class SelfAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_head)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, x):
        # Linear transformations
        Q = np.dot(x, self.W_q)  # Query
        K = np.dot(x, self.W_k)  # Key
        V = np.dot(x, self.W_v)  # Value
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        scores = scores / np.sqrt(self.d_head)
        attention_weights = np.softmax(scores, axis=-1)
        
        # Apply attention to values
        attention_output = np.matmul(attention_weights, V)
        
        return attention_output, attention_weights
```

Slide 4: Positional Encoding From Scratch

Positional encodings allow transformer models to understand token positions in sequences. This implementation demonstrates the sinusoidal positional encoding used in the original transformer paper, creating unique position-dependent patterns.

```python
import numpy as np

class PositionalEncoding:
    def __init__(self, d_model, max_seq_length=5000):
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_seq_length, d_model))
        position = np.arange(0, max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe[np.newaxis, :, :]  # Add batch dimension
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
        """
        seq_length = x.shape[1]
        return x + self.pe[:, :seq_length, :]

# Example usage
d_model = 512
pos_encoder = PositionalEncoding(d_model)
sequence = np.random.randn(1, 100, d_model)  # Batch size 1, sequence length 100
encoded_sequence = pos_encoder.forward(sequence)

print(f"Input shape: {sequence.shape}")
print(f"Encoded shape: {encoded_sequence.shape}")
print(f"First position encoding: {pos_encoder.pe[0, 0, :10]}")  # First 10 values
```

Slide 5: Multi-Head Attention Implementation

Multi-head attention allows the model to jointly attend to information from different representation subspaces. This implementation shows the complete mechanism including parallel attention heads and output projection.

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights for Q, K, V projections
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
        
        # Initialize scaling factor
        self.scale = 1 / np.sqrt(self.d_k)
    
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = np.dot(query, self.W_q)
        K = np.dot(key, self.W_k)
        V = np.dot(value, self.W_v)
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = np.ma.masked_array(scores, mask=mask)
        
        attention_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attention_weights /= attention_weights.sum(axis=-1, keepdims=True)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)
        
        # Combine heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = np.dot(context, self.W_o)
        
        return output, attention_weights
```

Slide 6: Feed-Forward Neural Network Layer

The feed-forward network in transformers processes each position independently with the same fully connected layer. This implementation shows the standard two-layer FFN with ReLU activation.

```python
import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)
    
    def forward(self, x, training=True):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            training: Boolean indicating training mode
        """
        # First linear layer
        hidden = np.dot(x, self.W1) + self.b1
        
        # ReLU activation
        hidden = self.relu(hidden)
        
        # Dropout during training
        if training:
            hidden = self.dropout(hidden)
        
        # Second linear layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
```

Slide 7: Layer Normalization Implementation

Layer normalization is crucial for stable training in transformers, normalizing the inputs across the feature dimension. This implementation shows the complete normalization process with learned affine parameters.

```python
import numpy as np

class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, features)
        """
        # Calculate mean and variance along features dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        
        # Scale and shift with learned parameters
        return self.gamma * x_norm + self.beta

# Example usage
batch_size, seq_length, features = 32, 50, 512
layer_norm = LayerNorm(features)
x = np.random.randn(batch_size, seq_length, features)
normalized = layer_norm.forward(x)

print(f"Input mean: {np.mean(x):.6f}, std: {np.std(x):.6f}")
print(f"Output mean: {np.mean(normalized):.6f}, std: {np.std(normalized):.6f}")
```

Slide 8: Transformer Encoder Layer

The encoder layer combines multi-head attention, feed-forward networks, and normalization layers. This implementation shows how these components work together in a single encoder block.

```python
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)
    
    def forward(self, x, mask=None, training=True):
        # Multi-head self-attention
        attn_output, _ = self.mha.forward(x, x, x, mask)
        if training:
            attn_output = self.dropout(attn_output)
        out1 = self.norm1.forward(x + attn_output)
        
        # Feed-forward network
        ff_output = self.ff.forward(out1, training)
        if training:
            ff_output = self.dropout(ff_output)
        out2 = self.norm2.forward(out1 + ff_output)
        
        return out2

# Example usage
d_model, num_heads, d_ff = 512, 8, 2048
encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
x = np.random.randn(32, 50, d_model)  # batch_size=32, seq_length=50
encoded = encoder_layer.forward(x)
print(f"Output shape: {encoded.shape}")
```

Slide 9: Building the Embedding Layer

The embedding layer converts token indices into dense vectors and scales them appropriately. This implementation includes both token embeddings and optional embedding weight sharing.

```python
class Embeddings:
    def __init__(self, vocab_size, d_model):
        self.d_model = d_model
        # Initialize embedding matrix
        self.embedding_matrix = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of token indices (batch_size, seq_length)
        """
        # Convert indices to embeddings
        embeddings = self.embedding_matrix[x]
        
        # Scale embeddings
        return embeddings * np.sqrt(self.d_model)
    
    def shared_weights(self):
        """Return embedding weights for potential weight sharing with output layer"""
        return self.embedding_matrix.T

# Example usage
vocab_size, d_model = 30000, 512
embedding_layer = Embeddings(vocab_size, d_model)
tokens = np.random.randint(0, vocab_size, (32, 50))  # batch_size=32, seq_length=50
embedded = embedding_layer.forward(tokens)
print(f"Input shape: {tokens.shape}")
print(f"Embedded shape: {embedded.shape}")
```

Slide 10: Complete Transformer Architecture

This implementation combines all previous components into a complete transformer model, showing how encoder and decoder layers work together to process sequences for various NLP tasks.

```python
class Transformer:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        self.embedding = Embeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Create encoder and decoder stacks
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Final layer normalization
        self.final_norm = LayerNorm(d_model)
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)
    
    def encode(self, src, training=True):
        # Embed and add positional encoding
        x = self.embedding.forward(src)
        x = self.positional_encoding.forward(x)
        
        # Pass through encoder layers
        enc_output = x
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer.forward(enc_output, training=training)
        
        return self.final_norm.forward(enc_output)
    
    def forward(self, src, training=True):
        enc_output = self.encode(src, training)
        
        # Project to vocabulary
        logits = np.dot(enc_output, self.output_projection)
        
        # Apply softmax
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        
        return probs

# Example usage
config = {
    'vocab_size': 30000,
    'd_model': 512,
    'num_heads': 8,
    'd_ff': 2048,
    'num_layers': 6
}

model = Transformer(**config)
src_tokens = np.random.randint(0, config['vocab_size'], (32, 50))
output_probs = model.forward(src_tokens)
print(f"Output probability shape: {output_probs.shape}")
```

Slide 11: Practical Example - Text Classification

This example shows how to use the transformer for a real-world text classification task, including data preprocessing and training loop implementation.

```python
class TextClassifier:
    def __init__(self, transformer_config, num_classes):
        self.transformer = Transformer(**transformer_config)
        self.classification_head = np.random.randn(
            transformer_config['d_model'], 
            num_classes
        ) / np.sqrt(transformer_config['d_model'])
    
    def preprocess_text(self, texts, tokenizer, max_length=512):
        # Convert texts to token indices
        tokenized = [tokenizer.encode(text) for text in texts]
        
        # Pad sequences
        padded = np.zeros((len(texts), max_length), dtype=np.int32)
        for i, tokens in enumerate(tokenized):
            length = min(len(tokens), max_length)
            padded[i, :length] = tokens[:length]
        
        return padded
    
    def forward(self, x, training=True):
        # Get transformer encodings
        encoded = self.transformer.encode(x, training)
        
        # Use [CLS] token representation (first token)
        cls_encoding = encoded[:, 0, :]
        
        # Project to class logits
        logits = np.dot(cls_encoding, self.classification_head)
        
        # Apply softmax
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        
        return probs

# Example usage with synthetic data
texts = [
    "This movie was amazing!",
    "The book was disappointing.",
    "I loved the performance."
]
tokenizer = SimpleTokenizer()  # Using the tokenizer from Slide 1
classifier = TextClassifier(config, num_classes=3)

# Preprocess and classify
inputs = classifier.preprocess_text(texts, tokenizer)
predictions = classifier.forward(inputs)
print(f"Predicted probabilities shape: {predictions.shape}")
```

Slide 12: Advanced Training Techniques

This implementation demonstrates key training optimizations used in modern transformer models, including gradient accumulation and mixed precision training for handling large models efficiently.

```python
import numpy as np

class TransformerTrainer:
    def __init__(self, model, learning_rate=1e-4, gradient_accumulation_steps=4):
        self.model = model
        self.learning_rate = learning_rate
        self.grad_acc_steps = gradient_accumulation_steps
        self.accumulated_gradients = None
        
    def compute_loss(self, logits, labels):
        """Cross entropy loss with label smoothing"""
        smoothing = 0.1
        confidence = 1.0 - smoothing
        
        # Create smoothed labels
        smooth_labels = np.full(logits.shape, smoothing / (logits.shape[-1] - 1))
        smooth_labels[np.arange(len(labels)), labels] = confidence
        
        log_probs = np.log(logits + 1e-10)
        loss = -np.sum(smooth_labels * log_probs) / len(labels)
        return loss
    
    def backward(self, loss, gradients):
        """Accumulate gradients with scaling"""
        if self.accumulated_gradients is None:
            self.accumulated_gradients = gradients
        else:
            for key in gradients:
                self.accumulated_gradients[key] += gradients[key]
    
    def optimizer_step(self):
        """Apply accumulated gradients with weight decay"""
        weight_decay = 0.01
        
        for key, grad in self.accumulated_gradients.items():
            # Scale gradients by accumulation steps
            grad /= self.grad_acc_steps
            
            # Add weight decay
            if 'weight' in key:
                grad += weight_decay * self.model.get_parameter(key)
            
            # Update parameters
            self.model.update_parameter(
                key, 
                -self.learning_rate * grad
            )
        
        self.accumulated_gradients = None

# Example training loop
trainer = TransformerTrainer(model)  # model from previous slide
batch_size = 32
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = len(train_data) // batch_size
    
    for i in range(num_batches):
        # Get batch
        batch_x = train_data[i * batch_size:(i + 1) * batch_size]
        batch_y = train_labels[i * batch_size:(i + 1) * batch_size]
        
        # Forward pass
        logits = model.forward(batch_x, training=True)
        loss = trainer.compute_loss(logits, batch_y)
        
        # Backward pass with gradient accumulation
        grads = model.backward(loss)
        trainer.backward(loss, grads)
        
        if (i + 1) % trainer.grad_acc_steps == 0:
            trainer.optimizer_step()
        
        total_loss += loss
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
```

Slide 13: Language Model Pre-training

This implementation shows how to pre-train a transformer model using masked language modeling, the technique used to train models like BERT.

```python
class MaskedLanguageModel:
    def __init__(self, transformer_config, mask_prob=0.15):
        self.transformer = Transformer(**transformer_config)
        self.mask_prob = mask_prob
        self.vocab_size = transformer_config['vocab_size']
        self.mask_token_id = self.vocab_size - 1  # Special mask token
    
    def create_masked_input(self, input_ids):
        """Create masked input and labels for MLM"""
        masked_inputs = input_ids.copy()
        labels = np.full_like(input_ids, -100)  # -100 indicates no loss
        
        # Create random mask
        probability_matrix = np.random.rand(*input_ids.shape)
        mask = probability_matrix < self.mask_prob
        
        # Create labels for masked tokens
        labels[mask] = input_ids[mask]
        
        # Replace masked tokens with [MASK]
        masked_inputs[mask] = self.mask_token_id
        
        return masked_inputs, labels
    
    def forward(self, input_ids, training=True):
        if training:
            masked_inputs, labels = self.create_masked_input(input_ids)
        else:
            masked_inputs = input_ids
            labels = None
        
        # Get transformer predictions
        logits = self.transformer.forward(masked_inputs, training=training)
        
        if training:
            return logits, labels
        return logits

# Example pre-training step
mlm = MaskedLanguageModel(config)
input_sequence = np.random.randint(0, config['vocab_size'], (16, 512))
logits, labels = mlm.forward(input_sequence)

print(f"MLM logits shape: {logits.shape}")
print(f"Masked positions: {np.sum(labels != -100)}")
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - Original Transformer Paper [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Language Models are Few-Shot Learners" - GPT-3 Paper [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
*   "On Layer Normalization in the Transformer Architecture" [https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)
*   "An Empirical Study of Training Self-Supervised Vision Transformers" [https://arxiv.org/abs/2104.02057](https://arxiv.org/abs/2104.02057)

