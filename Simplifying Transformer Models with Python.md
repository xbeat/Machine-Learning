## Simplifying Transformer Models with Python
Slide 1: Tokenization Fundamentals

The tokenization process is fundamental to transformer models, converting raw text into numerical tokens that can be processed. This implementation demonstrates a basic tokenizer using vocabulary building and token mapping techniques.

```python
class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        # Build vocabulary from input texts
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        # Create mappings
        for word in sorted(words):
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
    
    def encode(self, text):
        return [self.word_to_id.get(word.lower(), 0) for word in text.split()]
    
    def decode(self, tokens):
        return ' '.join([self.id_to_word.get(token, '<UNK>') for token in tokens])

# Example usage
tokenizer = SimpleTokenizer()
texts = ["The transformer model", "model architecture"]
tokenizer.fit(texts)
encoded = tokenizer.encode("The transformer")
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
```

Slide 2: Word Embeddings Implementation

Word embeddings transform tokens into dense vectors, capturing semantic relationships. This implementation creates a basic embedding layer with random initialization and lookup functionality.

```python
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim):
        # Initialize random embeddings
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        
    def forward(self, token_ids):
        # Look up embeddings for given token IDs
        return self.embeddings[token_ids]
    
    def get_embedding(self, token_id):
        return self.embeddings[token_id]

# Example usage
vocab_size, embedding_dim = 1000, 64
embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

# Sample token sequence
token_ids = np.array([1, 4, 2, 7])
token_embeddings = embedding_layer.forward(token_ids)
print(f"Input shape: {token_ids.shape}")
print(f"Output shape: {token_embeddings.shape}")
print(f"Sample embedding:\n{token_embeddings[0][:5]}")  # First 5 dimensions
```

Slide 3: Positional Encoding

Positional encoding adds information about token positions in the sequence. This implementation shows both sinusoidal and learned positional encodings methods.

```python
import numpy as np

def sinusoidal_position_encoding(seq_length, d_model):
    positions = np.arange(seq_length)[:, np.newaxis]
    dims = np.arange(0, d_model, 2)[np.newaxis, :]
    
    # Calculate angles using wavelengths
    angles = positions / np.power(10000, (2 * dims) / d_model)
    
    # Apply sin to even indices
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angles)
    pos_encoding[:, 1::2] = np.cos(angles)
    
    return pos_encoding

# Example usage
seq_length, d_model = 10, 64
pos_encoding = sinusoidal_position_encoding(seq_length, d_model)
print(f"Positional encoding shape: {pos_encoding.shape}")
print(f"First position encoding:\n{pos_encoding[0, :5]}")  # First 5 dimensions
```

Slide 4: Self-Attention Mechanism

The self-attention mechanism allows the model to weigh the importance of different tokens in relation to each other, forming the core of the transformer architecture.

```python
import numpy as np

class SelfAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Initialize weights
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        
    def forward(self, X):
        batch_size, seq_length, _ = X.shape
        
        # Linear projections
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Calculate attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1, 3))
        scores = scores / np.sqrt(self.head_dim)
        
        # Apply softmax
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output.reshape(batch_size, seq_length, self.d_model)

# Example usage
batch_size, seq_length, d_model = 2, 5, 64
attention = SelfAttention(d_model, num_heads=8)
X = np.random.randn(batch_size, seq_length, d_model)
output = attention.forward(X)
print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
```

Slide 5: Multi-Head Attention

Multi-head attention enables the model to focus on different aspects of the input simultaneously. This implementation demonstrates parallel attention heads processing and concatenation.

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights for all heads at once
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
        
    def split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and split heads
        Q = self.split_heads(np.dot(query, self.W_q))
        K = self.split_heads(np.dot(key, self.W_k))
        V = self.split_heads(np.dot(value, self.W_v))
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.ma.masked_array(scores, mask=mask)
        
        attention = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Apply attention to values
        context = np.matmul(attention, V)
        
        # Concatenate heads and apply final linear layer
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = np.dot(context, self.W_o)
        
        return output, attention

# Example usage
d_model, num_heads = 64, 8
batch_size, seq_length = 2, 10
mha = MultiHeadAttention(d_model, num_heads)

query = np.random.randn(batch_size, seq_length, d_model)
output, attention = mha.forward(query, query, query)
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")
```

Slide 6: Feed-Forward Network Implementation

The feed-forward network processes each position independently using two linear transformations with a ReLU activation in between, allowing the model to capture complex patterns.

```python
import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # First linear layer
        hidden = np.dot(x, self.W1) + self.b1
        
        # ReLU activation
        hidden = self.relu(hidden)
        
        # Second linear layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output

# Example usage
d_model, d_ff = 64, 256
batch_size, seq_length = 2, 10
ffn = FeedForward(d_model, d_ff)

x = np.random.randn(batch_size, seq_length, d_model)
output = ffn.forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Sample output:\n{output[0, 0, :5]}")  # First 5 dimensions of first position
```

Slide 7: Layer Normalization

Layer normalization stabilizes the learning process by normalizing the activations across features. This implementation shows both the forward pass and backward pass computations.

```python
import numpy as np

class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        
    def forward(self, x):
        # Calculate mean and variance along last axis
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # Scale and shift
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_output):
        # Compute gradients for gamma and beta
        grad_gamma = np.sum(grad_output * self.x_norm, axis=(0, 1))
        grad_beta = np.sum(grad_output, axis=(0, 1))
        
        # Compute gradient for input
        N = grad_output.shape[-1]
        grad_x = (1. / N) * self.gamma * (self.var + self.eps)**(-0.5) * (
            N * grad_output
            - np.sum(grad_output, axis=-1, keepdims=True)
            - self.x_norm * np.sum(grad_output * self.x_norm, axis=-1, keepdims=True)
        )
        
        return grad_x, grad_gamma, grad_beta

# Example usage
batch_size, seq_length, features = 2, 10, 64
ln = LayerNorm(features)

x = np.random.randn(batch_size, seq_length, features)
output = ln.forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Mean of normalized output: {np.mean(output):.6f}")
print(f"Std of normalized output: {np.std(output):.6f}")
```

Slide 8: Encoder Block Implementation

The encoder block combines self-attention and feed-forward networks with residual connections and layer normalization to process input sequences effectively.

```python
import numpy as np

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
        
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1-self.dropout_rate)
    
    def forward(self, x, mask=None, training=True):
        # Multi-head attention
        attn_output, _ = self.mha.forward(x, x, x, mask)
        if training:
            attn_output = self.dropout(attn_output)
        out1 = self.norm1.forward(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn.forward(out1)
        if training:
            ffn_output = self.dropout(ffn_output)
        out2 = self.norm2.forward(out1 + ffn_output)
        
        return out2

# Example usage
d_model, num_heads, d_ff = 64, 8, 256
encoder = EncoderBlock(d_model, num_heads, d_ff)

batch_size, seq_length = 2, 10
x = np.random.randn(batch_size, seq_length, d_model)
output = encoder.forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Sample output features:\n{output[0, 0, :5]}")
```

Slide 9: Decoder Block Implementation

The decoder block extends the encoder with masked self-attention and cross-attention mechanisms, enabling the model to generate sequential outputs while maintaining causality.

```python
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate
    
    def create_causal_mask(self, size):
        # Create lower triangular matrix
        mask = np.triu(np.ones((size, size)), k=1).astype(bool)
        return mask
    
    def forward(self, x, enc_output, training=True):
        seq_length = x.shape[1]
        causal_mask = self.create_causal_mask(seq_length)
        
        # Masked self-attention
        attn1_output, _ = self.masked_mha.forward(x, x, x, causal_mask)
        if training:
            attn1_output = self.dropout(attn1_output)
        out1 = self.norm1.forward(x + attn1_output)
        
        # Cross-attention with encoder output
        attn2_output, _ = self.cross_mha.forward(out1, enc_output, enc_output)
        if training:
            attn2_output = self.dropout(attn2_output)
        out2 = self.norm2.forward(out1 + attn2_output)
        
        # Feed-forward network
        ffn_output = self.ffn.forward(out2)
        if training:
            ffn_output = self.dropout(ffn_output)
        out3 = self.norm3.forward(out2 + ffn_output)
        
        return out3

# Example usage
decoder = DecoderBlock(d_model, num_heads, d_ff)
enc_output = np.random.randn(batch_size, seq_length, d_model)
dec_input = np.random.randn(batch_size, seq_length, d_model)

output = decoder.forward(dec_input, enc_output)
print(f"Decoder input shape: {dec_input.shape}")
print(f"Encoder output shape: {enc_output.shape}")
print(f"Decoder output shape: {output.shape}")
```

Slide 10: Complete Transformer Model

The complete transformer model integrates all components into a cohesive architecture capable of processing sequential data for various tasks.

```python
class Transformer:
    def __init__(self, 
                 vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout_rate=0.1):
        
        # Embedding layers
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding
        
        # Encoder and Decoder stacks
        self.encoder_layers = [
            EncoderBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_encoder_layers)
        ]
        
        self.decoder_layers = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_decoder_layers)
        ]
        
        # Final linear layer
        self.final_layer = np.random.randn(d_model, vocab_size) * 0.01
        
    def forward(self, src, tgt, training=True):
        # Embed and add positional encoding
        src_emb = self.embedding.forward(src)
        tgt_emb = self.embedding.forward(tgt)
        
        seq_length = src_emb.shape[1]
        pos_enc = self.pos_encoding(seq_length, src_emb.shape[-1])
        
        src_emb += pos_enc
        tgt_emb += pos_enc
        
        # Encoder
        enc_output = src_emb
        for encoder in self.encoder_layers:
            enc_output = encoder.forward(enc_output, training=training)
        
        # Decoder
        dec_output = tgt_emb
        for decoder in self.decoder_layers:
            dec_output = decoder.forward(dec_output, enc_output, training=training)
        
        # Final linear layer and softmax
        logits = np.dot(dec_output, self.final_layer)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        
        return probs

# Example usage
vocab_size = 10000
transformer = Transformer(vocab_size)

# Sample input
batch_size, seq_length = 2, 20
src_tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))
tgt_tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))

output = transformer.forward(src_tokens, tgt_tokens)
print(f"Input shapes: {src_tokens.shape}, {tgt_tokens.shape}")
print(f"Output shape: {output.shape}")
print(f"Output probabilities sum to 1: {np.allclose(np.sum(output, axis=-1), 1.0)}")
```

Slide 11: Training Implementation

The training process involves implementing loss calculation, backpropagation, and optimization for the transformer model using cross-entropy loss and Adam optimizer.

```python
import numpy as np

class TransformerTrainer:
    def __init__(self, model, learning_rate=0.0001, beta1=0.9, beta2=0.98):
        self.model = model
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-9
        
        # Initialize Adam optimizer parameters
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep
        
    def cross_entropy_loss(self, predictions, targets):
        """
        Calculate cross entropy loss with label smoothing
        """
        epsilon = 0.1  # Label smoothing factor
        n_classes = predictions.shape[-1]
        
        # Create smoothed labels
        smooth_targets = np.full_like(predictions, epsilon / (n_classes - 1))
        smooth_targets[np.arange(len(targets)), targets] = 1.0 - epsilon
        
        # Calculate loss
        log_probs = -np.log(predictions + 1e-10)
        loss = np.sum(smooth_targets * log_probs) / len(targets)
        return loss
    
    def train_step(self, src_batch, tgt_batch):
        # Forward pass
        predictions = self.model.forward(src_batch, tgt_batch, training=True)
        loss = self.cross_entropy_loss(predictions, tgt_batch)
        
        # Backward pass (simplified for demonstration)
        gradients = self._compute_gradients(loss, predictions, tgt_batch)
        
        # Update parameters using Adam
        self._adam_update(gradients)
        
        return loss
    
    def _compute_gradients(self, loss, predictions, targets):
        # Simplified gradient computation
        # In practice, you would use automatic differentiation
        gradients = {}
        for name, param in self.model.named_parameters():
            gradients[name] = np.random.randn(*param.shape)  # Placeholder
        return gradients
    
    def _adam_update(self, gradients):
        self.t += 1
        
        for name, grad in gradients.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(grad)
                self.v[name] = np.zeros_like(grad)
            
            # Update biased first moment
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second moment
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * np.square(grad)
            
            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param = getattr(self.model, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage
vocab_size = 10000
transformer = Transformer(vocab_size)
trainer = TransformerTrainer(transformer)

# Training loop example
batch_size, seq_length = 32, 20
for epoch in range(5):
    # Simulate batch data
    src_batch = np.random.randint(0, vocab_size, (batch_size, seq_length))
    tgt_batch = np.random.randint(0, vocab_size, (batch_size, seq_length))
    
    loss = trainer.train_step(src_batch, tgt_batch)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
```

Slide 12: Machine Translation Example

This implementation demonstrates how to use the transformer model for machine translation, including preprocessing, training, and inference phases.

```python
class TranslationTransformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, max_length=100):
        self.transformer = Transformer(
            vocab_size=max(src_vocab_size, tgt_vocab_size),
            d_model=512,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        self.max_length = max_length
        
    def translate(self, src_text, src_tokenizer, tgt_tokenizer):
        # Tokenize source text
        src_tokens = src_tokenizer.encode(src_text)
        src_tokens = np.array([src_tokens])  # Add batch dimension
        
        # Initialize target sequence with start token
        tgt_tokens = np.array([[tgt_tokenizer.token_to_id['<START>']]])
        
        # Generate translation autoregressively
        for _ in range(self.max_length):
            predictions = self.transformer.forward(src_tokens, tgt_tokens, training=False)
            next_token = np.argmax(predictions[0, -1])
            
            # Stop if end token is generated
            if next_token == tgt_tokenizer.token_to_id['<END>']:
                break
                
            # Append predicted token
            tgt_tokens = np.concatenate([
                tgt_tokens,
                np.array([[next_token]])
            ], axis=1)
        
        # Decode tokens to text
        translation = tgt_tokenizer.decode(tgt_tokens[0])
        return translation

# Example usage
src_vocab_size = 10000
tgt_vocab_size = 8000
translator = TranslationTransformer(src_vocab_size, tgt_vocab_size)

# Simulate tokenizers
class DummyTokenizer:
    def __init__(self):
        self.token_to_id = {'<START>': 1, '<END>': 2}
    def encode(self, text):
        return [1] + [random.randint(3, 100) for _ in text.split()] + [2]
    def decode(self, tokens):
        return " ".join([str(t) for t in tokens])

src_tokenizer = DummyTokenizer()
tgt_tokenizer = DummyTokenizer()

# Example translation
src_text = "Hello world!"
translation = translator.translate(src_text, src_tokenizer, tgt_tokenizer)
print(f"Source: {src_text}")
print(f"Translation: {translation}")
```

Slide 13: Additional Resources

*   Attention Is All You Need (Original Transformer Paper):
    *   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   BERT: Pre-training of Deep Bidirectional Transformers:
    *   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   The Annotated Transformer (Harvard NLP):
    *   [http://nlp.seas.harvard.edu/annotated-transformer/](http://nlp.seas.harvard.edu/annotated-transformer/)
*   Transformer Implementation Best Practices:
    *   Search "Transformer Implementation Guide" on Google Scholar
*   Practical Transformer Applications:
    *   Search "Recent Advances in Transformer Models" on ArXiv
*   Neural Machine Translation by Jointly Learning to Align and Translate:
    *   [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)

