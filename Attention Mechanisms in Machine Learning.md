## Attention Mechanisms in Machine Learning
Slide 1: Attention Mechanism Fundamentals

The attention mechanism allows neural networks to dynamically focus on relevant parts of input sequences by assigning importance weights to different elements, enabling more effective processing of sequential data compared to traditional RNN approaches.

```python
import numpy as np

def attention_score(query, key):
    # Calculate attention scores between query and key vectors
    scores = np.dot(query, key.T)
    # Apply softmax to get probability distribution
    attention_weights = np.exp(scores) / np.sum(np.exp(scores))
    return attention_weights

# Example usage
query = np.array([0.2, 0.5, 0.3])
key = np.array([[0.1, 0.4, 0.5],
                [0.2, 0.3, 0.5],
                [0.3, 0.2, 0.5]])

weights = attention_score(query, key)
print("Attention weights:", weights)
```

Slide 2: Self-Attention Implementation

Self-attention computes relationships between all positions in a sequence by using the same input as queries, keys, and values, enabling the model to capture complex dependencies within the data.

```python
import numpy as np

def self_attention(x, d_k):
    # x: input sequence [seq_len, d_model]
    # d_k: dimension of key vectors
    
    # Linear transformations for Q, K, V
    W_q = np.random.randn(x.shape[1], d_k)
    W_k = np.random.randn(x.shape[1], d_k)
    W_v = np.random.randn(x.shape[1], d_k)
    
    Q = np.dot(x, W_q)
    K = np.dot(x, W_k)
    V = np.dot(x, W_v)
    
    # Scaled dot-product attention
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Apply attention to values
    output = np.dot(attention_weights, V)
    return output, attention_weights

# Example usage
seq_len, d_model, d_k = 4, 8, 16
x = np.random.randn(seq_len, d_model)
output, weights = self_attention(x, d_k)
```

Slide 3: Multi-Head Attention Theory

Multi-head attention extends single-head attention by allowing the model to attend to information from different representation subspaces, creating multiple sets of query, key, and value transformations that operate in parallel.

```python
# Mathematical representation in code block
"""
$$
\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,...,head_h)W^O \\
\text{where }head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
"""
```

Slide 4: Multi-Head Attention Implementation

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(num_heads, d_model, self.d_k)
        self.W_k = np.random.randn(num_heads, d_model, self.d_k)
        self.W_v = np.random.randn(num_heads, d_model, self.d_k)
        self.W_o = np.random.randn(num_heads * self.d_k, d_model)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Transform input for each head
        Q = np.stack([np.dot(x, self.W_q[i]) for i in range(self.num_heads)])
        K = np.stack([np.dot(x, self.W_k[i]) for i in range(self.num_heads)])
        V = np.stack([np.dot(x, self.W_v[i]) for i in range(self.num_heads)])
        
        # Compute attention scores
        scores = np.matmul(Q, np.transpose(K, [0, 2, 1])) / np.sqrt(self.d_k)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Apply attention to values
        head_outputs = np.matmul(attention_weights, V)
        
        # Concatenate and project
        concat_output = np.concatenate(head_outputs, axis=-1)
        final_output = np.dot(concat_output, self.W_o)
        
        return final_output, attention_weights

# Example usage
batch_size, seq_len, d_model = 2, 4, 64
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)
x = np.random.randn(batch_size, seq_len, d_model)
output, weights = mha.forward(x)
```

Slide 5: Positional Encoding

Positional encoding adds position-dependent patterns to input embeddings, allowing attention mechanisms to understand sequence order since they lack inherent sequential processing capabilities.

```python
def positional_encoding(max_seq_len, d_model):
    pos_enc = np.zeros((max_seq_len, d_model))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sine to even indices
    pos_enc[:, 0::2] = np.sin(positions * div_term)
    # Apply cosine to odd indices
    pos_enc[:, 1::2] = np.cos(positions * div_term)
    
    return pos_enc

# Example usage
max_seq_len, d_model = 100, 512
pos_encoding = positional_encoding(max_seq_len, d_model)
```

Slide 6: Attention Layer Normalization

Layer normalization is crucial in attention mechanisms to stabilize the learning process by normalizing the activations across features, helping to prevent internal covariate shift and enabling faster training.

```python
import numpy as np

class LayerNorm:
    def __init__(self, features, eps=1e-6):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps
        
    def forward(self, x):
        # Calculate mean and variance along last axis
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize and scale
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta

# Example usage
batch_size, seq_len, features = 32, 10, 512
x = np.random.randn(batch_size, seq_len, features)
layer_norm = LayerNorm(features)
normalized_output = layer_norm.forward(x)
```

Slide 7: Feed-Forward Network in Transformer

The position-wise feed-forward network applies two linear transformations with a ReLU activation, processing each position independently and adding non-linearity to the model's representation capacity.

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff)
        self.w2 = np.random.randn(d_ff, d_model)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # First linear transformation
        hidden = self.relu(np.dot(x, self.w1) + self.b1)
        # Second linear transformation
        output = np.dot(hidden, self.w2) + self.b2
        return output

# Example usage
d_model, d_ff = 512, 2048
ff_layer = FeedForward(d_model, d_ff)
x = np.random.randn(32, 10, d_model)
output = ff_layer.forward(x)
```

Slide 8: Masked Attention Implementation

Masked attention prevents positions from attending to subsequent positions, crucial for training sequence-to-sequence models and maintaining causality in language generation tasks.

```python
def masked_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        # Apply mask by setting masked positions to large negative value
        scores = np.where(mask == 0, -1e9, scores)
    
    # Apply softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    output = np.matmul(attention_weights, value)
    return output, attention_weights

# Create causal mask for sequence length 4
seq_len = 4
mask = np.tril(np.ones((seq_len, seq_len)))

# Example usage
query = np.random.randn(1, seq_len, 64)
key = np.random.randn(1, seq_len, 64)
value = np.random.randn(1, seq_len, 64)
output, weights = masked_attention(query, key, value, mask)
```

Slide 9: Embedding Layer with Position Encoding

This implementation combines token embeddings with positional encodings to create input representations that capture both semantic meaning and sequential position information.

```python
class EmbeddingLayer:
    def __init__(self, vocab_size, d_model, max_seq_len):
        self.token_embedding = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.position_encoding = self._create_position_encoding(max_seq_len, d_model)
    
    def _create_position_encoding(self, max_seq_len, d_model):
        pos_enc = np.zeros((max_seq_len, d_model))
        positions = np.arange(max_seq_len)[:, np.newaxis]
        angles = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc[:, 0::2] = np.sin(positions * angles)
        pos_enc[:, 1::2] = np.cos(positions * angles)
        return pos_enc
    
    def forward(self, x):
        # Get sequence length from input
        seq_len = x.shape[1]
        # Get token embeddings
        embeddings = self.token_embedding[x]
        # Add position encoding
        return embeddings + self.position_encoding[:seq_len]

# Example usage
vocab_size, d_model, max_seq_len = 5000, 512, 100
embedding_layer = EmbeddingLayer(vocab_size, d_model, max_seq_len)
input_ids = np.random.randint(0, vocab_size, (32, 50))
embedded_output = embedding_layer.forward(input_ids)
```

Slide 10: Attention Dropout Implementation

Dropout in attention mechanisms helps prevent overfitting by randomly masking attention weights during training, improving model generalization and robustness to noise.

```python
import numpy as np

def attention_with_dropout(query, key, value, dropout_rate=0.1, training=True):
    d_k = query.shape[-1]
    
    # Compute attention scores
    scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    if training:
        # Create dropout mask
        dropout_mask = np.random.binomial(1, 1-dropout_rate, attention_weights.shape)
        # Apply dropout and scale
        attention_weights = (attention_weights * dropout_mask) / (1 - dropout_rate)
    
    output = np.matmul(attention_weights, value)
    return output, attention_weights

# Example usage
batch_size, seq_len, d_k = 2, 4, 64
query = np.random.randn(batch_size, seq_len, d_k)
key = np.random.randn(batch_size, seq_len, d_k)
value = np.random.randn(batch_size, seq_len, d_k)

output_train, weights_train = attention_with_dropout(query, key, value, training=True)
output_test, weights_test = attention_with_dropout(query, key, value, training=False)
```

Slide 11: Real-world Application: Text Classification

Implementation of attention-based text classification model showcasing practical usage in sentiment analysis tasks with preprocessing and evaluation metrics.

```python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class TextClassifier:
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        self.embedding = np.random.randn(vocab_size, embed_dim) / np.sqrt(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.classifier = np.random.randn(embed_dim, num_classes)
        
    def forward(self, x):
        # Embed tokens
        embedded = self.embedding[x]
        
        # Apply attention
        attended, _ = self.attention.forward(embedded)
        
        # Global average pooling
        pooled = np.mean(attended, axis=1)
        
        # Classification layer
        logits = np.dot(pooled, self.classifier)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)

# Example usage with dummy data
vocab_size, embed_dim, num_heads, num_classes = 5000, 128, 4, 2
classifier = TextClassifier(vocab_size, embed_dim, num_heads, num_classes)

# Simulate input data
x_train = np.random.randint(0, vocab_size, (100, 50))  # 100 sequences of length 50
y_train = np.random.randint(0, num_classes, 100)

# Make predictions
predictions = classifier.predict(x_train)
accuracy = accuracy_score(y_train, predictions)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_train, predictions))
```

Slide 12: Real-world Application: Machine Translation

Comprehensive implementation of an attention-based translation system demonstrating sequence-to-sequence modeling with beam search decoding.

```python
class TranslationModel:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads):
        self.encoder = Encoder(src_vocab_size, d_model, num_heads)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads)
        self.output_layer = np.random.randn(d_model, tgt_vocab_size)
        
    def beam_search(self, src_tokens, beam_size=4, max_len=50):
        # Encode source sequence
        encoder_output = self.encoder.forward(src_tokens)
        
        # Initialize beam
        beams = [([], 0.0)]  # (sequence, score)
        
        for _ in range(max_len):
            candidates = []
            for sequence, score in beams:
                if sequence and sequence[-1] == self.eos_token:
                    candidates.append((sequence, score))
                    continue
                    
                # Decode next token probabilities
                decoder_output = self.decoder.forward(sequence, encoder_output)
                logits = np.dot(decoder_output[-1], self.output_layer)
                probs = np.exp(logits) / np.sum(np.exp(logits))
                
                # Get top-k candidates
                top_k = np.argsort(probs)[-beam_size:]
                for token in top_k:
                    new_sequence = sequence + [token]
                    new_score = score + np.log(probs[token])
                    candidates.append((new_sequence, new_score))
            
            # Select top-k beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # Check if all beams ended
            if all(b[0][-1] == self.eos_token for b in beams):
                break
                
        return beams[0][0]  # Return best sequence

# Example preprocessing function
def preprocess_text(text, vocab):
    tokens = text.lower().split()
    return [vocab.get(token, vocab['<unk>']) for token in tokens]
```

Slide 13: Results Analysis for Translation Model

Detailed evaluation metrics and performance analysis of the attention-based translation system, including BLEU scores and attention visualization.

```python
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

def evaluate_translation_model(model, test_data, src_vocab, tgt_vocab):
    bleu_scores = []
    attention_maps = []
    
    for src_text, tgt_text in test_data:
        # Preprocess source and target
        src_tokens = preprocess_text(src_text, src_vocab)
        reference = preprocess_text(tgt_text, tgt_vocab)
        
        # Generate translation
        predicted_tokens = model.beam_search(src_tokens)
        predicted_text = [tgt_vocab.inverse[idx] for idx in predicted_tokens]
        
        # Calculate BLEU score
        bleu = sentence_bleu([reference], predicted_tokens)
        bleu_scores.append(bleu)
        
        # Store attention weights for visualization
        _, attention_weights = model.get_attention_weights(src_tokens)
        attention_maps.append(attention_weights)
    
    # Plot attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_maps[0], cmap='viridis')
    plt.colorbar()
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title('Attention Weights Visualization')
    
    print(f"Average BLEU Score: {np.mean(bleu_scores):.4f}")
    return np.mean(bleu_scores), attention_maps

# Example test results
test_results = {
    'BLEU Score': 0.342,
    'Translation Examples': [
        ('Hello world', 'Bonjour le monde'),
        ('How are you?', 'Comment allez-vous?')
    ],
    'Attention Statistics': {
        'Mean Weight': 0.125,
        'Max Weight': 0.876,
        'Min Weight': 0.001
    }
}
```

Slide 14: Optimization and Training Strategy

Advanced training techniques for attention-based models, including learning rate scheduling, gradient clipping, and warmup strategies.

```python
class AttentionOptimizer:
    def __init__(self, model_size, warmup_steps=4000):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def get_learning_rate(self):
        self.step += 1
        return self.model_size ** (-0.5) * min(
            self.step ** (-0.5),
            self.step * self.warmup_steps ** (-1.5)
        )
        
    def clip_gradients(self, gradients, max_norm=1.0):
        total_norm = np.sqrt(sum(np.sum(g * g) for g in gradients))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [g * clip_coef for g in gradients]
        return gradients

def train_step(model, optimizer, batch, clip_value=1.0):
    # Forward pass
    with torch.autograd.detect_anomaly():
        outputs = model(batch.src, batch.tgt)
        loss = criterion(outputs.view(-1, outputs.size(-1)), 
                        batch.tgt_y.view(-1))
        
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    # Update with warmup
    lr = optimizer.get_learning_rate()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()

# Training configuration
config = {
    'batch_size': 32,
    'epochs': 100,
    'warmup_steps': 4000,
    'max_grad_norm': 1.0,
    'learning_rate': 0.0001
}
```

Slide 15: Additional Resources

*   "Attention Is All You Need" - Original Transformer paper [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "Deep Residual Learning for Self-Attention" [https://arxiv.org/abs/2006.04704](https://arxiv.org/abs/2006.04704)
*   "BERT: Pre-training of Deep Bidirectional Transformers" [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
*   For more resources and implementation details:
    *   Google AI Blog: [https://ai.googleblog.com](https://ai.googleblog.com)
    *   Papers With Code: [https://paperswithcode.com/method/attention](https://paperswithcode.com/method/attention)
    *   Hugging Face Documentation: [https://huggingface.co/docs](https://huggingface.co/docs)

