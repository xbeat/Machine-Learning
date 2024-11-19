## Simplifying Transformer Architecture for Beginners
Slide 1: Transformer Architecture Overview

The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms and parallel processing capabilities. This implementation demonstrates the fundamental building blocks of a transformer model using NumPy for better understanding of the underlying mathematics.

```python
import numpy as np

class TransformerBlock:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize weights
        self.w_q = np.random.randn(embed_dim, embed_dim)
        self.w_k = np.random.randn(embed_dim, embed_dim)
        self.w_v = np.random.randn(embed_dim, embed_dim)
        
    def split_heads(self, x):
        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def attention(self, q, k, v, mask=None):
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = np.softmax(scores, axis=-1)
        return np.matmul(weights, v)

# Example usage
embed_dim, num_heads = 512, 8
transformer = TransformerBlock(embed_dim, num_heads)
```

Slide 2: Input Embedding and Positional Encoding

The first step in transformer processing involves converting input tokens into continuous vectors and adding positional information. This implementation shows how to create embeddings and incorporate sinusoidal positional encodings.

```python
class EmbeddingLayer:
    def __init__(self, vocab_size, embed_dim, max_seq_length):
        self.embed_dim = embed_dim
        self.embedding = np.random.randn(vocab_size, embed_dim)
        self.pos_encoding = self.create_positional_encoding(max_seq_length, embed_dim)
    
    def create_positional_encoding(self, max_seq_length, d_model):
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((max_seq_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x):
        seq_length = x.shape[1]
        embedded = self.embedding[x]
        return embedded + self.pos_encoding[:seq_length]

# Example usage
vocab_size, embed_dim, max_seq_length = 5000, 512, 100
embedding_layer = EmbeddingLayer(vocab_size, embed_dim, max_seq_length)
```

Slide 3: Multi-Head Attention Implementation

Multi-head attention allows the model to jointly attend to information from different representation subspaces. This implementation shows the complete multi-head attention mechanism with parallel processing of attention heads.

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize weight matrices
        self.w_q = np.random.randn(embed_dim, embed_dim)
        self.w_k = np.random.randn(embed_dim, embed_dim)
        self.w_v = np.random.randn(embed_dim, embed_dim)
        self.w_o = np.random.randn(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        Q = np.dot(query, self.w_q).reshape(batch_size, -1, self.num_heads, self.head_dim)
        K = np.dot(key, self.w_k).reshape(batch_size, -1, self.num_heads, self.head_dim)
        V = np.dot(value, self.w_v).reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = np.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = np.matmul(attention, V)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
        
        return np.dot(out, self.w_o)
```

Slide 4: Feed-Forward Neural Network

The feed-forward network in a transformer consists of two linear transformations with a ReLU activation in between. This component processes each position separately and identically, adding non-linearity to the model.

```python
class FeedForward:
    def __init__(self, embed_dim, ff_dim):
        self.w1 = np.random.randn(embed_dim, ff_dim)
        self.w2 = np.random.randn(ff_dim, embed_dim)
        self.b1 = np.zeros(ff_dim)
        self.b2 = np.zeros(embed_dim)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # First linear transformation
        hidden = np.dot(x, self.w1) + self.b1
        # ReLU activation
        hidden = self.relu(hidden)
        # Second linear transformation
        output = np.dot(hidden, self.w2) + self.b2
        return output

# Example usage
embed_dim, ff_dim = 512, 2048
ff_layer = FeedForward(embed_dim, ff_dim)
```

Slide 5: Layer Normalization

Layer normalization is crucial for stable training in transformers, normalizing the inputs across the features. This implementation shows both the mathematical computation and practical application with proper numerical stability considerations.

```python
class LayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        
    def forward(self, x):
        # Calculate mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example with numerical stability demonstration
class StableLayerNorm:
    def __init__(self, embed_dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
    
    def forward(self, x):
        max_val = np.max(np.abs(x), axis=-1, keepdims=True)
        x_scaled = x / (max_val + self.eps)
        
        mean = np.mean(x_scaled, axis=-1, keepdims=True)
        var = np.var(x_scaled, axis=-1, keepdims=True)
        
        x_norm = (x_scaled - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

Slide 6: Self-Attention Mechanism Details

The self-attention mechanism computes attention scores between all pairs of positions in the input sequence. This implementation demonstrates the detailed mathematics of scaled dot-product attention with masking support.

```python
class ScaledDotProductAttention:
    def __init__(self, dropout_rate=0.1):
        self.dropout_rate = dropout_rate
        
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: Query, Key, Value matrices
        mask: Optional mask tensor
        """
        # Compute attention scores
        d_k = K.shape[-1]
        scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Compute attention weights
        attention_weights = self.softmax_with_temperature(scores, temperature=1.0)
        
        # Apply dropout (simplified version)
        if self.dropout_rate > 0:
            dropout_mask = np.random.binomial(1, 1-self.dropout_rate, attention_weights.shape)
            attention_weights *= dropout_mask / (1 - self.dropout_rate)
        
        # Compute output
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def softmax_with_temperature(self, x, temperature=1.0):
        """Numerically stable softmax implementation"""
        exp_x = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temperature)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

Slide 7: Token and Position Embeddings

A detailed implementation of both token embeddings and positional encodings, showing how to combine them effectively while maintaining the proper scale of the embeddings.

```python
class TransformerEmbeddings:
    def __init__(self, vocab_size, embed_dim, max_seq_length):
        self.token_embedding = np.random.randn(vocab_size, embed_dim) * 0.02
        self.position_embedding = self.create_sinusoidal_embeddings(
            max_seq_length, embed_dim)
        self.embed_dim = embed_dim
        self.layer_norm = LayerNorm(embed_dim)
        
    def create_sinusoidal_embeddings(self, max_seq_length, d_model):
        # Create position indices
        position = np.arange(max_seq_length)[:, np.newaxis]
        
        # Create dimension indices
        div_term = np.exp(np.arange(0, d_model, 2) * 
                         (-np.log(10000.0) / d_model))
        
        # Calculate embeddings
        pe = np.zeros((max_seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, input_ids):
        seq_length = input_ids.shape[1]
        
        # Get token embeddings
        embeddings = self.token_embedding[input_ids]
        
        # Add position embeddings
        embeddings = embeddings + self.position_embedding[:seq_length]
        
        # Scale embeddings
        embeddings = embeddings * np.sqrt(self.embed_dim)
        
        # Apply layer normalization
        embeddings = self.layer_norm.forward(embeddings)
        
        return embeddings

# Example usage
vocab_size, embed_dim, max_seq_length = 30000, 512, 512
embedding_layer = TransformerEmbeddings(vocab_size, embed_dim, max_seq_length)
```

Slide 8: Encoder Block Implementation

A complete encoder block implementation combining multi-head attention, feed-forward network, and layer normalization with residual connections. This represents a single layer of the encoder stack in a transformer.

```python
class EncoderBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff_network = FeedForward(embed_dim, ff_dim)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.dropout_rate = dropout_rate
        
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1 - self.dropout_rate)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attention_output = self.attention.forward(x, x, x, mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1.forward(x + attention_output)
        
        # Feed-forward network with residual connection
        ff_output = self.ff_network.forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2.forward(x + ff_output)
        
        return x

# Example usage
embed_dim, num_heads, ff_dim = 512, 8, 2048
encoder = EncoderBlock(embed_dim, num_heads, ff_dim)
```

Slide 9: Decoder Block Implementation

The decoder block extends the encoder with masked self-attention and cross-attention mechanisms. This implementation shows how the decoder processes target sequences while attending to encoder outputs.

```python
class DecoderBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff_network = FeedForward(embed_dim, ff_dim)
        
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.norm3 = LayerNorm(embed_dim)
        
        self.dropout_rate = dropout_rate
    
    def create_causal_mask(self, size):
        """Create causal mask for decoder self-attention"""
        mask = np.triu(np.ones((size, size)), k=1)
        return mask == 0
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        if tgt_mask is None:
            tgt_mask = self.create_causal_mask(x.shape[1])
            
        self_att_output = self.self_attention.forward(x, x, x, tgt_mask)
        self_att_output = self.dropout(self_att_output)
        x = self.norm1.forward(x + self_att_output)
        
        # Cross-attention to encoder output
        cross_att_output = self.cross_attention.forward(
            x, encoder_output, encoder_output, src_mask)
        cross_att_output = self.dropout(cross_att_output)
        x = self.norm2.forward(x + cross_att_output)
        
        # Feed-forward network
        ff_output = self.ff_network.forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm3.forward(x + ff_output)
        
        return x
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1-self.dropout_rate, x.shape)
        return x * mask / (1 - self.dropout_rate)
```

Slide 10: Complete Transformer Model

A full implementation of the transformer model combining all previous components into a complete architecture capable of sequence-to-sequence tasks.

```python
class Transformer:
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
                 num_encoder_layers, num_decoder_layers, max_seq_length):
        # Embedding layers
        self.encoder_embed = TransformerEmbeddings(
            vocab_size, embed_dim, max_seq_length)
        self.decoder_embed = TransformerEmbeddings(
            vocab_size, embed_dim, max_seq_length)
        
        # Encoder and Decoder stacks
        self.encoder_layers = [
            EncoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_encoder_layers)
        ]
        
        self.decoder_layers = [
            DecoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_decoder_layers)
        ]
        
        # Output projection
        self.final_layer = np.random.randn(embed_dim, vocab_size) * 0.02
        
    def encode(self, src_tokens, src_mask=None):
        x = self.encoder_embed.forward(src_tokens)
        for encoder in self.encoder_layers:
            x = encoder.forward(x, src_mask)
        return x
    
    def decode(self, tgt_tokens, encoder_output, src_mask=None, tgt_mask=None):
        x = self.decoder_embed.forward(tgt_tokens)
        for decoder in self.decoder_layers:
            x = decoder.forward(x, encoder_output, src_mask, tgt_mask)
        return x
    
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src_tokens, src_mask)
        decoder_output = self.decode(
            tgt_tokens, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = np.dot(decoder_output, self.final_layer)
        return logits
```

Slide 11: Training Implementation

Implementation of the training loop with loss calculation, gradient accumulation, and learning rate scheduling for optimal transformer model training performance.

```python
class TransformerTrainer:
    def __init__(self, model, learning_rate=0.0001, warmup_steps=4000):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def calculate_loss(self, logits, targets, pad_idx=0):
        """Cross entropy loss with label smoothing"""
        vocab_size = logits.shape[-1]
        smoothing = 0.1
        
        # Create smoothed targets
        smooth_targets = np.zeros_like(logits)
        smooth_targets[targets != pad_idx] = smoothing / (vocab_size - 1)
        smooth_targets[np.arange(targets.shape[0]), targets] = 1.0 - smoothing
        
        # Calculate cross entropy
        log_probs = np.log_softmax(logits, axis=-1)
        loss = -np.sum(smooth_targets * log_probs) / np.sum(targets != pad_idx)
        
        return loss
    
    def get_learning_rate(self):
        """Implement learning rate scheduler with warmup"""
        step = self.step + 1
        lr = self.learning_rate * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )
        return lr
    
    def train_step(self, src_tokens, tgt_tokens):
        # Forward pass
        logits = self.model.forward(src_tokens, tgt_tokens[:, :-1])
        loss = self.calculate_loss(logits, tgt_tokens[:, 1:])
        
        # Update learning rate
        lr = self.get_learning_rate()
        self.step += 1
        
        return loss, logits

# Example usage
batch_size, seq_length = 32, 100
src_tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))
tgt_tokens = np.random.randint(0, vocab_size, (batch_size, seq_length))

trainer = TransformerTrainer(model)
loss, logits = trainer.train_step(src_tokens, tgt_tokens)
```

Slide 12: Inference and Beam Search

Implementation of beam search decoding for generating high-quality translations during inference, with support for diverse beam groups and length normalization.

```python
class TransformerInference:
    def __init__(self, model, max_length=100, beam_size=5):
        self.model = model
        self.max_length = max_length
        self.beam_size = beam_size
        
    def beam_search(self, src_tokens, start_token=1, end_token=2):
        # Encode source sequence
        encoder_output = self.model.encode(src_tokens)
        
        # Initialize beam
        beam = [(0., [start_token])]
        finished_beams = []
        
        for step in range(self.max_length):
            candidates = []
            
            for score, sequence in beam:
                if sequence[-1] == end_token:
                    finished_beams.append((score, sequence))
                    continue
                    
                # Decode next token probabilities
                tgt_tokens = np.array([sequence])
                decoder_output = self.model.decode(
                    tgt_tokens, encoder_output)
                logits = np.dot(decoder_output[:, -1], self.model.final_layer)
                probs = np.exp(np.log_softmax(logits, axis=-1))
                
                # Add top-k candidates to beam
                top_k = np.argpartition(probs[0], -self.beam_size)[-self.beam_size:]
                for token in top_k:
                    new_score = score + np.log(probs[0][token])
                    new_sequence = sequence + [token]
                    candidates.append((new_score, new_sequence))
            
            # Select top-k candidates
            candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
            beam = candidates[:self.beam_size]
            
            # Early stopping if all beams finished
            if all(seq[-1] == end_token for _, seq in beam):
                finished_beams.extend(beam)
                break
        
        # Length normalization
        finished_beams = [(score / len(seq) ** 0.6, seq) 
                         for score, seq in finished_beams]
        best_sequence = max(finished_beams, key=lambda x: x[0])[1]
        
        return best_sequence[1:-1]  # Remove start and end tokens

# Example usage
inference = TransformerInference(model)
translation = inference.beam_search(src_tokens)
```

Slide 13: Real-world Example - Machine Translation

This implementation shows a complete translation system using the transformer for English to French translation, including preprocessing and BLEU score calculation.

```python
class TranslationSystem:
    def __init__(self, model, src_tokenizer, tgt_tokenizer):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.inference = TransformerInference(model)
        
    def preprocess_text(self, text, tokenizer):
        # Basic preprocessing
        text = text.lower().strip()
        # Tokenization
        tokens = tokenizer.encode(text)
        return np.array([tokens])
    
    def calculate_bleu(self, reference, hypothesis):
        """Simple BLEU score implementation"""
        def get_ngrams(tokens, n):
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        
        # Calculate n-gram precision for n=1,2,3,4
        precisions = []
        for n in range(1, 5):
            ref_ngrams = get_ngrams(reference, n)
            hyp_ngrams = get_ngrams(hypothesis, n)
            matches = len(ref_ngrams.intersection(hyp_ngrams))
            if len(hyp_ngrams) > 0:
                precisions.append(matches / len(hyp_ngrams))
            else:
                precisions.append(0)
        
        # Calculate geometric mean
        if min(precisions) > 0:
            bleu = np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0
        
        # Apply brevity penalty
        bp = min(1, np.exp(1 - len(reference)/len(hypothesis)))
        return bp * bleu
    
    def translate(self, text):
        # Preprocess input
        src_tokens = self.preprocess_text(text, self.src_tokenizer)
        
        # Generate translation
        output_tokens = self.inference.beam_search(src_tokens)
        
        # Decode output
        translation = self.tgt_tokenizer.decode(output_tokens)
        return translation

# Example usage
text = "The transformer architecture has revolutionized natural language processing."
translator = TranslationSystem(model, src_tokenizer, tgt_tokenizer)
translation = translator.translate(text)
```

Slide 14: Real-world Example - Text Summarization

Implementation of an abstractive text summarization system using the transformer model, with support for length control and topic focus.

```python
class SummarizationSystem:
    def __init__(self, model, tokenizer, max_length=150):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inference = TransformerInference(
            model, max_length=max_length, beam_size=4)
        
    def preprocess_document(self, document):
        # Split into sentences
        sentences = document.split('.')
        # Remove empty sentences and normalize
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Tokenize
        tokens = []
        for sent in sentences:
            tokens.extend(self.tokenizer.encode(sent))
            tokens.append(self.tokenizer.sep_token_id)
        
        return np.array([tokens])
    
    def control_length(self, logits, desired_length, current_length):
        """Adjust token probabilities based on desired summary length"""
        if current_length >= desired_length:
            # Increase probability of end token
            logits[self.tokenizer.eos_token_id] *= 2.0
        return logits
    
    def summarize(self, document, desired_length=None):
        if desired_length is None:
            desired_length = min(len(document.split()) // 3, self.max_length)
            
        # Preprocess document
        input_tokens = self.preprocess_document(document)
        
        # Generate summary with length control
        summary_tokens = self.inference.beam_search(
            input_tokens,
            length_callback=lambda logits, length: 
                self.control_length(logits, desired_length, length)
        )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_tokens)
        return summary

# Example usage
document = """
The transformer architecture, introduced in the paper 'Attention is All You Need',
has fundamentally changed how we approach sequence processing tasks in machine
learning. Its self-attention mechanism allows for parallel processing of input
sequences and captures long-range dependencies more effectively than previous
architectures. This has led to state-of-the-art results across various natural
language processing tasks.
"""

summarizer = SummarizationSystem(model, tokenizer)
summary = summarizer.summarize(document)
```

Slide 15: Additional Resources

*   "Attention Is All You Need" - Original Transformer paper
    *   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    *   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Language Models are Few-Shot Learners" (GPT-3)
    *   [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
*   "Scaling Laws for Neural Language Models"
    *   [https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
*   "Training Language Models to Follow Instructions with Human Feedback"
    *   [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

