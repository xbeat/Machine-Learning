## Positional Encodings in Transformer LLMs
Slide 1: Understanding Positional Encodings Fundamentals

Positional encodings form the backbone of modern transformer architectures, enabling models to understand sequential information. They inject position-dependent signals into input embeddings through sophisticated mathematical transformations, preserving word order information during parallel processing.

```python
import numpy as np

def positional_encoding(position, d_model):
    # Create empty encoding matrix
    encoding = np.zeros((position, d_model))
    
    # Calculate positional encodings using sine and cosine
    for pos in range(position):
        for i in range(0, d_model, 2):
            denominator = np.power(10000, 2 * i / d_model)
            encoding[pos, i] = np.sin(pos / denominator)
            encoding[pos, i + 1] = np.cos(pos / denominator)
    
    return encoding

# Example usage
sequence_length = 10
embedding_dim = 512
encodings = positional_encoding(sequence_length, embedding_dim)
print(f"Shape of positional encodings: {encodings.shape}")
print("\nFirst position encoding (partial):")
print(encodings[0, :10])  # Show first 10 values
```

Slide 2: Implementing Absolute Positional Encodings

Absolute positional encodings assign unique position-dependent values to each token in the sequence. This implementation demonstrates how to create learnable position embeddings that can be trained alongside the model parameters.

```python
import torch
import torch.nn as nn

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, embed_dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _ = x.size()
        positions = torch.arange(seq_length, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings

# Example usage
seq_length, batch_size, embed_dim = 16, 4, 128
input_embeddings = torch.randn(batch_size, seq_length, embed_dim)
pos_encoder = AbsolutePositionalEncoding(max_seq_length=100, embed_dim=embed_dim)
output = pos_encoder(input_embeddings)
print(f"Output shape: {output.shape}")
```

Slide 3: Relative Positional Encodings Implementation

Relative positional encodings capture relationships between tokens based on their relative distances. This approach offers better generalization for varying sequence lengths and creates more flexible position-aware representations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_distance=32):
        super().__init__()
        self.max_distance = max_distance
        self.rel_embeddings = nn.Parameter(torch.randn(2 * max_distance + 1, dim))
        
    def forward(self, q, k):
        # q, k shapes: (batch, heads, seq_length, dim)
        seq_length = q.size(2)
        
        # Create relative position matrix
        positions = torch.arange(seq_length).unsqueeze(0) - torch.arange(seq_length).unsqueeze(1)
        positions = positions.clamp(-self.max_distance, self.max_distance) + self.max_distance
        rel_pos_emb = self.rel_embeddings[positions]
        
        # Calculate relative attention scores
        return torch.matmul(q, rel_pos_emb.transpose(-2, -1))

# Example usage
batch_size, heads, seq_length, dim = 2, 8, 20, 64
queries = torch.randn(batch_size, heads, seq_length, dim)
keys = torch.randn(batch_size, heads, seq_length, dim)

rel_pos = RelativePositionalEncoding(dim=dim)
rel_scores = rel_pos(queries, keys)
print(f"Relative attention scores shape: {rel_scores.shape}")
```

Slide 4: Sinusoidal Position Encoding Mathematics

The mathematical foundation of sinusoidal position encodings relies on wavelength variations across dimensions. This implementation demonstrates the core mathematical concepts using numpy, showing how different frequency components create unique position signatures.

```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_position_encoding(max_seq_length, d_model):
    """
    Mathematical implementation showing wavelength progression
    """
    position = np.arange(max_seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    # Demonstrate wavelength variation
    plt.figure(figsize=(12, 4))
    for i in range(4):
        plt.plot(pe[:, i], label=f'dim_{i}')
    plt.legend()
    plt.title('Sinusoidal Position Encoding Patterns')
    plt.show()
    
    return pe

# Generate and visualize
seq_length, d_model = 100, 64
encodings = sinusoidal_position_encoding(seq_length, d_model)
print(f"Position encoding matrix shape: {encodings.shape}")
```

Slide 5: Transformer Position-Aware Self-Attention

Position-aware self-attention integrates positional information directly into the attention mechanism. This implementation shows how positional encodings influence token relationships during the attention computation phase.

```python
import torch
import torch.nn as nn

class PositionAwareSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim))
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # Add positional embeddings
        positions = self.pos_embedding[:, :seq_length, :]
        x = x + positions
        
        # Transform input into Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        
        # Reshape and project
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_length, -1)
        return self.projection(context)

# Example usage
batch_size, seq_length, embed_dim = 8, 32, 256
x = torch.randn(batch_size, seq_length, embed_dim)
attention = PositionAwareSelfAttention(embed_dim, num_heads=8)
output = attention(x)
print(f"Output shape: {output.shape}")
```

Slide 6: Custom Learned Positional Encodings

This implementation showcases learned positional encodings that adapt to the specific characteristics of the training data. The model learns optimal position representations through backpropagation.

```python
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, embed_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create learnable position embeddings
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, max_seq_length, embed_dim)
        )
        
        # Position-dependent scaling factors
        self.scale_factors = nn.Parameter(
            torch.ones(1, max_seq_length, 1)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        seq_length = x.size(1)
        
        # Apply scaled positional embeddings
        positions = self.pos_embeddings[:, :seq_length, :] * self.scale_factors[:, :seq_length, :]
        x = x + positions
        
        # Normalize and apply dropout
        x = self.layer_norm(x)
        return self.dropout(x)

# Example usage
max_length, batch_size, dim = 50, 16, 256
input_tensor = torch.randn(batch_size, max_length, dim)
pos_encoder = LearnedPositionalEncoding(max_length, dim)
encoded = pos_encoder(input_tensor)
print(f"Encoded output shape: {encoded.shape}")

# Demonstrate learning process
optimizer = torch.optim.Adam(pos_encoder.parameters())
criterion = nn.MSELoss()

# Simple training loop example
for _ in range(5):
    encoded = pos_encoder(input_tensor)
    loss = criterion(encoded, torch.randn_like(encoded))  # Dummy target
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Training loss: {loss.item():.4f}")
```

Slide 7: Positional Encoding Visualization Tools

The visualization module provides comprehensive tools for analyzing and understanding positional encoding patterns. This implementation creates detailed visualizations of encoding matrices and attention patterns for debugging and analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncodingVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
        
    def visualize_encodings(self, encodings, title="Positional Encoding Heatmap"):
        plt.figure(figsize=self.fig_size)
        sns.heatmap(encodings, cmap='RdBu', center=0)
        plt.title(title)
        plt.xlabel('Encoding Dimension')
        plt.ylabel('Position')
        plt.show()
        
    def compare_encoding_methods(self, seq_length=50, d_model=128):
        # Generate different types of encodings
        sine_cos = self._generate_sinusoidal(seq_length, d_model)
        learned = self._generate_learned(seq_length, d_model)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        sns.heatmap(sine_cos[:20, :20], ax=ax1, cmap='RdBu', center=0)
        sns.heatmap(learned[:20, :20], ax=ax2, cmap='RdBu', center=0)
        ax1.set_title('Sinusoidal Encodings')
        ax2.set_title('Learned Encodings')
        plt.tight_layout()
        plt.show()
    
    def _generate_sinusoidal(self, seq_length, d_model):
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def _generate_learned(self, seq_length, d_model):
        return np.random.randn(seq_length, d_model)

# Example usage
visualizer = PositionalEncodingVisualizer()

# Generate and visualize encodings
seq_length, d_model = 100, 128
sine_cos_encodings = visualizer._generate_sinusoidal(seq_length, d_model)
visualizer.visualize_encodings(sine_cos_encodings, "Sinusoidal Positional Encodings")

# Compare different encoding methods
visualizer.compare_encoding_methods()
```

Slide 8: Real-world Application: Machine Translation

Implementation of a translation system demonstrating how positional encodings enhance sequence-to-sequence translation tasks. This example shows preprocessing, model implementation, and translation results.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class TranslatorWithPositionalEncoding(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings and positional encodings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length=5000)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def create_mask(self, src, tgt):
        src_mask = torch.ones((src.shape[1], src.shape[1]))
        tgt_mask = torch.triu(torch.ones((tgt.shape[1], tgt.shape[1])), diagonal=1) == 0
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_mask(src, tgt)
        
        # Apply embeddings and positional encodings
        src = self.src_embed(src) * np.sqrt(self.d_model)
        tgt = self.tgt_embed(tgt) * np.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transform sequences
        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        
        return self.output_layer(output)

# Example usage
src_vocab_size, tgt_vocab_size = 5000, 5000
model = TranslatorWithPositionalEncoding(src_vocab_size, tgt_vocab_size)

# Dummy translation data
src_tokens = torch.randint(0, src_vocab_size, (8, 32))
tgt_tokens = torch.randint(0, tgt_vocab_size, (8, 32))

# Forward pass
output = model(src_tokens, tgt_tokens)
print(f"Translation output shape: {output.shape}")
```

Slide 9: Attention Visualization with Position Information

This implementation creates detailed visualizations of attention patterns, showing how positional information influences token relationships in transformer models. The visualization helps understand position-aware attention mechanisms.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    def __init__(self, model_dim=512, num_heads=8):
        self.model_dim = model_dim
        self.num_heads = num_heads
        
    def compute_attention_patterns(self, query, key, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        return torch.softmax(scores, dim=-1)
    
    def plot_attention_heads(self, attention_weights, tokens=None):
        fig = plt.figure(figsize=(20, 10))
        
        for head in range(min(self.num_heads, 4)):  # Plot first 4 heads
            ax = fig.add_subplot(2, 2, head + 1)
            
            # Plot attention weights
            sns.heatmap(attention_weights[0, head].detach().numpy(),
                       xticklabels=tokens if tokens else 'auto',
                       yticklabels=tokens if tokens else 'auto',
                       cmap='viridis',
                       ax=ax)
            
            ax.set_title(f'Head {head + 1} Attention Pattern')
            
        plt.tight_layout()
        plt.show()
        
    def visualize_position_influence(self, seq_length=20):
        # Generate position-aware attention pattern
        positions = torch.arange(seq_length).unsqueeze(1)
        rel_positions = positions - positions.T
        
        # Create position-based attention bias
        position_bias = torch.exp(-torch.abs(rel_positions).float() / 5.0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(position_bias.numpy(), 
                    cmap='RdBu_r',
                    center=0,
                    xticklabels=range(seq_length),
                    yticklabels=range(seq_length))
        plt.title('Position-based Attention Bias')
        plt.show()

# Example usage
visualizer = AttentionVisualizer()

# Generate sample attention patterns
query = torch.randn(1, 8, 20, 64)  # (batch, heads, seq_length, dim)
key = torch.randn(1, 8, 20, 64)
attention_weights = visualizer.compute_attention_patterns(query, key)

# Visualize attention patterns
sample_tokens = [f'Token_{i}' for i in range(20)]
visualizer.plot_attention_heads(attention_weights, sample_tokens)

# Show position influence
visualizer.visualize_position_influence()
```

Slide 10: Position-Aware Text Generation Model

This implementation demonstrates how positional encodings enhance text generation capabilities. The model uses position information to maintain coherence and context awareness during generation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionAwareGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, d_model))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
        
    def forward(self, x, memory=None):
        # Apply embeddings and positional encoding
        seq_len = x.size(1)
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transform and generate
        if memory is None:
            memory = torch.zeros_like(x)
            
        output = self.transformer(x.transpose(0, 1), memory.transpose(0, 1), tgt_mask=mask)
        return self.output_layer(output.transpose(0, 1))
    
    def generate(self, start_tokens, max_length=50, temperature=1.0):
        self.eval()
        current_sequence = start_tokens
        
        with torch.no_grad():
            for _ in range(max_length):
                # Generate next token probabilities
                logits = self.forward(current_sequence)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
                
                # Append to sequence
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
                
                # Check for end of sequence token
                if next_token.item() == 2:  # Assuming 2 is EOS token
                    break
                    
        return current_sequence

# Example usage
vocab_size = 10000
model = PositionAwareGenerator(vocab_size)

# Generate text
start_sequence = torch.tensor([[1, 345, 678]])  # Example start tokens
generated = model.generate(start_sequence)
print(f"Generated sequence shape: {generated.shape}")
```

Slide 11: Performance Analysis and Benchmarking

This implementation provides tools for measuring and comparing the effectiveness of different positional encoding schemes, including metrics for sequence modeling tasks and attention pattern analysis.

```python
import torch
import time
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EncodingBenchmark:
    encoding_time: float
    memory_usage: int
    attention_quality: float
    sequence_coherence: float

class PositionalEncodingBenchmark:
    def __init__(self, max_seq_length: int, d_model: int):
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        
    def benchmark_encoding(self, encoding_fn, num_trials=100):
        total_time = 0
        max_memory = 0
        
        for _ in range(num_trials):
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            start_time = time.time()
            encoded = encoding_fn(self.max_seq_length, self.d_model)
            end_time = time.time()
            
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            total_time += (end_time - start_time)
            max_memory = max(max_memory, end_mem - start_mem)
            
        # Calculate attention quality metric
        attention_quality = self._compute_attention_quality(encoded)
        
        # Calculate sequence coherence
        sequence_coherence = self._measure_sequence_coherence(encoded)
        
        return EncodingBenchmark(
            encoding_time=total_time / num_trials,
            memory_usage=max_memory,
            attention_quality=attention_quality,
            sequence_coherence=sequence_coherence
        )
    
    def _compute_attention_quality(self, encoded_tensor):
        # Compute cosine similarity between positions
        encoded_norm = torch.nn.functional.normalize(encoded_tensor, dim=-1)
        similarity = torch.matmul(encoded_norm, encoded_norm.transpose(-2, -1))
        
        # Calculate average attention quality metric
        diagonal_mask = torch.eye(similarity.size(0))
        off_diagonal = similarity * (1 - diagonal_mask)
        
        return float(off_diagonal.abs().mean())
    
    def _measure_sequence_coherence(self, encoded_tensor):
        # Measure how well positions are distinguished
        positions = torch.arange(encoded_tensor.size(0))
        position_diffs = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # Calculate correlation with position differences
        encoded_flat = encoded_tensor.view(encoded_tensor.size(0), -1)
        encoding_diffs = torch.cdist(encoded_flat, encoded_flat)
        
        correlation = np.corrcoef(
            position_diffs.abs().flatten().numpy(),
            encoding_diffs.flatten().numpy()
        )[0, 1]
        
        return float(correlation)

# Example usage
def run_benchmarks():
    seq_length, d_model = 1024, 512
    benchmark = PositionalEncodingBenchmark(seq_length, d_model)
    
    # Define encoding methods for comparison
    encodings = {
        'sinusoidal': lambda s, d: torch.tensor([
            [pos / np.power(10000, 2 * (j // 2) / d) for j in range(d)]
            for pos in range(s)
        ]),
        'learned': lambda s, d: torch.randn(s, d),
        'relative': lambda s, d: torch.triu(torch.ones(s, s)) * \
                                torch.randn(d).unsqueeze(0).unsqueeze(0)
    }
    
    results = {}
    for name, enc_fn in encodings.items():
        results[name] = benchmark.benchmark_encoding(enc_fn)
        print(f"\nResults for {name} encoding:")
        print(f"Average encoding time: {results[name].encoding_time:.6f} seconds")
        print(f"Memory usage: {results[name].memory_usage} bytes")
        print(f"Attention quality: {results[name].attention_quality:.4f}")
        print(f"Sequence coherence: {results[name].sequence_coherence:.4f}")

# Run benchmarks
run_benchmarks()
```

Slide 12: Dynamic Position Adaptation System

This implementation showcases a dynamic positional encoding system that adapts to varying sequence lengths and content types, demonstrating advanced position-aware processing capabilities.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Learnable components
        self.content_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.position_scale = nn.Parameter(torch.ones(1, 1, d_model))
        
        # Position embedding generators
        self.pos_embedding_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Adaptive components
        self.length_factor = nn.Parameter(torch.ones(1))
        self.content_factor = nn.Parameter(torch.ones(1))
        
    def generate_position_codes(self, seq_length: int):
        position = torch.arange(seq_length, dtype=torch.float32)
        omega = torch.exp(
            torch.arange(0, self.d_model, 2) * 
            -(np.log(10000.0) / self.d_model)
        )
        
        out = torch.zeros(seq_length, self.d_model)
        out[:, 0::2] = torch.sin(position.unsqueeze(1) * omega)
        out[:, 1::2] = torch.cos(position.unsqueeze(1) * omega)
        
        return out
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_length, _ = x.shape
        
        # Generate base positional codes
        pos_codes = self.generate_position_codes(seq_length).to(x.device)
        
        # Apply content-based scaling
        content_importance = torch.sigmoid(
            self.pos_embedding_generator(x) * self.content_factor
        )
        
        # Combine with input taking sequence length into account
        length_scale = torch.sigmoid(seq_length / self.max_seq_length * self.length_factor)
        
        position_embedding = pos_codes.unsqueeze(0) * self.position_scale
        content_embedding = x * self.content_scale
        
        output = content_embedding + position_embedding * content_importance * length_scale
        
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1) == 0, 0)
            
        return output

# Example usage and testing
def test_dynamic_encoder():
    d_model = 256
    encoder = DynamicPositionalEncoder(d_model)
    
    # Test with different sequence lengths
    test_lengths = [10, 50, 100, 500]
    
    for length in test_lengths:
        x = torch.randn(2, length, d_model)
        encoded = encoder(x)
        
        print(f"\nTesting sequence length: {length}")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {encoded.shape}")
        
        # Verify position sensitivity
        pos_correlation = torch.corrcoef(
            encoded[0].flatten(),
            torch.arange(length).repeat_interleave(d_model).float()
        )
        print(f"Position correlation: {pos_correlation[0,1]:.4f}")

# Run tests
test_dynamic_encoder()
```

Slide 13: Advanced Position-Aware Attention Mechanism

This implementation introduces a sophisticated attention mechanism that dynamically adjusts to both local and global positional relationships, demonstrating enhanced context awareness in sequence processing.

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class AdvancedPositionAwareAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 16):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-scale position embeddings
        self.local_pos_embedding = nn.Parameter(
            torch.randn(2 * window_size - 1, self.head_dim)
        )
        self.global_pos_embedding = nn.Parameter(
            torch.randn(1024, self.head_dim)
        )
        
        # Attention projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # Dynamic position-aware components
        self.pos_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.content_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        
    def get_relative_positions(self, seq_length: int) -> torch.Tensor:
        positions = torch.arange(seq_length)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions += self.window_size - 1  # Shift to positive indices
        return relative_positions
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = x.shape
        
        # Generate QKV representations
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2),
            qkv
        )
        
        # Compute attention scores
        content_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add positional bias
        relative_positions = self.get_relative_positions(seq_length).to(x.device)
        local_pos_bias = self.local_pos_embedding[
            relative_positions.clamp(-self.window_size + 1, self.window_size - 1)
            + self.window_size - 1
        ]
        
        # Combine local and global position information
        position_scores = (
            (q.unsqueeze(-2) @ local_pos_bias.transpose(-2, -1))
            .squeeze(-2)
            * self.pos_scale
        )
        
        # Final attention scores
        attention_scores = (
            content_scores * self.content_scale
            + position_scores
        )
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2) == 0,
                float('-inf')
            )
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        output = (attention_probs @ v).transpose(1, 2).reshape(
            batch_size, seq_length, self.dim
        )
        
        return self.proj(output), attention_probs

# Example usage and testing
def test_advanced_attention():
    batch_size = 4
    seq_length = 32
    dim = 256
    
    attention = AdvancedPositionAwareAttention(dim)
    x = torch.randn(batch_size, seq_length, dim)
    mask = torch.ones(batch_size, seq_length)
    
    output, attention_weights = attention(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Analyze position sensitivity
    avg_attention_by_distance = []
    for dist in range(seq_length):
        diag_indices = torch.arange(seq_length - dist)
        attention_at_distance = attention_weights[0, 0, diag_indices, diag_indices + dist]
        avg_attention_by_distance.append(attention_at_distance.mean().item())
    
    print("\nAttention decay with distance:")
    for dist, avg_attn in enumerate(avg_attention_by_distance[:5]):
        print(f"Distance {dist}: {avg_attn:.4f}")

# Run tests
test_advanced_attention()
```

Slide 14: Additional Resources

1.  "Attention Is All You Need" - Original Transformer Paper [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "On Position Embeddings in BERT" [https://arxiv.org/abs/2010.15099](https://arxiv.org/abs/2010.15099)
3.  "RoFormer: Enhanced Transformer with Rotary Position Embedding" [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
4.  "Position Information in Transformers: An Overview" [https://arxiv.org/abs/2102.11090](https://arxiv.org/abs/2102.11090)
5.  "Realformer: Transformer Likes Residual Attention" [https://arxiv.org/abs/2012.11747](https://arxiv.org/abs/2012.11747)

