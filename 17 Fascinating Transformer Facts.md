## 17 Fascinating Transformer Facts
Slide 1: Transformer Architecture Overview

The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that capture long-range dependencies without recurrence. This implementation demonstrates the core components of a basic transformer model including multi-head attention and position-wise feed-forward networks.

```python
import numpy as np
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size)
        x = self.embedding(x) * np.sqrt(self.embedding.embedding_dim)
        output = self.transformer(x)
        return self.fc_out(output)
```

Slide 2: Self-Attention Mechanism Implementation

Self-attention calculates attention weights by comparing query-key pairs and uses these weights to create context-aware representations. This implementation shows the scaled dot-product attention mechanism that forms the foundation of transformer models.

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    # Calculate attention scores
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, value), attention_weights
```

Slide 3: Multi-Head Attention Implementation

Multi-head attention allows the model to attend to different representation subspaces simultaneously. This module splits queries, keys, and values into multiple heads, performs scaled dot-product attention independently, and concatenates the results.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
    def forward(self, query, key, value, mask=None):
        q = self.split_heads(self.W_q(query))
        k = self.split_heads(self.W_k(key))
        v = self.split_heads(self.W_v(value))
        
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        
        output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        return self.W_o(output)
```

Slide 4: Positional Encoding

Positional encodings inject information about token positions into the model since attention mechanisms are inherently permutation-invariant. This implementation uses sinusoidal functions to create unique position embeddings that maintain relative positions.

```python
def positional_encoding(max_seq_length, d_model):
    pos = torch.arange(max_seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = torch.zeros(max_seq_length, d_model)
    pos_encoding[:, 0::2] = torch.sin(pos * div_term)
    pos_encoding[:, 1::2] = torch.cos(pos * div_term)
    
    return pos_encoding
```

Slide 5: Feed-Forward Network

The position-wise feed-forward network applies two linear transformations with a ReLU activation in between. This component adds non-linearity and increases the model's capacity to learn complex patterns.

```python
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

Slide 6: Encoder Layer Implementation

The encoder layer combines multi-head attention with position-wise feed-forward networks, layer normalization, and residual connections. This implementation shows how these components work together to process input sequences effectively.

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))
```

Slide 7: Decoder Layer Implementation

The decoder layer extends the encoder by adding masked self-attention and cross-attention mechanisms. This prevents the decoder from attending to future tokens during training and enables it to utilize encoder outputs effectively.

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        
        # Cross-attention with encoder output
        attn2 = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        return self.norm3(x + self.dropout(ffn_out))
```

Slide 8: Attention Mask Generation

Attention masks are crucial for controlling information flow in transformers. This implementation shows how to create both padding masks for varying sequence lengths and look-ahead masks for autoregressive generation.

```python
def create_masks(src, tgt):
    # Source padding mask
    src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    # Target padding mask
    tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
    
    # Look-ahead mask for decoder
    seq_len = tgt.size(1)
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    look_ahead_mask = look_ahead_mask.bool()
    
    combined_mask = torch.logical_and(tgt_padding_mask, ~look_ahead_mask)
    
    return src_padding_mask, combined_mask
```

Slide 9: Training Loop Implementation

The training process for transformers requires careful attention to learning rate scheduling and loss computation. This implementation demonstrates a complete training loop with teacher forcing and label smoothing.

```python
def train_transformer(model, train_loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt) in enumerate(train_loader):
        src_mask, tgt_mask = create_masks(src, tgt)
        
        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = criterion(output.view(-1, output.size(-1)), 
                        tgt_output.contiguous().view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)
```

Slide 10: Inference and Beam Search

During inference, transformers can use beam search to generate better quality sequences. This implementation shows how to perform beam search with the trained transformer model.

```python
def beam_search(model, src, beam_size=4, max_len=50):
    model.eval()
    src_mask = create_masks(src, None)[0]
    
    # Encode source sequence
    enc_output = model.encoder(src, src_mask)
    
    # Initialize beams with start token
    beams = [(torch.tensor([[model.start_token]]), 0)]
    completed_beams = []
    
    for _ in range(max_len):
        candidates = []
        
        for seq, score in beams:
            if seq[0, -1] == model.end_token:
                completed_beams.append((seq, score))
                continue
                
            # Generate next token probabilities
            tgt_mask = create_masks(None, seq)[1]
            output = model.decoder(seq, enc_output, tgt_mask)
            probs = torch.softmax(output[:, -1], dim=-1)
            
            # Get top-k candidates
            values, indices = probs.topk(beam_size)
            for value, idx in zip(values[0], indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score - torch.log(value).item()
                candidates.append((new_seq, new_score))
        
        # Select top beam_size candidates
        beams = sorted(candidates, key=lambda x: x[1])[:beam_size]
        
    # Return best completed sequence
    completed_beams = sorted(completed_beams, key=lambda x: x[1])
    return completed_beams[0][0] if completed_beams else beams[0][0]
```

Slide 11: Transformer for Machine Translation

This implementation demonstrates a complete machine translation system using transformers. The code includes data preprocessing, tokenization, and evaluation metrics for translation quality assessment.

```python
class TranslationTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt):
        src_embed = self.encoder_embedding(src) * np.sqrt(512)
        tgt_embed = self.decoder_embedding(tgt) * np.sqrt(512)
        
        src_pos = positional_encoding(src.size(1), 512)
        tgt_pos = positional_encoding(tgt.size(1), 512)
        
        src_embed = src_embed + src_pos
        tgt_embed = tgt_embed + tgt_pos
        
        output = self.transformer(src_embed, tgt_embed)
        return self.fc_out(output)

# Example usage and training
def train_translation_model():
    # Initialize tokenizers and vocabularies
    src_vocab_size = 32000
    tgt_vocab_size = 32000
    
    model = TranslationTransformer(src_vocab_size, tgt_vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            src, tgt = batch
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            optimizer.zero_grad()
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, tgt_vocab_size), 
                           tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
```

Slide 12: Attention Visualization

Understanding attention patterns is crucial for interpreting transformer behavior. This implementation provides tools for visualizing attention weights and analyzing model decisions.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, src_tokens, tgt_tokens):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().numpy(),
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='viridis',
                annot=True)
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.title('Attention Weights Visualization')
    
    # Example usage
    def get_attention_weights(model, src, tgt):
        with torch.no_grad():
            attention = model.transformer.encoder.layers[-1].self_attn(
                model.encoder_embedding(src),
                model.encoder_embedding(src),
                model.encoder_embedding(src)
            )[1]
        return attention[0]  # Get weights from first head

# Calculate and visualize attention
src_sentence = ["The", "cat", "sat", "on", "the", "mat"]
tgt_sentence = ["Le", "chat", "s'assit", "sur", "le", "tapis"]
attention_weights = get_attention_weights(model, 
                                       tokenize(src_sentence), 
                                       tokenize(tgt_sentence))
visualize_attention(attention_weights, src_sentence, tgt_sentence)
```

Slide 13: Fine-tuning and Transfer Learning

Implementing transfer learning with pre-trained transformer models enables rapid adaptation to new tasks. This code shows how to fine-tune a pre-trained transformer for specific downstream tasks.

```python
class TransformerForClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.transformer = pretrained_model
        # Freeze transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

def fine_tune_transformer(base_model, train_dataset, num_classes):
    model = TransformerForClassification(base_model, num_classes)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=2e-5)
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataset:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = F.cross_entropy(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
```

Slide 14: Additional Resources

*   "Attention Is All You Need" - Original Transformer Paper [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" [https://arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683)
*   For more resources about transformers and deep learning:
    *   Google AI Blog: [https://ai.googleblog.com](https://ai.googleblog.com)
    *   Papers With Code: [https://paperswithcode.com/method/transformer](https://paperswithcode.com/method/transformer)
    *   Hugging Face Documentation: [https://huggingface.co/docs](https://huggingface.co/docs)

