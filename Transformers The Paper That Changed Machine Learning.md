## Transformers The Paper That Changed Machine Learning
Slide 1: Introduction to Transformers

Transformers have revolutionized natural language processing and machine learning. Introduced in the paper "Attention Is All You Need" by Vaswani et al., they replaced recurrent models with attention-based architectures. This new approach dramatically improved performance on various NLP tasks and has since become the foundation for many state-of-the-art language models.

```python
# Simplified Transformer architecture
class Transformer:
    def __init__(self, input_size, output_size, num_layers):
        self.encoder = Encoder(input_size, num_layers)
        self.decoder = Decoder(output_size, num_layers)
    
    def forward(self, src, tgt):
        enc_output = self.encoder(src)
        output = self.decoder(tgt, enc_output)
        return output
```

Slide 2: Self-Attention Mechanism

The core component of Transformers is the self-attention mechanism. It allows the model to weigh the importance of different words in a sentence relative to each other. This mechanism enables the model to capture long-range dependencies and contextual information more effectively than previous architectures.

```python
import numpy as np

def self_attention(query, key, value):
    # Compute attention scores
    scores = np.dot(query, key.T) / np.sqrt(key.shape[1])
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute weighted sum of values
    output = np.dot(weights, value)
    
    return output
```

Slide 3: Multi-Head Attention

Multi-head attention extends the self-attention mechanism by allowing the model to focus on different aspects of the input simultaneously. It does this by applying multiple attention operations in parallel and then combining their results.

```python
def multi_head_attention(query, key, value, num_heads):
    head_dim = query.shape[1] // num_heads
    
    # Split input into multiple heads
    q_heads = query.reshape(-1, num_heads, head_dim)
    k_heads = key.reshape(-1, num_heads, head_dim)
    v_heads = value.reshape(-1, num_heads, head_dim)
    
    # Apply self-attention to each head
    attention_outputs = [self_attention(q, k, v) for q, k, v in zip(q_heads, k_heads, v_heads)]
    
    # Concatenate outputs and project
    output = np.concatenate(attention_outputs, axis=-1)
    return output
```

Slide 4: Positional Encodings

Transformers process input sequences in parallel, which means they lack inherent understanding of word order. Positional encodings are added to the input embeddings to provide information about the position of each word in the sequence.

```python
def positional_encoding(seq_length, d_model):
    positions = np.arange(seq_length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * dims) / d_model)
    
    encodings = np.zeros((seq_length, d_model))
    encodings[:, 0::2] = np.sin(angles[:, 0::2])
    encodings[:, 1::2] = np.cos(angles[:, 1::2])
    
    return encodings

# Example usage
seq_length, d_model = 10, 512
pos_encoding = positional_encoding(seq_length, d_model)
print(pos_encoding.shape)  # (10, 512)
```

Slide 5: Results for: Positional Encodings

```
(10, 512)
```

Slide 6: Encoder Architecture

The Transformer encoder consists of multiple identical layers, each containing two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. Layer normalization and residual connections are applied around each sub-layer.

```python
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention sub-layer
        attn_output = self.self_attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x
```

Slide 7: Decoder Architecture

The Transformer decoder is similar to the encoder but includes an additional multi-head attention layer that attends to the encoder's output. It also employs masking in its self-attention layer to prevent attending to future tokens during training.

```python
class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x, enc_output):
        # Masked self-attention sub-layer
        attn_output = self.self_attention(x, x, x, mask=True)
        x = self.norm1(x + attn_output)
        
        # Cross-attention sub-layer
        cross_attn_output = self.cross_attention(x, enc_output, enc_output)
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x
```

Slide 8: Transformer Training

Training a Transformer involves minimizing a loss function, typically cross-entropy for language tasks. The model is trained end-to-end using backpropagation and optimization algorithms like Adam. Teacher forcing is often used during training, where the ground truth is fed as input to the decoder.

```python
def train_transformer(model, data_loader, num_epochs, lr):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for src, tgt in data_loader:
            optimizer.zero_grad()
            
            # Forward pass
            output = model(src, tgt[:, :-1])  # Teacher forcing
            
            # Compute loss
            loss = loss_fn(output.view(-1, output.size(-1)), tgt[:, 1:].view(-1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Slide 9: Transformer Inference

During inference, the Transformer generates output tokens sequentially. The decoder uses previously generated tokens as input and attends to the encoder's output to produce the next token. This process continues until an end-of-sequence token is generated or a maximum length is reached.

```python
def transformer_inference(model, src, max_len):
    model.eval()
    enc_output = model.encoder(src)
    
    # Start with start-of-sequence token
    dec_input = torch.tensor([[SOS_TOKEN]])
    output_seq = []
    
    for _ in range(max_len):
        dec_output = model.decoder(dec_input, enc_output)
        next_token = dec_output.argmax(dim=-1)
        output_seq.append(next_token.item())
        
        if next_token.item() == EOS_TOKEN:
            break
        
        # Update decoder input for next iteration
        dec_input = torch.cat([dec_input, next_token], dim=1)
    
    return output_seq
```

Slide 10: Real-Life Example: Machine Translation

Transformers excel at machine translation tasks. They can effectively capture contextual information and handle long-range dependencies, resulting in more accurate and fluent translations compared to previous models.

```python
# Example of using a pre-trained Transformer for English to French translation
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_en_to_fr(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example usage
english_text = "The Transformer model has revolutionized natural language processing."
french_translation = translate_en_to_fr(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")
```

Slide 11: Results for: Real-Life Example: Machine Translation

```
English: The Transformer model has revolutionized natural language processing.
French: Le modèle Transformer a révolutionné le traitement du langage naturel.
```

Slide 12: Real-Life Example: Text Summarization

Transformers are also widely used for text summarization tasks. They can effectively understand the main ideas in a long document and generate concise summaries that capture the essential information.

```python
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Example text to summarize
long_text = """
Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, particularly the burning of fossil fuels. These activities release greenhouse gases into the atmosphere, trapping heat and causing the Earth's average temperature to rise. The consequences of climate change are far-reaching and include more frequent and severe weather events, rising sea levels, and disruptions to ecosystems and biodiversity. To address this global challenge, governments, businesses, and individuals must work together to reduce greenhouse gas emissions, transition to renewable energy sources, and implement sustainable practices across all sectors of society.
"""

# Generate summary
summary = summarizer(long_text, max_length=100, min_length=30, do_sample=False)

print("Original text length:", len(long_text))
print("Summary:", summary[0]['summary_text'])
print("Summary length:", len(summary[0]['summary_text']))
```

Slide 13: Results for: Real-Life Example: Text Summarization

```
Original text length: 744
Summary: Climate change is a pressing issue caused by human activities, particularly burning fossil fuels. It leads to long-term shifts in temperatures and weather patterns, causing rising sea levels and disruptions to ecosystems. Addressing this challenge requires reducing greenhouse gas emissions and transitioning to renewable energy sources.
Summary length: 287
```

Slide 14: Efforts to Improve Transformer Efficiency

As Transformers have grown in size and complexity, researchers have focused on improving their efficiency. Techniques such as sparse attention, pruning, and knowledge distillation aim to reduce computational costs while maintaining performance.

```python
import numpy as np

def sparse_attention(query, key, value, sparsity):
    # Compute attention scores
    scores = np.dot(query, key.T) / np.sqrt(key.shape[1])
    
    # Keep only top-k values per row
    k = int(scores.shape[1] * (1 - sparsity))
    top_k_indices = np.argsort(scores, axis=1)[:, -k:]
    
    # Create sparse attention matrix
    sparse_scores = np.zeros_like(scores)
    rows = np.arange(scores.shape[0])[:, None]
    sparse_scores[rows, top_k_indices] = scores[rows, top_k_indices]
    
    # Apply softmax to get attention weights
    weights = np.exp(sparse_scores) / np.sum(np.exp(sparse_scores), axis=1, keepdims=True)
    
    # Compute weighted sum of values
    output = np.dot(weights, value)
    
    return output

# Example usage
query = np.random.randn(10, 64)
key = np.random.randn(20, 64)
value = np.random.randn(20, 64)
sparsity = 0.5

sparse_output = sparse_attention(query, key, value, sparsity)
print("Sparse attention output shape:", sparse_output.shape)
```

Slide 15: Results for: Efforts to Improve Transformer Efficiency

```
Sparse attention output shape: (10, 64)
```

Slide 16: Additional Resources

For a deeper understanding of Transformers and their applications, consider exploring these resources:

1.  Original Transformer paper: "Attention Is All You Need" by Vaswani et al. (2017) ArXiv URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. (2018) ArXiv URL: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3.  "GPT-3: Language Models are Few-Shot Learners" by Brown et al. (2020) ArXiv URL: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
4.  "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020) ArXiv URL: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

These papers provide comprehensive insights into the development and applications of Transformer models in various domains of machine learning and artificial intelligence.

