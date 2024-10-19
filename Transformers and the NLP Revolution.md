## Transformers and the NLP Revolution

Slide 1: Introduction to Transformers in NLP

The Transformer architecture, introduced in 2017, revolutionized Natural Language Processing (NLP). It addressed limitations of previous sequential models like RNNs, enabling efficient processing of long-range dependencies in text. The key innovation lies in the self-attention mechanism, which allows the model to weigh the importance of words in relation to each other, regardless of their position in the sequence.

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

# Example usage
query = np.random.rand(1, 64)  # Query vector
key = np.random.rand(10, 64)   # Key matrix (10 words, 64-dim embedding)
value = np.random.rand(10, 64) # Value matrix

result = self_attention(query, key, value)
print("Self-attention output shape:", result.shape)
```

Slide 2: Self-Attention Mechanism

The self-attention mechanism is the core component of Transformers. It allows the model to focus on different parts of the input sequence when processing each word. This mechanism computes three vectors for each word: Query, Key, and Value. The attention weights are calculated by comparing the Query of one word with the Keys of all words, determining how much focus to place on other words when encoding a particular word.

```python
def attention(query, key, value):
    # Scaled dot-product attention
    d_k = query.shape[-1]
    scores = np.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    attention_weights = np.softmax(scores, axis=-1)
    return np.matmul(attention_weights, value), attention_weights

# Example
seq_len, d_model = 4, 8
query = key = value = np.random.randn(seq_len, d_model)

output, weights = attention(query, key, value)
print("Attention output shape:", output.shape)
print("Attention weights shape:", weights.shape)
```

Slide 3: Multi-Head Attention

Multi-head attention extends the self-attention mechanism by applying multiple attention operations in parallel. Each "head" can focus on different aspects of the input, allowing the model to capture various types of relationships between words. This parallel processing contributes to the Transformer's efficiency and expressiveness.

```python
def multi_head_attention(query, key, value, num_heads):
    d_model = query.shape[-1]
    d_k = d_model // num_heads
    
    # Split into multiple heads
    def split_heads(x):
        return x.reshape(x.shape[0], num_heads, d_k)
    
    query_heads = split_heads(query)
    key_heads = split_heads(key)
    value_heads = split_heads(value)
    
    # Apply attention to each head
    head_outputs = []
    for i in range(num_heads):
        head_output, _ = attention(query_heads[:, i], key_heads[:, i], value_heads[:, i])
        head_outputs.append(head_output)
    
    # Concatenate head outputs
    multi_head_output = np.concatenate(head_outputs, axis=-1)
    return multi_head_output

# Example usage
seq_len, d_model, num_heads = 4, 64, 8
query = key = value = np.random.randn(seq_len, d_model)

output = multi_head_attention(query, key, value, num_heads)
print("Multi-head attention output shape:", output.shape)
```

Slide 4: Positional Encoding

Transformers process input sequences in parallel, losing the inherent order information. Positional encoding addresses this by adding position-dependent signals to the input embeddings. These encodings allow the model to consider the relative or absolute positions of words in the sequence.

```python
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    
    encodings = np.zeros((seq_len, d_model))
    encodings[:, 0::2] = np.sin(angles[:, 0::2])
    encodings[:, 1::2] = np.cos(angles[:, 1::2])
    
    return encodings

# Example
seq_len, d_model = 10, 64
pos_encoding = positional_encoding(seq_len, d_model)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.imshow(pos_encoding, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Sequence Position")
plt.show()
```

Slide 5: Feed-Forward Neural Networks

In addition to attention mechanisms, Transformers use feed-forward neural networks (FFNs) in each layer. These FFNs process each position independently, applying the same transformation to each element of the sequence. This component adds non-linearity and increases the model's capacity to learn complex patterns.

```python
def feed_forward_network(x, d_ff):
    # First linear transformation
    hidden = np.maximum(0, np.dot(x, np.random.randn(x.shape[-1], d_ff)))
    
    # Second linear transformation
    output = np.dot(hidden, np.random.randn(d_ff, x.shape[-1]))
    
    return output

# Example usage
seq_len, d_model, d_ff = 4, 64, 256
input_seq = np.random.randn(seq_len, d_model)

ffn_output = feed_forward_network(input_seq, d_ff)
print("Feed-forward network output shape:", ffn_output.shape)
```

Slide 6: Transformer Encoder Layer

A Transformer encoder layer combines multi-head attention, feed-forward networks, and normalization layers. This structure allows the model to process input sequences effectively, capturing complex relationships between words and transforming the representations.

```python
def layer_norm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon)

def encoder_layer(x, d_model, num_heads, d_ff):
    # Multi-head attention
    attention_output = multi_head_attention(x, x, x, num_heads)
    attention_output = layer_norm(x + attention_output)
    
    # Feed-forward network
    ffn_output = feed_forward_network(attention_output, d_ff)
    output = layer_norm(attention_output + ffn_output)
    
    return output

# Example usage
seq_len, d_model, num_heads, d_ff = 4, 64, 8, 256
input_seq = np.random.randn(seq_len, d_model)

encoder_output = encoder_layer(input_seq, d_model, num_heads, d_ff)
print("Encoder layer output shape:", encoder_output.shape)
```

Slide 7: Transformer Decoder Layer

The Transformer decoder layer is similar to the encoder but includes an additional multi-head attention layer that attends to the encoder's output. This structure allows the decoder to generate output sequences while considering both the input context and the previously generated tokens.

```python
def decoder_layer(x, enc_output, d_model, num_heads, d_ff):
    # Self-attention
    self_attention_output = multi_head_attention(x, x, x, num_heads)
    self_attention_output = layer_norm(x + self_attention_output)
    
    # Cross-attention with encoder output
    cross_attention_output = multi_head_attention(self_attention_output, enc_output, enc_output, num_heads)
    cross_attention_output = layer_norm(self_attention_output + cross_attention_output)
    
    # Feed-forward network
    ffn_output = feed_forward_network(cross_attention_output, d_ff)
    output = layer_norm(cross_attention_output + ffn_output)
    
    return output

# Example usage
seq_len, d_model, num_heads, d_ff = 4, 64, 8, 256
decoder_input = np.random.randn(seq_len, d_model)
encoder_output = np.random.randn(seq_len, d_model)

decoder_output = decoder_layer(decoder_input, encoder_output, d_model, num_heads, d_ff)
print("Decoder layer output shape:", decoder_output.shape)
```

Slide 8: Complete Transformer Architecture

The complete Transformer architecture consists of an encoder stack and a decoder stack. The encoder processes the input sequence, while the decoder generates the output sequence. This structure allows for efficient parallel processing and effective modeling of long-range dependencies in sequences.

```python
def transformer(input_seq, target_seq, num_layers, d_model, num_heads, d_ff):
    # Encoder
    enc_output = input_seq
    for _ in range(num_layers):
        enc_output = encoder_layer(enc_output, d_model, num_heads, d_ff)
    
    # Decoder
    dec_output = target_seq
    for _ in range(num_layers):
        dec_output = decoder_layer(dec_output, enc_output, d_model, num_heads, d_ff)
    
    return dec_output

# Example usage
seq_len, d_model, num_heads, d_ff, num_layers = 4, 64, 8, 256, 6
input_seq = np.random.randn(seq_len, d_model)
target_seq = np.random.randn(seq_len, d_model)

output = transformer(input_seq, target_seq, num_layers, d_model, num_heads, d_ff)
print("Transformer output shape:", output.shape)
```

Slide 9: Training Transformers

Training Transformers involves optimizing the model's parameters using techniques like gradient descent. The process typically includes tokenization, embedding, forward pass through the model, loss calculation, and backpropagation. Here's a simplified example of a training loop for a Transformer model.

```python
def train_step(model, optimizer, input_seq, target_seq):
    # Forward pass
    predictions = model(input_seq, target_seq)
    
    # Calculate loss
    loss = calculate_loss(predictions, target_seq)
    
    # Backpropagation
    gradients = calculate_gradients(loss, model.parameters())
    optimizer.apply_gradients(gradients, model.parameters())
    
    return loss

# Example training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in get_batches(training_data):
        input_seq, target_seq = batch
        loss = train_step(transformer_model, optimizer, input_seq, target_seq)
    
    print(f"Epoch {epoch + 1}, Loss: {loss}")

# Note: This is a simplified example. In practice, you would use a deep learning
# framework like PyTorch or TensorFlow for efficient training.
```

Slide 10: Transformer Variants

Since the introduction of the original Transformer, numerous variants have been developed to address specific tasks or improve performance. Some notable examples include BERT (Bidirectional Encoder Representations from Transformers), GPT (Generative Pre-trained Transformer), and T5 (Text-to-Text Transfer Transformer). These models have achieved state-of-the-art results on various NLP tasks.

```python
def bert_encoder(input_seq, num_layers, d_model, num_heads, d_ff):
    # BERT uses only the encoder part of the Transformer
    enc_output = input_seq
    for _ in range(num_layers):
        enc_output = encoder_layer(enc_output, d_model, num_heads, d_ff)
    return enc_output

def gpt_decoder(input_seq, num_layers, d_model, num_heads, d_ff):
    # GPT uses only the decoder part, but without cross-attention
    dec_output = input_seq
    for _ in range(num_layers):
        dec_output = decoder_layer(dec_output, None, d_model, num_heads, d_ff)
    return dec_output

# Example usage
seq_len, d_model, num_heads, d_ff, num_layers = 4, 64, 8, 256, 6
input_seq = np.random.randn(seq_len, d_model)

bert_output = bert_encoder(input_seq, num_layers, d_model, num_heads, d_ff)
gpt_output = gpt_decoder(input_seq, num_layers, d_model, num_heads, d_ff)

print("BERT output shape:", bert_output.shape)
print("GPT output shape:", gpt_output.shape)
```

Slide 11: Real-Life Example: Text Classification

One common application of Transformers is text classification. Here's a simplified example of how a pre-trained Transformer model could be used for sentiment analysis on movie reviews.

```python
import numpy as np

# Assume we have a pre-trained Transformer model
class PretrainedTransformer:
    def __call__(self, text):
        # Simulating the output of a pre-trained model
        return np.random.rand(768)  # 768-dimensional embedding

transformer = PretrainedTransformer()

def classify_sentiment(review):
    # Tokenize and encode the review (simplified)
    encoded_review = transformer(review)
    
    # Classification layer (simplified)
    weights = np.random.rand(768, 2)  # 2 classes: positive and negative
    logits = np.dot(encoded_review, weights)
    
    # Apply softmax to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    return "Positive" if probs[0] > probs[1] else "Negative"

# Example usage
reviews = [
    "This movie was absolutely fantastic!",
    "I've never been so bored in my life.",
    "The acting was great, but the plot was confusing."
]

for review in reviews:
    sentiment = classify_sentiment(review)
    print(f"Review: '{review}'\nSentiment: {sentiment}\n")
```

Slide 12: Real-Life Example: Machine Translation

Machine translation is a key application of Transformers. Here's a simplified example demonstrating how a Transformer-based model could be used for translating English to French.

```python
import numpy as np

class TranslationTransformer:
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model):
        self.src_embed = np.random.randn(src_vocab_size, d_model)
        self.tgt_embed = np.random.randn(tgt_vocab_size, d_model)
        self.output_proj = np.random.randn(d_model, tgt_vocab_size)
    
    def __call__(self, src_tokens, tgt_tokens):
        src_emb = self.src_embed[src_tokens]
        tgt_emb = self.tgt_embed[tgt_tokens]
        output = np.dot(src_emb + tgt_emb, self.output_proj)
        return output

def tokenize(text, vocab):
    return [vocab.get(word, 0) for word in text.lower().split()]

en_vocab = {"<start>": 0, "<end>": 1, "hello": 2, "world": 3}
fr_vocab = {"<start>": 0, "<end>": 1, "bonjour": 2, "monde": 3}

model = TranslationTransformer(len(en_vocab), len(fr_vocab), d_model=64)

def translate(text):
    src_tokens = tokenize(text, en_vocab)
    tgt_tokens = [fr_vocab["<start>"]]
    
    for _ in range(10):  # Max length of 10 tokens
        output = model(src_tokens, tgt_tokens)
        next_token = np.argmax(output[-1])
        if next_token == fr_vocab["<end>"]:
            break
        tgt_tokens.append(next_token)
    
    return " ".join([k for k, v in fr_vocab.items() if v in tgt_tokens[1:]])

input_text = "hello world"
translation = translate(input_text)
print(f"English: {input_text}")
print(f"French: {translation}")
```

Slide 13: Advantages and Limitations of Transformers

Transformers have revolutionized NLP with their ability to capture long-range dependencies and process sequences in parallel. However, they also have limitations, such as quadratic complexity with sequence length and the need for large amounts of training data.

```python
def transformer_complexity(seq_length, d_model):
    # Time complexity
    time_complexity = seq_length ** 2 * d_model
    
    # Space complexity
    space_complexity = seq_length * d_model
    
    return time_complexity, space_complexity

# Example: Compute complexity for different sequence lengths
seq_lengths = [10, 100, 1000, 10000]
d_model = 512

for length in seq_lengths:
    time, space = transformer_complexity(length, d_model)
    print(f"Sequence length: {length}")
    print(f"Time complexity: O({time})")
    print(f"Space complexity: O({space})")
    print()
```

Slide 14: Future Directions and Ongoing Research

Research in Transformer models continues to evolve rapidly. Current areas of focus include improving efficiency, reducing model size, and extending Transformers to new domains such as computer vision and speech processing.

```python
def simulate_research_progress(years, initial_performance, improvement_rate):
    performance = initial_performance
    advancements = []
    
    for year in range(years):
        performance *= (1 + improvement_rate)
        if np.random.rand() < 0.3:  # 30% chance of breakthrough
            performance *= 1.5
            advancements.append(f"Year {year + 1}: Major breakthrough!")
        advancements.append(f"Year {year + 1}: Performance = {performance:.2f}")
    
    return advancements

# Simulate 5 years of Transformer research
progress = simulate_research_progress(5, initial_performance=1.0, improvement_rate=0.2)

for advancement in progress:
    print(advancement)
```

Slide 15: Additional Resources

For those interested in diving deeper into Transformers and their applications in NLP, here are some valuable resources:

1.  "Attention Is All You Need" - The original Transformer paper (Vaswani et al., 2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
3.  "The Illustrated Transformer" by Jay Alammar - A visual guide to understanding Transformers (Note: As an AI, I can't confirm the current availability of this resource. Please check online for the most up-to-date link.)
4.  "Transformers from Scratch" tutorial series on machine learning blogs (Note: As an AI, I can't provide specific blog links. Please search for recent tutorials on implementing Transformers from scratch.)

These resources provide a mix of theoretical foundations and practical implementations to help deepen your understanding of Transformer models in NLP.

