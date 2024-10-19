## Transformers Architecture Explained
Slide 1: What is a Transformer?

A Transformer is a neural network architecture designed for processing sequential data, particularly in natural language processing tasks. It relies solely on self-attention mechanisms, abandoning recurrent neural networks (RNNs) and convolutions. This innovative approach allows Transformers to capture long-range dependencies and context more effectively than previous models.

Slide 2: Source Code for What is a Transformer?

```python
import random

class Transformer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
    def self_attention(self, sequence):
        # Simplified self-attention mechanism
        attention_weights = [random.random() for _ in range(len(sequence))]
        return [w * x for w, x in zip(attention_weights, sequence)]
    
    def forward(self, input_sequence):
        # Apply self-attention
        attended_sequence = self.self_attention(input_sequence)
        
        # Simple output generation (placeholder)
        output = [sum(attended_sequence) / len(attended_sequence)] * self.output_size
        
        return output

# Example usage
transformer = Transformer(input_size=10, output_size=5)
input_seq = [random.random() for _ in range(10)]
output = transformer.forward(input_seq)
print(f"Input: {input_seq}\nOutput: {output}")
```

Slide 3: Encoder-Decoder Architecture

The Transformer architecture consists of two main components: the encoder and the decoder. The encoder processes the input sequence, capturing its essence and context. The decoder then takes this encoded information and generates the output sequence. This structure is particularly effective for tasks like machine translation, where input in one language is transformed into output in another.

Slide 4: Source Code for Encoder-Decoder Architecture

```python
class EncoderDecoderTransformer:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
    def encode(self, input_sequence):
        # Simplified encoding process
        return [word * 2 for word in input_sequence]
    
    def decode(self, encoded_sequence, target_sequence):
        # Simplified decoding process
        return [min(word, self.vocab_size - 1) for word in encoded_sequence]
    
    def forward(self, input_sequence, target_sequence):
        encoded = self.encode(input_sequence)
        output = self.decode(encoded, target_sequence)
        return output

# Example usage
vocab_size = 1000
d_model = 512
transformer = EncoderDecoderTransformer(vocab_size, d_model)

input_seq = [5, 10, 15, 20]  # Simplified input (word indices)
target_seq = [0, 0, 0, 0]  # Placeholder target sequence
output = transformer.forward(input_seq, target_seq)
print(f"Input: {input_seq}\nOutput: {output}")
```

Slide 5: Self-Attention Mechanism

The self-attention mechanism is a key innovation in Transformers. It allows the model to weigh the importance of different parts of the input sequence when processing each element. This enables the capture of long-range dependencies and contextual information, leading to improved performance on various NLP tasks.

Slide 6: Source Code for Self-Attention Mechanism

```python
import math

def self_attention(query, key, value):
    # Simplified self-attention calculation
    d_k = len(query)
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(d_k) for _ in range(len(value))]
    
    # Softmax normalization
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    attention_weights = [score / sum_exp_scores for score in exp_scores]
    
    # Weighted sum of values
    return [sum(w * v for w, v in zip(attention_weights, value))]

# Example usage
query = [1, 2, 3]
key = [4, 5, 6]
value = [7, 8, 9]
result = self_attention(query, key, value)
print(f"Query: {query}\nKey: {key}\nValue: {value}\nAttention Result: {result}")
```

Slide 7: Multi-Head Attention

Multi-head attention extends the self-attention mechanism by applying multiple attention operations in parallel. This allows the model to focus on different aspects of the input simultaneously, capturing various types of relationships and dependencies within the data.

Slide 8: Source Code for Multi-Head Attention

```python
def multi_head_attention(query, key, value, num_heads):
    d_model = len(query)
    d_k = d_model // num_heads
    
    def split_heads(x):
        return [x[i:i+d_k] for i in range(0, d_model, d_k)]
    
    # Split inputs into multiple heads
    queries = split_heads(query)
    keys = split_heads(key)
    values = split_heads(value)
    
    # Apply self-attention for each head
    head_outputs = [self_attention(q, k, v) for q, k, v in zip(queries, keys, values)]
    
    # Concatenate head outputs
    return [val for head in head_outputs for val in head]

# Example usage
query = [i for i in range(8)]
key = [i * 2 for i in range(8)]
value = [i * 3 for i in range(8)]
num_heads = 2
result = multi_head_attention(query, key, value, num_heads)
print(f"Query: {query}\nKey: {key}\nValue: {value}\nMulti-Head Attention Result: {result}")
```

Slide 9: Positional Encoding

Transformers process input sequences in parallel, which means they lack inherent understanding of the order of elements. Positional encoding addresses this by adding position-dependent signals to the input embeddings, allowing the model to consider the sequence order.

Slide 10: Source Code for Positional Encoding

```python
import math

def positional_encoding(seq_length, d_model):
    pe = [[0.0 for _ in range(d_model)] for _ in range(seq_length)]
    
    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    
    return pe

# Example usage
seq_length = 4
d_model = 8
pos_encoding = positional_encoding(seq_length, d_model)
for i, encoding in enumerate(pos_encoding):
    print(f"Position {i}: {encoding}")
```

Slide 11: Feed-Forward Neural Networks

In addition to attention mechanisms, Transformers use feed-forward neural networks in each layer. These networks process the output of the attention layers, adding non-linearity and increasing the model's capacity to learn complex patterns.

Slide 12: Source Code for Feed-Forward Neural Networks

```python
def relu(x):
    return max(0, x)

def feed_forward(input_vector, weights1, biases1, weights2, biases2):
    # First layer
    hidden = [relu(sum(i * w for i, w in zip(input_vector, weight)) + b)
              for weight, b in zip(weights1, biases1)]
    
    # Second layer
    output = [sum(h * w for h, w in zip(hidden, weight)) + b
              for weight, b in zip(weights2, biases2)]
    
    return output

# Example usage
input_vector = [1, 2, 3, 4]
weights1 = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
biases1 = [0.1, 0.2]
weights2 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
biases2 = [0.3, 0.4, 0.5, 0.6]

result = feed_forward(input_vector, weights1, biases1, weights2, biases2)
print(f"Input: {input_vector}\nOutput: {result}")
```

Slide 13: Training Transformers

Training a Transformer involves optimizing its parameters to minimize a loss function. This process typically uses techniques like backpropagation and gradient descent. The model learns to generate accurate outputs for given inputs by iteratively adjusting its weights based on the computed gradients.

Slide 14: Source Code for Training Transformers

```python
import random

def simple_loss(predicted, target):
    return sum((p - t) ** 2 for p, t in zip(predicted, target)) / len(predicted)

def train_step(model, input_seq, target_seq, learning_rate):
    # Forward pass
    output = model.forward(input_seq, target_seq)
    
    # Compute loss
    loss = simple_loss(output, target_seq)
    
    # Simplified backpropagation (random weight updates)
    for param in model.__dict__.values():
        if isinstance(param, list):
            for i in range(len(param)):
                param[i] += (random.random() - 0.5) * learning_rate
    
    return loss

# Example usage
vocab_size = 1000
d_model = 512
transformer = EncoderDecoderTransformer(vocab_size, d_model)

input_seq = [5, 10, 15, 20]
target_seq = [25, 30, 35, 40]
learning_rate = 0.01

for epoch in range(5):
    loss = train_step(transformer, input_seq, target_seq, learning_rate)
    print(f"Epoch {epoch + 1}, Loss: {loss}")
```

Slide 15: Real-Life Example: Language Translation

One common application of Transformers is language translation. For instance, translating "Hello, how are you?" from English to French. The encoder processes the English input, capturing its meaning and context. The decoder then generates the French translation: "Bonjour, comment allez-vous?"

Slide 16: Source Code for Language Translation Example

```python
class SimpleTranslator:
    def __init__(self):
        self.en_to_fr = {
            "hello": "bonjour",
            "how": "comment",
            "are": "êtes",
            "you": "vous"
        }
    
    def translate(self, sentence):
        words = sentence.lower().replace("?", "").split()
        translated = [self.en_to_fr.get(word, word) for word in words]
        return " ".join(translated).capitalize() + "?"

# Example usage
translator = SimpleTranslator()
english_sentence = "Hello, how are you?"
french_translation = translator.translate(english_sentence)
print(f"English: {english_sentence}")
print(f"French: {french_translation}")
```

Slide 17: Real-Life Example: Text Summarization

Another application of Transformers is text summarization. Given a long article, a Transformer can generate a concise summary capturing the main points. This is useful for quickly understanding the essence of large documents or news articles.

Slide 18: Source Code for Text Summarization Example

```python
import random

def simple_summarize(text, summary_length=3):
    sentences = text.split('. ')
    word_count = {i: len(sentence.split()) for i, sentence in enumerate(sentences)}
    
    # Select sentences with the most words (simple heuristic)
    selected_indices = sorted(word_count, key=word_count.get, reverse=True)[:summary_length]
    summary = '. '.join(sentences[i] for i in sorted(selected_indices))
    
    return summary + '.'

# Example usage
article = """The Transformer model has revolutionized natural language processing. 
It uses self-attention mechanisms to process input sequences. 
This allows it to capture long-range dependencies effectively. 
Transformers have been applied to various tasks like translation and summarization. 
They form the basis of many state-of-the-art language models."""

summary = simple_summarize(article)
print(f"Original Article:\n{article}\n")
print(f"Summary:\n{summary}")
```

Slide 19: Additional Resources

For a deeper understanding of Transformers, refer to the original paper: Vaswani, A., et al. (2017). "Attention Is All You Need." arXiv:1706.03762 URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

This seminal work introduces the Transformer architecture and provides detailed explanations of its components and performance.

