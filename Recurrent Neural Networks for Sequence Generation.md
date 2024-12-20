## Recurrent Neural Networks for Sequence Generation
Slide 1: Introduction to Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of neural networks designed to process sequential data. Unlike traditional feedforward networks, RNNs have connections that form cycles, allowing them to maintain an internal state or "memory". This architecture makes them particularly well-suited for tasks involving time series, natural language processing, and other sequence-based problems.

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        for i in inputs:
            h = np.tanh(np.dot(self.Wxh, i) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            outputs.append(y)
        return outputs, h
```

Slide 2: The Vanishing Gradient Problem in RNNs

One of the main challenges in training RNNs is the vanishing gradient problem. As the network processes long sequences, the gradients can become extremely small, making it difficult for the network to learn long-term dependencies. This occurs because the gradient is multiplied many times by the weight matrix during backpropagation through time.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_vanishing_gradient(sequence_length, weight):
    gradients = [1.0]
    for _ in range(sequence_length - 1):
        gradients.append(gradients[-1] * weight)
    return gradients

seq_length = 100
weights = [0.9, 1.0, 1.1]

plt.figure(figsize=(10, 6))
for w in weights:
    grads = simulate_vanishing_gradient(seq_length, w)
    plt.plot(range(seq_length), grads, label=f'Weight = {w}')

plt.xlabel('Time Steps')
plt.ylabel('Gradient Magnitude')
plt.title('Vanishing Gradient Problem')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 3: Long Short-Term Memory (LSTM) Networks

To address the vanishing gradient problem, Long Short-Term Memory (LSTM) networks were introduced. LSTMs use a more complex structure with gates to control the flow of information. This allows them to learn long-term dependencies more effectively than simple RNNs.

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, prev_h, prev_c):
        # Concatenate input and previous hidden state
        combined = np.vstack((x, prev_h))
        
        # Compute gate activations
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # Update cell state and hidden state
        c = f * prev_c + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

Slide 4: Generating Sequences with RNNs

RNNs can be used to generate sequences by repeatedly sampling from the network's output distribution and feeding the sampled output back as input. This process allows the network to produce coherent sequences of arbitrary length.

```python
import numpy as np

def generate_sequence(rnn, seed, length):
    h = np.zeros((rnn.hidden_size, 1))
    x = seed
    generated_sequence = [x]
    
    for _ in range(length - 1):
        output, h = rnn.forward(x, h)
        x = np.random.choice(len(output), p=softmax(output.flatten()))
        generated_sequence.append(x)
    
    return generated_sequence

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Assuming we have a trained RNN model
rnn = SimpleRNN(input_size=10, hidden_size=64, output_size=10)
seed = np.random.randint(0, 10)
generated_seq = generate_sequence(rnn, seed, length=20)

print("Generated sequence:", generated_seq)
```

Slide 5: Non-Linear Representations in RNNs

RNNs use non-linear activation functions, such as tanh or ReLU, to create complex, non-linear representations of the input data. This allows them to capture intricate patterns and relationships in sequences that linear models cannot represent.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)
y_tanh = tanh(x)
y_relu = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_tanh, label='tanh')
plt.plot(x, y_relu, label='ReLU')
plt.title('Non-linear Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Training RNNs with Backpropagation Through Time (BPTT)

RNNs are typically trained using a technique called Backpropagation Through Time (BPTT). This involves unrolling the network for a fixed number of time steps and then applying standard backpropagation. However, due to the recurrent nature of the network, gradients are propagated backwards through time.

```python
import numpy as np

def bptt(inputs, targets, rnn, learning_rate, bptt_truncate):
    T = len(inputs)
    h = np.zeros((rnn.hidden_size, 1))
    loss = 0
    
    # Forward pass
    outputs = []
    for t in range(T):
        h = np.tanh(np.dot(rnn.Wxh, inputs[t]) + np.dot(rnn.Whh, h) + rnn.bh)
        y = np.dot(rnn.Why, h) + rnn.by
        loss += (y - targets[t])**2
        outputs.append(y)
    
    # Backward pass
    dWxh, dWhh, dWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
    dbh, dby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
    dhnext = np.zeros_like(h)
    
    for t in reversed(range(T)):
        dy = outputs[t] - targets[t]
        dWhy += np.dot(dy, h.T)
        dby += dy
        dh = np.dot(rnn.Why.T, dy) + dhnext
        dhraw = (1 - h**2) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, inputs[t].T)
        dWhh += np.dot(dhraw, h.T)
        dhnext = np.dot(rnn.Whh.T, dhraw)
        
        if t > 0:
            dhnext += np.dot(rnn.Whh.T, dhraw)
    
    # Clip gradients to mitigate exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    
    # Update parameters
    rnn.Wxh -= learning_rate * dWxh
    rnn.Whh -= learning_rate * dWhh
    rnn.Why -= learning_rate * dWhy
    rnn.bh -= learning_rate * dbh
    rnn.by -= learning_rate * dby
    
    return loss
```

Slide 7: Storing Information in RNNs

RNNs store information in their hidden state, which acts as a form of memory. This hidden state is updated at each time step based on the current input and the previous hidden state. The network learns to selectively update and maintain relevant information in this hidden state.

```python
import numpy as np

class MemoryCell:
    def __init__(self, input_size, hidden_size):
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))

    def forward(self, x, prev_h):
        return np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, prev_h) + self.b)

# Simulate memory retention
input_size, hidden_size = 10, 20
cell = MemoryCell(input_size, hidden_size)

h = np.zeros((hidden_size, 1))
inputs = [np.random.randn(input_size, 1) for _ in range(5)]

print("Initial hidden state:", h.flatten())
for i, x in enumerate(inputs):
    h = cell.forward(x, h)
    print(f"Hidden state after input {i+1}:", h.flatten())
```

Slide 8: Attention Mechanisms in RNNs

Attention mechanisms allow RNNs to focus on specific parts of the input sequence when generating each output. This helps the network handle long sequences more effectively and provides interpretability by showing which input elements are most relevant for each output.

```python
import numpy as np

def attention(query, keys, values):
    # Compute attention scores
    scores = np.dot(query, keys.T)
    
    # Apply softmax to get attention weights
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Compute weighted sum of values
    context = np.dot(weights, values)
    
    return context, weights

# Example usage
seq_length, hidden_size = 5, 10
query = np.random.randn(1, hidden_size)
keys = np.random.randn(seq_length, hidden_size)
values = np.random.randn(seq_length, hidden_size)

context, weights = attention(query, keys, values)

print("Attention weights:", weights.flatten())
print("Context vector:", context.flatten())
```

Slide 9: Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions, allowing the network to capture context from both past and future time steps. This is particularly useful in tasks where the entire sequence is available, such as in natural language processing.

```python
import numpy as np

class BidirectionalRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.forward_rnn = SimpleRNN(input_size, hidden_size, output_size)
        self.backward_rnn = SimpleRNN(input_size, hidden_size, output_size)
        self.Wy = np.random.randn(output_size, 2*hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        forward_outputs, _ = self.forward_rnn.forward(inputs)
        backward_outputs, _ = self.backward_rnn.forward(inputs[::-1])
        backward_outputs = backward_outputs[::-1]
        
        combined_outputs = []
        for f, b in zip(forward_outputs, backward_outputs):
            h = np.concatenate((f, b), axis=0)
            y = np.dot(self.Wy, h) + self.by
            combined_outputs.append(y)
        
        return combined_outputs

# Example usage
input_size, hidden_size, output_size = 10, 20, 5
bi_rnn = BidirectionalRNN(input_size, hidden_size, output_size)

inputs = [np.random.randn(input_size, 1) for _ in range(3)]
outputs = bi_rnn.forward(inputs)

for i, output in enumerate(outputs):
    print(f"Output {i+1}:", output.flatten())
```

Slide 10: Sequence-to-Sequence Models

Sequence-to-sequence models use RNNs to transform input sequences into output sequences of potentially different lengths. This architecture is commonly used in machine translation, text summarization, and other tasks that involve mapping between sequences.

```python
import numpy as np

class Seq2SeqModel:
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        self.encoder = SimpleRNN(input_vocab_size, hidden_size, hidden_size)
        self.decoder = SimpleRNN(output_vocab_size, hidden_size, output_vocab_size)

    def forward(self, input_seq, target_seq):
        # Encoder
        _, encoder_state = self.encoder.forward(input_seq)
        
        # Decoder
        decoder_state = encoder_state
        decoder_input = np.zeros((self.decoder.input_size, 1))  # Start token
        outputs = []
        
        for target in target_seq:
            output, decoder_state = self.decoder.forward([decoder_input])
            outputs.append(output[0])
            decoder_input = target  # Teacher forcing
        
        return outputs

# Example usage
input_vocab_size, output_vocab_size, hidden_size = 1000, 1000, 128
seq2seq = Seq2SeqModel(input_vocab_size, output_vocab_size, hidden_size)

input_seq = [np.random.randint(input_vocab_size) for _ in range(5)]
target_seq = [np.random.randint(output_vocab_size) for _ in range(6)]

input_vectors = [np.eye(input_vocab_size)[i].reshape(-1, 1) for i in input_seq]
target_vectors = [np.eye(output_vocab_size)[i].reshape(-1, 1) for i in target_seq]

outputs = seq2seq.forward(input_vectors, target_vectors)

for i, output in enumerate(outputs):
    print(f"Output {i+1}:", np.argmax(output))
```

Slide 11: Gated Recurrent Units (GRUs)

Gated Recurrent Units (GRUs) are a simplified version of LSTMs, designed to solve the vanishing gradient problem. GRUs use two gates: a reset gate and an update gate. These gates help the network learn to capture long-term dependencies in sequences while being computationally more efficient than LSTMs.

```python
import numpy as np

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size)
        
        # Initialize biases
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))

    def forward(self, x, prev_h):
        # Concatenate input and previous hidden state
        combined = np.vstack((x, prev_h))
        
        # Update gate
        z = self.sigmoid(np.dot(self.Wz, combined) + self.bz)
        
        # Reset gate
        r = self.sigmoid(np.dot(self.Wr, combined) + self.br)
        
        # Candidate hidden state
        h_candidate = np.tanh(np.dot(self.Wh, np.vstack((x, r * prev_h))) + self.bh)
        
        # New hidden state
        h = z * prev_h + (1 - z) * h_candidate
        
        return h

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
input_size, hidden_size = 10, 20
gru = GRUCell(input_size, hidden_size)

x = np.random.randn(input_size, 1)
prev_h = np.random.randn(hidden_size, 1)

new_h = gru.forward(x, prev_h)
print("New hidden state shape:", new_h.shape)
print("First few values of new hidden state:", new_h[:5].flatten())
```

Slide 12: Handling Variable-Length Sequences

RNNs can process sequences of varying lengths, which is crucial for many real-world applications. This is typically achieved through padding shorter sequences and using masking to ignore padded values during computation.

```python
import numpy as np

def pad_sequences(sequences, max_len, pad_value=0):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padded = seq + [pad_value] * (max_len - len(seq))
        else:
            padded = seq[:max_len]
        padded_sequences.append(padded)
    return np.array(padded_sequences)

def create_mask(sequences, max_len):
    return np.array([[1 if i < len(seq) else 0 for i in range(max_len)] for seq in sequences])

# Example usage
sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9]
]

max_len = 5
padded_sequences = pad_sequences(sequences, max_len)
mask = create_mask(sequences, max_len)

print("Padded sequences:")
print(padded_sequences)
print("\nMask:")
print(mask)

# Simulating RNN processing with masking
def masked_rnn_step(x, h, mask):
    # Simplified RNN step
    h_new = np.tanh(x + h)
    # Apply mask to keep or reset hidden state
    h_new = h_new * mask[:, np.newaxis] + h * (1 - mask[:, np.newaxis])
    return h_new

hidden_size = 2
h = np.zeros((len(sequences), hidden_size))

for t in range(max_len):
    x = padded_sequences[:, t]
    h = masked_rnn_step(x, h, mask[:, t])

print("\nFinal hidden states:")
print(h)
```

Slide 13: Real-life Example: Text Generation

RNNs can be used to generate text by learning patterns in a corpus of text data. This has applications in creative writing assistance, chatbots, and content generation.

```python
import numpy as np

class CharRNN:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def forward(self, inputs, h):
        outputs = []
        for x in inputs:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            outputs.append(y)
        return outputs, h

    def sample(self, h, seed_ix, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        indices = [seed_ix]
        for _ in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            indices.append(ix)
        return indices

# Example usage (assuming a trained model)
vocab_size, hidden_size = 27, 100  # 26 letters + space
rnn = CharRNN(vocab_size, hidden_size)

# Generate text
h = np.zeros((hidden_size, 1))
seed_ix = 0  # Start with 'a'
generated_indices = rnn.sample(h, seed_ix, 50)

# Convert indices to characters (assuming 0-25 for 'a'-'z' and 26 for space)
char_map = {i: chr(i + 97) if i < 26 else ' ' for i in range(27)}
generated_text = ''.join(char_map[i] for i in generated_indices)

print("Generated text:")
print(generated_text)
```

Slide 14: Real-life Example: Sentiment Analysis

RNNs are effective for sentiment analysis tasks, where the goal is to determine the emotional tone behind a sequence of words. This has applications in social media monitoring, customer feedback analysis, and market research.

```python
import numpy as np

class SentimentRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        self.embedding = np.random.randn(vocab_size, embedding_dim)
        self.Wxh = np.random.randn(hidden_size, embedding_dim) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(1, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((1, 1))

    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        for word_idx in inputs:
            x = self.embedding[word_idx].reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return self.sigmoid(y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage (assuming a trained model)
vocab_size, embedding_dim, hidden_size = 10000, 100, 128
rnn = SentimentRNN(vocab_size, embedding_dim, hidden_size)

# Sample sentences (represented as sequences of word indices)
positive_sentence = [42, 1337, 2468, 9876]  # "The movie was great"
negative_sentence = [24, 7331, 8642, 6789]  # "I didn't enjoy the book"

positive_sentiment = rnn.forward(positive_sentence)
negative_sentiment = rnn.forward(negative_sentence)

print(f"Positive sentence sentiment: {positive_sentiment[0][0]:.4f}")
print(f"Negative sentence sentiment: {negative_sentiment[0][0]:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into Recurrent Neural Networks and their applications, the following resources are recommended:

1. "Sequence Models" course by Andrew Ng on Coursera
2. "Deep Learning" book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (Chapter 10)
3. "Recurrent Neural Networks (RNN) with Keras" tutorial on TensorFlow's official website
4. ArXiv paper: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Cho et al. (2014) URL: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
5. ArXiv paper: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Chung et al. (2014) URL: [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)

These resources provide a mix of theoretical foundations and practical implementations to further your understanding of RNNs and their capabilities in sequence modeling tasks.

