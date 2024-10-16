## Exploring 5 Key Assumptions in Linear Regression
Slide 1: Challenges in Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are powerful models for processing sequential data, but they face significant challenges. This slideshow explores these challenges and presents solutions to overcome them, focusing on the vanishing and exploding gradient problems.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_rnn(input_size, hidden_size, sequence_length):
    # Initialize weights
    W_xh = np.random.randn(hidden_size, input_size) * 0.01
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
    b_h = np.zeros((hidden_size, 1))
    
    # Forward pass
    h = np.zeros((hidden_size, 1))
    hidden_states = []
    
    for t in range(sequence_length):
        x = np.random.randn(input_size, 1)
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h)
        hidden_states.append(h)
    
    return hidden_states

# Simulate RNN
hidden_states = simple_rnn(input_size=10, hidden_size=20, sequence_length=100)

# Plot hidden state activations
plt.figure(figsize=(12, 6))
plt.imshow(np.concatenate(hidden_states, axis=1), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Hidden State Activations Over Time')
plt.xlabel('Time Step')
plt.ylabel('Hidden Unit')
plt.show()
```

Slide 2: The Vanishing Gradient Problem

The vanishing gradient problem occurs when gradients become extremely small as they are propagated backwards through time. This issue makes it difficult for RNNs to learn and remember information from earlier time steps, limiting their effectiveness on long sequences.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simulate_gradient_flow(sequence_length, weight):
    gradients = [1.0]
    for _ in range(sequence_length - 1):
        gradients.append(gradients[-1] * weight * sigmoid(0.5) * (1 - sigmoid(0.5)))
    return gradients

# Simulate gradient flow for different weights
sequence_length = 100
weights = [0.5, 1.0, 1.5]

plt.figure(figsize=(12, 6))
for weight in weights:
    gradients = simulate_gradient_flow(sequence_length, weight)
    plt.plot(gradients, label=f'Weight = {weight}')

plt.title('Gradient Flow in RNN')
plt.xlabel('Time Steps (Backward)')
plt.ylabel('Gradient Magnitude')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 3: The Exploding Gradient Problem

Conversely to the vanishing gradient problem, gradients can also become excessively large, leading to unstable network updates. This phenomenon is known as the exploding gradient problem. It can cause the model to diverge during training, making it difficult to converge to a good solution.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_exploding_gradients(sequence_length, weight):
    gradients = [1.0]
    for _ in range(sequence_length - 1):
        gradients.append(gradients[-1] * weight)
    return gradients

# Simulate exploding gradients
sequence_length = 20
weights = [1.5, 2.0, 2.5]

plt.figure(figsize=(12, 6))
for weight in weights:
    gradients = simulate_exploding_gradients(sequence_length, weight)
    plt.plot(gradients, label=f'Weight = {weight}')

plt.title('Exploding Gradients in RNN')
plt.xlabel('Time Steps (Backward)')
plt.ylabel('Gradient Magnitude')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 4: Long Short-Term Memory (LSTM)

Long Short-Term Memory (LSTM) networks are a type of RNN designed to address the vanishing gradient problem. LSTMs introduce a memory cell along with input, output, and forget gates. These gates regulate the flow of information, allowing the network to retain information over long periods.

```python
import numpy as np

def lstm_cell(x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    # Concatenate input and previous hidden state
    combined = np.concatenate((x, h_prev), axis=0)
    
    # Forget gate
    f = sigmoid(np.dot(Wf, combined) + bf)
    
    # Input gate
    i = sigmoid(np.dot(Wi, combined) + bi)
    
    # Candidate memory cell
    c_tilde = np.tanh(np.dot(Wc, combined) + bc)
    
    # Update memory cell
    c = f * c_prev + i * c_tilde
    
    # Output gate
    o = sigmoid(np.dot(Wo, combined) + bo)
    
    # Update hidden state
    h = o * np.tanh(c)
    
    return h, c

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize LSTM parameters
input_size, hidden_size = 10, 20
Wf = np.random.randn(hidden_size, input_size + hidden_size)
Wi = np.random.randn(hidden_size, input_size + hidden_size)
Wc = np.random.randn(hidden_size, input_size + hidden_size)
Wo = np.random.randn(hidden_size, input_size + hidden_size)
bf, bi, bc, bo = (np.zeros((hidden_size, 1)) for _ in range(4))

# Simulate LSTM cell
x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))
c_prev = np.zeros((hidden_size, 1))

h, c = lstm_cell(x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo)
print("LSTM output shape:", h.shape)
print("LSTM cell state shape:", c.shape)
```

Slide 5: Gated Recurrent Unit (GRU)

Gated Recurrent Units (GRUs) are a simplified version of LSTMs, combining the forget and input gates into a single update gate. GRUs offer similar performance to LSTMs with a more streamlined architecture, often resulting in faster training times.

```python
import numpy as np

def gru_cell(x, h_prev, Wz, Wr, Wh, bz, br, bh):
    # Concatenate input and previous hidden state
    combined = np.concatenate((x, h_prev), axis=0)
    
    # Update gate
    z = sigmoid(np.dot(Wz, combined) + bz)
    
    # Reset gate
    r = sigmoid(np.dot(Wr, combined) + br)
    
    # Candidate hidden state
    h_tilde = np.tanh(np.dot(Wh, np.concatenate((x, r * h_prev), axis=0)) + bh)
    
    # Update hidden state
    h = (1 - z) * h_prev + z * h_tilde
    
    return h

# Helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize GRU parameters
input_size, hidden_size = 10, 20
Wz = np.random.randn(hidden_size, input_size + hidden_size)
Wr = np.random.randn(hidden_size, input_size + hidden_size)
Wh = np.random.randn(hidden_size, input_size + hidden_size)
bz, br, bh = (np.zeros((hidden_size, 1)) for _ in range(3))

# Simulate GRU cell
x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))

h = gru_cell(x, h_prev, Wz, Wr, Wh, bz, br, bh)
print("GRU output shape:", h.shape)
```

Slide 6: Gradient Clipping

Gradient clipping is a technique used to address the exploding gradient problem. It involves setting a threshold to clip gradients during backpropagation, preventing them from becoming too large. This helps in stabilizing the training process and avoiding divergence.

```python
import numpy as np
import matplotlib.pyplot as plt

def clip_gradients(gradients, threshold):
    norm = np.linalg.norm(gradients)
    if norm > threshold:
        return threshold * gradients / norm
    return gradients

# Generate random gradients
np.random.seed(42)
gradients = np.random.randn(1000)

# Apply gradient clipping
threshold = 5.0
clipped_gradients = clip_gradients(gradients, threshold)

# Visualize the effect of gradient clipping
plt.figure(figsize=(12, 6))
plt.scatter(range(len(gradients)), gradients, alpha=0.5, label='Original')
plt.scatter(range(len(clipped_gradients)), clipped_gradients, alpha=0.5, label='Clipped')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.axhline(y=-threshold, color='r', linestyle='--')
plt.title('Effect of Gradient Clipping')
plt.xlabel('Gradient Index')
plt.ylabel('Gradient Value')
plt.legend()
plt.show()

print(f"Original gradients range: [{gradients.min():.2f}, {gradients.max():.2f}]")
print(f"Clipped gradients range: [{clipped_gradients.min():.2f}, {clipped_gradients.max():.2f}]")
```

Slide 7: Dropout Regularization

Dropout is a regularization technique that helps prevent overfitting in neural networks, including RNNs. It works by randomly dropping out (i.e., setting to zero) a proportion of neurons during training, which forces the network to learn more robust features.

```python
import numpy as np

def dropout(X, keep_prob):
    mask = np.random.rand(*X.shape) < keep_prob
    return (X * mask) / keep_prob

# Example usage
X = np.random.randn(5, 10)  # 5 samples, 10 features
keep_prob = 0.8

# Apply dropout
X_dropped = dropout(X, keep_prob)

print("Original input:")
print(X)
print("\nInput after dropout:")
print(X_dropped)

# Visualize dropout effect
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(X, cmap='viridis', aspect='auto')
plt.title('Original Input')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(X_dropped, cmap='viridis', aspect='auto')
plt.title(f'After Dropout (keep_prob={keep_prob})')
plt.colorbar()

plt.tight_layout()
plt.show()
```

Slide 8: Layer Normalization

Layer Normalization is a technique used to stabilize the learning process in neural networks, including RNNs. It normalizes the inputs across the features to reduce the shift in the distribution of network activations, which can help accelerate training and improve generalization.

```python
import numpy as np

def layer_norm(x, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon)

# Example usage
batch_size, seq_length, hidden_size = 2, 3, 4
x = np.random.randn(batch_size, seq_length, hidden_size)

# Apply layer normalization
x_normalized = layer_norm(x)

print("Original input shape:", x.shape)
print("Normalized input shape:", x_normalized.shape)

# Visualize the effect of layer normalization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(x[0], cmap='viridis', aspect='auto')
plt.title('Original Input (First Sample)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(x_normalized[0], cmap='viridis', aspect='auto')
plt.title('After Layer Normalization (First Sample)')
plt.colorbar()

plt.tight_layout()
plt.show()
```

Slide 9: Real-Life Example: Sentiment Analysis

Sentiment analysis is a common application of RNNs in natural language processing. It involves determining the sentiment (positive, negative, or neutral) of a given text. Here's a simple example using a GRU-based RNN for sentiment classification.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
texts = [
    "I love this movie!",
    "This film is terrible.",
    "The acting was great.",
    "I didn't enjoy the plot.",
    "Amazing cinematography!"
]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to have the same length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Build the model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16
hidden_units = 32

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GRU(hidden_units),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=0)

# Make predictions
new_texts = ["This movie is fantastic!", "I didn't like the characters."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_length)
predictions = model.predict(new_padded)

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"Text: '{text}'\nPredicted sentiment: {sentiment} (confidence: {pred[0]:.2f})\n")
```

Slide 10: Real-Life Example: Time Series Forecasting

Time series forecasting is another important application of RNNs. In this example, we'll use an LSTM network to predict future values of a simple time series.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate a simple time series
np.random.seed(42)
t = np.linspace(0, 100, 1000)
series = np.sin(0.1 * t) + np.random.normal(0, 0.1, 1000)

# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 10
X, y = create_dataset(series, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(series, label='Actual')
plt.plot(range(look_back, len(train_predict) + look_back), train_predict, label='Train Predictions')
plt.plot(range(len(train_predict) + look_back * 2, len(series)), test_predict, label='Test Predictions')
plt.legend()
plt.title('Time Series Forecasting with LSTM')
plt.show()
```

Slide 11: Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions, allowing the network to capture context from both past and future states. This architecture is particularly useful for tasks where the entire sequence is available, such as in natural language processing.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model

# Define a simple bidirectional LSTM model
def create_bidirectional_lstm(input_shape, units):
    inputs = Input(shape=input_shape)
    bi_lstm = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    outputs = Dense(1)(bi_lstm)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
sequence_length = 10
features = 5
lstm_units = 32

model = create_bidirectional_lstm((sequence_length, features), lstm_units)
model.summary()

# Generate dummy data
import numpy as np
X = np.random.rand(100, sequence_length, features)
y = np.random.rand(100, sequence_length, 1)

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

Slide 12: Attention Mechanisms in RNNs

Attention mechanisms allow RNNs to focus on specific parts of the input sequence when generating each output. This approach has been particularly successful in tasks like machine translation and text summarization.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Example usage
sequence_length = 10
hidden_size = 32
attention = BahdanauAttention(hidden_size)

# Generate dummy data
query = tf.random.normal([1, hidden_size])
values = tf.random.normal([1, sequence_length, hidden_size])

context_vector, attention_weights = attention(query, values)

# Visualize attention weights
plt.figure(figsize=(10, 2))
plt.imshow(attention_weights.numpy().reshape(-1, sequence_length), cmap='viridis')
plt.colorbar()
plt.title('Attention Weights')
plt.xlabel('Sequence Position')
plt.show()

print("Context vector shape:", context_vector.shape)
print("Attention weights shape:", attention_weights.shape)
```

Slide 13: Transformer Architecture

While not strictly an RNN, the Transformer architecture has largely replaced traditional RNNs in many sequence-to-sequence tasks. It uses self-attention mechanisms to process entire sequences in parallel, addressing limitations of RNNs such as sequential computation.

```python
import tensorflow as tf
import numpy as np

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output, attention_weights

# Example usage
d_model = 512
num_heads = 8

mha = MultiHeadAttention(d_model, num_heads)
temp_q = tf.random.uniform((1, 60, d_model))
temp_k = tf.random.uniform((1, 60, d_model))
temp_v = tf.random.uniform((1, 60, d_model))

output, attention_weights = mha(temp_v, temp_k, temp_q, mask=None)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
```

Slide 14: Additional Resources

For those interested in diving deeper into RNNs and their variants, here are some valuable resources:

1. "Long Short-Term Memory" by Hochreiter and Schmidhuber (1997) ArXiv: [https://arxiv.org/abs/1909.09586](https://arxiv.org/abs/1909.09586) (Note: This is a link to a more recent paper discussing LSTM, as the original paper is not available on ArXiv)
2. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Chung et al. (2014) ArXiv: [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)
3. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

These papers provide in-depth discussions on LSTM, GRU, and the Transformer architecture, respectively. They offer valuable insights into the development and evolution of sequence modeling techniques in deep learning.

