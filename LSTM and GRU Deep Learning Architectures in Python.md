## LSTM and GRU Deep Learning Architectures in Python
Slide 1: Introduction to LSTMs and GRUs

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are advanced recurrent neural network architectures designed to address the vanishing gradient problem in traditional RNNs. These architectures are particularly effective for sequence modeling tasks, such as natural language processing and time series analysis.

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Sequential

# Create a simple LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(None, 1), return_sequences=True),
    LSTM(32),
    Dense(1)
])

# Create a simple GRU model
gru_model = Sequential([
    GRU(64, input_shape=(None, 1), return_sequences=True),
    GRU(32),
    Dense(1)
])

print("LSTM Model Summary:")
lstm_model.summary()

print("\nGRU Model Summary:")
gru_model.summary()
```

Slide 2: LSTM Architecture

LSTMs introduce a memory cell and three gates: input, forget, and output. The memory cell allows the network to maintain information over long sequences, while the gates control the flow of information. This architecture enables LSTMs to learn long-term dependencies effectively.

```python
import numpy as np
import matplotlib.pyplot as plt

def lstm_cell(x, h_prev, c_prev):
    # LSTM cell implementation (simplified)
    concat = np.concatenate((x, h_prev))
    
    f = sigmoid(np.dot(concat, Wf) + bf)
    i = sigmoid(np.dot(concat, Wi) + bi)
    o = sigmoid(np.dot(concat, Wo) + bo)
    g = np.tanh(np.dot(concat, Wg) + bg)
    
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    
    return h, c

# Placeholder weight matrices and bias vectors
Wf, Wi, Wo, Wg = np.random.randn(4, 10, 10)
bf, bi, bo, bg = np.zeros((4, 10))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Visualize LSTM cell
plt.figure(figsize=(10, 6))
plt.title("LSTM Cell")
plt.axis('off')
plt.text(0.5, 0.9, "x_t", ha='center')
plt.text(0.5, 0.1, "h_t-1", ha='center')
plt.text(0.5, 0.5, "LSTM\nCell", ha='center', bbox=dict(facecolor='white', edgecolor='black'))
plt.text(0.9, 0.5, "h_t", ha='center')
plt.text(0.1, 0.5, "c_t-1", ha='center')
plt.text(0.9, 0.7, "c_t", ha='center')
plt.show()
```

Slide 3: GRU Architecture

GRUs simplify the LSTM architecture by combining the forget and input gates into a single update gate. They also merge the cell state and hidden state. This results in a more computationally efficient model while maintaining performance comparable to LSTMs in many tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

def gru_cell(x, h_prev):
    # GRU cell implementation (simplified)
    concat = np.concatenate((x, h_prev))
    
    z = sigmoid(np.dot(concat, Wz) + bz)
    r = sigmoid(np.dot(concat, Wr) + br)
    h_tilde = np.tanh(np.dot(np.concatenate((x, r * h_prev)), Wh) + bh)
    h = (1 - z) * h_prev + z * h_tilde
    
    return h

# Placeholder weight matrices and bias vectors
Wz, Wr, Wh = np.random.randn(3, 10, 10)
bz, br, bh = np.zeros((3, 10))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Visualize GRU cell
plt.figure(figsize=(10, 6))
plt.title("GRU Cell")
plt.axis('off')
plt.text(0.5, 0.9, "x_t", ha='center')
plt.text(0.5, 0.1, "h_t-1", ha='center')
plt.text(0.5, 0.5, "GRU\nCell", ha='center', bbox=dict(facecolor='white', edgecolor='black'))
plt.text(0.9, 0.5, "h_t", ha='center')
plt.show()
```

Slide 4: LSTM Implementation

Let's implement a basic LSTM layer using NumPy. This implementation will help us understand the inner workings of an LSTM cell.

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
    
    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))
        
        # Compute gate activations
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
        
        # Update cell state and hidden state
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Example usage
input_size = 10
hidden_size = 20
lstm_cell = LSTMCell(input_size, hidden_size)

x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))
c_prev = np.zeros((hidden_size, 1))

h, c = lstm_cell.forward(x, h_prev, c_prev)
print("Hidden state shape:", h.shape)
print("Cell state shape:", c.shape)
```

Slide 5: GRU Implementation

Now, let's implement a basic GRU layer using NumPy. This implementation will highlight the differences between GRU and LSTM cells.

```python
import numpy as np

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights and biases
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size)
        
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))
    
    def forward(self, x, h_prev):
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))
        
        # Compute gate activations
        z = self.sigmoid(np.dot(self.Wz, concat) + self.bz)
        r = self.sigmoid(np.dot(self.Wr, concat) + self.br)
        
        # Compute candidate hidden state
        h_tilde = np.tanh(np.dot(self.Wh, np.vstack((r * h_prev, x))) + self.bh)
        
        # Update hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Example usage
input_size = 10
hidden_size = 20
gru_cell = GRUCell(input_size, hidden_size)

x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))

h = gru_cell.forward(x, h_prev)
print("Hidden state shape:", h.shape)
```

Slide 6: Comparing LSTM and GRU

LSTMs and GRUs have similarities and differences in their architecture and performance. LSTMs have separate cell states and hidden states, while GRUs combine them. GRUs have fewer parameters, making them computationally more efficient. Despite these differences, both architectures perform well in various sequence modeling tasks.

```python
import numpy as np
import matplotlib.pyplot as plt

# Function to generate sine wave data
def generate_sine_wave(num_samples, frequency, noise_level):
    x = np.linspace(0, 4 * np.pi, num_samples)
    y = np.sin(frequency * x) + np.random.normal(0, noise_level, num_samples)
    return x, y

# Generate data
num_samples = 1000
frequency = 1.5
noise_level = 0.1
x, y = generate_sine_wave(num_samples, frequency, noise_level)

# Prepare data for LSTM/GRU
sequence_length = 50
X = []
Y = []
for i in range(len(y) - sequence_length):
    X.append(y[i:i+sequence_length])
    Y.append(y[i+sequence_length])
X = np.array(X).reshape(-1, sequence_length, 1)
Y = np.array(Y)

# Create and train LSTM model
lstm_model = tf.keras.Sequential([
    LSTM(64, input_shape=(sequence_length, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_history = lstm_model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Create and train GRU model
gru_model = tf.keras.Sequential([
    GRU(64, input_shape=(sequence_length, 1)),
    Dense(1)
])
gru_model.compile(optimizer='adam', loss='mse')
gru_history = gru_model.fit(X, Y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.plot(gru_history.history['loss'], label='GRU Training Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
plt.title('LSTM vs GRU: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

Slide 7: Handling Long-Term Dependencies

Both LSTM and GRU are designed to handle long-term dependencies in sequential data. Let's demonstrate this capability by training models on a sequence with long-term patterns.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Generate data with long-term dependency
def generate_long_term_data(sequence_length, num_sequences):
    X = np.zeros((num_sequences, sequence_length))
    y = np.zeros((num_sequences, 1))
    
    for i in range(num_sequences):
        X[i, 0] = np.random.randint(0, 2)
        X[i, 1] = np.random.randint(0, 2)
        y[i] = X[i, 0] ^ X[i, 1]  # XOR of first two elements
    
    return X, y

# Generate data
sequence_length = 20
num_sequences = 10000
X, y = generate_long_term_data(sequence_length, num_sequences)

# Reshape input for LSTM/GRU
X = X.reshape(num_sequences, sequence_length, 1)

# Create and train LSTM model
lstm_model = Sequential([
    LSTM(32, input_shape=(sequence_length, 1)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Create and train GRU model
gru_model = Sequential([
    GRU(32, input_shape=(sequence_length, 1)),
    Dense(1, activation='sigmoid')
])
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_history = gru_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(lstm_history.history['accuracy'], label='LSTM Training Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation Accuracy')
plt.plot(gru_history.history['accuracy'], label='GRU Training Accuracy')
plt.plot(gru_history.history['val_accuracy'], label='GRU Validation Accuracy')
plt.title('LSTM vs GRU: Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 8: Gradient Flow in LSTM and GRU

One of the key advantages of LSTM and GRU over traditional RNNs is their ability to mitigate the vanishing gradient problem. Let's visualize how gradients flow through these architectures.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_flow(cell_type, sequence_length):
    if cell_type == 'lstm':
        forget_gate = np.random.uniform(0.5, 1, sequence_length)
        input_gate = 1 - forget_gate
        gradient = np.cumprod(forget_gate[::-1])[::-1]
    elif cell_type == 'gru':
        update_gate = np.random.uniform(0, 0.5, sequence_length)
        gradient = np.cumprod(1 - update_gate[::-1])[::-1]
    else:
        raise ValueError("Invalid cell type. Choose 'lstm' or 'gru'.")
    
    return gradient

sequence_length = 100
lstm_gradient = gradient_flow('lstm', sequence_length)
gru_gradient = gradient_flow('gru', sequence_length)

plt.figure(figsize=(12, 6))
plt.plot(lstm_gradient, label='LSTM')
plt.plot(gru_gradient, label='GRU')
plt.title('Gradient Flow in LSTM and GRU')
plt.xlabel('Time Steps')
plt.ylabel('Gradient Magnitude')
plt.legend()
plt.show()
```

Slide 9: Bidirectional LSTMs and GRUs

Bidirectional LSTMs and GRUs process sequences in both forward and backward directions, capturing context from both past and future time steps. This approach is particularly useful in tasks where the entire sequence is available, such as text classification or named entity recognition.

```python
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.models import Sequential

# Example of a Bidirectional LSTM model
bi_lstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(None, 1)),
    Bidirectional(LSTM(32)),
    Dense(1)
])

# Example of a Bidirectional GRU model
bi_gru_model = Sequential([
    Bidirectional(GRU(64, return_sequences=True), input_shape=(None, 1)),
    Bidirectional(GRU(32)),
    Dense(1)
])

print("Bidirectional LSTM Model Summary:")
bi_lstm_model.summary()

print("\nBidirectional GRU Model Summary:")
bi_gru_model.summary()
```

Slide 10: Attention Mechanism in LSTMs and GRUs

Attention mechanisms allow models to focus on specific parts of the input sequence when generating output. This technique has significantly improved performance in various sequence-to-sequence tasks, such as machine translation and text summarization.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_output, decoder_hidden):
        decoder_hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        score = self.V(tf.nn.tanh(self.W(encoder_output) + decoder_hidden_with_time_axis))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Example usage in a seq2seq model
encoder_output = tf.random.normal((64, 20, 256))  # (batch_size, max_length, hidden_size)
decoder_hidden = tf.random.normal((64, 256))  # (batch_size, hidden_size)

attention_layer = AttentionLayer(256)
context_vector, attention_weights = attention_layer(encoder_output, decoder_hidden)

print("Context vector shape:", context_vector.shape)
print("Attention weights shape:", attention_weights.shape)
```

Slide 11: Real-life Example: Sentiment Analysis

Let's implement a sentiment analysis model using LSTM for movie reviews. We'll use the IMDB dataset, which contains movie reviews labeled as positive or negative.

```python
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the IMDB dataset
vocab_size = 10000
max_length = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure uniform length
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Build the model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Make predictions
sample_review = X_test[0]
prediction = model.predict(sample_review.reshape(1, -1))[0][0]
print(f"Prediction: {'Positive' if prediction > 0.5 else 'Negative'} (confidence: {prediction:.2f})")
```

Slide 12: Real-life Example: Time Series Forecasting

Now, let's use a GRU model for time series forecasting. We'll predict future values of a sine wave with added noise.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Generate sine wave data
def generate_sine_wave(num_samples, frequency, noise_level):
    x = np.linspace(0, 4 * np.pi, num_samples)
    y = np.sin(frequency * x) + np.random.normal(0, noise_level, num_samples)
    return x, y

num_samples = 1000
frequency = 1.5
noise_level = 0.1
x, y = generate_sine_wave(num_samples, frequency, noise_level)

# Prepare data for GRU
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 50
X, Y = create_dataset(y, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Build and train the GRU model
model = Sequential([
    GRU(50, input_shape=(look_back, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y, label='Actual')
plt.plot(np.arange(look_back, look_back + len(train_predict)), train_predict, label='Train Predict')
plt.plot(np.arange(look_back + len(train_predict), look_back + len(train_predict) + len(test_predict)), test_predict, label='Test Predict')
plt.legend()
plt.title('GRU Time Series Forecasting')
plt.show()
```

Slide 13: Choosing Between LSTM and GRU

When deciding between LSTM and GRU for a specific task, consider the following factors:

1. Complexity of the problem: LSTMs may perform better on more complex sequence tasks due to their separate cell state.
2. Dataset size: GRUs might be preferable for smaller datasets due to fewer parameters.
3. Computational resources: GRUs are generally faster to train and require less memory.
4. Task-specific performance: Always compare both architectures empirically for your specific task.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

def create_model(cell_type, input_shape, output_units):
    model = Sequential()
    if cell_type == 'lstm':
        model.add(LSTM(64, input_shape=input_shape))
    elif cell_type == 'gru':
        model.add(GRU(64, input_shape=input_shape))
    model.add(Dense(output_units))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 50, 1)
y = np.random.randn(1000, 1)

# Train LSTM and GRU models
lstm_model = create_model('lstm', (50, 1), 1)
gru_model = create_model('gru', (50, 1), 1)

lstm_history = lstm_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
gru_history = gru_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Plot training curves
plt.figure(figsize=(12, 6))
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.plot(gru_history.history['loss'], label='GRU Training Loss')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
plt.title('LSTM vs GRU: Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For further exploration of LSTMs and GRUs, consider the following resources:

1. "Long Short-Term Memory" by Sepp Hochreiter and Jürgen Schmidhuber (1997) ArXiv: [https://arxiv.org/abs/1409.1259v2](https://arxiv.org/abs/1409.1259v2)
2. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Kyunghyun Cho et al. (2014) ArXiv: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
3. "LSTM: A Search Space Odyssey" by Klaus Greff et al. (2017) ArXiv: [https://arxiv.org/abs/1503.04069](https://arxiv.org/abs/1503.04069)
4. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Junyoung Chung et al. (2014) ArXiv: [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)

These papers provide in-depth explanations of the architectures, their variations, and applications in various domains of deep learning and natural language processing.

