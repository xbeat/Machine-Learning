## LSTM and GRU Basics
Slide 1: Introduction to LSTM and GRU

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are advanced recurrent neural network architectures designed to handle sequential data and long-term dependencies. These models have revolutionized natural language processing, time series analysis, and many other sequence-based tasks.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# Example of creating a simple LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(None, 1), return_sequences=True),
    LSTM(32),
    Dense(1)
])

# Example of creating a simple GRU model
gru_model = Sequential([
    GRU(64, input_shape=(None, 1), return_sequences=True),
    GRU(32),
    Dense(1)
])

print("LSTM model summary:")
lstm_model.summary()

print("\nGRU model summary:")
gru_model.summary()
```

Slide 2: LSTM Architecture

LSTM cells contain three gates: input, forget, and output. These gates control the flow of information through the cell, allowing it to selectively remember or forget information over long sequences.

```python
import tensorflow as tf

class SimpleLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SimpleLSTMCell, self).__init__()
        self.units = units
        self.state_size = [units, units]
        self.output_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1] + self.units, 4 * self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 4 * self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.bias = self.add_weight(shape=(4 * self.units,),
                                    initializer='uniform',
                                    name='bias')

    def call(self, inputs, states):
        h_prev, c_prev = states
        z = tf.matmul(tf.concat([inputs, h_prev], 1), self.kernel)
        z += tf.matmul(h_prev, self.recurrent_kernel)
        z = tf.nn.bias_add(z, self.bias)

        i, f, c, o = tf.split(z, 4, axis=1)

        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        c = f * c_prev + i * tf.tanh(c)
        o = tf.sigmoid(o)

        h = o * tf.tanh(c)
        return h, [h, c]

# Usage example
cell = SimpleLSTMCell(10)
x = tf.random.normal([1, 5])
h = tf.zeros([1, 10])
c = tf.zeros([1, 10])
output, next_state = cell(x, [h, c])
print("Output shape:", output.shape)
print("Next state shapes:", [s.shape for s in next_state])
```

Slide 3: GRU Architecture

GRU is a simplified version of LSTM with two gates: reset and update. It combines the forget and input gates into a single update gate, making it computationally more efficient while still maintaining good performance.

```python
import tensorflow as tf

class SimpleGRUCell(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SimpleGRUCell, self).__init__()
        self.units = units
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1] + self.units, 3 * self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, 3 * self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.bias = self.add_weight(shape=(3 * self.units,),
                                    initializer='uniform',
                                    name='bias')

    def call(self, inputs, states):
        h_prev = states[0]
        z = tf.matmul(tf.concat([inputs, h_prev], 1), self.kernel)
        z += tf.matmul(h_prev, self.recurrent_kernel)
        z = tf.nn.bias_add(z, self.bias)

        r, u, c = tf.split(z, 3, axis=1)

        r = tf.sigmoid(r)
        u = tf.sigmoid(u)
        c = tf.tanh(c)

        h = u * h_prev + (1 - u) * c
        return h, [h]

# Usage example
cell = SimpleGRUCell(10)
x = tf.random.normal([1, 5])
h = tf.zeros([1, 10])
output, next_state = cell(x, [h])
print("Output shape:", output.shape)
print("Next state shape:", next_state[0].shape)
```

Slide 4: Input Gate in LSTM

The input gate in LSTM determines which new information should be stored in the cell state. It uses a sigmoid function to decide which values to update and a tanh function to create candidate values.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def input_gate(x, h_prev, W_xi, W_hi, b_i):
    return tf.sigmoid(tf.matmul(x, W_xi) + tf.matmul(h_prev, W_hi) + b_i)

# Example usage
x = tf.constant(np.random.randn(1, 10), dtype=tf.float32)
h_prev = tf.constant(np.random.randn(1, 20), dtype=tf.float32)
W_xi = tf.constant(np.random.randn(10, 20), dtype=tf.float32)
W_hi = tf.constant(np.random.randn(20, 20), dtype=tf.float32)
b_i = tf.constant(np.random.randn(20), dtype=tf.float32)

i = input_gate(x, h_prev, W_xi, W_hi, b_i)

plt.hist(i.numpy().flatten(), bins=20)
plt.title("Distribution of Input Gate Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("Input gate output shape:", i.shape)
print("Input gate values range from", tf.reduce_min(i).numpy(), "to", tf.reduce_max(i).numpy())
```

Slide 5: Forget Gate in LSTM

The forget gate decides what information to discard from the cell state. It uses a sigmoid function to output values between 0 and 1, where 0 means "forget this" and 1 means "keep this".

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def forget_gate(x, h_prev, W_xf, W_hf, b_f):
    return tf.sigmoid(tf.matmul(x, W_xf) + tf.matmul(h_prev, W_hf) + b_f)

# Example usage
x = tf.constant(np.random.randn(1, 10), dtype=tf.float32)
h_prev = tf.constant(np.random.randn(1, 20), dtype=tf.float32)
W_xf = tf.constant(np.random.randn(10, 20), dtype=tf.float32)
W_hf = tf.constant(np.random.randn(20, 20), dtype=tf.float32)
b_f = tf.constant(np.random.randn(20), dtype=tf.float32)

f = forget_gate(x, h_prev, W_xf, W_hf, b_f)

plt.hist(f.numpy().flatten(), bins=20)
plt.title("Distribution of Forget Gate Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("Forget gate output shape:", f.shape)
print("Forget gate values range from", tf.reduce_min(f).numpy(), "to", tf.reduce_max(f).numpy())
```

Slide 6: Output Gate in LSTM

The output gate controls what information from the cell state will be output. It uses a sigmoid function to decide which parts of the cell state to output and a tanh function to scale the values.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def output_gate(x, h_prev, W_xo, W_ho, b_o):
    return tf.sigmoid(tf.matmul(x, W_xo) + tf.matmul(h_prev, W_ho) + b_o)

# Example usage
x = tf.constant(np.random.randn(1, 10), dtype=tf.float32)
h_prev = tf.constant(np.random.randn(1, 20), dtype=tf.float32)
W_xo = tf.constant(np.random.randn(10, 20), dtype=tf.float32)
W_ho = tf.constant(np.random.randn(20, 20), dtype=tf.float32)
b_o = tf.constant(np.random.randn(20), dtype=tf.float32)

o = output_gate(x, h_prev, W_xo, W_ho, b_o)

plt.hist(o.numpy().flatten(), bins=20)
plt.title("Distribution of Output Gate Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("Output gate output shape:", o.shape)
print("Output gate values range from", tf.reduce_min(o).numpy(), "to", tf.reduce_max(o).numpy())
```

Slide 7: Update Gate in GRU

The update gate in GRU is similar to the forget and input gates in LSTM combined. It decides what information to throw away and what new information to add.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def update_gate(x, h_prev, W_xz, W_hz, b_z):
    return tf.sigmoid(tf.matmul(x, W_xz) + tf.matmul(h_prev, W_hz) + b_z)

# Example usage
x = tf.constant(np.random.randn(1, 10), dtype=tf.float32)
h_prev = tf.constant(np.random.randn(1, 20), dtype=tf.float32)
W_xz = tf.constant(np.random.randn(10, 20), dtype=tf.float32)
W_hz = tf.constant(np.random.randn(20, 20), dtype=tf.float32)
b_z = tf.constant(np.random.randn(20), dtype=tf.float32)

z = update_gate(x, h_prev, W_xz, W_hz, b_z)

plt.hist(z.numpy().flatten(), bins=20)
plt.title("Distribution of Update Gate Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("Update gate output shape:", z.shape)
print("Update gate values range from", tf.reduce_min(z).numpy(), "to", tf.reduce_max(z).numpy())
```

Slide 8: Reset Gate in GRU

The reset gate in GRU allows the model to forget the previously computed state. It helps the model to drop information that is found to be irrelevant in the future.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def reset_gate(x, h_prev, W_xr, W_hr, b_r):
    return tf.sigmoid(tf.matmul(x, W_xr) + tf.matmul(h_prev, W_hr) + b_r)

# Example usage
x = tf.constant(np.random.randn(1, 10), dtype=tf.float32)
h_prev = tf.constant(np.random.randn(1, 20), dtype=tf.float32)
W_xr = tf.constant(np.random.randn(10, 20), dtype=tf.float32)
W_hr = tf.constant(np.random.randn(20, 20), dtype=tf.float32)
b_r = tf.constant(np.random.randn(20), dtype=tf.float32)

r = reset_gate(x, h_prev, W_xr, W_hr, b_r)

plt.hist(r.numpy().flatten(), bins=20)
plt.title("Distribution of Reset Gate Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("Reset gate output shape:", r.shape)
print("Reset gate values range from", tf.reduce_min(r).numpy(), "to", tf.reduce_max(r).numpy())
```

Slide 9: Implementing LSTM in TensorFlow

Let's implement a basic LSTM layer using TensorFlow's Keras API. This example demonstrates how to create an LSTM model for a simple sequence prediction task.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample data
def generate_sequence(length):
    return np.array([np.sin(i/10) for i in range(length)])

sequence_length = 100
input_sequence = generate_sequence(sequence_length)

# Prepare data for LSTM
X = np.array([input_sequence[i:i+10] for i in range(sequence_length-10)])
y = input_sequence[10:]

X = X.reshape((X.shape[0], X.shape[1], 1))

# Create and compile the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X, y, epochs=100, verbose=0)

# Plot the training loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Make predictions
test_sequence = generate_sequence(20)
test_X = test_sequence[:10].reshape((1, 10, 1))
predicted = model.predict(test_X)

plt.plot(range(20), test_sequence, label='Actual')
plt.plot(range(10, 11), predicted, 'ro', label='Predicted')
plt.legend()
plt.title('LSTM Prediction')
plt.show()

print("Actual value:", test_sequence[10])
print("Predicted value:", predicted[0][0])
```

Slide 10: Implementing GRU in TensorFlow

Now, let's implement a basic GRU layer using TensorFlow's Keras API. We'll use the same sequence prediction task as before to compare with the LSTM implementation.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# Generate sample data
def generate_sequence(length):
    return np.array([np.sin(i/10) for i in range(length)])

sequence_length = 100
input_sequence = generate_sequence(sequence_length)

# Prepare data for GRU
X = np.array([input_sequence[i:i+10] for i in range(sequence_length-10)])
y = input_sequence[10:]

X = X.reshape((X.shape[0], X.shape[1], 1))

# Create and compile the model
model = Sequential([
    GRU(50, activation='relu', input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X, y, epochs=100, verbose=0)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('GRU Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Make predictions
test_sequence = generate_sequence(20)
test_X = test_sequence[:10].reshape((1, 10, 1))
predicted = model.predict(test_X)

plt.plot(range(20), test_sequence, label='Actual')
plt.plot(range(10, 11), predicted, 'ro', label='Predicted')
plt.legend()
plt.title('GRU Prediction')
plt.show()

print("Actual value:", test_sequence[10])
print("Predicted value:", predicted[0][0])
```

Slide 11: Comparing LSTM and GRU Performance

Let's compare the performance of LSTM and GRU models on a simple time series prediction task. We'll use the same dataset for both models and evaluate their accuracy and training time.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import time
import matplotlib.pyplot as plt

# Generate sample data
def generate_sequence(length):
    return np.array([np.sin(i/10) + np.random.normal(0, 0.1) for i in range(length)])

# Prepare data
sequence_length = 1000
input_sequence = generate_sequence(sequence_length)
X = np.array([input_sequence[i:i+10] for i in range(sequence_length-10)])
y = input_sequence[10:]
X = X.reshape((X.shape[0], X.shape[1], 1))

# Create models
lstm_model = Sequential([LSTM(50, activation='relu', input_shape=(10, 1)), Dense(1)])
gru_model = Sequential([GRU(50, activation='relu', input_shape=(10, 1)), Dense(1)])

lstm_model.compile(optimizer='adam', loss='mse')
gru_model.compile(optimizer='adam', loss='mse')

# Train and evaluate models
start_time = time.time()
lstm_history = lstm_model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)
lstm_time = time.time() - start_time

start_time = time.time()
gru_history = gru_model.fit(X, y, epochs=100, validation_split=0.2, verbose=0)
gru_time = time.time() - start_time

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(lstm_history.history['val_loss'], label='LSTM')
plt.plot(gru_history.history['val_loss'], label='GRU')
plt.title('Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print(f"LSTM training time: {lstm_time:.2f} seconds")
print(f"GRU training time: {gru_time:.2f} seconds")
print(f"LSTM final validation loss: {lstm_history.history['val_loss'][-1]:.4f}")
print(f"GRU final validation loss: {gru_history.history['val_loss'][-1]:.4f}")
```

Slide 12: Real-life Example: Sentiment Analysis

Let's use an LSTM model for sentiment analysis on movie reviews. This example demonstrates how LSTMs can be applied to natural language processing tasks.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the IMDB dataset
vocab_size = 10000
max_length = 200
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# Create the model
model = Sequential([
    Embedding(vocab_size, 32, input_length=max_length),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Make a prediction
sample_review = X_test[0]
prediction = model.predict(np.expand_dims(sample_review, axis=0))[0][0]
print(f"Sample review sentiment: {'Positive' if prediction > 0.5 else 'Negative'} (confidence: {prediction:.2f})")
```

Slide 13: Real-life Example: Time Series Forecasting

In this example, we'll use a GRU model for time series forecasting of temperature data. This demonstrates how GRUs can be applied to predict future values in a sequence.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# Generate sample temperature data
def generate_temperature_data(days):
    temperatures = []
    for i in range(days):
        temp = 20 + 10 * np.sin(2 * np.pi * i / 365) + np.random.normal(0, 2)
        temperatures.append(temp)
    return pd.DataFrame({'temperature': temperatures})

# Prepare data
data = generate_temperature_data(1000)
sequence_length = 30

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data['temperature'].values, sequence_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train the model
model = Sequential([
    GRU(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions
predictions = model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Temperature Forecasting with GRU')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

print(f"Mean Squared Error: {np.mean((y_test - predictions.flatten())**2):.4f}")
```

Slide 14: Additional Resources

For those interested in diving deeper into LSTM and GRU architectures, here are some valuable resources:

1. "Long Short-Term Memory" by Sepp Hochreiter and Jürgen Schmidhuber (1997) ArXiv: [https://arxiv.org/abs/1909.09586](https://arxiv.org/abs/1909.09586) (Note: This is a more recent paper discussing the original LSTM paper)
2. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Kyunghyun Cho et al. (2014) ArXiv: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
3. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Junyoung Chung et al. (2014) ArXiv: [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)
4. "An Empirical Exploration of Recurrent Network Architectures" by Rafal Jozefowicz et al. (2015) Proceedings of the 32nd International Conference on Machine Learning

These resources provide in-depth explanations of the architectures, their variations, and comparative analyses. They serve as excellent starting points for understanding the theoretical foundations and practical applications of LSTM and GRU models in various domains of machine learning and artificial intelligence.

