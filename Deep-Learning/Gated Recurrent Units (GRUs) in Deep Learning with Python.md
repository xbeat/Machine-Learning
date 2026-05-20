## Gated Recurrent Units (GRUs) in Deep Learning with Python

Slide 1: Introduction to Gated Recurrent Units (GRUs)

Gated Recurrent Units (GRUs) are a type of recurrent neural network architecture designed to solve the vanishing gradient problem in traditional RNNs. Introduced by Cho et al. in 2014, GRUs have become popular due to their ability to capture long-term dependencies in sequential data while maintaining computational efficiency.

```python

# Creating a simple GRU layer
gru_layer = tf.keras.layers.GRU(units=64, activation='tanh', recurrent_activation='sigmoid')

# Example input shape: (batch_size, time_steps, features)
input_shape = (32, 10, 5)
output = gru_layer(tf.random.normal(input_shape))

print(f"Output shape: {output.shape}")
# Output shape: (32, 64)
```

Slide 2: GRU Architecture

The GRU architecture consists of two main gates: the update gate and the reset gate. These gates control the flow of information through the unit, allowing it to selectively update or reset its internal state. This mechanism enables GRUs to learn long-term dependencies while mitigating the vanishing gradient problem.

```python
import matplotlib.pyplot as plt

def gru_cell(x, h, W, U, b):
    z = sigmoid(np.dot(W[0], x) + np.dot(U[0], h) + b[0])
    r = sigmoid(np.dot(W[1], x) + np.dot(U[1], h) + b[1])
    h_tilde = np.tanh(np.dot(W[2], x) + np.dot(U[2], r * h) + b[2])
    h_new = (1 - z) * h + z * h_tilde
    return h_new

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Visualize GRU cell
plt.figure(figsize=(10, 6))
plt.title("GRU Cell Architecture")
plt.text(0.5, 0.9, "x_t", fontsize=12, ha='center')
plt.text(0.5, 0.1, "h_t-1", fontsize=12, ha='center')
plt.text(0.5, 0.5, "GRU Cell", fontsize=14, ha='center', bbox=dict(facecolor='white', edgecolor='black'))
plt.text(0.9, 0.5, "h_t", fontsize=12, ha='center')
plt.axis('off')
plt.show()
```

Slide 3: Update Gate

The update gate determines how much of the previous hidden state should be retained and how much of the new candidate state should be added. It helps the network decide which information to keep or discard over long sequences.

```python
    z = sigmoid(np.dot(Wz, x) + np.dot(Uz, h_prev) + bz)
    return z

# Example usage
x = np.random.randn(5, 1)  # Input
h_prev = np.random.randn(3, 1)  # Previous hidden state
Wz = np.random.randn(3, 5)  # Weight matrix for input
Uz = np.random.randn(3, 3)  # Weight matrix for previous hidden state
bz = np.random.randn(3, 1)  # Bias

z = update_gate(x, h_prev, Wz, Uz, bz)
print("Update gate output:")
print(z)
```

Slide 4: Reset Gate

The reset gate controls how much of the previous hidden state should be forgotten when computing the new candidate state. This allows the GRU to reset its memory and start fresh when necessary, which is particularly useful for capturing short-term dependencies.

```python
    r = sigmoid(np.dot(Wr, x) + np.dot(Ur, h_prev) + br)
    return r

# Example usage
x = np.random.randn(5, 1)  # Input
h_prev = np.random.randn(3, 1)  # Previous hidden state
Wr = np.random.randn(3, 5)  # Weight matrix for input
Ur = np.random.randn(3, 3)  # Weight matrix for previous hidden state
br = np.random.randn(3, 1)  # Bias

r = reset_gate(x, h_prev, Wr, Ur, br)
print("Reset gate output:")
print(r)
```

Slide 5: Candidate Hidden State

The candidate hidden state is computed using the current input and a filtered version of the previous hidden state. The reset gate determines how much of the previous hidden state to use in this computation.

```python
    h_tilde = np.tanh(np.dot(Wh, x) + np.dot(Uh, r * h_prev) + bh)
    return h_tilde

# Example usage
x = np.random.randn(5, 1)  # Input
h_prev = np.random.randn(3, 1)  # Previous hidden state
r = np.random.rand(3, 1)  # Reset gate output
Wh = np.random.randn(3, 5)  # Weight matrix for input
Uh = np.random.randn(3, 3)  # Weight matrix for previous hidden state
bh = np.random.randn(3, 1)  # Bias

h_tilde = candidate_hidden_state(x, h_prev, r, Wh, Uh, bh)
print("Candidate hidden state:")
print(h_tilde)
```

Slide 6: New Hidden State Computation

The new hidden state is a weighted combination of the previous hidden state and the candidate hidden state. The update gate determines the mixing proportion between these two states.

```python
    h_new = (1 - z) * h_prev + z * h_tilde
    return h_new

# Example usage
h_prev = np.random.randn(3, 1)  # Previous hidden state
z = np.random.rand(3, 1)  # Update gate output
h_tilde = np.random.randn(3, 1)  # Candidate hidden state

h_new = new_hidden_state(h_prev, z, h_tilde)
print("New hidden state:")
print(h_new)
```

Slide 7: GRU vs LSTM

GRUs are often compared to Long Short-Term Memory (LSTM) units. While both address the vanishing gradient problem, GRUs have fewer parameters and are computationally more efficient. GRUs combine the forget and input gates into a single update gate and merge the cell state with the hidden state, resulting in a simpler architecture.

```python
import time

# Function to create and train a model
def create_and_train_model(cell_type, units, input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        cell_type(units=units, return_sequences=True),
        tf.keras.layers.Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Generate dummy data
    x = tf.random.normal((1000, 50, 10))
    y = tf.random.normal((1000, 50, 1))
    
    # Train the model and measure time
    start_time = time.time()
    model.fit(x, y, epochs=5, batch_size=32, verbose=0)
    end_time = time.time()
    
    return end_time - start_time

# Compare GRU and LSTM
gru_time = create_and_train_model(tf.keras.layers.GRU, 64, (50, 10), 1)
lstm_time = create_and_train_model(tf.keras.layers.LSTM, 64, (50, 10), 1)

print(f"GRU training time: {gru_time:.2f} seconds")
print(f"LSTM training time: {lstm_time:.2f} seconds")
```

Slide 8: Implementing a GRU Layer in TensorFlow

TensorFlow provides a high-level API to create GRU layers easily. Here's an example of how to implement a GRU layer in a sequential model for a time series prediction task.

```python
import numpy as np

# Create a simple sequential model with a GRU layer
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(None, 1), return_sequences=True),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Generate dummy time series data
time_steps = 100
series = np.sin(0.1 * np.arange(time_steps))
x = series[:-1].reshape(-1, 1, 1)
y = series[1:].reshape(-1, 1, 1)

# Train the model
history = model.fit(x, y, epochs=50, verbose=0)

# Plot the training loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
```

Slide 9: Bidirectional GRU

Bidirectional GRUs process input sequences in both forward and backward directions, allowing the network to capture context from both past and future states. This is particularly useful for tasks where the entire sequence context is important, such as natural language processing.

```python

# Create a bidirectional GRU model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True), 
                                  input_shape=(None, 1)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()
```

Slide 10: GRU for Text Classification

GRUs are widely used in natural language processing tasks, such as text classification. Here's an example of using a GRU layer for sentiment analysis on movie reviews.

```python
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Load the IMDB dataset
max_features = 20000
maxlen = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_features, 128),
    tf.keras.layers.GRU(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test))

# Evaluate the model
score, acc = model.evaluate(x_test, y_test, batch_size=32)
print(f'Test score: {score:.3f}')
print(f'Test accuracy: {acc:.3f}')
```

Slide 11: GRU for Time Series Forecasting

GRUs are effective for time series forecasting due to their ability to capture long-term dependencies. Here's an example of using a GRU for predicting air pollution levels based on historical data.

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic air pollution data
np.random.seed(42)
time_steps = 1000
pollution_data = np.sin(0.1 * np.arange(time_steps)) + np.random.normal(0, 0.1, time_steps)

# Prepare data for training
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 50
X, Y = create_dataset(pollution_data, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(look_back, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(pollution_data, label='Actual')
plt.plot(range(look_back, look_back + len(train_predict)), train_predict, label='Train Predict')
plt.plot(range(look_back + len(train_predict), look_back + len(train_predict) + len(test_predict)), 
         test_predict, label='Test Predict')
plt.legend()
plt.title('Air Pollution Forecasting with GRU')
plt.show()
```

Slide 12: GRU for Music Generation

GRUs can be used for creative tasks such as music generation. This example demonstrates a simple model that generates a melody based on a given sequence of notes.

```python
import tensorflow as tf

# Define a simple vocabulary of musical notes
notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
note_to_int = {note: i for i, note in enumerate(notes)}
int_to_note = {i: note for i, note in enumerate(notes)}

# Generate a simple melody
melody = ['C', 'E', 'G', 'C', 'G', 'E', 'C', 'D', 'F', 'A', 'D', 'F', 'D', 'C']
melody_encoded = [note_to_int[note] for note in melody]

# Prepare sequences for training
sequence_length = 3
X, y = [], []
for i in range(len(melody_encoded) - sequence_length):
    X.append(melody_encoded[i:i+sequence_length])
    y.append(melody_encoded[i+sequence_length])

X = np.array(X)
y = np.array(y)
X = tf.keras.utils.to_categorical(X, num_classes=len(notes))
y = tf.keras.utils.to_categorical(y, num_classes=len(notes))

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(sequence_length, len(notes)), return_sequences=True),
    tf.keras.layers.GRU(64),
    tf.keras.layers.Dense(len(notes), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X, y, epochs=100, batch_size=32, verbose=0)

# Generate a new melody
seed = X[0]
generated_melody = []
for _ in range(10):
    prediction = model.predict(seed.reshape(1, sequence_length, len(notes)))
    next_note = np.argmax(prediction)
    generated_melody.append(int_to_note[next_note])
    seed = np.roll(seed, -1, axis=0)
    seed[-1] = tf.keras.utils.to_categorical(next_note, num_classes=len(notes))

print("Generated melody:", ' '.join(generated_melody))
```

Slide 13: GRU for Anomaly Detection in Time Series Data

GRUs can be effectively used for anomaly detection in time series data. This example demonstrates how to use a GRU-based autoencoder to detect anomalies in a synthetic dataset.

```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic time series data with anomalies
def generate_data(n_samples, seq_length):
    data = np.sin(np.linspace(0, 100, n_samples * seq_length)).reshape(n_samples, seq_length)
    # Add random noise
    data += np.random.normal(0, 0.1, data.shape)
    # Insert anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    data[anomaly_indices] += np.random.normal(0, 1, (len(anomaly_indices), seq_length))
    return data, anomaly_indices

n_samples, seq_length = 1000, 50
data, anomaly_indices = generate_data(n_samples, seq_length)

# Create and train the GRU autoencoder
model = tf.keras.Sequential([
    tf.keras.layers.GRU(32, activation='tanh', input_shape=(seq_length, 1), return_sequences=True),
    tf.keras.layers.GRU(16, activation='tanh', return_sequences=False),
    tf.keras.layers.RepeatVector(seq_length),
    tf.keras.layers.GRU(16, activation='tanh', return_sequences=True),
    tf.keras.layers.GRU(32, activation='tanh', return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])

model.compile(optimizer='adam', loss='mse')
model.fit(data, data, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Detect anomalies
reconstructed_data = model.predict(data)
mse = np.mean(np.power(data - reconstructed_data, 2), axis=1)
threshold = np.percentile(mse, 95)
predicted_anomalies = mse > threshold

# Visualize results
plt.figure(figsize=(12, 6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(anomaly_indices, mse[anomaly_indices], color='red', label='True Anomalies')
plt.legend()
plt.title('Anomaly Detection using GRU Autoencoder')
plt.show()

print(f"F1-score: {f1_score(anomaly_indices, predicted_anomalies):.2f}")
```

Slide 14: GRU for Natural Language Processing: Named Entity Recognition

GRUs are widely used in various Natural Language Processing (NLP) tasks. This example demonstrates how to use a GRU for Named Entity Recognition (NER), a task that involves identifying and classifying named entities in text.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data (simplified for demonstration)
sentences = [
    "John lives in New York",
    "Apple is headquartered in Cupertino",
    "The Eiffel Tower is in Paris"
]
labels = [
    ["B-PER", "O", "O", "B-LOC", "I-LOC"],
    ["B-ORG", "O", "O", "O", "B-LOC"],
    ["B-LOC", "I-LOC", "I-LOC", "O", "O", "B-LOC"]
]

# Create vocabularies
words = set([word for sentence in sentences for word in sentence.split()])
word_to_id = {word: i+1 for i, word in enumerate(words)}
tag_to_id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4, "B-ORG": 5, "I-ORG": 6}

# Convert sentences and labels to numerical data
X = [[word_to_id[word] for word in sentence.split()] for sentence in sentences]
y = [[tag_to_id[tag] for tag in sentence_labels] for sentence_labels in labels]

# Pad sequences
max_len = max(len(sentence) for sentence in X)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')

# Convert to categorical
y = tf.keras.utils.to_categorical(y)

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_to_id)+1, output_dim=50, input_length=max_len),
    tf.keras.layers.GRU(100, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tag_to_id), activation='softmax'))
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, batch_size=1, verbose=0)

# Make predictions
test_sentence = "Emma works at Google"
test_sequence = [word_to_id.get(word, 0) for word in test_sentence.split()]
test_padded = pad_sequences([test_sequence], maxlen=max_len, padding='post')
predictions = model.predict(test_padded)

# Decode predictions
id_to_tag = {i: tag for tag, i in tag_to_id.items()}
predicted_tags = [id_to_tag[np.argmax(pred)] for pred in predictions[0][:len(test_sequence)]]

print("Sentence:", test_sentence)
print("Predicted tags:", predicted_tags)
```

Slide 15: Additional Resources

For those interested in diving deeper into Gated Recurrent Units and their applications, here are some valuable resources:

1. Original GRU paper: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Cho et al. (2014) ArXiv link: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
2. "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Chung et al. (2014) ArXiv link: [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)
3. "An Empirical Exploration of Recurrent Network Architectures" by Jozefowicz et al. (2015) Proceedings of the 32nd International Conference on Machine Learning
4. TensorFlow GRU documentation: [https://www.tensorflow.org/api\_docs/python/tf/keras/layers/GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
5. PyTorch GRU documentation: [https://pytorch.org/docs/stable/generated/torch.nn.GRU.html](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

These resources provide a comprehensive understanding of GRUs, their implementation details, and comparisons with other recurrent neural network architectures.


