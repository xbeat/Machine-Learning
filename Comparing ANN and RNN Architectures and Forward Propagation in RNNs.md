## Comparing ANN and RNN Architectures and Forward Propagation in RNNs
Slide 1: Introduction to ANN and RNN

Artificial Neural Networks (ANNs) and Recurrent Neural Networks (RNNs) are two fundamental architectures in deep learning. ANNs are designed for static data, while RNNs excel at processing sequential information. This presentation will explore their differences and delve into the forward propagation process in RNNs.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_network(ax, is_rnn=False):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Input layer
    ax.add_patch(plt.Circle((1, 1), 0.2, fc='lightblue'))
    ax.add_patch(plt.Circle((1, 2), 0.2, fc='lightblue'))
    ax.add_patch(plt.Circle((1, 3), 0.2, fc='lightblue'))
    
    # Hidden layer
    ax.add_patch(plt.Circle((2, 1.5), 0.2, fc='lightgreen'))
    ax.add_patch(plt.Circle((2, 2.5), 0.2, fc='lightgreen'))
    
    # Output layer
    ax.add_patch(plt.Circle((3, 2), 0.2, fc='salmon'))
    
    # Connections
    for i in [1, 2, 3]:
        for j in [1.5, 2.5]:
            ax.arrow(1.2, i, 0.6, j-i, head_width=0.05, fc='gray', ec='gray')
    
    for i in [1.5, 2.5]:
        ax.arrow(2.2, i, 0.6, 2-i, head_width=0.05, fc='gray', ec='gray')
    
    if is_rnn:
        ax.add_patch(plt.Circle((2, 3.5), 0.2, fc='yellow'))
        ax.arrow(2.2, 3.5, -0.2, -1, head_width=0.05, fc='red', ec='red')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
plot_network(ax1)
ax1.set_title('ANN')
plot_network(ax2, is_rnn=True)
ax2.set_title('RNN')
plt.tight_layout()
plt.show()
```

Slide 2: ANN Architecture

An Artificial Neural Network (ANN) consists of interconnected layers of neurons. The input layer receives data, which is then processed through one or more hidden layers before reaching the output layer. Each connection between neurons has an associated weight, and neurons typically apply an activation function to their inputs.

```python
import numpy as np

class ANN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Example usage
ann = ANN(input_size=3, hidden_size=4, output_size=2)
input_data = np.array([[0.1, 0.2, 0.3]])
output = ann.forward(input_data)
print("ANN output:", output)
```

Slide 3: RNN Architecture

Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining an internal state or "memory". Unlike ANNs, RNNs have connections that loop back, allowing information to persist across time steps. This recurrent structure enables RNNs to process inputs of varying lengths and capture temporal dependencies.

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        self.last_inputs = inputs
        self.last_hs = { 0: h }
        
        for i, x in enumerate(inputs):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.last_hs[i + 1] = h
        
        y = np.dot(self.Why, h) + self.by
        return y, h

# Example usage
rnn = SimpleRNN(input_size=3, hidden_size=4, output_size=2)
input_sequence = [np.array([[0.1], [0.2], [0.3]]), np.array([[0.2], [0.3], [0.4]])]
output, final_hidden_state = rnn.forward(input_sequence)
print("RNN output:", output)
print("Final hidden state:", final_hidden_state)
```

Slide 4: Key Differences: ANN vs RNN

The main distinction between ANNs and RNNs lies in their ability to handle temporal dependencies. ANNs process each input independently, making them suitable for tasks with fixed-size inputs like image classification. RNNs, on the other hand, maintain a hidden state that evolves over time, allowing them to capture information from previous inputs. This makes RNNs ideal for sequential data processing tasks such as natural language processing and time series analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def process_sequence(model, sequence, is_rnn=False):
    outputs = []
    hidden_state = None
    for input_data in sequence:
        if is_rnn:
            output, hidden_state = model.forward(input_data, hidden_state)
        else:
            output = model.forward(input_data)
        outputs.append(output)
    return outputs

class SimpleANN:
    def forward(self, x):
        return np.sum(x)  # Simplified for demonstration

class SimpleRNN:
    def forward(self, x, h=None):
        if h is None:
            h = 0
        h = 0.5 * h + x  # Simplified RNN update
        return h, h

# Generate a sequence
sequence = np.random.rand(10, 1)

# Process with ANN and RNN
ann = SimpleANN()
rnn = SimpleRNN()
ann_outputs = process_sequence(ann, sequence)
rnn_outputs = process_sequence(rnn, sequence, is_rnn=True)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(ann_outputs, label='ANN output')
plt.plot(rnn_outputs, label='RNN output')
plt.xlabel('Time step')
plt.ylabel('Output')
plt.legend()
plt.title('ANN vs RNN: Processing a Sequence')
plt.show()
```

Slide 5: Forward Propagation in RNNs

Forward propagation in RNNs involves passing input data through the network to generate outputs at each time step. The process updates the hidden state, which serves as the network's memory. At each time step, the RNN combines the current input with the previous hidden state to produce a new hidden state and output.

```python
import numpy as np

def rnn_forward(x, h_prev, parameters):
    Wax, Waa, Wya, ba, by = parameters
    
    a = np.tanh(np.dot(Wax, x) + np.dot(Waa, h_prev) + ba)
    y = np.dot(Wya, a) + by
    
    return a, y

# Initialize parameters
hidden_size, input_size, output_size = 4, 3, 2
Wax = np.random.randn(hidden_size, input_size)
Waa = np.random.randn(hidden_size, hidden_size)
Wya = np.random.randn(output_size, hidden_size)
ba = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))
parameters = [Wax, Waa, Wya, ba, by]

# Example input sequence
x = np.random.randn(input_size, 1)
h_prev = np.zeros((hidden_size, 1))

# Perform forward propagation
a, y = rnn_forward(x, h_prev, parameters)
print("Hidden state:", a)
print("Output:", y)
```

Slide 6: Mathematical Representation of RNN Forward Propagation

The RNN forward pass can be represented by two key equations:

1. Hidden state update: h(t) = tanh(W\_ax \* x(t) + W\_aa \* h(t-1) + b\_a)
2. Output: y(t) = W\_ya \* h(t) + b\_y

Where h(t) is the hidden state at time t, x(t) is the input at time t, W\_ax, W\_aa, and W\_ya are weight matrices, and b\_a and b\_y are bias vectors. The tanh function is commonly used as the activation function for the hidden state.

```python
import numpy as np

def rnn_forward_math(x_sequence, h0, parameters):
    Wax, Waa, Wya, ba, by = parameters
    h, y = {}, {}
    h[-1] = np.(h0)
    
    for t in range(len(x_sequence)):
        h[t] = np.tanh(np.dot(Wax, x_sequence[t]) + np.dot(Waa, h[t-1]) + ba)
        y[t] = np.dot(Wya, h[t]) + by
    
    return h, y

# Initialize parameters (same as previous slide)
hidden_size, input_size, output_size = 4, 3, 2
Wax = np.random.randn(hidden_size, input_size)
Waa = np.random.randn(hidden_size, hidden_size)
Wya = np.random.randn(output_size, hidden_size)
ba = np.zeros((hidden_size, 1))
by = np.zeros((output_size, 1))
parameters = [Wax, Waa, Wya, ba, by]

# Example input sequence
x_sequence = [np.random.randn(input_size, 1) for _ in range(3)]
h0 = np.zeros((hidden_size, 1))

# Perform forward propagation
h, y = rnn_forward_math(x_sequence, h0, parameters)
for t in range(len(x_sequence)):
    print(f"Time step {t}:")
    print("Hidden state:", h[t])
    print("Output:", y[t])
    print()
```

Slide 7: Vanishing and Exploding Gradients in RNNs

RNNs face challenges when learning long-term dependencies due to the vanishing and exploding gradient problems. These issues arise from the repeated multiplication of gradients during backpropagation through time. Vanishing gradients occur when the gradient becomes extremely small, making it difficult for the network to learn from distant past information. Exploding gradients happen when the gradient grows exponentially, leading to unstable training.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_gradient_flow(num_time_steps, initial_gradient):
    gradients = [initial_gradient]
    for _ in range(num_time_steps - 1):
        new_gradient = gradients[-1] * np.random.uniform(0.5, 1.5)
        gradients.append(new_gradient)
    return gradients

num_time_steps = 100
initial_gradient = 1.0

vanishing_gradients = simulate_gradient_flow(num_time_steps, initial_gradient)
exploding_gradients = simulate_gradient_flow(num_time_steps, initial_gradient)

plt.figure(figsize=(12, 6))
plt.plot(vanishing_gradients, label='Vanishing Gradient')
plt.plot(exploding_gradients, label='Exploding Gradient')
plt.xlabel('Time Steps')
plt.ylabel('Gradient Magnitude')
plt.title('Vanishing and Exploding Gradients in RNNs')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 8: Long Short-Term Memory (LSTM) Networks

Long Short-Term Memory (LSTM) networks are a type of RNN designed to mitigate the vanishing gradient problem. LSTMs introduce a more complex structure with gates that control the flow of information. These gates allow the network to selectively remember or forget information over long sequences, making LSTMs particularly effective for tasks involving long-term dependencies.

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
    
    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))
        
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
        c = f * c_prev + i * c_tilde
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        h = o * np.tanh(c)
        
        return h, c
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
lstm_cell = LSTMCell(input_size=3, hidden_size=4)
x = np.random.randn(3, 1)
h_prev = np.zeros((4, 1))
c_prev = np.zeros((4, 1))

h, c = lstm_cell.forward(x, h_prev, c_prev)
print("New hidden state:", h)
print("New memory cell state:", c)
```

Slide 9: Gated Recurrent Units (GRU)

Gated Recurrent Units (GRU) are another variation of RNNs designed to solve the vanishing gradient problem. GRUs are simpler than LSTMs, using only two gates: a reset gate and an update gate. This simplification makes GRUs computationally more efficient while still capturing long-term dependencies effectively.

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
        concat = np.vstack((h_prev, x))
        
        z = self.sigmoid(np.dot(self.Wz, concat) + self.bz)
        r = self.sigmoid(np.dot(self.Wr, concat) + self.br)
        h_tilde = np.tanh(np.dot(self.Wh, np.vstack((r * h_prev, x))) + self.bh)
        h = (1 - z) * h_prev + z * h_tilde
        
        return h
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
gru_cell = GRUCell(input_size=3, hidden_size=4)
x = np.random.randn(3, 1)
h_prev = np.zeros((4, 1))

h = gru_cell.forward(x, h_prev)
print("New hidden state:", h)
```

Slide 10: Bidirectional RNNs

Bidirectional RNNs process input sequences in both forward and backward directions, allowing the network to capture context from both past and future states. This architecture is particularly useful for tasks where the entire input sequence is available, such as speech recognition or text translation.

```python
import numpy as np

class BidirectionalRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.forward_rnn = SimpleRNN(input_size, hidden_size)
        self.backward_rnn = SimpleRNN(input_size, hidden_size)
        self.output_layer = np.random.randn(output_size, 2 * hidden_size)
    
    def forward(self, sequence):
        forward_states = self.forward_rnn.forward(sequence)
        backward_states = self.backward_rnn.forward(sequence[::-1])[::-1]
        
        combined_states = np.concatenate((forward_states, backward_states), axis=1)
        output = np.dot(self.output_layer, combined_states.T).T
        
        return output

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
    
    def forward(self, sequence):
        h = np.zeros((self.Whh.shape[0], 1))
        states = []
        
        for x in sequence:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            states.append(h)
        
        return np.array(states)

# Example usage
input_size, hidden_size, output_size = 3, 4, 2
brnn = BidirectionalRNN(input_size, hidden_size, output_size)

sequence = [np.random.randn(input_size, 1) for _ in range(5)]
output = brnn.forward(sequence)
print("Bidirectional RNN output shape:", output.shape)
```

Slide 11: Attention Mechanism in RNNs

The attention mechanism allows RNNs to focus on different parts of the input sequence when producing each output. This approach has been particularly successful in tasks like machine translation and image captioning, where certain input elements are more relevant for generating specific outputs.

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
sequence_length, hidden_size = 5, 4
query = np.random.randn(1, hidden_size)
keys = np.random.randn(sequence_length, hidden_size)
values = np.random.randn(sequence_length, hidden_size)

context, weights = attention(query, keys, values)
print("Context vector shape:", context.shape)
print("Attention weights:", weights)
```

Slide 12: Real-life Example: Sentiment Analysis with RNNs

Sentiment analysis is a common application of RNNs in natural language processing. In this example, we'll use a simple RNN to classify the sentiment of movie reviews as positive or negative.

```python
import numpy as np

class SentimentRNN:
    def __init__(self, vocab_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = np.zeros((self.Whh.shape[0], 1))
        for x in inputs:
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return y

# Example usage
vocab_size, hidden_size, output_size = 1000, 128, 2
sentiment_rnn = SentimentRNN(vocab_size, hidden_size, output_size)

# Simulate a movie review (sequence of word indices)
review = np.random.randint(0, vocab_size, size=(20, 1))
review_vectors = np.eye(vocab_size)[review.flatten()].T

sentiment_score = sentiment_rnn.forward(review_vectors)
sentiment = "Positive" if sentiment_score[0] > sentiment_score[1] else "Negative"
print(f"Sentiment: {sentiment}")
```

Slide 13: Real-life Example: Time Series Prediction with RNNs

RNNs are well-suited for time series prediction tasks, such as forecasting weather patterns or stock prices. In this example, we'll use a simple RNN to predict the next value in a time series.

```python
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
    
    def forward(self, x, h):
        h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
        y = np.dot(self.Why, h) + self.by
        return y, h

# Generate a simple time series
time_steps = 100
time_series = np.sin(np.linspace(0, 4*np.pi, time_steps)) + np.random.randn(time_steps) * 0.1

# Prepare data for RNN
X = time_series[:-1].reshape(-1, 1)
y = time_series[1:].reshape(-1, 1)

# Initialize and use RNN for prediction
rnn = TimeSeriesRNN(input_size=1, hidden_size=10, output_size=1)
h = np.zeros((10, 1))
predictions = []

for i in range(len(X)):
    pred, h = rnn.forward(X[i].reshape(-1, 1), h)
    predictions.append(pred[0, 0])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_series, label='Actual')
plt.plot(range(1, len(predictions)+1), predictions, label='Predicted')
plt.legend()
plt.title('Time Series Prediction with RNN')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into RNNs and their applications, here are some recommended resources:

1. "Sequence Models" course by Andrew Ng on Coursera
2. "Deep Learning" book by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
3. "Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)" by Christopher Olah (blog post)
4. ArXiv paper: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Cho et al. (2014) URL: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
5. ArXiv paper: "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" by Chung et al. (2014) URL: [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)

These resources provide a mix of theoretical foundations and practical implementations of RNNs and their variants.
