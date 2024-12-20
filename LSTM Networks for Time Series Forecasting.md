## LSTM Networks for Time Series Forecasting
Slide 1: LSTM Architecture Fundamentals

LSTM networks fundamentally differ from traditional RNNs through their sophisticated gating mechanisms and memory cell structure. The architecture employs three gates: input, forget, and output, working in conjunction with a memory cell to regulate information flow through the network.

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weight matrices and biases
        self.hidden_size = hidden_size
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        
        # Initialize bias terms
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
```

Slide 2: LSTM Forward Pass Implementation

The forward pass in an LSTM involves computing the gates' activations and updating the cell state. This process maintains long-term dependencies through careful regulation of information flow, utilizing sigmoid and tanh activation functions.

```python
def forward(self, x, prev_h, prev_c):
    # Concatenate input and previous hidden state
    combined = np.vstack((x, prev_h))
    
    # Compute gate activations
    f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
    i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
    o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
    
    # Compute candidate cell state
    c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
    
    # Update cell state and hidden state
    c = f * prev_c + i * c_tilde
    h = o * np.tanh(c)
    
    return h, c
```

Slide 3: LSTM Mathematical Foundations

The core mathematical operations within an LSTM cell determine how information flows through the network. These equations represent the fundamental computations for each gate and state update within the cell.

```python
# Mathematical formulations for LSTM
"""
Input gate:     $$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$
Forget gate:    $$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$
Output gate:    $$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$$
Cell candidate: $$\tilde{c_t} = \tanh(W_c[h_{t-1}, x_t] + b_c)$$
Cell state:     $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c_t}$$
Hidden state:   $$h_t = o_t \odot \tanh(c_t)$$
"""
```

Slide 4: Helper Functions Implementation

Helper functions are essential components for LSTM operations, implementing activation functions and their derivatives for both forward propagation and backpropagation training processes.

```python
class LSTMCell:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        s = 1 / (1 + np.exp(-x))
        return s * (1 - s)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x) ** 2
    
    def init_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in [self.Wf, self.Wi, self.Wo, self.Wc]:
            weight.uniform_(-stdv, stdv)
```

Slide 5: Time Series Prediction Setup

Time series prediction represents one of the most common applications of LSTM networks. The preprocessing phase involves data normalization and sequence creation for effective training.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_time_series(data, sequence_length):
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    
    return np.array(X), np.array(y), scaler
```

Slide 6: LSTM Training Loop

The training process for LSTM networks requires careful management of sequences and state preservation. This implementation demonstrates a complete training loop with batch processing and gradient updates.

```python
def train_lstm(model, X_train, y_train, epochs, batch_size):
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        h_state = np.zeros((batch_size, model.hidden_size))
        c_state = np.zeros((batch_size, model.hidden_size))
        
        for batch_x, batch_y in get_batches(X_train, y_train, batch_size):
            # Forward pass
            h_state, c_state = model.forward(batch_x, h_state, c_state)
            loss = model.compute_loss(h_state, batch_y)
            
            # Backward pass
            gradients = model.backward(batch_x, batch_y, h_state)
            model.update_parameters(gradients, learning_rate=0.01)
            
            total_loss += loss
            
        losses.append(total_loss / len(X_train))
        print(f'Epoch {epoch + 1}, Loss: {losses[-1]:.4f}')
```

Slide 7: Stock Price Prediction Implementation

Stock market prediction represents a practical application of LSTM networks, requiring specialized data preprocessing and model configuration for financial time series analysis.

```python
import yfinance as yf
import numpy as np

def prepare_stock_data(symbol, start_date, end_date, sequence_length):
    # Download stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    # Extract features
    features = np.column_stack([
        stock_data['Close'].values,
        stock_data['Volume'].values,
        stock_data['High'].values - stock_data['Low'].values
    ])
    
    # Normalize features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i + sequence_length])
        y.append(scaled_features[i + sequence_length, 0])  # Predict next day's close price
        
    return np.array(X), np.array(y), scaler
```

Slide 8: LSTM Cell State Management

The cell state acts as the network's memory, carefully regulated by the gates. This implementation shows how the cell state is managed and updated during forward propagation.

```python
def manage_cell_state(self, x_t, prev_h, prev_c):
    # Concatenate input with previous hidden state
    combined = np.concatenate([x_t, prev_h], axis=1)
    
    # Compute all gates in parallel
    gates = np.dot(combined, self.W_gates.T) + self.b_gates
    
    # Split gates into individual components
    i, f, o, g = np.split(gates, 4, axis=1)
    
    # Apply activations
    i = self.sigmoid(i)  # input gate
    f = self.sigmoid(f)  # forget gate
    o = self.sigmoid(o)  # output gate
    g = np.tanh(g)      # cell candidate
    
    # Update cell state
    c = f * prev_c + i * g
    
    # Compute hidden state
    h = o * np.tanh(c)
    
    return h, c, (i, f, o, g)
```

Slide 9: Bidirectional LSTM Implementation

Bidirectional LSTMs process sequences in both forward and backward directions, capturing patterns that might be missed in unidirectional processing.

```python
class BidirectionalLSTM:
    def __init__(self, input_size, hidden_size):
        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        
    def forward(self, X):
        batch_size, seq_length, _ = X.shape
        
        # Forward direction
        forward_h = np.zeros((batch_size, seq_length, self.hidden_size))
        forward_c = np.zeros((batch_size, seq_length, self.hidden_size))
        
        # Backward direction
        backward_h = np.zeros((batch_size, seq_length, self.hidden_size))
        backward_c = np.zeros((batch_size, seq_length, self.hidden_size))
        
        h_state = np.zeros((batch_size, self.hidden_size))
        c_state = np.zeros((batch_size, self.hidden_size))
        
        # Process sequence in both directions
        for t in range(seq_length):
            forward_h[:, t], forward_c[:, t] = self.forward_lstm(
                X[:, t], h_state, c_state)
                
        for t in range(seq_length-1, -1, -1):
            backward_h[:, t], backward_c[:, t] = self.backward_lstm(
                X[:, t], h_state, c_state)
        
        # Concatenate results
        return np.concatenate([forward_h, backward_h], axis=2)
```

Slide 10: Natural Language Processing with LSTM

LSTM networks excel in processing sequential text data, making them ideal for NLP tasks. This implementation demonstrates text preprocessing and sequence generation for language modeling.

```python
class TextLSTM:
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.lstm = LSTMCell(embedding_dim, hidden_size)
        self.output_layer = np.random.randn(hidden_size, vocab_size) * 0.01
        
    def preprocess_text(self, text, sequence_length):
        # Create vocabulary
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Create sequences
        sequences = []
        next_chars = []
        for i in range(0, len(text) - sequence_length):
            sequences.append(text[i:i + sequence_length])
            next_chars.append(text[i + sequence_length])
            
        # Vectorize sequences
        X = np.zeros((len(sequences), sequence_length, len(chars)))
        y = np.zeros((len(sequences), len(chars)))
        
        for i, sequence in enumerate(sequences):
            for t, char in enumerate(sequence):
                X[i, t, char_to_idx[char]] = 1
            y[i, char_to_idx[next_chars[i]]] = 1
            
        return X, y, char_to_idx, idx_to_char
```

Slide 11: LSTM Dropout Implementation

Dropout is a crucial regularization technique for preventing overfitting in LSTM networks. This implementation shows how to apply dropout to both inputs and recurrent connections.

```python
class LSTMWithDropout:
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        self.lstm = LSTMCell(input_size, hidden_size)
        self.dropout_rate = dropout_rate
        self.training = True
        
    def forward(self, x, prev_h, prev_c):
        if self.training:
            # Create dropout masks
            input_mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            recurrent_mask = (np.random.rand(*prev_h.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            
            # Apply dropout
            x = x * input_mask
            prev_h = prev_h * recurrent_mask
            
        # Regular forward pass with dropped out values
        h, c = self.lstm.forward(x, prev_h, prev_c)
        return h, c

    def eval(self):
        self.training = False

    def train(self):
        self.training = True
```

Slide 12: LSTM Attention Mechanism

Attention mechanisms enhance LSTM's ability to focus on relevant parts of input sequences. This implementation demonstrates self-attention for sequence processing.

```python
class LSTMWithAttention:
    def __init__(self, input_size, hidden_size, attention_size):
        self.lstm = LSTMCell(input_size, hidden_size)
        self.attention_weights = np.random.randn(attention_size, hidden_size)
        self.attention_combine = np.random.randn(hidden_size * 2, hidden_size)
        
    def attention_score(self, hidden_states):
        # Calculate attention scores
        scores = np.dot(hidden_states, self.attention_weights.T)
        scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        scores = scores / np.sum(scores, axis=1, keepdims=True)
        
        # Apply attention to hidden states
        context = np.sum(hidden_states * scores[:, :, np.newaxis], axis=1)
        return context, scores
        
    def forward(self, x_sequence):
        batch_size, seq_length, _ = x_sequence.shape
        hidden_states = np.zeros((batch_size, seq_length, self.lstm.hidden_size))
        
        # Process sequence
        h = np.zeros((batch_size, self.lstm.hidden_size))
        c = np.zeros((batch_size, self.lstm.hidden_size))
        
        for t in range(seq_length):
            h, c = self.lstm.forward(x_sequence[:, t], h, c)
            hidden_states[:, t] = h
            
        # Apply attention
        context, attention_weights = self.attention_score(hidden_states)
        
        # Combine context with final hidden state
        combined = np.concatenate([h, context], axis=1)
        output = np.tanh(np.dot(combined, self.attention_combine))
        
        return output, attention_weights
```

Slide 13: Performance Metrics and Evaluation

LSTM networks require comprehensive evaluation metrics to assess their performance across different tasks. This implementation provides a suite of evaluation functions for both regression and classification tasks.

```python
class LSTMEvaluator:
    def __init__(self):
        self.metrics_history = defaultdict(list)
    
    def evaluate_regression(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics
    
    def evaluate_sequence(self, y_true, y_pred, vocab_size):
        # One-hot encode predictions
        y_pred_oh = np.eye(vocab_size)[np.argmax(y_pred, axis=1)]
        
        # Calculate accuracy
        accuracy = np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        
        # Calculate perplexity
        epsilon = 1e-10
        cross_entropy = -np.sum(y_true * np.log(y_pred + epsilon)) / len(y_true)
        perplexity = np.exp(cross_entropy)
        
        return {
            'accuracy': accuracy,
            'perplexity': perplexity,
            'cross_entropy': cross_entropy
        }
```

Slide 14: Model Checkpointing and State Management

Proper model state management is crucial for training large LSTM networks. This implementation provides functionality for saving and loading model states, enabling training resumption and model deployment.

```python
class LSTMStateManager:
    def __init__(self, model, checkpoint_dir='checkpoints'):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch, optimizer_state, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state': {
                'Wf': self.model.Wf,
                'Wi': self.model.Wi,
                'Wo': self.model.Wo,
                'Wc': self.model.Wc,
                'bf': self.model.bf,
                'bi': self.model.bi,
                'bo': self.model.bo,
                'bc': self.model.bc
            },
            'optimizer_state': optimizer_state,
            'metrics': metrics
        }
        
        filename = f'checkpoint_epoch_{epoch}.pkl'
        path = os.path.join(self.checkpoint_dir, filename)
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
    def load_checkpoint(self, epoch):
        filename = f'checkpoint_epoch_{epoch}.pkl'
        path = os.path.join(self.checkpoint_dir, filename)
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        # Restore model state
        for key, value in checkpoint['model_state'].items():
            setattr(self.model, key, value)
            
        return checkpoint['optimizer_state'], checkpoint['metrics']
```

Slide 15: Additional Resources

*   "Long Short-Term Memory Networks for Text Generation"
*   [https://arxiv.org/abs/1909.03858](https://arxiv.org/abs/1909.03858)
*   "Attention Mechanisms in LSTM Networks: A Comprehensive Survey"
*   [https://arxiv.org/abs/2001.11955](https://arxiv.org/abs/2001.11955)
*   "Bidirectional LSTM-CRF Models for Sequence Tagging"
*   [https://arxiv.org/abs/1508.01991](https://arxiv.org/abs/1508.01991)
*   "Dropout for LSTMs: Improving Neural Network Regularization"
*   [https://arxiv.org/abs/1512.05287](https://arxiv.org/abs/1512.05287)
*   "LSTM Neural Networks for Language Modeling"
*   [https://arxiv.org/abs/1902.07229](https://arxiv.org/abs/1902.07229)

