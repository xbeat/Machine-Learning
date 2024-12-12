## Recurrent Neural Networks for Sequential Data Processing
Slide 1: Basic RNN Architecture

A Recurrent Neural Network processes sequential data by maintaining a hidden state that gets updated with each input. The hidden state acts as the network's memory, allowing it to capture temporal dependencies in the data sequence. The basic RNN cell performs a transformation on both current input and previous hidden state.

```python
import numpy as np

class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        
        # Store dimensions
        self.hidden_size = hidden_size
        self.input_size = input_size
        
    def forward(self, x, h_prev):
        # Forward pass of RNN cell
        self.x = x
        self.h_prev = h_prev
        
        # Calculate new hidden state
        self.h_next = np.tanh(np.dot(self.Wxh, x) + 
                             np.dot(self.Whh, h_prev) + self.bh)
        return self.h_next
```

Slide 2: Mathematical Foundations of RNN

The core RNN computation involves matrix multiplications and a non-linear activation function. The hidden state at time t is computed using the current input and previous hidden state, following specific mathematical transformations that enable temporal learning.

```python
# Mathematical formulation of RNN in LaTeX notation
"""
Basic RNN equations:
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

Where:
- $$h_t$$ is the hidden state at time t
- $$x_t$$ is the input at time t
- $$W_{xh}$$ is input-to-hidden weight matrix
- $$W_{hh}$$ is hidden-to-hidden weight matrix
- $$W_{hy}$$ is hidden-to-output weight matrix
- $$b_h, b_y$$ are bias terms
"""
```

Slide 3: Backpropagation Through Time (BPTT)

BPTT is the algorithm used to train RNNs by unrolling the network through time and computing gradients. This process allows the network to learn from sequential data by propagating errors backwards through the temporal dimension while updating weights.

```python
def bptt(self, targets, learning_rate=0.1):
    # Initialize gradients
    dWxh = np.zeros_like(self.Wxh)
    dWhh = np.zeros_like(self.Whh)
    dbh = np.zeros_like(self.bh)
    dh_next = np.zeros_like(self.h_states[0])
    
    # Backward pass
    for t in reversed(range(len(targets))):
        # Gradient of loss with respect to hidden state
        dh = self.h_states[t] - targets[t]
        
        # Add gradient from next time step
        dh += dh_next
        
        # Compute gradients
        dWxh += np.dot(dh * (1 - self.h_states[t]**2), self.inputs[t].T)
        dWhh += np.dot(dh * (1 - self.h_states[t]**2), self.h_states[t-1].T)
        dbh += dh * (1 - self.h_states[t]**2)
        
        # Gradient for next iteration
        dh_next = np.dot(self.Whh.T, dh * (1 - self.h_states[t]**2))
    
    # Update weights
    self.Wxh -= learning_rate * dWxh
    self.Whh -= learning_rate * dWhh
    self.bh -= learning_rate * dbh
```

Slide 4: Implementing LSTM Cell

The Long Short-Term Memory (LSTM) cell is an advanced RNN architecture that solves the vanishing gradient problem using gates to control information flow. This implementation shows the core LSTM components including forget, input, and output gates.

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights for gates
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        # Initialize biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
    def forward(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.vstack((h_prev, x))
        
        # Calculate gates
        f = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
        o = self.sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # Update cell state and hidden state
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c
```

Slide 5: Character-Level Language Model

This implementation demonstrates a practical application of RNNs in natural language processing by creating a character-level language model that can generate text by learning patterns in the input sequence.

```python
class CharRNN:
    def __init__(self, vocab_size, hidden_size):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))
        
    def forward(self, inputs, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        
        # Forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1  # One-hot encode input
            
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + 
                           np.dot(self.Whh, hs[t-1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            
        return xs, hs, ps
```

Slide 6: Time Series Prediction with RNN

A practical implementation of RNN for time series forecasting, demonstrating how to process sequential numerical data. This model can be used for predicting future values based on historical patterns in financial, weather, or any temporal data.

```python
class TimeSeriesRNN:
    def __init__(self, input_size, hidden_size, sequence_length):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        
        # Initialize weights for time series processing
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(input_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((input_size, 1))
        
    def prepare_sequence(self, data):
        # Prepare sequences for training
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
```

Slide 7: Bidirectional RNN Implementation

Bidirectional RNNs process sequences in both forward and backward directions, capturing patterns that might be missed in unidirectional processing. This implementation shows how to combine information from both directions for improved sequence understanding.

```python
class BidirectionalRNN:
    def __init__(self, input_size, hidden_size):
        # Initialize forward and backward RNN components
        self.forward_rnn = SimpleRNNCell(input_size, hidden_size//2)
        self.backward_rnn = SimpleRNNCell(input_size, hidden_size//2)
        self.hidden_size = hidden_size
        
    def forward(self, sequence):
        # Process sequence in both directions
        forward_states = []
        backward_states = []
        h_forward = np.zeros((self.hidden_size//2, 1))
        h_backward = np.zeros((self.hidden_size//2, 1))
        
        # Forward pass
        for x in sequence:
            h_forward = self.forward_rnn.forward(x, h_forward)
            forward_states.append(h_forward)
            
        # Backward pass
        for x in reversed(sequence):
            h_backward = self.backward_rnn.forward(x, h_backward)
            backward_states.append(h_backward)
            
        # Combine states
        combined_states = [np.vstack((f, b)) 
                         for f, b in zip(forward_states, 
                                       reversed(backward_states))]
        return combined_states
```

Slide 8: GRU Cell Implementation

The Gated Recurrent Unit (GRU) is a simplified variant of LSTM that maintains good performance while reducing computational complexity. This implementation shows the reset and update gates that control information flow.

```python
class GRUCell:
    def __init__(self, input_size, hidden_size):
        # Initialize update gate parameters
        self.Wz = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bz = np.zeros((hidden_size, 1))
        
        # Initialize reset gate parameters
        self.Wr = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.br = np.zeros((hidden_size, 1))
        
        # Initialize candidate state parameters
        self.Wh = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        
    def forward(self, x, h_prev):
        # Concatenate input and previous hidden state
        concat = np.vstack((x, h_prev))
        
        # Update gate
        z = self.sigmoid(np.dot(self.Wz, concat) + self.bz)
        
        # Reset gate
        r = self.sigmoid(np.dot(self.Wr, concat) + self.br)
        
        # Candidate state
        h_candidate = np.tanh(np.dot(self.Wh, 
                    np.vstack((x, r * h_prev))) + self.bh)
        
        # New hidden state
        h = z * h_prev + (1 - z) * h_candidate
        
        return h

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

Slide 9: Attention Mechanism Integration

The attention mechanism allows RNNs to focus on different parts of the input sequence when generating each output element. This implementation demonstrates how to add attention layers to enhance sequence processing capabilities.

```python
class AttentionRNN:
    def __init__(self, input_size, hidden_size, attention_size):
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Initialize attention weights
        self.Wa = np.random.randn(attention_size, hidden_size) * 0.01
        self.Ua = np.random.randn(attention_size, hidden_size) * 0.01
        self.va = np.random.randn(1, attention_size) * 0.01
        
    def attention_score(self, h_t, h_s):
        # Calculate attention scores
        score = np.tanh(np.dot(self.Wa, h_t) + np.dot(self.Ua, h_s))
        return np.dot(self.va, score)
    
    def forward(self, encoder_states, decoder_state):
        attention_weights = []
        
        # Calculate attention weights for each encoder state
        for encoder_state in encoder_states:
            score = self.attention_score(decoder_state, encoder_state)
            attention_weights.append(score)
            
        # Normalize attention weights
        attention_weights = np.exp(attention_weights)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Calculate context vector
        context = np.sum([w * s for w, s in 
                         zip(attention_weights, encoder_states)], axis=0)
        
        return context, attention_weights
```

Slide 10: Real-world Example: Stock Price Prediction

Implementation of a practical RNN model for predicting stock prices using historical data. This example includes data preprocessing, model training, and evaluation metrics calculation.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockPricePredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.rnn = TimeSeriesRNN(1, 64, sequence_length)
        
    def prepare_data(self, prices):
        # Normalize the data
        scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def train(self, prices, epochs=100, learning_rate=0.01):
        X, y = self.prepare_data(prices)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            h = np.zeros((64, 1))
            
            for seq, target in zip(X, y):
                # Forward pass
                for t in range(self.sequence_length):
                    h = self.rnn.forward(seq[t].reshape(-1,1), h)
                
                # Calculate loss and update weights
                predicted = self.rnn.predict(h)
                loss = np.mean((predicted - target)**2)
                epoch_loss += loss
                
                # Backward pass (gradient update)
                self.rnn.backward(target, learning_rate)
                
            losses.append(epoch_loss/len(X))
            
        return losses
```

Slide 11: Results Analysis for Stock Price Prediction

A comprehensive evaluation of the stock price prediction model's performance, showing various metrics and visualization of predictions versus actual values.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_predictions(model, test_data):
    # Generate predictions
    predictions = model.predict(test_data)
    
    # Calculate metrics
    mse = mean_squared_error(test_data, predictions)
    mae = mean_absolute_error(test_data, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test_data, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Stock Price Prediction Results')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    # Calculate prediction accuracy
    accuracy = np.mean(np.abs((predictions - test_data) / test_data)) * 100
    print(f"Prediction Accuracy: {100 - accuracy:.2f}%")
    
    return mse, mae, rmse, accuracy
```

Slide 12: Language Translation Implementation

A complete implementation of a sequence-to-sequence RNN model for language translation, demonstrating encoder-decoder architecture with attention mechanism for translating between two languages.

```python
class Seq2SeqTranslator:
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        self.encoder = RNNEncoder(input_vocab_size, hidden_size)
        self.decoder = RNNDecoder(output_vocab_size, hidden_size)
        self.attention = AttentionRNN(hidden_size, hidden_size, hidden_size)
        
    def translate(self, input_sequence, max_length=50):
        # Encode input sequence
        encoder_states, final_state = self.encoder.forward(input_sequence)
        
        # Initialize decoder
        decoder_input = self.get_start_token()
        decoder_state = final_state
        output_sequence = []
        
        for _ in range(max_length):
            # Calculate attention
            context, attention_weights = self.attention.forward(
                encoder_states, decoder_state)
            
            # Generate next token
            decoder_output, decoder_state = self.decoder.forward(
                decoder_input, decoder_state, context)
            
            # Get most probable token
            predicted_token = np.argmax(decoder_output)
            output_sequence.append(predicted_token)
            
            # Break if end token is generated
            if predicted_token == self.get_end_token():
                break
                
            decoder_input = predicted_token
            
        return output_sequence, attention_weights
        
    def get_start_token(self):
        return np.zeros((1, 1))
        
    def get_end_token(self):
        return 1
```

Slide 13: Advanced RNN Training Techniques

Implementation of advanced training methods including gradient clipping, teacher forcing, and scheduled sampling to improve RNN training stability and performance.

```python
class AdvancedRNNTrainer:
    def __init__(self, model, clip_value=5.0, teacher_forcing_ratio=0.5):
        self.model = model
        self.clip_value = clip_value
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
    def clip_gradients(self, gradients):
        """Implement gradient clipping to prevent exploding gradients"""
        norm = np.sqrt(sum([np.sum(grad ** 2) 
                          for grad in gradients.values()]))
        if norm > self.clip_value:
            scale = self.clip_value / norm
            return {k: v * scale for k, v in gradients.items()}
        return gradients
        
    def train_step(self, input_seq, target_seq, learning_rate=0.01):
        # Forward pass with teacher forcing
        use_teacher_forcing = np.random.random() < self.teacher_forcing_ratio
        
        if use_teacher_forcing:
            outputs = self.model.forward_teacher_forcing(input_seq, target_seq)
        else:
            outputs = self.model.forward(input_seq)
            
        # Calculate gradients
        gradients = self.model.backward(outputs, target_seq)
        
        # Clip gradients
        clipped_gradients = self.clip_gradients(gradients)
        
        # Update weights
        self.model.update_parameters(clipped_gradients, learning_rate)
        
        return outputs
```

Slide 14: Additional Resources

*   Natural Language Processing with RNNs: "Understanding LSTM Networks" - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
*   Advanced RNN Architectures: "Empirical Evaluation of Gated Recurrent Neural Networks" - [https://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)
*   Attention Mechanisms in RNNs: "Neural Machine Translation by Jointly Learning to Align and Translate" - [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
*   For practical implementations and tutorials, refer to:
    *   Deep Learning Specialization on Coursera
    *   FastAI Deep Learning Course
    *   PyTorch Documentation on RNN implementations
*   Recommended search terms for further research:
    *   "RNN architecture comparison"
    *   "LSTM vs GRU performance analysis"
    *   "Attention mechanisms in sequence models"
    *   "Transformer vs RNN benchmarks"

