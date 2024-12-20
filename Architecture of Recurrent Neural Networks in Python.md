## Architecture of Recurrent Neural Networks in Python
Slide 1: Introduction to Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are a class of artificial neural networks designed to work with sequential data. Unlike feedforward networks, RNNs have loops that allow information to persist, making them ideal for tasks involving time series, natural language processing, and speech recognition.

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

# Example usage
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
inputs = [np.random.randn(10, 1) for _ in range(5)]  # 5 time steps
outputs, final_hidden_state = rnn.forward(inputs)
print(f"Number of outputs: {len(outputs)}")
print(f"Shape of final output: {outputs[-1].shape}")
print(f"Shape of final hidden state: {final_hidden_state.shape}")
```

Slide 2: The Architecture of RNNs

RNNs consist of a repeating module of neural network cells. Each cell takes input from the current time step and the hidden state from the previous time step. This architecture allows the network to maintain a form of memory, enabling it to process sequences of data.

```python
import torch
import torch.nn as nn

class BasicRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BasicRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.combined_size = input_size + hidden_size
        
        self.i2h = nn.Linear(self.combined_size, hidden_size)
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        return hidden

# Example usage
input_size = 10
hidden_size = 20
batch_size = 1
seq_length = 5

cell = BasicRNNCell(input_size, hidden_size)
hidden = torch.zeros(batch_size, hidden_size)
input_seq = torch.randn(seq_length, batch_size, input_size)

for i in range(seq_length):
    hidden = cell(input_seq[i], hidden)

print(f"Final hidden state shape: {hidden.shape}")
```

Slide 3: Unrolling the RNN

When processing a sequence, an RNN can be thought of as multiple copies of the same network, each passing a message to a successor. This unrolled view helps in understanding how RNNs handle sequences of varying lengths.

```python
import torch
import torch.nn as nn

class UnrolledRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(UnrolledRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
seq_length = 5
batch_size = 3

model = UnrolledRNN(input_size, hidden_size, num_layers)
input_seq = torch.randn(batch_size, seq_length, input_size)
output = model(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
```

Slide 4: Backpropagation Through Time (BPTT)

BPTT is the algorithm used to train RNNs. It unfolds the network through time and applies backpropagation to each time step. This allows the network to learn long-term dependencies in the data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Training loop with BPTT
def train_rnn(model, X, y, num_epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example usage
input_size = 1
hidden_size = 10
output_size = 1
seq_length = 20
num_epochs = 100
lr = 0.01

X = torch.sin(torch.linspace(0, 10, steps=seq_length)).unsqueeze(0).unsqueeze(2)
y = torch.sin(torch.linspace(0.1, 10.1, steps=seq_length)).unsqueeze(0)

model = SimpleRNN(input_size, hidden_size, output_size)
train_rnn(model, X, y, num_epochs, lr)
```

Slide 5: Vanishing and Exploding Gradients

RNNs often struggle with long-term dependencies due to vanishing or exploding gradients during training. This occurs when gradients become extremely small or large as they propagate through many time steps.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def simulate_gradient_flow(sequence_length):
    model = nn.RNN(input_size=1, hidden_size=1, num_layers=1, batch_first=True)
    
    # Initialize weights to 1 for demonstration
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.constant_(param, 1)
    
    input_sequence = torch.ones(1, sequence_length, 1)
    output, _ = model(input_sequence)
    
    loss = output.sum()
    loss.backward()
    
    gradients = []
    for t in range(sequence_length):
        grad = model.weight_hh_l0.grad[0, 0].item() ** t
        gradients.append(grad)
    
    return gradients

# Simulate gradient flow for different sequence lengths
seq_lengths = [10, 50, 100]
plt.figure(figsize=(10, 6))

for seq_len in seq_lengths:
    grads = simulate_gradient_flow(seq_len)
    plt.plot(range(seq_len), grads, label=f'Sequence length: {seq_len}')

plt.xlabel('Time steps')
plt.ylabel('Gradient magnitude')
plt.title('Gradient Flow in RNN')
plt.legend()
plt.yscale('log')
plt.show()
```

Slide 6: Long Short-Term Memory (LSTM) Networks

LSTMs are a special kind of RNN designed to avoid the long-term dependency problem. They use a more complex structure with gates to control the flow of information, allowing them to remember or forget information selectively.

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1
seq_length = 5
batch_size = 3

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
input_seq = torch.randn(batch_size, seq_length, input_size)
output = model(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
```

Slide 7: Gated Recurrent Units (GRUs)

GRUs are another variant of RNNs designed to solve the vanishing gradient problem. They are similar to LSTMs but with a simpler structure, using only two gates: reset and update gates.

```python
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1
seq_length = 5
batch_size = 3

model = GRUModel(input_size, hidden_size, num_layers, output_size)
input_seq = torch.randn(batch_size, seq_length, input_size)
output = model(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
```

Slide 8: Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions, allowing the network to capture context from both past and future states. This is particularly useful in tasks where the entire sequence is available at once, such as in natural language processing.

```python
import torch
import torch.nn as nn

class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BidirectionalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because of bidirectional
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1
seq_length = 5
batch_size = 3

model = BidirectionalRNN(input_size, hidden_size, num_layers, output_size)
input_seq = torch.randn(batch_size, seq_length, input_size)
output = model(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
```

Slide 9: Attention Mechanism in RNNs

The attention mechanism allows RNNs to focus on different parts of the input sequence when producing each output. This helps in handling long sequences and capturing important dependencies regardless of their position in the sequence.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        output, _ = self.rnn(x)
        
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
        
        # Apply attention
        context = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        
        # Final prediction
        out = self.fc(context)
        return out, attention_weights

# Example usage
input_size = 10
hidden_size = 20
output_size = 1
seq_length = 5
batch_size = 3

model = AttentionRNN(input_size, hidden_size, output_size)
input_seq = torch.randn(batch_size, seq_length, input_size)
output, attention_weights = model(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

Slide 10: Real-life Example: Sentiment Analysis

RNNs are widely used in Natural Language Processing tasks such as sentiment analysis. Here's a simplified example of using an LSTM for sentiment classification of movie reviews.

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# Example usage
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # Binary classification (positive/negative)

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim)
sample_input = torch.randint(0, vocab_size, (1, 20))  # Batch of 1, sequence length 20
output = model(sample_input)

print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

Slide 11: Real-life Example: Time Series Forecasting

RNNs are effective for time series forecasting tasks, such as predicting future values based on historical data. Here's a simple example of using an LSTM for temperature prediction.

```python
import torch
import torch.nn as nn

class TemperatureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TemperatureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 1  # One feature (temperature)
hidden_size = 64
num_layers = 2
output_size = 1  # Predict next temperature
seq_length = 30  # Use 30 days of historical data

model = TemperatureLSTM(input_size, hidden_size, num_layers, output_size)
sample_input = torch.randn(1, seq_length, input_size)  # Batch of 1, 30 days of data
prediction = model(sample_input)

print(f"Input shape: {sample_input.shape}")
print(f"Prediction shape: {prediction.shape}")
```

Slide 12: Sequence-to-Sequence Models

Sequence-to-sequence models use RNNs for tasks that involve transforming one sequence into another, such as machine translation or text summarization. These models typically consist of an encoder RNN and a decoder RNN.

```python
import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_vocab_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(source.device)
        _, (hidden, cell) = self.encoder(source)

        input = target[:, 0]

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoder(input.unsqueeze(1), (hidden, cell))
            prediction = self.fc(output.squeeze(1))
            outputs[:, t] = prediction
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input = target[:, t] if teacher_force else prediction.argmax(1)

        return outputs

# Example usage
input_vocab_size = 1000
output_vocab_size = 1000
hidden_size = 256
seq_length = 10
batch_size = 2

model = Seq2SeqModel(input_vocab_size, output_vocab_size, hidden_size)
source = torch.randint(0, input_vocab_size, (batch_size, seq_length))
target = torch.randint(0, output_vocab_size, (batch_size, seq_length))
output = model(source, target)

print(f"Source shape: {source.shape}")
print(f"Target shape: {target.shape}")
print(f"Output shape: {output.shape}")
```

Slide 13: Challenges and Future Directions

While RNNs have proven effective for many sequential tasks, they still face challenges such as difficulty in parallelization and capturing very long-term dependencies. Recent advancements like Transformer models have addressed some of these issues, but RNNs remain relevant for many applications.

Future directions for RNN research include:

1. Improving efficiency and parallelization
2. Developing hybrid architectures combining RNNs with other neural network types
3. Enhancing interpretability of RNN decisions
4. Adapting RNNs for continual learning scenarios

Slide 14: Challenges and Future Directions

```python
import torch
import torch.nn as nn

class HybridRNNTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, num_encoder_layers):
        super(HybridRNNTransformer, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_encoder_layers
        )
        
    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        transformer_out = self.transformer_encoder(rnn_out)
        return transformer_out

# Example usage
input_size = 10
hidden_size = 20
num_layers = 2
nhead = 4
num_encoder_layers = 2
seq_length = 5
batch_size = 3

model = HybridRNNTransformer(input_size, hidden_size, num_layers, nhead, num_encoder_layers)
input_seq = torch.randn(batch_size, seq_length, input_size)
output = model(input_seq)

print(f"Input shape: {input_seq.shape}")
print(f"Output shape: {output.shape}")
```

Slide 15: Additional Resources

For those interested in delving deeper into RNNs and their applications, here are some valuable resources:

1. "Long Short-Term Memory" by Hochreiter and Schmidhuber (1997) ArXiv: [https://arxiv.org/abs/1909.09586](https://arxiv.org/abs/1909.09586) (Note: This is a recent review paper, as the original 1997 paper predates ArXiv)
2. "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" by Cho et al. (2014) ArXiv: [https://arxiv.org/abs/1406.1078](https://arxiv.org/abs/1406.1078)
3. "Sequence to Sequence Learning with Neural Networks" by Sutskever et al. (2014) ArXiv: [https://arxiv.org/abs/1409.3215](https://arxiv.org/abs/1409.3215)
4. "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

