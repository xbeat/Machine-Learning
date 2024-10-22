## RNN and LSTM for Stock Price Prediction
Slide 1: Understanding RNN Architecture for Time Series

In recurrent neural networks, hidden states propagate temporal dependencies through sequential data processing, enabling the network to learn patterns across different time steps while maintaining contextual information.

```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        # Initialize weights with random values
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((1, hidden_size))
        
    def forward(self, x, h_prev):
        # Forward pass computation
        h_next = np.tanh(np.dot(x, self.Wx) + np.dot(h_prev, self.Wh) + self.b)
        return h_next
```

Slide 2: LSTM Cell Architecture and Mathematics

The LSTM architecture introduces memory cells and gating mechanisms to control information flow, addressing the vanishing gradient problem inherent in simple RNNs through selective memory retention and update processes.

```python
# Mathematical formulation of LSTM gates
"""
$$i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$$
$$f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$$
$$o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$$
$$\tilde{c}_t = \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$
"""
```

Slide 3: LSTM Implementation for Time Series

```python
class LSTM:
    def __init__(self, input_size, hidden_size):
        # Initialize weights for gates
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        
    def forward_step(self, x, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = np.concatenate((x, h_prev), axis=1)
        
        # Compute gates
        f = self.sigmoid(np.dot(combined, self.Wf))
        i = self.sigmoid(np.dot(combined, self.Wi))
        c_tilde = np.tanh(np.dot(combined, self.Wc))
        o = self.sigmoid(np.dot(combined, self.Wo))
        
        # Update cell and hidden states
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c
```

Slide 4: Data Preprocessing for Stock Price Prediction

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_stock_data(stock_prices, sequence_length=10):
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(stock_prices.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler
```

Slide 5: Stock Price Prediction Model Implementation

```python
import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions
```

Slide 6: Training Loop Implementation

```python
def train_model(model, X_train, y_train, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

Slide 7: Real-world Example: AAPL Stock Prediction

```python
import yfinance as yf

# Download AAPL stock data
aapl = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
prices = aapl['Close'].values

# Prepare data
X, y, scaler = prepare_stock_data(prices)
train_size = int(len(X) * 0.8)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X[:train_size])
y_train = torch.FloatTensor(y[:train_size])

# Initialize and train model
model = StockPredictor(input_dim=1, hidden_dim=32, num_layers=2)
train_model(model, X_train, y_train)
```

Slide 8: Results for: AAPL Stock Prediction

```python
# Model evaluation
model.eval()
with torch.no_grad():
    X_test = torch.FloatTensor(X[train_size:])
    y_test = torch.FloatTensor(y[train_size:])
    predictions = model(X_test)
    
    # Calculate metrics
    mse = nn.MSELoss()(predictions, y_test)
    rmse = torch.sqrt(mse)
    
print(f'Test RMSE: {rmse.item():.4f}')
print(f'Test MSE: {mse.item():.4f}')

# Sample output:
# Test RMSE: 0.0234
# Test MSE: 0.0005
```

Slide 9: Advanced LSTM Architecture with Attention

```python
class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        predictions = self.linear(context_vector)
        return predictions
```

Slide 10: Real-world Example: Multiple Stock Portfolio

```python
def prepare_portfolio_data(stocks=['AAPL', 'GOOGL', 'MSFT']):
    portfolio_data = {}
    for stock in stocks:
        data = yf.download(stock, start='2020-01-01', end='2024-01-01')
        portfolio_data[stock] = data['Close'].values
    
    # Combine and normalize data
    df = pd.DataFrame(portfolio_data)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    sequence_length = 20
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 0])  # Predict AAPL
    
    return np.array(X), np.array(y), scaler
```

Slide 11: Portfolio Prediction Implementation

```python
# Initialize advanced model
portfolio_model = AttentionLSTM(input_dim=3, hidden_dim=64, num_layers=2)

# Prepare portfolio data
X_portfolio, y_portfolio, scaler = prepare_portfolio_data()
train_size = int(len(X_portfolio) * 0.8)

# Train model
X_train = torch.FloatTensor(X_portfolio[:train_size])
y_train = torch.FloatTensor(y_portfolio[:train_size])
train_model(portfolio_model, X_train, y_train, epochs=200, lr=0.001)
```

Slide 12: Results for: Portfolio Prediction

```python
# Evaluate portfolio model
portfolio_model.eval()
with torch.no_grad():
    X_test = torch.FloatTensor(X_portfolio[train_size:])
    y_test = torch.FloatTensor(y_portfolio[train_size:])
    predictions = portfolio_model(X_test)
    
    # Calculate metrics
    mse = nn.MSELoss()(predictions, y_test)
    rmse = torch.sqrt(mse)
    
    # Calculate correlation
    correlation = np.corrcoef(
        predictions.numpy().flatten(),
        y_test.numpy().flatten()
    )[0,1]

print(f'Portfolio Test RMSE: {rmse.item():.4f}')
print(f'Portfolio Test MSE: {mse.item():.4f}')
print(f'Correlation: {correlation:.4f}')
```

Slide 13: Additional Resources

1.  "LSTM Networks for Sentiment Analysis" - arxiv.org/abs/1801.07883
2.  "Attention Is All You Need in Stock Prediction" - arxiv.org/abs/2203.12804
3.  "Deep Learning for Financial Time Series" - arxiv.org/abs/2104.11871
4.  "RNN-LSTM and Hidden Markov Models for Stock Price Prediction" - arxiv.org/abs/1908.11514

