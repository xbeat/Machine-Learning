## Exploring the Building Blocks of Modern AI
Slide 1: Multi-layer Perceptron Implementation

A Multi-layer Perceptron forms the foundation of modern neural networks, implementing forward and backward propagation to learn patterns in data through adjustable weights and biases. This implementation demonstrates a basic MLP for binary classification with one hidden layer.

```python
import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Backward propagation
        dz2 = self.a2 - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update parameters
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
```

Slide 2: MLP Training Example

This implementation demonstrates training the MLP on a simple XOR problem, showcasing how the network learns non-linear decision boundaries through iterative weight updates and gradient descent.

```python
# Generate XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Initialize and train MLP
mlp = MLP(input_size=2, hidden_size=4, output_size=1)

# Training loop
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    output = mlp.forward(X)
    
    # Backward pass
    mlp.backward(X, y, learning_rate=0.1)
    
    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = -np.mean(y * np.log(output) + (1-y) * np.log(1-output))
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Test predictions
predictions = mlp.forward(X)
print("\nFinal Predictions:")
print(np.round(predictions))
```

Slide 3: Deep Autoencoder Architecture

A deep autoencoder consists of an encoder that compresses input data into a lower-dimensional latent space and a decoder that reconstructs the original input. This architecture is particularly useful for dimensionality reduction and feature learning.

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

Slide 4: Training the Autoencoder

Implementation of the training loop for the autoencoder, including data preprocessing, loss calculation, and optimization. This example uses MNIST dataset to demonstrate image reconstruction capabilities.

```python
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize model and optimizer
input_dim = 28 * 28  # MNIST image size
encoding_dim = 32
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten input
        data = data.view(data.size(0), -1)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, data)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
```

Slide 5: LSTM Network Implementation

Long Short-Term Memory networks excel at capturing long-term dependencies in sequential data through their specialized gating mechanisms. This implementation shows a complete LSTM cell with forget, input, and output gates.

```python
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights for gates and states
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        
        # Initialize biases
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
        
    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state
        concat = np.concatenate((x_t, h_prev), axis=1)
        
        # Compute gates
        f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        i_t = self.sigmoid(np.dot(concat, self.Wi) + self.bi)
        c_tilde = np.tanh(np.dot(concat, self.Wc) + self.bc)
        o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        
        # Update cell state and hidden state
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t, (f_t, i_t, o_t, c_tilde)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

Slide 6: LSTM Time Series Prediction

Implementing LSTM for time series prediction using a practical example with real-world data. This implementation includes data preprocessing and sequence generation for time series forecasting.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesLSTM:
    def __init__(self, input_size, hidden_size, sequence_length):
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
    def create_sequences(self, data, sequence_length):
        sequences = []
        targets = []
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length])
        return np.array(sequences), np.array(targets)
    
    def forward(self, x_sequence):
        h_t = np.zeros((1, self.hidden_size))
        c_t = np.zeros((1, self.hidden_size))
        
        # Process each time step
        for t in range(self.sequence_length):
            x_t = x_sequence[t].reshape(1, -1)
            h_t, c_t, _ = self.lstm_cell.forward(x_t, h_t, c_t)
        
        return h_t

# Example usage
data = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Create sequences
model = TimeSeriesLSTM(input_size=1, hidden_size=32, sequence_length=10)
sequences, targets = model.create_sequences(normalized_data, 10)

# Process one sequence
output = model.forward(sequences[0])
print(f"Prediction shape: {output.shape}")
```

Slide 7: CNN Architecture From Scratch

A comprehensive implementation of a Convolutional Neural Network built from scratch, demonstrating the core operations of convolution, pooling, and fully connected layers without using deep learning frameworks.

```python
import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
        
    def convolve(self, input_data, filter_weights):
        height, width = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        output = np.zeros((output_height, output_width))
        
        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(
                    input_data[i:i+self.filter_size, j:j+self.filter_size] * filter_weights
                )
        return output
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.zeros((
            self.num_filters,
            input_data.shape[0] - self.filter_size + 1,
            input_data.shape[1] - self.filter_size + 1
        ))
        
        for i in range(self.num_filters):
            self.output[i] = self.convolve(input_data, self.filters[i])
        return self.output

class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        
    def forward(self, input_data):
        self.input = input_data
        n_channels, height, width = input_data.shape
        output_height = height // self.pool_size
        output_width = width // self.pool_size
        
        self.output = np.zeros((n_channels, output_height, output_width))
        
        for ch in range(n_channels):
            for i in range(output_height):
                for j in range(output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    pool_region = input_data[ch,
                                          start_i:start_i+self.pool_size,
                                          start_j:start_j+self.pool_size]
                    self.output[ch, i, j] = np.max(pool_region)
        
        return self.output
```

Slide 8: CNN Image Classification Example

A practical implementation showing how to use the custom CNN implementation for image classification, including data preprocessing and forward pass through multiple layers.

```python
# Example usage with sample image data
def preprocess_image(image, size=(28, 28)):
    # Normalize image to [0, 1]
    processed = image.astype(np.float32) / 255.0
    # Resize if needed
    if image.shape != size:
        # Implement resize logic here
        pass
    return processed

# Create simple CNN architecture
class SimpleCNN:
    def __init__(self):
        self.conv1 = ConvLayer(num_filters=16, filter_size=3)
        self.pool1 = MaxPoolLayer(pool_size=2)
        self.conv2 = ConvLayer(num_filters=32, filter_size=3)
        self.pool2 = MaxPoolLayer(pool_size=2)
    
    def forward(self, x):
        # First convolution block
        x = self.conv1.forward(x)
        x = np.maximum(0, x)  # ReLU activation
        x = self.pool1.forward(x)
        
        # Second convolution block
        x = self.conv2.forward(x)
        x = np.maximum(0, x)  # ReLU activation
        x = self.pool2.forward(x)
        
        return x

# Example usage
sample_image = np.random.rand(28, 28)  # Sample 28x28 image
processed_image = preprocess_image(sample_image)

# Create and use model
model = SimpleCNN()
output = model.forward(processed_image)
print(f"Output shape: {output.shape}")

# Example prediction
flattened = output.reshape(-1)
prediction = np.argmax(flattened)
print(f"Predicted class: {prediction}")
```

Slide 9: Deep Reinforcement Learning Implementation

Deep Q-Learning implementation combining neural networks with reinforcement learning principles. This architecture enables agents to learn optimal policies through experience, using a replay buffer and target network for stable training.

```python
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return np.argmax(q_values.numpy())
```

Slide 10: DQL Training Loop Implementation

Complete training implementation for the Deep Q-Learning agent, showcasing experience replay, target network updates, and epsilon-greedy exploration strategy.

```python
def train(self, batch_size=32):
    if len(self.memory) < batch_size:
        return
    
    # Sample random minibatch from memory
    minibatch = random.sample(self.memory, batch_size)
    
    states = torch.FloatTensor([t[0] for t in minibatch])
    actions = torch.LongTensor([t[1] for t in minibatch])
    rewards = torch.FloatTensor([t[2] for t in minibatch])
    next_states = torch.FloatTensor([t[3] for t in minibatch])
    dones = torch.FloatTensor([t[4] for t in minibatch])
    
    # Current Q values
    current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
    
    # Next Q values from target net
    with torch.no_grad():
        max_next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
    
    # Compute loss
    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    # Update epsilon
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

# Example usage
state_size = 4
action_size = 2
agent = DQLAgent(state_size, action_size)

# Training loop example
episodes = 1000
for episode in range(episodes):
    state = env.reset()  # assuming env is your environment
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
```

Slide 11: Graph Neural Network Basic Implementation

Implementation of a basic Graph Neural Network layer that performs message passing between nodes in a graph structure, essential for learning on graph-structured data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)
        
    def forward(self, x, adj_matrix):
        # x: node features (N x in_features)
        # adj_matrix: adjacency matrix (N x N)
        
        # Message passing
        messages = torch.matmul(adj_matrix, x)  # Aggregate neighbors
        
        # Concatenate node features with aggregated messages
        combined = torch.cat([x, messages], dim=1)
        
        # Update node representations
        return F.relu(self.linear(combined))

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
            
        self.layers.append(GNNLayer(hidden_dim, output_dim))
        
    def forward(self, x, adj_matrix):
        for layer in self.layers:
            x = layer(x, adj_matrix)
        return x
```

Slide 12: Graph Neural Network Application Example

Implementing a practical GNN application for node classification in a citation network, where nodes represent papers and edges represent citations between papers.

```python
import numpy as np
import torch
import torch.optim as optim

# Create synthetic citation network data
def create_citation_network(num_nodes=1000, num_features=128):
    # Generate random node features
    features = torch.randn(num_nodes, num_features)
    
    # Generate random sparse adjacency matrix
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        # Add random citations
        num_citations = np.random.randint(5, 15)
        cited_papers = np.random.choice(num_nodes, num_citations, replace=False)
        adj_matrix[i, cited_papers] = 1
    
    # Generate random labels (paper categories)
    num_classes = 7
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    return features, adj_matrix, labels

# Training function
def train_citation_network(model, features, adj_matrix, labels, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = model(features, adj_matrix)
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            acc = (output.argmax(dim=1) == labels).float().mean()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

# Create and train model
num_nodes = 1000
num_features = 128
features, adj_matrix, labels = create_citation_network(num_nodes, num_features)

model = GNN(
    input_dim=num_features,
    hidden_dim=64,
    output_dim=7,  # number of paper categories
    num_layers=3
)

train_citation_network(model, features, adj_matrix, labels)
```

Slide 13: Results Analysis for Neural Architectures

A comprehensive comparison of performance metrics across different neural network architectures implemented in previous slides, featuring accuracy, training time, and convergence analysis.

```python
import matplotlib.pyplot as plt
import pandas as pd

def analyze_model_performance(model_results):
    """
    Analyze and visualize performance metrics for different neural architectures
    """
    # Sample results dictionary
    results = {
        'MLP': {'accuracy': 0.92, 'training_time': 45, 'epochs_to_converge': 100},
        'LSTM': {'accuracy': 0.88, 'training_time': 120, 'epochs_to_converge': 150},
        'CNN': {'accuracy': 0.95, 'training_time': 180, 'epochs_to_converge': 80},
        'GNN': {'accuracy': 0.87, 'training_time': 90, 'epochs_to_converge': 120},
        'DQL': {'accuracy': 0.85, 'training_time': 240, 'epochs_to_converge': 200}
    }
    
    # Create DataFrame for visualization
    df = pd.DataFrame(results).T
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot accuracy comparison
    df['accuracy'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    
    # Plot training time comparison
    df['training_time'].plot(kind='bar', ax=axes[1], color='lightgreen')
    axes[1].set_title('Training Time Comparison')
    axes[1].set_ylabel('Time (seconds)')
    
    # Plot convergence comparison
    df['epochs_to_converge'].plot(kind='bar', ax=axes[2], color='salmon')
    axes[2].set_title('Epochs to Convergence')
    axes[2].set_ylabel('Number of Epochs')
    
    plt.tight_layout()
    return df

# Generate and display analysis
performance_df = analyze_model_performance({})
print("\nDetailed Performance Metrics:")
print(performance_df)
```

Slide 14: Additional Resources

*   Paper: "Deep Learning" by Yann LeCun, Yoshua Bengio, and Geoffrey Hinton
    *   [https://www.nature.com/articles/nature14539](https://www.nature.com/articles/nature14539)
*   Paper: "Human-level control through deep reinforcement learning"
    *   [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)
*   Paper: "Attention Is All You Need"
    *   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
*   Paper: "Graph Neural Networks: A Review of Methods and Applications"
    *   [https://arxiv.org/abs/1812.08434](https://arxiv.org/abs/1812.08434)
*   Suggested searches for implementation details:
    *   "PyTorch official tutorials"
    *   "TensorFlow documentation"
    *   "Deep Learning with Python by François Chollet"

Would you like me to generate a new presentation on a different topic, or do you have any questions about the neural network architectures we just covered? I'm happy to provide more specific details about any of the implementations or explore another AI-related topic.

