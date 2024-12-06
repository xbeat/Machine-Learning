## Neural Networks as Universal Function Approximators
Slide 1: Understanding Neural Networks as Function Approximators

Neural networks fundamentally operate as universal function approximators, capable of learning complex mappings between inputs and outputs through optimization of their internal parameters. The basic structure consists of layers of interconnected nodes that transform input data through nonlinear activation functions.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_neural_network(x, weights, biases, activation_fn):
    # Single layer neural network
    # weights shape: [input_dim, hidden_dim]
    hidden = np.dot(x, weights) + biases
    return activation_fn(hidden)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Example usage
x = np.linspace(-5, 5, 100).reshape(-1, 1)
weights = np.array([[1.0]])
biases = np.array([0.0])

output = simple_neural_network(x, weights, biases, relu)
```

Slide 2: Implementing the Universal Approximation Example

This implementation demonstrates how multiple ReLU functions can be combined to approximate a complex polynomial function. We'll create a neural network that approximates f(x) = x³ - 3x² + 2x + 5 using the approach described in the Universal Approximation Theorem.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    return x**3 - 3*x**2 + 2*x + 5

def relu_network(x, W1, B1, W2, B2):
    # Forward pass through the network
    hidden = np.maximum(0, np.dot(x.reshape(-1, 1), W1) + B1)
    output = np.dot(hidden, W2) + B2
    return output

# Network parameters
W1 = np.array([[1, 1, 1, 1, -1]])
B1 = np.array([0, 1, -2, -3, -1])
W2 = np.array([-5, 5, 5, 15, -20]).reshape(-1, 1)
B2 = np.array([0])

# Generate test points
x = np.linspace(-2, 4, 200)
y_true = target_function(x)
y_pred = relu_network(x, W1, B1, W2, B2)
```

Slide 3: Visualizing the Approximation

The visualization of how well our neural network approximates the target function provides insights into the effectiveness of the Universal Approximation Theorem. We can observe how combining multiple ReLU units creates a smooth approximation of the complex polynomial.

```python
plt.figure(figsize=(12, 6))
plt.plot(x, y_true, 'b-', label='Target Function')
plt.plot(x, y_pred, 'r--', label='Neural Network Approximation')
plt.title('Neural Network Approximation of Cubic Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Calculate approximation error
mse = np.mean((y_true - y_pred.flatten())**2)
print(f'Mean Squared Error: {mse:.6f}')
```

Slide 4: Implementing a Multi-Layer Universal Approximator

A more sophisticated implementation using multiple layers demonstrates how increasing network complexity can improve approximation accuracy. This implementation uses PyTorch to create a deeper network with customizable architecture.

```python
import torch
import torch.nn as nn

class UniversalApproximator(nn.Module):
    def __init__(self, hidden_sizes=[64, 32, 16]):
        super().__init__()
        layers = []
        input_size = 1
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size
            
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

Slide 5: Training the Universal Approximator

The training process demonstrates how the neural network learns to approximate the target function through gradient descent optimization. We use PyTorch's automatic differentiation to compute gradients and update network parameters.

```python
def train_approximator(model, x_train, y_train, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
            
    return model

# Prepare training data
x_train = torch.linspace(-2, 4, 200).reshape(-1, 1)
y_train = target_function(x_train.numpy()).reshape(-1, 1)
y_train = torch.FloatTensor(y_train)

model = UniversalApproximator()
trained_model = train_approximator(model, x_train, y_train)
```

Slide 6: Advanced ReLU Network Variations

Exploring different activation functions and their impact on function approximation capabilities. This implementation demonstrates how using variants of ReLU like Leaky ReLU and ELU can affect the network's approximation abilities.

```python
import torch.nn.functional as F

class AdvancedApproximator(nn.Module):
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.activation_type = activation_type
        
    def activation(self, x):
        if self.activation_type == 'relu':
            return F.relu(x)
        elif self.activation_type == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.01)
        elif self.activation_type == 'elu':
            return F.elu(x)
            
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)
```

Slide 7: Custom Loss Functions for Better Approximation

Implementing custom loss functions can improve the approximation quality by focusing on specific aspects of the target function. This example shows how to implement a weighted MSE loss that emphasizes certain regions of the function.

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, weight_fn):
        super().__init__()
        self.weight_fn = weight_fn
        
    def forward(self, pred, target, x):
        weights = self.weight_fn(x)
        return torch.mean(weights * (pred - target)**2)

def importance_weights(x):
    # Give more weight to regions with high curvature
    return 1 + torch.abs(x)

# Custom loss implementation
criterion = WeightedMSELoss(importance_weights)
loss = criterion(y_pred, y_train, x_train)
```

Slide 8: Approximating Discontinuous Functions

Neural networks can also approximate discontinuous functions, though they require more complex architectures. This implementation shows how to handle step functions and other discontinuities.

```python
def step_function(x):
    return np.where(x >= 0, 1, 0)

class DiscontinuousApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Scale parameter controls steepness of approximation
scale = 10.0
x = torch.linspace(-2, 2, 200).reshape(-1, 1)
y = torch.tensor(step_function(x.numpy()))
```

Slide 9: Mathematical Foundations in Code

Implementing the mathematical foundations of the Universal Approximation Theorem using matrix operations and activation functions. This code demonstrates the relationship between mathematical theory and practical implementation.

```python
def linear_combination(x, coefficients, points):
    """
    Mathematical representation of piecewise linear approximation
    """
    code_block = """
    # Mathematical formula in LaTeX (not rendered):
    # $$f(x) = \sum_{i=1}^n c_i \max(0, x - p_i)$$
    """
    
    result = np.zeros_like(x)
    for c, p in zip(coefficients, points):
        result += c * np.maximum(0, x - p)
    return result

# Example approximation of sin(x)
x = np.linspace(-np.pi, np.pi, 1000)
coefficients = [1, -2, 1, -0.5]
points = [-np.pi/2, 0, np.pi/2, np.pi]
approximation = linear_combination(x, coefficients, points)
```

Slide 10: Adaptive Learning Rate Implementation

The approximation quality can be significantly improved by implementing an adaptive learning rate schedule that adjusts based on the approximation error. This implementation demonstrates a custom learning rate scheduler for optimal convergence.

```python
class AdaptiveApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def adaptive_train(model, x_train, y_train, epochs=1000):
    initial_lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = F.mse_loss(y_pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())
        
    return model, losses
```

Slide 11: Error Analysis and Confidence Bounds

Implementing error analysis tools to understand the quality of function approximation and establish confidence bounds on the predictions. This code demonstrates how to compute and visualize approximation uncertainty.

```python
def compute_error_bounds(model, x_data, n_samples=100):
    predictions = []
    model.train()  # Enable dropout for uncertainty estimation
    
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(x_data)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    
    return mean_pred, std_pred

# Visualization of confidence bounds
plt.figure(figsize=(12, 6))
mean, std = compute_error_bounds(model, x_train)
plt.plot(x_train.numpy(), mean.numpy(), 'b-', label='Mean Prediction')
plt.fill_between(x_train.numpy().flatten(),
                 (mean - 2*std).numpy().flatten(),
                 (mean + 2*std).numpy().flatten(),
                 alpha=0.2, label='95% Confidence Interval')
plt.legend()
plt.show()
```

Slide 12: Real-world Application: Financial Time Series

Implementing the Universal Approximation Theorem for financial time series prediction, demonstrating how neural networks can approximate complex market patterns.

```python
class FinancialApproximator(nn.Module):
    def __init__(self, input_size, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def prepare_financial_data(prices, sequence_length):
    sequences = []
    targets = []
    
    for i in range(len(prices) - sequence_length):
        seq = prices[i:i+sequence_length]
        target = prices[i+sequence_length]
        sequences.append(seq)
        targets.append(target)
        
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)

# Example usage with synthetic data
prices = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.1, 1000)
X, y = prepare_financial_data(prices, sequence_length=10)
```

Slide 13: Real-world Application: Signal Processing

Implementing a neural network approximator for signal denoising and compression, demonstrating practical application in signal processing using autoencoder architecture based on the Universal Approximation Theorem.

```python
class SignalApproximator(nn.Module):
    def __init__(self, signal_length, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(signal_length, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, signal_length)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def process_signal(signal, noise_level=0.1):
    # Add noise to original signal
    noisy_signal = signal + noise_level * torch.randn_like(signal)
    
    # Prepare data for training
    model = SignalApproximator(signal.shape[1], encoding_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        reconstructed = model(noisy_signal)
        loss = F.mse_loss(reconstructed, signal)
        loss.backward()
        optimizer.step()
    
    return model, reconstructed
```

Slide 14: Optimization Techniques for Better Approximation

Advanced optimization strategies to improve the approximation capabilities of neural networks, including gradient clipping and custom initialization methods.

```python
class OptimizedApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        ])
        self.init_weights()
    
    def init_weights(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)

def train_with_gradient_clipping(model, x_train, y_train, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters())
    max_grad_norm = 1.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = F.mse_loss(y_pred, y_train)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
    return model
```

Slide 15: Additional Resources

*   "On the Approximation Properties of Neural Networks" - [https://arxiv.org/abs/1901.02220](https://arxiv.org/abs/1901.02220)
*   "Universal Approximation Bounds for Superpositions of a Sigmoidal Function" - [https://arxiv.org/abs/1905.08644](https://arxiv.org/abs/1905.08644)
*   "Deep Learning Theory: Approximation, Optimization, Generalization" - [https://arxiv.org/abs/1809.08561](https://arxiv.org/abs/1809.08561)
*   For practical implementations and tutorials, search for:
    *   "Neural Network Function Approximation PyTorch"
    *   "Universal Approximation Theorem Implementation"
    *   "Deep Learning for Function Approximation"

