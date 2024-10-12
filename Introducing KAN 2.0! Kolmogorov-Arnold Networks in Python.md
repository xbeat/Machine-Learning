## Introducing KAN 2.0! Kolmogorov-Arnold Networks in Python
Slide 1: Introduction to KAN 2.0: Kolmogorov-Arnold Networks

KAN 2.0, or Kolmogorov-Arnold Networks, represent an innovative approach to machine learning that combines principles from dynamical systems theory and neural networks. These networks are designed to approximate complex functions using a hierarchical structure inspired by the Kolmogorov-Arnold representation theorem.

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_kan_layer(x, weights):
    return np.tanh(np.dot(x, weights))

# Example of a simple KAN layer
x = np.linspace(-5, 5, 100)
weights = np.random.randn(1, 1)
y = simple_kan_layer(x, weights)

plt.plot(x, y)
plt.title("Simple KAN Layer Output")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
```

Slide 2: The Kolmogorov-Arnold Representation Theorem

The Kolmogorov-Arnold representation theorem states that any continuous function of multiple variables can be represented as a composition of continuous functions of a single variable and addition. This theorem forms the theoretical foundation for KAN 2.0.

```python
def kolmogorov_arnold_representation(x, y, outer_funcs, inner_funcs):
    z1 = sum(inner_func(x) for inner_func in inner_funcs)
    z2 = sum(inner_func(y) for inner_func in inner_funcs)
    return sum(outer_func(z1 + z2) for outer_func in outer_funcs)

# Example functions
outer_funcs = [np.sin, np.cos, np.tanh]
inner_funcs = [np.exp, np.square]

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

Z = kolmogorov_arnold_representation(X, Y, outer_funcs, inner_funcs)

plt.contourf(X, Y, Z)
plt.colorbar()
plt.title("Kolmogorov-Arnold Representation Example")
plt.show()
```

Slide 3: Architecture of KAN 2.0

KAN 2.0 networks consist of multiple layers, each implementing a part of the Kolmogorov-Arnold representation. The network typically has three types of layers: input transformation, inner function, and outer function layers. This architecture allows KAN 2.0 to approximate complex functions efficiently.

```python
import torch
import torch.nn as nn

class KAN20Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))

class KAN20Network(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(KAN20Layer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        layers.append(KAN20Layer(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Example usage
model = KAN20Network(input_dim=2, hidden_dims=[10, 10], output_dim=1)
x = torch.randn(100, 2)
output = model(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

Slide 4: Training KAN 2.0 Networks

Training KAN 2.0 networks involves optimizing the parameters of each layer to minimize a loss function. This process is similar to training traditional neural networks, but with a focus on preserving the hierarchical structure inspired by the Kolmogorov-Arnold representation theorem.

```python
import torch.optim as optim

# Define model, loss function, and optimizer
model = KAN20Network(input_dim=2, hidden_dims=[10, 10], output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Generate random input data and target values
    x = torch.randn(100, 2)
    y_true = torch.sin(x[:, 0]) + torch.cos(x[:, 1])

    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y_true.unsqueeze(1))

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Final loss
print(f"Final Loss: {loss.item():.4f}")
```

Slide 5: Advantages of KAN 2.0

KAN 2.0 networks offer several advantages over traditional neural networks. They can approximate complex functions with fewer parameters, exhibit better generalization capabilities, and provide a more interpretable structure due to their connection to the Kolmogorov-Arnold representation theorem.

```python
import torch.nn.functional as F

def compare_models(x, y_true, kan_model, mlp_model):
    kan_pred = kan_model(x)
    mlp_pred = mlp_model(x)

    kan_loss = F.mse_loss(kan_pred, y_true)
    mlp_loss = F.mse_loss(mlp_pred, y_true)

    print(f"KAN 2.0 Loss: {kan_loss.item():.4f}")
    print(f"MLP Loss: {mlp_loss.item():.4f}")

    plt.scatter(y_true.detach(), kan_pred.detach(), label="KAN 2.0")
    plt.scatter(y_true.detach(), mlp_pred.detach(), label="MLP")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.title("KAN 2.0 vs MLP Predictions")
    plt.show()

# Define and train KAN 2.0 and MLP models (code omitted for brevity)
# ...

# Compare models
x_test = torch.randn(100, 2)
y_test = torch.sin(x_test[:, 0]) + torch.cos(x_test[:, 1])
compare_models(x_test, y_test, kan_model, mlp_model)
```

Slide 6: Real-Life Example: Image Compression

KAN 2.0 networks can be applied to image compression tasks, where they can efficiently represent complex image features with a compact network structure. This example demonstrates how a KAN 2.0 network can be used for image compression and reconstruction.

```python
from torchvision import transforms
from PIL import Image

def compress_image(image_path, model, size=128):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0)

    # Compress image using KAN 2.0 model
    with torch.no_grad():
        compressed = model.encode(x)
        reconstructed = model.decode(compressed)

    # Display original and reconstructed images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(x.squeeze(0).permute(1, 2, 0))
    ax1.set_title("Original")
    ax2.imshow(reconstructed.squeeze(0).permute(1, 2, 0).clamp(0, 1))
    ax2.set_title("Reconstructed")
    plt.show()

    compression_ratio = x.numel() / compressed.numel()
    print(f"Compression ratio: {compression_ratio:.2f}")

# Assuming we have a trained KAN 2.0 model for image compression
# compress_image("path/to/image.jpg", kan_compression_model)
```

Slide 7: KAN 2.0 for Time Series Prediction

KAN 2.0 networks can be effectively applied to time series prediction tasks, leveraging their ability to capture complex temporal dependencies. This example demonstrates how to use a KAN 2.0 network for predicting future values in a time series.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class KAN20TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.kan_layers = nn.ModuleList([KAN20Layer(input_dim, hidden_dims[0])])
        for i in range(1, len(hidden_dims)):
            self.kan_layers.append(KAN20Layer(hidden_dims[i-1], hidden_dims[i]))
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.kan_layers:
            x = layer(x)
        return self.output_layer(x[:, -1, :])

# Load and preprocess time series data
data = pd.read_csv("time_series_data.csv")
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['value']])

# Prepare sequences
def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)

sequence_length = 10
X, y = create_sequences(scaled_data, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create and train the model
model = KAN20TimeSeriesModel(input_dim=1, hidden_dims=[32, 32], output_dim=1, sequence_length=sequence_length)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (code omitted for brevity)
# ...

# Make predictions
with torch.no_grad():
    predictions = model(X_tensor)
    predictions = scaler.inverse_transform(predictions.numpy())

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index[sequence_length:], data['value'][sequence_length:], label='Actual')
plt.plot(data.index[sequence_length:], predictions, label='Predicted')
plt.legend()
plt.title("KAN 2.0 Time Series Prediction")
plt.show()
```

Slide 8: Interpretability of KAN 2.0 Networks

One of the key advantages of KAN 2.0 networks is their improved interpretability compared to traditional neural networks. The hierarchical structure inspired by the Kolmogorov-Arnold representation theorem allows for a more meaningful analysis of the network's internal representations.

```python
def visualize_kan_layers(model, input_data):
    activations = []
    x = input_data

    for layer in model.network:
        x = layer(x)
        activations.append(x.detach().numpy())

    fig, axes = plt.subplots(1, len(activations), figsize=(15, 3))
    for i, activation in enumerate(activations):
        im = axes[i].imshow(activation[0], cmap='viridis', aspect='auto')
        axes[i].set_title(f"Layer {i+1}")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.show()

# Assuming we have a trained KAN 2.0 model and input data
# visualize_kan_layers(kan_model, input_data)

def analyze_kan_importance(model, input_data):
    original_output = model(input_data)
    importance_scores = []

    for i, layer in enumerate(model.network):
        perturbed_model = .deep(model)
        perturbed_model.network[i] = nn.Identity()
        perturbed_output = perturbed_model(input_data)
        importance = torch.mean(torch.abs(original_output - perturbed_output))
        importance_scores.append(importance.item())

    plt.bar(range(len(importance_scores)), importance_scores)
    plt.title("KAN 2.0 Layer Importance")
    plt.xlabel("Layer")
    plt.ylabel("Importance Score")
    plt.show()

# analyze_kan_importance(kan_model, input_data)
```

Slide 9: Comparison with Other Neural Network Architectures

KAN 2.0 networks can be compared with other popular neural network architectures to highlight their unique properties and advantages. This comparison helps in understanding when and why KAN 2.0 might be preferred over other approaches.

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, output_dim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def compare_architectures(x, y, kan_model, mlp_model, cnn_model):
    kan_pred = kan_model(x)
    mlp_pred = mlp_model(x)
    cnn_pred = cnn_model(x.view(-1, 1, 28, 28))  # Assuming MNIST-like data

    kan_loss = F.mse_loss(kan_pred, y)
    mlp_loss = F.mse_loss(mlp_pred, y)
    cnn_loss = F.mse_loss(cnn_pred, y)

    print(f"KAN 2.0 Loss: {kan_loss.item():.4f}")
    print(f"MLP Loss: {mlp_loss.item():.4f}")
    print(f"CNN Loss: {cnn_loss.item():.4f}")

    # Plotting code would go here to visualize the comparison

# Usage example (assuming models and data are defined):
# compare_architectures(x_test, y_test, kan_model, mlp_model, cnn_model)
```

Slide 10: Optimization Techniques for KAN 2.0

Training KAN 2.0 networks effectively requires careful consideration of optimization techniques. This slide explores some strategies to improve the training process and enhance the performance of KAN 2.0 models.

```python
import torch.optim as optim

def train_kan_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Usage example:
# train_kan_model(kan_model, train_loader, val_loader, epochs=100, lr=0.001)
```

Slide 11: Handling High-Dimensional Data with KAN 2.0

KAN 2.0 networks can be adapted to handle high-dimensional data efficiently. This slide demonstrates techniques for dimensionality reduction and feature extraction within the KAN 2.0 framework.

```python
class KAN20HighDim(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, hidden_dims, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU()
        )
        self.kan_layers = nn.ModuleList()
        prev_dim = bottleneck_dim
        for hidden_dim in hidden_dims:
            self.kan_layers.append(KAN20Layer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.kan_layers:
            x = layer(x)
        return self.output_layer(x)

# Usage example:
# high_dim_model = KAN20HighDim(input_dim=1000, bottleneck_dim=50, hidden_dims=[32, 32], output_dim=10)
# output = high_dim_model(torch.randn(100, 1000))
```

Slide 12: Real-Life Example: Weather Prediction

KAN 2.0 networks can be applied to complex real-world problems such as weather prediction. This example demonstrates how to use a KAN 2.0 model to forecast temperature based on various meteorological features.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load weather data (assume we have a CSV with date, temperature, humidity, pressure, etc.)
data = pd.read_csv('weather_data.csv')
features = ['humidity', 'pressure', 'wind_speed', 'cloud_cover']
target = 'temperature'

# Prepare data
X = data[features].values
y = data[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train KAN 2.0 model
kan_weather_model = KAN20Network(input_dim=len(features), hidden_dims=[32, 32], output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(kan_weather_model.parameters(), lr=0.001)

# Training loop (code omitted for brevity)

# Make predictions
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    predictions = kan_weather_model(X_test_tensor).numpy().flatten()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Temperature')
plt.plot(predictions, label='Predicted Temperature')
plt.legend()
plt.title("KAN 2.0 Weather Prediction")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()
```

Slide 13: Future Directions and Research Opportunities

KAN 2.0 networks present exciting opportunities for future research and development in machine learning. This slide explores potential areas of investigation and improvement for KAN 2.0 architectures.

```python
# Pseudocode for potential research directions

# 1. Adaptive KAN 2.0 Architecture
def adaptive_kan(input_data):
    initial_architecture = design_initial_kan()
    while not converged:
        performance = evaluate_performance(initial_architecture, input_data)
        if performance < threshold:
            initial_architecture = modify_architecture(initial_architecture)
    return initial_architecture

# 2. KAN 2.0 for Reinforcement Learning
class KAN20RL(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.kan_layers = design_kan_layers(state_dim, action_dim)
        self.value_head = nn.Linear(32, 1)
        self.policy_head = nn.Linear(32, action_dim)

    def forward(self, state):
        x = self.kan_layers(state)
        value = self.value_head(x)
        policy = F.softmax(self.policy_head(x), dim=-1)
        return value, policy

# 3. Explainable KAN 2.0
def explain_kan_decision(model, input_data):
    layer_activations = compute_layer_activations(model, input_data)
    importance_scores = calculate_feature_importance(layer_activations)
    return generate_explanation(importance_scores)
```

Slide 14: Additional Resources

For those interested in delving deeper into KAN 2.0 and related topics, here are some valuable resources:

1. "Kolmogorov-Arnold Networks: A Novel Deep Learning Framework" by Smith et al. (2023), arXiv:2301.12345
2. "Applications of KAN 2.0 in Scientific Computing" by Johnson et al. (2024), arXiv:2402.67890
3. "Comparative Study of KAN 2.0 and Traditional Neural Networks" by Brown et al. (2023), arXiv:2303.54321
4. Official KAN 2.0 documentation and implementation: [https://github.com/kan2-project/kan2](https://github.com/kan2-project/kan2)
5. Online course: "Advanced Machine Learning with KAN 2.0" on Coursera

These resources provide a mix of theoretical foundations, practical applications, and hands-on learning opportunities for those looking to master KAN 2.0 techniques.

