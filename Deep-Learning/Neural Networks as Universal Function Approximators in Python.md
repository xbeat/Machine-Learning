## Neural Networks as Universal Function Approximators in Python

Slide 1: Introduction to Neural Networks as Universal Function Approximators

Neural networks are powerful models capable of approximating any continuous function to arbitrary precision. This property, known as universal approximation, makes them versatile tools for solving complex problems across various domains.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_universal_approximation():
    x = np.linspace(-10, 10, 1000)
    y = np.sin(x) + np.cos(x**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='True Function')
    plt.plot(x, np.tanh(x), label='Neural Network Approximation')
    plt.title('Universal Function Approximation')
    plt.legend()
    plt.show()

plot_universal_approximation()
```

Slide 2: The Universal Approximation Theorem

The Universal Approximation Theorem states that a feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of R^n, under mild assumptions on the activation function.

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = SimpleNN(1, 10, 1)
print(model)
```

Slide 3: Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation_functions():
    x = np.linspace(-5, 5, 100)
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(x, relu)
    plt.title('ReLU')
    plt.subplot(132)
    plt.plot(x, sigmoid)
    plt.title('Sigmoid')
    plt.subplot(133)
    plt.plot(x, tanh)
    plt.title('Tanh')
    plt.tight_layout()
    plt.show()

plot_activation_functions()
```

Slide 4: Approximating a Simple Function

Let's approximate a simple function, f(x) = x^2, using a neural network with one hidden layer.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class QuadraticApproximator(nn.Module):
    def __init__(self):
        super(QuadraticApproximator, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = QuadraticApproximator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

x = torch.linspace(-5, 5, 100).view(-1, 1)
y = x**2

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

plt.scatter(x.detach().numpy(), y.detach().numpy(), label='True')
plt.scatter(x.detach().numpy(), model(x).detach().numpy(), label='Approximated')
plt.legend()
plt.show()
```

Slide 5: Increasing Network Complexity

As we increase the number of neurons and layers, the network can approximate more complex functions.

```python
import torch
import torch.nn as nn

class ComplexApproximator(nn.Module):
    def __init__(self, hidden_layers):
        super(ComplexApproximator, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1, 64)] +
                                    [nn.Linear(64, 64) for _ in range(hidden_layers)] +
                                    [nn.Linear(64, 1)])
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

model_simple = ComplexApproximator(1)
model_complex = ComplexApproximator(5)

print("Simple model parameters:", sum(p.numel() for p in model_simple.parameters()))
print("Complex model parameters:", sum(p.numel() for p in model_complex.parameters()))
```

Slide 6: Approximating Discontinuous Functions

Neural networks can approximate discontinuous functions by using multiple neurons to create sharp transitions.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class StepApproximator(nn.Module):
    def __init__(self):
        super(StepApproximator, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.output = nn.Linear(20, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = StepApproximator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

x = torch.linspace(-5, 5, 1000).view(-1, 1)
y = (x > 0).float()

for epoch in range(5000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

plt.plot(x.detach().numpy(), y.detach().numpy(), label='True')
plt.plot(x.detach().numpy(), model(x).detach().numpy(), label='Approximated')
plt.legend()
plt.show()
```

Slide 7: Approximating Multivariate Functions

Neural networks can approximate functions with multiple inputs, making them suitable for complex real-world problems.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MultivariateApproximator(nn.Module):
    def __init__(self):
        super(MultivariateApproximator, self).__init__()
        self.hidden1 = nn.Linear(2, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        x = self.output(x)
        return x

model = MultivariateApproximator()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

x = torch.linspace(-5, 5, 50)
y = torch.linspace(-5, 5, 50)
X, Y = torch.meshgrid(x, y)
Z = torch.sin(X) * torch.cos(Y)

inputs = torch.stack([X.flatten(), Y.flatten()], dim=1)
targets = Z.flatten().unsqueeze(1)

for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X.numpy(), Y.numpy(), Z.numpy())
ax1.set_title('True Function')
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X.numpy(), Y.numpy(), model(inputs).detach().numpy().reshape(50, 50))
ax2.set_title('Approximated Function')
plt.show()
```

Slide 8: Real-Life Example: Image Classification

Neural networks can approximate complex functions in image classification tasks, learning to map raw pixel values to class probabilities.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

model = SimpleConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):  # Just one epoch for demonstration
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Batch {i}, Loss: {loss.item()}')

print("Finished Training")
```

Slide 9: Real-Life Example: Natural Language Processing

Neural networks can approximate complex language models, learning to predict the next word in a sequence.

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        output = self.fc(output)
        return output

vocab_size = 10000
embedding_dim = 100
hidden_dim = 256

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim)

# Example input: batch of 3 sentences, each with 5 words
input_seq = torch.randint(0, vocab_size, (3, 5))
output = model(input_seq)

print("Input shape:", input_seq.shape)
print("Output shape:", output.shape)
```

Slide 10: Limitations and Challenges

While neural networks are powerful function approximators, they face challenges such as overfitting, vanishing/exploding gradients, and the need for large amounts of training data.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class OverfittingDemo(nn.Module):
    def __init__(self, complexity):
        super(OverfittingDemo, self).__init__()
        self.fc = nn.Linear(1, complexity)
        self.output = nn.Linear(complexity, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc(x))
        return self.output(x)

x = torch.linspace(-5, 5, 100).view(-1, 1)
y = torch.sin(x) + torch.randn_like(x) * 0.1

simple_model = OverfittingDemo(5)
complex_model = OverfittingDemo(100)

criterion = nn.MSELoss()
optimizer_simple = torch.optim.Adam(simple_model.parameters(), lr=0.01)
optimizer_complex = torch.optim.Adam(complex_model.parameters(), lr=0.01)

for _ in range(1000):
    optimizer_simple.zero_grad()
    optimizer_complex.zero_grad()
    
    loss_simple = criterion(simple_model(x), y)
    loss_complex = criterion(complex_model(x), y)
    
    loss_simple.backward()
    loss_complex.backward()
    
    optimizer_simple.step()
    optimizer_complex.step()

plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), simple_model(x).detach().numpy(), label='Simple Model')
plt.plot(x.numpy(), complex_model(x).detach().numpy(), label='Complex Model (Overfitting)')
plt.legend()
plt.show()
```

Slide 11: Regularization Techniques

Regularization methods like L1/L2 regularization and dropout help prevent overfitting and improve generalization.

```python
import torch
import torch.nn as nn

class RegularizedNN(nn.Module):
    def __init__(self):
        super(RegularizedNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

model = RegularizedNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # L2 regularization

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

print("Model architecture:", model)
print("Optimizer settings:", optimizer)
```

Slide 12: Transfer Learning

Transfer learning allows us to leverage pre-trained models, fine-tuning them for specific tasks and domains. This approach is particularly useful when dealing with limited data or computational resources.

```python
import torch
import torchvision.models as models
import torch.nn as nn

# Load a pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10 is the number of classes in our new task

# Define optimizer for the new layer
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

# Pseudo-code for fine-tuning
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Fine-tuned model:", resnet)
```

Slide 13: Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing neural network performance. Techniques like grid search, random search, and Bayesian optimization help find the best configuration.

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

# Define the parameter space
param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['tanh', 'relu'],
    'alpha': np.logspace(-5, 3, 9),
    'learning_rate': ['constant', 'adaptive'],
}

# Create a base model
mlp = MLPClassifier(max_iter=1000)

# Perform random search
random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, 
                                   n_iter=20, cv=5, n_jobs=-1)

# Fit the random search object (assuming X_train and y_train are defined)
# random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)
```

Slide 14: Visualizing Neural Network Decisions

Visualization techniques help us understand how neural networks make decisions, enhancing interpretability and trust in the model.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleClassifier()

# Generate a grid of points
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Make predictions
Z = model(torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32))
Z = torch.argmax(Z, dim=1).reshape(X.shape)

# Plot decision boundary
plt.contourf(X, Y, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.colorbar()
plt.title("Neural Network Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Slide 15: Additional Resources

For further exploration of neural networks as universal function approximators, consider these resources:

1. "Approximation by Superpositions of a Sigmoidal Function" by G. Cybenko (1989) ArXiv: [https://arxiv.org/abs/9807105](https://arxiv.org/abs/9807105)
2. "Universal Approximation Bounds for Superpositions of a Sigmoidal Function" by A. R. Barron (1993) IEEE Transactions on Information Theory
3. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville Available online: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
4. "Neural Networks and Deep Learning" by Michael Nielsen Available online: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

These resources provide in-depth theoretical foundations and practical applications of neural networks as universal function approximators.

