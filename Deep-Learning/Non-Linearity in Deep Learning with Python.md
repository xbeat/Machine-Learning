## Non-Linearity in Deep Learning with Python
Slide 1: Introduction to Non-Linearity in Deep Learning

Non-linearity is a fundamental concept in deep learning that allows neural networks to learn and represent complex patterns in data. It introduces the ability to model non-linear relationships between inputs and outputs, which is crucial for solving real-world problems that often involve intricate, non-linear dependencies.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(-10, 10, 100)
y_linear = 2 * x + 1
y_nonlinear = np.sin(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_linear, label='Linear')
plt.plot(x, y_nonlinear, label='Non-linear')
plt.legend()
plt.title('Linear vs Non-linear Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 2: Linear vs Non-linear Functions

Linear functions are characterized by a constant rate of change, resulting in straight lines when plotted. Non-linear functions, on the other hand, have varying rates of change, producing curves or more complex shapes. In deep learning, non-linear activation functions introduce this crucial non-linearity into neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def linear_function(x):
    return 2 * x + 1

def nonlinear_function(x):
    return x**2 + 3*x + 1

x = np.linspace(-5, 5, 100)
y_linear = linear_function(x)
y_nonlinear = nonlinear_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y_linear, label='Linear: y = 2x + 1')
plt.plot(x, y_nonlinear, label='Non-linear: y = x^2 + 3x + 1')
plt.legend()
plt.title('Linear vs Non-linear Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 3: Activation Functions in Neural Networks

Activation functions are crucial components in neural networks that introduce non-linearity. They determine whether a neuron should be activated or not based on the weighted sum of its inputs. Common activation functions include ReLU, Sigmoid, and Tanh.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(132)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(133)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.tight_layout()
plt.show()
```

Slide 4: Rectified Linear Unit (ReLU)

ReLU is one of the most popular activation functions in deep learning. It introduces non-linearity by outputting the input directly if it's positive, and zero otherwise. This simple function helps mitigate the vanishing gradient problem and allows for faster training of deep neural networks.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)
y = relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(1, 4, 'ReLU(x) = max(0, x)', fontsize=12)
plt.show()
```

Slide 5: Sigmoid Activation Function

The sigmoid function maps input values to a range between 0 and 1, making it useful for binary classification problems. It introduces non-linearity by compressing extreme input values and providing a smooth, S-shaped curve.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.grid(True)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(2, 0.8, 'Sigmoid(x) = 1 / (1 + e^(-x))', fontsize=12)
plt.show()
```

Slide 6: Hyperbolic Tangent (Tanh) Activation Function

The tanh function is similar to the sigmoid but maps input values to a range between -1 and 1. It introduces non-linearity while addressing some limitations of the sigmoid function, such as having a stronger gradient and being zero-centered.

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)
y = tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('Tanh(x)')
plt.grid(True)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.text(1.5, 0.5, 'Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))', fontsize=12)
plt.show()
```

Slide 7: Implementing a Simple Neural Network with Non-linearity

Let's create a basic neural network with non-linear activation functions to demonstrate how non-linearity is incorporated into deep learning models. We'll use the ReLU activation function in the hidden layer and the sigmoid function in the output layer.

```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Hidden layer with ReLU activation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Output layer with Sigmoid activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

# Example usage
nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output = nn.forward(X)
print("Network output:")
print(output)
```

Slide 8: Non-linearity in Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) also incorporate non-linearity through activation functions. In CNNs, non-linear activation functions are typically applied after each convolutional layer, allowing the network to learn complex patterns in image data.

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)  # Non-linearity after convolution
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create a random input tensor (1 channel, 28x28 image)
input_tensor = torch.randn(1, 1, 28, 28)

# Initialize and use the CNN
cnn = SimpleCNN()
output = cnn(input_tensor)
print("CNN output shape:", output.shape)
```

Slide 9: Vanishing and Exploding Gradients

Non-linear activation functions can help address the vanishing and exploding gradient problems in deep neural networks. These issues occur when gradients become extremely small or large during backpropagation, making it difficult to train deep networks effectively.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

x = np.linspace(-10, 10, 1000)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, sigmoid_derivative(x), label='Sigmoid derivative')
plt.title('Sigmoid and its derivative')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, relu_derivative(x), label='ReLU derivative')
plt.title('ReLU and its derivative')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

Slide 10: Non-linearity in Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) also utilize non-linear activation functions to process sequential data. The non-linearity allows RNNs to capture complex temporal dependencies in the input sequence.

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()  # Non-linear activation

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.tanh(out)  # Apply non-linearity to the output
        return out

# Create a random input tensor (batch_size=1, sequence_length=10, input_size=5)
input_tensor = torch.randn(1, 10, 5)

# Initialize and use the RNN
rnn = SimpleRNN(input_size=5, hidden_size=10, output_size=2)
output = rnn(input_tensor)
print("RNN output shape:", output.shape)
```

Slide 11: Non-linearity in Autoencoders

Autoencoders are neural networks used for unsupervised learning and dimensionality reduction. They incorporate non-linear activation functions in both the encoder and decoder parts to learn complex data representations.

```python
import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),  # Non-linear activation in encoder
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),  # Non-linear activation in decoder
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # Output activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create a random input tensor (batch_size=1, 784 features for a 28x28 image)
input_tensor = torch.randn(1, 784)

# Initialize and use the Autoencoder
autoencoder = SimpleAutoencoder()
output = autoencoder(input_tensor)
print("Autoencoder output shape:", output.shape)
```

Slide 12: Real-life Example: Image Classification

Image classification is a common application of deep learning that heavily relies on non-linearity. Convolutional Neural Networks (CNNs) with non-linear activation functions can learn complex features in images, enabling accurate classification of various objects.

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 112 * 112, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load and preprocess an image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Replace 'path/to/your/image.jpg' with an actual image path
image = Image.open('path/to/your/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Initialize and use the CNN
cnn = SimpleCNN()
output = cnn(input_tensor)
print("CNN output (class probabilities):", output)
```

Slide 13: Real-life Example: Natural Language Processing

Non-linearity plays a crucial role in Natural Language Processing (NLP) tasks, such as sentiment analysis. Recurrent Neural Networks (RNNs) or Transformer models with non-linear activation functions can capture complex patterns in text data, enabling accurate sentiment classification.

```python
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        dense_outputs = self.fc(hidden.squeeze(0))
        outputs = self.sigmoid(dense_outputs)
        return outputs

# Example usage
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 1

model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, output_dim)
input_sequence = torch.randint(0, vocab_size, (1, 20))  # Batch size 1, sequence length 20
output = model(input_sequence)
print("Sentiment prediction:", output.item())
```

Slide 14: Importance of Non-linearity in Deep Learning

Non-linearity is crucial in deep learning for several reasons:

1. Complexity Modeling: Non-linear functions allow neural networks to approximate complex, non-linear relationships in data.
2. Feature Hierarchy: Non-linear activations enable the network to learn hierarchical representations of features.
3. Universal Function Approximation: With non-linear activations, neural networks can theoretically approximate any continuous function.
4. Gradient Flow: Proper non-linear functions help mitigate vanishing and exploding gradient problems during training.
5. Decision Boundaries: Non-linearity allows for the creation of complex decision boundaries, essential for many classification tasks.

Slide 15: Importance of Non-linearity in Deep Learning

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Non-linear Decision Boundary')

# Simulated non-linear model
def non_linear_model(X):
    return (X[:, 0]**2 + X[:, 1]**2 > 0.5).astype(int)

# Generate sample data
np.random.seed(0)
X = np.random.randn(200, 2)
y = non_linear_model(X)

plt.figure(figsize=(10, 6))
plot_decision_boundary(X, y, non_linear_model)
plt.show()
```

Slide 16: Additional Resources

For those interested in delving deeper into non-linearity in deep learning, here are some valuable resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press)
2. "Neural Networks and Deep Learning" by Michael Nielsen (online book)
3. "Nonlinear Dynamics and Chaos in Neural Networks" by Walter Senn and Johanni Brea (ArXiv:1908.05033)
4. "On the Expressive Power of Deep Neural Networks" by Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein (ArXiv:1606.05336)
5. Coursera's "Deep Learning Specialization" by Andrew Ng

These resources provide in-depth explanations and mathematical foundations of non-linearity in neural networks and its impact on deep learning performance.

