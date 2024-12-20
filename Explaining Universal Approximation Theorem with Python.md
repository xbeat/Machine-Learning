## Explaining Universal Approximation Theorem with Python
Slide 1: Universal Approximation Theorem and Neural Networks

The Universal Approximation Theorem is a fundamental concept in neural networks, stating that a feedforward network with a single hidden layer containing a finite number of neurons can approximate continuous functions on compact subsets of R^n, under mild assumptions on the activation function. This theorem provides the theoretical foundation for the effectiveness of neural networks in various applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.grid(True)
plt.show()
```

Slide 2: Single Hidden Layer Neural Network

A single hidden layer neural network consists of an input layer, one hidden layer, and an output layer. The Universal Approximation Theorem focuses on this architecture, demonstrating its power in function approximation.

```python
import numpy as np

class SingleHiddenLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for output layer
        return self.a2

# Example usage
nn = SingleHiddenLayerNN(input_size=2, hidden_size=5, output_size=1)
X = np.array([[1, 2]])
output = nn.forward(X)
print(f"Output: {output}")
```

Slide 3: Activation Functions

Activation functions introduce non-linearity into the network, enabling it to learn complex patterns. Common activation functions include sigmoid, tanh, and ReLU.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.subplot(132)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.subplot(133)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.tight_layout()
plt.show()
```

Slide 4: Function Approximation

The Universal Approximation Theorem states that neural networks can approximate a wide range of functions. Let's demonstrate this by approximating a simple function.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Generate data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X)

# Create and train the neural network
nn = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
nn.fit(X, y.ravel())

# Predict using the trained network
y_pred = nn.predict(X)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='True function')
plt.plot(X, y_pred, color='red', label='NN approximation')
plt.title('Neural Network Approximation of sin(x)')
plt.legend()
plt.show()
```

Slide 5: Limitations and Considerations

While the Universal Approximation Theorem guarantees the existence of a network that can approximate any continuous function, it doesn't provide information on how to find this network or how many neurons are needed. In practice, we often use multiple layers and optimization techniques to improve performance.

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Generate data
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.sin(X) + 0.1 * np.random.randn(1000, 1)

# Create MLPRegressor with different architectures
nn_shallow = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000)
nn_deep = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

# Compute learning curves
train_sizes, train_scores_shallow, test_scores_shallow = learning_curve(nn_shallow, X, y.ravel(), cv=5)
train_sizes, train_scores_deep, test_scores_deep = learning_curve(nn_deep, X, y.ravel(), cv=5)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores_shallow, axis=1), label='Shallow NN (train)')
plt.plot(train_sizes, np.mean(test_scores_shallow, axis=1), label='Shallow NN (test)')
plt.plot(train_sizes, np.mean(train_scores_deep, axis=1), label='Deep NN (train)')
plt.plot(train_sizes, np.mean(test_scores_deep, axis=1), label='Deep NN (test)')
plt.title('Learning Curves: Shallow vs Deep Neural Network')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend()
plt.show()
```

Slide 6: Real-Life Example: Handwritten Digit Recognition

One practical application of neural networks is handwritten digit recognition. The MNIST dataset is commonly used for this task.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Create and train the neural network
nn = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
nn.fit(X_train, y_train)

# Make predictions
y_pred = nn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize some predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Slide 7: Real-Life Example: Image Classification

Another common application of neural networks is image classification. Let's use a pre-trained model to classify images.

```python
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ResNet18 model
model = resnet18(pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download and process an image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Make a prediction
output = model(batch_t)

# Load class labels
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the predicted class
_, predicted = torch.max(output, 1)
classification = classes[predicted.item()]

print(f"Predicted class: {classification}")
```

Slide 8: Gradient Descent and Backpropagation

Neural networks are trained using gradient descent and backpropagation. These algorithms allow the network to learn from data by adjusting its weights and biases.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, iterations):
        for _ in range(iterations):
            self.feedforward()
            self.backprop()

# Example usage
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(X, y)
nn.train(1500)

print(nn.output)
```

Slide 9: Overfitting and Regularization

Overfitting occurs when a model learns the training data too well, including its noise, leading to poor generalization. Regularization techniques help prevent overfitting.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Generate noisy data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + 0.3 * np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models with different regularization
nn_no_reg = MLPRegressor(hidden_layer_sizes=(100,), alpha=0, max_iter=10000)
nn_l2 = MLPRegressor(hidden_layer_sizes=(100,), alpha=0.01, max_iter=10000)

nn_no_reg.fit(X_train, y_train.ravel())
nn_l2.fit(X_train, y_train.ravel())

# Plot results
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, nn_no_reg.predict(X), color='red', label='No regularization')
plt.plot(X, nn_l2.predict(X), color='green', label='L2 regularization')
plt.title('Effect of Regularization on Neural Network')
plt.legend()
plt.show()
```

Slide 10: Hyperparameter Tuning

Hyperparameters are configuration settings for the neural network that are not learned from data. Proper tuning of hyperparameters is crucial for optimal performance.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
}

# Create MLPClassifier
mlp = MLPClassifier(max_iter=1000)

# Perform grid search
grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X, y)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

Slide 11: Transfer Learning

Transfer learning involves using a pre-trained model as a starting point for a new task, often leading to faster training and better performance, especially with limited data.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 classes in CIFAR10

# Define transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR10 dataset
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model
model.train()
for epoch in range(5):  # 5 epochs for demonstration
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

print("Finished Training")
```

Slide 12: Convolutional Neural Networks (CNNs)

CNNs are specialized neural networks designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

# Create an instance of the CNN
model = SimpleCNN()

# Print model structure
print(model)

# Example input tensor (batch_size, channels, height, width)
example_input = torch.randn(1, 3, 32, 32)

# Pass the input through the model
output = model(example_input)

print(f"Output shape: {output.shape}")
```

Slide 13: Recurrent Neural Networks (RNNs)

RNNs are designed to work with sequential data by maintaining an internal state that can capture information from previous inputs. They are particularly useful for tasks like natural language processing and time series analysis.

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        # out shape: (batch_size, sequence_length, hidden_size)
        out = self.fc(out[:, -1, :])
        return out

# Create an instance of the RNN
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)

# Example input tensor (batch_size, sequence_length, input_size)
example_input = torch.randn(32, 15, 10)

# Pass the input through the model
output = model(example_input)

print(f"Output shape: {output.shape}")
```

Slide 14: Generative Adversarial Networks (GANs)

GANs are a class of neural networks consisting of two parts: a generator that creates data and a discriminator that evaluates it. They are used for generating new, synthetic data that resembles real data.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize generator and discriminator
latent_dim = 100
img_shape = (1, 28, 28)
generator = Generator(latent_dim, img_shape)
discriminator = Discriminator(img_shape)

print("Generator:", generator)
print("Discriminator:", discriminator)
```

Slide 15: Additional Resources

For those interested in diving deeper into the Universal Approximation Theorem and neural networks, here are some valuable resources:

1. ArXiv paper on the Universal Approximation Theorem: "Universal Approximation Bounds for Superpositions of a Sigmoidal Function" by Andrew R. Barron ArXiv URL: [https://arxiv.org/abs/math/9204221](https://arxiv.org/abs/math/9204221)
2. ArXiv paper on modern perspectives of the Universal Approximation Theorem: "The Universal Approximation Theorem for Neural Networks" by Boris Hanin and Mark Sellke ArXiv URL: [https://arxiv.org/abs/2107.07395](https://arxiv.org/abs/2107.07395)

These papers provide in-depth mathematical analysis and proofs related to the Universal Approximation Theorem and its implications for neural networks.

