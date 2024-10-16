## Understanding Neural Network Classification with PyTorch
Slide 1: Introduction to Neural Network Classification

Neural network classification is a powerful machine learning technique used to categorize input data into predefined classes. It's based on artificial neural networks, which are inspired by the human brain's structure and function. These networks learn patterns from data to make predictions or decisions.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network for classification
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the classifier
model = SimpleClassifier(input_size=10, hidden_size=20, num_classes=3)
print(model)
```

Slide 2: PyTorch: A Dynamic Deep Learning Framework

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and intuitive interface for building and training neural networks. PyTorch uses dynamic computation graphs, allowing for easier debugging and more natural Python-like code.

```python
import torch

# Create a tensor (multi-dimensional array)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor x:")
print(x)

# Perform some operations
y = x * 2
z = torch.matmul(x, x.t())  # Matrix multiplication
print("\nTensor y (x * 2):")
print(y)
print("\nTensor z (x * x^T):")
print(z)
```

Slide 3: Preparing Data for Neural Network Classification

Before training a neural network, we need to prepare our data. This involves loading the dataset, splitting it into training and testing sets, and converting it into PyTorch tensors.

```python
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)
```

Slide 4: Defining a Neural Network Model

In PyTorch, we define neural network models by creating a class that inherits from nn.Module. This allows us to specify the layers and the forward pass of our network.

```python
import torch.nn as nn

class IrisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model = IrisClassifier(input_size=4, hidden_size=10, num_classes=3)
print(model)
```

Slide 5: Loss Function and Optimizer

To train our neural network, we need to define a loss function and an optimizer. The loss function measures how well our model is performing, while the optimizer updates the model's parameters to minimize the loss.

```python
import torch.optim as optim

# Define the loss function (Cross-Entropy Loss for classification)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Loss function:", criterion)
print("Optimizer:", optimizer)
```

Slide 6: Training Loop

The training loop is where we iterate over our dataset multiple times (epochs), make predictions, calculate the loss, and update the model's parameters.

```python
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
```

Slide 7: Evaluating the Model

After training, we need to evaluate our model's performance on the test set to see how well it generalizes to unseen data.

```python
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).float().mean()

print(f'Test Accuracy: {accuracy.item():.4f}')

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

Slide 8: Real-Life Example: Image Classification

Image classification is a common application of neural network classification. Let's look at how we can use a pre-trained model for image classification.

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image
image = Image.open("cat.jpg")
input_tensor = transform(image).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)

# Get the predicted class
_, predicted_idx = torch.max(output, 1)
print(f"Predicted class index: {predicted_idx.item()}")
```

Slide 9: Real-Life Example: Sentiment Analysis

Sentiment analysis is another popular application of neural network classification, used to determine the sentiment (positive, negative, or neutral) of text data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Sample data
texts = ["I love this movie", "This film is terrible", "The acting was okay"]
labels = [1, 0, 1]  # 1 for positive, 0 for negative

# Tokenize and build vocabulary
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Convert text to tensors
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: torch.tensor([x])

# Define the model
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        
    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)

# Create model instance
model = SentimentClassifier(len(vocab), 64, 2)

# Train the model (simplified for demonstration)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for text, label in zip(texts, labels):
        optimizer.zero_grad()
        predicted = model(torch.tensor(text_pipeline(text)))
        loss = criterion(predicted, label_pipeline(label))
        loss.backward()
        optimizer.step()

print("Training complete!")
```

Slide 10: Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are a specialized type of neural network designed for processing grid-like data, such as images. They use convolutional layers to automatically learn spatial hierarchies of features.

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 112 * 112, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create an instance of the CNN
model = SimpleCNN(num_classes=10)
print(model)

# Example input
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print("Output shape:", output.shape)
```

Slide 11: Recurrent Neural Networks (RNNs)

Recurrent Neural Networks are designed to work with sequence data, making them suitable for tasks like natural language processing and time series analysis. They maintain an internal state that can capture information about the sequence.

```python
import torch.nn as nn

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

# Create an instance of the RNN
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
print(model)

# Example input (batch_size=1, sequence_length=3, input_size=10)
input_tensor = torch.randn(1, 3, 10)
output = model(input_tensor)
print("Output shape:", output.shape)
```

Slide 12: Transfer Learning

Transfer learning is a technique where we use a pre-trained model as a starting point for a new task. This can significantly reduce training time and improve performance, especially when we have limited data.

```python
import torchvision.models as models
import torch.nn as nn

# Load a pre-trained ResNet model
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 5)  # 5 is the number of classes in our new task

# Now only the parameters of the new layer will be updated during training
print(resnet)

# Example usage
input_tensor = torch.randn(1, 3, 224, 224)
output = resnet(input_tensor)
print("Output shape:", output.shape)
```

Slide 13: Model Saving and Loading

After training a model, it's important to save it so that we can use it later without retraining. PyTorch provides easy ways to save and load models.

```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Create and save the model
model = SimpleModel()
torch.save(model.state_dict(), 'simple_model.pth')
print("Model saved!")

# Load the model
loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load('simple_model.pth'))
loaded_model.eval()
print("Model loaded!")

# Verify the loaded model
input_tensor = torch.randn(1, 10)
output = loaded_model(input_tensor)
print("Output from loaded model:", output)
```

Slide 14: Hyperparameter Tuning

Hyperparameter tuning is crucial for optimizing the performance of neural networks. We can use techniques like grid search or random search to find the best combination of hyperparameters.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Generate some dummy data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Define hyperparameters to search
param_grid = {
    'hidden_size': [5, 10, 20],
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64]
}

best_accuracy = 0
best_params = None

# Perform grid search
for params in ParameterGrid(param_grid):
    model = SimpleModel(params['hidden_size'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop (simplified)
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

print("Best parameters:", best_params)
print("Best accuracy:", best_accuracy.item())
```

Slide 15: Additional Resources

For further learning on neural network classification and PyTorch, consider exploring these resources:

1. PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html) The official documentation provides comprehensive guides and API references.
2. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville This book offers in-depth coverage of deep learning concepts and techniques.
3. FastAI course: [https://course.fast.ai/](https://course.fast.ai/) A practical deep learning course for coders, featuring PyTorch.
4. ArXiv papers on neural networks and deep learning:
   * "Attention Is All You Need" by Vaswani et al. (2017) ArXiv: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   * "Deep Residual Learning for Image Recognition" by He et al. (2015) ArXiv: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
5. PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/) Official tutorials covering various aspects of PyTorch and deep learning.
6. "Neural Networks and Deep Learning" by Michael Nielsen Free online book: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/)

These resources offer a mix of theoretical knowledge and practical implementation details to deepen your understanding of neural network classification and PyTorch.

