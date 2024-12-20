## PyTorch Tutorial Building Neural Networks with Python
Slide 1: Introduction to PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible and efficient framework for building and training neural networks. PyTorch is known for its dynamic computational graph, which allows for easier debugging and more intuitive development of complex models.

Slide 2: Source Code for Introduction to PyTorch

```python
import torch

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])

# Perform operations
y = x + 2
z = y * 3

print("Original tensor:", x)
print("After addition:", y)
print("After multiplication:", z)
```

Slide 3: Results for: Introduction to PyTorch

```
Original tensor: tensor([1, 2, 3, 4, 5])
After addition: tensor([3, 4, 5, 6, 7])
After multiplication: tensor([ 9, 12, 15, 18, 21])
```

Slide 4: Tensors in PyTorch

Tensors are the fundamental data structure in PyTorch. They are similar to NumPy arrays but can be used on GPUs to accelerate computing. Tensors can have different dimensions and data types, making them versatile for various machine learning tasks.

Slide 5: Source Code for Tensors in PyTorch

```python
import torch

# Create tensors
scalar = torch.tensor(42)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
print("3D Tensor:", tensor_3d)
```

Slide 6: Results for: Tensors in PyTorch

```
Scalar: tensor(42)
Vector: tensor([1, 2, 3])
Matrix: tensor([[1, 2],
                [3, 4]])
3D Tensor: tensor([[[1, 2],
                    [3, 4]],

                   [[5, 6],
                    [7, 8]]])
```

Slide 7: Autograd: Automatic Differentiation

PyTorch's autograd feature enables automatic computation of gradients, which is crucial for training neural networks. It keeps track of operations performed on tensors and automatically calculates gradients during backpropagation.

Slide 8: Source Code for Autograd: Automatic Differentiation

```python
import torch

# Create a tensor with requires_grad=True
x = torch.tensor([2.0], requires_grad=True)

# Define a simple function
y = x**2 + 3*x + 1

# Compute gradients
y.backward()

print("x:", x)
print("y:", y)
print("dy/dx:", x.grad)
```

Slide 9: Results for: Autograd: Automatic Differentiation

```
x: tensor([2.], requires_grad=True)
y: tensor([11.], grad_fn=<AddBackward0>)
dy/dx: tensor([7.])
```

Slide 10: Neural Networks with PyTorch

PyTorch provides a high-level API for building neural networks through the `torch.nn` module. This module contains various layers and loss functions that can be combined to create complex network architectures.

Slide 11: Source Code for Neural Networks with PyTorch

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model
model = SimpleNN()

# Create a random input tensor
input_tensor = torch.randn(1, 10)

# Forward pass
output = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
print("Model architecture:")
print(model)
```

Slide 12: Results for: Neural Networks with PyTorch

```
Input shape: torch.Size([1, 10])
Output shape: torch.Size([1, 1])
Model architecture:
SimpleNN(
  (fc1): Linear(in_features=10, out_features=5, bias=True)
  (fc2): Linear(in_features=5, out_features=1, bias=True)
  (relu): ReLU()
)
```

Slide 13: Training a Model

Training a model in PyTorch involves defining a loss function, an optimizer, and iterating through the training data. During each iteration, we perform a forward pass, compute the loss, and update the model parameters through backpropagation.

Slide 14: Source Code for Training a Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Linear(1, 1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate some fake data
X = torch.linspace(-10, 10, 100).view(-1, 1)
y = 2 * X + 1 + torch.randn(X.size()) * 0.1

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = criterion(y_pred, y)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test the model
test_input = torch.tensor([[5.0]])
predicted = model(test_input)
print(f"Predicted output for input {test_input.item()}: {predicted.item():.4f}")
```

Slide 15: Results for: Training a Model

```
Epoch [10/100], Loss: 0.0124
Epoch [20/100], Loss: 0.0123
Epoch [30/100], Loss: 0.0123
Epoch [40/100], Loss: 0.0123
Epoch [50/100], Loss: 0.0123
Epoch [60/100], Loss: 0.0123
Epoch [70/100], Loss: 0.0123
Epoch [80/100], Loss: 0.0123
Epoch [90/100], Loss: 0.0123
Epoch [100/100], Loss: 0.0123
Predicted output for input 5.0: 10.9854
```

Slide 16: Real-life Example: Image Classification

Image classification is a common application of deep learning. In this example, we'll use a pre-trained ResNet model to classify images. This demonstrates how PyTorch can be used for transfer learning and working with image data.

Slide 17: Source Code for Real-life Example: Image Classification

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet model
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
img = Image.open("sample_image.jpg")
img_tensor = transform(img).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(img_tensor)

# Load class labels
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the predicted class
_, predicted_idx = torch.max(output, 1)
predicted_label = classes[predicted_idx.item()]

print(f"The image is classified as: {predicted_label}")
```

Slide 18: Real-life Example: Natural Language Processing

Natural Language Processing (NLP) is another area where PyTorch excels. In this example, we'll use a pre-trained BERT model for sentiment analysis on a given text input.

Slide 19: Source Code for Real-life Example: Natural Language Processing

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare input text
text = "I love learning about artificial intelligence!"

# Tokenize and encode the text
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Make a prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the predicted sentiment
predicted_class = torch.argmax(predictions).item()
sentiment = "Positive" if predicted_class == 1 else "Negative"

print(f"Input text: {text}")
print(f"Predicted sentiment: {sentiment}")
print(f"Confidence: {predictions[0][predicted_class].item():.4f}")
```

Slide 20: Additional Resources

For more information on PyTorch and its applications, you can refer to the following resources:

1.  PyTorch official documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
2.  "Attention Is All You Need" paper (Transformer architecture): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3.  "Deep Residual Learning for Image Recognition" paper (ResNet): [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
4.  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" paper: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

These resources provide in-depth information on PyTorch and some of the key architectures used in modern deep learning.

