## Softmax in Machine Learning with Python

Slide 1: Introduction to Softmax in Machine Learning

Softmax is a crucial function in machine learning, particularly in multi-class classification problems. It transforms a vector of real numbers into a probability distribution, ensuring that the sum of all probabilities equals 1. This property makes softmax ideal for scenarios where we need to choose one class among several options.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example input vector
x = np.array([2.0, 1.0, 0.1])

probabilities = softmax(x)
print(f"Input: {x}")
print(f"Softmax output: {probabilities}")
print(f"Sum of probabilities: {np.sum(probabilities)}")

plt.bar(range(len(probabilities)), probabilities)
plt.title("Softmax Probability Distribution")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.show()
```

Slide 2: Mathematical Foundation of Softmax

The softmax function is defined as:

$\\text{softmax}(x\_i) = \\frac{e^{x\_i}}{\\sum\_{j=1}^n e^{x\_j}}$

Where $x\_i$ is the i-th element of the input vector, and n is the number of classes. This formula ensures that the output is always between 0 and 1, and the sum of all outputs is 1.

```python
import sympy as sp

# Define symbolic variables
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Define the softmax function symbolically
def symbolic_softmax(x):
    exp_x = [sp.exp(xi) for xi in x]
    return [ei / sum(exp_x) for ei in exp_x]

# Create a vector of symbolic variables
x_vector = [x1, x2, x3]

# Calculate the symbolic softmax
softmax_result = symbolic_softmax(x_vector)

# Print the symbolic result
for i, result in enumerate(softmax_result):
    print(f"Softmax({x_vector[i]}) = {result}")
```

Slide 3: Implementing Softmax in Python

Let's implement the softmax function and apply it to a simple example:

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example usage
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)

print("Input scores:", scores)
print("Softmax probabilities:", probabilities)
print("Sum of probabilities:", np.sum(probabilities))

# Verify that probabilities sum to 1
assert np.isclose(np.sum(probabilities), 1.0), "Probabilities should sum to 1"
```

Slide 4: Numerical Stability in Softmax Implementation

When implementing softmax, it's crucial to consider numerical stability. Large input values can lead to overflow in the exponential function. We can prevent this by subtracting the maximum value from each input:

```python
import numpy as np

def unstable_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def stable_softmax(x):
    shifted_x = x - np.max(x)
    return np.exp(shifted_x) / np.sum(np.exp(shifted_x))

# Large input values
x = np.array([1000, 2000, 3000])

print("Unstable softmax:")
try:
    print(unstable_softmax(x))
except OverflowError as e:
    print(f"OverflowError: {e}")

print("\nStable softmax:")
print(stable_softmax(x))
```

Slide 5: Softmax in Neural Networks

In neural networks, softmax is often used as the activation function in the output layer for multi-class classification problems. It converts the raw outputs (logits) into a probability distribution over the classes.

```python
import numpy as np
import torch
import torch.nn as nn

# Define a simple neural network with softmax output
class SimpleNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        logits = self.fc(x)
        return nn.functional.softmax(logits, dim=1)

# Example usage
input_size = 5
num_classes = 3
model = SimpleNet(input_size, num_classes)

# Generate random input
x = torch.randn(1, input_size)

# Forward pass
output = model(x)

print("Input:", x)
print("Output probabilities:", output)
print("Sum of probabilities:", torch.sum(output).item())
```

Slide 6: Softmax vs. Sigmoid

While softmax is used for multi-class classification, sigmoid is used for binary classification. Let's compare their behaviors:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

x = np.linspace(-10, 10, 100)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid(x))
plt.title("Sigmoid Function")
plt.xlabel("Input")
plt.ylabel("Output")

plt.subplot(1, 2, 2)
x_2d = np.column_stack((x, -x))
softmax_output = softmax(x_2d)
plt.plot(x, softmax_output[:, 0], label="Class 1")
plt.plot(x, softmax_output[:, 1], label="Class 2")
plt.title("Softmax Function (2 classes)")
plt.xlabel("Input")
plt.ylabel("Probability")
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 7: Cross-Entropy Loss with Softmax

In machine learning, softmax is often used in conjunction with cross-entropy loss for multi-class classification problems. Let's implement this combination:

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / n_samples
    return loss

# Example usage
logits = np.array([[2.0, 1.0, 0.1],
                   [0.3, 3.0, 1.0]])
true_labels = np.array([[1, 0, 0],
                        [0, 1, 0]])

probabilities = softmax(logits)
loss = cross_entropy_loss(true_labels, probabilities)

print("Logits:", logits)
print("True labels:", true_labels)
print("Softmax probabilities:", probabilities)
print("Cross-entropy loss:", loss)
```

Slide 8: Softmax in Image Classification

One common application of softmax is in image classification tasks. Here's a simplified example using a pre-trained ResNet model:

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image
img = Image.open("path_to_image.jpg")
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(input_batch)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get the top 5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Print results
for i in range(top5_prob.size(0)):
    print(f"Class {top5_catid[i]}: {top5_prob[i].item():.4f}")
```

Slide 9: Softmax in Natural Language Processing

Softmax is also widely used in natural language processing tasks, such as text classification. Here's a simple example using a basic neural network for sentiment analysis:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        hidden = F.relu(self.fc(pooled))
        return F.softmax(self.output(hidden), dim=1)

# Example usage
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # binary sentiment (positive/negative)

model = SimpleSentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# Simulate input (batch_size = 2, sequence_length = 5)
text = torch.LongTensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Forward pass
output = model(text)

print("Input text indices:", text)
print("Output probabilities:", output)
```

Slide 10: Softmax Temperature

The softmax function has a temperature parameter that can be used to control the "sharpness" of the probability distribution. Higher temperatures make the distribution more uniform, while lower temperatures make it more peaked.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax_with_temperature(x, temperature=1.0):
    exp_x = np.exp(x / temperature)
    return exp_x / exp_x.sum()

# Example logits
logits = np.array([2.0, 1.0, 0.1])

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
plt.figure(figsize=(12, 6))

for i, temp in enumerate(temperatures):
    probs = softmax_with_temperature(logits, temp)
    plt.bar(np.arange(len(logits)) + i*0.15, probs, width=0.15, label=f'T={temp}')

plt.xlabel('Class')
plt.ylabel('Probability')
plt.title('Softmax with Different Temperatures')
plt.legend()
plt.xticks(np.arange(len(logits)), ['Class 1', 'Class 2', 'Class 3'])
plt.show()
```

Slide 11: Softmax in Reinforcement Learning

In reinforcement learning, softmax is often used to implement exploration strategies. Here's an example of a softmax policy for action selection:

```python
import numpy as np

def softmax_policy(Q_values, temperature=1.0):
    probabilities = softmax_with_temperature(Q_values, temperature)
    return np.random.choice(len(Q_values), p=probabilities)

def softmax_with_temperature(x, temperature=1.0):
    exp_x = np.exp(x / temperature)
    return exp_x / exp_x.sum()

# Example Q-values for 4 actions
Q_values = np.array([1.0, 2.0, 1.5, 0.5])

# Simulate action selection
n_simulations = 1000
action_counts = np.zeros(len(Q_values))

for _ in range(n_simulations):
    action = softmax_policy(Q_values)
    action_counts[action] += 1

print("Q-values:", Q_values)
print("Action selection frequencies:")
for i, count in enumerate(action_counts):
    print(f"Action {i}: {count/n_simulations:.2f}")
```

Slide 12: Softmax Regression

Softmax regression, also known as multinomial logistic regression, is a generalization of logistic regression for multi-class classification. Let's implement a simple softmax regression model:

```python
import numpy as np

class SoftmaxRegression:
    def __init__(self, n_features, n_classes):
        self.weights = np.random.randn(n_features, n_classes)
        self.bias = np.zeros(n_classes)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def fit(self, X, y, learning_rate=0.01, n_iterations=100):
        n_samples, n_features = X.shape
        
        for _ in range(n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(z)
            
            # Backward pass
            error = y_pred - y
            d_weights = np.dot(X.T, error) / n_samples
            d_bias = np.sum(error, axis=0) / n_samples
            
            # Update parameters
            self.weights -= learning_rate * d_weights
            self.bias -= learning_rate * d_bias

# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])

model = SoftmaxRegression(n_features=2, n_classes=3)
model.fit(X, y)

predictions = model.predict(X)
print("Predictions:")
print(predictions)
```

Slide 13: Softmax in Attention Mechanisms

Attention mechanisms, crucial in modern deep learning, heavily rely on softmax for computing attention weights. Here's a simplified implementation of a basic attention mechanism:

```python
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        
        return torch.softmax(attention_scores, dim=1)

# Example usage
hidden_size = 256
attention = Attention(hidden_size)

# Simulating hidden state and encoder outputs
hidden = torch.randn(1, 1, hidden_size)
encoder_outputs = torch.randn(1, 10, hidden_size)

attention_weights = attention(hidden, encoder_outputs)
print("Attention weights:", attention_weights)
print("Sum of weights:", attention_weights.sum().item())
```

Slide 14: Softmax in Generative Models

Generative models, such as language models, often use softmax to produce probability distributions over the vocabulary. Here's a simple example of a character-level language model using softmax:

```python
import torch
import torch.nn as nn

class CharacterLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(CharacterLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return torch.softmax(logits, dim=-1)

# Example usage
vocab_size = 128  # ASCII characters
embedding_dim = 32
hidden_dim = 64

model = CharacterLM(vocab_size, embedding_dim, hidden_dim)

# Simulate input (batch_size = 1, sequence_length = 5)
input_seq = torch.randint(0, vocab_size, (1, 5))

output_probs = model(input_seq)
print("Input sequence:", input_seq)
print("Output probabilities shape:", output_probs.shape)
print("Sum of probabilities for first character:", output_probs[0, 0].sum().item())
```

Slide 15: Additional Resources

For those interested in diving deeper into softmax and its applications in machine learning, here are some valuable resources:

1.  "Softmax Classification" by Stanford CS231n: ArXiv link: [https://arxiv.org/abs/1511.06211](https://arxiv.org/abs/1511.06211)
2.  "Attention Is All You Need" - The transformer architecture, which heavily relies on softmax: ArXiv link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3.  "Deep Learning" by Goodfellow, Bengio, and Courville - Chapter 6.2 on Softmax Units: Book website: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

These resources provide in-depth explanations and advanced applications of softmax in various machine learning contexts.

