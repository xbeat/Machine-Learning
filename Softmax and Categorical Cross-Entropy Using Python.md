## Softmax and Categorical Cross-Entropy Using Python
Slide 1: Introduction to Softmax and Categorical Cross-Entropy

Softmax and Categorical Cross-Entropy are fundamental concepts in machine learning, particularly in classification tasks. Softmax transforms raw model outputs into probability distributions, while Categorical Cross-Entropy measures the dissimilarity between predicted and true distributions. This presentation will explore these concepts in depth, covering both forward and backward passes using Python.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-7))

# Example usage
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
true_labels = np.array([1, 0, 0])
loss = categorical_cross_entropy(true_labels, probabilities)

print(f"Probabilities: {probabilities}")
print(f"Loss: {loss}")
```

Slide 2: Softmax Function - Forward Pass

The Softmax function converts a vector of real numbers into a probability distribution. It's commonly used as the final activation function in multi-class classification neural networks. The function exponentiates each input and normalizes the results to ensure they sum to 1.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example
logits = np.array([2.0, 1.0, 0.1])
probabilities = softmax(logits)
print(f"Input logits: {logits}")
print(f"Softmax output: {probabilities}")
print(f"Sum of probabilities: {np.sum(probabilities)}")
```

Slide 3: Numerical Stability in Softmax

When implementing Softmax, it's crucial to consider numerical stability. Large input values can lead to overflow in the exponentiation step. To mitigate this, we subtract the maximum value from each input before exponentiation, which doesn't change the output probabilities but prevents overflow.

```python
import numpy as np

def unstable_softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def stable_softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Example with large numbers
large_inputs = np.array([1000, 2000, 3000])
print("Unstable Softmax:")
print(unstable_softmax(large_inputs))
print("\nStable Softmax:")
print(stable_softmax(large_inputs))
```

Slide 4: Categorical Cross-Entropy - Forward Pass

Categorical Cross-Entropy (CCE) is a loss function used to measure the dissimilarity between the predicted probability distribution and the true distribution in multi-class classification problems. It quantifies how well the predicted probabilities match the actual class labels.

```python
import numpy as np

def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-7))

# Example
true_labels = np.array([0, 1, 0])  # One-hot encoded
predicted_probs = np.array([0.1, 0.7, 0.2])
loss = categorical_cross_entropy(true_labels, predicted_probs)
print(f"True labels: {true_labels}")
print(f"Predicted probabilities: {predicted_probs}")
print(f"Categorical Cross-Entropy Loss: {loss}")
```

Slide 5: Numerical Stability in Categorical Cross-Entropy

Similar to Softmax, Categorical Cross-Entropy can face numerical issues, particularly when predicted probabilities are very close to 0 or 1. To prevent taking the logarithm of 0, we add a small epsilon value to the predictions.

```python
import numpy as np

def stable_categorical_cross_entropy(y_true, y_pred, epsilon=1e-7):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# Example
true_labels = np.array([0, 1, 0])
predicted_probs = np.array([0.001, 0.998, 0.001])
loss = stable_categorical_cross_entropy(true_labels, predicted_probs)
print(f"True labels: {true_labels}")
print(f"Predicted probabilities: {predicted_probs}")
print(f"Stable Categorical Cross-Entropy Loss: {loss}")
```

Slide 6: Softmax Derivative - Backward Pass

To perform backpropagation, we need to compute the derivative of the Softmax function. The derivative of Softmax with respect to its inputs has a special form due to its normalization property. We'll explore this derivative and its implementation.

```python
import numpy as np

def softmax_derivative(s):
    # Create a diagonal matrix from s
    diag_s = np.diag(s)
    # Outer product of s with itself
    outer_s = np.outer(s, s)
    # Return the Jacobian matrix
    return diag_s - outer_s

# Example
s = np.array([0.3, 0.5, 0.2])
jacobian = softmax_derivative(s)
print("Softmax output:", s)
print("Softmax Jacobian:")
print(jacobian)
```

Slide 7: Categorical Cross-Entropy Derivative - Backward Pass

The derivative of Categorical Cross-Entropy with respect to the predicted probabilities is straightforward. It's the difference between the predicted probabilities and the true labels. This simplicity is one reason why CCE is often used with Softmax in neural networks.

```python
import numpy as np

def categorical_cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

# Example
true_labels = np.array([0, 1, 0])
predicted_probs = np.array([0.1, 0.7, 0.2])
gradient = categorical_cross_entropy_derivative(true_labels, predicted_probs)
print(f"True labels: {true_labels}")
print(f"Predicted probabilities: {predicted_probs}")
print(f"Gradient: {gradient}")
```

Slide 8: Combined Softmax and Categorical Cross-Entropy Gradient

In practice, Softmax and Categorical Cross-Entropy are often used together in neural networks. When combined, their gradients simplify, leading to a more efficient backward pass. We'll implement this combined gradient computation.

```python
import numpy as np

def softmax_categorical_crossentropy_gradient(y_true, logits):
    # Compute softmax
    exp_logits = np.exp(logits - np.max(logits))
    y_pred = exp_logits / np.sum(exp_logits)
    
    # Compute gradient
    return y_pred - y_true

# Example
true_labels = np.array([0, 1, 0])
logits = np.array([2.0, 1.0, 0.1])
gradient = softmax_categorical_crossentropy_gradient(true_labels, logits)
print(f"True labels: {true_labels}")
print(f"Logits: {logits}")
print(f"Gradient: {gradient}")
```

Slide 9: Real-life Example: Image Classification

Let's apply Softmax and Categorical Cross-Entropy to a simple image classification task. We'll use a pre-trained model to classify an image and compute the loss.

```python
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Load and preprocess image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make prediction
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Compute loss (assuming the top prediction is correct)
true_label = np.zeros_like(preds[0])
true_label[np.argmax(preds)] = 1
loss = -np.sum(true_label * np.log(preds[0] + 1e-7))

print("Top 3 predictions:")
for _, label, prob in decoded_preds:
    print(f"{label}: {prob:.4f}")
print(f"\nLoss: {loss:.4f}")
```

Slide 10: Real-life Example: Sentiment Analysis

Another common application of Softmax and Categorical Cross-Entropy is in natural language processing tasks like sentiment analysis. We'll implement a simple sentiment classifier using a pre-trained model.

```python
import numpy as np
from transformers import pipeline

# Load pre-trained sentiment analysis model
classifier = pipeline('sentiment-analysis')

# Example texts
texts = [
    "I love this product! It's amazing!",
    "This movie was terrible, I hated it.",
    "The weather is okay today, nothing special."
]

# Perform sentiment analysis
results = classifier(texts)

# Convert results to probabilities and compute loss
for text, result in zip(texts, results):
    label = result['label']
    score = result['score']
    
    # Convert score to probabilities (simplified)
    probs = np.array([1 - score, score]) if label == 'POSITIVE' else np.array([score, 1 - score])
    
    # Assume the model's prediction is correct for this example
    true_label = np.array([0, 1]) if label == 'POSITIVE' else np.array([1, 0])
    
    # Compute loss
    loss = -np.sum(true_label * np.log(probs + 1e-7))
    
    print(f"Text: {text}")
    print(f"Sentiment: {label}")
    print(f"Probabilities: Negative={probs[0]:.4f}, Positive={probs[1]:.4f}")
    print(f"Loss: {loss:.4f}\n")
```

Slide 11: Visualizing Softmax and Cross-Entropy

To better understand how Softmax and Categorical Cross-Entropy work together, let's create a visualization of their behavior for different input values.

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-7))

# Generate input values
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)

# Compute Softmax and Cross-Entropy
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        logits = np.array([X[i,j], Y[i,j], 0])
        probs = softmax(logits)
        Z[i,j] = cross_entropy(np.array([1, 0, 0]), probs)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Cross-Entropy Loss')
plt.title('Softmax and Cross-Entropy Landscape')
plt.xlabel('Logit 1')
plt.ylabel('Logit 2')
plt.show()
```

Slide 12: Implementing a Simple Neural Network

Let's implement a basic neural network using Softmax and Categorical Cross-Entropy for multi-class classification on the Iris dataset.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.probs = self.softmax(self.z2)
        return self.probs
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        dz2 = self.probs - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        y_onehot = np.eye(3)[y]
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, y_onehot)
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# Train and evaluate
model = SimpleNN(4, 10, 3)
model.train(X_train, y_train)
train_acc = np.mean(model.predict(X_train) == y_train)
test_acc = np.mean(model.predict(X_test) == y_test)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

Slide 13: Handling Multi-class Classification in PyTorch

Let's explore how to implement Softmax and Categorical Cross-Entropy in PyTorch, a popular deep learning framework.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Define the model
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    train_outputs = model(X_train)
    test_outputs = model(X_test)
    _, train_preds = torch.max(train_outputs, 1)
    _, test_preds = torch.max(test_outputs, 1)
    train_acc = (train_preds == y_train).float().mean()
    test_acc = (test_preds == y_test).float().mean()

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

Slide 14: Visualizing Decision Boundaries

To better understand how our model classifies the Iris dataset, let's visualize its decision boundaries.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load data and reduce dimensionality
iris = load_iris()
X, y = iris.data, iris.target
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Create a mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Make predictions on the mesh grid
Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Decision Boundaries of Iris Classification')
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into Softmax, Categorical Cross-Entropy, and their applications in neural networks, here are some valuable resources:

1. "Understanding Categorical Cross-Entropy Loss, Binary Cross-Entropy Loss, Softmax Loss, Logistic Loss, Focal Loss and all those confusing names" by Raúl Gómez (arXiv:2101.08169)
2. "Efficient Backprop" by Yann LeCun et al. (1998) - This paper discusses various techniques for efficient training of neural networks, including the use of Softmax and Cross-Entropy. (Available on Yann LeCun's website)
3. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - This comprehensive book covers Softmax and Cross-Entropy in the context of deep learning. (Available online at [www.deeplearningbook.org](http://www.deeplearningbook.org))
4. "Neural Networks and Deep Learning" by Michael Nielsen - This free online book provides an excellent introduction to neural networks, including discussions on Softmax and Cross-Entropy. (Available at neuralnetworksanddeeplearning.com)

These resources provide a mix of theoretical background and practical implementations, suitable for readers looking to expand their understanding of these crucial concepts in machine learning.

