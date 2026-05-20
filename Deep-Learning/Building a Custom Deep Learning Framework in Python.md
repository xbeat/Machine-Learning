## Building a Custom Deep Learning Framework in Python
Slide 1: Introduction to Deep Learning Frameworks

Deep learning frameworks are essential tools for building and training neural networks. In this presentation, we'll create our own lightweight framework from scratch using Python. This will help us understand the core concepts behind popular frameworks like TensorFlow and PyTorch.

```python
import numpy as np

class DeepLearningFramework:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, X, y, epochs, batch_size):
        # Training logic will be implemented here
        pass

# Usage example
model = DeepLearningFramework()
```

Slide 2: Implementing Layers

Layers are the building blocks of neural networks. We'll start by implementing a simple dense (fully connected) layer.

```python
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, grad):
        self.dweights = np.dot(self.inputs.T, grad)
        self.dbias = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

# Usage
layer = Dense(input_size=10, output_size=5)
output = layer.forward(np.random.randn(1, 10))
print(output.shape)  # (1, 5)
```

Slide 3: Activation Functions

Activation functions introduce non-linearity to our network, allowing it to learn complex patterns. Let's implement the ReLU activation function.

```python
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad):
        return grad * (self.inputs > 0)

# Usage
relu = ReLU()
output = relu.forward(np.array([-1, 0, 1, 2]))
print(output)  # [0, 0, 1, 2]
```

Slide 4: Loss Functions

Loss functions measure the difference between predicted and actual values. We'll implement the Mean Squared Error (MSE) loss.

```python
class MSE:
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        return np.mean(np.square(y_true - y_pred))

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.size

# Usage
mse = MSE()
loss = mse.forward(np.array([1, 2, 3]), np.array([1.1, 2.2, 2.8]))
print(f"MSE Loss: {loss:.4f}")  # MSE Loss: 0.0233
```

Slide 5: Optimizers

Optimizers update the model's parameters to minimize the loss. We'll implement the Stochastic Gradient Descent (SGD) optimizer.

```python
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.bias -= self.learning_rate * layer.dbias

# Usage
optimizer = SGD(learning_rate=0.1)
layer = Dense(input_size=5, output_size=3)
# Assume we've computed gradients
layer.dweights = np.random.randn(5, 3)
layer.dbias = np.random.randn(1, 3)
optimizer.update(layer)
```

Slide 6: Building the Network

Now let's combine our components to create a simple neural network.

```python
class NeuralNetwork(DeepLearningFramework):
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

# Usage
model = NeuralNetwork()
model.add(Dense(input_size=2, output_size=3))
model.add(ReLU())
model.add(Dense(output_size=1))
model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.01))
```

Slide 7: Training Loop

The training loop is where our network learns from data. We'll implement a basic loop with batch processing.

```python
def fit(self, X, y, epochs, batch_size):
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Forward pass
            y_pred = self.forward(X_batch)

            # Compute loss
            loss = self.loss.forward(y_batch, y_pred)

            # Backward pass
            grad = self.loss.backward()
            self.backward(grad)

            # Update weights
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update(layer)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

NeuralNetwork.fit = fit
```

Slide 8: Making Predictions

After training, we need a way to use our model for predictions.

```python
def predict(self, X):
    return self.forward(X)

NeuralNetwork.predict = predict

# Usage example
X_train = np.random.randn(1000, 2)
y_train = np.sum(X_train, axis=1, keepdims=True)

model = NeuralNetwork()
model.add(Dense(input_size=2, output_size=3))
model.add(ReLU())
model.add(Dense(output_size=1))
model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.01))

model.fit(X_train, y_train, epochs=1000, batch_size=32)

X_test = np.array([[1.0, 2.0]])
prediction = model.predict(X_test)
print(f"Prediction: {prediction[0][0]:.2f}, Expected: {np.sum(X_test):.2f}")
```

Slide 9: Real-Life Example: Image Classification

Let's use our framework to build a simple image classifier for handwritten digits.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode targets
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

# Build model
model = NeuralNetwork()
model.add(Dense(input_size=64, output_size=32))
model.add(ReLU())
model.add(Dense(output_size=10))
model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.01))

# Train model
model.fit(X_train, y_train_onehot, epochs=1000, batch_size=32)

# Evaluate model
predictions = model.predict(X_test)
accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
print(f"Test accuracy: {accuracy:.2f}")
```

Slide 10: Real-Life Example: Text Generation

Let's use our framework to build a simple character-level language model for text generation.

```python
import string

# Prepare data
text = "Hello, world! This is a simple example of text generation."
chars = string.printable
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Create sequences
seq_length = 10
X = []
y = []
for i in range(len(text) - seq_length):
    X.append([char_to_idx[ch] for ch in text[i:i+seq_length]])
    y.append(char_to_idx[text[i+seq_length]])

X = np.array(X) / len(chars)
y = np.eye(len(chars))[y]

# Build model
model = NeuralNetwork()
model.add(Dense(input_size=seq_length, output_size=128))
model.add(ReLU())
model.add(Dense(output_size=len(chars)))
model.compile(loss=MSE(), optimizer=SGD(learning_rate=0.01))

# Train model
model.fit(X, y, epochs=1000, batch_size=32)

# Generate text
seed = "Hello, wo"
generated_text = seed
for _ in range(50):
    x = np.array([[char_to_idx[ch] for ch in seed]]) / len(chars)
    pred = model.predict(x)
    next_char = idx_to_char[np.argmax(pred)]
    generated_text += next_char
    seed = seed[1:] + next_char

print(f"Generated text: {generated_text}")
```

Slide 11: Implementing Convolutional Layers

Convolutional layers are crucial for image processing tasks. Let's implement a basic 2D convolutional layer.

```python
class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros((output_channels, 1))

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, in_h, in_w, _ = inputs.shape
        out_h = in_h - self.kernel_size + 1
        out_w = in_w - self.kernel_size + 1
        
        output = np.zeros((batch_size, out_h, out_w, self.output_channels))
        
        for i in range(out_h):
            for j in range(out_w):
                input_slice = inputs[:, i:i+self.kernel_size, j:j+self.kernel_size, :]
                for k in range(self.output_channels):
                    output[:, i, j, k] = np.sum(input_slice * self.weights[k], axis=(1,2,3)) + self.bias[k]
        
        return output

    def backward(self, grad):
        # Implement backward pass (omitted for brevity)
        pass

# Usage
conv = Conv2D(input_channels=3, output_channels=16, kernel_size=3)
input_data = np.random.randn(1, 28, 28, 3)
output = conv.forward(input_data)
print(output.shape)  # (1, 26, 26, 16)
```

Slide 12: Implementing Recurrent Layers

Recurrent layers are essential for sequence processing tasks. Let's implement a simple RNN layer.

```python
class SimpleRNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))

    def forward(self, inputs, h_prev):
        self.inputs = inputs
        self.h_prev = h_prev
        
        self.h_next = np.tanh(np.dot(self.Wx, inputs) + np.dot(self.Wh, h_prev) + self.b)
        return self.h_next

    def backward(self, grad):
        # Implement backward pass (omitted for brevity)
        pass

# Usage
rnn = SimpleRNN(input_size=10, hidden_size=20)
x = np.random.randn(10, 1)
h = np.zeros((20, 1))
output = rnn.forward(x, h)
print(output.shape)  # (20, 1)
```

Slide 13: Model Serialization

To save and load our trained models, we need to implement serialization.

```python
import pickle

class NeuralNetwork(DeepLearningFramework):
    # ... (previous implementation) ...

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Usage
model = NeuralNetwork()
# ... (build and train the model) ...

# Save the model
model.save('my_model.pkl')

# Load the model
loaded_model = NeuralNetwork.load('my_model.pkl')

# Use the loaded model
X_test = np.random.randn(1, 10)
prediction = loaded_model.predict(X_test)
print(f"Prediction: {prediction}")
```

Slide 14: Additional Resources

For further exploration of deep learning concepts and implementation details:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ArXiv: [https://arxiv.org/abs/1607.06036](https://arxiv.org/abs/1607.06036)
2. "Dive into Deep Learning" by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola ArXiv: [https://arxiv.org/abs/2106.11342](https://arxiv.org/abs/2106.11342)
3. "Automatic Differentiation in Machine Learning: a Survey" by Atılım Güneş Baydin et al. ArXiv: [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767)

These resources provide in-depth explanations of the concepts we've covered and more advanced topics in deep learning.

