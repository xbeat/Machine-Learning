## Neural Network Forward Pass in Python
Slide 1: Neural Network Forward Pass

A forward pass in a neural network is the process of propagating input data through the network to generate an output. This fundamental operation involves a series of matrix multiplications and activation functions applied layer by layer.

```python
import numpy as np

def forward_pass(x, weights, biases, activation_function):
    layer_output = np.dot(x, weights) + biases
    return activation_function(layer_output)

# Example usage
input_data = np.array([1, 2, 3])
weights = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
biases = np.array([0.1, 0.2])

def relu(x):
    return np.maximum(0, x)

output = forward_pass(input_data, weights, biases, relu)
print("Output:", output)
```

Slide 2: Input Layer

The input layer is the starting point of the forward pass. It receives the raw input data and passes it to the next layer without any transformation.

```python
import numpy as np

class InputLayer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input_data):
        return input_data

# Example usage
input_layer = InputLayer(3)
input_data = np.array([1.0, 2.0, 3.0])
output = input_layer.forward(input_data)
print("Input layer output:", output)
```

Slide 3: Dense Layer

A dense layer, also known as a fully connected layer, performs a linear transformation on its input by applying weights and biases.

```python
import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
    
    def forward(self, input_data):
        return np.dot(input_data, self.weights) + self.biases

# Example usage
dense_layer = DenseLayer(3, 2)
input_data = np.array([1.0, 2.0, 3.0])
output = dense_layer.forward(input_data)
print("Dense layer output:", output)
```

Slide 4: Activation Functions

Activation functions introduce non-linearity to the network, allowing it to learn complex patterns. Common activation functions include ReLU, sigmoid, and tanh.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Example usage
x = np.array([-2, -1, 0, 1, 2])
print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
```

Slide 5: Combining Layers

A neural network consists of multiple layers combined sequentially. The forward pass involves propagating the input through each layer in order.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

# Example usage
input_layer = InputLayer(3)
hidden_layer = DenseLayer(3, 4)
output_layer = DenseLayer(4, 2)

network = NeuralNetwork([input_layer, hidden_layer, output_layer])
input_data = np.array([1.0, 2.0, 3.0])
output = network.forward(input_data)
print("Network output:", output)
```

Slide 6: Softmax Activation

The softmax function is commonly used in the output layer for multi-class classification problems. It converts raw scores into probabilities that sum to 1.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Example usage
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)
print("Softmax probabilities:", probabilities)
print("Sum of probabilities:", np.sum(probabilities))
```

Slide 7: Loss Calculation

After the forward pass, we calculate the loss to measure the difference between the network's predictions and the true labels.

```python
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small value to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

# Example usage
y_true = np.array([0, 1, 0])  # One-hot encoded true label
y_pred = np.array([0.1, 0.7, 0.2])  # Predicted probabilities
loss = cross_entropy_loss(y_true, y_pred)
print("Cross-entropy loss:", loss)
```

Slide 8: Batch Processing

In practice, we often process multiple inputs simultaneously in batches to improve efficiency and stability during training.

```python
import numpy as np

def batch_forward_pass(batch, weights, biases, activation_function):
    layer_output = np.dot(batch, weights) + biases
    return activation_function(layer_output)

# Example usage
batch_size = 3
input_size = 4
output_size = 2

batch = np.random.randn(batch_size, input_size)
weights = np.random.randn(input_size, output_size)
biases = np.random.randn(output_size)

def relu(x):
    return np.maximum(0, x)

output = batch_forward_pass(batch, weights, biases, relu)
print("Batch output shape:", output.shape)
print("Batch output:\n", output)
```

Slide 9: Dropout Layer

Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 during training.

```python
import numpy as np

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
    
    def forward(self, input_data, training=True):
        if not training:
            return input_data
        
        mask = np.random.binomial(1, 1 - self.dropout_rate, input_data.shape) / (1 - self.dropout_rate)
        return input_data * mask

# Example usage
dropout_layer = DropoutLayer(dropout_rate=0.5)
input_data = np.random.randn(5, 10)
output_training = dropout_layer.forward(input_data, training=True)
output_inference = dropout_layer.forward(input_data, training=False)

print("Training output (with dropout):\n", output_training)
print("Inference output (without dropout):\n", output_inference)
```

Slide 10: Convolutional Layer

Convolutional layers are essential for processing grid-like data such as images. They apply filters to extract features from local regions.

```python
import numpy as np

def conv2d(input_data, kernel, stride=1, padding=0):
    input_height, input_width = input_data.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1
    
    padded_input = np.pad(input_data, padding, mode='constant')
    output = np.zeros((output_height, output_width))
    
    for i in range(0, output_height):
        for j in range(0, output_width):
            output[i, j] = np.sum(
                padded_input[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width] * kernel
            )
    
    return output

# Example usage
input_data = np.random.randn(5, 5)
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

output = conv2d(input_data, kernel, stride=1, padding=1)
print("Convolution output:\n", output)
```

Slide 11: Max Pooling Layer

Max pooling is a downsampling operation that reduces the spatial dimensions of the input, helping to achieve spatial invariance.

```python
import numpy as np

def max_pool2d(input_data, pool_size, stride=None):
    if stride is None:
        stride = pool_size
    
    input_height, input_width = input_data.shape
    pool_height, pool_width = pool_size
    
    output_height = (input_height - pool_height) // stride + 1
    output_width = (input_width - pool_width) // stride + 1
    
    output = np.zeros((output_height, output_width))
    
    for i in range(0, output_height):
        for j in range(0, output_width):
            output[i, j] = np.max(
                input_data[i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
            )
    
    return output

# Example usage
input_data = np.random.randn(6, 6)
output = max_pool2d(input_data, pool_size=(2, 2), stride=2)
print("Max pooling output:\n", output)
```

Slide 12: Real-life Example: Image Classification

Image classification is a common application of neural networks. Here's a simplified example of a forward pass for classifying handwritten digits.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def forward_pass(image, weights1, biases1, weights2, biases2):
    # Flatten the image
    flattened = image.reshape(1, -1)
    
    # First layer: Dense + ReLU
    hidden = relu(np.dot(flattened, weights1) + biases1)
    
    # Second layer: Dense + Softmax
    output = softmax(np.dot(hidden, weights2) + biases2)
    
    return output

# Example usage (assuming pre-trained weights and biases)
image = np.random.rand(28, 28)  # Simulated 28x28 grayscale image
weights1 = np.random.randn(784, 128)
biases1 = np.random.randn(128)
weights2 = np.random.randn(128, 10)
biases2 = np.random.randn(10)

probabilities = forward_pass(image, weights1, biases1, weights2, biases2)
predicted_digit = np.argmax(probabilities)

print("Predicted probabilities:", probabilities)
print("Predicted digit:", predicted_digit)
```

Slide 13: Real-life Example: Sentiment Analysis

Sentiment analysis is another common application of neural networks, particularly in natural language processing.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_text(text, vocab):
    words = text.lower().split()
    return [vocab.get(word, 0) for word in words]

def forward_pass(input_ids, embedding_matrix, weights, bias):
    # Embedding layer
    embedded = embedding_matrix[input_ids]
    
    # Average pooling
    pooled = np.mean(embedded, axis=0)
    
    # Dense layer with sigmoid activation
    output = sigmoid(np.dot(pooled, weights) + bias)
    
    return output

# Example usage
vocab = {"the": 1, "movie": 2, "was": 3, "great": 4, "terrible": 5}
embedding_matrix = np.random.randn(len(vocab) + 1, 50)  # +1 for unknown words
weights = np.random.randn(50, 1)
bias = np.random.randn(1)

text = "The movie was great"
input_ids = preprocess_text(text, vocab)
sentiment_score = forward_pass(input_ids, embedding_matrix, weights, bias)

print("Input text:", text)
print("Sentiment score:", sentiment_score[0])
print("Predicted sentiment:", "Positive" if sentiment_score > 0.5 else "Negative")
```

Slide 14: Additional Resources

For a deeper understanding of neural networks and their forward pass implementation, consider exploring these resources:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2. "Neural Networks and Deep Learning" by Michael Nielsen (Online book: [http://neuralnetworksanddeeplearning.com/](http://neuralnetworksanddeeplearning.com/))
3. "Efficient BackProp" by Yann LeCun et al. (1998) - ArXiv: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
4. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Sergey Ioffe and Christian Szegedy (2015) - ArXiv: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)

These resources provide in-depth explanations and advanced techniques for implementing and optimizing neural networks.

