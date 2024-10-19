## Artificial Neural Networks Fundamentals and Implementation

Slide 1: Introduction to Artificial Neural Networks

Artificial Neural Networks (ANNs) are computational models inspired by the human brain's structure and function. They consist of interconnected nodes (artificial neurons) organized in layers. ANNs are designed to recognize patterns, process complex data, and make predictions or decisions. This powerful machine learning technique has revolutionized various fields, including image and speech recognition, natural language processing, and more.

Slide 2: Source Code for Introduction to Artificial Neural Networks

```python
import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return 1 / (1 + math.exp(-weighted_sum))  # Sigmoid activation function

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.hidden_layer = [Neuron(num_inputs) for _ in range(num_hidden)]
        self.output_layer = [Neuron(num_hidden) for _ in range(num_outputs)]

    def forward(self, inputs):
        hidden_outputs = [neuron.activate(inputs) for neuron in self.hidden_layer]
        final_outputs = [neuron.activate(hidden_outputs) for neuron in self.output_layer]
        return final_outputs

# Example usage
nn = NeuralNetwork(num_inputs=3, num_hidden=4, num_outputs=2)
result = nn.forward([0.5, 0.3, 0.8])
print(f"Network output: {result}")
```

Slide 3: Neural Network Architecture

The architecture of an ANN typically consists of three main components: an input layer, one or more hidden layers, and an output layer. The input layer receives the initial data, hidden layers process the information, and the output layer produces the final result. Connections between neurons in adjacent layers have associated weights, which are adjusted during the learning process to improve the network's performance.

Slide 4: Source Code for Neural Network Architecture

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [Neuron(layer_sizes[i-1]) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            next_inputs = []
            for neuron in layer:
                next_inputs.append(neuron.activate(inputs))
            inputs = next_inputs
        return inputs

# Create a neural network with 3 input neurons, 4 hidden neurons, and 2 output neurons
nn = NeuralNetwork([3, 4, 2])
result = nn.forward([0.5, 0.3, 0.8])
print(f"Network output: {result}")
```

Slide 5: Activation Functions

Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include sigmoid, tanh, and ReLU (Rectified Linear Unit). These functions determine whether a neuron should be activated based on its input, enabling the network to make decisions and classifications.

Slide 6: Source Code for Activation Functions

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

def relu(x):
    return max(0, x)

# Example usage
x = 2.5
print(f"Sigmoid: {sigmoid(x)}")
print(f"Tanh: {tanh(x)}")
print(f"ReLU: {relu(x)}")

# Plotting activation functions
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
plt.plot(x, [sigmoid(i) for i in x], label='Sigmoid')
plt.plot(x, [tanh(i) for i in x], label='Tanh')
plt.plot(x, [relu(i) for i in x], label='ReLU')
plt.legend()
plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()
```

Slide 7: Results for Activation Functions

```
Sigmoid: 0.9241418199787566
Tanh: 0.9866142981514303
ReLU: 2.5
```

Slide 8: Feedforward Process

The feedforward process is the mechanism by which input data flows through the neural network to produce an output. During this process, each neuron receives inputs, applies weights and biases, and passes the result through an activation function. The output of one layer becomes the input for the next layer until the final output is produced.

Slide 9: Source Code for Feedforward Process

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [Neuron(layer_sizes[i-1]) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def feedforward(self, inputs):
        for layer in self.layers:
            new_inputs = []
            for neuron in layer:
                activation = neuron.activate(inputs)
                new_inputs.append(activation)
            inputs = new_inputs
        return inputs

# Example usage
nn = NeuralNetwork([3, 4, 2])
input_data = [0.5, 0.3, 0.8]
output = nn.feedforward(input_data)
print(f"Input: {input_data}")
print(f"Output: {output}")
```

Slide 10: Backpropagation and Training

Backpropagation is the algorithm used to train neural networks. It calculates the gradient of the loss function with respect to the network's weights and biases. This gradient is then used to update the weights and biases, minimizing the error between the network's predictions and the actual target values. The process is repeated iteratively to improve the network's performance.

Slide 11: Source Code for Backpropagation and Training

```python
import random
import math

def mse_loss(y_true, y_pred):
    return sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred)) / len(y_true)

class NeuralNetwork:
    # ... (previous code for initialization and feedforward)

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(X, y):
                # Feedforward
                y_pred = self.feedforward(x)
                
                # Backpropagation
                output_deltas = [y_p * (1 - y_p) * (y_t - y_p) for y_p, y_t in zip(y_pred, y_true)]
                
                for i, layer in enumerate(reversed(self.layers)):
                    layer_errors = output_deltas if i == 0 else [
                        sum(n.weights[j] * e for n, e in zip(self.layers[-i], layer_errors))
                        for j in range(len(layer))
                    ]
                    layer_deltas = [e * n.last_activation * (1 - n.last_activation) for n, e in zip(layer, layer_errors)]
                    
                    for neuron, delta in zip(layer, layer_deltas):
                        neuron.bias += learning_rate * delta
                        neuron.weights = [w + learning_rate * delta * act for w, act in zip(neuron.weights, neuron.last_inputs)]
                
                total_loss += mse_loss(y_true, y_pred)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

# Example usage
nn = NeuralNetwork([2, 3, 1])
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]
nn.train(X, y, learning_rate=0.1, epochs=1000)

# Test the trained network
for x in X:
    print(f"Input: {x}, Output: {nn.feedforward(x)}")
```

Slide 12: Real-Life Example: Image Classification

Image classification is a common application of neural networks. In this example, we'll create a simple neural network to classify handwritten digits using the MNIST dataset. The network will take a 28x28 pixel image as input and output the predicted digit (0-9).

Slide 13: Source Code for Image Classification

```python
import numpy as np

def load_mnist():
    # Simulated MNIST data loading (replace with actual data loading in practice)
    X_train = np.random.rand(1000, 784)  # 1000 images, 28x28 pixels flattened
    y_train = np.random.randint(0, 10, 1000)  # 1000 labels (0-9)
    return X_train, y_train

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(self.z1, 0)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate):
        num_examples = X.shape[0]
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        delta3 /= num_examples
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * (self.a1 > 0)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Train the network
X_train, y_train = load_mnist()
nn = NeuralNetwork(784, 100, 10)

for epoch in range(100):
    probs = nn.forward(X_train)
    nn.backward(X_train, y_train, learning_rate=0.1)
    if epoch % 10 == 0:
        predictions = np.argmax(probs, axis=1)
        accuracy = np.mean(predictions == y_train)
        print(f"Epoch {epoch}, Accuracy: {accuracy:.2f}")

# Test the network
X_test = np.random.rand(10, 784)
predictions = np.argmax(nn.forward(X_test), axis=1)
print(f"Test predictions: {predictions}")
```

Slide 14: Real-Life Example: Natural Language Processing

Neural networks are widely used in Natural Language Processing (NLP) tasks. In this example, we'll create a simple sentiment analysis model that classifies movie reviews as positive or negative based on their text content.

Slide 15: Source Code for Natural Language Processing

```python
import re
import numpy as np

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def create_vocab(texts):
    vocab = set()
    for text in texts:
        vocab.update(preprocess_text(text))
    return {word: i for i, word in enumerate(vocab)}

def text_to_vector(text, vocab):
    words = preprocess_text(text)
    vector = np.zeros(len(vocab))
    for word in words:
        if word in vocab:
            vector[vocab[word]] += 1
    return vector

class SentimentClassifier:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))
        return self.a2

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            # Backpropagation (simplified)
            dZ2 = predictions - y
            dW2 = np.dot(self.a1.T, dZ2) / X.shape[0]
            db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
            dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 * (1 - self.a1))
            dW1 = np.dot(X.T, dZ1) / X.shape[0]
            db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

# Example usage
reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "Terrible film, waste of time.",
    "Great acting and plot, highly recommended!",
    "Boring and predictable, don't bother watching."
]
labels = np.array([[1], [0], [1], [0]])  # 1 for positive, 0 for negative

vocab = create_vocab(reviews)
X = np.array([text_to_vector(review, vocab) for review in reviews])

classifier = SentimentClassifier(len(vocab), 10)
classifier.train(X, labels, learning_rate=0.01, epochs=1000)

# Test the classifier
test_reviews = [
    "I enjoyed this movie, it was very entertaining.",
    "Disappointing storyline and poor acting.",
]
test_X = np.array([text_to_vector(review, vocab) for review in test_reviews])
predictions = classifier.forward(test_X)
print("Test Predictions:", predictions)
```

Slide 16: Optimization Techniques

Neural network training can be improved through various optimization techniques. These methods aim to accelerate convergence, avoid local minima, and enhance the overall performance of the model. Common optimization algorithms include Stochastic Gradient Descent (SGD), Adam, and RMSprop. These techniques adapt the learning rate and momentum to achieve better results.

Slide 17: Source Code for Optimization Techniques

```python
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, param, grad):
        return param - self.learning_rate * grad

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, param, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(param)
            self.v = np.zeros_like(param)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage
params = np.array([1.0, 2.0, 3.0])
grads = np.array([0.1, 0.2, 0.3])

sgd = SGD(learning_rate=0.1)
adam = Adam()

print("Original params:", params)
print("SGD update:", sgd.update(params, grads))
print("Adam update:", adam.update(params, grads))
```

Slide 18: Regularization Techniques

Regularization is crucial for preventing overfitting in neural networks. Common regularization techniques include L1 and L2 regularization, dropout, and early stopping. These methods help the model generalize better to unseen data by adding constraints or noise to the learning process.

Slide 19: Source Code for Regularization Techniques

```python
import numpy as np

def l1_regularization(weights, lambda_):
    return lambda_ * np.sum(np.abs(weights))

def l2_regularization(weights, lambda_):
    return 0.5 * lambda_ * np.sum(weights ** 2)

def dropout(layer_output, keep_prob):
    mask = np.random.binomial(1, keep_prob, size=layer_output.shape) / keep_prob
    return layer_output * mask

class NeuralNetworkWithRegularization:
    def __init__(self, input_size, hidden_size, output_size, lambda_=0.01, dropout_prob=0.5):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lambda_ = lambda_
        self.dropout_prob = dropout_prob

    def forward(self, X, training=True):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(self.z1, 0)  # ReLU activation
        
        if training:
            self.a1 = dropout(self.a1, self.dropout_prob)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate):
        # Simplified backpropagation with L2 regularization
        num_examples = X.shape[0]
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        delta3 /= num_examples
        
        dW2 = np.dot(self.a1.T, delta3) + self.lambda_ * self.W2
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * (self.a1 > 0)
        dW1 = np.dot(X.T, delta2) + self.lambda_ * self.W1
        db1 = np.sum(delta2, axis=0)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Example usage
nn = NeuralNetworkWithRegularization(input_size=10, hidden_size=5, output_size=3)
X = np.random.randn(100, 10)
y = np.random.randint(0, 3, 100)

for _ in range(1000):
    nn.forward(X, training=True)
    nn.backward(X, y, learning_rate=0.01)

test_X = np.random.randn(10, 10)
predictions = nn.forward(test_X, training=False)
print("Test predictions:", predictions)
```

Slide 20: Additional Resources

For further exploration of Artificial Neural Networks and deep learning, consider the following resources:

1.  "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016)
2.  "Neural Networks and Deep Learning" by Michael Nielsen (free online book)
3.  ArXiv.org - Search for recent papers on neural networks and deep learning Example: "Attention Is All You Need" by Vaswani et al. (2017) - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
4.  Stanford CS231n: Convolutional Neural Networks for Visual Recognition (course materials available online)
5.  Coursera's Deep Learning Specialization by Andrew Ng

These resources provide in-depth coverage of neural network architectures, training techniques, and applications in various domains.

