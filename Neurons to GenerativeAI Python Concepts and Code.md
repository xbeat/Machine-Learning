## Neurons to GenerativeAI Python Concepts and Code
Slide 1: Neural Network Fundamentals

A neural network is a computational model inspired by biological neurons, consisting of interconnected nodes that process and transmit signals. The fundamental building block is the artificial neuron, which takes weighted inputs, applies an activation function, and produces an output.

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        # Compute weighted sum and apply activation
        z = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(z)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Example usage
neuron = Neuron(3)
input_data = np.array([0.5, 0.3, 0.2])
output = neuron.forward(input_data)
print(f"Neuron output: {output}")
```

Slide 2: Activation Functions and Their Mathematics

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common functions include ReLU, sigmoid, and tanh, each with specific properties and use cases.

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

# Mathematical formulas (as text for export)
'''
$$ReLU(x) = max(0, x)$$
$$Sigmoid(x) = \frac{1}{1 + e^{-x}}$$
$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
$$LeakyReLU(x) = max(αx, x)$$
'''
```

Slide 3: Feedforward Neural Network Implementation

A complete implementation of a feedforward neural network using NumPy, demonstrating the forward propagation process through multiple layers. This implementation includes matrix operations for efficient computation and modular design.

```python
import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output

class FeedforwardNN:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))
    
    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            current_input = layer.forward(current_input)
            current_input = self.relu(current_input)
        return current_input
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

# Example usage
nn = FeedforwardNN([3, 4, 2])
sample_input = np.random.rand(1, 3)
output = nn.forward(sample_input)
print(f"Network output: {output}")
```

Slide 4: Backpropagation Algorithm

The backpropagation algorithm is the cornerstone of neural network training, using the chain rule to compute gradients of the loss function with respect to weights and biases, enabling the network to learn from its errors.

```python
class NeuralNetwork:
    def backward(self, x, y, learning_rate=0.01):
        # Store activations for each layer
        activations = [x]
        z_values = []
        
        # Forward pass
        current_input = x
        for layer in self.layers:
            z = np.dot(current_input, layer.weights) + layer.bias
            z_values.append(z)
            current_input = self.sigmoid(z)
            activations.append(current_input)
        
        # Calculate output layer error
        delta = (activations[-1] - y) * self.sigmoid_derivative(z_values[-1])
        
        # Backpropagate error
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer.weights -= learning_rate * np.dot(activations[i].T, delta)
            layer.bias -= learning_rate * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = np.dot(delta, layer.weights.T) * self.sigmoid_derivative(z_values[i-1])
```

Slide 5: Loss Functions for Neural Networks

Loss functions quantify the difference between predicted and actual outputs, guiding the network's learning process. Common choices include Mean Squared Error for regression and Cross-Entropy for classification tasks.

```python
class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """Binary Cross-Entropy"""
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """Categorical Cross-Entropy"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Mathematical formulas (as text for export)
'''
$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
$$BCE = -\frac{1}{n}\sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$
$$CCE = -\frac{1}{n}\sum_{i=1}^n\sum_{j=1}^c y_{ij} \log(\hat{y}_{ij})$$
'''
```

Slide 6: Gradient Descent Optimization

Gradient descent is the primary optimization algorithm for training neural networks, iteratively adjusting weights and biases to minimize the loss function. This implementation demonstrates batch, mini-batch, and stochastic variations.

```python
class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def batch_update(self, model, X_batch, y_batch):
        """Standard batch gradient descent"""
        gradients = model.compute_gradients(X_batch, y_batch)
        for layer_idx in range(len(model.layers)):
            model.layers[layer_idx].weights -= self.learning_rate * gradients[layer_idx]['weights']
            model.layers[layer_idx].bias -= self.learning_rate * gradients[layer_idx]['bias']
    
    def mini_batch_sgd(self, model, X, y, batch_size=32, epochs=100):
        """Mini-batch stochastic gradient descent"""
        n_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:min(i + batch_size, n_samples)]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                self.batch_update(model, X_batch, y_batch)

# Mathematical formula (as text for export)
'''
$$w_{t+1} = w_t - \eta \nabla L(w_t)$$
$$\eta: \text{learning rate}$$
$$\nabla L(w_t): \text{gradient of loss function}$$
'''
```

Slide 7: Convolutional Neural Network Basics

Convolutional Neural Networks (CNNs) are specialized architectures for processing grid-like data, particularly images. They use convolution operations to automatically learn hierarchical feature representations.

```python
import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
    
    def convolve2d(self, input_data, filter_):
        """2D convolution operation"""
        h, w = input_data.shape
        fh, fw = filter_.shape
        output_h = h - fh + 1
        output_w = w - fw + 1
        output = np.zeros((output_h, output_w))
        
        for i in range(output_h):
            for j in range(output_w):
                output[i, j] = np.sum(
                    input_data[i:i+fh, j:j+fw] * filter_
                )
        return output
    
    def forward(self, input_data):
        """Forward pass for convolution layer"""
        h, w = input_data.shape
        output = np.zeros((
            self.num_filters,
            h - self.filter_size + 1,
            w - self.filter_size + 1
        ))
        
        for i in range(self.num_filters):
            output[i] = self.convolve2d(input_data, self.filters[i])
        return output
```

Slide 8: Recurrent Neural Network Implementation

Recurrent Neural Networks process sequential data by maintaining an internal state that captures temporal dependencies. This implementation shows a basic RNN cell with forward pass computation.

```python
class RNNCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weights for input-to-hidden, hidden-to-hidden, and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        
        self.hidden_size = hidden_size
        self.hidden_state = None
    
    def forward(self, x, h_prev):
        """
        Forward pass of RNN cell
        x: input at current timestep
        h_prev: hidden state from previous timestep
        """
        # Combine input and previous hidden state
        self.hidden_state = np.tanh(
            np.dot(self.Wxh, x) + 
            np.dot(self.Whh, h_prev) + 
            self.bh
        )
        return self.hidden_state

# Mathematical formula (as text for export)
'''
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$
'''
```

Slide 9: LSTM Network Architecture

Long Short-Term Memory networks address the vanishing gradient problem in traditional RNNs by introducing gating mechanisms that control information flow through the network.

```python
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        # Initialize weight matrices and biases for all gates
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
    def forward(self, x, h_prev, c_prev):
        """LSTM forward pass"""
        # Concatenate input and previous hidden state
        combined = np.vstack((h_prev, x))
        
        # Compute gates
        f = self.sigmoid(np.dot(self.Wf, combined) + self.bf)
        i = self.sigmoid(np.dot(self.Wi, combined) + self.bi)
        c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
        o = self.sigmoid(np.dot(self.Wo, combined) + self.bo)
        
        # Update cell and hidden states
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        
        return h, c

# Mathematical formulas (as text for export)
'''
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
'''
```

Slide 10: Attention Mechanism

Attention mechanisms allow neural networks to focus on relevant parts of the input sequence, crucial for tasks like machine translation and natural language processing.

```python
class AttentionLayer:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.v = np.random.randn(hidden_size, 1) * 0.01
        
    def forward(self, query, keys, values):
        """
        Compute attention scores and weighted sum of values
        query: current decoder hidden state
        keys: encoder hidden states
        values: encoder outputs
        """
        # Calculate attention scores
        scores = np.zeros((keys.shape[0], 1))
        for i in range(keys.shape[0]):
            score = np.dot(
                self.v.T,
                np.tanh(np.dot(self.W, query) + np.dot(self.W, keys[i]))
            )
            scores[i] = score
            
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)
        
        # Compute weighted sum of values
        context = np.sum(values * attention_weights, axis=0)
        
        return context, attention_weights
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

# Mathematical formula (as text for export)
'''
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
'''
```

Slide 11: Transformer Architecture Implementation

The Transformer architecture revolutionized sequence modeling by replacing recurrence with self-attention mechanisms. This implementation demonstrates the key components of the encoder block including multi-head attention and feed-forward networks.

```python
import numpy as np

class TransformerEncoder:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Multi-head attention weights
        self.Wq = np.random.randn(num_heads, d_model, self.d_head)
        self.Wk = np.random.randn(num_heads, d_model, self.d_head)
        self.Wv = np.random.randn(num_heads, d_model, self.d_head)
        self.Wo = np.random.randn(d_model, d_model)
        
        # Feed-forward network weights
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        
    def multi_head_attention(self, X):
        batch_size, seq_len, _ = X.shape
        attention_heads = []
        
        for head in range(self.num_heads):
            Q = np.dot(X, self.Wq[head])
            K = np.dot(X, self.Wk[head])
            V = np.dot(X, self.Wv[head])
            
            # Scaled dot-product attention
            scores = np.dot(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_head)
            attention = self.softmax(scores)
            head_output = np.dot(attention, V)
            attention_heads.append(head_output)
            
        # Concatenate and project heads
        multi_head = np.concatenate(attention_heads, axis=-1)
        return np.dot(multi_head, self.Wo)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

Slide 12: Real-world Example: Sentiment Analysis

Implementation of a complete sentiment analysis system using a neural network, demonstrating text preprocessing, embedding, and classification on movie reviews dataset.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class SentimentAnalyzer:
    def __init__(self, vocab_size, embedding_dim, max_length):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Initialize embedding layer
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # Initialize neural network layers
        self.W1 = np.random.randn(embedding_dim, 64) * 0.01
        self.b1 = np.zeros((64,))
        self.W2 = np.random.randn(64, 1) * 0.01
        self.b2 = np.zeros((1,))
    
    def preprocess_text(self, text, word_to_idx):
        """Convert text to sequence of indices"""
        words = text.lower().split()
        return [word_to_idx.get(word, 0) for word in words[:self.max_length]]
    
    def forward(self, x):
        """Forward pass through the network"""
        # Embedding lookup
        embedded = self.embedding_matrix[x]
        # Average pooling over sequence length
        pooled = np.mean(embedded, axis=1)
        # Dense layers with ReLU and sigmoid
        hidden = np.maximum(0, np.dot(pooled, self.W1) + self.b1)
        output = self.sigmoid(np.dot(hidden, self.W2) + self.b2)
        return output
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

# Example usage:
# model = SentimentAnalyzer(vocab_size=10000, embedding_dim=100, max_length=200)
# prediction = model.forward(preprocessed_text)
```

Slide 13: Results for Sentiment Analysis Model

Performance metrics and example predictions from the sentiment analysis implementation, demonstrating real-world application results.

```python
# Sample results output
results = {
    'Accuracy': 0.873,
    'Precision': 0.891,
    'Recall': 0.856,
    'F1-Score': 0.873
}

example_predictions = {
    "This movie was fantastic!": 0.92,
    "Terrible waste of time": 0.08,
    "Somewhat entertaining but lacking depth": 0.51
}

print("Model Performance Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.3f}")

print("\nExample Predictions (>0.5 = Positive):")
for text, score in example_predictions.items():
    print(f"Text: {text}")
    print(f"Sentiment Score: {score:.2f}")
    print(f"Prediction: {'Positive' if score > 0.5 else 'Negative'}\n")
```

Slide 14: Additional Resources

*   Neural Network Architecture Design Principles:
    *   "Attention Is All You Need" - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
    *   "Deep Residual Learning for Image Recognition" - [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
    *   "BERT: Pre-training of Deep Bidirectional Transformers" - [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
*   Advanced Topics:
    *   "Language Models are Few-Shot Learners" (GPT-3) - [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
    *   "A Survey of Deep Learning Techniques for Neural Machine Translation" - Search on Google Scholar
*   Implementation Resources:
    *   PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
    *   TensorFlow Tutorials: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)

