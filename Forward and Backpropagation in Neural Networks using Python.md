## Forward and Backpropagation in Neural Networks using Python

Slide 1: Introduction to Neural Networks

Neural Networks are computational models inspired by the human brain, capable of learning from data and making predictions or decisions. They are composed of interconnected nodes called neurons, organized in layers: input, hidden, and output layers. Neural Networks are widely used in various applications, including image recognition, natural language processing, and predictive analytics.

Code:

```python
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Example output data
y = np.array([[0], [1], [1], [0]])
```

Slide 2: Forward Propagation

Forward propagation is the process of computing the outputs of a neural network by passing the input data through the network's layers. It involves calculating the weighted sum of the inputs and applying an activation function at each neuron. The outputs of one layer become the inputs for the next layer, propagating forward until reaching the output layer.

Code:

```python
# Example neural network with one hidden layer
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
```

Slide 3: Backpropagation

Backpropagation is the algorithm used to train neural networks by adjusting the weights and biases to minimize the error between the predicted outputs and the actual outputs. It involves computing the gradients of the error with respect to the weights and biases, and updating them in the opposite direction of the gradients to reduce the error.

Code:

```python
# Backpropagation algorithm
def backpropagation(self, X, y, learning_rate):
    m = X.shape[0]
    y_pred = self.forward(X)
    
    # Compute gradients
    delta2 = y_pred - y
    dW2 = np.dot(self.a1.T, delta2) / m
    db2 = np.sum(delta2, axis=0, keepdims=True) / m
    delta1 = np.dot(delta2, self.W2.T) * self.a1 * (1 - self.a1)
    dW1 = np.dot(X.T, delta1) / m
    db1 = np.sum(delta1, axis=0, keepdims=True) / m
    
    # Update weights and biases
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
```

Slide 4: Activation Functions

Activation functions are mathematical functions applied to the weighted sum of inputs in a neural network's neurons. They introduce non-linearity, allowing the network to learn complex patterns in the data. Common activation functions include sigmoid, tanh, and ReLU (Rectified Linear Unit).

Code:

```python
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperbolic tangent (tanh) activation function
def tanh(x):
    return np.tanh(x)

# Rectified Linear Unit (ReLU) activation function
def relu(x):
    return np.maximum(0, x)
```

Slide 5: Loss Functions

Loss functions measure the difference between the predicted outputs of a neural network and the true outputs during training. Common loss functions include Mean Squared Error (MSE) for regression problems and Cross-Entropy Loss for classification problems. The goal of training is to minimize the loss function.

Code:

```python
import numpy as np

# Mean Squared Error (MSE) loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Binary Cross-Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

Slide 6: Optimization Algorithms

Optimization algorithms are used to update the weights and biases of a neural network during the training process, based on the computed gradients from backpropagation. Popular optimization algorithms include Stochastic Gradient Descent (SGD), Momentum, RMSprop, and Adam.

Code:

```python
# Stochastic Gradient Descent (SGD) optimizer
def sgd_update(params, grads, lr):
    for param, grad in zip(params, grads):
        param -= lr * grad

# Momentum optimizer
def momentum_update(params, grads, lr, velocities, momentum):
    for param, grad, velocity in zip(params, grads, velocities):
        velocity = momentum * velocity + lr * grad
        param -= velocity
```

Slide 7: Training a Neural Network

Training a neural network involves iteratively updating the weights and biases using an optimization algorithm and minimizing the loss function over the training data. This process is repeated for multiple epochs, allowing the network to learn the patterns in the data.

Code:

```python
import numpy as np

# Training function
def train(model, X_train, y_train, epochs, lr):
    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        loss = mse_loss(y_train, y_pred)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        model.backpropagation(X_train, y_train, lr)

# Example usage
model = NeuralNetwork(2, 4, 1)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
train(model, X_train, y_train, epochs=10000, lr=0.1)
```

Slide 8: Regularization Techniques

Regularization techniques are used to prevent overfitting in neural networks, which occurs when the model learns the training data too well and fails to generalize to new, unseen data. Common regularization techniques include L1/L2 regularization, dropout, and early stopping.

Code:

```python
# L2 regularization
def l2_regularization(weights, lambda_):
    regularization_term = 0
    for weight in weights:
        regularization_term += np.sum(np.square(weight))
    return lambda_ * regularization_term / (2 * len(weights))

# Dropout regularization
def dropout(X, keep_prob):
    mask = np.random.rand(*X.shape) < keep_prob
    return X * mask / keep_prob
```

Slide 9: Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a type of neural network designed for processing grid-like data, such as images. They use convolutional layers to extract local features and pooling layers to reduce the spatial dimensions, making them particularly effective for tasks like image recognition and classification.

Code:

```python
import numpy as np

# Example convolutional layer
def conv2d(X, W, stride=1, padding=0):
    n_x, d_x, h_x, w_x = X.shape
    n_f, d_f, f_h, f_w = W.shape
    h_out = (h_x - f_h + 2 * padding) // stride + 1
    w_out = (w_x - f_w + 2 * padding) // stride + 1
    
    outputs = np.zeros((n_x, n_f, h_out, w_out))
    
    # Convolutional operation
    for x in range(n_x):
        for f in range(n_f):
            for h in range(h_out):
                for w in range(w_out):
                    outputs[x, f, h, w] = np.sum(
                        X[x, :, h*stride:h*stride+f_h, w*stride:w*stride+f_w] *
                        W[f, :, :, :]
                    )
    
    return outputs
```

Slide 10: Pooling Layers

Pooling layers are used in CNNs to reduce the spatial dimensions of the feature maps, effectively summarizing the presence of features in different regions. Common pooling operations include max pooling and average pooling.

Code:

```python
import numpy as np

# Max pooling
def max_pool(X, f=2, stride=2):
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - f) // stride + 1
    w_out = (w_x - f) // stride + 1
    
    outputs = np.zeros((n_x, d_x, h_out, w_out))
    
    for x in range(n_x):
        for d in range(d_x):
            for h in range(h_out):
                for w in range(w_out):
                    outputs[x, d, h, w] = np.max(
                        X[x, d, h*stride:h*stride+f, w*stride:w*stride+f]
                    )
    
    return outputs
```

Slide 11: Transfer Learning

Transfer learning is a technique in deep learning where a pre-trained model, trained on a large dataset, is used as a starting point for a new task. The pre-trained weights are fine-tuned on the new dataset, leveraging the learned features and reducing the training time and data requirements.

Code:

```python
import torch
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Add custom layers
model.fc = torch.nn.Linear(512, 10)  # Example: 10 output classes

# Fine-tune the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001)

# Train the custom layers
for epoch in range(num_epochs):
    # Training loop
    ...
```

Slide 12: Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a type of neural network designed for processing sequential data, such as text or time series data. They maintain an internal state that captures information from previous inputs, allowing them to model dependencies and patterns in sequential data.

Code:

```python
import torch
import torch.nn as nn

# Simple RNN cell
class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_ih = nn.Parameter(torch.randn(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.randn(hidden_size))
        self.b_hh = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, x, h):
        # x: input, h: hidden state
        z = torch.mm(x, self.W_ih) + self.b_ih + torch.mm(h, self.W_hh) + self.b_hh
        h_next = torch.tanh(z)
        return h_next
```

Slide 13: Attention Mechanisms

Attention mechanisms are a technique used in deep learning models, particularly in natural language processing and computer vision tasks, to focus on the most relevant parts of the input data when making predictions. They allow the model to selectively attend to different parts of the input, improving performance on tasks that require capturing long-range dependencies.

Code:

```python
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        
    def forward(self, queries, keys, values):
        q = self.query(queries)
        k = self.key(keys)
        v = self.value(values)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        weights = nn.Softmax(dim=-1)(scores)
        output = torch.matmul(weights, v)
        
        return output
```

Slide 14 (Additional Resources): Further Reading

For those interested in exploring more advanced topics or diving deeper into the subject, here are some additional resources from arXiv.org:

1. "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville ([https://arxiv.org/abs/1609.08144](https://arxiv.org/abs/1609.08144))
2. "Attention Is All You Need" by Ashish Vaswani et al. ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))
3. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Nitish Srivastava et al. ([https://arxiv.org/abs/1207.0580](https://arxiv.org/abs/1207.0580))
4. "Adam: A Method for Stochastic Optimization" by Diederik P. Kingma and Jimmy Ba ([https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980))

These resources cover various aspects of neural networks, including attention mechanisms, regularization techniques, and optimization algorithms, providing a deeper understanding and insights into the field.

