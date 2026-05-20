## Fundamentals of Neural Networks
Slide 1: Neural Network Fundamentals

Neural networks consist of interconnected layers of artificial neurons that process information through weighted connections. Each neuron receives inputs, applies weights and biases, and produces an output through an activation function, mimicking biological neural systems in a simplified mathematical form.

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def activate(self, inputs):
        # Compute weighted sum and apply activation function
        z = np.dot(self.weights, inputs) + self.bias
        # Using sigmoid activation
        return 1 / (1 + np.exp(-z))

# Example usage
neuron = Neuron(3)
sample_input = np.array([0.5, 0.8, 0.1])
output = neuron.activate(sample_input)
print(f"Neuron output: {output}")
```

Slide 2: Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. These mathematical functions determine whether a neuron should be activated based on its input, transforming the weighted sum into a meaningful output signal.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Visualization
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.grid(True)
plt.legend()
plt.title('Common Activation Functions')
plt.show()
```

Slide 3: Forward Propagation

Forward propagation is the process where input data flows through the network layer by layer. Each layer applies weights, biases, and activation functions to transform the data, ultimately producing the network's prediction or output.

```python
import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.relu(self.output)
    
    def relu(self, x):
        return np.maximum(0, x)

# Create a simple network
layer1 = Layer(3, 4)
layer2 = Layer(4, 2)

# Forward pass
input_data = np.array([[1, 0.5, 0.2]])
hidden = layer1.forward(input_data)
output = layer2.forward(hidden)
print(f"Network output: {output}")
```

Slide 4: Backpropagation Theory

The backpropagation algorithm calculates gradients of the loss function with respect to network parameters using the chain rule. This mathematical process enables the network to learn by iteratively adjusting weights and biases to minimize prediction errors.

```python
# Mathematical representation in LaTeX notation
"""
$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}$$

$$\delta_j = \frac{\partial L}{\partial y_j} \cdot f'(z_j)$$

$$\Delta w_{ij} = -\eta \cdot \delta_j \cdot x_i$$
"""

# Note: These formulas represent:
# L: Loss function
# w_ij: Weight connecting neuron i to neuron j
# y_j: Output of neuron j
# z_j: Weighted sum input to neuron j
# η: Learning rate
```

Slide 5: Implementing Backpropagation

Neural networks learn through backpropagation by computing gradients and updating parameters. This implementation demonstrates the complete backward pass process, including gradient calculation and weight updates for a simple neural network layer.

```python
class NeuralLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.lr = learning_rate
        
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.output = self.sigmoid(self.z)
        return self.output
    
    def backward(self, output_error, output_delta):
        delta = output_delta * self.sigmoid_derivative(self.z)
        self.weights_grad = np.dot(self.inputs.T, delta)
        self.bias_grad = np.sum(delta, axis=0, keepdims=True)
        
        # Update parameters
        self.weights -= self.lr * self.weights_grad
        self.bias -= self.lr * self.bias_grad
        
        return np.dot(delta, self.weights.T)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
```

Slide 6: Loss Functions

Loss functions quantify the difference between predicted and actual outputs, guiding the network's learning process. Different tasks require specific loss functions - classification typically uses cross-entropy loss, while regression problems often employ mean squared error.

```python
import numpy as np

class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

# Example usage
y_true = np.array([[1, 0, 1, 0]])
y_pred = np.array([[0.9, 0.1, 0.8, 0.2]])

loss = LossFunctions()
mse_loss = loss.mse(y_true, y_pred)
bce_loss = loss.binary_cross_entropy(y_true, y_pred)

print(f"MSE Loss: {mse_loss}")
print(f"Binary Cross-Entropy Loss: {bce_loss}")
```

Slide 7: Complete Neural Network Implementation

A fully functional neural network implementation showcasing the integration of forward propagation, backpropagation, and training procedures. This implementation demonstrates the core concepts of deep learning in a practical context.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layers_dims):
        self.layers = []
        for i in range(len(layers_dims) - 1):
            self.layers.append(Layer(layers_dims[i], layers_dims[i+1]))
    
    def forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss and gradient
            loss = np.mean(np.square(output - y))
            error = 2 * (output - y) / y.shape[0]
            
            # Backward pass
            self.backward(error)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.relu(self.output)
    
    def backward(self, dout):
        dout = dout * self.relu_derivative(self.output)
        input_grad = np.dot(dout, self.weights.T)
        self.weights_grad = np.dot(self.input.T, dout)
        self.bias_grad = np.sum(dout, axis=0, keepdims=True)
        
        self.weights -= 0.01 * self.weights_grad
        self.bias -= 0.01 * self.bias_grad
        return input_grad
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

# Example usage
X = np.random.randn(100, 2)
y = np.sum(X, axis=1, keepdims=True) > 0

nn = NeuralNetwork([2, 4, 1])
nn.train(X, y, epochs=1000, learning_rate=0.01)
```

Slide 8: Data Preprocessing Techniques

Data preprocessing is crucial for neural network performance. This implementation demonstrates essential preprocessing steps including normalization, standardization, and one-hot encoding, which ensure optimal network training conditions.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse=False)
    
    def normalize(self, X):
        """Min-max normalization"""
        return (X - X.min()) / (X.max() - X.min())
    
    def standardize(self, X):
        """Z-score standardization"""
        return self.scaler.fit_transform(X)
    
    def one_hot_encode(self, y):
        """Convert categorical variables to binary format"""
        return self.encoder.fit_transform(y.reshape(-1, 1))

# Example usage
# Generate sample data
X = np.random.randn(100, 4)
y = np.random.randint(0, 3, size=(100,))

preprocessor = DataPreprocessor()

# Apply preprocessing
X_normalized = preprocessor.normalize(X)
X_standardized = preprocessor.standardize(X)
y_encoded = preprocessor.one_hot_encode(y)

print("Original data shape:", X.shape)
print("Processed features shape:", X_standardized.shape)
print("Encoded labels shape:", y_encoded.shape)
print("\nSample statistics:")
print("Normalized mean:", X_normalized.mean())
print("Normalized std:", X_normalized.std())
print("Standardized mean:", X_standardized.mean())
print("Standardized std:", X_standardized.std())
```

Slide 9: Image Classification Neural Network

A practical implementation of a convolutional neural network for image classification demonstrates how neural networks process visual data. This example shows the construction of layers specifically designed for image processing tasks.

```python
import numpy as np
from scipy.signal import convolve2d

class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
    
    def forward(self, input_data):
        self.input = input_data
        height, width = input_data.shape
        output = np.zeros((
            height - self.filter_size + 1,
            width - self.filter_size + 1,
            self.num_filters
        ))
        
        for i in range(self.num_filters):
            output[:,:,i] = convolve2d(input_data, self.filters[i], mode='valid')
        
        return output

class ImageClassifier:
    def __init__(self):
        self.conv1 = ConvolutionalLayer(num_filters=8, filter_size=3)
        self.conv2 = ConvolutionalLayer(num_filters=16, filter_size=3)
    
    def forward(self, image):
        # Assuming image is grayscale and normalized
        x = self.conv1.forward(image)
        x = np.maximum(0, x)  # ReLU activation
        x = self.conv2.forward(x[:,:,0])  # Using first channel for simplicity
        return x

# Example usage
sample_image = np.random.randn(28, 28)  # Simulating a MNIST-like image
classifier = ImageClassifier()
output = classifier.forward(sample_image)
print(f"Output shape: {output.shape}")
```

Slide 10: Natural Language Processing Implementation

Neural networks excel at processing sequential data like text. This implementation shows how to handle text data using embeddings and recurrent neural network structures for natural language processing tasks.

```python
import numpy as np

class TextProcessor:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1
        
    def embed_sequence(self, token_sequence):
        return np.array([self.embedding_matrix[token] for token in token_sequence])

class SimpleRNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.Wxh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))
        
    def forward(self, inputs):
        h = np.zeros((1, self.Whh.shape[0]))  # Initial hidden state
        self.hidden_states = []
        
        # Process sequence
        for x in inputs:
            h = np.tanh(np.dot(x, self.Wxh) + np.dot(h, self.Whh) + self.bh)
            self.hidden_states.append(h)
        
        # Output layer
        y = np.dot(h, self.Why) + self.by
        return self.softmax(y)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Example usage
vocab_size = 1000
embedding_dim = 50
hidden_dim = 128
output_dim = 10

text_processor = TextProcessor(vocab_size, embedding_dim)
rnn = SimpleRNN(embedding_dim, hidden_dim, output_dim)

# Simulate input sequence
token_sequence = np.random.randint(0, vocab_size, size=5)
embedded_sequence = text_processor.embed_sequence(token_sequence)
output = rnn.forward(embedded_sequence)

print(f"Input sequence shape: {embedded_sequence.shape}")
print(f"Output probabilities shape: {output.shape}")
```

Slide 11: Results Visualization and Model Evaluation

Proper evaluation and visualization of neural network results are crucial for understanding model performance. This implementation provides tools for analyzing model predictions and visualizing training progress.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

class ModelEvaluator:
    def __init__(self):
        self.training_loss = []
        self.validation_loss = []
        self.metrics = {}
    
    def log_training(self, train_loss, val_loss):
        self.training_loss.append(train_loss)
        self.validation_loss.append(val_loss)
    
    def plot_learning_curves(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_loss, label='Training Loss')
        plt.plot(self.validation_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def evaluate_classification(self, y_true, y_pred):
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        self.metrics['precision'] = precision
        self.metrics['recall'] = recall
        self.metrics['f1'] = f1
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.show()
        
        return self.metrics

# Example usage
evaluator = ModelEvaluator()

# Simulate training
for i in range(100):
    train_loss = np.exp(-i/20) + np.random.normal(0, 0.1)
    val_loss = np.exp(-i/20) + np.random.normal(0, 0.15)
    evaluator.log_training(train_loss, val_loss)

# Plot learning curves
evaluator.plot_learning_curves()

# Evaluate classification results
y_true = np.random.randint(0, 3, size=100)
y_pred = np.random.randint(0, 3, size=100)
metrics = evaluator.evaluate_classification(y_true, y_pred)
print("\nClassification Metrics:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")
```

Slide 12: Transfer Learning Implementation

Transfer learning leverages pre-trained models to solve new tasks efficiently. This implementation demonstrates how to adapt a pre-trained network for a new classification task while preserving learned feature representations.

```python
import numpy as np

class PretrainedNetwork:
    def __init__(self, input_size, hidden_sizes, num_classes):
        self.layers = []
        self.frozen_layers = []
        
        # Simulate pre-trained weights
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(layer_sizes)-1):
            layer = {
                'weights': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01,
                'bias': np.zeros((1, layer_sizes[i+1])),
                'frozen': True if i < len(layer_sizes)-2 else False
            }
            self.layers.append(layer)
        
        # Add new classification layer
        self.layers.append({
            'weights': np.random.randn(hidden_sizes[-1], num_classes) * 0.01,
            'bias': np.zeros((1, num_classes)),
            'frozen': False
        })
    
    def forward(self, X):
        self.activations = [X]
        for layer in self.layers:
            X = np.dot(X, layer['weights']) + layer['bias']
            X = self.relu(X)
            self.activations.append(X)
        return self.softmax(X)
    
    def backward(self, grad, learning_rate=0.01):
        for i in reversed(range(len(self.layers))):
            if not self.layers[i]['frozen']:
                layer_input = self.activations[i]
                weights_grad = np.dot(layer_input.T, grad)
                bias_grad = np.sum(grad, axis=0, keepdims=True)
                
                # Update parameters
                self.layers[i]['weights'] -= learning_rate * weights_grad
                self.layers[i]['bias'] -= learning_rate * bias_grad
                
                # Compute gradient for next layer
                grad = np.dot(grad, self.layers[i]['weights'].T)
                grad = grad * (self.activations[i] > 0)  # ReLU derivative
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Example usage
model = PretrainedNetwork(
    input_size=784,    # e.g., MNIST image size
    hidden_sizes=[512, 256],
    num_classes=10
)

# Simulate training data
X = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100, 1))

# Forward pass
predictions = model.forward(X)

# Calculate gradient (simplified)
y_one_hot = np.zeros((100, 10))
y_one_hot[np.arange(100), y.flatten()] = 1
gradient = predictions - y_one_hot

# Backward pass (update only non-frozen layers)
model.backward(gradient)

print("Frozen layers:", [i for i, layer in enumerate(model.layers) if layer['frozen']])
print("Trainable layers:", [i for i, layer in enumerate(model.layers) if not layer['frozen']])
```

Slide 13: Hyperparameter Optimization

Hyperparameter optimization is essential for achieving optimal neural network performance. This implementation showcases various techniques for automatically tuning network parameters using grid search and random search methods.

```python
import numpy as np
from itertools import product

class HyperparameterOptimizer:
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.results = []
    
    def grid_search(self, train_fn, X, y, cv_folds=3):
        param_combinations = [dict(zip(self.param_grid.keys(), v)) 
                            for v in product(*self.param_grid.values())]
        
        best_score = float('-inf')
        best_params = None
        
        for params in param_combinations:
            scores = []
            # Cross-validation
            fold_size = len(X) // cv_folds
            for i in range(cv_folds):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size
                
                X_val = X[start_idx:end_idx]
                y_val = y[start_idx:end_idx]
                X_train = np.concatenate([X[:start_idx], X[end_idx:]])
                y_train = np.concatenate([y[:start_idx], y[end_idx:]])
                
                score = train_fn(X_train, y_train, X_val, y_val, **params)
                scores.append(score)
            
            avg_score = np.mean(scores)
            self.results.append({
                'params': params,
                'score': avg_score,
                'std': np.std(scores)
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        return best_params, best_score
    
    def random_search(self, train_fn, X, y, n_iter=10):
        best_score = float('-inf')
        best_params = None
        
        for _ in range(n_iter):
            params = {k: np.random.choice(v) for k, v in self.param_grid.items()}
            score = train_fn(X, y, **params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score

# Example usage
def dummy_train_fn(X_train, y_train, X_val, y_val, learning_rate, hidden_size):
    # Simulate training and return validation score
    return np.random.random() * learning_rate * hidden_size

param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [64, 128, 256]
}

optimizer = HyperparameterOptimizer(param_grid)

# Generate dummy data
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, size=1000)

# Perform grid search
best_params, best_score = optimizer.grid_search(dummy_train_fn, X, y)
print("Grid Search Results:")
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")

# Perform random search
best_params, best_score = optimizer.random_search(dummy_train_fn, X, y)
print("\nRandom Search Results:")
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")
```

Slide 14: Additional Resources

*   ArXiv Papers:
*   Deep Learning Review: [https://arxiv.org/abs/1404.7828](https://arxiv.org/abs/1404.7828)
*   Neural Networks and Deep Learning: [https://arxiv.org/abs/1511.08458](https://arxiv.org/abs/1511.08458)
*   Optimization Methods for Deep Learning: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   Recommended Search Terms:
*   "Neural Network Architecture Design"
*   "Deep Learning Optimization Techniques"
*   "Transfer Learning in Neural Networks"
*   Online Resources:
*   Deep Learning Specialization (Coursera)
*   Fast.ai Practical Deep Learning Course
*   TensorFlow and PyTorch Documentation

