## Feedforward Neural Network Fundamentals
Slide 1: Neural Network Fundamentals and Architecture

In feedforward neural networks, information flows unidirectionally from input to output layers through hidden layers. Each neuron receives inputs, applies weights, adds a bias term, and processes the result through an activation function to produce an output signal.

```python
import numpy as np

class Neuron:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        # Calculate weighted sum and add bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply activation function (ReLU)
        return max(0, weighted_sum)

# Example usage
neuron = Neuron(input_size=3)
sample_input = np.array([1.0, 2.0, 3.0])
output = neuron.forward(sample_input)
print(f"Neuron output: {output}")
```

Slide 2: Mathematical Foundation of Forward Propagation

Forward propagation involves matrix multiplication between input values and weights, followed by bias addition and activation function application. The mathematical representation helps understand the computational flow in neural networks.

```python
# Mathematical representation of forward propagation
"""
For a single neuron:
$$z = \sum_{i=1}^n w_i x_i + b$$
$$a = f(z)$$

For a layer:
$$Z = XW^T + b$$
$$A = f(Z)$$

Where:
- X: input matrix
- W: weight matrix
- b: bias vector
- f: activation function
"""
```

Slide 3: Implementing a Basic Neural Network Layer

```python
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        self.inputs = inputs
        # Matrix multiplication and bias addition
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def relu(self, x):
        return np.maximum(0, x)

# Example
layer = Layer(3, 2)
input_data = np.array([[1, 2, 3]])
output = layer.forward(input_data)
print(f"Layer output shape: {output.shape}")
print(f"Layer output: \n{output}")
```

Slide 4: Activation Functions and Their Implementation

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Common functions include ReLU, Sigmoid, and Tanh, each serving different purposes in neural network architectures.

```python
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
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Demonstration
x = np.array([-2, -1, 0, 1, 2])
act = ActivationFunctions()
print(f"ReLU: {act.relu(x)}")
print(f"Sigmoid: {act.sigmoid(x)}")
print(f"Tanh: {act.tanh(x)}")
```

Slide 5: Complete Feedforward Neural Network Implementation

```python
class FeedforwardNN:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1]))
            
    def forward(self, X):
        current_input = X
        for layer in self.layers[:-1]:
            # Pass through hidden layers with ReLU
            current_input = ActivationFunctions.relu(layer.forward(current_input))
        # Output layer with softmax
        output = ActivationFunctions.softmax(self.layers[-1].forward(current_input))
        return output

# Create and test network
nn = FeedforwardNN([4, 8, 6, 3])
sample_input = np.random.randn(2, 4)
output = nn.forward(sample_input)
print(f"Network output shape: {output.shape}")
print(f"Network output:\n{output}")
```

Slide 6: Data Preprocessing for Neural Networks

Data preprocessing is crucial for neural network performance. This implementation demonstrates normalization, standardization, and one-hot encoding techniques commonly used before feeding data into neural networks.

```python
class DataPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def standardize(self, X):
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)
    
    def one_hot_encode(self, y):
        n_classes = len(np.unique(y))
        return np.eye(n_classes)[y]

# Example usage
X = np.random.randn(100, 4)
y = np.random.randint(0, 3, 100)

preprocessor = DataPreprocessor()
X_standardized = preprocessor.standardize(X)
y_encoded = preprocessor.one_hot_encode(y)

print(f"Standardized data stats:\nMean: {X_standardized.mean():.6f}\nStd: {X_standardized.std():.6f}")
print(f"\nOne-hot encoded shape: {y_encoded.shape}")
```

Slide 7: Loss Functions Implementation

Loss functions measure the difference between predicted and actual values, guiding the network's learning process. This implementation covers common loss functions used in neural networks.

```python
class LossFunctions:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        $$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$
        """
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        """
        $$CCE = -\sum_{i=1}^n y_i \log(\hat{y}_i)$$
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Example
y_true = np.array([[1, 0, 0], [0, 1, 0]])
y_pred = np.array([[0.9, 0.1, 0], [0.1, 0.8, 0.1]])

loss = LossFunctions()
mse = loss.mean_squared_error(y_true, y_pred)
cce = loss.categorical_crossentropy(y_true, y_pred)

print(f"MSE Loss: {mse:.6f}")
print(f"CCE Loss: {cce:.6f}")
```

Slide 8: Real-world Example: Binary Classification

```python
class BinaryClassifier:
    def __init__(self, input_size):
        self.network = FeedforwardNN([input_size, 16, 8, 2])
        self.preprocessor = DataPreprocessor()
    
    def prepare_data(self, X, y):
        X_processed = self.preprocessor.standardize(X)
        y_encoded = self.preprocessor.one_hot_encode(y)
        return X_processed, y_encoded
    
    def predict(self, X):
        X_processed = self.preprocessor.standardize(X)
        predictions = self.network.forward(X_processed)
        return np.argmax(predictions, axis=1)

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(1000, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and use classifier
classifier = BinaryClassifier(input_size=5)
X_processed, y_encoded = classifier.prepare_data(X, y)
predictions = classifier.predict(X[:10])

print("Sample predictions:", predictions)
print("Actual values:", y[:10])
```

Slide 9: Implementing Mini-batch Processing

Mini-batch processing improves training efficiency by processing small batches of data simultaneously, leveraging matrix operations for faster computation and better generalization.

```python
def create_mini_batches(X, y, batch_size=32):
    indices = np.random.permutation(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    n_batches = X.shape[0] // batch_size
    mini_batches = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        mini_batches.append((
            X_shuffled[start_idx:end_idx],
            y_shuffled[start_idx:end_idx]
        ))
    
    # Handle remaining samples
    if X.shape[0] % batch_size != 0:
        mini_batches.append((
            X_shuffled[n_batches*batch_size:],
            y_shuffled[n_batches*batch_size:]
        ))
    
    return mini_batches

# Example usage
X = np.random.randn(100, 4)
y = np.random.randint(0, 2, 100)
batches = create_mini_batches(X, y, batch_size=32)
print(f"Number of mini-batches: {len(batches)}")
print(f"First batch shapes: X={batches[0][0].shape}, y={batches[0][1].shape}")
```

Slide 10: Implementing Weight Updates and Gradient Descent

The core of neural network learning lies in updating weights based on calculated gradients. This implementation demonstrates the basic weight update mechanism using gradient descent optimization.

```python
class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
    def update_weights(self, weights, gradients):
        """
        Weight update formula:
        $$W_{new} = W_{old} - \alpha \nabla W$$
        where α is the learning rate
        """
        return weights - self.learning_rate * gradients
    
    def update_layer(self, layer, weight_gradients, bias_gradients):
        layer.weights = self.update_weights(layer.weights, weight_gradients)
        layer.bias = self.update_weights(layer.bias, bias_gradients)

# Example usage
optimizer = GradientDescent(learning_rate=0.01)
layer = Layer(input_size=4, output_size=3)

# Simulate gradients
weight_gradients = np.random.randn(*layer.weights.shape)
bias_gradients = np.random.randn(*layer.bias.shape)

print("Before update:")
print(f"Weights mean: {layer.weights.mean():.6f}")
print(f"Bias mean: {layer.bias.mean():.6f}")

optimizer.update_layer(layer, weight_gradients, bias_gradients)

print("\nAfter update:")
print(f"Weights mean: {layer.weights.mean():.6f}")
print(f"Bias mean: {layer.bias.mean():.6f}")
```

Slide 11: Real-world Example: Multiclass Classification

```python
class MulticlassClassifier:
    def __init__(self, input_features, num_classes):
        self.model = FeedforwardNN([input_features, 32, 16, num_classes])
        self.preprocessor = DataPreprocessor()
        self.loss_function = LossFunctions.categorical_crossentropy
        
    def train_step(self, X_batch, y_batch):
        # Forward pass
        predictions = self.model.forward(X_batch)
        loss = self.loss_function(y_batch, predictions)
        
        # Here we would normally do backpropagation
        return predictions, loss

# Generate synthetic multiclass data
n_samples = 1000
n_features = 10
n_classes = 5

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

# Initialize and use classifier
classifier = MulticlassClassifier(n_features, n_classes)
X_processed, y_encoded = classifier.preprocessor.standardize(X), classifier.preprocessor.one_hot_encode(y)

# Process one batch
batch_size = 32
X_batch, y_batch = X_processed[:batch_size], y_encoded[:batch_size]
predictions, loss = classifier.train_step(X_batch, y_batch)

print(f"Batch loss: {loss:.6f}")
print(f"Prediction shape: {predictions.shape}")
print(f"First sample predictions:\n{predictions[0]}")
```

Slide 12: Implementing Forward Propagation with Momentum

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_weights = {}
        self.velocity_bias = {}
        
    def initialize_velocity(self, layer_id, weights_shape, bias_shape):
        self.velocity_weights[layer_id] = np.zeros(weights_shape)
        self.velocity_bias[layer_id] = np.zeros(bias_shape)
    
    def update_layer(self, layer_id, layer, gradients_w, gradients_b):
        """
        Momentum update formula:
        $$v_t = \beta v_{t-1} + (1 - \beta)\nabla W$$
        $$W_{t+1} = W_t - \alpha v_t$$
        """
        # Update weights with momentum
        self.velocity_weights[layer_id] = (self.momentum * self.velocity_weights[layer_id] + 
                                         self.learning_rate * gradients_w)
        layer.weights -= self.velocity_weights[layer_id]
        
        # Update bias with momentum
        self.velocity_bias[layer_id] = (self.momentum * self.velocity_bias[layer_id] + 
                                      self.learning_rate * gradients_b)
        layer.bias -= self.velocity_bias[layer_id]

# Example usage
optimizer = MomentumOptimizer(learning_rate=0.01, momentum=0.9)
layer = Layer(input_size=4, output_size=3)

# Initialize velocity for the layer
optimizer.initialize_velocity(0, layer.weights.shape, layer.bias.shape)

# Simulate gradients
gradients_w = np.random.randn(*layer.weights.shape)
gradients_b = np.random.randn(*layer.bias.shape)

print("Initial weights mean:", layer.weights.mean())
optimizer.update_layer(0, layer, gradients_w, gradients_b)
print("Updated weights mean:", layer.weights.mean())
```

Slide 13: Implementing Batch Normalization

Batch normalization stabilizes training by normalizing layer inputs, reducing internal covariate shift. This implementation shows how to normalize activations across mini-batches during forward propagation.

```python
class BatchNormalization:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * x_normalized + self.beta

# Example usage
batch_norm = BatchNormalization(num_features=4)
input_data = np.random.randn(32, 4)  # Batch size 32, 4 features

normalized_output = batch_norm.forward(input_data, training=True)
print(f"Input mean: {input_data.mean():.6f}, std: {input_data.std():.6f}")
print(f"Output mean: {normalized_output.mean():.6f}, std: {normalized_output.std():.6f}")
```

Slide 14: Complete Training Pipeline

```python
class NeuralNetworkTrainer:
    def __init__(self, model, learning_rate=0.01, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.optimizer = GradientDescent(learning_rate)
        self.history = {'loss': [], 'accuracy': []}
    
    def train_epoch(self, X, y, validation_split=0.2):
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Training
        batches = create_mini_batches(X_train, y_train, self.batch_size)
        epoch_loss = 0
        
        for X_batch, y_batch in batches:
            predictions = self.model.forward(X_batch)
            batch_loss = LossFunctions.categorical_crossentropy(y_batch, predictions)
            epoch_loss += batch_loss
            
        # Validation
        val_predictions = self.model.forward(X_val)
        val_loss = LossFunctions.categorical_crossentropy(y_val, val_predictions)
        
        return epoch_loss / len(batches), val_loss

# Example usage
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.eye(3)[np.random.randint(0, 3, 1000)]  # 3 classes

model = FeedforwardNN([10, 16, 8, 3])
trainer = NeuralNetworkTrainer(model, learning_rate=0.01, batch_size=32)

train_loss, val_loss = trainer.train_epoch(X, y)
print(f"Training loss: {train_loss:.6f}")
print(f"Validation loss: {val_loss:.6f}")
```

Slide 15: Additional Resources

*   ArXiv Papers and Resources:

*   "Deep Learning Book" by Goodfellow et al.: [https://www.deeplearningbook.org](https://www.deeplearningbook.org)
*   "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift": [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
*   "Dropout: A Simple Way to Prevent Neural Networks from Overfitting": [http://jmlr.org/papers/v15/srivastava14a.html](http://jmlr.org/papers/v15/srivastava14a.html)
*   Search Google Scholar for: "Neural Networks Fundamentals Recent Advances"
*   Visit PyTorch documentation for implementation references: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
*   TensorFlow guides for theoretical background: [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide)

