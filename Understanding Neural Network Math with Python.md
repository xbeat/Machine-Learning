## Understanding Neural Network Math with Python
Slide 1: Neural Network Fundamentals

In artificial neural networks, neurons are the basic computational units that process input signals through weighted connections. Each neuron receives multiple inputs, applies weights, adds a bias term, and processes the result through an activation function to produce an output signal.

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights randomly from normal distribution
        self.weights = np.random.randn(num_inputs)
        # Initialize bias to zero
        self.bias = 0
        
    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))
        
    def forward(self, inputs):
        # Calculate weighted sum plus bias
        z = np.dot(self.weights, inputs) + self.bias
        # Apply activation function
        return self.activation(z)

# Example usage
neuron = Neuron(3)
inputs = np.array([0.5, 0.3, 0.2])
output = neuron.forward(inputs)
print(f"Neuron output: {output}")
```

Slide 2: Understanding Activation Functions

Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Common choices include sigmoid, tanh, and ReLU. These functions determine how the weighted sum of inputs is transformed into the neuron's output signal.

```python
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

# Generate sample data
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(10, 6))
plt.plot(x, ActivationFunctions.sigmoid(x), label='Sigmoid')
plt.plot(x, ActivationFunctions.tanh(x), label='Tanh')
plt.plot(x, ActivationFunctions.relu(x), label='ReLU')
plt.grid(True)
plt.legend()
plt.title('Common Activation Functions')
plt.show()
```

Slide 3: Forward Propagation

Forward propagation is the process where input data flows through the network layer by layer. Each layer applies weights, biases, and activation functions to transform the data. This process continues until the final output layer produces predictions.

```python
import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        
    def forward(self, inputs):
        # Store inputs for backpropagation
        self.inputs = inputs
        # Compute output
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output

# Create sample network
input_layer = Layer(3, 4)
hidden_layer = Layer(4, 2)
output_layer = Layer(2, 1)

# Forward pass
x = np.random.randn(3, 1)  # Input
h1 = input_layer.forward(x)
h2 = hidden_layer.forward(h1)
output = output_layer.forward(h2)

print(f"Network output shape: {output.shape}")
print(f"Output: \n{output}")
```

Slide 4: Loss Functions and Gradients

The loss function quantifies how well the network's predictions match the true values. For training, we need to compute gradients of the loss with respect to weights and biases. This enables the network to adjust its parameters to minimize prediction errors.

```python
class LossFunctions:
    @staticmethod
    def mse_loss(y_true, y_pred):
        """Mean Squared Error Loss"""
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_gradient(y_true, y_pred):
        """Gradient of MSE loss"""
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """Binary Cross Entropy Loss"""
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def bce_gradient(y_true, y_pred):
        """Gradient of Binary Cross Entropy"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true/y_pred - (1-y_true)/(1-y_pred)) / y_true.size

# Example usage
y_true = np.array([[1], [0], [1]])
y_pred = np.array([[0.9], [0.1], [0.8]])

mse = LossFunctions.mse_loss(y_true, y_pred)
bce = LossFunctions.binary_cross_entropy(y_true, y_pred)

print(f"MSE Loss: {mse:.4f}")
print(f"BCE Loss: {bce:.4f}")
```

Slide 5: Backpropagation Algorithm

Backpropagation computes gradients using the chain rule of calculus. It propagates the error signal backwards through the network, calculating how each parameter contributed to the final prediction error. This information guides weight updates during training.

```python
def backpropagation(self, x, y):
    # Forward pass
    output = self.forward(x)
    
    # Calculate initial gradient from loss
    gradient = self.loss_gradient(y, output)
    
    # Backward pass through layers
    for layer in reversed(self.layers):
        # Gradient of weights
        layer.dW = np.dot(gradient, layer.input.T)
        # Gradient of bias
        layer.db = np.sum(gradient, axis=1, keepdims=True)
        # Gradient for next layer
        gradient = np.dot(layer.weights.T, gradient)
        
        # Update parameters
        layer.weights -= self.learning_rate * layer.dW
        layer.bias -= self.learning_rate * layer.db
```

Slide 6: Building a Neural Network Class

A complete neural network implementation combines all previous concepts into a cohesive class. This implementation includes initialization, forward propagation, backpropagation, and training methods for handling batches of data.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append({
                'weights': np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01,
                'bias': np.zeros((layer_sizes[i+1], 1)),
                'activations': None
            })
        self.learning_rate = 0.01
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        current_input = X
        for layer in self.layers:
            z = np.dot(layer['weights'], current_input) + layer['bias']
            layer['activations'] = self.sigmoid(z)
            current_input = layer['activations']
        return current_input

# Example initialization
nn = NeuralNetwork([3, 4, 1])
sample_input = np.random.randn(3, 1)
output = nn.forward(sample_input)
print(f"Network output: {output}")
```

Slide 7: Training Loop Implementation

The training loop orchestrates the learning process by repeatedly presenting data to the network, computing predictions, calculating errors, and updating weights through backpropagation until the model converges to optimal parameters.

```python
def train(self, X, y, epochs=1000):
    losses = []
    
    for epoch in range(epochs):
        # Forward propagation
        output = self.forward(X)
        
        # Calculate loss
        loss = np.mean(np.square(y - output))
        losses.append(loss)
        
        # Backpropagation
        error = y - output
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                layer_error = error
            else:
                layer_error = np.dot(self.layers[i+1]['weights'].T, layer_error)
            
            # Calculate gradients
            delta = layer_error * self.sigmoid_derivative(layer['activations'])
            layer['weights'] += self.learning_rate * np.dot(delta, 
                              self.layers[i-1]['activations'].T if i > 0 else X.T)
            layer['bias'] += self.learning_rate * np.sum(delta, axis=1, keepdims=True)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return losses
```

Slide 8: XOR Problem Implementation

The XOR problem is a classic example that demonstrates the power of neural networks. It requires learning a non-linear decision boundary, which is impossible for single-layer perceptrons but achievable with a multi-layer network.

```python
# XOR problem implementation
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

# Create and train network
xor_nn = NeuralNetwork([2, 4, 1])
losses = xor_nn.train(X, y, epochs=1000)

# Test the network
test_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
for test_input in test_inputs:
    prediction = xor_nn.forward(np.array(test_input).reshape(2, 1))
    print(f"Input: {test_input}, Prediction: {prediction[0][0]:.4f}")

# Plot training progress
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress on XOR Problem')
plt.show()
```

Slide 9: Mini-batch Gradient Descent

Mini-batch gradient descent optimizes training by processing small batches of data instead of single examples or the entire dataset. This approach balances computational efficiency with update stability and helps avoid local minima.

```python
class MiniBatchTrainer:
    def __init__(self, network, batch_size=32):
        self.network = network
        self.batch_size = batch_size
    
    def create_mini_batches(self, X, y):
        indices = np.random.permutation(X.shape[1])
        n_batches = X.shape[1] // self.batch_size
        batches = []
        
        for i in range(n_batches):
            batch_indices = indices[i*self.batch_size:(i+1)*self.batch_size]
            batches.append((X[:, batch_indices], y[:, batch_indices]))
        
        return batches
    
    def train_epoch(self, X, y):
        batches = self.create_mini_batches(X, y)
        epoch_loss = 0
        
        for batch_X, batch_y in batches:
            # Forward and backward pass for each batch
            output = self.network.forward(batch_X)
            loss = np.mean(np.square(batch_y - output))
            self.network.backward(batch_X, batch_y)
            epoch_loss += loss
            
        return epoch_loss / len(batches)

# Usage example
trainer = MiniBatchTrainer(nn, batch_size=16)
loss = trainer.train_epoch(X_train, y_train)
print(f"Average batch loss: {loss:.4f}")
```

Slide 10: Implementing Momentum Optimization

Momentum optimization accelerates gradient descent by accumulating a velocity vector in directions of persistent reduction in the objective function. This technique helps overcome local minima and speeds up convergence significantly.

```python
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def initialize(self, layers):
        # Initialize velocity vectors for each parameter
        for i, layer in enumerate(layers):
            self.velocities[f'W{i}'] = np.zeros_like(layer['weights'])
            self.velocities[f'b{i}'] = np.zeros_like(layer['bias'])
    
    def update(self, layers, gradients):
        for i, layer in enumerate(layers):
            # Update velocities and parameters for weights
            self.velocities[f'W{i}'] = (self.momentum * self.velocities[f'W{i}'] - 
                                      self.learning_rate * gradients[f'dW{i}'])
            layer['weights'] += self.velocities[f'W{i}']
            
            # Update velocities and parameters for biases
            self.velocities[f'b{i}'] = (self.momentum * self.velocities[f'b{i}'] - 
                                      self.learning_rate * gradients[f'db{i}'])
            layer['bias'] += self.velocities[f'b{i}']

# Example usage
optimizer = MomentumOptimizer()
optimizer.initialize(nn.layers)
# During training:
optimizer.update(nn.layers, gradients)
```

Slide 11: Regularization Implementation

Regularization techniques prevent overfitting by adding constraints to the network's parameters. L1 and L2 regularization penalize large weights, while dropout randomly deactivates neurons during training to create more robust features.

```python
class RegularizedNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, l2_lambda=0.01, dropout_rate=0.5):
        super().__init__(layer_sizes)
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        
    def forward_with_dropout(self, X, training=True):
        current_input = X
        dropout_masks = []
        
        for layer in self.layers:
            # Forward pass
            z = np.dot(layer['weights'], current_input) + layer['bias']
            activation = self.sigmoid(z)
            
            if training:
                # Apply dropout
                mask = np.random.binomial(1, 1-self.dropout_rate, 
                                        size=activation.shape) / (1-self.dropout_rate)
                activation *= mask
                dropout_masks.append(mask)
            
            layer['activations'] = activation
            current_input = activation
            
        return current_input, dropout_masks
    
    def compute_cost(self, y_pred, y_true):
        # MSE Loss with L2 regularization
        mse = np.mean(np.square(y_pred - y_true))
        l2_cost = 0
        for layer in self.layers:
            l2_cost += np.sum(np.square(layer['weights']))
        return mse + (self.l2_lambda / 2) * l2_cost

# Example usage
reg_nn = RegularizedNetwork([3, 4, 1], l2_lambda=0.01, dropout_rate=0.2)
output, masks = reg_nn.forward_with_dropout(X_sample, training=True)
loss = reg_nn.compute_cost(output, y_true)
```

Slide 12: Real-world Example: Binary Classification

Implementation of a neural network for binary classification using the Wisconsin Breast Cancer dataset. This example demonstrates data preprocessing, model training, and evaluation metrics calculation.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = load_breast_cancer()
X = data.data.T  # Shape: (features, samples)
y = data.target.reshape(1, -1)  # Shape: (1, samples)

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.2)
X_train, X_test = X_train.T, X_test.T  # Back to (features, samples)
y_train, y_test = y_train.T, y_test.T

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.T).T
X_test = scaler.transform(X_test.T).T

# Train model
model = NeuralNetwork([30, 16, 8, 1])  # 30 features
histories = model.train(X_train, y_train, epochs=1000)

# Evaluate model
y_pred = model.forward(X_test)
accuracy = np.mean((y_pred > 0.5) == y_test)
print(f"Test accuracy: {accuracy:.4f}")
```

Slide 13: Real-world Example: Regression Problem

A practical implementation for predicting housing prices demonstrates regression with neural networks. This example includes feature engineering, model architecture design, and regression-specific loss function implementation.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import pandas as pd

class RegressionNetwork(NeuralNetwork):
    def __init__(self, layer_sizes, learning_rate=0.001):
        super().__init__(layer_sizes)
        self.learning_rate = learning_rate
        
    def custom_loss(self, y_true, y_pred):
        # Mean Absolute Error (MAE) for regression
        return np.mean(np.abs(y_true - y_pred))
        
    def train_regression(self, X, y, epochs=1000, batch_size=32):
        history = {'mae': [], 'mse': []}
        
        for epoch in range(epochs):
            # Mini-batch training
            for i in range(0, X.shape[1], batch_size):
                batch_X = X[:, i:i+batch_size]
                batch_y = y[:, i:i+batch_size]
                
                # Forward and backward passes
                predictions = self.forward(batch_X)
                self.backward(batch_X, batch_y)
                
            # Calculate epoch metrics
            full_predictions = self.forward(X)
            mae = self.custom_loss(y, full_predictions)
            mse = np.mean(np.square(y - full_predictions))
            
            history['mae'].append(mae)
            history['mse'].append(mse)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: MAE = {mae:.4f}, MSE = {mse:.4f}")
                
        return history

# Load and prepare housing data
housing = fetch_california_housing()
X = housing.data.T  # (features, samples)
y = housing.target.reshape(1, -1)  # (1, samples)

# Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X.T).T
y = scaler_y.fit_transform(y.T).T

# Create and train model
model = RegressionNetwork([8, 16, 8, 1])  # 8 input features
history = model.train_regression(X, y, epochs=1000)

# Make predictions
y_pred = model.forward(X)
y_pred = scaler_y.inverse_transform(y_pred.T).T
print(f"Final RMSE: {np.sqrt(np.mean(np.square(y_pred - y))):.4f}")
```

Slide 14: Understanding Gradients and Weight Updates

The process of weight updates in neural networks involves careful calculation of gradients and their application through various optimization techniques. This implementation shows detailed gradient computation and parameter updates.

```python
class GradientAnalyzer:
    def __init__(self, network):
        self.network = network
        self.gradient_history = []
        
    def compute_gradients(self, layer_outputs, error):
        gradients = []
        current_error = error
        
        for i in reversed(range(len(self.network.layers))):
            layer = self.network.layers[i]
            
            # Compute local gradient
            local_grad = current_error * self.network.sigmoid_derivative(layer['activations'])
            
            # Compute weight and bias gradients
            if i > 0:
                input_activations = self.network.layers[i-1]['activations']
            else:
                input_activations = layer_outputs[0]
                
            weight_grad = np.dot(local_grad, input_activations.T)
            bias_grad = np.sum(local_grad, axis=1, keepdims=True)
            
            # Store gradients
            gradients.insert(0, {
                'weight_grad': weight_grad,
                'bias_grad': bias_grad,
                'mean_grad': np.mean(np.abs(weight_grad)),
                'max_grad': np.max(np.abs(weight_grad))
            })
            
            # Compute error for next layer
            if i > 0:
                current_error = np.dot(layer['weights'].T, local_grad)
                
        self.gradient_history.append(gradients)
        return gradients

# Example usage
analyzer = GradientAnalyzer(nn)
gradients = analyzer.compute_gradients(layer_outputs, error)
for i, grad in enumerate(gradients):
    print(f"Layer {i+1}:")
    print(f"Mean gradient magnitude: {grad['mean_grad']:.6f}")
    print(f"Max gradient magnitude: {grad['max_grad']:.6f}")
```

Slide 15: Additional Resources

*   ArXiv paper on neural network fundamentals: [https://arxiv.org/abs/1901.05639](https://arxiv.org/abs/1901.05639)
*   Comprehensive review of optimization techniques: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   Deep learning book (Goodfellow et al.): [http://www.deeplearningbook.org](http://www.deeplearningbook.org)
*   Neural Networks and Deep Learning online book: [http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)
*   Modern backpropagation techniques: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)
*   Practical recommendations for gradient-based training: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)

