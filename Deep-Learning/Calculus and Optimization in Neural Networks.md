## Calculus and Optimization in Neural Networks
Slide 1: Neural Network Foundations and Calculus

The foundation of neural networks rests on calculus principles, particularly in optimization during training. Understanding these mathematical concepts is crucial for implementing effective deep learning solutions.

```python
import numpy as np

def sigmoid(z):
    """Basic sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid for backpropagation"""
    s = sigmoid(z)
    return s * (1 - s)

# Example usage
z = np.array([-2, -1, 0, 1, 2])
print(f"Sigmoid: {sigmoid(z)}")
print(f"Derivative: {sigmoid_derivative(z)}")
```

Slide 2: Gradient Descent Implementation

Gradient descent optimizes neural network parameters by iteratively adjusting weights and biases in the direction that minimizes the loss function, using partial derivatives to compute the steepest descent direction.

```python
class GradientDescent:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update_params(self, params, gradients):
        """Basic gradient descent update"""
        updated_params = {}
        for param_name in params:
            updated_params[param_name] = params[param_name] - self.learning_rate * gradients[param_name]
        return updated_params

# Example usage
params = {'W1': np.random.randn(3,2), 'b1': np.random.randn(2)}
gradients = {'W1': np.random.randn(3,2), 'b1': np.random.randn(2)}
optimizer = GradientDescent()
updated = optimizer.update_params(params, gradients)
```

Slide 3: Loss Function Derivatives

Loss functions measure model performance and guide optimization. Their derivatives indicate how to adjust parameters to minimize error, forming the basis for backpropagation.

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error loss and its derivative"""
    loss = np.mean((y_true - y_pred) ** 2)
    derivative = -2 * (y_true - y_pred) / y_true.shape[0]
    return loss, derivative

# Example
y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8])
loss, grad = mse_loss(y_true, y_pred)
print(f"Loss: {loss:.4f}\nGradient: {grad}")
```

Slide 4: Backpropagation Core Implementation

The backpropagation algorithm computes gradients efficiently by applying the chain rule of calculus, propagating errors backward through the network layers to update weights and biases.

```python
def backpropagation(layer_outputs, layer_inputs, weights, target):
    """Basic backpropagation implementation"""
    gradients = {}
    num_layers = len(layer_outputs)
    
    # Output layer error
    error = layer_outputs[-1] - target
    
    # Backward pass
    for l in reversed(range(num_layers)):
        delta = error * sigmoid_derivative(layer_outputs[l])
        gradients[f'W{l}'] = np.dot(layer_inputs[l].T, delta)
        error = np.dot(delta, weights[f'W{l}'].T)
    
    return gradients
```

Slide 5: Chain Rule Application

Understanding how the chain rule enables gradient computation through complex neural architectures is fundamental. This implementation demonstrates the mathematical principle in action.

```python
def chain_rule_example(x, w1, w2):
    """Demonstrates chain rule in neural networks"""
    # Forward pass
    $$z1 = w1 * x$$
    $$a1 = sigmoid(z1)$$
    $$z2 = w2 * a1$$
    $$output = sigmoid(z2)$$
    
    # Backward pass using chain rule
    $$\frac{\partial output}{\partial w1} = \frac{\partial output}{\partial z2} * 
                                          \frac{\partial z2}{\partial a1} * 
                                          \frac{\partial a1}{\partial z1} * 
                                          \frac{\partial z1}{\partial w1}$$
    
    d_output = sigmoid_derivative(z2)
    d_z2 = w2
    d_a1 = sigmoid_derivative(z1)
    d_z1 = x
    
    gradient_w1 = d_output * d_z2 * d_a1 * d_z1
    return gradient_w1
```

Slide 6: Optimization with Adam Algorithm

Adam combines momentum and adaptive learning rates for efficient neural network training. This implementation shows how to maintain running averages of gradients and their squares for parameter updates.

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, params, gradients):
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
            
        self.t += 1
        for param_name in params:
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradients[param_name]
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * np.square(gradients[param_name])
            
            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1**self.t)
            v_hat = self.v[param_name] / (1 - self.beta2**self.t)
            
            # Update parameters
            params[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
```

Slide 7: Neural Network Layer Implementation

A fundamental neural network layer combines linear transformation with non-linear activation, implementing forward and backward passes using matrix operations and calculus principles.

```python
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        self.activated = sigmoid(self.output)
        return self.activated
    
    def backward(self, grad_output):
        grad_activated = grad_output * sigmoid_derivative(self.output)
        grad_weights = np.dot(self.inputs.T, grad_activated)
        grad_bias = np.sum(grad_activated, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_activated, self.weights.T)
        return grad_inputs, grad_weights, grad_bias
```

Slide 8: Batch Normalization Implementation

Batch normalization accelerates training by normalizing layer inputs, requiring careful implementation of forward and backward passes with proper gradient computation.

```python
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
    def forward(self, x, training=True):
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Store for backward pass
            self.x = x
            self.x_norm = x_norm
            self.mean = mean
            self.var = var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            
        return self.gamma * x_norm + self.beta
```

Slide 9: Real-World Example - Time Series Prediction

Implementing a neural network for stock price prediction demonstrates practical application of calculus-based optimization in time series forecasting using historical data.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesNN:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.layer1 = Layer(lookback, 32)
        self.layer2 = Layer(32, 1)
        self.optimizer = Adam()
        
    def prepare_data(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y), scaler
    
    def forward(self, X):
        h1 = self.layer1.forward(X)
        output = self.layer2.forward(h1)
        return output
    
    def train_step(self, X_batch, y_batch):
        # Forward pass
        pred = self.forward(X_batch)
        loss = np.mean((pred - y_batch) ** 2)
        
        # Backward pass
        grad_output = 2 * (pred - y_batch) / len(y_batch)
        grad_h1, grad_w2, grad_b2 = self.layer2.backward(grad_output)
        _, grad_w1, grad_b1 = self.layer1.backward(grad_h1)
        
        # Update weights
        gradients = {
            'W1': grad_w1, 'b1': grad_b1,
            'W2': grad_w2, 'b2': grad_b2
        }
        params = {
            'W1': self.layer1.weights, 'b1': self.layer1.bias,
            'W2': self.layer2.weights, 'b2': self.layer2.bias
        }
        self.optimizer.update(params, gradients)
        return loss
```

Slide 10: Results for Time Series Prediction

```python
# Example usage with sample stock data
import yfinance as yf

# Download stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')['Close'].values

# Initialize and train model
model = TimeSeriesNN()
X, y, scaler = model.prepare_data(stock_data)

# Training loop
epochs = 50
batch_size = 32
losses = []

for epoch in range(epochs):
    epoch_losses = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        loss = model.train_step(X_batch, y_batch)
        epoch_losses.append(loss)
    losses.append(np.mean(epoch_losses))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")

# Results
print(f"Final Training Loss: {losses[-1]:.4f}")
```

Slide 11: Image Classification Neural Network

Real-world implementation of a convolutional neural network demonstrating the role of calculus in computer vision tasks through backpropagation and gradient-based learning.

```python
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(out_channels)
        
    def forward(self, x):
        self.input = x
        batch_size, in_channels, height, width = x.shape
        out_height = height - self.kernel.shape[2] + 1
        out_width = width - self.kernel.shape[3] + 1
        
        output = np.zeros((batch_size, self.kernel.shape[0], out_height, out_width))
        
        for b in range(batch_size):
            for c_out in range(self.kernel.shape[0]):
                for c_in in range(self.kernel.shape[1]):
                    for h in range(out_height):
                        for w in range(out_width):
                            output[b, c_out, h, w] += np.sum(
                                x[b, c_in, h:h+self.kernel.shape[2], w:w+self.kernel.shape[3]] * 
                                self.kernel[c_out, c_in]
                            )
                output[b, c_out] += self.bias[c_out]
                
        self.output = output
        return output

    def backward(self, grad_output):
        batch_size = self.input.shape[0]
        grad_kernel = np.zeros_like(self.kernel)
        grad_input = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for c_out in range(self.kernel.shape[0]):
                for c_in in range(self.kernel.shape[1]):
                    for h in range(grad_output.shape[2]):
                        for w in range(grad_output.shape[3]):
                            grad_kernel[c_out, c_in] += np.sum(
                                self.input[b, c_in, h:h+self.kernel.shape[2], w:w+self.kernel.shape[3]] * 
                                grad_output[b, c_out, h, w]
                            )
                            grad_input[b, c_in, h:h+self.kernel.shape[2], w:w+self.kernel.shape[3]] += \
                                self.kernel[c_out, c_in] * grad_output[b, c_out, h, w]
                                
        grad_bias = grad_output.sum(axis=(0, 2, 3))
        return grad_input, grad_kernel, grad_bias
```

Slide 12: Optimization Visualization

Implementation of gradient descent visualization to understand optimization trajectory in neural network training with real-time plotting.

```python
import matplotlib.pyplot as plt

def visualize_optimization(loss_function, start_point, learning_rate=0.1, iterations=100):
    """Visualize gradient descent optimization path"""
    x_history = [start_point[0]]
    y_history = [start_point[1]]
    z_history = [loss_function(start_point)]
    
    current_point = np.array(start_point)
    
    for _ in range(iterations):
        # Compute gradients
        grad = compute_gradient(loss_function, current_point)
        
        # Update point
        current_point = current_point - learning_rate * grad
        
        # Store history
        x_history.append(current_point[0])
        y_history.append(current_point[1])
        z_history.append(loss_function(current_point))
    
    # Create 3D visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot optimization path
    ax.plot(x_history, y_history, z_history, 'r-', linewidth=2, label='Optimization Path')
    ax.scatter(x_history[0], y_history[0], z_history[0], color='green', s=100, label='Start')
    ax.scatter(x_history[-1], y_history[-1], z_history[-1], color='red', s=100, label='End')
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Loss')
    ax.legend()
    
    return plt.gcf()

def compute_gradient(loss_function, point, epsilon=1e-7):
    """Compute numerical gradient of loss function"""
    grad = np.zeros_like(point)
    for i in range(len(point)):
        point_plus = point.copy()
        point_plus[i] += epsilon
        point_minus = point.copy()
        point_minus[i] -= epsilon
        grad[i] = (loss_function(point_plus) - loss_function(point_minus)) / (2 * epsilon)
    return grad
```

Slide 13: Additional Resources

*   "On the Convergence of Adam and Beyond" - [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   "Deep Learning with Differential Privacy" - [https://arxiv.org/abs/1607.00133](https://arxiv.org/abs/1607.00133)
*   "Neural Networks and Deep Learning: Mathematical Principles" - [https://arxiv.org/abs/1805.11544](https://arxiv.org/abs/1805.11544)
*   For additional reading on neural network optimization, search for:
    *   Gradient descent optimization algorithms
    *   Backpropagation through time
    *   Second-order optimization methods
    *   Natural gradient descent

