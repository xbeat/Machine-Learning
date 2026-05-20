## Calculus Foundations of Neural Networks
Slide 1: Understanding Gradient Descent in Neural Networks

Gradient descent forms the mathematical foundation of neural network training. It uses calculus to iteratively adjust network parameters by computing partial derivatives of the loss function with respect to each weight and bias, moving parameters in the direction that minimizes error.

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    # Initialize random weights
    weights = np.random.randn(X.shape[1])
    
    for epoch in range(epochs):
        # Forward pass
        prediction = np.dot(X, weights)
        
        # Compute gradients
        error = prediction - y
        gradient = np.dot(X.T, error) / len(X)
        
        # Update weights
        weights -= learning_rate * gradient
        
        # Compute loss
        loss = np.mean(error ** 2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    return weights
```

Slide 2: Implementing the Chain Rule for Backpropagation

The chain rule enables efficient computation of gradients through multiple layers by decomposing complex derivatives into simpler terms. This fundamental calculus concept allows us to determine how each parameter contributes to the final loss.

```python
def chain_rule_example(x, target):
    # Forward pass
    z = x * 2  # First operation
    y = z + 3  # Second operation
    loss = (y - target) ** 2  # Loss function
    
    # Backward pass using chain rule
    dloss_dy = 2 * (y - target)  # ∂loss/∂y
    dy_dz = 1  # ∂y/∂z
    dz_dx = 2  # ∂z/∂x
    
    # Final gradient using chain rule
    dloss_dx = dloss_dy * dy_dz * dz_dx
    
    return loss, dloss_dx
```

Slide 3: Sigmoid Activation Function and Its Derivative

The sigmoid activation function transforms linear inputs into probabilities between 0 and 1. Its derivative is crucial for backpropagation and exhibits a special property where it can be expressed in terms of the function itself.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# Visualize sigmoid and its derivative
x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, sigmoid_derivative(x), label='Derivative')
plt.legend()
plt.grid(True)
plt.title('Sigmoid Function and Its Derivative')
```

Slide 4: Neural Network Layer Implementation with Calculus

A neural network layer combines linear transformations with non-linear activations. The forward and backward passes utilize calculus principles to compute outputs and gradients respectively, forming the basis for learning.

```python
class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        self.activation = sigmoid(self.z)
        return self.activation
    
    def backward(self, gradient):
        batch_size = self.inputs.shape[0]
        
        # Gradient with respect to activation
        d_z = gradient * sigmoid_derivative(self.z)
        
        # Gradients with respect to weights and bias
        self.d_weights = np.dot(self.inputs.T, d_z) / batch_size
        self.d_bias = np.sum(d_z, axis=0, keepdims=True) / batch_size
        
        # Gradient for next layer
        d_inputs = np.dot(d_z, self.weights.T)
        return d_inputs
```

Slide 5: Loss Functions and Their Derivatives

Loss functions quantify the difference between predicted and actual outputs. Their derivatives provide the initial gradient signal for backpropagation. The mean squared error and cross-entropy loss are fundamental examples with distinct mathematical properties.

```python
class LossFunctions:
    @staticmethod
    def mse(y_true, y_pred):
        loss = np.mean((y_true - y_pred) ** 2)
        gradient = 2 * (y_pred - y_true) / y_true.shape[0]
        return loss, gradient
    
    @staticmethod
    def cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred))
        gradient = -(y_true / y_pred) / y_true.shape[0]
        return loss, gradient
```

Slide 6: Learning Rate Optimization with Calculus

Learning rate adaptation uses second-order derivatives to optimize the training process. This implementation shows how adaptive learning rates can be computed using gradient history.

```python
class AdaptiveLearningRate:
    def __init__(self, initial_lr=0.01, beta=0.9):
        self.lr = initial_lr
        self.beta = beta
        self.gradient_history = 0
        
    def compute_learning_rate(self, current_gradient):
        # Update gradient history using exponential moving average
        self.gradient_history = (self.beta * self.gradient_history + 
                               (1 - self.beta) * current_gradient**2)
        
        # Compute adaptive learning rate
        adapted_lr = self.lr / (np.sqrt(self.gradient_history) + 1e-8)
        return adapted_lr
```

Slide 7: Implementing ReLU and Its Derivative

The Rectified Linear Unit (ReLU) activation function demonstrates piecewise differentiation in calculus. Its derivative is a step function, making it computationally efficient and helping prevent vanishing gradients.

```python
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, gradient):
        # Derivative of ReLU is 1 for x > 0, 0 otherwise
        return gradient * (self.input > 0)

# Example usage and visualization
x = np.linspace(-5, 5, 100)
relu = ReLU()
y = relu.forward(x)
dy = relu.backward(np.ones_like(x))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, label='ReLU')
plt.title('ReLU Function')
plt.subplot(1, 2, 2)
plt.plot(x, dy, label='ReLU Derivative')
plt.title('ReLU Derivative')
plt.tight_layout()
```

Slide 8: Implementing Batch Normalization with Calculus

Batch normalization applies statistical normalization using calculus principles. The implementation requires computing means, variances, and their respective gradients during backpropagation.

```python
class BatchNormalization:
    def __init__(self, input_dim):
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        self.eps = 1e-8
        
    def forward(self, x):
        self.input = x
        self.mu = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.x_norm = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, gradient):
        m = self.input.shape[0]
        x_mu = self.input - self.mu
        
        # Gradients with respect to gamma and beta
        self.d_gamma = np.sum(gradient * self.x_norm, axis=0)
        self.d_beta = np.sum(gradient, axis=0)
        
        # Gradient with respect to input
        d_xnorm = gradient * self.gamma
        d_var = np.sum(d_xnorm * x_mu * -0.5 * 
                      (self.var + self.eps)**(-3/2), axis=0)
        d_mu = np.sum(d_xnorm * -1/np.sqrt(self.var + self.eps), axis=0)
        
        d_x = (d_xnorm / np.sqrt(self.var + self.eps) + 
               2 * d_var * x_mu/m + d_mu/m)
        
        return d_x
```

Slide 9: Numerical Gradient Checking Implementation

Gradient checking verifies analytical derivatives computed during backpropagation by comparing them with numerical approximations using finite differences. This essential debugging tool helps ensure correct gradient calculations.

```python
def gradient_check(model, x, y, epsilon=1e-7):
    # Get analytical gradients
    analytical_grads = model.get_gradients(x, y)
    
    # Compute numerical gradients for each parameter
    numerical_grads = []
    parameters = model.get_parameters()
    
    for param in parameters:
        param_grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'])
        
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            # Forward difference approximation
            param[idx] = old_value + epsilon
            cost_plus = model.compute_cost(x, y)
            
            param[idx] = old_value - epsilon
            cost_minus = model.compute_cost(x, y)
            
            # Restore parameter
            param[idx] = old_value
            
            # Compute numerical gradient
            param_grad[idx] = (cost_plus - cost_minus) / (2 * epsilon)
            it.iternext()
            
        numerical_grads.append(param_grad)
    
    # Compare gradients
    for analytical, numerical in zip(analytical_grads, numerical_grads):
        relative_error = np.abs(analytical - numerical) / \
                        (np.abs(analytical) + np.abs(numerical))
        print(f"Max relative error: {np.max(relative_error):.2e}")
```

Slide 10: Real-world Example: Time Series Prediction

This implementation demonstrates how calculus enables neural networks to learn temporal patterns in financial data through backpropagation through time (BPTT).

```python
class TimeSeriesNeuralNet:
    def __init__(self, input_size, hidden_size, sequence_length):
        self.Wx = np.random.randn(input_size, hidden_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b = np.zeros((1, hidden_size))
        self.sequence_length = sequence_length
        
    def forward(self, X):
        self.states = []
        h_t = np.zeros((X.shape[0], self.Wh.shape[1]))
        
        for t in range(self.sequence_length):
            # Store states for backpropagation
            self.states.append((X[:, t:t+1], h_t))
            
            # Forward step
            h_t = np.tanh(np.dot(X[:, t:t+1], self.Wx) + 
                         np.dot(h_t, self.Wh) + self.b)
        
        return h_t
    
    def backward(self, gradient, learning_rate=0.01):
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        
        # Backpropagate through time
        dh_next = gradient
        
        for t in reversed(range(self.sequence_length)):
            x_t, h_prev = self.states[t]
            
            # Compute gradients
            dtanh = dh_next * (1 - h_prev ** 2)
            dWx += np.dot(x_t.T, dtanh)
            dWh += np.dot(h_prev.T, dtanh)
            db += np.sum(dtanh, axis=0, keepdims=True)
            
            # Gradient for next iteration
            dh_next = np.dot(dtanh, self.Wh.T)
        
        # Update weights
        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.b -= learning_rate * db
```

Slide 11: Advanced Optimization: Natural Gradient Descent

Natural gradient descent uses differential geometry principles to optimize neural networks more efficiently by considering the manifold structure of the parameter space through the Fisher Information Matrix.

```python
class NaturalGradientOptimizer:
    def __init__(self, model, learning_rate=0.01, damping=1e-4):
        self.model = model
        self.lr = learning_rate
        self.damping = damping
        
    def compute_fisher_matrix(self, inputs, batch_size=32):
        fisher_matrix = 0
        samples = np.random.choice(len(inputs), batch_size)
        
        for idx in samples:
            # Get gradients for single sample
            grads = self.model.compute_gradients(inputs[idx:idx+1])
            grads_flat = np.concatenate([g.flatten() for g in grads])
            
            # Outer product to approximate Fisher
            fisher_matrix += np.outer(grads_flat, grads_flat)
            
        fisher_matrix /= batch_size
        # Add damping for numerical stability
        fisher_matrix += np.eye(fisher_matrix.shape[0]) * self.damping
        
        return fisher_matrix
    
    def update(self, inputs, gradients):
        # Compute Fisher Information Matrix
        fisher = self.compute_fisher_matrix(inputs)
        
        # Flatten gradients
        flat_grads = np.concatenate([g.flatten() for g in gradients])
        
        # Compute natural gradient direction
        natural_grad = np.linalg.solve(fisher, flat_grads)
        
        # Reshape back to original gradient shapes
        start = 0
        natural_gradients = []
        for g in gradients:
            size = g.size
            natural_gradients.append(
                natural_grad[start:start+size].reshape(g.shape)
            )
            start += size
            
        # Update model parameters
        self.model.update_parameters(natural_gradients, self.lr)
```

Slide 12: Implementing Second-Order Methods

Second-order optimization methods utilize Hessian calculations to achieve faster convergence by considering the curvature of the loss landscape.

```python
class NewtonOptimizer:
    def __init__(self, learning_rate=1.0):
        self.lr = learning_rate
        
    def compute_hessian(self, model, x, y):
        # Compute gradients
        grads = model.compute_gradients(x, y)
        param_shapes = [g.shape for g in grads]
        
        # Flatten parameters for Hessian computation
        flat_params = np.concatenate([p.flatten() for p in model.parameters])
        n_params = len(flat_params)
        
        # Initialize Hessian matrix
        hessian = np.zeros((n_params, n_params))
        
        # Compute second derivatives
        for i in range(n_params):
            # Compute diagonal elements
            grad_i = self.compute_directional_gradient(
                model, x, y, i, flat_params
            )
            hessian[i, i] = grad_i
            
            # Compute off-diagonal elements
            for j in range(i+1, n_params):
                grad_ij = self.compute_mixed_derivative(
                    model, x, y, i, j, flat_params
                )
                hessian[i, j] = grad_ij
                hessian[j, i] = grad_ij
        
        return hessian
    
    def compute_directional_gradient(self, model, x, y, idx, params, 
                                   epsilon=1e-7):
        # Compute second derivative in direction idx
        params_plus = params.copy()
        params_plus[idx] += epsilon
        grad_plus = model.compute_gradients(x, y, params_plus)
        
        params_minus = params.copy()
        params_minus[idx] -= epsilon
        grad_minus = model.compute_gradients(x, y, params_minus)
        
        return (grad_plus[idx] - grad_minus[idx]) / (2 * epsilon)
    
    def compute_mixed_derivative(self, model, x, y, i, j, params, 
                               epsilon=1e-7):
        # Compute mixed partial derivative
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_plus[j] += epsilon
        grad_plus = model.compute_gradients(x, y, params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= epsilon
        params_minus[j] -= epsilon
        grad_minus = model.compute_gradients(x, y, params_minus)
        
        return (grad_plus[i] - grad_minus[i]) / (4 * epsilon**2)
```

Slide 13: Practical Example: Image Classification with Calculus-Based Training

This implementation demonstrates a complete image classification system utilizing the calculus concepts covered, including gradient descent, backpropagation, and adaptive learning rates.

```python
class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.layers = []
        
        # Initialize network architecture
        self.add_conv_layer(32, (3, 3), activation='relu')
        self.add_max_pooling((2, 2))
        self.add_conv_layer(64, (3, 3), activation='relu')
        self.add_max_pooling((2, 2))
        self.add_dense(128, activation='relu')
        self.add_dense(num_classes, activation='softmax')
        
    def add_conv_layer(self, filters, kernel_size, activation):
        layer = ConvLayer(filters, kernel_size, activation)
        self.layers.append(layer)
    
    def forward(self, X):
        self.activations = []
        current_output = X
        
        for layer in self.layers:
            current_output = layer.forward(current_output)
            self.activations.append(current_output)
            
        return current_output
    
    def backward(self, gradient, learning_rate):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def train_step(self, X, y, learning_rate):
        # Forward pass
        predictions = self.forward(X)
        
        # Compute loss and gradient
        loss = cross_entropy_loss(y, predictions)
        gradient = softmax_gradient(y, predictions)
        
        # Backward pass
        self.backward(gradient, learning_rate)
        
        return loss

def train_network(model, X_train, y_train, epochs=10, batch_size=32):
    losses = []
    n_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            loss = model.train_step(X_batch, y_batch, 
                                  learning_rate=0.001)
            epoch_loss += loss
            
        losses.append(epoch_loss / n_batches)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses[-1]:.4f}")
```

Slide 14: Additional Resources

*   Backpropagation Through Time and Vanishing Gradients [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)
*   Natural Gradient Works Efficiently in Learning [https://arxiv.org/abs/1412.1193](https://arxiv.org/abs/1412.1193)
*   On the difficulty of training Recurrent Neural Networks [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063)
*   Second-Order Optimization for Neural Networks [https://www.google.com/search?q=second+order+optimization+neural+networks+research+papers](https://www.google.com/search?q=second+order+optimization+neural+networks+research+papers)
*   Adaptive Learning Methods for Neural Network Training [https://www.google.com/search?q=adaptive+learning+methods+neural+networks+papers](https://www.google.com/search?q=adaptive+learning+methods+neural+networks+papers)

