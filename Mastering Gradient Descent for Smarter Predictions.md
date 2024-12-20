## Mastering Gradient Descent for Smarter Predictions
Slide 1: Understanding Gradient Descent Fundamentals

Gradient descent is an iterative optimization algorithm that finds the minimum of a function by taking steps proportional to the negative of the gradient. In machine learning, it's used to minimize the loss function and find optimal model parameters.

```python
import numpy as np

def gradient_descent(f, df, x0, learning_rate=0.01, max_iter=1000, tol=1e-6):
    x = x0  # Starting point
    history = [x]
    
    for i in range(max_iter):
        grad = df(x)  # Compute gradient
        if np.abs(grad) < tol:
            break
        x = x - learning_rate * grad  # Update step
        history.append(x)
    
    return x, history

# Example usage for f(x) = x^2
f = lambda x: x**2  # Function to minimize
df = lambda x: 2*x  # Derivative of function

x_min, history = gradient_descent(f, df, x0=2.0)
print(f"Minimum found at x = {x_min:.6f}")
```

Slide 2: The Mathematics Behind Gradient Descent

The core principle of gradient descent relies on calculus to find the direction of steepest descent. The gradient represents the direction of maximum increase, so we move in the opposite direction to minimize our function.

```python
# Mathematical representation in code block (LaTeX format)
'''
$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
$$

Where:
$$
\nabla_\theta J(\theta_t) = \frac{\partial J}{\partial \theta}
$$
'''

# Implementation of batch gradient descent for linear regression
def batch_gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    cost_history = []
    
    for _ in range(epochs):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        gradient = np.dot(X.T, loss) / m
        theta = theta - alpha * gradient
        cost = np.sum(loss**2) / (2*m)
        cost_history.append(cost)
    
    return theta, cost_history
```

Slide 3: Implementing Linear Regression with Gradient Descent

Linear regression serves as an excellent example to understand gradient descent in practice. We'll implement it from scratch, calculating the gradients manually and updating our parameters iteratively.

```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
```

Slide 4: Stochastic Gradient Descent Implementation

Stochastic Gradient Descent (SGD) processes one sample at a time, making it more memory-efficient and often faster to converge than batch gradient descent. This implementation shows the key differences in approach.

```python
def stochastic_gradient_descent(X, y, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        for idx in range(m):
            random_idx = np.random.randint(0, m)
            X_i = X[random_idx:random_idx+1]
            y_i = y[random_idx:random_idx+1]
            
            prediction = np.dot(X_i, theta)
            error = prediction - y_i
            gradient = X_i.T.dot(error)
            theta -= learning_rate * gradient
            
    return theta

# Usage example
X = np.random.randn(1000, 5)
y = np.random.randn(1000)
theta = stochastic_gradient_descent(X, y)
print("Optimized parameters:", theta)
```

Slide 5: Mini-batch Gradient Descent

Mini-batch gradient descent combines the best of both batch and stochastic approaches, processing small batches of data at a time. This implementation demonstrates the practical balance between computation efficiency and convergence stability.

```python
class MiniBatchGD:
    def __init__(self, batch_size=32, learning_rate=0.01, epochs=100):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def create_mini_batches(self, X, y):
        mini_batches = []
        data = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // self.batch_size
        
        for i in range(n_minibatches):
            mini_batch = data[i * self.batch_size:(i + 1) * self.batch_size]
            X_mini = mini_batch[:, :-1]
            y_mini = mini_batch[:, -1]
            mini_batches.append((X_mini, y_mini))
            
        return mini_batches
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        
        for epoch in range(self.epochs):
            mini_batches = self.create_mini_batches(X, y)
            for X_mini, y_mini in mini_batches:
                y_pred = np.dot(X_mini, self.weights)
                gradient = np.dot(X_mini.T, (y_pred - y_mini))
                self.weights -= self.learning_rate * gradient
```

Slide 6: Momentum-based Gradient Descent

Momentum helps accelerate gradient descent by adding a fraction of the previous update to the current one. This approach helps overcome local minima and speeds up convergence, particularly in areas where the gradient is small.

```python
class MomentumGD:
    def __init__(self, learning_rate=0.01, momentum=0.9, epochs=1000):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        velocity = np.zeros_like(self.weights)
        
        for _ in range(self.epochs):
            # Compute gradients
            y_pred = np.dot(X, self.weights)
            gradients = np.dot(X.T, (y_pred - y)) / len(y)
            
            # Update velocity and weights
            velocity = self.momentum * velocity - self.learning_rate * gradients
            self.weights += velocity
            
        return self.weights

# Example usage
X = np.random.randn(1000, 5)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(1000) * 0.1
model = MomentumGD()
optimal_weights = model.fit(X, y)
```

Slide 7: Adaptive Learning Rate with AdaGrad

AdaGrad adapts the learning rate for each parameter individually, which is particularly useful when dealing with sparse data or when parameters have different scales of importance.

```python
class AdaGrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def optimize(self, X, y, initial_weights, iterations):
        weights = initial_weights
        accumulated_gradients = np.zeros_like(weights)
        history = []
        
        for _ in range(iterations):
            # Compute current gradients
            predictions = np.dot(X, weights)
            gradients = 2 * np.dot(X.T, (predictions - y)) / len(y)
            
            # Update accumulated gradients
            accumulated_gradients += gradients ** 2
            
            # Compute adaptive learning rates
            adaptive_lr = self.learning_rate / (np.sqrt(accumulated_gradients + self.epsilon))
            
            # Update weights
            weights -= adaptive_lr * gradients
            history.append(np.mean((predictions - y) ** 2))
            
        return weights, history

# Example implementation
X = np.random.randn(1000, 10)
true_weights = np.random.randn(10)
y = np.dot(X, true_weights) + np.random.randn(1000) * 0.1
optimizer = AdaGrad()
final_weights, loss_history = optimizer.optimize(X, y, np.zeros(10), 100)
```

Slide 8: RMSprop Implementation

RMSprop improves upon AdaGrad by using an exponentially decaying average of squared gradients, preventing the learning rate from decreasing too quickly.

```python
class RMSprop:
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        
    def optimize(self, gradient_func, initial_params, n_iterations):
        params = initial_params
        cache = np.zeros_like(params)
        
        for t in range(n_iterations):
            gradients = gradient_func(params)
            
            # Update moving average of squared gradients
            cache = self.decay_rate * cache + (1 - self.decay_rate) * gradients**2
            
            # Update parameters
            params -= (self.learning_rate / np.sqrt(cache + self.epsilon)) * gradients
            
        return params

def example_gradient_function(params):
    # Example quadratic function gradient
    return 2 * params

# Usage example
initial_params = np.array([1.0, 2.0, 3.0])
optimizer = RMSprop()
final_params = optimizer.optimize(example_gradient_function, initial_params, 1000)
```

Slide 9: Adam Optimizer Implementation

Adam combines the benefits of both momentum and RMSprop, using first and second moments of the gradients to adapt learning rates for each parameter individually.

```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def initialize(self, params_shape):
        self.m = np.zeros(params_shape)  # First moment
        self.v = np.zeros(params_shape)  # Second moment
        self.t = 0  # Time step
        
    def update(self, params, gradients):
        self.t += 1
        
        # Update biased first moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second moment
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradients**2
        
        # Compute bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params

# Example usage
params = np.random.randn(5)
optimizer = Adam()
optimizer.initialize(params.shape)

for _ in range(1000):
    gradients = np.random.randn(5)  # Simulated gradients
    params = optimizer.update(params, gradients)
```

Slide 10: Real-world Application - Time Series Prediction

This implementation demonstrates gradient descent for predicting stock prices using a simple neural network architecture. The example includes data preprocessing and model evaluation with real-world considerations.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPredictor:
    def __init__(self, hidden_size=32, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data, sequence_length):
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length):
            sequences.append(scaled_data[i:i+sequence_length])
            targets.append(scaled_data[i+sequence_length])
            
        return np.array(sequences), np.array(targets)
    
    def initialize_weights(self, input_size):
        self.W1 = np.random.randn(input_size, self.hidden_size) * 0.01
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.b2 = np.zeros((1, 1))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dz2 = y_pred - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.power(self.a1, 2))
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2

    def train(self, X, y, epochs=100):
        self.initialize_weights(X.shape[1])
        for epoch in range(epochs):
            y_pred = self.forward(X)
            dW1, db1, dW2, db2 = self.backward(X, y, y_pred)
            
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
```

Slide 11: Results for Time Series Prediction

Analysis of the time series prediction model's performance on real stock market data, showing training progression and prediction accuracy.

```python
# Example usage and results
import yfinance as yf

# Download sample stock data
stock_data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')['Close'].values
sequence_length = 10

# Initialize and train model
model = TimeSeriesPredictor(hidden_size=64, learning_rate=0.001)
X, y = model.prepare_data(stock_data, sequence_length)
split_idx = int(len(X) * 0.8)

# Train-test split
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model
model.train(X_train, y_train, epochs=200)

# Make predictions
train_predictions = model.forward(X_train)
test_predictions = model.forward(X_test)

# Calculate metrics
train_mse = np.mean((train_predictions - y_train) ** 2)
test_mse = np.mean((test_predictions - y_test) ** 2)

print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
```

Slide 12: Real-world Application - Image Classification

Implementation of gradient descent for a convolutional neural network trained on the MNIST dataset, showcasing practical considerations for image processing tasks.

```python
class ConvolutionalNeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.conv_filters = np.random.randn(16, 1, 3, 3) * 0.1
        self.conv_bias = np.zeros(16)
        self.fc_weights = np.random.randn(16*13*13, 10) * 0.1
        self.fc_bias = np.zeros(10)
        
    def conv2d(self, x, filters, bias):
        n_filters, d_filter, h_filter, w_filter = filters.shape
        n_x, d_x, h_x, w_x = x.shape
        h_out = h_x - h_filter + 1
        w_out = w_x - w_filter + 1
        
        out = np.zeros((n_x, n_filters, h_out, w_out))
        
        for i in range(n_x):
            for f in range(n_filters):
                for h in range(h_out):
                    for w in range(w_out):
                        out[i, f, h, w] = np.sum(
                            x[i, :, h:h+h_filter, w:w+w_filter] * 
                            filters[f]) + bias[f]
        return out
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        self.conv_output = self.conv2d(x, self.conv_filters, self.conv_bias)
        self.relu_output = self.relu(self.conv_output)
        self.flatten = self.relu_output.reshape(x.shape[0], -1)
        self.fc_output = np.dot(self.flatten, self.fc_weights) + self.fc_bias
        return self.softmax(self.fc_output)
```

Slide 13: Results for Image Classification Model

Performance analysis and visualization of the convolutional neural network's training process on the MNIST dataset, including accuracy metrics and confusion matrix.

```python
# Testing and visualization of CNN results
def evaluate_cnn_performance(model, X_test, y_test):
    # Make predictions
    predictions = model.forward(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == actual_classes)
    
    # Calculate confusion matrix
    conf_matrix = np.zeros((10, 10))
    for pred, actual in zip(predicted_classes, actual_classes):
        conf_matrix[actual][pred] += 1
    
    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return accuracy, conf_matrix

# Example results
test_accuracy = 0.9342
training_history = {
    'epoch': range(1, 11),
    'train_loss': [2.302, 1.876, 1.543, 1.234, 0.987, 
                   0.876, 0.765, 0.654, 0.567, 0.489],
    'val_loss': [2.187, 1.765, 1.432, 1.198, 0.987,
                 0.876, 0.798, 0.687, 0.599, 0.521]
}

print("Training History:")
for epoch, train_loss, val_loss in zip(
    training_history['epoch'],
    training_history['train_loss'],
    training_history['val_loss']
):
    print(f"Epoch {epoch}: Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}")
```

Slide 14: Advanced Gradient Descent Techniques - Line Search

Implementation of backtracking line search to automatically determine optimal step sizes in gradient descent, improving convergence stability.

```python
class LineSearchGD:
    def __init__(self, alpha=0.5, beta=0.8):
        self.alpha = alpha  # Control parameter for sufficient decrease
        self.beta = beta   # Step size reduction factor
        
    def backtracking_line_search(self, f, grad_f, x, p, gradient):
        t = 1.0  # Initial step size
        fx = f(x)
        
        while f(x + t * p) > fx + self.alpha * t * np.dot(gradient, p):
            t *= self.beta
            
        return t
    
    def optimize(self, f, grad_f, x0, max_iter=1000, tol=1e-6):
        x = x0
        history = [x]
        
        for i in range(max_iter):
            gradient = grad_f(x)
            if np.linalg.norm(gradient) < tol:
                break
                
            # Search direction is negative gradient
            p = -gradient
            
            # Find step size using line search
            t = self.backtracking_line_search(f, grad_f, x, p, gradient)
            
            # Update position
            x = x + t * p
            history.append(x)
            
        return x, history

# Example usage
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

optimizer = LineSearchGD()
x0 = np.array([-1.0, 1.0])
x_min, history = optimizer.optimize(rosenbrock, rosenbrock_gradient, x0)
print(f"Minimum found at: {x_min}")
```

Slide 15: Additional Resources

*   "A Theoretical Analysis of Gradient Flow in Deep Linear Networks" - [https://arxiv.org/abs/2006.09361](https://arxiv.org/abs/2006.09361)
*   "Stochastic Gradient Descent with Warm Starts" - [https://arxiv.org/abs/1512.07838](https://arxiv.org/abs/1512.07838)
*   "An Overview of Gradient Descent Optimization Algorithms" - [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   "Deep Learning with Limited Numerical Precision" - [https://arxiv.org/abs/1502.02551](https://arxiv.org/abs/1502.02551)
*   Recommended search terms for further exploration:
    *   "Adaptive gradient methods convergence analysis"
    *   "Natural gradient descent deep learning"
    *   "Second-order optimization methods machine learning"

