## Understanding Gradient Descent for AI Model Training
Slide 1: Introduction to Gradient Descent

Gradient descent is an iterative optimization algorithm used to minimize a loss function by iteratively adjusting parameters in the direction of steepest descent. The algorithm calculates partial derivatives to determine which direction leads to the minimum of the function.

```python
import numpy as np

def gradient_descent(f, df, x0, learning_rate=0.01, epochs=1000, tolerance=1e-6):
    x = x0  # Initial guess
    history = [x]
    
    for _ in range(epochs):
        gradient = df(x)
        x_new = x - learning_rate * gradient
        
        if np.abs(f(x_new) - f(x)) < tolerance:
            break
            
        x = x_new
        history.append(x)
    
    return x, history

# Example: Finding minimum of f(x) = x^2
f = lambda x: x**2
df = lambda x: 2*x

min_x, history = gradient_descent(f, df, x0=2.0)
print(f"Minimum found at x = {min_x:.6f}")
```

Slide 2: Implementing Linear Regression with Gradient Descent

Linear regression serves as a fundamental example of gradient descent application. We implement a simple linear regression model from scratch using numpy, demonstrating how partial derivatives guide the optimization process.

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, epochs=1000):
        n_samples = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Generate sample data
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Train model
model = LinearRegression()
model.fit(X, y)
```

Slide 3: Mathematical Foundations of Derivatives in Neural Networks

Understanding the chain rule is crucial for implementing backpropagation in neural networks. The mathematics behind gradient computation forms the basis for training deep learning models effectively.

```python
# Mathematical representation of chain rule in LaTeX
"""
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

$$
\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}}
$$
"""

def backward_pass_example(X, y, weights, activations):
    # Simplified backward pass implementation
    m = X.shape[0]
    dZ = activations - y
    dW = (1/m) * np.dot(X.T, dZ)
    db = (1/m) * np.sum(dZ, axis=0)
    return dW, db
```

Slide 4: Implementing a Neural Network Layer

This implementation demonstrates how derivatives enable backpropagation through a single neural network layer, showing the practical application of chain rule in deep learning computations.

```python
class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.bias
        return self.output

    def backward(self, upstream_gradient, learning_rate=0.01):
        dW = np.dot(self.input.T, upstream_gradient)
        db = np.sum(upstream_gradient, axis=0, keepdims=True)
        dinput = np.dot(upstream_gradient, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db
        return dinput

# Example usage
layer = NeuralLayer(10, 5)
X = np.random.randn(32, 10)  # Batch of 32 samples
output = layer.forward(X)
```

Slide 5: Stochastic Gradient Descent Implementation

Stochastic Gradient Descent (SGD) processes training data in small batches, making it more memory efficient and often leading to faster convergence than batch gradient descent. This implementation shows practical batch processing.

```python
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    
    for i in range(n_minibatches):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, y_mini))
        
    return mini_batches

class SGDOptimizer:
    def __init__(self, learning_rate=0.01, batch_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
    def optimize(self, model, X, y, epochs=100):
        losses = []
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X, y, self.batch_size)
            epoch_loss = 0
            
            for X_batch, y_batch in mini_batches:
                # Forward pass
                y_pred = model.forward(X_batch)
                loss = np.mean((y_pred - y_batch) ** 2)
                
                # Backward pass
                grad = 2 * (y_pred - y_batch) / self.batch_size
                model.backward(grad, self.learning_rate)
                
                epoch_loss += loss
                
            losses.append(epoch_loss / len(mini_batches))
        return losses
```

Slide 6: Advanced Gradient Optimization Techniques

Modern deep learning relies on sophisticated optimizers that extend basic gradient descent. This implementation showcases Adam optimizer, which combines momentum and adaptive learning rates for better convergence.

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def initialize(self, params_shape):
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        
    def update(self, params, grads):
        if self.m is None:
            self.initialize(params.shape)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grads)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params
```

Slide 7: Integration in Machine Learning: Probability Distributions

Integration plays a crucial role in calculating probabilities and expectations in machine learning. This implementation demonstrates numerical integration techniques for probability density functions.

```python
import scipy.stats as stats
import scipy.integrate as integrate

def probability_in_range(mu, sigma, a, b):
    """Calculate probability within range [a,b] for normal distribution"""
    # Integration of normal PDF
    def normal_pdf(x):
        return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    
    prob, error = integrate.quad(normal_pdf, a, b)
    return prob

def expected_value(func, distribution, lower, upper):
    """Calculate expected value of a function under a probability distribution"""
    def integrand(x):
        return func(x) * distribution.pdf(x)
    
    expectation, error = integrate.quad(integrand, lower, upper)
    return expectation

# Example usage
mu, sigma = 0, 1
normal_dist = stats.norm(mu, sigma)

# Calculate probability in range [-1, 1]
prob = probability_in_range(mu, sigma, -1, 1)
print(f"Probability in range [-1, 1]: {prob:.4f}")

# Calculate expected value of x^2 under standard normal
squared = lambda x: x**2
ev = expected_value(squared, normal_dist, -5, 5)
print(f"E[X^2] for standard normal: {ev:.4f}")
```

Slide 8: Implementing Backpropagation from Scratch

Backpropagation is the cornerstone of neural network training, using chain rule to compute gradients through multiple layers. This implementation shows the complete forward and backward passes through a neural network.

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            layer = {
                'W': np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01,
                'b': np.zeros((1, layer_sizes[i+1])),
                'activations': None,
                'Z': None
            }
            self.layers.append(layer)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def forward_propagation(self, X):
        current_input = X
        for layer in self.layers:
            Z = np.dot(current_input, layer['W']) + layer['b']
            A = self.sigmoid(Z)
            layer['Z'] = Z
            layer['activations'] = A
            current_input = A
        return current_input
    
    def backward_propagation(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        current_gradient = (self.layers[-1]['activations'] - y)
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == 0:
                previous_activations = X
            else:
                previous_activations = self.layers[i-1]['activations']
            
            dW = np.dot(previous_activations.T, current_gradient) / m
            db = np.sum(current_gradient, axis=0, keepdims=True) / m
            
            if i > 0:
                current_gradient = np.dot(current_gradient, layer['W'].T) * \
                                 self.sigmoid_derivative(self.layers[i-1]['Z'])
            
            layer['W'] -= learning_rate * dW
            layer['b'] -= learning_rate * db

# Example usage
X = np.random.randn(100, 3)
y = (np.sum(X, axis=1) > 0).astype(float).reshape(-1, 1)
nn = NeuralNetwork([3, 4, 1])
```

Slide 9: Real-world Application: House Price Prediction

This implementation demonstrates a complete machine learning pipeline for house price prediction, including data preprocessing, model training, and evaluation using gradient descent optimization.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HousePricePredictor:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X, y=None, training=True):
        if training:
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, y
        return self.scaler.transform(X)
    
    def train(self, X, y):
        X_scaled, y = self.preprocess_data(X, y)
        n_features = X_scaled.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            # Forward pass
            predictions = np.dot(X_scaled, self.weights) + self.bias
            
            # Compute gradients
            dw = (2/len(X)) * np.dot(X_scaled.T, (predictions - y))
            db = (2/len(X)) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        X_scaled = self.preprocess_data(X, training=False)
        return np.dot(X_scaled, self.weights) + self.bias
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y - predictions) ** 2) / 
                  np.sum((y - np.mean(y)) ** 2))
        return {'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Example usage with synthetic data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(1000) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = HousePricePredictor(learning_rate=0.01, epochs=1000)
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
print("Model Performance:", metrics)
```

Slide 10: Netflix-Style Recommendation System Implementation

A practical implementation of a recommendation system using gradient descent to optimize user and item embeddings, similar to how streaming services suggest content to users.

```python
class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=20, learning_rate=0.01, regularization=0.02):
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, n_factors))
        self.lr = learning_rate
        self.reg = regularization
        
    def predict(self, user_id, item_id):
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])
    
    def train(self, ratings, epochs=100):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            np.random.shuffle(ratings)
            
            for user_id, item_id, rating in ratings:
                # Compute prediction and error
                prediction = self.predict(user_id, item_id)
                error = rating - prediction
                
                # Store old factors for regularization
                user_vec = self.user_factors[user_id].copy()
                item_vec = self.item_factors[item_id].copy()
                
                # Update factors using gradient descent
                self.user_factors[user_id] += self.lr * (error * item_vec - self.reg * user_vec)
                self.item_factors[item_id] += self.lr * (error * user_vec - self.reg * item_vec)
                
                # Accumulate squared error
                epoch_loss += error ** 2
                
            losses.append(np.sqrt(epoch_loss / len(ratings)))
            
        return losses

# Example usage with synthetic rating data
n_users, n_items = 1000, 500
n_ratings = 10000
ratings_data = [
    (np.random.randint(n_users),
     np.random.randint(n_items),
     np.random.randint(1, 6))
    for _ in range(n_ratings)
]

model = MatrixFactorization(n_users, n_items)
training_losses = model.train(ratings_data)
```

Slide 11: Numerical Integration in Deep Learning: Monte Carlo Methods

Monte Carlo integration is crucial for estimating complex integrals in machine learning, particularly for calculating expectations and sampling from complex distributions.

```python
class MonteCarloIntegration:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
    
    def importance_sampling(self, target_dist, proposal_dist):
        # Generate samples from proposal distribution
        samples = proposal_dist.rvs(self.n_samples)
        
        # Calculate importance weights
        weights = target_dist.pdf(samples) / proposal_dist.pdf(samples)
        
        # Estimate expectation
        expectation = np.mean(weights)
        variance = np.var(weights)
        
        return {
            'expectation': expectation,
            'variance': variance,
            'std_error': np.sqrt(variance / self.n_samples)
        }
    
    def mcmc_metropolis(self, target_pdf, proposal_std, n_burnin=1000):
        samples = np.zeros(self.n_samples)
        current = np.random.randn()  # Initial state
        
        # Burn-in period
        for _ in range(n_burnin):
            proposal = current + np.random.normal(0, proposal_std)
            acceptance_ratio = target_pdf(proposal) / target_pdf(current)
            
            if np.random.rand() < acceptance_ratio:
                current = proposal
        
        # Sampling period
        for i in range(self.n_samples):
            proposal = current + np.random.normal(0, proposal_std)
            acceptance_ratio = target_pdf(proposal) / target_pdf(current)
            
            if np.random.rand() < acceptance_ratio:
                current = proposal
            
            samples[i] = current
            
        return samples

# Example usage
def target_pdf(x):
    return stats.norm.pdf(x, loc=2, scale=1.5)

mc = MonteCarloIntegration()
proposal_dist = stats.norm(loc=0, scale=2)
target_dist = stats.norm(loc=2, scale=1.5)

results = mc.importance_sampling(target_dist, proposal_dist)
samples = mc.mcmc_metropolis(target_pdf, proposal_std=0.5)
```

Slide 12: Advanced Optimization: Natural Gradient Descent

Natural gradient descent considers the geometry of the parameter space, leading to more efficient optimization in deep learning models by using the Fisher Information Matrix.

```python
class NaturalGradientOptimizer:
    def __init__(self, learning_rate=0.1, damping=1e-4):
        self.lr = learning_rate
        self.damping = damping
        
    def compute_fisher_matrix(self, model, X, batch_size=32):
        gradients = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            output = model.forward(batch)
            grad = model.compute_gradients(output)
            gradients.append(grad.reshape(1, -1))
            
        gradients = np.vstack(gradients)
        fisher = np.dot(gradients.T, gradients) / len(X)
        return fisher + np.eye(fisher.shape[0]) * self.damping
    
    def update_parameters(self, model, X, y):
        fisher = self.compute_fisher_matrix(model, X)
        gradients = model.compute_gradients(model.forward(X), y)
        
        # Solve Fisher * update = gradient
        natural_gradient = np.linalg.solve(fisher, gradients)
        
        # Update parameters
        model.parameters -= self.lr * natural_gradient.reshape(model.parameters.shape)

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.parameters = np.random.randn(input_size, hidden_size) * 0.01
        
    def forward(self, X):
        return np.tanh(np.dot(X, self.parameters))
    
    def compute_gradients(self, output, targets=None):
        if targets is None:
            return output
        return (output - targets).flatten()

# Example usage
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 5)

model = SimpleNeuralNet(10, 5, 1)
optimizer = NaturalGradientOptimizer()
optimizer.update_parameters(model, X, y)
```

Slide 13: Information Geometry in Machine Learning

Information geometry connects differential geometry with probability theory, providing insights into the structure of statistical manifolds and optimization landscapes in machine learning.

```python
class InformationGeometry:
    def __init__(self, distribution_family='gaussian'):
        self.family = distribution_family
        
    def fisher_metric(self, theta):
        """Compute Fisher metric for Gaussian distribution"""
        if self.family == 'gaussian':
            mu, sigma = theta
            # Fisher metric components for Gaussian
            g_mu_mu = 1 / (sigma**2)
            g_mu_sigma = 0
            g_sigma_sigma = 2 / (sigma**2)
            
            return np.array([[g_mu_mu, g_mu_sigma],
                           [g_mu_sigma, g_sigma_sigma]])
    
    def geodesic_distance(self, theta1, theta2, steps=100):
        """Compute approximate geodesic distance between distributions"""
        path = np.linspace(theta1, theta2, steps)
        distance = 0
        
        for i in range(steps-1):
            mid_point = (path[i] + path[i+1]) / 2
            metric = self.fisher_metric(mid_point)
            delta = path[i+1] - path[i]
            
            # Integrate along path
            distance += np.sqrt(np.dot(delta, np.dot(metric, delta)))
            
        return distance
    
    def parallel_transport(self, theta, vector, direction, epsilon=1e-5):
        """Parallel transport a vector along a geodesic"""
        metric = self.fisher_metric(theta)
        christoffel = self.compute_christoffel_symbols(theta)
        
        # Parallel transport equation
        derivative = -np.einsum('ijk,j,k', christoffel, direction, vector)
        
        return vector + epsilon * derivative
    
    def compute_christoffel_symbols(self, theta):
        """Compute Christoffel symbols of the Fisher metric"""
        h = 1e-7
        dim = len(theta)
        symbols = np.zeros((dim, dim, dim))
        
        for i in range(dim):
            theta_plus = theta.copy()
            theta_plus[i] += h
            theta_minus = theta.copy()
            theta_minus[i] -= h
            
            metric_plus = self.fisher_metric(theta_plus)
            metric_minus = self.fisher_metric(theta_minus)
            
            # Finite difference approximation
            symbols[:, :, i] = (metric_plus - metric_minus) / (2 * h)
            
        return symbols

# Example usage
geom = InformationGeometry()
theta1 = np.array([0, 1])  # (mu, sigma)
theta2 = np.array([1, 2])
distance = geom.geodesic_distance(theta1, theta2)
```

Slide 14: Practical Implementation of Modern Optimizer: Adam with Weight Decay

This implementation showcases the AdamW optimizer, which combines Adam's adaptive learning rates with proper weight decay regularization, commonly used in state-of-the-art models.

```python
class AdamW:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.moments = {}
        
    def init_param(self, param_id, param_shape):
        self.moments[param_id] = {
            'm': np.zeros(param_shape),  # First moment
            'v': np.zeros(param_shape)   # Second moment
        }
    
    def update(self, param_id, param, grad):
        if param_id not in self.moments:
            self.init_param(param_id, param.shape)
            
        self.t += 1
        m = self.moments[param_id]['m']
        v = self.moments[param_id]['v']
        
        # Apply weight decay
        grad = grad + self.weight_decay * param
        
        # Update biased first moment
        m = self.beta1 * m + (1 - self.beta1) * grad
        
        # Update biased second moment
        v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
        
        # Compute bias-corrected moments
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Store updated moments
        self.moments[param_id]['m'] = m
        self.moments[param_id]['v'] = v
        
        return param

# Example usage with neural network
class Layer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input, grad_weights, grad_biases

# Training example
X = np.random.randn(1000, 20)
y = np.random.randn(1000, 5)

layer = Layer(20, 5)
optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)

for epoch in range(100):
    # Forward pass
    output = layer.forward(X)
    loss = np.mean((output - y) ** 2)
    
    # Backward pass
    grad_output = 2 * (output - y) / len(X)
    grad_input, grad_weights, grad_biases = layer.backward(grad_output)
    
    # Update parameters
    layer.weights = optimizer.update('weights', layer.weights, grad_weights)
    layer.biases = optimizer.update('biases', layer.biases, grad_biases)
```

Slide 15: Additional Resources

*   Latest advancements in gradient-based optimization:
    *   [https://arxiv.org/abs/2103.00498](https://arxiv.org/abs/2103.00498) - "A Comprehensive Survey of Gradient-based Optimization"
    *   [https://arxiv.org/abs/2002.04839](https://arxiv.org/abs/2002.04839) - "Adaptive Gradient Methods with Dynamic Bound of Learning Rate"
    *   [https://arxiv.org/abs/2006.14372](https://arxiv.org/abs/2006.14372) - "On the Convergence of Adam and Beyond"
*   For practical implementations and tutorials:
    *   [https://www.deeplearning.ai](https://www.deeplearning.ai)
    *   [https://www.fast.ai](https://www.fast.ai)
    *   [https://pytorch.org/tutorials](https://pytorch.org/tutorials)
*   Recommended textbooks for deep learning optimization:
    *   "Deep Learning" by Goodfellow, Bengio, and Courville
    *   "Optimization Methods for Large-Scale Machine Learning" by Bottou et al.
    *   "Mathematics for Machine Learning" by Deisenroth, Faisal, and Ong

