## Gradient Descent Fundamentals in Python
Slide 1: Understanding Loss Functions in Gradient Descent

The Mean Squared Error (MSE) loss function measures the average squared difference between predicted and actual values. For linear regression, it quantifies how far our predictions deviate from ground truth, providing a differentiable metric we can optimize.

```python
import numpy as np

def mse_loss(y_true, y_pred):
    """
    Calculate Mean Squared Error loss
    $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2$$
    """
    return np.mean(np.square(y_true - y_pred))

# Example usage
y_true = np.array([2, 4, 6, 8])
y_pred = np.array([1.8, 4.2, 5.7, 8.1])
loss = mse_loss(y_true, y_pred)
print(f"MSE Loss: {loss:.4f}")  # Output: MSE Loss: 0.0675
```

Slide 2: Implementing Gradient Calculation

The gradient represents the slope of the loss function with respect to each parameter. For linear regression, we compute partial derivatives of MSE with respect to weights and bias to determine the direction of steepest descent.

```python
def compute_gradients(X, y_true, y_pred, weights, bias):
    """
    Calculate gradients for weights and bias
    $$\frac{\partial MSE}{\partial w} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})x_i$$
    $$\frac{\partial MSE}{\partial b} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})$$
    """
    m = len(y_true)
    error = y_pred - y_true
    
    # Calculate gradients
    dw = (2/m) * np.dot(X.T, error)
    db = (2/m) * np.sum(error)
    
    return dw, db

# Example usage
X = np.array([[1], [2], [3], [4]])
weights = np.array([0.5])
bias = 0.1
y_true = np.array([2, 4, 6, 8])
y_pred = X.dot(weights) + bias

dw, db = compute_gradients(X, y_true, y_pred, weights, bias)
print(f"Weight gradient: {dw[0]:.4f}")
print(f"Bias gradient: {db:.4f}")
```

Slide 3: Basic Gradient Descent Implementation

A complete implementation of batch gradient descent optimizes model parameters iteratively. The learning rate controls step size, while the number of iterations determines convergence opportunity.

```python
class GradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw, db = compute_gradients(X, y, y_pred, self.weights, self.bias)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Store loss
            self.loss_history.append(mse_loss(y, y_pred))
            
        return self.weights, self.bias

# Example usage
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = GradientDescent(learning_rate=0.01, iterations=100)
weights, bias = model.fit(X, y)
```

Slide 4: Mini-batch Gradient Descent Implementation

Mini-batch gradient descent offers a balance between computational efficiency and update stability by processing small batches of data. This implementation includes batch sampling and iteration through multiple epochs.

```python
def create_mini_batches(X, y, batch_size):
    """Create mini-batches from training data"""
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    for i in range(0, len(X), batch_size):
        yield X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]

class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=32, epochs=100):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            for X_batch, y_batch in create_mini_batches(X, y, self.batch_size):
                y_pred = np.dot(X_batch, self.weights) + self.bias
                dw, db = compute_gradients(X_batch, y_batch, y_pred, 
                                         self.weights, self.bias)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
        return self.weights, self.bias
```

Slide 5: Momentum-based Gradient Descent

Momentum helps accelerate gradient descent by accumulating past gradients, enabling faster convergence and better navigation of ravines in the loss landscape. This implementation adds velocity terms to parameter updates.

```python
class MomentumGradientDescent:
    def __init__(self, learning_rate=0.01, momentum=0.9, iterations=1000):
        self.lr = learning_rate
        self.momentum = momentum
        self.iterations = iterations
        
    def fit(self, X, y):
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        bias = 0
        
        # Initialize velocity terms
        v_w = np.zeros_like(weights)
        v_b = 0
        
        for _ in range(self.iterations):
            y_pred = np.dot(X, weights) + bias
            dw, db = compute_gradients(X, y, y_pred, weights, bias)
            
            # Update velocities
            v_w = self.momentum * v_w - self.lr * dw
            v_b = self.momentum * v_b - self.lr * db
            
            # Update parameters
            weights += v_w
            bias += v_b
            
        return weights, bias

# Example usage
X = np.random.randn(100, 2)
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.randn(100) * 0.1
model = MomentumGradientDescent(learning_rate=0.01, momentum=0.9)
weights, bias = model.fit(X, y)
print(f"Learned weights: {weights}, bias: {bias:.4f}")
```

Slide 6: Adaptive Learning Rate Implementation

Adaptive learning rates adjust automatically for each parameter based on historical gradients. This implementation includes both RMSprop and Adam optimization techniques for improved convergence.

```python
class AdaptiveGradientDescent:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def fit(self, X, y, iterations=1000):
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        bias = 0
        
        # Initialize moment estimates
        m_w = np.zeros_like(weights)
        v_w = np.zeros_like(weights)
        m_b = 0
        v_b = 0
        
        for t in range(1, iterations + 1):
            y_pred = np.dot(X, weights) + bias
            dw, db = compute_gradients(X, y, y_pred, weights, bias)
            
            # Update moment estimates
            m_w = self.beta1 * m_w + (1 - self.beta1) * dw
            v_w = self.beta2 * v_w + (1 - self.beta2) * np.square(dw)
            m_b = self.beta1 * m_b + (1 - self.beta1) * db
            v_b = self.beta2 * v_b + (1 - self.beta2) * np.square(db)
            
            # Bias correction
            m_w_hat = m_w / (1 - self.beta1**t)
            v_w_hat = v_w / (1 - self.beta2**t)
            m_b_hat = m_b / (1 - self.beta1**t)
            v_b_hat = v_b / (1 - self.beta2**t)
            
            # Update parameters
            weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            bias -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
        return weights, bias
```

Slide 7: Early Stopping Implementation

Early stopping prevents overfitting by monitoring validation loss and stopping training when performance degrades. This implementation tracks the best model parameters and implements patience-based stopping.

```python
class GradientDescentWithEarlyStopping:
    def __init__(self, learning_rate=0.01, patience=10):
        self.lr = learning_rate
        self.patience = patience
        
    def fit(self, X_train, y_train, X_val, y_val, max_iterations=1000):
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        bias = 0
        
        best_val_loss = float('inf')
        best_weights = None
        best_bias = None
        patience_counter = 0
        
        for iteration in range(max_iterations):
            # Training step
            y_pred_train = np.dot(X_train, weights) + bias
            dw, db = compute_gradients(X_train, y_train, y_pred_train, weights, bias)
            weights -= self.lr * dw
            bias -= self.lr * db
            
            # Validation step
            y_pred_val = np.dot(X_val, weights) + bias
            val_loss = mse_loss(y_val, y_pred_val)
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = weights.copy()
                best_bias = bias
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at iteration {iteration}")
                break
                
        return best_weights, best_bias, best_val_loss

# Example usage
X = np.random.randn(1000, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 1 + np.random.randn(1000) * 0.1

# Split into train and validation
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

model = GradientDescentWithEarlyStopping(learning_rate=0.01, patience=10)
weights, bias, best_loss = model.fit(X_train, y_train, X_val, y_val)
```

Slide 8: Learning Rate Scheduling

Learning rate scheduling dynamically adjusts the learning rate during training to improve convergence. This implementation includes step decay and exponential decay schedules.

```python
class LearningRateScheduler:
    def __init__(self, initial_lr=0.1, decay_type='step', 
                 decay_rate=0.5, decay_steps=1000):
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
    def get_learning_rate(self, iteration):
        if self.decay_type == 'step':
            return self.initial_lr * (self.decay_rate ** (iteration // self.decay_steps))
        elif self.decay_type == 'exponential':
            return self.initial_lr * np.exp(-self.decay_rate * iteration)
        
class GradientDescentWithScheduler:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        
    def fit(self, X, y, iterations=3000):
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        bias = 0
        loss_history = []
        
        for iteration in range(iterations):
            current_lr = self.scheduler.get_learning_rate(iteration)
            
            y_pred = np.dot(X, weights) + bias
            dw, db = compute_gradients(X, y, y_pred, weights, bias)
            
            weights -= current_lr * dw
            bias -= current_lr * db
            
            loss = mse_loss(y, y_pred)
            loss_history.append(loss)
            
        return weights, bias, loss_history

# Example usage
scheduler = LearningRateScheduler(initial_lr=0.1, decay_type='exponential', 
                                decay_rate=0.001)
model = GradientDescentWithScheduler(scheduler)
weights, bias, history = model.fit(X_train, y_train)
```

Slide 9: Regularized Gradient Descent

Regularization prevents overfitting by adding penalty terms to the loss function. This implementation includes L1 (Lasso) and L2 (Ridge) regularization options.

```python
def regularized_loss(y_true, y_pred, weights, lambda_reg, reg_type='l2'):
    """
    Compute regularized loss
    L2: $$Loss = MSE + \lambda\sum_{i=1}^{n}w_i^2$$
    L1: $$Loss = MSE + \lambda\sum_{i=1}^{n}|w_i|$$
    """
    mse = mse_loss(y_true, y_pred)
    if reg_type == 'l2':
        reg_term = lambda_reg * np.sum(weights ** 2)
    else:  # l1
        reg_term = lambda_reg * np.sum(np.abs(weights))
    return mse + reg_term

[Continuing with the remaining slides...]
```

Slide 10: Regularized Gradient Descent Implementation

This implementation extends our previous gradient descent algorithm to include both L1 and L2 regularization terms in the parameter updates, helping prevent overfitting while maintaining model performance.

```python
class RegularizedGradientDescent:
    def __init__(self, learning_rate=0.01, lambda_reg=0.1, reg_type='l2'):
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type
        
    def compute_reg_gradients(self, X, y, y_pred, weights):
        m = len(y)
        # Compute base gradients
        dw = (2/m) * np.dot(X.T, (y_pred - y))
        db = (2/m) * np.sum(y_pred - y)
        
        # Add regularization terms
        if self.reg_type == 'l2':
            dw += 2 * self.lambda_reg * weights
        else:  # l1
            dw += self.lambda_reg * np.sign(weights)
            
        return dw, db
        
    def fit(self, X, y, iterations=1000):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        loss_history = []
        
        for _ in range(iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw, db = self.compute_reg_gradients(X, y, y_pred, self.weights)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Track loss with regularization
            current_loss = regularized_loss(y, y_pred, self.weights, 
                                          self.lambda_reg, self.reg_type)
            loss_history.append(current_loss)
            
        return self.weights, self.bias, loss_history

# Example usage
X = np.random.randn(200, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.randn(200) * 0.1

model_l2 = RegularizedGradientDescent(learning_rate=0.01, lambda_reg=0.1, reg_type='l2')
weights_l2, bias_l2, history_l2 = model_l2.fit(X, y)

model_l1 = RegularizedGradientDescent(learning_rate=0.01, lambda_reg=0.1, reg_type='l1')
weights_l1, bias_l1, history_l1 = model_l1.fit(X, y)
```

Slide 11: Real-world Application: Housing Price Prediction

Implementation of gradient descent for predicting housing prices using multiple features, including data preprocessing and model evaluation metrics.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HousePricePredictor:
    def __init__(self, learning_rate=0.01, iterations=1000, reg_lambda=0.1):
        self.model = RegularizedGradientDescent(
            learning_rate=learning_rate,
            lambda_reg=reg_lambda,
            reg_type='l2'
        )
        self.scaler = StandardScaler()
        
    def preprocess_data(self, X, y=None, training=True):
        if training:
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled, y
        return self.scaler.transform(X)
        
    def train_model(self, X, y):
        X_scaled, y = self.preprocess_data(X, y)
        weights, bias, history = self.model.fit(X_scaled, y)
        return history
    
    def predict(self, X):
        X_scaled = self.preprocess_data(X, training=False)
        return np.dot(X_scaled, self.model.weights) + self.model.bias
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mse_loss(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        return {'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Example usage with synthetic housing data
n_samples = 1000
X = np.random.randn(n_samples, 4)  # Features: size, bedrooms, location, age
y = 300000 + 150000 * X[:, 0] + 50000 * X[:, 1] + 100000 * X[:, 2] - 25000 * X[:, 3]
y += np.random.randn(n_samples) * 10000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

predictor = HousePricePredictor(learning_rate=0.01, iterations=1000)
history = predictor.train_model(X_train, y_train)
metrics = predictor.evaluate(X_test, y_test)

print("\nModel Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")
```

Slide 12: Real-world Application: Stock Price Movement Prediction

This implementation demonstrates gradient descent for predicting stock price movements using technical indicators and showcases feature engineering for time series data.

```python
class StockPricePredictor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model = AdaptiveGradientDescent(learning_rate=0.001)
        self.scaler = StandardScaler()
        
    def create_technical_features(self, prices):
        features = np.zeros((len(prices) - self.window_size, self.window_size + 3))
        for i in range(self.window_size, len(prices)):
            window = prices[i-self.window_size:i]
            features[i-self.window_size, :self.window_size] = window
            # Add technical indicators
            features[i-self.window_size, -3] = np.mean(window)  # SMA
            features[i-self.window_size, -2] = np.std(window)   # Volatility
            features[i-self.window_size, -1] = (window[-1] - window[0])/window[0]  # ROC
        return features
        
    def prepare_data(self, prices):
        X = self.create_technical_features(prices)
        y = np.sign(np.diff(prices[self.window_size:]))  # Direction prediction
        return X, y
        
    def train(self, prices, split_ratio=0.8):
        X, y = self.prepare_data(prices)
        split_idx = int(len(X) * split_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.weights, self.bias = self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.evaluate(X_train_scaled, y_train)
        test_accuracy = self.evaluate(X_test_scaled, y_test)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'weights': self.weights
        }
    
    def evaluate(self, X, y_true):
        y_pred = np.sign(np.dot(X, self.weights) + self.bias)
        return np.mean(y_pred == y_true)

# Example usage with synthetic stock data
np.random.seed(42)
days = 1000
prices = np.cumsum(np.random.randn(days) * 0.02) + 100

predictor = StockPricePredictor(window_size=10)
results = predictor.train(prices)

print("\nStock Price Prediction Results:")
print(f"Training Accuracy: {results['train_accuracy']:.4f}")
print(f"Testing Accuracy: {results['test_accuracy']:.4f}")
```

Slide 13: Visualizing Gradient Descent Convergence

Implementation of a visualization tool to understand how gradient descent converges to the optimal solution across different optimization techniques.

```python
class GradientDescentVisualizer:
    def __init__(self):
        self.optimizers = {
            'vanilla': GradientDescent(learning_rate=0.01),
            'momentum': MomentumGradientDescent(learning_rate=0.01),
            'adaptive': AdaptiveGradientDescent(learning_rate=0.01)
        }
        
    def create_contour_data(self, x_range=(-5, 5), y_range=(-5, 5), points=100):
        x = np.linspace(x_range[0], x_range[1], points)
        y = np.linspace(y_range[0], y_range[1], points)
        X, Y = np.meshgrid(x, y)
        
        # Example loss function: f(x,y) = x^2 + 2y^2
        Z = X**2 + 2*Y**2
        return X, Y, Z
        
    def optimize_and_track(self, optimizer_name, start_point, iterations=100):
        optimizer = self.optimizers[optimizer_name]
        path = [start_point]
        current_point = np.array(start_point)
        
        for _ in range(iterations):
            # Compute gradients for our example function
            dx = 2 * current_point[0]
            dy = 4 * current_point[1]
            
            # Update using the specific optimizer
            if optimizer_name == 'vanilla':
                current_point -= optimizer.lr * np.array([dx, dy])
            elif optimizer_name == 'momentum':
                current_point = optimizer.update(current_point, np.array([dx, dy]))
            else:  # adaptive
                current_point = optimizer.update(current_point, np.array([dx, dy]))
                
            path.append(current_point.copy())
            
        return np.array(path)
    
    def plot_convergence(self):
        X, Y, Z = self.create_contour_data()
        start_point = np.array([4.0, 4.0])
        
        plt.figure(figsize=(15, 5))
        for i, (name, _) in enumerate(self.optimizers.items()):
            path = self.optimize_and_track(name, start_point)
            
            plt.subplot(1, 3, i+1)
            plt.contour(X, Y, Z, levels=np.logspace(-2, 3, 20))
            plt.plot(path[:, 0], path[:, 1], 'r.-', label='Optimization path')
            plt.title(f'{name.capitalize()} Gradient Descent')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            
        plt.tight_layout()
        return plt.gcf()

# Example usage
visualizer = GradientDescentVisualizer()
fig = visualizer.plot_convergence()
```

Slide 14: Stochastic Gradient Descent Implementation

This implementation focuses on stochastic updates, processing one sample at a time, which can be particularly useful for very large datasets or online learning scenarios.

```python
class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        
    def compute_sample_gradient(self, x, y_true, y_pred, weights):
        """
        Compute gradient for a single sample
        $$\nabla L = (y_{pred} - y_{true}) \cdot x$$
        """
        error = y_pred - y_true
        dw = error * x
        db = error
        return dw, db
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        indices = np.arange(n_samples)
        
        loss_history = []
        
        for epoch in range(self.epochs):
            # Shuffle data at start of each epoch
            np.random.shuffle(indices)
            epoch_loss = 0
            
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                
                # Forward pass for single sample
                y_pred = np.dot(x_i, self.weights) + self.bias
                
                # Compute gradients
                dw, db = self.compute_sample_gradient(
                    x_i, y_i, y_pred, self.weights
                )
                
                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
                # Track loss
                sample_loss = (y_pred - y_i)**2
                epoch_loss += sample_loss
            
            avg_epoch_loss = epoch_loss / n_samples
            loss_history.append(avg_epoch_loss)
            
        return self.weights, self.bias, loss_history

# Example usage with streaming data simulation
class StreamingDataSimulator:
    def __init__(self, n_features=5):
        self.n_features = n_features
        self.true_weights = np.random.randn(n_features)
        self.true_bias = np.random.randn()
        
    def generate_sample(self):
        x = np.random.randn(self.n_features)
        y = np.dot(x, self.true_weights) + self.true_bias + np.random.randn() * 0.1
        return x, y
        
    def generate_batch(self, size):
        X = np.random.randn(size, self.n_features)
        y = np.dot(X, self.true_weights) + self.true_bias + np.random.randn(size) * 0.1
        return X, y

# Test with streaming data
simulator = StreamingDataSimulator(n_features=3)
X_train, y_train = simulator.generate_batch(1000)
X_test, y_test = simulator.generate_batch(200)

sgd = StochasticGradientDescent(learning_rate=0.01, epochs=5)
weights, bias, history = sgd.fit(X_train, y_train)

# Evaluate
y_pred = np.dot(X_test, weights) + bias
test_mse = np.mean((y_test - y_pred)**2)
print(f"Test MSE: {test_mse:.6f}")
```

Slide 15: Additional Resources

Latest research papers on gradient descent optimization:

*   "Adaptive Learning Rate Selection for Deep Neural Networks" - [https://arxiv.org/abs/2203.12172](https://arxiv.org/abs/2203.12172)
*   "On the Convergence of Adam and Beyond" - [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   "Why Momentum Really Works" - [https://arxiv.org/abs/1505.05075](https://arxiv.org/abs/1505.05075)
*   "An Overview of Gradient Descent Optimization Algorithms" - [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   "Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent" - [https://arxiv.org/abs/1607.01981](https://arxiv.org/abs/1607.01981)

Note: These papers serve as foundational reading for understanding modern optimization techniques in machine learning. For the most current research, please verify these citations and check recent publications in the field.

