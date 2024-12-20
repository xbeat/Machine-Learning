## Gradient Descent Fundamentals for Machine Learning
Slide 1: Understanding Gradient Descent Fundamentals

Gradient Descent is an iterative optimization algorithm used to minimize a cost function by updating parameters in the opposite direction of the gradient. It forms the backbone of neural network training by finding local minima through calculating partial derivatives and adjusting weights accordingly.

```python
import numpy as np

def gradient_descent(x, y, learning_rate=0.01, epochs=100):
    # Initialize parameters
    m = b = 0
    n = len(x)
    
    for _ in range(epochs):
        # Calculate predictions
        y_pred = m * x + b
        
        # Calculate gradients
        dm = (-2/n) * sum(x * (y - y_pred))
        db = (-2/n) * sum(y - y_pred)
        
        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db
        
    return m, b

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
m, b = gradient_descent(X, y)
print(f"Final parameters: m={m:.4f}, b={b:.4f}")
```

Slide 2: Mathematical Foundation of Gradient Descent

The core principle of gradient descent relies on computing partial derivatives of the cost function with respect to each parameter. The cost function typically measures the difference between predicted and actual values, commonly using Mean Squared Error (MSE).

```python
def cost_function(X, y, theta):
    """
    Compute Mean Squared Error cost function
    X: input features
    y: target values
    theta: parameters
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost
```

Slide 3: Learning Rate and Convergence

The learning rate controls how large steps we take during optimization. Too large values can cause overshooting, while too small values lead to slow convergence. Adaptive learning rates can help balance this trade-off for more efficient training.

```python
def adaptive_gradient_descent(X, y, theta, alpha=0.01, epsilon=1e-8):
    costs = []
    m = len(y)
    prev_cost = float('inf')
    
    while True:
        # Compute predictions and error
        predictions = X.dot(theta)
        error = predictions - y
        
        # Compute gradients
        gradients = (1/m) * X.T.dot(error)
        
        # Update parameters
        theta = theta - alpha * gradients
        
        # Calculate current cost
        current_cost = cost_function(X, y, theta)
        costs.append(current_cost)
        
        # Check convergence
        if abs(prev_cost - current_cost) < epsilon:
            break
            
        prev_cost = current_cost
    
    return theta, costs
```

Slide 4: Batch vs. Stochastic Gradient Descent

In real-world applications, we can choose between processing all data points (batch), one point at a time (stochastic), or mini-batches. Each approach offers different trade-offs between computation speed and convergence stability.

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            predictions = X_batch.dot(theta)
            error = predictions - y_batch
            gradients = X_batch.T.dot(error) / len(X_batch)
            
            theta -= learning_rate * gradients
            
    return theta
```

Slide 5: Momentum-Based Gradient Descent

Momentum helps accelerate gradient descent by accumulating past gradients, enabling faster convergence and better navigation of ravines in the loss landscape while reducing oscillations during optimization.

```python
def momentum_gradient_descent(X, y, learning_rate=0.01, momentum=0.9, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    velocity = np.zeros(n)
    
    for _ in range(epochs):
        predictions = X.dot(theta)
        error = predictions - y
        gradients = X.T.dot(error) / m
        
        # Update velocity and parameters
        velocity = momentum * velocity - learning_rate * gradients
        theta += velocity
        
    return theta
```

Slide 6: Advanced Optimization with Adam

Adam combines the benefits of momentum and RMSprop, maintaining adaptive learning rates for each parameter. It efficiently handles sparse gradients and non-stationary objectives, making it particularly effective for deep learning applications.

```python
def adam_optimizer(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    
    # Initialize moment vectors
    v = np.zeros(n)  # First moment
    s = np.zeros(n)  # Second moment
    
    for t in range(1, epochs + 1):
        # Compute gradients
        predictions = X.dot(theta)
        gradients = X.T.dot(predictions - y) / m
        
        # Update biased first moment
        v = beta1 * v + (1 - beta1) * gradients
        
        # Update biased second moment
        s = beta2 * s + (1 - beta2) * np.square(gradients)
        
        # Bias correction
        v_corrected = v / (1 - beta1**t)
        s_corrected = s / (1 - beta2**t)
        
        # Update parameters
        theta -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
        
    return theta
```

Slide 7: Real-World Application: Stock Price Prediction

Implementing gradient descent for predicting stock prices using a simple linear regression model. This example demonstrates data preprocessing, model implementation, and evaluation on real market data.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def stock_price_prediction():
    # Generate sample stock data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = np.random.randn(len(dates)).cumsum() + 100
    
    # Create features (using last 5 days to predict next day)
    def create_features(data, lookback=5):
        X, y = [], []
        for i in range(len(data)-lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)
    
    # Preprocess data
    scaler = MinMaxScaler()
    normalized_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    X, y = create_features(normalized_prices)
    
    # Train model using gradient descent
    theta = adam_optimizer(X, y)
    
    return theta, scaler

# Run prediction
theta, scaler = stock_price_prediction()
print(f"Trained parameters: {theta}")
```

Slide 8: Source Code for Stock Price Prediction Results

```python
def evaluate_stock_predictions(theta, X, y, scaler):
    predictions = X.dot(theta)
    
    # Denormalize predictions and actual values
    predictions_actual = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    
    # Calculate metrics
    mse = np.mean((predictions_actual - y_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_actual - y_actual))
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    return predictions_actual, y_actual
```

Slide 9: Implementing Gradient Clipping

Gradient clipping prevents exploding gradients by scaling them when they exceed a threshold, ensuring stable training in deep networks or recurrent architectures where gradients can become numerically unstable.

```python
def gradient_descent_with_clipping(X, y, learning_rate=0.01, clip_value=5.0, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(epochs):
        predictions = X.dot(theta)
        gradients = X.T.dot(predictions - y) / m
        
        # Compute gradient norm
        grad_norm = np.linalg.norm(gradients)
        
        # Apply gradient clipping
        if grad_norm > clip_value:
            gradients = gradients * (clip_value / grad_norm)
        
        # Update parameters
        theta -= learning_rate * gradients
        
    return theta
```

Slide 10: Line Search Optimization

Line search methods adaptively determine step sizes by finding the optimal learning rate in the descent direction, improving convergence stability and speed compared to fixed learning rates.

```python
def line_search_gradient_descent(X, y, max_iter=100, c=0.5, tau=0.5):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(max_iter):
        predictions = X.dot(theta)
        gradients = X.T.dot(predictions - y) / m
        
        # Initial step size
        alpha = 1.0
        initial_cost = cost_function(X, y, theta)
        
        # Backtracking line search
        while cost_function(X, y, theta - alpha * gradients) > initial_cost - c * alpha * np.sum(gradients**2):
            alpha *= tau
            
        # Update parameters
        theta -= alpha * gradients
        
    return theta
```

Slide 11: Real-World Application: Housing Price Prediction

This implementation demonstrates gradient descent application in predicting housing prices using multiple features, incorporating regularization to prevent overfitting and handling multicollinear predictors.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def housing_price_predictor(X, y, alpha=0.01, lambda_reg=0.1, epochs=1000):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    m, n = X_scaled.shape
    
    # Initialize parameters
    theta = np.zeros(n)
    costs = []
    
    for _ in range(epochs):
        # Compute predictions
        predictions = X_scaled.dot(theta)
        
        # Compute gradients with regularization
        gradients = (1/m) * X_scaled.T.dot(predictions - y)
        gradients += (lambda_reg/m) * theta
        
        # Update parameters
        theta -= alpha * gradients
        
        # Compute cost
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost += (lambda_reg/(2*m)) * np.sum(theta**2)
        costs.append(cost)
    
    return theta, costs, scaler

# Example usage with synthetic data
X = np.random.randn(1000, 5)  # 5 features
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(1000)
theta, costs, scaler = housing_price_predictor(X, y)
```

Slide 12: Nesterov Accelerated Gradient

Nesterov momentum improves upon standard momentum by computing gradients at the anticipated next position, leading to better convergence rates and more accurate parameter updates.

```python
def nesterov_gradient_descent(X, y, learning_rate=0.01, momentum=0.9, epochs=100):
    m, n = X.shape
    theta = np.zeros(n)
    velocity = np.zeros(n)
    
    for _ in range(epochs):
        # Compute look-ahead position
        theta_lookahead = theta + momentum * velocity
        
        # Compute gradients at look-ahead position
        pred_lookahead = X.dot(theta_lookahead)
        gradients = X.T.dot(pred_lookahead - y) / m
        
        # Update velocity and parameters
        velocity = momentum * velocity - learning_rate * gradients
        theta += velocity
        
        # Optional: Add L2 regularization
        theta *= (1 - learning_rate * 0.01)  # 0.01 is regularization strength
    
    return theta
```

Slide 13: Advanced Learning Rate Scheduling

Learning rate scheduling dynamically adjusts the step size during training, enabling faster initial learning while ensuring fine-grained updates near convergence for optimal parameter estimation.

```python
def cosine_annealing_gradient_descent(X, y, init_lr=0.1, T_max=100, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for epoch in range(epochs):
        # Calculate current learning rate
        current_lr = init_lr * 0.5 * (1 + np.cos(np.pi * (epoch % T_max) / T_max))
        
        # Compute gradients
        predictions = X.dot(theta)
        gradients = X.T.dot(predictions - y) / m
        
        # Update parameters
        theta -= current_lr * gradients
        
        # Optional: Add warm restart
        if epoch % T_max == 0:
            current_lr = init_lr
    
    return theta

def plot_learning_rate_schedule(epochs=1000, T_max=100):
    import matplotlib.pyplot as plt
    
    lr_schedule = [0.1 * 0.5 * (1 + np.cos(np.pi * (e % T_max) / T_max)) 
                  for e in range(epochs)]
    
    plt.plot(lr_schedule)
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    return plt
```

Slide 14: Visualization of Gradient Descent Convergence

```python
import matplotlib.pyplot as plt

def visualize_convergence(costs, title="Gradient Descent Convergence"):
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.grid(True)
    return plt

def plot_contours(X, y, theta_history):
    # Create mesh grid for contour plot
    theta0_vals = np.linspace(theta_history[:, 0].min(), theta_history[:, 0].max(), 100)
    theta1_vals = np.linspace(theta_history[:, 1].min(), theta_history[:, 1].max(), 100)
    
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            theta = np.array([t0, t1])
            J_vals[i, j] = cost_function(X, y, theta)
    
    plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=50)
    plt.plot(theta_history[:, 0], theta_history[:, 1], 'r.-')
    plt.xlabel('θ₀')
    plt.ylabel('θ₁')
    return plt
```

Slide 15: Additional Resources

*   Gradient Descent Optimization Algorithms Overview
    *   [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   Adaptive Methods for Machine Learning
    *   [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   Modern Second-Order Methods for Large-Scale Machine Learning
    *   [https://arxiv.org/abs/2002.09018](https://arxiv.org/abs/2002.09018)
*   Convergence Analysis of Gradient Descent Algorithms
    *   [https://arxiv.org/abs/1810.08033](https://arxiv.org/abs/1810.08033)
*   Practical Recommendations for Gradient-Based Training
    *   Search "Neural Networks: Tricks of the Trade" on Google Scholar

