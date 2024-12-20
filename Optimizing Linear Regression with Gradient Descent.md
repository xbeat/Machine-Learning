## Optimizing Linear Regression with Gradient Descent
Slide 1: Sum of Squared Residuals (SSR) Fundamentals

The Sum of Squared Residuals represents the cumulative difference between observed values and predicted values in a regression model. It forms the foundation of optimization in linear regression by quantifying prediction errors through squared differences.

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_ssr(X, y, slope, intercept):
    # Calculate predicted values
    y_pred = slope * X + intercept
    # Calculate residuals and square them
    residuals = y - y_pred
    ssr = np.sum(residuals ** 2)
    return ssr, y_pred

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Calculate SSR for specific parameters
slope, intercept = 2.5, 0.5
ssr, y_pred = calculate_ssr(X, y, slope, intercept)
print(f"SSR for slope={slope}, intercept={intercept}: {ssr:.2f}")
```

Slide 2: Cost Function Implementation

The cost function quantifies how well our model fits the data by calculating the average squared difference between predictions and actual values. We implement it using vectorized operations for efficiency.

```python
def cost_function(X, y, theta):
    """
    X: Input features (with column of 1s prepended)
    y: Target values
    theta: Parameters [intercept, slope]
    """
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

# Prepare data
X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.array([intercept, slope])
cost = cost_function(X_b, y, theta)
print(f"Cost function value: {cost:.4f}")
```

Slide 3: Gradient Computation

Understanding how to compute gradients is crucial for implementing gradient descent. The gradient represents the direction of steepest ascent in the parameter space of our cost function.

```python
def compute_gradients(X, y, theta):
    """
    Computes partial derivatives of cost function
    Returns gradient vector for both parameters
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradients = (1/m) * X.T.dot(errors)
    return gradients

# Calculate gradients
gradients = compute_gradients(X_b, y, theta)
print(f"Gradients [intercept, slope]: {gradients}")
```

Slide 4: Implementing Gradient Descent

The gradient descent algorithm iteratively updates parameters in the opposite direction of the gradient, scaled by a learning rate, to minimize the cost function and find optimal parameters.

```python
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = []
    theta_history = []
    
    for i in range(iterations):
        prediction = X.dot(theta)
        error = prediction - y
        gradients = (1/m) * X.T.dot(error)
        theta = theta - learning_rate * gradients
        
        cost = (1/(2*m)) * np.sum(error ** 2)
        cost_history.append(cost)
        theta_history.append(theta.copy())
        
    return theta, cost_history, theta_history

# Run gradient descent
initial_theta = np.random.randn(2)
theta_final, cost_history, theta_history = gradient_descent(X_b, y, initial_theta)
print(f"Final parameters: {theta_final}")
```

Slide 5: Learning Rate Optimization

The learning rate significantly impacts convergence speed and stability. Too large values cause overshooting, while too small values result in slow convergence. We implement adaptive learning rate adjustment.

```python
def adaptive_gradient_descent(X, y, theta, initial_lr=0.01, decay=0.95):
    m = len(y)
    learning_rate = initial_lr
    cost_history = []
    
    for epoch in range(1000):
        gradients = compute_gradients(X, y, theta)
        theta = theta - learning_rate * gradients
        
        # Adaptive learning rate
        learning_rate *= decay
        
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
        
        if epoch > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-7:
            break
            
    return theta, cost_history

# Run adaptive gradient descent
theta_adaptive, cost_history = adaptive_gradient_descent(X_b, y, initial_theta)
print(f"Optimized parameters: {theta_adaptive}")
```

Slide 6: Real-world Data Preprocessing

Data preprocessing is crucial for gradient descent optimization. We implement robust scaling and outlier handling techniques using a real estate dataset to demonstrate practical application.

```python
def preprocess_data(X, y):
    # Remove outliers using IQR method
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
    
    X_cleaned = X[mask]
    y_cleaned = y[mask]
    
    # Feature scaling
    X_scaled = (X_cleaned - X_cleaned.mean()) / X_cleaned.std()
    y_scaled = (y_cleaned - y_cleaned.mean()) / y_cleaned.std()
    
    return X_scaled, y_scaled

# Example with real estate data
prices = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000])
sizes = np.array([1400, 1600, 1550, 1800, 1250, 1300, 2200])

X_scaled, y_scaled = preprocess_data(sizes, prices)
print("Scaled features shape:", X_scaled.shape)
print("Scaled target shape:", y_scaled.shape)
```

Slide 7: Mini-batch Gradient Descent Implementation

Mini-batch gradient descent offers a compromise between batch and stochastic gradient descent, providing better convergence stability while maintaining computational efficiency.

```python
def minibatch_gradient_descent(X, y, theta, batch_size=32, epochs=100, lr=0.01):
    m = len(y)
    cost_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            gradients = compute_gradients(X_batch, y_batch, theta)
            theta = theta - lr * gradients
            
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

# Run mini-batch gradient descent
theta_mini, cost_history_mini = minibatch_gradient_descent(X_b, y, initial_theta)
print(f"Final parameters (mini-batch): {theta_mini}")
```

Slide 8: Momentum-based Gradient Descent

Momentum helps accelerate gradient descent by accumulating past gradients, enabling faster convergence and better navigation of ravines in the loss landscape.

```python
def momentum_gradient_descent(X, y, theta, lr=0.01, beta=0.9, epochs=100):
    velocity = np.zeros_like(theta)
    cost_history = []
    
    for epoch in range(epochs):
        gradients = compute_gradients(X, y, theta)
        
        # Update velocity
        velocity = beta * velocity + (1 - beta) * gradients
        
        # Update parameters
        theta = theta - lr * velocity
        
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

# Apply momentum-based gradient descent
theta_momentum, cost_history_momentum = momentum_gradient_descent(X_b, y, initial_theta)
print(f"Final parameters (momentum): {theta_momentum}")
```

Slide 9: Early Stopping Implementation

Early stopping prevents overfitting by monitoring validation loss and stopping training when performance on validation set starts deteriorating.

```python
def gradient_descent_with_early_stopping(X_train, y_train, X_val, y_val, theta, 
                                       lr=0.01, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    best_theta = None
    
    while patience_counter < patience:
        gradients = compute_gradients(X_train, y_train, theta)
        theta = theta - lr * gradients
        
        val_loss = cost_function(X_val, y_val, theta)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_theta = theta.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
    return best_theta, best_val_loss

# Split data and apply early stopping
X_train, X_val = X_b[:80], X_b[80:]
y_train, y_val = y[:80], y[80:]
theta_early, val_loss = gradient_descent_with_early_stopping(X_train, y_train, 
                                                           X_val, y_val, initial_theta)
print(f"Best validation loss: {val_loss:.4f}")
```

Slide 10: Advanced Optimization with Adam

Adam optimization combines the benefits of momentum and RMSprop, adapting learning rates for each parameter while maintaining momentum for faster convergence.

```python
def adam_optimizer(X, y, theta, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = len(y)
    v = np.zeros_like(theta)  # First moment estimate
    s = np.zeros_like(theta)  # Second moment estimate
    t = 0  # Time step
    
    cost_history = []
    
    for epoch in range(1000):
        t += 1
        gradients = compute_gradients(X, y, theta)
        
        # Update biased first moment estimate
        v = beta1 * v + (1 - beta1) * gradients
        
        # Update biased second moment estimate
        s = beta2 * s + (1 - beta2) * np.square(gradients)
        
        # Bias correction
        v_corrected = v / (1 - beta1**t)
        s_corrected = s / (1 - beta2**t)
        
        # Update parameters
        theta = theta - lr * v_corrected / (np.sqrt(s_corrected) + epsilon)
        
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
        
        if epoch > 0 and abs(cost_history[-1] - cost_history[-2]) < 1e-8:
            break
    
    return theta, cost_history

# Run Adam optimization
theta_adam, cost_history_adam = adam_optimizer(X_b, y, initial_theta)
print(f"Final parameters (Adam): {theta_adam}")
```

Slide 11: Computing Loss Surface Visualization

Visualizing the loss surface helps understand optimization trajectory and identify potential challenges in convergence.

```python
def plot_loss_surface(X, y, theta_range=(-2, 2), resolution=100):
    w0 = np.linspace(theta_range[0], theta_range[1], resolution)
    w1 = np.linspace(theta_range[0], theta_range[1], resolution)
    W0, W1 = np.meshgrid(w0, w1)
    Z = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        for j in range(resolution):
            theta = np.array([W0[i,j], W1[i,j]])
            Z[i,j] = cost_function(X, y, theta)
    
    return W0, W1, Z

# Generate loss surface data
W0, W1, Z = plot_loss_surface(X_b, y)

# Create contour plot
plt.figure(figsize=(10, 8))
plt.contour(W0, W1, Z, levels=50)
plt.colorbar(label='Loss')
plt.xlabel('w0 (intercept)')
plt.ylabel('w1 (slope)')
plt.title('Loss Surface Contours')
plt.savefig('loss_surface.png')
plt.close()
```

Slide 12: Results Analysis and Visualization

Comprehensive analysis of optimization results across different algorithms, including convergence rates and final performance metrics.

```python
def analyze_optimization_results(results_dict):
    plt.figure(figsize=(12, 6))
    
    for name, history in results_dict.items():
        plt.plot(history, label=name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('convergence_comparison.png')
    plt.close()
    
    # Compare final costs
    final_costs = {name: history[-1] for name, history in results_dict.items()}
    for name, cost in final_costs.items():
        print(f"{name} final cost: {cost:.6f}")

# Analyze results
results = {
    'Standard GD': cost_history,
    'Momentum': cost_history_momentum,
    'Adam': cost_history_adam
}
analyze_optimization_results(results)
```

Slide 13: Additional Resources

*   "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" - [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
*   "On the Convergence of Adam and Beyond" - [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)
*   "Why Momentum Really Works" - [https://distill.pub/2017/momentum/](https://distill.pub/2017/momentum/)
*   "An Overview of Gradient Descent Optimization Algorithms" - [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
*   Search Google Scholar for: "Gradient Descent Optimization Algorithms Review"

