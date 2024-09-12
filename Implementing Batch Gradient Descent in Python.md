## Implementing Batch Gradient Descent in Python
Slide 1: Introduction to Batch Gradient Descent

Batch Gradient Descent is a fundamental optimization algorithm used in machine learning to minimize the cost function of a model. It updates the model parameters by computing the gradient of the entire training dataset in each iteration. This approach ensures stable convergence but can be computationally expensive for large datasets.

```python
import numpy as np

def batch_gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(num_iterations):
        h = np.dot(X, theta)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
    
    return theta
```

Slide 2: Cost Function and Gradient

The cost function measures the difference between predicted and actual values. For linear regression, we use the Mean Squared Error (MSE). The gradient of the cost function with respect to the parameters indicates the direction of steepest ascent.

```python
def cost_function(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    grad = (1/m) * np.dot(X.T, (predictions - y))
    return grad
```

Slide 3: Implementing the Optimizer

Our Batch Gradient Descent optimizer will iterate through a fixed number of steps, updating the parameters in each iteration based on the computed gradient.

```python
def batch_gradient_descent(X, y, learning_rate, num_iterations):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(num_iterations):
        grad = gradient(X, y, theta)
        theta -= learning_rate * grad
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history
```

Slide 4: Preparing the Data

Before applying the optimizer, we need to prepare our data. This includes normalization and adding a bias term to our feature matrix.

```python
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def add_bias_term(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

# Example usage
X_raw = np.random.randn(100, 3)
y = np.random.randn(100)

X_normalized = normalize_features(X_raw)
X = add_bias_term(X_normalized)
```

Slide 5: Hyperparameter Tuning

The learning rate and number of iterations are crucial hyperparameters. A learning rate that's too high may cause divergence, while one that's too low may result in slow convergence.

```python
learning_rates = [0.001, 0.01, 0.1, 1.0]
iterations = [100, 500, 1000]

best_cost = float('inf')
best_params = None

for lr in learning_rates:
    for iters in iterations:
        theta, cost_history = batch_gradient_descent(X, y, lr, iters)
        final_cost = cost_history[-1]
        if final_cost < best_cost:
            best_cost = final_cost
            best_params = (lr, iters)

print(f"Best parameters: Learning Rate = {best_params[0]}, Iterations = {best_params[1]}")
```

Slide 6: Visualizing Convergence

Plotting the cost function over iterations helps us understand the convergence behavior of our optimizer.

```python
import matplotlib.pyplot as plt

def plot_convergence(cost_history):
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Batch Gradient Descent')
    plt.show()

# Assuming we've run our optimizer
theta, cost_history = batch_gradient_descent(X, y, 0.01, 1000)
plot_convergence(cost_history)
```

Slide 7: Real-Life Example: House Price Prediction

Let's apply our Batch Gradient Descent optimizer to predict house prices based on features like square footage and number of bedrooms.

```python
# Simulated dataset
np.random.seed(42)
square_feet = np.random.randint(1000, 5000, 1000)
bedrooms = np.random.randint(1, 6, 1000)
prices = 100000 + 100 * square_feet + 20000 * bedrooms + np.random.randn(1000) * 50000

X = np.column_stack((square_feet, bedrooms))
y = prices

X_normalized = normalize_features(X)
X_with_bias = add_bias_term(X_normalized)

theta, cost_history = batch_gradient_descent(X_with_bias, y, 0.01, 1000)

print("Learned parameters:", theta)
plot_convergence(cost_history)
```

Slide 8: Making Predictions

Once we have our optimized parameters, we can use them to make predictions on new data.

```python
def predict(X, theta):
    return np.dot(X, theta)

# New house: 2500 sq ft, 3 bedrooms
new_house = np.array([[2500, 3]])
new_house_normalized = (new_house - np.mean(X, axis=0)) / np.std(X, axis=0)
new_house_with_bias = add_bias_term(new_house_normalized)

predicted_price = predict(new_house_with_bias, theta)
print(f"Predicted price for a 2500 sq ft house with 3 bedrooms: ${predicted_price[0]:.2f}")
```

Slide 9: Handling Non-Convergence

Sometimes, the optimizer may not converge due to issues like a high learning rate or ill-conditioned data. We can implement early stopping to handle this.

```python
def batch_gradient_descent_with_early_stopping(X, y, learning_rate, max_iterations, tolerance=1e-6):
    theta = np.zeros(X.shape[1])
    cost_history = []
    
    for i in range(max_iterations):
        prev_theta = theta.()
        grad = gradient(X, y, theta)
        theta -= learning_rate * grad
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
        
        if np.all(np.abs(theta - prev_theta) < tolerance):
            print(f"Converged after {i+1} iterations")
            break
    
    return theta, cost_history

# Example usage
theta, cost_history = batch_gradient_descent_with_early_stopping(X_with_bias, y, 0.01, 10000)
```

Slide 10: Mini-Batch Gradient Descent

For larger datasets, we can use Mini-Batch Gradient Descent, which combines the advantages of both Stochastic and Batch Gradient Descent.

```python
def mini_batch_gradient_descent(X, y, learning_rate, num_iterations, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for _ in range(num_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            grad = gradient(X_batch, y_batch, theta)
            theta -= learning_rate * grad
        
        cost = cost_function(X, y, theta)
        cost_history.append(cost)
    
    return theta, cost_history

# Example usage
theta, cost_history = mini_batch_gradient_descent(X_with_bias, y, 0.01, 1000, 32)
```

Slide 11: Real-Life Example: Iris Flower Classification

Let's use our Batch Gradient Descent optimizer for a classification task on the famous Iris dataset.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare the data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias term
X_train_with_bias = add_bias_term(X_train_scaled)
X_test_with_bias = add_bias_term(X_test_scaled)

# Train the model (using one-vs-rest strategy for multiclass)
theta_list = []
for class_label in range(3):
    y_binary = (y_train == class_label).astype(int)
    theta, _ = batch_gradient_descent(X_train_with_bias, y_binary, 0.01, 1000)
    theta_list.append(theta)

# Make predictions
def predict_iris(X, theta_list):
    predictions = np.array([predict(X, theta) for theta in theta_list]).T
    return np.argmax(predictions, axis=1)

y_pred = predict_iris(X_test_with_bias, theta_list)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on test set: {accuracy:.2f}")
```

Slide 12: Regularization

To prevent overfitting, we can add regularization to our cost function and gradient calculations.

```python
def cost_function_regularized(X, y, theta, lambda_):
    m = len(y)
    predictions = np.dot(X, theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    regularization = (lambda_ / (2*m)) * np.sum(theta[1:]**2)  # Exclude bias term
    return cost + regularization

def gradient_regularized(X, y, theta, lambda_):
    m = len(y)
    predictions = np.dot(X, theta)
    grad = (1/m) * np.dot(X.T, (predictions - y))
    grad[1:] += (lambda_ / m) * theta[1:]  # Regularize all but the bias term
    return grad

def batch_gradient_descent_regularized(X, y, learning_rate, num_iterations, lambda_):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(num_iterations):
        grad = gradient_regularized(X, y, theta, lambda_)
        theta -= learning_rate * grad
        cost = cost_function_regularized(X, y, theta, lambda_)
        cost_history.append(cost)

    return theta, cost_history

# Example usage
lambda_ = 0.1
theta_reg, cost_history_reg = batch_gradient_descent_regularized(X_with_bias, y, 0.01, 1000, lambda_)
```

Slide 13: Momentum-Based Gradient Descent

Momentum can help accelerate convergence, especially in areas where the gradient is small but consistent.

```python
def momentum_gradient_descent(X, y, learning_rate, num_iterations, momentum=0.9):
    theta = np.zeros(X.shape[1])
    velocity = np.zeros_like(theta)
    cost_history = []

    for _ in range(num_iterations):
        grad = gradient(X, y, theta)
        velocity = momentum * velocity - learning_rate * grad
        theta += velocity
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Example usage
theta_momentum, cost_history_momentum = momentum_gradient_descent(X_with_bias, y, 0.01, 1000)
plot_convergence(cost_history_momentum)
```

Slide 14: Comparing Optimizers

Let's compare the performance of our different optimizers on the same dataset.

```python
import time

optimizers = [
    ("Batch GD", batch_gradient_descent),
    ("Mini-Batch GD", lambda X, y, lr, iters: mini_batch_gradient_descent(X, y, lr, iters, 32)),
    ("Momentum GD", momentum_gradient_descent),
    ("Regularized GD", lambda X, y, lr, iters: batch_gradient_descent_regularized(X, y, lr, iters, 0.1))
]

results = {}

for name, optimizer in optimizers:
    start_time = time.time()
    theta, cost_history = optimizer(X_with_bias, y, 0.01, 1000)
    end_time = time.time()
    
    results[name] = {
        "final_cost": cost_history[-1],
        "time": end_time - start_time
    }

for name, result in results.items():
    print(f"{name}: Final Cost = {result['final_cost']:.4f}, Time = {result['time']:.2f} seconds")

# Plot convergence for all optimizers
plt.figure(figsize=(12, 8))
for name, optimizer in optimizers:
    _, cost_history = optimizer(X_with_bias, y, 0.01, 1000)
    plt.plot(range(len(cost_history)), cost_history, label=name)

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence Comparison of Different Optimizers')
plt.legend()
plt.show()
```

Slide 15: Additional Resources

For a deeper understanding of gradient descent and its variants, consider exploring these academic papers:

1. "An overview of gradient descent optimization algorithms" by Sebastian Ruder (2016) ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" by Duchi et al. (2011) ArXiv: [https://arxiv.org/abs/1101.3618](https://arxiv.org/abs/1101.3618)
3. "Adam: A Method for Stochastic Optimization" by Kingma and Ba (2014) ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

These papers provide in-depth analysis and comparisons of various optimization algorithms, including advanced techniques not covered in this presentation.

