## Gradient Descent for Linear Regression with Multiple Features
Slide 1: Gradient Descent for Linear Regression: Scaling to Multiple Features

Gradient descent is a fundamental optimization algorithm used in machine learning, particularly in linear regression. We'll explore how this algorithm scales from simple univariate cases to handling multiple features, making it a powerful tool for complex data analysis.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_descent(X, y, w_history):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='b', label='Data points')
    for w in w_history[::len(w_history)//10]:
        y_pred = X * w
        plt.plot(X, y_pred, color='r', alpha=0.1)
    plt.plot(X, X * w_history[-1], color='g', label='Final fit')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Gradient Descent Progress')
    plt.show()
```

Slide 2: Univariate Linear Regression

In univariate linear regression, we work with a single feature to predict an outcome. The goal is to find the best-fitting line that minimizes the difference between predicted and actual values.

```python
def univariate_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    w = 0
    w_history = [w]
    m = len(y)
    
    for _ in range(iterations):
        y_pred = w * X
        gradient = (1/m) * np.sum((y_pred - y) * X)
        w -= learning_rate * gradient
        w_history.append(w)
    
    return w, w_history

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

final_w, w_history = univariate_gradient_descent(X, y)
plot_gradient_descent(X, y, w_history)
print(f"Final weight: {final_w}")
```

Slide 3: Multivariate Linear Regression

As we scale to multiple features, our linear regression model becomes more complex. Each feature now has its own weight, and we need to update all weights simultaneously during gradient descent.

```python
def multivariate_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    w_history = [w.()]
    
    for _ in range(iterations):
        y_pred = np.dot(X, w)
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        w -= learning_rate * gradient
        w_history.append(w.())
    
    return w, w_history

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([2, 4, 5, 4, 5])

final_w, w_history = multivariate_gradient_descent(X, y)
print(f"Final weights: {final_w}")
```

Slide 4: Vectorized Implementation

Vectorization is crucial for efficient implementation of gradient descent, especially when dealing with large datasets and multiple features. It allows for faster computations and cleaner code.

```python
def vectorized_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    w_history = [w.()]
    
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(w) - y)
        w -= learning_rate * gradient
        w_history.append(w.())
    
    return w, w_history

# Example usage
X = np.column_stack((np.ones(5), np.array([1, 2, 3, 4, 5])))
y = np.array([2, 4, 5, 4, 5])

final_w, w_history = vectorized_gradient_descent(X, y)
print(f"Final weights: {final_w}")
```

Slide 5: Feature Scaling

When working with multiple features, it's important to scale them to ensure that gradient descent converges efficiently. Feature scaling helps prevent some features from dominating others due to differences in magnitude.

```python
def scale_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Example usage
X = np.array([[1, 1000], [2, 2000], [3, 3000], [4, 4000], [5, 5000]])
X_scaled = scale_features(X)

print("Original features:")
print(X)
print("\nScaled features:")
print(X_scaled)
```

Slide 6: Learning Rate and Convergence

The learning rate is a crucial hyperparameter in gradient descent. Too large, and the algorithm may overshoot; too small, and it may converge too slowly. Let's visualize the effect of different learning rates.

```python
def plot_learning_rates(X, y, learning_rates):
    plt.figure(figsize=(12, 8))
    for lr in learning_rates:
        _, w_history = vectorized_gradient_descent(X, y, learning_rate=lr)
        plt.plot(range(len(w_history)), [np.linalg.norm(w) for w in w_history], label=f'LR = {lr}')
    
    plt.xlabel('Iterations')
    plt.ylabel('Norm of weights')
    plt.title('Convergence with different learning rates')
    plt.legend()
    plt.show()

# Example usage
X = np.column_stack((np.ones(5), np.array([1, 2, 3, 4, 5])))
y = np.array([2, 4, 5, 4, 5])
learning_rates = [0.001, 0.01, 0.1, 1.0]

plot_learning_rates(X, y, learning_rates)
```

Slide 7: Batch vs. Stochastic Gradient Descent

Batch gradient descent uses the entire dataset for each update, while stochastic gradient descent uses a single example. Let's implement and compare both approaches.

```python
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(w) - y)
        w -= learning_rate * gradient
    
    return w

def stochastic_gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(iterations):
        for i in range(m):
            gradient = X[i].dot(X[i].dot(w) - y[i])
            w -= learning_rate * gradient
    
    return w

# Example usage
X = np.column_stack((np.ones(100), np.random.rand(100, 2)))
y = 2 + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(100)

w_batch = batch_gradient_descent(X, y)
w_sgd = stochastic_gradient_descent(X, y)

print("Batch GD weights:", w_batch)
print("SGD weights:", w_sgd)
```

Slide 8: Mini-batch Gradient Descent

Mini-batch gradient descent combines the best of both worlds, using a subset of the data for each update. This approach often leads to faster convergence and can be more robust.

```python
def mini_batch_gradient_descent(X, y, batch_size=32, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            gradient = X_batch.T.dot(X_batch.dot(w) - y_batch) / batch_size
            w -= learning_rate * gradient
    
    return w

# Example usage
X = np.column_stack((np.ones(1000), np.random.rand(1000, 2)))
y = 2 + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(1000)

w_mini_batch = mini_batch_gradient_descent(X, y)
print("Mini-batch GD weights:", w_mini_batch)
```

Slide 9: Regularization in Gradient Descent

Regularization helps prevent overfitting by adding a penalty term to the cost function. Let's implement L2 regularization (Ridge regression) in our gradient descent algorithm.

```python
def ridge_gradient_descent(X, y, lambda_reg=0.1, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(w) - y) + (lambda_reg / m) * w
        w -= learning_rate * gradient
    
    return w

# Example usage
X = np.column_stack((np.ones(100), np.random.rand(100, 5)))
y = 2 + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(100)

w_ridge = ridge_gradient_descent(X, y)
print("Ridge regression weights:", w_ridge)
```

Slide 10: Adaptive Learning Rates

Adaptive learning rate methods like AdaGrad adjust the learning rate for each parameter individually, which can lead to faster convergence. Let's implement a simple version of AdaGrad.

```python
def adagrad(X, y, learning_rate=0.01, epsilon=1e-8, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    G = np.zeros(n)
    
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(w) - y)
        G += gradient ** 2
        adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
        w -= adjusted_lr * gradient
    
    return w

# Example usage
X = np.column_stack((np.ones(100), np.random.rand(100, 2)))
y = 2 + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(100)

w_adagrad = adagrad(X, y)
print("AdaGrad weights:", w_adagrad)
```

Slide 11: Real-life Example: Housing Price Prediction

Let's apply our gradient descent algorithm to predict housing prices based on multiple features such as area, number of rooms, and distance to city center.

```python
# Generate synthetic housing data
np.random.seed(42)
n_samples = 1000
area = np.random.uniform(50, 250, n_samples)
rooms = np.random.randint(1, 6, n_samples)
distance = np.random.uniform(1, 30, n_samples)
X = np.column_stack((np.ones(n_samples), area, rooms, distance))
y = 100000 + 1000 * area + 10000 * rooms - 2000 * distance + np.random.normal(0, 10000, n_samples)

# Normalize features
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Apply gradient descent
w = vectorized_gradient_descent(X_normalized, y, learning_rate=0.1, iterations=10000)[0]

# Make predictions
new_house = np.array([1, (150 - np.mean(X[:, 1])) / np.std(X[:, 1]),
                      (3 - np.mean(X[:, 2])) / np.std(X[:, 2]),
                      (10 - np.mean(X[:, 3])) / np.std(X[:, 3])])
predicted_price = new_house.dot(w)

print(f"Predicted price for a house with 150 sq.m, 3 rooms, 10km from city center: ${predicted_price:.2f}")
```

Slide 12: Real-life Example: Customer Churn Prediction

Let's use gradient descent to predict customer churn based on features like usage time, customer support interactions, and product subscriptions.

```python
# Generate synthetic customer data
np.random.seed(42)
n_samples = 1000
usage_time = np.random.uniform(0, 100, n_samples)
support_interactions = np.random.randint(0, 10, n_samples)
subscriptions = np.random.randint(1, 5, n_samples)
X = np.column_stack((np.ones(n_samples), usage_time, support_interactions, subscriptions))
y = (0.5 - 0.01 * usage_time + 0.1 * support_interactions - 0.2 * subscriptions + 
     np.random.normal(0, 0.1, n_samples)) > 0

# Normalize features
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Apply logistic regression with gradient descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_gradient_descent(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    
    for _ in range(iterations):
        z = X.dot(w)
        h = sigmoid(z)
        gradient = X.T.dot(h - y) / m
        w -= learning_rate * gradient
    
    return w

w = logistic_gradient_descent(X_normalized, y)

# Make predictions
new_customer = np.array([1, (50 - np.mean(X[:, 1])) / np.std(X[:, 1]),
                         (5 - np.mean(X[:, 2])) / np.std(X[:, 2]),
                         (2 - np.mean(X[:, 3])) / np.std(X[:, 3])])
churn_probability = sigmoid(new_customer.dot(w))

print(f"Churn probability for a customer with 50 hours usage, 5 support interactions, and 2 subscriptions: {churn_probability:.2%}")
```

Slide 13: Conclusion and Future Directions

Gradient descent is a powerful optimization technique that scales well to handle multiple features in linear regression. We've explored its implementation, variations, and applications to real-world problems. Future directions include:

1. Exploring more advanced optimization algorithms like Adam or RMSprop
2. Implementing gradient descent for non-linear models like neural networks
3. Investigating techniques for handling very large datasets, such as online learning

By mastering gradient descent, you'll have a solid foundation for tackling more complex machine learning challenges.

Slide 14: Additional Resources

For those interested in diving deeper into gradient descent and its applications, here are some valuable resources:

1. "Gradient Descent Revisited" by Sebastian Ruder (arXiv:1609.04747) URL: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "An Overview of Gradient Descent Optimization Algorithms" by Sebastian Ruder (arXiv:1609.04747) URL: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Stochastic Gradient Descent Tricks" by Léon Bottou (arXiv:1206.5533) URL: [https://arxiv.org/abs/1206.5533](https://arxiv.org/abs/1206.5533)
4. "Optimization Methods for Large-Scale Machine Learning" by Léon Bottou, Frank E. Curtis, and Jorge Nocedal (arXiv:1606.04838) URL: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)

These papers provide in-depth analyses of gradient descent algorithms, their variations, and applications in machine learning. They offer valuable insights for both beginners and advanced practitioners looking to optimize their models and understand the theoretical foundations of these techniques.

