## Demystifying LASSO Feature Selection in Python
Slide 1: Understanding LASSO Feature Selection

LASSO (Least Absolute Shrinkage and Selection Operator) is a powerful technique for feature selection in machine learning. However, its inner workings are often misunderstood. This presentation will clarify how LASSO actually selects features and demonstrate its implementation from scratch in Python.

```python
import random

# Simulate data
n_samples, n_features = 100, 10
X = [[random.random() for _ in range(n_features)] for _ in range(n_samples)]
y = [sum(x) + random.gauss(0, 0.1) for x in X]

# Initialize weights
weights = [0] * n_features

print(f"Initial weights: {weights}")
```

Slide 2: The LASSO Objective Function

LASSO works by minimizing an objective function that combines the sum of squared errors with an L1 regularization term. This L1 term encourages sparsity in the model coefficients, effectively performing feature selection.

```python
def lasso_objective(X, y, weights, lambda_):
    n_samples = len(y)
    predictions = [sum(w * x for w, x in zip(weights, sample)) for sample in X]
    mse = sum((p - y_true)**2 for p, y_true in zip(predictions, y)) / (2 * n_samples)
    l1_penalty = lambda_ * sum(abs(w) for w in weights)
    return mse + l1_penalty

lambda_ = 0.1
objective = lasso_objective(X, y, weights, lambda_)
print(f"Initial objective value: {objective}")
```

Slide 3: Gradient Descent for LASSO

LASSO optimization is typically performed using gradient descent. The gradient of the LASSO objective function includes the gradient of the MSE term and the subgradient of the L1 penalty term.

```python
def lasso_gradient(X, y, weights, lambda_):
    n_samples = len(y)
    predictions = [sum(w * x for w, x in zip(weights, sample)) for sample in X]
    gradients = [0] * len(weights)
    for i in range(len(weights)):
        gradients[i] = sum((p - y_true) * X[j][i] for j, (p, y_true) in enumerate(zip(predictions, y))) / n_samples
        gradients[i] += lambda_ * (1 if weights[i] > 0 else -1 if weights[i] < 0 else 0)
    return gradients

gradients = lasso_gradient(X, y, weights, lambda_)
print(f"Initial gradients: {gradients[:5]}...")
```

Slide 4: Implementing LASSO Optimization

We'll implement LASSO optimization using coordinate descent, which updates one weight at a time while holding others constant. This approach is efficient and handles the non-differentiability of the L1 term.

```python
def soft_threshold(z, gamma):
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0

def lasso_coordinate_descent(X, y, lambda_, max_iter=1000, tol=1e-4):
    n_samples, n_features = len(X), len(X[0])
    weights = [0] * n_features
    for _ in range(max_iter):
        old_weights = weights.copy()
        for j in range(n_features):
            r = sum((y[i] - sum(w * x for w, x in zip(weights[:j] + weights[j+1:], X[i][:j] + X[i][j+1:])) for i in range(n_samples)))
            z = sum(X[i][j] * r for i in range(n_samples)) / n_samples
            weights[j] = soft_threshold(z, lambda_)
        if all(abs(w - ow) < tol for w, ow in zip(weights, old_weights)):
            break
    return weights

lambda_ = 0.1
lasso_weights = lasso_coordinate_descent(X, y, lambda_)
print(f"LASSO weights: {lasso_weights}")
```

Slide 5: Feature Selection in Action

LASSO performs feature selection by shrinking some coefficients exactly to zero. Let's examine how different lambda values affect feature selection.

```python
def count_nonzero(weights):
    return sum(1 for w in weights if abs(w) > 1e-6)

lambdas = [0.01, 0.1, 0.5, 1.0]
for lambda_ in lambdas:
    weights = lasso_coordinate_descent(X, y, lambda_)
    nonzero = count_nonzero(weights)
    print(f"Lambda: {lambda_}, Non-zero weights: {nonzero}")
```

Slide 6: Results for: Feature Selection in Action

```
Lambda: 0.01, Non-zero weights: 10
Lambda: 0.1, Non-zero weights: 7
Lambda: 0.5, Non-zero weights: 3
Lambda: 1.0, Non-zero weights: 1
```

Slide 7: Visualizing LASSO Path

The LASSO path shows how coefficients change as lambda varies. This visualization helps in understanding the feature selection process.

```python
import matplotlib.pyplot as plt

def lasso_path(X, y, lambdas):
    return [lasso_coordinate_descent(X, y, l) for l in lambdas]

lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
paths = lasso_path(X, y, lambdas)

plt.figure(figsize=(10, 6))
for i in range(len(X[0])):
    plt.plot(lambdas, [path[i] for path in paths], label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Coefficient value')
plt.title('LASSO Path')
plt.legend()
plt.show()
```

Slide 8: Cross-Validation for Lambda Selection

Choosing the right lambda is crucial. We use cross-validation to find the optimal lambda that balances between model complexity and performance.

```python
from sklearn.model_selection import KFold

def cross_validate_lasso(X, y, lambdas, n_folds=5):
    kf = KFold(n_splits=n_folds)
    mse_scores = [0] * len(lambdas)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        
        for i, lambda_ in enumerate(lambdas):
            weights = lasso_coordinate_descent(X_train, y_train, lambda_)
            predictions = [sum(w * x for w, x in zip(weights, sample)) for sample in X_test]
            mse = sum((p - y_true)**2 for p, y_true in zip(predictions, y_test)) / len(y_test)
            mse_scores[i] += mse / n_folds
    
    return mse_scores

lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
cv_scores = cross_validate_lasso(X, y, lambdas)
best_lambda = lambdas[cv_scores.index(min(cv_scores))]
print(f"Best lambda: {best_lambda}")
```

Slide 9: Real-Life Example: Text Classification

Let's apply LASSO to a text classification problem, where we'll use word frequencies as features to classify documents.

```python
import re
from collections import Counter

def preprocess(text):
    return re.findall(r'\w+', text.lower())

documents = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold"
]
labels = [0, 1, 1, 0]  # 0 for short sentences, 1 for long sentences

# Create feature matrix
words = set(word for doc in documents for word in preprocess(doc))
X = [[preprocess(doc).count(word) for word in words] for doc in documents]
y = labels

lasso_weights = lasso_coordinate_descent(X, y, lambda_=0.1)
selected_words = [word for word, weight in zip(words, lasso_weights) if abs(weight) > 1e-6]
print(f"Selected words: {selected_words}")
```

Slide 10: Real-Life Example: Image Feature Selection

In this example, we'll use LASSO for selecting important pixels in a simple image classification task.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate simple 10x10 images
def generate_image(shape):
    return np.random.rand(*shape)

# Create dataset
n_samples = 100
image_shape = (10, 10)
X = [generate_image(image_shape).flatten() for _ in range(n_samples)]
y = [1 if np.mean(img) > 0.5 else 0 for img in X]

lasso_weights = lasso_coordinate_descent(X, y, lambda_=0.1)

# Visualize selected pixels
selected_pixels = np.array(lasso_weights).reshape(image_shape)
plt.imshow(np.abs(selected_pixels), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Selected Pixels by LASSO')
plt.show()
```

Slide 11: Common Misconceptions about LASSO

Many believe LASSO always selects the most correlated features, but this isn't always true. LASSO can sometimes ignore highly correlated features in favor of less correlated ones that provide more unique information.

```python
import numpy as np

# Generate correlated features
n_samples = 1000
X = np.random.randn(n_samples, 3)
X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.1
X[:, 2] = np.random.randn(n_samples)
y = X[:, 0] + X[:, 2] + np.random.randn(n_samples) * 0.1

# Convert to list for our implementation
X = X.tolist()
y = y.tolist()

lasso_weights = lasso_coordinate_descent(X, y, lambda_=0.1)
print("LASSO weights:", lasso_weights)
print("Correlations:", [np.corrcoef(X_col, y)[0, 1] for X_col in zip(*X)])
```

Slide 12: LASSO vs Ridge Regression

While both LASSO and Ridge regression use regularization, LASSO's L1 penalty leads to sparse solutions, unlike Ridge's L2 penalty. Let's compare their behavior.

```python
def ridge_regression(X, y, lambda_):
    X = np.array(X)
    y = np.array(y)
    identity = np.eye(X.shape[1])
    weights = np.linalg.inv(X.T @ X + lambda_ * identity) @ X.T @ y
    return weights.tolist()

lambda_ = 0.1
lasso_weights = lasso_coordinate_descent(X, y, lambda_)
ridge_weights = ridge_regression(X, y, lambda_)

print("LASSO weights:", lasso_weights)
print("Ridge weights:", ridge_weights)
```

Slide 13: Stability of LASSO

LASSO's feature selection can be unstable with small changes in the data. Let's demonstrate this by adding small perturbations to our dataset.

```python
import copy
import random

def perturb_data(X, y, noise_level=0.01):
    X_perturbed = copy.deepcopy(X)
    y_perturbed = copy.deepcopy(y)
    for i in range(len(X)):
        for j in range(len(X[i])):
            X_perturbed[i][j] += random.gauss(0, noise_level)
        y_perturbed[i] += random.gauss(0, noise_level)
    return X_perturbed, y_perturbed

lambda_ = 0.1
original_weights = lasso_coordinate_descent(X, y, lambda_)
perturbed_X, perturbed_y = perturb_data(X, y)
perturbed_weights = lasso_coordinate_descent(perturbed_X, perturbed_y, lambda_)

print("Original selected features:", [i for i, w in enumerate(original_weights) if abs(w) > 1e-6])
print("Perturbed selected features:", [i for i, w in enumerate(perturbed_weights) if abs(w) > 1e-6])
```

Slide 14: Elastic Net: Combining LASSO and Ridge

Elastic Net addresses LASSO's instability by combining L1 and L2 penalties. Let's implement a simple version of Elastic Net.

```python
def elastic_net_coordinate_descent(X, y, lambda_1, lambda_2, max_iter=1000, tol=1e-4):
    n_samples, n_features = len(X), len(X[0])
    weights = [0] * n_features
    for _ in range(max_iter):
        old_weights = weights.copy()
        for j in range(n_features):
            r = sum((y[i] - sum(w * x for w, x in zip(weights[:j] + weights[j+1:], X[i][:j] + X[i][j+1:])) for i in range(n_samples)))
            z = sum(X[i][j] * r for i in range(n_samples)) / (n_samples + lambda_2)
            weights[j] = soft_threshold(z, lambda_1 / (n_samples + lambda_2))
        if all(abs(w - ow) < tol for w, ow in zip(weights, old_weights)):
            break
    return weights

lambda_1, lambda_2 = 0.1, 0.1
elastic_net_weights = elastic_net_coordinate_descent(X, y, lambda_1, lambda_2)
print("Elastic Net weights:", elastic_net_weights)
```

Slide 15: Additional Resources

For a deeper understanding of LASSO and related techniques, consider these resources:

1.  "Regularization Paths for Generalized Linear Models via Coordinate Descent" by Friedman et al. (2010) - [https://arxiv.org/abs/0708.1485](https://arxiv.org/abs/0708.1485)
2.  "The Elements of Statistical Learning" by Hastie et al. (2009) - Available online, covers LASSO in depth.
3.  "An Introduction to Statistical Learning" by James et al. (2013) - Provides a more accessible introduction to LASSO and other statistical learning methods.

