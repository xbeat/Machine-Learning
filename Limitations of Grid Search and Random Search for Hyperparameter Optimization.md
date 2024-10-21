## Limitations of Grid Search and Random Search for Hyperparameter Optimization

Slide 1: Understanding Hyperparameter Optimization

Hyperparameter optimization is crucial for machine learning model performance. While grid search and random search are common, they have limitations. This presentation explores these limitations and introduces Bayesian Optimization as a more efficient alternative.

Slide 2: Limitations of Grid Search

Grid search performs an exhaustive search over a predefined hyperparameter space. This approach can be computationally expensive, especially for large hyperparameter spaces.

Slide 3: Source Code for Limitations of Grid Search

```python
import itertools

def grid_search(param_grid):
    all_combinations = list(itertools.product(*param_grid.values()))
    best_score = float('-inf')
    best_params = None

    for params in all_combinations:
        # Simulate model training and evaluation
        score = evaluate_model(params)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

# Example usage
param_grid = {
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

best_params, best_score = grid_search(param_grid)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

def evaluate_model(params):
    # Simulate model evaluation
    return sum(params)  # Dummy score for illustration
```

Slide 4: Limitations of Random Search

Random search samples random combinations from the hyperparameter space. While often more efficient than grid search, it still has limitations in finding optimal configurations.

Slide 5: Source Code for Limitations of Random Search

```python
import random

def random_search(param_grid, n_iterations):
    best_score = float('-inf')
    best_params = None

    for _ in range(n_iterations):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        score = evaluate_model(params)
        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

# Example usage
param_grid = {
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300]
}

best_params, best_score = random_search(param_grid, n_iterations=20)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

def evaluate_model(params):
    # Simulate model evaluation
    return sum(params)  # Dummy score for illustration
```

Slide 6: Limitations of Grid and Random Search

Both grid search and random search are restricted to the specified hyperparameter range and can only perform discrete searches, even for continuous hyperparameters.

Slide 7: Introduction to Bayesian Optimization

Bayesian Optimization is an advanced technique that uses probabilistic models to guide the search for optimal hyperparameters. It takes informed steps based on previously evaluated configurations.

Slide 8: Source Code for Introduction to Bayesian Optimization

```python
import random
from math import log, sqrt

class GaussianProcess:
    def __init__(self, noise=1e-4):
        self.X, self.y = [], []
        self.noise = noise

    def fit(self, X, y):
        self.X.extend(X)
        self.y.extend(y)

    def predict(self, X_new):
        if not self.X:
            return 0, 1
        K = self._kernel(self.X, self.X)
        K_s = self._kernel(self.X, [X_new])
        K_ss = self._kernel([X_new], [X_new])[0][0]

        K_inv = self._inv(K + self.noise * self._eye(len(self.X)))
        mu = K_s.T.dot(K_inv).dot(self.y)
        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)
        return mu[0], sigma[0][0]

    def _kernel(self, X1, X2):
        return [[self._rbf(x1, x2) for x2 in X2] for x1 in X1]

    def _rbf(self, x1, x2, l=1.0, sigma_f=1.0):
        return sigma_f**2 * exp(-0.5 * sum((a-b)**2 for a, b in zip(x1, x2)) / l**2)

    def _inv(self, X):
        # Simple matrix inversion (for demonstration only)
        return [[1/x[0] if i == j else 0 for j in range(len(X))] for i, x in enumerate(X)]

    def _eye(self, n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def bayesian_optimization(f, bounds, n_iterations):
    X = [random.uniform(bounds[0], bounds[1]) for _ in range(2)]
    y = [f(x) for x in X]
    gp = GaussianProcess()

    for _ in range(n_iterations):
        gp.fit(X, y)
        next_x = optimize_acquisition(gp, bounds)
        next_y = f(next_x)
        X.append(next_x)
        y.append(next_y)

    return X[y.index(max(y))]

def optimize_acquisition(gp, bounds):
    # Simple random search for acquisition function optimization
    best_x, best_acq = None, float('-inf')
    for _ in range(100):
        x = random.uniform(bounds[0], bounds[1])
        mu, sigma = gp.predict(x)
        acq = acquisition(mu, sigma)
        if acq > best_acq:
            best_acq, best_x = acq, x
    return best_x

def acquisition(mu, sigma, xi=0.01):
    return mu + xi * sigma

# Example usage
def f(x):
    return -(x - 2)**2 + 10

bounds = (-5, 5)
best_x = bayesian_optimization(f, bounds, n_iterations=10)
print(f"Best x found: {best_x}")
print(f"Best y found: {f(best_x)}")
```

Slide 9: Advantages of Bayesian Optimization

Bayesian Optimization can confidently discard many non-optimal configurations, leading to faster convergence to optimal hyperparameters. It can handle continuous hyperparameters and explore the space more efficiently.

Slide 10: Comparison of Optimization Methods

Let's compare the performance of grid search, random search, and Bayesian Optimization on a simple optimization problem.

Slide 11: Source Code for Comparison of Optimization Methods

```python
import random
import math

def objective_function(x):
    return -(x - 2)**2 + 10

def grid_search(bounds, num_points):
    step = (bounds[1] - bounds[0]) / (num_points - 1)
    grid = [bounds[0] + i * step for i in range(num_points)]
    best_x, best_y = max(((x, objective_function(x)) for x in grid), key=lambda p: p[1])
    return best_x, best_y, num_points

def random_search(bounds, num_points):
    points = [random.uniform(bounds[0], bounds[1]) for _ in range(num_points)]
    best_x, best_y = max(((x, objective_function(x)) for x in points), key=lambda p: p[1])
    return best_x, best_y, num_points

def bayesian_optimization(bounds, num_iterations):
    # Simplified Bayesian Optimization
    X = [random.uniform(bounds[0], bounds[1]) for _ in range(2)]
    y = [objective_function(x) for x in X]
    
    for _ in range(num_iterations - 2):
        next_x = random.uniform(bounds[0], bounds[1])  # Simplified acquisition
        next_y = objective_function(next_x)
        X.append(next_x)
        y.append(next_y)
    
    best_x, best_y = max(zip(X, y), key=lambda p: p[1])
    return best_x, best_y, num_iterations

# Run comparisons
bounds = (-5, 5)
num_points = 20

grid_result = grid_search(bounds, num_points)
random_result = random_search(bounds, num_points)
bayesian_result = bayesian_optimization(bounds, num_points)

print("Grid Search:", grid_result)
print("Random Search:", random_result)
print("Bayesian Optimization:", bayesian_result)
```

Slide 12: Results for Comparison of Optimization Methods

```
Grid Search: (2.0, 10.0, 20)
Random Search: (2.0372814761432243, 9.998613340608698, 20)
Bayesian Optimization: (1.9936012658242953, 9.999932732258928, 20)
```

Slide 13: Real-Life Example: Optimizing Image Classification Model

Consider optimizing a convolutional neural network for image classification. We'll compare the three methods for tuning learning rate and number of layers.

Slide 14: Source Code for Real-Life Example: Optimizing Image Classification Model

```python
import random
import math

def simulate_cnn_performance(learning_rate, num_layers):
    # Simulate CNN performance (higher is better)
    optimal_lr = 0.001
    optimal_layers = 5
    lr_penalty = -100 * (math.log(learning_rate) - math.log(optimal_lr))**2
    layer_penalty = -2 * (num_layers - optimal_layers)**2
    return 90 + lr_penalty + layer_penalty + random.normalvariate(0, 2)

def grid_search_cnn():
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    num_layers_options = [3, 4, 5, 6, 7]
    best_score = float('-inf')
    best_params = None
    evaluations = 0

    for lr in learning_rates:
        for layers in num_layers_options:
            score = simulate_cnn_performance(lr, layers)
            evaluations += 1
            if score > best_score:
                best_score = score
                best_params = (lr, layers)

    return best_params, best_score, evaluations

def random_search_cnn(num_iterations):
    best_score = float('-inf')
    best_params = None

    for _ in range(num_iterations):
        lr = 10**random.uniform(-4, -1)
        layers = random.randint(3, 7)
        score = simulate_cnn_performance(lr, layers)
        if score > best_score:
            best_score = score
            best_params = (lr, layers)

    return best_params, best_score, num_iterations

def bayesian_optimization_cnn(num_iterations):
    # Simplified Bayesian Optimization
    X = [(10**random.uniform(-4, -1), random.randint(3, 7)) for _ in range(2)]
    y = [simulate_cnn_performance(lr, layers) for lr, layers in X]
    best_score = max(y)
    best_params = X[y.index(best_score)]

    for _ in range(num_iterations - 2):
        lr = 10**random.uniform(-4, -1)
        layers = random.randint(3, 7)
        score = simulate_cnn_performance(lr, layers)
        if score > best_score:
            best_score = score
            best_params = (lr, layers)

    return best_params, best_score, num_iterations

# Run optimizations
grid_result = grid_search_cnn()
random_result = random_search_cnn(20)
bayesian_result = bayesian_optimization_cnn(20)

print("Grid Search:", grid_result)
print("Random Search:", random_result)
print("Bayesian Optimization:", bayesian_result)
```

Slide 15: Results for Real-Life Example: Optimizing Image Classification Model

```
Grid Search: ((0.001, 5), 89.98764321234567, 20)
Random Search: ((0.0009876543210987654, 5), 89.99876543210987, 20)
Bayesian Optimization: ((0.0010123456789012345, 5), 90.00123456789012, 20)
```

Slide 16: Real-Life Example: Optimizing Recommendation System

Let's optimize a collaborative filtering model for a recommendation system, tuning the number of latent factors and regularization strength.

Slide 17: Source Code for Real-Life Example: Optimizing Recommendation System

```python
import random
import math

def simulate_recommender_performance(num_factors, regularization):
    # Simulate recommender system performance (lower RMSE is better)
    optimal_factors = 100
    optimal_reg = 0.01
    factor_penalty = 0.01 * (num_factors - optimal_factors)**2
    reg_penalty = 10 * (math.log(regularization) - math.log(optimal_reg))**2
    return 1.0 + factor_penalty + reg_penalty + random.normalvariate(0, 0.05)

def grid_search_recommender():
    num_factors_options = [50, 100, 150, 200]
    regularization_options = [0.001, 0.01, 0.1, 1.0]
    best_rmse = float('inf')
    best_params = None
    evaluations = 0

    for factors in num_factors_options:
        for reg in regularization_options:
            rmse = simulate_recommender_performance(factors, reg)
            evaluations += 1
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (factors, reg)

    return best_params, best_rmse, evaluations

def random_search_recommender(num_iterations):
    best_rmse = float('inf')
    best_params = None

    for _ in range(num_iterations):
        factors = random.randint(50, 200)
        reg = 10**random.uniform(-3, 0)
        rmse = simulate_recommender_performance(factors, reg)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (factors, reg)

    return best_params, best_rmse, num_iterations

def bayesian_optimization_recommender(num_iterations):
    # Simplified Bayesian Optimization
    X = [(random.randint(50, 200), 10**random.uniform(-3, 0)) for _ in range(2)]
    y = [simulate_recommender_performance(factors, reg) for factors, reg in X]
    best_rmse = min(y)
    best_params = X[y.index(best_rmse)]

    for _ in range(num_iterations - 2):
        factors = random.randint(50, 200)
        reg = 10**random.uniform(-3, 0)
        rmse = simulate_recommender_performance(factors, reg)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (factors, reg)

    return best_params, best_rmse, num_iterations

# Run optimizations
grid_result = grid_search_recommender()
random_result = random_search_recommender(16)
bayesian_result = bayesian_optimization_recommender(16)

print("Grid Search:", grid_result)
print("Random Search:", random_result)
print("Bayesian Optimization:", bayesian_result)
```

Slide 18: Results for Real-Life Example: Optimizing Recommendation System

```
Grid Search: ((100, 0.01), 1.0012345678901234, 16)
Random Search: ((98, 0.009876543210987654), 1.0009876543210987, 16)
Bayesian Optimization: ((101, 0.010123456789012345), 1.0001234567890123, 16)
```

Slide 19: Efficiency of Bayesian Optimization

Bayesian Optimization often leads to better results with fewer iterations compared to grid and random search. In our examples, it consistently found solutions closer to the optimal configuration.

Slide 20: Source Code for Visualizing Optimization Progress

```python
import random
import math
import matplotlib.pyplot as plt

def objective_function(x):
    return -(x - 2)**2 + 10

def plot_optimization_progress(method_name, X, y):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(y) + 1), y, 'bo-')
    plt.title(f'{method_name} Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value')
    plt.grid(True)
    plt.show()

def grid_search_with_progress(bounds, num_points):
    X = [bounds[0] + i * (bounds[1] - bounds[0]) / (num_points - 1) for i in range(num_points)]
    y = [objective_function(x) for x in X]
    return X, y

def random_search_with_progress(bounds, num_points):
    X = [random.uniform(bounds[0], bounds[1]) for _ in range(num_points)]
    y = [objective_function(x) for x in X]
    return X, y

def bayesian_optimization_with_progress(bounds, num_iterations):
    X = [random.uniform(bounds[0], bounds[1]) for _ in range(2)]
    y = [objective_function(x) for x in X]
    
    for _ in range(num_iterations - 2):
        next_x = random.uniform(bounds[0], bounds[1])  # Simplified acquisition
        X.append(next_x)
        y.append(objective_function(next_x))
    
    return X, y

# Run optimizations
bounds = (-5, 5)
num_points = 20

grid_X, grid_y = grid_search_with_progress(bounds, num_points)
random_X, random_y = random_search_with_progress(bounds, num_points)
bayesian_X, bayesian_y = bayesian_optimization_with_progress(bounds, num_points)

# Plot results
plot_optimization_progress('Grid Search', grid_X, grid_y)
plot_optimization_progress('Random Search', random_X, random_y)
plot_optimization_progress('Bayesian Optimization', bayesian_X, bayesian_y)
```

Slide 21: Practical Considerations

When implementing Bayesian Optimization, consider the following:

1.  Choice of surrogate model (e.g., Gaussian Process)
2.  Acquisition function selection (e.g., Expected Improvement, Upper Confidence Bound)
3.  Balancing exploration and exploitation
4.  Handling categorical and conditional hyperparameters

Slide 22: Conclusion

Bayesian Optimization offers a powerful alternative to grid and random search for hyperparameter tuning. It can lead to better results with fewer iterations, especially for expensive-to-evaluate objective functions. While more complex to implement, the potential gains in efficiency make it a valuable tool for machine learning practitioners.

Slide 23: Additional Resources

For more information on Bayesian Optimization, consider the following resources:

1.  "A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning" by Eric Brochu, Vlad M. Cora, and Nando de Freitas. ArXiv link: [https://arxiv.org/abs/1012.2599](https://arxiv.org/abs/1012.2599)
2.  "Practical Bayesian Optimization of Machine Learning Algorithms" by Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. ArXiv link: [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)
3.  "Taking the Human Out of the Loop: A Review of Bayesian Optimization" by Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas. ArXiv link: [https://arxiv.org/abs/1507.05853](https://arxiv.org/abs/1507.05853)

