## Mastering Grid Search and Random Search for Hyperparameter Optimization
Slide 1: Introduction to Grid Search

Grid search is a systematic hyperparameter optimization technique that works by exhaustively searching through a predefined set of hyperparameter values. This methodical approach evaluates every possible combination in the search space to find the optimal configuration for a machine learning model.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

# Create sample data
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1],
}

# Initialize model and grid search
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

Slide 2: Random Search Implementation

Random search offers a probabilistic alternative to grid search by randomly sampling from the parameter space. This approach often finds good parameters more efficiently than grid search, especially in high-dimensional spaces where many parameters may have minimal impact on the model.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define random distributions for parameters
param_distributions = {
    'C': uniform(0.1, 10),
    'kernel': ['rbf', 'linear'],
    'gamma': uniform(0.01, 1)
}

# Initialize random search
random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    random_state=42
)

# Fit and evaluate
random_search.fit(X, y)
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

Slide 3: Mathematical Foundation of Grid Search

The mathematical framework behind grid search involves exploring a discrete parameter space systematically. The optimization objective can be expressed using the following mathematical notation, where we aim to minimize the cross-validation error.

```python
"""
Grid Search Optimization Objective:

$$\theta^* = \argmin_{\theta \in \Theta} \frac{1}{K} \sum_{k=1}^K L(D_k, \theta)$$

where:
$$\Theta = \{\theta_1 \times \theta_2 \times ... \times \theta_n\}$$
$$L(D_k, \theta)$$ is the validation loss on fold k
$$K$$ is the number of cross-validation folds
"""

def grid_search_implementation(param_grid, model, X, y, k_folds):
    best_score = float('inf')
    best_params = None
    
    # Generate all combinations
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    return best_params, best_score
```

Slide 4: Cross-Validation in Search Strategies

Cross-validation plays a crucial role in both grid and random search by providing robust estimates of model performance across different hyperparameter combinations. This implementation demonstrates how to properly structure cross-validation within search procedures.

```python
from sklearn.model_selection import KFold
import numpy as np

def custom_cross_validation(model, param_grid, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    # For each fold
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train and evaluate
        model.set_params(**param_grid)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

Slide 5: Advanced Random Search Techniques

Random search can be enhanced by implementing adaptive sampling strategies that focus on promising regions of the parameter space. This implementation demonstrates an intelligent random search that adjusts sampling distributions based on previous results.

```python
import numpy as np
from scipy.stats import norm

class AdaptiveRandomSearch:
    def __init__(self, param_bounds, n_iterations=100):
        self.param_bounds = param_bounds
        self.n_iterations = n_iterations
        self.history = []
        
    def sample_parameters(self, previous_best=None):
        if previous_best is None or len(self.history) < 10:
            # Initial uniform sampling
            return {k: np.random.uniform(v[0], v[1]) 
                   for k, v in self.param_bounds.items()}
        
        # Adaptive sampling around best parameters
        params = {}
        for key, bounds in self.param_bounds.items():
            mean = previous_best[key]
            std = (bounds[1] - bounds[0]) / 4
            params[key] = np.clip(norm.rvs(mean, std), bounds[0], bounds[1])
        return params
```

Slide 6: Parallel Grid Search Implementation

Parallel processing can significantly accelerate grid search by distributing parameter combinations across multiple cores. This implementation showcases how to leverage parallel computing for efficient hyperparameter optimization.

```python
from joblib import Parallel, delayed
from sklearn.base import clone
import multiprocessing

def parallel_grid_search(estimator, param_grid, X, y, cv=5, n_jobs=-1):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    def evaluate_params(params):
        model = clone(estimator)
        model.set_params(**params)
        scores = []
        
        # Cross-validation
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
            
        return params, np.mean(scores)
    
    # Generate parameter combinations
    from itertools import product
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in product(*param_grid.values())]
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(params) 
        for params in param_combinations
    )
    
    # Find best parameters
    best_params, best_score = max(results, key=lambda x: x[1])
    return best_params, best_score
```

Slide 7: Hyperparameter Space Visualization

Understanding the hyperparameter landscape is crucial for optimization. This implementation creates visualization tools to analyze the relationship between parameters and model performance.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def visualize_parameter_space(results_df):
    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Assuming results_df has columns: param1, param2, score
    param1_unique = np.sort(results_df['param1'].unique())
    param2_unique = np.sort(results_df['param2'].unique())
    
    X, Y = np.meshgrid(param1_unique, param2_unique)
    Z = results_df.pivot_table(
        values='score', 
        index='param2', 
        columns='param1'
    ).values
    
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.colorbar(surface)
    
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Score')
    plt.title('Hyperparameter Space Landscape')
    
    return plt
```

Slide 8: Bayesian Optimization Integration

Combining random search with Bayesian optimization can lead to more efficient hyperparameter tuning. This implementation demonstrates how to integrate Gaussian Process-based optimization into the search strategy.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class BayesianOptimizer:
    def __init__(self, param_bounds, n_initial_points=5):
        self.param_bounds = param_bounds
        self.n_initial_points = n_initial_points
        self.X_observed = []
        self.y_observed = []
        
        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            random_state=42
        )
    
    def suggest_next_point(self):
        if len(self.X_observed) < self.n_initial_points:
            # Initial random sampling
            return self._random_sample()
        
        # Update GP model
        self.gp.fit(self.X_observed, self.y_observed)
        
        # Use expected improvement to suggest next point
        x_next = self._maximize_expected_improvement()
        return x_next
    
    def _random_sample(self):
        return np.random.uniform(
            low=[b[0] for b in self.param_bounds.values()],
            high=[b[1] for b in self.param_bounds.values()]
        )
```

Slide 9: Time-Based Early Stopping

Implementing intelligent stopping criteria for search algorithms can save computational resources without significantly compromising performance. This approach monitors convergence and stops when improvements become marginal.

```python
class TimeBasedSearchOptimizer:
    def __init__(self, max_time_hours=1, improvement_threshold=0.001):
        self.max_time = max_time_hours * 3600  # Convert to seconds
        self.threshold = improvement_threshold
        self.start_time = None
        self.best_score = float('-inf')
        self.iterations_without_improvement = 0
        
    def should_continue(self, current_score):
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_time:
            return False
            
        improvement = current_score - self.best_score
        if improvement > self.threshold:
            self.best_score = current_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
            
        return self.iterations_without_improvement < 10

# Example usage
optimizer = TimeBasedSearchOptimizer(max_time_hours=0.5)
while optimizer.should_continue(current_score):
    # Perform search iteration
    pass
```

Slide 10: Custom Parameter Distribution

Creating problem-specific parameter distributions can significantly improve search efficiency. This implementation shows how to define and use custom sampling distributions for different parameter types.

```python
import scipy.stats as stats

class CustomParameterSampler:
    def __init__(self):
        self.distributions = {}
        
    def add_log_uniform(self, name, low, high):
        self.distributions[name] = {
            'type': 'log_uniform',
            'low': np.log(low),
            'high': np.log(high)
        }
        
    def add_categorical(self, name, categories, probabilities=None):
        self.distributions[name] = {
            'type': 'categorical',
            'categories': categories,
            'probabilities': probabilities
        }
        
    def sample(self):
        params = {}
        for name, dist in self.distributions.items():
            if dist['type'] == 'log_uniform':
                value = np.exp(np.random.uniform(dist['low'], dist['high']))
            elif dist['type'] == 'categorical':
                value = np.random.choice(
                    dist['categories'],
                    p=dist['probabilities']
                )
            params[name] = value
        return params

# Example usage
sampler = CustomParameterSampler()
sampler.add_log_uniform('learning_rate', 1e-5, 1e-2)
sampler.add_categorical('activation', ['relu', 'tanh', 'sigmoid'])
```

Slide 11: Real-World Example - Gradient Boosting Optimization

This comprehensive example demonstrates the optimization of a gradient boosting model for a real-world classification task, including data preprocessing and performance evaluation.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess data
def optimize_gbm(X, y, n_trials=50):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter space
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4)
    }
    
    # Random search
    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(),
        param_distributions,
        n_iter=n_trials,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    # Fit and evaluate
    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    
    return best_model, random_search.best_params_, test_score
```

Slide 12: Multi-Objective Parameter Optimization

This implementation extends traditional search methods to handle multiple competing objectives simultaneously, such as model performance versus computational efficiency, using Pareto optimization principles.

```python
class MultiObjectiveOptimizer:
    def __init__(self, objectives):
        self.objectives = objectives
        self.pareto_front = []
        
    def dominates(self, scores1, scores2):
        better_in_any = False
        worse_in_none = True
        
        for s1, s2 in zip(scores1, scores2):
            if s1 < s2:  # Assuming minimization
                better_in_any = True
            elif s1 > s2:
                worse_in_none = False
                
        return better_in_any and worse_in_none
    
    def evaluate_solution(self, params):
        scores = []
        for objective in self.objectives:
            score = objective(params)
            scores.append(score)
        
        # Update Pareto front
        dominated = False
        for idx, (_, existing_scores) in enumerate(self.pareto_front):
            if self.dominates(existing_scores, scores):
                dominated = True
                break
            elif self.dominates(scores, existing_scores):
                self.pareto_front.pop(idx)
                
        if not dominated:
            self.pareto_front.append((params, scores))
        
        return scores
```

Slide 13: Exception Handling and Validation

Robust parameter search requires careful handling of edge cases and validation errors. This implementation shows how to manage common issues during the optimization process.

```python
class RobustParameterSearch:
    def __init__(self, param_bounds, timeout=30):
        self.param_bounds = param_bounds
        self.timeout = timeout
        self.failed_combinations = []
        
    def validate_parameters(self, params):
        try:
            # Check parameter types
            for name, value in params.items():
                expected_type = type(self.param_bounds[name][0])
                if not isinstance(value, expected_type):
                    raise ValueError(f"Invalid type for {name}")
                    
            # Check parameter ranges
            for name, value in params.items():
                low, high = self.param_bounds[name]
                if not (low <= value <= high):
                    raise ValueError(f"Value out of range for {name}")
                    
            return True
        except Exception as e:
            self.failed_combinations.append((params, str(e)))
            return False
            
    def safe_evaluate(self, model, params, X, y):
        if not self.validate_parameters(params):
            return None
            
        try:
            with timeout(seconds=self.timeout):
                model.set_params(**params)
                scores = cross_val_score(model, X, y, cv=5)
                return np.mean(scores)
        except Exception as e:
            self.failed_combinations.append((params, str(e)))
            return None
```

Slide 14: Additional Resources

*   Deep Exploration of Hyperparameter Optimization Methods - [https://arxiv.org/abs/2402.05070](https://arxiv.org/abs/2402.05070)
*   Neural Architecture Search (NAS) with Random Search - [https://arxiv.org/abs/2210.06423](https://arxiv.org/abs/2210.06423)
*   Bayesian Optimization for Automated Machine Learning - [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
*   Google Research Papers on Hyperparameter Tuning - [https://research.google.com/pubs/papers/hyperparameter-tuning](https://research.google.com/pubs/papers/hyperparameter-tuning)
*   Advanced Techniques in Grid Search - [https://machinelearning.org/research/grid-search-optimization](https://machinelearning.org/research/grid-search-optimization)
*   Efficient Hyperparameter Optimization - [https://papers.neurips.cc/paper/hyperparameter-optimization](https://papers.neurips.cc/paper/hyperparameter-optimization)

