## Bayesian Optimization for ML Hyperparameter Tuning
Slide 1: Bayesian Optimization Fundamentals

Bayesian optimization provides a sophisticated approach to hyperparameter tuning by modeling the objective function probabilistically. It constructs a surrogate model using Gaussian Processes to approximate the relationship between hyperparameters and model performance.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class BayesianOptimizer:
    def __init__(self, bounds):
        self.bounds = bounds
        self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.X_observed = []
        self.y_observed = []
        
    def suggest_next_point(self, n_samples=1000):
        X_random = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_samples, self.bounds.shape[0]))
        if len(self.X_observed) == 0:
            return X_random[0]
            
        mu, sigma = self.gp.predict(X_random, return_std=True)
        acquisition = mu - 1.96 * sigma
        return X_random[np.argmin(acquisition)]
```

Slide 2: Gaussian Process Surrogate Model

The surrogate model uses Gaussian Processes to estimate both the expected performance and uncertainty for any given hyperparameter configuration. This probabilistic approach enables informed decisions about which configurations to evaluate next.

```python
def fit_surrogate_model(X_observed, y_observed):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    
    # Reshape inputs if necessary
    X = np.array(X_observed).reshape(-1, 1)
    y = np.array(y_observed).reshape(-1, 1)
    
    # Fit the Gaussian Process model
    gp.fit(X, y)
    
    # Generate predictions for visualization
    X_test = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    mean_prediction, std_prediction = gp.predict(X_test, return_std=True)
    
    return gp, X_test, mean_prediction, std_prediction
```

Slide 3: Acquisition Functions Implementation

Acquisition functions guide the optimization process by balancing exploration and exploitation. The Expected Improvement (EI) function calculates the probability that a new point will improve upon the current best observation.

```python
def expected_improvement(X, X_observed, y_observed, gp, xi=0.01):
    mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)
    mu_sample = gp.predict(X_observed.reshape(-1, 1))
    
    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu_sample)
    
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        
    return ei
```

Slide 4: LASSO Hyperparameter Optimization

Real-world application demonstrating Bayesian optimization for tuning LASSO regression's alpha parameter. This implementation shows how to optimize regularization strength using cross-validation performance metrics.

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import numpy as np

def objective_function(alpha, X, y):
    model = Lasso(alpha=alpha)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -np.mean(cv_scores)  # Return negative MSE for minimization

class LassoBayesianOptimizer:
    def __init__(self, X, y):
        self.X_data = X
        self.y_data = y
        self.bounds = np.array([[1e-5, 100]])  # Alpha bounds
        self.optimizer = BayesianOptimizer(self.bounds)
    
    def optimize(self, n_iterations=20):
        for i in range(n_iterations):
            next_alpha = self.optimizer.suggest_next_point()
            score = objective_function(next_alpha[0], self.X_data, self.y_data)
            self.optimizer.X_observed.append(next_alpha)
            self.optimizer.y_observed.append(score)
            self.optimizer.gp.fit(
                np.array(self.optimizer.X_observed).reshape(-1, 1),
                np.array(self.optimizer.y_observed)
            )
```

Slide 5: Real-world Example - Housing Price Prediction

This implementation demonstrates Bayesian optimization for a housing price prediction model, showcasing data preprocessing, model training, and hyperparameter optimization in a practical context.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess data
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize optimizer
lasso_optimizer = LassoBayesianOptimizer(X_train, y_train)
lasso_optimizer.optimize(n_iterations=20)
```

Slide 6: Lower Confidence Bound Acquisition Function

The Lower Confidence Bound (LCB) acquisition function provides an alternative approach to balancing exploration and exploitation by considering both the predicted mean and variance, controlled by a trade-off parameter kappa.

```python
def lower_confidence_bound(X, gp, kappa=2.0):
    mu, sigma = gp.predict(X.reshape(-1, 1), return_std=True)
    lcb = mu - kappa * sigma
    return lcb

class LCBOptimizer:
    def __init__(self, bounds, kappa=2.0):
        self.bounds = bounds
        self.kappa = kappa
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
            n_restarts_optimizer=10
        )
    
    def suggest_next_point(self, n_samples=1000):
        X_random = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(n_samples, self.bounds.shape[0])
        )
        lcb_values = lower_confidence_bound(X_random, self.gp, self.kappa)
        return X_random[np.argmin(lcb_values)]
```

Slide 7: Probabilistic Model Update Mechanism

A crucial component of Bayesian optimization is the continuous update of the probabilistic model as new observations become available, refining the surrogate model's predictions and uncertainty estimates.

```python
class BayesianModelUpdater:
    def __init__(self, initial_X, initial_y):
        self.X_history = list(initial_X)
        self.y_history = list(initial_y)
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0),
            alpha=1e-6,
            normalize_y=True
        )
        
    def update_model(self, new_X, new_y):
        # Add new observations to history
        self.X_history.append(new_X)
        self.y_history.append(new_y)
        
        # Reshape data for GP fitting
        X = np.array(self.X_history).reshape(-1, 1)
        y = np.array(self.y_history).reshape(-1, 1)
        
        # Refit the GP model with all available data
        self.gp.fit(X, y)
        
        return self.gp
```

Slide 8: Real-world Example - XGBoost Hyperparameter Optimization

Implementation of Bayesian optimization for tuning XGBoost hyperparameters, demonstrating multi-dimensional optimization with practical constraints and evaluation metrics.

```python
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

class XGBoostBayesianOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Define parameter bounds
        self.bounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 300),
            'min_child_weight': (1, 7)
        }
        
    def objective(self, params):
        model = xgb.XGBRegressor(
            max_depth=int(params[0]),
            learning_rate=params[1],
            n_estimators=int(params[2]),
            min_child_weight=params[3]
        )
        
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_val)
        return mean_squared_error(self.y_val, predictions)
```

Slide 9: Mathematical Foundations of Bayesian Optimization

The theoretical underpinnings of Bayesian optimization involve complex mathematical concepts. Here are the key formulas used in the implementation of the Gaussian Process and acquisition functions.

```python
# Mathematical formulas in LaTeX notation:
"""
$$
\mu_{n+1}(x) = k_n(x)^T [K_n + \sigma^2I]^{-1}y_n
$$

$$
\sigma_{n+1}^2(x) = k(x,x) - k_n(x)^T[K_n + \sigma^2I]^{-1}k_n(x)
$$

$$
EI(x) = (\mu(x) - f(x^+) - \xi)\Phi(Z) + \sigma(x)\phi(Z)
$$

$$
LCB(x) = \mu(x) - \kappa\sigma(x)
$$
"""

def gaussian_process_posterior(X_new, X_train, y_train, kernel, noise=1e-6):
    K = kernel(X_train, X_train) + noise * np.eye(len(X_train))
    K_new = kernel(X_train, X_new)
    
    # Calculate posterior mean and variance
    mu = K_new.T @ np.linalg.solve(K, y_train)
    sigma = kernel(X_new, X_new) - K_new.T @ np.linalg.solve(K, K_new)
    
    return mu, sigma
```

Slide 10: Convergence Monitoring and Stopping Criteria

Implementation of mechanisms to monitor optimization progress and determine when to stop the optimization process based on convergence criteria or computational budget constraints.

```python
class ConvergenceMonitor:
    def __init__(self, tolerance=1e-4, patience=5):
        self.tolerance = tolerance
        self.patience = patience
        self.best_scores = []
        self.consecutive_no_improvement = 0
        
    def check_convergence(self, new_score):
        self.best_scores.append(new_score)
        
        if len(self.best_scores) < 2:
            return False
            
        improvement = self.best_scores[-2] - new_score
        
        if improvement < self.tolerance:
            self.consecutive_no_improvement += 1
        else:
            self.consecutive_no_improvement = 0
            
        return self.consecutive_no_improvement >= self.patience
    
    def get_best_score(self):
        return min(self.best_scores) if self.best_scores else None
```

Slide 11: Multi-Objective Bayesian Optimization

Implementation of multi-objective optimization for scenarios where multiple competing objectives need to be optimized simultaneously, using Pareto efficiency concepts and modified acquisition functions.

```python
class MultiObjectiveBayesianOptimizer:
    def __init__(self, objective_functions, bounds):
        self.objective_functions = objective_functions
        self.bounds = bounds
        self.gps = [GaussianProcessRegressor(kernel=RBF()) for _ in objective_functions]
        self.X_observed = []
        self.Y_observed = []
        
    def is_pareto_efficient(self, costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )
                is_efficient[i] = True
        return is_efficient
    
    def suggest_next_point(self, n_samples=1000):
        X_random = np.random.uniform(
            self.bounds[:, 0], 
            self.bounds[:, 1], 
            size=(n_samples, self.bounds.shape[0])
        )
        
        predictions = []
        for gp in self.gps:
            mu, sigma = gp.predict(X_random, return_std=True)
            predictions.append(mu - 1.96 * sigma)
            
        predictions = np.array(predictions).T
        efficient_mask = self.is_pareto_efficient(predictions)
        
        return X_random[efficient_mask][np.random.choice(sum(efficient_mask))]
```

Slide 12: Parallel Bayesian Optimization Implementation

Efficient implementation of parallel Bayesian optimization that can suggest multiple points for simultaneous evaluation, particularly useful for distributed computing environments.

```python
from concurrent.futures import ThreadPoolExecutor
import threading

class ParallelBayesianOptimizer:
    def __init__(self, objective_function, bounds, n_workers=4):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_workers = n_workers
        self.lock = threading.Lock()
        self.X_observed = []
        self.y_observed = []
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
        )
        
    def suggest_batch_points(self, batch_size):
        points = []
        X_pending = np.array(points)
        
        for _ in range(batch_size):
            X_random = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=(1000, self.bounds.shape[0])
            )
            
            if len(X_pending) > 0:
                # Adjust acquisition function for pending points
                mu, sigma = self.gp.predict(X_random, return_std=True)
                mu_pending, _ = self.gp.predict(X_pending)
                acquisition = mu - 1.96 * sigma - np.min(mu_pending)
            else:
                mu, sigma = self.gp.predict(X_random, return_std=True)
                acquisition = mu - 1.96 * sigma
                
            next_point = X_random[np.argmin(acquisition)]
            points.append(next_point)
            X_pending = np.vstack(points)
            
        return np.array(points)
    
    def optimize_parallel(self, n_iterations):
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for _ in range(n_iterations):
                batch_points = self.suggest_batch_points(self.n_workers)
                futures = [
                    executor.submit(self.objective_function, point)
                    for point in batch_points
                ]
                
                for future, point in zip(futures, batch_points):
                    with self.lock:
                        self.X_observed.append(point)
                        self.y_observed.append(future.result())
                        self.gp.fit(
                            np.array(self.X_observed),
                            np.array(self.y_observed)
                        )
```

Slide 13: Cross-Validation Integration

Implementation of robust cross-validation within the Bayesian optimization framework to ensure reliable performance estimation and prevent overfitting during hyperparameter tuning.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

class CrossValidatedBayesianOptimizer:
    def __init__(self, model_class, param_bounds, n_folds=5):
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.n_folds = n_folds
        self.optimizer = BayesianOptimizer(
            bounds=np.array(list(param_bounds.values()))
        )
        
    def objective_with_cv(self, X, y, params):
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
            
        return np.mean(scores), np.std(scores)
    
    def optimize(self, X, y, n_iterations=50):
        best_score = float('-inf')
        best_params = None
        
        for _ in range(n_iterations):
            next_point = self.optimizer.suggest_next_point()
            params = {
                key: value for key, value in zip(
                    self.param_bounds.keys(), 
                    next_point
                )
            }
            
            mean_score, std_score = self.objective_with_cv(X, y, params)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                
            self.optimizer.X_observed.append(next_point)
            self.optimizer.y_observed.append(-mean_score)  # Minimize negative score
            
        return best_params, best_score
```

Slide 14: Additional Resources

*   "A Tutorial on Bayesian Optimization of Expensive Cost Functions" - [https://arxiv.org/abs/1012.2599](https://arxiv.org/abs/1012.2599)
*   "Taking the Human Out of the Loop: A Review of Bayesian Optimization" - [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
*   "Practical Bayesian Optimization of Machine Learning Algorithms" - [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)
*   "Multi-Task Bayesian Optimization" - [https://arxiv.org/abs/1308.0042](https://arxiv.org/abs/1308.0042)
*   "Scalable Bayesian Optimization Using Deep Neural Networks" - [https://arxiv.org/abs/1502.05700](https://arxiv.org/abs/1502.05700)

