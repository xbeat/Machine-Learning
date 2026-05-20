## Hyperparameter Tuning Techniques in Python
Slide 1: Grid Search Cross-Validation

Grid search systematically works through multiple combinations of parameter tunes, cross validates each combination and then provides the best one. It's an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

# Sample data
X = np.random.randn(100, 2)
y = np.random.randint(0, 2, 100)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1],
}

# Initialize model
svm = SVC()

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the model
grid_search.fit(X, y)
print(f"Best parameters: {grid_search.best_params_}")
```

Slide 2: Random Search Optimization

Random search implements a randomized search over parameters, where each setting is sampled from a distribution over possible parameter values. It's often more efficient than grid search when dealing with high-dimensional spaces.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Define parameter distributions
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(5, 30)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Initialize model
rf = RandomForestClassifier()

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the model
random_search.fit(X, y)
print(f"Best parameters: {random_search.best_params_}")
```

Slide 3: Bayesian Optimization

Bayesian optimization uses probabilistic surrogate models to guide the search for optimal hyperparameters, making it more efficient than random or grid search by learning from previous evaluations to suggest better parameter combinations.

```python
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier

# Define search space
search_spaces = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'learning_rate_init': (0.0001, 0.1, 'log-uniform'),
    'max_iter': (100, 500),
    'alpha': (1e-5, 1e-1, 'log-uniform')
}

# Initialize model
mlp = MLPClassifier()

# Setup BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=mlp,
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the model
bayes_search.fit(X, y)
print(f"Best parameters: {bayes_search.best_params_}")
```

Slide 4: Hyperopt Optimization

Hyperopt provides a flexible framework for defining search spaces and implementing various optimization algorithms, including Tree of Parzen Estimators (TPE), which is particularly effective for neural network hyperparameter optimization.

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import numpy as np

def objective(params):
    clf = RandomForestClassifier(**params)
    accuracy = cross_val_score(clf, X, y, cv=5).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}

space = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'n_estimators': hp.choice('n_estimators', range(100,1000)),
    'min_samples_split': hp.uniform('min_samples_split', 2,10),
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(f"Best parameters: {best}")
```

Slide 5: Optuna Framework Implementation

Optuna is a modern hyperparameter optimization framework that provides efficient sampling strategies and pruning mechanisms. It uses a define-by-run API that allows for dynamic construction of search spaces.

```python
import optuna
from sklearn.metrics import accuracy_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'min_samples_split': trial.suggest_float('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }
    
    clf = RandomForestClassifier(**params)
    score = cross_val_score(clf, X, y, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
```

Slide 6: Population-Based Training (PBT)

Population-Based Training combines random search and evolutionary optimization, maintaining a population of models that train in parallel while adapting hyperparameters through competition and reproduction mechanisms, suitable for deep learning applications.

```python
import numpy as np
from typing import Dict, List

class PBTOptimizer:
    def __init__(self, population_size: int, exploit_fraction: float = 0.2):
        self.population_size = population_size
        self.exploit_fraction = exploit_fraction
        self.population = []
        
    def initialize_population(self, param_ranges: Dict):
        for _ in range(self.population_size):
            params = {
                key: np.random.uniform(val[0], val[1]) 
                for key, val in param_ranges.items()
            }
            self.population.append({
                'params': params,
                'score': 0.0
            })
    
    def exploit_and_explore(self, member: Dict) -> Dict:
        # Copy parameters from better performing population member
        better_member = np.random.choice(
            [m for m in self.population 
             if m['score'] > member['score']]
        )
        new_params = better_member['params'].copy()
        
        # Perturb parameters
        for key in new_params:
            if np.random.random() < 0.2:
                new_params[key] *= np.random.choice([0.8, 1.2])
        
        return new_params

    def step(self, scores: List[float]):
        for member, score in zip(self.population, scores):
            member['score'] = score
            
        # Sort population by score
        self.population.sort(key=lambda x: x['score'], reverse=True)
        
        # Replace worst performers
        cutoff = int(self.population_size * (1 - self.exploit_fraction))
        for i in range(cutoff, self.population_size):
            self.population[i]['params'] = self.exploit_and_explore(
                self.population[i]
            )

# Example usage
param_ranges = {
    'learning_rate': [0.0001, 0.1],
    'batch_size': [16, 256],
    'num_layers': [1, 5]
}

pbt = PBTOptimizer(population_size=10)
pbt.initialize_population(param_ranges)
```

Slide 7: Custom Parameter Scheduler

A parameter scheduler allows for dynamic adjustment of hyperparameters during training, implementing strategies like cyclic learning rates or warm restarts to improve model convergence and performance.

```python
import math
from typing import Callable

class ParameterScheduler:
    def __init__(self):
        self.schedulers = {}
        
    def cosine_annealing(self, initial_value: float, 
                        min_value: float, 
                        cycles: int) -> Callable:
        def schedule(epoch: int) -> float:
            cosine = math.cos(math.pi * (epoch % cycles) / cycles)
            value = min_value + 0.5 * (initial_value - min_value) * (1 + cosine)
            return value
        return schedule
    
    def exponential_decay(self, initial_value: float, 
                         decay_rate: float) -> Callable:
        def schedule(epoch: int) -> float:
            return initial_value * (decay_rate ** epoch)
        return schedule
    
    def cyclic_triangular(self, min_value: float, 
                         max_value: float, 
                         period: int) -> Callable:
        def schedule(epoch: int) -> float:
            cycle = epoch % period
            if cycle < period/2:
                return min_value + (max_value - min_value) * (2 * cycle / period)
            return max_value - (max_value - min_value) * (2 * (cycle - period/2) / period)
        return schedule
    
    def add_parameter(self, name: str, scheduler_fn: Callable):
        self.schedulers[name] = scheduler_fn
    
    def get_parameters(self, epoch: int) -> dict:
        return {name: scheduler(epoch) 
                for name, scheduler in self.schedulers.items()}

# Example usage
scheduler = ParameterScheduler()
scheduler.add_parameter(
    'learning_rate',
    scheduler.cosine_annealing(0.1, 0.0001, 10)
)
scheduler.add_parameter(
    'momentum',
    scheduler.cyclic_triangular(0.85, 0.95, 5)
)

# Get parameters for specific epoch
params = scheduler.get_parameters(epoch=5)
print(f"Parameters at epoch 5: {params}")
```

Slide 8: Real-world Implementation: NAS with DARTS

Neural Architecture Search (NAS) using Differentiable Architecture Search (DARTS) demonstrates an advanced hyperparameter optimization technique for finding optimal neural network architectures through gradient descent.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOperation(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)
        self.alpha = nn.Parameter(torch.randn(len(PRIMITIVES)))
            
    def forward(self, x):
        weights = F.softmax(self.alpha, dim=-1)
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class DARTSCell(nn.Module):
    def __init__(self, C_prev, C, reduction):
        super().__init__()
        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        op_names, indices = zip(*self.genotype.normal)
        self.compile(C, op_names, indices, reduction)
        
    def compile(self, C, op_names, indices, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)
```

Slide 9: Hyperparameter Optimization with Weights & Biases

Weights & Biases (wandb) provides a robust platform for tracking and visualizing hyperparameter optimization experiments, offering integration with popular machine learning frameworks and support for distributed optimization.

```python
import wandb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Initialize model with wandb config
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split
        )
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics to wandb
        wandb.log({
            "accuracy": accuracy,
            "feature_importance": model.feature_importances_.tolist()
        })

# Define sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'n_estimators': {'min': 100, 'max': 1000},
        'max_depth': {'min': 5, 'max': 30},
        'min_samples_split': {'min': 2, 'max': 10}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="hyperparameter-optimization")
wandb.agent(sweep_id, train_model, count=50)
```

Slide 10: Multi-Objective Hyperparameter Optimization

Multi-objective optimization handles scenarios where multiple competing objectives need to be optimized simultaneously, using Pareto efficiency concepts to find optimal trade-offs between different metrics.

```python
import numpy as np
from scipy.stats import norm
from typing import List, Tuple

class MultiObjectiveOptimizer:
    def __init__(self, n_objectives: int):
        self.n_objectives = n_objectives
        self.X = []
        self.Y = []
        
    def is_pareto_efficient(self, costs: np.ndarray) -> np.ndarray:
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                )
                is_efficient[i] = True
        return is_efficient
    
    def expected_improvement(self, X: np.ndarray, 
                           pareto_front: np.ndarray) -> np.ndarray:
        mu, sigma = self._gaussian_process(X)
        improvements = []
        
        for y_best in pareto_front:
            with np.errstate(divide='warn'):
                imp = (y_best - mu) / sigma
                ei = sigma * (imp * norm.cdf(imp) + norm.pdf(imp))
                ei[sigma == 0.0] = 0.0
            improvements.append(ei)
            
        return np.array(improvements).mean(axis=0)
    
    def suggest_next_point(self) -> np.ndarray:
        pareto_mask = self.is_pareto_efficient(np.array(self.Y))
        pareto_front = np.array(self.Y)[pareto_mask]
        
        X_test = self._generate_test_points()
        ei = self.expected_improvement(X_test, pareto_front)
        
        return X_test[ei.argmax()]
    
    def update(self, x: np.ndarray, y: List[float]):
        self.X.append(x)
        self.Y.append(y)
```

Slide 11: Implementation of Asynchronous Successive Halving

Asynchronous Successive Halving (ASHA) is a parallelizable hyperparameter optimization algorithm that adaptively allocates resources to more promising configurations while eliminating poor performers early.

```python
from typing import Dict, List
import heapq
import time

class ASHAOptimizer:
    def __init__(self, min_budget: int, max_budget: int, 
                 reduction_factor: int = 3):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.reduction_factor = reduction_factor
        self.brackets = self._create_brackets()
        self.running_trials = {}
        self.completed_trials = {}
        
    def _create_brackets(self) -> List[Dict]:
        brackets = []
        for s in range(self._get_num_brackets()):
            brackets.append({
                'config_queue': [],
                'promotion_queue': [],
                'current_rung': 0
            })
        return brackets
    
    def get_next_config(self) -> Dict:
        for bracket in self.brackets:
            if bracket['config_queue']:
                config = bracket['config_queue'].pop(0)
                trial_id = len(self.running_trials)
                self.running_trials[trial_id] = {
                    'config': config,
                    'bracket': bracket,
                    'current_iter': self.min_budget
                }
                return trial_id, config, self.min_budget
                
        return None, None, None
    
    def report_result(self, trial_id: int, result: float):
        trial = self.running_trials[trial_id]
        bracket = trial['bracket']
        
        if trial['current_iter'] >= self.max_budget:
            self.completed_trials[trial_id] = result
            del self.running_trials[trial_id]
        else:
            heapq.heappush(
                bracket['promotion_queue'],
                (-result, trial_id, trial['config'])
            )
            
        self._promote_configs(bracket)
    
    def _promote_configs(self, bracket: Dict):
        current_rung = bracket['current_rung']
        next_budget = self.min_budget * (
            self.reduction_factor ** (current_rung + 1)
        )
        
        if (len(bracket['promotion_queue']) >= 
            self.reduction_factor * len(bracket['config_queue'])):
            n_promote = len(bracket['promotion_queue']) // self.reduction_factor
            for _ in range(n_promote):
                _, trial_id, config = heapq.heappop(bracket['promotion_queue'])
                if trial_id in self.running_trials:
                    self.running_trials[trial_id]['current_iter'] = next_budget
                    bracket['config_queue'].append(config)
            bracket['current_rung'] += 1
```

Slide 12: Hyperband Implementation

Hyperband optimizes resource allocation by adaptively allocating more resources to promising configurations while using successive halving to efficiently eliminate poor performers, making it particularly effective for deep learning models.

```python
import numpy as np
from math import log, ceil
from typing import Callable, Dict, List

class Hyperband:
    def __init__(self, get_params_function: Callable, 
                 try_params_function: Callable,
                 max_iter: int = 81, eta: int = 3):
        self.get_params = get_params_function
        self.try_params = try_params_function
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(log(max_iter) / log(eta))
        self.B = (self.s_max + 1) * max_iter

    def run(self) -> Dict:
        best_loss = float('inf')
        best_params = None
        
        for s in reversed(range(self.s_max + 1)):
            n = ceil(int(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            
            # Generate configurations
            T = [self.get_params() for _ in range(n)]
            
            for i in range(s + 1):
                n_i = n * self.eta ** (-i)
                r_i = r * self.eta ** i
                
                # Run each configuration for r_i iterations
                val_losses = [self.try_params(t, r_i) for t in T]
                
                # Select top 1/eta configurations
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices[:int(n_i / self.eta)]]
                
                # Update best found configuration
                min_loss_idx = np.argmin(val_losses)
                if val_losses[min_loss_idx] < best_loss:
                    best_loss = val_losses[min_loss_idx]
                    best_params = T[0]
        
        return {
            'best_params': best_params,
            'best_loss': best_loss
        }

# Example usage
def get_random_params():
    return {
        'learning_rate': np.random.uniform(1e-6, 1e-2),
        'batch_size': np.random.choice([16, 32, 64, 128]),
        'n_layers': np.random.randint(1, 5)
    }

def evaluate_params(params, num_iters):
    # Simulate model training and return validation loss
    return np.random.random() * params['learning_rate'] * num_iters

optimizer = Hyperband(
    get_params_function=get_random_params,
    try_params_function=evaluate_params
)
result = optimizer.run()
print(f"Best parameters: {result['best_params']}")
print(f"Best loss: {result['best_loss']}")
```

Slide 13: Cross-Validation with Stratification for Hyperparameter Optimization

Stratified cross-validation ensures balanced representation of classes across folds while performing hyperparameter optimization, particularly important for imbalanced datasets and maintaining consistent evaluation metrics.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple

class StratifiedHyperparameterOptimizer:
    def __init__(self, model_class, param_space: Dict, 
                 n_splits: int = 5):
        self.model_class = model_class
        self.param_space = param_space
        self.n_splits = n_splits
        self.best_params = None
        self.best_score = float('-inf')
        
    def _sample_params(self) -> Dict:
        params = {}
        for param_name, param_range in self.param_space.items():
            if isinstance(param_range, list):
                params[param_name] = np.random.choice(param_range)
            elif isinstance(param_range, tuple):
                low, high = param_range
                params[param_name] = np.random.uniform(low, high)
        return params
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                n_trials: int = 100) -> Tuple[Dict, float]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        
        for _ in range(n_trials):
            params = self._sample_params()
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = self.model_class(**params)
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            mean_score = np.mean(scores)
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
                
        return self.best_params, self.best_score

# Example usage
from sklearn.ensemble import RandomForestClassifier

param_space = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}

optimizer = StratifiedHyperparameterOptimizer(
    RandomForestClassifier,
    param_space
)

X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

best_params, best_score = optimizer.optimize(X, y)
print(f"Best parameters found: {best_params}")
print(f"Best cross-validation score: {best_score:.4f}")
```

Slide 14: Additional Resources

*   arXiv:1905.04970 - "Hyperparameter Optimization: A Spectral Approach" [https://arxiv.org/abs/1905.04970](https://arxiv.org/abs/1905.04970)
*   arXiv:2006.03556 - "Population Based Training of Neural Networks" [https://arxiv.org/abs/2006.03556](https://arxiv.org/abs/2006.03556)
*   arXiv:1902.07638 - "BOHB: Robust and Efficient Hyperparameter Optimization" [https://arxiv.org/abs/1902.07638](https://arxiv.org/abs/1902.07638)
*   arXiv:2003.10865 - "A Survey of Hyperparameter Optimization Methods in Deep Learning" [https://arxiv.org/abs/2003.10865](https://arxiv.org/abs/2003.10865)
*   arXiv:1810.05934 - "Taking the Human Out of the Loop: A Review of Bayesian Optimization" [https://arxiv.org/abs/1810.05934](https://arxiv.org/abs/1810.05934)

