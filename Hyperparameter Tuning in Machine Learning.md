## Hyperparameter Tuning in Machine Learning
Slide 1: Hyperparameter Fundamentals in Machine Learning

The distinction between parameters and hyperparameters is fundamental in machine learning. Parameters are learned during training, while hyperparameters control the learning process itself. This implementation demonstrates a basic neural network with configurable hyperparameters.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate=0.01, n_hidden=64, epochs=100):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.epochs = epochs
        
    def initialize_weights(self, input_dim, output_dim):
        self.W1 = np.random.randn(input_dim, self.n_hidden)
        self.W2 = np.random.randn(self.n_hidden, output_dim)
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1)
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.W2)
        return self.z2
```

Slide 2: Grid Search Implementation

Grid search systematically works through multiple combinations of hyperparameter values, evaluating each model configuration. This implementation shows how to perform grid search with cross-validation for hyperparameter optimization.

```python
import itertools
from sklearn.model_selection import KFold

class GridSearch:
    def __init__(self, param_grid, model_class, n_folds=5):
        self.param_grid = param_grid
        self.model_class = model_class
        self.n_folds = n_folds
        
    def generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    def fit(self, X, y):
        best_score = float('-inf')
        kf = KFold(n_splits=self.n_folds, shuffle=True)
        
        for params in self.generate_param_combinations():
            scores = []
            for train_idx, val_idx in kf.split(X):
                model = self.model_class(**params)
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
                
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                self.best_params = params
```

Slide 3: Random Search Optimization

Random search offers a more efficient alternative to grid search by sampling random combinations of hyperparameters. This approach can often find good solutions faster than exhaustive grid search.

```python
import numpy as np
from scipy.stats import uniform, randint

class RandomSearch:
    def __init__(self, param_distributions, model_class, n_iter=100):
        self.param_distributions = param_distributions
        self.model_class = model_class
        self.n_iter = n_iter
    
    def sample_parameters(self):
        params = {}
        for param_name, distribution in self.param_distributions.items():
            if isinstance(distribution, tuple):
                low, high = distribution
                params[param_name] = np.random.uniform(low, high)
            else:
                params[param_name] = distribution.rvs()
        return params
    
    def optimize(self, X_train, y_train, X_val, y_val):
        best_score = float('-inf')
        for _ in range(self.n_iter):
            params = self.sample_parameters()
            model = self.model_class(**params)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                self.best_params = params
```

Slide 4: Bayesian Optimization Framework

Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters, making it more efficient than random or grid search for complex parameter spaces.

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimizer:
    def __init__(self, param_bounds, n_iterations=50):
        self.param_bounds = param_bounds
        self.n_iterations = n_iterations
        self.X_sample = []
        self.y_sample = []
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=25
        )
    
    def acquisition_function(self, X, gp):
        mean, std = gp.predict(X.reshape(1, -1), return_std=True)
        return mean + 1.96 * std  # Upper confidence bound
    
    def optimize(self, objective_function):
        for i in range(self.n_iterations):
            if i < 5:  # Initial random points
                X_next = np.random.uniform(
                    self.param_bounds[:, 0],
                    self.param_bounds[:, 1]
                )
            else:
                X_next = self._suggest_next_point()
            
            y_next = objective_function(X_next)
            self.X_sample.append(X_next)
            self.y_sample.append(y_next)
            self.gp.fit(np.array(self.X_sample), np.array(self.y_sample))
```

Slide 5: Learning Rate Optimization

Learning rate adjustment significantly impacts model convergence and performance. This implementation demonstrates an adaptive learning rate scheduler with exponential decay and warm-up periods for optimal training dynamics.

```python
import numpy as np

class AdaptiveLearningRate:
    def __init__(self, initial_lr=0.1, min_lr=1e-5, decay_rate=0.9, 
                 warmup_steps=1000, patience=10):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.steps = 0
        self.best_loss = float('inf')
        self.bad_epochs = 0
        
    def get_lr(self, current_loss):
        self.steps += 1
        
        # Warm-up phase
        if self.steps < self.warmup_steps:
            return self.initial_lr * (self.steps / self.warmup_steps)
            
        # Decay phase
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            
        if self.bad_epochs >= self.patience:
            self.initial_lr *= self.decay_rate
            self.bad_epochs = 0
            
        return max(self.initial_lr, self.min_lr)
```

Slide 6: Batch Size Dynamic Adjustment

Batch size significantly affects training stability and convergence speed. This implementation provides a dynamic batch size scheduler that adjusts based on training metrics and memory constraints.

```python
class DynamicBatchSizer:
    def __init__(self, initial_batch_size=32, max_batch_size=512, 
                 growth_rate=1.5, memory_limit_gb=8):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.growth_rate = growth_rate
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.performance_history = []
        
    def estimate_memory_usage(self, sample_size_bytes):
        return self.current_batch_size * sample_size_bytes
        
    def adjust_batch_size(self, current_loss, memory_per_sample):
        self.performance_history.append(current_loss)
        
        # Check if loss is stabilizing
        if len(self.performance_history) >= 3:
            loss_diff = abs(self.performance_history[-1] - self.performance_history[-2])
            
            if loss_diff < 0.01 and self.estimate_memory_usage(memory_per_sample) < self.memory_limit:
                self.current_batch_size = min(
                    int(self.current_batch_size * self.growth_rate),
                    self.max_batch_size
                )
                
        return self.current_batch_size
```

Slide 7: Neural Architecture Search

Neural Architecture Search (NAS) automates the process of finding optimal network architectures. This implementation provides a basic framework for architecture search using reinforcement learning.

```python
import numpy as np
from collections import namedtuple

Architecture = namedtuple('Architecture', ['n_layers', 'units_per_layer', 'activation'])

class NeuralArchitectureSearch:
    def __init__(self, max_layers=5, max_units=256, 
                 activations=['relu', 'tanh', 'sigmoid']):
        self.max_layers = max_layers
        self.max_units = max_units
        self.activations = activations
        self.architectures = []
        self.scores = []
        
    def sample_architecture(self):
        n_layers = np.random.randint(1, self.max_layers + 1)
        units = [2**np.random.randint(4, int(np.log2(self.max_units))+1) 
                for _ in range(n_layers)]
        activation = np.random.choice(self.activations)
        
        return Architecture(n_layers, units, activation)
        
    def update_architecture_pool(self, architecture, score):
        self.architectures.append(architecture)
        self.scores.append(score)
        
        # Keep only top 10 architectures
        if len(self.architectures) > 10:
            idx = np.argsort(self.scores)[-10:]
            self.architectures = [self.architectures[i] for i in idx]
            self.scores = [self.scores[i] for i in idx]
```

Slide 8: Cross-Validation Strategy Implementation

Cross-validation is crucial for reliable hyperparameter tuning. This implementation provides a stratified time-series cross-validation approach with proper handling of temporal dependencies.

```python
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin

class TimeSeriesCrossValidator(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, n_splits=5, gap=0, test_size=0.2):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        
    def split(self, X, y=None):
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Calculate test start index
            test_start = n_samples - (i + 1) * test_size
            test_end = n_samples - i * test_size
            
            # Calculate train end index considering gap
            train_end = test_start - self.gap
            
            if train_end > 0:
                train_indices = indices[:train_end]
                test_indices = indices[test_start:test_end]
                yield train_indices, test_indices
                
    def get_n_splits(self):
        return self.n_splits
```

Slide 9: Hyperparameter Evolution Strategy

Evolution strategies provide a nature-inspired approach to hyperparameter optimization. This implementation uses a genetic algorithm to evolve optimal hyperparameter combinations through successive generations.

```python
import numpy as np
from typing import Dict, List, Tuple

class HyperparameterEvolution:
    def __init__(self, param_ranges: Dict, population_size: int = 50, 
                 generations: int = 20, mutation_rate: float = 0.1):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = []
        self.fitness_history = []
        
    def initialize_population(self):
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    individual[param] = np.random.randint(low, high + 1)
                else:
                    individual[param] = np.random.uniform(low, high)
            self.population.append(individual)
    
    def mutate(self, individual: Dict) -> Dict:
        mutated = individual.copy()
        for param in mutated:
            if np.random.random() < self.mutation_rate:
                low, high = self.param_ranges[param]
                if isinstance(low, int) and isinstance(high, int):
                    mutated[param] = np.random.randint(low, high + 1)
                else:
                    mutated[param] = np.random.uniform(low, high)
        return mutated
```

Slide 10: Advanced Early Stopping with Patience

Early stopping is crucial to prevent overfitting while ensuring optimal model convergence. This implementation provides a sophisticated early stopping mechanism with multiple condition checks and state tracking.

```python
class AdvancedEarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 baseline: float = None, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_epoch = 0
        self.best_metrics = None
        self.wait = 0
        self.stopped_epoch = 0
        
    def check_improvement(self, current_metrics: Dict[str, float], 
                         model_weights: Dict) -> Tuple[bool, str]:
        if self.best_metrics is None:
            self.best_metrics = current_metrics
            self.best_weights = model_weights
            return False, "First epoch"
            
        improved = False
        message = "No improvement"
        
        # Check if current metrics are better than best metrics
        for metric, value in current_metrics.items():
            if abs(value - self.best_metrics[metric]) > self.min_delta:
                if value > self.best_metrics[metric]:
                    improved = True
                    self.best_metrics = current_metrics
                    self.best_weights = model_weights
                    self.wait = 0
                    message = f"Improved {metric}"
                    break
                    
        if not improved:
            self.wait += 1
            if self.wait >= self.patience:
                return True, "Patience exceeded"
                
        return False, message
```

Slide 11: Parameter-Free Optimization

This implementation demonstrates a parameter-free optimization approach that automatically adjusts hyperparameters based on the loss landscape and gradient statistics during training.

```python
import numpy as np
from typing import Callable, List, Optional

class ParameterFreeOptimizer:
    def __init__(self, loss_func: Callable, gradient_func: Callable):
        self.loss_func = loss_func
        self.gradient_func = gradient_func
        self.iteration = 0
        self.gradient_history: List[np.ndarray] = []
        self.parameter_history: List[np.ndarray] = []
        
    def estimate_lipschitz_constant(self, gradients: np.ndarray, 
                                  parameters: np.ndarray) -> float:
        if len(self.gradient_history) > 0:
            grad_diff = gradients - self.gradient_history[-1]
            param_diff = parameters - self.parameter_history[-1]
            return np.linalg.norm(grad_diff) / (np.linalg.norm(param_diff) + 1e-8)
        return 1.0
        
    def optimize_step(self, parameters: np.ndarray, 
                     gradients: Optional[np.ndarray] = None) -> np.ndarray:
        if gradients is None:
            gradients = self.gradient_func(parameters)
            
        # Estimate optimal step size using Lipschitz constant
        L = self.estimate_lipschitz_constant(gradients, parameters)
        step_size = 1.0 / (L + 1e-8)
        
        # Update parameters
        new_parameters = parameters - step_size * gradients
        
        # Store history
        self.gradient_history.append(gradients)
        self.parameter_history.append(parameters)
        self.iteration += 1
        
        return new_parameters
```

Slide 12: Hyperparameter Scheduling Framework

This advanced implementation provides a flexible framework for scheduling multiple hyperparameters simultaneously throughout the training process, supporting both cyclic and adaptive scheduling strategies.

```python
import numpy as np
from typing import Dict, Callable, Union
from dataclasses import dataclass

@dataclass
class ScheduleConfig:
    initial_value: float
    min_value: float
    max_value: float
    schedule_type: str  # 'cyclic', 'exponential', 'cosine'
    cycle_length: int = None
    decay_rate: float = None

class HyperparameterScheduler:
    def __init__(self, schedules: Dict[str, ScheduleConfig]):
        self.schedules = schedules
        self.current_step = 0
        self.current_values = {
            name: config.initial_value 
            for name, config in schedules.items()
        }
        
    def _cosine_schedule(self, config: ScheduleConfig) -> float:
        cycle_progress = (self.current_step % config.cycle_length) / config.cycle_length
        cosine_value = np.cos(np.pi * cycle_progress)
        value_range = config.max_value - config.min_value
        return config.min_value + 0.5 * value_range * (1 + cosine_value)
    
    def step(self) -> Dict[str, float]:
        for name, config in self.schedules.items():
            if config.schedule_type == 'cyclic':
                self.current_values[name] = self._cosine_schedule(config)
            elif config.schedule_type == 'exponential':
                self.current_values[name] *= config.decay_rate
                self.current_values[name] = max(
                    self.current_values[name], 
                    config.min_value
                )
        
        self.current_step += 1
        return self.current_values
```

Slide 13: Hyperparameter Importance Analysis

This implementation provides tools for analyzing the relative importance of different hyperparameters through sensitivity analysis and feature importance techniques.

```python
import numpy as np
from scipy.stats import spearmanr
from typing import List, Dict, Tuple

class HyperparameterImportance:
    def __init__(self, param_history: List[Dict], 
                 performance_history: List[float]):
        self.param_history = param_history
        self.performance_history = performance_history
        self.importance_scores = {}
        
    def calculate_correlation_importance(self) -> Dict[str, float]:
        param_matrix = []
        param_names = list(self.param_history[0].keys())
        
        for params in self.param_history:
            param_matrix.append([params[name] for name in param_names])
            
        param_matrix = np.array(param_matrix)
        
        for i, param_name in enumerate(param_names):
            correlation, _ = spearmanr(
                param_matrix[:, i], 
                self.performance_history
            )
            self.importance_scores[param_name] = abs(correlation)
            
        return self.importance_scores
        
    def get_top_parameters(self, n: int = 3) -> List[Tuple[str, float]]:
        sorted_scores = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_scores[:n]
```

Slide 14: Additional Resources

*   "Neural Architecture Search with Reinforcement Learning" - [https://arxiv.org/abs/1611.01578](https://arxiv.org/abs/1611.01578)
*   "Hyperparameter Optimization: A Spectral Approach" - [https://arxiv.org/abs/1706.00764](https://arxiv.org/abs/1706.00764)
*   "Population Based Training of Neural Networks" - [https://arxiv.org/abs/1711.09846](https://arxiv.org/abs/1711.09846)
*   "Random Search for Hyper-Parameter Optimization" - [https://jmlr.org/papers/v13/bergstra12a.html](https://jmlr.org/papers/v13/bergstra12a.html)
*   "Practical Guidelines for Hyperparameter Optimization" - Search on Google Scholar for "hyperparameter optimization best practices"
*   "Automated Machine Learning: Methods, Systems, Challenges" - Available on the Springer website
*   For implementation details, visit: scikit-learn.org/stable/modules/grid\_search.html

