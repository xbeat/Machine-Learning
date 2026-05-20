## Optimizing Machine Learning with Hyperparameter Tuning
Slide 1: Introduction to Hyperparameter Tuning

Hyperparameter tuning is a critical process in machine learning that involves optimizing the configuration settings used to control the learning process. These parameters, unlike model parameters, cannot be learned directly from the data and must be set before training begins. Effective tuning can significantly impact model performance.

```python
# Basic example of manual hyperparameter tuning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

# Generate sample data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Try different hyperparameters
hyperparams = [
    {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'},
    {'C': 10.0, 'kernel': 'linear'}
]

# Train and evaluate models with different hyperparameters
for params in hyperparams:
    model = SVC(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Parameters: {params}\nAccuracy: {score:.4f}\n")
```

Slide 2: Grid Search Implementation

Grid Search systematically works through a predefined set of hyperparameter values, training a model for each combination. This exhaustive approach ensures finding the optimal configuration within the search space, though it can be computationally expensive for large parameter sets.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create sample dataset
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

# Fit grid search
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

Slide 3: Random Search Optimization

Random Search offers a more efficient alternative to Grid Search by sampling random combinations of hyperparameters. This approach can often find good solutions more quickly than grid search, especially when not all hyperparameters are equally important to the final model performance.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform, randint

# Define parameter distributions
param_distributions = {
    'hidden_layer_sizes': [(100,), (100, 50), (50, 25)],
    'learning_rate_init': uniform(0.0001, 0.01),
    'max_iter': randint(100, 500),
    'alpha': uniform(0.0001, 0.01)
}

# Initialize model
mlp = MLPClassifier(random_state=42)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit random search
random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.4f}")
```

Slide 4: Bayesian Optimization

Bayesian optimization represents an advanced approach to hyperparameter tuning that uses probabilistic models to guide the search process. It constructs a surrogate model of the objective function and uses an acquisition function to determine the next set of parameters to evaluate.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

def objective_function(params):
    # Simulate an expensive model evaluation
    x, y = params
    return -(x**2 + (y-1)**2) + np.random.normal(0, 0.1)

class BayesianOptimizer:
    def __init__(self, bounds):
        self.bounds = bounds
        self.X_observed = []
        self.y_observed = []
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, random_state=42)
    
    def _acquisition_function(self, X, xi=0.01):
        mean, std = self.gp.predict(X.reshape(-1, 2), return_std=True)
        return mean + xi * std
    
    def optimize(self, n_iterations=20):
        for i in range(n_iterations):
            if len(self.X_observed) == 0:
                X_next = np.random.uniform(self.bounds[:, 0], 
                                         self.bounds[:, 1], 
                                         size=(2,))
            else:
                X = np.random.uniform(self.bounds[:, 0],
                                    self.bounds[:, 1],
                                    size=(1000, 2))
                acquisition_values = self._acquisition_function(X)
                X_next = X[np.argmax(acquisition_values)]
            
            y_next = objective_function(X_next)
            
            if len(self.X_observed) == 0:
                self.X_observed = X_next.reshape(1, -1)
                self.y_observed = np.array([y_next])
            else:
                self.X_observed = np.vstack((self.X_observed, X_next))
                self.y_observed = np.append(self.y_observed, y_next)
            
            self.gp.fit(self.X_observed, self.y_observed)
            
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]

# Example usage
bounds = np.array([[-5, 5], [-5, 5]])
optimizer = BayesianOptimizer(bounds)
best_params, best_value = optimizer.optimize()
print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")
```

Slide 5: Cross-Validation Strategies in Hyperparameter Tuning

Cross-validation plays a crucial role in hyperparameter tuning by providing robust estimates of model performance across different data splits. Various cross-validation strategies can be employed depending on the nature of the data and the specific requirements of the problem.

```python
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np

class CrossValidationTuning:
    def __init__(self, X, y, model, param_grid):
        self.X = X
        self.y = y
        self.model = model
        self.param_grid = param_grid
    
    def standard_cv(self, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        return self._perform_cv(kf)
    
    def stratified_cv(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return self._perform_cv(skf)
    
    def timeseries_cv(self, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return self._perform_cv(tscv)
    
    def _perform_cv(self, cv_splitter):
        best_score = -np.inf
        best_params = None
        
        for params in self._generate_param_combinations():
            scores = []
            for train_idx, val_idx in cv_splitter.split(self.X, self.y):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                self.model.set_params(**params)
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        return best_params, best_score
    
    def _generate_param_combinations(self):
        # Generate all combinations of parameters
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))

# Example usage
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

model = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20]
}

cv_tuner = CrossValidationTuning(X, y, model, param_grid)
best_params, best_score = cv_tuner.standard_cv()
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")
```

Slide 6: Learning Rate Optimization

Learning rate optimization is a crucial aspect of hyperparameter tuning that directly impacts model convergence and performance. Various scheduling techniques can be implemented to dynamically adjust the learning rate during training, helping avoid local minima and achieve better results.

```python
import numpy as np
import matplotlib.pyplot as plt

class LearningRateScheduler:
    def __init__(self, initial_lr=0.1):
        self.initial_lr = initial_lr
        
    def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10.0):
        """Step decay schedule"""
        lr = self.initial_lr * np.power(drop_rate, np.floor((1+epoch)/epochs_drop))
        return lr
    
    def exponential_decay(self, epoch, decay_rate=0.95):
        """Exponential decay schedule"""
        return self.initial_lr * np.power(decay_rate, epoch)
    
    def cosine_decay(self, epoch, total_epochs=100):
        """Cosine annealing schedule"""
        return self.initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    
    def plot_schedules(self, epochs=100):
        epochs_range = range(epochs)
        
        step_rates = [self.step_decay(e) for e in epochs_range]
        exp_rates = [self.exponential_decay(e) for e in epochs_range]
        cos_rates = [self.cosine_decay(e, epochs) for e in epochs_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, step_rates, label='Step Decay')
        plt.plot(epochs_range, exp_rates, label='Exponential Decay')
        plt.plot(epochs_range, cos_rates, label='Cosine Annealing')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedules')
        plt.legend()
        plt.grid(True)
        return plt.gcf()

# Example usage
scheduler = LearningRateScheduler(initial_lr=0.1)
fig = scheduler.plot_schedules()
```

Slide 7: Population-Based Training

Population-based training combines parallel optimization with adaptive hyperparameter scheduling, allowing for dynamic adjustment of hyperparameters during training. This approach is particularly effective for complex models with multiple interacting hyperparameters.

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Individual:
    hyperparameters: Dict
    fitness: float = 0.0
    
class PopulationBasedTraining:
    def __init__(self, population_size: int, generations: int):
        self.population_size = population_size
        self.generations = generations
        self.population: List[Individual] = []
        
    def initialize_population(self, param_ranges: Dict):
        """Initialize random population"""
        for _ in range(self.population_size):
            hyperparameters = {
                param: np.random.uniform(ranges[0], ranges[1])
                for param, ranges in param_ranges.items()
            }
            self.population.append(Individual(hyperparameters))
    
    def evaluate_individual(self, individual: Individual) -> float:
        """Simulate model training and evaluation"""
        # This would typically involve training a model with the given hyperparameters
        params = individual.hyperparameters
        # Simplified fitness function for demonstration
        fitness = -(params['learning_rate'] - 0.01)**2 - (params['dropout'] - 0.5)**2
        return fitness
    
    def evolve_population(self):
        """Main evolution loop"""
        for generation in range(self.generations):
            # Evaluate fitness for each individual
            with ProcessPoolExecutor() as executor:
                fitnesses = list(executor.map(self.evaluate_individual, self.population))
            
            for ind, fitness in zip(self.population, fitnesses):
                ind.fitness = fitness
            
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Replace bottom 20% with mutated versions of top 20%
            cutoff = int(self.population_size * 0.2)
            for i in range(cutoff):
                # Create mutated version of top performer
                parent = self.population[i]
                child_params = {
                    param: value * np.random.normal(1, 0.1)
                    for param, value in parent.hyperparameters.items()
                }
                self.population[-i-1] = Individual(child_params)
            
            print(f"Generation {generation + 1}: Best fitness = {self.population[0].fitness:.4f}")
            
        return self.population[0]

# Example usage
param_ranges = {
    'learning_rate': (0.0001, 0.1),
    'dropout': (0.1, 0.9)
}

pbt = PopulationBasedTraining(population_size=20, generations=10)
pbt.initialize_population(param_ranges)
best_individual = pbt.evolve_population()
print("\nBest hyperparameters found:")
for param, value in best_individual.hyperparameters.items():
    print(f"{param}: {value:.4f}")
```

Slide 8: Hyperparameter Importance Analysis

Understanding the relative importance of different hyperparameters can help focus tuning efforts and reduce computational costs. This implementation demonstrates methods for analyzing hyperparameter sensitivity and impact on model performance.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from scipy.stats import spearmanr

class HyperparameterImportanceAnalyzer:
    def __init__(self, base_model, param_ranges, n_samples=100):
        self.base_model = base_model
        self.param_ranges = param_ranges
        self.n_samples = n_samples
        self.results = None
        
    def sample_hyperparameters(self):
        """Generate random hyperparameter combinations"""
        samples = []
        for _ in range(self.n_samples):
            params = {}
            for param, (low, high) in self.param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param] = np.random.randint(low, high+1)
                else:
                    params[param] = np.random.uniform(low, high)
            samples.append(params)
        return samples
    
    def evaluate_importance(self, X, y):
        """Evaluate hyperparameter importance"""
        samples = self.sample_hyperparameters()
        scores = []
        
        for params in samples:
            model = self.base_model.set_params(**params)
            score = np.mean(cross_val_score(model, X, y, cv=3))
            scores.append(score)
            
        # Create DataFrame with results
        results_df = pd.DataFrame(samples)
        results_df['score'] = scores
        
        # Calculate correlations
        correlations = {}
        for param in self.param_ranges.keys():
            corr, _ = spearmanr(results_df[param], results_df['score'])
            correlations[param] = abs(corr)
            
        self.results = {
            'correlations': correlations,
            'samples': results_df
        }
        
        return self.results
    
    def plot_importance(self):
        """Plot hyperparameter importance"""
        if self.results is None:
            raise ValueError("Run evaluate_importance first")
            
        correlations = self.results['correlations']
        plt.figure(figsize=(10, 6))
        plt.bar(correlations.keys(), correlations.values())
        plt.xticks(rotation=45)
        plt.ylabel('Absolute Correlation with Performance')
        plt.title('Hyperparameter Importance Analysis')
        return plt.gcf()

# Example usage
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

param_ranges = {
    'n_estimators': (50, 200),
    'max_depth': (3, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5)
}

analyzer = HyperparameterImportanceAnalyzer(
    RandomForestClassifier(),
    param_ranges,
    n_samples=50
)

results = analyzer.evaluate_importance(X, y)
fig = analyzer.plot_importance()
```

Slide 9: Early Stopping Implementation

Early stopping is a crucial regularization technique that prevents overfitting by monitoring model performance on a validation set during training. This implementation shows how to effectively implement early stopping with patience and performance tracking.

```python
import numpy as np
from typing import Dict, List, Optional

class EarlyStoppingMonitor:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_weights: Optional[Dict] = None
        self.history: List[float] = []
        
    def __call__(self, current_score: float, weights: Dict) -> bool:
        self.history.append(current_score)
        
        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = weights.copy()
            return False
            
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.best_weights = weights.copy()
        else:
            self.counter += 1
            
        should_stop = self.counter >= self.patience
        return should_stop
    
    def get_best_weights(self) -> Dict:
        return self.best_weights

# Example usage with training loop
def train_with_early_stopping(model, X_train, y_train, X_val, y_val, epochs=100):
    early_stopping = EarlyStoppingMonitor(patience=5)
    
    for epoch in range(epochs):
        # Simulate training step
        train_loss = 1.0 / (epoch + 1) + np.random.normal(0, 0.1)
        
        # Simulate validation step
        val_score = 1.0 - (1.0 / (epoch + 1)) + np.random.normal(0, 0.05)
        
        # Get current weights (simulated)
        current_weights = {'layer1': np.random.randn(5, 5)}
        
        # Check early stopping
        if early_stopping(val_score, current_weights):
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_score={val_score:.4f}")
    
    # Restore best weights
    best_weights = early_stopping.get_best_weights()
    return best_weights, early_stopping.history

# Run example
X_train = np.random.randn(100, 10)
y_train = np.random.randint(0, 2, 100)
X_val = np.random.randn(20, 10)
y_val = np.random.randint(0, 2, 20)

best_weights, history = train_with_early_stopping(None, X_train, y_train, X_val, y_val)
```

Slide 10: Advanced Bayesian Optimization with Parallel Evaluation

Bayesian optimization with parallel evaluation capabilities allows for efficient exploration of hyperparameter space by evaluating multiple parameter configurations simultaneously while maintaining the benefits of guided search.

```python
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from concurrent.futures import ProcessPoolExecutor
import time

class ParallelBayesianOptimizer:
    def __init__(self, objective_function, bounds, n_parallel=4):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.n_parallel = n_parallel
        self.dim = len(bounds)
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
        self.X_observed = np.array([]).reshape(0, self.dim)
        self.y_observed = np.array([])
        
    def _acquisition_function(self, X, xi=0.01):
        """Expected Improvement acquisition function"""
        mu, sigma = self.gp.predict(X.reshape(-1, self.dim), return_std=True)
        
        if len(self.y_observed) > 0:
            current_best = np.max(self.y_observed)
            with np.errstate(divide='warn'):
                imp = mu - current_best - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0
        else:
            ei = mu
            
        return ei
        
    def get_next_points(self, n_points):
        """Get next batch of points to evaluate"""
        if len(self.X_observed) < n_points:
            # Initial random points
            points = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=(n_points, self.dim)
            )
        else:
            # Use acquisition function to select points
            candidates = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=(10000, self.dim)
            )
            
            acq_values = self._acquisition_function(candidates)
            points = candidates[np.argsort(acq_values)[-n_points:]]
            
        return points
        
    def optimize(self, n_iterations=20):
        for i in range(n_iterations):
            # Get next batch of points
            next_points = self.get_next_points(self.n_parallel)
            
            # Evaluate points in parallel
            with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
                results = list(executor.map(self.objective_function, next_points))
            
            # Update observations
            self.X_observed = np.vstack([self.X_observed, next_points])
            self.y_observed = np.append(self.y_observed, results)
            
            # Update Gaussian Process
            self.gp.fit(self.X_observed, self.y_observed)
            
            best_idx = np.argmax(self.y_observed)
            print(f"Iteration {i+1}: Best value = {self.y_observed[best_idx]:.4f}")
        
        return self.X_observed[best_idx], self.y_observed[best_idx]

# Example usage
def objective(params):
    """Example objective function (minimize negative quadratic)"""
    time.sleep(0.1)  # Simulate expensive evaluation
    return -(params[0]**2 + params[1]**2)

bounds = [(-5, 5), (-5, 5)]
optimizer = ParallelBayesianOptimizer(objective, bounds, n_parallel=4)
best_params, best_value = optimizer.optimize(n_iterations=10)

print(f"\nBest parameters found: {best_params}")
print(f"Best value found: {best_value:.4f}")
```

Slide 11: Multi-Objective Hyperparameter Optimization

Multi-objective optimization considers multiple competing objectives simultaneously when tuning hyperparameters. This implementation demonstrates how to handle trade-offs between different performance metrics while finding Pareto-optimal solutions.

```python
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class ParetoSolution:
    parameters: Dict
    objectives: List[float]
    dominated_by: int = 0

class MultiObjectiveOptimizer:
    def __init__(self, objective_functions: List, param_ranges: Dict):
        self.objective_functions = objective_functions
        self.param_ranges = param_ranges
        self.solutions: List[ParetoSolution] = []
        
    def dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """Check if solution 1 dominates solution 2"""
        better_in_any = False
        for obj1, obj2 in zip(sol1.objectives, sol2.objectives):
            if obj1 < obj2:  # Assuming minimization
                return False
            if obj1 > obj2:
                better_in_any = True
        return better_in_any
    
    def generate_random_solution(self) -> Dict:
        """Generate random hyperparameters"""
        return {
            param: np.random.uniform(ranges[0], ranges[1])
            for param, ranges in self.param_ranges.items()
        }
    
    def evaluate_solution(self, params: Dict) -> List[float]:
        """Evaluate all objectives for given parameters"""
        return [obj_func(params) for obj_func in self.objective_functions]
    
    def update_pareto_front(self):
        """Update domination count for all solutions"""
        for sol in self.solutions:
            sol.dominated_by = 0
            
        for i, sol1 in enumerate(self.solutions):
            for j, sol2 in enumerate(self.solutions):
                if i != j and self.dominates(sol2, sol1):
                    sol1.dominated_by += 1
    
    def optimize(self, n_iterations: int = 100, population_size: int = 50):
        """Main optimization loop"""
        # Initialize population
        for _ in range(population_size):
            params = self.generate_random_solution()
            objectives = self.evaluate_solution(params)
            self.solutions.append(ParetoSolution(params, objectives))
        
        for iteration in range(n_iterations):
            # Generate and evaluate new solution
            new_params = self.generate_random_solution()
            new_objectives = self.evaluate_solution(new_params)
            new_solution = ParetoSolution(new_params, new_objectives)
            
            # Add to population and update Pareto front
            self.solutions.append(new_solution)
            self.update_pareto_front()
            
            # Remove dominated solutions if population too large
            if len(self.solutions) > population_size:
                self.solutions.sort(key=lambda x: x.dominated_by)
                self.solutions = self.solutions[:population_size]
            
            if iteration % 10 == 0:
                non_dominated = [s for s in self.solutions if s.dominated_by == 0]
                print(f"Iteration {iteration}: {len(non_dominated)} Pareto-optimal solutions")
        
        return [s for s in self.solutions if s.dominated_by == 0]

# Example usage with competing objectives
def accuracy_objective(params):
    """Simulated accuracy objective"""
    return params['learning_rate'] * (1 - params['dropout'])

def complexity_objective(params):
    """Simulated model complexity objective"""
    return params['hidden_units'] * (1 - params['dropout'])

param_ranges = {
    'learning_rate': (0.0001, 0.1),
    'dropout': (0.1, 0.5),
    'hidden_units': (32, 256)
}

optimizer = MultiObjectiveOptimizer(
    [accuracy_objective, complexity_objective],
    param_ranges
)

pareto_solutions = optimizer.optimize(n_iterations=50)
print("\nPareto-optimal solutions:")
for sol in pareto_solutions[:5]:  # Show first 5 solutions
    print(f"Parameters: {sol.parameters}")
    print(f"Objectives: {sol.objectives}\n")
```

Slide 12: Automated Model Selection and Hyperparameter Tuning

This implementation combines model selection with hyperparameter optimization, automatically choosing the best model architecture and its corresponding hyperparameters based on cross-validated performance metrics.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from typing import Dict, Type, List
import numpy as np

class AutoML:
    def __init__(self, 
                 models: Dict[str, Type],
                 param_spaces: Dict[str, Dict],
                 n_trials: int = 100,
                 cv_folds: int = 5):
        self.models = models
        self.param_spaces = param_spaces
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_model = None
        self.best_params = None
        self.best_score = float('-inf')
        self.results_history = []
        
    def sample_parameters(self, model_name: str) -> Dict:
        """Sample parameters from parameter space"""
        params = {}
        param_space = self.param_spaces[model_name]
        
        for param_name, param_range in param_space.items():
            if isinstance(param_range[0], int):
                params[param_name] = np.random.randint(
                    param_range[0],
                    param_range[1] + 1
                )
            elif isinstance(param_range[0], float):
                params[param_name] = np.random.uniform(
                    param_range[0],
                    param_range[1]
                )
            else:
                params[param_name] = np.random.choice(param_range)
        return params
    
    def evaluate_model(self, X, y, model_name: str, params: Dict) -> float:
        """Evaluate model with given parameters"""
        model = self.models[model_name](**params)
        scores = cross_val_score(
            model, X, y,
            cv=self.cv_folds,
            n_jobs=-1
        )
        return np.mean(scores)
    
    def fit(self, X, y):
        """Main optimization loop"""
        for trial in range(self.n_trials):
            # Randomly select model
            model_name = np.random.choice(list(self.models.keys()))
            
            # Sample parameters
            params = self.sample_parameters(model_name)
            
            # Evaluate model
            score = self.evaluate_model(X, y, model_name, params)
            
            # Store results
            self.results_history.append({
                'trial': trial,
                'model': model_name,
                'params': params,
                'score': score
            })
            
            # Update best model if necessary
            if score > self.best_score:
                self.best_score = score
                self.best_model = model_name
                self.best_params = params
                
                print(f"\nNew best model found (trial {trial}):")
                print(f"Model: {model_name}")
                print(f"Parameters: {params}")
                print(f"Score: {score:.4f}")
        
        # Train final model
        final_model = self.models[self.best_model](**self.best_params)
        final_model.fit(X, y)
        self.best_model_fitted = final_model
        
        return self
    
    def predict(self, X):
        """Predict using best model"""
        if self.best_model_fitted is None:
            raise ValueError("Model not fitted yet")
        return self.best_model_fitted.predict(X)

# Example usage
models = {
    'random_forest': RandomForestClassifier,
    'gradient_boosting': GradientBoostingClassifier,
    'svm': SVC
}

param_spaces = {
    'random_forest': {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10)
    },
    'gradient_boosting': {
        'n_estimators': (50, 200),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10)
    },
    'svm': {
        'C': (0.1, 10.0),
        'kernel': ['rbf', 'linear']
    }
}

# Create and run AutoML
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

automl = AutoML(models, param_spaces, n_trials=50)
automl.fit(X, y)
```

Slide 13: Ensemble-Based Hyperparameter Optimization

Ensemble-based hyperparameter optimization leverages multiple optimization strategies to improve the robustness of the tuning process. This implementation combines different search strategies and aggregates their results to find optimal hyperparameter configurations.

```python
import numpy as np
from typing import List, Dict, Callable
from sklearn.model_selection import cross_val_score
from concurrent.futures import ProcessPoolExecutor

class EnsembleOptimizer:
    def __init__(self, 
                 objective_func: Callable,
                 param_space: Dict,
                 n_iterations: int = 100,
                 ensemble_size: int = 3):
        self.objective_func = objective_func
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.ensemble_size = ensemble_size
        self.best_params = None
        self.best_score = float('-inf')
        
    def random_search(self, n_trials: int) -> tuple:
        """Random search strategy"""
        best_params = None
        best_score = float('-inf')
        
        for _ in range(n_trials):
            params = {
                name: np.random.uniform(low, high)
                for name, (low, high) in self.param_space.items()
            }
            score = self.objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params, best_score
    
    def grid_search(self, n_points: int) -> tuple:
        """Grid search strategy"""
        best_params = None
        best_score = float('-inf')
        
        # Create grid points for each parameter
        param_grids = {
            name: np.linspace(low, high, n_points)
            for name, (low, high) in self.param_space.items()
        }
        
        # Generate all combinations
        from itertools import product
        param_names = list(self.param_space.keys())
        for values in product(*[param_grids[name] for name in param_names]):
            params = dict(zip(param_names, values))
            score = self.objective_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
                
        return best_params, best_score
    
    def hill_climbing(self, n_steps: int, step_size: float = 0.1) -> tuple:
        """Hill climbing strategy"""
        # Start from random point
        current_params = {
            name: np.random.uniform(low, high)
            for name, (low, high) in self.param_space.items()
        }
        current_score = self.objective_func(current_params)
        
        for _ in range(n_steps):
            # Generate neighbor
            neighbor_params = {
                name: value + np.random.normal(0, step_size)
                for name, value in current_params.items()
            }
            
            # Clip to bounds
            for name, value in neighbor_params.items():
                low, high = self.param_space[name]
                neighbor_params[name] = np.clip(value, low, high)
                
            neighbor_score = self.objective_func(neighbor_params)
            
            # Move if better
            if neighbor_score > current_score:
                current_score = neighbor_score
                current_params = neighbor_params
                
        return current_params, current_score
    
    def optimize(self) -> Dict:
        """Main optimization loop"""
        strategies = [
            (self.random_search, self.n_iterations // 3),
            (self.grid_search, int(np.cbrt(self.n_iterations))),
            (self.hill_climbing, self.n_iterations // 3)
        ]
        
        results = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for strategy, n_trials in strategies:
                for _ in range(self.ensemble_size):
                    futures.append(
                        executor.submit(strategy, n_trials)
                    )
            
            for future in futures:
                params, score = future.result()
                results.append((params, score))
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    
        return self.best_params

# Example usage
def objective_function(params):
    """Example objective function"""
    x, y = params['x'], params['y']
    return -(x**2 + (y-1)**2) + np.random.normal(0, 0.1)

param_space = {
    'x': (-5, 5),
    'y': (-5, 5)
}

optimizer = EnsembleOptimizer(
    objective_function,
    param_space,
    n_iterations=100,
    ensemble_size=3
)

best_params = optimizer.optimize()
print(f"Best parameters found: {best_params}")
print(f"Best score: {optimizer.best_score:.4f}")
```

Slide 14: Additional Resources

*   ArXiv Paper: "Automated Machine Learning: A Review and Analysis of the State-of-the-Art" - [https://arxiv.org/abs/2011.01538](https://arxiv.org/abs/2011.01538)
*   ArXiv Paper: "Population Based Training of Neural Networks" - [https://arxiv.org/abs/1711.09846](https://arxiv.org/abs/1711.09846)
*   ArXiv Paper: "A Tutorial on Bayesian Optimization" - [https://arxiv.org/abs/1807.02811](https://arxiv.org/abs/1807.02811)
*   Google Research Blog: "Improving Deep Learning Performance with AutoML" - [https://research.google/pubs/pub46180/](https://research.google/pubs/pub46180/)
*   Microsoft Research: "Efficient and Robust Automated Machine Learning" - [https://www.microsoft.com/en-us/research/publication/efficient-and-robust-automated-machine-learning/](https://www.microsoft.com/en-us/research/publication/efficient-and-robust-automated-machine-learning/)

