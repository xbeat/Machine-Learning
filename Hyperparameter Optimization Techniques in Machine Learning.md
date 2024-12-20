## Hyperparameter Optimization Techniques in Machine Learning
Slide 1: Grid Search Cross-Validation

Grid search systematically works through multiple combinations of parameter tunes, cross-validates each combination, and helps identify which combination produces the optimal model. This brute-force approach evaluates every possible combination of hyperparameter values specified.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

# Generate sample data
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
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X, y)

# Best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

Slide 2: Random Search Optimization

Random search tests a random combination of hyperparameters from a defined distribution. This method often finds optimal parameters more efficiently than grid search, especially when not all hyperparameters are equally important.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(range(5, 30)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Initialize model
rf = RandomForestClassifier()

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
random_search.fit(X, y)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

Slide 3: Bayesian Optimization

Bayesian optimization uses probabilistic models to guide the search for optimal hyperparameters. It builds a surrogate model of the objective function and uses an acquisition function to determine the next set of parameters to evaluate.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.neural_network import MLPClassifier

# Define search space
search_space = {
    'hidden_layer_sizes': Integer(50, 200),
    'learning_rate_init': Real(10**-4, 10**-1, prior='log-uniform'),
    'max_iter': Integer(100, 500),
    'alpha': Real(10**-5, 10**-1, prior='log-uniform')
}

# Initialize model
mlp = MLPClassifier()

# Setup BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=mlp,
    search_spaces=search_space,
    n_iter=50,
    cv=5,
    n_jobs=-1
)

# Fit the model
bayes_search.fit(X, y)

print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best score: {bayes_search.best_score_:.3f}")
```

Slide 4: Hyperparameter Optimization with Optuna

Optuna is a hyperparameter optimization framework that uses modern approaches like pruning unpromising trials and parallelization. It efficiently explores the parameter space using various sampling strategies.

```python
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

def objective(trial):
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    
    # Create model with current hyperparameters
    model = XGBClassifier(**params)
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    return scores.mean()

# Create study object and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.3f}")
```

Slide 5: Population-Based Training (PBT)

Population-Based Training combines random search with evolutionary optimization, maintaining a population of models that train in parallel. Models periodically adapt their hyperparameters by copying better performing peers, enabling dynamic optimization during training.

```python
import numpy as np
from typing import Dict, List
import copy

class PBTOptimizer:
    def __init__(self, population_size: int, exploit_threshold: float = 0.2):
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.population: List[Dict] = []
        
    def initialize_population(self, param_ranges: Dict):
        for _ in range(self.population_size):
            params = {
                key: np.random.uniform(ranges[0], ranges[1])
                for key, ranges in param_ranges.items()
            }
            self.population.append({
                'params': params,
                'score': 0.0
            })
    
    def exploit_and_explore(self, member_idx: int):
        # Sort population by score
        sorted_pop = sorted(self.population, key=lambda x: x['score'], reverse=True)
        
        if np.random.random() < self.exploit_threshold:
            # Copy parameters from better performing member
            better_member = sorted_pop[0]
            self.population[member_idx]['params'] = copy.deepcopy(better_member['params'])
            
            # Perturb parameters
            for param in self.population[member_idx]['params']:
                if np.random.random() < 0.2:
                    self.population[member_idx]['params'][param] *= np.random.choice([0.8, 1.2])

# Example usage
param_ranges = {
    'learning_rate': [0.0001, 0.1],
    'batch_size': [16, 256],
    'num_layers': [1, 5]
}

pbt = PBTOptimizer(population_size=10)
pbt.initialize_population(param_ranges)

# Simulate training for 10 steps
for step in range(10):
    # Update scores (in real scenario, this would be validation performance)
    for i in range(pbt.population_size):
        pbt.population[i]['score'] = np.random.random()
    
    # Perform exploitation and exploration
    for i in range(pbt.population_size):
        pbt.exploit_and_explore(i)
```

Slide 6: Hyperband Optimization

Hyperband is an optimization algorithm that adaptively allocates resources to configurations and uses successive halving to rapidly eliminate poor performers. It balances exploration of hyperparameters and exploitation of promising configurations.

```python
import math
from typing import Callable

class Hyperband:
    def __init__(self, max_iter: int, eta: int = 3):
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(math.log(max_iter) / math.log(eta))
        self.B = (self.s_max + 1) * max_iter

    def run_optimization(self, get_params_fn: Callable, evaluate_fn: Callable):
        for s in reversed(range(self.s_max + 1)):
            # Initial number of configurations
            n = int(math.ceil(int(self.B/self.max_iter/(s+1)) * self.eta**s))
            
            # Initial resource allocation
            r = self.max_iter * self.eta**(-s)

            # Generate configurations
            configs = [get_params_fn() for _ in range(n)]
            
            for i in range(s + 1):
                # Number of iterations for each config
                n_i = n * self.eta**(-i)
                r_i = r * self.eta**(i)
                
                # Evaluate configurations
                scores = []
                for config in configs:
                    score = evaluate_fn(config, r_i)
                    scores.append((config, score))
                
                # Select top configurations
                scores = sorted(scores, key=lambda x: x[1], reverse=True)
                n_survivors = int(n_i / self.eta)
                configs = [config for config, _ in scores[:n_survivors]]

        return configs[0]  # Return best configuration

# Example usage
def random_params():
    return {
        'learning_rate': np.random.uniform(0.0001, 0.1),
        'num_layers': np.random.randint(1, 5),
        'hidden_size': np.random.choice([32, 64, 128, 256])
    }

def evaluate_config(config, num_iters):
    # Simulate model training and return validation score
    return np.random.random()

hb = Hyperband(max_iter=81)  # max_iter should be divisible by eta^k for some integer k
best_config = hb.run_optimization(random_params, evaluate_config)
print(f"Best configuration found: {best_config}")
```

Slide 7: TPE (Tree-structured Parzen Estimators)

Tree-structured Parzen Estimators construct probabilistic models of the relationship between hyperparameters and their performance. It uses kernel density estimation to model the distribution of hyperparameters that lead to good and bad performance.

```python
import numpy as np
from scipy import stats
from typing import List, Tuple

class TPEOptimizer:
    def __init__(self, gamma: float = 0.15):
        self.gamma = gamma
        self.trials: List[Tuple] = []
    
    def suggest(self, param_range: Tuple[float, float]) -> float:
        if len(self.trials) < 10:
            return np.random.uniform(param_range[0], param_range[1])
        
        # Sort trials by score
        sorted_trials = sorted(self.trials, key=lambda x: x[1])
        n_good = max(1, int(self.gamma * len(sorted_trials)))
        
        # Split into good and bad trials
        good_params = [t[0] for t in sorted_trials[-n_good:]]
        bad_params = [t[0] for t in sorted_trials[:-n_good]]
        
        # Fit KDE to good and bad parameters
        kde_good = stats.gaussian_kde(good_params)
        kde_bad = stats.gaussian_kde(bad_params)
        
        # Generate candidates and compute EI
        candidates = np.linspace(param_range[0], param_range[1], 100)
        ei = kde_good(candidates) / (kde_bad(candidates) + 1e-6)
        
        return candidates[np.argmax(ei)]
    
    def update(self, param: float, score: float):
        self.trials.append((param, score))

# Example usage
tpe = TPEOptimizer()
param_range = (0.0001, 0.1)

# Simulate optimization process
for _ in range(50):
    # Get suggestion
    param = tpe.suggest(param_range)
    
    # Simulate evaluation (in real scenario, this would be model training)
    score = -(param - 0.01)**2 + np.random.normal(0, 0.01)
    
    # Update optimizer
    tpe.update(param, score)

# Get best trial
best_trial = max(tpe.trials, key=lambda x: x[1])
print(f"Best parameter: {best_trial[0]:.6f}, Score: {best_trial[1]:.6f}")
```

Slide 8: Custom Multi-Objective Optimization

Multi-objective optimization handles multiple competing objectives simultaneously, finding Pareto-optimal solutions. This implementation uses NSGA-II algorithm principles for hyperparameter optimization considering both model performance and computational cost.

```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Solution:
    params: dict
    objectives: List[float]
    crowding_distance: float = 0.0
    rank: int = 0

class MultiObjectiveOptimizer:
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.population: List[Solution] = []
    
    def dominate(self, sol1: Solution, sol2: Solution) -> bool:
        better_in_any = False
        for obj1, obj2 in zip(sol1.objectives, sol2.objectives):
            if obj1 > obj2:
                return False
            if obj1 < obj2:
                better_in_any = True
        return better_in_any
    
    def fast_non_dominated_sort(self):
        fronts = [[]]
        for p in self.population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in self.population:
                if self.dominate(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominate(q, p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]
    
    def calculate_crowding_distance(self, front: List[Solution]):
        if len(front) <= 2:
            for solution in front:
                solution.crowding_distance = float('inf')
            return
        
        for solution in front:
            solution.crowding_distance = 0
        
        num_objectives = len(front[0].objectives)
        
        for m in range(num_objectives):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue
                
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / obj_range

# Example usage
def evaluate_model(params):
    # Simulate model evaluation returning accuracy and training time
    accuracy = np.random.normal(0.8, 0.1)
    training_time = params['num_layers'] * params['hidden_size'] * 0.1
    return [accuracy, -training_time]  # Negative training time for minimization

optimizer = MultiObjectiveOptimizer(population_size=50)

# Generate initial population
for _ in range(optimizer.population_size):
    params = {
        'num_layers': np.random.randint(1, 5),
        'hidden_size': np.random.choice([32, 64, 128, 256]),
        'learning_rate': np.random.uniform(0.0001, 0.1)
    }
    objectives = evaluate_model(params)
    optimizer.population.append(Solution(params=params, objectives=objectives))

# Perform non-dominated sorting
fronts = optimizer.fast_non_dominated_sort()

# Calculate crowding distance for each front
for front in fronts:
    optimizer.calculate_crowding_distance(front)

# Get Pareto-optimal solutions (first front)
pareto_optimal = fronts[0]
print("Pareto-optimal solutions:")
for solution in pareto_optimal:
    print(f"Params: {solution.params}")
    print(f"Objectives: {solution.objectives}\n")
```

Slide 9: Early Stopping with Validation Performance

Early stopping is a regularization method that monitors validation performance during training to prevent overfitting. This implementation includes dynamic thresholding and patience mechanisms for robust hyperparameter optimization.

```python
class EarlyStoppingOptimizer:
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.reset()
        
    def reset(self):
        self.best_weights = None
        self.best_epoch = 0
        self.best_score = float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def is_improvement(self, score: float) -> bool:
        return score > (self.best_score + self.min_delta)
    
    def update(self, score: float, weights: dict, epoch: int) -> bool:
        if self.is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = weights.copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

# Example usage with hyperparameter optimization
def train_with_early_stopping(params, X_train, y_train, X_val, y_val):
    model = create_model(params)
    early_stopping = EarlyStoppingOptimizer(patience=5)
    
    best_score = float('-inf')
    for epoch in range(100):
        # Simulate training
        train_score = np.random.normal(0.8, 0.1)
        val_score = np.random.normal(0.75, 0.1)
        
        # Get current weights (simulated)
        current_weights = {'layer1': np.random.randn(10, 10)}
        
        # Check early stopping
        if early_stopping.update(val_score, current_weights, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        best_score = max(best_score, val_score)
    
    return best_score, early_stopping.best_weights

# Simulate optimization
param_configs = [
    {'learning_rate': 0.01, 'num_layers': 2},
    {'learning_rate': 0.001, 'num_layers': 3},
    {'learning_rate': 0.0001, 'num_layers': 4}
]

results = []
for params in param_configs:
    score, best_weights = train_with_early_stopping(
        params,
        X_train=np.random.randn(100, 10),
        y_train=np.random.randint(0, 2, 100),
        X_val=np.random.randn(50, 10),
        y_val=np.random.randint(0, 2, 50)
    )
    results.append((params, score))

best_params = max(results, key=lambda x: x[1])[0]
print(f"Best hyperparameters: {best_params}")
```

Slide 10: Evolutionary Hyperparameter Optimization

Evolutionary optimization applies genetic algorithms to hyperparameter tuning, using mutation, crossover, and selection operations to evolve optimal configurations across generations, mimicking natural selection processes.

```python
import numpy as np
from typing import List, Dict, Tuple
import random

class EvolutionaryOptimizer:
    def __init__(self, population_size: int, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_fitness_history = []
        
    def initialize_population(self, param_ranges: Dict) -> List[Dict]:
        population = []
        for _ in range(self.population_size):
            individual = {
                param: np.random.uniform(ranges[0], ranges[1])
                if isinstance(ranges[0], float)
                else np.random.randint(ranges[0], ranges[1])
                for param, ranges in param_ranges.items()
            }
            population.append(individual)
        return population
    
    def mutate(self, individual: Dict, param_ranges: Dict) -> Dict:
        mutated = individual.copy()
        for param in mutated:
            if random.random() < self.mutation_rate:
                ranges = param_ranges[param]
                if isinstance(ranges[0], float):
                    mutated[param] = np.random.uniform(ranges[0], ranges[1])
                else:
                    mutated[param] = np.random.randint(ranges[0], ranges[1])
        return mutated
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        child1, child2 = parent1.copy(), parent2.copy()
        for param in parent1:
            if random.random() < 0.5:
                child1[param], child2[param] = child2[param], child1[param]
        return child1, child2
    
    def select_parents(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        probs = np.array(fitness_scores) / sum(fitness_scores)
        selected_indices = np.random.choice(
            len(population),
            size=len(population),
            p=probs
        )
        return [population[idx] for idx in selected_indices]
    
    def evolve(self, population: List[Dict], fitness_scores: List[float],
               param_ranges: Dict) -> List[Dict]:
        # Select parents
        parents = self.select_parents(population, fitness_scores)
        
        # Create new population through crossover and mutation
        new_population = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1, child2 = self.crossover(parents[i], parents[i + 1])
                new_population.extend([
                    self.mutate(child1, param_ranges),
                    self.mutate(child2, param_ranges)
                ])
            else:
                new_population.append(self.mutate(parents[i], param_ranges))
                
        return new_population[:self.population_size]

# Example usage
def evaluate_model(params: Dict) -> float:
    # Simulate model training and return accuracy
    base_accuracy = 0.8
    penalty = 0.1 * (params['learning_rate'] - 0.01)**2
    noise = np.random.normal(0, 0.05)
    return base_accuracy - penalty + noise

# Define parameter ranges
param_ranges = {
    'learning_rate': (0.0001, 0.1),
    'hidden_size': (32, 256),
    'num_layers': (1, 5)
}

# Initialize optimizer
optimizer = EvolutionaryOptimizer(population_size=20)

# Initial population
population = optimizer.initialize_population(param_ranges)

# Evolution loop
n_generations = 10
for generation in range(n_generations):
    # Evaluate current population
    fitness_scores = [evaluate_model(params) for params in population]
    
    # Track best fitness
    best_fitness = max(fitness_scores)
    optimizer.best_fitness_history.append(best_fitness)
    
    # Evolve population
    population = optimizer.evolve(population, fitness_scores, param_ranges)
    
    print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}")

# Get best configuration
best_idx = np.argmax([evaluate_model(params) for params in population])
best_params = population[best_idx]
print(f"\nBest parameters found: {best_params}")
```

Slide 11: Hyperparameter Optimization with Pruning

Pruning eliminates underperforming hyperparameter configurations early in the training process, allowing more efficient exploration of the parameter space by reallocating computational resources to promising configurations.

```python
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
import time

@dataclass
class TrialHistory:
    params: Dict
    scores: List[float]
    epochs: List[int]
    resources_used: float

class PruningOptimizer:
    def __init__(self, max_epochs: int, evaluation_points: int = 5,
                 pruning_threshold: float = 0.5):
        self.max_epochs = max_epochs
        self.evaluation_points = evaluation_points
        self.pruning_threshold = pruning_threshold
        self.trials: List[TrialHistory] = []
        
    def should_prune(self, current_score: float, epoch: int) -> bool:
        if len(self.trials) < 5:  # Need some history for comparison
            return False
            
        # Get scores from previous trials at similar epochs
        comparable_scores = []
        for trial in self.trials:
            if epoch in trial.epochs:
                idx = trial.epochs.index(epoch)
                comparable_scores.append(trial.scores[idx])
                
        if not comparable_scores:
            return False
            
        # Compare current score against historical performance
        score_threshold = np.percentile(comparable_scores, 
                                      self.pruning_threshold * 100)
        return current_score < score_threshold
    
    def run_trial(self, params: Dict) -> Optional[float]:
        history = TrialHistory(params=params, scores=[], epochs=[], resources_used=0)
        start_time = time.time()
        
        # Define evaluation points
        eval_epochs = np.linspace(1, self.max_epochs, 
                                self.evaluation_points, dtype=int)
        
        for epoch in range(1, self.max_epochs + 1):
            # Simulate training and get validation score
            score = (1 - np.exp(-0.1 * epoch)) * (0.9 + 0.1 * np.random.random())
            
            if epoch in eval_epochs:
                history.epochs.append(epoch)
                history.scores.append(score)
                
                if self.should_prune(score, epoch):
                    print(f"Pruning trial at epoch {epoch}")
                    break
        
        history.resources_used = time.time() - start_time
        self.trials.append(history)
        
        return max(history.scores) if history.scores else None

# Example usage
param_configurations = [
    {'learning_rate': 0.1, 'batch_size': 32},
    {'learning_rate': 0.01, 'batch_size': 64},
    {'learning_rate': 0.001, 'batch_size': 128}
]

optimizer = PruningOptimizer(max_epochs=20)
results = []

for params in param_configurations:
    score = optimizer.run_trial(params)
    if score is not None:
        results.append((params, score))
        print(f"Configuration {params}: Score = {score:.4f}")

best_config = max(results, key=lambda x: x[1])
print(f"\nBest configuration: {best_config[0]}")
print(f"Best score: {best_config[1]:.4f}")
```

Slide 12: Parallel Asynchronous Hyperparameter Optimization

Asynchronous parallel optimization enables efficient utilization of multiple computational resources by running multiple trials simultaneously, using a shared optimization strategy to guide parameter selection based on completed trials.

```python
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple
import queue
import threading

class AsyncParallelOptimizer:
    def __init__(self, n_workers: int = 4):
        self.n_workers = n_workers
        self.results_queue = queue.Queue()
        self.pending_trials = []
        self.completed_trials = []
        self.lock = threading.Lock()
        
    def generate_params(self, param_ranges: Dict) -> Dict:
        with self.lock:
            # Use completed trials to inform parameter generation
            if len(self.completed_trials) > 0:
                best_trial = max(self.completed_trials, key=lambda x: x[1])
                best_params = best_trial[0]
                
                # Generate parameters with local search around best
                new_params = {}
                for param, ranges in param_ranges.items():
                    if isinstance(ranges[0], float):
                        perturbation = np.random.normal(0, 0.1)
                        new_value = best_params[param] * (1 + perturbation)
                        new_params[param] = np.clip(new_value, ranges[0], ranges[1])
                    else:
                        new_params[param] = np.random.randint(ranges[0], ranges[1])
                return new_params
            
            # Random initialization if no completed trials
            return {
                param: np.random.uniform(ranges[0], ranges[1])
                if isinstance(ranges[0], float)
                else np.random.randint(ranges[0], ranges[1])
                for param, ranges in param_ranges.items()
            }
    
    def evaluate_model(self, params: Dict) -> float:
        # Simulate model training with random duration
        time.sleep(np.random.uniform(0.1, 0.5))
        
        # Simulate performance score
        base_score = 0.8
        penalty = sum((v - 0.5)**2 for v in params.values())
        noise = np.random.normal(0, 0.05)
        return base_score - penalty + noise
    
    def worker_process(self, params: Dict) -> Tuple[Dict, float]:
        score = self.evaluate_model(params)
        return params, score
    
    def optimization_loop(self, param_ranges: Dict, n_trials: int):
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            
            # Initial batch of trials
            for _ in range(self.n_workers):
                if len(futures) < n_trials:
                    params = self.generate_params(param_ranges)
                    future = executor.submit(self.worker_process, params)
                    futures.append(future)
            
            # Process results and submit new trials
            completed = 0
            while completed < n_trials:
                for i, future in enumerate(futures):
                    if future.done():
                        params, score = future.result()
                        with self.lock:
                            self.completed_trials.append((params, score))
                            print(f"Trial completed: score = {score:.4f}")
                        
                        if len(self.completed_trials) < n_trials:
                            params = self.generate_params(param_ranges)
                            futures[i] = executor.submit(
                                self.worker_process, params
                            )
                        completed += 1

# Example usage
param_ranges = {
    'learning_rate': (0.0001, 0.1),
    'hidden_size': (32, 256),
    'dropout_rate': (0.1, 0.5)
}

optimizer = AsyncParallelOptimizer(n_workers=4)
optimizer.optimization_loop(param_ranges, n_trials=20)

# Get best configuration
best_trial = max(optimizer.completed_trials, key=lambda x: x[1])
print("\nOptimization Results:")
print(f"Best parameters: {best_trial[0]}")
print(f"Best score: {best_trial[1]:.4f}")

# Print optimization statistics
scores = [trial[1] for trial in optimizer.completed_trials]
print(f"\nMean score: {np.mean(scores):.4f}")
print(f"Std score: {np.std(scores):.4f}")
```

Slide 13: Real-world Application: NLP Model Optimization

Implementation of hyperparameter optimization for a real NLP task, demonstrating the complete workflow from data preprocessing to final model selection with practical considerations for computational efficiency.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
import time

class NLPHyperOptimizer:
    def __init__(self, X: List[str], y: np.ndarray):
        self.X = X
        self.y = y
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    def preprocess_data(self, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        vectorizer = TfidfVectorizer(
            max_features=params['max_features'],
            ngram_range=(1, params['max_ngram']),
            min_df=params['min_df']
        )
        
        X_train_vec = vectorizer.fit_transform(self.X_train)
        X_val_vec = vectorizer.transform(self.X_val)
        
        return X_train_vec, X_val_vec
    
    def train_evaluate_model(self, params: Dict) -> Dict:
        start_time = time.time()
        
        # Preprocess data
        X_train_vec, X_val_vec = self.preprocess_data(params)
        
        # Initialize model (example with simple neural network)
        model = MLPClassifier(
            hidden_layer_sizes=(params['hidden_size'],),
            learning_rate_init=params['learning_rate'],
            max_iter=params['max_iter'],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5
        )
        
        # Train model
        model.fit(X_train_vec, self.y_train)
        
        # Evaluate
        train_score = model.score(X_train_vec, self.y_train)
        val_score = model.score(X_val_vec, self.y_val)
        
        training_time = time.time() - start_time
        
        return {
            'params': params,
            'train_score': train_score,
            'val_score': val_score,
            'training_time': training_time,
            'n_epochs': model.n_iter_
        }
    
    def optimize(self, n_trials: int = 20) -> List[Dict]:
        results = []
        
        for trial in range(n_trials):
            # Generate parameters
            params = {
                'max_features': np.random.randint(1000, 10000),
                'max_ngram': np.random.randint(1, 4),
                'min_df': np.random.choice([1, 2, 3, 5]),
                'hidden_size': np.random.choice([32, 64, 128, 256]),
                'learning_rate': np.random.uniform(0.0001, 0.01),
                'max_iter': 100
            }
            
            # Train and evaluate
            result = self.train_evaluate_model(params)
            results.append(result)
            
            print(f"\nTrial {trial + 1}/{n_trials}")
            print(f"Validation Score: {result['val_score']:.4f}")
            print(f"Training Time: {result['training_time']:.2f}s")
        
        return results

# Example usage with synthetic data
np.random.seed(42)
X = [
    f"Sample text document {i}" for i in range(1000)
]
y = np.random.randint(0, 2, 1000)

optimizer = NLPHyperOptimizer(X, y)
results = optimizer.optimize(n_trials=5)

# Find best configuration
best_result = max(results, key=lambda x: x['val_score'])
print("\nBest Configuration:")
print(f"Parameters: {best_result['params']}")
print(f"Validation Score: {best_result['val_score']:.4f}")
print(f"Training Time: {best_result['training_time']:.2f}s")
```

Slide 14: Computer Vision Model Optimization

Real-world implementation of hyperparameter optimization for a computer vision task, incorporating transfer learning considerations and demonstrating efficient parameter search strategies for deep convolutional networks.

```python
import numpy as np
from typing import Dict, List, Tuple
import time

class CVHyperOptimizer:
    def __init__(self, train_data: Tuple[np.ndarray, np.ndarray],
                 val_data: Tuple[np.ndarray, np.ndarray]):
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = val_data
        self.best_model = None
        self.best_score = float('-inf')
    
    def create_model(self, params: Dict) -> Dict:
        # Simulate model creation with transfer learning
        base_model_options = {
            'resnet50': {'features': 2048, 'complexity': 1.0},
            'efficientnet': {'features': 1280, 'complexity': 0.8},
            'mobilenet': {'features': 1024, 'complexity': 0.5}
        }
        
        base_architecture = params['base_model']
        base_complexity = base_model_options[base_architecture]['complexity']
        
        # Calculate approximate training time based on model complexity
        training_time = (base_complexity * 
                        params['epochs'] * 
                        params['batch_size'] / 32)
        
        return {
            'architecture': base_architecture,
            'training_time': training_time,
            'params': params
        }
    
    def train_evaluate(self, model_config: Dict) -> Dict:
        # Simulate training and evaluation
        base_accuracy = 0.85
        
        # Adjust accuracy based on parameters
        params = model_config['params']
        accuracy_modifiers = {
            'learning_rate': lambda x: -0.1 * (np.log10(x) + 3)**2,
            'batch_size': lambda x: -0.05 * (np.log2(x/32))**2,
            'dropout': lambda x: -0.1 * (x - 0.5)**2
        }
        
        accuracy_adjustment = sum(
            modifier(params[param])
            for param, modifier in accuracy_modifiers.items()
        )
        
        # Add noise to simulate training variance
        noise = np.random.normal(0, 0.02)
        final_accuracy = base_accuracy + accuracy_adjustment + noise
        
        return {
            'model_config': model_config,
            'val_accuracy': min(max(final_accuracy, 0), 1),
            'training_time': model_config['training_time']
        }
    
    def optimize(self, n_trials: int = 20) -> List[Dict]:
        results = []
        param_ranges = {
            'base_model': ['resnet50', 'efficientnet', 'mobilenet'],
            'learning_rate': (1e-4, 1e-2),
            'batch_size': [16, 32, 64, 128],
            'dropout': (0.1, 0.5),
            'epochs': [30, 50, 100]
        }
        
        for trial in range(n_trials):
            # Generate parameters with adaptive sampling
            params = {
                'base_model': np.random.choice(param_ranges['base_model']),
                'learning_rate': np.exp(np.random.uniform(
                    np.log(param_ranges['learning_rate'][0]),
                    np.log(param_ranges['learning_rate'][1])
                )),
                'batch_size': np.random.choice(param_ranges['batch_size']),
                'dropout': np.random.uniform(*param_ranges['dropout']),
                'epochs': np.random.choice(param_ranges['epochs'])
            }
            
            # Create and evaluate model
            model_config = self.create_model(params)
            result = self.train_evaluate(model_config)
            results.append(result)
            
            # Update best model
            if result['val_accuracy'] > self.best_score:
                self.best_score = result['val_accuracy']
                self.best_model = model_config
            
            print(f"\nTrial {trial + 1}/{n_trials}")
            print(f"Model: {params['base_model']}")
            print(f"Accuracy: {result['val_accuracy']:.4f}")
            print(f"Training Time: {result['training_time']:.2f}h")
        
        return results

# Example usage with synthetic data
np.random.seed(42)

# Simulate image dataset
X_train = np.random.randn(1000, 224, 224, 3)
y_train = np.random.randint(0, 10, 1000)
X_val = np.random.randn(200, 224, 224, 3)
y_val = np.random.randint(0, 10, 200)

optimizer = CVHyperOptimizer(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)

results = optimizer.optimize(n_trials=5)

# Analyze results
best_trial = max(results, key=lambda x: x['val_accuracy'])
print("\nOptimization Results:")
print(f"Best Architecture: {best_trial['model_config']['architecture']}")
print(f"Best Parameters: {best_trial['model_config']['params']}")
print(f"Validation Accuracy: {best_trial['val_accuracy']:.4f}")
print(f"Training Time: {best_trial['training_time']:.2f}h")
```

Slide 15: Additional Resources

*   "Population Based Training of Neural Networks" [https://arxiv.org/abs/1711.09846](https://arxiv.org/abs/1711.09846)
*   "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" [https://arxiv.org/abs/1603.06560](https://arxiv.org/abs/1603.06560)
*   "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" [https://arxiv.org/abs/1807.01774](https://arxiv.org/abs/1807.01774)
*   "Automating the Art of Machine Learning: A Survey of AutoML Methods" [https://arxiv.org/abs/1908.08770](https://arxiv.org/abs/1908.08770)
*   "Neural Architecture Search: A Survey" [https://arxiv.org/abs/1808.05377](https://arxiv.org/abs/1808.05377)

