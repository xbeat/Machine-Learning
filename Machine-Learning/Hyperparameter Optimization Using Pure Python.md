## Hyperparameter Optimization Using Pure Python

Slide 1: Introduction to Hyperparameter Optimization

Hyperparameter optimization is a crucial step in machine learning to enhance model performance. It involves fine-tuning the settings that govern the learning process. By selecting the optimal set of hyperparameters along with quality data, we can achieve the best possible model performance.

```python
import random

def simple_model(x, a, b):
    return a * x + b

def evaluate_model(model, data, a, b):
    errors = [(y - model(x, a, b))**2 for x, y in data]
    return sum(errors) / len(errors)

# Generate some sample data
data = [(x, 2*x + 1 + random.uniform(-0.5, 0.5)) for x in range(100)]

# Simple grid search for hyperparameters
best_error = float('inf')
best_params = None

for a in [1.5, 2.0, 2.5]:
    for b in [0.5, 1.0, 1.5]:
        error = evaluate_model(simple_model, data, a, b)
        if error < best_error:
            best_error = error
            best_params = (a, b)

print(f"Best parameters: a={best_params[0]}, b={best_params[1]}")
print(f"Best error: {best_error}")
```

Slide 2: Grid Search

Grid search is a systematic approach to hyperparameter optimization. It involves defining a grid of hyperparameter values and evaluating the model's performance for each combination in the grid.

```python
import itertools

def grid_search(param_grid, model, data):
    best_params = None
    best_score = float('inf')
    
    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*param_grid.values()))
    
    for params in param_combinations:
        # Create a dictionary of parameter names and values
        current_params = dict(zip(param_grid.keys(), params))
        
        # Evaluate the model with current parameters
        score = evaluate_model(model, data, **current_params)
        
        # Update best parameters if current score is better
        if score < best_score:
            best_score = score
            best_params = current_params
    
    return best_params, best_score

# Define parameter grid
param_grid = {
    'a': [1.5, 2.0, 2.5],
    'b': [0.5, 1.0, 1.5]
}

best_params, best_score = grid_search(param_grid, simple_model, data)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 3: Random Search

Random search is a hyperparameter optimization technique that selects hyperparameter combinations randomly from the parameter space. It can be more efficient than grid search, especially when some parameters are more important than others.

```python
import random

def random_search(param_distributions, model, data, n_iter=10):
    best_params = None
    best_score = float('inf')
    
    for _ in range(n_iter):
        # Sample random parameters
        current_params = {
            param: random.choice(values) if isinstance(values, list) else random.uniform(*values)
            for param, values in param_distributions.items()
        }
        
        # Evaluate the model with current parameters
        score = evaluate_model(model, data, **current_params)
        
        # Update best parameters if current score is better
        if score < best_score:
            best_score = score
            best_params = current_params
    
    return best_params, best_score

# Define parameter distributions
param_distributions = {
    'a': [1.5, 2.0, 2.5],
    'b': (0.0, 2.0)  # Uniform distribution between 0 and 2
}

best_params, best_score = random_search(param_distributions, simple_model, data, n_iter=20)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 4: Bayesian Optimization

Bayesian optimization is a probabilistic model-based approach to hyperparameter optimization. It uses past evaluation results to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function.

```python
import math
import random

class GaussianProcess:
    def __init__(self, kernel, noise=1e-5):
        self.kernel = kernel
        self.noise = noise
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_new):
        K = self.kernel(self.X, self.X)
        K += self.noise * np.eye(len(self.X))
        K_new = self.kernel(self.X, X_new)
        K_inv = np.linalg.inv(K)
        
        mu = K_new.T.dot(K_inv).dot(self.y)
        sigma = self.kernel(X_new, X_new) - K_new.T.dot(K_inv).dot(K_new)
        return mu, np.diag(sigma)

def rbf_kernel(X1, X2, l=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 / l**2 * sqdist)

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X)
    mu_sample = gpr.predict(X_sample)[0]

    sigma = sigma.reshape(-1, 1)
    mu_sample_opt = np.max(mu_sample)
    
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def bayesian_optimization(model, data, param_space, n_iter=10):
    X_sample = np.array([list(param_space.keys()) for _ in range(5)])
    Y_sample = np.array([evaluate_model(model, data, **params) for params in X_sample])
    
    gpr = GaussianProcess(rbf_kernel)
    
    for i in range(n_iter):
        gpr.fit(X_sample, Y_sample)
        
        X = np.array([list(param_space.keys()) for _ in range(10000)])
        ei = expected_improvement(X, X_sample, Y_sample, gpr)
        
        next_point = X[np.argmax(ei)]
        next_value = evaluate_model(model, data, **dict(zip(param_space.keys(), next_point)))
        
        X_sample = np.vstack((X_sample, next_point))
        Y_sample = np.append(Y_sample, next_value)
    
    best_idx = np.argmin(Y_sample)
    return dict(zip(param_space.keys(), X_sample[best_idx])), Y_sample[best_idx]

param_space = {
    'a': (1.0, 3.0),
    'b': (0.0, 2.0)
}

best_params, best_score = bayesian_optimization(simple_model, data, param_space)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 5: Hyperband

Hyperband is an optimization technique that efficiently allocates computational resources by combining random search, bandit-based methods, and successive halving. It's particularly useful when dealing with computationally expensive models.

```python
import math
import random

def hyperband(get_params, evaluate, max_iter, eta=3, max_budget=None):
    if max_budget is None:
        max_budget = max_iter
    
    s_max = math.floor(math.log(max_budget, eta))
    B = (s_max + 1) * max_budget

    for s in reversed(range(s_max + 1)):
        n = math.ceil(int(B / max_budget / (s + 1)) * eta**s)
        r = max_budget * eta**(-s)

        T = [get_params() for _ in range(n)]
        
        for i in range(s + 1):
            n_i = n * eta**(-i)
            r_i = r * eta**i

            L = [evaluate(t, int(r_i)) for t in T]

            T = [T[i] for i in sorted(range(len(L)), key=lambda x: L[x])[:int(n_i / eta)]]

    return T[0]

# Example usage
def get_random_params():
    return {'a': random.uniform(1.0, 3.0), 'b': random.uniform(0.0, 2.0)}

def evaluate_with_iterations(params, iterations):
    # Simulate a more complex model where iterations affect performance
    error = evaluate_model(simple_model, data[:iterations], **params)
    return error

best_params = hyperband(get_random_params, evaluate_with_iterations, max_iter=100)
print(f"Best parameters: {best_params}")
print(f"Best score: {evaluate_model(simple_model, data, **best_params)}")
```

Slide 6: Genetic Algorithm

Genetic algorithms are optimization techniques inspired by natural selection. They represent hyperparameter sets as individuals in a population and evolve them over generations to find optimal solutions.

```python
import random

def initialize_population(pop_size, param_ranges):
    return [
        {param: random.uniform(range_[0], range_[1]) for param, range_ in param_ranges.items()}
        for _ in range(pop_size)
    ]

def crossover(parent1, parent2):
    child = {}
    for param in parent1:
        if random.random() < 0.5:
            child[param] = parent1[param]
        else:
            child[param] = parent2[param]
    return child

def mutate(individual, param_ranges, mutation_rate):
    for param in individual:
        if random.random() < mutation_rate:
            individual[param] = random.uniform(param_ranges[param][0], param_ranges[param][1])
    return individual

def genetic_algorithm(model, data, param_ranges, pop_size=50, generations=10, mutation_rate=0.1):
    population = initialize_population(pop_size, param_ranges)
    
    for _ in range(generations):
        # Evaluate fitness
        fitness = [1 / evaluate_model(model, data, **ind) for ind in population]
        
        # Select parents
        parents = random.choices(population, weights=fitness, k=pop_size)
        
        # Create next generation
        next_generation = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1 = mutate(crossover(parent1, parent2), param_ranges, mutation_rate)
            child2 = mutate(crossover(parent1, parent2), param_ranges, mutation_rate)
            next_generation.extend([child1, child2])
        
        population = next_generation
    
    # Find best individual
    best_individual = max(population, key=lambda ind: 1 / evaluate_model(model, data, **ind))
    return best_individual, evaluate_model(model, data, **best_individual)

param_ranges = {
    'a': (1.0, 3.0),
    'b': (0.0, 2.0)
}

best_params, best_score = genetic_algorithm(simple_model, data, param_ranges)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 7: AutoML Frameworks

AutoML frameworks integrate various hyperparameter optimization techniques into an automated workflow. While not a singular technique, they provide a comprehensive approach to model selection and hyperparameter tuning.

```python
class SimpleAutoML:
    def __init__(self, model_classes, param_ranges):
        self.model_classes = model_classes
        self.param_ranges = param_ranges
    
    def fit(self, X, y, time_budget=60):
        best_model = None
        best_score = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < time_budget:
            # Randomly select a model class
            model_class = random.choice(self.model_classes)
            
            # Randomly select hyperparameters
            params = {
                param: random.uniform(range_[0], range_[1])
                for param, range_ in self.param_ranges[model_class.__name__].items()
            }
            
            # Create and evaluate model
            model = model_class(**params)
            model.fit(X, y)
            score = model.score(X, y)
            
            if score < best_score:
                best_score = score
                best_model = model
        
        return best_model

# Example usage
class LinearRegression:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = None
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(100):  # Simple gradient descent
            y_pred = np.dot(X, self.weights)
            error = y_pred - y
            self.weights -= self.learning_rate * np.dot(X.T, error) / len(y)
    
    def score(self, X, y):
        y_pred = np.dot(X, self.weights)
        return np.mean((y - y_pred) ** 2)

model_classes = [LinearRegression]
param_ranges = {
    'LinearRegression': {'learning_rate': (0.01, 1.0)}
}

automl = SimpleAutoML(model_classes, param_ranges)
X = np.array([[x] for x, _ in data])
y = np.array([y for _, y in data])

best_model = automl.fit(X, y, time_budget=10)
print(f"Best model: {type(best_model).__name__}")
print(f"Best parameters: {best_model.learning_rate}")
print(f"Best score: {best_model.score(X, y)}")
```

Slide 8: Real-life Example: Image Classification

In this example, we'll use hyperparameter optimization for a simple image classification task using a basic neural network.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        d2 = output - y
        dw2 = np.dot(self.a1.T, d2) / m
        d1 = np.dot(d2, self.w2.T) * (self.a1 * (1 - self.a1))
        dw1 = np.dot(X.T, d1) / m
        
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2
    
    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def predict(self, X):
        return self.forward(X)

# Generate dummy data for binary classification
np.random.seed(0)
X = np.random.randn(1000, 20)  # 1000 samples, 20 features
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

# Define hyperparameter search space
param_space = {
    'hidden_size': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.5],
    'epochs': [100, 200, 300]
}

# Simple grid search
best_accuracy = 0
best_params = None

for hidden_size in param_space['hidden_size']:
    for learning_rate in param_space['learning_rate']:
        for epochs in param_space['epochs']:
            model = SimpleNN(20, hidden_size, 1, learning_rate)
            model.train(X, y, epochs)
            predictions = (model.predict(X) > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'hidden_size': hidden_size,
                    'learning_rate': learning_rate,
                    'epochs': epochs
                }

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {best_accuracy}")
```

Slide 9: Real-life Example: Text Classification

In this example, we'll optimize hyperparameters for a simple text classification model using the Naive Bayes algorithm.

```python
import re
from collections import defaultdict

class SimpleNaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha  # Smoothing parameter
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def preprocess(self, text):
        return re.findall(r'\w+', text.lower())

    def train(self, texts, labels):
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            words = self.preprocess(text)
            for word in words:
                self.feature_counts[label][word] += 1
                self.vocab.add(word)

    def predict(self, text):
        words = self.preprocess(text)
        scores = {}
        for label in self.class_counts:
            score = 0
            for word in words:
                likelihood = (self.feature_counts[label][word] + self.alpha) / (sum(self.feature_counts[label].values()) + self.alpha * len(self.vocab))
                score += likelihood
            scores[label] = score
        return max(scores, key=scores.get)

# Example data
texts = [
    "I love this movie",
    "This film is terrible",
    "Great acting and plot",
    "Waste of time and money",
    "Awesome special effects"
]
labels = ["positive", "negative", "positive", "negative", "positive"]

# Hyperparameter optimization
alpha_values = [0.1, 0.5, 1.0, 2.0]
best_accuracy = 0
best_alpha = None

for alpha in alpha_values:
    model = SimpleNaiveBayes(alpha)
    model.train(texts, labels)
    
    correct = sum(model.predict(text) == label for text, label in zip(texts, labels))
    accuracy = correct / len(texts)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_alpha = alpha

print(f"Best alpha: {best_alpha}")
print(f"Best accuracy: {best_accuracy}")
```

Slide 10: Importance of Cross-Validation

Cross-validation is crucial in hyperparameter optimization to prevent overfitting and ensure the model generalizes well to unseen data.

```python
import random

def cross_validate(model_class, param_grid, X, y, k=5):
    n = len(X)
    fold_size = n // k
    best_score = float('-inf')
    best_params = None

    for params in param_grid:
        scores = []
        for i in range(k):
            start = i * fold_size
            end = start + fold_size
            X_test = X[start:end]
            y_test = y[start:end]
            X_train = X[:start] + X[end:]
            y_train = y[:start] + y[end:]

            model = model_class(**params)
            model.train(X_train, y_train)
            score = model.evaluate(X_test, y_test)
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score

# Example usage
class DummyClassifier:
    def __init__(self, threshold):
        self.threshold = threshold

    def train(self, X, y):
        pass  # Dummy classifier doesn't need training

    def evaluate(self, X, y):
        predictions = [1 if x[0] > self.threshold else 0 for x in X]
        return sum(p == y_ for p, y_ in zip(predictions, y)) / len(y)

# Generate dummy data
X = [[random.random()] for _ in range(1000)]
y = [1 if x[0] > 0.5 else 0 for x in X]

param_grid = [{'threshold': t} for t in [0.3, 0.4, 0.5, 0.6, 0.7]]

best_params, best_score = cross_validate(DummyClassifier, param_grid, X, y)
print(f"Best parameters: {best_params}")
print(f"Best cross-validated score: {best_score}")
```

Slide 11: Handling Categorical Hyperparameters

Categorical hyperparameters require special handling in optimization processes. Here's an example of how to deal with them using one-hot encoding.

```python
import random

def one_hot_encode(value, categories):
    return [1 if cat == value else 0 for cat in categories]

class CategoricalModel:
    def __init__(self, activation, layers):
        self.activation = activation
        self.layers = layers

    def train(self, X, y):
        # Simulate training
        pass

    def evaluate(self, X, y):
        # Simulate evaluation
        return random.random()

def optimize_categorical_params(param_space, n_trials=100):
    best_score = float('-inf')
    best_params = None

    for _ in range(n_trials):
        params = {}
        for param, values in param_space.items():
            if isinstance(values, list):
                params[param] = random.choice(values)
            elif isinstance(values, tuple):
                params[param] = random.uniform(*values)

        # One-hot encode categorical parameters
        encoded_params = {}
        for param, value in params.items():
            if isinstance(param_space[param], list):
                encoded_params.update({f"{param}_{cat}": v for cat, v in zip(param_space[param], one_hot_encode(value, param_space[param]))})
            else:
                encoded_params[param] = value

        model = CategoricalModel(**params)
        model.train(X_train, y_train)
        score = model.evaluate(X_val, y_val)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

# Example usage
param_space = {
    'activation': ['relu', 'sigmoid', 'tanh'],
    'layers': (1, 5)
}

X_train, y_train = [1, 2, 3], [0, 1, 1]  # Dummy data
X_val, y_val = [4, 5], [1, 0]  # Dummy validation data

best_params, best_score = optimize_categorical_params(param_space)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 12: Handling Resource Constraints

When optimizing hyperparameters, it's important to consider resource constraints such as time and computational power. Here's an example of how to implement a time-based stopping criterion.

```python
import time

def time_constrained_optimization(param_space, time_budget):
    start_time = time.time()
    best_score = float('-inf')
    best_params = None
    iterations = 0

    while time.time() - start_time < time_budget:
        # Sample random parameters
        params = {
            param: random.choice(values) if isinstance(values, list) else random.uniform(*values)
            for param, values in param_space.items()
        }

        # Simulate model training and evaluation
        model = DummyModel(**params)
        score = model.train_and_evaluate()

        if score > best_score:
            best_score = score
            best_params = params

        iterations += 1

    elapsed_time = time.time() - start_time
    return best_params, best_score, iterations, elapsed_time

class DummyModel:
    def __init__(self, **params):
        self.params = params

    def train_and_evaluate(self):
        # Simulate training and evaluation time
        time.sleep(0.1)
        return random.random()

# Example usage
param_space = {
    'learning_rate': (0.001, 0.1),
    'batch_size': [32, 64, 128],
    'num_layers': (1, 5)
}

time_budget = 5  # seconds

best_params, best_score, iterations, elapsed_time = time_constrained_optimization(param_space, time_budget)

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
print(f"Number of iterations: {iterations}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")
```

Slide 13: Visualizing Hyperparameter Importance

Visualizing the importance of different hyperparameters can provide insights into which parameters have the most significant impact on model performance.

```python
import random
import matplotlib.pyplot as plt

def evaluate_model(params):
    # Simulate model evaluation
    return (
        params['learning_rate'] * 10 +
        params['num_layers'] * 5 +
        params['dropout'] * 2 +
        random.uniform(-1, 1)
    )

def random_search(param_space, n_iterations=1000):
    results = []
    for _ in range(n_iterations):
        params = {
            param: random.uniform(*values) if isinstance(values, tuple) else random.choice(values)
            for param, values in param_space.items()
        }
        score = evaluate_model(params)
        results.append((params, score))
    return results

def plot_parameter_importance(results):
    param_names = list(results[0][0].keys())
    param_values = {param: [] for param in param_names}
    scores = []

    for params, score in results:
        for param, value in params.items():
            param_values[param].append(value)
        scores.append(score)

    fig, axes = plt.subplots(1, len(param_names), figsize=(15, 5))
    for ax, param in zip(axes, param_names):
        ax.scatter(param_values[param], scores, alpha=0.5)
        ax.set_xlabel(param)
        ax.set_ylabel('Score')
        ax.set_title(f'{param} vs Score')

    plt.tight_layout()
    plt.show()

# Example usage
param_space = {
    'learning_rate': (0.001, 0.1),
    'num_layers': (1, 10),
    'dropout': (0.1, 0.5),
    'activation': ['relu', 'tanh', 'sigmoid']
}

results = random_search(param_space)
plot_parameter_importance(results)

# Find best parameters
best_params, best_score = max(results, key=lambda x: x[1])
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

Slide 14: Additional Resources

For those interested in delving deeper into hyperparameter optimization, here are some valuable resources:

1.  "Algorithms for Hyper-Parameter Optimization" by Bergstra et al. (2011) ArXiv: [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)
2.  "Practical Bayesian Optimization of Machine Learning Algorithms" by Snoek et al. (2012) ArXiv: [https://arxiv.org/abs/1206.2944](https://arxiv.org/abs/1206.2944)
3.  "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization" by Li et al. (2017) ArXiv: [https://arxiv.org/abs/1603.06560](https://arxiv.org/abs/1603.06560)
4.  "Auto-WEKA: Combined Selection and Hyperparameter Optimization of Classification Algorithms" by Thornton et al. (2013) ArXiv: [https://arxiv.org/abs/1208.3719](https://arxiv.org/abs/1208.3719)

These papers provide in-depth discussions on various hyperparameter optimization techniques and their applications in machine learning.

