## MonteCarlo in Python

Slide 1: 
What are Monte Carlo Simulations?

Monte Carlo simulations are computational algorithms that rely on repeated random sampling to obtain numerical results. They are used to model complex systems and processes, especially when deterministic solutions are difficult or impossible to obtain.

In Python, you can leverage libraries like NumPy and random to perform Monte Carlo simulations with ease.

Slide 2: Generating Random Numbers

The foundation of Monte Carlo simulations is the generation of random numbers. In Python, you can use the random module to generate random numbers within a specified range.

Code Example:

```python
import random

# Generate a random integer between 1 and 100
random_integer = random.randint(1, 100)

# Generate a random float between 0 and 1
random_float = random.random()
```

Slide 3: 
Simulating a Coin Toss

One of the simplest Monte Carlo simulations is simulating a coin toss. We can use the random.random() function to generate a random number between 0 and 1, and map it to either heads or tails.

Code Example:

```python
import random

def toss_coin():
    if random.random() < 0.5:
        return "Heads"
    else:
        return "Tails"

# Simulate 10 coin tosses
for _ in range(10):
    result = toss_coin()
    print(result)
```

Slide 4: 
Monte Carlo Integration

Monte Carlo integration is a technique for estimating the definite integral of a function by generating random points within the integration bounds and evaluating the function at those points.

Code Example:

```python
import random
import math

def monte_carlo_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

# Estimate pi using 1,000,000 samples
pi_estimate = monte_carlo_pi(1000000)
print(f"Estimated value of pi: {pi_estimate}")
```

Slide 5: 
Monte Carlo Simulations in Finance

Monte Carlo simulations are widely used in finance for tasks like option pricing, portfolio optimization, and risk analysis. One common application is estimating the value of an option using the Black-Scholes model.

Code Example:

```python
import numpy as np
from scipy.stats import norm

def monte_carlo_option_pricing(S0, K, r, sigma, T, num_sims):
    dt = T / num_sims
    S = np.zeros(num_sims + 1)
    S[0] = S0
    for i in range(num_sims):
        epsilon = norm.rvs()
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * epsilon * np.sqrt(dt))
    option_value = np.exp(-r * T) * np.mean(np.maximum(S[-1] - K, 0))
    return option_value
```

Slide 6: 
Monte Carlo Simulation in Science

Monte Carlo simulations are used in various scientific domains, including physics, chemistry, and biology. One example is simulating the behavior of particles in a system using the Metropolis algorithm.

Code Example:

```python
import random

def metropolis(energy, temperature, num_steps):
    current_state = initial_state()
    for step in range(num_steps):
        new_state = propose_new_state(current_state)
        delta_energy = energy(new_state) - energy(current_state)
        if delta_energy <= 0 or random.random() < np.exp(-delta_energy / temperature):
            current_state = new_state
    return current_state
```

Slide 7: 
Monte Carlo Simulation in Machine Learning

Monte Carlo simulations can be used in machine learning for tasks such as hyperparameter tuning, uncertainty estimation, and Bayesian optimization.

Code Example:

```python
from scipy.stats import uniform

def monte_carlo_hyperparameter_tuning(X, y, model, param_ranges):
    best_score = -np.inf
    best_params = None
    for _ in range(num_iterations):
        params = {name: uniform.rvs(*bounds) for name, bounds in param_ranges.items()}
        model.set_params(**params)
        score = cross_val_score(model, X, y).mean()
        if score > best_score:
            best_score = score
            best_params = params
    return best_params
```

Slide 8: 
Importance Sampling

Importance sampling is a variance reduction technique used in Monte Carlo simulations. It involves sampling from an alternative distribution and correcting the estimates using importance weights.

Code Example:

```python
import numpy as np

def importance_sampling(num_samples, target_dist, proposal_dist):
    samples = proposal_dist.rvs(num_samples)
    weights = target_dist.pdf(samples) / proposal_dist.pdf(samples)
    estimate = np.sum(weights * samples) / np.sum(weights)
    return estimate
```

Slide 9: 
Markov Chain Monte Carlo (MCMC)

MCMC is a class of Monte Carlo algorithms that generate samples from a probability distribution by constructing a Markov chain that has the desired distribution as its stationary distribution.

Code Example:

```python
import numpy as np

def metropolis_hastings(target_dist, proposal_dist, num_samples, initial_state):
    states = [initial_state]
    for _ in range(num_samples - 1):
        current_state = states[-1]
        proposed_state = proposal_dist(current_state)
        acceptance_ratio = target_dist(proposed_state) / target_dist(current_state)
        if np.random.rand() < min(1, acceptance_ratio):
            states.append(proposed_state)
        else:
            states.append(current_state)
    return states
```

Slide 10: 
Monte Carlo Tree Search

Monte Carlo Tree Search (MCTS) is a heuristic search algorithm used in game theory and decision-making problems. It combines random sampling with a tree search to find optimal decisions.

Code Example:

```python
def mcts(root_node, num_simulations):
    for _ in range(num_simulations):
        node = root_node
        state = node.state
        while not node.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child()
            else:
                new_node = node.expand()
                node = new_node
                break
        while node is not None:
            node.update(state)
            node = node.parent
    return root_node.best_child()
```

Slide 11: 
Monte Carlo Optimization

Monte Carlo optimization is a class of algorithms that use Monte Carlo simulations to find optimal solutions to optimization problems, especially when the objective function is complex or non-differentiable.

Code Example:

```python
import numpy as np

def monte_carlo_optimization(objective_function, param_ranges, num_iterations):
    best_params = None
    best_value = -np.inf
    for _ in range(num_iterations):
        params = {name: np.random.uniform(*bounds) for name, bounds in param_ranges.items()}
        value = objective_function(**params)
        if value > best_value:
            best_value = value
            best_params = params
    return best_params
```

Slide 12: 
Monte Carlo Simulation in Games

Monte Carlo simulations are used in game development for various purposes, such as generating random levels, simulating physics, and modeling agent behavior in game AI.

```python
import random

def generate_random_level(width, height, obstacle_probability):
    level = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            if random.random() < obstacle_probability:
                level[i][j] = 1  # 1 represents an obstacle
    return level
```

Slide 13: 
Further Exploration

Monte Carlo simulations are versatile and have applications in many domains beyond the examples covered in this presentation. Some additional areas to explore include:

* Computational biology and bioinformatics
* Quantum computing and simulations
* Cryptography and security
* Computer graphics and rendering
* Monte Carlo methods in deep learning

There are also advanced techniques like Quasi-Monte Carlo methods, Parallel Monte Carlo simulations, and more to dive into.

```python
# Example: Parallel Monte Carlo simulation using multiprocessing
import multiprocessing as mp

def parallel_simulation(func, num_processes, *args):
    pool = mp.Pool(processes=num_processes)
    results = pool.starmap(func, [(arg,) for arg in args])
    pool.close()
    pool.join()
    return results
```

