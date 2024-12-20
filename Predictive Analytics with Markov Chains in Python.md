## Predictive Analytics with Markov Chains in Python
Slide 1: Introduction to Markov Chains

Markov chains are mathematical systems that undergo transitions from one state to another according to certain probabilistic rules. They are named after the Russian mathematician Andrey Markov and have wide-ranging applications in various fields, including predictive analytics.

```python
import numpy as np

# Define the transition matrix
P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Initial state
initial_state = np.array([1, 0])

# Compute the state after one step
next_state = initial_state.dot(P)

print("Initial state:", initial_state)
print("State after one step:", next_state)
```

Slide 2: Properties of Markov Chains

Markov chains are characterized by the Markov property, which states that the probability of transitioning to any particular state depends solely on the current state and not on the sequence of events that preceded it. This "memoryless" property makes Markov chains particularly useful for modeling systems with limited history dependence.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_markov_chain(P, initial_state, steps):
    states = [initial_state]
    current_state = initial_state
    for _ in range(steps):
        current_state = np.random.choice([0, 1], p=P[current_state])
        states.append(current_state)
    return states

P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

states = simulate_markov_chain(P, 0, 100)

plt.plot(states)
plt.title("Markov Chain Simulation")
plt.xlabel("Step")
plt.ylabel("State")
plt.show()
```

Slide 3: Transition Matrices

The transition matrix is a fundamental concept in Markov chains. It represents the probabilities of transitioning from one state to another. Each element (i, j) in the matrix represents the probability of moving from state i to state j.

```python
import numpy as np

# Define a transition matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

print("Transition Matrix:")
print(P)

# Verify that each row sums to 1
row_sums = np.sum(P, axis=1)
print("\nRow sums:", row_sums)

# Calculate the probability of being in each state after 2 steps
initial_state = np.array([1, 0, 0])
after_two_steps = initial_state.dot(np.linalg.matrix_power(P, 2))

print("\nProbability distribution after 2 steps:")
print(after_two_steps)
```

Slide 4: Stationary Distribution

The stationary distribution of a Markov chain is a probability distribution that remains unchanged as the system evolves. It represents the long-term behavior of the system, regardless of the initial state.

```python
import numpy as np

def find_stationary_distribution(P, max_iter=1000, tol=1e-8):
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_next = pi.dot(P)
        if np.allclose(pi, pi_next, atol=tol):
            return pi_next
        pi = pi_next
    return pi

P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

stationary_dist = find_stationary_distribution(P)
print("Stationary distribution:", stationary_dist)

# Verify that the stationary distribution is unchanged by P
print("Verification:", stationary_dist.dot(P))
```

Slide 5: Predicting Future States

One of the key applications of Markov chains in predictive analytics is forecasting future states of a system. By using the transition matrix and the current state, we can compute the probability distribution of future states.

```python
import numpy as np

def predict_future_states(P, initial_state, steps):
    current_state = initial_state
    print(f"Initial state: {current_state}")
    for i in range(1, steps + 1):
        next_state = current_state.dot(P)
        print(f"State after {i} step(s): {next_state}")
        current_state = next_state

P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

initial_state = np.array([1, 0, 0])
predict_future_states(P, initial_state, 5)
```

Slide 6: Hidden Markov Models

Hidden Markov Models (HMMs) are an extension of Markov chains where the system's state is not directly observable. Instead, we observe outputs that are probabilistically related to the hidden states. HMMs are widely used in speech recognition, natural language processing, and bioinformatics.

```python
import numpy as np
from hmmlearn import hmm

# Define the HMM model
model = hmm.MultinomialHMM(n_components=2, random_state=42)

# Transition matrix
model.transmat_ = np.array([[0.7, 0.3],
                            [0.4, 0.6]])

# Emission probabilities
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                [0.6, 0.3, 0.1]])

# Generate a sequence of 100 observations
X, Z = model.sample(100)

print("Generated observation sequence:")
print(X.flatten())
print("\nHidden state sequence:")
print(Z)
```

Slide 7: Markov Chain Monte Carlo (MCMC)

Markov Chain Monte Carlo is a class of algorithms for sampling from probability distributions. It's particularly useful for solving complex problems in statistics, physics, and machine learning. MCMC uses Markov chains to explore the state space and generate samples from the target distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(target_pdf, proposal_pdf, proposal_sampler, initial_state, n_samples):
    samples = [initial_state]
    current_state = initial_state
    
    for _ in range(n_samples - 1):
        proposed_state = proposal_sampler(current_state)
        
        acceptance_ratio = (target_pdf(proposed_state) * proposal_pdf(current_state, proposed_state)) / \
                           (target_pdf(current_state) * proposal_pdf(proposed_state, current_state))
        
        if np.random.random() < acceptance_ratio:
            current_state = proposed_state
        
        samples.append(current_state)
    
    return samples

# Target distribution: Standard normal
target_pdf = lambda x: np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Proposal distribution: Gaussian with std dev 0.5
proposal_pdf = lambda x, mu: np.exp(-0.5 * ((x - mu) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
proposal_sampler = lambda mu: np.random.normal(mu, 0.5)

samples = metropolis_hastings(target_pdf, proposal_pdf, proposal_sampler, 0, 10000)

plt.hist(samples, bins=50, density=True, alpha=0.7)
plt.title("MCMC Sampling of Standard Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

Slide 8: Real-Life Example: Weather Prediction

Markov chains can be used to model and predict weather patterns. In this example, we'll use a simple Markov chain to predict weather states (sunny, cloudy, rainy) based on historical transition probabilities.

```python
import numpy as np

# Weather states: 0 - Sunny, 1 - Cloudy, 2 - Rainy
weather_transition = np.array([[0.7, 0.2, 0.1],
                               [0.3, 0.4, 0.3],
                               [0.2, 0.3, 0.5]])

def predict_weather(current_state, days):
    states = ["Sunny", "Cloudy", "Rainy"]
    print(f"Current weather: {states[current_state]}")
    
    for day in range(1, days + 1):
        next_state = np.random.choice(3, p=weather_transition[current_state])
        print(f"Day {day} forecast: {states[next_state]}")
        current_state = next_state

# Predict weather for the next 7 days, starting from a sunny day
predict_weather(0, 7)
```

Slide 9: Real-Life Example: Page Rank Algorithm

The PageRank algorithm, used by Google to rank web pages, is based on Markov chains. It models the behavior of a random surfer on the web and uses the stationary distribution of this Markov chain to determine the importance of web pages.

```python
import numpy as np

def pagerank(G, damping_factor=0.85, epsilon=1e-8, max_iterations=100):
    n = len(G)
    out_degree = np.sum(G, axis=1)
    out_degree[out_degree == 0] = 1  # Avoid division by zero
    
    P = (G / out_degree[:, np.newaxis]).T
    v = np.ones(n) / n
    
    for _ in range(max_iterations):
        v_prev = v
        v = (1 - damping_factor) / n + damping_factor * P.dot(v)
        if np.sum(np.abs(v - v_prev)) < epsilon:
            break
    
    return v

# Example web graph (adjacency matrix)
G = np.array([[0, 1, 1, 0],
              [0, 0, 1, 1],
              [1, 0, 0, 1],
              [0, 0, 1, 0]])

pageranks = pagerank(G)
print("PageRank values:")
for i, pr in enumerate(pageranks):
    print(f"Page {i+1}: {pr:.4f}")
```

Slide 10: Markov Decision Processes (MDPs)

Markov Decision Processes extend Markov chains by incorporating actions and rewards. They are fundamental in reinforcement learning and decision-making under uncertainty. MDPs consist of states, actions, transition probabilities, and rewards.

```python
import numpy as np

class SimpleMDP:
    def __init__(self, n_states, n_actions, transition_probs, rewards, gamma=0.9):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma

    def value_iteration(self, epsilon=1e-6):
        V = np.zeros(self.n_states)
        while True:
            V_prev = V.()
            for s in range(self.n_states):
                Q_sa = [sum([self.transition_probs[s, a, s_next] * 
                             (self.rewards[s, a, s_next] + self.gamma * V_prev[s_next])
                             for s_next in range(self.n_states)])
                        for a in range(self.n_actions)]
                V[s] = max(Q_sa)
            if np.max(np.abs(V - V_prev)) < epsilon:
                break
        return V

# Example MDP: 2 states, 2 actions
n_states, n_actions = 2, 2
transition_probs = np.array([
    [[0.7, 0.3], [0.4, 0.6]],
    [[0.2, 0.8], [0.9, 0.1]]
])
rewards = np.array([
    [[1, 0], [-1, 2]],
    [[0, 1], [2, -1]]
])

mdp = SimpleMDP(n_states, n_actions, transition_probs, rewards)
optimal_values = mdp.value_iteration()
print("Optimal state values:", optimal_values)
```

Slide 11: Continuous-Time Markov Chains

Continuous-Time Markov Chains (CTMCs) extend the concept of Markov chains to continuous time. They are useful for modeling systems where state transitions can occur at any time, rather than at discrete time steps.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ctmc(Q, initial_state, T):
    states = [initial_state]
    times = [0]
    current_state = initial_state
    current_time = 0
    
    while current_time < T:
        rates = -Q[current_state, current_state]
        time_to_next = np.random.exponential(1/rates)
        current_time += time_to_next
        
        if current_time > T:
            break
        
        transition_probs = Q[current_state, :] / rates
        transition_probs[current_state] = 0
        next_state = np.random.choice(len(Q), p=transition_probs)
        
        states.append(next_state)
        times.append(current_time)
        current_state = next_state
    
    return np.array(states), np.array(times)

# Example: Birth-Death process
Q = np.array([[-1, 1, 0],
              [0.5, -1, 0.5],
              [0, 1, -1]])

states, times = simulate_ctmc(Q, 0, 10)

plt.step(times, states, where='post')
plt.title("Continuous-Time Markov Chain Simulation")
plt.xlabel("Time")
plt.ylabel("State")
plt.yticks([0, 1, 2])
plt.show()
```

Slide 12: Limitations and Considerations

While Markov chains are powerful tools for predictive analytics, they have limitations. The Markov property assumes that the future state depends only on the current state, which may not always hold in real-world scenarios. Additionally, the accuracy of predictions depends heavily on the quality and representativeness of the transition probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

def markov_prediction(P, initial_state, steps):
    state = initial_state
    states = [state]
    for _ in range(steps):
        state = np.random.choice(len(P), p=P[state])
        states.append(state)
    return states

def non_markovian_prediction(P, initial_states, steps):
    states = initial_states.()
    for _ in range(steps - len(initial_states) + 1):
        prob = P[states[-2], states[-1]]
        state = np.random.choice(2, p=[1-prob, prob])
        states.append(state)
    return states

# Markov chain transition matrix
P = np.array([[0.7, 0.3],
              [0.4, 0.6]])

markov_states = markov_prediction(P, 0, 100)
non_markov_states = non_markovian_prediction(P, [0, 1], 100)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.step(range(len(markov_states)), markov_states, where='post')
plt.title("Markov Chain Prediction")
plt.ylabel("State")

plt.subplot(2, 1, 2)
plt.step(range(len(non_markov_states)), non_markov_states, where='post')
plt.title("Non-Markovian Process")
plt.xlabel("Step")
plt.ylabel("State")

plt.tight_layout()
plt.show()
```

Slide 13: Implementing Markov Chains in Python: Best Practices

When implementing Markov chains in Python, it's important to follow best practices for efficient and maintainable code. Use numpy for matrix operations, implement proper error handling, and document your code thoroughly. Consider using object-oriented programming to encapsulate Markov chain logic.

```python
import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix):
        self.P = np.array(transition_matrix)
        self._validate()

    def _validate(self):
        if not np.allclose(self.P.sum(axis=1), 1):
            raise ValueError("Transition matrix rows must sum to 1")

    def simulate(self, initial_state, steps):
        states = [initial_state]
        for _ in range(steps):
            states.append(np.random.choice(len(self.P), p=self.P[states[-1]]))
        return states

# Usage example
mc = MarkovChain([[0.7, 0.3], [0.4, 0.6]])
simulation = mc.simulate(0, 10)
print("Simulated states:", simulation)
```

Slide 14: Advanced Markov Chain Applications

Markov chains have diverse applications beyond simple state predictions. They are used in natural language processing for text generation, in bioinformatics for analyzing DNA sequences, and in queueing theory for modeling system performance.

```python
import numpy as np
from collections import defaultdict

def generate_text(corpus, n_words, n_gram=2):
    # Build n-gram model
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(corpus) - n_gram):
        state = tuple(corpus[i:i+n_gram])
        next_word = corpus[i+n_gram]
        model[state][next_word] += 1

    # Convert counts to probabilities
    for state in model:
        total = sum(model[state].values())
        model[state] = {word: count/total for word, count in model[state].items()}

    # Generate text
    current = tuple(corpus[:n_gram])
    result = list(current)
    for _ in range(n_words - n_gram):
        next_word = np.random.choice(list(model[current].keys()), p=list(model[current].values()))
        result.append(next_word)
        current = tuple(result[-n_gram:])

    return ' '.join(result)

# Example usage
corpus = "the quick brown fox jumps over the lazy dog".split()
generated_text = generate_text(corpus, 20)
print("Generated text:", generated_text)
```

Slide 15: Additional Resources

For those interested in diving deeper into Markov chains and their applications in predictive analytics, here are some valuable resources:

1. "Introduction to Probability Models" by Sheldon M. Ross
2. "Markov Chains and Stochastic Stability" by Sean Meyn and Richard L. Tweedie
3. ArXiv paper: "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" by Lawrence R. Rabiner ([https://arxiv.org/abs/stat.ML/0602019](https://arxiv.org/abs/stat.ML/0602019))
4. "Pattern Recognition and Machine Learning" by Christopher M. Bishop

These resources provide a comprehensive understanding of Markov chains, from theoretical foundations to practical applications in various fields.

