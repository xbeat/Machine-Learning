## Markov Process with Python Visualizations
Slide 1: Introduction to Markov Processes

Markov processes are stochastic models that describe a sequence of possible events where the probability of each event depends solely on the state attained in the previous event. Let's start with a simple example:

```python
import random

def coin_flip():
    return random.choice(['H', 'T'])

def markov_coin_flips(n):
    state = coin_flip()
    sequence = [state]
    for _ in range(n-1):
        if state == 'H':
            state = 'H' if random.random() < 0.7 else 'T'
        else:
            state = 'T' if random.random() < 0.6 else 'H'
        sequence.append(state)
    return sequence

print(markov_coin_flips(10))
```

Slide 2: Markov Chain Fundamentals

A Markov chain is a type of Markov process with a discrete set of states. The transition probabilities between states are represented by a transition matrix.

```python
import numpy as np

# Transition matrix for a weather model
# States: Sunny (0), Cloudy (1), Rainy (2)
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])

current_state = 0  # Start with Sunny
weather_sequence = [current_state]

for _ in range(7):  # Predict weather for a week
    current_state = np.random.choice(3, p=P[current_state])
    weather_sequence.append(current_state)

print(weather_sequence)
```

Slide 3: State Space and Transition Probabilities

The state space is the set of all possible states in a Markov process. Transition probabilities define the likelihood of moving from one state to another.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_weighted_edges_from([
    (0, 0, 0.7), (0, 1, 0.2), (0, 2, 0.1),
    (1, 0, 0.3), (1, 1, 0.4), (1, 2, 0.3),
    (2, 0, 0.2), (2, 1, 0.3), (2, 2, 0.5)
])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrows=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Weather Model State Transitions")
plt.show()
```

Slide 4: Chapman-Kolmogorov Equation

The Chapman-Kolmogorov equation describes the probability of transitioning from one state to another over multiple steps.

```python
def chapman_kolmogorov(P, n):
    return np.linalg.matrix_power(P, n)

# Using the weather model transition matrix
P_3_steps = chapman_kolmogorov(P, 3)
print("Probability matrix after 3 steps:")
print(P_3_steps)
```

Slide 5: Stationary Distribution

The stationary distribution is a probability distribution that remains unchanged in the Markov chain as time progresses.

```python
def find_stationary_distribution(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real
    return stationary / stationary.sum()

stationary_dist = find_stationary_distribution(P)
print("Stationary distribution:", stationary_dist)
```

Slide 6: Absorbing Markov Chains

An absorbing Markov chain has at least one absorbing state, which, once entered, cannot be left.

```python
# Transition matrix for a simple game
# States: Playing (0), Win (1), Lose (2)
P_game = np.array([
    [0.6, 0.3, 0.1],
    [0, 1, 0],
    [0, 0, 1]
])

def play_game():
    state = 0
    while state == 0:
        state = np.random.choice(3, p=P_game[state])
    return "Win" if state == 1 else "Lose"

results = [play_game() for _ in range(1000)]
print(f"Wins: {results.count('Win')}, Losses: {results.count('Lose')}")
```

Slide 7: Hidden Markov Models

Hidden Markov Models (HMMs) are Markov chains where the state is not directly observable, but outputs dependent on the state are visible.

```python
from hmmlearn import hmm

# Simple HMM for stock market trends
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Example data: daily stock returns
X = np.array([[0.7], [1.1], [0.6], [-0.2], [0.1], [1.5], [0.3], [-0.1]])

model.fit(X)

# Predict hidden states
hidden_states = model.predict(X)
print("Predicted market trends:", ["Bullish" if state == 0 else "Bearish" for state in hidden_states])
```

Slide 8: Markov Decision Processes

Markov Decision Processes (MDPs) extend Markov chains by adding actions and rewards, useful in reinforcement learning.

```python
import gym

env = gym.make('FrozenLake-v1', is_slippery=False)

def simple_policy(state):
    return env.action_space.sample()  # Random action

total_reward = 0
obs = env.reset()
for _ in range(100):
    action = simple_policy(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Total reward: {total_reward}")
```

Slide 9: Continuous-Time Markov Chains

Continuous-Time Markov Chains (CTMCs) model systems where state changes can occur at any time.

```python
import scipy.linalg as linalg

# Rate matrix for a simple CTMC
Q = np.array([
    [-3, 2, 1],
    [1, -2, 1],
    [1, 1, -2]
])

def ctmc_transition_prob(Q, t):
    return linalg.expm(Q * t)

P_t = ctmc_transition_prob(Q, 0.5)
print("Transition probabilities at t=0.5:")
print(P_t)
```

Slide 10: Monte Carlo Methods for Markov Chains

Monte Carlo methods can be used to estimate properties of Markov chains through repeated random sampling.

```python
def monte_carlo_stationary(P, n_simulations, n_steps):
    states = np.arange(P.shape[0])
    final_states = []
    
    for _ in range(n_simulations):
        state = np.random.choice(states)
        for _ in range(n_steps):
            state = np.random.choice(states, p=P[state])
        final_states.append(state)
    
    return np.bincount(final_states) / n_simulations

estimated_stationary = monte_carlo_stationary(P, 10000, 1000)
print("Estimated stationary distribution:", estimated_stationary)
```

Slide 11: Markov Chain Monte Carlo (MCMC)

MCMC is a class of algorithms for sampling from probability distributions based on constructing a Markov chain.

```python
import pymc3 as pm

def mcmc_example():
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sd=1)
        obs = pm.Normal('obs', mu=mu, sd=1, observed=np.random.randn(100))
        
        trace = pm.sample(1000, tune=1000)
    
    pm.plot_posterior(trace, var_names=['mu'])
    plt.show()

mcmc_example()
```

Slide 12: Applications in Natural Language Processing

Markov chains are used in various NLP tasks, including text generation and speech recognition.

```python
from collections import defaultdict
import random

def build_markov_model(text, n=2):
    model = defaultdict(list)
    words = text.split()
    for i in range(len(words) - n):
        state = tuple(words[i:i+n])
        next_word = words[i+n]
        model[state].append(next_word)
    return model

def generate_text(model, n_words=50, n=2):
    current = random.choice(list(model.keys()))
    result = list(current)
    for _ in range(n_words - n):
        if current not in model:
            break
        next_word = random.choice(model[current])
        result.append(next_word)
        current = tuple(result[-n:])
    return ' '.join(result)

text = """The quick brown fox jumps over the lazy dog. 
          The dog barks at the fox. The fox runs away quickly. 
          The lazy dog goes back to sleep."""
model = build_markov_model(text)
print(generate_text(model, n_words=20))
```

Slide 13: Markov Models in Bioinformatics

Markov models are extensively used in bioinformatics for sequence analysis and gene prediction.

```python
from Bio import SeqIO
from Bio.Seq import Seq

def calculate_transition_probabilities(sequences):
    transitions = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for i in range(len(seq) - 1):
            current, next = seq[i], seq[i+1]
            transitions[current][next] += 1
    
    for current in transitions:
        total = sum(transitions[current].values())
        for next in transitions[current]:
            transitions[current][next] /= total
    
    return transitions

# Example DNA sequences
sequences = [Seq("ATGCATGC"), Seq("ATGGATCC"), Seq("ATGCATCC")]
trans_prob = calculate_transition_probabilities(sequences)

for base in "ATGC":
    print(f"Transitions from {base}:", dict(trans_prob[base]))
```

Slide 14: Additional Resources

For further exploration of Markov Processes, consider these resources:

1. "An Introduction to Markov Processes" by Daniel W. Stroock ArXiv: [https://arxiv.org/abs/1106.4146](https://arxiv.org/abs/1106.4146)
2. "Hidden Markov Models for Time Series: An Introduction Using R" by Walter Zucchini et al. Book information: [https://www.crcpress.com/Hidden-Markov-Models-for-Time-Series-An-Introduction-Using-R-Second-Edition/Zucchini-MacDonald-Langrock/p/book/9781482253832](https://www.crcpress.com/Hidden-Markov-Models-for-Time-Series-An-Introduction-Using-R-Second-Edition/Zucchini-MacDonald-Langrock/p/book/9781482253832)
3. "Markov Chain Monte Carlo in Practice" by W.R. Gilks et al. Book information: [https://www.routledge.com/Markov-Chain-Monte-Carlo-in-Practice/Gilks-Richardson-Spiegelhalter/p/book/9780412055515](https://www.routledge.com/Markov-Chain-Monte-Carlo-in-Practice/Gilks-Richardson-Spiegelhalter/p/book/9780412055515)

These resources provide in-depth coverage of Markov processes and their applications in various fields.

