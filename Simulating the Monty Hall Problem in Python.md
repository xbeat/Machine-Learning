## Simulating the Monty Hall Problem in Python
Slide 1: The Monty Hall Problem in Machine Learning

The Monty Hall problem, a classic probability puzzle, has intriguing applications in machine learning. This slideshow explores how we can use Python to simulate and analyze this problem, demonstrating its relevance to decision-making algorithms and reinforcement learning.

```python
import random

def monty_hall_game(switch):
    doors = ['goat', 'goat', 'car']
    random.shuffle(doors)
    
    # Player's initial choice
    choice = random.randint(0, 2)
    
    # Monty opens a door with a goat
    remaining_doors = [i for i in range(3) if i != choice and doors[i] != 'car']
    opened_door = random.choice(remaining_doors)
    
    # Player switches or stays
    if switch:
        choice = [i for i in range(3) if i != choice and i != opened_door][0]
    
    return doors[choice] == 'car'

# Simulate 10000 games
wins_with_switch = sum(monty_hall_game(switch=True) for _ in range(10000))
wins_without_switch = sum(monty_hall_game(switch=False) for _ in range(10000))

print(f"Win rate with switching: {wins_with_switch / 100:.2f}%")
print(f"Win rate without switching: {wins_without_switch / 100:.2f}%")
```

Slide 2: Understanding the Monty Hall Problem

The Monty Hall problem involves a game show scenario where a contestant chooses one of three doors. Behind one door is a car, and behind the other two are goats. After the contestant makes their choice, the host (who knows what's behind each door) opens another door revealing a goat. The contestant is then given the option to switch their choice to the remaining unopened door or stick with their original choice.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_monty_hall_doors():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initial state
    ax1.bar(['Door 1', 'Door 2', 'Door 3'], [1, 1, 1], color=['brown']*3)
    ax1.set_title('Initial State')
    ax1.set_ylabel('Closed (1) / Opened (0)')
    
    # After host opens a door
    ax2.bar(['Door 1', 'Door 2', 'Door 3'], [1, 0, 1], color=['brown', 'lightgray', 'brown'])
    ax2.set_title('After Host Opens a Door')
    
    plt.tight_layout()
    plt.show()

plot_monty_hall_doors()
```

Slide 3: Probability Analysis

The counterintuitive aspect of the Monty Hall problem is that switching doors increases the probability of winning from 1/3 to 2/3. This can be explained by considering the initial probability distribution and how it changes after the host opens a door.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_probability_distribution():
    labels = ['Stay', 'Switch']
    probabilities = [1/3, 2/3]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probabilities, color=['blue', 'orange'])
    plt.title('Probability of Winning')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    
    for i, v in enumerate(probabilities):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.show()

plot_probability_distribution()
```

Slide 4: Simulating the Monty Hall Problem

We can use Python to simulate thousands of Monty Hall games to empirically verify the theoretical probabilities. This simulation helps us understand the problem through practical experimentation.

```python
import random
import matplotlib.pyplot as plt

def monty_hall_simulation(num_games):
    switch_wins, stay_wins = 0, 0
    
    for _ in range(num_games):
        doors = ['goat', 'goat', 'car']
        random.shuffle(doors)
        
        # Player's initial choice
        choice = random.randint(0, 2)
        
        # Monty opens a door with a goat
        remaining_doors = [i for i in range(3) if i != choice and doors[i] != 'car']
        opened_door = random.choice(remaining_doors)
        
        # If player switches
        switch_choice = [i for i in range(3) if i != choice and i != opened_door][0]
        
        if doors[switch_choice] == 'car':
            switch_wins += 1
        if doors[choice] == 'car':
            stay_wins += 1
    
    return switch_wins / num_games, stay_wins / num_games

num_games = 100000
switch_prob, stay_prob = monty_hall_simulation(num_games)

plt.figure(figsize=(10, 6))
plt.bar(['Switch', 'Stay'], [switch_prob, stay_prob])
plt.title(f'Monty Hall Simulation Results ({num_games:,} games)')
plt.ylabel('Probability of Winning')
plt.ylim(0, 1)

for i, v in enumerate([switch_prob, stay_prob]):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.show()
```

Slide 5: Applying Monty Hall to Machine Learning: Reinforcement Learning

The Monty Hall problem can be framed as a reinforcement learning task, where an agent learns the optimal strategy through trial and error. We can use a simple Q-learning algorithm to demonstrate this concept.

```python
import numpy as np
import random

class MontyHallEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.doors = ['goat', 'goat', 'car']
        random.shuffle(self.doors)
        self.initial_choice = random.randint(0, 2)
        return self.initial_choice
    
    def step(self, action):  # action: 0 (stay), 1 (switch)
        if action == 0:  # stay
            reward = 1 if self.doors[self.initial_choice] == 'car' else 0
        else:  # switch
            remaining_doors = [i for i in range(3) if i != self.initial_choice]
            new_choice = random.choice(remaining_doors)
            reward = 1 if self.doors[new_choice] == 'car' else 0
        
        return reward

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((3, 2))  # 3 initial choices, 2 actions (stay/switch)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
    
    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

# Training
env = MontyHallEnvironment()
agent = QLearningAgent()

episodes = 10000
for _ in range(episodes):
    state = env.reset()
    action = agent.get_action(state)
    reward = env.step(action)
    agent.update(state, action, reward, state)  # next_state is not used in this case

print("Learned Q-table:")
print(agent.q_table)
```

Slide 6: Interpreting Q-Learning Results

After training our Q-learning agent on the Monty Hall problem, we can interpret the resulting Q-table to understand the learned strategy. The Q-table represents the expected rewards for each action (stay or switch) given the initial door choice.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_q_table(q_table):
    plt.figure(figsize=(10, 6))
    sns.heatmap(q_table, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=['Stay', 'Switch'], yticklabels=['Door 1', 'Door 2', 'Door 3'])
    plt.title('Q-table: Expected Rewards for Each Action')
    plt.xlabel('Action')
    plt.ylabel('Initial Door Choice')
    plt.show()

# Using the Q-table from the previous slide
q_table = agent.q_table
plot_q_table(q_table)

# Interpret the results
best_actions = q_table.argmax(axis=1)
print("Best action for each initial choice:")
for i, action in enumerate(best_actions):
    print(f"Door {i+1}: {'Switch' if action == 1 else 'Stay'}")
```

Slide 7: Monty Hall and Bayesian Inference

The Monty Hall problem can be approached using Bayesian inference, which is a fundamental concept in machine learning for updating probabilities based on new evidence. Let's implement a Bayesian update for the Monty Hall problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def bayesian_monty_hall():
    # Prior probabilities
    prior = np.array([1/3, 1/3, 1/3])
    
    # Likelihood of Monty opening each door given the car's location
    likelihood = np.array([
        [0, 1/2, 1/2],  # Car in door 1
        [1, 0, 1/2],    # Car in door 2
        [1, 1/2, 0]     # Car in door 3
    ])
    
    # Assume Monty opens door 3
    evidence = np.array([0, 0, 1])
    
    # Calculate posterior probabilities
    posterior = prior * likelihood[:, 2]
    posterior /= posterior.sum()
    
    return prior, posterior

prior, posterior = bayesian_monty_hall()

plt.figure(figsize=(10, 6))
plt.bar(range(1, 4), prior, alpha=0.5, label='Prior')
plt.bar(range(1, 4), posterior, alpha=0.5, label='Posterior')
plt.title('Bayesian Update in Monty Hall Problem')
plt.xlabel('Door Number')
plt.ylabel('Probability')
plt.legend()
plt.xticks(range(1, 4))
plt.ylim(0, 1)

for i, (p, q) in enumerate(zip(prior, posterior)):
    plt.text(i+1, p, f'{p:.2f}', ha='center', va='bottom')
    plt.text(i+1, q, f'{q:.2f}', ha='center', va='top')

plt.show()
```

Slide 8: Monty Hall in Decision Trees

Decision trees, a popular machine learning algorithm, can be used to model the Monty Hall problem. We'll create a simple decision tree to illustrate the problem's structure and decision-making process.

```python
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n_samples = 1000
X = np.random.randint(0, 3, (n_samples, 2))  # Initial choice, Monty's choice
y = np.array([1 if (i != j and i != k and j != k) else 0 
              for i, j, k in zip(X[:, 0], X[:, 1], np.random.randint(0, 3, n_samples))])

# Train decision tree
clf = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Plot decision tree
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, feature_names=['Initial Choice', "Monty's Choice"], 
               class_names=['Stay', 'Switch'], filled=True, rounded=True)
plt.title("Decision Tree for Monty Hall Problem")
plt.show()

# Print feature importances
print("Feature importances:")
for name, importance in zip(['Initial Choice', "Monty's Choice"], clf.feature_importances_):
    print(f"{name}: {importance:.4f}")
```

Slide 9: Neural Network Approach

While the Monty Hall problem doesn't require the complexity of a neural network, we can use it to demonstrate how neural networks can learn decision-making strategies. We'll create a simple neural network to predict whether switching doors is beneficial.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
n_samples = 10000
X = np.random.randint(0, 3, (n_samples, 2))  # Initial choice, Monty's choice
y = np.array([1 if (i != j and i != k and j != k) else 0 
              for i, j, k in zip(X[:, 0], X[:, 1], np.random.randint(0, 3, n_samples))])

# Create and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, y, epochs=50, validation_split=0.2, verbose=0)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test the model
test_data = np.array([[0, 1], [1, 2], [2, 0]])
predictions = model.predict(test_data)

print("Predictions (probability of switching being beneficial):")
for initial, monty, pred in zip(test_data[:, 0], test_data[:, 1], predictions):
    print(f"Initial choice: {initial}, Monty's choice: {monty}, Prediction: {pred[0]:.4f}")
```

Slide 10: Real-Life Example: A/B Testing

The Monty Hall problem shares similarities with A/B testing, a common practice in machine learning for comparing two versions of a product or service. Let's simulate an A/B test scenario inspired by the Monty Hall problem.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def ab_test_simulation(n_trials):
    # Strategy A: Always stay (equivalent to not switching in Monty Hall)
    strategy_a = np.random.choice([0, 1], n_trials, p=[2/3, 1/3])
    
    # Strategy B: Always switch (equivalent to always switching in Monty Hall)
    strategy_b = np.random.choice([0, 1], n_trials, p=[1/3, 2/3])
    
    return strategy_a, strategy_b

n_trials = 1000
strategy_a, strategy_b = ab_test_simulation(n_trials)

# Calculate conversion rates
conv_rate_a = np.mean(strategy_a)
conv_rate_b = np.mean(strategy_b)

# Perform t-test
t_stat, p_value = stats.ttest_ind(strategy_a, strategy_b)

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(['Strategy A (Stay)', 'Strategy B (Switch)'], [conv_rate_a, conv_rate_b])
plt.title('A/B Test Results: Monty Hall Strategies')
plt.ylabel('Conversion Rate')
plt.ylim(0, 1)

for i, rate in enumerate([conv_rate_a, conv_rate_b]):
    plt.text(i, rate + 0.01, f'{rate:.3f}', ha='center')

plt.text(0.5, 0.8, f'p-value: {p_value:.4f}', ha='center', transform=plt.gca().transAxes)

plt.show()

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
```

Slide 11: Real-Life Example: Anomaly Detection

The Monty Hall problem can be related to anomaly detection in machine learning. In this example, we'll use the concept of unexpected outcomes (similar to the counterintuitive nature of the Monty Hall problem) to detect anomalies in a dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generate normal data (representing expected outcomes)
np.random.seed(42)
X_normal = np.random.normal(loc=0.33, scale=0.1, size=(1000, 1))

# Generate anomalies (representing unexpected outcomes, like always switching)
X_anomalies = np.random.normal(loc=0.66, scale=0.1, size=(50, 1))

# Combine the datasets
X = np.vstack((X_normal, X_anomalies))

# Train an Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
y_pred = clf.fit_predict(X)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(range(len(X)), X[y_pred == 1], c='blue', label='Normal', alpha=0.5)
plt.scatter(range(len(X)), X[y_pred == -1], c='red', label='Anomaly', alpha=0.5)
plt.title('Anomaly Detection in Monty Hall-like Outcomes')
plt.xlabel('Sample Index')
plt.ylabel('Outcome Probability')
plt.legend()
plt.show()

print(f"Number of detected anomalies: {sum(y_pred == -1)}")
```

Slide 12: Monty Hall and Ensemble Learning

Ensemble learning, a powerful technique in machine learning, can be applied to the Monty Hall problem. We'll create an ensemble of "weak learners" to make decisions about switching doors.

```python
import numpy as np
import matplotlib.pyplot as plt

class MontyHallWeakLearner:
    def __init__(self, switch_prob):
        self.switch_prob = switch_prob
    
    def predict(self, n_games):
        return np.random.choice([0, 1], size=n_games, p=[1-self.switch_prob, self.switch_prob])

class MontyHallEnsemble:
    def __init__(self, n_learners):
        self.learners = [MontyHallWeakLearner(np.random.uniform(0, 1)) for _ in range(n_learners)]
    
    def predict(self, n_games):
        predictions = np.array([learner.predict(n_games) for learner in self.learners])
        return np.mean(predictions, axis=0) > 0.5

# Create and evaluate the ensemble
n_learners = 50
n_games = 10000
ensemble = MontyHallEnsemble(n_learners)
decisions = ensemble.predict(n_games)

# Simulate Monty Hall games
wins = np.sum((np.random.random(n_games) > 1/3) == decisions)
win_rate = wins / n_games

plt.figure(figsize=(10, 6))
plt.bar(['Ensemble Strategy', 'Always Switch', 'Always Stay'], [win_rate, 2/3, 1/3])
plt.title('Monty Hall Ensemble Performance')
plt.ylabel('Win Rate')
plt.ylim(0, 1)

for i, rate in enumerate([win_rate, 2/3, 1/3]):
    plt.text(i, rate + 0.01, f'{rate:.3f}', ha='center')

plt.show()

print(f"Ensemble win rate: {win_rate:.3f}")
```

Slide 13: Monty Hall and Reinforcement Learning: Q-Learning

We can model the Monty Hall problem as a reinforcement learning task using Q-learning. This approach allows an agent to learn the optimal strategy through trial and error.

```python
import numpy as np
import matplotlib.pyplot as plt

class MontyHallEnvironment:
    def step(self, action):
        # 0: stay, 1: switch
        reward = np.random.choice([0, 1], p=[2/3, 1/3] if action == 0 else [1/3, 2/3])
        return reward

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros(2)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
    
    def get_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.q_table)
    
    def update(self, action, reward):
        self.q_table[action] += self.lr * (reward - self.q_table[action])

# Training
env = MontyHallEnvironment()
agent = QLearningAgent()

n_episodes = 10000
rewards = []

for _ in range(n_episodes):
    action = agent.get_action()
    reward = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(rewards) / (np.arange(n_episodes) + 1))
plt.title('Q-Learning Agent Performance in Monty Hall Problem')
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.show()

print("Final Q-values:")
print(f"Stay: {agent.q_table[0]:.3f}")
print(f"Switch: {agent.q_table[1]:.3f}")
```

Slide 14: Conclusion and Future Directions

The Monty Hall problem serves as an excellent example of how counterintuitive probabilistic reasoning can be applied to machine learning. We've explored various ML techniques, including reinforcement learning, Bayesian inference, neural networks, and ensemble methods, all in the context of this classic problem.

Future research directions could include:

1. Exploring more complex variations of the Monty Hall problem with additional doors or multiple rounds.
2. Applying advanced reinforcement learning algorithms like Deep Q-Networks to solve generalized Monty Hall-like problems.
3. Investigating the relationship between the Monty Hall problem and other areas of ML, such as active learning or multi-armed bandit problems.

By studying such problems, we can gain insights into decision-making processes and improve our understanding of probability in machine learning contexts.

Slide 15: Additional Resources

For those interested in diving deeper into the Monty Hall problem and its applications in machine learning, here are some valuable resources:

1. Rosenhouse, J. (2009). The Monty Hall Problem: The Remarkable Story of Math's Most Contentious Brain Teaser. Oxford University Press.
2. Grinstead, C. M., & Snell, J. L. (2006). Introduction to Probability. American Mathematical Society.
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.
4. ArXiv paper: "The Monty Hall problem revisited: Autonomous learning of domain knowledge" URL: [https://arxiv.org/abs/1909.03099](https://arxiv.org/abs/1909.03099)
5. ArXiv paper: "Generalized Monty Hall Problem and Its Applications in Reinforcement Learning" URL: [https://arxiv.org/abs/2006.09433](https://arxiv.org/abs/2006.09433)

These resources provide a mix of theoretical background and practical applications, helping to bridge the gap between the classic Monty Hall problem and modern machine learning techniques.

