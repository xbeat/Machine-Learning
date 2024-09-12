## Model-Based Reinforcement Learning with Python
Slide 1: Introduction to Model-Based Reinforcement Learning

Model-Based Reinforcement Learning (MBRL) is an approach in machine learning where an agent learns a model of its environment to make decisions. This method allows for more efficient learning and better generalization compared to model-free approaches. In MBRL, the agent builds an internal representation of the world, which it uses to plan and predict outcomes of its actions.

```python
import numpy as np
import matplotlib.pyplot as plt

class Environment:
    def __init__(self):
        self.state = 0

    def step(self, action):
        self.state += action
        return self.state, -abs(self.state)  # State and reward

class Model:
    def __init__(self):
        self.weights = np.random.randn(2)

    def predict(self, state, action):
        return np.dot(self.weights, [state, action])

    def update(self, state, action, next_state):
        prediction = self.predict(state, action)
        self.weights += 0.1 * (next_state - prediction) * np.array([state, action])

env = Environment()
model = Model()

states, predictions = [], []
for _ in range(100):
    state = env.state
    action = np.random.uniform(-1, 1)
    next_state, _ = env.step(action)
    
    model.update(state, action, next_state)
    
    states.append(state)
    predictions.append(model.predict(state, action))

plt.plot(states, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('Model-Based RL: Actual vs Predicted States')
plt.show()
```

Slide 2: Components of Model-Based Reinforcement Learning

MBRL consists of three main components: the environment model, the policy, and the planner. The environment model learns to predict state transitions and rewards. The policy determines the agent's actions based on its current state. The planner uses the model to simulate future outcomes and choose the best course of action.

```python
import gym
import numpy as np
from sklearn.linear_model import LinearRegression

class ModelBasedAgent:
    def __init__(self, env):
        self.env = env
        self.state_model = LinearRegression()
        self.reward_model = LinearRegression()
        self.experiences = []

    def collect_experience(self, num_episodes=10):
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.experiences.append((state, action, next_state, reward))
                state = next_state

    def train_models(self):
        states, actions, next_states, rewards = zip(*self.experiences)
        X = np.column_stack((states, actions))
        self.state_model.fit(X, next_states)
        self.reward_model.fit(X, rewards)

    def plan(self, state, num_steps=10):
        best_action = None
        best_return = float('-inf')

        for action in range(self.env.action_space.n):
            total_return = 0
            current_state = state

            for _ in range(num_steps):
                next_state = self.state_model.predict([[current_state, action]])[0]
                reward = self.reward_model.predict([[current_state, action]])[0]
                total_return += reward
                current_state = next_state

            if total_return > best_return:
                best_return = total_return
                best_action = action

        return best_action

# Usage
env = gym.make('CartPole-v1')
agent = ModelBasedAgent(env)
agent.collect_experience()
agent.train_models()

state = env.reset()
for _ in range(200):
    action = agent.plan(state)
    state, _, done, _ = env.step(action)
    if done:
        break

env.close()
```

Slide 3: Environment Model Learning

The environment model is a crucial component in MBRL. It learns to predict the next state and reward given the current state and action. This model can be implemented using various machine learning techniques, such as neural networks or Gaussian processes.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + 1)  # Predict next state and reward
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

# Example usage
state_dim, action_dim = 4, 2
model = EnvironmentModel(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for _ in range(1000):
    state = torch.randn(1, state_dim)
    action = torch.randn(1, action_dim)
    target = torch.randn(1, state_dim + 1)  # Next state and reward

    prediction = model(state, action)
    loss = nn.MSELoss()(prediction, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())
```

Slide 4: Policy Learning in Model-Based RL

In MBRL, the policy is often learned using the environment model. This allows the agent to improve its policy without directly interacting with the real environment, which can be more sample-efficient. The policy can be updated using techniques like policy gradient methods or Q-learning.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        return self.network(state)

class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)

# Example usage
state_dim, action_dim = 4, 2
policy = Policy(state_dim, action_dim)
value_function = ValueFunction(state_dim)
optimizer = optim.Adam(list(policy.parameters()) + list(value_function.parameters()), lr=0.001)

# Training loop (assuming we have an environment model)
for _ in range(1000):
    state = torch.randn(1, state_dim)
    action_probs = policy(state)
    action = torch.multinomial(action_probs, 1)
    value = value_function(state)

    # Use environment model to get next state and reward
    next_state = torch.randn(1, state_dim)  # Placeholder
    reward = torch.randn(1)  # Placeholder

    next_value = value_function(next_state)
    advantage = reward + 0.99 * next_value.detach() - value

    policy_loss = -torch.log(action_probs.gather(1, action)) * advantage.detach()
    value_loss = advantage.pow(2)

    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())
```

Slide 5: Planning in Model-Based RL

Planning is a key aspect of MBRL. The agent uses its learned model to simulate possible future trajectories and choose the best action. Common planning techniques include Monte Carlo Tree Search (MCTS) and Model Predictive Control (MPC).

```python
import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

def mcts(root, model, num_simulations=1000):
    for _ in range(num_simulations):
        node = root
        path = [node]

        # Selection
        while node.children and not model.is_terminal(node.state):
            action, node = max(node.children.items(), key=lambda x: ucb_score(x[1]))
            path.append(node)

        # Expansion
        if not model.is_terminal(node.state):
            action = model.get_random_action(node.state)
            next_state = model.step(node.state, action)
            child = Node(next_state, parent=node)
            node.children[action] = child
            node = child
            path.append(node)

        # Simulation
        value = model.rollout(node.state)

        # Backpropagation
        for node in reversed(path):
            node.visits += 1
            node.value += value

    return max(root.children.items(), key=lambda x: x[1].visits)[0]

def ucb_score(node, c=1.41):
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + c * np.sqrt(np.log(node.parent.visits) / node.visits)

# Example usage (assuming we have a model)
class DummyModel:
    def is_terminal(self, state):
        return np.random.random() < 0.1

    def get_random_action(self, state):
        return np.random.randint(4)

    def step(self, state, action):
        return state + action

    def rollout(self, state):
        return np.random.random()

model = DummyModel()
root = Node(state=0)
best_action = mcts(root, model)
print("Best action:", best_action)
```

Slide 6: Dyna-Q: Integrating Model-Based and Model-Free RL

Dyna-Q is an algorithm that combines model-based and model-free approaches. It uses real experiences to learn both a model of the environment and a Q-function. The learned model is then used to generate additional simulated experiences to update the Q-function.

```python
import numpy as np

class DynaQ:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, planning_steps=50):
        self.q_table = np.zeros((n_states, n_actions))
        self.model = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.planning_steps = planning_steps

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        # Q-learning update
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

        # Model learning
        self.model[(state, action)] = (reward, next_state)

        # Planning
        for _ in range(self.planning_steps):
            s, a = list(self.model.keys())[np.random.randint(len(self.model))]
            r, next_s = self.model[(s, a)]
            
            td_target = r + self.gamma * np.max(self.q_table[next_s])
            td_error = td_target - self.q_table[s, a]
            self.q_table[s, a] += self.lr * td_error

# Example usage
agent = DynaQ(n_states=10, n_actions=4)

for episode in range(1000):
    state = 0  # Initial state
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state = np.random.randint(10)  # Dummy environment
        reward = np.random.random()
        done = np.random.random() < 0.1

        agent.learn(state, action, reward, next_state)
        state = next_state

print("Final Q-table:")
print(agent.q_table)
```

Slide 7: Challenges in Model-Based Reinforcement Learning

MBRL faces several challenges, including model bias, computational complexity, and the exploration-exploitation trade-off. Model bias occurs when the learned model doesn't accurately represent the real environment, leading to suboptimal policies.

```python
import numpy as np
import matplotlib.pyplot as plt

class BiasedModel:
    def __init__(self, true_param, bias):
        self.true_param = true_param
        self.bias = bias
        self.estimated_param = true_param + np.random.normal(0, bias)

    def predict(self, x):
        return self.estimated_param * x

    def true_function(self, x):
        return self.true_param * x

# Create a biased model
true_param = 2.0
bias = 0.5
model = BiasedModel(true_param, bias)

# Generate data
x = np.linspace(0, 10, 100)
y_true = model.true_function(x)
y_pred = model.predict(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, label='True Function')
plt.plot(x, y_pred, label='Model Prediction')
plt.fill_between(x, y_true, y_pred, alpha=0.3, label='Model Bias')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Model Bias in MBRL')
plt.legend()
plt.show()

print(f"True parameter: {true_param}")
print(f"Estimated parameter: {model.estimated_param:.2f}")
print(f"Bias: {model.estimated_param - true_param:.2f}")
```

Slide 8: Exploration Strategies in Model-Based RL

Efficient exploration is crucial in MBRL to gather informative data for model learning. Common strategies include epsilon-greedy, upper confidence bound (UCB), and intrinsic motivation approaches. These methods help balance the exploration-exploitation trade-off, ensuring the agent discovers optimal policies while refining its knowledge of the environment.

```python
import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.true_values = np.random.normal(0, 1, n_arms)

    def pull(self, arm):
        return np.random.normal(self.true_values[arm], 1)

class UCBAgent:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def choose_arm(self, t):
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_values = self.values + np.sqrt(2 * np.log(t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward

# Simulation
env = BanditEnvironment(n_arms=10)
agent = UCBAgent(n_arms=10)
n_rounds = 1000
rewards = []

for t in range(1, n_rounds + 1):
    arm = agent.choose_arm(t)
    reward = env.pull(arm)
    agent.update(arm, reward)
    rewards.append(reward)

plt.plot(np.cumsum(rewards) / np.arange(1, n_rounds + 1))
plt.xlabel('Rounds')
plt.ylabel('Average Reward')
plt.title('UCB Agent Performance')
plt.show()
```

Slide 9: Model Predictive Control in MBRL

Model Predictive Control (MPC) is a popular planning method in MBRL. It uses the learned model to predict future states and optimize actions over a finite horizon. This approach allows for real-time decision-making while adapting to changing environments.

```python
import numpy as np

class SimpleModel:
    def predict(self, state, action):
        return state + action

def mpc_planning(model, initial_state, horizon=5, num_samples=100):
    best_sequence = None
    best_return = float('-inf')

    for _ in range(num_samples):
        state = initial_state
        action_sequence = np.random.uniform(-1, 1, horizon)
        total_return = 0

        for action in action_sequence:
            next_state = model.predict(state, action)
            reward = -abs(next_state)  # Simple reward function
            total_return += reward
            state = next_state

        if total_return > best_return:
            best_return = total_return
            best_sequence = action_sequence

    return best_sequence[0]  # Return the first action of the best sequence

# Example usage
model = SimpleModel()
initial_state = 0

for t in range(20):
    action = mpc_planning(model, initial_state)
    next_state = model.predict(initial_state, action)
    print(f"Step {t}: State = {initial_state:.2f}, Action = {action:.2f}, Next State = {next_state:.2f}")
    initial_state = next_state
```

Slide 10: Ensemble Methods in Model-Based RL

Ensemble methods combine multiple models to improve prediction accuracy and robustness. In MBRL, ensembles can help mitigate model bias and uncertainty, leading to more reliable planning and decision-making.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

class EnsembleModel:
    def __init__(self, num_models):
        self.models = [LinearRegression() for _ in range(num_models)]

    def fit(self, X, y):
        for model in self.models:
            indices = np.random.choice(len(X), len(X), replace=True)
            model.fit(X[indices], y[indices])

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

# Train ensemble model
ensemble = EnsembleModel(num_models=5)
ensemble.fit(X, y)

# Make predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_mean, y_std = ensemble.predict(X_test)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X_test, y_mean, 'r-', label='Mean Prediction')
plt.fill_between(X_test.ravel(), y_mean.ravel() - 2*y_std.ravel(), 
                 y_mean.ravel() + 2*y_std.ravel(), alpha=0.2, label='Uncertainty')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Ensemble Model Predictions')
plt.legend()
plt.show()
```

Slide 11: Real-Life Example: Robotic Arm Control

Model-Based RL can be applied to control robotic arms in manufacturing. The agent learns a model of the arm's dynamics and uses it to plan precise movements for tasks like assembly or pick-and-place operations.

```python
import numpy as np
import matplotlib.pyplot as plt

class RoboticArm:
    def __init__(self, length1, length2):
        self.l1 = length1
        self.l2 = length2

    def forward_kinematics(self, theta1, theta2):
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return x, y

class ArmModel:
    def __init__(self):
        self.arm = RoboticArm(1.0, 0.8)

    def predict(self, state, action):
        theta1, theta2 = state
        dtheta1, dtheta2 = action
        new_theta1 = theta1 + dtheta1
        new_theta2 = theta2 + dtheta2
        x, y = self.arm.forward_kinematics(new_theta1, new_theta2)
        return np.array([new_theta1, new_theta2]), np.array([x, y])

def mpc_control(model, current_state, target_position, horizon=10, num_samples=1000):
    best_sequence = None
    best_distance = float('inf')

    for _ in range(num_samples):
        state = current_state
        action_sequence = np.random.uniform(-0.1, 0.1, (horizon, 2))
        total_distance = 0

        for action in action_sequence:
            state, position = model.predict(state, action)
            distance = np.linalg.norm(position - target_position)
            total_distance += distance

        if total_distance < best_distance:
            best_distance = total_distance
            best_sequence = action_sequence

    return best_sequence[0]

# Example usage
model = ArmModel()
current_state = np.array([np.pi/4, np.pi/4])
target_position = np.array([1.5, 0.5])

positions = []
for _ range(20):
    action = mpc_control(model, current_state, target_position)
    current_state, position = model.predict(current_state, action)
    positions.append(position)

positions = np.array(positions)
plt.figure(figsize=(8, 8))
plt.plot(positions[:, 0], positions[:, 1], 'bo-')
plt.plot(target_position[0], target_position[1], 'r*', markersize=15)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robotic Arm Trajectory')
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 12: Real-Life Example: Autonomous Drone Navigation

MBRL can be used for autonomous drone navigation in complex environments. The drone learns a model of its flight dynamics and the environment, then uses this model to plan safe and efficient paths to its destination.

```python
import numpy as np
import matplotlib.pyplot as plt

class Drone:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

    def update(self, acceleration, dt=0.1):
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

class DroneModel:
    def predict(self, state, action, dt=0.1):
        position, velocity = state[:3], state[3:]
        new_velocity = velocity + action * dt
        new_position = position + new_velocity * dt
        return np.concatenate([new_position, new_velocity])

def mpc_planning(model, initial_state, target, horizon=10, num_samples=1000):
    best_sequence = None
    best_distance = float('inf')

    for _ in range(num_samples):
        state = initial_state
        action_sequence = np.random.uniform(-1, 1, (horizon, 3))
        total_distance = 0

        for action in action_sequence:
            state = model.predict(state, action)
            distance = np.linalg.norm(state[:3] - target)
            total_distance += distance

        if total_distance < best_distance:
            best_distance = total_distance
            best_sequence = action_sequence

    return best_sequence[0]

# Example usage
drone = Drone()
model = DroneModel()
target = np.array([5.0, 5.0, 5.0])

trajectory = [drone.position]
for _ in range(100):
    state = np.concatenate([drone.position, drone.velocity])
    action = mpc_planning(model, state, target)
    drone.update(action)
    trajectory.append(drone.position)

trajectory = np.array(trajectory)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'bo-')
ax.plot([target[0]], [target[1]], [target[2]], 'r*', markersize=15)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Drone Navigation Trajectory')
plt.show()
```

Slide 13: Future Directions in Model-Based RL

Model-Based Reinforcement Learning continues to evolve, with several promising research directions. These include improving sample efficiency, developing more accurate and robust environment models, and integrating MBRL with other AI techniques such as meta-learning and transfer learning.

```python
import numpy as np
import matplotlib.pyplot as plt

def sample_efficiency_comparison(num_episodes):
    model_free_performance = np.log(1 + np.arange(num_episodes))
    model_based_performance = np.sqrt(1 + np.arange(num_episodes))
    
    plt.figure(figsize=(10, 6))
    plt.plot(model_free_performance, label='Model-Free RL')
    plt.plot(model_based_performance, label='Model-Based RL')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Performance')
    plt.title('Sample Efficiency: Model-Based vs Model-Free RL')
    plt.legend()
    plt.grid(True)
    plt.show()

sample_efficiency_comparison(1000)

def future_performance_projection(years):
    current_performance = 100
    model_free_growth = np.array([current_performance * (1.1 ** year) for year in range(years)])
    model_based_growth = np.array([current_performance * (1.2 ** year) for year in range(years)])
    hybrid_growth = np.array([current_performance * (1.25 ** year) for year in range(years)])

    plt.figure(figsize=(10, 6))
    plt.plot(model_free_growth, label='Model-Free RL')
    plt.plot(model_based_growth, label='Model-Based RL')
    plt.plot(hybrid_growth, label='Hybrid Approaches')
    plt.xlabel('Years from Now')
    plt.ylabel('Relative Performance')
    plt.title('Projected Advancements in RL Approaches')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

future_performance_projection(10)
```

Slide 14: Additional Resources

For those interested in diving deeper into Model-Based Reinforcement Learning, here are some valuable resources:

1. "Model-Based Reinforcement Learning: A Survey" by T. Wang et al. (2019) ArXiv: [https://arxiv.org/abs/2006.16712](https://arxiv.org/abs/2006.16712)
2. "Benchmarking Model-Based Reinforcement Learning" by Y. Wang et al. (2019) ArXiv: [https://arxiv.org/abs/1907.02057](https://arxiv.org/abs/1907.02057)
3. "When to Trust Your Model: Model-Based Policy Optimization" by M. Janner et al. (2019) ArXiv: [https://arxiv.org/abs/1906.08253](https://arxiv.org/abs/1906.08253)
4. "Dream to Control: Learning Behaviors by Latent Imagination" by D. Hafner et al. (2019) ArXiv: [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603)

These papers provide comprehensive overviews, benchmarks, and state-of-the-art techniques in MBRL. They offer valuable insights into the current landscape and future directions of this exciting field.

