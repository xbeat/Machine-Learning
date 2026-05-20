## Reinforcement Learning with Python Code Examples
Slide 1: Fundamentals of Reinforcement Learning

Reinforcement learning operates on the principle of agents learning optimal behavior through interactions with an environment. The agent performs actions, receives rewards or penalties, and updates its policy to maximize cumulative rewards over time. This fundamental concept forms the basis of all RL algorithms.

```python
class RLEnvironment:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.current_state = 0
        
    def step(self, action):
        # Simulate environment transition
        next_state = (self.current_state + action) % len(self.states)
        reward = 1 if next_state > self.current_state else -1
        self.current_state = next_state
        return next_state, reward, False
        
    def reset(self):
        self.current_state = 0
        return self.current_state

# Example usage
env = RLEnvironment(states=range(5), actions=[-1, 0, 1])
state = env.reset()
next_state, reward, done = env.step(1)
print(f"State: {state} -> Action: 1 -> Next State: {next_state}, Reward: {reward}")
```

Slide 2: Q-Learning Algorithm Implementation

Q-Learning is a model-free reinforcement learning algorithm that learns to make optimal decisions by maintaining a Q-table of state-action values. The algorithm updates these values based on the Bellman equation, gradually improving its policy through experience.

```python
import numpy as np

class QLearning:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.95):
        self.q_table = np.zeros((states, actions))
        self.lr = learning_rate
        self.gamma = discount_factor

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning update formula
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
```

Slide 3: SARSA Implementation

SARSA (State-Action-Reward-State-Action) is an on-policy learning algorithm that differs from Q-learning by using the actual next action instead of the maximum Q-value for updates. This makes it more conservative in risky environments.

```python
class SARSA:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.95):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = gamma
    
    def update(self, state, action, reward, next_state, next_action):
        # SARSA update formula
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
        self.q_table[state, action] = new_q
        
    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
```

Slide 4: Deep Q-Network Architecture

Deep Q-Networks (DQN) combine Q-learning with deep neural networks to handle high-dimensional state spaces. This implementation includes experience replay and target networks to stabilize training and improve convergence in complex environments.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
```

Slide 5: Training Deep Q-Networks

The DQN training process involves iteratively sampling experiences from the replay buffer and updating the network weights using gradient descent. This implementation showcases the core training loop with target network updates and epsilon-greedy exploration.

```python
class DQNTrainer:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, 128, action_dim)
        self.target_net = DQN(state_dim, 128, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer()
        
    def train_step(self, batch_size=32, gamma=0.99):
        if len(self.memory) < batch_size:
            return
        
        # Sample transitions from memory
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        states = torch.FloatTensor(batch[0])
        actions = torch.LongTensor(batch[1])
        rewards = torch.FloatTensor(batch[2])
        next_states = torch.FloatTensor(batch[3])
        dones = torch.FloatTensor(batch[4])
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

Slide 6: Policy Gradient Methods

Policy gradient methods directly learn the policy function by optimizing the expected cumulative reward using gradient ascent. This implementation demonstrates the REINFORCE algorithm, a fundamental policy gradient method.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.policy = PolicyNetwork(state_dim, 128, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[action]
    
    def update(self, rewards, log_probs):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
```

Slide 7: Actor-Critic Architecture

Actor-Critic combines policy gradient methods with value function approximation. The actor learns the policy while the critic evaluates the policy through value estimation, reducing variance in policy updates.

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        value = self.critic(state)
        policy_dist = self.actor(state)
        return value, policy_dist

class A2CTrainer:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.ac_net = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        
    def compute_returns(self, rewards, values, gamma=0.99):
        returns = []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + gamma * R
            returns.insert(0, R - v.item())
        return returns
```

Slide 8: Proximal Policy Optimization (PPO)

Proximal Policy Optimization implements a clipped objective function to constrain policy updates, ensuring stable learning by preventing excessive policy changes while maintaining good sample efficiency and performance.

```python
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

class PPO:
    def __init__(self, state_dim, action_dim, clip_ratio=0.2, lr=3e-4):
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
    
    def update(self, states, actions, advantages, old_probs, returns):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advantages = torch.FloatTensor(advantages)
        old_probs = torch.FloatTensor(old_probs)
        returns = torch.FloatTensor(returns)
        
        # PPO policy loss computation
        new_probs, values = self.network(states)
        ratio = new_probs / old_probs
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Value loss computation
        value_loss = ((returns - values) ** 2).mean()
        
        # Combined loss and update
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

Slide 9: Multi-Agent Reinforcement Learning

Multi-agent reinforcement learning extends single-agent concepts to scenarios where multiple agents interact, compete, or cooperate. This implementation shows a basic framework for independent Q-learning in a multi-agent setting.

```python
class MultiAgentQLearning:
    def __init__(self, n_agents, state_dim, action_dim):
        self.agents = [QLearning(state_dim, action_dim) for _ in range(n_agents)]
        
    def select_actions(self, states, epsilon=0.1):
        return [agent.get_action(state, epsilon) 
                for agent, state in zip(self.agents, states)]
    
    def update_all(self, experiences):
        # experiences: list of (state, action, reward, next_state) tuples
        for agent_idx, (state, action, reward, next_state) in enumerate(experiences):
            self.agents[agent_idx].update(state, action, reward, next_state)

# Example usage with environment
class MultiAgentEnv:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.states = np.zeros((n_agents, 2))  # 2D state space
        
    def step(self, actions):
        next_states = np.zeros_like(self.states)
        rewards = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            # Simulate agent movement and interaction
            next_states[i] = self.states[i] + actions[i]
            rewards[i] = self._compute_reward(i, next_states)
            
        self.states = next_states
        return next_states, rewards
        
    def _compute_reward(self, agent_idx, states):
        # Example reward function based on distance to other agents
        distances = np.linalg.norm(states - states[agent_idx], axis=1)
        return -np.mean(distances[distances > 0])
```

Slide 10: Deep Deterministic Policy Gradient (DDPG)

DDPG combines insights from DQN and deterministic policy gradients to handle continuous action spaces. This algorithm uses actor-critic architecture with target networks and experience replay for stable learning in continuous control tasks.

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.network = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.max_action * self.network(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
    
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=1))

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
```

Slide 11: Implementing Prioritized Experience Replay

Prioritized Experience Replay enhances learning efficiency by sampling important transitions more frequently. This implementation uses a sum-tree data structure to efficiently store and sample transitions based on their TD-error priorities.

```python
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
        
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
```

Slide 12: Advanced Policy Optimization with Trust Region

Trust Region Policy Optimization (TRPO) implements sophisticated policy updates using natural gradient descent while respecting KL-divergence constraints between old and new policies.

```python
class TRPO:
    def __init__(self, state_dim, action_dim, max_kl=0.01):
        self.policy = GaussianMLPPolicy(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.max_kl = max_kl
        
    def compute_surrogate_loss(self, states, actions, advantages, old_dist_info):
        new_dist_info = self.policy.dist_info_sym(states)
        old_prob = old_dist_info.prob(actions)
        new_prob = new_dist_info.prob(actions)
        
        ratio = new_prob / old_prob
        surrogate_loss = -torch.mean(ratio * advantages)
        
        kl = torch.mean(old_dist_info.kl_div(new_dist_info))
        return surrogate_loss, kl
        
    def line_search(self, states, actions, advantages, old_dist_info, descent_direction):
        alpha = 1.0
        max_steps = 10
        
        for _ in range(max_steps):
            new_params = self.policy.get_params() + alpha * descent_direction
            self.policy.set_params(new_params)
            
            loss, kl = self.compute_surrogate_loss(
                states, actions, advantages, old_dist_info
            )
            
            if kl <= self.max_kl and loss < 0:
                return True, alpha
            
            alpha *= 0.5
        
        return False, None
```

Slide 13: Real-world Application - Stock Trading Agent

This implementation demonstrates a complete reinforcement learning solution for automated stock trading, including data preprocessing, custom environment, and a DQN agent optimized for financial markets.

```python
import pandas as pd
import numpy as np

class TradingEnvironment:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self._get_state()
        
    def _get_state(self):
        # Calculate technical indicators
        window = 10
        prices = self.data['close'].values
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[max(0, self.current_step-window):self.current_step])
        
        return np.array([
            self.balance / self.initial_balance,
            self.position,
            volatility,
            returns[self.current_step-1] if self.current_step > 0 else 0
        ])
        
    def step(self, action):  # 0: hold, 1: buy, 2: sell
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        if action == 1 and self.position <= 0:  # Buy
            shares = self.balance // current_price
            cost = shares * current_price
            self.balance -= cost
            self.position += shares
            reward = -cost * 0.001  # Transaction cost
            
        elif action == 2 and self.position >= 0:  # Sell
            revenue = self.position * current_price
            self.balance += revenue
            reward = revenue * 0.001 - current_price * 0.001  # Revenue minus transaction cost
            self.position = 0
            
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done

# Example usage
data = pd.DataFrame({
    'close': [100, 101, 99, 102, 98, 103],
    'volume': [1000, 1200, 900, 1500, 800, 1300]
})

env = TradingEnvironment(data)
trading_agent = DQN(state_dim=4, action_dim=3)  # Using previous DQN implementation
```

Slide 14: Real-world Application - Autonomous Navigation

This implementation demonstrates a practical application of reinforcement learning for autonomous navigation, featuring obstacle avoidance and path planning using the PPO algorithm with continuous action space.

```python
class NavigationEnv:
    def __init__(self, map_size=(100, 100), n_obstacles=10):
        self.map_size = map_size
        self.obstacles = self._generate_obstacles(n_obstacles)
        self.goal = self._generate_goal()
        
    def _generate_obstacles(self, n):
        obstacles = []
        for _ in range(n):
            pos = np.random.randint(0, self.map_size[0], 2)
            size = np.random.randint(5, 15, 2)
            obstacles.append((pos, size))
        return obstacles
        
    def _generate_goal(self):
        while True:
            goal = np.random.randint(0, self.map_size[0], 2)
            if not self._check_collision(goal):
                return goal
                
    def _check_collision(self, pos):
        for obs_pos, obs_size in self.obstacles:
            if (pos[0] >= obs_pos[0] and pos[0] <= obs_pos[0] + obs_size[0] and
                pos[1] >= obs_pos[1] and pos[1] <= obs_pos[1] + obs_size[1]):
                return True
        return False
        
    def step(self, action):
        # action: [dx, dy] continuous values between -1 and 1
        new_pos = self.current_pos + action * 5  # Scale action to actual movement
        
        # Check boundaries
        new_pos = np.clip(new_pos, 0, self.map_size[0]-1)
        
        # Check collision
        if self._check_collision(new_pos):
            return self.current_pos, -1, True
            
        self.current_pos = new_pos
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        reward = -distance_to_goal * 0.01
        
        if distance_to_goal < 5:
            reward += 100
            done = True
        else:
            done = False
            
        return self.current_pos, reward, done
```

Slide 15: Additional Resources

*   "Proximal Policy Optimization Algorithms" - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
*   "Deep Reinforcement Learning with Double Q-learning" - [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)
*   "Continuous Control with Deep Reinforcement Learning" - [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)
*   "Trust Region Policy Optimization" - [https://arxiv.org/abs/1502.05477](https://arxiv.org/abs/1502.05477)
*   Recommended search terms for further exploration:
    *   "Multi-Agent Deep Deterministic Policy Gradient"
    *   "Soft Actor-Critic Implementation"
    *   "Hierarchical Reinforcement Learning"
    *   "Meta-Reinforcement Learning"
*   Additional learning resources:
    *   Reinforcement Learning: An Introduction (Sutton & Barto)
    *   OpenAI Spinning Up documentation
    *   DeepMind's Advanced Deep Learning and Reinforcement Learning course materials

Note: For the most current research and implementations, please search on Google Scholar or arXiv using the topics mentioned above.

