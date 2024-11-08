## Reinforcement Learning for Sequential Decision-Making
Slide 1: Introduction to Reinforcement Learning

Reinforcement learning is a machine learning paradigm focused on training agents to make sequences of decisions. The agent learns to map situations to actions while maximizing a numerical reward signal through interaction with an environment over time.

```python
import gym
import numpy as np

# Create environment
env = gym.make('CartPole-v1')

# Basic Q-learning implementation
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = 0.1
        self.gamma = 0.95
        
    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(self.q_table[state])
```

Slide 2: Markov Decision Processes

MDPs provide the mathematical framework for modeling decision-making problems in reinforcement learning. A Markov process consists of states, actions, transition probabilities, and rewards, forming the basis for value-based learning methods.

```python
# Mathematical representation of MDP components
"""
States (S): Set of all possible states
Actions (A): Set of all possible actions
Transitions P(s'|s,a): Probability of next state given current state and action
Rewards R(s,a,s'): Immediate reward for transition
Value function V(s): Expected cumulative reward starting from state s

$$V(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]$$
"""

class MDP:
    def __init__(self, states, actions, transitions, rewards):
        self.states = states
        self.actions = actions
        self.P = transitions  # P[s][a][s'] = probability
        self.R = rewards      # R[s][a][s'] = reward
```

Slide 3: Value Iteration Algorithm

Value iteration is a dynamic programming algorithm that computes the optimal value function by iteratively updating state values based on the Bellman equation, converging to the optimal policy for solving MDPs.

```python
def value_iteration(mdp, theta=0.001, gamma=0.95):
    V = np.zeros(len(mdp.states))
    
    while True:
        delta = 0
        for s in range(len(mdp.states)):
            v = V[s]
            # Bellman optimality update
            V[s] = max(sum(mdp.P[s][a][s_next] * 
                          (mdp.R[s][a][s_next] + gamma * V[s_next])
                          for s_next in range(len(mdp.states)))
                      for a in range(len(mdp.actions)))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V
```

Slide 4: Q-Learning Implementation

Q-learning is an off-policy TD control algorithm that learns the action-value function directly through bootstrapping and experience replay, making it one of the most popular reinforcement learning algorithms.

```python
class QLearning:
    def __init__(self, state_space, action_space):
        self.q_table = np.zeros((state_space, action_space))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-learning update rule
        new_value = (1 - self.lr) * old_value + \
                   self.lr * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
```

Slide 5: Deep Q-Networks (DQN)

Deep Q-Networks extend traditional Q-learning by using neural networks to approximate the Q-function, enabling learning in environments with continuous state spaces and complex patterns, revolutionizing reinforcement learning applications.

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
        
    def select_action(self, state, epsilon=0.1):
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.output_dim, (1,))
        with torch.no_grad():
            return self.network(state).max(1)[1]
```

Slide 6: Experience Replay Buffer

Experience replay enables more efficient learning by storing and randomly sampling past experiences, breaking temporal correlations and improving sample efficiency in deep reinforcement learning algorithms.

```python
import random
from collections import namedtuple, deque

Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))
    
    def __len__(self):
        return len(self.buffer)
```

Slide 7: Policy Gradient Methods

Policy gradient methods directly optimize the policy by computing gradients with respect to the policy parameters, allowing for learning in continuous action spaces and providing more natural behavior in complex environments.

```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.policy(x)
    
    def get_log_prob(self, state, action):
        probs = self.forward(state)
        return torch.log(probs.squeeze(0)[action])
```

Slide 8: REINFORCE Algorithm Implementation

REINFORCE implements the policy gradient theorem using Monte Carlo sampling to estimate the gradient of the expected return, providing a fundamental approach to policy-based reinforcement learning.

```python
def reinforce(policy_net, optimizer, episodes, gamma=0.99):
    for episode in range(episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        
        while True:
            action_probs = policy_net(torch.FloatTensor(state))
            action = torch.distributions.Categorical(action_probs).sample()
            
            next_state, reward, done, _ = env.step(action.item())
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            if done:
                break
                
            state = next_state
            
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / returns.std()
        
        # Calculate loss and update policy
        loss = 0
        for log_prob, R in zip(action_probs, returns):
            loss -= log_prob * R
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Slide 9: Actor-Critic Architecture

Actor-Critic methods combine policy-based and value-based learning by maintaining both a policy (actor) and a value function (critic). The critic estimates the value function while the actor updates the policy using the critic's feedback.

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        # Returns action probabilities and state value
        return self.actor(state), self.critic(state)
```

Slide 10: Advantage Actor-Critic (A2C) Implementation

A2C improves upon basic actor-critic by using advantage estimation to reduce variance in policy gradients. The advantage function measures how much better an action is compared to the average action in that state.

```python
class A2C:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
    def compute_returns(self, rewards, values, gamma=0.99):
        returns = []
        R = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + gamma * R
            returns.insert(0, R - v.item())
        return torch.tensor(returns)
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        action_probs, values = self.actor_critic(states)
        
        # Calculate advantages
        advantages = self.compute_returns(rewards, values)
        
        # Actor loss
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))
        actor_loss = -(torch.log(selected_action_probs) * advantages).mean()
        
        # Critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Combined loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

Slide 11: Proximal Policy Optimization (PPO)

PPO implements a novel objective function that clips the policy ratio to prevent excessive policy updates, making it one of the most stable and efficient policy gradient algorithms for continuous control tasks.

```python
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.clip_epsilon = 0.2
        
    def compute_ppo_loss(self, states, actions, old_probs, advantages):
        # Current policy probabilities
        current_probs = self.policy(states)
        
        # Probability ratio
        ratio = (current_probs / old_probs)
        
        # Clipped objective function
        clipped_ratio = torch.clamp(ratio, 
                                  1 - self.clip_epsilon, 
                                  1 + self.clip_epsilon)
        
        # PPO loss
        loss = -torch.min(ratio * advantages,
                         clipped_ratio * advantages).mean()
        
        return loss
```

Slide 12: Deep Deterministic Policy Gradient (DDPG)

DDPG combines DQN and actor-critic methods to handle continuous action spaces. It uses deterministic policy gradients and implements target networks to stabilize learning in complex continuous control tasks.

```python
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim + action_dim)
        self.critic_target = Critic(state_dim + action_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.tau = 0.001  # Target network update rate
        
    def update_targets(self):
        # Soft update target networks
        for target, source in zip(self.actor_target.parameters(), 
                                self.actor.parameters()):
            target.data.copy_(
                self.tau * source.data + (1 - self.tau) * target.data
            )
            
        for target, source in zip(self.critic_target.parameters(), 
                                self.critic.parameters()):
            target.data.copy_(
                self.tau * source.data + (1 - self.tau) * target.data
            )
```

Slide 13: Real-world Application - Trading Agent

Implementation of a reinforcement learning agent for automated trading, demonstrating practical application in financial markets using historical price data and custom reward functions.

```python
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
        
    def step(self, action):  # 0: hold, 1: buy, 2: sell
        reward = 0
        done = False
        
        # Execute trade
        current_price = self.data[self.current_step]
        if action == 1 and self.position == 0:  # Buy
            self.position = self.balance / current_price
            self.balance = 0
        elif action == 2 and self.position > 0:  # Sell
            self.balance = self.position * current_price
            self.position = 0
            
        # Calculate reward
        next_price = self.data[self.current_step + 1]
        reward = ((next_price - current_price) / current_price) * \
                 (self.position * current_price + self.balance)
                 
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            
        return self._get_state(), reward, done, {}
        
    def _get_state(self):
        return np.array([
            self.balance,
            self.position,
            self.data[self.current_step]
        ])
```

Slide 14: Results Analysis and Visualization

Implementation of comprehensive evaluation metrics and visualization tools for analyzing reinforcement learning agent performance across different environments and scenarios.

```python
class ResultsAnalyzer:
    def __init__(self):
        self.episode_rewards = []
        self.learning_curves = []
        
    def track_episode(self, rewards):
        self.episode_rewards.append(sum(rewards))
        self.learning_curves.append(
            np.mean(self.episode_rewards[-100:])
        )
        
    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_curves)
        plt.title('Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (100 episodes)')
        plt.show()
```

Slide 15: Additional Resources

*   "Continuous Control with Deep Reinforcement Learning" - [https://arxiv.org/abs/1509.02971](https://arxiv.org/abs/1509.02971)
*   "Proximal Policy Optimization Algorithms" - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)
*   "Deep Reinforcement Learning that Matters" - [https://arxiv.org/abs/1709.06560](https://arxiv.org/abs/1709.06560)
*   "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" - [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)
*   "Benchmarking Deep Reinforcement Learning for Continuous Control" - [https://arxiv.org/abs/1604.06778](https://arxiv.org/abs/1604.06778)

