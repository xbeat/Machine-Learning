## Introduction to Agents and Multi-Agent Frameworks in Python
Slide 1: Introduction to Agents and Multi-Agent Frameworks

Agents are autonomous entities that can perceive their environment, make decisions, and take actions. Multi-agent frameworks provide a structure for multiple agents to interact and collaborate. This slideshow will guide you through the basics of implementing agents and multi-agent systems using Python.

```python
import random

class SimpleAgent:
    def __init__(self, name):
        self.name = name
    
    def perceive(self, environment):
        return environment
    
    def decide(self, perception):
        return random.choice(["move", "stay"])
    
    def act(self, decision):
        print(f"{self.name} decided to {decision}")

# Usage
agent = SimpleAgent("Agent1")
environment = {"obstacle": False, "goal": True}
perception = agent.perceive(environment)
decision = agent.decide(perception)
agent.act(decision)
```

Slide 2: Setting Up the Environment

The environment is a crucial component in agent-based systems. It represents the world in which agents operate and interact. Let's create a simple grid-based environment.

```python
import numpy as np

class GridEnvironment:
    def __init__(self, width, height):
        self.grid = np.zeros((height, width))
        self.width = width
        self.height = height
    
    def add_obstacle(self, x, y):
        self.grid[y, x] = 1
    
    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0

    def display(self):
        for row in self.grid:
            print(" ".join(["#" if cell == 1 else "." for cell in row]))

# Create and display a 5x5 grid environment
env = GridEnvironment(5, 5)
env.add_obstacle(2, 2)
env.display()
```

Slide 3: Implementing a Basic Agent

Now that we have an environment, let's create a more sophisticated agent that can navigate the grid environment.

```python
import random

class GridAgent:
    def __init__(self, x, y, environment):
        self.x = x
        self.y = y
        self.environment = environment
    
    def perceive(self):
        return {
            "up": self.environment.is_valid_position(self.x, self.y - 1),
            "down": self.environment.is_valid_position(self.x, self.y + 1),
            "left": self.environment.is_valid_position(self.x - 1, self.y),
            "right": self.environment.is_valid_position(self.x + 1, self.y)
        }
    
    def decide(self, perception):
        valid_moves = [move for move, is_valid in perception.items() if is_valid]
        return random.choice(valid_moves) if valid_moves else None
    
    def act(self, decision):
        if decision == "up":
            self.y -= 1
        elif decision == "down":
            self.y += 1
        elif decision == "left":
            self.x -= 1
        elif decision == "right":
            self.x += 1
        print(f"Agent moved {decision} to position ({self.x}, {self.y})")

# Usage
env = GridEnvironment(5, 5)
env.add_obstacle(2, 2)
agent = GridAgent(0, 0, env)

for _ in range(5):
    perception = agent.perceive()
    decision = agent.decide(perception)
    agent.act(decision)
```

Slide 4: Implementing a Goal-Oriented Agent

Let's enhance our agent to pursue a specific goal instead of moving randomly.

```python
import math

class GoalOrientedAgent(GridAgent):
    def __init__(self, x, y, environment, goal_x, goal_y):
        super().__init__(x, y, environment)
        self.goal_x = goal_x
        self.goal_y = goal_y
    
    def decide(self, perception):
        valid_moves = [move for move, is_valid in perception.items() if is_valid]
        if not valid_moves:
            return None
        
        # Calculate distances to the goal for each valid move
        distances = {
            "up": math.sqrt((self.x - self.goal_x)**2 + (self.y - 1 - self.goal_y)**2),
            "down": math.sqrt((self.x - self.goal_x)**2 + (self.y + 1 - self.goal_y)**2),
            "left": math.sqrt((self.x - 1 - self.goal_x)**2 + (self.y - self.goal_y)**2),
            "right": math.sqrt((self.x + 1 - self.goal_x)**2 + (self.y - self.goal_y)**2)
        }
        
        # Choose the move that brings the agent closest to the goal
        return min((move for move in valid_moves), key=lambda m: distances[m])

# Usage
env = GridEnvironment(5, 5)
env.add_obstacle(2, 2)
agent = GoalOrientedAgent(0, 0, env, 4, 4)

for _ in range(10):
    perception = agent.perceive()
    decision = agent.decide(perception)
    agent.act(decision)
    if agent.x == agent.goal_x and agent.y == agent.goal_y:
        print("Goal reached!")
        break
```

Slide 5: Introducing Multi-Agent Systems

Multi-agent systems involve multiple agents interacting within the same environment. Let's create a simple multi-agent system with two agents pursuing different goals.

```python
class MultiAgentEnvironment(GridEnvironment):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.agents = []
    
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def step(self):
        for agent in self.agents:
            perception = agent.perceive()
            decision = agent.decide(perception)
            agent.act(decision)
    
    def display(self):
        grid_ = self.grid.()
        for i, agent in enumerate(self.agents):
            grid_[agent.y, agent.x] = i + 2
        for row in grid_:
            print(" ".join(["#" if cell == 1 else f"A{int(cell-1)}" if cell > 1 else "." for cell in row]))

# Usage
env = MultiAgentEnvironment(7, 7)
env.add_obstacle(3, 3)
agent1 = GoalOrientedAgent(0, 0, env, 6, 6)
agent2 = GoalOrientedAgent(6, 0, env, 0, 6)
env.add_agent(agent1)
env.add_agent(agent2)

for _ in range(15):
    env.step()
    env.display()
    print("\n")
```

Slide 6: Implementing Agent Communication

In multi-agent systems, agents often need to communicate with each other. Let's implement a simple communication mechanism.

```python
class CommunicatingAgent(GoalOrientedAgent):
    def __init__(self, x, y, environment, goal_x, goal_y, name):
        super().__init__(x, y, environment, goal_x, goal_y)
        self.name = name
        self.messages = []
    
    def send_message(self, recipient, content):
        recipient.receive_message(self.name, content)
    
    def receive_message(self, sender, content):
        self.messages.append((sender, content))
    
    def process_messages(self):
        for sender, content in self.messages:
            print(f"{self.name} received message from {sender}: {content}")
        self.messages.clear()

# Modify MultiAgentEnvironment to include communication
class CommunicatingEnvironment(MultiAgentEnvironment):
    def step(self):
        for agent in self.agents:
            agent.process_messages()
        super().step()

# Usage
env = CommunicatingEnvironment(7, 7)
agent1 = CommunicatingAgent(0, 0, env, 6, 6, "Agent1")
agent2 = CommunicatingAgent(6, 0, env, 0, 6, "Agent2")
env.add_agent(agent1)
env.add_agent(agent2)

agent1.send_message(agent2, "Hello from Agent1!")
agent2.send_message(agent1, "Greetings, Agent1!")

env.step()
```

Slide 7: Implementing Cooperative Behavior

Let's enhance our agents to cooperate in achieving a common goal, such as finding the shortest path to a target.

```python
import heapq

class CooperativeAgent(CommunicatingAgent):
    def __init__(self, x, y, environment, goal_x, goal_y, name):
        super().__init__(x, y, environment, goal_x, goal_y, name)
        self.known_obstacles = set()
    
    def perceive(self):
        perception = super().perceive()
        for move, is_valid in perception.items():
            if not is_valid:
                if move == "up":
                    self.known_obstacles.add((self.x, self.y - 1))
                elif move == "down":
                    self.known_obstacles.add((self.x, self.y + 1))
                elif move == "left":
                    self.known_obstacles.add((self.x - 1, self.y))
                elif move == "right":
                    self.known_obstacles.add((self.x + 1, self.y))
        return perception
    
    def decide(self, perception):
        path = self.a_star((self.x, self.y), (self.goal_x, self.goal_y))
        if path:
            next_pos = path[1]
            if next_pos[0] > self.x:
                return "right"
            elif next_pos[0] < self.x:
                return "left"
            elif next_pos[1] > self.y:
                return "down"
            else:
                return "up"
        return None
    
    def a_star(self, start, goal):
        def heuristic(a, b):
            return abs(b[0] - a[0]) + abs(b[1] - a[1])
        
        neighbors = [(0,1), (0,-1), (1,0), (-1,0)]
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + 1
                if 0 <= neighbor[0] < self.environment.width and 0 <= neighbor[1] < self.environment.height:
                    if neighbor in self.known_obstacles:
                        continue
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue
                    if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return None

    def process_messages(self):
        for sender, content in self.messages:
            if isinstance(content, set):
                self.known_obstacles.update(content)
            print(f"{self.name} received obstacles from {sender}")
        self.messages.clear()
    
    def act(self, decision):
        super().act(decision)
        for other_agent in self.environment.agents:
            if other_agent != self:
                self.send_message(other_agent, self.known_obstacles)

# Usage
env = CommunicatingEnvironment(10, 10)
env.add_obstacle(5, 5)
env.add_obstacle(5, 6)
env.add_obstacle(6, 5)
agent1 = CooperativeAgent(0, 0, env, 9, 9, "Agent1")
agent2 = CooperativeAgent(9, 0, env, 0, 9, "Agent2")
env.add_agent(agent1)
env.add_agent(agent2)

for _ in range(20):
    env.step()
    env.display()
    print("\n")
```

Slide 8: Implementing a Belief-Desire-Intention (BDI) Agent

The BDI architecture is a popular framework for designing intelligent agents. Let's implement a simple BDI agent.

```python
class BDIAgent:
    def __init__(self, name, environment):
        self.name = name
        self.environment = environment
        self.beliefs = set()
        self.desires = set()
        self.intentions = []
    
    def update_beliefs(self, perception):
        # Update beliefs based on current perception
        self.beliefs = set(perception.items())
    
    def generate_options(self):
        # Generate possible desires based on current beliefs
        self.desires = set()
        for belief, value in self.beliefs:
            if value:
                self.desires.add(belief)
    
    def filter_intentions(self):
        # Choose intentions from desires
        self.intentions = list(self.desires)[:2]  # Limit to top 2 intentions
    
    def execute(self):
        # Execute the current intentions
        for intention in self.intentions:
            print(f"{self.name} is executing intention: {intention}")
    
    def bdi_loop(self):
        perception = self.environment.get_perception()
        self.update_beliefs(perception)
        self.generate_options()
        self.filter_intentions()
        self.execute()

class SimpleEnvironment:
    def __init__(self):
        self.state = {"move": True, "eat": False, "sleep": True}
    
    def get_perception(self):
        return self.state

# Usage
env = SimpleEnvironment()
agent = BDIAgent("BDIAgent", env)

for _ in range(3):
    agent.bdi_loop()
    # Change environment state
    env.state["eat"] = not env.state["eat"]
    print("\n")
```

Slide 9: Implementing a Learning Agent

Let's create an agent that can learn from its experiences using a simple Q-learning algorithm.

```python
import random

class LearningAgent:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {state: {action: 0 for action in actions} for state in states}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.actions = actions
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state][action] = new_q

# Example usage
states = ['A', 'B', 'C']
actions = ['left', 'right']
agent = LearningAgent(states, actions)

# Simulate learning
for _ in range(100):
    state = random.choice(states)
    action = agent.choose_action(state)
    next_state = random.choice(states)
    reward = 1 if next_state == 'C' else 0
    agent.learn(state, action, reward, next_state)

print("Q-table after learning:")
for state in states:
    print(f"{state}: {agent.q_table[state]}")
```

Slide 10: Implementing a Rule-Based Agent

Rule-based agents make decisions based on a set of predefined rules. Let's implement a simple rule-based agent for a traffic light system.

```python
class TrafficLightAgent:
    def __init__(self):
        self.state = "red"
        self.timer = 0
    
    def update(self):
        self.timer += 1
        if self.state == "red" and self.timer >= 30:
            self.state = "green"
            self.timer = 0
        elif self.state == "green" and self.timer >= 20:
            self.state = "yellow"
            self.timer = 0
        elif self.state == "yellow" and self.timer >= 5:
            self.state = "red"
            self.timer = 0
    
    def get_state(self):
        return self.state

# Simulation
agent = TrafficLightAgent()
for _ in range(100):
    agent.update()
    print(f"Current state: {agent.get_state()}, Timer: {agent.timer}")
```

Slide 11: Implementing a Reactive Agent

Reactive agents respond directly to their environment without maintaining internal state. Let's implement a simple reactive agent for a robot vacuum cleaner.

```python
import random

class ReactiveVacuumAgent:
    def sense(self, environment):
        return environment.is_dirty()
    
    def act(self, perception):
        if perception:
            return "clean"
        else:
            return random.choice(["move_left", "move_right"])

class Environment:
    def __init__(self):
        self.locations = [True, False]  # True means dirty
    
    def is_dirty(self):
        return self.locations[0]  # Check only the current location
    
    def clean(self):
        self.locations[0] = False
    
    def move_dirt(self):
        self.locations[0] = random.choice([True, False])

# Simulation
env = Environment()
agent = ReactiveVacuumAgent()

for _ in range(10):
    perception = agent.sense(env)
    action = agent.act(perception)
    print(f"Perception: {perception}, Action: {action}")
    
    if action == "clean":
        env.clean()
    env.move_dirt()  # Simulate changing environment
```

Slide 12: Implementing a Goal-Based Agent

Goal-based agents work towards achieving specific goals. Let's implement a simple goal-based agent for pathfinding in a maze.

```python
import heapq

class MazeAgent:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
    
    def heuristic(self, a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])
    
    def get_neighbors(self, pos):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return [(pos[0] + dx, pos[1] + dy) for dx, dy in neighbors
                if 0 <= pos[0] + dx < len(self.maze) and
                   0 <= pos[1] + dy < len(self.maze[0]) and
                   self.maze[pos[0] + dx][pos[1] + dy] != '#']
    
    def find_path(self):
        queue = [(0, self.start)]
        came_from = {}
        cost_so_far = {self.start: 0}
        
        while queue:
            _, current = heapq.heappop(queue)
            
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]
            
            for next in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(self.goal, next)
                    heapq.heappush(queue, (priority, next))
                    came_from[next] = current
        
        return None  # No path found

# Example usage
maze = [
    "S...#",
    ".##..",
    "...##",
    ".#..G"
]
start = (0, 0)
goal = (3, 4)

agent = MazeAgent(maze, start, goal)
path = agent.find_path()
print("Path found:", path)
```

Slide 13: Real-Life Example: Smart Home Automation

Let's implement a multi-agent system for smart home automation, including temperature control and lighting.

```python
import random

class TemperatureAgent:
    def __init__(self, name, ideal_temp):
        self.name = name
        self.ideal_temp = ideal_temp
    
    def sense(self, current_temp):
        return current_temp
    
    def act(self, sensed_temp):
        if sensed_temp < self.ideal_temp:
            return "increase"
        elif sensed_temp > self.ideal_temp:
            return "decrease"
        else:
            return "maintain"

class LightingAgent:
    def __init__(self, name):
        self.name = name
    
    def sense(self, is_daytime, motion_detected):
        return (is_daytime, motion_detected)
    
    def act(self, sensed_data):
        is_daytime, motion_detected = sensed_data
        if not is_daytime and motion_detected:
            return "turn_on"
        elif is_daytime or (not is_daytime and not motion_detected):
            return "turn_off"
        else:
            return "no_action"

class SmartHome:
    def __init__(self):
        self.temperature = 22
        self.is_daytime = True
        self.motion_detected = False
    
    def update_environment(self):
        self.temperature += random.uniform(-0.5, 0.5)
        self.is_daytime = random.choice([True, False])
        self.motion_detected = random.choice([True, False])
    
    def adjust_temperature(self, action):
        if action == "increase":
            self.temperature += 0.5
        elif action == "decrease":
            self.temperature -= 0.5
    
    def adjust_lighting(self, action):
        print(f"Lighting: {action}")

# Simulation
home = SmartHome()
temp_agent = TemperatureAgent("TempAgent", 23)
light_agent = LightingAgent("LightAgent")

for _ in range(10):
    home.update_environment()
    
    temp_action = temp_agent.act(temp_agent.sense(home.temperature))
    home.adjust_temperature(temp_action)
    
    light_action = light_agent.act(light_agent.sense(home.is_daytime, home.motion_detected))
    home.adjust_lighting(light_action)
    
    print(f"Temperature: {home.temperature:.1f}°C, Daytime: {home.is_daytime}, Motion: {home.motion_detected}")
    print(f"Temperature Action: {temp_action}")
    print("---")
```

Slide 14: Real-Life Example: Traffic Management System

Let's implement a multi-agent system for managing traffic at an intersection.

```python
import random

class TrafficLightAgent:
    def __init__(self, name):
        self.name = name
        self.state = "red"
        self.timer = 0
    
    def update(self, traffic_density):
        self.timer += 1
        if self.state == "red" and self.timer >= 30:
            self.state = "green"
            self.timer = 0
        elif self.state == "green":
            if traffic_density > 0.7 and self.timer >= 45:
                self.state = "yellow"
                self.timer = 0
            elif traffic_density <= 0.7 and self.timer >= 30:
                self.state = "yellow"
                self.timer = 0
        elif self.state == "yellow" and self.timer >= 5:
            self.state = "red"
            self.timer = 0
    
    def get_state(self):
        return self.state

class Intersection:
    def __init__(self):
        self.north_south = TrafficLightAgent("North-South")
        self.east_west = TrafficLightAgent("East-West")
        self.traffic_density = {"north_south": 0.5, "east_west": 0.5}
    
    def update_traffic_density(self):
        self.traffic_density["north_south"] = random.uniform(0, 1)
        self.traffic_density["east_west"] = random.uniform(0, 1)
    
    def update(self):
        self.update_traffic_density()
        self.north_south.update(self.traffic_density["north_south"])
        self.east_west.update(self.traffic_density["east_west"])
    
    def display_state(self):
        print(f"North-South: {self.north_south.get_state()}, Traffic Density: {self.traffic_density['north_south']:.2f}")
        print(f"East-West: {self.east_west.get_state()}, Traffic Density: {self.traffic_density['east_west']:.2f}")
        print("---")

# Simulation
intersection = Intersection()

for _ in range(20):
    intersection.update()
    intersection.display_state()
```

Slide 15: Additional Resources

For those interested in diving deeper into the world of agents and multi-agent systems, here are some valuable resources:

1. "An Introduction to MultiAgent Systems" by Michael Wooldridge ArXiv: [https://arxiv.org/abs/1909.12201](https://arxiv.org/abs/1909.12201)
2. "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig (Contains chapters on agents and multi-agent systems)
3. "Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations" by Yoav Shoham and Kevin Leyton-Brown ArXiv: [https://arxiv.org/abs/0812.2041](https://arxiv.org/abs/0812.2041)
4. "The Foundation of Multi-Agent Learning: An Introduction" by Peter Stone ArXiv: [https://arxiv.org/abs/2103.02373](https://arxiv.org/abs/2103.02373)

These resources provide in-depth knowledge about agent architectures, multi-agent interactions, and advanced topics in the field. Remember to verify the availability and content of these resources, as they may have been updated or moved since the time of this presentation.

