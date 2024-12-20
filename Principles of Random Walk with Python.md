## Principles of Random Walk with Python
Slide 1: What is a Random Walk?

A random walk is a mathematical concept that describes a path consisting of a succession of random steps. It's widely used in various fields, including physics, biology, and finance.

```python
import random
import matplotlib.pyplot as plt

def random_walk_1d(steps):
    position = 0
    positions = [position]
    for _ in range(steps):
        step = random.choice([-1, 1])
        position += step
        positions.append(position)
    return positions

walk = random_walk_1d(100)
plt.plot(walk)
plt.title("1D Random Walk")
plt.xlabel("Step")
plt.ylabel("Position")
plt.show()
```

Slide 2: Properties of Random Walks

Random walks exhibit several key properties:

1. Unpredictability of individual steps
2. Long-term trends may emerge
3. The average distance from the starting point increases with the square root of the number of steps

```python
import numpy as np

def average_distance(num_walks, steps):
    distances = []
    for _ in range(num_walks):
        walk = random_walk_1d(steps)
        distance = abs(walk[-1])
        distances.append(distance)
    return np.mean(distances)

steps = range(10, 1001, 10)
avg_distances = [average_distance(1000, s) for s in steps]
plt.plot(steps, avg_distances, label="Average Distance")
plt.plot(steps, np.sqrt(steps), label="Square Root of Steps")
plt.legend()
plt.xlabel("Number of Steps")
plt.ylabel("Average Distance")
plt.show()
```

Slide 3: Types of Random Walks

There are various types of random walks, including:

1. Simple random walk
2. Self-avoiding walk
3. Lévy flight
4. Brownian motion

Let's implement a 2D random walk:

```python
def random_walk_2d(steps):
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(steps):
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        x += dx
        y += dy
        path.append((x, y))
    return path

walk_2d = random_walk_2d(1000)
x, y = zip(*walk_2d)
plt.plot(x, y)
plt.title("2D Random Walk")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 4: Random Walks in Nature

Random walks are observed in various natural phenomena:

1. Brownian motion of particles in a fluid
2. Movement of animals foraging for food
3. Stock price fluctuations

Let's simulate Brownian motion:

```python
def brownian_motion(n, dt=0.1, sigma=1):
    sqrt_dt = np.sqrt(dt)
    r = np.random.normal(0, 1, n)
    x = np.cumsum(sqrt_dt * sigma * r)
    return x

t = np.linspace(0, 10, 1000)
x = brownian_motion(1000)
plt.plot(t, x)
plt.title("Brownian Motion")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()
```

Slide 5: Random Walks in Finance

In finance, random walks are used to model stock prices and other financial instruments. The efficient market hypothesis suggests that stock prices follow a random walk.

Let's simulate a stock price using geometric Brownian motion:

```python
def geometric_brownian_motion(S0, mu, sigma, T, N):
    dt = T/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) 
    W = np.cumsum(W)*np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X)
    return t, S

t, S = geometric_brownian_motion(S0=100, mu=0.1, sigma=0.3, T=1, N=252)
plt.plot(t, S)
plt.title("Stock Price Simulation")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()
```

Slide 6: Random Walks and Diffusion

Random walks are closely related to diffusion processes. The diffusion equation describes how the density of diffusing particles evolves over time.

Let's simulate a 1D diffusion process:

```python
def diffusion_1d(n_particles, n_steps, D=1):
    positions = np.zeros((n_particles, n_steps))
    for i in range(1, n_steps):
        step = np.random.normal(0, np.sqrt(2*D), n_particles)
        positions[:, i] = positions[:, i-1] + step
    return positions

positions = diffusion_1d(100, 1000)
plt.plot(positions.T)
plt.title("1D Diffusion")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.show()
```

Slide 7: Self-Avoiding Random Walks

Self-avoiding walks are random walks that do not visit the same point more than once. They are used to model polymer chains and other physical systems.

Here's a simple implementation of a 2D self-avoiding walk:

```python
def self_avoiding_walk_2d(steps):
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(steps):
        options = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(options)
        for dx, dy in options:
            new_x, new_y = x + dx, y + dy
            if (new_x, new_y) not in path:
                x, y = new_x, new_y
                path.append((x, y))
                break
        else:
            break
    return path

saw = self_avoiding_walk_2d(1000)
x, y = zip(*saw)
plt.plot(x, y)
plt.title("2D Self-Avoiding Walk")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 8: Random Walks on Graphs

Random walks can also be performed on graphs, where each step moves to a neighboring node. This has applications in network analysis and recommendation systems.

Let's implement a random walk on a simple graph:

```python
import networkx as nx

def random_walk_on_graph(G, start_node, steps):
    path = [start_node]
    current_node = start_node
    for _ in range(steps):
        neighbors = list(G.neighbors(current_node))
        if neighbors:
            current_node = random.choice(neighbors)
            path.append(current_node)
        else:
            break
    return path

G = nx.gnm_random_graph(20, 40)
walk = random_walk_on_graph(G, 0, 10)
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='lightblue', with_labels=True)
path_edges = list(zip(walk, walk[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
plt.title("Random Walk on a Graph")
plt.show()
```

Slide 9: First Passage Time

The first passage time is the time it takes for a random walker to reach a specific state or position for the first time. This concept is important in various applications, including chemical reactions and search algorithms.

Let's calculate the average first passage time to a certain distance:

```python
def first_passage_time(target_distance):
    position = 0
    time = 0
    while abs(position) < target_distance:
        step = random.choice([-1, 1])
        position += step
        time += 1
    return time

target_distances = range(1, 51)
avg_times = [np.mean([first_passage_time(d) for _ in range(1000)]) for d in target_distances]

plt.plot(target_distances, avg_times)
plt.title("Average First Passage Time")
plt.xlabel("Target Distance")
plt.ylabel("Average Time")
plt.show()
```

Slide 10: Random Walks and Monte Carlo Methods

Random walks are fundamental to many Monte Carlo methods, which use random sampling to solve problems that might be deterministic in principle.

Let's use a random walk to estimate π:

```python
def estimate_pi(n_points):
    inside_circle = 0
    for _ in range(n_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / n_points

n_points = [1000, 10000, 100000, 1000000]
estimates = [estimate_pi(n) for n in n_points]

plt.semilogx(n_points, estimates, 'o-')
plt.axhline(y=np.pi, color='r', linestyle='--')
plt.title("Monte Carlo Estimation of π")
plt.xlabel("Number of Points")
plt.ylabel("Estimated π")
plt.show()
```

Slide 11: Lévy Flights

Lévy flights are random walks where the step lengths have a probability distribution that is heavy-tailed. They are observed in various natural phenomena and can model optimal foraging strategies.

Let's implement a simple Lévy flight:

```python
def levy_flight(steps, alpha=1.5):
    x, y = 0, 0
    path = [(x, y)]
    for _ in range(steps):
        step_length = np.random.pareto(alpha)
        angle = random.uniform(0, 2 * np.pi)
        dx = step_length * np.cos(angle)
        dy = step_length * np.sin(angle)
        x += dx
        y += dy
        path.append((x, y))
    return path

levy_walk = levy_flight(1000)
x, y = zip(*levy_walk)
plt.plot(x, y)
plt.title("Lévy Flight")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 12: Continuous-Time Random Walks

Continuous-time random walks (CTRWs) are a generalization of random walks where both the step length and the waiting time between steps are random variables.

Let's implement a simple CTRW:

```python
def ctrw(steps, lambda_param=1, alpha=2):
    t = 0
    x = 0
    times = [t]
    positions = [x]
    for _ in range(steps):
        waiting_time = np.random.exponential(1/lambda_param)
        step = np.random.normal(0, 1)
        t += waiting_time
        x += step
        times.append(t)
        positions.append(x)
    return times, positions

times, positions = ctrw(1000)
plt.plot(times, positions)
plt.title("Continuous-Time Random Walk")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()
```

Slide 13: Applications of Random Walks

Random walks have numerous applications across various fields:

1. Ecology: Animal movement patterns
2. Physics: Particle diffusion
3. Computer Science: Page ranking algorithms
4. Finance: Option pricing models
5. Biology: Protein folding simulations

Here's a simple implementation of a random walk used in a page rank-like algorithm:

```python
def simple_page_rank(G, damping_factor=0.85, num_iterations=100):
    num_pages = len(G)
    ranks = {node: 1/num_pages for node in G}
    
    for _ in range(num_iterations):
        new_ranks = {}
        for node in G:
            rank_sum = sum(ranks[n] / len(G[n]) for n in G[node])
            new_ranks[node] = (1 - damping_factor) / num_pages + damping_factor * rank_sum
        ranks = new_ranks
    
    return ranks

G = nx.gnm_random_graph(10, 20)
ranks = simple_page_rank(G)

nx.draw(G, node_color=[ranks[n] for n in G], node_size=500, cmap=plt.cm.Reds)
plt.title("Simple Page Rank Visualization")
plt.show()
```

Slide 14: Further Resources

For those interested in diving deeper into random walks and related topics, here are some recommended resources:

1. "First-Passage Phenomena and Their Applications" by Redner, S. (2001). ArXiv: [https://arxiv.org/abs/cond-mat/0103384](https://arxiv.org/abs/cond-mat/0103384)
2. "Random Walks and Diffusion on Networks" by Masuda, N., Porter, M. A., & Lambiotte, R. (2017). ArXiv: [https://arxiv.org/abs/1612.03281](https://arxiv.org/abs/1612.03281)
3. "Lévy Flights and Related Topics in Physics" by Shlesinger, M. F., Zaslavsky, G. M., & Frisch, U. (1995). ArXiv: [https://arxiv.org/abs/cond-mat/9506086](https://arxiv.org/abs/cond-mat/9506086)

These papers provide in-depth discussions on various aspects of random walks and their applications in different fields.

