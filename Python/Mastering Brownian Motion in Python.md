## Mastering Brownian Motion in Python
Slide 1: Introduction to Brownian Motion

Brownian motion is a fundamental concept in physics and mathematics, describing the random motion of particles suspended in a fluid. This phenomenon was first observed by botanist Robert Brown in 1827 while studying pollen grains in water. In this presentation, we'll explore the basics of Brownian motion and how to simulate it using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def brownian_motion_1d(n_steps, dt=1):
    return np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))

t = np.linspace(0, 100, 1000)
x = brownian_motion_1d(1000)

plt.plot(t, x)
plt.title("1D Brownian Motion")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()
```

Slide 2: Properties of Brownian Motion

Brownian motion exhibits several key properties:

1. Continuous but nowhere differentiable paths
2. Gaussian increments
3. Statistical self-similarity
4. Markov property

These properties make Brownian motion a unique and important process in various fields, including physics, finance, and biology.

```python
import numpy as np
import matplotlib.pyplot as plt

def brownian_motion_2d(n_steps, dt=1):
    x = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))
    y = np.cumsum(np.random.normal(0, np.sqrt(dt), n_steps))
    return x, y

x, y = brownian_motion_2d(10000)

plt.figure(figsize=(10, 10))
plt.plot(x, y)
plt.title("2D Brownian Motion")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.show()
```

Slide 3: Mathematical Description

Mathematically, Brownian motion is described as a continuous-time stochastic process. For a one-dimensional Brownian motion B(t), the following properties hold:

1. B(0) = 0
2. For 0 ≤ s < t, the increment B(t) - B(s) follows a normal distribution with mean 0 and variance t - s
3. For non-overlapping time intervals, the increments are independent

```python
import numpy as np
import matplotlib.pyplot as plt

def brownian_bridge(n_steps, dt=1):
    t = np.linspace(0, 1, n_steps)
    dB = np.random.normal(0, np.sqrt(dt), n_steps)
    B = np.cumsum(dB)
    return t, B - t * B[-1]

t, B = brownian_bridge(1000)

plt.plot(t, B)
plt.title("Brownian Bridge")
plt.xlabel("Time")
plt.ylabel("Position")
plt.show()
```

Slide 4: Simulating Brownian Motion

To simulate Brownian motion in Python, we can use the property that increments are normally distributed. We'll create a function that generates a path of Brownian motion using NumPy's random number generator.

```python
import numpy as np
import matplotlib.pyplot as plt

def brownian_motion(n_steps, T=1, d=1, x0=None):
    dt = T / n_steps
    dB = np.random.normal(0, np.sqrt(dt), size=(n_steps, d))
    B0 = np.zeros((1, d)) if x0 is None else np.array(x0).reshape(1, d)
    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
    return B

t = np.linspace(0, 1, 1001)
B = brownian_motion(1000, d=3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(B[:, 0], B[:, 1], B[:, 2])
ax.set_title("3D Brownian Motion")
plt.show()
```

Slide 5: Visualizing Brownian Motion

Visualization is crucial for understanding Brownian motion. We'll create plots to show the paths of particles undergoing Brownian motion in different dimensions.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_brownian_motions(n_paths, n_steps):
    t = np.linspace(0, 1, n_steps + 1)
    for _ in range(n_paths):
        B = brownian_motion(n_steps)
        plt.plot(t, B)

    plt.title(f"{n_paths} Paths of Brownian Motion")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.show()

plot_multiple_brownian_motions(5, 1000)
```

Slide 6: Einstein's Theory of Brownian Motion

Albert Einstein's 1905 paper on Brownian motion provided a theoretical explanation for the phenomenon. He showed that the mean squared displacement of a Brownian particle is proportional to time:

⟨(x(t) - x(0))²⟩ = 2Dt

Where D is the diffusion coefficient. This relationship is fundamental to understanding diffusion processes.

```python
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_displacement(n_particles, n_steps, D=1, dt=1):
    paths = np.cumsum(np.random.normal(0, np.sqrt(2*D*dt), (n_particles, n_steps)), axis=1)
    msd = np.mean(np.square(paths), axis=0)
    t = np.arange(n_steps) * dt
    return t, msd

t, msd = mean_squared_displacement(1000, 1000)

plt.plot(t, msd, label="Simulated")
plt.plot(t, 2*t, 'r--', label="Theoretical")
plt.title("Mean Squared Displacement")
plt.xlabel("Time")
plt.ylabel("MSD")
plt.legend()
plt.show()
```

Slide 7: Brownian Motion in Finance

In finance, Brownian motion is used to model stock prices and other financial instruments. The geometric Brownian motion model assumes that the logarithm of the stock price follows a Brownian motion with drift.

```python
import numpy as np
import matplotlib.pyplot as plt

def geometric_brownian_motion(S0, mu, sigma, T, N):
    dt = T/N
    t = np.linspace(0, T, N+1)
    W = np.random.standard_normal(size=N+1)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return t, S

t, S = geometric_brownian_motion(S0=100, mu=0.1, sigma=0.3, T=1, N=252)

plt.plot(t, S)
plt.title("Geometric Brownian Motion: Stock Price Simulation")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()
```

Slide 8: Brownian Motion and the Heat Equation

Brownian motion is closely related to the heat equation, a partial differential equation that describes how heat diffuses through a material. The probability density function of a Brownian motion satisfies the heat equation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def heat_equation_solution(x, t, D=1):
    return norm.pdf(x, loc=0, scale=np.sqrt(2*D*t))

x = np.linspace(-5, 5, 1000)
t_values = [0.1, 0.5, 1.0, 2.0]

for t in t_values:
    plt.plot(x, heat_equation_solution(x, t), label=f't = {t}')

plt.title("Heat Equation Solution (Probability Density)")
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
```

Slide 9: Fractional Brownian Motion

Fractional Brownian motion (fBm) is a generalization of Brownian motion with long-range dependencies. It's characterized by the Hurst parameter H, which determines the nature of the correlations between increments.

```python
import numpy as np
import matplotlib.pyplot as plt

def fbm(n, H):
    t = np.arange(n)
    r = np.zeros(2*n-1)
    r[:n] = 0.5 * ((t+1)**(2*H) - 2*t**(2*H) + (t-1)**(2*H))
    r[n:] = r[n-2::-1]
    Lambda = np.real(np.fft.fft(r))
    W = np.fft.fft(np.random.randn(2*n-1))
    fBm = np.fft.ifft(W * np.sqrt(Lambda))
    return np.real(fBm[:n])

t = np.linspace(0, 1, 1000)
for H in [0.3, 0.5, 0.7]:
    plt.plot(t, fbm(1000, H), label=f'H = {H}')

plt.title("Fractional Brownian Motion")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.show()
```

Slide 10: Brownian Motion and Random Walks

Brownian motion can be thought of as the continuous-time limit of a random walk. As the step size decreases and the number of steps increases, a random walk converges to Brownian motion.

```python
import numpy as np
import matplotlib.pyplot as plt

def random_walk(n_steps):
    return np.cumsum(np.random.choice([-1, 1], size=n_steps))

def scaled_random_walk(n_steps, n_walks):
    t = np.linspace(0, 1, n_steps)
    for _ in range(n_walks):
        walk = random_walk(n_steps) / np.sqrt(n_steps)
        plt.plot(t, walk, alpha=0.5)
    
    plt.title(f"Scaled Random Walks (n={n_steps})")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.show()

scaled_random_walk(1000, 10)
```

Slide 11: Applications in Physics: Diffusion

Brownian motion is fundamental to understanding diffusion processes in physics. It describes the random motion of particles in a fluid, which leads to the spread of matter over time.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def diffusion_simulation(n_particles, n_steps, D=1, dt=0.01):
    positions = np.zeros((n_particles, n_steps))
    positions[:, 0] = np.random.normal(0, 1, n_particles)
    
    for i in range(1, n_steps):
        positions[:, i] = positions[:, i-1] + np.random.normal(0, np.sqrt(2*D*dt), n_particles)
    
    return positions

n_particles, n_steps = 1000, 100
positions = diffusion_simulation(n_particles, n_steps)

plt.hist(positions[:, -1], bins=50, density=True, alpha=0.7)
x = np.linspace(positions.min(), positions.max(), 100)
plt.plot(x, norm.pdf(x, 0, np.sqrt(2*n_steps*0.01)), 'r-', lw=2)
plt.title("Particle Distribution after Diffusion")
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.show()
```

Slide 12: Brownian Motion in Biology

In biology, Brownian motion plays a crucial role in many cellular processes, such as the movement of molecules within cells and the diffusion of nutrients across membranes.

```python
import numpy as np
import matplotlib.pyplot as plt

def cellular_brownian_motion(n_particles, n_steps, cell_radius):
    positions = np.zeros((n_particles, n_steps, 2))
    for i in range(1, n_steps):
        step = np.random.normal(0, 0.1, (n_particles, 2))
        new_positions = positions[:, i-1] + step
        
        # Keep particles within the cell
        distances = np.linalg.norm(new_positions, axis=1)
        mask = distances > cell_radius
        new_positions[mask] = positions[mask, i-1]
        
        positions[:, i] = new_positions
    
    return positions

cell_radius = 5
positions = cellular_brownian_motion(100, 1000, cell_radius)

plt.figure(figsize=(8, 8))
circle = plt.Circle((0, 0), cell_radius, fill=False)
plt.gca().add_artist(circle)
for particle in positions:
    plt.plot(particle[:, 0], particle[:, 1], alpha=0.5)
plt.title("Brownian Motion within a Cell")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.show()
```

Slide 13: Brownian Motion and Stochastic Differential Equations

Brownian motion is a key component in stochastic differential equations (SDEs), which are used to model various phenomena in physics, finance, and other fields. The general form of an SDE is:

dX(t) = μ(X(t), t)dt + σ(X(t), t)dW(t)

Where W(t) is a Brownian motion, μ is the drift term, and σ is the diffusion term.

```python
import numpy as np
import matplotlib.pyplot as plt

def ornstein_uhlenbeck(x0, theta, mu, sigma, T, N):
    dt = T/N
    t = np.linspace(0, T, N+1)
    x = np.zeros(N+1)
    x[0] = x0
    dW = np.random.normal(0, np.sqrt(dt), N)
    
    for i in range(1, N+1):
        x[i] = x[i-1] + theta * (mu - x[i-1]) * dt + sigma * dW[i-1]
    
    return t, x

t, x = ornstein_uhlenbeck(x0=1, theta=1, mu=0, sigma=0.5, T=10, N=1000)

plt.plot(t, x)
plt.title("Ornstein-Uhlenbeck Process")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into Brownian motion and its applications, here are some valuable resources:

1. "Brownian Motion and Stochastic Calculus" by Ioannis Karatzas and Steven E. Shreve ArXiv: [https://arxiv.org/abs/math/0601096](https://arxiv.org/abs/math/0601096)
2. "Introduction to Stochastic Processes" by Gregory F. Lawler ArXiv: [https://arxiv.org/abs/1604.07847](https://arxiv.org/abs/1604.07847)
3. "Stochastic Differential Equations: An Introduction with Applications" by Bernt Øksendal ArXiv: [https://arxiv.org/abs/math/0411401](https://arxiv.org/abs/math/0411401)

These resources provide in-depth coverage of Brownian motion theory and its applications in various fields.

