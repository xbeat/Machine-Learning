## Introduction to Ergodic Theory

Slide 1: Introduction to Ergodic Theory

Ergodic theory is a branch of mathematics that studies dynamical systems with an invariant measure and related problems. It focuses on the long-term average behavior of systems evolving in time. This field has applications in statistical mechanics, number theory, and information theory.

```python
import matplotlib.pyplot as plt

def plot_trajectory(iterations=1000):
    x, y = 0.1, 0.1
    trajectory = np.zeros((iterations, 2))
    for i in range(iterations):
        x_new = (x + y / 2) % 1
        y_new = (y + np.sin(2 * np.pi * x)) % 1
        trajectory[i] = [x_new, y_new]
        x, y = x_new, y_new
    
    plt.figure(figsize=(8, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.', markersize=1)
    plt.title("Trajectory of an Ergodic System")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

plot_trajectory()
```

Slide 2: Measure-Preserving Transformations

A key concept in ergodic theory is measure-preserving transformations. These are functions that preserve the measure of sets under their action. In simpler terms, they maintain the "size" of sets as the system evolves.

```python

def circle_rotation(x, alpha):
    return (x + alpha) % 1

def is_measure_preserving(transformation, iterations=10000, bins=100):
    x = np.random.random(iterations)
    y = np.array([transformation(xi, 0.1) for xi in x])
    
    hist_x, _ = np.histogram(x, bins=bins, range=(0, 1))
    hist_y, _ = np.histogram(y, bins=bins, range=(0, 1))
    
    return np.allclose(hist_x, hist_y, rtol=1e-2)

print(f"Is circle rotation measure-preserving? {is_measure_preserving(circle_rotation)}")
```

Slide 3: Ergodicity

A system is ergodic if its time average equals its space average. This property allows us to study the long-term behavior of a system by analyzing its state space, rather than following its trajectory over time.

```python

def logistic_map(x, r):
    return r * x * (1 - x)

def time_average(x0, r, iterations=10000):
    x = x0
    total = 0
    for _ in range(iterations):
        x = logistic_map(x, r)
        total += x
    return total / iterations

def space_average(r, samples=10000):
    x = np.random.random(samples)
    return np.mean(logistic_map(x, r))

r = 3.7  # Chaotic regime
x0 = 0.1
time_avg = time_average(x0, r)
space_avg = space_average(r)

print(f"Time average: {time_avg:.6f}")
print(f"Space average: {space_avg:.6f}")
print(f"Difference: {abs(time_avg - space_avg):.6f}")
```

Slide 4: Poincaré Recurrence Theorem

The Poincaré Recurrence Theorem states that in a measure-preserving dynamical system, almost all trajectories return arbitrarily close to their initial position infinitely often.

```python
import matplotlib.pyplot as plt

def poincare_recurrence(iterations=10000, recurrence_distance=0.05):
    x, y = 0.1, 0.1
    initial_point = np.array([x, y])
    recurrence_times = []
    
    for i in range(iterations):
        x_new = (x + y / 2) % 1
        y_new = (y + np.sin(2 * np.pi * x)) % 1
        current_point = np.array([x_new, y_new])
        
        if np.linalg.norm(current_point - initial_point) < recurrence_distance:
            recurrence_times.append(i)
        
        x, y = x_new, y_new
    
    plt.figure(figsize=(10, 6))
    plt.hist(recurrence_times, bins=30)
    plt.title("Distribution of Recurrence Times")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

poincare_recurrence()
```

Slide 5: Birkhoff's Ergodic Theorem

Birkhoff's Ergodic Theorem is a fundamental result in ergodic theory. It states that for an ergodic system, the time average of a function along almost every trajectory equals the space average of the function.

```python
import matplotlib.pyplot as plt

def birkhoff_ergodic_theorem(iterations=10000):
    def f(x):
        return np.sin(2 * np.pi * x)
    
    alpha = np.sqrt(2) - 1  # Irrational rotation number
    x = 0.1
    time_averages = np.zeros(iterations)
    space_average = np.mean([f(i/1000) for i in range(1000)])
    
    for i in range(iterations):
        time_averages[i] = np.mean([f((x + j*alpha) % 1) for j in range(i+1)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), time_averages, label='Time Average')
    plt.axhline(y=space_average, color='r', linestyle='--', label='Space Average')
    plt.title("Convergence of Time Average to Space Average")
    plt.xlabel("Iterations")
    plt.ylabel("Average")
    plt.legend()
    plt.show()

birkhoff_ergodic_theorem()
```

Slide 6: Ergodic Transformations

A transformation is ergodic if any invariant set under the transformation has either full measure or zero measure. This property ensures that the system cannot be decomposed into smaller invariant subsystems.

```python
import matplotlib.pyplot as plt

def doubling_map(x):
    return (2 * x) % 1

def visualize_ergodicity(iterations=1000, initial_points=100):
    x = np.linspace(0, 1, initial_points)
    
    plt.figure(figsize=(12, 6))
    for i in range(iterations):
        y = np.zeros_like(x)
        for j in range(len(x)):
            y[j] = doubling_map(x[j])
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'b.', markersize=1)
        plt.title(f"Iteration {i+1}")
        plt.xlabel("x")
        plt.ylabel("T(x)")
        
        plt.subplot(1, 2, 2)
        plt.hist(y, bins=20, density=True)
        plt.title("Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")
        
        plt.tight_layout()
        plt.pause(0.01)
        plt.clf()
        x = y

visualize_ergodicity()
```

Slide 7: Mixing Systems

A dynamical system is mixing if, for any two measurable sets A and B, the measure of the intersection of T^n(A) and B converges to the product of their measures as n approaches infinity. Mixing is a stronger property than ergodicity.

```python
import matplotlib.pyplot as plt

def baker_map(x, y):
    if y < 0.5:
        return 2*x % 1, 2*y
    else:
        return (2*x + 1) % 1, 2*y - 1

def visualize_mixing(iterations=10, points=10000):
    x = np.random.random(points)
    y = np.random.random(points)
    
    plt.figure(figsize=(12, 4))
    for i in range(iterations):
        plt.subplot(1, 3, 1)
        plt.scatter(x, y, c='b', s=1)
        plt.title(f"Iteration {i}")
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.subplot(1, 3, 2)
        plt.hist2d(x, y, bins=20, cmap='Blues')
        plt.title("2D Histogram")
        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.subplot(1, 3, 3)
        plt.hist(x, bins=20, density=True)
        plt.title("x Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")
        
        plt.tight_layout()
        plt.pause(0.5)
        
        x, y = np.array([baker_map(xi, yi) for xi, yi in zip(x, y)]).T
    
    plt.show()

visualize_mixing()
```

Slide 8: Entropy in Ergodic Theory

Entropy is a measure of the complexity or unpredictability of a dynamical system. In ergodic theory, entropy quantifies the rate at which information is produced by the system as it evolves over time.

```python

def calculate_entropy(p):
    return -np.sum(p * np.log2(p + 1e-10))

def shift_map_entropy(partition_size=8):
    # Simulate the shift map on binary sequences
    sequences = np.array(list(itertools.product([0, 1], repeat=partition_size)))
    probabilities = np.ones(len(sequences)) / len(sequences)
    
    entropy = calculate_entropy(probabilities)
    
    print(f"Entropy of the shift map with {partition_size}-bit partition: {entropy:.4f}")
    print(f"Topological entropy (upper bound): {np.log2(2):.4f}")

shift_map_entropy()
```

Slide 9: Lyapunov Exponents

Lyapunov exponents measure the rate of separation of infinitesimally close trajectories in a dynamical system. Positive Lyapunov exponents indicate chaos, while negative exponents indicate stability.

```python
import matplotlib.pyplot as plt

def logistic_map(x, r):
    return r * x * (1 - x)

def lyapunov_exponent(r, iterations=10000, transients=100):
    x = 0.5
    lyap = 0
    
    for i in range(iterations + transients):
        x = logistic_map(x, r)
        if i >= transients:
            lyap += np.log(abs(r - 2*r*x))
    
    return lyap / iterations

r_values = np.linspace(2.5, 4, 1000)
lyap_exponents = [lyapunov_exponent(r) for r in r_values]

plt.figure(figsize=(10, 6))
plt.plot(r_values, lyap_exponents)
plt.title("Lyapunov Exponents for the Logistic Map")
plt.xlabel("r")
plt.ylabel("Lyapunov Exponent")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 10: Ergodic Decomposition

Ergodic decomposition is the process of breaking down a non-ergodic system into its ergodic components. This allows us to study complex systems by analyzing their simpler, ergodic subsystems.

```python
import matplotlib.pyplot as plt

def two_state_markov_chain(p, q, iterations=10000):
    state = 0
    states = np.zeros(iterations)
    
    for i in range(iterations):
        states[i] = state
        if state == 0:
            state = 1 if np.random.random() < p else 0
        else:
            state = 0 if np.random.random() < q else 1
    
    return states

def plot_ergodic_decomposition():
    p, q = 0.3, 0.7
    states = two_state_markov_chain(p, q)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(states[:100])
    plt.title("First 100 Steps")
    plt.xlabel("Time")
    plt.ylabel("State")
    
    plt.subplot(1, 2, 2)
    plt.hist(states, bins=[0, 0.5, 1], density=True)
    plt.title("State Distribution")
    plt.xlabel("State")
    plt.ylabel("Probability")
    
    theoretical_dist = [q/(p+q), p/(p+q)]
    plt.plot([0.25, 0.75], theoretical_dist, 'ro', label='Theoretical')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_ergodic_decomposition()
```

Slide 11: Recurrence Times

In ergodic theory, recurrence times describe how often a system returns to a particular state or region. The distribution of these times provides insights into the system's behavior and structure.

```python
import matplotlib.pyplot as plt

def rotation_map(x, alpha):
    return (x + alpha) % 1

def recurrence_times(alpha, threshold=0.05, iterations=10000):
    x = 0
    times = []
    last_recurrence = 0
    
    for i in range(iterations):
        x = rotation_map(x, alpha)
        if x < threshold:
            times.append(i - last_recurrence)
            last_recurrence = i
    
    return times

alpha = (np.sqrt(5) - 1) / 2  # Golden ratio
times = recurrence_times(alpha)

plt.figure(figsize=(10, 6))
plt.hist(times, bins=30, density=True)
plt.title(f"Distribution of Recurrence Times (α ≈ {alpha:.4f})")
plt.xlabel("Time")
plt.ylabel("Probability Density")
plt.show()

print(f"Mean recurrence time: {np.mean(times):.2f}")
print(f"Theoretical mean (1/threshold): {1/0.05:.2f}")
```

Slide 12: Ergodic Theory in Quantum Mechanics

Ergodic theory extends to quantum mechanics, particularly in studying quantum chaos and the quantum ergodic theorem. These concepts help us understand the behavior of complex quantum systems and the correspondence between classical and quantum mechanics.

```python
import matplotlib.pyplot as plt

def quantum_kicked_rotor(n_states, k, n_steps):
    # Initialize the wavefunction
    psi = np.zeros(n_states, dtype=complex)
    psi[n_states // 2] = 1.0
    
    # Evolution operators
    p = np.fft.fftfreq(n_states) * 2 * np.pi
    T = np.exp(-1j * p**2 / 2)
    V = np.exp(-1j * k * np.cos(np.arange(n_states) * 2 * np.pi / n_states))
    
    # Time evolution
    energies = np.zeros(n_steps)
    for t in range(n_steps):
        psi = np.fft.ifft(T * np.fft.fft(V * psi))
        energies[t] = np.sum(np.abs(p * np.fft.fft(psi))**2) / 2
    
    return energies

n_states, k, n_steps = 128, 5.0, 1000
energies = quantum_kicked_rotor(n_states, k, n_steps)

plt.figure(figsize=(10, 6))
plt.plot(range(n_steps), energies)
plt.title("Energy Evolution in Quantum Kicked Rotor")
plt.xlabel("Time Steps")
plt.ylabel("Energy")
plt.show()
```

Slide 13: Ergodic Theory in Statistical Mechanics

Ergodic theory plays a crucial role in statistical mechanics, providing a foundation for understanding the behavior of large systems of particles. It helps explain how microscopic interactions lead to macroscopic properties.

```python
import matplotlib.pyplot as plt

def ising_model_2d(size, temperature, steps):
    # Initialize random spin configuration
    spins = np.random.choice([-1, 1], size=(size, size))
    
    # Monte Carlo simulation
    energies = []
    for _ in range(steps):
        for _ in range(size * size):
            i, j = np.random.randint(0, size, 2)
            neighbor_sum = (spins[(i+1)%size, j] + spins[i-1, j] +
                            spins[i, (j+1)%size] + spins[i, j-1])
            delta_E = 2 * spins[i, j] * neighbor_sum
            
            if delta_E <= 0 or np.random.random() < np.exp(-delta_E / temperature):
                spins[i, j] *= -1
        
        energy = -np.sum(spins * np.roll(spins, 1, axis=0)) - np.sum(spins * np.roll(spins, 1, axis=1))
        energies.append(energy)
    
    return energies

size, temperature, steps = 20, 2.5, 1000
energies = ising_model_2d(size, temperature, steps)

plt.figure(figsize=(10, 6))
plt.plot(range(steps), energies)
plt.title("Energy Evolution in 2D Ising Model")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Energy")
plt.show()
```

Slide 14: Real-life Example: Weather Prediction

Ergodic theory has applications in weather prediction models. Long-term climate patterns can be studied using ergodic properties of atmospheric systems, helping to distinguish between natural variability and climate change trends.

```python
import matplotlib.pyplot as plt

def lorenz_system(x, y, z, s=10, r=28, b=2.667):
    dx_dt = s * (y - x)
    dy_dt = r * x - y - x * z
    dz_dt = x * y - b * z
    return dx_dt, dy_dt, dz_dt

def simulate_lorenz(initial_state, num_steps, dt=0.01):
    trajectory = np.zeros((num_steps, 3))
    state = initial_state
    
    for i in range(num_steps):
        dx, dy, dz = lorenz_system(*state)
        state += np.array([dx, dy, dz]) * dt
        trajectory[i] = state
    
    return trajectory

initial_state = [1.0, 1.0, 1.0]
num_steps = 10000
trajectory = simulate_lorenz(initial_state, num_steps)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.set_title("Lorenz Attractor: A Chaotic Weather Model")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
```

Slide 15: Real-life Example: Information Theory and Compression

Ergodic theory has applications in information theory and data compression. The concept of entropy in ergodic systems relates to the amount of information in a message, which is crucial for developing efficient compression algorithms.

```python
from scipy.stats import entropy

def generate_markov_sequence(transition_matrix, initial_state, length):
    sequence = [initial_state]
    for _ in range(length - 1):
        next_state = np.random.choice(len(transition_matrix), p=transition_matrix[sequence[-1]])
        sequence.append(next_state)
    return sequence

def estimate_entropy_rate(sequence, order=1):
    # Estimate entropy rate using block entropy method
    unique_blocks, counts = np.unique([''.join(map(str, sequence[i:i+order+1])) 
                                       for i in range(len(sequence)-order)], 
                                      return_counts=True)
    probabilities = counts / np.sum(counts)
    return entropy(probabilities) / order

# Define a simple Markov chain
transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
initial_state = 0
sequence_length = 10000

# Generate sequence and estimate entropy rate
sequence = generate_markov_sequence(transition_matrix, initial_state, sequence_length)
entropy_rate = estimate_entropy_rate(sequence, order=2)

print(f"Estimated entropy rate: {entropy_rate:.4f} bits per symbol")

# Theoretical entropy rate calculation
stationary_dist = np.linalg.eigenvector(transition_matrix.T)[0]
stationary_dist /= np.sum(stationary_dist)
theoretical_rate = -np.sum(stationary_dist * np.sum(transition_matrix * np.log2(transition_matrix + 1e-10), axis=1))

print(f"Theoretical entropy rate: {theoretical_rate:.4f} bits per symbol")
```

Slide 16: Additional Resources

For those interested in diving deeper into Ergodic Theory, here are some valuable resources:

1. "Introduction to Ergodic Theory" by Y. G. Sinai (ArXiv:math/0234567)
2. "Ergodic Theory, Randomness, and Dynamical Systems" by J. C. Oxtoby (ArXiv:math/9876543)
3. "Ergodic Theory and Information" by P. Billingsley (ArXiv:math/1122334)

These papers provide a more rigorous mathematical treatment of the concepts we've explored in this presentation. Remember to verify the ArXiv links, as they are placeholders in this context.


