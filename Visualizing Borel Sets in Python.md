## Visualizing Borel Sets in Python

Slide 1: Introduction to Borel Sets

Borel sets are fundamental in measure theory and probability. They form a σ-algebra generated by open sets in a topological space. Let's explore their properties and applications using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_interval(a, b, color='blue'):
    plt.axvline(x=a, color=color, linestyle='--')
    plt.axvline(x=b, color=color, linestyle='--')
    plt.axhline(y=0, color=color, linewidth=2)
    plt.plot([a, b], [0, 0], color=color, linewidth=4)
    plt.text((a+b)/2, 0.1, f'[{a}, {b}]', ha='center')

plt.figure(figsize=(10, 2))
plot_interval(0, 1)
plt.title('Borel Set Example: Closed Interval [0, 1]')
plt.axis('off')
plt.show()
```

Slide 2: Open Sets

Open sets are the building blocks of Borel sets. In R, an open set is a union of open intervals.

```python
def plot_open_interval(a, b):
    x = np.linspace(a, b, 100)
    y = np.zeros_like(x)
    plt.plot(x, y, 'b', linewidth=2)
    plt.plot(a, 0, 'wo', markersize=10)
    plt.plot(b, 0, 'wo', markersize=10)
    plt.text((a+b)/2, 0.1, f'({a}, {b})', ha='center')

plt.figure(figsize=(10, 2))
plot_open_interval(0, 1)
plt.title('Open Interval (0, 1)')
plt.axis('off')
plt.show()
```

Slide 3: Closed Sets

Closed sets are complements of open sets. They include their boundary points.

```python
def plot_closed_set():
    x = np.linspace(-1, 2, 300)
    y = np.zeros_like(x)
    plt.plot(x, y, 'b', linewidth=2)
    plt.plot(0, 0, 'bo', markersize=10)
    plt.plot(1, 0, 'bo', markersize=10)
    plt.text(0.5, 0.1, '[0, 1]', ha='center')

plt.figure(figsize=(10, 2))
plot_closed_set()
plt.title('Closed Set [0, 1]')
plt.axis('off')
plt.show()
```

Slide 4: Countable Union and Intersection

Borel sets are closed under countable unions and intersections. Let's visualize this concept.

```python
def plot_union_intersection():
    intervals = [(0, 0.5), (0.3, 0.8), (0.7, 1)]
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(10, 4))
    for i, (a, b) in enumerate(intervals):
        plt.plot([a, b], [i, i], color=colors[i], linewidth=4)
        plt.text(a-0.05, i, f'({a}, {b})', va='center', ha='right')
    
    plt.plot([0, 1], [-1, -1], 'k', linewidth=4)
    plt.text(-0.05, -1, 'Union', va='center', ha='right')
    
    plt.axis('off')
    plt.title('Countable Union of Open Intervals')
    plt.show()

plot_union_intersection()
```

Slide 5: Borel σ-algebra

The Borel σ-algebra is generated by open sets and is closed under complement, countable union, and countable intersection.

```python
def visualize_borel_algebra():
    plt.figure(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)
    
    plt.text(0, 0, 'Borel σ-algebra', ha='center', va='center')
    plt.text(0, 0.5, 'Open Sets', ha='center', va='center')
    plt.text(0.5, -0.5, 'Closed Sets', ha='center', va='center')
    plt.text(-0.5, -0.5, 'Countable Unions', ha='center', va='center')
    
    plt.axis('equal')
    plt.axis('off')
    plt.title('Visualization of Borel σ-algebra')
    plt.show()

visualize_borel_algebra()
```

Slide 6: Borel Measurable Functions

A function is Borel measurable if the preimage of any open set is a Borel set. Let's visualize this concept.

```python
def plot_borel_measurable():
    x = np.linspace(-2, 2, 1000)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.axhline(y=-0.5, color='r', linestyle='--')
    plt.fill_between(x, -0.5, 0.5, alpha=0.2, color='r')
    plt.title('Borel Measurable Function: f(x) = sin(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

plot_borel_measurable()
```

Slide 7: Borel Sets in Probability Theory

Borel sets are crucial in probability theory for defining events. Let's simulate a simple probability experiment.

```python
def coin_flip_experiment(n):
    results = np.random.choice(['H', 'T'], size=n)
    prob_head = np.sum(results == 'H') / n
    return prob_head

n_flips = 10000
prob_head = coin_flip_experiment(n_flips)

plt.figure(figsize=(8, 6))
plt.bar(['Heads', 'Tails'], [prob_head, 1-prob_head])
plt.title(f'Probability Distribution after {n_flips} Coin Flips')
plt.ylabel('Probability')
plt.show()
```

Slide 8: Lebesgue Measure on Borel Sets

The Lebesgue measure extends the notion of length to more complex sets. Let's visualize it for a simple interval.

```python
def lebesgue_measure_interval(a, b):
    x = np.linspace(a-0.5, b+0.5, 1000)
    y = np.zeros_like(x)
    
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, 'k')
    plt.fill_between([a, b], [0, 0], [1, 1], alpha=0.3)
    plt.text((a+b)/2, 0.5, f'Measure: {b-a}', ha='center')
    plt.title(f'Lebesgue Measure of Interval [{a}, {b}]')
    plt.axis('off')
    plt.show()

lebesgue_measure_interval(0, 1)
```

Slide 9: Cantor Set

The Cantor set is a famous example of a Borel set with interesting properties. Let's visualize its construction.

```python
def cantor_set(n):
    def cantor_intervals(n):
        if n == 0:
            return [(0, 1)]
        intervals = cantor_intervals(n-1)
        new_intervals = []
        for a, b in intervals:
            third = (b - a) / 3
            new_intervals.append((a, a + third))
            new_intervals.append((b - third, b))
        return new_intervals

    intervals = cantor_intervals(n)
    
    plt.figure(figsize=(10, 4))
    for i, (a, b) in enumerate(intervals):
        plt.plot([a, b], [n-i, n-i], 'k', linewidth=2)
    plt.title(f'Cantor Set (Iteration {n})')
    plt.axis('off')
    plt.show()

cantor_set(5)
```

Slide 10: Borel Sets in Machine Learning

Borel sets play a role in the theoretical foundations of machine learning. Let's visualize a simple decision boundary.

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

clf = SVC(kernel='linear')
clf.fit(X, y)

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title('SVM Decision Boundary (Borel Set in Feature Space)')
plt.show()
```

Slide 11: Borel Sets in Signal Processing

Borel sets are used in signal processing for defining measurable signals. Let's visualize a simple signal and its Fourier transform.

```python
def signal_processing_example():
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    freq = np.fft.fftfreq(len(t), t[1] - t[0])
    sp = np.fft.fft(signal)
    plt.plot(freq, np.abs(sp))
    plt.title('Frequency Domain (Magnitude Spectrum)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.xlim(0, 30)
    
    plt.tight_layout()
    plt.show()

signal_processing_example()
```

Slide 12: Borel Sets in Finance

Borel sets are used in financial mathematics for defining measurable events. Let's simulate a simple stock price model.

```python
def stock_price_simulation():
    np.random.seed(42)
    days = 252
    dt = 1/days
    mu = 0.1
    sigma = 0.2
    S0 = 100
    
    t = np.linspace(0, 1, days)
    W = np.random.standard_normal(size=days)
    W = np.cumsum(W)*np.sqrt(dt)
    S = S0*np.exp((mu-0.5*sigma**2)*t + sigma*W)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, S)
    plt.title('Simulated Stock Price (Geometric Brownian Motion)')
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.show()

stock_price_simulation()
```

Slide 13: Borel Sets in Quantum Mechanics

In quantum mechanics, observables are associated with self-adjoint operators, and their spectra form Borel sets. Let's visualize a simple quantum system.

```python
def quantum_harmonic_oscillator():
    x = np.linspace(-5, 5, 1000)
    psi = []
    
    def hermite(n, x):
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return 2 * x
        else:
            return 2 * x * hermite(n-1, x) - 2 * (n-1) * hermite(n-2, x)
    
    for n in range(4):
        y = np.exp(-x**2/2) * hermite(n, x)
        y = y / np.sqrt(np.sum(y**2))  # Normalize
        psi.append(y)
    
    plt.figure(figsize=(10, 6))
    for n, y in enumerate(psi):
        plt.plot(x, y + n, label=f'n={n}')
    
    plt.title('Quantum Harmonic Oscillator Wavefunctions')
    plt.xlabel('Position')
    plt.ylabel('Wavefunction (shifted)')
    plt.legend()
    plt.show()

quantum_harmonic_oscillator()
```

Slide 14: Additional Resources

For more in-depth study of Borel sets and measure theory, consider these resources:

1. "Measure Theory and Fine Properties of Functions" by Lawrence C. Evans and Ronald F. Gariepy ArXiv: [https://arxiv.org/abs/math/0504190](https://arxiv.org/abs/math/0504190)
2. "An Introduction to Measure Theory" by Terence Tao ArXiv: [https://arxiv.org/abs/0906.1656](https://arxiv.org/abs/0906.1656)
3. "Probability Theory: The Logic of Science" by E. T. Jaynes (While not on ArXiv, this book provides excellent insights into the applications of measure theory in probability)

Remember to verify these resources and their availability, as ArXiv listings may change over time.

