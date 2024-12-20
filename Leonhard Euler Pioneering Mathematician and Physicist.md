## Leonhard Euler Pioneering Mathematician and Physicist:
Slide 1: Leonhard Euler (1707-1783): Mathematical Genius and Prolific Contributor

Leonhard Euler, a Swiss mathematician and physicist, revolutionized multiple fields of mathematics and laid the groundwork for modern mathematical notation. His contributions span calculus, graph theory, mechanics, optics, and astronomy, making him one of the most influential mathematicians in history.

```python
import matplotlib.pyplot as plt
import numpy as np

def euler_identity(x):
    return np.exp(1j * x)

x = np.linspace(0, 2*np.pi, 100)
y = euler_identity(x)

plt.figure(figsize=(8, 8))
plt.plot(y.real, y.imag)
plt.title("Euler's Identity: e^(iπ) + 1 = 0")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 2: Early Life and Education

Born in Basel, Switzerland, Euler showed exceptional mathematical aptitude from a young age. He studied under Johann Bernoulli at the University of Basel, graduating at 16. Euler's early exposure to the Bernoulli family, renowned mathematicians, significantly influenced his career trajectory.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([
    ("Leonhard Euler", "Johann Bernoulli"),
    ("Johann Bernoulli", "Jacob Bernoulli"),
    ("Johann Bernoulli", "Daniel Bernoulli"),
    ("Johann Bernoulli", "Nicolaus Bernoulli")
])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, font_weight='bold')
edge_labels = {("Leonhard Euler", "Johann Bernoulli"): "Student"}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Euler's Early Influences")
plt.axis('off')
plt.show()
```

Slide 3: Euler's Number and Natural Logarithms

Euler introduced the concept of e, the base of natural logarithms, now known as Euler's number. This irrational and transcendental number plays a crucial role in calculus and complex analysis. Euler's work on exponential functions and logarithms laid the foundation for many areas of modern mathematics.

```python
import numpy as np
import matplotlib.pyplot as plt

def e_approximation(n):
    return (1 + 1/n)**n

n_values = np.logspace(0, 6, 100)
e_approx = [e_approximation(n) for n in n_values]

plt.figure(figsize=(10, 6))
plt.semilogx(n_values, e_approx)
plt.axhline(y=np.e, color='r', linestyle='--', label="e")
plt.xlabel("n")
plt.ylabel("(1 + 1/n)^n")
plt.title("Approximation of e as n increases")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Euler's Formula and Complex Analysis

Euler's formula, eiπ + 1 = 0, is often regarded as the most beautiful equation in mathematics. It elegantly connects five fundamental mathematical constants. This formula is central to complex analysis and has applications in physics, engineering, and signal processing.

```python
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2*np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Unit Circle')
plt.plot([0, 1], [0, 0], 'r', linewidth=2, label='cos(θ)')
plt.plot([1, 1], [0, 0], 'g', linewidth=2, label='sin(θ)')
plt.plot([0, 1], [0, 0], 'k--')
plt.plot([1, 1], [0, 0], 'k--')

plt.title("Euler's Formula: e^(iθ) = cos(θ) + i*sin(θ)")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 5: Graph Theory and the Königsberg Bridge Problem

Euler laid the foundation for graph theory by solving the Königsberg Bridge Problem. This problem, which asked whether it was possible to walk through the city crossing each of its seven bridges exactly once, led Euler to develop concepts of vertices, edges, and paths in graphs.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.MultiGraph()
G.add_edges_from([
    (0, 1), (0, 1), (0, 2), (0, 2),
    (1, 2), (1, 3), (2, 3)
])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
        node_size=3000, font_size=12, font_weight='bold')
edge_labels = {(u, v): f'Bridge {i+1}' for i, (u, v) in enumerate(G.edges())}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Königsberg Bridge Problem")
plt.axis('off')
plt.show()
```

Slide 6: Euler's Identity in Number Theory

Euler's work in number theory produced several groundbreaking results. His identity relating exponential functions to trigonometric functions is a cornerstone of complex analysis. Euler also made significant contributions to the theory of partitions and the distribution of prime numbers.

```python
import sympy as sp

def euler_totient(n):
    phi = n
    for p in sp.prime_factors(n):
        phi *= (1 - 1/p)
    return int(phi)

def verify_euler_theorem(a, n):
    if sp.gcd(a, n) != 1:
        return False
    phi = euler_totient(n)
    left = pow(a, phi, n)
    return left == 1

a, n = 7, 15
result = verify_euler_theorem(a, n)
print(f"Euler's theorem holds for a={a}, n={n}: {result}")
```

Slide 7: Differential Equations and the Euler Method

Euler made substantial contributions to the field of differential equations. He developed the Euler method, a numerical procedure for solving ordinary differential equations with a given initial value. This method forms the basis for more advanced numerical integration techniques used in various scientific and engineering applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, y0, t0, tn, h):
    t = np.arange(t0, tn+h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + h * f(t[i-1], y[i-1])
    return t, y

def f(t, y):
    return y

t0, y0 = 0, 1
tn, h = 5, 0.1

t, y = euler_method(f, y0, t0, tn, h)

plt.figure(figsize=(10, 6))
plt.plot(t, y, 'b-', label='Euler Method')
plt.plot(t, np.exp(t), 'r--', label='Exact Solution')
plt.title("Euler Method for dy/dt = y")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Euler's Work in Physics and Mechanics

Euler's contributions extended beyond pure mathematics into physics and mechanics. He formulated the Euler-Lagrange equations, fundamental in classical mechanics, and made significant advancements in fluid dynamics with the Euler equations describing inviscid fluid flow.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_ode(y, t, L, g):
    theta, omega = y
    dydt = [omega, -g/L * np.sin(theta)]
    return dydt

L, g = 1.0, 9.81
y0 = [np.pi/4, 0.0]
t = np.linspace(0, 10, 1000)

sol = odeint(pendulum_ode, y0, t, args=(L, g))

plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], 'b', label='θ(t)')
plt.plot(t, sol[:, 1], 'g', label='ω(t)')
plt.title("Simple Pendulum Motion")
plt.xlabel("Time")
plt.ylabel("Angle (rad) / Angular Velocity (rad/s)")
plt.legend()
plt.grid(True)
plt.show()
```

Slide 9: Euler's Contributions to Astronomy

Euler made significant contributions to celestial mechanics, particularly in the study of lunar motion and the three-body problem. His work on planetary orbits and perturbation theory advanced the field of astronomy and laid groundwork for future space exploration.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def three_body_problem(state, t, G, m1, m2, m3):
    r1, r2, r3 = state[:2], state[2:4], state[4:6]
    v1, v2, v3 = state[6:8], state[8:10], state[10:]
    
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)
    
    dv1 = G * (m2 * (r2 - r1) / r12**3 + m3 * (r3 - r1) / r13**3)
    dv2 = G * (m1 * (r1 - r2) / r12**3 + m3 * (r3 - r2) / r23**3)
    dv3 = G * (m1 * (r1 - r3) / r13**3 + m2 * (r2 - r3) / r23**3)
    
    return np.concatenate([v1, v2, v3, dv1, dv2, dv3])

G = 1
m1, m2, m3 = 1, 1, 1
initial_state = np.array([0, 0, 1, 0, 0, 1, 0.5, 0, -0.5, 0, 0, -1])
t = np.linspace(0, 20, 1000)

solution = odeint(three_body_problem, initial_state, t, args=(G, m1, m2, m3))

plt.figure(figsize=(10, 10))
plt.plot(solution[:, 0], solution[:, 1], label='Body 1')
plt.plot(solution[:, 2], solution[:, 3], label='Body 2')
plt.plot(solution[:, 4], solution[:, 5], label='Body 3')
plt.title("Three-Body Problem Simulation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 10: Euler's Number Theory and the Riemann Hypothesis

Euler's work in number theory laid the groundwork for future developments, including the Riemann Hypothesis. His product formula for the Riemann zeta function connected prime numbers to complex analysis, a relationship that continues to fascinate mathematicians today.

```python
import numpy as np
import matplotlib.pyplot as plt

def riemann_zeta(s, terms=1000):
    return np.sum(1 / np.power(np.arange(1, terms + 1), s))

s_values = np.linspace(0, 20, 1000)
zeta_values = [riemann_zeta(s) for s in s_values]

plt.figure(figsize=(10, 6))
plt.plot(s_values, zeta_values)
plt.title("Riemann Zeta Function for Real s > 1")
plt.xlabel("s")
plt.ylabel("ζ(s)")
plt.grid(True)
plt.show()
```

Slide 11: Euler's Legacy in Mathematical Notation

Euler introduced much of the mathematical notation used today, including the concept of a function f(x), the use of e for the base of natural logarithms, i for the imaginary unit, and π for the ratio of a circle's circumference to its diameter. His notational innovations greatly simplified mathematical expression and communication.

```python
import sympy as sp

x, y = sp.symbols('x y')
expr = sp.exp(sp.I * sp.pi) + 1

print("Euler's Identity in SymPy:")
sp.pprint(expr)
print("\nSimplified:")
sp.pprint(sp.simplify(expr))
```

Slide 12: Euler's Influence on Modern Mathematics

Euler's work continues to influence modern mathematics and its applications. His contributions to calculus, complex analysis, and number theory form the basis of many contemporary mathematical and scientific fields, from cryptography to quantum mechanics.

```python
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(h, w, max_iter):
    y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)
    
    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2
    
    return divtime

h, w = 1000, 1500
max_iter = 100

plt.figure(figsize=(12, 8))
plt.imshow(mandelbrot(h, w, max_iter), cmap='hot', extent=[-2, 0.8, -1.4, 1.4])
plt.title("Mandelbrot Set: A Modern Application of Complex Analysis")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.colorbar(label='Iteration count')
plt.show()
```

Slide 13: Additional Resources

* "Euler: The Master of Us All" by William Dunham (MAA, 1999)
* "An Introduction to the Theory of Numbers" by G.H. Hardy and E.M. Wright (Oxford University Press, 1979)
* "A First Course in Complex Analysis" by Matthias Beck, Gerald Marchesi, Dennis Pixton, and Lucas Sabalka (arXiv:1605.01059)
* "Euler's Gem: The Polyhedron Formula and the Birth of Topology" by David S. Richeson (Princeton University Press, 2008)
* "The Euler Archive" ([https://scholarlycommons.pacific.edu/euler/](https://scholarlycommons.pacific.edu/euler/))

