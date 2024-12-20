## Hyperbolic Manifolds using Python
Hyperbolic Manifolds

Slide 1: Introduction to Hyperbolic Geometry

Hyperbolic geometry is a non-Euclidean geometry that challenges our intuition about parallel lines and the nature of space. In hyperbolic space, the sum of angles in a triangle is less than 180 degrees, and there are infinitely many parallel lines through a point not on a given line.

```python
import matplotlib.pyplot as plt
import numpy as np

def hyperbolic_line(t):
    return np.cosh(t), np.sinh(t)

t = np.linspace(-2, 2, 100)
x, y = hyperbolic_line(t)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.title("A Hyperbolic Line")
plt.axis('equal')
plt.show()
```

Slide 2: Poincaré Disk Model

The Poincaré disk model represents the entire hyperbolic plane as the interior of a unit disk. In this model, straight lines in hyperbolic space appear as circular arcs perpendicular to the boundary of the disk. This visualization helps us understand the unique properties of hyperbolic geometry.

```python
import matplotlib.pyplot as plt
import numpy as np

def poincare_disk():
    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

def hyperbolic_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    center = (x1*y2 - x2*y1) / (x1*dy - y1*dx), (x2*x2 + y2*y2 - x1*x1 - y1*y1) / (2*(x2*dy - y2*dx))
    radius = np.sqrt((center[0] - x1)**2 + (center[1] - y1)**2)
    return plt.Circle(center, radius, fill=False)

plt.figure(figsize=(8, 8))
poincare_disk()

lines = [
    ((0, 0), (0.8, 0)),
    ((0, 0), (0.6, 0.6)),
    ((0, 0), (0, 0.8)),
    ((0.5, 0), (0.5, 0.8)),
]

for start, end in lines:
    plt.gca().add_artist(hyperbolic_line(start, end))

plt.title("Poincaré Disk Model")
plt.show()
```

Slide 3: Hyperbolic Manifolds

A hyperbolic manifold is a topological space that locally resembles hyperbolic space. These manifolds play a crucial role in geometry, topology, and theoretical physics. They exhibit fascinating properties such as negative curvature and exponential volume growth.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hyperbolic_surface(u, v):
    x = np.sinh(u) * np.cos(v)
    y = np.sinh(u) * np.sin(v)
    z = np.cosh(u)
    return x, y, z

u = np.linspace(0, 2, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

X, Y, Z = hyperbolic_surface(U, V)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("Hyperbolic Surface")
plt.show()
```

Slide 4: Curvature in Hyperbolic Manifolds

The curvature of a hyperbolic manifold is constant and negative. This property leads to unique geometric behaviors, such as exponential divergence of geodesics and the absence of parallel transport. Understanding curvature is crucial for analyzing the global structure of hyperbolic manifolds.

```python
import numpy as np
import matplotlib.pyplot as plt

def geodesic(t, k):
    return np.sinh(k*t) / k, np.cosh(k*t)

t = np.linspace(0, 2, 1000)
k_values = [0.5, 1, 2]

plt.figure(figsize=(10, 6))
for k in k_values:
    x, y = geodesic(t, k)
    plt.plot(x, y, label=f'k = {k}')

plt.title("Geodesics in Hyperbolic Space")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

Slide 5: Fundamental Domain

A fundamental domain is a region of hyperbolic space that, when acted upon by a group of isometries, tiles the entire space without overlaps. This concept is essential for understanding the structure of hyperbolic manifolds and their quotients.

```python
import matplotlib.pyplot as plt
import numpy as np

def poincare_disk():
    circle = plt.Circle((0, 0), 1, fill=False)
    plt.gca().add_artist(circle)
    plt.axis('equal')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

def hyperbolic_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    center = (x1*y2 - x2*y1) / (x1*dy - y1*dx), (x2*x2 + y2*y2 - x1*x1 - y1*y1) / (2*(x2*dy - y2*dx))
    radius = np.sqrt((center[0] - x1)**2 + (center[1] - y1)**2)
    return plt.Circle(center, radius, fill=False)

plt.figure(figsize=(8, 8))
poincare_disk()

# Define a fundamental domain
lines = [
    ((0, 0), (0.5, 0.866)),
    ((0.5, 0.866), (1, 0)),
    ((1, 0), (0, 0)),
]

for start, end in lines:
    plt.gca().add_artist(hyperbolic_line(start, end))

plt.title("Fundamental Domain in Poincaré Disk")
plt.show()
```

Slide 6: Thurston's Geometrization Conjecture

Thurston's geometrization conjecture, now a theorem, states that every closed 3-manifold can be decomposed into geometric pieces, each with one of eight model geometries. Hyperbolic geometry plays a central role in this classification, being the most common and complex of these geometries.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([
    ('S3', 'E3'), ('E3', 'H3'),
    ('S2xE1', 'E3'), ('H2xE1', 'E3'),
    ('S2xE1', 'S3'), ('H2xE1', 'H3'),
    ('SL2R', 'H2xE1'), ('Nil', 'E3'),
    ('Sol', 'E3')
])

pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=10, font_weight='bold')

plt.title("Thurston's Eight Geometries")
plt.axis('off')
plt.show()
```

Slide 7: Hyperbolic 3-Manifolds

Hyperbolic 3-manifolds are particularly important in topology and geometry. They exhibit rich structures and are closely related to knot theory and quantum field theory. The volume of a hyperbolic 3-manifold is a topological invariant, providing a powerful tool for classification.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def figure_eight_knot_complement(u, v):
    x = (np.cos(u) + 2) * np.cos(v)
    y = (np.cos(u) + 2) * np.sin(v)
    z = np.sin(u)
    return x, y, z

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

X, Y, Z = figure_eight_knot_complement(U, V)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("Figure-Eight Knot Complement")
plt.show()
```

Slide 8: Hyperbolic Manifolds in Physics

Hyperbolic manifolds have significant applications in physics, particularly in cosmology and string theory. They provide models for negatively curved spacetimes and play a role in understanding the large-scale structure of the universe.

```python
import numpy as np
import matplotlib.pyplot as plt

def ads_metric(r):
    return 1 / (1 + r**2)

r = np.linspace(0, 5, 1000)
g = ads_metric(r)

plt.figure(figsize=(10, 6))
plt.plot(r, g)
plt.title("Anti-de Sitter (AdS) Metric")
plt.xlabel("r")
plt.ylabel("g(r)")
plt.grid(True)
plt.show()
```

Slide 9: Hyperbolic Structures on Surfaces

Surfaces of genus g > 1 admit hyperbolic structures. These structures are related to Teichmüller space, which parametrizes different hyperbolic metrics on a given surface. Understanding these structures is crucial for the study of Riemann surfaces and complex analysis.

```python
import matplotlib.pyplot as plt
import numpy as np

def hyperbolic_octagon():
    theta = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(theta)
    y = np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, 'k-')
    
    for i in range(8):
        angle = i * np.pi / 4
        plt.plot([0, np.cos(angle)], [0, np.sin(angle)], 'r-')
    
    plt.title("Hyperbolic Structure on a Genus 2 Surface")
    plt.axis('equal')
    plt.axis('off')
    plt.show()

hyperbolic_octagon()
```

Slide 10: Geodesics on Hyperbolic Manifolds

Geodesics on hyperbolic manifolds exhibit unique behaviors, such as exponential divergence and the absence of conjugate points. These properties have profound implications for the dynamics of systems modeled on hyperbolic spaces.

```python
import numpy as np
import matplotlib.pyplot as plt

def hyperbolic_geodesic(t, x0, y0, vx, vy):
    x = x0 * np.cosh(t) + vx * np.sinh(t)
    y = y0 * np.cosh(t) + vy * np.sinh(t)
    return x, y

t = np.linspace(0, 2, 1000)

plt.figure(figsize=(10, 8))
for i in range(5):
    x0, y0 = np.random.rand(2) * 0.2
    vx, vy = np.random.rand(2) - 0.5
    x, y = hyperbolic_geodesic(t, x0, y0, vx, vy)
    plt.plot(x, y)

plt.title("Geodesics on a Hyperbolic Manifold")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 11: Hyperbolic Knot Complements

The complement of a knot in the 3-sphere often admits a hyperbolic structure. This connection between knot theory and hyperbolic geometry has led to powerful invariants and classification results for knots and 3-manifolds.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def trefoil_knot(t):
    x = np.sin(t) + 2 * np.sin(2*t)
    y = np.cos(t) - 2 * np.cos(2*t)
    z = -np.sin(3*t)
    return x, y, z

t = np.linspace(0, 2*np.pi, 1000)
x, y, z = trefoil_knot(t)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_title("Trefoil Knot")
plt.show()
```

Slide 12: Dehn Filling

Dehn filling is a fundamental operation in 3-manifold topology that creates new 3-manifolds from existing ones. This process is intimately connected with hyperbolic geometry and plays a crucial role in the study of hyperbolic 3-manifolds.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def torus(R, r):
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    x = (R + r*np.cos(phi)) * np.cos(theta)
    y = (R + r*np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x, y, z = torus(3, 1)
ax.plot_surface(x, y, z, alpha=0.7)

ax.set_title("Torus Boundary for Dehn Filling")
plt.show()
```

Slide 13: Mostow Rigidity Theorem

The Mostow Rigidity Theorem states that for hyperbolic manifolds of dimension greater than two, the geometric structure is uniquely determined by the topology. This powerful result has far-reaching consequences in the study of hyperbolic manifolds and their invariants.

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([
    ('Topology', 'Hyperbolic Structure'),
    ('Hyperbolic Structure', 'Volume'),
    ('Hyperbolic Structure', 'Geodesics'),
    ('Volume', 'Topological Invariant'),
    ('Geodesics', 'Topological Invariant')
])

pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
        node_size=3000, font_size=10, font_weight='bold')

plt.title("Mostow Rigidity Theorem")
plt.axis('off')
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into the fascinating world of hyperbolic manifolds, here are some valuable resources:

1. Thurston, W. P. (1997). Three-dimensional geometry and topology. Princeton University Press. ArXiv: [https://arxiv.org/abs/math/9712268](https://arxiv.org/abs/math/9712268)
2. Ratcliffe, J. G. (2019). Foundations of hyperbolic manifolds. Springer. ArXiv: [https://arxiv.org/abs/math/0111045](https://arxiv.org/abs/math/0111045)
3. Benedetti, R., & Petronio, C. (1992). Lectures on hyperbolic geometry. Springer. ArXiv: [https://arxiv.org/abs/math/9903079](https://arxiv.org/abs/math/9903079)
4. Martelli, B. (2016). An introduction to geometric topology. ArXiv: [https://arxiv.org/abs/1610.02592](https://arxiv.org/abs/1610.02592)
5. Kapovich, M. (2009). Hyperbolic manifolds and discrete groups. Birkhäuser. ArXiv: [https://arxiv.org/abs/math/0201038](https://arxiv.org/abs/math/0201038)

These resources provide a comprehensive overview of hyperbolic manifolds, from foundational concepts to advanced topics. They offer both theoretical insights and practical applications, suitable for readers at various levels of mathematical sophistication.

