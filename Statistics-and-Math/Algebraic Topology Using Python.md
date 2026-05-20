## Algebraic Topology Using Python

Slide 1: Introduction to Algebraic Topology

Algebraic topology is a branch of mathematics that uses algebraic structures to study topological spaces. It provides powerful tools for analyzing the shape and structure of geometric objects.

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# Draw the graph
nx.draw(G, with_labels=True)
plt.title("A Simple Graph Representation")
plt.show()
```

Slide 2: Simplicial Complexes

Simplicial complexes are fundamental objects in algebraic topology, representing higher-dimensional generalizations of graphs.

```python
from scipy.spatial import Delaunay
import numpy as np

# Generate random points in 2D
points = np.random.rand(20, 2)

# Create Delaunay triangulation
tri = Delaunay(points)

# Plot the triangulation
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.title("2D Simplicial Complex")
plt.show()
```

Slide 3: Homology Groups

Homology groups are algebraic structures that capture the essence of holes in topological spaces. They provide a way to count and classify different types of holes.

```python
import gudhi

# Create a simplicial complex
simplex_tree = gudhi.SimplexTree()
simplex_tree.insert([1, 2, 3])
simplex_tree.insert([2, 3, 4])
simplex_tree.insert([3, 4, 5])

# Compute homology
homology = simplex_tree.persistence()

print("Homology groups:")
for interval in homology:
    dim, (birth, death) = interval
    print(f"Dimension {dim}: birth = {birth}, death = {death}")
```

Slide 4: Fundamental Group

The fundamental group is a topological invariant that captures information about loops in a space.

```python
import sympy as sp

def fundamental_group_presentation(generators, relations):
    G = sp.generators(' '.join(generators))
    return sp.presentation_free(G, relations)

# Example: Fundamental group of a torus
torus_group = fundamental_group_presentation(['a', 'b'], ['a*b*a^-1*b^-1'])
print("Fundamental group of a torus:", torus_group)
```

Slide 5: Euler Characteristic

The Euler characteristic is a topological invariant that relates the number of vertices, edges, and faces in a polyhedron.

```python
def euler_characteristic(vertices, edges, faces):
    return vertices - edges + faces

# Example: Cube
cube_vertices = 8
cube_edges = 12
cube_faces = 6

print("Euler characteristic of a cube:", 
      euler_characteristic(cube_vertices, cube_edges, cube_faces))
```

Slide 6: Betti Numbers

Betti numbers are topological invariants that count the number of holes in each dimension of a space.

```python
import numpy as np
from scipy.spatial import Delaunay
from persim import plot_diagrams
import ripser

# Generate a point cloud in the shape of a circle
t = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack((np.cos(t), np.sin(t)))

# Compute persistent homology
diagrams = ripser.ripser(circle)['dgms']

# Plot persistence diagrams
plot_diagrams(diagrams, show=True)
```

Slide 7: Homotopy Equivalence

Homotopy equivalence is a relation between topological spaces that preserves their essential topological properties.

```python
import networkx as nx
import matplotlib.pyplot as plt

def contract_edge(G, u, v):
    G.add_edges_from((u, w) for w in G.neighbors(v) if w != u)
    G.remove_node(v)

# Create two homotopy equivalent graphs
G1 = nx.cycle_graph(4)
G2 = nx.cycle_graph(4)
contract_edge(G2, 0, 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
nx.draw(G1, with_labels=True, ax=ax1)
ax1.set_title("Original Graph")
nx.draw(G2, with_labels=True, ax=ax2)
ax2.set_title("Homotopy Equivalent Graph")
plt.show()
```

Slide 8: Cohomology

Cohomology is the dual notion to homology, providing additional algebraic tools for studying topological spaces.

```python
import numpy as np
from scipy.linalg import null_space

def compute_cohomology(boundary_matrix):
    kernel = null_space(boundary_matrix.T)
    image = np.column_stack([boundary_matrix[:, i] 
                             for i in range(boundary_matrix.shape[1])])
    
    betti_number = kernel.shape[1] - np.linalg.matrix_rank(image)
    return betti_number

# Example: Compute 1-dimensional cohomology of a triangle
boundary_matrix = np.array([
    [-1, 1, 0],
    [-1, 0, 1],
    [0, -1, 1]
])

print("1-dimensional cohomology:", compute_cohomology(boundary_matrix))
```

Slide 9: Persistent Homology

Persistent homology is a method for computing topological features of a space at different spatial resolutions.

```python
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# Generate a noisy circle
t = np.linspace(0, 2*np.pi, 100)
circle = np.column_stack((np.cos(t), np.sin(t)))
noisy_circle = circle + 0.1 * np.random.randn(*circle.shape)

# Compute persistent homology
diagrams = ripser(noisy_circle)['dgms']

# Plot the persistence diagram
plot_diagrams(diagrams, show=True)
```

Slide 10: Simplicial Homology Computation

Simplicial homology is a concrete way to compute homology groups for simplicial complexes.

```python
import numpy as np
from scipy.linalg import null_space

def simplicial_homology(boundary_matrix):
    kernel = null_space(boundary_matrix)
    image = np.column_stack([boundary_matrix[:, i] 
                             for i in range(boundary_matrix.shape[1])])
    
    betti_number = kernel.shape[1] - np.linalg.matrix_rank(image)
    return betti_number

# Example: Compute 1-dimensional homology of a triangle
boundary_matrix = np.array([
    [-1, -1, 0],
    [1, 0, -1],
    [0, 1, 1]
])

print("1-dimensional homology:", simplicial_homology(boundary_matrix))
```

Slide 11: Topological Data Analysis

Topological Data Analysis (TDA) applies techniques from algebraic topology to analyze and extract insights from complex datasets.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from ripser import ripser
from persim import plot_diagrams

# Generate a noisy dataset
X, _ = datasets.make_circles(n_samples=300, noise=0.05, factor=0.5)

# Compute persistent homology
diagrams = ripser(X)['dgms']

# Plot the dataset and persistence diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X[:, 0], X[:, 1])
ax1.set_title("Noisy Circles Dataset")
plot_diagrams(diagrams, show=False, ax=ax2)
ax2.set_title("Persistence Diagram")
plt.show()
```

Slide 12: Mapping Cylinder

The mapping cylinder is a construction in algebraic topology that helps visualize continuous maps between topological spaces.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mapping_cylinder(t, theta):
    x = (1 - t) * np.cos(theta)
    y = (1 - t) * np.sin(theta)
    z = t
    return x, y, z

t = np.linspace(0, 1, 100)
theta = np.linspace(0, 2*np.pi, 100)
T, Theta = np.meshgrid(t, theta)

X, Y, Z = mapping_cylinder(T, Theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title("Mapping Cylinder")
plt.show()
```

Slide 13: Real-Life Example: Protein Structure Analysis

Algebraic topology can be applied to analyze the structure of proteins, helping to understand their function and interactions.

```python
from Bio.PDB import PDBParser
import numpy as np
from ripser import ripser
from persim import plot_diagrams

# Load a protein structure (you need to download a PDB file)
parser = PDBParser()
structure = parser.get_structure("protein", "path/to/protein.pdb")

# Extract alpha carbon coordinates
coords = np.array([atom.coord for atom in structure.get_atoms() if atom.name == 'CA'])

# Compute persistent homology
diagrams = ripser(coords)['dgms']

# Plot persistence diagram
plot_diagrams(diagrams, show=True)
```

Slide 14: Real-Life Example: Network Analysis

Algebraic topology techniques can be used to analyze complex networks, revealing hidden structures and patterns.

```python
import networkx as nx
import numpy as np
from ripser import ripser
from persim import plot_diagrams

# Generate a random graph
G = nx.erdos_renyi_graph(100, 0.1)

# Compute the adjacency matrix
adj_matrix = nx.adjacency_matrix(G).todense()

# Compute persistent homology
diagrams = ripser(adj_matrix, distance_matrix=True)['dgms']

# Plot persistence diagram
plot_diagrams(diagrams, show=True)
```

Slide 15: Additional Resources

For further exploration of Algebraic Topology:

1. "Computational Topology: An Introduction" by Herbert Edelsbrunner and John Harer ArXiv: [https://arxiv.org/abs/cs/0503002](https://arxiv.org/abs/cs/0503002)
2. "Algebraic Topology" by Allen Hatcher Available online: [https://pi.math.cornell.edu/~hatcher/AT/ATpage.html](https://pi.math.cornell.edu/~hatcher/AT/ATpage.html)
3. "Topological Data Analysis" by Gunnar Carlsson ArXiv: [https://arxiv.org/abs/0907.2721](https://arxiv.org/abs/0907.2721)

These resources provide in-depth coverage of the topics discussed in this presentation and offer advanced concepts for further study.

