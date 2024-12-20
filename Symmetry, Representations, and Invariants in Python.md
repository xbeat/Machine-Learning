## Symmetry, Representations, and Invariants in Python
Slide 1: Introduction to Symmetry in Mathematics

Symmetry is a fundamental concept in mathematics, describing the invariance of an object under certain transformations. It plays a crucial role in various fields, from geometry to physics. Let's explore symmetry using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple symmetric function
x = np.linspace(-10, 10, 1000)
y = x**2

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Symmetry of y = x^2")
plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.grid(True)
plt.show()
```

Slide 2: Types of Symmetry

Symmetry comes in various forms, including reflection, rotation, and translation. Each type preserves certain properties of the original object. Let's visualize these types using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple triangle
triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# Function to plot triangle
def plot_triangle(tri, color='b'):
    plt.plot(np.append(tri[:, 0], tri[0, 0]), np.append(tri[:, 1], tri[0, 1]), color)

# Original triangle
plt.subplot(131)
plot_triangle(triangle)
plt.title("Original")

# Reflection symmetry
plt.subplot(132)
plot_triangle(triangle)
plot_triangle(triangle * [-1, 1], 'r')
plt.title("Reflection")

# Rotation symmetry
plt.subplot(133)
plot_triangle(triangle)
rotated = np.dot(triangle, [[np.cos(np.pi/3), -np.sin(np.pi/3)], 
                            [np.sin(np.pi/3), np.cos(np.pi/3)]])
plot_triangle(rotated, 'g')
plt.title("Rotation")

plt.tight_layout()
plt.show()
```

Slide 3: Group Theory and Symmetry

Group theory provides a mathematical framework for understanding symmetry. A symmetry group consists of all transformations that leave an object invariant. Let's implement a simple group operation using Python.

```python
class SymmetryGroup:
    def __init__(self, elements):
        self.elements = elements
    
    def operate(self, a, b):
        return (a + b) % len(self.elements)
    
    def print_operation_table(self):
        n = len(self.elements)
        table = [[self.operate(i, j) for j in range(n)] for i in range(n)]
        print("Operation Table:")
        for row in table:
            print(row)

# Create a cyclic group of order 4
C4 = SymmetryGroup([0, 1, 2, 3])
C4.print_operation_table()
```

Slide 4: Representations in Mathematics

Representations allow us to study abstract mathematical structures using concrete objects like matrices. They play a crucial role in physics and chemistry. Let's implement a simple representation of a symmetry group.

```python
import numpy as np

class Representation:
    def __init__(self, group):
        self.group = group
        self.matrices = self._generate_matrices()
    
    def _generate_matrices(self):
        n = len(self.group.elements)
        return [np.eye(n, dtype=int) for _ in range(n)]
    
    def print_matrices(self):
        for i, matrix in enumerate(self.matrices):
            print(f"Matrix for element {self.group.elements[i]}:")
            print(matrix)
            print()

# Create a representation for C4
rep = Representation(C4)
rep.print_matrices()
```

Slide 5: Character Tables

Character tables summarize the properties of representations, providing a compact way to describe symmetry groups. Let's create a simple character table for our C4 group.

```python
import numpy as np

def character_table(group):
    n = len(group.elements)
    table = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            table[i, j] = np.exp(2j * np.pi * i * j / n)
    
    return table

# Generate and print character table for C4
char_table = character_table(C4)
print("Character Table for C4:")
print(char_table)
```

Slide 6: Invariants in Mathematics

Invariants are properties or quantities that remain unchanged under certain transformations. They are crucial in various areas of mathematics and physics. Let's explore a simple invariant: the determinant of a matrix under similarity transformations.

```python
import numpy as np

def is_invariant(A, P):
    """Check if det(A) is invariant under similarity transformation."""
    similarity = np.linalg.inv(P) @ A @ P
    return np.isclose(np.linalg.det(A), np.linalg.det(similarity))

# Example matrix and transformation
A = np.array([[1, 2], [3, 4]])
P = np.array([[2, 1], [1, 1]])

print("Original matrix A:")
print(A)
print("Determinant of A:", np.linalg.det(A))
print("Is determinant invariant?", is_invariant(A, P))
```

Slide 7: Symmetry in Differential Equations

Symmetries in differential equations can help us find solutions or simplify complex problems. Let's implement a simple symmetry analysis for a basic differential equation.

```python
import sympy as sp

def find_scaling_symmetry(equation):
    t, y = sp.symbols('t y')
    f = sp.Function('f')
    
    # Apply scaling transformation
    epsilon = sp.Symbol('epsilon')
    t_scaled = sp.exp(epsilon) * t
    y_scaled = sp.exp(sp.Symbol('a') * epsilon) * y
    
    # Substitute scaled variables into the equation
    eq_scaled = equation.subs({t: t_scaled, y: y_scaled, f(t): y_scaled})
    
    # Find the value of 'a' that preserves the equation
    a_value = sp.solve(sp.expand(eq_scaled) - equation, sp.Symbol('a'))[0]
    
    return a_value

# Example: y' = y/t
eq = sp.Eq(sp.diff(sp.Function('f')(sp.Symbol('t')), sp.Symbol('t')), 
           sp.Function('f')(sp.Symbol('t')) / sp.Symbol('t'))

scaling_factor = find_scaling_symmetry(eq)
print(f"Scaling symmetry: y -> e^({scaling_factor}ε) * y, t -> e^ε * t")
```

Slide 8: Symmetry in Quantum Mechanics

Symmetry principles are fundamental in quantum mechanics, leading to conservation laws and selection rules. Let's implement a simple example of rotational symmetry in a quantum system.

```python
import numpy as np

def rotation_matrix(theta):
    """2D rotation matrix"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def is_rotationally_invariant(wavefunction, theta):
    """Check if a 2D wavefunction is rotationally invariant"""
    x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    psi = wavefunction(x, y)
    
    rotated_coords = np.dot(rotation_matrix(theta), np.array([x.flatten(), y.flatten()]))
    psi_rotated = wavefunction(rotated_coords[0].reshape(x.shape), 
                               rotated_coords[1].reshape(y.shape))
    
    return np.allclose(psi, psi_rotated)

# Example: 2D harmonic oscillator ground state
def harmonic_oscillator_2d(x, y):
    return np.exp(-(x**2 + y**2) / 2)

print("Is 2D harmonic oscillator ground state rotationally invariant?")
print(is_rotationally_invariant(harmonic_oscillator_2d, np.pi/4))
```

Slide 9: Symmetry in Crystallography

Crystallography heavily relies on symmetry to describe and classify crystal structures. Let's implement a simple 2D lattice generator to visualize crystal symmetry.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_2d_lattice(a1, a2, n=5):
    lattice_points = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            point = i*a1 + j*a2
            lattice_points.append(point)
    return np.array(lattice_points)

# Define lattice vectors
a1 = np.array([1, 0])
a2 = np.array([0.5, np.sqrt(3)/2])

# Generate lattice points
lattice = generate_2d_lattice(a1, a2)

# Plot lattice
plt.figure(figsize=(8, 8))
plt.scatter(lattice[:, 0], lattice[:, 1], c='b')
plt.title("2D Hexagonal Lattice")
plt.axis('equal')
plt.grid(True)
plt.show()
```

Slide 10: Symmetry in Data Analysis

Symmetry concepts can be applied in data analysis to detect patterns and reduce dimensionality. Let's implement a simple Principal Component Analysis (PCA) to find symmetries in data.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data with hidden symmetry
np.random.seed(42)
n_samples = 300
t = np.random.uniform(0, 2*np.pi, n_samples)
x = np.cos(t) + 0.1*np.random.randn(n_samples)
y = np.sin(t) + 0.1*np.random.randn(n_samples)
data = np.column_stack((x, y))

# Apply PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title("Original Data")

plt.subplot(122)
plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
plt.title("PCA Transformed Data")

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 11: Symmetry in Graph Theory

Graph theory uses symmetry to analyze network structures. Let's implement a simple algorithm to detect automorphisms in a graph.

```python
import networkx as nx
import matplotlib.pyplot as plt

def is_automorphism(G, mapping):
    return all(set(G.neighbors(v)) == set(mapping[u] for u in G.neighbors(mapping[v]))
               for v in G)

# Create a simple graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

# Define a potential automorphism
automorphism = {1: 2, 2: 1, 3: 3, 4: 4}

# Check if it's a valid automorphism
print("Is the mapping a valid automorphism?", is_automorphism(G, automorphism))

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=16)
plt.title("Graph with Automorphism")
plt.show()
```

Slide 12: Real-life Example: Symmetry in Image Processing

Symmetry plays a crucial role in image processing and computer vision. Let's implement a simple algorithm to detect horizontal symmetry in an image.

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

def detect_horizontal_symmetry(image):
    gray = color.rgb2gray(image)
    height, width = gray.shape
    left_half = gray[:, :width//2]
    right_half = np.fliplr(gray[:, width//2:])
    
    symmetry_score = np.mean(np.abs(left_half - right_half))
    return 1 - symmetry_score  # Higher score means more symmetric

# Load and process an image (replace with your own image path)
image = io.imread('path_to_your_image.jpg')
image = transform.resize(image, (200, 200))

symmetry_score = detect_horizontal_symmetry(image)

plt.imshow(image)
plt.title(f"Horizontal Symmetry Score: {symmetry_score:.2f}")
plt.axis('off')
plt.show()
```

Slide 13: Real-life Example: Symmetry in Music Theory

Symmetry concepts are widely used in music theory and composition. Let's create a simple program to generate a palindromic melody, demonstrating musical symmetry.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def generate_tone(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * frequency * t)

def create_palindrome_melody():
    notes = [261.63, 293.66, 329.63, 349.23, 392.00]  # C4, D4, E4, F4, G4
    durations = [0.5, 0.25, 0.25, 0.5]
    
    melody = []
    for note, duration in zip(notes, durations):
        melody.append(generate_tone(note, duration))
    
    return np.concatenate(melody + melody[::-1])

# Generate and play the melody
melody = create_palindrome_melody()
sample_rate = 44100

# Save the melody as a WAV file
wavfile.write('palindrome_melody.wav', sample_rate, (melody * 32767).astype(np.int16))

# Plot the waveform
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(melody)) / sample_rate, melody)
plt.title('Palindromic Melody Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

print("Melody saved as 'palindrome_melody.wav'")
```

Slide 14: Additional Resources

For further exploration of symmetry, representations, and invariants in mathematics and physics, consider the following resources:

1. "Group Theory in Physics" by John F. Cornwell (ArXiv:physics/9709043)
2. "Symmetry and the Standard Model" by Matthew Robinson (ArXiv:1112.4888)
3. "Invariants, Symmetry, and Conservation Laws" by Peter J. Olver (ArXiv:math-ph/0107008)
4. "Representation Theory: A First Course" by William Fulton and Joe Harris
5. "Symmetry Methods for Differential Equations: A Beginner's Guide" by Peter E. Hydon

These resources provide in-depth discussions on the topics covered in this presentation and can help deepen your understanding of these fundamental concepts.

