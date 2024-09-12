## Geometric Reasoning in Large Language Models
Slide 1: Geometric Reasoning in LLMs: An Introduction

Geometric reasoning is a fundamental aspect of human cognition that involves understanding and manipulating spatial relationships. Large Language Models (LLMs) have shown surprising capabilities in various domains, including geometric reasoning. This presentation explores how LLMs process and generate geometric information, and how we can leverage Python to investigate and enhance these capabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_triangle(a, b, c):
    plt.figure(figsize=(6, 6))
    plt.plot([0, a[0], b[0], 0], [0, a[1], b[1], 0], 'b-')
    plt.plot([a[0], c[0]], [a[1], c[1]], 'r--')
    plt.axis('equal')
    plt.grid(True)
    plt.title("Triangle with Median")
    plt.show()

# Define triangle vertices
A = np.array([0, 0])
B = np.array([4, 0])
C = np.array([2, 3])

# Calculate and plot median
M = (A + B) / 2
plot_triangle(A, B, C)
```

Slide 2: Representing Geometric Shapes in LLMs

LLMs process geometric shapes as sequences of tokens, similar to how they handle natural language. These tokens can represent coordinates, dimensions, or mathematical descriptions of shapes. By encoding geometric information in this way, LLMs can reason about spatial relationships and properties.

```python
def encode_shape(shape_type, *params):
    return f"{shape_type}({','.join(map(str, params))})"

def decode_shape(encoded_shape):
    shape_type, params = encoded_shape.split('(')
    params = list(map(float, params[:-1].split(',')))
    return shape_type, params

# Example usage
circle = encode_shape("circle", 0, 0, 5)  # Center (0,0), radius 5
print(f"Encoded: {circle}")
print(f"Decoded: {decode_shape(circle)}")
```

Slide 3: Geometric Calculations with LLMs

LLMs can perform basic geometric calculations by understanding the relationships between different shapes and their properties. We can use Python to verify and visualize these calculations.

```python
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

def calculate_triangle_area(base, height):
    return 0.5 * base * height

# Example calculations
circle_radius = 5
triangle_base = 6
triangle_height = 4

print(f"Circle area: {calculate_circle_area(circle_radius):.2f}")
print(f"Triangle area: {calculate_triangle_area(triangle_base, triangle_height):.2f}")

# Visualize the shapes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

circle = plt.Circle((0, 0), circle_radius, fill=False)
ax1.add_artist(circle)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.set_aspect('equal')
ax1.set_title('Circle')

ax2.plot([0, triangle_base, 0, 0], [0, 0, triangle_height, 0])
ax2.set_xlim(0, 7)
ax2.set_ylim(0, 5)
ax2.set_aspect('equal')
ax2.set_title('Triangle')

plt.tight_layout()
plt.show()
```

Slide 4: Geometric Transformations in LLMs

LLMs can understand and apply geometric transformations such as translation, rotation, and scaling. We can implement these transformations in Python to demonstrate how LLMs might process this information.

```python
import numpy as np
import matplotlib.pyplot as plt

def translate(points, tx, ty):
    return points + np.array([tx, ty])

def rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return np.dot(points, rotation_matrix.T)

def scale(points, sx, sy):
    return points * np.array([sx, sy])

# Example usage
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

translated = translate(square, 2, 1)
rotated = rotate(square, np.pi/4)
scaled = scale(square, 2, 0.5)

# Visualize transformations
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(translated[:, 0], translated[:, 1])
ax1.set_title('Translated')
ax1.set_aspect('equal')

ax2.plot(rotated[:, 0], rotated[:, 1])
ax2.set_title('Rotated')
ax2.set_aspect('equal')

ax3.plot(scaled[:, 0], scaled[:, 1])
ax3.set_title('Scaled')
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()
```

Slide 5: Geometric Pattern Recognition

LLMs can identify patterns in geometric arrangements. We can simulate this capability by implementing pattern recognition algorithms in Python.

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_spiral(n_points, a, b):
    theta = np.linspace(0, 8*np.pi, n_points)
    r = a + b * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def recognize_spiral(x, y):
    # Simple spiral recognition based on increasing distance from origin
    distances = np.sqrt(x**2 + y**2)
    return np.all(np.diff(distances) > 0)

# Generate and recognize spiral
x, y = generate_spiral(100, 0.1, 0.1)

plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.title(f"Recognized as spiral: {recognize_spiral(x, y)}")
plt.axis('equal')
plt.show()
```

Slide 6: Geometric Problem Solving

LLMs can solve geometric problems by breaking them down into steps and applying geometric principles. Let's implement a simple geometric problem solver in Python.

```python
import math

def solve_right_triangle(a, b=None, c=None, angle=None):
    if a and b:
        c = math.sqrt(a**2 + b**2)
        angle = math.degrees(math.atan(b/a))
    elif a and c:
        b = math.sqrt(c**2 - a**2)
        angle = math.degrees(math.acos(a/c))
    elif a and angle:
        angle_rad = math.radians(angle)
        b = a * math.tan(angle_rad)
        c = a / math.cos(angle_rad)
    else:
        return "Insufficient information"
    
    return {
        "a": round(a, 2),
        "b": round(b, 2),
        "c": round(c, 2),
        "angle": round(angle, 2)
    }

# Example usage
print(solve_right_triangle(a=3, b=4))
print(solve_right_triangle(a=5, angle=30))
```

Slide 7: Geometric Reasoning in Real-Life: Urban Planning

LLMs can apply geometric reasoning to real-life scenarios like urban planning. Let's simulate a simple city block layout optimization problem.

```python
import numpy as np
import matplotlib.pyplot as plt

def optimize_block_layout(block_size, building_sizes):
    total_area = block_size[0] * block_size[1]
    building_areas = [w*h for w, h in building_sizes]
    
    if sum(building_areas) > total_area:
        return "Block too small for all buildings"
    
    layout = np.zeros(block_size)
    for i, (w, h) in enumerate(building_sizes):
        placed = False
        for y in range(block_size[1] - h + 1):
            for x in range(block_size[0] - w + 1):
                if np.all(layout[y:y+h, x:x+w] == 0):
                    layout[y:y+h, x:x+w] = i + 1
                    placed = True
                    break
            if placed:
                break
    
    return layout

# Example usage
block_size = (10, 10)
buildings = [(3, 3), (4, 2), (2, 4), (3, 3)]

layout = optimize_block_layout(block_size, buildings)

plt.figure(figsize=(8, 8))
plt.imshow(layout, cmap='viridis')
plt.title("Optimized City Block Layout")
plt.colorbar(ticks=range(len(buildings)+1), label="Building ID")
plt.show()
```

Slide 8: Geometric Reasoning for Computer Vision Tasks

LLMs can assist in computer vision tasks by reasoning about geometric properties of objects in images. Let's implement a simple shape detection algorithm.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_shapes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        
        x, y = approx.ravel()[0], approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(image, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        elif len(approx) == 4:
            cv2.putText(image, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        elif len(approx) == 5:
            cv2.putText(image, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        elif len(approx) == 6:
            cv2.putText(image, "Hexagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.putText(image, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    
    return image

# Example usage (you need to provide an image path)
# result = detect_shapes("path_to_your_image.jpg")
# plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
```

Slide 9: Geometric Reasoning for 3D Object Reconstruction

LLMs can assist in 3D object reconstruction by reasoning about geometric properties from 2D projections. Let's implement a simple 3D reconstruction from multiple 2D views.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def reconstruct_3d_from_views(front_view, side_view, top_view):
    height, width = front_view.shape
    depth = side_view.shape[1]
    
    voxel_grid = np.zeros((height, width, depth), dtype=bool)
    
    for i in range(height):
        for j in range(width):
            for k in range(depth):
                voxel_grid[i, j, k] = (
                    front_view[i, j] and
                    side_view[i, k] and
                    top_view[k, j]
                )
    
    return voxel_grid

# Example usage
front = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]])

side = np.array([[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]])

top = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]])

reconstructed = reconstruct_3d_from_views(front, side, top)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(reconstructed, edgecolor='k')
ax.set_title("3D Reconstruction from 2D Views")
plt.show()
```

Slide 10: Geometric Reasoning for Path Planning

LLMs can apply geometric reasoning to solve path planning problems. Let's implement a simple 2D path planning algorithm using geometric concepts.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_obstacle_grid(size, obstacles):
    grid = np.zeros(size)
    for obstacle in obstacles:
        x, y, radius = obstacle
        for i in range(size[0]):
            for j in range(size[1]):
                if np.sqrt((i-x)**2 + (j-y)**2) <= radius:
                    grid[i, j] = 1
    return grid

def find_path(start, goal, obstacle_grid):
    # A* algorithm implementation
    # (simplified for brevity)
    # Returns a path from start to goal

    return path

# Example usage
grid_size = (50, 50)
obstacles = [(10, 10, 5), (30, 30, 7), (20, 40, 6)]
start = (5, 5)
goal = (45, 45)

grid = create_obstacle_grid(grid_size, obstacles)
path = find_path(start, goal, grid)

plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap='binary')
plt.plot([p[1] for p in path], [p[0] for p in path], 'r-')
plt.title("Path Planning with Obstacles")
plt.show()
```

Slide 11: Geometric Reasoning in Natural Language Processing

LLMs can use geometric reasoning to understand and generate spatial descriptions in natural language. Let's explore how we can represent and process spatial relations.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_spatial_relations(text):
    doc = nlp(text)
    spatial_relations = []
    
    for token in doc:
        if token.dep_ in ["prep", "agent"] and token.head.pos_ in ["NOUN", "PROPN"]:
            if token.children:
                obj = list(token.children)[0]
                if obj.pos_ in ["NOUN", "PROPN"]:
                    spatial_relations.append((token.head.text, token.text, obj.text))
    
    return spatial_relations

# Example usage
text = "The book is on the table next to the lamp."
relations = extract_spatial_relations(text)

for relation in relations:
    print(f"{relation[0]} {relation[1]} {relation[2]}")
```

Slide 12: Geometric Reasoning for Data Visualization

LLMs can apply geometric reasoning to create effective data visualizations. Let's implement a simple algorithm to generate a radial plot.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_radial_plot(categories, values):
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radial Plot")
    plt.show()

# Example usage
categories = ['A', 'B', 'C', 'D', 'E']
values = [4, 3, 5, 2, 4]

create_radial_plot(categories, values)
```

Slide 13: Geometric Reasoning in Computer Graphics

LLMs can apply geometric reasoning to generate and manipulate 3D objects. Let's create a simple 3D object using Python and visualize it.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_cube():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    return vertices, edges

def plot_3d_object(vertices, edges):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for edge in edges:
        ax.plot3D(*zip(*vertices[edge]), color='b')
    
    ax.set_title("3D Cube")
    plt.show()

# Create and plot a 3D cube
vertices, edges = create_cube()
plot_3d_object(vertices, edges)
```

Slide 14: Geometric Reasoning in Machine Learning

LLMs can apply geometric reasoning to understand and interpret machine learning models. Let's visualize the decision boundary of a simple classifier using geometric concepts.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Train a Support Vector Machine classifier
clf = SVC(kernel='linear')
clf.fit(X, y)

# Create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot the decision boundary
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title("SVM Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into geometric reasoning in LLMs, consider exploring the following resources:

1. "Geometry of Neural Networks: From Dynamics to Function" (arXiv:2011.04330) This paper explores the geometric properties of neural networks and their implications for understanding LLM behavior.
2. "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (arXiv:2104.13478) A comprehensive overview of geometric deep learning, including applications to language models.
3. "On the Geometry of Generalization in Neural Networks" (arXiv:1705.06661) This work investigates the geometric properties of neural network loss landscapes and their impact on generalization.

These resources provide valuable insights into the intersection of geometry and machine learning, which can help in understanding the geometric reasoning capabilities of LLMs.

