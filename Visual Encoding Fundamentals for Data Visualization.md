## Visual Encoding Fundamentals for Data Visualization
Slide 1: Visual Encoding Foundations in Python

Data visualization requires systematic mapping between data attributes and visual properties. Understanding how to programmatically implement these mappings forms the foundation for creating effective visualizations that leverage human perceptual capabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

class VisualEncoder:
    def __init__(self, data):
        self.data = np.array(data)
        self.min_val = np.min(data)
        self.max_val = np.max(data)
    
    def normalize(self):
        return (self.data - self.min_val) / (self.max_val - self.min_val)
    
    def encode_size(self, min_size=50, max_size=500):
        normalized = self.normalize()
        return min_size + normalized * (max_size - min_size)
    
    def encode_color(self, colormap='viridis'):
        normalized = self.normalize()
        return plt.cm.get_cmap(colormap)(normalized)

# Example usage
data = [10, 25, 45, 60, 90]
encoder = VisualEncoder(data)
sizes = encoder.encode_size()
colors = encoder.encode_color()

plt.scatter(range(len(data)), data, s=sizes, c=colors)
plt.show()
```

Slide 2: Retinal Variable Implementation

Retinal variables represent the fundamental visual properties that our visual system can perceive pre-attentively. This implementation demonstrates how to systematically map data values to position, size, and color intensity simultaneously.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_retinal_encoding(values, feature='all'):
    normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
    
    if feature == 'position':
        return normalized * 100
    elif feature == 'size':
        return 20 + normalized * 180
    elif feature == 'color':
        return plt.cm.viridis(normalized)
    else:
        return {
            'position': normalized * 100,
            'size': 20 + normalized * 180,
            'color': plt.cm.viridis(normalized)
        }

# Example with multiple encodings
data = np.array([15, 35, 55, 75, 95])
encodings = create_retinal_encoding(data)

plt.figure(figsize=(10, 6))
plt.scatter(encodings['position'], 
           np.ones_like(data), 
           s=encodings['size'], 
           c=encodings['color'])
plt.ylim(0, 2)
plt.show()
```

Slide 3: Spatial Encoding Generation

Spatial encodings involve the systematic organization of visual elements in space. This implementation creates a framework for positioning visual elements based on data relationships and hierarchical structures.

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SpatialEncoder:
    def __init__(self, data_matrix):
        self.data = np.array(data_matrix)
        self.graph = nx.from_numpy_array(self.data)
    
    def hierarchical_layout(self):
        return nx.spring_layout(self.graph)
    
    def force_directed_layout(self):
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        return pos
    
    def circular_layout(self):
        return nx.circular_layout(self.graph)

# Example usage
data_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
])

encoder = SpatialEncoder(data_matrix)
layout = encoder.force_directed_layout()

plt.figure(figsize=(8, 8))
nx.draw(encoder.graph, layout, node_size=500, 
        node_color='lightblue', with_labels=True)
plt.show()
```

Slide 4: Color Encoding Strategies

Color encoding requires careful consideration of perceptual principles and color spaces. This implementation provides tools for creating effective color mappings that account for both sequential and diverging data patterns.

```python
import numpy as np
import matplotlib.pyplot as plt
import colorsys

class ColorEncoder:
    def __init__(self, data_range=(0, 1)):
        self.min_val, self.max_val = data_range
        
    def sequential_colormap(self, n_colors=256):
        hsv_colors = [(0.6, i/n_colors, 1.0) for i in range(n_colors)]
        rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]
        return LinearSegmentedColormap.from_list('sequential', rgb_colors)
    
    def diverging_colormap(self, n_colors=256):
        half = n_colors // 2
        neg_colors = [(0.0, i/half, 1.0) for i in range(half)]
        pos_colors = [(0.6, i/half, 1.0) for i in range(half)]
        rgb_colors = [colorsys.hsv_to_rgb(*hsv) for hsv in neg_colors + pos_colors]
        return LinearSegmentedColormap.from_list('diverging', rgb_colors)

# Example usage
data = np.random.normal(size=(10, 10))
encoder = ColorEncoder(data_range=(np.min(data), np.max(data)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(data, cmap=encoder.sequential_colormap())
ax2.imshow(data, cmap=encoder.diverging_colormap())
plt.show()
```

Slide 5: Shape Encoding Framework

Shape encoding provides a systematic approach to mapping categorical data to distinct visual forms. This implementation creates a flexible framework for generating and managing different shape encodings while maintaining perceptual effectiveness.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

class ShapeEncoder:
    def __init__(self):
        self.shapes = {
            'triangle': self._create_triangle(),
            'square': self._create_square(),
            'diamond': self._create_diamond(),
            'pentagon': self._create_pentagon()
        }
    
    def _create_triangle(self):
        return np.array([[0., -0.5], [0.5, 0.5], [-0.5, 0.5]])
    
    def _create_square(self):
        return np.array([[-0.5, -0.5], [0.5, -0.5], 
                        [0.5, 0.5], [-0.5, 0.5]])
    
    def _create_diamond(self):
        return np.array([[0., -0.5], [0.5, 0.], 
                        [0., 0.5], [-0.5, 0.]])
    
    def _create_pentagon(self):
        angles = np.linspace(0, 2*np.pi, 6)[:-1]
        return 0.5 * np.array([[np.cos(a), np.sin(a)] for a in angles])

    def plot_shapes(self, categories, positions):
        fig, ax = plt.subplots(figsize=(10, 10))
        for cat, pos in zip(categories, positions):
            shape = self.shapes[cat]
            path = Path(shape + pos)
            patch = patches.PathPatch(path, facecolor='blue', alpha=0.6)
            ax.add_patch(patch)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return fig, ax

# Example usage
encoder = ShapeEncoder()
categories = ['triangle', 'square', 'diamond', 'pentagon']
positions = np.array([[0,0], [1,1], [-1,1], [0,-1]])
encoder.plot_shapes(categories, positions)
plt.show()
```

Slide 6: Multi-dimensional Visual Encoding

Implementing multi-dimensional visual encoding requires careful consideration of how different visual channels can work together effectively. This implementation demonstrates combining multiple visual variables to represent complex data relationships.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class MultiDimensionalEncoder:
    def __init__(self, data):
        self.data = np.array(data)
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
    
    def encode_dimensions(self, x_dim=0, y_dim=1, 
                         size_dim=2, color_dim=3):
        encoded = {
            'x': self.scaled_data[:, x_dim],
            'y': self.scaled_data[:, y_dim],
            'sizes': 100 + self.scaled_data[:, size_dim] * 400,
            'colors': plt.cm.viridis(self.scaled_data[:, color_dim])
        }
        return encoded
    
    def plot_encoding(self, x_dim=0, y_dim=1, 
                     size_dim=2, color_dim=3):
        encoded = self.encode_dimensions(x_dim, y_dim, 
                                      size_dim, color_dim)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(encoded['x'], encoded['y'],
                   s=encoded['sizes'],
                   c=encoded['colors'])
        plt.colorbar(label=f'Dimension {color_dim}')
        plt.xlabel(f'Dimension {x_dim}')
        plt.ylabel(f'Dimension {y_dim}')
        return plt

# Example usage
data = np.random.rand(50, 5)  # 5-dimensional data
encoder = MultiDimensionalEncoder(data)
encoder.plot_encoding()
plt.show()
```

Slide 7: Time Series Visual Encoding

Time series data requires specialized visual encoding techniques to effectively communicate temporal patterns and relationships. This implementation provides a framework for encoding time-based data with multiple visual variables.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class TimeSeriesEncoder:
    def __init__(self, timestamps, values):
        self.df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'value': values
        })
        self.normalize_values()
    
    def normalize_values(self):
        self.df['normalized'] = (self.df['value'] - self.df['value'].min()) / \
                               (self.df['value'].max() - self.df['value'].min())
    
    def encode_time_series(self, style='composite'):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if style == 'composite':
            scatter = ax.scatter(self.df['timestamp'], 
                               self.df['value'],
                               c=self.df['normalized'],
                               s=100 * self.df['normalized'],
                               cmap='viridis')
            ax.plot(self.df['timestamp'], 
                   self.df['value'], 
                   alpha=0.3)
            plt.colorbar(scatter, label='Normalized Value')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        return fig, ax

# Example usage
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.sin(np.linspace(0, 4*np.pi, 100)) + \
         np.random.normal(0, 0.1, 100)

encoder = TimeSeriesEncoder(dates, values)
encoder.encode_time_series()
plt.show()
```

Slide 8: Hierarchical Data Encoding

Hierarchical data structures require specialized visual encoding techniques to effectively represent relationships between different levels. This implementation creates a framework for encoding tree-like data structures with multiple visual attributes.

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class HierarchicalEncoder:
    def __init__(self, adjacency_list, node_values):
        self.G = nx.DiGraph(adjacency_list)
        self.node_values = node_values
        self.norm = Normalize(vmin=min(node_values.values()), 
                            vmax=max(node_values.values()))
    
    def encode_hierarchy(self):
        pos = nx.spring_layout(self.G, k=1, iterations=50)
        
        # Calculate node sizes and colors based on values
        node_sizes = {node: 1000 * self.norm(val) 
                     for node, val in self.node_values.items()}
        node_colors = {node: plt.cm.viridis(self.norm(val)) 
                      for node, val in self.node_values.items()}
        
        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos,
                node_size=[node_sizes[node] for node in self.G.nodes()],
                node_color=[node_colors[node] for node in self.G.nodes()],
                with_labels=True,
                arrows=True,
                edge_color='gray',
                alpha=0.7)
        
        return plt

# Example usage
adjacency_list = [
    (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)
]
node_values = {i: np.random.rand() for i in range(1, 8)}

encoder = HierarchicalEncoder(adjacency_list, node_values)
encoder.encode_hierarchy()
plt.show()
```

Slide 9: Pattern-Based Encoding

Pattern-based encoding leverages visual textures and repeating elements to represent data characteristics. This implementation provides methods for creating and applying various pattern encodings to represent both categorical and continuous data.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.path import Path

class PatternEncoder:
    def __init__(self, data_range=(0, 1)):
        self.min_val, self.max_val = data_range
        self.patterns = {
            'dots': self._create_dot_pattern,
            'stripes': self._create_stripe_pattern,
            'grid': self._create_grid_pattern
        }
    
    def _create_dot_pattern(self, density):
        def pattern(x, y, w, h):
            points = []
            spacing = np.sqrt(1/density) * 0.1
            for i in np.arange(x, x+w, spacing):
                for j in np.arange(y, y+h, spacing):
                    points.append(Circle((i, j), spacing/4))
            return points
        return pattern
    
    def _create_stripe_pattern(self, density):
        def pattern(x, y, w, h):
            lines = []
            spacing = 1/density * 0.1
            for i in np.arange(x, x+w, spacing):
                lines.append(Rectangle((i, y), spacing/3, h))
            return lines
        return pattern
    
    def encode_value(self, value, pattern_type='dots'):
        normalized = (value - self.min_val)/(self.max_val - self.min_val)
        density = 1 + normalized * 9  # Map to density range [1, 10]
        return self.patterns[pattern_type](density)

# Example usage
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
encoder = PatternEncoder()

for idx, (pattern_name, value) in enumerate([
    ('dots', 0.3), ('stripes', 0.6), ('grid', 0.9)
]):
    pattern_func = encoder.encode_value(value, pattern_name)
    patterns = pattern_func(0, 0, 1, 1)
    for pattern in patterns:
        axs[idx].add_patch(pattern)
    axs[idx].set_xlim(-0.1, 1.1)
    axs[idx].set_ylim(-0.1, 1.1)
    axs[idx].set_title(f'{pattern_name.capitalize()} Pattern')

plt.show()
```

Slide 10: Motion and Animation Encoding

While static visualizations are important, motion and animation can add an additional dimension to data encoding. This implementation demonstrates how to create animated visualizations that encode data through movement patterns.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

class MotionEncoder:
    def __init__(self, data, time_steps):
        self.data = np.array(data)
        self.time_steps = time_steps
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    
    def create_animation(self, interval=50):
        scatter = self.ax.scatter([], [], c=[], s=[], cmap='viridis')
        
        def init():
            self.ax.set_xlim(0, self.time_steps)
            self.ax.set_ylim(np.min(self.data)-1, np.max(self.data)+1)
            return scatter,
        
        def update(frame):
            positions = np.arange(frame+1)
            values = self.data[:frame+1]
            scatter.set_offsets(np.c_[positions, values])
            scatter.set_array(self.norm(values))
            scatter.set_sizes(100 * self.norm(values))
            return scatter,
        
        anim = FuncAnimation(self.fig, update, frames=self.time_steps,
                            init_func=init, interval=interval, blit=True)
        return anim

# Example usage
time_series = np.sin(np.linspace(0, 4*np.pi, 100)) + \
              np.random.normal(0, 0.1, 100)
encoder = MotionEncoder(time_series, len(time_series))
anim = encoder.create_animation()
plt.colorbar(plt.cm.ScalarMappable(norm=encoder.norm, cmap='viridis'))
plt.show()
```

Slide 11: Texture-Based Data Encoding

Texture-based encoding provides an alternative way to represent data through varying surface patterns. This implementation creates a framework for generating and applying procedural textures based on data values.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class TextureEncoder:
    def __init__(self, resolution=100):
        self.resolution = resolution
        self.x = np.linspace(0, 1, resolution)
        self.y = np.linspace(0, 1, resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def generate_texture(self, value, pattern_type='noise'):
        if pattern_type == 'noise':
            return self._perlin_noise(value)
        elif pattern_type == 'waves':
            return self._wave_pattern(value)
        elif pattern_type == 'dots':
            return self._dot_pattern(value)
    
    def _perlin_noise(self, frequency):
        noise = np.zeros((self.resolution, self.resolution))
        frequency = 1 + frequency * 20
        
        for i in range(6):
            freq = frequency * (2 ** i)
            noise += (1 / (2 ** i)) * np.random.randn(
                self.resolution, self.resolution
            )
            noise = signal.convolve2d(
                noise, 
                np.ones((3,3))/9, 
                mode='same'
            )
        return noise
    
    def _wave_pattern(self, frequency):
        frequency = 1 + frequency * 10
        return np.sin(2 * np.pi * frequency * self.X) * \
               np.cos(2 * np.pi * frequency * self.Y)
    
    def _dot_pattern(self, density):
        density = 1 + density * 10
        return np.sin(2 * np.pi * density * self.X) ** 2 * \
               np.sin(2 * np.pi * density * self.Y) ** 2

# Example usage
encoder = TextureEncoder()
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

patterns = ['noise', 'waves', 'dots']
values = [0.2, 0.5, 0.8]

for ax, (pattern, value) in zip(axes.ravel(), 
                               zip(patterns, values)):
    texture = encoder.generate_texture(value, pattern)
    im = ax.imshow(texture, cmap='viridis')
    ax.set_title(f'{pattern.capitalize()} Pattern\nValue: {value}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

Slide 12: Composite Visual Encoding

Composite visual encoding combines multiple encoding techniques to represent complex data relationships. This implementation demonstrates how to create sophisticated visualizations that leverage multiple visual channels simultaneously.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Rectangle

class CompositeEncoder:
    def __init__(self, data_matrix):
        self.data = np.array(data_matrix)
        self.dim = self.data.shape[1]
    
    def encode_composite(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        patches = []
        colors = []
        sizes = []
        
        for i, row in enumerate(self.data):
            # Position encoding
            x = row[0]  # First dimension for x
            y = row[1]  # Second dimension for y
            
            # Size encoding
            size = 0.1 + 0.2 * row[2] if self.dim > 2 else 0.15
            
            # Shape encoding
            if self.dim > 3:
                shape_val = row[3]
                if shape_val < 0.33:
                    patch = Circle((x, y), size/2)
                elif shape_val < 0.66:
                    patch = Rectangle((x-size/2, y-size/2), 
                                   size, size)
                else:
                    patch = self._create_triangle(x, y, size)
            else:
                patch = Circle((x, y), size/2)
            
            patches.append(patch)
            
            # Color encoding
            color_val = row[4] if self.dim > 4 else 0.5
            colors.append(color_val)
            sizes.append(size)
        
        collection = PatchCollection(patches, cmap='viridis')
        collection.set_array(np.array(colors))
        collection.set_alpha(0.6)
        
        ax.add_collection(collection)
        plt.colorbar(collection)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig, ax
    
    def _create_triangle(self, x, y, size):
        vertices = np.array([
            [x, y + size/2],
            [x - size/2, y - size/2],
            [x + size/2, y - size/2]
        ])
        return plt.Polygon(vertices)

# Example usage
data = np.random.rand(20, 5)  # 5-dimensional data
encoder = CompositeEncoder(data)
encoder.encode_composite()
plt.show()
```

Slide 13: Additional Resources

1.  arXiv:2007.12604 - "Visual Encoding: A Framework for Information Visualization" [https://arxiv.org/abs/2007.12604](https://arxiv.org/abs/2007.12604)
2.  arXiv:1909.01315 - "Perceptual Models for Visual Data Encoding" [https://arxiv.org/abs/1909.01315](https://arxiv.org/abs/1909.01315)
3.  arXiv:2103.05448 - "Interactive Visual Analytics: From Data to Insight" [https://arxiv.org/abs/2103.05448](https://arxiv.org/abs/2103.05448)
4.  arXiv:1711.00467 - "Understanding Visual Encoding: Theory and Application" [https://arxiv.org/abs/1711.00467](https://arxiv.org/abs/1711.00467)
5.  arXiv:2105.09978 - "Modern Approaches to Data Visualization and Visual Analytics" [https://arxiv.org/abs/2105.09978](https://arxiv.org/abs/2105.09978)

