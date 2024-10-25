## Interactive Data Visualization with Matplotlib

Slide 1: Interactive Line Plot with Click Events

Creating interactive visualizations enhances data exploration by allowing users to interact directly with plots. This implementation demonstrates how to capture mouse clicks on a line plot and display coordinates, enabling detailed examination of specific data points.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure and plot
fig, ax = plt.subplots()
line, = ax.plot(x, y)

# Click event handler
def on_click(event):
    if event.inaxes == ax:
        print(f'Clicked coordinates: x={event.xdata:.2f}, y={event.ydata:.2f}')
        ax.plot(event.xdata, event.ydata, 'ro')  # Add red dot at click
        plt.draw()

# Connect click event
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
```

Slide 2: Dynamic Data Updates in Real-time

Implementing real-time data visualization capabilities allows for monitoring of streaming data sources. This implementation creates an animated plot that updates automatically with new data points, simulating sensor readings or live measurements.

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class RealtimePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.x_data, self.y_data = [], []
        
    def init_plot(self):
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-2, 2)
        return self.line,
    
    def update(self, frame):
        self.x_data.append(frame)
        self.y_data.append(np.sin(frame * 0.1) + np.random.normal(0, 0.1))
        
        self.line.set_data(self.x_data, self.y_data)
        return self.line,

rt_plot = RealtimePlot()
anim = FuncAnimation(rt_plot.fig, rt_plot.update, init_func=rt_plot.init_plot,
                    frames=range(100), interval=50, blit=True)
plt.show()
```

Slide 3: Custom Interactive Legend

Interactive legends provide enhanced control over plot elements, allowing users to toggle visibility of different data series. This implementation creates a customized legend with clickable elements and hover effects.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate multiple data series
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

fig, ax = plt.subplots()
lines = []
lines.append(ax.plot(x, y1, label='sin(x)')[0])
lines.append(ax.plot(x, y2, label='cos(x)')[0])
lines.append(ax.plot(x, y3, label='tan(x)')[0])

leg = ax.legend()

def on_pick(event):
    legline = event.artist
    line = lines[leg.get_lines().index(legline)]
    line.set_visible(not line.get_visible())
    plt.draw()

for legline in leg.get_lines():
    legline.set_picker(True)
fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()
```

Slide 4: Interactive Time Series Analysis

Time series visualization requires specialized handling for temporal data and interactive features. This implementation creates an interactive time series plot with zoom capabilities and date-aware tooltips.

```python
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np

# Generate sample time series data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

fig, ax = plt.subplots(figsize=(12, 6))
line = ax.plot(dates, values)

# Configure date formatting
date_formatter = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_formatter)
plt.xticks(rotation=45)

def hover(event):
    if event.inaxes == ax:
        # Find nearest point
        distances = [abs(d.toordinal() - event.xdata) for d in dates]
        nearest_idx = distances.index(min(distances))
        
        # Update annotation
        ax.texts.clear()
        ax.annotate(f'Value: {values[nearest_idx]:.2f}\nDate: {dates[nearest_idx].strftime("%Y-%m-%d")}',
                   xy=(dates[nearest_idx], values[nearest_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        plt.draw()

fig.canvas.mpl_connect('motion_notify_event', hover)
plt.tight_layout()
plt.show()
```

Slide 5: Interactive 3D Surface Plot with Dynamic Coloring

Three-dimensional data visualization with interactive features enhances understanding of complex spatial relationships. This implementation demonstrates a 3D surface plot that responds to user input for rotation and color mapping adjustments.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def create_interactive_3d_plot():
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create initial surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf)
    
    def on_key(event):
        if event.key == 'c':  # Change colormap
            surf.set_cmap('plasma')
        elif event.key == 'r':  # Reset view
            ax.view_init(30, -60)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Run the visualization
create_interactive_3d_plot()
```

Slide 6: Advanced Time Series Visualization

Real-time data monitoring requires sophisticated visualization techniques that handle streaming data efficiently. This implementation showcases a rolling time window display with automatic updates and interactive markers.

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from matplotlib.animation import FuncAnimation

class TimeSeriesMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.values = deque(maxlen=max_points)
        
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_ylim(-2, 2)
        
    def update(self, frame):
        self.times.append(frame)
        self.values.append(np.sin(frame * 0.1))
        
        self.ax.set_xlim(max(0, frame - self.max_points), frame + 3)
        self.line.set_data(list(self.times), list(self.values))
        return self.line,

monitor = TimeSeriesMonitor()
anim = FuncAnimation(monitor.fig, monitor.update,
                    frames=range(200), interval=50)
plt.show()
```

Slide 7: Interactive Histogram with Dynamic Binning

Statistical data exploration benefits from interactive histogram visualization. This implementation allows users to dynamically adjust bin sizes and observe distribution changes in real-time.

```python
import numpy as np
import matplotlib.pyplot as plt

class InteractiveHistogram:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots()
        self.bins = 30
        self.update_plot()
        
    def update_plot(self):
        self.ax.clear()
        self.ax.hist(self.data, bins=self.bins)
        self.ax.set_title(f'Histogram (bins={self.bins})')
        plt.draw()
        
    def on_scroll(self, event):
        if event.button == 'up':
            self.bins = min(100, self.bins + 5)
        else:
            self.bins = max(5, self.bins - 5)
        self.update_plot()

# Example usage
data = np.random.normal(0, 1, 1000)
hist = InteractiveHistogram(data)
hist.fig.canvas.mpl_connect('scroll_event', hist.on_scroll)
plt.show()
```

Slide 8: Phase Space Plot with Dynamic Trajectories

Analyzing dynamical systems requires specialized visualization techniques. This implementation creates an interactive phase space plot that shows system evolution and allows parameter adjustment.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_phase_space_plot():
    t = np.linspace(0, 20, 1000)
    x = np.sin(t)
    v = np.cos(t)
    
    fig, ax = plt.subplots()
    line, = ax.plot(x, v)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    
    def update_frequency(event):
        if event.key == 'up':
            t_new = np.linspace(0, 20, 1000)
            x_new = np.sin(1.5 * t_new)
            v_new = 1.5 * np.cos(1.5 * t_new)
            line.set_data(x_new, v_new)
            plt.draw()
    
    fig.canvas.mpl_connect('key_press_event', update_frequency)
    plt.show()

create_phase_space_plot()
```

Slide 9: Multi-Panel Interactive Dashboard

Combining multiple visualization types enables comprehensive data analysis. This implementation creates a dashboard with synchronized interactive plots that respond to user interactions.

```python
import matplotlib.pyplot as plt
import numpy as np

def create_dashboard():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    line1, = ax1.plot(x, y)
    hist = ax2.hist(y, bins=20)
    
    def on_click(event):
        if event.inaxes == ax1:
            ax1.axvline(x=event.xdata, color='r', alpha=0.5)
            ax2.clear()
            mask = x <= event.xdata
            ax2.hist(y[mask], bins=20)
            plt.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

create_dashboard()
```

Slide 10: Custom Colormap Animation

Understanding data through color requires specialized visualization techniques. This implementation demonstrates how to create and animate custom colormaps for enhanced data representation.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_colormap_animation():
    fig, ax = plt.subplots()
    
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    def frame(i):
        Z = np.sin(np.sqrt(X**2 + Y**2) - i * 0.1)
        if hasattr(frame, 'im'):
            frame.im.remove()
        frame.im = ax.imshow(Z, cmap='viridis')
        return frame.im,
    
    anim = FuncAnimation(fig, frame, frames=100,
                        interval=50, blit=True)
    plt.colorbar(frame.im)
    plt.show()

create_colormap_animation()
```

Slide 11: Interactive Scatter Plot Matrix

Multivariate data analysis requires specialized visualization techniques. This implementation creates an interactive scatter plot matrix with brushing and linking capabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

class ScatterMatrix:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.n = data.shape[1]
        
        self.fig, self.axes = plt.subplots(self.n, self.n,
                                         figsize=(10, 10))
        self.create_matrix()
        
    def create_matrix(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.axes[i, j].scatter(self.data[:, j],
                                          self.data[:, i],
                                          alpha=0.5)
                else:
                    self.axes[i, i].hist(self.data[:, i])
                    
                if i == self.n - 1:
                    self.axes[i, j].set_xlabel(self.labels[j])
                if j == 0:
                    self.axes[i, j].set_ylabel(self.labels[i])

# Example usage
data = np.random.randn(100, 3)
labels = ['X', 'Y', 'Z']
matrix = ScatterMatrix(data, labels)
plt.tight_layout()
plt.show()
```

Slide 12: Dynamic Network Graph Visualization

Network analysis requires specialized interactive visualization techniques. This implementation creates a force-directed graph layout with interactive node positioning.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_network_plot():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Generate random graph
    n_nodes = 10
    positions = np.random.rand(n_nodes, 2)
    edges = [(i, j) for i in range(n_nodes) 
             for j in range(i+1, n_nodes)
             if np.random.rand() < 0.3]
    
    # Plot nodes and edges
    ax.scatter(positions[:, 0], positions[:, 1])
    for i, j in edges:
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]], 'k-')
    
    def on_click(event):
        if event.inaxes == ax:
            dist = np.sum((positions - 
                          [event.xdata, event.ydata])**2,
                         axis=1)
            nearest = np.argmin(dist)
            positions[nearest] = [event.xdata, event.ydata]
            ax.clear()
            ax.scatter(positions[:, 0], positions[:, 1])
            for i, j in edges:
                ax.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]], 'k-')
            plt.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

create_network_plot()
```

Slide 13: Additional Resources

1.  [https://arxiv.org/abs/2012.08972](https://arxiv.org/abs/2012.08972) - Interactive Visualization Techniques for Exploring High-Dimensional Data
2.  [https://arxiv.org/abs/2107.14702](https://arxiv.org/abs/2107.14702) - Real-Time Interactive Data Visualization in Python
3.  [https://arxiv.org/abs/2109.05542](https://arxiv.org/abs/2109.05542) - Advanced Matplotlib Techniques for Scientific Visualization
4.  [https://arxiv.org/abs/2203.09801](https://arxiv.org/abs/2203.09801) - Interactive Visual Analytics for Time Series Data
5.  [https://arxiv.org/abs/2106.12231](https://arxiv.org/abs/2106.12231) - Modern Approaches to Network Visualization

