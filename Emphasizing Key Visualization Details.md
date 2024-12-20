## Emphasizing Key Visualization Details
Slide 1: Understanding Matplotlib's Zoom Indicator

The indicate\_inset\_zoom function in Matplotlib enables creation of magnified views of specific plot regions. This powerful visualization technique helps emphasize important details by creating a secondary axes that displays a zoomed portion of the main plot while maintaining context.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 1000)
y = np.sin(x) * np.exp(-x/3)

# Create main plot
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax.plot(x, y, 'b-')
ax.set_title('Original Plot')

# Define region of interest
x1, x2 = 1.5, 2.5  # x boundaries
y1, y2 = 0.4, 0.6  # y boundaries

# Create zoomed plot
ax2.plot(x, y, 'b-')
ax2.set_xlim(x1, x2)
ax2.set_ylim(y1, y2)
ax2.set_title('Zoomed Region')

# Add zoom indicator
from mpl_toolkits.axes_grid1.inset_locator import indicate_inset_zoom
indicate_inset_zoom(ax2, ax)

plt.tight_layout()
plt.show()
```

Slide 2: Advanced Zoom Customization

Enhancing zoom indicators requires careful consideration of visual elements. We can customize the appearance of both the indicator box and connecting lines to make the relationship between main and zoomed views more apparent through styling parameters.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Create sample data
x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(3*x) * np.exp(-x/2)

# Main figure setup
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x, y, 'k-', linewidth=1.5)

# Create inset axes
axins = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
axins.plot(x, y, 'k-', linewidth=1.5)

# Define zoom region
region = (1.5, 2.5, 0.2, 0.6)
axins.set_xlim(region[0], region[1])
axins.set_ylim(region[2], region[3])

# Customize zoom indicator
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", 
          ec="red", linestyle="--", alpha=0.8)

plt.show()
```

Slide 3: Multiple Zoom Regions

When dealing with complex data, highlighting multiple regions of interest simultaneously can provide comprehensive insights. This technique allows viewers to compare different features within the same visualization context effectively.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Generate complex data
x = np.linspace(0, 10, 1000)
y = np.sin(x) + 0.5*np.sin(3*x) + 0.2*np.random.randn(len(x))

# Create main plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, y, 'b-', label='Signal')

# First zoom region
axins1 = inset_axes(ax, width="30%", height="30%", 
                    bbox_to_anchor=(0.2, 0.8))
axins1.plot(x, y, 'b-')
x1, x2, y1, y2 = 1, 2, -0.5, 0.5
axins1.set_xlim(x1, x2)
axins1.set_ylim(y1, y2)
mark_inset(ax, axins1, loc1=1, loc2=2, fc="none", ec="r")

# Second zoom region
axins2 = inset_axes(ax, width="30%", height="30%", 
                    bbox_to_anchor=(0.8, 0.2))
axins2.plot(x, y, 'b-')
x3, x4, y3, y4 = 7, 8, 0, 1
axins2.set_xlim(x3, x4)
axins2.set_ylim(y3, y4)
mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="g")

plt.show()
```

Slide 4: Interactive Zoom Selection

Creating an interactive system for zoom selection enhances user exploration capabilities. This implementation allows users to dynamically select regions of interest using mouse clicks, providing a more engaging visualization experience.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class ZoomSelector:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.x = np.linspace(0, 10, 1000)
        self.y = np.sin(self.x) * np.exp(-self.x/3)
        
        # Plot original data
        self.line, = self.ax1.plot(self.x, self.y, 'b-')
        self.ax2.plot(self.x, self.y, 'b-')
        
        # Initialize selector
        self.rs = RectangleSelector(
            self.ax1, self.line_select_callback,
            useblit=True, button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels'
        )
        
    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        self.ax2.set_xlim(min(x1, x2), max(x1, x2))
        self.ax2.set_ylim(min(y1, y2), max(y1, y2))
        self.fig.canvas.draw_idle()

# Create interactive plot
zoom_selector = ZoomSelector()
plt.show()
```

Slide 5: Zoom Navigator for Time Series

When analyzing time series data, incorporating a zoom navigator provides an efficient way to explore temporal patterns. This implementation creates a secondary view that acts as both a zoom control and context provider for the main visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

class TimeSeriesZoomNavigator:
    def __init__(self, data, timestamps):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                                      height_ratios=[3, 1])
        self.data = data
        self.timestamps = timestamps
        
        # Plot data in both axes
        self.ax1.plot(timestamps, data, 'b-')
        self.ax2.plot(timestamps, data, 'b-')
        
        # Configure navigator
        self.span = SpanSelector(
            self.ax2, self.onselect, 'horizontal',
            useblit=True, props=dict(alpha=0.3, facecolor='red')
        )
        
        # Style configuration
        self.ax2.set_xlabel('Time')
        self.ax1.set_ylabel('Value')
        plt.tight_layout()
        
    def onselect(self, xmin, xmax):
        self.ax1.set_xlim(xmin, xmax)
        self.fig.canvas.draw_idle()

# Example usage
t = np.linspace(0, 100, 1000)
signal = np.sin(t/5) + np.sin(t/2) + 0.2*np.random.randn(len(t))

navigator = TimeSeriesZoomNavigator(signal, t)
plt.show()
```

Slide 6: Hierarchical Zoom Levels

Complex datasets often require multiple levels of detail examination. This implementation creates a hierarchical zoom system that allows users to progressively zoom into nested regions of interest with maintained context.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def create_hierarchical_zoom(x, y, zoom_levels):
    fig, ax_main = plt.subplots(figsize=(12, 8))
    ax_main.plot(x, y, 'b-')
    ax_main.set_title('Main View')
    
    axes = [ax_main]
    for level, (x_range, y_range) in enumerate(zoom_levels, 1):
        # Create inset with increasing zoom
        ax_inset = inset_axes(axes[-1], 
                            width="40%", height="40%",
                            bbox_to_anchor=(1.0, 0.5),
                            bbox_transform=axes[-1].transAxes,
                            loc='center right')
        
        ax_inset.plot(x, y, 'b-')
        ax_inset.set_xlim(x_range)
        ax_inset.set_ylim(y_range)
        ax_inset.set_title(f'Zoom Level {level}')
        
        # Connect views with arrows
        axes[-1].annotate('', 
                         xy=(x_range[0], y_range[0]),
                         xytext=(x_range[1], y_range[1]),
                         arrowprops=dict(arrowstyle='->'))
        axes.append(ax_inset)

# Example usage
x = np.linspace(0, 10, 1000)
y = np.sin(2*x) * np.exp(-x/5) + 0.1*np.random.randn(len(x))

zoom_levels = [
    ((2, 3), (-0.5, 0.5)),
    ((2.2, 2.4), (-0.2, 0.2)),
    ((2.25, 2.35), (-0.1, 0.1))
]

create_hierarchical_zoom(x, y, zoom_levels)
plt.tight_layout()
plt.show()
```

Slide 7: Dynamic Resolution Enhancement

When zooming into specific regions, we can dynamically increase the resolution of the data visualization. This technique ensures that zoomed regions maintain high detail levels while optimizing memory usage.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class DynamicResolutionZoom:
    def __init__(self, x, y, zoom_region):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.original_x = x
        self.original_y = y
        
        # Original plot
        self.ax1.plot(x, y, 'b.-', label='Original')
        self.ax1.set_title('Original Resolution')
        
        # Create high-resolution zoom
        x_zoom = np.linspace(zoom_region[0], zoom_region[1], 1000)
        interpolator = interp1d(x, y, kind='cubic')
        y_zoom = interpolator(x_zoom)
        
        # Zoomed plot with enhanced resolution
        self.ax2.plot(x_zoom, y_zoom, 'r-', label='Enhanced')
        self.ax2.set_xlim(zoom_region)
        self.ax2.set_title('Enhanced Resolution')
        
        # Add indicators
        self.ax1.axvspan(zoom_region[0], zoom_region[1], 
                        alpha=0.2, color='red')
        
        self.ax1.legend()
        self.ax2.legend()

# Example usage
x = np.linspace(0, 10, 50)  # Sparse original data
y = np.sin(x) + 0.1*np.random.randn(len(x))
zoom_region = (4, 5)

resolution_zoom = DynamicResolutionZoom(x, y, zoom_region)
plt.tight_layout()
plt.show()
```

Slide 8: Comparative Zoom Analysis

When analyzing multiple datasets, synchronized zoom regions enable effective comparison of corresponding features. This implementation creates linked zoom views for multiple data series.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class ComparativeZoomAnalyzer:
    def __init__(self, datasets):
        self.datasets = datasets
        n_datasets = len(datasets)
        
        # Create figure with two rows
        self.fig, self.axes = plt.subplots(2, n_datasets, 
                                          figsize=(4*n_datasets, 8))
        
        # Plot original data
        for i, (label, data) in enumerate(datasets.items()):
            self.axes[0, i].plot(data['x'], data['y'], '-')
            self.axes[0, i].set_title(f'{label} - Original')
            self.axes[1, i].plot(data['x'], data['y'], '-')
            self.axes[1, i].set_title(f'{label} - Zoomed')
        
        # Initialize selector on first plot
        self.selector = RectangleSelector(
            self.axes[0, 0], self.update_zoom,
            useblit=True, button=[1]
        )
    
    def update_zoom(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Update all zoomed views
        for ax in self.axes[1, :]:
            ax.set_xlim(min(x1, x2), max(x1, x2))
            ax.set_ylim(min(y1, y2), max(y1, y2))
        
        self.fig.canvas.draw_idle()

# Example usage
x = np.linspace(0, 10, 1000)
datasets = {
    'Signal A': {'x': x, 'y': np.sin(2*x) + 0.1*np.random.randn(len(x))},
    'Signal B': {'x': x, 'y': np.cos(3*x) + 0.1*np.random.randn(len(x))},
    'Signal C': {'x': x, 'y': np.sin(x)*np.exp(-x/5) + 0.1*np.random.randn(len(x))}
}

analyzer = ComparativeZoomAnalyzer(datasets)
plt.tight_layout()
plt.show()
```

Slide 9: Context-Preserving Zoom Transitions

Context preservation during zoom transitions helps maintain spatial awareness while examining details. This implementation creates smooth transitions between zoom levels while keeping surrounding context visible through transparency effects.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

class ContextPreservingZoom:
    def __init__(self, x, y):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.x, self.y = x, y
        
        # Plot main data
        self.main_line, = self.ax.plot(x, y, 'b-', alpha=0.6)
        
        # Initialize zoom box
        self.zoom_rect = Rectangle((0, 0), 0, 0, 
                                 facecolor='white', alpha=0.9,
                                 edgecolor='red', linewidth=2)
        self.ax.add_patch(self.zoom_rect)
        
        # Create zoom line
        self.zoom_line, = self.ax.plot([], [], 'r-', linewidth=2)
        
        # Animation setup
        self.zoom_frames = np.linspace(0, 1, 30)
        self.current_zoom = None
        
    def zoom_to_region(self, x_range, y_range):
        self.current_zoom = (x_range, y_range)
        
        def update(frame):
            # Update zoom rectangle
            width = x_range[1] - x_range[0]
            height = y_range[1] - y_range[0]
            self.zoom_rect.set_bounds(x_range[0], y_range[0], 
                                    width*frame, height*frame)
            
            # Update zoomed line
            mask = (self.x >= x_range[0]) & (self.x <= x_range[1])
            self.zoom_line.set_data(self.x[mask], self.y[mask])
            
            return self.zoom_rect, self.zoom_line
        
        self.anim = FuncAnimation(self.fig, update,
                                frames=self.zoom_frames,
                                interval=50, blit=True)

# Example usage
x = np.linspace(0, 10, 1000)
y = np.sin(2*x) * np.exp(-x/5) + 0.1*np.random.randn(len(x))

cpz = ContextPreservingZoom(x, y)
cpz.zoom_to_region([2, 3], [-0.5, 0.5])
plt.show()
```

Slide 10: Linked Coordinate Zoom Navigation

Implementing a linked coordinate system allows for synchronized navigation across multiple views of the same dataset. This approach is particularly useful when analyzing relationships between different aspects of the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, SpanSelector

class LinkedCoordinateZoom:
    def __init__(self, data_dict):
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 2)
        
        # Create four linked views
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Main view
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # X profile
        self.ax3 = self.fig.add_subplot(gs[1, 0])  # Y profile
        self.ax4 = self.fig.add_subplot(gs[1, 1])  # Zoom view
        
        self.data = data_dict
        self.setup_plots()
        
        # Initialize selectors
        self.rect_selector = RectangleSelector(
            self.ax1, self.on_rect_select,
            useblit=True, props=dict(facecolor='red', alpha=0.3)
        )
        
        self.x_selector = SpanSelector(
            self.ax2, self.on_xspan_select, 'horizontal',
            useblit=True, props=dict(facecolor='blue', alpha=0.3)
        )
        
        self.y_selector = SpanSelector(
            self.ax3, self.on_yspan_select, 'vertical',
            useblit=True, props=dict(facecolor='green', alpha=0.3)
        )
    
    def setup_plots(self):
        # Main scatter plot
        self.ax1.scatter(self.data['x'], self.data['y'], 
                        c=self.data['z'], cmap='viridis', alpha=0.6)
        
        # X profile
        self.ax2.plot(self.data['x'], np.mean(self.data['z'], axis=0))
        
        # Y profile
        self.ax3.plot(np.mean(self.data['z'], axis=1), self.data['y'])
        
        # Initialize zoom view
        self.scatter_zoom = self.ax4.scatter([], [], c=[], cmap='viridis')
    
    def on_rect_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.update_zoom_view(x1, x2, y1, y2)
    
    def on_xspan_select(self, xmin, xmax):
        ymin, ymax = self.ax1.get_ylim()
        self.update_zoom_view(xmin, xmax, ymin, ymax)
    
    def on_yspan_select(self, ymin, ymax):
        xmin, xmax = self.ax1.get_xlim()
        self.update_zoom_view(xmin, xmax, ymin, ymax)
    
    def update_zoom_view(self, xmin, xmax, ymin, ymax):
        mask = ((self.data['x'] >= xmin) & (self.data['x'] <= xmax) &
                (self.data['y'] >= ymin) & (self.data['y'] <= ymax))
        
        self.ax4.cla()
        self.ax4.scatter(self.data['x'][mask], self.data['y'][mask],
                        c=self.data['z'][mask], cmap='viridis')
        self.ax4.set_xlim(xmin, xmax)
        self.ax4.set_ylim(ymin, ymax)
        self.fig.canvas.draw_idle()

# Example usage
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

data = {
    'x': X.flatten(),
    'y': Y.flatten(),
    'z': Z.flatten()
}

lcz = LinkedCoordinateZoom(data)
plt.tight_layout()
plt.show()
```

Slide 11: Adaptive Zoom Resolution Control

This technique implements dynamic resolution adjustment based on zoom level, ensuring optimal performance and detail visibility. The system automatically increases sampling density in zoomed regions while maintaining lower resolution in the overview.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class AdaptiveZoomController:
    def __init__(self, x, y, z, base_resolution=50):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.original_data = {'x': x, 'y': y, 'z': z}
        self.base_resolution = base_resolution
        
        # Initialize both views
        self.plot_overview()
        self.current_zoom = None
        
        # Connect zoom event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def plot_overview(self):
        # Create base resolution grid
        xi = np.linspace(min(self.original_data['x']), 
                        max(self.original_data['x']), 
                        self.base_resolution)
        yi = np.linspace(min(self.original_data['y']), 
                        max(self.original_data['y']), 
                        self.base_resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate at base resolution
        zi = griddata((self.original_data['x'], self.original_data['y']), 
                     self.original_data['z'], (xi, yi))
        
        self.overview = self.ax1.pcolormesh(xi, yi, zi, shading='auto')
        self.ax1.set_title('Overview (Base Resolution)')
    
    def update_zoom(self, x_range, y_range, zoom_factor=4):
        # Calculate adaptive resolution based on zoom level
        zoom_resolution = int(self.base_resolution * zoom_factor)
        
        # Create high-resolution grid for zoomed region
        xi = np.linspace(x_range[0], x_range[1], zoom_resolution)
        yi = np.linspace(y_range[0], y_range[1], zoom_resolution)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate at higher resolution
        zi = griddata((self.original_data['x'], self.original_data['y']), 
                     self.original_data['z'], (xi, yi))
        
        self.ax2.clear()
        self.ax2.pcolormesh(xi, yi, zi, shading='auto')
        self.ax2.set_xlim(x_range)
        self.ax2.set_ylim(y_range)
        self.ax2.set_title(f'Zoomed View ({zoom_resolution}x{zoom_resolution})')
        
        # Add zoom indicator to overview
        if self.current_zoom:
            self.current_zoom.remove()
        self.current_zoom = self.ax1.add_patch(
            plt.Rectangle((x_range[0], y_range[0]),
                         x_range[1]-x_range[0],
                         y_range[1]-y_range[0],
                         fill=False, color='red')
        )
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        if event.inaxes == self.ax1:
            x, y = event.xdata, event.ydata
            zoom_width = (max(self.original_data['x']) - 
                         min(self.original_data['x'])) / 5
            zoom_height = (max(self.original_data['y']) - 
                         min(self.original_data['y'])) / 5
            
            x_range = [x - zoom_width/2, x + zoom_width/2]
            y_range = [y - zoom_height/2, y + zoom_height/2]
            self.update_zoom(x_range, y_range)

# Example usage
x = np.random.uniform(-5, 5, 1000)
y = np.random.uniform(-5, 5, 1000)
z = np.sin(np.sqrt(x**2 + y**2))

controller = AdaptiveZoomController(x, y, z)
plt.tight_layout()
plt.show()
```

Slide 12: Multi-Scale Feature Detection

This implementation combines zoom functionality with feature detection algorithms to automatically identify and highlight regions of interest across different scales of the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter

class MultiScaleFeatureDetector:
    def __init__(self, data, scales=[1, 2, 4, 8]):
        self.data = data
        self.scales = scales
        self.fig, self.axes = plt.subplots(2, len(scales), 
                                          figsize=(4*len(scales), 8))
        
        self.detect_features()
        self.plot_results()
    
    def detect_features(self):
        self.features = {}
        for scale in self.scales:
            # Smooth data at current scale
            smoothed = gaussian_filter(self.data, sigma=scale)
            
            # Detect local maxima
            peaks = signal.find_peaks_cwt(smoothed.ravel(), 
                                        np.array([scale]))
            
            # Calculate feature importance
            importance = smoothed.ravel()[peaks]
            
            self.features[scale] = {
                'smoothed': smoothed,
                'peaks': peaks,
                'importance': importance
            }
    
    def plot_results(self):
        for i, scale in enumerate(self.scales):
            # Plot smoothed data
            self.axes[0, i].imshow(self.features[scale]['smoothed'],
                                 cmap='viridis')
            self.axes[0, i].set_title(f'Scale {scale}')
            
            # Plot detected features
            peak_coords = np.unravel_index(
                self.features[scale]['peaks'],
                self.data.shape
            )
            
            self.axes[1, i].imshow(self.data, cmap='viridis', alpha=0.5)
            self.axes[1, i].scatter(
                peak_coords[1], peak_coords[0],
                c=self.features[scale]['importance'],
                cmap='hot', s=50*scale
            )
            self.axes[1, i].set_title(f'Features at Scale {scale}')
    
    def get_features_in_region(self, x_range, y_range, scale):
        features = self.features[scale]
        peak_coords = np.unravel_index(features['peaks'], 
                                     self.data.shape)
        
        mask = ((peak_coords[1] >= x_range[0]) & 
                (peak_coords[1] <= x_range[1]) &
                (peak_coords[0] >= y_range[0]) & 
                (peak_coords[0] <= y_range[1]))
        
        return {
            'x': peak_coords[1][mask],
            'y': peak_coords[0][mask],
            'importance': features['importance'][mask]
        }

# Example usage
size = 100
x = np.linspace(-5, 5, size)
y = np.linspace(-5, 5, size)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y) + 0.1*np.random.randn(size, size)

detector = MultiScaleFeatureDetector(Z)
plt.tight_layout()
plt.show()

# Get features in specific region
region_features = detector.get_features_in_region([25, 75], [25, 75], 4)
```

Slide 13: Magnification Lens Implementation

The magnification lens provides an interactive way to explore data by creating a movable magnified region that follows the cursor. This implementation includes smooth transitions and maintains context while examining details.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class MagnificationLens:
    def __init__(self, data, magnification=2.0):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.data = data
        self.mag = magnification
        
        # Display base image
        self.base_img = self.ax.imshow(self.data, cmap='viridis')
        
        # Initialize lens
        self.lens = Circle((0, 0), radius=30, 
                          fill=False, color='red', linewidth=2)
        self.ax.add_patch(self.lens)
        
        # Create magnified view
        self.mag_ax = self.fig.add_axes([0.7, 0.7, 0.2, 0.2])
        self.mag_img = self.mag_ax.imshow(self.data, cmap='viridis')
        self.mag_ax.set_xticks([])
        self.mag_ax.set_yticks([])
        
        # Connect events
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        
    def on_move(self, event):
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            
            # Update lens position
            self.lens.center = (x, y)
            
            # Calculate magnified region
            radius = int(self.lens.radius)
            x_slice = slice(max(0, y-radius), min(self.data.shape[0], y+radius))
            y_slice = slice(max(0, x-radius), min(self.data.shape[1], x+radius))
            
            # Update magnified view
            magnified = self.data[x_slice, y_slice]
            self.mag_img.set_data(magnified)
            self.mag_img.set_extent([y_slice.start, y_slice.stop,
                                   x_slice.stop, x_slice.start])
            
            self.fig.canvas.draw_idle()

# Example usage
size = 200
x = np.linspace(-4, 4, size)
y = np.linspace(-4, 4, size)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) * np.exp(-0.1*(X**2 + Y**2))

lens = MagnificationLens(Z)
plt.show()
```

Slide 14: Coordinated Multi-View Zoom System

This implementation creates a coordinated system of multiple views that respond to zoom actions simultaneously, enabling exploration of relationships between different data representations.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

class CoordinatedMultiView:
    def __init__(self, dataset):
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(2, 3)
        self.dataset = dataset
        
        # Create different views
        self.setup_views()
        self.setup_selectors()
        
    def setup_views(self):
        # Scatter plot view
        self.scatter_ax = self.fig.add_subplot(self.gs[0, 0])
        self.scatter = self.scatter_ax.scatter(
            self.dataset['x'], self.dataset['y'],
            c=self.dataset['color'], cmap='viridis'
        )
        self.scatter_ax.set_title('Scatter View')
        
        # Histogram view
        self.hist_ax = self.fig.add_subplot(self.gs[0, 1])
        self.hist_ax.hist(self.dataset['x'], bins=50)
        self.hist_ax.set_title('Distribution View')
        
        # Time series view
        self.time_ax = self.fig.add_subplot(self.gs[0, 2])
        self.time_ax.plot(self.dataset['time'], self.dataset['value'])
        self.time_ax.set_title('Time Series View')
        
        # Zoomed views
        self.zoom_scatter = self.fig.add_subplot(self.gs[1, 0])
        self.zoom_hist = self.fig.add_subplot(self.gs[1, 1])
        self.zoom_time = self.fig.add_subplot(self.gs[1, 2])
        
    def setup_selectors(self):
        self.scatter_selector = RectangleSelector(
            self.scatter_ax,
            self.on_scatter_select,
            useblit=True,
            props=dict(facecolor='red', alpha=0.3)
        )
        
    def on_scatter_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Update all zoomed views
        self.update_zoomed_views(
            [min(x1, x2), max(x1, x2)],
            [min(y1, y2), max(y1, y2)]
        )
        
    def update_zoomed_views(self, x_range, y_range):
        # Clear previous zoomed views
        self.zoom_scatter.clear()
        self.zoom_hist.clear()
        self.zoom_time.clear()
        
        # Create mask for selected region
        mask = ((self.dataset['x'] >= x_range[0]) & 
                (self.dataset['x'] <= x_range[1]) &
                (self.dataset['y'] >= y_range[0]) & 
                (self.dataset['y'] <= y_range[1]))
        
        # Update scatter zoom
        self.zoom_scatter.scatter(
            self.dataset['x'][mask],
            self.dataset['y'][mask],
            c=self.dataset['color'][mask],
            cmap='viridis'
        )
        self.zoom_scatter.set_title('Zoomed Scatter')
        
        # Update histogram zoom
        self.zoom_hist.hist(self.dataset['x'][mask], bins=30)
        self.zoom_hist.set_title('Zoomed Distribution')
        
        # Update time series zoom
        time_mask = ((self.dataset['time'] >= x_range[0]) &
                    (self.dataset['time'] <= x_range[1]))
        self.zoom_time.plot(
            self.dataset['time'][time_mask],
            self.dataset['value'][time_mask]
        )
        self.zoom_time.set_title('Zoomed Time Series')
        
        self.fig.canvas.draw_idle()

# Example usage
np.random.seed(42)
n_points = 1000
dataset = {
    'x': np.random.normal(0, 1, n_points),
    'y': np.random.normal(0, 1, n_points),
    'color': np.random.uniform(0, 1, n_points),
    'time': np.linspace(0, 10, n_points),
    'value': np.sin(np.linspace(0, 10, n_points)) + 
             0.1*np.random.randn(n_points)
}

cmv = CoordinatedMultiView(dataset)
plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

1.  "Interactive Data Visualization: Foundations, Techniques, and Applications" [https://arxiv.org/abs/2107.08239](https://arxiv.org/abs/2107.08239)
2.  "A Survey of Visual Analytics Techniques for Machine Learning" [https://arxiv.org/abs/1808.04926](https://arxiv.org/abs/1808.04926)
3.  "Visualization Techniques for Time Series Data Analysis" [https://arxiv.org/abs/2009.12816](https://arxiv.org/abs/2009.12816)
4.  "Multi-Scale Visualization Techniques for Scientific Data" [https://arxiv.org/abs/1907.11928](https://arxiv.org/abs/1907.11928)
5.  "Interactive Visual Analysis of High-Dimensional Data" [https://arxiv.org/abs/2011.07458](https://arxiv.org/abs/2011.07458)

