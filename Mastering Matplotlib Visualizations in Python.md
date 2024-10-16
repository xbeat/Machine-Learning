## Mastering Matplotlib Visualizations in Python
Slide 1: Introduction to Matplotlib

Matplotlib is a powerful plotting library for Python, widely used for creating static, animated, and interactive visualizations. It provides a MATLAB-like interface and can produce publication-quality figures in various formats.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('A Simple Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

Slide 2: Basic Line Plot

Line plots are fundamental in data visualization. They show trends over a continuous interval and are great for displaying time series data.

```python
import matplotlib.pyplot as plt

years = [2015, 2016, 2017, 2018, 2019, 2020]
temperatures = [15.2, 15.5, 15.8, 16.1, 16.3, 16.5]

plt.figure(figsize=(10, 6))
plt.plot(years, temperatures, marker='o')
plt.title('Average Annual Temperatures')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
```

Slide 3: Customizing Plot Styles

Matplotlib offers various styles to customize the appearance of your plots. You can change colors, line styles, markers, and more.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.figure(figsize=(12, 6))
plt.plot(x, np.sin(x), 'r--', label='sin(x)')
plt.plot(x, np.cos(x), 'b-.', label='cos(x)')
plt.plot(x, -np.sin(x), 'g:', label='-sin(x)')
plt.legend()
plt.title('Trigonometric Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Slide 4: Scatter Plots

Scatter plots are useful for displaying the relationship between two variables. They can reveal patterns, correlations, or clusters in data.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.rand(50)
y = 2 * x + np.random.rand(50)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='purple', alpha=0.6, s=100)
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
```

Slide 5: Bar Charts

Bar charts are excellent for comparing quantities across different categories. They can be vertical or horizontal.

```python
import matplotlib.pyplot as plt

fruits = ['Apples', 'Oranges', 'Bananas', 'Pears', 'Grapes']
quantities = [30, 25, 40, 20, 35]

plt.figure(figsize=(10, 6))
plt.bar(fruits, quantities, color=['red', 'orange', 'yellow', 'green', 'purple'])
plt.title('Fruit Quantities')
plt.xlabel('Fruit')
plt.ylabel('Quantity')
plt.ylim(0, 50)
for i, v in enumerate(quantities):
    plt.text(i, v + 1, str(v), ha='center')
plt.show()
```

Slide 6: Histograms

Histograms display the distribution of a dataset. They're useful for understanding the underlying frequency distribution of a set of continuous data.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.normal(170, 10, 250)  # Generate 250 heights with mean 170 and std 10

plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, edgecolor='black')
plt.title('Distribution of Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

Slide 7: Subplots

Subplots allow you to create multiple plots in a single figure, which is useful for comparing different datasets or visualizing various aspects of the same data.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Different Plot Types in Subplots')

axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title('Sine Wave')

axs[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axs[0, 1].set_title('Scatter Plot')

axs[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
axs[1, 0].set_title('Bar Chart')

axs[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axs[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

Slide 8: 3D Plots

Matplotlib can create 3D plots, which are useful for visualizing three-dimensional data or surfaces.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
fig.colorbar(surf)
plt.show()
```

Slide 9: Pie Charts

Pie charts are used to show the composition of a whole, divided into parts. They're effective for displaying percentage or proportional data.

```python
import matplotlib.pyplot as plt

activities = ['Work', 'Sleep', 'Leisure', 'Eat', 'Commute']
hours = [8, 7, 5, 2, 2]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

plt.figure(figsize=(10, 8))
plt.pie(hours, labels=activities, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Daily Activities')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
```

Slide 10: Heatmaps

Heatmaps are useful for visualizing matrix data, showing patterns, correlations, or relative intensities.

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 10)
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(data, cmap='YlOrRd')
plt.colorbar(heatmap)

plt.title('Heatmap Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add value annotations
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()
```

Slide 11: Animation

Matplotlib can create animated plots, which are great for visualizing changing data over time or iterations.

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10))
    return line,

ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.title('Animated Sine Wave')
plt.show()
```

Slide 12: Customizing Ticks and Labels

Fine-tuning axis ticks and labels can greatly improve the readability and appearance of your plots.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

ax.set_title('Customized Sine Wave Plot')
ax.set_xlabel('Angle (radians)')
ax.set_ylabel('Sine value')

# Customize x-ticks
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])

# Customize y-ticks
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])

ax.grid(True)
plt.show()
```

Slide 13: Real-Life Example: Weather Data Visualization

Let's visualize monthly temperature data for a city, showcasing how to handle time series data and create informative plots.

```python
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate sample weather data
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
temps = np.random.normal(15, 10, 365) + 10 * np.sin(np.arange(365) * 2 * np.pi / 365)

# Calculate monthly averages
monthly_temps = [temps[i:i+30].mean() for i in range(0, 360, 30)]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Weather Data Visualization', fontsize=16)

# Daily temperature plot
ax1.plot(dates, temps)
ax1.set_title('Daily Temperatures')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C)')
ax1.grid(True)

# Monthly average bar plot
ax2.bar(months, monthly_temps)
ax2.set_title('Monthly Average Temperatures')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Temperature (°C)')
ax2.set_ylim(0, max(monthly_temps) + 5)

for i, temp in enumerate(monthly_temps):
    ax2.text(i, temp + 0.5, f'{temp:.1f}°C', ha='center')

plt.tight_layout()
plt.show()
```

Slide 14: Real-Life Example: Population Pyramid

A population pyramid is a graphical representation of the age and sex distribution of a population. Let's create one using Matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with real data for actual use)
ages = np.arange(0, 101, 10)
male_pop = [5, 7, 8, 10, 9, 8, 6, 4, 2, 1, 0.5]
female_pop = [5.2, 7.3, 8.4, 10.2, 9.4, 8.2, 6.3, 4.4, 2.3, 1.2, 0.6]

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(ages, male_pop, height=8, align='center', color='skyblue', label='Male')
ax.barh(ages, [-pop for pop in female_pop], height=8, align='center', color='pink', label='Female')

ax.set_xlabel('Population (%)')
ax.set_ylabel('Age Group')
ax.set_title('Population Pyramid Example')

ax.legend()

ax.set_xticks(np.arange(-10, 11, 2))
ax.set_xticklabels([str(abs(x)) for x in ax.get_xticks()])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0, 105, 'Male', ha='right', va='bottom')
ax.text(0, 105, 'Female', ha='left', va='bottom')

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of Matplotlib and its capabilities, consider the following resources:

1. Matplotlib Official Documentation: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
2. "Visualization with Matplotlib" by Jake VanderPlas (ArXiv:1412.3590): [https://arxiv.org/abs/1412.3590](https://arxiv.org/abs/1412.3590)
3. "Scientific Visualization: Python + Matplotlib" by Nicolas P. Rougier (ArXiv:1401.4127): [https://arxiv.org/abs/1401.4127](https://arxiv.org/abs/1401.4127)

These resources provide in-depth tutorials, examples, and advanced techniques for mastering Matplotlib.

