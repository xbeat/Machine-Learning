## Visualizing Data Beyond Bar and Line Charts
Slide 1: Size-encoded Heatmap

Size-encoded heatmaps improve upon traditional heatmaps by adding a size component to each cell. This enhancement makes it easier to interpret exact values and reduces visual clutter, especially for values close to zero. The technique is particularly useful when dealing with datasets where both color and size can convey meaningful information.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
data = np.random.randn(10, 10)

# Create a size-encoded heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", 
            linewidths=0.5, ax=ax)

# Add size encoding
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        size = np.abs(data[i, j]) * 1000  # Scale size based on absolute value
        ax.scatter(j + 0.5, i + 0.5, s=size, color='none', 
                   edgecolor='black', linewidth=1.5)

plt.title("Size-encoded Heatmap")
plt.show()
```

Slide 2: Waterfall Chart

Waterfall charts effectively visualize incremental changes in a value over time or across categories. Unlike line plots, which show absolute values, waterfall charts display color-encoded rolling differences at each step. This makes it easier to understand the scale and direction of changes between data points.

```python
import plotly.graph_objects as go

# Sample data
categories = ['Start', 'Category A', 'Category B', 'Category C', 'Category D', 'End']
values = [10, 2, -3, 5, -1, 13]

# Calculate cumulative sum
cumulative = [0] * len(categories)
cumulative[0] = values[0]
for i in range(1, len(categories) - 1):
    cumulative[i] = cumulative[i-1] + values[i]
cumulative[-1] = cumulative[-2] + values[-1]

# Create waterfall chart
fig = go.Figure(go.Waterfall(
    name = "Waterfall", orientation = "v",
    measure = ["absolute"] + ["relative"] * (len(categories) - 2) + ["total"],
    x = categories,
    textposition = "outside",
    text = [f"{value:+}" for value in values],
    y = values,
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
))

fig.update_layout(
    title = "Waterfall Chart Example",
    showlegend = False
)

fig.show()
```

Slide 3: Raincloud Plot

Raincloud plots combine box plots, strip plots, and kernel density estimation (KDE) plots to provide a comprehensive view of data distribution. This visualization addresses limitations of box plots and histograms by showing individual data points, summary statistics, and probability distributions simultaneously.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(0)
data1 = np.random.normal(0, 1, 200)
data2 = np.random.normal(2, 1.5, 200)

# Create raincloud plot
fig, ax = plt.subplots(figsize=(10, 6))

# Function to create half violins
def half_violin(data, pos, side, width=0.3, ax=None):
    parts = ax.violinplot(data, positions=[pos], showmeans=False, 
                          showextrema=False, showmedians=False, vert=False)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        if side == 'left':
            pc.set_clip_path(plt.Rectangle((0, pos - width), 100, 2*width, 
                             transform=ax.transData))
        else:
            pc.set_clip_path(plt.Rectangle((-100, pos - width), 100, 2*width, 
                             transform=ax.transData))

# Plot the raincloud components
half_violin(data1, 1, 'right', ax=ax)
half_violin(data2, 2, 'right', ax=ax)
ax.boxplot([data1, data2], positions=[1, 2], vert=False, widths=0.15, 
           patch_artist=True, boxprops=dict(facecolor="white", color="black"), 
           medianprops=dict(color="red"))
ax.scatter(data1, np.random.normal(1, 0.04, len(data1)), alpha=0.5, s=5)
ax.scatter(data2, np.random.normal(2, 0.04, len(data2)), alpha=0.5, s=5)

ax.set_yticks([1, 2])
ax.set_yticklabels(['Data 1', 'Data 2'])
ax.set_xlabel('Value')
ax.set_title('Raincloud Plot Example')

plt.tight_layout()
plt.show()
```

Slide 4: Hexbin Plot

Hexbin plots are excellent for visualizing large datasets with thousands of points, where traditional scatter plots become too dense to interpret. By binning data points into hexagonal regions and assigning color intensity based on point density, hexbin plots provide a clear view of data distribution and concentration.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(0)
n = 100000
x = np.random.standard_normal(n)
y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)

# Create hexbin plot
fig, ax = plt.subplots(figsize=(10, 8))
hb = ax.hexbin(x, y, gridsize=50, cmap='YlOrRd')

ax.set_title("Hexbin Plot Example")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Add colorbar
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Count in bin')

plt.tight_layout()
plt.show()
```

Slide 5: Kernel Density Estimation (KDE) Plot

KDE plots offer a smooth representation of data distribution in two-dimensional space. By connecting points of equal density, these plots create contours that highlight areas of high and low data concentration. KDE plots are particularly useful for identifying patterns and clusters in complex datasets.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data
np.random.seed(0)
n = 1000
x = np.random.normal(0, 1, n)
y = x + np.random.normal(0, 1, n)

# Calculate the point density
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

# Create KDE plot
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(x, y, c=z, s=50, alpha=0.5, cmap='viridis')

ax.set_title("Kernel Density Estimation (KDE) Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Density')

plt.tight_layout()
plt.show()
```

Slide 6: Bump Chart

Bump charts are ideal for visualizing changes in rank over time across multiple categories. Similar to line plots, bump charts invert the y-axis labels because a smaller rank often indicates a better position. This makes it easy to track the relative performance of different entities over time.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Company A': [3, 2, 1, 1, 2],
    'Company B': [1, 1, 2, 3, 1],
    'Company C': [2, 3, 3, 2, 3],
    'Company D': [4, 4, 4, 4, 4]
}

df = pd.DataFrame(data)
df = df.set_index('Year')

# Create bump chart
fig, ax = plt.subplots(figsize=(12, 6))

for column in df.columns:
    ax.plot(df.index, df[column], marker='o', linewidth=2, markersize=8, label=column)

ax.set_yticks(range(1, len(df.columns) + 1))
ax.set_yticklabels(range(1, len(df.columns) + 1))
ax.invert_yaxis()

ax.set_xlabel('Year')
ax.set_ylabel('Rank')
ax.set_title('Bump Chart: Company Rankings Over Time')

ax.legend(title='Companies', loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

Slide 7: Dot Plot

Dot plots offer a clean alternative to bar plots, especially when dealing with many categories. By displaying only the endpoints as dots, dot plots reduce visual clutter while still effectively conveying the total value for each category. This makes them particularly useful for comparing multiple items across a single metric.

```python
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = {
    'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Value': [45, 30, 60, 22, 38, 51, 43, 57, 33, 40]
}

df = pd.DataFrame(data)

# Sort the dataframe by value
df = df.sort_values('Value', ascending=True)

# Create dot plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.hlines(y=df['Category'], xmin=0, xmax=df['Value'], color='skyblue', alpha=0.7, linewidth=2)
ax.plot(df['Value'], df['Category'], "o", markersize=10, color='blue', alpha=0.8)

# Add value labels
for i, v in enumerate(df['Value']):
    ax.text(v + 0.5, i, str(v), va='center')

ax.set_xlabel('Value')
ax.set_title('Dot Plot Example')
ax.set_xlim(0, max(df['Value']) * 1.1)

plt.tight_layout()
plt.show()
```

Slide 8: Bubble Chart

Bubble charts extend the concept of dot plots by encoding an additional dimension of data through bubble size. This makes them excellent for visualizing three variables simultaneously: two on the x and y axes, and a third represented by the size of each bubble. Bubble charts are particularly useful for comparing multiple entities across different metrics.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(0)
n = 20
x = np.random.rand(n)
y = np.random.rand(n)
sizes = np.random.rand(n) * 1000

# Create bubble chart
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(x, y, s=sizes, alpha=0.5, cmap='viridis')

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Bubble Chart Example')

# Add colorbar
plt.colorbar(scatter)

# Add size legend
sizes_legend = [100, 300, 500]
legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.5, label=str(size))
                   for size in sizes_legend]
ax.legend(handles=legend_elements, title="Bubble Sizes", loc="upper right")

plt.tight_layout()
plt.show()
```

Slide 9: Real-Life Example: Environmental Data Visualization

Let's apply some of these visualization techniques to environmental data. We'll use a size-encoded heatmap to display air quality index (AQI) data for different cities over a week.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data: AQI for 7 cities over 7 days
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
aqi_data = np.random.randint(30, 150, size=(7, 7))

# Create a size-encoded heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(aqi_data, annot=True, fmt="d", cmap="YlOrRd", 
            xticklabels=days, yticklabels=cities, ax=ax)

# Add size encoding
for i in range(aqi_data.shape[0]):
    for j in range(aqi_data.shape[1]):
        size = aqi_data[i, j] * 2  # Scale size based on AQI value
        ax.scatter(j + 0.5, i + 0.5, s=size, color='none', 
                   edgecolor='black', linewidth=1.5)

plt.title("Air Quality Index (AQI) by City and Day")
plt.tight_layout()
plt.show()
```

Slide 10: Real-Life Example: Sports Performance Analysis

In this example, we'll use a bump chart to visualize the performance of different teams in a sports league over multiple seasons.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data: Team rankings over 5 seasons
data = {
    'Season': [2018, 2019, 2020, 2021, 2022],
    'Team A': [3, 1, 2, 1, 3],
    'Team B': [1, 2, 1, 3, 1],
    'Team C': [2, 3, 4, 2, 2],
    'Team D': [4, 4, 3, 4, 4],
    'Team E': [5, 5, 5, 5, 5]
}

df = pd.DataFrame(data)
df = df.set_index('Season')

# Create bump chart
fig, ax = plt.subplots(figsize=(12, 8))

for column in df.columns:
    ax.plot(df.index, df[column], marker='o', linewidth=2, markersize=10, label=column)

ax.set_yticks(range(1, len(df.columns) + 1))
ax.set_yticklabels(range(1, len(df.columns) + 1))
ax.invert_yaxis()

ax.set_xlabel('Season')
ax.set_ylabel('Rank')
ax.set_title('Team Performance Over Seasons')

ax.legend(title='Teams', loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

Slide 11: Choosing the Right Visualization

Selecting the appropriate visualization technique depends on the nature of your data and the story you want to tell. Consider these factors when choosing a plot type:

1. Data dimensions: How many variables are you comparing?
2. Data type: Is your data categorical, continuous, or time-series?
3. Data distribution: Are you interested in showing outliers, trends, or patterns?
4. Audience: Who will be interpreting the visualization?
5. Purpose: Are you comparing values, showing relationships, or displaying trends over time?

Slide 12: Choosing the Right Visualization

For example, use size-encoded heatmaps for multidimensional data, waterfall charts for cumulative changes, and bump charts for ranking changes over time. Always prioritize clarity and ease of interpretation in your visualizations.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data_types = ['Categorical', 'Continuous', 'Time-series']
plot_types = ['Bar/Column', 'Scatter/Line', 'Line/Area']

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(data_types))
width = 0.25

for i, plot_type in enumerate(plot_types):
    ax.bar(x + i*width, [1]*len(data_types), width, label=plot_type, alpha=0.7)

ax.set_ylabel('Suitability')
ax.set_title('Plot Types for Different Data Types')
ax.set_xticks(x + width)
ax.set_xticklabels(data_types)
ax.legend()

plt.tight_layout()
plt.show()
```

Slide 13: Advanced Techniques: Combining Visualizations

Sometimes, combining multiple visualization techniques can provide deeper insights. For instance, we can create a hybrid chart that combines a dot plot with a box plot to show both individual data points and summary statistics.

Slide 14: Advanced Techniques: Combining Visualizations

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Generate sample data
np.random.seed(0)
categories = ['A', 'B', 'C', 'D', 'E']
data = [np.random.normal(loc, scale, 100) for loc, scale in 
        zip(range(3, 8), range(1, 6))]

# Create the hybrid chart
fig, ax = plt.subplots(figsize=(12, 6))

# Box plot
sns.boxplot(data=data, ax=ax, palette='pastel', width=0.3)

# Dot plot (jittered scatter plot)
for i, d in enumerate(data):
    y = np.random.normal(i, 0.04, len(d))
    ax.scatter(d, y, alpha=0.6, s=30, edgecolor='none')

ax.set_xticklabels(categories)
ax.set_ylabel('Value')
ax.set_title('Hybrid Chart: Box Plot + Dot Plot')

plt.tight_layout()
plt.show()
```

Slide 14: Interactive Visualizations

Interactive visualizations can greatly enhance data exploration and user engagement. While static plots are useful, interactive elements allow users to dive deeper into the data. Here's an example using Plotly to create an interactive scatter plot:

Slide 15: Interactive Visualizations

```python
import plotly.graph_objects as go
import numpy as np

# Generate sample data
np.random.seed(0)
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
sizes = np.random.randint(5, 50, n)

# Create interactive scatter plot
fig = go.Figure(data=go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=sizes,
        color=sizes,
        colorscale='Viridis',
        showscale=True
    ),
    text=[f'Point {i+1}<br>Size: {s}' for i, s in enumerate(sizes)],
    hoverinfo='text'
))

fig.update_layout(
    title='Interactive Scatter Plot',
    xaxis_title='X Axis',
    yaxis_title='Y Axis'
)

fig.show()
```

Slide 16: Additional Resources

For those interested in diving deeper into advanced data visualization techniques, here are some valuable resources:

1. "Fundamentals of Data Visualization" by Claus O. Wilke ArXiv: [https://arxiv.org/abs/2001.06451](https://arxiv.org/abs/2001.06451)
2. "Statistical Rethinking: A Bayesian Course with Examples in R and Stan" by Richard McElreath ArXiv: [https://arxiv.org/abs/2006.16949](https://arxiv.org/abs/2006.16949)
3. "Data Visualization: A Practical Introduction" by Kieran Healy

These resources provide in-depth discussions on choosing appropriate visualizations, design principles, and advanced techniques for effective data communication.

