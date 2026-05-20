## Raincloud Plots Visualizing Data Distributions in Python
Slide 1: Introduction to Raincloud Plots

Raincloud plots are an innovative data visualization technique that combines aspects of box plots, violin plots, and scatter plots. They provide a comprehensive view of data distribution, central tendency, and individual data points, making them a powerful tool for exploratory data analysis and statistical reporting.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create the raincloud plot
f, ax = plt.subplots(figsize=(7, 5))
pt.RainCloud(data, ax=ax, orient="h", width_viol=.6, width_box=.25, 
             box_col="#D8BFD8", box_showfliers=False, 
             cloud_color="#D8BFD8", cloud_alpha=0.5)

plt.title("Basic Raincloud Plot")
plt.xlabel("Value")
plt.show()
```

Slide 2: Anatomy of a Raincloud Plot

A raincloud plot consists of three main components: the 'rain' (individual data points), the 'cloud' (probability density), and the 'box' (summary statistics). This combination allows for a detailed examination of data distribution while still providing an overview of key statistical measures.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot components
sns.stripplot(data=data, ax=ax, color="navy", alpha=0.4, jitter=True)
sns.kdeplot(data=data, ax=ax, color="skyblue", shade=True)
sns.boxplot(data=data, ax=ax, color="lightblue", width=0.15, boxprops=dict(alpha=0.7))

# Customize plot
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Anatomy of a Raincloud Plot")

# Add annotations
ax.annotate("Rain (Data Points)", xy=(2, 0.05), xytext=(2.5, 0.1),
            arrowprops=dict(arrowstyle="->"))
ax.annotate("Cloud (Density)", xy=(0, 0.3), xytext=(1, 0.35),
            arrowprops=dict(arrowstyle="->"))
ax.annotate("Box (Summary Statistics)", xy=(-0.5, -0.02), xytext=(-2, -0.1),
            arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.show()
```

Slide 3: Advantages of Raincloud Plots

Raincloud plots offer several advantages over traditional visualization methods. They provide a more complete picture of data distribution, reveal potential outliers or multimodality, and allow for easy comparison between groups. This makes them particularly useful for analyzing complex datasets in fields such as psychology, biology, and social sciences.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

# Generate sample data
np.random.seed(42)
group1 = np.random.normal(0, 1, 1000)
group2 = np.concatenate([np.random.normal(-1, 0.5, 600), np.random.normal(1, 0.5, 400)])

# Create figure and axes
f, axes = plt.subplots(2, 2, figsize=(12, 10))

# Traditional plots
sns.boxplot(data=[group1, group2], ax=axes[0, 0])
axes[0, 0].set_title("Box Plot")
sns.violinplot(data=[group1, group2], ax=axes[0, 1])
axes[0, 1].set_title("Violin Plot")
sns.stripplot(data=[group1, group2], ax=axes[1, 0], jitter=True)
axes[1, 0].set_title("Strip Plot")

# Raincloud plot
pt.RainCloud(data=[group1, group2], ax=axes[1, 1], orient="h", width_viol=.6, width_box=.25,
             box_col=["#1f77b4", "#ff7f0e"], box_showfliers=False,
             cloud_color=["#1f77b4", "#ff7f0e"], cloud_alpha=0.5)
axes[1, 1].set_title("Raincloud Plot")

plt.tight_layout()
plt.show()
```

Slide 4: Creating a Basic Raincloud Plot

Let's start by creating a basic raincloud plot using the `ptitprince` library, which provides a convenient interface for generating these plots. We'll use a simple dataset to demonstrate the process.

```python
import numpy as np
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create the raincloud plot
f, ax = plt.subplots(figsize=(8, 6))
pt.RainCloud(data, ax=ax, orient="h", width_viol=.6, width_box=.25,
             box_col="#D8BFD8", box_showfliers=False,
             cloud_color="#D8BFD8", cloud_alpha=0.5)

# Customize the plot
ax.set_title("Basic Raincloud Plot")
ax.set_xlabel("Value")
ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
```

Slide 5: Customizing Raincloud Plots

Raincloud plots can be customized to suit specific needs. We can adjust colors, orientations, and other visual elements to enhance readability and emphasize particular aspects of the data.

```python
import numpy as np
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Create the customized raincloud plot
f, ax = plt.subplots(figsize=(8, 6))
pt.RainCloud(data, ax=ax, orient="v", width_viol=.7, width_box=.3,
             box_col="#FFA07A", box_showfliers=True, # Show outliers
             cloud_color="#20B2AA", cloud_alpha=0.8,
             rain_point_size=3, rain_alpha=0.7)

# Customize the plot
ax.set_title("Customized Raincloud Plot", fontsize=16)
ax.set_xlabel("Density", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

# Add a grid for better readability
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

Slide 6: Comparing Multiple Groups

One of the strengths of raincloud plots is their ability to effectively compare multiple groups. Let's create a plot that compares three different distributions side by side.

```python
import numpy as np
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data for three groups
np.random.seed(42)
group1 = np.random.normal(0, 1, 1000)
group2 = np.random.normal(1, 1.5, 1000)
group3 = np.random.exponential(1, 1000)

# Combine data and create labels
data = [group1, group2, group3]
labels = ["Normal", "Wide Normal", "Exponential"]

# Create the raincloud plot
f, ax = plt.subplots(figsize=(12, 6))
pt.RainCloud(data, ax=ax, orient="h", width_viol=.6, width_box=.25,
             box_col=["#1f77b4", "#ff7f0e", "#2ca02c"],
             box_showfliers=False,
             cloud_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
             cloud_alpha=0.5,
             rain_point_size=3, rain_alpha=0.7,
             labels=labels)

# Customize the plot
ax.set_title("Comparison of Multiple Distributions", fontsize=16)
ax.set_xlabel("Value", fontsize=12)
ax.set_ylabel("Group", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 7: Handling Categorical Data

Raincloud plots can also be used to visualize categorical data. Let's create a plot that shows the distribution of heights for different tree species.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'Species': np.repeat(['Oak', 'Pine', 'Maple'], 100),
    'Height': np.concatenate([
        np.random.normal(20, 5, 100),  # Oak
        np.random.normal(30, 7, 100),  # Pine
        np.random.normal(25, 6, 100)   # Maple
    ])
})

# Create the raincloud plot
f, ax = plt.subplots(figsize=(10, 6))
pt.RainCloud(x='Species', y='Height', data=data, ax=ax, orient="v",
             width_viol=.6, width_box=.25,
             box_col=['#8B4513', '#228B22', '#D2691E'],
             box_showfliers=False,
             cloud_color=['#8B4513', '#228B22', '#D2691E'],
             cloud_alpha=0.5,
             rain_point_size=3, rain_alpha=0.7)

# Customize the plot
ax.set_title("Distribution of Tree Heights by Species", fontsize=16)
ax.set_xlabel("Species", fontsize=12)
ax.set_ylabel("Height (meters)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 8: Adding Statistical Information

We can enhance our raincloud plots by adding statistical information such as mean, median, or confidence intervals. This provides more context to the visualization and aids in data interpretation.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
from scipy import stats

# Generate sample data
np.random.seed(42)
data = pd.DataFrame({
    'Group': np.repeat(['A', 'B'], 100),
    'Value': np.concatenate([
        np.random.normal(10, 2, 100),  # Group A
        np.random.normal(12, 3, 100)   # Group B
    ])
})

# Calculate means and confidence intervals
means = data.groupby('Group')['Value'].mean()
ci = data.groupby('Group')['Value'].agg(lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x)))

# Create the raincloud plot
f, ax = plt.subplots(figsize=(10, 6))
pt.RainCloud(x='Group', y='Value', data=data, ax=ax, orient="h",
             width_viol=.6, width_box=.25,
             box_col=['#4169E1', '#FF6347'],
             box_showfliers=False,
             cloud_color=['#4169E1', '#FF6347'],
             cloud_alpha=0.5,
             rain_point_size=3, rain_alpha=0.7)

# Add means and confidence intervals
for i, group in enumerate(['A', 'B']):
    ax.plot([means[group], means[group]], [i-0.2, i+0.2], color='black', linewidth=2)
    ax.plot([ci[group][0][0], ci[group][0][1]], [i, i], color='black', linewidth=2)

# Customize the plot
ax.set_title("Raincloud Plot with Statistical Information", fontsize=16)
ax.set_ylabel("Group", fontsize=12)
ax.set_xlabel("Value", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 9: Raincloud Plots for Time Series Data

Raincloud plots can be adapted to visualize time series data, providing insights into how distributions change over time. Let's create a plot that shows temperature distributions for different months.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
data = pd.DataFrame({
    'Month': np.repeat(months, 100),
    'Temperature': np.concatenate([
        np.random.normal(0, 5, 100),   # January
        np.random.normal(2, 5, 100),   # February
        np.random.normal(8, 6, 100),   # March
        np.random.normal(14, 7, 100),  # April
        np.random.normal(20, 6, 100),  # May
        np.random.normal(25, 5, 100)   # June
    ])
})

# Create the raincloud plot
f, ax = plt.subplots(figsize=(12, 6))
pt.RainCloud(x='Month', y='Temperature', data=data, ax=ax, orient="v",
             width_viol=.6, width_box=.25,
             box_col=['#4169E1', '#6495ED', '#87CEEB', '#98FB98', '#FFFF00', '#FFA500'],
             box_showfliers=False,
             cloud_color=['#4169E1', '#6495ED', '#87CEEB', '#98FB98', '#FFFF00', '#FFA500'],
             cloud_alpha=0.5,
             rain_point_size=3, rain_alpha=0.7)

# Customize the plot
ax.set_title("Temperature Distribution by Month", fontsize=16)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 10: Combining Raincloud Plots with Other Visualizations

We can enhance our data presentation by combining raincloud plots with other types of visualizations. Here, we'll create a plot that includes both a raincloud plot and a line plot to show trends alongside distributions.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns

# Generate sample data
np.random.seed(42)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
data = pd.DataFrame({
    'Month': np.repeat(months, 100),
    'Temperature': np.concatenate([
        np.random.normal(0, 5, 100),   # January
        np.random.normal(2, 5, 100),   # February
        np.random.normal(8, 6, 100),   # March
        np.random.normal(14, 7, 100),  # April
        np.random.normal(20, 6, 100),  # May
        np.random.normal(25, 5, 100)   # June
    ])
})

# Calculate monthly averages
monthly_avg = data.groupby('Month')['Temperature'].mean()

# Create the plot
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Raincloud plot
pt.RainCloud(x='Month', y='Temperature', data=data, ax=ax1, orient="h",
             width_viol=.6, width_box=.25,
             box_col='skyblue', box_showfliers=False,
             cloud_color='skyblue', cloud_alpha=0.5,
             rain_point_size=3, rain_alpha=0.7)

# Line plot
sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax2, marker='o')

# Customize the plots
ax1.set_title("Temperature Distribution by Month", fontsize=16)
ax1.set_ylabel("Month", fontsize=12)
ax2.set_title("Average Temperature Trend", fontsize=16)
ax2.set_xlabel("Month", fontsize=12)
ax2.set_ylabel("Average Temperature (°C)", fontsize=12)

plt.tight_layout()
plt.show()
```

Slide 11: Real-Life Example - Student Performance

Let's use a raincloud plot to visualize student performance across different subjects. This example demonstrates how raincloud plots can be used in educational settings to compare distributions and identify patterns.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
subjects = ['Math', 'Science', 'Literature', 'History']
data = pd.DataFrame({
    'Subject': np.repeat(subjects, 100),
    'Score': np.concatenate([
        np.random.normal(75, 10, 100),  # Math
        np.random.normal(70, 12, 100),  # Science
        np.random.normal(80, 8, 100),   # Literature
        np.random.normal(72, 15, 100)   # History
    ])
})

# Create the raincloud plot
f, ax = plt.subplots(figsize=(12, 6))
pt.RainCloud(x='Subject', y='Score', data=data, ax=ax, orient="h",
             width_viol=.6, width_box=.25,
             box_col=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
             box_showfliers=False,
             cloud_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
             cloud_alpha=0.5,
             rain_point_size=3, rain_alpha=0.7)

# Customize the plot
ax.set_title("Student Performance Across Subjects", fontsize=16)
ax.set_xlabel("Score", fontsize=12)
ax.set_ylabel("Subject", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 12: Real-Life Example - Climate Data Analysis

In this example, we'll use raincloud plots to visualize and compare temperature distributions across different cities. This demonstrates how raincloud plots can be applied in climate science to analyze and compare complex datasets.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
cities = ['New York', 'London', 'Tokyo', 'Sydney']
data = pd.DataFrame({
    'City': np.repeat(cities, 1000),
    'Temperature': np.concatenate([
        np.random.normal(15, 10, 1000),  # New York
        np.random.normal(12, 7, 1000),   # London
        np.random.normal(18, 8, 1000),   # Tokyo
        np.random.normal(20, 6, 1000)    # Sydney
    ])
})

# Create the raincloud plot
f, ax = plt.subplots(figsize=(12, 6))
pt.RainCloud(x='City', y='Temperature', data=data, ax=ax, orient="v",
             width_viol=.6, width_box=.25,
             box_col=['#1E90FF', '#32CD32', '#FF69B4', '#FFA500'],
             box_showfliers=False,
             cloud_color=['#1E90FF', '#32CD32', '#FF69B4', '#FFA500'],
             cloud_alpha=0.5,
             rain_point_size=2, rain_alpha=0.5)

# Customize the plot
ax.set_title("Temperature Distribution Across Cities", fontsize=16)
ax.set_xlabel("City", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 13: Best Practices for Raincloud Plots

When creating raincloud plots, consider these best practices:

1. Use appropriate color schemes to distinguish between groups.
2. Adjust point size and transparency for large datasets to avoid overplotting.
3. Choose the orientation (horizontal or vertical) based on the number of groups and available space.
4. Include clear labels and titles to enhance interpretability.
5. Consider adding summary statistics or confidence intervals when appropriate.
6. Use consistent scales when comparing multiple raincloud plots.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt

# Generate sample data
np.random.seed(42)
groups = ['A', 'B', 'C']
data = pd.DataFrame({
    'Group': np.repeat(groups, 500),
    'Value': np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(2, 1.5, 500),
        np.random.normal(-1, 2, 500)
    ])
})

# Create the raincloud plot with best practices
f, ax = plt.subplots(figsize=(10, 6))
pt.RainCloud(x='Group', y='Value', data=data, ax=ax, orient="h",
             width_viol=.6, width_box=.25,
             box_col=['#3498db', '#e74c3c', '#2ecc71'],
             box_showfliers=False,
             cloud_color=['#3498db', '#e74c3c', '#2ecc71'],
             cloud_alpha=0.5,
             rain_point_size=2, rain_alpha=0.5)

# Customize the plot
ax.set_title("Best Practices Example", fontsize=16)
ax.set_xlabel("Value", fontsize=12)
ax.set_ylabel("Group", fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in learning more about raincloud plots and their applications, here are some valuable resources:

1. Allen M, et al. (2019). "Raincloud plots: a multi-platform tool for robust data visualization." Wellcome Open Research, 4:63. DOI: 10.12688/wellcomeopenres.15191.1
2. Micah Allen's GitHub repository: [https://github.com/RainCloudPlots/RainCloudPlots](https://github.com/RainCloudPlots/RainCloudPlots)
3. Tutorial on creating raincloud plots in Python: [https://www.python-graph-gallery.com/raincloud-plot](https://www.python-graph-gallery.com/raincloud-plot)
4. R package for raincloud plots: [https://github.com/jorvlan/raincloudplots](https://github.com/jorvlan/raincloudplots)
5. Allen M, et al. (2021). "Raincloud plots: a multi-platform tool for robust data visualization \[version 2; peer review: 2 approved\]." Wellcome Open Research, 4:63. DOI: 10.12688/wellcomeopenres.15191.2

These resources provide in-depth explanations, tutorials, and code examples to help you master the art of creating effective raincloud plots for your data visualization needs.

