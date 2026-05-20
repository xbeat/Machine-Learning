## Enhancing Data Visualizations with Seaborn in Python
Slide 1: Introduction to Seaborn

Seaborn is a powerful Python library for creating statistical data visualizations. Built on top of Matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics. Seaborn is particularly useful for exploring and understanding data through various plot types.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load a sample dataset
tips = sns.load_dataset("tips")

# Create a simple scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Relationship between Total Bill and Tip")
plt.show()
```

Slide 2: Setting up Seaborn

Before diving into Seaborn's features, it's essential to set up your environment. Seaborn can be installed using pip, and it's often used alongside Pandas for data manipulation.

```python
# Install Seaborn (run this in your terminal or command prompt)
# pip install seaborn

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set the default Seaborn style
sns.set_theme()

# Load a built-in dataset
df = sns.load_dataset("penguins")
print(df.head())
```

Slide 3: Customizing Plot Aesthetics

Seaborn offers various built-in themes and color palettes to enhance the appearance of your plots. You can easily customize the overall look of your visualizations.

```python
# Set a specific style
sns.set_style("whitegrid")

# Create a plot with a custom color palette
sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", palette="deep")
plt.title("Penguin Bill Dimensions by Species")
plt.show()

# Reset to default style
sns.set_style("darkgrid")
```

Slide 4: Distribution Plots

Seaborn excels at visualizing distributions. The distplot function (now replaced by displot) allows you to create histograms with kernel density estimation.

```python
# Create a distribution plot
sns.displot(df, x="flipper_length_mm", kde=True, hue="species")
plt.title("Distribution of Flipper Lengths")
plt.show()
```

Slide 5: Categorical Plots

Categorical plots in Seaborn help visualize the distribution of a quantitative variable across different categories.

```python
# Create a box plot
sns.boxplot(data=df, x="species", y="body_mass_g")
plt.title("Body Mass Distribution by Penguin Species")
plt.show()

# Create a violin plot
sns.violinplot(data=df, x="species", y="body_mass_g")
plt.title("Body Mass Distribution (Violin Plot)")
plt.show()
```

Slide 6: Regression Plots

Seaborn's regression plots are useful for visualizing the relationship between two variables and fitting a regression model.

```python
# Create a regression plot
sns.regplot(data=df, x="flipper_length_mm", y="body_mass_g")
plt.title("Relationship between Flipper Length and Body Mass")
plt.show()
```

Slide 7: Pair Plots

Pair plots are an excellent way to visualize relationships between multiple variables in a dataset.

```python
# Create a pair plot
sns.pairplot(df, hue="species")
plt.suptitle("Pair Plot of Penguin Measurements", y=1.02)
plt.show()
```

Slide 8: Heatmaps

Heatmaps are useful for visualizing the correlation between variables in a dataset.

```python
# Create a correlation matrix
corr_matrix = df.corr()

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Penguin Measurements")
plt.show()
```

Slide 9: Facet Grids

Facet grids allow you to create multiple plots for different subsets of your data.

```python
# Create a facet grid
g = sns.FacetGrid(df, col="species", height=4, aspect=1.2)
g.map(sns.scatterplot, "bill_length_mm", "bill_depth_mm")
g.add_legend()
plt.suptitle("Bill Dimensions by Species", y=1.05)
plt.show()
```

Slide 10: Real-Life Example: Environmental Data Analysis

Let's analyze air quality data using Seaborn to visualize pollution levels across different cities.

```python
# Create sample air quality data
air_quality = pd.DataFrame({
    'city': ['New York', 'London', 'Tokyo', 'Beijing', 'Mumbai'] * 12,
    'month': list(range(1, 13)) * 5,
    'aqi': [50, 45, 40, 80, 70, 55, 48, 42, 85, 75, 60, 52,
            40, 35, 30, 75, 65, 45, 38, 32, 78, 68, 50, 42,
            35, 30, 25, 70, 60, 40, 33, 27, 73, 63, 45, 37,
            85, 80, 75, 120, 110, 90, 83, 77, 125, 115, 95, 87,
            75, 70, 65, 110, 100, 80, 73, 67, 115, 105, 85, 77]
})

# Create a line plot to show AQI trends
sns.lineplot(data=air_quality, x='month', y='aqi', hue='city')
plt.title('Air Quality Index (AQI) Trends Across Cities')
plt.xlabel('Month')
plt.ylabel('AQI')
plt.show()
```

Slide 11: Real-Life Example: Scientific Data Visualization

Visualizing scientific data is crucial for understanding complex phenomena. Let's use Seaborn to analyze the relationship between a planet's mass and its orbital period.

```python
import numpy as np

# Generate sample exoplanet data
np.random.seed(42)
n_planets = 100
planet_data = pd.DataFrame({
    'mass': np.random.uniform(0.1, 10, n_planets),  # Earth masses
    'orbital_period': np.random.uniform(1, 1000, n_planets),  # Earth days
    'star_type': np.random.choice(['G', 'K', 'M'], n_planets)
})

# Create a scatter plot with logarithmic scales
sns.scatterplot(data=planet_data, x='mass', y='orbital_period', hue='star_type', alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.title('Exoplanet Mass vs. Orbital Period')
plt.xlabel('Planet Mass (Earth masses)')
plt.ylabel('Orbital Period (Earth days)')
plt.show()
```

Slide 12: Advanced Customization

Seaborn allows for advanced customization of plots, including combining multiple plot types and adjusting various parameters.

```python
# Create a complex plot combining multiple Seaborn features
g = sns.JointGrid(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species")
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.kdeplot)
g.add_legend()
plt.suptitle("Bill Length vs. Depth with Marginal Distributions", y=1.02)
plt.tight_layout()
plt.show()
```

Slide 13: Seaborn with Time Series Data

Seaborn can be used effectively with time series data, providing insights into trends and patterns over time.

```python
# Generate sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(len(dates))) + 100
})

# Create a time series plot
sns.lineplot(data=ts_data, x='date', y='value')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 14: Combining Seaborn with Matplotlib

While Seaborn provides high-level plotting functions, you can still use Matplotlib for fine-grained control over your visualizations.

```python
# Create a Seaborn plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species", ax=ax)

# Add Matplotlib customizations
ax.set_title("Penguin Flipper Length vs. Body Mass", fontsize=16)
ax.set_xlabel("Flipper Length (mm)", fontsize=12)
ax.set_ylabel("Body Mass (g)", fontsize=12)
ax.legend(title="Species", title_fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For further exploration of Seaborn and data visualization techniques, consider the following resources:

1. Seaborn Official Documentation: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
2. "Data Visualization: A Practical Introduction" by Kieran Healy
3. ArXiv paper: "Visualization of High-Dimensional Data using t-SNE" by L.J.P. van der Maaten and G.E. Hinton ([https://arxiv.org/abs/1307.1662](https://arxiv.org/abs/1307.1662))
4. Kaggle's Data Visualization Tutorials
5. PyData conference talks and workshops (available on YouTube)

These resources provide in-depth explanations, advanced techniques, and real-world applications of data visualization using Seaborn and other Python libraries.

