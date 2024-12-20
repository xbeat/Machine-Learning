## Data Visualization with Python Empowering Analysis
Slide 1: Data Visualization with Python

Data visualization is a powerful tool for understanding complex datasets and extracting meaningful insights. Python offers a rich ecosystem of libraries for creating sophisticated and interactive visualizations. This presentation will explore common types of visualizations and their importance in data analysis using Python.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate sample data
data = np.random.randn(1000)

# Create a histogram
plt.figure(figsize=(10, 6))
sns.histplot(data, kde=True)
plt.title('Sample Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Bar Charts

Bar charts are ideal for comparing categorical data. They can display frequencies, proportions, or averages. Let's create a bar chart using Matplotlib to show the average temperature of different cities.

```python
import matplotlib.pyplot as plt

cities = ['New York', 'London', 'Tokyo', 'Sydney']
temperatures = [20, 15, 25, 22]

plt.figure(figsize=(10, 6))
plt.bar(cities, temperatures)
plt.title('Average Temperature by City')
plt.xlabel('City')
plt.ylabel('Temperature (°C)')
plt.show()
```

Slide 3: Line Charts

Line charts are suitable for visualizing trends over time or across continuous variables. They can help identify patterns, correlations, and anomalies. Let's create a line chart showing the population growth of a city over time.

```python
import matplotlib.pyplot as plt

years = list(range(2000, 2021))
population = [1000000 + i * 50000 for i in range(len(years))]

plt.figure(figsize=(10, 6))
plt.plot(years, population, marker='o')
plt.title('City Population Growth')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.show()
```

Slide 4: Scatter Plots

Scatter plots are effective for showing relationships between two numerical variables. They can reveal correlations, clusters, and outliers. Let's create a scatter plot to visualize the relationship between study time and exam scores.

```python
import matplotlib.pyplot as plt
import numpy as np

study_time = np.random.uniform(1, 10, 50)
exam_scores = 60 + 3 * study_time + np.random.normal(0, 5, 50)

plt.figure(figsize=(10, 6))
plt.scatter(study_time, exam_scores)
plt.title('Study Time vs. Exam Scores')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.show()
```

Slide 5: Histograms

Histograms are useful for understanding the distribution of a single numerical variable. They can help identify skewness, modality, and outliers. Let's create a histogram of ages in a sample population.

```python
import matplotlib.pyplot as plt
import numpy as np

ages = np.random.normal(35, 10, 1000)

plt.figure(figsize=(10, 6))
plt.hist(ages, bins=20, edgecolor='black')
plt.title('Age Distribution in Sample Population')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

Slide 6: Box Plots

Box plots show the distribution of a numerical variable across different categories. They can compare medians, quartiles, and outliers. Let's create a box plot to compare the distribution of heights across different sports.

```python
import matplotlib.pyplot as plt
import numpy as np

sports = ['Basketball', 'Soccer', 'Tennis', 'Swimming']
heights = [
    np.random.normal(195, 10, 100),
    np.random.normal(180, 8, 100),
    np.random.normal(185, 7, 100),
    np.random.normal(188, 9, 100)
]

plt.figure(figsize=(10, 6))
plt.boxplot(heights, labels=sports)
plt.title('Height Distribution by Sport')
plt.xlabel('Sport')
plt.ylabel('Height (cm)')
plt.show()
```

Slide 7: Heatmaps

Heatmaps represent data as a colored grid, often used to visualize matrices or correlations. They can identify patterns, clusters, and outliers in multi-dimensional data. Let's create a heatmap to visualize the correlation between different variables in a dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Generate sample data
data = np.random.randn(100, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

# Calculate correlation matrix
corr = df.corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()
```

Slide 8: Treemaps

Treemaps display hierarchical data as nested rectangles, representing different levels of aggregation. They can visualize part-to-whole relationships and proportions. Let's create a treemap to show the distribution of global land area by continent.

```python
import matplotlib.pyplot as plt
import squarify

continents = ['Asia', 'Africa', 'North America', 'South America', 'Antarctica', 'Europe', 'Australia']
areas = [44579000, 30370000, 24709000, 17840000, 14200000, 10180000, 8600000]

plt.figure(figsize=(12, 8))
squarify.plot(sizes=areas, label=continents, alpha=0.8)
plt.title('Global Land Area Distribution by Continent')
plt.axis('off')
plt.show()
```

Slide 9: Geographic Maps

Geographic maps visualize data on a geographic map, often used for spatial analysis. They can show locations, densities, and distributions. Let's create a simple map using Folium to display the locations of major cities.

```python
import folium

# Create a map centered on the world
world_map = folium.Map(location=[0, 0], zoom_start=2)

# Add markers for major cities
cities = [
    ('New York', 40.7128, -74.0060),
    ('London', 51.5074, -0.1278),
    ('Tokyo', 35.6762, 139.6503),
    ('Sydney', -33.8688, 151.2093)
]

for city, lat, lon in cities:
    folium.Marker([lat, lon], popup=city).add_to(world_map)

# Save the map
world_map.save('world_cities_map.html')

print("Map saved as 'world_cities_map.html'")
```

Slide 10: Real-Life Example: Air Quality Analysis

Let's analyze air quality data using various visualization techniques. We'll use a dataset containing information about air pollutants in different cities.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (you would typically load this from a file)
data = {
    'City': ['A', 'B', 'C', 'D', 'E'] * 4,
    'Pollutant': ['PM2.5', 'PM10', 'NO2', 'SO2'] * 5,
    'Concentration': [25, 40, 35, 10, 15, 60, 80, 55, 20, 30, 30, 45, 40, 15, 25, 5, 10, 8, 3, 7]
}
df = pd.DataFrame(data)

# Create a grouped bar chart
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Concentration', hue='Pollutant', data=df)
plt.title('Air Pollutant Concentrations by City')
plt.ylabel('Concentration (μg/m³)')
plt.legend(title='Pollutant')
plt.show()

# Create a heatmap
pivot_df = df.pivot(index='City', columns='Pollutant', values='Concentration')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, annot=True, cmap='YlOrRd')
plt.title('Air Pollutant Concentrations Heatmap')
plt.show()
```

Slide 11: Real-Life Example: Climate Change Visualization

Let's visualize global temperature anomalies over time to illustrate climate change trends.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create sample data (you would typically load this from a file)
years = list(range(1900, 2021))
temp_anomalies = [0.1 * (year - 1900) + np.random.normal(0, 0.2) for year in years]

df = pd.DataFrame({'Year': years, 'Temperature Anomaly': temp_anomalies})

# Create a line plot with a trend line
plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['Temperature Anomaly'], label='Annual anomaly')
plt.plot(df['Year'], df['Temperature Anomaly'].rolling(window=10).mean(), 
         color='red', label='10-year moving average')

plt.title('Global Temperature Anomalies (1900-2020)')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Create a histogram of temperature anomalies
plt.figure(figsize=(10, 6))
plt.hist(df['Temperature Anomaly'], bins=20, edgecolor='black')
plt.title('Distribution of Temperature Anomalies')
plt.xlabel('Temperature Anomaly (°C)')
plt.ylabel('Frequency')
plt.show()
```

Slide 12: Interactive Visualizations

Python libraries like Plotly allow for the creation of interactive visualizations. Let's create an interactive scatter plot of life expectancy vs. GDP per capita.

```python
import plotly.express as px
import numpy as np

# Generate sample data
countries = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
gdp_per_capita = np.random.uniform(1000, 50000, 10)
life_expectancy = 50 + 0.0006 * gdp_per_capita + np.random.normal(0, 2, 10)
population = np.random.uniform(1e6, 1e9, 10)

# Create interactive scatter plot
fig = px.scatter(
    x=gdp_per_capita,
    y=life_expectancy,
    size=population,
    color=countries,
    hover_name=countries,
    log_x=True,
    size_max=60,
    labels={'x': 'GDP per capita', 'y': 'Life Expectancy'}
)

fig.update_layout(title='Life Expectancy vs. GDP per capita')
fig.show()

print("Interactive plot created. Open in a web browser to interact.")
```

Slide 13: Importance of Visualization in Data Analysis

Data visualization plays a crucial role in data analysis. It helps identify patterns and trends that might be difficult to detect through numerical analysis alone. Visualizations are powerful tools for communicating complex ideas to both technical and non-technical audiences. They allow for interactive exploration of data, enabling users to manipulate and examine different aspects of the dataset. By providing clear and concise representations of data, visualizations support informed decision-making processes.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a plot demonstrating pattern recognition
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(x, y1, label='sin(x)')
ax1.plot(x, y2, label='cos(x)')
ax1.set_title('Pattern Recognition in Trigonometric Functions')
ax1.legend()

# Create a plot demonstrating outlier detection
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
outliers = np.random.uniform(-5, 5, 20)
all_data = np.concatenate([data, outliers])

ax2.boxplot(all_data)
ax2.set_title('Outlier Detection using Box Plot')

plt.tight_layout()
plt.show()
```

Slide 14: Best Practices in Data Visualization

When creating visualizations, it's important to follow best practices to ensure clarity and effectiveness. Choose appropriate chart types for your data and analysis goals. Use color effectively to highlight important information without overwhelming the viewer. Ensure your visualizations are accessible by using color-blind friendly palettes and providing alternative text descriptions. Label axes and data points clearly, and include titles and legends where necessary. Avoid cluttering your visualizations with unnecessary elements that may distract from the main message.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a color-blind friendly palette
sns.set_palette("colorblind")

# Generate sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

# Create a simple bar chart with best practices
plt.figure(figsize=(10, 6))
sns.barplot(x=categories, y=values)
plt.title('Sample Data Visualization')
plt.xlabel('Categories')
plt.ylabel('Values')

# Add value labels on top of each bar
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha='center', va='bottom')

# Remove top and right spines
sns.despine()

plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into data visualization with Python, here are some valuable resources:

1.  "Fundamentals of Data Visualization" by Claus O. Wilke ArXiv: [https://arxiv.org/abs/1901.08879](https://arxiv.org/abs/1901.08879)
2.  "Interactive Data Visualization in Python with Bokeh" by Bryan Van de Ven ArXiv: [https://arxiv.org/abs/1602.00103](https://arxiv.org/abs/1602.00103)
3.  "Matplotlib for Python Developers" by Sandro Tosi ArXiv: [https://arxiv.org/abs/1711.06385](https://arxiv.org/abs/1711.06385)

These resources provide in-depth explanations and advanced techniques for creating effective and impactful data visualizations using Python.

