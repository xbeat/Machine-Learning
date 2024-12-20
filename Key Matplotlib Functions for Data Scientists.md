## Key Matplotlib Functions for Data Scientists
Slide 1: Introduction to Matplotlib

Matplotlib is a powerful data visualization library for Python. It provides a wide range of functions to create various types of plots and charts. This presentation will cover key Matplotlib functions that every data scientist should learn, along with practical examples and code snippets.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('A Simple Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

Slide 2: The plot() Function

The plot() function is used to create line plots. It's versatile and can be used to visualize trends over time or relationships between continuous variables. In this example, we'll plot the growth of a bacterial population over time.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data for bacterial growth
time = np.linspace(0, 24, 100)
population = 1000 * np.exp(0.2 * time)

plt.plot(time, population)
plt.title('Bacterial Growth Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Population')
plt.show()
```

Slide 3: The scatter() Function

The scatter() function is used to create scatter plots, which are excellent for visualizing relationships between two variables. Let's use it to explore the correlation between study time and exam scores.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random data for study time and exam scores
study_time = np.random.randn(50) * 2 + 10
exam_scores = 5 * study_time + np.random.randn(50) * 10 + 60

plt.scatter(study_time, exam_scores)
plt.title('Study Time vs. Exam Scores')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.show()
```

Slide 4: The hist() Function

The hist() function creates histograms, which are useful for visualizing the distribution of a dataset. Let's use it to analyze the distribution of heights in a population.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random height data
heights = np.random.normal(170, 10, 1000)

plt.hist(heights, bins=30, edgecolor='black')
plt.title('Distribution of Heights in Population')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()
```

Slide 5: The bar() Function

The bar() function creates bar charts, which are ideal for comparing quantities across different categories. Let's use it to visualize the popularity of different programming languages.

```python
import matplotlib.pyplot as plt

languages = ['Python', 'Java', 'JavaScript', 'C++', 'Ruby']
popularity = [68, 45, 63, 38, 22]

plt.bar(languages, popularity)
plt.title('Programming Language Popularity')
plt.xlabel('Programming Language')
plt.ylabel('Popularity Index')
plt.show()
```

Slide 6: The pie() Function

The pie() function creates pie charts, which are useful for showing proportions of a whole. Let's use it to display the distribution of different transportation modes in a city.

```python
import matplotlib.pyplot as plt

transport_modes = ['Car', 'Bus', 'Bicycle', 'Walking', 'Train']
percentages = [45, 20, 15, 12, 8]

plt.pie(percentages, labels=transport_modes, autopct='%1.1f%%')
plt.title('Transportation Mode Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
```

Slide 7: The subplots() Function

The subplots() function allows you to create multiple plots in a single figure. This is useful for comparing different datasets or visualizing various aspects of the same data. Let's create a figure with four different plots.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title('Sine Wave')

# Plot 2: Scatter plot
axs[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axs[0, 1].set_title('Random Scatter')

# Plot 3: Bar plot
axs[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
axs[1, 0].set_title('Bar Chart')

# Plot 4: Histogram
axs[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axs[1, 1].set_title('Normal Distribution')

plt.tight_layout()
plt.show()
```

Slide 8: The imshow() Function

The imshow() function is used to display images or 2D arrays as color-encoded images. It's particularly useful for visualizing matrices, heatmaps, or actual images. Let's create a simple heatmap using random data.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate a random 2D array
data = np.random.rand(10, 10)

plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title('Heatmap of Random Data')
plt.show()
```

Slide 9: The title() Function

The title() function adds a title to your plot, providing context and clarity. It's a simple yet crucial function for making your visualizations more informative. Let's create a plot with a descriptive title.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave Over One Complete Cycle', fontsize=16, fontweight='bold')
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')
plt.show()
```

Slide 10: The legend() Function

The legend() function adds a legend to your plot, which is essential when you have multiple datasets in a single plot. It helps viewers understand what each line or point series represents.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title('Sine and Cosine Waves')
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
```

Slide 11: The xlabel() and ylabel() Functions

The xlabel() and ylabel() functions are used to label the x and y axes of your plot, respectively. These labels provide crucial information about what the axes represent, making your plot more understandable.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data for a quadratic function
x = np.linspace(-10, 10, 100)
y = x**2

plt.plot(x, y)
plt.title('Quadratic Function')
plt.xlabel('X-axis: Input Values', fontsize=12)
plt.ylabel('Y-axis: Output Values (x^2)', fontsize=12)
plt.grid(True)
plt.show()
```

Slide 12: Customizing Plot Appearance

Matplotlib offers numerous options to customize the appearance of your plots. You can change colors, line styles, markers, and more. Let's create a plot with custom styling.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='blue', linestyle='--', linewidth=2, label='sin(x)')
plt.plot(x, y2, color='red', linestyle=':', linewidth=2, label='cos(x)')
plt.title('Customized Sin and Cos Waves', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='-', alpha=0.7)
plt.show()
```

Slide 13: Saving Plots

Matplotlib allows you to save your plots in various file formats. The savefig() function is used for this purpose. Let's create a plot and save it as both PNG and PDF files.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')

# Save as PNG
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Save as PDF
plt.savefig('sine_wave.pdf', bbox_inches='tight')

plt.show()
```

Slide 14: Real-Life Example: Weather Data Visualization

Let's create a more complex, real-life example by visualizing weather data. We'll plot temperature and precipitation data for a city over a year.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate mock weather data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperature = [5, 7, 10, 15, 20, 25, 28, 27, 22, 15, 10, 6]
precipitation = [50, 40, 45, 60, 70, 80, 85, 90, 80, 70, 60, 55]

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Months')
ax1.set_ylabel('Temperature (°C)', color=color)
ax1.plot(months, temperature, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Precipitation (mm)', color=color)
ax2.bar(months, precipitation, alpha=0.3, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Temperature and Precipitation Over a Year', fontsize=16)
fig.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in deepening their understanding of Matplotlib and data visualization in Python, here are some valuable resources:

1.  Matplotlib Official Documentation: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
2.  "Visualization with Matplotlib" by Nicolas P. Rougier (ArXiv): [https://arxiv.org/abs/1805.03383](https://arxiv.org/abs/1805.03383)
3.  "Ten Simple Rules for Better Figures" by Nicolas P. Rougier et al. (ArXiv): [https://arxiv.org/abs/1411.7396](https://arxiv.org/abs/1411.7396)

These resources provide in-depth explanations, advanced techniques, and best practices for creating effective visualizations using Matplotlib.

