## Comparing Sankey Diagrams and Bar Plots in Python
Slide 1: 

Introduction to Sankey Diagrams

Sankey diagrams are a type of flow diagram where the width of the arrows is proportional to the quantity of flow. They are often used to visualize the flow of energy, materials, or costs in a process. Sankey diagrams can help identify inefficiencies and bottlenecks in a system.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define the flows and their values
flows = [('A', 'B', 5), ('A', 'C', 3), ('B', 'D', 4), ('C', 'D', 2)]

# Create the Sankey diagram
fig = plt.figure()
sankey = plt.subplot(1, 1, 1, xticks=[], yticks=[], title="Simple Sankey Diagram")
sankey.set_box_aspect(1)  # To make the diagram square
sankey.axis('off')

# Add the flows to the diagram
sankey.add(flows=flows, fc='k', alpha=0.5, orientations=[0, 1, 0, -1])

# Draw the diagram
plt.show()
```

Slide 2: 

Grouped Bar Plots

Grouped bar plots are a type of chart that displays multiple sets of data using rectangular bars. Each set of data is represented by a different color or pattern, and the bars are grouped together to facilitate comparison across categories.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = ['A', 'B', 'C', 'D']
data1 = [10, 20, 15, 25]
data2 = [8, 18, 12, 22]

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.arange(len(labels))
width = 0.35

# Create the bars
bar1 = ax.bar(x - width/2, data1, width, label='Data 1')
bar2 = ax.bar(x + width/2, data2, width, label='Data 2')

# Add labels and titles
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title('Grouped Bar Plot')

# Display the plot
plt.show()
```

Slide 3: 

Comparing Sankey Diagrams and Grouped Bar Plots

Sankey diagrams are better suited for visualizing flows or transfers between different stages or components of a system, while grouped bar plots are more appropriate for comparing values across different categories or groups.

```python
# No code for this slide
```

Slide 4: 

Advantages of Sankey Diagrams

Sankey diagrams provide a clear visual representation of the flow of quantities between different stages or components. They can help identify bottlenecks, inefficiencies, and potential areas for optimization in a system.

```python
import matplotlib.pyplot as plt

# Sample data
flows = [('A', 'B', 10), ('A', 'C', 5), ('B', 'D', 8), ('C', 'D', 2)]

# Create the Sankey diagram
fig = plt.figure()
sankey = plt.subplot(1, 1, 1, xticks=[], yticks=[], title="Advantages of Sankey Diagrams")
sankey.set_box_aspect(1)
sankey.axis('off')

# Add the flows to the diagram
sankey.add(flows=flows, fc='k', alpha=0.5, orientations=[0, 1, 0, -1])

# Draw the diagram
plt.show()
```

Slide 5: 

Limitations of Sankey Diagrams

Sankey diagrams can become complex and difficult to interpret when there are many flows or stages involved. They may not be suitable for visualizing time-series data or data with a large number of categories.

```python
# Pseudocode for a complex Sankey diagram
# Define flows and their values
# Create a figure and subplot
# Set the aspect ratio and remove axis ticks
# Add the flows to the diagram with appropriate orientations
# Adjust the layout and spacing for better readability
# Draw the diagram
```

Slide 6: 

Advantages of Grouped Bar Plots

Grouped bar plots are effective for comparing values across different categories or groups. They are easy to read and interpret, and can be used to visualize both numerical and categorical data.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = ['A', 'B', 'C', 'D']
data1 = [10, 20, 15, 25]
data2 = [8, 18, 12, 22]
data3 = [12, 16, 18, 20]

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.arange(len(labels))
width = 0.2

# Create the bars
bar1 = ax.bar(x - width, data1, width, label='Data 1')
bar2 = ax.bar(x, data2, width, label='Data 2')
bar3 = ax.bar(x + width, data3, width, label='Data 3')

# Add labels and titles
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title('Grouped Bar Plot')

# Display the plot
plt.show()
```

Slide 7: 

Limitations of Grouped Bar Plots

Grouped bar plots can become cluttered and difficult to read when there are many groups or categories. They may not be suitable for visualizing flows or transfers between different stages or components.

```python
# Pseudocode for a cluttered grouped bar plot
# Define labels and data for multiple groups
# Set up the figure and axis
# Calculate the bar width based on the number of groups
# Create the bars for each group
# Add labels, titles, and legend
# Display the plot
```

Slide 8: 

Customizing Sankey Diagrams

Sankey diagrams can be customized in various ways to improve their visual appeal and readability. This can include adjusting the colors, line styles, and orientations of the flows, as well as adding labels and annotations.

```python
import matplotlib.pyplot as plt

# Sample data
flows = [('A', 'B', 10), ('A', 'C', 5), ('B', 'D', 8), ('C', 'D', 2)]
labels = ['A', 'B', 'C', 'D']

# Create the Sankey diagram
fig = plt.figure()
sankey = plt.subplot(1, 1, 1, xticks=[], yticks=[], title="Customized Sankey Diagram")
sankey.set_box_aspect(1)
sankey.axis('off')

# Customize the diagram
sankey.add(flows=flows, fc='#008000', alpha=0.5, orientations=[0, 1, 0, -1])
sankey.set_offset(0.2, 0.1)  # Adjust the diagram position
sankey.set_fontsize(12)  # Adjust the font size

# Add labels
for label, pos in enumerate(sankey.get_offsets()):
    sankey.annotate(labels[label], pos)

# Draw the diagram
plt.show()
```

Slide 9: 

Customizing Grouped Bar Plots

Grouped bar plots can be customized with different colors, hatching patterns, and labels to improve their visual appeal and clarity. Additionally, error bars can be added to represent uncertainty or variability in the data.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
labels = ['A', 'B', 'C', 'D']
data1 = [10, 20, 15, 25]
data2 = [8, 18, 12, 22]
errors = [1, 2, 1.5, 1]

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.arange(len(labels))
width = 0.35

# Create the bars
bar1 = ax.bar(x - width/2, data1, width, label='Data 1', color='C0', hatch='//')
bar2 = ax.bar(x + width/2, data2, width, label='Data 2', color='C1', edgecolor='black')

# Add error bars
plt.errorbar(x - width/2, data1, yerr=errors, fmt='o', color='black', capsize=5, capthick=2)
plt.errorbar(x + width/2, data2, yerr=errors, fmt='o', color='black', capsize=5, capthick=2)

# Add labels and titles
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title('Customized Grouped Bar Plot')

# Display the plot
plt.show()
```

Slide 10: 

Combining Sankey Diagrams and Grouped Bar Plots

In some cases, it may be beneficial to combine Sankey diagrams and grouped bar plots in a single visualization. This can provide a comprehensive view of the system, showing both the flows between components and the comparison of values across categories.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data for Sankey diagram
flows = [('A', 'B', 10), ('A', 'C', 5), ('B', 'D', 8), ('C', 'D', 2)]

# Sample data for grouped bar plot
labels = ['A', 'B', 'C', 'D']
data1 = [15, 10, 20, 25]
data2 = [12, 8, 18, 22]

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

# Create the Sankey diagram
sankey = ax1.subplot(xticks=[], yticks=[], title="Sankey Diagram")
sankey.set_box_aspect(1)
sankey.axis('off')
sankey.add(flows=flows, fc='k', alpha=0.5, orientations=[0, 1, 0, -1])

# Create the grouped bar plot
x = np.arange(len(labels))
width = 0.35
bar1 = ax2.bar(x - width/2, data1, width, label='Data 1')
bar2 = ax2.bar(x + width/2, data2, width, label='Data 2')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.set_title('Grouped Bar Plot')

plt.subplots_adjust(hspace=0.5)  # Adjust the spacing between subplots

# Display the plot
plt.show()
```

Slide 11: 

Interactive Visualizations

Both Sankey diagrams and grouped bar plots can be made interactive using libraries like Plotly or Bokeh. Interactive visualizations allow users to explore the data by hovering over elements, zooming, and filtering, enhancing the overall user experience.

```python
import plotly.graph_objects as go

# Sample data for Sankey diagram
flows = [('A', 'B', 10), ('A', 'C', 5), ('B', 'D', 8), ('C', 'D', 2)]
labels = ['A', 'B', 'C', 'D']

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="blue"
    ),
    link=dict(
        source=[0, 0, 1, 2],
        target=[1, 2, 3, 3],
        value=[10, 5, 8, 2]
    ))])

# Adjust the layout
fig.update_layout(title_text="Interactive Sankey Diagram", font_size=10)

# Display the interactive visualization
fig.show()
```

Slide 12: 

Choosing the Right Visualization

The choice between using Sankey diagrams or grouped bar plots depends on the nature of the data and the specific goals of the visualization. Sankey diagrams are better suited for visualizing flows or transfers, while grouped bar plots are more appropriate for comparing values across different categories or groups.


Slide 13: 

Additional Resources

For more information and examples on Sankey diagrams and grouped bar plots, you can refer to the following resources:

* ArXiv: [link](https://arxiv.org/abs/1807.06365) - "Visualizing Flows with Sankey Diagrams"
* ArXiv: [link](https://arxiv.org/abs/1712.03708) - "Grouped Bar Charts: A Bridge Between Simple and Complex Visualization"

