## Stem-Leaf Plots in Python A Visual Representation of Data
Slide 1:

Introduction to Stem-Leaf Plots

A stem-leaf plot is a statistical method for organizing and displaying numerical data. It provides a visual representation of the distribution of data, making it easier to identify patterns, clusters, and outliers. This plot is particularly useful for smaller datasets and can help in understanding the shape and spread of the data.

```python
import matplotlib.pyplot as plt

def stem_leaf_plot(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Extract stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Create a dictionary to store leaves for each stem
    stem_leaf_dict = {}
    for stem, leaf in zip(stems, leaves):
        if stem in stem_leaf_dict:
            stem_leaf_dict[stem].append(leaf)
        else:
            stem_leaf_dict[stem] = [leaf]
    
    # Plot the stem-leaf plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for stem, leaves in stem_leaf_dict.items():
        ax.text(0, int(stem), f"{stem} | {''.join(leaves)}")
    
    ax.set_ylim(min(map(int, stems))-1, max(map(int, stems))+1)
    ax.set_axis_off()
    plt.title("Stem-Leaf Plot")
    plt.show()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
stem_leaf_plot(data)
```

Slide 2:

Components of a Stem-Leaf Plot

A stem-leaf plot consists of two main components: stems and leaves. The stem represents the leading digit(s) of each data point, while the leaf represents the trailing digit. This structure allows for a compact representation of data while preserving individual values.

```python
def create_stem_leaf(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Create a dictionary to store leaves for each stem
    stem_leaf_dict = {}
    for stem, leaf in zip(stems, leaves):
        if stem in stem_leaf_dict:
            stem_leaf_dict[stem].append(leaf)
        else:
            stem_leaf_dict[stem] = [leaf]
    
    # Print the stem-leaf plot
    for stem, leaves in stem_leaf_dict.items():
        print(f"{stem} | {''.join(leaves)}")

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
create_stem_leaf(data)
```

Slide 3:

Creating a Basic Stem-Leaf Plot

To create a stem-leaf plot, we first sort the data in ascending order. Then, we separate each number into its stem (leading digit(s)) and leaf (trailing digit). Finally, we arrange the leaves next to their corresponding stems.

```python
def basic_stem_leaf(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Print the stem-leaf plot
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(leaves[i], end="")
        print()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
basic_stem_leaf(data)
```

Slide 4:

Interpreting a Stem-Leaf Plot

Stem-leaf plots provide valuable insights into data distribution. They allow us to quickly identify the median, mode, and range of the dataset. Additionally, we can easily spot clusters, gaps, and outliers in the data.

```python
import numpy as np

def interpret_stem_leaf(data):
    # Calculate basic statistics
    median = np.median(data)
    mode = max(set(data), key=data.count)
    data_range = max(data) - min(data)
    
    print(f"Median: {median}")
    print(f"Mode: {mode}")
    print(f"Range: {data_range}")
    
    # Create and print the stem-leaf plot
    basic_stem_leaf(data)

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 20, 28]
interpret_stem_leaf(data)
```

Slide 5:

Back-to-Back Stem-Leaf Plots

Back-to-back stem-leaf plots are useful for comparing two datasets. They display two stem-leaf plots facing each other, sharing a common stem. This arrangement allows for easy comparison of the distribution and spread of two datasets.

```python
def back_to_back_stem_leaf(data1, data2):
    # Combine and sort both datasets
    all_data = sorted(data1 + data2)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in all_data]
    leaves1 = [str(x)[-1] if x in data1 else '' for x in all_data]
    leaves2 = [str(x)[-1] if x in data2 else '' for x in all_data]
    
    # Print the back-to-back stem-leaf plot
    unique_stems = sorted(set(stems), reverse=True)
    for stem in unique_stems:
        left_leaves = ''.join(leaves1[i] for i, s in enumerate(stems) if s == stem)
        right_leaves = ''.join(leaves2[i] for i, s in enumerate(stems) if s == stem)
        print(f"{left_leaves:>10} | {stem} | {right_leaves}")

# Example data
data1 = [23, 29, 20, 32, 27, 25, 28]
data2 = [31, 24, 26, 22, 30, 33, 35]
back_to_back_stem_leaf(data1, data2)
```

Slide 6:

Handling Large Datasets

For larger datasets, we may need to adjust our stem-leaf plot to maintain readability. One approach is to use larger stem intervals or to split stems when there are too many leaves for a single stem.

```python
def large_dataset_stem_leaf(data, stem_interval=10):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x // stem_interval) for x in sorted_data]
    leaves = [str(x % stem_interval) for x in sorted_data]
    
    # Print the stem-leaf plot
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{int(stem) * stem_interval:3d} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(f"{leaves[i]:2s}", end=" ")
        print()

# Example data (larger dataset)
import random
data = [random.randint(0, 199) for _ in range(50)]
large_dataset_stem_leaf(data)
```

Slide 7:

Stem-Leaf Plot with Outlier Detection

Stem-leaf plots can be enhanced to highlight outliers in the dataset. We can use statistical methods like the Interquartile Range (IQR) to identify and mark potential outliers.

```python
import numpy as np

def stem_leaf_with_outliers(data):
    # Calculate Q1, Q3, and IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Print the stem-leaf plot with outlier detection
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                if lower_bound <= sorted_data[i] <= upper_bound:
                    print(leaves[i], end="")
                else:
                    print(f"*{leaves[i]}*", end="")  # Mark outliers with asterisks
        print()

# Example data with outliers
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 15, 40]
stem_leaf_with_outliers(data)
```

Slide 8:

Colored Stem-Leaf Plot

We can enhance the visual appeal and interpretability of stem-leaf plots by adding color. This can be particularly useful for highlighting different ranges or categories within the data.

```python
import matplotlib.pyplot as plt
import numpy as np

def colored_stem_leaf(data):
    # Sort the data
    sorted_data = sorted(data)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    # Create a dictionary to store leaves for each stem
    stem_leaf_dict = {}
    for stem, leaf in zip(stems, leaves):
        if stem in stem_leaf_dict:
            stem_leaf_dict[stem].append(leaf)
        else:
            stem_leaf_dict[stem] = [leaf]
    
    # Plot the colored stem-leaf plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(stem_leaf_dict)))
    
    for i, (stem, leaves) in enumerate(stem_leaf_dict.items()):
        ax.text(0, int(stem), f"{stem} | {''.join(leaves)}", color=colors[i])
    
    ax.set_ylim(min(map(int, stems))-1, max(map(int, stems))+1)
    ax.set_axis_off()
    plt.title("Colored Stem-Leaf Plot")
    plt.show()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
colored_stem_leaf(data)
```

Slide 9:

Real-Life Example: Student Heights

Let's create a stem-leaf plot to visualize the distribution of heights (in centimeters) for a group of students in a classroom.

```python
def student_heights_stem_leaf(heights):
    # Sort the heights
    sorted_heights = sorted(heights)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_heights]
    leaves = [str(x)[-1] for x in sorted_heights]
    
    # Print the stem-leaf plot
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(leaves[i], end="")
        print()

# Example student heights (in cm)
heights = [158, 162, 165, 170, 171, 173, 175, 176, 178, 180, 182, 185]
print("Student Heights (cm) Stem-Leaf Plot:")
student_heights_stem_leaf(heights)
```

Slide 10:

Real-Life Example: Daily Temperature Readings

Let's create a stem-leaf plot to visualize daily temperature readings (in Celsius) for a month.

```python
import random

def temperature_stem_leaf(temperatures):
    # Sort the temperatures
    sorted_temps = sorted(temperatures)
    
    # Create stems and leaves
    stems = [str(x)[:-1] for x in sorted_temps]
    leaves = [str(x)[-1] for x in sorted_temps]
    
    # Print the stem-leaf plot
    unique_stems = sorted(set(stems))
    for stem in unique_stems:
        print(f"{stem} | ", end="")
        for i, s in enumerate(stems):
            if s == stem:
                print(leaves[i], end="")
        print()

# Generate random temperature data for a month (in Celsius)
temperatures = [random.randint(15, 35) for _ in range(30)]
print("Daily Temperature Readings (°C) Stem-Leaf Plot:")
temperature_stem_leaf(temperatures)
```

Slide 11:

Stem-Leaf Plot with Summary Statistics

We can enhance our stem-leaf plot by including summary statistics such as mean, median, and mode. This provides a more comprehensive view of the data distribution.

```python
import numpy as np

def stem_leaf_with_stats(data):
    # Calculate summary statistics
    mean = np.mean(data)
    median = np.median(data)
    mode = max(set(data), key=data.count)
    
    # Create and print the stem-leaf plot
    basic_stem_leaf(data)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median}")
    print(f"Mode: {mode}")

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 20, 28]
stem_leaf_with_stats(data)
```

Slide 12:

Interactive Stem-Leaf Plot

Creating an interactive stem-leaf plot allows users to explore the data more dynamically. Here's a simple example using ipywidgets for Jupyter notebooks.

```python
import ipywidgets as widgets
from IPython.display import display

def interactive_stem_leaf(data):
    def update_plot(stem_width):
        stems = [str(x // stem_width) for x in sorted(data)]
        leaves = [str(x % stem_width) for x in sorted(data)]
        
        unique_stems = sorted(set(stems))
        output = []
        for stem in unique_stems:
            stem_line = f"{int(stem) * stem_width:3d} | "
            stem_line += " ".join(leaves[i] for i, s in enumerate(stems) if s == stem)
            output.append(stem_line)
        
        plot_output.value = "\n".join(output)
    
    stem_width_slider = widgets.IntSlider(min=1, max=20, step=1, value=10, description="Stem Width:")
    plot_output = widgets.Textarea(layout={'width': '400px', 'height': '300px'})
    
    widgets.interactive(update_plot, stem_width=stem_width_slider)
    display(stem_width_slider, plot_output)

# Example data
data = [random.randint(0, 199) for _ in range(50)]
interactive_stem_leaf(data)
```

Slide 13:

Stem-Leaf Plot vs. Histogram

While stem-leaf plots and histograms are both used to visualize data distribution, they have different strengths. Let's compare them using the same dataset.

```python
import matplotlib.pyplot as plt
import numpy as np

def compare_stem_leaf_histogram(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Stem-leaf plot
    sorted_data = sorted(data)
    stems = [str(x)[:-1] for x in sorted_data]
    leaves = [str(x)[-1] for x in sorted_data]
    
    for stem in set(stems):
        ax1.text(0, int(stem), f"{stem} | {''.join([l for s, l in zip(stems, leaves) if s == stem])}")
    
    ax1.set_ylim(min(map(int, stems))-1, max(map(int, stems))+1)
    ax1.set_title("Stem-Leaf Plot")
    ax1.set_axis_off()
    
    # Histogram
    ax2.hist(data, bins='auto', edgecolor='black')
    ax2.set_title("Histogram")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

# Example data
data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30, 20, 28]
compare_stem_leaf_histogram(data)
```

Slide 14:

Limitations and Considerations

While stem-leaf plots are useful, they have limitations. They work best for small to medium-sized datasets and can become unwieldy for large datasets or those with a wide range of values. In such cases, other visualization methods like histograms or box plots might be more appropriate.

```python
def stem_leaf_limitations(data):
    if len(data) > 100:
        print("Warning: Large dataset. Consider using a histogram instead.")
        return
    
    range_of_data = max(data) - min(data)
    if range_of_data > 100:
        print("Warning: Wide range of values. Consider using larger stem intervals.")
    
    # Proceed with stem-leaf plot
    basic_stem_leaf(data)

# Example data
small_data = [23, 29, 20, 32, 27, 25, 28, 31, 24, 26, 22, 30]
large_data = list(range(1, 200))

print("Small dataset:")
stem_leaf_limitations(small_data)

print("\nLarge dataset:")
stem_leaf_limitations(large_data)
```

Slide 15:

Additional Resources

For those interested in diving deeper into stem-leaf plots and data visualization techniques, here are some valuable resources:

1. "Visualizing Data Patterns with Micromaps" by Daniel B. Carr and Linda Williams Pickle (2010) ArXiv reference: [https://arxiv.org/abs/1005.1937](https://arxiv.org/abs/1005.1937)
2. "Statistical Graphics Procedures by Example: Effective Graphs Using SAS" by Sanjay Matange and Dan Heath (2011) While not on ArXiv, this book provides comprehensive examples of various statistical graphics, including stem-leaf plots.
3. "The Grammar of Graphics" by Leland Wilkinson (2005) This seminal work discusses the principles behind effective data visualization techniques.

Remember to verify these resources and their availability, as they may have been updated or moved since the last knowledge update.

