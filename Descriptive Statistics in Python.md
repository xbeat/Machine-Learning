## Descriptive Statistics in Python

Slide 1: Introduction to Descriptive Statistics

Descriptive statistics are used to summarize and describe the main features of a dataset. Python provides powerful libraries like NumPy, Pandas, and SciPy to perform various descriptive statistical operations on data.

Slide 2: Importing Libraries

To get started with descriptive statistics in Python, we need to import the necessary libraries.

```python
import numpy as np
import pandas as pd
```

Slide 3: Mean

The mean is the average value of a dataset. It is calculated by summing all the values and dividing by the total number of values.

```python
data = [5, 8, 2, 9, 6]
mean = sum(data) / len(data)
print(f'Mean: {mean}')  # Output: Mean: 6.0
```

Slide 4: Median

The median is the middle value of a sorted dataset. If the dataset has an even number of values, the median is the average of the two middle values.

```python
data = [5, 8, 2, 9, 6]
data.sort()
n = len(data)
if n % 2 == 0:
    median = (data[n//2-1] + data[n//2]) / 2
else:
    median = data[n//2]
print(f'Median: {median}')  # Output: Median: 6.0
```

Slide 5: Mode

The mode is the value that appears most frequently in a dataset. If multiple values occur the same number of times, there are multiple modes.

```python
from collections import Counter

data = [5, 8, 2, 9, 6, 8, 2]
counts = Counter(data)
modes = [value for value, count in counts.items() if count == max(counts.values())]
print(f'Mode(s): {modes}')  # Output: Mode(s): [2, 8]
```

Slide 6: Range

The range is the difference between the maximum and minimum values in a dataset.

```python
data = [5, 8, 2, 9, 6]
range_val = max(data) - min(data)
print(f'Range: {range_val}')  # Output: Range: 7
```

Slide 7: Variance and Standard Deviation

Variance and standard deviation measure the spread of a dataset. Variance is the average of the squared differences from the mean, while standard deviation is the square root of the variance.

```python
import math

data = [5, 8, 2, 9, 6]
mean = sum(data) / len(data)
squared_diffs = [(x - mean)**2 for x in data]
variance = sum(squared_diffs) / len(data)
std_dev = math.sqrt(variance)
print(f'Variance: {variance}')  # Output: Variance: 6.8
print(f'Standard Deviation: {std_dev}')  # Output: Standard Deviation: 2.6076809620810597
```

Slide 8: Percentiles

Percentiles divide a dataset into 100 equal parts. The nth percentile is the value below which n percent of the data falls.

```python
import numpy as np

data = [5, 8, 2, 9, 6]
quartiles = np.percentile(data, [25, 50, 75])
print(f'Quartiles: {quartiles}')  # Output: Quartiles: [ 4.5  6.   8. ]
```

Slide 9: Data Visualization

Descriptive statistics can be visualized using various plots, such as histograms, box plots, and scatter plots. This helps in understanding the data distribution and identifying patterns.

```python
import matplotlib.pyplot as plt

data = [5, 8, 2, 9, 6]
plt.hist(data, bins=5, edgecolor='black')
plt.title('Histogram')
plt.show()
```

Slide 10: Exploring Pandas DataFrame

Pandas provides a powerful DataFrame object for working with structured data. It offers built-in methods for descriptive statistics.

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 45]}
df = pd.DataFrame(data)
print(df.describe())
```

Slide 11: Grouping and Aggregating

Pandas allows grouping and aggregating data based on one or more columns, enabling descriptive statistics calculations for each group.

```python
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['New York', 'London', 'Paris', 'New York', 'London']}
df = pd.DataFrame(data)
grouped = df.groupby('City')['Age'].agg(['mean', 'std'])
print(grouped)
```

Slide 12: Correlation

Correlation measures the strength and direction of the linear relationship between two variables.

```python
import pandas as pd

data = {'X': [1, 2, 3, 4, 5],
        'Y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)
corr = df['X'].corr(df['Y'])
print(f'Correlation: {corr}')  # Output: Correlation: 1.0
```

Slide 13: Missing Data Handling

Handling missing data is crucial for accurate descriptive statistics. Pandas provides methods like dropna(), fillna(), and interpolate() to handle missing values.

```python
import pandas as pd
import numpy as np

data = {'A': [1, np.nan, 3, 4, 5],
        'B': [2, 6, np.nan, 8, 10]}
df = pd.DataFrame(data)
print(df.dropna())  # Drop rows with missing values
print(df.fillna(0))  # Fill missing values with 0
```

Slide 14: Conclusion

Descriptive statistics provide a powerful way to summarize and understand data. Python's extensive libraries offer a wide range of tools for computing and visualizing descriptive statistics, enabling effective data exploration and analysis.

## Meta
Here's a title, description, and hashtags for a TikTok presentation on Descriptive Statistics in Python with an institutional tone:

Mastering Descriptive Statistics in Python

Explore the fundamentals of descriptive statistics using Python's powerful data analysis libraries. This comprehensive guide covers essential concepts such as measures of central tendency, dispersion, and data visualization techniques. Enhance your data analysis skills and gain insights into your datasets with ease.

Hashtags: #DescriptiveStatistics #Python #DataAnalysis #NumPy #Pandas #DataScience #Statistics #DataVisualization #AcademicContent #EducationalTikTok

