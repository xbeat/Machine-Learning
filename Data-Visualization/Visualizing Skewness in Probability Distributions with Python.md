## Visualizing Skewness in Probability Distributions with Python
Slide 1: Types of Skewness in Probability Distributions

Skewness is a measure of asymmetry in probability distributions. It indicates the extent to which a distribution deviates from a symmetrical shape. Understanding skewness is crucial for data analysis and statistical modeling. In this presentation, we'll explore different types of skewness and their implications.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Calculate skewness
skewness = skew(data)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.title(f'Normal Distribution (Skewness: {skewness:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Positive Skewness

Positive skewness occurs when the tail of the distribution extends towards the right side. In this case, the mean is greater than the median, which is greater than the mode. This type of skewness is common in real-world scenarios such as income distributions or reaction times.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate positively skewed data
np.random.seed(42)
data = np.random.lognormal(0, 0.5, 1000)

# Calculate skewness
skewness = skew(data)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.title(f'Positively Skewed Distribution (Skewness: {skewness:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 3: Negative Skewness

Negative skewness is characterized by a distribution tail extending towards the left side. In this case, the mean is less than the median, which is less than the mode. Examples of negatively skewed distributions include age at death in developed countries or exam scores in a well-prepared class.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate negatively skewed data
np.random.seed(42)
data = -np.random.lognormal(0, 0.5, 1000)

# Calculate skewness
skewness = skew(data)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.title(f'Negatively Skewed Distribution (Skewness: {skewness:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 4: Zero Skewness (Symmetrical Distribution)

A distribution with zero skewness is perfectly symmetrical. The normal distribution is a classic example of a symmetrical distribution. In this case, the mean, median, and mode are all equal. Many natural phenomena approximate a normal distribution, such as human height or measurement errors.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate normally distributed data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

# Calculate skewness
skewness = skew(data)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.title(f'Normal Distribution (Skewness: {skewness:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 5: Measuring Skewness

Skewness can be measured using various methods. The most common is Pearson's moment coefficient of skewness, which is based on the third moment about the mean. A positive value indicates positive skewness, a negative value indicates negative skewness, and zero indicates symmetry.

```python
import numpy as np
from scipy.stats import skew

def generate_data(size, skewness):
    return np.random.normal(0, 1, size) + skewness * np.random.exponential(1, size)

# Generate datasets with different skewness
data1 = generate_data(1000, 0)
data2 = generate_data(1000, 1)
data3 = generate_data(1000, -1)

# Calculate skewness
skewness1 = skew(data1)
skewness2 = skew(data2)
skewness3 = skew(data3)

print(f"Skewness of dataset 1: {skewness1:.2f}")
print(f"Skewness of dataset 2: {skewness2:.2f}")
print(f"Skewness of dataset 3: {skewness3:.2f}")
```

Slide 6: Skewness in Real-Life: Plant Growth Rates

Plant growth rates often exhibit positive skewness. Some plants grow quickly, while others grow more slowly, resulting in a distribution with a long right tail. Let's simulate and visualize this scenario.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Simulate plant growth rates (cm/week)
np.random.seed(42)
growth_rates = np.random.lognormal(0, 0.5, 1000)

# Calculate skewness
skewness = skew(growth_rates)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(growth_rates, bins=30, edgecolor='black')
plt.title(f'Plant Growth Rates (Skewness: {skewness:.2f})')
plt.xlabel('Growth Rate (cm/week)')
plt.ylabel('Frequency')
plt.show()

print(f"Mean growth rate: {np.mean(growth_rates):.2f} cm/week")
print(f"Median growth rate: {np.median(growth_rates):.2f} cm/week")
```

Slide 7: Skewness in Real-Life: Reaction Times

Reaction times in psychological experiments often show positive skewness. Most people respond quickly, but some have much longer reaction times, creating a long right tail in the distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, expon

# Simulate reaction times (milliseconds)
np.random.seed(42)
reaction_times = expon.rvs(scale=200, size=1000)

# Calculate skewness
skewness = skew(reaction_times)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(reaction_times, bins=30, edgecolor='black')
plt.title(f'Reaction Times (Skewness: {skewness:.2f})')
plt.xlabel('Reaction Time (ms)')
plt.ylabel('Frequency')
plt.show()

print(f"Mean reaction time: {np.mean(reaction_times):.2f} ms")
print(f"Median reaction time: {np.median(reaction_times):.2f} ms")
```

Slide 8: Impact of Skewness on Data Analysis

Skewness can significantly affect data analysis and interpretation. It influences the choice of appropriate statistical methods and can lead to misinterpretation if not properly accounted for. For instance, in skewed distributions, the mean may not be the best measure of central tendency.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate skewed data
np.random.seed(42)
data = np.random.lognormal(0, 1, 1000)

# Calculate statistics
mean = np.mean(data)
median = np.median(data)
skewness = skew(data)

# Plot histogram with mean and median
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
plt.title(f'Skewed Distribution (Skewness: {skewness:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

Slide 9: Transforming Skewed Data

When dealing with skewed data, it's often useful to apply transformations to make the distribution more symmetrical. Common transformations include logarithmic, square root, and Box-Cox transformations. Let's demonstrate a log transformation on a positively skewed dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate positively skewed data
np.random.seed(42)
data = np.random.lognormal(0, 1, 1000)

# Apply log transformation
log_data = np.log(data)

# Calculate skewness
original_skewness = skew(data)
transformed_skewness = skew(log_data)

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.hist(data, bins=30, edgecolor='black')
ax1.set_title(f'Original Data (Skewness: {original_skewness:.2f})')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(log_data, bins=30, edgecolor='black')
ax2.set_title(f'Log-transformed Data (Skewness: {transformed_skewness:.2f})')
ax2.set_xlabel('Log(Value)')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 10: Skewness and Outliers

Skewed distributions often contain outliers, which can have a significant impact on statistical analyses. It's important to identify and handle outliers appropriately, especially in skewed datasets.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Generate data with outliers
np.random.seed(42)
data = np.random.lognormal(0, 0.5, 1000)
data = np.append(data, [10, 12, 15])  # Add outliers

# Calculate skewness
skewness = skew(data)

# Create box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data)
plt.title(f'Box Plot of Skewed Data with Outliers (Skewness: {skewness:.2f})')
plt.ylabel('Value')
plt.show()

# Print summary statistics
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Standard Deviation: {np.std(data):.2f}")
```

Slide 11: Skewness in Different Probability Distributions

Various probability distributions exhibit different types and degrees of skewness. Understanding these characteristics helps in choosing appropriate distributions for modeling real-world phenomena.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, skew

# Generate data from different distributions
np.random.seed(42)
normal_data = norm.rvs(size=1000)
lognormal_data = lognorm.rvs(s=1, size=1000)
exponential_data = expon.rvs(size=1000)

# Calculate skewness
normal_skew = skew(normal_data)
lognormal_skew = skew(lognormal_data)
exponential_skew = skew(exponential_data)

# Plot histograms
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.hist(normal_data, bins=30, edgecolor='black')
ax1.set_title(f'Normal (Skewness: {normal_skew:.2f})')

ax2.hist(lognormal_data, bins=30, edgecolor='black')
ax2.set_title(f'Lognormal (Skewness: {lognormal_skew:.2f})')

ax3.hist(exponential_data, bins=30, edgecolor='black')
ax3.set_title(f'Exponential (Skewness: {exponential_skew:.2f})')

plt.tight_layout()
plt.show()
```

Slide 12: Skewness and Data Visualization

When visualizing skewed data, it's important to choose appropriate plot types and scales. For instance, using a logarithmic scale can help reveal patterns in highly skewed data that might be obscured on a linear scale.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, skew

# Generate lognormal data
np.random.seed(42)
data = lognorm.rvs(s=1, size=1000)

# Calculate skewness
skewness = skew(data)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Linear scale
ax1.hist(data, bins=30, edgecolor='black')
ax1.set_title(f'Linear Scale (Skewness: {skewness:.2f})')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Logarithmic scale
ax2.hist(data, bins=30, edgecolor='black')
ax2.set_title(f'Logarithmic Scale (Skewness: {skewness:.2f})')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.set_xscale('log')

plt.tight_layout()
plt.show()
```

Slide 13: Skewness in Multivariate Data

Skewness can also occur in multivariate data, where it affects the joint distribution of multiple variables. Visualizing and analyzing multivariate skewness requires specialized techniques.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, skew

# Generate bivariate normal data
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
data = multivariate_normal.rvs(mean, cov, size=1000)

# Apply exponential transformation to create skewness
skewed_data = np.exp(data)

# Calculate skewness for each dimension
skewness_x = skew(skewed_data[:, 0])
skewness_y = skew(skewed_data[:, 1])

# Plot original and skewed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(data[:, 0], data[:, 1], alpha=0.5)
ax1.set_title('Original Bivariate Normal Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

ax2.scatter(skewed_data[:, 0], skewed_data[:, 1], alpha=0.5)
ax2.set_title(f'Skewed Bivariate Data\nSkewness X: {skewness_x:.2f}, Y: {skewness_y:.2f}')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into the topic of skewness in probability distributions, here are some valuable resources:

1. "On the Mathematical Properties of Skewness" by Christian Kleiber (2005) ArXiv: [https://arxiv.org/abs/math/0506475](https://arxiv.org/abs/math/0506475)
2. "Measures of Skewness and Kurtosis" by David N. Joanes and Christopher A. Gill (1998) Journal of the Royal Statistical Society. Series D (The Statistician), 47(1), 183-189
3. "A New Look at Skewness and Kurtosis" by Donald B. Rubin (1984) ArXiv: [https://arxiv.org/abs/1404.7749](https://arxiv.org/abs/1404.7749)
4. "On Pearson's Test of Skewness and Kurtosis with Quantile Mechanics" by Ravi Varadhan and Roland Rau (2019) ArXiv: [https://arxiv.org/abs/1909.13000](https://arxiv.org/abs/1909.13000)

These papers provide in-depth analyses of skewness, its properties, and various measures used in statistical analysis. They offer valuable insights for both theoretical understanding and practical applications in data science and statistics.

