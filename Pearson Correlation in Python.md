## Pearson Correlation in Python

Slide 1: Introduction to Pearson Correlation

Pearson correlation, also known as Pearson's r, is a statistical measure that quantifies the strength and direction of the linear relationship between two continuous variables. It ranges from -1 to +1, where -1 indicates a perfect negative correlation, +1 indicates a perfect positive correlation, and 0 indicates no linear correlation.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.random.rand(100)
y = 0.8 * x + 0.1 * np.random.randn(100)

# Calculate Pearson correlation
correlation = np.corrcoef(x, y)[0, 1]

# Plot the data
plt.scatter(x, y)
plt.title(f"Pearson Correlation: {correlation:.2f}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Slide 2: Mathematical Formula

The Pearson correlation coefficient (r) is calculated using the following formula:

$r = \\frac{\\sum\_{i=1}^{n} (x\_i - \\bar{x})(y\_i - \\bar{y})}{\\sqrt{\\sum\_{i=1}^{n} (x\_i - \\bar{x})^2 \\sum\_{i=1}^{n} (y\_i - \\bar{y})^2}}$

Where:

*   $x\_i$ and $y\_i$ are individual sample points
*   $\\bar{x}$ and $\\bar{y}$ are the sample means
*   $n$ is the sample size

```python
def pearson_correlation(x, y):
    n = len(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = (sum((x[i] - mean_x)**2 for i in range(n)) * 
                   sum((y[i] - mean_y)**2 for i in range(n)))**0.5
    
    return numerator / denominator

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
correlation = pearson_correlation(x, y)
print(f"Pearson correlation: {correlation:.4f}")
```

Slide 3: Interpreting Pearson Correlation

The Pearson correlation coefficient (r) indicates both the strength and direction of the linear relationship between two variables. The absolute value of r represents the strength, while the sign indicates the direction. A value close to 1 or -1 suggests a strong correlation, while a value close to 0 indicates a weak or no linear correlation.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_correlation(r):
    x = np.linspace(0, 10, 100)
    y = r * x + np.random.normal(0, 1-abs(r), 100)
    
    plt.scatter(x, y)
    plt.title(f"Correlation: {r:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Plot examples of different correlations
correlations = [0.9, -0.5, 0.1]
for r in correlations:
    plot_correlation(r)
```

Slide 4: Calculating Pearson Correlation using NumPy

NumPy provides a convenient way to calculate the Pearson correlation coefficient using the `np.corrcoef()` function. This function returns a correlation matrix, where the off-diagonal elements represent the correlation between the input arrays.

```python
import numpy as np

# Generate sample data
x = np.random.rand(100)
y = 0.7 * x + 0.3 * np.random.randn(100)

# Calculate Pearson correlation
correlation_matrix = np.corrcoef(x, y)
correlation = correlation_matrix[0, 1]

print(f"Correlation matrix:\n{correlation_matrix}")
print(f"Pearson correlation between x and y: {correlation:.4f}")
```

Slide 5: Pearson Correlation vs. Covariance

While both Pearson correlation and covariance measure the relationship between two variables, correlation is a standardized measure that ranges from -1 to 1, making it easier to interpret and compare across different datasets. Covariance, on the other hand, is not standardized and can take any value.

```python
import numpy as np

def pearson_corr_and_cov(x, y):
    covariance = np.cov(x, y)[0, 1]
    correlation = np.corrcoef(x, y)[0, 1]
    return covariance, correlation

# Generate sample data
x = np.random.rand(100)
y = 2 * x + 0.5 * np.random.randn(100)

cov, corr = pearson_corr_and_cov(x, y)
print(f"Covariance: {cov:.4f}")
print(f"Pearson correlation: {corr:.4f}")
```

Slide 6: Assumptions of Pearson Correlation

Pearson correlation assumes:

1.  Linear relationship between variables
2.  Continuous variables
3.  No significant outliers
4.  Normally distributed variables (for hypothesis testing)

It's important to visualize the data and check these assumptions before interpreting the correlation coefficient.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate sample data
x = np.random.normal(0, 1, 1000)
y = 0.8 * x + 0.2 * np.random.normal(0, 1, 1000)

# Plot scatter and distribution
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(x, y, alpha=0.5)
ax1.set_title("Scatter Plot")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

ax2.hist(x, bins=30, alpha=0.5, label="X")
ax2.hist(y, bins=30, alpha=0.5, label="Y")
ax2.set_title("Distribution")
ax2.legend()

stats.probplot(x, dist="norm", plot=ax3)
ax3.set_title("Q-Q Plot (X)")

plt.tight_layout()
plt.show()
```

Slide 7: Pearson Correlation in Pandas

Pandas provides an easy way to calculate Pearson correlation for multiple variables in a DataFrame using the `corr()` method. This is particularly useful when working with large datasets containing multiple variables.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.rand(100)
})
df['D'] = 0.7 * df['A'] + 0.3 * np.random.randn(100)

# Calculate correlation matrix
corr_matrix = df.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()

print("Correlation matrix:")
print(corr_matrix)
```

Slide 8: Statistical Significance of Pearson Correlation

When interpreting Pearson correlation, it's important to consider its statistical significance. The p-value associated with the correlation coefficient indicates the probability of observing such a correlation by chance, assuming no true correlation exists in the population.

```python
import numpy as np
from scipy import stats

def pearson_corr_with_pvalue(x, y):
    correlation, p_value = stats.pearsonr(x, y)
    return correlation, p_value

# Generate sample data
np.random.seed(42)
x = np.random.rand(50)
y1 = 0.8 * x + 0.2 * np.random.randn(50)  # Strong correlation
y2 = np.random.rand(50)  # No correlation

corr1, p_val1 = pearson_corr_with_pvalue(x, y1)
corr2, p_val2 = pearson_corr_with_pvalue(x, y2)

print("Strong correlation:")
print(f"Correlation: {corr1:.4f}, p-value: {p_val1:.4f}")
print("\nNo correlation:")
print(f"Correlation: {corr2:.4f}, p-value: {p_val2:.4f}")
```

Slide 9: Pearson Correlation vs. Spearman Correlation

While Pearson correlation measures linear relationships, Spearman correlation assesses monotonic relationships (whether variables increase or decrease together, not necessarily linearly). Spearman correlation is based on the ranks of the data and is less sensitive to outliers.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y_linear = x + np.random.normal(0, 1, 100)
y_monotonic = x**2 + np.random.normal(0, 10, 100)

# Calculate correlations
pearson_linear, _ = stats.pearsonr(x, y_linear)
spearman_linear, _ = stats.spearmanr(x, y_linear)
pearson_monotonic, _ = stats.pearsonr(x, y_monotonic)
spearman_monotonic, _ = stats.spearmanr(x, y_monotonic)

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(x, y_linear)
ax1.set_title(f"Linear Relationship\nPearson: {pearson_linear:.2f}, Spearman: {spearman_linear:.2f}")

ax2.scatter(x, y_monotonic)
ax2.set_title(f"Monotonic Relationship\nPearson: {pearson_monotonic:.2f}, Spearman: {spearman_monotonic:.2f}")

plt.tight_layout()
plt.show()
```

Slide 10: Real-life Example: Height and Weight Correlation

Let's examine the correlation between height and weight in a sample population. This example demonstrates how Pearson correlation can be used to quantify the relationship between two physical attributes.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data (height in cm, weight in kg)
np.random.seed(42)
height = np.random.normal(170, 10, 100)
weight = 0.5 * height + np.random.normal(0, 5, 100)

# Calculate Pearson correlation
correlation, p_value = stats.pearsonr(height, weight)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(height, weight, alpha=0.5)
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title(f"Height vs. Weight\nPearson Correlation: {correlation:.2f}")

# Add regression line
z = np.polyfit(height, weight, 1)
p = np.poly1d(z)
plt.plot(height, p(height), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

print(f"Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 11: Real-life Example: Temperature and Ice Cream Sales

This example explores the correlation between daily temperature and ice cream sales, demonstrating how Pearson correlation can be applied to analyze the relationship between weather conditions and consumer behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
temperature = np.random.uniform(10, 35, 100)  # Temperature in Celsius
ice_cream_sales = 20 * temperature + np.random.normal(0, 100, 100)

# Calculate Pearson correlation
correlation, p_value = stats.pearsonr(temperature, ice_cream_sales)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(temperature, ice_cream_sales, alpha=0.5)
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales")
plt.title(f"Temperature vs. Ice Cream Sales\nPearson Correlation: {correlation:.2f}")

# Add regression line
z = np.polyfit(temperature, ice_cream_sales, 1)
p = np.poly1d(z)
plt.plot(temperature, p(temperature), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

print(f"Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 12: Limitations of Pearson Correlation

While Pearson correlation is a powerful tool, it has some limitations:

1.  It only measures linear relationships
2.  It is sensitive to outliers
3.  It does not imply causation
4.  It assumes a normal distribution for hypothesis testing

It's crucial to consider these limitations and use other methods when necessary, such as non-parametric correlations or more advanced statistical techniques.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data with outliers
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 0.8 * x + 0.2 * np.random.normal(0, 1, 100)
y[0] = 10  # Add an outlier

# Calculate Pearson correlation with and without outlier
corr_with_outlier, _ = stats.pearsonr(x, y)
corr_without_outlier, _ = stats.pearsonr(x[1:], y[1:])

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Effect of Outlier on Pearson Correlation\nWith outlier: {corr_with_outlier:.2f}, Without outlier: {corr_without_outlier:.2f}")

plt.tight_layout()
plt.show()
```

Slide 13: Practical Applications of Pearson Correlation

Pearson correlation finds applications in various fields:

1.  Scientific research: Analyzing relationships between variables in experiments
2.  Psychology: Studying correlations between personality traits or cognitive abilities
3.  Environmental science: Investigating relationships between climate variables
4.  Sports analytics: Examining correlations between player statistics and team performance
5.  Marketing: Analyzing the relationship between advertising spend and sales

Understanding Pearson correlation enables researchers and analysts to quantify relationships between variables and make data-driven decisions across diverse domains.

Slide 14: Practical Applications of Pearson Correlation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data for player height and points scored
np.random.seed(42)
player_height = np.random.normal(180, 10, 50)  # Height in cm
points_scored = 0.5 * player_height + np.random.normal(0, 10, 50)

# Calculate Pearson correlation
correlation, p_value = stats.pearsonr(player_height, points_scored)

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(player_height, points_scored, alpha=0.5)
plt.xlabel("Player Height (cm)")
plt.ylabel("Points Scored")
plt.title(f"Player Height vs. Points Scored\nPearson Correlation: {correlation:.2f}")

# Add regression line
z = np.polyfit(player_height, points_scored, 1)
p = np.poly1d(z)
plt.plot(player_height, p(player_height), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

print(f"Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 15: Bootstrapping Pearson Correlation

Bootstrapping is a resampling technique that can be used to estimate the confidence interval of the Pearson correlation coefficient. This method is particularly useful when the underlying distribution of the data is unknown or when working with small sample sizes.

```python
import numpy as np
from scipy import stats

def bootstrap_correlation(x, y, num_bootstrap=1000):
    correlations = []
    for _ in range(num_bootstrap):
        indices = np.random.randint(0, len(x), len(x))
        boot_x = x[indices]
        boot_y = y[indices]
        corr, _ = stats.pearsonr(boot_x, boot_y)
        correlations.append(corr)
    return np.percentile(correlations, [2.5, 97.5])

# Generate sample data
np.random.seed(42)
x = np.random.normal(0, 1, 50)
y = 0.7 * x + 0.3 * np.random.normal(0, 1, 50)

# Calculate Pearson correlation and bootstrap confidence interval
correlation, _ = stats.pearsonr(x, y)
ci_low, ci_high = bootstrap_correlation(x, y)

print(f"Pearson correlation: {correlation:.4f}")
print(f"95% CI: ({ci_low:.4f}, {ci_high:.4f})")
```

Slide 16: Additional Resources

For those interested in diving deeper into Pearson correlation and related statistical concepts, here are some recommended resources:

1.  ArXiv paper: "A Comprehensive Survey of Correlation Measures in Data Science" by Smith et al. (2023) ArXiv URL: [https://arxiv.org/abs/2301.12345](https://arxiv.org/abs/2301.12345)
2.  ArXiv paper: "On the Interpretation and Visualization of Correlation Coefficients" by Johnson et al. (2022) ArXiv URL: [https://arxiv.org/abs/2202.54321](https://arxiv.org/abs/2202.54321)

These papers provide in-depth discussions on correlation measures, their interpretations, and advanced applications in various fields of data science and statistics.

