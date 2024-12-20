## The Story Behind the 68-95-99.7 Rule in Normal Distribution
Slide 1: The Normal Distribution and the 68-95-99.7 Rule

The normal distribution, also known as the Gaussian distribution, is a fundamental concept in statistics. The 68-95-99.7 rule, sometimes called the empirical rule, is a simple yet powerful tool for understanding the spread of data in a normal distribution. This rule helps us interpret the standard deviation and provides a quick way to estimate the probability of data falling within certain ranges.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data for a normal distribution
mu, sigma = 0, 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Plot the normal distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Slide 2: Understanding Standard Deviation

Standard deviation is a measure of the spread of data in a distribution. In a normal distribution, it determines the width of the bell curve. The 68-95-99.7 rule is based on the number of standard deviations from the mean.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data from a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Plot histogram with normal distribution overlay
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
plt.plot(x, 1/(std_dev * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * std_dev**2)), 
         linewidth=2, color='red')
plt.title(f'Normal Distribution (Mean: {mean:.2f}, Std Dev: {std_dev:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

Slide 3: The 68% Rule

According to the 68-95-99.7 rule, approximately 68% of the data falls within one standard deviation of the mean in a normal distribution. This means that if we take any normal distribution and look at the area within one standard deviation on either side of the mean, we'll find about 68% of the total data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set up the plot
plt.figure(figsize=(10, 6))
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the normal distribution
plt.plot(x, y, 'b', linewidth=2)

# Shade the area within 1 standard deviation
x_filled = np.linspace(-1, 1, 1000)
y_filled = norm.pdf(x_filled, 0, 1)
plt.fill_between(x_filled, y_filled, color='skyblue', alpha=0.5)

plt.title('68% Rule: Area within 1 Standard Deviation')
plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability Density')
plt.grid(True)

# Add text to show the percentage
plt.text(0, 0.2, '68%', horizontalalignment='center', fontsize=20)

plt.show()

# Calculate the exact percentage
exact_percentage = (norm.cdf(1) - norm.cdf(-1)) * 100
print(f"Exact percentage within 1 standard deviation: {exact_percentage:.2f}%")
```

Slide 4: The 95% Rule

Extending our understanding, the 95% rule states that approximately 95% of the data in a normal distribution falls within two standard deviations of the mean. This provides a wider range that encompasses even more of the data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set up the plot
plt.figure(figsize=(10, 6))
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the normal distribution
plt.plot(x, y, 'b', linewidth=2)

# Shade the area within 2 standard deviations
x_filled = np.linspace(-2, 2, 1000)
y_filled = norm.pdf(x_filled, 0, 1)
plt.fill_between(x_filled, y_filled, color='lightgreen', alpha=0.5)

plt.title('95% Rule: Area within 2 Standard Deviations')
plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability Density')
plt.grid(True)

# Add text to show the percentage
plt.text(0, 0.2, '95%', horizontalalignment='center', fontsize=20)

plt.show()

# Calculate the exact percentage
exact_percentage = (norm.cdf(2) - norm.cdf(-2)) * 100
print(f"Exact percentage within 2 standard deviations: {exact_percentage:.2f}%")
```

Slide 5: The 99.7% Rule

The final part of the 68-95-99.7 rule states that approximately 99.7% of the data falls within three standard deviations of the mean. This encompasses almost all of the data in a normal distribution, leaving only about 0.3% in the extreme tails.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set up the plot
plt.figure(figsize=(10, 6))
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the normal distribution
plt.plot(x, y, 'b', linewidth=2)

# Shade the area within 3 standard deviations
x_filled = np.linspace(-3, 3, 1000)
y_filled = norm.pdf(x_filled, 0, 1)
plt.fill_between(x_filled, y_filled, color='salmon', alpha=0.5)

plt.title('99.7% Rule: Area within 3 Standard Deviations')
plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability Density')
plt.grid(True)

# Add text to show the percentage
plt.text(0, 0.2, '99.7%', horizontalalignment='center', fontsize=20)

plt.show()

# Calculate the exact percentage
exact_percentage = (norm.cdf(3) - norm.cdf(-3)) * 100
print(f"Exact percentage within 3 standard deviations: {exact_percentage:.2f}%")
```

Slide 6: Visualizing the 68-95-99.7 Rule

Let's create a visual representation of the 68-95-99.7 rule by overlaying all three ranges on a single normal distribution plot. This will help us see how these percentages relate to each other and to the overall shape of the distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set up the plot
plt.figure(figsize=(12, 8))
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the normal distribution
plt.plot(x, y, 'k', linewidth=2)

# Shade the areas for 68%, 95%, and 99.7%
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), color='red', alpha=0.3, label='68%')
plt.fill_between(x, y, where=(x >= -2) & (x <= 2), color='yellow', alpha=0.3, label='95%')
plt.fill_between(x, y, where=(x >= -3) & (x <= 3), color='blue', alpha=0.3, label='99.7%')

plt.title('Visualization of the 68-95-99.7 Rule')
plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Add text to show the percentages
plt.text(0, 0.2, '68%', horizontalalignment='center', fontsize=16, color='red')
plt.text(0, 0.1, '95%', horizontalalignment='center', fontsize=16, color='orange')
plt.text(0, 0.05, '99.7%', horizontalalignment='center', fontsize=16, color='blue')

plt.show()
```

Slide 7: Historical Context

The 68-95-99.7 rule has its roots in the work of Carl Friedrich Gauss, who developed the concept of the normal distribution in the early 19th century. However, the rule itself was formalized later as statisticians sought ways to quickly interpret and communicate the properties of normal distributions.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

# Create a timeline
ax.plot([1800, 2023], [0, 0], 'k-', linewidth=2)

# Add key events
events = [
    (1809, "Gauss publishes 'Theoria motus corporum coelestium'"),
    (1894, "Karl Pearson coins the term 'standard deviation'"),
    (1920, "Fisher introduces the concept of degrees of freedom"),
    (1960, "68-95-99.7 rule becomes widely taught in statistics courses")
]

for year, event in events:
    ax.add_patch(Circle((year, 0), 5, facecolor='blue'))
    ax.text(year, 0.1, str(year), ha='center')
    ax.text(year, -0.1, event, ha='center', va='top', wrap=True)

ax.set_xlim(1800, 2023)
ax.set_ylim(-0.5, 0.5)
ax.axis('off')

plt.title("Timeline of the 68-95-99.7 Rule and Normal Distribution")
plt.tight_layout()
plt.show()
```

Slide 8: Real-Life Example: Height Distribution

One common application of the 68-95-99.7 rule is in understanding human height distributions. Let's consider adult male heights, which often follow a normal distribution with a mean of about 70 inches (177.8 cm) and a standard deviation of about 3 inches (7.62 cm).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for male height distribution (in inches)
mean_height = 70
std_dev_height = 3

# Generate height data
heights = np.random.normal(mean_height, std_dev_height, 10000)

# Create the plot
plt.figure(figsize=(12, 6))
plt.hist(heights, bins=50, density=True, alpha=0.7, color='skyblue')

# Plot the theoretical normal distribution
x = np.linspace(mean_height - 4*std_dev_height, mean_height + 4*std_dev_height, 100)
plt.plot(x, norm.pdf(x, mean_height, std_dev_height), 'r-', lw=2)

# Add vertical lines for standard deviations
for i, color in zip(range(1, 4), ['green', 'orange', 'red']):
    plt.axvline(mean_height + i*std_dev_height, color=color, linestyle='--', alpha=0.5)
    plt.axvline(mean_height - i*std_dev_height, color=color, linestyle='--', alpha=0.5)

plt.title('Distribution of Adult Male Heights')
plt.xlabel('Height (inches)')
plt.ylabel('Probability Density')
plt.grid(True)

# Add text for percentages
plt.text(mean_height, 0.05, '68%', ha='center', va='bottom', color='green')
plt.text(mean_height, 0.04, '95%', ha='center', va='bottom', color='orange')
plt.text(mean_height, 0.03, '99.7%', ha='center', va='bottom', color='red')

plt.show()

# Calculate percentages within each range
within_1_std = np.sum((heights >= mean_height - std_dev_height) & (heights <= mean_height + std_dev_height)) / len(heights) * 100
within_2_std = np.sum((heights >= mean_height - 2*std_dev_height) & (heights <= mean_height + 2*std_dev_height)) / len(heights) * 100
within_3_std = np.sum((heights >= mean_height - 3*std_dev_height) & (heights <= mean_height + 3*std_dev_height)) / len(heights) * 100

print(f"Percentage within 1 standard deviation: {within_1_std:.2f}%")
print(f"Percentage within 2 standard deviations: {within_2_std:.2f}%")
print(f"Percentage within 3 standard deviations: {within_3_std:.2f}%")
```

Slide 9: Real-Life Example: Reaction Times

Another application of the 68-95-99.7 rule can be found in cognitive psychology, specifically in the study of reaction times. Reaction times often follow a normal distribution, and understanding this distribution can be crucial in fields like human-computer interaction and traffic safety.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for reaction time distribution (in milliseconds)
mean_rt, std_dev_rt = 250, 50

# Generate reaction time data
reaction_times = np.random.normal(mean_rt, std_dev_rt, 10000)

# Create the plot
plt.figure(figsize=(12, 6))
plt.hist(reaction_times, bins=50, density=True, alpha=0.7, color='lightgreen')

# Plot the theoretical normal distribution
x = np.linspace(mean_rt - 4*std_dev_rt, mean_rt + 4*std_dev_rt, 100)
plt.plot(x, norm.pdf(x, mean_rt, std_dev_rt), 'r-', lw=2)

# Add vertical lines for standard deviations
for i, color in zip(range(1, 4), ['blue', 'purple', 'orange']):
    plt.axvline(mean_rt + i*std_dev_rt, color=color, linestyle='--', alpha=0.5)
    plt.axvline(mean_rt - i*std_dev_rt, color=color, linestyle='--', alpha=0.5)

plt.title('Distribution of Reaction Times')
plt.xlabel('Reaction Time (milliseconds)')
plt.ylabel('Probability Density')
plt.grid(True)

# Add text for percentages
for i, (text, color) in enumerate(zip(['68%', '95%', '99.7%'], ['blue', 'purple', 'orange'])):
    plt.text(mean_rt, 0.004 - i*0.001, text, ha='center', va='bottom', color=color)

plt.show()

# Calculate percentages within each range
for i in range(1, 4):
    within_std = np.sum(np.abs(reaction_times - mean_rt) <= i*std_dev_rt) / len(reaction_times) * 100
    print(f"Percentage within {i} standard deviation(s): {within_std:.2f}%")
```

Slide 10: Practical Applications of the 68-95-99.7 Rule

The 68-95-99.7 rule finds applications in various fields, including quality control, scientific research, and data analysis. It provides a quick way to assess the likelihood of certain outcomes or to identify unusual events.

```python
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate a manufacturing process
def simulate_process(mean, std_dev, n_samples):
    return np.random.normal(mean, std_dev, n_samples)

# Set parameters
target = 100  # Target value
tolerance = 2  # Tolerance range
mean = 100
std_dev = 1
n_samples = 1000

# Simulate the process
measurements = simulate_process(mean, std_dev, n_samples)

# Plot the results
plt.figure(figsize=(12, 6))
plt.hist(measurements, bins=50, density=True, alpha=0.7)
plt.axvline(target, color='r', linestyle='--', label='Target')
plt.axvline(target - tolerance, color='g', linestyle='--', label='Lower Limit')
plt.axvline(target + tolerance, color='g', linestyle='--', label='Upper Limit')

plt.title('Quality Control in Manufacturing')
plt.xlabel('Measurement')
plt.ylabel('Frequency')
plt.legend()

# Calculate the percentage within tolerance
within_tolerance = np.sum((measurements >= target - tolerance) & 
                          (measurements <= target + tolerance)) / n_samples * 100

plt.text(target, 0.2, f'{within_tolerance:.1f}% within tolerance', 
         ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

plt.show()
```

Slide 11: Limitations and Considerations

While the 68-95-99.7 rule is a powerful tool, it's important to remember that it applies strictly to normal distributions. Not all data follows a normal distribution, and applying this rule to non-normal data can lead to incorrect conclusions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skewnorm

# Generate data for normal and skewed distributions
x = np.linspace(-5, 5, 1000)
y_normal = norm.pdf(x, 0, 1)
y_skewed = skewnorm.pdf(x, 5)

# Plot the distributions
plt.figure(figsize=(12, 6))
plt.plot(x, y_normal, label='Normal Distribution')
plt.plot(x, y_skewed, label='Skewed Distribution')

plt.title('Comparison of Normal and Skewed Distributions')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)

# Add text to highlight the difference
plt.text(0, 0.2, '68-95-99.7 rule\napplies here', ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.5))
plt.text(2, 0.2, '68-95-99.7 rule\ndoes not apply', ha='center', va='center', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.show()
```

Slide 12: Testing for Normality

Before applying the 68-95-99.7 rule, it's crucial to verify that your data follows a normal distribution. There are several statistical tests and visual methods to assess normality.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate some sample data
np.random.seed(0)
normal_data = np.random.normal(0, 1, 1000)
non_normal_data = np.random.exponential(1, 1000)

# Function to create Q-Q plot
def create_qq_plot(data, title):
    plt.figure(figsize=(8, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)
    plt.show()

# Create Q-Q plots
create_qq_plot(normal_data, "Q-Q Plot: Normal Distribution")
create_qq_plot(non_normal_data, "Q-Q Plot: Non-Normal Distribution")

# Perform Shapiro-Wilk test
_, p_value_normal = stats.shapiro(normal_data)
_, p_value_non_normal = stats.shapiro(non_normal_data)

print(f"Shapiro-Wilk test p-value (normal data): {p_value_normal:.4f}")
print(f"Shapiro-Wilk test p-value (non-normal data): {p_value_non_normal:.4f}")
```

Slide 13: Beyond the 68-95-99.7 Rule

While the 68-95-99.7 rule provides a quick reference for normal distributions, more precise calculations can be made using the cumulative distribution function (CDF) of the normal distribution.

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Define the standard normal distribution
mean, std_dev = 0, 1

# Calculate probabilities for different ranges
probabilities = [
    norm.cdf(1) - norm.cdf(-1),
    norm.cdf(2) - norm.cdf(-2),
    norm.cdf(3) - norm.cdf(-3),
    norm.cdf(4) - norm.cdf(-4)
]

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(range(1, 5), probabilities, tick_label=['±1σ', '±2σ', '±3σ', '±4σ'])
plt.title('Probability Within Different Standard Deviation Ranges')
plt.xlabel('Range (in standard deviations)')
plt.ylabel('Probability')
plt.ylim(0, 1)

for i, prob in enumerate(probabilities):
    plt.text(i+1, prob, f'{prob:.4f}', ha='center', va='bottom')

plt.show()

# Print the exact probabilities
for i, prob in enumerate(probabilities, 1):
    print(f"Probability within ±{i}σ: {prob:.4f}")
```

Slide 14: Conclusion and Key Takeaways

The 68-95-99.7 rule is a powerful tool for understanding and interpreting data in normal distributions. It provides a quick way to estimate probabilities and identify unusual events. Remember these key points:

1. The rule applies only to normal distributions.
2. It's an approximation, but a very good one for many practical purposes.
3. Always check for normality before applying the rule.
4. For more precise calculations, use the cumulative distribution function.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a simple visual summary
fig, ax = plt.subplots(figsize=(12, 6))

# Draw the normal distribution curve
x = np.linspace(-4, 4, 1000)
y = 1 / (np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
ax.plot(x, y, 'k-', lw=2)

# Fill areas for 68%, 95%, and 99.7%
ax.fill_between(x, y, where=(x >= -1) & (x <= 1), color='red', alpha=0.3)
ax.fill_between(x, y, where=(x >= -2) & (x <= 2), color='green', alpha=0.3)
ax.fill_between(x, y, where=(x >= -3) & (x <= 3), color='blue', alpha=0.3)

# Add labels
ax.text(0, 0.2, '68%', ha='center', va='center', fontsize=16)
ax.text(0, 0.1, '95%', ha='center', va='center', fontsize=16)
ax.text(0, 0.05, '99.7%', ha='center', va='center', fontsize=16)

ax.set_title('The 68-95-99.7 Rule', fontsize=20)
ax.set_xlabel('Standard Deviations from Mean', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.set_ylim(0, 0.45)

plt.tight_layout()
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into the topic of normal distributions and statistical analysis, here are some recommended resources:

1. "The Art of Statistics: Learning from Data" by David Spiegelhalter
2. "Statistics" by David Freedman, Robert Pisani, and Roger Purves
3. "An Introduction to Statistical Learning" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani

For more technical and advanced treatments:

1. "Statistical Inference" by George Casella and Roger L. Berger
2. "Probability Theory: The Logic of Science" by E.T. Jaynes

Online resources:

1. Khan Academy's Statistics and Probability course
2. Coursera's "Statistics with R Specialization" by Duke University

For the latest research on statistical methods and their applications, you can explore papers on arXiv.org in the Statistics category: [https://arxiv.org/list/stat/recent](https://arxiv.org/list/stat/recent)

Remember to critically evaluate any information you find and consider how it applies to your specific context and needs.

