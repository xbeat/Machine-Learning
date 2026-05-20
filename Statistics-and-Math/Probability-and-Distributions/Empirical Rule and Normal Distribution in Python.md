## Empirical Rule and Normal Distribution in Python
Slide 1: The Empirical Rule in Normal Distribution

The Empirical Rule, also known as the 68-95-99.7 rule, is a fundamental concept in statistics that describes the distribution of data in a normal distribution. This rule provides a quick way to understand the spread of data around the mean.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for a standard normal distribution
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Standard Normal Distribution")
plt.xlabel("Standard Deviations from Mean")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
```

Slide 2: Understanding the Standard Deviation

The standard deviation is a measure of the spread of data in a distribution. In a normal distribution, it determines the width of the bell curve. The Empirical Rule uses standard deviations to describe data percentages.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate random data
data = np.random.normal(0, 1, 1000)

# Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7)
plt.axvline(mean, color='r', linestyle='dashed', linewidth=2)
plt.axvline(mean + std, color='g', linestyle='dashed', linewidth=2)
plt.axvline(mean - std, color='g', linestyle='dashed', linewidth=2)
plt.title("Normal Distribution with Mean and Standard Deviation")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend(["Mean", "±1 Standard Deviation"])
plt.show()

print(f"Mean: {mean:.2f}")
print(f"Standard Deviation: {std:.2f}")
```

Slide 3: The 68% Rule

According to the Empirical Rule, approximately 68% of the data falls within one standard deviation of the mean in a normal distribution. This forms the core of the rule and is often the most practical for quick estimations.

```python
import numpy as np
from scipy import stats

# Generate a large sample from a normal distribution
sample = np.random.normal(0, 1, 100000)

# Calculate the percentage within 1 standard deviation
within_1_std = np.sum((sample >= -1) & (sample <= 1)) / len(sample) * 100

print(f"Percentage within 1 standard deviation: {within_1_std:.2f}%")
```

Slide 4: The 95% Rule

Extending the concept, the Empirical Rule states that approximately 95% of the data falls within two standard deviations of the mean. This provides a wider range for data analysis and prediction.

```python
import numpy as np
from scipy import stats

# Generate a large sample from a normal distribution
sample = np.random.normal(0, 1, 100000)

# Calculate the percentage within 2 standard deviations
within_2_std = np.sum((sample >= -2) & (sample <= 2)) / len(sample) * 100

print(f"Percentage within 2 standard deviations: {within_2_std:.2f}%")
```

Slide 5: The 99.7% Rule

The final part of the Empirical Rule states that approximately 99.7% of the data falls within three standard deviations of the mean. This encompasses almost all the data in a normal distribution.

```python
import numpy as np
from scipy import stats

# Generate a large sample from a normal distribution
sample = np.random.normal(0, 1, 100000)

# Calculate the percentage within 3 standard deviations
within_3_std = np.sum((sample >= -3) & (sample <= 3)) / len(sample) * 100

print(f"Percentage within 3 standard deviations: {within_3_std:.2f}%")
```

Slide 6: Visualizing the Empirical Rule

To better understand the Empirical Rule, we can create a visual representation of the percentages within each standard deviation range.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data for a standard normal distribution
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Plot the distribution with shaded areas
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'k', linewidth=2)
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), color='red', alpha=0.3)
plt.fill_between(x, y, where=((x >= -2) & (x <= -1)) | ((x >= 1) & (x <= 2)), color='green', alpha=0.3)
plt.fill_between(x, y, where=((x >= -3) & (x <= -2)) | ((x >= 2) & (x <= 3)), color='blue', alpha=0.3)

plt.title("Empirical Rule Visualization")
plt.xlabel("Standard Deviations from Mean")
plt.ylabel("Probability Density")
plt.legend(["Normal Distribution", "68%", "95%", "99.7%"])
plt.grid(True)
plt.show()
```

Slide 7: Applying the Empirical Rule

The Empirical Rule is particularly useful for quick estimations and understanding data spread. Let's apply it to a real-world scenario involving height data.

```python
import numpy as np

# Example: Adult male height data (in cm)
mean_height = 175
std_dev = 7

# Calculate ranges
one_std_range = (mean_height - std_dev, mean_height + std_dev)
two_std_range = (mean_height - 2*std_dev, mean_height + 2*std_dev)
three_std_range = (mean_height - 3*std_dev, mean_height + 3*std_dev)

print(f"68% of heights are between {one_std_range[0]:.1f} cm and {one_std_range[1]:.1f} cm")
print(f"95% of heights are between {two_std_range[0]:.1f} cm and {two_std_range[1]:.1f} cm")
print(f"99.7% of heights are between {three_std_range[0]:.1f} cm and {three_std_range[1]:.1f} cm")
```

Slide 8: Real-life Example: Quality Control

In manufacturing, the Empirical Rule can be applied to quality control processes. Let's consider a light bulb factory where the target lifespan is 1000 hours with a standard deviation of 50 hours.

```python
import numpy as np
from scipy import stats

# Light bulb lifespan parameters
mean_lifespan = 1000  # hours
std_dev = 50  # hours

# Calculate the percentage of bulbs within acceptable range (900-1100 hours)
lower_bound = (900 - mean_lifespan) / std_dev
upper_bound = (1100 - mean_lifespan) / std_dev
acceptable_percentage = (stats.norm.cdf(upper_bound) - stats.norm.cdf(lower_bound)) * 100

print(f"Percentage of bulbs within acceptable range: {acceptable_percentage:.2f}%")

# Calculate the number of standard deviations for the acceptable range
num_std_dev = (1100 - mean_lifespan) / std_dev

print(f"The acceptable range is within {num_std_dev:.2f} standard deviations of the mean")
```

Slide 9: Real-life Example: Weather Forecasting

Meteorologists often use the Empirical Rule to predict temperature ranges. Let's apply this to a summer temperature dataset.

```python
import numpy as np
from scipy import stats

# Summer temperature data (in Celsius)
temperatures = np.array([25, 28, 30, 32, 27, 29, 31, 26, 28, 30])

mean_temp = np.mean(temperatures)
std_dev_temp = np.std(temperatures)

# Calculate temperature ranges
one_std_range = (mean_temp - std_dev_temp, mean_temp + std_dev_temp)
two_std_range = (mean_temp - 2*std_dev_temp, mean_temp + 2*std_dev_temp)

print(f"Mean temperature: {mean_temp:.1f}°C")
print(f"Standard deviation: {std_dev_temp:.1f}°C")
print(f"68% of temperatures are likely between {one_std_range[0]:.1f}°C and {one_std_range[1]:.1f}°C")
print(f"95% of temperatures are likely between {two_std_range[0]:.1f}°C and {two_std_range[1]:.1f}°C")

# Calculate probability of temperature above 35°C
prob_above_35 = 1 - stats.norm.cdf((35 - mean_temp) / std_dev_temp)
print(f"Probability of temperature above 35°C: {prob_above_35:.2%}")
```

Slide 10: Limitations of the Empirical Rule

While the Empirical Rule is powerful, it's important to understand its limitations. It assumes a perfectly normal distribution, which is not always the case in real-world scenarios.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate a skewed distribution
data_skewed = np.random.gamma(2, 2, 10000)

# Calculate mean and standard deviation
mean_skewed = np.mean(data_skewed)
std_skewed = np.std(data_skewed)

# Calculate percentages within 1, 2, and 3 standard deviations
within_1_std = np.sum((data_skewed >= mean_skewed - std_skewed) & (data_skewed <= mean_skewed + std_skewed)) / len(data_skewed) * 100
within_2_std = np.sum((data_skewed >= mean_skewed - 2*std_skewed) & (data_skewed <= mean_skewed + 2*std_skewed)) / len(data_skewed) * 100
within_3_std = np.sum((data_skewed >= mean_skewed - 3*std_skewed) & (data_skewed <= mean_skewed + 3*std_skewed)) / len(data_skewed) * 100

# Plot the skewed distribution
plt.figure(figsize=(10, 6))
plt.hist(data_skewed, bins=50, density=True, alpha=0.7)
plt.title("Skewed Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print(f"Percentage within 1 standard deviation: {within_1_std:.2f}%")
print(f"Percentage within 2 standard deviations: {within_2_std:.2f}%")
print(f"Percentage within 3 standard deviations: {within_3_std:.2f}%")
```

Slide 11: Testing for Normality

Before applying the Empirical Rule, it's crucial to test if your data follows a normal distribution. The Shapiro-Wilk test is a common method for this purpose.

```python
import numpy as np
from scipy import stats

# Generate two datasets: one normal, one non-normal
normal_data = np.random.normal(0, 1, 1000)
non_normal_data = np.random.exponential(1, 1000)

# Perform Shapiro-Wilk test
_, p_value_normal = stats.shapiro(normal_data)
_, p_value_non_normal = stats.shapiro(non_normal_data)

print("Normal data:")
print(f"p-value: {p_value_normal:.4f}")
print("Conclusion: Normal" if p_value_normal > 0.05 else "Conclusion: Not normal")

print("\nNon-normal data:")
print(f"p-value: {p_value_non_normal:.4f}")
print("Conclusion: Normal" if p_value_non_normal > 0.05 else "Conclusion: Not normal")
```

Slide 12: Applying the Empirical Rule to Non-normal Data

When dealing with non-normal data, we can use transformation techniques to make the data more normal-like, allowing for the application of the Empirical Rule.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate non-normal data (right-skewed)
data = np.random.lognormal(0, 0.5, 1000)

# Apply log transformation
transformed_data = np.log(data)

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(data, bins=30, density=True, alpha=0.7)
ax1.set_title("Original Data")
ax1.set_xlabel("Value")
ax1.set_ylabel("Frequency")

ax2.hist(transformed_data, bins=30, density=True, alpha=0.7)
ax2.set_title("Log-transformed Data")
ax2.set_xlabel("Log(Value)")
ax2.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Test for normality
_, p_value_original = stats.shapiro(data)
_, p_value_transformed = stats.shapiro(transformed_data)

print(f"Original data p-value: {p_value_original:.4f}")
print(f"Transformed data p-value: {p_value_transformed:.4f}")
```

Slide 13: Practical Applications of the Empirical Rule

The Empirical Rule finds applications in various fields, including education, psychology, and social sciences. Let's explore how it can be used to analyze exam scores.

```python
import numpy as np
from scipy import stats

# Sample exam scores
scores = np.array([65, 70, 75, 80, 85, 90, 95, 100, 85, 80, 75, 70, 85, 90, 80])

mean_score = np.mean(scores)
std_dev_score = np.std(scores)

# Calculate score ranges
one_std_range = (mean_score - std_dev_score, mean_score + std_dev_score)
two_std_range = (mean_score - 2*std_dev_score, mean_score + 2*std_dev_score)

print(f"Mean score: {mean_score:.2f}")
print(f"Standard deviation: {std_dev_score:.2f}")
print(f"68% of scores are likely between {one_std_range[0]:.2f} and {one_std_range[1]:.2f}")
print(f"95% of scores are likely between {two_std_range[0]:.2f} and {two_std_range[1]:.2f}")

# Calculate probability of scoring above 90
prob_above_90 = 1 - stats.norm.cdf((90 - mean_score) / std_dev_score)
print(f"Probability of scoring above 90: {prob_above_90:.2%}")
```

Slide 14: Conclusion and Key Takeaways

The Empirical Rule is a powerful tool for understanding and interpreting data in a normal distribution. It provides quick insights into data spread and probabilities. Remember its key points: 68% within one standard deviation, 95% within two, and 99.7% within three. While useful, always consider its limitations and test for normality before application.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

plt.figure(figsize=(12, 6))
plt.plot(x, y, 'k', linewidth=2)
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), color='red', alpha=0.3)
plt.fill_between(x, y, where=((x >= -2) & (x <= -1)) | ((x >= 1) & (x <= 2)), color='green', alpha=0.3)
plt.fill_between(x, y, where=((x >= -3) & (x <= -2)) | ((x >= 2) & (x <= 3)), color='blue', alpha=0.3)

plt.title("Empirical Rule Summary")
plt.xlabel("Standard Deviations from Mean")
plt.ylabel("Probability Density")
plt.legend(["Normal Distribution", "68%", "95%", "99.7%"])
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into the Empirical Rule and its applications in statistics, here are some valuable resources:

1. ArXiv.org paper: "On the Origins and Development of the Method of Least Squares" by Stephen M. Stigler URL: [https://arxiv.org/abs/1011.0923](https://arxiv.org/abs/1011.0923)
2. ArXiv.org paper: "Gaussianity tests of the COBE DMR data" by K. M. Górski et al. URL: [https://arxiv.org/abs/astro-ph/9403067](https://arxiv.org/abs/astro-ph/9403067)

These papers provide historical context and practical applications of normal distributions and related statistical concepts.

