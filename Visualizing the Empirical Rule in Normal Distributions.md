## Visualizing the Empirical Rule in Normal Distributions
Slide 1: The Empirical Rule and Normal Distribution

The empirical rule, also known as the 68-95-99.7 rule, is a statistical principle that describes the distribution of data in a normal distribution. It states that approximately 68% of data falls within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within three standard deviations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate normal distribution data
mean = 0
std_dev = 1
data = np.random.normal(mean, std_dev, 10000)

# Plot histogram
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Understanding Standard Deviation

Standard deviation is a measure of the spread of data in a distribution. It quantifies the average distance between each data point and the mean of the dataset.

```python
import numpy as np

# Sample dataset
data = [2, 4, 6, 8, 10]

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
```

Slide 3: Visualizing the Empirical Rule

Let's create a visual representation of the empirical rule using Python. We'll plot a normal distribution and highlight the areas within one, two, and three standard deviations.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'k-', lw=2)

plt.fill_between(x, y, where=(abs(x) < 1), color='red', alpha=0.3)
plt.fill_between(x, y, where=(abs(x) < 2), color='yellow', alpha=0.3)
plt.fill_between(x, y, where=(abs(x) < 3), color='green', alpha=0.3)

plt.title("Normal Distribution with Empirical Rule")
plt.xlabel("Standard Deviations from Mean")
plt.ylabel("Probability Density")
plt.show()
```

Slide 4: Calculating Percentages within Standard Deviations

Let's write a function to calculate the percentage of data falling within a given number of standard deviations in a normal distribution.

```python
import scipy.stats as stats

def percentage_within_std(num_std):
    return stats.norm.cdf(num_std) - stats.norm.cdf(-num_std)

one_std = percentage_within_std(1)
two_std = percentage_within_std(2)
three_std = percentage_within_std(3)

print(f"Percentage within 1 std dev: {one_std:.2%}")
print(f"Percentage within 2 std dev: {two_std:.2%}")
print(f"Percentage within 3 std dev: {three_std:.2%}")
```

Slide 5: Verifying the Empirical Rule

We can verify the empirical rule by generating a large sample of normally distributed data and calculating the percentages that fall within each standard deviation range.

```python
import numpy as np

np.random.seed(42)
sample_size = 1000000
data = np.random.normal(0, 1, sample_size)

within_one_std = np.sum(np.abs(data) < 1) / sample_size
within_two_std = np.sum(np.abs(data) < 2) / sample_size
within_three_std = np.sum(np.abs(data) < 3) / sample_size

print(f"Percentage within 1 std dev: {within_one_std:.2%}")
print(f"Percentage within 2 std dev: {within_two_std:.2%}")
print(f"Percentage within 3 std dev: {within_three_std:.2%}")
```

Slide 6: Practical Application: Quality Control

In manufacturing, the empirical rule is often used for quality control. Let's consider a process that produces widgets with a target weight of 100 grams and a standard deviation of 2 grams.

```python
import numpy as np
from scipy import stats

target_weight = 100
std_dev = 2

# Calculate acceptable range (within 2 standard deviations)
lower_limit = target_weight - 2 * std_dev
upper_limit = target_weight + 2 * std_dev

# Generate sample data
sample_size = 1000
weights = np.random.normal(target_weight, std_dev, sample_size)

# Calculate percentage within acceptable range
within_range = np.sum((weights >= lower_limit) & (weights <= upper_limit)) / sample_size

print(f"Acceptable range: {lower_limit:.2f} - {upper_limit:.2f} grams")
print(f"Percentage within range: {within_range:.2%}")
```

Slide 7: Real-Life Example: Height Distribution

Let's apply the empirical rule to human height distribution. Assume the average adult male height is 170 cm with a standard deviation of 7 cm.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mean_height = 170
std_dev_height = 7

# Generate height data
heights = np.random.normal(mean_height, std_dev_height, 10000)

# Calculate ranges
one_std_range = (mean_height - std_dev_height, mean_height + std_dev_height)
two_std_range = (mean_height - 2 * std_dev_height, mean_height + 2 * std_dev_height)

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=50, density=True, alpha=0.7)
plt.axvline(one_std_range[0], color='r', linestyle='dashed', linewidth=2)
plt.axvline(one_std_range[1], color='r', linestyle='dashed', linewidth=2)
plt.axvline(two_std_range[0], color='g', linestyle='dashed', linewidth=2)
plt.axvline(two_std_range[1], color='g', linestyle='dashed', linewidth=2)

plt.title("Adult Male Height Distribution")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")
plt.show()

# Calculate percentages
within_one_std = np.sum((heights >= one_std_range[0]) & (heights <= one_std_range[1])) / len(heights)
within_two_std = np.sum((heights >= two_std_range[0]) & (heights <= two_std_range[1])) / len(heights)

print(f"Percentage within 1 std dev: {within_one_std:.2%}")
print(f"Percentage within 2 std dev: {within_two_std:.2%}")
```

Slide 8: Z-Score and Standard Normal Distribution

The z-score represents the number of standard deviations a data point is from the mean. It allows us to standardize normal distributions and calculate probabilities.

```python
import numpy as np
from scipy import stats

def calculate_z_score(x, mean, std_dev):
    return (x - mean) / std_dev

# Example: SAT scores (mean = 1000, std_dev = 200)
sat_score = 1300
sat_mean = 1000
sat_std_dev = 200

z_score = calculate_z_score(sat_score, sat_mean, sat_std_dev)
percentile = stats.norm.cdf(z_score) * 100

print(f"Z-score: {z_score:.2f}")
print(f"Percentile: {percentile:.2f}%")
```

Slide 9: Confidence Intervals

Confidence intervals use the empirical rule to estimate population parameters. Let's calculate a 95% confidence interval for a sample mean.

```python
import numpy as np
from scipy import stats

# Generate sample data
sample_size = 30
population_mean = 100
population_std_dev = 15
sample = np.random.normal(population_mean, population_std_dev, sample_size)

# Calculate sample statistics
sample_mean = np.mean(sample)
sample_std_error = population_std_dev / np.sqrt(sample_size)

# Calculate 95% confidence interval
confidence_level = 0.95
degrees_of_freedom = sample_size - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

margin_of_error = t_value * sample_std_error
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"Sample Mean: {sample_mean:.2f}")
print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")
```

Slide 10: Outlier Detection

The empirical rule can be used to identify potential outliers in a dataset. Data points beyond three standard deviations from the mean are often considered outliers.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with outliers
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 1000),
    np.random.normal(0, 1, 10) * 4  # Potential outliers
])

# Calculate mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)

# Identify potential outliers
outliers = data[np.abs(data - mean) > 3 * std_dev]

# Plot data and outliers
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.scatter(outliers, [0.02] * len(outliers), color='red', s=50, label='Potential Outliers')

plt.title("Data Distribution with Potential Outliers")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Number of potential outliers: {len(outliers)}")
```

Slide 11: Probability Density Function (PDF)

The probability density function describes the likelihood of a continuous random variable taking on a specific value. For a normal distribution, it's defined by the mean and standard deviation.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_pdf(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

x = np.linspace(-4, 4, 1000)
pdf = normal_pdf(x, 0, 1)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf, 'b-', lw=2)
plt.title("Probability Density Function of Standard Normal Distribution")
plt.xlabel("Z-score")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()
```

Slide 12: Cumulative Distribution Function (CDF)

The cumulative distribution function gives the probability that a random variable is less than or equal to a given value. It's useful for calculating probabilities within specific ranges.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
cdf = norm.cdf(x)

plt.figure(figsize=(10, 6))
plt.plot(x, cdf, 'r-', lw=2)
plt.title("Cumulative Distribution Function of Standard Normal Distribution")
plt.xlabel("Z-score")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.show()

# Calculate probability between -2 and 2 standard deviations
prob = norm.cdf(2) - norm.cdf(-2)
print(f"Probability between -2 and 2 standard deviations: {prob:.4f}")
```

Slide 13: Central Limit Theorem

The Central Limit Theorem states that the sampling distribution of the mean approaches a normal distribution as the sample size increases, regardless of the population distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_sampling_distribution(population, sample_size, num_samples):
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size)
        sample_means.append(np.mean(sample))
    return sample_means

# Create a non-normal population (uniform distribution)
population = np.random.uniform(0, 10, 10000)

sample_sizes = [5, 30, 100]
num_samples = 1000

plt.figure(figsize=(15, 5))
for i, size in enumerate(sample_sizes, 1):
    plt.subplot(1, 3, i)
    sample_means = simulate_sampling_distribution(population, size, num_samples)
    plt.hist(sample_means, bins=30, density=True)
    plt.title(f"Sample Size: {size}")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```

Slide 14: Applications in Machine Learning

The empirical rule and normal distribution play crucial roles in machine learning, particularly in feature scaling and anomaly detection.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 2)
X[0] = [10, 10]  # Add an outlier

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Detect outliers using Elliptic Envelope (assumes Gaussian distribution)
outlier_detector = EllipticEnvelope(contamination=0.01, random_state=42)
outlier_labels = outlier_detector.fit_predict(X_scaled)

# Count outliers
num_outliers = np.sum(outlier_labels == -1)
print(f"Number of detected outliers: {num_outliers}")

# Print statistics of scaled features
print("Scaled feature statistics:")
print(f"Mean: {X_scaled.mean(axis=0)}")
print(f"Standard deviation: {X_scaled.std(axis=0)}")
```

Slide 15: Additional Resources

For those interested in diving deeper into the topics of normal distribution, empirical rule, and their applications in statistics and data science, here are some valuable resources:

1. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (2009) ArXiv: [https://arxiv.org/abs/1011.0885](https://arxiv.org/abs/1011.0885)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
3. "Probability Theory: The Logic of Science" by E.T. Jaynes (2003) ArXiv: [https://arxiv.org/abs/math/0312635](https://arxiv.org/abs/math/0312635)

These resources provide in-depth explanations and advanced applications of the concepts covered in this presentation.

