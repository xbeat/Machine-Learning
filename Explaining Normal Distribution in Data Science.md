## Explaining Normal Distribution in Data Science

Slide 1: What is Normal Distribution?

The Normal Distribution, also known as the Gaussian distribution, is a symmetrical, bell-shaped curve that represents the probability distribution of many natural phenomena. It's characterized by its mean (μ) and standard deviation (σ), which determine the center and spread of the distribution, respectively.

```python
import matplotlib.pyplot as plt

def normal_distribution(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

x = np.linspace(-5, 5, 1000)
y = normal_distribution(x, 0, 1)

plt.plot(x, y)
plt.title('Standard Normal Distribution (μ=0, σ=1)')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Slide 2: Properties of Normal Distribution

The Normal Distribution has several key properties that make it useful in data science. It's symmetric about the mean, the mean, median, and mode are all equal, and approximately 68%, 95%, and 99.7% of the data fall within one, two, and three standard deviations of the mean, respectively.

```python
import matplotlib.pyplot as plt

mu, sigma = 0, 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sigma**2))

plt.plot(x, y)
plt.fill_between(x, y, where=(x >= mu-sigma) & (x <= mu+sigma), alpha=0.3, label='68%')
plt.fill_between(x, y, where=(x >= mu-2*sigma) & (x <= mu+2*sigma), alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mu-3*sigma) & (x <= mu+3*sigma), alpha=0.1, label='99.7%')

plt.title('Standard Normal Distribution with Percentages')
plt.xlabel('Standard Deviations from Mean')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Central Limit Theorem

The Central Limit Theorem states that the distribution of sample means approximates a normal distribution as the sample size becomes larger, regardless of the population's distribution. This theorem is fundamental in statistics and explains why many natural phenomena follow a normal distribution.

```python
import matplotlib.pyplot as plt

def sample_means(population, sample_size, num_samples):
    return [np.mean(np.random.choice(population, sample_size)) for _ in range(num_samples)]

population = np.random.exponential(scale=1.0, size=10000)  # Non-normal population
sample_sizes = [1, 10, 30, 100]
num_samples = 1000

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Central Limit Theorem Demonstration')

for i, sample_size in enumerate(sample_sizes):
    means = sample_means(population, sample_size, num_samples)
    ax = axes[i//2, i%2]
    ax.hist(means, bins=30, density=True)
    ax.set_title(f'Sample Size: {sample_size}')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

Slide 4: Z-Score and Standardization

Z-score represents the number of standard deviations a data point is from the mean. Standardization transforms data to have a mean of 0 and a standard deviation of 1, allowing for easier comparison between different datasets.

```python
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=1000)

# Calculate z-scores
z_scores = (data - np.mean(data)) / np.std(data)

# Plot original data and z-scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(data, bins=30, edgecolor='black')
ax1.set_title('Original Data')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

ax2.hist(z_scores, bins=30, edgecolor='black')
ax2.set_title('Z-Scores (Standardized Data)')
ax2.set_xlabel('Z-Score')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(f"Original data - Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
print(f"Z-scores - Mean: {np.mean(z_scores):.2f}, Std: {np.std(z_scores):.2f}")
```

Slide 5: Probability Density Function (PDF)

The Probability Density Function (PDF) of a Normal Distribution gives the relative likelihood of a random variable taking on a particular value. It's represented by the familiar bell-shaped curve.

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))

x = np.linspace(-5, 5, 1000)
pdf_standard = norm.pdf(x, 0, 1)
pdf_custom = normal_pdf(x, 1, 1.5)

plt.plot(x, pdf_standard, label='Standard Normal (μ=0, σ=1)')
plt.plot(x, pdf_custom, label='Custom Normal (μ=1, σ=1.5)')
plt.title('Probability Density Functions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 6: Cumulative Distribution Function (CDF)

The Cumulative Distribution Function (CDF) of a Normal Distribution gives the probability that a random variable will take a value less than or equal to a given value. It's the integral of the PDF.

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
cdf_standard = norm.cdf(x, 0, 1)
cdf_custom = norm.cdf(x, 1, 1.5)

plt.plot(x, cdf_standard, label='Standard Normal (μ=0, σ=1)')
plt.plot(x, cdf_custom, label='Custom Normal (μ=1, σ=1.5)')
plt.title('Cumulative Distribution Functions')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 7: Generating Normal Distribution Data

In data science, we often need to generate normally distributed data for simulations or testing. Python's NumPy library provides functions to easily generate such data.

```python
import matplotlib.pyplot as plt

# Generate normally distributed data
np.random.seed(42)
data1 = np.random.normal(loc=0, scale=1, size=1000)
data2 = np.random.normal(loc=2, scale=1.5, size=1000)

# Plot histograms
plt.hist(data1, bins=30, alpha=0.7, label='μ=0, σ=1')
plt.hist(data2, bins=30, alpha=0.7, label='μ=2, σ=1.5')
plt.title('Generated Normal Distributions')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Print summary statistics
print("Data1 - Mean:", np.mean(data1), "Std:", np.std(data1))
print("Data2 - Mean:", np.mean(data2), "Std:", np.std(data2))
```

Slide 8: Checking for Normality

Before applying statistical methods that assume normality, it's crucial to check if your data follows a normal distribution. We can use visual methods like Q-Q plots or statistical tests like the Shapiro-Wilk test.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Generate data
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
non_normal_data = np.random.exponential(1, 1000)

# Q-Q plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(normal_data, dist="norm", plot=ax1)
ax1.set_title("Q-Q plot (Normal Data)")

stats.probplot(non_normal_data, dist="norm", plot=ax2)
ax2.set_title("Q-Q plot (Non-Normal Data)")

plt.tight_layout()
plt.show()

# Shapiro-Wilk test
print("Normal data - Shapiro-Wilk test:")
print(stats.shapiro(normal_data))

print("\nNon-normal data - Shapiro-Wilk test:")
print(stats.shapiro(non_normal_data))
```

Slide 9: Real-Life Example: Height Distribution

Human height is often approximated by a normal distribution. Let's simulate the height distribution of adults and analyze it using our knowledge of normal distributions.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Simulate height data (in cm)
np.random.seed(42)
male_heights = np.random.normal(175, 7, 1000)
female_heights = np.random.normal(162, 6, 1000)

# Plot histograms
plt.hist(male_heights, bins=30, alpha=0.7, label='Male')
plt.hist(female_heights, bins=30, alpha=0.7, label='Female')
plt.title('Height Distribution')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Calculate percentiles
male_percentiles = np.percentile(male_heights, [2.5, 50, 97.5])
female_percentiles = np.percentile(female_heights, [2.5, 50, 97.5])

print("Male height (cm) - 2.5th, 50th, 97.5th percentiles:", male_percentiles)
print("Female height (cm) - 2.5th, 50th, 97.5th percentiles:", female_percentiles)
```

Slide 10: Real-Life Example: Reaction Time Experiment

Reaction times in psychological experiments often follow a normal distribution. Let's simulate a reaction time experiment and analyze the results.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Simulate reaction time data (in milliseconds)
np.random.seed(42)
reaction_times = np.random.normal(250, 50, 1000)

# Plot histogram and fit a normal distribution
plt.hist(reaction_times, bins=30, density=True, alpha=0.7, color='skyblue')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(reaction_times), np.std(reaction_times))
plt.plot(x, p, 'k', linewidth=2)
plt.title('Reaction Time Distribution')
plt.xlabel('Reaction Time (ms)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Calculate summary statistics
print("Mean reaction time:", np.mean(reaction_times))
print("Standard deviation:", np.std(reaction_times))
print("Percentage within 1 standard deviation:", 
      np.sum((reaction_times >= np.mean(reaction_times) - np.std(reaction_times)) & 
             (reaction_times <= np.mean(reaction_times) + np.std(reaction_times))) / len(reaction_times) * 100)
```

Slide 11: Confidence Intervals

Confidence intervals provide a range of plausible values for a population parameter. For normally distributed data, we can easily calculate confidence intervals using the mean and standard error.

```python
from scipy import stats

# Generate sample data
np.random.seed(42)
sample = np.random.normal(100, 15, 100)

# Calculate 95% confidence interval
confidence_level = 0.95
degrees_freedom = len(sample) - 1
sample_mean = np.mean(sample)
sample_standard_error = stats.sem(sample)
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)

print(f"{confidence_level*100}% Confidence Interval: {confidence_interval}")

# Visualize the confidence interval
import matplotlib.pyplot as plt

plt.hist(sample, bins=20, density=True, alpha=0.7, color='skyblue')
plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(confidence_interval[0], color='green', linestyle='dotted', linewidth=2, label='Confidence Interval')
plt.axvline(confidence_interval[1], color='green', linestyle='dotted', linewidth=2)
plt.title('Sample Distribution with Confidence Interval')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Hypothesis Testing

Hypothesis testing is a crucial application of normal distribution in data science. It allows us to make inferences about population parameters based on sample data. Let's perform a one-sample t-test as an example.

```python
from scipy import stats

# Generate sample data
np.random.seed(42)
sample = np.random.normal(102, 5, 100)  # Sample with mean 102 and std 5

# Perform one-sample t-test
hypothesized_mean = 100
t_statistic, p_value = stats.ttest_1samp(sample, hypothesized_mean)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Visualize the results
import matplotlib.pyplot as plt

plt.hist(sample, bins=20, density=True, alpha=0.7, color='skyblue')
plt.axvline(np.mean(sample), color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(hypothesized_mean, color='green', linestyle='dotted', linewidth=2, label='Hypothesized Mean')
plt.title('Sample Distribution with Hypothesis Test')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

Slide 13: Normal Distribution in Machine Learning

The normal distribution plays a crucial role in many machine learning algorithms. For example, in linear regression, we often assume that the residuals are normally distributed. Let's visualize this assumption using a simple linear regression model.

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.normal(0, 1, (100, 1))

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate residuals
residuals = y - model.predict(X)

# Plot residuals
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.scatter(X, residuals)
plt.title('Residuals vs. Predictor')
plt.xlabel('X')
plt.ylabel('Residuals')

plt.subplot(122)
stats.probplot(residuals.ravel(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

# Test for normality of residuals
_, p_value = stats.normaltest(residuals)
print(f"p-value for normality test: {p_value[0]:.4f}")
```

Slide 14: Limitations and Alternatives

While the normal distribution is widely applicable, it's not always appropriate. Some data may be skewed or have heavy tails. In such cases, we might consider alternative distributions or transformations.

```python
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate data from different distributions
normal_data = np.random.normal(0, 1, 1000)
skewed_data = np.random.lognormal(0, 0.5, 1000)
heavy_tailed_data = np.random.standard_t(3, 1000)

# Plot histograms
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.hist(normal_data, bins=30, density=True)
plt.title('Normal Distribution')

plt.subplot(132)
plt.hist(skewed_data, bins=30, density=True)
plt.title('Skewed Distribution')

plt.subplot(133)
plt.hist(heavy_tailed_data, bins=30, density=True)
plt.title('Heavy-tailed Distribution')

plt.tight_layout()
plt.show()

# Test for normality
for data, name in zip([normal_data, skewed_data, heavy_tailed_data], 
                      ['Normal', 'Skewed', 'Heavy-tailed']):
    _, p_value = stats.normaltest(data)
    print(f"{name} data - p-value for normality test: {p_value:.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into the normal distribution and its applications in data science, here are some valuable resources:

1. "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman ArXiv: [https://arxiv.org/abs/1011.0885](https://arxiv.org/abs/1011.0885)
2. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani ArXiv: [https://arxiv.org/abs/1501.07274](https://arxiv.org/abs/1501.07274)
3. "Probability Theory: The Logic of Science" by E.T. Jaynes ArXiv: [https://arxiv.org/abs/math/0312635](https://arxiv.org/abs/math/0312635)

These resources provide in-depth coverage of probability theory, statistical learning, and the role of the normal distribution in various data science applications.


