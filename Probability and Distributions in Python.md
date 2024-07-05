## Probability and Distributions in Python

Slide 2: Introduction to Probability Probability is the mathematical study of the likelihood of events occurring. In Python, we can use various libraries and functions to work with probability concepts. The most commonly used library for this purpose is NumPy.

Slide 3: Random Numbers Before diving into probability and distributions, we need to understand how to generate random numbers in Python. The random module provides functions for generating random numbers. Code Example:

```python
import random

# Generate a random float between 0 and 1
random_float = random.random()
print(random_float)

# Generate a random integer between 1 and 6 (inclusive)
random_int = random.randint(1, 6)
print(random_int)
```

Slide 4: Discrete Probability Distributions A discrete probability distribution is a probability distribution that describes the likelihood of different possible outcomes for a random variable that can take on a countable number of values. In Python, we can use the math and statistics modules to work with discrete distributions. Code Example:

```python
import math

# Calculate the probability mass function (PMF) for a binomial distribution
n = 10  # Number of trials
p = 0.3  # Probability of success
k = 3  # Number of successes
pmf = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
print(f"Binomial PMF: {pmf}")
```

Slide 5: Continuous Probability Distributions A continuous probability distribution is a probability distribution that describes the likelihood of different possible outcomes for a random variable that can take on any value within a continuous range. In Python, we can use the scipy.stats module to work with continuous distributions. Code Example:

```python
import scipy.stats as stats

# Calculate the probability density function (PDF) for a normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation
x = 1.5  # Value to evaluate
pdf = stats.norm.pdf(x, mu, sigma)
print(f"Normal PDF at x={x}: {pdf}")
```

Slide 6: Central Limit Theorem The Central Limit Theorem states that the sum of many independent and identically distributed random variables tends toward a normal distribution, regardless of the underlying distribution. This theorem is fundamental in probability and statistics. Code Example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a sample of 10000 random numbers from a uniform distribution
sample = np.random.uniform(size=10000)

# Calculate the mean and standard deviation of the sample
sample_mean = np.mean(sample)
sample_std = np.std(sample)

# Plot the histogram of the sample
plt.hist(sample, bins=30, density=True)
plt.title("Histogram of Uniform Sample")
plt.show()
```

Slide 7: Sampling and Bootstrapping Sampling and bootstrapping are techniques used to estimate population parameters or test hypotheses based on a sample of data. In Python, we can use the random module and NumPy to perform sampling and bootstrapping. Code Example:

```python
import numpy as np

# Generate a population of 1000 random numbers
population = np.random.normal(loc=0, scale=1, size=1000)

# Take a simple random sample of size 100 from the population
sample = np.random.choice(population, size=100, replace=False)

# Perform bootstrapping to estimate the population mean
bootstrap_means = []
for _ in range(1000):
    bootstrap_sample = np.random.choice(sample, size=len(sample), replace=True)
    bootstrap_means.append(np.mean(bootstrap_sample))

print(f"Bootstrap estimate of population mean: {np.mean(bootstrap_means)}")
```

Slide 8: Hypothesis Testing Hypothesis testing is a statistical method used to make inferences about a population parameter based on a sample of data. In Python, we can use the scipy.stats module to perform hypothesis testing. Code Example:

```python
import scipy.stats as stats

# Generate two samples
sample1 = np.random.normal(loc=0, scale=1, size=100)
sample2 = np.random.normal(loc=0.5, scale=1, size=100)

# Perform a two-sample t-test
t_stat, p_val = stats.ttest_ind(sample1, sample2)

# Print the results
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_val}")
```

Slide 9: Confidence Intervals A confidence interval is a range of values that is likely to contain an unknown population parameter with a certain level of confidence. In Python, we can use the scipy.stats module to calculate confidence intervals. Code Example:

```python
import scipy.stats as stats

# Generate a sample
sample = np.random.normal(loc=0, scale=1, size=100)

# Calculate the 95% confidence interval for the population mean
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)
confidence_interval = stats.norm.interval(0.95, loc=sample_mean, scale=sample_std / np.sqrt(n))

print(f"95% Confidence Interval: {confidence_interval}")
```

Slide 10: Monte Carlo Simulation Monte Carlo simulation is a technique used to approximate the probability of different outcomes by running multiple trial runs, often using random sampling. In Python, we can use NumPy and other libraries to perform Monte Carlo simulations. Code Example:

```python
import numpy as np

# Define the function to be simulated
def func(x):
    return x ** 2 + np.random.normal(0, 1)

# Set up the simulation
num_simulations = 10000
x_values = np.linspace(-5, 5, 100)
results = np.zeros((len(x_values), num_simulations))

# Run the simulation
for i in range(num_simulations):
    results[:, i] = [func(x) for x in x_values]

# Calculate the mean and confidence intervals
means = np.mean(results, axis=1)
lower_bounds = np.percentile(results, 2.5, axis=1)
upper_bounds = np.percentile(results, 97.5, axis=1)

# Plot the results
plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.3)
plt.plot(x_values, means, label="Mean")
plt.legend()
plt.show()
```

Slide 11: Bayesian Statistics Bayesian statistics is a branch of statistics that uses Bayes' theorem to update the probabilities of hypotheses as more evidence or information becomes available. In Python, we can use libraries like PyMC3 to perform Bayesian analysis. Code Example:

```python
import pymc3 as pm

# Define the data
data = np.random.normal(loc=0, scale=1, size=100)

# Define the Bayesian model
with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

    # Run the MCMC sampler
    trace = pm.sample(1000, cores=2)

# Print the summary statistics
print(pm.summary(trace))
```

Slide 12: Conclusion In this presentation, we covered various concepts and techniques related to probability and distributions in Python. We explored random number generation, discrete and continuous probability distributions, the Central Limit Theorem, sampling and bootstrapping, hypothesis testing, confidence intervals, Monte Carlo simulations, and Bayesian statistics. Python provides powerful libraries and tools for working with probability and statistics, making it an excellent choice for data analysis and modeling tasks.

## Meta:
Mastering Probability and Distributions in Python

Unlock the power of probability and distributions in Python with our comprehensive TikTok series. From random number generation to Bayesian statistics, we'll guide you through key concepts and techniques, complete with code examples and clear explanations. Enhance your data analysis and modeling skills with this essential resource for Python enthusiasts and aspiring data scientists. Join us on this educational journey and elevate your Python proficiency to new heights.

Hashtags: #PythonTutorials #ProbabilityAndDistributions #DataScience #CodeExamples #LearningTikTok #InstitutionalContent

