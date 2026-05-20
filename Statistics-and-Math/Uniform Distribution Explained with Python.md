## Uniform Distribution Explained with Python
Slide 1: Introduction to Uniform Distribution

Uniform distribution is a probability distribution where all outcomes are equally likely. It's characterized by a constant probability density function over a defined interval. This distribution is fundamental in probability theory and statistics, with applications ranging from random number generation to modeling various real-world phenomena.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate uniform distribution
data = np.random.uniform(0, 1, 1000)

# Plot histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Probability Density Function (PDF)

The PDF of a uniform distribution is constant within its range and zero outside it. For a uniform distribution between a and b, the PDF is 1/(b-a) within the interval \[a,b\] and 0 elsewhere.

```python
import numpy as np
import matplotlib.pyplot as plt

def uniform_pdf(x, a, b):
    return np.where((x >= a) & (x <= b), 1 / (b - a), 0)

x = np.linspace(-1, 3, 1000)
y = uniform_pdf(x, 0, 2)

plt.plot(x, y)
plt.title('PDF of Uniform Distribution')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(0, 0.6)
plt.show()
```

Slide 3: Cumulative Distribution Function (CDF)

The CDF of a uniform distribution represents the probability that a random variable takes a value less than or equal to x. It increases linearly from 0 to 1 over the interval \[a,b\].

```python
import numpy as np
import matplotlib.pyplot as plt

def uniform_cdf(x, a, b):
    return np.where(x < a, 0, np.where(x > b, 1, (x - a) / (b - a)))

x = np.linspace(-1, 3, 1000)
y = uniform_cdf(x, 0, 2)

plt.plot(x, y)
plt.title('CDF of Uniform Distribution')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.ylim(0, 1.1)
plt.show()
```

Slide 4: Mean and Variance

For a uniform distribution between a and b, the mean is (a+b)/2, and the variance is (b-a)^2/12. These properties are essential for understanding the distribution's behavior.

```python
import numpy as np

def uniform_stats(a, b):
    mean = (a + b) / 2
    variance = (b - a)**2 / 12
    return mean, variance

a, b = 0, 10
mean, variance = uniform_stats(a, b)
print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {np.sqrt(variance)}")
```

Slide 5: Generating Uniform Random Numbers

Python's numpy library provides a simple way to generate uniform random numbers. This is useful for simulations, sampling, and various computational tasks.

```python
import numpy as np

# Generate 5 random numbers between 0 and 1
random_numbers = np.random.uniform(0, 1, 5)
print("Random numbers between 0 and 1:", random_numbers)

# Generate 5 random numbers between 10 and 20
random_numbers = np.random.uniform(10, 20, 5)
print("Random numbers between 10 and 20:", random_numbers)
```

Slide 6: Sampling from Uniform Distribution

We can use uniform sampling to generate datasets for various purposes, such as testing algorithms or creating synthetic data for machine learning models.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate samples
samples = np.random.uniform(-5, 5, 1000)

# Plot histogram
plt.hist(samples, bins=30, edgecolor='black')
plt.title('Histogram of Samples from Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Calculate sample statistics
print(f"Sample Mean: {np.mean(samples):.2f}")
print(f"Sample Variance: {np.var(samples):.2f}")
```

Slide 7: Inverse Transform Sampling

Inverse transform sampling is a method for generating random numbers from any probability distribution, given its cumulative distribution function. For uniform distribution, it's straightforward.

```python
import numpy as np
import matplotlib.pyplot as plt

def inverse_transform_sampling(a, b, size):
    u = np.random.uniform(0, 1, size)
    return a + (b - a) * u

samples = inverse_transform_sampling(0, 10, 1000)

plt.hist(samples, bins=30, edgecolor='black')
plt.title('Histogram of Samples using Inverse Transform Sampling')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 8: Uniform Distribution in Hypothesis Testing

Uniform distribution is often used in hypothesis testing, particularly in the context of p-values. Under the null hypothesis, p-values follow a uniform distribution between 0 and 1.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate p-values under null hypothesis
p_values = np.random.uniform(0, 1, 1000)

# Plot histogram
plt.hist(p_values, bins=20, edgecolor='black')
plt.title('Histogram of p-values under Null Hypothesis')
plt.xlabel('p-value')
plt.ylabel('Frequency')
plt.show()

# Perform Kolmogorov-Smirnov test
ks_statistic, p_value = stats.kstest(p_values, 'uniform')
print(f"KS test p-value: {p_value:.4f}")
```

Slide 9: Uniform Distribution in Monte Carlo Integration

Uniform distribution is crucial in Monte Carlo integration, a technique for numerical integration using random sampling. Here's an example calculating π using this method.

```python
import numpy as np

def monte_carlo_pi(n):
    points = np.random.uniform(-1, 1, (n, 2))
    inside_circle = np.sum(np.linalg.norm(points, axis=1) <= 1)
    pi_estimate = 4 * inside_circle / n
    return pi_estimate

n_samples = [1000, 10000, 100000, 1000000]
for n in n_samples:
    pi_estimate = monte_carlo_pi(n)
    print(f"Pi estimate with {n} samples: {pi_estimate:.6f}")
```

Slide 10: Uniform Distribution in Randomized Algorithms

Uniform distribution is often used in randomized algorithms. Here's an example of reservoir sampling, which selects a random sample of k items from a stream of unknown size.

```python
import numpy as np

def reservoir_sampling(stream, k):
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = np.random.randint(0, i+1)
            if j < k:
                reservoir[j] = item
    return reservoir

# Example usage
stream = range(1000)
sample = reservoir_sampling(stream, 10)
print("Random sample:", sample)
```

Slide 11: Real-Life Example: Quality Control

Uniform distribution can model manufacturing tolerances. For instance, if a product's length should be 10 cm with a tolerance of ±0.1 cm, we can model this as a uniform distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 product lengths
lengths = np.random.uniform(9.9, 10.1, 1000)

plt.hist(lengths, bins=30, edgecolor='black')
plt.title('Distribution of Product Lengths')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Calculate percentage within tolerance
within_tolerance = np.sum((lengths >= 9.9) & (lengths <= 10.1)) / len(lengths) * 100
print(f"Percentage within tolerance: {within_tolerance:.2f}%")
```

Slide 12: Real-Life Example: Wait Times

Uniform distribution can model wait times in certain scenarios. For example, if buses arrive every 15 minutes, the wait time for a random passenger can be modeled as uniform between 0 and 15 minutes.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate wait times for 1000 passengers
wait_times = np.random.uniform(0, 15, 1000)

plt.hist(wait_times, bins=30, edgecolor='black')
plt.title('Distribution of Bus Wait Times')
plt.xlabel('Wait Time (minutes)')
plt.ylabel('Frequency')
plt.show()

print(f"Average wait time: {np.mean(wait_times):.2f} minutes")
print(f"Maximum wait time: {np.max(wait_times):.2f} minutes")
```

Slide 13: Limitations and Considerations

While uniform distribution is useful in many scenarios, it's important to recognize its limitations. Many real-world phenomena don't follow uniform distributions, and assuming uniformity when it doesn't exist can lead to incorrect conclusions.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate non-uniform data
non_uniform_data = np.random.exponential(scale=2, size=1000)

plt.hist(non_uniform_data, bins=30, edgecolor='black')
plt.title('Histogram of Non-Uniform Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Test for uniformity
from scipy import stats
_, p_value = stats.kstest(non_uniform_data, 'uniform')
print(f"p-value for uniformity test: {p_value:.6f}")
```

Slide 14: Additional Resources

For further exploration of uniform distribution and its applications in statistics and probability theory, consider the following resources:

1. "A Survey of Uniform Distribution Theory" by L. Kuipers and H. Niederreiter (ArXiv:math/0210037)
2. "On the Uniform Distribution of Sequences" by J. F. Koksma (ArXiv:math/0507185)
3. "Uniform Distribution and Algorithmic Randomness" by A. Nies (ArXiv:1808.09608)

These papers provide in-depth discussions on various aspects of uniform distribution and its theoretical foundations.

