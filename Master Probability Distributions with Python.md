## Master Probability Distributions with Python
Slide 1: Understanding Probability Distributions

Probability distributions are mathematical functions that describe the likelihood of different outcomes in a random event. They are fundamental to statistics and data science, helping us model uncertainty and make predictions. In this presentation, we'll explore key probability distributions and how to work with them using Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data
data = np.random.randn(1000)

# Plot histogram
plt.hist(data, bins=30, density=True)
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Normal Distribution

The normal distribution, also known as the Gaussian distribution, is a symmetric bell-shaped curve. It's widely used in natural and social sciences to represent real-valued random variables. In Python, we can generate and visualize a normal distribution using NumPy and Matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data points
x = np.linspace(-5, 5, 100)
y = stats.norm.pdf(x, 0, 1)

# Plot the distribution
plt.plot(x, y)
plt.title('Standard Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Slide 3: Uniform Distribution

The uniform distribution represents a constant probability over a specified range. It's often used in simulations and random number generation. Here's how to create and visualize a uniform distribution in Python:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate uniform random numbers
data = np.random.uniform(0, 1, 1000)

# Plot histogram
plt.hist(data, bins=30, density=True)
plt.title('Uniform Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 4: Binomial Distribution

The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials. It's commonly used in scenarios involving yes/no outcomes, such as coin flips or quality control. Let's simulate coin flips using the binomial distribution:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 10  # number of trials
p = 0.5  # probability of success

# Generate binomial distribution
x = np.arange(0, n+1)
y = binom.pmf(x, n, p)

# Plot
plt.bar(x, y)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()
```

Slide 5: Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space. It's often used in queueing theory, traffic flow, and rare event modeling. Here's an example of generating a Poisson distribution:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameter
lambda_param = 3  # average number of events

# Generate Poisson distribution
x = np.arange(0, 15)
y = poisson.pmf(x, lambda_param)

# Plot
plt.bar(x, y)
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.show()
```

Slide 6: Exponential Distribution

The exponential distribution models the time between events in a Poisson process. It's commonly used in reliability engineering and queueing theory. Let's create an exponential distribution and plot its probability density function:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Parameter
lambda_param = 0.5  # rate parameter

# Generate data points
x = np.linspace(0, 10, 100)
y = expon.pdf(x, scale=1/lambda_param)

# Plot
plt.plot(x, y)
plt.title(f'Exponential Distribution (λ={lambda_param})')
plt.xlabel('Time')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Slide 7: Real-life Example: Customer Arrivals

Let's model customer arrivals at a coffee shop using a Poisson distribution. Assume that on average, 20 customers arrive per hour. We'll simulate the number of arrivals for a 12-hour day:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_param = 20  # average arrivals per hour
hours = 12

# Simulate customer arrivals
arrivals = np.random.poisson(lambda_param, hours)

# Plot
plt.bar(range(1, hours+1), arrivals)
plt.title('Customer Arrivals at Coffee Shop')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Arrivals')
plt.show()

print(f"Total customers: {sum(arrivals)}")
```

Slide 8: Real-life Example: Manufacturing Quality Control

In a manufacturing process, we can use the binomial distribution to model the number of defective items in a batch. Let's simulate quality control for a production line where each item has a 5% chance of being defective:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 100  # items per batch
p = 0.05  # probability of defect
num_batches = 1000

# Simulate batches
defects = np.random.binomial(n, p, num_batches)

# Plot histogram
plt.hist(defects, bins=range(0, max(defects)+2), align='left', rwidth=0.8)
plt.title('Defective Items per Batch')
plt.xlabel('Number of Defective Items')
plt.ylabel('Frequency')
plt.show()

print(f"Average defects per batch: {np.mean(defects):.2f}")
```

Slide 9: Probability Distribution Fitting

Often, we need to determine which probability distribution best fits our data. SciPy provides tools for distribution fitting. Let's generate some random data and try to fit a distribution to it:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data (let's assume it's from a gamma distribution)
true_shape, true_scale = 2.0, 2.0
data = np.random.gamma(true_shape, true_scale, 1000)

# Fit a gamma distribution to the data
fitted_params = stats.gamma.fit(data)
fitted_shape, _, fitted_scale = fitted_params

# Plot the results
x = np.linspace(0, 20, 100)
plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')
plt.plot(x, stats.gamma.pdf(x, *fitted_params), 'r-', label='Fitted')
plt.title('Gamma Distribution Fitting')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"True shape: {true_shape}, Fitted shape: {fitted_shape:.2f}")
print(f"True scale: {true_scale}, Fitted scale: {fitted_scale:.2f}")
```

Slide 10: Multivariate Normal Distribution

The multivariate normal distribution is an extension of the one-dimensional normal distribution to higher dimensions. It's useful for modeling correlated random variables. Let's create and visualize a 2D multivariate normal distribution:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Parameters
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]

# Create grid and multivariate normal
x, y = np.mgrid[-3:3:.1, -3:3:.1]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)

# Plot
plt.contourf(x, y, rv.pdf(pos))
plt.title('2D Multivariate Normal Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.show()
```

Slide 11: Kernel Density Estimation

Kernel Density Estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable. It's useful when you don't know the underlying distribution of your data. Let's use KDE to estimate the distribution of some sample data:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
data = np.concatenate([np.random.normal(0, 1, 1000), 
                       np.random.normal(4, 1.5, 500)])

# Compute KDE
kde = stats.gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)

# Plot
plt.hist(data, bins=50, density=True, alpha=0.7, label='Data')
plt.plot(x_range, kde(x_range), label='KDE')
plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Slide 12: Cumulative Distribution Function (CDF)

The Cumulative Distribution Function (CDF) gives the probability that a random variable is less than or equal to a certain value. It's useful for calculating probabilities and quantiles. Let's plot the CDF of a normal distribution:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data points
x = np.linspace(-4, 4, 100)
y = stats.norm.cdf(x)

# Plot CDF
plt.plot(x, y)
plt.title('Cumulative Distribution Function (CDF) of Standard Normal')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

# Calculate probability P(X <= 1)
print(f"P(X <= 1) = {stats.norm.cdf(1):.4f}")
```

Slide 13: Monte Carlo Simulation

Monte Carlo simulations use repeated random sampling to solve problems that might be deterministic in principle. They're widely used in finance, physics, and engineering. Let's use a Monte Carlo simulation to estimate π:

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n):
    points_inside_circle = 0
    total_points = n
    
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    
    distances = np.sqrt(x**2 + y**2)
    points_inside_circle = np.sum(distances <= 1)
    
    pi_estimate = 4 * points_inside_circle / total_points
    return pi_estimate, x, y

n = 10000
pi_estimate, x, y = estimate_pi(n)

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=np.sqrt(x**2 + y**2) <= 1, cmap='coolwarm', alpha=0.5)
plt.title(f'Monte Carlo Pi Estimation\nEstimate: {pi_estimate:.4f}, True: {np.pi:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
```

Slide 14: Additional Resources

For further exploration of probability distributions and their applications in Python:

1. SciPy Documentation: Comprehensive guide on statistical functions and probability distributions. [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
2. "Probability Theory: The Logic of Science" by E. T. Jaynes: A foundational text on probability theory. ArXiv: [https://arxiv.org/abs/math/0312635](https://arxiv.org/abs/math/0312635)
3. "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani: Covers statistical learning methods with applications in R. (Note: While not on ArXiv, this is a widely recognized resource in the field)
4. "Probabilistic Programming & Bayesian Methods for Hackers" by Cameron Davidson-Pilon: A practical introduction to Bayesian methods and probabilistic programming. GitHub: [https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

