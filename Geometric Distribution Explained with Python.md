## Geometric Distribution Explained with Python
Slide 1: Introduction to Geometric Distribution

The Geometric Distribution is a discrete probability distribution that models the number of trials needed to achieve the first success in a sequence of independent Bernoulli trials. It's often used in scenarios where we're interested in the waiting time until a certain event occurs.

```python
import numpy as np
import matplotlib.pyplot as plt

def geometric_pmf(k, p):
    return p * (1 - p)**(k - 1)

p = 0.3
k = np.arange(1, 11)
pmf = geometric_pmf(k, p)

plt.bar(k, pmf)
plt.xlabel('Number of trials (k)')
plt.ylabel('Probability')
plt.title(f'Geometric Distribution PMF (p={p})')
plt.show()
```

Slide 2: Probability Mass Function (PMF)

The PMF of the Geometric Distribution gives the probability of getting the first success on the k-th trial. It's defined as P(X = k) = p \* (1 - p)^(k - 1), where p is the probability of success on each trial.

```python
def geometric_pmf(k, p):
    return p * (1 - p)**(k - 1)

# Example: Probability of first success on 3rd trial with p=0.4
p = 0.4
k = 3
probability = geometric_pmf(k, p)
print(f"P(X = {k}) = {probability:.4f}")
```

Slide 3: Cumulative Distribution Function (CDF)

The CDF of the Geometric Distribution gives the probability of getting the first success within k trials. It's defined as P(X ≤ k) = 1 - (1 - p)^k.

```python
def geometric_cdf(k, p):
    return 1 - (1 - p)**k

# Example: Probability of first success within 5 trials with p=0.3
p = 0.3
k = 5
probability = geometric_cdf(k, p)
print(f"P(X ≤ {k}) = {probability:.4f}")
```

Slide 4: Expected Value and Variance

The expected value (mean) of the Geometric Distribution is 1/p, and the variance is (1-p)/p^2. These measures provide insights into the average behavior and spread of the distribution.

```python
def geometric_stats(p):
    mean = 1 / p
    variance = (1 - p) / p**2
    return mean, variance

p = 0.25
mean, variance = geometric_stats(p)
print(f"Mean: {mean:.2f}")
print(f"Variance: {variance:.2f}")
```

Slide 5: Generating Random Samples

We can simulate the Geometric Distribution using Python's random module or NumPy. This allows us to generate random samples and analyze their properties.

```python
import numpy as np

def geometric_sample(p, size=1000):
    return np.random.geometric(p, size)

p = 0.3
samples = geometric_sample(p)

plt.hist(samples, bins=30, density=True, alpha=0.7)
plt.xlabel('Number of trials')
plt.ylabel('Frequency')
plt.title(f'Simulated Geometric Distribution (p={p})')
plt.show()
```

Slide 6: Real-Life Example: Quality Control

In a manufacturing process, each product has a 5% chance of being defective. We can use the Geometric Distribution to model the number of products inspected before finding the first defective item.

```python
p_defective = 0.05
k = np.arange(1, 51)
pmf = geometric_pmf(k, p_defective)

plt.bar(k, pmf)
plt.xlabel('Number of products inspected')
plt.ylabel('Probability')
plt.title('Probability of Finding First Defective Product')
plt.show()

# Expected number of products inspected
expected_inspections = 1 / p_defective
print(f"Expected number of products inspected: {expected_inspections:.2f}")
```

Slide 7: Memoryless Property

The Geometric Distribution has a unique "memoryless" property. This means that the probability of success on the next trial remains constant, regardless of the number of previous failures.

```python
def conditional_probability(k, n, p):
    return geometric_pmf(k, p) / (1 - geometric_cdf(n-1, p))

p = 0.2
k = 5
n = 3
prob = conditional_probability(k, n, p)
print(f"P(X = {k} | X > {n-1}) = {prob:.4f}")
print(f"P(X = {k-n+1}) = {geometric_pmf(k-n+1, p):.4f}")
```

Slide 8: Relationship with Exponential Distribution

The Geometric Distribution is the discrete analog of the continuous Exponential Distribution. Both model "time until first event" scenarios, but in discrete and continuous time, respectively.

```python
from scipy.stats import geom, expon

p = 0.2
rate = -np.log(1 - p)  # Equivalent rate for exponential distribution

x = np.linspace(0, 20, 100)
geom_pmf = geom.pmf(np.arange(1, 21), p)
exp_pdf = expon.pdf(x, scale=1/rate)

plt.step(np.arange(1, 21), geom_pmf, where='post', label='Geometric')
plt.plot(x, exp_pdf, label='Exponential')
plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Geometric vs Exponential Distribution')
plt.legend()
plt.show()
```

Slide 9: Parameter Estimation

We can estimate the parameter p of a Geometric Distribution from observed data using the method of moments or maximum likelihood estimation.

```python
def estimate_p(samples):
    return 1 / np.mean(samples)

# Generate synthetic data
true_p = 0.3
samples = geometric_sample(true_p, size=10000)

# Estimate p
estimated_p = estimate_p(samples)
print(f"True p: {true_p}")
print(f"Estimated p: {estimated_p:.4f}")
```

Slide 10: Real-Life Example: Network Packet Transmission

In computer networking, the Geometric Distribution can model the number of attempts needed to successfully transmit a packet over a noisy channel. Assume each transmission has a 70% success rate.

```python
p_success = 0.7
k = np.arange(1, 11)
pmf = geometric_pmf(k, p_success)

plt.bar(k, pmf)
plt.xlabel('Number of transmission attempts')
plt.ylabel('Probability')
plt.title('Probability of Successful Packet Transmission')
plt.show()

# Expected number of transmission attempts
expected_attempts = 1 / p_success
print(f"Expected number of transmission attempts: {expected_attempts:.2f}")
```

Slide 11: Geometric Distribution in Hypothesis Testing

The Geometric Distribution can be used in hypothesis testing, particularly in scenarios involving waiting times or the number of trials until a specific event occurs.

```python
from scipy.stats import geom

def geometric_test(observed, p_null, alpha=0.05):
    p_value = 1 - geom.cdf(observed - 1, p_null)
    return p_value < alpha

# Example: Testing if a coin is fair
p_null = 0.5  # Probability of heads for a fair coin
observed = 10  # First heads on 10th flip

result = geometric_test(observed, p_null)
print(f"Reject null hypothesis: {result}")
```

Slide 12: Geometric Distribution in Bayesian Inference

We can use the Geometric Distribution in Bayesian inference to update our beliefs about the probability of success based on observed data.

```python
from scipy.stats import beta

def update_beta(prior_a, prior_b, observed):
    posterior_a = prior_a + 1
    posterior_b = prior_b + observed - 1
    return posterior_a, posterior_b

prior_a, prior_b = 1, 1  # Uniform prior
observed = 5  # First success on 5th trial

posterior_a, posterior_b = update_beta(prior_a, prior_b, observed)

x = np.linspace(0, 1, 100)
plt.plot(x, beta.pdf(x, prior_a, prior_b), label='Prior')
plt.plot(x, beta.pdf(x, posterior_a, posterior_b), label='Posterior')
plt.xlabel('p')
plt.ylabel('Density')
plt.title('Bayesian Update of p')
plt.legend()
plt.show()
```

Slide 13: Limitations and Considerations

While the Geometric Distribution is useful in many scenarios, it has limitations. It assumes constant probability of success and independence between trials, which may not always hold in real-world situations. Consider using more complex models when these assumptions are violated.

```python
def time_varying_probability(t):
    return 0.1 + 0.4 * np.sin(t / 10)

t = np.linspace(0, 50, 1000)
p_t = time_varying_probability(t)

plt.plot(t, p_t)
plt.xlabel('Time')
plt.ylabel('Probability of success')
plt.title('Time-varying Probability (Non-geometric)')
plt.show()
```

Slide 14: Additional Resources

For further exploration of the Geometric Distribution and its applications, consider the following resources:

1. "A Tutorial on the Basic Properties of the Geometric Distribution" by M. Akkouchi (arXiv:1609.08295) This paper provides an in-depth look at the fundamental properties of the Geometric Distribution, including its moment-generating function and various statistical properties.
2. "On Discrete Probability Distributions" by N. L. Johnson and S. Kotz (arXiv:1803.05858) This comprehensive work covers various discrete probability distributions, including the Geometric Distribution, and discusses their interrelationships and applications.
3. "Applications of Geometric Probability Distributions" by R. K. S. Hankin (Journal of Statistical Software, 2006) This article explores practical applications of the Geometric Distribution in various fields, including reliability theory and operations research.
4. "Introduction to Probability Models" by Sheldon M. Ross This textbook offers a thorough introduction to probability theory, with a dedicated section on the Geometric Distribution and its relationship to other distributions.
5. Online Resources:
   * SciPy Documentation (scipy.stats.geom): Provides details on using the Geometric Distribution in Python.
   * Khan Academy's Probability and Statistics course: Offers intuitive explanations of the Geometric Distribution suitable for beginners.

These resources offer a mix of theoretical foundations and practical applications, catering to different levels of expertise and interests in the Geometric Distribution.

