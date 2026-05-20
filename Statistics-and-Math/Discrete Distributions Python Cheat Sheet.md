## Discrete Distributions Python Cheat Sheet
Slide 1: Introduction to Discrete Distributions

Discrete distributions are probability distributions that describe random variables that can only take on specific, distinct values. These distributions are fundamental in statistics and probability theory, used to model various real-world phenomena where outcomes are countable or finite.

```python
# Example: Rolling a fair six-sided die
import random

def roll_die():
    return random.randint(1, 6)

# Simulate 1000 die rolls
rolls = [roll_die() for _ in range(1000)]

# Calculate probabilities
probabilities = {i: rolls.count(i) / 1000 for i in range(1, 7)}

print("Empirical probabilities:")
for outcome, prob in probabilities.items():
    print(f"P(X = {outcome}) ≈ {prob:.3f}")
```

Slide 2: Bernoulli Distribution

The Bernoulli distribution models a single trial with two possible outcomes: success (1) or failure (0). It is characterized by a single parameter p, which represents the probability of success.

```python
import random

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

# Simulate 1000 Bernoulli trials with p = 0.3
p = 0.3
trials = [bernoulli_trial(p) for _ in range(1000)]

success_rate = sum(trials) / len(trials)
print(f"Empirical success rate: {success_rate:.3f}")
print(f"Theoretical probability: {p:.3f}")
```

Slide 3: Binomial Distribution

The Binomial distribution models the number of successes in a fixed number of independent Bernoulli trials. It is characterized by two parameters: n (number of trials) and p (probability of success in each trial).

```python
import math

def binomial_pmf(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

n, p = 10, 0.3
for k in range(n + 1):
    prob = binomial_pmf(n, k, p)
    print(f"P(X = {k}) = {prob:.4f}")
```

Slide 4: Geometric Distribution

The Geometric distribution models the number of failures before the first success in a sequence of independent Bernoulli trials. It is characterized by a single parameter p, which is the probability of success on each trial.

```python
def geometric_pmf(k, p):
    return p * ((1 - p) ** k)

p = 0.2
for k in range(10):
    prob = geometric_pmf(k, p)
    print(f"P(X = {k}) = {prob:.4f}")
```

Slide 5: Negative Binomial Distribution

The Negative Binomial distribution models the number of failures before a specified number of successes occur in a sequence of independent Bernoulli trials. It is characterized by two parameters: r (number of successes) and p (probability of success on each trial).

```python
def negative_binomial_pmf(r, k, p):
    return math.comb(k + r - 1, k) * (p ** r) * ((1 - p) ** k)

r, p = 3, 0.4
for k in range(10):
    prob = negative_binomial_pmf(r, k, p)
    print(f"P(X = {k}) = {prob:.4f}")
```

Slide 6: Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space, given that these events occur with a known average rate and independently of each other. It is characterized by a single parameter λ (lambda), which represents the average number of events in the interval.

```python
import math

def poisson_pmf(k, lambda_param):
    return (math.exp(-lambda_param) * (lambda_param ** k)) / math.factorial(k)

lambda_param = 3
for k in range(10):
    prob = poisson_pmf(k, lambda_param)
    print(f"P(X = {k}) = {prob:.4f}")
```

Slide 7: Hypergeometric Distribution

The Hypergeometric distribution models the number of successes in a sample drawn without replacement from a finite population. It is characterized by three parameters: N (population size), K (number of successes in the population), and n (sample size).

```python
def hypergeometric_pmf(N, K, n, k):
    return (math.comb(K, k) * math.comb(N - K, n - k)) / math.comb(N, n)

N, K, n = 50, 20, 10
for k in range(min(n, K) + 1):
    prob = hypergeometric_pmf(N, K, n, k)
    print(f"P(X = {k}) = {prob:.4f}")
```

Slide 8: Discrete Uniform Distribution

The Discrete Uniform distribution models a finite set of equally likely outcomes. It is characterized by two parameters: a (minimum value) and b (maximum value).

```python
def discrete_uniform_pmf(a, b):
    return 1 / (b - a + 1)

a, b = 1, 6  # Example: fair six-sided die
prob = discrete_uniform_pmf(a, b)
for k in range(a, b + 1):
    print(f"P(X = {k}) = {prob:.4f}")
```

Slide 9: Real-life Example: Quality Control

Consider a manufacturing process where defects occur randomly and independently. The number of defects in a batch can be modeled using a Poisson distribution.

```python
import math
import random

def poisson_pmf(k, lambda_param):
    return (math.exp(-lambda_param) * (lambda_param ** k)) / math.factorial(k)

def simulate_defects(lambda_param, n_batches):
    return [sum(random.random() < lambda_param / 100 for _ in range(100)) for _ in range(n_batches)]

# Average defect rate: 2 per 100 items
lambda_param = 2
n_batches = 1000

defects = simulate_defects(lambda_param, n_batches)
observed_freq = {k: defects.count(k) / n_batches for k in range(max(defects) + 1)}

print("Defects | Observed Freq | Theoretical Prob")
for k in range(10):
    observed = observed_freq.get(k, 0)
    theoretical = poisson_pmf(k, lambda_param)
    print(f"{k:7d} | {observed:13.4f} | {theoretical:16.4f}")
```

Slide 10: Real-life Example: Customer Service

A call center receives calls according to a Poisson process. The time between consecutive calls follows an Exponential distribution, which is the continuous counterpart of the Geometric distribution.

```python
import random
import math

def exponential_random(lambda_param):
    return -math.log(1 - random.random()) / lambda_param

def simulate_call_center(lambda_param, duration):
    time = 0
    calls = []
    while time < duration:
        time += exponential_random(lambda_param)
        if time < duration:
            calls.append(time)
    return calls

# Simulate 8 hours (480 minutes) with an average of 4 calls per hour
lambda_param = 4 / 60  # 4 calls per hour = 1/15 calls per minute
duration = 480  # 8 hours in minutes

calls = simulate_call_center(lambda_param, duration)
print(f"Number of calls received: {len(calls)}")
print("Call times (in minutes):")
for i, call_time in enumerate(calls[:10], 1):
    print(f"Call {i}: {call_time:.2f}")

if len(calls) > 10:
    print("...")
```

Slide 11: Cumulative Distribution Function (CDF)

The Cumulative Distribution Function (CDF) of a discrete random variable X is the probability that X takes on a value less than or equal to x. It's useful for calculating probabilities over ranges of values.

```python
def binomial_pmf(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binomial_cdf(n, x, p):
    return sum(binomial_pmf(n, k, p) for k in range(x + 1))

n, p = 10, 0.3
print("x | P(X <= x)")
for x in range(n + 1):
    cdf = binomial_cdf(n, x, p)
    print(f"{x:2d} | {cdf:.4f}")
```

Slide 12: Expectation and Variance

The expectation (mean) and variance are important measures of central tendency and dispersion for discrete distributions. Here's how to calculate them for the Binomial distribution.

```python
def binomial_expectation(n, p):
    return n * p

def binomial_variance(n, p):
    return n * p * (1 - p)

n, p = 10, 0.3
expectation = binomial_expectation(n, p)
variance = binomial_variance(n, p)
std_dev = math.sqrt(variance)

print(f"Expectation: {expectation:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
```

Slide 13: Generating Random Variables

Generating random variables from discrete distributions is crucial for simulations and Monte Carlo methods. Here's an example of generating random variables from a Binomial distribution.

```python
import random

def binomial_random(n, p):
    return sum(random.random() < p for _ in range(n))

n, p = 10, 0.3
samples = [binomial_random(n, p) for _ in range(10000)]

observed_mean = sum(samples) / len(samples)
observed_variance = sum((x - observed_mean) ** 2 for x in samples) / len(samples)

print(f"Observed Mean: {observed_mean:.2f}")
print(f"Theoretical Mean: {n * p:.2f}")
print(f"Observed Variance: {observed_variance:.2f}")
print(f"Theoretical Variance: {n * p * (1 - p):.2f}")
```

Slide 14: Additional Resources

For further exploration of discrete distributions and their applications, consider the following resources:

1.  "A Survey of Discrete Probability Distributions" by M. H. DeGroot and M. J. Schervish (2012), ArXiv:1209.3404 \[math.ST\] URL: [https://arxiv.org/abs/1209.3404](https://arxiv.org/abs/1209.3404)
2.  "Probability Distributions: A Guide for Data Scientists" by J. Shao and D. Tu (2019), ArXiv:1909.13011 \[stat.ML\] URL: [https://arxiv.org/abs/1909.13011](https://arxiv.org/abs/1909.13011)

These papers provide comprehensive overviews of discrete probability distributions and their applications in various fields, including data science and machine learning.

