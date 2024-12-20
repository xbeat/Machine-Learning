## Bernoulli Distribution Explained with Python
Slide 1: Introduction to Bernoulli Distribution

The Bernoulli distribution is a discrete probability distribution for a binary random variable. It models scenarios with two possible outcomes, often referred to as "success" and "failure." This distribution is named after Swiss mathematician Jacob Bernoulli and serves as a foundation for more complex probability distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_trial(p):
    return np.random.random() < p

# Simulate 1000 Bernoulli trials with p=0.6
p = 0.6
trials = 1000
results = [bernoulli_trial(p) for _ in range(trials)]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(results, 'o', markersize=2)
plt.title(f'Bernoulli Trials (p={p})')
plt.xlabel('Trial Number')
plt.ylabel('Outcome (1=Success, 0=Failure)')
plt.yticks([0, 1])
plt.show()
```

Slide 2: Probability Mass Function (PMF)

The Probability Mass Function (PMF) of a Bernoulli distribution gives the probability of each outcome. For a Bernoulli random variable X with parameter p:

P(X = 1) = p (success) P(X = 0) = 1 - p (failure)

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_pmf(x, p):
    return p if x == 1 else 1 - p

p = 0.6
x_values = [0, 1]
probabilities = [bernoulli_pmf(x, p) for x in x_values]

plt.figure(figsize=(8, 6))
plt.bar(x_values, probabilities)
plt.title(f'Bernoulli PMF (p={p})')
plt.xlabel('X')
plt.ylabel('Probability')
plt.xticks(x_values)
plt.ylim(0, 1)
for i, prob in enumerate(probabilities):
    plt.text(i, prob, f'{prob:.2f}', ha='center', va='bottom')
plt.show()
```

Slide 3: Expected Value and Variance

The expected value (mean) of a Bernoulli distribution is equal to the probability of success, p. The variance is p(1-p).

```python
import numpy as np

def bernoulli_stats(p):
    mean = p
    variance = p * (1 - p)
    std_dev = np.sqrt(variance)
    return mean, variance, std_dev

p = 0.6
mean, variance, std_dev = bernoulli_stats(p)

print(f"For Bernoulli(p={p}):")
print(f"Expected Value (Mean): {mean}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev:.4f}")
```

Slide 4: Generating Bernoulli Random Variables

We can generate Bernoulli random variables using NumPy's random module. This is useful for simulations and Monte Carlo methods.

```python
import numpy as np

def generate_bernoulli(p, size=1):
    return np.random.binomial(n=1, p=p, size=size)

# Generate 10 Bernoulli random variables with p=0.6
p = 0.6
samples = generate_bernoulli(p, size=10)

print(f"10 Bernoulli samples with p={p}:")
print(samples)

# Calculate the sample mean
sample_mean = np.mean(samples)
print(f"Sample mean: {sample_mean}")
print(f"True mean (p): {p}")
```

Slide 5: Bernoulli Trials and Binomial Distribution

A sequence of independent Bernoulli trials forms a Binomial distribution. The number of successes in n Bernoulli trials follows a Binomial(n, p) distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def bernoulli_to_binomial(p, n, num_experiments):
    results = np.random.binomial(n, p, num_experiments)
    return results

p = 0.6
n = 20
num_experiments = 10000

results = bernoulli_to_binomial(p, n, num_experiments)

plt.figure(figsize=(10, 6))
plt.hist(results, bins=range(n+2), density=True, alpha=0.7)
x = range(n+1)
plt.plot(x, binom.pmf(x, n, p), 'ro-', ms=8)
plt.title(f'Binomial Distribution from Bernoulli Trials (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()
```

Slide 6: Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a method to estimate the parameter p of a Bernoulli distribution from observed data.

```python
import numpy as np

def bernoulli_mle(data):
    return np.mean(data)

# Generate some sample data
true_p = 0.7
sample_size = 1000
data = np.random.binomial(n=1, p=true_p, size=sample_size)

# Estimate p using MLE
estimated_p = bernoulli_mle(data)

print(f"True p: {true_p}")
print(f"Estimated p (MLE): {estimated_p:.4f}")
print(f"Sample size: {sample_size}")
```

Slide 7: Confidence Intervals

We can construct confidence intervals for the parameter p using the normal approximation to the binomial distribution.

```python
import numpy as np
from scipy import stats

def bernoulli_confidence_interval(data, confidence=0.95):
    n = len(data)
    p_hat = np.mean(data)
    z = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z * np.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat - margin_of_error, p_hat + margin_of_error

# Generate sample data
true_p = 0.6
sample_size = 1000
data = np.random.binomial(n=1, p=true_p, size=sample_size)

# Calculate 95% confidence interval
ci_lower, ci_upper = bernoulli_confidence_interval(data)

print(f"True p: {true_p}")
print(f"Sample mean: {np.mean(data):.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
```

Slide 8: Hypothesis Testing

Hypothesis testing for a Bernoulli distribution often involves testing whether the probability of success is equal to a specific value.

```python
import numpy as np
from scipy import stats

def bernoulli_hypothesis_test(data, p0, alpha=0.05):
    n = len(data)
    p_hat = np.mean(data)
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value < alpha, p_value

# Generate sample data
true_p = 0.55
sample_size = 1000
data = np.random.binomial(n=1, p=true_p, size=sample_size)

# Test H0: p = 0.5 vs H1: p ≠ 0.5
reject_null, p_value = bernoulli_hypothesis_test(data, p0=0.5)

print(f"True p: {true_p}")
print(f"Sample mean: {np.mean(data):.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Reject null hypothesis: {reject_null}")
```

Slide 9: Real-Life Example: Quality Control

In a manufacturing process, each product is either defective (failure) or non-defective (success). The probability of producing a non-defective item is 0.95. We can model this as a Bernoulli distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_production(p_success, n_items):
    return np.random.binomial(n=1, p=p_success, size=n_items)

p_success = 0.95
n_items = 1000

production_run = simulate_production(p_success, n_items)
defect_rate = 1 - np.mean(production_run)

plt.figure(figsize=(10, 5))
plt.plot(production_run, 'o', markersize=2)
plt.title('Quality Control Simulation')
plt.xlabel('Item Number')
plt.ylabel('Quality (1=Good, 0=Defective)')
plt.yticks([0, 1])
plt.text(n_items/2, 0.5, f'Defect Rate: {defect_rate:.2%}', 
         horizontalalignment='center', verticalalignment='center')
plt.show()
```

Slide 10: Real-Life Example: A/B Testing

A/B testing in web design often uses Bernoulli trials to model user behavior. For instance, we might test if a new website design increases the click-through rate.

```python
import numpy as np
import matplotlib.pyplot as plt

def ab_test(p_a, p_b, n_samples):
    results_a = np.random.binomial(n=1, p=p_a, size=n_samples)
    results_b = np.random.binomial(n=1, p=p_b, size=n_samples)
    return results_a, results_b

p_a, p_b = 0.1, 0.15  # Click-through rates for designs A and B
n_samples = 1000

results_a, results_b = ab_test(p_a, p_b, n_samples)

plt.figure(figsize=(10, 6))
plt.hist([results_a.mean(), results_b.mean()], label=['Design A', 'Design B'])
plt.title('A/B Test Results: Click-through Rates')
plt.xlabel('Click-through Rate')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f"Design A click-through rate: {results_a.mean():.2%}")
print(f"Design B click-through rate: {results_b.mean():.2%}")
```

Slide 11: Bernoulli Distribution in Machine Learning

The Bernoulli distribution is fundamental in many machine learning algorithms, particularly in logistic regression and neural networks with binary outputs.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(1000, 1)
y = (1 / (1 + np.exp(-X - 0.5))).ravel() > 0.5

# Fit logistic regression
model = LogisticRegression()
model.fit(X, y)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='b', alpha=0.5)
X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict_proba(X_test)[:, 1]
plt.plot(X_test, y_pred, color='r', lw=2)
plt.title('Logistic Regression (Bernoulli Distribution)')
plt.xlabel('X')
plt.ylabel('Probability of Success')
plt.show()
```

Slide 12: Bernoulli Distribution in Information Theory

The Bernoulli distribution plays a crucial role in information theory, particularly in the concept of entropy. The entropy of a Bernoulli distribution measures the uncertainty of the outcome.

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

p_values = np.linspace(0, 1, 100)
entropies = [bernoulli_entropy(p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, entropies)
plt.title('Entropy of Bernoulli Distribution')
plt.xlabel('Probability of Success (p)')
plt.ylabel('Entropy (bits)')
plt.grid(True)
plt.show()

print(f"Maximum entropy: {max(entropies):.4f} bits")
print(f"Occurs at p = {p_values[np.argmax(entropies)]:.2f}")
```

Slide 13: Limitations and Extensions

While the Bernoulli distribution is simple and widely applicable, it has limitations. It can only model binary outcomes and assumes independence between trials. Extensions and related distributions include:

1. Binomial distribution (multiple Bernoulli trials)
2. Beta distribution (conjugate prior for Bernoulli)
3. Categorical distribution (multiple categories)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, beta, multinomial

# Bernoulli vs Binomial
n, p = 10, 0.3
x = np.arange(0, n+1)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.bar(x, binom.pmf(x, n, p))
plt.title('Binomial(10, 0.3)')

# Beta distribution
plt.subplot(132)
x = np.linspace(0, 1, 100)
for a, b in [(0.5, 0.5), (5, 1), (1, 3), (2, 2)]:
    plt.plot(x, beta.pdf(x, a, b), label=f'Beta({a},{b})')
plt.title('Beta Distributions')
plt.legend()

# Categorical distribution
plt.subplot(133)
probs = [0.1, 0.3, 0.4, 0.2]
x = np.arange(len(probs))
plt.bar(x, probs)
plt.title('Categorical Distribution')
plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For further exploration of the Bernoulli distribution and related concepts, consider the following resources:

1. "Probability Theory: The Logic of Science" by E. T. Jaynes (2003) ArXiv: [https://arxiv.org/abs/math/0312635](https://arxiv.org/abs/math/0312635)
2. "Information Theory, Inference, and Learning Algorithms" by David J.C. MacKay (2003) Available at: [http://www.inference.org.uk/mackay/itila/](http://www.inference.org.uk/mackay/itila/)
3. "Pattern Recognition and Machine Learning" by Christopher M. Bishop (2006)

These resources provide in-depth discussions of probability theory, including the Bernoulli distribution and its applications in various fields.

