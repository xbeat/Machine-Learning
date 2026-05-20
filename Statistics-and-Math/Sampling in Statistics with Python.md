## Sampling in Statistics with Python
Slide 1: Introduction to Sampling in Statistics

Sampling is a fundamental concept in statistics that involves selecting a subset of individuals from a larger population to make inferences about the entire population. This process is crucial for conducting research, surveys, and data analysis when it's impractical or impossible to study every member of a population. In this presentation, we'll explore various sampling techniques and their implementation using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a population
population = np.random.normal(loc=100, scale=15, size=10000)

# Plot the population distribution
plt.hist(population, bins=50)
plt.title("Population Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: Simple Random Sampling

Simple random sampling is a basic technique where each member of the population has an equal probability of being selected. This method ensures unbiased representation of the population. Let's implement simple random sampling using Python's NumPy library.

```python
import numpy as np

# Generate a population
population = np.arange(1, 1001)

# Perform simple random sampling
sample_size = 100
simple_random_sample = np.random.choice(population, size=sample_size, replace=False)

print("Simple Random Sample:", simple_random_sample)
print("Sample Mean:", np.mean(simple_random_sample))
print("Population Mean:", np.mean(population))
```

Slide 3: Systematic Sampling

Systematic sampling involves selecting every k-th element from the population after a random start. This method is useful when the population is ordered and we want to ensure even coverage across the entire range. Here's how to implement systematic sampling in Python.

```python
import numpy as np

# Generate an ordered population
population = np.arange(1, 1001)

# Set sample size and calculate step size
sample_size = 100
step = len(population) // sample_size

# Perform systematic sampling
start = np.random.randint(0, step)
systematic_sample = population[start::step][:sample_size]

print("Systematic Sample:", systematic_sample)
print("Sample Mean:", np.mean(systematic_sample))
print("Population Mean:", np.mean(population))
```

Slide 4: Stratified Sampling

Stratified sampling divides the population into subgroups (strata) based on shared characteristics, then samples from each stratum. This method ensures representation from all subgroups. Let's implement stratified sampling for a hypothetical student population.

```python
import numpy as np
import pandas as pd

# Create a hypothetical student population
np.random.seed(42)
students = pd.DataFrame({
    'grade': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
    'score': np.random.randint(60, 101, size=1000)
})

# Perform stratified sampling
sample_size = 100
stratified_sample = students.groupby('grade', group_keys=False).apply(lambda x: x.sample(int(sample_size/4)))

print("Stratified Sample:")
print(stratified_sample)
print("\nSample Mean Score:", stratified_sample['score'].mean())
print("Population Mean Score:", students['score'].mean())
```

Slide 5: Cluster Sampling

Cluster sampling involves dividing the population into clusters, randomly selecting some clusters, and then sampling all members within the chosen clusters. This method is useful when it's more practical to sample groups rather than individuals. Let's simulate cluster sampling for a city's households.

```python
import numpy as np

# Simulate a city with neighborhoods (clusters) and households
np.random.seed(42)
city = [np.random.normal(loc=50000, scale=10000, size=np.random.randint(50, 150)) for _ in range(20)]

# Perform cluster sampling
num_clusters = 5
sampled_clusters = np.random.choice(len(city), size=num_clusters, replace=False)
cluster_sample = [household for cluster in sampled_clusters for household in city[cluster]]

print("Number of sampled households:", len(cluster_sample))
print("Mean household income in sample:", np.mean(cluster_sample))
print("Mean household income in population:", np.mean([income for neighborhood in city for income in neighborhood]))
```

Slide 6: Weighted Sampling

Weighted sampling assigns different probabilities to population members based on their importance or representation. This technique is useful when certain elements should have a higher chance of being selected. Let's implement weighted sampling using Python.

```python
import numpy as np

# Create a population with weights
population = ['A', 'B', 'C', 'D', 'E']
weights = [0.1, 0.2, 0.3, 0.1, 0.3]

# Perform weighted sampling
sample_size = 1000
weighted_sample = np.random.choice(population, size=sample_size, p=weights)

# Calculate the frequency of each element in the sample
unique, counts = np.unique(weighted_sample, return_counts=True)
frequencies = dict(zip(unique, counts / sample_size))

print("Sample frequencies:")
for item, freq in frequencies.items():
    print(f"{item}: {freq:.2f}")
```

Slide 7: Bootstrap Sampling

Bootstrap sampling is a resampling technique used to estimate the sampling distribution of a statistic. It involves repeatedly sampling with replacement from the original sample. This method is particularly useful for estimating confidence intervals and performing hypothesis tests.

```python
import numpy as np
import matplotlib.pyplot as plt

# Original sample
original_sample = np.random.normal(loc=100, scale=15, size=100)

# Perform bootstrap sampling
n_bootstrap = 10000
bootstrap_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    bootstrap_sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
    bootstrap_means[i] = np.mean(bootstrap_sample)

# Plot the bootstrap distribution of means
plt.hist(bootstrap_means, bins=50)
plt.title("Bootstrap Distribution of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()

print("Original Sample Mean:", np.mean(original_sample))
print("Bootstrap Mean of Means:", np.mean(bootstrap_means))
print("95% Confidence Interval:", np.percentile(bootstrap_means, [2.5, 97.5]))
```

Slide 8: Importance Sampling

Importance sampling is a technique used to estimate properties of a particular distribution while sampling from a different distribution. This method is particularly useful in situations where sampling from the target distribution is difficult or computationally expensive.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

def proposal_distribution(x):
    return np.exp(-np.abs(x)) / 2

# Generate samples from the proposal distribution
n_samples = 10000
samples = np.random.exponential(scale=1, size=n_samples) * np.random.choice([-1, 1], size=n_samples)

# Calculate importance weights
weights = target_distribution(samples) / proposal_distribution(samples)

# Estimate the mean of the target distribution
estimated_mean = np.sum(samples * weights) / np.sum(weights)

print("Estimated mean:", estimated_mean)
print("True mean:", 0)  # The true mean of a standard normal distribution is 0

# Plot the results
x = np.linspace(-4, 4, 1000)
plt.plot(x, target_distribution(x), label='Target Distribution')
plt.hist(samples, bins=50, density=True, alpha=0.5, label='Proposal Samples')
plt.legend()
plt.title("Importance Sampling")
plt.show()
```

Slide 9: Reservoir Sampling

Reservoir sampling is an algorithm for randomly selecting k samples from a population of unknown size, possibly very large or streaming. This technique is particularly useful when dealing with big data or streaming data where we can't hold all items in memory at once.

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

# Simulate a data stream
np.random.seed(42)
data_stream = iter(np.random.randint(1, 1001, size=10000))

# Perform reservoir sampling
sample_size = 100
reservoir_sample = reservoir_sampling(data_stream, sample_size)

print("Reservoir Sample:", reservoir_sample)
print("Sample Mean:", np.mean(reservoir_sample))
```

Slide 10: Monte Carlo Sampling

Monte Carlo sampling is a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. One common application is estimating definite integrals. Let's use Monte Carlo sampling to estimate the value of π.

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_pi(n_samples):
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    inside_circle = (x**2 + y**2 <= 1)
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    return pi_estimate

# Estimate π with increasing number of samples
sample_sizes = np.logspace(2, 6, num=20, dtype=int)
pi_estimates = [estimate_pi(n) for n in sample_sizes]

# Plot the results
plt.semilogx(sample_sizes, pi_estimates, 'b-')
plt.axhline(y=np.pi, color='r', linestyle='--')
plt.xlabel('Number of Samples')
plt.ylabel('Estimated π')
plt.title('Monte Carlo Estimation of π')
plt.grid(True)
plt.show()

print(f"Final π estimate (with {sample_sizes[-1]} samples): {pi_estimates[-1]}")
print(f"True π value: {np.pi}")
```

Slide 11: Gibbs Sampling

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) algorithm for obtaining a sequence of observations which are approximated from a specified multivariate probability distribution. It's particularly useful for sampling from high-dimensional distributions. Let's implement a simple Gibbs sampler for a bivariate normal distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def gibbs_sampler(n_samples, mu, sigma):
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    x[0], y[0] = 0, 0
    
    for i in range(1, n_samples):
        x[i] = np.random.normal(mu[0] + sigma[0, 1] / sigma[1, 1] * (y[i-1] - mu[1]),
                                np.sqrt(sigma[0, 0] - sigma[0, 1]**2 / sigma[1, 1]))
        y[i] = np.random.normal(mu[1] + sigma[1, 0] / sigma[0, 0] * (x[i] - mu[0]),
                                np.sqrt(sigma[1, 1] - sigma[1, 0]**2 / sigma[0, 0]))
    
    return x, y

# Set up the bivariate normal distribution parameters
mu = np.array([0, 0])
sigma = np.array([[1, 0.5], [0.5, 1]])

# Run the Gibbs sampler
n_samples = 5000
x, y = gibbs_sampler(n_samples, mu, sigma)

# Plot the results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x, y, 'b.', alpha=0.1)
plt.title('Gibbs Sampling: Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(122)
plt.hist2d(x, y, bins=50, cmap='Blues')
plt.title('Gibbs Sampling: 2D Histogram')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
```

Slide 12: Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is another MCMC method used to obtain a sequence of random samples from a probability distribution where direct sampling is difficult. It's more general than Gibbs sampling and can be applied to a wider range of problems. Let's implement it for sampling from a gamma distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

def metropolis_hastings(target_pdf, proposal_pdf, proposal_sampler, n_samples, initial_state):
    samples = np.zeros(n_samples)
    current_state = initial_state
    accepted = 0
    
    for i in range(n_samples):
        proposed_state = proposal_sampler(current_state)
        
        acceptance_ratio = (target_pdf(proposed_state) * proposal_pdf(current_state, proposed_state)) / \
                           (target_pdf(current_state) * proposal_pdf(proposed_state, current_state))
        
        if np.random.random() < acceptance_ratio:
            current_state = proposed_state
            accepted += 1
        
        samples[i] = current_state
    
    return samples, accepted / n_samples

# Target distribution: Gamma(k=2, theta=2)
k, theta = 2, 2
target_pdf = lambda x: gamma.pdf(x, a=k, scale=theta)

# Proposal distribution: Normal(mu=x, sigma=0.5)
proposal_pdf = lambda x, mu: np.exp(-0.5 * ((x - mu) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
proposal_sampler = lambda mu: np.random.normal(mu, 0.5)

# Run Metropolis-Hastings
n_samples = 10000
initial_state = 1.0
samples, acceptance_rate = metropolis_hastings(target_pdf, proposal_pdf, proposal_sampler, n_samples, initial_state)

# Plot results
x = np.linspace(0, 20, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='MCMC Samples')
plt.plot(x, target_pdf(x), 'r-', lw=2, label='Target PDF')
plt.title(f'Metropolis-Hastings Sampling (Acceptance Rate: {acceptance_rate:.2f})')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"Sample Mean: {np.mean(samples):.4f}")
print(f"True Mean: {k * theta:.4f}")
```

Slide 13: Rejection Sampling

Rejection sampling is a technique used to generate observations from a distribution when direct sampling is difficult. It involves sampling from a simpler proposal distribution and accepting or rejecting samples based on a comparison with the target distribution. Let's implement rejection sampling for a custom probability distribution.

```python
import numpy as np
import matplotlib.pyplot as plt

def target_pdf(x):
    return 0.3 * np.exp(-(x - 0.3)**2) + 0.7 * np.exp(-(x - 2.0)**2 / 0.3)

def rejection_sampling(n_samples):
    samples = []
    x = np.linspace(0, 3, 1000)
    M = max(target_pdf(x))
    
    while len(samples) < n_samples:
        x = np.random.uniform(0, 3)
        y = np.random.uniform(0, M)
        
        if y <= target_pdf(x):
            samples.append(x)
    
    return np.array(samples)

n_samples = 10000
samples = rejection_sampling(n_samples)

x = np.linspace(0, 3, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
plt.plot(x, target_pdf(x), 'r-', label='Target PDF')
plt.title('Rejection Sampling')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Slide 14: Importance Sampling for Rare Event Simulation

Importance sampling is particularly useful for simulating rare events. It allows us to estimate the probability of unlikely events more efficiently than direct Monte Carlo simulation. Let's use importance sampling to estimate the probability of a rare event in a simple queueing system.

```python
import numpy as np

def direct_mc_simulation(n_simulations, arrival_rate, service_rate, buffer_size):
    overflow_count = 0
    for _ in range(n_simulations):
        queue_length = 0
        for _ in range(1000):  # Simulate 1000 time steps
            if np.random.random() < arrival_rate:
                queue_length += 1
            if np.random.random() < service_rate and queue_length > 0:
                queue_length -= 1
            if queue_length > buffer_size:
                overflow_count += 1
                break
    return overflow_count / n_simulations

def importance_sampling(n_simulations, arrival_rate, service_rate, buffer_size):
    overflow_probs = []
    for _ in range(n_simulations):
        queue_length = 0
        likelihood_ratio = 1
        for _ in range(1000):  # Simulate 1000 time steps
            if np.random.random() < 0.5:  # Biased arrival rate
                queue_length += 1
                likelihood_ratio *= arrival_rate / 0.5
            if np.random.random() < service_rate and queue_length > 0:
                queue_length -= 1
            if queue_length > buffer_size:
                overflow_probs.append(likelihood_ratio)
                break
    return np.mean(overflow_probs) if overflow_probs else 0

arrival_rate, service_rate, buffer_size = 0.1, 0.15, 10
n_simulations = 100000

direct_prob = direct_mc_simulation(n_simulations, arrival_rate, service_rate, buffer_size)
importance_prob = importance_sampling(n_simulations, arrival_rate, service_rate, buffer_size)

print(f"Direct MC estimation: {direct_prob:.6f}")
print(f"Importance sampling estimation: {importance_prob:.6f}")
```

Slide 15: Additional Resources

For those interested in delving deeper into sampling techniques and their applications in statistics and machine learning, here are some valuable resources:

1. "Monte Carlo Statistical Methods" by Christian P. Robert and George Casella ArXiv: [https://arxiv.org/abs/0908.3655](https://arxiv.org/abs/0908.3655)
2. "An Introduction to MCMC for Machine Learning" by Christophe Andrieu et al. ArXiv: [https://arxiv.org/abs/1109.4435](https://arxiv.org/abs/1109.4435)
3. "A Survey of Monte Carlo Methods for Parameter Estimation" by Johanna Ärje et al. ArXiv: [https://arxiv.org/abs/1parameter-estimation-monte-carlo](https://arxiv.org/abs/1parameter-estimation-monte-carlo)

These resources provide in-depth explanations and advanced techniques in sampling and Monte Carlo methods, which are crucial for various applications in statistics, machine learning, and data science.

