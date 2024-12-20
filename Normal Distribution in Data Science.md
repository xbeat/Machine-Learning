## Normal Distribution in Data Science
Slide 1: Probability Distributions in Data Science

Probability distributions are fundamental concepts in statistics and data science. They describe the likelihood of different outcomes in a random event or experiment. Understanding these distributions is crucial for data analysis, machine learning, and statistical inference. Let's explore some of the most common probability distributions and their applications.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Function to plot probability distributions
def plot_distribution(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()
```

Slide 2: Normal Distribution (Continuous)

The Normal distribution, also known as the Gaussian distribution, is characterized by its symmetric bell-shaped curve. It's widely used in data science due to its prevalence in natural phenomena. For example, the heights of individuals in a population often follow a normal distribution.

```python
# Generate data for a normal distribution
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, 0, 1)

plot_distribution(x, y, 'Normal Distribution')

# Example: Simulating heights of individuals
heights = np.random.normal(170, 10, 1000)  # Mean: 170 cm, Std Dev: 10 cm
print(f"Average height: {heights.mean():.2f} cm")
print(f"Standard deviation: {heights.std():.2f} cm")
```

Slide 3: Bernoulli Distribution (Discrete)

The Bernoulli distribution models binary outcomes, such as success or failure, yes or no. It's useful for representing events with only two possible outcomes. A practical example is modeling the outcome of a single coin flip.

```python
# Simulating Bernoulli trials (coin flips)
p = 0.5  # Probability of success (fair coin)
n_flips = 1000
coin_flips = np.random.binomial(n=1, p=p, size=n_flips)

heads = np.sum(coin_flips)
tails = n_flips - heads

print(f"Number of heads: {heads}")
print(f"Number of tails: {tails}")
print(f"Proportion of heads: {heads/n_flips:.2f}")
```

Slide 4: Binomial Distribution (Discrete)

The Binomial distribution extends the Bernoulli distribution to model the number of successes in a fixed number of independent trials. It's commonly used in scenarios involving repeated experiments with binary outcomes.

```python
# Simulating exam pass/fail rates
n_students = 100
pass_probability = 0.7
exam_results = np.random.binomial(n=1, p=pass_probability, size=n_students)

passed = np.sum(exam_results)
failed = n_students - passed

print(f"Number of students who passed: {passed}")
print(f"Number of students who failed: {failed}")
print(f"Pass rate: {passed/n_students:.2f}")

# Plotting the binomial distribution
x = np.arange(0, n_students + 1)
y = stats.binom.pmf(x, n_students, pass_probability)
plot_distribution(x, y, 'Binomial Distribution')
```

Slide 5: Poisson Distribution (Discrete)

The Poisson distribution models the number of events occurring in a fixed interval of time or space, given a known average rate. It's useful for modeling rare events or counting processes. For instance, it can be used to model the number of goals scored in a soccer match.

```python
# Simulating goals scored in soccer matches
average_goals = 2.5
n_matches = 1000

goals = np.random.poisson(lam=average_goals, size=n_matches)

print(f"Average goals per match: {goals.mean():.2f}")
print(f"Maximum goals in a single match: {goals.max()}")

# Plotting the Poisson distribution
x = np.arange(0, 15)
y = stats.poisson.pmf(x, average_goals)
plot_distribution(x, y, 'Poisson Distribution')
```

Slide 6: Exponential Distribution (Continuous)

The Exponential distribution models the time between events in a Poisson process. It's often used to represent waiting times or the lifespan of electronic components. For example, it can model the time between customer arrivals at a store.

```python
# Simulating time between customer arrivals
average_time = 5  # minutes
n_customers = 1000

arrival_times = np.random.exponential(scale=average_time, size=n_customers)

print(f"Average time between arrivals: {arrival_times.mean():.2f} minutes")
print(f"Longest wait time: {arrival_times.max():.2f} minutes")

# Plotting the Exponential distribution
x = np.linspace(0, 20, 100)
y = stats.expon.pdf(x, scale=average_time)
plot_distribution(x, y, 'Exponential Distribution')
```

Slide 7: Gamma Distribution (Continuous)

The Gamma distribution is a generalization of the Exponential distribution. It models the waiting time for a specific number of events in a Poisson process. This distribution is versatile and can be used to model various phenomena, such as rainfall amounts or the time required to complete a task.

```python
# Simulating task completion times
shape, scale = 2, 2
n_tasks = 1000

completion_times = np.random.gamma(shape, scale, n_tasks)

print(f"Average completion time: {completion_times.mean():.2f} hours")
print(f"Longest completion time: {completion_times.max():.2f} hours")

# Plotting the Gamma distribution
x = np.linspace(0, 20, 100)
y = stats.gamma.pdf(x, a=shape, scale=scale)
plot_distribution(x, y, 'Gamma Distribution')
```

Slide 8: Beta Distribution (Continuous)

The Beta distribution models probabilities, making it useful for Bayesian inference and modeling proportions. Unlike the Binomial distribution where probability is a parameter, in the Beta distribution, probability itself is a random variable. It's often used to model the distribution of probabilities in A/B testing scenarios.

```python
# Simulating conversion rates in A/B testing
a, b = 10, 20  # Shape parameters
n_simulations = 1000

conversion_rates = np.random.beta(a, b, n_simulations)

print(f"Average conversion rate: {conversion_rates.mean():.4f}")
print(f"95% confidence interval: ({np.percentile(conversion_rates, 2.5):.4f}, {np.percentile(conversion_rates, 97.5):.4f})")

# Plotting the Beta distribution
x = np.linspace(0, 1, 100)
y = stats.beta.pdf(x, a, b)
plot_distribution(x, y, 'Beta Distribution')
```

Slide 9: Uniform Distribution (Continuous/Discrete)

The Uniform distribution represents scenarios where all outcomes within a given range are equally likely. It can be continuous (e.g., selecting a random point on a line segment) or discrete (e.g., rolling a fair die). It's often used in simulations and random number generation.

```python
# Simulating fair die rolls
n_rolls = 1000
die_rolls = np.random.randint(1, 7, n_rolls)

for i in range(1, 7):
    print(f"Number of {i}s rolled: {np.sum(die_rolls == i)}")

# Plotting the Uniform distribution (continuous)
x = np.linspace(0, 1, 100)
y = stats.uniform.pdf(x)
plot_distribution(x, y, 'Uniform Distribution (Continuous)')
```

Slide 10: Student's t-Distribution (Continuous)

The Student's t-distribution is similar to the Normal distribution but has heavier tails. It's commonly used in hypothesis testing and estimating confidence intervals for small sample sizes. In machine learning, it's utilized in t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction.

```python
# Simulating t-distribution
degrees_of_freedom = 5
n_samples = 1000

t_samples = np.random.standard_t(degrees_of_freedom, n_samples)

print(f"Mean: {t_samples.mean():.4f}")
print(f"Standard deviation: {t_samples.std():.4f}")

# Plotting the t-distribution
x = np.linspace(-4, 4, 100)
y = stats.t.pdf(x, degrees_of_freedom)
plot_distribution(x, y, "Student's t-Distribution")
```

Slide 11: Log-Normal Distribution (Continuous)

The Log-Normal distribution occurs when the logarithm of a random variable follows a normal distribution. It's often used to model right-skewed data, such as income distributions or particle sizes in physical processes.

```python
# Simulating particle sizes
mu, sigma = 0, 0.5
n_particles = 1000

particle_sizes = np.random.lognormal(mu, sigma, n_particles)

print(f"Median particle size: {np.median(particle_sizes):.4f}")
print(f"Mean particle size: {particle_sizes.mean():.4f}")

# Plotting the Log-Normal distribution
x = np.linspace(0, 5, 100)
y = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
plot_distribution(x, y, 'Log-Normal Distribution')
```

Slide 12: Weibull Distribution (Continuous)

The Weibull distribution is versatile and often employed to analyze time-to-failure data in reliability engineering. It can model increasing, decreasing, or constant failure rates depending on its shape parameter. This distribution is useful in predicting product lifetimes and planning maintenance schedules.

```python
# Simulating product lifetimes
shape, scale = 2, 1000  # Shape and scale parameters
n_products = 1000

lifetimes = np.random.weibull(shape, n_products) * scale

print(f"Average product lifetime: {lifetimes.mean():.2f} hours")
print(f"Median product lifetime: {np.median(lifetimes):.2f} hours")

# Plotting the Weibull distribution
x = np.linspace(0, 2000, 100)
y = stats.weibull_min.pdf(x, shape, scale=scale)
plot_distribution(x, y, 'Weibull Distribution')
```

Slide 13: Real-Life Example: Height Distribution

Let's apply the Normal distribution to model the heights of adults in a population. This example demonstrates how probability distributions can be used to analyze and make inferences about real-world data.

```python
# Simulating height data
mean_height = 170  # cm
std_dev = 10  # cm
n_people = 1000

heights = np.random.normal(mean_height, std_dev, n_people)

print(f"Average height: {heights.mean():.2f} cm")
print(f"Standard deviation: {heights.std():.2f} cm")

# Plotting the height distribution
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=30, density=True, alpha=0.7)
plt.title('Distribution of Adult Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Probability Density')

# Overlay the theoretical normal distribution
x = np.linspace(mean_height - 4*std_dev, mean_height + 4*std_dev, 100)
y = stats.norm.pdf(x, mean_height, std_dev)
plt.plot(x, y, 'r-', lw=2)

plt.grid(True)
plt.show()

# Calculating probabilities
print(f"Probability of height > 190 cm: {(heights > 190).mean():.4f}")
print(f"Probability of height between 160 and 180 cm: {((heights >= 160) & (heights <= 180)).mean():.4f}")
```

Slide 14: Real-Life Example: Customer Arrivals

Let's use the Poisson and Exponential distributions to model customer arrivals at a coffee shop. This example shows how these distributions can be applied to analyze and predict customer behavior.

```python
# Simulating customer arrivals
avg_customers_per_hour = 20
simulation_hours = 8

# Poisson distribution for number of customers per hour
customers_per_hour = np.random.poisson(avg_customers_per_hour, simulation_hours)

print("Customers per hour:")
for hour, customers in enumerate(customers_per_hour, 1):
    print(f"Hour {hour}: {customers} customers")

print(f"\nTotal customers: {np.sum(customers_per_hour)}")
print(f"Average customers per hour: {np.mean(customers_per_hour):.2f}")

# Exponential distribution for time between arrivals
avg_time_between_arrivals = 60 / avg_customers_per_hour  # minutes
n_intervals = 100

inter_arrival_times = np.random.exponential(avg_time_between_arrivals, n_intervals)

plt.figure(figsize=(10, 6))
plt.hist(inter_arrival_times, bins=20, density=True, alpha=0.7)
plt.title('Distribution of Time Between Customer Arrivals')
plt.xlabel('Time (minutes)')
plt.ylabel('Probability Density')

# Overlay the theoretical exponential distribution
x = np.linspace(0, max(inter_arrival_times), 100)
y = stats.expon.pdf(x, scale=avg_time_between_arrivals)
plt.plot(x, y, 'r-', lw=2)

plt.grid(True)
plt.show()

print(f"\nAverage time between arrivals: {np.mean(inter_arrival_times):.2f} minutes")
print(f"Probability of waiting more than 5 minutes: {(inter_arrival_times > 5).mean():.4f}")
```

Slide 15: Additional Resources

For those interested in diving deeper into probability distributions and their applications in data science, here are some recommended resources:

1. "A Survey of Probability Distributions" by Hao Zhang (arXiv:1807.11347) URL: [https://arxiv.org/abs/1807.11347](https://arxiv.org/abs/1807.11347)
2. "Probability Distributions in Machine Learning" by Shubham Jain (arXiv:2106.02252) URL: [https://arxiv.org/abs/2106.02252](https://arxiv.org/abs/2106.02252)
3. "An Introduction to Probability and Statistics Using Python" by Paul Fackler (arXiv:2002.04292) URL: [https://arxiv.org/abs/2002.04292](https://arxiv.org/abs/2002.04292)

These papers provide comprehensive overviews and advanced applications of probability distributions in various fields of data science and machine learning.

