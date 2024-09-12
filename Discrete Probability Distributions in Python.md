## Discrete Probability Distributions in Python
Slide 1: Introduction to Discrete Distributions

Discrete distributions are probability distributions that describe random variables with a finite or countably infinite set of possible values. They are fundamental in statistics and probability theory, used to model various real-world phenomena where outcomes are distinct and separate.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate discrete data
x = np.arange(1, 7)
y = np.random.randint(1, 7, size=1000)

# Plot histogram
plt.hist(y, bins=x - 0.5, rwidth=0.8)
plt.xticks(x)
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.title('Histogram of Discrete Data (Dice Rolls)')
plt.show()
```

Slide 2: Bernoulli Distribution

The Bernoulli distribution models a single binary outcome, such as success/failure or yes/no. It's named after Jacob Bernoulli and is the simplest discrete probability distribution. The probability mass function is defined by a single parameter p, which represents the probability of success.

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli(p, size=1000):
    return np.random.random(size) < p

p = 0.7
results = bernoulli(p)

plt.bar(['Failure', 'Success'], [np.sum(results == 0), np.sum(results == 1)])
plt.title(f'Bernoulli Distribution (p={p})')
plt.ylabel('Count')
plt.show()
```

Slide 3: Bernoulli Distribution - Real-Life Example

Consider a quality control process in a manufacturing plant. Each product is inspected and classified as either defective or non-defective. This scenario can be modeled using a Bernoulli distribution, where success (1) represents a non-defective item and failure (0) represents a defective item.

```python
def quality_control(defect_rate, num_items):
    return bernoulli(1 - defect_rate, num_items)

defect_rate = 0.05
num_items = 1000
inspection_results = quality_control(defect_rate, num_items)

print(f"Number of non-defective items: {np.sum(inspection_results)}")
print(f"Number of defective items: {num_items - np.sum(inspection_results)}")
```

Slide 4: Binomial Distribution

The Binomial distribution models the number of successes in a fixed number of independent Bernoulli trials. It's characterized by two parameters: n (number of trials) and p (probability of success on each trial). The Binomial distribution is widely used in various fields, including biology, physics, and social sciences.

```python
from scipy.stats import binom

n, p = 20, 0.3
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.bar(x, pmf)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()
```

Slide 5: Binomial Distribution - Real-Life Example

Imagine a call center that receives an average of 100 calls per hour. Each call has a 20% chance of requiring escalation to a supervisor. We can use the Binomial distribution to model the number of escalated calls in a given hour.

```python
n_calls = 100
p_escalation = 0.2

# Simulate one hour of calls
escalated_calls = np.random.binomial(n_calls, p_escalation)

print(f"Number of escalated calls in one hour: {escalated_calls}")

# Simulate multiple hours
hours = 1000
escalated_calls_per_hour = np.random.binomial(n_calls, p_escalation, hours)

plt.hist(escalated_calls_per_hour, bins=range(0, max(escalated_calls_per_hour)+2), align='left', rwidth=0.8)
plt.title('Distribution of Escalated Calls per Hour')
plt.xlabel('Number of Escalated Calls')
plt.ylabel('Frequency')
plt.show()
```

Slide 6: Geometric Distribution

The Geometric distribution models the number of Bernoulli trials needed to get the first success. It's characterized by a single parameter p, which is the probability of success on each trial. This distribution is memoryless, meaning the probability of success doesn't depend on previous outcomes.

```python
from scipy.stats import geom

p = 0.3
x = np.arange(1, 15)
pmf = geom.pmf(x, p)

plt.bar(x, pmf)
plt.title(f'Geometric Distribution (p={p})')
plt.xlabel('Number of Trials until First Success')
plt.ylabel('Probability')
plt.show()
```

Slide 7: Geometric Distribution - Application

The Geometric distribution can be used to model the number of attempts needed to achieve a desired outcome. For instance, in a game where players need to roll a six on a fair die, the number of rolls until the first six appears follows a Geometric distribution.

```python
def roll_until_six():
    rolls = 0
    while True:
        rolls += 1
        if np.random.randint(1, 7) == 6:
            return rolls

# Simulate 1000 games
games = 1000
results = [roll_until_six() for _ in range(games)]

plt.hist(results, bins=range(1, max(results)+2), align='left', rwidth=0.8)
plt.title('Number of Rolls Until First Six')
plt.xlabel('Number of Rolls')
plt.ylabel('Frequency')
plt.show()

print(f"Average number of rolls: {np.mean(results):.2f}")
```

Slide 8: Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval of time or space, given that these events happen with a known average rate and independently of each other. It's characterized by a single parameter λ (lambda), which represents both the mean and variance of the distribution.

```python
from scipy.stats import poisson

lambda_param = 3
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_param)

plt.bar(x, pmf)
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.show()
```

Slide 9: Poisson Distribution - Real-Life Example

The Poisson distribution can model various real-world phenomena, such as the number of customers arriving at a store in a given hour, or the number of typos in a document of a certain length. Let's simulate the number of earthquakes occurring in a seismically active region over a period of time.

```python
avg_earthquakes_per_year = 5
years = 100

earthquake_counts = np.random.poisson(avg_earthquakes_per_year, years)

plt.hist(earthquake_counts, bins=range(0, max(earthquake_counts)+2), align='left', rwidth=0.8)
plt.title('Annual Earthquake Counts over 100 Years')
plt.xlabel('Number of Earthquakes')
plt.ylabel('Frequency')
plt.show()

print(f"Average earthquakes per year: {np.mean(earthquake_counts):.2f}")
print(f"Maximum earthquakes in a year: {np.max(earthquake_counts)}")
```

Slide 10: Uniform Distribution (Discrete)

The Discrete Uniform distribution assigns equal probability to a finite set of outcomes. It's characterized by two parameters: a (minimum value) and b (maximum value). This distribution is often used to model situations where each outcome is equally likely, such as rolling a fair die.

```python
from scipy.stats import randint

a, b = 1, 6  # Min and max values for a die
x = np.arange(a, b+1)
pmf = randint.pmf(x, a, b+1)

plt.bar(x, pmf)
plt.title(f'Discrete Uniform Distribution (a={a}, b={b})')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.xticks(x)
plt.show()
```

Slide 11: Uniform Distribution - Application

The Discrete Uniform distribution can be used to model various scenarios where outcomes are equally likely. Let's simulate a simple game where a player wins if they correctly guess a randomly chosen number between 1 and 10.

```python
def play_guessing_game(num_games):
    wins = 0
    for _ in range(num_games):
        secret_number = np.random.randint(1, 11)
        guess = np.random.randint(1, 11)  # Simulate a random guess
        if guess == secret_number:
            wins += 1
    return wins

num_games = 1000
wins = play_guessing_game(num_games)

print(f"Number of wins: {wins}")
print(f"Win rate: {wins/num_games:.2%}")

# Theoretical probability
print(f"Theoretical win probability: {1/10:.2%}")
```

Slide 12: Comparing Discrete Distributions

Different discrete distributions can be used to model various phenomena. Here's a visual comparison of the probability mass functions for the distributions we've discussed.

```python
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Bernoulli
p = 0.7
x = [0, 1]
axs[0, 0].bar(x, [1-p, p])
axs[0, 0].set_title('Bernoulli (p=0.7)')

# Binomial
n, p = 10, 0.3
x = np.arange(0, n+1)
axs[0, 1].bar(x, binom.pmf(x, n, p))
axs[0, 1].set_title(f'Binomial (n={n}, p={p})')

# Geometric
p = 0.3
x = np.arange(1, 15)
axs[1, 0].bar(x, geom.pmf(x, p))
axs[1, 0].set_title(f'Geometric (p={p})')

# Poisson
lambda_param = 3
x = np.arange(0, 15)
axs[1, 1].bar(x, poisson.pmf(x, lambda_param))
axs[1, 1].set_title(f'Poisson (λ={lambda_param})')

plt.tight_layout()
plt.show()
```

Slide 13: Choosing the Right Distribution

Selecting the appropriate discrete distribution depends on the nature of the problem:

1. Bernoulli: For binary outcomes (success/failure).
2. Binomial: For the number of successes in fixed trials.
3. Geometric: For the number of trials until first success.
4. Poisson: For the number of events in a fixed interval.
5. Uniform: For equally likely outcomes.

Consider the underlying process and assumptions when choosing a distribution to model your data.

```python
# Example: Deciding between Binomial and Poisson
n, p = 1000, 0.003
lambda_param = n * p

x = np.arange(0, 15)
binom_pmf = binom.pmf(x, n, p)
poisson_pmf = poisson.pmf(x, lambda_param)

plt.plot(x, binom_pmf, 'bo-', label='Binomial')
plt.plot(x, poisson_pmf, 'ro-', label='Poisson')
plt.title(f'Binomial vs Poisson (n={n}, p={p}, λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into discrete distributions and their applications, here are some valuable resources:

1. "A Survey of Discrete Probability Distributions" by Aleksandar Nanevski (arXiv:2102.07850) URL: [https://arxiv.org/abs/2102.07850](https://arxiv.org/abs/2102.07850)
2. "Probability Distributions in the Physical Sciences" by Michael Trott (arXiv:1611.08318) URL: [https://arxiv.org/abs/1611.08318](https://arxiv.org/abs/1611.08318)
3. "Statistical Distributions" by Catherine Forbes et al. (Book, Wiley)
4. Online courses on probability theory and statistics from platforms like Coursera, edX, or MIT OpenCourseWare.

