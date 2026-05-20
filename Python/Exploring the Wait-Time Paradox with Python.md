## Exploring the Wait-Time Paradox with Python

Slide 1: Introduction to Wait-Time Analysis

Have you ever wondered why the longer you wait on hold, the more time you expect to keep waiting? This phenomenon is known as the "wait-time paradox," and it can be explored using Python programming and data analysis techniques. In this presentation, we will delve into curve fitting, plotting, and understanding exponential and Weibull distributions to gain insights into this intriguing phenomenon.

```python
import numpy as np
import matplotlib.pyplot as plt
```

Slide 2: Generating Random Wait Times

To analyze wait times, we first need to generate random wait-time data. We can use the exponential distribution, which is commonly used to model wait times in queuing systems.

```python
# Generate random wait times from an exponential distribution
wait_times = np.random.exponential(scale=5, size=1000)
```

Slide 3: Plotting Wait Times

Let's visualize the wait-time data using a histogram to get a better understanding of its distribution.

```python
# Plot a histogram of wait times
plt.hist(wait_times, bins=30, edgecolor='black')
plt.xlabel('Wait Time')
plt.ylabel('Frequency')
plt.title('Distribution of Wait Times')
plt.show()
```

Slide 4: Curve Fitting with Exponential Distribution

We can fit an exponential distribution to the wait-time data using the `scipy.stats` module and compare it to the actual data.

```python
from scipy.stats import expon

# Fit an exponential distribution to the data
params = expon.fit(wait_times)
fitted_pdf = expon.pdf(wait_times, *params)

# Plot the fitted distribution against the data
plt.hist(wait_times, bins=30, density=True, alpha=0.5, label='Data')
plt.plot(wait_times, fitted_pdf, 'r-', label='Exponential Fit')
plt.legend()
plt.show()
```

Slide 5: Limitations of Exponential Distribution

While the exponential distribution provides a good starting point, it may not accurately capture the wait-time paradox. Let's explore another distribution that better models this phenomenon.

```python
from scipy.stats import exponweib

# Fit a Weibull distribution to the data
params = exponweib.fit(wait_times, floc=0)
fitted_pdf = exponweib.pdf(wait_times, *params)

# Plot the fitted distribution against the data
plt.hist(wait_times, bins=30, density=True, alpha=0.5, label='Data')
plt.plot(wait_times, fitted_pdf, 'r-', label='Weibull Fit')
plt.legend()
plt.show()
```

Slide 6: Comparing Distributions

Let's compare the exponential and Weibull distributions side by side to see which one better captures the wait-time data.

```python
# Generate wait times from exponential and Weibull distributions
exp_times = np.random.exponential(scale=5, size=1000)
weib_times = exponweib.rvs(*params, size=1000)

# Plot the distributions
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(exp_times, bins=30, density=True, alpha=0.5, label='Exponential')
ax[0].set_title('Exponential Distribution')
ax[1].hist(weib_times, bins=30, density=True, alpha=0.5, label='Weibull')
ax[1].set_title('Weibull Distribution')
plt.show()
```

Slide 7: Understanding the Wait-Time Paradox

The wait-time paradox suggests that the longer you wait, the more time you expect to keep waiting. Let's explore this phenomenon by simulating a queuing system.

```python
# Simulate a queuing system
queue = []
for i in range(1000):
    wait_time = np.random.weibull(a=2, scale=10)
    queue.append(wait_time)
    if i % 10 == 0:
        print(f"After waiting {i} seconds, the expected remaining wait time is {sum(queue) / len(queue)}")
```

Slide 8: Visualizing the Wait-Time Paradox

Let's plot the expected remaining wait time as a function of the time already waited to visualize the wait-time paradox.

```python
# Simulate a queuing system and record expected remaining wait times
waited_times = []
expected_times = []
queue = []
for i in range(1000):
    wait_time = np.random.weibull(a=2, scale=10)
    queue.append(wait_time)
    waited_times.append(i)
    expected_times.append(sum(queue) / len(queue))

# Plot the wait-time paradox
plt.plot(waited_times, expected_times)
plt.xlabel('Time Waited (seconds)')
plt.ylabel('Expected Remaining Wait Time (seconds)')
plt.title('Wait-Time Paradox')
plt.show()
```

Slide 9: Exploring Other Distributions

While the Weibull distribution provides a better fit for the wait-time data, there are other distributions that can be explored, such as the log-normal distribution.

```python
from scipy.stats import lognorm

# Fit a log-normal distribution to the data
params = lognorm.fit(wait_times, floc=0)
fitted_pdf = lognorm.pdf(wait_times, *params)

# Plot the fitted distribution against the data
plt.hist(wait_times, bins=30, density=True, alpha=0.5, label='Data')
plt.plot(wait_times, fitted_pdf, 'r-', label='Log-normal Fit')
plt.legend()
plt.show()
```

Slide 10: Comparing Multiple Distributions

Let's compare the exponential, Weibull, and log-normal distributions side by side to see which one best captures the wait-time data.

```python
# Generate wait times from different distributions
exp_times = np.random.exponential(scale=5, size=1000)
weib_times = exponweib.rvs(*params, size=1000)
lognorm_times = lognorm.rvs(*params, size=1000)

# Plot the distributions
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].hist(exp_times, bins=30, density=True, alpha=0.5, label='Exponential')
ax[0].set_title('Exponential Distribution')
ax[1].hist(weib_times, bins=30, density=True, alpha=0.5, label='Weibull')
ax[1].set_title('Weibull Distribution')
ax[2].hist(lognorm_times, bins=30, density=True, alpha=0.5, label='Log-normal')
ax[2].set_title('Log-normal Distribution')
plt.show()
```

Slide 11: Goodness-of-Fit Tests

To quantitatively assess the fit of different distributions to the wait-time data, we can perform goodness-of-fit tests, such as the Kolmogorov-Smirnov test or the Anderson-Darling test.

```python
from scipy.stats import kstest, anderson

# Perform Kolmogorov-Smirnov test
exp_ks_stat, exp_ks_pvalue = kstest(wait_times, 'expon', args=params)
weib_ks_stat, weib_ks_pvalue = kstest(wait_times, 'exponweib', args=params)
lognorm_ks_stat, lognorm_ks_pvalue = kstest(wait_times, 'lognorm', args=params)

print(f"Exponential KS test: statistic={exp_ks_stat:.4f}, p-value={exp_ks_pvalue:.4f}")
print(f"Weibull KS test: statistic={weib_ks_stat:.4f}, p-value={weib_ks_pvalue:.4f}")
print(f"Log-normal KS test: statistic={lognorm_ks_stat:.4f}, p-value={lognorm_ks_pvalue:.4f}")

# Perform Anderson-Darling test
exp_ad_stat, exp_ad_pvalue, _ = anderson(wait_times, dist='expon', args=params)
weib_ad_stat, weib_ad_pvalue, _ = anderson(wait_times, dist='exponweib', args=params)
lognorm_ad_stat, lognorm_ad_pvalue, _ = anderson(wait_times, dist='lognorm', args=params)

print(f"\nExponential AD test: statistic={exp_ad_stat:.4f}, p-value={exp_ad_pvalue:.4f}")
print(f"Weibull AD test: statistic={weib_ad_stat:.4f}, p-value={weib_ad_pvalue:.4f}")
print(f"Log-normal AD test: statistic={lognorm_ad_stat:.4f}, p-value={lognorm_ad_pvalue:.4f}")
```

Slide 12: Interpreting Goodness-of-Fit Results

The goodness-of-fit test results can help us determine which distribution provides the best fit for the wait-time data. Lower test statistic values and higher p-values indicate a better fit.

```python
# Print the best-fitting distribution based on test results
best_dist = min([('exponential', exp_ks_stat, exp_ad_stat),
                 ('weibull', weib_ks_stat, weib_ad_stat),
                 ('lognormal', lognorm_ks_stat, lognorm_ad_stat)],
                key=lambda x: x[1] + x[2])

print(f"\nThe best-fitting distribution for the wait-time data is: {best_dist[0]}")
```

Slide 13: Practical Applications

Understanding the wait-time paradox and modeling wait times accurately can have practical applications in various domains, such as customer service, queuing systems, and resource allocation.

```python
# Simulate a queuing system and optimize resource allocation
queue = []
service_rate = 5  # Number of customers served per minute
for i in range(1000):
    wait_time = np.random.weibull(a=2, scale=10)
    queue.append(wait_time)
    if sum(queue) > service_rate * 60:
        print(f"Queue is getting too long, allocating more resources at time {i} minutes.")
        service_rate += 2
```

Slide 14: Conclusion

In this presentation, we explored the wait-time paradox and demonstrated how Python programming and data analysis techniques can be used to model and understand this phenomenon. By fitting various distributions to wait-time data and performing goodness-of-fit tests, we gained insights into the underlying patterns and characteristics of wait times. These insights can be applied in practical scenarios, such as optimizing resource allocation in queuing systems and improving customer service.

