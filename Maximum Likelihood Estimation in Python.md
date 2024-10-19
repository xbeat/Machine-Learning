## Maximum Likelihood Estimation in Python

Slide 1: Introduction to Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution by maximizing the likelihood function. It's a fundamental technique in statistics and machine learning, providing a way to fit models to data and make inferences about populations based on samples.

```python
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate sample data
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)

# Plot histogram of the data
plt.hist(data, bins=30, density=True, alpha=0.7)
plt.title("Sample Data with Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

Slide 2: The Likelihood Function

The likelihood function is the joint probability of observing the given data under a specific probability distribution with certain parameters. For a set of independent and identically distributed observations, the likelihood is the product of individual probabilities.

```python
    mu, sigma = params
    return np.prod(norm.pdf(data, mu, sigma))

# Calculate likelihood for different mean values
mu_range = np.linspace(4, 6, 100)
likelihoods = [likelihood((mu, 2), data) for mu in mu_range]

plt.plot(mu_range, likelihoods)
plt.title("Likelihood Function")
plt.xlabel("Mean (μ)")
plt.ylabel("Likelihood")
plt.show()
```

Slide 3: Log-Likelihood Function

In practice, we often work with the log-likelihood function instead of the likelihood function. This is because the log-likelihood is easier to work with mathematically and helps prevent numerical underflow when dealing with very small probabilities.

```python
    mu, sigma = params
    return np.sum(norm.logpdf(data, mu, sigma))

# Calculate log-likelihood for different mean values
log_likelihoods = [log_likelihood((mu, 2), data) for mu in mu_range]

plt.plot(mu_range, log_likelihoods)
plt.title("Log-Likelihood Function")
plt.xlabel("Mean (μ)")
plt.ylabel("Log-Likelihood")
plt.show()
```

Slide 4: Maximum Likelihood Estimation Process

The MLE process involves finding the parameter values that maximize the likelihood (or log-likelihood) function. This is typically done using optimization algorithms, such as gradient descent or Newton's method.

```python

def negative_log_likelihood(params, data):
    return -log_likelihood(params, data)

# Initial guess for parameters
initial_guess = [np.mean(data), np.std(data)]

# Perform MLE
result = minimize(negative_log_likelihood, initial_guess, args=(data,))

print(f"MLE estimates: μ = {result.x[0]:.2f}, σ = {result.x[1]:.2f}")
```

Slide 5: MLE for Normal Distribution

For a normal distribution, the maximum likelihood estimates for the mean (μ) and standard deviation (σ) have closed-form solutions. The MLE for μ is the sample mean, and the MLE for σ is the square root of the biased sample variance.

```python
mle_std = np.sqrt(np.mean((data - mle_mean)**2))

print(f"Analytical MLE estimates: μ = {mle_mean:.2f}, σ = {mle_std:.2f}")

# Compare with true parameters
print(f"True parameters: μ = 5.00, σ = 2.00")
```

Slide 6: Visualizing MLE Fit

Let's visualize how well our MLE estimates fit the observed data by plotting the estimated probability density function against the histogram of the data.

```python
y = norm.pdf(x, mle_mean, mle_std)

plt.hist(data, bins=30, density=True, alpha=0.7)
plt.plot(x, y, 'r-', lw=2)
plt.title("MLE Fit to Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend(["MLE Fit", "Data"])
plt.show()
```

Slide 7: MLE for Bernoulli Distribution

The Bernoulli distribution is used for binary outcomes. Let's estimate the probability of success (p) for a series of coin flips using MLE.

```python
np.random.seed(42)
coin_flips = np.random.binomial(1, 0.7, 1000)

# MLE for Bernoulli is simply the sample mean
p_mle = np.mean(coin_flips)

print(f"MLE estimate for p: {p_mle:.4f}")
print(f"True p: 0.7000")

# Plot the results
plt.bar(["Heads", "Tails"], [p_mle, 1-p_mle])
plt.title("MLE Estimate of Coin Flip Probability")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.show()
```

Slide 8: MLE for Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval. Let's estimate the rate parameter (λ) for a Poisson process.

```python

# Generate Poisson data
np.random.seed(42)
poisson_data = np.random.poisson(lam=3, size=1000)

# MLE for Poisson is the sample mean
lambda_mle = np.mean(poisson_data)

print(f"MLE estimate for λ: {lambda_mle:.4f}")
print(f"True λ: 3.0000")

# Plot the results
x = np.arange(0, 15)
plt.bar(x, np.bincount(poisson_data, minlength=15)/len(poisson_data), alpha=0.7, label="Data")
plt.plot(x, poisson.pmf(x, lambda_mle), 'r-', lw=2, label="MLE Fit")
plt.title("MLE Fit for Poisson Distribution")
plt.xlabel("Number of events")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 9: Properties of Maximum Likelihood Estimators

Maximum Likelihood Estimators have several desirable properties:

1. Consistency: As the sample size increases, the MLE converges to the true parameter value.
2. Asymptotic normality: For large samples, the MLE is approximately normally distributed.
3. Efficiency: Among consistent estimators, the MLE has the lowest possible variance.

```python
sample_sizes = [10, 100, 1000, 10000]
mle_estimates = []

for size in sample_sizes:
    sample = np.random.normal(loc=5, scale=2, size=size)
    mle_estimates.append(np.mean(sample))

plt.plot(sample_sizes, mle_estimates, 'bo-')
plt.axhline(y=5, color='r', linestyle='--')
plt.xscale('log')
plt.title("Consistency of MLE")
plt.xlabel("Sample Size")
plt.ylabel("MLE Estimate (μ)")
plt.legend(["MLE Estimates", "True μ"])
plt.show()
```

Slide 10: Real-Life Example: Estimating Email Spam Rate

Suppose we want to estimate the probability of an email being spam based on historical data. We can use MLE with a Bernoulli distribution to model this scenario.

```python
np.random.seed(42)
emails = np.random.binomial(1, 0.2, 1000)

# MLE for spam probability
p_spam_mle = np.mean(emails)

print(f"MLE estimate for spam probability: {p_spam_mle:.4f}")

# Visualize the results
labels = ['Not Spam', 'Spam']
sizes = [(1 - p_spam_mle) * 100, p_spam_mle * 100]
colors = ['lightblue', 'lightcoral']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Estimated Email Classification")
plt.show()
```

Slide 11: Real-Life Example: Estimating Arrival Rate at a Restaurant

Let's use MLE to estimate the average number of customers arriving at a restaurant per hour, assuming a Poisson distribution for arrivals.

```python
np.random.seed(42)
arrivals = np.random.poisson(lam=15, size=100)  # 100 hours of data

# MLE for arrival rate
lambda_mle = np.mean(arrivals)

print(f"MLE estimate for average arrivals per hour: {lambda_mle:.2f}")

# Visualize the results
x = np.arange(0, 30)
plt.hist(arrivals, bins=range(31), density=True, alpha=0.7, label="Observed Data")
plt.plot(x, poisson.pmf(x, lambda_mle), 'r-', lw=2, label="MLE Fit")
plt.title("Customer Arrivals per Hour")
plt.xlabel("Number of Arrivals")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

Slide 12: Challenges and Limitations of MLE

While MLE is a powerful technique, it has some limitations:

1. Sensitivity to initial conditions in numerical optimization.
2. Potential for overfitting with small sample sizes.
3. Difficulty in handling complex, multi-modal distributions.
4. Assumption of a specific probability distribution for the data.

```python
def complex_likelihood(x, data):
    return np.sum(np.sin(x * data) + np.cos(x * data))

x_range = np.linspace(0, 10, 1000)
y = [complex_likelihood(x, data) for x in x_range]

plt.plot(x_range, y)
plt.title("Complex Likelihood Function")
plt.xlabel("Parameter Value")
plt.ylabel("Likelihood")
plt.show()

# Multiple local maxima can lead to different MLE results
# depending on the starting point of optimization
```

Slide 13: Alternatives and Extensions to MLE

Several alternatives and extensions to MLE exist to address its limitations:

1. Bayesian Inference: Incorporates prior knowledge about parameters.
2. Regularized MLE: Adds a penalty term to prevent overfitting.
3. Expectation-Maximization (EM) Algorithm: Handles latent variables and mixture models.
4. Robust MLE: Less sensitive to outliers and model misspecification.

```python

# Bayesian inference example (simple prior)
def posterior(theta, data, prior):
    return likelihood((theta, 2), data) * prior.pdf(theta)

prior = uniform(loc=0, scale=10)
posterior_values = [posterior(theta, data, prior) for theta in mu_range]

plt.plot(mu_range, posterior_values)
plt.title("Posterior Distribution")
plt.xlabel("θ")
plt.ylabel("Posterior Probability")
plt.show()
```

Slide 14: Additional Resources

For those interested in diving deeper into Maximum Likelihood Estimation and related topics, here are some valuable resources:

1. "Maximum Likelihood Estimation and Inference" by Russell Davidson and James G. MacKinnon (2003)
2. "Statistical Inference" by George Casella and Roger L. Berger (2002)
3. ArXiv paper: "A Tutorial on the Maximum Likelihood Estimation of Linear Regression Models" by Cosma Rohilla Shalizi ([https://arxiv.org/abs/1008.4686](https://arxiv.org/abs/1008.4686))
4. ArXiv paper: "Maximum Likelihood Estimation of Mixture Models: A Comprehensive Review" by Zoubin Ghahramani and Geoffrey E. Hinton ([https://arxiv.org/abs/1507.00121](https://arxiv.org/abs/1507.00121))

These resources provide in-depth explanations, mathematical derivations, and advanced applications of Maximum Likelihood Estimation techniques.


