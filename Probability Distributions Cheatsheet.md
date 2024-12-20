## Probability Distributions Cheatsheet
Slide 1: Foundations of Probability Distributions

Probability distributions form the backbone of statistical modeling and machine learning. They describe the likelihood of different outcomes occurring in a random process, providing essential mathematical tools for data analysis, inference, and prediction.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate random data from different distributions
normal_data = np.random.normal(loc=0, scale=1, size=1000)
uniform_data = np.random.uniform(low=-3, high=3, size=1000)

# Create visualization
plt.figure(figsize=(12, 6))
plt.hist(normal_data, bins=30, alpha=0.5, label='Normal')
plt.hist(uniform_data, bins=30, alpha=0.5, label='Uniform')
plt.legend()
plt.title('Comparing Normal and Uniform Distributions')
plt.show()

# Calculate basic statistics
print(f"Normal mean: {normal_data.mean():.2f}, std: {normal_data.std():.2f}")
print(f"Uniform mean: {uniform_data.mean():.2f}, std: {uniform_data.std():.2f}")
```

Slide 2: Normal Distribution Mathematics

The normal distribution, also known as Gaussian distribution, is characterized by its probability density function (PDF). The mathematical foundation involves key parameters μ (mean) and σ (standard deviation).

```python
# Mathematical representation of Normal Distribution PDF
"""
PDF formula:
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where:
$$\mu$$ is the mean
$$\sigma$$ is the standard deviation
"""

def normal_pdf(x, mu, sigma):
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2)/(2 * sigma**2))

x = np.linspace(-5, 5, 1000)
pdf = normal_pdf(x, mu=0, sigma=1)

plt.plot(x, pdf)
plt.title('Standard Normal Distribution PDF')
plt.grid(True)
plt.show()
```

Slide 3: Exponential Distribution Implementation

The exponential distribution models the time between events in a Poisson process. It's commonly used in reliability engineering and queuing theory to model time intervals between independent events.

```python
def exponential_pdf(x, lambda_param):
    """
    $$f(x) = \lambda e^{-\lambda x}$$
    where λ is the rate parameter
    """
    return lambda_param * np.exp(-lambda_param * x)

x = np.linspace(0, 5, 1000)
lambdas = [0.5, 1, 2]

plt.figure(figsize=(10, 6))
for l in lambdas:
    plt.plot(x, exponential_pdf(x, l), label=f'λ={l}')

plt.title('Exponential Distribution PDF')
plt.legend()
plt.grid(True)
plt.show()

# Generate random samples
samples = np.random.exponential(scale=1/2, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: {1/2})")
```

Slide 4: Chi-Square Distribution Analysis

The chi-square distribution emerges from the sum of squared standard normal variables. It's fundamental in hypothesis testing and confidence interval construction for variance estimation.

```python
def chi_square_pdf(x, df):
    """
    $$f(x) = \frac{x^{(k/2-1)}e^{-x/2}}{2^{k/2}\Gamma(k/2)}$$
    where k is degrees of freedom
    """
    return stats.chi2.pdf(x, df)

x = np.linspace(0, 15, 1000)
dfs = [1, 2, 5]

plt.figure(figsize=(10, 6))
for df in dfs:
    plt.plot(x, chi_square_pdf(x, df), label=f'df={df}')

plt.title('Chi-Square Distribution PDF')
plt.legend()
plt.grid(True)
plt.show()

# Generate chi-square samples
samples = np.random.chisquare(df=2, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: 2)")
```

Slide 5: Poisson Distribution Implementation

The Poisson distribution models the number of events occurring in a fixed interval when these events happen with a known average rate and independently of the time since the last event.

```python
def poisson_pmf(k, lambda_param):
    """
    $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
    where λ is the rate parameter
    """
    return (lambda_param**k * np.exp(-lambda_param)) / np.math.factorial(k)

k = np.arange(0, 15)
lambdas = [1, 4, 8]

plt.figure(figsize=(10, 6))
for l in lambdas:
    pmf = [poisson_pmf(ki, l) for ki in k]
    plt.plot(k, pmf, 'o-', label=f'λ={l}')

plt.title('Poisson Distribution PMF')
plt.legend()
plt.grid(True)
plt.show()

# Generate Poisson samples
samples = np.random.poisson(lam=4, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: 4)")
```

Slide 6: Binomial Distribution and Applications

The binomial distribution models the number of successes in a fixed number of independent Bernoulli trials. Each trial has the same probability of success and is independent of other trials.

```python
def binomial_pmf(n, k, p):
    """
    $$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$
    where:
    n = number of trials
    k = number of successes
    p = probability of success
    """
    return stats.binom.pmf(k, n, p)

n = 20  # number of trials
p = 0.3  # probability of success
k = np.arange(0, n+1)

plt.figure(figsize=(10, 6))
pmf = binomial_pmf(n, k, p)
plt.bar(k, pmf, alpha=0.8)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.grid(True)
plt.show()

# Generate samples
samples = np.random.binomial(n=20, p=0.3, size=1000)
print(f"Mean: {samples.mean():.2f} (Expected: {n*p})")
```

Slide 7: Beta Distribution and Bayesian Applications

The Beta distribution is crucial in Bayesian statistics, serving as a conjugate prior for the Bernoulli and Binomial distributions. It models continuous probabilities in the interval \[0,1\].

```python
def plot_beta_distribution(alphas, betas):
    """
    $$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$$
    where B(α,β) is the Beta function
    """
    x = np.linspace(0, 1, 1000)
    plt.figure(figsize=(12, 6))
    
    for a, b in zip(alphas, betas):
        plt.plot(x, stats.beta.pdf(x, a, b), 
                label=f'α={a}, β={b}')
    
    plt.title('Beta Distribution PDF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot different parameter combinations
alphas = [0.5, 5, 1]
betas = [0.5, 1, 3]
plot_beta_distribution(alphas, betas)

# Generate samples
samples = np.random.beta(a=2, b=5, size=1000)
print(f"Mean: {samples.mean():.3f}")
```

Slide 8: Multivariate Normal Distribution

The multivariate normal distribution extends the normal distribution to higher dimensions, essential for modeling correlated random variables and in many machine learning applications.

```python
def multivariate_normal_example():
    """
    $$f(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} 
    \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$
    """
    mean = [0, 0]
    cov = [[1, 0.5], 
           [0.5, 2]]
    
    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, 1000)
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
    plt.title('Multivariate Normal Distribution Samples')
    plt.axis('equal')
    plt.grid(True)
    plt.show()
    
    # Calculate empirical correlation
    print(f"Empirical correlation: {np.corrcoef(samples.T)[0,1]:.3f}")
    print(f"Theoretical correlation: {cov[0][1]/np.sqrt(cov[0][0]*cov[1][1]):.3f}")

multivariate_normal_example()
```

Slide 9: Gamma Distribution Implementation

The Gamma distribution generalizes the exponential distribution and is widely used in modeling waiting times, life testing, and as a conjugate prior in Bayesian statistics.

```python
def plot_gamma_distribution(alphas, betas):
    """
    $$f(x; \alpha, \beta) = \frac{\beta^\alpha x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}$$
    where α is shape and β is rate
    """
    x = np.linspace(0, 10, 1000)
    plt.figure(figsize=(12, 6))
    
    for a, b in zip(alphas, betas):
        plt.plot(x, stats.gamma.pdf(x, a, scale=1/b), 
                label=f'α={a}, β={b}')
    
    plt.title('Gamma Distribution PDF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot different parameter combinations
alphas = [1, 2, 5]
betas = [1, 2, 1]
plot_gamma_distribution(alphas, betas)

# Generate samples
samples = np.random.gamma(shape=2, scale=1/2, size=1000)
print(f"Mean: {samples.mean():.3f} (Expected: {2/(2)})")
```

Slide 10: Real-world Application - Network Traffic Analysis

Network packet arrivals are commonly modeled using probability distributions. This example demonstrates analyzing network traffic patterns using Poisson and Exponential distributions for inter-arrival times.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Simulate network packet arrivals
np.random.seed(42)
num_packets = 1000
arrival_rate = 5  # packets per second

# Generate inter-arrival times (exponential distribution)
inter_arrival_times = np.random.exponential(1/arrival_rate, num_packets)
arrival_times = np.cumsum(inter_arrival_times)

# Count packets in fixed intervals
interval_size = 1.0  # 1 second intervals
num_intervals = int(np.ceil(arrival_times[-1]))
packet_counts = np.zeros(num_intervals)

for time in arrival_times:
    interval = int(time)
    if interval < num_intervals:
        packet_counts[interval] += 1

# Statistical analysis
mean_packets = np.mean(packet_counts)
std_packets = np.std(packet_counts)

plt.figure(figsize=(12, 6))
plt.hist(packet_counts, bins=20, density=True, alpha=0.7)
x = np.arange(0, max(packet_counts)+1)
plt.plot(x, stats.poisson.pmf(x, mean_packets), 'r-', label='Poisson fit')
plt.title('Network Packet Arrivals Distribution')
plt.xlabel('Packets per Second')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

print(f"Mean packets per second: {mean_packets:.2f}")
print(f"Standard deviation: {std_packets:.2f}")
print(f"Theoretical std (Poisson): {np.sqrt(mean_packets):.2f}")
```

Slide 11: Real-world Application - Financial Risk Modeling

This implementation demonstrates using probability distributions to model financial returns and estimate Value at Risk (VaR) using both normal and Student's t-distributions.

```python
def calculate_var_metrics(returns, confidence_levels=[0.95, 0.99]):
    """
    Returns Value at Risk (VaR) and Expected Shortfall (ES)
    $$VaR_\alpha = \mu + \sigma \Phi^{-1}(\alpha)$$
    where Φ⁻¹ is the inverse CDF of the standard normal distribution
    """
    mu = returns.mean()
    sigma = returns.std()
    
    results = {}
    for conf in confidence_levels:
        # Normal VaR
        var_normal = -stats.norm.ppf(1-conf, mu, sigma)
        
        # Student's t VaR (with estimated degrees of freedom)
        t_params = stats.t.fit(returns)
        var_student = -stats.t.ppf(1-conf, *t_params)
        
        results[conf] = {
            'VaR_normal': var_normal,
            'VaR_student': var_student
        }
    
    return results

# Generate sample financial returns
np.random.seed(42)
n_days = 1000
returns = np.random.normal(0.0001, 0.01, n_days)

# Add some fat-tail events
returns = np.append(returns, np.random.standard_t(df=3, size=50) * 0.02)

# Calculate VaR
var_results = calculate_var_metrics(returns)

plt.figure(figsize=(12, 6))
plt.hist(returns, bins=50, density=True, alpha=0.7)
x = np.linspace(min(returns), max(returns), 100)
plt.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 
         'r-', label='Normal fit')
plt.plot(x, stats.t.pdf(x, *stats.t.fit(returns)), 
         'g-', label='Student t fit')
plt.title('Financial Returns Distribution')
plt.legend()
plt.grid(True)
plt.show()

for conf, metrics in var_results.items():
    print(f"\nConfidence Level: {conf*100}%")
    print(f"Normal VaR: {metrics['VaR_normal']:.4f}")
    print(f"Student's t VaR: {metrics['VaR_student']:.4f}")
```

Slide 12: Kernel Density Estimation (KDE)

KDE is a non-parametric method to estimate probability density functions. It's particularly useful when data doesn't follow standard distributions and requires flexible density estimation.

```python
def kde_estimation(data, bandwidths=[0.1, 0.3, 0.5]):
    """
    $$\hat{f}_h(x) = \frac{1}{nh}\sum_{i=1}^n K\left(\frac{x-x_i}{h}\right)$$
    where K is the kernel function and h is the bandwidth
    """
    x_grid = np.linspace(min(data)-1, max(data)+1, 200)
    
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=30, density=True, alpha=0.3, label='Data')
    
    for bw in bandwidths:
        kde = stats.gaussian_kde(data, bw_method=bw)
        plt.plot(x_grid, kde(x_grid), 
                label=f'KDE (bandwidth={bw})')
    
    plt.title('Kernel Density Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate mixture of normal distributions
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(1, 1, 700)
])

kde_estimation(data)
print(f"Sample statistics:")
print(f"Mean: {np.mean(data):.3f}")
print(f"Std: {np.std(data):.3f}")
print(f"Skewness: {stats.skew(data):.3f}")
print(f"Kurtosis: {stats.kurtosis(data):.3f}")
```

Slide 13: Mixture Models Implementation

Mixture models combine multiple probability distributions to model complex data patterns. This implementation showcases Gaussian Mixture Models (GMM) with expectation-maximization for parameter estimation.

```python
from sklearn.mixture import GaussianMixture

def fit_gaussian_mixture(data, n_components=2):
    """
    Gaussian Mixture Model:
    $$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k,\Sigma_k)$$
    where πk are mixing coefficients
    """
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data.reshape(-1, 1))
    
    # Plot results
    x = np.linspace(data.min()-1, data.max()+1, 1000).reshape(-1, 1)
    scores = np.exp(gmm.score_samples(x))
    responsibilities = gmm.predict_proba(x)
    
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=50, density=True, alpha=0.5)
    plt.plot(x, scores, 'r-', label='GMM density')
    
    for i in range(n_components):
        plt.plot(x, responsibilities[:, i] * scores,
                '--', label=f'Component {i+1}')
    
    plt.title('Gaussian Mixture Model Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return gmm

# Generate synthetic data from mixture
np.random.seed(42)
data = np.concatenate([
    np.random.normal(-2, 0.5, 300),
    np.random.normal(2, 1.0, 700)
])

gmm = fit_gaussian_mixture(data)
print("Mixture Parameters:")
for i, (mean, covar, weight) in enumerate(zip(
    gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_
)):
    print(f"\nComponent {i+1}:")
    print(f"Mean: {mean:.3f}")
    print(f"Variance: {covar:.3f}")
    print(f"Weight: {weight:.3f}")
```

Slide 14: Distribution Testing and Goodness of Fit

Statistical tests help determine whether data follows a particular distribution. This implementation covers multiple goodness-of-fit tests and their interpretations.

```python
def distribution_testing(data, alpha=0.05):
    """
    Implements multiple distribution tests:
    - Shapiro-Wilk test for normality
    - Anderson-Darling test
    - Kolmogorov-Smirnov test
    """
    # Visual QQ plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title("Q-Q Plot")
    
    plt.subplot(122)
    plt.hist(data, bins='auto', density=True, alpha=0.7)
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(data), np.std(data)),
            'r-', label='Normal fit')
    plt.title("Histogram with Normal Fit")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    shapiro_stat, shapiro_p = stats.shapiro(data)
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    ad_result = stats.anderson(data, dist='norm')
    
    print("\nNormality Tests Results:")
    print(f"Shapiro-Wilk test: p-value = {shapiro_p:.4f}")
    print(f"Kolmogorov-Smirnov test: p-value = {ks_p:.4f}")
    print("\nAnderson-Darling test:")
    for i in range(len(ad_result.critical_values)):
        sig = (1 - float(ad_result.significance_level[i])/100)
        print(f"At {sig:.2f} significance level: ", end="")
        if ad_result.statistic < ad_result.critical_values[i]:
            print("Normal")
        else:
            print("Non-normal")

# Generate test data
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
skewed_data = np.random.gamma(2, 2, 1000)

print("Testing Normal Data:")
distribution_testing(normal_data)
print("\nTesting Skewed Data:")
distribution_testing(skewed_data)
```

Slide 15: Additional Resources

*   "A Survey of Probability Distributions with Applications" - arXiv:1907.09952
*   "Modern Statistical Methods for Heavy-Tailed Distributions" - arXiv:2104.12883
*   "Nonparametric Statistical Testing of Distributions" - arXiv:1904.12956
*   "Practical Methods for Fitting Mixture Models" - [https://www.sciencedirect.com/topics/mathematics/mixture-distribution](https://www.sciencedirect.com/topics/mathematics/mixture-distribution)
*   "Computational Methods for Distribution Testing" - [https://dl.acm.org/doi/10.1145/3460120](https://dl.acm.org/doi/10.1145/3460120)

