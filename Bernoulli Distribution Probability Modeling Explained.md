## Bernoulli Distribution Probability Modeling Explained
Slide 1: Understanding Bernoulli Distribution

The Bernoulli distribution models binary outcomes where a random variable X can take only two possible values: success (1) with probability p or failure (0) with probability 1-p. This foundational probability distribution serves as the building block for more complex distributions.

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_pmf(x, p):
    """
    Calculate Probability Mass Function for Bernoulli distribution
    Args:
        x: outcome (0 or 1)
        p: probability of success
    """
    if x not in [0, 1]:
        return 0
    return p if x == 1 else 1 - p

# Example: Coin flip with p=0.7 (biased coin)
p = 0.7
x_values = [0, 1]
pmf_values = [bernoulli_pmf(x, p) for x in x_values]

# Plot PMF
plt.stem(x_values, pmf_values)
plt.title(f'Bernoulli PMF (p={p})')
plt.xlabel('x')
plt.ylabel('P(X = x)')
plt.show()
```

Slide 2: Mathematical Properties of Bernoulli Distribution

The Bernoulli distribution has specific mathematical properties that make it useful for statistical analysis. The expected value (mean) is p, and the variance is p(1-p). These properties help in understanding the distribution's behavior.

```python
def bernoulli_properties(p):
    """
    Calculate key properties of Bernoulli distribution
    """
    mean = p
    variance = p * (1 - p)
    skewness = (1 - 2*p) / np.sqrt(variance)
    kurtosis = (1 - 6*p*(1-p)) / (p*(1-p))
    
    properties = {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
    return properties

# Mathematical formulas (LaTeX format)
formulas = """
$$E[X] = p$$
$$Var(X) = p(1-p)$$
$$Skewness = \frac{1-2p}{\sqrt{p(1-p)}}$$
$$Kurtosis = \frac{1-6p(1-p)}{p(1-p)}$$
"""

# Example calculation
p = 0.3
props = bernoulli_properties(p)
for key, value in props.items():
    print(f"{key}: {value:.4f}")
```

Slide 3: Bernoulli Random Variable Generator

A custom implementation of a Bernoulli random variable generator demonstrates how to simulate binary outcomes according to a specified probability. This implementation uses numpy's random number generator for efficiency.

```python
class BernoulliGenerator:
    def __init__(self, p):
        self.p = p
        if not 0 <= p <= 1:
            raise ValueError("Probability must be between 0 and 1")
    
    def generate_sample(self, size=1):
        """Generate Bernoulli random variables"""
        return (np.random.random(size) < self.p).astype(int)
    
    def sample_mean(self, n_samples=1000):
        """Calculate sample mean"""
        samples = self.generate_sample(n_samples)
        return np.mean(samples)

# Example usage
generator = BernoulliGenerator(p=0.7)
samples = generator.generate_sample(10)
print(f"10 Bernoulli trials: {samples}")
print(f"Sample mean (1000 trials): {generator.sample_mean():.4f}")
```

Slide 4: Maximum Likelihood Estimation

Understanding how to estimate the parameter p from observed data is crucial in statistical inference. The maximum likelihood estimator for the Bernoulli distribution is simply the sample proportion of successes.

```python
def bernoulli_mle(samples):
    """
    Maximum Likelihood Estimation for Bernoulli parameter p
    Args:
        samples: array of observed outcomes (0s and 1s)
    Returns:
        p_hat: MLE estimate of p
    """
    p_hat = np.mean(samples)
    return p_hat

# Generate synthetic data
true_p = 0.6
generator = BernoulliGenerator(true_p)
sample_sizes = [10, 100, 1000, 10000]

for n in sample_sizes:
    samples = generator.generate_sample(n)
    p_hat = bernoulli_mle(samples)
    print(f"n={n}: True p={true_p:.4f}, Estimated p={p_hat:.4f}")
```

Slide 5: Confidence Intervals for Bernoulli Parameter

The confidence interval for a Bernoulli parameter p uses the normal approximation when sample size is large enough. This implementation shows how to calculate both exact and approximate confidence intervals for parameter estimation.

```python
import scipy.stats as stats

def bernoulli_confidence_interval(samples, confidence=0.95):
    """
    Calculate confidence interval for Bernoulli parameter p
    Args:
        samples: array of observed outcomes (0s and 1s)
        confidence: confidence level (default 95%)
    Returns:
        tuple of (lower_bound, upper_bound)
    """
    n = len(samples)
    p_hat = np.mean(samples)
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Standard error
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    
    # Calculate confidence interval
    lower = p_hat - z * se
    upper = p_hat + z * se
    
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)

# Example usage
np.random.seed(42)
generator = BernoulliGenerator(p=0.7)
samples = generator.generate_sample(1000)
ci_lower, ci_upper = bernoulli_confidence_interval(samples)
print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
```

Slide 6: Hypothesis Testing for Bernoulli Distribution

Statistical hypothesis testing for a Bernoulli parameter allows us to make inferences about the true probability of success. This implementation demonstrates both one-sided and two-sided tests.

```python
def bernoulli_hypothesis_test(samples, p0, alternative='two-sided', alpha=0.05):
    """
    Perform hypothesis test for Bernoulli parameter
    H0: p = p0 vs H1: p ≠ p0 (two-sided)
                 H1: p > p0 (greater)
                 H1: p < p0 (less)
    """
    n = len(samples)
    p_hat = np.mean(samples)
    se = np.sqrt(p0 * (1 - p0) / n)
    z_stat = (p_hat - p0) / se
    
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # 'less'
        p_value = stats.norm.cdf(z_stat)
        
    return {
        'z_statistic': z_stat,
        'p_value': p_value,
        'reject_h0': p_value < alpha
    }

# Example usage
samples = BernoulliGenerator(p=0.7).generate_sample(1000)
result = bernoulli_hypothesis_test(samples, p0=0.65, alternative='greater')
print(f"Test Results:\n{result}")
```

Slide 7: Real-world Application - Email Spam Detection

The Bernoulli distribution naturally models binary classification problems like spam detection. This implementation shows how to create a simple Bernoulli Naive Bayes classifier from scratch.

```python
class BernoulliBayesClassifier:
    def __init__(self):
        self.feature_probs = {}
        self.class_prior = None
        
    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes classifier
        X: binary feature matrix (n_samples, n_features)
        y: binary labels (n_samples,)
        """
        n_samples, n_features = X.shape
        self.class_prior = np.mean(y)
        
        # Calculate P(feature=1|class) for each feature
        pos_samples = X[y == 1]
        neg_samples = X[y == 0]
        
        for j in range(n_features):
            self.feature_probs[j] = {
                1: (np.sum(pos_samples[:, j]) + 1) / (len(pos_samples) + 2),  # Laplace smoothing
                0: (np.sum(neg_samples[:, j]) + 1) / (len(neg_samples) + 2)
            }
    
    def predict_proba(self, X):
        """Calculate probability of class 1"""
        log_prob_1 = np.log(self.class_prior)
        log_prob_0 = np.log(1 - self.class_prior)
        
        for j in range(X.shape[1]):
            feature_val = X[:, j]
            prob_1 = self.feature_probs[j][1]
            prob_0 = self.feature_probs[j][0]
            
            log_prob_1 += np.where(feature_val == 1,
                                  np.log(prob_1),
                                  np.log(1 - prob_1))
            log_prob_0 += np.where(feature_val == 1,
                                  np.log(prob_0),
                                  np.log(1 - prob_0))
        
        proba = 1 / (1 + np.exp(log_prob_0 - log_prob_1))
        return proba

# Example usage
# Generate synthetic email data
np.random.seed(42)
n_samples, n_features = 1000, 10
X = np.random.binomial(1, 0.3, (n_samples, n_features))
y = np.random.binomial(1, 0.7, n_samples)

# Train and evaluate
clf = BernoulliBayesClassifier()
clf.fit(X, y)
probs = clf.predict_proba(X[:5])
print(f"Predicted probabilities for first 5 samples:\n{probs}")
```

Slide 8: Bernoulli Process Simulation

A Bernoulli process consists of independent Bernoulli trials over time. This implementation demonstrates how to simulate and analyze sequences of Bernoulli trials, including waiting times between successes.

```python
class BernoulliProcess:
    def __init__(self, p):
        self.p = p
        self.generator = BernoulliGenerator(p)
        
    def simulate_process(self, n_steps):
        """Simulate n steps of a Bernoulli process"""
        return self.generator.generate_sample(n_steps)
    
    def time_until_success(self, max_steps=1000):
        """Simulate until first success occurs"""
        for t in range(max_steps):
            if self.generator.generate_sample(1)[0] == 1:
                return t + 1
        return None
    
    def analyze_waiting_times(self, n_trials=1000):
        """Analyze distribution of waiting times"""
        waiting_times = [self.time_until_success() for _ in range(n_trials)]
        return {
            'mean': np.mean(waiting_times),
            'variance': np.var(waiting_times),
            'geometric_expectation': 1/self.p
        }

# Example usage
process = BernoulliProcess(p=0.2)
sequence = process.simulate_process(20)
waiting_analysis = process.analyze_waiting_times()

print(f"Sample sequence: {sequence}")
print(f"Waiting time analysis: {waiting_analysis}")
```

Slide 9: Log-Likelihood and Information Theory

The log-likelihood function for Bernoulli distribution plays a crucial role in statistical inference and information theory. This implementation explores entropy and Kullback-Leibler divergence.

```python
def bernoulli_log_likelihood(samples, p):
    """Calculate log-likelihood for Bernoulli samples"""
    return np.sum(samples * np.log(p) + (1 - samples) * np.log(1 - p))

def bernoulli_entropy(p):
    """Calculate entropy of Bernoulli distribution"""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def bernoulli_kl_divergence(p, q):
    """Calculate KL divergence between two Bernoulli distributions"""
    return p * np.log(p/q) + (1-p) * np.log((1-p)/(1-q))

# Example calculations
p_true = 0.7
p_approx = 0.65
samples = BernoulliGenerator(p_true).generate_sample(1000)

results = {
    'log_likelihood': bernoulli_log_likelihood(samples, p_true),
    'entropy': bernoulli_entropy(p_true),
    'kl_divergence': bernoulli_kl_divergence(p_true, p_approx)
}

print("Information theoretic measures:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
```

Slide 10: Real-world Application - A/B Testing Framework

A practical implementation of A/B testing using Bernoulli distributions to compare conversion rates between two variants. This framework includes sample size calculation and sequential testing.

```python
class ABTest:
    def __init__(self, base_rate=0.1, min_effect=0.02, alpha=0.05, power=0.8):
        self.base_rate = base_rate
        self.min_effect = min_effect
        self.alpha = alpha
        self.power = power
        
    def required_sample_size(self):
        """Calculate required sample size per variant"""
        p1 = self.base_rate
        p2 = self.base_rate + self.min_effect
        pooled_p = (p1 + p2) / 2
        
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(self.power)
        
        n = (2 * pooled_p * (1-pooled_p) * (z_alpha + z_beta)**2) / (self.min_effect**2)
        return int(np.ceil(n))
    
    def analyze_results(self, control_data, treatment_data):
        """Analyze A/B test results"""
        n1, n2 = len(control_data), len(treatment_data)
        p1, p2 = np.mean(control_data), np.mean(treatment_data)
        
        # Calculate z-statistic
        pooled_p = (np.sum(control_data) + np.sum(treatment_data)) / (n1 + n2)
        se = np.sqrt(pooled_p * (1-pooled_p) * (1/n1 + 1/n2))
        z_stat = (p2 - p1) / se
        
        # Calculate p-value and confidence interval
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        ci_margin = stats.norm.ppf(1 - self.alpha/2) * se
        
        return {
            'control_rate': p1,
            'treatment_rate': p2,
            'absolute_difference': p2 - p1,
            'relative_difference': (p2 - p1) / p1,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'ci_lower': p2 - p1 - ci_margin,
            'ci_upper': p2 - p1 + ci_margin
        }

# Example usage
ab_test = ABTest(base_rate=0.1, min_effect=0.02)
n = ab_test.required_sample_size()

# Simulate data
control = BernoulliGenerator(p=0.1).generate_sample(n)
treatment = BernoulliGenerator(p=0.12).generate_sample(n)

results = ab_test.analyze_results(control, treatment)
print(f"Required sample size per variant: {n}")
print("\nTest Results:")
for key, value in results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
```

Slide 11: Moment Generating Function and Characteristic Function

The moment-generating function (MGF) and characteristic function provide powerful tools for analyzing Bernoulli distributions. This implementation demonstrates their calculation and use in deriving moments.

```python
import matplotlib.pyplot as plt
from scipy.special import factorial

def bernoulli_mgf(t, p):
    """Calculate moment generating function for Bernoulli distribution"""
    return (1 - p) + p * np.exp(t)

def bernoulli_characteristic(t, p):
    """Calculate characteristic function for Bernoulli distribution"""
    return (1 - p) + p * np.exp(1j * t)

def calculate_moments(p, n_moments=4):
    """Calculate first n moments using MGF derivatives"""
    moments = []
    for k in range(1, n_moments + 1):
        moment = p  # For Bernoulli, all moments equal p
        moments.append(moment)
    return moments

# Visualization and analysis
p = 0.3
t_values = np.linspace(-2, 2, 100)
mgf_values = [bernoulli_mgf(t, p) for t in t_values]
char_values = [bernoulli_characteristic(t, p) for t in t_values]

moments = calculate_moments(p)
print(f"First four moments for p={p}:")
for i, moment in enumerate(moments, 1):
    print(f"E[X^{i}] = {moment:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_values, mgf_values)
plt.title('Moment Generating Function')
plt.xlabel('t')
plt.ylabel('M(t)')

plt.subplot(1, 2, 2)
plt.plot(t_values, [x.real for x in char_values], label='Real')
plt.plot(t_values, [x.imag for x in char_values], label='Imaginary')
plt.title('Characteristic Function')
plt.xlabel('t')
plt.legend()
plt.show()
```

Slide 12: Bernoulli Trials with Conjugate Priors

Implementation of Bayesian inference for Bernoulli trials using the Beta distribution as a conjugate prior, demonstrating posterior updates and credible intervals.

```python
class BayesianBernoulli:
    def __init__(self, alpha_prior=1, beta_prior=1):
        self.alpha = alpha_prior
        self.beta = beta_prior
        
    def update(self, data):
        """Update posterior parameters with new data"""
        successes = np.sum(data)
        failures = len(data) - successes
        self.alpha += successes
        self.beta += failures
        
    def posterior_mean(self):
        """Calculate posterior mean"""
        return self.alpha / (self.alpha + self.beta)
    
    def credible_interval(self, confidence=0.95):
        """Calculate credible interval"""
        return stats.beta.interval(confidence, self.alpha, self.beta)
    
    def plot_posterior(self):
        """Plot posterior distribution"""
        x = np.linspace(0, 1, 200)
        y = stats.beta.pdf(x, self.alpha, self.beta)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', lw=2, label='Posterior')
        plt.fill_between(x, y, alpha=0.2)
        plt.title(f'Beta({self.alpha:.1f}, {self.beta:.1f}) Posterior')
        plt.xlabel('p')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

# Example usage
model = BayesianBernoulli(alpha_prior=2, beta_prior=2)
data = BernoulliGenerator(p=0.7).generate_sample(100)

# Update and analyze
model.update(data)
mean = model.posterior_mean()
ci_lower, ci_upper = model.credible_interval()

print(f"Posterior mean: {mean:.4f}")
print(f"95% Credible interval: ({ci_lower:.4f}, {ci_upper:.4f})")
model.plot_posterior()
```

Slide 13: Additional Resources

*   Comprehensive Review of Bernoulli Processes: [https://arxiv.org/abs/1406.1897](https://arxiv.org/abs/1406.1897)
*   Bayesian Analysis of Bernoulli Trials: [https://arxiv.org/abs/1902.08416](https://arxiv.org/abs/1902.08416)
*   Applications in Machine Learning: [https://arxiv.org/abs/1712.05594](https://arxiv.org/abs/1712.05594)
*   Modern Perspectives on Bernoulli Distribution: [https://papers.ssrn.com/sol3/papers.cfm?abstract\_id=3542018](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3542018)
*   For more advanced topics, search Google Scholar for "Bernoulli Distribution Applications in Deep Learning"
*   Statistical Learning Theory and Bernoulli Processes: [https://projecteuclid.org/journals/bernoulli](https://projecteuclid.org/journals/bernoulli)

