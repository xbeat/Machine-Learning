## Logistic Distribution Probability Regression and Neural Networks

Slide 1: Introduction to Logistic Distribution

The logistic distribution is a continuous probability distribution characterized by its location parameter μ and scale parameter s. Its probability density function forms a symmetric curve similar to the normal distribution but with heavier tails. Understanding its implementation is crucial for various statistical applications.

```python
import numpy as np
import matplotlib.pyplot as plt

def logistic_pdf(x, mu=0, s=1):
    """
    Calculate the probability density function of logistic distribution
    Args:
        x: input values
        mu: location parameter
        s: scale parameter
    """
    z = (x - mu) / s
    exp_term = np.exp(-z)
    return exp_term / (s * (1 + exp_term)**2)

# Generate sample data
x = np.linspace(-10, 10, 1000)
y = logistic_pdf(x)

# Plot the PDF
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='μ=0, s=1')
plt.plot(x, logistic_pdf(x, mu=2, s=1.5), 'r--', label='μ=2, s=1.5')
plt.title('Logistic Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 2: Cumulative Distribution Function

The cumulative distribution function (CDF) of the logistic distribution represents the probability that a random variable takes a value less than or equal to a specific point. It's crucial for calculating probabilities and quantiles.

```python
def logistic_cdf(x, mu=0, s=1):
    """
    Calculate the cumulative distribution function of logistic distribution
    Args:
        x: input values
        mu: location parameter
        s: scale parameter
    """
    return 1 / (1 + np.exp(-(x - mu) / s))

# Generate sample data
x = np.linspace(-10, 10, 1000)
y = logistic_cdf(x)

# Plot the CDF
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='μ=0, s=1')
plt.plot(x, logistic_cdf(x, mu=2, s=1.5), 'r--', label='μ=2, s=1.5')
plt.title('Logistic Distribution CDF')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Random Number Generation

Generating random numbers from a logistic distribution is essential for simulation studies and bootstrapping. This implementation uses the inverse transform sampling method to generate random samples.

```python
def generate_logistic_samples(size, mu=0, s=1):
    """
    Generate random samples from logistic distribution
    Args:
        size: number of samples
        mu: location parameter
        s: scale parameter
    """
    u = np.random.uniform(0, 1, size)
    return mu + s * np.log(u / (1 - u))

# Generate and visualize samples
samples = generate_logistic_samples(10000)
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
x = np.linspace(-10, 10, 1000)
plt.plot(x, logistic_pdf(x), 'r-', label='PDF')
plt.title('Generated Samples vs. Theoretical PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Print summary statistics
print(f"Sample mean: {np.mean(samples):.4f}")
print(f"Sample variance: {np.var(samples):.4f}")
```

Slide 4: Parameter Estimation

Maximum likelihood estimation (MLE) is used to estimate the parameters of a logistic distribution from observed data. This implementation uses scipy's optimization to find the best parameters.

```python
from scipy.optimize import minimize

def negative_log_likelihood(params, data):
    """
    Calculate negative log-likelihood for logistic distribution
    Args:
        params: [mu, s]
        data: observed samples
    """
    mu, s = params
    z = (data - mu) / s
    return np.sum(np.log(s) + z + 2 * np.log(1 + np.exp(-z)))

# Generate sample data
true_mu, true_s = 2, 1.5
data = generate_logistic_samples(1000, true_mu, true_s)

# Estimate parameters
result = minimize(negative_log_likelihood, x0=[0, 1], args=(data,))
est_mu, est_s = result.x

print(f"True parameters: μ={true_mu}, s={true_s}")
print(f"Estimated parameters: μ={est_mu:.4f}, s={est_s:.4f}")
```

Slide 5: Logistic Regression for Binary Classification

The logistic regression model uses the logistic distribution for binary classification tasks. This implementation showcases a complete example with synthetic data generation, model training, and evaluation metrics for a binary classification problem.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            pred = self.sigmoid(linear_pred)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (pred - y))
            db = (1/n_samples) * np.sum(pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        pred = self.sigmoid(linear_pred)
        return (pred > 0.5).astype(int)

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
```

Slide 6: Quantile Function Implementation

The quantile function, also known as the inverse cumulative distribution function, is essential for generating random numbers and calculating confidence intervals in logistic distribution applications.

```python
def logistic_quantile(p, mu=0, s=1):
    """
    Calculate the quantile function (inverse CDF) of logistic distribution
    Args:
        p: probability values between 0 and 1
        mu: location parameter
        s: scale parameter
    """
    return mu + s * np.log(p / (1 - p))

# Generate example quantiles
probs = np.linspace(0.01, 0.99, 100)
quantiles = logistic_quantile(probs)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(probs, quantiles, 'b-')
plt.title('Logistic Distribution Quantile Function')
plt.xlabel('Probability')
plt.ylabel('Quantile')
plt.grid(True)
plt.show()

# Print some common quantiles
print(f"25th percentile: {logistic_quantile(0.25):.4f}")
print(f"50th percentile: {logistic_quantile(0.50):.4f}")
print(f"75th percentile: {logistic_quantile(0.75):.4f}")
```

Slide 7: Comparing Normal and Logistic Distributions

The logistic distribution shares similarities with the normal distribution but has heavier tails. This implementation compares both distributions and highlights their differences through visualization.

```python
def normal_pdf(x, mu=0, sigma=1):
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

# Generate comparison data
x = np.linspace(-5, 5, 1000)
logistic_y = logistic_pdf(x, mu=0, s=1)
normal_y = normal_pdf(x, mu=0, sigma=1)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(x, logistic_y, 'b-', label='Logistic')
plt.plot(x, normal_y, 'r--', label='Normal')
plt.title('Comparison of Logistic and Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Calculate kurtosis
x_samples = generate_logistic_samples(10000)
logistic_kurtosis = np.mean((x_samples - np.mean(x_samples))**4) / (np.var(x_samples)**2)
print(f"Empirical kurtosis of logistic distribution: {logistic_kurtosis:.4f}")
```

Slide 8: Maximum Likelihood Estimation with Real Data

This implementation demonstrates how to estimate logistic distribution parameters from real data using maximum likelihood estimation and evaluate the goodness of fit.

```python
from scipy import stats
import numpy as np

def fit_logistic_distribution(data):
    """
    Fit logistic distribution to data using MLE
    Returns location and scale parameters
    """
    params = stats.logistic.fit(data)
    return params[0], params[1]  # location, scale

# Generate sample data with noise
true_mu, true_s = 2.5, 1.2
np.random.seed(42)
sample_data = generate_logistic_samples(1000, true_mu, true_s)
sample_data += np.random.normal(0, 0.1, 1000)  # Add noise

# Fit distribution
est_mu, est_s = fit_logistic_distribution(sample_data)

# Evaluate fit
x_fit = np.linspace(min(sample_data), max(sample_data), 100)
y_fit = logistic_pdf(x_fit, est_mu, est_s)

plt.figure(figsize=(10, 6))
plt.hist(sample_data, bins=50, density=True, alpha=0.7, label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted PDF')
plt.title('Fitted Logistic Distribution to Noisy Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated parameters: μ={est_mu:.4f}, s={est_s:.4f}")
```

Slide 9: Logistic Growth Model

The logistic distribution can model population growth with carrying capacity. This implementation shows how to fit and predict population growth using the logistic function.

```python
def logistic_growth(t, K, r, t0):
    """
    Logistic growth function
    Args:
        t: time points
        K: carrying capacity
        r: growth rate
        t0: time at midpoint
    """
    return K / (1 + np.exp(-r * (t - t0)))

# Generate synthetic growth data
t = np.linspace(0, 10, 100)
true_K, true_r, true_t0 = 1000, 0.8, 5
y_true = logistic_growth(t, true_K, true_r, true_t0)
y_noisy = y_true + np.random.normal(0, 20, len(t))

# Fit model using scipy's curve_fit
from scipy.optimize import curve_fit
popt, _ = curve_fit(logistic_growth, t, y_noisy, p0=[1000, 1, 5])

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(t, y_noisy, alpha=0.5, label='Data')
plt.plot(t, logistic_growth(t, *popt), 'r-', label='Fitted curve')
plt.title('Logistic Growth Model Fit')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated parameters: K={popt[0]:.1f}, r={popt[1]:.4f}, t0={popt[2]:.4f}")
```

Slide 10: Confidence Intervals

Computing confidence intervals using the logistic distribution properties helps quantify uncertainty in parameter estimates and predictions.

```python
def logistic_confidence_interval(x, mu, s, confidence=0.95):
    """
    Calculate confidence intervals for logistic distribution
    Args:
        x: data points
        mu: location parameter
        s: scale parameter
        confidence: confidence level
    """
    alpha = 1 - confidence
    z = stats.logistic.ppf(1 - alpha/2)
    margin = z * s / np.sqrt(len(x))
    return mu - margin, mu + margin

# Generate sample data
np.random.seed(42)
sample_size = 1000
data = generate_logistic_samples(sample_size, mu=2, s=1.5)

# Calculate confidence intervals
mu_hat = np.mean(data)
s_hat = np.std(data) * np.pi / np.sqrt(3)
ci_lower, ci_upper = logistic_confidence_interval(data, mu_hat, s_hat)

print(f"Sample mean: {mu_hat:.4f}")
print(f"95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.axvline(ci_lower, color='r', linestyle='--', label='95% CI')
plt.axvline(ci_upper, color='r', linestyle='--')
plt.axvline(mu_hat, color='g', label='Sample mean')
plt.title('Logistic Distribution with Confidence Intervals')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 11: Application in Chess Ratings

Implementation of the logistic distribution for chess rating calculations, following the approach used by FIDE and USCF.

```python
def calculate_expected_score(rating_a, rating_b, K=400):
    """
    Calculate expected score for player A vs player B
    Args:
        rating_a: rating of player A
        rating_b: rating of player B
        K: rating scale factor (400 for FIDE)
    """
    return 1 / (1 + 10**((rating_b - rating_a) / K))

def update_rating(old_rating, expected_score, actual_score, K=20):
    """
    Update player rating based on game outcome
    Args:
        old_rating: current rating
        expected_score: expected score from calculate_expected_score
        actual_score: actual game result (1 for win, 0.5 for draw, 0 for loss)
        K: rating change factor
    """
    return old_rating + K * (actual_score - expected_score)

# Example usage
player_a_rating = 1500
player_b_rating = 1700

# Calculate expected score
expected = calculate_expected_score(player_a_rating, player_b_rating)

# Update ratings after a draw (0.5)
new_rating_a = update_rating(player_a_rating, expected, 0.5)
new_rating_b = update_rating(player_b_rating, 1-expected, 0.5)

print(f"Player A: {player_a_rating} → {new_rating_a:.1f}")
print(f"Player B: {player_b_rating} → {new_rating_b:.1f}")
```

Slide 12: Survival Analysis with Logistic Distribution

Implementation of survival analysis using the logistic distribution to model time-to-event data with censoring.

```python
def survival_function(t, mu, s):
    """
    Calculate survival function S(t) = 1 - F(t)
    where F(t) is the logistic CDF
    """
    return 1 - logistic_cdf(t, mu, s)

def hazard_function(t, mu, s):
    """
    Calculate hazard function h(t) = f(t)/S(t)
    """
    return logistic_pdf(t, mu, s) / survival_function(t, mu, s)

# Generate example data
t = np.linspace(0, 10, 100)
mu, s = 5, 1

# Calculate survival and hazard
survival = survival_function(t, mu, s)
hazard = hazard_function(t, mu, s)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(t, survival, 'b-')
ax1.set_title('Survival Function')
ax1.set_xlabel('Time')
ax1.set_ylabel('S(t)')
ax1.grid(True)

ax2.plot(t, hazard, 'r-')
ax2.set_title('Hazard Function')
ax2.set_xlabel('Time')
ax2.set_ylabel('h(t)')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

Slide 13: Additional Resources

arxiv.org/abs/1908.00441 - "On the Implementation of the Logistic Distribution in Statistical Software" arxiv.org/abs/2003.07528 - "The Logistic Distribution and Its Applications in Machine Learning" arxiv.org/abs/1912.04642 - "Robust Parameter Estimation for the Logistic Distribution" arxiv.org/abs/2105.14045 - "A Comprehensive Review of the Logistic Distribution in Modern Statistics" arxiv.org/abs/1906.08295 - "Applications of the Logistic Distribution in Survival Analysis"

