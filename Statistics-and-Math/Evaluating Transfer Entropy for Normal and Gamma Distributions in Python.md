## Evaluating Transfer Entropy for Normal and Gamma Distributions in Python
Slide 1: Introduction to Transfer Entropy

Transfer entropy is a measure of directed information transfer between two random processes. It quantifies the amount of uncertainty reduced in future values of one process by knowing the past values of another process, beyond the uncertainty already reduced by knowing its own past.

```python
import numpy as np
from scipy import stats

def transfer_entropy(source, target, k=1, l=1):
    """
    Calculate transfer entropy from source to target.
    k: history length for target
    l: history length for source
    """
    joint = np.array(list(zip(target[k:], target[:len(target)-k], source[:len(source)-l])))
    p_joint = stats.gaussian_kde(joint.T)(joint.T)
    p_cond_target = stats.gaussian_kde(joint[:, :2].T)(joint[:, :2].T)
    p_cond_both = stats.gaussian_kde(joint[:, [0, 1, 2]].T)(joint[:, [0, 1, 2]].T)
    return np.mean(np.log2(p_cond_both / p_cond_target))

# Example usage
np.random.seed(0)
source = np.random.normal(0, 1, 1000)
target = np.roll(source, 1) + np.random.normal(0, 0.1, 1000)

te = transfer_entropy(source, target)
print(f"Transfer entropy: {te:.4f}")
```

Slide 2: Normal Distribution

The normal distribution, also known as the Gaussian distribution, is a continuous probability distribution characterized by its bell-shaped curve. It is symmetric about the mean and is fully defined by two parameters: the mean (μ) and the standard deviation (σ).

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data points
x = np.linspace(-5, 5, 1000)

# Create normal distributions
mu1, sigma1 = 0, 1
mu2, sigma2 = 1, 1.5

y1 = norm.pdf(x, mu1, sigma1)
y2 = norm.pdf(x, mu2, sigma2)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=f'μ={mu1}, σ={sigma1}')
plt.plot(x, y2, label=f'μ={mu2}, σ={sigma2}')
plt.title('Normal Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 3: Gamma Distribution

The gamma distribution is a continuous probability distribution with two parameters: shape (k) and scale (θ). It is often used to model waiting times and is a generalization of the exponential and chi-squared distributions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Generate data points
x = np.linspace(0, 20, 1000)

# Create gamma distributions
k1, theta1 = 2, 2
k2, theta2 = 5, 1

y1 = gamma.pdf(x, k1, scale=theta1)
y2 = gamma.pdf(x, k2, scale=theta2)

# Plot the distributions
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=f'k={k1}, θ={theta1}')
plt.plot(x, y2, label=f'k={k2}, θ={theta2}')
plt.title('Gamma Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 4: Generating Normal and Gamma Distributions

To evaluate transfer entropy, we first need to generate data from normal and gamma distributions. Here's how we can create synthetic data using NumPy:

```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate normal distribution
mu, sigma = 0, 1
normal_data = np.random.normal(mu, sigma, 1000)

# Generate gamma distribution
k, theta = 2, 2
gamma_data = np.random.gamma(k, theta, 1000)

print("Normal distribution statistics:")
print(f"Mean: {np.mean(normal_data):.4f}")
print(f"Standard deviation: {np.std(normal_data):.4f}")

print("\nGamma distribution statistics:")
print(f"Mean: {np.mean(gamma_data):.4f}")
print(f"Standard deviation: {np.std(gamma_data):.4f}")
```

Slide 5: Calculating Transfer Entropy

Now that we have our data, let's calculate the transfer entropy between the normal and gamma distributions. We'll use the `transfer_entropy` function defined earlier:

```python
def transfer_entropy(source, target, k=1, l=1):
    joint = np.array(list(zip(target[k:], target[:len(target)-k], source[:len(source)-l])))
    p_joint = stats.gaussian_kde(joint.T)(joint.T)
    p_cond_target = stats.gaussian_kde(joint[:, :2].T)(joint[:, :2].T)
    p_cond_both = stats.gaussian_kde(joint[:, [0, 1, 2]].T)(joint[:, [0, 1, 2]].T)
    return np.mean(np.log2(p_cond_both / p_cond_target))

# Calculate transfer entropy
te_normal_to_gamma = transfer_entropy(normal_data, gamma_data)
te_gamma_to_normal = transfer_entropy(gamma_data, normal_data)

print(f"Transfer entropy (Normal to Gamma): {te_normal_to_gamma:.4f}")
print(f"Transfer entropy (Gamma to Normal): {te_gamma_to_normal:.4f}")
```

Slide 6: Interpreting Transfer Entropy Results

The transfer entropy values we calculated provide insights into the information flow between the normal and gamma distributions. A higher value indicates stronger information transfer, while a value close to zero suggests minimal information transfer.

```python
import matplotlib.pyplot as plt

# Visualize the transfer entropy results
plt.figure(figsize=(10, 6))
plt.bar(['Normal to Gamma', 'Gamma to Normal'], [te_normal_to_gamma, te_gamma_to_normal])
plt.title('Transfer Entropy Between Normal and Gamma Distributions')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()

# Interpret the results
if te_normal_to_gamma > te_gamma_to_normal:
    print("The normal distribution provides more information about the gamma distribution than vice versa.")
elif te_normal_to_gamma < te_gamma_to_normal:
    print("The gamma distribution provides more information about the normal distribution than vice versa.")
else:
    print("The information transfer between the normal and gamma distributions is symmetric.")
```

Slide 7: Time-Lagged Transfer Entropy

Transfer entropy can also be calculated with time lags to explore delayed information transfer. Let's implement a function to calculate time-lagged transfer entropy:

```python
def time_lagged_transfer_entropy(source, target, lag, k=1, l=1):
    source_lagged = np.roll(source, lag)
    return transfer_entropy(source_lagged, target, k, l)

# Calculate time-lagged transfer entropy for different lags
lags = range(-10, 11)
te_values = [time_lagged_transfer_entropy(normal_data, gamma_data, lag) for lag in lags]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(lags, te_values)
plt.title('Time-Lagged Transfer Entropy (Normal to Gamma)')
plt.xlabel('Time Lag')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(True)
plt.show()

# Find the lag with maximum transfer entropy
max_lag = lags[np.argmax(te_values)]
print(f"Maximum transfer entropy occurs at lag {max_lag}")
```

Slide 8: Comparing Transfer Entropy with Correlation

While transfer entropy measures directed information flow, correlation measures the linear relationship between variables. Let's compare these two measures:

```python
from scipy.stats import pearsonr

# Calculate Pearson correlation
correlation, _ = pearsonr(normal_data, gamma_data)

# Calculate transfer entropy in both directions
te_normal_to_gamma = transfer_entropy(normal_data, gamma_data)
te_gamma_to_normal = transfer_entropy(gamma_data, normal_data)

print(f"Pearson correlation: {correlation:.4f}")
print(f"Transfer entropy (Normal to Gamma): {te_normal_to_gamma:.4f}")
print(f"Transfer entropy (Gamma to Normal): {te_gamma_to_normal:.4f}")

# Visualize the comparison
plt.figure(figsize=(10, 6))
plt.bar(['Correlation', 'TE (N to G)', 'TE (G to N)'], [abs(correlation), te_normal_to_gamma, te_gamma_to_normal])
plt.title('Comparison of Correlation and Transfer Entropy')
plt.ylabel('Magnitude')
plt.grid(axis='y')
plt.show()
```

Slide 9: Real-Life Example: Stock Market Analysis

Let's apply transfer entropy to analyze information flow between two stock prices. We'll use Yahoo Finance to fetch real stock data:

```python
import yfinance as yf
import pandas as pd

# Fetch stock data
apple = yf.Ticker("AAPL")
google = yf.Ticker("GOOGL")

start_date = "2022-01-01"
end_date = "2023-01-01"

apple_data = apple.history(start=start_date, end=end_date)['Close']
google_data = google.history(start=start_date, end=end_date)['Close']

# Calculate daily returns
apple_returns = apple_data.pct_change().dropna()
google_returns = google_data.pct_change().dropna()

# Calculate transfer entropy
te_apple_to_google = transfer_entropy(apple_returns, google_returns)
te_google_to_apple = transfer_entropy(google_returns, apple_returns)

print(f"Transfer entropy (Apple to Google): {te_apple_to_google:.4f}")
print(f"Transfer entropy (Google to Apple): {te_google_to_apple:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['Apple to Google', 'Google to Apple'], [te_apple_to_google, te_google_to_apple])
plt.title('Transfer Entropy Between Apple and Google Stock Returns')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()
```

Slide 10: Real-Life Example: Climate Data Analysis

Another application of transfer entropy is in climate data analysis. Let's examine the information flow between temperature and humidity:

```python
import pandas as pd
import numpy as np

# Generate synthetic climate data
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
temperature = np.random.normal(20, 5, len(dates)) + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
humidity = 50 + 0.5 * temperature + np.random.normal(0, 5, len(dates))

climate_data = pd.DataFrame({"Temperature": temperature, "Humidity": humidity}, index=dates)

# Calculate transfer entropy
te_temp_to_humid = transfer_entropy(climate_data['Temperature'], climate_data['Humidity'])
te_humid_to_temp = transfer_entropy(climate_data['Humidity'], climate_data['Temperature'])

print(f"Transfer entropy (Temperature to Humidity): {te_temp_to_humid:.4f}")
print(f"Transfer entropy (Humidity to Temperature): {te_humid_to_temp:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['Temperature to Humidity', 'Humidity to Temperature'], [te_temp_to_humid, te_humid_to_temp])
plt.title('Transfer Entropy in Climate Data')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()
```

Slide 11: Conditional Transfer Entropy

Conditional transfer entropy measures the information flow from one variable to another, given a third variable. This can help identify indirect influences in complex systems:

```python
def conditional_transfer_entropy(source, target, condition, k=1, l=1, m=1):
    joint = np.array(list(zip(target[k:], target[:len(target)-k], source[:len(source)-l], condition[:len(condition)-m])))
    p_joint = stats.gaussian_kde(joint.T)(joint.T)
    p_cond_target = stats.gaussian_kde(joint[:, :3].T)(joint[:, :3].T)
    p_cond_all = stats.gaussian_kde(joint.T)(joint.T)
    return np.mean(np.log2(p_cond_all * p_cond_target[:, :2] / (p_cond_target * p_joint[:, :3])))

# Generate synthetic data
np.random.seed(42)
x = np.random.normal(0, 1, 1000)
y = 0.5 * x + np.random.normal(0, 0.5, 1000)
z = 0.3 * x + 0.7 * y + np.random.normal(0, 0.3, 1000)

# Calculate conditional transfer entropy
cte_x_to_z_given_y = conditional_transfer_entropy(x, z, y)
cte_y_to_z_given_x = conditional_transfer_entropy(y, z, x)

print(f"Conditional TE (X to Z given Y): {cte_x_to_z_given_y:.4f}")
print(f"Conditional TE (Y to Z given X): {cte_y_to_z_given_x:.4f}")
```

Slide 12: Transfer Entropy in Time Series Analysis

Transfer entropy is particularly useful in time series analysis for detecting causal relationships. Let's apply it to a simple autoregressive model:

```python
def generate_ar_process(n, a):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i-1] + np.random.normal(0, 1)
    return x

# Generate two AR(1) processes
n = 1000
x = generate_ar_process(n, 0.8)
y = generate_ar_process(n, 0.6)

# Introduce causal relationship: y affects x
x[1:] += 0.3 * y[:-1]

# Calculate transfer entropy
te_x_to_y = transfer_entropy(x, y)
te_y_to_x = transfer_entropy(y, x)

print(f"Transfer entropy (X to Y): {te_x_to_y:.4f}")
print(f"Transfer entropy (Y to X): {te_y_to_x:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.bar(['X to Y', 'Y to X'], [te_x_to_y, te_y_to_x])
plt.title('Transfer Entropy in AR Processes')
plt.ylabel('Transfer Entropy (bits)')
plt.grid(axis='y')
plt.show()
```

Slide 13: Limitations and Considerations

While transfer entropy is a powerful tool for analyzing information flow, it has some limitations and considerations:

1. Computational complexity: Calculating transfer entropy can be computationally expensive, especially for large datasets.
2. Data requirements: Accurate estimation requires a sufficient amount of data.
3. Nonlinearity: Transfer entropy can capture nonlinear relationships, but interpreting results can be challenging.
4. Parameter selection: Choosing appropriate history lengths (k and l) can affect results.

To address some of these issues, we can use techniques such as bootstrapping to estimate confidence intervals:

```python
import numpy as np
from scipy import stats

def bootstrap_transfer_entropy(source, target, n_bootstrap=1000, k=1, l=1):
    original_te = transfer_entropy(source, target, k, l)
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(source), len(source), replace=True)
        boot_source = source[indices]
        boot_target = target[indices]
        bootstrap_samples.append(transfer_entropy(boot_source, boot_target, k, l))
    
    ci_lower, ci_upper = np.percentile(bootstrap_samples, [2.5, 97.5])
    return original_te, ci_lower, ci_upper

# Example usage
np.random.seed(42)
source = np.random.normal(0, 1, 1000)
target = 0.5 * source + np.random.normal(0, 0.5, 1000)

te, ci_lower, ci_upper = bootstrap_transfer_entropy(source, target)
print(f"Transfer Entropy: {te:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

Slide 14: Surrogate Data Testing

To determine if the observed transfer entropy is statistically significant, we can use surrogate data testing. This involves creating randomized versions of the original data and comparing the transfer entropy:

```python
def surrogate_data_test(source, target, n_surrogates=1000, k=1, l=1):
    original_te = transfer_entropy(source, target, k, l)
    surrogate_te = []
    
    for _ in range(n_surrogates):
        surrogate_source = np.random.permutation(source)
        surrogate_te.append(transfer_entropy(surrogate_source, target, k, l))
    
    p_value = np.mean(np.array(surrogate_te) >= original_te)
    return original_te, p_value

# Example usage
np.random.seed(42)
source = np.random.normal(0, 1, 1000)
target = 0.5 * source + np.random.normal(0, 0.5, 1000)

te, p_value = surrogate_data_test(source, target)
print(f"Transfer Entropy: {te:.4f}")
print(f"p-value: {p_value:.4f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.hist(surrogate_te, bins=30, edgecolor='black')
plt.axvline(original_te, color='red', linestyle='dashed', linewidth=2)
plt.title('Surrogate Data Test for Transfer Entropy')
plt.xlabel('Transfer Entropy')
plt.ylabel('Frequency')
plt.legend(['Original TE', 'Surrogate TE'])
plt.show()
```

Slide 15: Additional Resources

For those interested in delving deeper into transfer entropy and its applications, here are some valuable resources:

1. Schreiber, T. (2000). "Measuring Information Transfer". Physical Review Letters, 85(2), 461-464. ArXiv: [https://arxiv.org/abs/nlin/0001042](https://arxiv.org/abs/nlin/0001042)
2. Lizier, J. T. (2014). "JIDT: An information-theoretic toolkit for studying the dynamics of complex systems". Frontiers in Robotics and AI, 1, 11. ArXiv: [https://arxiv.org/abs/1408.3270](https://arxiv.org/abs/1408.3270)
3. Bossomaier, T., Barnett, L., Harré, M., & Lizier, J. T. (2016). "An Introduction to Transfer Entropy: Information Flow in Complex Systems". Springer International Publishing. (Book)
4. Vicente, R., Wibral, M., Lindner, M., & Pipa, G. (2011). "Transfer entropy—a model-free measure of effective connectivity for the neurosciences". Journal of Computational Neuroscience, 30(1), 45-67. ArXiv: [https://arxiv.org/abs/0902.3616](https://arxiv.org/abs/0902.3616)

These resources provide a mix of theoretical foundations and practical applications of transfer entropy in various fields, from physics to neuroscience.

