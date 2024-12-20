## Casual Modeling with Stationary Diffusions in Python
Slide 1: Introduction to Casual Modeling with Stationary Diffusions

Casual modeling with stationary diffusions is a powerful technique for analyzing and simulating complex systems. This approach combines the principles of causal inference with the mathematical framework of diffusion processes, allowing us to model and understand the relationships between variables in a system that evolves over time.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def stationary_diffusion(n_steps, mu, sigma):
    return np.cumsum(norm.rvs(loc=mu, scale=sigma, size=n_steps))

x = np.linspace(0, 100, 1000)
y = stationary_diffusion(1000, 0, 0.1)

plt.plot(x, y)
plt.title("Simple Stationary Diffusion Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```

Slide 2: Setting Up the Environment

Before we dive into casual modeling with stationary diffusions, it's essential to set up our Python environment with the necessary libraries. We'll be using NumPy for numerical computations, SciPy for statistical functions, and Matplotlib for visualization.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
print("SciPy version:", scipy.__version__)
print("Matplotlib version:", matplotlib.__version__)

# Check if we can create a simple plot
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Test Plot")
plt.show()
```

Slide 3: Defining a Stationary Diffusion Process

A stationary diffusion process is a continuous-time stochastic process where the probability distribution of the process at any fixed time interval is the same for all time points. In Python, we can simulate this using the Wiener process, also known as Brownian motion.

```python
def stationary_diffusion(n_steps, dt=0.1, mu=0, sigma=1):
    """
    Generate a stationary diffusion process.
    
    :param n_steps: Number of steps
    :param dt: Time step
    :param mu: Drift parameter
    :param sigma: Diffusion parameter
    :return: Array of process values
    """
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    W = np.cumsum(dW)
    return mu * np.arange(n_steps) * dt + sigma * W

# Generate and plot a sample path
t = np.linspace(0, 10, 1000)
X = stationary_diffusion(1000)
plt.plot(t, X)
plt.title("Sample Path of a Stationary Diffusion Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
```

Slide 4: Implementing a Simple Causal Model

Let's implement a simple causal model using a stationary diffusion process. We'll create two variables, X and Y, where X influences Y with some time delay.

```python
def causal_model(n_steps, delay, influence_strength):
    X = stationary_diffusion(n_steps)
    Y = np.zeros(n_steps)
    
    for i in range(delay, n_steps):
        Y[i] = 0.8 * Y[i-1] + influence_strength * X[i-delay] + np.random.normal(0, 0.1)
    
    return X, Y

X, Y = causal_model(1000, delay=50, influence_strength=0.5)

plt.figure(figsize=(12, 6))
plt.plot(X, label='X')
plt.plot(Y, label='Y')
plt.title("Causal Relationship between X and Y")
plt.legend()
plt.show()
```

Slide 5: Analyzing Causal Relationships

To analyze causal relationships in our model, we can use techniques like cross-correlation and Granger causality. Let's implement a function to compute the cross-correlation between two time series.

```python
def cross_correlation(x, y, max_lag):
    corr = np.correlate(x - x.mean(), y - y.mean(), mode='full')
    corr = corr[len(corr)//2:]
    corr /= np.sqrt(np.sum((x - x.mean())**2) * np.sum((y - y.mean())**2))
    lags = np.arange(0, min(max_lag, len(corr)))
    return lags, corr[:max_lag]

X, Y = causal_model(1000, delay=50, influence_strength=0.5)
lags, corr = cross_correlation(X, Y, 100)

plt.plot(lags, corr)
plt.title("Cross-correlation between X and Y")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.show()
```

Slide 6: Implementing Granger Causality

Granger causality is a statistical concept of causality based on prediction. If a signal X1 "Granger-causes" a signal X2, then past values of X1 should contain information that helps predict X2 above and beyond the information contained in past values of X2 alone.

```python
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality(x, y, max_lag):
    data = np.column_stack((y, x))
    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    
    p_values = [round(result[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
    return p_values

X, Y = causal_model(1000, delay=50, influence_strength=0.5)
p_values = granger_causality(X, Y, 10)

plt.plot(range(1, 11), p_values, marker='o')
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title("Granger Causality Test: X -> Y")
plt.xlabel("Lag")
plt.ylabel("p-value")
plt.show()
```

Slide 7: Introducing Time-Varying Causality

In real-world scenarios, causal relationships may change over time. Let's modify our model to incorporate time-varying causality.

```python
def time_varying_causal_model(n_steps, influence_func):
    X = stationary_diffusion(n_steps)
    Y = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        influence = influence_func(i)
        Y[i] = 0.8 * Y[i-1] + influence * X[i-1] + np.random.normal(0, 0.1)
    
    return X, Y

def influence_function(t):
    return 0.5 * np.sin(t / 100) + 0.5

X, Y = time_varying_causal_model(1000, influence_function)

plt.figure(figsize=(12, 6))
plt.plot(X, label='X')
plt.plot(Y, label='Y')
plt.title("Time-Varying Causal Relationship")
plt.legend()
plt.show()
```

Slide 8: Detecting Time-Varying Causality

To detect time-varying causality, we can use rolling window analysis or more advanced techniques like wavelet-based methods. Let's implement a simple rolling window Granger causality test.

```python
def rolling_granger_causality(x, y, window_size, max_lag):
    n = len(x)
    p_values = np.zeros(n - window_size + 1)
    
    for i in range(n - window_size + 1):
        x_window = x[i:i+window_size]
        y_window = y[i:i+window_size]
        p_values[i] = granger_causality(x_window, y_window, max_lag)[0]
    
    return p_values

X, Y = time_varying_causal_model(1000, influence_function)
p_values = rolling_granger_causality(X, Y, 200, 5)

plt.plot(range(200, 1001), p_values)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title("Rolling Window Granger Causality Test")
plt.xlabel("Time")
plt.ylabel("p-value")
plt.show()
```

Slide 9: Multivariate Causal Modeling

Real-world systems often involve multiple variables. Let's extend our model to incorporate multiple causal relationships.

```python
def multivariate_causal_model(n_steps, n_vars):
    X = np.zeros((n_steps, n_vars))
    for i in range(n_vars):
        X[:, i] = stationary_diffusion(n_steps)
    
    Y = np.zeros(n_steps)
    for i in range(1, n_steps):
        Y[i] = 0.8 * Y[i-1] + np.sum(0.1 * X[i-1, :]) + np.random.normal(0, 0.1)
    
    return X, Y

X, Y = multivariate_causal_model(1000, 3)

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(X[:, i], label=f'X{i+1}')
plt.plot(Y, label='Y', linewidth=2)
plt.title("Multivariate Causal Model")
plt.legend()
plt.show()
```

Slide 10: Causal Discovery in Multivariate Systems

Causal discovery aims to infer the causal structure from observational data. Let's implement a simple constraint-based algorithm for causal discovery.

```python
def partial_correlation(X, Y, Z):
    XZ = np.column_stack((X, Z))
    YZ = np.column_stack((Y, Z))
    resX = np.linalg.lstsq(XZ, Y, rcond=None)[0]
    resY = np.linalg.lstsq(YZ, X, rcond=None)[0]
    return np.corrcoef(resX[:, 0], resY[:, 0])[0, 1]

def pc_algorithm(data, alpha=0.05):
    n_vars = data.shape[1]
    G = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if abs(np.corrcoef(data[:, i], data[:, j])[0, 1]) < alpha:
                G[i, j] = G[j, i] = 0
    
    return G

X, Y = multivariate_causal_model(1000, 3)
data = np.column_stack((X, Y.reshape(-1, 1)))
G = pc_algorithm(data)

plt.imshow(G, cmap='binary')
plt.title("Inferred Causal Graph")
plt.xticks(range(4), ['X1', 'X2', 'X3', 'Y'])
plt.yticks(range(4), ['X1', 'X2', 'X3', 'Y'])
plt.colorbar()
plt.show()
```

Slide 11: Incorporating Prior Knowledge

In many cases, we have prior knowledge about the system we're modeling. Let's modify our causal discovery algorithm to incorporate this information.

```python
def pc_algorithm_with_prior(data, prior, alpha=0.05):
    n_vars = data.shape[1]
    G = prior.()
    
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if G[i, j] == 1:
                if abs(np.corrcoef(data[:, i], data[:, j])[0, 1]) < alpha:
                    G[i, j] = G[j, i] = 0
    
    return G

# Define prior knowledge
prior = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0]
])

X, Y = multivariate_causal_model(1000, 3)
data = np.column_stack((X, Y.reshape(-1, 1)))
G = pc_algorithm_with_prior(data, prior)

plt.imshow(G, cmap='binary')
plt.title("Inferred Causal Graph with Prior Knowledge")
plt.xticks(range(4), ['X1', 'X2', 'X3', 'Y'])
plt.yticks(range(4), ['X1', 'X2', 'X3', 'Y'])
plt.colorbar()
plt.show()
```

Slide 12: Evaluating Causal Models

To assess the performance of our causal models, we can use techniques like cross-validation and out-of-sample prediction. Let's implement a simple evaluation framework.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def evaluate_causal_model(X, Y, test_size=0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    
    # Train a simple linear model
    beta = np.linalg.lstsq(X_train, Y_train, rcond=None)[0]
    
    # Make predictions
    Y_pred = X_test @ beta
    
    # Calculate MSE
    mse = mean_squared_error(Y_test, Y_pred)
    
    return mse

X, Y = multivariate_causal_model(1000, 3)
mse = evaluate_causal_model(X, Y)

print(f"Model performance (MSE): {mse:.4f}")

# Visualize predictions vs actual values
plt.scatter(Y[-200:], X[-200:] @ np.linalg.lstsq(X, Y, rcond=None)[0])
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Model Predictions vs Actual Values")
plt.show()
```

Slide 13: Dealing with Non-Stationarity

In practice, many real-world systems exhibit non-stationary behavior. Let's explore how to detect and handle non-stationarity in our causal models.

```python
from statsmodels.tsa.stattools import adfuller

def is_stationary(x, significance=0.05):
    result = adfuller(x)
    return result[1] < significance

def differencing(x):
    return np.diff(x)

def non_stationary_causal_model(n_steps):
    X = np.cumsum(stationary_diffusion(n_steps, mu=0.01))
    Y = np.zeros(n_steps)
    
    for i in range(1, n_steps):
        Y[i] = Y[i-1] + 0.5 * X[i-1] + np.random.normal(0, 0.1)
    
    return X, Y

X, Y = non_stationary_causal_model(1000)

plt.figure(figsize=(12, 6))
plt.plot(X, label='X')
plt.plot(Y, label='Y')
plt.title("Non-Stationary Causal Relationship")
plt.legend()
plt.show()

print("X is stationary:", is_stationary(X))
print("Y is stationary:", is_stationary(Y))

# Apply differencing
X_diff = differencing(X)
Y_diff = differencing(Y)

print("Differenced X is stationary:", is_stationary(X_diff))
print("Differenced Y is stationary:", is_stationary(Y_diff))

plt.figure(figsize=(12, 6))
plt.plot(X_diff, label='Differenced X')
plt.plot(Y_diff, label='Differenced Y')
plt.title("Differenced Non-Stationary Series")
plt.legend()
plt.show()
```

Slide 14: Cointegration in Causal Modeling

Cointegration is an important concept when dealing with non-stationary time series. It occurs when two or more non-stationary series have a long-term equilibrium relationship.

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def test_cointegration(y, x):
    # Combine the series
    data = np.column_stack((y, x))
    
    # Perform Johansen cointegration test
    result = coint_johansen(data, det_order=0, k_ar_diff=1)
    
    # Extract test statistics and critical values
    trace_stat = result.lr1
    trace_crit = result.cvt[:, 1]  # 5% critical values
    
    return trace_stat, trace_crit

X, Y = non_stationary_causal_model(1000)
trace_stat, trace_crit = test_cointegration(Y, X)

print("Trace statistic:", trace_stat)
print("5% critical values:", trace_crit)
print("Cointegration exists:", any(trace_stat > trace_crit))

plt.figure(figsize=(10, 6))
plt.plot(trace_stat, label='Trace Statistic')
plt.plot(trace_crit, label='5% Critical Value')
plt.title("Johansen Cointegration Test")
plt.legend()
plt.show()
```

Slide 15: Error Correction Models (ECM)

When dealing with cointegrated series, we can use Error Correction Models to capture both short-term dynamics and long-term equilibrium relationships.

```python
from statsmodels.tsa.vector_ar.vecm import VECM

def fit_ecm(y, x):
    data = np.column_stack((y, x))
    model = VECM(data, deterministic='ci', k_ar_diff=2, coint_rank=1)
    fitted_model = model.fit()
    return fitted_model

X, Y = non_stationary_causal_model(1000)
ecm_model = fit_ecm(Y, X)

# Print summary of the ECM model
print(ecm_model.summary())

# Plot impulse response function
irf = ecm_model.irf(10)
irf.plot(orth=False)
plt.title("Impulse Response Function")
plt.show()
```

Slide 16: Causal Inference with Instrumental Variables

Instrumental variables (IV) are used in causal inference when there's potential for confounding. Let's implement a simple IV regression.

```python
from statsmodels.sandbox.regression.gmm import IV2SLS

def generate_iv_data(n_samples):
    Z = np.random.normal(0, 1, n_samples)
    X = 0.5 * Z + np.random.normal(0, 0.5, n_samples)
    Y = 2 * X + np.random.normal(0, 1, n_samples)
    return Z, X, Y

def iv_regression(y, x, z):
    model = IV2SLS(y, x, z).fit()
    return model

Z, X, Y = generate_iv_data(1000)
iv_model = iv_regression(Y, X, Z)

print(iv_model.summary())

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5)
plt.plot(X, iv_model.predict(), color='r', label='IV Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Instrumental Variable Regression')
plt.legend()
plt.show()
```

Slide 17: Additional Resources

For those interested in diving deeper into casual modeling with stationary diffusions, here are some valuable resources:

1. "Causal Inference in Statistics: A Primer" by Judea Pearl, Madelyn Glymour, and Nicholas P. Jewell ArXiv: [https://arxiv.org/abs/1305.5506](https://arxiv.org/abs/1305.5506)
2. "Causality: Models, Reasoning, and Inference" by Judea Pearl Cambridge University Press, 2009
3. "Elements of Causal Inference: Foundations and Learning Algorithms" by Jonas Peters, Dominik Janzing, and Bernhard Schölkopf ArXiv: [https://arxiv.org/abs/1703.01424](https://arxiv.org/abs/1703.01424)
4. "Causal Inference and Data Science" by Miguel A. Hernán, James M. Robins, and Babette A. Brumback ArXiv: [https://arxiv.org/abs/1603.01639](https://arxiv.org/abs/1603.01639)

These resources provide a comprehensive overview of causal inference techniques, including advanced topics not covered in this presentation.

