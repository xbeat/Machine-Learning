## Exploring Bootstrap Techniques

Slide 1: Understanding Bootstrap

Bootstrap is a powerful statistical technique used for estimating the sampling distribution of a statistic by resampling with replacement from the original dataset. It's widely applied in various fields, including machine learning, to assess the variability and uncertainty of statistical estimates.

```python
import random

def bootstrap_sample(data, sample_size):
    return [random.choice(data) for _ in range(sample_size)]

# Example dataset
original_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Generate a bootstrap sample
bootstrap_sample_result = bootstrap_sample(original_data, len(original_data))
print("Original data:", original_data)
print("Bootstrap sample:", bootstrap_sample_result)
```

Slide 2: Traditional Bootstrap

Traditional bootstrap, also known as non-parametric bootstrap, involves repeatedly sampling with replacement from the original dataset. This method assumes that the observations are independent and identically distributed (i.i.d.). It's particularly useful when the underlying distribution of the data is unknown or complex.

```python
def traditional_bootstrap(data, num_iterations, statistic_func):
    results = []
    for _ in range(num_iterations):
        sample = bootstrap_sample(data, len(data))
        results.append(statistic_func(sample))
    return results

# Calculate mean as our statistic
mean = lambda x: sum(x) / len(x)

# Perform traditional bootstrap
bootstrap_means = traditional_bootstrap(original_data, 1000, mean)
print(f"Original mean: {mean(original_data)}")
print(f"Bootstrap mean estimates: {bootstrap_means[:5]}...")
```

Slide 3: Block Bootstrap

Block bootstrap is a variation of the traditional bootstrap method designed for time series or dependent data. It preserves the dependency structure within the data by sampling blocks of consecutive observations instead of individual data points. This approach is crucial when dealing with autocorrelated data.

```python
def block_bootstrap_sample(data, block_size):
    n = len(data)
    start = random.randint(0, n - block_size)
    return data[start:start + block_size]

def block_bootstrap(data, num_iterations, block_size, statistic_func):
    results = []
    for _ in range(num_iterations):
        blocks = []
        while len(blocks) * block_size < len(data):
            blocks.append(block_bootstrap_sample(data, block_size))
        sample = [item for block in blocks for item in block]
        results.append(statistic_func(sample[:len(data)]))
    return results

# Example time series data
time_series_data = [i + random.random() for i in range(100)]

# Perform block bootstrap
block_bootstrap_means = block_bootstrap(time_series_data, 1000, 10, mean)
print(f"Original mean: {mean(time_series_data)}")
print(f"Block bootstrap mean estimates: {block_bootstrap_means[:5]}...")
```

Slide 4: Differences Between Traditional and Block Bootstrap

The main difference between traditional and block bootstrap lies in how they handle data dependencies. Traditional bootstrap assumes independence between observations, while block bootstrap preserves the temporal structure of the data. This makes block bootstrap more suitable for time series analysis and other scenarios with dependent data.

```python
import matplotlib.pyplot as plt

def plot_bootstrap_distributions(traditional, block):
    plt.figure(figsize=(10, 5))
    plt.hist(traditional, bins=30, alpha=0.5, label='Traditional')
    plt.hist(block, bins=30, alpha=0.5, label='Block')
    plt.xlabel('Mean Estimate')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Bootstrap Mean Estimates')
    plt.show()

# Assuming we have already calculated traditional_means and block_means
plot_bootstrap_distributions(bootstrap_means, block_bootstrap_means)
```

Slide 5: Bootstrap in Machine Learning - Random Forest

Bootstrap is a key component in ensemble methods like Random Forest. In Random Forest, each decision tree is trained on a bootstrap sample of the original dataset. This process, known as bagging (Bootstrap Aggregating), helps to reduce overfitting and improve model robustness.

```python
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        # Simplified tree fitting logic
        pass

    def predict(self, X):
        # Simplified prediction logic
        return [random.choice([0, 1]) for _ in range(len(X))]

class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            bootstrap_indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
            X_bootstrap = [X[i] for i in bootstrap_indices]
            y_bootstrap = [y[i] for i in bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = [tree.predict(X) for tree in self.trees]
        return [round(sum(pred) / len(pred)) for pred in zip(*predictions)]

# Example usage
X = [[random.random() for _ in range(5)] for _ in range(100)]
y = [random.choice([0, 1]) for _ in range(100)]

rf = SimpleRandomForest(n_trees=10, max_depth=3)
rf.fit(X, y)
predictions = rf.predict(X[:5])
print("Random Forest predictions:", predictions)
```

Slide 6: Confidence Intervals with Bootstrap

Bootstrap can be used to estimate confidence intervals for various statistics. This is particularly useful when the sampling distribution of the statistic is unknown or difficult to derive analytically. We'll demonstrate how to calculate a 95% confidence interval for the mean using the percentile method.

```python
def bootstrap_confidence_interval(data, num_iterations, statistic_func, alpha=0.05):
    bootstrap_estimates = traditional_bootstrap(data, num_iterations, statistic_func)
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    return (
        percentile(bootstrap_estimates, lower_percentile),
        percentile(bootstrap_estimates, upper_percentile)
    )

def percentile(data, p):
    k = (len(data) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted(data)[int(k)]
    d0 = sorted(data)[int(f)] * (c - k)
    d1 = sorted(data)[int(c)] * (k - f)
    return d0 + d1

# Calculate 95% confidence interval for the mean
data = [random.gauss(0, 1) for _ in range(1000)]
ci_lower, ci_upper = bootstrap_confidence_interval(data, 10000, mean)
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
```

Slide 7: Bootstrap for Hypothesis Testing

Bootstrap can be used for hypothesis testing when traditional parametric tests are not applicable. We'll demonstrate how to perform a two-sample bootstrap hypothesis test to compare the means of two groups.

```python
def two_sample_bootstrap_test(group1, group2, num_iterations=10000):
    observed_diff = mean(group1) - mean(group2)
    combined = group1 + group2
    n1, n2 = len(group1), len(group2)
    
    diffs = []
    for _ in range(num_iterations):
        sample = random.sample(combined, n1 + n2)
        bootstrap_group1 = sample[:n1]
        bootstrap_group2 = sample[n1:]
        diffs.append(mean(bootstrap_group1) - mean(bootstrap_group2))
    
    p_value = sum(1 for diff in diffs if abs(diff) >= abs(observed_diff)) / num_iterations
    return p_value

# Example: Test if two groups have different means
group1 = [random.gauss(0, 1) for _ in range(100)]
group2 = [random.gauss(0.5, 1) for _ in range(100)]

p_value = two_sample_bootstrap_test(group1, group2)
print(f"Bootstrap hypothesis test p-value: {p_value:.4f}")
```

Slide 8: Bootstrap for Time Series Forecasting

Bootstrap can be applied to time series forecasting to estimate prediction intervals. We'll demonstrate a simple implementation of bootstrap prediction intervals for an autoregressive model.

```python
def ar_model(data, lag=1):
    X = [data[i:i+lag] for i in range(len(data)-lag)]
    y = data[lag:]
    
    # Simple linear regression
    X_mean = mean([sum(x)/len(x) for x in X])
    y_mean = mean(y)
    numerator = sum((X[i][0] - X_mean) * (y[i] - y_mean) for i in range(len(X)))
    denominator = sum((x[0] - X_mean)**2 for x in X)
    
    slope = numerator / denominator
    intercept = y_mean - slope * X_mean
    
    return lambda x: intercept + slope * x

def bootstrap_forecast(data, horizon, num_iterations=1000):
    model = ar_model(data)
    forecasts = []
    
    for _ in range(num_iterations):
        bootstrap_data = bootstrap_sample(data, len(data))
        bootstrap_model = ar_model(bootstrap_data)
        
        forecast = []
        last_value = bootstrap_data[-1]
        for _ in range(horizon):
            next_value = bootstrap_model(last_value)
            forecast.append(next_value)
            last_value = next_value
        
        forecasts.append(forecast)
    
    return list(zip(*forecasts))

# Example usage
time_series = [random.gauss(i*0.1, 1) for i in range(100)]
forecast_distribution = bootstrap_forecast(time_series, horizon=5)

for i, dist in enumerate(forecast_distribution):
    lower, upper = percentile(dist, 2.5), percentile(dist, 97.5)
    print(f"Step {i+1} 95% Prediction Interval: ({lower:.2f}, {upper:.2f})")
```

Slide 9: Bootstrap for Model Validation

Bootstrap can be used for model validation, particularly in assessing the stability of model performance metrics. We'll demonstrate how to use bootstrap to estimate the confidence interval of a model's accuracy.

```python
def simple_model(X, y):
    # Dummy model that predicts the majority class
    return lambda x: max(set(y), key=y.count)

def accuracy(y_true, y_pred):
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)

def bootstrap_model_validation(X, y, num_iterations=1000):
    accuracies = []
    for _ in range(num_iterations):
        indices = bootstrap_sample(range(len(X)), len(X))
        X_boot, y_boot = [X[i] for i in indices], [y[i] for i in indices]
        
        model = simple_model(X_boot, y_boot)
        y_pred = [model(x) for x in X]
        accuracies.append(accuracy(y, y_pred))
    
    return accuracies

# Example usage
X = [[random.random() for _ in range(5)] for _ in range(1000)]
y = [random.choice([0, 1]) for _ in range(1000)]

accuracy_distribution = bootstrap_model_validation(X, y)
ci_lower, ci_upper = percentile(accuracy_distribution, 2.5), percentile(accuracy_distribution, 97.5)
print(f"Model accuracy 95% CI: ({ci_lower:.4f}, {ci_upper:.4f})")
```

Slide 10: Jackknife Resampling

Jackknife is another resampling technique related to bootstrap. It involves systematically leaving out one observation at a time from the dataset. Jackknife can be used to estimate the bias and standard error of a statistic.

```python
def jackknife_resampling(data, statistic_func):
    n = len(data)
    jackknife_estimates = []
    
    for i in range(n):
        sample = data[:i] + data[i+1:]
        jackknife_estimates.append(statistic_func(sample))
    
    jackknife_estimate = sum(jackknife_estimates) / n
    bias = (n - 1) * (jackknife_estimate - statistic_func(data))
    variance = sum((est - jackknife_estimate)**2 for est in jackknife_estimates) * (n - 1) / n
    
    return jackknife_estimate, bias, math.sqrt(variance)

# Example usage
data = [random.gauss(0, 1) for _ in range(100)]
jack_est, jack_bias, jack_se = jackknife_resampling(data, mean)

print(f"Jackknife estimate: {jack_est:.4f}")
print(f"Estimated bias: {jack_bias:.4f}")
print(f"Estimated standard error: {jack_se:.4f}")
```

Slide 11: Cross-Validation vs. Bootstrap

Cross-validation and bootstrap are both resampling methods used for model evaluation, but they have different approaches and use cases. We'll compare these methods using a simple example.

```python
def cross_validation(X, y, k_folds=5):
    fold_size = len(X) // k_folds
    accuracies = []
    
    for i in range(k_folds):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test, y_test = X[start:end], y[start:end]
        X_train = X[:start] + X[end:]
        y_train = y[:start] + y[end:]
        
        model = simple_model(X_train, y_train)
        y_pred = [model(x) for x in X_test]
        accuracies.append(accuracy(y_test, y_pred))
    
    return sum(accuracies) / len(accuracies)

def bootstrap_validation(X, y, num_iterations=1000):
    accuracies = []
    for _ in range(num_iterations):
        indices = bootstrap_sample(range(len(X)), len(X))
        X_train, y_train = [X[i] for i in indices], [y[i] for i in indices]
        
        model = simple_model(X_train, y_train)
        y_pred = [model(x) for x in X]
        accuracies.append(accuracy(y, y_pred))
    
    return sum(accuracies) / len(accuracies)

# Example usage
X = [[random.random() for _ in range(5)] for _ in range(1000)]
y = [random.choice([0, 1]) for _ in range(1000)]

cv_accuracy = cross_validation(X, y)
bootstrap_accuracy = bootstrap_validation(X, y)

print(f"Cross-validation accuracy: {cv_accuracy:.4f}")
print(f"Bootstrap validation accuracy: {bootstrap_accuracy:.4f}")
```

Slide 12: Bootstrap in Neural Network Ensembles

Bootstrap can be applied to create ensembles of neural networks, improving model robustness and performance. This technique involves training multiple neural networks on different bootstrap samples of the training data.

```python
import random
import math

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = [random.uniform(-1, 1) for _ in range(hidden_size * input_size)]
        self.output_layer = [random.uniform(-1, 1) for _ in range(output_size * hidden_size)]

    def predict(self, input_data):
        # Simplified forward pass
        hidden = [math.tanh(sum(x * w for x, w in zip(input_data, self.hidden_layer[i:i+len(input_data)])))
                  for i in range(0, len(self.hidden_layer), len(input_data))]
        output = sum(h * w for h, w in zip(hidden, self.output_layer))
        return 1 / (1 + math.exp(-output))

def create_bootstrap_ensemble(X, y, n_models=5):
    ensemble = []
    for _ in range(n_models):
        bootstrap_indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_bootstrap = [X[i] for i in bootstrap_indices]
        y_bootstrap = [y[i] for i in bootstrap_indices]
        model = SimpleNeuralNetwork(len(X[0]), 5, 1)
        # Training would occur here (omitted for simplicity)
        ensemble.append(model)
    return ensemble

# Example usage
X = [[random.random() for _ in range(3)] for _ in range(100)]
y = [random.choice([0, 1]) for _ in range(100)]

ensemble = create_bootstrap_ensemble(X, y)
ensemble_predictions = [sum(model.predict(x) for model in ensemble) / len(ensemble) for x in X[:5]]
print("Ensemble predictions for first 5 samples:", ensemble_predictions)
```

Slide 13: Parametric Bootstrap

Parametric bootstrap is a variation of the bootstrap method that assumes a specific probability distribution for the data. It generates bootstrap samples by drawing from the assumed distribution with parameters estimated from the original data.

```python
import random
import statistics

def parametric_bootstrap(data, num_iterations, distribution='normal'):
    if distribution == 'normal':
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        return [
            [random.gauss(mean, std_dev) for _ in range(len(data))]
            for _ in range(num_iterations)
        ]
    else:
        raise ValueError("Unsupported distribution")

def estimate_confidence_interval(bootstrap_samples, alpha=0.05):
    means = [statistics.mean(sample) for sample in bootstrap_samples]
    means.sort()
    lower_index = int(alpha / 2 * len(means))
    upper_index = int((1 - alpha / 2) * len(means))
    return means[lower_index], means[upper_index]

# Example usage
original_data = [random.gauss(0, 1) for _ in range(100)]
bootstrap_samples = parametric_bootstrap(original_data, 1000)
ci_lower, ci_upper = estimate_confidence_interval(bootstrap_samples)

print(f"Original data mean: {statistics.mean(original_data):.4f}")
print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")
```

Slide 14: Bootstrap in Regression Analysis

Bootstrap can be applied to regression analysis to estimate the uncertainty of regression coefficients. This is particularly useful when the assumptions of traditional regression methods are violated.

```python
import random

def simple_linear_regression(x, y):
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept

def bootstrap_regression(x, y, num_iterations=1000):
    n = len(x)
    slope_estimates, intercept_estimates = [], []
    
    for _ in range(num_iterations):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        x_sample = [x[i] for i in indices]
        y_sample = [y[i] for i in indices]
        slope, intercept = simple_linear_regression(x_sample, y_sample)
        slope_estimates.append(slope)
        intercept_estimates.append(intercept)
    
    return slope_estimates, intercept_estimates

# Example usage
x = [i + random.gauss(0, 0.5) for i in range(100)]
y = [2 * xi + 1 + random.gauss(0, 1) for xi in x]

slope_estimates, intercept_estimates = bootstrap_regression(x, y)
print(f"Slope 95% CI: ({percentile(slope_estimates, 2.5):.4f}, {percentile(slope_estimates, 97.5):.4f})")
print(f"Intercept 95% CI: ({percentile(intercept_estimates, 2.5):.4f}, {percentile(intercept_estimates, 97.5):.4f})")
```

Slide 15: Additional Resources

For those interested in diving deeper into bootstrap methods and their applications, here are some valuable resources:

1.  Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. Chapman and Hall/CRC. ArXiv URL: [https://arxiv.org/abs/1708.00273](https://arxiv.org/abs/1708.00273)
2.  Davison, A. C., & Hinkley, D. V. (1997). Bootstrap Methods and their Application. Cambridge University Press.
3.  Hall, P. (1992). The Bootstrap and Edgeworth Expansion. Springer Series in Statistics.
4.  Chernick, M. R. (2011). Bootstrap Methods: A Guide for Practitioners and Researchers. Wiley Series in Probability and Statistics.

These resources provide comprehensive coverage of bootstrap techniques, from foundational concepts to advanced applications in various fields of statistics and machine learning.

