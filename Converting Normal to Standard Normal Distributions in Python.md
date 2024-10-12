## Converting Normal to Standard Normal Distributions in Python
Slide 1: Converting a Normal Distribution to a Standard Normal Distribution

The process of transforming a normal distribution to a standard normal distribution is a fundamental technique in statistics and data analysis. This conversion allows for easier comparison and interpretation of data from different normal distributions. In this presentation, we'll explore the mathematical concept behind this transformation and implement it using Python.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a normal distribution
mean = 5
std_dev = 2
data = np.random.normal(mean, std_dev, 1000)

# Plot the original distribution
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
plt.title("Original Normal Distribution")
plt.show()
```

Slide 2: Understanding the Normal Distribution

A normal distribution, also known as a Gaussian distribution, is characterized by its mean (μ) and standard deviation (σ). The mean determines the center of the distribution, while the standard deviation influences its spread. Normal distributions are symmetric and bell-shaped, with data clustered around the mean.

```python
# Visualize the normal distribution
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev)**2)

plt.plot(x, y)
plt.title(f"Normal Distribution (μ={mean}, σ={std_dev})")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.show()
```

Slide 3: The Standard Normal Distribution

The standard normal distribution is a special case of the normal distribution with a mean of 0 and a standard deviation of 1. It's often denoted as N(0, 1) and is widely used as a reference distribution in statistics. Converting to a standard normal distribution simplifies probability calculations and allows for easier comparison between different datasets.

```python
# Generate and plot a standard normal distribution
standard_normal = np.random.standard_normal(1000)
plt.hist(standard_normal, bins=30, density=True, alpha=0.7, color='lightgreen')
plt.title("Standard Normal Distribution")
plt.show()
```

Slide 4: The Z-Score Transformation

To convert a normal distribution to a standard normal distribution, we use the z-score transformation. The z-score represents the number of standard deviations a data point is from the mean. The formula for calculating the z-score is:

z = (x - μ) / σ

Where: x is the original value μ is the mean of the original distribution σ is the standard deviation of the original distribution

```python
# Calculate z-scores
z_scores = (data - mean) / std_dev

# Plot the transformed distribution
plt.hist(z_scores, bins=30, density=True, alpha=0.7, color='salmon')
plt.title("Transformed Standard Normal Distribution")
plt.show()
```

Slide 5: Implementing the Transformation in Python

Let's implement the z-score transformation using Python's NumPy library. We'll create a function that takes the original data, mean, and standard deviation as inputs and returns the standardized data.

```python
def standardize(data, mean, std_dev):
    return (data - mean) / std_dev

# Apply the transformation
standardized_data = standardize(data, mean, std_dev)

# Verify the mean and standard deviation of the transformed data
print(f"Mean of standardized data: {np.mean(standardized_data):.4f}")
print(f"Standard deviation of standardized data: {np.std(standardized_data):.4f}")
```

Slide 6: Visualizing the Transformation

To better understand the transformation, let's create a side-by-side comparison of the original and standardized distributions.

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
ax1.set_title(f"Original Distribution\n(μ={mean}, σ={std_dev})")

ax2.hist(standardized_data, bins=30, density=True, alpha=0.7, color='salmon')
ax2.set_title("Standardized Distribution\n(μ=0, σ=1)")

plt.tight_layout()
plt.show()
```

Slide 7: Probability Calculations

One of the main advantages of using the standard normal distribution is the simplicity of probability calculations. The area under the curve between two z-scores represents the probability of a value falling within that range.

```python
from scipy import stats

# Calculate probability of a value falling between -1 and 1 standard deviations
prob = stats.norm.cdf(1) - stats.norm.cdf(-1)

print(f"Probability of a value falling within 1 standard deviation: {prob:.4f}")

# Visualize the probability
x = np.linspace(-3, 3, 100)
y = stats.norm.pdf(x)
plt.plot(x, y)
plt.fill_between(x, y, where=(x >= -1) & (x <= 1), alpha=0.3)
plt.title("Standard Normal Distribution\nProbability within 1 Standard Deviation")
plt.show()
```

Slide 8: Real-Life Example: Height Distribution

Consider a dataset of adult human heights. The mean height is 170 cm with a standard deviation of 10 cm. We can use the z-score transformation to standardize this distribution and calculate probabilities.

```python
# Generate sample height data
heights = np.random.normal(170, 10, 1000)

# Standardize the heights
standardized_heights = standardize(heights, 170, 10)

# Calculate the probability of a person being taller than 190 cm
z_score_190 = (190 - 170) / 10
prob_taller_190 = 1 - stats.norm.cdf(z_score_190)

print(f"Probability of a person being taller than 190 cm: {prob_taller_190:.4f}")

# Visualize the height distribution and the 190 cm threshold
plt.hist(heights, bins=30, density=True, alpha=0.7)
plt.axvline(190, color='red', linestyle='--', label='190 cm threshold')
plt.title("Height Distribution")
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
```

Slide 9: Real-Life Example: Test Scores

Imagine a standardized test where scores follow a normal distribution with a mean of 500 and a standard deviation of 100. We can use the z-score transformation to compare individual scores and calculate percentiles.

```python
# Generate sample test scores
scores = np.random.normal(500, 100, 1000)

# Function to calculate percentile
def calculate_percentile(score, mean, std_dev):
    z_score = (score - mean) / std_dev
    percentile = stats.norm.cdf(z_score) * 100
    return percentile

# Calculate percentile for a score of 650
percentile_650 = calculate_percentile(650, 500, 100)

print(f"A score of 650 is in the {percentile_650:.2f}th percentile")

# Visualize the score distribution and the 650 score threshold
plt.hist(scores, bins=30, density=True, alpha=0.7)
plt.axvline(650, color='red', linestyle='--', label='650 score')
plt.title("Test Score Distribution")
plt.xlabel("Score")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
```

Slide 10: Inverse Transformation

Sometimes, we need to convert z-scores back to the original scale. This process is called inverse transformation and can be useful when working with confidence intervals or generating random samples from a specific normal distribution.

```python
def inverse_standardize(z_scores, mean, std_dev):
    return z_scores * std_dev + mean

# Generate standard normal data
standard_data = np.random.standard_normal(1000)

# Transform to a normal distribution with mean=75 and std_dev=5
transformed_data = inverse_standardize(standard_data, 75, 5)

# Visualize the transformed data
plt.hist(transformed_data, bins=30, density=True, alpha=0.7)
plt.title("Transformed Normal Distribution\n(μ=75, σ=5)")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.show()

print(f"Mean of transformed data: {np.mean(transformed_data):.2f}")
print(f"Standard deviation of transformed data: {np.std(transformed_data):.2f}")
```

Slide 11: Handling Multivariate Normal Distributions

In some cases, we may need to work with multivariate normal distributions, where we have multiple correlated variables. The standardization process for multivariate normal distributions involves matrix operations.

```python
import numpy as np
from scipy import stats

# Generate a 2D normal distribution
mean = [1, 2]
cov = [[1, 0.5], [0.5, 2]]
data_2d = np.random.multivariate_normal(mean, cov, 1000)

# Standardize the 2D data
standardized_2d = stats.zscore(data_2d)

# Visualize the original and standardized data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5)
ax1.set_title("Original 2D Normal Distribution")

ax2.scatter(standardized_2d[:, 0], standardized_2d[:, 1], alpha=0.5)
ax2.set_title("Standardized 2D Normal Distribution")

plt.tight_layout()
plt.show()
```

Slide 12: Applications in Machine Learning

Standardization is crucial in many machine learning algorithms, especially those sensitive to the scale of input features. Let's demonstrate how standardization can improve the performance of a simple linear regression model.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Generate sample data
X = np.random.normal(0, 100, (1000, 1))
y = 2 * X + np.random.normal(0, 10, (1000, 1))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate without standardization
model_raw = LinearRegression()
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
mse_raw = mean_squared_error(y_test, y_pred_raw)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate with standardization
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = model_scaled.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)

print(f"MSE without standardization: {mse_raw:.4f}")
print(f"MSE with standardization: {mse_scaled:.4f}")
```

Slide 13: Best Practices and Considerations

When working with normal distributions and standardization:

1. Always check for normality assumptions before applying z-score transformation.
2. Be cautious with outliers, as they can significantly affect the mean and standard deviation.
3. Consider using robust standardization methods for datasets with extreme values.
4. Remember that standardization changes the scale of your data, which may affect interpretability.
5. When working with time series data, be mindful of temporal dependencies and consider alternatives like differencing or rolling standardization.

```python
import scipy.stats as stats

# Generate sample data with outliers
data_with_outliers = np.concatenate([np.random.normal(0, 1, 990), np.random.normal(0, 10, 10)])

# Perform normality test
_, p_value = stats.normaltest(data_with_outliers)

print(f"p-value for normality test: {p_value:.4f}")

# Visualize the data
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(data_with_outliers, bins=30, density=True, alpha=0.7)
plt.title("Data with Outliers")

plt.subplot(122)
stats.probplot(data_with_outliers, dist="norm", plot=plt)
plt.title("Q-Q Plot")

plt.tight_layout()
plt.show()
```

Slide 14: Additional Resources

For those interested in delving deeper into the topic of normal distributions and standardization, here are some recommended resources:

1. "The Art of Statistics: Learning from Data" by David Spiegelhalter
2. "Statistics" by David Freedman, Robert Pisani, and Roger Purves
3. "Introduction to Probability and Statistics for Engineers and Scientists" by Sheldon M. Ross
4. ArXiv paper: "A Tutorial on the Cross-Entropy Method" by Pieter-Tjerk de Boer et al. ([https://arxiv.org/abs/cs/0312015](https://arxiv.org/abs/cs/0312015))
5. ArXiv paper: "Probability Distributions in the Statistical Sciences" by Narayanaswamy Balakrishnan ([https://arxiv.org/abs/1808.05870](https://arxiv.org/abs/1808.05870))

These resources provide in-depth explanations and advanced applications of normal distributions and standardization techniques in various fields of study.

