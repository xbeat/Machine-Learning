## Key Statistical Methods for Data Analysis and Decision-Making
Slide 1: Introduction to Statistical Methods

Statistical methods are essential tools for understanding and interpreting data. They help us make informed decisions based on empirical evidence. This presentation will explore key statistical methods and their practical applications in various fields.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=1000)

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.title('Distribution of Sample Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Descriptive Statistics

Descriptive statistics summarize and describe the main features of a dataset. They provide a concise overview of the data's central tendency, dispersion, and shape. Common measures include mean, median, mode, standard deviation, and percentiles.

```python
import numpy as np

# Generate sample data
data = np.random.normal(loc=50, scale=10, size=1000)

# Calculate descriptive statistics
mean = np.mean(data)
median = np.median(data)
std_dev = np.std(data)
percentiles = np.percentile(data, [25, 50, 75])

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"25th, 50th, 75th Percentiles: {percentiles}")
```

Slide 3: Probability Distributions

Probability distributions show how the values of a random variable are distributed. They are fundamental in statistical analysis and modeling. Common distributions include normal, binomial, and Poisson distributions.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data for normal and binomial distributions
x = np.linspace(-4, 4, 100)
y_normal = stats.norm.pdf(x, 0, 1)
y_binomial = stats.binom.pmf(np.arange(11), n=10, p=0.5)

# Plot distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(x, y_normal)
ax1.set_title('Normal Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Probability Density')

ax2.bar(np.arange(11), y_binomial)
ax2.set_title('Binomial Distribution (n=10, p=0.5)')
ax2.set_xlabel('Number of Successes')
ax2.set_ylabel('Probability')

plt.tight_layout()
plt.show()
```

Slide 4: Hypothesis Testing

Hypothesis testing is a method for making decisions about a population based on sample data. It involves formulating null and alternative hypotheses, calculating test statistics, and determining p-values to assess the evidence against the null hypothesis.

```python
import numpy as np
from scipy import stats

# Generate two samples
control = np.random.normal(loc=50, scale=10, size=100)
treatment = np.random.normal(loc=52, scale=10, size=100)

# Perform t-test
t_statistic, p_value = stats.ttest_ind(control, treatment)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

Slide 5: ANOVA (Analysis of Variance)

ANOVA tests for differences between the means of three or more groups. It is widely used in experimental design and helps determine if there are significant differences among group means.

```python
import numpy as np
from scipy import stats

# Generate sample data for three groups
group1 = np.random.normal(loc=50, scale=5, size=30)
group2 = np.random.normal(loc=52, scale=5, size=30)
group3 = np.random.normal(loc=48, scale=5, size=30)

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

Slide 6: Chi-Squared Test

The Chi-Squared test assesses the association between categorical variables. It is commonly used to determine if there is a significant relationship between two categorical variables in a population.

```python
import numpy as np
from scipy.stats import chi2_contingency

# Create a contingency table
observed = np.array([[30, 20], [15, 35]])

# Perform Chi-Squared test
chi2, p_value, dof, expected = chi2_contingency(observed)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

Slide 7: Linear Regression

Linear regression models the relationship between a dependent variable and one or more independent variables. It is widely used for prediction and understanding the strength of relationships between variables.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

print(f"Slope: {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")
```

Slide 8: Logistic Regression

Logistic regression is used for predicting binary outcomes. It models the probability of an instance belonging to a particular class and is widely used in classification problems.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Generate sample data
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()

print(f"Model Accuracy: {model.score(X_test, y_test):.4f}")
```

Slide 9: Correlation

Correlation measures the strength and direction of the relationship between two variables. It is useful for understanding how variables are related and can guide further analysis.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate correlated data
x = np.random.randn(100)
y = 0.7 * x + 0.3 * np.random.randn(100)

# Calculate Pearson correlation coefficient
correlation, p_value = stats.pearsonr(x, y)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Scatter Plot (Correlation: {correlation:.2f})')
plt.show()

print(f"Correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")
```

Slide 10: Time Series Analysis

Time series analysis involves analyzing data points collected or recorded at specific time intervals. It is crucial for understanding trends, seasonality, and making forecasts based on historical data.

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series data
np.random.seed(42)
time = np.arange(0, 100)
trend = 0.5 * time
seasonal = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 2, 100)
data = trend + seasonal + noise

# Perform time series decomposition
result = seasonal_decompose(data, model='additive', period=12)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()
```

Slide 11: Non-Parametric Tests

Non-parametric tests are used when data doesn't fit normal distribution assumptions. They are versatile and can be applied to various types of data, including ordinal and nominal data.

```python
import numpy as np
from scipy import stats

# Generate sample data
group1 = np.random.exponential(scale=2, size=50)
group2 = np.random.exponential(scale=2.5, size=50)

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(group1, group2)

print(f"Mann-Whitney U statistic: {statistic}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# Plot histograms
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(group1, bins=20, alpha=0.5, label='Group 1')
plt.hist(group2, bins=20, alpha=0.5, label='Group 2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Groups')
plt.legend()
plt.show()
```

Slide 12: Real-Life Example: A/B Testing in Web Design

A/B testing is a practical application of hypothesis testing in web design. It involves comparing two versions of a webpage to determine which one performs better in terms of user engagement or conversion rates.

```python
import numpy as np
from scipy import stats

# Simulate click-through rates for two webpage versions
np.random.seed(42)
version_a = np.random.binomial(n=1, p=0.1, size=1000)
version_b = np.random.binomial(n=1, p=0.12, size=1000)

# Perform chi-squared test
contingency_table = np.array([[sum(version_a), len(version_a) - sum(version_a)],
                              [sum(version_b), len(version_b) - sum(version_b)]])
chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

print(f"Chi-squared statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("There is a significant difference between the two versions.")
else:
    print("There is no significant difference between the two versions.")

# Calculate and print conversion rates
conversion_rate_a = sum(version_a) / len(version_a)
conversion_rate_b = sum(version_b) / len(version_b)
print(f"Conversion rate A: {conversion_rate_a:.2%}")
print(f"Conversion rate B: {conversion_rate_b:.2%}")
```

Slide 13: Real-Life Example: Quality Control in Manufacturing

Statistical process control is crucial in manufacturing for maintaining product quality. This example demonstrates how to use control charts to monitor a production process and detect any out-of-control situations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate production data
np.random.seed(42)
data = np.random.normal(loc=100, scale=2, size=100)

# Calculate control limits
mean = np.mean(data)
std_dev = np.std(data)
ucl = mean + 3 * std_dev
lcl = mean - 3 * std_dev

# Create control chart
plt.figure(figsize=(12, 6))
plt.plot(data, marker='o')
plt.axhline(y=mean, color='g', linestyle='-', label='Mean')
plt.axhline(y=ucl, color='r', linestyle='--', label='Upper Control Limit')
plt.axhline(y=lcl, color='r', linestyle='--', label='Lower Control Limit')
plt.xlabel('Sample Number')
plt.ylabel('Measurement')
plt.title('Control Chart for Manufacturing Process')
plt.legend()
plt.show()

# Check for out-of-control points
out_of_control = np.where((data > ucl) | (data < lcl))[0]
if len(out_of_control) > 0:
    print(f"Out-of-control points detected at samples: {out_of_control}")
else:
    print("Process is in control.")
```

Slide 14: Additional Resources

For those interested in delving deeper into statistical methods and their applications, here are some valuable resources:

1. ArXiv.org: A repository of scientific papers, including many on statistical methods and their applications. ([https://arxiv.org/list/stat.ME/recent](https://arxiv.org/list/stat.ME/recent))
2. "Statistical Rethinking" by Richard McElreath: A comprehensive book on Bayesian statistics and probabilistic modeling.
3. Online courses: Platforms like Coursera, edX, and DataCamp offer courses on statistics and data analysis.
4. R and Python libraries: Explore libraries like scipy, statsmodels, and scikit-learn for implementing statistical methods in your projects.
5. Open-source textbooks: OpenIntro Statistics provides free, high-quality textbooks on statistics.

Remember to verify the credibility and relevance of these resources before using them in your work or studies.

