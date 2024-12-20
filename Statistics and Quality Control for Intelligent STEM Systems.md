## Statistics and Quality Control for Intelligent STEM Systems
Slide 1: Introduction to Statistics and Quality Control in STEM Projects

Statistics and quality control play crucial roles in STEM projects, particularly when developing intelligent systems. These methodologies help ensure reliability, efficiency, and accuracy in data-driven decision-making processes. This presentation will explore various analytical approaches using Python to implement statistical methods and quality control techniques in intelligent systems.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
data = np.random.normal(100, 15, 1000)

# Create histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Distribution of Sample Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Descriptive Statistics: Measures of Central Tendency

Descriptive statistics provide a summary of the main characteristics of a dataset. Measures of central tendency, such as mean, median, and mode, are fundamental in understanding the typical values in a distribution.

```python
import numpy as np

# Generate sample data
data = np.random.normal(100, 15, 1000)

# Calculate measures of central tendency
mean = np.mean(data)
median = np.median(data)
mode = float(max(set(data), key=list(data).count))

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {mode:.2f}")
```

Slide 3: Measures of Variability: Standard Deviation and Variance

Measures of variability help quantify the spread of data points in a dataset. Standard deviation and variance are commonly used metrics for this purpose, providing insights into data dispersion.

```python
import numpy as np

# Generate sample data
data = np.random.normal(100, 15, 1000)

# Calculate measures of variability
std_dev = np.std(data)
variance = np.var(data)

print(f"Standard Deviation: {std_dev:.2f}")
print(f"Variance: {variance:.2f}")

# Visualize data dispersion
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, alpha=0.5)
plt.axhline(y=np.mean(data), color='r', linestyle='--', label='Mean')
plt.axhline(y=np.mean(data) + std_dev, color='g', linestyle='--', label='Mean + 1 Std Dev')
plt.axhline(y=np.mean(data) - std_dev, color='g', linestyle='--', label='Mean - 1 Std Dev')
plt.legend()
plt.title('Data Dispersion Visualization')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.show()
```

Slide 4: Probability Distributions: Normal Distribution

The normal distribution is a fundamental concept in statistics, often used to model real-world phenomena. Understanding and working with normal distributions is crucial for many statistical analyses and quality control processes.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from a normal distribution
mu, sigma = 0, 1
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = stats.norm.pdf(x, mu, sigma)

# Plot the normal distribution
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Standard Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.fill_between(x, y, where=(x >= -sigma) & (x <= sigma), alpha=0.3, label='68% of data')
plt.legend()
plt.show()

# Calculate probability within 1 standard deviation
prob_within_1_sigma = stats.norm.cdf(1) - stats.norm.cdf(-1)
print(f"Probability within 1 standard deviation: {prob_within_1_sigma:.4f}")
```

Slide 5: Hypothesis Testing: t-test

Hypothesis testing is a critical component of statistical analysis, allowing us to make inferences about population parameters based on sample data. The t-test is commonly used to compare means between two groups.

```python
import numpy as np
from scipy import stats

# Generate two sample datasets
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a significant difference between the groups.")
else:
    print("Fail to reject null hypothesis: There is no significant difference between the groups.")
```

Slide 6: Correlation and Regression Analysis

Correlation and regression analyses are essential tools for understanding relationships between variables. These techniques help identify patterns and make predictions based on observed data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate correlated data
x = np.random.normal(0, 1, 100)
y = 2 * x + np.random.normal(0, 0.5, 100)

# Calculate correlation coefficient
correlation_coeff, _ = stats.pearsonr(x, y)

# Perform linear regression
slope, intercept, r_value, _, _ = stats.linregress(x, y)

# Plot data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)
plt.plot(x, slope * x + intercept, color='r', label='Regression Line')
plt.title(f'Correlation and Regression Analysis (r = {correlation_coeff:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print(f"Correlation coefficient: {correlation_coeff:.4f}")
print(f"Regression equation: Y = {slope:.2f}X + {intercept:.2f}")
```

Slide 7: Quality Control: Control Charts

Control charts are powerful tools for monitoring process stability and detecting unusual variations. They help identify when a process is out of control and requires intervention.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 5, 50)

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
plt.title('Control Chart')
plt.xlabel('Sample Number')
plt.ylabel('Measurement')
plt.legend()
plt.show()

# Check for out-of-control points
out_of_control = np.where((data > ucl) | (data < lcl))[0]
print(f"Out-of-control points: {out_of_control}")
```

Slide 8: Sampling Techniques: Simple Random Sampling

Sampling is a crucial aspect of statistical analysis and quality control. Simple random sampling ensures that each item in a population has an equal chance of being selected, reducing bias in the sample.

```python
import numpy as np

# Create a population
population = np.arange(1, 1001)

# Perform simple random sampling
sample_size = 50
sample = np.random.choice(population, size=sample_size, replace=False)

print(f"Population size: {len(population)}")
print(f"Sample size: {len(sample)}")
print(f"Sample: {sample[:10]}...")  # Display first 10 elements

# Calculate sample statistics
sample_mean = np.mean(sample)
sample_std = np.std(sample)

print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample standard deviation: {sample_std:.2f}")
```

Slide 9: Data Visualization: Box Plots

Data visualization is essential for understanding and communicating statistical information. Box plots provide a concise summary of the distribution of a dataset, showing key statistics such as median, quartiles, and potential outliers.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate multiple datasets
data1 = np.random.normal(100, 10, 100)
data2 = np.random.normal(90, 20, 100)
data3 = np.random.normal(110, 5, 100)

# Create box plot
plt.figure(figsize=(10, 6))
plt.boxplot([data1, data2, data3], labels=['Dataset 1', 'Dataset 2', 'Dataset 3'])
plt.title('Box Plot Comparison of Multiple Datasets')
plt.ylabel('Value')
plt.show()

# Calculate and print summary statistics
for i, data in enumerate([data1, data2, data3], 1):
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    print(f"Dataset {i}:")
    print(f"  Median: {median:.2f}")
    print(f"  Q1: {q1:.2f}")
    print(f"  Q3: {q3:.2f}")
    print(f"  IQR: {iqr:.2f}")
```

Slide 10: Confidence Intervals

Confidence intervals provide a range of plausible values for a population parameter, taking into account the uncertainty in our estimates. They are crucial for making inferences about populations based on sample data.

```python
import numpy as np
from scipy import stats

# Generate sample data
sample_size = 100
sample = np.random.normal(100, 15, sample_size)

# Calculate sample statistics
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)  # ddof=1 for sample standard deviation

# Calculate confidence interval (95% confidence level)
confidence_level = 0.95
degrees_of_freedom = sample_size - 1
t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
margin_of_error = t_value * (sample_std / np.sqrt(sample_size))

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"Sample Mean: {sample_mean:.2f}")
print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")

# Visualize confidence interval
plt.figure(figsize=(10, 6))
plt.hist(sample, bins=20, density=True, alpha=0.7, color='skyblue')
plt.axvline(sample_mean, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.axvline(ci_lower, color='green', linestyle='dotted', linewidth=2, label='95% CI')
plt.axvline(ci_upper, color='green', linestyle='dotted', linewidth=2)
plt.title('Sample Distribution with Confidence Interval')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

Slide 11: Real-Life Example: Quality Control in Manufacturing

In a manufacturing setting, statistical quality control is crucial for maintaining product consistency and identifying potential issues in the production process. Let's consider a hypothetical example of a factory producing electronic components.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simulate resistor measurements (target: 100 ohms, tolerance: ±5%)
np.random.seed(42)
measurements = np.random.normal(100, 2.5, 1000)

# Define specification limits
lower_spec = 95
upper_spec = 105

# Calculate process capability indices
mean = np.mean(measurements)
std_dev = np.std(measurements)
cp = (upper_spec - lower_spec) / (6 * std_dev)
cpk = min((upper_spec - mean) / (3 * std_dev), (mean - lower_spec) / (3 * std_dev))

print(f"Process Capability (Cp): {cp:.2f}")
print(f"Process Capability Index (Cpk): {cpk:.2f}")

# Visualize data and specification limits
plt.figure(figsize=(12, 6))
plt.hist(measurements, bins=30, density=True, alpha=0.7, color='skyblue')
plt.axvline(lower_spec, color='red', linestyle='dashed', label='Spec Limits')
plt.axvline(upper_spec, color='red', linestyle='dashed')
plt.axvline(mean, color='green', linestyle='dotted', label='Mean')
plt.title('Resistor Measurements Distribution')
plt.xlabel('Resistance (ohms)')
plt.ylabel('Density')
plt.legend()
plt.show()

# Calculate percentage of out-of-spec products
out_of_spec = np.sum((measurements < lower_spec) | (measurements > upper_spec))
percentage_out_of_spec = (out_of_spec / len(measurements)) * 100
print(f"Percentage of out-of-spec products: {percentage_out_of_spec:.2f}%")
```

Slide 12: Real-Life Example: A/B Testing in User Interface Design

A/B testing is a common application of statistics in user interface design and user experience optimization. Let's consider an example where we're testing two different button designs on a website to see which one leads to higher click-through rates.

```python
import numpy as np
from scipy import stats

# Simulate click-through data for two button designs
np.random.seed(42)
design_a_clicks = np.random.binomial(1, 0.12, 1000)  # 12% click-through rate
design_b_clicks = np.random.binomial(1, 0.15, 1000)  # 15% click-through rate

# Calculate click-through rates
ctr_a = np.mean(design_a_clicks)
ctr_b = np.mean(design_b_clicks)

print(f"Design A Click-through Rate: {ctr_a:.4f}")
print(f"Design B Click-through Rate: {ctr_b:.4f}")

# Perform statistical test (chi-square test of independence)
contingency_table = np.array([[np.sum(design_a_clicks), len(design_a_clicks) - np.sum(design_a_clicks)],
                              [np.sum(design_b_clicks), len(design_b_clicks) - np.sum(design_b_clicks)]])

chi2, p_value = stats.chi2_contingency(contingency_table)[:2]

print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("There is a statistically significant difference between the two designs.")
else:
    print("There is no statistically significant difference between the two designs.")

# Calculate relative improvement
relative_improvement = (ctr_b - ctr_a) / ctr_a * 100
print(f"Relative improvement of Design B over Design A: {relative_improvement:.2f}%")
```

Slide 13: Machine Learning Integration: Feature Selection

In intelligent systems, feature selection is a crucial step in preparing data for machine learning models. Statistical techniques can help identify the most relevant features, improving model performance and reducing computational complexity.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train and evaluate model with selected features
model = LogisticRegression()
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy with selected features: {accuracy:.4f}")

# Display selected feature indices
selected_features = selector.get_support(indices=True)
print(f"Selected feature indices: {selected_features}")
```

Slide 14: Time Series Analysis: Moving Averages

Time series analysis is essential for understanding and forecasting trends in data that change over time. Moving averages are a simple yet powerful technique for smoothing out short-term fluctuations and highlighting longer-term trends.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
time = np.arange(100)
trend = 0.5 * time
seasonality = 10 * np.sin(2 * np.pi * time / 12)
noise = np.random.normal(0, 5, 100)
data = trend + seasonality + noise

# Calculate simple moving average
window_size = 12
moving_average = np.convolve(data, np.ones(window_size), 'valid') / window_size

# Plot original data and moving average
plt.figure(figsize=(12, 6))
plt.plot(time, data, label='Original Data')
plt.plot(time[window_size-1:], moving_average, label=f'{window_size}-point Moving Average', color='red')
plt.title('Time Series Data with Moving Average')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and print statistics
original_std = np.std(data)
smoothed_std = np.std(moving_average)
print(f"Standard deviation of original data: {original_std:.2f}")
print(f"Standard deviation of smoothed data: {smoothed_std:.2f}")
print(f"Reduction in variability: {(1 - smoothed_std/original_std)*100:.2f}%")
```

Slide 15: Additional Resources

For those interested in further exploring statistics and quality control in STEM projects, here are some valuable resources:

1. ArXiv.org: A repository of scholarly articles in various fields, including statistics and machine learning. Example: "Statistical Methods for Machine Learning" ([https://arxiv.org/abs/2012.05951](https://arxiv.org/abs/2012.05951))
2. Python libraries documentation:
   * NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)
   * SciPy: [https://docs.scipy.org/doc/scipy/](https://docs.scipy.org/doc/scipy/)
   * Scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
3. Online courses and tutorials on platforms like Coursera, edX, and DataCamp.
4. Textbooks on statistical quality control and data analysis in Python.

Remember to always verify the credibility and relevance of additional resources before incorporating them into your projects.

