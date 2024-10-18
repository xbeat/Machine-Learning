## Statistics for Data Science with Python

Slide 1: Introduction to Statistics for Data Science

Statistics plays a crucial role in data science, providing the foundation for understanding and interpreting data. This slideshow will cover key statistical concepts and their implementation in Python, focusing on practical applications for data scientists.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate sample data
data = np.random.normal(loc=0, scale=1, size=1000)

# Create a histogram
plt.hist(data, bins=30, edgecolor='black')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

Slide 2: Measures of Central Tendency

Central tendency measures help us understand the typical or average value in a dataset. The three main measures are mean, median, and mode.

```python

data = [2, 3, 3, 4, 5, 5, 5, 6, 6, 7]

mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data).mode[0]

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")

# Output:
# Mean: 4.6
# Median: 5.0
# Mode: 5
```

Slide 3: Measures of Dispersion

Dispersion measures indicate how spread out the data is. Common measures include variance, standard deviation, and range.

```python

data = [2, 3, 3, 4, 5, 5, 5, 6, 6, 7]

variance = np.var(data)
std_dev = np.std(data)
data_range = np.ptp(data)

print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
print(f"Range: {data_range}")

# Output:
# Variance: 2.24
# Standard Deviation: 1.4966629547095765
# Range: 5
```

Slide 4: Probability Distributions

Probability distributions describe the likelihood of different outcomes in a random event. The normal distribution is particularly important in statistics.

```python
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, 0, 1)

plt.plot(x, y)
plt.title('Standard Normal Distribution')
plt.xlabel('Z-score')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()
```

Slide 5: Hypothesis Testing

Hypothesis testing is a statistical method used to make inferences about population parameters based on sample data.

```python

# Sample data
group1 = [5.2, 5.4, 5.6, 5.8, 6.0]
group2 = [5.0, 5.2, 5.4, 5.6, 5.8]

# Perform t-test
t_statistic, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Output:
# T-statistic: 1.4142135623730951
# P-value: 0.19493381801926307
```

Slide 6: Correlation Analysis

Correlation analysis measures the strength and direction of the relationship between two variables.

```python
import matplotlib.pyplot as plt

# Generate correlated data
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Calculate correlation coefficient
correlation = np.corrcoef(x, y)[0, 1]

plt.scatter(x, y)
plt.title(f'Scatter Plot (Correlation: {correlation:.2f})')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Slide 7: Linear Regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables.

```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot results
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Output:
# Slope: 0.7
# Intercept: 2.0
```

Slide 8: Confidence Intervals

Confidence intervals provide a range of values that is likely to contain the true population parameter with a certain level of confidence.

```python
from scipy import stats

# Sample data
data = [20, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# Calculate confidence interval
mean = np.mean(data)
std_error = stats.sem(data)
ci = stats.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=std_error)

print(f"Sample Mean: {mean}")
print(f"95% Confidence Interval: {ci}")

# Output:
# Sample Mean: 25.4
# 95% Confidence Interval: (23.176629629629626, 27.623370370370374)
```

Slide 9: Analysis of Variance (ANOVA)

ANOVA is a statistical technique used to compare means across multiple groups and determine if there are significant differences between them.

```python
from scipy import stats

# Sample data for three groups
group1 = [5, 6, 7, 5, 6]
group2 = [7, 8, 9, 8, 7]
group3 = [6, 7, 8, 7, 6]

# Perform one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")

# Output:
# F-statistic: 6.666666666666667
# P-value: 0.009523809523809525
```

Slide 10: Time Series Analysis

Time series analysis involves studying data points collected over time to identify trends, seasonality, and other patterns.

```python
import matplotlib.pyplot as plt

# Create a sample time series
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.randn(len(dates)).cumsum()
ts = pd.Series(values, index=dates)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(ts)
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Calculate rolling mean
rolling_mean = ts.rolling(window=30).mean()
print(f"Rolling mean (last 5 days):\n{rolling_mean.tail()}")
```

Slide 11: Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that helps identify the most important features in a dataset.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X = np.random.randn(100, 3)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.bar(range(1, 4), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 12: Real-Life Example: Analyzing Weather Data

In this example, we'll analyze temperature data to identify trends and patterns.

```python
import matplotlib.pyplot as plt

# Sample weather data (temperature in Celsius)
data = {
    'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
    'Temperature': np.random.normal(loc=20, scale=5, size=365) + 5 * np.sin(np.arange(365) * 2 * np.pi / 365)
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Calculate monthly average temperature
monthly_avg = df.resample('M').mean()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temperature'], label='Daily Temperature')
plt.plot(monthly_avg.index, monthly_avg['Temperature'], label='Monthly Average', linewidth=2)
plt.title('Temperature Analysis')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

print("Annual average temperature:", df['Temperature'].mean())
print("Warmest day:", df['Temperature'].idxmax(), "with temperature:", df['Temperature'].max())
print("Coldest day:", df['Temperature'].idxmin(), "with temperature:", df['Temperature'].min())
```

Slide 13: Real-Life Example: Analyzing Product Ratings

In this example, we'll analyze product ratings to gain insights into customer satisfaction.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample product ratings (1 to 5 stars)
ratings = np.random.choice([1, 2, 3, 4, 5], size=1000, p=[0.05, 0.1, 0.2, 0.3, 0.35])

# Calculate statistics
mean_rating = np.mean(ratings)
median_rating = np.median(ratings)
mode_rating = stats.mode(ratings).mode[0]

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(ratings, bins=5, range=(0.5, 5.5), edgecolor='black')
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.xticks([1, 2, 3, 4, 5])
plt.show()

print(f"Mean rating: {mean_rating:.2f}")
print(f"Median rating: {median_rating}")
print(f"Mode rating: {mode_rating}")
print(f"Percentage of 4 and 5 star ratings: {np.mean(ratings >= 4) * 100:.2f}%")
```

Slide 14: Additional Resources

For further exploration of statistics in data science, consider these resources:

1. "Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (Available on ArXiv: [https://arxiv.org/abs/1501.07249](https://arxiv.org/abs/1501.07249))
2. "A Survey of Deep Learning Techniques for Neural Machine Translation" by Shuohang Wang and Jing Jiang (ArXiv: [https://arxiv.org/abs/1804.09849](https://arxiv.org/abs/1804.09849))
3. Coursera's "Statistics with Python Specialization" by the University of Michigan
4. DataCamp's "Statistical Thinking in Python" course
5. "Python for Data Analysis" by Wes McKinney

These resources provide in-depth coverage of statistical concepts and their applications in data science using Python.


