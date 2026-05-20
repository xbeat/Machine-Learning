## Univariate Analysis of Continuous Variables in Python
Slide 1: Introduction to Univariate Analysis of Continuous Variables

Univariate analysis is a fundamental statistical method that focuses on examining and describing a single variable at a time. For continuous variables, this analysis involves understanding the distribution, central tendency, and spread of the data. Python, with its rich ecosystem of libraries, provides powerful tools for conducting univariate analysis efficiently.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample continuous data
data = np.random.normal(loc=0, scale=1, size=1000)
df = pd.DataFrame(data, columns=['Value'])

print(df.head())
```

Slide 2: Measures of Central Tendency

Central tendency measures provide a single value that attempts to describe the center of the data distribution. The three main measures are mean, median, and mode. For continuous variables, the mean and median are most commonly used.

```python
# Calculate mean and median
mean_value = df['Value'].mean()
median_value = df['Value'].median()

print(f"Mean: {mean_value:.2f}")
print(f"Median: {median_value:.2f}")

# Visualize central tendency
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Value', kde=True)
plt.axvline(mean_value, color='r', linestyle='--', label='Mean')
plt.axvline(median_value, color='g', linestyle='--', label='Median')
plt.legend()
plt.title('Distribution with Mean and Median')
plt.show()
```

Slide 3: Measures of Dispersion

Dispersion measures quantify the spread of data around the central tendency. Common measures include variance, standard deviation, and range. These metrics help understand the variability and consistency of the data.

```python
# Calculate dispersion measures
variance = df['Value'].var()
std_dev = df['Value'].std()
data_range = df['Value'].max() - df['Value'].min()

print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Range: {data_range:.2f}")

# Visualize dispersion
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Value'])
plt.title('Boxplot Showing Data Dispersion')
plt.show()
```

Slide 4: Percentiles and Quartiles

Percentiles divide the data into 100 equal parts, while quartiles divide it into four parts. These measures provide insights into the distribution of data and help identify potential outliers.

```python
# Calculate quartiles and percentiles
q1, q2, q3 = df['Value'].quantile([0.25, 0.5, 0.75])
p10, p90 = df['Value'].quantile([0.1, 0.9])

print(f"Q1 (25th percentile): {q1:.2f}")
print(f"Q2 (Median): {q2:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")
print(f"10th percentile: {p10:.2f}")
print(f"90th percentile: {p90:.2f}")

# Visualize quartiles
plt.figure(figsize=(10, 6))
sns.boxenplot(x=df['Value'])
plt.title('Enhanced Box Plot with Quartiles')
plt.show()
```

Slide 5: Skewness and Kurtosis

Skewness measures the asymmetry of the distribution, while kurtosis quantifies the tailedness of the distribution. These metrics provide additional insights into the shape of the data distribution.

```python
from scipy.stats import skew, kurtosis

# Calculate skewness and kurtosis
skewness = skew(df['Value'])
kurt = kurtosis(df['Value'])

print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurt:.2f}")

# Visualize skewness
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Value', kde=True)
plt.title(f'Distribution Shape (Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f})')
plt.show()
```

Slide 6: Histogram Analysis

Histograms provide a visual representation of the data distribution by dividing the range of values into bins and showing the frequency of data points in each bin.

```python
# Create histogram
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Value', bins=30, kde=True)
plt.title('Histogram of Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Analyze bin statistics
bin_counts, bin_edges, _ = plt.hist(df['Value'], bins=30)
most_frequent_bin = bin_edges[np.argmax(bin_counts)]
print(f"Most frequent bin center: {most_frequent_bin:.2f}")
```

Slide 7: Kernel Density Estimation (KDE)

KDE is a non-parametric way to estimate the probability density function of a random variable. It provides a smooth, continuous estimation of the data distribution.

```python
from scipy.stats import gaussian_kde

# Calculate KDE
kde = gaussian_kde(df['Value'])
x_range = np.linspace(df['Value'].min(), df['Value'].max(), 100)
y_kde = kde(x_range)

# Plot KDE
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_kde, label='KDE')
plt.hist(df['Value'], bins=30, density=True, alpha=0.5, label='Histogram')
plt.title('Kernel Density Estimation vs Histogram')
plt.legend()
plt.show()

# Find mode using KDE
mode_kde = x_range[np.argmax(y_kde)]
print(f"Estimated mode from KDE: {mode_kde:.2f}")
```

Slide 8: Cumulative Distribution Function (CDF)

The CDF shows the probability that a random variable takes a value less than or equal to a given value. It's useful for understanding the entire distribution of the data.

```python
# Calculate empirical CDF
sorted_data = np.sort(df['Value'])
y_values = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

# Plot CDF
plt.figure(figsize=(10, 6))
plt.plot(sorted_data, y_values)
plt.title('Cumulative Distribution Function')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

# Find median using CDF
median_cdf = np.interp(0.5, y_values, sorted_data)
print(f"Median from CDF: {median_cdf:.2f}")
```

Slide 9: Z-Score Analysis

Z-scores indicate how many standard deviations an element is from the mean. They are useful for identifying outliers and comparing values from different normal distributions.

```python
# Calculate Z-scores
z_scores = (df['Value'] - df['Value'].mean()) / df['Value'].std()

# Add Z-scores to dataframe
df['Z-Score'] = z_scores

# Identify potential outliers
outliers = df[abs(df['Z-Score']) > 3]

print("Potential outliers:")
print(outliers)

# Visualize Z-scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.index, y=df['Z-Score'])
plt.axhline(y=3, color='r', linestyle='--')
plt.axhline(y=-3, color='r', linestyle='--')
plt.title('Z-Scores of Data Points')
plt.xlabel('Index')
plt.ylabel('Z-Score')
plt.show()
```

Slide 10: Quantile-Quantile (Q-Q) Plot

Q-Q plots compare the quantiles of the data distribution to the quantiles of a theoretical distribution (often normal). They help assess whether the data follows a particular distribution.

```python
import statsmodels.api as sm

# Create Q-Q plot
fig, ax = plt.subplots(figsize=(10, 6))
sm.qqplot(df['Value'], line='s', ax=ax)
plt.title('Q-Q Plot')
plt.show()

# Perform Shapiro-Wilk test for normality
from scipy.stats import shapiro
stat, p = shapiro(df['Value'])
print(f"Shapiro-Wilk test - statistic: {stat:.4f}, p-value: {p:.4f}")
```

Slide 11: Time Series Analysis (if applicable)

If the continuous variable has a time component, time series analysis can reveal trends, seasonality, and other temporal patterns.

```python
# Generate sample time series data
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
time_series = pd.Series(np.cumsum(np.random.randn(365)) + 10, index=dates)

# Plot time series
plt.figure(figsize=(12, 6))
plt.plot(time_series)
plt.title('Time Series of Continuous Variable')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# Compute rolling statistics
rolling_mean = time_series.rolling(window=30).mean()
rolling_std = time_series.rolling(window=30).std()

print("First 5 values of rolling mean:")
print(rolling_mean.head())
```

Slide 12: Correlation with Other Variables

While univariate analysis focuses on a single variable, it's often useful to examine its relationship with other variables in the dataset.

```python
# Generate correlated data
np.random.seed(42)
x = np.random.randn(1000)
y = 2*x + np.random.randn(1000)*0.5
df_corr = pd.DataFrame({'X': x, 'Y': y})

# Calculate correlation
correlation = df_corr['X'].corr(df_corr['Y'])

# Visualize correlation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_corr, x='X', y='Y')
plt.title(f'Correlation between X and Y (r = {correlation:.2f})')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
```

Slide 13: Real-Life Example 1: Height Analysis

Let's analyze a dataset of adult heights to demonstrate univariate analysis in a real-world context.

```python
# Generate sample height data (in cm)
np.random.seed(42)
heights = np.random.normal(loc=170, scale=10, size=1000)
df_heights = pd.DataFrame({'Height': heights})

# Basic statistics
print(df_heights['Height'].describe())

# Visualize distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df_heights, x='Height', kde=True)
plt.title('Distribution of Adult Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()

# Calculate and print percentiles
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(heights, p):.2f} cm")
```

Slide 14: Real-Life Example 2: Stock Price Analysis

Let's analyze the closing prices of a hypothetical stock to demonstrate univariate analysis in financial data.

```python
import yfinance as yf

# Download stock data (e.g., Apple Inc.)
stock_data = yf.download('AAPL', start='2022-01-01', end='2023-01-01')

# Focus on closing prices
closing_prices = stock_data['Close']

# Basic statistics
print(closing_prices.describe())

# Visualize price distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=closing_prices, kde=True)
plt.title('Distribution of AAPL Closing Prices (2022)')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()

# Calculate daily returns
daily_returns = closing_prices.pct_change().dropna()

# Plot daily returns
plt.figure(figsize=(10, 6))
sns.lineplot(data=daily_returns)
plt.title('Daily Returns of AAPL Stock (2022)')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()

# Calculate and print volatility (standard deviation of returns)
volatility = daily_returns.std()
print(f"Annualized volatility: {volatility * np.sqrt(252):.2%}")
```

Slide 15: Additional Resources

For further exploration of univariate analysis and statistical methods using Python, consider the following resources:

1. "Statistical Data Analysis in Python" by Thomas Haslwanter (arXiv:1807.09209) URL: [https://arxiv.org/abs/1807.09209](https://arxiv.org/abs/1807.09209)
2. "Python for Data Analysis" by Wes McKinney (O'Reilly Media)
3. SciPy Documentation: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
4. Seaborn Tutorial: [https://seaborn.pydata.org/tutorial.html](https://seaborn.pydata.org/tutorial.html)
5. Statsmodels Documentation: [https://www.statsmodels.org/stable/index.html](https://www.statsmodels.org/stable/index.html)

These resources provide in-depth explanations and additional techniques for univariate analysis and beyond.

