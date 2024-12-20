## Hypothesis Testing in Python
Slide 1: Importing Libraries Before we begin with hypothesis testing, we need to import the necessary libraries in Python. Code:

```python
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
```

Slide 2: One-Sample t-Test The one-sample t-test is used to determine if the sample mean is significantly different from a hypothesized population mean. Code:

```python
# Sample data
sample_data = [12.5, 14.2, 11.8, 13.1, 12.7]

# Hypothesized population mean
mu0 = 13

# Perform one-sample t-test
t_statistic, p_value = ttest_1samp(sample_data, mu0)

# Print results
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
```

Slide 3: Two-Sample t-Test The two-sample t-test is used to determine if the means of two independent samples are significantly different. Code:

```python
# Sample data
sample1 = [22.1, 24.6, 23.3, 21.8, 20.9]
sample2 = [18.7, 19.2, 20.1, 21.4, 22.5]

# Perform two-sample t-test
t_statistic, p_value = ttest_ind(sample1, sample2)

# Print results
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
```

Slide 4: Finance API: Alpha Vantage Alpha Vantage is a free API that provides stock market data, including time series data, global quotes, and more. Code:

```python
import requests

# API key (replace with your own key)
api_key = "YOUR_API_KEY"

# Stock ticker symbol
ticker = "AAPL"

# API endpoint for stock time series data
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"

# Send API request
response = requests.get(url)

# Print the data
print(response.text)
```

Slide 5: Hypothesis Testing on Stock Data We can use hypothesis testing to analyze stock data and make informed investment decisions. Code:

```python
# Load stock data into a DataFrame
stock_data = pd.read_csv("stock_data.csv")

# Calculate daily returns
stock_data["Returns"] = stock_data["Close"].pct_change()

# Hypothesis: The mean daily return is not significantly different from 0
mu0 = 0

# Perform one-sample t-test
t_statistic, p_value = ttest_1samp(stock_data["Returns"], mu0)

# Print results
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
```

Slide 6: Normality Assumption Many hypothesis tests, including the t-test, assume that the data is normally distributed. We can check this assumption using a normality test or visualizations. Code:

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Perform Shapiro-Wilk test for normality
stat, p_value = shapiro(stock_data["Returns"])

# Print results
print(f"Shapiro-Wilk statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Plot a histogram to visually assess normality
stock_data["Returns"].hist(bins=20)
plt.title("Distribution of Daily Returns")
plt.show()
```

Slide 7: Paired t-Test The paired t-test is used to compare the means of two related samples, such as stock prices before and after a specific event. Code:

```python
# Stock prices before and after an event
before = [120.5, 118.2, 122.1, 119.7, 121.3]
after = [123.1, 120.4, 125.2, 121.8, 124.5]

# Perform paired t-test
t_statistic, p_value = ttest_1samp(np.array(after) - np.array(before), 0)

# Print results
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
```

Slide 8: ANOVA Analysis of Variance (ANOVA) is used to compare the means of three or more independent groups. This can be useful for analyzing stock performance across different sectors or time periods. Code:

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
stock_data = pd.read_csv("stock_data.csv")

# Fit ANOVA model
model = ols("Returns ~ Sector", data=stock_data).fit()
anova_table = sm.stats.anov_lm(model, typ=2)

# Print ANOVA table
print(anova_table)
```

Slide 9: Effect Size Effect size measures the magnitude of the difference between groups or the strength of the relationship between variables. It provides additional context beyond statistical significance. Code:

```python
from scipy.stats import cohen_d

# Calculate Cohen's d effect size
group1 = stock_data[stock_data["Sector"] == "Tech"]["Returns"]
group2 = stock_data[stock_data["Sector"] == "Finance"]["Returns"]
cohen_d_value = cohen_d(group1, group2)

# Print effect size
print(f"Cohen's d effect size: {cohen_d_value:.2f}")
```

Slide 10: Multiple Testing When conducting multiple hypothesis tests, the probability of making at least one Type I error (false positive) increases. Adjustments, such as the Bonferroni correction, can be applied to control the familywise error rate. Code:

```python
from statsmodels.stats.multitest import multipletests

# Perform multiple t-tests
p_values = [ttest_1samp(stock_data[stock_data["Sector"] == sector]["Returns"], 0)[1]
            for sector in stock_data["Sector"].unique()]

# Apply Bonferroni correction
rejected, corrected_p_values, _, _ = multipletests(p_values, method="bonferroni")

# Print corrected p-values
for sector, p_value in zip(stock_data["Sector"].unique(), corrected_p_values):
    print(f"Sector: {sector}, Corrected p-value: {p_value:.4f}")
```

Slide 11: Correlation Analysis Correlation analysis measures the strength and direction of the relationship between two variables. In finance, it can be used to analyze the relationship between stock returns and other factors. Code:

```python
import seaborn as sns

# Load stock data
stock_data = pd.read_csv("stock_data.csv")

# Calculate correlation matrix
corr_matrix = stock_data[["Returns", "Volume", "Volatility"]].corr()

# Plot correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```

Slide 12: Regression Analysis Regression analysis is used to model the relationship between a dependent variable and one or more independent variables. It can be used to predict stock prices or returns based on various factors. Code:

```python
import statsmodels.api as sm

# Load stock data
stock_data = pd.read_csv("stock_data.csv")

# Fit linear regression model
X = stock_data[["Volume", "Volatility"]]
y = stock_data["Returns"]
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())
```

Slide 13: Time Series Analysis Time series analysis is a specialized branch of statistics that deals with data collected over time. It can be used to analyze and forecast stock prices, trading volumes, and other financial time series data. Code:

```python
from statsmodels.tsa.arima.model import ARIMA

# Load stock price data
stock_prices = pd.read_csv("stock_prices.csv", index_col="Date", parse_dates=True)

# Fit ARIMA model
model = ARIMA(stock_prices["Close"], order=(1, 1, 1)).fit()

# Forecast future stock prices
forecast = model.forecast(steps=30)[0]

# Plot actual and forecasted prices
plt.figure(figsize=(12, 6))
plt.plot(stock_prices.index, stock_prices["Close"], label="Actual")
plt.plot(forecast.index, forecast.values, label="Forecast")
plt.title("Stock Price Forecast")
plt.legend()
plt.show()
```

Slide 14: Backtesting Trading Strategies Backtesting involves applying a trading strategy to historical data to evaluate its performance. Hypothesis testing can be used to assess the significance of the strategy's returns. Code:

```python
# Load stock data
stock_data = pd.read_csv("stock_data.csv")

# Define trading strategy
stock_data["Signal"] = np.where(stock_data["Returns"].shift(1) > 0, 1, -1)
stock_data["Strategy_Returns"] = stock_data["Signal"] * stock_data["Returns"]

# Hypothesis: The mean strategy return is not significantly different from 0
mu0 = 0
t_statistic, p_value = ttest_1samp(stock_data["Strategy_Returns"], mu0)

# Print results
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
```
