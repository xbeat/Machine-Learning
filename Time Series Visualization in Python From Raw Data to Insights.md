## Time Series Visualization in Python From Raw Data to Insights
Slide 1: Introduction to Time Series Visualization

Time series visualization is a powerful tool for understanding and analyzing data that changes over time. It allows us to identify patterns, trends, and anomalies that might be difficult to spot in raw data. In this presentation, we'll explore various techniques for visualizing time series data using Python, starting from basic plots and progressing to more advanced visualizations.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100

# Create a simple line plot
plt.figure(figsize=(12, 6))
plt.plot(dates, values)
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

Slide 2: Loading and Preparing Time Series Data

Before we can visualize time series data, we need to load and prepare it. Python's pandas library is excellent for handling time series data. Let's look at how to load data from a CSV file and ensure it's properly formatted for time series analysis.

```python
import pandas as pd

# Load data from CSV file
df = pd.read_csv('time_series_data.csv', parse_dates=['date'], index_col='date')

# Ensure the index is in datetime format
df.index = pd.to_datetime(df.index)

# Sort the index to ensure chronological order
df = df.sort_index()

print(df.head())
```

Slide 3: Basic Line Plot

The simplest and most common way to visualize time series data is using a line plot. This type of plot connects data points with lines, making it easy to see trends and patterns over time.

```python
import matplotlib.pyplot as plt

# Create a line plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'])
plt.title('Time Series Line Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

Slide 4: Multiple Time Series

Often, we need to compare multiple time series. We can do this by plotting multiple lines on the same graph, using different colors and labels to distinguish between them.

```python
# Assuming df has multiple columns: 'value1', 'value2', 'value3'
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value1'], label='Series 1')
plt.plot(df.index, df['value2'], label='Series 2')
plt.plot(df.index, df['value3'], label='Series 3')
plt.title('Multiple Time Series Comparison')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 5: Seasonal Decomposition

Seasonal decomposition is a technique used to separate a time series into its constituent components: trend, seasonality, and residuals. This can help us understand the underlying patterns in our data.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition
result = seasonal_decompose(df['value'], model='additive', period=30)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()
```

Slide 6: Rolling Statistics

Rolling statistics, such as moving averages and standard deviations, can help smooth out short-term fluctuations and highlight longer-term trends or cycles in the data.

```python
# Calculate rolling mean and standard deviation
rolling_mean = df['value'].rolling(window=30).mean()
rolling_std = df['value'].rolling(window=30).std()

# Plot original data with rolling statistics
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['value'], label='Original')
plt.plot(df.index, rolling_mean, label='30-day Rolling Mean')
plt.plot(df.index, rolling_std, label='30-day Rolling Std')
plt.title('Time Series with Rolling Statistics')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 7: Heatmaps for Seasonal Patterns

Heatmaps can be an effective way to visualize seasonal patterns in time series data, especially when dealing with multiple years of data.

```python
import seaborn as sns

# Reshape the data into a matrix (assuming daily data)
data_matrix = df['value'].values.reshape(-1, 365)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data_matrix, cmap='YlOrRd')
plt.title('Time Series Heatmap (Daily Pattern)')
plt.xlabel('Day of Year')
plt.ylabel('Year')
plt.show()
```

Slide 8: Autocorrelation Plot

Autocorrelation plots help identify repeating patterns or periodicity in a time series by showing the correlation of the series with itself at different lag times.

```python
from pandas.plotting import autocorrelation_plot

# Create autocorrelation plot
plt.figure(figsize=(12, 6))
autocorrelation_plot(df['value'])
plt.title('Autocorrelation Plot')
plt.grid(True)
plt.show()
```

Slide 9: Box Plots for Seasonal Comparisons

Box plots can be used to compare the distribution of values across different seasons or time periods, helping to identify seasonal patterns and outliers.

```python
# Add month and year columns
df['month'] = df.index.month
df['year'] = df.index.year

# Create box plot
plt.figure(figsize=(12, 6))
df.boxplot(column='value', by='month')
plt.title('Monthly Distribution of Values')
plt.xlabel('Month')
plt.ylabel('Value')
plt.show()
```

Slide 10: Interactive Time Series Visualization with Plotly

For more interactive visualizations, we can use libraries like Plotly. These allow users to zoom, pan, and hover over data points for more information.

```python
import plotly.graph_objects as go

# Create an interactive line plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['value'], mode='lines', name='Time Series'))
fig.update_layout(title='Interactive Time Series Plot',
                  xaxis_title='Date',
                  yaxis_title='Value')
fig.show()
```

Slide 11: Real-Life Example: Weather Data Visualization

Let's visualize temperature data for a city over a year, showing daily temperatures and a moving average.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample weather data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
temperatures = 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.randn(len(dates)) * 3
weather_df = pd.DataFrame({'date': dates, 'temperature': temperatures})
weather_df.set_index('date', inplace=True)

# Calculate 7-day moving average
weather_df['moving_avg'] = weather_df['temperature'].rolling(window=7).mean()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(weather_df.index, weather_df['temperature'], label='Daily Temperature', alpha=0.7)
plt.plot(weather_df.index, weather_df['moving_avg'], label='7-day Moving Average', linewidth=2)
plt.title('Daily Temperature and Moving Average')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 12: Real-Life Example: Energy Consumption Analysis

Let's visualize energy consumption data, comparing weekday and weekend patterns.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample energy consumption data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
consumption = 100 + 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24) + np.random.randn(len(dates)) * 10
energy_df = pd.DataFrame({'datetime': dates, 'consumption': consumption})
energy_df.set_index('datetime', inplace=True)

# Separate weekday and weekend data
weekday = energy_df[energy_df.index.dayofweek < 5].groupby(energy_df.index.hour).mean()
weekend = energy_df[energy_df.index.dayofweek >= 5].groupby(energy_df.index.hour).mean()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(weekday.index, weekday['consumption'], label='Weekday Average', marker='o')
plt.plot(weekend.index, weekend['consumption'], label='Weekend Average', marker='o')
plt.title('Average Hourly Energy Consumption: Weekday vs Weekend')
plt.xlabel('Hour of Day')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()
```

Slide 13: Handling Missing Data in Time Series

Missing data is a common issue in time series analysis. Let's explore how to visualize and handle missing data points.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create sample data with missing values
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.randn(len(dates)).cumsum()
ts = pd.Series(values, index=dates)

# Introduce missing values
ts[ts.index[10:20]] = np.nan
ts[ts.index[50:60]] = np.nan

# Visualize the data with missing values
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts.values)
plt.title('Time Series with Missing Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)

# Highlight missing data periods
missing_periods = ts[ts.isnull()].index
for period in missing_periods:
    plt.axvspan(period, period, color='red', alpha=0.3)

plt.show()

# Fill missing values using forward fill method
ts_filled = ts.fillna(method='ffill')

# Visualize the filled data
plt.figure(figsize=(12, 6))
plt.plot(ts_filled.index, ts_filled.values)
plt.title('Time Series with Missing Data Filled')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

Slide 14: Time Series Forecasting Visualization

Visualizing time series forecasts can help in understanding future trends and the uncertainty associated with predictions.

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.cumsum(np.random.randn(len(dates))) + 100
ts = pd.Series(values, index=dates)

# Fit ARIMA model
model = ARIMA(ts, order=(1, 1, 1))
results = model.fit()

# Make forecast
forecast = results.forecast(steps=30)

# Visualize the forecast
plt.figure(figsize=(12, 6))
plt.plot(ts.index, ts.values, label='Observed')
plt.plot(forecast.index, forecast.values, color='red', label='Forecast')
plt.fill_between(forecast.index,
                 forecast.values - 1.96 * forecast.se_mean,
                 forecast.values + 1.96 * forecast.se_mean,
                 color='pink', alpha=0.3, label='95% Confidence Interval')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 15: Additional Resources

For those interested in diving deeper into time series visualization and analysis using Python, here are some valuable resources:

1. "Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos (available online: [https://otexts.com/fpp3/](https://otexts.com/fpp3/))
2. "Time Series Analysis and Its Applications: With R Examples" by Robert H. Shumway and David S. Stoffer (ArXiv: [https://arxiv.org/abs/1706.09838](https://arxiv.org/abs/1706.09838))
3. Pandas documentation on time series functionality: [https://pandas.pydata.org/pandas-docs/stable/user\_guide/timeseries.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
4. Statsmodels documentation on time series analysis: [https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)

These resources provide in-depth explanations of time series concepts, advanced visualization techniques, and practical examples using Python and other tools.

