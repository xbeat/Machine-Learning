## Time Shifting in Pandas Manipulating Time Series Data
Slide 1: Time Shifting in Pandas

Time shifting is a powerful feature in pandas that allows you to manipulate and analyze time series data efficiently. It enables you to shift index labels forward or backward, creating new time-based perspectives on your data. This slideshow will guide you through the concepts and practical applications of time shifting in pandas using Python.

```python
import pandas as pd
import numpy as np

# Create a sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
data = np.random.randn(10)
ts = pd.Series(data, index=dates)
print(ts)
```

Slide 2: Basic Time Shifting with shift()

The shift() method is the fundamental tool for time shifting in pandas. It moves the index by a specified number of periods, filling in NaN values for the newly created positions.

```python
# Shift the time series forward by 2 periods
shifted_forward = ts.shift(2)
print("Original series:\n", ts)
print("\nShifted forward by 2 periods:\n", shifted_forward)

# Shift the time series backward by 1 period
shifted_backward = ts.shift(-1)
print("\nShifted backward by 1 period:\n", shifted_backward)
```

Slide 3: Filling Shifted Values

When shifting time series data, you might want to fill the newly created NaN values. The fillna() method can be used in combination with shift() to achieve this.

```python
# Shift forward by 2 periods and fill with the last valid observation
shifted_filled = ts.shift(2).fillna(method='ffill')
print("Shifted and filled:\n", shifted_filled)

# Shift backward by 1 period and fill with a specific value
shifted_filled_value = ts.shift(-1).fillna(0)
print("\nShifted and filled with 0:\n", shifted_filled_value)
```

Slide 4: Time Shifting with Different Frequencies

Pandas allows you to shift time series data using different time frequencies, such as days, hours, or minutes. This is particularly useful when working with data that has irregular time intervals.

```python
# Create a time series with irregular intervals
irregular_dates = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-07', '2023-01-10'])
irregular_ts = pd.Series(np.random.randn(4), index=irregular_dates)

# Shift by 2 days
shifted_days = irregular_ts.shift(periods=2, freq='D')
print("Original irregular series:\n", irregular_ts)
print("\nShifted by 2 days:\n", shifted_days)
```

Slide 5: Rolling Window Operations

Time shifting is often used in combination with rolling window operations to perform calculations over a moving time frame. The rolling() method creates a window of specified size that moves along the time series.

```python
# Calculate a 3-day rolling mean
rolling_mean = ts.rolling(window=3).mean()
print("Original series:\n", ts)
print("\n3-day rolling mean:\n", rolling_mean)

# Calculate a 2-day rolling sum with a 1-day shift
rolling_sum_shifted = ts.shift(1).rolling(window=2).sum()
print("\n2-day rolling sum with 1-day shift:\n", rolling_sum_shifted)
```

Slide 6: Lagged Differences

Time shifting is crucial for calculating lagged differences, which are often used in time series analysis to remove trends and make the data stationary.

```python
# Calculate the 1-day lagged difference
lagged_diff = ts - ts.shift(1)
print("Original series:\n", ts)
print("\n1-day lagged difference:\n", lagged_diff)

# Calculate the 3-day lagged difference
lagged_diff_3 = ts - ts.shift(3)
print("\n3-day lagged difference:\n", lagged_diff_3)
```

Slide 7: Percentage Change Calculation

Time shifting is essential for calculating percentage changes over time, a common operation in data analysis and finance.

```python
# Calculate daily percentage change
pct_change = ts.pct_change()
print("Original series:\n", ts)
print("\nDaily percentage change:\n", pct_change)

# Calculate 3-day percentage change
pct_change_3 = ts.pct_change(periods=3)
print("\n3-day percentage change:\n", pct_change_3)
```

Slide 8: Time Shifting in DataFrames

Time shifting can be applied to entire DataFrames, allowing you to shift multiple columns simultaneously or apply different shifts to different columns.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'A': np.random.randn(5),
    'B': np.random.randn(5)
}, index=pd.date_range(start='2023-01-01', periods=5))

# Shift entire DataFrame
shifted_df = df.shift(1)
print("Original DataFrame:\n", df)
print("\nShifted DataFrame:\n", shifted_df)

# Shift individual columns differently
df_mixed_shift = df.()
df_mixed_shift['A'] = df['A'].shift(1)
df_mixed_shift['B'] = df['B'].shift(-1)
print("\nMixed shifted DataFrame:\n", df_mixed_shift)
```

Slide 9: Real-Life Example: Weather Data Analysis

Let's explore how time shifting can be applied to analyze temperature changes in weather data.

```python
# Create a sample weather dataset
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
temperatures = [20, 22, 19, 21, 23, 22, 20, 18, 19, 21]
weather_data = pd.Series(temperatures, index=dates, name='Temperature')

# Calculate day-over-day temperature change
temp_change = weather_data - weather_data.shift(1)
print("Temperature data:\n", weather_data)
print("\nDay-over-day temperature change:\n", temp_change)

# Identify days with temperature increase
temp_increase = temp_change > 0
print("\nDays with temperature increase:\n", temp_increase)
```

Slide 10: Real-Life Example: Product Inventory Analysis

Time shifting can be useful in analyzing product inventory levels and predicting stock needs.

```python
# Create a sample inventory dataset
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
inventory_levels = [100, 95, 105, 110, 100, 90, 85, 95, 100, 105]
inventory_data = pd.Series(inventory_levels, index=dates, name='Inventory')

# Calculate 3-day moving average of inventory levels
inventory_ma = inventory_data.rolling(window=3).mean()
print("Inventory data:\n", inventory_data)
print("\n3-day moving average of inventory:\n", inventory_ma)

# Predict next day's inventory based on 3-day trend
next_day_prediction = inventory_ma.shift(-1)
print("\nNext day's inventory prediction:\n", next_day_prediction)
```

Slide 11: Time Shifting with Custom Business Days

Pandas allows you to define custom business day calendars, which can be useful for time shifting in business-related analyses.

```python
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

# Create a custom business day calendar
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# Create a sample business time series
business_dates = pd.date_range(start='2023-01-01', end='2023-01-15', freq=us_bd)
business_data = pd.Series(np.random.randn(len(business_dates)), index=business_dates)

# Shift by 2 business days
shifted_business = business_data.shift(2, freq=us_bd)
print("Original business series:\n", business_data)
print("\nShifted by 2 business days:\n", shifted_business)
```

Slide 12: Handling Time Zones in Time Shifting

When working with time series data from different time zones, it's important to consider time zone information during time shifting operations.

```python
# Create a time series with time zone information
tz_dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D', tz='US/Eastern')
tz_data = pd.Series(np.random.randn(5), index=tz_dates)

# Shift the time series
shifted_tz = tz_data.shift(1)
print("Original time zone aware series:\n", tz_data)
print("\nShifted time zone aware series:\n", shifted_tz)

# Convert to a different time zone and shift
tz_data_pacific = tz_data.tz_convert('US/Pacific')
shifted_tz_pacific = tz_data_pacific.shift(1)
print("\nShifted series in Pacific time:\n", shifted_tz_pacific)
```

Slide 13: Combining Time Shifting with Resampling

Time shifting can be combined with resampling to analyze data at different time frequencies.

```python
# Create a high-frequency time series
high_freq_dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='H')
high_freq_data = pd.Series(np.random.randn(len(high_freq_dates)), index=high_freq_dates)

# Resample to daily frequency and calculate shifted difference
daily_data = high_freq_data.resample('D').mean()
daily_shift_diff = daily_data - daily_data.shift(1)

print("Original high-frequency data:\n", high_freq_data)
print("\nResampled daily data:\n", daily_data)
print("\nDaily shifted difference:\n", daily_shift_diff)
```

Slide 14: Time Shifting in Seasonal Decomposition

Time shifting plays a crucial role in seasonal decomposition, which is used to separate time series data into trend, seasonal, and residual components.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Create a sample seasonal time series
seasonal_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
seasonal_data = pd.Series(np.sin(np.arange(len(seasonal_dates)) * 2 * np.pi / 365) + 
                          np.random.normal(0, 0.1, len(seasonal_dates)), 
                          index=seasonal_dates)

# Perform seasonal decomposition
result = seasonal_decompose(seasonal_data, model='additive', period=365)

# Plot the components
import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
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

Slide 15: Additional Resources

For further exploration of time shifting and time series analysis in pandas, consider the following resources:

1. Pandas Official Documentation: Time Series / Date functionality [https://pandas.pydata.org/pandas-docs/stable/user\_guide/timeseries.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)
2. "Time Series Analysis in Python with statsmodels" by Aileen Nielsen ArXiv URL: [https://arxiv.org/abs/1810.07704](https://arxiv.org/abs/1810.07704)
3. "Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos Available online at: [https://otexts.com/fpp3/](https://otexts.com/fpp3/)

These resources provide in-depth explanations and advanced techniques for working with time series data in Python and pandas.

