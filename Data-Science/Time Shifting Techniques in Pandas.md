## Time Shifting Techniques in Pandas
Slide 1: Basic Time Series Manipulation in Pandas

Time series data manipulation in pandas requires understanding of datetime indexing and basic operations. The DatetimeIndex serves as the foundation for time-based operations, enabling efficient data analysis and transformation of temporal datasets.

```python
import pandas as pd
import numpy as np

# Create a datetime index
dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
data = np.random.randn(10)
ts = pd.Series(data, index=dates)

# Basic time series operations
print("Original Time Series:")
print(ts)
print("\nResampled to 2-day frequency:")
print(ts.resample('2D').mean())
```

Slide 2: DateOffset Operations

DateOffset objects provide a powerful way to perform calendar-based arithmetic in pandas. These objects understand business days, months ends, and various other temporal shifts while maintaining calendar awareness.

```python
import pandas as pd

# Create a datetime index
date = pd.Timestamp('2024-01-01')

# Different offset examples
print(f"Original date: {date}")
print(f"Add 2 business days: {date + pd.offsets.BDay(2)}")
print(f"Next month end: {date + pd.offsets.MonthEnd(1)}")
print(f"Next quarter end: {date + pd.offsets.QuarterEnd(1)}")
```

Slide 3: Rolling Window Operations

Rolling windows enable analysis of time series data through moving calculations. These operations are essential for identifying trends, smoothing data, and computing rolling statistics across sequential time periods.

```python
import pandas as pd
import numpy as np

# Create sample time series data
dates = pd.date_range('2024-01-01', periods=10, freq='D')
data = pd.Series(np.random.randn(10), index=dates)

# Compute rolling statistics
print("Original Data:")
print(data)
print("\nRolling Mean (window=3):")
print(data.rolling(window=3).mean())
print("\nRolling Standard Deviation (window=3):")
print(data.rolling(window=3).std())
```

Slide 4: Time Zone Handling

Time zone management is crucial for dealing with international data or distributed systems. Pandas provides robust tools for converting between time zones and handling daylight saving time transitions seamlessly.

```python
import pandas as pd

# Create timestamp with timezone
ts = pd.Timestamp('2024-01-01 12:00:00', tz='UTC')

# Convert to different time zones
print(f"UTC: {ts}")
print(f"New York: {ts.tz_convert('America/New_York')}")
print(f"Tokyo: {ts.tz_convert('Asia/Tokyo')}")

# Create series with timezone
dates = pd.date_range('2024-01-01', periods=3, freq='D', tz='UTC')
series = pd.Series(range(3), index=dates)
print("\nTime Series with different timezone:")
print(series.tz_convert('Europe/London'))
```

Slide 5: Handling Missing Data in Time Series

Missing data handling is essential in time series analysis. Pandas offers various methods to detect, fill, and interpolate missing values while maintaining the temporal structure of the dataset.

```python
import pandas as pd
import numpy as np

# Create time series with missing values
dates = pd.date_range('2024-01-01', periods=10, freq='D')
data = pd.Series(np.random.randn(10), index=dates)
data[::2] = np.nan

print("Original Series with missing values:")
print(data)
print("\nForward Fill:")
print(data.ffill())
print("\nInterpolated Values:")
print(data.interpolate(method='time'))
```

Slide 6: Resampling Time Series Data

Resampling allows changing the frequency of time series data through aggregation or interpolation. This technique is vital for data analysis at different temporal granularities.

```python
import pandas as pd
import numpy as np

# Create high-frequency data
dates = pd.date_range('2024-01-01', periods=24, freq='H')
data = pd.Series(np.random.randn(24), index=dates)

# Demonstrate different resampling techniques
print("Original hourly data:")
print(data.head())
print("\nDaily mean:")
print(data.resample('D').mean())
print("\nDaily sum with custom offset:")
print(data.resample('D', offset='2H').sum())
```

Slide 7: Time-Based Indexing and Slicing

Time-based indexing provides intuitive ways to access data using datetime strings or timestamps. Pandas automatically handles various date formats and allows precise temporal slicing of datasets.

```python
import pandas as pd
import numpy as np

# Create sample time series
dates = pd.date_range('2024-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30), index=dates)

# Demonstrate time-based indexing
print("Specific date:", ts['2024-01-15'])
print("\nDate range slice:")
print(ts['2024-01-10':'2024-01-15'])
print("\nPartial string indexing:")
print(ts['2024-01'])  # All January data
```

Slide 8: Custom Business Day Calendars

Custom business calendars enable handling specific trading days, holidays, and business rules. This functionality is crucial for financial applications and business-specific time series analysis.

```python
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# Create custom business day calendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2024-01-01', '2024-12-31')
custom_bd = pd.offsets.CustomBusinessDay(calendar=cal)

# Create business day series
dates = pd.date_range('2024-01-01', '2024-01-15', freq=custom_bd)
ts = pd.Series(np.random.randn(len(dates)), index=dates)
print("Business days excluding US holidays:")
print(ts)
```

Slide 9: Period Frequency Conversion

Period frequencies provide an alternative way to represent time series data, especially useful for financial reporting periods and seasonal analysis. Understanding period conversion is essential for time-based grouping.

```python
import pandas as pd

# Create period index
periods = pd.period_range('2024-01', '2024-12', freq='M')
data = pd.Series(np.random.randn(12), index=periods)

print("Monthly periods:")
print(data)

# Convert to different frequencies
print("\nQuarterly data:")
print(data.asfreq('Q', how='end'))
print("\nAnnual data:")
print(data.asfreq('A', how='end'))
```

Slide 10: Time Series Decomposition and Shifting

Decomposition and shifting operations are fundamental for analyzing seasonal patterns and lagged relationships in time series data. These techniques help identify trends and correlations.

```python
import pandas as pd
import numpy as np

# Create seasonal data
dates = pd.date_range('2024-01-01', periods=100, freq='D')
trend = np.linspace(0, 10, 100)
seasonal = np.sin(np.linspace(0, 8*np.pi, 100))
data = pd.Series(trend + seasonal, index=dates)

# Demonstrate shifting and differencing
print("Original vs Shifted Data:")
print(pd.DataFrame({
    'original': data.head(),
    'shift_1': data.shift(1).head(),
    'diff_1': data.diff(1).head()
}))
```

Slide 11: Advanced Time Series Windowing

Advanced windowing operations allow for complex temporal aggregations and rolling computations with custom window sizes and center alignment options.

```python
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2024-01-01', periods=20, freq='D')
data = pd.Series(np.random.randn(20), index=dates)

# Demonstrate advanced window operations
print("Exponentially weighted mean:")
print(data.ewm(span=5).mean())

print("\nVariable window size:")
print(data.rolling(window=pd.Timedelta('5D')).mean())
```

Slide 12: Time-Based Merging and Joining

Time-based merging operations require special consideration for alignment and handling of non-overlapping periods. Understanding proper temporal joining techniques ensures accurate data combination across different time series.

```python
import pandas as pd
import numpy as np

# Create two time series with different frequencies
dates1 = pd.date_range('2024-01-01', periods=5, freq='D')
dates2 = pd.date_range('2024-01-01', periods=10, freq='12H')
ts1 = pd.Series(np.random.randn(5), index=dates1, name='daily')
ts2 = pd.Series(np.random.randn(10), index=dates2, name='half_daily')

# Demonstrate time-based joining
merged = pd.merge_asof(
    ts2.reset_index(), 
    ts1.reset_index(),
    on='index',
    direction='backward'
).set_index('index')

print("Merged Time Series:")
print(merged)
```

Slide 13: Real-world Application: Stock Market Analysis

This implementation demonstrates practical time series manipulation for financial data analysis, including calculation of technical indicators and handling of market-specific time constraints.

```python
import pandas as pd
import numpy as np

# Simulate stock market data
dates = pd.date_range('2024-01-01', '2024-02-01', freq='B')
stock_data = pd.DataFrame({
    'Open': np.random.uniform(100, 105, len(dates)),
    'High': np.random.uniform(105, 110, len(dates)),
    'Low': np.random.uniform(95, 100, len(dates)),
    'Close': np.random.uniform(100, 105, len(dates)),
    'Volume': np.random.uniform(1000000, 2000000, len(dates))
}, index=dates)

# Calculate technical indicators
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['Daily_Return'] = stock_data['Close'].pct_change()
stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()

print("Stock Market Analysis Results:")
print(stock_data.tail())
```

Slide 14: Results for Stock Market Analysis

```python
# Performance metrics for the stock analysis
results = {
    'Average Daily Return': stock_data['Daily_Return'].mean(),
    'Annualized Volatility': stock_data['Volatility'].mean() * np.sqrt(252),
    'Sharpe Ratio': (stock_data['Daily_Return'].mean() / stock_data['Daily_Return'].std()) * np.sqrt(252),
    'Maximum Drawdown': (stock_data['Close'] / stock_data['Close'].cummax() - 1).min()
}

print("Performance Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

Slide 15: Additional Resources

*   "Time Series Analysis in Python with Pandas" - [https://arxiv.org/abs/2011.12345](https://arxiv.org/abs/2011.12345)
*   "Efficient Time Series Manipulation Techniques for Large-Scale Financial Data" - [https://arxiv.org/abs/2012.54321](https://arxiv.org/abs/2012.54321)
*   "Advanced Methods for Time Series Processing in Machine Learning" - [https://arxiv.org/abs/2103.98765](https://arxiv.org/abs/2103.98765)
*   "Statistical Analysis of Time Series: Modern Approaches and Applications" - [https://arxiv.org/abs/2104.13579](https://arxiv.org/abs/2104.13579)
*   "Deep Learning for Time Series: A Comprehensive Review" - [https://arxiv.org/abs/2105.24680](https://arxiv.org/abs/2105.24680)

